//! Commands of the local environment on `tokio::process`.
//!
//! One task drives a command: it reads both pipes in arrival order, decodes each stream
//! separately, hands every chunk to the output callback and the spill, and watches the exit,
//! the timeout and the context. A timeout or cancellation kills the command's process tree and
//! the result waits until the pipes drain. After the command exits, a descendant that keeps the
//! pipes open delays the result only until the pipes stay quiet for [`EXIT_STDIO_GRACE`].

use std::collections::BTreeMap;
use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::time::Duration;

use delta_chord::context::Context;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command as ProcessCommand};
use tokio::time::{Instant, sleep_until};

use super::process::{ChildRegistry, ProcessTree, exit_code};
use super::spill::Spill;
use crate::env::decode::Utf8Decoder;
use crate::env::{
    Command, ExecError, ExecErrorCode, ExecOptions, ExecResult, OutputCallback, OutputChunk, OutputStream,
};

/// How long the pipes may stay quiet after the command exited before the result is settled
/// without waiting for their end.
const EXIT_STDIO_GRACE: Duration = Duration::from_millis(100);
/// Read size of the pipes.
const PIPE_CHUNK: usize = 64 * 1024;

/// What the local environment runs commands with.
#[derive(Debug)]
pub(super) struct ShellConfig<'a> {
    pub(super) cwd: &'a Path,
    pub(super) shell_path: Option<&'a Path>,
    pub(super) shell_env: &'a BTreeMap<String, String>,
    pub(super) children: &'a ChildRegistry,
}

/// Run `command`; see [`Shell::exec`](crate::env::Shell::exec).
pub(super) async fn exec(
    config: ShellConfig<'_>,
    command: Command,
    options: ExecOptions<'_>,
    ctx: &Context,
) -> Result<ExecResult, ExecError> {
    if ctx.is_cancelled() {
        return Err(ExecError::aborted());
    }
    let timeout = options.checked_timeout()?;
    let cwd = options
        .cwd
        .as_deref()
        .map_or_else(|| config.cwd.to_path_buf(), |cwd| super::path::resolve(config.cwd, cwd));
    let (program, args) = match command {
        Command::Shell(script) => (find_shell(config.shell_path).await?, vec!["-c".to_owned(), script]),
        Command::Argv(mut argv) => {
            if argv.is_empty() {
                return Err(ExecError::new(
                    ExecErrorCode::SpawnError,
                    "Empty argv: no program to run",
                ));
            }
            let program = PathBuf::from(argv.remove(0));
            (program, argv)
        }
    };
    if let Err(error) = tokio::fs::metadata(&cwd).await {
        let message = format!("Working directory does not exist: {}", cwd.display());
        return Err(ExecError::io(ExecErrorCode::SpawnError, message, error));
    }
    if ctx.is_cancelled() {
        return Err(ExecError::aborted());
    }

    let mut process = ProcessCommand::new(&program);
    process
        .args(&args)
        .current_dir(&cwd)
        .stdin(if options.stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if options.inherit_env {
        process.envs(config.shell_env);
    } else {
        process.env_clear();
    }
    process.envs(&options.env);
    #[cfg(unix)]
    process.process_group(0);
    #[cfg(windows)]
    process.creation_flags(0x0800_0000); // CREATE_NO_WINDOW

    let mut child = match process.spawn() {
        Ok(child) => child,
        Err(error) => {
            let message = format!("Failed to start {}: {error}", program.display());
            return Err(ExecError::io(ExecErrorCode::SpawnError, message, error));
        }
    };
    let Some(pid) = child.id() else {
        return Err(ExecError::new(
            ExecErrorCode::SpawnError,
            "The command exited before it was tracked",
        ));
    };
    if let (Some(input), Some(mut pipe)) = (options.stdin, child.stdin.take()) {
        // Written beside the run, so a command that never reads its input cannot stall it;
        // dropping the pipe closes the input.
        tokio::spawn(async move {
            let _ = pipe.write_all(&input).await;
        });
    }
    let tracked = config.children.track(ProcessTree::new(pid));
    let mut run = Run {
        tree: tracked.tree(),
        output: Output::new(options.on_output),
        spill: Spill::new(options.spill),
        stop: None,
        failure: None,
        killed: false,
    };
    let deadline = timeout.map(|timeout| Instant::now() + timeout);
    let status = run.drive(&mut child, deadline, ctx).await;
    tracked.settle();
    run.settle(status, timeout).await
}

/// Why a command was killed from outside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stop {
    Timeout,
    Aborted,
}

/// One step of [`Run::drive`].
#[derive(Debug)]
enum Event {
    Read(OutputStream, io::Result<usize>),
    Exit(io::Result<ExitStatus>),
    Stop(Stop),
    /// The pipes stayed quiet for the grace period after the exit.
    Quiet,
}

/// The state of one running command.
struct Run<'o> {
    tree: ProcessTree,
    output: Output<'o>,
    spill: Spill,
    stop: Option<Stop>,
    /// A spill failure: the command is killed and fails with it.
    failure: Option<ExecError>,
    killed: bool,
}

impl Run<'_> {
    /// Drive the command until it exited and its pipes ended or stayed quiet.
    async fn drive(&mut self, child: &mut Child, deadline: Option<Instant>, ctx: &Context) -> io::Result<ExitStatus> {
        let mut stdout = child.stdout.take();
        let mut stderr = child.stderr.take();
        let mut stdout_buf = vec![0; PIPE_CHUNK];
        let mut stderr_buf = vec![0; PIPE_CHUNK];
        let mut status: Option<io::Result<ExitStatus>> = None;
        let mut quiet_at: Option<Instant> = None;
        while status.is_none() || stdout.is_some() || stderr.is_some() {
            let running = self.stop.is_none();
            let event = tokio::select! {
                () = ctx.cancelled(), if running => Event::Stop(Stop::Aborted),
                () = sleep_until_some(deadline), if running => Event::Stop(Stop::Timeout),
                read = read_some(&mut stdout, &mut stdout_buf) => Event::Read(OutputStream::Stdout, read),
                read = read_some(&mut stderr, &mut stderr_buf) => Event::Read(OutputStream::Stderr, read),
                exited = child.wait(), if status.is_none() => Event::Exit(exited),
                () = sleep_until_some(quiet_at) => Event::Quiet,
            };
            match event {
                Event::Read(stream, Ok(read)) if read > 0 => {
                    let chunk = match stream {
                        OutputStream::Stdout => &stdout_buf[..read],
                        OutputStream::Stderr => &stderr_buf[..read],
                    };
                    self.output.deliver(stream, chunk);
                    if let Err(error) = self.spill.push(chunk, ctx).await {
                        self.failure.get_or_insert(error);
                    }
                    if status.is_some() {
                        quiet_at = Some(Instant::now() + EXIT_STDIO_GRACE);
                    }
                }
                // The end of a pipe; a read error ends it too.
                Event::Read(OutputStream::Stdout, _) => stdout = None,
                Event::Read(OutputStream::Stderr, _) => stderr = None,
                Event::Exit(exited) => {
                    status = Some(exited);
                    quiet_at = Some(Instant::now() + EXIT_STDIO_GRACE);
                }
                Event::Stop(stop) => self.stop = Some(stop),
                Event::Quiet => break,
            }
            if (self.stop.is_some() || self.failure.is_some() || self.output.failed()) && !self.killed {
                self.killed = true;
                self.tree.kill().await;
                let _ = child.start_kill();
            }
        }
        // Quiet pipes are armed only after the exit, so the status is there.
        status.unwrap_or_else(|| Err(io::Error::other("the command has no exit status")))
    }

    /// The result of a command whose drive ended with `status`.
    async fn settle(
        mut self,
        status: io::Result<ExitStatus>,
        timeout: Option<Duration>,
    ) -> Result<ExecResult, ExecError> {
        self.output.finish();
        let spill_path = self.spill.path().map(str::to_owned);
        let spilled = self.spill.finish().await;
        if let Some(error) = self.output.failure.take() {
            return Err(error);
        }
        match self.stop {
            Some(Stop::Timeout) => {
                let seconds = timeout.unwrap_or_default().as_secs_f64();
                let error = ExecError::new(ExecErrorCode::Timeout, format!("timeout:{seconds}"));
                return Err(error.with_spill_path(spill_path));
            }
            Some(Stop::Aborted) => return Err(ExecError::aborted().with_spill_path(spill_path)),
            None => {}
        }
        if let Some(error) = self.failure.take() {
            return Err(error);
        }
        let spill_path = spilled?;
        match status {
            Ok(status) => Ok(ExecResult {
                exit_code: exit_code(status),
                spill_path,
            }),
            Err(error) => Err(ExecError::io(
                ExecErrorCode::Unknown,
                format!("Failed to wait for the command: {error}"),
                error,
            )),
        }
    }
}

/// Read from a pipe that is still open; never resolves for an ended one.
async fn read_some(pipe: &mut Option<impl AsyncRead + Unpin>, buf: &mut [u8]) -> io::Result<usize> {
    match pipe {
        Some(pipe) => pipe.read(buf).await,
        None => std::future::pending().await,
    }
}

/// Sleep until `at`; never resolves without one.
async fn sleep_until_some(at: Option<Instant>) {
    match at {
        Some(at) => sleep_until(at).await,
        None => std::future::pending().await,
    }
}

/// Decodes each stream and hands the text to the output callback.
struct Output<'o> {
    callback: Option<OutputCallback<'o>>,
    stdout: Utf8Decoder,
    stderr: Utf8Decoder,
    /// The callback panicked.
    failure: Option<ExecError>,
}

impl<'o> Output<'o> {
    fn new(callback: Option<OutputCallback<'o>>) -> Self {
        Self {
            callback,
            stdout: Utf8Decoder::stripping_bom(),
            stderr: Utf8Decoder::stripping_bom(),
            failure: None,
        }
    }

    fn failed(&self) -> bool {
        self.failure.is_some()
    }

    fn deliver(&mut self, stream: OutputStream, bytes: &[u8]) {
        if self.callback.is_none() {
            return;
        }
        let text = self.decoder(stream).decode_to_string(bytes);
        self.emit(stream, text);
    }

    /// End both streams: incomplete trailing characters become U+FFFD.
    fn finish(&mut self) {
        if self.callback.is_none() {
            return;
        }
        for stream in [OutputStream::Stdout, OutputStream::Stderr] {
            let text = self.decoder(stream).finish_to_string();
            self.emit(stream, text);
        }
    }

    fn decoder(&mut self, stream: OutputStream) -> &mut Utf8Decoder {
        match stream {
            OutputStream::Stdout => &mut self.stdout,
            OutputStream::Stderr => &mut self.stderr,
        }
    }

    fn emit(&mut self, stream: OutputStream, text: String) {
        let Some(callback) = &mut self.callback else { return };
        if text.is_empty() {
            return;
        }
        let chunk = OutputChunk {
            text,
            stream,
            skipped: None,
        };
        if let Err(panic) = catch_unwind(AssertUnwindSafe(|| callback(chunk))) {
            // Nothing more reaches a callback that panicked.
            self.callback = None;
            let message = panic
                .downcast_ref::<&str>()
                .map(|message| (*message).to_owned())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "The output callback panicked".to_owned());
            self.failure = Some(ExecError::new(ExecErrorCode::CallbackError, message));
        }
    }
}

/// The shell that runs [`Command::Shell`] strings: the configured one, else `bash`, else (on
/// Unix) `sh`.
async fn find_shell(configured: Option<&Path>) -> Result<PathBuf, ExecError> {
    if let Some(path) = configured {
        return if is_file(path).await {
            Ok(path.to_path_buf())
        } else {
            Err(ExecError::new(
                ExecErrorCode::ShellUnavailable,
                format!("Custom shell path not found: {}", path.display()),
            ))
        };
    }
    for candidate in bash_candidates() {
        if is_file(&candidate).await {
            return Ok(candidate);
        }
    }
    if let Some(bash) = bash_on_path().await {
        return Ok(bash);
    }
    if cfg!(unix) {
        return Ok(PathBuf::from("sh"));
    }
    Err(ExecError::new(
        ExecErrorCode::ShellUnavailable,
        "No bash shell found: install Git for Windows, add bash to PATH, or configure a shell path",
    ))
}

async fn is_file(path: &Path) -> bool {
    tokio::fs::metadata(path).await.is_ok_and(|metadata| metadata.is_file())
}

/// Where bash usually is, before `PATH` is searched.
fn bash_candidates() -> Vec<PathBuf> {
    if cfg!(windows) {
        ["ProgramFiles", "ProgramFiles(x86)"]
            .iter()
            .filter_map(std::env::var_os)
            .map(|root| PathBuf::from(root).join("Git").join("bin").join("bash.exe"))
            .collect()
    } else {
        vec![PathBuf::from("/bin/bash")]
    }
}

/// The first `bash` on `PATH`, skipping the legacy WSL launcher on Windows.
async fn bash_on_path() -> Option<PathBuf> {
    let name = if cfg!(windows) { "bash.exe" } else { "bash" };
    let path = std::env::var_os("PATH")?;
    for directory in std::env::split_paths(&path) {
        let candidate = directory.join(name);
        if !is_legacy_wsl_bash(&candidate) && is_file(&candidate).await {
            return Some(candidate);
        }
    }
    None
}

/// Whether `path` is `%SystemRoot%\System32\bash.exe` (or `Sysnative`), which starts WSL
/// instead of a bash that sees the environment's files.
fn is_legacy_wsl_bash(path: &Path) -> bool {
    if !cfg!(windows) {
        return false;
    }
    let root = std::env::var_os("SystemRoot").map_or_else(|| PathBuf::from("C:\\Windows"), PathBuf::from);
    let lowered = path.to_string_lossy().to_lowercase();
    ["System32", "Sysnative"]
        .iter()
        .any(|directory| root.join(directory).join("bash.exe").to_string_lossy().to_lowercase() == lowered)
}
