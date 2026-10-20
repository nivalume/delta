//! What [`Shell::exec`](super::Shell::exec) takes and returns: the command, its options, the
//! output chunks it streams and its result.

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use delta_chord::output::{OutputLimits, OutputSkip, PROGRESS_BYTES_PER_SECOND, Retain};

use super::error::{ExecError, ExecErrorCode};

/// The longest timeout an environment accepts, as upstream (the largest timer a JavaScript
/// runtime arms); longer ones fail before anything is spawned.
pub const MAX_TIMEOUT: Duration = Duration::from_millis(2_147_483_647);

/// A command to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// A script for the environment's shell (`bash -c` locally).
    Shell(String),
    /// A program and its arguments, passed as they are: no shell parses them.
    Argv(Vec<String>),
}

impl Command {
    pub fn shell(script: impl Into<String>) -> Self {
        Self::Shell(script.into())
    }

    pub fn argv<I, S>(argv: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::Argv(argv.into_iter().map(Into::into).collect())
    }
}

/// Receives every decoded output chunk of a command. A panic in it kills the command, which
/// then fails with [`ExecErrorCode::CallbackError`].
pub type OutputCallback<'a> = Box<dyn FnMut(OutputChunk) + Send + 'a>;

/// How to run a command. `ExecOptions::default()` runs it in the environment's `cwd` with the
/// inherited environment variables, no timeout and no output callback.
pub struct ExecOptions<'a> {
    /// Working directory, resolved against the environment's `cwd`.
    pub cwd: Option<String>,
    /// Variables set for the command, over the inherited ones.
    pub env: BTreeMap<String, String>,
    /// Start from the environment's variables (default); `false` passes only [`env`](Self::env).
    pub inherit_env: bool,
    /// Kill the command after this long; at most [`MAX_TIMEOUT`].
    pub timeout: Option<Duration>,
    pub on_output: Option<OutputCallback<'a>>,
    /// Stream the complete output to a temporary file once it crosses these thresholds.
    pub spill: Option<SpillOptions>,
    /// The tail of the output the caller keeps; see [`ShellOutputWindow`].
    pub window: Option<ShellOutputWindow>,
}

impl Default for ExecOptions<'_> {
    fn default() -> Self {
        Self {
            cwd: None,
            env: BTreeMap::new(),
            inherit_env: true,
            timeout: None,
            on_output: None,
            spill: None,
            window: None,
        }
    }
}

impl fmt::Debug for ExecOptions<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecOptions")
            .field("cwd", &self.cwd)
            .field("env", &self.env.keys().collect::<Vec<_>>())
            .field("inherit_env", &self.inherit_env)
            .field("timeout", &self.timeout)
            .field("on_output", &self.on_output.is_some())
            .field("spill", &self.spill)
            .field("window", &self.window)
            .finish()
    }
}

impl<'a> ExecOptions<'a> {
    #[must_use]
    pub fn with_cwd(mut self, cwd: impl Into<String>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    #[must_use]
    pub fn with_env(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(name.into(), value.into());
        self
    }

    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    #[must_use]
    pub fn on_output(mut self, callback: impl FnMut(OutputChunk) + Send + 'a) -> Self {
        self.on_output = Some(Box::new(callback));
        self
    }

    #[must_use]
    pub fn with_spill(mut self, spill: SpillOptions) -> Self {
        self.spill = Some(spill);
        self
    }

    #[must_use]
    pub fn with_window(mut self, window: ShellOutputWindow) -> Self {
        self.window = Some(window);
        self
    }

    /// The timeout, checked against [`MAX_TIMEOUT`]. An invalid timeout fails with
    /// [`ExecErrorCode::Timeout`] and an `Invalid timeout` message, as upstream.
    pub fn checked_timeout(&self) -> Result<Option<Duration>, ExecError> {
        match self.timeout {
            Some(timeout) if timeout.is_zero() => Err(ExecError::new(
                ExecErrorCode::Timeout,
                "Invalid timeout: must be positive",
            )),
            Some(timeout) if timeout > MAX_TIMEOUT => Err(ExecError::new(
                ExecErrorCode::Timeout,
                format!("Invalid timeout: maximum is {} seconds", MAX_TIMEOUT.as_secs_f64()),
            )),
            timeout => Ok(timeout),
        }
    }
}

/// Spill thresholds: once the output seen so far exceeds either, the complete raw output goes
/// to a temporary file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpillOptions {
    pub after_bytes: u64,
    /// Complete or partial lines.
    pub after_lines: u64,
}

/// The tail of the combined output a caller keeps, and how often it samples it.
///
/// An environment that moves output over a slow link may omit output the caller would drop
/// anyway and report it as [`OutputChunk::skipped`]; it omits only output followed by more than
/// the window (more than `max_bytes` bytes or more than `max_lines` newlines), delivers all of
/// that following output in the same chunk, and sends no faster than the pace. The local
/// environment never omits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShellOutputWindow {
    /// UTF-8 bytes of decoded text kept at the end of the output.
    pub max_bytes: usize,
    /// Lines kept at the end of the output.
    pub max_lines: usize,
    /// Minimum pause between the caller's samples.
    pub min_interval: Duration,
    /// Each sample also pauses the caller in proportion to its size at this rate.
    pub bytes_per_second: usize,
}

impl ShellOutputWindow {
    /// A window paced as [`delta_chord::output::Progress`] commits are.
    pub fn new(max_bytes: usize, max_lines: usize, min_interval: Duration) -> Self {
        Self {
            max_bytes,
            max_lines,
            min_interval,
            bytes_per_second: PROGRESS_BYTES_PER_SECOND,
        }
    }

    /// The tail limits of an [`OutputBuffer`](delta_chord::output::OutputBuffer) that keeps
    /// this window.
    pub fn limits(&self) -> OutputLimits {
        OutputLimits {
            max_bytes: self.max_bytes,
            max_lines: self.max_lines,
            retain: Retain::Tail,
        }
    }
}

/// Which pipe a chunk came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputStream {
    Stdout,
    Stderr,
}

/// One decoded piece of a command's output, raw: not bounded, sanitized or throttled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputChunk {
    pub text: String,
    /// The stream of the chunk; for a windowed chunk spanning both, the last one.
    pub stream: OutputStream,
    /// Output the environment omitted right before this chunk; only with a window. Feed it to
    /// [`OutputBuffer::skip`](delta_chord::output::OutputBuffer::skip) before the text.
    pub skipped: Option<OutputSkip>,
}

/// A command that ran to completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecResult {
    /// The process's exit code; `128 + signal` when a signal killed it; `1` when unknown.
    pub exit_code: i32,
    /// The temporary file holding the complete raw output, when it crossed the spill
    /// thresholds.
    pub spill_path: Option<String>,
}
