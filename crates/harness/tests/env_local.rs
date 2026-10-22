//! Behaviour of `LocalEnv` beyond the conformance suite, ported from upstream's
//! `env-node.test.ts` and `env-node-spill.test.ts`: file operations, text line readers, the
//! shell, spills and process trees.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use delta_chord::context::Context;
use delta_harness::env::{
    Command, ExecErrorCode, ExecOptions, FileErrorCode, FileKind, FileSystem, LocalEnv, RemoveOptions, Shell,
    SpillOptions, TempFileOptions, TextLine,
};
use pretty_assertions::assert_eq;

/// A unique directory under the system temp directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("delta-env-local-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir(&path).expect("create a temp dir");
        // Canonical, so paths the environment reports compare equal on systems with a linked
        // temp directory.
        Self(std::fs::canonicalize(&path).expect("canonical temp dir"))
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn join(&self, path: &str) -> String {
        self.0.join(path).to_string_lossy().into_owned()
    }

    fn env(&self) -> LocalEnv {
        LocalEnv::new(&self.0)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Removes a spill file on drop.
struct Spilled(String);

impl Drop for Spilled {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn ctx() -> Context {
    Context::background()
}

fn aborted() -> Context {
    let (ctx, canceller) = Context::background().with_cancel();
    canceller.cancel();
    ctx
}

/// Run a shell string and collect the combined output.
async fn run(env: &LocalEnv, script: &str, options: ExecOptions<'_>) -> (Result<i32, (ExecErrorCode, String)>, String) {
    let mut output = String::new();
    let options = ExecOptions {
        on_output: Some(Box::new(|chunk| output.push_str(&chunk.text))),
        ..options
    };
    let result = env.exec(Command::shell(script), options, &ctx()).await;
    let result = result
        .map(|result| result.exit_code)
        .map_err(|error| (error.code, error.message));
    (result, output)
}

#[tokio::test]
async fn reads_writes_lists_and_removes_files() {
    let dir = TempDir::new();
    let env = dir.env();
    let ctx = ctx();
    assert_eq!(env.id(), "local");
    assert_eq!(env.absolute_path("nested/./child/..").unwrap(), dir.join("nested"));
    assert_eq!(env.join_path(&[env.cwd(), "a", "b"]).unwrap(), dir.join("a/b"));
    env.create_dir("nested/child", true, &ctx).await.unwrap();
    env.write_file("nested/child/file.txt", b"hel", &ctx).await.unwrap();
    env.append_file("nested/child/file.txt", b"lo", &ctx).await.unwrap();
    assert_eq!(
        env.read_text_file("nested/child/file.txt", &ctx).await.unwrap(),
        "hello"
    );
    assert_eq!(
        env.read_binary_file("nested/child/file.txt", &ctx).await.unwrap(),
        b"hello"
    );
    assert_eq!(
        env.read_text_lines("nested/child/file.txt", Some(1), &ctx)
            .await
            .unwrap(),
        ["hello"]
    );

    let entries = env.list_dir("nested/child", &ctx).await.unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(
        (
            entries[0].name.as_str(),
            entries[0].path.clone(),
            entries[0].kind,
            entries[0].size
        ),
        ("file.txt", dir.join("nested/child/file.txt"), FileKind::File, 5)
    );
    assert!(entries[0].mtime_ms > 0);

    assert!(env.exists("nested/child/file.txt", &ctx).await.unwrap());
    env.remove("nested/child/file.txt", RemoveOptions::default(), &ctx)
        .await
        .unwrap();
    assert!(!env.exists("nested/child/file.txt", &ctx).await.unwrap());
    let missing = env.file_info("missing.txt", &ctx).await.unwrap_err();
    assert_eq!(
        (missing.code, missing.path),
        (FileErrorCode::NotFound, Some(dir.join("missing.txt")))
    );
    env.write_file("plain.txt", b"x", &ctx).await.unwrap();
    assert_eq!(
        env.list_dir("plain.txt", &ctx).await.unwrap_err().code,
        FileErrorCode::NotDirectory
    );
}

#[cfg(unix)]
#[tokio::test]
async fn reports_symbolic_links_without_following_them() {
    let dir = TempDir::new();
    let env = dir.env();
    let ctx = ctx();
    env.write_file("dir/file.txt", b"hello", &ctx).await.unwrap();
    std::os::unix::fs::symlink(dir.join("dir/file.txt"), dir.join("file-link")).unwrap();
    let info = env.file_info("file-link", &ctx).await.unwrap();
    assert_eq!((info.name.as_str(), info.kind), ("file-link", FileKind::Symlink));
    assert_eq!(
        env.canonical_path("file-link", &ctx).await.unwrap(),
        dir.join("dir/file.txt")
    );
    let mut kinds: Vec<_> = env
        .list_dir(".", &ctx)
        .await
        .unwrap()
        .into_iter()
        .map(|entry| (entry.name, entry.kind))
        .collect();
    kinds.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        kinds,
        [
            ("dir".to_owned(), FileKind::Directory),
            ("file-link".to_owned(), FileKind::Symlink)
        ]
    );
}

#[tokio::test]
async fn renames_truncates_and_flushes() {
    let dir = TempDir::new();
    let env = dir.env();
    let ctx = ctx();
    env.write_file("source.txt", b"new", &ctx).await.unwrap();
    env.write_file("destination.txt", b"old", &ctx).await.unwrap();
    env.rename_file("source.txt", "destination.txt", &ctx).await.unwrap();
    assert_eq!(env.read_text_file("destination.txt", &ctx).await.unwrap(), "new");
    let failed = env
        .rename_file("missing.txt", "destination.txt", &ctx)
        .await
        .unwrap_err();
    assert_eq!(
        (failed.code, failed.path),
        (FileErrorCode::NotFound, Some(dir.join("missing.txt")))
    );

    env.write_file("file.bin", &[0x61, 0xc3, 0xa9, 0x0a], &ctx)
        .await
        .unwrap();
    env.truncate_file("file.bin", 2, &ctx).await.unwrap();
    assert_eq!(env.read_binary_file("file.bin", &ctx).await.unwrap(), [0x61, 0xc3]);
    env.truncate_file("file.bin", 4, &ctx).await.unwrap();
    assert_eq!(
        env.read_binary_file("file.bin", &ctx).await.unwrap(),
        [0x61, 0xc3, 0, 0]
    );
    assert_eq!(
        env.truncate_file("missing.bin", 0, &ctx).await.unwrap_err().code,
        FileErrorCode::NotFound
    );
    assert!(!env.exists("missing.bin", &ctx).await.unwrap());

    env.flush_file("file.bin", &ctx).await.unwrap();
    assert_eq!(
        env.flush_file("missing.bin", &ctx).await.unwrap_err().code,
        FileErrorCode::NotFound
    );
    env.create_dir("dir", true, &ctx).await.unwrap();
    assert_eq!(
        env.flush_file("dir", &ctx).await.unwrap_err().code,
        FileErrorCode::IsDirectory
    );
}

#[tokio::test]
async fn honors_create_dir_and_remove_options() {
    let dir = TempDir::new();
    let env = dir.env();
    let ctx = ctx();
    assert_eq!(
        env.create_dir("missing/child", false, &ctx).await.unwrap_err().code,
        FileErrorCode::NotFound
    );
    env.write_file("dir/child/file.txt", b"x", &ctx).await.unwrap();
    let plain = env.remove("dir", RemoveOptions::default(), &ctx).await;
    assert_eq!(plain.unwrap_err().code, FileErrorCode::IsDirectory);
    let recursive = RemoveOptions {
        recursive: true,
        force: false,
    };
    env.remove("dir", recursive, &ctx).await.unwrap();
    assert!(!env.exists("dir", &ctx).await.unwrap());
    assert_eq!(
        env.remove("missing", RemoveOptions::default(), &ctx)
            .await
            .unwrap_err()
            .code,
        FileErrorCode::NotFound
    );
    let force = RemoveOptions {
        recursive: false,
        force: true,
    };
    env.remove("missing", force, &ctx).await.unwrap();
}

#[tokio::test]
async fn creates_private_temp_dirs_and_files() {
    let dir = TempDir::new();
    let env = dir.env();
    let ctx = ctx();
    let temp_dir = env.create_temp_dir(Some("delta-env-test-"), &ctx).await.unwrap();
    let options = TempFileOptions {
        prefix: Some("prefix-".to_owned()),
        suffix: Some(".txt".to_owned()),
    };
    let temp_file = env.create_temp_file(options, &ctx).await.unwrap();
    let file_name = Path::new(&temp_file)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert!(file_name.starts_with("prefix-"), "{file_name}");
    assert_eq!(Path::new(&file_name).extension(), Some("txt".as_ref()));
    assert_eq!(env.read_text_file(&temp_file, &ctx).await.unwrap(), "");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = |path: &str| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!((mode(&temp_dir), mode(&temp_file)), (0o700, 0o600));
    }
    std::fs::remove_dir(&temp_dir).unwrap();
    std::fs::remove_file(&temp_file).unwrap();
}

#[tokio::test]
async fn pre_aborted_file_operations_fail_without_side_effects() {
    let dir = TempDir::new();
    let env = dir.env();
    env.write_file("file.txt", b"hello", &ctx()).await.unwrap();
    let ctx = aborted();
    let codes = [
        env.read_text_file("file.txt", &ctx).await.err(),
        env.read_text_lines("file.txt", None, &ctx).await.err(),
        env.read_binary_file("file.txt", &ctx).await.err(),
        env.open_text_line_reader("file.txt", &ctx).await.err(),
        env.write_file("other.txt", b"hello", &ctx).await.err(),
        env.append_file("file.txt", b" world", &ctx).await.err(),
        env.truncate_file("file.txt", 1, &ctx).await.err(),
        env.flush_file("file.txt", &ctx).await.err(),
        env.rename_file("file.txt", "renamed.txt", &ctx).await.err(),
        env.file_info("file.txt", &ctx).await.err(),
        env.list_dir(".", &ctx).await.err(),
        env.canonical_path("file.txt", &ctx).await.err(),
        env.exists("file.txt", &ctx).await.err(),
        env.create_dir("dir", true, &ctx).await.err(),
        env.remove("file.txt", RemoveOptions::default(), &ctx).await.err(),
        env.create_temp_dir(None, &ctx).await.err(),
        env.create_temp_file(TempFileOptions::default(), &ctx).await.err(),
    ];
    for code in codes.map(|error| error.map(|error| error.code)) {
        assert_eq!(code, Some(FileErrorCode::Aborted));
    }
    assert_eq!(env.read_text_file("file.txt", &self::ctx()).await.unwrap(), "hello");
    let names: Vec<_> = env
        .list_dir(".", &self::ctx())
        .await
        .unwrap()
        .into_iter()
        .map(|entry| entry.name)
        .collect();
    assert_eq!(names, ["file.txt"]);
}

#[tokio::test]
async fn text_line_reader_reports_terminated_lines_and_keeps_carriage_returns() {
    let dir = TempDir::new();
    let env = dir.env();
    let ctx = ctx();
    env.write_file("lines.txt", b"\xef\xbb\xbfone\r\n\ntwo\npartial", &ctx)
        .await
        .unwrap();
    let mut reader = env.open_text_line_reader("lines.txt", &ctx).await.unwrap();
    let mut lines = Vec::new();
    while let Some(line) = reader.read_line(&ctx).await.unwrap() {
        lines.push(line);
    }
    let line = |text: &str, terminated| TextLine {
        text: text.to_owned(),
        terminated,
    };
    assert_eq!(
        lines,
        [
            line("one\r", true),
            line("", true),
            line("two", true),
            line("partial", false)
        ]
    );
    assert_eq!(reader.read_line(&ctx).await.unwrap(), None);
    reader.close(&ctx).await;

    env.write_file("newline.txt", b"\n", &ctx).await.unwrap();
    assert_eq!(env.read_text_lines("newline.txt", None, &ctx).await.unwrap(), [""]);
    env.write_file("empty.txt", b"", &ctx).await.unwrap();
    assert!(env.read_text_lines("empty.txt", None, &ctx).await.unwrap().is_empty());
    assert_eq!(
        env.open_text_line_reader("missing.txt", &ctx)
            .await
            .err()
            .map(|error| error.code),
        Some(FileErrorCode::NotFound)
    );
}

#[tokio::test]
async fn text_line_reader_decodes_characters_split_across_reads() {
    let dir = TempDir::new();
    let env = dir.env();
    let ctx = ctx();
    // The reader reads 64 KiB at a time; a four-byte character straddles the first boundary.
    let first = format!("{}😀tail", "a".repeat(64 * 1024 - 2));
    let second = "é".repeat(100_000);
    env.write_file("large.txt", format!("{first}\n{second}").as_bytes(), &ctx)
        .await
        .unwrap();
    let mut reader = env.open_text_line_reader("large.txt", &ctx).await.unwrap();
    assert_eq!(
        reader.read_line(&ctx).await.unwrap(),
        Some(TextLine {
            text: first,
            terminated: true
        })
    );
    assert_eq!(
        reader.read_line(&ctx).await.unwrap(),
        Some(TextLine {
            text: second,
            terminated: false
        })
    );
    assert_eq!(reader.read_line(&ctx).await.unwrap(), None);
    // An aborted read consumes nothing.
    assert_eq!(
        reader.read_line(&aborted()).await.unwrap_err().code,
        FileErrorCode::Aborted
    );
    reader.close(&ctx).await;
}

#[tokio::test]
async fn runs_commands_in_cwd_with_env_overrides() {
    let dir = TempDir::new();
    let mut shell_env = BTreeMap::new();
    shell_env.insert("DELTA_ENV_CONFIGURED".to_owned(), "configured".to_owned());
    shell_env.insert("DELTA_ENV_SHADOWED".to_owned(), "stale".to_owned());
    let env = dir.env().with_shell_env(shell_env);
    // Git Bash's `$PWD` is `/c/...`; `pwd -W` prints the Windows path.
    let pwd = if cfg!(windows) { "$(pwd -W)" } else { "$PWD" };
    let script = format!(
        r#"printf '%s|%s|%s|%s' "{pwd}" "$DELTA_ENV_CONFIGURED" "${{DELTA_ENV_SHADOWED-unset}}" "${{HOME+home}}""#
    );
    let options = ExecOptions::default().with_env("DELTA_ENV_SHADOWED", "");
    let (result, output) = run(&env, &script, options).await;
    assert_eq!(result, Ok(0));
    let (cwd, rest) = output.split_once('|').unwrap();
    assert_eq!(
        std::fs::canonicalize(cwd).unwrap(),
        std::fs::canonicalize(dir.path()).unwrap()
    );
    assert_eq!(rest, "configured||home");

    let isolated = ExecOptions {
        inherit_env: false,
        ..ExecOptions::default().with_env("DELTA_ENV_EXPLICIT", "explicit")
    };
    // An inherited variable must not reach the command. Git Bash fills in `HOME` itself when it
    // is missing, so Windows probes `USERPROFILE`.
    let inherited = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    let script = format!(r#"printf '%s:%s:%s' "${{{inherited}-}}" "${{DELTA_ENV_CONFIGURED-}}" "$DELTA_ENV_EXPLICIT""#);
    let (result, output) = run(&env, &script, isolated).await;
    assert_eq!((result, output.as_str()), (Ok(0), "::explicit"));
}

#[tokio::test]
async fn reports_exit_codes_signals_and_spawn_failures() {
    let dir = TempDir::new();
    let env = dir.env();
    assert_eq!(run(&env, "exit 7", ExecOptions::default()).await.0, Ok(7));
    #[cfg(unix)]
    assert_eq!(run(&env, "kill -9 $$", ExecOptions::default()).await.0, Ok(128 + 9));

    let missing_cwd = LocalEnv::new(dir.path().join("missing"));
    let (result, _) = run(&missing_cwd, "printf ok", ExecOptions::default()).await;
    let (code, message) = result.unwrap_err();
    assert_eq!(code, ExecErrorCode::SpawnError);
    assert!(message.starts_with("Working directory does not exist: "), "{message}");

    let missing_shell = dir.env().with_shell(dir.path().join("missing-shell"));
    let (result, _) = run(&missing_shell, "printf ok", ExecOptions::default()).await;
    assert_eq!(result.unwrap_err().0, ExecErrorCode::ShellUnavailable);
    env.write_file("not-executable", b"not executable", &ctx())
        .await
        .unwrap();
    let not_executable = dir.env().with_shell(dir.path().join("not-executable"));
    let (result, _) = run(&not_executable, "printf ok", ExecOptions::default()).await;
    assert_eq!(result.unwrap_err().0, ExecErrorCode::SpawnError);
}

#[tokio::test]
async fn rejects_invalid_timeouts_before_spawning() {
    let dir = TempDir::new();
    let env = dir.env();
    for timeout in [Duration::ZERO, Duration::from_millis(2_147_483_648)] {
        let (result, _) = run(&env, "touch spawned", ExecOptions::default().with_timeout(timeout)).await;
        let (code, message) = result.unwrap_err();
        assert_eq!(code, ExecErrorCode::Timeout);
        assert!(message.starts_with("Invalid timeout"), "{message}");
    }
    assert!(!env.exists("spawned", &ctx()).await.unwrap());
    let (result, _) = run(
        &env,
        "sleep 5",
        ExecOptions::default().with_timeout(Duration::from_millis(50)),
    )
    .await;
    assert_eq!(result.unwrap_err(), (ExecErrorCode::Timeout, "timeout:0.05".to_owned()));
}

#[tokio::test]
async fn pre_aborted_commands_never_start() {
    let dir = TempDir::new();
    let env = dir.env();
    let result = env
        .exec(Command::shell("touch spawned"), ExecOptions::default(), &aborted())
        .await;
    assert_eq!(result.unwrap_err().code, ExecErrorCode::Aborted);
    assert!(!env.exists("spawned", &ctx()).await.unwrap());
}

#[tokio::test]
async fn a_panicking_output_callback_kills_the_command() {
    let dir = TempDir::new();
    let env = dir.env();
    let options = ExecOptions::default().on_output(|_| panic!("callback failed"));
    let started = Instant::now();
    let result = env.exec(Command::shell("printf out; sleep 5"), options, &ctx()).await;
    let error = result.unwrap_err();
    assert_eq!(
        (error.code, error.message.as_str()),
        (ExecErrorCode::CallbackError, "callback failed")
    );
    assert!(started.elapsed() < Duration::from_secs(4), "the command was killed");
}

#[tokio::test]
async fn decodes_characters_split_across_pipe_chunks() {
    let dir = TempDir::new();
    let env = dir.env();
    let (result, output) = run(
        &env,
        r"printf '\360\237'; sleep 0.05; printf '\230\200'",
        ExecOptions::default(),
    )
    .await;
    assert_eq!((result, output.as_str()), (Ok(0), "😀"));
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn a_timeout_kills_the_whole_process_tree() {
    let dir = TempDir::new();
    let env = dir.env();
    let options = ExecOptions::default().with_timeout(Duration::from_millis(200));
    let started = Instant::now();
    let (result, _) = run(&env, "sleep 30 & echo $! > child.pid; wait", options).await;
    assert_eq!(result.unwrap_err().0, ExecErrorCode::Timeout);
    assert!(started.elapsed() < Duration::from_secs(5));
    let pid = env.read_text_file("child.pid", &ctx()).await.unwrap();
    assert!(is_dead(pid.trim()));
}

/// Whether process `pid` is gone, or a zombie where no init process reaps orphans.
#[cfg(target_os = "linux")]
fn is_dead(pid: &str) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat")).map_or(true, |stat| {
        stat.rsplit_once(") ").is_some_and(|(_, rest)| rest.starts_with('Z'))
    })
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn dropping_an_exec_future_kills_its_process_tree() {
    let dir = TempDir::new();
    let env = dir.env();
    let ctx = ctx();
    let running = env.exec(
        Command::shell("sleep 30 & echo $! > child.pid; wait"),
        ExecOptions::default(),
        &ctx,
    );
    assert!(tokio::time::timeout(Duration::from_millis(300), running).await.is_err());
    let pid = env.read_text_file("child.pid", &ctx).await.unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !is_dead(pid.trim()) {
        assert!(
            Instant::now() < deadline,
            "the background child outlived the dropped command"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn settles_while_a_background_descendant_holds_the_pipes() {
    let dir = TempDir::new();
    let env = dir.env();
    let started = Instant::now();
    let (result, output) = run(&env, "sleep 3 & printf done", ExecOptions::default()).await;
    assert_eq!((result, output.as_str()), (Ok(0), "done"));
    assert!(started.elapsed() < Duration::from_secs(2));
    env.cleanup(&ctx()).await;
}

#[tokio::test]
async fn cleanup_terminates_running_commands() {
    let dir = TempDir::new();
    let env = dir.env();
    let ctx = ctx();
    let running = env.exec(Command::shell("touch started; sleep 60"), ExecOptions::default(), &ctx);
    let stop = async {
        while !env.exists("started", &ctx).await.unwrap() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        env.cleanup(&ctx).await;
    };
    let (result, ()) = tokio::time::timeout(Duration::from_secs(10), async { tokio::join!(running, stop) })
        .await
        .expect("cleanup ends the command");
    // Killed by SIGKILL; Windows has no signals and `TerminateProcess` leaves exit code 1.
    assert_eq!(result.unwrap().exit_code, if cfg!(windows) { 1 } else { 128 + 9 });
}

#[tokio::test]
async fn does_not_spill_below_the_thresholds() {
    let dir = TempDir::new();
    let env = dir.env();
    let spill = SpillOptions {
        after_bytes: 100,
        after_lines: 10,
    };
    let result = env
        .exec(
            Command::shell("printf short"),
            ExecOptions::default().with_spill(spill),
            &ctx(),
        )
        .await;
    assert_eq!(result.unwrap().spill_path, None);
}

#[tokio::test]
async fn spills_exact_raw_bytes_while_streaming_decoded_text() {
    let dir = TempDir::new();
    let env = dir.env();
    let spill = SpillOptions {
        after_bytes: 1,
        after_lines: 10,
    };
    let (result, output) = {
        let mut output = String::new();
        let options = ExecOptions::default()
            .with_spill(spill)
            .on_output(|chunk| output.push_str(&chunk.text));
        let result = env.exec(Command::shell(r"printf 'f\200\000o'"), options, &ctx()).await;
        (result, output)
    };
    let spilled = Spilled(result.unwrap().spill_path.expect("a spill"));
    assert_eq!(output, "f\u{FFFD}\0o");
    assert_eq!(env.read_binary_file(&spilled.0, &ctx()).await.unwrap(), b"f\x80\x00o");
}

#[tokio::test]
async fn reports_the_spill_of_a_command_that_times_out() {
    let dir = TempDir::new();
    let env = dir.env();
    let spill = SpillOptions {
        after_bytes: 10,
        after_lines: 10,
    };
    let options = ExecOptions::default()
        .with_spill(spill)
        .with_timeout(Duration::from_millis(300));
    let error = env
        .exec(Command::shell("printf 12345678901234567890; sleep 5"), options, &ctx())
        .await
        .unwrap_err();
    assert_eq!(error.code, ExecErrorCode::Timeout);
    let spilled = Spilled(error.spill_path.expect("a spill"));
    assert_eq!(
        env.read_text_file(&spilled.0, &ctx()).await.unwrap(),
        "12345678901234567890"
    );
}

#[tokio::test]
async fn streams_every_line_and_spills_them_all_past_the_line_threshold() {
    let dir = TempDir::new();
    let env = dir.env();
    let spill = SpillOptions {
        after_bytes: 1024 * 1024,
        after_lines: 100,
    };
    let script = "i=1; while [ $i -le 15000 ]; do echo line-$i; i=$((i+1)); done";
    let (result, output) = run(&env, script, ExecOptions::default().with_spill(spill)).await;
    assert_eq!(result, Ok(0));
    let expected: Vec<String> = (1..=15_000).map(|index| format!("line-{index}\n")).collect();
    assert_eq!(output, expected.concat());

    let options = ExecOptions::default().with_spill(spill);
    let result = env.exec(Command::shell(script), options, &ctx()).await.unwrap();
    let spilled = Spilled(result.spill_path.expect("a spill"));
    let lines = env.read_text_lines(&spilled.0, None, &ctx()).await.unwrap();
    assert_eq!(
        (lines.len(), lines.last().map(String::as_str)),
        (15_000, Some("line-15000"))
    );
}

#[tokio::test]
async fn preserves_complete_large_output_in_the_spill() {
    let dir = TempDir::new();
    let env = dir.env();
    let spill = SpillOptions {
        after_bytes: 10,
        after_lines: 10,
    };
    let script = "head -c 500000 /dev/zero | tr '\\0' x";
    let result = env
        .exec(Command::shell(script), ExecOptions::default().with_spill(spill), &ctx())
        .await
        .unwrap();
    let spilled = Spilled(result.spill_path.expect("a spill"));
    assert_eq!(env.read_text_file(&spilled.0, &ctx()).await.unwrap().len(), 500_000);
}
