//! The conformance suite of `ExecutionEnv`, ported from upstream's `env-conformance.ts`.
//!
//! Every case is a generic function over `&dyn ExecutionEnv` whose `cwd` is a fresh, empty,
//! writable directory; [`Suite`] says what the environment under test offers. The cases run
//! here against `LocalEnv` in unique temp directories that are removed afterwards; a remote
//! environment runs the same functions against its own.
//!
//! The `watch` and `list_files` cases of upstream are ported when `LocalEnv` implements those
//! optional methods; until then [`optional_methods_report_not_supported`] checks the default.

#![allow(clippy::unwrap_used)] // Cases are functions outside `#[test]`, as for any environment.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use delta_chord::context::Context;
use delta_harness::env::{
    Command, DirPage, ExecError, ExecErrorCode, ExecOptions, ExecResult, ExecutionEnv, FileErrorCode, FileInfo,
    FileKind, FileResult, ListFilesOptions, LocalEnv, OpenOptions, OutputStream, RemoveOptions, ShellOutputWindow,
    WatchTarget,
};
use futures::future::BoxFuture;

/// What the environment under test offers.
#[derive(Debug, Clone)]
struct Suite {
    /// Argv prefix that runs a POSIX shell script: `[..shell, script, args..]`.
    shell: Vec<String>,
    /// The environment can create symbolic links.
    symlinks: bool,
    /// [`FileSystem::watch`](delta_harness::env::FileSystem::watch) is implemented.
    watch: bool,
    /// [`FileSystem::list_files`](delta_harness::env::FileSystem::list_files) is implemented.
    list_files: bool,
}

impl Suite {
    fn local() -> Self {
        Self {
            shell: vec!["sh".to_owned(), "-c".to_owned()],
            symlinks: cfg!(unix),
            watch: false,
            list_files: false,
        }
    }

    /// An argv command running `script` with `args` as `$1…`.
    fn script(&self, script: &str, args: &[&str]) -> Command {
        let mut argv = self.shell.clone();
        argv.push(script.to_owned());
        argv.push("argv0".to_owned());
        argv.extend(args.iter().map(|&arg| arg.to_owned()));
        Command::Argv(argv)
    }
}

/// One conformance case.
type Case = for<'a> fn(&'a Suite, &'a dyn ExecutionEnv) -> BoxFuture<'a, ()>;

fn ctx() -> Context {
    Context::background()
}

fn aborted() -> Context {
    let (ctx, canceller) = Context::background().with_cancel();
    canceller.cancel();
    ctx
}

fn file_code<T>(result: &FileResult<T>) -> Option<FileErrorCode> {
    result.as_ref().err().map(|error| error.code)
}

fn exec_code(result: &Result<ExecResult, ExecError>) -> Option<ExecErrorCode> {
    result.as_ref().err().map(|error| error.code)
}

/// Run `command` and collect each stream.
async fn exec_collect(
    env: &dyn ExecutionEnv,
    command: Command,
    cwd: Option<&str>,
) -> (Result<ExecResult, ExecError>, String, String) {
    let mut stdout = String::new();
    let mut stderr = String::new();
    let mut options = ExecOptions::default().on_output(|chunk| match chunk.stream {
        OutputStream::Stdout => stdout.push_str(&chunk.text),
        OutputStream::Stderr => stderr.push_str(&chunk.text),
    });
    options.cwd = cwd.map(str::to_owned);
    let result = env.exec(command, options, &ctx()).await;
    (result, stdout, stderr)
}

/// Every page of a directory, `max_entries` at a time.
async fn read_all(env: &dyn ExecutionEnv, path: &str, max_entries: usize) -> (Vec<Vec<FileInfo>>, bool) {
    let ctx = ctx();
    let mut reader = env.open_dir_reader(path, &ctx).await.expect("open dir reader");
    let mut pages = Vec::new();
    let mut done = false;
    for _ in 0..1000 {
        let page = reader.next(max_entries, &ctx).await.expect("next page");
        pages.push(page.entries);
        if page.done {
            done = true;
            break;
        }
    }
    reader.close(&ctx).await;
    (pages, done)
}

mod cases {
    use pretty_assertions::assert_eq;

    use super::*;

    pub(super) fn binary_reader_reads_byte_ranges<'a>(_: &'a Suite, env: &'a dyn ExecutionEnv) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let ctx = ctx();
            env.write_file("data.txt", b"hello world", &ctx).await.unwrap();
            let reader = env
                .open_binary_reader("data.txt", OpenOptions::default(), &ctx)
                .await
                .unwrap();
            let info = reader.info(&ctx).await.unwrap();
            assert_eq!(
                (info.name.as_str(), info.kind, info.size),
                ("data.txt", FileKind::File, 11)
            );
            assert_eq!(reader.read(0, 5, &ctx).await.unwrap(), b"hello");
            assert_eq!(reader.read(6, 100, &ctx).await.unwrap(), b"world");
            assert!(reader.read(11, 4, &ctx).await.unwrap().is_empty());
            assert!(reader.read(50, 1, &ctx).await.unwrap().is_empty());
            assert!(reader.read(3, 0, &ctx).await.unwrap().is_empty());
            assert_eq!(
                file_code(&reader.read(u64::MAX, 2, &ctx).await),
                Some(FileErrorCode::Invalid)
            );
            assert_eq!(
                file_code(&reader.read(0, 1, &aborted()).await),
                Some(FileErrorCode::Aborted)
            );
            assert_eq!(file_code(&reader.info(&aborted()).await), Some(FileErrorCode::Aborted));
            // Use after close is a compile-time error: `close` consumes the reader.
            reader.close(&ctx).await;
        })
    }

    pub(super) fn binary_reader_scans_lines_like_decoding_the_whole_file<'a>(
        _: &'a Suite,
        env: &'a dyn ExecutionEnv,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let ctx = ctx();
            // A byte-order mark, an invalid sequence before a newline, an empty line, a later
            // U+FEFF, and no final newline.
            let bytes: &[u8] = &[
                0xef, 0xbb, 0xbf, 0x61, 0x0a, 0xe2, 0x82, 0x0a, 0x0a, 0xef, 0xbb, 0xbf, 0x62, 0x0a, 0xc3, 0xa9,
            ];
            env.write_file("lines.txt", bytes, &ctx).await.unwrap();
            let text = decode(bytes, true);
            let lines: Vec<&str> = text.split('\n').collect();
            // Decoding a range drops the mark only at the start of the file.
            let range = |from: u64, to: u64| decode(&bytes[from as usize..to as usize], from == 0);
            let reader = env
                .open_binary_reader("lines.txt", OpenOptions::default(), &ctx)
                .await
                .unwrap();
            for (start_line, end_line) in [
                (0, None),
                (0, Some(1)),
                (1, Some(3)),
                (2, Some(3)),
                (3, None),
                (4, Some(9)),
            ] {
                let scan = reader.scan_lines(start_line, end_line, &ctx).await.unwrap();
                let start = start_line as usize;
                let end = end_line.map_or(lines.len(), |end| (end as usize).min(lines.len()));
                let selected = lines[start..end].join("\n");
                assert_eq!(scan.newlines as usize, lines.len() - 1);
                assert_eq!(range(scan.start, scan.end), selected);
                assert_eq!(scan.selected_bytes as usize, selected.len());
                assert_eq!(range(scan.start, scan.first_line_end), lines[start]);
                assert_eq!(scan.first_line_bytes as usize, lines[start].len());
            }
            let past = reader.scan_lines(9, None, &ctx).await.unwrap();
            let size = bytes.len() as u64;
            assert_eq!((past.start, past.end, past.selected_bytes), (size, size, 0));
            assert_eq!(
                file_code(&reader.scan_lines(2, Some(2), &ctx).await),
                Some(FileErrorCode::Invalid)
            );
            reader.close(&ctx).await;
        })
    }

    /// The text decoding the whole input yields, with or without a leading mark.
    fn decode(bytes: &[u8], strip_bom: bool) -> String {
        let text = String::from_utf8_lossy(bytes).into_owned();
        match text.strip_prefix('\u{FEFF}') {
            Some(rest) if strip_bom => rest.to_owned(),
            _ => text,
        }
    }

    pub(super) fn binary_reader_keeps_reading_the_opened_file_after_a_rename<'a>(
        _: &'a Suite,
        env: &'a dyn ExecutionEnv,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let ctx = ctx();
            env.write_file("a.txt", b"one", &ctx).await.unwrap();
            let reader = env
                .open_binary_reader("a.txt", OpenOptions::default(), &ctx)
                .await
                .unwrap();
            env.rename_file("a.txt", "b.txt", &ctx).await.unwrap();
            env.write_file("a.txt", b"two", &ctx).await.unwrap();
            assert_eq!(reader.read(0, 10, &ctx).await.unwrap(), b"one");
            reader.close(&ctx).await;
        })
    }

    pub(super) fn binary_reader_refuses_directories_missing_files_and_aborted_opens<'a>(
        _: &'a Suite,
        env: &'a dyn ExecutionEnv,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let ctx = ctx();
            env.create_dir("dir", true, &ctx).await.unwrap();
            env.write_file("file.txt", b"x", &ctx).await.unwrap();
            let open = |path: &'a str, ctx: Context| async move {
                file_code(&env.open_binary_reader(path, OpenOptions::default(), &ctx).await)
            };
            assert_eq!(open("dir", ctx.clone()).await, Some(FileErrorCode::IsDirectory));
            assert_eq!(open("missing.txt", ctx.clone()).await, Some(FileErrorCode::NotFound));
            assert_eq!(open("file.txt", aborted()).await, Some(FileErrorCode::Aborted));
        })
    }

    pub(super) fn binary_reader_follows_symlinks_unless_no_follow_refuses_the_final_one<'a>(
        suite: &'a Suite,
        env: &'a dyn ExecutionEnv,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            if !suite.symlinks {
                return;
            }
            let ctx = ctx();
            env.write_file("target.txt", b"target", &ctx).await.unwrap();
            env.write_file("sub/inner.txt", b"inner", &ctx).await.unwrap();
            let linked = env
                .exec(
                    suite.script("ln -s target.txt link.txt && ln -s sub dirlink", &[]),
                    ExecOptions::default(),
                    &ctx,
                )
                .await;
            assert_eq!(linked.unwrap().exit_code, 0);

            let followed = env
                .open_binary_reader("link.txt", OpenOptions::default(), &ctx)
                .await
                .unwrap();
            assert_eq!(followed.read(0, 10, &ctx).await.unwrap(), b"target");
            followed.close(&ctx).await;

            let no_follow = OpenOptions { no_follow: true };
            let refused = env.open_binary_reader("link.txt", no_follow, &ctx).await;
            assert_eq!(file_code(&refused), Some(FileErrorCode::Invalid));

            // Only the final component is refused; earlier symlinked directories still resolve.
            let inner = env
                .open_binary_reader("dirlink/inner.txt", no_follow, &ctx)
                .await
                .unwrap();
            assert_eq!(inner.read(0, 10, &ctx).await.unwrap(), b"inner");
            inner.close(&ctx).await;
        })
    }

    pub(super) fn dir_reader_pages_every_entry_exactly_once<'a>(
        _: &'a Suite,
        env: &'a dyn ExecutionEnv,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let ctx = ctx();
            let names = ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"];
            for name in names {
                env.write_file(name, name.as_bytes(), &ctx).await.unwrap();
            }
            env.create_dir("sub", true, &ctx).await.unwrap();
            let (pages, done) = read_all(env, ".", 2).await;
            assert!(done, "the reader reached the end");
            assert!(pages.iter().all(|page| page.len() <= 2), "pages within max_entries");
            let entries: Vec<FileInfo> = pages.into_iter().flatten().collect();
            let mut listed: Vec<&str> = entries.iter().map(|entry| entry.name.as_str()).collect();
            listed.sort_unstable();
            assert_eq!(listed, ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt", "sub"]);
            let kind = |name: &str| {
                entries
                    .iter()
                    .find(|entry| entry.name == name)
                    .map(|entry| (entry.kind, entry.size))
            };
            assert_eq!(kind("sub").map(|(kind, _)| kind), Some(FileKind::Directory));
            assert_eq!(kind("a.txt"), Some((FileKind::File, 5)));
        })
    }

    pub(super) fn dir_reader_reports_the_end_and_closes_after_an_aborted_call<'a>(
        _: &'a Suite,
        env: &'a dyn ExecutionEnv,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let ctx = ctx();
            env.create_dir("empty", true, &ctx).await.unwrap();
            let mut reader = env.open_dir_reader("empty", &ctx).await.unwrap();
            let end = DirPage {
                entries: Vec::new(),
                done: true,
            };
            assert_eq!(reader.next(10, &ctx).await.unwrap(), end);
            assert_eq!(reader.next(10, &ctx).await.unwrap(), end);
            assert_eq!(file_code(&reader.next(0, &ctx).await), Some(FileErrorCode::Invalid));
            assert_eq!(
                file_code(&reader.next(1, &aborted()).await),
                Some(FileErrorCode::Aborted)
            );
            // An aborted call closes the reader.
            assert_eq!(file_code(&reader.next(1, &ctx).await), Some(FileErrorCode::Invalid));
            reader.close(&ctx).await;
        })
    }

    pub(super) fn dir_reader_refuses_missing_paths_and_files<'a>(
        _: &'a Suite,
        env: &'a dyn ExecutionEnv,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let ctx = ctx();
            env.write_file("file.txt", b"x", &ctx).await.unwrap();
            assert_eq!(
                file_code(&env.open_dir_reader("missing", &ctx).await),
                Some(FileErrorCode::NotFound)
            );
            assert_eq!(
                file_code(&env.open_dir_reader("file.txt", &ctx).await),
                Some(FileErrorCode::NotDirectory)
            );
            assert_eq!(
                file_code(&env.open_dir_reader(".", &aborted()).await),
                Some(FileErrorCode::Aborted)
            );
        })
    }

    pub(super) fn dir_reader_skips_entries_removed_during_enumeration<'a>(
        _: &'a Suite,
        env: &'a dyn ExecutionEnv,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let ctx = ctx();
            for name in ["x", "y", "z"] {
                env.write_file(&format!("dir/{name}"), name.as_bytes(), &ctx)
                    .await
                    .unwrap();
            }
            let mut reader = env.open_dir_reader("dir", &ctx).await.unwrap();
            for name in ["x", "y", "z"] {
                env.remove(&format!("dir/{name}"), RemoveOptions::default(), &ctx)
                    .await
                    .unwrap();
            }
            let mut entries = Vec::new();
            for _ in 0..10 {
                let page = reader.next(10, &ctx).await.unwrap();
                entries.extend(page.entries);
                if page.done {
                    break;
                }
            }
            assert_eq!(entries, []);
            reader.close(&ctx).await;
        })
    }

    pub(super) fn argv_exec_passes_arguments_without_shell_parsing<'a>(
        suite: &'a Suite,
        env: &'a dyn ExecutionEnv,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let hostile = "it's $(touch pwned) `touch pwned` *; touch pwned";
            let command = suite.script(r#"printf "%s|%s" "$1" "$2""#, &[hostile, "a b"]);
            let (result, stdout, _) = exec_collect(env, command, None).await;
            assert_eq!(result.unwrap().exit_code, 0);
            assert_eq!(stdout, format!("{hostile}|a b"));
            assert!(!env.exists("pwned", &ctx()).await.unwrap());
        })
    }

    pub(super) fn exec_reports_the_stream_of_every_chunk_in_both_forms<'a>(
        suite: &'a Suite,
        env: &'a dyn ExecutionEnv,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let script = "printf out; printf err >&2; printf more";
            for command in [suite.script(script, &[]), Command::shell(script)] {
                let (result, stdout, stderr) = exec_collect(env, command, None).await;
                assert_eq!(result.unwrap().exit_code, 0);
                assert_eq!((stdout.as_str(), stderr.as_str()), ("outmore", "err"));
            }
        })
    }

    pub(super) fn argv_exec_honors_cwd_and_exit_codes<'a>(
        suite: &'a Suite,
        env: &'a dyn ExecutionEnv,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let ctx = ctx();
            env.create_dir("sub", true, &ctx).await.unwrap();
            let command = suite.script("printf x > made.txt; exit 3", &[]);
            let (result, _, _) = exec_collect(env, command, Some("sub")).await;
            assert_eq!(result.unwrap().exit_code, 3);
            assert_eq!(env.read_text_file("sub/made.txt", &ctx).await.unwrap(), "x");
        })
    }

    pub(super) fn argv_exec_reports_missing_programs_and_empty_argv_as_spawn_errors<'a>(
        _: &'a Suite,
        env: &'a dyn ExecutionEnv,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let ctx = ctx();
            let missing = Command::argv(["delta-harness-conformance-missing-program"]);
            let result = env.exec(missing, ExecOptions::default(), &ctx).await;
            assert_eq!(exec_code(&result), Some(ExecErrorCode::SpawnError));
            let empty = env.exec(Command::Argv(Vec::new()), ExecOptions::default(), &ctx).await;
            assert_eq!(exec_code(&empty), Some(ExecErrorCode::SpawnError));
        })
    }

    pub(super) fn windowed_exec_keeps_the_exact_tail_and_counts_what_it_skips<'a>(
        suite: &'a Suite,
        env: &'a dyn ExecutionEnv,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            const LINES: usize = 2000;
            let window = ShellOutputWindow {
                max_bytes: 200,
                max_lines: 5,
                min_interval: Duration::ZERO,
                bytes_per_second: 1_000_000_000,
            };
            let (mut bytes, mut newlines, mut tail) = (0, 0, String::new());
            let script = format!("i=0; while [ $i -lt {LINES} ]; do echo line-$i; i=$((i+1)); done");
            let options = ExecOptions::default().with_window(window).on_output(|chunk| {
                let chunk_newlines = chunk.text.matches('\n').count();
                if let Some(skipped) = chunk.skipped {
                    bytes += skipped.bytes;
                    newlines += skipped.newlines;
                    assert!(
                        chunk.text.len() > window.max_bytes || chunk_newlines > window.max_lines,
                        "a skip is followed by more than the window"
                    );
                    tail.clear();
                }
                bytes += chunk.text.len();
                newlines += chunk_newlines;
                tail.push_str(&chunk.text);
            });
            let result = env.exec(suite.script(&script, &[]), options, &ctx()).await;
            assert_eq!(result.unwrap().exit_code, 0);
            let expected: Vec<String> = (0..LINES).map(|index| format!("line-{index}\n")).collect();
            assert_eq!(bytes, expected.concat().len());
            assert_eq!(newlines, LINES);
            assert!(
                tail.ends_with(&expected[LINES - window.max_lines..].concat()),
                "the delivered output ends with the tail"
            );
        })
    }

    pub(super) fn argv_exec_distinguishes_timeout_from_abort<'a>(
        suite: &'a Suite,
        env: &'a dyn ExecutionEnv,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let options = ExecOptions::default().with_timeout(Duration::from_millis(100));
            let timed_out = env.exec(suite.script("sleep 2", &[]), options, &ctx()).await;
            assert_eq!(exec_code(&timed_out), Some(ExecErrorCode::Timeout));

            let (ctx, canceller) = Context::background().with_cancel();
            let abort = async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                canceller.cancel();
            };
            let (aborted, ()) = tokio::join!(
                env.exec(suite.script("sleep 2", &[]), ExecOptions::default(), &ctx),
                abort
            );
            assert_eq!(exec_code(&aborted), Some(ExecErrorCode::Aborted));
        })
    }

    pub(super) fn optional_methods_report_not_supported<'a>(
        suite: &'a Suite,
        env: &'a dyn ExecutionEnv,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let ctx = ctx();
            if !suite.list_files {
                let listed = env.list_files(".", ListFilesOptions::default(), &ctx).await;
                assert_eq!(file_code(&listed), Some(FileErrorCode::NotSupported));
            }
            if !suite.watch {
                let target = WatchTarget {
                    path: "AGENTS.md".to_owned(),
                    ..WatchTarget::default()
                };
                let watched = env.watch(vec![target], &ctx).await;
                assert_eq!(watched.err().map(|error| error.code), Some(FileErrorCode::NotSupported));
            }
        })
    }
}

/// A unique directory under the system temp directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("delta-env-conformance-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir(&path).expect("create a temp dir");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Run `case` against a `LocalEnv` in a fresh directory.
async fn run_local(case: Case) {
    let dir = TempDir::new();
    let env: Arc<dyn ExecutionEnv> = Arc::new(LocalEnv::new(dir.path()));
    case(&Suite::local(), &*env).await;
    env.cleanup(&ctx()).await;
}

macro_rules! local_conformance {
    ($($name:ident),* $(,)?) => {
        mod local {
            $(
                #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
                async fn $name() {
                    super::run_local(super::cases::$name).await;
                }
            )*
        }
    };
}

local_conformance!(
    binary_reader_reads_byte_ranges,
    binary_reader_scans_lines_like_decoding_the_whole_file,
    binary_reader_keeps_reading_the_opened_file_after_a_rename,
    binary_reader_refuses_directories_missing_files_and_aborted_opens,
    binary_reader_follows_symlinks_unless_no_follow_refuses_the_final_one,
    dir_reader_pages_every_entry_exactly_once,
    dir_reader_reports_the_end_and_closes_after_an_aborted_call,
    dir_reader_refuses_missing_paths_and_files,
    dir_reader_skips_entries_removed_during_enumeration,
    argv_exec_passes_arguments_without_shell_parsing,
    exec_reports_the_stream_of_every_chunk_in_both_forms,
    argv_exec_honors_cwd_and_exit_codes,
    argv_exec_reports_missing_programs_and_empty_argv_as_spawn_errors,
    windowed_exec_keeps_the_exact_tail_and_counts_what_it_skips,
    argv_exec_distinguishes_timeout_from_abort,
    optional_methods_report_not_supported,
);
