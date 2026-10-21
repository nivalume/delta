# Execution environments (`env`)

Follows the `ExecutionEnv` parts of spec §7.3 (`api.env`, output windows and skips,
`exec` forms, binary and directory readers, `listFiles`, `watch`, trust) and §2.2
(`HarnessOptions.env`, built per call). Corresponds to upstream `src/env/index.ts` (the
interface), `src/env/node.ts` and `src/env/node-watch.ts` (the local implementation) and
`src/testing/env-conformance.ts`.

## Problem

Tools reach files and processes only through an environment the harness builds for each
call from the conversation's `cwd`, so a conversation's directory, container or remote
host needs no tool wrapper, and a rerun after recovery gets the environment of that time.
The environment streams raw command output unbounded and spills the complete output to a
file; bounding, sanitizing and throttling happen once, in `ToolApi::output`
([tool.md](tool.md)). Files are read positionally so a `read` of a huge file transfers
only what the output limits allow.

## Errors and results

```rust
#[derive(Serialize, Deserialize)] #[serde(rename_all = "snake_case")]
pub enum FileErrorCode { Aborted, NotFound, PermissionDenied, NotDirectory, IsDirectory, Invalid, NotSupported, Unknown }
#[derive(Debug, thiserror::Error)]
pub struct FileError { pub code: FileErrorCode, pub message: String, pub path: Option<String>, source: Option<io::Error> }

#[serde(rename_all = "snake_case")]
pub enum ExecErrorCode { Aborted, Timeout, ShellUnavailable, SpawnError, CallbackError, Unknown }
pub struct ExecError { pub code: ExecErrorCode, pub message: String, pub spill_path: Option<String>, source: Option<io::Error> }

pub type FileResult<T> = Result<T, FileError>;
```

Expected failures are returned, never panicked. `io::Error` maps by kind: `NotFound`,
`PermissionDenied`, `NotADirectory`, `IsADirectory`, `InvalidInput` and `InvalidFilename` →
`Invalid`, else `Unknown`; a cancelled context is `Aborted` on every operation. Reads race the
context; mutations check it before each step but never race it, so a write the operating
system already started is not reported as aborted while it completes.

## The traits

```rust
pub trait FileSystem: Send + Sync {
    /// The file namespace: equal ids see the same files at the same paths. Every local environment shares one.
    fn id(&self) -> &str;
    fn cwd(&self) -> &str;
    fn absolute_path(&self, path: &str) -> FileResult<String>;
    fn join_path(&self, parts: &[&str]) -> FileResult<String>;
    fn read_text_file<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<String>>;
    fn read_binary_file(..) -> BoxFuture<FileResult<Vec<u8>>>;
    fn read_text_lines(&self, path, max_lines: Option<usize>, ctx) -> BoxFuture<FileResult<Vec<String>>> { /* over open_text_line_reader */ }
    fn open_text_line_reader(..) -> BoxFuture<FileResult<Box<dyn TextLineReader>>>;
    fn open_binary_reader(&self, path, options: OpenOptions /* no_follow */, ctx) -> BoxFuture<FileResult<Box<dyn BinaryReader>>>;
    fn write_file(&self, path, content: &[u8], ctx) -> BoxFuture<FileResult<()>>;   // creates parents
    fn append_file(..); fn truncate_file(&self, path, size: u64, ctx); fn flush_file(..); fn rename_file(from, to, ctx);
    fn file_info(..) -> BoxFuture<FileResult<FileInfo>>;
    fn list_dir(..) -> BoxFuture<FileResult<Vec<FileInfo>>>;
    fn open_dir_reader(..) -> BoxFuture<FileResult<Box<dyn DirReader>>>;
    fn list_files(&self, path, options: ListFilesOptions, ctx) -> BoxFuture<FileResult<FileListing>> { not_supported() }
    fn watch(&self, targets: Vec<WatchTarget>, ctx) -> BoxFuture<FileResult<Box<dyn FileWatcher>>> { not_supported() }
    fn canonical_path(..) -> BoxFuture<FileResult<String>>;
    fn exists(..) -> BoxFuture<FileResult<bool>> { /* over file_info */ }
    fn create_dir(&self, path, recursive: bool, ctx); fn remove(&self, path, options: RemoveOptions /* recursive, force */, ctx);
    fn create_temp_dir(&self, prefix: Option<&str>, ctx) -> BoxFuture<FileResult<String>>;
    fn create_temp_file(&self, options: TempFileOptions /* prefix, suffix */, ctx) -> BoxFuture<FileResult<String>>;
}

pub trait Shell: Send + Sync {
    fn exec<'a>(&'a self, command: Command, options: ExecOptions<'a>, ctx: &'a Context) -> BoxFuture<'a, Result<ExecResult, ExecError>>;
    /// Kill every command still running; for the owner's shutdown, never for one request.
    fn cleanup<'a>(&'a self, ctx: &'a Context) -> BoxFuture<'a, ()>;
}

pub trait ExecutionEnv: FileSystem + Shell {}
impl<T: FileSystem + Shell + ?Sized> ExecutionEnv for T {}
```

`cleanup` lives on `Shell` only: with one on each trait, `env.cleanup(ctx)` would be
ambiguous on `dyn ExecutionEnv`, and upstream's file-system half has nothing to clean up (its
readers and watchers are closed by their owners). `read_text_lines` and `exists` are provided
methods, so another environment gets them from `open_text_line_reader` and `file_info`.

Tools hold `Arc<dyn ExecutionEnv>`. Methods return `BoxFuture` so the traits are object
safe (as `delta_agent::AgentTool`); `id`, `cwd`, `absolute_path` and `join_path` are
synchronous because they are pure path arithmetic. `cwd` is fixed per environment
instance: the harness builds one per call from the conversation's `cwd`, so upstream's
mutable `cwd` field is not needed.

```rust
pub struct FileInfo { pub name: String, pub path: String, pub kind: FileKind /* File | Directory | Symlink */, pub size: u64, pub mtime_ms: i64 }

pub trait BinaryReader: Send + Sync {
    fn info(&self, ctx) -> BoxFuture<FileResult<FileInfo>>;                   // of the opened file, whatever its path names now
    fn read(&self, offset: u64, length: usize, ctx) -> BoxFuture<FileResult<Vec<u8>>>;   // fewer bytes only at end of file
    fn scan_lines(&self, start_line: u64, end_line: Option<u64>, ctx) -> BoxFuture<FileResult<LineScan>>;
    fn close(self: Box<Self>, ctx: &Context) -> BoxFuture<'_, ()>;
}
pub struct LineScan { pub newlines: u64, pub start: u64, pub end: u64, pub first_line_end: u64, pub last_line_start: u64, pub selected_bytes: u64, pub first_line_bytes: u64 }

pub trait TextLineReader: Send { fn read_line(&mut self, ctx) -> BoxFuture<FileResult<Option<TextLine /* text, terminated */>>>; fn close(..); }
pub trait DirReader: Send { fn next(&mut self, max_entries: usize, ctx) -> BoxFuture<FileResult<DirPage /* entries, done */>>; fn close(..); }
```

Every reader's `close` consumes it, so use after close is a compile-time error rather than an
`Invalid` result; dropping a local reader releases its file too. `BinaryReader` reads through
`&self`, so it is `Sync` as well.

- `open_binary_reader` opens one regular file: a directory fails `IsDirectory`, another
  non-regular file `Invalid`; with `no_follow`, a symbolic link as the final component
  fails `Invalid` (earlier components still resolve). Reads see the opened file after a
  rename. A single `read` allocates at most 1 MiB at a time, whatever `length` says; an
  `offset + length` past `u64` fails `Invalid` (negative or fractional values are
  unrepresentable). The local reader checks the kind before opening, so a FIFO is never
  opened (upstream's `O_NONBLOCK`), and implements `no_follow` without `O_NOFOLLOW` (the
  crate has no libc binding): `lstat` refuses a link, and on Unix the opened file must have
  the device and inode that `lstat` saw, so a path swapped for a link in between is refused.
- Text: `read_text_file` decodes lossily and keeps a byte-order mark (as Node's `readFile`);
  text line readers and command output drop a leading one. Line readers split at `\n` only,
  keeping `\r`; an aborted `read_line` consumes nothing.
- `scan_lines` makes one pass inside the environment: line `k` starts after the `k`-th
  newline byte; `[start_line, end_line)` (absent: to the end); a selection past the last
  line is empty at the end of the file; decoded sizes are those of decoding the whole
  file as UTF-8 (a leading BOM is not counted; invalid bytes count as U+FFFD, 3 bytes).
- `DirReader::next` pages in file-system order, reads metadata only for returned entries,
  skips entries removed meanwhile and unsupported kinds; `done` may come with the last
  page or an empty one; after a failed or aborted call the reader is closed.
- `list_files` (optional): files, directories and symlinks (not followed) below a
  directory, relative paths with `/`, depth first, siblings in byte order of names;
  `.git` never listed; `hidden: true` default; inside a git work tree what git ignores is
  skipped (`gitignore: true` default: the directory's, its parents' and subdirectories'
  `.gitignore`, `.git/info/exclude`, the global excludes; a tracked file matching a rule
  is skipped too); `exclude` names skip subtrees; `max_entries` (50 000), `max_bytes`
  (4 MiB, at most 15.9 MiB; each entry counts its path plus two), `timeout_ms` (10 000)
  end it early with `truncated: true`. Negative or fractional bounds fail `Invalid`.

## `exec`

```rust
pub enum Command { Shell(String), Argv(Vec<String>) }

pub struct ExecOptions<'a> {
    pub cwd: Option<String>,                  // resolved against the environment's cwd
    pub env: BTreeMap<String, String>,
    pub inherit_env: bool,                    // default true
    pub timeout: Option<Duration>,
    pub on_output: Option<Box<dyn FnMut(OutputChunk) + Send + 'a>>,   // may borrow the caller's buffers
    pub spill: Option<SpillOptions>,          // after_bytes, after_lines
    pub window: Option<ShellOutputWindow>,    // max_bytes, max_lines, min_interval: Duration, bytes_per_second; limits() -> OutputLimits
}
pub struct OutputChunk { pub text: String, pub stream: OutputStream /* Stdout | Stderr */, pub skipped: Option<OutputSkip> }
pub struct ExecResult { pub exit_code: i32, pub spill_path: Option<String> }
```

- `Shell(s)` runs `<shell> -c s`: the configured path (missing: `ShellUnavailable`), else
  `/bin/bash` (Windows: Git Bash under `ProgramFiles`), else `bash` on `PATH` (searched
  in-process, not through `which`/`where`; a legacy WSL `bash.exe` is skipped), else `sh` on
  Unix and `ShellUnavailable` on Windows, as upstream. `Argv(v)` spawns `v[0]` with the rest
  as arguments, no parsing; an empty vector is `SpawnError`. A missing working directory is `SpawnError` (`Working directory does not
  exist: <cwd>`).
- Output: every decoded chunk of stdout and stderr, in arrival order, raw, unbounded,
  unthrottled, each stream decoded separately (a character split across chunks survives).
  Nothing reaches the callback after `exec` settled, even from a descendant holding the
  pipes: after the exit, pipes that stay quiet for 100 ms are dropped. A panic in the
  callback (caught with `catch_unwind`, so it needs `panic = "unwind"`) kills the command and
  fails `CallbackError` with the panic message.
- `spill`: once the seen output exceeds `after_bytes` or `after_lines` (complete or
  partial lines), the complete raw output (the prefix kept so far and everything after)
  streams to a temp file `delta-output-*.log` created through `create_temp_file`; the
  writer's backpressure pauses the child's pipes. The path is `ExecResult.spill_path`, or
  `ExecError.spill_path` when the command then timed out or was aborted. A spill write
  failure kills the command and fails `Unknown`.
- `window`: the caller keeps only this tail, so the environment may omit output outside
  it and report the omission as `skipped` on the next chunk; it may omit only output
  followed by more than the window (more than `max_bytes` bytes or more than `max_lines`
  newlines), delivers all of that following output in the same chunk, and should deliver
  no faster than the pace. The local environment never omits (it is in-process); the
  conformance case checks the contract for environments that do. A window and separate
  streams are exclusive: a windowed chunk's `stream` names its last stream.
- Timeout and abort kill only this command's process tree and fail `Timeout`
  (`timeout:<secs>`) or `Aborted` after the pipes drain, with `spill_path` when one exists.
  On Unix each command leads its own process group (`process_group(0)`, i.e. `setpgid`;
  upstream's `detached` is `setsid`, which also drops the controlling terminal, not needed
  with stdin closed). The crate forbids `unsafe` and has no libc binding, so the group is
  killed by `/bin/sh -c 'kill -9 -<pgid>'` rather than `killpg(2)`, and the direct child
  through tokio as well; a descendant that leaves the group (`setsid`) is out of reach, as
  upstream. Windows uses `taskkill /F /T`. Dropping the `exec` future kills the tree too.
- A zero timeout or one above 2 147 483 647 ms fails before spawning with `Timeout` and an
  `Invalid timeout: …` message, as upstream (`ExecErrorCode` has no `Invalid`); the bound
  also keeps `Instant + timeout` from overflowing.
- Exit code: the process's; killed by a signal, `128 + signal`; unknown, `1`.
- `cleanup` kills every child the environment still tracks.

## Watch and temp files

```rust
pub struct WatchTarget { pub path: String, pub recursive: bool, pub exclude: Exclude /* hidden, names */ }
pub enum WatchChange { Paths(Vec<String>), Overflow, Error(FileError) }
pub trait FileWatcher: Send { fn mode(&self) -> WatchMode /* Native | Polling */; fn next(&mut self) -> BoxFuture<Option<WatchChange>>; fn close(self: Box<Self>, ctx: &Context) -> BoxFuture<'_, ()>; }
```

- A target may be missing; creating it is a change. A recursive target covers its subtree
  except excluded entries, without following symbolic links below it; a watched symlink
  itself reports changes to what it points to. When `watch` returns, coverage is
  established. Paths reported cover their subtree; reports may be spurious; `Overflow`
  means rescan everything; `Error` is final.
- The local watcher uses platform events only to trigger rescans and reports differences
  between snapshots (identity, size, mtime, a content hash for small files), so replaced
  files, renamed parents and directories created with their contents are reported
  whatever events arrive. `Polling` (default on Windows and on file systems that do not
  report remote changes) compares snapshots every 2 s and can miss a change undone in
  between; at most 10 000 directories per watcher, beyond which `Error`.
- `create_temp_dir(prefix)` (default prefix `delta-`) / `create_temp_file { prefix, suffix }`
  create `<prefix><random><suffix>` directly under the system temp directory with
  `0700`/`0600` permissions (`create_new`, so a name is never reused; upstream makes a
  directory per file). `remove` with `recursive` and `force` (missing is fine) is `rm -rf`
  semantics; a directory without `recursive` fails `IsDirectory`.
- An environment is trusted, not a confinement boundary: path checks are hygiene, not
  security.

## `LocalEnv`

```rust
pub struct LocalEnv { /* cwd, shell_path, shell_env, running children */ }
impl LocalEnv {
    pub fn new(cwd: impl Into<PathBuf>) -> Self;          // a relative cwd resolves against the process's
    pub fn with_shell(self, path: impl Into<PathBuf>) -> Self;
    pub fn with_shell_env(self, env: BTreeMap<String, String>) -> Self;
}
```

`id()` is `"local"`. The tool task builds `api.env` through `HarnessOptions.env`, an
`Arc<dyn EnvBuilder>` ([harness.md](harness.md)) whose `build(EnvTarget { conversation_id,
cwd }, &Context)` returns `Option<Arc<dyn ExecutionEnv>>`, never on the session line;
`LocalEnvBuilder::new(default_cwd)` (one cached `LocalEnv` per directory; the conversation's
`cwd`, else the default) is the usual value. The builder and its cache live with
`EnvBuilder` in `agent::env`, not in `env`: `EnvTarget` names a conversation, and a free `local_env(cwd)` with a process-wide cache would be the
module-level state the architecture rules out. `with_watch` arrives with `watch`.

Paths: `~` and `~/…` name the home directory, a `file://` URL its file path (a malformed one
stays an ordinary path), and the result is normalized lexically as Node's `path.resolve`.
Text decoding is a streaming UTF-8 decoder (`utf8_chunks`, crate-private `env::decode`), as
`delta_chord::OutputBuffer`; `LineScanner` computes `LineScan`s from it in one pass.

Layout: `env/mod.rs` (traits and shared types), `error.rs`, `exec.rs`, `decode.rs`,
`line_scan.rs`, `watch.rs` and `list.rs` (types of the optional capabilities), and
`env/local/` (`mod.rs`, `path.rs`, `fs.rs`, `readers.rs`, `shell.rs`, `process.rs`,
`spill.rs`). `env` names only `delta_chord`, `tokio`, `futures`, `url` and `uuid`.

## Conformance suite

`tests/env_conformance.rs`: each case is a `for<'a> fn(&'a Suite, &'a dyn ExecutionEnv) ->
BoxFuture<'a, ()>` run in a fresh, empty, writable `cwd`; `Suite` holds `shell` (argv prefix,
`["sh", "-c"]`), `symlinks` (default true on Unix), `watch` and `list_files` (whether the
optional methods are implemented; when not, a case checks that they fail `not_supported`).
The functions move to a public `delta_harness::testing` module when a second environment
needs them. Cases ported from upstream:

- binary reader: reads byte ranges of the opened file; scans lines like decoding the whole
  file; keeps reading the file it opened after a rename; refuses directories, missing
  files and aborted opens; follows symlinks unless `no_follow` refuses the final one.
- directory reader: pages every entry exactly once; reports the end and is closed after an
  aborted call (use after `close` does not compile); refuses missing paths and files; skips
  entries removed during enumeration.
- watch (30 s timeout): reports a missing file's creation, changes, replacement and
  removal; a missing target whose ancestors are created; directories created together
  with their contents; a path whose parent is renamed and recreated; skips excluded
  entries and reports a rename out of them; keeps recursive coverage where a
  non-recursive target overlaps; follows a directory replaced at the same path; stops
  reporting once closed; reports changes to the file a watched symbolic link points to.
- exec: argv passes arguments without shell parsing; reports the stream of every chunk in
  both forms; argv honors `cwd` and exit codes; missing programs and empty argv are spawn
  errors; windowed exec keeps the exact tail and counts what it skips; distinguishes
  timeout from abort.
- `list_files`: lists depth first in name order; skips what git ignores in work trees,
  `.git` and excluded names; stops at its bounds and reports `truncated`; fails on missing
  directories, files and aborts; lists symbolic links without following them.

The suite runs against `LocalEnv` in `tests/`, in temp directories, without the network; the
watch and `list_files` cases are ported with those methods. `tests/env_local.rs` covers the
local behaviour beyond it (spills, process trees, environment variables, file operations).

## Phase 1 and later

- Phase 1 ships the traits and `LocalEnv`. `watch` and `list_files` are optional methods
  whose default returns `not_supported`; `LocalEnv` keeps that default until `notify`
  (for `watch`) and `ignore` (for `list_files`) are dependencies, and their conformance
  cases are skipped meanwhile.
- A remote environment on pi-env's daemon protocol is later; the `window` and `skipped`
  contract exists for it, and the conformance suite is what it must pass.

## Differences from upstream

- Two traits, `FileSystem` and `Shell`, with `ExecutionEnv` as their union; errors are
  `Result`s with `thiserror` types instead of `{ ok, value }` objects.
- Path functions are synchronous; `cwd` is immutable per instance.
- `on_output` is `FnMut(OutputChunk)` that may borrow (`ExecOptions<'a>`); a panic maps to
  `CallbackError`. The stream enum is `OutputStream` (`Stream` is `futures`' trait).
- `cleanup` only on `Shell`; a reader's `close` consumes it.
- `FileWatcher` is pull-based (`next().await`) instead of a callback.
- `id` is `"local"` (upstream `node:local`); it only keys the in-process mutation queue.
- Reads are `Vec<u8>` / `String`; no `Uint8Array` or string duality on writes
  (`&[u8]`, and `&str` through `as_bytes`).
- No PowerShell support in the shell discovery beyond `shell_path`.

## Decisions

- `watch` and `list_files` are optional methods returning `not_supported` until `notify` /
  `ignore` are dependencies.
- `BoxFuture` methods (as `delta_agent`), no `trait_variant` dependency.
- Process trees: a process group per command (`process_group(0)`), killed through
  `/bin/sh`'s `kill`, since `unsafe` is forbidden and there is no `nix`/`rustix` dependency.

## Open (minor)

- Whether a tool may opt out of the per-command process group (a command that must receive
  the terminal's signals).
- `rustix` for `killpg(2)` instead of spawning `kill`, if the maintainer accepts the
  dependency.
