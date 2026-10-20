//! Execution environments: the files and processes tools reach.
//!
//! An [`ExecutionEnv`] is a [`FileSystem`] and a [`Shell`]. The harness builds one per tool
//! call from the conversation's `cwd`, so a conversation's directory, container or remote host
//! needs no tool wrapper. [`LocalEnv`] is the implementation on the local machine; a remote
//! environment implements the same traits and passes the same conformance suite
//! (`tests/env_conformance.rs`).
//!
//! The traits are object safe (tools hold `Arc<dyn ExecutionEnv>`): asynchronous methods return
//! a [`BoxFuture`], and expected failures are returned as [`FileError`] / [`ExecError`], never
//! panicked. Every asynchronous operation takes a [`Context`] and fails `aborted` once it is
//! cancelled. An environment is trusted, not a confinement boundary: its path checks are
//! hygiene, not security.
//!
//! This module is a side module: it depends on `delta_chord` and the standard library only,
//! never on the harness's kernel or agent modules. See `docs/design/env.md`.

mod decode;
mod error;
mod exec;
mod line_scan;
mod list;
pub mod local;
mod watch;

use futures::future::BoxFuture;

use delta_chord::context::Context;

pub use error::{ExecError, ExecErrorCode, FileError, FileErrorCode, FileResult};
pub use exec::{
    Command, ExecOptions, ExecResult, MAX_TIMEOUT, OutputCallback, OutputChunk, OutputStream, ShellOutputWindow,
    SpillOptions,
};
pub use line_scan::LineScan;
pub use list::{
    DEFAULT_LIST_MAX_BYTES, DEFAULT_LIST_MAX_ENTRIES, DEFAULT_LIST_TIMEOUT, FileListEntry, FileListing,
    ListFilesOptions, MAX_LIST_BYTES,
};
pub use local::LocalEnv;
pub use watch::{Exclude, FileWatcher, WatchChange, WatchMode, WatchTarget};

/// What a path names. Other kinds (sockets, devices, FIFOs) are not reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileKind {
    File,
    Directory,
    /// A symbolic link, not followed.
    Symlink,
}

/// Metadata of one file, directory or symbolic link.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileInfo {
    /// The last path component.
    pub name: String,
    /// The absolute path.
    pub path: String,
    pub kind: FileKind,
    /// Size in bytes.
    pub size: u64,
    /// Modification time in milliseconds since the Unix epoch.
    pub mtime_ms: i64,
}

/// One line of a [`TextLineReader`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextLine {
    /// The line without its newline; a carriage return before it is kept.
    pub text: String,
    /// Whether a newline ended the line; only the last line of a file can lack one.
    pub terminated: bool,
}

/// One page of a [`DirReader`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DirPage {
    pub entries: Vec<FileInfo>,
    /// The directory has no more entries. It may come with the last entries or with an empty
    /// page.
    pub done: bool,
}

/// How [`FileSystem::open_binary_reader`] opens a file.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpenOptions {
    /// Refuse a symbolic link as the final path component (`invalid`); earlier components
    /// still resolve.
    pub no_follow: bool,
}

/// How [`FileSystem::remove`] removes, as `rm`'s flags.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RemoveOptions {
    /// Remove a directory with everything below it; without it a directory fails
    /// `is_directory`.
    pub recursive: bool,
    /// A missing path is not an error.
    pub force: bool,
}

/// Name parts of a [`FileSystem::create_temp_file`]; a random part goes between them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TempFileOptions {
    pub prefix: Option<String>,
    pub suffix: Option<String>,
}

/// Lines of one text file, decoded as UTF-8 (a leading byte-order mark dropped, invalid bytes
/// as U+FFFD) and split at `\n` only.
pub trait TextLineReader: Send {
    /// The next line; `None` at the end of the file. An aborted call consumes nothing, so the
    /// next call returns the same line.
    fn read_line<'a>(&'a mut self, ctx: &'a Context) -> BoxFuture<'a, FileResult<Option<TextLine>>>;

    fn close(self: Box<Self>, ctx: &Context) -> BoxFuture<'_, ()>;
}

/// Positional reads from one opened regular file. All calls see the file that was opened, even
/// after its path is renamed or replaced.
pub trait BinaryReader: Send + Sync {
    /// Metadata of the opened file, whatever its path names now.
    fn info<'a>(&'a self, ctx: &'a Context) -> BoxFuture<'a, FileResult<FileInfo>>;

    /// Up to `length` bytes at `offset`; fewer only at the end of the file. One call
    /// allocates at most 1 MiB at a time, whatever `length` says, so a huge `length` costs
    /// only what the file yields.
    fn read<'a>(&'a self, offset: u64, length: usize, ctx: &'a Context) -> BoxFuture<'a, FileResult<Vec<u8>>>;

    /// One pass over the file inside the environment that locates lines
    /// `[start_line, end_line)` (`None`: to the end); see [`LineScan`]. An empty range fails
    /// `invalid`.
    fn scan_lines<'a>(
        &'a self,
        start_line: u64,
        end_line: Option<u64>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, FileResult<LineScan>>;

    fn close(self: Box<Self>, ctx: &Context) -> BoxFuture<'_, ()>;
}

/// Pages of one directory's entries.
pub trait DirReader: Send {
    /// Up to `max_entries` (at least one) entries in file-system order, continuing where the
    /// previous call stopped. Metadata is read only for returned entries; an entry removed
    /// meanwhile, or of an unsupported kind, is skipped. After a failed or aborted call the
    /// reader is closed and later calls fail `invalid`.
    fn next<'a>(&'a mut self, max_entries: usize, ctx: &'a Context) -> BoxFuture<'a, FileResult<DirPage>>;

    fn close(self: Box<Self>, ctx: &Context) -> BoxFuture<'_, ()>;
}

/// The files of an environment. Relative paths resolve against [`cwd`](Self::cwd).
pub trait FileSystem: Send + Sync {
    /// The file namespace: equal ids see the same files at the same paths, whatever their
    /// `cwd`. Every local environment shares one (`"local"`); each container or remote host has
    /// its own.
    fn id(&self) -> &str;

    /// The directory relative paths resolve against; fixed per environment.
    fn cwd(&self) -> &str;

    /// `path` resolved against the `cwd` and normalized. Pure path arithmetic.
    fn absolute_path(&self, path: &str) -> FileResult<String>;

    /// `parts` joined with the environment's separator and normalized. Pure path arithmetic.
    fn join_path(&self, parts: &[&str]) -> FileResult<String>;

    /// The whole file as UTF-8, invalid bytes as U+FFFD; a byte-order mark is kept.
    fn read_text_file<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<String>>;

    fn read_binary_file<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<Vec<u8>>>;

    /// The first `max_lines` lines (all with `None`) as a [`TextLineReader`] yields them.
    fn read_text_lines<'a>(
        &'a self,
        path: &'a str,
        max_lines: Option<usize>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, FileResult<Vec<String>>> {
        Box::pin(async move {
            if max_lines == Some(0) {
                return Ok(Vec::new());
            }
            let mut reader = self.open_text_line_reader(path, ctx).await?;
            let mut lines = Vec::new();
            let result = loop {
                if max_lines.is_some_and(|max| lines.len() >= max) {
                    break Ok(lines);
                }
                match reader.read_line(ctx).await {
                    Ok(Some(line)) => lines.push(line.text),
                    Ok(None) => break Ok(lines),
                    Err(error) => break Err(error),
                }
            };
            reader.close(ctx).await;
            result
        })
    }

    fn open_text_line_reader<'a>(
        &'a self,
        path: &'a str,
        ctx: &'a Context,
    ) -> BoxFuture<'a, FileResult<Box<dyn TextLineReader>>>;

    /// Open one regular file for positional reads. A directory fails `is_directory`, another
    /// non-regular file `invalid`; see [`OpenOptions::no_follow`].
    fn open_binary_reader<'a>(
        &'a self,
        path: &'a str,
        options: OpenOptions,
        ctx: &'a Context,
    ) -> BoxFuture<'a, FileResult<Box<dyn BinaryReader>>>;

    /// Replace the file's contents, creating it and its parent directories.
    fn write_file<'a>(&'a self, path: &'a str, content: &'a [u8], ctx: &'a Context) -> BoxFuture<'a, FileResult<()>>;

    /// Append to the file, creating it and its parent directories.
    fn append_file<'a>(&'a self, path: &'a str, content: &'a [u8], ctx: &'a Context) -> BoxFuture<'a, FileResult<()>>;

    /// Truncate or extend an existing file to exactly `size` bytes.
    fn truncate_file<'a>(&'a self, path: &'a str, size: u64, ctx: &'a Context) -> BoxFuture<'a, FileResult<()>>;

    /// Flush the file's contents and the metadata needed to retrieve them to storage.
    fn flush_file<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<()>>;

    /// Rename atomically, replacing an existing destination. Errors name the source.
    fn rename_file<'a>(&'a self, from: &'a str, to: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<()>>;

    /// Metadata of the path itself; a symbolic link is not followed.
    fn file_info<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<FileInfo>>;

    /// Every entry of a directory, in file-system order; see [`DirReader`] for large ones.
    fn list_dir<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<Vec<FileInfo>>>;

    fn open_dir_reader<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<Box<dyn DirReader>>>;

    /// Optional: the files, directories and symbolic links below a directory; see
    /// [`ListFilesOptions`]. Fails `not_supported` when the environment cannot list, so callers
    /// fall back to their own walk.
    fn list_files<'a>(
        &'a self,
        path: &'a str,
        options: ListFilesOptions,
        ctx: &'a Context,
    ) -> BoxFuture<'a, FileResult<FileListing>> {
        let _ = (path, options, ctx);
        Box::pin(async { Err(FileError::not_supported("list_files")) })
    }

    /// Optional: report changes to `targets`; see [`WatchTarget`] and [`WatchChange`]. Fails
    /// `not_supported` when the environment cannot watch.
    fn watch<'a>(
        &'a self,
        targets: Vec<WatchTarget>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, FileResult<Box<dyn FileWatcher>>> {
        let _ = (targets, ctx);
        Box::pin(async { Err(FileError::not_supported("watch")) })
    }

    /// The absolute path with every symbolic link resolved; the path must exist.
    fn canonical_path<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<String>>;

    /// Whether the path names anything, a dangling symbolic link included.
    fn exists<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<bool>> {
        Box::pin(async move {
            match self.file_info(path, ctx).await {
                Ok(_) => Ok(true),
                Err(error) if error.code == FileErrorCode::NotFound => Ok(false),
                Err(error) => Err(error),
            }
        })
    }

    /// Create a directory; with `recursive`, its missing parents too, and an existing directory
    /// is fine.
    fn create_dir<'a>(&'a self, path: &'a str, recursive: bool, ctx: &'a Context) -> BoxFuture<'a, FileResult<()>>;

    /// Remove a file, a symbolic link or, with [`RemoveOptions::recursive`], a directory tree.
    fn remove<'a>(&'a self, path: &'a str, options: RemoveOptions, ctx: &'a Context) -> BoxFuture<'a, FileResult<()>>;

    /// Create a new private directory (`0700`) under the system temp directory; returns its
    /// absolute path.
    fn create_temp_dir<'a>(&'a self, prefix: Option<&'a str>, ctx: &'a Context) -> BoxFuture<'a, FileResult<String>>;

    /// Create a new, empty private file (`0600`) under the system temp directory; returns its
    /// absolute path.
    fn create_temp_file<'a>(&'a self, options: TempFileOptions, ctx: &'a Context) -> BoxFuture<'a, FileResult<String>>;
}

/// The commands of an environment.
pub trait Shell: Send + Sync {
    /// Run a command: a [`Command::Shell`] string through the environment's shell, a
    /// [`Command::Argv`] directly. A non-zero exit is a result, not an error. A timeout or a
    /// cancelled context kills only this command's process tree.
    fn exec<'a>(
        &'a self,
        command: Command,
        options: ExecOptions<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<ExecResult, ExecError>>;

    /// Kill every command still running; for the owner's shutdown, never for one request.
    fn cleanup<'a>(&'a self, ctx: &'a Context) -> BoxFuture<'a, ()>;
}

/// A file system and a shell that see the same files.
pub trait ExecutionEnv: FileSystem + Shell {}

impl<T: FileSystem + Shell + ?Sized> ExecutionEnv for T {}
