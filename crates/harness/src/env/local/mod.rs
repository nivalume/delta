//! [`LocalEnv`]: the execution environment of the local machine, on `tokio::fs` and
//! `tokio::process`.
//!
//! Path arithmetic lives in `path`, file operations in `fs` and `readers`, commands in `shell`
//! with their process trees in `process` and spills in `spill`. `watch` and `list_files` keep
//! the traits' `not_supported` defaults until `notify` and `ignore` are dependencies.

mod fs;
mod path;
mod process;
mod readers;
mod shell;
mod spill;

use std::collections::BTreeMap;
use std::path::PathBuf;

use delta_chord::context::Context;
use futures::future::BoxFuture;

use self::process::ChildRegistry;
use self::shell::ShellConfig;
use super::error::path_string;
use super::{
    BinaryReader, Command, DirReader, ExecError, ExecOptions, ExecResult, FileInfo, FileResult, FileSystem,
    OpenOptions, RemoveOptions, Shell, TempFileOptions, TextLineReader,
};

/// The local machine as an [`ExecutionEnv`](super::ExecutionEnv).
///
/// Its `cwd` is fixed; the harness builds one per tool call from the conversation's `cwd`.
/// Commands run in their own process group, which a timeout, a cancellation or
/// [`Shell::cleanup`] kills as a whole; dropping an `exec` future kills its command too.
#[derive(Debug)]
pub struct LocalEnv {
    cwd: PathBuf,
    /// `cwd` as reported by [`FileSystem::cwd`].
    cwd_string: String,
    shell_path: Option<PathBuf>,
    shell_env: BTreeMap<String, String>,
    children: ChildRegistry,
}

impl LocalEnv {
    /// An environment whose relative paths resolve against `cwd`; a relative `cwd` resolves
    /// against the process's current directory.
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        let cwd = cwd.into();
        let cwd = path::normalize(&std::path::absolute(&cwd).unwrap_or(cwd));
        Self {
            cwd_string: path_string(&cwd),
            cwd,
            shell_path: None,
            shell_env: BTreeMap::new(),
            children: ChildRegistry::default(),
        }
    }

    /// Run [`Command::Shell`] strings with this shell (as `<shell> -c <script>`) instead of
    /// the discovered `bash`.
    #[must_use]
    pub fn with_shell(mut self, path: impl Into<PathBuf>) -> Self {
        self.shell_path = Some(path.into());
        self
    }

    /// Variables every command gets over the inherited ones, below
    /// [`ExecOptions::env`]; not applied with `inherit_env: false`.
    #[must_use]
    pub fn with_shell_env(mut self, env: BTreeMap<String, String>) -> Self {
        self.shell_env = env;
        self
    }

    fn resolve(&self, path: &str) -> PathBuf {
        path::resolve(&self.cwd, path)
    }
}

impl FileSystem for LocalEnv {
    fn id(&self) -> &'static str {
        "local"
    }

    fn cwd(&self) -> &str {
        &self.cwd_string
    }

    fn absolute_path(&self, path: &str) -> FileResult<String> {
        Ok(path_string(&self.resolve(path)))
    }

    fn join_path(&self, parts: &[&str]) -> FileResult<String> {
        Ok(path_string(&path::join(parts)))
    }

    fn read_text_file<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<String>> {
        Box::pin(async move { fs::read_text_file(&self.resolve(path), ctx).await })
    }

    fn read_binary_file<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<Vec<u8>>> {
        Box::pin(async move { fs::read_binary_file(&self.resolve(path), ctx).await })
    }

    fn open_text_line_reader<'a>(
        &'a self,
        path: &'a str,
        ctx: &'a Context,
    ) -> BoxFuture<'a, FileResult<Box<dyn TextLineReader>>> {
        Box::pin(readers::open_text_line_reader(self.resolve(path), ctx))
    }

    fn open_binary_reader<'a>(
        &'a self,
        path: &'a str,
        options: OpenOptions,
        ctx: &'a Context,
    ) -> BoxFuture<'a, FileResult<Box<dyn BinaryReader>>> {
        Box::pin(readers::open_binary_reader(self.resolve(path), options, ctx))
    }

    fn write_file<'a>(&'a self, path: &'a str, content: &'a [u8], ctx: &'a Context) -> BoxFuture<'a, FileResult<()>> {
        Box::pin(async move { fs::write_file(&self.resolve(path), content, ctx).await })
    }

    fn append_file<'a>(&'a self, path: &'a str, content: &'a [u8], ctx: &'a Context) -> BoxFuture<'a, FileResult<()>> {
        Box::pin(async move { fs::append_file(&self.resolve(path), content, ctx).await })
    }

    fn truncate_file<'a>(&'a self, path: &'a str, size: u64, ctx: &'a Context) -> BoxFuture<'a, FileResult<()>> {
        Box::pin(async move { fs::truncate_file(&self.resolve(path), size, ctx).await })
    }

    fn flush_file<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<()>> {
        Box::pin(async move { fs::flush_file(&self.resolve(path), ctx).await })
    }

    fn rename_file<'a>(&'a self, from: &'a str, to: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<()>> {
        Box::pin(async move { fs::rename_file(&self.resolve(from), &self.resolve(to), ctx).await })
    }

    fn file_info<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<FileInfo>> {
        Box::pin(async move { fs::file_info(&self.resolve(path), ctx).await })
    }

    fn list_dir<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<Vec<FileInfo>>> {
        Box::pin(async move { fs::list_dir(&self.resolve(path), ctx).await })
    }

    fn open_dir_reader<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<Box<dyn DirReader>>> {
        Box::pin(readers::open_dir_reader(self.resolve(path), ctx))
    }

    fn canonical_path<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<String>> {
        Box::pin(async move { fs::canonical_path(&self.resolve(path), ctx).await })
    }

    fn create_dir<'a>(&'a self, path: &'a str, recursive: bool, ctx: &'a Context) -> BoxFuture<'a, FileResult<()>> {
        Box::pin(async move { fs::create_dir(&self.resolve(path), recursive, ctx).await })
    }

    fn remove<'a>(&'a self, path: &'a str, options: RemoveOptions, ctx: &'a Context) -> BoxFuture<'a, FileResult<()>> {
        Box::pin(async move { fs::remove(&self.resolve(path), options, ctx).await })
    }

    fn create_temp_dir<'a>(&'a self, prefix: Option<&'a str>, ctx: &'a Context) -> BoxFuture<'a, FileResult<String>> {
        Box::pin(fs::create_temp_dir(prefix, ctx))
    }

    fn create_temp_file<'a>(&'a self, options: TempFileOptions, ctx: &'a Context) -> BoxFuture<'a, FileResult<String>> {
        Box::pin(async move { fs::create_temp_file(&options, ctx).await })
    }
}

impl Shell for LocalEnv {
    fn exec<'a>(
        &'a self,
        command: Command,
        options: ExecOptions<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<ExecResult, ExecError>> {
        let config = ShellConfig {
            cwd: &self.cwd,
            shell_path: self.shell_path.as_deref(),
            shell_env: &self.shell_env,
            children: &self.children,
        };
        Box::pin(shell::exec(config, command, options, ctx))
    }

    fn cleanup<'a>(&'a self, _ctx: &'a Context) -> BoxFuture<'a, ()> {
        Box::pin(self.children.kill_all())
    }
}
