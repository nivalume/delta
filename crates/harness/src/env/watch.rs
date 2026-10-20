//! Change reports for hosts that load resources from an environment:
//! [`FileSystem::watch`](super::FileSystem::watch) and what it returns. An optional capability;
//! the default method fails with `not_supported`.

use futures::future::BoxFuture;

use delta_chord::context::Context;

use super::error::FileError;

/// A file or directory to watch. It may be missing; creating it is a change.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WatchTarget {
    pub path: String,
    /// Watch everything below a directory, not only its entries. Symbolic links below it are
    /// not followed; a watched symbolic link itself reports changes to what it points to.
    pub recursive: bool,
    /// Entries below `path` that are neither watched nor reported.
    pub exclude: Exclude,
}

/// Entries a [`WatchTarget`] skips with everything below them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Exclude {
    /// Names starting with `.`.
    pub hidden: bool,
    /// These names, such as `node_modules`.
    pub names: Vec<String>,
}

/// What changed.
#[derive(Debug)]
pub enum WatchChange {
    /// Something at or below each path may have changed; a directory covers its subtree.
    /// Reports may be spurious; a change is never missed while the watcher is healthy.
    Paths(Vec<String>),
    /// Coverage was uncertain for a while (lost events, a reconnect): rescan everything.
    Overflow,
    /// The watcher stopped, for example past the environment's directory limit; final.
    Error(FileError),
}

/// How reliably a [`FileWatcher`] reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WatchMode {
    /// Platform events trigger rescans; changes are reported within about two seconds.
    Native,
    /// Snapshots are compared periodically, because the file system does not report changes
    /// reliably (network and FUSE file systems); a change undone in between can be missed.
    Polling,
}

/// An established watch. Coverage starts when [`FileSystem::watch`](super::FileSystem::watch)
/// returns, so a host that watches before it loads cannot miss a change made during the load.
pub trait FileWatcher: Send {
    fn mode(&self) -> WatchMode;

    /// The next change; `None` once the watcher is closed or after a [`WatchChange::Error`].
    fn next(&mut self) -> BoxFuture<'_, Option<WatchChange>>;

    /// Stop watching; nothing is reported after this resolves.
    fn close(self: Box<Self>, ctx: &Context) -> BoxFuture<'_, ()>;
}
