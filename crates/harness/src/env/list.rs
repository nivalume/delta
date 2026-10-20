//! Recursive listings for indexes such as file completion:
//! [`FileSystem::list_files`](super::FileSystem::list_files) and what it returns. An optional
//! capability; the default method fails with `not_supported`, and callers fall back to their
//! own walk over [`DirReader`](super::DirReader)s.

use std::time::Duration;

use super::FileKind;

/// Default of [`ListFilesOptions::max_entries`].
pub const DEFAULT_LIST_MAX_ENTRIES: usize = 50_000;
/// Default of [`ListFilesOptions::max_bytes`].
pub const DEFAULT_LIST_MAX_BYTES: usize = 4 * 1024 * 1024;
/// Largest accepted [`ListFilesOptions::max_bytes`] (15.9 MiB); more fails `invalid`.
pub const MAX_LIST_BYTES: usize = 16_672_358;
/// Default of [`ListFilesOptions::timeout`].
pub const DEFAULT_LIST_TIMEOUT: Duration = Duration::from_secs(10);

/// What [`FileSystem::list_files`](super::FileSystem::list_files) lists. Any bound ends the
/// listing early with [`FileListing::truncated`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListFilesOptions {
    /// Include names starting with `.`; default true. Entries named `.git` are never listed.
    pub hidden: bool,
    /// Inside a git work tree, skip what git ignores (the `.gitignore` files of the directory,
    /// its parents and subdirectories, `.git/info/exclude`, the global excludes); default true.
    /// A tracked file matching a rule is skipped too. Outside work trees it has no effect.
    pub gitignore: bool,
    /// Names of entries to skip with everything below them, such as `node_modules`.
    pub exclude: Vec<String>,
    pub max_entries: usize,
    /// Most bytes of the listing: each entry's UTF-8 path plus two; at most
    /// [`MAX_LIST_BYTES`].
    pub max_bytes: usize,
    pub timeout: Duration,
}

impl Default for ListFilesOptions {
    fn default() -> Self {
        Self {
            hidden: true,
            gitignore: true,
            exclude: Vec::new(),
            max_entries: DEFAULT_LIST_MAX_ENTRIES,
            max_bytes: DEFAULT_LIST_MAX_BYTES,
            timeout: DEFAULT_LIST_TIMEOUT,
        }
    }
}

/// One entry of a [`FileListing`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileListEntry {
    /// Relative to the listed directory, with `/` separators on every system.
    pub path: String,
    /// A symbolic link is listed as such and not followed.
    pub kind: FileKind,
}

/// The files, directories and symbolic links below a directory.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileListing {
    /// Depth first, parents before their entries, siblings in byte order of their names.
    pub entries: Vec<FileListEntry>,
    /// A bound of [`ListFilesOptions`] ended the listing early.
    pub truncated: bool,
}
