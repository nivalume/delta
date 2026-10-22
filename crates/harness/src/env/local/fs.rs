//! File operations of the local environment on `tokio::fs`, over paths already resolved
//! against the environment's `cwd`.
//!
//! Reads race the context and return `aborted` as soon as it is cancelled. Mutations check it
//! before each step but never race it: a write the operating system already started is not
//! reported as aborted while it completes in the background.

use std::fs::Metadata;
use std::future::Future;
use std::io;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use delta_chord::context::Context;
use tokio::fs;

use crate::env::error::path_string;
use crate::env::{FileError, FileErrorCode, FileInfo, FileKind, FileResult, RemoveOptions, TempFileOptions};

/// Fail `aborted` once `ctx` is cancelled.
pub(super) fn ensure_live(ctx: &Context, path: &Path) -> FileResult<()> {
    if ctx.is_cancelled() {
        Err(FileError::aborted().with_path(path))
    } else {
        Ok(())
    }
}

/// Run a read until it completes or `ctx` is cancelled.
pub(super) async fn reading<T>(
    ctx: &Context,
    path: &Path,
    operation: impl Future<Output = io::Result<T>>,
) -> FileResult<T> {
    ensure_live(ctx, path)?;
    match ctx.run(operation).await {
        Ok(result) => result.map_err(|error| FileError::from_io(error, path)),
        Err(_) => Err(FileError::aborted().with_path(path)),
    }
}

/// Run one step of a mutation unless `ctx` is already cancelled.
async fn mutating<T>(ctx: &Context, path: &Path, operation: impl Future<Output = io::Result<T>>) -> FileResult<T> {
    ensure_live(ctx, path)?;
    operation.await.map_err(|error| FileError::from_io(error, path))
}

/// [`FileInfo`] of `path` from its metadata; other kinds than files, directories and symbolic
/// links fail `invalid`.
pub(super) fn file_info_of(path: &Path, metadata: &Metadata) -> FileResult<FileInfo> {
    let file_type = metadata.file_type();
    let kind = if file_type.is_symlink() {
        FileKind::Symlink
    } else if file_type.is_dir() {
        FileKind::Directory
    } else if file_type.is_file() {
        FileKind::File
    } else {
        return Err(FileError::invalid(format!("Unsupported file type: {}", path.display())).with_path(path));
    };
    Ok(FileInfo {
        name: path
            .file_name()
            .map_or_else(|| path_string(path), |name| name.to_string_lossy().into_owned()),
        path: path_string(path),
        kind,
        size: metadata.len(),
        mtime_ms: metadata.modified().map_or(0, millis_since_epoch),
    })
}

fn millis_since_epoch(time: SystemTime) -> i64 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(after) => i64::try_from(after.as_millis()).unwrap_or(i64::MAX),
        Err(before) => i64::try_from(before.duration().as_millis()).map_or(i64::MIN, |millis| -millis),
    }
}

pub(super) async fn read_text_file(path: &Path, ctx: &Context) -> FileResult<String> {
    let bytes = read_binary_file(path, ctx).await?;
    Ok(match String::from_utf8(bytes) {
        Ok(text) => text,
        Err(error) => String::from_utf8_lossy(error.as_bytes()).into_owned(),
    })
}

pub(super) async fn read_binary_file(path: &Path, ctx: &Context) -> FileResult<Vec<u8>> {
    reading(ctx, path, fs::read(path)).await
}

/// Create the parent directories of `path`.
async fn create_parents(path: &Path, ctx: &Context) -> FileResult<()> {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => mutating(ctx, parent, fs::create_dir_all(parent)).await,
        _ => Ok(()),
    }
}

pub(super) async fn write_file(path: &Path, content: &[u8], ctx: &Context) -> FileResult<()> {
    create_parents(path, ctx).await?;
    mutating(ctx, path, fs::write(path, content)).await
}

pub(super) async fn append_file(path: &Path, content: &[u8], ctx: &Context) -> FileResult<()> {
    use tokio::io::AsyncWriteExt;

    create_parents(path, ctx).await?;
    mutating(ctx, path, async {
        let mut file = fs::OpenOptions::new().append(true).create(true).open(path).await?;
        file.write_all(content).await?;
        file.flush().await
    })
    .await
}

pub(super) async fn truncate_file(path: &Path, size: u64, ctx: &Context) -> FileResult<()> {
    if i64::try_from(size).is_err() {
        return Err(FileError::invalid("File size is too large").with_path(path));
    }
    mutating(ctx, path, async {
        let file = fs::OpenOptions::new().write(true).open(path).await?;
        file.set_len(size).await
    })
    .await
}

pub(super) async fn flush_file(path: &Path, ctx: &Context) -> FileResult<()> {
    // POSIX refuses to open a directory for writing and Windows to open one at all (access
    // denied), so check first and answer the same everywhere.
    let metadata = mutating(ctx, path, fs::metadata(path)).await?;
    if metadata.is_dir() {
        return Err(FileError::new(
            FileErrorCode::IsDirectory,
            format!("Is a directory: {}", path.display()),
        )
        .with_path(path));
    }
    let file = mutating(ctx, path, fs::OpenOptions::new().read(true).write(true).open(path)).await?;
    mutating(ctx, path, file.sync_all()).await
}

pub(super) async fn rename_file(from: &Path, to: &Path, ctx: &Context) -> FileResult<()> {
    mutating(ctx, from, fs::rename(from, to)).await
}

pub(super) async fn file_info(path: &Path, ctx: &Context) -> FileResult<FileInfo> {
    let metadata = reading(ctx, path, fs::symlink_metadata(path)).await?;
    file_info_of(path, &metadata)
}

pub(super) async fn list_dir(path: &Path, ctx: &Context) -> FileResult<Vec<FileInfo>> {
    let mut entries = reading(ctx, path, fs::read_dir(path)).await?;
    let mut infos = Vec::new();
    while let Some(entry) = reading(ctx, path, entries.next_entry()).await? {
        let entry_path = entry.path();
        match reading(ctx, &entry_path, fs::symlink_metadata(&entry_path)).await {
            Ok(metadata) => infos.extend(file_info_of(&entry_path, &metadata).ok()),
            // Removed since the directory was read: no longer an entry.
            Err(error) if error.code == FileErrorCode::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(infos)
}

pub(super) async fn canonical_path(path: &Path, ctx: &Context) -> FileResult<String> {
    reading(ctx, path, fs::canonicalize(path))
        .await
        .map(|path| path_string(&path))
}

pub(super) async fn create_dir(path: &Path, recursive: bool, ctx: &Context) -> FileResult<()> {
    if recursive {
        mutating(ctx, path, fs::create_dir_all(path)).await
    } else {
        mutating(ctx, path, fs::create_dir(path)).await
    }
}

/// `rm` semantics: `-r` for directories, `-f` for missing paths.
pub(super) async fn remove(path: &Path, options: RemoveOptions, ctx: &Context) -> FileResult<()> {
    let removed = match mutating(ctx, path, fs::symlink_metadata(path)).await {
        Ok(metadata) if metadata.is_dir() => {
            if !options.recursive {
                return Err(FileError::new(
                    FileErrorCode::IsDirectory,
                    format!("Is a directory (remove it recursively): {}", path.display()),
                )
                .with_path(path));
            }
            mutating(ctx, path, fs::remove_dir_all(path)).await
        }
        Ok(_) => mutating(ctx, path, fs::remove_file(path)).await,
        Err(error) => Err(error),
    };
    match removed {
        Err(error) if options.force && error.code == FileErrorCode::NotFound => Ok(()),
        removed => removed,
    }
}

/// A fresh random name part.
fn random_name() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

pub(super) async fn create_temp_dir(prefix: Option<&str>, ctx: &Context) -> FileResult<String> {
    let root = std::env::temp_dir();
    ensure_live(ctx, &root)?;
    let path = root.join(format!("{}{}", prefix.unwrap_or("delta-"), random_name()));
    #[cfg_attr(not(unix), allow(unused_mut))] // Only Unix sets the mode.
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    builder.mode(0o700);
    mutating(ctx, &path, builder.create(&path)).await?;
    Ok(path_string(&path))
}

pub(super) async fn create_temp_file(options: &TempFileOptions, ctx: &Context) -> FileResult<String> {
    let root = std::env::temp_dir();
    ensure_live(ctx, &root)?;
    let name = format!(
        "{}{}{}",
        options.prefix.as_deref().unwrap_or_default(),
        random_name(),
        options.suffix.as_deref().unwrap_or_default()
    );
    let path = root.join(name);
    let mut open = fs::OpenOptions::new();
    open.write(true).create_new(true);
    #[cfg(unix)]
    open.mode(0o600);
    mutating(ctx, &path, open.open(&path)).await?;
    Ok(path_string(&path))
}
