//! The local readers: positional binary reads and line scans, text lines and directory pages.
//!
//! Binary and text readers hold a `std::fs::File` and read positionally on the blocking pool,
//! so they share no cursor, a read that was aborted consumes nothing, and a renamed or
//! replaced path does not change what they read.

use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use delta_chord::context::Context;
use futures::future::BoxFuture;
use tokio::fs;

use super::fs::{ensure_live, file_info_of, reading};
use crate::env::decode::Utf8Decoder;
use crate::env::line_scan::LineScanner;
use crate::env::{
    BinaryReader, DirPage, DirReader, FileError, FileErrorCode, FileInfo, FileResult, LineScan, OpenOptions, TextLine,
    TextLineReader,
};

/// Largest single allocation of a [`BinaryReader::read`].
const BINARY_READ_CHUNK: usize = 1024 * 1024;
/// Read size of line scans and text line readers.
const SCAN_CHUNK: usize = 64 * 1024;

/// Run blocking file work on tokio's blocking pool.
async fn blocking<T: Send + 'static>(
    path: &Path,
    work: impl FnOnce() -> io::Result<T> + Send + 'static,
) -> FileResult<T> {
    match tokio::task::spawn_blocking(work).await {
        Ok(result) => result.map_err(|error| FileError::from_io(error, path)),
        Err(error) => Err(FileError::new(FileErrorCode::Unknown, format!("File task failed: {error}")).with_path(path)),
    }
}

/// Read into `buf` at `offset`; fewer bytes only at the end of the file.
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        #[cfg(unix)]
        let read = std::os::unix::fs::FileExt::read_at(file, &mut buf[filled..], offset + filled as u64);
        #[cfg(windows)]
        let read = std::os::windows::fs::FileExt::seek_read(file, &mut buf[filled..], offset + filled as u64);
        match read {
            Ok(0) => break,
            Ok(read) => filled += read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(filled)
}

fn symlink_refused(path: &Path) -> FileError {
    FileError::invalid(format!("Refusing to follow a symbolic link: {}", path.display())).with_path(path)
}

/// Open `path` as one regular file; see [`FileSystem::open_binary_reader`](crate::env::FileSystem::open_binary_reader).
///
/// The kind is checked before opening, so a FIFO is never opened (which would block for a
/// writer). With `no_follow` the final component is checked with `lstat`, and on Unix the
/// opened file must be the one that check saw (same device and inode), so a path swapped for a
/// link in between is refused too.
pub(super) async fn open_binary_reader(
    path: PathBuf,
    options: OpenOptions,
    ctx: &Context,
) -> FileResult<Box<dyn BinaryReader>> {
    let before = if options.no_follow {
        reading(ctx, &path, fs::symlink_metadata(&path)).await?
    } else {
        reading(ctx, &path, fs::metadata(&path)).await?
    };
    if before.file_type().is_symlink() {
        return Err(symlink_refused(&path));
    }
    check_regular(&path, &before)?;
    let opened = path.clone();
    let file = blocking(&path, move || File::open(opened)).await?;
    let file = Arc::new(file);
    let handle = Arc::clone(&file);
    let after = blocking(&path, move || handle.metadata()).await?;
    check_regular(&path, &after)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if options.no_follow && (before.dev(), before.ino()) != (after.dev(), after.ino()) {
            return Err(symlink_refused(&path));
        }
    }
    ensure_live(ctx, &path)?;
    Ok(Box::new(LocalBinaryReader { file, path }))
}

fn check_regular(path: &Path, metadata: &std::fs::Metadata) -> FileResult<()> {
    if metadata.is_dir() {
        Err(FileError::new(
            FileErrorCode::IsDirectory,
            format!("Is a directory: {}", path.display()),
        )
        .with_path(path))
    } else if metadata.is_file() {
        Ok(())
    } else {
        Err(FileError::invalid(format!("Not a regular file: {}", path.display())).with_path(path))
    }
}

#[derive(Debug)]
struct LocalBinaryReader {
    file: Arc<File>,
    path: PathBuf,
}

impl BinaryReader for LocalBinaryReader {
    fn info<'a>(&'a self, ctx: &'a Context) -> BoxFuture<'a, FileResult<FileInfo>> {
        Box::pin(async move {
            ensure_live(ctx, &self.path)?;
            let file = Arc::clone(&self.file);
            let metadata = blocking(&self.path, move || file.metadata()).await?;
            file_info_of(&self.path, &metadata)
        })
    }

    fn read<'a>(&'a self, offset: u64, length: usize, ctx: &'a Context) -> BoxFuture<'a, FileResult<Vec<u8>>> {
        Box::pin(async move {
            ensure_live(ctx, &self.path)?;
            if offset.checked_add(length as u64).is_none() {
                return Err(FileError::invalid("Offset and length overflow").with_path(&self.path));
            }
            let file = Arc::clone(&self.file);
            let worker = ctx.clone();
            let bytes = blocking(&self.path, move || {
                let mut bytes = Vec::new();
                while bytes.len() < length && !worker.is_cancelled() {
                    // Grow by at most one chunk, so the allocation follows what the file yields.
                    let start = bytes.len();
                    let wanted = (length - start).min(BINARY_READ_CHUNK);
                    bytes.resize(start + wanted, 0);
                    let read = read_at(&file, &mut bytes[start..], offset + start as u64)?;
                    bytes.truncate(start + read);
                    if read < wanted {
                        break; // The end of the file.
                    }
                }
                Ok(bytes)
            })
            .await?;
            ensure_live(ctx, &self.path)?;
            Ok(bytes)
        })
    }

    fn scan_lines<'a>(
        &'a self,
        start_line: u64,
        end_line: Option<u64>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, FileResult<LineScan>> {
        Box::pin(async move {
            ensure_live(ctx, &self.path)?;
            let Some(mut scanner) = LineScanner::new(start_line, end_line) else {
                return Err(FileError::invalid("Invalid line range").with_path(&self.path));
            };
            let file = Arc::clone(&self.file);
            let worker = ctx.clone();
            let scan = blocking(&self.path, move || {
                let mut chunk = vec![0; SCAN_CHUNK];
                let mut position = 0;
                while !worker.is_cancelled() {
                    let read = read_at(&file, &mut chunk, position)?;
                    if read == 0 {
                        break;
                    }
                    scanner.push(&chunk[..read]);
                    position += read as u64;
                }
                Ok(scanner.finish())
            })
            .await?;
            ensure_live(ctx, &self.path)?;
            Ok(scan)
        })
    }

    fn close(self: Box<Self>, _ctx: &Context) -> BoxFuture<'_, ()> {
        // Dropping the last handle closes the file.
        Box::pin(async move { drop(self) })
    }
}

/// Open `path` for [`TextLineReader`] reads.
pub(super) async fn open_text_line_reader(path: PathBuf, ctx: &Context) -> FileResult<Box<dyn TextLineReader>> {
    ensure_live(ctx, &path)?;
    let opened = path.clone();
    let file = blocking(&path, move || File::open(opened)).await?;
    ensure_live(ctx, &path)?;
    Ok(Box::new(LocalTextLineReader {
        file: Arc::new(file),
        path,
        offset: 0,
        decoder: Utf8Decoder::stripping_bom(),
        buffer: String::new(),
        consumed: 0,
        searched: 0,
        ended: false,
    }))
}

/// A strict `\n` reader that reports whether each line was terminated.
#[derive(Debug)]
struct LocalTextLineReader {
    file: Arc<File>,
    path: PathBuf,
    /// File offset of the next read.
    offset: u64,
    decoder: Utf8Decoder,
    /// Decoded text not yet returned starts at `consumed`.
    buffer: String,
    consumed: usize,
    /// `buffer[consumed..searched]` holds no newline.
    searched: usize,
    ended: bool,
}

/// What the buffer of a [`LocalTextLineReader`] yields.
#[derive(Debug)]
enum Buffered {
    Line(TextLine),
    /// The end of the file.
    End,
    /// No complete line yet: read more.
    NeedMore,
}

impl LocalTextLineReader {
    /// The next complete line in the buffer, or the rest at the end of the file.
    fn take_line(&mut self) -> Buffered {
        if let Some(index) = self.buffer[self.searched..].find('\n') {
            let end = self.searched + index;
            let text = self.buffer[self.consumed..end].to_owned();
            self.consumed = end + 1;
            self.searched = self.consumed;
            self.compact();
            return Buffered::Line(TextLine { text, terminated: true });
        }
        self.searched = self.buffer.len();
        if !self.ended {
            return Buffered::NeedMore;
        }
        if self.consumed == self.buffer.len() {
            return Buffered::End;
        }
        let text = self.buffer[self.consumed..].to_owned();
        self.buffer.clear();
        self.consumed = 0;
        self.searched = 0;
        Buffered::Line(TextLine {
            text,
            terminated: false,
        })
    }

    /// Drop returned text once it is most of the buffer.
    fn compact(&mut self) {
        if self.consumed > self.buffer.len() / 2 {
            self.buffer.drain(..self.consumed);
            self.searched -= self.consumed;
            self.consumed = 0;
        }
    }
}

impl TextLineReader for LocalTextLineReader {
    fn read_line<'a>(&'a mut self, ctx: &'a Context) -> BoxFuture<'a, FileResult<Option<TextLine>>> {
        Box::pin(async move {
            loop {
                ensure_live(ctx, &self.path)?;
                match self.take_line() {
                    Buffered::Line(line) => return Ok(Some(line)),
                    Buffered::End => return Ok(None),
                    Buffered::NeedMore => {}
                }
                let file = Arc::clone(&self.file);
                let offset = self.offset;
                let chunk = blocking(&self.path, move || {
                    let mut chunk = vec![0; SCAN_CHUNK];
                    let read = read_at(&file, &mut chunk, offset)?;
                    chunk.truncate(read);
                    Ok(chunk)
                })
                .await?;
                // An aborted read consumes nothing: the next call reads the same bytes again.
                ensure_live(ctx, &self.path)?;
                self.offset += chunk.len() as u64;
                let text = if chunk.is_empty() {
                    self.ended = true;
                    self.decoder.finish_to_string()
                } else {
                    self.decoder.decode_to_string(&chunk)
                };
                self.buffer.push_str(&text);
            }
        })
    }

    fn close(self: Box<Self>, _ctx: &Context) -> BoxFuture<'_, ()> {
        Box::pin(async move { drop(self) })
    }
}

/// Open `path` for [`DirReader`] pages.
pub(super) async fn open_dir_reader(path: PathBuf, ctx: &Context) -> FileResult<Box<dyn DirReader>> {
    let entries = reading(ctx, &path, fs::read_dir(&path)).await?;
    ensure_live(ctx, &path)?;
    Ok(Box::new(LocalDirReader {
        entries,
        path,
        done: false,
        closed: false,
    }))
}

#[derive(Debug)]
struct LocalDirReader {
    entries: fs::ReadDir,
    path: PathBuf,
    done: bool,
    /// Set by a failed or aborted call.
    closed: bool,
}

impl LocalDirReader {
    fn fail(&mut self, error: FileError) -> FileResult<DirPage> {
        self.closed = true;
        Err(error)
    }
}

impl DirReader for LocalDirReader {
    fn next<'a>(&'a mut self, max_entries: usize, ctx: &'a Context) -> BoxFuture<'a, FileResult<DirPage>> {
        Box::pin(async move {
            if let Err(error) = ensure_live(ctx, &self.path) {
                return self.fail(error);
            }
            if self.closed {
                return Err(FileError::invalid("Directory reader is closed").with_path(&self.path));
            }
            if max_entries == 0 {
                return Err(FileError::invalid("max_entries must be positive").with_path(&self.path));
            }
            let mut entries = Vec::new();
            while !self.done && entries.len() < max_entries {
                let entry = match self.entries.next_entry().await {
                    Ok(Some(entry)) => entry,
                    Ok(None) => {
                        self.done = true;
                        break;
                    }
                    Err(error) => return self.fail(FileError::from_io(error, &self.path)),
                };
                if let Err(error) = ensure_live(ctx, &self.path) {
                    return self.fail(error);
                }
                let path = entry.path();
                match fs::symlink_metadata(&path).await {
                    Ok(metadata) => entries.extend(file_info_of(&path, &metadata).ok()),
                    // Removed since the directory was read: no longer an entry.
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                    Err(error) => return self.fail(FileError::from_io(error, &path)),
                }
            }
            Ok(DirPage {
                entries,
                done: self.done,
            })
        })
    }

    fn close(self: Box<Self>, _ctx: &Context) -> BoxFuture<'_, ()> {
        Box::pin(async move { drop(self) })
    }
}
