//! The JSONL backend: the storage contract on append-only JSON Lines files.
//!
//! A directory holds `main.jsonl`, one commit marker per commit, and a sidecar per document
//! incarnation (`doc-<id>.jsonl`, its bases and deltas) and per live task
//! (`task-<id>.jsonl`, its whole records). A commit appends its sidecar records, then its
//! marker, and only then publishes the batch to the in-memory tables every read is served
//! from; opening replays the confirmed records and drops what no marker confirms. Bases and
//! retirements make earlier sidecar records of current-only documents and finished tasks
//! unnecessary, and the backend reclaims them after the authorizing commit.

mod format;
mod recover;

use std::collections::{BTreeMap, HashSet};
use std::future;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use delta_chord::context::Context;
use futures::FutureExt;

pub use self::format::JSONL_FORMAT_VERSION;
use self::format::{EncodedCommit, MAIN_FILE, RECLAIM_SUFFIX, current_only, doc_file, encode, task_file};
use super::batch::{Batch, RecordWrite, Target};
use super::error::StorageError;
use super::memory::MemoryStorage;
use super::{Storage, StorageFuture};
use crate::env::{FileError, FileErrorCode, FileSystem, LocalEnv, RemoveOptions};
use crate::records::{
    ConversationId, ConversationQuery, ConversationRecord, Cursor, DocumentAddress, DocumentContent, DocumentId,
    DocumentPoint, DocumentQuery, DocumentRecord, EntryId, EntryQuery, EntryRecord, Page, RawId, Seq, StorageWrite,
    StoredDocument, StoredEntry, SubmissionId, SubmissionQuery, SubmissionRecord, TaskId, TaskQuery, TaskRecord,
    TaskStatus,
};

/// Options of [`JsonlStorage::open`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JsonlOptions {
    /// Flush every affected sidecar before the commit marker is appended, flush `main.jsonl`
    /// before reclaiming, and flush a replacement before it is renamed. Without it, ordinary
    /// process crashes leave a consistent directory, but an acknowledged commit may be lost
    /// to a power, host, kernel or file-system failure.
    pub fsync: bool,
}

/// Storage in a directory of JSON Lines files.
///
/// Reads are served from memory tables rebuilt at open. A commit whose writes fail half
/// way poisons the storage: every later call fails with [`StorageError::Poisoned`] until it
/// is reopened, which recovers exactly the commits whose marker was written.
pub struct JsonlStorage {
    memory: MemoryStorage,
    files: Files,
    /// Serializes commits and close, and holds the tasks that have a sidecar.
    commit: tokio::sync::Mutex<LiveTasks>,
    closed: AtomicBool,
    poison: Mutex<Option<String>>,
}

/// Tasks whose newest record is live, so a sidecar holds them.
#[derive(Debug, Default)]
struct LiveTasks(HashSet<TaskId>);

/// The directory and how to write it.
struct Files {
    fs: Arc<dyn FileSystem>,
    dir: String,
    main: String,
    fsync: bool,
}

impl std::fmt::Debug for JsonlStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonlStorage")
            .field("dir", &self.files.dir)
            .field("fsync", &self.files.fsync)
            .field("closed", &self.closed.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

impl JsonlStorage {
    /// Opens the storage in `dir` on `fs`, creating the directory when it is missing, and
    /// recovers it: torn lines and unconfirmed sidecar tails are removed, confirmed records
    /// replayed. Missing confirmed data is [`StorageError::Corrupt`].
    pub async fn open(
        dir: &str,
        fs: Arc<dyn FileSystem>,
        options: JsonlOptions,
        ctx: &Context,
    ) -> Result<Self, StorageError> {
        let dir = fs.absolute_path(dir)?;
        fs.create_dir(&dir, true, ctx).await?;
        let main = fs.join_path(&[&dir, MAIN_FILE])?;
        let files = Files {
            fs,
            dir,
            main,
            fsync: options.fsync,
        };
        let memory = MemoryStorage::new();
        let live_tasks = recover::recover(&files, &memory, ctx).await?;
        Ok(Self {
            memory,
            files,
            commit: tokio::sync::Mutex::new(LiveTasks(live_tasks)),
            closed: AtomicBool::new(false),
            poison: Mutex::new(None),
        })
    }

    /// [`open`](Self::open) on this machine's file system; a relative `dir` is below the
    /// current directory.
    pub async fn open_local(dir: impl AsRef<Path>, options: JsonlOptions) -> Result<Self, StorageError> {
        let dir = dir.as_ref();
        let cwd = std::env::current_dir().map_err(|error| FileError::from_io(error, Path::new(".")))?;
        let dir = cwd.join(dir);
        let fs: Arc<dyn FileSystem> = Arc::new(LocalEnv::new(cwd));
        Self::open(&dir.to_string_lossy(), fs, options, &Context::background()).await
    }

    /// The directory's absolute path.
    pub fn dir(&self) -> &str {
        &self.files.dir
    }

    /// Fails when the storage is closed or poisoned.
    fn usable(&self) -> Result<(), StorageError> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(StorageError::Closed);
        }
        match &*self.poison.lock().unwrap_or_else(PoisonError::into_inner) {
            Some(cause) => Err(StorageError::Poisoned(cause.clone())),
            None => Ok(()),
        }
    }

    /// Poisons the storage with the first uncertain write failure.
    fn poison(&self, cause: String) -> StorageError {
        let mut poison = self.poison.lock().unwrap_or_else(PoisonError::into_inner);
        StorageError::Poisoned(poison.get_or_insert(cause).clone())
    }

    /// A read of the memory tables, when the storage is usable.
    fn read<'a, T: Send + 'a>(
        &'a self,
        read: impl FnOnce(&'a MemoryStorage) -> StorageFuture<'a, T>,
    ) -> StorageFuture<'a, T> {
        match self.usable() {
            Ok(()) => read(&self.memory),
            Err(error) => future::ready(Err(error)).boxed(),
        }
    }

    async fn commit_batch(&self, writes: Vec<StorageWrite>) -> Result<Seq, StorageError> {
        let mut live_tasks = self.commit.lock().await;
        self.usable()?;
        // Planning or encoding fails before anything is written; the storage stays usable.
        let prepared = self.memory.prepare(writes)?;
        let encoded = encode(prepared.seq, &prepared.batch)?;
        let reclamations = reclamations(&prepared.batch, &encoded, &live_tasks.0);
        let paths = encoded
            .sidecars
            .keys()
            .map(|file| self.files.path(file).map(|path| (file.as_str(), path)))
            .collect::<Result<Vec<_>, FileError>>()?;
        self.files
            .publish(&encoded, &paths)
            .await
            .map_err(|cause| self.poison(cause))?;
        live_tasks.adopt(&prepared.batch);
        let seq = self.memory.publish(prepared);
        self.files.reclaim(reclamations).await;
        Ok(seq)
    }
}

impl LiveTasks {
    /// Follows the tasks a published batch wrote.
    fn adopt(&mut self, batch: &Batch) {
        for write in &batch.records {
            if let RecordWrite::Task(task) = write {
                if task.state.status() == TaskStatus::Terminal {
                    self.0.remove(&task.id);
                } else {
                    self.0.insert(task.id);
                }
            }
        }
    }
}

/// The sidecars a batch makes reclaimable, with their new content (empty: remove): a
/// retired current-only incarnation's, a current-only incarnation's after a new base (just
/// that base), a task's once it is terminal. Rewindable history is never reclaimed.
fn reclamations(batch: &Batch, encoded: &EncodedCommit, live_tasks: &HashSet<TaskId>) -> BTreeMap<String, Vec<u8>> {
    let mut replacements = BTreeMap::new();
    for action in &batch.documents {
        let current = match &action.target {
            Target::Created(create) => current_only(create),
            Target::Existing(record) => !record.keeps_history(),
        };
        if !current {
            continue;
        }
        let file = doc_file(action.id);
        if action.retire {
            replacements.insert(file, Vec::new());
        } else if matches!(
            (&action.target, &action.content),
            (Target::Existing(_), Some(DocumentContent::Base { .. }))
        ) && let Some(content) = encoded.sidecars.get(&file)
        {
            replacements.insert(file, content.clone());
        }
    }
    let mut final_tasks = BTreeMap::new();
    for write in &batch.records {
        if let RecordWrite::Task(task) = write {
            final_tasks.insert(task.id, task.state.status());
        }
    }
    for (id, status) in final_tasks {
        let file = task_file(id);
        if status == TaskStatus::Terminal && (live_tasks.contains(&id) || encoded.sidecars.contains_key(&file)) {
            replacements.insert(file, Vec::new());
        }
    }
    replacements
}

impl Files {
    fn path(&self, file: &str) -> Result<String, FileError> {
        self.fs.join_path(&[&self.dir, file])
    }

    /// The publication protocol: every sidecar append, the flushes in durable mode, then the
    /// marker. Runs to the end once started (a cancelled call would leave an uncertain
    /// write); an error is the cause to poison with.
    async fn publish(&self, encoded: &EncodedCommit, paths: &[(&str, String)]) -> Result<(), String> {
        let ctx = Context::background();
        for (file, path) in paths {
            let content = encoded.sidecars.get(*file).map_or(&[][..], Vec::as_slice);
            self.fs
                .append_file(path, content, &ctx)
                .await
                .map_err(|error| format!("the append to {file} failed: {error}"))?;
        }
        if self.fsync {
            for (file, path) in paths {
                self.fs
                    .flush_file(path, &ctx)
                    .await
                    .map_err(|error| format!("the flush of {file} failed: {error}"))?;
            }
        }
        self.fs
            .append_file(&self.main, &encoded.marker, &ctx)
            .await
            .map_err(|error| format!("the append to {MAIN_FILE} failed: {error}"))
    }

    /// Replaces or removes sidecars whose committed state no longer needs their records.
    /// Best effort: the marker already published the state, and a reclamation left undone
    /// is redone by the next open. In durable mode `main.jsonl` is flushed first, so the
    /// authorizing marker cannot be lost while the reclamation survives.
    async fn reclaim(&self, replacements: BTreeMap<String, Vec<u8>>) {
        if replacements.is_empty() {
            return;
        }
        let ctx = Context::background();
        if self.fsync && self.fs.flush_file(&self.main, &ctx).await.is_err() {
            return;
        }
        for (file, content) in replacements {
            let _ = self.replace(&file, &content, &ctx).await;
        }
    }

    /// Writes `content` to a temporary file and renames it over `file`; empty content
    /// removes `file`.
    async fn replace(&self, file: &str, content: &[u8], ctx: &Context) -> Result<(), FileError> {
        let path = self.path(file)?;
        if content.is_empty() {
            let options = RemoveOptions {
                recursive: false,
                force: true,
            };
            return self.fs.remove(&path, options, ctx).await;
        }
        let temporary = self.path(&format!("{file}{RECLAIM_SUFFIX}"))?;
        self.fs.write_file(&temporary, content, ctx).await?;
        if self.fsync {
            self.fs.flush_file(&temporary, ctx).await?;
        }
        self.fs.rename_file(&temporary, &path, ctx).await
    }

    /// The file's bytes; empty when it does not exist.
    async fn read(&self, path: &str, ctx: &Context) -> Result<Vec<u8>, FileError> {
        match self.fs.read_binary_file(path, ctx).await {
            Err(error) if error.code == FileErrorCode::NotFound => Ok(Vec::new()),
            result => result,
        }
    }
}

impl Storage for JsonlStorage {
    fn commit(&self, writes: Vec<StorageWrite>, _ctx: &Context) -> StorageFuture<'_, Seq> {
        self.commit_batch(writes).boxed()
    }

    fn mint_id(&self) -> StorageFuture<'_, RawId> {
        self.read(MemoryStorage::mint_id)
    }

    fn conversation(&self, id: ConversationId, ctx: &Context) -> StorageFuture<'_, Option<ConversationRecord>> {
        self.read(|memory| memory.conversation(id, ctx))
    }

    fn scan_conversations(
        &self,
        query: ConversationQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<ConversationRecord>> {
        self.read(|memory| memory.scan_conversations(query, limit, cursor, ctx))
    }

    fn entry(&self, id: EntryId, ctx: &Context) -> StorageFuture<'_, Option<StoredEntry>> {
        self.read(|memory| memory.entry(id, ctx))
    }

    fn visible_entry(
        &self,
        conversation: ConversationId,
        id: EntryId,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<StoredEntry>> {
        self.read(|memory| memory.visible_entry(conversation, id, ctx))
    }

    fn find_latest_head_marker(
        &self,
        conversation: ConversationId,
        at_or_before: Option<EntryId>,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<EntryRecord>> {
        self.read(|memory| memory.find_latest_head_marker(conversation, at_or_before, ctx))
    }

    fn scan_entries(
        &self,
        query: EntryQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<EntryRecord>> {
        self.read(|memory| memory.scan_entries(query, limit, cursor, ctx))
    }

    fn task(&self, id: TaskId, ctx: &Context) -> StorageFuture<'_, Option<TaskRecord>> {
        self.read(|memory| memory.task(id, ctx))
    }

    fn scan_tasks(
        &self,
        query: TaskQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<TaskRecord>> {
        self.read(|memory| memory.scan_tasks(query, limit, cursor, ctx))
    }

    fn submission(&self, id: SubmissionId, ctx: &Context) -> StorageFuture<'_, Option<SubmissionRecord>> {
        self.read(|memory| memory.submission(id, ctx))
    }

    fn scan_submissions(
        &self,
        query: SubmissionQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<SubmissionRecord>> {
        self.read(|memory| memory.scan_submissions(query, limit, cursor, ctx))
    }

    fn submission_by_request(
        &self,
        conversation: ConversationId,
        request_id: &str,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<SubmissionRecord>> {
        self.read(|memory| memory.submission_by_request(conversation, request_id, ctx))
    }

    fn find_document(
        &self,
        address: &DocumentAddress,
        at: DocumentPoint,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<DocumentRecord>> {
        self.read(|memory| memory.find_document(address, at, ctx))
    }

    fn document(&self, id: DocumentId, at: DocumentPoint, ctx: &Context) -> StorageFuture<'_, Option<StoredDocument>> {
        self.read(|memory| memory.document(id, at, ctx))
    }

    fn scan_documents(
        &self,
        query: DocumentQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<DocumentRecord>> {
        self.read(|memory| memory.scan_documents(query, limit, cursor, ctx))
    }

    /// Refuses later calls, waits for a commit underway, then releases the tables.
    fn close(&self, ctx: &Context) -> StorageFuture<'_, ()> {
        self.closed.store(true, Ordering::SeqCst);
        let ctx = ctx.clone();
        async move {
            let _commit = self.commit.lock().await;
            self.memory.close(&ctx).await
        }
        .boxed()
    }
}
