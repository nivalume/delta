//! What only the JSONL backend has: its files, the publication protocol, reclamation and
//! recovery, with failures injected into the file system. After upstream's
//! `test/jsonl-storage.test.ts`; the contract itself is the conformance suite's.

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

mod support;

use std::path::Path;
use std::sync::{Arc, Mutex};

use delta_chord::context::Context;
use delta_harness::env::{
    BinaryReader, DirReader, FileError, FileErrorCode, FileInfo, FileResult, FileSystem, LocalEnv, OpenOptions,
    RemoveOptions, TempFileOptions, TextLineReader,
};
use delta_harness::records::{
    BaseContent, ConversationPolicy, DocumentContent, DocumentCreate, DocumentId, DocumentPoint, DocumentScope,
    EntryId, Fork, History, ROOT_CONVERSATION_ID, Seq, StorageWrite, TaskId, TaskOutcome, TaskRecord, TaskState,
};
use delta_harness::storage::{JsonlOptions, JsonlStorage, Storage, StorageError};
use futures::future::BoxFuture;
use pretty_assertions::assert_eq;
use serde_json::json;
use support::{TempDir, commit, create_root, ctx, entry, mint, pending_task, value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Operation {
    Append,
    Flush,
    Write,
    Rename,
    Remove,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Fails without doing anything.
    Before,
    /// Does it, then fails.
    After,
    /// Writes half the bytes (at least one), then fails.
    Short,
}

#[derive(Clone, Copy, Debug)]
struct Failure {
    operation: Operation,
    /// 1 for the first call of the operation since the failure was armed.
    call: usize,
    mode: Mode,
}

#[derive(Debug, Default)]
struct Log {
    operations: Vec<String>,
    calls: [usize; 5],
    failure: Option<Failure>,
}

/// The local file system, with the mutating operations logged and one failure injectable.
struct Instrumented {
    inner: LocalEnv,
    log: Arc<Mutex<Log>>,
}

/// The test's handle on an [`Instrumented`] file system.
#[derive(Clone)]
struct Probe(Arc<Mutex<Log>>);

impl Probe {
    fn fail(&self, failure: Failure) {
        *self.0.lock().unwrap() = Log {
            failure: Some(failure),
            ..Log::default()
        };
    }

    fn clear(&self) {
        *self.0.lock().unwrap() = Log::default();
    }

    fn operations(&self) -> Vec<String> {
        self.0.lock().unwrap().operations.clone()
    }
}

fn name(path: &str) -> String {
    Path::new(path).file_name().unwrap().to_string_lossy().into_owned()
}

impl Instrumented {
    fn new(dir: &Path) -> (Self, Probe) {
        let log = Arc::new(Mutex::new(Log::default()));
        let fs = Self {
            inner: LocalEnv::new(dir),
            log: Arc::clone(&log),
        };
        (fs, Probe(log))
    }

    /// Logs the call; the mode it fails with, when this is the armed call.
    fn record(&self, operation: Operation, entry: String) -> Option<Mode> {
        let mut log = self.log.lock().unwrap();
        log.operations.push(entry);
        let index = operation as usize;
        log.calls[index] += 1;
        let call = log.calls[index];
        log.failure
            .filter(|failure| failure.operation == operation && failure.call == call)
            .map(|failure| failure.mode)
    }
}

fn injected(what: &str, path: &str) -> FileError {
    FileError::new(
        FileErrorCode::Unknown,
        format!("injected {what} failure at {}", name(path)),
    )
}

fn half(content: &[u8]) -> &[u8] {
    &content[..(content.len() / 2).max(1)]
}

impl FileSystem for Instrumented {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn cwd(&self) -> &str {
        self.inner.cwd()
    }

    fn absolute_path(&self, path: &str) -> FileResult<String> {
        self.inner.absolute_path(path)
    }

    fn join_path(&self, parts: &[&str]) -> FileResult<String> {
        self.inner.join_path(parts)
    }

    fn read_text_file<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<String>> {
        self.inner.read_text_file(path, ctx)
    }

    fn read_binary_file<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<Vec<u8>>> {
        self.inner.read_binary_file(path, ctx)
    }

    fn open_text_line_reader<'a>(
        &'a self,
        path: &'a str,
        ctx: &'a Context,
    ) -> BoxFuture<'a, FileResult<Box<dyn TextLineReader>>> {
        self.inner.open_text_line_reader(path, ctx)
    }

    fn open_binary_reader<'a>(
        &'a self,
        path: &'a str,
        options: OpenOptions,
        ctx: &'a Context,
    ) -> BoxFuture<'a, FileResult<Box<dyn BinaryReader>>> {
        self.inner.open_binary_reader(path, options, ctx)
    }

    fn write_file<'a>(&'a self, path: &'a str, content: &'a [u8], ctx: &'a Context) -> BoxFuture<'a, FileResult<()>> {
        let mode = self.record(Operation::Write, format!("write:{}", name(path)));
        Box::pin(async move {
            match mode {
                None => self.inner.write_file(path, content, ctx).await,
                Some(Mode::Before) => Err(injected("write", path)),
                Some(Mode::Short) => {
                    self.inner.write_file(path, half(content), ctx).await?;
                    Err(injected("short write", path))
                }
                Some(Mode::After) => {
                    self.inner.write_file(path, content, ctx).await?;
                    Err(injected("post-write", path))
                }
            }
        })
    }

    fn append_file<'a>(&'a self, path: &'a str, content: &'a [u8], ctx: &'a Context) -> BoxFuture<'a, FileResult<()>> {
        let mode = self.record(Operation::Append, format!("append:{}", name(path)));
        Box::pin(async move {
            match mode {
                None => self.inner.append_file(path, content, ctx).await,
                Some(Mode::Before) => Err(injected("append", path)),
                Some(Mode::Short) => {
                    self.inner.append_file(path, half(content), ctx).await?;
                    Err(injected("short append", path))
                }
                Some(Mode::After) => {
                    self.inner.append_file(path, content, ctx).await?;
                    Err(injected("post-append", path))
                }
            }
        })
    }

    fn truncate_file<'a>(&'a self, path: &'a str, size: u64, ctx: &'a Context) -> BoxFuture<'a, FileResult<()>> {
        self.inner.truncate_file(path, size, ctx)
    }

    fn flush_file<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<()>> {
        let mode = self.record(Operation::Flush, format!("flush:{}", name(path)));
        Box::pin(async move {
            match mode {
                None => self.inner.flush_file(path, ctx).await,
                Some(Mode::After) => {
                    self.inner.flush_file(path, ctx).await?;
                    Err(injected("flush", path))
                }
                Some(_) => Err(injected("flush", path)),
            }
        })
    }

    fn rename_file<'a>(&'a self, from: &'a str, to: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<()>> {
        let mode = self.record(Operation::Rename, format!("rename:{}->{}", name(from), name(to)));
        Box::pin(async move {
            match mode {
                None => self.inner.rename_file(from, to, ctx).await,
                Some(Mode::After) => {
                    self.inner.rename_file(from, to, ctx).await?;
                    Err(injected("rename", from))
                }
                Some(_) => Err(injected("rename", from)),
            }
        })
    }

    fn file_info<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<FileInfo>> {
        self.inner.file_info(path, ctx)
    }

    fn list_dir<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<Vec<FileInfo>>> {
        self.inner.list_dir(path, ctx)
    }

    fn open_dir_reader<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<Box<dyn DirReader>>> {
        self.inner.open_dir_reader(path, ctx)
    }

    fn canonical_path<'a>(&'a self, path: &'a str, ctx: &'a Context) -> BoxFuture<'a, FileResult<String>> {
        self.inner.canonical_path(path, ctx)
    }

    fn create_dir<'a>(&'a self, path: &'a str, recursive: bool, ctx: &'a Context) -> BoxFuture<'a, FileResult<()>> {
        self.inner.create_dir(path, recursive, ctx)
    }

    fn remove<'a>(&'a self, path: &'a str, options: RemoveOptions, ctx: &'a Context) -> BoxFuture<'a, FileResult<()>> {
        let mode = self.record(Operation::Remove, format!("remove:{}", name(path)));
        Box::pin(async move {
            match mode {
                None => self.inner.remove(path, options, ctx).await,
                Some(Mode::After) => {
                    self.inner.remove(path, options, ctx).await?;
                    Err(injected("remove", path))
                }
                Some(_) => Err(injected("remove", path)),
            }
        })
    }

    fn create_temp_dir<'a>(&'a self, prefix: Option<&'a str>, ctx: &'a Context) -> BoxFuture<'a, FileResult<String>> {
        self.inner.create_temp_dir(prefix, ctx)
    }

    fn create_temp_file<'a>(&'a self, options: TempFileOptions, ctx: &'a Context) -> BoxFuture<'a, FileResult<String>> {
        self.inner.create_temp_file(options, ctx)
    }
}

fn dir_str(dir: &Path) -> String {
    dir.to_string_lossy().into_owned()
}

/// Opens `dir` on the plain local file system.
async fn open(dir: &Path) -> JsonlStorage {
    open_with(dir, JsonlOptions::default()).await
}

async fn open_with(dir: &Path, options: JsonlOptions) -> JsonlStorage {
    let fs: Arc<dyn FileSystem> = Arc::new(LocalEnv::new(dir));
    JsonlStorage::open(&dir_str(dir), fs, options, &ctx()).await.unwrap()
}

/// Opens `dir` on an instrumented file system.
async fn open_instrumented(dir: &Path, options: JsonlOptions) -> (JsonlStorage, Probe) {
    let (fs, probe) = Instrumented::new(dir);
    let storage = JsonlStorage::open(&dir_str(dir), Arc::new(fs), options, &ctx())
        .await
        .unwrap();
    (storage, probe)
}

/// The error opening `dir` fails with.
async fn open_error(dir: &Path) -> StorageError {
    let fs: Arc<dyn FileSystem> = Arc::new(LocalEnv::new(dir));
    JsonlStorage::open(&dir_str(dir), fs, JsonlOptions::default(), &ctx())
        .await
        .unwrap_err()
}

fn lines(path: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(path).unwrap();
    text.lines().map(str::to_owned).collect()
}

fn reclaim_files(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".reclaim"))
        .collect()
}

fn doc_path(dir: &Path, id: DocumentId) -> std::path::PathBuf {
    dir.join(format!("doc-{id}.jsonl"))
}

fn task_path(dir: &Path, id: TaskId) -> std::path::PathBuf {
    dir.join(format!("task-{id}.jsonl"))
}

fn session_document(id: DocumentId, kind: &str) -> DocumentCreate {
    DocumentCreate {
        id,
        kind: kind.to_owned(),
        key: None,
        scope: DocumentScope::Session,
        policy: None,
    }
}

fn create(record: DocumentCreate, content: serde_json::Value) -> StorageWrite {
    StorageWrite::DocumentCreate {
        record,
        content: BaseContent {
            version: 1,
            value: value(content),
        },
    }
}

fn base(id: DocumentId, content: serde_json::Value) -> StorageWrite {
    StorageWrite::DocumentChange {
        id,
        content: DocumentContent::Base {
            version: 1,
            value: value(content),
        },
    }
}

fn delta(id: DocumentId, ops: serde_json::Value) -> StorageWrite {
    StorageWrite::DocumentChange {
        id,
        content: DocumentContent::Delta {
            version: 1,
            ops: serde_json::from_value(ops).unwrap(),
        },
    }
}

fn task(record: TaskRecord) -> StorageWrite {
    StorageWrite::Task(record)
}

fn live_task(id: TaskId, phase: &str) -> TaskRecord {
    let mut record = pending_task(id, ROOT_CONVERSATION_ID);
    record.state = TaskState::Pending {
        checkpoint: value(json!({ "phase": phase })),
    };
    record
}

fn terminal_task(id: TaskId) -> TaskRecord {
    let mut record = pending_task(id, ROOT_CONVERSATION_ID);
    record.state = TaskState::Terminal {
        outcome: TaskOutcome::Completed {
            result: value(json!({ "ok": true })),
        },
    };
    record
}

async fn current(storage: &dyn Storage, id: DocumentId) -> Option<serde_json::Value> {
    storage
        .document(id, DocumentPoint::Current, &ctx())
        .await
        .unwrap()
        .map(|document| serde_json::to_value(&document.value).unwrap())
}

#[tokio::test]
async fn every_commit_publishes_before_current_only_documents_and_finished_tasks_are_reclaimed() {
    let dir = TempDir::new();
    let storage = open(dir.path()).await;
    create_root(&storage).await;
    let task_id: TaskId = mint(&storage).await;
    let document_id: DocumentId = mint(&storage).await;
    commit(&storage, vec![task(live_task(task_id, "ready"))]).await;
    commit(
        &storage,
        vec![create(
            session_document(document_id, "test.document"),
            json!({ "count": 0 }),
        )],
    )
    .await;
    commit(&storage, vec![delta(document_id, json!([]))]).await;
    commit(&storage, vec![base(document_id, json!({ "count": 1 }))]).await;
    assert_eq!(lines(&task_path(dir.path(), task_id)).len(), 1);
    assert_eq!(
        lines(&doc_path(dir.path(), document_id)).len(),
        1,
        "the base reclaimed the earlier revisions"
    );

    commit(&storage, vec![StorageWrite::DocumentRetire { id: document_id }]).await;
    commit(&storage, vec![task(terminal_task(task_id))]).await;
    let markers = lines(&dir.path().join("main.jsonl"));
    assert_eq!(markers.len(), 7);
    for marker in &markers {
        let marker: serde_json::Value = serde_json::from_str(marker).unwrap();
        assert_eq!(
            (marker["format"].as_u64(), marker["type"].as_str()),
            (Some(1), Some("commit"))
        );
    }
    assert!(!task_path(dir.path(), task_id).exists());
    assert!(!doc_path(dir.path(), document_id).exists());

    let reopened = open(dir.path()).await;
    assert_eq!(
        reopened.task(task_id, &ctx()).await.unwrap(),
        Some(terminal_task(task_id))
    );
    assert_eq!(current(&reopened, document_id).await, None);
}

#[tokio::test]
async fn live_task_replacements_in_one_commit_keep_their_order() {
    let dir = TempDir::new();
    let storage = open(dir.path()).await;
    create_root(&storage).await;
    let id: TaskId = mint(&storage).await;
    commit(
        &storage,
        vec![task(live_task(id, "first")), task(live_task(id, "second"))],
    )
    .await;
    assert_eq!(lines(&task_path(dir.path(), id)).len(), 2);
    let reopened = open(dir.path()).await;
    assert_eq!(reopened.task(id, &ctx()).await.unwrap(), Some(live_task(id, "second")));
}

#[tokio::test]
async fn a_torn_multi_record_sidecar_append_is_removed() {
    let dir = TempDir::new();
    let (storage, probe) = open_instrumented(dir.path(), JsonlOptions::default()).await;
    create_root(&storage).await;
    let id: TaskId = mint(&storage).await;
    probe.fail(Failure {
        operation: Operation::Append,
        call: 1,
        mode: Mode::Short,
    });
    let long = format!("second-{}", "x".repeat(512));
    let error = storage
        .commit(vec![task(live_task(id, "first")), task(live_task(id, &long))], &ctx())
        .await
        .unwrap_err();
    assert!(matches!(error, StorageError::Poisoned(_)), "{error:?}");
    let partial = std::fs::read(task_path(dir.path(), id)).unwrap();
    assert_ne!(partial.last(), Some(&b'\n'));
    assert_eq!(String::from_utf8_lossy(&partial).matches('\n').count(), 1);

    let reopened = open(dir.path()).await;
    assert_eq!(reopened.task(id, &ctx()).await.unwrap(), None);
    assert_eq!(std::fs::metadata(task_path(dir.path(), id)).unwrap().len(), 0);
}

#[tokio::test]
async fn a_refused_batch_writes_nothing_and_leaves_the_storage_usable() {
    let dir = TempDir::new();
    let (storage, probe) = open_instrumented(dir.path(), JsonlOptions::default()).await;
    create_root(&storage).await;
    let id: EntryId = mint(&storage).await;
    probe.clear();
    let error = storage
        .commit(
            vec![
                StorageWrite::Entry(entry(id, ROOT_CONVERSATION_ID)),
                StorageWrite::Entry(entry(id, ROOT_CONVERSATION_ID)),
            ],
            &ctx(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, StorageError::Rejected(_)), "{error:?}");
    assert_eq!(probe.operations(), Vec::<String>::new());
    assert_eq!(storage.entry(id, &ctx()).await.unwrap(), None);
    assert_eq!(
        commit(&storage, vec![StorageWrite::Entry(entry(id, ROOT_CONVERSATION_ID))]).await,
        Seq(2)
    );
}

#[tokio::test]
async fn an_append_failure_poisons_and_reopening_recovers_only_confirmed_commits() {
    for (call, mode) in [
        (1, Mode::Before),
        (1, Mode::After),
        (1, Mode::Short),
        (2, Mode::Before),
        (2, Mode::After),
        (3, Mode::Before),
        (3, Mode::Short),
        (3, Mode::After),
    ] {
        let dir = TempDir::new();
        let (storage, probe) = open_instrumented(dir.path(), JsonlOptions::default()).await;
        create_root(&storage).await;
        let first: DocumentId = mint(&storage).await;
        let second: DocumentId = mint(&storage).await;
        probe.fail(Failure {
            operation: Operation::Append,
            call,
            mode,
        });
        let error = storage
            .commit(
                vec![
                    create(session_document(first, "first"), json!({ "text": "α" })),
                    create(session_document(second, "second"), json!({ "text": "β" })),
                ],
                &ctx(),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, StorageError::Poisoned(_)), "{call} {mode:?}: {error:?}");
        let read = storage
            .document(first, DocumentPoint::Current, &ctx())
            .await
            .unwrap_err();
        assert!(matches!(read, StorageError::Poisoned(_)), "reads fail too: {read:?}");
        assert!(matches!(storage.mint_id().await, Err(StorageError::Poisoned(_))));

        let reopened = open(dir.path()).await;
        // Only the marker's own append (the third) having happened commits the batch.
        let survived = call == 3 && mode == Mode::After;
        assert_eq!(
            current(&reopened, first).await,
            survived.then(|| json!({ "text": "α" })),
            "{call} {mode:?}"
        );
        assert_eq!(
            current(&reopened, second).await,
            survived.then(|| json!({ "text": "β" })),
            "{call} {mode:?}"
        );
    }
}

#[tokio::test]
async fn a_sidecar_flush_failure_poisons_before_any_marker() {
    for (call, mode) in [(1, Mode::Before), (1, Mode::After), (2, Mode::Before), (2, Mode::After)] {
        let dir = TempDir::new();
        let (storage, probe) = open_instrumented(dir.path(), JsonlOptions { fsync: true }).await;
        create_root(&storage).await;
        let first: DocumentId = mint(&storage).await;
        let second: DocumentId = mint(&storage).await;
        probe.fail(Failure {
            operation: Operation::Flush,
            call,
            mode,
        });
        let error = storage
            .commit(
                vec![
                    create(session_document(first, "flush.first"), json!({})),
                    create(session_document(second, "flush.second"), json!({})),
                ],
                &ctx(),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, StorageError::Poisoned(_)), "{call} {mode:?}: {error:?}");
        assert!(!probe.operations().contains(&"append:main.jsonl".to_owned()));
        let reopened = open(dir.path()).await;
        assert_eq!(current(&reopened, first).await, None);
        assert_eq!(current(&reopened, second).await, None);
    }
}

#[tokio::test]
async fn flushes_order_publication_and_flush_main_only_to_authorize_reclamation() {
    for fsync in [false, true] {
        let dir = TempDir::new();
        let (storage, probe) = open_instrumented(dir.path(), JsonlOptions { fsync }).await;
        create_root(&storage).await;
        let first: DocumentId = mint(&storage).await;
        let second: DocumentId = mint(&storage).await;
        probe.clear();
        commit(
            &storage,
            vec![
                create(session_document(second, "second"), json!({})),
                create(session_document(first, "first"), json!({})),
            ],
        )
        .await;
        let mut expected = vec![
            format!("append:doc-{second}.jsonl"),
            format!("append:doc-{first}.jsonl"),
        ];
        if fsync {
            expected.extend([format!("flush:doc-{second}.jsonl"), format!("flush:doc-{first}.jsonl")]);
        }
        expected.push("append:main.jsonl".to_owned());
        assert_eq!(probe.operations(), expected, "fsync {fsync}");

        probe.clear();
        commit(&storage, vec![base(first, json!({ "checkpoint": true }))]).await;
        let mut expected = vec![format!("append:doc-{first}.jsonl")];
        if fsync {
            expected.push(format!("flush:doc-{first}.jsonl"));
        }
        expected.push("append:main.jsonl".to_owned());
        if fsync {
            expected.push("flush:main.jsonl".to_owned());
        }
        expected.push(format!("write:doc-{first}.jsonl.reclaim"));
        if fsync {
            expected.push(format!("flush:doc-{first}.jsonl.reclaim"));
        }
        expected.push(format!("rename:doc-{first}.jsonl.reclaim->doc-{first}.jsonl"));
        assert_eq!(probe.operations(), expected, "fsync {fsync}");

        probe.clear();
        let entry_id: EntryId = mint(&storage).await;
        commit(
            &storage,
            vec![StorageWrite::Entry(entry(entry_id, ROOT_CONVERSATION_ID))],
        )
        .await;
        assert_eq!(probe.operations(), ["append:main.jsonl"], "fsync {fsync}");

        let task_id: TaskId = mint(&storage).await;
        commit(&storage, vec![task(live_task(task_id, "ready"))]).await;
        probe.clear();
        commit(&storage, vec![task(terminal_task(task_id))]).await;
        let mut expected = vec!["append:main.jsonl".to_owned()];
        if fsync {
            expected.push("flush:main.jsonl".to_owned());
        }
        expected.push(format!("remove:task-{task_id}.jsonl"));
        assert_eq!(probe.operations(), expected, "fsync {fsync}");
    }
}

/// A document at seq 2 with a delta at seq 3; the caller commits a base at seq 4.
async fn document_with_a_delta(storage: &JsonlStorage) -> DocumentId {
    create_root(storage).await;
    let id: DocumentId = mint(storage).await;
    commit(
        storage,
        vec![create(session_document(id, "test.document"), json!({ "count": 0 }))],
    )
    .await;
    commit(storage, vec![delta(id, json!([["s", ["count"], 1]]))]).await;
    id
}

#[tokio::test]
async fn a_failed_base_reclamation_is_redone_by_the_next_open() {
    for (operation, mode) in [
        (Operation::Write, Mode::Before),
        (Operation::Write, Mode::Short),
        (Operation::Write, Mode::After),
        (Operation::Rename, Mode::Before),
        (Operation::Rename, Mode::After),
    ] {
        let dir = TempDir::new();
        let (storage, probe) = open_instrumented(dir.path(), JsonlOptions::default()).await;
        let id = document_with_a_delta(&storage).await;
        probe.fail(Failure {
            operation,
            call: 1,
            mode,
        });
        assert_eq!(commit(&storage, vec![base(id, json!({ "count": 2 }))]).await, Seq(4));
        assert_eq!(current(&storage, id).await, Some(json!({ "count": 2 })));
        storage.close(&ctx()).await.unwrap();

        let reopened = open(dir.path()).await;
        assert_eq!(
            current(&reopened, id).await,
            Some(json!({ "count": 2 })),
            "{operation:?} {mode:?}"
        );
        assert_eq!(lines(&doc_path(dir.path(), id)).len(), 1, "{operation:?} {mode:?}");
        assert_eq!(reclaim_files(dir.path()), Vec::<String>::new());
    }
}

#[tokio::test]
async fn a_failed_retirement_reclamation_is_redone_by_the_next_open() {
    for mode in [Mode::Before, Mode::After] {
        let dir = TempDir::new();
        let (storage, probe) = open_instrumented(dir.path(), JsonlOptions::default()).await;
        create_root(&storage).await;
        let task_id: TaskId = mint(&storage).await;
        let id: DocumentId = mint(&storage).await;
        let record = DocumentCreate {
            scope: DocumentScope::Task { task_id },
            ..session_document(id, "task.document")
        };
        commit(
            &storage,
            vec![task(live_task(task_id, "ready")), create(record, json!({ "count": 1 }))],
        )
        .await;
        probe.fail(Failure {
            operation: Operation::Remove,
            call: 1,
            mode,
        });
        assert_eq!(
            commit(&storage, vec![StorageWrite::DocumentRetire { id }]).await,
            Seq(3)
        );
        assert_eq!(current(&storage, id).await, None);
        storage.close(&ctx()).await.unwrap();

        let reopened = open(dir.path()).await;
        assert_eq!(current(&reopened, id).await, None);
        assert_eq!(
            reopened.task(task_id, &ctx()).await.unwrap(),
            Some(live_task(task_id, "ready"))
        );
        assert!(!doc_path(dir.path(), id).exists(), "{mode:?}");
        assert_eq!(reclaim_files(dir.path()), Vec::<String>::new());
    }
}

#[tokio::test]
async fn a_failed_authorizing_main_flush_defers_reclamation() {
    for mode in [Mode::Before, Mode::After] {
        let dir = TempDir::new();
        let (storage, probe) = open_instrumented(dir.path(), JsonlOptions { fsync: true }).await;
        create_root(&storage).await;
        let id: DocumentId = mint(&storage).await;
        commit(
            &storage,
            vec![create(session_document(id, "test.document"), json!({ "count": 0 }))],
        )
        .await;
        probe.fail(Failure {
            operation: Operation::Flush,
            call: 2,
            mode,
        });
        assert_eq!(commit(&storage, vec![base(id, json!({ "count": 2 }))]).await, Seq(3));
        assert_eq!(
            probe.operations(),
            [
                format!("append:doc-{id}.jsonl"),
                format!("flush:doc-{id}.jsonl"),
                "append:main.jsonl".to_owned(),
                "flush:main.jsonl".to_owned(),
            ]
        );
        assert_eq!(current(&storage, id).await, Some(json!({ "count": 2 })));
        assert_eq!(lines(&doc_path(dir.path(), id)).len(), 2);
        storage.close(&ctx()).await.unwrap();

        // The recovery's own reclamation is deferred the same way.
        let (fs, probe) = Instrumented::new(dir.path());
        probe.fail(Failure {
            operation: Operation::Flush,
            call: 1,
            mode,
        });
        let deferred = JsonlStorage::open(&dir_str(dir.path()), Arc::new(fs), JsonlOptions { fsync: true }, &ctx())
            .await
            .unwrap();
        assert_eq!(current(&deferred, id).await, Some(json!({ "count": 2 })));
        assert_eq!(probe.operations(), ["flush:main.jsonl"]);
        assert_eq!(lines(&doc_path(dir.path(), id)).len(), 2);
        deferred.close(&ctx()).await.unwrap();

        let reclaimed = open_with(dir.path(), JsonlOptions { fsync: true }).await;
        assert_eq!(current(&reclaimed, id).await, Some(json!({ "count": 2 })));
        assert_eq!(lines(&doc_path(dir.path(), id)).len(), 1, "{mode:?}");
    }
}

#[tokio::test]
async fn a_failed_replacement_flush_keeps_the_committed_base() {
    for mode in [Mode::Before, Mode::After] {
        let dir = TempDir::new();
        let (storage, probe) = open_instrumented(dir.path(), JsonlOptions { fsync: true }).await;
        create_root(&storage).await;
        let id: DocumentId = mint(&storage).await;
        commit(
            &storage,
            vec![create(session_document(id, "test.document"), json!({ "count": 0 }))],
        )
        .await;
        // Sidecar, main, then the replacement.
        probe.fail(Failure {
            operation: Operation::Flush,
            call: 3,
            mode,
        });
        assert_eq!(commit(&storage, vec![base(id, json!({ "count": 2 }))]).await, Seq(3));
        probe.clear();
        commit(&storage, vec![delta(id, json!([["s", ["count"], 3]]))]).await;
        storage.close(&ctx()).await.unwrap();

        let reopened = open_with(dir.path(), JsonlOptions { fsync: true }).await;
        assert_eq!(current(&reopened, id).await, Some(json!({ "count": 3 })), "{mode:?}");
        assert_eq!(lines(&doc_path(dir.path(), id)).len(), 2, "{mode:?}");
        assert_eq!(reclaim_files(dir.path()), Vec::<String>::new());
    }
}

#[tokio::test]
async fn a_failed_task_reclamation_is_redone_by_the_next_open() {
    for mode in [Mode::Before, Mode::After] {
        let dir = TempDir::new();
        let (storage, probe) = open_instrumented(dir.path(), JsonlOptions::default()).await;
        create_root(&storage).await;
        let id: TaskId = mint(&storage).await;
        commit(&storage, vec![task(live_task(id, "ready"))]).await;
        probe.fail(Failure {
            operation: Operation::Remove,
            call: 1,
            mode,
        });
        assert_eq!(commit(&storage, vec![task(terminal_task(id))]).await, Seq(3));
        assert_eq!(storage.task(id, &ctx()).await.unwrap(), Some(terminal_task(id)));
        storage.close(&ctx()).await.unwrap();

        let reopened = open(dir.path()).await;
        assert_eq!(reopened.task(id, &ctx()).await.unwrap(), Some(terminal_task(id)));
        assert!(!task_path(dir.path(), id).exists(), "{mode:?}");
        assert_eq!(reclaim_files(dir.path()), Vec::<String>::new());
    }
}

#[tokio::test]
async fn deltas_after_a_current_only_base_append_to_the_replacement() {
    let dir = TempDir::new();
    let storage = open(dir.path()).await;
    create_root(&storage).await;
    let id: DocumentId = mint(&storage).await;
    commit(
        &storage,
        vec![create(session_document(id, "test.document"), json!({ "count": 0 }))],
    )
    .await;
    commit(&storage, vec![base(id, json!({ "count": 10 }))]).await;
    commit(&storage, vec![delta(id, json!([["s", ["count"], 11]]))]).await;
    assert_eq!(lines(&doc_path(dir.path(), id)).len(), 2);
    let reopened = open(dir.path()).await;
    assert_eq!(current(&reopened, id).await, Some(json!({ "count": 11 })));
}

#[tokio::test]
async fn rewindable_history_is_never_reclaimed() {
    let dir = TempDir::new();
    let storage = open(dir.path()).await;
    create_root(&storage).await;
    let id: DocumentId = mint(&storage).await;
    let record = DocumentCreate {
        scope: DocumentScope::Conversation {
            conversation_id: ROOT_CONVERSATION_ID,
        },
        policy: Some(ConversationPolicy {
            history: History::Rewindable,
            fork: Fork::AsOf,
        }),
        ..session_document(id, "rewindable")
    };
    let created_at = commit(&storage, vec![create(record, json!({ "count": 0 }))]).await;
    let changed_at = commit(&storage, vec![delta(id, json!([["s", ["count"], 1]]))]).await;
    commit(&storage, vec![base(id, json!({ "count": 2 }))]).await;
    commit(&storage, vec![StorageWrite::DocumentRetire { id }]).await;
    assert_eq!(lines(&doc_path(dir.path(), id)).len(), 3);

    let reopened = open(dir.path()).await;
    let at = |seq| {
        let reopened = &reopened;
        async move {
            reopened
                .document(id, DocumentPoint::At(seq), &ctx())
                .await
                .unwrap()
                .map(|document| serde_json::to_value(&document.value).unwrap())
        }
    };
    assert_eq!(at(created_at).await, Some(json!({ "count": 0 })));
    assert_eq!(at(changed_at).await, Some(json!({ "count": 1 })));
    assert_eq!(current(&reopened, id).await, None);
    assert_eq!(lines(&doc_path(dir.path(), id)).len(), 3);
}

#[tokio::test]
async fn retired_task_session_and_latest_documents_are_reclaimed() {
    let dir = TempDir::new();
    let storage = open(dir.path()).await;
    create_root(&storage).await;
    let task_id: TaskId = mint(&storage).await;
    let session_id: DocumentId = mint(&storage).await;
    let latest_id: DocumentId = mint(&storage).await;
    let task_document_id: DocumentId = mint(&storage).await;
    let latest = DocumentCreate {
        scope: DocumentScope::Conversation {
            conversation_id: ROOT_CONVERSATION_ID,
        },
        policy: Some(ConversationPolicy {
            history: History::Latest,
            fork: Fork::Current,
        }),
        ..session_document(latest_id, "latest")
    };
    let task_document = DocumentCreate {
        scope: DocumentScope::Task { task_id },
        ..session_document(task_document_id, "task")
    };
    let created_at = commit(
        &storage,
        vec![
            task(live_task(task_id, "ready")),
            create(session_document(session_id, "session"), json!({})),
            create(latest, json!({})),
            create(task_document, json!({})),
        ],
    )
    .await;
    let retired_at = commit(
        &storage,
        vec![
            StorageWrite::DocumentRetire { id: session_id },
            StorageWrite::DocumentRetire { id: latest_id },
            StorageWrite::DocumentRetire { id: task_document_id },
            task(terminal_task(task_id)),
        ],
    )
    .await;
    for path in [
        doc_path(dir.path(), session_id),
        doc_path(dir.path(), latest_id),
        doc_path(dir.path(), task_document_id),
        task_path(dir.path(), task_id),
    ] {
        assert!(!path.exists(), "{}", path.display());
    }

    let reopened = open(dir.path()).await;
    assert_eq!(
        reopened.task(task_id, &ctx()).await.unwrap(),
        Some(terminal_task(task_id))
    );
    for id in [session_id, latest_id, task_document_id] {
        assert_eq!(current(&reopened, id).await, None);
    }
    let address = session_document(session_id, "session").address();
    let found = reopened
        .find_document(&address, DocumentPoint::At(created_at), &ctx())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (found.id, found.created_at, found.retired_at),
        (session_id, created_at, Some(retired_at))
    );
    assert_eq!(
        reopened
            .find_document(&address, DocumentPoint::At(retired_at), &ctx())
            .await
            .unwrap(),
        None
    );
}

#[tokio::test]
async fn torn_utf8_tails_are_truncated_at_their_byte_and_their_sequence_reused() {
    let dir = TempDir::new();
    let storage = open(dir.path()).await;
    create_root(&storage).await;
    let id: DocumentId = mint(&storage).await;
    commit(
        &storage,
        vec![create(session_document(id, "test.document"), json!({ "text": "kept" }))],
    )
    .await;
    storage.close(&ctx()).await.unwrap();
    let sidecar = doc_path(dir.path(), id);
    let main = dir.path().join("main.jsonl");
    let sizes = (
        std::fs::metadata(&sidecar).unwrap().len(),
        std::fs::metadata(&main).unwrap().len(),
    );
    let torn = "{\"text\":\"€".as_bytes();
    for path in [&sidecar, &main] {
        let mut bytes = std::fs::read(path).unwrap();
        bytes.extend_from_slice(&torn[..torn.len() - 1]);
        std::fs::write(path, bytes).unwrap();
    }

    let reopened = open(dir.path()).await;
    assert_eq!(
        (
            std::fs::metadata(&sidecar).unwrap().len(),
            std::fs::metadata(&main).unwrap().len()
        ),
        sizes
    );
    assert_eq!(commit(&reopened, vec![delta(id, json!([]))]).await, Seq(3));
}

#[tokio::test]
async fn unconfirmed_sidecar_tails_are_removed_without_resurrecting_finished_tasks() {
    let dir = TempDir::new();
    let storage = open(dir.path()).await;
    create_root(&storage).await;
    let id: TaskId = mint(&storage).await;
    commit(&storage, vec![task(live_task(id, "ready"))]).await;
    commit(&storage, vec![task(terminal_task(id))]).await;
    storage.close(&ctx()).await.unwrap();
    let sidecar = task_path(dir.path(), id);
    assert!(!sidecar.exists());
    let stale = json!({
        "format": 1,
        "type": "record",
        "seq": 4,
        "ordinal": 0,
        "payload": { "type": "task", "value": live_task(id, "stale") },
    });
    std::fs::write(&sidecar, format!("{stale}\n")).unwrap();

    let reopened = open(dir.path()).await;
    assert!(!sidecar.exists());
    assert_eq!(reopened.task(id, &ctx()).await.unwrap(), Some(terminal_task(id)));
    assert_eq!(commit(&reopened, vec![]).await, Seq(4));
}

#[tokio::test]
async fn missing_confirmed_data_fails_the_open() {
    let dir = TempDir::new();
    let storage = open(dir.path()).await;
    create_root(&storage).await;
    let id: DocumentId = mint(&storage).await;
    commit(&storage, vec![create(session_document(id, "test.document"), json!({}))]).await;
    storage.close(&ctx()).await.unwrap();
    std::fs::write(doc_path(dir.path(), id), "").unwrap();
    let error = open_error(dir.path()).await;
    assert!(
        matches!(&error, StorageError::Corrupt(message) if message.contains("Missing confirmed sidecar record")),
        "{error:?}"
    );
}

#[tokio::test]
async fn a_confirmed_record_after_an_unconfirmed_one_fails_the_open() {
    let dir = TempDir::new();
    let storage = open(dir.path()).await;
    create_root(&storage).await;
    let id: DocumentId = mint(&storage).await;
    commit(
        &storage,
        vec![create(session_document(id, "test.document"), json!({ "count": 0 }))],
    )
    .await;
    commit(&storage, vec![delta(id, json!([["s", ["count"], 1]]))]).await;
    storage.close(&ctx()).await.unwrap();
    let path = doc_path(dir.path(), id);
    let existing = lines(&path);
    let unconfirmed = json!({
        "format": 1,
        "type": "record",
        "seq": 2,
        "ordinal": 999,
        "payload": { "type": "document", "id": id, "content": { "kind": "delta", "version": 1, "ops": [] } },
    });
    std::fs::write(&path, format!("{}\n{unconfirmed}\n{}\n", existing[0], existing[1])).unwrap();
    let error = open_error(dir.path()).await;
    assert!(
        matches!(&error, StorageError::Corrupt(message) if message.contains("Confirmed record follows an unconfirmed tail")),
        "{error:?}"
    );
}

#[tokio::test]
async fn sequences_that_do_not_increase_fail_the_open() {
    let dir = TempDir::new();
    let storage = open(dir.path()).await;
    create_root(&storage).await;
    storage.close(&ctx()).await.unwrap();
    let path = dir.path().join("main.jsonl");
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, text.repeat(2)).unwrap();
    let error = open_error(dir.path()).await;
    assert!(
        matches!(&error, StorageError::Corrupt(message) if message.contains("Commit sequence does not strictly increase")),
        "{error:?}"
    );
}

#[tokio::test]
async fn invalid_document_content_fails_the_open() {
    let dir = TempDir::new();
    let storage = open(dir.path()).await;
    create_root(&storage).await;
    let id: DocumentId = mint(&storage).await;
    commit(&storage, vec![create(session_document(id, "test.document"), json!({}))]).await;
    storage.close(&ctx()).await.unwrap();
    let path = doc_path(dir.path(), id);
    let mut record: serde_json::Value = serde_json::from_str(lines(&path)[0].as_str()).unwrap();
    record["payload"]["content"].as_object_mut().unwrap().remove("value");
    std::fs::write(&path, format!("{record}\n")).unwrap();
    let error = open_error(dir.path()).await;
    assert!(
        matches!(&error, StorageError::Corrupt(message) if message.contains("Invalid document content")),
        "{error:?}"
    );
}

#[tokio::test]
async fn malformed_complete_lines_fail_the_open() {
    let dir = TempDir::new();
    let storage = open(dir.path()).await;
    create_root(&storage).await;
    storage.close(&ctx()).await.unwrap();
    let main = dir.path().join("main.jsonl");
    let mut text = std::fs::read_to_string(&main).unwrap();
    text.push_str("{bad}\n");
    std::fs::write(&main, text).unwrap();
    let error = open_error(dir.path()).await;
    assert!(
        matches!(&error, StorageError::Corrupt(message) if message.contains("Malformed complete main.jsonl line 2")),
        "{error:?}"
    );

    let dir = TempDir::new();
    let storage = open(dir.path()).await;
    create_root(&storage).await;
    storage.close(&ctx()).await.unwrap();
    std::fs::write(dir.path().join("doc-99.jsonl"), "{bad}\n").unwrap();
    let error = open_error(dir.path()).await;
    assert!(
        matches!(&error, StorageError::Corrupt(message) if message.contains("Malformed complete doc-99.jsonl line 1")),
        "{error:?}"
    );
}

#[tokio::test]
async fn a_new_directory_opens_empty_and_files_carry_delta_s_own_records() {
    let dir = TempDir::new();
    let storage = JsonlStorage::open_local(dir.path().join("a/b"), JsonlOptions::default())
        .await
        .unwrap();
    assert!(Path::new(storage.dir()).ends_with("a/b"));
    assert_eq!(storage.conversation(ROOT_CONVERSATION_ID, &ctx()).await.unwrap(), None);
    create_root(&storage).await;
    let marker: serde_json::Value = serde_json::from_str(&lines(&dir.path().join("a/b/main.jsonl"))[0]).unwrap();
    assert_eq!(marker["seq"], 1);
    assert_eq!(marker["writes"][0]["type"], "conversation");
    assert_eq!(marker["writes"][0]["value"]["id"], 1);
    storage.close(&ctx()).await.unwrap();
    storage.close(&ctx()).await.unwrap();
    assert!(matches!(storage.mint_id().await, Err(StorageError::Closed)));
}
