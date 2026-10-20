//! Helpers shared by the storage test files.

#![allow(dead_code, unreachable_pub)] // Each test file uses a subset.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use delta_chord::context::Context;
use delta_chord::json::Json;
use delta_harness::records::{
    ConversationId, ConversationRecord, EntryId, EntryRecord, Page, ROOT_CONVERSATION_ID, RecordId, Seq, StorageWrite,
    TaskId, TaskRecord, TaskState,
};
use delta_harness::storage::{MemoryStorage, SqliteOptions, SqliteStorage, Storage, StorageError};

/// A directory under the system temp directory, removed on drop.
#[derive(Debug)]
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new() -> Self {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let name = format!(
            "delta-harness-storage-{}-{nanos}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let path = std::env::temp_dir().join(name);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The backends every conformance case runs against.
#[derive(Clone, Copy, Debug)]
pub enum Backend {
    Memory,
    SqliteInMemory,
    SqliteFile,
    SqlitePrefixed,
}

/// One open storage for one case, with what the case needs to know about its backend.
pub struct Fixture {
    pub storage: Arc<dyn Storage>,
    /// The backend's largest id, which it never mints.
    pub max_id: u64,
    /// Keeps a file-backed database's directory until the case ends.
    pub dir: Option<TempDir>,
}

impl Backend {
    pub async fn open(self) -> Fixture {
        let sqlite_max = u64::try_from(i64::MAX).unwrap();
        match self {
            Self::Memory => Fixture {
                storage: Arc::new(MemoryStorage::new()),
                max_id: u64::MAX,
                dir: None,
            },
            Self::SqliteInMemory => Fixture {
                storage: Arc::new(SqliteStorage::open(":memory:", SqliteOptions::default()).await.unwrap()),
                max_id: sqlite_max,
                dir: None,
            },
            Self::SqliteFile | Self::SqlitePrefixed => {
                let dir = TempDir::new();
                let options = SqliteOptions {
                    prefix: matches!(self, Self::SqlitePrefixed).then(|| "delta_".to_owned()),
                    ..SqliteOptions::default()
                };
                let storage = SqliteStorage::open(dir.path().join("storage.sqlite"), options)
                    .await
                    .unwrap();
                Fixture {
                    storage: Arc::new(storage),
                    max_id: sqlite_max,
                    dir: Some(dir),
                }
            }
        }
    }
}

pub fn ctx() -> Context {
    Context::background()
}

pub async fn commit(storage: &dyn Storage, writes: Vec<StorageWrite>) -> Seq {
    storage.commit(writes, &ctx()).await.unwrap()
}

/// Commits `writes`, expects a `Rejected` error whose message contains `needle`.
pub async fn reject(storage: &dyn Storage, writes: Vec<StorageWrite>, needle: &str) -> StorageError {
    let error = storage.commit(writes, &ctx()).await.unwrap_err();
    assert!(
        matches!(error, StorageError::Rejected(_)) && error.to_string().contains(needle),
        "expected a rejection containing {needle:?}, got {error:?}"
    );
    error
}

pub async fn mint<I: RecordId>(storage: &dyn Storage) -> I {
    I::from(storage.mint_id().await.unwrap())
}

pub async fn create_root(storage: &dyn Storage) -> ConversationId {
    commit(
        storage,
        vec![StorageWrite::Conversation(ConversationRecord::new(
            ROOT_CONVERSATION_ID,
        ))],
    )
    .await;
    ROOT_CONVERSATION_ID
}

pub fn entry(id: EntryId, conversation: ConversationId) -> EntryRecord {
    EntryRecord::new(id, conversation, "message")
}

pub fn pending_task(id: TaskId, conversation: ConversationId) -> TaskRecord {
    TaskRecord::new(
        id,
        conversation,
        "test.task",
        1,
        delta_chord::json!({ "value": id.0 }),
        TaskState::Pending {
            checkpoint: delta_chord::json!({ "phase": "ready" }),
        },
    )
}

/// The ids of a page's items.
pub fn ids<T, I: RecordId>(page: &Page<T>, id: impl Fn(&T) -> I) -> Vec<I> {
    page.items.iter().map(id).collect()
}

/// A JSON value from `serde_json::json!` syntax.
pub fn value(value: serde_json::Value) -> Json {
    Json::from(value)
}
