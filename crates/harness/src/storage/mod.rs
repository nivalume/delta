//! Durable storage: the [`Storage`] contract and its memory and SQLite backends.
//!
//! The Session commits one batch of [`StorageWrite`]s per transaction; storage makes it
//! atomic, stamps it with a sequence, keeps one id namespace, materializes documents from a
//! base and its delta tail, and answers keyed lookups and cursor scans in id order. The
//! commit rules and the scan, ancestry and replay algorithms are shared by the backends, so
//! a backend only maps them onto its tables. Corresponds to upstream `src/types.ts`
//! (`Storage`) and `src/storage/`. See `docs/design/storage.md`.

mod ancestry;
mod batch;
mod error;
mod memory;
mod replay;
mod scan;
mod sqlite;

use delta_chord::context::Context;
use futures::future::BoxFuture;

pub use error::StorageError;
pub use memory::MemoryStorage;
pub use sqlite::{CURRENT_SCHEMA_VERSION, Schema, SqliteOptions, SqliteStorage, migrations};

use crate::records::{
    ConversationId, ConversationQuery, ConversationRecord, Cursor, DocumentAddress, DocumentId, DocumentPoint,
    DocumentQuery, DocumentRecord, EntryId, EntryQuery, EntryRecord, Page, RawId, Seq, StorageWrite, StoredDocument,
    StoredEntry, SubmissionId, SubmissionQuery, SubmissionRecord, TaskId, TaskQuery, TaskRecord,
};

/// The future every [`Storage`] method returns.
pub type StorageFuture<'a, T> = BoxFuture<'a, Result<T, StorageError>>;

/// A storage backend.
///
/// Object-safe, so the Session holds an `Arc<dyn Storage>`. Every value crossing the
/// boundary is owned, so neither side can change what the other retains.
///
/// The trait trusts the Session for semantic validity (references, ancestry, transitions)
/// and enforces atomicity, id ownership, create-once records and document record
/// consistency. The Session serializes commits. A backend retries its own transient
/// failures; what it returns is final. `ctx` carries cancellation for reads; the built-in
/// backends finish a call once it started.
pub trait Storage: Send + Sync + 'static {
    /// Persists one batch atomically and returns its sequence; once it returns, later reads
    /// observe it. A refused batch ([`StorageError::Rejected`]) has no effect.
    fn commit(&self, writes: Vec<StorageWrite>, ctx: &Context) -> StorageFuture<'_, Seq>;

    /// An id above every id minted or stored before, also across reopen. Minted ids that no
    /// commit used may be minted again after a reopen.
    fn mint_id(&self) -> StorageFuture<'_, RawId>;

    fn conversation(&self, id: ConversationId, ctx: &Context) -> StorageFuture<'_, Option<ConversationRecord>>;

    fn scan_conversations(
        &self,
        query: ConversationQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<ConversationRecord>>;

    /// An entry by id, whatever its conversation.
    fn entry(&self, id: EntryId, ctx: &Context) -> StorageFuture<'_, Option<StoredEntry>>;

    /// The entry only when it is visible from `conversation` through its fork ancestry.
    /// An unknown conversation is a [`StorageError::Request`].
    fn visible_entry(
        &self,
        conversation: ConversationId,
        id: EntryId,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<StoredEntry>>;

    /// The newest visible entry with a `head` at or below `at_or_before`.
    fn find_latest_head_marker(
        &self,
        conversation: ConversationId,
        at_or_before: Option<EntryId>,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<EntryRecord>>;

    /// Entries visible from `query.conversation_id`, through every ancestor up to its cap.
    fn scan_entries(
        &self,
        query: EntryQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<EntryRecord>>;

    fn task(&self, id: TaskId, ctx: &Context) -> StorageFuture<'_, Option<TaskRecord>>;

    fn scan_tasks(
        &self,
        query: TaskQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<TaskRecord>>;

    fn submission(&self, id: SubmissionId, ctx: &Context) -> StorageFuture<'_, Option<SubmissionRecord>>;

    fn scan_submissions(
        &self,
        query: SubmissionQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<SubmissionRecord>>;

    fn submission_by_request(
        &self,
        conversation: ConversationId,
        request_id: &str,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<SubmissionRecord>>;

    /// The incarnation at one exact address alive at `at`; a missing key means the singleton.
    fn find_document(
        &self,
        address: &DocumentAddress,
        at: DocumentPoint,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<DocumentRecord>>;

    /// One incarnation materialized at `at`; `None` when unknown or not alive at `at`. A
    /// historical point on a current-only incarnation is a [`StorageError::Request`].
    fn document(&self, id: DocumentId, at: DocumentPoint, ctx: &Context) -> StorageFuture<'_, Option<StoredDocument>>;

    /// Incarnations alive in one exact scope at the point, ascending by id.
    fn scan_documents(
        &self,
        query: DocumentQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<DocumentRecord>>;

    /// Refuses new calls ([`StorageError::Closed`]), waits for calls underway, then releases
    /// the backend. Closing twice succeeds.
    fn close(&self, ctx: &Context) -> StorageFuture<'_, ()>;
}
