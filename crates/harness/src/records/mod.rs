//! The persisted record model: ids, conversations, entries, submissions, tasks, document
//! incarnations, storage writes and scan queries.
//!
//! Records are values with serde-native, snake_case JSON; status unions are tagged enums
//! and `Option` stands for an absent field. Corresponds to upstream `src/types.ts` and
//! `src/ids.ts`. See `docs/design/records.md`.

mod conversation;
mod document;
mod entry;
mod error;
mod ids;
mod query;
mod submission;
mod task;
mod write;

pub use conversation::{ConversationOwner, ConversationOwnership, ConversationParent, ConversationRecord};
pub use document::{
    BaseContent, ConversationPolicy, DocumentAddress, DocumentContent, DocumentCopySource, DocumentCreate,
    DocumentPoint, DocumentRecord, DocumentScope, DocumentScopeKind, Fork, History, StoredDocument,
};
pub use entry::{
    ContextEdit, EditAction, EntryDraft, EntryKind, EntryRecord, Head, StoredEntry, TypedEntry, TypedEntryDraft,
};
pub use error::RecordError;
pub use ids::{ConversationId, DocumentId, EntryId, ROOT_CONVERSATION_ID, RawId, RecordId, Seq, SubmissionId, TaskId};
pub use query::{ConversationQuery, Cursor, DocumentQuery, EntryQuery, Page, ScanOrder, SubmissionQuery, TaskQuery};
pub use submission::{
    SubmissionCreate, SubmissionRecord, SubmissionSettlement, SubmissionState, SubmissionStatus, SubmissionType,
};
pub use task::{AbortReason, JoinPolicy, TaskOutcome, TaskOutcomeError, TaskRecord, TaskState, TaskStatus};
pub use write::StorageWrite;

use serde::{Deserialize, Deserializer};

/// Deserializes a present field as `Some`, `null` included, so `null` stays a value in
/// `data`, `detail`, `result` and `trace`; an absent field is `None` through `default`.
fn present<'de, D: Deserializer<'de>, T: Deserialize<'de>>(deserializer: D) -> Result<Option<T>, D::Error> {
    T::deserialize(deserializer).map(Some)
}

#[cfg(test)]
mod tests;
