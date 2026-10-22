//! Storage writes: the commands of one atomic commit batch.

use serde::{Deserialize, Serialize};

use super::conversation::ConversationRecord;
use super::document::{BaseContent, DocumentContent, DocumentCopySource, DocumentCreate};
use super::entry::EntryRecord;
use super::ids::DocumentId;
use super::submission::SubmissionRecord;
use super::task::TaskRecord;

/// One command of a commit batch.
///
/// Conversations, entries and document incarnations are created once; tasks and submissions
/// are replaced whole. The serde form (`{"type":"document_create","value":{...}}`) is what a
/// later JSONL backend writes as its log lines; memory and SQLite do not persist it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum StorageWrite {
    Conversation(ConversationRecord),
    Entry(EntryRecord),
    Task(TaskRecord),
    Submission(SubmissionRecord),
    /// A new incarnation with its first base.
    DocumentCreate {
        record: DocumentCreate,
        content: BaseContent,
    },
    /// A new incarnation whose first base is the source's value and version.
    DocumentCopy {
        record: DocumentCreate,
        source: DocumentCopySource,
    },
    /// A base or delta on an existing incarnation.
    DocumentChange {
        id: DocumentId,
        content: DocumentContent,
    },
    /// Ends an incarnation's lifetime at this commit.
    DocumentRetire {
        id: DocumentId,
    },
}
