//! Scan queries, pages and cursors.

use delta_chord::json::Json;
use serde::{Deserialize, Serialize};

use super::document::{DocumentPoint, DocumentScope};
use super::ids::{ConversationId, EntryId, TaskId};
use super::submission::SubmissionStatus;
use super::task::TaskStatus;

/// The id order of a scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScanOrder {
    Ascending,
    Descending,
}

impl ScanOrder {
    /// The serialized name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ascending => "ascending",
            Self::Descending => "descending",
        }
    }
}

/// One page of a scan; `next` is present only when more items remain.
#[derive(Clone, Debug, PartialEq)]
pub struct Page<T> {
    pub items: Vec<T>,
    pub next: Option<Cursor>,
}

/// A backend-owned scan continuation, round-tripped unread by callers. The built-in
/// backends use `{"after": <id>, "order": "ascending" | "descending"}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Cursor(pub Json);

/// Conversations by owner edge; the filters are conjunctive. Default order ascending.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConversationQuery {
    pub owner_conversation_id: Option<ConversationId>,
    pub owner_task_id: Option<TaskId>,
    pub order: Option<ScanOrder>,
}

/// Entries visible from a conversation, through its fork ancestry. Bounds are inclusive.
/// Default order descending (newest first).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryQuery {
    pub conversation_id: ConversationId,
    pub min_entry_id: Option<EntryId>,
    pub max_entry_id: Option<EntryId>,
    pub order: Option<ScanOrder>,
}

impl EntryQuery {
    /// Every visible entry of `conversation_id`, newest first.
    pub fn new(conversation_id: ConversationId) -> Self {
        Self {
            conversation_id,
            min_entry_id: None,
            max_entry_id: None,
            order: None,
        }
    }
}

/// Tasks by conjunctive filters. Default order ascending.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TaskQuery {
    pub conversation_id: Option<ConversationId>,
    pub kind: Option<String>,
    pub status: Option<TaskStatus>,
    pub abort_requested: Option<bool>,
    pub background: Option<bool>,
    pub order: Option<ScanOrder>,
}

/// Submissions by conjunctive filters. Default order ascending.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SubmissionQuery {
    pub conversation_id: Option<ConversationId>,
    pub status: Option<SubmissionStatus>,
    pub order: Option<ScanOrder>,
}

/// Incarnations alive in one exact scope at one point, optionally of one kind. Always
/// ascending by incarnation id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentQuery {
    pub scope: DocumentScope,
    pub at: DocumentPoint,
    pub kind: Option<String>,
}

impl DocumentQuery {
    /// Every current incarnation of `scope`.
    pub fn current(scope: DocumentScope) -> Self {
        Self {
            scope,
            at: DocumentPoint::Current,
            kind: None,
        }
    }
}
