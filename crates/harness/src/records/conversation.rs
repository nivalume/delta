//! Conversation records: identity, fork parent and owning task.

use serde::{Deserialize, Serialize};

use super::ids::{ConversationId, EntryId, TaskId};

/// A conversation. Created once and never changed.
///
/// JSON: `{"id":1}`;
/// `{"id":7,"parent":{"conversation_id":1,"at":42},"owner":{"conversation_id":1,"task_id":40}}`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationRecord {
    pub id: ConversationId,
    /// History: the conversation sees its parent's entries up to `parent.at`, and fork copies
    /// of the parent's documents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<ConversationParent>,
    /// The task that created the conversation: task attribution, abort and idle traversal.
    /// Not access control. Stays recorded after the task is terminal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<ConversationOwner>,
}

impl ConversationRecord {
    /// A conversation without parent or owner.
    pub fn new(id: ConversationId) -> Self {
        Self {
            id,
            parent: None,
            owner: None,
        }
    }
}

/// Where a fork starts: `at` is an entry visible in `conversation_id`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConversationParent {
    pub conversation_id: ConversationId,
    pub at: EntryId,
}

/// The persisted owner edge: the owning task and its conversation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConversationOwner {
    pub conversation_id: ConversationId,
    pub task_id: TaskId,
}

/// Creation input. The Session derives the persisted [`ConversationOwner`] from the task;
/// callers never build it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConversationOwnership {
    #[default]
    Ownerless,
    Task(TaskId),
}
