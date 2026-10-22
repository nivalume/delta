//! Document records: incarnations, scopes, conversation policies, points and content.

use std::fmt;

use delta_chord::delta::Op;
use delta_chord::json::Json;
use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize, Serializer};

use super::error::RecordError;
use super::ids::{ConversationId, DocumentId, Seq, TaskId};

/// One incarnation of a document.
///
/// JSON: `{"id":12,"kind":"delta.live","scope":{"kind":"conversation","conversation_id":1},
/// "history":"latest","fork":"initial","created_at":3}`; a family member adds `"key"`,
/// retirement adds `"retired_at"`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentRecord {
    /// One incarnation; never reused.
    pub id: DocumentId,
    pub kind: String,
    /// The family member's key; absent for a singleton.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    pub scope: DocumentScope,
    /// Present iff `scope` is a conversation.
    #[serde(flatten)]
    pub policy: Option<ConversationPolicy>,
    /// Stamped by storage: the commit that created the incarnation.
    pub created_at: Seq,
    /// Stamped by storage: the commit that retired it. The lifetime is
    /// `created_at <= at < retired_at`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retired_at: Option<Seq>,
}

impl DocumentRecord {
    /// The record of `create` created at `created_at`.
    pub fn created(create: DocumentCreate, created_at: Seq) -> Self {
        Self {
            id: create.id,
            kind: create.kind,
            key: create.key,
            scope: create.scope,
            policy: create.policy,
            created_at,
            retired_at: None,
        }
    }

    /// The logical address the incarnation occupies.
    pub fn address(&self) -> DocumentAddress {
        DocumentAddress {
            kind: self.kind.clone(),
            scope: self.scope,
            key: self.key.clone(),
        }
    }

    /// Whether the incarnation exists at `at`: unretired for `Current`, inside
    /// `[created_at, retired_at)` for a sequence.
    pub fn is_alive_at(&self, at: DocumentPoint) -> bool {
        match at {
            DocumentPoint::Current => self.retired_at.is_none(),
            DocumentPoint::At(seq) => self.created_at <= seq && self.retired_at.is_none_or(|retired| seq < retired),
        }
    }

    /// Whether storage keeps every revision: only rewindable conversation documents do.
    /// Session, task and `latest` conversation documents are current-only.
    pub fn keeps_history(&self) -> bool {
        matches!(
            self.policy,
            Some(ConversationPolicy {
                history: History::Rewindable,
                ..
            })
        ) && self.scope.kind() == DocumentScopeKind::Conversation
    }

    /// Refuses a policy that does not match the scope and `as_of` without `rewindable`.
    pub fn check(&self) -> Result<(), RecordError> {
        check_policy(self.id, self.scope, self.policy)
    }
}

fn check_policy(id: DocumentId, scope: DocumentScope, policy: Option<ConversationPolicy>) -> Result<(), RecordError> {
    let problem = match (scope, policy) {
        (DocumentScope::Conversation { .. }, None) => "a conversation document requires history and fork",
        (DocumentScope::Session | DocumentScope::Task { .. }, Some(_)) => {
            "only a conversation document has history and fork"
        }
        (
            _,
            Some(ConversationPolicy {
                history: History::Latest,
                fork: Fork::AsOf,
            }),
        ) => "fork \"as_of\" requires history \"rewindable\"",
        _ => return Ok(()),
    };
    Err(RecordError::Document {
        id,
        problem: problem.to_owned(),
    })
}

/// Who owns a document and so when it dies, tagged by `kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DocumentScope {
    /// Lives as long as the storage.
    Session,
    Conversation {
        conversation_id: ConversationId,
    },
    /// Retired with its task.
    Task {
        task_id: TaskId,
    },
}

impl DocumentScope {
    pub fn kind(self) -> DocumentScopeKind {
        match self {
            Self::Session => DocumentScopeKind::Session,
            Self::Conversation { .. } => DocumentScopeKind::Conversation,
            Self::Task { .. } => DocumentScopeKind::Task,
        }
    }

    /// The owner's id; 0 for the session scope (the SQLite `owner_id` column).
    pub fn owner_id(self) -> u64 {
        match self {
            Self::Session => 0,
            Self::Conversation { conversation_id } => conversation_id.0,
            Self::Task { task_id } => task_id.0,
        }
    }
}

/// The kind of a [`DocumentScope`] without its owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentScopeKind {
    Session,
    Conversation,
    Task,
}

impl DocumentScopeKind {
    /// The serialized name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Session => "session",
            Self::Conversation => "conversation",
            Self::Task => "task",
        }
    }
}

/// What a conversation document keeps and what a fork does with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConversationPolicy {
    pub history: History,
    pub fork: Fork,
}

/// Whether a conversation document keeps its revisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum History {
    /// Current value only; a base reclaims earlier revisions.
    Latest,
    /// Every revision; readable at any commit of its lifetime.
    Rewindable,
}

/// What a fork's child gets of a conversation document.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fork {
    /// A copy of the value at the fork entry's commit; requires [`History::Rewindable`].
    AsOf,
    /// A copy of the value when the fork commits.
    Current,
    /// Nothing; the child starts from the definition's initial value.
    Initial,
}

/// Creation input: a [`DocumentRecord`] without its lifetime stamps.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentCreate {
    pub id: DocumentId,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    pub scope: DocumentScope,
    #[serde(flatten)]
    pub policy: Option<ConversationPolicy>,
}

impl DocumentCreate {
    /// The logical address the incarnation will occupy.
    pub fn address(&self) -> DocumentAddress {
        DocumentAddress {
            kind: self.kind.clone(),
            scope: self.scope,
            key: self.key.clone(),
        }
    }

    /// Refuses a policy that does not match the scope and `as_of` without `rewindable`.
    pub fn check(&self) -> Result<(), RecordError> {
        check_policy(self.id, self.scope, self.policy)
    }
}

/// A document's logical identity: kind and scope, plus the key of a family member. Each
/// incarnation of the address has its own [`DocumentId`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocumentAddress {
    pub kind: String,
    pub scope: DocumentScope,
    /// Absent: the singleton. A member keyed `""` is a different address.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
}

/// A point in a document's history: JSON `"current"` or a commit sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DocumentPoint {
    Current,
    At(Seq),
}

impl Serialize for DocumentPoint {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Current => serializer.serialize_str("current"),
            Self::At(seq) => seq.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for DocumentPoint {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct PointVisitor;

        impl Visitor<'_> for PointVisitor {
            type Value = DocumentPoint;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("\"current\" or a commit sequence")
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<DocumentPoint, E> {
                Ok(DocumentPoint::At(Seq(value)))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<DocumentPoint, E> {
                u64::try_from(value)
                    .map(|value| DocumentPoint::At(Seq(value)))
                    .map_err(|_| E::invalid_value(de::Unexpected::Signed(value), &self))
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<DocumentPoint, E> {
                if value == "current" {
                    Ok(DocumentPoint::Current)
                } else {
                    Err(E::invalid_value(de::Unexpected::Str(value), &self))
                }
            }
        }

        deserializer.deserialize_any(PointVisitor)
    }
}

/// One stored revision of a document, tagged by `kind`. The version is the definition
/// version that wrote it; a delta must have its base's version.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DocumentContent {
    /// The complete value.
    Base { version: u32, value: Json },
    /// An op batch on the previous revision.
    Delta { version: u32, ops: Vec<Op> },
}

impl DocumentContent {
    pub fn version(&self) -> u32 {
        match self {
            Self::Base { version, .. } | Self::Delta { version, .. } => *version,
        }
    }
}

/// The base a new incarnation starts from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BaseContent {
    pub version: u32,
    pub value: Json,
}

impl From<BaseContent> for DocumentContent {
    fn from(base: BaseContent) -> Self {
        Self::Base {
            version: base.version,
            value: base.value,
        }
    }
}

/// A document materialized at one point: the selected base's version, the value after its
/// delta tail, and the tail's length.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredDocument {
    pub record: DocumentRecord,
    pub version: u32,
    pub value: Json,
    /// Deltas after the newest base at the point.
    pub deltas_since_base: u64,
}

/// What a fork copy reads: one incarnation at one point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentCopySource {
    pub id: DocumentId,
    pub at: DocumentPoint,
}
