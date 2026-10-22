//! Entry records: immutable conversation items, their context edits and typed entry kinds.

use std::fmt;
use std::marker::PhantomData;

use delta_ai::types::Message;
use delta_chord::json::{self, Json};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::error::RecordError;
use super::ids::{ConversationId, EntryId, Seq, TaskId};

/// An immutable entry of a conversation.
///
/// JSON: `{"id":5,"conversation_id":1,"kind":"delta.user","model":[...],"data":{...}}`; the
/// messages in `model` keep `delta_ai`'s (upstream's camelCase) shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntryRecord {
    pub id: EntryId,
    pub conversation_id: ConversationId,
    /// What the entry is; `delta.*` kinds belong to the crate's own entries by convention.
    pub kind: String,
    /// The model messages the entry contributes; absent: none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<Vec<Message>>,
    /// Application data; `null` is a value here.
    #[serde(
        default,
        deserialize_with = "super::present",
        skip_serializing_if = "Option::is_none"
    )]
    pub data: Option<Json>,
    /// Head marker: later context starts at this entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head: Option<EntryId>,
    /// Edits of earlier entries' contribution to later context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edits: Option<Vec<ContextEdit>>,
    /// The task whose runtime commit appended the entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by_task_id: Option<TaskId>,
}

impl EntryRecord {
    /// An entry with only its identity and kind.
    pub fn new(id: EntryId, conversation_id: ConversationId, kind: impl Into<String>) -> Self {
        Self {
            id,
            conversation_id,
            kind: kind.into(),
            model: None,
            data: None,
            head: None,
            edits: None,
            by_task_id: None,
        }
    }
}

/// An entry read from storage with the sequence of the commit that wrote it.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredEntry {
    pub entry: EntryRecord,
    pub commit_seq: Seq,
}

/// One edit of later context: `{"target":5,"action":"omit"}` or
/// `{"target":5,"action":"replace","messages":[...]}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RawEdit", into = "RawEdit")]
pub struct ContextEdit {
    pub target: EntryId,
    pub action: EditAction,
}

/// What a [`ContextEdit`] does to its target's messages.
#[derive(Clone, Debug, PartialEq)]
pub enum EditAction {
    /// The target contributes nothing.
    Omit,
    /// The target contributes these messages instead.
    Replace(Vec<Message>),
}

/// The flat wire form of a [`ContextEdit`]: the action tag and its payload beside `target`.
#[derive(Serialize, Deserialize)]
struct RawEdit {
    target: EntryId,
    action: RawAction,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    messages: Option<Vec<Message>>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RawAction {
    Omit,
    Replace,
}

impl TryFrom<RawEdit> for ContextEdit {
    type Error = String;

    fn try_from(raw: RawEdit) -> Result<Self, String> {
        let action = match (raw.action, raw.messages) {
            (RawAction::Omit, None) => EditAction::Omit,
            (RawAction::Replace, Some(messages)) => EditAction::Replace(messages),
            (RawAction::Omit, Some(_)) => return Err("an \"omit\" edit carries no messages".to_owned()),
            (RawAction::Replace, None) => return Err("a \"replace\" edit requires messages".to_owned()),
        };
        Ok(Self {
            target: raw.target,
            action,
        })
    }
}

impl From<ContextEdit> for RawEdit {
    fn from(edit: ContextEdit) -> Self {
        let (action, messages) = match edit.action {
            EditAction::Omit => (RawAction::Omit, None),
            EditAction::Replace(messages) => (RawAction::Replace, Some(messages)),
        };
        Self {
            target: edit.target,
            action,
            messages,
        }
    }
}

/// Where a new entry's head marker points (upstream `head: EntryId | "self"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Head {
    /// An earlier entry.
    At(EntryId),
    /// The new entry itself, once its id is minted.
    SelfRef,
}

/// An entry to append; the Session assigns the id, conversation and task stamp.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EntryDraft {
    pub kind: String,
    pub model: Option<Vec<Message>>,
    pub data: Option<Json>,
    pub head: Option<Head>,
    pub edits: Option<Vec<ContextEdit>>,
}

impl EntryDraft {
    /// A draft of `kind` with nothing else set.
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            ..Self::default()
        }
    }
}

/// A typed entry kind: the `kind` string plus the type of `data` (upstream `Entry<D>`).
pub struct EntryKind<D> {
    pub kind: &'static str,
    _data: PhantomData<fn() -> D>,
}

impl<D> EntryKind<D> {
    /// The kind named `kind` whose data is a `D`.
    pub const fn new(kind: &'static str) -> Self {
        Self {
            kind,
            _data: PhantomData,
        }
    }
}

impl<D> Clone for EntryKind<D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D> Copy for EntryKind<D> {}

impl<D> fmt::Debug for EntryKind<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("EntryKind").field(&self.kind).finish()
    }
}

impl<D: DeserializeOwned> EntryKind<D> {
    /// The record with its decoded data, or `None` when the record is of another kind.
    /// A record of this kind without `data` decodes from `null`.
    pub fn decode(&self, record: EntryRecord) -> Result<Option<TypedEntry<D>>, RecordError> {
        if record.kind != self.kind {
            return Ok(None);
        }
        let data =
            json::from_json(record.data.as_ref().unwrap_or(&Json::Null)).map_err(|source| RecordError::EntryData {
                kind: self.kind.to_owned(),
                source,
            })?;
        Ok(Some(TypedEntry { record, data }))
    }
}

impl<D: Serialize> EntryKind<D> {
    /// The untyped draft of a typed one.
    pub fn encode(&self, draft: TypedEntryDraft<D>) -> Result<EntryDraft, RecordError> {
        let data = json::to_json(&draft.data).map_err(|source| RecordError::EntryData {
            kind: self.kind.to_owned(),
            source,
        })?;
        Ok(EntryDraft {
            kind: self.kind.to_owned(),
            model: draft.model,
            data: Some(data),
            head: draft.head,
            edits: draft.edits,
        })
    }
}

/// An entry whose data was decoded by an [`EntryKind`].
#[derive(Clone, Debug, PartialEq)]
pub struct TypedEntry<D> {
    pub record: EntryRecord,
    pub data: D,
}

/// An [`EntryDraft`] without `kind`, with typed `data`.
#[derive(Clone, Debug, PartialEq)]
pub struct TypedEntryDraft<D> {
    pub data: D,
    pub model: Option<Vec<Message>>,
    pub head: Option<Head>,
    pub edits: Option<Vec<ContextEdit>>,
}

impl<D> TypedEntryDraft<D> {
    /// A draft with `data` and nothing else.
    pub fn new(data: D) -> Self {
        Self {
            data,
            model: None,
            head: None,
            edits: None,
        }
    }
}
