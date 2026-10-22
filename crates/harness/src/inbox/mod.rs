//! Submissions and the inbox (spec §6): host input and passive entry writes, admitted
//! durably in one commit, queued in `delta.inbox` while the conversation is busy, and placed
//! at turn boundaries where they change the model's context predictably.
//!
//! Corresponds to upstream `src/harness/inbox.ts` and `src/harness/submissions.ts` (the
//! admission half; waits and handles are the surface's). See `docs/design/inbox.md`.

mod admission;
mod boundary;
mod error;

use delta_ai::types::{Message, UserContent};
use delta_chord::delta::Op;
use delta_chord::json::{self, Json};
use serde::{Deserialize, Deserializer, Serialize};

pub use admission::{
    Admission, AdmissionStatus, admit_submission, append_input, remove_inbox_item, withdraw_inputs,
    withdraw_queued_inputs,
};
pub use boundary::{
    At, Boundary, BoundaryResult, apply_boundary, has_queued_reset, is_display_only, is_stale, prepare_boundary,
};
pub use error::SubmitError;

use crate::agent::{QueueMode, Settings};
use crate::documents::{CheckpointInfo, ConversationScope, DocAddress, DocDefinition, ForkInitial, Latest};
use crate::records::{ContextEdit, ConversationId, EntryDraft, EntryId, Head, SubmissionId};
use crate::session::{Tx, TxError};

/// The entry kind of placed user input.
pub const USER_ENTRY: &str = "delta.user";
/// The entry kind of a reset: a head marker at itself, with the handoff as a user message.
pub const RESET_ENTRY: &str = "delta.reset";

/// The `delta.inbox` document: conversation scope, `latest`, a fork starts empty, a complete
/// base whenever `items` is empty.
#[derive(Clone, Copy, Debug, Default)]
pub struct InboxDoc;

impl DocDefinition for InboxDoc {
    const KIND: &'static str = "delta.inbox";
    const VERSION: u32 = 1;
    type Scope = ConversationScope<Latest, ForkInitial>;
    type Seed = ();

    fn initial((): &()) -> Json {
        delta_chord::json!({ "items": [] })
    }

    fn checkpoint_when(value: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        value
            .get("items")
            .and_then(Json::as_array)
            .is_some_and(|items| items.is_empty())
    }
}

/// What a busy conversation does with new input.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WhenBusy {
    /// Placed at the next boundary, between a tool round and the next request.
    Steer,
    /// Placed after the run's answer.
    #[default]
    FollowUp,
    /// Refused with `ConversationBusy`; nothing is written.
    Reject,
}

/// The entry an input is placed as when not `delta.user`: its kind and data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputEntry {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Json>,
}

/// What a host submits.
#[derive(Clone, Debug, PartialEq)]
pub enum SubmissionDraft {
    /// User input for a run.
    Input {
        content: UserContent,
        when_busy: WhenBusy,
        entry: Option<InputEntry>,
        request_id: Option<String>,
    },
    /// A passive entry write.
    Write {
        entry: EntryDraft,
        request_id: Option<String>,
    },
}

impl SubmissionDraft {
    /// Input `content` with the defaults: a follow-up placed as `delta.user`.
    pub fn input(content: impl Into<UserContent>) -> Self {
        Self::Input {
            content: content.into(),
            when_busy: WhenBusy::FollowUp,
            entry: None,
            request_id: None,
        }
    }

    /// A write of `entry`.
    pub fn write(entry: EntryDraft) -> Self {
        Self::Write {
            entry,
            request_id: None,
        }
    }

    /// The same draft with `when_busy` (inputs only).
    #[must_use]
    pub fn when_busy(mut self, mode: WhenBusy) -> Self {
        if let Self::Input { when_busy, .. } = &mut self {
            *when_busy = mode;
        }
        self
    }

    /// The same draft with the host deduplication key `id`.
    #[must_use]
    pub fn request_id(mut self, id: impl Into<String>) -> Self {
        match &mut self {
            Self::Input { request_id, .. } | Self::Write { request_id, .. } => *request_id = Some(id.into()),
        }
        self
    }

    fn request(&self) -> Option<&str> {
        match self {
            Self::Input { request_id, .. } | Self::Write { request_id, .. } => request_id.as_deref(),
        }
    }
}

/// One queued submission, in ID order: `{"mode":"follow_up","id":7,"content":...}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum InboxItem {
    Steer {
        id: SubmissionId,
        content: UserContent,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        entry: Option<InputEntry>,
    },
    FollowUp {
        id: SubmissionId,
        content: UserContent,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        entry: Option<InputEntry>,
    },
    /// The write's `EntryDraft` as plain JSON.
    Write { id: SubmissionId, entry: Json },
}

impl InboxItem {
    pub fn id(&self) -> SubmissionId {
        match self {
            Self::Steer { id, .. } | Self::FollowUp { id, .. } | Self::Write { id, .. } => *id,
        }
    }

    /// Whether the item is user input (a steer or a follow-up).
    pub fn is_input(&self) -> bool {
        !matches!(self, Self::Write { .. })
    }
}

/// The value of a `delta.inbox` document.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InboxState {
    #[serde(default)]
    pub items: Vec<InboxItem>,
}

/// The queue modes a boundary reads, from the settings resolved on the line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueueModes {
    pub steering: QueueMode,
    pub follow_up: QueueMode,
}

impl From<&Settings> for QueueModes {
    fn from(settings: &Settings) -> Self {
        Self {
            steering: settings.steering_mode,
            follow_up: settings.follow_up_mode,
        }
    }
}

/// The JSON form of an [`EntryDraft`] in the inbox: `head` is an entry id or `"self"`.
#[derive(Serialize, Deserialize)]
struct WireDraft {
    kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<Vec<Message>>,
    #[serde(default, deserialize_with = "present", skip_serializing_if = "Option::is_none")]
    data: Option<Json>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    head: Option<WireHead>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    edits: Option<Vec<ContextEdit>>,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum WireHead {
    At(EntryId),
    SelfRef(SelfTag),
}

#[derive(Serialize, Deserialize)]
enum SelfTag {
    #[serde(rename = "self")]
    SelfRef,
}

/// A present `null` stays a value.
fn present<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Json>, D::Error> {
    Json::deserialize(deserializer).map(Some)
}

/// The inbox JSON of a write's draft.
pub(crate) fn draft_to_json(draft: &EntryDraft) -> Result<Json, serde_json::Error> {
    let wire = WireDraft {
        kind: draft.kind.clone(),
        model: draft.model.clone(),
        data: draft.data.clone(),
        head: draft.head.map(|head| match head {
            Head::At(entry) => WireHead::At(entry),
            Head::SelfRef => WireHead::SelfRef(SelfTag::SelfRef),
        }),
        edits: draft.edits.clone(),
    };
    json::to_json(&wire)
}

/// The draft of a queued write.
pub(crate) fn draft_from_json(value: &Json) -> Result<EntryDraft, serde_json::Error> {
    let wire: WireDraft = json::from_json(value)?;
    Ok(EntryDraft {
        kind: wire.kind,
        model: wire.model,
        data: wire.data,
        head: wire.head.map(|head| match head {
            WireHead::At(entry) => Head::At(entry),
            WireHead::SelfRef(SelfTag::SelfRef) => Head::SelfRef,
        }),
        edits: wire.edits,
    })
}

/// The queued items of `conversation`.
pub(crate) async fn inbox_items(tx: &mut Tx<'_>, conversation: ConversationId) -> Result<Vec<InboxItem>, TxError> {
    let doc = tx.doc(DocAddress::<InboxDoc>::new(conversation)).await?;
    let state: InboxState = json::from_json(doc.value()).map_err(TxError::callback)?;
    Ok(state.items)
}

#[cfg(test)]
mod tests;
