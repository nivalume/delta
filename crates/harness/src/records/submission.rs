//! Submission records: host inputs and passive writes with their lifecycle states.

use delta_chord::json::Json;
use serde::{Deserialize, Serialize};

use super::error::RecordError;
use super::ids::{ConversationId, EntryId, SubmissionId};

/// A host submission.
///
/// JSON: `{"id":9,"conversation_id":1,"request_id":"r1","type":"input","status":"placed","entry":12}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubmissionRecord {
    pub id: SubmissionId,
    pub conversation_id: ConversationId,
    /// Host deduplication key, unique per conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(rename = "type")]
    pub kind: SubmissionType,
    #[serde(flatten)]
    pub state: SubmissionState,
}

/// Whether a submission asks for an answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubmissionType {
    /// Placed as an entry, then answered by a run.
    Input,
    /// Written as an entry; nothing answers it.
    Write,
}

/// A submission's lifecycle state, tagged by `status`.
///
/// | type | legal states |
/// | --- | --- |
/// | `input` | `queued` → `placed{entry}` → `done{entry, answer}` or `unanswered{entry?, reason, detail?}` |
/// | `write` | `queued` → `done{entry}` or `unanswered{reason, detail?}` |
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum SubmissionState {
    Queued,
    /// Input only: the entry is in the conversation, the answer is pending.
    Placed {
        entry: EntryId,
    },
    /// `answer` is present for an input and absent for a write.
    Done {
        entry: EntryId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        answer: Option<EntryId>,
    },
    /// `entry` may be present for an input only.
    Unanswered {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        entry: Option<EntryId>,
        reason: String,
        #[serde(
            default,
            deserialize_with = "super::present",
            skip_serializing_if = "Option::is_none"
        )]
        detail: Option<Json>,
    },
}

impl SubmissionState {
    pub fn status(&self) -> SubmissionStatus {
        match self {
            Self::Queued => SubmissionStatus::Queued,
            Self::Placed { .. } => SubmissionStatus::Placed,
            Self::Done { .. } => SubmissionStatus::Done,
            Self::Unanswered { .. } => SubmissionStatus::Unanswered,
        }
    }
}

/// The status alone: query filters and the SQLite `status` column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubmissionStatus {
    Queued,
    Placed,
    Done,
    Unanswered,
}

impl SubmissionStatus {
    /// The serialized name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Placed => "placed",
            Self::Done => "done",
            Self::Unanswered => "unanswered",
        }
    }
}

/// How a submission ends.
#[derive(Clone, Debug, PartialEq)]
pub enum SubmissionSettlement {
    Done { answer: EntryId },
    Unanswered { reason: String, detail: Option<Json> },
}

/// Creation input: a [`SubmissionRecord`] without its id.
#[derive(Clone, Debug, PartialEq)]
pub struct SubmissionCreate {
    pub conversation_id: ConversationId,
    pub request_id: Option<String>,
    pub kind: SubmissionType,
    pub state: SubmissionState,
}

impl SubmissionCreate {
    /// The record with id `id`.
    pub fn into_record(self, id: SubmissionId) -> SubmissionRecord {
        SubmissionRecord {
            id,
            conversation_id: self.conversation_id,
            request_id: self.request_id,
            kind: self.kind,
            state: self.state,
        }
    }
}

impl SubmissionRecord {
    /// Refuses the state combinations the type table forbids: a placed write, a write with
    /// an answer, an unanswered write with an entry, a done input without an answer.
    pub fn check(&self) -> Result<(), RecordError> {
        let problem = match (self.kind, &self.state) {
            (SubmissionType::Write, SubmissionState::Placed { .. }) => "a write submission cannot be placed",
            (SubmissionType::Write, SubmissionState::Done { answer: Some(_), .. }) => {
                "a write submission has no answer"
            }
            (SubmissionType::Write, SubmissionState::Unanswered { entry: Some(_), .. }) => {
                "an unanswered write submission has no entry"
            }
            (SubmissionType::Input, SubmissionState::Done { answer: None, .. }) => {
                "a done input submission requires an answer"
            }
            _ => return Ok(()),
        };
        Err(RecordError::Submission {
            id: self.id,
            problem: problem.to_owned(),
        })
    }
}
