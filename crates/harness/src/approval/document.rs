//! `delta.approvals`: a conversation's pending approval requests and session grants, and the
//! commit that records a decision.

use std::fmt;

use delta_ai::types::JsonObject;
use delta_chord::delta::Op;
use delta_chord::json::{self, Json};
use serde::{Deserialize, Serialize};

use super::APPROVAL_ENTRY;
use crate::documents::{CheckpointInfo, ConversationScope, DocAddress, DocDefinition, ForkInitial, Latest};
use crate::inbox::{QueueModes, SubmissionDraft, SubmitError, admit_submission};
use crate::records::{ConversationId, EntryDraft, TaskId};
use crate::session::{Tx, TxError};

/// An approval request's id: the id of the tool task that asks, which asks at most once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ApprovalId(pub TaskId);

impl fmt::Display for ApprovalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// The answer to a request (Codex's `ReviewDecision`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReviewDecision {
    Approved,
    /// Approved, and later calls with the same tool and approval key are too.
    ApprovedForSession,
    Denied {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rejection: Option<String>,
    },
    /// Written by the harness when the deadline passes; refused from callers.
    TimedOut,
    /// Aborts the conversation's run.
    Abort,
}

/// One pending request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub id: ApprovalId,
    pub call_id: String,
    pub name: String,
    /// The final arguments, exactly as the call runs when approved.
    pub arguments: JsonObject,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    pub requested_at: i64,
    /// Harness clock, epoch ms; also in the task's checkpoint.
    pub deadline: i64,
    /// Set by the deciding commit; the task consumes it and removes the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision: Option<ReviewDecision>,
}

/// A session grant: calls of `tool` whose approval key is `key` are approved without asking.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalGrant {
    pub tool: String,
    pub key: String,
    pub granted_by: ApprovalId,
}

/// The value of a `delta.approvals` document.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ApprovalsState {
    /// In id order.
    #[serde(default)]
    pub requests: Vec<ApprovalRequest>,
    /// In grant order.
    #[serde(default)]
    pub grants: Vec<ApprovalGrant>,
}

impl ApprovalsState {
    pub fn request(&self, id: ApprovalId) -> Option<&ApprovalRequest> {
        self.requests.iter().find(|request| request.id == id)
    }

    /// Whether a grant covers `tool` with `key`.
    pub fn granted(&self, tool: &str, key: &str) -> bool {
        self.grants.iter().any(|grant| grant.tool == tool && grant.key == key)
    }
}

/// The `delta.approvals` document: conversation scope, latest, a fork starts without
/// requests or grants, a complete base whenever no request is pending.
#[derive(Clone, Copy, Debug, Default)]
pub struct ApprovalsDoc;

impl DocDefinition for ApprovalsDoc {
    const KIND: &'static str = "delta.approvals";
    const VERSION: u32 = 1;
    type Scope = ConversationScope<Latest, ForkInitial>;
    type Seed = ();

    fn initial((): &()) -> Json {
        delta_chord::json!({ "requests": [], "grants": [] })
    }

    fn checkpoint_when(value: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        value
            .get("requests")
            .and_then(Json::as_array)
            .is_none_or(|requests| requests.is_empty())
    }
}

/// `conversation`'s `delta.approvals`, as of this transaction.
pub async fn approvals_state(tx: &mut Tx<'_>, conversation: ConversationId) -> Result<ApprovalsState, TxError> {
    let doc = tx.doc(DocAddress::<ApprovalsDoc>::new(conversation)).await?;
    json::from_json(doc.value()).map_err(TxError::callback)
}

/// Edits `conversation`'s `delta.approvals` as a typed state.
pub async fn edit_approvals<R>(
    tx: &mut Tx<'_>,
    conversation: ConversationId,
    edit: impl FnOnce(&mut ApprovalsState) -> R,
) -> Result<R, TxError> {
    let mut doc = tx.doc(DocAddress::<ApprovalsDoc>::new(conversation)).await?;
    doc.update_as::<ApprovalsState, R>(edit).map_err(TxError::callback)
}

/// Removes request `id`.
pub async fn remove_request(tx: &mut Tx<'_>, conversation: ConversationId, id: ApprovalId) -> Result<(), TxError> {
    edit_approvals(tx, conversation, |state| {
        state.requests.retain(|request| request.id != id);
    })
    .await
}

/// What [`record_decision`] found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Recorded {
    /// The decision was written.
    Written,
    /// The request already had this decision.
    Already(ReviewDecision),
    /// There is no such request.
    Missing,
}

/// Writes `decision` into the pending request `id`: the request's `decision`, a grant for
/// `ApprovedForSession` when the request has an approval key, and the display-only
/// `delta.approval` entry, admitted once under request id `approval:<id>`. Admission comes
/// first, before any table write of the commit.
pub async fn record_decision(
    tx: &mut Tx<'_>,
    conversation: ConversationId,
    id: ApprovalId,
    decision: ReviewDecision,
    now: i64,
    modes: QueueModes,
) -> Result<Recorded, TxError> {
    let state = approvals_state(tx, conversation).await?;
    let Some(request) = state.request(id) else {
        return Ok(Recorded::Missing);
    };
    if let Some(decided) = &request.decision {
        return Ok(Recorded::Already(decided.clone()));
    }
    let data = delta_chord::json!({
        "approval_id": json::to_json(&id).map_err(TxError::callback)?,
        "call_id": request.call_id.clone(),
        "name": request.name.clone(),
        "decision": json::to_json(&decision).map_err(TxError::callback)?,
    });
    let entry = EntryDraft {
        data: Some(data),
        ..EntryDraft::new(APPROVAL_ENTRY)
    };
    let draft = SubmissionDraft::write(entry).request_id(format!("approval:{id}"));
    admit_submission(tx, conversation, draft, now, modes)
        .await
        .map_err(SubmitError::into_tx)?;
    let grant = match (&decision, &request.approval_key) {
        (ReviewDecision::ApprovedForSession, Some(key)) => Some(ApprovalGrant {
            tool: request.name.clone(),
            key: key.clone(),
            granted_by: id,
        }),
        _ => None,
    };
    edit_approvals(tx, conversation, |state| {
        if let Some(request) = state.requests.iter_mut().find(|request| request.id == id) {
            request.decision = Some(decision);
        }
        if let Some(grant) = grant
            && !state.granted(&grant.tool, &grant.key)
        {
            state.grants.push(grant);
        }
    })
    .await?;
    Ok(Recorded::Written)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decisions_and_states_have_their_json_shape() {
        let denied = ReviewDecision::Denied {
            rejection: Some("no".to_owned()),
        };
        assert_eq!(
            serde_json::to_value(&denied).unwrap(),
            serde_json::json!({ "type": "denied", "rejection": "no" })
        );
        assert_eq!(
            serde_json::to_value(ReviewDecision::ApprovedForSession).unwrap(),
            serde_json::json!({ "type": "approved_for_session" })
        );
        let state: ApprovalsState = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(state, ApprovalsState::default());
        let grant = ApprovalGrant {
            tool: "bash".to_owned(),
            key: "ls".to_owned(),
            granted_by: ApprovalId(TaskId(4)),
        };
        assert_eq!(
            serde_json::to_value(&grant).unwrap(),
            serde_json::json!({ "tool": "bash", "key": "ls", "granted_by": 4 })
        );
    }
}
