//! Approvals: the policy that decides, per tool call, whether it runs, is denied or must be
//! approved, and the durable records of a request and its decision.
//!
//! codex-rs's flow made durable. When the policy asks, the tool task commits a request into
//! the conversation's `delta.approvals` with its deadline, then waits in its `approval` phase
//! for the decision: a oneshot from [`PendingApprovals`], or the committed decision on a
//! rerun. `Conversation::decide` commits the decision (and a display-only `delta.approval`
//! entry, and a session grant for `ApprovedForSession`) before it notifies the waiter, so
//! a crash at any point leaves one durable outcome. See `docs/design/approval.md`.

mod document;
mod pending;
mod policy;
pub mod rules;

pub use document::{
    ApprovalGrant, ApprovalId, ApprovalRequest, ApprovalsDoc, ApprovalsState, Recorded, ReviewDecision,
    approvals_state, edit_approvals, record_decision, remove_request,
};
pub use pending::{ApprovalWaiter, PendingApprovals, RunAbort};
pub use policy::{AllowAll, ApprovalCall, ApprovalPolicy, AskAll, DenyAll, PolicyDecision, PolicyError};

/// The entry kind of a decision: display-only, the audit record observers see.
pub const APPROVAL_ENTRY: &str = "delta.approval";

/// Why `decide` refused a decision; nothing was written.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ApprovalError {
    /// No tool task with this id exists.
    #[error("Approval {0} does not exist")]
    UnknownApproval(ApprovalId),
    /// The request is not waiting for a decision: decided already, timed out, withdrawn or
    /// never made.
    #[error("Approval {0} is not pending")]
    NotPending(ApprovalId),
    /// `TimedOut` is the harness's own decision.
    #[error("A caller cannot decide {0}: only the harness times out")]
    InvalidDecision(&'static str),
}

impl ApprovalError {
    /// As a commit closure's error, so the caller's error type can take it back.
    pub fn into_tx(self) -> crate::session::TxError {
        crate::session::TxError::callback(self)
    }
}
