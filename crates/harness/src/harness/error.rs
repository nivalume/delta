//! The surface's errors.

use delta_chord::context::Cancelled;

use crate::agent::AgentError;
use crate::approval::ApprovalError;
use crate::context::ContextError;
use crate::inbox::SubmitError;
use crate::records::{ConversationId, SubmissionId, TaskId};
use crate::scheduler::SchedulerError;
use crate::session::{SessionError, TxError};
use crate::storage::StorageError;

/// Why a Harness operation failed.
#[derive(Debug, thiserror::Error)]
pub enum HarnessError {
    /// The Harness is closing or closed.
    #[error("The harness is closed")]
    Closed,
    #[error("Conversation {0} does not exist")]
    UnknownConversation(ConversationId),
    #[error("Submission {0} does not exist")]
    UnknownSubmission(SubmissionId),
    #[error("Task {0} does not exist")]
    UnknownTask(TaskId),
    /// Admission refused the submission (`ConversationBusy`, an invalid entry kind, a
    /// request id of another type); nothing was written.
    #[error(transparent)]
    Submit(SubmitError),
    /// `decide` refused the decision; nothing was written.
    #[error(transparent)]
    Approval(ApprovalError),
    /// The caller's context was cancelled; shared work is unaffected.
    #[error(transparent)]
    Cancelled(Cancelled),
    /// A built-in document does not convert to its typed state.
    #[error("A stored document does not convert: {0}")]
    Document(serde_json::Error),
    #[error(transparent)]
    Agent(AgentError),
    #[error(transparent)]
    Context(ContextError),
    #[error(transparent)]
    Storage(StorageError),
    #[error(transparent)]
    Scheduler(SchedulerError),
    #[error(transparent)]
    Session(SessionError),
}

impl From<SessionError> for HarnessError {
    fn from(error: SessionError) -> Self {
        match error {
            SessionError::Closed => Self::Closed,
            SessionError::Cancelled(cancelled) => Self::Cancelled(cancelled),
            SessionError::Tx(TxError::UnknownConversation(id)) => Self::UnknownConversation(id),
            SessionError::Tx(TxError::UnknownSubmission(id)) => Self::UnknownSubmission(id),
            SessionError::Tx(TxError::UnknownTask(id)) => Self::UnknownTask(id),
            SessionError::Tx(TxError::Callback(source)) => match source.downcast::<SubmitError>() {
                Ok(refusal) => Self::Submit(*refusal),
                Err(source) => match source.downcast::<ApprovalError>() {
                    Ok(refusal) => Self::Approval(*refusal),
                    Err(source) => Self::Session(SessionError::Tx(TxError::Callback(source))),
                },
            },
            other => Self::Session(other),
        }
    }
}

impl From<SchedulerError> for HarnessError {
    fn from(error: SchedulerError) -> Self {
        match error {
            SchedulerError::Closed => Self::Closed,
            SchedulerError::UnknownTask(id) => Self::UnknownTask(id),
            SchedulerError::Cancelled(cancelled) => Self::Cancelled(cancelled),
            SchedulerError::Session(error) => Self::from(error),
        }
    }
}

impl From<ContextError> for HarnessError {
    fn from(error: ContextError) -> Self {
        Self::Context(error)
    }
}

impl From<StorageError> for HarnessError {
    fn from(error: StorageError) -> Self {
        Self::Storage(error)
    }
}

impl From<AgentError> for HarnessError {
    fn from(error: AgentError) -> Self {
        Self::Agent(error)
    }
}

/// Why [`Harness::open`](super::Harness::open) failed; nothing keeps running.
#[derive(Debug, thiserror::Error)]
pub enum OpenError {
    #[error("The registry lacks the built-in task {0}; create it with Registry::new()")]
    MissingBuiltin(String),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
}
