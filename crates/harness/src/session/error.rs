//! The Session's errors: a failed Session, a refused call, a rolled-back transaction.

use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;

use delta_chord::context::Cancelled;

use crate::documents::DocumentError;
use crate::records::{ConversationId, RecordError, SubmissionId, TaskId};
use crate::storage::StorageError;

/// The Session failed; `cause` is the first error. Close it and open the storage again.
///
/// Every call after the failure, and every call underway when it happened, returns this
/// error. Retrying cannot succeed: memory may be behind storage.
#[derive(Clone, thiserror::Error)]
#[error("Session failed after a storage error; close and reopen it")]
pub struct SessionFailed {
    #[source]
    pub cause: Arc<dyn StdError + Send + Sync>,
}

impl SessionFailed {
    /// A failure caused by `cause`.
    pub fn new(cause: impl Into<Box<dyn StdError + Send + Sync>>) -> Self {
        Self {
            cause: Arc::from(cause.into()),
        }
    }
}

impl fmt::Debug for SessionFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionFailed")
            .field("cause", &self.cause.to_string())
            .finish()
    }
}

/// Why a Session call did not succeed.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    /// The Session failed, during this call or before it.
    #[error(transparent)]
    Failed(SessionFailed),
    /// The Session is closing or closed.
    #[error("Session is closed")]
    Closed,
    /// The caller's context was cancelled before the call took effect.
    #[error(transparent)]
    Cancelled(Cancelled),
    /// An invalid read ([`StorageError::Request`]); every other storage error fails the
    /// Session and arrives as [`Failed`](Self::Failed).
    #[error(transparent)]
    Storage(StorageError),
    /// A rejected commit or read; nothing was written.
    #[error(transparent)]
    Tx(TxError),
}

impl From<TxError> for SessionError {
    /// Lifts the variants that are not about the transaction itself.
    fn from(error: TxError) -> Self {
        match error {
            TxError::Failed(failed) => Self::Failed(failed),
            TxError::Cancelled(cancelled) => Self::Cancelled(cancelled),
            TxError::Storage(error) => Self::Storage(error),
            other => Self::Tx(other),
        }
    }
}

impl From<SessionFailed> for SessionError {
    fn from(failed: SessionFailed) -> Self {
        Self::Failed(failed)
    }
}

/// Why a transaction rolled back, or why a document read was refused. Nothing was written.
#[derive(Debug, thiserror::Error)]
pub enum TxError {
    #[error("Tx::{method}() cannot read tables after the first table write")]
    ReadAfterWrite { method: &'static str },
    #[error("Conversation {0} does not exist")]
    UnknownConversation(ConversationId),
    #[error("Task {0} does not exist")]
    UnknownTask(TaskId),
    #[error("Submission {0} does not exist")]
    UnknownSubmission(SubmissionId),
    /// New owned work whose owner is `completing`, terminal or abort-marked.
    #[error("{what} {task} is {status}")]
    OwnerUnavailable {
        what: &'static str,
        task: TaskId,
        status: &'static str,
    },
    #[error("Task {0} is terminal")]
    TerminalTask(TaskId),
    #[error("Task {0} cannot change conversations")]
    TaskConversationChange(TaskId),
    /// Task options or a task replacement that break a rule: a background child, a child in
    /// another conversation than its owner, no conversation, a second terminal candidate.
    #[error("{0}")]
    InvalidTask(String),
    /// Definition mismatch, version too new, migration required or failed, non-object value.
    #[error(transparent)]
    Document(DocumentError),
    /// A record's `check()` refused it, or typed entry data did not convert.
    #[error(transparent)]
    Record(RecordError),
    /// An illegal settlement or placement.
    #[error("{0}")]
    Submission(String),
    /// An invisible fork entry, a duplicate copy, a write to a fork source.
    #[error("{0}")]
    Fork(String),
    /// A [`StorageError::Request`] seen by the transaction; any other storage error failed
    /// the Session ([`Failed`](Self::Failed)).
    #[error(transparent)]
    Storage(StorageError),
    /// The transaction's context was cancelled during a read.
    #[error(transparent)]
    Cancelled(Cancelled),
    /// The Session failed while the transaction ran.
    #[error(transparent)]
    Failed(SessionFailed),
    /// The commit closure or a definition function (`initial`, `migrate`,
    /// `checkpoint_when`) panicked; the commit rolled back.
    #[error("panicked: {0}")]
    Panic(String),
    /// The commit closure's own error.
    #[error(transparent)]
    Callback(Box<dyn StdError + Send + Sync>),
}

impl TxError {
    /// Wraps the commit closure's own error.
    pub fn callback(error: impl Into<Box<dyn StdError + Send + Sync>>) -> Self {
        Self::Callback(error.into())
    }
}

impl From<DocumentError> for TxError {
    fn from(error: DocumentError) -> Self {
        Self::Document(error)
    }
}

impl From<RecordError> for TxError {
    fn from(error: RecordError) -> Self {
        Self::Record(error)
    }
}

impl From<SessionFailed> for TxError {
    fn from(failed: SessionFailed) -> Self {
        Self::Failed(failed)
    }
}

/// The message of a caught panic.
pub(super) fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "a panic without a message".to_owned()
    }
}
