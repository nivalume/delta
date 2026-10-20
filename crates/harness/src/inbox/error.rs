//! Why an admission was refused.

use crate::records::{ConversationId, SubmissionType};
use crate::session::TxError;

/// Why [`admit_submission`](super::admit_submission) refused a draft; nothing was written.
#[derive(Debug, thiserror::Error)]
pub enum SubmitError {
    /// A busy conversation refused input with `WhenBusy::Reject`.
    #[error("Conversation {0} is busy")]
    ConversationBusy(ConversationId),
    #[error("An input entry's kind must be non-empty and not start with \"delta.\", got {0:?}")]
    InvalidEntryKind(String),
    #[error("Request {request_id} already identifies a submission of type {}", type_name(*existing))]
    RequestIdTypeMismatch {
        request_id: String,
        existing: SubmissionType,
    },
    #[error(transparent)]
    Tx(#[from] TxError),
}

fn type_name(kind: SubmissionType) -> &'static str {
    match kind {
        SubmissionType::Input => "input",
        SubmissionType::Write => "write",
    }
}

impl SubmitError {
    /// The error a commit closure returns: a transaction error as is, a refusal wrapped so
    /// the caller can take it back out of the rolled-back commit.
    pub fn into_tx(self) -> TxError {
        match self {
            Self::Tx(error) => error,
            refusal => TxError::callback(refusal),
        }
    }
}
