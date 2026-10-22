//! Errors of record validation and typed entry conversion.

use super::ids::{DocumentId, SubmissionId, TaskId};

/// A record that breaks a rule its type cannot express, or typed entry data that does not
/// convert.
#[derive(Debug, thiserror::Error)]
pub enum RecordError {
    #[error("Submission {id}: {problem}")]
    Submission { id: SubmissionId, problem: String },
    #[error("Task {id}: {problem}")]
    Task { id: TaskId, problem: String },
    #[error("Document {id}: {problem}")]
    Document { id: DocumentId, problem: String },
    #[error("Entry data of kind {kind} does not convert")]
    EntryData {
        kind: String,
        #[source]
        source: serde_json::Error,
    },
}
