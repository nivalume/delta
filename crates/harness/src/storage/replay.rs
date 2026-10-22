//! Document materialization: a base and the delta tail after it, shared by the backends.

use std::borrow::Borrow;

use delta_chord::delta;
use delta_chord::json::Json;

use super::error::StorageError;
use crate::records::{DocumentContent, DocumentId, DocumentRecord, Seq, StoredDocument};

/// Replays `tail` (the revisions after the selected base, in sequence order) onto the base
/// `(version, value)` of `record`. A tail revision that is a base, has another version or
/// does not apply is corruption, not absence.
pub(crate) fn materialize<C: Borrow<DocumentContent>>(
    record: DocumentRecord,
    version: u32,
    mut value: Json,
    tail: impl IntoIterator<Item = Result<(Seq, C), StorageError>>,
) -> Result<StoredDocument, StorageError> {
    let id = record.id;
    let mut deltas_since_base = 0;
    for revision in tail {
        let (seq, content) = revision?;
        match content.borrow() {
            DocumentContent::Delta {
                version: delta_version,
                ops,
            } if *delta_version == version => {
                delta::apply(&mut value, ops).map_err(|error| {
                    StorageError::Corrupt(format!("Document {id} revision {seq} does not apply: {error}"))
                })?;
                deltas_since_base += 1;
            }
            _ => {
                return Err(StorageError::Corrupt(format!(
                    "Document {id} crosses a stored version boundary without a base at revision {seq}"
                )));
            }
        }
    }
    Ok(StoredDocument {
        record,
        version,
        value,
        deltas_since_base,
    })
}

/// The error for an incarnation with no base at or before the point.
pub(crate) fn missing_base(id: DocumentId) -> StorageError {
    StorageError::Corrupt(format!("Document {id} is missing a required base"))
}

/// The error for a historical read of a current-only incarnation.
pub(crate) fn no_history(id: DocumentId) -> StorageError {
    StorageError::Request(format!("Document {id} does not retain historical content"))
}
