//! Column encoding: checked integer conversions, record JSON and error mapping.

use delta_chord::sqlite::SqliteError;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::records::{DocumentContent, Seq};
use crate::storage::error::StorageError;

/// The largest id SQLite's `INTEGER` holds; it is never minted.
pub(super) const MAX_ID: u64 = i64::MAX as u64;

/// A `u64` id or sequence as an `INTEGER`; one past `i64::MAX` is refused, never wrapped.
pub(super) fn int(value: u64) -> Result<i64, StorageError> {
    i64::try_from(value).map_err(|_| StorageError::Rejected(format!("{value} is outside SQLite's integer range")))
}

/// An `INTEGER` read back as a `u64`; a negative value is corruption.
pub(super) fn uint(value: i64) -> Result<u64, StorageError> {
    u64::try_from(value).map_err(|_| StorageError::Corrupt(format!("Stored integer {value} is negative")))
}

pub(super) fn encode<T: Serialize>(value: &T) -> Result<String, StorageError> {
    Ok(serde_json::to_string(value)?)
}

/// Decodes a `record` or `content` column; `what` names the row for the error.
pub(super) fn decode<T: DeserializeOwned>(text: &str, what: impl FnOnce() -> String) -> Result<T, StorageError> {
    serde_json::from_str(text).map_err(|error| StorageError::Corrupt(format!("{} is unreadable: {error}", what())))
}

/// The `kind` and `content` columns of a revision.
pub(super) fn encode_content(content: &DocumentContent) -> Result<(&'static str, u32, String), StorageError> {
    Ok(match content {
        DocumentContent::Base { version, value } => ("base", *version, encode(value)?),
        DocumentContent::Delta { version, ops } => ("delta", *version, encode(ops)?),
    })
}

/// A revision from its columns.
pub(super) fn decode_content(
    document: u64,
    seq: Seq,
    kind: &str,
    version: i64,
    content: &str,
) -> Result<DocumentContent, StorageError> {
    let what = || format!("Document {document} revision {seq}");
    let version =
        u32::try_from(version).map_err(|_| StorageError::Corrupt(format!("{} has version {version}", what())))?;
    Ok(match kind {
        "base" => DocumentContent::Base {
            version,
            value: decode(content, what)?,
        },
        "delta" => DocumentContent::Delta {
            version,
            ops: decode(content, what)?,
        },
        other => return Err(StorageError::Corrupt(format!("{} has kind {other:?}", what()))),
    })
}

/// A database error as the storage error it carries: closures return storage errors
/// wrapped in [`SqliteError::Other`].
pub(super) fn from_sqlite(error: SqliteError) -> StorageError {
    match error {
        SqliteError::Other(other) => match other.downcast::<StorageError>() {
            Ok(error) => *error,
            Err(other) => StorageError::Sqlite(SqliteError::Other(other)),
        },
        SqliteError::Closed => StorageError::Closed,
        error => StorageError::Sqlite(error),
    }
}
