//! The storage error type.

use delta_chord::sqlite::{SqliteError, rusqlite};

use crate::env::FileError;

/// What a [`Storage`](super::Storage) method returns on failure.
///
/// Only [`Request`](Self::Request) is an ordinary failure of one call; the Session treats
/// every other error as fatal and reopens.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    /// An invalid read with no durable effect: an unknown conversation, a bad cursor or
    /// limit, history the document does not keep (upstream `StorageRequestError`).
    #[error("{0}")]
    Request(String),
    /// The storage was closed.
    #[error("storage is closed")]
    Closed,
    /// No id is left below the backend's integer maximum; unreachable in practice.
    #[error("ID space is exhausted")]
    IdSpaceExhausted,
    /// A batch the backend refuses, with no effect: id ownership, document consistency,
    /// copy source.
    #[error("{0}")]
    Rejected(String),
    /// Stored data that cannot be read back: a missing base, a version change inside a delta
    /// tail, an op that does not apply, an unreadable row or line.
    #[error("{0}")]
    Corrupt(String),
    /// A write whose outcome is uncertain left the JSONL backend unusable: every later call
    /// fails until the storage is reopened, which recovers what was committed.
    #[error("JSONL storage is poisoned and must be reopened: {0}")]
    Poisoned(String),
    /// The SQLite backend's driver failed.
    #[error(transparent)]
    Sqlite(#[from] SqliteError),
    /// The JSONL backend's file system failed outside a commit (opening, recovery).
    #[error(transparent)]
    File(#[from] FileError),
    /// A record did not encode as JSON.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl StorageError {
    /// Whether the error fails only its own call (the Session's exemption rule).
    pub fn is_request(&self) -> bool {
        matches!(self, Self::Request(_))
    }
}

impl From<rusqlite::Error> for StorageError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Sqlite(SqliteError::Sqlite(error))
    }
}
