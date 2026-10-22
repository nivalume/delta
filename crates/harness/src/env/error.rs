//! Errors of the execution environment: [`FileError`] for the file system, [`ExecError`] for
//! commands.
//!
//! Expected failures are returned with a stable, serializable code that callers branch on; the
//! message is for people, and the underlying [`io::Error`], when there is one, is the
//! [`source`](std::error::Error::source).

use std::fmt;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Result of a file-system operation.
pub type FileResult<T> = Result<T, FileError>;

/// What kind of file-system failure a [`FileError`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileErrorCode {
    /// The context was cancelled.
    Aborted,
    NotFound,
    PermissionDenied,
    /// A path component that must be a directory is not one.
    NotDirectory,
    /// A directory where a file is required.
    IsDirectory,
    /// An invalid argument, a closed reader, an unsupported file kind or a refused symbolic link.
    Invalid,
    /// The environment does not offer the operation (an optional method).
    NotSupported,
    Unknown,
}

impl FileErrorCode {
    /// The code as serialized, such as `"not_found"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Aborted => "aborted",
            Self::NotFound => "not_found",
            Self::PermissionDenied => "permission_denied",
            Self::NotDirectory => "not_directory",
            Self::IsDirectory => "is_directory",
            Self::Invalid => "invalid",
            Self::NotSupported => "not_supported",
            Self::Unknown => "unknown",
        }
    }

    /// The code of an I/O error: by [`io::ErrorKind`], invalid input as [`Invalid`](Self::Invalid),
    /// anything else [`Unknown`](Self::Unknown).
    pub fn from_io_kind(kind: io::ErrorKind) -> Self {
        match kind {
            io::ErrorKind::NotFound => Self::NotFound,
            io::ErrorKind::PermissionDenied => Self::PermissionDenied,
            io::ErrorKind::NotADirectory => Self::NotDirectory,
            io::ErrorKind::IsADirectory => Self::IsDirectory,
            io::ErrorKind::InvalidInput | io::ErrorKind::InvalidFilename => Self::Invalid,
            _ => Self::Unknown,
        }
    }
}

impl fmt::Display for FileErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A failed file-system operation.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct FileError {
    pub code: FileErrorCode,
    pub message: String,
    /// The path the operation failed on, when there is one: absolute for the local environment.
    pub path: Option<String>,
    #[source]
    source: Option<io::Error>,
}

impl FileError {
    pub fn new(code: FileErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            path: None,
            source: None,
        }
    }

    /// The error of an operation whose context was cancelled.
    pub fn aborted() -> Self {
        Self::new(FileErrorCode::Aborted, "aborted")
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(FileErrorCode::Invalid, message)
    }

    /// The error of an optional operation the environment does not offer.
    pub fn not_supported(operation: &str) -> Self {
        Self::new(
            FileErrorCode::NotSupported,
            format!("{operation} is not supported by this environment"),
        )
    }

    /// Map an I/O error by its kind ([`FileErrorCode::from_io_kind`]); the message names the path.
    pub fn from_io(error: io::Error, path: &Path) -> Self {
        Self {
            code: FileErrorCode::from_io_kind(error.kind()),
            message: format!("{error}: {}", path.display()),
            path: Some(path_string(path)),
            source: Some(error),
        }
    }

    #[must_use]
    pub fn with_path(mut self, path: &Path) -> Self {
        self.path = Some(path_string(path));
        self
    }
}

/// What kind of command failure an [`ExecError`] is. A command that runs and exits with a
/// non-zero code is not a failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecErrorCode {
    /// The context was cancelled; the command was killed.
    Aborted,
    /// The timeout elapsed and the command was killed, or the timeout itself was invalid.
    Timeout,
    /// No shell to run a [`Command::Shell`](super::Command::Shell) string.
    ShellUnavailable,
    /// The command could not be started: an empty argv, a missing program or working directory.
    SpawnError,
    /// The output callback panicked; the command was killed.
    CallbackError,
    Unknown,
}

impl ExecErrorCode {
    /// The code as serialized, such as `"spawn_error"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Aborted => "aborted",
            Self::Timeout => "timeout",
            Self::ShellUnavailable => "shell_unavailable",
            Self::SpawnError => "spawn_error",
            Self::CallbackError => "callback_error",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for ExecErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A command that could not run to completion.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ExecError {
    pub code: ExecErrorCode,
    pub message: String,
    /// The spill file of a command that timed out or was aborted after its output crossed the
    /// spill thresholds; it holds the complete output up to the kill.
    pub spill_path: Option<String>,
    #[source]
    source: Option<io::Error>,
}

impl ExecError {
    pub fn new(code: ExecErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            spill_path: None,
            source: None,
        }
    }

    /// The error of a command whose context was cancelled.
    pub fn aborted() -> Self {
        Self::new(ExecErrorCode::Aborted, "aborted")
    }

    /// An error caused by an I/O failure.
    pub fn io(code: ExecErrorCode, message: impl Into<String>, source: io::Error) -> Self {
        Self {
            source: Some(source),
            ..Self::new(code, message)
        }
    }

    #[must_use]
    pub fn with_spill_path(mut self, spill_path: Option<String>) -> Self {
        self.spill_path = spill_path;
        self
    }
}

/// A path as the environment reports it; not UTF-8 parts are replaced.
pub(crate) fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
