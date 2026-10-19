//! Error types.

use std::fmt;

/// Category of a [`ModelsError`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelsErrorCode {
    /// A dynamic model source failed.
    ModelSource,
    /// A model definition is invalid.
    ModelValidation,
    /// Unknown provider, unsupported operation or model type.
    Provider,
    /// A provider has no implementation for a model's API.
    Stream,
    /// API key resolution or the credential store failed.
    Auth,
    /// OAuth refresh or derivation failed. The stored credential is kept; re-login fixes it.
    OAuth,
    /// The operation was cancelled.
    Aborted,
}

impl ModelsErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ModelSource => "model_source",
            Self::ModelValidation => "model_validation",
            Self::Provider => "provider",
            Self::Stream => "stream",
            Self::Auth => "auth",
            Self::OAuth => "oauth",
            Self::Aborted => "aborted",
        }
    }
}

type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Error raised by [`crate::Models`] operations, auth resolution and credential stores.
///
/// As upstream, the message includes the cause's message so callers that only surface
/// `to_string()` still see the underlying reason.
#[derive(Debug)]
pub struct ModelsError {
    code: ModelsErrorCode,
    message: String,
    source: Option<BoxError>,
}

impl ModelsError {
    pub fn new(code: ModelsErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            source: None,
        }
    }

    /// An error whose message is `message: <cause>`.
    pub fn with_source(code: ModelsErrorCode, message: impl Into<String>, source: impl Into<BoxError>) -> Self {
        let source = source.into();
        let mut message = message.into();
        let detail = source.to_string();
        let detail = detail.trim();
        if !detail.is_empty() && !message.contains(detail) {
            message = format!("{message}: {detail}");
        }
        Self {
            code,
            message,
            source: Some(source),
        }
    }

    pub fn aborted() -> Self {
        Self::new(ModelsErrorCode::Aborted, "Request was aborted")
    }

    pub fn code(&self) -> ModelsErrorCode {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ModelsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ModelsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn std::error::Error + 'static))
    }
}

/// Error inside an API implementation. It never escapes a stream: the stream ends with an
/// `error` event whose message carries `to_string()`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Models(#[from] ModelsError),
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// HTTP error response from a provider.
    #[error("{message}")]
    Status {
        status: u16,
        message: String,
        /// The raw response body, trimmed and truncated to
        /// [`crate::http::MAX_PROVIDER_ERROR_BODY_CHARS`] (upstream `normalizeProviderError`'s
        /// `body`); `None` when empty. APIs whose upstream error text comes from the vendor
        /// SDK's error object format their message from it.
        body: Option<String>,
        /// Server-requested retry delay, when present.
        retry_after_ms: Option<u64>,
    },
    #[error("Request was aborted")]
    Aborted,
}

impl Error {
    pub fn msg(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

/// Result alias for API implementation internals.
pub type Result<T, E = Error> = std::result::Result<T, E>;
