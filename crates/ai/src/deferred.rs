//! Deferred responses: a capable provider answers a request with a durable
//! [`DeferredHandle`] (stop reason `deferred`) and continues it asynchronously. The caller
//! later redeems the handle with [`crate::Models::fetch_deferred`] or cancels it with
//! [`crate::Models::cancel_deferred`].
//!
//! Request one with [`crate::SimpleStreamOptions::deferred`]. Providers declare support with
//! [`crate::Provider::supports_fetch_deferred`] / [`crate::Provider::supports_cancel_deferred`];
//! API implementations with the methods of the same names on [`crate::ProviderStreams`].

use std::time::Duration;

use crate::model::Model;
use crate::options::RequestOptions;

pub use crate::types::DeferredHandle;

/// How long a deferred request may run before the provider gives up.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeferredWindow {
    /// `15m`
    FifteenMinutes,
    /// `1h`
    OneHour,
    /// `24h`
    TwentyFourHours,
}

impl DeferredWindow {
    /// Upstream's value: `15m`, `1h` or `24h`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FifteenMinutes => "15m",
            Self::OneHour => "1h",
            Self::TwentyFourHours => "24h",
        }
    }
}

/// The `deferred` option of a simple request. Upstream `deferred: true` is
/// `Some(DeferredRequest::default())`; `deferred: { window }` sets `window`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DeferredRequest {
    pub window: Option<DeferredWindow>,
}

impl DeferredRequest {
    /// A request with a window.
    pub fn with_window(window: DeferredWindow) -> Self {
        Self { window: Some(window) }
    }
}

/// Options of [`crate::Models::stream_deferred`] / [`crate::Models::fetch_deferred`].
#[derive(Clone, Debug, Default)]
pub struct DeferredFetchOptions {
    pub base: RequestOptions<Model>,
    /// Maximum provider long-poll duration. Default: zero, which performs one status check.
    pub wait: Option<Duration>,
}

impl From<RequestOptions<Model>> for DeferredFetchOptions {
    fn from(base: RequestOptions<Model>) -> Self {
        Self { base, wait: None }
    }
}

/// Options of the best-effort [`crate::Models::cancel_deferred`].
pub type DeferredCancelOptions = RequestOptions<Model>;
