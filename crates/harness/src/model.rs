//! Model access: the [`ModelClient`] seam the agent layer streams through, its
//! [`ModelsClient`] adapter over `delta_ai::Models`, and the request types pinned in task
//! checkpoints ([`ModelRef`], [`ConversationStreamOptions`]).
//!
//! Tasks never hold a `Models` of their own: `HarnessOptions.models` is read at each use, so
//! a host that swaps providers or credentials is followed without a session write. Tests
//! inject a client that replays scripted responses. See `docs/design/harness.md`.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use delta_ai::deferred::{DeferredFetchOptions, DeferredWindow};
use delta_ai::model::{CacheRetention, Model, ModelThinkingLevel};
use delta_ai::options::{RequestOptions, SimpleStreamOptions, StreamOptions, Transport};
use delta_ai::types::{AssistantMessage, DeferredHandle, JsonObject, Message, ProviderHeaders};
use delta_ai::{AssistantMessageEventStream, DeferredRequest, Models};
use delta_chord::context::Context;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

/// A model by provider and id, as `delta.agent` stores it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelRef {
    pub provider: String,
    pub model_id: String,
}

impl ModelRef {
    pub fn new(provider: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model_id: model_id.into(),
        }
    }

    /// `provider/model_id`: the key of usage and compaction overrides.
    pub fn key(&self) -> String {
        format!("{}/{}", self.provider, self.model_id)
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.provider, self.model_id)
    }
}

/// The transport a request prefers, as JSON (`delta_ai::Transport` has no serde form).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamTransport {
    Sse,
    Websocket,
    WebsocketCached,
    Auto,
}

impl From<StreamTransport> for Transport {
    fn from(transport: StreamTransport) -> Self {
        match transport {
            StreamTransport::Sse => Self::Sse,
            StreamTransport::Websocket => Self::Websocket,
            StreamTransport::WebsocketCached => Self::WebsocketCached,
            StreamTransport::Auto => Self::Auto,
        }
    }
}

/// How long a deferred request may run, as JSON.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamDeferredWindow {
    #[serde(rename = "15m")]
    FifteenMinutes,
    #[serde(rename = "1h")]
    OneHour,
    #[serde(rename = "24h")]
    TwentyFourHours,
}

impl From<StreamDeferredWindow> for DeferredWindow {
    fn from(window: StreamDeferredWindow) -> Self {
        match window {
            StreamDeferredWindow::FifteenMinutes => Self::FifteenMinutes,
            StreamDeferredWindow::OneHour => Self::OneHour,
            StreamDeferredWindow::TwentyFourHours => Self::TwentyFourHours,
        }
    }
}

/// The `deferred` request option: ask the provider for a durable handle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamDeferred {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<StreamDeferredWindow>,
}

/// The curated, JSON subset of `delta_ai::StreamOptions` a conversation's requests use
/// (spec §7.1). Pinned in the generation's `request` checkpoint, so a resend after recovery
/// uses the options the request was prepared with; callbacks are not pinned.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConversationStreamOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<StreamTransport>,
    /// Request timeout in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// The provider client's own retries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_retry_delay_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<ProviderHeaders>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JsonObject>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_retention: Option<CacheRetention>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deferred: Option<StreamDeferred>,
}

/// One model request: the resolved model, the messages, the thinking level and the pinned
/// options, with the conversation's provider-facing session id.
#[derive(Clone, Debug)]
pub struct ModelRequest<'a> {
    pub model: &'a Model,
    pub messages: Vec<Message>,
    /// `Off` sends no reasoning option.
    pub thinking_level: ModelThinkingLevel,
    pub options: &'a ConversationStreamOptions,
    /// `delta.provider.session_id`, for providers' prompt caches.
    pub session_id: Option<&'a str>,
    /// The output token limit; `None`: the provider's default.
    pub max_tokens: Option<u64>,
}

/// A failure of a model call that has no message to carry it (cancelling a deferred
/// response).
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ModelError(pub String);

/// Model access for the agent layer. Object-safe; `HarnessOptions.models` holds one.
///
/// Every call is cancelled with `ctx`: a stream ends and a future stops at its next await.
/// Failures are messages with stop reason `error`, as `delta_ai` reports them, so the
/// generation classifies them like any other response.
pub trait ModelClient: Send + Sync {
    /// The model `reference` names, or `None` when no provider knows it (`no_model`).
    fn model(&self, reference: &ModelRef) -> Option<Model>;

    /// Streams one response with partial updates.
    fn stream(&self, request: ModelRequest<'_>, ctx: &Context) -> AssistantMessageEventStream;

    /// Polls or redeems a deferred response; a response still running ends `deferred`.
    fn fetch_deferred<'a>(
        &'a self,
        model: &'a Model,
        handle: &'a DeferredHandle,
        ctx: &'a Context,
    ) -> BoxFuture<'a, AssistantMessage>;

    /// Best-effort cancellation of a deferred response.
    fn cancel_deferred<'a>(
        &'a self,
        model: &'a Model,
        handle: &'a DeferredHandle,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<(), ModelError>>;
}

impl fmt::Debug for dyn ModelClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelClient").finish_non_exhaustive()
    }
}

/// Where a [`ModelsClient`] gets its `Models`: one value, or a closure called at each use.
#[derive(Clone)]
pub enum ModelsSource {
    Fixed(Models),
    Dynamic(Arc<dyn Fn() -> Models + Send + Sync>),
}

impl fmt::Debug for ModelsSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fixed(models) => f.debug_tuple("Fixed").field(models).finish(),
            Self::Dynamic(_) => f.write_str("Dynamic"),
        }
    }
}

/// [`ModelClient`] over `delta_ai::Models`: provider-neutral streaming with the request's
/// reasoning level, options and session id.
#[derive(Clone, Debug)]
pub struct ModelsClient {
    source: ModelsSource,
}

impl ModelsClient {
    /// A client over `models`.
    pub fn new(models: Models) -> Self {
        Self {
            source: ModelsSource::Fixed(models),
        }
    }

    /// A client that reads its `Models` from `models` at each use.
    pub fn dynamic(models: impl Fn() -> Models + Send + Sync + 'static) -> Self {
        Self {
            source: ModelsSource::Dynamic(Arc::new(models)),
        }
    }

    fn models(&self) -> Models {
        match &self.source {
            ModelsSource::Fixed(models) => models.clone(),
            ModelsSource::Dynamic(models) => models(),
        }
    }
}

impl From<ModelsSource> for ModelsClient {
    fn from(source: ModelsSource) -> Self {
        Self { source }
    }
}

/// The transport options of a request: cancellation, timeout, retries, headers.
fn request_options<M>(options: Option<&ConversationStreamOptions>, ctx: &Context) -> RequestOptions<M> {
    let mut request = RequestOptions::<M> {
        cancel: Some(ctx.token()),
        ..RequestOptions::default()
    };
    if let Some(options) = options {
        request.timeout = options.timeout_ms.map(Duration::from_millis);
        request.max_retries = options.max_retries;
        request.max_retry_delay = options.max_retry_delay_ms.map(Duration::from_millis);
        request.headers.clone_from(&options.headers);
    }
    request
}

impl ModelClient for ModelsClient {
    fn model(&self, reference: &ModelRef) -> Option<Model> {
        self.models().get_model(&reference.provider, &reference.model_id)
    }

    fn stream(&self, request: ModelRequest<'_>, ctx: &Context) -> AssistantMessageEventStream {
        let options = request.options;
        let base = StreamOptions {
            cancel: Some(ctx.token()),
            headers: options.headers.clone(),
            timeout: options.timeout_ms.map(Duration::from_millis),
            max_retries: options.max_retries,
            max_retry_delay: options.max_retry_delay_ms.map(Duration::from_millis),
            transport: options.transport.map(Transport::from),
            cache_retention: options.cache_retention,
            session_id: request.session_id.map(str::to_owned),
            metadata: options.metadata.clone(),
            max_tokens: request.max_tokens,
            ..StreamOptions::default()
        };
        let mut simple = SimpleStreamOptions::from(base);
        simple.reasoning = request.thinking_level.reasoning();
        simple.deferred = options.deferred.map(|deferred| DeferredRequest {
            window: deferred.window.map(DeferredWindow::from),
        });
        let context = delta_ai::types::Context::new(request.messages);
        self.models().stream_simple(request.model, &context, simple)
    }

    fn fetch_deferred<'a>(
        &'a self,
        model: &'a Model,
        handle: &'a DeferredHandle,
        ctx: &'a Context,
    ) -> BoxFuture<'a, AssistantMessage> {
        let options = DeferredFetchOptions::from(request_options(None, ctx));
        Box::pin(async move { self.models().fetch_deferred(model, handle, options).await })
    }

    fn cancel_deferred<'a>(
        &'a self,
        model: &'a Model,
        handle: &'a DeferredHandle,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<(), ModelError>> {
        let options = request_options(None, ctx);
        Box::pin(async move {
            self.models()
                .cancel_deferred(model, handle, options)
                .await
                .map_err(|error| ModelError(error.to_string()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_options_are_snake_case_json_without_absent_fields() {
        let options = ConversationStreamOptions {
            transport: Some(StreamTransport::WebsocketCached),
            timeout_ms: Some(5),
            deferred: Some(StreamDeferred {
                window: Some(StreamDeferredWindow::OneHour),
            }),
            ..ConversationStreamOptions::default()
        };
        let json = serde_json::to_value(&options).unwrap();
        assert_eq!(
            json,
            serde_json::json!({ "transport": "websocket_cached", "timeout_ms": 5, "deferred": { "window": "1h" } })
        );
        assert_eq!(
            serde_json::from_value::<ConversationStreamOptions>(json).unwrap(),
            options
        );
    }
}
