//! Request options shared by every API, plus the typed per-API options.

use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde_json::Value;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::model::{CacheRetention, Model, ThinkingLevel};
use crate::types::{JsonObject, ProviderEnv, ProviderHeaders};

/// Error a request callback may return; it fails the request.
pub type CallbackError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Status and headers of a provider HTTP response.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProviderResponse {
    pub status: u16,
    pub headers: IndexMap<String, String>,
}

/// Inspect or replace the provider payload before it is sent. `Ok(None)` keeps it.
pub type OnPayload = OnPayloadFor<Model>;
/// [`OnPayload`] of a request for a model of type `M`.
pub type OnPayloadFor<M> =
    Arc<dyn Fn(&Value, &M) -> BoxFuture<'static, Result<Option<Value>, CallbackError>> + Send + Sync>;
/// Called after an HTTP response is received and before its body is consumed.
pub type OnResponse = OnResponseFor<Model>;
/// [`OnResponse`] of a request for a model of type `M`.
pub type OnResponseFor<M> =
    Arc<dyn Fn(&ProviderResponse, &M) -> BoxFuture<'static, Result<(), CallbackError>> + Send + Sync>;
/// Observe each parsed provider stream event before normalization. Awaited in stream order.
pub type OnProviderStreamEvent =
    Arc<dyn Fn(&Value, &Model) -> BoxFuture<'static, Result<(), CallbackError>> + Send + Sync>;
/// Transform the fully assembled request headers before provider dispatch (`Models` only).
pub type TransformHeaders = Arc<dyn Fn(ProviderHeaders) -> BoxFuture<'static, ProviderHeaders> + Send + Sync>;

/// Preferred transport for providers that support several.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Transport {
    #[default]
    Sse,
    Websocket,
    WebsocketCached,
    Auto,
}

/// Provider-neutral tool selection for simple requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
}

/// Token budgets per thinking level (token-based providers only).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ThinkingBudgets {
    pub minimal: Option<u64>,
    pub low: Option<u64>,
    pub medium: Option<u64>,
    pub high: Option<u64>,
}

/// Options every API accepts. Unsupported options are ignored by APIs that cannot apply them.
#[derive(Clone, Default)]
pub struct StreamOptions {
    /// Cancels the request; the stream then ends with an `aborted` error.
    pub cancel: Option<CancellationToken>,
    /// Explicit API key. Wins over anything the provider would resolve.
    pub api_key: Option<String>,
    /// Provider-scoped environment values, used before the process environment.
    pub env: Option<ProviderEnv>,
    /// Extra request headers; `None` values suppress API defaults of the same name.
    pub headers: Option<ProviderHeaders>,
    /// HTTP client to send requests with. Defaults to a shared client.
    pub http_client: Option<reqwest::Client>,
    /// Request timeout. Default: 10 minutes.
    pub timeout: Option<Duration>,
    /// Maximum client-side retry attempts. Default: 2.
    pub max_retries: Option<u32>,
    /// Longest server-requested retry delay to wait for; longer requests fail immediately.
    /// Default: 60 s. `Some(Duration::ZERO)` disables the cap.
    pub max_retry_delay: Option<Duration>,
    pub on_payload: Option<OnPayload>,
    pub on_response: Option<OnResponse>,
    pub on_provider_stream_event: Option<OnProviderStreamEvent>,
    pub temperature: Option<f64>,
    /// Sampling parameters merged into the request body as-is (OpenAI-compatible APIs only),
    /// over `Model::sampling_params`.
    pub sampling_params: Option<JsonObject>,
    pub max_tokens: Option<u64>,
    pub transport: Option<Transport>,
    /// Prompt cache retention preference. Default: short.
    pub cache_retention: Option<CacheRetention>,
    /// Session identifier for session-based caching and routing.
    pub session_id: Option<String>,
    pub websocket_connect_timeout: Option<Duration>,
    /// Request metadata; providers extract the fields they understand.
    pub metadata: Option<JsonObject>,
    /// Options of the model's API. Ignored when they do not match `model.api`.
    pub api: Option<ApiOptions>,
    /// Final transform of the assembled headers. Consumed by [`crate::Models`]: it runs after
    /// provider auth, model headers and `headers` are merged, and providers never see it.
    pub transform_headers: Option<TransformHeaders>,
}

impl fmt::Debug for StreamOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamOptions")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("env", &self.env.as_ref().map(|env| env.keys().collect::<Vec<_>>()))
            .field(
                "headers",
                &self.headers.as_ref().map(|headers| headers.keys().collect::<Vec<_>>()),
            )
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("cache_retention", &self.cache_retention)
            .field("session_id", &self.session_id)
            .field("api", &self.api)
            .finish_non_exhaustive()
    }
}

impl StreamOptions {
    pub fn is_cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(CancellationToken::is_cancelled)
    }

    /// A value from `env`, else from the process environment. Blank values count as unset.
    pub fn env_var(&self, name: &str) -> Option<String> {
        crate::auth::env_value(self.env.as_ref(), name)
    }
}

/// Unified options for `stream_simple()`/`complete_simple()`: one reasoning knob mapped to
/// each API's native thinking controls.
#[derive(Clone, Debug, Default)]
pub struct SimpleStreamOptions {
    pub base: StreamOptions,
    pub tool_choice: Option<ToolChoice>,
    pub reasoning: Option<ThinkingLevel>,
    /// Custom token budgets for thinking levels.
    pub thinking_budgets: Option<ThinkingBudgets>,
    /// Ask a capable provider to return a durable handle and continue the request
    /// asynchronously (upstream `deferred: true | { window }`). See [`crate::deferred`].
    pub deferred: Option<crate::deferred::DeferredRequest>,
}

impl From<StreamOptions> for SimpleStreamOptions {
    fn from(base: StreamOptions) -> Self {
        Self {
            base,
            ..Self::default()
        }
    }
}

/// Auth, HTTP transport and callbacks shared by provider requests for a model of type `M`
/// (upstream `ProviderRequestOptions<TModel>`). Image generation, classification and deferred
/// fetches use it; chat streams use the flat [`StreamOptions`].
pub struct RequestOptions<M> {
    /// Cancels the request; the result then has stop reason `aborted`.
    pub cancel: Option<CancellationToken>,
    /// Explicit API key. Wins over anything the provider would resolve.
    pub api_key: Option<String>,
    /// Provider-scoped environment values, used before the process environment.
    pub env: Option<ProviderEnv>,
    /// Extra request headers; `None` values suppress API defaults of the same name.
    pub headers: Option<ProviderHeaders>,
    /// HTTP client to send requests with. Defaults to a shared client.
    pub http_client: Option<reqwest::Client>,
    /// Request timeout. Default: 10 minutes.
    pub timeout: Option<Duration>,
    /// Maximum client-side retry attempts. Default: 2.
    pub max_retries: Option<u32>,
    /// Longest server-requested retry delay to wait for. Default: 60 s; zero disables the cap.
    pub max_retry_delay: Option<Duration>,
    pub on_payload: Option<OnPayloadFor<M>>,
    pub on_response: Option<OnResponseFor<M>>,
    /// Final transform of the assembled headers. Consumed by [`crate::Models`]; providers never
    /// see it.
    pub transform_headers: Option<TransformHeaders>,
}

impl<M> Default for RequestOptions<M> {
    fn default() -> Self {
        Self {
            cancel: None,
            api_key: None,
            env: None,
            headers: None,
            http_client: None,
            timeout: None,
            max_retries: None,
            max_retry_delay: None,
            on_payload: None,
            on_response: None,
            transform_headers: None,
        }
    }
}

impl<M> Clone for RequestOptions<M> {
    fn clone(&self) -> Self {
        Self {
            cancel: self.cancel.clone(),
            api_key: self.api_key.clone(),
            env: self.env.clone(),
            headers: self.headers.clone(),
            http_client: self.http_client.clone(),
            timeout: self.timeout,
            max_retries: self.max_retries,
            max_retry_delay: self.max_retry_delay,
            on_payload: self.on_payload.clone(),
            on_response: self.on_response.clone(),
            transform_headers: self.transform_headers.clone(),
        }
    }
}

impl<M> fmt::Debug for RequestOptions<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RequestOptions")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("env", &self.env.as_ref().map(|env| env.keys().collect::<Vec<_>>()))
            .field(
                "headers",
                &self.headers.as_ref().map(|headers| headers.keys().collect::<Vec<_>>()),
            )
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .finish_non_exhaustive()
    }
}

impl<M> RequestOptions<M> {
    pub fn is_cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(CancellationToken::is_cancelled)
    }

    /// A value from `env`, else from the process environment. Blank values count as unset.
    pub fn env_var(&self, name: &str) -> Option<String> {
        crate::auth::env_value(self.env.as_ref(), name)
    }

    /// The auth and transport settings as [`StreamOptions`], without callbacks, for the
    /// [`crate::http`] helpers (`client`, `send_with_retry`).
    pub fn transport_options(&self) -> StreamOptions {
        StreamOptions {
            cancel: self.cancel.clone(),
            api_key: self.api_key.clone(),
            env: self.env.clone(),
            headers: self.headers.clone(),
            http_client: self.http_client.clone(),
            timeout: self.timeout,
            max_retries: self.max_retries,
            max_retry_delay: self.max_retry_delay,
            ..StreamOptions::default()
        }
    }
}

/// Typed options of one API.
#[derive(Clone, Debug, PartialEq)]
pub enum ApiOptions {
    AnthropicMessages(AnthropicOptions),
    OpenAICompletions(OpenAICompletionsOptions),
    OpenAIResponses(OpenAIResponsesOptions),
    AzureOpenAIResponses(AzureOpenAIResponsesOptions),
    OpenAICodexResponses(OpenAICodexResponsesOptions),
    GoogleGenerativeAi(GoogleOptions),
    GoogleVertex(GoogleVertexOptions),
    MistralConversations(MistralOptions),
    BedrockConverseStream(BedrockOptions),
    PiMessages(PiMessagesOptions),
    /// Options of a custom API.
    Custom(JsonObject),
}

/// Anthropic adaptive-thinking effort.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnthropicEffort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
    /// A provider value from a model's thinking level map.
    Other(String),
}

impl AnthropicEffort {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
            Self::Other(value) => value,
        }
    }
}

/// How thinking content is returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingDisplay {
    Summarized,
    Omitted,
}

/// Tool choice of the Anthropic and Bedrock APIs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnthropicToolChoice {
    Auto,
    Any,
    None,
    Tool { name: String },
}

/// Options of the `anthropic-messages` API.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AnthropicOptions {
    /// Enable extended thinking (adaptive for models that support it, budget-based otherwise).
    pub thinking_enabled: Option<bool>,
    /// Budget for budget-based thinking. Default 1024 when thinking is enabled.
    pub thinking_budget_tokens: Option<u64>,
    /// Effort for adaptive thinking models.
    pub effort: Option<AnthropicEffort>,
    /// Default: summarized when thinking is enabled.
    pub thinking_display: Option<ThinkingDisplay>,
    /// Request the interleaved thinking beta for non-adaptive models. Default: true.
    pub interleaved_thinking: Option<bool>,
    pub tool_choice: Option<AnthropicToolChoice>,
}

/// OpenAI reasoning effort.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
    /// A provider value from a model's thinking level map.
    Other(String),
}

impl ReasoningEffort {
    pub fn as_str(&self) -> &str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
            Self::Other(value) => value,
        }
    }
}

impl From<ThinkingLevel> for ReasoningEffort {
    fn from(level: ThinkingLevel) -> Self {
        match level {
            ThinkingLevel::Minimal => Self::Minimal,
            ThinkingLevel::Low => Self::Low,
            ThinkingLevel::Medium => Self::Medium,
            ThinkingLevel::High => Self::High,
            ThinkingLevel::Xhigh => Self::Xhigh,
            ThinkingLevel::Max => Self::Max,
        }
    }
}

/// Options of the `openai-completions` API.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OpenAICompletionsOptions {
    /// Raw `tool_choice` value (`"auto"`, `"none"`, `"required"` or a function object).
    pub tool_choice: Option<Value>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub thinking_budgets: Option<ThinkingBudgets>,
}

/// Options of the `openai-responses` API (and Azure Responses).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OpenAIResponsesOptions {
    pub reasoning_effort: Option<ReasoningEffort>,
    /// `auto`, `detailed` or `concise`; `Some(None)` sends no summary request.
    pub reasoning_summary: Option<Option<String>>,
    pub service_tier: Option<String>,
    /// Raw `tool_choice` value.
    pub tool_choice: Option<Value>,
}

/// Options of the `azure-openai-responses` API.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AzureOpenAIResponsesOptions {
    pub responses: OpenAIResponsesOptions,
    /// Default: `AZURE_OPENAI_API_VERSION`, else `v1`.
    pub azure_api_version: Option<String>,
    pub azure_resource_name: Option<String>,
    pub azure_base_url: Option<String>,
    /// Deployment name; defaults to the model id or `AZURE_OPENAI_DEPLOYMENT_NAME_MAP`.
    pub azure_deployment_name: Option<String>,
}

/// Options of the `openai-codex-responses` API (upstream `OpenAICodexResponsesOptions`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OpenAICodexResponsesOptions {
    /// `none` sends the model's `off` level.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// `auto`, `concise`, `detailed`, `off` or `on`. Default: `auto`.
    pub reasoning_summary: Option<String>,
    pub service_tier: Option<String>,
    /// `low`, `medium` or `high`. Default: `low`.
    pub text_verbosity: Option<String>,
    /// `auto`, `none` or `required`. Default: `auto`.
    pub tool_choice: Option<String>,
}

/// Gemini thinking level.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GoogleThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
    Other(String),
}

/// Gemini thinking configuration.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GoogleThinking {
    pub enabled: bool,
    /// `-1` for dynamic, `0` to disable.
    pub budget_tokens: Option<i64>,
    pub level: Option<GoogleThinkingLevel>,
}

/// Gemini tool choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoogleToolChoice {
    Auto,
    None,
    Any,
}

/// Options of the `google-generative-ai` API.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GoogleOptions {
    pub tool_choice: Option<GoogleToolChoice>,
    pub thinking: Option<GoogleThinking>,
}

/// Options of the `google-vertex` API.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GoogleVertexOptions {
    pub tool_choice: Option<GoogleToolChoice>,
    pub thinking: Option<GoogleThinking>,
    pub project: Option<String>,
    pub location: Option<String>,
}

/// Options of the `mistral-conversations` API.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MistralOptions {
    /// Raw `tool_choice` value.
    pub tool_choice: Option<Value>,
    /// Only `reasoning` is defined.
    pub prompt_mode: Option<String>,
    pub reasoning_effort: Option<String>,
}

/// Options of the `bedrock-converse-stream` API.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BedrockOptions {
    pub region: Option<String>,
    pub profile: Option<String>,
    pub tool_choice: Option<AnthropicToolChoice>,
    pub reasoning: Option<ThinkingLevel>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    /// Claude 4.x only.
    pub interleaved_thinking: Option<bool>,
    /// Claude only. Default: summarized.
    pub thinking_display: Option<ThinkingDisplay>,
    /// Cost allocation tags.
    pub request_metadata: Option<IndexMap<String, String>>,
    /// Bedrock API key; bypasses SigV4 signing.
    pub bearer_token: Option<String>,
}

/// Options of the `pi-messages` API. They are forwarded to the backend as they are.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PiMessagesOptions {
    pub reasoning: Option<ThinkingLevel>,
    /// Raw `toolChoice` value: `"auto"`, `"none"`, `"required"` or
    /// `{"type": "function", "function": {"name": ...}}`.
    pub tool_choice: Option<Value>,
    /// Ask the backend for debug metadata (e.g. routing response headers).
    pub debug: Option<bool>,
}
