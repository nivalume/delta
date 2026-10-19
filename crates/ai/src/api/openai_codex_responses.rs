//! The `openai-codex-responses` API: the Responses API of the ChatGPT Codex backend
//! (`{baseUrl}/codex/responses`), authenticated with a ChatGPT OAuth access token.
//!
//! Port of upstream `src/api/openai-codex-responses.ts`. Two transports:
//!
//! - WebSocket (`wss://…/codex/responses`): one `response.create` message per request.
//!   With a session id the connection is cached per session and ChatGPT account and reused
//!   for the next request; in `websocket-cached` and `auto` mode a follow-up request on a
//!   cached connection sends only the new input items with `previous_response_id`.
//! - SSE (`POST …/codex/responses`, zstd-compressed body) with its own retry loop.
//!
//! `auto` (the default) tries the WebSocket first and falls back to SSE when it fails before
//! the first event; a session whose WebSocket failed uses SSE from then on. Input conversion
//! and event normalization come from [`super::openai_responses_shared`].

use futures::{SinkExt, StreamExt};
use serde_json::{Map, Value, json};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex, Once, PoisonError};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_util::sync::CancellationToken;

use crate::api::common::{apply_on_payload, notify_response, observe_provider_event, spawn_stream};
use crate::api::openai_responses::mapped_effort;
use crate::api::openai_responses_shared::{
    ConvertResponsesMessagesOptions, ConvertResponsesToolsOptions, GrammarToolInputProperties, ResponsesEventProcessor,
    ResponsesStreamOptions, clamp_openai_prompt_cache_key, convert_responses_messages, convert_responses_tools,
    create_grammar_tool_input_properties, responses_compat,
};
use crate::error::{Error, Result};
use crate::events::{AssistantMessageEventStream, MessageBuilder};
use crate::http;
use crate::model::{CacheRetention, Model, ModelThinkingLevel};
use crate::oauth::openai_codex::chatgpt_account_id;
use crate::options::{
    ApiOptions, OpenAICodexResponsesOptions, ReasoningEffort, SimpleStreamOptions, StreamOptions, ToolChoice, Transport,
};
use crate::provider::ProviderStreams;
use crate::session_resources::{SessionResourceCleanup, register_session_resource_cleanup};
use crate::simple_options::build_base_options;
use crate::transcript::{
    get_declared_tools, get_initial_system_message, get_system_message_text, normalize_context, resolve_transcript,
    resolve_transcript_tools,
};
use crate::types::{
    AssistantMessageDiagnostic, Context, DiagnosticErrorInfo, JsonObject, Message, ProviderHeaders, StopReason,
    TranscriptContext, Usage, now_ms,
};

// ============================================================================
// Configuration
// ============================================================================

const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api";
const DEFAULT_MAX_RETRIES: u32 = 0;
const BASE_DELAY_MS: u64 = 1000;
const DEFAULT_MAX_RETRY_DELAY: Duration = Duration::from_secs(60);
const DEFAULT_WEBSOCKET_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const CODEX_TOOL_CALL_PROVIDERS: [&str; 3] = ["openai", "openai-codex", "opencode"];
const WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE: u16 = 1009;
const WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE: &str = "websocket_connection_limit_reached";
const PREVIOUS_RESPONSE_NOT_FOUND_CODE: &str = "previous_response_not_found";
const CODEX_RESPONSE_STATUSES: [&str; 6] = [
    "completed",
    "incomplete",
    "failed",
    "cancelled",
    "queued",
    "in_progress",
];
const OPENAI_BETA_RESPONSES_WEBSOCKETS: &str = "responses_websockets=2026-02-06";
/// A cached connection closes after this long without a request.
const SESSION_WEBSOCKET_CACHE_TTL: Duration = Duration::from_secs(5 * 60);
/// A cached connection is replaced at this age, before the backend's one-hour limit.
const SESSION_WEBSOCKET_MAX_AGE: Duration = Duration::from_secs(55 * 60);
/// How long a background close waits for the server's close frame.
const CLOSE_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

/// The `openai-codex-responses` API.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAICodexResponsesApi;

impl ProviderStreams for OpenAICodexResponsesApi {
    fn stream(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        stream(model, context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        stream_simple(model, context, options)
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Failure of one transport attempt. The WebSocket retry and SSE fallback logic tells
/// errors of the Codex service (which no other transport fixes) from transport errors.
#[derive(Debug)]
enum CodexFailure {
    /// An `error` or `response.failed` event (upstream `CodexApiError`).
    Api { message: String, code: Option<String> },
    /// A payload that is not JSON (upstream `CodexProtocolError`).
    Protocol(String),
    /// `on_provider_stream_event` failed (upstream `ProviderStreamEventCallbackError`).
    Callback(String),
    /// The WebSocket closed (upstream `WebSocketCloseError`).
    Closed { message: String, code: Option<u16> },
    /// Anything else: connection errors, timeouts, normalization errors, cancellation.
    Other(Error),
}

impl From<Error> for CodexFailure {
    fn from(error: Error) -> Self {
        Self::Other(error)
    }
}

impl From<CodexFailure> for Error {
    fn from(failure: CodexFailure) -> Self {
        match failure {
            CodexFailure::Api { message, .. }
            | CodexFailure::Protocol(message)
            | CodexFailure::Callback(message)
            | CodexFailure::Closed { message, .. } => Error::msg(message),
            CodexFailure::Other(error) => error,
        }
    }
}

impl CodexFailure {
    fn message(&self) -> String {
        match self {
            Self::Api { message, .. }
            | Self::Protocol(message)
            | Self::Callback(message)
            | Self::Closed { message, .. } => message.clone(),
            Self::Other(error) => error.to_string(),
        }
    }

    /// Upstream `isCodexNonTransportError`.
    fn is_non_transport(&self) -> bool {
        matches!(self, Self::Api { .. } | Self::Protocol(_) | Self::Callback(_))
    }

    fn has_code(&self, expected: &str) -> bool {
        matches!(self, Self::Api { code: Some(code), .. } if code == expected)
    }

    fn diagnostic_error(&self) -> DiagnosticErrorInfo {
        let (name, code) = match self {
            Self::Api { code, .. } => ("CodexApiError", code.clone().map(Value::String)),
            Self::Protocol(_) => ("CodexProtocolError", None),
            Self::Callback(_) => ("ProviderStreamEventCallbackError", None),
            Self::Closed { code, .. } => ("WebSocketCloseError", code.map(Value::from)),
            Self::Other(_) => ("Error", None),
        };
        DiagnosticErrorInfo {
            name: Some(name.to_owned()),
            message: self.message(),
            stack: None,
            code,
        }
    }
}

type CodexResult<T> = std::result::Result<T, CodexFailure>;

fn assert_successful_output(builder: &MessageBuilder) -> Result<()> {
    match builder.output.stop_reason {
        StopReason::Pending => Err(Error::msg("Codex stream ended without a stop reason")),
        StopReason::Error | StopReason::Aborted => Err(Error::msg(
            builder
                .output
                .error_message
                .clone()
                .filter(|message| !message.is_empty())
                .unwrap_or_else(|| "An unknown error occurred".into()),
        )),
        _ => Ok(()),
    }
}

// ============================================================================
// Retry helpers
// ============================================================================

/// Whether `text` contains `first`, then optionally one character, then `second`
/// (the regular expression `first.?second`; `.` does not match line terminators).
fn contains_with_optional_gap(text: &str, first: &str, second: &str) -> bool {
    text.match_indices(first).any(|(index, _)| {
        let rest = &text[index + first.len()..];
        if rest.starts_with(second) {
            return true;
        }
        let mut chars = rest.chars();
        match chars.next() {
            Some(c) if !matches!(c, '\n' | '\r' | '\u{2028}' | '\u{2029}') => chars.as_str().starts_with(second),
            _ => false,
        }
    })
}

fn is_terminal_rate_limit_error(error_text: &str) -> bool {
    let text = error_text.to_lowercase();
    [
        "gousagelimiterror",
        "freeusagelimiterror",
        "monthly usage limit reached",
        "available balance",
        "insufficient_quota",
        "out of budget",
        "quota exceeded",
        "billing",
    ]
    .iter()
    .any(|pattern| text.contains(pattern))
}

fn is_retryable_error(status: u16, error_text: &str) -> bool {
    if status == 429 && is_terminal_rate_limit_error(error_text) {
        return false;
    }
    if matches!(status, 429 | 500 | 502 | 503 | 504) {
        return true;
    }
    let text = error_text.to_lowercase();
    contains_with_optional_gap(&text, "rate", "limit")
        || text.contains("overloaded")
        || contains_with_optional_gap(&text, "service", "unavailable")
        || contains_with_optional_gap(&text, "upstream", "connect")
        || contains_with_optional_gap(&text, "connection", "refused")
}

fn ceil_seconds(duration: Duration) -> u64 {
    (duration.as_millis() as u64).div_ceil(1000)
}

/// The server-requested delay, or the error when it exceeds `max_retry_delay`.
fn validate_retry_delay(delay: Duration, options: &StreamOptions) -> std::result::Result<Duration, String> {
    let max = options.max_retry_delay.unwrap_or(DEFAULT_MAX_RETRY_DELAY);
    if !max.is_zero() && delay > max {
        return Err(format!(
            "Server requested {}s retry delay (max: {}s)",
            ceil_seconds(delay),
            ceil_seconds(max)
        ));
    }
    Ok(delay)
}

async fn sleep(delay: Duration, cancel: Option<&tokio_util::sync::CancellationToken>) -> Result<()> {
    http::or_cancel(cancel, async {
        tokio::time::sleep(delay).await;
        Ok(())
    })
    .await
}

// ============================================================================
// Request compression
// ============================================================================

/// zstd frame of the request body. The Codex backend decodes `Content-Encoding: zstd` on the
/// SSE endpoint, as the official Codex client sends it.
fn compress_request_body_zstd(body: &[u8]) -> Vec<u8> {
    ruzstd::encoding::compress_to_vec(body, ruzstd::encoding::CompressionLevel::Fastest)
}

// ============================================================================
// Stream functions
// ============================================================================

fn codex_options_of(options: &StreamOptions) -> OpenAICodexResponsesOptions {
    match &options.api {
        Some(ApiOptions::OpenAICodexResponses(api)) => api.clone(),
        _ => OpenAICodexResponsesOptions::default(),
    }
}

fn transport_name(transport: Transport) -> &'static str {
    match transport {
        Transport::Sse => "sse",
        Transport::Websocket => "websocket",
        Transport::WebsocketCached => "websocket-cached",
        Transport::Auto => "auto",
    }
}

/// Stream a response from the OpenAI Codex Responses API. `options.api_key` must be a
/// ChatGPT OAuth access token (a JWT that names the ChatGPT account).
pub fn stream(model: &Model, context: &TranscriptContext, options: StreamOptions) -> AssistantMessageEventStream {
    ensure_session_cleanup_registered();
    let supports_mid_convo_system_messages = responses_compat(model)
        .supports_mid_convo_system_messages
        .unwrap_or(false);
    let context = resolve_transcript(context, supports_mid_convo_system_messages);
    let target = model.clone();
    spawn_stream(model, options.cancel.clone(), move |mut builder| async move {
        let result = run(&mut builder, &target, &context, &options).await;
        (builder, result)
    })
}

/// Stream with the provider-neutral reasoning options: the reasoning level, clamped to the
/// model's supported levels, becomes `reasoning_effort` (`off` sends none).
#[allow(clippy::needless_pass_by_value)] // The API contract takes the options by value.
pub fn stream_simple(
    model: &Model,
    context: &TranscriptContext,
    options: SimpleStreamOptions,
) -> AssistantMessageEventStream {
    let mut base = build_base_options(model, context, &options);
    let reasoning_effort = options
        .reasoning
        .map(|level| model.clamp_thinking_level(level.into()))
        .and_then(ModelThinkingLevel::reasoning)
        .map(ReasoningEffort::from);
    base.api = Some(ApiOptions::OpenAICodexResponses(OpenAICodexResponsesOptions {
        reasoning_effort,
        tool_choice: options.tool_choice.map(|choice| {
            match choice {
                ToolChoice::Auto => "auto",
                ToolChoice::None => "none",
            }
            .to_owned()
        }),
        ..OpenAICodexResponsesOptions::default()
    }));
    stream(model, context, base)
}

/// Request state shared by both transports.
struct CodexRequest<'a> {
    model: &'a Model,
    options: &'a StreamOptions,
    api: OpenAICodexResponsesOptions,
    grammar: GrammarToolInputProperties,
    cache_session_id: Option<String>,
    account_id: String,
}

impl CodexRequest<'_> {
    fn stream_options(&self) -> ResponsesStreamOptions {
        ResponsesStreamOptions {
            service_tier: self.api.service_tier.clone(),
            grammar_tool_input_properties: self.grammar.clone(),
            resolve_service_tier: Some(resolve_codex_service_tier),
            apply_service_tier_pricing: Some(apply_service_tier_pricing),
        }
    }
}

async fn run(
    builder: &mut MessageBuilder,
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
) -> Result<()> {
    let Some(api_key) = options.api_key.clone().filter(|key| !key.is_empty()) else {
        return Err(Error::msg(format!("No API key for provider: {}", model.provider)));
    };
    let account_id = extract_account_id(&api_key)?;
    let api = codex_options_of(options);
    let compat = responses_compat(model);
    let grammar = create_grammar_tool_input_properties(
        &get_declared_tools(context.messages()),
        compat.supports_openai_grammar_tools.unwrap_or(false),
    )?;
    let cache_session_id = options
        .session_id
        .clone()
        .filter(|_| options.cache_retention != Some(CacheRetention::None));
    let codex_session_id = clamp_openai_prompt_cache_key(cache_session_id.as_deref());
    let body = build_request_body(model, context, &api, options, codex_session_id.as_deref(), &grammar)?;
    let body = apply_on_payload(options, model, body).await?;
    let codex_session_id = codex_session_id.filter(|id| !id.is_empty());
    let cache_session_id = cache_session_id.filter(|id| !id.is_empty());
    let websocket_request_id = codex_session_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::now_v7().to_string());
    let mut sse_headers = build_sse_headers(model, options, &account_id, &api_key, codex_session_id.as_deref())?;
    let websocket_headers = build_websocket_headers(model, options, &account_id, &api_key, &websocket_request_id)?;
    let body_json = serde_json::to_string(&body)?;
    let http_timeout = options.timeout.filter(|timeout| !timeout.is_zero());
    let transport = options.transport.unwrap_or(Transport::Auto);
    let request = CodexRequest {
        model,
        options,
        api,
        grammar,
        cache_session_id,
        account_id,
    };
    let session = request.cache_session_id.as_deref();

    let websocket_disabled_for_session = transport != Transport::Sse && is_websocket_sse_fallback_active(session);
    if websocket_disabled_for_session {
        record_websocket_sse_fallback(session);
    }

    if transport != Transport::Sse && !websocket_disabled_for_session {
        let url = resolve_codex_websocket_url(&model.base_url);
        let mut retried_connection_limit = false;
        let mut retried_missing_continuation = false;
        loop {
            let mut websocket_started = false;
            let attempt = async {
                process_websocket_stream(
                    &url,
                    &body,
                    &websocket_headers,
                    builder,
                    &request,
                    transport,
                    http_timeout,
                    &mut websocket_started,
                )
                .await?;
                if options.is_cancelled() {
                    return Err(CodexFailure::Other(Error::Aborted));
                }
                assert_successful_output(builder)?;
                Ok(())
            }
            .await;
            let failure = match attempt {
                Ok(()) => return Ok(()),
                Err(failure) => failure,
            };
            let aborted = options.is_cancelled();
            let connection_limit_before_start =
                !websocket_started && failure.has_code(WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE);
            if !aborted && failure.has_code(PREVIOUS_RESPONSE_NOT_FOUND_CODE) && !retried_missing_continuation {
                retried_missing_continuation = true;
                continue;
            }
            if !aborted && connection_limit_before_start && !retried_connection_limit {
                retried_connection_limit = true;
                continue;
            }
            if aborted || (failure.is_non_transport() && !connection_limit_before_start) {
                return Err(failure.into());
            }
            let mut details = JsonObject::new();
            details.insert("configuredTransport".into(), transport_name(transport).into());
            if !websocket_started {
                details.insert("fallbackTransport".into(), "sse".into());
            }
            details.insert("eventsEmitted".into(), websocket_started.into());
            details.insert(
                "phase".into(),
                if websocket_started {
                    "after_message_stream_start"
                } else {
                    "before_message_stream_start"
                }
                .into(),
            );
            details.insert("requestBytes".into(), body_json.len().into());
            builder.output.push_diagnostic(AssistantMessageDiagnostic {
                kind: "provider_transport_failure".into(),
                timestamp: now_ms(),
                error: Some(failure.diagnostic_error()),
                details: Some(details),
            });
            record_websocket_failure(session, &failure.message());
            if websocket_started {
                return Err(failure.into());
            }
            record_websocket_sse_fallback(session);
            break;
        }
    }

    // The SSE body is compressed; the WebSocket transport above sends the uncompressed JSON
    // message, as the official Codex client does.
    let sse_body = bytes::Bytes::from(compress_request_body_zstd(body_json.as_bytes()));
    sse_headers.insert(
        reqwest::header::CONTENT_ENCODING,
        reqwest::header::HeaderValue::from_static("zstd"),
    );
    let response = send_sse_request(&request, &sse_headers, &sse_body, http_timeout).await?;

    builder.start();
    let mut reader = SseEventReader::new(response, options.cancel.clone());
    let mut started = true;
    process_codex_events(&mut reader, builder, &request, &mut started)
        .await
        .map_err(Error::from)?;

    if options.is_cancelled() {
        return Err(Error::Aborted);
    }
    assert_successful_output(builder)
}

/// Why an SSE attempt failed: errors the catch-all retry may retry, and errors it may not.
enum AttemptError {
    Retryable(Error),
    Final(Error),
}

/// Send the SSE request with upstream's retry loop: retryable statuses wait for the
/// server-requested delay (bounded by `max_retry_delay`) or `1 s · 2^attempt`; other
/// failures, except usage-limit errors and too long server delays, are retried with the
/// exponential delay too. `max_retries` defaults to 0.
async fn send_sse_request(
    request: &CodexRequest<'_>,
    headers: &reqwest::header::HeaderMap,
    body: &bytes::Bytes,
    http_timeout: Option<Duration>,
) -> Result<reqwest::Response> {
    let options = request.options;
    let cancel = options.cancel.as_ref();
    let max_retries = options.max_retries.unwrap_or(DEFAULT_MAX_RETRIES);
    let client = http::client(options);
    let url = resolve_codex_url(&request.model.base_url);

    let mut attempt: u32 = 0;
    loop {
        if options.is_cancelled() {
            return Err(Error::Aborted);
        }
        let outcome: std::result::Result<Option<reqwest::Response>, AttemptError> = async {
            let send = client.post(&url).headers(headers.clone()).body(body.clone()).send();
            let response = match http_timeout {
                Some(timeout) => {
                    http::or_cancel(cancel, async {
                        match tokio::time::timeout(timeout, send).await {
                            Ok(response) => response.map_err(Error::from),
                            Err(_) => Err(Error::msg(format!(
                                "Codex SSE response headers timed out after {}ms",
                                timeout.as_millis()
                            ))),
                        }
                    })
                    .await
                }
                None => http::or_cancel(cancel, async { send.await.map_err(Error::from) }).await,
            }
            .map_err(AttemptError::Retryable)?;
            notify_response(options, request.model, &response)
                .await
                .map_err(AttemptError::Retryable)?;
            if response.status().is_success() {
                return Ok(Some(response));
            }
            let status = response.status();
            let response_headers = response.headers().clone();
            let error_text = http::or_cancel(cancel, async { response.text().await.map_err(Error::from) })
                .await
                .map_err(AttemptError::Retryable)?;
            if attempt < max_retries && is_retryable_error(status.as_u16(), &error_text) {
                let delay = match http::retry_after(&response_headers) {
                    None => Duration::from_millis(BASE_DELAY_MS * 2u64.pow(attempt)),
                    Some(delay) => validate_retry_delay(delay, options)
                        .map_err(|message| AttemptError::Final(Error::msg(message)))?,
                };
                sleep(delay, cancel).await.map_err(AttemptError::Retryable)?;
                return Ok(None);
            }
            let (message, friendly_message) = parse_error_response(status, &error_text);
            Err(AttemptError::Retryable(Error::msg(friendly_message.unwrap_or(message))))
        }
        .await;
        match outcome {
            Ok(Some(response)) => return Ok(response),
            Ok(None) => {}
            Err(AttemptError::Retryable(Error::Aborted) | AttemptError::Final(Error::Aborted)) => {
                return Err(Error::Aborted);
            }
            Err(AttemptError::Final(error)) => return Err(error),
            Err(AttemptError::Retryable(error)) => {
                // Network errors are retryable.
                if attempt < max_retries && !error.to_string().contains("usage limit") {
                    sleep(Duration::from_millis(BASE_DELAY_MS * 2u64.pow(attempt)), cancel).await?;
                } else {
                    return Err(error);
                }
            }
        }
        attempt += 1;
    }
}

// ============================================================================
// Request building
// ============================================================================

fn build_request_body(
    model: &Model,
    context: &TranscriptContext,
    api: &OpenAICodexResponsesOptions,
    options: &StreamOptions,
    cache_session_id: Option<&str>,
    grammar: &GrammarToolInputProperties,
) -> Result<Value> {
    let compat = responses_compat(model);
    let supports_strict_mode = compat.supports_strict_mode.unwrap_or(true);
    let supports_openai_grammar_tools = compat.supports_openai_grammar_tools.unwrap_or(false);
    let supports_additional_tools = compat.supports_additional_tools.unwrap_or(false);
    let supports_tool_search = compat.supports_tool_search.unwrap_or(false);
    let transcript_tools =
        resolve_transcript_tools(context.messages(), supports_additional_tools || supports_tool_search);
    let tool_options = ConvertResponsesToolsOptions {
        strict: Some(None),
        supports_strict_mode: Some(supports_strict_mode),
        supports_openai_grammar_tools: Some(supports_openai_grammar_tools),
        ..ConvertResponsesToolsOptions::default()
    };
    let input = convert_responses_messages(
        model,
        context,
        &CODEX_TOOL_CALL_PROVIDERS,
        &ConvertResponsesMessagesOptions {
            include_system_prompt: Some(false),
            grammar_tool_input_properties: Some(grammar),
            supports_mid_convo_system_messages: compat.supports_mid_convo_system_messages.unwrap_or(false),
            supports_additional_tools,
            supports_tool_search,
            tool_options,
            ..ConvertResponsesMessagesOptions::default()
        },
    )?;

    let instructions = get_initial_system_message(context.messages())
        .map(get_system_message_text)
        .unwrap_or_default();
    let mut body = Map::new();
    body.insert("model".into(), model.id.clone().into());
    body.insert("store".into(), false.into());
    body.insert("stream".into(), true.into());
    body.insert(
        "instructions".into(),
        if instructions.is_empty() {
            "You are a helpful assistant.".to_owned()
        } else {
            instructions
        }
        .into(),
    );
    body.insert("input".into(), Value::Array(input));
    let verbosity = api
        .text_verbosity
        .clone()
        .filter(|verbosity| !verbosity.is_empty())
        .unwrap_or_else(|| "low".into());
    body.insert("text".into(), json!({"verbosity": verbosity}));
    body.insert("include".into(), json!(["reasoning.encrypted_content"]));
    if let Some(key) = cache_session_id {
        body.insert("prompt_cache_key".into(), key.into());
    }
    body.insert(
        "tool_choice".into(),
        api.tool_choice.clone().unwrap_or_else(|| "auto".into()).into(),
    );
    body.insert("parallel_tool_calls".into(), true.into());

    if let Some(temperature) = options.temperature {
        body.insert("temperature".into(), json!(temperature));
    }
    if let Some(tier) = &api.service_tier {
        body.insert("service_tier".into(), tier.clone().into());
    }
    if !transcript_tools.request_tools.is_empty() {
        body.insert(
            "tools".into(),
            Value::Array(convert_responses_tools(&transcript_tools.request_tools, &tool_options)?),
        );
    }

    let off = model.mapped_thinking_level(ModelThinkingLevel::Off);
    if let Some(requested) = &api.reasoning_effort {
        let effort = if *requested == ReasoningEffort::None {
            match off {
                None => Some("none".to_owned()),
                Some(mapped) => mapped.map(str::to_owned),
            }
        } else {
            Some(mapped_effort(model, requested))
        };
        if let Some(effort) = effort {
            let summary = api.reasoning_summary.clone().unwrap_or_else(|| "auto".into());
            body.insert("reasoning".into(), json!({"effort": effort, "summary": summary}));
        }
    } else if model.reasoning && off != Some(None) {
        body.insert("reasoning".into(), json!({"effort": off.flatten().unwrap_or("none")}));
    }

    Ok(Value::Object(body))
}

fn service_tier_cost_multiplier(model: &Model, service_tier: Option<&str>) -> f64 {
    match service_tier {
        Some("flex") => 0.5,
        Some("priority") => {
            if model.id == "gpt-5.5" {
                2.5
            } else {
                2.0
            }
        }
        _ => 1.0,
    }
}

/// Scale the cost of `usage` by the Codex service tier price multiplier.
pub fn apply_service_tier_pricing(model: &Model, usage: &mut Usage, service_tier: Option<&str>) {
    let multiplier = service_tier_cost_multiplier(model, service_tier);
    if (multiplier - 1.0).abs() < f64::EPSILON {
        return;
    }
    let cost = &mut usage.cost;
    cost.input *= multiplier;
    cost.output *= multiplier;
    cost.cache_read *= multiplier;
    cost.cache_write *= multiplier;
    cost.total = cost.input + cost.output + cost.cache_read + cost.cache_write;
}

/// The tier to price with: Codex echoes `default` for flex and priority requests, so the
/// requested tier wins then.
fn resolve_codex_service_tier(response: Option<&str>, request: Option<&str>) -> Option<String> {
    if response == Some("default") && matches!(request, Some("flex" | "priority")) {
        return request.map(str::to_owned);
    }
    response.or(request).map(str::to_owned)
}

/// The SSE endpoint: `{baseUrl}/codex/responses`, unless the base URL already ends there.
pub fn resolve_codex_url(base_url: &str) -> String {
    let raw = if base_url.trim().is_empty() {
        DEFAULT_CODEX_BASE_URL
    } else {
        base_url
    };
    let normalized = raw.trim_end_matches('/');
    if normalized.ends_with("/codex/responses") {
        normalized.to_owned()
    } else if normalized.ends_with("/codex") {
        format!("{normalized}/responses")
    } else {
        format!("{normalized}/codex/responses")
    }
}

/// The WebSocket endpoint: [`resolve_codex_url`] with `https` → `wss` and `http` → `ws`.
pub fn resolve_codex_websocket_url(base_url: &str) -> String {
    let http_url = resolve_codex_url(base_url);
    let Ok(mut url) = url::Url::parse(&http_url) else {
        return http_url;
    };
    let scheme = match url.scheme() {
        "https" => Some("wss"),
        "http" => Some("ws"),
        _ => None,
    };
    if let Some(scheme) = scheme {
        // `set_scheme` refuses special → non-special changes; rebuild the URL instead.
        let rebuilt = format!("{scheme}{}", &url.as_str()[url.scheme().len()..]);
        if let Ok(rebuilt) = url::Url::parse(&rebuilt) {
            url = rebuilt;
        }
    }
    url.to_string()
}

// ============================================================================
// Event processing
// ============================================================================

/// A source of parsed provider events (SSE or WebSocket).
trait CodexEventSource {
    async fn next_event(&mut self) -> CodexResult<Option<Value>>;
}

fn codex_event_error(event: &Value) -> (Option<String>, Option<String>) {
    let nested = event.get("error").filter(|error| error.is_object());
    let string = |value: Option<&Value>| value.and_then(Value::as_str).map(str::to_owned);
    let code = string(event.get("code")).or_else(|| string(nested.and_then(|error| error.get("code"))));
    let message = string(event.get("message")).or_else(|| string(nested.and_then(|error| error.get("message"))));
    (code, message)
}

/// Map a Codex event the way upstream's `mapCodexEvents` does. `Ok(None)` skips the event;
/// `Ok(Some((event, true)))` is the terminal event, renamed `response.completed`.
fn map_codex_event(event: Value, builder: &mut MessageBuilder) -> CodexResult<Option<(Value, bool)>> {
    let Some(kind) = event.get("type").and_then(Value::as_str).map(str::to_owned) else {
        return Ok(None);
    };
    match kind.as_str() {
        "error" => {
            let (code, message) = codex_event_error(&event);
            let detail = message
                .clone()
                .filter(|message| !message.is_empty())
                .or_else(|| code.clone().filter(|code| !code.is_empty()))
                .unwrap_or_else(|| event.to_string());
            Err(CodexFailure::Api {
                message: format!("Codex error: {detail}"),
                code,
            })
        }
        "response.failed" => {
            let error = event.get("response").and_then(|response| response.get("error"));
            let field = |name: &str| {
                error
                    .and_then(|error| error.get(name))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            };
            Err(CodexFailure::Api {
                message: field("message")
                    .filter(|message| !message.is_empty())
                    .unwrap_or_else(|| "Codex response failed".into()),
                code: field("code"),
            })
        }
        "response.done" | "response.completed" | "response.incomplete" => {
            let mut event = event;
            if let Some(Value::Object(response)) = event.get_mut("response") {
                if let Some(Value::Bool(end_turn)) = response.get("end_turn") {
                    builder.output.end_turn = Some(*end_turn);
                }
                let status = response
                    .get("status")
                    .and_then(Value::as_str)
                    .filter(|status| CODEX_RESPONSE_STATUSES.contains(status))
                    .map(str::to_owned);
                match status {
                    Some(status) => {
                        response.insert("status".into(), status.into());
                    }
                    None => {
                        response.shift_remove("status");
                    }
                }
            }
            if let Value::Object(object) = &mut event {
                object.insert("type".into(), "response.completed".into());
            }
            Ok(Some((event, true)))
        }
        _ => Ok(Some((event, false))),
    }
}

/// Read events from `source`, observe and map them, and normalize them into `builder`.
/// When `started` is false, the first mapped event emits `start` and sets it (the WebSocket
/// transport starts the output on its first event). Reading stops after the terminal event,
/// even when the server keeps the stream open.
async fn process_codex_events(
    source: &mut impl CodexEventSource,
    builder: &mut MessageBuilder,
    request: &CodexRequest<'_>,
    started: &mut bool,
) -> CodexResult<()> {
    let mut processor = ResponsesEventProcessor::new(request.stream_options());
    while let Some(event) = source.next_event().await? {
        if builder.is_closed() {
            return Err(CodexFailure::Other(Error::Aborted));
        }
        // Keep callback failures out of the WebSocket retry and SSE fallback path.
        observe_provider_event(request.options, request.model, &event)
            .await
            .map_err(|error| CodexFailure::Callback(error.to_string()))?;
        let Some((event, terminal)) = map_codex_event(event, builder)? else {
            continue;
        };
        if !*started {
            *started = true;
            builder.start();
        }
        processor.handle_event(builder, request.model, &event)?;
        if terminal {
            break;
        }
    }
    if builder.is_closed() {
        return Err(CodexFailure::Other(Error::Aborted));
    }
    processor.finish(builder)?;
    Ok(())
}

// ============================================================================
// SSE parsing
// ============================================================================

/// Upstream's own SSE parser: frames end at a blank line (`\n\n`); a frame's `data:` lines
/// are trimmed and joined; `[DONE]` and empty data are skipped. A final frame without a
/// trailing blank line is processed at the end of the body.
struct SseEventReader {
    body: futures::stream::BoxStream<'static, reqwest::Result<bytes::Bytes>>,
    buffer: Vec<u8>,
    queue: VecDeque<Value>,
    ended: bool,
    cancel: Option<CancellationToken>,
}

impl SseEventReader {
    fn new(response: reqwest::Response, cancel: Option<CancellationToken>) -> Self {
        Self {
            body: response.bytes_stream().boxed(),
            buffer: Vec::new(),
            queue: VecDeque::new(),
            ended: false,
            cancel,
        }
    }

    fn parse_frame(&mut self, frame: &[u8]) -> CodexResult<()> {
        let frame = String::from_utf8_lossy(frame);
        let data_lines: Vec<&str> = frame
            .split('\n')
            .filter_map(|line| line.strip_prefix("data:"))
            .map(http::js_trim)
            .collect();
        if data_lines.is_empty() {
            return Ok(());
        }
        let data = data_lines.join("\n");
        let data = http::js_trim(&data);
        if data.is_empty() || data == "[DONE]" {
            return Ok(());
        }
        let event = serde_json::from_str(data)
            .map_err(|error| CodexFailure::Protocol(format!("Invalid Codex SSE JSON: {error}")))?;
        self.queue.push_back(event);
        Ok(())
    }

    fn drain_frames(&mut self) -> CodexResult<()> {
        while let Some(index) = self.buffer.windows(2).position(|window| window == b"\n\n") {
            let frame: Vec<u8> = self.buffer.drain(..index + 2).take(index).collect();
            self.parse_frame(&frame)?;
        }
        Ok(())
    }
}

impl CodexEventSource for SseEventReader {
    async fn next_event(&mut self) -> CodexResult<Option<Value>> {
        loop {
            if self.cancel.as_ref().is_some_and(CancellationToken::is_cancelled) {
                return Err(CodexFailure::Other(Error::Aborted));
            }
            if let Some(event) = self.queue.pop_front() {
                return Ok(Some(event));
            }
            if self.ended {
                return Ok(None);
            }
            let next = http::or_cancel(self.cancel.as_ref(), async { Ok(self.body.next().await) }).await?;
            match next {
                Some(Ok(chunk)) => {
                    self.buffer.extend_from_slice(&chunk);
                    self.drain_frames()?;
                }
                Some(Err(error)) => return Err(CodexFailure::Other(Error::from(error))),
                None => {
                    self.ended = true;
                    // Treat the end of the body as the end of the last frame.
                    if !http::js_trim(&String::from_utf8_lossy(&self.buffer)).is_empty() {
                        self.buffer.extend_from_slice(b"\n\n");
                    }
                    self.drain_frames()?;
                }
            }
        }
    }
}

// ============================================================================
// Error responses and auth
// ============================================================================

/// The error message of a failed SSE request, and the friendly usage-limit message when
/// the error is a usage or rate limit (upstream `parseErrorResponse`).
fn parse_error_response(status: reqwest::StatusCode, raw: &str) -> (String, Option<String>) {
    let mut message = if raw.is_empty() {
        status.canonical_reason().unwrap_or("Request failed").to_owned()
    } else {
        raw.to_owned()
    };
    let mut friendly_message = None;
    if let Ok(parsed) = serde_json::from_str::<Value>(raw)
        && let Some(error) = parsed
            .get("error")
            .filter(|error| crate::api::constrained_sampling::js_truthy(error))
    {
        let string = |name: &str| {
            error
                .get(name)
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
        };
        let code = string("code").or_else(|| string("type")).unwrap_or("").to_lowercase();
        if code.contains("usage_limit_reached")
            || code.contains("usage_not_included")
            || code.contains("rate_limit_exceeded")
            || status.as_u16() == 429
        {
            let plan = string("plan_type")
                .map(|plan| format!(" ({} plan)", plan.to_lowercase()))
                .unwrap_or_default();
            let when = error
                .get("resets_at")
                .and_then(Value::as_f64)
                .filter(|resets_at| *resets_at != 0.0)
                .map(|resets_at| {
                    let minutes = ((resets_at * 1000.0 - now_ms() as f64) / 60_000.0).round().max(0.0);
                    format!(" Try again in ~{minutes} min.")
                })
                .unwrap_or_default();
            friendly_message = Some(
                format!("You have hit your ChatGPT usage limit{plan}.{when}")
                    .trim()
                    .to_owned(),
            );
        }
        if let Some(error_message) = string("message") {
            error_message.clone_into(&mut message);
        } else if let Some(friendly) = &friendly_message {
            message.clone_from(friendly);
        }
    }
    (message, friendly_message)
}

fn extract_account_id(token: &str) -> Result<String> {
    chatgpt_account_id(token).ok_or_else(|| Error::msg("Failed to extract accountId from token"))
}

/// Defaults first so model and caller headers can override them; the token and account id
/// always win.
fn base_codex_headers(model: &Model, options: &StreamOptions, account_id: &str, token: &str) -> Vec<ProviderHeaders> {
    let user_agent = http::user_agent();
    let defaults = http::headers_of([("originator", "pi"), ("User-Agent", user_agent.as_str())]);
    let model_headers: ProviderHeaders = model
        .headers
        .iter()
        .flatten()
        .map(|(name, value)| (name.clone(), Some(value.clone())))
        .collect();
    let authorization = format!("Bearer {token}");
    let forced = http::headers_of([
        ("Authorization", authorization.as_str()),
        ("chatgpt-account-id", account_id),
    ]);
    vec![
        defaults,
        model_headers,
        options.headers.clone().unwrap_or_default(),
        forced,
    ]
}

fn build_sse_headers(
    model: &Model,
    options: &StreamOptions,
    account_id: &str,
    token: &str,
    session_id: Option<&str>,
) -> Result<reqwest::header::HeaderMap> {
    let mut layers = base_codex_headers(model, options, account_id, token);
    let mut sse = http::headers_of([
        ("OpenAI-Beta", "responses=experimental"),
        ("accept", "text/event-stream"),
        ("content-type", "application/json"),
    ]);
    if let Some(session_id) = session_id {
        sse.insert("session-id".into(), Some(session_id.to_owned()));
        sse.insert("x-client-request-id".into(), Some(session_id.to_owned()));
    }
    layers.push(sse);
    http::build_headers(&layers)
}

fn build_websocket_headers(
    model: &Model,
    options: &StreamOptions,
    account_id: &str,
    token: &str,
    request_id: &str,
) -> Result<reqwest::header::HeaderMap> {
    let mut layers = base_codex_headers(model, options, account_id, token);
    let mut websocket = ProviderHeaders::new();
    websocket.insert("accept".into(), None);
    websocket.insert("content-type".into(), None);
    websocket.insert("OpenAI-Beta".into(), Some(OPENAI_BETA_RESPONSES_WEBSOCKETS.into()));
    websocket.insert("x-client-request-id".into(), Some(request_id.to_owned()));
    websocket.insert("session-id".into(), Some(request_id.to_owned()));
    layers.push(websocket);
    http::build_headers(&layers)
}

// ============================================================================
// WebSocket connection cache and debug stats
// ============================================================================

type WsStream = tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// WebSocket counters of one session, for debugging and tests (upstream
/// `OpenAICodexWebSocketDebugStats`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpenAICodexWebSocketDebugStats {
    pub requests: u64,
    pub connections_created: u64,
    pub connections_reused: u64,
    pub cached_context_requests: u64,
    pub store_true_requests: u64,
    pub full_context_requests: u64,
    pub delta_requests: u64,
    pub last_input_items: usize,
    pub last_delta_input_items: Option<usize>,
    pub last_previous_response_id: Option<String>,
    pub websocket_failures: u64,
    pub sse_fallbacks: u64,
    pub websocket_fallback_active: Option<bool>,
    pub last_websocket_error: Option<String>,
}

/// What a follow-up request on the same connection may leave out.
#[derive(Clone, Debug)]
#[allow(clippy::struct_field_names)] // Upstream's names.
struct ContinuationState {
    last_request_body: Value,
    last_response_id: String,
    last_response_items: Vec<Value>,
}

enum ParkCommand {
    /// Hand the socket back for a request.
    Return,
    /// Close it with this reason.
    Close(&'static str),
}

/// An idle cached socket. Its task watches the socket while no request uses it: a close by
/// the server makes it unusable, and after [`SESSION_WEBSOCKET_CACHE_TTL`] the task closes
/// it and removes the cache entry. Dropping the command sender closes the socket.
struct ParkedSocket {
    command: oneshot::Sender<ParkCommand>,
    task: JoinHandle<Option<WsStream>>,
}

struct CachedConnection {
    id: u64,
    busy: bool,
    created_at: Instant,
    parked: Option<ParkedSocket>,
    continuation: Option<ContinuationState>,
    /// Fires when [`close_openai_codex_websocket_sessions`] closes the entry; a request
    /// using the socket then fails as if the socket closed.
    closed: CancellationToken,
}

/// Identity of a cache entry (upstream compares entry objects).
#[derive(Clone, Debug)]
struct EntryKey {
    session_id: String,
    account_id: String,
    id: u64,
}

type SessionCache = HashMap<String, HashMap<String, CachedConnection>>;

static SESSION_CACHE: LazyLock<Mutex<SessionCache>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static DEBUG_STATS: LazyLock<Mutex<HashMap<String, OpenAICodexWebSocketDebugStats>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static SSE_FALLBACK_SESSIONS: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(|| Mutex::new(HashSet::new()));
static NEXT_CONNECTION_ID: AtomicU64 = AtomicU64::new(1);
static SESSION_CLEANUP: Once = Once::new();

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Upstream registers the cache cleanup when the module loads, which happens on the first
/// Codex request; this does the same on the first [`stream`] call.
fn ensure_session_cleanup_registered() {
    SESSION_CLEANUP.call_once(|| {
        let cleanup: SessionResourceCleanup = std::sync::Arc::new(|session_id: Option<&str>| {
            close_openai_codex_websocket_sessions(session_id);
            Ok(())
        });
        // The registration lives as long as the process.
        let _registration = register_session_resource_cleanup(cleanup);
    });
}

fn with_debug_stats(session_id: &str, update: impl FnOnce(&mut OpenAICodexWebSocketDebugStats)) {
    update(lock(&DEBUG_STATS).entry(session_id.to_owned()).or_default());
}

/// A copy of the WebSocket counters of `session_id` (upstream
/// `getOpenAICodexWebSocketDebugStats`).
pub fn get_openai_codex_websocket_debug_stats(session_id: &str) -> Option<OpenAICodexWebSocketDebugStats> {
    lock(&DEBUG_STATS).get(session_id).cloned()
}

/// Forget the counters and the SSE fallback of `session_id`, or of every session (upstream
/// `resetOpenAICodexWebSocketDebugStats`).
pub fn reset_openai_codex_websocket_debug_stats(session_id: Option<&str>) {
    match session_id {
        Some(session_id) => {
            lock(&DEBUG_STATS).remove(session_id);
            lock(&SSE_FALLBACK_SESSIONS).remove(session_id);
        }
        None => {
            lock(&DEBUG_STATS).clear();
            lock(&SSE_FALLBACK_SESSIONS).clear();
        }
    }
}

/// Close the cached WebSockets of `session_id`, or of every session (upstream
/// `closeOpenAICodexWebSocketSessions`). Registered as a session resource cleanup.
pub fn close_openai_codex_websocket_sessions(session_id: Option<&str>) {
    let close_entry = |entry: CachedConnection| {
        entry.closed.cancel();
        if let Some(parked) = entry.parked {
            let _ = parked.command.send(ParkCommand::Close("debug_close"));
        }
    };
    let mut cache = lock(&SESSION_CACHE);
    match session_id {
        Some(session_id) => {
            for (_, entry) in cache.remove(session_id).into_iter().flatten() {
                close_entry(entry);
            }
        }
        None => {
            for (_, entry) in cache.drain().flat_map(|(_, entries)| entries) {
                close_entry(entry);
            }
        }
    }
}

fn is_websocket_sse_fallback_active(session_id: Option<&str>) -> bool {
    session_id.is_some_and(|session_id| lock(&SSE_FALLBACK_SESSIONS).contains(session_id))
}

fn record_websocket_sse_fallback(session_id: Option<&str>) {
    let Some(session_id) = session_id else {
        return;
    };
    let active = is_websocket_sse_fallback_active(Some(session_id));
    with_debug_stats(session_id, |stats| {
        stats.sse_fallbacks += 1;
        stats.websocket_fallback_active = Some(active);
    });
}

fn record_websocket_failure(session_id: Option<&str>, error: &str) {
    let Some(session_id) = session_id else {
        return;
    };
    lock(&SSE_FALLBACK_SESSIONS).insert(session_id.to_owned());
    with_debug_stats(session_id, |stats| {
        stats.websocket_failures += 1;
        stats.last_websocket_error = Some(error.to_owned());
        stats.websocket_fallback_active = Some(true);
    });
}

fn close_frame(code: u16, reason: &'static str) -> CloseFrame {
    CloseFrame {
        code: CloseCode::from(code),
        reason: reason.into(),
    }
}

/// Send a close frame and wait briefly for the server's.
async fn close_socket(mut socket: WsStream, reason: &'static str) {
    let _ = tokio::time::timeout(CLOSE_HANDSHAKE_TIMEOUT, async {
        if socket.close(Some(close_frame(1000, reason))).await.is_ok() {
            while let Some(Ok(_)) = socket.next().await {}
        }
    })
    .await;
}

fn close_socket_detached(socket: WsStream, reason: &'static str) {
    tokio::spawn(close_socket(socket, reason));
}

fn remove_entry(cache: &mut SessionCache, key: &EntryKey) -> Option<CachedConnection> {
    let entries = cache.get_mut(&key.session_id)?;
    let removed = if entries.get(&key.account_id).is_some_and(|entry| entry.id == key.id) {
        entries.remove(&key.account_id)
    } else {
        None
    };
    if entries.is_empty() {
        cache.remove(&key.session_id);
    }
    removed
}

fn park(socket: WsStream, key: EntryKey) -> ParkedSocket {
    let (command, mut commands) = oneshot::channel();
    let task = tokio::spawn(async move {
        let mut socket = socket;
        let idle = tokio::time::sleep(SESSION_WEBSOCKET_CACHE_TTL);
        tokio::pin!(idle);
        loop {
            tokio::select! {
                command = &mut commands => {
                    return match command {
                        Ok(ParkCommand::Return) => Some(socket),
                        Ok(ParkCommand::Close(reason)) => {
                            close_socket(socket, reason).await;
                            None
                        }
                        Err(_) => {
                            close_socket(socket, "done").await;
                            None
                        }
                    };
                }
                () = &mut idle => {
                    let removed = remove_entry(&mut lock(&SESSION_CACHE), &key);
                    // The removed entry holds this task's command sender; dropping it here
                    // would only close the socket again.
                    drop(removed);
                    close_socket(socket, "idle_timeout").await;
                    return None;
                }
                message = socket.next() => match message {
                    // Messages on an idle socket have no listener upstream.
                    Some(Ok(WsMessage::Text(_) | WsMessage::Binary(_) | WsMessage::Ping(_) | WsMessage::Pong(_) | WsMessage::Frame(_))) => {}
                    // Closed by the server: the socket is no longer reusable.
                    _ => return None,
                },
            }
        }
    });
    ParkedSocket { command, task }
}

/// A socket checked out for one request.
struct Lease {
    socket: WsStream,
    entry: Option<EntryKey>,
    reused: bool,
    closed: CancellationToken,
}

fn release(lease: Lease, keep: bool) {
    let Some(key) = lease.entry else {
        close_socket_detached(lease.socket, "done");
        return;
    };
    let mut cache = lock(&SESSION_CACHE);
    let entry = cache
        .get_mut(&key.session_id)
        .and_then(|entries| entries.get_mut(&key.account_id))
        .filter(|entry| entry.id == key.id && !entry.closed.is_cancelled());
    match entry {
        Some(entry) if keep => {
            entry.busy = false;
            entry.parked = Some(park(lease.socket, key));
        }
        _ => {
            remove_entry(&mut cache, &key);
            drop(cache);
            close_socket_detached(lease.socket, "done");
        }
    }
}

fn websocket_error(error: &tokio_tungstenite::tungstenite::Error) -> CodexFailure {
    let message = error.to_string();
    CodexFailure::Other(Error::msg(if message.is_empty() {
        "WebSocket error".to_owned()
    } else {
        message
    }))
}

async fn connect_websocket(
    url: &str,
    headers: &reqwest::header::HeaderMap,
    cancel: Option<&CancellationToken>,
    connect_timeout: Duration,
) -> CodexResult<WsStream> {
    let mut request = url.into_client_request().map_err(|error| websocket_error(&error))?;
    for (name, value) in headers {
        request.headers_mut().insert(name.clone(), value.clone());
    }
    let connect = async {
        // Boxed: the TLS handshake future is large.
        Box::pin(tokio_tungstenite::connect_async(request))
            .await
            .map(|(socket, _)| socket)
            .map_err(|error| websocket_error(&error))
    };
    let connect = async {
        if connect_timeout.is_zero() {
            return connect.await;
        }
        match tokio::time::timeout(connect_timeout, connect).await {
            Ok(result) => result,
            Err(_) => Err(CodexFailure::Other(Error::msg(format!(
                "WebSocket connect timeout after {}ms",
                connect_timeout.as_millis()
            )))),
        }
    };
    match cancel {
        Some(cancel) => tokio::select! {
            () = cancel.cancelled() => Err(CodexFailure::Other(Error::Aborted)),
            result = connect => result,
        },
        None => connect.await,
    }
}

enum CacheDecision {
    Reuse(ParkedSocket, EntryKey, CancellationToken),
    OneShot,
    Connect,
}

async fn acquire_websocket(
    url: &str,
    headers: &reqwest::header::HeaderMap,
    session_id: Option<&str>,
    account_id: &str,
    options: &StreamOptions,
) -> CodexResult<Lease> {
    let cancel = options.cancel.as_ref();
    let connect_timeout = options
        .websocket_connect_timeout
        .unwrap_or(DEFAULT_WEBSOCKET_CONNECT_TIMEOUT);
    let Some(session_id) = session_id else {
        let socket = connect_websocket(url, headers, cancel, connect_timeout).await?;
        return Ok(Lease {
            socket,
            entry: None,
            reused: false,
            closed: CancellationToken::new(),
        });
    };

    let decision = {
        let mut cache = lock(&SESSION_CACHE);
        let cached = cache
            .get_mut(session_id)
            .and_then(|entries| entries.get_mut(account_id));
        match cached {
            None => CacheDecision::Connect,
            Some(cached) => {
                let key = EntryKey {
                    session_id: session_id.to_owned(),
                    account_id: account_id.to_owned(),
                    id: cached.id,
                };
                if !cached.busy && cached.created_at.elapsed() >= SESSION_WEBSOCKET_MAX_AGE {
                    if let Some(parked) = cached.parked.take() {
                        let _ = parked.command.send(ParkCommand::Close("connection_age_limit"));
                    }
                    remove_entry(&mut cache, &key);
                    CacheDecision::Connect
                } else if cached.busy {
                    CacheDecision::OneShot
                } else if let Some(parked) = cached.parked.take() {
                    cached.busy = true;
                    CacheDecision::Reuse(parked, key, cached.closed.clone())
                } else {
                    remove_entry(&mut cache, &key);
                    CacheDecision::Connect
                }
            }
        }
    };

    match decision {
        CacheDecision::Reuse(parked, key, closed) => {
            let _ = parked.command.send(ParkCommand::Return);
            if let Ok(Some(socket)) = parked.task.await {
                return Ok(Lease {
                    socket,
                    entry: Some(key),
                    reused: true,
                    closed,
                });
            }
            // The server closed the idle socket: replace it.
            remove_entry(&mut lock(&SESSION_CACHE), &key);
        }
        CacheDecision::OneShot => {
            let socket = connect_websocket(url, headers, cancel, connect_timeout).await?;
            return Ok(Lease {
                socket,
                entry: None,
                reused: false,
                closed: CancellationToken::new(),
            });
        }
        CacheDecision::Connect => {}
    }

    let socket = connect_websocket(url, headers, cancel, connect_timeout).await?;
    let id = NEXT_CONNECTION_ID.fetch_add(1, Ordering::Relaxed);
    let closed = CancellationToken::new();
    lock(&SESSION_CACHE).entry(session_id.to_owned()).or_default().insert(
        account_id.to_owned(),
        CachedConnection {
            id,
            busy: true,
            created_at: Instant::now(),
            parked: None,
            continuation: None,
            closed: closed.clone(),
        },
    );
    Ok(Lease {
        socket,
        entry: Some(EntryKey {
            session_id: session_id.to_owned(),
            account_id: account_id.to_owned(),
            id,
        }),
        reused: false,
        closed,
    })
}

fn with_entry<T>(key: &EntryKey, update: impl FnOnce(&mut CachedConnection) -> T) -> Option<T> {
    lock(&SESSION_CACHE)
        .get_mut(&key.session_id)
        .and_then(|entries| entries.get_mut(&key.account_id))
        .filter(|entry| entry.id == key.id)
        .map(update)
}

// ============================================================================
// WebSocket parsing
// ============================================================================

fn close_error(code: u16, reason: &str) -> CodexFailure {
    let reason_text = if !reason.is_empty() {
        format!(" {reason}")
    } else if code == WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE {
        " message too big".to_owned()
    } else {
        String::new()
    };
    CodexFailure::Closed {
        message: format!("WebSocket closed {code}{reason_text}").trim().to_owned(),
        code: Some(code),
    }
}

/// Reads the events of one response from a WebSocket (upstream `parseWebSocket`).
struct WebSocketEventReader<'a> {
    socket: &'a mut WsStream,
    cancel: Option<CancellationToken>,
    closed: CancellationToken,
    idle_timeout: Option<Duration>,
    done: bool,
    saw_completion: bool,
}

impl WebSocketEventReader<'_> {
    async fn next_message(&mut self) -> CodexResult<Option<tokio_tungstenite::tungstenite::Result<WsMessage>>> {
        let cancelled = async {
            match &self.cancel {
                Some(cancel) => cancel.cancelled().await,
                None => std::future::pending().await,
            }
        };
        let idle = async {
            match self.idle_timeout {
                Some(timeout) => tokio::time::sleep(timeout).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            () = cancelled => Err(CodexFailure::Other(Error::Aborted)),
            () = self.closed.cancelled() => Err(close_error(1000, "debug_close")),
            () = idle => {
                let timeout = self.idle_timeout.unwrap_or_default();
                let _ = tokio::time::timeout(CLOSE_HANDSHAKE_TIMEOUT, self.socket.close(Some(close_frame(1000, "idle_timeout")))).await;
                Err(CodexFailure::Other(Error::msg(format!(
                    "WebSocket idle timeout after {}ms",
                    timeout.as_millis()
                ))))
            }
            message = self.socket.next() => Ok(message),
        }
    }
}

impl CodexEventSource for WebSocketEventReader<'_> {
    async fn next_event(&mut self) -> CodexResult<Option<Value>> {
        loop {
            if self.cancel.as_ref().is_some_and(CancellationToken::is_cancelled) {
                return Err(CodexFailure::Other(Error::Aborted));
            }
            if self.done {
                return if self.saw_completion {
                    Ok(None)
                } else {
                    Err(CodexFailure::Other(Error::msg(
                        "WebSocket stream closed before response.completed",
                    )))
                };
            }
            let text = match self.next_message().await? {
                Some(Ok(WsMessage::Text(text))) => text.to_string(),
                Some(Ok(WsMessage::Binary(data))) => String::from_utf8_lossy(&data).into_owned(),
                Some(Ok(WsMessage::Ping(_) | WsMessage::Pong(_) | WsMessage::Frame(_))) => continue,
                Some(Ok(WsMessage::Close(frame))) => {
                    if self.saw_completion {
                        self.done = true;
                        continue;
                    }
                    // A close frame without a status code is reported as 1005, as browsers do.
                    let (code, reason) = frame.map_or((1005, String::new()), |frame| {
                        (u16::from(frame.code), frame.reason.to_string())
                    });
                    return Err(close_error(code, &reason));
                }
                Some(Err(error)) => return Err(websocket_error(&error)),
                // The connection ended without a close frame (1006, as browsers report it).
                None => {
                    if self.saw_completion {
                        self.done = true;
                        continue;
                    }
                    return Err(close_error(1006, ""));
                }
            };
            if text.is_empty() {
                continue;
            }
            let event: Value = serde_json::from_str(&text)
                .map_err(|error| CodexFailure::Protocol(format!("Invalid Codex WebSocket JSON: {error}")))?;
            let kind = event.get("type").and_then(Value::as_str).unwrap_or("");
            if matches!(kind, "response.completed" | "response.done" | "response.incomplete") {
                self.saw_completion = true;
                self.done = true;
            }
            return Ok(Some(event));
        }
    }
}

/// `body` without `input` and `previous_response_id`, as JSON text.
fn request_body_without_input(body: &Value) -> String {
    let mut body = body.clone();
    if let Value::Object(object) = &mut body {
        object.shift_remove("input");
        object.shift_remove("previous_response_id");
    }
    body.to_string()
}

fn input_items(body: &Value) -> Vec<Value> {
    body.get("input").and_then(Value::as_array).cloned().unwrap_or_default()
}

/// The input items a follow-up request adds after the last request's input and response
/// items, or `None` when the request does not continue it.
fn cached_websocket_input_delta(body: &Value, continuation: &ContinuationState) -> Option<Vec<Value>> {
    if request_body_without_input(body) != request_body_without_input(&continuation.last_request_body) {
        return None;
    }
    let current = input_items(body);
    let mut baseline = input_items(&continuation.last_request_body);
    baseline.extend(continuation.last_response_items.iter().cloned());
    if current.len() < baseline.len() {
        return None;
    }
    let json = |items: &[Value]| serde_json::to_string(items).unwrap_or_default();
    if json(&current[..baseline.len()]) != json(&baseline) {
        return None;
    }
    Some(current[baseline.len()..].to_vec())
}

/// The body to send on a cached connection: only the input delta with
/// `previous_response_id` when the request continues the last one; otherwise the full body
/// (and the continuation is forgotten).
fn build_cached_websocket_request_body(key: &EntryKey, body: &Value) -> Value {
    let continuation = with_entry(key, |entry| entry.continuation.clone()).flatten();
    let Some(continuation) = continuation else {
        return body.clone();
    };
    let delta = cached_websocket_input_delta(body, &continuation);
    match delta {
        Some(delta) if !continuation.last_response_id.is_empty() => {
            let mut body = body.clone();
            if let Value::Object(object) = &mut body {
                object.insert(
                    "previous_response_id".into(),
                    continuation.last_response_id.clone().into(),
                );
                object.insert("input".into(), Value::Array(delta));
            }
            body
        }
        _ => {
            with_entry(key, |entry| entry.continuation = None);
            body.clone()
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn process_websocket_stream(
    url: &str,
    body: &Value,
    headers: &reqwest::header::HeaderMap,
    builder: &mut MessageBuilder,
    request: &CodexRequest<'_>,
    transport: Transport,
    idle_timeout: Option<Duration>,
    started: &mut bool,
) -> CodexResult<()> {
    let options = request.options;
    let session_id = request.cache_session_id.as_deref();
    let mut lease = acquire_websocket(url, headers, session_id, &request.account_id, options).await?;
    let use_cached_context = matches!(transport, Transport::WebsocketCached | Transport::Auto);
    // ChatGPT Codex Responses rejects `store: true`; continuation still works through the
    // connection-scoped `previous_response_id` state.
    let request_body = match &lease.entry {
        Some(key) if use_cached_context => build_cached_websocket_request_body(key, body),
        _ => body.clone(),
    };
    if let Some(session_id) = session_id {
        let input_len = request_body.get("input").and_then(Value::as_array).map_or(0, Vec::len);
        let previous_response_id = request_body
            .get("previous_response_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_owned);
        with_debug_stats(session_id, |stats| {
            stats.requests += 1;
            if lease.reused {
                stats.connections_reused += 1;
            } else {
                stats.connections_created += 1;
            }
            if use_cached_context {
                stats.cached_context_requests += 1;
            }
            if request_body.get("store") == Some(&Value::Bool(true)) {
                stats.store_true_requests += 1;
            }
            stats.last_input_items = input_len;
            if let Some(previous) = previous_response_id {
                stats.delta_requests += 1;
                stats.last_delta_input_items = Some(input_len);
                stats.last_previous_response_id = Some(previous);
            } else {
                stats.full_context_requests += 1;
                stats.last_delta_input_items = None;
                stats.last_previous_response_id = None;
            }
        });
    }

    let result = async {
        let mut message = Map::new();
        message.insert("type".into(), "response.create".into());
        if let Value::Object(fields) = &request_body {
            for (name, value) in fields {
                message.insert(name.clone(), value.clone());
            }
        }
        lease
            .socket
            .send(WsMessage::text(Value::Object(message).to_string()))
            .await
            .map_err(|error| websocket_error(&error))?;
        let mut reader = WebSocketEventReader {
            socket: &mut lease.socket,
            cancel: options.cancel.clone(),
            closed: lease.closed.clone(),
            idle_timeout,
            done: false,
            saw_completion: false,
        };
        process_codex_events(&mut reader, builder, request, started).await
    }
    .await;

    let keep = match &result {
        Err(_) => {
            if let Some(key) = &lease.entry {
                with_entry(key, |entry| entry.continuation = None);
            }
            false
        }
        Ok(()) if options.is_cancelled() => false,
        Ok(()) => {
            if use_cached_context
                && let Some(key) = &lease.entry
                && let Some(response_id) = builder.output.response_id.clone().filter(|id| !id.is_empty())
            {
                let output = normalize_context(&Context::new(vec![Message::Assistant(builder.output.clone())]));
                let response_items = convert_responses_messages(
                    request.model,
                    &output,
                    &CODEX_TOOL_CALL_PROVIDERS,
                    &ConvertResponsesMessagesOptions {
                        include_system_prompt: Some(false),
                        grammar_tool_input_properties: Some(&request.grammar),
                        ..ConvertResponsesMessagesOptions::default()
                    },
                );
                match response_items {
                    Ok(items) => {
                        let items: Vec<Value> = items
                            .into_iter()
                            .filter(|item| {
                                !matches!(
                                    item.get("type").and_then(Value::as_str),
                                    Some("function_call_output" | "custom_tool_call_output")
                                )
                            })
                            .collect();
                        with_entry(key, |entry| {
                            entry.continuation = Some(ContinuationState {
                                last_request_body: body.clone(),
                                last_response_id: response_id,
                                last_response_items: items,
                            });
                        });
                    }
                    Err(_) => {
                        with_entry(key, |entry| entry.continuation = None);
                    }
                }
            }
            true
        }
    };
    release(lease, keep);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::normalize_context;
    use pretty_assertions::assert_eq;

    fn model(id: &str, map: Option<Value>) -> Model {
        let mut value = json!({
            "id": id, "name": id, "api": "openai-codex-responses", "provider": "openai-codex",
            "baseUrl": "https://chatgpt.com/backend-api", "reasoning": true, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 400_000, "maxTokens": 128_000
        });
        if let Some(map) = map {
            value["thinkingLevelMap"] = map;
        }
        serde_json::from_value(value).unwrap()
    }

    fn body(model: &Model, api: &OpenAICodexResponsesOptions) -> Value {
        let context = normalize_context(&Context::new(vec![Message::user("hi")]).with_system_prompt("sys"));
        build_request_body(
            model,
            &context,
            api,
            &StreamOptions::default(),
            Some("session"),
            &GrammarToolInputProperties::new(),
        )
        .unwrap()
    }

    #[test]
    fn request_body_fields_and_order() {
        let body = body(&model("gpt-5.1-codex", None), &OpenAICodexResponsesOptions::default());
        let keys: Vec<&str> = body.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "model",
                "store",
                "stream",
                "instructions",
                "input",
                "text",
                "include",
                "prompt_cache_key",
                "tool_choice",
                "parallel_tool_calls",
                "reasoning"
            ]
        );
        assert_eq!(body["instructions"], "sys");
        assert_eq!(
            body["input"],
            json!([{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}])
        );
        assert_eq!(body["text"], json!({"verbosity": "low"}));
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["reasoning"], json!({"effort": "none"}));
    }

    #[test]
    fn reasoning_effort_mapping() {
        let effort = |map: Option<Value>, effort: ReasoningEffort| {
            body(
                &model("gpt-5.5", map),
                &OpenAICodexResponsesOptions {
                    reasoning_effort: Some(effort),
                    ..OpenAICodexResponsesOptions::default()
                },
            )
            .get("reasoning")
            .cloned()
        };
        assert_eq!(
            effort(Some(json!({"minimal": "low"})), ReasoningEffort::Minimal),
            Some(json!({"effort": "low", "summary": "auto"}))
        );
        assert_eq!(
            effort(None, ReasoningEffort::None),
            Some(json!({"effort": "none", "summary": "auto"}))
        );
        assert_eq!(effort(Some(json!({"off": null})), ReasoningEffort::None), None);
        // A null map entry for a level sends the level itself.
        assert_eq!(
            effort(Some(json!({"xhigh": null})), ReasoningEffort::Xhigh),
            Some(json!({"effort": "xhigh", "summary": "auto"}))
        );
        // Without an effort: the off level, unless it maps to null.
        let no_off = body(
            &model("gpt-5.5", Some(json!({"off": null}))),
            &OpenAICodexResponsesOptions::default(),
        );
        assert!(no_off.get("reasoning").is_none());
    }

    #[test]
    fn resolves_endpoints() {
        assert_eq!(
            resolve_codex_url("https://chatgpt.com/backend-api/"),
            "https://chatgpt.com/backend-api/codex/responses"
        );
        assert_eq!(resolve_codex_url("https://x/codex"), "https://x/codex/responses");
        assert_eq!(
            resolve_codex_url("https://x/codex/responses"),
            "https://x/codex/responses"
        );
        assert_eq!(
            resolve_codex_url(" "),
            "https://chatgpt.com/backend-api/codex/responses"
        );
        assert_eq!(
            resolve_codex_websocket_url("https://chatgpt.com/backend-api"),
            "wss://chatgpt.com/backend-api/codex/responses"
        );
        assert_eq!(
            resolve_codex_websocket_url("http://127.0.0.1:8080"),
            "ws://127.0.0.1:8080/codex/responses"
        );
    }

    #[test]
    fn retry_classification() {
        assert!(is_retryable_error(503, ""));
        assert!(is_retryable_error(429, "rate limited"));
        assert!(!is_retryable_error(429, "insufficient_quota"));
        assert!(!is_retryable_error(429, "GoUsageLimitError"));
        assert!(is_retryable_error(400, "Rate-limit hit"));
        assert!(is_retryable_error(400, "service unavailable"));
        assert!(is_retryable_error(400, "upstream_connect error"));
        assert!(!is_retryable_error(400, "rate\nlimit"));
        assert!(!is_retryable_error(400, "bad request"));
    }

    #[test]
    fn friendly_usage_limit_messages() {
        let status = reqwest::StatusCode::TOO_MANY_REQUESTS;
        let (message, friendly) = parse_error_response(
            status,
            r#"{"error":{"code":"usage_limit_reached","plan_type":"PLUS","message":"limit"}}"#,
        );
        assert_eq!(message, "limit");
        assert_eq!(
            friendly.as_deref(),
            Some("You have hit your ChatGPT usage limit (plus plan).")
        );
        let resets_at = (now_ms() / 1000 + 600) as f64;
        let raw = json!({"error": {"code": "usage_limit_reached", "resets_at": resets_at}}).to_string();
        let (_, friendly) = parse_error_response(reqwest::StatusCode::BAD_REQUEST, &raw);
        assert!(
            friendly
                .unwrap()
                .starts_with("You have hit your ChatGPT usage limit. Try again in ~")
        );
        let (message, friendly) = parse_error_response(reqwest::StatusCode::BAD_REQUEST, "nope");
        assert_eq!((message.as_str(), friendly), ("nope", None));
        let (message, _) = parse_error_response(reqwest::StatusCode::BAD_GATEWAY, "");
        assert_eq!(message, "Bad Gateway");
    }

    #[test]
    fn maps_terminal_events_and_errors() {
        let (sender, _stream) = AssistantMessageEventStream::channel();
        let mut builder = MessageBuilder::new(sender, crate::api::common::new_output(&model("m", None)));
        let (event, terminal) = map_codex_event(
            json!({"type": "response.done", "response": {"status": "weird", "end_turn": false}}),
            &mut builder,
        )
        .unwrap()
        .unwrap();
        assert!(terminal);
        assert_eq!(
            event,
            json!({"type": "response.completed", "response": {"end_turn": false}})
        );
        assert_eq!(builder.output.end_turn, Some(false));
        let error = map_codex_event(
            json!({"type": "error", "error": {"code": "websocket_connection_limit_reached"}}),
            &mut builder,
        )
        .unwrap_err();
        assert!(error.has_code(WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE));
        assert_eq!(error.message(), "Codex error: websocket_connection_limit_reached");
        let error = map_codex_event(json!({"type": "error"}), &mut builder).unwrap_err();
        assert_eq!(error.message(), r#"Codex error: {"type":"error"}"#);
        let error = map_codex_event(json!({"type": "response.failed", "response": {}}), &mut builder).unwrap_err();
        assert_eq!(error.message(), "Codex response failed");
        assert!(map_codex_event(json!({"no": "type"}), &mut builder).unwrap().is_none());
    }

    #[test]
    fn computes_websocket_input_deltas() {
        let continuation = ContinuationState {
            last_request_body: json!({"model": "m", "input": [{"a": 1}]}),
            last_response_id: "resp_1".into(),
            last_response_items: vec![json!({"b": 2})],
        };
        assert_eq!(
            cached_websocket_input_delta(
                &json!({"model": "m", "input": [{"a": 1}, {"b": 2}, {"c": 3}]}),
                &continuation
            ),
            Some(vec![json!({"c": 3})])
        );
        // Other fields changed, or the prefix differs: no delta.
        assert_eq!(
            cached_websocket_input_delta(&json!({"model": "x", "input": [{"a": 1}, {"b": 2}]}), &continuation),
            None
        );
        assert_eq!(
            cached_websocket_input_delta(&json!({"model": "m", "input": [{"a": 2}, {"b": 2}]}), &continuation),
            None
        );
        assert_eq!(
            cached_websocket_input_delta(&json!({"model": "m", "input": [{"a": 1}]}), &continuation),
            None
        );
    }
}
