//! OpenAI Chat Completions (`openai-completions`) and every OpenAI-compatible endpoint.
//!
//! Port of upstream `api/openai-completions.ts` (plus `openai-prompt-cache.ts`; the
//! constrained-sampling helpers live in [`crate::api::constrained_sampling`]). Requests go to `{baseUrl}/chat/completions` with a bearer
//! key. Server differences are described by [`OpenAICompletionsCompat`] on the model; unset
//! fields are auto-detected from the provider id and base URL ([`detect_compat`]).
//!
//! Design notes: `docs/design/api-openai-completions.md`.

/// Re-exported from its old home here (upstream `utils/hash.ts`).
pub use crate::api::common::short_hash;
/// Re-exported from its old home here; the module is shared by every API.
pub use crate::api::constrained_sampling;

use bytes::Bytes;
use futures::StreamExt;
use indexmap::IndexMap;
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::hash::BuildHasher;

use crate::api::common::{
    self, apply_on_payload, finalize_usage, join_url, notify_response, observe_provider_event, spawn_stream,
};
use crate::api::constrained_sampling::{
    GrammarToolInputJsonBuffer, append_grammar_tool_input_json_delta, create_grammar_tool_input_properties,
    get_grammar_tool_input, get_json_schema_tool_parameters, js_truthy, resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
};
use crate::api::github_copilot_headers::copilot_dynamic_headers;
use crate::compat::{
    ChatTemplateKwargValue, MaxTokensField, ModelCompat, OpenAICompletionsCompat, SessionAffinityFormat,
    ThinkingFormat, ThinkingTokenBudgetField, ThinkingVariable,
};
use crate::error::{Error, Result};
use crate::events::{AssistantMessageEventStream, MessageBuilder};
use crate::http::{self, js_trim};
use crate::model::{CacheRetention, Model, ModelThinkingLevel};
use crate::options::{
    ApiOptions, OpenAICompletionsOptions, ReasoningEffort, SimpleStreamOptions, StreamOptions, ToolChoice,
};
use crate::provider::ProviderStreams;
use crate::simple_options::{
    build_base_options, clamp_thinking_budget_to_answer_room, resolve_sampling_params, thinking_budget_for_level,
};
use crate::transcript::{
    get_declared_tools, get_system_message_text, render_system_message_update, resolve_transcript,
    resolve_transcript_tools,
};
use crate::transform::transform_messages;
use crate::types::{
    AssistantBlock, AssistantMessage, JsonObject, Message, ProviderHeaders, StopReason, Tool, ToolResultMessage,
    TranscriptContext, Usage, UserBlock, UserContent,
};

/// Base URL the OpenAI SDK falls back to when the model has none.
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
/// OpenAI's limit on `prompt_cache_key`, in characters.
pub const OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH: usize = 64;
/// Reasoning delta fields, in priority order: the first non-empty one wins.
const REASONING_FIELDS: [&str; 3] = ["reasoning_content", "reasoning", "reasoning_text"];
const ASSISTANT_BRIDGE_TEXT: &str = "I have processed the tool results.";

/// The `openai-completions` API.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAICompletionsApi;

impl ProviderStreams for OpenAICompletionsApi {
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

// ---------------------------------------------------------------------------
// Compatibility settings
// ---------------------------------------------------------------------------

/// [`OpenAICompletionsCompat`] with every field resolved: explicit model settings over
/// auto-detected values.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedOpenAICompletionsCompat {
    pub supports_store: bool,
    pub supports_developer_role: bool,
    pub supports_reasoning_effort: bool,
    pub supports_usage_in_streaming: bool,
    pub supports_finish_reason: bool,
    pub max_tokens_field: MaxTokensField,
    pub requires_tool_result_name: bool,
    pub requires_assistant_after_tool_result: bool,
    pub requires_thinking_as_text: bool,
    pub requires_reasoning_content_on_assistant_messages: bool,
    pub thinking_format: ThinkingFormat,
    pub chat_template_kwargs: IndexMap<String, ChatTemplateKwargValue>,
    pub chat_template_args: IndexMap<String, ChatTemplateKwargValue>,
    pub zai_tool_stream: bool,
    pub supports_thinking_token_budget: bool,
    pub thinking_token_budget_field: Option<ThinkingTokenBudgetField>,
    pub supports_strict_mode: bool,
    pub supports_openai_grammar_tools: bool,
    pub supports_mid_convo_system_messages: bool,
    pub supports_mid_convo_tool_additions: bool,
    pub cache_control_format: Option<String>,
    pub send_session_affinity_headers: bool,
    pub session_affinity_format: SessionAffinityFormat,
    pub supports_long_cache_retention: bool,
    pub vllm_priority: Option<f64>,
}

/// Compatibility settings detected from the provider id and base URL.
pub fn detect_compat(model: &Model) -> ResolvedOpenAICompletionsCompat {
    let provider = model.provider.as_str();
    let base_url = model.base_url.as_str();
    let has = |needle: &str| base_url.contains(needle);

    let is_zai = matches!(provider, "zai" | "zai-coding-cn") || has("api.z.ai") || has("open.bigmodel.cn");
    let is_together = provider == "together" || has("api.together.ai") || has("api.together.xyz");
    let is_moonshot = matches!(provider, "moonshotai" | "moonshotai-cn") || has("api.moonshot.");
    let is_open_router = provider == "openrouter" || has("openrouter.ai");
    let is_cloudflare_workers_ai = provider == "cloudflare-workers-ai" || has("api.cloudflare.com");
    let is_cloudflare_ai_gateway = provider == "cloudflare-ai-gateway" || has("gateway.ai.cloudflare.com");
    let is_nvidia = provider == "nvidia" || has("integrate.api.nvidia.com");
    let is_ant_ling = provider == "ant-ling" || has("api.ant-ling.com");
    let is_cerebras = provider == "cerebras" || has("cerebras.ai");
    let is_deepseek = provider == "deepseek" || base_url.to_lowercase().contains("deepseek.com");
    let is_chutes = has("chutes.ai");
    let is_grok = provider == "xai" || has("api.x.ai");

    let is_non_standard = is_nvidia
        || is_cerebras
        || is_grok
        || is_together
        || is_chutes
        || is_deepseek
        || is_zai
        || is_moonshot
        || provider == "opencode"
        || has("opencode.ai")
        || is_cloudflare_workers_ai
        || is_cloudflare_ai_gateway
        || is_ant_ling;
    let use_max_tokens = is_chutes
        || is_deepseek
        || is_moonshot
        || is_cloudflare_ai_gateway
        || is_together
        || is_nvidia
        || is_ant_ling
        || is_zai;
    let is_open_router_developer_role_model =
        is_open_router && (model.id.starts_with("anthropic/") || model.id.starts_with("openai/"));
    let cache_control_format =
        (provider == "openrouter" && model.id.starts_with("anthropic/")).then(|| "anthropic".to_owned());
    let thinking_format = if is_deepseek {
        ThinkingFormat::Deepseek
    } else if is_zai {
        ThinkingFormat::Zai
    } else if is_together {
        ThinkingFormat::Together
    } else if is_ant_ling {
        ThinkingFormat::AntLing
    } else if is_open_router {
        ThinkingFormat::Openrouter
    } else {
        ThinkingFormat::Openai
    };

    ResolvedOpenAICompletionsCompat {
        supports_store: !is_non_standard,
        supports_developer_role: is_open_router_developer_role_model || (!is_non_standard && !is_open_router),
        supports_reasoning_effort: !(is_grok
            || is_zai
            || is_moonshot
            || is_together
            || is_cloudflare_ai_gateway
            || is_nvidia
            || is_ant_ling),
        supports_usage_in_streaming: true,
        supports_finish_reason: true,
        max_tokens_field: if use_max_tokens {
            MaxTokensField::MaxTokens
        } else {
            MaxTokensField::MaxCompletionTokens
        },
        requires_tool_result_name: false,
        requires_assistant_after_tool_result: false,
        requires_thinking_as_text: false,
        requires_reasoning_content_on_assistant_messages: is_deepseek,
        thinking_format,
        chat_template_kwargs: IndexMap::new(),
        chat_template_args: IndexMap::new(),
        zai_tool_stream: false,
        supports_thinking_token_budget: false,
        thinking_token_budget_field: None,
        // OpenAI compatibility alone does not imply strict JSON-schema tool support.
        supports_strict_mode: false,
        supports_openai_grammar_tools: false,
        supports_mid_convo_system_messages: false,
        supports_mid_convo_tool_additions: false,
        cache_control_format,
        send_session_affinity_headers: is_open_router,
        session_affinity_format: if is_open_router {
            SessionAffinityFormat::Openrouter
        } else {
            SessionAffinityFormat::Openai
        },
        supports_long_cache_retention: !(is_together
            || is_cloudflare_workers_ai
            || is_cloudflare_ai_gateway
            || is_nvidia
            || is_ant_ling),
        vllm_priority: None,
    }
}

/// The model's explicit compat settings, when they are typed for this API.
fn explicit_compat(model: &Model) -> Option<&OpenAICompletionsCompat> {
    match &model.compat {
        Some(ModelCompat::OpenAICompletions(compat)) => Some(compat),
        _ => None,
    }
}

/// Resolved compatibility settings: auto-detection overridden by explicit `model.compat`.
pub fn get_compat(model: &Model) -> ResolvedOpenAICompletionsCompat {
    let detected = detect_compat(model);
    let Some(c) = explicit_compat(model) else {
        return detected;
    };
    ResolvedOpenAICompletionsCompat {
        supports_store: c.supports_store.unwrap_or(detected.supports_store),
        supports_developer_role: c.supports_developer_role.unwrap_or(detected.supports_developer_role),
        supports_reasoning_effort: c
            .supports_reasoning_effort
            .unwrap_or(detected.supports_reasoning_effort),
        supports_usage_in_streaming: c
            .supports_usage_in_streaming
            .unwrap_or(detected.supports_usage_in_streaming),
        supports_finish_reason: c.supports_finish_reason.unwrap_or(detected.supports_finish_reason),
        max_tokens_field: c.max_tokens_field.unwrap_or(detected.max_tokens_field),
        requires_tool_result_name: c
            .requires_tool_result_name
            .unwrap_or(detected.requires_tool_result_name),
        requires_assistant_after_tool_result: c
            .requires_assistant_after_tool_result
            .unwrap_or(detected.requires_assistant_after_tool_result),
        requires_thinking_as_text: c
            .requires_thinking_as_text
            .unwrap_or(detected.requires_thinking_as_text),
        requires_reasoning_content_on_assistant_messages: c
            .requires_reasoning_content_on_assistant_messages
            .unwrap_or(detected.requires_reasoning_content_on_assistant_messages),
        thinking_format: c.thinking_format.unwrap_or(detected.thinking_format),
        chat_template_kwargs: c.chat_template_kwargs.clone().unwrap_or(detected.chat_template_kwargs),
        chat_template_args: c.chat_template_args.clone().unwrap_or(detected.chat_template_args),
        zai_tool_stream: c.zai_tool_stream.unwrap_or(detected.zai_tool_stream),
        supports_thinking_token_budget: c
            .supports_thinking_token_budget
            .unwrap_or(detected.supports_thinking_token_budget),
        thinking_token_budget_field: c.thinking_token_budget_field.or(detected.thinking_token_budget_field),
        supports_strict_mode: c.supports_strict_mode.unwrap_or(detected.supports_strict_mode),
        supports_openai_grammar_tools: c
            .supports_openai_grammar_tools
            .unwrap_or(detected.supports_openai_grammar_tools),
        supports_mid_convo_system_messages: c
            .supports_mid_convo_system_messages
            .unwrap_or(detected.supports_mid_convo_system_messages),
        supports_mid_convo_tool_additions: c
            .supports_mid_convo_tool_additions
            .unwrap_or(detected.supports_mid_convo_tool_additions),
        cache_control_format: c.cache_control_format.clone().or(detected.cache_control_format),
        send_session_affinity_headers: c
            .send_session_affinity_headers
            .unwrap_or(detected.send_session_affinity_headers),
        session_affinity_format: c.session_affinity_format.unwrap_or(detected.session_affinity_format),
        supports_long_cache_retention: c
            .supports_long_cache_retention
            .unwrap_or(detected.supports_long_cache_retention),
        vllm_priority: c.vllm_priority,
    }
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// The options of this API carried in `options.api`, or the defaults.
fn api_options(options: &StreamOptions) -> OpenAICompletionsOptions {
    match &options.api {
        Some(ApiOptions::OpenAICompletions(api)) => api.clone(),
        _ => OpenAICompletionsOptions::default(),
    }
}

/// Stream a Chat Completions response.
pub fn stream(model: &Model, context: &TranscriptContext, options: StreamOptions) -> AssistantMessageEventStream {
    let compat = get_compat(model);
    let context = resolve_transcript(context, compat.supports_mid_convo_system_messages);
    let task_model = model.clone();
    spawn_stream(model, options.cancel.clone(), move |mut builder| async move {
        let mut state = StreamState::default();
        let result = run(&mut builder, &mut state, &task_model, &context, &options, &compat).await;
        if result.is_err() {
            // Keep the reasoning replay data of a failed response.
            state.apply_reasoning_details(&mut builder);
        }
        (builder, result)
    })
}

/// Stream with provider-neutral options: `reasoning` becomes `reasoning_effort` (clamped to
/// the model's levels), `tool_choice` and `thinking_budgets` pass through.
#[allow(clippy::needless_pass_by_value)] // Owned options are the API contract.
pub fn stream_simple(
    model: &Model,
    context: &TranscriptContext,
    options: SimpleStreamOptions,
) -> AssistantMessageEventStream {
    let mut base = build_base_options(model, context, &options);
    let reasoning = options
        .reasoning
        .map(|level| model.clamp_thinking_level(ModelThinkingLevel::from(level)))
        .and_then(ModelThinkingLevel::reasoning);
    base.api = Some(ApiOptions::OpenAICompletions(OpenAICompletionsOptions {
        tool_choice: options.tool_choice.map(|choice| {
            Value::String(
                match choice {
                    ToolChoice::Auto => "auto",
                    ToolChoice::None => "none",
                }
                .into(),
            )
        }),
        reasoning_effort: reasoning.map(ReasoningEffort::from),
        thinking_budgets: options.thinking_budgets,
    }));
    stream(model, context, base)
}

/// Whether `headers` holds a non-blank value for `name`.
fn has_header(headers: Option<&ProviderHeaders>, name: &str) -> bool {
    headers.into_iter().flatten().any(|(key, value)| {
        key.eq_ignore_ascii_case(name) && value.as_deref().is_some_and(|value| !js_trim(value).is_empty())
    })
}

/// The bearer key: the API key, or `unused` when an authorization header stands in for it.
fn client_api_key(model: &Model, options: &StreamOptions) -> Result<String> {
    if let Some(key) = options.api_key.as_deref().filter(|key| !key.is_empty()) {
        return Ok(key.to_owned());
    }
    let headers = options.headers.as_ref();
    if has_header(headers, "authorization") || has_header(headers, "cf-aig-authorization") {
        return Ok("unused".into());
    }
    Err(Error::msg(format!("No API key for provider: {}", model.provider)))
}

fn resolve_cache_retention(options: &StreamOptions) -> CacheRetention {
    if let Some(retention) = options.cache_retention {
        return retention;
    }
    if options.env_var("PI_CACHE_RETENTION").as_deref() == Some("long") {
        return CacheRetention::Long;
    }
    CacheRetention::Short
}

/// Truncate a prompt cache key to OpenAI's limit, counting characters.
pub fn clamp_openai_prompt_cache_key(key: &str) -> String {
    key.chars().take(OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH).collect()
}

/// Request headers: JSON and bearer defaults, the user agent, model headers, GitHub Copilot's
/// dynamic headers, session affinity headers, then `options.headers` (where `None` deletes a default).
fn request_headers(
    model: &Model,
    messages: &[Message],
    options: &StreamOptions,
    api_key: &str,
    session_id: Option<&str>,
    compat: &ResolvedOpenAICompletionsCompat,
) -> Result<reqwest::header::HeaderMap> {
    let mut defaults = http::headers_of([
        ("Accept", "application/json"),
        ("Content-Type", "application/json"),
        ("Authorization", &format!("Bearer {api_key}")),
        ("User-Agent", &http::user_agent()),
    ]);
    for (name, value) in model.headers.iter().flatten() {
        defaults.insert(name.clone(), Some(value.clone()));
    }
    defaults.extend(copilot_dynamic_headers(&model.provider, messages).into_iter().flatten());
    if let Some(session_id) = session_id.filter(|_| compat.send_session_affinity_headers) {
        let mut set = |name: &str| {
            defaults.insert(name.to_owned(), Some(session_id.to_owned()));
        };
        if compat.session_affinity_format == SessionAffinityFormat::Openrouter {
            set("x-session-id");
        } else {
            if compat.session_affinity_format == SessionAffinityFormat::Openai {
                set("session_id");
            }
            set("x-client-request-id");
            set("x-session-affinity");
        }
    }
    let none = ProviderHeaders::new();
    http::build_headers([&defaults, options.headers.as_ref().unwrap_or(&none)])
}

async fn run(
    builder: &mut MessageBuilder,
    state: &mut StreamState,
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
    compat: &ResolvedOpenAICompletionsCompat,
) -> Result<()> {
    let api_key = client_api_key(model, options)?;
    let api = api_options(options);
    state.grammar = create_grammar_tool_input_properties(
        &get_declared_tools(context.messages()),
        compat.supports_openai_grammar_tools,
    )?;
    let cache_retention = resolve_cache_retention(options);
    let session_id = options
        .session_id
        .as_deref()
        .filter(|_| cache_retention != CacheRetention::None);
    let headers = request_headers(model, context.messages(), options, &api_key, session_id, compat)?;
    let params = build_params_with(model, context, options, &api, compat, cache_retention, &state.grammar)?;
    let params = apply_on_payload(options, model, params).await?;
    let body = Bytes::from(serde_json::to_vec(&params)?);
    let base_url = if model.base_url.is_empty() {
        DEFAULT_BASE_URL
    } else {
        model.base_url.as_str()
    };
    let url = join_url(base_url, "chat/completions");
    let client = http::client(options);
    let response = http::send_with_retry(options, || {
        Ok(client.post(&url).headers(headers.clone()).body(body.clone()))
    })
    .await
    .map_err(format_http_error)?;
    notify_response(options, model, &response).await?;
    builder.start();

    let events = http::sse_events(response, options.cancel.clone());
    let mut events = std::pin::pin!(events);
    while let Some(event) = events.next().await {
        if builder.is_closed() {
            return Err(Error::Aborted);
        }
        let event = match event {
            // As with the OpenAI SDK, cancellation ends the chunk loop; the blocks received
            // so far are closed below and the request then fails as aborted.
            Err(Error::Aborted) => break,
            other => other?,
        };
        let data = event.data.as_str();
        if data.starts_with("[DONE]") {
            break;
        }
        if data.trim().is_empty() {
            continue;
        }
        let chunk: Value = serde_json::from_str(data)
            .map_err(|error| Error::msg(format!("Could not parse stream chunk: {error}: {data}")))?;
        if let Some(error) = chunk_error(&chunk) {
            return Err(error);
        }
        observe_provider_event(options, model, &chunk).await?;
        state.handle_chunk(builder, model, &chunk)?;
    }

    state.finish(builder)?;
    if options.is_cancelled() {
        return Err(Error::Aborted);
    }
    let output = &mut builder.output;
    if !state.has_finish_reason && !compat.supports_finish_reason {
        output.stop_reason = if output.tool_calls().next().is_some() {
            StopReason::ToolUse
        } else {
            StopReason::Stop
        };
    }
    if output.stop_reason == StopReason::Error {
        return Err(Error::msg(
            output
                .error_message
                .clone()
                .unwrap_or_else(|| "Provider returned an error stop reason".into()),
        ));
    }
    if (compat.supports_finish_reason && !state.has_finish_reason) || output.stop_reason == StopReason::Pending {
        return Err(Error::msg("Stream ended without finish_reason"));
    }
    Ok(())
}

/// The message upstream builds for an HTTP error: `formatProviderError` over the OpenAI SDK
/// error (`"<status>: <error JSON>"`, or the SDK message such as `"<status> <text>"` when the
/// body has no `error` object), plus OpenRouter's `error.metadata.raw` on its own line when
/// the message does not already contain it.
fn format_http_error(error: Error) -> Error {
    let Error::Status {
        status,
        body,
        retry_after_ms,
        ..
    } = error
    else {
        return error;
    };
    let (sdk_message, error_object) = common::openai_sdk_error(status, body.as_deref());
    let mut message = common::format_provider_error(status, &sdk_message, error_object.as_ref(), None);
    // Some providers via OpenRouter give additional information in this field.
    if let Some(raw) = error_object
        .as_ref()
        .and_then(|error| error.pointer("/metadata/raw"))
        .filter(|raw| js_truthy(raw))
    {
        let raw = raw.as_str().map_or_else(|| raw.to_string(), str::to_owned);
        if !message.contains(&raw) {
            message.push('\n');
            message.push_str(&raw);
        }
    }
    Error::Status {
        status,
        message,
        body,
        retry_after_ms,
    }
}

/// An error carried inside the stream: a truthy top-level `error` (as the OpenAI SDK
/// raises it), or an error envelope without completion fields (`{"detail": ...}`,
/// `{"object": "error", ...}`).
fn chunk_error(chunk: &Value) -> Option<Error> {
    let object = chunk.as_object()?;
    if let Some(error) = object.get("error").filter(|error| js_truthy(error)) {
        let mut message = match error.get("message") {
            Some(Value::String(text)) if !text.is_empty() => text.clone(),
            Some(value) if js_truthy(value) => value.to_string(),
            _ => error.to_string(),
        };
        // Some providers behind OpenRouter put the upstream reason in metadata.raw.
        if let Some(raw) = error.pointer("/metadata/raw").filter(|raw| js_truthy(raw)) {
            let raw = raw.as_str().map_or_else(|| raw.to_string(), str::to_owned);
            if !message.contains(&raw) {
                message.push('\n');
                message.push_str(&raw);
            }
        }
        return Some(Error::msg(message));
    }
    let is_completion = object.contains_key("choices") || object.contains_key("usage") || object.contains_key("id");
    let is_error_envelope = object.contains_key("detail")
        || object.get("object").and_then(Value::as_str) == Some("error")
        || object.get("message").is_some_and(Value::is_string);
    if is_completion || !is_error_envelope {
        return None;
    }
    let detail = object.get("detail").or_else(|| object.get("message")).map_or_else(
        || chunk.to_string(),
        |value| value.as_str().map_or_else(|| value.to_string(), str::to_owned),
    );
    Some(Error::msg(detail))
}

// ---------------------------------------------------------------------------
// Request parameters
// ---------------------------------------------------------------------------

/// Build the request body for `model`, `context` and `options` (`stream: true`).
pub fn build_params(model: &Model, context: &TranscriptContext, options: &StreamOptions) -> Result<Value> {
    let compat = get_compat(model);
    let context = resolve_transcript(context, compat.supports_mid_convo_system_messages);
    let grammar = create_grammar_tool_input_properties(
        &get_declared_tools(context.messages()),
        compat.supports_openai_grammar_tools,
    )?;
    build_params_with(
        model,
        &context,
        options,
        &api_options(options),
        &compat,
        resolve_cache_retention(options),
        &grammar,
    )
}

/// Whether the transcript contains tool calls or tool results.
fn has_tool_history(messages: &[Message]) -> bool {
    messages.iter().any(|message| match message {
        Message::ToolResult(_) => true,
        Message::Assistant(assistant) => assistant.tool_calls().next().is_some(),
        _ => false,
    })
}

fn build_params_with(
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
    api: &OpenAICompletionsOptions,
    compat: &ResolvedOpenAICompletionsCompat,
    cache_retention: CacheRetention,
    grammar: &HashMap<String, String>,
) -> Result<Value> {
    let transcript_tools = resolve_transcript_tools(
        context.messages(),
        compat.supports_mid_convo_system_messages && compat.supports_mid_convo_tool_additions,
    );
    let mut messages = convert_messages(model, context, compat, grammar)?;
    let cache_control = compat_cache_control(compat, cache_retention);
    let long_retention = cache_retention == CacheRetention::Long && compat.supports_long_cache_retention;

    let mut params = JsonObject::new();
    params.insert("model".into(), Value::String(model.id.clone()));
    // Filled in below, once cache markers are applied; inserted now to keep key order.
    params.insert("messages".into(), Value::Null);
    params.insert("stream".into(), Value::Bool(true));
    if ((model.base_url.contains("api.openai.com") && cache_retention != CacheRetention::None) || long_retention)
        && let Some(session_id) = &options.session_id
    {
        params.insert(
            "prompt_cache_key".into(),
            Value::String(clamp_openai_prompt_cache_key(session_id)),
        );
    }
    if long_retention {
        params.insert("prompt_cache_retention".into(), Value::String("24h".into()));
    }
    if compat.supports_usage_in_streaming {
        params.insert("stream_options".into(), json!({"include_usage": true}));
    }
    if compat.supports_store {
        params.insert("store".into(), Value::Bool(false));
    }
    if let Some(max_tokens) = options.max_tokens.filter(|max| *max > 0) {
        let field = match compat.max_tokens_field {
            // Deprecated by OpenAI, but some OpenAI-compatible providers only accept max_tokens.
            MaxTokensField::MaxTokens => "max_tokens",
            MaxTokensField::MaxCompletionTokens => "max_completion_tokens",
        };
        params.insert(field.into(), Value::from(max_tokens));
    }
    if let Some(temperature) = options.temperature {
        params.insert("temperature".into(), number_value(temperature));
    }

    let mut tools = None;
    if !transcript_tools.request_tools.is_empty() {
        tools = Some(convert_tools(&transcript_tools.request_tools, compat)?);
        if compat.zai_tool_stream {
            params.insert("tool_stream".into(), Value::Bool(true));
        }
    } else if has_tool_history(context.messages()) {
        // Anthropic (via LiteLLM/proxy) requires tools when the conversation has tool calls.
        tools = Some(Vec::new());
    }
    if let Some(cache_control) = &cache_control {
        apply_anthropic_cache_control(&mut messages, tools.as_mut(), cache_control);
    }
    params.insert("messages".into(), Value::Array(messages));
    if let Some(tools) = tools {
        // `tools` precedes `tool_stream` upstream; key order has no meaning on the wire.
        params.insert("tools".into(), Value::Array(tools));
    }
    if let Some(tool_choice) = api.tool_choice.as_ref().filter(|choice| js_truthy(choice)) {
        params.insert("tool_choice".into(), tool_choice.clone());
    }
    if let Some(priority) = compat.vllm_priority {
        params.insert("priority".into(), number_value(priority));
    }

    let effort = api.reasoning_effort.as_ref().map(ReasoningEffort::as_str);
    let budget_field = resolve_thinking_token_budget_field(compat);
    let thinking_budget = resolve_clamped_thinking_budget(model, options, api);
    apply_thinking_params(&mut params, model, compat, effort, thinking_budget);

    // Cap reasoning with a top-level budget field. Independent of thinkingFormat: the same
    // server can serve zai, qwen or chat-template models. Reasoning and the answer share
    // max_tokens here, so an uncapped reasoning phase can consume the whole response.
    if let (Some(field), Some(budget)) = (budget_field, thinking_budget) {
        let name = match field {
            ThinkingTokenBudgetField::ThinkingTokenBudget => "thinking_token_budget",
            ThinkingTokenBudgetField::ThinkingBudget => "thinking_budget",
            ThinkingTokenBudgetField::ThinkingBudgetTokens => "thinking_budget_tokens",
        };
        params.insert(name.into(), Value::from(budget));
    }

    if let Some(explicit) = explicit_compat(model) {
        // OpenRouter provider routing preferences.
        if let Some(routing) = &explicit.open_router_routing {
            params.insert("provider".into(), Value::Object(routing.clone()));
        }
        // Vercel AI Gateway provider routing preferences.
        if let Some(routing) = &explicit.vercel_gateway_routing {
            let mut gateway = JsonObject::new();
            for key in ["only", "order"] {
                if let Some(value) = routing.get(key).filter(|value| js_truthy(value)) {
                    gateway.insert(key.into(), value.clone());
                }
            }
            if !gateway.is_empty() {
                params.insert("providerOptions".into(), json!({"gateway": gateway}));
            }
        }
    }

    // Last so model and request sampling parameters override named request fields.
    let level = effort.and_then(level_of).unwrap_or(ModelThinkingLevel::Off);
    if let Some(sampling) = resolve_sampling_params(model, level, options.sampling_params.as_ref()) {
        for (key, value) in sampling {
            params.insert(key, value);
        }
    }
    Ok(Value::Object(params))
}

/// A JSON number; integral values are written without a fraction, as JavaScript does.
fn number_value(number: f64) -> Value {
    if number.is_finite() && number.fract() == 0.0 && number.abs() < 9_007_199_254_740_992.0 {
        Value::from(number as i64)
    } else {
        serde_json::Number::from_f64(number).map_or(Value::Null, Value::Number)
    }
}

/// The thinking level named `name`.
fn level_of(name: &str) -> Option<ModelThinkingLevel> {
    ModelThinkingLevel::ALL.into_iter().find(|level| level.as_str() == name)
}

/// `model.thinkingLevelMap?.[name]`: `None` when unmapped, `Some(None)` for an explicit null.
#[allow(clippy::option_option)] // Mirrors `Model::mapped_thinking_level`.
fn level_value<'a>(model: &'a Model, name: &str) -> Option<Option<&'a str>> {
    level_of(name).and_then(|level| model.mapped_thinking_level(level))
}

/// `model.thinkingLevelMap?.[name] ?? fallback`.
fn level_value_or(model: &Model, name: &str, fallback: &str) -> String {
    level_value(model, name).flatten().unwrap_or(fallback).to_owned()
}

fn resolve_thinking_token_budget_field(compat: &ResolvedOpenAICompletionsCompat) -> Option<ThinkingTokenBudgetField> {
    compat.thinking_token_budget_field.or_else(|| {
        compat
            .supports_thinking_token_budget
            .then_some(ThinkingTokenBudgetField::ThinkingTokenBudget)
    })
}

/// The thinking budget for the requested effort, capped to leave room for the answer.
fn resolve_clamped_thinking_budget(
    model: &Model,
    options: &StreamOptions,
    api: &OpenAICompletionsOptions,
) -> Option<u64> {
    if !model.reasoning {
        return None;
    }
    let level = level_of(api.reasoning_effort.as_ref()?.as_str())?.reasoning()?;
    let ceiling = options.max_tokens.filter(|max| *max > 0).unwrap_or(model.max_tokens);
    let budget =
        clamp_thinking_budget_to_answer_room(thinking_budget_for_level(level, api.thinking_budgets.as_ref()), ceiling);
    (budget > 0).then_some(budget)
}

/// Reasoning request fields per `thinkingFormat`.
fn apply_thinking_params(
    params: &mut JsonObject,
    model: &Model,
    compat: &ResolvedOpenAICompletionsCompat,
    effort: Option<&str>,
    thinking_budget: Option<u64>,
) {
    if !model.reasoning {
        return;
    }
    let supports_effort = compat.supports_reasoning_effort;
    let off_is_null = level_value(model, "off") == Some(None);
    let mut set = |key: &str, value: Value| {
        params.insert(key.to_owned(), value);
    };
    match compat.thinking_format {
        ThinkingFormat::Zai => {
            set(
                "thinking",
                if effort.is_some() {
                    json!({"type": "enabled", "clear_thinking": false})
                } else {
                    json!({"type": "disabled"})
                },
            );
            if let Some(effort) = effort.filter(|_| supports_effort) {
                match level_value(model, effort) {
                    None => set("reasoning_effort", Value::String(effort.to_owned())),
                    Some(Some(mapped)) => set("reasoning_effort", Value::String(mapped.to_owned())),
                    Some(None) => {}
                }
            }
        }
        ThinkingFormat::Qwen => {
            set("enable_thinking", Value::Bool(effort.is_some()));
            if let Some(effort) = effort.filter(|_| supports_effort) {
                set("reasoning_effort", Value::String(level_value_or(model, effort, effort)));
            }
        }
        ThinkingFormat::QwenChatTemplate => {
            set(
                "chat_template_kwargs",
                json!({"enable_thinking": effort.is_some(), "preserve_thinking": true}),
            );
        }
        ThinkingFormat::ChatTemplate => {
            if let Some(kwargs) =
                build_chat_template_values(model, effort, &compat.chat_template_kwargs, thinking_budget)
            {
                set("chat_template_kwargs", Value::Object(kwargs));
            }
        }
        ThinkingFormat::Baseten => {
            if let Some(args) = build_chat_template_values(model, effort, &compat.chat_template_args, thinking_budget) {
                set("chat_template_args", Value::Object(args));
            }
            if supports_effort {
                let mapped = level_value(model, effort.unwrap_or("off"));
                let value = match mapped {
                    None => effort,
                    Some(mapped) => mapped,
                };
                if let Some(value) = value {
                    set("reasoning_effort", Value::String(value.to_owned()));
                }
            }
        }
        ThinkingFormat::Deepseek => {
            if effort.is_some() {
                set("thinking", json!({"type": "enabled"}));
            } else if !off_is_null {
                set("thinking", json!({"type": "disabled"}));
            }
            if let Some(effort) = effort.filter(|_| supports_effort) {
                set("reasoning_effort", Value::String(level_value_or(model, effort, effort)));
            }
        }
        ThinkingFormat::Openrouter => {
            // OpenRouter normalizes reasoning across providers via a nested reasoning object.
            if let Some(effort) = effort {
                set("reasoning", json!({"effort": level_value_or(model, effort, effort)}));
            } else if !off_is_null {
                set("reasoning", json!({"effort": level_value_or(model, "off", "none")}));
            }
        }
        ThinkingFormat::AntLing => {
            if let Some(Some(mapped)) = effort.map(|effort| level_value(model, effort).flatten()) {
                set("reasoning", json!({"effort": mapped}));
            }
        }
        ThinkingFormat::Together => {
            set("reasoning", json!({"enabled": effort.is_some()}));
            if let Some(effort) = effort.filter(|_| supports_effort) {
                set("reasoning_effort", Value::String(level_value_or(model, effort, effort)));
            }
        }
        ThinkingFormat::StringThinking => {
            if let Some(effort) = effort {
                set("thinking", Value::String(level_value_or(model, effort, effort)));
            } else if !off_is_null {
                set("thinking", Value::String(level_value_or(model, "off", "none")));
            }
        }
        ThinkingFormat::Openai => {
            if !supports_effort {
                return;
            }
            match effort {
                // OpenAI-style reasoning_effort.
                Some(effort) => set("reasoning_effort", Value::String(level_value_or(model, effort, effort))),
                None => {
                    if let Some(Some(off)) = level_value(model, "off") {
                        set("reasoning_effort", Value::String(off.to_owned()));
                    }
                }
            }
        }
    }
}

/// Resolve chat template kwargs/args; `None` when nothing remains.
fn build_chat_template_values(
    model: &Model,
    effort: Option<&str>,
    values: &IndexMap<String, ChatTemplateKwargValue>,
    thinking_budget: Option<u64>,
) -> Option<JsonObject> {
    let mut resolved = JsonObject::new();
    for (key, value) in values {
        let value = match value {
            ChatTemplateKwargValue::Literal(literal) => Some(literal.clone()),
            ChatTemplateKwargValue::Var { var, omit_when_off } => {
                if effort.is_none() && *omit_when_off == Some(true) {
                    None
                } else {
                    match var {
                        ThinkingVariable::Enabled => Some(Value::Bool(effort.is_some())),
                        ThinkingVariable::Budget => thinking_budget.map(Value::from),
                        ThinkingVariable::Effort => match level_value(model, effort.unwrap_or("off")) {
                            None => effort.map(|effort| Value::String(effort.to_owned())),
                            Some(mapped) => mapped.map(|mapped| Value::String(mapped.to_owned())),
                        },
                    }
                }
            }
        };
        if let Some(value) = value {
            resolved.insert(key.clone(), value);
        }
    }
    (!resolved.is_empty()).then_some(resolved)
}

// ---------------------------------------------------------------------------
// Anthropic-style cache_control markers
// ---------------------------------------------------------------------------

fn compat_cache_control(compat: &ResolvedOpenAICompletionsCompat, retention: CacheRetention) -> Option<Value> {
    if compat.cache_control_format.as_deref() != Some("anthropic") || retention == CacheRetention::None {
        return None;
    }
    let mut cache_control = json!({"type": "ephemeral"});
    if retention == CacheRetention::Long && compat.supports_long_cache_retention {
        cache_control["ttl"] = Value::String("1h".into());
    }
    Some(cache_control)
}

/// Mark the system prompt, the last tool and the last conversation text for caching.
fn apply_anthropic_cache_control(messages: &mut [Value], tools: Option<&mut Vec<Value>>, cache_control: &Value) {
    let role = |message: &Value| {
        message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    if let Some(system) = messages
        .iter_mut()
        .find(|message| matches!(role(message).as_str(), "system" | "developer"))
    {
        add_cache_control_to_text_content(system, cache_control);
    }
    if let Some(Value::Object(tool)) = tools.and_then(|tools| tools.last_mut()) {
        tool.insert("cache_control".into(), cache_control.clone());
    }
    for message in messages.iter_mut().rev() {
        if matches!(role(message).as_str(), "user" | "assistant" | "tool")
            && add_cache_control_to_text_content(message, cache_control)
        {
            return;
        }
    }
}

fn add_cache_control_to_text_content(message: &mut Value, cache_control: &Value) -> bool {
    let Some(content) = message.get_mut("content") else {
        return false;
    };
    match content {
        Value::String(text) => {
            if text.is_empty() {
                return false;
            }
            *content = json!([{"type": "text", "text": std::mem::take(text), "cache_control": cache_control}]);
            true
        }
        Value::Array(parts) => {
            for part in parts.iter_mut().rev() {
                if let Some(part) = part.as_object_mut()
                    && part.get("type").and_then(Value::as_str) == Some("text")
                {
                    part.insert("cache_control".into(), cache_control.clone());
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Messages and tools
// ---------------------------------------------------------------------------

fn utf16_len(text: &str) -> usize {
    text.chars().map(char::len_utf16).sum()
}

/// Replace every character outside `[a-zA-Z0-9_-]` with `_` per UTF-16 code unit.
fn sanitize_id_part(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
            out.push(c);
        } else {
            out.extend(std::iter::repeat_n('_', c.len_utf16()));
        }
    }
    out
}

/// Tool call id from another API, made acceptable to Chat Completions.
///
/// Responses-style ids (`{call_id}|{item_id}`, where the item id can be 400+ characters
/// with `+`, `/`, `=`) keep both parts, sanitized; ids longer than 40 characters keep a
/// prefix plus a hash, so calls sharing a `call_id` stay distinct. OpenAI ids are cut to 40.
pub fn normalize_tool_call_id(id: &str, provider: &str) -> String {
    if let Some((call, item)) = id.split_once('|') {
        let call_id = sanitize_id_part(call);
        let item_id = sanitize_id_part(item);
        let combined = if item_id.is_empty() {
            call_id.clone()
        } else {
            format!("{call_id}_{item_id}")
        };
        if combined.len() <= 40 {
            return combined;
        }
        let hash: String = short_hash(id).chars().take(8).collect();
        let keep = 40usize.saturating_sub(hash.len() + 1).max(1).min(call_id.len());
        return format!("{}_{hash}", &call_id[..keep]);
    }
    if provider == "openai" && utf16_len(id) > 40 {
        let mut units = 0;
        return id
            .chars()
            .take_while(|c| {
                units += c.len_utf16();
                units <= 40
            })
            .collect();
    }
    id.to_owned()
}

fn image_url_part(mime_type: &str, data: &str) -> Value {
    json!({"type": "image_url", "image_url": {"url": format!("data:{mime_type};base64,{data}")}})
}

/// Convert a transcript into Chat Completions `messages`.
///
/// `grammar` maps grammar tool names to their input property; their calls replay as
/// `custom` tool calls.
pub fn convert_messages<S: BuildHasher>(
    model: &Model,
    context: &TranscriptContext,
    compat: &ResolvedOpenAICompletionsCompat,
    grammar: &HashMap<String, String, S>,
) -> Result<Vec<Value>> {
    let context = resolve_transcript(context, compat.supports_mid_convo_system_messages);
    let normalize = |id: &str, model: &Model, _source: &AssistantMessage| normalize_tool_call_id(id, &model.provider);
    let transformed = transform_messages(context.messages(), model, Some(&normalize));
    let transcript_tools = resolve_transcript_tools(
        context.messages(),
        compat.supports_mid_convo_system_messages && compat.supports_mid_convo_tool_additions,
    );
    let instruction_role = if model.reasoning && compat.supports_developer_role {
        "developer"
    } else {
        "system"
    };

    let mut params: Vec<Value> = Vec::new();
    let mut last_role: Option<&'static str> = None;
    let mut i = 0;
    while i < transformed.len() {
        let message = &transformed[i];
        // Some providers don't allow user messages directly after tool results.
        if compat.requires_assistant_after_tool_result
            && last_role == Some("toolResult")
            && matches!(message, Message::User(_))
        {
            params.push(json!({"role": "assistant", "content": ASSISTANT_BRIDGE_TEXT}));
        }
        match message {
            Message::System(system) => {
                if i > 0 && transcript_tools.anchors_additions {
                    let added = system.tools_added.as_deref().unwrap_or_default();
                    if !added.is_empty() {
                        // Kimi-style tool additions at this point of the conversation.
                        params.push(json!({"role": "system", "tools": convert_tools(added, compat)?}));
                    }
                }
                let text = if i == 0 {
                    get_system_message_text(system)
                } else {
                    render_system_message_update(system)
                };
                if !text.is_empty() {
                    params.push(json!({"role": instruction_role, "content": text}));
                }
            }
            Message::User(user) => match &user.content {
                UserContent::Text(text) => params.push(json!({"role": "user", "content": text})),
                UserContent::Blocks(blocks) => {
                    let content: Vec<Value> = blocks
                        .iter()
                        .filter_map(|block| match block {
                            UserBlock::Text(text) if text.text.is_empty() => None,
                            UserBlock::Text(text) => Some(json!({"type": "text", "text": text.text})),
                            UserBlock::Image(image) => Some(image_url_part(&image.mime_type, &image.data)),
                        })
                        .collect();
                    if content.is_empty() {
                        i += 1;
                        continue;
                    }
                    params.push(json!({"role": "user", "content": content}));
                }
            },
            Message::Assistant(assistant) => match convert_assistant(assistant, model, compat, grammar)? {
                Some(converted) => params.push(converted),
                None => {
                    i += 1;
                    continue;
                }
            },
            Message::ToolResult(_) => {
                let mut images: Vec<Value> = Vec::new();
                let mut j = i;
                while let Some(Message::ToolResult(result)) = transformed.get(j) {
                    params.push(convert_tool_result(result, compat));
                    if model.supports_images() {
                        images.extend(result.content.iter().filter_map(|block| match block {
                            UserBlock::Image(image) => Some(image_url_part(&image.mime_type, &image.data)),
                            UserBlock::Text(_) => None,
                        }));
                    }
                    j += 1;
                }
                i = j;
                if images.is_empty() {
                    last_role = Some("toolResult");
                } else {
                    if compat.requires_assistant_after_tool_result {
                        params.push(json!({"role": "assistant", "content": ASSISTANT_BRIDGE_TEXT}));
                    }
                    let mut content = vec![json!({"type": "text", "text": "Attached image(s) from tool result:"})];
                    content.extend(images);
                    params.push(json!({"role": "user", "content": content}));
                    last_role = Some("user");
                }
                continue;
            }
        }
        last_role = Some(message.role());
        i += 1;
    }
    Ok(params)
}

fn convert_tool_result(result: &ToolResultMessage, compat: &ResolvedOpenAICompletionsCompat) -> Value {
    let text = result
        .content
        .iter()
        .filter_map(|block| match block {
            UserBlock::Text(text) => Some(text.text.as_str()),
            UserBlock::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let has_images = result.content.iter().any(|block| matches!(block, UserBlock::Image(_)));
    // Always send text: a placeholder when the result has only images or nothing.
    let text = if !js_trim(&text).is_empty() {
        text
    } else if has_images {
        "(see attached image)".to_owned()
    } else {
        "(no tool output)".to_owned()
    };
    let mut message = json!({"role": "tool", "content": text, "tool_call_id": result.tool_call_id});
    // Some providers require the 'name' field in tool results.
    if compat.requires_tool_result_name && !result.tool_name.is_empty() {
        message["name"] = Value::String(result.tool_name.clone());
    }
    message
}

/// Whether a value is a well-formed OpenRouter-style `reasoning_details` entry.
fn is_reasoning_detail(detail: &Value) -> bool {
    let Some(object) = detail.as_object() else {
        return false;
    };
    let id_ok = matches!(object.get("id"), None | Some(Value::Null | Value::String(_)));
    let format_ok = matches!(object.get("format"), None | Some(Value::String(_)));
    let index_ok = matches!(object.get("index"), None | Some(Value::Number(_)));
    if !(id_ok && format_ok && index_ok) {
        return false;
    }
    match object.get("type").and_then(Value::as_str) {
        Some("reasoning.summary") => object.get("summary").is_some_and(Value::is_string),
        Some("reasoning.encrypted") => object.get("data").is_some_and(Value::is_string),
        Some("reasoning.text") => {
            object.get("text").is_some_and(Value::is_string)
                && matches!(object.get("signature"), None | Some(Value::Null | Value::String(_)))
        }
        _ => false,
    }
}

/// `reasoning_details` stored in a thinking signature.
fn parse_reasoning_details(signature: Option<&str>) -> Option<Vec<Value>> {
    let parsed: Value = serde_json::from_str(signature.filter(|s| !s.is_empty())?).ok()?;
    match parsed {
        Value::Array(details) if !details.is_empty() && details.iter().all(is_reasoning_detail) => Some(details),
        _ => None,
    }
}

/// An encrypted reasoning detail stored in a tool call's thought signature (older
/// transcripts).
fn parse_legacy_encrypted_reasoning_detail(signature: Option<&str>) -> Option<Value> {
    let parsed: Value = serde_json::from_str(signature.filter(|s| !s.is_empty())?).ok()?;
    let valid = is_reasoning_detail(&parsed)
        && parsed["type"] == "reasoning.encrypted"
        && parsed
            .get("id")
            .and_then(Value::as_str)
            .is_some_and(|id| !id.is_empty())
        && parsed
            .get("data")
            .and_then(Value::as_str)
            .is_some_and(|data| !data.is_empty());
    valid.then_some(parsed)
}

fn convert_assistant<S: BuildHasher>(
    message: &AssistantMessage,
    model: &Model,
    compat: &ResolvedOpenAICompletionsCompat,
    grammar: &HashMap<String, String, S>,
) -> Result<Option<Value>> {
    let mut assistant = JsonObject::new();
    assistant.insert("role".into(), Value::String("assistant".into()));
    // Some providers don't accept null content; they get an empty string instead.
    assistant.insert(
        "content".into(),
        if compat.requires_assistant_after_tool_result {
            Value::String(String::new())
        } else {
            Value::Null
        },
    );

    let text_parts: Vec<&str> = message
        .content
        .iter()
        .filter_map(AssistantBlock::as_text)
        .map(|text| text.text.as_str())
        .filter(|text| !js_trim(text).is_empty())
        .collect();
    let assistant_text = text_parts.concat();
    let thinking_blocks: Vec<_> = message.content.iter().filter_map(AssistantBlock::as_thinking).collect();
    let tool_calls: Vec<_> = message.tool_calls().collect();

    let signed_details = thinking_blocks
        .iter()
        .find_map(|block| parse_reasoning_details(block.thinking_signature.as_deref()));
    let legacy_details: Vec<Value> = tool_calls
        .iter()
        .filter_map(|call| parse_legacy_encrypted_reasoning_detail(call.thought_signature.as_deref()))
        .collect();
    let preserved_details = signed_details.or_else(|| (!legacy_details.is_empty()).then_some(legacy_details));

    let non_empty_thinking: Vec<_> = thinking_blocks
        .iter()
        .filter(|block| !js_trim(&block.thinking).is_empty())
        .collect();
    if !non_empty_thinking.is_empty() {
        if compat.requires_thinking_as_text {
            // Thinking as plain text (no tags, so the model does not mimic them).
            let thinking_text = non_empty_thinking
                .iter()
                .map(|block| block.thinking.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            let mut content = vec![json!({"type": "text", "text": thinking_text})];
            content.extend(text_parts.iter().map(|text| json!({"type": "text", "text": text})));
            assistant.insert("content".into(), Value::Array(content));
        } else {
            // Assistant content always goes out as a plain string: some models (DeepSeek
            // V3.2 via NVIDIA NIM) mirror a content-part array literally in their output.
            if !assistant_text.is_empty() {
                assistant.insert("content".into(), Value::String(assistant_text));
            }
            // reasoning_details is the structured alternative to a raw reasoning field.
            if preserved_details.is_none() {
                // The first thinking block's signature names the field (llama.cpp + gpt-oss).
                let mut signature = non_empty_thinking[0].thinking_signature.as_deref();
                if model.provider == "opencode-go" && signature == Some("reasoning") {
                    signature = Some("reasoning_content");
                }
                if let Some(field) = signature.filter(|field| REASONING_FIELDS.contains(field)) {
                    let text = non_empty_thinking
                        .iter()
                        .map(|block| block.thinking.as_str())
                        .collect::<Vec<_>>()
                        .join("\n");
                    assistant.insert(field.to_owned(), Value::String(text));
                }
            }
        }
    } else if !assistant_text.is_empty() {
        assistant.insert("content".into(), Value::String(assistant_text));
    }

    if !tool_calls.is_empty() {
        let mut converted = Vec::with_capacity(tool_calls.len());
        for call in &tool_calls {
            if let Some(property) = grammar.get(&call.name) {
                let input = get_grammar_tool_input(&call.name, &call.arguments, property)?;
                converted.push(json!({
                    "id": call.id,
                    "type": "custom",
                    "custom": {"name": call.name, "input": input},
                }));
            } else {
                converted.push(json!({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": Value::Object(call.arguments.clone()).to_string(),
                    },
                }));
            }
        }
        assistant.insert("tool_calls".into(), Value::Array(converted));
    }
    if let Some(details) = preserved_details {
        assistant.insert("reasoning_details".into(), Value::Array(details));
    }
    if compat.requires_reasoning_content_on_assistant_messages
        && model.reasoning
        && !assistant.contains_key("reasoning_content")
    {
        assistant.insert("reasoning_content".into(), Value::String(String::new()));
    }

    // Skip messages with neither content nor tool calls (e.g. aborted empty responses):
    // providers reject them.
    let has_content = match assistant.get("content") {
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(parts)) => !parts.is_empty(),
        _ => false,
    };
    if !has_content && !assistant.contains_key("tool_calls") {
        return Ok(None);
    }
    Ok(Some(Value::Object(assistant)))
}

/// Convert tools to Chat Completions tool definitions: OpenAI grammar `custom` tools when
/// supported, else `function` tools (strict when constrained sampling asks for it).
pub fn convert_tools(tools: &[Tool], compat: &ResolvedOpenAICompletionsCompat) -> Result<Vec<Value>> {
    let mut converted = Vec::with_capacity(tools.len());
    for tool in tools {
        if let Some(grammar) = resolve_grammar_constrained_sampling(tool, compat.supports_openai_grammar_tools)? {
            converted.push(json!({
                "type": "custom",
                "custom": {
                    "name": tool.name,
                    "description": tool.description,
                    "format": {
                        "type": "grammar",
                        "grammar": {"syntax": grammar.format.as_str(), "definition": grammar.definition},
                    },
                },
            }));
            continue;
        }
        let strict = resolve_json_schema_strict_sampling(tool, compat.supports_strict_mode, None)?;
        let mut function = Map::new();
        function.insert("name".into(), Value::String(tool.name.clone()));
        function.insert("description".into(), Value::String(tool.description.clone()));
        function.insert("parameters".into(), get_json_schema_tool_parameters(tool, strict)?);
        // Only include strict when the provider supports it; some reject unknown fields.
        if compat.supports_strict_mode {
            function.insert("strict".into(), Value::Bool(strict.unwrap_or(false)));
        }
        converted.push(json!({"type": "function", "function": function}));
    }
    Ok(converted)
}

// ---------------------------------------------------------------------------
// Usage and stop reasons
// ---------------------------------------------------------------------------

/// A non-negative token count; `null`, missing and non-numbers count as absent.
fn token_count(value: Option<&Value>) -> Option<u64> {
    let number = value?.as_f64()?;
    Some(if number > 0.0 { number as u64 } else { 0 })
}

/// Usage from a chunk's `usage` object.
fn parse_chunk_usage(raw: &Value, model: &Model) -> Usage {
    let prompt_tokens = token_count(raw.get("prompt_tokens")).unwrap_or(0);
    // `cached_tokens` is cache reads (hits). OpenAI/OpenRouter report it in
    // prompt_tokens_details, DeepSeek as prompt_cache_hit_tokens, Kimi at the top level.
    // cache_write_tokens (OpenRouter-compatible providers) is a separate write count, so
    // writes are not subtracted from cached_tokens.
    let cache_read = token_count(raw.pointer("/prompt_tokens_details/cached_tokens"))
        .or_else(|| token_count(raw.get("prompt_cache_hit_tokens")))
        .or_else(|| token_count(raw.get("cached_tokens")))
        .unwrap_or(0);
    let cache_write = token_count(raw.pointer("/prompt_tokens_details/cache_write_tokens")).unwrap_or(0);
    let mut usage = Usage {
        input: prompt_tokens.saturating_sub(cache_read).saturating_sub(cache_write),
        // completion_tokens already includes reasoning tokens.
        output: token_count(raw.get("completion_tokens")).unwrap_or(0),
        cache_read,
        cache_write,
        reasoning: Some(token_count(raw.pointer("/completion_tokens_details/reasoning_tokens")).unwrap_or(0)),
        ..Usage::default()
    };
    finalize_usage(model, &mut usage);
    usage
}

/// Map a `finish_reason` to a stop reason, with an error message for failures.
pub fn map_stop_reason(reason: &str) -> (StopReason, Option<String>) {
    match reason {
        "stop" | "end" => (StopReason::Stop, None),
        "length" => (StopReason::Length, None),
        "function_call" | "tool_calls" => (StopReason::ToolUse, None),
        other => (StopReason::Error, Some(format!("Provider finish_reason: {other}"))),
    }
}

// ---------------------------------------------------------------------------
// Stream state
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct CustomInput {
    property: String,
    buffer: GrammarToolInputJsonBuffer,
}

/// A tool call being streamed.
#[derive(Debug)]
struct ToolSlot {
    content_index: usize,
    stream_index: Option<i64>,
    /// Set for OpenAI grammar (`custom`) tool calls: the raw input is rendered as JSON
    /// arguments `{"<property>": "<input>"}`.
    custom: Option<CustomInput>,
}

/// Accumulates Chat Completions chunks into the builder.
#[derive(Debug, Default)]
struct StreamState {
    grammar: HashMap<String, String>,
    text: Option<usize>,
    thinking: Option<usize>,
    tools: Vec<ToolSlot>,
    by_index: HashMap<i64, usize>,
    by_id: HashMap<String, usize>,
    /// `reasoning_details` are replay metadata, not visible deltas: kept here and stored
    /// as the thinking signature when the block ends.
    reasoning_details: Option<Vec<JsonObject>>,
    has_finish_reason: bool,
}

/// `target[key] = source[key]`, where an absent source value deletes the key.
fn assign_from(target: &mut JsonObject, source: &JsonObject, key: &str) {
    match source.get(key) {
        Some(value) => {
            target.insert(key.to_owned(), value.clone());
        }
        None => {
            target.shift_remove(key);
        }
    }
}

fn is_nullish(value: Option<&Value>) -> bool {
    matches!(value, None | Some(Value::Null))
}

fn fill_missing_common_fields(target: &mut JsonObject, source: &JsonObject) {
    if is_nullish(target.get("id")) {
        assign_from(target, source, "id");
    }
    if !target.get("format").is_some_and(js_truthy) {
        assign_from(target, source, "format");
    }
    if is_nullish(target.get("index")) {
        assign_from(target, source, "index");
    }
}

/// Append a streamed detail: consecutive text and summary deltas merge into one entry,
/// encrypted entries stay discrete.
fn append_reasoning_detail(details: &mut Vec<JsonObject>, detail: &JsonObject) {
    let kind = detail.get("type").and_then(Value::as_str);
    if let Some(last) = details.last_mut() {
        let last_kind = last.get("type").and_then(Value::as_str).map(str::to_owned);
        let field = match (kind, last_kind.as_deref()) {
            (Some("reasoning.text"), Some("reasoning.text")) => Some("text"),
            (Some("reasoning.summary"), Some("reasoning.summary")) => Some("summary"),
            _ => None,
        };
        if let Some(field) = field {
            let addition = detail.get(field).and_then(Value::as_str).unwrap_or_default();
            if let Some(Value::String(text)) = last.get_mut(field) {
                text.push_str(addition);
            }
            if field == "text" && !last.get("signature").is_some_and(js_truthy) {
                assign_from(last, detail, "signature");
            }
            fill_missing_common_fields(last, detail);
            return;
        }
    }
    details.push(detail.clone());
}

impl StreamState {
    fn ensure_text(&mut self, builder: &mut MessageBuilder) -> usize {
        *self.text.get_or_insert_with(|| builder.start_text())
    }

    fn ensure_thinking(&mut self, builder: &mut MessageBuilder, signature: &str) -> usize {
        *self
            .thinking
            .get_or_insert_with(|| builder.start_thinking(String::new(), false, Some(signature.to_owned())))
    }

    /// Store the streamed reasoning details as the thinking block's signature.
    fn apply_reasoning_details(&self, builder: &mut MessageBuilder) {
        if let (Some(index), Some(details)) = (self.thinking, &self.reasoning_details) {
            let details: Vec<Value> = details.iter().cloned().map(Value::Object).collect();
            builder.set_thinking_signature(index, Some(Value::Array(details).to_string()));
        }
    }

    fn handle_chunk(&mut self, builder: &mut MessageBuilder, model: &Model, chunk: &Value) -> Result<()> {
        let Some(object) = chunk.as_object() else {
            return Ok(());
        };
        let output = &mut builder.output;
        // Every chunk of a completion carries the same id.
        if output.response_id.is_none()
            && let Some(id) = object.get("id").and_then(Value::as_str).filter(|id| !id.is_empty())
        {
            output.response_id = Some(id.to_owned());
        }
        if output.response_model.is_none()
            && let Some(name) = object
                .get("model")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty() && *name != model.id)
        {
            output.response_model = Some(name.to_owned());
        }
        let usage = object.get("usage").filter(|usage| js_truthy(usage));
        if let Some(usage) = usage {
            output.usage = parse_chunk_usage(usage, model);
        }

        let Some(choice) = object
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(Value::as_object)
        else {
            return Ok(());
        };
        // Some providers (e.g. Moonshot) report usage in choice.usage.
        if usage.is_none()
            && let Some(choice_usage) = choice.get("usage").filter(|usage| js_truthy(usage))
        {
            output.usage = parse_chunk_usage(choice_usage, model);
        }
        if let Some(reason) = choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .filter(|reason| !reason.is_empty())
        {
            output.raw_stop_reason = Some(reason.to_owned());
            let (stop_reason, error_message) = map_stop_reason(reason);
            output.stop_reason = stop_reason;
            if error_message.is_some() {
                output.error_message = error_message;
            }
            self.has_finish_reason = true;
        }

        let Some(delta) = choice.get("delta").and_then(Value::as_object) else {
            return Ok(());
        };
        // Reasoning before content: some proxies (LiteLLM) put both in one delta, and the
        // thinking block must precede the text block. The first non-empty reasoning field
        // wins (chutes.ai sends the same text in reasoning_content and reasoning).
        if let Some((field, text)) = REASONING_FIELDS.iter().find_map(|field| {
            delta
                .get(*field)
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .map(|text| (*field, text))
        }) {
            let signature = if model.provider == "opencode-go" && field == "reasoning" {
                "reasoning_content"
            } else {
                field
            };
            let index = self.ensure_thinking(builder, signature);
            builder.thinking_delta(index, text);
        }
        if let Some(details) = delta.get("reasoning_details").and_then(Value::as_array) {
            for detail in details.iter().filter(|detail| is_reasoning_detail(detail)) {
                let Some(detail) = detail.as_object() else { continue };
                self.ensure_thinking(builder, "");
                append_reasoning_detail(self.reasoning_details.get_or_insert_with(Vec::new), detail);
            }
        }
        if let Some(text) = delta
            .get("content")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
        {
            let index = self.ensure_text(builder);
            builder.text_delta(index, text);
        }
        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for call in calls.iter().filter_map(Value::as_object) {
                self.handle_tool_call_delta(builder, call)?;
            }
        }
        Ok(())
    }

    /// Find or start the tool call a delta belongs to. A delta is matched by its `index`
    /// first (some servers change the id mid-call), then by its `id`; a delta with neither
    /// goes to the only call in flight.
    fn tool_slot(&mut self, builder: &mut MessageBuilder, call: &JsonObject) -> Result<usize> {
        let stream_index = call.get("index").and_then(Value::as_i64).filter(|index| *index >= 0);
        let id = call.get("id").and_then(Value::as_str).filter(|id| !id.is_empty());
        let function = call.get("function").filter(|function| js_truthy(function));
        let custom = call.get("custom").filter(|custom| js_truthy(custom));
        let name = function
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .or_else(|| custom.and_then(|custom| custom.get("name")).and_then(Value::as_str))
            .unwrap_or_default();
        let is_custom = custom.is_some() && function.is_none();

        let mut slot = stream_index.and_then(|index| self.by_index.get(&index).copied());
        if slot.is_none() {
            slot = id.and_then(|id| self.by_id.get(id).copied());
        }
        if slot.is_none() && id.is_none() && stream_index.is_none() {
            match self.tools.len() {
                0 => {}
                1 => slot = Some(0),
                _ => {
                    return Err(Error::msg(
                        "Tool call delta without id or index while several tool calls are in progress",
                    ));
                }
            }
        }
        let slot = match slot {
            Some(slot) => slot,
            None => {
                // The "input" fallback is only for grammar calls to tools we do not know.
                let custom_input = is_custom.then(|| CustomInput {
                    property: self.grammar.get(name).cloned().unwrap_or_else(|| "input".into()),
                    buffer: GrammarToolInputJsonBuffer::default(),
                });
                let mut arguments = JsonObject::new();
                if let Some(custom) = &custom_input {
                    arguments.insert(custom.property.clone(), Value::String(String::new()));
                }
                let content_index =
                    builder.start_tool_call(id.unwrap_or_default().to_owned(), name.to_owned(), arguments);
                self.tools.push(ToolSlot {
                    content_index,
                    stream_index,
                    custom: custom_input,
                });
                let slot = self.tools.len() - 1;
                if let Some(index) = stream_index {
                    self.by_index.insert(index, slot);
                }
                slot
            }
        };
        if let Some(index) = stream_index
            && self.tools[slot].stream_index.is_none()
        {
            self.tools[slot].stream_index = Some(index);
            self.by_index.insert(index, slot);
        }
        if let Some(id) = id {
            self.by_id.insert(id.to_owned(), slot);
        }
        let content_index = self.tools[slot].content_index;
        if let Some(tool_call) = builder.tool_call_mut(content_index) {
            // A late id adopts the anonymous call; names are only set when non-empty.
            if tool_call.id.is_empty()
                && let Some(id) = id
            {
                id.clone_into(&mut tool_call.id);
            }
            if tool_call.name.is_empty() && !name.is_empty() {
                name.clone_into(&mut tool_call.name);
            }
            if is_custom && self.tools[slot].custom.is_none() {
                let property = self
                    .grammar
                    .get(&tool_call.name)
                    .cloned()
                    .unwrap_or_else(|| "input".into());
                tool_call.arguments = JsonObject::new();
                tool_call
                    .arguments
                    .insert(property.clone(), Value::String(String::new()));
                self.tools[slot].custom = Some(CustomInput {
                    property,
                    buffer: GrammarToolInputJsonBuffer::default(),
                });
            }
        }
        Ok(slot)
    }

    fn handle_tool_call_delta(&mut self, builder: &mut MessageBuilder, call: &JsonObject) -> Result<()> {
        let slot = self.tool_slot(builder, call)?;
        let content_index = self.tools[slot].content_index;
        let arguments = call
            .get("function")
            .and_then(|function| function.get("arguments"))
            .filter(|arguments| js_truthy(arguments));
        let custom_input = call
            .get("custom")
            .and_then(|custom| custom.get("input"))
            .filter(|input| js_truthy(input));
        if let Some(arguments) = arguments {
            let delta = arguments.as_str().map_or_else(|| arguments.to_string(), str::to_owned);
            builder.tool_call_delta(content_index, &delta);
        } else if let Some(input) = custom_input {
            let input = input.as_str().map_or_else(|| input.to_string(), str::to_owned);
            if let Some(delta) = self.append_custom_input(builder, slot, &input, false)? {
                builder.tool_call_delta(content_index, &delta);
            }
        }
        Ok(())
    }

    /// Grow a grammar call's raw input by `addition` (or close it) and return the JSON
    /// text to stream.
    fn append_custom_input(
        &mut self,
        builder: &mut MessageBuilder,
        slot: usize,
        addition: &str,
        close: bool,
    ) -> Result<Option<String>> {
        let content_index = self.tools[slot].content_index;
        let Some(custom) = self.tools[slot].custom.as_mut() else {
            return Ok(None);
        };
        let Some(tool_call) = builder.tool_call_mut(content_index) else {
            return Ok(None);
        };
        let current = tool_call
            .arguments
            .get(&custom.property)
            .and_then(Value::as_str)
            .unwrap_or_default();
        let next_input = format!("{current}{addition}");
        let delta = append_grammar_tool_input_json_delta(&mut custom.buffer, &custom.property, &next_input, close)?;
        tool_call.arguments = JsonObject::new();
        tool_call
            .arguments
            .insert(custom.property.clone(), Value::String(next_input));
        Ok(delta)
    }

    /// End every block in content order.
    fn finish(&mut self, builder: &mut MessageBuilder) -> Result<()> {
        for index in 0..builder.output.content.len() {
            match &builder.output.content[index] {
                AssistantBlock::Text(_) => builder.end_text(index),
                AssistantBlock::Thinking(_) => {
                    self.apply_reasoning_details(builder);
                    builder.end_thinking(index);
                }
                AssistantBlock::ToolCall(_) => {
                    if let Some(slot) = self.tools.iter().position(|slot| slot.content_index == index)
                        && let Some(delta) = self.append_custom_input(builder, slot, "", true)?
                    {
                        builder.tool_call_delta(index, &delta);
                    }
                    builder.end_tool_call(index);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
