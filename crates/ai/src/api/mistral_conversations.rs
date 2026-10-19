//! `mistral-conversations`: Mistral's native Chat Completions endpoint
//! (`POST {baseUrl}/v1/chat/completions` with `stream: true`), streamed over SSE.
//!
//! Port of pi-ai `api/mistral-conversations.ts`. As upstream, the payload is built in the
//! Mistral SDK's camelCase shape (what `on_payload` sees) and remapped to the snake_case wire
//! format just before it is sent. Design notes: `docs/design/api-mistral-conversations.md`.

use futures::StreamExt;
use reqwest::header::{HeaderName, HeaderValue};
use serde_json::{Map, Value, json};
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Duration;

use crate::api::common::{self, short_hash};
use crate::api::constrained_sampling::{get_json_schema_tool_parameters, resolve_json_schema_strict_sampling};
use crate::error::{Error, Result};
use crate::events::{AssistantMessageEventStream, MessageBuilder};
use crate::http;
use crate::json::parse_streaming_json;
use crate::model::{CacheRetention, Model, ModelThinkingLevel, calculate_cost};
use crate::options::{ApiOptions, MistralOptions, SimpleStreamOptions, StreamOptions, ToolChoice};
use crate::provider::ProviderStreams;
use crate::simple_options::build_base_options;
use crate::transcript::{get_current_tools, get_system_message_text, render_system_message_update, resolve_transcript};
use crate::transform::transform_messages;
use crate::types::{
    AssistantBlock, AssistantMessage, ImageContent, JsonObject, Message, ProviderHeaders, StopReason, Tool,
    TranscriptContext, UserBlock, UserContent,
};

const MISTRAL_TOOL_CALL_ID_LENGTH: usize = 9;
const MAX_MISTRAL_ERROR_BODY_CHARS: usize = 4000;
/// Upstream default of `timeoutMs` for this API. It bounds only the wait for response headers.
const DEFAULT_HEADER_TIMEOUT: Duration = Duration::from_secs(60);

/// The `mistral-conversations` API.
#[derive(Clone, Copy, Debug, Default)]
pub struct MistralConversationsApi;

impl ProviderStreams for MistralConversationsApi {
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

/// Stream a response from the native Mistral Chat Completions endpoint.
pub fn stream(model: &Model, context: &TranscriptContext, options: StreamOptions) -> AssistantMessageEventStream {
    let context = resolve_transcript(context, model.supports_mid_convo_system_messages());
    let request_model = model.clone();
    common::spawn_stream(model, options.cancel.clone(), move |mut builder| async move {
        let result = run(&request_model, &context, &options, &mut builder).await;
        (builder, result)
    })
}

/// Map provider-neutral simple options to [`MistralOptions`] and stream.
///
/// Models with a thinking level map send `reasoning_effort`; other reasoning models send
/// `prompt_mode: "reasoning"`.
#[allow(clippy::needless_pass_by_value)] // The signature is shared by every API module.
pub fn stream_simple(
    model: &Model,
    context: &TranscriptContext,
    options: SimpleStreamOptions,
) -> AssistantMessageEventStream {
    let mut base = build_base_options(model, context, &options);
    let SimpleStreamOptions {
        reasoning, tool_choice, ..
    } = options;
    let reasoning = reasoning
        .map(|level| model.clamp_thinking_level(ModelThinkingLevel::from(level)))
        .filter(|level| *level != ModelThinkingLevel::Off);
    let effort_map = if model.reasoning {
        model.thinking_level_map.as_ref()
    } else {
        None
    };
    let reasoning_effort = effort_map.and_then(|map| match reasoning {
        Some(level) => Some(map.get(&level).cloned().flatten().unwrap_or_else(|| "high".to_owned())),
        None => map.get(&ModelThinkingLevel::Off).cloned().flatten(),
    });
    let prompt_mode = (model.reasoning && effort_map.is_none() && reasoning.is_some()).then(|| "reasoning".to_owned());
    base.api = Some(ApiOptions::MistralConversations(MistralOptions {
        tool_choice: tool_choice.map(|choice| {
            Value::from(match choice {
                ToolChoice::Auto => "auto",
                ToolChoice::None => "none",
            })
        }),
        prompt_mode,
        reasoning_effort,
    }));
    stream(model, context, base)
}

async fn run(
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
    builder: &mut MessageBuilder,
) -> Result<()> {
    // Upstream requires an explicit API key; an `authorization` header alone is not enough.
    common::require_auth(model, options, &[])?;
    let api_key = options.api_key.clone().unwrap_or_default();
    let mistral = match &options.api {
        Some(ApiOptions::MistralConversations(mistral)) => mistral.clone(),
        _ => MistralOptions::default(),
    };

    let payload = build_chat_payload(model, context, options, &mistral)?;
    let payload = common::apply_on_payload(options, model, payload).await?;
    let response = request_stream(model, &payload, &api_key, options).await?;
    builder.start();
    consume_chat_stream(model, builder, response, options).await?;

    if options.is_cancelled() {
        return Err(Error::Aborted);
    }
    if builder.output.stop_reason == StopReason::Pending {
        return Err(Error::msg("Mistral stream ended without a finish reason"));
    }
    // `error` stop reasons end the stream through `spawn_stream` with `error_message`.
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Tool call ids
// ---------------------------------------------------------------------------------------------

/// Mistral accepts only 9-character alphanumeric tool call ids.
fn derive_mistral_tool_call_id(id: &str, attempt: u32) -> String {
    let normalized: String = id.chars().filter(char::is_ascii_alphanumeric).collect();
    if attempt == 0 && normalized.len() == MISTRAL_TOOL_CALL_ID_LENGTH {
        return normalized;
    }
    let seed_base = if normalized.is_empty() { id } else { &normalized };
    let seed = if attempt == 0 {
        seed_base.to_owned()
    } else {
        format!("{seed_base}:{attempt}")
    };
    short_hash(&seed)
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .take(MISTRAL_TOOL_CALL_ID_LENGTH)
        .collect()
}

/// Per-request normalizer: the same source id always maps to the same Mistral id, and two
/// source ids never share one (collisions retry with a salted seed).
#[derive(Debug, Default)]
struct ToolCallIdNormalizer {
    ids: HashMap<String, String>,
    owners: HashMap<String, String>,
}

impl ToolCallIdNormalizer {
    fn normalize(&mut self, id: &str) -> String {
        if let Some(existing) = self.ids.get(id) {
            return existing.clone();
        }
        let mut attempt = 0;
        loop {
            let candidate = derive_mistral_tool_call_id(id, attempt);
            if self.owners.get(&candidate).is_none_or(|owner| owner == id) {
                self.ids.insert(id.to_owned(), candidate.clone());
                self.owners.insert(candidate.clone(), id.to_owned());
                return candidate;
            }
            attempt += 1;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Request payload (SDK camelCase shape)
// ---------------------------------------------------------------------------------------------

fn should_use_prompt_caching(options: &StreamOptions) -> Option<&str> {
    if options.cache_retention == Some(CacheRetention::None) {
        return None;
    }
    options.session_id.as_deref().filter(|id| !id.is_empty())
}

fn non_empty(value: Option<&String>) -> Option<&String> {
    value.filter(|value| !value.is_empty())
}

/// Build the payload `on_payload` sees: the Mistral SDK's camelCase request shape.
fn build_chat_payload(
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
    mistral: &MistralOptions,
) -> Result<Value> {
    let normalizer = RefCell::new(ToolCallIdNormalizer::default());
    let normalize = |id: &str, _: &Model, _: &AssistantMessage| normalizer.borrow_mut().normalize(id);
    let messages = transform_messages(context.messages(), model, Some(&normalize));

    let mut payload = Map::new();
    payload.insert("model".into(), Value::from(model.id.as_str()));
    payload.insert("stream".into(), Value::Bool(true));
    payload.insert(
        "messages".into(),
        Value::Array(to_chat_messages(&messages, model.supports_images())),
    );
    let tools = get_current_tools(context.messages());
    if !tools.is_empty() {
        payload.insert("tools".into(), Value::Array(to_function_tools(&tools)?));
    }
    if let Some(temperature) = options.temperature {
        payload.insert("temperature".into(), json!(temperature));
    }
    if let Some(max_tokens) = options.max_tokens {
        payload.insert("maxTokens".into(), json!(max_tokens));
    }
    if let Some(choice) = mistral.tool_choice.as_ref().and_then(map_tool_choice) {
        payload.insert("toolChoice".into(), choice);
    }
    if let Some(mode) = non_empty(mistral.prompt_mode.as_ref()) {
        payload.insert("promptMode".into(), Value::from(mode.as_str()));
    }
    if let Some(effort) = non_empty(mistral.reasoning_effort.as_ref()) {
        payload.insert("reasoningEffort".into(), Value::from(effort.as_str()));
    }
    if let Some(session_id) = should_use_prompt_caching(options) {
        payload.insert("promptCacheKey".into(), Value::from(session_id));
    }
    Ok(Value::Object(payload))
}

/// `auto`/`none`/`any`/`required` pass through; a function choice is reduced to its name.
fn map_tool_choice(choice: &Value) -> Option<Value> {
    match choice {
        Value::Null | Value::Bool(false) => None,
        Value::String(text) if text.is_empty() => None,
        Value::Object(object) => match object.get("function").and_then(|function| function.get("name")) {
            Some(name) => Some(json!({"type": "function", "function": {"name": name}})),
            // Upstream would fail on a malformed object; it is passed through instead.
            None => Some(choice.clone()),
        },
        other => Some(other.clone()),
    }
}

fn image_url(image: &ImageContent) -> String {
    format!("data:{};base64,{}", image.mime_type, image.data)
}

fn to_chat_messages(messages: &[Message], supports_images: bool) -> Vec<Value> {
    let mut result = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        match message {
            Message::System(system) => {
                let text = if index == 0 {
                    get_system_message_text(system)
                } else {
                    render_system_message_update(system)
                };
                if !text.is_empty() {
                    result.push(json!({"role": "system", "content": text}));
                }
            }
            Message::User(user) => {
                let blocks = match &user.content {
                    UserContent::Text(text) => {
                        result.push(json!({"role": "user", "content": text}));
                        continue;
                    }
                    UserContent::Blocks(blocks) => blocks,
                };
                let had_images = blocks.iter().any(|block| matches!(block, UserBlock::Image(_)));
                let content: Vec<Value> = blocks
                    .iter()
                    .filter_map(|block| match block {
                        UserBlock::Text(text) => Some(json!({"type": "text", "text": text.text})),
                        UserBlock::Image(image) if supports_images => {
                            Some(json!({"type": "image_url", "imageUrl": image_url(image)}))
                        }
                        UserBlock::Image(_) => None,
                    })
                    .collect();
                if !content.is_empty() {
                    result.push(json!({"role": "user", "content": content}));
                } else if had_images && !supports_images {
                    result.push(json!({"role": "user", "content": "(image omitted: model does not support images)"}));
                }
            }
            Message::Assistant(assistant) => {
                let mut content = Vec::new();
                let mut tool_calls = Vec::new();
                for block in &assistant.content {
                    match block {
                        AssistantBlock::Text(text) => {
                            if !text.text.trim().is_empty() {
                                content.push(json!({"type": "text", "text": text.text}));
                            }
                        }
                        AssistantBlock::Thinking(thinking) => {
                            if !thinking.thinking.trim().is_empty() {
                                content.push(json!({
                                    "type": "thinking",
                                    "thinking": [{"type": "text", "text": thinking.thinking}],
                                }));
                            }
                        }
                        AssistantBlock::ToolCall(call) => {
                            let arguments = Value::Object(call.arguments.clone()).to_string();
                            tool_calls.push(json!({
                                "id": call.id,
                                "type": "function",
                                "function": {"name": call.name, "arguments": arguments},
                                "index": 0,
                            }));
                        }
                    }
                }
                if content.is_empty() && tool_calls.is_empty() {
                    continue;
                }
                let mut wire = Map::new();
                wire.insert("role".into(), Value::from("assistant"));
                wire.insert("prefix".into(), Value::Bool(false));
                if !content.is_empty() {
                    wire.insert("content".into(), Value::Array(content));
                }
                if !tool_calls.is_empty() {
                    wire.insert("toolCalls".into(), Value::Array(tool_calls));
                }
                result.push(Value::Object(wire));
            }
            Message::ToolResult(tool_result) => {
                let text = tool_result
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        UserBlock::Text(text) => Some(text.text.as_str()),
                        UserBlock::Image(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let has_images = tool_result
                    .content
                    .iter()
                    .any(|block| matches!(block, UserBlock::Image(_)));
                let mut content = vec![json!({
                    "type": "text",
                    "text": build_tool_result_text(&text, has_images, supports_images, tool_result.is_error),
                })];
                if supports_images {
                    content.extend(tool_result.content.iter().filter_map(|block| match block {
                        UserBlock::Image(image) => Some(json!({"type": "image_url", "imageUrl": image_url(image)})),
                        UserBlock::Text(_) => None,
                    }));
                }
                result.push(json!({
                    "role": "tool",
                    "toolCallId": tool_result.tool_call_id,
                    "name": tool_result.tool_name,
                    "content": content,
                }));
            }
        }
    }
    result
}

fn build_tool_result_text(text: &str, has_images: bool, supports_images: bool, is_error: bool) -> String {
    let trimmed = text.trim();
    let error_prefix = if is_error { "[tool error] " } else { "" };
    if !trimmed.is_empty() {
        let image_suffix = if has_images && !supports_images {
            "\n[tool image omitted: model does not support images]"
        } else {
            ""
        };
        return format!("{error_prefix}{trimmed}{image_suffix}");
    }
    let body = if has_images {
        if supports_images {
            "(see attached image)"
        } else {
            "(image omitted: model does not support images)"
        }
    } else {
        "(no tool output)"
    };
    format!("{error_prefix}{body}")
}

fn to_function_tools(tools: &[Tool]) -> Result<Vec<Value>> {
    tools
        .iter()
        .map(|tool| {
            // Mistral supports strict mode, so a `require` tool fails only when its schema is
            // outside the strict subset.
            let strict = resolve_json_schema_strict_sampling(tool, true, None)?;
            let parameters = get_json_schema_tool_parameters(tool, strict)?;
            let strict = strict.unwrap_or(false);
            Ok(json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                    "strict": strict,
                },
            }))
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------------------------

const WIRE_PAYLOAD_KEYS: [(&str, &str); 12] = [
    ("topP", "top_p"),
    ("maxTokens", "max_tokens"),
    ("randomSeed", "random_seed"),
    ("responseFormat", "response_format"),
    ("toolChoice", "tool_choice"),
    ("presencePenalty", "presence_penalty"),
    ("frequencyPenalty", "frequency_penalty"),
    ("parallelToolCalls", "parallel_tool_calls"),
    ("reasoningEffort", "reasoning_effort"),
    ("promptMode", "prompt_mode"),
    ("promptCacheKey", "prompt_cache_key"),
    ("safePrompt", "safe_prompt"),
];

const WIRE_CHUNK_KEYS: [(&str, &str); 6] = [
    ("imageUrl", "image_url"),
    ("documentUrl", "document_url"),
    ("documentName", "document_name"),
    ("fileId", "file_id"),
    ("referenceIds", "reference_ids"),
    ("inputAudio", "input_audio"),
];

fn remap(record: &mut Map<String, Value>, source: &str, target: &str) {
    if let Some(value) = record.remove(source) {
        record.insert(target.to_owned(), value);
    }
}

/// Rename the SDK's camelCase fields to the snake_case wire names, as the Mistral SDK does.
fn to_wire_payload(payload: Value) -> Value {
    let Value::Object(mut wire) = payload else {
        return payload;
    };
    for (source, target) in WIRE_PAYLOAD_KEYS {
        remap(&mut wire, source, target);
    }
    if let Some(Value::Array(messages)) = wire.get_mut("messages") {
        for message in messages {
            to_wire_message(message);
        }
    }
    if let Some(Value::Object(format)) = wire.get_mut("response_format") {
        remap(format, "jsonSchema", "json_schema");
        if let Some(Value::Object(schema)) = format.get_mut("json_schema") {
            remap(schema, "schemaDefinition", "schema");
        }
    }
    Value::Object(wire)
}

fn to_wire_message(message: &mut Value) {
    let Value::Object(message) = message else { return };
    remap(message, "toolCalls", "tool_calls");
    remap(message, "toolCallId", "tool_call_id");
    if let Some(Value::Array(chunks)) = message.get_mut("content") {
        for chunk in chunks {
            if let Value::Object(chunk) = chunk {
                for (source, target) in WIRE_CHUNK_KEYS {
                    remap(chunk, source, target);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------------------------

fn chat_completions_url(base_url: &str) -> Result<url::Url> {
    let mut url = url::Url::parse(base_url).map_err(|error| Error::msg(format!("Invalid URL: {base_url}: {error}")))?;
    let path = format!("{}/v1/chat/completions", url.path().trim_end_matches('/'));
    url.set_path(&path);
    url.set_query(None);
    url.set_fragment(None);
    Ok(url)
}

fn has_header_override(options: &StreamOptions, target: &str) -> bool {
    options
        .headers
        .iter()
        .flatten()
        .any(|(name, _)| name.eq_ignore_ascii_case(target))
}

/// Default headers, then `options.headers` (model headers reach the API merged into them by
/// [`crate::Models`]). `x-affinity` carries the session id unless a header source names it.
fn build_mistral_headers(api_key: &str, options: &StreamOptions) -> Result<reqwest::header::HeaderMap> {
    let authorization = format!("Bearer {api_key}");
    let user_agent = http::user_agent();
    let defaults = http::headers_of([
        ("User-Agent", user_agent.as_str()),
        ("accept", "text/event-stream"),
        ("authorization", authorization.as_str()),
        ("content-type", "application/json"),
    ]);
    let empty = ProviderHeaders::default();
    let mut headers = http::build_headers([&defaults, options.headers.as_ref().unwrap_or(&empty)])?;
    if let Some(session_id) = should_use_prompt_caching(options)
        && !has_header_override(options, "x-affinity")
    {
        let value = HeaderValue::from_str(session_id).map_err(|_| Error::msg("Invalid value for header x-affinity"))?;
        headers.insert(HeaderName::from_static("x-affinity"), value);
    }
    Ok(headers)
}

/// Send the request once. As upstream, there are no client-side retries, and the timeout
/// covers only the wait for response headers: long streams are never cut off by it.
// Not ported: upstream's `options.fetch` override; `StreamOptions::http_client` plays that role.
async fn request_stream(
    model: &Model,
    payload: &Value,
    api_key: &str,
    options: &StreamOptions,
) -> Result<reqwest::Response> {
    let url = chat_completions_url(&model.base_url)?;
    let headers = build_mistral_headers(api_key, options)?;
    let body = serde_json::to_vec(&to_wire_payload(payload.clone()))?;
    let request = http::client(options).post(url).headers(headers).body(body);
    let timeout = options.timeout.unwrap_or(DEFAULT_HEADER_TIMEOUT);
    let response = http::or_cancel(options.cancel.as_ref(), async {
        match tokio::time::timeout(timeout, request.send()).await {
            Ok(response) => response.map_err(Error::from),
            Err(_) => Err(Error::msg(format!(
                "Mistral response headers timed out after {}ms",
                timeout.as_millis()
            ))),
        }
    })
    .await?;

    common::notify_response(options, model, &response).await?;

    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let retry_after_ms = http::retry_after(response.headers()).map(|delay| delay.as_millis() as u64);
    let body = http::or_cancel(options.cancel.as_ref(), async {
        response.text().await.map_err(Error::from)
    })
    .await?;
    let body = body.trim();
    let detail = if body.is_empty() {
        status.canonical_reason().map_or_else(
            || format!("Request failed with status {}", status.as_u16()),
            str::to_owned,
        )
    } else {
        http::truncate_error_text(body, MAX_MISTRAL_ERROR_BODY_CHARS)
    };
    Err(Error::Status {
        status: status.as_u16(),
        message: format!("Mistral API error ({}): {detail}", status.as_u16()),
        retry_after_ms,
    })
}

// ---------------------------------------------------------------------------------------------
// Stream consumption
// ---------------------------------------------------------------------------------------------

/// Map a Mistral `finish_reason` to a stop reason and, for failures, an error message.
fn map_chat_stop_reason(reason: &str) -> (StopReason, Option<String>) {
    match reason {
        "stop" => (StopReason::Stop, None),
        "length" | "model_length" => (StopReason::Length, None),
        "tool_calls" => (StopReason::ToolUse, None),
        // Mistral reports transient server failures this way; "server error" makes the
        // message retryable.
        "error" => (
            StopReason::Error,
            Some("Provider stopped with: error (server error)".into()),
        ),
        other => (StopReason::Error, Some(format!("Provider stopped with: {other}"))),
    }
}

/// JavaScript `value || 0` for a token count.
fn token_count(value: Option<&Value>) -> u64 {
    value
        .and_then(Value::as_f64)
        .filter(|count| count.is_finite() && *count > 0.0)
        .map_or(0, |count| count as u64)
}

/// Cached prompt tokens from any of the usage shapes Mistral and its SDK use, capped at the
/// prompt tokens.
fn cached_prompt_tokens(usage: &Value, prompt_tokens: u64) -> u64 {
    let candidates = [
        usage.pointer("/promptTokensDetails/cachedTokens"),
        usage.pointer("/prompt_tokens_details/cached_tokens"),
        usage.pointer("/promptTokenDetails/cachedTokens"),
        usage.pointer("/prompt_token_details/cached_tokens"),
        usage.get("numCachedTokens"),
        usage.get("num_cached_tokens"),
    ];
    let raw = candidates.into_iter().flatten().find(|value| !value.is_null());
    let cached = raw
        .and_then(Value::as_f64)
        .filter(|count| count.is_finite())
        .unwrap_or(0.0)
        .max(0.0);
    (cached as u64).min(prompt_tokens)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpenBlock {
    Text(usize),
    Thinking(usize),
}

#[derive(Debug)]
struct ToolSlot {
    /// Upstream key: the call's `index`, else its id.
    key: String,
    content_index: usize,
    /// The provider id, when the call had a real one.
    provider_id: Option<String>,
}

#[derive(Debug, Default)]
struct ChatStreamState {
    current: Option<OpenBlock>,
    tool_slots: Vec<ToolSlot>,
}

impl ChatStreamState {
    fn finish_current(&mut self, builder: &mut MessageBuilder) {
        match self.current.take() {
            Some(OpenBlock::Text(index)) => builder.end_text(index),
            Some(OpenBlock::Thinking(index)) => builder.end_thinking(index),
            None => {}
        }
    }

    fn append_text(&mut self, builder: &mut MessageBuilder, text: &str) {
        let index = match self.current {
            Some(OpenBlock::Text(index)) => index,
            _ => {
                self.finish_current(builder);
                let index = builder.start_text();
                self.current = Some(OpenBlock::Text(index));
                index
            }
        };
        builder.text_delta(index, text);
    }

    fn append_thinking(&mut self, builder: &mut MessageBuilder, text: &str) {
        let index = match self.current {
            Some(OpenBlock::Thinking(index)) => index,
            _ => {
                self.finish_current(builder);
                let index = builder.start_thinking(String::new(), false, None);
                self.current = Some(OpenBlock::Thinking(index));
                index
            }
        };
        builder.thinking_delta(index, text);
    }

    fn handle_chunk(&mut self, model: &Model, builder: &mut MessageBuilder, chunk: &Value) {
        // Keep the first non-empty completion id as the response id.
        if builder.output.response_id.is_none()
            && let Some(id) = chunk.get("id").and_then(Value::as_str).filter(|id| !id.is_empty())
        {
            builder.output.response_id = Some(id.to_owned());
        }

        if let Some(usage @ Value::Object(raw)) = chunk.get("usage") {
            let prompt_tokens = token_count(raw.get("prompt_tokens"));
            let cached = cached_prompt_tokens(usage, prompt_tokens);
            let output = &mut builder.output.usage;
            output.input = prompt_tokens.saturating_sub(cached);
            output.output = token_count(raw.get("completion_tokens"));
            output.cache_read = cached;
            output.cache_write = 0;
            let total = token_count(raw.get("total_tokens"));
            output.total_tokens = if total > 0 {
                total
            } else {
                output.input + output.output + output.cache_read + output.cache_write
            };
            calculate_cost(&model.cost, output);
        }

        let Some(choice) = chunk
            .get("choices")
            .and_then(|choices| choices.get(0))
            .filter(|choice| !choice.is_null())
        else {
            return;
        };

        if let Some(reason) = choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .filter(|reason| !reason.is_empty())
        {
            builder.output.raw_stop_reason = Some(reason.to_owned());
            let (stop_reason, error_message) = map_chat_stop_reason(reason);
            builder.output.stop_reason = stop_reason;
            if error_message.is_some() {
                builder.output.error_message = error_message;
            }
        }

        let Some(delta) = choice.get("delta") else { return };
        match delta.get("content") {
            Some(item @ Value::String(_)) => self.handle_content_item(builder, item),
            Some(Value::Array(items)) => {
                for item in items {
                    self.handle_content_item(builder, item);
                }
            }
            _ => {}
        }

        if let Some(Value::Array(tool_calls)) = delta.get("tool_calls") {
            for tool_call in tool_calls {
                self.handle_tool_call(builder, tool_call);
            }
        }
    }

    fn handle_content_item(&mut self, builder: &mut MessageBuilder, item: &Value) {
        // GLM models on Mistral send empty content deltas around thinking and tool calls.
        // Opening a block for them splits thinking into several blocks, which Mistral rejects
        // on replay, so empty deltas are skipped.
        match item {
            Value::String(text) if !text.is_empty() => self.append_text(builder, text),
            Value::Object(object) => match object.get("type").and_then(Value::as_str) {
                Some("thinking") => {
                    let Some(Value::Array(parts)) = object.get("thinking") else {
                        return;
                    };
                    let text: String = parts
                        .iter()
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .collect();
                    if !text.is_empty() {
                        self.append_thinking(builder, &text);
                    }
                }
                Some("text") => {
                    if let Some(text) = object
                        .get("text")
                        .and_then(Value::as_str)
                        .filter(|text| !text.is_empty())
                    {
                        self.append_text(builder, text);
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn handle_tool_call(&mut self, builder: &mut MessageBuilder, tool_call: &Value) {
        let Value::Object(call) = tool_call else { return };
        self.finish_current(builder);

        // Mistral sends the literal string "null" as the id of continuation chunks.
        let provider_id = call
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty() && *id != "null");
        let index = call.get("index").filter(|index| !index.is_null());
        let call_id = match provider_id {
            Some(id) => id.to_owned(),
            None => {
                let index = index.map_or_else(|| "0".to_owned(), Value::to_string);
                derive_mistral_tool_call_id(&format!("toolcall:{index}"), 0)
            }
        };
        let key = match index {
            Some(index) => format!("index:{index}"),
            None => format!("id:{call_id}"),
        };
        let function = call.get("function");
        let name = function
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let arguments_delta = match function.and_then(|function| function.get("arguments")) {
            Some(Value::String(arguments)) => arguments.clone(),
            None | Some(Value::Null) => "{}".to_owned(),
            Some(other) => other.to_string(),
        };

        // A different real id at the same index is a new call, not a continuation.
        let existing = self
            .tool_slots
            .iter()
            .rposition(|slot| slot.key == key)
            .filter(|&slot| match (&self.tool_slots[slot].provider_id, provider_id) {
                (Some(existing), Some(incoming)) => existing == incoming,
                _ => true,
            });
        let content_index = if let Some(slot) = existing {
            let content_index = self.tool_slots[slot].content_index;
            if let Some(call) = builder.tool_call_mut(content_index)
                && call.name.is_empty()
                && !name.is_empty()
            {
                name.clone_into(&mut call.name);
            }
            content_index
        } else {
            let content_index = builder.start_tool_call(call_id, name.to_owned(), JsonObject::new());
            self.tool_slots.push(ToolSlot {
                key,
                content_index,
                provider_id: provider_id.map(str::to_owned),
            });
            content_index
        };

        builder.tool_call_delta(content_index, &arguments_delta);
        let partial = builder.tool_call_json(content_index).map(parse_streaming_json);
        if let (Some(arguments), Some(call)) = (partial, builder.tool_call_mut(content_index)) {
            call.arguments = arguments;
        }
    }

    fn finish(&mut self, builder: &mut MessageBuilder) {
        self.finish_current(builder);
        for slot in &self.tool_slots {
            builder.end_tool_call(slot.content_index);
        }
    }
}

async fn consume_chat_stream(
    model: &Model,
    builder: &mut MessageBuilder,
    response: reqwest::Response,
    options: &StreamOptions,
) -> Result<()> {
    let mut state = ChatStreamState::default();
    let mut events = std::pin::pin!(http::sse_events(response, options.cancel.clone()));
    while let Some(event) = events.next().await {
        if builder.is_closed() {
            return Err(Error::Aborted);
        }
        let event = event?;
        let data = event.data.trim();
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            break;
        }
        let chunk: Value = serde_json::from_str(data)?;
        if !chunk.get("choices").is_some_and(Value::is_array) {
            return Err(Error::msg("Invalid Mistral streaming event"));
        }
        common::observe_provider_event(options, model, &chunk).await?;
        state.handle_chunk(model, builder, &chunk);
    }
    state.finish(builder);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ConstrainedSampling, StrictMode};
    use crate::types::{TextContent, ThinkingContent, ToolCall, ToolResultMessage, UserMessage};
    use pretty_assertions::assert_eq;

    #[test]
    fn tool_call_ids_match_upstream() {
        // Vector from pi-py's port of the upstream normalizer test.
        assert_eq!(derive_mistral_tool_call_id("call_abc123|fc_xyz", 0), "nrregu2qr");
        assert_eq!(derive_mistral_tool_call_id("abc123456", 0), "abc123456");
        assert_eq!(derive_mistral_tool_call_id("a-b-c-1-2-3-4-5-6", 0), "abc123456");
        assert!(derive_mistral_tool_call_id("x", 0).len() <= MISTRAL_TOOL_CALL_ID_LENGTH);
    }

    #[test]
    fn normalizer_is_stable_and_collision_free() {
        let mut normalizer = ToolCallIdNormalizer::default();
        let a = normalizer.normalize("abc-123-456");
        let b = normalizer.normalize("abc_123_456");
        assert_eq!(a, "abc123456");
        assert_ne!(a, b);
        assert_eq!(b, derive_mistral_tool_call_id("abc_123_456", 1));
        assert_eq!(normalizer.normalize("abc-123-456"), a);
        assert!(b.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn stop_reasons_map_like_upstream() {
        assert_eq!(map_chat_stop_reason("stop"), (StopReason::Stop, None));
        assert_eq!(map_chat_stop_reason("length"), (StopReason::Length, None));
        assert_eq!(map_chat_stop_reason("model_length"), (StopReason::Length, None));
        assert_eq!(map_chat_stop_reason("tool_calls"), (StopReason::ToolUse, None));
        assert_eq!(
            map_chat_stop_reason("error"),
            (
                StopReason::Error,
                Some("Provider stopped with: error (server error)".into())
            )
        );
        assert_eq!(
            map_chat_stop_reason("weird"),
            (StopReason::Error, Some("Provider stopped with: weird".into()))
        );
    }

    #[test]
    fn tool_choice_mapping() {
        assert_eq!(map_tool_choice(&json!("any")), Some(json!("any")));
        assert_eq!(map_tool_choice(&json!("")), None);
        assert_eq!(map_tool_choice(&Value::Null), None);
        assert_eq!(
            map_tool_choice(&json!({"type": "function", "function": {"name": "f", "extra": 1}, "x": 2})),
            Some(json!({"type": "function", "function": {"name": "f"}}))
        );
    }

    #[test]
    fn tool_result_texts() {
        assert_eq!(build_tool_result_text(" ok ", false, true, false), "ok");
        assert_eq!(
            build_tool_result_text("ok", true, false, true),
            "[tool error] ok\n[tool image omitted: model does not support images]"
        );
        assert_eq!(build_tool_result_text("", true, true, false), "(see attached image)");
        assert_eq!(
            build_tool_result_text("", true, true, true),
            "[tool error] (see attached image)"
        );
        assert_eq!(
            build_tool_result_text("", true, false, false),
            "(image omitted: model does not support images)"
        );
        assert_eq!(
            build_tool_result_text("  ", false, false, true),
            "[tool error] (no tool output)"
        );
    }

    #[test]
    fn wire_payload_renames_sdk_fields() {
        let payload = json!({
            "model": "m",
            "maxTokens": 5,
            "promptCacheKey": "s",
            "responseFormat": {"type": "json_schema", "jsonSchema": {"name": "r", "schemaDefinition": {"type": "object"}}},
            "messages": [
                {"role": "tool", "toolCallId": "abc123456", "name": "t", "content": [{"type": "image_url", "imageUrl": "data:x"}]},
                {"role": "assistant", "prefix": false, "toolCalls": []},
                "not a message",
            ],
        });
        assert_eq!(
            to_wire_payload(payload),
            json!({
                "model": "m",
                "max_tokens": 5,
                "prompt_cache_key": "s",
                "response_format": {"type": "json_schema", "json_schema": {"name": "r", "schema": {"type": "object"}}},
                "messages": [
                    {"role": "tool", "tool_call_id": "abc123456", "name": "t", "content": [{"type": "image_url", "image_url": "data:x"}]},
                    {"role": "assistant", "prefix": false, "tool_calls": []},
                    "not a message",
                ],
            })
        );
    }

    #[test]
    fn urls_keep_base_path_and_drop_query() {
        assert_eq!(
            chat_completions_url("https://api.mistral.ai").unwrap().as_str(),
            "https://api.mistral.ai/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://proxy.example/mistral/?x=1#f")
                .unwrap()
                .as_str(),
            "https://proxy.example/mistral/v1/chat/completions"
        );
        assert!(chat_completions_url("not a url").is_err());
    }

    #[test]
    fn strict_tools_follow_constrained_sampling() {
        let mut tool = Tool::new(
            "t",
            "d",
            json!({"type": "object", "properties": {"a": {"type": "string"}}}),
        );
        let plain = to_function_tools(std::slice::from_ref(&tool)).unwrap();
        assert_eq!(plain[0]["function"]["strict"], json!(false));
        assert_eq!(plain[0]["function"]["parameters"], tool.parameters);

        tool.constrained_sampling = Some(ConstrainedSampling::JsonSchema {
            strict: StrictMode::Prefer,
        });
        let strict = to_function_tools(std::slice::from_ref(&tool)).unwrap();
        assert_eq!(strict[0]["function"]["strict"], json!(true));
        assert_eq!(
            strict[0]["function"]["parameters"]["additionalProperties"],
            json!(false)
        );

        tool.parameters = json!({"type": "object", "oneOf": []});
        let fallback = to_function_tools(std::slice::from_ref(&tool)).unwrap();
        assert_eq!(fallback[0]["function"]["strict"], json!(false));

        tool.constrained_sampling = Some(ConstrainedSampling::JsonSchema {
            strict: StrictMode::Require,
        });
        assert_eq!(
            to_function_tools(&[tool]).unwrap_err().to_string(),
            "Tool \"t\" requires JSON-schema constrained sampling, but oneOf schemas are unsupported."
        );
    }

    #[test]
    fn converts_messages_for_replay() {
        let mut assistant = AssistantMessage::empty("mistral-conversations", "mistral", "m");
        assistant.content = vec![
            AssistantBlock::Thinking(ThinkingContent::new("  ")),
            AssistantBlock::Text(TextContent::new("")),
        ];
        let messages = vec![
            Message::user("plain"),
            Message::User(UserMessage::new(UserContent::Blocks(vec![UserBlock::Image(
                ImageContent {
                    data: "AA==".into(),
                    mime_type: "image/png".into(),
                },
            )]))),
            Message::Assistant(assistant),
            Message::ToolResult(ToolResultMessage::text(
                &ToolCall {
                    id: "abc123456".into(),
                    name: "t".into(),
                    ..ToolCall::default()
                },
                "",
                true,
            )),
        ];
        assert_eq!(
            to_chat_messages(&messages, false),
            vec![
                json!({"role": "user", "content": "plain"}),
                json!({"role": "user", "content": "(image omitted: model does not support images)"}),
                json!({"role": "tool", "toolCallId": "abc123456", "name": "t",
                       "content": [{"type": "text", "text": "[tool error] (no tool output)"}]}),
            ]
        );
    }
}
