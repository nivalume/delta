//! Anthropic Messages API (`anthropic-messages`).
//!
//! Port of pi-ai `api/anthropic-messages.ts`. Upstream sends requests through the Anthropic
//! SDK (`client.beta.messages.create(...).asResponse()`) and parses the SSE body itself; this
//! module writes the same HTTP request by hand: `POST {baseUrl}/v1/messages?beta=true` with the
//! headers the SDK would send, and the same response parsing.
//!
//! Design notes: `docs/design/api-anthropic-messages.md`.

use futures::StreamExt;
use serde_json::{Map, Value, json};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::api::common::{self, spawn_stream};
use crate::api::constrained_sampling::{get_json_schema_tool_parameters, resolve_json_schema_strict_sampling};
use crate::api::github_copilot_headers::{build_copilot_dynamic_headers, has_copilot_vision_input};
use crate::compat::{AnthropicAllowedFallbackModel, ModelCompat, SessionAffinityFormat};
use crate::error::{Error, Result};
use crate::events::{AssistantMessageEventStream, MessageBuilder};
use crate::http::{self, SseEvent};
use crate::json::{parse_json_with_repair, parse_streaming_json};
use crate::model::{CacheRetention, Model, ModelCost, ModelThinkingLevel, ThinkingLevel, calculate_cost};
use crate::options::{
    AnthropicEffort, AnthropicOptions, AnthropicToolChoice, ApiOptions, SimpleStreamOptions, StreamOptions,
    ThinkingDisplay, ToolChoice,
};
use crate::provider::ProviderStreams;
use crate::simple_options::{adjust_max_tokens_for_thinking, build_base_options, clamp_max_tokens_to_context};
use crate::transcript::{
    get_current_tools, get_initial_system_message, get_system_message_text, render_system_message_update,
    resolve_transcript,
};
use crate::transform::{split_error_tool_result_images, transform_messages};
use crate::types::{
    AssistantBlock, AssistantMessage, AssistantMessageDiagnostic, ConstrainedSampling, ImageContent, JsonObject,
    Message, ProviderHeaders, StopReason, StrictMode, Tool, ToolResultMessage, TranscriptContext, UserBlock,
    UserContent,
};

/// The `anthropic-messages` API implementation.
#[derive(Clone, Copy, Debug, Default)]
pub struct AnthropicMessagesApi;

impl ProviderStreams for AnthropicMessagesApi {
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
// Constants
// ---------------------------------------------------------------------------

const API_ID: &str = "anthropic-messages";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";

const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
const SERVER_SIDE_FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
const MID_CONVERSATION_OUTPUT_CONFIG_BETA: &str = "mid-conversation-output-config-2026-07-01";
const THINKING_BINDING_CONTROLS_BETA: &str = "thinking-binding-controls-2026-08-01";
const INLINE_TOOLS_BETA: &str = "inline-tools-2026-09-15";
const CLAUDE_CODE_BETA: &str = "claude-code-20250219";
const OAUTH_BETA: &str = "oauth-2025-04-20";

/// Stealth mode: mimic Claude Code's identity and tool naming exactly.
pub const CLAUDE_CODE_VERSION: &str = "2.1.280";
/// First system block of every OAuth (Claude Code) request.
pub const CLAUDE_CODE_SYSTEM_PROMPT: &str = "You are Claude Code, Anthropic's official CLI for Claude.";

/// Claude Code 2.x tool names (canonical casing).
/// Source: <https://cchistory.mariozechner.at/data/prompts-2.1.11.md>
pub const CLAUDE_CODE_TOOLS: [&str; 17] = [
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
];

/// SSE event names that carry Messages stream events; everything else (`ping`, proxy
/// extensions, `[DONE]`) is skipped.
const MESSAGE_EVENTS: [&str; 6] = [
    "message_start",
    "message_delta",
    "message_stop",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
];

/// 400 messages for which a request with preferred (non-required) strict tools is retried
/// with those tools non-strict.
const STRICT_TOOL_LIMIT_MESSAGES: [&str; 3] = [
    "The compiled grammar is too large",
    "Too many strict tools (",
    "Schemas contains too many parameters with union types (",
];

/// Keywords Anthropic strict tool use rejects with a 400 for the whole request.
/// <https://platform.claude.com/docs/en/build-with-claude/structured-outputs#json-schema-limitations>
const STRICT_UNSUPPORTED_KEYWORDS: [&str; 11] = [
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "maxItems",
    "uniqueItems",
    "minContains",
    "maxContains",
    "minProperties",
    "maxProperties",
];
const STRICT_STRING_FORMATS: [&str; 10] = [
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "uri",
    "ipv4",
    "ipv6",
    "uuid",
];

const ANTHROPIC_EFFORTS: [&str; 5] = ["low", "medium", "high", "xhigh", "max"];

/// Stable deferred tool declared whenever native tool changes are in use. Anthropic adds
/// hidden prompt scaffolding for mid-conversation tool changes; declaring this placeholder
/// from the first request keeps that scaffolding in the cached prefix, so the first tool
/// change does not invalidate the cache. It is never activated and the model cannot see it.
fn deferred_tool_placeholder() -> Value {
    json!({
        "name": "__pi_deferred_placeholder__",
        "description": "Reserved placeholder. Never available. Never call this.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "defer_loading": true,
    })
}

// ---------------------------------------------------------------------------
// Compat, cache control and auth
// ---------------------------------------------------------------------------

/// `getAnthropicCompat` plus the raw flags upstream reads from `model.compat` directly.
#[derive(Clone, Debug)]
struct Compat {
    supports_eager_tool_input_streaming: bool,
    supports_long_cache_retention: bool,
    send_session_affinity_headers: bool,
    /// `x-session-id` (OpenRouter) instead of `x-session-affinity`.
    openrouter_session_affinity: bool,
    supports_cache_control_on_tools: bool,
    supports_temperature: bool,
    allow_empty_signature: bool,
    supports_strict_tools: bool,
    supports_mid_convo_system_messages: bool,
    supports_mid_convo_tool_changes: bool,
    force_adaptive_thinking: bool,
    supports_mid_convo_effort: bool,
    allowed_fallback_models: Vec<AnthropicAllowedFallbackModel>,
}

impl Compat {
    fn of(model: &Model) -> Self {
        let raw = match &model.compat {
            Some(ModelCompat::AnthropicMessages(compat)) => Some(compat),
            _ => None,
        };
        let flag = |get: fn(&crate::compat::AnthropicMessagesCompat) -> Option<bool>| raw.and_then(get);
        let is_openrouter = model.provider == "openrouter" || model.base_url.contains("openrouter.ai");
        let format = raw.and_then(|c| c.session_affinity_format);
        Self {
            supports_eager_tool_input_streaming: flag(|c| c.supports_eager_tool_input_streaming).unwrap_or(true),
            supports_long_cache_retention: flag(|c| c.supports_long_cache_retention).unwrap_or(true),
            send_session_affinity_headers: flag(|c| c.send_session_affinity_headers).unwrap_or(is_openrouter),
            openrouter_session_affinity: format.map_or(is_openrouter, |f| f == SessionAffinityFormat::Openrouter),
            supports_cache_control_on_tools: flag(|c| c.supports_cache_control_on_tools).unwrap_or(true),
            supports_temperature: flag(|c| c.supports_temperature).unwrap_or(true),
            allow_empty_signature: flag(|c| c.allow_empty_signature).unwrap_or(false),
            supports_strict_tools: flag(|c| c.supports_strict_tools).unwrap_or(false),
            supports_mid_convo_system_messages: flag(|c| c.supports_mid_convo_system_messages).unwrap_or(false),
            supports_mid_convo_tool_changes: flag(|c| c.supports_mid_convo_tool_changes).unwrap_or(false),
            force_adaptive_thinking: flag(|c| c.force_adaptive_thinking) == Some(true),
            supports_mid_convo_effort: flag(|c| c.supports_mid_convo_effort) == Some(true),
            allowed_fallback_models: raw.and_then(|c| c.allowed_fallback_models.clone()).unwrap_or_default(),
        }
    }
}

/// Cache retention preference: the option, else `PI_CACHE_RETENTION=long`, else short.
fn resolve_cache_retention(options: &StreamOptions) -> CacheRetention {
    if let Some(retention) = options.cache_retention {
        return retention;
    }
    if options.env_var("PI_CACHE_RETENTION").as_deref() == Some("long") {
        return CacheRetention::Long;
    }
    CacheRetention::Short
}

/// `cache_control` for the resolved retention; `None` when caching is off.
fn cache_control(compat: &Compat, retention: CacheRetention) -> Option<Value> {
    match retention {
        CacheRetention::None => None,
        CacheRetention::Long if compat.supports_long_cache_retention => Some(json!({"type": "ephemeral", "ttl": "1h"})),
        _ => Some(json!({"type": "ephemeral"})),
    }
}

fn has_header(headers: Option<&ProviderHeaders>, name: &str) -> bool {
    headers.into_iter().flatten().any(|(key, value)| {
        key.eq_ignore_ascii_case(name) && value.as_deref().is_some_and(|value| !value.trim().is_empty())
    })
}

fn has_request_auth(api_key: Option<&str>, headers: Option<&ProviderHeaders>) -> bool {
    api_key.is_some_and(|key| !key.is_empty())
        || has_header(headers, "authorization")
        || has_header(headers, "x-api-key")
        || has_header(headers, "cf-aig-authorization")
}

/// Whether `api_key` is a Claude Code OAuth access token.
pub fn is_oauth_token(api_key: &str) -> bool {
    api_key.contains("sk-ant-oat")
}

/// A tool name in Claude Code's canonical casing when it matches one (case-insensitive).
pub fn to_claude_code_name(name: &str) -> String {
    let lower = name.to_lowercase();
    CLAUDE_CODE_TOOLS
        .iter()
        .find(|tool| tool.to_lowercase() == lower)
        .map_or_else(|| name.to_owned(), |tool| (*tool).to_owned())
}

/// Map a tool name returned in OAuth mode back to the caller's tool name.
pub fn from_claude_code_name(name: &str, tools: &[Tool]) -> String {
    let lower = name.to_lowercase();
    tools
        .iter()
        .find(|tool| tool.name.to_lowercase() == lower)
        .map_or_else(|| name.to_owned(), |tool| tool.name.clone())
}

/// Normalize a tool call id to Anthropic's `^[a-zA-Z0-9_-]{1,64}$`: other characters become
/// `_` (one per UTF-16 unit, as upstream's regex replaces), then the id is cut to 64.
pub fn normalize_tool_call_id(id: &str) -> String {
    let mut normalized = String::with_capacity(id.len());
    for ch in id.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            normalized.push(ch);
        } else {
            for _ in 0..ch.len_utf16() {
                normalized.push('_');
            }
        }
    }
    normalized.chars().take(64).collect()
}

// ---------------------------------------------------------------------------
// Strict tool schemas (shared logic in `api::constrained_sampling`)
// ---------------------------------------------------------------------------

/// Whether Anthropic's strict mode rejects this schema keyword with this value (upstream
/// `isAnthropicStrictUnsupportedKeyword`).
#[allow(clippy::float_cmp)] // Exact, like upstream's `value !== 0 && value !== 1`.
fn is_strict_unsupported_keyword(key: &str, value: &Value) -> bool {
    if STRICT_UNSUPPORTED_KEYWORDS.contains(&key) {
        return true;
    }
    match key {
        "minItems" => !value.as_f64().is_some_and(|n| n == 0.0 || n == 1.0),
        "format" => !value
            .as_str()
            .is_some_and(|format| STRICT_STRING_FORMATS.contains(&format)),
        _ => false,
    }
}

fn json_schema_strict_mode(tool: &Tool) -> Option<StrictMode> {
    match &tool.constrained_sampling {
        Some(ConstrainedSampling::JsonSchema { strict }) => Some(*strict),
        _ => None,
    }
}

fn is_preferred_strict(tool: &Tool) -> bool {
    json_schema_strict_mode(tool) == Some(StrictMode::Prefer)
}

// ---------------------------------------------------------------------------
// Request payload
// ---------------------------------------------------------------------------

/// Converts tool definitions and records whether a preferred strict tool was sent strict.
#[derive(Debug)]
struct ToolConverter {
    is_oauth: bool,
    eager_input_streaming: bool,
    supports_strict_tools: bool,
    disable_preferred_strict_tools: bool,
    has_preferred_strict_tools: bool,
}

impl ToolConverter {
    fn convert(&mut self, tools: &[Tool], cache_control: Option<&Value>) -> Result<Vec<Value>> {
        let mut definitions = Vec::with_capacity(tools.len());
        for (index, tool) in tools.iter().enumerate() {
            let strict = if self.disable_preferred_strict_tools && is_preferred_strict(tool) {
                None
            } else {
                resolve_json_schema_strict_sampling(
                    tool,
                    self.supports_strict_tools,
                    Some(is_strict_unsupported_keyword),
                )?
            };
            let parameters = get_json_schema_tool_parameters(tool, strict)?;
            let strict = strict == Some(true);
            let non_null = |key: &str, default: Value| match parameters.get(key) {
                Some(value) if !value.is_null() => value.clone(),
                _ => default,
            };
            let mut legacy = Map::new();
            legacy.insert("type".into(), json!("object"));
            legacy.insert("properties".into(), non_null("properties", json!({})));
            legacy.insert("required".into(), non_null("required", json!([])));
            let input_schema = match (&parameters, strict) {
                (Value::Object(parameters), true) => {
                    let mut schema = parameters.clone();
                    schema.extend(legacy);
                    schema
                }
                _ => legacy,
            };
            let mut definition = Map::new();
            let name = if self.is_oauth {
                to_claude_code_name(&tool.name)
            } else {
                tool.name.clone()
            };
            definition.insert("name".into(), json!(name));
            definition.insert("description".into(), json!(tool.description));
            if self.eager_input_streaming {
                definition.insert("eager_input_streaming".into(), json!(true));
            }
            if strict {
                definition.insert("strict".into(), json!(true));
                if is_preferred_strict(tool) {
                    self.has_preferred_strict_tools = true;
                }
            }
            definition.insert("input_schema".into(), Value::Object(input_schema));
            if let Some(cache_control) = cache_control
                && index + 1 == tools.len()
            {
                definition.insert("cache_control".into(), cache_control.clone());
            }
            definitions.push(Value::Object(definition));
        }
        Ok(definitions)
    }
}

fn convert_image_block(image: &ImageContent) -> Value {
    json!({
        "type": "image",
        "source": {"type": "base64", "media_type": image.mime_type, "data": image.data},
    })
}

/// Tool result content: a string when text-only, otherwise blocks (with a placeholder text
/// block when there is no text).
fn convert_content_blocks(content: &[UserBlock]) -> Value {
    let has_images = content.iter().any(|block| matches!(block, UserBlock::Image(_)));
    if !has_images {
        let text: Vec<&str> = content
            .iter()
            .filter_map(|block| match block {
                UserBlock::Text(text) => Some(text.text.as_str()),
                UserBlock::Image(_) => None,
            })
            .collect();
        return json!(text.join("\n"));
    }
    let mut blocks: Vec<Value> = content
        .iter()
        .map(|block| match block {
            UserBlock::Text(text) => json!({"type": "text", "text": text.text}),
            UserBlock::Image(image) => convert_image_block(image),
        })
        .collect();
    if !content.iter().any(|block| matches!(block, UserBlock::Text(_))) {
        blocks.insert(0, json!({"type": "text", "text": "(see attached image)"}));
    }
    Value::Array(blocks)
}

fn convert_tool_result(message: &ToolResultMessage, content: &[UserBlock]) -> Value {
    json!({
        "type": "tool_result",
        "tool_use_id": message.tool_call_id,
        "content": convert_content_blocks(content),
        "is_error": message.is_error,
    })
}

/// Converted messages plus the effort each replayed assistant message was generated with,
/// keyed by its index in `messages`.
#[derive(Debug, Default)]
struct ConvertedMessages {
    messages: Vec<Value>,
    assistant_levels: HashMap<usize, String>,
}

/// Convert transformed messages (without the leading system message) to Anthropic messages.
/// `tool_converter` is set when tool changes are sent natively (`tool_addition` blocks).
fn convert_messages(
    messages: &[Message],
    is_oauth: bool,
    cache_control: Option<&Value>,
    allow_empty_signature: bool,
    managed_provider: Option<&str>,
    mut tool_converter: Option<&mut ToolConverter>,
) -> Result<ConvertedMessages> {
    let mut params: Vec<Value> = Vec::new();
    let mut assistant_levels = HashMap::new();
    // Later system messages are held back and emitted directly before the next assistant
    // message (or at the end). Anthropic requires `tool_result` blocks to immediately follow
    // their `tool_use`, so a system message between them is rejected; this also mirrors where
    // the managed-effort system messages are inserted.
    let mut pending_system: Vec<Value> = Vec::new();

    let mut index = 0;
    while index < messages.len() {
        match &messages[index] {
            Message::System(system) => {
                // Later system messages only reach this point when the model accepts them
                // natively; otherwise the transcript was collapsed before conversion.
                let text = render_system_message_update(system);
                let mut blocks = Vec::new();
                if !text.is_empty() {
                    blocks.push(json!({"type": "text", "text": text}));
                }
                if let Some(converter) = tool_converter.as_deref_mut() {
                    let added = system.tools_added.clone().unwrap_or_default();
                    let redefined: HashSet<&str> = added.iter().map(|tool| tool.name.as_str()).collect();
                    for removed in system.tools_removed.iter().flatten() {
                        // A new definition under the same name replaces the old one.
                        if redefined.contains(removed.name.as_str()) {
                            continue;
                        }
                        let name = if is_oauth {
                            to_claude_code_name(&removed.name)
                        } else {
                            removed.name.clone()
                        };
                        blocks.push(json!({"type": "tool_removal", "tool": {"type": "tool_reference", "name": name}}));
                    }
                    for definition in converter.convert(&added, None)? {
                        blocks.push(json!({"type": "tool_addition", "tool": {"type": "tool_definition", "definition": definition}}));
                    }
                }
                if !blocks.is_empty() {
                    pending_system.push(json!({"role": "system", "content": blocks}));
                }
            }
            Message::User(user) => match &user.content {
                UserContent::Text(text) => {
                    if !text.trim().is_empty() {
                        params.push(json!({"role": "user", "content": text}));
                    }
                }
                UserContent::Blocks(blocks) => {
                    let blocks: Vec<Value> = blocks
                        .iter()
                        .filter_map(|block| match block {
                            UserBlock::Text(text) if text.text.trim().is_empty() => None,
                            UserBlock::Text(text) => Some(json!({"type": "text", "text": text.text})),
                            UserBlock::Image(image) => Some(convert_image_block(image)),
                        })
                        .collect();
                    if !blocks.is_empty() {
                        params.push(json!({"role": "user", "content": blocks}));
                    }
                }
            },
            Message::Assistant(assistant) => {
                params.append(&mut pending_system);
                let blocks = convert_assistant_blocks(assistant, is_oauth, allow_empty_signature);
                if !blocks.is_empty() {
                    let message_index = params.len();
                    params.push(json!({"role": "assistant", "content": blocks}));
                    if let Some(provider) = managed_provider
                        && assistant.api == API_ID
                        && assistant.provider == provider
                        && let Some(level) = assistant
                            .provider_thinking_level
                            .as_deref()
                            .filter(|level| ANTHROPIC_EFFORTS.contains(level))
                    {
                        assistant_levels.insert(message_index, level.to_owned());
                    }
                }
            }
            Message::ToolResult(_) => {
                // Consecutive tool results go in one user message. Anthropic rejects error tool
                // results that contain anything but text, so their images follow all tool
                // results of the turn, which must come first in the message.
                let mut content = Vec::new();
                let mut error_images = Vec::new();
                while let Some(Message::ToolResult(result)) = messages.get(index) {
                    match split_error_tool_result_images(result) {
                        Some(split) => {
                            let text: Vec<UserBlock> = split.content.into_iter().map(UserBlock::Text).collect();
                            content.push(convert_tool_result(result, &text));
                            error_images.push(json!({"type": "text", "text": split.caption}));
                            error_images.extend(split.images.iter().map(convert_image_block));
                        }
                        None => content.push(convert_tool_result(result, &result.content)),
                    }
                    index += 1;
                }
                content.append(&mut error_images);
                params.push(json!({"role": "user", "content": content}));
                continue;
            }
        }
        index += 1;
    }
    params.append(&mut pending_system);

    // Cache the conversation history at the last user or system message.
    if let Some(cache_control) = cache_control
        && let Some(Value::Object(last)) = params.last_mut()
        && matches!(last.get("role").and_then(Value::as_str), Some("user" | "system"))
    {
        match last.get_mut("content") {
            Some(Value::Array(blocks)) => {
                if let Some(Value::Object(block)) = blocks.last_mut()
                    && matches!(
                        block.get("type").and_then(Value::as_str),
                        Some("text" | "image" | "tool_result" | "tool_addition" | "tool_removal")
                    )
                {
                    block.insert("cache_control".into(), cache_control.clone());
                }
            }
            Some(content @ Value::String(_)) => {
                let text = std::mem::take(content);
                *content = json!([{"type": "text", "text": text, "cache_control": cache_control}]);
            }
            _ => {}
        }
    }

    Ok(ConvertedMessages {
        messages: params,
        assistant_levels,
    })
}

fn convert_assistant_blocks(assistant: &AssistantMessage, is_oauth: bool, allow_empty_signature: bool) -> Vec<Value> {
    let mut blocks = Vec::new();
    for block in &assistant.content {
        match block {
            AssistantBlock::Text(text) => {
                if !text.text.trim().is_empty() {
                    blocks.push(json!({"type": "text", "text": text.text}));
                }
            }
            AssistantBlock::Thinking(thinking) => {
                // Redacted thinking: pass the opaque payload back as redacted_thinking.
                if thinking.is_redacted() {
                    let mut block = Map::new();
                    block.insert("type".into(), json!("redacted_thinking"));
                    if let Some(data) = &thinking.thinking_signature {
                        block.insert("data".into(), json!(data));
                    }
                    blocks.push(Value::Object(block));
                    continue;
                }
                let signature = thinking
                    .thinking_signature
                    .as_deref()
                    .filter(|signature| !signature.trim().is_empty());
                if thinking.thinking.trim().is_empty() && signature.is_none() {
                    continue;
                }
                // A missing/empty signature (e.g. from an aborted stream) becomes plain text,
                // unless the model's provider emits and accepts empty signatures.
                blocks.push(match signature {
                    Some(signature) => {
                        json!({"type": "thinking", "thinking": thinking.thinking, "signature": signature})
                    }
                    None if allow_empty_signature => {
                        json!({"type": "thinking", "thinking": thinking.thinking, "signature": ""})
                    }
                    None => json!({"type": "text", "text": thinking.thinking}),
                });
            }
            AssistantBlock::ToolCall(call) => {
                let name = if is_oauth {
                    to_claude_code_name(&call.name)
                } else {
                    call.name.clone()
                };
                blocks.push(json!({"type": "tool_use", "id": call.id, "name": name, "input": call.arguments}));
            }
        }
    }
    blocks
}

/// Effort-only system messages before replayed assistants with a known effort, plus a final
/// one for this turn.
fn insert_thinking_level_messages(converted: ConvertedMessages, active_effort: &str) -> Vec<Value> {
    let mut messages = Vec::with_capacity(converted.messages.len() + 1);
    for (index, message) in converted.messages.into_iter().enumerate() {
        if let Some(effort) = converted.assistant_levels.get(&index) {
            messages.push(json!({"role": "system", "content": [], "output_config": {"effort": effort}}));
        }
        messages.push(message);
    }
    messages.push(json!({"role": "system", "content": [], "output_config": {"effort": active_effort}}));
    messages
}

/// The `betas` of a request. An `anthropic-beta` header in the model or request headers
/// replaces them (`null` suppresses them).
fn beta_features(
    model: &Model,
    compat: &Compat,
    current_tools: &[Tool],
    is_oauth: bool,
    native_tool_changes: bool,
    options: &StreamOptions,
    api: &AnthropicOptions,
) -> Vec<String> {
    let mut configured: Option<Option<&str>> = None;
    for (name, value) in model.headers.iter().flatten() {
        if name.eq_ignore_ascii_case("anthropic-beta") {
            configured = Some(Some(value));
        }
    }
    for (name, value) in options.headers.iter().flatten() {
        if name.eq_ignore_ascii_case("anthropic-beta") {
            configured = Some(value.as_deref());
        }
    }
    let mut features: Vec<String> = Vec::new();
    let mut push = |feature: &str| {
        if !features.iter().any(|existing| existing == feature) {
            features.push(feature.to_owned());
        }
    };
    match configured {
        Some(None) => return Vec::new(),
        Some(Some(value)) => {
            for feature in value.split(',').map(str::trim).filter(|feature| !feature.is_empty()) {
                push(feature);
            }
            return features;
        }
        None => {}
    }
    if is_oauth {
        push(CLAUDE_CODE_BETA);
        push(OAUTH_BETA);
    }
    if !current_tools.is_empty() && !compat.supports_eager_tool_input_streaming {
        push(FINE_GRAINED_TOOL_STREAMING_BETA);
    }
    if model.reasoning
        && api.thinking_enabled == Some(true)
        && api.interleaved_thinking.unwrap_or(true)
        && !compat.force_adaptive_thinking
    {
        push(INTERLEAVED_THINKING_BETA);
    }
    if !compat.allowed_fallback_models.is_empty() {
        push(SERVER_SIDE_FALLBACK_BETA);
    }
    if compat.supports_mid_convo_effort {
        push(MID_CONVERSATION_OUTPUT_CONFIG_BETA);
        push(THINKING_BINDING_CONTROLS_BETA);
    }
    if native_tool_changes {
        push(INLINE_TOOLS_BETA);
    }
    features
}

/// A built request body.
#[derive(Debug)]
struct BuiltParams {
    params: Value,
    has_preferred_strict_tools: bool,
    /// Effort of the final effort marker, recorded on the reply so later requests can replay it.
    provider_thinking_level: Option<String>,
}

fn display_str(display: Option<ThinkingDisplay>) -> &'static str {
    match display {
        Some(ThinkingDisplay::Omitted) => "omitted",
        Some(ThinkingDisplay::Summarized) | None => "summarized",
    }
}

/// Build the request body (including `betas`, which become the `anthropic-beta` header) for
/// a resolved transcript.
fn build_params(
    model: &Model,
    context: &TranscriptContext,
    is_oauth: bool,
    options: &StreamOptions,
    api: &AnthropicOptions,
    disable_preferred_strict_tools: bool,
) -> Result<BuiltParams> {
    let compat = Compat::of(model);
    let cache_control = cache_control(&compat, resolve_cache_retention(options));
    let initial = get_initial_system_message(context.messages());
    let initial_text = initial.map(get_system_message_text).unwrap_or_default();
    let transformed = transform_messages(
        context.messages(),
        model,
        Some(&|id: &str, _: &Model, _: &AssistantMessage| normalize_tool_call_id(id)),
    );
    let conversation = if initial.is_some() {
        &transformed[1..]
    } else {
        &transformed[..]
    };

    let mut converter = ToolConverter {
        is_oauth,
        eager_input_streaming: compat.supports_eager_tool_input_streaming,
        supports_strict_tools: compat.supports_strict_tools,
        disable_preferred_strict_tools,
        has_preferred_strict_tools: false,
    };
    // Native tool changes keep the request-level tool list fixed and define every later tool
    // by value in a `tool_addition` block. Anthropic rejects a tool list where every tool is
    // deferred, so there must be an initial active tool to anchor the placeholder.
    let initial_tools = initial
        .and_then(|system| system.tools_added.clone())
        .unwrap_or_default();
    let native_tool_changes = compat.supports_mid_convo_system_messages
        && compat.supports_mid_convo_tool_changes
        && !initial_tools.is_empty();
    let thinking_disabled = model.reasoning
        && api.thinking_enabled == Some(false)
        && model.mapped_thinking_level(ModelThinkingLevel::Off) != Some(None);
    // Effort cannot change between messages while thinking is disabled, so "off" sends no
    // effort markers.
    let effort_messages = compat.supports_mid_convo_effort && !thinking_disabled;
    let converted = convert_messages(
        conversation,
        is_oauth,
        cache_control.as_ref(),
        compat.allow_empty_signature,
        effort_messages.then_some(model.provider.as_str()),
        native_tool_changes.then_some(&mut converter),
    )?;
    let active_effort = api.effort.clone().unwrap_or(AnthropicEffort::High);
    let current_tools = get_current_tools(context.messages());
    let betas = beta_features(
        model,
        &compat,
        &current_tools,
        is_oauth,
        native_tool_changes,
        options,
        api,
    );

    let mut params = Map::new();
    params.insert("model".into(), json!(model.id));
    let messages = if effort_messages {
        insert_thinking_level_messages(converted, active_effort.as_str())
    } else {
        converted.messages
    };
    params.insert("messages".into(), Value::Array(messages));
    params.insert(
        "max_tokens".into(),
        json!(options.max_tokens.unwrap_or(model.max_tokens)),
    );
    params.insert("stream".into(), json!(true));
    if !betas.is_empty() {
        params.insert("betas".into(), json!(betas));
    }

    let system_block = |text: &str| {
        let mut block = Map::new();
        block.insert("type".into(), json!("text"));
        block.insert("text".into(), json!(text));
        if let Some(cache_control) = &cache_control {
            block.insert("cache_control".into(), cache_control.clone());
        }
        Value::Object(block)
    };
    // OAuth requests must carry the Claude Code identity.
    if is_oauth {
        let mut system = vec![system_block(CLAUDE_CODE_SYSTEM_PROMPT)];
        if !initial_text.is_empty() {
            system.push(system_block(&initial_text));
        }
        params.insert("system".into(), Value::Array(system));
    } else if !initial_text.is_empty() {
        params.insert("system".into(), json!([system_block(&initial_text)]));
    }

    // Temperature is incompatible with extended thinking and unsupported on Claude Opus 4.7+.
    if let Some(temperature) = options.temperature
        && api.thinking_enabled != Some(true)
        && !compat.supports_mid_convo_effort
        && compat.supports_temperature
    {
        params.insert("temperature".into(), json!(temperature));
    }

    let tool_cache_control = cache_control
        .as_ref()
        .filter(|_| compat.supports_cache_control_on_tools);
    if native_tool_changes {
        // Initial tools stay active with the cache breakpoint on the last one, followed by the
        // placeholder; later tools are defined by value in `tool_addition` blocks.
        let mut tools = converter.convert(&initial_tools, tool_cache_control)?;
        tools.push(deferred_tool_placeholder());
        params.insert("tools".into(), Value::Array(tools));
    } else if !current_tools.is_empty() {
        params.insert(
            "tools".into(),
            Value::Array(converter.convert(&current_tools, tool_cache_control)?),
        );
    }

    let display = display_str(api.thinking_display);
    if thinking_disabled {
        // No effort is sent, so the model default applies. Anthropic rejects `block_binding`
        // with disabled thinking.
        params.insert("thinking".into(), json!({"type": "disabled"}));
    } else if effort_messages {
        // Managed effort models use adaptive thinking so prefix mismatches can be dropped
        // instead of surfacing as persistent 400 responses.
        params.insert(
            "thinking".into(),
            json!({"type": "adaptive", "display": display, "block_binding": {"prefix_mismatch_behavior": "drop_block"}}),
        );
        params.insert("output_config".into(), json!({"effort": "high"}));
    } else if model.reasoning && api.thinking_enabled == Some(true) {
        if compat.force_adaptive_thinking {
            params.insert("thinking".into(), json!({"type": "adaptive", "display": display}));
            if let Some(effort) = &api.effort {
                params.insert("output_config".into(), json!({"effort": effort.as_str()}));
            }
        } else {
            let budget = api.thinking_budget_tokens.filter(|budget| *budget > 0).unwrap_or(1024);
            params.insert(
                "thinking".into(),
                json!({"type": "enabled", "budget_tokens": budget, "display": display}),
            );
        }
    }

    if let Some(user_id) = options
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("user_id"))
        .and_then(Value::as_str)
    {
        params.insert("metadata".into(), json!({"user_id": user_id}));
    }

    if let Some(choice) = &api.tool_choice {
        params.insert(
            "tool_choice".into(),
            match choice {
                AnthropicToolChoice::Auto => json!({"type": "auto"}),
                AnthropicToolChoice::Any => json!({"type": "any"}),
                AnthropicToolChoice::None => json!({"type": "none"}),
                AnthropicToolChoice::Tool { name } => json!({"type": "tool", "name": name}),
            },
        );
    }

    if !compat.allowed_fallback_models.is_empty() {
        let fallbacks: Vec<Value> = compat
            .allowed_fallback_models
            .iter()
            .map(|fallback| json!({"model": fallback.model}))
            .collect();
        params.insert("fallbacks".into(), Value::Array(fallbacks));
    }

    Ok(BuiltParams {
        params: Value::Object(params),
        has_preferred_strict_tools: converter.has_preferred_strict_tools,
        provider_thinking_level: effort_messages.then(|| active_effort.as_str().to_owned()),
    })
}

// ---------------------------------------------------------------------------
// HTTP request
// ---------------------------------------------------------------------------

/// How the request authenticates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AuthMode {
    /// `x-api-key`, or header-owned auth when there is no key.
    ApiKey,
    /// Claude Code OAuth token: `Authorization: Bearer`, Claude Code identity.
    OAuth,
    /// GitHub Copilot: `Authorization: Bearer` and the dynamic headers of
    /// [`crate::api::github_copilot_headers`].
    Copilot,
}

fn auth_mode(model: &Model, api_key: Option<&str>) -> AuthMode {
    if model.provider == "github-copilot" {
        AuthMode::Copilot
    } else if api_key.is_some_and(is_oauth_token) {
        AuthMode::OAuth
    } else {
        AuthMode::ApiKey
    }
}

/// Header sources in the order the Anthropic SDK applies them: SDK defaults, auth, the
/// client's default headers (pi's User-Agent, then model headers, GitHub Copilot's dynamic
/// headers for `messages`, and request headers).
fn base_header_sources(
    model: &Model,
    messages: &[Message],
    compat: &Compat,
    mode: AuthMode,
    api_key: Option<&str>,
    session_id: Option<&str>,
    options: &StreamOptions,
) -> Vec<ProviderHeaders> {
    let mut sources = vec![http::headers_of([
        ("Accept", "application/json"),
        ("anthropic-version", ANTHROPIC_VERSION),
    ])];
    match (mode, api_key) {
        (AuthMode::ApiKey, Some(key)) => sources.push(http::headers_of([("X-Api-Key", key)])),
        (AuthMode::OAuth | AuthMode::Copilot, Some(token)) => {
            sources.push(http::headers_of([("Authorization", &format!("Bearer {token}"))]));
        }
        (_, None) => {}
    }
    sources.push(http::headers_of([
        ("User-Agent", &http::user_agent()),
        ("accept", "application/json"),
        ("anthropic-dangerous-direct-browser-access", "true"),
    ]));
    match mode {
        AuthMode::OAuth => sources.push(http::headers_of([
            ("user-agent", &format!("claude-cli/{CLAUDE_CODE_VERSION}")),
            ("x-app", "cli"),
        ])),
        AuthMode::ApiKey => {
            if let Some(session_id) = session_id
                && compat.send_session_affinity_headers
            {
                let name = if compat.openrouter_session_affinity {
                    "x-session-id"
                } else {
                    "x-session-affinity"
                };
                sources.push(http::headers_of([(name, session_id)]));
            }
        }
        AuthMode::Copilot => {}
    }
    if let Some(headers) = &model.headers {
        sources.push(
            headers
                .iter()
                .map(|(name, value)| (name.clone(), Some(value.clone())))
                .collect(),
        );
    }
    // Copilot's dynamic headers come after the model's headers and before the request's.
    if mode == AuthMode::Copilot {
        sources.push(build_copilot_dynamic_headers(
            messages,
            has_copilot_vision_input(messages),
        ));
    }
    if let Some(headers) = &options.headers {
        sources.push(headers.clone());
    }
    sources.push(http::headers_of([("Content-Type", "application/json")]));
    sources
}

/// Request-level headers the SDK derives from body fields: `betas` and `user_profile_id`
/// leave the body and become headers.
fn take_request_headers(params: &mut Value) -> ProviderHeaders {
    let mut headers = ProviderHeaders::new();
    let Some(body) = params.as_object_mut() else {
        return headers;
    };
    if let Some(betas) = body.remove("betas")
        && let Some(betas) = betas.as_array()
        && !betas.is_empty()
    {
        let joined: Vec<String> = betas
            .iter()
            .map(|beta| beta.as_str().map_or_else(|| beta.to_string(), str::to_owned))
            .collect();
        headers.insert("anthropic-beta".into(), Some(joined.join(",")));
    }
    if let Some(profile) = body.remove("user_profile_id")
        && let Some(profile) = profile.as_str()
    {
        headers.insert("anthropic-user-profile-id".into(), Some(profile.to_owned()));
    }
    headers
}

/// Give an HTTP error the message the Anthropic SDK gives it (`APIError.makeMessage` over
/// the whole parsed body), which upstream surfaces as is: `400 {"type":"error","error":{...}}`
/// for a JSON body, `"<status> <text>"` otherwise.
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
    let (json, text) = common::sdk_error_body(body.as_deref());
    let message = common::sdk_api_error_message(status, json.as_ref(), text);
    Error::Status {
        status,
        message,
        body,
        retry_after_ms,
    }
}

/// A 400 `invalid_request_error` (the SDK's `error.type`, from the body's `error.type`) whose
/// message names one of the strict tool limits.
fn is_strict_tool_limit_error(error: &Error) -> bool {
    let Error::Status {
        status: 400,
        message,
        body,
        ..
    } = error
    else {
        return false;
    };
    let error_type = body
        .as_deref()
        .and_then(|body| serde_json::from_str::<Value>(body).ok())
        .and_then(|body| body.pointer("/error/type").and_then(Value::as_str).map(str::to_owned));
    error_type.as_deref() == Some("invalid_request_error")
        && STRICT_TOOL_LIMIT_MESSAGES.iter().any(|text| message.contains(text))
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

fn api_options(options: &StreamOptions) -> AnthropicOptions {
    match &options.api {
        Some(ApiOptions::AnthropicMessages(api)) => api.clone(),
        _ => AnthropicOptions::default(),
    }
}

/// Stream a response from the Anthropic Messages API. API options come from
/// `options.api` ([`ApiOptions::AnthropicMessages`]).
pub fn stream(model: &Model, context: &TranscriptContext, options: StreamOptions) -> AssistantMessageEventStream {
    let owned_model = model.clone();
    let context = context.clone();
    let cancel = options.cancel.clone();
    spawn_stream(model, cancel, move |mut builder| async move {
        let result = run(&owned_model, &context, &options, &mut builder).await;
        (builder, result)
    })
}

async fn run(
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
    builder: &mut MessageBuilder,
) -> Result<()> {
    let compat = Compat::of(model);
    let context = resolve_transcript(context, compat.supports_mid_convo_system_messages);
    let current_tools = get_current_tools(context.messages());
    let api = api_options(options);

    let api_key = options.api_key.as_deref().filter(|key| !key.is_empty());
    // Not ported: workload identity federation (ANTHROPIC_FEDERATION_RULE_ID & co.), which
    // upstream delegates to the Anthropic SDK's token exchange.
    if !has_request_auth(api_key, options.headers.as_ref()) {
        return Err(Error::msg(format!("No API key for provider: {}", model.provider)));
    }
    let mode = auth_mode(model, api_key);
    let is_oauth = mode == AuthMode::OAuth;
    let retention = resolve_cache_retention(options);
    let session_id = options
        .session_id
        .as_deref()
        .filter(|_| retention != CacheRetention::None);
    let header_sources = base_header_sources(model, context.messages(), &compat, mode, api_key, session_id, options);
    let base_url = if model.base_url.is_empty() {
        DEFAULT_BASE_URL
    } else {
        model.base_url.as_str()
    };
    let url = common::join_url(base_url, "/v1/messages?beta=true");
    // Upstream retries through `retryProviderRequest`, whose default is no retries.
    let retry_options = StreamOptions {
        max_retries: Some(options.max_retries.unwrap_or(0)),
        ..options.clone()
    };
    let client = http::client(options);

    let mut disable_preferred_strict_tools = false;
    let response = loop {
        if options.is_cancelled() {
            return Err(Error::Aborted);
        }
        let built = build_params(model, &context, is_oauth, options, &api, disable_preferred_strict_tools)?;
        if let Some(level) = &built.provider_thinking_level {
            builder.output.provider_thinking_level = Some(level.clone());
        }
        let mut params = common::apply_on_payload(options, model, built.params).await?;
        if let Some(body) = params.as_object_mut() {
            body.insert("stream".into(), Value::Bool(true));
        }
        let request_headers = take_request_headers(&mut params);
        let headers = http::build_headers(header_sources.iter().chain(std::iter::once(&request_headers)))?;
        let body = serde_json::to_vec(&params)?;
        let sent = http::send_with_retry(&retry_options, || {
            Ok(client.post(&url).headers(headers.clone()).body(body.clone()))
        })
        .await
        .map_err(format_http_error);
        match sent {
            Ok(response) => break response,
            Err(error) => {
                // Only downgrade a rejected request, never an already-started stream.
                if options.is_cancelled()
                    || disable_preferred_strict_tools
                    || !built.has_preferred_strict_tools
                    || !is_strict_tool_limit_error(&error)
                {
                    return Err(error);
                }
                disable_preferred_strict_tools = true;
                let mut details = JsonObject::new();
                details.insert("reason".into(), json!("strict_tool_limits_exceeded"));
                builder.output.push_diagnostic(AssistantMessageDiagnostic::new(
                    "anthropic_strict_tools_fallback",
                    Some(error.to_string()),
                    Some(details),
                ));
            }
        }
    };
    common::notify_response(options, model, &response).await?;
    builder.start();

    let mut state = StreamState::new(model, &compat, &current_tools, is_oauth);
    let mut events = Box::pin(http::sse_events(response, options.cancel.clone()));
    let mut saw_message_start = false;
    let mut saw_message_stop = false;
    while let Some(event) = events.next().await {
        let event = event?;
        if builder.is_closed() || options.is_cancelled() {
            return Err(Error::Aborted);
        }
        if event.event == "error" {
            return Err(Error::msg(event.data));
        }
        if !MESSAGE_EVENTS.contains(&event.event.as_str()) {
            continue;
        }
        let value = parse_event(&event)?;
        match value.get("type").and_then(Value::as_str) {
            Some("message_start") => saw_message_start = true,
            Some("message_stop") => saw_message_stop = true,
            _ => {}
        }
        common::observe_provider_event(options, model, &value).await?;
        state.handle(&value, builder)?;
    }
    if saw_message_start && !saw_message_stop {
        return Err(Error::msg("Anthropic stream ended before message_stop"));
    }
    if options.is_cancelled() {
        return Err(Error::Aborted);
    }
    match builder.output.stop_reason {
        StopReason::Pending => return Err(Error::msg("Anthropic stream ended without a stop reason")),
        StopReason::Error | StopReason::Aborted => {
            let message = builder
                .output
                .error_message
                .clone()
                .unwrap_or_else(|| "An unknown error occurred".into());
            return Err(Error::msg(message));
        }
        _ => {}
    }
    if let Some(transformations) = state.input_transformations.filter(|list| !list.is_empty()) {
        builder
            .output
            .push_diagnostic(input_transformations_diagnostic(&transformations));
    }
    Ok(())
}

/// Parse a Messages SSE event, repairing malformed JSON as upstream does.
fn parse_event(event: &SseEvent) -> Result<Value> {
    let fail = |reason: String| {
        let mut raw = vec![format!("event: {}", event.event)];
        raw.extend(event.data.split('\n').map(|line| format!("data: {line}")));
        Error::msg(format!(
            "Could not parse Anthropic SSE event {}: {reason}; data={}; raw={}",
            event.event,
            event.data,
            raw.join("\\n")
        ))
    };
    match parse_json_with_repair(&event.data) {
        Ok(value @ Value::Object(_)) => Ok(value),
        Ok(other) => Err(fail(format!("expected a JSON object, got {other}"))),
        Err(error) => Err(fail(error.to_string())),
    }
}

fn input_transformations_diagnostic(transformations: &[Value]) -> AssistantMessageDiagnostic {
    let entries: Vec<Value> = transformations
        .iter()
        .map(|transformation| {
            let mut entry = Map::new();
            for key in ["type", "path", "reason"] {
                if let Some(value) = transformation.get(key).filter(|value| !value.is_null()) {
                    entry.insert(key.into(), value.clone());
                }
            }
            Value::Object(entry)
        })
        .collect();
    let mut details = JsonObject::new();
    details.insert("transformations".into(), Value::Array(entries));
    AssistantMessageDiagnostic::new("anthropic_input_transformations", None, Some(details))
}

fn token_count(value: Option<&Value>) -> Option<u64> {
    let value = value?;
    value.as_u64().or_else(|| value.as_f64().map(|n| n.max(0.0) as u64))
}

/// Map an Anthropic stop reason to a stop reason and an error message.
fn map_stop_reason(reason: &str, stop_details: Option<&Value>) -> Result<(StopReason, Option<String>)> {
    Ok(match reason {
        // `pause_turn`: stop is good enough, the caller resubmits. `stop_sequence`: we never
        // send stop sequences.
        "end_turn" | "pause_turn" | "stop_sequence" => (StopReason::Stop, None),
        "max_tokens" => (StopReason::Length, None),
        "tool_use" => (StopReason::ToolUse, None),
        "refusal" => {
            let explanation = stop_details
                .and_then(|details| details.get("explanation"))
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .unwrap_or("The model refused to complete the request");
            (StopReason::Error, Some(explanation.to_owned()))
        }
        // Content flagged by safety filters.
        "sensitive" => (StopReason::Error, Some("Provider stopped with: sensitive".into())),
        other => return Err(Error::msg(format!("Unhandled stop reason: {other}"))),
    })
}

/// Response parsing state of one stream.
struct StreamState<'a> {
    model: &'a Model,
    compat: &'a Compat,
    current_tools: &'a [Tool],
    is_oauth: bool,
    /// Prices used for the usage cost: a returned fallback model's when it differs.
    cost: ModelCost,
    input_transformations: Option<Vec<Value>>,
    /// Anthropic block index -> positions in the output of blocks not yet stopped.
    open_blocks: HashMap<String, VecDeque<usize>>,
}

fn block_key(event: &Value) -> String {
    event.get("index").map_or_else(|| "undefined".into(), Value::to_string)
}

impl<'a> StreamState<'a> {
    fn new(model: &'a Model, compat: &'a Compat, current_tools: &'a [Tool], is_oauth: bool) -> Self {
        Self {
            model,
            compat,
            current_tools,
            is_oauth,
            cost: model.cost.clone(),
            input_transformations: None,
            open_blocks: HashMap::new(),
        }
    }

    fn find(&self, event: &Value) -> Option<usize> {
        self.open_blocks
            .get(&block_key(event))
            .and_then(|positions| positions.front().copied())
    }

    fn open(&mut self, event: &Value, position: usize) {
        self.open_blocks
            .entry(block_key(event))
            .or_default()
            .push_back(position);
    }

    fn update_usage(&self, builder: &mut MessageBuilder) {
        let usage = &mut builder.output.usage;
        usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write;
        calculate_cost(&self.cost, usage);
    }

    fn handle(&mut self, event: &Value, builder: &mut MessageBuilder) -> Result<()> {
        match event.get("type").and_then(Value::as_str) {
            Some("message_start") => self.message_start(event, builder),
            Some("content_block_start") => self.block_start(event, builder)?,
            Some("content_block_delta") => self.block_delta(event, builder),
            Some("content_block_stop") => self.block_stop(event, builder),
            Some("message_delta") => self.message_delta(event, builder)?,
            _ => {}
        }
        Ok(())
    }

    fn message_start(&mut self, event: &Value, builder: &mut MessageBuilder) {
        let message = event.get("message").unwrap_or(&Value::Null);
        builder.output.response_id = message.get("id").and_then(Value::as_str).map(str::to_owned);
        if let Some(transformations) = message.get("input_transformations").and_then(Value::as_array) {
            self.input_transformations = Some(transformations.clone());
        }
        let response_model = message.get("model").and_then(Value::as_str);
        if let Some(response_model) = response_model
            && response_model != self.model.id
        {
            builder.output.response_model = Some(response_model.to_owned());
        }
        let fallback_cost = response_model.filter(|id| *id != self.model.id).and_then(|id| {
            self.compat
                .allowed_fallback_models
                .iter()
                .find(|fallback| fallback.provider == self.model.provider && fallback.model == id)
                .map(|fallback| fallback.cost.clone())
        });
        self.cost = fallback_cost.unwrap_or_else(|| self.model.cost.clone());
        // Initial usage, so input counts survive a stream aborted early.
        let usage = message.get("usage").unwrap_or(&Value::Null);
        let count = |key: &str| token_count(usage.get(key)).unwrap_or(0);
        let output = &mut builder.output.usage;
        output.input = count("input_tokens");
        output.output = count("output_tokens");
        output.cache_read = count("cache_read_input_tokens");
        output.cache_write = count("cache_creation_input_tokens");
        output.cache_write_1h = Some(
            token_count(
                usage
                    .get("cache_creation")
                    .and_then(|creation| creation.get("ephemeral_1h_input_tokens")),
            )
            .unwrap_or(0),
        );
        self.update_usage(builder);
    }

    fn block_start(&mut self, event: &Value, builder: &mut MessageBuilder) -> Result<()> {
        let block = event.get("content_block").unwrap_or(&Value::Null);
        let text_field = |key: &str| block.get(key).and_then(Value::as_str).unwrap_or_default().to_owned();
        match block.get("type").and_then(Value::as_str) {
            Some("fallback") => {
                if !builder.output.content.is_empty() {
                    return Err(Error::msg(
                        "Anthropic performed an unsupported mid-output model fallback",
                    ));
                }
            }
            Some("text") => {
                let position = builder.start_text();
                builder.text_delta(position, &text_field("text"));
                self.open(event, position);
            }
            Some("thinking") => {
                let position = builder.start_thinking(text_field("thinking"), false, Some(text_field("signature")));
                self.open(event, position);
            }
            Some("redacted_thinking") => {
                let data = block.get("data").and_then(Value::as_str).map(str::to_owned);
                let position = builder.start_thinking("[Reasoning redacted]".into(), true, data);
                self.open(event, position);
            }
            Some("tool_use") => {
                let name = text_field("name");
                let name = if self.is_oauth {
                    from_claude_code_name(&name, self.current_tools)
                } else {
                    name
                };
                let arguments = block
                    .get("input")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                let position = builder.start_tool_call(text_field("id"), name, arguments);
                self.open(event, position);
            }
            _ => {}
        }
        Ok(())
    }

    fn block_delta(&self, event: &Value, builder: &mut MessageBuilder) {
        let Some(position) = self.find(event) else { return };
        let delta = event.get("delta").unwrap_or(&Value::Null);
        let field = |key: &str| delta.get(key).and_then(Value::as_str).unwrap_or_default();
        let block = builder.output.content.get(position);
        match (delta.get("type").and_then(Value::as_str), block) {
            (Some("text_delta"), Some(AssistantBlock::Text(_))) => builder.text_delta(position, field("text")),
            (Some("thinking_delta"), Some(AssistantBlock::Thinking(_))) => {
                builder.thinking_delta(position, field("thinking"));
            }
            (Some("input_json_delta"), Some(AssistantBlock::ToolCall(_))) => {
                builder.tool_call_delta(position, field("partial_json"));
                let arguments = parse_streaming_json(builder.tool_call_json(position).unwrap_or_default());
                if let Some(call) = builder.tool_call_mut(position) {
                    call.arguments = arguments;
                }
            }
            (Some("signature_delta"), Some(AssistantBlock::Thinking(_))) => {
                builder.append_thinking_signature(position, field("signature"));
            }
            _ => {}
        }
    }

    fn block_stop(&mut self, event: &Value, builder: &mut MessageBuilder) {
        let Some(position) = self
            .open_blocks
            .get_mut(&block_key(event))
            .and_then(VecDeque::pop_front)
        else {
            return;
        };
        match builder.output.content.get(position) {
            Some(AssistantBlock::Text(_)) => builder.end_text(position),
            Some(AssistantBlock::Thinking(_)) => builder.end_thinking(position),
            Some(AssistantBlock::ToolCall(_)) => {
                // Arguments come from the streamed JSON only; none streamed means `{}`.
                if builder
                    .tool_call_json(position)
                    .is_none_or(|json| json.trim().is_empty())
                    && let Some(call) = builder.tool_call_mut(position)
                {
                    call.arguments = JsonObject::new();
                }
                builder.end_tool_call(position);
            }
            None => {}
        }
    }

    fn message_delta(&mut self, event: &Value, builder: &mut MessageBuilder) -> Result<()> {
        if let Some(transformations) = event.get("input_transformations").and_then(Value::as_array) {
            self.input_transformations = Some(transformations.clone());
        }
        let delta = event.get("delta").unwrap_or(&Value::Null);
        if let Some(reason) = delta
            .get("stop_reason")
            .and_then(Value::as_str)
            .filter(|reason| !reason.is_empty())
        {
            builder.output.raw_stop_reason = Some(reason.to_owned());
            let (stop_reason, error_message) = map_stop_reason(reason, delta.get("stop_details"))?;
            builder.output.stop_reason = stop_reason;
            if error_message.is_some() {
                builder.output.error_message = error_message;
            }
        }
        // Only present fields overwrite: proxies may omit input_tokens in message_delta.
        if let Some(usage) = event.get("usage").filter(|usage| usage.is_object()) {
            let output = &mut builder.output.usage;
            if let Some(input) = token_count(usage.get("input_tokens")) {
                output.input = input;
            }
            if let Some(tokens) = token_count(usage.get("output_tokens")) {
                output.output = tokens;
            }
            if let Some(read) = token_count(usage.get("cache_read_input_tokens")) {
                output.cache_read = read;
            }
            if let Some(write) = token_count(usage.get("cache_creation_input_tokens")) {
                output.cache_write = write;
            }
            // Vercel AI Gateway reports the TTL breakdown in deltas.
            if let Some(long) = token_count(
                usage
                    .get("cache_creation")
                    .and_then(|creation| creation.get("ephemeral_1h_input_tokens")),
            ) {
                output.cache_write_1h = Some(long);
            }
            // Anthropic reports reasoning tokens as a subset of output tokens.
            if let Some(thinking) = token_count(
                usage
                    .get("output_tokens_details")
                    .and_then(|details| details.get("thinking_tokens")),
            ) {
                output.reasoning = Some(thinking);
            }
        }
        self.update_usage(builder);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Simple options
// ---------------------------------------------------------------------------

/// Map a reasoning level to an Anthropic effort for adaptive thinking. The model's thinking
/// level map wins; otherwise `xhigh` and `max` become `high`.
fn map_thinking_level_to_effort(model: &Model, level: ThinkingLevel) -> AnthropicEffort {
    if let Some(Some(mapped)) = model.mapped_thinking_level(level.into()) {
        return match mapped {
            "low" => AnthropicEffort::Low,
            "medium" => AnthropicEffort::Medium,
            "high" => AnthropicEffort::High,
            "xhigh" => AnthropicEffort::Xhigh,
            "max" => AnthropicEffort::Max,
            other => AnthropicEffort::Other(other.to_owned()),
        };
    }
    match level {
        ThinkingLevel::Minimal | ThinkingLevel::Low => AnthropicEffort::Low,
        ThinkingLevel::Medium => AnthropicEffort::Medium,
        ThinkingLevel::High | ThinkingLevel::Xhigh | ThinkingLevel::Max => AnthropicEffort::High,
    }
}

/// Stream with provider-neutral options: `reasoning` maps to an adaptive effort or a
/// thinking budget, `tool_choice` to Anthropic's tool choice.
#[allow(clippy::needless_pass_by_value)] // The shared API module signature takes options by value.
pub fn stream_simple(
    model: &Model,
    context: &TranscriptContext,
    options: SimpleStreamOptions,
) -> AssistantMessageEventStream {
    let mut base = build_base_options(model, context, &options);
    let tool_choice = options.tool_choice.map(|choice| match choice {
        ToolChoice::Auto => AnthropicToolChoice::Auto,
        ToolChoice::None => AnthropicToolChoice::None,
    });
    let mut api = AnthropicOptions {
        tool_choice,
        ..AnthropicOptions::default()
    };
    match options.reasoning {
        None => api.thinking_enabled = Some(false),
        Some(level) if Compat::of(model).force_adaptive_thinking => {
            api.thinking_enabled = Some(true);
            api.effort = Some(map_thinking_level_to_effort(model, level));
        }
        Some(level) => {
            let (max_tokens, budget) = adjust_max_tokens_for_thinking(
                base.max_tokens,
                model.max_tokens,
                level,
                options.thinking_budgets.as_ref(),
            );
            let max_tokens = clamp_max_tokens_to_context(model, context, max_tokens);
            base.max_tokens = Some(max_tokens);
            api.thinking_enabled = Some(true);
            api.thinking_budget_tokens = Some(budget.min(max_tokens.saturating_sub(1024)));
        }
    }
    base.api = Some(ApiOptions::AnthropicMessages(api));
    stream(model, context, base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::constrained_sampling::{UnsupportedStrictSchemaKeywordCheck, make_strict_json_schema};
    use crate::types::{TextContent, ThinkingContent, ToolCall};
    use pretty_assertions::assert_eq;

    fn tool(name: &str, parameters: Value, strict: Option<StrictMode>) -> Tool {
        Tool {
            constrained_sampling: strict.map(|strict| ConstrainedSampling::JsonSchema { strict }),
            ..Tool::new(name, "Look up a value", parameters)
        }
    }

    #[test]
    fn normalizes_tool_call_ids() {
        assert_eq!(normalize_tool_call_id("call_1|fc/abc"), "call_1_fc_abc");
        assert_eq!(normalize_tool_call_id(&"a".repeat(80)), "a".repeat(64));
        // An astral character is two UTF-16 units, so two underscores (as upstream's regex).
        assert_eq!(normalize_tool_call_id("x\u{1F600}y"), "x__y");
    }

    #[test]
    fn claude_code_names_round_trip() {
        let tools = vec![Tool::new("todowrite", "", json!({})), Tool::new("read", "", json!({}))];
        assert_eq!(to_claude_code_name("todowrite"), "TodoWrite");
        assert_eq!(to_claude_code_name("find"), "find");
        assert_eq!(from_claude_code_name("TodoWrite", &tools), "todowrite");
        assert_eq!(from_claude_code_name("Read", &tools), "read");
        assert_eq!(from_claude_code_name("Glob", &tools), "Glob");
        assert_eq!(from_claude_code_name("Glob", &[]), "Glob");
        assert!(is_oauth_token("sk-ant-oat01-x"));
        assert!(!is_oauth_token("sk-ant-api03-x"));
    }

    #[test]
    fn strict_schema_makes_optional_properties_nullable() {
        let schema = json!({
            "type": "object",
            "title": "T",
            "properties": {"value": {"type": "string"}, "optional": {"type": "number"}},
            "required": ["value"]
        });
        let strict = make_strict_json_schema(&schema, Some(is_strict_unsupported_keyword)).unwrap();
        assert_eq!(
            strict,
            json!({
                "type": "object",
                "title": "T",
                "properties": {
                    "value": {"type": "string"},
                    "optional": {"anyOf": [{"type": "number"}, {"type": "null"}]}
                },
                "required": ["value", "optional"],
                "additionalProperties": false
            })
        );
    }

    #[test]
    fn strict_schema_rejections() {
        let check = Some(is_strict_unsupported_keyword as UnsupportedStrictSchemaKeywordCheck);
        let reject = |schema: Value| make_strict_json_schema(&schema, check).unwrap_err().0;
        assert_eq!(
            reject(json!({"type": "object", "properties": {"a": {"type": "integer", "minimum": 1}}})),
            "minimum: 1 is unsupported"
        );
        assert_eq!(
            reject(json!({"type": "object", "properties": {"a": {"type": "array", "minItems": 2}}})),
            "minItems: 2 is unsupported"
        );
        assert_eq!(
            reject(json!({"type": "object", "properties": {"a": {"type": "string", "format": "regex"}}})),
            "format: \"regex\" is unsupported"
        );
        assert_eq!(
            reject(json!({"type": "object", "$defs": {}})),
            "$defs schemas are unsupported"
        );
        assert_eq!(
            reject(json!({"type": "object", "additionalProperties": true})),
            "schema-valued or true additionalProperties is unsupported"
        );
        assert_eq!(
            reject(json!({"type": "object", "required": ["x"]})),
            "required contains an unknown property"
        );
        assert_eq!(
            reject(json!({"anyOf": [{"type": "object"}]})),
            "object and array unions are unsupported"
        );
        assert_eq!(reject(json!({"type": "string"})), "root schema must have type object");
        assert_eq!(reject(json!(true)), "root schema must have type object");
        // Accepted keywords.
        make_strict_json_schema(
            &json!({"type": "object", "properties": {
                "code": {"type": "string", "minLength": 1, "pattern": "^[a-z]+$"},
                "url": {"type": "string", "format": "uri"},
                "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1}
            }}),
            check,
        )
        .unwrap();
    }

    #[test]
    fn strict_resolution_by_mode() {
        let resolve = |tool: &Tool, supported: bool| {
            resolve_json_schema_strict_sampling(tool, supported, Some(is_strict_unsupported_keyword))
        };
        let params = json!({"type": "object", "properties": {"v": {"type": "string"}}});
        let unsupported = json!({"type": "object", "properties": {"v": {"type": "number", "minimum": 1}}});
        assert_eq!(
            resolve(&tool("t", params.clone(), Some(StrictMode::Prefer)), true).unwrap(),
            Some(true)
        );
        assert_eq!(
            resolve(&tool("t", unsupported.clone(), Some(StrictMode::Prefer)), true).unwrap(),
            None
        );
        assert_eq!(
            resolve(&tool("t", params.clone(), Some(StrictMode::Prefer)), false).unwrap(),
            None
        );
        assert_eq!(resolve(&tool("t", params.clone(), None), true).unwrap(), None);
        let error = resolve(&tool("t", unsupported, Some(StrictMode::Require)), true)
            .unwrap_err()
            .to_string();
        assert_eq!(
            error,
            "Tool \"t\" requires JSON-schema constrained sampling, but minimum: 1 is unsupported."
        );
        let error = resolve(&tool("t", params, Some(StrictMode::Require)), false)
            .unwrap_err()
            .to_string();
        assert!(error.ends_with("but strict tools are unsupported."));
    }

    #[test]
    fn tool_definitions_have_upstream_key_order() {
        let mut converter = ToolConverter {
            is_oauth: false,
            eager_input_streaming: true,
            supports_strict_tools: true,
            disable_preferred_strict_tools: false,
            has_preferred_strict_tools: false,
        };
        let tools = vec![
            tool(
                "a",
                json!({"type": "object", "title": "A", "properties": {"v": {"type": "string"}}}),
                Some(StrictMode::Prefer),
            ),
            tool("b", json!({"type": "object", "properties": {}}), None),
        ];
        let cache = json!({"type": "ephemeral"});
        let definitions = converter.convert(&tools, Some(&cache)).unwrap();
        let keys: Vec<&str> = definitions[0].as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            ["name", "description", "eager_input_streaming", "strict", "input_schema"]
        );
        let schema_keys: Vec<&str> = definitions[0]["input_schema"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            schema_keys,
            ["type", "title", "properties", "required", "additionalProperties"]
        );
        assert!(converter.has_preferred_strict_tools);
        assert_eq!(
            definitions[1],
            json!({"name": "b", "description": "Look up a value", "eager_input_streaming": true,
                   "input_schema": {"type": "object", "properties": {}, "required": []},
                   "cache_control": {"type": "ephemeral"}})
        );
    }

    #[test]
    fn stop_reasons() {
        assert_eq!(map_stop_reason("end_turn", None).unwrap().0, StopReason::Stop);
        assert_eq!(map_stop_reason("pause_turn", None).unwrap().0, StopReason::Stop);
        assert_eq!(map_stop_reason("stop_sequence", None).unwrap().0, StopReason::Stop);
        assert_eq!(map_stop_reason("max_tokens", None).unwrap().0, StopReason::Length);
        assert_eq!(map_stop_reason("tool_use", None).unwrap().0, StopReason::ToolUse);
        assert_eq!(
            map_stop_reason("refusal", Some(&json!({"explanation": "no"}))).unwrap(),
            (StopReason::Error, Some("no".into()))
        );
        assert_eq!(
            map_stop_reason("refusal", None).unwrap().1.as_deref(),
            Some("The model refused to complete the request")
        );
        assert_eq!(
            map_stop_reason("sensitive", None).unwrap().1.as_deref(),
            Some("Provider stopped with: sensitive")
        );
        assert_eq!(
            map_stop_reason("new_reason", None).unwrap_err().to_string(),
            "Unhandled stop reason: new_reason"
        );
    }

    #[test]
    fn effort_messages_reconstruct_the_marker_prefix() {
        let converted = ConvertedMessages {
            messages: vec![json!({"role": "user", "content": "one"}), json!({"role": "assistant"})],
            assistant_levels: [(1, "low".to_owned())].into_iter().collect(),
        };
        assert_eq!(
            insert_thinking_level_messages(converted, "high"),
            vec![
                json!({"role": "user", "content": "one"}),
                json!({"role": "system", "content": [], "output_config": {"effort": "low"}}),
                json!({"role": "assistant"}),
                json!({"role": "system", "content": [], "output_config": {"effort": "high"}}),
            ]
        );
    }

    #[test]
    fn assistant_blocks_replay_rules() {
        let mut assistant = AssistantMessage::empty(API_ID, "anthropic", "m");
        assistant.content = vec![
            AssistantBlock::Thinking(ThinkingContent {
                thinking: "[Reasoning redacted]".into(),
                thinking_signature: Some("opaque".into()),
                redacted: Some(true),
            }),
            AssistantBlock::Thinking(ThinkingContent::new("  ")),
            AssistantBlock::Thinking(ThinkingContent::new("unsigned")),
            AssistantBlock::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: Some("sig".into()),
                redacted: None,
            }),
            AssistantBlock::Text(TextContent::new("   ")),
            AssistantBlock::Text(TextContent::new("answer")),
            AssistantBlock::ToolCall(ToolCall {
                id: "t1".into(),
                name: "read".into(),
                ..ToolCall::default()
            }),
        ];
        assert_eq!(
            convert_assistant_blocks(&assistant, true, false),
            vec![
                json!({"type": "redacted_thinking", "data": "opaque"}),
                json!({"type": "text", "text": "unsigned"}),
                json!({"type": "thinking", "thinking": "", "signature": "sig"}),
                json!({"type": "text", "text": "answer"}),
                json!({"type": "tool_use", "id": "t1", "name": "Read", "input": {}}),
            ]
        );
        assert_eq!(
            convert_assistant_blocks(&assistant, false, true)[1],
            json!({"type": "thinking", "thinking": "unsigned", "signature": ""})
        );
    }

    #[test]
    fn request_headers_leave_the_body() {
        let mut params = json!({"model": "m", "betas": ["a", "b"], "user_profile_id": "p", "stream": true});
        let headers = take_request_headers(&mut params);
        assert_eq!(params, json!({"model": "m", "stream": true}));
        assert_eq!(headers.get("anthropic-beta"), Some(&Some("a,b".to_owned())));
        assert_eq!(headers.get("anthropic-user-profile-id"), Some(&Some("p".to_owned())));
        let mut params = json!({"betas": []});
        assert!(take_request_headers(&mut params).is_empty());
    }
}
