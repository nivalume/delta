//! Logic shared by the Gemini API (`google-generative-ai`) and Vertex AI (`google-vertex`)
//! implementations. Port of upstream `src/api/google-shared.ts`.
//!
//! Everything here works on plain data and does not depend on a transport: the Gemini API
//! module serializes [`GenerateRequest`] to REST JSON and parses REST chunks into
//! [`ResponseChunk`]; the Vertex AI module maps the same types to and from the
//! `google-cloud-aiplatform-v1` proto types. The decisions (thinking configuration, which
//! parts to replay, thought signature rules, schema handling, finish reasons, usage) and the
//! streaming state machine ([`StreamState`]) are shared.

use serde_json::Value;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::api::constrained_sampling;
use crate::error::{Error, Result};
use crate::events::MessageBuilder;
use crate::http;
use crate::model::{Model, ModelThinkingLevel, ThinkingLevel, calculate_cost};
use crate::options::{
    GoogleThinking, GoogleThinkingLevel, GoogleToolChoice, StreamOptions, ThinkingBudgets, ToolChoice,
};
use crate::transcript::{
    collapse_system_messages, get_current_tools, get_initial_system_message, get_system_message_text,
    without_initial_system_message,
};
use crate::transform::transform_messages;
use crate::types::{
    AssistantBlock, AssistantMessage, JsonObject, Message, StopReason, Tool, TranscriptContext, Usage, UserBlock,
    UserContent, now_ms,
};

// ---------------------------------------------------------------------------
// Thinking levels and thinking configuration
// ---------------------------------------------------------------------------

/// A pi thinking level resolved to one of Gemini's standard levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResolvedGoogleThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
}

impl ResolvedGoogleThinkingLevel {
    /// Lowercase pi name (`minimal`, `low`, ...).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// Resolve a supported pi level, or the model's `thinking_level_map` value for it, to a
/// standard Google level. Mapped values are compared case-insensitively.
pub fn resolve_google_thinking_level(model: &Model, level: ModelThinkingLevel) -> Result<ResolvedGoogleThinkingLevel> {
    let mapped = model.mapped_thinking_level(level);
    let resolved = match mapped {
        Some(Some(value)) => value.to_lowercase(),
        _ => level.as_str().to_owned(),
    };
    match resolved.as_str() {
        "minimal" => Ok(ResolvedGoogleThinkingLevel::Minimal),
        "low" => Ok(ResolvedGoogleThinkingLevel::Low),
        "medium" => Ok(ResolvedGoogleThinkingLevel::Medium),
        "high" => Ok(ResolvedGoogleThinkingLevel::High),
        _ => {
            // `String(mapped)` upstream: the value, `null`, or `undefined`.
            let mapped_text = match mapped {
                Some(Some(value)) => value.to_owned(),
                Some(None) => "null".to_owned(),
                None => "undefined".to_owned(),
            };
            Err(Error::msg(format!(
                "Unsupported Google thinking level mapping for {}/{}: {} -> {mapped_text}",
                model.provider,
                model.id,
                level.as_str()
            )))
        }
    }
}

/// Every position right after an occurrence of `needle` in `haystack`.
fn after_occurrences<'a>(haystack: &'a str, needle: &'a str) -> impl Iterator<Item = &'a str> + 'a {
    haystack
        .match_indices(needle)
        .map(move |(index, _)| &haystack[index + needle.len()..])
}

fn starts_with_digit(text: &str) -> bool {
    text.bytes().next().is_some_and(|byte| byte.is_ascii_digit())
}

/// Whether this model uses Gemini's discrete `thinkingLevel` control instead of the
/// token-based `thinkingBudget`. Supported levels come from the model's thinking level map;
/// this only selects the wire format.
pub fn uses_google_thinking_level(model: &Model) -> bool {
    let id = model.id.to_lowercase();
    // /gemini-3(?:\.\d+)?-(?:pro|flash)/: Gemini 3 Pro/Flash with or without a minor version.
    let gemini_3 = after_occurrences(&id, "gemini-3").any(|rest| {
        let rest = match rest.strip_prefix('.') {
            Some(minor) if starts_with_digit(minor) => minor.trim_start_matches(|c: char| c.is_ascii_digit()),
            _ => rest,
        };
        rest.starts_with("-pro") || rest.starts_with("-flash")
    });
    // /gemini-(?:[4-9]|[1-9]\d)/: Gemini 4 and later.
    let gemini_4_plus = after_occurrences(&id, "gemini-").any(|rest| {
        let bytes = rest.as_bytes();
        match bytes.first() {
            Some(b'4'..=b'9') => true,
            Some(b'1'..=b'9') => bytes.get(1).is_some_and(u8::is_ascii_digit),
            _ => false,
        }
    });
    // /gemma-?4/: hosted Gemma 4 naming forms gemma-4-* and gemma4-*.
    let gemma_4 = after_occurrences(&id, "gemma").any(|rest| rest.starts_with('4') || rest.starts_with("-4"));
    gemini_3 || gemini_4_plus || id == "gemini-flash-latest" || id == "gemini-flash-lite-latest" || gemma_4
}

/// The Google API level of a resolved level.
pub fn to_google_thinking_level(level: ResolvedGoogleThinkingLevel) -> GoogleThinkingLevel {
    match level {
        ResolvedGoogleThinkingLevel::Minimal => GoogleThinkingLevel::Minimal,
        ResolvedGoogleThinkingLevel::Low => GoogleThinkingLevel::Low,
        ResolvedGoogleThinkingLevel::Medium => GoogleThinkingLevel::Medium,
        ResolvedGoogleThinkingLevel::High => GoogleThinkingLevel::High,
    }
}

/// Wire name of a Google `ThinkingLevel` enum value (`MINIMAL`, ..., or the raw value of
/// [`GoogleThinkingLevel::Other`], e.g. `THINKING_LEVEL_UNSPECIFIED`).
pub fn google_thinking_level_name(level: &GoogleThinkingLevel) -> &str {
    match level {
        GoogleThinkingLevel::Minimal => "MINIMAL",
        GoogleThinkingLevel::Low => "LOW",
        GoogleThinkingLevel::Medium => "MEDIUM",
        GoogleThinkingLevel::High => "HIGH",
        GoogleThinkingLevel::Other(value) => value,
    }
}

/// Gemini `ThinkingConfig`, transport-independent.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ThinkingConfig {
    pub include_thoughts: Option<bool>,
    pub thinking_budget: Option<i64>,
    pub thinking_level: Option<GoogleThinkingLevel>,
}

/// Configuration that turns thinking off, or the lowest supported level for models that
/// cannot disable it.
pub fn get_disabled_google_thinking_config(model: &Model) -> Result<ThinkingConfig> {
    let disabled = ThinkingConfig {
        thinking_budget: Some(0),
        ..ThinkingConfig::default()
    };
    if !uses_google_thinking_level(model) {
        return Ok(disabled);
    }
    let fallback = model.clamp_thinking_level(ModelThinkingLevel::Off);
    if fallback == ModelThinkingLevel::Off {
        return Ok(disabled);
    }
    let resolved = resolve_google_thinking_level(model, fallback)?;
    Ok(ThinkingConfig {
        thinking_level: Some(to_google_thinking_level(resolved)),
        ..ThinkingConfig::default()
    })
}

/// The request's thinking configuration for the `thinking` option (upstream `buildParams`).
///
/// - enabled on a reasoning model: `includeThoughts` plus the level, else the budget;
/// - explicitly disabled on a reasoning model: [`get_disabled_google_thinking_config`];
/// - otherwise none.
pub fn resolve_thinking_config(model: &Model, thinking: Option<&GoogleThinking>) -> Result<Option<ThinkingConfig>> {
    let Some(thinking) = thinking else {
        return Ok(None);
    };
    if !model.reasoning {
        return Ok(None);
    }
    if !thinking.enabled {
        return get_disabled_google_thinking_config(model).map(Some);
    }
    let mut config = ThinkingConfig {
        include_thoughts: Some(true),
        ..ThinkingConfig::default()
    };
    if let Some(level) = &thinking.level {
        config.thinking_level = Some(level.clone());
    } else if let Some(budget) = thinking.budget_tokens {
        config.thinking_budget = Some(budget);
    }
    Ok(Some(config))
}

/// Default token budget for a level on budget-based Gemini models; `-1` is dynamic.
pub fn get_google_budget(model: &Model, level: ResolvedGoogleThinkingLevel, custom: Option<&ThinkingBudgets>) -> i64 {
    let custom_budget = custom.and_then(|budgets| match level {
        ResolvedGoogleThinkingLevel::Minimal => budgets.minimal,
        ResolvedGoogleThinkingLevel::Low => budgets.low,
        ResolvedGoogleThinkingLevel::Medium => budgets.medium,
        ResolvedGoogleThinkingLevel::High => budgets.high,
    });
    if let Some(budget) = custom_budget {
        return budget as i64;
    }
    let table: [i64; 4] = if model.id.contains("2.5-pro") {
        [128, 2048, 8192, 32768]
    } else if model.id.contains("2.5-flash-lite") {
        [512, 2048, 8192, 24576]
    } else if model.id.contains("2.5-flash") {
        [128, 2048, 8192, 24576]
    } else {
        return -1;
    };
    match level {
        ResolvedGoogleThinkingLevel::Minimal => table[0],
        ResolvedGoogleThinkingLevel::Low => table[1],
        ResolvedGoogleThinkingLevel::Medium => table[2],
        ResolvedGoogleThinkingLevel::High => table[3],
    }
}

/// The `thinking` option of a simple request (upstream `streamSimple`): disabled without a
/// reasoning level or when the model clamps it to `off`; otherwise a level for models with
/// discrete levels and a token budget for the rest.
pub fn simple_thinking(
    model: &Model,
    reasoning: Option<ThinkingLevel>,
    budgets: Option<&ThinkingBudgets>,
) -> Result<GoogleThinking> {
    simple_thinking_with(model, reasoning, budgets, get_google_budget)
}

/// Token budget for a level, given custom budgets; `-1` is dynamic. The Gemini API and Vertex
/// have different default tables.
pub type GoogleBudgetFn = fn(&Model, ResolvedGoogleThinkingLevel, Option<&ThinkingBudgets>) -> i64;

/// [`simple_thinking`] with the budget table of the API.
pub fn simple_thinking_with(
    model: &Model,
    reasoning: Option<ThinkingLevel>,
    budgets: Option<&ThinkingBudgets>,
    budget: GoogleBudgetFn,
) -> Result<GoogleThinking> {
    let disabled = GoogleThinking {
        enabled: false,
        ..GoogleThinking::default()
    };
    let Some(reasoning) = reasoning else {
        return Ok(disabled);
    };
    let clamped = model.clamp_thinking_level(reasoning.into());
    if clamped == ModelThinkingLevel::Off {
        return Ok(disabled);
    }
    let resolved = resolve_google_thinking_level(model, clamped)?;
    if uses_google_thinking_level(model) {
        return Ok(GoogleThinking {
            enabled: true,
            budget_tokens: None,
            level: Some(to_google_thinking_level(resolved)),
        });
    }
    Ok(GoogleThinking {
        enabled: true,
        budget_tokens: Some(budget(model, resolved, budgets)),
        level: None,
    })
}

/// Provider-neutral tool choice as a Gemini tool choice.
pub fn simple_tool_choice(choice: ToolChoice) -> GoogleToolChoice {
    match choice {
        ToolChoice::Auto => GoogleToolChoice::Auto,
        ToolChoice::None => GoogleToolChoice::None,
    }
}

// ---------------------------------------------------------------------------
// Thought signatures
// ---------------------------------------------------------------------------

/// Whether a streamed part is thinking content.
///
/// `thought: true` is the definitive marker. `thoughtSignature` is an encrypted
/// representation of the model's internal reasoning that can appear on ANY part (text,
/// function call, ...) and does not make the part thinking. Signature-bearing parts are
/// replayed as-is; signatures are never merged or moved across parts.
/// See <https://ai.google.dev/gemini-api/docs/thought-signatures>.
pub fn is_thinking_part(thought: Option<bool>) -> bool {
    thought == Some(true)
}

/// Keep the last non-empty signature of the current streamed block. Some backends send the
/// signature only on the first delta of a part; this never moves signatures across parts.
pub fn retain_thought_signature(existing: Option<String>, incoming: Option<&str>) -> Option<String> {
    match incoming {
        Some(incoming) if !incoming.is_empty() => Some(incoming.to_owned()),
        _ => existing,
    }
}

/// Thought signatures are base64 (`TYPE_BYTES`): standard alphabet, padded to a multiple of 4.
pub fn is_valid_thought_signature(signature: &str) -> bool {
    if signature.is_empty() || !signature.len().is_multiple_of(4) {
        return false;
    }
    let body = signature.trim_end_matches('=');
    let padding = signature.len() - body.len();
    padding <= 2
        && !body.is_empty()
        && body
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'+' || byte == b'/')
}

/// A signature to replay: only from the same provider and model, and only when valid base64.
pub fn resolve_thought_signature(same_provider_and_model: bool, signature: Option<&str>) -> Option<String> {
    signature
        .filter(|signature| same_provider_and_model && is_valid_thought_signature(signature))
        .map(str::to_owned)
}

// ---------------------------------------------------------------------------
// Model capabilities
// ---------------------------------------------------------------------------

/// Major version of a `gemini-<n>` or `gemini-live-<n>` model id.
pub fn gemini_major_version(model_id: &str) -> Option<u64> {
    let id = model_id.to_lowercase();
    let rest = id.strip_prefix("gemini")?;
    let rest = rest.strip_prefix("-live").unwrap_or(rest);
    let rest = rest.strip_prefix('-')?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    if digits.is_empty() {
        return None;
    }
    Some(digits.parse().unwrap_or(u64::MAX))
}

/// Models behind Google APIs that require explicit tool call ids in function calls and
/// responses.
pub fn requires_tool_call_id(model_id: &str) -> bool {
    model_id.starts_with("claude-")
        || model_id.starts_with("gpt-oss-")
        || gemini_major_version(model_id).is_some_and(|major| major >= 3)
}

/// Gemini 3+ accepts images nested inside `functionResponse.parts`; older Gemini models get
/// a separate user image turn. Non-Gemini models are assumed to support it.
pub fn supports_multimodal_function_response(model_id: &str) -> bool {
    gemini_major_version(model_id).is_none_or(|major| major >= 3)
}

/// Gemini 3+ enforces required function parameters in validated tool-calling modes.
pub fn supports_google_strict_tool_sampling(model_id: &str) -> bool {
    gemini_major_version(model_id).is_some_and(|major| major >= 3)
}

/// Tool call id for a model that requires ids: characters outside `[a-zA-Z0-9_-]` become
/// `_` (one per UTF-16 unit, as upstream's regex replace), truncated to 64.
pub fn normalize_tool_call_id(model_id: &str, id: &str) -> String {
    if !requires_tool_call_id(model_id) {
        return id.to_owned();
    }
    let mut normalized = String::with_capacity(id.len());
    for c in id.chars() {
        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
            normalized.push(c);
        } else {
            for _ in 0..c.len_utf16() {
                normalized.push('_');
            }
        }
    }
    normalized.truncate(64);
    normalized
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/// Role of a [`Content`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    User,
    Model,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Model => "model",
        }
    }
}

/// Inline media data (base64).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InlineData {
    pub mime_type: String,
    pub data: String,
}

/// A request part.
#[derive(Clone, Debug, PartialEq)]
pub enum Part {
    /// Text; `thought` marks replayed thinking.
    Text {
        text: String,
        thought: bool,
        thought_signature: Option<String>,
    },
    InlineData(InlineData),
    FunctionCall {
        /// Sent only to models that require tool call ids.
        id: Option<String>,
        name: String,
        args: JsonObject,
        thought_signature: Option<String>,
    },
    FunctionResponse {
        /// Sent only to models that require tool call ids.
        id: Option<String>,
        name: String,
        /// `{"output": ...}` or `{"error": ...}`.
        response: JsonObject,
        /// Images nested in the response (Gemini 3+).
        parts: Vec<InlineData>,
    },
}

impl Part {
    fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            thought: false,
            thought_signature: None,
        }
    }
}

/// A request turn.
#[derive(Clone, Debug, PartialEq)]
pub struct Content {
    pub role: Role,
    pub parts: Vec<Part>,
}

fn user_block_part(block: &UserBlock) -> Part {
    match block {
        UserBlock::Text(text) => Part::text(text.text.clone()),
        UserBlock::Image(image) => Part::InlineData(InlineData {
            mime_type: image.mime_type.clone(),
            data: image.data.clone(),
        }),
    }
}

fn is_blank(text: &str) -> bool {
    text.trim().is_empty()
}

fn assistant_parts(model: &Model, message: &AssistantMessage, include_id: bool) -> Vec<Part> {
    // Thinking and signatures are kept only for the same provider and model.
    let same = message.provider == model.provider && message.model == model.id;
    let mut parts = Vec::new();
    for block in &message.content {
        match block {
            AssistantBlock::Text(text) => {
                let signature = resolve_thought_signature(same, text.text_signature.as_deref());
                // Skip empty text unless it carries a signature: Gemini can sign a part with
                // empty visible text and requires the signature echoed back.
                if is_blank(&text.text) && signature.is_none() {
                    continue;
                }
                parts.push(Part::Text {
                    text: text.text.clone(),
                    thought: false,
                    thought_signature: signature,
                });
            }
            AssistantBlock::Thinking(thinking) => {
                if same {
                    let signature = resolve_thought_signature(same, thinking.thinking_signature.as_deref());
                    if is_blank(&thinking.thinking) && signature.is_none() {
                        continue;
                    }
                    parts.push(Part::Text {
                        text: thinking.thinking.clone(),
                        thought: true,
                        thought_signature: signature,
                    });
                } else {
                    // Another model's signature is unusable; send readable thinking as text.
                    // (transform_messages already drops foreign thinking, as upstream.)
                    if is_blank(&thinking.thinking) {
                        continue;
                    }
                    parts.push(Part::text(thinking.thinking.clone()));
                }
            }
            AssistantBlock::ToolCall(call) => parts.push(Part::FunctionCall {
                id: include_id.then(|| call.id.clone()),
                name: call.name.clone(),
                args: call.arguments.clone(),
                thought_signature: resolve_thought_signature(same, call.thought_signature.as_deref()),
            }),
        }
    }
    parts
}

/// Convert a transcript to Gemini contents. Gemini has no mid-conversation system messages:
/// they are collapsed into the leading prompt, which is sent as `systemInstruction`.
pub fn convert_messages(model: &Model, context: &TranscriptContext) -> Vec<Content> {
    let collapsed = collapse_system_messages(context);
    let conversation = without_initial_system_message(collapsed.messages());
    let include_id = requires_tool_call_id(&model.id);
    let normalize = |id: &str, _: &Model, _: &AssistantMessage| normalize_tool_call_id(&model.id, id);
    let mut contents: Vec<Content> = Vec::new();

    for message in transform_messages(conversation, model, Some(&normalize)) {
        match message {
            Message::User(user) => {
                let parts: Vec<Part> = match &user.content {
                    UserContent::Text(text) => vec![Part::text(text.clone())],
                    UserContent::Blocks(blocks) => blocks.iter().map(user_block_part).collect(),
                };
                if parts.is_empty() {
                    continue;
                }
                contents.push(Content {
                    role: Role::User,
                    parts,
                });
            }
            Message::Assistant(assistant) => {
                let parts = assistant_parts(model, &assistant, include_id);
                if parts.is_empty() {
                    continue;
                }
                contents.push(Content {
                    role: Role::Model,
                    parts,
                });
            }
            Message::ToolResult(result) => {
                let text = result
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        UserBlock::Text(text) => Some(text.text.as_str()),
                        UserBlock::Image(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let images: Vec<InlineData> = if model.supports_images() {
                    result
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            UserBlock::Image(image) => Some(InlineData {
                                mime_type: image.mime_type.clone(),
                                data: image.data.clone(),
                            }),
                            UserBlock::Text(_) => None,
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                let has_text = !is_blank(&text);
                let has_images = !images.is_empty();
                let multimodal = supports_multimodal_function_response(&model.id);
                let value = if has_text {
                    text
                } else if has_images {
                    "(see attached image)".to_owned()
                } else {
                    String::new()
                };
                let mut response = JsonObject::new();
                response.insert(
                    if result.is_error { "error" } else { "output" }.to_owned(),
                    Value::String(value),
                );
                let part = Part::FunctionResponse {
                    id: include_id.then(|| result.tool_call_id.clone()),
                    name: result.tool_name.clone(),
                    response,
                    parts: if has_images && multimodal {
                        images.clone()
                    } else {
                        Vec::new()
                    },
                };
                // All function responses of a turn go in one user turn (Cloud Code Assist
                // requires it): merge into the previous turn when it holds responses.
                match contents.last_mut() {
                    Some(last)
                        if last.role == Role::User
                            && last
                                .parts
                                .iter()
                                .any(|part| matches!(part, Part::FunctionResponse { .. })) =>
                    {
                        last.parts.push(part);
                    }
                    _ => contents.push(Content {
                        role: Role::User,
                        parts: vec![part],
                    }),
                }
                // Gemini < 3: images follow in a separate user turn.
                if has_images && !multimodal {
                    let mut parts = vec![Part::text("Tool result image:")];
                    parts.extend(images.into_iter().map(Part::InlineData));
                    contents.push(Content {
                        role: Role::User,
                        parts,
                    });
                }
            }
            Message::System(_) => {}
        }
    }
    contents
}

// ---------------------------------------------------------------------------
// Tools and JSON schemas
// ---------------------------------------------------------------------------

const JSON_SCHEMA_META_DECLARATIONS: [&str; 8] = [
    "$schema",
    "$id",
    "$anchor",
    "$dynamicAnchor",
    "$vocabulary",
    "$comment",
    "$defs",
    // Pre-2019-09 equivalent of $defs.
    "definitions",
];

/// Copy of `schema` without JSON Schema meta declarations, at every depth of nested objects
/// (`$ref` is kept). For the legacy OpenAPI `parameters` field.
pub fn sanitize_for_open_api(schema: &Value) -> Value {
    match schema {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .filter(|(key, _)| !JSON_SCHEMA_META_DECLARATIONS.contains(&key.as_str()))
                .map(|(key, value)| (key.clone(), sanitize_for_open_api(value)))
                .collect(),
        ),
        other => other.clone(),
    }
}

pub use crate::api::constrained_sampling::{UnsupportedStrictSchema, get_json_schema_tool_parameters};

/// Convert a tool schema to the strict subset used by provider constrained sampling: every
/// property required (optional ones become nullable) and no additional properties. Google
/// adds no keyword check; see [`crate::api::constrained_sampling::make_strict_json_schema`].
pub fn make_strict_json_schema(schema: &Value) -> std::result::Result<Value, UnsupportedStrictSchema> {
    constrained_sampling::make_strict_json_schema(schema, None)
}

/// Whether a JSON-schema constrained tool is sent in strict mode: `Some(true)` when strict,
/// `None` otherwise. Fails when the tool requires strict sampling that cannot be honored.
/// See [`crate::api::constrained_sampling::resolve_json_schema_strict_sampling`].
pub fn resolve_json_schema_strict_sampling(tool: &Tool, supports_strict_mode: bool) -> Result<Option<bool>> {
    constrained_sampling::resolve_json_schema_strict_sampling(tool, supports_strict_mode, None)
}

/// Parameters of a function declaration.
#[derive(Clone, Debug, PartialEq)]
pub enum ToolParameters {
    /// `parametersJsonSchema`: full JSON Schema (`anyOf`, `oneOf`, `const`, ...).
    JsonSchema(Value),
    /// Legacy `parameters` (OpenAPI 3.0 schema), meta declarations stripped.
    OpenApi(Value),
}

/// A Gemini function declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: ToolParameters,
}

/// Convert tools to function declarations (one `tools` entry upstream). `use_parameters`
/// selects the legacy OpenAPI `parameters` field, which Cloud Code Assist needs for Claude
/// models. `None` for an empty tool list.
pub fn convert_tools(
    tools: &[Tool],
    use_parameters: bool,
    supports_strict_mode: bool,
) -> Result<Option<Vec<FunctionDeclaration>>> {
    if tools.is_empty() {
        return Ok(None);
    }
    let mut declarations = Vec::with_capacity(tools.len());
    for tool in tools {
        let strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)?;
        let parameters = get_json_schema_tool_parameters(tool, strict)?;
        declarations.push(FunctionDeclaration {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: if use_parameters {
                ToolParameters::OpenApi(sanitize_for_open_api(&parameters))
            } else {
                ToolParameters::JsonSchema(parameters)
            },
        });
    }
    Ok(Some(declarations))
}

/// Gemini `FunctionCallingConfigMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FunctionCallingMode {
    Auto,
    None,
    Any,
    Validated,
}

impl FunctionCallingMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "AUTO",
            Self::None => "NONE",
            Self::Any => "ANY",
            Self::Validated => "VALIDATED",
        }
    }
}

/// A tool choice as a function calling mode.
pub fn map_tool_choice(choice: GoogleToolChoice) -> FunctionCallingMode {
    match choice {
        GoogleToolChoice::Auto => FunctionCallingMode::Auto,
        GoogleToolChoice::None => FunctionCallingMode::None,
        GoogleToolChoice::Any => FunctionCallingMode::Any,
    }
}

/// The function calling mode: `none`/`any` as chosen, else `VALIDATED` when a tool is sent
/// in strict mode, else the chosen mode (none when unset).
pub fn resolve_google_function_calling_mode(
    tools: &[Tool],
    tool_choice: Option<GoogleToolChoice>,
    supports_strict_mode: bool,
) -> Result<Option<FunctionCallingMode>> {
    let mut use_strict_mode = false;
    for tool in tools {
        if resolve_json_schema_strict_sampling(tool, supports_strict_mode)? == Some(true) {
            use_strict_mode = true;
            break;
        }
    }
    if let Some(choice @ (GoogleToolChoice::None | GoogleToolChoice::Any)) = tool_choice {
        return Ok(Some(map_tool_choice(choice)));
    }
    if use_strict_mode {
        return Ok(Some(FunctionCallingMode::Validated));
    }
    Ok(tool_choice.map(map_tool_choice))
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// The request options the shared request builder reads.
#[derive(Clone, Copy, Debug, Default)]
pub struct RequestSettings<'a> {
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub tool_choice: Option<GoogleToolChoice>,
    pub thinking: Option<&'a GoogleThinking>,
}

/// A transport-independent `generateContent` request (upstream `GenerateContentParameters`
/// minus the model).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerateRequest {
    pub contents: Vec<Content>,
    pub system_instruction: Option<String>,
    pub temperature: Option<f64>,
    pub max_output_tokens: Option<u64>,
    pub tools: Option<Vec<FunctionDeclaration>>,
    pub function_calling_mode: Option<FunctionCallingMode>,
    pub thinking_config: Option<ThinkingConfig>,
}

/// Build the request for a transcript (upstream `buildParams`).
pub fn build_request(
    model: &Model,
    context: &TranscriptContext,
    settings: &RequestSettings<'_>,
) -> Result<GenerateRequest> {
    let collapsed = collapse_system_messages(context);
    let messages = collapsed.messages();
    let contents = convert_messages(model, &collapsed);
    let tools = get_current_tools(messages);
    let supports_strict_mode = supports_google_strict_tool_sampling(&model.id);
    let function_calling_mode = if tools.is_empty() {
        None
    } else {
        resolve_google_function_calling_mode(&tools, settings.tool_choice, supports_strict_mode)?
    };
    let system_instruction = get_initial_system_message(messages)
        .map(get_system_message_text)
        .filter(|text| !text.is_empty());
    Ok(GenerateRequest {
        contents,
        system_instruction,
        temperature: settings.temperature,
        max_output_tokens: settings.max_tokens,
        tools: convert_tools(&tools, false, supports_strict_mode)?,
        function_calling_mode,
        thinking_config: resolve_thinking_config(model, settings.thinking)?,
    })
}

// ---------------------------------------------------------------------------
// Finish reasons and usage
// ---------------------------------------------------------------------------

/// Map a Gemini finish reason to a stop reason: `STOP` → stop, `MAX_TOKENS` → length,
/// everything else (safety, recitation, malformed calls, unknown values) → error.
pub fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "STOP" => StopReason::Stop,
        "MAX_TOKENS" => StopReason::Length,
        _ => StopReason::Error,
    }
}

/// Gemini `usageMetadata` counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UsageMetadata {
    pub prompt_token_count: Option<u64>,
    pub cached_content_token_count: Option<u64>,
    pub candidates_token_count: Option<u64>,
    pub thoughts_token_count: Option<u64>,
    pub total_token_count: Option<u64>,
}

/// Usage with cost from usage metadata. `None` until `promptTokenCount` is reported, so a
/// partial early report does not replace a complete one.
pub fn parse_usage(model: &Model, metadata: &UsageMetadata) -> Option<Usage> {
    let prompt = metadata.prompt_token_count?;
    let cached = metadata.cached_content_token_count.unwrap_or(0);
    let thoughts = metadata.thoughts_token_count.unwrap_or(0);
    let mut usage = Usage {
        input: prompt.saturating_sub(cached),
        output: metadata.candidates_token_count.unwrap_or(0).saturating_add(thoughts),
        cache_read: cached,
        cache_write: 0,
        reasoning: Some(thoughts),
        total_tokens: metadata.total_token_count.unwrap_or(0),
        ..Usage::default()
    };
    calculate_cost(&model.cost, &mut usage);
    Some(usage)
}

// ---------------------------------------------------------------------------
// Responses and the stream state machine
// ---------------------------------------------------------------------------

/// A function call of a response part.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResponseFunctionCall {
    pub id: Option<String>,
    pub name: Option<String>,
    pub args: Option<JsonObject>,
}

/// A response part. Fields the stream does not use are not represented.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResponsePart {
    pub text: Option<String>,
    pub thought: Option<bool>,
    /// Base64 text.
    pub thought_signature: Option<String>,
    pub function_call: Option<ResponseFunctionCall>,
}

/// The first candidate of a streamed response chunk.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Candidate {
    pub parts: Vec<ResponsePart>,
    pub finish_reason: Option<String>,
}

/// One streamed `GenerateContentResponse`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResponseChunk {
    pub response_id: Option<String>,
    pub candidate: Option<Candidate>,
    pub usage: Option<UsageMetadata>,
}

static TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpenBlock {
    Text(usize),
    Thinking(usize),
}

/// Turns response chunks into message events (upstream's stream loop). Consecutive text
/// parts extend one text block, consecutive thought parts one thinking block; a function
/// call closes the open block and is emitted complete.
#[derive(Debug, Default)]
pub struct StreamState {
    open: Option<OpenBlock>,
}

impl StreamState {
    pub fn new() -> Self {
        Self::default()
    }

    fn end_open_block(&mut self, builder: &mut MessageBuilder) {
        match self.open.take() {
            Some(OpenBlock::Text(index)) => builder.end_text(index),
            Some(OpenBlock::Thinking(index)) => builder.end_thinking(index),
            None => {}
        }
    }

    fn text_part(&mut self, builder: &mut MessageBuilder, text: &str, thinking: bool, signature: Option<&str>) {
        let reuse = matches!(
            (self.open, thinking),
            (Some(OpenBlock::Thinking(_)), true) | (Some(OpenBlock::Text(_)), false)
        );
        if !reuse {
            self.end_open_block(builder);
            self.open = Some(if thinking {
                OpenBlock::Thinking(builder.start_thinking(String::new(), false, None))
            } else {
                OpenBlock::Text(builder.start_text())
            });
        }
        match self.open {
            Some(OpenBlock::Thinking(index)) => {
                builder.thinking_delta(index, text);
                let existing = builder
                    .output
                    .content
                    .get(index)
                    .and_then(AssistantBlock::as_thinking)
                    .and_then(|block| block.thinking_signature.clone());
                builder.set_thinking_signature(index, retain_thought_signature(existing, signature));
            }
            Some(OpenBlock::Text(index)) => {
                builder.text_delta(index, text);
                let existing = builder
                    .output
                    .content
                    .get(index)
                    .and_then(AssistantBlock::as_text)
                    .and_then(|block| block.text_signature.clone());
                builder.set_text_signature(index, retain_thought_signature(existing, signature));
            }
            None => {}
        }
    }

    fn function_call(&mut self, builder: &mut MessageBuilder, call: &ResponseFunctionCall, signature: Option<&str>) {
        self.end_open_block(builder);
        // Generate a unique id when none is provided or it repeats one of this message.
        let provided = call.id.as_deref().filter(|id| !id.is_empty());
        let id = match provided {
            Some(id) if !builder.output.tool_calls().any(|existing| existing.id == id) => id.to_owned(),
            _ => format!(
                "{}_{}_{}",
                call.name.as_deref().unwrap_or("undefined"),
                now_ms(),
                TOOL_CALL_COUNTER.fetch_add(1, Ordering::Relaxed) + 1
            ),
        };
        let args = call.args.clone().unwrap_or_default();
        let json = Value::Object(args.clone()).to_string();
        let index = builder.start_tool_call(id, call.name.clone().unwrap_or_default(), args);
        if let Some(signature) = signature.filter(|signature| !signature.is_empty())
            && let Some(tool_call) = builder.tool_call_mut(index)
        {
            tool_call.thought_signature = Some(signature.to_owned());
        }
        builder.tool_call_delta(index, &json);
        builder.end_tool_call(index);
    }

    /// Apply one chunk: content parts, finish reason and usage.
    pub fn apply_chunk(&mut self, builder: &mut MessageBuilder, model: &Model, chunk: &ResponseChunk) {
        // The response id identifies the response; keep the first non-empty one.
        if builder.output.response_id.as_deref().is_none_or(str::is_empty)
            && let Some(id) = chunk.response_id.as_deref().filter(|id| !id.is_empty())
        {
            builder.output.response_id = Some(id.to_owned());
        }
        if let Some(candidate) = &chunk.candidate {
            for part in &candidate.parts {
                let signature = part.thought_signature.as_deref();
                if let Some(text) = &part.text {
                    self.text_part(builder, text, is_thinking_part(part.thought), signature);
                }
                if let Some(call) = &part.function_call {
                    self.function_call(builder, call, signature);
                }
            }
            if let Some(reason) = candidate.finish_reason.as_deref().filter(|reason| !reason.is_empty()) {
                let output = &mut builder.output;
                output.raw_stop_reason = Some(reason.to_owned());
                output.stop_reason = map_stop_reason(reason);
                // Gemini reports STOP after function calls.
                if output.stop_reason == StopReason::Stop && output.tool_calls().next().is_some() {
                    output.stop_reason = StopReason::ToolUse;
                }
            }
        }
        if let Some(usage) = chunk.usage.as_ref().and_then(|metadata| parse_usage(model, metadata)) {
            builder.output.usage = usage;
        }
    }

    /// End the open text or thinking block after the last chunk.
    pub fn finish(&mut self, builder: &mut MessageBuilder) {
        self.end_open_block(builder);
    }
}

/// The outcome of a stream that ended: an error when no finish reason arrived
/// (`missing_finish_reason`) or the finish reason maps to an error.
pub fn check_finished(output: &AssistantMessage, missing_finish_reason: &str) -> Result<()> {
    match output.stop_reason {
        StopReason::Pending => Err(Error::msg(missing_finish_reason)),
        StopReason::Error | StopReason::Aborted => Err(Error::msg(match &output.raw_stop_reason {
            Some(raw) => format!("Provider stopped with: {raw}"),
            None => "An unknown error occurred".to_owned(),
        })),
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Retries
// ---------------------------------------------------------------------------

fn backoff(retry_index: u32) -> Duration {
    let base = (0.5 * 2f64.powi(retry_index as i32)).min(8.0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    let jitter = f64::from(nanos % 1000) / 1000.0 * 0.25;
    Duration::from_secs_f64(base * (1.0 - jitter))
}

/// Give an HTTP error the message `@google/genai` gives it (`ApiError` from
/// `throwErrorIfNotOK`): the JSON body re-serialized, or for a body that is not JSON
/// `{"error":{"message":<text>,"code":<status>,"status":<status text>}}`. Upstream's
/// `formatProviderError` keeps that message as it is. Other errors are returned unchanged.
pub fn format_google_http_error(error: Error) -> Error {
    let Error::Status {
        status,
        body,
        retry_after_ms,
        ..
    } = error
    else {
        return error;
    };
    let text = body.as_deref().unwrap_or("");
    let message = match serde_json::from_str::<Value>(text) {
        Ok(json) => json.to_string(),
        Err(_) => {
            let status_text = reqwest::StatusCode::from_u16(status)
                .ok()
                .and_then(|code| code.canonical_reason())
                .unwrap_or("");
            serde_json::json!({"error": {"message": text, "code": status, "status": status_text}}).to_string()
        }
    };
    Error::Status {
        status,
        message,
        body,
        retry_after_ms,
    }
}

/// Run a request with the shared provider retry policy (upstream `retryGoogleRequest`), for
/// transports that do not go through [`http::send_with_retry`] (the Vertex AI SDK).
///
/// [`Error::Status`] errors with a retryable status (408, 409, 429, 5xx) are retried up to
/// `options.max_retries` times; a server delay in `retry_after_ms` is honored and fails at
/// once when above `options.max_retry_delay`, otherwise exponential backoff applies. Other
/// errors (transport failures, error events inside an accepted stream) are not retried.
/// Sleeping and each attempt are interrupted by `options.cancel`.
pub async fn retry_google_request<T, F, Fut>(options: &StreamOptions, mut request: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let max_retries = options.max_retries.unwrap_or(http::DEFAULT_MAX_RETRIES);
    let max_delay = options.max_retry_delay.unwrap_or(http::DEFAULT_MAX_RETRY_DELAY);
    let cancel = options.cancel.as_ref();
    let mut retry_index = 0;
    loop {
        let error = match http::or_cancel(cancel, request()).await {
            Ok(value) => return Ok(value),
            Err(error) => error,
        };
        let Error::Status {
            status, retry_after_ms, ..
        } = &error
        else {
            return Err(error);
        };
        let retryable = matches!(status, 408 | 409 | 429) || *status >= 500;
        if retry_index >= max_retries || !retryable {
            return Err(error);
        }
        let delay = match retry_after_ms.map(Duration::from_millis) {
            Some(delay) if !max_delay.is_zero() && delay > max_delay => {
                return Err(Error::msg(format!(
                    "Server requested {}s retry delay (max: {}s). {error}",
                    delay.as_secs_f64().ceil(),
                    max_delay.as_secs_f64().ceil()
                )));
            }
            Some(delay) => delay,
            None => backoff(retry_index),
        };
        retry_index += 1;
        http::or_cancel(cancel, async {
            tokio::time::sleep(delay).await;
            Ok(())
        })
        .await?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::normalize_context;
    use crate::types::{ConstrainedSampling, StrictMode};
    use crate::types::{Context, ImageContent, TextContent, ThinkingContent, ToolCall, ToolResultMessage, UserMessage};
    use pretty_assertions::assert_eq;
    use serde_json::json;
    use std::sync::atomic::AtomicU32;

    const VALID_SIG: &str = "AAAAAAAAAAAAAAAAAAAAAA==";

    fn model_with(id: &str, extra: Value) -> Model {
        let mut value = json!({"id": id, "name": id, "api": "google-generative-ai", "provider": "google",
            "baseUrl": "https://example.com", "reasoning": true, "input": ["text"],
            "cost": {"input": 1, "output": 2, "cacheRead": 0.5, "cacheWrite": 0},
            "contextWindow": 128_000, "maxTokens": 8192});
        if let (Value::Object(target), Value::Object(extra)) = (&mut value, extra) {
            target.extend(extra);
        }
        serde_json::from_value(value).unwrap()
    }

    fn model(id: &str) -> Model {
        model_with(id, json!({}))
    }

    fn assistant(api: &str, provider: &str, id: &str, content: Vec<AssistantBlock>) -> Message {
        let mut message = AssistantMessage::empty(api, provider, id);
        message.content = content;
        message.stop_reason = StopReason::ToolUse;
        Message::Assistant(message)
    }

    fn call(id: &str, args: Value) -> AssistantBlock {
        AssistantBlock::ToolCall(ToolCall {
            id: id.into(),
            name: "bash".into(),
            arguments: match args {
                Value::Object(arguments) => arguments,
                _ => JsonObject::new(),
            },
            ..ToolCall::default()
        })
    }

    fn result(id: &str, name: &str, content: Vec<UserBlock>, is_error: bool) -> Message {
        Message::ToolResult(ToolResultMessage {
            tool_call_id: id.into(),
            tool_name: name.into(),
            content,
            is_error,
            ..ToolResultMessage::default()
        })
    }

    fn convert(model: &Model, messages: Vec<Message>) -> Vec<Content> {
        convert_messages(model, &normalize_context(&Context::new(messages)))
    }

    fn all_parts(contents: &[Content]) -> Vec<&Part> {
        contents.iter().flat_map(|content| &content.parts).collect()
    }

    fn tool(parameters: Value) -> Tool {
        Tool::new("test_tool", "A test tool", parameters)
    }

    // --- convertTools (google-shared-convert-tools.test.ts) ---

    fn open_api_parameters(parameters: Value) -> Value {
        let declarations = convert_tools(&[tool(parameters)], true, true).unwrap().unwrap();
        match &declarations[0].parameters {
            ToolParameters::OpenApi(value) => value.clone(),
            ToolParameters::JsonSchema(_) => panic!("expected parameters"),
        }
    }

    #[test]
    fn strips_meta_keys_recursively_and_keeps_refs() {
        let parameters = json!({
            "$schema": "http://json-schema.org/draft-07/schema#", "$id": "urn:bash-tool",
            "$comment": "demo", "$defs": {"commandDef": {"type": "string"}},
            "definitions": {"legacyDef": {"type": "number"}},
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "deep": {"$schema": "x", "$id": "urn:nested", "type": "string"},
                "refProp": {"$ref": "#/$defs/someDef", "type": "string"}
            },
            "required": ["command"]
        });
        let original = parameters.clone();
        assert_eq!(
            open_api_parameters(parameters.clone()),
            json!({"type": "object", "properties": {
                "command": {"type": "string"},
                "deep": {"type": "string"},
                "refProp": {"$ref": "#/$defs/someDef", "type": "string"}
            }, "required": ["command"]})
        );
        assert_eq!(parameters, original);
        assert_eq!(
            open_api_parameters(json!({"type": "object", "properties": {"path": {"type": "string"}}})),
            json!({"type": "object", "properties": {"path": {"type": "string"}}})
        );
    }

    #[test]
    fn keeps_schema_in_parameters_json_schema() {
        let parameters = json!({"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
            "properties": {"command": {"type": "string"}}, "required": ["command"]});
        let declarations = convert_tools(&[tool(parameters.clone())], false, true)
            .unwrap()
            .unwrap();
        assert_eq!(declarations[0].parameters, ToolParameters::JsonSchema(parameters));
        assert_eq!(convert_tools(&[], false, true).unwrap(), None);
        assert_eq!(convert_tools(&[], true, true).unwrap(), None);
    }

    #[test]
    fn validated_function_calling_for_strict_tools() {
        let mut strict = tool(json!({"type": "object", "properties": {}}));
        strict.constrained_sampling = Some(ConstrainedSampling::JsonSchema {
            strict: StrictMode::Require,
        });
        assert!(supports_google_strict_tool_sampling("gemini-3.1-pro-preview"));
        assert!(!supports_google_strict_tool_sampling("gemini-2.5-pro"));
        assert_eq!(
            resolve_google_function_calling_mode(std::slice::from_ref(&strict), None, true).unwrap(),
            Some(FunctionCallingMode::Validated)
        );
        let error = resolve_google_function_calling_mode(&[strict], None, false).unwrap_err();
        assert!(
            error
                .to_string()
                .starts_with("Tool \"test_tool\" requires JSON-schema constrained sampling")
        );
    }

    #[test]
    fn strict_parameters_and_explicit_choices() {
        let mut prefer = tool(
            json!({"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "null"}},
            "required": ["b"]}),
        );
        prefer.constrained_sampling = Some(ConstrainedSampling::JsonSchema {
            strict: StrictMode::Prefer,
        });
        let declarations = convert_tools(std::slice::from_ref(&prefer), false, true)
            .unwrap()
            .unwrap();
        assert_eq!(
            declarations[0].parameters,
            ToolParameters::JsonSchema(json!({"type": "object", "properties": {
                "a": {"anyOf": [{"type": "string"}, {"type": "null"}]}, "b": {"type": "null"}},
                "required": ["a", "b"], "additionalProperties": false}))
        );
        let tools = std::slice::from_ref(&prefer);
        let mode = |choice, strict| resolve_google_function_calling_mode(tools, choice, strict).unwrap();
        assert_eq!(
            mode(Some(GoogleToolChoice::None), true),
            Some(FunctionCallingMode::None)
        );
        assert_eq!(mode(Some(GoogleToolChoice::Any), true), Some(FunctionCallingMode::Any));
        assert_eq!(
            mode(Some(GoogleToolChoice::Auto), true),
            Some(FunctionCallingMode::Validated)
        );
        // Not strict when the provider cannot honor it and the tool only prefers it.
        assert_eq!(mode(None, false), None);
        assert_eq!(
            mode(Some(GoogleToolChoice::Auto), false),
            Some(FunctionCallingMode::Auto)
        );
    }

    // --- Gemini 3 unsigned tool calls (google-shared-gemini3-unsigned-tool-call.test.ts) ---

    fn unsigned_calls(source: (&str, &str, &str), signature: Option<&str>) -> Vec<Message> {
        let mut first = call("call_1", json!({"command": "echo hi"}));
        if let AssistantBlock::ToolCall(call) = &mut first {
            call.thought_signature = signature.map(str::to_owned);
        }
        vec![
            Message::user("Hi"),
            assistant(
                source.0,
                source.1,
                source.2,
                vec![first, call("call_2", json!({"command": "ls -la"}))],
            ),
            result("call_1", "bash", vec![UserBlock::text("hi")], false),
            result("call_2", "bash", vec![UserBlock::text("files")], false),
        ]
    }

    fn call_ids(contents: &[Content]) -> (Vec<Option<String>>, Vec<Option<String>>) {
        let mut calls = Vec::new();
        let mut responses = Vec::new();
        for part in all_parts(contents) {
            match part {
                Part::FunctionCall { id, .. } => calls.push(id.clone()),
                Part::FunctionResponse { id, .. } => responses.push(id.clone()),
                _ => {}
            }
        }
        (calls, responses)
    }

    fn call_signatures(contents: &[Content]) -> Vec<Option<String>> {
        all_parts(contents)
            .into_iter()
            .filter_map(|part| match part {
                Part::FunctionCall { thought_signature, .. } => Some(thought_signature.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn gemini_3_keeps_tool_call_ids() {
        for id in ["gemini-3-pro-preview", "gemini-3.6-flash"] {
            let m = model(id);
            let contents = convert(&m, unsigned_calls(("google-generative-ai", "google", id), None));
            let ids = Some(vec![Some("call_1".to_owned()), Some("call_2".to_owned())]);
            let (calls, responses) = call_ids(&contents);
            assert_eq!(Some(calls), ids);
            assert_eq!(Some(responses), ids);
        }
    }

    #[test]
    fn unsigned_tool_calls_get_no_signature() {
        let m = model("gemini-3-pro-preview");
        let contents = convert(
            &m,
            unsigned_calls(("google-generative-ai", "google", "other-model"), None),
        );
        assert_eq!(call_signatures(&contents), vec![None, None]);
        assert!(!format!("{contents:?}").contains("skip_thought_signature_validator"));
        let contents = convert(
            &m,
            unsigned_calls(
                ("google-generative-ai", "google", "gemini-3-pro-preview"),
                Some(VALID_SIG),
            ),
        );
        assert_eq!(call_signatures(&contents), vec![Some(VALID_SIG.to_owned()), None]);
        for invalid in ["not base64!", "AAA", "AAAA\n", "===="] {
            let contents = convert(
                &m,
                unsigned_calls(
                    ("google-generative-ai", "google", "gemini-3-pro-preview"),
                    Some(invalid),
                ),
            );
            assert_eq!(call_signatures(&contents), vec![None, None], "{invalid:?}");
        }
    }

    #[test]
    fn non_gemini_3_gets_no_ids_or_signatures() {
        let m = model("gemini-2.5-flash");
        let contents = convert(
            &m,
            unsigned_calls(("google-generative-ai", "google", "other-model"), Some(VALID_SIG)),
        );
        let (calls, responses) = call_ids(&contents);
        assert_eq!(calls, vec![None, None]);
        assert_eq!(responses, vec![None, None]);
        assert_eq!(call_signatures(&contents), vec![None, None]);
    }

    #[test]
    fn requires_tool_call_id_by_model() {
        for (expected, id) in [
            (false, "gemini-2.5-flash"),
            (true, "gemini-3.6-flash"),
            (true, "gemini-live-3-x"),
            (true, "claude-sonnet-4-5"),
            (true, "gpt-oss-120b"),
            (false, "gemma-3"),
        ] {
            assert_eq!(requires_tool_call_id(id), expected, "{id}");
        }
    }

    #[test]
    fn normalizes_foreign_tool_call_ids() {
        let m = model("gemini-3-pro-preview");
        let long_id = format!("call|{}", "x".repeat(80));
        let contents = convert(
            &m,
            vec![
                Message::user("Hi"),
                assistant("openai-responses", "openai", "gpt-5", vec![call(&long_id, json!({}))]),
                result(&long_id, "bash", vec![UserBlock::text("ok")], false),
            ],
        );
        let expected: String = format!("call_{}", "x".repeat(80)).chars().take(64).collect();
        let (calls, responses) = call_ids(&contents);
        assert_eq!(calls, vec![Some(expected.clone())]);
        assert_eq!(responses, vec![Some(expected)]);
        assert_eq!(normalize_tool_call_id("gemini-3-pro", "a\u{1F600}b"), "a__b");
        assert_eq!(normalize_tool_call_id("gemini-2.5-pro", "a|b"), "a|b");
    }

    // --- Image tool result routing (google-shared-image-tool-result-routing.test.ts) ---

    fn image_routing(model: &Model) -> Vec<Message> {
        vec![
            Message::user("read the files"),
            assistant(
                &model.api,
                &model.provider,
                &model.id,
                vec![
                    call("call_a", json!({"path": "a.txt"})),
                    call("call_img", json!({"path": "image.png"})),
                    call("call_b", json!({"path": "b.txt"})),
                ],
            ),
            result("call_a", "read", vec![UserBlock::text("alpha text")], false),
            result(
                "call_img",
                "read",
                vec![UserBlock::Image(ImageContent {
                    data: "abc".into(),
                    mime_type: "image/png".into(),
                })],
                false,
            ),
            result("call_b", "read", vec![UserBlock::text("beta text")], false),
        ]
    }

    fn vision(id: &str) -> Model {
        model_with(id, json!({"input": ["text", "image"]}))
    }

    #[test]
    fn separate_image_turn_for_gemini_2() {
        let m = vision("gemini-2.5-flash");
        let contents = convert(&m, image_routing(&m));
        assert_eq!(contents.len(), 5);
        assert!(
            contents[2]
                .parts
                .iter()
                .all(|part| matches!(part, Part::FunctionResponse { .. }))
        );
        assert_eq!(contents[3].parts[0], Part::text("Tool result image:"));
        assert!(matches!(contents[3].parts[1], Part::InlineData(_)));
        assert!(matches!(contents[4].parts[0], Part::FunctionResponse { .. }));
    }

    #[test]
    fn nests_images_for_gemini_3() {
        let m = vision("gemini-3-pro-preview");
        let contents = convert(&m, image_routing(&m));
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[2].parts.len(), 3);
        let Part::FunctionResponse { parts, response, .. } = &contents[2].parts[1] else {
            panic!("expected a function response");
        };
        assert_eq!(parts.len(), 1);
        assert_eq!(response, json!({"output": "(see attached image)"}).as_object().unwrap());
    }

    #[test]
    fn whitespace_only_tool_result_text_counts_as_no_text() {
        let m = vision("gemini-3-pro-preview");
        let mut messages = image_routing(&m);
        if let Message::ToolResult(r) = &mut messages[2] {
            r.content = vec![UserBlock::text("\r\n")];
        }
        if let Message::ToolResult(r) = &mut messages[3] {
            r.content.insert(0, UserBlock::text(" \n"));
        }
        let contents = convert(&m, messages);
        let responses: Vec<Value> = contents[2]
            .parts
            .iter()
            .map(|part| match part {
                Part::FunctionResponse { response, .. } => Value::Object(response.clone()),
                _ => Value::Null,
            })
            .collect();
        assert_eq!(
            responses,
            vec![
                json!({"output": ""}),
                json!({"output": "(see attached image)"}),
                json!({"output": "beta text"})
            ]
        );
    }

    #[test]
    fn error_results_use_error_key() {
        let m = model("gemini-2.5-flash");
        let contents = convert(
            &m,
            vec![
                Message::user("go"),
                assistant(&m.api, &m.provider, &m.id, vec![call("c1", json!({}))]),
                result("c1", "bash", vec![UserBlock::text("boom")], true),
            ],
        );
        assert_eq!(
            contents.last().unwrap().parts[0],
            Part::FunctionResponse {
                id: None,
                name: "bash".into(),
                response: json!({"error": "boom"}).as_object().unwrap().clone(),
                parts: vec![]
            }
        );
    }

    #[test]
    fn orphaned_tool_calls_get_error_results() {
        let m = model("gemini-3-pro-preview");
        let contents = convert(
            &m,
            vec![
                Message::user("go"),
                assistant(&m.api, &m.provider, &m.id, vec![call("c1", json!({}))]),
                Message::user("never mind"),
            ],
        );
        assert_eq!(contents.len(), 4);
        assert_eq!(
            contents[2].parts[0],
            Part::FunctionResponse {
                id: Some("c1".into()),
                name: "bash".into(),
                response: json!({"error": "No result provided"}).as_object().unwrap().clone(),
                parts: vec![]
            }
        );
    }

    // --- Signed empty blocks (google-shared-signed-empty-blocks.test.ts) ---

    fn signed(source_id: &str, content: Vec<AssistantBlock>) -> Vec<Content> {
        let m = model("gemini-3-pro-preview");
        let contents = convert(
            &m,
            vec![
                Message::user("Hi"),
                assistant("google-generative-ai", "google", source_id, content),
            ],
        );
        contents.into_iter().filter(|c| c.role == Role::Model).collect()
    }

    fn thinking(text: &str, signature: Option<&str>) -> AssistantBlock {
        AssistantBlock::Thinking(ThinkingContent {
            thinking: text.into(),
            thinking_signature: signature.map(str::to_owned),
            redacted: None,
        })
    }

    fn text(text: &str, signature: Option<&str>) -> AssistantBlock {
        AssistantBlock::Text(TextContent {
            text: text.into(),
            text_signature: signature.map(str::to_owned),
        })
    }

    #[test]
    fn keeps_signed_empty_blocks_of_the_same_model() {
        let turns = signed(
            "gemini-3-pro-preview",
            vec![thinking("", Some(VALID_SIG)), call("call_1", json!({"command": "ls"}))],
        );
        assert_eq!(
            turns[0].parts[0],
            Part::Text {
                text: String::new(),
                thought: true,
                thought_signature: Some(VALID_SIG.into())
            }
        );
        let turns = signed(
            "gemini-3-pro-preview",
            vec![text("", Some(VALID_SIG)), call("call_1", json!({"command": "ls"}))],
        );
        assert_eq!(
            turns[0].parts[0],
            Part::Text {
                text: String::new(),
                thought: false,
                thought_signature: Some(VALID_SIG.into())
            }
        );
    }

    #[test]
    fn drops_unsigned_and_foreign_empty_blocks() {
        let turns = signed(
            "gemini-3-pro-preview",
            vec![
                thinking("", None),
                text("   ", None),
                call("call_1", json!({"command": "ls"})),
            ],
        );
        assert_eq!(turns[0].parts.len(), 1);
        assert!(matches!(turns[0].parts[0], Part::FunctionCall { .. }));
        let turns = signed(
            "other-model",
            vec![
                thinking("", Some(VALID_SIG)),
                text("", Some(VALID_SIG)),
                call("call_1", json!({"command": "ls"})),
            ],
        );
        assert_eq!(turns[0].parts.len(), 1);
        assert!(!format!("{turns:?}").contains(VALID_SIG));
    }

    #[test]
    fn user_content_and_same_model_thinking() {
        let m = vision("gemini-3-pro-preview");
        let mut answer = AssistantMessage::empty(&m.api, &m.provider, &m.id);
        answer.content = vec![thinking("hmm", Some(VALID_SIG)), text("answer", None)];
        answer.stop_reason = StopReason::Stop;
        let mut failed = AssistantMessage::empty(&m.api, &m.provider, &m.id);
        failed.content = vec![text("partial", None)];
        failed.stop_reason = StopReason::Error;
        let context = Context::new(vec![
            Message::user("plain"),
            Message::User(UserMessage {
                content: UserContent::Blocks(vec![
                    UserBlock::text("look"),
                    UserBlock::Image(ImageContent {
                        data: "b64".into(),
                        mime_type: "image/jpeg".into(),
                    }),
                ]),
                timestamp: 0,
            }),
            Message::Assistant(answer),
            Message::Assistant(failed),
        ])
        .with_system_prompt("system");
        let contents = convert_messages(&m, &normalize_context(&context));
        assert_eq!(
            contents,
            vec![
                Content {
                    role: Role::User,
                    parts: vec![Part::text("plain")]
                },
                Content {
                    role: Role::User,
                    parts: vec![
                        Part::text("look"),
                        Part::InlineData(InlineData {
                            mime_type: "image/jpeg".into(),
                            data: "b64".into()
                        })
                    ]
                },
                Content {
                    role: Role::Model,
                    parts: vec![
                        Part::Text {
                            text: "hmm".into(),
                            thought: true,
                            thought_signature: Some(VALID_SIG.into())
                        },
                        Part::text("answer")
                    ]
                },
            ]
        );
    }

    // --- Thought signatures (google-thinking-signature.test.ts) ---

    #[test]
    fn thinking_detection_and_signature_retention() {
        assert!(is_thinking_part(Some(true)));
        assert!(!is_thinking_part(Some(false)));
        assert!(!is_thinking_part(None));
        let first = retain_thought_signature(None, Some("sig-1"));
        assert_eq!(first.as_deref(), Some("sig-1"));
        let second = retain_thought_signature(first, None);
        assert_eq!(second.as_deref(), Some("sig-1"));
        let third = retain_thought_signature(second, Some(""));
        assert_eq!(third.as_deref(), Some("sig-1"));
        assert_eq!(retain_thought_signature(third, Some("sig-2")).as_deref(), Some("sig-2"));
    }

    // --- Thinking levels (google-thinking-level-map.test.ts, resolver part) ---

    fn mapped(id: &str, map: Value) -> Model {
        let mut extra = json!({"provider": "test-google"});
        extra["thinkingLevelMap"] = map;
        model_with(id, extra)
    }

    #[test]
    fn resolves_levels_and_mappings() {
        use ModelThinkingLevel as L;
        use ResolvedGoogleThinkingLevel as R;
        for (level, expected) in [
            (L::Minimal, R::Minimal),
            (L::Low, R::Low),
            (L::Medium, R::Medium),
            (L::High, R::High),
        ] {
            assert_eq!(
                resolve_google_thinking_level(&mapped("gemini-3.7-flash", json!({})), level).unwrap(),
                expected
            );
        }
        for (value, expected) in [
            ("minimal", R::Minimal),
            ("low", R::Low),
            ("MEDIUM", R::Medium),
            ("HIGH", R::High),
        ] {
            let m = mapped("gemini-3.7-flash", json!({"high": value, "xhigh": value, "max": value}));
            for level in [L::High, L::Xhigh, L::Max] {
                assert_eq!(resolve_google_thinking_level(&m, level).unwrap(), expected);
            }
        }
        assert_eq!(
            resolve_google_thinking_level(&mapped("gemini-3.7-flash", json!({"xhigh": "extreme"})), L::Xhigh)
                .unwrap_err()
                .to_string(),
            "Unsupported Google thinking level mapping for test-google/gemini-3.7-flash: xhigh -> extreme"
        );
        assert_eq!(
            resolve_google_thinking_level(&mapped("gemini-3.7-flash", json!({})), L::Max)
                .unwrap_err()
                .to_string(),
            "Unsupported Google thinking level mapping for test-google/gemini-3.7-flash: max -> undefined"
        );
    }

    #[test]
    fn selects_thinking_level_models() {
        for (id, expected) in [
            ("gemini-3-flash-preview", true),
            ("gemini-3.1-pro-preview", true),
            ("gemini-3.8-flash", true),
            ("models/Gemini-3-Pro", true),
            ("gemini-4-argon", true),
            ("gemini-12-x", true),
            ("gemini-flash-latest", true),
            ("gemini-flash-lite-latest", true),
            ("gemma-4-31b-it", true),
            ("gemma4-e2b", true),
            ("gemini-2.5-flash", false),
            ("gemini-2.5-pro", false),
            ("gemini-3-image", false),
            ("gemini-3.-pro", false),
            ("gemini-1.5-pro", false),
        ] {
            assert_eq!(uses_google_thinking_level(&model(id)), expected, "{id}");
        }
    }

    #[test]
    fn disabled_thinking_config() {
        let budget_zero = ThinkingConfig {
            thinking_budget: Some(0),
            ..ThinkingConfig::default()
        };
        assert_eq!(
            get_disabled_google_thinking_config(&model("gemini-2.5-flash")).unwrap(),
            budget_zero
        );
        assert_eq!(
            get_disabled_google_thinking_config(&mapped("gemini-3-flash-preview", json!({}))).unwrap(),
            budget_zero
        );
        let no_off = mapped("gemini-3-pro-preview", json!({"off": null, "minimal": null}));
        assert_eq!(
            get_disabled_google_thinking_config(&no_off).unwrap(),
            ThinkingConfig {
                thinking_level: Some(GoogleThinkingLevel::Low),
                ..ThinkingConfig::default()
            }
        );
    }

    #[test]
    fn simple_thinking_and_budgets() {
        let flash = model("gemini-2.5-flash");
        assert_eq!(
            simple_thinking(&flash, None, None).unwrap(),
            GoogleThinking {
                enabled: false,
                ..GoogleThinking::default()
            }
        );
        assert_eq!(
            simple_thinking(&flash, Some(ThinkingLevel::Xhigh), None).unwrap(),
            GoogleThinking {
                enabled: true,
                budget_tokens: Some(24576),
                level: None
            }
        );
        assert_eq!(
            get_google_budget(&model("gemini-2.5-pro"), ResolvedGoogleThinkingLevel::High, None),
            32768
        );
        assert_eq!(
            get_google_budget(
                &model("gemini-2.5-flash-lite"),
                ResolvedGoogleThinkingLevel::Minimal,
                None
            ),
            512
        );
        assert_eq!(
            get_google_budget(&flash, ResolvedGoogleThinkingLevel::Minimal, None),
            128
        );
        assert_eq!(
            get_google_budget(&model("gemini-3-flash-preview"), ResolvedGoogleThinkingLevel::Low, None),
            -1
        );
        let custom = ThinkingBudgets {
            low: Some(77),
            ..ThinkingBudgets::default()
        };
        assert_eq!(
            get_google_budget(&flash, ResolvedGoogleThinkingLevel::Low, Some(&custom)),
            77
        );
        let not_reasoning = model_with("gemini-2.5-flash", json!({"reasoning": false}));
        assert_eq!(
            simple_thinking(&not_reasoning, Some(ThinkingLevel::High), None).unwrap(),
            GoogleThinking {
                enabled: false,
                ..GoogleThinking::default()
            }
        );
    }

    // --- Finish reasons and usage ---

    #[test]
    fn maps_finish_reasons() {
        assert_eq!(map_stop_reason("STOP"), StopReason::Stop);
        assert_eq!(map_stop_reason("MAX_TOKENS"), StopReason::Length);
        for reason in [
            "SAFETY",
            "MALFORMED_FUNCTION_CALL",
            "OTHER",
            "TOO_MANY_TOOL_CALLS",
            "NO_IMAGE",
            "SOMETHING_NEW",
        ] {
            assert_eq!(map_stop_reason(reason), StopReason::Error);
        }
    }

    #[test]
    fn usage_waits_for_prompt_count_and_saturates() {
        let m = model("gemini-2.5-flash");
        assert_eq!(
            parse_usage(
                &m,
                &UsageMetadata {
                    candidates_token_count: Some(3),
                    ..UsageMetadata::default()
                }
            ),
            None
        );
        let usage = parse_usage(
            &m,
            &UsageMetadata {
                prompt_token_count: Some(2),
                cached_content_token_count: Some(5),
                candidates_token_count: Some(3),
                thoughts_token_count: Some(4),
                total_token_count: Some(9),
            },
        )
        .unwrap();
        assert_eq!(
            (
                usage.input,
                usage.output,
                usage.cache_read,
                usage.reasoning,
                usage.total_tokens
            ),
            (0, 7, 5, Some(4), 9)
        );
        assert!(usage.cost.total > 0.0);
    }

    // --- Retries (google-shared-retry.test.ts) ---

    fn status(code: u16) -> Error {
        Error::Status {
            status: code,
            message: format!("got status: {code}"),
            body: None,
            retry_after_ms: None,
        }
    }

    async fn run_retry(max_retries: Option<u32>, outcomes: Vec<u16>) -> (Result<&'static str>, u32) {
        let calls = AtomicU32::new(0);
        let options = StreamOptions {
            max_retries,
            ..StreamOptions::default()
        };
        let result = retry_google_request(&options, || {
            let index = calls.fetch_add(1, Ordering::SeqCst) as usize;
            let outcome = outcomes.get(index).copied();
            async move {
                match outcome {
                    Some(code) => Err(status(code)),
                    None => Ok("ok"),
                }
            }
        })
        .await;
        (result, calls.load(Ordering::SeqCst))
    }

    #[tokio::test(start_paused = true)]
    async fn retries_retryable_statuses_only() {
        let (result, calls) = run_retry(Some(1), vec![429]).await;
        assert_eq!(result.unwrap(), "ok");
        assert_eq!(calls, 2);
        let (result, calls) = run_retry(Some(0), vec![429]).await;
        assert!(matches!(result, Err(Error::Status { status: 429, .. })));
        assert_eq!(calls, 1);
        let (result, calls) = run_retry(Some(2), vec![400]).await;
        assert!(matches!(result, Err(Error::Status { status: 400, .. })));
        assert_eq!(calls, 1);
    }

    #[tokio::test(start_paused = true)]
    async fn retry_honors_server_delay_cap() {
        let options = StreamOptions {
            max_retries: Some(1),
            max_retry_delay: Some(Duration::from_secs(1)),
            ..StreamOptions::default()
        };
        let result: Result<()> = retry_google_request(&options, || async {
            Err(Error::Status {
                status: 503,
                message: "503 busy".into(),
                body: None,
                retry_after_ms: Some(5000),
            })
        })
        .await;
        assert_eq!(
            result.unwrap_err().to_string(),
            "Server requested 5s retry delay (max: 1s). 503 busy"
        );
    }

    #[test]
    fn http_errors_carry_the_sdk_json_message() {
        let error = |status: u16, body: Option<&str>| {
            format_google_http_error(Error::Status {
                status,
                message: String::new(),
                body: body.map(str::to_owned),
                retry_after_ms: None,
            })
            .to_string()
        };
        // A JSON body is re-serialized (compact), as `JSON.stringify(await response.json())`.
        assert_eq!(
            error(
                400,
                Some(
                    "{\n  \"error\": {\n    \"code\": 400,\n    \"message\": \"API key not valid.\",\n    \"status\": \"INVALID_ARGUMENT\"\n  }\n}"
                )
            ),
            r#"{"error":{"code":400,"message":"API key not valid.","status":"INVALID_ARGUMENT"}}"#
        );
        // Other bodies are wrapped like the SDK does.
        assert_eq!(
            error(403, Some("blocked by gateway WAF")),
            r#"{"error":{"message":"blocked by gateway WAF","code":403,"status":"Forbidden"}}"#
        );
        assert_eq!(
            error(502, None),
            r#"{"error":{"message":"","code":502,"status":"Bad Gateway"}}"#
        );
        assert_eq!(format_google_http_error(Error::msg("boom")).to_string(), "boom");
    }
}
