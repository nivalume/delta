//! Message/tool conversion and stream processing shared by the OpenAI Responses API family
//! (`openai-responses`, `azure-openai-responses`).
//!
//! Port of upstream `src/api/openai-responses-shared.ts`, plus the parts of
//! `openai-prompt-cache.ts` it needs; constrained sampling and `shortHash` come from
//! [`crate::api::constrained_sampling`] and [`crate::api::common`]. Request items and stream
//! events are plain JSON values in the Responses wire format.

use futures::{Stream, StreamExt};
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::collections::HashMap;

use crate::api::common::observe_provider_event;
use crate::api::constrained_sampling::{
    GrammarToolInputJsonBuffer, append_grammar_tool_input_json_delta, get_grammar_tool_input,
    get_json_schema_tool_parameters, resolve_grammar_constrained_sampling, resolve_json_schema_strict_sampling,
};
use crate::compat::{ModelCompat, OpenAIResponsesCompat};
use crate::error::{Error, Result};
use crate::events::MessageBuilder;
use crate::http::SseEvent;
use crate::json::parse_tool_call_arguments;
use crate::model::{Model, calculate_cost};
use crate::options::StreamOptions;
use crate::transcript::{
    get_system_message_text, render_system_message_update, resolve_transcript, resolve_transcript_tools,
};
use crate::transform::transform_messages;
use crate::types::{
    AssistantBlock, AssistantMessage, JsonObject, Message, StopReason, SystemMessage, Tool, TranscriptContext, Usage,
    UserBlock, UserContent,
};

// =============================================================================
// Utilities
// =============================================================================

/// Re-exported from its old home here (upstream `utils/hash.ts`).
pub use crate::api::common::short_hash;

/// OpenAI's limit on `prompt_cache_key`.
pub const OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH: usize = 64;

/// Clamp a prompt cache key to [`OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH`] code points.
pub fn clamp_openai_prompt_cache_key(key: Option<&str>) -> Option<String> {
    key.map(|key| key.chars().take(OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH).collect())
}

/// The model's compat as the Responses schema. Compat parsed with another schema is
/// re-read leniently; unknown or invalid compat yields the defaults.
pub fn responses_compat(model: &Model) -> OpenAIResponsesCompat {
    match &model.compat {
        Some(ModelCompat::OpenAIResponses(compat)) => compat.clone(),
        Some(other) => serde_json::to_value(other)
            .ok()
            .and_then(|value| serde_json::from_value(value).ok())
            .unwrap_or_default(),
        None => OpenAIResponsesCompat::default(),
    }
}

/// Render a JSON field the way a JavaScript template literal would (`undefined`, `null`, ...).
fn js_template(value: Option<&Value>) -> String {
    match value {
        None => "undefined".into(),
        Some(Value::Null) => "null".into(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Bool(flag)) => flag.to_string(),
        Some(Value::Number(number)) => number.to_string(),
        Some(Value::Object(_)) => "[object Object]".into(),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item {
                Value::Null => String::new(),
                other => js_template(Some(other)),
            })
            .collect::<Vec<_>>()
            .join(","),
    }
}

/// JavaScript truthiness of a JSON value.
fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(flag)) => *flag,
        Some(Value::Number(number)) => number.as_f64().is_some_and(|n| n != 0.0),
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(_) | Value::Object(_)) => true,
    }
}

fn str_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

/// A non-empty string field (JavaScript `value || fallback`).
fn truthy_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    str_field(value, key).filter(|text| !text.is_empty())
}

/// A token count (`value || 0`).
fn count_field(value: Option<&Value>) -> u64 {
    value
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_f64().filter(|n| n.is_finite() && *n > 0.0).map(|n| n as u64))
        })
        .unwrap_or(0)
}

#[derive(Serialize)]
struct TextSignatureV1Out<'a> {
    v: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase: Option<&'a str>,
}

/// `{"v":1,"id":...,"phase":...}`; a missing id or empty phase is omitted.
fn encode_text_signature_v1(id: Option<&str>, phase: Option<&str>) -> String {
    serde_json::to_string(&TextSignatureV1Out {
        v: 1,
        id,
        phase: phase.filter(|phase| !phase.is_empty()),
    })
    .unwrap_or_default()
}

/// Message id and phase of a text signature: [`crate::types::TextSignatureV1`] JSON, or a
/// legacy plain id string. Only `commentary` and `final_answer` phases are kept.
fn parse_text_signature(signature: Option<&str>) -> Option<(String, Option<String>)> {
    let signature = signature.filter(|signature| !signature.is_empty())?;
    if signature.starts_with('{')
        && let Ok(Value::Object(parsed)) = serde_json::from_str::<Value>(signature)
        && parsed.get("v").and_then(Value::as_f64) == Some(1.0)
        && let Some(id) = parsed.get("id").and_then(Value::as_str)
    {
        let phase = parsed
            .get("phase")
            .and_then(Value::as_str)
            .filter(|phase| matches!(*phase, "commentary" | "final_answer"));
        return Some((id.to_owned(), phase.map(str::to_owned)));
    }
    Some((signature.to_owned(), None))
}

// =============================================================================
// Constrained sampling (shared logic in `api::constrained_sampling`)
// =============================================================================

/// Re-exported from its old home here.
pub use crate::api::constrained_sampling::create_grammar_tool_input_properties;

/// Tool name to the argument property that carries a grammar tool's raw input.
pub type GrammarToolInputProperties = HashMap<String, String>;

// =============================================================================
// Tool conversion
// =============================================================================

/// Options of [`convert_responses_tools`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConvertResponsesToolsOptions {
    /// Default `strict` of function tools: `None` is `false`, `Some(None)` sends `null`.
    pub strict: Option<Option<bool>>,
    /// Default: true.
    pub supports_strict_mode: Option<bool>,
    /// Default: false.
    pub supports_openai_grammar_tools: Option<bool>,
    /// Tools are a `tool_search_output` result (`defer_loading: true`).
    pub tool_search_result: bool,
}

/// Convert tool definitions to Responses `function` and grammar `custom` tools.
pub fn convert_responses_tools(tools: &[Tool], options: &ConvertResponsesToolsOptions) -> Result<Vec<Value>> {
    let default_strict = options.strict.unwrap_or(Some(false));
    let supports_strict_mode = options.supports_strict_mode.unwrap_or(true);
    let supports_grammar = options.supports_openai_grammar_tools.unwrap_or(false);
    tools
        .iter()
        .map(|tool| {
            if let Some(grammar) = resolve_grammar_constrained_sampling(tool, supports_grammar)? {
                let mut custom = json!({
                    "type": "custom",
                    "name": tool.name,
                    "description": tool.description,
                    "format": {"type": "grammar", "syntax": grammar.format.as_str(), "definition": grammar.definition},
                });
                if options.tool_search_result {
                    custom["defer_loading"] = Value::Bool(true);
                }
                return Ok(custom);
            }
            let strict =
                resolve_json_schema_strict_sampling(tool, supports_strict_mode, None)?.map_or(default_strict, Some);
            let mut function = json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": get_json_schema_tool_parameters(tool, strict)?,
            });
            if options.tool_search_result {
                function["defer_loading"] = Value::Bool(true);
            }
            if supports_strict_mode {
                function["strict"] = strict.map_or(Value::Null, Value::Bool);
            }
            Ok(function)
        })
        .collect()
}

// =============================================================================
// Message conversion
// =============================================================================

/// Options of [`convert_responses_messages`].
#[derive(Clone, Debug, Default)]
pub struct ConvertResponsesMessagesOptions<'a> {
    /// Send the leading system message as an input item. Default: true.
    pub include_system_prompt: Option<bool>,
    pub grammar_tool_input_properties: Option<&'a GrammarToolInputProperties>,
    /// Send later system messages in place; otherwise they fold into the leading prompt.
    pub supports_mid_convo_system_messages: bool,
    pub supports_additional_tools: bool,
    pub supports_tool_search: bool,
    pub tool_options: ConvertResponsesToolsOptions,
    /// Effort of the reply being generated. When set, thinking levels are sent as
    /// `configuration_update` items (one before each earlier reply of this model that recorded
    /// its level, one at the end) instead of a changing `reasoning.effort`.
    pub configuration_update_effort: Option<String>,
}

fn normalize_id_part(part: &str) -> String {
    let mut sanitized = String::with_capacity(part.len());
    for ch in part.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            sanitized.push(ch);
        } else {
            // JavaScript replaces each UTF-16 code unit.
            for _ in 0..ch.len_utf16() {
                sanitized.push('_');
            }
        }
    }
    sanitized.truncate(64);
    sanitized.trim_end_matches('_').to_owned()
}

fn build_foreign_responses_item_id(item_id: &str) -> String {
    let mut normalized = format!("fc_{}", short_hash(item_id));
    normalized.truncate(64);
    normalized
}

fn image_item(mime_type: &str, data: &str) -> Value {
    json!({"type": "input_image", "detail": "auto", "image_url": format!("data:{mime_type};base64,{data}")})
}

fn convert_tool_result_output(model: &Model, content: &[UserBlock]) -> Value {
    let text_result = content
        .iter()
        .filter_map(|block| match block {
            UserBlock::Text(text) => Some(text.text.as_str()),
            UserBlock::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let images: Vec<_> = content
        .iter()
        .filter_map(|block| match block {
            UserBlock::Image(image) => Some(image),
            UserBlock::Text(_) => None,
        })
        .collect();
    let has_text = !text_result.trim().is_empty();
    if images.is_empty() || !model.supports_images() {
        return Value::String(if has_text {
            text_result
        } else if images.is_empty() {
            "(no tool output)".into()
        } else {
            "(see attached image)".into()
        });
    }
    let mut output = Vec::with_capacity(images.len() + 1);
    if has_text {
        output.push(json!({"type": "input_text", "text": text_result}));
    }
    output.extend(images.iter().map(|image| image_item(&image.mime_type, &image.data)));
    Value::Array(output)
}

fn configuration_update(effort: &str) -> Value {
    json!({"type": "configuration_update", "reasoning": {"effort": effort}})
}

/// Convert a transcript to Responses input items.
///
/// Tool call ids are `call_id|item_id`. Calls from providers outside
/// `allowed_tool_call_providers` or from another API are normalized to OpenAI's id rules
/// (foreign item ids are hashed into `fc_<hash>`).
pub fn convert_responses_messages(
    model: &Model,
    context: &TranscriptContext,
    allowed_tool_call_providers: &[&str],
    options: &ConvertResponsesMessagesOptions<'_>,
) -> Result<Vec<Value>> {
    let normalized_context = resolve_transcript(context, options.supports_mid_convo_system_messages);
    let target_allowed = allowed_tool_call_providers.contains(&model.provider.as_str());
    let normalize_tool_call_id = |id: &str, _target: &Model, source: &AssistantMessage| -> String {
        if !target_allowed {
            return normalize_id_part(id);
        }
        let mut parts = id.split('|');
        let (Some(call_id), Some(item_id)) = (parts.next(), parts.next()) else {
            return normalize_id_part(id);
        };
        let normalized_call_id = normalize_id_part(call_id);
        let is_foreign = source.provider != model.provider || source.api != model.api;
        let mut normalized_item_id = if is_foreign {
            build_foreign_responses_item_id(item_id)
        } else {
            normalize_id_part(item_id)
        };
        // The Responses API requires function call item ids to start with "fc".
        if !normalized_item_id.starts_with("fc_") {
            normalized_item_id = normalize_id_part(&format!("fc_{normalized_item_id}"));
        }
        format!("{normalized_call_id}|{normalized_item_id}")
    };
    let transformed = transform_messages(normalized_context.messages(), model, Some(&normalize_tool_call_id));
    let transcript_tools = resolve_transcript_tools(
        normalized_context.messages(),
        options.supports_additional_tools || options.supports_tool_search,
    );
    let empty_grammar = GrammarToolInputProperties::new();
    let grammar = options.grammar_tool_input_properties.unwrap_or(&empty_grammar);
    let include_initial_system_message = options.include_system_prompt.unwrap_or(true);
    let compat = responses_compat(model);
    let instruction_role = if model.reasoning && compat.supports_developer_role != Some(false) {
        "developer"
    } else {
        "system"
    };

    let mut items: Vec<Value> = Vec::new();
    let append_system_tool_additions = |items: &mut Vec<Value>, message: &SystemMessage, seed: &str| -> Result<()> {
        let tools: &[Tool] = if transcript_tools.anchors_additions {
            message.tools_added.as_deref().unwrap_or_default()
        } else {
            &[]
        };
        if tools.is_empty() {
            return Ok(());
        }
        if options.supports_additional_tools {
            items.push(json!({
                "type": "additional_tools",
                "role": "developer",
                "tools": convert_responses_tools(tools, &options.tool_options)?,
            }));
            return Ok(());
        }
        if !options.supports_tool_search {
            return Ok(());
        }
        let names: Vec<&str> = tools.iter().map(|tool| tool.name.as_str()).collect();
        let call_id = format!("pi_tool_load_{}", short_hash(&format!("{seed}:{}", names.join(","))));
        items.push(json!({
            "type": "tool_search_call",
            "call_id": call_id,
            "execution": "client",
            "status": "completed",
            "arguments": {"query": names.join(" "), "limit": names.len()},
        }));
        let search_options = ConvertResponsesToolsOptions {
            tool_search_result: true,
            ..options.tool_options
        };
        items.push(json!({
            "type": "tool_search_output",
            "call_id": call_id,
            "execution": "client",
            "status": "completed",
            "tools": convert_responses_tools(tools, &search_options)?,
        }));
        Ok(())
    };

    let mut msg_index = 0usize;
    for (source_index, message) in transformed.iter().enumerate() {
        let is_leading_system_message = source_index == 0 && matches!(message, Message::System(_));
        match message {
            Message::System(system) => {
                if !is_leading_system_message {
                    append_system_tool_additions(&mut items, system, &format!("system:{msg_index}"))?;
                }
                if !is_leading_system_message || include_initial_system_message {
                    let text = if is_leading_system_message {
                        get_system_message_text(system)
                    } else {
                        render_system_message_update(system)
                    };
                    if !text.is_empty() {
                        items.push(json!({"role": instruction_role, "content": text}));
                    }
                }
            }
            Message::User(user) => match &user.content {
                UserContent::Text(text) => {
                    items.push(json!({"role": "user", "content": [{"type": "input_text", "text": text}]}));
                }
                UserContent::Blocks(blocks) => {
                    if blocks.is_empty() {
                        continue;
                    }
                    let content: Vec<Value> = blocks
                        .iter()
                        .map(|block| match block {
                            UserBlock::Text(text) => json!({"type": "input_text", "text": text.text}),
                            UserBlock::Image(image) => image_item(&image.mime_type, &image.data),
                        })
                        .collect();
                    items.push(json!({"role": "user", "content": content}));
                }
            },
            Message::Assistant(assistant) => {
                let same_provider_and_api = assistant.provider == model.provider && assistant.api == model.api;
                let same_model = same_provider_and_api && assistant.model == model.id;
                let different_model = same_provider_and_api && assistant.model != model.id;
                let mut output: Vec<Value> = Vec::new();
                let mut text_block_index = 0usize;
                for block in &assistant.content {
                    match block {
                        AssistantBlock::Thinking(thinking) => {
                            if let Some(signature) = thinking.thinking_signature.as_deref().filter(|s| !s.is_empty()) {
                                output.push(serde_json::from_str(signature)?);
                            }
                        }
                        AssistantBlock::Text(text) => {
                            let parsed = parse_text_signature(text.text_signature.as_deref());
                            let fallback_id = if text_block_index == 0 {
                                format!("msg_pi_{msg_index}")
                            } else {
                                format!("msg_pi_{msg_index}_{text_block_index}")
                            };
                            text_block_index += 1;
                            // OpenAI requires message ids of at most 64 characters.
                            let message_id = match parsed.as_ref().map(|(id, _)| id.as_str()) {
                                None | Some("") => fallback_id,
                                Some(id) if id.encode_utf16().count() > 64 => format!("msg_{}", short_hash(id)),
                                Some(id) => id.to_owned(),
                            };
                            let mut item = json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": text.text, "annotations": []}],
                                "status": "completed",
                                "id": message_id,
                            });
                            if let Some(phase) = parsed.and_then(|(_, phase)| phase) {
                                item["phase"] = Value::String(phase);
                            }
                            output.push(item);
                        }
                        AssistantBlock::ToolCall(call) => {
                            let mut parts = call.id.split('|');
                            let call_id = parts.next().unwrap_or_default();
                            let custom_input_property = grammar.get(&call.name);
                            // Different-model replies omit item ids: OpenAI validates which item ids
                            // were paired with reasoning items. Ids that do not match the replayed
                            // item type (function_call: fc_*, custom_tool_call: ctc_*) are dropped too.
                            let prefix = if custom_input_property.is_some() { "ctc_" } else { "fc_" };
                            let item_id = parts
                                .next()
                                .filter(|item_id| !different_model && item_id.starts_with(prefix));
                            let mut item = Map::new();
                            if let Some(property) = custom_input_property {
                                item.insert("type".into(), "custom_tool_call".into());
                                if let Some(item_id) = item_id {
                                    item.insert("id".into(), item_id.into());
                                }
                                item.insert("call_id".into(), call_id.into());
                                item.insert("name".into(), call.name.clone().into());
                                item.insert(
                                    "input".into(),
                                    get_grammar_tool_input(&call.name, &call.arguments, property)?.into(),
                                );
                            } else {
                                item.insert("type".into(), "function_call".into());
                                if let Some(item_id) = item_id {
                                    item.insert("id".into(), item_id.into());
                                }
                                item.insert("call_id".into(), call_id.into());
                                item.insert("name".into(), call.name.clone().into());
                                item.insert("arguments".into(), serde_json::to_string(&call.arguments)?.into());
                            }
                            if same_model && let Some(namespace) = &call.namespace {
                                item.insert("namespace".into(), namespace.clone().into());
                            }
                            output.push(Value::Object(item));
                        }
                    }
                }
                if output.is_empty() {
                    continue;
                }
                if options.configuration_update_effort.is_some()
                    && same_model
                    && let Some(level) = &assistant.provider_thinking_level
                {
                    items.push(configuration_update(level));
                }
                items.extend(output);
            }
            Message::ToolResult(result) => {
                let call_id = result.tool_call_id.split('|').next().unwrap_or_default();
                let output = convert_tool_result_output(model, &result.content);
                let kind = if grammar.contains_key(&result.tool_name) {
                    "custom_tool_call_output"
                } else {
                    "function_call_output"
                };
                items.push(json!({"type": kind, "call_id": call_id, "output": output}));
            }
        }
        if !is_leading_system_message {
            msg_index += 1;
        }
    }
    if let Some(effort) = &options.configuration_update_effort {
        items.push(configuration_update(effort));
    }
    Ok(items)
}

// =============================================================================
// Stream processing
// =============================================================================

/// Picks the service tier used for pricing from the response's and the request's tier.
pub type ResolveServiceTier = fn(response: Option<&str>, request: Option<&str>) -> Option<String>;
/// Applies service-tier pricing to the computed usage cost.
pub type ApplyServiceTierPricing = fn(model: &Model, usage: &mut Usage, service_tier: Option<&str>);

/// Options of [`process_responses_stream`].
#[derive(Clone, Debug, Default)]
pub struct ResponsesStreamOptions {
    /// Requested service tier.
    pub service_tier: Option<String>,
    pub grammar_tool_input_properties: GrammarToolInputProperties,
    pub resolve_service_tier: Option<ResolveServiceTier>,
    pub apply_service_tier_pricing: Option<ApplyServiceTierPricing>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotKind {
    Thinking,
    Text,
    ToolCall,
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    kind: SlotKind,
    content_index: usize,
}

/// Scratch state of a tool call whose `output_item.done` has not arrived.
#[derive(Debug)]
enum OpenToolCall {
    Function {
        partial_json: String,
    },
    Custom {
        property: String,
        input: String,
        buffer: GrammarToolInputJsonBuffer,
    },
}

/// Normalizes Responses stream events into a [`MessageBuilder`].
#[derive(Debug)]
struct ResponsesStreamState {
    options: ResponsesStreamOptions,
    /// Keyed by the event's `output_index` as JSON text; `None` when it is missing (some
    /// servers omit it, which then shares one slot as upstream does).
    slots: HashMap<Option<String>, Slot>,
    open_tool_calls: HashMap<usize, OpenToolCall>,
    reasoning_blocks_by_id: HashMap<String, usize>,
    saw_terminal_response_event: bool,
}

fn output_index_key(event: &Value) -> Option<String> {
    event.get("output_index").map(Value::to_string)
}

impl ResponsesStreamState {
    fn new(options: ResponsesStreamOptions) -> Self {
        Self {
            options,
            slots: HashMap::new(),
            open_tool_calls: HashMap::new(),
            reasoning_blocks_by_id: HashMap::new(),
            saw_terminal_response_event: false,
        }
    }

    fn slot(&self, event: &Value, kind: SlotKind) -> Option<Slot> {
        self.slots
            .get(&output_index_key(event))
            .copied()
            .filter(|slot| slot.kind == kind)
    }

    fn apply_message_phase_stop_reason(builder: &mut MessageBuilder, item: &Value) {
        if str_field(item, "type") == Some("message") && str_field(item, "phase") == Some("final_answer") {
            builder.output.stop_reason = StopReason::Stop;
        }
    }

    fn create_slot(&mut self, builder: &mut MessageBuilder, key: Option<String>, item: &Value) -> Option<Slot> {
        let slot = match str_field(item, "type") {
            Some("reasoning") => Slot {
                kind: SlotKind::Thinking,
                content_index: builder.start_thinking(String::new(), false, None),
            },
            Some("message") => {
                Self::apply_message_phase_stop_reason(builder, item);
                Slot {
                    kind: SlotKind::Text,
                    content_index: builder.start_text(),
                }
            }
            Some(kind @ ("function_call" | "custom_tool_call")) => {
                let id = format!("{}|{}", js_template(item.get("call_id")), js_template(item.get("id")));
                let name = str_field(item, "name").unwrap_or_default().to_owned();
                let (arguments, open, initial_json) = if kind == "function_call" {
                    let partial_json = truthy_str(item, "arguments").unwrap_or_default().to_owned();
                    let open = OpenToolCall::Function {
                        partial_json: partial_json.clone(),
                    };
                    (JsonObject::new(), open, partial_json)
                } else {
                    let property = self
                        .options
                        .grammar_tool_input_properties
                        .get(&name)
                        .cloned()
                        .unwrap_or_else(|| "input".into());
                    let input = truthy_str(item, "input").unwrap_or_default().to_owned();
                    let mut arguments = JsonObject::new();
                    arguments.insert(property.clone(), Value::String(input.clone()));
                    let open = OpenToolCall::Custom {
                        property,
                        input,
                        buffer: GrammarToolInputJsonBuffer::default(),
                    };
                    (arguments, open, String::new())
                };
                let content_index = builder.start_tool_call(id, name, arguments);
                if let Some(call) = builder.tool_call_mut(content_index) {
                    call.namespace = str_field(item, "namespace").map(str::to_owned);
                }
                // Arguments sent with `output_item.added` go through the builder too, so the
                // streamed JSON it parses at the end matches upstream's `partialJson`.
                builder.tool_call_delta(content_index, &initial_json);
                self.open_tool_calls.insert(content_index, open);
                Slot {
                    kind: SlotKind::ToolCall,
                    content_index,
                }
            }
            _ => return None,
        };
        self.slots.insert(key, slot);
        Some(slot)
    }

    fn push_custom_input(
        &mut self,
        builder: &mut MessageBuilder,
        index: usize,
        next_input: Option<&str>,
        close: bool,
    ) -> Result<()> {
        let Some(OpenToolCall::Custom {
            property,
            input,
            buffer,
        }) = self.open_tool_calls.get_mut(&index)
        else {
            return Ok(());
        };
        let next_input = next_input.map_or_else(|| input.clone(), str::to_owned);
        let delta = append_grammar_tool_input_json_delta(buffer, property, &next_input, close)?;
        let mut arguments = JsonObject::new();
        arguments.insert(property.clone(), Value::String(next_input.clone()));
        *input = next_input;
        if let Some(call) = builder.tool_call_mut(index) {
            call.arguments = arguments;
        }
        if let Some(delta) = delta {
            builder.tool_call_delta(index, &delta);
        }
        Ok(())
    }

    fn set_thinking_text(builder: &mut MessageBuilder, index: usize, text: String) {
        if let Some(AssistantBlock::Thinking(thinking)) = builder.output.content.get_mut(index) {
            thinking.thinking = text;
        }
    }

    fn output_item_done(&mut self, builder: &mut MessageBuilder, event: &Value) -> Result<()> {
        let Some(item) = event.get("item") else {
            return Ok(());
        };
        Self::apply_message_phase_stop_reason(builder, item);
        let key = output_index_key(event);
        let slot = match self.slots.get(&key).copied() {
            Some(slot) => Some(slot),
            None => self.create_slot(builder, key.clone(), item),
        };
        let Some(slot) = slot else {
            return Ok(());
        };
        let index = slot.content_index;
        match (str_field(item, "type"), slot.kind) {
            (Some("reasoning"), SlotKind::Thinking) => {
                let join = |field: &str| {
                    item.get(field)
                        .and_then(Value::as_array)
                        .map(|parts| {
                            parts
                                .iter()
                                .map(|part| str_field(part, "text").unwrap_or_default())
                                .collect::<Vec<_>>()
                                .join("\n\n")
                        })
                        .unwrap_or_default()
                };
                let summary = join("summary");
                let content = join("content");
                if !summary.is_empty() {
                    Self::set_thinking_text(builder, index, summary);
                } else if !content.is_empty() {
                    Self::set_thinking_text(builder, index, content);
                }
                builder.set_thinking_signature(index, Some(serde_json::to_string(item)?));
                if let Some(id) = str_field(item, "id") {
                    self.reasoning_blocks_by_id.insert(id.to_owned(), index);
                }
                builder.end_thinking(index);
                self.slots.remove(&key);
            }
            (Some("message"), SlotKind::Text) => {
                let text = item
                    .get("content")
                    .and_then(Value::as_array)
                    .map(|parts| {
                        parts
                            .iter()
                            .map(|part| {
                                let field = if str_field(part, "type") == Some("output_text") {
                                    "text"
                                } else {
                                    "refusal"
                                };
                                str_field(part, field).unwrap_or_default()
                            })
                            .collect::<String>()
                    })
                    .unwrap_or_default();
                if let Some(AssistantBlock::Text(block)) = builder.output.content.get_mut(index) {
                    block.text = text;
                }
                let signature = encode_text_signature_v1(str_field(item, "id"), str_field(item, "phase"));
                builder.set_text_signature(index, Some(signature));
                builder.end_text(index);
                self.slots.remove(&key);
            }
            (Some("function_call"), SlotKind::ToolCall) => {
                let Some(OpenToolCall::Function { partial_json }) = self.open_tool_calls.get(&index) else {
                    return Ok(());
                };
                let json_text = truthy_str(item, "arguments").unwrap_or(partial_json).to_owned();
                let (arguments, arguments_error) = parse_tool_call_arguments(&json_text);
                let namespace = item.get("namespace").map(|value| value.as_str().map(str::to_owned));
                let finalize = |builder: &mut MessageBuilder| {
                    if let Some(call) = builder.tool_call_mut(index) {
                        call.arguments.clone_from(&arguments);
                        call.arguments_error.clone_from(&arguments_error);
                        if let Some(namespace) = &namespace {
                            call.namespace.clone_from(namespace);
                        }
                    }
                };
                finalize(builder);
                builder.end_tool_call(index);
                // The builder parses the JSON it streamed; the final arguments are those of the
                // completed item, as upstream (they differ only when the server's final
                // arguments do not extend the streamed deltas).
                finalize(builder);
                self.open_tool_calls.remove(&index);
                self.slots.remove(&key);
            }
            (Some("custom_tool_call"), SlotKind::ToolCall) => {
                if !matches!(self.open_tool_calls.get(&index), Some(OpenToolCall::Custom { .. })) {
                    return Ok(());
                }
                let input = item.get("input").filter(|value| !value.is_null());
                self.push_custom_input(
                    builder,
                    index,
                    input.map(|value| value.as_str().unwrap_or_default()),
                    true,
                )?;
                if let Some(namespace) = item.get("namespace")
                    && let Some(call) = builder.tool_call_mut(index)
                {
                    call.namespace = namespace.as_str().map(str::to_owned);
                }
                builder.end_tool_call(index);
                self.open_tool_calls.remove(&index);
                self.slots.remove(&key);
            }
            _ => {}
        }
        Ok(())
    }

    /// Azure can omit `encrypted_content` from `output_item.done` and send it only in the
    /// terminal response; copy it into the stored reasoning item for stateless replay.
    fn backfill_reasoning_signatures(&self, builder: &mut MessageBuilder, response_output: &[Value]) -> Result<()> {
        for item in response_output {
            if str_field(item, "type") != Some("reasoning") || !truthy(item.get("encrypted_content")) {
                continue;
            }
            let Some(&index) = str_field(item, "id").and_then(|id| self.reasoning_blocks_by_id.get(id)) else {
                continue;
            };
            let Some(AssistantBlock::Thinking(block)) = builder.output.content.get_mut(index) else {
                continue;
            };
            let Some(signature) = block.thinking_signature.as_deref().filter(|s| !s.is_empty()) else {
                continue;
            };
            let mut stored: Value = serde_json::from_str(signature)?;
            if truthy(stored.get("encrypted_content")) {
                continue;
            }
            if let Some(object) = stored.as_object_mut() {
                object.insert("encrypted_content".into(), item["encrypted_content"].clone());
            }
            block.thinking_signature = Some(serde_json::to_string(&stored)?);
        }
        Ok(())
    }

    fn finalize_response(&mut self, builder: &mut MessageBuilder, model: &Model, response: &Value) -> Result<()> {
        self.saw_terminal_response_event = true;
        let output_items = response
            .get("output")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        self.backfill_reasoning_signatures(builder, &output_items)?;
        if let Some(id) = truthy_str(response, "id") {
            builder.output.response_id = Some(id.to_owned());
        }
        if let Some(usage) = response.get("usage").filter(|usage| truthy(Some(usage))) {
            let input_details = usage.get("input_tokens_details");
            let cached = count_field(input_details.and_then(|details| details.get("cached_tokens")));
            let cache_write = count_field(input_details.and_then(|details| details.get("cache_write_tokens")));
            // OpenAI includes cached and cache-write tokens in input_tokens.
            builder.output.usage = Usage {
                input: count_field(usage.get("input_tokens")).saturating_sub(cached + cache_write),
                output: count_field(usage.get("output_tokens")),
                cache_read: cached,
                cache_write,
                reasoning: Some(count_field(
                    usage
                        .get("output_tokens_details")
                        .and_then(|details| details.get("reasoning_tokens")),
                )),
                total_tokens: count_field(usage.get("total_tokens")),
                ..Usage::default()
            };
        }
        calculate_cost(&model.cost, &mut builder.output.usage);
        if let Some(apply) = self.options.apply_service_tier_pricing {
            let response_tier = str_field(response, "service_tier");
            let request_tier = self.options.service_tier.as_deref();
            let service_tier = match self.options.resolve_service_tier {
                Some(resolve) => resolve(response_tier, request_tier),
                None => response_tier.or(request_tier).map(str::to_owned),
            };
            apply(model, &mut builder.output.usage, service_tier.as_deref());
        }
        // Incomplete responses keep the provider's reason so max-output truncation and
        // content filtering stay distinct.
        let status = response.get("status").filter(|status| truthy(Some(status)));
        let incomplete_reason = response
            .get("incomplete_details")
            .and_then(|details| str_field(details, "reason"));
        builder.output.raw_stop_reason = match (status, incomplete_reason) {
            (_, Some(reason)) if !reason.is_empty() => Some(format!("{}.{reason}", js_template(status))),
            (Some(status), _) => Some(js_template(Some(status))),
            (None, _) => None,
        };
        let (stop_reason, error_message) = map_stop_reason(status, incomplete_reason)?;
        builder.output.stop_reason = stop_reason;
        builder.output.error_message = error_message;
        if builder.output.stop_reason == StopReason::Stop && builder.output.tool_calls().next().is_some() {
            builder.output.stop_reason = StopReason::ToolUse;
        }
        Ok(())
    }

    fn handle_event(&mut self, builder: &mut MessageBuilder, model: &Model, event: &Value) -> Result<()> {
        let Some(kind) = str_field(event, "type") else {
            return Ok(());
        };
        match kind {
            "response.created" => {
                if let Some(response) = event.get("response") {
                    builder.output.response_id = str_field(response, "id").map(str::to_owned);
                }
            }
            "response.output_item.added" => {
                if let Some(item) = event.get("item") {
                    self.create_slot(builder, output_index_key(event), item);
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                if let Some(slot) = self.slot(event, SlotKind::Thinking) {
                    builder.thinking_delta(slot.content_index, str_field(event, "delta").unwrap_or_default());
                }
            }
            "response.reasoning_summary_part.done" => {
                if let Some(slot) = self.slot(event, SlotKind::Thinking) {
                    builder.thinking_delta(slot.content_index, "\n\n");
                }
            }
            "response.output_text.delta" | "response.refusal.delta" => {
                if let Some(slot) = self.slot(event, SlotKind::Text) {
                    builder.text_delta(slot.content_index, str_field(event, "delta").unwrap_or_default());
                }
            }
            "response.function_call_arguments.delta" => {
                let Some(slot) = self.slot(event, SlotKind::ToolCall) else {
                    return Ok(());
                };
                if let Some(OpenToolCall::Function { partial_json }) = self.open_tool_calls.get_mut(&slot.content_index)
                {
                    let delta = str_field(event, "delta").unwrap_or_default();
                    partial_json.push_str(delta);
                    builder.tool_call_delta(slot.content_index, delta);
                }
            }
            "response.function_call_arguments.done" => {
                let Some(slot) = self.slot(event, SlotKind::ToolCall) else {
                    return Ok(());
                };
                let Some(arguments) = str_field(event, "arguments") else {
                    return Ok(());
                };
                if let Some(OpenToolCall::Function { partial_json }) = self.open_tool_calls.get_mut(&slot.content_index)
                {
                    let previous = std::mem::replace(partial_json, arguments.to_owned());
                    if let Some(delta) = arguments.strip_prefix(previous.as_str()) {
                        builder.tool_call_delta(slot.content_index, delta);
                    }
                }
            }
            "response.custom_tool_call_input.delta" => {
                let Some(slot) = self.slot(event, SlotKind::ToolCall) else {
                    return Ok(());
                };
                if let Some(OpenToolCall::Custom { input, .. }) = self.open_tool_calls.get(&slot.content_index) {
                    let next = format!("{input}{}", str_field(event, "delta").unwrap_or_default());
                    self.push_custom_input(builder, slot.content_index, Some(&next), false)?;
                }
            }
            "response.custom_tool_call_input.done" => {
                let Some(slot) = self.slot(event, SlotKind::ToolCall) else {
                    return Ok(());
                };
                let input = str_field(event, "input").unwrap_or_default().to_owned();
                self.push_custom_input(builder, slot.content_index, Some(&input), true)?;
            }
            "response.output_item.done" => self.output_item_done(builder, event)?,
            "response.completed" | "response.incomplete" => {
                let response = event.get("response").cloned().unwrap_or(Value::Null);
                self.finalize_response(builder, model, &response)?;
            }
            "error" => {
                return Err(Error::msg(format!(
                    "Error Code {}: {}",
                    js_template(event.get("code")),
                    js_template(event.get("message"))
                )));
            }
            "response.failed" => {
                self.saw_terminal_response_event = true;
                let response = event.get("response");
                builder.output.raw_stop_reason = response.and_then(|r| str_field(r, "status")).map(str::to_owned);
                let error = response.and_then(|r| r.get("error")).filter(|e| truthy(Some(e)));
                let reason = response
                    .and_then(|r| r.get("incomplete_details"))
                    .and_then(|details| details.get("reason"))
                    .filter(|reason| truthy(Some(reason)));
                let message = match (error, reason) {
                    (Some(error), _) => {
                        let field = |name: &str, fallback: &str| {
                            if truthy(error.get(name)) {
                                js_template(error.get(name))
                            } else {
                                fallback.to_owned()
                            }
                        };
                        format!("{}: {}", field("code", "unknown"), field("message", "no message"))
                    }
                    (None, Some(reason)) => format!("incomplete: {}", js_template(Some(reason))),
                    (None, None) => "Unknown error (no error details in response)".into(),
                };
                return Err(Error::msg(message));
            }
            _ => {}
        }
        Ok(())
    }

    fn finish(&self, builder: &MessageBuilder) -> Result<()> {
        if !self.saw_terminal_response_event {
            return Err(Error::msg(
                "OpenAI Responses stream ended before a terminal response event",
            ));
        }
        // The agent runs every tool call of the final message. Calls whose output_item.done
        // never arrived may have cut-off or mixed-up arguments (e.g. a server that omits
        // output_index), so they fail the response.
        if builder.output.stop_reason == StopReason::ToolUse {
            for (index, block) in builder.output.content.iter().enumerate() {
                if let AssistantBlock::ToolCall(call) = block
                    && self.open_tool_calls.contains_key(&index)
                {
                    return Err(Error::msg(format!(
                        "OpenAI Responses stream completed with an unfinished tool call: {} ({})",
                        call.name, call.id
                    )));
                }
            }
        }
        Ok(())
    }
}

fn map_stop_reason(status: Option<&Value>, incomplete_reason: Option<&str>) -> Result<(StopReason, Option<String>)> {
    let Some(status) = status else {
        return Ok((StopReason::Stop, None));
    };
    Ok(match status.as_str() {
        Some("completed" | "in_progress" | "queued") => (StopReason::Stop, None),
        Some("incomplete") => match incomplete_reason {
            Some("max_output_tokens") => (StopReason::Length, None),
            Some(reason) if !reason.is_empty() => (StopReason::Error, Some(format!("Response incomplete: {reason}"))),
            _ => (
                StopReason::Error,
                Some("Response incomplete without a provider reason".into()),
            ),
        },
        Some("failed" | "cancelled") => (StopReason::Error, None),
        _ => {
            return Err(Error::msg(format!(
                "Unhandled stop reason: {}",
                js_template(Some(status))
            )));
        }
    })
}

/// The error a JSON SSE payload with an `error` field raises, as the OpenAI SDK words it.
fn stream_payload_error(error: &Value) -> Error {
    let message = match error.get("message") {
        Some(Value::String(message)) if !message.is_empty() => message.clone(),
        Some(message) if truthy(Some(message)) => message.to_string(),
        _ => error.to_string(),
    };
    Error::msg(message)
}

/// Normalize a Responses SSE stream into `builder`.
///
/// Mirrors the OpenAI SDK's stream layer first: `[DONE]` ends the stream, payloads that are
/// not JSON fail it, and a payload with an `error` field fails it with that error's message.
/// Every other payload goes to `options.on_provider_stream_event`, then is normalized; unknown
/// event and item types are skipped.
///
/// Errors when the stream ends without a terminal response event, on `error` and
/// `response.failed` events, and when a `toolUse` response still has an unfinished tool call.
pub async fn process_responses_stream<S>(
    events: S,
    builder: &mut MessageBuilder,
    model: &Model,
    options: &StreamOptions,
    stream_options: ResponsesStreamOptions,
) -> Result<()>
where
    S: Stream<Item = Result<SseEvent>>,
{
    let mut state = ResponsesStreamState::new(stream_options);
    let mut events = std::pin::pin!(events);
    while let Some(event) = events.next().await {
        if builder.is_closed() {
            return Err(Error::Aborted);
        }
        let event = event?;
        if event.data.starts_with("[DONE]") {
            break;
        }
        let payload: Value = serde_json::from_str(&event.data)
            .map_err(|error| Error::msg(format!("Could not parse message into JSON: {error}: {}", event.data)))?;
        if let Some(error) = payload.get("error").filter(|error| truthy(Some(error))) {
            return Err(stream_payload_error(error));
        }
        observe_provider_event(options, model, &payload).await?;
        state.handle_event(builder, model, &payload)?;
    }
    if builder.is_closed() {
        return Err(Error::Aborted);
    }
    state.finish(builder)
}

/// Normalize already-parsed Responses events (no SSE layer, no callbacks). For tests and
/// transports that decode events themselves.
pub fn process_responses_events<'a>(
    events: impl IntoIterator<Item = &'a Value>,
    builder: &mut MessageBuilder,
    model: &Model,
    stream_options: ResponsesStreamOptions,
) -> Result<()> {
    let mut state = ResponsesStreamState::new(stream_options);
    for event in events {
        state.handle_event(builder, model, event)?;
    }
    state.finish(builder)
}

/// Incremental form of [`process_responses_events`], for transports that read, map and
/// observe events themselves (the OpenAI Codex SSE and WebSocket transports, which hand
/// over each event as upstream's `mapCodexEvents` yields it).
#[derive(Debug)]
pub struct ResponsesEventProcessor(ResponsesStreamState);

impl ResponsesEventProcessor {
    pub fn new(stream_options: ResponsesStreamOptions) -> Self {
        Self(ResponsesStreamState::new(stream_options))
    }

    /// Normalize one parsed event into `builder`.
    pub fn handle_event(&mut self, builder: &mut MessageBuilder, model: &Model, event: &Value) -> Result<()> {
        self.0.handle_event(builder, model, event)
    }

    /// The end-of-stream checks: a terminal response event arrived and no `toolUse` response
    /// has an unfinished tool call.
    pub fn finish(&self, builder: &MessageBuilder) -> Result<()> {
        self.0.finish(builder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{AssistantMessageEvent, AssistantMessageEventStream};
    use crate::transcript::normalize_context;
    use crate::types::{ConstrainedSampling, StrictMode};
    use crate::types::{
        Context, GrammarVariants, ImageContent, TextContent, ThinkingContent, ToolCall, ToolResultMessage,
    };
    use pretty_assertions::assert_eq;

    fn model() -> Model {
        serde_json::from_value(json!({
            "id": "gpt-5.4", "name": "GPT-5.4", "api": "openai-responses", "provider": "openai",
            "baseUrl": "https://api.openai.com/v1", "reasoning": true, "input": ["text", "image"],
            "cost": {"input": 1, "output": 2, "cacheRead": 0.5, "cacheWrite": 0},
            "contextWindow": 400_000, "maxTokens": 128_000
        }))
        .unwrap()
    }

    fn assistant(content: Vec<AssistantBlock>, provider: &str, model_id: &str) -> AssistantMessage {
        AssistantMessage {
            content,
            stop_reason: StopReason::ToolUse,
            ..AssistantMessage::empty("openai-responses", provider, model_id)
        }
    }

    fn convert(model: &Model, messages: Vec<Message>, options: &ConvertResponsesMessagesOptions<'_>) -> Vec<Value> {
        let context = normalize_context(&Context::new(messages));
        convert_responses_messages(model, &context, &["openai", "openai-codex", "opencode"], options).unwrap()
    }

    fn run(events: &[Value], options: ResponsesStreamOptions) -> (AssistantMessage, Result<()>, Vec<String>) {
        run_with(&model(), events, options)
    }

    fn run_with(
        model: &Model,
        events: &[Value],
        options: ResponsesStreamOptions,
    ) -> (AssistantMessage, Result<()>, Vec<String>) {
        let (sender, mut stream) = AssistantMessageEventStream::channel();
        let mut builder = MessageBuilder::new(sender, crate::api::common::new_output(model));
        builder.start();
        let result = process_responses_events(events, &mut builder, model, options);
        let output = builder.output.clone();
        drop(builder);
        let mut kinds = Vec::new();
        while let Some(Some(event)) = futures::FutureExt::now_or_never(stream.next()) {
            kinds.push(event.kind().to_owned());
            if let AssistantMessageEvent::ToolCallEnd { tool_call, .. } = &event {
                kinds.push(format!("args:{}", Value::Object(tool_call.arguments.clone())));
            }
        }
        (output, result, kinds)
    }

    #[test]
    fn prompt_cache_key_is_clamped_by_code_points() {
        assert_eq!(
            clamp_openai_prompt_cache_key(Some(&"x".repeat(67))),
            Some("x".repeat(64))
        );
        assert_eq!(clamp_openai_prompt_cache_key(None), None);
        let emoji = "\u{1F648}".repeat(70);
        assert_eq!(clamp_openai_prompt_cache_key(Some(&emoji)).unwrap().chars().count(), 64);
    }

    #[test]
    fn empty_tool_results_use_placeholder() {
        let m = model();
        for text in ["", "\r\n", " \n\t"] {
            let call = ToolCall {
                id: "tool-1".into(),
                name: "bash".into(),
                ..ToolCall::default()
            };
            let messages = vec![
                Message::user("Run"),
                assistant(vec![AssistantBlock::ToolCall(call.clone())], "openai", "gpt-5.4").into(),
                ToolResultMessage::text(&call, text, false).into(),
            ];
            let input = convert(&m, messages, &ConvertResponsesMessagesOptions::default());
            let output = input
                .iter()
                .find(|item| item["type"] == "function_call_output")
                .unwrap();
            assert_eq!(output["output"], "(no tool output)");
        }
    }

    #[test]
    fn tool_result_images_stay_in_function_call_output() {
        let mut m = model();
        let call = ToolCall {
            id: "call_1|fc_1".into(),
            name: "shot".into(),
            ..ToolCall::default()
        };
        let mut result = ToolResultMessage::text(&call, "A red circle", false);
        result.content.push(UserBlock::Image(ImageContent {
            data: "AAA".into(),
            mime_type: "image/png".into(),
        }));
        let messages = vec![
            Message::user("go"),
            assistant(vec![AssistantBlock::ToolCall(call.clone())], "openai", "gpt-5.4").into(),
            result.into(),
        ];
        let input = convert(&m, messages.clone(), &ConvertResponsesMessagesOptions::default());
        assert_eq!(
            input.last().unwrap(),
            &json!({"type": "function_call_output", "call_id": "call_1", "output": [
                {"type": "input_text", "text": "A red circle"},
                {"type": "input_image", "detail": "auto", "image_url": "data:image/png;base64,AAA"}
            ]})
        );
        // Text-only models get the transform placeholder instead of the image.
        m.input = vec![crate::model::InputModality::Text];
        let input = convert(&m, messages, &ConvertResponsesMessagesOptions::default());
        assert_eq!(
            input.last().unwrap()["output"],
            "A red circle\n(tool image omitted: model does not support images)"
        );
    }

    #[test]
    fn foreign_tool_call_ids_are_hashed() {
        let raw = "call_4VnzVawQXPB9|I9b95oN1wD/cHXKTw3PpRkL6KkCtzTJhUxMouMWYwHeTo2j3htzfSk7YPx2vifiIM4g3A8XXyOj8q4Bt6SLUG7gqY1E3ELkrkVQNHglRfUmWj84lqxJY+Puieb3VKyX0FB+83TUzn91c==";
        let call = ToolCall {
            id: raw.into(),
            name: "edit".into(),
            ..ToolCall::default()
        };
        let mut foreign = assistant(
            vec![AssistantBlock::ToolCall(call.clone())],
            "github-copilot",
            "gpt-5.5",
        );
        foreign.api = "openai-responses".into();
        let messages = vec![
            Message::user("Use the tool."),
            foreign.into(),
            ToolResultMessage::text(&call, "ok", false).into(),
        ];
        let input = convert(&model(), messages, &ConvertResponsesMessagesOptions::default());
        let function_call = input.iter().find(|item| item["type"] == "function_call").unwrap();
        let expected = format!("fc_{}", short_hash(raw.split('|').nth(1).unwrap()));
        assert_eq!(function_call["id"], expected.as_str());
        assert_eq!(function_call["call_id"], "call_4VnzVawQXPB9");
        let output = input
            .iter()
            .find(|item| item["type"] == "function_call_output")
            .unwrap();
        assert_eq!(output["call_id"], "call_4VnzVawQXPB9");
    }

    #[test]
    fn cross_provider_replay_drops_thinking_and_signatures() {
        let foreign = AssistantMessage {
            content: vec![
                AssistantBlock::Thinking(ThinkingContent {
                    thinking: "hm".into(),
                    thinking_signature: Some("sig".into()),
                    redacted: None,
                }),
                AssistantBlock::Text(TextContent {
                    text: "first answer".into(),
                    text_signature: Some("sig".into()),
                }),
                AssistantBlock::Text(TextContent::new("visible answer")),
                AssistantBlock::ToolCall(ToolCall {
                    id: "toolu_01".into(),
                    name: "read".into(),
                    ..ToolCall::default()
                }),
            ],
            stop_reason: StopReason::ToolUse,
            ..AssistantMessage::empty("anthropic-messages", "anthropic", "claude-opus-4-8")
        };
        let messages = vec![Message::user("hello"), foreign.into()];
        let context = normalize_context(&Context::new(messages).with_system_prompt("You are concise."));
        let input = convert_responses_messages(
            &model(),
            &context,
            &["openai"],
            &ConvertResponsesMessagesOptions::default(),
        )
        .unwrap();
        assert_eq!(input[0], json!({"role": "developer", "content": "You are concise."}));
        let ids: Vec<&Value> = input
            .iter()
            .filter(|item| item["type"] == "message")
            .map(|item| &item["id"])
            .collect();
        assert_eq!(ids, [&json!("msg_pi_1"), &json!("msg_pi_1_1")]);
        assert!(!input.iter().any(|item| item["type"] == "reasoning"));
        let call = input.iter().find(|item| item["type"] == "function_call").unwrap();
        assert_eq!(
            call,
            &json!({"type": "function_call", "call_id": "toolu_01", "name": "read", "arguments": "{}"})
        );
        // Orphaned call gets a synthetic result.
        assert_eq!(
            input.last().unwrap(),
            &json!({"type": "function_call_output", "call_id": "toolu_01", "output": "No result provided"})
        );
    }

    #[test]
    fn text_signatures_replay_ids_and_phases() {
        let long = "x".repeat(80);
        let cases = [
            (
                Some(r#"{"v":1,"id":"msg_a","phase":"commentary"}"#.to_owned()),
                json!("msg_a"),
                Some("commentary"),
            ),
            (
                Some(r#"{"v":1,"id":"msg_b","phase":"other"}"#.to_owned()),
                json!("msg_b"),
                None,
            ),
            (Some("msg_legacy".to_owned()), json!("msg_legacy"), None),
            (Some(long.clone()), json!(format!("msg_{}", short_hash(&long))), None),
            (None, json!("msg_pi_1"), None),
        ];
        for (signature, id, phase) in cases {
            let message = AssistantMessage {
                content: vec![AssistantBlock::Text(TextContent {
                    text: "t".into(),
                    text_signature: signature,
                })],
                stop_reason: StopReason::Stop,
                ..AssistantMessage::empty("openai-responses", "openai", "gpt-5.4")
            };
            let input = convert(
                &model(),
                vec![Message::user("q"), message.into()],
                &ConvertResponsesMessagesOptions::default(),
            );
            assert_eq!(input[1]["id"], id);
            assert_eq!(input[1]["phase"].as_str(), phase);
        }
    }

    #[test]
    fn system_role_and_omission() {
        let mut m = model();
        let context = normalize_context(&Context::new(vec![Message::user("hi")]).with_system_prompt("sys"));
        let options = ConvertResponsesMessagesOptions::default();
        assert_eq!(
            convert_responses_messages(&m, &context, &[], &options).unwrap()[0]["role"],
            "developer"
        );
        m.compat = Some(ModelCompat::OpenAIResponses(OpenAIResponsesCompat {
            supports_developer_role: Some(false),
            ..OpenAIResponsesCompat::default()
        }));
        assert_eq!(
            convert_responses_messages(&m, &context, &[], &options).unwrap()[0]["role"],
            "system"
        );
        let omit = ConvertResponsesMessagesOptions {
            include_system_prompt: Some(false),
            ..ConvertResponsesMessagesOptions::default()
        };
        let input = convert_responses_messages(&m, &context, &[], &omit).unwrap();
        assert_eq!(
            input,
            [json!({"role": "user", "content": [{"type": "input_text", "text": "hi"}]})]
        );
    }

    fn tool(name: &str) -> Tool {
        Tool::new(
            name,
            name,
            json!({"type": "object", "properties": {"a": {"type": "string"}}}),
        )
    }

    #[test]
    fn tool_additions_as_additional_tools_and_tool_search() {
        let messages: Vec<Message> = vec![
            SystemMessage {
                content: "sys".into(),
                tools_added: Some(vec![tool("a")]),
                ..SystemMessage::default()
            }
            .into(),
            Message::user("hi"),
            SystemMessage {
                content: "more".into(),
                tools_added: Some(vec![tool("b")]),
                ..SystemMessage::default()
            }
            .into(),
        ];
        let context = normalize_context(&Context::new(messages));
        let additional = ConvertResponsesMessagesOptions {
            supports_mid_convo_system_messages: true,
            supports_additional_tools: true,
            ..ConvertResponsesMessagesOptions::default()
        };
        let input = convert_responses_messages(&model(), &context, &[], &additional).unwrap();
        assert_eq!(input[2]["type"], "additional_tools");
        assert_eq!(input[2]["role"], "developer");
        assert_eq!(input[2]["tools"][0]["name"], "b");
        assert_eq!(input[2]["tools"][0]["strict"], false);
        assert_eq!(input[3], json!({"role": "developer", "content": "more"}));

        let search = ConvertResponsesMessagesOptions {
            supports_mid_convo_system_messages: true,
            supports_tool_search: true,
            ..ConvertResponsesMessagesOptions::default()
        };
        let input = convert_responses_messages(&model(), &context, &[], &search).unwrap();
        let call_id = format!("pi_tool_load_{}", short_hash("system:1:b"));
        assert_eq!(
            input[2],
            json!({"type": "tool_search_call", "call_id": call_id, "execution": "client", "status": "completed",
                   "arguments": {"query": "b", "limit": 1}})
        );
        assert_eq!(input[3]["type"], "tool_search_output");
        assert_eq!(input[3]["call_id"], call_id.as_str());
        assert_eq!(input[3]["tools"][0]["defer_loading"], true);

        // Without mid-conversation system messages, the additions fold into the leading prompt.
        let folded = ConvertResponsesMessagesOptions {
            supports_additional_tools: true,
            ..ConvertResponsesMessagesOptions::default()
        };
        let input = convert_responses_messages(&model(), &context, &[], &folded).unwrap();
        assert_eq!(input[0], json!({"role": "developer", "content": "sys\n\nmore"}));
        assert_eq!(input.len(), 2);
    }

    #[test]
    fn convert_tools_strict_and_grammar() {
        let mut preferred = Tool::new(
            "preferred",
            "d",
            json!({"type": "object", "properties": {"path": {"type": "string"}, "n": {"type": "number"}}, "required": ["path"]}),
        );
        preferred.constrained_sampling = Some(ConstrainedSampling::JsonSchema {
            strict: StrictMode::Prefer,
        });
        let options = ConvertResponsesToolsOptions {
            supports_strict_mode: Some(true),
            ..ConvertResponsesToolsOptions::default()
        };
        let tools = convert_responses_tools(&[tool("plain"), preferred.clone()], &options).unwrap();
        assert_eq!(tools[0]["strict"], false);
        assert_eq!(tools[1]["strict"], true);
        assert_eq!(
            tools[1]["parameters"],
            json!({"type": "object", "properties": {"path": {"type": "string"},
                   "n": {"anyOf": [{"type": "number"}, {"type": "null"}]}},
                   "required": ["path", "n"], "additionalProperties": false})
        );
        // Without strict support the key is omitted and "prefer" falls back.
        let no_strict = ConvertResponsesToolsOptions {
            supports_strict_mode: Some(false),
            ..ConvertResponsesToolsOptions::default()
        };
        let tools = convert_responses_tools(std::slice::from_ref(&preferred), &no_strict).unwrap();
        assert!(tools[0].get("strict").is_none());
        // "require" fails when it cannot be honored.
        let mut required = preferred.clone();
        required.constrained_sampling = Some(ConstrainedSampling::JsonSchema {
            strict: StrictMode::Require,
        });
        let error = convert_responses_tools(&[required.clone()], &no_strict).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Tool \"preferred\" requires JSON-schema constrained sampling, but strict tools are unsupported."
        );
        required.parameters = json!({"type": "object", "properties": {"a": {"oneOf": []}}});
        let error = convert_responses_tools(&[required], &options).unwrap_err();
        assert!(error.to_string().ends_with("but oneOf schemas are unsupported."));

        let mut grammar = Tool::new(
            "query",
            "q",
            json!({"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}),
        );
        grammar.constrained_sampling = Some(ConstrainedSampling::Grammar {
            variants: GrammarVariants {
                openai_lark: Some("start: /.+/".into()),
                openai_regex: None,
            },
        });
        let with_grammar = ConvertResponsesToolsOptions {
            supports_openai_grammar_tools: Some(true),
            ..ConvertResponsesToolsOptions::default()
        };
        assert_eq!(
            convert_responses_tools(std::slice::from_ref(&grammar), &with_grammar).unwrap()[0],
            json!({"type": "custom", "name": "query", "description": "q",
                   "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}})
        );
        assert_eq!(
            create_grammar_tool_input_properties(std::slice::from_ref(&grammar), true)
                .unwrap()
                .get("query")
                .map(String::as_str),
            Some("sql")
        );
        // Unsupported grammar tools are plain function tools.
        assert_eq!(
            convert_responses_tools(std::slice::from_ref(&grammar), &options).unwrap()[0]["type"],
            "function"
        );
        grammar.parameters = json!({"type": "object", "properties": {"sql": {"type": "number"}}, "required": ["sql"]});
        assert_eq!(
            create_grammar_tool_input_properties(&[grammar], true)
                .unwrap_err()
                .to_string(),
            "Tool \"query\" cannot use grammar constrained sampling: grammar constrained sampling property sql must have type string."
        );
    }

    #[test]
    fn mid_convo_effort_configuration_updates() {
        let mut recorded = assistant(vec![AssistantBlock::Text(TextContent::new("a"))], "openai", "gpt-5.4");
        recorded.stop_reason = StopReason::Stop;
        recorded.provider_thinking_level = Some("high".into());
        let mut other_model = recorded.clone();
        other_model.model = "gpt-6-sol".into();
        let mut unstamped = recorded.clone();
        unstamped.provider_thinking_level = None;
        let options = ConvertResponsesMessagesOptions {
            configuration_update_effort: Some("low".into()),
            ..ConvertResponsesMessagesOptions::default()
        };
        let input = convert(
            &model(),
            vec![
                Message::user("one"),
                unstamped.into(),
                Message::user("two"),
                other_model.into(),
                Message::user("three"),
                recorded.into(),
                Message::user("four"),
            ],
            &options,
        );
        let kinds: Vec<&str> = input
            .iter()
            .map(|item| item["type"].as_str().or(item["role"].as_str()).unwrap())
            .collect();
        assert_eq!(
            kinds,
            [
                "user",
                "message",
                "user",
                "message",
                "user",
                "configuration_update",
                "message",
                "user",
                "configuration_update"
            ]
        );
        assert_eq!(input[5], configuration_update("high"));
        assert_eq!(input[8], configuration_update("low"));
    }

    #[test]
    fn namespaces_replay_only_to_the_same_model() {
        let message = AssistantMessage {
            content: vec![
                AssistantBlock::ToolCall(ToolCall {
                    id: "call_function|fc_test".into(),
                    name: "lookup".into(),
                    namespace: Some("dynamic_tools".into()),
                    ..ToolCall::default()
                }),
                AssistantBlock::ToolCall(ToolCall {
                    id: "call_custom|ctc_test".into(),
                    name: "query".into(),
                    arguments: [("input".to_owned(), json!("hello"))].into_iter().collect(),
                    namespace: Some("dynamic_tools".into()),
                    ..ToolCall::default()
                }),
            ],
            stop_reason: StopReason::ToolUse,
            ..AssistantMessage::empty("openai-responses", "openai", "gpt-5.4")
        };
        let grammar: GrammarToolInputProperties = [("query".to_owned(), "input".to_owned())].into_iter().collect();
        let options = ConvertResponsesMessagesOptions {
            grammar_tool_input_properties: Some(&grammar),
            ..ConvertResponsesMessagesOptions::default()
        };
        let input = convert(&model(), vec![message.clone().into()], &options);
        assert_eq!(
            input[0],
            json!({"type": "function_call", "id": "fc_test", "call_id": "call_function", "name": "lookup",
                   "arguments": "{}", "namespace": "dynamic_tools"})
        );
        assert_eq!(
            input[1],
            json!({"type": "custom_tool_call", "id": "ctc_test", "call_id": "call_custom", "name": "query",
                   "input": "hello", "namespace": "dynamic_tools"})
        );
        let mut other = model();
        other.id = "gpt-5.2".into();
        let input = convert(&other, vec![message.into()], &options);
        assert!(input[0].get("namespace").is_none() && input[0].get("id").is_none());
        assert!(input[1].get("namespace").is_none() && input[1].get("id").is_none());
    }

    fn completed(id: &str) -> Value {
        json!({"type": "response.completed", "response": {"id": id, "status": "completed"}})
    }

    #[test]
    fn rejects_streams_without_terminal_event() {
        let events = [
            json!({"type": "response.created", "response": {"id": "resp_eof"}}),
            json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs", "summary": []}}),
            json!({"type": "response.reasoning_text.delta", "output_index": 0, "delta": "partial"}),
        ];
        let (output, result, _) = run(&events, ResponsesStreamOptions::default());
        assert_eq!(
            result.unwrap_err().to_string(),
            "OpenAI Responses stream ended before a terminal response event"
        );
        assert_eq!(output.response_id.as_deref(), Some("resp_eof"));
        assert_eq!(output.content[0].as_thinking().unwrap().thinking, "partial");
    }

    #[test]
    fn rejects_unfinished_tool_calls() {
        let events = [
            json!({"type": "response.output_item.added", "output_index": 0,
                   "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "bash", "arguments": ""}}),
            json!({"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "{\"command\":\"rm -rf /tmp/build"}),
            completed("resp"),
        ];
        let (_, result, _) = run(&events, ResponsesStreamOptions::default());
        assert_eq!(
            result.unwrap_err().to_string(),
            "OpenAI Responses stream completed with an unfinished tool call: bash (call_1|fc_1)"
        );
    }

    #[test]
    fn rejects_parallel_tool_calls_without_output_index() {
        let call = |n: &str, arguments: &str| json!({"type": "function_call", "id": format!("fc_{n}"), "call_id": format!("call_{n}"), "name": "bash", "arguments": arguments});
        let events = [
            json!({"type": "response.output_item.added", "item": call("a", "")}),
            json!({"type": "response.function_call_arguments.delta", "delta": "{\"command\":\"echo a\"}"}),
            json!({"type": "response.output_item.added", "item": call("b", "")}),
            json!({"type": "response.function_call_arguments.delta", "delta": "{\"command\":\"echo b\"}"}),
            json!({"type": "response.output_item.done", "item": call("a", "{\"command\":\"echo a\"}")}),
            json!({"type": "response.output_item.done", "item": call("b", "{\"command\":\"echo b\"}")}),
            completed("resp"),
        ];
        let (_, result, _) = run(&events, ResponsesStreamOptions::default());
        assert_eq!(
            result.unwrap_err().to_string(),
            "OpenAI Responses stream completed with an unfinished tool call: bash (call_a|fc_a)"
        );
    }

    fn phased(first: &str, second: &str, terminal: Value) -> Vec<Value> {
        vec![
            json!({"type": "response.output_item.added", "output_index": 0,
                   "item": {"type": "message", "id": "msg_phase", "role": "assistant", "status": "in_progress", "content": [], "phase": first}}),
            json!({"type": "response.output_item.done", "output_index": 0,
                   "item": {"type": "message", "id": "msg_phase", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": "answer", "annotations": []}], "phase": second}}),
            terminal,
        ]
    }

    #[test]
    fn message_phases_and_text_signature() {
        for (first, second) in [
            ("commentary", "commentary"),
            ("final_answer", "final_answer"),
            ("commentary", "final_answer"),
        ] {
            let (output, result, _) = run(
                &phased(first, second, completed("r")),
                ResponsesStreamOptions::default(),
            );
            result.unwrap();
            assert_eq!(output.stop_reason, StopReason::Stop);
            let text = output.content[0].as_text().unwrap();
            assert_eq!(text.text, "answer");
            assert_eq!(
                text.text_signature.as_deref(),
                Some(format!(r#"{{"v":1,"id":"msg_phase","phase":"{second}"}}"#).as_str())
            );
        }
        let incomplete = json!({"type": "response.incomplete", "response": {"id": "r", "status": "incomplete",
                                "incomplete_details": {"reason": "max_output_tokens"}}});
        let (output, result, _) = run(
            &phased("final_answer", "final_answer", incomplete),
            ResponsesStreamOptions::default(),
        );
        result.unwrap();
        assert_eq!(output.stop_reason, StopReason::Length);
    }

    #[test]
    fn completed_terminal_usage_and_cost() {
        let events = [
            json!({"type": "response.completed", "response": {"id": "resp_completed", "status": "completed",
            "usage": {"input_tokens": 20, "output_tokens": 7, "total_tokens": 27,
                      "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 3},
                      "output_tokens_details": {"reasoning_tokens": 4}}}}),
        ];
        let (output, result, _) = run(&events, ResponsesStreamOptions::default());
        result.unwrap();
        assert_eq!(output.response_id.as_deref(), Some("resp_completed"));
        assert_eq!(output.stop_reason, StopReason::Stop);
        assert_eq!(output.raw_stop_reason.as_deref(), Some("completed"));
        assert_eq!(
            (
                output.usage.input,
                output.usage.output,
                output.usage.cache_read,
                output.usage.cache_write
            ),
            (15, 7, 2, 3)
        );
        assert_eq!(output.usage.reasoning, Some(4));
        assert_eq!(output.usage.total_tokens, 27);
        assert!((output.usage.cost.input - 15.0 / 1e6).abs() < 1e-12);
        assert!((output.usage.cost.output - 14.0 / 1e6).abs() < 1e-12);
        assert!((output.usage.cost.cache_read - 1.0 / 1e6).abs() < 1e-12);
    }

    #[test]
    fn incomplete_terminal_events() {
        for (reason, stop, message) in [
            ("max_output_tokens", StopReason::Length, None),
            (
                "content_filter",
                StopReason::Error,
                Some("Response incomplete: content_filter"),
            ),
            (
                "max_time_limit",
                StopReason::Error,
                Some("Response incomplete: max_time_limit"),
            ),
        ] {
            let events = [
                json!({"type": "response.incomplete", "response": {"id": "resp_incomplete", "status": "incomplete",
                "incomplete_details": {"reason": reason},
                "usage": {"input_tokens": 30, "output_tokens": 12, "total_tokens": 42, "input_tokens_details": {"cached_tokens": 5}}}}),
            ];
            let (output, result, _) = run(&events, ResponsesStreamOptions::default());
            result.unwrap();
            assert_eq!(output.stop_reason, stop);
            assert_eq!(output.raw_stop_reason, Some(format!("incomplete.{reason}")));
            assert_eq!(output.error_message.as_deref(), message);
            assert_eq!((output.usage.input, output.usage.cache_read), (25, 5));
        }
    }

    #[test]
    fn failed_and_error_events() {
        let events = [
            json!({"type": "response.failed", "response": {"id": "r", "status": "failed",
                                "error": {"code": "server_error", "message": "boom"}}}),
        ];
        let (output, result, _) = run(&events, ResponsesStreamOptions::default());
        assert_eq!(result.unwrap_err().to_string(), "server_error: boom");
        assert_eq!(output.raw_stop_reason.as_deref(), Some("failed"));
        let events = [json!({"type": "response.failed", "response": {"status": "failed"}})];
        assert_eq!(
            run(&events, ResponsesStreamOptions::default())
                .1
                .unwrap_err()
                .to_string(),
            "Unknown error (no error details in response)"
        );
        let events = [json!({"type": "error", "code": "rate_limit_exceeded", "message": "slow down"})];
        assert_eq!(
            run(&events, ResponsesStreamOptions::default())
                .1
                .unwrap_err()
                .to_string(),
            "Error Code rate_limit_exceeded: slow down"
        );
        let events = [json!({"type": "response.completed", "response": {"status": "weird"}})];
        assert_eq!(
            run(&events, ResponsesStreamOptions::default())
                .1
                .unwrap_err()
                .to_string(),
            "Unhandled stop reason: weird"
        );
    }

    #[test]
    fn reasoning_summary_and_signature() {
        let item = json!({"type": "reasoning", "id": "rs_1", "summary": [
            {"type": "summary_text", "text": "First"}, {"type": "summary_text", "text": "Second"}],
            "encrypted_content": "enc"});
        let events = [
            json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
            json!({"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "First"}),
            json!({"type": "response.reasoning_summary_part.done", "output_index": 0}),
            json!({"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "Second"}),
            json!({"type": "response.output_item.done", "output_index": 0, "item": item}),
            json!({"type": "response.unknown_future_event", "output_index": 0}),
            completed("r"),
        ];
        let (output, result, kinds) = run(&events, ResponsesStreamOptions::default());
        result.unwrap();
        let thinking = output.content[0].as_thinking().unwrap();
        assert_eq!(thinking.thinking, "First\n\nSecond");
        assert_eq!(thinking.thinking_signature.as_deref(), Some(item.to_string().as_str()));
        assert_eq!(
            kinds,
            [
                "start",
                "thinking_start",
                "thinking_delta",
                "thinking_delta",
                "thinking_delta",
                "thinking_end"
            ]
        );
        // Replays as the stored reasoning item.
        let mut message = output;
        message.stop_reason = StopReason::Stop;
        let input = convert(
            &model(),
            vec![Message::user("q"), message.into()],
            &ConvertResponsesMessagesOptions::default(),
        );
        assert_eq!(input[1], item);
    }

    #[test]
    fn reasoning_encrypted_content_backfill() {
        for (done_content, expected) in [
            (Some("from-output-item-done"), "from-output-item-done"),
            (None, "from-response-completed"),
        ] {
            let mut done = json!({"type": "reasoning", "id": "rs_x", "summary": []});
            if let Some(content) = done_content {
                done["encrypted_content"] = json!(content);
            }
            let mut completed_item = done.clone();
            completed_item["encrypted_content"] = json!("from-response-completed");
            let events = [
                json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs_x", "summary": []}}),
                json!({"type": "response.output_item.done", "output_index": 0, "item": done}),
                json!({"type": "response.completed", "response": {"id": "r", "status": "completed", "output": [completed_item]}}),
            ];
            let (output, result, _) = run(&events, ResponsesStreamOptions::default());
            result.unwrap();
            let signature: Value = serde_json::from_str(
                output.content[0]
                    .as_thinking()
                    .unwrap()
                    .thinking_signature
                    .as_ref()
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(signature["encrypted_content"], expected);
        }
    }

    #[test]
    fn tool_call_arguments_stream_and_finalize() {
        let added =
            json!({"type": "function_call", "id": "fc_test", "call_id": "call_test", "name": "edit", "arguments": ""});
        let events = |final_json: &str| {
            let mut done = added.clone();
            done["arguments"] = json!(final_json);
            vec![
                json!({"type": "response.output_item.added", "item": added}),
                json!({"type": "response.function_call_arguments.delta", "delta": "{\"path\":\"README.md\""}),
                json!({"type": "response.function_call_arguments.delta", "delta": ",\"content\":\"updated\"}"}),
                json!({"type": "response.function_call_arguments.done", "arguments": final_json}),
                json!({"type": "response.output_item.done", "item": done}),
                completed("r"),
            ]
        };
        let (output, result, kinds) = run(
            &events(r#"{"path":"README.md","content":"updated"}"#),
            ResponsesStreamOptions::default(),
        );
        result.unwrap();
        assert_eq!(output.stop_reason, StopReason::ToolUse);
        let call = output.tool_calls().next().unwrap();
        assert_eq!(call.id, "call_test|fc_test");
        assert_eq!(
            Value::Object(call.arguments.clone()),
            json!({"path": "README.md", "content": "updated"})
        );
        assert_eq!(
            kinds,
            [
                "start",
                "toolcall_start",
                "toolcall_delta",
                "toolcall_delta",
                "toolcall_end",
                r#"args:{"path":"README.md","content":"updated"}"#
            ]
        );
        // Malformed final arguments are marked instead of keeping the parsed prefix.
        let malformed = r#"{"path":"README.md","content":"updated","mode":.5}"#;
        let (output, result, _) = run(&events(malformed), ResponsesStreamOptions::default());
        result.unwrap();
        let call = output.tool_calls().next().unwrap();
        assert!(call.arguments.is_empty());
        assert!(
            call.arguments_error
                .as_deref()
                .unwrap()
                .ends_with(&format!("Received arguments:\n{malformed}"))
        );
    }

    #[test]
    fn function_namespace_from_output_item_done() {
        let events = [
            json!({"type": "response.output_item.added", "output_index": 0,
                   "item": {"type": "function_call", "id": "fc_test", "call_id": "call_test", "name": "lookup", "arguments": ""}}),
            json!({"type": "response.output_item.done", "output_index": 0,
                   "item": {"type": "function_call", "id": "fc_test", "call_id": "call_test", "name": "lookup",
                            "arguments": "{\"value\":\"hello\"}", "namespace": "dynamic_tools"}}),
            completed("resp_test"),
        ];
        let (output, result, kinds) = run(&events, ResponsesStreamOptions::default());
        result.unwrap();
        assert!(output.error_message.is_none());
        let call = output.tool_calls().next().unwrap();
        assert_eq!(call.namespace.as_deref(), Some("dynamic_tools"));
        assert_eq!(Value::Object(call.arguments.clone()), json!({"value": "hello"}));
        assert!(kinds.contains(&r#"args:{"value":"hello"}"#.to_owned()));
    }

    #[test]
    fn custom_tool_call_input_streams_as_json() {
        let grammar: GrammarToolInputProperties = [("query".to_owned(), "sql".to_owned())].into_iter().collect();
        let events = [
            json!({"type": "response.output_item.added", "output_index": 1,
                   "item": {"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "query", "input": ""}}),
            json!({"type": "response.custom_tool_call_input.delta", "output_index": 1, "delta": "SELECT \"a\""}),
            json!({"type": "response.custom_tool_call_input.delta", "output_index": 1, "delta": "\nFROM t"}),
            json!({"type": "response.custom_tool_call_input.done", "output_index": 1, "input": "SELECT \"a\"\nFROM t"}),
            json!({"type": "response.output_item.done", "output_index": 1,
                   "item": {"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "query",
                            "input": "SELECT \"a\"\nFROM t", "namespace": "ns"}}),
            completed("r"),
        ];
        let options = ResponsesStreamOptions {
            grammar_tool_input_properties: grammar.clone(),
            ..ResponsesStreamOptions::default()
        };
        let (output, result, kinds) = run(&events, options);
        result.unwrap();
        let call = output.tool_calls().next().unwrap();
        assert_eq!(call.id, "call_1|ctc_1");
        assert_eq!(call.namespace.as_deref(), Some("ns"));
        assert_eq!(
            Value::Object(call.arguments.clone()),
            json!({"sql": "SELECT \"a\"\nFROM t"})
        );
        assert_eq!(
            kinds,
            [
                "start",
                "toolcall_start",
                "toolcall_delta",
                "toolcall_delta",
                "toolcall_delta",
                "toolcall_end",
                r#"args:{"sql":"SELECT \"a\"\nFROM t"}"#
            ]
        );
        // Round-trips as a custom tool call with its result.
        let mut message = output;
        message.stop_reason = StopReason::ToolUse;
        let result = ToolResultMessage::text(message.tool_calls().next().unwrap(), "rows", false);
        let convert_options = ConvertResponsesMessagesOptions {
            grammar_tool_input_properties: Some(&grammar),
            ..ConvertResponsesMessagesOptions::default()
        };
        let input = convert(&model(), vec![message.into(), result.into()], &convert_options);
        assert_eq!(
            input,
            [
                json!({"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "query",
                       "input": "SELECT \"a\"\nFROM t", "namespace": "ns"}),
                json!({"type": "custom_tool_call_output", "call_id": "call_1", "output": "rows"})
            ]
        );
    }

    #[test]
    fn grammar_input_must_grow_monotonically() {
        let mut buffer = GrammarToolInputJsonBuffer::default();
        assert_eq!(
            append_grammar_tool_input_json_delta(&mut buffer, "p", "ab", false).unwrap(),
            Some("{\"p\":\"ab".into())
        );
        assert!(append_grammar_tool_input_json_delta(&mut buffer, "p", "x", false).is_err());
        assert_eq!(
            append_grammar_tool_input_json_delta(&mut buffer, "p", "ab", true).unwrap(),
            Some("\"}".into())
        );
        assert_eq!(
            append_grammar_tool_input_json_delta(&mut buffer, "p", "ab", true).unwrap(),
            None
        );
        assert!(append_grammar_tool_input_json_delta(&mut buffer, "p", "abc", true).is_err());
    }

    #[test]
    fn refusal_and_text_deltas() {
        let events = [
            json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg_1", "content": []}}),
            json!({"type": "response.output_text.delta", "output_index": 0, "delta": "Hel"}),
            json!({"type": "response.refusal.delta", "output_index": 0, "delta": "lo"}),
            json!({"type": "response.output_text.delta", "output_index": 3, "delta": "ignored"}),
            json!({"type": "response.output_item.done", "output_index": 0, "item": {"type": "message", "id": "msg_1",
                   "content": [{"type": "output_text", "text": "Hel"}, {"type": "refusal", "refusal": "lo"}]}}),
            completed("r"),
        ];
        let (output, result, kinds) = run(&events, ResponsesStreamOptions::default());
        result.unwrap();
        assert_eq!(output.text(), "Hello");
        assert_eq!(
            output.content[0].as_text().unwrap().text_signature.as_deref(),
            Some(r#"{"v":1,"id":"msg_1"}"#)
        );
        assert_eq!(kinds, ["start", "text_start", "text_delta", "text_delta", "text_end"]);
    }

    #[test]
    fn service_tier_pricing_uses_response_tier() {
        fn double(_: &Model, usage: &mut Usage, tier: Option<&str>) {
            if tier == Some("priority") {
                usage.cost.total *= 2.0;
            }
        }
        let events = [
            json!({"type": "response.completed", "response": {"status": "completed", "service_tier": "priority",
            "usage": {"input_tokens": 1_000_000, "output_tokens": 0, "total_tokens": 1_000_000}}}),
        ];
        let options = ResponsesStreamOptions {
            service_tier: Some("flex".into()),
            apply_service_tier_pricing: Some(double),
            ..ResponsesStreamOptions::default()
        };
        let (output, result, _) = run(&events, options);
        result.unwrap();
        assert!((output.usage.cost.total - 2.0).abs() < 1e-9);
    }
}
