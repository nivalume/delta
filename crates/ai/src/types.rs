//! Message, content, tool and usage types.
//!
//! The JSON shape of every type matches upstream pi-ai (camelCase fields, `type`/`role`
//! tags), so a context serialized by pi, pi-py or tau can be read by the others.

use indexmap::IndexMap;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

use crate::model::ModelThinkingLevel;

/// A JSON object, in insertion order.
pub type JsonObject = serde_json::Map<String, Value>;

/// Provider-scoped environment overrides. Values take precedence over the process environment.
pub type ProviderEnv = BTreeMap<String, String>;

/// Request headers. A `None` value suppresses a provider/API default header with the same name.
pub type ProviderHeaders = IndexMap<String, Option<String>>;

/// Milliseconds since the Unix epoch.
pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}

/// Text content block.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextContent {
    pub text: String,
    /// Provider message metadata, e.g. the OpenAI Responses message id (legacy id string or
    /// [`TextSignatureV1`] JSON).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}

impl TextContent {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            text_signature: None,
        }
    }
}

/// Structured form of [`TextContent::text_signature`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextSignatureV1 {
    pub v: u8,
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
}

/// Thinking/reasoning content block.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingContent {
    pub thinking: String,
    /// Provider-specific opaque or serialized reasoning replay data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    /// The thinking was redacted by safety filters; the encrypted payload is in
    /// `thinking_signature` so it can be passed back for multi-turn continuity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redacted: Option<bool>,
}

impl ThinkingContent {
    pub fn new(thinking: impl Into<String>) -> Self {
        Self {
            thinking: thinking.into(),
            ..Self::default()
        }
    }

    pub fn is_redacted(&self) -> bool {
        self.redacted == Some(true)
    }
}

/// Base64-encoded image content block.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    /// Base64-encoded image data.
    pub data: String,
    /// E.g. `image/png`.
    pub mime_type: String,
}

/// A tool call made by the model.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: JsonObject,
    /// Google: opaque signature for reusing thought context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    /// OpenAI Responses namespace for calls to dynamically loaded or namespaced tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// Set when the model's complete arguments text was not a valid JSON object. `arguments`
    /// is then empty; tool argument validation fails with this message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments_error: Option<String>,
}

/// Content of user messages and tool results.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum UserBlock {
    Text(TextContent),
    Image(ImageContent),
}

impl UserBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(TextContent::new(text))
    }
}

/// Content of assistant messages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AssistantBlock {
    Text(TextContent),
    Thinking(ThinkingContent),
    ToolCall(ToolCall),
}

impl AssistantBlock {
    pub fn as_text(&self) -> Option<&TextContent> {
        match self {
            Self::Text(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_thinking(&self) -> Option<&ThinkingContent> {
        match self {
            Self::Thinking(thinking) => Some(thinking),
            _ => None,
        }
    }

    pub fn as_tool_call(&self) -> Option<&ToolCall> {
        match self {
            Self::ToolCall(call) => Some(call),
            _ => None,
        }
    }
}

/// Text-only block list of system messages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum SystemBlock {
    Text(TextContent),
}

/// Message content that is either a plain string or a list of blocks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content<B> {
    Text(String),
    Blocks(Vec<B>),
}

impl<B> Default for Content<B> {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl<B> From<&str> for Content<B> {
    fn from(text: &str) -> Self {
        Self::Text(text.to_owned())
    }
}

impl<B> From<String> for Content<B> {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

/// Content of a user message.
pub type UserContent = Content<UserBlock>;
/// Content of a system message.
pub type SystemContent = Content<SystemBlock>;

impl UserContent {
    /// The content as blocks; a plain string becomes one text block.
    pub fn to_blocks(&self) -> Vec<UserBlock> {
        match self {
            Self::Text(text) => vec![UserBlock::text(text.clone())],
            Self::Blocks(blocks) => blocks.clone(),
        }
    }

    /// Joined text of every text block.
    pub fn text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Blocks(blocks) => blocks
                .iter()
                .filter_map(|block| match block {
                    UserBlock::Text(text) => Some(text.text.as_str()),
                    UserBlock::Image(_) => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

impl SystemContent {
    /// Joined text of every block.
    pub fn text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Blocks(blocks) => blocks
                .iter()
                .map(|SystemBlock::Text(text)| text.text.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

/// Token usage and cost of a response.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    /// Subset of `cache_write` written with 1h retention. Only Anthropic reports this split.
    #[serde(default, rename = "cacheWrite1h", skip_serializing_if = "Option::is_none")]
    pub cache_write_1h: Option<u64>,
    /// Reasoning tokens, when reported. A subset of `output`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
    pub total_tokens: u64,
    pub cost: UsageCost,
}

/// Cost of a response in USD.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}

/// How a generation ended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    /// Only in partial messages, while the stop reason is unknown.
    #[default]
    Pending,
    /// The model finished its turn.
    Stop,
    /// Output hit the maximum token limit.
    Length,
    /// The model called tools and expects tool results.
    ToolUse,
    /// The request failed.
    Error,
    /// The request was cancelled.
    Aborted,
    /// The provider continues the request asynchronously; see [`AssistantMessage::deferred`].
    Deferred,
}

impl StopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolUse => "toolUse",
            Self::Error => "error",
            Self::Aborted => "aborted",
            Self::Deferred => "deferred",
        }
    }

    /// `error` or `aborted`.
    pub fn is_failure(self) -> bool {
        matches!(self, Self::Error | Self::Aborted)
    }
}

/// Durable handle of a request a provider continues asynchronously.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeferredHandle {
    pub provider: String,
    pub model_id: String,
    pub api: String,
    /// Provider token, such as a response id or batch id plus row id.
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub poll_after_ms: Option<u64>,
    /// Provider conversion data required to reconstruct the final assistant message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// Error details of a diagnostic.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticErrorInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<Value>,
}

/// Redacted provider/runtime diagnostic for failures and recoveries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessageDiagnostic {
    #[serde(rename = "type")]
    pub kind: String,
    pub timestamp: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<DiagnosticErrorInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<JsonObject>,
}

impl AssistantMessageDiagnostic {
    pub fn new(kind: impl Into<String>, message: Option<String>, details: Option<JsonObject>) -> Self {
        Self {
            kind: kind.into(),
            timestamp: now_ms(),
            error: message.map(|message| DiagnosticErrorInfo {
                name: None,
                message,
                stack: None,
                code: None,
            }),
            details,
        }
    }
}

/// System instructions and tool declarations at one point in the transcript.
///
/// The leading system message is the system prompt. Later system messages change it:
/// `content` adds instructions from that point on, `sections` replace or remove named prompt
/// sections, and `tools_added`/`tools_removed` change the tool set. Replaying every system
/// message in order yields the current prompt and tools.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemMessage {
    pub content: SystemContent,
    /// Named, ordered prompt sections rendered verbatim after `content`. Later messages replace
    /// sections by name; `None` removes one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sections: Option<IndexMap<String, Option<String>>>,
    /// Complete definitions of tools that become available at this point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools_added: Option<Vec<Tool>>,
    /// Tools that stop being available at this point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools_removed: Option<Vec<ToolReference>>,
    pub timestamp: i64,
}

/// A user message.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessage {
    pub content: UserContent,
    pub timestamp: i64,
}

impl UserMessage {
    /// A text message stamped with the current time.
    pub fn new(content: impl Into<UserContent>) -> Self {
        Self {
            content: content.into(),
            timestamp: now_ms(),
        }
    }
}

/// An assistant response.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    pub content: Vec<AssistantBlock>,
    pub api: String,
    pub provider: String,
    pub model: String,
    /// Concrete model reported by the provider when different from the requested `model`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_model: Option<String>,
    /// Provider response/message identifier when the upstream API exposes one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Exact provider-native effort level used for this response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_thinking_level: Option<String>,
    /// Thinking level the agent loop requested for this response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<ModelThinkingLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<AssistantMessageDiagnostic>>,
    pub usage: Usage,
    pub stop_reason: StopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deferred: Option<DeferredHandle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_stop_reason: Option<String>,
    /// Provider indication of whether the model explicitly ended its turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_turn: Option<bool>,
    /// Unix timestamp in milliseconds when the request started.
    pub timestamp: i64,
    /// Milliseconds from `timestamp` until the response ended, from a monotonic clock.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

impl AssistantMessage {
    /// An empty, pending message for `model`, stamped with the current time.
    pub fn empty(api: impl Into<String>, provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api: api.into(),
            provider: provider.into(),
            model: model.into(),
            timestamp: now_ms(),
            ..Self::default()
        }
    }

    /// Joined text of every text block.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(AssistantBlock::as_text)
            .map(|text| text.text.as_str())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Every tool call, in order.
    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.content.iter().filter_map(AssistantBlock::as_tool_call)
    }

    pub fn push_diagnostic(&mut self, diagnostic: AssistantMessageDiagnostic) {
        self.diagnostics.get_or_insert_with(Vec::new).push(diagnostic);
    }
}

/// A tool call that another tool made while it ran.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NestedToolCallRecord {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<JsonObject>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments_bytes: Option<u64>,
    /// `ok`, `error` or `unfinished`.
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Bounded record of the nested calls a tool made.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NestedToolCalls {
    pub calls: Vec<NestedToolCallRecord>,
    pub complete: bool,
}

/// The result of a tool call.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<UserBlock>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    /// Usage from the tool execution itself. Not part of main LLM context accounting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Calls this tool made to other tools. Not sent to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nested_calls: Option<NestedToolCalls>,
    pub is_error: bool,
    pub timestamp: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

impl ToolResultMessage {
    /// A text result for `call`, stamped with the current time.
    pub fn text(call: &ToolCall, text: impl Into<String>, is_error: bool) -> Self {
        Self {
            tool_call_id: call.id.clone(),
            tool_name: call.name.clone(),
            content: vec![UserBlock::text(text)],
            is_error,
            timestamp: now_ms(),
            ..Self::default()
        }
    }
}

/// A transcript message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum Message {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
}

impl Message {
    pub fn role(&self) -> &'static str {
        match self {
            Self::System(_) => "system",
            Self::User(_) => "user",
            Self::Assistant(_) => "assistant",
            Self::ToolResult(_) => "toolResult",
        }
    }

    /// A user text message stamped with the current time.
    pub fn user(content: impl Into<UserContent>) -> Self {
        Self::User(UserMessage::new(content))
    }
}

impl From<SystemMessage> for Message {
    fn from(message: SystemMessage) -> Self {
        Self::System(message)
    }
}

impl From<UserMessage> for Message {
    fn from(message: UserMessage) -> Self {
        Self::User(message)
    }
}

impl From<AssistantMessage> for Message {
    fn from(message: AssistantMessage) -> Self {
        Self::Assistant(message)
    }
}

impl From<ToolResultMessage> for Message {
    fn from(message: ToolResultMessage) -> Self {
        Self::ToolResult(message)
    }
}

/// OpenAI grammar variants for constrained sampling.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarVariants {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_lark: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_regex: Option<String>,
}

/// How strictly a JSON-schema constraint is applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StrictMode {
    /// Use provider-side strict schema enforcement when supported, else plain tool calling.
    Prefer,
    /// Fail the request when the provider/model cannot honor it.
    Require,
}

/// Provider-side constrained sampling for a tool.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConstrainedSampling {
    JsonSchema { strict: StrictMode },
    Grammar { variants: GrammarVariants },
}

/// A tool the model may call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    pub name: String,
    pub description: String,
    /// JSON Schema of the arguments object.
    pub parameters: Value,
    /// `false` in JSON means the same as omitting it.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_constrained_sampling"
    )]
    pub constrained_sampling: Option<ConstrainedSampling>,
}

impl Tool {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            constrained_sampling: None,
        }
    }
}

fn deserialize_constrained_sampling<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<ConstrainedSampling>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Field {
        Flag(bool),
        Config(ConstrainedSampling),
    }
    match Option::<Field>::deserialize(deserializer)? {
        None | Some(Field::Flag(false)) => Ok(None),
        Some(Field::Flag(true)) => Err(serde::de::Error::custom(
            "constrainedSampling must be false or a config",
        )),
        Some(Field::Config(config)) => Ok(Some(config)),
    }
}

/// Reference to a tool by name.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolReference {
    pub name: String,
}

/// Request input accepted by the public stream entry points. `system_prompt` and `tools` are
/// shorthand for a leading system message; [`crate::normalize_context`] folds them into one.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Context {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

impl Context {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            system_prompt: None,
            messages,
            tools: None,
        }
    }

    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    #[must_use]
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }
}

/// Normalized request context passed to providers and API implementations. The prompt and
/// tool declarations are carried by the transcript's system messages. Only
/// [`crate::normalize_context`] and the transcript helpers produce it, so a raw [`Context`]
/// cannot reach provider code by accident.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct TranscriptContext {
    messages: Vec<Message>,
}

impl TranscriptContext {
    pub(crate) fn from_messages(messages: Vec<Message>) -> Self {
        Self { messages }
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn into_messages(self) -> Vec<Message> {
        self.messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn messages_round_trip_upstream_json() {
        let value = json!([
            {"role": "system", "content": "Be brief.", "sections": {"rules": "<r/>", "old": null}, "timestamp": 1},
            {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image", "data": "AA==", "mimeType": "image/png"}], "timestamp": 2},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hm", "thinkingSignature": "sig"},
                {"type": "text", "text": "ok"},
                {"type": "toolCall", "id": "c1", "name": "read", "arguments": {"path": "a"}}
            ], "api": "anthropic-messages", "provider": "anthropic", "model": "m",
             "usage": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0, "cacheWrite1h": 0, "totalTokens": 3,
                       "cost": {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0}},
             "stopReason": "toolUse", "timestamp": 3},
            {"role": "toolResult", "toolCallId": "c1", "toolName": "read", "content": [{"type": "text", "text": "x"}], "isError": false, "timestamp": 4}
        ]);
        let messages: Vec<Message> = serde_json::from_value(value.clone()).unwrap();
        assert!(matches!(&messages[2], Message::Assistant(m) if m.stop_reason == StopReason::ToolUse));
        assert_eq!(serde_json::to_value(&messages).unwrap(), value);
    }

    #[test]
    fn constrained_sampling_false_is_none() {
        let tool: Tool = serde_json::from_value(
            json!({"name": "t", "description": "d", "parameters": {}, "constrainedSampling": false}),
        )
        .unwrap();
        assert_eq!(tool.constrained_sampling, None);
        let tool: Tool = serde_json::from_value(json!({"name": "t", "description": "d", "parameters": {},
            "constrainedSampling": {"type": "json_schema", "strict": "prefer"}}))
        .unwrap();
        assert_eq!(
            tool.constrained_sampling,
            Some(ConstrainedSampling::JsonSchema {
                strict: StrictMode::Prefer
            })
        );
    }
}
