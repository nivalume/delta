//! Agent messages, events, tool results and loop configuration types.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Debug;
use std::sync::Arc;

use delta_ai::{AssistantMessage, AssistantMessageEvent, Message, ToolCall, ToolResultMessage, Usage, UserBlock};

/// App-defined message types that live in the transcript next to the LLM messages, for
/// example UI notifications. [`crate::AgentHooks::convert_to_llm`] filters or converts
/// them before each request.
///
/// Upstream extends `AgentMessage` through TypeScript declaration merging; delta uses a type
/// parameter instead.
pub trait CustomMessage: Clone + Debug + Send + Sync + Serialize + DeserializeOwned + 'static {
    /// The message role, e.g. `"notification"`. Must not be one of the LLM roles.
    fn role(&self) -> &str;
    /// Unix timestamp in milliseconds.
    fn timestamp(&self) -> i64;
}

/// The default: no custom messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoCustom {}

impl CustomMessage for NoCustom {
    fn role(&self) -> &str {
        match *self {}
    }

    fn timestamp(&self) -> i64 {
        match *self {}
    }
}

/// A transcript entry: an LLM message or an app-defined message.
///
/// Serializes as the inner message, so transcripts without custom messages have exactly
/// the shape of upstream's.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AgentMessage<C = NoCustom> {
    Llm(Message),
    Custom(C),
}

impl<C: CustomMessage> AgentMessage<C> {
    pub fn role(&self) -> &str {
        match self {
            Self::Llm(message) => message.role(),
            Self::Custom(custom) => custom.role(),
        }
    }

    pub fn timestamp(&self) -> i64 {
        match self {
            Self::Llm(Message::System(m)) => m.timestamp,
            Self::Llm(Message::User(m)) => m.timestamp,
            Self::Llm(Message::Assistant(m)) => m.timestamp,
            Self::Llm(Message::ToolResult(m)) => m.timestamp,
            Self::Custom(custom) => custom.timestamp(),
        }
    }

    pub fn as_llm(&self) -> Option<&Message> {
        match self {
            Self::Llm(message) => Some(message),
            Self::Custom(_) => None,
        }
    }

    pub fn as_assistant(&self) -> Option<&AssistantMessage> {
        match self {
            Self::Llm(Message::Assistant(message)) => Some(message),
            _ => None,
        }
    }

    pub fn is_system(&self) -> bool {
        matches!(self, Self::Llm(Message::System(_)))
    }
}

impl<C> From<Message> for AgentMessage<C> {
    fn from(message: Message) -> Self {
        Self::Llm(message)
    }
}

impl<C> From<AssistantMessage> for AgentMessage<C> {
    fn from(message: AssistantMessage) -> Self {
        Self::Llm(Message::Assistant(message))
    }
}

impl<C> From<ToolResultMessage> for AgentMessage<C> {
    fn from(message: ToolResultMessage) -> Self {
        Self::Llm(Message::ToolResult(message))
    }
}

impl<C> From<delta_ai::UserMessage> for AgentMessage<C> {
    fn from(message: delta_ai::UserMessage) -> Self {
        Self::Llm(Message::User(message))
    }
}

impl<C> From<delta_ai::SystemMessage> for AgentMessage<C> {
    fn from(message: delta_ai::SystemMessage) -> Self {
        Self::Llm(Message::System(message))
    }
}

/// How the tool calls of one assistant message run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolExecutionMode {
    /// Each call is prepared, executed and finalized before the next starts.
    Sequential,
    /// Calls are prepared in order, then allowed calls execute concurrently.
    /// `ToolExecutionEnd` follows completion order; tool result messages follow source order.
    #[default]
    Parallel,
}

/// How many queued messages a queue drain point injects.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QueueMode {
    All,
    #[default]
    OneAtATime,
}

/// Final or partial result of a tool.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentToolResult {
    /// Text or image content returned to the model.
    pub content: Vec<UserBlock>,
    /// Structured details for logs or UI rendering. Not sent to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    /// Machine-readable result matching the tool's output schema. Not sent to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_content: Option<Value>,
    /// Usage of the tool execution itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Report a failure without returning an error: the model sees `content` as an error
    /// result, `details` and `structured_content` are kept.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
    /// Hint to stop after the current tool batch; effective only when every finalized
    /// result of the batch sets it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub terminate: bool,
}

impl AgentToolResult {
    /// A text result.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![UserBlock::text(text)],
            ..Self::default()
        }
    }

    /// The result recorded for a failed call: the message as text, empty details.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![UserBlock::text(message)],
            details: Some(Value::Object(serde_json::Map::new())),
            ..Self::default()
        }
    }
}

/// Final outcome of a tool call after the hooks ran.
#[derive(Clone, Debug, PartialEq)]
pub struct AgentToolCallOutcome {
    pub tool_call: ToolCall,
    pub result: AgentToolResult,
    pub is_error: bool,
    /// Milliseconds `execute` took; `None` when the tool did not run.
    pub duration_ms: Option<u64>,
}

/// Events of an agent run.
///
/// `AgentEnd` is the last event of a run; the agent becomes idle after the listeners for
/// it finish.
#[derive(Clone, Debug)]
pub enum AgentEvent<C = NoCustom> {
    AgentStart,
    AgentEnd {
        messages: Vec<AgentMessage<C>>,
    },
    /// A turn is one assistant response plus its tool calls.
    TurnStart,
    TurnEnd {
        message: AgentMessage<C>,
        tool_results: Vec<ToolResultMessage>,
    },
    /// Emitted for system, user, assistant, tool result and custom messages.
    MessageStart {
        message: AgentMessage<C>,
    },
    /// Assistant messages only, while streaming. `message` is the response as of this event.
    MessageUpdate {
        message: Arc<AssistantMessage>,
        assistant_message_event: AssistantMessageEvent,
    },
    MessageEnd {
        message: AgentMessage<C>,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: Value,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: Value,
        partial_result: AgentToolResult,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: AgentToolResult,
        is_error: bool,
        duration_ms: Option<u64>,
    },
}

impl<C> AgentEvent<C> {
    /// Upstream event type name (`agent_start`, `message_update`, ...).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::AgentStart => "agent_start",
            Self::AgentEnd { .. } => "agent_end",
            Self::TurnStart => "turn_start",
            Self::TurnEnd { .. } => "turn_end",
            Self::MessageStart { .. } => "message_start",
            Self::MessageUpdate { .. } => "message_update",
            Self::MessageEnd { .. } => "message_end",
            Self::ToolExecutionStart { .. } => "tool_execution_start",
            Self::ToolExecutionUpdate { .. } => "tool_execution_update",
            Self::ToolExecutionEnd { .. } => "tool_execution_end",
        }
    }
}

/// Result of `before_tool_call`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BeforeToolCallResult {
    /// Prevent execution; the loop records an error result instead.
    pub block: bool,
    /// Text of the error result. Default: "Tool execution was blocked".
    pub reason: Option<String>,
    /// Early-termination hint for a blocked call.
    pub terminate: bool,
}

/// Field-by-field override returned by `after_tool_call`. Omitted fields keep the executed
/// result; there is no deep merge. Replacing `content` without `structured_content` drops
/// the structured content, which may no longer match.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AfterToolCallResult {
    pub content: Option<Vec<UserBlock>>,
    pub details: Option<Value>,
    pub structured_content: Option<Value>,
    pub is_error: Option<bool>,
    pub usage: Option<Usage>,
    pub terminate: Option<bool>,
}

/// Decision of `finish_turn`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentTurnDecision {
    /// Ensure one more provider request. Tool results, steering or follow-ups that already
    /// cause one satisfy it.
    Continue,
    /// End the run after `TurnEnd`, without polling queues.
    End,
}
