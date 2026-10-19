//! Loop configuration and the hooks an application plugs into the agent loop.

use futures::future::BoxFuture;
use serde_json::Value;
use std::fmt;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use delta_ai::{
    AssistantMessage, AssistantMessageEventStream, Message, Model, ModelThinkingLevel, Models, SimpleStreamOptions,
    ToolCall, ToolResultMessage, TranscriptContext,
};

use crate::tool::AgentTool;
use crate::types::{
    AfterToolCallResult, AgentMessage, AgentToolResult, AgentTurnDecision, BeforeToolCallResult, CustomMessage,
    NoCustom, ToolExecutionMode,
};

/// Stream function used by the loop; [`models_stream_fn`] adapts [`Models::stream_simple`].
///
/// It receives a normalized transcript (the prompt and tools are in system messages) and
/// must encode failures in the returned stream.
pub type StreamFn = Arc<
    dyn Fn(Model, TranscriptContext, SimpleStreamOptions) -> BoxFuture<'static, AssistantMessageEventStream>
        + Send
        + Sync,
>;

/// [`StreamFn`] that streams through a [`Models`] collection.
pub fn models_stream_fn(models: Models) -> StreamFn {
    Arc::new(move |model, context, options| {
        let models = models.clone();
        // The transcript is already normalized; normalizing it again is a no-op.
        let context = delta_ai::Context::new(context.into_messages());
        Box::pin(async move { models.stream_simple(&model, &context, options) })
    })
}

/// Transcript and executable tools of a run.
pub struct AgentContext<C = NoCustom> {
    pub messages: Vec<AgentMessage<C>>,
    pub tools: Vec<Arc<dyn AgentTool>>,
}

impl<C: Clone> Clone for AgentContext<C> {
    fn clone(&self) -> Self {
        Self {
            messages: self.messages.clone(),
            tools: self.tools.clone(),
        }
    }
}

impl<C: fmt::Debug> fmt::Debug for AgentContext<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tools: Vec<&str> = self.tools.iter().map(|tool| tool.definition().name.as_str()).collect();
        f.debug_struct("AgentContext")
            .field("messages", &self.messages)
            .field("tools", &tools)
            .finish()
    }
}

impl<C> Default for AgentContext<C> {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            tools: Vec::new(),
        }
    }
}

/// Input of [`AgentHooks::before_tool_call`].
#[derive(Debug)]
pub struct BeforeToolCallContext<'a, C = NoCustom> {
    /// The assistant message that requested the call.
    pub assistant_message: &'a AssistantMessage,
    pub tool_call: &'a ToolCall,
    /// Validated arguments.
    pub args: &'a Value,
    pub context: &'a AgentContext<C>,
}

/// Input of [`AgentHooks::after_tool_call`].
#[derive(Debug)]
pub struct AfterToolCallContext<'a, C = NoCustom> {
    pub assistant_message: &'a AssistantMessage,
    pub tool_call: &'a ToolCall,
    pub args: &'a Value,
    /// The executed result before overrides.
    pub result: &'a AgentToolResult,
    pub is_error: bool,
    pub context: &'a AgentContext<C>,
}

/// A completed turn, for [`AgentHooks::finish_turn`] and [`AgentHooks::prepare_next_turn`].
#[derive(Debug)]
pub struct AgentTurnContext<'a, C = NoCustom> {
    pub message: &'a AssistantMessage,
    pub tool_results: &'a [ToolResultMessage],
    /// Context after the turn's messages were appended.
    pub context: &'a AgentContext<C>,
    /// Messages this run returns if it exits now.
    pub new_messages: &'a [AgentMessage<C>],
}

/// Input of [`AgentHooks::prepare_request`].
#[derive(Debug)]
pub struct PrepareRequestContext<'a, C = NoCustom> {
    pub context: &'a AgentContext<C>,
    pub model: &'a Model,
    pub thinking_level: ModelThinkingLevel,
}

/// Replacement state for the request being prepared and later requests of the run.
#[derive(Debug)]
pub struct AgentRequestUpdate<C = NoCustom> {
    pub context: Option<AgentContext<C>>,
    pub model: Option<Model>,
    pub thinking_level: Option<ModelThinkingLevel>,
}

impl<C> Default for AgentRequestUpdate<C> {
    fn default() -> Self {
        Self {
            context: None,
            model: None,
            thinking_level: None,
        }
    }
}

/// Replacement state for the next turn.
#[derive(Debug)]
pub struct AgentLoopTurnUpdate<C = NoCustom> {
    pub context: Option<AgentContext<C>>,
    /// Messages to append before the next request, with normal lifecycle events.
    pub messages: Option<Vec<AgentMessage<C>>>,
    pub model: Option<Model>,
    pub thinking_level: Option<ModelThinkingLevel>,
}

impl<C> Default for AgentLoopTurnUpdate<C> {
    fn default() -> Self {
        Self {
            context: None,
            messages: None,
            model: None,
            thinking_level: None,
        }
    }
}

/// Application hooks of the agent loop. Every method has a default; implement the ones you
/// need. Upstream configures each hook as a separate callback; one trait object keeps the
/// borrowed contexts easy to use from Rust.
#[allow(unused_variables)]
pub trait AgentHooks<C: CustomMessage = NoCustom>: Send + Sync {
    /// Convert agent messages to LLM messages before each request. Default: keep the LLM
    /// messages, drop custom ones.
    fn convert_to_llm<'a>(&'a self, messages: &'a [AgentMessage<C>]) -> BoxFuture<'a, Vec<Message>> {
        Box::pin(async move {
            messages
                .iter()
                .filter_map(|message| message.as_llm().cloned())
                .collect()
        })
    }

    /// Transform the transcript before `convert_to_llm` (pruning, injected context).
    fn transform_context<'a>(
        &'a self,
        messages: Vec<AgentMessage<C>>,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Vec<AgentMessage<C>>> {
        Box::pin(async move { messages })
    }

    /// API key for each request, for short-lived tokens. `None` keeps the configured key.
    fn get_api_key<'a>(&'a self, provider: &'a str) -> BoxFuture<'a, Option<String>> {
        Box::pin(async { None })
    }

    /// Runs after argument validation, before execution; can block the call.
    fn before_tool_call<'a>(
        &'a self,
        context: BeforeToolCallContext<'a, C>,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Option<BeforeToolCallResult>> {
        Box::pin(async { None })
    }

    /// Runs after execution, before `ToolExecutionEnd`; can override parts of the result.
    fn after_tool_call<'a>(
        &'a self,
        context: AfterToolCallContext<'a, C>,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Option<AfterToolCallResult>> {
        Box::pin(async { None })
    }

    /// Runs after the assistant message and tool results of a turn, before `TurnEnd`.
    /// The decision is ignored for error and aborted responses.
    fn finish_turn<'a>(
        &'a self,
        turn: AgentTurnContext<'a, C>,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Option<AgentTurnDecision>> {
        Box::pin(async { None })
    }

    /// Runs before every provider request, including the first, after pending messages were
    /// appended. Does not poll queues.
    fn prepare_request<'a>(
        &'a self,
        request: PrepareRequestContext<'a, C>,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Option<AgentRequestUpdate<C>>> {
        Box::pin(async { None })
    }

    /// Runs after `TurnEnd` when the loop continues, before the next turn starts.
    fn prepare_next_turn<'a>(
        &'a self,
        turn: AgentTurnContext<'a, C>,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Option<AgentLoopTurnUpdate<C>>> {
        Box::pin(async { None })
    }
}

/// Hooks that keep every default.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultHooks;

impl<C: CustomMessage> AgentHooks<C> for DefaultHooks {}

/// Source of queued messages (steering or follow-up).
pub type MessageSource<C> = Arc<dyn Fn() -> BoxFuture<'static, Vec<AgentMessage<C>>> + Send + Sync>;

/// Configuration of the low-level loop.
pub struct AgentLoopConfig<C: CustomMessage = NoCustom> {
    pub model: Model,
    /// Request options; `reasoning` is the thinking level (`None` = off). The loop sets
    /// `cancel` and `api_key` per request.
    pub stream_options: SimpleStreamOptions,
    /// Default: parallel. A tool with sequential mode makes its batch sequential.
    pub tool_execution: ToolExecutionMode,
    pub hooks: Arc<dyn AgentHooks<C>>,
    /// Steering messages, polled after each turn (and at the start of a run).
    pub get_steering_messages: Option<MessageSource<C>>,
    /// Follow-up messages, polled when the agent would otherwise stop.
    pub get_follow_up_messages: Option<MessageSource<C>>,
}

impl<C: CustomMessage> Clone for AgentLoopConfig<C> {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            stream_options: self.stream_options.clone(),
            tool_execution: self.tool_execution,
            hooks: self.hooks.clone(),
            get_steering_messages: self.get_steering_messages.clone(),
            get_follow_up_messages: self.get_follow_up_messages.clone(),
        }
    }
}

impl<C: CustomMessage> fmt::Debug for AgentLoopConfig<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentLoopConfig")
            .field("model", &self.model.id)
            .field("tool_execution", &self.tool_execution)
            .finish_non_exhaustive()
    }
}

impl<C: CustomMessage> AgentLoopConfig<C> {
    /// Config with default hooks and options.
    pub fn new(model: Model) -> Self {
        Self {
            model,
            stream_options: SimpleStreamOptions::default(),
            tool_execution: ToolExecutionMode::Parallel,
            hooks: Arc::new(DefaultHooks),
            get_steering_messages: None,
            get_follow_up_messages: None,
        }
    }

    pub(crate) fn thinking_level(&self) -> ModelThinkingLevel {
        self.stream_options
            .reasoning
            .map_or(ModelThinkingLevel::Off, ModelThinkingLevel::from)
    }

    pub(crate) fn set_thinking_level(&mut self, level: ModelThinkingLevel) {
        self.stream_options.reasoning = level.reasoning();
    }
}
