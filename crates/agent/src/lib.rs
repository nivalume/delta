//! Stateful agent with tool execution and event streaming, built on [`delta_ai`].
//!
//! Rust port of [`@earendil-works/pi-agent-core`](https://github.com/earendil-works/pi/tree/main/packages/agent).
//! Design notes live in `docs/design/` of this crate.

pub mod agent;
pub mod agent_loop;
pub mod hooks;
pub mod proxy;
pub mod tool;
pub mod types;

pub use agent::{
    Agent, AgentError, AgentInitialState, AgentListener, AgentOptions, AgentState, StreamingMessage, SubscriptionId,
};
pub use agent_loop::{
    AgentEventSink, AgentEventStream, AgentLoopError, RunToolCallOptions, agent_loop, agent_loop_continue,
    declare_tool_changes, run_agent_loop, run_agent_loop_continue, run_tool_call,
};
pub use hooks::{
    AfterToolCallContext, AgentContext, AgentHooks, AgentLoopConfig, AgentLoopTurnUpdate, AgentRequestUpdate,
    AgentTurnContext, BeforeToolCallContext, DefaultHooks, MessageSource, PrepareRequestContext, StreamFn,
    models_stream_fn,
};
pub use proxy::{ProxyAssistantMessageEvent, ProxyStreamOptions, stream_proxy};
pub use tool::{AgentTool, FnTool, ToolContext, ToolError, ToolReplay, ToolUpdates};
#[cfg(feature = "schemars")]
pub use tool::{schema_for, typed_tool};
pub use types::{
    AfterToolCallResult, AgentEvent, AgentMessage, AgentToolCallOutcome, AgentToolResult, AgentTurnDecision,
    BeforeToolCallResult, CustomMessage, NoCustom, QueueMode, ToolExecutionMode,
};

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
