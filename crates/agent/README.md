# delta-agent

Stateful agent with tool execution and event streaming. Built on
[`delta-ai`](https://github.com/nivalume/pi-rs/tree/main/crates/ai).

Rust port of [`@earendil-works/pi-agent-core`](https://github.com/earendil-works/pi/tree/main/packages/agent),
part of [delta](https://github.com/nivalume/pi-rs). Import it as `delta_agent`.

- Tool execution, parallel by default or sequential, with argument validation and
  `before_tool_call`/`after_tool_call` hooks
- Steering and follow-up queues
- Lifecycle events (`AgentStart`, `TurnStart`, `MessageUpdate`, `ToolExecutionEnd`, ...)
- App-defined transcript messages through a type parameter
- A low-level loop (`agent_loop`) for callers that manage state themselves, and `stream_proxy`
  for sending model requests through your own server

The API follows upstream. Where Rust needs a different shape (one `AgentHooks` trait instead of
separate callbacks, `CancellationToken` instead of `AbortSignal`, custom messages through a type
parameter, no global default stream function), the sections below say so. The design notes in
[`docs/design/`](docs/design/README.md) list every difference from upstream.

## Installation

```bash
cargo add delta-agent
```

Nothing is published to crates.io yet. Use a git dependency meanwhile:

```toml
[dependencies]
delta-agent = { git = "https://github.com/nivalume/pi-rs" }
delta-ai = { git = "https://github.com/nivalume/pi-rs" }
```

The examples below also use `tokio`, `tokio-util` (`CancellationToken`), `futures` (`BoxFuture`,
`StreamExt`) and `serde_json`.

Cargo features:

| Feature | Default | Contents |
| --- | --- | --- |
| `schemars` | on | `typed_tool` and `schema_for`: tool parameter schemas generated from Rust types |

## Quick Start

```rust,no_run
use std::io::Write;

use delta_agent::{Agent, AgentEvent, AgentInitialState, AgentOptions, models_stream_fn};
use delta_ai::providers::builtin::anthropic_provider;
use delta_ai::{AssistantMessageEvent, Models};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = Models::default();
    models.set_provider(anthropic_provider());
    let model = models
        .get_model("anthropic", "claude-sonnet-4-6")
        .ok_or("model not found")?;

    let agent: Agent = Agent::new(AgentOptions {
        initial_state: AgentInitialState {
            system_prompt: Some("You are a helpful assistant.".into()),
            model: Some(model),
            ..Default::default()
        },
        ..AgentOptions::new(models_stream_fn(models))
    });

    agent.subscribe(|event, _cancel| async move {
        if let AgentEvent::MessageUpdate {
            assistant_message_event: AssistantMessageEvent::TextDelta { delta, .. },
            ..
        } = &*event
        {
            // Print just the new text chunk.
            print!("{delta}");
            let _ = std::io::stdout().flush();
        }
    });

    agent.prompt_text("Hello!", Vec::new()).await?;
    Ok(())
}
```

`AgentOptions::new` takes the stream function. There is no global default:
`models_stream_fn(models)` streams through a `delta_ai::Models` collection, `stream_proxy` (see
[Proxy Usage](#proxy-usage)) streams through your own server, and tests can pass a closure.

## Core Concepts

### AgentMessage vs LLM Message

The agent works with `AgentMessage<C>`, an enum with two variants:

- `AgentMessage::Llm(Message)`: a standard LLM message (`system`, `user`, `assistant`,
  `toolResult`)
- `AgentMessage::Custom(C)`: an app-defined message type, chosen with the type parameter `C`
  (see [Custom Message Types](#custom-message-types))

`C` defaults to `NoCustom`, an empty enum, so `AgentMessage`, `Agent` and the other generic types
can be written without it. `AgentMessage` serializes as the inner message, so a transcript without
custom messages has the same JSON shape as upstream's.

LLMs only understand the LLM messages. The `convert_to_llm` hook bridges this gap by filtering
and converting messages before each LLM call.

### Message Flow

```text
Vec<AgentMessage<C>> → transform_context() → Vec<AgentMessage<C>> → convert_to_llm() → Vec<Message> → LLM
                         (optional)                                  (default: drop custom messages)
```

1. **transform_context**: prune old messages, inject external context
2. **convert_to_llm**: filter out UI-only messages, convert custom types to LLM messages

Both are methods of the `AgentHooks` trait. Every method of the trait has a default
implementation, so you implement only the ones you need. The default `convert_to_llm` keeps the
LLM messages and drops custom ones. After `convert_to_llm`, the loop normalizes the messages into a
`delta_ai::TranscriptContext` and passes it to the stream function.

## Event Flow

The agent emits events for UI updates. The event sequence below helps build responsive
interfaces. `AgentEvent::kind()` returns upstream's event name (`agent_start`, `message_update`,
...) for logs.

### prompt() Event Sequence

When you call `prompt_text("Hello", Vec::new())`:

```text
prompt_text("Hello", vec![])
├─ AgentStart
├─ TurnStart
├─ MessageStart   { message: user message }      // Your prompt
├─ MessageEnd     { message: user message }
├─ MessageStart   { message: assistant message } // LLM starts responding
├─ MessageUpdate  { message: partial, .. }       // Streaming chunks
├─ MessageUpdate  { message: partial, .. }
├─ MessageEnd     { message: assistant message } // Complete response
├─ TurnEnd        { message, tool_results: [] }
└─ AgentEnd       { messages: [...] }
```

`AgentEnd.messages` holds the messages this run added. When a response ends with stop reason
`error` or `aborted`, the run emits `TurnEnd` and `AgentEnd` right after the assistant
`MessageEnd`. It runs no tools and reads no queues.

### With Tool Calls

If the assistant calls tools, the loop continues:

```text
prompt_text("Read config.json", vec![])
├─ AgentStart
├─ TurnStart
├─ MessageStart/End    { user message }
├─ MessageStart        { assistant message with tool call }
├─ MessageUpdate...
├─ MessageEnd          { assistant message }
├─ ToolExecutionStart  { tool_call_id, tool_name, args }
├─ ToolExecutionUpdate { partial_result }                  // If the tool streams
├─ ToolExecutionEnd    { tool_call_id, result, is_error }
├─ MessageStart/End    { tool result message }
├─ TurnEnd             { message, tool_results: [tool result] }
│
├─ TurnStart                                               // Next turn
├─ MessageStart        { assistant message }               // LLM responds to the tool result
├─ MessageUpdate...
├─ MessageEnd
├─ TurnEnd
└─ AgentEnd
```

Tool execution mode is configurable with `ToolExecutionMode`:

- `Parallel` (default): prepare tool calls one by one (`ToolExecutionStart`, argument
  validation, `before_tool_call`, in source order), execute allowed tools concurrently, emit
  `ToolExecutionEnd` as soon as each tool is finalized, then emit tool result messages and
  `TurnEnd.tool_results` in assistant source order
- `Sequential`: execute tool calls one by one; each call is prepared, executed and finalized
  (including its tool result message) before the next starts

In parallel mode, tool completion events follow tool completion order, but persisted tool result
messages still follow assistant source order.

The mode can be set for the agent with `AgentOptions::tool_execution` (or
`Agent::set_tool_execution`, or `AgentLoopConfig::tool_execution` for the low-level loop), or per
tool with `AgentTool::execution_mode()` (`FnTool::with_execution_mode`). If any tool call in a
batch targets a tool whose mode is `Sequential`, the entire batch executes sequentially regardless
of the agent setting.

The `before_tool_call` hook runs after `ToolExecutionStart` and argument validation. It can block
execution by returning `BeforeToolCallResult { block: true, .. }`, and can set `terminate: true` on
the blocked result. The `after_tool_call` hook runs after the tool finishes, before
`ToolExecutionEnd` and the tool result message events. It returns an `AfterToolCallResult` that
overrides fields of the result.

Tools (`AgentToolResult::terminate`), blocked `before_tool_call` results and `after_tool_call`
overrides can set `terminate` to hint that the automatic follow-up LLM call should be skipped. The
loop only stops early when every finalized tool result in that batch sets `terminate`. Mixed
batches continue normally.

A call to an unknown tool, a call whose arguments fail validation, and a blocked call get an error
result without running. When a response stops with reason `length`, the arguments of its tool
calls may be truncated, so every call in it gets an error result without running.

When you use `Agent`, assistant `MessageEnd` processing is a barrier before tool preflight begins:
the agent updates its state and awaits every listener for an event before the loop continues. That
means `before_tool_call` sees agent state that already includes the assistant message that
requested the tool call.

### Request preparation and turn finalization

`prepare_request` runs immediately before every conversational provider request, including the
first. Use it to install canonical persisted context after pending input has been emitted. It
returns an `AgentRequestUpdate` that can replace the context, the model and the thinking level for
this request and later ones:

```rust
use delta_agent::{AgentContext, AgentHooks, AgentMessage, AgentRequestUpdate, PrepareRequestContext};
use futures::future::BoxFuture;
use tokio_util::sync::CancellationToken;

struct SessionHooks;

impl SessionHooks {
    async fn load_model_context(&self) -> Vec<AgentMessage> {
        // Read the persisted transcript.
        Vec::new()
    }
}

impl AgentHooks for SessionHooks {
    fn prepare_request<'a>(
        &'a self,
        request: PrepareRequestContext<'a>,
        _cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Option<AgentRequestUpdate>> {
        Box::pin(async move {
            let context = AgentContext {
                messages: self.load_model_context().await,
                tools: request.context.tools.clone(),
            };
            Some(AgentRequestUpdate {
                context: Some(context),
                ..Default::default()
            })
        })
    }
}
```

`prepare_request` does not poll queues. Steering queued while it runs waits for the next normal
steering poll.

`finish_turn` runs after the assistant message and all tool results are finalized, but before
`TurnEnd`. It runs for normal, error and aborted responses:

```rust
use delta_agent::{AgentHooks, AgentTurnContext, AgentTurnDecision};
use delta_ai::AssistantMessage;
use futures::future::BoxFuture;
use tokio_util::sync::CancellationToken;

fn should_end_run(_message: &AssistantMessage) -> bool {
    false
}

fn needs_another_response(_message: &AssistantMessage) -> bool {
    false
}

struct TurnHooks;

impl AgentHooks for TurnHooks {
    fn finish_turn<'a>(
        &'a self,
        turn: AgentTurnContext<'a>,
        _cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Option<AgentTurnDecision>> {
        Box::pin(async move {
            if turn.message.stop_reason.is_failure() {
                return None;
            }
            if should_end_run(turn.message) {
                return Some(AgentTurnDecision::End);
            }
            needs_another_response(turn.message).then_some(AgentTurnDecision::Continue)
        })
    }
}
```

Returning `None` preserves normal scheduling. `AgentTurnDecision::End` stops immediately after
`TurnEnd`, before polling steering or follow-up queues or preparing another request. On a normal
response, `AgentTurnDecision::Continue` ensures one next provider request. If tool results,
steering or a follow-up already cause that request, they satisfy the decision and no additional
request is made; otherwise the loop makes one context-only request. Error and aborted responses
remain hard exits, so their decisions are ignored. `finish_turn` runs again after the next request,
so returning `Continue` unconditionally creates an endless loop.

Upstream's removed `shouldStopAfterTurn` was never ported. To stop after a turn, return
`Some(AgentTurnDecision::End)`, and check `turn.message.stop_reason.is_failure()` first if the
check should only see normal responses.

Each provider turn follows this lifecycle:

```text
selected input events
→ prepare_request
→ provider response
→ tool results
→ finish_turn
→ TurnEnd
→ existing continuation scheduling or AgentEnd
```

When the loop continues after `TurnEnd`, the `prepare_next_turn` hook runs before the next
`TurnStart`. It returns an `AgentLoopTurnUpdate`, which can also add messages to send with the next
request.

### continue() and queued input

`continue` is a Rust keyword, so the method is `continue_run()`. It keeps upstream's queue
behavior. Empty and system-only transcripts return `AgentError::NoMessages` without consuming
queues. A non-assistant tail continues from the existing context: steering is polled at startup,
while follow-up input waits until the response naturally stops.

```rust
use delta_agent::{Agent, AgentError};
use delta_ai::UserMessage;

async fn retry(agent: &Agent) -> Result<(), AgentError> {
    agent.follow_up(UserMessage::new("After the retry"));
    // The first request retries the existing user or tool result tail.
    agent.continue_run().await
}
```

An assistant tail cannot be sent directly, so `continue_run()` falls back to one queued steering
batch, then one queued follow-up batch, and returns `AgentError::AssistantTail` when both queues
are empty. The queue mode still controls whether the selected batch contains one message or all
messages:

```rust
use delta_agent::{Agent, AgentError};
use delta_ai::UserMessage;

async fn resume(agent: &Agent) -> Result<(), AgentError> {
    agent.steer(UserMessage::new("Continue from here"));
    // Uses the queued message only because the tail is an assistant message.
    agent.continue_run().await
}
```

### Event Types

| Event | Description |
|-------|-------------|
| `AgentStart` | Agent begins processing |
| `AgentEnd { messages }` | Final event for the run. Awaited listeners for this event still count toward settlement |
| `TurnStart` | New turn begins (one LLM call + tool executions) |
| `TurnEnd { message, tool_results }` | Turn completes with the assistant message and tool results |
| `MessageStart { message }` | Any message begins (system, user, assistant, tool result, custom) |
| `MessageUpdate { message, assistant_message_event }` | **Assistant only.** `assistant_message_event` is the provider event, e.g. `TextDelta`; `message` is an `Arc<AssistantMessage>` snapshot of the response so far |
| `MessageEnd { message }` | Message completes |
| `ToolExecutionStart { tool_call_id, tool_name, args }` | Tool begins |
| `ToolExecutionUpdate { .., partial_result }` | Tool streams progress |
| `ToolExecutionEnd { .., result, is_error, duration_ms }` | Tool completes. `duration_ms` is `None` when the tool did not run |

`Agent::subscribe()` listeners are awaited in registration order. `AgentEnd` means no more loop
events will be emitted, but `agent.wait_for_idle().await` and `agent.prompt(..).await` only return
after the awaited `AgentEnd` listeners finish.

## Agent Options

```rust
use std::sync::Arc;
use std::time::Duration;

use delta_agent::{
    AfterToolCallContext, AfterToolCallResult, AgentHooks, AgentInitialState, AgentMessage, AgentOptions,
    BeforeToolCallContext, BeforeToolCallResult, QueueMode, ToolExecutionMode, models_stream_fn,
};
use delta_ai::options::{ThinkingBudgets, Transport};
use delta_ai::{Message, Model, ModelThinkingLevel, Models};
use futures::future::BoxFuture;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

fn options(models: Models, model: Model) -> AgentOptions {
    AgentOptions {
        // Initial state. `system_prompt` and `tools` become the leading system message
        // unless `messages` already starts with one.
        initial_state: AgentInitialState {
            system_prompt: Some("You are a helpful assistant.".into()),
            model: Some(model),
            thinking_level: ModelThinkingLevel::Medium,
            tools: Vec::new(),
            messages: Vec::new(),
        },

        // Required stream function. It receives a `TranscriptContext`: the prompt and tools
        // are in the transcript's system messages.
        stream_fn: models_stream_fn(models),

        // `convert_to_llm`, `transform_context`, `get_api_key`, `before_tool_call`,
        // `after_tool_call`, `prepare_request`, `finish_turn` and `prepare_next_turn`.
        hooks: Arc::new(AppHooks),

        // Steering mode: `OneAtATime` (default) or `All`.
        steering_mode: QueueMode::OneAtATime,

        // Follow-up mode: `OneAtATime` (default) or `All`.
        follow_up_mode: QueueMode::OneAtATime,

        // Session ID for provider caching.
        session_id: Some("session-123".into()),

        // Custom thinking budgets for token-based providers.
        thinking_budgets: Some(ThinkingBudgets {
            minimal: Some(128),
            low: Some(512),
            medium: Some(1024),
            high: Some(2048),
        }),

        // Tool execution mode: `Parallel` (default) or `Sequential`.
        tool_execution: ToolExecutionMode::Parallel,

        // Passed to every request in `SimpleStreamOptions`.
        transport: Transport::Auto,
        max_retry_delay: Some(Duration::from_secs(60)),
        on_payload: None,
        on_response: None,
        on_provider_stream_event: None,
    }
}

struct AppHooks;

impl AgentHooks for AppHooks {
    // Convert agent messages to LLM messages. Needed for custom message types; the default
    // keeps the LLM messages and drops custom ones.
    fn convert_to_llm<'a>(&'a self, messages: &'a [AgentMessage]) -> BoxFuture<'a, Vec<Message>> {
        Box::pin(async move { messages.iter().filter_map(|m| m.as_llm().cloned()).collect() })
    }

    // Transform the transcript before `convert_to_llm` (pruning, compaction).
    fn transform_context<'a>(
        &'a self,
        messages: Vec<AgentMessage>,
        _cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Vec<AgentMessage>> {
        Box::pin(async move { prune_old_messages(messages) })
    }

    // Dynamic API key resolution (for expiring OAuth tokens). `None` keeps the configured key.
    fn get_api_key<'a>(&'a self, _provider: &'a str) -> BoxFuture<'a, Option<String>> {
        Box::pin(refresh_token())
    }

    // Preflight each tool call after its arguments are validated. Can block execution.
    fn before_tool_call<'a>(
        &'a self,
        context: BeforeToolCallContext<'a>,
        _cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Option<BeforeToolCallResult>> {
        Box::pin(async move {
            (context.tool_call.name == "bash").then(|| BeforeToolCallResult {
                block: true,
                reason: Some("bash is disabled".into()),
                terminate: true,
            })
        })
    }

    // Postprocess each tool result before the final tool events are emitted.
    fn after_tool_call<'a>(
        &'a self,
        context: AfterToolCallContext<'a>,
        _cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Option<AfterToolCallResult>> {
        Box::pin(async move {
            if context.is_error {
                return None;
            }
            if context.tool_call.name == "notify_done" {
                return Some(AfterToolCallResult {
                    terminate: Some(true),
                    ..Default::default()
                });
            }
            let mut details = context.result.details.clone().unwrap_or_else(|| json!({}));
            if let Value::Object(fields) = &mut details {
                fields.insert("audited".into(), Value::Bool(true));
            }
            Some(AfterToolCallResult {
                details: Some(details),
                ..Default::default()
            })
        })
    }

    // `prepare_request` and `finish_turn`: see "Request preparation and turn finalization".
}

fn prune_old_messages(messages: Vec<AgentMessage>) -> Vec<AgentMessage> {
    messages
}

async fn refresh_token() -> Option<String> {
    None
}
```

Most applications only set a few fields and take the rest from `AgentOptions::new(stream_fn)`
with `..AgentOptions::new(stream_fn)`, as in the [Quick Start](#quick-start). `AgentHooks` methods
receive a borrowed context struct and the run's `CancellationToken`, and return a boxed future.
Hooks do not return `Result`.

## Agent State

```rust
use std::collections::HashSet;
use std::sync::Arc;

use delta_agent::{AgentMessage, AgentTool, NoCustom, StreamingMessage};
use delta_ai::{Model, ModelThinkingLevel};

pub struct AgentState<C = NoCustom> {
    pub model: Model,
    pub thinking_level: ModelThinkingLevel,
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub messages: Vec<AgentMessage<C>>,
    pub is_streaming: bool,
    pub streaming_message: Option<StreamingMessage<C>>,
    pub pending_tool_calls: HashSet<String>,
    pub error_message: Option<String>,
}
```

`Agent` is a handle: clones share the same agent, so one task can `prompt` while another steers,
aborts or reads state. The state sits behind a lock. Read it with `agent.with_state(|state| ..)`
and change it with `agent.with_state_mut(|state| ..)`; the closures keep lock guards out of your
code. Common fields also have methods: `messages()` (a copy of the transcript), `set_messages()`,
`model()`, `set_model()`, `set_thinking_level()`, `set_tools()`, `is_streaming()`,
`error_message()`.

```rust
use delta_agent::Agent;

fn inspect(agent: &Agent) {
    let (count, pending) = agent.with_state(|state| (state.messages.len(), state.pending_tool_calls.len()));
    println!("{count} messages, {pending} tool calls running");
}
```

`is_streaming`, `streaming_message` and `pending_tool_calls` are run state. The next run event
overwrites changes you make to them.

The transcript owns the system prompt and tool declarations: the leading system message is the
prompt, and later system messages patch it (see `SystemMessage` in `delta_ai`).
`agent.system_prompt()` replays the transcript; there is no separate field. `tools` is the
executable loadout. Before every request the loop compares it with the tools the transcript
declares. If they differ, it announces the change in a system message (merged into a pending system
message when one exists). The function that does this, `declare_tool_changes`, is public.
`delta_ai::transcript::get_current_system_message(&messages)` returns the replayed head, including
declared tools, for a slice of LLM messages. For an agent transcript with custom messages, pass the
LLM messages (`AgentMessage::as_llm`).

To change the prompt mid-conversation, append a system message with `content` (added
instructions) or `sections` (named replacements):

```rust,no_run
use delta_agent::{Agent, AgentError};
use delta_ai::types::now_ms;
use delta_ai::{SystemMessage, UserMessage};

async fn add_skills(agent: &Agent) -> Result<(), AgentError> {
    let skills = SystemMessage {
        sections: Some(
            [("skills".to_owned(), Some("<skills>...</skills>".to_owned()))]
                .into_iter()
                .collect(),
        ),
        timestamp: now_ms(),
        ..SystemMessage::default()
    };
    agent
        .prompt(vec![skills.into(), UserMessage::new("Continue").into()])
        .await
}
```

During streaming, `streaming_message` holds the current message: `StreamingMessage::Message` from
`MessageStart` to `MessageEnd`, and `StreamingMessage::Assistant` with the `Arc` snapshot of the
last `MessageUpdate` while the assistant response streams.

`is_streaming` remains `true` until the run fully settles, including awaited `AgentEnd`
listeners.

## Methods

### Prompting

```rust,no_run
use delta_agent::{Agent, AgentError};
use delta_ai::{ImageContent, UserMessage};

async fn prompting(agent: &Agent, base64_data: String) -> Result<(), AgentError> {
    // Text prompt
    agent.prompt_text("Hello", Vec::new()).await?;

    // With images
    let image = ImageContent {
        data: base64_data,
        mime_type: "image/jpeg".into(),
    };
    agent.prompt_text("What's in this image?", vec![image]).await?;

    // Messages directly
    agent.prompt(vec![UserMessage::new("Hello").into()]).await?;

    // Continue existing non-assistant input; an assistant tail may use queued input as fallback
    agent.continue_run().await?;
    Ok(())
}
```

`prompt` takes a `Vec<AgentMessage<C>>`. Upstream overloads `prompt` for a string, a message and
an array; delta has `prompt_text` for text and images. While a run is active, `prompt` returns
`AgentError::AlreadyRunning` and `continue_run` returns `AgentError::ContinueWhileRunning`. Use
`steer` or `follow_up` to queue messages instead.

### State Management

```rust
use std::sync::Arc;

use delta_agent::{Agent, AgentError, AgentMessage, AgentTool, DefaultHooks, ToolExecutionMode};
use delta_ai::{ModelThinkingLevel, Models};

fn manage(
    agent: &Agent,
    models: &Models,
    my_tool: Arc<dyn AgentTool>,
    new_messages: Vec<AgentMessage>,
    message: AgentMessage,
) -> Result<(), AgentError> {
    if let Some(model) = models.get_model("openai", "gpt-4o") {
        agent.set_model(model);
    }
    agent.set_thinking_level(ModelThinkingLevel::Medium);
    agent.set_tools(vec![my_tool]);
    agent.set_tool_execution(ToolExecutionMode::Sequential);
    // Replaces all hooks: before_tool_call, after_tool_call, prepare_request, finish_turn, ...
    agent.set_hooks(Arc::new(DefaultHooks));
    agent.set_messages(new_messages);
    agent.with_state_mut(|state| state.messages.push(message));
    // Respects the queue modes; does not consume.
    let next_queued_messages = agent.peek_queued_messages();
    println!("{} queued", next_queued_messages.len());
    agent.reset()
}
```

`set_hooks` replaces the whole `AgentHooks` object; upstream assigns each callback separately.
`set_stream_fn` replaces the stream function. `reset()` clears the transcript and queues, keeps the
replayed system message (prompt and tool baseline), and returns `AgentError::ResetWhileRunning`
during a run.

### Session and Thinking Budgets

```rust
use delta_agent::Agent;
use delta_ai::options::ThinkingBudgets;

fn configure(agent: &Agent) {
    agent.set_session_id(Some("session-123".into()));

    agent.set_thinking_budgets(Some(ThinkingBudgets {
        minimal: Some(128),
        low: Some(512),
        medium: Some(1024),
        high: Some(2048),
    }));
}
```

### Control

```rust,no_run
use delta_agent::Agent;

async fn control(agent: &Agent) {
    agent.abort(); // Cancel the current run
    agent.wait_for_idle().await; // Wait for completion
}
```

`abort()` cancels the run's `CancellationToken`. `cancel_token()` returns that token while a run is
active. Dropping the future returned by `prompt` or `continue_run` also cancels the run; in that
case no `AgentEnd` is emitted.

### Events

```rust
use delta_agent::{Agent, AgentEvent};
use tokio_util::sync::CancellationToken;

async fn flush_session_state(_cancel: &CancellationToken) {}

fn listen(agent: &Agent) {
    let subscription = agent.subscribe(|event, cancel| async move {
        if let AgentEvent::AgentEnd { .. } = &*event {
            // Final barrier work for the run
            flush_session_state(&cancel).await;
        }
    });
    agent.unsubscribe(subscription);
}
```

A listener receives the event as `Arc<AgentEvent<C>>` and the run's `CancellationToken`.
`subscribe` returns a `SubscriptionId` for `unsubscribe`; upstream returns an unsubscribe function.

## Steering and Follow-up

Steering messages let you interrupt the agent while tools are running. Follow-up messages let you
queue work after the agent would otherwise stop.

```rust
use delta_agent::{Agent, QueueMode};
use delta_ai::UserMessage;

fn queue(agent: &Agent) {
    agent.set_steering_mode(QueueMode::OneAtATime);
    agent.set_follow_up_mode(QueueMode::OneAtATime);

    // While the agent is running tools
    agent.steer(UserMessage::new("Stop! Do this instead."));

    // After the agent finishes its current work
    agent.follow_up(UserMessage::new("Also summarize the result."));

    let steering_mode = agent.steering_mode();
    let follow_up_mode = agent.follow_up_mode();
    println!("{steering_mode:?} {follow_up_mode:?}");

    agent.clear_steering_queue();
    agent.clear_follow_up_queue();
    agent.clear_all_queues();
}
```

Use `clear_steering_queue`, `clear_follow_up_queue` or `clear_all_queues` to drop queued messages.
`has_queued_messages` reports whether either queue holds a message.

When steering messages are detected after a turn completes:
1. All tool calls from the current assistant message have already finished
2. Steering messages are injected
3. The LLM responds on the next turn

Follow-up messages are checked only when there are no more tool calls and no steering messages.
If any are queued, they are injected and another turn runs.

Steering messages queued before a run starts are injected before its first request.

## Custom Message Types

Upstream extends `AgentMessage` through TypeScript declaration merging. In delta, define a type
that implements `CustomMessage` and pass it as the type parameter:

```rust
use delta_agent::{AgentMessage, CustomMessage};
use delta_ai::types::now_ms;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
enum AppMessage {
    Notification { text: String, timestamp: i64 },
}

impl CustomMessage for AppMessage {
    fn role(&self) -> &str {
        match self {
            Self::Notification { .. } => "notification",
        }
    }

    fn timestamp(&self) -> i64 {
        match self {
            Self::Notification { timestamp, .. } => *timestamp,
        }
    }
}

// Now valid
let msg: AgentMessage<AppMessage> = AgentMessage::Custom(AppMessage::Notification {
    text: "Info".into(),
    timestamp: now_ms(),
});
assert_eq!(msg.role(), "notification");
```

The role must not be one of the LLM roles. With `#[serde(tag = "role")]` the message serializes
as `{"role":"notification","text":"Info","timestamp":...}`, the same JSON as upstream.

Handle custom types in `convert_to_llm`. The default implementation drops them. This one shows
them to the model as user text instead:

```rust
use std::sync::Arc;

use delta_agent::{Agent, AgentHooks, AgentMessage, AgentOptions, CustomMessage, StreamFn};
use delta_ai::{Message, UserMessage};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
enum AppMessage {
    Notification { text: String, timestamp: i64 },
}

impl CustomMessage for AppMessage {
    fn role(&self) -> &str {
        "notification"
    }

    fn timestamp(&self) -> i64 {
        match self {
            Self::Notification { timestamp, .. } => *timestamp,
        }
    }
}

struct AppHooks;

impl AgentHooks<AppMessage> for AppHooks {
    fn convert_to_llm<'a>(&'a self, messages: &'a [AgentMessage<AppMessage>]) -> BoxFuture<'a, Vec<Message>> {
        Box::pin(async move {
            messages
                .iter()
                .map(|message| match message {
                    AgentMessage::Llm(message) => message.clone(),
                    AgentMessage::Custom(AppMessage::Notification { text, timestamp }) => Message::User(UserMessage {
                        content: format!("Notification: {text}").into(),
                        timestamp: *timestamp,
                    }),
                })
                .collect()
        })
    }
}

fn app_agent(stream_fn: StreamFn) -> Agent<AppMessage> {
    Agent::new(AgentOptions {
        hooks: Arc::new(AppHooks),
        ..AgentOptions::new(stream_fn)
    })
}
```

## Tools

Define tools by implementing the `AgentTool` trait:

```rust
use std::sync::Arc;

use delta_agent::{Agent, AgentTool, AgentToolResult, ToolContext, ToolError, ToolExecutionMode};
use delta_ai::Tool;
use futures::future::BoxFuture;
use serde_json::{Value, json};

struct ReadFileTool {
    definition: Tool,
}

impl ReadFileTool {
    fn new() -> Self {
        let parameters = json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path" },
            },
            "required": ["path"],
        });
        Self {
            definition: Tool::new("read_file", "Read a file's contents", parameters),
        }
    }
}

impl AgentTool for ReadFileTool {
    fn definition(&self) -> &Tool {
        &self.definition
    }

    // For UI display. Default: the tool name.
    fn label(&self) -> &str {
        "Read File"
    }

    // Override the execution mode for this tool (optional).
    // `Sequential` forces the entire batch to run one at a time.
    // `Parallel` allows concurrent execution with other tool calls.
    // If `None` (the default), the agent's `tool_execution` setting applies.
    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        Some(ToolExecutionMode::Sequential)
    }

    fn execute(&self, arguments: Value, context: ToolContext) -> BoxFuture<'_, Result<AgentToolResult, ToolError>> {
        Box::pin(async move {
            let path = arguments["path"].as_str().unwrap_or_default().to_owned();

            // Optional: stream progress
            context.updates.send(AgentToolResult::text("Reading..."));

            let file = path.clone();
            let content = tokio::task::spawn_blocking(move || std::fs::read_to_string(file)).await??;
            let size = content.len();

            // Optional: set `terminate: true` here to skip the automatic follow-up LLM call
            // when every finalized tool result in the batch does the same.
            Ok(AgentToolResult {
                details: Some(json!({ "path": path, "size": size })),
                ..AgentToolResult::text(content)
            })
        })
    }
}

fn install(agent: &Agent) {
    agent.set_tools(vec![Arc::new(ReadFileTool::new())]);
}
```

The arguments are validated against `parameters` before `execute` runs. `ToolContext` holds the
tool call ID, the run's `CancellationToken` (upstream's `signal`) and `updates` (upstream's
`onUpdate`). `AgentTool` also has optional `output_schema()`, `replay()` and
`prepare_arguments()` methods, which correspond to upstream's fields of the same names.

Two shorter ways to build a tool:

- `FnTool::new(definition, closure)` wraps a `Tool` declaration and an async closure.
  `with_label` and `with_execution_mode` set the optional fields.
- `typed_tool(name, description, closure)` (feature `schemars`) generates the parameter schema
  from a Rust type and deserializes the validated arguments into it. Upstream builds schemas with
  TypeBox at runtime.

```rust
use delta_agent::{AgentToolResult, FnTool, ToolExecutionMode};
use delta_ai::Tool;
use serde_json::json;

fn echo_tool() -> FnTool {
    let parameters = json!({
        "type": "object",
        "properties": { "text": { "type": "string" } },
        "required": ["text"],
    });
    FnTool::new(
        Tool::new("echo", "Echo the input", parameters),
        |args, _ctx| async move { Ok(AgentToolResult::text(args["text"].as_str().unwrap_or_default())) },
    )
    .with_label("Echo")
    .with_execution_mode(ToolExecutionMode::Parallel)
}

// Requires the `schemars` feature (on by default).
#[cfg(feature = "schemars")]
fn get_time_tool() -> FnTool {
    use delta_agent::typed_tool;

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct GetTime {
        /// IANA timezone name.
        timezone: Option<String>,
    }

    typed_tool("get_time", "Get the current time", |args: GetTime, _ctx| async move {
        Ok(AgentToolResult::text(format!(
            "12:00 in {}",
            args.timezone.unwrap_or_else(|| "UTC".into())
        )))
    })
}
```

### Error Handling

**Return `Err`** when a tool fails. Do not return error messages as content.

```rust
use delta_agent::{AgentToolResult, FnTool, ToolError};
use delta_ai::Tool;
use serde_json::json;

fn checked_read_file() -> FnTool {
    let parameters = json!({
        "type": "object",
        "properties": { "path": { "type": "string" } },
        "required": ["path"],
    });
    FnTool::new(
        Tool::new("read_file", "Read a file's contents", parameters),
        |args, _ctx| async move {
            let path = args["path"].as_str().unwrap_or_default();
            if !std::path::Path::new(path).exists() {
                return Err(ToolError::from(format!("File not found: {path}")));
            }
            // Return content only on success
            Ok(AgentToolResult::text("..."))
        },
    )
}
```

The agent records the error's message as the tool result and reports it to the LLM with
`is_error: true`. To keep `details` or `structured_content` on a failure, return a result with
`is_error: true` instead of `Err`.

Set `terminate: true` on the result of `execute()`, on a blocked `before_tool_call` result, or in
an `after_tool_call` override to hint that the agent should stop after the current tool batch. This
only takes effect when every finalized tool result in the batch is terminating. The hint is
runtime-only; emitted tool result transcript messages remain standard LLM tool results.

### MCP and Codemode

Not ported: there are no Rust counterparts of `pi-mcp`, `pi-codemode` or `examples/mcp-codemode`
yet; a tool that calls other tools can use `run_tool_call`, which runs a call through the same
validation, `before_tool_call` and `after_tool_call` steps as a model-issued call.

## Proxy Usage

For apps that send model requests through their own server, which holds the provider credentials:

```rust
use std::sync::Arc;

use delta_agent::{Agent, AgentOptions, ProxyStreamOptions, StreamFn, stream_proxy};

let stream_fn: StreamFn = Arc::new(|model, context, options| {
    Box::pin(async move {
        stream_proxy(
            &model,
            &context,
            ProxyStreamOptions {
                options,
                auth_token: "...".into(),
                proxy_url: "https://your-server.com".into(),
            },
        )
    })
});

let agent: Agent = Agent::new(AgentOptions::new(stream_fn));
```

`stream_proxy` sends `POST {proxy_url}/api/stream` with a bearer token and reads the server's SSE
events (`ProxyAssistantMessageEvent`). The wire format matches upstream's; see
[`docs/design/proxy.md`](docs/design/proxy.md).

## Low-Level API

For direct control without `Agent`:

```rust,no_run
use std::sync::Arc;

use delta_agent::{
    AgentContext, AgentEvent, AgentLoopConfig, AgentLoopError, AgentMessage, DefaultHooks, StreamFn, ToolExecutionMode,
    agent_loop, agent_loop_continue,
};
use delta_ai::types::now_ms;
use delta_ai::{Model, SystemMessage, UserMessage};
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

async fn run(model: Model, stream_fn: StreamFn) -> Result<(), AgentLoopError> {
    let system = SystemMessage {
        content: "You are helpful.".into(),
        timestamp: now_ms(),
        ..SystemMessage::default()
    };
    let mut context: AgentContext = AgentContext {
        messages: vec![system.into()],
        tools: Vec::new(),
    };

    let config = AgentLoopConfig {
        // Overridden by a tool's `execution_mode()` if set.
        tool_execution: ToolExecutionMode::Parallel,
        // `convert_to_llm`, `before_tool_call`, `after_tool_call`, ...
        hooks: Arc::new(DefaultHooks),
        ..AgentLoopConfig::new(model)
    };

    let user_message: AgentMessage = UserMessage::new("Hello").into();

    let mut events = agent_loop(
        vec![user_message],
        context.clone(),
        config.clone(),
        CancellationToken::new(),
        stream_fn.clone(),
    );
    while let Some(event) = events.next().await {
        println!("{}", event.kind());
        if let AgentEvent::AgentEnd { messages } = event {
            context.messages.extend(messages);
        }
    }

    // Continue from existing context. The last message must not be an assistant message,
    // e.g. retry after a failed request.
    let last_role = context.messages.last().map(AgentMessage::role);
    if last_role.is_some_and(|role| role != "assistant") {
        let mut events = agent_loop_continue(context, config, CancellationToken::new(), stream_fn)?;
        while let Some(event) = events.next().await {
            println!("{}", event.kind());
        }
    }
    Ok(())
}
```

`agent_loop` and `agent_loop_continue` run the loop on a Tokio task and return an
`AgentEventStream`. `AgentEventStream::result()` drains the events and returns the messages the
run added. Dropping the stream before the run ends cancels the run (a child of your token, so your
token stays uncancelled). `agent_loop_continue` returns `AgentLoopError` when the context is empty
or ends with an assistant message.

These streams are observational. They preserve event order, but they do not wait for your event
handling to settle before later producer phases continue. If you need message processing to act as
a barrier before tool preflight, use `Agent`, or call `run_agent_loop` / `run_agent_loop_continue`
with your own `AgentEventSink`. The loop awaits each `emit` call:

```rust,no_run
use delta_agent::{AgentContext, AgentEvent, AgentEventSink, AgentLoopConfig, StreamFn, run_agent_loop};
use delta_ai::UserMessage;
use futures::future::BoxFuture;
use tokio_util::sync::CancellationToken;

struct PrintSink;

impl AgentEventSink<delta_agent::NoCustom> for PrintSink {
    fn emit(&self, event: AgentEvent) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            // The loop waits here before it continues.
            println!("{}", event.kind());
        })
    }
}

async fn run(context: AgentContext, config: AgentLoopConfig, stream_fn: StreamFn) {
    let prompts = vec![UserMessage::new("Hello").into()];
    let new_messages = run_agent_loop(
        prompts,
        context,
        config,
        &PrintSink,
        CancellationToken::new(),
        &stream_fn,
    )
    .await;
    println!("{} new messages", new_messages.len());
}
```

## License

MIT
