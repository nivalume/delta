# `Agent` (`agent`)

Corresponds to upstream `src/agent.ts`. `Agent<C>` holds state and wraps the low-level loop in an object that can `prompt` many times.

## Handle

`Agent<C>` is an `Arc<Inner>` inside; `clone()` gives a handle that shares the same agent. While one task runs `await agent.prompt(..)`, another task can `steer`, `abort` or read state. Upstream is a single-threaded JS object and does not need this layer.

Internally, a `std::sync::Mutex` protects the state, settings, listeners and the two queues. Locks are held only briefly in synchronous code and never across `await`.

## Construction

```rust
let agent: Agent = Agent::new(AgentOptions {
    initial_state: AgentInitialState {
        system_prompt: Some("You are helpful.".into()),
        model: Some(model),
        tools: vec![Arc::new(tool)],
        ..Default::default()
    },
    hooks: Arc::new(MyHooks),
    ..AgentOptions::new(models_stream_fn(models))
});
```

- `AgentOptions::new(stream_fn)` gives defaults for the other fields: `DefaultHooks`, `OneAtATime` for both queues, transport `Auto`, parallel tool execution.
- `AgentOptions` also has `session_id`, `thinking_budgets`, `transport`, `max_retry_delay`, `on_payload`, `on_response` and `on_provider_stream_event`, which go into `SimpleStreamOptions` on every run.
- `AgentInitialState`: when `messages` does not start with a system message, `system_prompt` and `tools` form the leading system message (same as upstream: the system prompt and tool declarations live in the transcript). Without a `model`, the placeholder model `unknown` is used.

## State

```rust
pub struct AgentState<C> {
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

- Read: `with_state(|state| ..)`; write: `with_state_mut(|state| ..)`. Common fields have getters/setters (`messages()`, `set_model()`, `set_tools()`, `system_prompt()` and so on).
- `system_prompt()` is obtained by replaying the system messages in the transcript; there is no separate field.
- `set_tools()` changes only the executable tools. The loop writes the difference from what the transcript declares as a system message before the next request (see [loop.md](loop.md#tool-declaration-changes-declare_tool_changes)).
- `streaming_message` is `StreamingMessage::Message(AgentMessage<C>)` (between `MessageStart` and `MessageEnd`) or `StreamingMessage::Assistant(Arc<AssistantMessage>)` (the snapshot from the latest `MessageUpdate`). Upstream types this field as `AgentMessage`. Rust splits it into two variants so the `Arc` from `MessageUpdate` can be stored directly without another clone.

### State reduction

`Agent` implements `AgentEventSink`. Each event is first reduced into state, then passed to listeners, same as upstream `processEvents`:

| Event | State change |
| --- | --- |
| `MessageStart` | `streaming_message = Message(msg)` |
| `MessageUpdate` | `streaming_message = Assistant(arc)` |
| `MessageEnd` | Clear `streaming_message`, append the message to `messages` |
| `ToolExecutionStart` / `End` | Add to / remove from `pending_tool_calls` |
| `TurnEnd` | If the assistant message has `error_message`, write it to `state.error_message` |
| `AgentEnd` | Clear `streaming_message` |

## Listeners

```rust
pub type AgentListener<C> =
    Arc<dyn Fn(Arc<AgentEvent<C>>, CancellationToken) -> BoxFuture<'static, ()> + Send + Sync>;
```

- `subscribe(listener)` returns a `SubscriptionId`; `unsubscribe(id)` removes the subscription (upstream `subscribe` returns an unsubscribe function).
- Listeners are awaited one by one in subscription order. Events are shared as `Arc` and not cloned per listener.
- Listener execution counts as part of the run: the agent becomes idle only after all listeners for `AgentEnd` have returned.

## Running

| Method | Upstream | Notes |
| --- | --- | --- |
| `prompt(messages)` | `prompt(messages)` | Start a run with a batch of messages |
| `prompt_text(text, images)` | `prompt(text, images)` | Text plus optional images form one user message |
| `continue_run()` | `continue()` | If the last message is not assistant, retry. If it is assistant, try one batch of steering messages, then one batch of follow-up messages; if there are none, `AssistantTail` |
| `abort()` | `abort()` | Cancel the current run |
| `wait_for_idle()` | `waitForIdle()` | Returns after the current run and its listeners have finished |
| `reset()` | `reset()` | Clear the conversation and queues, keeping the system message obtained by replay (prompt and tool baseline). Returns `ResetWhileRunning` if called during a run |

`continue` is a Rust keyword, so the method is named `continue_run`.

When a run is already active, `prompt` returns `AgentError::AlreadyRunning` and `continue_run` returns `ContinueWhileRunning`. The error text is the same as upstream.

At the start of a run: set `is_streaming = true`, clear `streaming_message` and `error_message`, and create the run's `CancellationToken`. At the end of a run, a drop guard cleans up (`is_streaming = false`, clear `pending_tool_calls`, notify idle). If the future returned by `prompt` is dropped before the run ends, the guard also cleans up state and cancels the run token, so tools still holding the token stop. In this case `AgentEnd` is not emitted.

`wait_for_idle` is implemented with `tokio::sync::watch`, so several tasks can wait at the same time.

## Queues

- `steer(msg)`: injected after the current assistant reply (and its tool calls) ends.
- `follow_up(msg)`: injected only when the agent would otherwise stop.
- The mode is set by `set_steering_mode` / `set_follow_up_mode`: `OneAtATime` takes one message at a time, `All` takes all at once.
- `clear_steering_queue`, `clear_follow_up_queue`, `clear_all_queues`, `has_queued_messages`, `peek_queued_messages` (the messages the next turn will take, steering first, without consuming them).

Steering messages queued before a run starts are injected before the first request. When `continue_run` restarts a run with a steering batch, it skips the initial steering read, so `OneAtATime` mode does not inject two messages at once.

## Differences from upstream

- A cloneable shared handle with internal locking.
- State is read and written through closures; `streaming_message` is split into two variants.
- `subscribe` returns a `SubscriptionId`.
- Errors are returned as `Result<(), AgentError>`. Upstream `runWithLifecycle` catches exceptions thrown by the loop and synthesizes a failed assistant message. The Rust loop never ends with an error (stream failures are encoded in the stream, hooks do not return `Result`), so this path does not exist.
- Dropping the `prompt` future cancels the run. Upstream's promise cannot be cancelled; only `abort()` works.
