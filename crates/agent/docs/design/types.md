# Messages, events and tool results (`types`)

Corresponds to upstream `src/types.ts`.

## `AgentMessage<C>` and custom messages

Besides LLM messages, an application can put its own messages in the transcript (for example UI notifications or compaction summaries). Upstream extends the `AgentMessage` union type with TypeScript declaration merging; delta uses a type parameter:

```rust
pub trait CustomMessage: Clone + Debug + Send + Sync + Serialize + DeserializeOwned + 'static {
    fn role(&self) -> &str;      // must not reuse an LLM message role
    fn timestamp(&self) -> i64;  // milliseconds
}

#[serde(untagged)]
pub enum AgentMessage<C = NoCustom> {
    Llm(Message),
    Custom(C),
}
```

- `NoCustom` is an empty enum and the default parameter. Applications without custom messages just write `Agent` and `AgentMessage`.
- `untagged` serialization: the `Llm` variant serializes as the inner `Message`, so a transcript without custom messages is exactly the same JSON as upstream. Deserialization tries `Message` first (distinguished by `role`), then `C`.
- Custom messages are not sent to the model. The default `AgentHooks::convert_to_llm` drops them; applications that need to turn them into user messages or similar implement it themselves.
- `role()`, `timestamp()`, `as_llm()`, `as_assistant()` and `is_system()` are the common accessors. `From<Message>`, `From<AssistantMessage>`, `From<UserMessage>` and others are used for construction.

## Tool results

```rust
pub struct AgentToolResult {
    pub content: Vec<UserBlock>,              // text or images sent to the model
    pub details: Option<Value>,               // for logs and UI, not sent to the model
    pub structured_content: Option<Value>,    // structured result matching the tool's output schema, not sent to the model
    pub usage: Option<Usage>,                 // the tool's own usage (for example a sub-agent)
    pub is_error: bool,
    pub terminate: bool,
}
```

The JSON shape matches upstream (camelCase; `false`/`None` fields are omitted).

- `is_error`: a tool can report failure directly instead of returning `Err`. This keeps `details` and `structured_content`.
- `terminate`: a hint to stop after the current batch of tool calls. It takes effect only when **every** final result in the batch sets it, same as upstream.
- `AgentToolResult::text(..)` builds a text result. `AgentToolResult::error(..)` builds the result the loop records on failure (text + an empty `details` object, same as upstream).

`AgentToolCallOutcome` is the final result after hooks have run. It carries `tool_call`, `is_error` and `duration_ms` (`None` when the tool did not run).

## Events

```rust
pub enum AgentEvent<C = NoCustom> {
    AgentStart,
    AgentEnd { messages: Vec<AgentMessage<C>> },
    TurnStart,
    TurnEnd { message: AgentMessage<C>, tool_results: Vec<ToolResultMessage> },
    MessageStart { message: AgentMessage<C> },
    MessageUpdate { message: Arc<AssistantMessage>, assistant_message_event: AssistantMessageEvent },
    MessageEnd { message: AgentMessage<C> },
    ToolExecutionStart { tool_call_id, tool_name, args },
    ToolExecutionUpdate { tool_call_id, tool_name, args, partial_result },
    ToolExecutionEnd { tool_call_id, tool_name, result, is_error, duration_ms },
}
```

`kind()` returns upstream's event type name (`agent_start`, `message_update` and so on), for logs and cross-language comparison.

Event order is the same as upstream:

- One run: `AgentStart`, then some turns, then `AgentEnd`. `AgentEnd.messages` holds the messages added in this run.
- One turn: `TurnStart`; `MessageStart`/`MessageEnd` for this turn's input messages; `MessageStart`, some `MessageUpdate`s and `MessageEnd` for the assistant reply; tool execution events; `MessageStart`/`MessageEnd` for the tool result messages; finally `TurnEnd`.
- System, user, tool result and custom messages only get `MessageStart`/`MessageEnd`. `MessageUpdate` is only for an assistant message that is being streamed.

### The `MessageUpdate` snapshot

Upstream's `message_update` event carries the partial object that is mutated in place during streaming. In Rust, listeners must be able to store events and pass them across tasks, so `message` is an `Arc<AssistantMessage>` snapshot: for every provider event the loop takes, it clones `AssistantMessageEventStream::partial()` once into an `Arc`. `Agent` stores the same `Arc` in `state.streaming_message`, so listeners get the same data with no second clone.

The cost is one message clone per delta. During streaming a message is usually a few KB to a few tens of KB, so this cost is negligible next to network latency.

### No `Serialize`

`AgentEvent` is only passed within the process and does not implement serialization. Upstream does not treat events as a persistence format either. Applications that pass events across processes (for example RPC mode) define their own wire format.

## Hook return values

| Type | Used by | Notes |
| --- | --- | --- |
| `BeforeToolCallResult { block, reason, terminate }` | `before_tool_call` | With `block`, the tool does not run and an error result is recorded; the default text is `Tool execution was blocked` |
| `AfterToolCallResult { content, details, structured_content, is_error, usage, terminate }` | `after_tool_call` | Overrides field by field; `None` keeps the original value; no deep merge. If `content` is replaced but `structured_content` is not given, the original structured result is dropped, because it may no longer match |
| `AgentTurnDecision::{Continue, End}` | `finish_turn` | `Continue` guarantees one more request (no extra turn is added when tool results or queued messages already trigger a request). `End` stops the run after `TurnEnd` without reading the queues |

## Other enums

- `ToolExecutionMode::{Sequential, Parallel}`, default `Parallel`, serialized in lowercase.
- `QueueMode::{All, OneAtATime}`, default `OneAtATime`, serialized as `all` / `one-at-a-time`.
