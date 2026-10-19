# Low-level loop and hooks (`agent_loop`, `hooks`)

Corresponds to upstream `src/agent-loop.ts`, `src/stream-fn.ts` and `AgentLoopConfig` in `types.ts`.

## Entry points

| Function | Upstream | Notes |
| --- | --- | --- |
| `run_agent_loop(prompts, context, config, sink, cancel, stream_fn)` | `runAgentLoop` | Adds `prompts` to the context and emits their events, then enters the loop. Returns the messages added in this run |
| `run_agent_loop_continue(context, config, sink, cancel, stream_fn)` | `runAgentLoopContinue` | Adds no new messages and continues from the current context (retry). The last message must not be assistant, otherwise `AgentLoopError` |
| `agent_loop(..)` / `agent_loop_continue(..)` | `agentLoop` / `agentLoopContinue` | Run the two functions above in a Tokio task and return an `AgentEventStream` |

The `run_*` versions hand events to an `AgentEventSink` and await every `emit`: the loop continues only after the sink has processed the event. `Agent` relies on this so that listeners run in order and count toward the end of the run (see [agent.md](agent.md)).

The `agent_loop` versions put events into an unbounded channel; the loop does not wait for the consumer. `AgentEventStream` implements `Stream<Item = AgentEvent<C>>`; `result()` drains the events and returns the added messages. Dropping this stream before the run ends cancels the run: the loop gets a child token of the caller's token, and drop cancels only the child token, not the caller's token.

## Config

```rust
pub struct AgentLoopConfig<C> {
    pub model: Model,
    pub stream_options: SimpleStreamOptions,   // reasoning is the thinking level; cancel and api_key are set by the loop for each request
    pub tool_execution: ToolExecutionMode,
    pub hooks: Arc<dyn AgentHooks<C>>,
    pub get_steering_messages: Option<MessageSource<C>>,
    pub get_follow_up_messages: Option<MessageSource<C>>,
}
```

Upstream `AgentLoopConfig` extends `SimpleStreamOptions` and puts all callbacks flat on the same object. delta puts the request options in the `stream_options` field and collects the callbacks in `hooks`. `MessageSource<C>` is a closure that returns `BoxFuture<Vec<AgentMessage<C>>>`, corresponding to upstream's `getSteeringMessages` / `getFollowUpMessages`.

## `StreamFn`

```rust
pub type StreamFn = Arc<
    dyn Fn(Model, TranscriptContext, SimpleStreamOptions) -> BoxFuture<'static, AssistantMessageEventStream> + Send + Sync,
>;
```

- The input is a normalized `TranscriptContext` (the system prompt and tool declarations are in system messages).
- Failures must be encoded in the returned stream (an `Error` event), not raised as a panic.
- `models_stream_fn(models)` adapts `delta_ai::Models::stream_simple` into a `StreamFn`; this is the normal usage. `stream_proxy` (see [proxy.md](proxy.md)) is another implementation. Tests construct one directly.
- Upstream has a global `setDefaultStreamFn`, used when the caller passes no `streamFn`. delta keeps no global state; `StreamFn` must be passed explicitly.

## `AgentHooks<C>`

Upstream's callbacks are merged into one trait. Every method has a default implementation, and an application implements only the methods it needs:

| Method | Upstream | When called | Default |
| --- | --- | --- | --- |
| `convert_to_llm(messages)` | `convertToLlm` | Before each request, after `transform_context` | Keep LLM messages, drop custom messages |
| `transform_context(messages, cancel)` | `transformContext` | Before each request, before `convert_to_llm` (trim, inject context) | Return unchanged |
| `get_api_key(provider)` | `getApiKey` | Before each request, for short-lived tokens | `None`: use the key configured in `stream_options` |
| `before_tool_call(ctx, cancel)` | `beforeToolCall` | After argument validation, before execution | Do not block |
| `after_tool_call(ctx, cancel)` | `afterToolCall` | After execution, before `ToolExecutionEnd` | Do not override |
| `finish_turn(turn, cancel)` | `finishTurn` | After this turn's assistant message and tool results are in the context, before `TurnEnd` | Do not intervene. The decision is ignored when the reply failed |
| `prepare_request(request, cancel)` | `prepareRequest` | Before each request (including the first), after pending messages are added | Do not modify. Does not read the queues |
| `prepare_next_turn(turn, cancel)` | `prepareNextTurn` | When the loop continues, after `TurnEnd` and before the next `TurnStart` | Do not modify |

Hook inputs are borrowed context structs (`BeforeToolCallContext<'a, C>`, `AgentTurnContext<'a, C>` and so on). Method signatures look like `fn(&'a self, ctx: Ctx<'a, C>, cancel: &'a CancellationToken) -> BoxFuture<'a, Option<R>>`. Writing these eight callbacks as separate `Arc<dyn Fn>` would need a higher-ranked lifetime bound on each; trait methods are easier to write and to implement. `DefaultHooks` is an empty struct that uses all the default implementations.

`prepare_request` returns `AgentRequestUpdate { context, model, thinking_level }`. `prepare_next_turn` returns `AgentLoopTurnUpdate`, which has one more field, `messages`: messages to append before the next request; they emit lifecycle events as usual. Replaced values apply to this request and all later ones.

## The loop

The flow chart is in [README.md](README.md#overall-structure). Step by step, matching upstream:

1. **Start**: `AgentStart`, `TurnStart`. `run_agent_loop` first passes the prompts through `declare_tool_changes`, adds them to the context and emits `MessageStart`/`MessageEnd`. Then it reads the steering queue once (messages queued before the run started).
2. **Prepare the turn**: if this is not the first turn, call `prepare_next_turn`. If there are no pending messages at this point, read the steering queue once more (preparation can be slow, for example when compacting the context), then emit `TurnStart`.
3. **Append pending messages**: the messages from `prepare_next_turn` plus the queued messages go through `declare_tool_changes`, then each one emits its events and is added to the context.
4. **`prepare_request`**.
5. **Streaming request**: `transform_context` → `convert_to_llm` → `normalize_context`, resolve the API key, call `StreamFn`. Provider events become `MessageStart` / `MessageUpdate` / `MessageEnd`. If the stream ends without `Done`/`Error`, it is recorded as the error `Stream ended without a final message`. The final message's `thinking_level` records the requested level.
6. **Failure**: when the reply ends with `error`/`aborted`, call `finish_turn` (decision ignored), emit `TurnEnd` and `AgentEnd`, and stop.
7. **Tools**: if there are tool calls, run them (see [tools.md](tools.md)) and add the results to the context. When all results request `terminate`, the loop does not continue because of tool calls after this batch.
8. **End the turn**: `finish_turn`, `TurnEnd`. If the decision is `End`, emit `AgentEnd` and stop.
9. **Continue condition**: read the steering queue. If there are tool calls to continue with or steering messages, go back to step 2. Otherwise read the follow-up queue; if it has messages, go back to step 2. If neither has messages but `finish_turn` returned `Continue`, run one more request with only the current context. Otherwise emit `AgentEnd` and stop.

### Tool declaration changes (`declare_tool_changes`)

`context.tools` holds the tools that can run at runtime. System messages in the transcript declare which tools the model may call (the `delta_ai` transcript design, see [delta_ai transcript.md](../../../ai/docs/design/transcript.md)). When the two disagree (for example after the application calls `Agent::set_tools`), the difference is written as `tools_added` / `tools_removed` on a system message before the next request:

- If the messages to append include a system message, rewrite the last one: its tool fields are replaced by the difference.
- Otherwise insert a new system message before the first non-system message.
- If there is no difference and the caller's system message declares no change itself, keep it unchanged.

This function is public, as upstream, so applications can use it when they build messages themselves.

## Differences from upstream

- Config is split into `stream_options` + `hooks` + queue sources; see above.
- No global default `StreamFn`.
- The context stores the final message carried by the `Done`/`Error` event. Upstream stores the partial object mutated in place during streaming. The content is the same.
- Dropping the stream returned by `agent_loop` cancels the run. Upstream's `EventStream` has no cancellation semantics.
