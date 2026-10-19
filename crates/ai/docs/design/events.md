# Event protocol (`events`)

Upstream: `src/types.ts` (`AssistantMessageEvent`), `src/utils/event-stream.ts`, `src/api/lazy.ts`.

## Protocol

Success: `Start → update events* → Done`. Failure after generation has started: `Start → update events* → Error`. When request preparation (auth, building the request) fails, the stream holds a single `Error`. Update events and `Done` never come before `Start`, and no events follow a terminal event.

| Event | Upstream type | Fields | Notes |
| --- | --- | --- | --- |
| `Start` | `start` | `message` | Response metadata, usually no content blocks |
| `TextStart` | `text_start` | `content_index` | A text block starts, with empty content |
| `TextDelta` | `text_delta` | `content_index`, `delta` | |
| `TextEnd` | `text_end` | `content_index`, `content`, `signature` | `content` is the final text |
| `ThinkingStart` | `thinking_start` | `content_index`, `thinking`, `redacted`, `signature` | Empty for normal thinking; redacted thinking may be complete at start |
| `ThinkingDelta` | `thinking_delta` | `content_index`, `delta` | |
| `ThinkingEnd` | `thinking_end` | `content_index`, `content`, `signature` | |
| `ToolCallStart` | `toolcall_start` | `content_index`, `id`, `name`, `arguments` | `arguments` depends on the provider, usually empty |
| `ToolCallDelta` | `toolcall_delta` | `content_index`, `delta` | A fragment of the argument JSON |
| `ToolCallEnd` | `toolcall_end` | `content_index`, `tool_call` | The complete tool call, not validated against the schema |
| `Done` | `done` | `reason`, `message` | `reason`: stop / length / toolUse / deferred |
| `Error` | `error` | `reason`, `error` | `reason`: error / aborted; `error` carries the partial content and `error_message` |

Events of different content blocks can interleave (for example `text_start, text_delta, toolcall_start, text_delta, toolcall_delta`). Consumers match them by `content_index`.

## Differences from upstream: `partial` becomes consumer-side reduction

In upstream, every non-terminal event carries `partial`: the same `AssistantMessage` object the provider is modifying, not a snapshot taken when the event happened. In Rust, a shared mutable object needs a lock, and the semantics are vague, so (after confirmation) this changes to:

- Events do not carry `partial`.
- Each time `AssistantMessageEventStream` yields an event, it reduces that event into its own copy of the message. `partial()` returns the message as of the event just yielded, an exact snapshot at that event.
- So that the consumer can reduce, events carry extra block data: `ToolCallStart` carries `id`, `name` and the initial arguments; `ThinkingStart` carries the full content and signature of redacted thinking; `TextEnd`/`ThinkingEnd` carry the final signature.
- Tool call arguments: the consumer accumulates the JSON fragments of `ToolCallDelta` and gets the current arguments with streaming JSON parsing each time (as upstream re-parses on every delta).
- Consumers that only call `result()` do no reduction and pay no extra cost.

Response metadata (usage, response id and so on) is only updated in the terminal event during streaming, so these fields of `partial()` may lag. This is compatible with upstream's convention that partial is not a snapshot.

## Types

```rust
let (sender, stream) = AssistantMessageEventStream::channel();
```

- **`EventSender`**: the producer side. Events after a terminal event are ignored. It writes `duration_ms` on the terminal message, timed with a monotonic clock from when the sender was created. It does not time the message if the message already has `duration_ms`, or if its `timestamp` is earlier than the stream's creation time (a forwarded response that started elsewhere), as in upstream.
- **`AssistantMessageEventStream`**: the consumer side; implements `futures::Stream<Item = AssistantMessageEvent>`. For `partial()` see above. `result().await` reads up to the terminal event and returns the final message. If the producer ends without sending a terminal event, it returns a message with `stop_reason = error` ("Stream ended without a final message"). `from_final(message)` builds a stream that holds only one terminal event.
- **`MessageBuilder`**: a helper type API implementations use to build the response. It holds `output: AssistantMessage`. All content changes go through its methods, and each method also emits the matching event, so the producer's and consumer's messages stay identical:

| Method | Event |
| --- | --- |
| `start()` | `Start` (sent only once) |
| `start_text()` / `text_delta()` / `end_text()` | `TextStart` / `TextDelta` / `TextEnd` |
| `start_thinking()` / `thinking_delta()` / `end_thinking()` | `ThinkingStart` / `ThinkingDelta` / `ThinkingEnd` |
| `start_tool_call()` / `tool_call_delta()` / `end_tool_call()` | `ToolCallStart` / `ToolCallDelta` / `ToolCallEnd` |
| `finish()` | `Done` (sends `Error` when `stop_reason` is error/aborted) |
| `fail(message, aborted)` | `Error` |

`end_tool_call` parses the accumulated argument text with `json::parse_tool_call_arguments`. If it is not a JSON object, the arguments are empty, and `ToolCall::arguments_error` records the error and the raw text so tool validation can report the problem back to the model (upstream behavior). If no argument fragments arrived, the arguments from the start are kept.

## Runtime requirements

API implementations and `lazy_stream` run in Tokio tasks, so a Tokio runtime is required. The channel is unbounded. When the consumer is slow, events wait in the queue, as with upstream's in-memory queue. After the consumer drops the stream (`is_closed()`), the producer stops working (see "Dropping the event stream" under Confirmed design decisions in the README).
