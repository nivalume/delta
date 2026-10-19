# Proxy stream (`proxy`)

Corresponds to upstream `src/proxy.ts`. The application forwards model requests to its own server, and the server holds the provider credentials. `stream_proxy` is the `StreamFn` implementation for this deployment style.

## Protocol

Request: `POST {proxy_url}/api/stream`, `Authorization: Bearer {auth_token}`, JSON body:

```json
{ "model": { ... }, "context": { "messages": [ ... ] }, "options": { ... } }
```

`options` is the serializable part of `SimpleStreamOptions`, with the same field names as upstream: `temperature`, `samplingParams`, `maxTokens`, `reasoning`, `cacheRetention`, `sessionId`, `headers`, `metadata`, `transport`, `thinkingBudgets`, `maxRetryDelayMs`. Empty fields are omitted.

Response: SSE; each `data:` is one `ProxyAssistantMessageEvent`. It is the same as `AssistantMessageEvent` without `partial` (the server does not resend the whole message), and `done`/`error` carry only the stop reason, usage and error text:

| Event | Fields |
| --- | --- |
| `start` | |
| `text_start` / `thinking_start` | `contentIndex` |
| `text_delta` / `thinking_delta` | `contentIndex`, `delta` |
| `text_end` / `thinking_end` | `contentIndex`, `contentSignature?` |
| `toolcall_start` | `contentIndex`, `id`, `toolName` |
| `toolcall_delta` | `contentIndex`, `delta` |
| `toolcall_end` | `contentIndex`, `toolCall` |
| `done` | `reason`, `usage`, `providerThinkingLevel?` |
| `error` | `reason`, `errorMessage?`, `usage`, `providerThinkingLevel?` |

## Client-side reconstruction

`stream_proxy` rebuilds the message with `delta_ai::events::MessageBuilder`, the same producer-side tooling the hand-written providers use. So the events sent to the loop, and the `partial()` reduction result, are the same as with a direct provider connection:

- The `contentIndex` of `*_start` must equal the current number of content blocks, and `*_delta`/`*_end` must point to a block of the matching type; otherwise the stream ends with an error.
- `toolcall_delta` arguments are parsed with the streaming JSON parser in `delta_ai::json`. `toolcall_end` replaces them with the final `toolCall` from the server (`MessageBuilder::end_tool_call_with`).
- `done`/`error` write the stop reason, usage and error text, then end the stream.

## Errors

| Case | Result |
| --- | --- |
| HTTP non-2xx | `Proxy error: {status} {reason}`; if the response body is JSON with an `error` field, `Proxy error: {error}` instead |
| Connection or parse failure | The error text is the underlying error |
| Stream closes normally before `done`/`error` | `Connection closed by proxy server before the response completed` |
| Cancellation | Stop reason `aborted`, text `Request aborted by user` |

## Differences from upstream

- Upstream splits SSE into lines by hand from the `ReadableStream` of `fetch`. delta uses `delta_ai::http::sse_events`, shared with the other hand-written providers.
- Upstream keeps a partial object on the client and mutates it in place. delta uses `MessageBuilder`, and the consumer side of the event stream reduces the partial (the `delta_ai` design).
- When the event stream is dropped, reading stops and the connection is closed (the `delta_ai` convention).
