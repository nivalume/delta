# API: pi-messages

## Purpose

`pi-messages` streams pi's own message protocol to a backend. The request is one `POST {baseUrl}/messages` with `{model, context, options}`; the response is an SSE stream of serialized assistant-message events that ends with a `done` or `error` event. It is the wire protocol of the Radius gateway ([radius.md](radius.md)), but any backend that implements it works: a custom provider only needs models with `"api": "pi-messages"` and `ProviderApi::single(PiMessagesApi)`.

The implementation is in `crates/ai/src/api/pi_messages.rs`. It exposes:

- `PiMessagesApi` (unit struct, implements `ProviderStreams`);
- `stream(model, context, options)` and `stream_simple(model, context, options)`;
- `PiMessagesEvent` and `PiMessagesRewriteImpact`, the wire event types (upstream exports them as types). They derive `Serialize` and `Deserialize`, so a Rust backend can produce the wire format with them.

API options come from `ApiOptions::PiMessages(PiMessagesOptions)` in `options.api` (`reasoning`, `tool_choice`, `debug`); other variants are ignored.

## Upstream source files

- `packages/ai/src/api/pi-messages.ts` (and `pi-messages.lazy.ts`, the lazy registration)
- `packages/ai/src/utils/diagnostics.ts` (`createAssistantMessageDiagnostic`, `appendAssistantMessageDiagnostic`)
- Tests: `test/pi-messages.test.ts`, ported to `tests/pi_messages.rs` (one test per `it`, plus cases for thinking, parallel tool calls, rewrites, malformed events, request options, transcript replay and cancellation)
- Behavior reference: `test/radius-thinking-replay-e2e.test.ts` (live network, not ported; see below)
- Reference: pi-py `pi_ai/api/pi_messages.py`

## Request mapping

### Flow

1. `options.api_key` is required; otherwise fail with `No API key provided for provider "<provider>"`. An `authorization` header alone does not count (as upstream).
2. URL: `model.base_url` without trailing slashes, plus `/messages`. With `debug: Some(true)` the query parameter `debug=1` is set (`URLSearchParams.set`: an existing `debug` parameter is replaced).
3. Build the body and pass it to `on_payload`.
4. Send one request. There are no client retries and no client timeout: upstream uses a plain `fetch` and the backend owns retries. `options.max_retries`, `options.max_retry_delay` and `options.timeout` are ignored. `options.cancel` stops the request at any point.
5. Call `on_response` with the status and headers, for successful and failed responses.
6. A non-2xx status fails the request (see Errors). Otherwise read SSE events until `done` or `error`.

### Request body

| Field | Source |
| --- | --- |
| `model` | `model.id` |
| `context` | The `TranscriptContext` as delta serializes it: `{"messages": [...]}` in upstream's JSON shape. It is sent unchanged: no `transform_messages`, no system message collapsing, no tool call id normalization. The backend handles replay. |
| `options.temperature` | `options.temperature` |
| `options.maxTokens` | `options.max_tokens` |
| `options.reasoning` | `PiMessagesOptions::reasoning` (`minimal` ... `max`) |
| `options.cacheRetention` | `options.cache_retention`; when unset, `long` if `PI_CACHE_RETENTION=long` (from `options.env` or the process environment), else absent so the backend default applies |
| `options.sessionId` | `options.session_id` |
| `options.toolChoice` | `PiMessagesOptions::tool_choice`, a raw JSON value: `"auto"`, `"none"`, `"required"` or `{"type": "function", "function": {"name": ...}}` |

Unset fields are left out, as `JSON.stringify` drops `undefined`.

Because the context is sent as-is, foreign assistant messages keep their thinking blocks, signatures and `providerThinkingLevel`. The e2e test `radius-thinking-replay-e2e.test.ts` documents why this matters: Claude models with managed effort bind a signed thinking block to the effort level of the request that produced it, and `anthropic-messages` replays that level from `providerThinkingLevel`. A reply whose `done` event lacks `providerThinkingLevel` loses its thinking on replay. The client side of that contract is covered here: `done`/`error` copy `providerThinkingLevel` onto the message, and the field is serialized back into the next request (`sends_the_transcript_unchanged`).

### Request headers

Merged with `http::build_headers`, later sources win:

1. `authorization: Bearer <api_key>`, `accept: text/event-stream`, `content-type: application/json`
2. `options.headers`

### `stream_simple`

As upstream's `streamSimple`: the base options pass through unchanged, `reasoning` is forwarded as it is (no clamping to the model's thinking levels; the backend applies them), `tool_choice` becomes `"auto"` or `"none"`, and `debug` is kept from `ApiOptions::PiMessages` when the caller set it. No thinking budgets, no `max_tokens` adjustment.

## Stream event mapping

The SSE `data` of each event is parsed as JSON, passed to `on_provider_stream_event` unchanged (unknown fields included), then replayed through `MessageBuilder`. Empty data and `[DONE]` are skipped. Events with an unknown `type` are skipped after the observer saw them.

| Wire event | delta |
| --- | --- |
| `start` | `MessageBuilder::start()` (`Start` with stop reason `pending`) |
| `text_start {contentIndex}` | `start_text()` |
| `text_delta {contentIndex, delta}` | `text_delta` |
| `text_end {contentIndex, content, contentSignature?}` | the block's text becomes `content`, `text_signature` becomes `contentSignature`; `end_text` |
| `thinking_start {contentIndex}` | `start_thinking("", false, None)` |
| `thinking_delta {contentIndex, delta}` | `thinking_delta` |
| `thinking_end {contentIndex, content, contentSignature?, redacted?}` | thinking, `thinking_signature` and `redacted` set from the event; `end_thinking` |
| `toolcall_start {contentIndex, id, toolName}` | `start_tool_call(id, toolName, {})` |
| `toolcall_delta {contentIndex, delta}` | `tool_call_delta` (consumers see best-effort parsed partial arguments) |
| `toolcall_end {contentIndex, toolCall}` | `end_tool_call_with(index, toolCall)` |
| `done {reason, usage, responseId?, providerThinkingLevel?, rewrite?}` | stop reason, usage and `response_id` copied; `provider_thinking_level` only when present; `Done` |
| `error {reason, usage, errorMessage?, responseId?, providerThinkingLevel?, rewrite?}` | the same plus `error_message`; `Error` with the partial content kept |

Usage and cost come from the backend as they are; delta does not recompute them from the model's prices. A `rewrite` impact (`{policyId, policyVersion, changed, tokenCountChange, messageCountChange, systemPromptChanged}`) on a terminal event adds a `pi_messages_rewrite` diagnostic whose `details` are the impact's fields.

A stream that ends without `done` or `error` fails with `<provider> stream ended without a terminal event`.

## Errors

- A non-2xx response fails with `<status> <statusText>: <message>[ (<code>)]`. `message` and `code` come from the body's `error` object when the body is JSON with one (`{"error": {"message", "code"}}`); otherwise the whole body is the message. The status text is the canonical reason phrase.
- That failure carries a `pi_messages_response_failure` diagnostic: `error` is `{name: "PiMessagesResponseError", message, code?}`; `details` are `{version: 1, provider, model, url, status, statusText, error | body, timestampMs}`, with `body` truncated to 8192 characters plus `…`. No diagnostic is added when the request was cancelled.
- Every error raised before a terminal event (missing key, HTTP failure, malformed JSON, missing terminal event, cancellation) ends the stream with a fresh error message: no partial content, empty usage. This is upstream's `createErrorEvent`. A server-sent `error` event instead keeps the partial content.
- Cancellation ends with stop reason `aborted` and `Request was aborted`, as in the other delta APIs.

## Rust-specific differences

- Events carry no shared `partial`; the consumer stream reduces them (see [events.md](events.md)). Wire content indexes are mapped to the builder's indexes. A delta or end event for an index that was never started is skipped; upstream throws a `TypeError` there.
- `MessageBuilder` emits `Start` before the first content event even if the backend sent no `start`, so consumers always see a started stream. For backends that send `start` (all valid ones) nothing changes.
- Unknown event types are skipped. Upstream forwards them to the consumer as events of that unknown type, which Rust's event enum cannot represent.
- `toolcall_end` needs a complete `toolCall` (`id`, `name`, `arguments`) and replaces the block. Upstream merges the wire object into the block with `Object.assign`; for complete calls the result is the same.
- `done.reason` and `error.reason` are parsed as `StopReason`. A `done` whose reason is not `stop`/`length`/`toolUse` ends through `spawn_stream` like other APIs: `error`/`aborted` become an `Error` event, `pending` becomes `stop`. An `error` event without `errorMessage` gets `An unknown error occurred`; upstream leaves `errorMessage` undefined. Tool calls left open at `done` are closed by `spawn_stream`.
- A missing `usage` deserializes as empty usage; a malformed one fails the event.
- SSE is decoded by `http::sse_events` (WHATWG rules). Upstream splits on blank lines after CRLF normalization and reads only the first `data:` line of an event. For pi-messages backends (one JSON line per event) the results are the same; a multi-line `data` field is joined with `\n` here.
- `options.headers` entries with value `None` remove the default headers of the same name (`http::build_headers`). Upstream's `providerHeadersToRecord` drops null entries without removing the defaults.
- The diagnostic body is truncated by characters; upstream truncates UTF-16 code units.
- An invalid base URL fails with `Invalid URL: <url> (<reason>)` instead of the JavaScript `TypeError` text. Malformed event JSON fails with serde's message instead of `JSON.parse`'s.
- `response.body` is always present in `reqwest`, so upstream's `<provider> response has no body` branch has no counterpart.

## Not ported

- The registry tests of `pi-messages.test.ts` (`getApiProvider("pi-messages")`, the `Api` type): delta has no global API registry (see the README's "Migrating from the Old Global API"). The API is used through `PiMessagesApi`.
- `radius-thinking-replay-e2e.test.ts`: it needs real credentials and paid requests. Its client-side contract is covered offline (see Request body).

## Improvements pending confirmation

1. A response-header or idle timeout. Upstream sets none, so a gateway that accepts the connection and never answers holds the request until `options.cancel` fires. `options.timeout` could bound the wait for the response headers, as `mistral-conversations` does.
2. Retries for connection errors before any response arrived (the backend cannot retry a request it never received). Upstream sends once.
