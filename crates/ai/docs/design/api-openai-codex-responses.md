# API: openai-codex-responses (and `session_resources`)

## Purpose

Implements the Responses API of the ChatGPT Codex backend (`{baseUrl}/codex/responses`, default base URL `https://chatgpt.com/backend-api`). It serves the `openai-codex` provider: a ChatGPT Plus/Pro subscription signed in with `OpenAICodexOAuth` (see [oauth.md](oauth.md)). The access token is a JWT that names the ChatGPT account.

Two transports:

- **WebSocket** (`wss://…/codex/responses`): one `response.create` message per request. With a session id, the connection is cached per session and ChatGPT account and reused by the next request. In `websocket-cached` and `auto` mode, a follow-up request on a cached connection sends only the new input items with `previous_response_id`.
- **SSE** (`POST …/codex/responses`): the body is zstd-compressed; the API has its own retry loop.

`auto` (the default) tries the WebSocket first and falls back to SSE when it fails before the first event. A session whose WebSocket failed uses SSE from then on.

Input conversion and event normalization come from `api::openai_responses_shared` (see [api-openai-responses.md](api-openai-responses.md)).

`session_resources` is the per-session cleanup registry the WebSocket cache registers with. Apps call `cleanup_session_resources(Some(session_id))` when a session ends.

## Upstream source files

- `packages/ai/src/api/openai-codex-responses.ts`: `stream`, `streamSimple`, request body, headers, SSE parser, retry loop, WebSocket cache, debug stats.
- `packages/ai/src/api/openai-prompt-cache.ts`: `clampOpenAIPromptCacheKey` (already in `openai_responses_shared::clamp_openai_prompt_cache_key`).
- `packages/ai/src/session-resources.ts`: `registerSessionResourceCleanup`, `cleanupSessionResources`.
- `packages/ai/src/providers/openai-codex.ts`, `openai-codex.models.ts`: the provider.
- Tests: `test/openai-codex-stream.test.ts` (ported to `tests/openai_codex_responses.rs`; the connection age case is a unit test in the module), the Codex case of `test/pre-generation-error.test.ts`. Not ported: `test/openai-codex-cache-affinity-e2e.test.ts` and `test/codex-websocket-cached-probe.ts` (live probes that need a real ChatGPT token).

## Public API

| Upstream | Rust |
| --- | --- |
| `openAICodexResponsesApi()` | `api::openai_codex_responses::OpenAICodexResponsesApi` |
| `stream`, `streamSimple` | `stream`, `stream_simple` |
| `OpenAICodexResponsesOptions` | `options::OpenAICodexResponsesOptions` in `ApiOptions::OpenAICodexResponses` (`reasoning_effort`, `reasoning_summary`, `service_tier`, `text_verbosity`, `tool_choice`) |
| `OpenAICodexWebSocketDebugStats` | `OpenAICodexWebSocketDebugStats` (snake_case fields) |
| `getOpenAICodexWebSocketDebugStats(id)` | `get_openai_codex_websocket_debug_stats(id)` |
| `resetOpenAICodexWebSocketDebugStats(id?)` | `reset_openai_codex_websocket_debug_stats(Option<&str>)` |
| `closeOpenAICodexWebSocketSessions(id?)` | `close_openai_codex_websocket_sessions(Option<&str>)` |
| `registerSessionResourceCleanup(fn)` → unregister function | `session_resources::register_session_resource_cleanup(Arc<dyn Fn>)` → `SessionResourceRegistration` (`unregister()`) |
| `cleanupSessionResources(id?)` (throws `AggregateError`) | `cleanup_session_resources(Option<&str>) -> Result<(), SessionResourcesCleanupError>` |
| — | `resolve_codex_url`, `resolve_codex_websocket_url`, `apply_service_tier_pricing` |

`StreamOptions` fields used: `api_key`, `headers`, `session_id`, `cache_retention`, `transport` (default `Auto`), `timeout` (SSE header timeout and WebSocket idle timeout; none by default), `websocket_connect_timeout` (default 15 s; zero disables), `max_retries` (default 0), `max_retry_delay` (default 60 s; zero disables the cap), `temperature`, `on_payload`, `on_response`, `on_provider_stream_event`, `cancel`, `http_client` (SSE only).

## Request mapping

### Auth and session

- No API key (or an empty one): `No API key for provider: <provider>`.
- The account id is the JWT claim `https://api.openai.com/auth` → `chatgpt_account_id`, decoded like `atob` (standard base64, padding optional). A token without it fails with `Failed to extract accountId from token`.
- The cache session id is `session_id` unless `cache_retention` is `none` (`PI_CACHE_RETENTION` is not read, as upstream). The Codex session id is that id clamped to 64 code points.

### Request body (in insertion order)

| Field | Rule |
| --- | --- |
| `model` | `model.id` |
| `store` | `false` (the backend rejects `true`) |
| `stream` | `true` |
| `instructions` | Text of the initial system message, or `You are a helpful assistant.` when empty |
| `input` | `convert_responses_messages` without the system prompt; tool call providers `openai`, `openai-codex`, `opencode` |
| `text` | `{"verbosity": text_verbosity or "low"}` |
| `include` | `["reasoning.encrypted_content"]` |
| `prompt_cache_key` | The Codex session id, when present |
| `tool_choice` | `tool_choice` or `auto` |
| `parallel_tool_calls` | `true` |
| `temperature` | When set |
| `service_tier` | When set |
| `tools` | Request-level tools with `strict: null` default; `supportsStrictMode` defaults to `true` here |
| `reasoning` | See below |

Reasoning: with `reasoning_effort = none`, the effort is the model's `off` level (`none` when unmapped; no `reasoning` when it maps to `null`); other efforts go through `thinkingLevelMap` (a `null` entry keeps the effort). `summary` is `reasoning_summary` or `auto`. Without an effort, a reasoning model whose `off` level is not `null` sends `{"effort": off or "none"}`.

`stream_simple` clamps the reasoning level to the model's levels (`off` sends none) and maps `ToolChoice` to `auto` / `none`.

`on_payload` may replace the body; the replaced body is sent on both transports.

### Headers

Merge order (later wins, case-insensitive, `None` deletes): `originator: pi`, `User-Agent: delta/…`; `model.headers`; `options.headers`; then `Authorization: Bearer <token>` and `chatgpt-account-id` always win.

- SSE adds `OpenAI-Beta: responses=experimental`, `accept: text/event-stream`, `content-type: application/json`, `content-encoding: zstd`, and with a session `session-id` and `x-client-request-id`.
- WebSocket removes `accept` and `content-type`, and sets `OpenAI-Beta: responses_websockets=2026-02-06`, `x-client-request-id` and `session-id` to the Codex session id or a fresh UUID v7. (Upstream deletes `OpenAI-Beta` from the lower-cased header record with a mixed-case key, which is a no-op, so the beta header is sent; delta sends it too.)

## Transports

### Flow

1. `transport` other than `sse`, and the session has not fallen back to SSE: up to three WebSocket attempts.
   - `previous_response_not_found` (the backend lost the continuation): retry once with a fresh connection and the full body.
   - `websocket_connection_limit_reached` before the first event: retry once.
   - Cancelled, or a Codex error (an `error`/`response.failed` event, invalid JSON, an `on_provider_stream_event` failure) other than the connection limit before start: fail.
   - Any other failure (connect error or timeout, close, idle timeout, normalization error): add a `provider_transport_failure` diagnostic (`configuredTransport`, `fallbackTransport: "sse"` before the first event, `eventsEmitted`, `phase`, `requestBytes`) to the message, record the failure for the session; after the first event fail, otherwise fall back to SSE.
2. SSE with retries.

`start` is emitted once: on the first WebSocket event, or when the SSE response is accepted.

### SSE

- Retry loop (`max_retries`, default 0): a 429/500/502/503/504 response, or a body matching `rate limit`, `overloaded`, `service unavailable`, `upstream connect`, `connection refused` (one optional character between the words, case-insensitive), is retried unless it is a 429 with a terminal limit (`GoUsageLimitError`, `FreeUsageLimitError`, `Monthly usage limit reached`, `available balance`, `insufficient_quota`, `out of budget`, `quota exceeded`, `billing`). The delay is `retry-after-ms`, `retry-after` (seconds or HTTP date), or `1 s · 2^attempt`. A server delay above `max_retry_delay` fails at once with `Server requested <n>s retry delay (max: <m>s)`.
- Other failed responses throw the parsed error message, and like connection errors they are retried with the exponential delay while attempts remain, unless the message contains `usage limit` (upstream's catch-all; see "Improvements pending confirmation").
- Error message: the body's `error.message`, else the raw body, else the status text. With an `error.code`/`error.type` matching `usage_limit_reached`, `usage_not_included` or `rate_limit_exceeded`, or status 429, the friendly message `You have hit your ChatGPT usage limit[ (<plan> plan)].[ Try again in ~<n> min.]` is used instead.
- `timeout` bounds the wait for the response headers: `Codex SSE response headers timed out after <ms>ms`. `on_response` runs for every response.
- Parsing follows upstream's own parser: frames end at `\n\n`; `data:` lines are trimmed and joined; empty data and `[DONE]` are skipped; invalid JSON fails with `Invalid Codex SSE JSON: <reason>`. A last frame without a blank line is processed at the end of the body.

### Event mapping (both transports)

Each event goes to `on_provider_stream_event` first. Then:

| Event | Mapping |
| --- | --- |
| no `type` | Skipped |
| `error` | Error `Codex error: <message or code or event JSON>` with the code (`message`/`code` at the top level or under `error`) |
| `response.failed` | Error `response.error.message` or `Codex response failed` |
| `response.done`, `response.completed`, `response.incomplete` | `end_turn` (boolean) is copied to the message; the event becomes `response.completed` with the status kept when it is a known Codex status (otherwise removed); reading stops after it, even when the stream stays open |
| other | Passed to the shared Responses normalizer |

The service tier for pricing is the response's tier, except that `default` echoed for a `flex` or `priority` request uses the requested tier. Multipliers: `flex` 0.5, `priority` 2 (`gpt-5.5`: 2.5).

At the end: a pending stop reason fails with `Codex stream ended without a stop reason`; an `error`/`aborted` stop reason fails with the message's error.

### WebSocket

- Connect: tokio-tungstenite over rustls with the platform's native roots; `WebSocket connect timeout after <ms>ms`.
- Read: text and binary messages are JSON events; invalid JSON is `Invalid Codex WebSocket JSON: <reason>`. A close before the terminal event is `WebSocket closed <code>[ <reason>]` (`message too big` for 1009 without a reason; 1005 for a close frame without a code, 1006 for a dropped connection, as browsers report them). `timeout` is an idle timeout between messages: the socket is closed with `idle_timeout` and the attempt fails with `WebSocket idle timeout after <ms>ms`. A stream that ends without a terminal event fails with `WebSocket stream closed before response.completed`.
- Cache (session id present): one entry per session and account id.
  - A free entry younger than 55 minutes is reused; an older one is closed (`connection_age_limit`) and replaced.
  - A busy entry (concurrent request) gets a one-shot connection.
  - After a successful request the socket goes back to the cache; it is closed after 5 idle minutes (`idle_timeout`). On failure or cancellation it is closed and removed.
  - Without a session id every request uses a one-shot connection that is closed afterwards.
- Continuation (`websocket-cached` and `auto` on a cached entry): after a response with an id, the entry keeps the full request body, the response id and the response's input items (tool outputs left out). The next request on the entry sends only the new input items and `previous_response_id` when its body equals the last one apart from `input` (JSON text comparison) and its input starts with the last input plus the response items. Otherwise the continuation is dropped and the full body is sent. A failed request drops it.
- Debug stats per session: `requests`, `connections_created`, `connections_reused`, `cached_context_requests`, `store_true_requests`, `full_context_requests`, `delta_requests`, `last_input_items`, `last_delta_input_items`, `last_previous_response_id`, `websocket_failures`, `sse_fallbacks`, `websocket_fallback_active`, `last_websocket_error`.
- `close_openai_codex_websocket_sessions` closes cached sockets (`debug_close`); a request that is using one fails as if the socket closed. It is registered with `session_resources` on the first `stream` call (upstream registers when the module loads, which is lazy too).

## Rust-specific differences

- `timeoutMs`, `websocketConnectTimeoutMs`, `maxRetryDelayMs` are `Duration`s; upstream's `Invalid timeoutMs` check has no counterpart.
- zstd uses the pure-Rust `ruzstd` encoder at its `Fastest` level (about zstd level 1; upstream uses level 3 through `node:zlib`). The frame format is the same.
- WebSocket state: upstream reads `readyState` to decide whether a cached socket is reusable. Here an idle cached socket is owned by a small task that watches it: a close by the server makes it unusable, and the task closes it after the 5-minute TTL. Closing a socket sends a close frame and waits up to 5 s for the server's frame in the background.
- The WebSocket ignores `HTTP(S)_PROXY`; upstream uses a proxy only under Bun (Node relies on the app's global dispatcher). Behind a proxy-only network the WebSocket fails and `auto` falls back to SSE, which uses the `reqwest` client and its proxy settings.
- WebSocket connect errors carry tungstenite's message (for example `HTTP error: 401 Unauthorized`) where browsers report `WebSocket error`.
- JSON parse errors carry `serde_json`'s message after the fixed prefix.
- `cleanup_session_resources` returns an error value instead of throwing; `register_session_resource_cleanup` returns a handle instead of a function. Registering the same `Arc` twice keeps one entry, like upstream's `Set`.
- The diagnostic's `error.name` is the upstream class name (`CodexApiError`, `CodexProtocolError`, `ProviderStreamEventCallbackError`, `WebSocketCloseError`, `Error`); there is no stack.
- `stream_simple` without an API key returns an error stream instead of throwing synchronously.

## Not ported

- `test/openai-codex-cache-affinity-e2e.test.ts` and the `codex-websocket-cached-probe.ts` script: they need a live ChatGPT token.
- Telemetry.

## Improvements pending confirmation

1. The SSE catch-all retries every failed response (including 400s) while attempts remain, except usage-limit messages and too long server delays, because upstream throws the parsed error inside its `try`. Retrying only retryable statuses and connection errors would avoid repeating requests that cannot succeed.
2. Upstream's `assertSuccessfulOutput` runs inside the WebSocket `try`, so a response that ends with an `error` stop reason after the first event is recorded as a WebSocket failure and switches the session to SSE. Keeping the WebSocket for such responses would avoid the needless fallback.
3. The account id is decoded with `atob` semantics (standard base64). JWT payloads are base64url, so a payload with `-` or `_` fails; accepting base64url too would be more robust.
4. Proxy support for the WebSocket transport (HTTP CONNECT through `HTTPS_PROXY`).
