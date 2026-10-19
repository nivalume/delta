# API design: `anthropic-messages`

## Purpose

`delta_ai::api::anthropic_messages` implements the Anthropic Messages protocol (streaming). Every model with `model.api == "anthropic-messages"` goes through it: the official Anthropic endpoint, OAuth (Claude Code subscription), the GitHub Copilot Claude endpoint, and compatible endpoints such as OpenRouter, Vercel AI Gateway, Cloudflare AI Gateway, Fireworks and Kimi.

The module exposes:

- `AnthropicMessagesApi`: a unit struct that implements `ProviderStreams` and delegates directly to the two functions below.
- `stream(model, context, options)`: API-specific options come from `ApiOptions::AnthropicMessages(AnthropicOptions)` in `options.api`; other variants are ignored.
- `stream_simple(model, context, options)`: maps `reasoning` / `thinking_budgets` / `tool_choice` to `AnthropicOptions`, then calls `stream`.
- Some pure functions and constants: `normalize_tool_call_id`, `to_claude_code_name`, `from_claude_code_name`, `is_oauth_token`, `CLAUDE_CODE_VERSION`, `CLAUDE_CODE_SYSTEM_PROMPT`, `CLAUDE_CODE_TOOLS`.

Upstream sends the request through the Anthropic SDK (`client.beta.messages.create(params).asResponse()`) and then parses the SSE response body itself. Following the rules in AGENTS.md (Anthropic has no official Rust SDK, and no third-party SDKs are allowed), this module uses `reqwest` + `delta_ai::http` to hand-write the same request the SDK puts on the wire.

## Upstream source files

- `packages/ai/src/api/anthropic-messages.ts` (main body: options, request building, SSE parsing, event mapping, `streamSimple`; workload identity federation is described in [anthropic-federation.md](anthropic-federation.md))
- `packages/ai/src/api/constrained-sampling.ts` (strict JSON schema conversion, `resolveJsonSchemaStrictSampling`; in Rust this is the shared module `api::constrained_sampling`)
- `packages/ai/src/api/simple-options.ts` (`buildBaseOptions`, `adjustMaxTokensForThinking`, `clampMaxTokensToContext`; already in `simple_options.rs`)
- `packages/ai/src/api/transform-messages.ts` (`transformMessages`, `splitErrorToolResultImages`; already in `transform.rs`)
- `packages/ai/src/utils/provider-retry.ts` (retry policy; already in `http.rs`)
- `packages/ai/src/utils/transcript.ts`, `utils/text.ts` (system message replay; already in `transcript.rs`)
- Reference: pi-py `pi_ai/api/anthropic_messages.py` and `tests/test_anthropic_*.py`

The ported test cases come from upstream `test/anthropic-*.test.ts`, `cache-retention.test.ts`, `interleaved-thinking.test.ts` and `max-thinking.test.ts` (for e2e cases that need a real network, only the assertions that can be checked offline were ported), plus the three pi-py test files.

## Request mapping

### Endpoint and authentication

- `POST {baseUrl}/v1/messages?beta=true`. `?beta=true` is a query parameter built into the SDK's `beta.messages` path; requests from both pi and pi-py carry it. When `baseUrl` is an empty string, `https://api.anthropic.com` is used (same as the JS SDK's `baseURL || default`).
- The authentication check matches upstream `assertRequestAuth`: there must be a non-empty `api_key`, or a non-blank `authorization` / `x-api-key` / `cf-aig-authorization` in `options.headers`, or (provider `anthropic` only) a workload identity federation config. Otherwise the request fails with `No API key for provider: <provider>` (the stream contains a single `error` event).
- Workload identity federation (upstream `getAnthropicFederation`): for provider `anthropic`, when there is no key and no auth header, and `ANTHROPIC_FEDERATION_RULE_ID`, `ANTHROPIC_ORGANIZATION_ID` and `ANTHROPIC_IDENTITY_TOKEN_FILE` are set in `options.env` (else the process environment), the request sends `Authorization: Bearer <access token>` from a shared token cache in the auth header position and appends `oauth-2025-04-20` to `anthropic-beta`, as the SDK does. A 401 response drops the cached token. The exchange, the cache and their errors are in [anthropic-federation.md](anthropic-federation.md).
- Three authentication modes:
  - `provider == "github-copilot"`: `Authorization: Bearer <key>`, no OAuth rewriting, plus Copilot's dynamic headers (see [github-copilot.md](github-copilot.md)).
  - `api_key` contains `sk-ant-oat`: OAuth (Claude Code) mode. `Authorization: Bearer <token>`, plus `user-agent: claude-cli/2.1.280` and `x-app: cli`. The betas get `claude-code-20250219` and `oauth-2025-04-20`. The system prompt is prefixed with `You are Claude Code, Anthropic's official CLI for Claude.`. Tool names are rewritten to Claude Code's canonical casing, and tool names in the response are mapped back to the caller's names.
  - Otherwise: send `X-Api-Key` when there is a key. Without a key, the request headers carry the authentication (for example `Authorization: Bearer <ANTHROPIC_AUTH_TOKEN>`, or Cloudflare's `cf-aig-authorization`).

### Request headers

The merge order matches the SDK. Later entries override earlier ones, names are case-insensitive, and `None` deletes an earlier header with the same name:

1. SDK defaults: `Accept: application/json`, `anthropic-version: 2023-06-01`
2. Authentication header (`X-Api-Key` or `Authorization`)
3. Client default headers (upstream `mergeClientHeaders`): `User-Agent: delta/...`, `accept`, `anthropic-dangerous-direct-browser-access: true`; `user-agent`/`x-app` in OAuth mode; session affinity headers in API key mode; `model.headers`; for `github-copilot`, the dynamic headers `X-Initiator` (`agent` when the last message of the resolved transcript is not a user message, else `user`), `Openai-Intent: conversation-edits` and, when a user or tool result message carries an image, `Copilot-Vision-Request: true`; `options.headers`
4. `Content-Type: application/json`
5. Headers generated from request body fields: `betas` → `anthropic-beta` (comma-joined), `user_profile_id` → `anthropic-user-profile-id`. These two fields are taken out of the request body after `on_payload`, same as the SDK.

Session affinity: sent only in API key mode, when cache retention is not `none`, there is a `session_id`, and `compat.sendSessionAffinityHeaders` is set (default: true for OpenRouter). `sessionAffinityFormat == openrouter` (the OpenRouter default) uses `x-session-id`; otherwise `x-session-affinity`.

### Betas (`getBetaFeatures`)

When `anthropic-beta` appears in `model.headers` or `options.headers`, its value (trimmed, deduplicated) replaces the whole list, and `None` means no betas are sent. Otherwise betas are collected in order:

| Condition | beta |
| --- | --- |
| OAuth | `claude-code-20250219`, `oauth-2025-04-20` |
| Has tools and `supportsEagerToolInputStreaming == false` | `fine-grained-tool-streaming-2025-05-14` |
| `model.reasoning`, `thinking_enabled == true`, `interleaved_thinking != false`, not `forceAdaptiveThinking` | `interleaved-thinking-2025-05-14` |
| `allowedFallbackModels` is non-empty | `server-side-fallback-2026-07-01` |
| `supportsMidConvoEffort` | `mid-conversation-output-config-2026-07-01`, `thinking-binding-controls-2026-08-01` |
| Native tool changes | `inline-tools-2026-09-15` |

### Request body

Field order is the same as upstream: `model`, `messages`, `max_tokens` (`options.max_tokens`, otherwise `model.max_tokens`), `stream: true`, `betas`, `system`, `temperature`, `tools`, `thinking`, `output_config`, `metadata`, `tool_choice`, `fallbacks`. A replacement body returned by `on_payload` is forced back to `stream: true`.

- **Transcript**: `resolve_transcript` keeps or folds mid-conversation system messages according to `compat.supportsMidConvoSystemMessages`. `transform_messages` handles cross-model replay (drops foreign thinking and text signatures, adds `No result provided` for orphaned tool results, replaces images with placeholders for non-vision models). Tool call ids are normalized by `normalize_tool_call_id` to `^[a-zA-Z0-9_-]{1,64}$` (illegal characters are replaced with `_` one UTF-16 unit at a time, matching the upstream regex, then truncated to 64 characters).
- **System prompt**: the full text of the first system message (content + sections) becomes a `system` text block with `cache_control`.
- **User messages**: blank plain text is skipped. In a block list, blank text blocks are filtered out; if nothing is left, the message is skipped. Images become `{"type":"image","source":{"type":"base64",...}}`.
- **Assistant messages**: blank text is skipped. `redacted` thinking is replayed as `redacted_thinking`. Thinking with a signature is replayed as is (the thinking text may be empty). Thinking without a signature becomes a text block; with `compat.allowEmptySignature` it stays a thinking block with `signature: ""`. Tool calls become `tool_use`.
- **Tool results**: consecutive tool results are merged into one user message. For plain text, `content` is a string joined with `\n`; with images it is a block list (when there are only images, `(see attached image)` is prepended). Images in error results are split out with `transform::split_error_tool_result_images` and placed after all `tool_result` blocks of the turn (Anthropic rejects error results that contain images).
- **Mid-conversation system messages** (when the model supports them): rendered as `role: "system"` messages (`render_system_message_update`) and deferred until just before the next assistant message (or the end), because `tool_result` must directly follow `tool_use`.
- **Native tool changes**: enabled when `supportsMidConvoSystemMessages && supportsMidConvoToolChanges` and the first system message has tools. Top-level `tools` is fixed to the initial tools (the last one has a cache breakpoint) plus `__pi_deferred_placeholder__` (`defer_loading: true`). Later changes go into system messages as `tool_removal` / `tool_addition` blocks; redefining a tool with the same name only sends an addition. Otherwise top-level `tools` is the full current tool set.
- **Tool definitions**: `name` (rewritten in OAuth mode), `description`, `eager_input_streaming: true` (`supportsEagerToolInputStreaming`, default true), `strict: true` (in strict mode), `input_schema`, and `cache_control` on the last tool (`supportsCacheControlOnTools`, default true). In non-strict mode, `input_schema` keeps only `type/properties/required`. In strict mode it is the full strict schema, with these three keys then overwritten.
- **Strict tools**: tools with `constrainedSampling: {type: json_schema}` are made strict when `compat.supportsStrictTools` is set (by calling `resolve_json_schema_strict_sampling` in the shared module `api::constrained_sampling`, passing Anthropic's keyword check `is_strict_unsupported_keyword`: `minimum`, `maxItems`, `minItems` other than 0/1, `format` values not on the allow list, and so on). When a `prefer` tool cannot be made strict, it falls back to non-strict; for `require`, the request fails.
- **Cache**: the retention policy is `options.cache_retention`; otherwise long if `PI_CACHE_RETENTION=long` (checked in `options.env` first, then the process environment); otherwise short. `none` sends no `cache_control`. Long with `supportsLongCacheRetention` (default true) gives `{"type":"ephemeral","ttl":"1h"}`. Breakpoint positions: the system block, the last tool, and the last text/image/tool_result/tool_addition/tool_removal block of the last user/system message (string content is rewritten as a text block with `cache_control`).
- **thinking**:
  - `model.reasoning`, `thinking_enabled == false`, and `thinkingLevelMap.off` is not `null`: `{"type":"disabled"}`.
  - `supportsMidConvoEffort` (and not disabled): `{"type":"adaptive","display":...,"block_binding":{"prefix_mismatch_behavior":"drop_block"}}`, `output_config: {"effort":"high"}`. In the messages, `{"role":"system","content":[],"output_config":{"effort":...}}` is inserted before each replayed assistant message with a known effort (same provider, and `providerThinkingLevel` is a valid effort). The current turn's effort (default high) is inserted at the end and written to the output's `provider_thinking_level`.
  - `thinking_enabled == true`: with `forceAdaptiveThinking`, `{"type":"adaptive","display":...}`, plus `output_config` when there is an effort. Otherwise `{"type":"enabled","budget_tokens": budget or 1024,"display":...}`. `display` defaults to `summarized`.
- **temperature**: sent only when thinking is off, the model is not `supportsMidConvoEffort`, and `supportsTemperature` is set (default true). `sampling_params` is not used by this API (same as upstream).
- **metadata**: only the string `user_id` is taken.
- **tool_choice**: `auto`/`any`/`none` → `{"type":...}`; a named tool → `{"type":"tool","name":...}`.
- **fallbacks**: when `allowedFallbackModels` is non-empty, send `[{"model":...}]`.

### `stream_simple`

Same as upstream `streamSimple`:

- `build_base_options` produces the base options (tightening `max_tokens` to the context window), and `tool_choice` is mapped to `AnthropicToolChoice`.
- No `reasoning`: `thinking_enabled = false`.
- `forceAdaptiveThinking`: `thinking_enabled = true`. The effort is the string from `thinkingLevelMap`; otherwise minimal/low → low, medium → medium, everything else → high.
- Otherwise: `adjust_max_tokens_for_thinking` computes the limit and the budget, then `clamp_max_tokens_to_context`; the budget is `min(budget, max(0, max_tokens - 1024))`.

## Stream event mapping

Only SSE events named `message_start`, `message_delta`, `message_stop`, `content_block_start`, `content_block_delta` and `content_block_stop` are handled (data is parsed with `parse_json_with_repair`). `ping`, `[DONE]` and proxy extension events are skipped. `event: error` ends the stream directly, using the raw data as the error message. Each parsed event is first passed to `on_provider_stream_event`. `start` is emitted only after the HTTP response is accepted and `on_response` has been called.

| Anthropic event | Handling | delta events |
| --- | --- | --- |
| `message_start` | `response_id`, `input_transformations`. If `message.model` differs from the requested model, record `response_model`; if it is in `allowedFallbackModels`, use its pricing instead. Initial usage (`cacheWrite1h` from `cache_creation.ephemeral_1h_input_tokens`, default 0); recompute totals and cost | none |
| `content_block_start` `text` | New text block; when `content_block.text` is non-empty, it is the first delta | `text_start` (+ `text_delta`) |
| `content_block_start` `thinking` | New thinking block; initial thinking and signature (empty string by default) are given with the start event | `thinking_start` |
| `content_block_start` `redacted_thinking` | Thinking block `[Reasoning redacted]`, `redacted: true`, signature is `data` | `thinking_start` |
| `content_block_start` `tool_use` | Tool call (in OAuth mode the name is mapped back to the caller's tool name); initial arguments are `input` | `toolcall_start` |
| `content_block_start` `fallback` | If there is already output, error `Anthropic performed an unsupported mid-output model fallback`; otherwise ignored | none |
| `content_block_delta` `text_delta` | Append text | `text_delta` |
| `content_block_delta` `thinking_delta` | Append thinking | `thinking_delta` |
| `content_block_delta` `input_json_delta` | Append argument JSON and update the partial arguments with `parse_streaming_json` | `toolcall_delta` |
| `content_block_delta` `signature_delta` | Append thinking signature | none |
| `content_block_stop` | End the matching block. Tool call arguments are parsed from the full JSON (`{}` when there is no JSON; record `arguments_error` when it is not an object or cannot be parsed) | `text_end` / `thinking_end` / `toolcall_end` |
| `message_delta` | `input_transformations`; `stop_reason` → `raw_stop_reason` and the stop reason; usage is overwritten field by field (missing fields keep the `message_start` values), including `cache_creation.ephemeral_1h_input_tokens` and `output_tokens_details.thinking_tokens` (→ `reasoning`); recompute totals and cost | none |
| `message_stop` | Mark the stream as ended normally | none |
| Other event/block/delta types | Skipped | none |

Blocks are located by Anthropic's `index`: each index keeps a FIFO queue of unfinished blocks. This is equivalent to upstream `findIndex(b => b.index === event.index)` (the index is removed when the block ends).

Stop reasons: `end_turn`, `pause_turn`, `stop_sequence` → `stop`; `max_tokens` → `length`; `tool_use` → `toolUse`; `refusal` → `error` (message is `stop_details.explanation`, otherwise `The model refused to complete the request`); `sensitive` → `error` (`Provider stopped with: sensitive`); anything else → error `Unhandled stop reason: <value>` (`raw_stop_reason` is already recorded).

After the stream ends, these checks run in order: `message_start` seen but no `message_stop` → `Anthropic stream ended before message_stop`; cancelled → aborted; no stop reason → `Anthropic stream ended without a stop reason`; stop reason is error → fail with its message. On success, if there are `input_transformations` (the list from the last occurrence), an `anthropic_input_transformations` diagnostic is appended; each item in `details.transformations` keeps only the non-empty `type`/`path`/`reason`.

## Errors and retries

- Requests are sent through `http::send_with_retry`. Upstream `retryProviderRequest` does **not retry** when `maxRetries` is not specified, so here the default of `options.max_retries` is set to 0 (the core default is 2). When specified, retries follow the SDK policy for 408/409/429/5xx/connection errors and honor `x-should-retry`, `retry-after-ms`, `retry-after` and `max_retry_delay`.
- Strict tool fallback: when the request is rejected (HTTP 400) and the message contains one of `The compiled grammar is too large`, `Too many strict tools (` or `Schemas contains too many parameters with union types (`, and this request has `prefer` tools that were made strict, has not already fallen back, and was not cancelled: append an `anthropic_strict_tools_fallback` diagnostic (`details.reason = "strict_tool_limits_exceeded"`), make the `prefer` tools non-strict, rebuild the request and send it once more (calling `on_payload` again). `require` tools stay strict. The fallback happens only once; a second rejection is reported as an error. A stream that has already started (including an in-stream `event: error`) does not fall back.
- Cancellation: `options.cancel` is checked before each request is built, while waiting for the response and during retry backoff, and when reading each SSE event. The stream ends as aborted with `Request was aborted` and keeps the partial content already received. The stream also stops when the consumer drops it (`builder.is_closed()`).
- When the `on_payload` / `on_response` / `on_provider_stream_event` callbacks fail, the stream ends with their error message.

## Rust-specific differences

- **Event shape**: delta events do not carry a shared `partial`. When `content_block_start` carries non-empty text, an extra `text_delta` is emitted so that the partial message reduced by the consumer contains that text. The initial thinking text and signature are given with `thinking_start`. Upstream only emits `text_start`/`thinking_start`.
- **Unfinished tool calls**: on a successful end, `spawn_stream` emits a `toolcall_end` for any tool call that has not received `content_block_stop` (shared contract). Upstream does not.
- **HTTP error messages**: like the SDK's `APIError.makeMessage`, built from the raw response body in `Error::Status`. A JSON body gives `"<status> <entire JSON body>"` (for example `400 {"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long"}}`); otherwise `"<status> <text>"`; an empty body gives `"<status> status code (no body)"`. Differences: the body is first trimmed and truncated to 4000 characters (`Error::Status.body`), so a JSON body longer than 4000 characters cannot be parsed and is treated as text. The JSON is re-serialized by `serde_json`, so float formatting may differ from `JSON.stringify` (for example `1.0`). Connection error and timeout texts come from reqwest, not the SDK's `Connection error.` / `Request timed out.`.
- **Strict tool fallback check**: as upstream, requires status 400, `error.type == "invalid_request_error"` in the error body, and a message containing one of the limit texts above.
- **SSE parsing**: uses the shared `http::SseDecoder`. It drops events that have only `event:` and no `data:` (upstream fails to parse them as empty data). The `raw=` in parse failure messages is rebuilt from the event name and data lines, without comment/`id:` lines. JSON error texts come from serde, not JS `SyntaxError`. A parse result that is not an object also reports `Could not parse Anthropic SSE event ...`.
- **String sanitizing**: upstream calls `sanitizeSurrogates` on outgoing text to remove lone surrogates. A Rust `String` cannot contain lone surrogates, so nothing is needed.
- **Usage fields**: when `message_start` lacks `usage`, it is treated as 0 (upstream throws `TypeError`). Float token counts are truncated to integers.
- **Custom client**: upstream's `options.client` (injecting an SDK client such as `AnthropicVertex`) has no equivalent; a `reqwest::Client` can be injected through `StreamOptions::http_client`.
- **SDK telemetry headers**: `X-Stainless-*` is not sent (the SDK's own telemetry, not required by the protocol).

## Not ported

- The `telemetry` parent span option: upstream's `streamSimple` only forwards it (through `buildBaseOptions`); spans and metrics are recorded by `Models`, and telemetry as a whole waits for `delta-telemetry`.

The Anthropic OAuth login flow is ported in `oauth` (see [oauth.md](oauth.md)). Upstream's `anthropic-messages` API has no deferred responses and no WebSocket transport, so there is nothing to port for those.

## Improvements pending confirmation

- An in-stream `overloaded_error` (`event: error`) currently fails directly, as upstream. provider-references.md suggests treating it as a retryable error, but the stream has already started and events have been emitted, so a retry would change the event sequence. Whether and how to do it needs confirmation (for example, retry only when there are no content blocks yet).
- Upstream throws when `message_start` lacks `usage`; here it is treated as 0. It can be changed back to an error if strict parity is needed.
- When `baseUrl` is empty, pi-py reads `ANTHROPIC_BASE_URL`, while upstream JS uses the default address directly. This module follows upstream JS.
