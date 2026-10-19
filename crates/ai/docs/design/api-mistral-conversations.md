# API: mistral-conversations

## Purpose

`mistral-conversations` talks to Mistral's native Chat Completions endpoint: `POST {baseUrl}/v1/chat/completions` with `stream: true`, returning deltas over SSE. Upstream calls this endpoint through the HTTP layer of the Mistral SDK. delta has no official Mistral Rust SDK, so the client is hand-written on `reqwest` + `delta_ai::http`, and the request fields are exactly the ones upstream sends.

The implementation is in `crates/ai/src/api/mistral_conversations.rs`. It exposes:

- `MistralConversationsApi` (unit struct, implements `ProviderStreams`);
- `stream(model, context, options)`;
- `stream_simple(model, context, options)`.

API-specific options come from `ApiOptions::MistralConversations(MistralOptions)` in `options.api`; other variants are always ignored. The model compat setting is `ModelCompat::Mistral`; for now only `supportsMidConvoSystemMessages` is used.

## Upstream source files

- `packages/ai/src/api/mistral-conversations.ts` (main body)
- `packages/ai/src/api/constrained-sampling.ts` (strict JSON Schema, in the shared module `api::constrained_sampling`)
- `packages/ai/src/utils/hash.ts` (`shortHash`, in `api::common::short_hash`)
- Tests: `test/mistral-http-transport.test.ts`, `test/mistral-raw-stop-reason.test.ts`, `test/mistral-reasoning-mode.test.ts`, `test/mistral-tool-schema.test.ts`
- Reference: pi-py `pi_ai/api/mistral_conversations.py` and `tests/test_mistral_conversations.py`

## Request mapping

### Flow

1. `resolve_transcript`: when the model's `compat.supportsMidConvoSystemMessages` is true, mid-conversation system messages are kept; otherwise they are collapsed into one at the start.
2. `options.api_key` is required (matching upstream; an `authorization` header alone does not count), otherwise fail with `No API key for provider: <provider>`.
3. `transform_messages`, passing in the Mistral tool call id normalization function (see below).
4. Build the request body in the camelCase shape of the Mistral SDK and hand it to `on_payload` (it sees camelCase, e.g. `maxTokens`, `promptMode`, `promptCacheKey`).
5. Before sending, rename camelCase fields to the snake_case wire names (`to_wire_payload`).
6. Call `on_response` after receiving the response headers (for both successful and failed responses). Only on a success status code emit the `start` event and start reading SSE.

### Request body fields

| SDK field (as seen by `on_payload`) | Wire field | Source |
| --- | --- | --- |
| `model` | `model` | `model.id` |
| `stream` | `stream` | Always `true` |
| `messages` | `messages` | See the table below |
| `tools` | `tools` | Current tool set (`get_current_tools`); not sent when empty |
| `temperature` | `temperature` | `options.temperature` |
| `maxTokens` | `max_tokens` | `options.max_tokens` |
| `toolChoice` | `tool_choice` | `MistralOptions.tool_choice`: `auto`/`none`/`any`/`required` as is; a function object keeps only `{type: "function", function: {name}}` |
| `promptMode` | `prompt_mode` | `MistralOptions.prompt_mode` (sent only when non-empty) |
| `reasoningEffort` | `reasoning_effort` | `MistralOptions.reasoning_effort` (sent only when non-empty) |
| `promptCacheKey` | `prompt_cache_key` | `options.session_id`, when `cache_retention != none` and the session id is non-empty |

The new request body returned by `on_payload` is renamed too, so callers can add fields in SDK style such as `topP`, `randomSeed`, `responseFormat.jsonSchema.schemaDefinition`, `presencePenalty`, `frequencyPenalty`, `parallelToolCalls`, `safePrompt`, and on the wire they become `top_p`, `random_seed`, `response_format.json_schema.schema`, etc. `toolCalls`/`toolCallId` inside messages and `imageUrl`, `documentUrl`, `documentName`, `fileId`, `referenceIds`, `inputAudio` in content blocks are renamed as well.

### Message conversion

| Internal message | Mistral message |
| --- | --- |
| First system | `{role: "system", content}`, content is `get_system_message_text` (body plus sections); not sent when empty |
| Later system (only present when the compat setting allows it) | `{role: "system", content}`, content is `render_system_message_update` |
| user, string content | `{role: "user", content: "<text>"}` |
| user, block content | Text blocks `{type: "text", text}`; image blocks `{type: "image_url", imageUrl: "data:<mime>;base64,<data>"}` when the model supports images; if everything is filtered out and there were images, send the placeholder text `(image omitted: model does not support images)` instead |
| assistant | `{role: "assistant", prefix: false, content?, toolCalls?}`; blocks whose text is empty after trimming are skipped; thinking blocks become `{type: "thinking", thinking: [{type: "text", text}]}`; tool calls become `{id, type: "function", function: {name, arguments: <compact JSON string>}, index: 0}`; when both content and tool calls are empty, the whole message is not sent |
| toolResult | `{role: "tool", toolCallId, name, content}`; the first block is text (see below), followed by image blocks when the model supports images |

Tool result text rules (matching upstream): text blocks are joined with newlines and trimmed. On error, prefix `[tool error] `. When there is text, and there are images but the model does not support images, append `\n[tool image omitted: model does not support images]`. When there is no text, use in order `(see attached image)`, `(image omitted: model does not support images)`, `(no tool output)`.

Cross-model replay is handled by `transform_messages`: thinking in foreign assistant messages is dropped, failed or aborted assistant messages are not replayed, tool calls without results get a `No result provided` error result, and models that do not support images get placeholder text instead of images.

### Tool call id normalization

Mistral accepts only 9-character alphanumeric ids. The upstream normalization function is ported:

- Remove all non-ASCII-alphanumeric characters; on the first attempt, if the result is exactly 9 characters, use it directly;
- Otherwise hash the stripped string (or the original id if it is empty) with `shortHash` (cyrb53 style, computed over UTF-16 code units, base36 output) and take the first 9 alphanumeric characters;
- One mapping table per request: the same source id always maps to the same result; when a result is already taken by another source id, rehash with `<seed>:<attempt>`.

Example: `call_abc123|fc_xyz` → `nrregu2qr` (same as pi and pi-py). Normalization applies only to assistant messages from foreign models; tool results are rewritten to follow the mapping.

### Tool definitions and strict mode

Each tool is sent as `{type: "function", function: {name, description, parameters, strict}}`. When a tool has `constrainedSampling: {type: "json_schema"}`, try to convert the parameter schema to the strict subset (all properties listed in `required`, originally optional properties become `anyOf: [original schema, {type: "null"}]`, `additionalProperties: false` added to every object):

- Conversion succeeds: send the converted schema with `strict: true`;
- Conversion fails and `strict: "prefer"`: send the original schema with `strict: false`;
- Conversion fails and `strict: "require"`: fail before the request with `Tool "<name>" requires JSON-schema constrained sampling, but <reason>.`.

The checks for unsupported keywords, union types, tuples, etc. correspond one to one with upstream `makeStrictJsonSchema`. Upstream's `stripSymbolKeys` removes TypeBox symbol keys; Rust's `serde_json::Value` has no such keys, so it is not needed.

### Request headers

Merge order: API default headers, then `options.headers` (names are case-insensitive; `None` removes a default header).

| Header | Value |
| --- | --- |
| `User-Agent` | `delta/<version> (<os>; <arch>)` |
| `accept` | `text/event-stream` |
| `authorization` | `Bearer <api_key>` |
| `content-type` | `application/json` |
| `x-affinity` | Session affinity: when prompt caching is enabled (same condition as `promptCacheKey`) and no header source mentions `x-affinity` (including removal entries set to `None`), set to `session_id` |

### `stream_simple`

Matches upstream `streamSimple`:

- `build_base_options`: resolve sampling parameters by reasoning level, clamp `max_tokens` to the context window;
- `tool_choice`: `Auto` → `"auto"`, `None` → `"none"`;
- The reasoning level is first clamped to a level the model supports with `clamp_thinking_level`; `off` counts as not enabled;
- Model `reasoning` is true and it has a `thinkingLevelMap`: send `reasoning_effort` with the mapped value; when the mapping is missing or `null`, use `"high"`; when reasoning is not enabled, use the mapped value for `off` (e.g. `"none"`; not sent when `null`);
- Model `reasoning` is true but there is no mapping (Magistral): send `prompt_mode: "reasoning"` when reasoning is enabled;
- Non-reasoning models send no reasoning control fields.

## Stream event mapping

Each SSE `data` is trimmed. Empty data is skipped; `[DONE]` stops reading. The data must be a JSON object with a `choices` array, otherwise fail with `Invalid Mistral streaming event`. Each valid chunk goes first to `on_provider_stream_event`, then is processed.

| Mistral chunk content | Handling |
| --- | --- |
| Top-level `id` | The first non-empty value is written to `response_id` |
| Top-level `usage` | `input = prompt_tokens - cached`, `cache_read = cached`, `output = completion_tokens`, `cache_write = 0`; `total_tokens` prefers `total_tokens`, otherwise the sum; then cost is computed from the model's pricing. Cached tokens are the first non-empty value from, in order, `promptTokensDetails.cachedTokens`, `prompt_tokens_details.cached_tokens`, `promptTokenDetails.cachedTokens`, `prompt_token_details.cached_tokens`, `numCachedTokens`, `num_cached_tokens`, clamped to `[0, prompt_tokens]` |
| `choices[0].finish_reason` (non-empty) | Written to `raw_stop_reason`, and the stop reason is mapped (see the table below) |
| `delta.content` is a string | Text delta: `text_start` (when needed) + `text_delta` |
| String or `{type: "text"}` in a `delta.content` array | Same as above |
| `{type: "thinking", thinking: [{text}]}` in a `delta.content` array | The text segments are concatenated into a thinking delta: `thinking_start` (when needed) + `thinking_delta` |
| Empty string or empty text | Skipped, no new block opened (GLM models send empty deltas around thinking and tool calls; opening a new block would split thinking into several blocks, which Mistral rejects on replay) |
| Unknown block type | Skipped |
| `delta.tool_calls[]` | First end the current text/thinking block (`text_end`/`thinking_end`); find an existing call by `index` (by id when there is none) or create a new one (`toolcall_start`); string arguments are appended as is, object arguments are serialized then appended (`toolcall_delta`) |
| End of stream | End the current text/thinking block, then end all tool calls in order of appearance (`toolcall_end`; arguments are fully parsed here; a parse failure is written to `ToolCall::arguments_error`, not a stream error) |

Tool call id: take `id`, but the empty string and the literal `"null"` in Mistral continuation chunks count as no id. When there is no id, generate a 9-character id with `derive("toolcall:<index or 0>")`.

Across chunks, text and thinking keep appending while the type stays the same; when the type switches, the previous block is ended first.

### Stop reasons

| `finish_reason` | `stop_reason` | `error_message` |
| --- | --- | --- |
| `stop` | `stop` | — |
| `length`, `model_length` | `length` | — |
| `tool_calls` | `toolUse` | — |
| `error` | `error` | `Provider stopped with: error (server error)` (contains "server error", which the retry logic above uses to treat it as retryable) |
| Other | `error` | `Provider stopped with: <raw value>` |

At the end of the stream: if cancelled, the result is `aborted`; if no `finish_reason` was ever received, fail with `Mistral stream ended without a finish reason`; when the stop reason is `error`, end with an `error` event (keeping the content received); otherwise emit `done`.

## Errors and retries

- No client-side retry (upstream calls `fetch` directly and ignores `maxRetries`). `max_retries` and `max_retry_delay` have no effect for this API.
- The timeout covers only waiting for response headers: 60 seconds by default (upstream `timeoutMs ?? 60000`), overridden by `options.timeout`. On timeout, fail with `Mistral response headers timed out after <milliseconds>ms`. Reading the response body has no overall time limit, so long thinking streams are not cut off (upstream #10609).
- HTTP errors: call `on_response` first, then read the response body. The error message is `Mistral API error (<status code>): <trimmed raw response body, at most 4000 characters>`; when the body is empty, use the status phrase; when there is no phrase, `Request failed with status <status code>`. Internally this is `Error::Status`, recording `retry-after` and the raw response body (`body`).
- Cancellation: `options.cancel` takes effect while waiting for response headers, reading the response body and reading SSE. The stream ends with `aborted` and `Request was aborted`. When the consumer drops the stream, the producer stops reading.
- Failures in the request phase (before `start` is emitted) produce only a single `error` event.

## Rust-specific differences

- Upstream merges `model.headers` once more inside the API. In delta, `Models` merges the model headers into `options.headers` before calling the API (and runs `transform_headers` after that), and the API merges only the default headers and `options.headers`. The "explicitly set" check for `x-affinity` therefore looks only at `options.headers`, with the same effect. When calling `stream` directly, put the model headers into `options.headers` yourself.
- `User-Agent` uses delta's own value, not `pi (...)`.
- `stream_simple` in upstream throws synchronously when the API key is missing; Rust returns a stream with a single `error` event and the same message.
- Upstream applies `sanitizeSurrogates` to every string; Rust strings are guaranteed valid UTF-8, so it is not needed.
- Events carry no shared `partial`. Each time a tool call receives an argument segment, `parse_streaming_json` updates the arguments in the output message (matching upstream's partial parsing), and they are fully parsed again at the end. Empty argument deltas produce no `toolcall_delta` event (the common behavior of `MessageBuilder`); upstream emits an empty delta.
- SSE parsing uses the shared `http::SseDecoder` (handles `\r\n`, `\r`, multi-line `data`, UTF-8 across chunks); upstream has its own splitting logic. Results are the same for valid streams.
- Custom `fetch` (upstream `options.fetch`) is not ported; `StreamOptions::http_client` serves the same role.
- Error text for JSON parse failures comes from `serde_json` and differs from the JavaScript `SyntaxError` text.
- For inputs that would crash upstream with a type error, Rust is lenient: a `tool_choice` object missing `function.name` is sent as is; a missing `delta`, `content` that is neither a string nor an array, `null` content blocks, and tool calls missing `function` cause that part to be skipped.
- The following two items come from `docs/research/provider-references.md`. They only make the client more robust and do not change results for valid input. Implemented:
  - When another real id appears on the same `index`, it is treated as a new tool call (upstream concatenates the arguments of the two calls, producing invalid JSON);
  - The tool call name is set only when non-empty: when the first chunk has an empty name and a later chunk gives the name, it is filled in.
- In tests, each case uses its own `reqwest::Client`: the connection pool of a shared default client belongs to the Tokio runtime that opened the connection, `#[tokio::test]` uses one runtime per case, and reusing connections fails.

## Improvements pending confirmation

1. Retry 408/409/429/5xx per `options.max_retries` using the `http::send_with_retry` policy (pi-py already supports `max_retries`, default 0). Upstream does not retry, so this is not implemented yet.
2. In-stream error objects (e.g. `{"object": "error", "message": ...}`) currently only report `Invalid Mistral streaming event`; the `message` in them could be surfaced.
3. Anonymous tool calls (first without an id, later with a real id) keep the generated id; consider switching to the real id that arrives later.
4. Response body idle timeout: upstream leaves a stuck response body to the HTTP client's idle timeout; delta's default client has no read timeout.
5. The `model` field in chunks could be written to `response_model` (upstream does not).
