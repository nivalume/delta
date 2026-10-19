# Gemini API (`google-generative-ai`) and Google shared logic (`api::google_shared`)

`google-generative-ai` talks to the Gemini API (`generativelanguage.googleapis.com`). Upstream sends requests through the `@google/genai` SDK. There is no official Rust SDK here, so per AGENTS.md the REST client is hand-written. The request and response JSON shapes match what that SDK actually sends and receives.

`google_shared` holds the logic shared by the Gemini API and Vertex AI (`google-vertex`). Vertex AI will later use the official `google-cloud-aiplatform-v1` crate and its proto types, so the shared module handles only plain data and does not depend on the transport.

## Upstream source files

- `packages/ai/src/api/google-shared.ts`: message/tool conversion, thinking config, thought signatures, stop reasons, retry.
- `packages/ai/src/api/google-generative-ai.ts`: `stream`, `streamSimple`, `buildParams`, `getGoogleBudget`.
- `packages/ai/src/api/constrained-sampling.ts`: strict JSON Schema (`google-shared` depends on it; in Rust it is the shared module `api::constrained_sampling`).
- Tests: `test/google-raw-stop-reason.test.ts`, `google-shared-convert-tools.test.ts`, `google-shared-gemini3-unsigned-tool-call.test.ts`, `google-shared-image-tool-result-routing.test.ts`, `google-shared-retry.test.ts`, `google-shared-signed-empty-blocks.test.ts`, `google-thinking-level-map.test.ts`, `google-thinking-signature.test.ts`; also see pi-py's `test_google_shared.py` and `test_google_streams.py`.

## Module layout

`google_shared` (transport-independent, reused directly by Vertex):

- Thinking: `resolve_google_thinking_level`, `uses_google_thinking_level`, `get_disabled_google_thinking_config`, `resolve_thinking_config` (the thinking branch in upstream `buildParams`), `get_google_budget`, `simple_thinking` (the thinking branch in upstream `streamSimple`), `ThinkingConfig`.
- Signatures: `is_thinking_part`, `retain_thought_signature`, `is_valid_thought_signature`, `resolve_thought_signature`.
- Model capabilities: `gemini_major_version`, `requires_tool_call_id`, `supports_multimodal_function_response`, `supports_google_strict_tool_sampling`, `normalize_tool_call_id`.
- Request: `convert_messages` → `Vec<Content>` (`Part` enum: text/thinking, inline data, functionCall, functionResponse); `convert_tools` → `Vec<FunctionDeclaration>`; `resolve_google_function_calling_mode`; `build_request` → `GenerateRequest` (upstream `GenerateContentParameters` without model).
- Schema: `sanitize_for_open_api`, `make_strict_json_schema`, `resolve_json_schema_strict_sampling`, `get_json_schema_tool_parameters`.
- Response: `ResponseChunk`/`Candidate`/`ResponsePart`/`UsageMetadata` (only the fields stream handling uses), `StreamState` (the state machine of the upstream stream loop), `check_finished`, `map_stop_reason`, `parse_usage`.
- Retry: `retry_google_request`, for transports that do not go through `http::send_with_retry` (the Vertex SDK).

`google_generative_ai` (REST transport): `GoogleGenerativeAiApi`, `stream`, `stream_simple`, `request_body` (`GenerateRequest` → REST JSON), `parse_chunk` (REST JSON → `ResponseChunk`), endpoint and request headers.

The Vertex module only has to: convert `GenerateRequest` into a proto request, convert proto responses into `ResponseChunk` (thought signatures are bytes in proto; convert them to base64 text), then call the same `StreamState`, and at the end use `check_finished(output, "Google Vertex stream ended without a finish reason")`.

## Request mapping

### Endpoint and request headers

- `POST {base}/{model}:streamGenerateContent?alt=sse`.
- `base`: the model's `base_url` (already includes the API version, matching upstream `apiVersion: ""`). When empty, use `GOOGLE_GEMINI_BASE_URL` (check `options.env` first, then the process environment) or `https://generativelanguage.googleapis.com`, then append `/v1beta` (the SDK default).
- Model path: if the id already starts with `models/` or `tunedModels/`, use it as is; otherwise prefix `models/` (SDK `tModel`). If the id contains `..`, `?` or `&`, fail with "invalid model parameter" (the SDK rejects these too).
- Request headers (`build_headers` merge order: defaults → `options.headers`): `Content-Type: application/json`, `User-Agent: delta/…`, `x-goog-api-key: <key>`. The key goes only in the header, never in the URL. `options.headers` can override any of them (matching upstream, where explicit headers override the default User-Agent). `Models` merges the model's `headers` into `options.headers`.

### Request body

| `GenerateRequest` field | REST JSON | Source |
| --- | --- | --- |
| `contents` | `contents[]`: `{role: "user"/"model", parts}` | `convert_messages` |
| `system_instruction` | `systemInstruction: {role: "user", parts: [{text}]}` (the shape of SDK `tContent`) | Text of the first system message after collapsing (including sections); not sent when empty |
| `temperature` | `generationConfig.temperature` | `options.temperature` |
| `max_output_tokens` | `generationConfig.maxOutputTokens` | `options.max_tokens` |
| `thinking_config` | `generationConfig.thinkingConfig: {includeThoughts, thinkingLevel, thinkingBudget}` | `resolve_thinking_config` |
| `tools` | `tools: [{functionDeclarations: [{name, description, parametersJsonSchema}]}]` | Current tool set after collapsing |
| `function_calling_mode` | `toolConfig.functionCallingConfig.mode` | Decided by `resolve_google_function_calling_mode` when there are tools |

`generationConfig` is always present (possibly an empty object), as in the SDK. Gemini does not support system messages in the middle of a conversation: all mid-conversation system messages are collapsed into `systemInstruction`, and tools added mid-conversation are merged into `tools`. `options.sampling_params`, `cache_retention`, `session_id` and `metadata` do not apply to this API and are ignored.

### Thinking config

The `thinking` option comes from `ApiOptions::GoogleGenerativeAi` (options for other APIs are ignored):

- `enabled` and the model has `reasoning`: `{includeThoughts: true}`, plus `thinkingLevel` (when there is a level) or `thinkingBudget` (when there is a budget).
- Explicitly disabled and the model has `reasoning`: `get_disabled_google_thinking_config`. Models that do not use `thinkingLevel` send `{thinkingBudget: 0}`. Models that use `thinkingLevel` clamp `off` against the thinking level table; if it is still `off`, send `{thinkingBudget: 0}`, otherwise send the lowest level after clamping (for example, Gemini 3.1 Pro cannot disable thinking, so it sends `LOW`).
- `thinking` not set: do not send `thinkingConfig`.

`uses_google_thinking_level`: the id (lowercased) matches `gemini-3(.N)-pro|flash`, `gemini-4` and later, `gemini-flash-latest`, `gemini-flash-lite-latest`, `gemma-4`/`gemma4`. Upstream uses a regex. There is no regex dependency here, so an equivalent matcher is hand-written and covered by table-driven tests.

`resolve_google_thinking_level`: take the value for that level from the model's thinking level table (case-insensitive); with no mapping, use the level name itself. The result must be minimal/low/medium/high, otherwise fail with `Unsupported Google thinking level mapping for <provider>/<id>: <level> -> <mapped value|null|undefined>`.

### `stream_simple`

Same as upstream `streamSimple`:

- `build_base_options` (`max_tokens` clamped to the context window).
- `tool_choice`: `Auto`/`None` map to Gemini's options of the same name.
- No `reasoning`, or the model clamps it to `off`: `thinking: {enabled: false}`.
- Otherwise resolve to a standard level. Models that use `thinkingLevel` send the level; other models send a budget: custom `thinking_budgets` take precedence, otherwise `2.5-pro` is 128/2048/8192/32768, `2.5-flash-lite` is 512/2048/8192/24576, `2.5-flash` is 128/2048/8192/24576 (minimal/low/medium/high), and other models `-1` (dynamic).
- When the level mapping is invalid, upstream throws synchronously; here it returns a stream with a single `Error` event.

### Message conversion (`convert_messages`)

First `collapse_system_messages` and drop the first system message, then `transform_messages` (models that need ids use `normalize_tool_call_id`: characters outside `[a-zA-Z0-9_-]` become `_`, truncated to 64). Rules:

- User message: string → one text part; blocks → text part / `inlineData`; skipped when there are no parts.
- Assistant message: signatures and thinking are kept only for the same provider and the same model (api is not compared, matching upstream).
  - Text: the signature goes through `resolve_thought_signature` (same model and valid base64) and becomes `thoughtSignature`. Whitespace-only text is kept only when it has a signature (Gemini signs parts with empty visible text and requires them to be sent back).
  - Thinking: for the same model, send `{thought: true, text, thoughtSignature?}`; drop it when it is whitespace-only with no signature. For a different model, send it as plain text (in practice `transform_messages` has already dropped foreign thinking, same as upstream).
  - Tool call: `functionCall {name, args, id?}`. Only models that need ids (`claude-*`, `gpt-oss-*`, Gemini 3+) get the id. Signature rules as above. `skip_thought_signature_validator` is never added.
  - Assistant messages with no parts are skipped; `error`/`aborted` assistant messages are dropped by `transform_messages`.
- Tool result: text blocks are joined with `\n`; whitespace-only counts as no text. Images are taken only when the model supports images. `response` is `{output: …}` (`{error: …}` on error). The value is the text, or `"(see attached image)"` when there is no text but there are images, otherwise the empty string. Gemini 3+ (and non-Gemini models) put images in `functionResponse.parts`; below Gemini 3, a user turn with `"Tool result image:"` plus the images is appended after it. Consecutive tool results are merged into the previous user turn that contains a `functionResponse`.
- Orphaned tool calls get a `"No result provided"` error result from `transform_messages`.

### Tools and Schema

- By default send `parametersJsonSchema` (the full JSON Schema, sent as is, including `$schema`, `$ref`, `$defs`). With `use_parameters = true` (upstream uses it for Claude models on Cloud Code Assist), send the old `parameters` and recursively remove `$schema`, `$id`, `$anchor`, `$dynamicAnchor`, `$vocabulary`, `$comment`, `$defs`, `definitions` (`$ref` is kept).
- When a tool declares `constrained_sampling: json_schema` and the model is Gemini 3+, the parameters are converted to strict form (all properties required, optional properties become nullable, `additionalProperties: false`) and the mode is `VALIDATED`. When strict mode is not supported: `prefer` falls back to a normal tool, `require` fails with `Tool "<name>" requires JSON-schema constrained sampling, but …`.
- Function calling mode: `none`/`any` as is; otherwise `VALIDATED` when there are strict tools; otherwise follow `tool_choice` (when unset, `toolConfig` is not sent). With no tools, neither `tools` nor `toolConfig` is sent.

## Stream event mapping

Each SSE `data:` is one `GenerateContentResponse` JSON. Empty data and proxy-injected `[DONE]` are skipped. An `{"error": …}` envelope fails immediately. Everything else goes first to `on_provider_stream_event`, then is parsed leniently by `parse_chunk` (only the first candidate is read; unknown fields and unknown part types are ignored; fields of the wrong type count as missing), then goes to `StreamState::apply_chunk`.

| Gemini response | Event / output field |
| --- | --- |
| HTTP response accepted | `on_response`, then `Start` |
| `responseId` | `response_id`, keep the first non-empty value |
| part `text`, `thought: true` | If the current block is not thinking, end the current block and `ThinkingStart`; `ThinkingDelta` (no event for empty strings) |
| part `text`, not thinking | If the current block is not text, end the current block and `TextStart`; `TextDelta` |
| part `thoughtSignature` (text/thinking part) | `thinking_signature`/`text_signature` of the current block, keep the last non-empty value, emitted with `ThinkingEnd`/`TextEnd` |
| part `functionCall` | End the current block; `ToolCallStart` (arguments are `args`), `ToolCallDelta` (compact JSON of `args`), `ToolCallEnd`; a non-empty `thoughtSignature` is recorded on the tool call |
| `functionCall.id` missing, empty, or duplicating an existing call in this message | Generate `<name>_<millisecond timestamp>_<in-process incrementing counter>`, unique across chunks |
| `finishReason` | `raw_stop_reason` stored as is; `STOP` → `stop`, `MAX_TOKENS` → `length`, others (including unknown values) → `error`; if there are tool calls and it is `stop`, change it to `toolUse` |
| `usageMetadata` (when it has `promptTokenCount`) | See below |
| End of stream | End the current block; no stop reason → error "Google stream ended without a finish reason"; stop reason maps to `error` → error "Provider stopped with: <raw>"; otherwise `Done` |

Usage: `input = promptTokenCount - cachedContentTokenCount` (saturating subtraction), `output = candidatesTokenCount + thoughtsTokenCount`, `cache_read = cachedContentTokenCount`, `cache_write = 0`, `reasoning = thoughtsTokenCount`, `total_tokens = totalTokenCount` (the server's value, as in upstream, not recomputed), cost via `calculate_cost`. Each chunk with usage overwrites the previous value, but usage without `promptTokenCount` does not overwrite (an early report with only some fields does not replace a complete one). Counts accept numbers or numeric strings.

## Errors and retry

- Missing key: `No API key for provider: <provider>`. `require_auth` also accepts `x-goog-api-key` in `options.headers`.
- HTTP errors: `http::send_with_retry`. The message matches `@google/genai`'s `ApiError` (`google_shared::format_google_http_error`): for a JSON body, the whole JSON re-serialized (e.g. `{"error":{"code":400,"message":"API key not valid.","status":"INVALID_ARGUMENT"}}`); for a non-JSON body, `{"error":{"message":"<text>","code":<status>,"status":"<status phrase>"}}`. Upstream's `formatProviderError` keeps it as is. Retryable: 408, 409, 429, 5xx, connection errors. `retry-after-ms`/`retry-after` are honored; exceeding `max_retry_delay` fails immediately.
- In-stream error envelope: `got status: <error.status>. <JSON of the whole chunk>` (the SDK's wording). Not retried. Content received so far is kept in the error message.
- Unparseable chunk: `Failed to parse Google stream chunk: <reason>`.
- Cancellation: `options.cancel` is honored before the request, while waiting to retry, and while reading the stream. It ends with `aborted` and keeps partial content. Reading stops when the consumer drops the stream.
- When `on_payload`, `on_response` or `on_provider_stream_event` returns an error, the request fails.
- `retry_google_request` (for Vertex): retries `Error::Status` 408/409/429/5xx, using `retry_after_ms` or exponential backoff. Other errors (transport errors, in-stream errors) are not retried. The default count matches the core.

## Rust-specific differences

- `on_payload` receives the REST request body (`contents`, `systemInstruction`, `tools`, `toolConfig`, `generationConfig`), not the SDK's `{model, contents, config}`. Changes must use REST field names (e.g. `generationConfig.topK`). The return value is sent as the request body as is.
- `on_provider_stream_event` receives the raw JSON of the chunk, not an SDK object (same field names).
- HTTP error message: the SDK decides whether to parse or wrap as text based on whether `content-type` is JSON; here it is decided by whether the body parses as JSON (`Error::Status` carries no response headers). When `content-type` is JSON but the body does not parse, upstream throws `SyntaxError`; here it is wrapped as text. The body is first trimmed of leading and trailing whitespace and truncated to 4000 characters; JSON bodies over 4000 characters are wrapped as text. The status phrase uses `StatusCode::canonical_reason` (fetch's `statusText` is the reason phrase from the server, and is empty under HTTP/2).
- Retry uses the core policy: `max_retries` defaults to 2 (upstream Google adapter defaults to 0), the `retry-after` header is used (the SDK's `ApiError` has no headers, so upstream never uses it), and connection errors are retried too (upstream does not).
- When cancelled up front the message is "Request was aborted" (in upstream `buildParams` it is "Request aborted").
- Stop reason: REST gets a string, so this uses the semantics of upstream `mapStopReasonString`. Unknown values map to `error` with the message "Provider stopped with: <raw>". The upstream SDK path throws "Unhandled stop reason: <raw>" for unknown enum values. Both end as `error`.
- In-stream error envelope: the SDK throws only when the raw chunk itself is JSON and `code` is in 400–599. Under `alt=sse` the envelope is in practice ignored as an empty response, and the final report is "stream ended without a finish reason". Here any non-empty `error` fails immediately (matching pi-py).
- `sanitizeSurrogates` is not needed: Rust strings cannot contain lone surrogates.
- Thought signatures are stored and sent back as the base64 text received (matching the JS SDK; pi-py's SDK converts them to URL-safe base64).
- Events carry no shared `partial`. `ToolCallStart` carries no signature; the signature is in the complete call in `ToolCallEnd`. Empty text parts emit no `TextDelta` (`MessageBuilder` skips empty deltas), but still open a block and record the signature.
- Usage is updated only when it contains `promptTokenCount`, with saturating subtraction (a provider-references requirement; no change for normal streams).
- Skip `[DONE]`, parse parts leniently (a struct with optional fields, not a tagged enum), ignore unknown part types (provider-references requirements).
- The tool call id counter is a process-level `AtomicU64`; upstream uses a module-level counter.
- Strict JSON Schema (upstream `constrained-sampling.ts`) is in the shared module `api::constrained_sampling`. `google_shared` keeps two wrapper functions, `make_strict_json_schema(&schema)` and `resolve_json_schema_strict_sampling(tool, supports)` (without keyword checks), and re-exports `get_json_schema_tool_parameters` and `UnsupportedStrictSchema`, so existing callers do not change.

## Not ported

- Upstream rejects a custom `fetch`: Rust uses `options.http_client` and has no matching check.
- The SDK's own `x-goog-api-client` header and User-Agent prefix (`google-genai-sdk/…`): the hand-written client is not the SDK and does not pretend to be.
- Telemetry: the upstream Google adapter has none.

## provider-references checks

- `x-goog-api-key` header, key not in the URL: yes.
- `STOP` changed to `toolUse` when a `functionCall` appears: yes (`MAX_TOKENS` stays `length`).
- Unknown stop reasons do not crash: mapped to `error` with the raw value kept.
- Empty `thoughtSignature` treated as none; signatures on non-thinking text parts kept: yes.
- Thinking parts become thinking blocks, not merged into text: yes.
- Calls without an id get an id unique across chunks: yes (duplicate ids are also regenerated, same as upstream).
- Usage waits for `promptTokenCount`, saturating subtraction, output includes `thoughtsTokenCount`: yes.
- In-stream `{"error": …}` envelope: yes.
- Lenient part parsing, unknown types not fatal, `[DONE]` skipped, error when there is no finish event: yes.
- `parametersJsonSchema`: yes. Inlining `$defs`/`definitions` references: not implemented, see below.

## Core changes needed (not done, left to the maintainer)

- Optional: `http` exposes a generic `retry(options, f)` (or makes `backoff` public), so `retry_google_request` does not have to copy the backoff calculation.
- Optional: add `as_str()` to `options::GoogleThinkingLevel` returning the wire enum name; for now `google_shared::google_thinking_level_name` is used.

## Improvements pending confirmation

- Inline `$defs`/`definitions` references per provider-references (recursive references become `{}`, unresolvable references stay as is). Upstream sends `parametersJsonSchema` as is, so this is not implemented.
- Fill `response_model` from the response's `modelVersion` (upstream does not).
- Whether in-stream `UNAVAILABLE`/`RESOURCE_EXHAUSTED` errors should be retryable like Anthropic's `overloaded_error` (currently matches upstream: not retried).
- `toolUsePromptTokenCount` is not counted in `input` (`total_tokens` takes the server value, so it may not equal the sum of the parts).
- Below Gemini 3, the "tool result image" turn breaks the merging of function responses: later tool results (including synthesized orphan results) land in a new user turn, same as upstream. Consider placing the image turn after all function responses.
