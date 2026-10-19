# API: openai-responses (including the shared Responses layer)

## Purpose

Implements the OpenAI Responses API (`POST {baseUrl}/responses`, SSE streaming). The module has two layers:

- `api::openai_responses_shared`: input item conversion, tool conversion and stream event handling shared by the Responses family. `openai-responses`, `azure-openai-responses` and `openai-codex-responses` (see [api-openai-codex-responses.md](api-openai-codex-responses.md)) use it. Transports that read and map events themselves use `ResponsesEventProcessor`, the incremental form of `process_responses_events`.
- `api::openai_responses`: the unit struct `OpenAIResponsesApi`. It handles compat configuration, the request body, request headers, cache and session affinity, service tier pricing, and `stream` / `stream_simple`.

The client is hand-written on `reqwest` + `delta_ai::http` and does not depend on the OpenAI SDK.

## Upstream source files

- `packages/ai/src/api/openai-responses-shared.ts`: `convertResponsesMessages`, `convertResponsesTools`, `processResponsesStream`, `mapStopReason`.
- `packages/ai/src/api/openai-responses.ts`: `getCompat`, `buildParams`, `createClient`, `stream`, `streamSimple`, service tier pricing.
- `packages/ai/src/api/constrained-sampling.ts`: strict JSON Schema and grammar tools, in the shared module `api::constrained_sampling`.
- `packages/ai/src/api/openai-prompt-cache.ts`: truncates `prompt_cache_key` to 64 code points.
- `packages/ai/src/utils/hash.ts`: `shortHash`, in `api::common::short_hash`, bit-for-bit identical to TypeScript (over UTF-16 code units, `Math.imul` semantics, base-36 output).
- Test references: `test/openai-responses-*.test.ts` (including `openai-responses-chatgpt-sign-in.test.ts` and `openai-responses-usage-limit.test.ts`, ported to `tests/openai_responses_chatgpt.rs`), `test/cache-retention.test.ts` (the Responses part), `test/xai-responses.test.ts` (the request part), and pi-py's `tests/test_openai_responses.py`.

## Request mapping

### URL and request headers

- URL: `join_url(model.base_url, "responses")`.
- Header merge order (later entries override earlier ones, names are case-insensitive, `None` deletes):
  1. API defaults: `Authorization: Bearer <api_key>` (`unused` when there is no key but there is an `authorization` / `cf-aig-authorization` header), `Accept`, `Content-Type: application/json`, `User-Agent: delta/...`;
  2. `model.headers`;
  3. For `provider == "github-copilot"`, Copilot's dynamic headers: `X-Initiator`, `Openai-Intent: conversation-edits`, and `Copilot-Vision-Request: true` when the transcript has an image (see [github-copilot.md](github-copilot.md));
  4. Session affinity headers (see below);
  5. `options.headers`.
- With no API key and no `authorization` / `cf-aig-authorization` header, the error is `No API key for provider: <provider>`.
- Session affinity: sent only when `session_id` is present and cache retention is not `none`. The format comes from `compat.sessionAffinityFormat`. When it is not set, `openrouter` is used if the provider is `openrouter` or the baseUrl contains `openrouter.ai`; otherwise `openai`.
  - `openai`: `session_id` and `x-client-request-id`;
  - `openai-nosession`: only `x-client-request-id`;
  - `openrouter`: `x-session-id`.

### Compat defaults

| Field | Default |
| --- | --- |
| `supportsDeveloperRole` | true (instructions for reasoning models use the `developer` role, otherwise `system`) |
| `supportsMidConvoSystemMessages` | false (when false, later system messages are folded into the initial prompt) |
| `sessionAffinityFormat` | Inferred from provider / baseUrl |
| `supportsLongCacheRetention` | true |
| `supportsStrictMode` | false |
| `supportsOpenAIGrammarTools` | false |
| `supportsAdditionalTools` / `supportsToolSearch` | false |
| `supportsExplicitPromptCacheMode` | false |
| `supportsMaxOutputTokens` | true |
| `supportsMidConvoEffort` | false |

If the model's `compat` does not have the Responses shape (for example custom JSON), it is re-read leniently as the Responses shape; on failure, the defaults are used.

### Request body fields (in insertion order)

| Field | Rule |
| --- | --- |
| `model` | `model.id` |
| `input` | See "Input item conversion" |
| `stream` | `true` |
| `prompt_cache_key` | `session_id` when cache retention is not `none`, truncated to 64 code points |
| `prompt_cache_retention` | `"24h"` when retention is `long`, `supportsLongCacheRetention` is set, and explicit cache mode is not used |
| `prompt_cache_options` | Explicit cache mode only: retention `none` → `{"mode":"explicit"}`; `long` with long cache support → `{"ttl":"30m"}` |
| `store` | `false` |
| `max_output_tokens` | Sent when `max_tokens > 0` and `supportsMaxOutputTokens`; minimum 16 |
| `temperature` | Sent when set |
| `service_tier` | `OpenAIResponsesOptions::service_tier` |
| `tools` | Request-level tools (see "Tool conversion"); not sent when empty |
| `tool_choice` | `OpenAIResponsesOptions::tool_choice` sent as is |
| `reasoning` / `include` | See below |
| Sampling parameters | `resolve_sampling_params(model, level, options.sampling_params)` merged last; existing keys are overwritten in place |

Cache retention: `options.cache_retention`; otherwise `long` when `PI_CACHE_RETENTION=long`; otherwise `short`.

Sign in with ChatGPT: a request with `provider = openai`, `baseUrl = https://api.openai.com/v1` and an API key that does not start with `sk-` carries a ChatGPT access token (from `OpenAIChatGPTOAuth`, see [oauth.md](oauth.md)). Token sharing rejects some fields, so such a request leaves out `prompt_cache_retention`, `prompt_cache_options`, `max_output_tokens` and `temperature`, and never uses `configuration_update` items (the level goes into `reasoning.effort`, and `provider_thinking_level` stays unset). `prompt_cache_key` is still sent.

Reasoning parameters (only when `model.reasoning`):

- The actual effort is `reasoning_effort`; when it is absent and a non-empty `reasoning_summary` is set, it is `medium`.
- With an effort: `reasoning = {effort, summary}`. The effort is mapped through `thinkingLevelMap` (the original value is used when the mapped value is `null`); summary defaults to `auto`. Also `include = ["reasoning.encrypted_content"]`.
- Without an effort: when the provider is not `github-copilot` and `thinkingLevelMap.off` is not `null`, send `reasoning = {effort: thinkingLevelMap.off ?? "none"}`.
- When the provider is `xai`, always `include = ["reasoning.encrypted_content"]`.
- Changing effort mid-conversation (`supportsMidConvoEffort` and an effort was requested): `reasoning.effort` is fixed to `medium`, and the real level is written into a `configuration_update` input item (see below) and recorded in the output message's `provider_thinking_level`. This way the earlier input does not change and the prefix cache stays valid.

### Input item conversion (`convert_responses_messages`)

1. `resolve_transcript` keeps or folds later system messages according to `supportsMidConvoSystemMessages`.
2. `transform_messages` handles cross-model handoff. Tool call id normalization rules:
   - The target provider is not on the allow list (OpenAI: `openai`, `openai-codex`, `opencode`) or the id has no `|`: replace `[^a-zA-Z0-9_-]` with `_` (per UTF-16 code unit), truncate to 64, and strip trailing `_`.
   - Otherwise the id is `call_id|item_id`: `call_id` is normalized as above; an `item_id` from another provider or API becomes `fc_<shortHash>`, otherwise it is normalized; if it does not start with `fc_`, `fc_` is prepended.
3. For each message (`msg_index` counts only non-initial system messages; messages skipped with `continue` are not counted, same as upstream):
   - Initial system message: `{role: developer|system, content: full prompt text}` (omitted when `include_system_prompt` is false).
   - Later system messages: first output tool addition items if needed (see below), then the text rendered by `render_system_message_update`.
   - user: string → one `input_text`; block list → `input_text` / `input_image` (`detail: auto`, data URL); empty block lists are skipped.
   - assistant:
     - thinking: the signature is the JSON of a Responses reasoning item and is parsed back into an input item as is (including `encrypted_content`);
     - text: the message id and phase are parsed from `textSignature` (`TextSignatureV1` JSON or an old plain-string id; only `commentary` / `final_answer` are kept). Without an id, `msg_pi_<i>` is used, and the Nth text block of the same message uses `msg_pi_<i>_<N>`; ids longer than 64 characters become `msg_<shortHash>`;
     - toolCall: grammar tools send `custom_tool_call` (`input` taken from the matching argument), otherwise `function_call` (`arguments` is compact JSON). For the same provider/API but a different model, the item id is dropped; an item id whose prefix does not match the type (`fc_` / `ctc_`) is also dropped. `namespace` is replayed only to the same model;
     - assistant messages with empty output are skipped; when needed, a `configuration_update` is inserted before the message (same model and `provider_thinking_level` recorded).
   - toolResult: `function_call_output` or `custom_tool_call_output`; `call_id` is the part before `|`. Output: a string when there are no images or the model does not support images (`(no tool output)` when the text is blank, `(see attached image)` when there are only images); otherwise an array of `input_text` + `input_image`.
4. Finally, if mid-conversation effort is enabled, append `{type: configuration_update, reasoning: {effort}}`.

Tool addition items (`supportsAdditionalTools` or `supportsToolSearch`, and the tool history has only additions): request-level `tools` holds only the tools of the initial system message, and tools added by later system messages are loaded at their original position:

- `supportsAdditionalTools`: `{type: additional_tools, role: developer, tools}`;
- otherwise `supportsToolSearch`: a `tool_search_call` / `tool_search_output` pair (`execution: client`, `status: completed`), `call_id = pi_tool_load_<shortHash("system:<i>:<names>")>`; tools in the output carry `defer_loading: true`.

### Tool conversion (`convert_responses_tools`)

- Grammar tools (`constrainedSampling.type = grammar` and grammar tools are supported): `{type: custom, name, description, format: {type: grammar, syntax: lark|regex, definition}}`. lark is preferred. The parameter schema must be an object with exactly one required string property; otherwise it is an error.
- Other tools: `{type: function, name, description, parameters, strict?}`.
  - When `constrainedSampling.type = json_schema`, strict mode is supported, and the schema can be converted to the strict subset: `strict = true`, and `parameters` is replaced with the strict schema (all properties required, originally optional properties changed to `anyOf [original schema, null]`, `additionalProperties: false`).
  - When `strict: require` cannot be satisfied, it is an error; `prefer` falls back to non-strict.
  - The `strict` key is sent only when strict mode is supported.

## Stream event mapping

The SSE layer first mimics the OpenAI SDK: `data: [DONE]` stops reading; data that cannot be parsed as JSON is an error; a payload with a non-empty `error` field reports its `error.message` (it is not passed to the callback). Other payloads are passed to `on_provider_stream_event` first, then normalized. Unknown event types and item types are skipped.

| Responses event | Handling |
| --- | --- |
| `response.created` | Record `response_id` |
| `response.output_item.added` (reasoning) | New thinking block, `thinking_start` |
| `response.output_item.added` (message) | New text block, `text_start`; when `phase = final_answer`, temporarily set the stop reason to `stop` |
| `response.output_item.added` (function_call) | New tool call with id `call_id|id`, `toolcall_start` |
| `response.output_item.added` (custom_tool_call) | New tool call with arguments `{<input property>: input}`, `toolcall_start` |
| `response.reasoning_summary_text.delta` / `response.reasoning_text.delta` | `thinking_delta` |
| `response.reasoning_summary_part.done` | `thinking_delta("\n\n")` |
| `response.output_text.delta` / `response.refusal.delta` | `text_delta` |
| `response.function_call_arguments.delta` | Append argument JSON, `toolcall_delta` |
| `response.function_call_arguments.done` | Replace with the full arguments; if it continues the content already received, emit the rest as a `toolcall_delta` |
| `response.custom_tool_call_input.delta` / `.done` | The raw input is re-encoded as a JSON delta `{"<property>":"..."}`, `toolcall_delta`; the input must grow monotonically |
| `response.output_item.done` (reasoning) | Text from summary (joined with `\n\n`), otherwise content, otherwise the accumulated text; the signature is the JSON of the whole reasoning item; `thinking_end` |
| `response.output_item.done` (message) | Text is the concatenation of `output_text` / `refusal`; signature `{"v":1,"id":...,"phase":...}`; `text_end` |
| `response.output_item.done` (function_call / custom_tool_call) | Parse the item's final arguments (`arguments_error` on failure), set `namespace`, `toolcall_end` |
| `response.completed` / `response.incomplete` | Terminal event: backfill the reasoning items' `encrypted_content`, `response_id`, usage, cost, stop reason |
| `response.failed` | Error `<code>: <message>`, or `incomplete: <reason>`, or `Unknown error (no error details in response)`; record `raw_stop_reason` |
| `error` | Error `Error Code <code>: <message>` |

Slots are keyed by the event's `output_index`. Events without `output_index` share one slot (matching the `undefined` key of upstream's `Map`). So when parallel tool calls lack `output_index`, an unfinished call is left and an error is reported, instead of mixing their arguments together and running them.

### Usage and cost

- `input = input_tokens - cached_tokens - cache_write_tokens` (not below 0), `cache_read = cached_tokens`, `cache_write = cache_write_tokens`, `output = output_tokens`, `reasoning = reasoning_tokens` (always set), and `total_tokens` uses the server's `total_tokens` (not recomputed).
- Cost is computed by `calculate_cost`; `openai-responses` then applies a service tier multiplier: `flex` ×0.5, `priority` / `fast` ×2 (×2.5 for `gpt-5.5`). The service tier is taken from the response's `service_tier` first, and from the request when absent.

### Stop reasons

| `status` | Result |
| --- | --- |
| Missing, `completed`, `in_progress`, `queued` | `stop` |
| `incomplete` + `max_output_tokens` | `length` |
| `incomplete` + other reason | `error`, `Response incomplete: <reason>` |
| `incomplete` with no reason | `error`, `Response incomplete without a provider reason` |
| `failed`, `cancelled` | `error` (no message; ends up as `An unknown error occurred`) |
| Other | Error `Unhandled stop reason: <status>` |

`raw_stop_reason` is `status`, or `<status>.<reason>` when there is an incomplete reason. When the result is `stop` and there are tool calls, it becomes `toolUse`.

At stream end: if no terminal event was received, error `OpenAI Responses stream ended before a terminal response event`; if the stop reason is `toolUse` but a tool call did not receive `output_item.done`, error `OpenAI Responses stream completed with an unfinished tool call: <name> (<id>)`.

## Errors and retries

- Sending uses `http::send_with_retry`: retries 408/409/429/5xx and connection errors, honors `retry-after-ms` / `retry-after` and `max_retry_delay`, and can be interrupted by `options.cancel`.
- The HTTP error format matches upstream `formatProviderError(normalizeProviderError(error), "<prefix> API error")`: when the body's `error` is a non-empty object, `<prefix> API error (<status>): <error JSON>`; otherwise `<prefix> API error (<status>): <SDK message>` (for example `... (502): 502 Bad Gateway`). The prefix is `OpenAI` when the provider is `openai`, otherwise the provider id. The raw body comes from `Error::Status.body`. Other errors (in-stream errors, configuration errors) keep their original message.
- An error message that contains `subscription_sharing_usage_limit_exceeded` (from an HTTP error or a `response.failed` event) gets a second line: `Check your ChatGPT usage: https://chatgpt.com/settings/usage`. This applies to every provider of the API, as upstream.
- The `start` event is emitted after the response is accepted and the `on_response` callback has run. Later failures end with an `error` event carrying the content already received. On cancellation the stop reason is `aborted` with the message `Request was aborted`; when the consumer drops the stream, the read loop returns `Error::Aborted` and stops.

## Rust-specific differences

1. Events do not carry a shared `partial`. The temporary `stop` stop reason set by `phase = final_answer` only shows up in the final message; the consumer's partial does not see this intermediate state.
2. `MessageBuilder::end_tool_call` parses arguments from the streamed JSON it has accumulated itself. Upstream uses the full arguments in `output_item.done`. The implementation sets them first, calls `end_tool_call`, and then overwrites them again, so the final message matches upstream. But when the server's final arguments do not continue the streamed content (for example the malformed-arguments test case), the arguments in the `toolcall_end` event are parsed from the streamed content. See "Core changes needed".
3. Function arguments already present in `output_item.added` are emitted as one `toolcall_delta` (upstream only puts them into `partialJson` and emits no event), so the JSON parsed by the builder matches upstream's `partialJson`.
4. `MessageBuilder` skips empty text/thinking/toolcall deltas; upstream emits empty delta events.
5. When the stop reason is `length`, `spawn_stream` calls `end_open_tool_calls` for tool calls still unfinished and emits `toolcall_end`; upstream does not.
6. The HTTP error body is first trimmed and truncated to 4000 characters (`Error::Status.body`): a JSON body longer than 4000 characters cannot be parsed and is treated as text; the `error` object is re-serialized by `serde_json`, so float formatting may differ from `JSON.stringify`.
7. `sanitizeSurrogates` is not needed: Rust strings never contain lone surrogates.
8. When `stream_simple` has no API key, upstream throws synchronously; here an error stream ending with the same message is returned.
9. Constrained sampling is in `api::constrained_sampling` and `shortHash` is in `api::common`; `openai_responses_shared` re-exports `short_hash` and `create_grammar_tool_input_properties`. Prompt cache key truncation is still a public helper of this module.
10. When SSE data is not JSON, the error message format is `Could not parse message into JSON: <reason>: <data>` (the SDK throws the JSON parse exception itself).
11. Usage fields accept floats and round them to integers; negative values are treated as 0.

## Not ported

- Telemetry, the `fetch` option (replaced by `StreamOptions::http_client`).

## Core changes needed

- `events.rs`: add `end_tool_call_with(index: usize, arguments: JsonObject, arguments_error: Option<String>)` to `MessageBuilder` (or a method that replaces the accumulated JSON), so that the `toolcall_end` event carries the final arguments given by the API directly. This removes difference 2.

## Improvements pending confirmation

- An unknown `status` currently reports an error as upstream does (`Unhandled stop reason`). provider-references suggests that an unknown finish reason should not be a hard error; it could become `stop` while keeping `raw_stop_reason`.
- When `function_call_arguments.done` / `custom_tool_call_input.done` lack the argument field, it is currently ignored (upstream throws TypeError); an explicit error could be considered.
- A tool call id with a missing `item.id` becomes `call_x|undefined` (same as upstream and pi-py); using only `call_id` could be considered.
