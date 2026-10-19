# openai-completions API

## Purpose

`openai-completions` implements the OpenAI Chat Completions protocol. It also covers all
OpenAI-compatible endpoints (DeepSeek, z.ai, Moonshot, Together, OpenRouter, Vercel AI Gateway,
Cloudflare, NVIDIA NIM, vLLM, llama.cpp, SGLang, and others). Differences between servers are
described by `OpenAICompletionsCompat` on the model; fields that are not set are detected
automatically from the provider id and the base URL.

Module: `crates/ai/src/api/openai_completions.rs` (unit struct `OpenAICompletionsApi`, which
implements `ProviderStreams`). Strict JSON schema tools and OpenAI grammar custom tools live in the
shared module `api/constrained_sampling.rs`, and `shortHash` is in `api::common::short_hash`. For
compatibility with the old paths, this module re-exports `openai_completions::constrained_sampling`
and `openai_completions::short_hash`. Options are
`ApiOptions::OpenAICompletions(OpenAICompletionsOptions)`.

## Upstream source files

- `packages/ai/src/api/openai-completions.ts` (main body: compat detection, request parameters,
  message conversion, stream handling)
- `packages/ai/src/api/openai-prompt-cache.ts` (truncates `prompt_cache_key` to 64 characters)
- `packages/ai/src/api/constrained-sampling.ts` (strict schema conversion, grammar tools)
- `packages/ai/src/utils/hash.ts` (`shortHash`, used to normalize tool call ids)
- Reference tests: `test/openai-completions-*.test.ts`, `constrained-sampling.test.ts`; pi-py's
  `tests/test_openai_completions.py`.

## Compat resolution

`get_compat(model)` = `detect_compat(model)`, overridden field by field by the fields explicitly set
in `model.compat` (`vllmPriority` only takes an explicit value). The detection rules match upstream:

| Condition | Result |
| --- | --- |
| Non-standard endpoints: NVIDIA, Cerebras, xAI, Together, chutes.ai, DeepSeek, z.ai, Moonshot, OpenCode, Cloudflare Workers AI / AI Gateway, Ant Ling | Do not send `store`; do not use the developer role |
| chutes.ai, DeepSeek, Moonshot, Cloudflare AI Gateway, Together, NVIDIA, Ant Ling, z.ai | Use `max_tokens`; otherwise `max_completion_tokens` |
| xAI, z.ai, Moonshot, Together, AI Gateway, NVIDIA, Ant Ling | `reasoning_effort` not supported |
| OpenRouter | `thinkingFormat=openrouter`; sends the session affinity header (`x-session-id`) by default; only `anthropic/` and `openai/` models use the developer role |
| provider is `openrouter` and the model id starts with `anthropic/` | `cacheControlFormat=anthropic` |
| DeepSeek / z.ai / Together / Ant Ling | The matching `thinkingFormat`; DeepSeek also requires `reasoning_content` on assistant messages |
| Together, Cloudflare, NVIDIA, Ant Ling | Long cache retention not supported |
| Other | `thinkingFormat=openai`, `supportsStrictMode=false` (being OpenAI-compatible does not mean strict tools are supported) |

## Request mapping

`POST {baseUrl}/chat/completions` (when `baseUrl` is empty, `https://api.openai.com/v1` is used,
same as the OpenAI SDK).

Request headers are merged in order (later entries override earlier ones, `None` deletes):
`Accept`/`Content-Type: application/json`, `Authorization: Bearer <key>`, `User-Agent`,
`model.headers`, GitHub Copilot's dynamic headers (only for `provider == "github-copilot"`:
`X-Initiator`, `Openai-Intent: conversation-edits`, `Copilot-Vision-Request: true` when the
transcript has an image; see [github-copilot.md](github-copilot.md)), session affinity headers,
`options.headers`. When there is no API key but the
`authorization` or `cf-aig-authorization` header is non-empty, the key is recorded as `unused` (same
as upstream; the default header can be deleted with `Authorization: None`). Otherwise the stream
fails with `No API key for provider: <p>`.

Session affinity headers are sent only when `sendSessionAffinityHeaders` is set, cache retention is
not `none`, and there is a `session_id`. The `openrouter` format sends `x-session-id`; other formats
send `x-client-request-id` and `x-session-affinity`, and the `openai` format also adds `session_id`.

Request body fields (same order as upstream):

| Field | Source |
| --- | --- |
| `model`, `messages`, `stream: true` | Model id, converted messages |
| `prompt_cache_key` | The base URL contains `api.openai.com` and retention is not `none`, or retention is long and `supportsLongCacheRetention`; the value is `session_id` truncated to 64 characters |
| `prompt_cache_retention: "24h"` | Long retention and `supportsLongCacheRetention` |
| `stream_options.include_usage` | `supportsUsageInStreaming` is not false |
| `store: false` | `supportsStore` |
| `max_tokens` / `max_completion_tokens` | `options.max_tokens` (when greater than 0); the field name is decided by `maxTokensField` |
| `temperature` | `options.temperature` |
| `tools` | Current tools (see below); when there are no tools but the history has tool calls/results, send `tools: []` (needed when LiteLLM proxies Anthropic) |
| `tool_stream: true` | Has tools and `zaiToolStream` |
| `tool_choice` | Raw value from the API options |
| `priority` | `vllmPriority` |
| Reasoning fields | See the table below |
| `thinking_token_budget` / `thinking_budget` / `thinking_budget_tokens` | `thinkingTokenBudgetField` (defaults to the first one when `supportsThinkingTokenBudget`) and there is a budget |
| `provider` | Explicit `openRouterRouting` |
| `providerOptions.gateway` | `only`/`order` of explicit `vercelGatewayRouting` |
| Sampling parameters | Merged last: model `samplingParams`, per-reasoning-level overrides, request `sampling_params`; they overwrite fields with the same name |

Cache retention: `options.cache_retention`; otherwise the environment variable
`PI_CACHE_RETENTION=long`; otherwise `short`.

Reasoning fields (only when `model.reasoning`; `effort` is the requested reasoning level, `map[x]` is
`thinkingLevelMap`):

| `thinkingFormat` | Fields |
| --- | --- |
| `zai` | `thinking: {type: enabled, clear_thinking: false}` or `{type: disabled}`; `reasoning_effort` when supported (the level name when unmapped; not sent when mapped to null) |
| `qwen` | `enable_thinking`; `reasoning_effort = map[effort] ?? effort` when supported |
| `qwen-chat-template` | `chat_template_kwargs: {enable_thinking, preserve_thinking: true}` |
| `chat-template` | `chat_template_kwargs`: literals as is; `$var` is `thinking.enabled` (boolean), `thinking.effort` (mapped value or level name), `thinking.budget` (budget after clamping); `omitWhenOff` omits it when reasoning is off |
| `baseten` | `chat_template_args` (same as above), `reasoning_effort` when supported (`map.off` when off) |
| `deepseek` | `thinking: {type: enabled}`, `{type: disabled}` when off (not sent when `map.off` is null); `reasoning_effort` when supported |
| `openrouter` | `reasoning: {effort}`, `{effort: map.off ?? "none"}` when off (not sent when `map.off` is null) |
| `ant-ling` | Sends `reasoning: {effort}` only when `map[effort]` is a string |
| `together` | `reasoning: {enabled}`; `reasoning_effort` when supported |
| `string-thinking` | `thinking: map[effort] ?? effort`, `map.off ?? "none"` when off |
| `openai` | `reasoning_effort = map[effort] ?? effort` when supported; when off, sent only if `map.off` is a string |

Thinking budget: computed only when there is a reasoning level. `xhigh`/`max` count as `high`. The
upper limit is `max_tokens` (the model's `maxTokens` when not given) minus 1024. Nothing is sent when
the result is 0.

### Tools

`convert_tools`: when `supportsOpenAIGrammarTools` is set and a tool declares a grammar constraint,
send `{type: custom, custom: {name, description, format: {type: grammar, grammar: {syntax, definition}}}}`
(lark first, then regex; error when no variant is usable or the parameter schema is not "exactly one
required string property"). Otherwise send a function tool. With `supportsStrictMode` it carries
`strict`; when the `json_schema` constraint can be converted, it is `true` and the strict schema is
sent (all properties required, optional properties changed to `anyOf [..., null]`,
`additionalProperties: false`). When `require` cannot be satisfied, the request fails.

When `supportsMidConvoSystemMessages` and `supportsMidConvoToolAdditions` are set and the tool
history has only additions, top-level `tools` sends only the initial tools. Tools added by later
system messages are inserted at their original position as `{role: system, tools: [...]}` (Kimi
style). Otherwise `tools` is the full current tool list.

### Message conversion

First `resolve_transcript` (merged into one message when mid-conversation system messages are not
supported), then `transform_messages`. Tool call id normalization: a Responses-style id containing
`|` is joined as `{call}_{item}` (anything not in `[a-zA-Z0-9_-]` is replaced with `_` per UTF-16
unit). When longer than 40 characters, it becomes a prefix plus the first 8 characters of
`shortHash`. Other ids for provider `openai` are truncated to 40 UTF-16 units.

- System messages: the first one sends the full text (including sections); later ones send
  `render_system_message_update`. The role is `developer` for reasoning models with
  `supportsDeveloperRole`, otherwise `system`.
- User messages: strings as is; block lists drop empty text, images become `image_url` data URLs;
  skipped when everything is empty.
- assistant: text blocks (blank text removed) are joined into a string `content`; with no content it
  is `null` (`""` with `requiresAssistantAfterToolResult`). Thinking: with `requiresThinkingAsText`,
  it goes first as a text part. Otherwise, if the signature of the first thinking block is
  `reasoning_content`/`reasoning`/`reasoning_text`, it is replayed in that field (opencode-go's
  `reasoning` is changed to `reasoning_content`). When the signature is `reasoning_details` JSON,
  `reasoning_details` is replayed (taking precedence over the raw field); encrypted details in a tool
  call's `thoughtSignature` from old transcripts are also replayed. With
  `requiresReasoningContentOnAssistantMessages`, `reasoning_content: ""` is added. Grammar tool calls
  are replayed as `{type: custom, custom: {name, input}}`. Messages with neither content nor tool
  calls are skipped.
- Tool results: consecutive results are sent one by one as `role: tool` (when the text is empty,
  `(see attached image)` or `(no tool output)` is used; with `requiresToolResultName` they carry
  `name`). When the model supports images, the images are merged after this batch of results into
  one user message `Attached image(s) from tool result:`.
- `requiresAssistantAfterToolResult`: when a user message directly follows tool results,
  `I have processed the tool results.` is inserted.
- `cacheControlFormat=anthropic` and retention is not `none`: add `cache_control: {type: ephemeral}`
  to the first system/developer message, the last tool, and the last user/assistant/tool message
  with text (with `ttl: "1h"` for long retention when supported). String content is first converted
  to a single text part.

## Stream event mapping

Each SSE `data:` is one chunk. `[DONE]` stops reading, empty data is skipped, and JSON that cannot be
parsed fails the request. `on_provider_stream_event` is called for every parsed chunk (including
`null`), except error chunks.

| Chunk content | Handling |
| --- | --- |
| HTTP response accepted | `on_response`, then `start` |
| Non-empty `error` | Request fails: `error.message` (JSON-encoded when not a string; the whole error JSON-encoded when there is no message); `error.metadata.raw` is appended on a new line when it is not already in the message |
| No `choices`/`usage`/`id`, but has `detail`, `message` or `object: "error"` | Request fails; the message is `detail`/`message` |
| `id` | The first non-empty value becomes `response_id` |
| `model` | Becomes `response_model` when non-empty and different from the requested model id (only the first one) |
| `usage` (or `choices[0].usage`, Moonshot) | Overwrites usage: cache read from `prompt_tokens_details.cached_tokens` → `prompt_cache_hit_tokens` → `cached_tokens`; cache write from `prompt_tokens_details.cache_write_tokens`; `input = prompt - read - write`; `reasoning = completion_tokens_details.reasoning_tokens` (default 0); cost from the model's pricing |
| `finish_reason` | `raw_stop_reason`; `stop`/`end` → stop, `length` → length, `tool_calls`/`function_call` → toolUse, other → error (`Provider finish_reason: <x>`) |
| `delta.reasoning_content` / `reasoning` / `reasoning_text` | The first non-empty field → `thinking_start` (signature is the field name)/`thinking_delta` |
| `delta.reasoning_details` | Valid entries are kept in memory (adjacent text/summary entries merged, encrypted entries kept separate); a thinking block is started when needed; when the block ends, they are JSON-encoded into the thinking signature |
| `delta.content` | `text_start` / `text_delta` (thinking is handled before text) |
| `delta.tool_calls[]` | Find the call by `index` first, then by `id`. If neither matches and there is only one call, it goes to that call; with several calls the request fails; if none is found, `toolcall_start`. A late id is filled in for an anonymous call; the name is set only when non-empty. `function.arguments` → `toolcall_delta`; `custom.input` is rendered as incremental JSON `{"<property>":"<escaped input>"}` |
| Stream end | `text_end`/`thinking_end`/`toolcall_end` in content order (argument parse failures are recorded in `arguments_error`) |

End rules: cancelled → aborted; no `finish_reason` and `supportsFinishReason=false` → toolUse if
there are tool calls, otherwise stop; stop reason is error → fail with its message;
`supportsFinishReason` but no `finish_reason` received → `Stream ended without finish_reason`. Tool
calls are emitted whatever the finish reason is.

## Errors and retries

- Requests go through `http::send_with_retry`: 408/409/429/5xx and connection errors are retried,
  `retry-after-ms` / `retry-after` are honored, and the request fails immediately when the delay
  exceeds `max_retry_delay`. Cancellation takes effect at any time.
- HTTP error messages match upstream. First an `APIError` is built as in the OpenAI SDK (message
  `"<status> <error.message>"`; when the body has no `error` object, `"<status> <text>"` or
  `"<status> status code (no body)"`). Then `formatProviderError` is applied: when the body's `error`
  is a non-empty object not already contained in the message, the result is
  `"<status>: <error JSON>"`, otherwise the SDK message. OpenRouter's `error.metadata.raw` is
  appended on a new line when it is not in the message. The raw body comes from
  `Error::Status.body`; the helpers are in `api::common` (`openai_sdk_error`,
  `format_provider_error`).
- On failure (including before an HTTP error, in-stream errors, and cancellation) the stream ends
  with an `error` event and keeps the content already received; collected `reasoning_details` are
  still written into the thinking signature.
- Cancellation: the read loop ends, open blocks end normally, and then the stream fails with
  `Request was aborted` (aborted). The event order matches upstream (the SDK ends the iteration
  silently on cancellation).

## Rust-specific differences

- Events are produced through `MessageBuilder`: an empty `toolcall_delta` is not emitted (upstream
  also emits an empty delta on chunks that carry only id/name); the `thinking_start` event carries
  the signature.
- The arguments of grammar tool calls come from the builder parsing the streamed JSON; the result is
  the same as upstream `{property: input}`. If the same call starts with function deltas and later
  switches to custom (upstream drops the existing function argument buffer), the JSON already
  accumulated in the builder cannot be cleared, and argument parsing may fail.
- The HTTP error body is first trimmed and truncated to 4000 characters (`Error::Status.body`): a
  JSON body longer than 4000 characters cannot be parsed and is treated as text; the `error` object
  is re-serialized by `serde_json`, so float formatting may differ from `JSON.stringify` (for
  example `1.0`). Connection error texts come from reqwest, not the SDK's `Connection error.`.
- The default retry count is decided by the core (`DEFAULT_MAX_RETRIES = 2`); upstream
  `retryProviderRequest` does not retry by default.
- The OpenAI SDK's `X-Stainless-*` headers are not sent; `User-Agent` is the core's
  `delta/<version> (...)`.
- Rust strings do not contain lone surrogates, so `sanitizeSurrogates` does not need to be ported.
- JavaScript semantics follow upstream: the whitespace set of `trim()` (includes `\u{feff}`,
  excludes `\u{85}`), `shortHash` over UTF-16 units, and `??=`/`||=` assignments to the
  `reasoning_details` field (the field is deleted when the source is missing).
- Integer-valued `f64` values (such as `vllmPriority`, `temperature`) are serialized as integers,
  matching `JSON.stringify`.
- Objects in `ChatTemplateKwargValue::Literal` are sent as is (upstream treats any object as a `$var`
  reference).
- `ReasoningEffort::None` / `Other(s)` are handled as strings: reasoning counts as on, the name is
  looked up in `thinkingLevelMap`, names that are not levels get no thinking budget, and sampling
  parameters resolve at the `off` level. This is the same as upstream when given a non-standard
  string.
- Not ported: telemetry.

The following behaviors come from `docs/research/provider-references.md`. They only make the client
more tolerant of invalid input or give clearer errors, and do not change the result for valid input:

- A negative tool call `index` is treated as missing (upstream would merge all calls under `-1`).
- A delta with neither id nor index goes to the only call in progress; with several calls it is an
  error (upstream would create a new anonymous call).
- When `function.arguments` is not a string, it is appended as JSON text (upstream would produce
  `[object Object]`).
- Error packets without completion fields (`{"detail": ...}`, `{"object": "error"}`) are reported as
  errors directly, instead of ending with `Stream ended without finish_reason`.

Note: the provider-references rule "start a new call when a different id appears on the same index"
conflicts with upstream. An upstream test (Kimi changes the id mid-stream) requires merging by
index; this module follows upstream.

## Improvements pending confirmation

- Whether in-stream `overloaded`/`529` errors should be treated as retryable (upstream does not
  retry).
- Whether the provider-references rule "a different id on the same index is a new call" should apply
  to some providers (it conflicts with upstream's Kimi behavior and is not adopted for now).
