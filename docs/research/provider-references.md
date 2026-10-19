# Provider implementation references: Zed and litellm-rs

Notes from reading how two Rust projects implement LLM provider clients, used as a
checklist for delta's hand-written API implementations (Anthropic Messages, OpenAI Chat
Completions and Responses, Gemini, Mistral).

Read at: Zed `042e0406` (2026-10-19), litellm-rs `81077510` (2026-10-15).

## Licenses

| Source | License | Use in delta (MIT) |
| --- | --- | --- |
| Zed provider crates (`anthropic`, `open_ai`, `google_ai`, `mistral`, `bedrock`, `language_model_core`, ...) | GPL-3.0-or-later | Study only. Behavior is re-implemented from the wire specs; no code is copied. |
| Zed `http_client`, `cloud_llm_client` | Apache-2.0 | Nothing relevant to provider logic. |
| [litellm-rs](https://github.com/majiayu000/litellm-rs) | MIT | May be adapted with attribution. delta re-implements instead, because the code is tied to gateway types. |

Neither is usable as a dependency: Zed crates are `publish = false` (the crates.io names
`anthropic`, `open_ai`, `mistral` belong to unrelated crates), and litellm-rs is a
gateway-sized crate pinned to reqwest 0.12.

## Behaviors delta adopts

### SSE

- One shared byte-level decoder (`delta_ai::http::SseDecoder`) for every API: `\n`, `\r\n`
  and lone `\r` line ends; multi-line `data:` joined with `\n`; only one leading space
  stripped; `:` comments ignored; `event:` exposed; final event flushed at EOF; bytes
  buffered per line so UTF-8 split across chunks decodes correctly.
- `data: [DONE]` is the end marker on OpenAI-compatible streams and is skipped on
  Anthropic and Gemini streams (proxies inject it).
- A stream that ends without its protocol's terminal event is an error ("stream ended
  unexpectedly"), not a silent success.

### In-stream errors

- Anthropic `event: error` is mapped like an HTTP error of the same type; `overloaded_error`
  mid-stream is retryable.
- OpenAI-compatible chunks: a non-null top-level `error` wins over any payload; a chunk
  with neither `choices` nor `usage` (e.g. `{"detail": ...}`, `{"object": "error"}`) is an
  error, not an empty event.
- Responses: `response.failed` (with `response.error`), bare `type: "error"`,
  `response.incomplete` (`incomplete_details.reason`).
- Gemini: `{"error": ...}` envelopes inside the stream.

### Deserialization

- Every tagged enum of provider events, blocks, deltas and parts has an `Unknown`
  fallback (`#[serde(other)]` or a catch-all); new provider event types must not kill the
  stream.
- Gemini parts are structs with optional fields, not an untagged enum.
- Lenient OpenAI-compatible shapes: `choices` and `index` may be missing or null; tool
  call `index` may be missing, null or `-1`; `function` may be null; usage-only chunks may
  have `choices: []`, null or absent; nulls in `*_tokens_details`.

### Tool calls

- OpenAI-compatible accumulation keyed by (id, index): a late id adopts the anonymous
  call at the same index; a different id at the same index starts a new call; a chunk with
  neither id nor index goes to the only call in flight and is an error when several are in
  flight. Calls keep arrival order (never drain a hash map).
- Names are set only when non-empty; an empty id counts as absent; Mistral's literal
  `"null"` id on continuation chunks counts as absent.
- Accumulated tool calls are emitted on any `finish_reason` and at `[DONE]`/EOF, not only
  on `finish_reason == "tool_calls"` (compat servers send `stop` after tool calls).
- Empty arguments `""` are `{}`; a parse failure at block end is a tool-level error
  (`ToolCall::arguments_error`), not a stream error.
- Streaming previews: a trailing unpaired `\` is dropped before partial parsing.

### Usage

- Anthropic usage merges field by field across `message_start` and `message_delta`:
  present fields overwrite, absent fields keep their value, an explicit 0 replaces.
- OpenAI: cached tokens are split out of `prompt_tokens`; `stream_options.include_usage`
  is sent where supported.
- Gemini: usage is held back until `promptTokenCount` exists; subtraction uses
  `saturating_sub`; output includes `thoughtsTokenCount` (upstream pi-ai behavior).

### Gemini specifics

- `x-goog-api-key` header, never the key in the URL.
- Force `toolUse` when a `functionCall` was seen even if `finishReason` is `STOP`.
- Unknown finish reasons are not hard errors.
- Empty `thoughtSignature` is `None`; a signature on a non-thought text part is kept.
- Thought parts are thinking blocks, never folded into text.
- Synthetic ids for calls without `id` are unique across chunks.
- Tool schemas go out as `parametersJsonSchema`; `$defs`/`definitions` refs are inlined
  (recursive refs become `{}`; an unresolvable ref leaves the schema untouched).

### Anthropic specifics

- Replay: trim trailing whitespace of text and drop empty text; drop unsigned thinking and
  empty redacted thinking.
- Errors: "prompt is too long: N tokens" is context overflow; "credit balance is too low"
  is billing (not retryable); 529 and `overloaded_error` are retryable.

### Requests

- `tool_choice` only when tools are present.
- No `service_tier` or `reasoning_effort: "none"` to endpoints/models that reject them.

### Retries

- Retry 408, 409, 429, 5xx (529 included), connection errors, and in-stream
  overloaded errors; `x-should-retry` overrides. Never retry billing or quota errors.
- Delays: `retry-after-ms`, then `retry-after` in seconds or as an HTTP date, else
  exponential backoff with jitter. Server delays above `max_retry_delay` fail at once.
- A `retry-after` header does not turn an arbitrary status into a rate limit, and no delay
  is guessed from body text.

## Behaviors avoided

- Line-only SSE parsing that ignores `event:` and multi-line data.
- API keys in URLs.
- Hard errors on unknown finish reasons or event types.
- Per-chunk tool indices for Gemini.
