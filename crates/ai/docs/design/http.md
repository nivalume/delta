# HTTP, SSE and the API task framework (`http`, `json`, `api::common`)

Upstream has no separate HTTP layer: each API uses a vendor SDK, and the SDK handles retries, timeouts and SSE. Hand-written clients must implement these themselves, so they are collected in the `http` module. The behavior follows the policies of the SDKs upstream uses (`src/utils/provider-retry.ts`, `src/utils/error-body.ts`, `src/utils/headers.ts`), and draws on the Zed and litellm-rs implementations (see `docs/research/provider-references.md`).

## Client and request headers

- `http::client(&options)`: `options.http_client`, or the default client of the current Tokio runtime (rustls, 30-second connect timeout, User-Agent `delta/<version> (<os>; <arch>)`). Proxies follow `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` and the system proxy settings.
  The default client is cached per runtime (only the most recent runtime's is kept). Connections in the pool belong to the runtime that opened them and become invalid after that runtime shuts down; reusing them across runtimes makes requests fail at random (for example, each `#[tokio::test]` has its own runtime). When called outside a runtime, a new client is built each time.
- `http::build_headers([..])`: merges several sources in order, with case-insensitive names. Later sources override earlier ones, and `None` removes an earlier header with the same name. Order in API implementations: API default headers → `options.headers`.

## Retries (`send_with_retry`)

Same as upstream `retryProviderRequest` (upstream copies the OpenAI/Anthropic SDK policy and makes the backoff interruptible):

- Retryable: `x-should-retry: true`; or 408, 409, 429, 5xx; or a connection error. `x-should-retry: false` is not retried.
- Count: `max_retries`, default 2.
- Delay: `retry-after-ms`, then `retry-after` (seconds or an HTTP date), otherwise exponential backoff of `min(0.5·2^n, 8)` seconds minus up to 25% jitter.
- If the delay the server asks for exceeds `max_retry_delay` (default 60 seconds, 0 means no limit), fail immediately. The error message includes the requested number of seconds.
- `options.cancel` is honored both during backoff and during the request.
- Request timeout: `timeout`, default 10 minutes (same as the SDK default).

An error response becomes `Error::Status { status, message: "<status> <body message>", body, retry_after_ms }`. The body message is taken from the JSON `error.message` / `message` / `error` / `detail`, otherwise the whole text, truncated to 4000 characters (upstream `MAX_PROVIDER_ERROR_BODY_CHARS`). `body` is the raw body, processed as upstream `normalizeProviderError` does: leading and trailing whitespace trimmed (the JavaScript `trim` whitespace set), `None` for an empty body, truncated to 4000 characters with a note giving the number of characters cut (`http::error_body_text`).

Upstream's HTTP error text comes from the vendor SDK's error object, so each API uses `body` to rewrite `message` in upstream's format:

- `openai-completions`, `openai-responses`, `azure-openai-responses`: the OpenAI SDK `APIError` message plus `formatProviderError` (`api::common::openai_sdk_error`, `api::common::format_provider_error`).
- `anthropic-messages`: the Anthropic SDK `APIError` message, `"<status> <whole JSON body>"` (`api::common::sdk_api_error_message`).
- `google-generative-ai`: the `@google/genai` `ApiError` message, which is the whole JSON body (`google_shared::format_google_http_error`).
- `mistral-conversations` does not go through `send_with_retry`. It builds the message itself in upstream's format `Mistral API error (<status>): <body>`, and also fills in `body`.

Truncation counts characters; upstream counts UTF-16 code units. A JSON body over 4000 characters cannot be parsed after truncation and is treated as text.

## SSE decoding (`SseDecoder`, `sse_events`)

Implemented according to the WHATWG event-stream specification:

- Line terminators `\n`, `\r\n` and a lone `\r` (including `\r` + `\n` split across chunks).
- Bytes are buffered per line before UTF-8 decoding, so multi-byte characters split across chunks decode correctly.
- Multi-line `data:` is joined with `\n`. Only one leading space is stripped from a field value. Comments starting with `:` are ignored. `event` and `id` are exposed.
- An empty line dispatches the event. At end of stream, the last event that did not end with an empty line is flushed.
- `[DONE]` gets no special handling in the decoder; each API decides (OpenAI-compatible streams use it to mark the end; Anthropic and Gemini skip a `[DONE]` injected by proxies).

`sse_events(response, cancel)` turns the response body into a `Stream<Item = Result<SseEvent>>`. Read errors and cancellation are returned as `Err`.

## Streaming JSON (`json`)

Upstream uses the `partial-json` package to parse unfinished tool arguments. Rust hand-writes it (there is no suitable high-quality crate):

- `repair_json`: escapes raw control characters inside strings and doubles the backslash before an invalid escape.
- `parse_json_with_repair`: parses strictly first; on failure, repairs and parses once more.
- `parse_partial_json`: recursive-descent parsing of a JSON prefix. At end of input it closes unfinished strings, arrays and objects. An incomplete number falls back to its longest valid prefix. Literal prefixes such as `tr` are completed. Keys without a value in an object are dropped. Invalid (rather than truncated) input returns `None`.
- `parse_streaming_json`: tries a full parse, a prefix parse and a repaired prefix parse in turn. Returns an empty object when the result is not an object.
- `parse_tool_call_arguments`: for completed tool call arguments. For a non-object, returns empty arguments and an error message that includes the raw text.

## API task framework (`api::common`)

`spawn_stream(model, cancel, run)` runs one API request in a Tokio task and ends the event stream in a uniform way:

- `run` returns `Err`, or the cancellation token has fired: emit `Error` (`aborted` on cancellation, with message "Request was aborted"), keeping the partial content received so far.
- `run` returns `Ok`, but `stop_reason` is error/aborted: emit `Error`, with the message taken from `error_message`.
- Otherwise, close any unfinished tool calls and emit `Done`.
- The consumer has already dropped the stream: emit nothing.

Also provided: callback wrappers (`apply_on_payload`, `notify_response`, `observe_provider_event`), `finalize_usage` (recomputes the total and cost), `require_auth`, `join_url`, and `short_hash` (upstream `shortHash` from `utils/hash.ts`, computed over UTF-16 code units, with output identical bit for bit to TypeScript).

The constrained sampling logic shared by the APIs (upstream `api/constrained-sampling.ts`: strict JSON schema conversion, grammar tools) is in `api::constrained_sampling`. As in upstream, `make_strict_json_schema` and `resolve_json_schema_strict_sampling` accept an optional keyword check (`Option<fn(&str, &Value) -> bool>`). A provider uses it to reject keywords its strict mode does not support, with the error text `<key>: <JSON of the value> is unsupported`. Currently only Anthropic passes a check.
