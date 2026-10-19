# API design: bedrock-converse-stream

## Purpose

`bedrock-converse-stream` streams models on Bedrock (Claude, Nova, OpenAI GPT/gpt-oss, etc.) through the Amazon Bedrock `ConverseStream` endpoint,
and converts the Bedrock event stream into delta's unified `AssistantMessageEvent` protocol.

Per `AGENTS.md`, Bedrock uses the official AWS Rust SDK (`aws-sdk-bedrockruntime` + `aws-config`); SigV4 and event stream parsing are not hand-written.
The whole module is behind the `bedrock` feature (`#[cfg(feature = "bedrock")]`); when it is off, the AWS SDK is not compiled.

Module: `crates/ai/src/api/bedrock_converse_stream.rs`, with the submodules `bedrock_converse_stream/proxy.rs` (proxy resolution) and
`bedrock_converse_stream/http_client.rs` (the HTTP clients handed to the SDK). Unit tests are in `bedrock_converse_stream/tests.rs` and
`proxy.rs`, end-to-end tests in `crates/ai/tests/bedrock_converse_stream.rs`.

Public interface:

- `BedrockConverseStreamApi` (unit struct, implements `ProviderStreams`);
- `stream(model, context, StreamOptions)`;
- `stream_simple(model, context, SimpleStreamOptions)`.

API-specific options come from `ApiOptions::BedrockConverseStream(BedrockOptions)` in `options.api`; other variants are ignored.
Compat settings come from `ModelCompat::Bedrock` (currently only `supportsStrictMode`).

## Upstream source files

- `packages/ai/src/api/bedrock-converse-stream.ts` (main body, 1416 lines).
- `packages/ai/src/api/constrained-sampling.ts` (strict tool schema conversion, in the shared module `api::constrained_sampling`).
- `packages/ai/src/utils/node-http-proxy.ts` (proxy resolution, in `bedrock_converse_stream/proxy.rs`; upstream also uses it for the
  OpenAI Codex WebSocket under Bun, which delta does not port).
- `packages/ai/src/api/bedrock-converse-stream.lazy.ts`, `src/bedrock-provider.ts`: only lazy-load Node-only modules.
  In Rust a feature flag replaces them; there is no matching code.
- Tests: `packages/ai/test/bedrock-*.test.ts`, `node-http-proxy.test.ts`; the Python port `pi_ai/api/bedrock_converse_stream.py` as a second reference.

## Request mapping

### Flow

1. `stream()` first uses `collapse_system_messages` to merge mid-conversation system messages into the system prompt at the start (Bedrock does not support mid-conversation system messages).
2. Resolve the client configuration (region, endpoint, profile, credentials, bearer token); see the next section.
3. Build the request body as JSON, in the same shape as the input of upstream `ConverseStreamCommand` (camelCase members; binary members such as image bytes and encrypted reasoning are base64 strings).
4. Call `on_payload`; the callback can replace the whole JSON.
5. Convert the JSON into the SDK's strongly typed input (`ConverseRequest::from_payload`). Top-level members the SDK does not know are ignored, matching how the JS SDK serializes commands.
6. Build the client with `aws_config` and the HTTP client chosen by the transport rules below, send, and read the event stream.
   The transport is resolved right after the client configuration, before the payload is built (same order as upstream).

### Client configuration (matches the upstream Node branch)

| Item | Rule |
| --- | --- |
| profile | `BedrockOptions::profile`, otherwise `AWS_PROFILE` in `options.env` ("explicit profile"), otherwise `AWS_PROFILE` in the process environment |
| region | Region in the inference profile ARN > `BedrockOptions::region` > `AWS_REGION` > `AWS_DEFAULT_REGION` > region in the standard endpoint (only when the endpoint is pinned) > `us-east-1` when there is no environment `AWS_PROFILE`; otherwise left to the SDK to resolve from the profile |
| endpoint | Always used when `model.base_url` is not a standard `bedrock-runtime[-fips].<region>.amazonaws.com[.cn]`; a standard endpoint is pinned only when neither a region nor an environment `AWS_PROFILE` is configured |
| Static credentials | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (+ `AWS_SESSION_TOKEN`), looked up in `options.env` first, then the process environment; not used when there is an explicit profile (upstream #6957) |
| bearer token | `BedrockOptions::bearer_token` > `options.api_key` > `AWS_BEARER_TOKEN_BEDROCK`; when set, `auth_scheme_preference = httpBearerAuth` and no SigV4 |
| `AWS_BEDROCK_SKIP_AUTH=1` | Use dummy static credentials, no bearer token, and pin the auth scheme to SigV4 (the SDK also reads `AWS_BEARER_TOKEN_BEDROCK` itself, so it has to be overridden) |

Environment variable lookup: values in `options.env` take precedence, then the process environment; blank values count as unset.

### Transport: proxies and HTTP/1.1

Upstream sets `requestHandler` on the client config; delta sets the SDK's `http_client`:

| Condition (checked in this order) | HTTP client |
| --- | --- |
| A proxy applies to `model.base_url` | HTTP/1.1 only, every request through that proxy (upstream: `NodeHttpHandler` with `HttpProxyAgent`/`HttpsProxyAgent`) |
| `AWS_BEDROCK_FORCE_HTTP1=1` | HTTP/1.1 only, no proxy (upstream: `NodeHttpHandler`); for custom endpoints that require HTTP/1.1 |
| otherwise | The SDK's default HTTPS client (HTTP/2 or HTTP/1.1 through ALPN), with its process-environment proxy lookup turned off |

Proxy resolution (`resolve_http_proxy_url_for_target`, port of `node-http-proxy.ts`):

- The target is `model.base_url`, also when the SDK derives the endpoint from the region. An empty or unparsable base URL, or one without a host, gets no proxy.
- Each name is looked up as lowercase in `options.env`, uppercase in `options.env`, lowercase in the process environment, uppercase in the process environment.
- `NO_PROXY`: `*` disables proxies. Otherwise entries are split on commas and whitespace and compared case-insensitively.
  An entry matches the host itself and its subdomains; a leading `.`, `*` or `*.` is dropped. `host:port` and `[ipv6]:port` entries match only that port
  (the target's port defaults to 80/443 by scheme); bare IPv6 addresses (`::1`) and bracketed ones are accepted. CIDR ranges are not supported.
- The proxy is `<scheme>_proxy` of the target (`HTTPS_PROXY` for `https://`), else `ALL_PROXY`. A value without `://` gets the target's scheme as prefix.
- An unparsable proxy fails with `Invalid proxy URL "<value>": Invalid URL`; a scheme other than `http`/`https` fails with
  `Unsupported proxy protocol. SOCKS and PAC proxy URLs are not supported; use an HTTP or HTTPS proxy URL. Got <scheme>:`.
- `https://` targets go through `CONNECT`; `http://` targets are sent to the proxy in absolute form. Credentials in the proxy URL become
  `Proxy-Authorization: Basic ...`. `NO_PROXY` is applied once, to the base URL; the chosen proxy then carries every request of the client.

### Request body fields

| Field | Source |
| --- | --- |
| `modelId` | `model.id` |
| `messages` | `transform_messages` (tool call ids normalized to `[a-zA-Z0-9_-]`, at most 64, length counted in UTF-16), then converted per the rules below |
| `system` | Full text of the leading system message (including sections); `cachePoint` appended for models that support caching |
| `inferenceConfig` | `maxTokens`: `options.max_tokens`, defaulting to `model.max_tokens` for Claude models; `temperature` |
| `toolConfig` | Current tool set; omitted when there are no tools or `tool_choice = None`; `toolChoice` is `auto` / `any` / `tool{name}` |
| `additionalModelRequestFields` | Reasoning config, see below |
| `requestMetadata` | `BedrockOptions::request_metadata` |

Message conversion:

- User messages: string content that is whitespace-only sends `<empty>`; in block content, whitespace-only text blocks are dropped, and if all are dropped, send `<empty>`. Images map by MIME type to `jpeg/png/gif/webp`;
  other types fail with `Unknown image type: ...`. base64 decoding is as lenient as the browser's `atob` (tolerates whitespace and missing padding).
- Assistant messages: messages with empty content are skipped; whitespace-only text is skipped; empty key names are removed from tool call arguments (Bedrock rejects empty property names; this only affects the copy that is sent);
  thinking: redacted blocks are base64-decoded and replayed as `reasoningContent.redactedContent` (the block is dropped if decoding fails);
  Claude models with a signature send `reasoningText{text, signature}`, and without a signature fall back to plain text; non-Claude models send only `reasoningText{text}`.
  When all blocks are filtered out, the whole message is skipped.
- Tool results: consecutive tool results are merged into one user message; empty content sends `<empty>`. For Claude models, images in error results are split out with
  `split_error_tool_result_images` and placed after all tool results of the turn (upstream #10794); other models keep them in place.
- Cache points: when the cache retention policy is not `none` and the model supports prompt caching (Claude 3.5 Haiku, 3.7 Sonnet, 4.x, 5, or `AWS_BEDROCK_FORCE_CACHE=1`),
  append `{"cachePoint": {"type": "default"}}` to the end of the system prompt and of the last user message, with `"ttl": "1h"` for `long`.
  The cache policy defaults to `short`, and is `long` when `PI_CACHE_RETENTION=long`.

Strict tools: when `compat.supportsStrictMode` is true and a tool declares `constrainedSampling: json_schema`, convert the schema to the strict subset and set `strict: true`.
When it cannot be converted, `prefer` falls back to a normal tool and `require` fails; when the model does not support strict mode, `require` fails.

### Reasoning (`additionalModelRequestFields`)

Sent only when `BedrockOptions::reasoning` has a value and `model.reasoning` is true. The model family is matched on id and name (so that application inference profile ARNs work).

- Claude, adaptive thinking models (Opus 4.6/4.7/4.8/5, Sonnet 4.6/5, Haiku 5, Fable 5):
  `thinking: {type: "adaptive", display, block_binding?}`, `output_config: {effort}`;
  models that support block binding (other than 4.6) add `block_binding: {prefix_mismatch_behavior: "drop_block"}` and
  `anthropic_beta: ["thinking-binding-controls-2026-08-01"]`.
  effort: `xhigh` is used directly on models that support it natively; otherwise look up `thinking_level_map`; otherwise minimal/low→low, medium→medium, everything else→high.
- Claude, budget thinking models: `thinking: {type: "enabled", budget_tokens, display}`. The budget comes from custom budgets (xhigh/max use high), otherwise the default table;
  with `interleaved_thinking` (default true), add `anthropic_beta: ["interleaved-thinking-2025-05-14"]`.
- `display` defaults to `summarized`. GovCloud (region starts with `us-gov-`, or id starts with `us-gov.` / `arn:aws-us-gov:`) sends neither display nor block binding.
- gpt-oss: `reasoning_effort` (low/medium/high); other GPT: `reasoning: {effort}` (`thinking_level_map` takes precedence).

### `stream_simple`

Same as upstream `streamSimple`: `build_base_options` + `tool_choice` (Auto/None map to Bedrock's auto/none).
Without reasoning, no reasoning fields are sent. Claude adaptive models and non-Claude models pass `reasoning` and custom budgets through directly.
Claude budget models compute `max_tokens` with `adjust_max_tokens_for_thinking`, then truncate to the context window,
and set the budget for that level to `min(budget, max(0, max_tokens - 1024))`.

### Custom request headers

`options.headers` are merged case-insensitively by name (`None` removes). The reserved headers `x-amz-*`, `authorization` and `host` are skipped.
They are written into the request through the SDK's `mutate_request` (which runs before signing), so they take part in SigV4 signing. When there are no headers to write, no interceptor is registered.
`options.headers` already include the model headers merged by `Models`.

## Stream event mapping

| Bedrock event | Handling |
| --- | --- |
| `messageStart` | role must be assistant, otherwise fail; `builder.start()` |
| `contentBlockStart.toolUse` | `start_tool_call(toolUseId, name)`, record the Bedrock block index |
| `contentBlockDelta.text` | If no block is open at that index, first `start_text` (text blocks have no start event), then `text_delta` |
| `contentBlockDelta.toolUse` | `tool_call_delta` for the matching tool block |
| `contentBlockDelta.reasoningContent.text` | If there is no block, `start_thinking` (initial signature is the empty string, matching upstream), then `thinking_delta` |
| `contentBlockDelta.reasoningContent.signature` | Appended to `thinking_signature` for non-redacted blocks |
| `contentBlockDelta.reasoningContent.redactedContent` | On first appearance, mark the block as redacted, clear the signature, and append the placeholder text `[Reasoning redacted]` (one `thinking_delta`); bytes accumulate in a temporary buffer |
| `contentBlockStop` | Text: `end_text`; thinking: first encode the accumulated bytes as base64 into the signature, then `end_thinking`; tool: `end_tool_call` (parse the full arguments); the block's Bedrock index is cleared |
| `messageStop` | Record `raw_stop_reason`; `end_turn`/`stop_sequence`→stop, `max_tokens`/`model_context_window_exceeded`→length, `tool_use`→toolUse, others→error with `Provider stopped with: <reason>` |
| `metadata.usage` | input/output/cacheRead/cacheWrite; `cacheWrite1h` is the total with ttl=1h in `cacheDetails` (not set when there are no details); `total_tokens` uses Bedrock's totalTokens, or input+output when it is 0; cost is computed from the model's pricing (1h cache writes at 2x the input price) |
| Unknown event/delta type | Ignored |
| In-stream exceptions (`internalServerException` and the other four) | First handed to `on_provider_stream_event`, then ended as an error |

Before processing, each event is passed to `on_provider_stream_event` in the upstream JS SDK shape (e.g. `{"contentBlockDelta": {...}}`, binary as base64).

After the stream ends: cancelled→aborted; stop reason still pending→`Bedrock stream ended without a stop reason`; stop reason is error→fail with its message.
On both success and failure, encrypted reasoning still in the buffer is written back to the signature (the stream may end without sending `contentBlockStop`).

## Errors and retries

- Error messages match upstream `formatBedrockError`: Bedrock exceptions get a readable prefix (`Validation error:`, `Throttling error:`, `Internal server error:`,
  `Model stream error:`, `Service unavailable:`; others use the exception name itself, and `Unknown` when the SDK cannot identify it).
  When the server gives no message, use `"<status>: <body>"` (truncated to 4000 characters), so a gateway 403 does not become `Unknown: UnknownError`.
  When the message contains "data retention mode", a hint with a documentation link is appended.
- Diagnostics: on failure (not abort), append a `bedrock_response_failure` diagnostic. `details` contains only known fields: `status`, `errorCode` (ending in `Exception`,
  at most 200 characters), `requestId` (from the error response headers, or the successful response's `x-amzn-requestid` as a fallback for in-stream errors). `error_message` is not affected.
- `on_response`: an interceptor records the status and all response headers of every HTTP response before deserialization. After the request succeeds, it is called once for the last (successful) response,
  so custom gateway headers are visible too.
- Retry: the SDK's standard retry policy, `max_attempts = max_retries + 1` (3 attempts by default, same as the JS SDK default).
- Timeout: `options.timeout` (default 10 minutes) is used as the SDK's operation timeout (covering the send and all retries), and as the idle timeout of the event stream (the longest wait between two events).
  The SDK's stalled-stream protection is turned off so that long reasoning pauses are not misjudged.
- Cancellation: `options.cancel` is `select`ed against SDK construction, `send()` and every `recv()`. After cancellation, dropping the future interrupts the request. The same happens when the consumer drops the stream.
- On failure, partial content received so far is kept; unfinished tool calls keep a best-effort parse of the JSON streamed in so far (matching upstream's partial message).

## Rust-specific differences

- Uses the official AWS SDK instead of upstream's JS SDK: the request is first built as JSON (for `on_payload`) and then converted to SDK types. In the JSON returned by `on_payload`,
  members the SDK does not support are ignored, and malformed members fail with `Invalid Bedrock payload: ...`.
- The SDK's `Document` uses `HashMap`, so object key order on the wire for tool schemas, tool arguments and `additionalModelRequestFields` is not fixed (upstream JS keeps insertion order).
- Rust's default credential chain always reads environment variables first. To keep upstream's "explicit profile wins over environment keys" semantics, when there is a profile and no static credentials,
  `ProfileFileCredentialsProvider` is used directly (covers SSO, credential_process, assume role, but not ECS/IMDS).
- `aws-config` does not export `AuthSchemeId`, so the auth scheme preference is built through type inference (`scheme_preference`).
- Rust strings cannot contain unpaired surrogates, so upstream's `sanitizeSurrogates` is a no-op and is not ported.
- In-stream modeled exceptions (such as `throttlingException`) are typed errors in the Rust SDK, so diagnostics include `errorCode`; in upstream tests such exceptions are bare objects with only `requestId`.
- Upstream's region fallback branch outside Node (browsers) does not apply and is not ported.
- The HTTP/1.1 client runs on `reqwest` (`http1_only`, no redirects, the SDK's connect timeout), because the SDK's own HTTP client has no public way to restrict
  ALPN to `http/1.1`. Differences from the SDK's client: TLS uses delta's `reqwest` stack (rustls with the platform verifier) instead of rustls with aws-lc and the
  native root store, and `reqwest` adds `Accept: */*` when the request has no `Accept` header. The SDK's read timeout covers the time until the response head, as in
  the SDK's own connector.
- The default transport is the SDK's hyper client, which offers HTTP/2 and HTTP/1.1 through ALPN. Upstream's default `NodeHttp2Handler` speaks only HTTP/2
  (prior knowledge on `http://` endpoints); delta speaks HTTP/1.1 to `http://` endpoints. The SDK client's own `HTTP(S)_PROXY`/`NO_PROXY` lookup is turned off
  so that only the upstream rules decide about proxies.
- Upstream throws proxy errors outside its `try` block, so the stream never ends (an unhandled rejection). delta ends the stream with the error message.
- Whitespace-only proxy variables count as unset, like every other variable of this module; upstream treats them as set and fails on them.
- `aws-config`'s credential providers (SSO, STS, IMDS) keep the SDK's default HTTP client, which reads proxy settings from the process environment.
  Upstream's credential providers use Node's default handlers, which do not.
- `options.http_client` (reqwest) has no effect for this API.
- `max_retry_delay` is not mapped: the SDK's backoff policy does not read `retry-after` and has no matching concept.
- When a `thinking` block becomes redacted mid-stream, `MessageBuilder` has no matching event, so the consumer's partial message does not see `redacted: true` until the end event; the final message is correct.

## Tests

- Unit tests (`bedrock_converse_stream/tests.rs`, 55 of them): client configuration (credential precedence, endpoint/region resolution, ARN, bearer token, skip auth,
  transport selection, proxy errors, redacted proxy credentials in `Debug`),
  request body (strict tools, tool choice, metadata, placeholder text, splitting images from error results, empty key names, thinking replay, redacted replay, cross-model replay and orphaned tool calls,
  cache points, every branch of the reasoning fields), JSON→SDK conversion, event handling (text, parallel tool calls, signatures, redacted concatenation, no stop event, usage and 1h cache cost),
  error formatting and diagnostics, budget mapping in `stream_simple`.
- Proxy resolution tests (`bedrock_converse_stream/proxy.rs`, 15 of them): the five cases of `node-http-proxy.test.ts`, plus `NO_PROXY=*`, scoped `NO_PROXY`,
  lowercase-before-uppercase, `ALL_PROXY` fallback, scheme-less proxies, default ports in `NO_PROXY`, invalid proxy URLs, host-less targets, blank entries and `parseInt`.
- End-to-end tests (`tests/bedrock_converse_stream.rs`, 23 of them): a real SDK client connects to wiremock, and the response body is event stream frames encoded with `aws-smithy-eventstream`.
  They cover text + usage + response headers, thinking + parallel tool calls, redacted round trip, wire request body shape, `on_payload` replacement,
  custom headers in signing, bearer token, HTTP service errors, gateway errors, 5xx retry, in-stream exceptions, error stop reason, missing stop reason, cancellation, timeout, cross-model replay.
  Transport tests use local servers only: `AWS_BEDROCK_FORCE_HTTP1` with response headers (port of `bedrock-response-headers.test.ts`), an `http://` target through a
  scoped proxy (wiremock answers the absolute-form request; `Proxy-Authorization` from the proxy URL), `NO_PROXY` bypass, an `https://` target through a `CONNECT` proxy,
  and the ALPN offer read from the TLS ClientHello (`http/1.1` only through a proxy and with `AWS_BEDROCK_FORCE_HTTP1`, `h2` + `http/1.1` by default), and an unsupported proxy scheme.
- Dependencies of the `bedrock` feature for the transports: `aws-smithy-http-client` (the SDK's HTTPS client without the environment proxy lookup) and
  `aws-smithy-runtime-api` (the `HttpClient`/`HttpConnector` traits for the `reqwest` client); `aws-smithy-types` gains the `http-body-1-x` feature. All were already in `Cargo.lock`.
- New dev dependency: `aws-smithy-eventstream = "0.61"` (in the dev-dependencies of the workspace and `crates/ai`), used only to encode event stream frames in tests.

## Improvements pending confirmation

- Cache `SdkConfig`/`Client`: upstream creates a new client for each request (kept the same here), so the credential chain (SSO, assume role, IMDS) is resolved again each time; pi-py caches clients by configuration.
- Unstable `Document` key order may affect prompt cache hits on the Bedrock side (tool definitions are in the cache prefix). Options: rewrite the request body with order-preserving JSON before signing, or wait for the SDK to support ordered documents.
- Accept CIDR ranges (`10.0.0.0/8`) in `NO_PROXY`, as the SDK's and `reqwest`'s own proxy matchers do. Upstream compares them as host names, so they never match.
- Move `proxy.rs` to a shared module once another API needs it (upstream also uses it for the OpenAI Codex WebSocket under Bun).
- Bring more `ConverseStream` members (`guardrailConfig`, `performanceConfig`, `serviceTier`, etc.) into the JSON→SDK conversion for `on_payload`.
