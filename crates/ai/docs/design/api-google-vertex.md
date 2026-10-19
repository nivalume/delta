# API design: google-vertex

## Purpose

`google-vertex` streams Gemini models through Vertex AI and converts the responses into delta's unified `AssistantMessageEvent` protocol.

Upstream sends REST requests through the Vertex mode of `@google/genai`. Per `AGENTS.md`, Vertex AI uses the official Google Cloud Rust client libraries:

- `google-cloud-aiplatform-v1` (feature `prediction-service`): `PredictionService::stream_generate_content`, a gRPC server stream;
- `google-cloud-auth`: Application Default Credentials (ADC), credential files, Vertex API key.

The whole module is behind the `vertex` feature (`#[cfg(feature = "vertex")]`).

Module: `crates/ai/src/api/google_vertex.rs`. Unit tests are in `google_vertex/tests.rs`, end-to-end tests in `crates/ai/tests/google_vertex.rs`.

Public interface:

- `GoogleVertexApi` (unit struct, implements `ProviderStreams`);
- `stream(model, context, StreamOptions)`, `stream_simple(model, context, SimpleStreamOptions)`;
- `stream_with_client(model, context, StreamOptions, PredictionService)`: sends the request with a client supplied by the caller (custom endpoint, credentials, transport or test stub); the rest of the flow is the same as `stream`;
- `get_google_budget` (Vertex's thinking budget table), `proto_request`, `response_chunk`, `response_json` (conversion functions, public for testing and reuse).

API-specific options come from `ApiOptions::GoogleVertex(GoogleVertexOptions)` (`tool_choice`, `thinking`, `project`, `location`); other variants are ignored.

## Upstream source files

- `packages/ai/src/api/google-vertex.ts`: `stream`, `streamSimple`, client creation, `getGoogleBudget`.
- `packages/ai/src/api/google-shared.ts`: message/tool conversion, thinking config, signatures, stop reasons, retry. Already ported on the Rust side as `api::google_shared` (see `api-google-generative-ai.md`); this module reuses it directly without changes.
- `packages/ai/src/providers/google-vertex.ts`: provider factory and auth resolution (`GOOGLE_CLOUD_API_KEY`, ADC file check), ported in the `vertex` module of `providers/builtin.rs` (`VertexAuth`, including the login flow). This module only does the resolution that the upstream API module itself does.
- Tests: the Vertex parts of `test/google-vertex-api-key-resolution.test.ts`, `google-raw-stop-reason.test.ts`, `google-thinking-level-map.test.ts`; pi-py's `google_vertex.py` and `test_google_streams.py` as a second reference.

## Division of work with google_shared

`google_shared` handles only plain data (`GenerateRequest`, `ResponseChunk`, `StreamState`) and is transport-independent. This module only does:

1. Resolution of auth, endpoint and model resource name, and client construction;
2. `GenerateRequest` → proto `GenerateContentRequest`;
3. proto `GenerateContentResponse` → `ResponseChunk`, handed to the same `StreamState`;
4. At the end, `check_finished(output, "Google Vertex stream ended without a finish reason")`.

### Request conversion: through REST JSON + serde

The generated proto types implement serde. Field names are the camelCase names of REST/proto JSON, bytes use base64, and enums accept name strings (and also numbers). So the request is first rendered to REST JSON with the Gemini module's `google_generative_ai::request_body` (this is exactly the request body `@google/genai` sends to Vertex), handed to `on_payload`, then converted to proto with `serde_json::from_value::<GenerateContentRequest>`, and finally `model` is filled in.

Reasons:

- The JSON that `on_payload` sees is the same as in the Gemini module, and its fields match the request body the upstream SDK actually sends. Callbacks can edit it by REST field names (e.g. `generationConfig.topK`, `labels`), and every field the proto defines takes effect;
- No need to hand-write proto construction code for `Part`/`Content`/`Tool`;
- serde decodes base64 text such as thought signatures and images into bytes.

Notes:

- Fields the proto does not define (e.g. `functionCall.id`, `functionResponse.id`) go into the generated type's unknown-field table and are dropped during gRPC encoding.
- When the JSON cannot be converted to proto (wrong type, integer out of range, invalid base64), the request fails before it is sent: `Invalid Google Vertex request payload: <reason>`.
- If the JSON returned by `on_payload` has a `model` field, it is kept (it must be a full resource name); otherwise the resolved resource name is used.

### Response conversion: read the proto directly

The response direction does not go through JSON: when serializing enums, the generated types output numbers (`finishReason: 1`), unlike REST. `response_chunk` reads proto fields directly:

- Only the first candidate is read;
- part: `text()`, `thought` (recorded as thinking only when `true`), `thought_signature` (non-empty bytes are encoded to text with standard base64, the same form REST returns), `function_call()` (an empty `name` counts as missing; `args` as is);
- `finish_reason`: `FINISH_REASON_UNSPECIFIED` (the proto default) counts as no stop reason; known values use their name (`STOP`, `SAFETY`, ...); unknown numeric values use the number as text;
- `usage_metadata`: proto3 scalars have no "missing", so 0 and negative values count as missing. REST JSON also omits 0 values, so the result matches the Gemini module (usage is not updated when `promptTokenCount` is 0);
- `response_id`: the empty string counts as missing.

`on_provider_stream_event` receives `response_json`: the serde JSON of the generated type (REST field names, base64 bytes, default values omitted), with `candidates[].finishReason` replaced by the name string. Other enum fields (safety ratings, `trafficType`, etc.) stay numeric. It is serialized only when the callback is set.

## Request mapping

### Flow

1. Already cancelled: end with `aborted` directly, without sending a request.
2. Resolve auth, endpoint and model resource name (same order as upstream: API key, otherwise project, then location).
3. Build the client (`stream_with_client` skips this step).
4. `google_shared::build_request` → `request_body` → `on_payload` → `proto_request`.
5. `retry_google_request` wraps `stream_generate_content().send()`; on success, `start`.
6. Read the stream one item at a time (`http::or_cancel` makes cancellation take effect immediately). Each response goes first to `on_provider_stream_event`, then to `StreamState::apply_chunk`.

### Auth

- `options.api_key` is non-empty after trimming, is not `gcp-vertex-credentials`, and is not a placeholder of the form `<...>` (such as `<authenticated>`): use the Vertex API key (express mode). `api_key_credentials` sends `x-goog-api-key`; project/location are not needed.
- Otherwise use ADC:
  - project: `GoogleVertexOptions::project` → `GOOGLE_CLOUD_PROJECT` → `GCLOUD_PROJECT`; if none is set, fail with `Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options.`
  - location: `GoogleVertexOptions::location` → `GOOGLE_CLOUD_LOCATION`; if not set, fail with `Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options.`
  - Credentials: when `GOOGLE_APPLICATION_CREDENTIALS` has a value (upstream `keyFilename`), read that file and build by `type`: `service_account`, `authorized_user`, `impersonated_service_account`, `external_account`. Other types fail with `Unsupported Google credentials type in <path>: <type>`; read/parse failures fail with `Failed to read/parse Google credentials file <path>: …` respectively. When it is not set, use the default ADC of `google-cloud-auth` (gcloud's well-known file, then falling back to the metadata server).
- Environment variables are looked up in `options.env` first, then the process environment. Blank values count as unset (`StreamOptions::env_var`).
- The upstream API module itself does not read `GOOGLE_CLOUD_API_KEY` (the provider factory resolves it into `apiKey`), and neither does this one.

### Endpoint

The gRPC path is fixed (`/google.cloud.aiplatform.v1.PredictionService/StreamGenerateContent`), so only the scheme and authority are taken from the endpoint:

1. The model's `base_url` (trimmed, non-empty, and without the `{location}` placeholder; this is upstream `resolveCustomBaseUrl`);
2. Otherwise `GOOGLE_VERTEX_BASE_URL` (the JS SDK reads it from the process environment);
3. Otherwise by location: API key and `global` use `https://aiplatform.googleapis.com`; the multi-regions `us`/`eu` use `https://aiplatform.{us|eu}.rep.googleapis.com` (matching the Python SDK and pi-py); everything else uses `https://{location}-aiplatform.googleapis.com` (the SDK's regional endpoint).

In the built-in catalog the `baseUrl` of Vertex models is `https://{location}-aiplatform.googleapis.com`, so rule 3 applies by default.

### Model resource name

First get the path per the SDK's `tModel` (Vertex mode): if the id contains `..`, `?` or `&`, fail with `invalid model parameter`; ids starting with `publishers/`, `projects/` or `models/` are used as is; ids containing `/` take the first two segments as `publishers/{a}/models/{b}`; everything else becomes `publishers/google/models/{id}`.

- API key: use this path directly;
- ADC: when the path does not start with `projects/`, prefix it with `projects/{project}/locations/{location}/`.

### Request headers

Upstream sets `{User-Agent: pi UA, ...model.headers, ...options.headers}` as the SDK's `httpOptions.headers`. Here `options.headers` (`Models` has already merged the model's headers; entries with value `None` are removed) become client-level request metadata (`ClientBuilder::with_extension(HeaderMap)`). The SDK drops custom values for `user-agent`, `authorization`, `x-goog-api-key`, `x-goog-api-client`, `x-goog-user-project` and `x-goog-request-params`, and sets them itself. The User-Agent is set per request through `RequestOptionsBuilder::with_user_agent`: delta's User-Agent by default, or the value of `user-agent` in `options.headers` when present, matching upstream.

### Request body

Same as the Gemini module (see the "Request body" section of `api-google-generative-ai.md`). The corresponding proto fields:

| `GenerateRequest` field | REST JSON | proto field |
| --- | --- | --- |
| `contents` | `contents[]` | `contents` (`role`, `parts`: `text`, `thought`, `thoughtSignature`, `inlineData`, `functionCall`, `functionResponse`) |
| `system_instruction` | `systemInstruction: {role: "user", parts: [{text}]}` | `system_instruction` |
| `temperature` | `generationConfig.temperature` | `generation_config.temperature` (f32) |
| `max_output_tokens` | `generationConfig.maxOutputTokens` | `generation_config.max_output_tokens` (i32) |
| `thinking_config` | `generationConfig.thinkingConfig` | `generation_config.thinking_config` (`include_thoughts`, `thinking_level` enum, `thinking_budget` i32) |
| `tools` | `tools: [{functionDeclarations: [{name, description, parametersJsonSchema}]}]` | `tools[0].function_declarations[].parameters_json_schema` |
| `function_calling_mode` | `toolConfig.functionCallingConfig.mode` | `tool_config.function_calling_config.mode` (`AUTO`/`NONE`/`ANY`/`VALIDATED`) |

The `model` field is the resolved resource name. The rules for message conversion, thinking config, strict tool schema and function calling mode all come from `google_shared` and are the same as in the Gemini module (including adding ids to tool calls for models that need them, but the v1 proto has no id field; see below).

### `stream_simple`

Same as upstream `streamSimple`: `build_base_options`, `tool_choice` mapping, `thinking: {enabled: false}` when `reasoning` is empty or clamped to `off`; otherwise models that use `thinkingLevel` send the level and the rest send a budget.

Budgets use Vertex's own table (`getGoogleBudget` in upstream `google-vertex.ts`): custom `thinking_budgets` take precedence; `2.5-pro` is 128/2048/8192/32768; `2.5-flash` (**including** `2.5-flash-lite`; the upstream Vertex table has no Flash-Lite row) is 128/2048/8192/24576; others `-1`. Because this table differs from the Gemini API one, this module calls `google_shared::simple_thinking_with` with its own budget function (see "Core changes").

As in upstream, `stream_simple` does not keep `project`/`location` (`build_base_options` clears `api`); they can only come from `options.env` or the process environment.

## Stream event mapping

| proto response | Event / output field |
| --- | --- |
| `send()` succeeds (gRPC response headers arrived) | `Start` |
| `response_id` | `response_id`, keep the first non-empty value |
| part `text`, `thought: true` | If the current block is not thinking, end the current block and `ThinkingStart`; `ThinkingDelta` |
| part `text`, not thinking | If the current block is not text, end the current block and `TextStart`; `TextDelta` |
| part `thought_signature` (text/thinking part) | base64 text, used as the signature of the current block, keep the last non-empty value |
| part `function_call` | End the current block; `ToolCallStart`, `ToolCallDelta` (compact JSON of `args`), `ToolCallEnd`; a non-empty signature is recorded on the tool call |
| Function call id | Not in the v1 proto; always generate `<name>_<millisecond timestamp>_<counter>` (unique across chunks) |
| `finish_reason` | `raw_stop_reason` is its name; `STOP` → `stop`, `MAX_TOKENS` → `length`, others → `error`; if there are tool calls and it is `stop`, change it to `toolUse` |
| `usage_metadata` (when `prompt_token_count` > 0) | `input = prompt - cached`, `output = candidates + thoughts`, `cache_read = cached`, `reasoning = thoughts`, `total_tokens` takes the server value, cost via `calculate_cost` |
| Error status in the stream | Fail immediately, no retry, content received so far is kept |
| End of stream | End the current block; cancelled → `aborted`; no stop reason → `Google Vertex stream ended without a finish reason`; stop reason maps to `error` → `Provider stopped with: <raw>`; otherwise `Done` |

## Errors, retry, timeout and cancellation

- SDK error conversion (`vertex_error`): service errors with a gRPC status are converted to `Error::Status` using the HTTP mapping of `google.rpc.Code` (`INVALID_ARGUMENT`/`FAILED_PRECONDITION`/`OUT_OF_RANGE` 400, `UNAUTHENTICATED` 401, `PERMISSION_DENIED` 403, `NOT_FOUND` 404, `ALREADY_EXISTS`/`ABORTED` 409, `RESOURCE_EXHAUSTED` 429, `CANCELLED` 499, `UNKNOWN`/`INTERNAL`/`DATA_LOSS` 500, `UNIMPLEMENTED` 501, `UNAVAILABLE` 503, `DEADLINE_EXCEEDED` 504). The message is `"<status> <status.message>"` (the status code name when the message is empty). Errors with only an HTTP status (returned by a proxy or load balancer) are also converted to `Error::Status`. Other errors (connection, auth, timeout, serialization) use the SDK's error text.
- Retry: `google_shared::retry_google_request` (upstream `retryGoogleRequest`) wraps the call that opens the stream. When `Error::Status` is 408/409/429/5xx, it retries with exponential backoff up to `max_retries` (default 2). Other errors and in-stream errors are not retried. gax's retry policy does not apply to server streams, so it is not used. There is no `retry_after_ms` (gRPC's `RetryInfo` needs `google-cloud-rpc`, and the upstream SDK never uses the server delay either).
- Timeout: `options.timeout` (default 10 minutes) is set as the client's `attempt_timeout`. gRPC uses it as the deadline for the whole call (including the stream), the same semantics as the HTTP timeout of other APIs. With `stream_with_client`, the caller's client configuration wins.
- Cancellation: checked before the request; opening the stream and waiting to retry respond to cancellation through `retry_google_request`; while reading the stream, each `next()` races against `options.cancel`. After cancellation it ends with `aborted`, message `Request was aborted`, and keeps partial content. Reading stops when the consumer drops the stream.
- When `on_payload` or `on_provider_stream_event` returns an error, the request fails. There is no `on_response` (the upstream Vertex adapter does not call it, and a gRPC stream has no HTTP response).

## Rust-specific differences

- The transport is gRPC, not REST:
  - Only the scheme and authority of the model's `base_url` are used; the path (API version, `projects/.../locations/...` collection path) is not. Upstream's `baseUrlResourceScope: COLLECTION` and "`apiVersion: ""` when the URL already contains the version" have no counterpart under gRPC.
  - With a custom `base_url`, upstream does not add the `projects/{p}/locations/{l}` prefix. gRPC's `model` must be a full resource name, so ADC mode always adds the prefix.
  - The v1 proto's `FunctionCall`/`FunctionResponse` have no `id`: call ids for models such as Gemini 3 are not sent (upstream REST sends them), ids in responses are not available either, and tool call ids are always generated locally. Vertex pairs calls and results by order and name.
- `on_payload` receives the REST request body (same as the Gemini module), not the SDK's `{model, contents, config}`; fields the proto does not have are not sent.
- `on_provider_stream_event` receives the proto's serde JSON, see above; enums other than `finishReason` are numbers.
- Error messages use `"<status> <message>"`; upstream uses the JSON body assembled by the SDK.
- Retry count defaults to 2 (upstream Google adapter defaults to 0), same as the Gemini module.
- Upstream sets no timeout on the Vertex SDK; here `options.timeout` is applied (the core's unified semantics).
- The SDK additionally sends its own `x-goog-api-client`.
- `options.http_client` is not used (upstream errors on a custom `fetch`; there is no such check here).
- Credential files are dispatched by `type`. The supported types are the same as `google-cloud-auth`'s ADC (excluding `gdch_service_account`). Upstream `google-auth-library`'s `keyFilename` also supports older formats such as `.p12`/`.pem`.
- When cancelled up front the message is `Request was aborted` (in upstream `buildParams` it is `Request aborted`).
- Unknown numeric stop reasons are recorded as the number as text; under REST it would be the name from the server. Both map to `error`.
- A new client and credentials are created for each request (same as upstream creating `new GoogleGenAI` each time).
- `sanitizeSurrogates` is not needed: Rust strings contain no lone surrogates.

## Not ported

- Upstream's check that rejects a custom `fetch` (see above).
- Telemetry: the upstream Vertex adapter has none.

## provider-references checks

- Key not in the URL: the SDK puts the API key in `x-goog-api-key` metadata.
- `STOP` changed to `toolUse` when a function call appears (`MAX_TOKENS` stays `length`): yes.
- Unknown stop reasons do not crash: mapped to `error` with the raw value kept.
- Empty signatures treated as none; signatures on non-thinking text parts kept; thinking parts become thinking blocks: yes.
- Calls without an id get an id unique across chunks: yes (the v1 proto never has ids).
- Usage waits for `promptTokenCount`, saturating subtraction, output includes `thoughtsTokenCount`: yes.
- In-stream errors: a gRPC status fails immediately, no retry; error when there is no stop reason: yes.
- Lenient parsing: the SDK decodes proto types; unknown fields/enum values are not fatal (unknown enums are kept as numbers).
- `parametersJsonSchema`: yes; inlining `$defs`/`definitions` is not implemented (same as the Gemini module, see its document).
- Retry: 408/409/429/5xx are retried, connection errors are not (matching upstream `retryGoogleRequest`).

## Core changes

- The `vertex` feature adds the optional dependency `google-cloud-gax` (an official crate that `aiplatform` already depends on indirectly), for `with_user_agent`.
- `google_shared::simple_thinking_with` takes a budget function; Vertex passes its own `get_google_budget` (no 2.5-flash-lite row, matching upstream).

## Improvements pending confirmation

- Cache `PredictionService` by (endpoint, auth method, key/project/credential file, request headers) to reuse gRPC connections and the token cache (pi-py does this; upstream creates a new client each time).
- Read `RetryInfo.retry_delay` from gRPC error details as the retry delay.
- `stream_simple` keeps `project`/`location` from `GoogleVertexOptions` in `options.base.api` (upstream drops them).
- Replace the `{location}` placeholder in the catalog's `baseUrl` with the resolved location instead of ignoring it (same result, just more direct).
- Fill `response_model` from `model_version` (upstream does not).
- With an API key (express mode) over gRPC, whether the server accepts `model` as `publishers/google/models/{id}` needs to be verified with a real key; if not, switch to REST transport or require project/location.
- If function call ids need to be sent (Gemini 3), the only options are an API version with an `id` field or REST transport.
- Streaming function arguments (`partialArgs`/`willContinue`, `streamFunctionCallArguments`) are not used, same as upstream.
