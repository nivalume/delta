# API: azure-openai-responses

## Purpose

Implements the Azure OpenAI v1 Responses endpoint: `POST {baseUrl}/responses?api-version=<version>`, SSE streaming, authenticated with the `api-key` request header. Input item conversion, tool conversion and stream event handling all reuse `api::openai_responses_shared` (see `api-openai-responses.md`). This module only handles Azure's endpoint, deployment name, request body differences and error prefix. The unit struct is `AzureOpenAIResponsesApi`.

## Upstream source files

- `packages/ai/src/api/azure-openai-responses.ts`: `stream`, `streamSimple`, `buildParams`.
- `packages/ai/src/api/azure-openai-config.ts`: deployment name mapping, endpoint resolution and normalization, API version.
- Test references: `test/azure-openai-base-url.test.ts`, `test/azure-openai-responses-reasoning-replay.test.ts`, `test/azure-openai-tool-choice.test.ts`, and pi-py's `tests/test_azure_openai_responses.py`.

## Endpoint and configuration

Azure models have no baseUrl that can be used directly: each user has their own resource, resolved per request. Options come from `ApiOptions::AzureOpenAIResponses(AzureOpenAIResponsesOptions)`. Environment variables are looked up in `options.env` first, then in the process environment (empty values count as unset).

- Base URL, the first non-empty value in this order:
  1. `azure_base_url` (trimmed);
  2. `AZURE_OPENAI_BASE_URL` (trimmed);
  3. `azure_resource_name` or `AZURE_OPENAI_RESOURCE_NAME` → `https://<name>.openai.azure.com/openai/v1`;
  4. `model.base_url`;
  5. If none is set, error `Azure OpenAI base URL is required. ...`.
- Normalization: strip the trailing `/`, then parse the URL; on failure, error `Invalid Azure OpenAI base URL: <original value>`. When the host ends with `.openai.azure.com`, `.cognitiveservices.azure.com` or `.ai.azure.com`, and the path is empty, `/`, `/openai` or `/openai/v1/responses`, the path is changed to `/openai/v1` and the query string is removed. Other URLs (including proxies) are left unchanged.
- API version: `azure_api_version`, otherwise `AZURE_OPENAI_API_VERSION`, otherwise `v1`.
- Request URL: the base URL path plus `/responses`, with the query string replaced by `api-version=<version>`.
- Deployment name (sent as the request body's `model`): `azure_deployment_name`, otherwise this model's entry in `AZURE_OPENAI_DEPLOYMENT_NAME_MAP`, otherwise `model.id`. The map format is comma-separated `model=deployment`; empty entries and entries missing either side are skipped. As with JavaScript `split("=", 2)`, anything after a second `=` is dropped. A later entry with the same name overrides an earlier one.

## Request mapping

- Request headers (later entries override earlier ones): `api-key: <api_key>`, `Accept`, `Content-Type: application/json`, `User-Agent: delta/...`, then `model.headers`, and finally `options.headers`. `Authorization` is not sent, and neither are session affinity headers.
- `options.api_key` is required and request headers cannot replace it (same as upstream); otherwise the error is `No API key for provider: <provider>`.
- Tool call id allow list: `openai`, `openai-codex`, `opencode`, `azure`.

Request body fields (in insertion order):

| Field | Rule |
| --- | --- |
| `model` | Deployment name |
| `input` | Shared conversion; `supportsMidConvoSystemMessages`, `supportsAdditionalTools`, `supportsToolSearch` default to false |
| `stream` | `true` |
| `prompt_cache_key` | `session_id` truncated to 64 code points (cache retention is ignored) |
| `store` | `false` |
| `max_output_tokens` | Sent when `max_tokens > 0`, minimum 16 (`supportsMaxOutputTokens` is ignored) |
| `temperature` | Sent when set |
| `tools` | Request-level tools; `supportsStrictMode` defaults to **true** (false for OpenAI), `supportsOpenAIGrammarTools` defaults to false |
| `tool_choice` | Sent as is |
| `reasoning` / `include` | Same as `openai-responses`, but without the GitHub Copilot and xAI branches and without mid-conversation effort |
| Sampling parameters | Merged last |

Upstream Azure does not send `service_tier`, `prompt_cache_retention` or `prompt_cache_options`, does no service tier pricing, and does not use `configuration_update`. `OpenAIResponsesOptions::service_tier` is ignored for Azure.

## Stream event mapping

Exactly the same as `openai-responses` (shared `process_responses_stream`), without service tier pricing. Azure may provide a reasoning item's `encrypted_content` only in the `output` of `response.completed`; the shared layer backfills it into the saved reasoning signature, so multi-turn replay works with `store: false`.

The `api` of the output message is fixed to `azure-openai-responses` (same as upstream).

## Errors and retries

- Retries and cancellation are the same as `openai-responses` (`http::send_with_retry`).
- The HTTP error format is the same as `openai-responses`, with the prefix `Azure OpenAI API error`: `Azure OpenAI API error (<status>): <error JSON>`; when the body has no `error` object, `Azure OpenAI API error (<status>): <SDK message>`. Endpoint resolution errors, a missing key and in-stream errors keep their original message.
- If the stop reason is still pending when the stream ends, the error is `Azure OpenAI Responses stream ended without a stop reason` (in practice the terminal event always sets the stop reason).

## stream_simple

Same as upstream: it passes only the base options, `tool_choice` (`auto` / `none`), and the reasoning level clamped to what the model supports (`off` sends no effort). `build_base_options` clears `options.api`, so with the simple interface the Azure endpoint can only come from environment variables or `model.base_url`.

## Rust-specific differences

1. The request URL is built with the `url` crate: the path gets `/responses`, and the query string is replaced with `api-version`. The upstream SDK appends `/responses` directly to the whole base URL string, so when the base URL has a query string, `/responses` ends up in the query string and is then replaced by `api-version`. This module follows pi-py's behavior (`/v1/responses?api-version=...`). It only affects proxy URLs with a query string.
2. When the API key is missing, upstream `streamSimple` throws synchronously; here an error stream ending with the same message is returned.
3. For the other differences (events without partial, tool call end arguments, empty deltas, and so on), see `api-openai-responses.md`.

## Not ported

- Upstream's `azure` provider wrapper for `openai-completions` (`resolveAzureModel` / `withDeploymentName`) belongs to the provider layer and is not in this API module.
- Telemetry, the `fetch` option (replaced by `StreamOptions::http_client`).

## Improvements pending confirmation

- The simple interface drops explicit Azure configuration (upstream behavior). `stream_simple` could keep the `azure_*` fields passed by the caller.
- Azure ignores `service_tier`, cache retention and session affinity headers (upstream behavior). Whether the Azure v1 endpoint accepts these fields is pending confirmation.
- A base URL with a query string currently loses that query string; it could keep the original query parameters and append `api-version`.
