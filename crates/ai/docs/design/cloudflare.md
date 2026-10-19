# Providers: Cloudflare AI Gateway and Cloudflare Workers AI

## Purpose

Two built-in providers route requests through Cloudflare:

- `cloudflare-ai-gateway`: Cloudflare AI Gateway. Anthropic models go through the gateway's Anthropic passthrough (`anthropic-messages`), OpenAI models through its OpenAI passthrough (`openai-responses`), and Workers AI models (`workers-ai/@cf/...`) through the Unified API `/compat` endpoint (`openai-completions`).
- `cloudflare-workers-ai`: Workers AI's OpenAI-compatible endpoint (`openai-completions`) for chat models, and the REST `/ai/run` endpoint (`cloudflare-workers-ai-system-one`, see [api-system-one.md](api-system-one.md)) for the System One classifier models (`@cf/cloudflare/clef`, `@cf/cloudflare/clef-flash`, `typesafe/jev`).

Neither provider has its own chat wire protocol. Both reuse the existing API implementations and add two things: auth that resolves an account id (and a gateway id) next to the API token, and wrappers (`CloudflareStreams` for chat, `CloudflareClassifier` for classifiers) that fill the account and gateway ids into the model's base URL.

Code:

- `crates/ai/src/api/cloudflare.rs`: endpoint templates.
- `crates/ai/src/providers/cloudflare.rs`: auth, the stream and classifier wrappers, the provider factories.
- `crates/ai/src/providers/builtin.rs`: `builtin_provider("cloudflare-ai-gateway" | "cloudflare-workers-ai")`.

## Upstream source files

- `packages/ai/src/api/cloudflare.ts`: endpoint templates → `api::cloudflare` constants.
- `packages/ai/src/providers/cloudflare-auth.ts` → `CloudflareWorkersAiAuth`, `CloudflareAiGatewayAuth` (`cloudflare_workers_ai_auth()`, `cloudflare_ai_gateway_auth()`).
- `packages/ai/src/providers/cloudflare-stream.ts` → `resolve_cloudflare_model`, `resolve_cloudflare_base_url`, `CloudflareStreams` (`cloudflare_streams()`), `CloudflareClassifier` (`cloudflare_classifier()`).
- `packages/ai/src/providers/cloudflare-ai-gateway.ts` (+ `.models.ts`) → `cloudflare_ai_gateway_provider()`.
- `packages/ai/src/providers/cloudflare-workers-ai.ts` (+ `.models.ts`) → `cloudflare_workers_ai_provider()`.
- Catalog data: `data/providers/cloudflare-ai-gateway.json`, `data/providers/cloudflare-workers-ai.json` (already synced).
- `packages/ai/src/api/cloudflare-ai-binding.ts`: not ported (see below).
- Tests: `test/cloudflare-stream.test.ts`, `test/cloudflare-workers-ai-system-one.test.ts` (in `tests/cloudflare_workers_ai_system_one.rs`), the two Cloudflare cases of `test/providers.test.ts`, the Cloudflare cases of `test/openai-completions-empty-tools.test.ts` and `test/openai-responses-compat.test.ts`. `test/cloudflare-utils.ts` only gates the live-network suites (`stream.test.ts`, `empty.test.ts`), which are not ported because tests must not hit the network.

## Endpoints

| Constant | Template |
| --- | --- |
| `CLOUDFLARE_WORKERS_AI_BASE_URL` | `https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1` |
| `CLOUDFLARE_WORKERS_AI_REST_BASE_URL` | `https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai` (model runs are posted to `run`; used by the classifier API) |
| `CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL` | `https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/compat` |
| `CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL` | `https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/openai` |
| `CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL` | `https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/anthropic` |

The catalog entries carry these templates as `baseUrl`. The placeholders are filled in per request, so one catalog serves every account.

## Auth

Both auths read three values: `CLOUDFLARE_API_KEY`, `CLOUDFLARE_ACCOUNT_ID` and (gateway only) `CLOUDFLARE_GATEWAY_ID`. Each value is merged per field: the stored credential's value wins (the key from `credential.key`, the ids from `credential.env`), otherwise the `AuthContext` environment (which includes `StreamOptions::env`). A credential with only a key still picks up the ids from the environment.

A value that is missing or empty means "not configured" (`resolve` returns `None`). As in upstream, an empty stored key does not fall back to the environment key.

| | Workers AI | AI Gateway |
| --- | --- | --- |
| Required | key, account id | key, account id, gateway id |
| `auth` | `api_key = <key>` (sent as `Authorization: Bearer <key>`) | `headers = {cf-aig-authorization: "Bearer <key>", Authorization: None, x-api-key: None}` |
| `env` | `{CLOUDFLARE_ACCOUNT_ID}` | `{CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_GATEWAY_ID}` |
| `source` | `stored credential` when a credential exists, else `CLOUDFLARE_API_KEY` | same |
| Login prompts | secret "Enter Cloudflare API key", text "Enter Cloudflare account ID" | the same two, then text "Enter Cloudflare AI Gateway ID" |

Login stores `{type: "api_key", key, env: {CLOUDFLARE_ACCOUNT_ID[, CLOUDFLARE_GATEWAY_ID]}}`, the `auth.json` shape upstream writes.

The gateway's `None` headers remove the API implementations' own `Authorization` / `x-api-key` headers. Otherwise the gateway would read them as a BYOK provider key. `Models` merges explicit request headers after auth headers, so an explicit `Authorization: Bearer <provider key>` (BYOK) replaces the `None` and reaches the gateway next to `cf-aig-authorization`. The API implementations already accept `cf-aig-authorization` in place of an API key (`openai-completions`, `openai-responses`, `anthropic-messages`).

## Endpoint placeholders

`Models::apply_auth` merges the resolved `env` into `options.env` (explicit request `env` wins). `CloudflareStreams` wraps an API implementation; before `stream` / `stream_simple` it replaces `{CLOUDFLARE_ACCOUNT_ID}` and `{CLOUDFLARE_GATEWAY_ID}` in `model.base_url` with the `options.env` values. A placeholder without a value stays in the URL. Without `options.env` the model is passed unchanged.

The gateway provider maps all three APIs (`anthropic-messages`, `openai-completions`, `openai-responses`), each wrapped in `CloudflareStreams`, even when the catalog lacks one of them: models.dev's gateway catalog drops and restores the `workers-ai/*` entries over time. The Workers AI provider uses one wrapped `openai-completions` implementation.

The `openai-completions` compat detection for both providers (no `store`, no developer role, `max_tokens` and no `reasoning_effort` on the gateway, no long cache retention) was already ported in `api::openai_completions`.

## Classifier models

`cloudflare_workers_ai_provider()` registers `cloudflare-workers-ai-system-one` (see [api-system-one.md](api-system-one.md)) wrapped in `CloudflareClassifier` (upstream `cloudflareClassifier`). The classifier models use the same auth as the chat models. Before `classify`, the wrapper fills `{CLOUDFLARE_ACCOUNT_ID}` in `model.base_url` from `options.base.env` with `resolve_cloudflare_base_url`; the catalog base URL is `CLOUDFLARE_WORKERS_AI_REST_BASE_URL`, so requests go to `https://api.cloudflare.com/client/v4/accounts/<account>/ai/run`. Without `options.base.env`, or when nothing changes, the model is passed through unchanged.

## Rust-specific differences

- Upstream's `resolve` checks `signal.throwIfAborted()` between env reads. `ApiKeyAuth::resolve` takes no signal in delta (async work is cancelled by dropping the future, see the design README), and `AuthContext::env` is synchronous, so there is nothing to check.
- `resolve_cloudflare_model` returns `Cow<Model>`: it borrows the model when nothing changes, where upstream returns the same object.
- Upstream fills placeholders with JavaScript `String.replaceAll`, which interprets `$&`-style patterns in the replacement string. Rust's `str::replace` inserts the value literally. Cloudflare account and gateway ids do not contain `$`.
- `AuthContext::env` treats blank values as unset (delta's general rule); upstream's `ctx.env` returns them and the truthiness check then rejects only empty strings.
- Upstream's legacy global API (`compat.ts`) sends Cloudflare models without resolved Cloudflare auth through the `Models` path. delta has no legacy global API, so this does not apply.

## Not ported

- `api/cloudflare-ai-binding.ts` (`createAiBindingFetch`, `AiBinding`, `CLOUDFLARE_GATEWAY_BINDING_AUTH_SENTINEL`) and its test `test/cloudflare-ai-binding.test.ts`. The module adapts the Workers runtime's `env.AI` binding (`env.AI.fetch()`) into a `fetch` function, so a Worker can reach its gateway without an API token. That binding exists only inside a Cloudflare Worker (JavaScript, `workerd`). A native Rust process has no `env.AI` binding, and delta has no pluggable `fetch` (`StreamOptions::http_client` takes a `reqwest::Client`). The sentinel constant alone has no use without the binding. Outside the Workers runtime, the gateway is reached over HTTPS with `cf-aig-authorization`, which is what these providers do.
- Live-network tests gated by `test/cloudflare-utils.ts`.

## Improvements pending confirmation

1. `CloudflareStreams` could end the stream with a clear error when a placeholder is still in the base URL after the fill-in, instead of sending a request to a URL that contains `{CLOUDFLARE_GATEWAY_ID}`. Through `Models` this cannot happen (auth fails first with `Provider is not configured`); it happens when a caller uses the wrapper or the provider directly without `options.env`. Upstream sends the request as is, so nothing is changed.
