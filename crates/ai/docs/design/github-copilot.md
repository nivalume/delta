# GitHub Copilot (`github-copilot`)

## Purpose

The built-in `github-copilot` provider sends requests to the GitHub Copilot API with a Copilot subscription. It has no wire protocol of its own: Claude models use `anthropic-messages`, Gemini and Kimi models `openai-completions`, GPT, Grok and MAI models `openai-responses`. Copilot adds three things on top of those APIs:

- auth: a Copilot token from `COPILOT_GITHUB_TOKEN`, or the GitHub device-code login, which exchanges the GitHub token for a short-lived Copilot token and reads the account's model list;
- a per-credential API base URL (individual, business and enterprise accounts use different hosts);
- per-request headers that the three APIs add when `model.provider == "github-copilot"`.

Code:

- `crates/ai/src/oauth/github_copilot.rs`: `GitHubCopilotOAuth`, the OAuth flow.
- `crates/ai/src/api/github_copilot_headers.rs`: the dynamic request headers.
- `crates/ai/src/providers/github_copilot.rs`: `github_copilot_provider()` and the model filter.
- `crates/ai/src/providers/builtin.rs`: `builtin_provider("github-copilot")`.
- Catalog data: `crates/ai/data/providers/github-copilot.json` (already synced).

## Upstream source files

- `packages/ai/src/auth/oauth/github-copilot.ts` → `oauth::github_copilot`
- `packages/ai/src/api/github-copilot-headers.ts` → `api::github_copilot_headers`
- The Copilot branches of `packages/ai/src/api/anthropic-messages.ts`, `openai-completions.ts`, `openai-responses.ts` → the header code of `api::anthropic_messages`, `api::openai_completions`, `api::openai_responses`
- `packages/ai/src/providers/github-copilot.ts` (+ `.models.ts`) → `providers::github_copilot`
- `packages/ai/src/env-api-keys.ts` (the `COPILOT_GITHUB_TOKEN` entry)
- `packages/ai/src/api/transform-messages.ts` has no Copilot branch. Its Copilot test (`transform-messages-copilot-openai-to-anthropic.test.ts`) checks the generic cross-model rules for a Copilot Claude target; those rules are already ported in `transform`.
- Tests: `test/github-copilot-oauth.test.ts`, `test/github-copilot-anthropic.test.ts`, `test/transform-messages-copilot-openai-to-anthropic.test.ts`, the Copilot cases of `test/oauth-auth.test.ts`, `test/env-api-keys.test.ts`, `test/openai-responses-compat.test.ts` and `test/model-catalog-types.test.ts`. Ported to `tests/github_copilot.rs`, plus unit tests in the three modules.

## OAuth flow (`GitHubCopilotOAuth`)

Name `GitHub Copilot`, subscription, no login label. The client id is upstream's VS Code Copilot Chat id, kept base64-encoded in the source as upstream keeps it (`client_id()`).

Every GitHub and Copilot request sends the editor headers (`COPILOT_HEADERS`): `User-Agent: GitHubCopilotChat/0.35.0`, `Editor-Version: vscode/1.107.0`, `Editor-Plugin-Version: copilot-chat/0.35.0`, `Copilot-Integration-Id: vscode-chat`. The device-code and token-poll requests send only that `User-Agent`.

### Endpoints

| Request | URL |
| --- | --- |
| Device code | `POST https://<domain>/login/device/code` |
| Token poll | `POST https://<domain>/login/oauth/access_token` |
| Copilot token | `GET https://api.<domain>/copilot_internal/v2/token` |
| Model catalog | `GET <base url>/models` |
| Model policy | `POST <base url>/models/<model id>/policy` |

`<domain>` is the enterprise domain, else `github.com`. The base URL (`github_copilot_base_url`) comes from the Copilot token's `proxy-ep=<host>` field with a leading `proxy.` replaced by `api.` (`proxy.individual.githubcopilot.com` → `https://api.individual.githubcopilot.com`); without one, `https://copilot-api.<enterprise domain>`; else `https://api.individual.githubcopilot.com`.

`normalize_domain(input)` turns the user's input into a host: the input is trimmed, parsed as a URL (with `https://` prepended when it has no `://`), and its host is returned; blank or unparsable input gives `None`.

### Login

1. `Text` prompt `GitHub Enterprise URL/domain (blank for github.com)` with placeholder `company.ghe.com`. A cancelled interaction fails with `Login cancelled`. Non-blank input that does not normalize fails with `Invalid GitHub Enterprise URL/domain`.
2. Device code: form body `client_id`, `scope=read:user`. The response must be a JSON object with string `device_code`, `user_code`, `verification_uri`, a number `expires_in` and an optional number `interval`; otherwise `Invalid device code response` (not an object) or `Invalid device code response fields`. `verification_uri` must parse as an `http` or `https` URL (`Untrusted verification_uri in device code response`) and is passed on in its normalized form (the URL's `href`).
3. `DeviceCode` event with `user_code`, `verification_uri`, `interval` and `expires_in`.
4. Polling through `poll_oauth_device_code_flow` with `wait_before_first_poll`, the server's interval and expiry. Form body `client_id`, `device_code`, `grant_type=urn:ietf:params:oauth:grant-type:device_code`. A string `access_token` completes; `error: authorization_pending` → pending; `slow_down` → the server's `interval` when it is a number, else +5 s; other errors → `Device flow failed: <error>[: <error_description>]`; anything else → `Invalid device token response`.
5. Copilot token exchange with `Authorization: Bearer <GitHub token>`. The response needs a string `token` and a number `expires_at` (`Invalid Copilot token response` / `Invalid Copilot token response fields`).
6. Model catalog (two retries within five seconds, see below). Errors fail the login.
7. When the catalog has policy models: `Progress` `Enabling models...`, then a policy update per model (best effort).
8. `availableModelIds` is the catalog's available ids followed by the newly enabled ids, without duplicates.

A request that fails with a non-2xx status fails with `<status> <status text>: <body>`.

### Account model catalog

The catalog request sends `Accept: application/json`, `Authorization: Bearer <Copilot token>`, the editor headers and `X-GitHub-Api-Version: 2026-06-01`. The response's `data` must be an array (`Invalid Copilot models response`). Entries without a string `id` and entries with `capabilities.supports.tool_calls == false` are skipped. Then:

- picker models: `model_picker_enabled == true` and `policy.state != "disabled"`;
- the policy fallback applies only when the base URL is `https://api.individual.githubcopilot.com` (some individual accounts report `false` for every picker flag) and there are no picker models;
- available ids: the picker models, or with the fallback the models with `policy.state == "enabled"`;
- policy ids: models with `policy.state == "unconfigured"` that are in the built-in `github-copilot` catalog and are picker models (or any model, with the fallback).

### Policy updates

`POST <base url>/models/<id>/policy` with `{"state":"enabled"}`, `Content-Type: application/json`, the bearer token, the editor headers, `openai-intent: chat-policy` and `x-interaction-type: chat-policy`, two retries within five seconds. A 2xx response enables the model. A transport error or another status skips it. A `429` that outlasts the retries is an error that stops the batch; the login still succeeds with the models enabled so far. Cancellation fails the login.

### Rate-limit retry

Catalog and policy requests go through `fetch_with_rate_limit_retry` (upstream `fetchWithRateLimitRetry`). Each attempt times out after 5 s. A `429` is retried up to `max_retries` times. The delay is `Retry-After` (seconds as JavaScript `parseFloat` reads them, else an HTTP date minus now), or 500 ms doubled per retry without the header. An unreadable `Retry-After`, or a delay that reaches the end of the time budget, returns the `429` at once. Login uses two retries within 5 s; refresh uses none.

### Refresh and request auth

- `refresh`: the Copilot token exchange with the stored GitHub token (`refresh`) and the stored enterprise domain, then the catalog without retries. A catalog error (for example a `429`) fails the refresh.
- `to_auth`: `api_key` is the Copilot token, `base_url` the credential's base URL. `Models` puts that base URL on the request model, so business and enterprise accounts reach their own hosts.

### Credential

```json
{"type": "oauth", "refresh": "<GitHub token>", "access": "<Copilot token>", "expires": 1730000000000,
 "enterpriseUrl": "company.ghe.com", "availableModelIds": ["claude-sonnet-4.6", "gpt-5-mini"]}
```

`expires` is `expires_at * 1000 - 5 minutes`. `enterpriseUrl` holds the normalized domain and is left out for github.com. A refresh writes a new credential with only these fields, as upstream does. The extra fields live in `OAuthCredential::extra`, so the JSON shape is upstream's.

## Dynamic request headers

`api::github_copilot_headers` (upstream `github-copilot-headers.ts`):

- `infer_copilot_initiator(messages)`: `agent` when the last message is not a user message, else `user`.
- `has_copilot_vision_input(messages)`: a user message with block content or a tool result message contains an image.
- `build_copilot_dynamic_headers(messages, has_images)`: `X-Initiator`, `Openai-Intent: conversation-edits`, and `Copilot-Vision-Request: true` when `has_images`.
- `copilot_dynamic_headers(provider, messages)`: the headers for `github-copilot`, `None` for other providers.

The messages are the resolved transcript, as upstream passes `normalizedContext.messages`. Placement in each API's header order:

| API | Order |
| --- | --- |
| `anthropic-messages` | after `model.headers`, before `options.headers` (upstream `mergeClientHeaders(..., model.headers, dynamicHeaders, optionsHeaders)`); Copilot also uses `Authorization: Bearer` without the OAuth rewriting |
| `openai-completions` | after `model.headers`, before session affinity headers and `options.headers` |
| `openai-responses` | after `model.headers`, before session affinity headers and `options.headers` |

Request headers in `options.headers` therefore override the dynamic ones.

## Provider

`github_copilot_provider()` (upstream `githubCopilotProvider`):

- id `github-copilot`, name `GitHub Copilot`, base URL `https://api.individual.githubcopilot.com`;
- auth: `env_api_key_auth("GitHub Copilot token", ["COPILOT_GITHUB_TOKEN"])` with `GitHubCopilotOAuth`. `GH_TOKEN` and `GITHUB_TOKEN` are not used;
- models: the built-in catalog;
- APIs by `model.api`: `anthropic-messages`, `openai-completions`, `openai-responses`;
- `filter_github_copilot_models`: with an OAuth credential whose `availableModelIds` is an array of strings, only those models are available (`Models::get_available`). API-key credentials, no credential, or a missing or malformed list keep every model.

## Rust-specific differences

- Cancellation is the interaction's `CancellationToken` instead of an `AbortSignal`. `OAuthAuth::refresh` takes no signal (see [auth.md](auth.md)), so refresh requests cannot be cancelled by the caller; `Models` bounds a refresh with its 15 s timeout.
- Upstream tests stub the global `fetch`. Here `GitHubCopilotOAuth::rewrite_url` maps each request URL before it is sent; the URLs are still derived as upstream derives them, so tests see the host each request went to. `Default` leaves URLs unchanged.
- The time budget of `fetch_with_rate_limit_retry` is applied as each attempt's timeout (`min(5 s, remaining budget)`) instead of a combined `AbortSignal`; a budget that is already used up before an attempt fails with `The operation was aborted due to timeout`.
- `Retry-After` dates are parsed as IMF-fixdate (`Sun, 06 Nov 1994 08:49:37 GMT`, the format HTTP uses) through `http::httpdate_to_unix`; JavaScript's `Date.parse` accepts more formats.
- The status text in error messages is the canonical reason phrase of the status code (`reqwest::StatusCode::canonical_reason`), not the server's reason phrase.
- An invalid JSON body fails with `Invalid JSON response: <serde error>` instead of the JavaScript `SyntaxError` text. Transport errors carry reqwest's message and its cause chain instead of `fetch failed`.
- `normalize_domain` uses the `url` crate (WHATWG URL parsing, like `new URL`). A URL without a host gives `None`; upstream gives `""`, which every caller treats as no domain.
- `DeviceCode` event intervals are `u64` (fractional seconds are truncated), as for the other device flows.
- The dynamic headers come from a helper that returns `None` for other providers (`copilot_dynamic_headers`) instead of the `if` repeated in each API.
- Timing tests run in real time with 1-second intervals instead of fake timers. The repeated `slow_down` timeout test uses a 1 s interval and a 13 s expiry (polls at 1 s and 7 s) instead of 5 s and 25 s. The transport-failure test sends the first policy request to a closed port.

## Not ported

- `bun-oauth.ts` (bundled OAuth loaders for Bun standalone binaries) and `lazyOAuth`/`load.ts`: not needed in Rust (see [oauth.md](oauth.md)).
- Live end-to-end tests that need a Copilot account (`context-overflow.test.ts`, `tool-call-id-normalization.test.ts`, `openai-responses-tool-result-images.test.ts`, `stream.test.ts` and similar): tests must not hit the network.

## Improvements pending confirmation

1. Pass a cancellation token to `OAuthAuth::refresh` so the Copilot token exchange and catalog request stop with the caller (same as item 2 of [oauth.md](oauth.md)).
2. Percent-encode the model id in the policy URL. Upstream inserts it as is; catalog ids are plain today.
3. Give the device-code, token-poll and Copilot-token requests a request timeout. Upstream has none for them (only the catalog and policy requests have the 5 s timeout); a stalled GitHub endpoint keeps the login waiting until it is cancelled.
4. Keep unknown extra fields of the stored credential on refresh. Upstream builds a new credential with only the Copilot fields.
