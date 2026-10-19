# Anthropic workload identity federation (`api::anthropic_federation`)

## Purpose

Workload identity federation lets a workload (a CI job, a Kubernetes pod) call the Anthropic API without an API key. The workload holds an OIDC identity token (a JWT) issued by its platform. It exchanges that token at Anthropic's token endpoint for a short-lived access token and sends the access token as `Authorization: Bearer`.

Upstream pi resolves the federation variables in `anthropicApiKeyAuth` and hands an `oidc_federation` config to the Anthropic TypeScript SDK, which runs the exchange, caches the token and refreshes it. delta has no SDK, so `delta_ai::api::anthropic_federation` ports that part of the SDK:

- `AnthropicFederationConfig`: the config upstream builds (organization, workspace, federation rule, service account, identity token file).
- `exchange_identity_token(client, config, base_url)`: one jwt-bearer exchange (SDK `oidcFederationProvider` with `identityTokenFromFile`).
- `FederationTokenCache`: the SDK's `TokenCache` around that exchange.
- `federation_token_cache(base_url, config)`: the shared cache of the current config (upstream's `federationClient`).
- The `ANTHROPIC_*_ENV` constants (upstream `env-api-keys.ts`), `TOKEN_ENDPOINT`, `GRANT_TYPE_JWT_BEARER`, `OAUTH_API_BETA_HEADER`, `FEDERATION_BETA_HEADER`, and the helpers `require_secure_token_endpoint` and `redact_sensitive`.

`api::anthropic_messages` decides when a request federates and sends the token (see [api-anthropic-messages.md](api-anthropic-messages.md)). `providers::builtin` resolves the variables.

## Upstream source files

- `packages/ai/src/providers/anthropic.ts`: `anthropicApiKeyAuth().resolve`
- `packages/ai/src/env-api-keys.ts`: the variable names
- `packages/ai/src/api/anthropic-messages.ts`: `getAnthropicFederation`, `federationClient`, `PiAnthropic`, `createClient`, `assertRequestAuth`
- `@anthropic-ai/sdk` 0.129.0 (the version in `packages/ai/package.json`): `src/lib/credentials/identity-token.ts`, `oidc-federation.ts`, `token-cache.ts`, `types.ts`, `credential-chain.ts` (`resolveCredentialsFromConfig`), `src/client.ts` (`authHeaders`, `prepareRequest`, `shouldRetry`, `withOptions`)
- Tests: `test/anthropic-federation.test.ts`, `test/anthropic-federation-sdk.test.ts`, ported to `tests/anthropic_federation.rs` with extra cases for the SDK parts (refresh windows, 401, error texts, identity token file errors, non-https refusal) and unit tests in the module.

## Resolution (`AnthropicApiKeyAuth`)

Order, as upstream:

1. A stored API key (with the credential's `env`).
2. `ANTHROPIC_AUTH_TOKEN`: `Authorization: Bearer <token>` header.
3. `ANTHROPIC_OAUTH_TOKEN`, then `ANTHROPIC_API_KEY`: the API key.
4. Federation: `ANTHROPIC_FEDERATION_RULE_ID`, `ANTHROPIC_ORGANIZATION_ID` and `ANTHROPIC_IDENTITY_TOKEN_FILE` must all be set, else the provider is not configured. `ANTHROPIC_SERVICE_ACCOUNT_ID` and `ANTHROPIC_WORKSPACE_ID` are optional. The result has empty auth, the set variables in `env`, and source `workload identity federation`.

Keys and `ANTHROPIC_AUTH_TOKEN` win over federation, as in the SDK. The ids are provider config, not auth, so they travel in `env`; `Models` merges that `env` into `StreamOptions::env`.

## When a request federates

`get_anthropic_federation` in `api::anthropic_messages`: only for `model.provider == "anthropic"` (the exchange is an Anthropic endpoint), and only when the request has no key and no `authorization` / `x-api-key` / `cf-aig-authorization` header. The three required variables come from `options.env`, else the process environment (`auth::env_value`); a missing one means no federation, and the request fails with `No API key for provider: anthropic`. Other `anthropic-messages` providers never federate.

Upstream's `PiAnthropic` turns off the SDK's own credential chain, so header-owned auth never triggers an exchange. delta has no such chain; federation runs only through this path.

## Token exchange

`POST {baseUrl}/v1/oauth/token`, where `baseUrl` is the model's base URL with trailing slashes removed:

- Headers: `Content-Type: application/json`, `anthropic-beta: oauth-2025-04-20,oidc-federation-2026-04-01`, `User-Agent: delta/...`.
- Body (in this order): `grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer"`, `assertion` (the identity token), `federation_rule_id`, `organization_id`, then `service_account_id` and `workspace_id` when set. The workspace goes in the exchange, not in a request header: the minted token is workspace-scoped.
- The identity token file is read on every exchange (so rotated tokens, such as Kubernetes projected tokens, are picked up) and trimmed.
- The access token lives `expires_in` seconds from the exchange (whole seconds now plus `expires_in`).

Checks and errors, with the SDK's texts:

| Condition | Error |
| --- | --- |
| Base URL is not https and not `http://` on `localhost`, `127.0.0.1` or `::1` | `Refusing to send credential over non-https token endpoint "<url>"` |
| Base URL does not parse | `Invalid token endpoint base URL "<url>": <error>` |
| File cannot be read | `Failed to read identity token file at <path>: <error>` |
| File is blank | `Identity token file at <path> is empty` |
| Token longer than 16 KiB | `Identity token is <n> KiB, exceeds the 16 KiB assertion limit` |
| Connection error | `Failed to reach token endpoint <url>: <error>` |
| Non-2xx status | `Token exchange failed with status <status>[ (request-id <id>)]: <redacted body>[hint]` |
| Body is not JSON (first 1 MiB is read) | `Token endpoint returned non-JSON response (status <status>)` |
| No `access_token` | `Token endpoint response missing access_token: <redacted body>` |
| `token_type` set and not Bearer (any case) | `Token endpoint response: unsupported token_type "<type>" (want Bearer)` |
| `expires_in` is not a finite number | `Token endpoint response missing required fields: <redacted body>` |

The 401 hint is ` Ensure your federation rule matches your identity token. ` plus, when no workspace is set, the advice to set `ANTHROPIC_WORKSPACE_ID`, plus a pointer to the Workload identity page of Claude Console. Redaction keeps only `error`, `error_description` and `error_uri` from a JSON object body; a non-JSON body is truncated to 2000 characters; other JSON values become `null`.

A failed exchange fails the request with that message. It is not retried (upstream's `retryProviderRequest` only retries SDK API errors).

## Token cache

`FederationTokenCache` follows the SDK's `TokenCache`. On each request:

- No cached token: exchange and wait.
- More than 120 s left: use the cached token.
- 30 to 120 s left (advisory window): use the cached token and start a refresh in the background. A failed background refresh is ignored (the cached token stays) and blocks further background refreshes for 5 s.
- Less than 30 s left: exchange and wait; a failure fails the request.

Concurrent requests share one exchange. A Messages response with status 401 drops the cached token (the SDK's `invalidate()`), so the next request exchanges again; the failed request itself is not retried, since upstream calls the SDK with `maxRetries: 0`.

`federation_token_cache` keeps one cache for the current `(base URL, config)` pair, as upstream keeps one SDK client for `JSON.stringify([model.baseUrl, federation])` and clones it per request with `withOptions()`, which shares the token cache. A request with another pair replaces it.

## Request

A federated request is an API-key-mode request without a key (no Claude Code identity, no OAuth betas from pi). The SDK adds two things, which `api::anthropic_messages` reproduces:

- `Authorization: Bearer <access token>` in the auth header position (after the SDK defaults, before client and request headers). There is no `X-Api-Key`.
- After all headers are merged, `oauth-2025-04-20` is appended to `anthropic-beta` unless it is already there (the list is split on `,`, trimmed and joined with `,`).

## Rust-specific differences

- **No SDK**: this module ports the SDK pieces an environment-only `oidc_federation` config reaches. The SDK's `WorkloadIdentityError` fields (`statusCode`, `body`, `requestId`) are not kept; upstream surfaces only the message, and so does `Error::Message`.
- **Cache key**: upstream also replaces its client when the `fetch` function changes. A `reqwest::Client` has no identity to compare, so the key is the base URL and config only. The exchange uses the HTTP client of the request that triggers it (`StreamOptions::http_client` or the default client).
- **Token per attempt**: the token is fetched once per request attempt (the first request and the strict-tools fallback request). When `max_retries` is set, `http::send_with_retry` reuses that token for its retries; upstream's `retryProviderRequest` makes a new SDK request per retry, which asks the cache again. This differs only when a token enters the refresh window during a retry backoff.
- **Shared exchange**: concurrent waiters serialize on an async mutex and reuse a token that another waiter just minted. When that exchange fails, the next waiter runs its own exchange instead of receiving the same error.
- **Cancellation**: waiting for a token races `options.cancel`. Cancelling a request drops its in-flight exchange (no token is stored). A background refresh runs in its own Tokio task and is not cancelled.
- **File reading**: the identity token file is read with blocking `std::fs::read` (a small file) and decoded as lossy UTF-8, as Node's `readFile(path, 'utf-8')`. Read errors carry Rust's `io::Error` text (`No such file or directory (os error 2)`) instead of Node's (`ENOENT: no such file or directory, open '...'`).
- **User-Agent**: the exchange sends `http::user_agent()` (`delta/...`), where the SDK sends its own `Anthropic/JS 0.129.0`.
- **Timeouts**: the exchange has no request timeout of its own, as the SDK's raw `fetch`; the default client's 30 s connect timeout applies.
- **Logging**: the SDK logs failed background refreshes at debug level; this crate has no logger, so they are dropped.
- **Unusual token responses**: an `access_token` that is a JSON object or array counts as missing (JavaScript would stringify it). A non-string `token_type` is ignored (JavaScript throws a `TypeError`). A JSON `null` body reports `missing access_token: null` (JavaScript throws a `TypeError`).

## Not ported

SDK features that upstream's federation path does not reach, because `PiAnthropic` disables the SDK's credential chain and pi passes the config directly:

- Profile files (`ANTHROPIC_PROFILE`, `ANTHROPIC_CONFIG_DIR`, `configs/<profile>.json`, `active_config`), the `user_oauth` authentication type, and the on-disk credentials cache with its file permission checks.
- `ANTHROPIC_IDENTITY_TOKEN` (a static token value), `ANTHROPIC_SCOPE`, and a `base_url` from the config.
- The `forceRefresh` flag after a 401, which only matters for the on-disk cache.

## Improvements pending confirmation

- Upstream keeps a single cache slot. A process that alternates between two federation configs (for example two workspaces) exchanges again at every switch. A map keyed by config would keep both tokens.
- The exchange could use `options.timeout` so a hanging token endpoint cannot stall a request indefinitely (cancellation still works).
