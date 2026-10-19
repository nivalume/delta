# Auth (`auth`)

Upstream: `src/auth/types.ts`, `src/auth/resolve.ts`, `src/auth/helpers.ts`, `src/auth/context.ts`, `src/auth/credential-store.ts`.

## Types

| Upstream | Rust |
| --- | --- |
| `ModelAuth { apiKey, headers, baseUrl }` | `ModelAuth` |
| `ApiKeyCredential` / `OAuthCredential` / `Credential` | Same names; `Credential` is tagged by `type` (`api_key` / `oauth`), matching pi's `auth.json` format; extra OAuth fields go in `extra` |
| `CredentialStore` | trait: `read`, `list`, `modify`, `delete` |
| `InMemoryCredentialStore` | Same name; serializes writes per provider |
| `AuthContext` | trait: `env(name)`, `file_exists(path)`; default implementation `DefaultAuthContext` |
| `ApiKeyAuth` | trait: `name`, `resolve`, optional `check`, optional `login` |
| `OAuthAuth` | trait: `name`, `login`, `refresh`, `to_auth` |
| `ProviderAuth { apiKey?, oauth? }` | `ProviderAuth { api_key, oauth }` (`Arc<dyn ..>`) |
| `envApiKeyAuth(name, envVars)` | `env_api_key_auth(name, &[..])` |
| `resolve: () => ({ auth: {} })` for local servers without a key | `ProviderAuth::keyless(name)` |
| `AuthInteraction` / `AuthPrompt` / `AuthEvent` | trait / enums with the same names |

`CredentialStore::modify` is the only write path. The function passed in receives the current credential and returns a new credential, or `None` (unchanged). The store implementation serializes access per provider (with a cross-process lock where supported, for example a file lock).

## Resolution order (`resolve_provider_auth`)

Same as upstream:

1. An explicit `api_key` (request option) and the provider has API key auth: build a temporary credential from it and resolve.
2. A stored credential exists:
   - OAuth and the provider supports OAuth: if less than 5 minutes of validity remain (or a longer period the caller asks for), refresh inside the lock, then call `to_auth`.
   - API key and the provider supports it: resolve with the stored key and env (the request's env overrides the stored one).
   - Type mismatch: treated as not configured; no fallback to environment variables.
3. No stored credential: resolve from environment sources such as environment variables.

A stored credential owns the provider: environment variables are only consulted when no credential is stored. A failed OAuth refresh does not silently fall back to a key from environment variables.

The request's `env` overlay: `OverlayAuthContext` checks the request's env first, then the process environment. Blank values count as unset.

## OAuth refresh locking

`refresh_stored_oauth_credential`: inside `modify`, check again whether a refresh is needed (double check). If so, call `oauth.refresh` and write the result to the store before the lock is released. Concurrent requests refresh only once (covered by tests).

In upstream, the caller's signal only cancels the wait for the lock. Once the refresh starts, it ignores the signal and is bounded only by a 15-second timeout, so that a rotated refresh token is not lost. In Rust, cancelling means dropping the future, so the whole refresh runs in a separate Tokio task. Dropping the caller's future does not stop the refresh from completing and writing back. The refresh itself still has a 15-second timeout.

Error codes: refresh failure → `OAuth` (the credential is kept; logging in again fixes it); store read/write failure or API key resolution failure → `Auth`.

## Scope of the first phase

- Implemented: types, in-memory store, resolution order, locked refresh, `env_api_key_auth`, `keyless`, the `Models::login/logout` framework.
- OAuth login flows and their helpers (PKCE, the loopback callback server, device-code polling) live in `oauth`; see [oauth.md](oauth.md). Every upstream flow is ported: Anthropic, OpenRouter, xAI, OpenAI Codex, Sign in with ChatGPT, Kimi Code, Meta, GitHub Copilot, Radius.
