# OAuth flows (`oauth`)

## Purpose

`delta_ai::oauth` holds the OAuth login flows of the built-in providers and the helpers they share. The framework (the `OAuthAuth` trait, `AuthInteraction`, `Models::login`, locked refresh) lives in `auth` and is described in [auth.md](auth.md). This module adds:

- shared helpers: PKCE, the loopback callback server, the race between the callback and a pasted code, device-code polling, the browser pages;
- five flows: Anthropic (Claude Pro/Max), OpenRouter (PKCE that mints an API key), and the device-code flows of xAI, Kimi Code and Meta;
- the `delta-ai` login CLI (`src/bin/delta-ai.rs`, feature `cli`).

The helpers keep upstream's names and shapes, so the remaining flows (OpenAI Codex, OpenAI ChatGPT, GitHub Copilot, Radius) can be ported one-to-one.

## Upstream source files

- `packages/ai/src/auth/oauth/pkce.ts`, `callback-server.ts`, `device-code.ts`
- `packages/ai/src/utils/oauth-page.ts`
- `packages/ai/src/auth/oauth/anthropic.ts`, `openrouter.ts`, `xai.ts`, `kimi-coding.ts`, `meta.ts`
- `packages/ai/src/providers/anthropic.ts`, `openrouter.ts`, `xai.ts`, `kimi-coding.ts`, `meta.ts` (the `auth.oauth` entries)
- `packages/ai/src/cli.ts`
- Tests: `test/oauth-callback-server.test.ts`, `test/oauth-device-code.test.ts`, `test/oauth-auth.test.ts` (the parts for these flows), `test/anthropic-oauth.test.ts`, `test/openrouter-oauth.test.ts`, `test/xai-oauth.test.ts`, `test/kimi-coding-oauth.test.ts`, `test/meta-oauth.test.ts`. Ported to `tests/oauth_helpers.rs`, `tests/oauth_flows.rs` and `tests/oauth_kimi_meta.rs`.

## Module layout

| Upstream | Rust |
| --- | --- |
| `auth/oauth/pkce.ts` | `oauth::pkce` |
| `auth/oauth/callback-server.ts` | `oauth::callback_server` |
| `auth/oauth/device-code.ts` | `oauth::device_code` |
| `utils/oauth-page.ts` | `oauth::page` |
| `auth/oauth/anthropic.ts` | `oauth::anthropic` (`AnthropicOAuth`) |
| `auth/oauth/openrouter.ts` | `oauth::openrouter` (`OpenRouterOAuth`) |
| `auth/oauth/xai.ts` | `oauth::xai` (`XaiOAuth`) |
| `auth/oauth/kimi-coding.ts` | `oauth::kimi_coding` (`KimiCodingOAuth`) |
| `auth/oauth/meta.ts` | `oauth::meta` (`MetaOAuth`) |
| `auth/oauth/load.ts`, `lazyOAuth` | Not needed: the flows are plain structs compiled into the crate |

The helpers are re-exported from `delta_ai::oauth`.

## Shared helpers

### PKCE

| Upstream | Rust |
| --- | --- |
| `generatePKCE(): Promise<{verifier, challenge}>` | `generate_pkce() -> Pkce { verifier, challenge }` |
| `base64urlEncode` (private) | `pkce::base64url_encode(&[u8]) -> String` |
| — | `pkce::pkce_challenge(verifier) -> String` (S256), used by tests |

The verifier is 32 bytes from the OS random source (`getrandom`), base64url without padding; the challenge is `base64url(SHA-256(verifier))`.

### Callback server

```rust
pub struct OAuthCallbackServerOptions<T> {
    pub provider_name: String,
    pub host: String,
    pub port: u16,                      // 0 picks a free port
    pub path: String,
    pub redirect_host: Option<String>,
    pub state: Option<String>,          // None: no state check
    pub complete: OAuthCallbackComplete<T>,
    pub cancel: Option<CancellationToken>, // upstream `signal`
    pub timeout: Option<Duration>,      // upstream `timeoutMs`
}
OAuthCallbackServerOptions::new(provider_name, host, port, path, |code| async move { .. })
    .redirect_host(..).state(..).cancel(..).timeout(..)

pub async fn start_oauth_callback_server<T: Send + 'static>(options)
    -> Result<OAuthCallbackServer<T>, ModelsError>;

impl<T> OAuthCallbackServer<T> {
    pub fn redirect_uri(&self) -> &str;
    pub async fn wait(&self) -> Result<Option<T>, ModelsError>;
    pub fn cancel(&self);
    pub fn close(&self);
}

pub async fn wait_for_callback_or_manual_input<T>(
    interaction: &dyn AuthInteraction,
    callback: Option<&OAuthCallbackServer<T>>,
    message: impl Into<String>,
    placeholder: impl Into<String>,
) -> Result<CallbackOrManual<T>, ModelsError>;   // Callback(T) | Manual(String)
```

Behavior, as upstream:

- `start` fails with `Login cancelled` when the cancel token already fired, and with the bind error when the port is taken (no fallback; callers such as Anthropic retry with port 0 themselves). The `std::io::Error` is the error's `source()`.
- `redirect_uri` is `http://<redirect_host or host>:<bound port><path>`; a host containing `:` is bracketed.
- Request handling, in order: not `GET <path>` → 404 `Callback route not found.`; expected state and `state` differs → 400 `State mismatch.`; already claimed or settled → 409 `This sign-in has already been handled.`; non-empty `error` → 400 page `<provider> authorization failed.` with `error_description` (or `error`) as details, and the wait fails with `<provider> authorization failed: <description>`; no `code` → 400 `Missing authorization code.`. Otherwise the callback is claimed and `complete(code)` runs before the page is sent: success → 200 `Signed in to <provider>. You may now close this page.`; failure → 502 `<provider> sign-in failed.` with the error message as details, and the wait fails with that error.
- Pages are `text/html; charset=utf-8` with `cache-control: no-store`; their markup is upstream's `oauth-page.ts`, byte for byte.
- `cancel()` resolves the wait with `None` unless a callback is already being completed. `close()` fails a pending wait with `OAuth callback server closed` and stops listening. The cancel token fails the wait with `Login cancelled`; the timeout with `<provider> sign-in timed out`.
- `wait_for_callback_or_manual_input` sends an `AuthPrompt::ManualCode` whose `cancel` token fires when the function returns. When the prompt answers first, the server is cancelled; a callback that is already being completed still wins. A failed prompt fails the call. Without a server only the prompt is used.

### Device-code polling

```rust
pub struct OAuthDeviceCodePollOptions {
    pub interval_seconds: Option<f64>,
    pub expires_in_seconds: Option<f64>,
    pub wait_before_first_poll: bool,
    pub cancel: CancellationToken,      // upstream `signal`
}
pub enum OAuthDeviceCodePollResult<T> {
    Pending, SlowDown { interval_seconds: Option<f64> }, Failed { message: String }, Complete(T),
}
pub async fn poll_oauth_device_code_flow<T, F, Fut>(options, poll: F) -> Result<T, ModelsError>
where F: FnMut() -> Fut, Fut: Future<Output = Result<OAuthDeviceCodePollResult<T>, ModelsError>>;
pub async fn abortable_sleep(duration, &CancellationToken, cancel_message) -> Result<(), ModelsError>;
```

Rules (RFC 8628, as upstream): the interval starts at `interval_seconds` or 5 s, never below 1 s. The first poll runs at once unless `wait_before_first_poll`. `SlowDown` with a finite positive server interval sets the interval to it (floor 1 s); otherwise the interval grows by 5 s. Sleeps are cut at the deadline. Error texts: `Login cancelled`, `Device flow timed out`, and after any `slow_down` the long clock-drift hint (`SLOW_DOWN_TIMEOUT_MESSAGE`). `Failed { message }` fails with `message`.

### Other helpers

- `oauth_callback_host()`: `PI_OAUTH_CALLBACK_HOST`, else `127.0.0.1`.
- `login_cancelled()`, `oauth_error(message)`: the two error kinds flows return.
- `form_urlencode(pairs)`, `search_param(query, name)`, `url_search_param(&Url, name)`: `URLSearchParams` serialization and lookup.
- `oauth_http_client()`: the default client of `delta_ai::http`.
- `format_error_details(&error)`: message plus `; cause=` chain (Anthropic's `formatErrorDetails`).

### Errors

Flows return `ModelsError`. Cancellation is `Login cancelled` with code `Aborted`. Everything else has code `OAuth` and upstream's message text.

## Flows

Each flow is a struct implementing `OAuthAuth`. `Default` uses upstream's endpoints. The endpoint fields are public so tests can point them at wiremock; upstream tests stub `fetch` instead.

### Anthropic (`AnthropicOAuth`)

- Name `Anthropic (Claude Pro/Max)`, subscription, no login label.
- `login` first asks a `Select` prompt `Select Anthropic login method:` with `browser` (`Browser login (default)`) and `copy_code` (`Copy code login (headless)`). Other answers fail with `Unknown Anthropic login method: <id>`.
- Browser login: PKCE (the verifier doubles as `state`); a callback server on `callback_host:53692/callback` with `redirect_host` `localhost` and the state check; if port 53692 is taken, port 0; if that fails too, no server and the redirect URI `http://localhost:53692/callback`. It emits `AuthUrl` with the authorize URL (`code=true`, `client_id`, `response_type=code`, `redirect_uri`, `scope`, `code_challenge`, `code_challenge_method=S256`, `state`), then races the callback against a pasted code / redirect URL. Pasted input is parsed as a URL, `code#state`, a query string, or a bare code; a different non-empty state fails with `OAuth state mismatch`; no code fails with `Missing authorization code`. Then `Progress` `Exchanging authorization code for tokens...` and the token exchange.
- Copy-code login: the redirect URI is `https://platform.claude.com/oauth/code/callback`; the user pastes `code#state` into a `ManualCode` prompt.
- Token exchange and refresh: JSON `POST` to `https://platform.claude.com/v1/oauth/token`, 30 s timeout. The refresh body has no `scope`. Errors keep upstream's text (`Token exchange request failed. url=…; redirect_uri=…; response_type=authorization_code; details=…`, `Anthropic token refresh request failed. url=…; details=…`, the `invalid JSON` variants). `expires` is now + `expires_in` − 5 minutes.
- `to_auth`: `api_key = access`. The access token starts with `sk-ant-oat`, which `anthropic-messages` already detects: it sends `Authorization: Bearer`, the `claude-code-20250219` and `oauth-2025-04-20` betas and the Claude Code identity (see [api-anthropic-messages.md](api-anthropic-messages.md)). Nothing else changes in the request path.

### OpenRouter (`OpenRouterOAuth`)

- Name `OpenRouter OAuth`, login label `Sign in with OpenRouter`, not a subscription.
- Login: PKCE; a callback server on `callback_host`, port 0, path `/oauth/callback/<uuid v4>` (OpenRouter sends no `state`), 5-minute timeout, and `complete` runs the key exchange so the browser page shows exchange failures. Events: `Progress` `Listening for OpenRouter OAuth callback on <uri>`, then `AuthUrl` for `https://openrouter.ai/auth?callback_url=…&code_challenge=…&code_challenge_method=S256`. A pasted URL, query string or bare code is exchanged after `Progress` `Exchanging authorization code for an API key...`.
- Exchange: JSON `POST https://openrouter.ai/api/v1/auth/keys` with `code`, `code_verifier`, `code_challenge_method`; 30 s timeout (`OpenRouter OAuth token exchange timed out`). Non-2xx → `OpenRouter OAuth key exchange failed (HTTP <status>)[: <detail>]`, the detail from `error_description`, `message`, `error`, or `error.message`. No `key` → `OpenRouter OAuth response carries no "key"`.
- Credential: `access = key`, `refresh = ""`, `expires = 9007199254740991` (`Number.MAX_SAFE_INTEGER`). `refresh` returns the credential unchanged; `to_auth` gives `api_key = access`.

### xAI (`XaiOAuth`)

- Name `xAI (Grok/X subscription)`, subscription, login label `Sign in with SuperGrok or X Premium`.
- Login: form `POST https://auth.x.ai/oauth2/device/code` (`client_id`, `scope`, `referrer=pi`). The response is validated: `device_code`, `user_code`, `verification_uri` non-empty strings, `expires_in` positive, `interval` used only when positive; `verification_uri` and `verification_uri_complete` must be `https` URLs (`Untrusted verification URI in xAI OAuth response`). A `DeviceCode` event carries `verification_uri_complete` when present. Polling uses `poll_oauth_device_code_flow` with `wait_before_first_poll`; `authorization_pending` → pending, `slow_down` → server interval, `access_denied`/`authorization_denied` → `xAI device authorization was denied`, `expired_token` → `xAI device code expired`, other errors → `xAI OAuth device token polling failed (HTTP <status>)[: <error>: <description>]`.
- Refresh: form `POST https://auth.x.ai/oauth2/token` with `grant_type=refresh_token`. A missing `refresh_token` keeps the old one; a missing `expires_in` means one hour; `expires` is now + lifetime − 5 minutes. Field errors: `Invalid xAI OAuth response field: <field>`.
- `to_auth`: `api_key = access`.

### Kimi Code (`KimiCodingOAuth`)

- Name `Kimi Code (subscription)`, subscription, login label `Sign in with Kimi Code`.
- OAuth host: the `oauth_host` field, else `KIMI_CODE_OAUTH_HOST`, else `KIMI_OAUTH_HOST`, else `https://auth.kimi.com`. Trailing slashes are removed. The host is resolved at each login and refresh.
- Requests are form `POST`s with `Accept: application/json` and a 30 s timeout. A response body counts only when it parses as a JSON object or array (upstream `readJson`).
- Login: `POST <host>/api/oauth/device_authorization` with `client_id`. Non-2xx → `Kimi Code device authorization failed with status <status>[: <body text>]`. `device_code`, `user_code`, `verification_uri` and `verification_uri_complete` must be strings, and both URIs must be `http` or `https` URLs; otherwise `Invalid Kimi Code device authorization response: <JSON>`. `interval` defaults to 5 s and `expires_in` to 15 minutes when they are not positive. The `DeviceCode` event carries `verification_uri_complete`.
- Polling: `POST <host>/api/oauth/token` with `client_id`, `device_code` and the device-code grant type, through `poll_oauth_device_code_flow` with `wait_before_first_poll`. Status 500 or higher → `Kimi Code device token request failed with status <status>[: <body text>]`. A 2xx with a string `access_token` completes. `authorization_pending` → pending; `slow_down` → the server's positive `interval`; `expired_token` → `Kimi Code device authorization expired. Please restart login.`; `access_denied` → `Kimi Code login was denied.`; anything else → `Kimi Code device token request failed (status <status>)[: <error>[: <description>]]`.
- A token response needs a non-empty `access_token` and `refresh_token` and a positive `expires_in`, else `Kimi Code token <poll|refresh> response missing fields: <JSON>`. `expires` is now + `expires_in`, with no skew.
- Refresh: `POST <host>/api/oauth/token` with `client_id`, `grant_type=refresh_token` and `refresh_token`. Network errors, 429 and 5xx are retried up to three times, after 1 s, 2 s and 4 s. 401, 403 or `invalid_grant` fail at once with `Kimi Code token refresh unauthorized (status <status>)[: <description>]`. Other failures: `Kimi Code token refresh failed with status <status>: <JSON>`. When the retries run out, the last error is returned.
- `to_auth`: `headers = {Authorization: Bearer <access>}` and no API key. `openai-completions` uses the bearer key `unused` when an `Authorization` header is present, and the auth header replaces the default one, so the request carries exactly `Authorization: Bearer <access>`. The base URL and the model list do not change.

### Meta (`MetaOAuth`)

- Name `Meta (Muse subscription)`, subscription, login label `Sign in with Meta`.
- Meta splits identity from API access. The device flow yields an identity token, which is exchanged for a Model API key at the Muse Code key-mint endpoint. The credential stores the identity token as `refresh` and the key as `access`, with `expires` now + 24 hours, so the standard refresh mints a new key. The identity token itself cannot be renewed.
- Login: form `POST https://auth.meta.com/oidc/device/authorization/` with `client_id`. Non-2xx → `Meta device authorization failed with status <status><detail>`, where the detail is `: ` and the first non-blank of `error_description`, `detail`, `message`, `error` (trimmed). The verification URI is `verification_uri_complete` if it is an `http(s)` URL, else `verification_uri` under the same rule. It must be present, and `device_code` and `user_code` must be non-empty strings, else `Invalid Meta device authorization response: <JSON>`. `interval` and `expires_in` are used only when positive; without `expires_in` polling runs until cancelled.
- Polling: form `POST https://auth.meta.com/oidc/device/token/` with the device-code grant type, `device_code` and `client_id`, with `wait_before_first_poll`. A 2xx with a non-empty `access_token` completes. `authorization_pending` → pending; `slow_down` → the positive `interval`; `access_denied` → `Meta login was denied.`; `expired_token` → `Meta device authorization expired. Please restart login.`; anything else → `Meta device token request failed with status <status><detail>`.
- Then `Progress` `Enabling Meta Model API access...` and the mint: `POST https://api.meta.ai/muse-code/key` with body `{}` and headers `Accept: application/json`, `Authorization: Bearer <identity token>`, `Content-Type: application/json`, `x-api-version: 1.0.0`. 401/403 → ``Meta session expired (status <status>). Run `/login meta` to sign in again.<detail>``; other non-2xx → `Meta API key mint failed with status <status><detail>`; no `api_key` → `Meta did not issue an API key.[ Complete setup at <action_url>]` (only an `http(s)` `action_url` is shown).
- A login that fails after the cancel token fired reports `Login cancelled`.
- Refresh: the mint request with the stored identity token.
- `to_auth`: `api_key = access`. The `openai-responses` request does not change otherwise.

### Provider wiring

| Provider | Upstream | Rust |
| --- | --- | --- |
| `anthropic` | `auth: { apiKey, oauth: lazyOAuth(anthropic) }` | `anthropic_provider()`: `ProviderAuth::api_key(..).with_oauth(AnthropicOAuth::default())` |
| `openrouter` | `auth: { apiKey: envApiKeyAuth(..), oauth: lazyOAuth(openrouter) }` | `builtin_oauth("openrouter")` in `providers::builtin` |
| `xai` | `auth: { apiKey: envApiKeyAuth(..), oauth: lazyOAuth(xai) }` | `builtin_oauth("xai")` |
| `kimi-coding` | `auth: { apiKey: envApiKeyAuth(..), oauth: lazyOAuth(kimiCoding) }` | `builtin_oauth("kimi-coding")` |
| `meta` | `auth: { apiKey: envApiKeyAuth(..), oauth: lazyOAuth(meta) }` | `builtin_oauth("meta")` |

`builtin_oauth(id)` adds the OAuth method to a table-driven provider. Later flows add their entries there.

## CLI (`delta-ai`)

`src/bin/delta-ai.rs`, built only with `--features cli` (`[[bin]] required-features = ["cli"]`), so the library build is unchanged.

- `delta-ai`, `help`, `--help`, `-h`: usage and the providers with OAuth (`  <id padded to 20> <name>`).
- `delta-ai list`: `<id padded to 20> <name>` per provider with OAuth.
- `delta-ai login [provider]`: without a provider, a numbered list and `Enter number (1-N): `. Unknown providers fail with `Unknown provider: <id>`, unknown commands with `Unknown command: <command>`. Errors print `Error: <message>` to stderr and exit 1.
- Prompts: `Select` prints the message and numbered labels and asks `Enter number (1-N): ` (`Invalid selection` otherwise); other prompts ask `<message>[ (<placeholder>)]: `. Events: `AuthUrl` and `DeviceCode` print `Open this URL in your browser:` and the URL (plus instructions or `Enter code: <code>`); `Info` and `Progress` print the message.
- The flow runs `provider.auth().oauth.login(..)` directly, with a fresh UUID as device id. The credential is written to `auth.json` in the current directory: the file is read as a JSON object (missing or invalid → empty), the provider's entry is replaced by the `{"type":"oauth",…}` credential, other entries are kept, and the file is written with two-space indentation (`JSON.stringify(auth, null, 2)`). Then `Credentials saved to auth.json`.

## Rust-specific differences

- Cancellation is a `CancellationToken` (`AuthInteraction::cancel()`, `ManualCode { cancel }`, the helpers' `cancel` fields) instead of an `AbortSignal`. Flows call `interaction.cancel()` once and keep the token.
- `OAuthAuth::refresh` takes no signal (see [auth.md](auth.md)); the Anthropic, xAI, Kimi and Meta refresh requests cannot be cancelled by the caller. `Models` bounds a refresh with its 15 s timeout; Anthropic, Kimi and Meta also keep their own 30 s request timeouts. Kimi's refresh therefore has no `Kimi Code token refresh aborted` error; its backoff sleeps end only when the future is dropped (for example by the `Models` timeout).
- `generate_pkce` is synchronous.
- `OAuthCallbackServer::wait` hands out the result once (a second call fails); upstream returns the same promise each time. Dropping the server closes it.
- `start_oauth_callback_server` is a small HTTP/1.1 handler on Tokio's `TcpListener`: one request per connection, `connection: close`, the request body is ignored. A connection accepted after `close()` is dropped unanswered, so later requests fail as with a closed Node server.
- `poll_oauth_device_code_flow` takes `poll` as its second argument instead of an options field. It measures time with `tokio::time::Instant`, so tests run on paused time instead of fake timers. `interval_seconds` and `expires_in_seconds` are `f64` (JavaScript numbers); the `DeviceCode` event's fields are `u64` and are truncated.
- Anthropic's `formatErrorDetails` has no error name, `code`/`errno` or stack in Rust; `format_error_details` gives the message and the `source()` chain. A token response missing `access_token`, `refresh_token` or `expires_in` is reported as invalid JSON (upstream would store `undefined` values).
- When the Anthropic token exchange or a Kimi login request is cancelled, the error is `Login cancelled` instead of an `AbortError`. A Kimi or Meta request that hits its 30 s timeout fails with reqwest's timeout message instead of the DOM `TimeoutError` text; other network errors carry reqwest's message instead of `fetch failed`.
- Kimi and Meta print response bodies with `serde_json` (`JSON.stringify` upstream). The output is the same compact JSON for the bodies these endpoints return; a float such as `5.0` prints as `5.0` instead of `5`.
- The Kimi host variables are read through delta's `env_value`, so blank values count as unset (upstream treats only the empty string as unset). Tests set the `oauth_host` field; the environment lookup is a unit test.
- Meta's endpoints are public struct fields (`device_authorization_url`, `device_token_url`, `api_key_mint_url`).
- `PI_OAUTH_CALLBACK_HOST` is read when a login starts (upstream's Anthropic module reads it once at import). Blank values count as unset, as everywhere in delta. The flow structs also take a `callback_host` field, which the tests use instead of mutating the process environment (forbidden without `unsafe` in Rust 2024).
- Endpoints are public struct fields (`token_url`, `authorize_url`, `device_code_url`, `callback_port`) so tests can use wiremock and free ports; upstream stubs global `fetch` and binds the real port 53692.
- CLI: the usage line names `delta-ai` instead of `npx @earendil-works/pi-ai`. Stdin is read by a background thread; end of input fails the pending prompt with `Input closed` (Node's readline would wait forever).
- `lazyOAuth`/`load.ts` are not needed: Rust has no bundle-size reason to load flows lazily.

## Not ported

- The other flows: OpenAI Codex, OpenAI ChatGPT, GitHub Copilot, Radius. They are ported separately on top of these helpers.
- `src/oauth.ts` (type-only compatibility exports for coding-agent extensions).
- `registerBundledOAuthFlowLoaders` (Bun standalone binaries).

## Improvements pending confirmation

1. A read timeout for callback connections: a client that connects and never sends a request keeps its task alive (upstream relies on Node's server timeouts).
2. Pass a cancellation token to `OAuthAuth::refresh`, so a refresh request can stop with the caller (needs a change to the framework trait).
3. Device-code event fields could be `f64` to keep fractional intervals exactly.
