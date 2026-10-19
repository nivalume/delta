# Provider: Radius

## Purpose

Radius is a gateway that speaks the `pi-messages` API ([api-pi-messages.md](api-pi-messages.md)). The built-in `radius` provider:

- serves the public `radius.pi.dev` catalog shipped with the crate, so model lookups work before any network access;
- replaces that catalog with the account's gateway catalog (`GET {gateway}/v1/config`) once a refresh with configured auth has run, and persists it in the `ModelsStore`;
- authenticates with `RADIUS_API_KEY` (or a stored key) or with OAuth against the gateway (browser login with PKCE, or device code).

`radius_provider(RadiusProviderOptions { id, name, gateway })` builds the provider for any gateway, e.g. a private deployment registered as `radius-dev`.

Code:

- `crates/ai/src/providers/radius.rs`: gateway config loading, `RadiusProvider`, `radius_provider()`.
- `crates/ai/src/oauth/radius.rs`: `RadiusOAuth`, `create_radius_oauth()`.
- `crates/ai/src/providers/builtin.rs`: `builtin_provider("radius")`.

## Upstream source files

- `packages/ai/src/providers/radius.ts` → `providers::radius::{radius_provider, RadiusProvider, RadiusProviderOptions}`
- `packages/ai/src/providers/radius-config.ts` → `providers::radius::{DEFAULT_RADIUS_GATEWAY, RadiusGatewayConfig, normalize_radius_gateway_url, get_radius_credential_config, get_radius_models_from_config, get_radius_models, load_radius_gateway_config}`
- `packages/ai/src/auth/oauth/radius.ts` → `oauth::radius::{create_radius_oauth, RadiusOAuth, RadiusOAuthOptions}`
- `packages/ai/src/providers/radius.models.ts` and `data/radius.json` → `data/providers/radius.json` (already synced; read through `providers::catalog`)
- `packages/ai/src/env-api-keys.ts` (`radius: "RADIUS_API_KEY"`)
- Tests: `test/radius-oauth.test.ts`, `test/radius-provider.test.ts`, ported to `tests/radius.rs` (one test per `it`, plus device-code outcomes, error texts, the legacy catalog import and a request through `Models` with an OAuth credential). `test/radius-thinking-replay-e2e.test.ts` needs the network and is not ported; see [api-pi-messages.md](api-pi-messages.md).
- Reference: pi-py `pi_ai/providers/radius.py`, `radius_config.py`, `auth/oauth/radius.py`

## Gateway URL

`normalize_radius_gateway_url` adds `https://` when the value has no `http://` or `https://` scheme (case-insensitive) and removes trailing slashes. Gateway endpoints are resolved like `new URL(path, gateway)`: the absolute paths `/v1/config`, `/v1/oauth`, `/v1/oauth/token` and `/v1/oauth/device` replace any path of the gateway URL.

## Catalog

| State | `models()` |
| --- | --- |
| Default gateway, no gateway catalog yet | the shipped catalog, with `provider` set to the provider id |
| Custom gateway, no gateway catalog yet | empty |
| Gateway catalog known (restored or fetched) | exactly the gateway catalog |

Radius organization owners can disable models, so the gateway catalog replaces the baseline instead of overlaying it (unlike `create_provider`'s dynamic overlay). An empty gateway catalog therefore means no models. `is_dynamic()` is true, so `Models::refresh()` refreshes Radius; the shipped catalog stays available for synchronous lookups.

`refresh_models(context)`, as upstream:

1. `context.stored` present: publish its chat models of this provider (`Persist::Keep`) and stop if the publication was superseded.
2. Nothing stored and the credential is OAuth: import the catalog the pre-`ModelsStore` implementation cached in the credential (`gatewayConfig`), publishing it with `Persist::Write`.
3. Stop without network access or when cancelled.
4. `GET {gateway}/v1/config` with `accept: application/json` and `authorization: Bearer <token>` (the OAuth access token, or the API key). Non-2xx fails with `Could not load Radius config from <gateway>: <status>: <body>` (body trimmed, at most 512 characters plus `…`); a body without a string `baseUrl` and a `models` array fails with `Invalid Radius config from <gateway>`. Errors have code `ModelSource`; the previous catalog stays.
5. Publish the models with `Persist::Write { models, checked_at }`.

Config sanitizing keeps model entries with a string `id` and `name`, a boolean `reasoning`, an `input` array, a `cost` object and numeric `contextWindow` and `maxTokens`. `get_radius_models_from_config` copies each entry and sets `api: "pi-messages"`, `provider` and `baseUrl` from the config.

## Auth

| Method | Upstream | Rust |
| --- | --- | --- |
| API key | `envApiKeyAuth("Radius API key", ["RADIUS_API_KEY"])` | `env_api_key_auth("Radius API key", &["RADIUS_API_KEY"])` |
| OAuth | `lazyOAuth({name, load: () => loadRadiusOAuth({name, gateway})})` | `create_radius_oauth(RadiusOAuthOptions { name, gateway })` |

The OAuth flow uses the provider's name and gateway, so a custom gateway signs in against itself.

### OAuth flow (`RadiusOAuth`)

- Name: the `name` option; not a subscription; no login label.
- `login` asks a `Select` prompt `Sign in to <name>:` with `browser` (`Sign in with browser (recommended)`) and `device-code` (`Sign in with device code (when signing in from another device)`). Other answers fail with `Unknown <name> sign-in method: <id>`.
- Browser: `GET {gateway}/v1/oauth` discovers `authorizationEndpoint` (non-2xx: `Could not load Radius OAuth config from <gateway>: <status> <body>`; no string endpoint: `Invalid Radius OAuth config from <gateway>`). Then PKCE, a UUID `state`, and the authorize URL: the endpoint with its query replaced by `response_type=code`, `client_id=pi-gateway`, `redirect_uri`, `scope=gateway offline_access`, `code_challenge`, `code_challenge_method=S256`, `handoff=url`, `state`. A callback server on `127.0.0.1:1456/oauth/callback` checks the state and exchanges the code before answering, so the browser page shows exchange failures. Events: `Progress` `Listening for OAuth callback on <redirect_uri>`, then `AuthUrl` with instructions `Continue in your browser.`. There is no pasted-code fallback and no timeout (as upstream); cancellation stops the wait. A wait that ends without a credential fails with `OAuth callback did not complete.`.
- Device code: form `POST {gateway}/v1/oauth/device` (`client_id`, `scope`). Non-2xx fails with `Radius OAuth device authorization failed: <detail>`; a response without truthy `device_code`, `user_code`, `verification_uri` and `expires_in` fails with `Radius OAuth device authorization response is missing required fields`. A `DeviceCode` event, then `poll_oauth_device_code_flow` (first poll at once): `authorization_pending` → pending, `slow_down` → interval + 5 s, `expired_token` → `Device authorization expired.`, `access_denied` → `Device authorization was denied.`, other OAuth errors fail with the token error.
- Token requests (code exchange, device polling, refresh): form `POST {gateway}/v1/oauth/token`. Non-2xx fails with `Radius OAuth token request failed: <detail>`, where the detail is `<error>: <error_description>`, `<error>`, the description, the raw body when it is not JSON, or the status. The credential is `{access, refresh, expires: now + expires_in − 60 s, scope?}`; `scope` is kept in `extra`.
- `refresh` posts `grant_type=refresh_token` without discovery. `to_auth` gives `api_key = access`.

## Rust-specific differences

- The provider is a hand-written `Provider` (`RadiusProvider`) instead of `create_provider`, because the gateway catalog replaces the baseline rather than overlaying it. The upstream closure state (`dynamicModels`) is an `Arc<RwLock<Option<Vec<Model>>>>` updated in `ModelsPublication::update`.
- `RadiusProvider::with_baseline_models(models)` replaces the shipped baseline. Upstream derives the baseline from the gateway only and its tests stub the global `fetch` for the default gateway; the Rust tests point the provider at a wiremock gateway and set the shipped catalog with this method.
- `RadiusOAuth::callback_port` is public (default 1456) so tests can use a free port; upstream binds 1456. The redirect URI always uses the configured port, so it must not be `0`.
- `lazyOAuth`/`loadRadiusOAuth` are not needed: the flow is compiled into the crate.
- `OAuthAuth::refresh` takes no cancellation token (see [auth.md](auth.md)), so a refresh request cannot be cancelled by the caller.
- Every gateway request is raced against the cancel token and fails with `Login cancelled` (code `Aborted`). Upstream maps only token and device requests to `Login cancelled`; a cancelled discovery request fails there with the `AbortError` text. Config loading fails with `Request was aborted` (code `Aborted`) when the refresh is cancelled.
- A token response without a string `access_token` or `refresh_token`, or without a numeric `expires_in`, fails with `Invalid Radius OAuth token response`. Upstream stores `undefined` in the credential.
- Gateway model entries that pass upstream's field check but do not form a valid `Model` (for example an unknown input modality) are skipped with a warning; upstream keeps them as they are. Fields the `Model` type does not know (`lab`, `providers`, ...) are dropped when the model is built.
- The device `interval` and `expires_in` are passed to the poller as `f64`; the `DeviceCode` event truncates them to whole seconds (see [oauth.md](oauth.md)).
- Body truncation counts characters; upstream counts UTF-16 code units.

## Not ported

- `src/bun-oauth.ts` (`registerBundledOAuthFlowLoaders` for Bun standalone binaries).
- The live e2e test `radius-thinking-replay-e2e.test.ts`.

## Improvements pending confirmation

1. A timeout for the config and OAuth requests. Upstream sets none; a gateway that never answers holds a refresh until the caller cancels it.
2. Fall back to a free callback port when 1456 is taken, as the Anthropic flow does. The gateway would have to accept the changed redirect URI.
