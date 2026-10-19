//! Anthropic workload identity federation: exchange an OIDC identity token for a short-lived
//! Anthropic access token and cache it.
//!
//! Upstream pi hands an `oidc_federation` config to the Anthropic TypeScript SDK
//! (`@anthropic-ai/sdk` 0.129.0), which runs the exchange, caches the token per client and
//! refreshes it. This crate has no SDK, so this module ports that part of the SDK:
//! `lib/credentials/identity-token.ts` (`identityTokenFromFile`), `oidc-federation.ts`
//! (`oidcFederationProvider`), `token-cache.ts` (`TokenCache`) and the helpers in `types.ts`.
//! `api::anthropic_messages` decides when federation applies (upstream
//! `getAnthropicFederation`) and sends the token.
//!
//! Design notes: `docs/design/anthropic-federation.md`.

use serde_json::{Map, Value};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{Error, Result};
use crate::http;

/// Tagged id (`fdrl_...`) of the federation rule. Required for federation.
pub const ANTHROPIC_FEDERATION_RULE_ID_ENV: &str = "ANTHROPIC_FEDERATION_RULE_ID";
/// Organization id. Required for federation.
pub const ANTHROPIC_ORGANIZATION_ID_ENV: &str = "ANTHROPIC_ORGANIZATION_ID";
/// Optional `svac_...` service account id the token must target.
pub const ANTHROPIC_SERVICE_ACCOUNT_ID_ENV: &str = "ANTHROPIC_SERVICE_ACCOUNT_ID";
/// Path of the file holding the OIDC identity token (JWT). Required for federation; read on
/// every exchange, so rotated tokens (Kubernetes projected tokens) are picked up.
pub const ANTHROPIC_IDENTITY_TOKEN_FILE_ENV: &str = "ANTHROPIC_IDENTITY_TOKEN_FILE";
/// Optional `wrkspc_...` id (or `default`) the minted token is scoped to.
pub const ANTHROPIC_WORKSPACE_ID_ENV: &str = "ANTHROPIC_WORKSPACE_ID";

/// The token endpoint, relative to the base URL.
pub const TOKEN_ENDPOINT: &str = "/v1/oauth/token";
/// RFC 7523 jwt-bearer grant type.
pub const GRANT_TYPE_JWT_BEARER: &str = "urn:ietf:params:oauth:grant-type:jwt-bearer";
/// `anthropic-beta` value required on requests authenticated with an OAuth bearer token.
pub const OAUTH_API_BETA_HEADER: &str = "oauth-2025-04-20";
/// `anthropic-beta` value that routes a jwt-bearer exchange to the federation service.
pub const FEDERATION_BETA_HEADER: &str = "oidc-federation-2026-04-01";

/// More than this many seconds left: the cached token is used as is.
const ADVISORY_REFRESH_THRESHOLD_IN_SECONDS: f64 = 120.0;
/// Fewer than this many seconds left: the request waits for a fresh token.
const MANDATORY_REFRESH_THRESHOLD_IN_SECONDS: f64 = 30.0;
/// Pause after a failed background refresh before the next one.
const ADVISORY_REFRESH_BACKOFF_IN_SECONDS: f64 = 5.0;
/// The token endpoint rejects larger assertions.
const MAX_ASSERTION_BYTES: usize = 16 * 1024;
const MAX_TOKEN_RESPONSE_BYTES: usize = 1 << 20;
const MAX_ERROR_BODY_CHARS: usize = 2000;
/// RFC 6749 §5.2 error fields: the only keys of a token endpoint body kept in error messages.
const SAFE_ERROR_KEYS: [&str; 3] = ["error", "error_description", "error_uri"];

/// An `oidc_federation` config, the fields upstream builds from the federation variables
/// (`organization_id`, `workspace_id`, `authentication.federation_rule_id`,
/// `authentication.service_account_id`, `authentication.identity_token.path`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AnthropicFederationConfig {
    pub organization_id: String,
    pub workspace_id: Option<String>,
    pub federation_rule_id: String,
    pub service_account_id: Option<String>,
    pub identity_token_file: String,
}

/// A minted access token.
#[derive(Clone, Debug, PartialEq)]
pub struct AccessToken {
    pub token: String,
    /// Unix epoch seconds.
    pub expires_at: f64,
}

/// Current time as whole Unix epoch seconds (the SDK's `nowAsSeconds`).
fn now_as_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs() as f64)
}

/// Read the identity token from `path` on every call, trimmed (SDK `identityTokenFromFile`).
fn read_identity_token(path: &str) -> Result<String> {
    let bytes = std::fs::read(path)
        .map_err(|error| Error::msg(format!("Failed to read identity token file at {path}: {error}")))?;
    let content = String::from_utf8_lossy(&bytes);
    let token = http::js_trim(&content);
    if token.is_empty() {
        return Err(Error::msg(format!("Identity token file at {path} is empty")));
    }
    Ok(token.to_owned())
}

/// Reject base URLs that would send the identity token over cleartext HTTP. Loopback hosts
/// are allowed for local development.
pub fn require_secure_token_endpoint(base_url: &str) -> Result<()> {
    if base_url.is_empty() {
        return Ok(());
    }
    let url = url::Url::parse(base_url)
        .map_err(|error| Error::msg(format!("Invalid token endpoint base URL \"{base_url}\": {error}")))?;
    if url.scheme() == "https" {
        return Ok(());
    }
    let host = url
        .host_str()
        .unwrap_or_default()
        .to_lowercase()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_owned();
    if url.scheme() == "http" && matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1") {
        return Ok(());
    }
    Err(Error::msg(format!(
        "Refusing to send credential over non-https token endpoint \"{base_url}\""
    )))
}

/// `text` cut to `max` UTF-16 code units (JavaScript string length), noting what was dropped.
fn truncate_js(text: &str, max: usize) -> String {
    let length = text.encode_utf16().count();
    if length <= max {
        return text.to_owned();
    }
    let mut units = 0;
    let mut kept = String::new();
    for c in text.chars() {
        units += c.len_utf16();
        if units > max {
            break;
        }
        kept.push(c);
    }
    format!("{kept}... <{} more chars>", length - max)
}

/// A redacted copy of a token endpoint body for error messages (SDK `redactSensitive`):
/// strings are truncated (or, when they hold JSON, parsed, redacted and serialized again),
/// objects keep only the RFC 6749 error fields, anything else becomes `null`.
pub fn redact_sensitive(body: &Value) -> Value {
    match body {
        Value::String(text) => match serde_json::from_str::<Value>(text) {
            Ok(parsed) => Value::String(redact_sensitive(&parsed).to_string()),
            Err(_) => Value::String(truncate_js(text, MAX_ERROR_BODY_CHARS)),
        },
        Value::Object(object) => Value::Object(
            object
                .iter()
                .filter(|(key, _)| SAFE_ERROR_KEYS.contains(&key.as_str()))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        ),
        // `null` stays `null`; arrays and other values become `null`.
        _ => Value::Null,
    }
}

/// A redacted token endpoint body as text: what `${redactSensitive(text)}` prints.
fn redact_text(text: &str) -> String {
    match redact_sensitive(&Value::String(text.to_owned())) {
        Value::String(text) => text,
        other => other.to_string(),
    }
}

/// JavaScript `Number(value)` for the JSON values a token response can carry.
fn js_number(value: Option<&Value>) -> f64 {
    match value {
        Some(Value::Number(number)) => number.as_f64().unwrap_or(f64::NAN),
        Some(Value::String(text)) => {
            let text = http::js_trim(text);
            if text.is_empty() {
                0.0
            } else {
                text.parse().unwrap_or(f64::NAN)
            }
        }
        Some(Value::Bool(flag)) => f64::from(u8::from(*flag)),
        Some(Value::Null) => 0.0,
        _ => f64::NAN,
    }
}

/// The `access_token` field when it is truthy in JavaScript (a non-empty string, a non-zero
/// number or `true`).
fn access_token_of(data: &Value) -> Option<String> {
    match data.get("access_token")? {
        Value::String(token) if !token.is_empty() => Some(token.clone()),
        Value::Number(number) if number.as_f64().is_some_and(|value| value != 0.0) => Some(number.to_string()),
        Value::Bool(true) => Some("true".into()),
        _ => None,
    }
}

/// Read at most [`MAX_TOKEN_RESPONSE_BYTES`] of the body as UTF-8 (lossy).
async fn read_limited_text(mut response: reqwest::Response) -> Result<String> {
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if body.len() + chunk.len() > MAX_TOKEN_RESPONSE_BYTES {
            body.extend_from_slice(&chunk[..MAX_TOKEN_RESPONSE_BYTES - body.len()]);
            break;
        }
        body.extend_from_slice(&chunk);
    }
    Ok(String::from_utf8_lossy(&body).into_owned())
}

/// Parse a token endpoint response: JSON with a truthy `access_token` and, when present, a
/// Bearer `token_type` (SDK `parseTokenResponse`).
async fn parse_token_response(response: reqwest::Response, status: u16) -> Result<(Value, String)> {
    let text = read_limited_text(response).await?;
    let Ok(data) = serde_json::from_str::<Value>(&text) else {
        return Err(Error::msg(format!(
            "Token endpoint returned non-JSON response (status {status})"
        )));
    };
    let Some(token) = access_token_of(&data) else {
        return Err(Error::msg(format!(
            "Token endpoint response missing access_token: {}",
            redact_sensitive(&data)
        )));
    };
    if let Some(Value::String(token_type)) = data.get("token_type")
        && !token_type.is_empty()
        && token_type.to_lowercase() != "bearer"
    {
        return Err(Error::msg(format!(
            "Token endpoint response: unsupported token_type \"{token_type}\" (want Bearer)"
        )));
    }
    Ok((data, token))
}

/// One RFC 7523 jwt-bearer exchange against `{base_url}/v1/oauth/token` (SDK
/// `oidcFederationProvider`). `base_url` has no trailing slash.
pub async fn exchange_identity_token(
    client: &reqwest::Client,
    config: &AnthropicFederationConfig,
    base_url: &str,
) -> Result<AccessToken> {
    require_secure_token_endpoint(base_url)?;

    let jwt = read_identity_token(&config.identity_token_file)?;
    let length = jwt.encode_utf16().count();
    if length > MAX_ASSERTION_BYTES {
        return Err(Error::msg(format!(
            "Identity token is {} KiB, exceeds the 16 KiB assertion limit",
            length.div_ceil(1024)
        )));
    }

    let mut body = Map::new();
    body.insert("grant_type".into(), GRANT_TYPE_JWT_BEARER.into());
    body.insert("assertion".into(), jwt.into());
    body.insert("federation_rule_id".into(), config.federation_rule_id.clone().into());
    body.insert("organization_id".into(), config.organization_id.clone().into());
    if let Some(id) = config.service_account_id.as_deref().filter(|id| !id.is_empty()) {
        body.insert("service_account_id".into(), id.into());
    }
    if let Some(id) = config.workspace_id.as_deref().filter(|id| !id.is_empty()) {
        body.insert("workspace_id".into(), id.into());
    }

    let url = format!("{base_url}{TOKEN_ENDPOINT}");
    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .header(
            "anthropic-beta",
            format!("{OAUTH_API_BETA_HEADER},{FEDERATION_BETA_HEADER}"),
        )
        .header("User-Agent", http::user_agent())
        .body(Value::Object(body).to_string())
        .send()
        .await
        .map_err(|error| Error::msg(format!("Failed to reach token endpoint {url}: {error}")))?;

    let status = response.status().as_u16();
    let request_id = response
        .headers()
        .get("Request-Id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);

    if !response.status().is_success() {
        let text = response.text().await.unwrap_or_default();
        let redacted = redact_text(&text);
        // A 401 is hard to debug from the status alone, so point at the likely fixes.
        let hint = if status == 401 {
            let middle = if config.workspace_id.as_deref().is_some_and(|id| !id.is_empty()) {
                ""
            } else {
                "If your federation rule is scoped to multiple workspaces, set the ANTHROPIC_WORKSPACE_ID environment variable, the 'workspace_id' config key, or the `workspaceId` option. "
            };
            format!(
                " Ensure your federation rule matches your identity token. {middle}View your authentication events in the Workload identity page of Claude Console for more details."
            )
        } else {
            String::new()
        };
        let request_id = request_id.map(|id| format!(" (request-id {id})")).unwrap_or_default();
        return Err(Error::msg(format!(
            "Token exchange failed with status {status}{request_id}: {redacted}{hint}"
        )));
    }

    let (data, token) = parse_token_response(response, status).await?;
    let expires_in = js_number(data.get("expires_in"));
    if !expires_in.is_finite() {
        return Err(Error::msg(format!(
            "Token endpoint response missing required fields: {}",
            redact_sensitive(&data)
        )));
    }
    Ok(AccessToken {
        token,
        expires_at: now_as_seconds() + expires_in,
    })
}

#[derive(Debug, Default)]
struct CacheState {
    cached: Option<AccessToken>,
    /// When the last background refresh failed (epoch seconds).
    last_advisory_error: f64,
}

/// Access token cache for one federation config (SDK `TokenCache` around
/// `oidcFederationProvider`), with two-tier proactive refresh:
///
/// - no cached token: exchange and wait;
/// - more than 120 s left: use the cached token;
/// - 30 to 120 s left: use the cached token and refresh in the background (a failure is
///   ignored and pauses background refreshes for 5 s);
/// - less than 30 s left: exchange and wait; a failure fails the request.
///
/// Concurrent waiting callers share one exchange.
#[derive(Debug)]
pub struct FederationTokenCache {
    config: AnthropicFederationConfig,
    /// The token endpoint host, without trailing slashes.
    base_url: String,
    state: Mutex<CacheState>,
    /// Held for the duration of an exchange.
    refresh: Arc<tokio::sync::Mutex<()>>,
}

impl FederationTokenCache {
    /// A cache that exchanges against `{base_url}/v1/oauth/token`.
    pub fn new(config: AnthropicFederationConfig, base_url: &str) -> Self {
        Self {
            config,
            base_url: base_url.trim_end_matches('/').to_owned(),
            state: Mutex::new(CacheState::default()),
            refresh: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    /// The federation config this cache mints tokens for.
    pub fn config(&self) -> &AnthropicFederationConfig {
        &self.config
    }

    fn state(&self) -> std::sync::MutexGuard<'_, CacheState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The cached token and its remaining lifetime in seconds.
    fn cached(&self) -> Option<(String, f64)> {
        let state = self.state();
        let cached = state.cached.as_ref()?;
        Some((cached.token.clone(), cached.expires_at - now_as_seconds()))
    }

    /// A valid access token; `client` sends the exchange when one is needed.
    pub async fn get_token(self: &Arc<Self>, client: &reqwest::Client) -> Result<String> {
        match self.cached() {
            Some((token, remaining)) if remaining > ADVISORY_REFRESH_THRESHOLD_IN_SECONDS => Ok(token),
            Some((token, remaining)) if remaining > MANDATORY_REFRESH_THRESHOLD_IN_SECONDS => {
                self.background_refresh(client);
                Ok(token)
            }
            _ => self.refresh(client).await,
        }
    }

    /// Drop the cached token, so the next request exchanges again. Called after a 401: the
    /// server rejected the token even if its expiry looks fine.
    pub fn invalidate(&self) {
        self.state().cached = None;
    }

    /// Mandatory refresh. A caller that waited for another caller's exchange uses its token.
    async fn refresh(&self, client: &reqwest::Client) -> Result<String> {
        let _guard = self.refresh.lock().await;
        if let Some((token, remaining)) = self.cached()
            && remaining > MANDATORY_REFRESH_THRESHOLD_IN_SECONDS
        {
            return Ok(token);
        }
        let token = exchange_identity_token(client, &self.config, &self.base_url).await?;
        let value = token.token.clone();
        self.state().cached = Some(token);
        Ok(value)
    }

    /// Advisory refresh in a background task, skipped while an exchange runs or within the
    /// backoff after a failed one.
    fn background_refresh(self: &Arc<Self>, client: &reqwest::Client) {
        let Ok(guard) = Arc::clone(&self.refresh).try_lock_owned() else {
            return;
        };
        if now_as_seconds() - self.state().last_advisory_error < ADVISORY_REFRESH_BACKOFF_IN_SECONDS {
            return;
        }
        let cache = Arc::clone(self);
        let client = client.clone();
        tokio::spawn(async move {
            let _guard = guard;
            match exchange_identity_token(&client, &cache.config, &cache.base_url).await {
                Ok(token) => cache.state().cached = Some(token),
                // Keep serving the cached token.
                Err(_) => cache.state().last_advisory_error = now_as_seconds(),
            }
        });
    }
}

/// The token cache of the current federation config. Upstream keeps one SDK client for the
/// current `[baseUrl, config]` pair and replaces it when the pair changes, so requests with
/// the same config share the token.
static FEDERATION_CACHE: Mutex<Option<(String, Arc<FederationTokenCache>)>> = Mutex::new(None);

/// The shared token cache for `config` against `base_url` (the model's base URL).
pub fn federation_token_cache(base_url: &str, config: &AnthropicFederationConfig) -> Arc<FederationTokenCache> {
    let mut slot = FEDERATION_CACHE.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((cached_base_url, cache)) = &*slot
        && cached_base_url == base_url
        && cache.config() == config
    {
        return Arc::clone(cache);
    }
    let cache = Arc::new(FederationTokenCache::new(config.clone(), base_url));
    *slot = Some((base_url.to_owned(), Arc::clone(&cache)));
    cache
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    #[test]
    fn secure_token_endpoint() {
        for url in [
            "",
            "https://api.anthropic.com",
            "http://localhost:8080",
            "http://127.0.0.1:1234",
            "http://[::1]:80",
            "http://LOCALHOST",
        ] {
            assert!(require_secure_token_endpoint(url).is_ok(), "{url}");
        }
        assert_eq!(
            require_secure_token_endpoint("http://api.anthropic.com")
                .unwrap_err()
                .to_string(),
            "Refusing to send credential over non-https token endpoint \"http://api.anthropic.com\""
        );
        assert!(
            require_secure_token_endpoint("not a url")
                .unwrap_err()
                .to_string()
                .starts_with("Invalid token endpoint base URL \"not a url\": ")
        );
    }

    #[test]
    fn redacts_token_endpoint_bodies() {
        let body = json!({"error": "invalid_grant", "error_description": "bad", "assertion": "secret"});
        assert_eq!(
            redact_sensitive(&body),
            json!({"error": "invalid_grant", "error_description": "bad"})
        );
        assert_eq!(
            redact_text(r#"{"error":"invalid_grant","access_token":"secret"}"#),
            r#"{"error":"invalid_grant"}"#
        );
        assert_eq!(redact_text("plain text"), "plain text");
        assert_eq!(redact_text(""), "");
        assert_eq!(redact_text("[1, 2]"), "null");
        let long = "x".repeat(2005);
        assert_eq!(redact_text(&long), format!("{}... <5 more chars>", "x".repeat(2000)));
        assert_eq!(redact_sensitive(&json!([1])), Value::Null);
    }

    #[test]
    #[allow(clippy::float_cmp)] // Exact values.
    fn numbers_like_javascript() {
        assert_eq!(js_number(Some(&json!(3600))), 3600.0);
        assert_eq!(js_number(Some(&json!(" 60 "))), 60.0);
        assert_eq!(js_number(Some(&Value::Null)), 0.0);
        assert!(js_number(None).is_nan());
        assert!(js_number(Some(&json!("soon"))).is_nan());
        assert!(js_number(Some(&json!({}))).is_nan());
    }

    #[test]
    fn reuses_the_cache_for_the_same_config() {
        let config = AnthropicFederationConfig {
            organization_id: "org".into(),
            workspace_id: None,
            federation_rule_id: "fdrl".into(),
            service_account_id: None,
            identity_token_file: "/tmp/token".into(),
        };
        let first = federation_token_cache("https://unit.test", &config);
        let second = federation_token_cache("https://unit.test", &config);
        assert!(Arc::ptr_eq(&first, &second));
        let other = AnthropicFederationConfig {
            workspace_id: Some("wrkspc".into()),
            ..config
        };
        let third = federation_token_cache("https://unit.test", &other);
        assert!(!Arc::ptr_eq(&first, &third));
    }
}
