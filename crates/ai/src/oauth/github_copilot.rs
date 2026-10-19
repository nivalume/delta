//! GitHub Copilot OAuth: the GitHub device flow, the Copilot token exchange, the account's
//! model catalog and model policy enablement. Port of upstream `auth/oauth/github-copilot.ts`.
//!
//! The stored credential's `refresh` is the GitHub access token and `access` the short-lived
//! Copilot token. Extra credential fields (upstream's JSON names): `enterpriseUrl` (the
//! GitHub Enterprise domain, absent for github.com) and `availableModelIds` (the models the
//! account may use, read by the provider's model filter).

use base64::Engine as _;
use futures::future::BoxFuture;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::auth::{AuthEvent, AuthInteraction, AuthPrompt, LoginOptions, ModelAuth, OAuthAuth, OAuthCredential};
use crate::error::ModelsError;
use crate::http::{httpdate_to_unix, is_js_whitespace, js_trim};
use crate::oauth::{
    CANCEL_MESSAGE, OAuthDeviceCodePollOptions, OAuthDeviceCodePollResult, abortable_sleep, form_urlencode,
    format_error_details, login_cancelled, oauth_error, oauth_http_client, poll_oauth_device_code_flow,
};
use crate::providers::catalog::all_builtin_models;
use crate::types::{JsonObject, now_ms};

/// Upstream keeps the client id base64-encoded in its source; so does this port.
const CLIENT_ID_BASE64: &str = "SXYxLmI1MDdhMDhjODdlY2ZlOTg=";

/// The `User-Agent` of every GitHub and Copilot request.
pub const USER_AGENT: &str = "GitHubCopilotChat/0.35.0";
/// Headers that identify the client as VS Code Copilot Chat (upstream `COPILOT_HEADERS`).
pub const COPILOT_HEADERS: [(&str, &str); 4] = [
    ("User-Agent", USER_AGENT),
    ("Editor-Version", "vscode/1.107.0"),
    ("Editor-Plugin-Version", "copilot-chat/0.35.0"),
    ("Copilot-Integration-Id", "vscode-chat"),
];
/// `X-GitHub-Api-Version` of the model catalog request.
pub const COPILOT_API_VERSION: &str = "2026-06-01";
/// The Copilot API of individual accounts, used when neither the token nor an enterprise
/// domain names another one.
pub const INDIVIDUAL_BASE_URL: &str = "https://api.individual.githubcopilot.com";

/// Each catalog or policy request gives up after 5 seconds (upstream `AbortSignal.timeout(5000)`).
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const REFRESH_SKEW_MS: i64 = 5 * 60 * 1000;

/// The GitHub OAuth client id of VS Code Copilot Chat.
pub fn client_id() -> String {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(CLIENT_ID_BASE64)
        .unwrap_or_default();
    String::from_utf8(bytes).unwrap_or_default()
}

/// Maps a request URL before it is sent.
pub type UrlRewrite = Arc<dyn Fn(&str) -> String + Send + Sync>;

/// GitHub Copilot OAuth (`github-copilot`). `Default` sends every request to upstream's URLs.
#[derive(Clone, Default)]
pub struct GitHubCopilotOAuth {
    /// Rewrites each request URL before it is sent. The URLs themselves (GitHub or GitHub
    /// Enterprise hosts, the Copilot API host from the token) are always derived as upstream
    /// derives them; tests map them to a local server here.
    pub rewrite_url: Option<UrlRewrite>,
}

impl fmt::Debug for GitHubCopilotOAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GitHubCopilotOAuth")
            .field("rewrite_url", &self.rewrite_url.is_some())
            .finish()
    }
}

/// The host of a GitHub Enterprise URL or bare domain: `company.ghe.com` and
/// `https://company.ghe.com/path` both give `company.ghe.com`. `None` for blank or invalid
/// input.
pub fn normalize_domain(input: &str) -> Option<String> {
    let trimmed = js_trim(input);
    if trimmed.is_empty() {
        return None;
    }
    let url = if trimmed.contains("://") {
        url::Url::parse(trimmed)
    } else {
        url::Url::parse(&format!("https://{trimmed}"))
    };
    url.ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .filter(|host| !host.is_empty())
}

struct GitHubUrls {
    device_code: String,
    access_token: String,
    copilot_token: String,
}

fn get_urls(domain: &str) -> GitHubUrls {
    GitHubUrls {
        device_code: format!("https://{domain}/login/device/code"),
        access_token: format!("https://{domain}/login/oauth/access_token"),
        copilot_token: format!("https://api.{domain}/copilot_internal/v2/token"),
    }
}

/// The API base URL named by a Copilot token's `proxy-ep` field
/// (`tid=...;exp=...;proxy-ep=proxy.individual.githubcopilot.com;...` gives
/// `https://api.individual.githubcopilot.com`).
fn get_base_url_from_token(token: &str) -> Option<String> {
    // The first `proxy-ep=` followed by a non-empty value (upstream `/proxy-ep=([^;]+)/`).
    let host = token.match_indices("proxy-ep=").find_map(|(start, key)| {
        let value = token[start + key.len()..].split(';').next().unwrap_or_default();
        (!value.is_empty()).then_some(value)
    })?;
    let api_host = match host.strip_prefix("proxy.") {
        Some(rest) => format!("api.{rest}"),
        None => host.to_owned(),
    };
    Some(format!("https://{api_host}"))
}

/// The Copilot API base URL: from the token's `proxy-ep`, else
/// `https://copilot-api.<enterprise domain>`, else [`INDIVIDUAL_BASE_URL`].
pub fn github_copilot_base_url(token: Option<&str>, enterprise_domain: Option<&str>) -> String {
    if let Some(url) = token
        .filter(|token| !token.is_empty())
        .and_then(get_base_url_from_token)
    {
        return url;
    }
    match enterprise_domain.filter(|domain| !domain.is_empty()) {
        Some(domain) => format!("https://copilot-api.{domain}"),
        None => INDIVIDUAL_BASE_URL.to_owned(),
    }
}

/// The normalized enterprise domain stored in a credential's `enterpriseUrl`.
fn copilot_enterprise_domain(credential: &OAuthCredential) -> Option<String> {
    match credential.extra.get("enterpriseUrl") {
        Some(Value::String(url)) if !url.is_empty() => normalize_domain(url),
        _ => None,
    }
}

/// Model ids of the account's Copilot catalog.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CopilotModelCatalog {
    /// Models the account may use.
    available_model_ids: Vec<String>,
    /// Known models whose policy is `unconfigured` and can be enabled.
    policy_model_ids: Vec<String>,
}

struct AccountModel {
    id: String,
    picker_enabled: bool,
    policy_state: Option<Value>,
}

impl AccountModel {
    fn policy_is(&self, state: &str) -> bool {
        self.policy_state.as_ref().and_then(Value::as_str) == Some(state)
    }
}

fn is_known_model(id: &str) -> bool {
    all_builtin_models("github-copilot")
        .iter()
        .any(|model| model.id() == id)
}

fn parse_github_copilot_model_catalog(
    raw: &Value,
    allow_policy_fallback: bool,
) -> Result<CopilotModelCatalog, ModelsError> {
    let Some(Value::Array(data)) = raw.get("data") else {
        return Err(oauth_error("Invalid Copilot models response"));
    };

    let account_models: Vec<AccountModel> = data
        .iter()
        .filter_map(|item| {
            let id = item.get("id")?.as_str()?;
            let supports = item
                .get("capabilities")
                .and_then(|capabilities| capabilities.get("supports"));
            if supports.and_then(|supports| supports.get("tool_calls")) == Some(&Value::Bool(false)) {
                return None;
            }
            Some(AccountModel {
                id: id.to_owned(),
                picker_enabled: item.get("model_picker_enabled") == Some(&Value::Bool(true)),
                policy_state: item.get("policy").and_then(|policy| policy.get("state")).cloned(),
            })
        })
        .collect();
    let picker_model_ids: Vec<String> = account_models
        .iter()
        .filter(|model| model.picker_enabled && !model.policy_is("disabled"))
        .map(|model| model.id.clone())
        .collect();
    let use_policy_fallback = allow_policy_fallback && picker_model_ids.is_empty();
    let available_model_ids = if !picker_model_ids.is_empty() || !allow_policy_fallback {
        picker_model_ids
    } else {
        account_models
            .iter()
            .filter(|model| model.policy_is("enabled"))
            .map(|model| model.id.clone())
            .collect()
    };
    let policy_model_ids = account_models
        .iter()
        .filter(|model| {
            model.policy_is("unconfigured")
                && is_known_model(&model.id)
                && (model.picker_enabled || use_policy_fallback)
        })
        .map(|model| model.id.clone())
        .collect();
    Ok(CopilotModelCatalog {
        available_model_ids,
        policy_model_ids,
    })
}

/// Retry budget of [`GitHubCopilotOAuth::fetch_with_rate_limit_retry`].
#[derive(Clone, Copy, Debug)]
struct RetryPolicy {
    max_retries: u32,
    max_elapsed: Duration,
}

/// Login: two retries within five seconds.
const LOGIN_RETRY_POLICY: RetryPolicy = RetryPolicy {
    max_retries: 2,
    max_elapsed: Duration::from_secs(5),
};
/// Refresh: no retries.
const REFRESH_RETRY_POLICY: RetryPolicy = RetryPolicy {
    max_retries: 0,
    max_elapsed: Duration::ZERO,
};

/// JavaScript's `parseFloat`: the longest numeric prefix after leading whitespace, `None`
/// for `NaN`.
fn js_parse_float(text: &str) -> Option<f64> {
    let text = text.trim_start_matches(is_js_whitespace);
    let bytes = text.as_bytes();
    let mut end = 0;
    if matches!(bytes.first(), Some(b'+' | b'-')) {
        end = 1;
    }
    if text[end..].starts_with("Infinity") {
        return Some(if text.starts_with('-') {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        });
    }
    let digits = |from: usize| bytes[from..].iter().take_while(|byte| byte.is_ascii_digit()).count();
    let integer = digits(end);
    end += integer;
    let mut fraction = 0;
    if bytes.get(end) == Some(&b'.') {
        fraction = digits(end + 1);
        if integer > 0 || fraction > 0 {
            end += 1 + fraction;
        }
    }
    if integer == 0 && fraction == 0 {
        return None;
    }
    if matches!(bytes.get(end), Some(b'e' | b'E')) {
        let mut exponent = end + 1;
        if matches!(bytes.get(exponent), Some(b'+' | b'-')) {
            exponent += 1;
        }
        let exponent_digits = digits(exponent);
        if exponent_digits > 0 {
            end = exponent + exponent_digits;
        }
    }
    text[..end].parse().ok()
}

/// The `Retry-After` delay in milliseconds as upstream computes it: seconds, else an HTTP
/// date minus now; `NaN` (not finite) when neither parses.
fn retry_after_delay_ms(value: &str) -> f64 {
    match js_parse_float(value) {
        Some(seconds) => seconds * 1000.0,
        None => match httpdate_to_unix(value) {
            Some(date) => date * 1000.0 - now_ms() as f64,
            None => f64::NAN,
        },
    }
}

fn status_line(status: reqwest::StatusCode) -> String {
    format!("{} {}", status.as_u16(), status.canonical_reason().unwrap_or_default())
}

/// `<status> <status text>: <body>`, upstream's error for a failed request.
async fn http_failure(response: reqwest::Response) -> ModelsError {
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    oauth_error(format!("{}: {text}", status_line(status)))
}

fn transport_error(error: &reqwest::Error) -> ModelsError {
    oauth_error(format_error_details(error))
}

async fn json_body(response: reqwest::Response) -> Result<Value, ModelsError> {
    let text = response.text().await.map_err(|error| transport_error(&error))?;
    serde_json::from_str(&text).map_err(|error| oauth_error(format!("Invalid JSON response: {error}")))
}

/// Runs `future` unless `cancel` fires first (`Login cancelled`).
async fn cancellable<T>(
    cancel: &CancellationToken,
    future: impl Future<Output = Result<T, ModelsError>>,
) -> Result<T, ModelsError> {
    if cancel.is_cancelled() {
        return Err(login_cancelled());
    }
    tokio::select! {
        () = cancel.cancelled() => Err(login_cancelled()),
        result = future => result,
    }
}

#[derive(Debug)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    interval: Option<f64>,
    expires_in: f64,
}

fn parse_device_code_response(data: &Value) -> Result<DeviceCodeResponse, ModelsError> {
    if !(data.is_object() || data.is_array()) {
        return Err(oauth_error("Invalid device code response"));
    }
    let string = |name: &str| data.get(name).and_then(Value::as_str).map(str::to_owned);
    let invalid = || oauth_error("Invalid device code response fields");
    let interval = match data.get("interval") {
        None => None,
        Some(Value::Number(number)) => number.as_f64(),
        Some(_) => return Err(invalid()),
    };
    let (Some(device_code), Some(user_code), Some(verification_uri), Some(expires_in)) = (
        string("device_code"),
        string("user_code"),
        string("verification_uri"),
        data.get("expires_in").and_then(Value::as_f64),
    ) else {
        return Err(invalid());
    };

    // The verification URI is opened in the user's browser; to keep the opener from
    // launching an executable or similar, it must be an http(s) URL.
    let untrusted = || oauth_error("Untrusted verification_uri in device code response");
    let parsed = url::Url::parse(&verification_uri).map_err(|_| untrusted())?;
    if parsed.scheme() != "https" && parsed.scheme() != "http" {
        return Err(untrusted());
    }

    Ok(DeviceCodeResponse {
        device_code,
        user_code,
        verification_uri: parsed.to_string(),
        interval,
        expires_in,
    })
}

impl GitHubCopilotOAuth {
    fn url(&self, url: &str) -> String {
        match &self.rewrite_url {
            Some(rewrite) => rewrite(url),
            None => url.to_owned(),
        }
    }

    /// A form `POST` to a GitHub login endpoint; the JSON response body.
    async fn post_form(
        &self,
        url: &str,
        fields: &[(&str, &str)],
        cancel: &CancellationToken,
    ) -> Result<Value, ModelsError> {
        let request = oauth_http_client()
            .post(self.url(url))
            .header("Accept", "application/json")
            .header("Content-Type", "application/x-www-form-urlencoded")
            .header("User-Agent", USER_AGENT)
            .body(form_urlencode(fields.iter().copied()));
        cancellable(cancel, async {
            let response = request.send().await.map_err(|error| transport_error(&error))?;
            if !response.status().is_success() {
                return Err(http_failure(response).await);
            }
            json_body(response).await
        })
        .await
    }

    async fn start_device_flow(
        &self,
        domain: &str,
        cancel: &CancellationToken,
    ) -> Result<DeviceCodeResponse, ModelsError> {
        let client_id = client_id();
        let data = self
            .post_form(
                &get_urls(domain).device_code,
                &[("client_id", &client_id), ("scope", "read:user")],
                cancel,
            )
            .await?;
        parse_device_code_response(&data)
    }

    async fn poll_for_github_access_token(
        &self,
        domain: &str,
        device: &DeviceCodeResponse,
        cancel: &CancellationToken,
    ) -> Result<String, ModelsError> {
        let url = get_urls(domain).access_token;
        let client_id = client_id();
        let options = OAuthDeviceCodePollOptions {
            interval_seconds: device.interval,
            expires_in_seconds: Some(device.expires_in),
            wait_before_first_poll: true,
            cancel: cancel.clone(),
        };
        poll_oauth_device_code_flow(options, || async {
            let raw = self
                .post_form(
                    &url,
                    &[
                        ("client_id", &client_id),
                        ("device_code", &device.device_code),
                        ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                    ],
                    cancel,
                )
                .await?;
            if let Some(token) = raw.get("access_token").and_then(Value::as_str) {
                return Ok(OAuthDeviceCodePollResult::Complete(token.to_owned()));
            }
            if let Some(error) = raw.get("error").and_then(Value::as_str) {
                return Ok(match error {
                    "authorization_pending" => OAuthDeviceCodePollResult::Pending,
                    "slow_down" => OAuthDeviceCodePollResult::SlowDown {
                        interval_seconds: raw.get("interval").and_then(Value::as_f64),
                    },
                    _ => {
                        let suffix = match raw.get("error_description").and_then(Value::as_str) {
                            Some(description) if !description.is_empty() => format!(": {description}"),
                            _ => String::new(),
                        };
                        OAuthDeviceCodePollResult::Failed {
                            message: format!("Device flow failed: {error}{suffix}"),
                        }
                    }
                });
            }
            Ok(OAuthDeviceCodePollResult::Failed {
                message: "Invalid device token response".into(),
            })
        })
        .await
    }

    /// Exchange the GitHub access token for a Copilot token.
    async fn refresh_github_copilot_access_token(
        &self,
        refresh_token: &str,
        enterprise_domain: Option<&str>,
        cancel: &CancellationToken,
    ) -> Result<OAuthCredential, ModelsError> {
        let domain = enterprise_domain.unwrap_or("github.com");
        let mut request = oauth_http_client()
            .get(self.url(&get_urls(domain).copilot_token))
            .header("Accept", "application/json")
            .header("Authorization", format!("Bearer {refresh_token}"));
        for (name, value) in COPILOT_HEADERS {
            request = request.header(name, value);
        }
        let raw = cancellable(cancel, async {
            let response = request.send().await.map_err(|error| transport_error(&error))?;
            if !response.status().is_success() {
                return Err(http_failure(response).await);
            }
            json_body(response).await
        })
        .await?;

        if !(raw.is_object() || raw.is_array()) {
            return Err(oauth_error("Invalid Copilot token response"));
        }
        let (Some(token), Some(expires_at)) = (
            raw.get("token").and_then(Value::as_str),
            raw.get("expires_at").and_then(Value::as_f64),
        ) else {
            return Err(oauth_error("Invalid Copilot token response fields"));
        };

        let mut extra = JsonObject::new();
        if let Some(domain) = enterprise_domain {
            extra.insert("enterpriseUrl".into(), Value::String(domain.to_owned()));
        }
        Ok(OAuthCredential {
            refresh: refresh_token.to_owned(),
            access: token.to_owned(),
            expires: (expires_at * 1000.0) as i64 - REFRESH_SKEW_MS,
            extra,
        })
    }

    /// Send a catalog or policy request, retrying `429` responses after `Retry-After` (or
    /// 500 ms, doubling) within the policy's retry count and time budget. Each attempt times
    /// out after five seconds. Returns the last response; a `429` comes back when the
    /// retries are used up, the delay is unreadable, or the delay would pass the budget.
    async fn fetch_with_rate_limit_retry(
        &self,
        request: impl Fn() -> reqwest::RequestBuilder,
        cancel: &CancellationToken,
        policy: RetryPolicy,
    ) -> Result<reqwest::Response, ModelsError> {
        let deadline =
            (policy.max_retries > 0 && !policy.max_elapsed.is_zero()).then(|| Instant::now() + policy.max_elapsed);
        let budget_expired = || oauth_error("The operation was aborted due to timeout");
        for retry in 0u32.. {
            let timeout = match deadline {
                Some(deadline) => REQUEST_TIMEOUT.min(deadline.saturating_duration_since(Instant::now())),
                None => REQUEST_TIMEOUT,
            };
            if timeout.is_zero() {
                return Err(budget_expired());
            }
            let response = cancellable(cancel, async {
                request()
                    .timeout(timeout)
                    .send()
                    .await
                    .map_err(|error| transport_error(&error))
            })
            .await?;
            if response.status() != reqwest::StatusCode::TOO_MANY_REQUESTS || retry == policy.max_retries {
                return Ok(response);
            }

            let mut delay_ms = 500.0 * 2f64.powi(retry as i32);
            if let Some(value) = response
                .headers()
                .get("retry-after")
                .and_then(|value| value.to_str().ok())
                .filter(|value| !value.is_empty())
            {
                delay_ms = retry_after_delay_ms(value);
                if !delay_ms.is_finite() {
                    return Ok(response);
                }
            }
            let delay = Duration::from_secs_f64(delay_ms.max(0.0) / 1000.0);
            if let Some(deadline) = deadline
                && delay >= deadline.saturating_duration_since(Instant::now())
            {
                return Ok(response);
            }
            drop(response);
            abortable_sleep(delay, cancel, CANCEL_MESSAGE).await?;
        }
        unreachable!("the retry loop returns")
    }

    async fn fetch_github_copilot_models(
        &self,
        copilot_token: &str,
        enterprise_domain: Option<&str>,
        cancel: &CancellationToken,
        policy: RetryPolicy,
    ) -> Result<CopilotModelCatalog, ModelsError> {
        let base_url = github_copilot_base_url(Some(copilot_token), enterprise_domain);
        // Some Individual accounts return false for every picker flag despite explicit enabled
        // policies. Limit the fallback to that endpoint so other account types keep strict
        // picker semantics.
        let allow_policy_fallback = base_url == INDIVIDUAL_BASE_URL;
        let url = self.url(&format!("{base_url}/models"));
        let response = self
            .fetch_with_rate_limit_retry(
                || {
                    let mut request = oauth_http_client()
                        .get(&url)
                        .header("Accept", "application/json")
                        .header("Authorization", format!("Bearer {copilot_token}"));
                    for (name, value) in COPILOT_HEADERS {
                        request = request.header(name, value);
                    }
                    request.header("X-GitHub-Api-Version", COPILOT_API_VERSION)
                },
                cancel,
                policy,
            )
            .await?;
        if !response.status().is_success() {
            return Err(http_failure(response).await);
        }
        let raw = cancellable(cancel, json_body(response)).await?;
        parse_github_copilot_model_catalog(&raw, allow_policy_fallback)
    }

    /// Enable a model for the account (required for some models, such as Claude and Grok,
    /// before they can be used). A failed request gives `Ok(false)`; a `429` that outlasts
    /// the retries is an error.
    async fn enable_github_copilot_model(
        &self,
        token: &str,
        model_id: &str,
        enterprise_domain: Option<&str>,
        cancel: &CancellationToken,
    ) -> Result<bool, ModelsError> {
        let base_url = github_copilot_base_url(Some(token), enterprise_domain);
        let url = self.url(&format!("{base_url}/models/{model_id}/policy"));
        let sent = self
            .fetch_with_rate_limit_retry(
                || {
                    let mut request = oauth_http_client()
                        .post(&url)
                        .header("Content-Type", "application/json")
                        .header("Authorization", format!("Bearer {token}"));
                    for (name, value) in COPILOT_HEADERS {
                        request = request.header(name, value);
                    }
                    request
                        .header("openai-intent", "chat-policy")
                        .header("x-interaction-type", "chat-policy")
                        .body(json!({"state": "enabled"}).to_string())
                },
                cancel,
                LOGIN_RETRY_POLICY,
            )
            .await;
        let response = match sent {
            Ok(response) => response,
            Err(error) if cancel.is_cancelled() => return Err(error),
            Err(_) => return Ok(false),
        };
        if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(http_failure(response).await);
        }
        Ok(response.status().is_success())
    }

    /// Enable the requested models and return the ids that succeeded. Policy updates are
    /// best effort; rate limiting that outlasts the retries stops the batch.
    async fn enable_github_copilot_models(
        &self,
        token: &str,
        model_ids: &[String],
        enterprise_domain: Option<&str>,
        cancel: &CancellationToken,
    ) -> Result<Vec<String>, ModelsError> {
        let mut enabled = Vec::new();
        for model_id in model_ids {
            match self
                .enable_github_copilot_model(token, model_id, enterprise_domain, cancel)
                .await
            {
                Ok(true) => enabled.push(model_id.clone()),
                Ok(false) => {}
                Err(error) if cancel.is_cancelled() => return Err(error),
                Err(_) => break,
            }
        }
        Ok(enabled)
    }

    async fn login_github_copilot(&self, interaction: &dyn AuthInteraction) -> Result<OAuthCredential, ModelsError> {
        let cancel = interaction.cancel();
        let input = interaction
            .prompt(AuthPrompt::Text {
                message: "GitHub Enterprise URL/domain (blank for github.com)".into(),
                placeholder: Some("company.ghe.com".into()),
            })
            .await?;
        if cancel.is_cancelled() {
            return Err(login_cancelled());
        }

        let enterprise_domain = normalize_domain(&input);
        if !js_trim(&input).is_empty() && enterprise_domain.is_none() {
            return Err(oauth_error("Invalid GitHub Enterprise URL/domain"));
        }
        let enterprise_domain = enterprise_domain.as_deref();
        let domain = enterprise_domain.unwrap_or("github.com");

        let device = self.start_device_flow(domain, &cancel).await?;
        interaction.notify(AuthEvent::DeviceCode {
            user_code: device.user_code.clone(),
            verification_uri: device.verification_uri.clone(),
            interval_seconds: device.interval.map(|seconds| seconds as u64),
            expires_in_seconds: Some(device.expires_in as u64),
        });

        let github_access_token = self.poll_for_github_access_token(domain, &device, &cancel).await?;
        let mut credential = self
            .refresh_github_copilot_access_token(&github_access_token, enterprise_domain, &cancel)
            .await?;
        let models = self
            .fetch_github_copilot_models(&credential.access, enterprise_domain, &cancel, LOGIN_RETRY_POLICY)
            .await?;
        let mut enabled_model_ids = Vec::new();
        if !models.policy_model_ids.is_empty() {
            interaction.notify(AuthEvent::Progress {
                message: "Enabling models...".into(),
            });
            enabled_model_ids = self
                .enable_github_copilot_models(&credential.access, &models.policy_model_ids, enterprise_domain, &cancel)
                .await?;
        }

        let mut seen = HashSet::new();
        let available: Vec<Value> = models
            .available_model_ids
            .into_iter()
            .chain(enabled_model_ids)
            .filter(|id| seen.insert(id.clone()))
            .map(Value::String)
            .collect();
        credential
            .extra
            .insert("availableModelIds".into(), Value::Array(available));
        Ok(credential)
    }

    async fn refresh_github_copilot_token(&self, credential: &OAuthCredential) -> Result<OAuthCredential, ModelsError> {
        let enterprise_domain = copilot_enterprise_domain(credential);
        let cancel = CancellationToken::new();
        let mut refreshed = self
            .refresh_github_copilot_access_token(&credential.refresh, enterprise_domain.as_deref(), &cancel)
            .await?;
        let models = self
            .fetch_github_copilot_models(
                &refreshed.access,
                enterprise_domain.as_deref(),
                &cancel,
                REFRESH_RETRY_POLICY,
            )
            .await?;
        let available = models.available_model_ids.into_iter().map(Value::String).collect();
        refreshed
            .extra
            .insert("availableModelIds".into(), Value::Array(available));
        Ok(refreshed)
    }
}

impl OAuthAuth for GitHubCopilotOAuth {
    fn name(&self) -> &'static str {
        "GitHub Copilot"
    }

    fn is_subscription(&self) -> bool {
        true
    }

    fn login<'a>(
        &'a self,
        interaction: &'a dyn AuthInteraction,
        _options: &'a LoginOptions,
    ) -> BoxFuture<'a, Result<OAuthCredential, ModelsError>> {
        Box::pin(self.login_github_copilot(interaction))
    }

    fn refresh(&self, credential: OAuthCredential) -> BoxFuture<'_, Result<OAuthCredential, ModelsError>> {
        Box::pin(async move { self.refresh_github_copilot_token(&credential).await })
    }

    /// The Copilot token as the API key and the credential's own API base URL.
    fn to_auth<'a>(&'a self, credential: &'a OAuthCredential) -> BoxFuture<'a, Result<ModelAuth, ModelsError>> {
        Box::pin(async move {
            let enterprise_domain = copilot_enterprise_domain(credential);
            Ok(ModelAuth {
                api_key: Some(credential.access.clone()),
                base_url: Some(github_copilot_base_url(
                    Some(&credential.access),
                    enterprise_domain.as_deref(),
                )),
                ..ModelAuth::default()
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_id_decodes() {
        assert_eq!(client_id(), "Iv1.b507a08c87ecfe98");
    }

    #[test]
    fn normalizes_enterprise_domains() {
        assert_eq!(normalize_domain(""), None);
        assert_eq!(normalize_domain("   "), None);
        assert_eq!(normalize_domain("company.ghe.com").as_deref(), Some("company.ghe.com"));
        assert_eq!(
            normalize_domain(" Company.GHE.com ").as_deref(),
            Some("company.ghe.com")
        );
        assert_eq!(
            normalize_domain("https://company.ghe.com/some/path").as_deref(),
            Some("company.ghe.com")
        );
        assert_eq!(normalize_domain("not a domain"), None);
    }

    #[test]
    fn derives_the_base_url_from_the_token_then_the_enterprise_domain() {
        assert_eq!(
            github_copilot_base_url(Some("tid=1;exp=2;proxy-ep=proxy.business.githubcopilot.com;x"), None),
            "https://api.business.githubcopilot.com"
        );
        assert_eq!(
            github_copilot_base_url(Some("proxy-ep=copilot.example"), Some("company.ghe.com")),
            "https://copilot.example"
        );
        assert_eq!(
            github_copilot_base_url(Some("no-proxy-ep"), Some("company.ghe.com")),
            "https://copilot-api.company.ghe.com"
        );
        assert_eq!(github_copilot_base_url(None, None), INDIVIDUAL_BASE_URL);
    }

    #[test]
    fn parse_float_reads_the_numeric_prefix() {
        assert_eq!(js_parse_float("1"), Some(1.0));
        assert_eq!(js_parse_float(" 1.5s"), Some(1.5));
        assert_eq!(js_parse_float(".5"), Some(0.5));
        assert_eq!(js_parse_float("2e3"), Some(2000.0));
        assert_eq!(js_parse_float("2e"), Some(2.0));
        assert_eq!(js_parse_float("-Infinity"), Some(f64::NEG_INFINITY));
        assert_eq!(js_parse_float("Wed, 21 Oct 2015 07:28:00 GMT"), None);
        assert_eq!(js_parse_float("."), None);
    }

    #[test]
    fn retry_after_accepts_http_dates() {
        let delay = retry_after_delay_ms("Wed, 21 Oct 2015 07:28:00 GMT");
        assert!(delay.is_finite() && delay < 0.0);
        assert!(retry_after_delay_ms("soon").is_nan());
    }
}
