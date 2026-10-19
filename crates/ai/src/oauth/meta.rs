//! Meta Model API OAuth flow. Port of upstream `auth/oauth/meta.ts`.
//!
//! RFC 8628 device authorization grant against `https://auth.meta.com` (JSON responses).
//! Meta splits identity from API access: the identity token is not accepted for
//! inference, so it is exchanged for a Model API key at the Muse Code key-mint endpoint
//! (minted keys live about a day). The identity token is stored as `refresh` and the
//! minted key as `access`, so the standard OAuth refresh re-mints the key when it expires.
//! The identity token itself is not renewable (auth.meta.com answers
//! `grant_type=refresh_token` with 404 and issues no refresh token), so a 401/403 from the
//! mint endpoint means the session is dead and the user must sign in again.

use futures::future::BoxFuture;
use serde_json::Value;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::auth::{AuthEvent, AuthInteraction, LoginOptions, ModelAuth, OAuthAuth, OAuthCredential};
use crate::error::ModelsError;
use crate::oauth::{
    OAuthDeviceCodePollOptions, OAuthDeviceCodePollResult, form_urlencode, login_cancelled, oauth_error,
    oauth_http_client, poll_oauth_device_code_flow,
};
use crate::types::{JsonObject, now_ms};

/// Muse Code CLI client id.
pub const CLIENT_ID: &str = "1031625952748946";
pub const DEVICE_AUTHORIZATION_URL: &str = "https://auth.meta.com/oidc/device/authorization/";
pub const DEVICE_TOKEN_URL: &str = "https://auth.meta.com/oidc/device/token/";
pub const API_KEY_MINT_URL: &str = "https://api.meta.ai/muse-code/key";
/// Lifetime of a minted Model API key.
pub const API_KEY_LIFETIME_MS: i64 = 24 * 60 * 60 * 1000;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Meta (Muse subscription) OAuth. `Default` uses upstream's endpoints.
#[derive(Clone, Debug)]
pub struct MetaOAuth {
    pub device_authorization_url: String,
    pub device_token_url: String,
    pub api_key_mint_url: String,
}

impl Default for MetaOAuth {
    fn default() -> Self {
        Self {
            device_authorization_url: DEVICE_AUTHORIZATION_URL.to_owned(),
            device_token_url: DEVICE_TOKEN_URL.to_owned(),
            api_key_mint_url: API_KEY_MINT_URL.to_owned(),
        }
    }
}

#[derive(Debug)]
struct DeviceAuthorization {
    device_code: String,
    user_code: String,
    verification_uri: String,
    interval_seconds: Option<f64>,
    expires_in_seconds: Option<f64>,
}

/// Status and parsed body of a response (upstream `readJson`: an object or array, else
/// `None`).
struct JsonResponse {
    status: u16,
    json: Option<Value>,
}

impl JsonResponse {
    fn ok(&self) -> bool {
        (200..300).contains(&self.status)
    }

    fn field(&self, name: &str) -> Option<&Value> {
        self.json.as_ref().and_then(|json| json.get(name))
    }

    fn str_field(&self, name: &str) -> Option<&str> {
        self.field(name).and_then(Value::as_str)
    }

    /// `JSON.stringify(json)`.
    fn stringify(&self) -> String {
        self.json.as_ref().map_or_else(|| "null".to_owned(), Value::to_string)
    }

    /// `: <detail>` from the first non-blank `error_description`, `detail`, `message` or
    /// `error` field, else empty (upstream `errorDetail`).
    fn error_detail(&self) -> String {
        ["error_description", "detail", "message", "error"]
            .iter()
            .find_map(|key| self.str_field(key).map(str::trim).filter(|value| !value.is_empty()))
            .map(|value| format!(": {value}"))
            .unwrap_or_default()
    }
}

/// Send `request` with a 30 s timeout. A cancelled `cancel` fails with `Login cancelled`.
async fn send(
    request: reqwest::RequestBuilder,
    cancel: Option<&CancellationToken>,
) -> Result<JsonResponse, ModelsError> {
    let request = async {
        let response = request
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await
            .map_err(|error| oauth_error(error.to_string()))?;
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        let json = serde_json::from_str::<Value>(&text)
            .ok()
            .filter(|json| json.is_object() || json.is_array());
        Ok(JsonResponse { status, json })
    };
    let Some(cancel) = cancel else {
        return request.await;
    };
    if cancel.is_cancelled() {
        return Err(login_cancelled());
    }
    tokio::select! {
        () = cancel.cancelled() => Err(login_cancelled()),
        result = request => result,
    }
}

async fn post_form(
    url: &str,
    fields: &[(&str, &str)],
    cancel: &CancellationToken,
) -> Result<JsonResponse, ModelsError> {
    let request = oauth_http_client()
        .post(url)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .body(form_urlencode(fields.iter().copied()));
    send(request, Some(cancel)).await
}

/// The verification URI is opened in the user's browser; only http(s) URLs are trusted.
fn trusted_http_url(value: Option<&str>) -> Option<String> {
    let value = value.filter(|value| !value.is_empty())?;
    match url::Url::parse(value) {
        Ok(url) if matches!(url.scheme(), "https" | "http") => Some(url.to_string()),
        _ => None,
    }
}

fn positive_number(value: Option<&Value>) -> Option<f64> {
    value
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value > 0.0)
}

impl MetaOAuth {
    async fn start_device_authorization(&self, cancel: &CancellationToken) -> Result<DeviceAuthorization, ModelsError> {
        let response = post_form(&self.device_authorization_url, &[("client_id", CLIENT_ID)], cancel).await?;
        if !response.ok() {
            return Err(oauth_error(format!(
                "Meta device authorization failed with status {}{}",
                response.status,
                response.error_detail()
            )));
        }
        let non_empty = |name: &str| response.str_field(name).filter(|value| !value.is_empty());
        let verification_uri = trusted_http_url(response.str_field("verification_uri_complete"))
            .or_else(|| trusted_http_url(response.str_field("verification_uri")));
        let (Some(device_code), Some(user_code), Some(verification_uri)) =
            (non_empty("device_code"), non_empty("user_code"), verification_uri)
        else {
            return Err(oauth_error(format!(
                "Invalid Meta device authorization response: {}",
                response.stringify()
            )));
        };
        Ok(DeviceAuthorization {
            device_code: device_code.to_owned(),
            user_code: user_code.to_owned(),
            verification_uri,
            interval_seconds: positive_number(response.field("interval")),
            expires_in_seconds: positive_number(response.field("expires_in")),
        })
    }

    async fn poll_for_identity_token(
        &self,
        device: &DeviceAuthorization,
        cancel: &CancellationToken,
    ) -> Result<String, ModelsError> {
        let options = OAuthDeviceCodePollOptions {
            interval_seconds: device.interval_seconds,
            expires_in_seconds: device.expires_in_seconds,
            wait_before_first_poll: true,
            cancel: cancel.clone(),
        };
        poll_oauth_device_code_flow(options, || async {
            let response = post_form(
                &self.device_token_url,
                &[
                    ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                    ("device_code", &device.device_code),
                    ("client_id", CLIENT_ID),
                ],
                cancel,
            )
            .await?;
            if response.ok()
                && let Some(token) = response.str_field("access_token").filter(|token| !token.is_empty())
            {
                return Ok(OAuthDeviceCodePollResult::Complete(token.to_owned()));
            }
            Ok(match response.str_field("error") {
                Some("authorization_pending") => OAuthDeviceCodePollResult::Pending,
                Some("slow_down") => OAuthDeviceCodePollResult::SlowDown {
                    interval_seconds: positive_number(response.field("interval")),
                },
                Some("access_denied") => OAuthDeviceCodePollResult::Failed {
                    message: "Meta login was denied.".into(),
                },
                Some("expired_token") => OAuthDeviceCodePollResult::Failed {
                    message: "Meta device authorization expired. Please restart login.".into(),
                },
                _ => OAuthDeviceCodePollResult::Failed {
                    message: format!(
                        "Meta device token request failed with status {}{}",
                        response.status,
                        response.error_detail()
                    ),
                },
            })
        })
        .await
    }

    /// Exchange an identity token for a Model API key. Keys are valid for about a day.
    async fn mint_api_key(
        &self,
        identity_token: &str,
        cancel: Option<&CancellationToken>,
    ) -> Result<OAuthCredential, ModelsError> {
        let request = oauth_http_client()
            .post(&self.api_key_mint_url)
            .header("Accept", "application/json")
            .header("Authorization", format!("Bearer {identity_token}"))
            .header("Content-Type", "application/json")
            .header("x-api-version", "1.0.0")
            .body("{}");
        let response = send(request, cancel).await?;
        if response.status == 401 || response.status == 403 {
            // The identity token is not renewable (see the module docs); only a fresh device
            // flow helps.
            return Err(oauth_error(format!(
                "Meta session expired (status {}). Run `/login meta` to sign in again.{}",
                response.status,
                response.error_detail()
            )));
        }
        if !response.ok() {
            return Err(oauth_error(format!(
                "Meta API key mint failed with status {}{}",
                response.status,
                response.error_detail()
            )));
        }
        let Some(api_key) = response.str_field("api_key").filter(|key| !key.is_empty()) else {
            let setup = trusted_http_url(response.str_field("action_url"))
                .map(|url| format!(" Complete setup at {url}"))
                .unwrap_or_default();
            return Err(oauth_error(format!("Meta did not issue an API key.{setup}")));
        };
        Ok(OAuthCredential {
            refresh: identity_token.to_owned(),
            access: api_key.to_owned(),
            expires: now_ms() + API_KEY_LIFETIME_MS,
            extra: JsonObject::new(),
        })
    }

    async fn login_meta(&self, interaction: &dyn AuthInteraction) -> Result<OAuthCredential, ModelsError> {
        let cancel = interaction.cancel();
        let result = async {
            let device = self.start_device_authorization(&cancel).await?;
            interaction.notify(AuthEvent::DeviceCode {
                user_code: device.user_code.clone(),
                verification_uri: device.verification_uri.clone(),
                interval_seconds: device.interval_seconds.map(|seconds| seconds as u64),
                expires_in_seconds: device.expires_in_seconds.map(|seconds| seconds as u64),
            });
            let identity_token = self.poll_for_identity_token(&device, &cancel).await?;
            interaction.notify(AuthEvent::Progress {
                message: "Enabling Meta Model API access...".into(),
            });
            self.mint_api_key(&identity_token, Some(&cancel)).await
        }
        .await;
        match result {
            // Any failure after cancellation reports as a cancelled login.
            Err(_) if cancel.is_cancelled() => Err(login_cancelled()),
            result => result,
        }
    }
}

impl OAuthAuth for MetaOAuth {
    fn name(&self) -> &'static str {
        "Meta (Muse subscription)"
    }

    fn is_subscription(&self) -> bool {
        true
    }

    fn login_label(&self) -> Option<&str> {
        Some("Sign in with Meta")
    }

    fn login<'a>(
        &'a self,
        interaction: &'a dyn AuthInteraction,
        _options: &'a LoginOptions,
    ) -> BoxFuture<'a, Result<OAuthCredential, ModelsError>> {
        Box::pin(self.login_meta(interaction))
    }

    /// Re-mint the API key from the stored identity token.
    fn refresh(&self, credential: OAuthCredential) -> BoxFuture<'_, Result<OAuthCredential, ModelsError>> {
        Box::pin(async move { self.mint_api_key(&credential.refresh, None).await })
    }

    fn to_auth<'a>(&'a self, credential: &'a OAuthCredential) -> BoxFuture<'a, Result<ModelAuth, ModelsError>> {
        Box::pin(async move {
            Ok(ModelAuth {
                api_key: Some(credential.access.clone()),
                ..ModelAuth::default()
            })
        })
    }
}
