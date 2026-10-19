//! Kimi Code (subscription) OAuth flow. Port of upstream `auth/oauth/kimi-coding.ts`.
//!
//! RFC 8628 device authorization grant against `https://auth.kimi.com` with JSON
//! responses. The access token authenticates requests to `https://api.kimi.com/coding` as
//! an `Authorization: Bearer` header.

use futures::future::BoxFuture;
use serde_json::Value;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::auth::{AuthEvent, AuthInteraction, LoginOptions, ModelAuth, OAuthAuth, OAuthCredential, env_value};
use crate::error::ModelsError;
use crate::http::headers_of;
use crate::oauth::{
    OAuthDeviceCodePollOptions, OAuthDeviceCodePollResult, form_urlencode, login_cancelled, oauth_error,
    oauth_http_client, poll_oauth_device_code_flow,
};
use crate::types::{JsonObject, now_ms};

pub const CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516cb48c098";
pub const DEFAULT_OAUTH_HOST: &str = "https://auth.kimi.com";
/// Environment variables that override the OAuth host, in order.
pub const OAUTH_HOST_ENV: [&str; 2] = ["KIMI_CODE_OAUTH_HOST", "KIMI_OAUTH_HOST"];
const DEVICE_CODE_TIMEOUT_SECONDS: f64 = 15.0 * 60.0;
const DEFAULT_POLL_INTERVAL_SECONDS: f64 = 5.0;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const REFRESH_MAX_RETRIES: u32 = 3;

/// Kimi Code (subscription) OAuth. `Default` resolves the OAuth host from the environment
/// at each login and refresh, as upstream does.
#[derive(Clone, Debug, Default)]
pub struct KimiCodingOAuth {
    /// OAuth host. `None`: `KIMI_CODE_OAUTH_HOST`, else `KIMI_OAUTH_HOST`, else
    /// [`DEFAULT_OAUTH_HOST`]. Trailing slashes are removed.
    pub oauth_host: Option<String>,
}

/// The OAuth host from the environment (upstream `getOauthHost`).
pub fn kimi_oauth_host() -> String {
    oauth_host_from(|name| env_value(None, name))
}

fn oauth_host_from(lookup: impl Fn(&str) -> Option<String>) -> String {
    let host = OAUTH_HOST_ENV
        .iter()
        .find_map(|name| lookup(name))
        .unwrap_or_else(|| DEFAULT_OAUTH_HOST.to_owned());
    host.trim_end_matches('/').to_owned()
}

#[derive(Debug)]
struct DeviceAuthorization {
    device_code: String,
    user_code: String,
    verification_uri_complete: String,
    interval_seconds: f64,
    expires_in_seconds: f64,
}

/// Status, text and parsed body of a response (upstream `readJson`: an object or array,
/// else `None`).
struct JsonResponse {
    status: u16,
    text: String,
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
}

fn read_json(text: &str) -> Option<Value> {
    serde_json::from_str::<Value>(text)
        .ok()
        .filter(|json| json.is_object() || json.is_array())
}

/// Form `POST` with a 30 s timeout. A cancelled `cancel` fails with `Login cancelled`.
async fn post_form(
    url: &str,
    fields: &[(&str, &str)],
    cancel: Option<&CancellationToken>,
) -> Result<JsonResponse, ModelsError> {
    let request = async {
        let response = oauth_http_client()
            .post(url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .header("Accept", "application/json")
            .timeout(REQUEST_TIMEOUT)
            .body(form_urlencode(fields.iter().copied()))
            .send()
            .await
            .map_err(|error| oauth_error(error.to_string()))?;
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        let json = read_json(&text);
        Ok(JsonResponse { status, text, json })
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

/// The verification URI is opened in the user's browser; only http(s) URLs are trusted.
fn trusted_http_url(value: Option<&str>) -> Option<String> {
    let value = value.filter(|value| !value.is_empty())?;
    match url::Url::parse(value) {
        Ok(url) if matches!(url.scheme(), "https" | "http") => Some(url.to_string()),
        _ => None,
    }
}

fn positive(value: Option<&Value>) -> Option<f64> {
    value
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn with_text(text: &str) -> String {
    if text.is_empty() {
        String::new()
    } else {
        format!(": {text}")
    }
}

fn parse_token_response(response: &JsonResponse, operation: &str) -> Result<OAuthCredential, String> {
    let non_empty = |name: &str| response.str_field(name).filter(|value| !value.is_empty());
    match (
        non_empty("access_token"),
        non_empty("refresh_token"),
        positive(response.field("expires_in")),
    ) {
        (Some(access), Some(refresh), Some(expires_in)) => Ok(OAuthCredential {
            refresh: refresh.to_owned(),
            access: access.to_owned(),
            expires: now_ms() + (expires_in * 1000.0) as i64,
            extra: JsonObject::new(),
        }),
        _ => Err(format!(
            "Kimi Code token {operation} response missing fields: {}",
            response.stringify()
        )),
    }
}

impl KimiCodingOAuth {
    fn host(&self) -> String {
        match &self.oauth_host {
            Some(host) => host.trim_end_matches('/').to_owned(),
            None => kimi_oauth_host(),
        }
    }

    async fn start_device_authorization(
        &self,
        host: &str,
        cancel: &CancellationToken,
    ) -> Result<DeviceAuthorization, ModelsError> {
        let response = post_form(
            &format!("{host}/api/oauth/device_authorization"),
            &[("client_id", CLIENT_ID)],
            Some(cancel),
        )
        .await?;
        if !response.ok() {
            return Err(oauth_error(format!(
                "Kimi Code device authorization failed with status {}{}",
                response.status,
                with_text(&response.text)
            )));
        }

        let device_code = response.str_field("device_code");
        let user_code = response.str_field("user_code");
        let verification_uri = response.str_field("verification_uri");
        let verification_uri_complete = response.str_field("verification_uri_complete");
        let (Some(device_code), Some(user_code), Some(_), Some(verification_uri_complete)) =
            (device_code, user_code, verification_uri, verification_uri_complete)
        else {
            return Err(invalid_device_authorization(&response));
        };
        if trusted_http_url(Some(verification_uri_complete)).is_none() || trusted_http_url(verification_uri).is_none() {
            return Err(invalid_device_authorization(&response));
        }

        Ok(DeviceAuthorization {
            device_code: device_code.to_owned(),
            user_code: user_code.to_owned(),
            verification_uri_complete: verification_uri_complete.to_owned(),
            interval_seconds: positive(response.field("interval")).unwrap_or(DEFAULT_POLL_INTERVAL_SECONDS),
            expires_in_seconds: positive(response.field("expires_in")).unwrap_or(DEVICE_CODE_TIMEOUT_SECONDS),
        })
    }

    async fn poll_for_token(
        &self,
        host: &str,
        device: &DeviceAuthorization,
        cancel: &CancellationToken,
    ) -> Result<OAuthCredential, ModelsError> {
        let token_url = format!("{host}/api/oauth/token");
        let options = OAuthDeviceCodePollOptions {
            interval_seconds: Some(device.interval_seconds),
            expires_in_seconds: Some(device.expires_in_seconds),
            wait_before_first_poll: true,
            cancel: cancel.clone(),
        };
        poll_oauth_device_code_flow(options, || async {
            let response = post_form(
                &token_url,
                &[
                    ("client_id", CLIENT_ID),
                    ("device_code", &device.device_code),
                    ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ],
                Some(cancel),
            )
            .await?;

            if response.status >= 500 {
                return Ok(OAuthDeviceCodePollResult::Failed {
                    message: format!(
                        "Kimi Code device token request failed with status {}{}",
                        response.status,
                        with_text(&response.text)
                    ),
                });
            }

            if response.ok() && response.field("access_token").is_some_and(Value::is_string) {
                return Ok(match parse_token_response(&response, "poll") {
                    Ok(credential) => OAuthDeviceCodePollResult::Complete(credential),
                    Err(message) => OAuthDeviceCodePollResult::Failed { message },
                });
            }

            let error = response.str_field("error");
            Ok(match error {
                Some("authorization_pending") => OAuthDeviceCodePollResult::Pending,
                Some("slow_down") => OAuthDeviceCodePollResult::SlowDown {
                    interval_seconds: response
                        .field("interval")
                        .and_then(Value::as_f64)
                        .filter(|interval| *interval > 0.0),
                },
                Some("expired_token") => OAuthDeviceCodePollResult::Failed {
                    message: "Kimi Code device authorization expired. Please restart login.".into(),
                },
                Some("access_denied") => OAuthDeviceCodePollResult::Failed {
                    message: "Kimi Code login was denied.".into(),
                },
                _ => {
                    let detail = match error {
                        Some(error) => {
                            let description = response
                                .str_field("error_description")
                                .map(|description| format!(": {description}"))
                                .unwrap_or_default();
                            format!(": {error}{description}")
                        }
                        None => String::new(),
                    };
                    OAuthDeviceCodePollResult::Failed {
                        message: format!(
                            "Kimi Code device token request failed (status {}){detail}",
                            response.status
                        ),
                    }
                }
            })
        })
        .await
    }

    async fn login_kimi_coding(&self, interaction: &dyn AuthInteraction) -> Result<OAuthCredential, ModelsError> {
        let cancel = interaction.cancel();
        let host = self.host();
        let device = self.start_device_authorization(&host, &cancel).await?;
        interaction.notify(AuthEvent::DeviceCode {
            user_code: device.user_code.clone(),
            verification_uri: device.verification_uri_complete.clone(),
            interval_seconds: Some(device.interval_seconds as u64),
            expires_in_seconds: Some(device.expires_in_seconds as u64),
        });
        self.poll_for_token(&host, &device, &cancel).await
    }

    /// Refresh with up to three retries on network errors, 429 and 5xx (1 s, 2 s, 4 s
    /// backoff). 401, 403 and `invalid_grant` fail at once as unauthorized.
    async fn refresh_token(&self, refresh_token: &str) -> Result<OAuthCredential, ModelsError> {
        let token_url = format!("{}/api/oauth/token", self.host());
        let mut last_error = None;
        for attempt in 0..=REFRESH_MAX_RETRIES {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(1000 * 2u64.pow(attempt - 1))).await;
            }

            let response = match post_form(
                &token_url,
                &[
                    ("client_id", CLIENT_ID),
                    ("grant_type", "refresh_token"),
                    ("refresh_token", refresh_token),
                ],
                None,
            )
            .await
            {
                Ok(response) => response,
                Err(error) => {
                    last_error = Some(error);
                    continue;
                }
            };

            if response.ok() {
                return parse_token_response(&response, "refresh").map_err(oauth_error);
            }

            // Unauthorized: the stored credential is dead; the user has to log in again.
            if response.status == 401 || response.status == 403 || response.str_field("error") == Some("invalid_grant")
            {
                let description = response
                    .str_field("error_description")
                    .map(|description| format!(": {description}"))
                    .unwrap_or_default();
                return Err(oauth_error(format!(
                    "Kimi Code token refresh unauthorized (status {}){description}",
                    response.status
                )));
            }

            if (response.status == 429 || response.status >= 500) && attempt < REFRESH_MAX_RETRIES {
                last_error = Some(oauth_error(format!(
                    "Kimi Code token refresh failed with status {}",
                    response.status
                )));
                continue;
            }

            return Err(oauth_error(format!(
                "Kimi Code token refresh failed with status {}: {}",
                response.status,
                response.stringify()
            )));
        }

        Err(last_error.unwrap_or_else(|| oauth_error("Kimi Code token refresh failed")))
    }
}

fn invalid_device_authorization(response: &JsonResponse) -> ModelsError {
    oauth_error(format!(
        "Invalid Kimi Code device authorization response: {}",
        response.stringify()
    ))
}

impl OAuthAuth for KimiCodingOAuth {
    fn name(&self) -> &'static str {
        "Kimi Code (subscription)"
    }

    fn is_subscription(&self) -> bool {
        true
    }

    fn login_label(&self) -> Option<&str> {
        Some("Sign in with Kimi Code")
    }

    fn login<'a>(
        &'a self,
        interaction: &'a dyn AuthInteraction,
        _options: &'a LoginOptions,
    ) -> BoxFuture<'a, Result<OAuthCredential, ModelsError>> {
        Box::pin(self.login_kimi_coding(interaction))
    }

    fn refresh(&self, credential: OAuthCredential) -> BoxFuture<'_, Result<OAuthCredential, ModelsError>> {
        Box::pin(async move { self.refresh_token(&credential.refresh).await })
    }

    /// The access token as an `Authorization: Bearer` header; no API key.
    fn to_auth<'a>(&'a self, credential: &'a OAuthCredential) -> BoxFuture<'a, Result<ModelAuth, ModelsError>> {
        Box::pin(async move {
            Ok(ModelAuth {
                headers: Some(headers_of([(
                    "Authorization",
                    &format!("Bearer {}", credential.access),
                )])),
                ..ModelAuth::default()
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oauth_host_prefers_kimi_code_oauth_host_and_strips_trailing_slashes() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                pairs
                    .iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| (*value).to_owned())
            }
        };
        assert_eq!(oauth_host_from(env(&[])), "https://auth.kimi.com");
        assert_eq!(
            oauth_host_from(env(&[("KIMI_CODE_OAUTH_HOST", "https://auth.example.com/")])),
            "https://auth.example.com"
        );
        assert_eq!(
            oauth_host_from(env(&[("KIMI_OAUTH_HOST", "https://legacy.example.com//")])),
            "https://legacy.example.com"
        );
        assert_eq!(
            oauth_host_from(env(&[
                ("KIMI_OAUTH_HOST", "https://legacy.example.com"),
                ("KIMI_CODE_OAUTH_HOST", "https://auth.example.com"),
            ])),
            "https://auth.example.com"
        );
    }

    #[test]
    fn trusts_only_http_and_https_verification_uris() {
        assert_eq!(
            trusted_http_url(Some("https://www.kimi.com/code")).as_deref(),
            Some("https://www.kimi.com/code")
        );
        assert!(trusted_http_url(Some("http://localhost:8080/code")).is_some());
        for value in ["", "file:///etc/passwd", "javascript:alert(1)", "not a url"] {
            assert!(trusted_http_url(Some(value)).is_none(), "{value}");
        }
        assert!(trusted_http_url(None).is_none());
    }
}
