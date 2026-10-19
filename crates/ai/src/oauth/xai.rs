//! xAI OAuth device-code flow (SuperGrok / X Premium). Port of upstream `auth/oauth/xai.ts`.

use futures::future::BoxFuture;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::auth::{AuthEvent, AuthInteraction, LoginOptions, ModelAuth, OAuthAuth, OAuthCredential};
use crate::error::ModelsError;
use crate::oauth::{
    OAuthDeviceCodePollOptions, OAuthDeviceCodePollResult, form_urlencode, login_cancelled, oauth_error,
    oauth_http_client, poll_oauth_device_code_flow,
};
use crate::types::{JsonObject, now_ms};

pub const CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
pub const SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";
pub const DEVICE_CODE_URL: &str = "https://auth.x.ai/oauth2/device/code";
pub const TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";
/// Refresh slightly before the reported expiry to avoid using a token that dies mid-request.
const REFRESH_SKEW_MS: i64 = 5 * 60 * 1000;
const DEFAULT_TOKEN_LIFETIME_SECONDS: f64 = 3600.0;

/// xAI (Grok/X subscription) OAuth. `Default` uses upstream's endpoints.
#[derive(Clone, Debug)]
pub struct XaiOAuth {
    pub device_code_url: String,
    pub token_url: String,
}

impl Default for XaiOAuth {
    fn default() -> Self {
        Self {
            device_code_url: DEVICE_CODE_URL.to_owned(),
            token_url: TOKEN_URL.to_owned(),
        }
    }
}

struct OAuthHttpResponse {
    ok: bool,
    status: u16,
    body: JsonObject,
}

#[derive(Debug)]
struct XaiDeviceCode {
    device_code: String,
    user_code: String,
    verification_uri: String,
    verification_uri_complete: Option<String>,
    interval_seconds: Option<f64>,
    expires_in_seconds: f64,
}

fn invalid_field(field: &str) -> ModelsError {
    oauth_error(format!("Invalid xAI OAuth response field: {field}"))
}

fn required_string(body: &JsonObject, field: &str) -> Result<String, ModelsError> {
    match body.get(field) {
        Some(Value::String(value)) if !value.is_empty() => Ok(value.clone()),
        _ => Err(invalid_field(field)),
    }
}

fn positive_number(body: &JsonObject, field: &str) -> Result<f64, ModelsError> {
    match body.get(field).and_then(Value::as_f64) {
        Some(value) if value.is_finite() && value > 0.0 => Ok(value),
        _ => Err(invalid_field(field)),
    }
}

/// The verification URI is opened in the user's browser; force it to be an https URL so a
/// malicious response cannot make the opener launch something else.
fn validate_verification_uri(raw: &str) -> Result<String, ModelsError> {
    match url::Url::parse(raw) {
        Ok(url) if url.scheme() == "https" => Ok(url.to_string()),
        _ => Err(oauth_error("Untrusted verification URI in xAI OAuth response")),
    }
}

async fn post_form(
    url: &str,
    fields: &[(&str, &str)],
    cancel: &CancellationToken,
) -> Result<OAuthHttpResponse, ModelsError> {
    let request = async {
        let response = oauth_http_client()
            .post(url)
            .header("Accept", "application/json")
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(form_urlencode(fields.iter().copied()))
            .send()
            .await
            .map_err(|error| oauth_error(error.to_string()))?;
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        let body = match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(body)) => body,
            Ok(_) => JsonObject::new(),
            Err(_) => {
                return Err(oauth_error(format!(
                    "xAI OAuth returned invalid JSON (HTTP {})",
                    status.as_u16()
                )));
            }
        };
        Ok(OAuthHttpResponse {
            ok: status.is_success(),
            status: status.as_u16(),
            body,
        })
    };
    if cancel.is_cancelled() {
        return Err(login_cancelled());
    }
    tokio::select! {
        () = cancel.cancelled() => Err(login_cancelled()),
        result = request => result,
    }
}

fn request_failure(action: &str, response: &OAuthHttpResponse) -> ModelsError {
    let field = |name: &str| match response.body.get(name) {
        Some(Value::String(value)) if !value.is_empty() => Some(value.clone()),
        _ => None,
    };
    let detail = [field("error"), field("error_description")]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(": ");
    let detail = if detail.is_empty() {
        String::new()
    } else {
        format!(": {detail}")
    };
    oauth_error(format!("xAI OAuth {action} failed (HTTP {}){detail}", response.status))
}

fn parse_device_code(body: &JsonObject) -> Result<XaiDeviceCode, ModelsError> {
    // RFC 8628 allows interval 0 (no minimum wait); fall back to the poller's default
    // instead of failing on non-positive or malformed values.
    let interval_seconds = body
        .get("interval")
        .and_then(Value::as_f64)
        .filter(|interval| interval.is_finite() && *interval > 0.0);
    let verification_uri_complete = match body.get("verification_uri_complete") {
        Some(Value::String(uri)) if !uri.is_empty() => Some(validate_verification_uri(uri)?),
        _ => None,
    };
    Ok(XaiDeviceCode {
        device_code: required_string(body, "device_code")?,
        user_code: required_string(body, "user_code")?,
        verification_uri: validate_verification_uri(&required_string(body, "verification_uri")?)?,
        verification_uri_complete,
        interval_seconds,
        expires_in_seconds: positive_number(body, "expires_in")?,
    })
}

fn credentials_from_token_response(
    body: &JsonObject,
    previous_refresh_token: Option<&str>,
) -> Result<OAuthCredential, ModelsError> {
    let access = required_string(body, "access_token")?;
    // xAI may omit refresh_token on refresh when the token is not rotated.
    let refresh = match previous_refresh_token {
        Some(previous) if !previous.is_empty() && !body.contains_key("refresh_token") => previous.to_owned(),
        _ => required_string(body, "refresh_token")?,
    };
    let expires_in_seconds = if body.contains_key("expires_in") {
        positive_number(body, "expires_in")?
    } else {
        DEFAULT_TOKEN_LIFETIME_SECONDS
    };
    Ok(OAuthCredential {
        refresh,
        access,
        expires: now_ms() + (expires_in_seconds * 1000.0) as i64 - REFRESH_SKEW_MS,
        extra: JsonObject::new(),
    })
}

impl XaiOAuth {
    async fn request_device_code(&self, cancel: &CancellationToken) -> Result<XaiDeviceCode, ModelsError> {
        let response = post_form(
            &self.device_code_url,
            &[("client_id", CLIENT_ID), ("scope", SCOPE), ("referrer", "pi")],
            cancel,
        )
        .await?;
        if !response.ok {
            return Err(request_failure("device authorization", &response));
        }
        parse_device_code(&response.body)
    }

    async fn poll_for_tokens(
        &self,
        device: &XaiDeviceCode,
        cancel: &CancellationToken,
    ) -> Result<OAuthCredential, ModelsError> {
        let options = OAuthDeviceCodePollOptions {
            interval_seconds: device.interval_seconds,
            expires_in_seconds: Some(device.expires_in_seconds),
            wait_before_first_poll: true,
            cancel: cancel.clone(),
        };
        poll_oauth_device_code_flow(options, || async {
            let response = post_form(
                &self.token_url,
                &[
                    ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                    ("client_id", CLIENT_ID),
                    ("device_code", &device.device_code),
                ],
                cancel,
            )
            .await?;

            if response.ok {
                return Ok(OAuthDeviceCodePollResult::Complete(credentials_from_token_response(
                    &response.body,
                    None,
                )?));
            }
            Ok(match response.body.get("error").and_then(Value::as_str) {
                Some("authorization_pending") => OAuthDeviceCodePollResult::Pending,
                Some("slow_down") => OAuthDeviceCodePollResult::SlowDown {
                    interval_seconds: response.body.get("interval").and_then(Value::as_f64),
                },
                Some("access_denied" | "authorization_denied") => OAuthDeviceCodePollResult::Failed {
                    message: "xAI device authorization was denied".into(),
                },
                Some("expired_token") => OAuthDeviceCodePollResult::Failed {
                    message: "xAI device code expired".into(),
                },
                _ => OAuthDeviceCodePollResult::Failed {
                    message: request_failure("device token polling", &response).message().to_owned(),
                },
            })
        })
        .await
    }

    async fn login_xai(&self, interaction: &dyn AuthInteraction) -> Result<OAuthCredential, ModelsError> {
        let cancel = interaction.cancel();
        let device = self.request_device_code(&cancel).await?;
        interaction.notify(AuthEvent::DeviceCode {
            user_code: device.user_code.clone(),
            verification_uri: device
                .verification_uri_complete
                .clone()
                .unwrap_or_else(|| device.verification_uri.clone()),
            interval_seconds: device.interval_seconds.map(|seconds| seconds as u64),
            expires_in_seconds: Some(device.expires_in_seconds as u64),
        });
        self.poll_for_tokens(&device, &cancel).await
    }

    async fn refresh_token(&self, refresh_token: &str) -> Result<OAuthCredential, ModelsError> {
        let response = post_form(
            &self.token_url,
            &[
                ("grant_type", "refresh_token"),
                ("client_id", CLIENT_ID),
                ("refresh_token", refresh_token),
            ],
            &CancellationToken::new(),
        )
        .await?;
        if !response.ok {
            return Err(request_failure("token refresh", &response));
        }
        credentials_from_token_response(&response.body, Some(refresh_token))
    }
}

impl OAuthAuth for XaiOAuth {
    fn name(&self) -> &'static str {
        "xAI (Grok/X subscription)"
    }

    fn is_subscription(&self) -> bool {
        true
    }

    fn login_label(&self) -> Option<&str> {
        Some("Sign in with SuperGrok or X Premium")
    }

    fn login<'a>(
        &'a self,
        interaction: &'a dyn AuthInteraction,
        _options: &'a LoginOptions,
    ) -> BoxFuture<'a, Result<OAuthCredential, ModelsError>> {
        Box::pin(self.login_xai(interaction))
    }

    fn refresh(&self, credential: OAuthCredential) -> BoxFuture<'_, Result<OAuthCredential, ModelsError>> {
        Box::pin(async move { self.refresh_token(&credential.refresh).await })
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
