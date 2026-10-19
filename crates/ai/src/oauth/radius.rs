//! Radius gateway OAuth flow. Port of upstream `auth/oauth/radius.ts`.
//!
//! Radius is a pi-messages gateway. The OAuth client APIs live on the configured gateway;
//! only the interactive browser authorization endpoint is discovered. The model catalog is
//! loaded by the Radius provider ([`crate::providers::radius`]).

use futures::future::BoxFuture;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::auth::{
    AuthEvent, AuthInteraction, AuthPrompt, AuthSelectOption, LoginOptions, ModelAuth, OAuthAuth, OAuthCredential,
};
use crate::error::ModelsError;
use crate::oauth::{
    OAuthCallbackServerOptions, OAuthDeviceCodePollOptions, OAuthDeviceCodePollResult, form_urlencode, generate_pkce,
    login_cancelled, oauth_error, oauth_http_client, poll_oauth_device_code_flow, start_oauth_callback_server,
};
use crate::providers::radius::normalize_radius_gateway_url;
use crate::types::{JsonObject, now_ms};

pub const CALLBACK_HOST: &str = "127.0.0.1";
pub const CALLBACK_PORT: u16 = 1456;
pub const CALLBACK_PATH: &str = "/oauth/callback";
pub const CLIENT_ID: &str = "pi-gateway";
pub const SCOPE: &str = "gateway offline_access";
const TOKEN_EXPIRY_SKEW_MS: i64 = 60_000;
const LOGIN_METHOD_BROWSER: &str = "browser";
const LOGIN_METHOD_DEVICE_CODE: &str = "device-code";
const DEVICE_CODE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:device_code";

/// Options of [`create_radius_oauth`] (upstream `RadiusOAuthOptions`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RadiusOAuthOptions {
    /// Display name, used in the login prompt and as the OAuth method's name.
    pub name: String,
    /// Gateway URL. A missing scheme means `https://`; trailing slashes are removed.
    pub gateway: String,
}

/// OAuth of a Radius gateway (upstream `createRadiusOAuth`).
pub fn create_radius_oauth(options: RadiusOAuthOptions) -> RadiusOAuth {
    RadiusOAuth {
        name: options.name,
        gateway: normalize_radius_gateway_url(&options.gateway),
        callback_port: CALLBACK_PORT,
    }
}

/// Radius gateway OAuth: browser login (PKCE with a loopback callback) or device code.
#[derive(Clone, Debug)]
pub struct RadiusOAuth {
    name: String,
    gateway: String,
    /// Loopback callback port of the browser login. Default [`CALLBACK_PORT`]; it is part
    /// of the redirect URI, so it must not be `0`.
    pub callback_port: u16,
}

impl RadiusOAuth {
    /// The normalized gateway URL.
    pub fn gateway(&self) -> &str {
        &self.gateway
    }

    fn redirect_uri(&self) -> String {
        format!("http://{CALLBACK_HOST}:{}{CALLBACK_PATH}", self.callback_port)
    }
}

/// An OAuth error response (upstream `OAuthResponseError`): the message plus the OAuth
/// `error` code, which device polling maps to its outcomes.
struct OAuthResponseError {
    oauth_error: Option<String>,
    error: ModelsError,
}

enum TokenError {
    Response(OAuthResponseError),
    Other(ModelsError),
}

impl From<TokenError> for ModelsError {
    fn from(error: TokenError) -> Self {
        match error {
            TokenError::Response(response) => response.error,
            TokenError::Other(error) => error,
        }
    }
}

/// `<message>: <detail>`, the detail from the body's `error` and `error_description`
/// (upstream `readOAuthResponseError`).
fn oauth_response_error(status: u16, text: &str, message: &str) -> OAuthResponseError {
    let mut oauth_error_code = None;
    let mut description = None;
    if !text.is_empty() {
        match serde_json::from_str::<Value>(text) {
            Ok(Value::Object(data)) => {
                oauth_error_code = data.get("error").and_then(Value::as_str).map(str::to_owned);
                description = data.get("error_description").and_then(Value::as_str).map(str::to_owned);
            }
            // `null.error` throws in JavaScript; the text is then the description.
            Ok(Value::Null) | Err(_) => description = Some(text.to_owned()),
            Ok(_) => {}
        }
    }
    // JavaScript truthiness: empty strings count as missing.
    let code = oauth_error_code.as_deref().filter(|code| !code.is_empty());
    let description_text = description.as_deref().filter(|description| !description.is_empty());
    let detail = match (code, description_text) {
        (Some(code), Some(description)) => format!("{code}: {description}"),
        (Some(code), None) => code.to_owned(),
        (None, Some(description)) => description.to_owned(),
        (None, None) => status.to_string(),
    };
    OAuthResponseError {
        oauth_error: oauth_error_code,
        error: oauth_error(format!("{message}: {detail}")),
    }
}

/// `new URL(path, gateway)`: an absolute path replaces the gateway's path.
fn gateway_url(gateway: &str, path: &str) -> Result<url::Url, ModelsError> {
    url::Url::parse(gateway)
        .and_then(|base| base.join(path))
        .map_err(|error| oauth_error(format!("Invalid URL: {gateway} ({error})")))
}

/// Run `request` unless `cancel` fires first (`Login cancelled`).
async fn cancellable<T>(
    cancel: &CancellationToken,
    request: impl Future<Output = Result<T, ModelsError>>,
) -> Result<T, ModelsError> {
    if cancel.is_cancelled() {
        return Err(login_cancelled());
    }
    tokio::select! {
        () = cancel.cancelled() => Err(login_cancelled()),
        result = request => result,
    }
}

/// `GET <gateway>/v1/oauth`: the browser authorization endpoint.
async fn load_authorization_endpoint(gateway: &str, cancel: &CancellationToken) -> Result<String, ModelsError> {
    let url = gateway_url(gateway, "/v1/oauth")?;
    cancellable(cancel, async {
        let response = oauth_http_client()
            .get(url)
            .header("accept", "application/json")
            .send()
            .await
            .map_err(|error| oauth_error(error.to_string()))?;
        let status = response.status();
        let text = response.text().await.map_err(|error| oauth_error(error.to_string()))?;
        if !status.is_success() {
            return Err(oauth_error(format!(
                "Could not load Radius OAuth config from {gateway}: {} {text}",
                status.as_u16()
            )));
        }
        let discovery: Value = serde_json::from_str(&text).map_err(|error| oauth_error(error.to_string()))?;
        match discovery.get("authorizationEndpoint") {
            Some(Value::String(endpoint)) => Ok(endpoint.clone()),
            _ => Err(oauth_error(format!("Invalid Radius OAuth config from {gateway}"))),
        }
    })
    .await
}

/// A form `POST` to the gateway: the status and body text.
async fn post_form(
    gateway: &str,
    path: &str,
    fields: &[(&str, &str)],
    cancel: &CancellationToken,
) -> Result<(u16, String), ModelsError> {
    let url = gateway_url(gateway, path)?;
    cancellable(cancel, async {
        let response = oauth_http_client()
            .post(url)
            .header("accept", "application/json")
            .header("content-type", "application/x-www-form-urlencoded")
            .body(form_urlencode(fields.iter().copied()))
            .send()
            .await
            .map_err(|error| oauth_error(error.to_string()))?;
        let status = response.status().as_u16();
        let text = response.text().await.map_err(|error| oauth_error(error.to_string()))?;
        Ok((status, text))
    })
    .await
}

fn is_success(status: u16) -> bool {
    (200..300).contains(&status)
}

/// `POST <gateway>/v1/oauth/token` (upstream `requestOAuthToken`).
async fn request_oauth_token(
    gateway: &str,
    fields: &[(&str, &str)],
    cancel: &CancellationToken,
) -> Result<OAuthCredential, TokenError> {
    let (status, text) = post_form(gateway, "/v1/oauth/token", fields, cancel)
        .await
        .map_err(TokenError::Other)?;
    if !is_success(status) {
        return Err(TokenError::Response(oauth_response_error(
            status,
            &text,
            "Radius OAuth token request failed",
        )));
    }
    let data: JsonObject =
        serde_json::from_str(&text).map_err(|error| TokenError::Other(oauth_error(error.to_string())))?;
    let invalid = || TokenError::Other(oauth_error("Invalid Radius OAuth token response"));
    let access = data.get("access_token").and_then(Value::as_str).ok_or_else(invalid)?;
    let refresh = data.get("refresh_token").and_then(Value::as_str).ok_or_else(invalid)?;
    let expires_in = data.get("expires_in").and_then(Value::as_f64).ok_or_else(invalid)?;
    let mut extra = JsonObject::new();
    if let Some(scope) = data.get("scope").filter(|scope| !scope.is_null()) {
        extra.insert("scope".into(), scope.clone());
    }
    Ok(OAuthCredential {
        access: access.to_owned(),
        refresh: refresh.to_owned(),
        expires: now_ms() + (expires_in * 1000.0) as i64 - TOKEN_EXPIRY_SKEW_MS,
        extra,
    })
}

/// Device authorization response (RFC 8628).
struct DeviceAuthorization {
    device_code: String,
    user_code: String,
    verification_uri: String,
    expires_in: f64,
    interval: Option<f64>,
}

/// A truthy string field: non-empty text, or a number as JavaScript would accept it.
fn truthy_string(data: &JsonObject, field: &str) -> Option<String> {
    match data.get(field)? {
        Value::String(value) if !value.is_empty() => Some(value.clone()),
        Value::Number(number) if number.as_f64().is_some_and(|value| value != 0.0) => Some(number.to_string()),
        _ => None,
    }
}

async fn request_device_authorization(
    gateway: &str,
    cancel: &CancellationToken,
) -> Result<DeviceAuthorization, ModelsError> {
    let (status, text) = post_form(
        gateway,
        "/v1/oauth/device",
        &[("client_id", CLIENT_ID), ("scope", SCOPE)],
        cancel,
    )
    .await?;
    if !is_success(status) {
        return Err(oauth_response_error(status, &text, "Radius OAuth device authorization failed").error);
    }
    let data: JsonObject = serde_json::from_str(&text).map_err(|error| oauth_error(error.to_string()))?;
    let missing = || oauth_error("Radius OAuth device authorization response is missing required fields");
    let expires_in = data
        .get("expires_in")
        .and_then(Value::as_f64)
        .filter(|value| *value != 0.0 && !value.is_nan())
        .ok_or_else(missing)?;
    Ok(DeviceAuthorization {
        device_code: truthy_string(&data, "device_code").ok_or_else(missing)?,
        user_code: truthy_string(&data, "user_code").ok_or_else(missing)?,
        verification_uri: truthy_string(&data, "verification_uri").ok_or_else(missing)?,
        expires_in,
        interval: data.get("interval").and_then(Value::as_f64),
    })
}

impl RadiusOAuth {
    async fn login_with_browser(
        &self,
        authorization_endpoint: &str,
        interaction: &dyn AuthInteraction,
    ) -> Result<OAuthCredential, ModelsError> {
        let cancel = interaction.cancel();
        let pkce = generate_pkce();
        let state = uuid::Uuid::new_v4().to_string();
        let redirect_uri = self.redirect_uri();
        let mut authorize_url =
            url::Url::parse(authorization_endpoint).map_err(|error| oauth_error(format!("Invalid URL: {error}")))?;
        authorize_url.set_query(Some(&form_urlencode([
            ("response_type", "code"),
            ("client_id", CLIENT_ID),
            ("redirect_uri", redirect_uri.as_str()),
            ("scope", SCOPE),
            ("code_challenge", pkce.challenge.as_str()),
            ("code_challenge_method", "S256"),
            ("handoff", "url"),
            ("state", state.as_str()),
        ])));

        let complete = {
            let gateway = self.gateway.clone();
            let redirect_uri = redirect_uri.clone();
            let verifier = pkce.verifier.clone();
            let cancel = cancel.clone();
            move |code: String| {
                let gateway = gateway.clone();
                let redirect_uri = redirect_uri.clone();
                let verifier = verifier.clone();
                let cancel = cancel.clone();
                async move {
                    let fields = [
                        ("grant_type", "authorization_code"),
                        ("client_id", CLIENT_ID),
                        ("redirect_uri", redirect_uri.as_str()),
                        ("code", code.as_str()),
                        ("code_verifier", verifier.as_str()),
                    ];
                    request_oauth_token(&gateway, &fields, &cancel)
                        .await
                        .map_err(ModelsError::from)
                }
            }
        };
        let callback = start_oauth_callback_server(
            OAuthCallbackServerOptions::new("Radius", CALLBACK_HOST, self.callback_port, CALLBACK_PATH, complete)
                .state(state.clone())
                .cancel(cancel.clone()),
        )
        .await?;
        interaction.notify(AuthEvent::Progress {
            message: format!("Listening for OAuth callback on {redirect_uri}"),
        });
        interaction.notify(AuthEvent::AuthUrl {
            url: authorize_url.to_string(),
            instructions: Some("Continue in your browser.".into()),
        });

        let result = callback.wait().await;
        callback.close();
        result?.ok_or_else(|| oauth_error("OAuth callback did not complete."))
    }

    async fn login_with_device_code(&self, interaction: &dyn AuthInteraction) -> Result<OAuthCredential, ModelsError> {
        let cancel = interaction.cancel();
        let device = request_device_authorization(&self.gateway, &cancel).await?;
        interaction.notify(AuthEvent::DeviceCode {
            user_code: device.user_code.clone(),
            verification_uri: device.verification_uri.clone(),
            interval_seconds: device.interval.map(|seconds| seconds as u64),
            expires_in_seconds: Some(device.expires_in as u64),
        });

        let options = OAuthDeviceCodePollOptions {
            interval_seconds: device.interval,
            expires_in_seconds: Some(device.expires_in),
            wait_before_first_poll: false,
            cancel: cancel.clone(),
        };
        poll_oauth_device_code_flow(options, || async {
            let fields = [
                ("grant_type", DEVICE_CODE_GRANT_TYPE),
                ("client_id", CLIENT_ID),
                ("device_code", device.device_code.as_str()),
            ];
            match request_oauth_token(&self.gateway, &fields, &cancel).await {
                Ok(credential) => Ok(OAuthDeviceCodePollResult::Complete(credential)),
                Err(TokenError::Other(error)) => Err(error),
                Err(TokenError::Response(response)) => match response.oauth_error.as_deref() {
                    Some("authorization_pending") => Ok(OAuthDeviceCodePollResult::Pending),
                    Some("slow_down") => Ok(OAuthDeviceCodePollResult::SlowDown { interval_seconds: None }),
                    Some("expired_token") => Ok(OAuthDeviceCodePollResult::Failed {
                        message: "Device authorization expired.".into(),
                    }),
                    Some("access_denied") => Ok(OAuthDeviceCodePollResult::Failed {
                        message: "Device authorization was denied.".into(),
                    }),
                    _ => Err(response.error),
                },
            }
        })
        .await
    }

    async fn login_radius(&self, interaction: &dyn AuthInteraction) -> Result<OAuthCredential, ModelsError> {
        let method = interaction
            .prompt(AuthPrompt::Select {
                message: format!("Sign in to {}:", self.name),
                options: vec![
                    AuthSelectOption {
                        id: LOGIN_METHOD_BROWSER.into(),
                        label: "Sign in with browser (recommended)".into(),
                        description: None,
                    },
                    AuthSelectOption {
                        id: LOGIN_METHOD_DEVICE_CODE.into(),
                        label: "Sign in with device code (when signing in from another device)".into(),
                        description: None,
                    },
                ],
            })
            .await?;
        match method.as_str() {
            LOGIN_METHOD_DEVICE_CODE => self.login_with_device_code(interaction).await,
            LOGIN_METHOD_BROWSER => {
                let endpoint = load_authorization_endpoint(&self.gateway, &interaction.cancel()).await?;
                self.login_with_browser(&endpoint, interaction).await
            }
            other => Err(oauth_error(format!("Unknown {} sign-in method: {other}", self.name))),
        }
    }
}

impl OAuthAuth for RadiusOAuth {
    fn name(&self) -> &str {
        &self.name
    }

    fn login<'a>(
        &'a self,
        interaction: &'a dyn AuthInteraction,
        _options: &'a LoginOptions,
    ) -> BoxFuture<'a, Result<OAuthCredential, ModelsError>> {
        Box::pin(self.login_radius(interaction))
    }

    fn refresh(&self, credential: OAuthCredential) -> BoxFuture<'_, Result<OAuthCredential, ModelsError>> {
        Box::pin(async move {
            let fields = [
                ("grant_type", "refresh_token"),
                ("client_id", CLIENT_ID),
                ("refresh_token", credential.refresh.as_str()),
            ];
            request_oauth_token(&self.gateway, &fields, &CancellationToken::new())
                .await
                .map_err(ModelsError::from)
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_oauth_error_responses() {
        let error = |status, text: &str| oauth_response_error(status, text, "failed");
        let response = error(400, r#"{"error":"invalid_grant","error_description":"code expired"}"#);
        assert_eq!(response.oauth_error.as_deref(), Some("invalid_grant"));
        assert_eq!(response.error.message(), "failed: invalid_grant: code expired");
        assert_eq!(
            error(400, r#"{"error":"authorization_pending"}"#).error.message(),
            "failed: authorization_pending"
        );
        assert_eq!(error(502, "Bad Gateway").error.message(), "failed: Bad Gateway");
        assert_eq!(error(500, "").error.message(), "failed: 500");
        assert_eq!(error(500, "{}").error.message(), "failed: 500");
        assert_eq!(error(500, "null").error.message(), "failed: null");
    }

    #[test]
    fn gateway_paths_replace_the_gateway_path() {
        assert_eq!(
            gateway_url("https://radius.example/base", "/v1/oauth")
                .unwrap()
                .as_str(),
            "https://radius.example/v1/oauth"
        );
    }
}
