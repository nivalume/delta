//! OpenAI Codex OAuth flow (ChatGPT Plus/Pro subscription). Port of upstream
//! `auth/oauth/openai-codex.ts`.
//!
//! Two login methods: a browser login whose redirect reaches a loopback callback server on
//! the port registered with OpenAI (raced against a pasted redirect URL), and a device-code
//! login for headless machines. The credential stores the ChatGPT account id from the access
//! token (`accountId`); the `openai-codex-responses` API sends it as `chatgpt-account-id`.

use base64::Engine as _;
use base64::alphabet;
use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
use futures::future::BoxFuture;
use serde_json::{Value, json};
use std::fmt::Write as _;
use tokio_util::sync::CancellationToken;

use crate::auth::{
    AuthEvent, AuthInteraction, AuthPrompt, AuthSelectOption, LoginOptions, ModelAuth, OAuthAuth, OAuthCredential,
};
use crate::error::ModelsError;
use crate::oauth::anthropic::parse_authorization_input;
use crate::oauth::{
    CallbackOrManual, OAuthCallbackServerOptions, OAuthDeviceCodePollOptions, OAuthDeviceCodePollResult,
    form_urlencode, generate_pkce, login_cancelled, oauth_callback_host, oauth_error, oauth_http_client,
    poll_oauth_device_code_flow, start_oauth_callback_server, wait_for_callback_or_manual_input,
};
use crate::types::{JsonObject, now_ms};

pub const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
pub const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
/// Registered with OpenAI and shared with the Codex CLI.
pub const CALLBACK_PORT: u16 = 1455;
const CALLBACK_PATH: &str = "/auth/callback";
pub const DEVICE_USER_CODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
pub const DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
pub const DEVICE_VERIFICATION_URI: &str = "https://auth.openai.com/codex/device";
pub const DEVICE_REDIRECT_URI: &str = "https://auth.openai.com/deviceauth/callback";
const DEVICE_CODE_TIMEOUT_SECONDS: u64 = 15 * 60;
const BROWSER_LOGIN_METHOD: &str = "browser";
const DEVICE_CODE_LOGIN_METHOD: &str = "device_code";
const SCOPE: &str = "openid profile email offline_access";
/// JWT claim that holds the ChatGPT account id.
pub const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";

/// OpenAI Codex (ChatGPT Plus/Pro) OAuth. `Default` uses upstream's endpoints and port.
#[derive(Clone, Debug)]
pub struct OpenAICodexOAuth {
    pub authorize_url: String,
    pub token_url: String,
    pub device_user_code_url: String,
    pub device_token_url: String,
    /// Callback listen host. `None`: `PI_OAUTH_CALLBACK_HOST`, else `127.0.0.1`.
    pub callback_host: Option<String>,
    /// Callback port; the redirect URI is `http://localhost:<port>/auth/callback`. There is
    /// no fallback port: when it is taken, login continues with the pasted redirect URL.
    pub callback_port: u16,
}

impl Default for OpenAICodexOAuth {
    fn default() -> Self {
        Self {
            authorize_url: AUTHORIZE_URL.to_owned(),
            token_url: TOKEN_URL.to_owned(),
            device_user_code_url: DEVICE_USER_CODE_URL.to_owned(),
            device_token_url: DEVICE_TOKEN_URL.to_owned(),
            callback_host: None,
            callback_port: CALLBACK_PORT,
        }
    }
}

/// `atob` as browsers implement it ("forgiving base64"): the standard alphabet, padding
/// optional, trailing bits ignored.
const FORGIVING_BASE64: GeneralPurpose = GeneralPurpose::new(
    &alphabet::STANDARD,
    GeneralPurposeConfig::new()
        .with_decode_padding_mode(DecodePaddingMode::Indifferent)
        .with_decode_allow_trailing_bits(true),
);

/// The payload of a JWT, decoded as upstream does (`JSON.parse(atob(parts[1]))`): exactly
/// three dot-separated parts, the payload in standard base64, its bytes read as Latin-1.
/// `None` when any step fails.
pub fn decode_jwt_payload(token: &str) -> Option<Value> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    let payload: String = parts[1].chars().filter(|c| !c.is_ascii_whitespace()).collect();
    let bytes = FORGIVING_BASE64.decode(payload).ok()?;
    let text: String = bytes.into_iter().map(char::from).collect();
    serde_json::from_str(&text).ok()
}

/// The ChatGPT account id of an access token (`https://api.openai.com/auth` →
/// `chatgpt_account_id`), when it is a non-empty string.
pub fn chatgpt_account_id(access_token: &str) -> Option<String> {
    let payload = decode_jwt_payload(access_token)?;
    match payload.get(JWT_CLAIM_PATH)?.get("chatgpt_account_id")? {
        Value::String(id) if !id.is_empty() => Some(id.clone()),
        _ => None,
    }
}

fn create_state() -> String {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).expect("OS random number generator is unavailable");
    bytes.iter().fold(String::with_capacity(32), |mut state, byte| {
        let _ = write!(state, "{byte:02x}");
        state
    })
}

/// Access token, refresh token and absolute expiry of a token response.
struct OAuthToken {
    access: String,
    refresh: String,
    expires: i64,
}

#[derive(Clone, Copy)]
enum TokenOperation {
    Exchange,
    Refresh,
}

impl TokenOperation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Exchange => "exchange",
            Self::Refresh => "refresh",
        }
    }
}

struct HttpResponse {
    status: reqwest::StatusCode,
    text: String,
}

impl HttpResponse {
    fn ok(&self) -> bool {
        self.status.is_success()
    }

    fn status_text(&self) -> &str {
        self.status.canonical_reason().unwrap_or("")
    }
}

/// Send `request`, failing with `Login cancelled` when `cancel` fires (upstream
/// `fetchWithLoginCancellation`).
async fn send(request: reqwest::RequestBuilder, cancel: &CancellationToken) -> Result<HttpResponse, ModelsError> {
    if cancel.is_cancelled() {
        return Err(login_cancelled());
    }
    let request = async {
        let response = request.send().await.map_err(|error| oauth_error(error.to_string()))?;
        let status = response.status();
        let text = response.text().await.map_err(|error| oauth_error(error.to_string()))?;
        Ok(HttpResponse { status, text })
    };
    tokio::select! {
        () = cancel.cancelled() => Err(login_cancelled()),
        result = request => result,
    }
}

fn post_json(url: &str, body: &Value) -> reqwest::RequestBuilder {
    oauth_http_client()
        .post(url)
        .header("Content-Type", "application/json")
        .body(body.to_string())
}

fn post_form(url: &str, fields: &[(&str, &str)]) -> reqwest::RequestBuilder {
    oauth_http_client()
        .post(url)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(form_urlencode(fields.iter().copied()))
}

/// `JSON.parse` of a response body, with the parser's message on failure.
fn parse_json(text: &str) -> Result<Value, ModelsError> {
    serde_json::from_str(text).map_err(|error| oauth_error(error.to_string()))
}

fn non_empty_string(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::String(text)) if !text.is_empty() => Some(text.clone()),
        _ => None,
    }
}

fn read_token_response(response: &HttpResponse, operation: TokenOperation) -> Result<OAuthToken, ModelsError> {
    let operation = operation.as_str();
    if !response.ok() {
        let detail = if response.text.is_empty() {
            response.status_text()
        } else {
            response.text.as_str()
        };
        return Err(oauth_error(format!(
            "OpenAI Codex token {operation} failed ({}): {detail}",
            response.status.as_u16()
        )));
    }
    let json = parse_json(&response.text)?;
    let access = non_empty_string(json.get("access_token"));
    let refresh = non_empty_string(json.get("refresh_token"));
    let expires_in = json.get("expires_in").and_then(Value::as_f64);
    match (access, refresh, expires_in) {
        (Some(access), Some(refresh), Some(expires_in)) => Ok(OAuthToken {
            access,
            refresh,
            expires: now_ms() + (expires_in * 1000.0) as i64,
        }),
        _ => Err(oauth_error(format!(
            "OpenAI Codex token {operation} response missing fields: {json}"
        ))),
    }
}

fn credentials_from_token(token: OAuthToken) -> Result<OAuthCredential, ModelsError> {
    let account_id =
        chatgpt_account_id(&token.access).ok_or_else(|| oauth_error("Failed to extract accountId from token"))?;
    let mut extra = JsonObject::new();
    extra.insert("accountId".into(), Value::String(account_id));
    Ok(OAuthCredential {
        refresh: token.refresh,
        access: token.access,
        expires: token.expires,
        extra,
    })
}

/// The device authorization of the headless login.
#[derive(Debug)]
struct DeviceAuthInfo {
    device_auth_id: String,
    user_code: String,
    interval_seconds: f64,
}

/// `Number(value)` for the device response's `interval`, which may be a string.
fn js_number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => {
            let text = text.trim();
            if text.is_empty() {
                Some(0.0)
            } else {
                Some(text.parse::<f64>().unwrap_or(f64::NAN))
            }
        }
        _ => None,
    }
}

fn status_with_body(prefix: &str, status: reqwest::StatusCode, body: &str) -> String {
    if body.is_empty() {
        format!("{prefix} {}", status.as_u16())
    } else {
        format!("{prefix} {}: {body}", status.as_u16())
    }
}

impl OpenAICodexOAuth {
    fn redirect_uri(&self) -> String {
        format!("http://localhost:{}{CALLBACK_PATH}", self.callback_port)
    }

    async fn exchange_authorization_code(
        &self,
        code: &str,
        verifier: &str,
        redirect_uri: &str,
        cancel: &CancellationToken,
    ) -> Result<OAuthCredential, ModelsError> {
        let request = post_form(
            &self.token_url,
            &[
                ("grant_type", "authorization_code"),
                ("client_id", CLIENT_ID),
                ("code", code),
                ("code_verifier", verifier),
                ("redirect_uri", redirect_uri),
            ],
        );
        let response = send(request, cancel).await?;
        credentials_from_token(read_token_response(&response, TokenOperation::Exchange)?)
    }

    async fn refresh_access_token(&self, refresh_token: &str) -> Result<OAuthCredential, ModelsError> {
        let request = post_form(
            &self.token_url,
            &[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token),
                ("client_id", CLIENT_ID),
            ],
        );
        let response = send(request, &CancellationToken::new())
            .await
            .map_err(|error| oauth_error(format!("OpenAI Codex token refresh error: {}", error.message())))?;
        credentials_from_token(read_token_response(&response, TokenOperation::Refresh)?)
    }

    async fn start_device_auth(&self, cancel: &CancellationToken) -> Result<DeviceAuthInfo, ModelsError> {
        let response = send(
            post_json(&self.device_user_code_url, &json!({"client_id": CLIENT_ID})),
            cancel,
        )
        .await?;
        if !response.ok() {
            if response.status == reqwest::StatusCode::NOT_FOUND {
                return Err(oauth_error(
                    "OpenAI Codex device code login is not enabled for this server. Use browser login or verify the server URL.",
                ));
            }
            return Err(oauth_error(status_with_body(
                "OpenAI Codex device code request failed with status",
                response.status,
                &response.text,
            )));
        }
        let json = parse_json(&response.text)?;
        let interval_seconds = json.get("interval").and_then(js_number);
        match (
            non_empty_string(json.get("device_auth_id")),
            non_empty_string(json.get("user_code")),
            interval_seconds,
        ) {
            (Some(device_auth_id), Some(user_code), Some(interval_seconds))
                if interval_seconds.is_finite() && interval_seconds >= 0.0 =>
            {
                Ok(DeviceAuthInfo {
                    device_auth_id,
                    user_code,
                    interval_seconds,
                })
            }
            _ => Err(oauth_error(format!(
                "Invalid OpenAI Codex device code response: {json}"
            ))),
        }
    }

    /// Poll until the user approves; returns the authorization code and its PKCE verifier.
    async fn poll_device_auth(
        &self,
        device: &DeviceAuthInfo,
        cancel: &CancellationToken,
    ) -> Result<(String, String), ModelsError> {
        let options = OAuthDeviceCodePollOptions {
            interval_seconds: Some(device.interval_seconds),
            expires_in_seconds: Some(DEVICE_CODE_TIMEOUT_SECONDS as f64),
            wait_before_first_poll: false,
            cancel: cancel.clone(),
        };
        poll_oauth_device_code_flow(options, || async {
            let body = json!({"device_auth_id": device.device_auth_id, "user_code": device.user_code});
            let response = send(post_json(&self.device_token_url, &body), cancel).await?;

            if response.ok() {
                let json = parse_json(&response.text)?;
                return Ok(
                    match (
                        non_empty_string(json.get("authorization_code")),
                        non_empty_string(json.get("code_verifier")),
                    ) {
                        (Some(code), Some(verifier)) => OAuthDeviceCodePollResult::Complete((code, verifier)),
                        _ => OAuthDeviceCodePollResult::Failed {
                            message: format!("Invalid OpenAI Codex device auth token response: {json}"),
                        },
                    },
                );
            }

            if matches!(response.status.as_u16(), 403 | 404) {
                return Ok(OAuthDeviceCodePollResult::Pending);
            }

            let error_code =
                serde_json::from_str::<Value>(&response.text)
                    .ok()
                    .and_then(|json| match json.get("error") {
                        Some(Value::Object(error)) => error.get("code").and_then(Value::as_str).map(str::to_owned),
                        Some(Value::String(error)) => Some(error.clone()),
                        _ => None,
                    });
            Ok(match error_code.as_deref() {
                Some("deviceauth_authorization_pending") => OAuthDeviceCodePollResult::Pending,
                Some("slow_down") => OAuthDeviceCodePollResult::SlowDown { interval_seconds: None },
                _ => OAuthDeviceCodePollResult::Failed {
                    message: status_with_body(
                        "OpenAI Codex device auth failed with status",
                        response.status,
                        &response.text,
                    ),
                },
            })
        })
        .await
    }

    async fn login_device_code(&self, interaction: &dyn AuthInteraction) -> Result<OAuthCredential, ModelsError> {
        let cancel = interaction.cancel();
        let device = self.start_device_auth(&cancel).await?;
        interaction.notify(AuthEvent::DeviceCode {
            user_code: device.user_code.clone(),
            verification_uri: DEVICE_VERIFICATION_URI.into(),
            interval_seconds: Some(device.interval_seconds as u64),
            expires_in_seconds: Some(DEVICE_CODE_TIMEOUT_SECONDS),
        });
        let (code, verifier) = self.poll_device_auth(&device, &cancel).await?;
        self.exchange_authorization_code(&code, &verifier, DEVICE_REDIRECT_URI, &cancel)
            .await
    }

    /// The authorize URL (upstream `createAuthorizationFlow`).
    fn authorization_url(&self, challenge: &str, state: &str, originator: &str) -> String {
        let redirect_uri = self.redirect_uri();
        let params = form_urlencode([
            ("response_type", "code"),
            ("client_id", CLIENT_ID),
            ("redirect_uri", redirect_uri.as_str()),
            ("scope", SCOPE),
            ("code_challenge", challenge),
            ("code_challenge_method", "S256"),
            ("state", state),
            ("id_token_add_organizations", "true"),
            ("codex_cli_simplified_flow", "true"),
            ("originator", originator),
        ]);
        format!("{}?{params}", self.authorize_url)
    }

    async fn login_browser(
        &self,
        interaction: &dyn AuthInteraction,
        options: &LoginOptions,
    ) -> Result<OAuthCredential, ModelsError> {
        let cancel = interaction.cancel();
        let pkce = generate_pkce();
        let state = create_state();
        let originator = options.agent_name.as_deref().unwrap_or("pi");
        let url = self.authorization_url(&pkce.challenge, &state, originator);
        let redirect_uri = self.redirect_uri();
        let host = self.callback_host.clone().unwrap_or_else(oauth_callback_host);
        // The port is shared with the Codex CLI; when it is taken, fall back to the pasted
        // redirect URL.
        let callback = start_oauth_callback_server(
            OAuthCallbackServerOptions::new("OpenAI", host, self.callback_port, CALLBACK_PATH, |code| async move {
                Ok(code)
            })
            .state(state.clone())
            .cancel(cancel.clone()),
        )
        .await
        .ok();

        interaction.notify(AuthEvent::AuthUrl {
            url,
            instructions: Some("A browser window should open. Complete login to finish.".into()),
        });

        let result = async {
            let result = wait_for_callback_or_manual_input(
                interaction,
                callback.as_ref(),
                "Complete login in your browser, or paste the authorization code / redirect URL here:",
                redirect_uri.clone(),
            )
            .await?;
            let code = match result {
                CallbackOrManual::Callback(code) => Some(code),
                CallbackOrManual::Manual(input) => {
                    let parsed = parse_authorization_input(&input);
                    if let Some(parsed_state) = parsed.state.as_deref()
                        && !parsed_state.is_empty()
                        && parsed_state != state
                    {
                        return Err(oauth_error("State mismatch"));
                    }
                    parsed.code
                }
            };
            let Some(code) = code.filter(|code| !code.is_empty()) else {
                return Err(oauth_error("Missing authorization code"));
            };
            self.exchange_authorization_code(&code, &pkce.verifier, &redirect_uri, &cancel)
                .await
        }
        .await;
        if let Some(callback) = &callback {
            callback.close();
        }
        result
    }
}

impl OAuthAuth for OpenAICodexOAuth {
    fn name(&self) -> &'static str {
        "OpenAI (ChatGPT Plus/Pro)"
    }

    fn is_subscription(&self) -> bool {
        true
    }

    fn login<'a>(
        &'a self,
        interaction: &'a dyn AuthInteraction,
        options: &'a LoginOptions,
    ) -> BoxFuture<'a, Result<OAuthCredential, ModelsError>> {
        Box::pin(async move {
            let method = interaction
                .prompt(AuthPrompt::Select {
                    message: "Select OpenAI Codex login method:".into(),
                    options: vec![
                        AuthSelectOption {
                            id: BROWSER_LOGIN_METHOD.into(),
                            label: "Browser login (default)".into(),
                            description: None,
                        },
                        AuthSelectOption {
                            id: DEVICE_CODE_LOGIN_METHOD.into(),
                            label: "Device code login (headless)".into(),
                            description: None,
                        },
                    ],
                })
                .await?;
            match method.as_str() {
                DEVICE_CODE_LOGIN_METHOD => self.login_device_code(interaction).await,
                BROWSER_LOGIN_METHOD => self.login_browser(interaction, options).await,
                _ => Err(oauth_error(format!("Unknown OpenAI Codex login method: {method}"))),
            }
        })
    }

    fn refresh(&self, credential: OAuthCredential) -> BoxFuture<'_, Result<OAuthCredential, ModelsError>> {
        Box::pin(async move { self.refresh_access_token(&credential.refresh).await })
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

    fn token(payload: &Value) -> String {
        let encoded = base64::engine::general_purpose::STANDARD.encode(payload.to_string());
        format!("aaa.{encoded}.bbb")
    }

    #[test]
    fn reads_the_account_id_from_the_access_token() {
        let payload = json!({JWT_CLAIM_PATH: {"chatgpt_account_id": "acc_1"}});
        assert_eq!(chatgpt_account_id(&token(&payload)).as_deref(), Some("acc_1"));
        // Padding is optional, as for `atob`.
        let unpadded = token(&payload).replace('=', "");
        assert_eq!(chatgpt_account_id(&unpadded).as_deref(), Some("acc_1"));
        assert_eq!(chatgpt_account_id("only.two"), None);
        assert_eq!(
            chatgpt_account_id(&token(&json!({JWT_CLAIM_PATH: {"chatgpt_account_id": ""}}))),
            None
        );
        assert_eq!(chatgpt_account_id(&token(&json!({"sub": "x"}))), None);
        assert_eq!(chatgpt_account_id("a.!!!.c"), None);
    }

    #[test]
    fn state_is_sixteen_random_bytes_in_hex() {
        let state = create_state();
        assert_eq!(state.len(), 32);
        assert!(state.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(state, create_state());
    }

    #[test]
    fn device_interval_accepts_numbers_and_numeric_strings() {
        assert_eq!(js_number(&json!(5)), Some(5.0));
        assert_eq!(js_number(&json!(" 5 ")), Some(5.0));
        assert_eq!(js_number(&json!("")), Some(0.0));
        assert!(js_number(&json!("soon")).unwrap().is_nan());
        assert_eq!(js_number(&json!(null)), None);
    }
}
