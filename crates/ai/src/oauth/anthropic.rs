//! Anthropic OAuth flow (Claude Pro/Max). Port of upstream `auth/oauth/anthropic.ts`.
//!
//! Two login methods: a browser login whose redirect reaches a loopback callback server
//! (raced against a pasted redirect URL), and a copy-code login for headless machines.
//! The access token (`sk-ant-oat…`) is used as the API key; the `anthropic-messages` API
//! recognizes it and switches to Claude Code identity and the OAuth beta header.

use futures::future::BoxFuture;
use serde::Deserialize;
use serde_json::{Value, json};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::auth::{
    AuthEvent, AuthInteraction, AuthPrompt, AuthSelectOption, LoginOptions, ModelAuth, OAuthAuth, OAuthCredential,
};
use crate::error::{ModelsError, ModelsErrorCode};
use crate::oauth::{
    CallbackOrManual, OAuthCallbackServerOptions, form_urlencode, format_error_details, generate_pkce, login_cancelled,
    oauth_callback_host, oauth_error, oauth_http_client, search_param, start_oauth_callback_server, url_search_param,
    wait_for_callback_or_manual_input,
};
use crate::types::{JsonObject, now_ms};

/// Public client id of Claude Code (upstream stores it base64-encoded).
pub const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
pub const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
pub const TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
/// Preferred so the port can be forwarded into containers or over SSH. Anthropic accepts
/// any loopback port, so login falls back to a free port when this one cannot be bound.
pub const CALLBACK_PORT: u16 = 53692;
const CALLBACK_PATH: &str = "/callback";
pub const COPY_CODE_REDIRECT_URI: &str = "https://platform.claude.com/oauth/code/callback";
const BROWSER_LOGIN_METHOD: &str = "browser";
const COPY_CODE_LOGIN_METHOD: &str = "copy_code";
const SCOPES: &str =
    "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Credentials expire five minutes before the server-reported expiry.
const EXPIRY_SKEW_MS: i64 = 5 * 60 * 1000;

/// Anthropic (Claude Pro/Max) OAuth. `Default` uses upstream's endpoints and port.
#[derive(Clone, Debug)]
pub struct AnthropicOAuth {
    pub authorize_url: String,
    pub token_url: String,
    /// Callback listen host. `None`: `PI_OAUTH_CALLBACK_HOST`, else `127.0.0.1`.
    pub callback_host: Option<String>,
    /// Preferred callback port; a free port is used when it is taken.
    pub callback_port: u16,
}

impl Default for AnthropicOAuth {
    fn default() -> Self {
        Self {
            authorize_url: AUTHORIZE_URL.to_owned(),
            token_url: TOKEN_URL.to_owned(),
            callback_host: None,
            callback_port: CALLBACK_PORT,
        }
    }
}

/// Code and state parsed from pasted input (upstream `parseAuthorizationInput`).
#[derive(Debug, Default, PartialEq, Eq)]
struct AuthorizationInput {
    code: Option<String>,
    state: Option<String>,
}

fn parse_authorization_input(input: &str) -> AuthorizationInput {
    let value = input.trim();
    if value.is_empty() {
        return AuthorizationInput::default();
    }
    if let Ok(url) = url::Url::parse(value) {
        return AuthorizationInput {
            code: url_search_param(&url, "code"),
            state: url_search_param(&url, "state"),
        };
    }
    if value.contains('#') {
        let mut parts = value.split('#');
        return AuthorizationInput {
            code: parts.next().map(str::to_owned),
            state: parts.next().map(str::to_owned),
        };
    }
    if value.contains("code=") {
        return AuthorizationInput {
            code: search_param(value, "code"),
            state: search_param(value, "state"),
        };
    }
    AuthorizationInput {
        code: Some(value.to_owned()),
        state: None,
    }
}

enum PostError {
    Cancelled,
    Failed(String),
}

/// POST a JSON body; the response text on 2xx. Bounded by a 30 s timeout.
async fn post_json(url: &str, body: &Value, cancel: &CancellationToken) -> Result<String, PostError> {
    let request = async {
        let response = oauth_http_client()
            .post(url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .timeout(REQUEST_TIMEOUT)
            .body(body.to_string())
            .send()
            .await
            .map_err(|error| PostError::Failed(format_error_details(&error)))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|error| PostError::Failed(format_error_details(&error)))?;
        if !status.is_success() {
            return Err(PostError::Failed(format!(
                "HTTP request failed. status={}; url={url}; body={text}",
                status.as_u16()
            )));
        }
        Ok(text)
    };
    tokio::select! {
        () = cancel.cancelled() => Err(PostError::Cancelled),
        result = request => result,
    }
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: String,
    expires_in: f64,
}

fn credential_from(tokens: TokenResponse) -> OAuthCredential {
    OAuthCredential {
        refresh: tokens.refresh_token,
        access: tokens.access_token,
        expires: now_ms() + (tokens.expires_in * 1000.0) as i64 - EXPIRY_SKEW_MS,
        extra: JsonObject::new(),
    }
}

impl AnthropicOAuth {
    async fn exchange_authorization_code(
        &self,
        code: &str,
        state: &str,
        verifier: &str,
        redirect_uri: &str,
        cancel: &CancellationToken,
    ) -> Result<OAuthCredential, ModelsError> {
        let token_url = &self.token_url;
        let body = json!({
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "state": state,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        });
        let response_body = post_json(token_url, &body, cancel).await.map_err(|error| match error {
            PostError::Cancelled => login_cancelled(),
            PostError::Failed(details) => oauth_error(format!(
                "Token exchange request failed. url={token_url}; redirect_uri={redirect_uri}; response_type=authorization_code; details={details}"
            )),
        })?;
        let tokens: TokenResponse = serde_json::from_str(&response_body).map_err(|error| {
            oauth_error(format!(
                "Token exchange returned invalid JSON. url={token_url}; body={response_body}; details={}",
                format_error_details(&error)
            ))
        })?;
        Ok(credential_from(tokens))
    }

    fn authorize_url_for(&self, redirect_uri: &str, challenge: &str, state: &str) -> String {
        let params = form_urlencode([
            ("code", "true"),
            ("client_id", CLIENT_ID),
            ("response_type", "code"),
            ("redirect_uri", redirect_uri),
            ("scope", SCOPES),
            ("code_challenge", challenge),
            ("code_challenge_method", "S256"),
            ("state", state),
        ]);
        format!("{}?{params}", self.authorize_url)
    }

    async fn login_browser(&self, interaction: &dyn AuthInteraction) -> Result<OAuthCredential, ModelsError> {
        let cancel = interaction.cancel();
        let pkce = generate_pkce();
        let host = self.callback_host.clone().unwrap_or_else(oauth_callback_host);
        let start = |port: u16| {
            start_oauth_callback_server(
                OAuthCallbackServerOptions::new("Anthropic", host.clone(), port, CALLBACK_PATH, |code| async move {
                    Ok(code)
                })
                .redirect_host("localhost")
                .state(pkce.verifier.clone())
                .cancel(cancel.clone()),
            )
        };
        // Without a callback server, login continues with the pasted redirect URL.
        let callback = match start(self.callback_port).await {
            Ok(callback) => Some(callback),
            Err(_) => start(0).await.ok(),
        };
        let fallback_redirect_uri = format!("http://localhost:{}{CALLBACK_PATH}", self.callback_port);
        let redirect_uri = callback
            .as_ref()
            .map_or(fallback_redirect_uri, |callback| callback.redirect_uri().to_owned());

        let result = async {
            interaction.notify(AuthEvent::AuthUrl {
                url: self.authorize_url_for(&redirect_uri, &pkce.challenge, &pkce.verifier),
                instructions: Some(
                    "Complete login in your browser. If the browser is on another machine, paste the final redirect URL here."
                        .into(),
                ),
            });

            let result = wait_for_callback_or_manual_input(
                interaction,
                callback.as_ref(),
                "Complete login in your browser, or paste the authorization code / redirect URL here:",
                redirect_uri.clone(),
            )
            .await?;
            let (code, state) = match result {
                CallbackOrManual::Callback(code) => (Some(code), pkce.verifier.clone()),
                CallbackOrManual::Manual(input) => {
                    let parsed = parse_authorization_input(&input);
                    if let Some(state) = parsed.state.as_deref()
                        && !state.is_empty()
                        && state != pkce.verifier
                    {
                        return Err(oauth_error("OAuth state mismatch"));
                    }
                    (parsed.code, parsed.state.unwrap_or_else(|| pkce.verifier.clone()))
                }
            };
            let Some(code) = code.filter(|code| !code.is_empty()) else {
                return Err(oauth_error("Missing authorization code"));
            };
            interaction.notify(AuthEvent::Progress {
                message: "Exchanging authorization code for tokens...".into(),
            });
            self.exchange_authorization_code(&code, &state, &pkce.verifier, &redirect_uri, &cancel)
                .await
        }
        .await;
        if let Some(callback) = &callback {
            callback.close();
        }
        result
    }

    async fn login_copy_code(&self, interaction: &dyn AuthInteraction) -> Result<OAuthCredential, ModelsError> {
        let cancel = interaction.cancel();
        let pkce = generate_pkce();
        interaction.notify(AuthEvent::AuthUrl {
            url: self.authorize_url_for(COPY_CODE_REDIRECT_URI, &pkce.challenge, &pkce.verifier),
            instructions: Some(
                "Complete login in your browser, then copy the code Anthropic shows and paste it here.".into(),
            ),
        });

        let input = interaction
            .prompt(AuthPrompt::ManualCode {
                message: "Paste the code Anthropic shows after you sign in:".into(),
                placeholder: Some("code#state".into()),
                cancel: cancel.clone(),
            })
            .await?;
        let parsed = parse_authorization_input(&input);
        if let Some(state) = parsed.state.as_deref()
            && !state.is_empty()
            && state != pkce.verifier
        {
            return Err(oauth_error("OAuth state mismatch"));
        }
        let Some(code) = parsed.code.filter(|code| !code.is_empty()) else {
            return Err(oauth_error("Missing authorization code"));
        };
        interaction.notify(AuthEvent::Progress {
            message: "Exchanging authorization code for tokens...".into(),
        });
        let state = parsed.state.unwrap_or_else(|| pkce.verifier.clone());
        self.exchange_authorization_code(&code, &state, &pkce.verifier, COPY_CODE_REDIRECT_URI, &cancel)
            .await
    }

    async fn refresh_token(&self, refresh_token: &str) -> Result<OAuthCredential, ModelsError> {
        let token_url = &self.token_url;
        let body = json!({
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": refresh_token,
        });
        let response_body =
            post_json(token_url, &body, &CancellationToken::new())
                .await
                .map_err(|error| match error {
                    PostError::Cancelled => login_cancelled(),
                    PostError::Failed(details) => oauth_error(format!(
                        "Anthropic token refresh request failed. url={token_url}; details={details}"
                    )),
                })?;
        let tokens: TokenResponse = serde_json::from_str(&response_body).map_err(|error| {
            oauth_error(format!(
                "Anthropic token refresh returned invalid JSON. url={token_url}; body={response_body}; details={}",
                format_error_details(&error)
            ))
        })?;
        Ok(credential_from(tokens))
    }
}

impl OAuthAuth for AnthropicOAuth {
    fn name(&self) -> &'static str {
        "Anthropic (Claude Pro/Max)"
    }

    fn is_subscription(&self) -> bool {
        true
    }

    fn login<'a>(
        &'a self,
        interaction: &'a dyn AuthInteraction,
        _options: &'a LoginOptions,
    ) -> BoxFuture<'a, Result<OAuthCredential, ModelsError>> {
        Box::pin(async move {
            let method = interaction
                .prompt(AuthPrompt::Select {
                    message: "Select Anthropic login method:".into(),
                    options: vec![
                        AuthSelectOption {
                            id: BROWSER_LOGIN_METHOD.into(),
                            label: "Browser login (default)".into(),
                            description: None,
                        },
                        AuthSelectOption {
                            id: COPY_CODE_LOGIN_METHOD.into(),
                            label: "Copy code login (headless)".into(),
                            description: None,
                        },
                    ],
                })
                .await?;
            match method.as_str() {
                COPY_CODE_LOGIN_METHOD => self.login_copy_code(interaction).await,
                BROWSER_LOGIN_METHOD => self.login_browser(interaction).await,
                _ => Err(ModelsError::new(
                    ModelsErrorCode::OAuth,
                    format!("Unknown Anthropic login method: {method}"),
                )),
            }
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn input(code: Option<&str>, state: Option<&str>) -> AuthorizationInput {
        AuthorizationInput {
            code: code.map(str::to_owned),
            state: state.map(str::to_owned),
        }
    }

    #[test]
    fn parses_pasted_authorization_input() {
        assert_eq!(parse_authorization_input("  "), input(None, None));
        assert_eq!(
            parse_authorization_input("http://localhost:53692/callback?code=c&state=s"),
            input(Some("c"), Some("s"))
        );
        assert_eq!(parse_authorization_input("c#s#extra"), input(Some("c"), Some("s")));
        assert_eq!(parse_authorization_input("code=c&state=s"), input(Some("c"), Some("s")));
        assert_eq!(parse_authorization_input(" bare "), input(Some("bare"), None));
    }
}
