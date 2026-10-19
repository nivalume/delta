//! Sign in with ChatGPT: OpenAI Responses API token sharing. Port of upstream
//! `auth/oauth/openai-chatgpt.ts`.
//!
//! A public-client flow without a client secret: every login registers a new client
//! (`dynamic_agent_client`), and OpenAI returns the issued client id in the callback. The
//! resulting user access token is sent directly to `api.openai.com`; the `openai-responses`
//! API recognizes it (a key that does not start with `sk-`) and leaves out the request
//! fields token sharing rejects.
//!
//! The flow runs its own loopback server instead of
//! [`crate::oauth::start_oauth_callback_server`]: the callback carries a `client_id` next to
//! the code, and invalid callbacks keep the login waiting.

use futures::future::BoxFuture;
use serde_json::Value;
use std::sync::{Arc, Mutex, PoisonError};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::auth::{AuthEvent, AuthInteraction, AuthPrompt, LoginOptions, ModelAuth, OAuthAuth, OAuthCredential};
use crate::error::ModelsError;
use crate::oauth::callback_server::{read_request_head, send_page};
use crate::oauth::pkce::base64url_encode;
use crate::oauth::{
    form_urlencode, generate_pkce, login_cancelled, oauth_callback_host, oauth_error, oauth_error_html,
    oauth_http_client, oauth_success_html, url_search_param,
};
use crate::types::{JsonObject, now_ms};

/// Every login registers a new client with this id; OpenAI returns the issued client id in
/// the callback.
pub const DYNAMIC_CLIENT_ID: &str = "dynamic_agent_client";
const AGENT_NAME_HINT: &str = "Pi";
pub const AUTHORIZE_URL: &str = "https://auth.openai.com/api/accounts/authorize";
pub const TOKEN_URL: &str = "https://auth.openai.com/api/accounts/oauth/token";
pub const RESOURCE: &str = "https://api.openai.com/v1";
pub const CALLBACK_PORT: u16 = 1455;
const CALLBACK_PATH: &str = "/auth/callback";
/// The scope that allows sending the access token directly to the OpenAI API.
pub const DIRECT_TOKEN_SCOPE: &str = "chatgpt.tokens.use.direct";
const SCOPE: &str = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct";
/// Refresh this long before the real expiry so a request never starts with a token about
/// to expire.
const EXPIRY_MARGIN_MS: i64 = 3 * 60 * 1000;

/// OpenAI Sign in with ChatGPT. `Default` uses upstream's endpoints and port.
#[derive(Clone, Debug)]
pub struct OpenAIChatGPTOAuth {
    pub authorize_url: String,
    pub token_url: String,
    /// Callback listen host. `None`: `PI_OAUTH_CALLBACK_HOST`, else `127.0.0.1`.
    pub callback_host: Option<String>,
    /// Callback port; the redirect URI is `http://127.0.0.1:<port>/auth/callback`. Login
    /// fails when the port is taken.
    pub callback_port: u16,
}

impl Default for OpenAIChatGPTOAuth {
    fn default() -> Self {
        Self {
            authorize_url: AUTHORIZE_URL.to_owned(),
            token_url: TOKEN_URL.to_owned(),
            callback_host: None,
            callback_port: CALLBACK_PORT,
        }
    }
}

/// The authorization code and the client id OpenAI issued for this login.
#[derive(Clone, Debug, PartialEq, Eq)]
struct AuthorizationResult {
    code: String,
    client_id: String,
}

fn random_value() -> String {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).expect("OS random number generator is unavailable");
    base64url_encode(&bytes)
}

fn authorization_result_from_callback(url: &url::Url, expected_state: &str) -> Result<AuthorizationResult, String> {
    let Some(code) = url_search_param(url, "code").filter(|code| !code.is_empty()) else {
        return Err("Missing authorization code".into());
    };
    let Some(state) = url_search_param(url, "state").filter(|state| !state.is_empty()) else {
        return Err("Missing OAuth state".into());
    };
    if state != expected_state {
        return Err("OAuth state mismatch".into());
    }
    let client_id = url_search_param(url, "client_id")
        .map(|id| id.trim().to_owned())
        .filter(|id| !id.is_empty());
    let Some(client_id) = client_id else {
        return Err("OpenAI OAuth registration callback did not contain an issued client ID".into());
    };
    Ok(AuthorizationResult { code, client_id })
}

fn authorization_result_from_manual_input(
    input: &str,
    expected_state: &str,
    redirect_uri: &str,
) -> Result<AuthorizationResult, ModelsError> {
    let url = url::Url::parse(input.trim()).map_err(|_| oauth_error("Paste the full callback URL from the browser"))?;
    let expected = url::Url::parse(redirect_uri).map_err(|error| oauth_error(error.to_string()))?;
    if url.origin() != expected.origin() || url.path() != expected.path() {
        return Err(oauth_error(format!(
            "The pasted callback URL must start with {redirect_uri}"
        )));
    }
    if let Some(error) = url_search_param(&url, "error").filter(|error| !error.is_empty()) {
        return Err(oauth_error(format!("ChatGPT authorization failed: {error}")));
    }
    authorization_result_from_callback(&url, expected_state).map_err(oauth_error)
}

type CallbackResult = Result<AuthorizationResult, ModelsError>;

/// State shared by the callback server's tasks.
struct CallbackShared {
    expected_state: String,
    redirect_uri: String,
    /// Settles the login once; later callbacks still get their page.
    sender: Mutex<Option<oneshot::Sender<CallbackResult>>>,
}

impl CallbackShared {
    fn settle(&self, result: CallbackResult) {
        let sender = self.sender.lock().unwrap_or_else(PoisonError::into_inner).take();
        if let Some(sender) = sender {
            let _ = sender.send(result);
        }
    }
}

/// The running loopback server. Cancelling `shutdown` stops listening and closes every
/// open connection, including spare connections the browser opened ahead of time.
struct CallbackServer {
    result: oneshot::Receiver<CallbackResult>,
    shutdown: CancellationToken,
}

async fn start_callback_server(
    host: &str,
    port: u16,
    expected_state: &str,
    redirect_uri: &str,
) -> std::io::Result<CallbackServer> {
    let listener = TcpListener::bind((host, port)).await?;
    let (sender, result) = oneshot::channel();
    let shared = Arc::new(CallbackShared {
        expected_state: expected_state.to_owned(),
        redirect_uri: redirect_uri.to_owned(),
        sender: Mutex::new(Some(sender)),
    });
    let shutdown = CancellationToken::new();
    tokio::spawn(accept_loop(listener, shared, shutdown.clone()));
    Ok(CallbackServer { result, shutdown })
}

async fn accept_loop(listener: TcpListener, shared: Arc<CallbackShared>, shutdown: CancellationToken) {
    loop {
        let accepted = tokio::select! {
            biased;
            () = shutdown.cancelled() => break,
            accepted = listener.accept() => accepted,
        };
        match accepted {
            Ok((stream, _)) => {
                let shared = shared.clone();
                let shutdown = shutdown.clone();
                tokio::spawn(async move {
                    tokio::select! {
                        () = shutdown.cancelled() => {}
                        () = handle_connection(stream, &shared) => {}
                    }
                });
            }
            Err(error) => {
                shared.settle(Err(oauth_error(error.to_string())));
                break;
            }
        }
    }
}

async fn handle_connection(mut stream: TcpStream, shared: &CallbackShared) {
    let Some((_method, target)) = read_request_head(&mut stream).await else {
        let html = oauth_error_html("Internal error while processing the callback.", None);
        send_page(&mut stream, 500, &html).await;
        return;
    };
    let Ok(url) = url::Url::parse(&shared.redirect_uri).and_then(|base| base.join(&target)) else {
        let html = oauth_error_html("Internal error while processing the callback.", None);
        send_page(&mut stream, 500, &html).await;
        return;
    };
    if url.path() != CALLBACK_PATH {
        send_page(&mut stream, 404, &oauth_error_html("Callback route not found.", None)).await;
        return;
    }
    if let Some(error) = url_search_param(&url, "error").filter(|error| !error.is_empty()) {
        let html = oauth_error_html("ChatGPT was not connected.", Some(&format!("Error: {error}")));
        send_page(&mut stream, 400, &html).await;
        shared.settle(Err(oauth_error(format!("ChatGPT authorization failed: {error}"))));
        return;
    }
    match authorization_result_from_callback(&url, &shared.expected_state) {
        Ok(result) => {
            let html = oauth_success_html("ChatGPT authentication completed. You can close this window.");
            send_page(&mut stream, 200, &html).await;
            shared.settle(Ok(result));
        }
        Err(message) => send_page(&mut stream, 400, &oauth_error_html(&message, None)).await,
    }
}

/// `urn:uuid:<lowercase uuid>` for the installation's device id (upstream `agentHostId`).
fn agent_host_id(device_id: Option<&str>) -> Result<String, ModelsError> {
    match device_id {
        Some(id) if is_uuid(id) => Ok(format!("urn:uuid:{}", id.to_ascii_lowercase())),
        _ => Err(oauth_error(
            "Sign in with ChatGPT requires a device ID (UUID) for this installation",
        )),
    }
}

/// `/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i`.
fn is_uuid(value: &str) -> bool {
    let groups: Vec<&str> = value.split('-').collect();
    groups.len() == 5
        && groups
            .iter()
            .zip([8, 4, 4, 4, 12])
            .all(|(group, len)| group.len() == len && group.chars().all(|c| c.is_ascii_hexdigit()))
}

fn require_token_string(token: &JsonObject, field: &str) -> Result<String, ModelsError> {
    match token.get(field) {
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(value.clone()),
        _ => Err(oauth_error(format!("OpenAI OAuth token response has invalid {field}"))),
    }
}

fn credential_from_token_response(token: &JsonObject, client_id: &str) -> Result<OAuthCredential, ModelsError> {
    let access = require_token_string(token, "access_token")?;
    let refresh = require_token_string(token, "refresh_token")?;
    let scope = require_token_string(token, "scope")?;
    let expires_in = match token.get("expires_in").and_then(Value::as_f64) {
        Some(seconds) if seconds.is_finite() && seconds > 0.0 => seconds,
        _ => return Err(oauth_error("OpenAI OAuth token response has invalid expires_in")),
    };
    let scopes: Vec<&str> = scope.split_whitespace().collect();
    if !scopes.contains(&DIRECT_TOKEN_SCOPE) {
        return Err(oauth_error(format!(
            "OpenAI OAuth grant did not include {DIRECT_TOKEN_SCOPE}"
        )));
    }
    let mut extra = JsonObject::new();
    extra.insert("clientId".into(), Value::String(client_id.to_owned()));
    extra.insert(
        "scopes".into(),
        Value::Array(
            scopes
                .into_iter()
                .map(|scope| Value::String(scope.to_owned()))
                .collect(),
        ),
    );
    Ok(OAuthCredential {
        refresh,
        access,
        expires: now_ms() + (expires_in * 1000.0) as i64 - EXPIRY_MARGIN_MS,
        extra,
    })
}

impl OpenAIChatGPTOAuth {
    fn redirect_uri(&self) -> String {
        format!("http://127.0.0.1:{}{CALLBACK_PATH}", self.callback_port)
    }

    async fn request_token(
        &self,
        fields: &[(&str, &str)],
        cancel: &CancellationToken,
    ) -> Result<JsonObject, ModelsError> {
        let request = async {
            let response = oauth_http_client()
                .post(&self.token_url)
                .header("accept", "application/json")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(form_urlencode(fields.iter().copied()))
                .send()
                .await
                .map_err(|error| oauth_error(error.to_string()))?;
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            if !status.is_success() {
                let detail = if text.is_empty() {
                    status.canonical_reason().unwrap_or("").to_owned()
                } else {
                    text
                };
                return Err(oauth_error(format!(
                    "OpenAI OAuth token request failed ({}): {detail}",
                    status.as_u16()
                )));
            }
            match serde_json::from_str::<Value>(&text) {
                Ok(Value::Object(object)) => Ok(object),
                Ok(_) => Err(oauth_error("OpenAI OAuth token response must be an object")),
                Err(error) => Err(oauth_error(error.to_string())),
            }
        };
        if cancel.is_cancelled() {
            return Err(login_cancelled());
        }
        tokio::select! {
            () = cancel.cancelled() => Err(login_cancelled()),
            result = request => result,
        }
    }

    async fn exchange_authorization_code(
        &self,
        result: &AuthorizationResult,
        verifier: &str,
        cancel: &CancellationToken,
    ) -> Result<OAuthCredential, ModelsError> {
        let redirect_uri = self.redirect_uri();
        let token = self
            .request_token(
                &[
                    ("grant_type", "authorization_code"),
                    ("client_id", &result.client_id),
                    ("code", &result.code),
                    ("code_verifier", verifier),
                    ("redirect_uri", &redirect_uri),
                    ("resource", RESOURCE),
                ],
                cancel,
            )
            .await?;
        // The ID token is not used to identify the user or read profile data; its presence
        // is part of the token-response contract.
        match token.get("id_token") {
            Some(Value::String(id_token)) if !id_token.trim().is_empty() => {}
            _ => return Err(oauth_error("OpenAI OAuth token response did not contain an ID token")),
        }
        credential_from_token_response(&token, &result.client_id)
    }

    async fn refresh_access_token(&self, credential: &OAuthCredential) -> Result<OAuthCredential, ModelsError> {
        let client_id = match credential.extra.get("clientId") {
            Some(Value::String(id)) if !id.trim().is_empty() => id.clone(),
            _ => {
                return Err(oauth_error(
                    "Stored OpenAI OAuth credential does not contain an issued client ID; reconnect ChatGPT",
                ));
            }
        };
        let token = self
            .request_token(
                &[
                    ("grant_type", "refresh_token"),
                    ("client_id", &client_id),
                    ("refresh_token", &credential.refresh),
                    ("resource", RESOURCE),
                ],
                &CancellationToken::new(),
            )
            .await?;
        credential_from_token_response(&token, &client_id)
    }

    async fn login_chatgpt(
        &self,
        interaction: &dyn AuthInteraction,
        options: &LoginOptions,
    ) -> Result<OAuthCredential, ModelsError> {
        let device_id = options.device_id.as_ref().map(|device_id| device_id());
        let host_id = agent_host_id(device_id.as_deref())?;
        let cancel = interaction.cancel();
        let pkce = generate_pkce();
        let state = random_value();
        let nonce = random_value();
        let redirect_uri = self.redirect_uri();
        let host = self.callback_host.clone().unwrap_or_else(oauth_callback_host);
        // Without this server, the browser's callback would reach whatever else holds the
        // port (another pending login or the Codex CLI), which rejects it as a state
        // mismatch. Fail with a clear error instead.
        let callback = start_callback_server(&host, self.callback_port, &state, &redirect_uri)
            .await
            .map_err(|error| {
                if error.kind() == std::io::ErrorKind::AddrInUse {
                    oauth_error(format!(
                        "Port {} is in use, probably by an unfinished login in another pi session or by the Codex CLI. Cancel that login and try again.",
                        self.callback_port
                    ))
                } else {
                    oauth_error(error.to_string())
                }
            })?;

        let params = form_urlencode([
            ("client_id", DYNAMIC_CLIENT_ID),
            (
                "agent_name_hint",
                options.agent_name.as_deref().unwrap_or(AGENT_NAME_HINT),
            ),
            ("ext_agent_host_id", host_id.as_str()),
            ("response_type", "code"),
            ("redirect_uri", redirect_uri.as_str()),
            ("resource", RESOURCE),
            ("scope", SCOPE),
            ("state", state.as_str()),
            ("code_challenge", pkce.challenge.as_str()),
            ("code_challenge_method", "S256"),
            ("nonce", nonce.as_str()),
        ]);
        interaction.notify(AuthEvent::AuthUrl {
            url: format!("{}?{params}", self.authorize_url),
            instructions: Some(
                "Complete sign-in in your browser. If the callback does not complete, paste the final redirect URL here."
                    .into(),
            ),
        });

        let manual_cancel = cancel.child_token();
        let _abort_manual = manual_cancel.clone().drop_guard();
        let manual = async {
            let input = interaction
                .prompt(AuthPrompt::ManualCode {
                    message: "Complete login in your browser, or paste the final redirect URL here:".into(),
                    placeholder: Some(redirect_uri.clone()),
                    cancel: manual_cancel.clone(),
                })
                .await?;
            authorization_result_from_manual_input(&input, &state, &redirect_uri)
        };
        let CallbackServer { result, shutdown } = callback;
        // Stops the server and closes its connections when login ends, however it ends.
        let _close_server = shutdown.drop_guard();
        let callback_result = async {
            result
                .await
                .unwrap_or_else(|_| Err(oauth_error("OAuth callback server closed")))
        };

        let outcome = async {
            let result = tokio::select! {
                result = callback_result => result?,
                result = manual => result?,
                () = cancel.cancelled() => return Err(login_cancelled()),
            };
            interaction.notify(AuthEvent::Progress {
                message: "Exchanging authorization code for tokens...".into(),
            });
            self.exchange_authorization_code(&result, &pkce.verifier, &cancel).await
        }
        .await;
        match outcome {
            Err(_) if cancel.is_cancelled() => Err(login_cancelled()),
            outcome => outcome,
        }
    }
}

impl OAuthAuth for OpenAIChatGPTOAuth {
    fn name(&self) -> &'static str {
        "OpenAI (ChatGPT subscription)"
    }

    fn is_subscription(&self) -> bool {
        true
    }

    fn login_label(&self) -> Option<&str> {
        Some("Sign in with ChatGPT")
    }

    fn login<'a>(
        &'a self,
        interaction: &'a dyn AuthInteraction,
        options: &'a LoginOptions,
    ) -> BoxFuture<'a, Result<OAuthCredential, ModelsError>> {
        Box::pin(self.login_chatgpt(interaction, options))
    }

    fn refresh(&self, credential: OAuthCredential) -> BoxFuture<'_, Result<OAuthCredential, ModelsError>> {
        Box::pin(async move { self.refresh_access_token(&credential).await })
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

    const REDIRECT: &str = "http://127.0.0.1:1455/auth/callback";

    #[test]
    fn device_ids_must_be_uuids() {
        assert_eq!(
            agent_host_id(Some("E61BBE28-07EF-466D-8E5D-A344F94AB305")).unwrap(),
            "urn:uuid:e61bbe28-07ef-466d-8e5d-a344f94ab305"
        );
        for id in [None, Some("not-a-uuid"), Some("e61bbe28-07ef-466d-8e5d-a344f94ab30g")] {
            assert!(
                agent_host_id(id)
                    .unwrap_err()
                    .message()
                    .contains("requires a device ID")
            );
        }
    }

    #[test]
    fn callback_urls_need_code_state_and_client_id() {
        let check = |query: &str| {
            let url = url::Url::parse(&format!("{REDIRECT}?{query}")).unwrap();
            authorization_result_from_callback(&url, "s")
        };
        assert_eq!(
            check("code=c&state=s&client_id=%20oaiapp_1%20"),
            Ok(AuthorizationResult {
                code: "c".into(),
                client_id: "oaiapp_1".into()
            })
        );
        assert_eq!(check("state=s").unwrap_err(), "Missing authorization code");
        assert_eq!(check("code=c").unwrap_err(), "Missing OAuth state");
        assert_eq!(check("code=c&state=x").unwrap_err(), "OAuth state mismatch");
        assert_eq!(
            check("code=c&state=s").unwrap_err(),
            "OpenAI OAuth registration callback did not contain an issued client ID"
        );
    }

    #[test]
    fn pasted_input_must_be_the_callback_url() {
        let error = |input: &str| {
            authorization_result_from_manual_input(input, "s", REDIRECT)
                .unwrap_err()
                .message()
                .to_owned()
        };
        assert_eq!(error("just-a-code"), "Paste the full callback URL from the browser");
        assert_eq!(
            error("http://localhost:1455/auth/callback?code=c"),
            format!("The pasted callback URL must start with {REDIRECT}")
        );
        assert_eq!(
            error(&format!("{REDIRECT}?error=access_denied")),
            "ChatGPT authorization failed: access_denied"
        );
        assert!(
            authorization_result_from_manual_input(&format!(" {REDIRECT}?code=c&state=s&client_id=x "), "s", REDIRECT)
                .is_ok()
        );
    }
}
