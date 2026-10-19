//! OpenRouter OAuth PKCE flow. Port of upstream `auth/oauth/openrouter.ts`.
//!
//! OpenRouter exchanges an authorization code for a permanent, user-controlled API key
//! rather than an expiring access/refresh token pair. The callback is handled by a
//! one-shot loopback server on an ephemeral port, raced against a manual prompt so
//! remote/headless sessions can paste the redirect URL when the browser cannot reach the
//! loopback server.

use futures::future::BoxFuture;
use serde_json::{Value, json};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::auth::{AuthEvent, AuthInteraction, LoginOptions, ModelAuth, OAuthAuth, OAuthCredential};
use crate::error::ModelsError;
use crate::oauth::{
    CallbackOrManual, OAuthCallbackServerOptions, form_urlencode, generate_pkce, login_cancelled, oauth_callback_host,
    oauth_error, oauth_http_client, search_param, start_oauth_callback_server, url_search_param,
    wait_for_callback_or_manual_input,
};
use crate::types::JsonObject;

pub const AUTHORIZE_URL: &str = "https://openrouter.ai/auth";
pub const TOKEN_URL: &str = "https://openrouter.ai/api/v1/auth/keys";
const LOGIN_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const TOKEN_EXCHANGE_TIMEOUT: Duration = Duration::from_secs(30);
/// `Number.MAX_SAFE_INTEGER`: the minted key does not expire.
pub const PERMANENT_EXPIRY: i64 = 9_007_199_254_740_991;

/// OpenRouter OAuth. `Default` uses upstream's endpoints.
#[derive(Clone, Debug)]
pub struct OpenRouterOAuth {
    pub authorize_url: String,
    pub token_url: String,
    /// Callback listen host. `None`: `PI_OAUTH_CALLBACK_HOST`, else `127.0.0.1`.
    pub callback_host: Option<String>,
}

impl Default for OpenRouterOAuth {
    fn default() -> Self {
        Self {
            authorize_url: AUTHORIZE_URL.to_owned(),
            token_url: TOKEN_URL.to_owned(),
            callback_host: None,
        }
    }
}

fn parse_authorization_input(input: &str) -> Option<String> {
    let value = input.trim();
    if value.is_empty() {
        return None;
    }
    if let Ok(url) = url::Url::parse(value) {
        return url_search_param(&url, "code");
    }
    if value.contains("code=") {
        return search_param(value, "code");
    }
    Some(value.to_owned())
}

fn error_detail(body: &JsonObject) -> Option<String> {
    for field in ["error_description", "message", "error"] {
        if let Some(Value::String(detail)) = body.get(field) {
            return Some(detail.clone());
        }
    }
    match body.get("error") {
        Some(Value::Object(error)) => error.get("message").and_then(Value::as_str).map(str::to_owned),
        _ => None,
    }
}

/// Exchange an authorization code for a permanent API key.
async fn exchange_authorization_code(
    token_url: &str,
    code: &str,
    verifier: &str,
    cancel: &CancellationToken,
) -> Result<OAuthCredential, ModelsError> {
    if cancel.is_cancelled() {
        return Err(login_cancelled());
    }
    let request = async {
        let response = oauth_http_client()
            .post(token_url)
            .header("accept", "application/json")
            .header("content-type", "application/json")
            .body(json!({"code": code, "code_verifier": verifier, "code_challenge_method": "S256"}).to_string())
            .send()
            .await
            .map_err(|error| oauth_error(error.to_string()))?;
        let status = response.status();
        let text = response.text().await.map_err(|error| oauth_error(error.to_string()))?;
        let body = match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(body)) => body,
            Err(_) if status.is_success() => return Err(oauth_error("OpenRouter OAuth returned invalid JSON")),
            Ok(_) | Err(_) => JsonObject::new(),
        };
        Ok((status, body))
    };
    let (status, body) = tokio::select! {
        () = cancel.cancelled() => return Err(login_cancelled()),
        result = tokio::time::timeout(TOKEN_EXCHANGE_TIMEOUT, request) => match result {
            Ok(result) => result?,
            Err(_) => return Err(oauth_error("OpenRouter OAuth token exchange timed out")),
        },
    };

    if !status.is_success() {
        let detail = error_detail(&body)
            .map(|detail| format!(": {detail}"))
            .unwrap_or_default();
        return Err(oauth_error(format!(
            "OpenRouter OAuth key exchange failed (HTTP {}){detail}",
            status.as_u16()
        )));
    }
    let Some(key) = body.get("key").and_then(Value::as_str).filter(|key| !key.is_empty()) else {
        return Err(oauth_error("OpenRouter OAuth response carries no \"key\""));
    };
    Ok(OAuthCredential {
        access: key.to_owned(),
        refresh: String::new(),
        expires: PERMANENT_EXPIRY,
        extra: JsonObject::new(),
    })
}

impl OpenRouterOAuth {
    async fn login_openrouter(&self, interaction: &dyn AuthInteraction) -> Result<OAuthCredential, ModelsError> {
        let cancel = interaction.cancel();
        let pkce = generate_pkce();
        // OpenRouter sends no `state`; the random path keeps stray requests from completing the sign-in.
        let path = format!("/oauth/callback/{}", uuid::Uuid::new_v4());
        let host = self.callback_host.clone().unwrap_or_else(oauth_callback_host);
        let complete = {
            let token_url = self.token_url.clone();
            let verifier = pkce.verifier.clone();
            let cancel = cancel.clone();
            move |code: String| {
                let token_url = token_url.clone();
                let verifier = verifier.clone();
                let cancel = cancel.clone();
                async move { exchange_authorization_code(&token_url, &code, &verifier, &cancel).await }
            }
        };
        let callback = start_oauth_callback_server(
            OAuthCallbackServerOptions::new("OpenRouter", host, 0, path, complete)
                .cancel(cancel.clone())
                .timeout(LOGIN_TIMEOUT),
        )
        .await?;

        let result = async {
            let redirect_uri = callback.redirect_uri().to_owned();
            let params = form_urlencode([
                ("callback_url", redirect_uri.as_str()),
                ("code_challenge", pkce.challenge.as_str()),
                ("code_challenge_method", "S256"),
            ]);
            let mut authorize_url =
                url::Url::parse(&self.authorize_url).map_err(|error| oauth_error(error.to_string()))?;
            authorize_url.set_query(Some(&params));

            interaction.notify(AuthEvent::Progress {
                message: format!("Listening for OpenRouter OAuth callback on {redirect_uri}"),
            });
            interaction.notify(AuthEvent::AuthUrl {
                url: authorize_url.to_string(),
                instructions: Some(
                    "Complete sign-in in your browser. If the browser is on another machine, paste the final redirect URL here."
                        .into(),
                ),
            });

            let result = wait_for_callback_or_manual_input(
                interaction,
                Some(&callback),
                "Complete sign-in in your browser, or paste the authorization code / redirect URL here:",
                redirect_uri.clone(),
            )
            .await?;
            let input = match result {
                CallbackOrManual::Callback(credential) => return Ok(credential),
                CallbackOrManual::Manual(input) => input,
            };
            let Some(code) = parse_authorization_input(&input) else {
                return Err(oauth_error("Missing authorization code"));
            };
            interaction.notify(AuthEvent::Progress {
                message: "Exchanging authorization code for an API key...".into(),
            });
            exchange_authorization_code(&self.token_url, &code, &pkce.verifier, &cancel).await
        }
        .await;
        callback.close();
        result
    }
}

impl OAuthAuth for OpenRouterOAuth {
    fn name(&self) -> &'static str {
        "OpenRouter OAuth"
    }

    fn login_label(&self) -> Option<&str> {
        Some("Sign in with OpenRouter")
    }

    fn login<'a>(
        &'a self,
        interaction: &'a dyn AuthInteraction,
        _options: &'a LoginOptions,
    ) -> BoxFuture<'a, Result<OAuthCredential, ModelsError>> {
        Box::pin(self.login_openrouter(interaction))
    }

    /// The minted key is permanent: refresh returns the credential unchanged.
    fn refresh(&self, credential: OAuthCredential) -> BoxFuture<'_, Result<OAuthCredential, ModelsError>> {
        Box::pin(async move { Ok(credential) })
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
    fn parses_pasted_input_and_error_details() {
        assert_eq!(parse_authorization_input(" "), None);
        assert_eq!(
            parse_authorization_input("http://127.0.0.1:1/oauth/callback/x?code=abc").as_deref(),
            Some("abc")
        );
        assert_eq!(parse_authorization_input("code=abc&x=1").as_deref(), Some("abc"));
        assert_eq!(parse_authorization_input("  abc  ").as_deref(), Some("abc"));

        let body = |value: Value| value.as_object().cloned().unwrap_or_default();
        assert_eq!(
            error_detail(&body(json!({"error": {"message": "invalid code"}}))).as_deref(),
            Some("invalid code")
        );
        assert_eq!(
            error_detail(&body(json!({"message": "m", "error": "e"}))).as_deref(),
            Some("m")
        );
        assert_eq!(error_detail(&body(json!({}))), None);
    }
}
