//! Tests of the built-in OAuth flows (Anthropic, OpenRouter, xAI) and their provider
//! wiring. Ported from upstream `test/oauth-auth.test.ts` (the parts for these flows),
//! `test/anthropic-oauth.test.ts`, `test/openrouter-oauth.test.ts` and
//! `test/xai-oauth.test.ts`.
//!
//! Token endpoints are wiremock servers; browser callbacks are real loopback requests.
//! Upstream's xAI timing tests use fake timers; here the server sends 1-second intervals
//! and the test checks the gaps in real time.

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

use delta_ai::auth::{
    AuthEvent, AuthInteraction, AuthPrompt, AuthResolutionOverrides, AuthSelectOption, LoginOptions, OAuthAuth,
};
use delta_ai::models::CreateModelsOptions;
use delta_ai::oauth::openrouter::PERMANENT_EXPIRY;
use delta_ai::oauth::{AnthropicOAuth, OpenRouterOAuth, XaiOAuth, pkce};
use delta_ai::providers::builtin::{anthropic_provider, builtin_provider};
use delta_ai::types::now_ms;
use delta_ai::{
    Credential, CredentialStore, InMemoryCredentialStore, JsonObject, Models, ModelsError, ModelsErrorCode,
    OAuthCredential, Provider,
};
use futures::future::BoxFuture;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn http() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

type PromptFn = Arc<dyn Fn(AuthPrompt) -> BoxFuture<'static, Result<String, ModelsError>> + Send + Sync>;
type NotifyFn = Arc<dyn Fn(&AuthEvent) + Send + Sync>;

/// Records events and prompts; `prompt` and `on_event` script the user.
struct TestInteraction {
    prompt: PromptFn,
    on_event: NotifyFn,
    cancel: CancellationToken,
    events: Mutex<Vec<AuthEvent>>,
    prompts: Mutex<Vec<AuthPrompt>>,
}

impl TestInteraction {
    fn new(
        prompt: impl Fn(AuthPrompt) -> BoxFuture<'static, Result<String, ModelsError>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            prompt: Arc::new(prompt),
            on_event: Arc::new(|_| {}),
            cancel: CancellationToken::new(),
            events: Mutex::new(Vec::new()),
            prompts: Mutex::new(Vec::new()),
        }
    }

    fn on_event(mut self, on_event: impl Fn(&AuthEvent) + Send + Sync + 'static) -> Self {
        self.on_event = Arc::new(on_event);
        self
    }

    fn with_cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }

    fn auth_url(&self) -> Option<url::Url> {
        self.events.lock().unwrap().iter().find_map(|event| match event {
            AuthEvent::AuthUrl { url, .. } => Some(url::Url::parse(url).unwrap()),
            _ => None,
        })
    }
}

impl AuthInteraction for TestInteraction {
    fn prompt(&self, prompt: AuthPrompt) -> BoxFuture<'_, Result<String, ModelsError>> {
        self.prompts.lock().unwrap().push(prompt.clone());
        (self.prompt)(prompt)
    }

    fn notify(&self, event: AuthEvent) {
        (self.on_event)(&event);
        self.events.lock().unwrap().push(event);
    }

    fn cancel(&self) -> CancellationToken {
        self.cancel.clone()
    }
}

fn ok(text: impl Into<String>) -> BoxFuture<'static, Result<String, ModelsError>> {
    let text = text.into();
    Box::pin(async move { Ok(text) })
}

/// A prompt that never answers (until its cancel token fires, for manual-code prompts).
fn never(prompt: AuthPrompt) -> BoxFuture<'static, Result<String, ModelsError>> {
    Box::pin(async move {
        if let AuthPrompt::ManualCode { cancel, .. } = prompt {
            cancel.cancelled().await;
            return Err(ModelsError::new(ModelsErrorCode::Aborted, "aborted"));
        }
        std::future::pending().await
    })
}

fn cancelled_prompt(_: AuthPrompt) -> BoxFuture<'static, Result<String, ModelsError>> {
    Box::pin(async { Err(ModelsError::new(ModelsErrorCode::Aborted, "Login cancelled")) })
}

fn param(url: &url::Url, name: &str) -> Option<String> {
    url.query_pairs()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.into_owned())
}

fn json_response(body: Value, status: u16) -> ResponseTemplate {
    ResponseTemplate::new(status).set_body_json(body)
}

fn json_body(request: &Request) -> Value {
    serde_json::from_slice(&request.body).unwrap()
}

fn form_body(request: &Request) -> Vec<(String, String)> {
    url::form_urlencoded::parse(&request.body).into_owned().collect()
}

fn form_field(form: &[(String, String)], name: &str) -> Option<String> {
    form.iter().find(|(key, _)| key == name).map(|(_, value)| value.clone())
}

fn credential(access: &str, refresh: &str, expires: i64) -> OAuthCredential {
    OAuthCredential {
        refresh: refresh.into(),
        access: access.into(),
        expires,
        extra: JsonObject::new(),
    }
}

async fn token_server(path_: &str, body: Value, status: u16) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(path_))
        .respond_with(json_response(body, status))
        .mount(&server)
        .await;
    server
}

async fn free_port() -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    listener.local_addr().unwrap().port()
}

/// The browser's request to the loopback callback, spawned from a notify handler.
type PageSlot = Arc<Mutex<Option<JoinHandle<reqwest::Result<(u16, String)>>>>>;

/// Spawn a GET of the loopback callback from a notify handler.
fn spawn_get(slot: &PageSlot, url: String) {
    *slot.lock().unwrap() = Some(tokio::spawn(async move {
        let response = http().get(url).send().await?;
        let status = response.status().as_u16();
        Ok((status, response.text().await?))
    }));
}

async fn take_page(slot: &PageSlot) -> (u16, String) {
    let handle = slot.lock().unwrap().take().unwrap();
    handle.await.unwrap().unwrap()
}

// ---------------------------------------------------------------------------
// OAuthAuth adapters (oauth-auth.test.ts)
// ---------------------------------------------------------------------------

#[test]
fn identifies_only_subscription_backed_oauth_flows_as_subscriptions() {
    assert!(AnthropicOAuth::default().is_subscription());
    assert!(XaiOAuth::default().is_subscription());
    assert!(!OpenRouterOAuth::default().is_subscription());
}

#[tokio::test]
async fn to_auth_derives_the_api_key_from_the_access_token() {
    let token = credential("token", "r", 0);
    for oauth in [
        Arc::new(AnthropicOAuth::default()) as Arc<dyn OAuthAuth>,
        Arc::new(XaiOAuth::default()),
        Arc::new(OpenRouterOAuth::default()),
    ] {
        let auth = oauth.to_auth(&token).await.unwrap();
        assert_eq!(auth.api_key.as_deref(), Some("token"));
        assert_eq!(auth.headers, None);
        assert_eq!(auth.base_url, None);
    }
}

#[tokio::test]
async fn openrouter_keeps_the_permanent_credential_on_refresh() {
    let stored = credential("token", "", PERMANENT_EXPIRY);
    let refreshed = OpenRouterOAuth::default().refresh(stored.clone()).await.unwrap();
    assert_eq!(refreshed, stored);
}

#[tokio::test]
async fn anthropic_refresh_exchanges_the_refresh_token_and_omits_scope() {
    let server = token_server(
        "/v1/oauth/token",
        json!({"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600}),
        200,
    )
    .await;
    let oauth = AnthropicOAuth {
        token_url: format!("{}/v1/oauth/token", server.uri()),
        ..AnthropicOAuth::default()
    };
    let refreshed = oauth.refresh(credential("old", "refresh-token", 0)).await.unwrap();
    assert_eq!(refreshed.access, "new-access");
    assert_eq!(refreshed.refresh, "new-refresh");
    assert!(refreshed.expires > now_ms());

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let body = json_body(&requests[0]);
    assert_eq!(body["grant_type"], "refresh_token");
    assert_eq!(body["client_id"], delta_ai::oauth::anthropic::CLIENT_ID);
    assert_eq!(body["refresh_token"], "refresh-token");
    assert!(body.get("scope").is_none());
}

#[tokio::test]
async fn anthropic_refresh_reports_http_failures() {
    let server = token_server("/v1/oauth/token", json!({"error": "invalid_grant"}), 400).await;
    let token_url = format!("{}/v1/oauth/token", server.uri());
    let oauth = AnthropicOAuth {
        token_url: token_url.clone(),
        ..AnthropicOAuth::default()
    };
    let error = oauth.refresh(credential("old", "r", 0)).await.unwrap_err();
    assert_eq!(
        error.message(),
        format!(
            "Anthropic token refresh request failed. url={token_url}; details=HTTP request failed. status=400; url={token_url}; body={{\"error\":\"invalid_grant\"}}"
        )
    );
}

#[test]
fn providers_expose_oauth_alongside_api_key_auth() {
    let anthropic = anthropic_provider();
    assert!(anthropic.auth().api_key.is_some());
    assert_eq!(
        anthropic.auth().oauth.as_ref().unwrap().name(),
        "Anthropic (Claude Pro/Max)"
    );

    let openrouter = builtin_provider("openrouter").unwrap();
    assert!(openrouter.auth().api_key.is_some());
    assert_eq!(
        openrouter.auth().oauth.as_ref().unwrap().login_label(),
        Some("Sign in with OpenRouter")
    );

    let xai = builtin_provider("xai").unwrap();
    assert!(xai.auth().api_key.is_some());
    let oauth = xai.auth().oauth.as_ref().unwrap();
    assert_eq!(oauth.name(), "xAI (Grok/X subscription)");
    assert!(oauth.is_subscription());
    assert_eq!(oauth.login_label(), Some("Sign in with SuperGrok or X Premium"));

    assert!(builtin_provider("groq").unwrap().auth().oauth.is_none());
}

async fn models_with(provider_id: &str, stored: OAuthCredential) -> Models {
    let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
    credentials
        .modify(
            provider_id,
            Box::new(move |_| Box::pin(async move { Ok(Some(Credential::OAuth(stored))) })),
        )
        .await
        .unwrap();
    let models = Models::new(CreateModelsOptions {
        credentials: Some(credentials),
        ..CreateModelsOptions::default()
    });
    models.set_provider_arc(builtin_provider(provider_id).unwrap());
    models
}

#[tokio::test]
async fn resolves_stored_anthropic_oauth_credentials_through_models() {
    // Keep this beyond get_auth()'s refresh window.
    let models = models_with(
        "anthropic",
        credential("oauth-access-token", "r", now_ms() + 10 * 60_000),
    )
    .await;
    let result = models
        .get_auth("anthropic", &AuthResolutionOverrides::default())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.auth.api_key.as_deref(), Some("oauth-access-token"));
    assert_eq!(result.source.as_deref(), Some("OAuth"));
}

#[tokio::test]
async fn resolves_the_same_stored_openrouter_key_for_chat_and_image_models() {
    let models = models_with("openrouter", credential("sk-or-stored", "", PERMANENT_EXPIRY)).await;
    let chat = models.get_models(Some("openrouter"));
    assert!(!chat.is_empty());
    let all = models.get_all_models(Some("openrouter"));
    let image = all
        .iter()
        .find(|model| model.model_type() == delta_ai::ModelType::Image)
        .unwrap();
    for model in [&delta_ai::AnyModel::Chat(chat[0].clone()), image] {
        let result = models
            .get_model_auth(model, &AuthResolutionOverrides::default())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.auth.api_key.as_deref(), Some("sk-or-stored"));
    }
}

// ---------------------------------------------------------------------------
// Anthropic OAuth (anthropic-oauth.test.ts)
// ---------------------------------------------------------------------------

const ANTHROPIC_TOKEN_PATH: &str = "/v1/oauth/token";

async fn anthropic_server() -> (MockServer, AnthropicOAuth) {
    let server = token_server(
        ANTHROPIC_TOKEN_PATH,
        json!({"access_token": "access-token", "refresh_token": "refresh-token", "expires_in": 3600}),
        200,
    )
    .await;
    let oauth = AnthropicOAuth {
        token_url: format!("{}{ANTHROPIC_TOKEN_PATH}", server.uri()),
        callback_host: Some("127.0.0.1".into()),
        callback_port: free_port().await,
        ..AnthropicOAuth::default()
    };
    (server, oauth)
}

#[tokio::test]
async fn anthropic_keeps_the_localhost_redirect_uri_for_manual_callback_login() {
    let (server, oauth) = anthropic_server().await;
    let auth_url: Arc<Mutex<Option<url::Url>>> = Arc::default();
    let seen = auth_url.clone();
    let interaction = TestInteraction::new(move |prompt| match prompt {
        AuthPrompt::Select { .. } => ok("browser"),
        AuthPrompt::ManualCode { .. } => {
            let url = seen.lock().unwrap().clone().unwrap();
            let state = param(&url, "state").unwrap();
            let redirect_uri = param(&url, "redirect_uri").unwrap();
            ok(format!("{redirect_uri}?code=manual-code&state={state}"))
        }
        other => panic!("Unexpected prompt: {other:?}"),
    });
    let slot = auth_url.clone();
    let interaction = interaction.on_event(move |event| {
        if let AuthEvent::AuthUrl { url, .. } = event {
            *slot.lock().unwrap() = Some(url::Url::parse(url).unwrap());
        }
    });

    let credential = oauth.login(&interaction, &LoginOptions::default()).await.unwrap();
    assert_eq!(credential.access, "access-token");
    assert_eq!(credential.refresh, "refresh-token");

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let body = json_body(&requests[0]);
    assert_eq!(body["grant_type"], "authorization_code");
    assert_eq!(body["code"], "manual-code");
    assert_eq!(
        body["redirect_uri"],
        format!("http://localhost:{}/callback", oauth.callback_port)
    );
}

#[tokio::test]
async fn anthropic_offers_browser_login_first_and_uses_the_selected_copy_code_flow() {
    let (server, oauth) = anthropic_server().await;
    let auth_url: Arc<Mutex<Option<url::Url>>> = Arc::default();
    let seen = auth_url.clone();
    let interaction = TestInteraction::new(move |prompt| match prompt {
        AuthPrompt::Select { .. } => ok("copy_code"),
        AuthPrompt::ManualCode { .. } => {
            let url = seen.lock().unwrap().clone().unwrap();
            ok(format!("copied-code#{}", param(&url, "state").unwrap()))
        }
        other => panic!("Unexpected prompt: {other:?}"),
    });
    let slot = auth_url.clone();
    let interaction = interaction.on_event(move |event| {
        if let AuthEvent::AuthUrl { url, .. } = event {
            *slot.lock().unwrap() = Some(url::Url::parse(url).unwrap());
        }
    });

    let credential = oauth.login(&interaction, &LoginOptions::default()).await.unwrap();
    assert_eq!(credential.access, "access-token");
    assert_eq!(credential.refresh, "refresh-token");

    let url = auth_url.lock().unwrap().clone().unwrap();
    assert_eq!(
        param(&url, "redirect_uri").as_deref(),
        Some("https://platform.claude.com/oauth/code/callback")
    );
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let body = json_body(&requests[0]);
    assert_eq!(body["grant_type"], "authorization_code");
    assert_eq!(body["code"], "copied-code");
    assert_eq!(body["state"], param(&url, "state").unwrap());
    assert_eq!(body["redirect_uri"], "https://platform.claude.com/oauth/code/callback");

    let prompts = interaction.prompts.lock().unwrap();
    let AuthPrompt::Select { message, options } = &prompts[0] else {
        panic!("expected a select prompt first");
    };
    assert_eq!(message, "Select Anthropic login method:");
    assert_eq!(
        options,
        &[
            AuthSelectOption {
                id: "browser".into(),
                label: "Browser login (default)".into(),
                description: None,
            },
            AuthSelectOption {
                id: "copy_code".into(),
                label: "Copy code login (headless)".into(),
                description: None,
            },
        ]
    );
    assert_eq!(
        prompts
            .iter()
            .filter(|p| matches!(p, AuthPrompt::Select { .. }))
            .count(),
        1
    );
}

#[tokio::test]
async fn anthropic_cancels_when_login_method_selection_is_cancelled() {
    let interaction = TestInteraction::new(cancelled_prompt);
    let error = AnthropicOAuth::default()
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.message(), "Login cancelled");
}

#[tokio::test]
async fn anthropic_resolves_through_the_manual_code_prompt_and_cancels_it_after_settling() {
    let (_server, oauth) = anthropic_server().await;
    let manual: Arc<Mutex<Option<CancellationToken>>> = Arc::default();
    let seen = manual.clone();
    let interaction = TestInteraction::new(move |prompt| match prompt {
        AuthPrompt::Select { .. } => ok("browser"),
        AuthPrompt::ManualCode { cancel, .. } => {
            *seen.lock().unwrap() = Some(cancel);
            ok("the-code")
        }
        other => panic!("Unexpected prompt: {other:?}"),
    });

    let credential = oauth.login(&interaction, &LoginOptions::default()).await.unwrap();
    assert_eq!(credential.access, "access-token");
    assert!(interaction.auth_url().is_some());
    // The prompt's token fires once login settles, so UIs can dismiss it.
    assert!(manual.lock().unwrap().as_ref().unwrap().is_cancelled());
}

struct BrowserLogin {
    credential: OAuthCredential,
    redirect_uri: String,
    exchanged_code: String,
    exchanged_redirect_uri: String,
    page_status: u16,
    page_text: String,
}

async fn anthropic_login_through_browser_callback(callback_port: u16) -> BrowserLogin {
    let (server, mut oauth) = anthropic_server().await;
    oauth.callback_port = callback_port;
    let page: PageSlot = Arc::default();
    let slot = page.clone();
    let interaction = TestInteraction::new(|prompt| match prompt {
        AuthPrompt::Select { .. } => ok("browser"),
        other => never(other),
    })
    .on_event(move |event| {
        let AuthEvent::AuthUrl { url, .. } = event else { return };
        let url = url::Url::parse(url).unwrap();
        let mut callback = url::Url::parse(&param(&url, "redirect_uri").unwrap()).unwrap();
        callback.set_host(Some("127.0.0.1")).unwrap();
        callback
            .query_pairs_mut()
            .append_pair("code", "browser-code")
            .append_pair("state", &param(&url, "state").unwrap());
        spawn_get(&slot, callback.to_string());
    });

    let credential = oauth.login(&interaction, &LoginOptions::default()).await.unwrap();
    let (page_status, page_text) = take_page(&page).await;
    let redirect_uri = param(&interaction.auth_url().unwrap(), "redirect_uri").unwrap();
    let body = json_body(&server.received_requests().await.unwrap()[0]);
    BrowserLogin {
        credential,
        redirect_uri,
        exchanged_code: body["code"].as_str().unwrap().to_owned(),
        exchanged_redirect_uri: body["redirect_uri"].as_str().unwrap().to_owned(),
        page_status,
        page_text,
    }
}

#[tokio::test]
async fn anthropic_completes_login_through_the_browser_callback_and_shows_the_sign_in_page() {
    let login = anthropic_login_through_browser_callback(free_port().await).await;
    assert_eq!(login.credential.access, "access-token");
    assert_eq!(login.exchanged_code, "browser-code");
    assert_eq!(login.exchanged_redirect_uri, login.redirect_uri);
    assert_eq!(login.page_status, 200);
    assert!(login.page_text.contains("Signed in to Anthropic."));
}

// Upstream #10571.
#[tokio::test]
async fn anthropic_falls_back_to_a_free_callback_port_when_the_preferred_port_cannot_be_bound() {
    let blocker = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let blocked = blocker.local_addr().unwrap().port();
    let login = anthropic_login_through_browser_callback(blocked).await;
    let redirect = url::Url::parse(&login.redirect_uri).unwrap();
    assert_eq!(redirect.host_str(), Some("localhost"));
    assert_eq!(redirect.path(), "/callback");
    assert_ne!(redirect.port(), Some(blocked));
    assert_eq!(login.credential.access, "access-token");
    assert_eq!(login.exchanged_redirect_uri, login.redirect_uri);
    assert_eq!(login.page_status, 200);
}

#[tokio::test]
async fn anthropic_rejects_a_pasted_state_mismatch() {
    let (server, oauth) = anthropic_server().await;
    let interaction = TestInteraction::new(|prompt| match prompt {
        AuthPrompt::Select { .. } => ok("copy_code"),
        _ => ok("code#other-state"),
    });
    let error = oauth.login(&interaction, &LoginOptions::default()).await.unwrap_err();
    assert_eq!(error.message(), "OAuth state mismatch");
    assert!(server.received_requests().await.unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// OpenRouter OAuth (openrouter-oauth.test.ts)
// ---------------------------------------------------------------------------

const OPENROUTER_KEYS_PATH: &str = "/api/v1/auth/keys";

fn openrouter(server: &MockServer) -> OpenRouterOAuth {
    OpenRouterOAuth {
        token_url: format!("{}{OPENROUTER_KEYS_PATH}", server.uri()),
        callback_host: Some("127.0.0.1".into()),
        ..OpenRouterOAuth::default()
    }
}

/// Answers the browser callback with `code` as soon as the authorize URL is shown.
fn openrouter_browser(code: &'static str) -> (TestInteraction, PageSlot) {
    let page: PageSlot = Arc::default();
    let slot = page.clone();
    let interaction = TestInteraction::new(never).on_event(move |event| {
        let AuthEvent::AuthUrl { url, .. } = event else { return };
        let url = url::Url::parse(url).unwrap();
        let mut callback = url::Url::parse(&param(&url, "callback_url").unwrap()).unwrap();
        callback.query_pairs_mut().append_pair("code", code);
        spawn_get(&slot, callback.to_string());
    });
    (interaction, page)
}

#[tokio::test]
async fn openrouter_runs_pkce_on_a_one_shot_loopback_callback_and_mints_a_permanent_key() {
    let server = token_server(OPENROUTER_KEYS_PATH, json!({"key": "sk-or-test"}), 200).await;
    let (interaction, page) = openrouter_browser("authorization-code");
    let manual: Arc<Mutex<Option<CancellationToken>>> = Arc::default();
    let seen = manual.clone();
    let interaction = TestInteraction {
        prompt: Arc::new(move |prompt| {
            if let AuthPrompt::ManualCode { cancel, .. } = &prompt {
                *seen.lock().unwrap() = Some(cancel.clone());
            }
            never(prompt)
        }),
        ..interaction
    };

    let credential = openrouter(&server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap();
    assert_eq!(credential, credential_of("sk-or-test"));
    assert_eq!(take_page(&page).await.0, 200);
    assert!(manual.lock().unwrap().as_ref().unwrap().is_cancelled());

    let authorize = interaction.auth_url().unwrap();
    assert_eq!(authorize.origin().ascii_serialization(), "https://openrouter.ai");
    assert_eq!(authorize.path(), "/auth");
    assert_eq!(param(&authorize, "code_challenge_method").as_deref(), Some("S256"));
    let callback = url::Url::parse(&param(&authorize, "callback_url").unwrap()).unwrap();
    assert_eq!(callback.host_str(), Some("127.0.0.1"));
    let id = callback.path().strip_prefix("/oauth/callback/").unwrap();
    assert!(!id.is_empty() && id.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let body = json_body(&requests[0]);
    assert_eq!(body["code"], "authorization-code");
    assert_eq!(body["code_challenge_method"], "S256");
    let verifier = body["code_verifier"].as_str().unwrap();
    assert_eq!(
        param(&authorize, "code_challenge").unwrap(),
        pkce::pkce_challenge(verifier)
    );
    let events = interaction.events.lock().unwrap();
    assert!(matches!(
        &events[0],
        AuthEvent::Progress { message } if message.starts_with("Listening for OpenRouter OAuth callback on http://127.0.0.1:")
    ));
}

fn credential_of(key: &str) -> OAuthCredential {
    credential(key, "", PERMANENT_EXPIRY)
}

#[tokio::test]
async fn openrouter_reports_token_exchange_failures_through_both_the_callback_page_and_login() {
    let server = token_server(OPENROUTER_KEYS_PATH, json!({"error": {"message": "invalid code"}}), 403).await;
    let (interaction, page) = openrouter_browser("bad-code");
    let error = openrouter(&server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        error.message(),
        "OpenRouter OAuth key exchange failed (HTTP 403): invalid code"
    );
    assert_eq!(take_page(&page).await.0, 502);
}

#[tokio::test]
async fn openrouter_allows_only_one_token_exchange_for_a_callback() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(OPENROUTER_KEYS_PATH))
        .respond_with(json_response(json!({"key": "sk-or-test"}), 200).set_delay(Duration::from_millis(500)))
        .mount(&server)
        .await;
    let callback: Arc<Mutex<Option<String>>> = Arc::default();
    let (interaction, page) = openrouter_browser("authorization-code");
    let slot = callback.clone();
    let previous = interaction.on_event.clone();
    let interaction = interaction.on_event(move |event| {
        if let AuthEvent::AuthUrl { url, .. } = event {
            let url = url::Url::parse(url).unwrap();
            let mut callback = url::Url::parse(&param(&url, "callback_url").unwrap()).unwrap();
            callback.query_pairs_mut().append_pair("code", "authorization-code");
            *slot.lock().unwrap() = Some(callback.to_string());
        }
        previous(event);
    });
    let oauth = openrouter(&server);
    let login_options = LoginOptions::default();
    let login = oauth.login(&interaction, &login_options);
    let second = async {
        // Wait until the first callback started the exchange.
        loop {
            if server.received_requests().await.unwrap().len() == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let url = callback.lock().unwrap().clone().unwrap();
        http().get(url).send().await.unwrap().status().as_u16()
    };
    let (credential, second) = tokio::join!(login, second);
    assert_eq!(second, 409);
    assert_eq!(credential.unwrap().access, "sk-or-test");
    assert_eq!(take_page(&page).await.0, 200);
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn openrouter_rejects_a_successful_response_that_does_not_contain_a_key() {
    let server = token_server(OPENROUTER_KEYS_PATH, json!({"user_id": "user-1"}), 200).await;
    let (interaction, page) = openrouter_browser("code-without-key");
    let error = openrouter(&server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.message(), "OpenRouter OAuth response carries no \"key\"");
    assert_eq!(take_page(&page).await.0, 502);
}

#[tokio::test]
async fn openrouter_mints_a_key_from_a_pasted_redirect_url_when_the_callback_never_arrives() {
    let server = token_server(OPENROUTER_KEYS_PATH, json!({"key": "sk-or-manual"}), 200).await;
    let callback_url: Arc<Mutex<Option<String>>> = Arc::default();
    let seen = callback_url.clone();
    let slot = callback_url.clone();
    let interaction = TestInteraction::new(move |prompt| {
        assert!(
            matches!(prompt, AuthPrompt::ManualCode { .. }),
            "Unexpected prompt: {prompt:?}"
        );
        ok(format!("{}?code=manual-code", seen.lock().unwrap().clone().unwrap()))
    })
    .on_event(move |event| {
        if let AuthEvent::AuthUrl { url, .. } = event {
            *slot.lock().unwrap() = param(&url::Url::parse(url).unwrap(), "callback_url");
        }
    });
    let credential = openrouter(&server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap();
    assert_eq!(credential, credential_of("sk-or-manual"));
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let body = json_body(&requests[0]);
    assert_eq!(body["code"], "manual-code");
    assert_eq!(body["code_challenge_method"], "S256");
}

#[tokio::test]
async fn openrouter_accepts_a_bare_authorization_code_from_the_manual_prompt() {
    let server = token_server(OPENROUTER_KEYS_PATH, json!({"key": "sk-or-manual"}), 200).await;
    let interaction = TestInteraction::new(|_| ok("  manual-code  "));
    let credential = openrouter(&server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap();
    assert_eq!(credential.access, "sk-or-manual");
    assert_eq!(
        json_body(&server.received_requests().await.unwrap()[0])["code"],
        "manual-code"
    );
}

#[tokio::test]
async fn openrouter_fails_login_when_the_manual_prompt_is_cancelled() {
    let server = token_server(OPENROUTER_KEYS_PATH, json!({"key": "sk-or-unexpected"}), 200).await;
    let interaction = TestInteraction::new(cancelled_prompt);
    let error = openrouter(&server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.message(), "Login cancelled");
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn openrouter_rejects_empty_manual_input_without_exchanging_a_code() {
    let server = token_server(OPENROUTER_KEYS_PATH, json!({"key": "sk-or-unexpected"}), 200).await;
    let interaction = TestInteraction::new(|_| ok("   "));
    let error = openrouter(&server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.message(), "Missing authorization code");
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn openrouter_closes_the_pending_callback_when_login_is_cancelled() {
    let cancel = CancellationToken::new();
    let callback: Arc<Mutex<Option<String>>> = Arc::default();
    let slot = callback.clone();
    let token = cancel.clone();
    let interaction = TestInteraction::new(never).with_cancel(cancel).on_event(move |event| {
        if let AuthEvent::AuthUrl { url, .. } = event {
            *slot.lock().unwrap() = param(&url::Url::parse(url).unwrap(), "callback_url");
            token.cancel();
        }
    });
    let error = OpenRouterOAuth {
        callback_host: Some("127.0.0.1".into()),
        ..OpenRouterOAuth::default()
    }
    .login(&interaction, &LoginOptions::default())
    .await
    .unwrap_err();
    assert_eq!(error.message(), "Login cancelled");
    let url = callback.lock().unwrap().clone().unwrap();
    assert!(http().get(url).send().await.is_err());
}

#[tokio::test]
async fn openrouter_rejects_before_opening_a_callback_server_when_login_is_already_cancelled() {
    let cancel = CancellationToken::new();
    cancel.cancel();
    let interaction = TestInteraction::new(|_| ok(""))
        .with_cancel(cancel)
        .on_event(|_| panic!("Cancelled login must not emit events"));
    let error = OpenRouterOAuth::default()
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.message(), "Login cancelled");
}

#[tokio::test]
async fn openrouter_uses_the_configured_oauth_callback_host() {
    // Upstream sets PI_OAUTH_CALLBACK_HOST; delta's `callback_host` field overrides it the
    // same way without mutating the process environment.
    let cancel = CancellationToken::new();
    let callback: Arc<Mutex<Option<String>>> = Arc::default();
    let slot = callback.clone();
    let token = cancel.clone();
    let interaction = TestInteraction::new(never).with_cancel(cancel).on_event(move |event| {
        if let AuthEvent::AuthUrl { url, .. } = event {
            *slot.lock().unwrap() = param(&url::Url::parse(url).unwrap(), "callback_url");
            token.cancel();
        }
    });
    let error = OpenRouterOAuth {
        callback_host: Some("localhost".into()),
        ..OpenRouterOAuth::default()
    }
    .login(&interaction, &LoginOptions::default())
    .await
    .unwrap_err();
    assert_eq!(error.message(), "Login cancelled");
    let url = url::Url::parse(&callback.lock().unwrap().clone().unwrap()).unwrap();
    assert_eq!(url.host_str(), Some("localhost"));
}

// ---------------------------------------------------------------------------
// xAI OAuth device flow (xai-oauth.test.ts)
// ---------------------------------------------------------------------------

const XAI_DEVICE_PATH: &str = "/oauth2/device/code";
const XAI_TOKEN_PATH: &str = "/oauth2/token";

fn device_code_response(overrides: Value) -> Value {
    let mut body = json!({
        "device_code": "device-code",
        "user_code": "ABCD-1234",
        "verification_uri": "https://accounts.x.ai/oauth2/device",
        "expires_in": 900,
        "interval": 1,
    });
    merge(&mut body, overrides);
    body
}

fn token_response(overrides: Value) -> Value {
    let mut body = json!({
        "access_token": "access-token",
        "refresh_token": "refresh-token",
        "expires_in": 21_600,
        "token_type": "Bearer",
    });
    merge(&mut body, overrides);
    body
}

/// Merge `overrides` into `body`; `null` removes a field (upstream's `undefined`).
fn merge(body: &mut Value, overrides: Value) {
    let Value::Object(overrides) = overrides else { return };
    let body = body.as_object_mut().unwrap();
    for (key, value) in overrides {
        if value.is_null() {
            body.remove(&key);
        } else {
            body.insert(key, value);
        }
    }
}

/// Token endpoint that replies from a queue and records when each poll arrived.
struct TokenReplies {
    replies: Mutex<Vec<ResponseTemplate>>,
    times: Arc<Mutex<Vec<Instant>>>,
}

impl Respond for TokenReplies {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let form = form_body(request);
        assert_eq!(
            form_field(&form, "grant_type").as_deref(),
            Some("urn:ietf:params:oauth:grant-type:device_code")
        );
        assert_eq!(
            form_field(&form, "client_id").as_deref(),
            Some(delta_ai::oauth::xai::CLIENT_ID)
        );
        assert_eq!(form_field(&form, "device_code").as_deref(), Some("device-code"));
        self.times.lock().unwrap().push(Instant::now());
        let mut replies = self.replies.lock().unwrap();
        assert!(!replies.is_empty(), "Unexpected token poll");
        replies.remove(0)
    }
}

async fn xai_server(
    device: ResponseTemplate,
    replies: Vec<ResponseTemplate>,
) -> (MockServer, Arc<Mutex<Vec<Instant>>>) {
    let server = MockServer::start().await;
    let times = Arc::new(Mutex::new(Vec::new()));
    Mock::given(method("POST"))
        .and(path(XAI_DEVICE_PATH))
        .respond_with(device)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(XAI_TOKEN_PATH))
        .respond_with(TokenReplies {
            replies: Mutex::new(replies),
            times: times.clone(),
        })
        .mount(&server)
        .await;
    (server, times)
}

fn xai(server: &MockServer) -> XaiOAuth {
    XaiOAuth {
        device_code_url: format!("{}{XAI_DEVICE_PATH}", server.uri()),
        token_url: format!("{}{XAI_TOKEN_PATH}", server.uri()),
    }
}

fn device_codes(interaction: &TestInteraction) -> Vec<AuthEvent> {
    interaction
        .events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| matches!(event, AuthEvent::DeviceCode { .. }))
        .cloned()
        .collect()
}

fn no_prompt(_: AuthPrompt) -> BoxFuture<'static, Result<String, ModelsError>> {
    panic!("Unexpected prompt")
}

fn assert_gap(later: Instant, earlier: Instant, at_least: Duration) {
    let gap = later.duration_since(earlier);
    assert!(
        gap + Duration::from_millis(20) >= at_least,
        "gap {gap:?} < {at_least:?}"
    );
}

#[tokio::test]
async fn xai_uses_the_device_grant_delays_polling_and_handles_pending_and_slow_down() {
    let (server, times) = xai_server(
        json_response(device_code_response(json!({})), 200),
        vec![
            json_response(json!({"error": "authorization_pending"}), 400),
            json_response(json!({"error": "slow_down", "interval": 2}), 400),
            json_response(token_response(json!({})), 200),
        ],
    )
    .await;
    let interaction = TestInteraction::new(no_prompt);
    let start = Instant::now();
    let before = now_ms();
    let credential = xai(&server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap();
    let after = now_ms();

    let device = server.received_requests().await.unwrap();
    let form = form_body(&device[0]);
    assert_eq!(
        form_field(&form, "client_id").as_deref(),
        Some(delta_ai::oauth::xai::CLIENT_ID)
    );
    assert_eq!(
        form_field(&form, "scope").as_deref(),
        Some("openid profile email offline_access grok-cli:access api:access")
    );
    assert_eq!(form_field(&form, "referrer").as_deref(), Some("pi"));
    assert_eq!(
        device_codes(&interaction),
        [AuthEvent::DeviceCode {
            user_code: "ABCD-1234".into(),
            verification_uri: "https://accounts.x.ai/oauth2/device".into(),
            interval_seconds: Some(1),
            expires_in_seconds: Some(900),
        }]
    );

    let times = times.lock().unwrap().clone();
    assert_eq!(times.len(), 3);
    // waitBeforeFirstPoll: one interval before the first poll.
    assert_gap(times[0], start, Duration::from_secs(1));
    assert_gap(times[1], times[0], Duration::from_secs(1));
    // slow_down raised the interval to the server's 2 seconds.
    assert_gap(times[2], times[1], Duration::from_secs(2));

    assert_eq!(credential.access, "access-token");
    assert_eq!(credential.refresh, "refresh-token");
    assert!(credential.expires >= before + 21_600_000 - 300_000);
    assert!(credential.expires <= after + 21_600_000 - 300_000);
}

#[tokio::test]
async fn xai_falls_back_to_the_default_poll_interval_when_the_response_reports_interval_0() {
    let (server, times) = xai_server(
        json_response(device_code_response(json!({"interval": 0})), 200),
        vec![json_response(token_response(json!({})), 200)],
    )
    .await;
    let start = Instant::now();
    xai(&server)
        .login(&TestInteraction::new(no_prompt), &LoginOptions::default())
        .await
        .unwrap();
    // RFC 8628 default interval is 5 seconds when the server does not require a wait.
    assert_gap(times.lock().unwrap()[0], start, Duration::from_secs(5));
}

#[tokio::test]
async fn xai_prefers_verification_uri_complete_when_the_server_provides_it() {
    let (server, _) = xai_server(
        json_response(
            device_code_response(json!({
                "verification_uri_complete": "https://accounts.x.ai/oauth2/device?user_code=ABCD-1234"
            })),
            200,
        ),
        vec![json_response(token_response(json!({})), 200)],
    )
    .await;
    let interaction = TestInteraction::new(no_prompt);
    xai(&server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap();
    assert_eq!(
        device_codes(&interaction),
        [AuthEvent::DeviceCode {
            user_code: "ABCD-1234".into(),
            verification_uri: "https://accounts.x.ai/oauth2/device?user_code=ABCD-1234".into(),
            interval_seconds: Some(1),
            expires_in_seconds: Some(900),
        }]
    );
}

#[tokio::test]
async fn xai_rejects_a_non_https_verification_uri_complete() {
    let (server, _) = xai_server(
        json_response(
            device_code_response(json!({
                "verification_uri_complete": "http://accounts.x.ai/oauth2/device?user_code=ABCD-1234"
            })),
            200,
        ),
        vec![],
    )
    .await;
    let error = xai(&server)
        .login(&TestInteraction::new(no_prompt), &LoginOptions::default())
        .await
        .unwrap_err();
    assert!(error.message().contains("Untrusted verification URI"));
}

#[tokio::test]
async fn xai_rejects_non_https_verification_uris() {
    for uri in ["http://accounts.x.ai/oauth2/device", "file:///etc/passwd", "not a url"] {
        let (server, _) = xai_server(
            json_response(device_code_response(json!({"verification_uri": uri})), 200),
            vec![],
        )
        .await;
        let error = xai(&server)
            .login(&TestInteraction::new(no_prompt), &LoginOptions::default())
            .await
            .unwrap_err();
        assert!(error.message().contains("Untrusted verification URI"), "{uri}");
    }
}

#[tokio::test]
async fn xai_fails_when_device_authorization_is_denied() {
    for error_code in ["access_denied", "authorization_denied"] {
        let (server, _) = xai_server(
            json_response(device_code_response(json!({})), 200),
            vec![json_response(json!({"error": error_code}), 400)],
        )
        .await;
        let error = xai(&server)
            .login(&TestInteraction::new(no_prompt), &LoginOptions::default())
            .await
            .unwrap_err();
        assert_eq!(error.message(), "xAI device authorization was denied", "{error_code}");
    }
}

#[tokio::test]
async fn xai_cancels_while_waiting_for_the_first_token_poll() {
    let (server, times) = xai_server(json_response(device_code_response(json!({})), 200), vec![]).await;
    let cancel = CancellationToken::new();
    let token = cancel.clone();
    let interaction = TestInteraction::new(no_prompt)
        .with_cancel(cancel)
        .on_event(move |event| {
            if matches!(event, AuthEvent::DeviceCode { .. }) {
                token.cancel();
            }
        });
    let error = xai(&server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.message(), "Login cancelled");
    assert_eq!(error.code(), ModelsErrorCode::Aborted);
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
    assert!(times.lock().unwrap().is_empty());
}

async fn refresh_server(reply: ResponseTemplate) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(XAI_TOKEN_PATH))
        .respond_with(reply)
        .mount(&server)
        .await;
    server
}

#[tokio::test]
async fn xai_refreshes_tokens_and_preserves_an_unrotated_refresh_token() {
    let rotated_server = refresh_server(json_response(
        token_response(json!({"access_token": "new-access", "refresh_token": "new-refresh"})),
        200,
    ))
    .await;
    let rotated = xai(&rotated_server)
        .refresh(credential("old-access", "old-refresh", 0))
        .await
        .unwrap();
    let form = form_body(&rotated_server.received_requests().await.unwrap()[0]);
    assert_eq!(form_field(&form, "grant_type").as_deref(), Some("refresh_token"));
    assert_eq!(
        form_field(&form, "client_id").as_deref(),
        Some(delta_ai::oauth::xai::CLIENT_ID)
    );
    assert_eq!(form_field(&form, "refresh_token").as_deref(), Some("old-refresh"));
    assert_eq!(rotated.refresh, "new-refresh");
    assert_eq!(rotated.access, "new-access");

    let preserved_server = refresh_server(json_response(
        token_response(json!({"access_token": "newer-access", "refresh_token": null})),
        200,
    ))
    .await;
    let oauth = xai(&preserved_server);
    let preserved = oauth
        .refresh(credential("old-access", "keep-refresh", 0))
        .await
        .unwrap();
    let form = form_body(&preserved_server.received_requests().await.unwrap()[0]);
    assert_eq!(form_field(&form, "refresh_token").as_deref(), Some("keep-refresh"));
    assert_eq!(preserved.refresh, "keep-refresh");
    assert_eq!(preserved.access, "newer-access");
    assert_eq!(oauth.name(), "xAI (Grok/X subscription)");
    assert_eq!(
        oauth.to_auth(&preserved).await.unwrap().api_key.as_deref(),
        Some("newer-access")
    );
}

#[tokio::test]
async fn xai_assumes_a_one_hour_lifetime_when_expires_in_is_missing() {
    let server = refresh_server(json_response(token_response(json!({"expires_in": null})), 200)).await;
    let before = now_ms();
    let refreshed = xai(&server).refresh(credential("a", "old-refresh", 0)).await.unwrap();
    let after = now_ms();
    assert!(refreshed.expires >= before + 3_600_000 - 300_000);
    assert!(refreshed.expires <= after + 3_600_000 - 300_000);
}

#[tokio::test]
async fn xai_rejects_token_responses_with_missing_fields() {
    let server = refresh_server(json_response(token_response(json!({"access_token": null})), 200)).await;
    let error = xai(&server)
        .refresh(credential("a", "old-refresh", 0))
        .await
        .unwrap_err();
    assert_eq!(error.message(), "Invalid xAI OAuth response field: access_token");
}

#[tokio::test]
async fn xai_surfaces_the_upstream_error_code_and_description_on_refresh_failure() {
    let server = refresh_server(json_response(
        json!({"error": "invalid_grant", "error_description": "refresh token revoked"}),
        400,
    ))
    .await;
    let error = xai(&server)
        .refresh(credential("a", "old-refresh", 0))
        .await
        .unwrap_err();
    assert_eq!(
        error.message(),
        "xAI OAuth token refresh failed (HTTP 400): invalid_grant: refresh token revoked"
    );
}
