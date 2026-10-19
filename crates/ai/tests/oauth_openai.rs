//! Tests of the OpenAI OAuth flows (OpenAI Codex, Sign in with ChatGPT) and their provider
//! wiring. Ported from upstream `test/openai-codex-oauth.test.ts`,
//! `test/openai-chatgpt-oauth.test.ts` and the OpenAI parts of `test/oauth-auth.test.ts`.
//!
//! Token endpoints are wiremock servers; browser callbacks are real loopback requests on
//! free ports. Upstream's device-code timing tests use fake timers; here the server sends
//! 1-second intervals and the test checks the gaps in real time, and the 15-minute timeout
//! runs on Tokio's paused clock.

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

use base64::Engine as _;
use delta_ai::auth::{AuthEvent, AuthInteraction, AuthPrompt, AuthSelectOption, LoginOptions, OAuthAuth};
use delta_ai::oauth::{OpenAIChatGPTOAuth, OpenAICodexOAuth};
use delta_ai::providers::builtin::{builtin_provider, openai_codex_provider};
use delta_ai::types::now_ms;
use delta_ai::{JsonObject, ModelsError, ModelsErrorCode, OAuthCredential, Provider};
use futures::future::BoxFuture;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The browser's request to the loopback callback, spawned from a notify handler.
type PageSlot = Arc<Mutex<Option<tokio::task::JoinHandle<(u16, String)>>>>;
type PageSlots = Arc<Mutex<Vec<tokio::task::JoinHandle<(u16, String)>>>>;

type PromptFn = Arc<dyn Fn(AuthPrompt) -> BoxFuture<'static, Result<String, ModelsError>> + Send + Sync>;

/// Records events and prompts; `prompt` scripts the user. `auth_url` holds the last
/// `AuthUrl` event so prompts can answer with its state.
struct TestInteraction {
    prompt: PromptFn,
    cancel: CancellationToken,
    events: Mutex<Vec<AuthEvent>>,
    prompts: Mutex<Vec<AuthPrompt>>,
    auth_url: Arc<Mutex<Option<url::Url>>>,
    on_event: Arc<dyn Fn(&AuthEvent) + Send + Sync>,
}

impl TestInteraction {
    fn new(
        prompt: impl Fn(AuthPrompt) -> BoxFuture<'static, Result<String, ModelsError>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            prompt: Arc::new(prompt),
            cancel: CancellationToken::new(),
            events: Mutex::new(Vec::new()),
            prompts: Mutex::new(Vec::new()),
            auth_url: Arc::new(Mutex::new(None)),
            on_event: Arc::new(|_| {}),
        }
    }

    /// A user who answers the method selection with `method` and pastes the redirect URL
    /// built by `paste` from the authorize URL.
    fn pasting(method: &'static str, paste: impl Fn(&url::Url) -> String + Send + Sync + 'static) -> Self {
        let auth_url: Arc<Mutex<Option<url::Url>>> = Arc::new(Mutex::new(None));
        let slot = auth_url.clone();
        let paste = Arc::new(paste);
        let mut interaction = Self::new(move |prompt| {
            let slot = slot.clone();
            let paste = paste.clone();
            Box::pin(async move {
                match prompt {
                    AuthPrompt::Select { .. } => Ok(method.to_owned()),
                    AuthPrompt::ManualCode { .. } => {
                        let url = slot.lock().unwrap().clone().expect("auth URL before the manual prompt");
                        Ok(paste(&url))
                    }
                    other => panic!("unexpected prompt: {other:?}"),
                }
            })
        });
        interaction.auth_url = auth_url;
        interaction
    }

    fn with_cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }

    fn on_event(mut self, on_event: impl Fn(&AuthEvent) + Send + Sync + 'static) -> Self {
        self.on_event = Arc::new(on_event);
        self
    }

    fn events(&self) -> Vec<AuthEvent> {
        self.events.lock().unwrap().clone()
    }
}

impl AuthInteraction for TestInteraction {
    fn prompt(&self, prompt: AuthPrompt) -> BoxFuture<'_, Result<String, ModelsError>> {
        self.prompts.lock().unwrap().push(prompt.clone());
        (self.prompt)(prompt)
    }

    fn notify(&self, event: AuthEvent) {
        if let AuthEvent::AuthUrl { url, .. } = &event {
            *self.auth_url.lock().unwrap() = Some(url::Url::parse(url).unwrap());
        }
        (self.on_event)(&event);
        self.events.lock().unwrap().push(event);
    }

    fn cancel(&self) -> CancellationToken {
        self.cancel.clone()
    }
}

fn select_device_code() -> TestInteraction {
    TestInteraction::new(|prompt| {
        Box::pin(async move {
            match prompt {
                AuthPrompt::Select { .. } => Ok("device_code".to_owned()),
                other => panic!("unexpected prompt: {other:?}"),
            }
        })
    })
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

async fn free_port() -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    listener.local_addr().unwrap().port()
}

fn http() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

fn access_token(account_id: &str) -> String {
    let encode = |value: Value| base64::engine::general_purpose::STANDARD.encode(value.to_string());
    format!(
        "{}.{}.signature",
        encode(json!({"alg": "none"})),
        encode(json!({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}))
    )
}

fn credential(access: &str, refresh: &str, expires: i64, extra: Value) -> OAuthCredential {
    OAuthCredential {
        refresh: refresh.into(),
        access: access.into(),
        expires,
        extra: match extra {
            Value::Object(extra) => extra,
            _ => JsonObject::new(),
        },
    }
}

// ---------------------------------------------------------------------------
// OAuthAuth adapters and provider wiring (oauth-auth.test.ts)
// ---------------------------------------------------------------------------

#[test]
fn identifies_the_openai_flows_as_subscriptions() {
    assert!(OpenAIChatGPTOAuth::default().is_subscription());
    assert!(OpenAICodexOAuth::default().is_subscription());
}

#[test]
fn openai_exposes_chatgpt_oauth_alongside_api_key_auth() {
    let provider = builtin_provider("openai").unwrap();
    let auth = provider.auth();
    assert_eq!(auth.api_key.as_ref().unwrap().name(), "OpenAI API key");
    let oauth = auth.oauth.as_ref().unwrap();
    assert_eq!(oauth.name(), "OpenAI (ChatGPT subscription)");
    assert!(oauth.is_subscription());
    assert_eq!(oauth.login_label(), Some("Sign in with ChatGPT"));
}

#[test]
fn openai_codex_is_built_in_with_oauth_only() {
    let provider = builtin_provider("openai-codex").unwrap();
    assert_eq!(provider.name(), "OpenAI Codex (legacy)");
    assert!(provider.auth().api_key.is_none());
    let oauth = provider.auth().oauth.clone().unwrap();
    assert_eq!(oauth.name(), "OpenAI (ChatGPT Plus/Pro)");
    assert_eq!(oauth.login_label(), None);
    let models = openai_codex_provider().models();
    assert!(models.iter().any(|model| model.id == "gpt-5.5"));
    assert!(
        models
            .iter()
            .all(|model| model.api == "openai-codex-responses" && model.base_url == "https://chatgpt.com/backend-api")
    );
}

#[tokio::test]
async fn openai_codex_to_auth_derives_the_api_key_from_the_access_token() {
    for oauth in [
        Arc::new(OpenAICodexOAuth::default()) as Arc<dyn OAuthAuth>,
        Arc::new(OpenAIChatGPTOAuth::default()),
    ] {
        let auth = oauth.to_auth(&credential("token", "r", 0, Value::Null)).await.unwrap();
        assert_eq!(auth.api_key.as_deref(), Some("token"));
        assert_eq!(auth.base_url, None);
        assert_eq!(auth.headers, None);
    }
}

// ---------------------------------------------------------------------------
// OpenAI Codex OAuth (openai-codex-oauth.test.ts)
// ---------------------------------------------------------------------------

const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

fn codex(server: &MockServer) -> OpenAICodexOAuth {
    OpenAICodexOAuth {
        authorize_url: format!("{}/oauth/authorize", server.uri()),
        token_url: format!("{}/oauth/token", server.uri()),
        device_user_code_url: format!("{}/api/accounts/deviceauth/usercode", server.uri()),
        device_token_url: format!("{}/api/accounts/deviceauth/token", server.uri()),
        callback_host: Some("127.0.0.1".into()),
        callback_port: 0,
    }
}

fn device_auth_pending() -> ResponseTemplate {
    json_response(
        json!({"error": {"message": "Device authorization is pending. Please try again.",
            "type": "invalid_request_error", "param": null, "code": "deviceauth_authorization_pending"}}),
        403,
    )
}

/// Answers the device token polls in order and records when each arrived.
struct PollResponses {
    responses: Mutex<Vec<ResponseTemplate>>,
    times: Arc<Mutex<Vec<Instant>>>,
    bodies: Arc<Mutex<Vec<Value>>>,
}

impl Respond for PollResponses {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        self.times.lock().unwrap().push(Instant::now());
        self.bodies.lock().unwrap().push(json_body(request));
        let mut responses = self.responses.lock().unwrap();
        assert!(!responses.is_empty(), "Unexpected extra device auth poll");
        responses.remove(0)
    }
}

struct CodexDeviceServer {
    server: MockServer,
    poll_times: Arc<Mutex<Vec<Instant>>>,
    poll_bodies: Arc<Mutex<Vec<Value>>>,
}

async fn codex_device_server(interval: Value, polls: Vec<ResponseTemplate>, access: &str) -> CodexDeviceServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/accounts/deviceauth/usercode"))
        .respond_with(json_response(
            json!({"device_auth_id": "device-auth-id", "user_code": "ABCD-1234", "interval": interval}),
            200,
        ))
        .mount(&server)
        .await;
    let poll_times = Arc::new(Mutex::new(Vec::new()));
    let poll_bodies = Arc::new(Mutex::new(Vec::new()));
    Mock::given(method("POST"))
        .and(path("/api/accounts/deviceauth/token"))
        .respond_with(PollResponses {
            responses: Mutex::new(polls),
            times: poll_times.clone(),
            bodies: poll_bodies.clone(),
        })
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(json_response(
            json!({"access_token": access, "refresh_token": "refresh-token", "expires_in": 3600}),
            200,
        ))
        .mount(&server)
        .await;
    CodexDeviceServer {
        server,
        poll_times,
        poll_bodies,
    }
}

fn device_code_completion() -> ResponseTemplate {
    json_response(
        json!({"authorization_code": "oauth-code", "code_challenge": "device-code-challenge",
            "code_verifier": "device-code-verifier"}),
        200,
    )
}

#[tokio::test]
async fn codex_logs_in_with_the_device_code_flow() {
    let access = access_token("account-123");
    let device = codex_device_server(
        json!("1"),
        vec![device_auth_pending(), device_code_completion()],
        &access,
    )
    .await;
    let interaction = select_device_code();

    let before = now_ms();
    let credential = codex(&device.server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap();

    assert_eq!(
        interaction.events(),
        [AuthEvent::DeviceCode {
            user_code: "ABCD-1234".into(),
            verification_uri: "https://auth.openai.com/codex/device".into(),
            interval_seconds: Some(1),
            expires_in_seconds: Some(900),
        }]
    );
    // The first poll runs at once, the second one interval later.
    let times = device.poll_times.lock().unwrap().clone();
    assert_eq!(times.len(), 2);
    assert!(times[1] - times[0] >= Duration::from_millis(950));
    assert_eq!(
        device.poll_bodies.lock().unwrap()[0],
        json!({"device_auth_id": "device-auth-id", "user_code": "ABCD-1234"})
    );

    let requests = device.server.received_requests().await.unwrap();
    let usercode = &requests[0];
    assert_eq!(usercode.headers["content-type"], "application/json");
    assert_eq!(json_body(usercode), json!({"client_id": CODEX_CLIENT_ID}));
    let exchange = requests
        .iter()
        .find(|request| request.url.path() == "/oauth/token")
        .unwrap();
    assert_eq!(exchange.headers["content-type"], "application/x-www-form-urlencoded");
    let form = form_body(exchange);
    assert_eq!(form_field(&form, "grant_type").as_deref(), Some("authorization_code"));
    assert_eq!(form_field(&form, "client_id").as_deref(), Some(CODEX_CLIENT_ID));
    assert_eq!(form_field(&form, "code").as_deref(), Some("oauth-code"));
    assert_eq!(
        form_field(&form, "redirect_uri").as_deref(),
        Some("https://auth.openai.com/deviceauth/callback")
    );
    assert_eq!(
        form_field(&form, "code_verifier").as_deref(),
        Some("device-code-verifier")
    );

    assert_eq!(credential.access, access);
    assert_eq!(credential.refresh, "refresh-token");
    assert_eq!(credential.extra["accountId"], "account-123");
    assert!(credential.expires >= before + 3_600_000 && credential.expires <= now_ms() + 3_600_000);
}

#[tokio::test]
async fn codex_offers_browser_login_first_and_uses_the_selected_device_code_flow() {
    let access = access_token("account-456");
    let device = codex_device_server(json!("5"), vec![device_code_completion()], &access).await;
    let interaction = select_device_code().on_event(|event| {
        assert!(
            !matches!(event, AuthEvent::AuthUrl { .. }),
            "Browser login should not start"
        );
    });

    let credential = codex(&device.server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap();
    assert_eq!(credential.access, access);
    assert_eq!(credential.extra["accountId"], "account-456");

    let prompts = interaction.prompts.lock().unwrap().clone();
    assert_eq!(prompts.len(), 1);
    let AuthPrompt::Select { message, options } = &prompts[0] else {
        panic!("expected a select prompt");
    };
    assert_eq!(message, "Select OpenAI Codex login method:");
    assert_eq!(
        options,
        &[
            AuthSelectOption {
                id: "browser".into(),
                label: "Browser login (default)".into(),
                description: None
            },
            AuthSelectOption {
                id: "device_code".into(),
                label: "Device code login (headless)".into(),
                description: None
            },
        ]
    );
    assert_eq!(
        interaction.events(),
        [AuthEvent::DeviceCode {
            user_code: "ABCD-1234".into(),
            verification_uri: "https://auth.openai.com/codex/device".into(),
            interval_seconds: Some(5),
            expires_in_seconds: Some(900),
        }]
    );
}

#[tokio::test]
async fn codex_cancels_when_login_method_selection_is_cancelled() {
    let interaction = TestInteraction::new(|_| {
        Box::pin(async { Err(ModelsError::new(ModelsErrorCode::Aborted, "Login cancelled")) })
    });
    let error = OpenAICodexOAuth::default()
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.message(), "Login cancelled");
}

#[tokio::test]
async fn codex_cancels_the_device_code_flow_while_waiting() {
    let device = codex_device_server(json!("5"), vec![device_auth_pending(), device_auth_pending()], "unused").await;
    let cancel = CancellationToken::new();
    let interaction = select_device_code().with_cancel(cancel.clone());
    let poll_times = device.poll_times.clone();
    let canceller = tokio::spawn(async move {
        while poll_times.lock().unwrap().is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        cancel.cancel();
    });

    let error = codex(&device.server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap_err();
    canceller.await.unwrap();
    assert_eq!(error.message(), "Login cancelled");
    assert_eq!(error.code(), ModelsErrorCode::Aborted);
    assert_eq!(device.poll_times.lock().unwrap().len(), 1);
}

#[tokio::test(start_paused = true)]
async fn codex_times_out_the_device_code_flow_after_15_minutes() {
    let device = codex_device_server(json!("60"), (0..16).map(|_| device_auth_pending()).collect(), "unused").await;
    let interaction = select_device_code();
    let started = tokio::time::Instant::now();

    let error = codex(&device.server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.message(), "Device flow timed out");
    assert_eq!(started.elapsed(), Duration::from_secs(15 * 60));
    // At most one poll at once and one per minute until the deadline. The paused clock may
    // also move on while a request waits for wiremock (which runs on its own clock), so
    // fewer polls can happen.
    let polls = device.poll_times.lock().unwrap().len();
    assert!((1..=15).contains(&polls), "{polls} polls");
}

#[tokio::test]
async fn codex_treats_device_auth_403_and_404_responses_as_pending() {
    let access = access_token("account-403-404");
    let device = codex_device_server(
        json!("1"),
        vec![
            json_response(json!({"error": "access_denied", "error_description": "denied"}), 403),
            ResponseTemplate::new(404).set_body_raw("not ready", "text/plain"),
            device_code_completion(),
        ],
        &access,
    )
    .await;

    let credential = codex(&device.server)
        .login(&select_device_code(), &LoginOptions::default())
        .await
        .unwrap();
    assert_eq!(credential.access, access);
    assert_eq!(credential.extra["accountId"], "account-403-404");
    assert_eq!(device.poll_times.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn codex_includes_the_response_body_in_device_auth_poll_failures() {
    let device = codex_device_server(
        json!("5"),
        vec![ResponseTemplate::new(500).set_body_raw(
            r#"{"error":"server_error","error_description":"try again later"}"#,
            "application/json",
        )],
        "unused",
    )
    .await;
    let error = codex(&device.server)
        .login(&select_device_code(), &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        error.message(),
        r#"OpenAI Codex device auth failed with status 500: {"error":"server_error","error_description":"try again later"}"#
    );
}

#[tokio::test]
async fn codex_reports_disabled_device_code_login() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/accounts/deviceauth/usercode"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;
    let error = codex(&server)
        .login(&select_device_code(), &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        error.message(),
        "OpenAI Codex device code login is not enabled for this server. Use browser login or verify the server URL."
    );
}

#[tokio::test]
async fn codex_reports_token_refresh_failures() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(json_response(
            json!({"error": {"message": "Could not validate your token. Please try signing in again.",
                "type": "invalid_request_error"}}),
            401,
        ))
        .mount(&server)
        .await;
    let error = codex(&server)
        .refresh(credential(
            "invalid-access-token",
            "invalid-refresh-token",
            0,
            Value::Null,
        ))
        .await
        .unwrap_err();
    assert!(error.message().starts_with("OpenAI Codex token refresh failed (401): "));
    assert!(error.message().contains("Could not validate your token"));

    let requests = server.received_requests().await.unwrap();
    let form = form_body(&requests[0]);
    assert_eq!(form_field(&form, "grant_type").as_deref(), Some("refresh_token"));
    assert_eq!(
        form_field(&form, "refresh_token").as_deref(),
        Some("invalid-refresh-token")
    );
    assert_eq!(form_field(&form, "client_id").as_deref(), Some(CODEX_CLIENT_ID));
}

#[tokio::test]
async fn codex_refresh_rejects_tokens_without_an_account_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(json_response(
            json!({"access_token": "not-a-jwt", "refresh_token": "r", "expires_in": 60}),
            200,
        ))
        .mount(&server)
        .await;
    let error = codex(&server)
        .refresh(credential("a", "r", 0, Value::Null))
        .await
        .unwrap_err();
    assert_eq!(error.message(), "Failed to extract accountId from token");

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(json_response(json!({"access_token": "a"}), 200))
        .mount(&server)
        .await;
    let error = codex(&server)
        .refresh(credential("a", "r", 0, Value::Null))
        .await
        .unwrap_err();
    assert_eq!(
        error.message(),
        r#"OpenAI Codex token refresh response missing fields: {"access_token":"a"}"#
    );
}

async fn codex_token_server(access: &str) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(json_response(
            json!({"access_token": access, "refresh_token": "refresh", "expires_in": 3600}),
            200,
        ))
        .mount(&server)
        .await;
    server
}

fn paste_redirect(port: u16) -> impl Fn(&url::Url) -> String + Send + Sync + 'static {
    move |auth_url| {
        let state = param(auth_url, "state").unwrap();
        format!("http://localhost:{port}/auth/callback?code=pasted-code&state={state}")
    }
}

#[tokio::test]
async fn codex_uses_the_apps_agent_name_as_the_browser_login_originator() {
    let server = codex_token_server(&access_token("acct")).await;
    let port = free_port().await;
    let oauth = OpenAICodexOAuth {
        callback_port: port,
        ..codex(&server)
    };
    let interaction = TestInteraction::pasting("browser", paste_redirect(port));
    let options = LoginOptions {
        agent_name: Some("my-app".into()),
        ..LoginOptions::default()
    };
    oauth.login(&interaction, &options).await.unwrap();

    let auth_url = interaction.auth_url.lock().unwrap().clone().unwrap();
    assert_eq!(param(&auth_url, "originator").as_deref(), Some("my-app"));
    assert_eq!(param(&auth_url, "client_id").as_deref(), Some(CODEX_CLIENT_ID));
    assert_eq!(param(&auth_url, "response_type").as_deref(), Some("code"));
    assert_eq!(
        param(&auth_url, "redirect_uri"),
        Some(format!("http://localhost:{port}/auth/callback"))
    );
    assert_eq!(
        param(&auth_url, "scope").as_deref(),
        Some("openid profile email offline_access")
    );
    assert_eq!(param(&auth_url, "code_challenge_method").as_deref(), Some("S256"));
    assert_eq!(param(&auth_url, "id_token_add_organizations").as_deref(), Some("true"));
    assert_eq!(param(&auth_url, "codex_cli_simplified_flow").as_deref(), Some("true"));

    // Without an agent name the originator is `pi`.
    let interaction = TestInteraction::pasting("browser", paste_redirect(port));
    oauth.login(&interaction, &LoginOptions::default()).await.unwrap();
    let auth_url = interaction.auth_url.lock().unwrap().clone().unwrap();
    assert_eq!(param(&auth_url, "originator").as_deref(), Some("pi"));
}

#[tokio::test]
async fn codex_falls_back_to_the_pasted_redirect_url_when_the_callback_port_is_taken() {
    // The port is registered with OpenAI; the Codex CLI may hold it.
    let blocker = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = blocker.local_addr().unwrap().port();
    let server = codex_token_server(&access_token("acct")).await;
    let oauth = OpenAICodexOAuth {
        callback_port: port,
        ..codex(&server)
    };

    let credential = oauth
        .login(
            &TestInteraction::pasting("browser", paste_redirect(port)),
            &LoginOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(credential.extra["accountId"], "acct");
    let requests = server.received_requests().await.unwrap();
    let form = form_body(&requests[0]);
    assert_eq!(form_field(&form, "code").as_deref(), Some("pasted-code"));
    assert_eq!(
        form_field(&form, "redirect_uri"),
        Some(format!("http://localhost:{port}/auth/callback"))
    );
    drop(blocker);
}

#[tokio::test]
async fn codex_rejects_a_pasted_state_mismatch() {
    let server = codex_token_server(&access_token("acct")).await;
    let port = free_port().await;
    let oauth = OpenAICodexOAuth {
        callback_port: port,
        ..codex(&server)
    };
    let interaction = TestInteraction::pasting("browser", move |_| {
        format!("http://localhost:{port}/auth/callback?code=c&state=other")
    });
    let error = oauth.login(&interaction, &LoginOptions::default()).await.unwrap_err();
    assert_eq!(error.message(), "State mismatch");
    // A bare code is accepted.
    let interaction = TestInteraction::pasting("browser", |_| "bare-code".to_owned());
    oauth.login(&interaction, &LoginOptions::default()).await.unwrap();
}

#[tokio::test]
async fn codex_completes_browser_login_through_the_loopback_callback() {
    let server = codex_token_server(&access_token("acct-browser")).await;
    let port = free_port().await;
    let oauth = OpenAICodexOAuth {
        callback_port: port,
        ..codex(&server)
    };
    let page: PageSlot = Arc::new(Mutex::new(None));
    let slot = page.clone();
    let interaction = TestInteraction::new(|prompt| {
        Box::pin(async move {
            match prompt {
                AuthPrompt::Select { .. } => Ok("browser".to_owned()),
                AuthPrompt::ManualCode { cancel, .. } => {
                    cancel.cancelled().await;
                    Err(ModelsError::new(ModelsErrorCode::Aborted, "aborted"))
                }
                other => panic!("unexpected prompt: {other:?}"),
            }
        })
    })
    .on_event(move |event| {
        if let AuthEvent::AuthUrl { url, instructions } = event {
            assert_eq!(
                instructions.as_deref(),
                Some("A browser window should open. Complete login to finish.")
            );
            let url = url::Url::parse(url).unwrap();
            let state = param(&url, "state").unwrap();
            let callback = format!("http://127.0.0.1:{port}/auth/callback?code=browser-code&state={state}");
            *slot.lock().unwrap() = Some(tokio::spawn(async move {
                let response = http().get(callback).send().await.unwrap();
                (response.status().as_u16(), response.text().await.unwrap())
            }));
        }
    });

    let credential = oauth.login(&interaction, &LoginOptions::default()).await.unwrap();
    assert_eq!(credential.extra["accountId"], "acct-browser");
    let handle = page.lock().unwrap().take().unwrap();
    let (status, body) = handle.await.unwrap();
    assert_eq!(status, 200);
    assert!(body.contains("Signed in to OpenAI"));
    let requests = server.received_requests().await.unwrap();
    assert_eq!(
        form_field(&form_body(&requests[0]), "code").as_deref(),
        Some("browser-code")
    );
}

// ---------------------------------------------------------------------------
// Sign in with ChatGPT (openai-chatgpt-oauth.test.ts)
// ---------------------------------------------------------------------------

const REQUIRED_SCOPE: &str = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct";
const DEVICE_ID: &str = "e61bbe28-07ef-466d-8e5d-a344f94ab305";

fn token_response(scope: &str) -> Value {
    json!({"access_token": "access-token", "refresh_token": "refresh-token", "expires_in": 3600,
        "id_token": "id-token", "scope": scope})
}

async fn chatgpt_token_server(response: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/accounts/oauth/token"))
        .respond_with(json_response(response, 200))
        .mount(&server)
        .await;
    server
}

async fn chatgpt(server: &MockServer) -> OpenAIChatGPTOAuth {
    OpenAIChatGPTOAuth {
        authorize_url: format!("{}/api/accounts/authorize", server.uri()),
        token_url: format!("{}/api/accounts/oauth/token", server.uri()),
        callback_host: Some("127.0.0.1".into()),
        callback_port: free_port().await,
    }
}

fn device_options(device_id: &'static str, agent_name: Option<&str>) -> LoginOptions {
    LoginOptions {
        device_id: Some(Arc::new(move || device_id.to_owned())),
        agent_name: agent_name.map(str::to_owned),
    }
}

/// A user who pastes the redirect URL with the authorization code, the state and, when
/// given, the issued client id.
fn chatgpt_paste(client_id: Option<&'static str>) -> TestInteraction {
    TestInteraction::pasting("unused", move |auth_url| {
        let mut callback = url::Url::parse(&param(auth_url, "redirect_uri").unwrap()).unwrap();
        callback
            .query_pairs_mut()
            .append_pair("code", "authorization-code")
            .append_pair("state", &param(auth_url, "state").unwrap());
        if let Some(client_id) = client_id {
            callback.query_pairs_mut().append_pair("client_id", client_id);
        }
        callback.to_string()
    })
}

#[tokio::test]
async fn chatgpt_registers_a_user_owned_client_and_stores_its_issued_id_and_granted_scopes() {
    let server = chatgpt_token_server(token_response(REQUIRED_SCOPE)).await;
    let oauth = chatgpt(&server).await;
    let interaction = chatgpt_paste(Some("oaiapp_issued"));

    let credential = oauth
        .login(&interaction, &device_options(DEVICE_ID, None))
        .await
        .unwrap();

    let auth_url = interaction.auth_url.lock().unwrap().clone().unwrap();
    assert_eq!(param(&auth_url, "client_id").as_deref(), Some("dynamic_agent_client"));
    assert_eq!(param(&auth_url, "agent_name_hint").as_deref(), Some("Pi"));
    assert_eq!(
        param(&auth_url, "ext_agent_host_id"),
        Some(format!("urn:uuid:{DEVICE_ID}"))
    );
    assert_eq!(param(&auth_url, "scope").as_deref(), Some(REQUIRED_SCOPE));
    assert_eq!(
        param(&auth_url, "redirect_uri"),
        Some(format!("http://127.0.0.1:{}/auth/callback", oauth.callback_port))
    );
    assert_eq!(
        param(&auth_url, "resource").as_deref(),
        Some("https://api.openai.com/v1")
    );
    assert_eq!(param(&auth_url, "code_challenge_method").as_deref(), Some("S256"));
    assert!(param(&auth_url, "nonce").is_some_and(|nonce| nonce.len() == 43));

    let requests = server.received_requests().await.unwrap();
    let exchange = form_body(&requests[0]);
    assert_eq!(requests[0].headers["accept"], "application/json");
    assert_eq!(
        form_field(&exchange, "grant_type").as_deref(),
        Some("authorization_code")
    );
    assert_eq!(form_field(&exchange, "client_id").as_deref(), Some("oaiapp_issued"));
    assert_eq!(form_field(&exchange, "code").as_deref(), Some("authorization-code"));
    assert_eq!(
        form_field(&exchange, "resource").as_deref(),
        Some("https://api.openai.com/v1")
    );
    assert!(form_field(&exchange, "code_verifier").is_some_and(|verifier| !verifier.is_empty()));

    assert_eq!(credential.access, "access-token");
    assert_eq!(credential.refresh, "refresh-token");
    assert_eq!(credential.extra["clientId"], "oaiapp_issued");
    assert_eq!(
        credential.extra["scopes"],
        json!(REQUIRED_SCOPE.split(' ').collect::<Vec<_>>())
    );
    assert!(interaction.events().contains(&AuthEvent::Progress {
        message: "Exchanging authorization code for tokens...".into()
    }));
}

#[tokio::test]
async fn chatgpt_uses_the_apps_agent_name_as_the_name_hint() {
    let server = chatgpt_token_server(token_response(REQUIRED_SCOPE)).await;
    let oauth = chatgpt(&server).await;
    let interaction = chatgpt_paste(Some("oaiapp_issued"));
    oauth
        .login(&interaction, &device_options(DEVICE_ID, Some("my-app")))
        .await
        .unwrap();
    let auth_url = interaction.auth_url.lock().unwrap().clone().unwrap();
    assert_eq!(param(&auth_url, "agent_name_hint").as_deref(), Some("my-app"));
}

#[tokio::test]
async fn chatgpt_rejects_registration_without_an_issued_client_id() {
    let server = chatgpt_token_server(token_response(REQUIRED_SCOPE)).await;
    let error = chatgpt(&server)
        .await
        .login(&chatgpt_paste(None), &device_options(DEVICE_ID, None))
        .await
        .unwrap_err();
    assert!(
        error
            .message()
            .contains("registration callback did not contain an issued client ID")
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn chatgpt_rejects_a_token_response_that_did_not_grant_direct_token_use() {
    let server = chatgpt_token_server(token_response("openid profile email offline_access resource.invoke")).await;
    let error = chatgpt(&server)
        .await
        .login(&chatgpt_paste(Some("oaiapp_issued")), &device_options(DEVICE_ID, None))
        .await
        .unwrap_err();
    assert!(
        error
            .message()
            .contains("grant did not include chatgpt.tokens.use.direct")
    );
}

#[tokio::test]
async fn chatgpt_requires_an_id_token_in_the_exchange() {
    let mut response = token_response(REQUIRED_SCOPE);
    response.as_object_mut().unwrap().remove("id_token");
    let server = chatgpt_token_server(response).await;
    let error = chatgpt(&server)
        .await
        .login(&chatgpt_paste(Some("oaiapp_issued")), &device_options(DEVICE_ID, None))
        .await
        .unwrap_err();
    assert_eq!(
        error.message(),
        "OpenAI OAuth token response did not contain an ID token"
    );
}

#[tokio::test]
async fn chatgpt_requires_a_device_id_before_starting_authorization() {
    let server = chatgpt_token_server(token_response(REQUIRED_SCOPE)).await;
    let oauth = chatgpt(&server).await;
    for options in [LoginOptions::default(), device_options("not-a-uuid", None)] {
        let interaction = chatgpt_paste(Some("oaiapp_issued"));
        let error = oauth.login(&interaction, &options).await.unwrap_err();
        assert!(error.message().contains("requires a device ID"));
        assert!(interaction.events().is_empty());
    }
}

#[tokio::test]
async fn chatgpt_reports_a_taken_callback_port() {
    let blocker = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = blocker.local_addr().unwrap().port();
    let server = chatgpt_token_server(token_response(REQUIRED_SCOPE)).await;
    let oauth = OpenAIChatGPTOAuth {
        callback_port: port,
        ..chatgpt(&server).await
    };
    let interaction = chatgpt_paste(Some("oaiapp_issued"));
    let error = oauth
        .login(&interaction, &device_options(DEVICE_ID, None))
        .await
        .unwrap_err();
    assert_eq!(
        error.message(),
        format!(
            "Port {port} is in use, probably by an unfinished login in another pi session or by the Codex CLI. Cancel that login and try again."
        )
    );
    assert!(interaction.events().is_empty());
}

#[tokio::test]
async fn chatgpt_completes_login_through_the_loopback_callback() {
    let server = chatgpt_token_server(token_response(REQUIRED_SCOPE)).await;
    let oauth = chatgpt(&server).await;
    let pages: PageSlots = Arc::new(Mutex::new(Vec::new()));
    let slot = pages.clone();
    let port = oauth.callback_port;
    let interaction = TestInteraction::new(|prompt| {
        Box::pin(async move {
            let AuthPrompt::ManualCode { cancel, .. } = prompt else {
                panic!("unexpected prompt");
            };
            cancel.cancelled().await;
            Err(ModelsError::new(ModelsErrorCode::Aborted, "aborted"))
        })
    })
    .on_event(move |event| {
        let AuthEvent::AuthUrl { url, .. } = event else {
            return;
        };
        let url = url::Url::parse(url).unwrap();
        let state = param(&url, "state").unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let requests = [
            format!("{base}/other"),
            format!("{base}/auth/callback?code=c&state=wrong&client_id=x"),
            format!("{base}/auth/callback?code=browser-code&state={state}&client_id=oaiapp_browser"),
        ];
        // Sequential, so the invalid callbacks are answered before the valid one.
        *slot.lock().unwrap() = vec![tokio::spawn(async move {
            let mut last = (0, String::new());
            for (index, request) in requests.into_iter().enumerate() {
                let response = http().get(request).send().await.unwrap();
                let page = (response.status().as_u16(), response.text().await.unwrap());
                match index {
                    0 => assert_eq!(page.0, 404),
                    1 => {
                        assert_eq!(page.0, 400);
                        assert!(page.1.contains("OAuth state mismatch"));
                    }
                    _ => last = page,
                }
            }
            last
        })];
    });

    let credential = oauth
        .login(&interaction, &device_options(DEVICE_ID, None))
        .await
        .unwrap();
    assert_eq!(credential.extra["clientId"], "oaiapp_browser");
    let handle = pages.lock().unwrap().pop().unwrap();
    let (status, body) = handle.await.unwrap();
    assert_eq!(status, 200);
    assert!(body.contains("ChatGPT authentication completed. You can close this window."));
    // The server is closed once login returns.
    assert!(
        http()
            .get(format!("http://127.0.0.1:{port}/auth/callback"))
            .send()
            .await
            .is_err()
    );
}

#[tokio::test]
async fn chatgpt_fails_when_the_provider_redirects_with_an_error() {
    let server = chatgpt_token_server(token_response(REQUIRED_SCOPE)).await;
    let oauth = chatgpt(&server).await;
    let port = oauth.callback_port;
    let page: PageSlot = Arc::new(Mutex::new(None));
    let slot = page.clone();
    let interaction = TestInteraction::new(|prompt| {
        Box::pin(async move {
            let AuthPrompt::ManualCode { cancel, .. } = prompt else {
                panic!("unexpected prompt");
            };
            cancel.cancelled().await;
            Err(ModelsError::new(ModelsErrorCode::Aborted, "aborted"))
        })
    })
    .on_event(move |event| {
        if matches!(event, AuthEvent::AuthUrl { .. }) {
            let url = format!("http://127.0.0.1:{port}/auth/callback?error=access_denied");
            *slot.lock().unwrap() = Some(tokio::spawn(async move {
                let response = http().get(url).send().await.unwrap();
                (response.status().as_u16(), response.text().await.unwrap())
            }));
        }
    });
    let error = oauth
        .login(&interaction, &device_options(DEVICE_ID, None))
        .await
        .unwrap_err();
    assert_eq!(error.message(), "ChatGPT authorization failed: access_denied");
    let handle = page.lock().unwrap().take().unwrap();
    let (status, body) = handle.await.unwrap();
    assert_eq!(status, 400);
    assert!(body.contains("ChatGPT was not connected.") && body.contains("Error: access_denied"));
}

#[tokio::test]
async fn chatgpt_requires_refresh_responses_to_rotate_the_refresh_token() {
    let mut response = token_response(REQUIRED_SCOPE);
    response.as_object_mut().unwrap().remove("refresh_token");
    let server = chatgpt_token_server(response).await;
    let error = chatgpt(&server)
        .await
        .refresh(connected_credential())
        .await
        .unwrap_err();
    assert!(error.message().contains("token response has invalid refresh_token"));
}

fn connected_credential() -> OAuthCredential {
    credential(
        "old-access",
        "old-refresh",
        0,
        json!({"clientId": "oaiapp_existing", "scopes": REQUIRED_SCOPE.split(' ').collect::<Vec<_>>()}),
    )
}

#[tokio::test]
async fn chatgpt_refreshes_with_the_credentials_issued_client_id_and_stores_replacement_scopes() {
    let mut response = token_response(REQUIRED_SCOPE);
    response["access_token"] = json!("new-access");
    response["refresh_token"] = json!("new-refresh");
    let server = chatgpt_token_server(response).await;

    let before = now_ms();
    let credential = chatgpt(&server).await.refresh(connected_credential()).await.unwrap();

    // expires_in is 3600 seconds; the credential expires 3 minutes early.
    assert!(credential.expires >= before + (3600 - 180) * 1000);
    assert!(credential.expires <= now_ms() + (3600 - 180) * 1000);
    let requests = server.received_requests().await.unwrap();
    let form = form_body(&requests[0]);
    assert_eq!(form_field(&form, "grant_type").as_deref(), Some("refresh_token"));
    assert_eq!(form_field(&form, "client_id").as_deref(), Some("oaiapp_existing"));
    assert_eq!(form_field(&form, "refresh_token").as_deref(), Some("old-refresh"));
    assert_eq!(
        form_field(&form, "resource").as_deref(),
        Some("https://api.openai.com/v1")
    );
    assert_eq!(form_field(&form, "scope"), None);
    assert_eq!(credential.access, "new-access");
    assert_eq!(credential.refresh, "new-refresh");
    assert_eq!(credential.extra["clientId"], "oaiapp_existing");
    assert_eq!(
        credential.extra["scopes"],
        json!(REQUIRED_SCOPE.split(' ').collect::<Vec<_>>())
    );
}

#[tokio::test]
async fn chatgpt_refresh_needs_the_issued_client_id() {
    let server = chatgpt_token_server(token_response(REQUIRED_SCOPE)).await;
    let error = chatgpt(&server)
        .await
        .refresh(credential("a", "r", 0, Value::Null))
        .await
        .unwrap_err();
    assert_eq!(
        error.message(),
        "Stored OpenAI OAuth credential does not contain an issued client ID; reconnect ChatGPT"
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}
