//! Tests of the Kimi Code and Meta OAuth device flows and their provider wiring. Ported
//! from upstream `test/kimi-coding-oauth.test.ts` and `test/meta-oauth.test.ts`.
//!
//! Auth endpoints are wiremock servers. Upstream's timing tests use fake timers with
//! 5-second intervals; here the server sends 1-second intervals and the tests check the
//! gaps in real time (the HTTP requests need a real clock).

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

use delta_ai::auth::{AuthEvent, AuthInteraction, AuthPrompt, AuthResolutionOverrides, LoginOptions, OAuthAuth};
use delta_ai::models::CreateModelsOptions;
use delta_ai::oauth::{KimiCodingOAuth, MetaOAuth, kimi_coding, meta};
use delta_ai::providers::builtin::builtin_provider;
use delta_ai::types::now_ms;
use delta_ai::{
    Context, Credential, CredentialStore, InMemoryCredentialStore, JsonObject, Message, Model, Models, ModelsError,
    ModelsErrorCode, OAuthCredential, StopReason, StreamOptions,
};
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

/// Records events; fails every prompt (device flows never prompt).
struct TestInteraction {
    cancel: CancellationToken,
    on_event: Box<dyn Fn(&AuthEvent) + Send + Sync>,
    events: Mutex<Vec<AuthEvent>>,
}

impl TestInteraction {
    fn new() -> Self {
        Self {
            cancel: CancellationToken::new(),
            on_event: Box::new(|_| {}),
            events: Mutex::new(Vec::new()),
        }
    }

    fn events(&self) -> Vec<AuthEvent> {
        self.events.lock().unwrap().clone()
    }
}

impl AuthInteraction for TestInteraction {
    fn prompt(&self, _prompt: AuthPrompt) -> BoxFuture<'_, Result<String, ModelsError>> {
        panic!("Device-code login should not prompt")
    }

    fn notify(&self, event: AuthEvent) {
        (self.on_event)(&event);
        self.events.lock().unwrap().push(event);
    }

    fn cancel(&self) -> CancellationToken {
        self.cancel.clone()
    }
}

fn json_response(body: Value, status: u16) -> ResponseTemplate {
    ResponseTemplate::new(status).set_body_json(body)
}

fn form_body(request: &Request) -> Vec<(String, String)> {
    url::form_urlencoded::parse(&request.body).into_owned().collect()
}

fn form_field(form: &[(String, String)], name: &str) -> Option<String> {
    form.iter().find(|(key, _)| key == name).map(|(_, value)| value.clone())
}

fn header<'a>(request: &'a Request, name: &str) -> Option<&'a str> {
    request.headers.get(name).map(|value| value.to_str().unwrap())
}

fn credential(access: &str, refresh: &str, expires: i64) -> OAuthCredential {
    OAuthCredential {
        refresh: refresh.into(),
        access: access.into(),
        expires,
        extra: JsonObject::new(),
    }
}

fn assert_gap(later: Instant, earlier: Instant, at_least: Duration) {
    let gap = later.duration_since(earlier);
    assert!(
        gap + Duration::from_millis(20) >= at_least,
        "gap {gap:?} < {at_least:?}"
    );
}

/// Replies from a queue (the last reply repeats) and records each request and its time.
struct Replies {
    replies: Mutex<Vec<ResponseTemplate>>,
    requests: Arc<Mutex<Vec<(Instant, Request)>>>,
}

impl Respond for Replies {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        self.requests.lock().unwrap().push((Instant::now(), request.clone()));
        let mut replies = self.replies.lock().unwrap();
        assert!(!replies.is_empty(), "Unexpected request to {}", request.url);
        if replies.len() == 1 {
            replies[0].clone()
        } else {
            replies.remove(0)
        }
    }
}

type Recorded = Arc<Mutex<Vec<(Instant, Request)>>>;

async fn mount(server: &MockServer, path_: &str, replies: Vec<ResponseTemplate>) -> Recorded {
    let requests = Recorded::default();
    Mock::given(method("POST"))
        .and(path(path_))
        .respond_with(Replies {
            replies: Mutex::new(replies),
            requests: requests.clone(),
        })
        .mount(server)
        .await;
    requests
}

fn recorded(requests: &Recorded) -> Vec<(Instant, Request)> {
    requests.lock().unwrap().clone()
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

/// The provider's first catalog model, sent to `server` instead of `host`.
fn mock_model(models: &Models, provider_id: &str, host: &str, server: &MockServer) -> Model {
    let mut model = models.get_models(Some(provider_id))[0].clone();
    model.base_url = model.base_url.replace(host, &server.uri());
    model
}

fn stream_options() -> StreamOptions {
    StreamOptions {
        http_client: Some(reqwest::Client::builder().no_proxy().build().unwrap()),
        max_retries: Some(0),
        ..StreamOptions::default()
    }
}

#[test]
fn kimi_coding_and_meta_expose_oauth_alongside_api_key_auth() {
    let kimi = builtin_provider("kimi-coding").unwrap();
    assert!(kimi.auth().api_key.is_some());
    let oauth = kimi.auth().oauth.as_ref().unwrap();
    assert_eq!(oauth.name(), "Kimi Code (subscription)");
    assert!(oauth.is_subscription());
    assert_eq!(oauth.login_label(), Some("Sign in with Kimi Code"));

    let meta = builtin_provider("meta").unwrap();
    assert!(meta.auth().api_key.is_some());
    let oauth = meta.auth().oauth.as_ref().unwrap();
    assert_eq!(oauth.name(), "Meta (Muse subscription)");
    assert!(oauth.is_subscription());
    assert_eq!(oauth.login_label(), Some("Sign in with Meta"));
}

// ---------------------------------------------------------------------------
// Kimi Code OAuth (kimi-coding-oauth.test.ts)
// ---------------------------------------------------------------------------

const KIMI_DEVICE_PATH: &str = "/api/oauth/device_authorization";
const KIMI_TOKEN_PATH: &str = "/api/oauth/token";

fn kimi_device_response(overrides: &Value) -> ResponseTemplate {
    let mut body = json!({
        "user_code": "ABCD-1234",
        "device_code": "device-code-123",
        "verification_uri": "https://www.kimi.com/code",
        "verification_uri_complete": "https://www.kimi.com/code?user_code=ABCD-1234",
        "interval": 1,
        "expires_in": 600,
    });
    for (key, value) in overrides.as_object().unwrap() {
        body[key] = value.clone();
    }
    json_response(body, 200)
}

fn kimi(server: &MockServer) -> KimiCodingOAuth {
    KimiCodingOAuth {
        oauth_host: Some(server.uri()),
    }
}

async fn kimi_server(device: ResponseTemplate, tokens: Vec<ResponseTemplate>) -> (MockServer, Recorded, Recorded) {
    let server = MockServer::start().await;
    let device = mount(&server, KIMI_DEVICE_PATH, vec![device]).await;
    let tokens = mount(&server, KIMI_TOKEN_PATH, tokens).await;
    (server, device, tokens)
}

#[tokio::test]
async fn kimi_logs_in_with_the_device_authorization_flow() {
    let (server, device, tokens) = kimi_server(
        kimi_device_response(&json!({})),
        vec![
            json_response(json!({"error": "authorization_pending"}), 400),
            json_response(
                json!({"access_token": "access-token", "refresh_token": "refresh-token", "expires_in": 3600}),
                200,
            ),
        ],
    )
    .await;
    let interaction = TestInteraction::new();
    let start = Instant::now();
    let credential = kimi(&server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap();
    let after = now_ms();

    let (_, device_request) = &recorded(&device)[0];
    assert_eq!(
        header(device_request, "content-type"),
        Some("application/x-www-form-urlencoded")
    );
    assert_eq!(header(device_request, "accept"), Some("application/json"));
    assert_eq!(
        form_field(&form_body(device_request), "client_id").as_deref(),
        Some(kimi_coding::CLIENT_ID)
    );
    assert_eq!(
        interaction.events(),
        [AuthEvent::DeviceCode {
            user_code: "ABCD-1234".into(),
            verification_uri: "https://www.kimi.com/code?user_code=ABCD-1234".into(),
            interval_seconds: Some(1),
            expires_in_seconds: Some(600),
        }]
    );

    let polls = recorded(&tokens);
    assert_eq!(polls.len(), 2);
    for (_, request) in &polls {
        let form = form_body(request);
        assert_eq!(
            form_field(&form, "grant_type").as_deref(),
            Some("urn:ietf:params:oauth:grant-type:device_code")
        );
        assert_eq!(form_field(&form, "client_id").as_deref(), Some(kimi_coding::CLIENT_ID));
        assert_eq!(form_field(&form, "device_code").as_deref(), Some("device-code-123"));
    }
    // waitBeforeFirstPoll: the first poll happens after one interval.
    assert_gap(polls[0].0, start, Duration::from_secs(1));
    assert_gap(polls[1].0, polls[0].0, Duration::from_secs(1));

    assert_eq!(credential.access, "access-token");
    assert_eq!(credential.refresh, "refresh-token");
    // No expiry skew: now + expires_in.
    assert!(credential.expires <= after + 3_600_000);
    assert!(credential.expires >= after + 3_600_000 - 1000);
}

#[tokio::test]
async fn kimi_fails_when_the_device_code_expires() {
    let (server, _, _) = kimi_server(
        kimi_device_response(&json!({})),
        vec![json_response(json!({"error": "expired_token"}), 400)],
    )
    .await;
    let error = kimi(&server)
        .login(&TestInteraction::new(), &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        error.message(),
        "Kimi Code device authorization expired. Please restart login."
    );
    assert_eq!(error.code(), ModelsErrorCode::OAuth);
}

#[tokio::test]
async fn kimi_fails_when_the_user_denies_the_login() {
    let (server, _, _) = kimi_server(
        kimi_device_response(&json!({})),
        vec![json_response(json!({"error": "access_denied"}), 400)],
    )
    .await;
    let error = kimi(&server)
        .login(&TestInteraction::new(), &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.message(), "Kimi Code login was denied.");
}

#[tokio::test]
async fn kimi_honors_the_oauth_host_override_and_strips_trailing_slashes() {
    // Upstream sets KIMI_CODE_OAUTH_HOST; the environment lookup itself is a unit test in
    // `oauth::kimi_coding`.
    let (server, device, tokens) = kimi_server(
        kimi_device_response(&json!({"interval": 1})),
        vec![json_response(
            json!({"access_token": "a", "refresh_token": "r", "expires_in": 60}),
            200,
        )],
    )
    .await;
    let oauth = KimiCodingOAuth {
        oauth_host: Some(format!("{}/", server.uri())),
    };
    let credential = oauth
        .login(&TestInteraction::new(), &LoginOptions::default())
        .await
        .unwrap();
    assert_eq!((credential.access.as_str(), credential.refresh.as_str()), ("a", "r"));
    assert_eq!(recorded(&device)[0].1.url.path(), KIMI_DEVICE_PATH);
    assert_eq!(recorded(&tokens)[0].1.url.path(), KIMI_TOKEN_PATH);
}

#[tokio::test]
async fn kimi_refreshes_tokens_and_returns_a_bearer_header_for_requests() {
    let (server, _, tokens) = kimi_server(
        kimi_device_response(&json!({})),
        vec![json_response(
            json!({"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600}),
            200,
        )],
    )
    .await;
    let oauth = kimi(&server);
    let before = now_ms();
    let refreshed = oauth
        .refresh(credential("old-access", "old-refresh", before))
        .await
        .unwrap();
    let form = form_body(&recorded(&tokens)[0].1);
    assert_eq!(form_field(&form, "grant_type").as_deref(), Some("refresh_token"));
    assert_eq!(form_field(&form, "refresh_token").as_deref(), Some("old-refresh"));
    assert_eq!(form_field(&form, "client_id").as_deref(), Some(kimi_coding::CLIENT_ID));
    assert_eq!(refreshed.access, "new-access");
    assert_eq!(refreshed.refresh, "new-refresh");
    assert!(refreshed.extra.is_empty());
    assert!(refreshed.expires >= before + 3_600_000);

    let auth = oauth.to_auth(&refreshed).await.unwrap();
    assert_eq!(auth.api_key, None);
    assert_eq!(auth.base_url, None);
    let headers = auth.headers.unwrap();
    assert_eq!(headers.len(), 1);
    assert_eq!(
        headers.get("Authorization").cloned().flatten().as_deref(),
        Some("Bearer new-access")
    );
}

#[tokio::test]
async fn kimi_retries_refresh_on_429_and_fails_unauthorized_on_invalid_grant() {
    // 429 once, then success.
    let (server, _, tokens) = kimi_server(
        kimi_device_response(&json!({})),
        vec![
            json_response(json!({"error": "temporarily_unavailable"}), 429),
            json_response(
                json!({"access_token": "a", "refresh_token": "r", "expires_in": 60}),
                200,
            ),
        ],
    )
    .await;
    let refreshed = kimi(&server).refresh(credential("old", "old", 0)).await.unwrap();
    assert_eq!(refreshed.access, "a");
    let calls = recorded(&tokens);
    assert_eq!(calls.len(), 2);
    assert_gap(calls[1].0, calls[0].0, Duration::from_secs(1));

    // invalid_grant is not retried.
    let (server, _, tokens) = kimi_server(
        kimi_device_response(&json!({})),
        vec![json_response(json!({"error": "invalid_grant"}), 400)],
    )
    .await;
    let error = kimi(&server).refresh(credential("old", "old", 0)).await.unwrap_err();
    assert_eq!(error.message(), "Kimi Code token refresh unauthorized (status 400)");
    assert_eq!(recorded(&tokens).len(), 1);
}

#[tokio::test]
async fn kimi_refresh_reports_unauthorized_and_non_retryable_failures() {
    let (server, _, _) = kimi_server(
        kimi_device_response(&json!({})),
        vec![json_response(
            json!({"error": "unauthorized", "error_description": "token revoked"}),
            401,
        )],
    )
    .await;
    let error = kimi(&server).refresh(credential("old", "old", 0)).await.unwrap_err();
    assert_eq!(
        error.message(),
        "Kimi Code token refresh unauthorized (status 401): token revoked"
    );

    let (server, _, tokens) = kimi_server(
        kimi_device_response(&json!({})),
        vec![json_response(json!({"error": "bad_request"}), 400)],
    )
    .await;
    let error = kimi(&server).refresh(credential("old", "old", 0)).await.unwrap_err();
    assert_eq!(
        error.message(),
        r#"Kimi Code token refresh failed with status 400: {"error":"bad_request"}"#
    );
    assert_eq!(recorded(&tokens).len(), 1);

    let (server, _, _) = kimi_server(
        kimi_device_response(&json!({})),
        vec![json_response(json!({"access_token": "a", "expires_in": 60}), 200)],
    )
    .await;
    let error = kimi(&server).refresh(credential("old", "old", 0)).await.unwrap_err();
    assert_eq!(
        error.message(),
        r#"Kimi Code token refresh response missing fields: {"access_token":"a","expires_in":60}"#
    );
}

#[tokio::test]
async fn kimi_rejects_untrusted_verification_uris_and_failed_device_authorization() {
    let (server, _, _) = kimi_server(
        kimi_device_response(&json!({"verification_uri_complete": "file:///etc/passwd"})),
        vec![],
    )
    .await;
    let error = kimi(&server)
        .login(&TestInteraction::new(), &LoginOptions::default())
        .await
        .unwrap_err();
    assert!(
        error
            .message()
            .starts_with("Invalid Kimi Code device authorization response: {"),
        "{}",
        error.message()
    );

    let (server, _, _) = kimi_server(ResponseTemplate::new(503).set_body_string("unavailable"), vec![]).await;
    let error = kimi(&server)
        .login(&TestInteraction::new(), &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        error.message(),
        "Kimi Code device authorization failed with status 503: unavailable"
    );
}

#[tokio::test]
async fn kimi_cancels_while_waiting_for_the_first_token_poll() {
    let (server, _, tokens) = kimi_server(kimi_device_response(&json!({})), vec![]).await;
    let mut interaction = TestInteraction::new();
    let token = interaction.cancel.clone();
    interaction.on_event = Box::new(move |_| token.cancel());
    let error = kimi(&server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.message(), "Login cancelled");
    assert_eq!(error.code(), ModelsErrorCode::Aborted);
    assert!(recorded(&tokens).is_empty());
}

#[tokio::test]
async fn kimi_requests_send_the_oauth_access_token_as_a_bearer_header() {
    let models = models_with("kimi-coding", credential("kimi-access", "r", now_ms() + 3_600_000)).await;
    let auth = models
        .get_auth("kimi-coding", &AuthResolutionOverrides::default())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(auth.source.as_deref(), Some("OAuth"));
    assert_eq!(auth.auth.api_key, None);

    let server = MockServer::start().await;
    let chunk = json!({"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}});
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(format!("data: {chunk}\n\ndata: [DONE]\n\n"), "text/event-stream"),
        )
        .mount(&server)
        .await;
    let model = mock_model(&models, "kimi-coding", "https://api.kimi.com", &server);
    let message = models
        .complete(&model, &Context::new(vec![Message::user("hi")]), stream_options())
        .await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);

    let request = server.received_requests().await.unwrap().pop().unwrap();
    assert_eq!(request.url.path(), "/coding/v1/chat/completions");
    let authorization: Vec<_> = request.headers.get_all("authorization").iter().collect();
    assert_eq!(authorization, ["Bearer kimi-access"]);
}

// ---------------------------------------------------------------------------
// Meta OAuth (meta-oauth.test.ts)
// ---------------------------------------------------------------------------

const META_DEVICE_PATH: &str = "/oidc/device/authorization/";
const META_TOKEN_PATH: &str = "/oidc/device/token/";
const META_MINT_PATH: &str = "/muse-code/key";
const DAY_MS: i64 = 24 * 60 * 60 * 1000;

fn meta_oauth(server: &MockServer) -> MetaOAuth {
    MetaOAuth {
        device_authorization_url: format!("{}{META_DEVICE_PATH}", server.uri()),
        device_token_url: format!("{}{META_TOKEN_PATH}", server.uri()),
        api_key_mint_url: format!("{}{META_MINT_PATH}", server.uri()),
    }
}

fn meta_device_response() -> ResponseTemplate {
    json_response(
        json!({
            "device_code": "device-code-123",
            "user_code": "ABCD-1234",
            "verification_uri": "https://auth.meta.com/oauth/device/",
            "verification_uri_complete": "https://auth.meta.com/oauth/device/?code=ABCD-1234",
            "interval": 1,
            "expires_in": 600,
        }),
        200,
    )
}

struct MetaServer {
    server: MockServer,
    device: Recorded,
    tokens: Recorded,
    mint: Recorded,
}

async fn meta_server(
    device: ResponseTemplate,
    tokens: Vec<ResponseTemplate>,
    mint: Vec<ResponseTemplate>,
) -> MetaServer {
    let server = MockServer::start().await;
    MetaServer {
        device: mount(&server, META_DEVICE_PATH, vec![device]).await,
        tokens: mount(&server, META_TOKEN_PATH, tokens).await,
        mint: mount(&server, META_MINT_PATH, mint).await,
        server,
    }
}

#[tokio::test]
async fn meta_logs_in_with_the_device_flow_and_mints_a_model_api_key() {
    let meta = meta_server(
        meta_device_response(),
        vec![
            json_response(json!({"error": "authorization_pending"}), 400),
            json_response(json!({"access_token": "identity-token", "token_type": "Bearer"}), 200),
        ],
        vec![json_response(json!({"api_key": "LLM|minted-key"}), 200)],
    )
    .await;
    let interaction = TestInteraction::new();
    let start = Instant::now();
    let credential = meta_oauth(&meta.server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap();
    let after = now_ms();

    let (_, device_request) = &recorded(&meta.device)[0];
    assert_eq!(
        form_field(&form_body(device_request), "client_id").as_deref(),
        Some(meta::CLIENT_ID)
    );
    assert_eq!(
        interaction.events(),
        [
            AuthEvent::DeviceCode {
                user_code: "ABCD-1234".into(),
                verification_uri: "https://auth.meta.com/oauth/device/?code=ABCD-1234".into(),
                interval_seconds: Some(1),
                expires_in_seconds: Some(600),
            },
            AuthEvent::Progress {
                message: "Enabling Meta Model API access...".into(),
            },
        ]
    );

    let polls = recorded(&meta.tokens);
    assert_eq!(polls.len(), 2);
    for (_, request) in &polls {
        let form = form_body(request);
        assert_eq!(
            form_field(&form, "grant_type").as_deref(),
            Some("urn:ietf:params:oauth:grant-type:device_code")
        );
        assert_eq!(form_field(&form, "client_id").as_deref(), Some(meta::CLIENT_ID));
        assert_eq!(form_field(&form, "device_code").as_deref(), Some("device-code-123"));
    }
    assert_gap(polls[0].0, start, Duration::from_secs(1));
    assert_gap(polls[1].0, polls[0].0, Duration::from_secs(1));

    let (_, mint) = &recorded(&meta.mint)[0];
    assert_eq!(header(mint, "authorization"), Some("Bearer identity-token"));
    assert_eq!(header(mint, "accept"), Some("application/json"));
    assert_eq!(header(mint, "content-type"), Some("application/json"));
    assert_eq!(header(mint, "x-api-version"), Some("1.0.0"));
    assert_eq!(mint.body, b"{}");

    assert_eq!(credential.refresh, "identity-token");
    assert_eq!(credential.access, "LLM|minted-key");
    assert!(credential.expires <= after + DAY_MS);
    assert!(credential.expires >= after + DAY_MS - 1000);
}

#[tokio::test]
async fn meta_re_mints_the_api_key_from_the_stored_identity_token_on_refresh() {
    let meta = meta_server(
        meta_device_response(),
        vec![],
        vec![json_response(json!({"api_key": "LLM|fresh-key"}), 200)],
    )
    .await;
    let before = now_ms();
    let refreshed = meta_oauth(&meta.server)
        .refresh(credential("LLM|old-key", "identity-token", 1))
        .await
        .unwrap();
    let after = now_ms();
    assert_eq!(
        header(&recorded(&meta.mint)[0].1, "authorization"),
        Some("Bearer identity-token")
    );
    assert_eq!(refreshed.refresh, "identity-token");
    assert_eq!(refreshed.access, "LLM|fresh-key");
    assert!(refreshed.expires >= before + DAY_MS && refreshed.expires <= after + DAY_MS);
}

#[tokio::test]
async fn meta_reports_the_setup_url_when_meta_issues_no_key() {
    let meta = meta_server(
        meta_device_response(),
        vec![],
        vec![json_response(
            json!({"require_payment": true, "action_url": "https://dev.meta.ai/billing"}),
            200,
        )],
    )
    .await;
    let error = meta_oauth(&meta.server)
        .refresh(credential("", "identity-token", 1))
        .await
        .unwrap_err();
    assert_eq!(
        error.message(),
        "Meta did not issue an API key. Complete setup at https://dev.meta.ai/billing"
    );
}

#[tokio::test]
async fn meta_uses_the_minted_key_as_the_request_api_key() {
    let auth = MetaOAuth::default()
        .to_auth(&credential("LLM|key", "identity-token", 1))
        .await
        .unwrap();
    assert_eq!(auth.api_key.as_deref(), Some("LLM|key"));
    assert_eq!(auth.headers, None);
    assert_eq!(auth.base_url, None);
}

#[tokio::test]
async fn meta_reports_an_expired_session_and_mint_failures() {
    let meta = meta_server(
        meta_device_response(),
        vec![],
        vec![json_response(json!({"detail": " invalid token "}), 401)],
    )
    .await;
    let error = meta_oauth(&meta.server)
        .refresh(credential("", "identity-token", 1))
        .await
        .unwrap_err();
    assert_eq!(
        error.message(),
        "Meta session expired (status 401). Run `/login meta` to sign in again.: invalid token"
    );

    let meta = meta_server(
        meta_device_response(),
        vec![],
        vec![json_response(json!({"message": "boom"}), 500)],
    )
    .await;
    let error = meta_oauth(&meta.server)
        .refresh(credential("", "identity-token", 1))
        .await
        .unwrap_err();
    assert_eq!(error.message(), "Meta API key mint failed with status 500: boom");
}

#[tokio::test]
async fn meta_fails_on_denied_logins_and_invalid_device_responses() {
    let meta = meta_server(
        meta_device_response(),
        vec![json_response(json!({"error": "access_denied"}), 400)],
        vec![],
    )
    .await;
    let error = meta_oauth(&meta.server)
        .login(&TestInteraction::new(), &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.message(), "Meta login was denied.");

    let meta = meta_server(
        json_response(
            json!({"device_code": "d", "user_code": "u", "verification_uri": "file:///etc/passwd"}),
            200,
        ),
        vec![],
        vec![],
    )
    .await;
    let error = meta_oauth(&meta.server)
        .login(&TestInteraction::new(), &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        error.message(),
        r#"Invalid Meta device authorization response: {"device_code":"d","user_code":"u","verification_uri":"file:///etc/passwd"}"#
    );

    let meta = meta_server(
        json_response(
            json!({"error": "invalid_client", "error_description": "unknown client"}),
            400,
        ),
        vec![],
        vec![],
    )
    .await;
    let error = meta_oauth(&meta.server)
        .login(&TestInteraction::new(), &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        error.message(),
        "Meta device authorization failed with status 400: unknown client"
    );
}

#[tokio::test]
async fn meta_falls_back_to_verification_uri_and_cancels_while_waiting() {
    let meta = meta_server(
        json_response(
            json!({"device_code": "d", "user_code": "u", "verification_uri": "https://auth.meta.com/oauth/device/"}),
            200,
        ),
        vec![],
        vec![],
    )
    .await;
    let mut interaction = TestInteraction::new();
    let token = interaction.cancel.clone();
    interaction.on_event = Box::new(move |_| token.cancel());
    let error = meta_oauth(&meta.server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.message(), "Login cancelled");
    assert_eq!(error.code(), ModelsErrorCode::Aborted);
    assert_eq!(
        interaction.events(),
        [AuthEvent::DeviceCode {
            user_code: "u".into(),
            verification_uri: "https://auth.meta.com/oauth/device/".into(),
            interval_seconds: None,
            expires_in_seconds: None,
        }]
    );
    assert!(recorded(&meta.tokens).is_empty());
}

#[tokio::test]
async fn meta_requests_send_the_minted_key_as_the_api_key() {
    let models = models_with("meta", credential("LLM|minted", "identity", now_ms() + DAY_MS)).await;
    let server = MockServer::start().await;
    let completed = json!({"type": "response.completed", "response": {"id": "r", "status": "completed",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}});
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(format!("data: {completed}\n\n"), "text/event-stream"))
        .mount(&server)
        .await;
    let model = mock_model(&models, "meta", "https://api.meta.ai", &server);
    let message = models
        .complete(&model, &Context::new(vec![Message::user("hi")]), stream_options())
        .await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);

    let request = server.received_requests().await.unwrap().pop().unwrap();
    assert_eq!(request.url.path(), "/v1/responses");
    assert_eq!(header(&request, "authorization"), Some("Bearer LLM|minted"));
}
