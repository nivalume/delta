//! Tests of the Radius OAuth flow and the Radius provider. Ported from upstream
//! `test/radius-oauth.test.ts` and `test/radius-provider.test.ts`, plus device-code outcomes
//! and the request path through `Models`.
//!
//! The gateway is a wiremock server; browser callbacks are real loopback requests. Upstream
//! stubs `fetch` against the default gateway; here the provider points at the mock gateway
//! and gets the shipped catalog through `with_baseline_models`.

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

use delta_ai::auth::{AuthEvent, AuthInteraction, AuthPrompt, AuthType, LoginOptions, OAuthAuth};
use delta_ai::models::{CreateModelsOptions, InMemoryModelsStore};
use delta_ai::oauth::radius::{CALLBACK_PORT, CLIENT_ID, SCOPE};
use delta_ai::oauth::{RadiusOAuth, RadiusOAuthOptions, create_radius_oauth};
use delta_ai::provider::ModelsStoreEntry;
use delta_ai::providers::builtin::builtin_provider;
use delta_ai::providers::catalog::builtin_models;
use delta_ai::providers::radius::{
    DEFAULT_RADIUS_GATEWAY, RadiusGatewayConfig, RadiusProviderOptions, get_radius_models_from_config, radius_provider,
};
use delta_ai::types::now_ms;
use delta_ai::{
    AnyModel, ApiKeyCredential, Context, Credential, CredentialStore, InMemoryCredentialStore, JsonObject, Message,
    Models, ModelsError, ModelsErrorCode, ModelsRefreshOptions, ModelsStore, OAuthCredential, Provider, StopReason,
    StreamOptions,
};
use futures::future::BoxFuture;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn http() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

type NotifyFn = Arc<dyn Fn(&AuthEvent) + Send + Sync>;

/// Answers the login-method prompt with `method` and records events.
struct TestInteraction {
    method: String,
    on_event: NotifyFn,
    cancel: CancellationToken,
    events: Mutex<Vec<AuthEvent>>,
    prompts: Mutex<Vec<AuthPrompt>>,
}

impl TestInteraction {
    fn new(method: &str) -> Self {
        Self {
            method: method.to_owned(),
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

    fn events(&self) -> Vec<AuthEvent> {
        self.events.lock().unwrap().clone()
    }
}

impl AuthInteraction for TestInteraction {
    fn prompt(&self, prompt: AuthPrompt) -> BoxFuture<'_, Result<String, ModelsError>> {
        self.prompts.lock().unwrap().push(prompt);
        let method = self.method.clone();
        Box::pin(async move { Ok(method) })
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

fn form_field(request: &Request, name: &str) -> Option<String> {
    form_body(request)
        .into_iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
}

fn param(url: &url::Url, name: &str) -> Option<String> {
    url.query_pairs()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.into_owned())
}

fn oauth(server: &MockServer) -> RadiusOAuth {
    create_radius_oauth(RadiusOAuthOptions {
        name: "Radius".into(),
        gateway: server.uri(),
    })
}

/// Responds with `responses` in order, repeating the last one.
struct Sequence {
    responses: Vec<ResponseTemplate>,
    calls: AtomicUsize,
}

impl Sequence {
    fn new(responses: Vec<ResponseTemplate>) -> Self {
        Self {
            responses,
            calls: AtomicUsize::new(0),
        }
    }
}

impl Respond for Sequence {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        self.responses[call.min(self.responses.len() - 1)].clone()
    }
}

async fn free_port() -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    listener.local_addr().unwrap().port()
}

fn device_response() -> ResponseTemplate {
    json_response(
        json!({
            "device_code": "device-code",
            "user_code": "ABCD-1234",
            "verification_uri": "https://radius-ui.example/pair",
            "expires_in": 600,
            "interval": 1
        }),
        200,
    )
}

async fn device_server(token: impl Respond + 'static) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/oauth/device"))
        .respond_with(device_response())
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/oauth/token"))
        .respond_with(token)
        .mount(&server)
        .await;
    server
}

// ---------------------------------------------------------------------------
// radius-oauth.test.ts
// ---------------------------------------------------------------------------

#[tokio::test]
async fn uses_gateway_endpoints_directly_for_device_login() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/oauth/device"))
        .respond_with(json_response(
            json!({
                "device_code": "device-code",
                "user_code": "ABCD-1234",
                "verification_uri": "https://radius-ui.example/pair",
                "expires_in": 600,
                "interval": 5
            }),
            200,
        ))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/oauth/token"))
        .respond_with(json_response(
            json!({
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
                "scope": "gateway offline_access"
            }),
            200,
        ))
        .mount(&server)
        .await;

    let interaction = TestInteraction::new("device-code");
    let before = now_ms();
    let credential = oauth(&server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap();
    let after = now_ms();

    assert_eq!(credential.access, "access-token");
    assert_eq!(credential.refresh, "refresh-token");
    let lifetime = 3600 * 1000 - 60_000;
    assert!((before + lifetime..=after + lifetime).contains(&credential.expires));
    assert_eq!(
        serde_json::to_value(Credential::OAuth(credential)).unwrap()["scope"],
        "gateway offline_access"
    );
    assert_eq!(
        interaction.events(),
        [AuthEvent::DeviceCode {
            user_code: "ABCD-1234".into(),
            verification_uri: "https://radius-ui.example/pair".into(),
            interval_seconds: Some(5),
            expires_in_seconds: Some(600),
        }]
    );

    let requests = server.received_requests().await.unwrap();
    let paths: Vec<_> = requests.iter().map(|request| request.url.path().to_owned()).collect();
    assert_eq!(paths, ["/v1/oauth/device", "/v1/oauth/token"]);
    assert_eq!(form_field(&requests[0], "client_id").as_deref(), Some(CLIENT_ID));
    assert_eq!(form_field(&requests[0], "scope").as_deref(), Some(SCOPE));
    assert_eq!(
        form_field(&requests[1], "grant_type").as_deref(),
        Some("urn:ietf:params:oauth:grant-type:device_code")
    );
    assert_eq!(form_field(&requests[1], "client_id").as_deref(), Some("pi-gateway"));
    assert_eq!(form_field(&requests[1], "device_code").as_deref(), Some("device-code"));
}

#[tokio::test]
async fn refreshes_directly_through_the_gateway_without_discovery() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/oauth/token"))
        .respond_with(json_response(
            json!({"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600}),
            200,
        ))
        .mount(&server)
        .await;

    let refreshed = oauth(&server)
        .refresh(OAuthCredential {
            access: "old-access".into(),
            refresh: "old-refresh".into(),
            expires: 0,
            extra: JsonObject::new(),
        })
        .await
        .unwrap();

    assert_eq!(refreshed.access, "new-access");
    assert_eq!(refreshed.refresh, "new-refresh");
    assert!(refreshed.extra.is_empty());
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(form_field(&requests[0], "grant_type").as_deref(), Some("refresh_token"));
    assert_eq!(form_field(&requests[0], "client_id").as_deref(), Some("pi-gateway"));
    assert_eq!(
        form_field(&requests[0], "refresh_token").as_deref(),
        Some("old-refresh")
    );
}

#[tokio::test]
async fn discovers_only_the_interactive_browser_authorization_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/oauth"))
        .respond_with(json_response(json!({"issuer": "https://radius-ui.example"}), 200))
        .mount(&server)
        .await;

    let error = oauth(&server)
        .login(&TestInteraction::new("browser"), &LoginOptions::default())
        .await
        .unwrap_err();

    assert_eq!(
        error.message(),
        format!("Invalid Radius OAuth config from {}", server.uri())
    );
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

type PageSlot = Arc<Mutex<Option<JoinHandle<reqwest::Result<(u16, String)>>>>>;

struct BrowserLogin {
    result: Result<OAuthCredential, ModelsError>,
    page_status: u16,
    page_text: String,
    authorize_url: url::Url,
    token_request: Request,
}

async fn browser_login(token: ResponseTemplate) -> BrowserLogin {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/oauth"))
        .respond_with(json_response(
            json!({"authorizationEndpoint": format!("{}/authorize?stale=1", server.uri())}),
            200,
        ))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/oauth/token"))
        .respond_with(token)
        .mount(&server)
        .await;
    let mut oauth = oauth(&server);
    oauth.callback_port = free_port().await;

    let page: PageSlot = Arc::default();
    let slot = page.clone();
    let interaction = TestInteraction::new("browser").on_event(move |event| {
        let AuthEvent::AuthUrl { url, .. } = event else { return };
        let authorize = url::Url::parse(url).unwrap();
        let mut callback = url::Url::parse(&param(&authorize, "redirect_uri").unwrap()).unwrap();
        callback
            .query_pairs_mut()
            .append_pair("code", "browser-code")
            .append_pair("state", &param(&authorize, "state").unwrap());
        *slot.lock().unwrap() = Some(tokio::spawn(async move {
            let response = http().get(callback).send().await?;
            let status = response.status().as_u16();
            Ok((status, response.text().await?))
        }));
    });

    let result = oauth.login(&interaction, &LoginOptions::default()).await;
    let handle = page.lock().unwrap().take().unwrap();
    let (page_status, page_text) = handle.await.unwrap().unwrap();
    let events = interaction.events();
    assert!(matches!(&events[0], AuthEvent::Progress { message }
        if message == &format!("Listening for OAuth callback on http://127.0.0.1:{}/oauth/callback", oauth.callback_port)));
    let authorize_url = events
        .iter()
        .find_map(|event| match event {
            AuthEvent::AuthUrl { url, instructions } => {
                assert_eq!(instructions.as_deref(), Some("Continue in your browser."));
                Some(url::Url::parse(url).unwrap())
            }
            _ => None,
        })
        .unwrap();
    let token_request = server
        .received_requests()
        .await
        .unwrap()
        .into_iter()
        .find(|request| request.url.path() == "/v1/oauth/token")
        .unwrap();
    BrowserLogin {
        result,
        page_status,
        page_text,
        authorize_url,
        token_request,
    }
}

#[tokio::test]
async fn exchanges_the_browser_callback_code_before_showing_the_sign_in_page() {
    let failed = browser_login(json_response(
        json!({"error": "invalid_grant", "error_description": "code expired"}),
        400,
    ))
    .await;
    let error = failed.result.unwrap_err();
    assert_eq!(
        error.message(),
        "Radius OAuth token request failed: invalid_grant: code expired"
    );
    assert_eq!(failed.page_status, 502);
    assert!(failed.page_text.contains("code expired"));

    let succeeded = browser_login(json_response(
        json!({"access_token": "access", "refresh_token": "refresh", "expires_in": 3600}),
        200,
    ))
    .await;
    assert_eq!(succeeded.result.unwrap().access, "access");
    assert_eq!(succeeded.page_status, 200);
    assert!(succeeded.page_text.contains("Signed in to Radius."));

    // The authorize URL replaces the endpoint's query with the PKCE request.
    let url = &succeeded.authorize_url;
    assert_eq!(url.path(), "/authorize");
    assert_eq!(param(url, "stale"), None);
    assert_eq!(param(url, "response_type").as_deref(), Some("code"));
    assert_eq!(param(url, "client_id").as_deref(), Some("pi-gateway"));
    assert_eq!(param(url, "scope").as_deref(), Some("gateway offline_access"));
    assert_eq!(param(url, "code_challenge_method").as_deref(), Some("S256"));
    assert_eq!(param(url, "handoff").as_deref(), Some("url"));
    let redirect_uri = param(url, "redirect_uri").unwrap();
    let request = &succeeded.token_request;
    assert_eq!(form_field(request, "grant_type").as_deref(), Some("authorization_code"));
    assert_eq!(form_field(request, "code").as_deref(), Some("browser-code"));
    assert_eq!(form_field(request, "redirect_uri"), Some(redirect_uri));
    let verifier = form_field(request, "code_verifier").unwrap();
    assert_eq!(
        param(url, "code_challenge").unwrap(),
        delta_ai::oauth::pkce::pkce_challenge(&verifier)
    );
}

// ---------------------------------------------------------------------------
// Further OAuth cases
// ---------------------------------------------------------------------------

#[test]
fn uses_the_fixed_callback_port_and_normalizes_the_gateway() {
    let oauth = create_radius_oauth(RadiusOAuthOptions {
        name: "Radius Dev".into(),
        gateway: "localhost:8788/".into(),
    });
    assert_eq!(oauth.callback_port, CALLBACK_PORT);
    assert_eq!(oauth.gateway(), "https://localhost:8788");
    assert_eq!(oauth.name(), "Radius Dev");
    assert!(!oauth.is_subscription());
    assert_eq!(oauth.login_label(), None);
}

#[tokio::test]
async fn offers_browser_and_device_code_sign_in() {
    let server = MockServer::start().await;
    let interaction = TestInteraction::new("password");
    let error = create_radius_oauth(RadiusOAuthOptions {
        name: "Acme Gateway".into(),
        gateway: server.uri(),
    })
    .login(&interaction, &LoginOptions::default())
    .await
    .unwrap_err();
    assert_eq!(error.message(), "Unknown Acme Gateway sign-in method: password");
    let prompts = interaction.prompts.lock().unwrap().clone();
    let AuthPrompt::Select { message, options } = &prompts[0] else {
        panic!("expected a select prompt");
    };
    assert_eq!(message, "Sign in to Acme Gateway:");
    let options: Vec<_> = options
        .iter()
        .map(|option| (option.id.as_str(), option.label.as_str()))
        .collect();
    assert_eq!(
        options,
        [
            ("browser", "Sign in with browser (recommended)"),
            (
                "device-code",
                "Sign in with device code (when signing in from another device)"
            )
        ]
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn polls_until_the_device_is_approved() {
    let server = device_server(Sequence::new(vec![
        json_response(json!({"error": "authorization_pending"}), 400),
        json_response(
            json!({"access_token": "access", "refresh_token": "refresh", "expires_in": 60}),
            200,
        ),
    ]))
    .await;
    let credential = oauth(&server)
        .login(&TestInteraction::new("device-code"), &LoginOptions::default())
        .await
        .unwrap();
    assert_eq!(credential.access, "access");
    let token_requests = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|request| request.url.path() == "/v1/oauth/token")
        .count();
    assert_eq!(token_requests, 2);
}

#[tokio::test]
async fn maps_device_code_failures() {
    for (body, message) in [
        (json!({"error": "access_denied"}), "Device authorization was denied."),
        (json!({"error": "expired_token"}), "Device authorization expired."),
        (
            json!({"error": "invalid_client", "error_description": "unknown client"}),
            "Radius OAuth token request failed: invalid_client: unknown client",
        ),
    ] {
        let server = device_server(json_response(body, 400)).await;
        let error = oauth(&server)
            .login(&TestInteraction::new("device-code"), &LoginOptions::default())
            .await
            .unwrap_err();
        assert_eq!(error.message(), message);
    }
}

#[tokio::test]
async fn rejects_incomplete_device_authorizations() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/oauth/device"))
        .respond_with(json_response(
            json!({"device_code": "d", "user_code": "u", "expires_in": 600}),
            200,
        ))
        .mount(&server)
        .await;
    let error = oauth(&server)
        .login(&TestInteraction::new("device-code"), &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        error.message(),
        "Radius OAuth device authorization response is missing required fields"
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/oauth/device"))
        .respond_with(json_response(json!({"error": "invalid_scope"}), 400))
        .mount(&server)
        .await;
    let error = oauth(&server)
        .login(&TestInteraction::new("device-code"), &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        error.message(),
        "Radius OAuth device authorization failed: invalid_scope"
    );
}

#[tokio::test]
async fn reports_discovery_failures_and_cancellation() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/oauth"))
        .respond_with(ResponseTemplate::new(503).set_body_string("down"))
        .mount(&server)
        .await;
    let error = oauth(&server)
        .login(&TestInteraction::new("browser"), &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        error.message(),
        format!("Could not load Radius OAuth config from {}: 503 down", server.uri())
    );

    let interaction = TestInteraction::new("device-code");
    interaction.cancel.cancel();
    let error = oauth(&server)
        .login(&interaction, &LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.code(), ModelsErrorCode::Aborted);
    assert_eq!(error.message(), "Login cancelled");
}

// ---------------------------------------------------------------------------
// radius-provider.test.ts
// ---------------------------------------------------------------------------

fn radius_config(base_url: &str) -> Value {
    json!({
        "baseUrl": base_url,
        "models": [
            {
                "id": "balanced",
                "name": "Fresh Balanced",
                "reasoning": true,
                "input": ["text"],
                "cost": {"input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 0},
                "contextWindow": 424_242,
                "maxTokens": 32000
            },
            {
                "id": "organization-only",
                "name": "Organization Only",
                "reasoning": false,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 128_000,
                "maxTokens": 16000
            }
        ]
    })
}

async fn config_server(config: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/config"))
        .respond_with(json_response(config, 200))
        .mount(&server)
        .await;
    server
}

/// The default provider's shipped catalog on a mock gateway.
fn provider_on(server: &MockServer) -> delta_ai::providers::radius::RadiusProvider {
    radius_provider(RadiusProviderOptions {
        gateway: Some(server.uri()),
        ..RadiusProviderOptions::default()
    })
    .with_baseline_models(builtin_models("radius"))
}

async fn models_with_credential(credential: Credential) -> Models {
    let credentials = Arc::new(InMemoryCredentialStore::new());
    credentials
        .modify(
            "radius",
            Box::new(move |_| Box::pin(async move { Ok(Some(credential)) })),
        )
        .await
        .unwrap();
    Models::new(CreateModelsOptions {
        credentials: Some(credentials),
        ..CreateModelsOptions::default()
    })
}

fn api_key(key: &str) -> Credential {
    Credential::ApiKey(ApiKeyCredential {
        key: Some(key.into()),
        env: None,
    })
}

fn ids(models: &Models) -> Vec<String> {
    models
        .get_models(Some("radius"))
        .into_iter()
        .map(|model| model.id)
        .collect()
}

fn refresh_radius() -> ModelsRefreshOptions {
    ModelsRefreshOptions {
        providers: Some(vec!["radius".into()]),
        ..ModelsRefreshOptions::default()
    }
}

#[test]
fn ships_a_static_public_catalog_for_the_default_gateway() {
    let provider = radius_provider(RadiusProviderOptions::default());
    assert_eq!(provider.gateway(), DEFAULT_RADIUS_GATEWAY);
    let models = provider.models();
    assert!(!models.is_empty());
    assert!(
        models
            .iter()
            .any(|model| model.id == "balanced" && model.provider == "radius" && model.api == "pi-messages")
    );
    assert!(provider.is_dynamic());

    // A renamed provider owns the shipped models under its id.
    let renamed = radius_provider(RadiusProviderOptions {
        id: Some("radius-eu".into()),
        gateway: Some("radius.pi.dev/".into()),
        ..RadiusProviderOptions::default()
    });
    assert!(renamed.models().iter().all(|model| model.provider == "radius-eu"));
}

#[test]
fn does_not_apply_the_public_radius_catalog_to_custom_gateways() {
    let provider = radius_provider(RadiusProviderOptions {
        id: Some("radius-dev".into()),
        gateway: Some("http://localhost:8788".into()),
        ..RadiusProviderOptions::default()
    });
    assert!(provider.models().is_empty());
}

#[tokio::test]
async fn replaces_the_static_public_catalog_with_refreshed_models() {
    let server = config_server(radius_config("https://radius.example/v1")).await;
    let models = models_with_credential(api_key("radius-key")).await;
    models.set_provider(provider_on(&server));
    assert!(ids(&models).len() > 2);

    let result = models.refresh(refresh_radius()).await;

    assert!(result.errors.is_empty(), "{:?}", result.errors);
    let balanced = models.get_model("radius", "balanced").unwrap();
    assert_eq!(balanced.name, "Fresh Balanced");
    assert_eq!(balanced.base_url, "https://radius.example/v1");
    assert_eq!(balanced.context_window, 424_242);
    assert!(models.get_model("radius", "organization-only").is_some());
    // Models disabled by the organization must not reappear from the shipped catalog.
    assert_eq!(ids(&models), ["balanced", "organization-only"]);

    let request = &server.received_requests().await.unwrap()[0];
    assert_eq!(request.headers["authorization"], "Bearer radius-key");
    assert_eq!(request.headers["accept"], "application/json");
}

#[tokio::test]
async fn replaces_the_static_public_catalog_with_a_cached_catalog_without_network_access() {
    let config = RadiusGatewayConfig {
        base_url: "https://radius.example/v1".into(),
        models: radius_config("")["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|model| model.as_object().unwrap().clone())
            .collect(),
    };
    let store = Arc::new(InMemoryModelsStore::default());
    store
        .write(
            "radius",
            ModelsStoreEntry {
                models: get_radius_models_from_config("radius", &config)
                    .into_iter()
                    .map(AnyModel::Chat)
                    .collect(),
                checked_at: Some(now_ms()),
                ..ModelsStoreEntry::default()
            },
        )
        .await
        .unwrap();
    let models = Models::new(CreateModelsOptions {
        models_store: Some(store),
        ..CreateModelsOptions::default()
    });
    models.set_provider(radius_provider(RadiusProviderOptions::default()));

    models
        .refresh(ModelsRefreshOptions {
            allow_network: Some(false),
            ..refresh_radius()
        })
        .await;

    assert_eq!(models.get_model("radius", "balanced").unwrap().name, "Fresh Balanced");
    assert!(models.get_model("radius", "organization-only").is_some());
    assert_eq!(ids(&models), ["balanced", "organization-only"]);
}

#[tokio::test]
async fn exposes_no_models_when_the_organization_disabled_all_of_them() {
    let server = config_server(json!({"baseUrl": "https://radius.example/v1", "models": []})).await;
    let models = models_with_credential(api_key("radius-key")).await;
    models.set_provider(provider_on(&server));

    let result = models.refresh(refresh_radius()).await;

    assert!(result.errors.is_empty(), "{:?}", result.errors);
    assert!(models.get_models(Some("radius")).is_empty());
}

// ---------------------------------------------------------------------------
// Further provider cases
// ---------------------------------------------------------------------------

#[tokio::test]
async fn reports_config_failures_and_keeps_the_previous_catalog() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/config"))
        .respond_with(ResponseTemplate::new(403).set_body_string(format!("  {}  ", "x".repeat(600))))
        .mount(&server)
        .await;
    let models = models_with_credential(api_key("radius-key")).await;
    models.set_provider(provider_on(&server));
    let before = ids(&models);

    let result = models.refresh(refresh_radius()).await;

    let error = &result.errors["radius"];
    assert_eq!(error.code(), ModelsErrorCode::ModelSource);
    assert_eq!(
        error.message(),
        format!(
            "Could not load Radius config from {}: 403: {}…",
            server.uri(),
            "x".repeat(512)
        )
    );
    assert_eq!(ids(&models), before);

    let server = config_server(json!({"models": []})).await;
    models.set_provider(provider_on(&server));
    let result = models.refresh(refresh_radius()).await;
    assert_eq!(
        result.errors["radius"].message(),
        format!("Invalid Radius config from {}", server.uri())
    );
}

#[tokio::test]
async fn imports_the_legacy_credential_catalog_and_uses_the_oauth_access_token() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/config"))
        .and(header("authorization", "Bearer oauth-access"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    let credential: OAuthCredential = serde_json::from_value(json!({
        "access": "oauth-access",
        "refresh": "oauth-refresh",
        "expires": now_ms() + 3_600_000,
        "gatewayConfig": radius_config("https://legacy.example/v1")
    }))
    .unwrap();
    let store = Arc::new(InMemoryModelsStore::default());
    let credentials = Arc::new(InMemoryCredentialStore::new());
    credentials
        .modify(
            "radius",
            Box::new(move |_| Box::pin(async move { Ok(Some(Credential::OAuth(credential))) })),
        )
        .await
        .unwrap();
    let models = Models::new(CreateModelsOptions {
        credentials: Some(credentials),
        models_store: Some(store.clone()),
        ..CreateModelsOptions::default()
    });
    models.set_provider(provider_on(&server));

    let result = models.refresh(refresh_radius()).await;

    // The gateway failed, but the legacy catalog was imported and persisted first.
    assert!(result.errors.contains_key("radius"));
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
    assert_eq!(ids(&models), ["balanced", "organization-only"]);
    assert_eq!(
        models.get_model("radius", "balanced").unwrap().base_url,
        "https://legacy.example/v1"
    );
    let stored = store.read("radius").await.unwrap().unwrap();
    assert_eq!(stored.models.len(), 2);
}

#[tokio::test]
async fn is_built_in_with_api_key_and_oauth_auth() {
    let provider = builtin_provider("radius").unwrap();
    assert_eq!(provider.name(), "Radius");
    assert_eq!(provider.auth().api_key.as_ref().unwrap().name(), "Radius API key");
    assert_eq!(provider.auth().oauth.as_ref().unwrap().name(), "Radius");
    assert!(delta_ai::providers::builtin::builtin_provider_ids().contains(&"radius"));

    let oauth = provider.auth().oauth.clone().unwrap();
    let credential = OAuthCredential {
        access: "access".into(),
        refresh: "refresh".into(),
        expires: 0,
        extra: JsonObject::new(),
    };
    assert_eq!(
        oauth.to_auth(&credential).await.unwrap().api_key.as_deref(),
        Some("access")
    );
}

#[tokio::test]
async fn streams_through_models_with_the_oauth_access_token() {
    // The gateway serves the config and the pi-messages endpoint.
    let server = MockServer::start().await;
    let config = radius_config(&format!("{}/v1", server.uri()));
    Mock::given(method("GET"))
        .and(path("/v1/config"))
        .respond_with(json_response(config, 200))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("authorization", "Bearer oauth-access"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            format!(
                "data: {}\n\ndata: {}\n\n",
                json!({"type": "text_start", "contentIndex": 0}),
                json!({"type": "done", "reason": "stop", "usage": {"input": 1, "output": 1, "cacheRead": 0,
                       "cacheWrite": 0, "totalTokens": 2, "cost": {"input": 0.0, "output": 0.0, "cacheRead": 0.0,
                       "cacheWrite": 0.0, "total": 0.0}}})
            ),
            "text/event-stream",
        ))
        .mount(&server)
        .await;
    let models = models_with_credential(Credential::OAuth(OAuthCredential {
        access: "oauth-access".into(),
        refresh: "oauth-refresh".into(),
        expires: now_ms() + 3_600_000,
        extra: JsonObject::new(),
    }))
    .await;
    models.set_provider(provider_on(&server));
    let result = models.refresh(refresh_radius()).await;
    assert!(result.errors.is_empty(), "{:?}", result.errors);
    assert!(
        models
            .check_auth("radius")
            .await
            .unwrap()
            .is_some_and(|check| check.auth_type == AuthType::OAuth)
    );

    let model = models.get_model("radius", "organization-only").unwrap();
    let message = models
        .complete(
            &model,
            &Context::new(vec![Message::user("Hello")]),
            StreamOptions {
                http_client: Some(http()),
                ..StreamOptions::default()
            },
        )
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    assert_eq!(message.provider, "radius");
    assert_eq!(message.model, "organization-only");
}
