//! Anthropic workload identity federation: auth resolution, the jwt-bearer token exchange,
//! token caching and refresh, against wiremock servers for the token endpoint and the
//! Messages endpoint.
//!
//! Ported from upstream `test/anthropic-federation.test.ts` and
//! `test/anthropic-federation-sdk.test.ts`, plus cases for the parts of the Anthropic SDK this
//! crate ports (`lib/credentials/oidc-federation.ts`, `token-cache.ts`).

// Test helpers panic on unexpected shapes by design.
#![allow(clippy::unwrap_used)]

use delta_ai::api::anthropic_federation::{AnthropicFederationConfig, federation_token_cache};
use delta_ai::api::anthropic_messages::stream;
use delta_ai::auth::{ApiKeyInput, AuthContext};
use delta_ai::providers::builtin::anthropic_provider;
use delta_ai::types::ProviderEnv;
use delta_ai::{
    AssistantMessage, AuthResult, Context, CreateModelsOptions, Message, Model, ModelAuth, Models, Provider,
    SimpleStreamOptions, StopReason, StreamOptions, UserMessage,
};
use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// Tests that federate share the process-wide token cache slot, which keeps one config at a
/// time (as upstream keeps one SDK client). They run one at a time so a test's cache is not
/// replaced by another test's config halfway through.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

const OFFLINE: &str = "http://127.0.0.1:9";
const TOKEN_PATH: &str = "/v1/oauth/token";
const MESSAGES_PATH: &str = "/v1/messages";

fn federation_env(token_file: &str) -> ProviderEnv {
    [
        ("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_test"),
        ("ANTHROPIC_ORGANIZATION_ID", "org-test"),
        ("ANTHROPIC_SERVICE_ACCOUNT_ID", "svac_test"),
        ("ANTHROPIC_IDENTITY_TOKEN_FILE", token_file),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_owned(), value.to_owned()))
    .collect()
}

fn without(env: &ProviderEnv, name: &str) -> ProviderEnv {
    let mut env = env.clone();
    env.remove(name);
    env
}

fn with(env: &ProviderEnv, name: &str, value: &str) -> ProviderEnv {
    let mut env = env.clone();
    env.insert(name.into(), value.into());
    env
}

fn model(base_url: &str, provider: &str) -> Model {
    serde_json::from_value(json!({
        "id": "claude-test", "name": "Claude Test", "api": "anthropic-messages", "provider": provider,
        "baseUrl": base_url, "reasoning": false, "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 100_000, "maxTokens": 4096
    }))
    .unwrap()
}

fn context() -> Context {
    let mut context = Context::new(vec![Message::User(UserMessage {
        content: "Hello".into(),
        timestamp: 1,
    })]);
    context.system_prompt = Some("System prompt.".into());
    context
}

/// A client per test: the shared default client's pool must not outlive a test's runtime.
fn client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

fn options(env: ProviderEnv) -> StreamOptions {
    StreamOptions {
        env: Some(env),
        http_client: Some(client()),
        ..StreamOptions::default()
    }
}

async fn run(model: &Model, options: StreamOptions) -> AssistantMessage {
    stream(model, &delta_ai::normalize_context(&context()), options)
        .result()
        .await
}

/// A temporary identity token file, removed on drop.
struct TokenFile {
    dir: PathBuf,
    path: PathBuf,
}

impl TokenFile {
    fn new(content: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "delta-anthropic-federation-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("identity.jwt");
        std::fs::write(&path, content).unwrap();
        Self { dir, path }
    }

    fn path(&self) -> &str {
        self.path.to_str().unwrap()
    }

    fn write(&self, content: &str) {
        std::fs::write(&self.path, content).unwrap();
    }
}

impl Drop for TokenFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// SSE body like upstream's `createSseResponse`.
fn sse_body() -> String {
    [
        json!({"type": "message_start", "message": {"id": "msg_test", "usage": {"input_tokens": 1, "output_tokens": 0}}}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
        json!({"type": "message_stop"}),
    ]
    .iter()
    .fold(String::new(), |mut body, event| {
        let _ = write!(body, "event: {}\ndata: {event}\n\n", event["type"].as_str().unwrap());
        body
    })
}

/// Token endpoint answers: `federated-token-<n>` for the n-th exchange (from 1), with the
/// lifetime `expires_in[n - 1]` (the last entry repeats).
struct TokenResponder {
    calls: AtomicUsize,
    expires_in: Vec<u64>,
}

impl Respond for TokenResponder {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let expires_in = self.expires_in[call.min(self.expires_in.len() - 1)];
        ResponseTemplate::new(200).set_body_json(json!({
            "access_token": format!("federated-token-{}", call + 1),
            "expires_in": expires_in,
        }))
    }
}

/// A server with a token endpoint (tokens live `expires_in` seconds) and a Messages endpoint.
async fn serve(expires_in: &[u64]) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(TOKEN_PATH))
        .respond_with(TokenResponder {
            calls: AtomicUsize::new(0),
            expires_in: expires_in.to_vec(),
        })
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(MESSAGES_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body(), "text/event-stream"))
        .mount(&server)
        .await;
    server
}

/// A server whose token endpoint answers with `response`.
async fn serve_token(response: ResponseTemplate) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(TOKEN_PATH))
        .respond_with(response)
        .mount(&server)
        .await;
    server
}

async fn requests_to(server: &MockServer, path: &str) -> Vec<Request> {
    server
        .received_requests()
        .await
        .unwrap()
        .into_iter()
        .filter(|request| request.url.path() == path)
        .collect()
}

fn header<'a>(request: &'a Request, name: &str) -> Option<&'a str> {
    request.headers.get(name).map(|value| value.to_str().unwrap())
}

fn authorizations(requests: &[Request]) -> Vec<Option<&str>> {
    requests
        .iter()
        .map(|request| header(request, "authorization"))
        .collect()
}

struct Env(BTreeMap<String, String>);

impl AuthContext for Env {
    fn env(&self, name: &str) -> Option<String> {
        self.0.get(name).cloned()
    }

    fn file_exists(&self, _: &str) -> bool {
        false
    }
}

async fn resolve_with_env(env: &ProviderEnv) -> Option<AuthResult> {
    let provider = anthropic_provider();
    let api_key = provider.auth().api_key.clone().unwrap();
    api_key
        .resolve(ApiKeyInput {
            ctx: &Env(env.clone()),
            credential: None,
        })
        .await
        .unwrap()
}

fn federation_result(env: ProviderEnv) -> AuthResult {
    AuthResult {
        auth: ModelAuth::default(),
        env: Some(env),
        source: Some("workload identity federation".into()),
    }
}

// ---------------------------------------------------------------------------
// anthropic-federation.test.ts
// ---------------------------------------------------------------------------

#[tokio::test]
async fn resolves_the_federation_variables_as_provider_env_with_no_request_auth() {
    let env = federation_env("/tmp/identity.jwt");
    assert_eq!(resolve_with_env(&env).await, Some(federation_result(env)));
}

#[tokio::test]
async fn passes_anthropic_workspace_id_through_when_set() {
    let env = with(
        &federation_env("/tmp/identity.jwt"),
        "ANTHROPIC_WORKSPACE_ID",
        "wrkspc_test",
    );
    let result = resolve_with_env(&env).await.unwrap();
    assert_eq!(
        result.env.unwrap().get("ANTHROPIC_WORKSPACE_ID").map(String::as_str),
        Some("wrkspc_test")
    );
}

#[tokio::test]
async fn is_not_configured_when_a_federation_variable_is_missing() {
    let env = federation_env("/tmp/identity.jwt");
    for name in [
        "ANTHROPIC_IDENTITY_TOKEN_FILE",
        "ANTHROPIC_FEDERATION_RULE_ID",
        "ANTHROPIC_ORGANIZATION_ID",
    ] {
        assert_eq!(resolve_with_env(&without(&env, name)).await, None, "{name}");
    }
}

#[tokio::test]
async fn treats_anthropic_service_account_id_as_optional_like_the_sdk() {
    let _serial = SERIAL.lock().await;
    let token = TokenFile::new("header.payload.signature");
    let partial = without(&federation_env(token.path()), "ANTHROPIC_SERVICE_ACCOUNT_ID");
    assert_eq!(
        resolve_with_env(&partial).await,
        Some(federation_result(partial.clone()))
    );

    let server = serve(&[3600]).await;
    let message = run(&model(&server.uri(), "anthropic"), options(partial)).await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    let exchanges = requests_to(&server, TOKEN_PATH).await;
    assert_eq!(
        exchanges[0].body_json::<Value>().unwrap(),
        json!({
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": "header.payload.signature",
            "federation_rule_id": "fdrl_test",
            "organization_id": "org-test",
        })
    );
}

#[tokio::test]
async fn keeps_api_key_and_auth_token_precedence_over_federation() {
    let env = federation_env("/tmp/identity.jwt");
    assert_eq!(
        resolve_with_env(&with(&env, "ANTHROPIC_API_KEY", "api-key")).await,
        Some(AuthResult {
            auth: ModelAuth {
                api_key: Some("api-key".into()),
                ..ModelAuth::default()
            },
            env: None,
            source: Some("ANTHROPIC_API_KEY".into()),
        })
    );
    let result = resolve_with_env(&with(&env, "ANTHROPIC_AUTH_TOKEN", "auth-token"))
        .await
        .unwrap();
    assert_eq!(result.source.as_deref(), Some("ANTHROPIC_AUTH_TOKEN"));
    assert_eq!(result.auth.api_key, None);
    assert_eq!(
        result
            .auth
            .headers
            .unwrap()
            .get("Authorization")
            .cloned()
            .flatten()
            .as_deref(),
        Some("Bearer auth-token")
    );
    assert_eq!(result.env, None);
}

/// Upstream checks the config handed to the SDK; here the exchange and the federated request
/// are checked on the wire.
#[tokio::test]
async fn sends_a_federated_access_token_instead_of_a_key() {
    let _serial = SERIAL.lock().await;
    let token = TokenFile::new("header.payload.signature\n");
    let server = serve(&[3600]).await;
    let env = with(&federation_env(token.path()), "ANTHROPIC_WORKSPACE_ID", "wrkspc_test");
    let message = run(&model(&server.uri(), "anthropic"), options(env)).await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);

    let exchanges = requests_to(&server, TOKEN_PATH).await;
    assert_eq!(exchanges.len(), 1);
    let exchange = &exchanges[0];
    assert_eq!(
        exchange.body_json::<Value>().unwrap(),
        json!({
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": "header.payload.signature",
            "federation_rule_id": "fdrl_test",
            "organization_id": "org-test",
            "service_account_id": "svac_test",
            "workspace_id": "wrkspc_test",
        })
    );
    assert_eq!(header(exchange, "content-type"), Some("application/json"));
    assert_eq!(
        header(exchange, "anthropic-beta"),
        Some("oauth-2025-04-20,oidc-federation-2026-04-01")
    );
    assert!(header(exchange, "user-agent").unwrap().starts_with("delta/"));
    assert_eq!(header(exchange, "authorization"), None);

    let messages = requests_to(&server, MESSAGES_PATH).await;
    let request = &messages[0];
    assert_eq!(header(request, "authorization"), Some("Bearer federated-token-1"));
    assert_eq!(header(request, "x-api-key"), None);
    assert_eq!(header(request, "anthropic-version"), Some("2023-06-01"));
    // Not an OAuth (Claude Code) request: only the SDK's token beta is added.
    assert_eq!(header(request, "anthropic-beta"), Some("oauth-2025-04-20"));
    assert_eq!(header(request, "x-app"), None);
    // The workspace is in the token; the SDK sends no workspace header for federation.
    assert_eq!(header(request, "anthropic-workspace-id"), None);
    let body = request.body_json::<Value>().unwrap();
    assert_eq!(body["system"][0]["text"], "System prompt.");
}

#[tokio::test]
async fn appends_the_token_beta_to_configured_betas() {
    let _serial = SERIAL.lock().await;
    let token = TokenFile::new("jwt");
    let server = serve(&[3600]).await;
    let mut options = options(federation_env(token.path()));
    options.headers = Some(
        [(
            "anthropic-beta".to_owned(),
            Some("files-api-2025-04-14, interleaved-thinking-2025-05-14".to_owned()),
        )]
        .into_iter()
        .collect(),
    );
    let message = run(&model(&server.uri(), "anthropic"), options).await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    let messages = requests_to(&server, MESSAGES_PATH).await;
    assert_eq!(
        header(&messages[0], "anthropic-beta"),
        Some("files-api-2025-04-14,interleaved-thinking-2025-05-14,oauth-2025-04-20")
    );
}

#[tokio::test]
async fn threads_auth_context_federation_variables_through_models() {
    let _serial = SERIAL.lock().await;
    let token = TokenFile::new("jwt");
    let server = serve(&[3600]).await;
    let models = Models::new(CreateModelsOptions {
        auth_context: Some(Arc::new(Env(federation_env(token.path())))),
        ..CreateModelsOptions::default()
    });
    models.set_provider(anthropic_provider());
    let options = SimpleStreamOptions {
        base: StreamOptions {
            http_client: Some(client()),
            ..StreamOptions::default()
        },
        ..SimpleStreamOptions::default()
    };
    let message = models
        .complete_simple(&model(&server.uri(), "anthropic"), &context(), options)
        .await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    assert_eq!(requests_to(&server, TOKEN_PATH).await.len(), 1);
    let messages = requests_to(&server, MESSAGES_PATH).await;
    assert_eq!(authorizations(&messages), [Some("Bearer federated-token-1")]);
}

#[tokio::test]
async fn lets_an_explicit_api_key_win_over_federation_env() {
    let token = TokenFile::new("jwt");
    let server = serve(&[3600]).await;
    let mut options = options(federation_env(token.path()));
    options.api_key = Some("explicit-key".into());
    let message = run(&model(&server.uri(), "anthropic"), options).await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    assert!(requests_to(&server, TOKEN_PATH).await.is_empty());
    let messages = requests_to(&server, MESSAGES_PATH).await;
    assert_eq!(header(&messages[0], "x-api-key"), Some("explicit-key"));
    assert_eq!(header(&messages[0], "authorization"), None);
}

#[tokio::test]
async fn does_not_federate_other_anthropic_messages_providers() {
    let message = run(
        &model(OFFLINE, "kimi-coding"),
        options(federation_env("/tmp/identity.jwt")),
    )
    .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("No API key for provider: kimi-coding")
    );
}

// ---------------------------------------------------------------------------
// anthropic-federation-sdk.test.ts
// ---------------------------------------------------------------------------

#[tokio::test]
async fn exchanges_the_identity_token_once_across_requests() {
    let _serial = SERIAL.lock().await;
    let token = TokenFile::new("header.payload.signature");
    let server = serve(&[3600]).await;
    let model = model(&server.uri(), "anthropic");
    for _ in 0..3 {
        let message = run(&model, options(federation_env(token.path()))).await;
        assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    }
    assert_eq!(requests_to(&server, TOKEN_PATH).await.len(), 1);
    let messages = requests_to(&server, MESSAGES_PATH).await;
    assert_eq!(authorizations(&messages), [Some("Bearer federated-token-1"); 3]);
}

/// Upstream stubs the federation variables in the process environment; this test passes
/// them in `options.env`, which the federation lookup reads first.
#[tokio::test]
async fn does_not_federate_header_owned_auth() {
    let token = TokenFile::new("jwt");
    let server = serve(&[3600]).await;
    let mut options = options(federation_env(token.path()));
    options.headers = Some(
        [("Authorization".to_owned(), Some("Bearer auth-token".to_owned()))]
            .into_iter()
            .collect(),
    );
    let message = run(&model(&server.uri(), "anthropic"), options).await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    let requests = server.received_requests().await.unwrap();
    let seen: Vec<(&str, Option<&str>)> = requests
        .iter()
        .map(|request| (request.url.path(), header(request, "authorization")))
        .collect();
    assert_eq!(seen, [(MESSAGES_PATH, Some("Bearer auth-token"))]);
    assert_eq!(header(&requests[0], "anthropic-beta"), None);
}

// ---------------------------------------------------------------------------
// The SDK's token cache (token-cache.ts) and exchange (oidc-federation.ts)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn exchanges_again_when_the_token_is_about_to_expire() {
    let _serial = SERIAL.lock().await;
    let token = TokenFile::new("first-jwt");
    // Under 30 s left: every request waits for a new token.
    let server = serve(&[10]).await;
    let model = model(&server.uri(), "anthropic");
    run(&model, options(federation_env(token.path()))).await;
    // The identity token file is read on every exchange, so a rotated token is used.
    token.write("second-jwt");
    run(&model, options(federation_env(token.path()))).await;

    let exchanges = requests_to(&server, TOKEN_PATH).await;
    let assertions: Vec<Value> = exchanges
        .iter()
        .map(|request| request.body_json::<Value>().unwrap()["assertion"].clone())
        .collect();
    assert_eq!(assertions, [json!("first-jwt"), json!("second-jwt")]);
    let messages = requests_to(&server, MESSAGES_PATH).await;
    assert_eq!(
        authorizations(&messages),
        [Some("Bearer federated-token-1"), Some("Bearer federated-token-2")]
    );
}

#[tokio::test]
async fn refreshes_in_the_background_inside_the_advisory_window() {
    let _serial = SERIAL.lock().await;
    let token = TokenFile::new("jwt");
    // The first token has 60 s left (30 to 120 s: advisory window), later ones an hour.
    let server = serve(&[60, 3600]).await;
    let model = model(&server.uri(), "anthropic");
    let env = federation_env(token.path());
    run(&model, options(env.clone())).await;
    // Served from the cache while a background exchange runs.
    run(&model, options(env.clone())).await;

    let config = AnthropicFederationConfig {
        organization_id: "org-test".into(),
        workspace_id: None,
        federation_rule_id: "fdrl_test".into(),
        service_account_id: Some("svac_test".into()),
        identity_token_file: token.path().into(),
    };
    let cache = federation_token_cache(&server.uri(), &config);
    let refreshed = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if cache.get_token(&client()).await.unwrap() == "federated-token-2" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert!(refreshed.is_ok(), "the background refresh did not finish");

    run(&model, options(env)).await;
    assert_eq!(requests_to(&server, TOKEN_PATH).await.len(), 2);
    let messages = requests_to(&server, MESSAGES_PATH).await;
    assert_eq!(
        authorizations(&messages),
        [
            Some("Bearer federated-token-1"),
            Some("Bearer federated-token-1"),
            Some("Bearer federated-token-2")
        ]
    );
}

#[tokio::test]
async fn drops_the_token_after_a_401() {
    let _serial = SERIAL.lock().await;
    let token = TokenFile::new("jwt");
    let server = serve(&[3600]).await;
    Mock::given(method("POST"))
        .and(path(MESSAGES_PATH))
        .respond_with(ResponseTemplate::new(401).set_body_json(
            json!({"type": "error", "error": {"type": "authentication_error", "message": "invalid token"}}),
        ))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    let model = model(&server.uri(), "anthropic");
    let rejected = run(&model, options(federation_env(token.path()))).await;
    assert_eq!(rejected.stop_reason, StopReason::Error);
    assert!(rejected.error_message.unwrap().starts_with("401 "));

    let message = run(&model, options(federation_env(token.path()))).await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    assert_eq!(requests_to(&server, TOKEN_PATH).await.len(), 2);
    let messages = requests_to(&server, MESSAGES_PATH).await;
    assert_eq!(
        authorizations(&messages),
        [Some("Bearer federated-token-1"), Some("Bearer federated-token-2")]
    );
}

/// The error message of a federated request whose token exchange gets `response`.
async fn exchange_error(response: ResponseTemplate, env: ProviderEnv) -> String {
    let server = serve_token(response).await;
    let message = run(&model(&server.uri(), "anthropic"), options(env)).await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert!(requests_to(&server, MESSAGES_PATH).await.is_empty());
    message.error_message.unwrap()
}

#[tokio::test]
async fn reports_token_exchange_failures_like_the_sdk() {
    let _serial = SERIAL.lock().await;
    let token = TokenFile::new("jwt");
    let env = federation_env(token.path());
    let hint = " Ensure your federation rule matches your identity token. If your federation rule is scoped to multiple workspaces, set the ANTHROPIC_WORKSPACE_ID environment variable, the 'workspace_id' config key, or the `workspaceId` option. View your authentication events in the Workload identity page of Claude Console for more details.";

    let unauthorized = ResponseTemplate::new(401)
        .insert_header("request-id", "req_123")
        .set_body_json(json!({"error": "invalid_grant", "assertion": "secret"}));
    assert_eq!(
        exchange_error(unauthorized.clone(), env.clone()).await,
        format!("Token exchange failed with status 401 (request-id req_123): {{\"error\":\"invalid_grant\"}}{hint}")
    );
    // With a workspace configured, the hint leaves out the workspace advice.
    assert_eq!(
        exchange_error(unauthorized, with(&env, "ANTHROPIC_WORKSPACE_ID", "wrkspc_test")).await,
        "Token exchange failed with status 401 (request-id req_123): {\"error\":\"invalid_grant\"} Ensure your federation rule matches your identity token. View your authentication events in the Workload identity page of Claude Console for more details."
    );
    assert_eq!(
        exchange_error(ResponseTemplate::new(503).set_body_string("upstream down"), env.clone()).await,
        "Token exchange failed with status 503: upstream down"
    );
    assert_eq!(
        exchange_error(ResponseTemplate::new(200).set_body_string("<html>"), env.clone()).await,
        "Token endpoint returned non-JSON response (status 200)"
    );
    assert_eq!(
        exchange_error(
            ResponseTemplate::new(200).set_body_json(json!({"expires_in": 3600, "error": "none"})),
            env.clone()
        )
        .await,
        "Token endpoint response missing access_token: {\"error\":\"none\"}"
    );
    assert_eq!(
        exchange_error(
            ResponseTemplate::new(200)
                .set_body_json(json!({"access_token": "t", "token_type": "MAC", "expires_in": 1})),
            env.clone()
        )
        .await,
        "Token endpoint response: unsupported token_type \"MAC\" (want Bearer)"
    );
    assert_eq!(
        exchange_error(
            ResponseTemplate::new(200).set_body_json(json!({"access_token": "t", "token_type": "bearer"})),
            env
        )
        .await,
        "Token endpoint response missing required fields: {}"
    );
}

#[tokio::test]
async fn reports_identity_token_file_problems() {
    let _serial = SERIAL.lock().await;
    let server = serve(&[3600]).await;
    let model = model(&server.uri(), "anthropic");

    let empty = TokenFile::new(" \n");
    let message = run(&model, options(federation_env(empty.path()))).await;
    assert_eq!(
        message.error_message,
        Some(format!("Identity token file at {} is empty", empty.path()))
    );

    let large = TokenFile::new(&"a".repeat(16 * 1024 + 1));
    let message = run(&model, options(federation_env(large.path()))).await;
    assert_eq!(
        message.error_message.as_deref(),
        Some("Identity token is 17 KiB, exceeds the 16 KiB assertion limit")
    );

    let missing = format!("{}/missing.jwt", empty.dir.to_str().unwrap());
    let message = run(&model, options(federation_env(&missing))).await;
    let error = message.error_message.unwrap();
    assert!(
        error.starts_with(&format!("Failed to read identity token file at {missing}: ")),
        "{error}"
    );

    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn refuses_to_send_the_identity_token_over_plain_http() {
    let _serial = SERIAL.lock().await;
    let token = TokenFile::new("jwt");
    let message = run(
        &model("http://api.example.test", "anthropic"),
        options(federation_env(token.path())),
    )
    .await;
    assert_eq!(
        message.error_message.as_deref(),
        Some("Refusing to send credential over non-https token endpoint \"http://api.example.test\"")
    );
}
