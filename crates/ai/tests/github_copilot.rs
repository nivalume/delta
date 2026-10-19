//! Tests of GitHub Copilot: the OAuth flow, the provider, the dynamic request headers of the
//! three chat APIs and the Copilot-specific message transforms.
//!
//! Ported from upstream `test/github-copilot-oauth.test.ts`, the Copilot cases of
//! `test/oauth-auth.test.ts` and `test/env-api-keys.test.ts`,
//! `test/github-copilot-anthropic.test.ts`,
//! `test/transform-messages-copilot-openai-to-anthropic.test.ts`, the Copilot case of
//! `test/openai-responses-compat.test.ts` and `test/model-catalog-types.test.ts`.
//!
//! GitHub and Copilot endpoints are one wiremock server: `GitHubCopilotOAuth::rewrite_url`
//! turns `https://<host>/<path>` into `<server>/<host>/<path>`, so the tests still see which
//! host each request went to. Upstream's timing tests use fake timers; here the server sends
//! 1-second intervals and the tests check the gaps in real time.

// Test helpers outside #[test] functions; `mapped` mirrors `Model::mapped_thinking_level`.
#![allow(clippy::unwrap_used, clippy::needless_pass_by_value, clippy::option_option)]

use delta_ai::api::anthropic_messages::AnthropicMessagesApi;
use delta_ai::api::openai_completions::OpenAICompletionsApi;
use delta_ai::api::openai_responses::OpenAIResponsesApi;
use delta_ai::auth::{
    ApiKeyInput, AuthContext, AuthEvent, AuthInteraction, AuthPrompt, AuthResolutionOverrides, AuthType, LoginOptions,
    ModelAuth, OAuthAuth, ProviderAuth, env_api_key_auth,
};
use delta_ai::model::ModelThinkingLevel;
use delta_ai::models::CreateModelsOptions;
use delta_ai::oauth::GitHubCopilotOAuth;
use delta_ai::oauth::github_copilot::client_id;
use delta_ai::options::{AnthropicOptions, ApiOptions};
use delta_ai::provider::BuiltProvider;
use delta_ai::providers::builtin::builtin_provider;
use delta_ai::providers::catalog::all_builtin_models;
use delta_ai::providers::github_copilot::{filter_github_copilot_models, github_copilot_provider};
use delta_ai::transform::transform_messages;
use delta_ai::types::{AssistantBlock, AssistantMessage, ImageContent, StopReason, ToolCall, now_ms};
use delta_ai::{
    Context, CreateProviderOptions, Credential, CredentialStore, InMemoryCredentialStore, JsonObject, Message, Model,
    Models, ModelsError, OAuthCredential, Provider, ProviderApi, ProviderStreams, StreamOptions, TextContent,
    ToolResultMessage, UserBlock, UserContent, UserMessage, create_provider, normalize_context,
};
use futures::StreamExt;
use futures::future::BoxFuture;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::any;
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TEST_COPILOT_ACCESS_TOKEN: &str = "tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;";
const TEST_COPILOT_MODELS_URL: &str = "https://api.individual.githubcopilot.com/models";

fn json_response(body: Value, status: u16) -> ResponseTemplate {
    ResponseTemplate::new(status).set_body_json(body)
}

fn empty_ok() -> ResponseTemplate {
    ResponseTemplate::new(200)
}

/// The upstream URL a rewritten request was meant for.
fn upstream_url(request: &Request) -> String {
    format!("https:/{}", request.url.path())
}

fn form(request: &Request) -> Vec<(String, String)> {
    url::form_urlencoded::parse(&request.body).into_owned().collect()
}

fn form_field(request: &Request, name: &str) -> Option<String> {
    form(request)
        .into_iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
}

fn header<'a>(request: &'a Request, name: &str) -> Option<&'a str> {
    request.headers.get(name).and_then(|value| value.to_str().ok())
}

/// A server that answers every request through `handler(upstream url, request)`.
async fn stub(handler: impl Fn(&str, &Request) -> ResponseTemplate + Send + Sync + 'static) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(any())
        .respond_with(move |request: &Request| handler(&upstream_url(request), request))
        .mount(&server)
        .await;
    server
}

fn oauth(server: &MockServer) -> GitHubCopilotOAuth {
    let uri = server.uri();
    GitHubCopilotOAuth {
        rewrite_url: Some(Arc::new(move |url: &str| {
            url.replacen("https://", &format!("{uri}/"), 1)
        })),
    }
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

fn provider_model_id(index: usize) -> String {
    github_copilot_provider().models()[index].id.clone()
}

fn device_code_body(overrides: Value) -> Value {
    let mut body = json!({
        "device_code": "device-code",
        "user_code": "ABCD-EFGH",
        "verification_uri": "https://github.com/login/device",
        "interval": 1,
        "expires_in": 900,
    });
    for (key, value) in overrides.as_object().unwrap() {
        body[key] = value.clone();
    }
    body
}

/// Records events and prompts; text prompts answer `answer`.
struct TestInteraction {
    answer: String,
    cancel: CancellationToken,
    events: Mutex<Vec<AuthEvent>>,
    prompts: Mutex<Vec<AuthPrompt>>,
}

impl TestInteraction {
    fn new(answer: &str) -> Self {
        Self {
            answer: answer.into(),
            cancel: CancellationToken::new(),
            events: Mutex::new(Vec::new()),
            prompts: Mutex::new(Vec::new()),
        }
    }

    fn device_codes(&self) -> Vec<AuthEvent> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| matches!(event, AuthEvent::DeviceCode { .. }))
            .cloned()
            .collect()
    }

    fn progress(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|event| match event {
                AuthEvent::Progress { message } => Some(message.clone()),
                _ => None,
            })
            .collect()
    }
}

impl AuthInteraction for TestInteraction {
    fn prompt(&self, prompt: AuthPrompt) -> BoxFuture<'_, Result<String, ModelsError>> {
        assert!(
            matches!(prompt, AuthPrompt::Text { .. }),
            "Unexpected prompt: {prompt:?}"
        );
        self.prompts.lock().unwrap().push(prompt);
        let answer = self.answer.clone();
        Box::pin(async move { Ok(answer) })
    }

    fn notify(&self, event: AuthEvent) {
        self.events.lock().unwrap().push(event);
    }

    fn cancel(&self) -> CancellationToken {
        self.cancel.clone()
    }
}

type Reply = Arc<dyn Fn(&str) -> ResponseTemplate + Send + Sync>;

/// Login endpoints: device code, an immediate access token, the Copilot token, then
/// `models` for the catalog and `policy(model id)` for policy updates.
async fn login_server(
    models: impl Fn() -> ResponseTemplate + Send + Sync + 'static,
    policy: Option<Reply>,
) -> MockServer {
    stub(move |url, _| {
        if url.ends_with("/login/device/code") {
            return json_response(device_code_body(json!({})), 200);
        }
        if url.ends_with("/login/oauth/access_token") {
            return json_response(json!({"access_token": "ghu_refresh_token"}), 200);
        }
        if url.contains("/copilot_internal/v2/token") {
            return json_response(
                json!({"token": TEST_COPILOT_ACCESS_TOKEN, "expires_at": 9_999_999_999_i64}),
                200,
            );
        }
        if url == TEST_COPILOT_MODELS_URL {
            return models();
        }
        if let Some(rest) = url.strip_prefix(&format!("{TEST_COPILOT_MODELS_URL}/"))
            && let Some(model_id) = rest.strip_suffix("/policy")
        {
            let policy = policy.as_ref().expect("Unexpected policy request");
            return policy(model_id);
        }
        ResponseTemplate::new(599).set_body_string(format!("Unexpected fetch URL: {url}"))
    })
    .await
}

async fn requests_to(server: &MockServer, suffix: &str) -> Vec<Request> {
    server
        .received_requests()
        .await
        .unwrap()
        .into_iter()
        .filter(|request| upstream_url(request).ends_with(suffix))
        .collect()
}

async fn login(oauth: &GitHubCopilotOAuth, interaction: &TestInteraction) -> Result<OAuthCredential, ModelsError> {
    oauth.login(interaction, &LoginOptions::default()).await
}

/// Refresh a credential against a server whose catalog is `data` and whose token names
/// `proxy_host`.
async fn refresh_with_models(data: Value, proxy_host: &'static str) -> OAuthCredential {
    let access_token = format!("tid=test;exp=9999999999;proxy-ep={proxy_host};");
    let models_url = format!("https://{}/models", proxy_host.replacen("proxy.", "api.", 1));
    let token = access_token.clone();
    let server = stub(move |url, request| {
        if url.contains("/copilot_internal/v2/token") {
            return json_response(json!({"token": token, "expires_at": 9_999_999_999_i64}), 200);
        }
        if url == models_url {
            assert_eq!(
                header(request, "authorization"),
                Some(format!("Bearer {token}").as_str())
            );
            return json_response(json!({"data": data}), 200);
        }
        ResponseTemplate::new(599).set_body_string(format!("Unexpected fetch URL: {url}"))
    })
    .await;
    oauth(&server)
        .refresh(credential("old-access-token", "ghu_refresh_token", 0, json!({})))
        .await
        .unwrap()
}

fn available_model_ids(credential: &OAuthCredential) -> Vec<String> {
    credential.extra["availableModelIds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_str().unwrap().to_owned())
        .collect()
}

async fn models_with_stored(provider: Arc<dyn Provider>, stored: Credential) -> (Models, Arc<dyn CredentialStore>) {
    let store: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
    store
        .modify(
            "github-copilot",
            Box::new(move |_| Box::pin(async move { Ok(Some(stored)) })),
        )
        .await
        .unwrap();
    let models = Models::new(CreateModelsOptions {
        credentials: Some(store.clone()),
        ..CreateModelsOptions::default()
    });
    models.set_provider_arc(provider);
    (models, store)
}

/// The built-in provider with `oauth` in place of the default flow.
fn copilot_provider_with(oauth: GitHubCopilotOAuth) -> BuiltProvider {
    let auth =
        ProviderAuth::api_key(env_api_key_auth("GitHub Copilot token", &["COPILOT_GITHUB_TOKEN"])).with_oauth(oauth);
    let mut options = CreateProviderOptions::new(
        "github-copilot",
        auth,
        all_builtin_models("github-copilot").to_vec(),
        ProviderApi::single(OpenAIResponsesApi),
    );
    options.filter_models = Some(Arc::new(filter_github_copilot_models));
    create_provider(options)
}

fn assert_gap(later: Instant, earlier: Instant, at_least: Duration) {
    let gap = later.duration_since(earlier);
    assert!(
        gap + Duration::from_millis(20) >= at_least,
        "gap {gap:?} < {at_least:?}"
    );
}

fn tool_capable(id: &str, picker: bool, policy: Option<&str>) -> Value {
    let mut model =
        json!({"id": id, "model_picker_enabled": picker, "capabilities": {"supports": {"tool_calls": true}}});
    if let Some(state) = policy {
        model["policy"] = json!({"state": state});
    }
    model
}

// ---------------------------------------------------------------------------
// Account model catalog (github-copilot-oauth.test.ts)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn filters_models_to_the_authenticated_account_picker_catalog() {
    let picker_model_id = provider_model_id(0);
    let disabled_model_id = provider_model_id(1);
    let hidden_model_id = provider_model_id(2);

    let credential = refresh_with_models(
        json!([
            tool_capable(&picker_model_id, true, None),
            tool_capable(&disabled_model_id, true, Some("disabled")),
            tool_capable(&hidden_model_id, false, Some("enabled")),
        ]),
        "proxy.individual.githubcopilot.com",
    )
    .await;
    assert_eq!(available_model_ids(&credential), std::slice::from_ref(&picker_model_id));

    let (models, _) = models_with_stored(Arc::new(github_copilot_provider()), Credential::OAuth(credential)).await;
    let available: Vec<String> = models
        .get_available(Some("github-copilot"))
        .await
        .unwrap()
        .into_iter()
        .map(|model| model.id)
        .collect();
    assert_eq!(available, [picker_model_id]);
}

#[tokio::test]
async fn falls_back_to_explicitly_enabled_policy_models_when_the_picker_catalog_is_empty() {
    let enabled_model_id = provider_model_id(0);
    let credential = refresh_with_models(
        json!([
            tool_capable(&enabled_model_id, false, Some("enabled")),
            tool_capable("policy-disabled-model", false, Some("disabled")),
            tool_capable("unconfigured-model", false, None),
            {"id": "tool-incapable-model", "model_picker_enabled": false, "policy": {"state": "enabled"},
             "capabilities": {"supports": {"tool_calls": false}}},
        ]),
        "proxy.individual.githubcopilot.com",
    )
    .await;
    assert_eq!(
        available_model_ids(&credential),
        std::slice::from_ref(&enabled_model_id)
    );

    let (models, _) = models_with_stored(Arc::new(github_copilot_provider()), Credential::OAuth(credential)).await;
    let available: Vec<String> = models
        .get_available(Some("github-copilot"))
        .await
        .unwrap()
        .into_iter()
        .map(|model| model.id)
        .collect();
    assert_eq!(available, [enabled_model_id]);
}

#[tokio::test]
async fn does_not_fall_back_to_policy_models_for_non_individual_accounts() {
    let credential = refresh_with_models(
        json!([tool_capable("gpt-4.1", false, Some("enabled"))]),
        "proxy.business.githubcopilot.com",
    )
    .await;
    assert!(available_model_ids(&credential).is_empty());
}

#[tokio::test]
async fn does_not_retry_model_catalog_throttling_during_credential_refresh() {
    let catalog_requests = Arc::new(Mutex::new(0));
    let count = catalog_requests.clone();
    let server = stub(move |url, _| {
        if url.contains("/copilot_internal/v2/token") {
            return json_response(
                json!({"token": TEST_COPILOT_ACCESS_TOKEN, "expires_at": 9_999_999_999_i64}),
                200,
            );
        }
        if url == TEST_COPILOT_MODELS_URL {
            *count.lock().unwrap() += 1;
            return json_response(json!({"error": "too many requests"}), 429).insert_header("Retry-After", "0");
        }
        ResponseTemplate::new(599)
    })
    .await;
    let error = oauth(&server)
        .refresh(credential("old-access-token", "ghu_refresh_token", 0, json!({})))
        .await
        .unwrap_err();
    assert!(error.message().contains("429"), "{error}");
    assert_eq!(*catalog_requests.lock().unwrap(), 1);
}

#[tokio::test]
async fn filter_keeps_every_model_without_an_oauth_model_list() {
    let models = github_copilot_provider().models();
    let count = models.len();
    assert_eq!(filter_github_copilot_models(models.clone(), None).len(), count);
    let api_key = Credential::ApiKey(delta_ai::ApiKeyCredential::default());
    assert_eq!(
        filter_github_copilot_models(models.clone(), Some(&api_key)).len(),
        count
    );
    let malformed = Credential::OAuth(credential("a", "r", 0, json!({"availableModelIds": ["x", 1]})));
    assert_eq!(
        filter_github_copilot_models(models.clone(), Some(&malformed)).len(),
        count
    );
    let missing = Credential::OAuth(credential("a", "r", 0, json!({})));
    assert_eq!(filter_github_copilot_models(models, Some(&missing)).len(), count);
}

// ---------------------------------------------------------------------------
// Device flow and login (github-copilot-oauth.test.ts)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn reports_device_code_details_through_the_device_code_event() {
    let server = login_server(
        || json_response(json!({"data": []}), 200),
        Some(Arc::new(|_: &str| empty_ok())),
    )
    .await;
    let interaction = TestInteraction::new("");
    let credential = login(&oauth(&server), &interaction).await.unwrap();

    assert_eq!(
        interaction.device_codes(),
        [AuthEvent::DeviceCode {
            user_code: "ABCD-EFGH".into(),
            verification_uri: "https://github.com/login/device".into(),
            interval_seconds: Some(1),
            expires_in_seconds: Some(900),
        }]
    );
    match &interaction.prompts.lock().unwrap()[..] {
        [AuthPrompt::Text { message, placeholder }] => {
            assert_eq!(message, "GitHub Enterprise URL/domain (blank for github.com)");
            assert_eq!(placeholder.as_deref(), Some("company.ghe.com"));
        }
        prompts => panic!("Unexpected prompts: {prompts:?}"),
    }

    assert_eq!(credential.refresh, "ghu_refresh_token");
    assert_eq!(credential.access, TEST_COPILOT_ACCESS_TOKEN);
    assert_eq!(credential.expires, 9_999_999_999_000 - 5 * 60 * 1000);
    assert_eq!(Value::Object(credential.extra), json!({"availableModelIds": []}));

    // The Copilot token request carries the GitHub token and the editor headers; the
    // catalog request adds the API version.
    let token = requests_to(&server, "/copilot_internal/v2/token").await;
    assert_eq!(
        upstream_url(&token[0]),
        "https://api.github.com/copilot_internal/v2/token"
    );
    assert_eq!(header(&token[0], "authorization"), Some("Bearer ghu_refresh_token"));
    assert_eq!(header(&token[0], "user-agent"), Some("GitHubCopilotChat/0.35.0"));
    assert_eq!(header(&token[0], "editor-version"), Some("vscode/1.107.0"));
    assert_eq!(header(&token[0], "editor-plugin-version"), Some("copilot-chat/0.35.0"));
    assert_eq!(header(&token[0], "copilot-integration-id"), Some("vscode-chat"));
    let catalog = requests_to(&server, "/models").await;
    assert_eq!(
        header(&catalog[0], "authorization"),
        Some(format!("Bearer {TEST_COPILOT_ACCESS_TOKEN}").as_str())
    );
    assert_eq!(header(&catalog[0], "x-github-api-version"), Some("2026-06-01"));
    assert_eq!(header(&catalog[0], "accept"), Some("application/json"));
}

#[tokio::test]
async fn updates_only_known_tool_capable_unconfigured_account_model_policies() {
    let configured_model_id = provider_model_id(0);
    let unconfigured_model_id = provider_model_id(1);
    let tool_incapable_model_id = provider_model_id(2);
    let catalog_requests = Arc::new(Mutex::new(0));
    let policy_model_ids = Arc::new(Mutex::new(Vec::<String>::new()));

    let (count, data) = (
        catalog_requests.clone(),
        json!({"data": [
            tool_capable(&configured_model_id, true, Some("enabled")),
            tool_capable(&unconfigured_model_id, true, Some("unconfigured")),
            tool_capable("remote-only-model", true, Some("unconfigured")),
            {"id": tool_incapable_model_id, "model_picker_enabled": true, "policy": {"state": "unconfigured"},
             "capabilities": {"supports": {"tool_calls": false}}},
        ]}),
    );
    let ids = policy_model_ids.clone();
    let server = login_server(
        move || {
            *count.lock().unwrap() += 1;
            json_response(data.clone(), 200)
        },
        Some(Arc::new(move |model_id: &str| {
            ids.lock().unwrap().push(model_id.to_owned());
            empty_ok()
        })),
    )
    .await;

    let interaction = TestInteraction::new("");
    let credential = login(&oauth(&server), &interaction).await.unwrap();

    assert_eq!(*catalog_requests.lock().unwrap(), 1);
    assert_eq!(
        *policy_model_ids.lock().unwrap(),
        std::slice::from_ref(&unconfigured_model_id)
    );
    assert_eq!(interaction.progress(), ["Enabling models..."]);
    assert_eq!(
        available_model_ids(&credential),
        [
            configured_model_id,
            unconfigured_model_id.clone(),
            "remote-only-model".to_owned()
        ]
    );

    let policy = requests_to(&server, "/policy").await;
    assert_eq!(
        upstream_url(&policy[0]),
        format!("{TEST_COPILOT_MODELS_URL}/{unconfigured_model_id}/policy")
    );
    assert_eq!(policy[0].method.as_str(), "POST");
    assert_eq!(
        serde_json::from_slice::<Value>(&policy[0].body).unwrap(),
        json!({"state": "enabled"})
    );
    assert_eq!(header(&policy[0], "content-type"), Some("application/json"));
    assert_eq!(header(&policy[0], "openai-intent"), Some("chat-policy"));
    assert_eq!(header(&policy[0], "x-interaction-type"), Some("chat-policy"));
    assert_eq!(header(&policy[0], "copilot-integration-id"), Some("vscode-chat"));
}

#[tokio::test]
async fn retries_a_throttled_policy_update_after_retry_after() {
    let model_id = provider_model_id(0);
    let policy_times = Arc::new(Mutex::new(Vec::<Instant>::new()));
    let times = policy_times.clone();
    let data = json!({"data": [{"id": model_id, "model_picker_enabled": true, "policy": {"state": "unconfigured"}}]});
    let server = login_server(
        move || json_response(data.clone(), 200),
        Some(Arc::new(move |_: &str| {
            let mut times = times.lock().unwrap();
            times.push(Instant::now());
            if times.len() == 1 {
                json_response(json!({"error": "too many requests"}), 429).insert_header("Retry-After", "1")
            } else {
                empty_ok()
            }
        })),
    )
    .await;

    let credential = login(&oauth(&server), &TestInteraction::new("")).await.unwrap();
    let times = policy_times.lock().unwrap().clone();
    assert_eq!(times.len(), 2);
    assert_gap(times[1], times[0], Duration::from_secs(1));
    assert_eq!(available_model_ids(&credential), [model_id]);
}

#[tokio::test]
async fn continues_policy_updates_after_a_transport_failure() {
    let model_ids = [provider_model_id(0), provider_model_id(1)];
    let policy_model_ids = Arc::new(Mutex::new(Vec::<String>::new()));
    let ids = policy_model_ids.clone();
    let data = json!({"data": model_ids.iter().map(|id| json!({"id": id, "model_picker_enabled": true, "policy": {"state": "unconfigured"}})).collect::<Vec<_>>()});
    let server = login_server(
        move || json_response(data.clone(), 200),
        Some(Arc::new(move |model_id: &str| {
            ids.lock().unwrap().push(model_id.to_owned());
            empty_ok()
        })),
    )
    .await;

    // The first policy request goes to a closed port.
    let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let closed_uri = format!("http://{}", closed.local_addr().unwrap());
    drop(closed);
    let uri = server.uri();
    let failing = format!("/{}/policy", model_ids[0]);
    let oauth = GitHubCopilotOAuth {
        rewrite_url: Some(Arc::new(move |url: &str| {
            let base = if url.ends_with(&failing) { &closed_uri } else { &uri };
            url.replacen("https://", &format!("{base}/"), 1)
        })),
    };

    let credential = login(&oauth, &TestInteraction::new("")).await.unwrap();
    // Only the second update reached the server; it alone is enabled.
    assert_eq!(*policy_model_ids.lock().unwrap(), [model_ids[1].clone()]);
    assert_eq!(available_model_ids(&credential), model_ids);
}

#[tokio::test]
async fn stops_policy_updates_and_persists_authentication_when_the_retry_delay_exceeds_the_login_budget() {
    let first_model_id = provider_model_id(0);
    let second_model_id = provider_model_id(1);
    let policy_model_ids = Arc::new(Mutex::new(Vec::<String>::new()));
    let ids = policy_model_ids.clone();
    let data = json!({"data": [
        {"id": first_model_id, "model_picker_enabled": true, "policy": {"state": "unconfigured"}},
        {"id": second_model_id, "model_picker_enabled": true, "policy": {"state": "unconfigured"}},
    ]});
    let server = login_server(
        move || json_response(data.clone(), 200),
        Some(Arc::new(move |model_id: &str| {
            ids.lock().unwrap().push(model_id.to_owned());
            json_response(json!({"error": "too many requests"}), 429).insert_header("Retry-After", "5")
        })),
    )
    .await;

    let store: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
    let models = Models::new(CreateModelsOptions {
        credentials: Some(store.clone()),
        ..CreateModelsOptions::default()
    });
    models.set_provider(copilot_provider_with(oauth(&server)));
    let credential = models
        .login(
            "github-copilot",
            AuthType::OAuth,
            &TestInteraction::new(""),
            &LoginOptions::default(),
        )
        .await
        .unwrap();
    let Credential::OAuth(oauth_credential) = &credential else {
        panic!("Unexpected credential: {credential:?}");
    };
    assert_eq!(oauth_credential.access, TEST_COPILOT_ACCESS_TOKEN);
    assert_eq!(*policy_model_ids.lock().unwrap(), std::slice::from_ref(&first_model_id));
    assert_eq!(available_model_ids(oauth_credential), [first_model_id, second_model_id]);
    assert_eq!(store.read("github-copilot").await.unwrap(), Some(credential));
}

#[tokio::test]
async fn rejects_a_non_http_verification_uri_before_it_reaches_the_device_code_event() {
    // A malicious enterprise OAuth server could return a verification_uri that the browser
    // launcher would otherwise hand to the OS.
    let server = stub(|url, _| {
        if url.ends_with("/login/device/code") {
            return json_response(device_code_body(json!({"verification_uri": "$(id>/tmp/pwned)"})), 200);
        }
        ResponseTemplate::new(599)
    })
    .await;
    let interaction = TestInteraction::new("");
    let error = login(&oauth(&server), &interaction).await.unwrap_err();
    assert!(error.message().contains("Untrusted verification_uri"), "{error}");
    assert!(interaction.device_codes().is_empty());

    let server = stub(|url, _| {
        if url.ends_with("/login/device/code") {
            return json_response(device_code_body(json!({"verification_uri": "file:///etc/passwd"})), 200);
        }
        ResponseTemplate::new(599)
    })
    .await;
    let error = login(&oauth(&server), &interaction).await.unwrap_err();
    assert_eq!(error.message(), "Untrusted verification_uri in device code response");
}

#[tokio::test]
async fn normalizes_verification_uri_before_it_reaches_the_device_code_event() {
    let raw_verification_uri = "https://github.com/login/\u{1b}]8;;evil";
    let normalized = url::Url::parse(raw_verification_uri).unwrap().to_string();
    assert_ne!(normalized, raw_verification_uri);

    let server = login_server(|| json_response(json!({"data": []}), 200), None).await;
    // A second server answers the device code with the raw URI and forwards the rest.
    let forward = server.uri();
    let device = stub(move |url, _| {
        if url.ends_with("/login/device/code") {
            return json_response(device_code_body(json!({"verification_uri": raw_verification_uri})), 200);
        }
        ResponseTemplate::new(599)
    })
    .await;
    let device_uri = device.uri();
    let oauth = GitHubCopilotOAuth {
        rewrite_url: Some(Arc::new(move |url: &str| {
            let base = if url.ends_with("/login/device/code") {
                &device_uri
            } else {
                &forward
            };
            url.replacen("https://", &format!("{base}/"), 1)
        })),
    };

    let interaction = TestInteraction::new("");
    login(&oauth, &interaction).await.unwrap();
    assert_eq!(
        interaction.device_codes(),
        [AuthEvent::DeviceCode {
            user_code: "ABCD-EFGH".into(),
            verification_uri: normalized,
            interval_seconds: Some(1),
            expires_in_seconds: Some(900),
        }]
    );
}

/// Login endpoints whose access-token poll replies from `replies` and records poll times.
async fn polling_server(device: Value, replies: Vec<Value>) -> (MockServer, Arc<Mutex<Vec<Instant>>>) {
    let poll_times = Arc::new(Mutex::new(Vec::new()));
    let times = poll_times.clone();
    let replies = Mutex::new(replies);
    let server = stub(move |url, request| {
        if url.ends_with("/login/device/code") {
            assert_eq!(request.method.as_str(), "POST");
            assert_eq!(header(request, "accept"), Some("application/json"));
            assert_eq!(
                header(request, "content-type"),
                Some("application/x-www-form-urlencoded")
            );
            assert_eq!(header(request, "user-agent"), Some("GitHubCopilotChat/0.35.0"));
            assert_eq!(form_field(request, "client_id"), Some(client_id()));
            assert_eq!(form_field(request, "scope").as_deref(), Some("read:user"));
            return json_response(device.clone(), 200);
        }
        if url.ends_with("/login/oauth/access_token") {
            times.lock().unwrap().push(Instant::now());
            assert_eq!(request.method.as_str(), "POST");
            assert_eq!(
                header(request, "content-type"),
                Some("application/x-www-form-urlencoded")
            );
            assert_eq!(form_field(request, "client_id"), Some(client_id()));
            assert_eq!(form_field(request, "device_code").as_deref(), Some("device-code"));
            assert_eq!(
                form_field(request, "grant_type").as_deref(),
                Some("urn:ietf:params:oauth:grant-type:device_code")
            );
            let mut replies = replies.lock().unwrap();
            assert!(!replies.is_empty(), "Unexpected extra access token poll");
            return json_response(replies.remove(0), 200);
        }
        if url.contains("/copilot_internal/v2/token") {
            return json_response(
                json!({"token": TEST_COPILOT_ACCESS_TOKEN, "expires_at": 9_999_999_999_i64}),
                200,
            );
        }
        if url == TEST_COPILOT_MODELS_URL {
            return json_response(json!({"data": []}), 200);
        }
        ResponseTemplate::new(599)
    })
    .await;
    (server, poll_times)
}

#[tokio::test]
async fn waits_before_polling_and_uses_the_server_interval_after_slow_down() {
    let (server, poll_times) = polling_server(
        device_code_body(json!({"interval": 1})),
        vec![
            json!({"error": "authorization_pending", "error_description": "pending"}),
            json!({"error": "slow_down", "error_description": "slow down", "interval": 2}),
            json!({"access_token": "ghu_refresh_token"}),
        ],
    )
    .await;
    let start = Instant::now();
    login(&oauth(&server), &TestInteraction::new("")).await.unwrap();

    let times = poll_times.lock().unwrap().clone();
    assert_eq!(times.len(), 3);
    assert_gap(times[0], start, Duration::from_secs(1));
    assert_gap(times[1], times[0], Duration::from_secs(1));
    // slow_down carried a server-provided interval of 2 seconds.
    assert_gap(times[2], times[1], Duration::from_secs(2));
}

#[tokio::test]
async fn times_out_after_repeated_slow_down_responses() {
    // Interval 1 s, 13 s to expiry: polls at 1 s and 7 s (slow_down adds 5 s each time),
    // then the next wait reaches the deadline.
    let (server, poll_times) = polling_server(
        device_code_body(json!({"interval": 1, "expires_in": 13})),
        vec![
            json!({"error": "slow_down", "error_description": "slow down"}),
            json!({"error": "slow_down", "error_description": "still too fast"}),
            json!({"error": "authorization_pending", "error_description": "pending"}),
        ],
    )
    .await;
    let start = Instant::now();
    let error = login(&oauth(&server), &TestInteraction::new("")).await.unwrap_err();
    assert!(
        error
            .message()
            .contains("Device flow timed out after one or more slow_down responses"),
        "{error}"
    );
    let times = poll_times.lock().unwrap().clone();
    assert_eq!(times.len(), 2);
    assert_gap(times[0], start, Duration::from_secs(1));
    assert_gap(times[1], times[0], Duration::from_secs(6));
    assert_gap(Instant::now(), start, Duration::from_secs(13));
}

#[tokio::test]
async fn reports_device_flow_errors_and_invalid_responses() {
    let (server, _) = polling_server(
        device_code_body(json!({})),
        vec![json!({"error": "access_denied", "error_description": "The user denied access"})],
    )
    .await;
    let error = login(&oauth(&server), &TestInteraction::new("")).await.unwrap_err();
    assert_eq!(
        error.message(),
        "Device flow failed: access_denied: The user denied access"
    );

    let (server, _) = polling_server(device_code_body(json!({})), vec![json!({"unexpected": true})]).await;
    let error = login(&oauth(&server), &TestInteraction::new("")).await.unwrap_err();
    assert_eq!(error.message(), "Invalid device token response");

    let (server, _) = polling_server(device_code_body(json!({"expires_in": "900"})), vec![]).await;
    let error = login(&oauth(&server), &TestInteraction::new("")).await.unwrap_err();
    assert_eq!(error.message(), "Invalid device code response fields");

    let server = stub(|url, _| {
        if url.ends_with("/login/device/code") {
            return ResponseTemplate::new(404).set_body_string("Not Found");
        }
        ResponseTemplate::new(599)
    })
    .await;
    let error = login(&oauth(&server), &TestInteraction::new("")).await.unwrap_err();
    assert_eq!(error.message(), "404 Not Found: Not Found");
}

#[tokio::test]
async fn logs_in_to_github_enterprise_and_stores_the_domain() {
    let server = stub(|url, _| match url {
        "https://company.ghe.com/login/device/code" => json_response(device_code_body(json!({})), 200),
        "https://company.ghe.com/login/oauth/access_token" => {
            json_response(json!({"access_token": "ghu_enterprise"}), 200)
        }
        "https://api.company.ghe.com/copilot_internal/v2/token" => {
            json_response(json!({"token": "no-proxy-ep", "expires_at": 9_999_999_999_i64}), 200)
        }
        "https://copilot-api.company.ghe.com/models" => {
            json_response(json!({"data": [tool_capable("gpt-4.1", true, None)]}), 200)
        }
        _ => ResponseTemplate::new(599).set_body_string(format!("Unexpected fetch URL: {url}")),
    })
    .await;
    let oauth = oauth(&server);
    let credential = login(&oauth, &TestInteraction::new(" https://Company.GHE.com/ "))
        .await
        .unwrap();
    assert_eq!(credential.refresh, "ghu_enterprise");
    assert_eq!(
        Value::Object(credential.extra.clone()),
        json!({"enterpriseUrl": "company.ghe.com", "availableModelIds": ["gpt-4.1"]})
    );
    let auth = oauth.to_auth(&credential).await.unwrap();
    assert_eq!(auth.base_url.as_deref(), Some("https://copilot-api.company.ghe.com"));
}

#[tokio::test]
async fn rejects_an_invalid_enterprise_domain_before_any_request() {
    let server = stub(|_, _| ResponseTemplate::new(599)).await;
    let error = login(&oauth(&server), &TestInteraction::new("not a domain"))
        .await
        .unwrap_err();
    assert_eq!(error.message(), "Invalid GitHub Enterprise URL/domain");
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn cancels_while_waiting_for_the_first_token_poll() {
    let (server, poll_times) = polling_server(device_code_body(json!({"interval": 5})), vec![]).await;
    let interaction = TestInteraction::new("");
    let cancel = interaction.cancel.clone();
    let oauth = oauth(&server);
    let task = async { login(&oauth, &interaction).await };
    let (result, ()) = tokio::join!(task, async {
        tokio::time::sleep(Duration::from_millis(300)).await;
        cancel.cancel();
    });
    let error = result.unwrap_err();
    assert_eq!(error.message(), "Login cancelled");
    assert_eq!(error.code(), delta_ai::ModelsErrorCode::Aborted);
    assert!(poll_times.lock().unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// OAuth auth and provider wiring (oauth-auth.test.ts, env-api-keys.test.ts)
// ---------------------------------------------------------------------------

#[test]
fn provider_exposes_subscription_oauth_alongside_the_copilot_token() {
    let provider = builtin_provider("github-copilot").unwrap();
    assert_eq!(provider.name(), "GitHub Copilot");
    assert_eq!(provider.base_url(), Some("https://api.individual.githubcopilot.com"));
    assert_eq!(provider.auth().api_key.as_ref().unwrap().name(), "GitHub Copilot token");
    let oauth = provider.auth().oauth.as_ref().unwrap();
    assert_eq!(oauth.name(), "GitHub Copilot");
    assert!(oauth.is_subscription());
    assert_eq!(oauth.login_label(), None);
}

#[tokio::test]
async fn to_auth_derives_the_base_url_from_the_token_proxy_endpoint() {
    let access = "tid=abc;exp=123;proxy-ep=proxy.enterprise.example;rest";
    let auth = GitHubCopilotOAuth::default()
        .to_auth(&credential(access, "r", 0, json!({})))
        .await
        .unwrap();
    assert_eq!(
        auth,
        ModelAuth {
            api_key: Some(access.into()),
            base_url: Some("https://api.enterprise.example".into()),
            headers: None,
        }
    );
}

#[tokio::test]
async fn to_auth_falls_back_to_the_enterprise_domain_then_the_individual_endpoint() {
    let oauth = GitHubCopilotOAuth::default();
    let enterprise = oauth
        .to_auth(&credential(
            "no-proxy-ep",
            "r",
            0,
            json!({"enterpriseUrl": "https://company.ghe.com"}),
        ))
        .await
        .unwrap();
    assert_eq!(
        enterprise.base_url.as_deref(),
        Some("https://copilot-api.company.ghe.com")
    );
    let individual = oauth
        .to_auth(&credential("no-proxy-ep", "r", 0, json!({})))
        .await
        .unwrap();
    assert_eq!(
        individual.base_url.as_deref(),
        Some("https://api.individual.githubcopilot.com")
    );
}

#[tokio::test]
async fn refresh_preserves_the_enterprise_domain() {
    let server = stub(|url, _| {
        if url.ends_with("/models") {
            return json_response(json!({"data": []}), 200);
        }
        json_response(json!({"token": "new-token", "expires_at": 9_999_999_999_i64}), 200)
    })
    .await;
    let refreshed = oauth(&server)
        .refresh(credential(
            "old",
            "gh-token",
            0,
            json!({"enterpriseUrl": "company.ghe.com"}),
        ))
        .await
        .unwrap();
    assert_eq!(refreshed.access, "new-token");
    assert_eq!(refreshed.refresh, "gh-token");
    assert_eq!(refreshed.extra["enterpriseUrl"], "company.ghe.com");
    let requests = server.received_requests().await.unwrap();
    assert!(upstream_url(&requests[0]).contains("api.company.ghe.com"));
    assert_eq!(upstream_url(&requests[1]), "https://copilot-api.company.ghe.com/models");
}

#[tokio::test]
async fn refresh_reports_copilot_token_failures() {
    let server = stub(|_, _| ResponseTemplate::new(401).set_body_string("bad credentials")).await;
    let error = oauth(&server)
        .refresh(credential("old", "gh-token", 0, json!({})))
        .await
        .unwrap_err();
    assert_eq!(error.message(), "401 Unauthorized: bad credentials");

    let server = stub(|_, _| json_response(json!({"token": "t"}), 200)).await;
    let error = oauth(&server)
        .refresh(credential("old", "gh-token", 0, json!({})))
        .await
        .unwrap_err();
    assert_eq!(error.message(), "Invalid Copilot token response fields");

    let server = stub(|url, _| {
        if url.ends_with("/models") {
            return json_response(json!({"models": []}), 200);
        }
        json_response(json!({"token": "t", "expires_at": 1}), 200)
    })
    .await;
    let error = oauth(&server)
        .refresh(credential("old", "gh-token", 0, json!({})))
        .await
        .unwrap_err();
    assert_eq!(error.message(), "Invalid Copilot models response");
}

#[tokio::test]
async fn resolves_stored_oauth_credentials_including_the_per_credential_base_url() {
    let access = "tid=abc;exp=123;proxy-ep=proxy.business.githubcopilot.com;rest";
    // Keep this beyond get_auth()'s refresh window.
    let stored = credential(access, "r", now_ms() + 10 * 60_000, json!({}));
    let (models, _) = models_with_stored(builtin_provider("github-copilot").unwrap(), Credential::OAuth(stored)).await;
    let result = models
        .get_auth("github-copilot", &AuthResolutionOverrides::default())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.auth.api_key.as_deref(), Some(access));
    assert_eq!(
        result.auth.base_url.as_deref(),
        Some("https://api.business.githubcopilot.com")
    );
}

struct Env(HashMap<&'static str, &'static str>);

impl AuthContext for Env {
    fn env(&self, name: &str) -> Option<String> {
        self.0.get(name).map(|value| (*value).to_owned())
    }

    fn file_exists(&self, _: &str) -> bool {
        false
    }
}

#[tokio::test]
async fn resolves_the_api_key_only_from_copilot_github_token() {
    let provider = builtin_provider("github-copilot").unwrap();
    let api_key = provider.auth().api_key.clone().unwrap();

    let generic = Env([("GH_TOKEN", "gh-token"), ("GITHUB_TOKEN", "github-token")].into());
    let result = api_key
        .resolve(ApiKeyInput {
            ctx: &generic,
            credential: None,
        })
        .await
        .unwrap();
    assert!(result.is_none());

    let copilot = Env([
        ("COPILOT_GITHUB_TOKEN", "copilot-token"),
        ("GH_TOKEN", "gh-token"),
        ("GITHUB_TOKEN", "github-token"),
    ]
    .into());
    let result = api_key
        .resolve(ApiKeyInput {
            ctx: &copilot,
            credential: None,
        })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.auth.api_key.as_deref(), Some("copilot-token"));
    assert_eq!(result.source.as_deref(), Some("COPILOT_GITHUB_TOKEN"));
}

// ---------------------------------------------------------------------------
// Catalog (github-copilot-anthropic.test.ts, model-catalog-types.test.ts)
// ---------------------------------------------------------------------------

fn copilot_model(id: &str) -> Model {
    github_copilot_provider()
        .models()
        .into_iter()
        .find(|model| model.id == id)
        .unwrap()
}

fn mapped(model: &Model, level: ModelThinkingLevel) -> Option<Option<&str>> {
    model.mapped_thinking_level(level)
}

#[test]
fn applies_copilot_specific_adaptive_thinking_effort_overrides() {
    use ModelThinkingLevel::{High, Low, Max, Medium, Minimal, Xhigh};

    let opus47 = copilot_model("claude-opus-4.7");
    assert_eq!(mapped(&opus47, Minimal), Some(Some("low")));
    assert_eq!(mapped(&opus47, Xhigh), Some(Some("xhigh")));
    assert_eq!(mapped(&opus47, Max), Some(Some("max")));
    assert!(opus47.supported_thinking_levels().contains(&Xhigh));
    assert!(opus47.supported_thinking_levels().contains(&Max));

    let opus5 = copilot_model("claude-opus-5");
    assert_eq!(opus5.api, "anthropic-messages");
    assert_eq!(opus5.context_window, 1_000_000);
    assert_eq!(mapped(&opus5, Minimal), Some(Some("low")));
    assert_eq!(mapped(&opus5, Xhigh), Some(Some("xhigh")));
    assert_eq!(mapped(&opus5, Max), Some(Some("max")));

    let opus55 = copilot_model("claude-opus-5.5");
    assert_eq!(opus55.api, "anthropic-messages");
    assert_eq!(opus55.context_window, 1_000_000);
    assert_eq!(opus55.supported_thinking_levels(), [Low, Medium, High, Xhigh, Max]);

    let sonnet46 = copilot_model("claude-sonnet-4.6");
    assert_eq!(mapped(&sonnet46, Minimal), Some(Some("low")));
    assert_eq!(mapped(&sonnet46, Max), Some(Some("max")));
    assert!(sonnet46.supported_thinking_levels().contains(&Max));
    assert!(!sonnet46.supported_thinking_levels().contains(&Xhigh));
}

#[test]
fn routes_grok_and_every_gpt_model_through_the_responses_api() {
    assert_eq!(copilot_model("grok-4.5").api, "openai-responses");
    let models = github_copilot_provider().models();
    let gpt: Vec<&Model> = models.iter().filter(|model| model.id.starts_with("gpt-")).collect();
    assert!(!gpt.is_empty());
    assert!(gpt.iter().all(|model| model.api == "openai-responses"));
    for id in ["gpt-6-sol", "gpt-6-luna"] {
        let model = copilot_model(id);
        assert_eq!(model.context_window, 1_000_000);
        assert_eq!(model.max_tokens, 128_000);
        assert_eq!(mapped(&model, ModelThinkingLevel::Off), Some(Some("none")));
        assert_eq!(mapped(&model, ModelThinkingLevel::Max), Some(Some("max")));
    }
}

// ---------------------------------------------------------------------------
// Dynamic request headers (github-copilot-anthropic.test.ts and the OpenAI APIs)
// ---------------------------------------------------------------------------

fn client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("tid_copilot_session_test_token".into()),
        http_client: Some(client()),
        max_retries: Some(0),
        ..StreamOptions::default()
    }
}

fn at(model: &Model, server: &MockServer) -> Model {
    let mut model = model.clone();
    model.base_url = server.uri();
    model
}

fn user(text: &str) -> Message {
    Message::User(UserMessage {
        content: text.into(),
        timestamp: 1,
    })
}

fn image_block() -> UserBlock {
    UserBlock::Image(ImageContent {
        data: "iVBORw0KGgo=".into(),
        mime_type: "image/png".into(),
    })
}

fn copilot_assistant(api: &str, model: &str, content: Vec<AssistantBlock>) -> AssistantMessage {
    let mut message = AssistantMessage::empty(api, "github-copilot", model);
    message.content = content;
    message.stop_reason = StopReason::ToolUse;
    message.timestamp = 1;
    message
}

fn call(id: &str, name: &str, arguments: Value) -> AssistantBlock {
    AssistantBlock::ToolCall(ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: arguments.as_object().cloned().unwrap(),
        ..ToolCall::default()
    })
}

fn result(id: &str, name: &str, content: Vec<UserBlock>, is_error: bool) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: id.into(),
        tool_name: name.into(),
        content,
        is_error,
        timestamp: 2,
        ..ToolResultMessage::default()
    })
}

/// A tool round trip whose result carries an image; the last message is a tool result.
fn agent_turn_with_image(api: &str, model: &str) -> Vec<Message> {
    vec![
        user("read the screenshot"),
        Message::Assistant(copilot_assistant(
            api,
            model,
            vec![call("call_1", "read", json!({"path": "shot.png"}))],
        )),
        result(
            "call_1",
            "read",
            vec![UserBlock::Text(TextContent::new("done")), image_block()],
            false,
        ),
    ]
}

async fn sse_server(route: &'static str, body: String) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path(route))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;
    server
}

async fn drain(mut stream: delta_ai::AssistantMessageEventStream) {
    while stream.next().await.is_some() {}
}

async fn only_request(server: &MockServer) -> Request {
    let mut requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    requests.pop().unwrap()
}

fn anthropic_sse() -> String {
    [
        json!({"type": "message_start", "message": {"id": "msg_test", "usage": {"input_tokens": 10, "output_tokens": 0}}}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}),
        json!({"type": "message_stop"}),
    ]
    .iter()
    .fold(String::new(), |mut body, event| {
        let _ = write!(body, "event: {}\ndata: {event}\n\n", event["type"].as_str().unwrap());
        body
    })
}

fn hello() -> delta_ai::TranscriptContext {
    normalize_context(&Context::new(vec![user("Hello")]).with_system_prompt("You are a helpful assistant."))
}

#[tokio::test]
async fn anthropic_uses_bearer_auth_copilot_headers_and_a_valid_messages_payload() {
    let server = sse_server("/v1/messages", anthropic_sse()).await;
    let model = at(&copilot_model("claude-sonnet-4.6"), &server);
    assert_eq!(model.api, "anthropic-messages");
    drain(AnthropicMessagesApi.stream(&model, &hello(), options())).await;

    let request = only_request(&server).await;
    // Auth: Bearer, no x-api-key.
    assert_eq!(
        header(&request, "authorization"),
        Some("Bearer tid_copilot_session_test_token")
    );
    assert_eq!(header(&request, "x-api-key"), None);
    // Copilot static headers from model.headers.
    assert!(header(&request, "user-agent").unwrap().contains("GitHubCopilotChat"));
    assert_eq!(header(&request, "copilot-integration-id"), Some("vscode-chat"));
    // Dynamic headers.
    assert_eq!(header(&request, "x-initiator"), Some("user"));
    assert_eq!(header(&request, "openai-intent"), Some("conversation-edits"));
    assert_eq!(header(&request, "copilot-vision-request"), None);

    // The payload is valid Anthropic Messages format.
    let betas = header(&request, "anthropic-beta").unwrap_or_default();
    assert!(!betas.contains("fine-grained-tool-streaming-2025-05-14"));
    let body: Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(body["model"], "claude-sonnet-4.6");
    assert_eq!(body["stream"], true);
    assert_eq!(body["max_tokens"], model.max_tokens);
    assert!(body["messages"].is_array());
}

#[tokio::test]
async fn anthropic_omits_the_interleaved_thinking_beta_for_adaptive_thinking_models() {
    let server = sse_server("/v1/messages", anthropic_sse()).await;
    let model = at(&copilot_model("claude-sonnet-4.6"), &server);
    let options = StreamOptions {
        api: Some(ApiOptions::AnthropicMessages(AnthropicOptions {
            interleaved_thinking: Some(true),
            ..AnthropicOptions::default()
        })),
        ..options()
    };
    drain(AnthropicMessagesApi.stream(&model, &hello(), options)).await;
    let request = only_request(&server).await;
    let betas = header(&request, "anthropic-beta").unwrap_or_default();
    assert!(!betas.contains("interleaved-thinking-2025-05-14"), "{betas}");
}

#[tokio::test]
async fn anthropic_marks_agent_follow_ups_and_vision_requests() {
    let server = sse_server("/v1/messages", anthropic_sse()).await;
    let model = at(&copilot_model("claude-sonnet-4.6"), &server);
    let context = normalize_context(&Context::new(agent_turn_with_image(
        "anthropic-messages",
        "claude-sonnet-4.6",
    )));
    // Request headers still override the dynamic ones.
    let mut options = options();
    options.headers = Some(
        [("Openai-Intent".to_owned(), Some("custom".to_owned()))]
            .into_iter()
            .collect(),
    );
    drain(AnthropicMessagesApi.stream(&model, &context, options)).await;
    let request = only_request(&server).await;
    assert_eq!(header(&request, "x-initiator"), Some("agent"));
    assert_eq!(header(&request, "copilot-vision-request"), Some("true"));
    assert_eq!(header(&request, "openai-intent"), Some("custom"));
}

#[tokio::test]
async fn anthropic_sends_no_copilot_headers_for_other_providers() {
    let server = sse_server("/v1/messages", anthropic_sse()).await;
    let mut model = at(&copilot_model("claude-sonnet-4.6"), &server);
    model.provider = "anthropic".into();
    drain(AnthropicMessagesApi.stream(&model, &hello(), options())).await;
    let request = only_request(&server).await;
    assert_eq!(header(&request, "x-initiator"), None);
    assert_eq!(header(&request, "openai-intent"), None);
}

fn completions_sse() -> String {
    let chunk = json!({"id": "c1", "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}});
    format!("data: {chunk}\n\ndata: [DONE]\n\n")
}

#[tokio::test]
async fn openai_completions_sends_copilot_dynamic_headers() {
    let server = sse_server("/chat/completions", completions_sse()).await;
    let model = at(&copilot_model("gemini-3.5-flash"), &server);
    assert_eq!(model.api, "openai-completions");
    drain(OpenAICompletionsApi.stream(&model, &hello(), options())).await;
    let request = only_request(&server).await;
    assert_eq!(
        header(&request, "authorization"),
        Some("Bearer tid_copilot_session_test_token")
    );
    assert_eq!(header(&request, "copilot-integration-id"), Some("vscode-chat"));
    assert_eq!(header(&request, "x-initiator"), Some("user"));
    assert_eq!(header(&request, "openai-intent"), Some("conversation-edits"));
    assert_eq!(header(&request, "copilot-vision-request"), None);

    let server = sse_server("/chat/completions", completions_sse()).await;
    let model = at(&copilot_model("gemini-3.5-flash"), &server);
    let context = normalize_context(&Context::new(agent_turn_with_image(
        "openai-completions",
        "gemini-3.5-flash",
    )));
    drain(OpenAICompletionsApi.stream(&model, &context, options())).await;
    let request = only_request(&server).await;
    assert_eq!(header(&request, "x-initiator"), Some("agent"));
    assert_eq!(header(&request, "copilot-vision-request"), Some("true"));
}

fn responses_sse() -> String {
    let completed = json!({"type": "response.completed", "response": {"id": "resp_1", "status": "completed",
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}});
    format!("data: {completed}\n\n")
}

#[tokio::test]
async fn openai_responses_sends_copilot_dynamic_headers() {
    let server = sse_server("/responses", responses_sse()).await;
    let model = at(&copilot_model("gpt-5-mini"), &server);
    assert_eq!(model.api, "openai-responses");
    drain(OpenAIResponsesApi.stream(&model, &hello(), options())).await;
    let request = only_request(&server).await;
    assert_eq!(
        header(&request, "authorization"),
        Some("Bearer tid_copilot_session_test_token")
    );
    assert_eq!(header(&request, "editor-version"), Some("vscode/1.107.0"));
    assert_eq!(header(&request, "x-initiator"), Some("user"));
    assert_eq!(header(&request, "openai-intent"), Some("conversation-edits"));
    assert_eq!(header(&request, "copilot-vision-request"), None);
    // openai-responses-compat.test.ts: no reasoning when none is requested.
    let body: Value = serde_json::from_slice(&request.body).unwrap();
    assert!(body.get("reasoning").is_none(), "{body}");

    let server = sse_server("/responses", responses_sse()).await;
    let model = at(&copilot_model("gpt-5-mini"), &server);
    let context = normalize_context(&Context::new(agent_turn_with_image("openai-responses", "gpt-5-mini")));
    drain(OpenAIResponsesApi.stream(&model, &context, options())).await;
    let request = only_request(&server).await;
    assert_eq!(header(&request, "x-initiator"), Some("agent"));
    assert_eq!(header(&request, "copilot-vision-request"), Some("true"));
}

// ---------------------------------------------------------------------------
// OpenAI to Anthropic session migration for Copilot Claude
// (transform-messages-copilot-openai-to-anthropic.test.ts)
// ---------------------------------------------------------------------------

/// The tool call id normalization `anthropic-messages` uses.
fn anthropic_normalize_tool_call_id(id: &str, _: &Model, _: &AssistantMessage) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect()
}

fn transform_for_copilot_claude(messages: &[Message]) -> Vec<Message> {
    transform_messages(
        messages,
        &copilot_model("claude-sonnet-4.6"),
        Some(&anthropic_normalize_tool_call_id),
    )
}

fn synthetic_results(messages: &[Message]) -> Vec<&ToolResultMessage> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult(result) if result.is_error => Some(result),
            _ => None,
        })
        .collect()
}

fn text_of(result: &ToolResultMessage) -> Vec<&str> {
    result
        .content
        .iter()
        .map(|block| match block {
            UserBlock::Text(text) => text.text.as_str(),
            UserBlock::Image(_) => "<image>",
        })
        .collect()
}

#[test]
fn removes_thought_signatures_from_tool_calls_when_migrating_between_models() {
    let signature = json!({"type": "reasoning.encrypted", "id": "call_123", "data": "encrypted"}).to_string();
    let messages = vec![
        user("run a command"),
        Message::Assistant(copilot_assistant(
            "openai-responses",
            "gpt-5",
            vec![AssistantBlock::ToolCall(ToolCall {
                id: "call_123".into(),
                name: "bash".into(),
                arguments: json!({"command": "ls"}).as_object().cloned().unwrap(),
                thought_signature: Some(signature),
                ..ToolCall::default()
            })],
        )),
        result(
            "call_123",
            "bash",
            vec![UserBlock::Text(TextContent::new("output"))],
            false,
        ),
    ];

    let transformed = transform_for_copilot_claude(&messages);
    let tool_call = transformed
        .iter()
        .find_map(|message| match message {
            Message::Assistant(assistant) => assistant.content.iter().find_map(|block| match block {
                AssistantBlock::ToolCall(call) => Some(call),
                _ => None,
            }),
            _ => None,
        })
        .unwrap();
    assert_eq!(tool_call.thought_signature, None);
}

#[test]
fn adds_synthetic_tool_results_for_trailing_orphaned_tool_calls() {
    let messages = vec![
        user("read the file"),
        Message::Assistant(copilot_assistant(
            "openai-responses",
            "gpt-5",
            vec![call("call_123|fc_123", "read", json!({"path": "README.md"}))],
        )),
    ];

    let transformed = transform_for_copilot_claude(&messages);
    let Some(Message::ToolResult(last)) = transformed.last() else {
        panic!("Unexpected last message: {:?}", transformed.last());
    };
    assert_eq!(last.tool_call_id, "call_123_fc_123");
    assert_eq!(last.tool_name, "read");
    assert!(last.is_error);
    assert_eq!(text_of(last), ["No result provided"]);
}

#[test]
fn adds_synthetic_results_only_for_trailing_tool_calls_that_are_still_missing_results() {
    let messages = vec![
        user("run commands"),
        Message::Assistant(copilot_assistant(
            "openai-responses",
            "gpt-5",
            vec![
                call("call_1|fc_1", "read", json!({"path": "README.md"})),
                call("call_2|fc_2", "bash", json!({"command": "pwd"})),
            ],
        )),
        result(
            "call_1|fc_1",
            "read",
            vec![UserBlock::Text(TextContent::new("done"))],
            false,
        ),
    ];

    let transformed = transform_for_copilot_claude(&messages);
    let synthetic = synthetic_results(&transformed);
    assert_eq!(synthetic.len(), 1);
    assert_eq!(synthetic[0].tool_call_id, "call_2_fc_2");
    assert_eq!(synthetic[0].tool_name, "bash");
    assert_eq!(text_of(synthetic[0]), ["No result provided"]);
}

#[test]
fn user_content_blocks_without_images_are_not_vision_requests() {
    use delta_ai::api::github_copilot_headers::has_copilot_vision_input;
    let blocks = Message::User(UserMessage {
        content: UserContent::Blocks(vec![UserBlock::Text(TextContent::new("hi"))]),
        timestamp: 1,
    });
    assert!(!has_copilot_vision_input(&[blocks]));
}
