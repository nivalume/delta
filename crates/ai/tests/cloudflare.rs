//! End-to-end tests of the Cloudflare AI Gateway and Workers AI providers: auth resolution,
//! endpoint placeholders and auth headers, against local mock servers.

// Shared helpers outside `#[test]` functions panic on setup failures too.
#![allow(clippy::unwrap_used)]

use delta_ai::auth::AuthContext;
use delta_ai::providers::cloudflare::{cloudflare_ai_gateway_provider, cloudflare_workers_ai_provider};
use delta_ai::types::{ConstrainedSampling, ProviderEnv, StrictMode};
use delta_ai::{
    AssistantMessage, Context, CreateModelsOptions, Message, Model, Models, SimpleStreamOptions, StopReason,
    StreamOptions, ThinkingLevel, Tool,
};
use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Arc;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

const GATEWAY_ENV: &[(&str, &str)] = &[
    ("CLOUDFLARE_API_KEY", "cf-token"),
    ("CLOUDFLARE_ACCOUNT_ID", "account-id"),
    ("CLOUDFLARE_GATEWAY_ID", "gateway-id"),
];

#[derive(Default)]
struct Ctx {
    env: BTreeMap<String, String>,
}

impl AuthContext for Ctx {
    fn env(&self, name: &str) -> Option<String> {
        self.env.get(name).cloned()
    }

    fn file_exists(&self, _path: &str) -> bool {
        false
    }
}

fn models(env: &[(&str, &str)]) -> Models {
    let models = Models::new(CreateModelsOptions {
        auth_context: Some(Arc::new(Ctx {
            env: env
                .iter()
                .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
                .collect(),
        })),
        ..CreateModelsOptions::default()
    });
    models.set_provider(cloudflare_ai_gateway_provider());
    models.set_provider(cloudflare_workers_ai_provider());
    models
}

/// A catalog model whose endpoint host is the mock server; the placeholders stay.
fn catalog_model(models: &Models, provider: &str, id: &str, server: &MockServer) -> Model {
    let mut model = models.get_model(provider, id).unwrap();
    model.base_url = model
        .base_url
        .replace("https://gateway.ai.cloudflare.com", &server.uri())
        .replace("https://api.cloudflare.com", &server.uri());
    assert!(model.base_url.contains("{CLOUDFLARE_ACCOUNT_ID}"), "{}", model.base_url);
    model
}

async fn serve(body: String) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;
    server
}

/// One `chat.completions` chunk that ends the response.
fn completions_body() -> String {
    let chunk = json!({"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}});
    format!("data: {chunk}\n\ndata: [DONE]\n\n")
}

fn responses_body() -> String {
    let completed = json!({"type": "response.completed", "response": {"id": "r", "status": "completed",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}});
    format!("data: {completed}\n\n")
}

fn anthropic_body() -> String {
    let events = [
        json!({"type": "message_start", "message": {"id": "msg", "model": "claude",
            "usage": {"input_tokens": 1, "output_tokens": 0}}}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 1, "output_tokens": 1}}),
        json!({"type": "message_stop"}),
    ];
    events.iter().fold(String::new(), |mut body, event| {
        let _ = write!(body, "event: {}\ndata: {event}\n\n", event["type"].as_str().unwrap());
        body
    })
}

fn base() -> StreamOptions {
    StreamOptions {
        http_client: Some(reqwest::Client::builder().no_proxy().build().unwrap()),
        max_retries: Some(0),
        ..StreamOptions::default()
    }
}

fn simple(base: StreamOptions) -> SimpleStreamOptions {
    SimpleStreamOptions {
        base,
        ..SimpleStreamOptions::default()
    }
}

fn hi() -> Context {
    Context::new(vec![Message::user("hi")])
}

async fn last_request(server: &MockServer) -> Request {
    server.received_requests().await.unwrap().pop().unwrap()
}

fn header<'a>(request: &'a Request, name: &str) -> Option<&'a str> {
    request.headers.get(name).map(|value| value.to_str().unwrap())
}

fn body(request: &Request) -> Value {
    request.body_json().unwrap()
}

fn assert_done(message: &AssistantMessage) {
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
}

#[tokio::test]
async fn uses_conservative_openai_compatible_fields_for_ai_gateway_compat_models() {
    let server = serve(completions_body()).await;
    let models = models(GATEWAY_ENV);
    let model = catalog_model(
        &models,
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
        &server,
    );
    let mut options = simple(StreamOptions {
        max_tokens: Some(1234),
        ..base()
    });
    options.reasoning = Some(ThinkingLevel::High);

    let message = models
        .complete_simple(&model, &hi().with_system_prompt("You are helpful."), options)
        .await;
    assert_done(&message);

    let request = last_request(&server).await;
    assert_eq!(request.url.path(), "/v1/account-id/gateway-id/compat/chat/completions");
    let params = body(&request);
    assert_eq!(params["messages"][0]["role"], "system");
    assert_eq!(params["max_tokens"], 1234);
    assert!(params.get("max_completion_tokens").is_none());
    assert!(params.get("reasoning_effort").is_none());
    assert!(params.get("store").is_none());
    assert_eq!(header(&request, "authorization"), None);
    assert_eq!(header(&request, "cf-aig-authorization"), Some("Bearer cf-token"));
}

#[tokio::test]
async fn resolves_ai_gateway_base_url_through_provider_auth() {
    let server = serve(completions_body()).await;
    let models = models(GATEWAY_ENV);
    let model = catalog_model(
        &models,
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
        &server,
    );

    assert_done(&models.complete_simple(&model, &hi(), simple(base())).await);

    let request = last_request(&server).await;
    assert_eq!(request.url.path(), "/v1/account-id/gateway-id/compat/chat/completions");
}

#[tokio::test]
async fn preserves_inline_upstream_authorization_for_ai_gateway_byok_requests() {
    let server = serve(responses_body()).await;
    let models = models(GATEWAY_ENV);
    let model = catalog_model(&models, "cloudflare-ai-gateway", "gpt-5.1", &server);
    let options = StreamOptions {
        headers: Some(
            [("Authorization".to_owned(), Some("Bearer upstream-token".to_owned()))]
                .into_iter()
                .collect(),
        ),
        ..base()
    };

    assert_done(&models.complete_simple(&model, &hi(), simple(options)).await);

    let request = last_request(&server).await;
    assert_eq!(request.url.path(), "/v1/account-id/gateway-id/openai/responses");
    assert_eq!(header(&request, "authorization"), Some("Bearer upstream-token"));
    assert_eq!(header(&request, "cf-aig-authorization"), Some("Bearer cf-token"));
}

#[tokio::test]
async fn sends_session_affinity_headers_for_workers_ai_through_ai_gateway() {
    let server = serve(completions_body()).await;
    let models = models(GATEWAY_ENV);
    let model = catalog_model(
        &models,
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
        &server,
    );
    let options = StreamOptions {
        session_id: Some("session-1".into()),
        ..base()
    };

    assert_done(&models.complete_simple(&model, &hi(), simple(options)).await);

    let request = last_request(&server).await;
    assert_eq!(header(&request, "session_id"), Some("session-1"));
    assert_eq!(header(&request, "x-client-request-id"), Some("session-1"));
    assert_eq!(header(&request, "x-session-affinity"), Some("session-1"));
}

#[tokio::test]
async fn sets_strict_mode_explicitly_for_ai_gateway_openai_responses_tools() {
    let server = serve(responses_body()).await;
    let models = models(GATEWAY_ENV);
    let model = catalog_model(&models, "cloudflare-ai-gateway", "gpt-5.6-sol", &server);
    let parameters = |properties: Value, required: &[&str]| json!({"type": "object", "properties": properties, "required": required});
    let mut constrained = Tool::new(
        "constrained",
        "A constrained tool",
        parameters(json!({"value": {"type": "string"}}), &["value"]),
    );
    constrained.constrained_sampling = Some(ConstrainedSampling::JsonSchema {
        strict: StrictMode::Prefer,
    });
    let context = Context::new(vec![Message::user("Use a tool.")]).with_tools(vec![
        Tool::new(
            "ordinary",
            "An ordinary tool",
            parameters(
                json!({"path": {"type": "string"}, "offset": {"type": "number"}}),
                &["path"],
            ),
        ),
        constrained,
    ]);

    assert_done(&models.complete(&model, &context, base()).await);

    let params = body(&last_request(&server).await);
    let tools: Vec<(&str, bool)> = params["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| (tool["name"].as_str().unwrap(), tool["strict"].as_bool().unwrap()))
        .collect();
    assert_eq!(tools, [("ordinary", false), ("constrained", true)]);
}

#[tokio::test]
async fn ai_gateway_anthropic_models_send_only_the_gateway_token() {
    let server = serve(anthropic_body()).await;
    let models = models(GATEWAY_ENV);
    let model = catalog_model(&models, "cloudflare-ai-gateway", "claude-sonnet-4-5", &server);

    assert_done(&models.complete(&model, &hi(), base()).await);

    let request = last_request(&server).await;
    assert_eq!(request.url.path(), "/v1/account-id/gateway-id/anthropic/v1/messages");
    assert_eq!(header(&request, "cf-aig-authorization"), Some("Bearer cf-token"));
    assert_eq!(header(&request, "authorization"), None);
    assert_eq!(header(&request, "x-api-key"), None);
}

#[tokio::test]
async fn workers_ai_sends_the_token_as_bearer_key_to_the_account_endpoint() {
    let server = serve(completions_body()).await;
    let models = models(&[
        ("CLOUDFLARE_API_KEY", "cf-key"),
        ("CLOUDFLARE_ACCOUNT_ID", "account-id"),
    ]);
    let model = catalog_model(&models, "cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6", &server);

    assert_done(&models.complete(&model, &hi(), base()).await);

    let request = last_request(&server).await;
    assert_eq!(
        request.url.path(),
        "/client/v4/accounts/account-id/ai/v1/chat/completions"
    );
    assert_eq!(header(&request, "authorization"), Some("Bearer cf-key"));
    assert_eq!(header(&request, "cf-aig-authorization"), None);
    assert_eq!(body(&request)["model"], "@cf/moonshotai/kimi-k2.6");
}

#[tokio::test]
async fn request_scoped_env_configures_the_gateway() {
    let server = serve(completions_body()).await;
    let models = models(&[]);
    let model = catalog_model(
        &models,
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
        &server,
    );

    let unconfigured = models.complete(&model, &hi(), base()).await;
    assert_eq!(unconfigured.stop_reason, StopReason::Error);
    assert_eq!(
        unconfigured.error_message.as_deref(),
        Some("Provider is not configured: cloudflare-ai-gateway")
    );

    let env: ProviderEnv = GATEWAY_ENV
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect();
    let options = StreamOptions {
        env: Some(env),
        ..base()
    };
    assert_done(&models.complete(&model, &hi(), options).await);

    let request = last_request(&server).await;
    assert_eq!(request.url.path(), "/v1/account-id/gateway-id/compat/chat/completions");
    assert_eq!(header(&request, "cf-aig-authorization"), Some("Bearer cf-token"));
}
