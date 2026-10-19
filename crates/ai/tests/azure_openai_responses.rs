//! End-to-end tests of the `azure-openai-responses` API against recorded SSE bodies.

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

use futures::StreamExt;
use futures::future::BoxFuture;
use serde_json::{Value, json};
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use tau_ai::api::azure_openai_responses::{self, AzureOpenAIResponsesApi};
use tau_ai::options::{AzureOpenAIResponsesOptions, OpenAIResponsesOptions};
use tau_ai::types::ProviderHeaders;
use tau_ai::{
    ApiOptions, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream, Context, Message, Model,
    ProviderStreams, SimpleStreamOptions, StopReason, StreamOptions, Tool, ToolChoice, normalize_context,
};
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn model() -> Model {
    serde_json::from_value(json!({
        "id": "gpt-4o-mini", "name": "GPT-4o mini", "api": "azure-openai-responses", "provider": "azure",
        "baseUrl": "", "reasoning": false, "input": ["text", "image"],
        "cost": {"input": 0.15, "output": 0.6, "cacheRead": 0.075, "cacheWrite": 0},
        "contextWindow": 128_000, "maxTokens": 16_384
    }))
    .unwrap()
}

fn sse(events: &[Value]) -> String {
    events.iter().fold(String::new(), |mut body, event| {
        let _ = write!(body, "data: {event}\n\n");
        body
    })
}

fn completed_events() -> Vec<Value> {
    vec![
        json!({"type": "response.created", "response": {"id": "resp_azure"}}),
        json!({"type": "response.completed", "response": {"id": "resp_azure", "status": "completed"}}),
    ]
}

async fn server_with(body: String) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;
    server
}

fn azure_options(server: &MockServer, azure: AzureOpenAIResponsesOptions) -> StreamOptions {
    StreamOptions {
        api_key: Some("test-api-key".into()),
        max_retries: Some(0),
        api: Some(ApiOptions::AzureOpenAIResponses(AzureOpenAIResponsesOptions {
            azure_base_url: Some(format!("{}/openai/v1", server.uri())),
            ..azure
        })),
        ..StreamOptions::default()
    }
}

fn hello() -> tau_ai::TranscriptContext {
    normalize_context(&Context::new(vec![Message::user("hello")]))
}

async fn collect(mut stream: AssistantMessageEventStream) -> (Vec<AssistantMessageEvent>, AssistantMessage) {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    let message = events
        .last()
        .and_then(AssistantMessageEvent::final_message)
        .cloned()
        .unwrap();
    (events, message)
}

async fn last_request(server: &MockServer) -> Request {
    server.received_requests().await.unwrap().pop().unwrap()
}

fn body(request: &Request) -> Value {
    serde_json::from_slice(&request.body).unwrap()
}

fn header<'a>(request: &'a Request, name: &str) -> Option<&'a str> {
    request.headers.get(name).and_then(|value| value.to_str().ok())
}

#[tokio::test]
async fn request_wire_format() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/responses"))
        .and(query_param("api-version", "2025-04-01-preview"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse(&completed_events()), "text/event-stream"))
        .mount(&server)
        .await;
    let mut options = azure_options(
        &server,
        AzureOpenAIResponsesOptions {
            azure_api_version: Some("2025-04-01-preview".into()),
            azure_deployment_name: Some("prod-mini".into()),
            responses: OpenAIResponsesOptions {
                tool_choice: Some(json!("required")),
                service_tier: Some("priority".into()),
                ..OpenAIResponsesOptions::default()
            },
            ..AzureOpenAIResponsesOptions::default()
        },
    );
    options.session_id = Some("x".repeat(67));
    options.max_tokens = Some(8);
    let tool = Tool::new(
        "read",
        "Read a file",
        json!({"type": "object", "properties": {"path": {"type": "string"}}}),
    );
    let context = normalize_context(&Context::new(vec![Message::user("hello")]).with_tools(vec![tool]));
    let (events, message) = collect(AzureOpenAIResponsesApi.stream(&model(), &context, options)).await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    assert_eq!(message.api, "azure-openai-responses");
    assert_eq!(message.response_id.as_deref(), Some("resp_azure"));
    assert_eq!(events.first().unwrap().kind(), "start");

    let request = last_request(&server).await;
    assert_eq!(header(&request, "api-key"), Some("test-api-key"));
    assert_eq!(header(&request, "authorization"), None);
    assert!(header(&request, "user-agent").unwrap().starts_with("tau/"));
    let body = body(&request);
    assert_eq!(
        body,
        json!({
            "model": "prod-mini",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
            "stream": true,
            "prompt_cache_key": "x".repeat(64),
            "store": false,
            "max_output_tokens": 16,
            "tools": [{"type": "function", "name": "read", "description": "Read a file",
                       "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}, "strict": false}],
            "tool_choice": "required"
        })
    );
}

#[tokio::test]
async fn deployment_map_reasoning_and_strict_compat() {
    let server = server_with(sse(&completed_events())).await;
    let mut target = model();
    target.reasoning = true;
    target.compat = Some(
        tau_ai::compat::ModelCompat::from_value("azure-openai-responses", json!({"supportsStrictMode": false}))
            .unwrap(),
    );
    let mut options = azure_options(&server, AzureOpenAIResponsesOptions::default());
    options.env = Some(
        [(
            "AZURE_OPENAI_DEPLOYMENT_NAME_MAP".to_owned(),
            "gpt-4o-mini=mapped-deploy".to_owned(),
        )]
        .into_iter()
        .collect(),
    );
    let mut tool = Tool::new(
        "preferred",
        "d",
        json!({"type": "object", "properties": {"value": {"type": "string"}}}),
    );
    tool.constrained_sampling = Some(tau_ai::types::ConstrainedSampling::JsonSchema {
        strict: tau_ai::types::StrictMode::Prefer,
    });
    let context = normalize_context(&Context::new(vec![Message::user("hello")]).with_tools(vec![tool]));
    collect(azure_openai_responses::stream(&target, &context, options)).await;
    let body = body(&last_request(&server).await);
    assert_eq!(body["model"], "mapped-deploy");
    assert_eq!(body["reasoning"], json!({"effort": "none"}));
    assert!(body["tools"][0].get("strict").is_none());
}

#[tokio::test]
async fn endpoint_errors() {
    let options = StreamOptions {
        api_key: Some("k".into()),
        env: Some(
            [("AZURE_OPENAI_BASE_URL".to_owned(), "not-a-url".to_owned())]
                .into_iter()
                .collect(),
        ),
        ..StreamOptions::default()
    };
    let (events, message) = collect(azure_openai_responses::stream(&model(), &hello(), options)).await;
    assert_eq!(events.len(), 1);
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Invalid Azure OpenAI base URL: not-a-url")
    );

    let (_, message) = collect(azure_openai_responses::stream(
        &model(),
        &hello(),
        StreamOptions::default(),
    ))
    .await;
    assert_eq!(message.error_message.as_deref(), Some("No API key for provider: azure"));
}

#[tokio::test]
async fn http_error_prefix() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(401).set_body_json(json!({"error": {"code": "401", "message": "Access denied"}})),
        )
        .mount(&server)
        .await;
    let (_, message) = collect(azure_openai_responses::stream(
        &model(),
        &hello(),
        azure_options(&server, AzureOpenAIResponsesOptions::default()),
    ))
    .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Azure OpenAI API error (401): Access denied")
    );
}

#[tokio::test]
async fn forwards_provider_events_and_user_agent_override() {
    let server = server_with(sse(&completed_events())).await;
    let seen: Arc<Mutex<Vec<Value>>> = Arc::default();
    let log = seen.clone();
    let mut options = azure_options(&server, AzureOpenAIResponsesOptions::default());
    options.on_provider_stream_event = Some(Arc::new(move |event: &Value, _: &Model| -> BoxFuture<'static, _> {
        log.lock().unwrap().push(event.clone());
        Box::pin(async { Ok(()) })
    }));
    let headers: ProviderHeaders = [("User-Agent".to_owned(), Some("custom-agent".to_owned()))]
        .into_iter()
        .collect();
    options.headers = Some(headers);
    let (_, message) = collect(azure_openai_responses::stream(&model(), &hello(), options)).await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(*seen.lock().unwrap(), completed_events());
    assert_eq!(header(&last_request(&server).await, "user-agent"), Some("custom-agent"));
}

#[tokio::test]
async fn stream_simple_forwards_tool_choice() {
    let server = server_with(sse(&completed_events())).await;
    let mut target = model();
    target.base_url = format!("{}/openai/v1", server.uri());
    let tool = Tool::new("read", "Read a file", json!({"type": "object", "properties": {}}));
    let context = normalize_context(&Context::new(vec![Message::user("Summarize this")]).with_tools(vec![tool]));
    let options = SimpleStreamOptions {
        base: StreamOptions {
            api_key: Some("test-key".into()),
            max_retries: Some(0),
            ..StreamOptions::default()
        },
        tool_choice: Some(ToolChoice::None),
        ..SimpleStreamOptions::default()
    };
    let (_, message) = collect(azure_openai_responses::stream_simple(&target, &context, options)).await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    let body = body(&last_request(&server).await);
    assert_eq!(body["tool_choice"], "none");
    assert_eq!(body["tools"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn reasoning_replay_backfills_encrypted_content() {
    let done = json!({"type": "reasoning", "id": "rs_missing", "summary": []});
    let mut completed_item = done.clone();
    completed_item["encrypted_content"] = json!("from-response-completed");
    let server = server_with(sse(&[
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs_missing", "summary": []}}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": done}),
        json!({"type": "response.output_item.added", "output_index": 1, "item": {"type": "message", "id": "msg_1", "content": []}}),
        json!({"type": "response.output_item.done", "output_index": 1, "item": {"type": "message", "id": "msg_1",
               "content": [{"type": "output_text", "text": "ok"}]}}),
        json!({"type": "response.completed", "response": {"id": "r", "status": "completed", "output": [completed_item]}}),
    ]))
    .await;
    let mut target = model();
    target.reasoning = true;
    let options = azure_options(&server, AzureOpenAIResponsesOptions::default());
    let (_, message) = collect(azure_openai_responses::stream(&target, &hello(), options.clone())).await;
    assert_eq!(message.stop_reason, StopReason::Stop);

    // Replaying the reply sends the reasoning item with the backfilled encrypted content.
    let context = normalize_context(&Context::new(vec![
        Message::user("first"),
        message.into(),
        Message::user("follow-up"),
    ]));
    collect(azure_openai_responses::stream(&target, &context, options)).await;
    let body = body(&last_request(&server).await);
    let reasoning = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "reasoning")
        .unwrap();
    assert_eq!(reasoning["id"], "rs_missing");
    assert_eq!(reasoning["encrypted_content"], "from-response-completed");
}
