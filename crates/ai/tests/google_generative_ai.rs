//! End-to-end tests of the `google-generative-ai` API against recorded SSE bodies.
//!
//! Ports of upstream `test/google-raw-stop-reason.test.ts` and
//! `google-thinking-level-map.test.ts` (which mock `@google/genai`), plus the wire-level
//! checks of pi-py's `test_google_streams.py`.
#![allow(clippy::unwrap_used, clippy::too_many_lines)]

use futures::StreamExt;
use futures::future::BoxFuture;
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

use tau_ai::api::google_generative_ai::{GoogleGenerativeAiApi, stream, stream_simple};
use tau_ai::options::{
    ApiOptions, CallbackError, GoogleOptions, GoogleThinking, GoogleThinkingLevel, GoogleToolChoice, OnPayload,
    OnProviderStreamEvent, OnResponse, ProviderResponse, ThinkingBudgets,
};
use tau_ai::types::{AssistantBlock, ImageContent, TextContent, ThinkingContent, ToolCall, ToolResultMessage};
use tau_ai::{
    AssistantMessage, AssistantMessageEvent, Context, Message, Model, ProviderStreams, SimpleStreamOptions, StopReason,
    StreamOptions, ThinkingLevel, Tool, TranscriptContext, UserBlock, UserContent, UserMessage, normalize_context,
};

const MODEL_PATH: &str = "/v1beta/models/gemini-2.5-flash:streamGenerateContent";

fn model(base_url: &str, id: &str) -> Model {
    serde_json::from_value(json!({
        "id": id, "name": id, "api": "google-generative-ai", "provider": "google",
        "baseUrl": base_url, "reasoning": true, "input": ["text", "image"],
        "cost": {"input": 0.3, "output": 2.5, "cacheRead": 0.075, "cacheWrite": 0},
        "contextWindow": 1_048_576, "maxTokens": 65536
    }))
    .unwrap()
}

fn custom_model(id: &str, thinking_level_map: &Value) -> Model {
    serde_json::from_value(json!({
        "id": id, "name": id, "api": "google-generative-ai", "provider": "test-google",
        "baseUrl": "https://example.invalid/v1beta", "reasoning": true, "input": ["text"],
        "thinkingLevelMap": thinking_level_map,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 128_000, "maxTokens": 4096
    }))
    .unwrap()
}

fn hello() -> TranscriptContext {
    normalize_context(&Context::new(vec![Message::user("hello")]))
}

fn options(key: &str) -> StreamOptions {
    StreamOptions {
        api_key: Some(key.into()),
        max_retries: Some(0),
        ..StreamOptions::default()
    }
}

fn sse(chunks: &[Value]) -> ResponseTemplate {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str("data: ");
        body.push_str(&chunk.to_string());
        body.push_str("\r\n\r\n");
    }
    ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
}

fn stop_chunk(finish_reason: &str, function_call: bool) -> Value {
    let mut candidate = json!({"finishReason": finish_reason});
    if function_call {
        candidate["content"] =
            json!({"parts": [{"functionCall": {"id": "call-1", "name": "echo", "args": {"value": "truncated"}}}]});
    }
    json!({
        "responseId": "google-response-id",
        "candidates": [candidate],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 0, "totalTokenCount": 1}
    })
}

/// A server that answers successive requests with `responses`, then 404.
async fn server(responses: Vec<ResponseTemplate>) -> MockServer {
    server_at(MODEL_PATH, responses).await
}

async fn server_at(model_path: &str, responses: Vec<ResponseTemplate>) -> MockServer {
    let server = MockServer::start().await;
    for (index, response) in responses.into_iter().enumerate() {
        Mock::given(method("POST"))
            .and(path(model_path))
            .and(query_param("alt", "sse"))
            .respond_with(response)
            .up_to_n_times(1)
            .with_priority(u8::try_from(index + 1).unwrap())
            .mount(&server)
            .await;
    }
    server
}

fn base_url(server: &MockServer) -> String {
    format!("{}/v1beta", server.uri())
}

async fn collect(mut events: tau_ai::AssistantMessageEventStream) -> (Vec<AssistantMessageEvent>, AssistantMessage) {
    let mut collected = Vec::new();
    while let Some(event) = events.next().await {
        collected.push(event);
    }
    let message = collected
        .last()
        .and_then(AssistantMessageEvent::final_message)
        .cloned()
        .unwrap();
    (collected, message)
}

fn kinds(events: &[AssistantMessageEvent]) -> Vec<&'static str> {
    events.iter().map(AssistantMessageEvent::kind).collect()
}

async fn last_body(server: &MockServer) -> Value {
    server
        .received_requests()
        .await
        .unwrap()
        .last()
        .unwrap()
        .body_json()
        .unwrap()
}

// ---------------------------------------------------------------------------
// Text, thinking, tool calls, usage
// ---------------------------------------------------------------------------

#[tokio::test]
async fn text_and_thinking_stream() {
    let server = server(vec![sse(&[
        json!({"responseId": "resp_1", "candidates": [{"content": {"parts": [
            {"text": "Let me ", "thought": true, "thoughtSignature": "c2ln"}]}}]}),
        json!({"candidates": [{"content": {"parts": [{"text": "think.", "thought": true}]}}]}),
        json!({"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}),
        json!({"candidates": [{"content": {"parts": [{"text": " world", "thoughtSignature": "dGV4dA=="}]},
            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "cachedContentTokenCount": 4, "candidatesTokenCount": 5,
                "thoughtsTokenCount": 3, "totalTokenCount": 18}}),
    ])])
    .await;
    let m = model(&base_url(&server), "gemini-2.5-flash");
    let (events, message) = collect(stream(&m, &hello(), options("secret-key"))).await;
    assert_eq!(
        kinds(&events),
        [
            "start",
            "thinking_start",
            "thinking_delta",
            "thinking_delta",
            "thinking_end",
            "text_start",
            "text_delta",
            "text_delta",
            "text_end",
            "done"
        ]
    );
    assert!(matches!(
        &events[4],
        AssistantMessageEvent::ThinkingEnd { content, signature: Some(signature), .. }
            if content == "Let me think." && signature == "c2ln"
    ));
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("STOP"));
    assert_eq!(message.response_id.as_deref(), Some("resp_1"));
    assert_eq!(message.api, "google-generative-ai");
    assert_eq!(
        message.content,
        vec![
            AssistantBlock::Thinking(ThinkingContent {
                thinking: "Let me think.".into(),
                thinking_signature: Some("c2ln".into()),
                redacted: None,
            }),
            AssistantBlock::Text(TextContent {
                text: "Hello world".into(),
                text_signature: Some("dGV4dA==".into()),
            }),
        ]
    );
    let usage = &message.usage;
    assert_eq!(
        (
            usage.input,
            usage.output,
            usage.cache_read,
            usage.cache_write,
            usage.reasoning,
            usage.total_tokens
        ),
        (6, 8, 4, 0, Some(3), 18)
    );
    let expected_cost = (6.0 * 0.3 + 8.0 * 2.5 + 4.0 * 0.075) / 1_000_000.0;
    assert!((usage.cost.total - expected_cost).abs() < 1e-12);

    let requests = server.received_requests().await.unwrap();
    let request = &requests[0];
    assert_eq!(request.url.path(), MODEL_PATH);
    assert_eq!(request.url.query(), Some("alt=sse"));
    assert!(!request.url.as_str().contains("secret-key"));
    assert_eq!(request.headers.get("x-goog-api-key").unwrap(), "secret-key");
    assert_eq!(request.headers.get("content-type").unwrap(), "application/json");
    assert!(
        request
            .headers
            .get("user-agent")
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("tau/")
    );
    assert_eq!(
        request.body_json::<Value>().unwrap(),
        json!({"contents": [{"role": "user", "parts": [{"text": "hello"}]}], "generationConfig": {}})
    );
}

#[tokio::test]
async fn tool_calls_generated_ids_and_request_shape() {
    let server = server(vec![sse(&[
        json!({"candidates": [{"content": {"parts": [{"text": "Calling"}]}}]}),
        json!({"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "bash", "args": {"command": "ls"}}, "thoughtSignature": "c2ln"},
            {"functionCall": {"id": "dup", "name": "bash", "args": {"command": "pwd"}}},
            {"functionCall": {"id": "dup", "name": "bash"}}
        ]}, "finishReason": "STOP"}]}),
    ])])
    .await;
    let tool = Tool::new(
        "bash",
        "Run a command",
        json!({"type": "object", "properties": {"command": {"type": "string"}}}),
    );
    let context = normalize_context(
        &Context::new(vec![Message::user("go")])
            .with_system_prompt("sys")
            .with_tools(vec![tool]),
    );
    let mut opts = options("k");
    opts.temperature = Some(0.25);
    opts.max_tokens = Some(512);
    opts.api = Some(ApiOptions::GoogleGenerativeAi(GoogleOptions {
        tool_choice: Some(GoogleToolChoice::Auto),
        thinking: None,
    }));
    let m = model(&base_url(&server), "gemini-2.5-flash");
    let (events, message) = collect(GoogleGenerativeAiApi.stream(&m, &context, opts)).await;
    assert_eq!(message.stop_reason, StopReason::ToolUse);
    let calls: Vec<&ToolCall> = message.tool_calls().collect();
    assert_eq!(calls.len(), 3);
    assert!(calls[0].id.starts_with("bash_"));
    assert_eq!(calls[0].thought_signature.as_deref(), Some("c2ln"));
    assert_eq!(Value::Object(calls[0].arguments.clone()), json!({"command": "ls"}));
    assert_eq!(calls[1].id, "dup");
    assert_ne!(calls[2].id, "dup");
    assert_ne!(calls[2].id, calls[0].id);
    assert!(calls[2].arguments.is_empty());
    let deltas: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            AssistantMessageEvent::ToolCallDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, [r#"{"command":"ls"}"#, r#"{"command":"pwd"}"#, "{}"]);
    assert_eq!(kinds(&events)[1..4], ["text_start", "text_delta", "text_end"]);
    assert_eq!(
        last_body(&server).await,
        json!({
            "contents": [{"role": "user", "parts": [{"text": "go"}]}],
            "systemInstruction": {"role": "user", "parts": [{"text": "sys"}]},
            "tools": [{"functionDeclarations": [{"name": "bash", "description": "Run a command",
                "parametersJsonSchema": {"type": "object", "properties": {"command": {"type": "string"}}}}]}],
            "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
            "generationConfig": {"temperature": 0.25, "maxOutputTokens": 512}
        })
    );
}

// ---------------------------------------------------------------------------
// Finish reasons (google-raw-stop-reason.test.ts)
// ---------------------------------------------------------------------------

async fn run_chunks(chunks: &[Value]) -> AssistantMessage {
    let server = server(vec![sse(chunks)]).await;
    let m = model(&base_url(&server), "gemini-2.5-flash");
    stream(&m, &hello(), options("k")).result().await
}

#[tokio::test]
async fn raw_finish_reasons() {
    let message = run_chunks(&[stop_chunk("MALFORMED_FUNCTION_CALL", false)]).await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("MALFORMED_FUNCTION_CALL"));
    assert_eq!(
        message.error_message.as_deref(),
        Some("Provider stopped with: MALFORMED_FUNCTION_CALL")
    );

    let message = run_chunks(&[stop_chunk("SOMETHING_NEW", false)]).await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Provider stopped with: SOMETHING_NEW")
    );

    let message = run_chunks(&[stop_chunk("MAX_TOKENS", true)]).await;
    assert_eq!(message.stop_reason, StopReason::Length);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("MAX_TOKENS"));
    assert_eq!(message.tool_calls().count(), 1);

    let message = run_chunks(&[stop_chunk("STOP", true)]).await;
    assert_eq!(message.stop_reason, StopReason::ToolUse);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("STOP"));
    assert_eq!(message.response_id.as_deref(), Some("google-response-id"));
    assert_eq!(message.tool_calls().next().unwrap().id, "call-1");
}

#[tokio::test]
async fn missing_finish_reason_and_empty_stream() {
    let message = run_chunks(&[json!({"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})]).await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Google stream ended without a finish reason")
    );
    assert_eq!(message.text(), "hi");
    let message = run_chunks(&[]).await;
    assert_eq!(
        message.error_message.as_deref(),
        Some("Google stream ended without a finish reason")
    );
}

#[tokio::test]
async fn lenient_parsing_and_held_usage() {
    let server = server(vec![
        ResponseTemplate::new(200).set_body_raw(
            concat!(
                ": keep-alive\n\n",
                "data: {\"candidates\":[{\"content\":{\"parts\":[{\"executableCode\":{\"code\":\"x\"}},{\"text\":\"a\"}]}}],",
                "\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":1,\"totalTokenCount\":6},",
                "\"somethingNew\":true}\n\n",
                "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"b\"}]},\"finishReason\":\"STOP\"}],",
                "\"usageMetadata\":{\"candidatesTokenCount\":2}}\n\n",
                "data: [DONE]\n\n"
            ),
            "text/event-stream",
        ),
    ])
    .await;
    let m = model(&base_url(&server), "gemini-2.5-flash");
    let message = stream(&m, &hello(), options("k")).result().await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    assert_eq!(message.text(), "ab");
    // The last report has no promptTokenCount, so the complete earlier one is kept.
    assert_eq!(
        (message.usage.input, message.usage.output, message.usage.total_tokens),
        (5, 1, 6)
    );
}

#[tokio::test]
async fn forwards_provider_stream_events_in_order() {
    let chunks = [
        json!({"responseId": "resp_google", "candidates": [{"content": {"parts": [{"text": "hello"}]}}]}),
        json!({"candidates": [{"finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3}}),
    ];
    let server = server(vec![sse(&chunks)]).await;
    let received: Arc<Mutex<Vec<(Value, String)>>> = Arc::default();
    let sink = received.clone();
    let hook: OnProviderStreamEvent = Arc::new(move |event: &Value, model: &Model| -> BoxFuture<'static, _> {
        let sink = sink.clone();
        let entry = (event.clone(), model.id.clone());
        Box::pin(async move {
            tokio::task::yield_now().await;
            sink.lock().unwrap().push(entry);
            Ok(())
        })
    });
    let m = model(&base_url(&server), "gemini-2.5-flash");
    let mut opts = options("k");
    opts.on_provider_stream_event = Some(hook);
    let message = stream(&m, &hello(), opts).result().await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.response_id.as_deref(), Some("resp_google"));
    assert_eq!(message.content, vec![AssistantBlock::Text(TextContent::new("hello"))]);
    let received = received.lock().unwrap().clone();
    assert_eq!(
        received,
        chunks
            .iter()
            .map(|chunk| (chunk.clone(), "gemini-2.5-flash".to_owned()))
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Headers and endpoints
// ---------------------------------------------------------------------------

#[tokio::test]
async fn header_overrides() {
    let server = server(vec![sse(&[stop_chunk("STOP", false)])]).await;
    let m = model(&base_url(&server), "gemini-2.5-flash");
    let mut opts = options("k");
    opts.headers = Some(
        [
            ("User-Agent".to_owned(), Some("custom-agent".to_owned())),
            ("x-goog-api-key".to_owned(), Some("mine".to_owned())),
            ("x-extra".to_owned(), Some("1".to_owned())),
        ]
        .into_iter()
        .collect(),
    );
    let message = stream(&m, &hello(), opts).result().await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    let request = &server.received_requests().await.unwrap()[0];
    assert_eq!(request.headers.get("user-agent").unwrap(), "custom-agent");
    assert_eq!(request.headers.get("x-goog-api-key").unwrap(), "mine");
    assert_eq!(request.headers.get("x-extra").unwrap(), "1");
}

#[tokio::test]
async fn env_base_url_when_model_has_none() {
    let server = server(vec![sse(&[stop_chunk("STOP", false)])]).await;
    let m = model("", "gemini-2.5-flash");
    let mut opts = options("k");
    opts.env = Some([("GOOGLE_GEMINI_BASE_URL".to_owned(), server.uri())].into());
    let message = stream(&m, &hello(), opts).result().await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
}

#[tokio::test]
async fn missing_api_key() {
    let m = model("https://example.invalid/v1beta", "gemini-2.5-flash");
    let message = stream(&m, &hello(), StreamOptions::default()).result().await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("No API key for provider: google")
    );
    let message = stream_simple(&m, &hello(), SimpleStreamOptions::default())
        .result()
        .await;
    assert_eq!(
        message.error_message.as_deref(),
        Some("No API key for provider: google")
    );
}

// ---------------------------------------------------------------------------
// Errors, retries, abort
// ---------------------------------------------------------------------------

#[tokio::test]
async fn http_errors_carry_status_and_body_message() {
    let body = json!({"error": {"code": 400, "message": "API key not valid.", "status": "INVALID_ARGUMENT"}});
    let server = server(vec![ResponseTemplate::new(400).set_body_json(body)]).await;
    let m = model(&base_url(&server), "gemini-2.5-flash");
    let message = stream(&m, &hello(), options("k")).result().await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_message.as_deref(), Some("400 API key not valid."));
    assert!(message.content.is_empty());

    let server = server_html_error().await;
    let m = model(&base_url(&server), "gemini-2.5-flash");
    let message = stream(&m, &hello(), options("k")).result().await;
    assert_eq!(message.error_message.as_deref(), Some("502 bad gateway"));
}

async fn server_html_error() -> MockServer {
    server(vec![
        ResponseTemplate::new(502).set_body_raw("bad gateway", "text/html"),
    ])
    .await
}

#[tokio::test]
async fn retries_retryable_status() {
    let server = server(vec![
        ResponseTemplate::new(503)
            .set_body_json(json!({"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}))
            .insert_header("retry-after-ms", "10"),
        sse(&[stop_chunk("STOP", false)]),
    ])
    .await;
    let m = model(&base_url(&server), "gemini-2.5-flash");
    let mut opts = options("k");
    opts.max_retries = Some(1);
    let message = stream(&m, &hello(), opts).result().await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn in_stream_error_is_reported_and_not_retried() {
    let error_chunk = json!({"error": {"code": 429, "message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}});
    let server = server(vec![
        sse(&[
            json!({"candidates": [{"content": {"parts": [{"text": "x"}]}}]}),
            error_chunk.clone(),
        ]),
        sse(&[stop_chunk("STOP", false)]),
    ])
    .await;
    let m = model(&base_url(&server), "gemini-2.5-flash");
    let mut opts = options("k");
    opts.max_retries = Some(2);
    let (events, message) = collect(stream(&m, &hello(), opts)).await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message,
        Some(format!("got status: RESOURCE_EXHAUSTED. {error_chunk}"))
    );
    assert_eq!(message.text(), "x");
    assert_eq!(kinds(&events), ["start", "text_start", "text_delta", "error"]);
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn cancelled_before_the_request() {
    let cancel = CancellationToken::new();
    cancel.cancel();
    let mut opts = options("k");
    opts.cancel = Some(cancel);
    let m = model("https://example.invalid/v1beta", "gemini-2.5-flash");
    let (events, message) = collect(stream(&m, &hello(), opts)).await;
    assert_eq!(kinds(&events), ["error"]);
    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert_eq!(message.error_message.as_deref(), Some("Request was aborted"));
}

/// A server that sends one SSE event and then holds the response open until dropped.
fn hanging_server(first_event: &str) -> (String, std::sync::mpsc::Sender<()>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (done, wait) = std::sync::mpsc::channel::<()>();
    let event = format!("data: {first_event}\n\n");
    std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        let mut buffer = vec![0u8; 65536];
        let _ = socket.read(&mut buffer);
        let chunk = format!("{:x}\r\n{event}\r\n", event.len());
        let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n";
        socket.write_all(head.as_bytes()).unwrap();
        socket.write_all(chunk.as_bytes()).unwrap();
        socket.flush().unwrap();
        let _ = wait.recv_timeout(Duration::from_secs(30));
    });
    (format!("http://{address}/v1beta"), done)
}

#[tokio::test]
async fn abort_mid_stream_keeps_partial_content() {
    let (url, done) = hanging_server(r#"{"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}"#);
    let cancel = CancellationToken::new();
    let mut opts = options("k");
    opts.cancel = Some(cancel.clone());
    let m = model(&url, "gemini-2.5-flash");
    let mut events = stream(&m, &hello(), opts);
    let mut kinds_seen = Vec::new();
    let mut final_message = None;
    while let Some(event) = events.next().await {
        kinds_seen.push(event.kind());
        if matches!(event, AssistantMessageEvent::TextDelta { .. }) {
            cancel.cancel();
        }
        if let Some(message) = event.final_message() {
            final_message = Some(message.clone());
        }
    }
    drop(done);
    let message = final_message.unwrap();
    assert_eq!(kinds_seen, ["start", "text_start", "text_delta", "error"]);
    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert_eq!(message.error_message.as_deref(), Some("Request was aborted"));
    assert_eq!(message.text(), "partial");
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

#[tokio::test]
async fn on_payload_replacement_and_on_response() {
    let server = server(vec![
        sse(&[stop_chunk("STOP", false)]).insert_header("x-request-id", "req-1"),
    ])
    .await;
    let on_payload: OnPayload = Arc::new(|payload: &Value, _: &Model| -> BoxFuture<'static, _> {
        let mut payload = payload.clone();
        payload["generationConfig"]["topK"] = json!(3);
        Box::pin(async move { Ok(Some(payload)) })
    });
    let responses: Arc<Mutex<Vec<ProviderResponse>>> = Arc::default();
    let sink = responses.clone();
    let on_response: OnResponse = Arc::new(move |response: &ProviderResponse, _: &Model| -> BoxFuture<'static, _> {
        sink.lock().unwrap().push(response.clone());
        Box::pin(async { Ok(()) })
    });
    let m = model(&base_url(&server), "gemini-2.5-flash");
    let mut opts = options("k");
    opts.on_payload = Some(on_payload);
    opts.on_response = Some(on_response);
    let message = stream(&m, &hello(), opts).result().await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(last_body(&server).await["generationConfig"], json!({"topK": 3}));
    let responses = responses.lock().unwrap();
    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0].status, 200);
    assert_eq!(
        responses[0].headers.get("x-request-id").map(String::as_str),
        Some("req-1")
    );
}

// ---------------------------------------------------------------------------
// Thinking configuration through stream_simple (google-thinking-level-map.test.ts)
// ---------------------------------------------------------------------------

async fn capture_payload(model: &Model, reasoning: Option<ThinkingLevel>, budgets: Option<ThinkingBudgets>) -> Value {
    let captured: Arc<Mutex<Option<Value>>> = Arc::default();
    let sink = captured.clone();
    let on_payload: OnPayload = Arc::new(move |payload: &Value, _: &Model| -> BoxFuture<'static, _> {
        *sink.lock().unwrap() = Some(payload.clone());
        Box::pin(async { Err::<Option<Value>, CallbackError>("payload captured".into()) })
    });
    let options = SimpleStreamOptions {
        base: StreamOptions {
            api_key: Some("test".into()),
            on_payload: Some(on_payload),
            ..StreamOptions::default()
        },
        reasoning,
        thinking_budgets: budgets,
        ..SimpleStreamOptions::default()
    };
    let result = stream_simple(model, &hello(), options).result().await;
    assert!(
        result
            .error_message
            .as_deref()
            .unwrap_or_default()
            .contains("payload captured"),
        "{:?}",
        result.error_message
    );
    captured.lock().unwrap().take().unwrap()
}

fn thinking_config(payload: &Value) -> &Value {
    &payload["generationConfig"]["thinkingConfig"]
}

#[tokio::test]
async fn simple_thinking_levels() {
    let no_minimal = json!({"off": null, "minimal": null, "low": "low", "medium": "medium", "high": "high",
        "xhigh": null, "max": null});
    // Lowest supported level when reasoning is omitted (pi #9455).
    let payload = capture_payload(&custom_model("gemini-3.8-flash", &no_minimal), None, None).await;
    assert_eq!(thinking_config(&payload), &json!({"thinkingLevel": "LOW"}));
    // Native medium effort for Gemini 3.1 Pro.
    let payload = capture_payload(
        &custom_model("gemini-3.1-pro-preview", &no_minimal),
        Some(ThinkingLevel::Medium),
        None,
    )
    .await;
    assert_eq!(
        thinking_config(&payload),
        &json!({"includeThoughts": true, "thinkingLevel": "MEDIUM"})
    );
    // Gemini 2.5 thinking is disabled when reasoning is omitted.
    let payload = capture_payload(&custom_model("gemini-2.5-flash", &json!({})), None, None).await;
    assert_eq!(thinking_config(&payload), &json!({"thinkingBudget": 0}));
    // Gemini 4 uses thinking levels (pi #10890).
    let payload = capture_payload(
        &custom_model("gemini-4-argon", &json!({})),
        Some(ThinkingLevel::Medium),
        None,
    )
    .await;
    assert_eq!(
        thinking_config(&payload),
        &json!({"includeThoughts": true, "thinkingLevel": "MEDIUM"})
    );
    for reasoning in [ThinkingLevel::Xhigh, ThinkingLevel::Max] {
        let payload = capture_payload(
            &custom_model("gemini-3.7-flash", &json!({"xhigh": "high", "max": "high"})),
            Some(reasoning),
            None,
        )
        .await;
        assert_eq!(
            thinking_config(&payload),
            &json!({"includeThoughts": true, "thinkingLevel": "HIGH"})
        );
    }
    // Uppercase provider values for standard levels.
    let payload = capture_payload(
        &custom_model("gemini-3.7-flash", &json!({"high": "LOW"})),
        Some(ThinkingLevel::High),
        None,
    )
    .await;
    assert_eq!(thinking_config(&payload)["thinkingLevel"], "LOW");
    // Mapped levels for token budgets.
    let payload = capture_payload(
        &custom_model("gemini-2.5-flash", &json!({"xhigh": "high"})),
        Some(ThinkingLevel::Xhigh),
        Some(ThinkingBudgets {
            high: Some(1234),
            ..ThinkingBudgets::default()
        }),
    )
    .await;
    assert_eq!(
        thinking_config(&payload),
        &json!({"includeThoughts": true, "thinkingBudget": 1234})
    );
    // Default budgets.
    let payload = capture_payload(
        &custom_model("gemini-2.5-pro", &json!({})),
        Some(ThinkingLevel::Low),
        None,
    )
    .await;
    assert_eq!(
        thinking_config(&payload),
        &json!({"includeThoughts": true, "thinkingBudget": 2048})
    );
}

#[tokio::test]
async fn simple_options_map_tool_choice_and_max_tokens() {
    let tool = Tool::new("t", "d", json!({"type": "object"}));
    let context = normalize_context(&Context::new(vec![Message::user("hi")]).with_tools(vec![tool]));
    let captured: Arc<Mutex<Option<Value>>> = Arc::default();
    let sink = captured.clone();
    let on_payload: OnPayload = Arc::new(move |payload: &Value, _: &Model| -> BoxFuture<'static, _> {
        *sink.lock().unwrap() = Some(payload.clone());
        Box::pin(async { Err::<Option<Value>, CallbackError>("stop".into()) })
    });
    let options = SimpleStreamOptions {
        base: StreamOptions {
            api_key: Some("test".into()),
            on_payload: Some(on_payload),
            max_tokens: Some(100),
            ..StreamOptions::default()
        },
        tool_choice: Some(tau_ai::ToolChoice::None),
        ..SimpleStreamOptions::default()
    };
    let m = custom_model("gemini-2.5-flash", &json!({}));
    let _ = stream_simple(&m, &context, options).result().await;
    let payload = captured.lock().unwrap().take().unwrap();
    assert_eq!(
        payload["toolConfig"],
        json!({"functionCallingConfig": {"mode": "NONE"}})
    );
    assert_eq!(payload["generationConfig"]["maxOutputTokens"], 100);
}

#[tokio::test]
async fn invalid_thinking_mapping_fails_the_stream() {
    let m = custom_model("gemini-3.7-flash", &json!({"xhigh": "extreme"}));
    let options = SimpleStreamOptions {
        base: options("k"),
        reasoning: Some(ThinkingLevel::Xhigh),
        ..SimpleStreamOptions::default()
    };
    let (events, message) = collect(stream_simple(&m, &hello(), options)).await;
    assert_eq!(kinds(&events), ["error"]);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Unsupported Google thinking level mapping for test-google/gemini-3.7-flash: xhigh -> extreme")
    );
}

#[tokio::test]
async fn explicit_thinking_options() {
    let server = server(vec![sse(&[stop_chunk("STOP", false)])]).await;
    let m = model(&base_url(&server), "gemini-2.5-flash");
    let mut opts = options("k");
    opts.api = Some(ApiOptions::GoogleGenerativeAi(GoogleOptions {
        tool_choice: Some(GoogleToolChoice::Any),
        thinking: Some(GoogleThinking {
            enabled: true,
            budget_tokens: Some(-1),
            level: None,
        }),
    }));
    stream(&m, &hello(), opts.clone()).result().await;
    let body = last_body(&server).await;
    assert_eq!(
        body["generationConfig"],
        json!({"thinkingConfig": {"includeThoughts": true, "thinkingBudget": -1}})
    );
    // No tools: no toolConfig even with a tool choice.
    assert!(body.get("toolConfig").is_none());

    // Options of another API are ignored.
    let server2 = server_ok().await;
    let m = model(&base_url(&server2), "gemini-2.5-flash");
    opts.api = Some(ApiOptions::GoogleVertex(tau_ai::options::GoogleVertexOptions {
        thinking: Some(GoogleThinking {
            enabled: true,
            budget_tokens: None,
            level: Some(GoogleThinkingLevel::High),
        }),
        ..Default::default()
    }));
    stream(&m, &hello(), opts).result().await;
    assert_eq!(last_body(&server2).await["generationConfig"], json!({}));
}

async fn server_ok() -> MockServer {
    server(vec![sse(&[stop_chunk("STOP", false)])]).await
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

#[tokio::test]
async fn thought_signature_round_trip() {
    let server = server_at(
        "/v1beta/models/gemini-3-flash-preview:streamGenerateContent",
        vec![
            sse(&[json!({"candidates": [{"content": {"parts": [
                {"text": "a", "thoughtSignature": "AAAAAAAA/+8="},
                {"functionCall": {"id": "c1", "name": "read", "args": {"path": "x"}}, "thoughtSignature": "c2lnMQ=="}
            ]}, "finishReason": "STOP"}]})]),
            sse(&[stop_chunk("STOP", false)]),
        ],
    )
    .await;
    let m = model(&base_url(&server), "gemini-3-flash-preview");
    let first = stream(&m, &hello(), options("k")).result().await;
    assert_eq!(first.stop_reason, StopReason::ToolUse);
    let result = ToolResultMessage {
        tool_call_id: "c1".into(),
        tool_name: "read".into(),
        content: vec![UserBlock::text("contents")],
        ..ToolResultMessage::default()
    };
    let context = normalize_context(&Context::new(vec![
        Message::user("hello"),
        Message::Assistant(first),
        Message::ToolResult(result),
    ]));
    let message = stream(&m, &context, options("k")).result().await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(
        last_body(&server).await["contents"],
        json!([
            {"role": "user", "parts": [{"text": "hello"}]},
            {"role": "model", "parts": [
                {"text": "a", "thoughtSignature": "AAAAAAAA/+8="},
                {"functionCall": {"name": "read", "args": {"path": "x"}, "id": "c1"}, "thoughtSignature": "c2lnMQ=="}
            ]},
            {"role": "user", "parts": [{"functionResponse": {"name": "read", "response": {"output": "contents"},
                "id": "c1"}}]}
        ])
    );
}

#[tokio::test]
async fn cross_provider_replay_images_and_orphans() {
    let server = server(vec![sse(&[stop_chunk("STOP", false)])]).await;
    let m = model(&base_url(&server), "gemini-2.5-flash");
    let mut foreign = AssistantMessage::empty("anthropic-messages", "anthropic", "claude-sonnet-4-5");
    foreign.stop_reason = StopReason::ToolUse;
    foreign.content = vec![
        AssistantBlock::Thinking(ThinkingContent {
            thinking: "secret reasoning".into(),
            thinking_signature: Some("c2ln".into()),
            redacted: None,
        }),
        AssistantBlock::Text(TextContent {
            text: "Let me look.".into(),
            text_signature: Some("c2ln".into()),
        }),
        AssistantBlock::ToolCall(ToolCall {
            id: "toolu_01|x".into(),
            name: "read".into(),
            arguments: json!({"path": "a.png"}).as_object().unwrap().clone(),
            thought_signature: Some("c2ln".into()),
            ..ToolCall::default()
        }),
        AssistantBlock::ToolCall(ToolCall {
            id: "toolu_02".into(),
            name: "read".into(),
            ..ToolCall::default()
        }),
    ];
    let image = ImageContent {
        data: "iVBORw0KGgo=".into(),
        mime_type: "image/png".into(),
    };
    let context = normalize_context(
        &Context::new(vec![
            Message::User(UserMessage {
                content: UserContent::Blocks(vec![UserBlock::text("what is this?"), UserBlock::Image(image.clone())]),
                timestamp: 0,
            }),
            Message::Assistant(foreign),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "toolu_01|x".into(),
                tool_name: "read".into(),
                content: vec![UserBlock::Image(image)],
                ..ToolResultMessage::default()
            }),
            Message::user("thanks"),
        ])
        .with_system_prompt("be brief"),
    );
    let message = stream(&m, &context, options("k")).result().await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    let body = last_body(&server).await;
    assert_eq!(
        body["contents"],
        json!([
            {"role": "user", "parts": [{"text": "what is this?"},
                {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}]},
            // Foreign thinking is dropped; signatures are not replayed; Gemini 2.5 gets no ids.
            {"role": "model", "parts": [
                {"text": "Let me look."},
                {"functionCall": {"name": "read", "args": {"path": "a.png"}}},
                {"functionCall": {"name": "read", "args": {}}}
            ]},
            {"role": "user", "parts": [
                {"functionResponse": {"name": "read", "response": {"output": "(see attached image)"}}}
            ]},
            // Gemini < 3: tool result images follow in their own user turn, which ends the
            // run of merged function responses (as upstream).
            {"role": "user", "parts": [{"text": "Tool result image:"},
                {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}]},
            // The orphaned call gets a synthetic error result.
            {"role": "user", "parts": [
                {"functionResponse": {"name": "read", "response": {"error": "No result provided"}}}
            ]},
            {"role": "user", "parts": [{"text": "thanks"}]}
        ])
    );
    assert_eq!(
        body["systemInstruction"],
        json!({"role": "user", "parts": [{"text": "be brief"}]})
    );
}

#[tokio::test]
async fn mid_conversation_system_messages_are_collapsed() {
    let server = server(vec![sse(&[stop_chunk("STOP", false)])]).await;
    let m = model(&base_url(&server), "gemini-2.5-flash");
    let context = normalize_context(
        &Context::new(vec![
            Message::user("one"),
            Message::System(tau_ai::SystemMessage {
                content: "Also be kind.".into(),
                tools_added: Some(vec![Tool::new("late", "Late tool", json!({"type": "object"}))]),
                ..tau_ai::SystemMessage::default()
            }),
            Message::user("two"),
        ])
        .with_system_prompt("Be brief."),
    );
    stream(&m, &context, options("k")).result().await;
    let body = last_body(&server).await;
    assert_eq!(
        body["systemInstruction"],
        json!({"role": "user", "parts": [{"text": "Be brief.\n\nAlso be kind."}]})
    );
    assert_eq!(
        body["contents"],
        json!([{"role": "user", "parts": [{"text": "one"}]}, {"role": "user", "parts": [{"text": "two"}]}])
    );
    assert_eq!(body["tools"][0]["functionDeclarations"][0]["name"], "late");
}
