//! End-to-end tests of the `pi-messages` API against local servers. Ported from upstream
//! `test/pi-messages.test.ts`, plus cases for thinking, parallel tool calls, cancellation and
//! the request options.

#![allow(clippy::unwrap_used)] // Test helpers outside `#[test]` functions.

use delta_ai::api::pi_messages::{PiMessagesApi, PiMessagesEvent, stream, stream_simple};
use delta_ai::options::{OnProviderStreamEvent, OnResponse, PiMessagesOptions};
use delta_ai::{
    ApiOptions, AssistantMessageEvent, CacheRetention, Context, Message, Model, ProviderStreams, SimpleStreamOptions,
    StopReason, StreamOptions, ThinkingLevel, ToolChoice, TranscriptContext, UserMessage, normalize_context,
};
use futures::StreamExt;
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

fn model(base_url: &str) -> Model {
    serde_json::from_value(json!({
        "id": "auto",
        "name": "Radius Auto",
        "api": "pi-messages",
        "provider": "radius",
        "baseUrl": base_url,
        "reasoning": false,
        "input": ["text"],
        "cost": {"input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 0.2},
        "contextWindow": 128_000,
        "maxTokens": 16_384
    }))
    .unwrap()
}

fn context() -> Context {
    Context::new(vec![Message::User(UserMessage {
        content: "Hello".into(),
        timestamp: 1,
    })])
}

fn transcript() -> TranscriptContext {
    normalize_context(&context())
}

fn usage() -> Value {
    json!({
        "input": 10,
        "output": 5,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 15,
        "cost": {"input": 0.1, "output": 0.2, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.3}
    })
}

fn sse(events: &[Value]) -> String {
    events.iter().fold(String::new(), |mut body, event| {
        body.push_str("data: ");
        body.push_str(&event.to_string());
        body.push_str("\n\n");
        body
    })
}

async fn server_with(template: ResponseTemplate) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(template)
        .mount(&server)
        .await;
    server
}

async fn sse_server(events: &[Value]) -> MockServer {
    server_with(ResponseTemplate::new(200).set_body_raw(sse(events), "text/event-stream")).await
}

fn base_url(server: &MockServer) -> String {
    format!("{}/v1", server.uri())
}

/// Options with an API key. Each test gets its own HTTP client: pooled connections of the
/// shared default client belong to the Tokio runtime of the test that opened them.
fn options(api_key: &str) -> StreamOptions {
    StreamOptions {
        api_key: Some(api_key.into()),
        http_client: Some(reqwest::Client::builder().no_proxy().build().unwrap()),
        ..StreamOptions::default()
    }
}

fn pi_options(options: PiMessagesOptions) -> ApiOptions {
    ApiOptions::PiMessages(options)
}

async fn last_request(server: &MockServer) -> wiremock::Request {
    server.received_requests().await.unwrap().pop().unwrap()
}

/// A server that sends `chunks` with delays, then keeps the connection open.
fn stalling_server(chunks: Vec<(String, Duration)>) -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        socket.set_nodelay(true).unwrap();
        let mut request = Vec::new();
        let mut buffer = [0u8; 4096];
        loop {
            let read = socket.read(&mut buffer).unwrap();
            request.extend_from_slice(&buffer[..read]);
            let text = String::from_utf8_lossy(&request).to_ascii_lowercase();
            if let Some(end) = text.find("\r\n\r\n") {
                let length = text
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length:"))
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                if request.len() >= end + 4 + length {
                    break;
                }
            }
            if read == 0 {
                break;
            }
        }
        let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n";
        socket.write_all(head.as_bytes()).unwrap();
        for (chunk, delay) in chunks {
            std::thread::sleep(delay);
            let frame = format!("{:x}\r\n{chunk}\r\n", chunk.len());
            if socket.write_all(frame.as_bytes()).is_err() {
                return;
            }
        }
        std::thread::sleep(Duration::from_secs(30));
    });
    format!("http://{address}/v1")
}

// ---------------------------------------------------------------------------------------------
// pi-messages.test.ts
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn streams_text_and_tool_calls_and_resolves_the_terminal_message() {
    let server = sse_server(&[
        json!({"type": "start"}),
        json!({"type": "text_start", "contentIndex": 0}),
        json!({"type": "text_delta", "contentIndex": 0, "delta": "Hel"}),
        json!({"type": "text_delta", "contentIndex": 0, "delta": "lo"}),
        json!({"type": "text_end", "contentIndex": 0, "content": "Hello"}),
        json!({"type": "toolcall_start", "contentIndex": 1, "id": "call_1", "toolName": "read"}),
        json!({"type": "toolcall_delta", "contentIndex": 1, "delta": "{\"path\":"}),
        json!({"type": "toolcall_delta", "contentIndex": 1, "delta": "\"a.txt\"}"}),
        json!({"type": "toolcall_end", "contentIndex": 1,
               "toolCall": {"type": "toolCall", "id": "call_1", "name": "read", "arguments": {"path": "a.txt"}}}),
        json!({"type": "done", "reason": "toolUse", "usage": usage(), "responseId": "resp_1",
               "providerThinkingLevel": "high"}),
    ])
    .await;
    let model = model(&base_url(&server));

    let mut events = stream(
        &model,
        &transcript(),
        StreamOptions {
            session_id: Some("session-1".into()),
            max_tokens: Some(100),
            headers: Some([("x-custom".to_owned(), Some("1".to_owned()))].into_iter().collect()),
            api: Some(pi_options(PiMessagesOptions {
                tool_choice: Some(json!("auto")),
                ..PiMessagesOptions::default()
            })),
            ..options("test-key")
        },
    );
    let mut kinds = Vec::new();
    let mut first_partial_stop_reason = None;
    while let Some(event) = events.next().await {
        if first_partial_stop_reason.is_none() && !event.is_terminal() {
            first_partial_stop_reason = events.partial().map(|partial| partial.stop_reason);
        }
        kinds.push(event.kind());
    }
    let message = events.result().await;

    assert_eq!(first_partial_stop_reason, Some(StopReason::Pending));
    assert_eq!(message.stop_reason, StopReason::ToolUse);
    assert_eq!(serde_json::to_value(&message.usage).unwrap(), usage());
    assert_eq!(message.response_id.as_deref(), Some("resp_1"));
    assert_eq!(message.provider_thinking_level.as_deref(), Some("high"));
    assert_eq!(message.model, "auto");
    assert_eq!(message.provider, "radius");
    assert_eq!(
        serde_json::to_value(&message.content).unwrap(),
        json!([
            {"type": "text", "text": "Hello"},
            {"type": "toolCall", "id": "call_1", "name": "read", "arguments": {"path": "a.txt"}}
        ])
    );
    assert!(kinds.contains(&"text_delta"));
    assert_eq!(kinds.iter().filter(|kind| **kind == "toolcall_end").count(), 1);

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.url.path(), "/v1/messages");
    assert_eq!(request.url.query(), None);
    assert_eq!(request.headers["authorization"], "Bearer test-key");
    assert_eq!(request.headers["accept"], "text/event-stream");
    assert_eq!(request.headers["content-type"], "application/json");
    assert_eq!(request.headers["x-custom"], "1");
    assert_eq!(
        request.body_json::<Value>().unwrap(),
        json!({
            "model": "auto",
            "context": {"messages": [{"role": "user", "content": "Hello", "timestamp": 1}]},
            "options": {"maxTokens": 100, "sessionId": "session-1", "toolChoice": "auto"}
        })
    );
}

#[tokio::test]
async fn forwards_parsed_wire_events_in_order_before_converting_them() {
    let wire = vec![
        json!({"type": "start"}),
        json!({"type": "text_start", "contentIndex": 0}),
        json!({"type": "text_delta", "contentIndex": 0, "delta": "Hello", "gatewayField": "upstream-value"}),
        json!({"type": "text_end", "contentIndex": 0, "content": "Hello"}),
        json!({"type": "done", "reason": "stop", "usage": usage(), "responseId": "resp_1"}),
    ];
    let server = sse_server(&wire).await;
    let model = model(&base_url(&server));
    let received: Arc<Mutex<Vec<(Value, String)>>> = Arc::default();
    let sink = received.clone();
    let observer: OnProviderStreamEvent = Arc::new(move |event: &Value, model: &Model| {
        sink.lock().unwrap().push((event.clone(), model.id.clone()));
        Box::pin(async { Ok(()) })
    });

    let message = stream_simple(
        &model,
        &transcript(),
        SimpleStreamOptions::from(StreamOptions {
            on_provider_stream_event: Some(observer),
            ..options("test-key")
        }),
    )
    .result()
    .await;

    let received = received.lock().unwrap().clone();
    assert_eq!(
        received.iter().map(|(event, _)| event.clone()).collect::<Vec<_>>(),
        wire
    );
    assert!(received.iter().all(|(_, id)| id == "auto"));
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.response_id.as_deref(), Some("resp_1"));
    assert_eq!(
        serde_json::to_value(&message.content).unwrap(),
        json!([{"type": "text", "text": "Hello"}])
    );
}

#[tokio::test]
async fn appends_debug_and_reports_response_headers_via_on_response() {
    let server = server_with(
        ResponseTemplate::new(200)
            .insert_header("x-pi-gateway-upstream-provider", "anthropic")
            .set_body_raw(
                sse(&[json!({"type": "done", "reason": "stop", "usage": usage()})]),
                "text/event-stream",
            ),
    )
    .await;
    let observed: Arc<Mutex<Option<(u16, String)>>> = Arc::default();
    let sink = observed.clone();
    let on_response: OnResponse = Arc::new(move |response, _model| {
        let provider = response
            .headers
            .get("x-pi-gateway-upstream-provider")
            .cloned()
            .unwrap_or_default();
        *sink.lock().unwrap() = Some((response.status, provider));
        Box::pin(async { Ok(()) })
    });

    let message = stream_simple(
        &model(&base_url(&server)),
        &transcript(),
        SimpleStreamOptions::from(StreamOptions {
            on_response: Some(on_response),
            api: Some(pi_options(PiMessagesOptions {
                debug: Some(true),
                ..PiMessagesOptions::default()
            })),
            ..options("test-key")
        }),
    )
    .result()
    .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    let request = last_request(&server).await;
    assert_eq!(request.url.path(), "/v1/messages");
    assert_eq!(request.url.query(), Some("debug=1"));
    assert_eq!(observed.lock().unwrap().clone(), Some((200, "anthropic".to_owned())));
}

#[tokio::test]
async fn surfaces_backend_error_responses_with_diagnostics() {
    let server = server_with(ResponseTemplate::new(401).set_body_raw(
        json!({"error": {"message": "Token expired", "code": "unauthorized"}}).to_string(),
        "application/json",
    ))
    .await;
    let observed: Arc<Mutex<Option<u16>>> = Arc::default();
    let sink = observed.clone();
    let on_response: OnResponse = Arc::new(move |response, _model| {
        *sink.lock().unwrap() = Some(response.status);
        Box::pin(async { Ok(()) })
    });

    let message = stream(
        &model(&base_url(&server)),
        &transcript(),
        StreamOptions {
            on_response: Some(on_response),
            ..options("stale")
        },
    )
    .result()
    .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("401 Unauthorized: Token expired (unauthorized)")
    );
    // `on_response` sees error responses too.
    assert_eq!(*observed.lock().unwrap(), Some(401));
    let diagnostics = message.diagnostics.unwrap();
    assert_eq!(diagnostics[0].kind, "pi_messages_response_failure");
    let details = diagnostics[0].details.as_ref().unwrap();
    assert_eq!(details["status"], 401);
    assert_eq!(details["statusText"], "Unauthorized");
    assert_eq!(details["provider"], "radius");
    assert_eq!(details["model"], "auto");
    assert_eq!(details["url"], format!("{}/messages", base_url(&server)));
    assert_eq!(details["version"], 1);
    // Errors are not retried: the backend owns retries.
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn surfaces_non_json_error_bodies_and_does_not_retry_overloads() {
    let server = server_with(ResponseTemplate::new(503).set_body_string("gateway overloaded")).await;
    let message = stream(&model(&base_url(&server)), &transcript(), options("key"))
        .result()
        .await;
    assert_eq!(
        message.error_message.as_deref(),
        Some("503 Service Unavailable: gateway overloaded")
    );
    let diagnostics = message.diagnostics.unwrap();
    assert_eq!(diagnostics[0].details.as_ref().unwrap()["body"], "gateway overloaded");
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn propagates_server_sent_error_events() {
    let server = sse_server(&[
        json!({"type": "start"}),
        json!({"type": "text_start", "contentIndex": 0}),
        json!({"type": "text_delta", "contentIndex": 0, "delta": "partial"}),
        json!({"type": "error", "reason": "error", "usage": usage(), "errorMessage": "Upstream failed"}),
    ])
    .await;

    let message = stream(&model(&base_url(&server)), &transcript(), options("test-key"))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_message.as_deref(), Some("Upstream failed"));
    assert_eq!(serde_json::to_value(&message.usage).unwrap(), usage());
    // The partial content of an error event is kept.
    assert_eq!(message.text(), "partial");
    assert!(message.diagnostics.is_none());
}

#[tokio::test]
async fn propagates_server_sent_aborted_events() {
    let server = sse_server(&[json!({"type": "error", "reason": "aborted", "usage": usage(),
                                     "errorMessage": "Request was cancelled upstream"})])
    .await;
    let message = stream(&model(&base_url(&server)), &transcript(), options("test-key"))
        .result()
        .await;
    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert_eq!(message.error_message.as_deref(), Some("Request was cancelled upstream"));
}

#[tokio::test]
async fn errors_when_no_api_key_is_provided() {
    let message = stream(&model("http://127.0.0.1:1/v1"), &transcript(), StreamOptions::default())
        .result()
        .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("No API key provided for provider \"radius\"")
    );
}

#[tokio::test]
async fn errors_when_the_stream_ends_without_a_terminal_event() {
    let server = sse_server(&[
        json!({"type": "start"}),
        json!({"type": "text_start", "contentIndex": 0}),
        json!({"type": "text_delta", "contentIndex": 0, "delta": "partial"}),
    ])
    .await;

    let message = stream(&model(&base_url(&server)), &transcript(), options("test-key"))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("radius stream ended without a terminal event")
    );
    // Thrown errors report a fresh message, as upstream's `createErrorEvent`.
    assert!(message.content.is_empty());
}

#[test]
fn is_a_provider_streams_implementation() {
    fn assert_streams<T: ProviderStreams>(_: &T) {}
    assert_streams(&PiMessagesApi);
}

// ---------------------------------------------------------------------------------------------
// Further cases
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn streams_thinking_with_signatures_and_redaction() {
    let server = sse_server(&[
        json!({"type": "start"}),
        json!({"type": "thinking_start", "contentIndex": 0}),
        json!({"type": "thinking_delta", "contentIndex": 0, "delta": "Let me "}),
        json!({"type": "thinking_delta", "contentIndex": 0, "delta": "think"}),
        json!({"type": "thinking_end", "contentIndex": 0, "content": "Let me think", "contentSignature": "sig-1"}),
        json!({"type": "thinking_start", "contentIndex": 1}),
        json!({"type": "thinking_end", "contentIndex": 1, "content": "", "contentSignature": "opaque", "redacted": true}),
        json!({"type": "text_start", "contentIndex": 2}),
        json!({"type": "text_delta", "contentIndex": 2, "delta": "Answer"}),
        json!({"type": "text_end", "contentIndex": 2, "content": "Answer", "contentSignature": "msg_1"}),
        json!({"type": "done", "reason": "stop", "usage": usage(), "providerThinkingLevel": "high"}),
    ])
    .await;

    let mut events = stream(&model(&base_url(&server)), &transcript(), options("key"));
    let mut thinking_end = None;
    while let Some(event) = events.next().await {
        if let AssistantMessageEvent::ThinkingEnd {
            content_index: 0,
            content,
            signature,
        } = &event
        {
            thinking_end = Some((content.clone(), signature.clone()));
        }
    }
    let message = events.result().await;

    assert_eq!(
        thinking_end,
        Some(("Let me think".to_owned(), Some("sig-1".to_owned())))
    );
    assert_eq!(
        serde_json::to_value(&message.content).unwrap(),
        json!([
            {"type": "thinking", "thinking": "Let me think", "thinkingSignature": "sig-1"},
            {"type": "thinking", "thinking": "", "thinkingSignature": "opaque", "redacted": true},
            {"type": "text", "text": "Answer", "textSignature": "msg_1"}
        ])
    );
    assert_eq!(message.provider_thinking_level.as_deref(), Some("high"));
}

#[tokio::test]
async fn streams_parallel_tool_calls_and_reports_partial_arguments() {
    let server = sse_server(&[
        json!({"type": "start"}),
        json!({"type": "toolcall_start", "contentIndex": 0, "id": "call_a", "toolName": "read"}),
        json!({"type": "toolcall_start", "contentIndex": 1, "id": "call_b", "toolName": "write"}),
        json!({"type": "toolcall_delta", "contentIndex": 0, "delta": "{\"path\": \"a"}),
        json!({"type": "toolcall_delta", "contentIndex": 1, "delta": "{\"path\": \"b\"}"}),
        json!({"type": "toolcall_delta", "contentIndex": 0, "delta": ".txt\"}"}),
        json!({"type": "toolcall_end", "contentIndex": 1,
               "toolCall": {"type": "toolCall", "id": "call_b", "name": "write", "arguments": {"path": "b"}}}),
        json!({"type": "toolcall_end", "contentIndex": 0,
               "toolCall": {"type": "toolCall", "id": "call_a", "name": "read", "arguments": {"path": "a.txt"}}}),
        json!({"type": "done", "reason": "toolUse", "usage": usage()}),
    ])
    .await;

    let mut events = stream(&model(&base_url(&server)), &transcript(), options("key"));
    let mut partial_paths = Vec::new();
    while let Some(event) = events.next().await {
        if let AssistantMessageEvent::ToolCallDelta { content_index: 0, .. } = event {
            let partial = events.partial().unwrap();
            let call = partial.content[0].as_tool_call().unwrap();
            partial_paths.push(call.arguments.get("path").cloned());
        }
    }
    let message = events.result().await;

    assert_eq!(partial_paths, [Some(json!("a")), Some(json!("a.txt"))]);
    let calls: Vec<_> = message
        .tool_calls()
        .map(|call| (call.id.clone(), call.name.clone()))
        .collect();
    assert_eq!(
        calls,
        [
            ("call_a".to_owned(), "read".to_owned()),
            ("call_b".to_owned(), "write".to_owned())
        ]
    );
    assert_eq!(message.stop_reason, StopReason::ToolUse);
}

#[tokio::test]
async fn skips_unknown_events_and_done_markers_and_records_rewrites() {
    let body = format!(
        "{}data: [DONE]\n\n",
        sse(&[
            json!({"type": "ping"}),
            json!({"type": "text_start", "contentIndex": 0}),
            json!({"type": "text_delta", "contentIndex": 0, "delta": "ok"}),
            json!({"type": "text_end", "contentIndex": 0, "content": "ok"}),
            json!({"type": "done", "reason": "length", "usage": usage(),
                   "rewrite": {"policyId": "compact", "policyVersion": 3, "changed": true, "tokenCountChange": -120,
                               "messageCountChange": -2, "systemPromptChanged": false}}),
        ])
    );
    let server = server_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")).await;

    let mut events = stream(&model(&base_url(&server)), &transcript(), options("key"));
    let mut kinds = Vec::new();
    while let Some(event) = events.next().await {
        kinds.push(event.kind());
    }
    let message = events.result().await;

    // A backend that sends no `start` still produces one before the first content event.
    assert_eq!(kinds, ["start", "text_start", "text_delta", "text_end", "done"]);
    assert_eq!(message.stop_reason, StopReason::Length);
    assert_eq!(message.text(), "ok");
    let diagnostics = message.diagnostics.unwrap();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].kind, "pi_messages_rewrite");
    assert_eq!(
        Value::Object(diagnostics[0].details.clone().unwrap()),
        json!({"policyId": "compact", "policyVersion": 3, "changed": true, "tokenCountChange": -120,
               "messageCountChange": -2, "systemPromptChanged": false})
    );
}

#[tokio::test]
async fn reports_malformed_events() {
    let server = server_with(ResponseTemplate::new(200).set_body_raw(
        "data: {\"type\": \"start\"}\n\ndata: {not json\n\n",
        "text/event-stream",
    ))
    .await;
    let message = stream(&model(&base_url(&server)), &transcript(), options("key"))
        .result()
        .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert!(message.error_message.unwrap().contains("key must be a string"));
}

#[tokio::test]
async fn sends_reasoning_temperature_cache_retention_and_tool_choice() {
    let server = sse_server(&[json!({"type": "done", "reason": "stop", "usage": usage()})]).await;
    let model = model(&base_url(&server));

    stream(
        &model,
        &transcript(),
        StreamOptions {
            temperature: Some(0.5),
            cache_retention: Some(CacheRetention::None),
            api: Some(pi_options(PiMessagesOptions {
                reasoning: Some(ThinkingLevel::High),
                tool_choice: Some(json!({"type": "function", "function": {"name": "read"}})),
                debug: None,
            })),
            ..options("key")
        },
    )
    .result()
    .await;
    assert_eq!(
        last_request(&server).await.body_json::<Value>().unwrap()["options"],
        json!({
            "temperature": 0.5,
            "reasoning": "high",
            "cacheRetention": "none",
            "toolChoice": {"type": "function", "function": {"name": "read"}}
        })
    );

    // The legacy environment opt-in maps to `long`; simple options forward reasoning and tool
    // choice as they are.
    stream_simple(
        &model,
        &transcript(),
        SimpleStreamOptions {
            base: StreamOptions {
                env: Some(
                    [("PI_CACHE_RETENTION".to_owned(), "long".to_owned())]
                        .into_iter()
                        .collect(),
                ),
                api: Some(pi_options(PiMessagesOptions {
                    tool_choice: Some(json!("required")),
                    ..PiMessagesOptions::default()
                })),
                ..options("key")
            },
            reasoning: Some(ThinkingLevel::Minimal),
            tool_choice: Some(ToolChoice::None),
            thinking_budgets: None,
            deferred: None,
        },
    )
    .result()
    .await;
    let request = last_request(&server).await;
    assert_eq!(request.url.query(), None);
    assert_eq!(
        request.body_json::<Value>().unwrap()["options"],
        json!({"reasoning": "minimal", "cacheRetention": "long", "toolChoice": "none"})
    );
}

#[tokio::test]
async fn sends_the_transcript_unchanged() {
    let server = sse_server(&[json!({"type": "done", "reason": "stop", "usage": usage()})]).await;
    let assistant = json!({
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "hm", "thinkingSignature": "sig"},
            {"type": "toolCall", "id": "toolu_1", "name": "read", "arguments": {"path": "a"}}
        ],
        "api": "anthropic-messages", "provider": "anthropic", "model": "claude-opus-5-5",
        "providerThinkingLevel": "high",
        "usage": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 3,
                  "cost": {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0}},
        "stopReason": "toolUse", "timestamp": 2
    });
    let messages: Vec<Message> = serde_json::from_value(json!([
        {"role": "user", "content": [{"type": "text", "text": "Look"},
                                     {"type": "image", "data": "AA==", "mimeType": "image/png"}], "timestamp": 1},
        assistant,
        {"role": "toolResult", "toolCallId": "toolu_1", "toolName": "read",
         "content": [{"type": "text", "text": "x"}], "isError": false, "timestamp": 3}
    ]))
    .unwrap();
    let context = Context::new(messages).with_system_prompt("Be brief.");
    let transcript = normalize_context(&context);

    stream(&model(&base_url(&server)), &transcript, options("key"))
        .result()
        .await;

    let body: Value = last_request(&server).await.body_json().unwrap();
    // Foreign assistant messages keep their thinking signatures and provider thinking level.
    assert_eq!(body["context"], serde_json::to_value(&transcript).unwrap());
    assert_eq!(body["context"]["messages"][0]["role"], "system");
    assert_eq!(body["context"]["messages"][2]["providerThinkingLevel"], "high");
    assert_eq!(body["context"]["messages"][2]["content"][0]["thinkingSignature"], "sig");
}

#[tokio::test]
async fn aborts_while_waiting_for_an_event() {
    let base_url = stalling_server(vec![(
        sse(&[
            json!({"type": "start"}),
            json!({"type": "text_start", "contentIndex": 0}),
            json!({"type": "text_delta", "contentIndex": 0, "delta": "hel"}),
        ]),
        Duration::ZERO,
    )]);
    let cancel = CancellationToken::new();
    let mut events = stream(
        &model(&base_url),
        &transcript(),
        StreamOptions {
            cancel: Some(cancel.clone()),
            ..options("key")
        },
    );
    while let Some(event) = events.next().await {
        if matches!(event, AssistantMessageEvent::TextDelta { .. }) {
            break;
        }
    }
    cancel.cancel();
    let message = events.result().await;
    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert_eq!(message.error_message.as_deref(), Some("Request was aborted"));
    assert!(message.content.is_empty());
}

#[tokio::test]
async fn cancelled_before_the_request_is_aborted() {
    let server = sse_server(&[json!({"type": "done", "reason": "stop", "usage": usage()})]).await;
    let cancel = CancellationToken::new();
    cancel.cancel();
    let message = stream(
        &model(&base_url(&server)),
        &transcript(),
        StreamOptions {
            cancel: Some(cancel),
            ..options("key")
        },
    )
    .result()
    .await;
    assert_eq!(message.stop_reason, StopReason::Aborted);
}

#[test]
fn serializes_events_for_backends() {
    let event = PiMessagesEvent::ToolCallStart {
        content_index: 1,
        id: "call_1".into(),
        tool_name: "read".into(),
    };
    assert_eq!(
        serde_json::to_value(&event).unwrap(),
        json!({"type": "toolcall_start", "contentIndex": 1, "id": "call_1", "toolName": "read"})
    );
}
