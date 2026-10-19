//! End-to-end tests of the `bedrock-converse-stream` API: the official AWS SDK client talks to a
//! local wiremock server that replays recorded `application/vnd.amazon.eventstream` bodies.
#![cfg(feature = "bedrock")]
// Test helpers outside `#[test]` functions unwrap freely.
#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_smithy_eventstream::frame::write_message_to;
use aws_smithy_types::event_stream::{Header, HeaderValue, Message as Frame};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use delta_ai::api::bedrock_converse_stream::{BedrockConverseStreamApi, stream, stream_simple};
use delta_ai::options::{ApiOptions, BedrockOptions, ProviderResponse, ThinkingBudgets};
use delta_ai::types::ProviderEnv;
use delta_ai::{
    AssistantBlock, AssistantMessage, Context, ImageContent, Message, Model, ProviderStreams, SimpleStreamOptions,
    StopReason, StreamOptions, ThinkingContent, ThinkingLevel, Tool, ToolCall, ToolResultMessage, UserBlock,
    UserContent, UserMessage, normalize_context,
};
use futures::StreamExt;
use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn string_header(name: &'static str, value: &str) -> Header {
    Header::new(name, HeaderValue::String(value.to_owned().into()))
}

fn encode(frame: &Frame) -> Vec<u8> {
    let mut buffer = Vec::new();
    write_message_to(frame, &mut buffer).unwrap();
    buffer
}

/// One `event` frame of the ConverseStream output.
#[allow(clippy::needless_pass_by_value)] // Called with `json!` literals.
fn event(kind: &str, payload: Value) -> Vec<u8> {
    encode(
        &Frame::new(serde_json::to_vec(&payload).unwrap())
            .add_header(string_header(":message-type", "event"))
            .add_header(string_header(":event-type", kind))
            .add_header(string_header(":content-type", "application/json")),
    )
}

/// One modeled `exception` frame.
fn exception(kind: &str, message: &str) -> Vec<u8> {
    encode(
        &Frame::new(serde_json::to_vec(&json!({"message": message})).unwrap())
            .add_header(string_header(":message-type", "exception"))
            .add_header(string_header(":exception-type", kind))
            .add_header(string_header(":content-type", "application/json")),
    )
}

fn message_start() -> Vec<u8> {
    event("messageStart", json!({"role": "assistant"}))
}

fn text_delta(index: i64, text: &str) -> Vec<u8> {
    event(
        "contentBlockDelta",
        json!({"contentBlockIndex": index, "delta": {"text": text}}),
    )
}

fn block_stop(index: i64) -> Vec<u8> {
    event("contentBlockStop", json!({"contentBlockIndex": index}))
}

fn message_stop(reason: &str) -> Vec<u8> {
    event("messageStop", json!({"stopReason": reason}))
}

fn body(frames: &[Vec<u8>]) -> Vec<u8> {
    frames.concat()
}

fn eventstream(frames: &[Vec<u8>]) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "application/vnd.amazon.eventstream")
        .insert_header("x-amzn-requestid", "req-123")
        .insert_header("x-bifrost-provider", "bedrock")
        .set_body_bytes(body(frames))
}

fn model(base_url: &str, id: &str, name: &str) -> Model {
    serde_json::from_value(json!({
        "id": id, "name": name, "api": "bedrock-converse-stream", "provider": "amazon-bedrock",
        "baseUrl": base_url, "reasoning": true, "input": ["text", "image"],
        "cost": {"input": 3, "output": 15, "cacheRead": 0.3, "cacheWrite": 3.75},
        "contextWindow": 200_000, "maxTokens": 64000, "compat": {"supportsStrictMode": true}
    }))
    .unwrap()
}

fn claude(server: &MockServer) -> Model {
    model(
        &server.uri(),
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "Claude Sonnet 4.5 (US)",
    )
}

fn nova(server: &MockServer) -> Model {
    let mut model = model(&server.uri(), "amazon.nova-lite-v1:0", "Nova Lite");
    model.reasoning = false;
    model.compat = None;
    model
}

fn env() -> ProviderEnv {
    [
        ("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE"),
        ("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"),
        ("AWS_REGION", "us-east-1"),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect()
}

fn options() -> StreamOptions {
    StreamOptions {
        env: Some(env()),
        max_retries: Some(0),
        cache_retention: Some(delta_ai::CacheRetention::None),
        ..StreamOptions::default()
    }
}

fn hello() -> delta_ai::TranscriptContext {
    normalize_context(&Context::new(vec![Message::user("hello")]))
}

async fn mount(server: &MockServer, response: ResponseTemplate) {
    Mock::given(method("POST"))
        .and(path_regex("^/model/.+/converse-stream$"))
        .respond_with(response)
        .mount(server)
        .await;
}

async fn requests(server: &MockServer) -> Vec<Request> {
    server.received_requests().await.unwrap()
}

fn request_json(request: &Request) -> Value {
    serde_json::from_slice(&request.body).unwrap()
}

fn header<'a>(request: &'a Request, name: &str) -> Option<&'a str> {
    request.headers.get(name).and_then(|value| value.to_str().ok())
}

#[tokio::test]
async fn streams_text_usage_and_response_headers() {
    let server = MockServer::start().await;
    mount(
        &server,
        eventstream(&[
            message_start(),
            text_delta(0, "Hel"),
            text_delta(0, "lo"),
            block_stop(0),
            event(
                "messageStop",
                json!({"stopReason": "end_turn", "additionalModelResponseFields": {"source": "test"}}),
            ),
            event(
                "metadata",
                json!({"usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12, "cacheReadInputTokens": 4},
                       "metrics": {"latencyMs": 5}}),
            ),
        ]),
    )
    .await;
    let model = nova(&server);
    let responses: Arc<Mutex<Vec<ProviderResponse>>> = Arc::default();
    let events: Arc<Mutex<Vec<Value>>> = Arc::default();
    let (responses_sink, events_sink) = (responses.clone(), events.clone());
    let options = StreamOptions {
        on_response: Some(Arc::new(move |response: &ProviderResponse, _: &Model| {
            responses_sink.lock().unwrap().push(response.clone());
            Box::pin(async { Ok(()) })
        })),
        on_provider_stream_event: Some(Arc::new(move |event: &Value, _: &Model| {
            events_sink.lock().unwrap().push(event.clone());
            Box::pin(async { Ok(()) })
        })),
        ..options()
    };
    let mut events_stream = BedrockConverseStreamApi.stream(&model, &hello(), options);
    let mut kinds = Vec::new();
    while let Some(event) = events_stream.next().await {
        kinds.push(event.kind());
    }
    assert_eq!(
        kinds,
        ["start", "text_start", "text_delta", "text_delta", "text_end", "done"]
    );
    let message = events_stream.partial().unwrap().clone();
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(message.text(), "Hello");
    assert_eq!(message.usage.input, 10);
    assert_eq!(message.usage.output, 2);
    assert_eq!(message.usage.cache_read, 4);
    assert_eq!(message.usage.total_tokens, 12);
    assert!(message.usage.cost.total > 0.0);

    let responses = responses.lock().unwrap().clone();
    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0].status, 200);
    assert_eq!(responses[0].headers["x-amzn-requestid"], "req-123");
    assert_eq!(responses[0].headers["x-bifrost-provider"], "bedrock");

    let events = events.lock().unwrap().clone();
    assert_eq!(events[0], json!({"messageStart": {"role": "assistant"}}));
    assert_eq!(
        events[1],
        json!({"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hel"}}})
    );
    assert_eq!(
        events[4],
        json!({"messageStop": {"stopReason": "end_turn", "additionalModelResponseFields": {"source": "test"}}})
    );
    assert_eq!(events.len(), 6);

    let requests = requests(&server).await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url.path(), "/model/amazon.nova-lite-v1%3A0/converse-stream");
    assert!(
        header(&requests[0], "authorization")
            .unwrap()
            .starts_with("AWS4-HMAC-SHA256")
    );
    assert_eq!(
        request_json(&requests[0]),
        json!({"messages": [{"role": "user", "content": [{"text": "hello"}]}], "inferenceConfig": {}})
    );
}

#[tokio::test]
async fn thinking_and_parallel_tool_calls() {
    let server = MockServer::start().await;
    mount(
        &server,
        eventstream(&[
            message_start(),
            event(
                "contentBlockDelta",
                json!({"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "Plan."}}}),
            ),
            event(
                "contentBlockDelta",
                json!({"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig-1"}}}),
            ),
            block_stop(0),
            event(
                "contentBlockStart",
                json!({"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "tool-a", "name": "read"}}}),
            ),
            event(
                "contentBlockStart",
                json!({"contentBlockIndex": 2, "start": {"toolUse": {"toolUseId": "tool-b", "name": "read"}}}),
            ),
            event(
                "contentBlockDelta",
                json!({"contentBlockIndex": 1, "delta": {"toolUse": {"input": "{\"path\":"}}}),
            ),
            event(
                "contentBlockDelta",
                json!({"contentBlockIndex": 2, "delta": {"toolUse": {"input": "{\"path\":\"b\"}"}}}),
            ),
            event(
                "contentBlockDelta",
                json!({"contentBlockIndex": 1, "delta": {"toolUse": {"input": "\"a\"}"}}}),
            ),
            block_stop(1),
            block_stop(2),
            message_stop("tool_use"),
        ]),
    )
    .await;
    let message = stream(&claude(&server), &hello(), options()).result().await;
    assert_eq!(message.stop_reason, StopReason::ToolUse, "{:?}", message.error_message);
    assert_eq!(
        message.content[0],
        AssistantBlock::Thinking(ThinkingContent {
            thinking: "Plan.".into(),
            thinking_signature: Some("sig-1".into()),
            redacted: None,
        })
    );
    let calls: Vec<&ToolCall> = message.tool_calls().collect();
    assert_eq!(calls.len(), 2);
    assert_eq!((calls[0].id.as_str(), calls[1].id.as_str()), ("tool-a", "tool-b"));
    assert_eq!(Value::Object(calls[0].arguments.clone()), json!({"path": "a"}));
    assert_eq!(Value::Object(calls[1].arguments.clone()), json!({"path": "b"}));
}

#[tokio::test]
async fn redacted_reasoning_round_trips_as_base64() {
    let server = MockServer::start().await;
    let payload = b"\x00encrypted\xff";
    mount(
        &server,
        eventstream(&[
            message_start(),
            event(
                "contentBlockDelta",
                json!({"contentBlockIndex": 0, "delta": {"reasoningContent": {"redactedContent": STANDARD.encode(payload)}}}),
            ),
            block_stop(0),
            text_delta(1, "done"),
            block_stop(1),
            message_stop("end_turn"),
        ]),
    )
    .await;
    let model = model(&server.uri(), "global.openai.gpt-5.6-terra", "GPT-5.6 Terra (Global)");
    let message = stream(&model, &hello(), options()).result().await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    let thinking = message.content[0].as_thinking().unwrap();
    assert!(thinking.is_redacted());
    assert_eq!(thinking.thinking, "[Reasoning redacted]");
    assert_eq!(
        thinking.thinking_signature.as_deref(),
        Some(STANDARD.encode(payload).as_str())
    );

    // Replay: the payload goes back as redactedContent bytes.
    let replay = normalize_context(&Context::new(vec![
        Message::user("hello"),
        Message::Assistant(message),
        Message::user("continue"),
    ]));
    let _ = stream(&model, &replay, options()).result().await;
    let requests = requests(&server).await;
    let sent = request_json(&requests[1]);
    assert_eq!(
        sent["messages"][1]["content"],
        json!([{"reasoningContent": {"redactedContent": STANDARD.encode(payload)}}, {"text": "done"}])
    );
}

#[tokio::test]
async fn request_payload_shape_on_the_wire() {
    let server = MockServer::start().await;
    mount(&server, eventstream(&[message_start(), message_stop("end_turn")])).await;
    let model = claude(&server);
    let mut assistant = AssistantMessage::empty("bedrock-converse-stream", "amazon-bedrock", &model.id);
    assistant.stop_reason = StopReason::ToolUse;
    assistant.content = vec![AssistantBlock::ToolCall(ToolCall {
        id: "tool-1".into(),
        name: "read".into(),
        arguments: [("path".to_owned(), json!("a.png"))].into_iter().collect(),
        ..ToolCall::default()
    })];
    let context = Context::new(vec![
        Message::User(UserMessage {
            content: UserContent::Blocks(vec![
                UserBlock::text("look"),
                UserBlock::Image(ImageContent {
                    data: "AAEC".into(),
                    mime_type: "image/webp".into(),
                }),
            ]),
            timestamp: 0,
        }),
        Message::Assistant(assistant),
        Message::ToolResult(ToolResultMessage {
            tool_call_id: "tool-1".into(),
            tool_name: "read".into(),
            content: vec![UserBlock::text("bytes")],
            is_error: false,
            ..ToolResultMessage::default()
        }),
    ])
    .with_system_prompt("Be brief.")
    .with_tools(vec![Tool::new(
        "read",
        "Read a file",
        json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
    )]);
    let options = StreamOptions {
        cache_retention: Some(delta_ai::CacheRetention::Long),
        temperature: Some(0.5),
        api: Some(ApiOptions::BedrockConverseStream(BedrockOptions {
            reasoning: Some(ThinkingLevel::Medium),
            thinking_budgets: Some(ThinkingBudgets {
                medium: Some(4096),
                ..ThinkingBudgets::default()
            }),
            request_metadata: Some([("project".to_owned(), "delta".to_owned())].into_iter().collect()),
            ..BedrockOptions::default()
        })),
        ..options()
    };
    let message = stream(&model, &normalize_context(&context), options).result().await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    let sent = request_json(&requests(&server).await[0]);
    assert_eq!(
        sent,
        json!({
            "messages": [
                {"role": "user", "content": [
                    {"text": "look"},
                    {"image": {"format": "webp", "source": {"bytes": "AAEC"}}}
                ]},
                {"role": "assistant", "content": [
                    {"toolUse": {"toolUseId": "tool-1", "name": "read", "input": {"path": "a.png"}}}
                ]},
                {"role": "user", "content": [
                    {"toolResult": {"toolUseId": "tool-1", "content": [{"text": "bytes"}], "status": "success"}},
                    {"cachePoint": {"type": "default", "ttl": "1h"}}
                ]}
            ],
            "system": [{"text": "Be brief."}, {"cachePoint": {"type": "default", "ttl": "1h"}}],
            "inferenceConfig": {"maxTokens": 64000, "temperature": 0.5},
            "toolConfig": {"tools": [{"toolSpec": {
                "name": "read",
                "description": "Read a file",
                "inputSchema": {"json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
            }}]},
            "additionalModelRequestFields": {
                "thinking": {"type": "enabled", "budget_tokens": 4096, "display": "summarized"},
                "anthropic_beta": ["interleaved-thinking-2025-05-14"]
            },
            "requestMetadata": {"project": "delta"}
        })
    );
}

#[tokio::test]
async fn on_payload_replaces_the_request() {
    let server = MockServer::start().await;
    mount(&server, eventstream(&[message_start(), message_stop("end_turn")])).await;
    let options = StreamOptions {
        on_payload: Some(Arc::new(|payload: &Value, _: &Model| {
            let mut payload = payload.clone();
            payload["inferenceConfig"]["maxTokens"] = json!(7);
            payload["inferenceConfig"]["stopSequences"] = json!(["END"]);
            Box::pin(async move { Ok(Some(payload)) })
        })),
        ..options()
    };
    let message = stream(&nova(&server), &hello(), options).result().await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    let sent = request_json(&requests(&server).await[0]);
    assert_eq!(
        sent["inferenceConfig"],
        json!({"maxTokens": 7, "stopSequences": ["END"]})
    );
}

#[tokio::test]
async fn custom_headers_are_signed_and_reserved_headers_ignored() {
    let server = MockServer::start().await;
    mount(&server, eventstream(&[message_start(), message_stop("end_turn")])).await;
    let headers = [
        ("x-custom", "v"),
        ("Authorization", "evil"),
        ("X-Amz-Date", "evil"),
        ("host", "evil"),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_owned(), Some(value.to_owned())))
    .collect();
    let simple = SimpleStreamOptions {
        base: StreamOptions {
            headers: Some(headers),
            ..options()
        },
        ..SimpleStreamOptions::default()
    };
    let message = stream_simple(&nova(&server), &hello(), simple).result().await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    let request = &requests(&server).await[0];
    assert_eq!(header(request, "x-custom"), Some("v"));
    let authorization = header(request, "authorization").unwrap();
    assert!(authorization.starts_with("AWS4-HMAC-SHA256"), "{authorization}");
    let signed = authorization.split("SignedHeaders=").nth(1).unwrap();
    assert!(
        signed.split([';', ',']).any(|name| name == "x-custom"),
        "{authorization}"
    );
    assert_ne!(header(request, "x-amz-date"), Some("evil"));
    assert_ne!(header(request, "host"), Some("evil"));
}

#[tokio::test]
async fn bearer_token_replaces_sigv4() {
    let server = MockServer::start().await;
    mount(&server, eventstream(&[message_start(), message_stop("end_turn")])).await;
    let options = StreamOptions {
        api_key: Some("bedrock-api-key".into()),
        ..options()
    };
    let message = stream(&nova(&server), &hello(), options).result().await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    let request = &requests(&server).await[0];
    assert_eq!(header(request, "authorization"), Some("Bearer bedrock-api-key"));
}

#[tokio::test]
async fn service_errors_carry_prefix_and_diagnostics() {
    let server = MockServer::start().await;
    mount(
        &server,
        ResponseTemplate::new(400)
            .insert_header(
                "x-amzn-errortype",
                "ValidationException:http://internal.amazon.com/coral/",
            )
            .insert_header("x-amzn-requestid", "11111111-2222-3333-4444-555555555555")
            .insert_header("content-type", "application/json")
            .set_body_json(json!({"message": "The provided model identifier is invalid."})),
    )
    .await;
    let message = stream(&nova(&server), &hello(), options()).result().await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Validation error: The provided model identifier is invalid.")
    );
    let diagnostics = message.diagnostics.unwrap();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].kind, "bedrock_response_failure");
    assert_eq!(diagnostics[0].error, None);
    assert_eq!(
        Value::Object(diagnostics[0].details.clone().unwrap()),
        json!({"status": 400, "errorCode": "ValidationException", "requestId": "11111111-2222-3333-4444-555555555555"})
    );
}

#[tokio::test]
async fn gateway_errors_surface_status_and_body() {
    let server = MockServer::start().await;
    mount(
        &server,
        ResponseTemplate::new(403)
            .insert_header("x-amzn-requestid", "req-403")
            .set_body_string("<html>Forbidden by gateway</html>"),
    )
    .await;
    let message = stream(&nova(&server), &hello(), options()).result().await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Unknown: 403: <html>Forbidden by gateway</html>")
    );
    assert_eq!(
        Value::Object(message.diagnostics.unwrap()[0].details.clone().unwrap()),
        json!({"status": 403, "requestId": "req-403"})
    );
}

#[tokio::test]
async fn server_errors_are_retried() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(503)
                .insert_header("x-amzn-errortype", "ServiceUnavailableException")
                .set_body_json(json!({"message": "try again"})),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;
    mount(
        &server,
        eventstream(&[message_start(), text_delta(0, "ok"), message_stop("end_turn")]),
    )
    .await;
    let options = StreamOptions {
        max_retries: Some(2),
        ..options()
    };
    let message = stream(&nova(&server), &hello(), options).result().await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    assert_eq!(message.text(), "ok");
    assert_eq!(requests(&server).await.len(), 2);
}

#[tokio::test]
async fn in_stream_exceptions_are_reported_after_observation() {
    let server = MockServer::start().await;
    mount(
        &server,
        eventstream(&[
            message_start(),
            text_delta(0, "partial"),
            exception("throttlingException", "Too many requests, please wait."),
        ]),
    )
    .await;
    let events: Arc<Mutex<Vec<Value>>> = Arc::default();
    let sink = events.clone();
    let options = StreamOptions {
        on_provider_stream_event: Some(Arc::new(move |event: &Value, _: &Model| {
            sink.lock().unwrap().push(event.clone());
            Box::pin(async { Ok(()) })
        })),
        ..options()
    };
    let message = stream(&nova(&server), &hello(), options).result().await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Throttling error: Too many requests, please wait.")
    );
    assert_eq!(message.text(), "partial");
    assert_eq!(
        Value::Object(message.diagnostics.unwrap()[0].details.clone().unwrap()),
        json!({"errorCode": "ThrottlingException", "requestId": "req-123"})
    );
    assert_eq!(
        events.lock().unwrap().last().unwrap(),
        &json!({"throttlingException": {"message": "Too many requests, please wait."}})
    );
}

#[tokio::test]
async fn provider_error_stops_keep_the_raw_reason() {
    let server = MockServer::start().await;
    mount(
        &server,
        eventstream(&[message_start(), message_stop("guardrail_intervened")]),
    )
    .await;
    let message = stream(&nova(&server), &hello(), options()).result().await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("guardrail_intervened"));
    assert_eq!(
        message.error_message.as_deref(),
        Some("Provider stopped with: guardrail_intervened")
    );
    assert_eq!(
        Value::Object(message.diagnostics.unwrap()[0].details.clone().unwrap()),
        json!({"requestId": "req-123"})
    );
}

#[tokio::test]
async fn streams_without_a_stop_reason_fail() {
    let server = MockServer::start().await;
    mount(&server, eventstream(&[message_start(), text_delta(0, "cut")])).await;
    let message = stream(&nova(&server), &hello(), options()).result().await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Bedrock stream ended without a stop reason")
    );
    assert_eq!(message.text(), "cut");

    // An empty event stream still reports the response headers first.
    let server = MockServer::start().await;
    mount(&server, eventstream(&[])).await;
    let seen = Arc::new(Mutex::new(0));
    let sink = seen.clone();
    let options = StreamOptions {
        on_response: Some(Arc::new(move |_: &ProviderResponse, _: &Model| {
            *sink.lock().unwrap() += 1;
            Box::pin(async { Ok(()) })
        })),
        ..options()
    };
    let message = stream(&nova(&server), &hello(), options).result().await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(*seen.lock().unwrap(), 1);
}

#[tokio::test]
async fn cancellation_aborts_a_pending_request() {
    let server = MockServer::start().await;
    mount(
        &server,
        eventstream(&[message_start(), message_stop("end_turn")]).set_delay(Duration::from_secs(30)),
    )
    .await;
    let cancel = CancellationToken::new();
    let options = StreamOptions {
        cancel: Some(cancel.clone()),
        ..options()
    };
    let pending = stream(&nova(&server), &hello(), options);
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        cancel.cancel();
    });
    let started = std::time::Instant::now();
    let message = tokio::time::timeout(Duration::from_secs(10), pending.result())
        .await
        .unwrap();
    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert_eq!(message.error_message.as_deref(), Some("Request was aborted"));
    assert_eq!(message.diagnostics, None);
    assert!(started.elapsed() < Duration::from_secs(10));
}

#[tokio::test]
async fn foreign_history_and_orphaned_tool_calls_on_the_wire() {
    let server = MockServer::start().await;
    mount(&server, eventstream(&[message_start(), message_stop("end_turn")])).await;
    let mut foreign = AssistantMessage::empty("openai-responses", "openai", "gpt-5");
    foreign.stop_reason = StopReason::ToolUse;
    foreign.content = vec![
        AssistantBlock::Thinking(ThinkingContent {
            thinking: "hidden".into(),
            thinking_signature: Some("enc".into()),
            redacted: None,
        }),
        AssistantBlock::ToolCall(ToolCall {
            id: "call_1|fc_1".into(),
            name: "read".into(),
            ..ToolCall::default()
        }),
    ];
    let context = normalize_context(&Context::new(vec![
        Message::user("hi"),
        Message::Assistant(foreign),
        Message::user("next"),
    ]));
    let message = stream(&claude(&server), &context, options()).result().await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    let sent = request_json(&requests(&server).await[0]);
    assert_eq!(
        sent["messages"][1],
        json!({"role": "assistant", "content": [{"toolUse": {"toolUseId": "call_1_fc_1", "name": "read", "input": {}}}]})
    );
    assert_eq!(
        sent["messages"][2],
        json!({"role": "user", "content": [{"toolResult": {"toolUseId": "call_1_fc_1",
            "content": [{"text": "No result provided"}], "status": "error"}}]})
    );
}

#[tokio::test]
async fn timeouts_fail_the_request() {
    let server = MockServer::start().await;
    mount(
        &server,
        eventstream(&[message_start(), message_stop("end_turn")]).set_delay(Duration::from_secs(30)),
    )
    .await;
    let options = StreamOptions {
        timeout: Some(Duration::from_millis(300)),
        ..options()
    };
    let started = std::time::Instant::now();
    let message = stream(&nova(&server), &hello(), options).result().await;
    assert_eq!(message.stop_reason, StopReason::Error);
    let error = message.error_message.unwrap();
    assert!(error.contains("operation timeout"), "{error}");
    assert!(started.elapsed() < Duration::from_secs(10));
}
