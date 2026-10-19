//! End-to-end tests of the `mistral-conversations` API against local servers (ports of
//! upstream `test/mistral-*.test.ts` and pi-py's `tests/test_mistral_conversations.py`).

#![allow(clippy::unwrap_used)] // Test helpers outside `#[test]` functions.

use delta_ai::api::mistral_conversations::{MistralConversationsApi, stream, stream_simple};
use delta_ai::options::{MistralOptions, OnPayload, OnProviderStreamEvent, OnResponse};
use delta_ai::types::{ConstrainedSampling, StrictMode};
use delta_ai::{
    ApiOptions, AssistantBlock, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream, CacheRetention,
    Context, ImageContent, Message, Model, ProviderStreams, SimpleStreamOptions, StopReason, StreamOptions,
    SystemMessage, TextContent, ThinkingContent, ThinkingLevel, Tool, ToolCall, ToolChoice, ToolResultMessage,
    TranscriptContext, UserBlock, UserContent, UserMessage, normalize_context,
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

fn model_with(base_url: &str, id: &str, reasoning: bool, level_map: Option<Value>, images: bool) -> Model {
    let mut value = json!({
        "id": id, "name": id, "api": "mistral-conversations", "provider": "mistral",
        "baseUrl": base_url, "reasoning": reasoning,
        "input": if images { json!(["text", "image"]) } else { json!(["text"]) },
        "cost": {"input": 2.0, "output": 6.0, "cacheRead": 0.2, "cacheWrite": 0.0},
        "contextWindow": 128_000, "maxTokens": 16_384,
    });
    if let Some(map) = level_map {
        value["thinkingLevelMap"] = map;
    }
    serde_json::from_value(value).unwrap()
}

fn model(base_url: &str) -> Model {
    model_with(base_url, "mistral-large-latest", false, None, true)
}

fn sse(events: &[Value], separator: &str) -> String {
    let body = events
        .iter()
        .map(|event| format!("data: {event}"))
        .collect::<Vec<_>>()
        .join(separator);
    format!("{body}{separator}data: [DONE]{separator}")
}

fn terminal_event(finish_reason: &str) -> Value {
    json!({
        "id": "mistral-response-id",
        "model": "mistral-large-latest",
        "choices": [{"index": 0, "finish_reason": finish_reason, "delta": {}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })
}

fn chunk(delta: &Value, finish_reason: Option<&str>) -> Value {
    json!({
        "id": "response-1",
        "model": "mistral-large-latest",
        "choices": [{"index": 0, "finish_reason": finish_reason, "delta": delta}],
    })
}

async fn server_with(template: ResponseTemplate) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(template)
        .mount(&server)
        .await;
    server
}

async fn sse_server(events: &[Value]) -> MockServer {
    server_with(ResponseTemplate::new(200).set_body_raw(sse(events, "\r\n\r\n"), "text/event-stream")).await
}

async fn last_body(server: &MockServer) -> Value {
    let requests = server.received_requests().await.unwrap();
    requests.last().unwrap().body_json().unwrap()
}

fn hello() -> TranscriptContext {
    normalize_context(&Context::new(vec![Message::User(UserMessage {
        content: "hello".into(),
        timestamp: 1,
    })]))
}

/// Options with an API key. Each test gets its own HTTP client: pooled connections of the
/// shared default client belong to the Tokio runtime of the test that opened them.
fn options(api_key: &str) -> StreamOptions {
    StreamOptions {
        api_key: Some(api_key.into()),
        http_client: Some(reqwest::Client::new()),
        ..StreamOptions::default()
    }
}

fn mistral_options(options: MistralOptions) -> ApiOptions {
    ApiOptions::MistralConversations(options)
}

fn capture_payload() -> (OnPayload, Arc<Mutex<Option<Value>>>) {
    let captured = Arc::new(Mutex::new(None));
    let sink = captured.clone();
    let hook: OnPayload = Arc::new(move |payload: &Value, _model: &Model| {
        *sink.lock().unwrap() = Some(payload.clone());
        Box::pin(async { Ok(None) })
    });
    (hook, captured)
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

fn assistant(provider: &str, api: &str, model: &str, content: Vec<AssistantBlock>) -> AssistantMessage {
    AssistantMessage {
        content,
        stop_reason: StopReason::ToolUse,
        ..AssistantMessage::empty(api, provider, model)
    }
}

/// A one-connection HTTP/1.1 server that answers with a chunked SSE body, writing each chunk
/// after its delay. With `hang`, the body never ends.
fn raw_sse_server(chunks: Vec<(Vec<u8>, Duration)>, hang: bool) -> String {
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
        for (bytes, delay) in chunks {
            std::thread::sleep(delay);
            let mut frame = format!("{:x}\r\n", bytes.len()).into_bytes();
            frame.extend_from_slice(&bytes);
            frame.extend_from_slice(b"\r\n");
            if socket.write_all(&frame).is_err() {
                return;
            }
        }
        if hang {
            std::thread::sleep(Duration::from_secs(30));
            return;
        }
        let _ = socket.write_all(b"0\r\n\r\n");
    });
    format!("http://{address}")
}

// ---------------------------------------------------------------------------------------------
// Request payload and transport
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn serializes_sdk_style_payloads_to_the_wire_format() {
    let server = server_with(
        ResponseTemplate::new(200)
            .insert_header("x-request-id", "request-1")
            .set_body_raw(sse(&[terminal_event("stop")], "\r\n\r\n"), "text/event-stream"),
    )
    .await;
    let model = model(&server.uri());
    let context = normalize_context(
        &Context::new(vec![Message::User(UserMessage {
            content: UserContent::Blocks(vec![
                UserBlock::text("describe"),
                UserBlock::Image(ImageContent {
                    data: "aGVsbG8=".into(),
                    mime_type: "image/png".into(),
                }),
            ]),
            timestamp: 1,
        })])
        .with_system_prompt("Be precise")
        .with_tools(vec![Tool::new(
            "lookup",
            "Look something up",
            json!({"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
        )]),
    );
    let callback_payload = Arc::new(Mutex::new(None::<Value>));
    let sink = callback_payload.clone();
    let on_payload: OnPayload = Arc::new(move |payload: &Value, _model: &Model| {
        *sink.lock().unwrap() = Some(payload.clone());
        let mut replaced = payload.clone();
        let object = replaced.as_object_mut().unwrap();
        object.insert("topP".into(), json!(0.9));
        object.insert("randomSeed".into(), json!(42));
        object.insert(
            "responseFormat".into(),
            json!({"type": "json_schema", "jsonSchema": {"name": "result",
                "schemaDefinition": {"type": "object", "properties": {"maxTokens": {"type": "number"}}}}}),
        );
        object.insert("presencePenalty".into(), json!(0.1));
        object.insert("frequencyPenalty".into(), json!(0.2));
        object.insert("parallelToolCalls".into(), json!(true));
        object.insert("safePrompt".into(), json!(true));
        Box::pin(async move { Ok(Some(replaced)) })
    });
    let responses = Arc::new(Mutex::new(Vec::new()));
    let response_sink = responses.clone();
    let on_response: OnResponse = Arc::new(move |response: &delta_ai::options::ProviderResponse, _model: &Model| {
        response_sink.lock().unwrap().push(response.clone());
        Box::pin(async { Ok(()) })
    });

    let message = stream(
        &model,
        &context,
        StreamOptions {
            headers: Some(
                [("x-custom".to_owned(), Some("value".to_owned()))]
                    .into_iter()
                    .collect(),
            ),
            max_tokens: Some(123),
            session_id: Some("session-1".into()),
            on_payload: Some(on_payload),
            on_response: Some(on_response),
            api: Some(mistral_options(MistralOptions {
                tool_choice: Some(json!({"type": "function", "function": {"name": "lookup"}})),
                prompt_mode: Some("reasoning".into()),
                reasoning_effort: Some("high".into()),
            })),
            ..options("secret")
        },
    )
    .result()
    .await;

    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    let requests = server.received_requests().await.unwrap();
    let request = &requests[0];
    assert_eq!(request.url.path(), "/v1/chat/completions");
    let header = |name: &str| request.headers.get(name).unwrap().to_str().unwrap().to_owned();
    assert_eq!(header("authorization"), "Bearer secret");
    assert_eq!(header("accept"), "text/event-stream");
    assert_eq!(header("content-type"), "application/json");
    assert_eq!(header("x-affinity"), "session-1");
    assert_eq!(header("x-custom"), "value");
    assert!(header("user-agent").starts_with("delta/"));

    let payload = callback_payload.lock().unwrap().clone().unwrap();
    assert_eq!(payload["maxTokens"], json!(123));
    assert_eq!(payload["promptMode"], json!("reasoning"));
    assert_eq!(payload["promptCacheKey"], json!("session-1"));
    {
        let responses = responses.lock().unwrap();
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].status, 200);
        assert_eq!(responses[0].headers.get("x-request-id").unwrap(), "request-1");
        assert_eq!(responses[0].headers.get("content-type").unwrap(), "text/event-stream");
    }

    let wire: Value = request.body_json().unwrap();
    assert_eq!(wire["model"], json!("mistral-large-latest"));
    assert_eq!(wire["stream"], json!(true));
    assert_eq!(wire["max_tokens"], json!(123));
    assert_eq!(wire["prompt_mode"], json!("reasoning"));
    assert_eq!(wire["reasoning_effort"], json!("high"));
    assert_eq!(
        wire["tool_choice"],
        json!({"type": "function", "function": {"name": "lookup"}})
    );
    assert_eq!(wire["prompt_cache_key"], json!("session-1"));
    assert_eq!(wire["top_p"], json!(0.9));
    assert_eq!(wire["random_seed"], json!(42));
    assert_eq!(wire["presence_penalty"], json!(0.1));
    assert_eq!(wire["frequency_penalty"], json!(0.2));
    assert_eq!(wire["parallel_tool_calls"], json!(true));
    assert_eq!(wire["safe_prompt"], json!(true));
    assert_eq!(
        wire["response_format"],
        json!({"type": "json_schema", "json_schema": {"name": "result",
            "schema": {"type": "object", "properties": {"maxTokens": {"type": "number"}}}}})
    );
    for camel in ["maxTokens", "promptMode", "promptCacheKey", "toolChoice", "topP"] {
        assert!(wire.get(camel).is_none(), "{camel} left in the wire payload");
    }
    assert_eq!(
        wire["tools"],
        json!([{"type": "function", "function": {
            "name": "lookup", "description": "Look something up",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            "strict": false,
        }}])
    );
    assert_eq!(
        wire["messages"],
        json!([
            {"role": "system", "content": "Be precise"},
            {"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": "data:image/png;base64,aGVsbG8="},
            ]},
        ])
    );
}

#[tokio::test]
async fn serializes_assistant_thinking_tool_calls_and_tool_results_for_replay() {
    let server = sse_server(&[terminal_event("stop")]).await;
    let model = model(&server.uri());
    let mut arguments = serde_json::Map::new();
    arguments.insert("query".into(), json!("pi"));
    let context = normalize_context(&Context::new(vec![
        Message::Assistant(assistant(
            "mistral",
            "mistral-conversations",
            &model.id,
            vec![
                AssistantBlock::Thinking(ThinkingContent::new("reason")),
                AssistantBlock::Text(TextContent::new("answer")),
                AssistantBlock::ToolCall(ToolCall {
                    id: "abc123456".into(),
                    name: "lookup".into(),
                    arguments,
                    ..ToolCall::default()
                }),
            ],
        )),
        Message::ToolResult(ToolResultMessage {
            tool_call_id: "abc123456".into(),
            tool_name: "lookup".into(),
            content: vec![
                UserBlock::text("found"),
                UserBlock::Image(ImageContent {
                    data: "aGVsbG8=".into(),
                    mime_type: "image/png".into(),
                }),
            ],
            is_error: false,
            timestamp: 2,
            ..ToolResultMessage::default()
        }),
    ]));

    let message = stream(&model, &context, options("test")).result().await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(
        last_body(&server).await["messages"],
        json!([
            {
                "role": "assistant",
                "prefix": false,
                "content": [
                    {"type": "thinking", "thinking": [{"type": "text", "text": "reason"}]},
                    {"type": "text", "text": "answer"},
                ],
                "tool_calls": [{
                    "id": "abc123456",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"query\":\"pi\"}"},
                    "index": 0,
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "abc123456",
                "name": "lookup",
                "content": [
                    {"type": "text", "text": "found"},
                    {"type": "image_url", "image_url": "data:image/png;base64,aGVsbG8="},
                ],
            },
        ])
    );
}

#[tokio::test]
async fn normalizes_foreign_tool_call_ids_with_short_hash() {
    let server = sse_server(&[terminal_event("stop")]).await;
    let mut foreign = assistant(
        "openai",
        "openai-responses",
        "gpt-5",
        vec![
            AssistantBlock::Thinking(ThinkingContent {
                thinking: "secret".into(),
                thinking_signature: Some("sig".into()),
                redacted: None,
            }),
            AssistantBlock::ToolCall(ToolCall {
                id: "call_abc123|fc_xyz".into(),
                name: "lookup".into(),
                ..ToolCall::default()
            }),
        ],
    );
    foreign.stop_reason = StopReason::ToolUse;
    let context = normalize_context(&Context::new(vec![
        Message::Assistant(foreign),
        Message::ToolResult(ToolResultMessage {
            tool_call_id: "call_abc123|fc_xyz".into(),
            tool_name: "lookup".into(),
            is_error: true,
            ..ToolResultMessage::default()
        }),
    ]));

    stream(&model(&server.uri()), &context, options("test")).result().await;

    let messages = last_body(&server).await["messages"].clone();
    // Thinking from another model is dropped; the id is shortHash("callabc123fcxyz")[..9].
    assert_eq!(
        messages,
        json!([
            {"role": "assistant", "prefix": false, "tool_calls": [{
                "id": "nrregu2qr", "type": "function", "function": {"name": "lookup", "arguments": "{}"}, "index": 0,
            }]},
            {"role": "tool", "tool_call_id": "nrregu2qr", "name": "lookup",
             "content": [{"type": "text", "text": "[tool error] (no tool output)"}]},
        ])
    );
}

#[tokio::test]
async fn orphaned_tool_calls_get_synthetic_results_and_errored_turns_are_dropped() {
    let server = sse_server(&[terminal_event("stop")]).await;
    let model = model(&server.uri());
    let mut failed = assistant("mistral", "mistral-conversations", &model.id, vec![]);
    failed.stop_reason = StopReason::Error;
    failed.content = vec![AssistantBlock::Text(TextContent::new("partial"))];
    let context = normalize_context(&Context::new(vec![
        Message::Assistant(assistant(
            "mistral",
            "mistral-conversations",
            &model.id,
            vec![AssistantBlock::ToolCall(ToolCall {
                id: "abc123456".into(),
                name: "lookup".into(),
                ..ToolCall::default()
            })],
        )),
        Message::user("never mind"),
        Message::Assistant(failed),
    ]));

    stream(&model, &context, options("test")).result().await;

    let messages = last_body(&server).await["messages"].clone();
    let roles: Vec<&str> = messages
        .as_array()
        .unwrap()
        .iter()
        .map(|message| message["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["assistant", "tool", "user"]);
    assert_eq!(
        messages[1]["content"],
        json!([{"type": "text", "text": "[tool error] No result provided"}])
    );
}

#[tokio::test]
async fn images_become_placeholders_for_text_only_models() {
    let server = sse_server(&[terminal_event("stop")]).await;
    let model = model_with(&server.uri(), "devstral", false, None, false);
    let image = UserBlock::Image(ImageContent {
        data: "aGVsbG8=".into(),
        mime_type: "image/png".into(),
    });
    let context = normalize_context(&Context::new(vec![
        Message::user(UserContent::Blocks(vec![image.clone()])),
        Message::Assistant(assistant(
            "mistral",
            "mistral-conversations",
            "devstral",
            vec![AssistantBlock::ToolCall(ToolCall {
                id: "abc123456".into(),
                name: "screenshot".into(),
                ..ToolCall::default()
            })],
        )),
        Message::ToolResult(ToolResultMessage {
            tool_call_id: "abc123456".into(),
            tool_name: "screenshot".into(),
            content: vec![image],
            ..ToolResultMessage::default()
        }),
    ]));

    stream(&model, &context, options("test")).result().await;

    let messages = last_body(&server).await["messages"].clone();
    assert_eq!(
        messages[0],
        json!({"role": "user", "content": [{"type": "text", "text": "(image omitted: model does not support images)"}]})
    );
    assert_eq!(
        messages[2]["content"],
        json!([{"type": "text", "text": "(tool image omitted: model does not support images)"}])
    );
}

#[tokio::test]
async fn keeps_mid_conversation_system_messages_only_when_supported() {
    let server = sse_server(&[terminal_event("stop")]).await;
    let messages = vec![
        Message::System(SystemMessage {
            content: "Base".into(),
            ..SystemMessage::default()
        }),
        Message::user("hi"),
        Message::System(SystemMessage {
            content: "Later".into(),
            sections: Some([("style".to_owned(), Some("terse".to_owned()))].into_iter().collect()),
            ..SystemMessage::default()
        }),
        Message::user("again"),
    ];
    let context = normalize_context(&Context::new(messages));

    let mut supporting = model(&server.uri());
    supporting.compat = Some(
        delta_ai::compat::ModelCompat::from_value(
            "mistral-conversations",
            json!({"supportsMidConvoSystemMessages": true}),
        )
        .unwrap(),
    );
    stream(&supporting, &context, options("k")).result().await;
    let wire = last_body(&server).await["messages"].clone();
    let roles: Vec<&str> = wire
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["system", "user", "system", "user"]);
    assert_eq!(
        wire[2]["content"],
        json!("Later\n\nUpdated system prompt section \"style\":\n\nterse")
    );

    stream(&model(&server.uri()), &context, options("k")).result().await;
    let wire = last_body(&server).await["messages"].clone();
    let roles: Vec<&str> = wire
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["system", "user", "user"]);
    assert_eq!(wire[0]["content"], json!("Base\n\nLater\n\nterse"));
}

#[tokio::test]
async fn honors_case_insensitive_header_overrides_and_affinity_suppression() {
    let server = sse_server(&[terminal_event("stop")]).await;
    // Model headers reach the API merged into `options.headers` by `Models`.
    let headers = [
        ("Authorization", Some("Bearer model-key")),
        ("X-Affinity", Some("model-affinity")),
        ("authorization", None),
        ("x-affinity", None),
        ("User-Agent", Some("custom-agent")),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_owned(), value.map(str::to_owned)))
    .collect();
    stream(
        &model(&server.uri()),
        &hello(),
        StreamOptions {
            session_id: Some("automatic-affinity".into()),
            headers: Some(headers),
            ..options("request-key")
        },
    )
    .result()
    .await;

    let requests = server.received_requests().await.unwrap();
    let request = &requests[0];
    assert!(request.headers.get("authorization").is_none());
    assert!(request.headers.get("x-affinity").is_none());
    assert_eq!(request.headers.get("user-agent").unwrap(), "custom-agent");
}

#[tokio::test]
async fn prompt_cache_key_follows_session_and_cache_retention() {
    let server = sse_server(&[terminal_event("stop")]).await;
    let model = model(&server.uri());
    let (hook, captured) = capture_payload();
    stream(
        &model,
        &hello(),
        StreamOptions {
            session_id: Some("session-123".into()),
            on_payload: Some(hook.clone()),
            ..options("k")
        },
    )
    .result()
    .await;
    assert_eq!(
        captured.lock().unwrap().as_ref().unwrap()["promptCacheKey"],
        json!("session-123")
    );

    stream(
        &model,
        &hello(),
        StreamOptions {
            session_id: Some("session-123".into()),
            cache_retention: Some(CacheRetention::None),
            on_payload: Some(hook),
            ..options("k")
        },
    )
    .result()
    .await;
    assert!(
        captured
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .get("promptCacheKey")
            .is_none()
    );
    let requests = server.received_requests().await.unwrap();
    assert!(requests[1].headers.get("x-affinity").is_none());
    assert!(
        requests[1]
            .body_json::<Value>()
            .unwrap()
            .get("prompt_cache_key")
            .is_none()
    );
}

#[tokio::test]
async fn strict_tool_schemas_are_serialized_for_constrained_sampling() {
    let server = server_with(ResponseTemplate::new(400).set_body_string(r#"{"message":"nope"}"#)).await;
    let mut tool = Tool::new(
        "inspect_schema",
        "Inspect the schema",
        json!({"type": "object", "properties": {"nested": {"type": "object",
            "properties": {"value": {"type": "string"}}, "required": ["value"]}}, "required": ["nested"]}),
    );
    tool.constrained_sampling = Some(ConstrainedSampling::JsonSchema {
        strict: StrictMode::Require,
    });
    let context = normalize_context(&Context::new(vec![Message::user("Hi")]).with_tools(vec![tool]));
    let (hook, captured) = capture_payload();

    let message = stream(
        &model(&server.uri()),
        &context,
        StreamOptions {
            on_payload: Some(hook),
            ..options("fake-key")
        },
    )
    .result()
    .await;

    let payload = captured.lock().unwrap().clone().unwrap();
    let function = &payload["tools"][0]["function"];
    assert_eq!(function["strict"], json!(true));
    assert_eq!(function["parameters"]["additionalProperties"], json!(false));
    assert_eq!(
        function["parameters"]["properties"]["nested"]["required"],
        json!(["value"])
    );
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some(r#"Mistral API error (400): {"message":"nope"}"#)
    );
}

#[tokio::test]
async fn required_strict_tool_with_unsupported_schema_fails_before_the_request() {
    let server = sse_server(&[terminal_event("stop")]).await;
    let mut tool = Tool::new("t", "d", json!({"type": "object", "oneOf": [{}]}));
    tool.constrained_sampling = Some(ConstrainedSampling::JsonSchema {
        strict: StrictMode::Require,
    });
    let context = normalize_context(&Context::new(vec![Message::user("Hi")]).with_tools(vec![tool]));
    let (events, message) = collect(stream(&model(&server.uri()), &context, options("k"))).await;
    assert_eq!(events.len(), 1);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Tool \"t\" requires JSON-schema constrained sampling, but oneOf schemas are unsupported.")
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}

// ---------------------------------------------------------------------------------------------
// Stream parsing
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn parses_native_thinking_text_fragmented_tool_calls_and_cached_usage() {
    let mut last = chunk(
        &json!({"tool_calls": [{"index": 0, "function": {"name": "", "arguments": "\"pi\"}"}}]}),
        Some("tool_calls"),
    );
    last["usage"] = json!({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
        "prompt_tokens_details": {"cached_tokens": 3}});
    let server = sse_server(&[
        chunk(
            &json!({"content": [{"type": "thinking", "thinking": [{"type": "text", "text": "reason"}]}]}),
            None,
        ),
        chunk(&json!({"content": [{"type": "text", "text": "answer"}]}), None),
        chunk(
            &json!({"tool_calls": [{"id": "abc123456", "index": 0, "function": {"name": "lookup", "arguments": "{\"query\":"}}]}),
            None,
        ),
        last,
    ])
    .await;

    let (events, message) = collect(stream(&model(&server.uri()), &hello(), options("test"))).await;

    let kinds: Vec<&str> = events.iter().map(AssistantMessageEvent::kind).collect();
    assert_eq!(
        kinds,
        [
            "start",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "text_start",
            "text_delta",
            "text_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_delta",
            "toolcall_end",
            "done"
        ]
    );
    assert_eq!(message.stop_reason, StopReason::ToolUse);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("tool_calls"));
    assert_eq!(message.response_id.as_deref(), Some("response-1"));
    assert_eq!(
        serde_json::to_value(&message.content).unwrap(),
        json!([
            {"type": "thinking", "thinking": "reason"},
            {"type": "text", "text": "answer"},
            {"type": "toolCall", "id": "abc123456", "name": "lookup", "arguments": {"query": "pi"}},
        ])
    );
    let usage = &message.usage;
    assert_eq!(
        (
            usage.input,
            usage.output,
            usage.cache_read,
            usage.cache_write,
            usage.total_tokens
        ),
        (7, 4, 3, 0, 14)
    );
    let expected_cost = (7.0 * 2.0 + 4.0 * 6.0 + 3.0 * 0.2) / 1_000_000.0;
    assert!((usage.cost.total - expected_cost).abs() < 1e-12);
}

#[tokio::test]
async fn ignores_empty_content_deltas() {
    // GLM models on Mistral send empty content deltas at the start, around tool calls and
    // mid-thinking; they must not open blocks or split thinking (#9674).
    let thinking = |text: &str| json!({"type": "thinking", "thinking": [{"type": "text", "text": text}]});
    let tool_call = |arguments: &str, first: bool| {
        if first {
            json!({"index": 0, "id": "abc123456", "function": {"name": "read", "arguments": arguments}})
        } else {
            json!({"index": 0, "function": {"name": "", "arguments": arguments}})
        }
    };
    let deltas = [
        json!({"content": ""}),
        json!({"content": [thinking("first part,")]}),
        json!({"content": ""}),
        json!({"content": [{"type": "text", "text": ""}]}),
        json!({"content": [thinking(" second part."), {"type": "text", "text": "Reading."}]}),
        json!({"content": "", "tool_calls": [tool_call("", true)]}),
        json!({"content": "", "tool_calls": [tool_call("{\"path\":", false)]}),
        json!({"content": "", "tool_calls": [tool_call("\"a.txt\"}", false)]}),
        json!({"content": ""}),
    ];
    let last = deltas.len() - 1;
    let events: Vec<Value> = deltas
        .into_iter()
        .enumerate()
        .map(|(i, delta)| chunk(&delta, (i == last).then_some("tool_calls")))
        .collect();
    let server = sse_server(&events).await;

    let message = stream(
        &model_with(&server.uri(), "zai-glm-5-3", true, None, false),
        &hello(),
        options("test"),
    )
    .result()
    .await;

    assert_eq!(
        serde_json::to_value(&message.content).unwrap(),
        json!([
            {"type": "thinking", "thinking": "first part, second part."},
            {"type": "text", "text": "Reading."},
            {"type": "toolCall", "id": "abc123456", "name": "read", "arguments": {"path": "a.txt"}},
        ])
    );
}

#[tokio::test]
async fn handles_whole_parallel_tool_calls_and_null_ids() {
    let server = sse_server(&[
        chunk(
            &json!({"content": "Calling.", "tool_calls": [
                {"id": "aaaaaaaa1", "index": 0, "function": {"name": "a", "arguments": {"x": 1}}},
                {"id": "bbbbbbbb2", "index": 1, "function": {"name": "b", "arguments": "{\"y\":2}"}},
            ]}),
            None,
        ),
        // Continuation chunk with Mistral's literal "null" id.
        chunk(
            &json!({"tool_calls": [{"id": "null", "index": 1, "function": {"name": "", "arguments": ""}}]}),
            None,
        ),
        // Same index, different real id: a new call, not a continuation.
        chunk(
            &json!({"tool_calls": [{"id": "ccccccccc", "index": 0, "function": {"name": "c", "arguments": "{}"}}]}),
            None,
        ),
        // No id and no index: an id is derived from the index fallback.
        chunk(
            &json!({"tool_calls": [{"function": {"name": "d", "arguments": "{\"z\":"}}]}),
            Some("tool_calls"),
        ),
    ])
    .await;

    let message = stream(&model(&server.uri()), &hello(), options("test")).result().await;

    assert_eq!(message.stop_reason, StopReason::ToolUse);
    let calls: Vec<&ToolCall> = message.tool_calls().collect();
    let summary: Vec<(&str, &str, Value)> = calls
        .iter()
        .map(|call| {
            (
                call.id.as_str(),
                call.name.as_str(),
                Value::Object(call.arguments.clone()),
            )
        })
        .collect();
    assert_eq!(summary[0], ("aaaaaaaa1", "a", json!({"x": 1})));
    assert_eq!(summary[1], ("bbbbbbbb2", "b", json!({"y": 2})));
    assert_eq!(summary[2], ("ccccccccc", "c", json!({})));
    assert_eq!(summary[3].1, "d");
    assert_eq!(summary[3].0.len(), 9);
    assert!(summary[3].0.chars().all(|c| c.is_ascii_alphanumeric()));
    // Incomplete arguments: a tool-level error, not a stream error.
    assert!(calls[3].arguments_error.is_some());
    assert_eq!(message.text(), "Calling.");
}

#[tokio::test]
async fn forwards_each_parsed_sse_payload_before_normalizing_it() {
    let events = [
        json!({"id": "response-1", "provider_metadata": {"request": "test"},
               "choices": [{"index": 0, "finish_reason": null, "delta": {"content": "hello"}}]}),
        json!({"id": "response-1", "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}),
    ];
    let server = sse_server(&events).await;
    let received = Arc::new(Mutex::new(Vec::new()));
    let sink = received.clone();
    let hook: OnProviderStreamEvent = Arc::new(move |event: &Value, model: &Model| {
        sink.lock().unwrap().push((event.clone(), model.id.clone()));
        Box::pin(async { Ok(()) })
    });

    let message = stream(
        &model(&server.uri()),
        &hello(),
        StreamOptions {
            on_provider_stream_event: Some(hook),
            ..options("test")
        },
    )
    .result()
    .await;

    let received = received.lock().unwrap().clone();
    assert_eq!(
        received,
        events
            .iter()
            .map(|event| (event.clone(), "mistral-large-latest".to_owned()))
            .collect::<Vec<_>>()
    );
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "hello");
}

#[tokio::test]
async fn parses_sse_and_utf8_sequences_split_across_transport_chunks() {
    let event = json!({"id": "response-bytewise", "model": "mistral-large-latest",
        "choices": [{"index": 0, "finish_reason": "stop", "delta": {"content": "héllo 🌍"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}});
    let body = format!("data: {event}\r\n\r\ndata: [DONE]\r\n\r\n");
    let chunks = body
        .into_bytes()
        .into_iter()
        .map(|byte| (vec![byte], Duration::ZERO))
        .collect();
    let base_url = raw_sse_server(chunks, false);

    let message = stream(&model(&base_url), &hello(), options("test")).result().await;

    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    assert_eq!(message.text(), "héllo 🌍");
}

#[tokio::test]
async fn raw_stop_reasons_are_preserved() {
    for (finish_reason, stop_reason, error) in [
        ("stop", StopReason::Stop, None),
        ("length", StopReason::Length, None),
        ("model_length", StopReason::Length, None),
        (
            "error",
            StopReason::Error,
            Some("Provider stopped with: error (server error)"),
        ),
        (
            "unmapped_error",
            StopReason::Error,
            Some("Provider stopped with: unmapped_error"),
        ),
    ] {
        let server = server_with(
            ResponseTemplate::new(200).set_body_raw(sse(&[terminal_event(finish_reason)], "\n\n"), "text/event-stream"),
        )
        .await;
        let message = stream(&model(&server.uri()), &hello(), options("test")).result().await;
        assert_eq!(message.stop_reason, stop_reason, "{finish_reason}");
        assert_eq!(message.raw_stop_reason.as_deref(), Some(finish_reason));
        assert_eq!(message.error_message.as_deref(), error);
    }
}

#[tokio::test]
async fn stream_without_finish_reason_is_an_error() {
    let server = sse_server(&[chunk(&json!({"content": "partial"}), None)]).await;
    let message = stream(&model(&server.uri()), &hello(), options("test")).result().await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Mistral stream ended without a finish reason")
    );
    assert_eq!(message.text(), "partial");
}

#[tokio::test]
async fn in_stream_error_payloads_fail_the_stream() {
    let server = server_with(ResponseTemplate::new(200).set_body_raw(
        "data: {\"object\":\"error\",\"message\":\"boom\"}\n\n",
        "text/event-stream",
    ))
    .await;
    let message = stream(&model(&server.uri()), &hello(), options("test")).result().await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Invalid Mistral streaming event")
    );
}

// ---------------------------------------------------------------------------------------------
// Errors, timeouts and cancellation
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn preserves_http_status_and_response_bodies_in_errors() {
    let server = server_with(ResponseTemplate::new(403).set_body_string(r#"{"message":"blocked by gateway"}"#)).await;
    let statuses = Arc::new(Mutex::new(Vec::new()));
    let sink = statuses.clone();
    let on_response: OnResponse = Arc::new(move |response: &delta_ai::options::ProviderResponse, _model: &Model| {
        sink.lock().unwrap().push(response.status);
        Box::pin(async { Ok(()) })
    });

    let (events, message) = collect(stream(
        &model(&server.uri()),
        &hello(),
        StreamOptions {
            on_response: Some(on_response),
            ..options("test")
        },
    ))
    .await;

    assert_eq!(events.len(), 1, "no start event before the request is accepted");
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some(r#"Mistral API error (403): {"message":"blocked by gateway"}"#)
    );
    assert_eq!(*statuses.lock().unwrap(), [403]);
    // As upstream, there are no client-side retries.
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn empty_error_bodies_use_the_status_text() {
    let server = server_with(ResponseTemplate::new(503)).await;
    let message = stream(&model(&server.uri()), &hello(), options("test")).result().await;
    assert_eq!(
        message.error_message.as_deref(),
        Some("Mistral API error (503): Service Unavailable")
    );
}

#[tokio::test]
async fn missing_api_key_fails() {
    let server = sse_server(&[terminal_event("stop")]).await;
    let model = model(&server.uri());
    let message = stream(&model, &hello(), StreamOptions::default()).result().await;
    assert_eq!(
        message.error_message.as_deref(),
        Some("No API key for provider: mistral")
    );
    let message = stream_simple(&model, &hello(), SimpleStreamOptions::default())
        .result()
        .await;
    assert_eq!(
        message.error_message.as_deref(),
        Some("No API key for provider: mistral")
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn applies_the_request_timeout_while_waiting_for_response_headers() {
    let server = server_with(
        ResponseTemplate::new(200)
            .set_delay(Duration::from_millis(500))
            .set_body_raw(sse(&[terminal_event("stop")], "\n\n"), "text/event-stream"),
    )
    .await;
    let message = stream(
        &model(&server.uri()),
        &hello(),
        StreamOptions {
            timeout: Some(Duration::from_millis(20)),
            ..options("test")
        },
    )
    .result()
    .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Mistral response headers timed out after 20ms")
    );
}

#[tokio::test]
async fn does_not_abort_an_active_stream_that_lasts_longer_than_the_timeout() {
    // Regression for upstream #10609.
    let thinking = json!({"choices": [{"index": 0,
        "delta": {"content": [{"type": "thinking", "thinking": [{"type": "text", "text": "x"}]}]}}]});
    let mut chunks: Vec<(Vec<u8>, Duration)> = (0..5)
        .map(|_| (format!("data: {thinking}\n\n").into_bytes(), Duration::from_millis(15)))
        .collect();
    chunks.push((
        format!("data: {}\n\ndata: [DONE]\n\n", terminal_event("stop")).into_bytes(),
        Duration::from_millis(15),
    ));
    let base_url = raw_sse_server(chunks, false);

    let message = stream(
        &model(&base_url),
        &hello(),
        StreamOptions {
            timeout: Some(Duration::from_millis(20)),
            ..options("test")
        },
    )
    .result()
    .await;

    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    assert_eq!(
        serde_json::to_value(&message.content).unwrap(),
        json!([{"type": "thinking", "thinking": "xxxxx"}])
    );
}

#[tokio::test]
async fn aborts_while_waiting_for_an_sse_chunk() {
    let first = chunk(&json!({"content": "hel"}), None);
    let base_url = raw_sse_server(vec![(format!("data: {first}\n\n").into_bytes(), Duration::ZERO)], true);
    let cancel = CancellationToken::new();
    let mut events = stream(
        &model(&base_url),
        &hello(),
        StreamOptions {
            cancel: Some(cancel.clone()),
            ..options("test")
        },
    );
    // Wait until the first text delta arrived, then abort while the body stalls.
    while let Some(event) = events.next().await {
        if matches!(event, AssistantMessageEvent::TextDelta { .. }) {
            break;
        }
    }
    cancel.cancel();
    let message = events.result().await;
    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert_eq!(message.error_message.as_deref(), Some("Request was aborted"));
    assert_eq!(message.text(), "hel");
}

#[tokio::test]
async fn cancelled_before_the_request_is_aborted() {
    let server = sse_server(&[terminal_event("stop")]).await;
    let cancel = CancellationToken::new();
    cancel.cancel();
    let message = stream(
        &model(&server.uri()),
        &hello(),
        StreamOptions {
            cancel: Some(cancel),
            ..options("test")
        },
    )
    .result()
    .await;
    assert_eq!(message.stop_reason, StopReason::Aborted);
}

// ---------------------------------------------------------------------------------------------
// stream_simple: reasoning mode selection
// ---------------------------------------------------------------------------------------------

fn none_high_levels() -> Value {
    json!({"off": "none", "minimal": null, "low": null, "medium": null, "high": "high", "xhigh": null, "max": null})
}

fn glm_5_2_levels() -> Value {
    let mut map = none_high_levels();
    map["max"] = json!("max");
    map
}

fn glm_5_3_levels() -> Value {
    let mut map = none_high_levels();
    map["off"] = Value::Null;
    map["low"] = json!("low");
    map["max"] = json!("max");
    map
}

async fn simple_payload(model: &Model, reasoning: Option<ThinkingLevel>) -> Value {
    let (hook, captured) = capture_payload();
    let options = SimpleStreamOptions {
        base: StreamOptions {
            on_payload: Some(hook),
            ..options("fake-key")
        },
        reasoning,
        ..SimpleStreamOptions::default()
    };
    // The base URL refuses connections; only the payload matters.
    stream_simple(model, &hello(), options).result().await;
    captured.lock().unwrap().clone().unwrap()
}

const DEAD_URL: &str = "http://127.0.0.1:9";

#[tokio::test]
async fn uses_prompt_mode_for_reasoning_models_without_a_level_map() {
    let magistral = model_with(DEAD_URL, "magistral-medium-latest", true, None, false);
    let payload = simple_payload(&magistral, Some(ThinkingLevel::Medium)).await;
    assert_eq!(payload["promptMode"], json!("reasoning"));
    assert!(payload.get("reasoningEffort").is_none());

    let payload = simple_payload(&magistral, None).await;
    assert!(payload.get("promptMode").is_none());
    assert!(payload.get("reasoningEffort").is_none());
}

#[tokio::test]
async fn uses_reasoning_effort_for_models_with_a_level_map() {
    for (id, map) in [
        ("mistral-small-2603", none_high_levels()),
        ("mistral-medium-latest", none_high_levels()),
        ("zai-glm-5-2", glm_5_2_levels()),
    ] {
        let model = model_with(DEAD_URL, id, true, Some(map), false);
        let payload = simple_payload(&model, Some(ThinkingLevel::High)).await;
        assert_eq!(payload["reasoningEffort"], json!("high"), "{id}");
        assert!(payload.get("promptMode").is_none());
        // Unsupported levels clamp to a supported effort.
        let payload = simple_payload(&model, Some(ThinkingLevel::Low)).await;
        assert_eq!(payload["reasoningEffort"], json!("high"), "{id}");
        // Thinking off sends the mapped `off` value.
        let payload = simple_payload(&model, None).await;
        assert_eq!(payload["reasoningEffort"], json!("none"), "{id}");
        assert!(payload.get("promptMode").is_none());
    }

    let glm_5_2 = model_with(DEAD_URL, "zai-glm-5-2", true, Some(glm_5_2_levels()), false);
    assert_eq!(
        simple_payload(&glm_5_2, Some(ThinkingLevel::Max)).await["reasoningEffort"],
        json!("max")
    );

    let glm_5_3 = model_with(DEAD_URL, "zai-glm-5-3", true, Some(glm_5_3_levels()), false);
    for (level, expected) in [
        (ThinkingLevel::Low, "low"),
        (ThinkingLevel::High, "high"),
        (ThinkingLevel::Max, "max"),
        (ThinkingLevel::Medium, "high"),
    ] {
        let payload = simple_payload(&glm_5_3, Some(level)).await;
        assert_eq!(payload["reasoningEffort"], json!(expected), "{level:?}");
        assert!(payload.get("promptMode").is_none());
    }
    // `off` maps to null: no effort is sent.
    assert!(simple_payload(&glm_5_3, None).await.get("reasoningEffort").is_none());
}

#[tokio::test]
async fn omits_reasoning_controls_for_non_reasoning_models() {
    let model = model_with(DEAD_URL, "mistral-medium-2505", false, Some(none_high_levels()), false);
    let payload = simple_payload(&model, Some(ThinkingLevel::Medium)).await;
    assert!(payload.get("reasoningEffort").is_none());
    assert!(payload.get("promptMode").is_none());
}

#[tokio::test]
async fn simple_options_map_tool_choice_and_clamp_max_tokens() {
    let server = sse_server(&[terminal_event("stop")]).await;
    let model = model(&server.uri());
    let message = MistralConversationsApi
        .stream_simple(
            &model,
            &hello(),
            SimpleStreamOptions {
                base: StreamOptions {
                    temperature: Some(0.5),
                    ..options("k")
                },
                tool_choice: Some(ToolChoice::None),
                ..SimpleStreamOptions::default()
            },
        )
        .result()
        .await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    let wire = last_body(&server).await;
    assert_eq!(wire["tool_choice"], json!("none"));
    assert_eq!(wire["temperature"], json!(0.5));
    assert_eq!(wire["max_tokens"], json!(16_384));
}
