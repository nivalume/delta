//! End-to-end tests of the `anthropic-messages` API against recorded SSE bodies served by
//! wiremock, plus request payload checks through `on_payload`.
//!
//! Cases are ported from upstream `test/anthropic-*.test.ts`, `cache-retention.test.ts` and
//! pi-py's `test_anthropic_*.py`.

// Test helpers panic on unexpected shapes by design.
#![allow(clippy::unwrap_used, clippy::needless_pass_by_value)]

use futures::StreamExt;
use futures::future::FutureExt;
use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tau_ai::api::anthropic_messages::{AnthropicMessagesApi, stream, stream_simple};
use tau_ai::model::ThinkingLevel;
use tau_ai::options::{
    AnthropicEffort, AnthropicOptions, AnthropicToolChoice, ApiOptions, OnPayload, OnProviderStreamEvent,
    SimpleStreamOptions, StreamOptions, ThinkingDisplay, ToolChoice,
};
use tau_ai::types::{
    AssistantBlock, AssistantMessage, ConstrainedSampling, ImageContent, Message, StopReason, StrictMode,
    SystemMessage, TextContent, ThinkingContent, ToolCall, ToolReference, ToolResultMessage, UserBlock, UserContent,
    UserMessage,
};
use tau_ai::{AssistantMessageEvent, AssistantMessageEventStream, Context, Model, ProviderStreams, Tool};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const OFFLINE: &str = "http://127.0.0.1:9";

fn model(base_url: &str, extra: Value) -> Model {
    let mut value = json!({
        "id": "claude-test", "name": "Claude Test", "api": "anthropic-messages", "provider": "anthropic",
        "baseUrl": base_url, "reasoning": false, "input": ["text"],
        "cost": {"input": 3, "output": 15, "cacheRead": 0.3, "cacheWrite": 3.75},
        "contextWindow": 200_000, "maxTokens": 1024
    });
    if let Value::Object(extra) = extra {
        for (key, field) in extra {
            value[key] = field;
        }
    }
    serde_json::from_value(value).unwrap()
}

fn transcript(context: &Context) -> tau_ai::TranscriptContext {
    tau_ai::normalize_context(context)
}

fn hello() -> Context {
    Context::new(vec![user("Hello", 1)])
}

fn user(text: &str, timestamp: i64) -> Message {
    Message::User(UserMessage {
        content: text.into(),
        timestamp,
    })
}

fn assistant(content: Vec<AssistantBlock>, stop_reason: StopReason) -> Message {
    let mut message = AssistantMessage::empty("anthropic-messages", "anthropic", "claude-test");
    message.content = content;
    message.stop_reason = stop_reason;
    message.timestamp = 1;
    Message::Assistant(message)
}

fn tool_call(id: &str, name: &str, arguments: &Value) -> AssistantBlock {
    AssistantBlock::ToolCall(ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: arguments.as_object().cloned().unwrap_or_default(),
        ..ToolCall::default()
    })
}

fn tool_result(id: &str, name: &str, content: Vec<UserBlock>, is_error: bool) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: id.into(),
        tool_name: name.into(),
        content,
        is_error,
        timestamp: 2,
        ..ToolResultMessage::default()
    })
}

fn text(text: &str) -> UserBlock {
    UserBlock::Text(TextContent::new(text))
}

fn image(data: &str) -> UserBlock {
    UserBlock::Image(ImageContent {
        data: data.into(),
        mime_type: "image/png".into(),
    })
}

fn empty_tool(name: &str) -> Tool {
    Tool::new(
        name,
        format!("{name} tool"),
        json!({"type": "object", "properties": {}}),
    )
}

#[allow(clippy::unnecessary_wraps)] // Shaped for the `StreamOptions::api` field.
fn api(options: AnthropicOptions) -> Option<ApiOptions> {
    Some(ApiOptions::AnthropicMessages(options))
}

/// A client per test: the shared default client's pool must not outlive a test's runtime.
fn client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

fn key() -> StreamOptions {
    StreamOptions {
        api_key: Some("k".into()),
        http_client: Some(client()),
        ..StreamOptions::default()
    }
}

/// SSE body like upstream's `createSseResponse` (`event: <type>` + `data:`).
fn sse(events: &[Value]) -> String {
    events
        .iter()
        .map(|event| match event {
            Value::Array(pair) => format!(
                "event: {}\ndata: {}\n\n",
                pair[0].as_str().unwrap(),
                pair[1].as_str().unwrap()
            ),
            event => format!("event: {}\ndata: {event}\n\n", event["type"].as_str().unwrap()),
        })
        .collect()
}

/// A raw SSE event: name and unparsed data.
fn raw(name: &str, data: &str) -> Value {
    json!([name, data])
}

fn message_start() -> Value {
    json!({"type": "message_start", "message": {"id": "msg_test", "model": "claude-test",
        "usage": {"input_tokens": 12, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}})
}

fn message_delta(stop_reason: &str) -> Value {
    json!({"type": "message_delta", "delta": {"stop_reason": stop_reason},
        "usage": {"input_tokens": 12, "output_tokens": 5, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}})
}

fn minimal_events() -> Vec<Value> {
    vec![
        message_start(),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}),
        json!({"type": "content_block_stop", "index": 0}),
        message_delta("end_turn"),
        json!({"type": "message_stop"}),
    ]
}

fn sse_response(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
}

async fn serve(body: String) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;
    server
}

async fn collect(mut stream: AssistantMessageEventStream) -> (Vec<AssistantMessageEvent>, AssistantMessage) {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    let result = events
        .last()
        .and_then(AssistantMessageEvent::final_message)
        .unwrap()
        .clone();
    (events, result)
}

fn kinds(events: &[AssistantMessageEvent]) -> Vec<&'static str> {
    events.iter().map(AssistantMessageEvent::kind).collect()
}

/// Stream `events` from a mock server and return the server, events and final message.
async fn run_events(
    model_extra: Value,
    events: &[Value],
    context: &Context,
    mut options: StreamOptions,
) -> (MockServer, Vec<AssistantMessageEvent>, AssistantMessage) {
    options.http_client.get_or_insert_with(client);
    let server = serve(sse(events)).await;
    let model = model(&server.uri(), model_extra);
    let (events, result) = collect(stream(&model, &transcript(context), options)).await;
    (server, events, result)
}

async fn requests(server: &MockServer) -> Vec<Request> {
    server.received_requests().await.unwrap()
}

fn header<'a>(request: &'a Request, name: &str) -> Option<&'a str> {
    request.headers.get(name).map(|value| value.to_str().unwrap())
}

fn on_payload_capture(store: Arc<Mutex<Vec<Value>>>, fail: bool) -> OnPayload {
    Arc::new(move |payload: &Value, _model: &Model| {
        store.lock().unwrap().push(payload.clone());
        async move { if fail { Err("payload captured".into()) } else { Ok(None) } }.boxed()
    })
}

/// Capture the payload passed to `on_payload`; the request then fails before any HTTP call.
async fn capture(model: &Model, context: &Context, mut options: StreamOptions) -> (Value, AssistantMessage) {
    let store = Arc::new(Mutex::new(Vec::new()));
    if options.api_key.is_none() {
        options.api_key = Some("fake-key".into());
    }
    options.on_payload = Some(on_payload_capture(store.clone(), true));
    let (_, result) = collect(stream(model, &transcript(context), options)).await;
    assert_eq!(result.error_message.as_deref(), Some("payload captured"));
    let payload = store.lock().unwrap()[0].clone();
    (payload, result)
}

async fn capture_simple(
    model: &Model,
    context: &Context,
    mut options: SimpleStreamOptions,
) -> (Value, AssistantMessage) {
    let store = Arc::new(Mutex::new(Vec::new()));
    if options.base.api_key.is_none() {
        options.base.api_key = Some("fake-key".into());
    }
    options.base.on_payload = Some(on_payload_capture(store.clone(), true));
    let (_, result) = collect(stream_simple(model, &transcript(context), options)).await;
    assert_eq!(result.error_message.as_deref(), Some("payload captured"));
    let payload = store.lock().unwrap()[0].clone();
    (payload, result)
}

fn simple(reasoning: Option<ThinkingLevel>) -> SimpleStreamOptions {
    SimpleStreamOptions {
        reasoning,
        ..SimpleStreamOptions::default()
    }
}

fn no_cache() -> StreamOptions {
    StreamOptions {
        cache_retention: Some(tau_ai::CacheRetention::None),
        ..StreamOptions::default()
    }
}

fn assistant_content(payload: &Value) -> &Value {
    &payload["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "assistant")
        .unwrap()["content"]
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streams_text_and_sends_sdk_headers() {
    let (server, events, result) = run_events(json!({}), &minimal_events(), &hello(), key()).await;
    assert_eq!(
        kinds(&events),
        ["start", "text_start", "text_delta", "text_end", "done"]
    );
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text(), "Hello");
    assert_eq!(result.response_id.as_deref(), Some("msg_test"));
    assert_eq!(result.response_model, None);
    assert_eq!(result.raw_stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(
        (result.usage.input, result.usage.output, result.usage.total_tokens),
        (12, 5, 17)
    );
    assert_eq!(result.usage.cache_write_1h, Some(0));

    let requests = requests(&server).await;
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.url.path(), "/v1/messages");
    assert_eq!(request.url.query(), Some("beta=true"));
    assert_eq!(header(request, "x-api-key"), Some("k"));
    assert_eq!(header(request, "authorization"), None);
    assert_eq!(header(request, "anthropic-version"), Some("2023-06-01"));
    assert_eq!(header(request, "content-type"), Some("application/json"));
    assert_eq!(header(request, "accept"), Some("application/json"));
    assert_eq!(
        header(request, "anthropic-dangerous-direct-browser-access"),
        Some("true")
    );
    assert!(header(request, "user-agent").unwrap().starts_with("tau/"));
    assert_eq!(header(request, "anthropic-beta"), None);
    let body: Value = request.body_json().unwrap();
    assert_eq!(body["model"], "claude-test");
    assert_eq!(body["stream"], true);
    assert!(body.get("betas").is_none());
}

#[tokio::test]
async fn provider_streams_impl_delegates() {
    let server = serve(sse(&minimal_events())).await;
    let model = model(&server.uri(), json!({}));
    let (_, result) = collect(AnthropicMessagesApi.stream(&model, &transcript(&hello()), key())).await;
    assert_eq!(result.text(), "Hello", "{:?}", result.error_message);
}

#[tokio::test]
async fn forwards_parsed_provider_events_in_order() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let store = seen.clone();
    let hook: OnProviderStreamEvent = Arc::new(move |event: &Value, _: &Model| {
        store.lock().unwrap().push(event["type"].as_str().unwrap().to_owned());
        async { Ok(()) }.boxed()
    });
    let mut events = minimal_events();
    events.insert(1, raw("ping", r#"{"type": "ping"}"#));
    events.push(raw("done", "[DONE]"));
    events.push(raw("proxy.stats", "not json"));
    let options = StreamOptions {
        on_provider_stream_event: Some(hook),
        ..key()
    };
    let (_, _, result) = run_events(json!({}), &events, &hello(), options).await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(
        *seen.lock().unwrap(),
        [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop"
        ]
    );
}

#[tokio::test]
async fn replays_recorded_thinking_and_parallel_tool_calls() {
    let body = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/anthropic-messages/thinking_parallel_tools.sse"
    ))
    .unwrap();
    let server = serve(body).await;
    let model = model(&server.uri(), json!({"reasoning": true}));
    let (events, result) = collect(stream(&model, &transcript(&hello()), key())).await;

    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert_eq!(result.response_model.as_deref(), Some("claude-sonnet-4-5-20250929"));
    assert_eq!(result.content.len(), 4);
    let thinking = result.content[0].as_thinking().unwrap();
    assert_eq!(
        thinking.thinking,
        "The user wants the weather in two cities. I'll call the tool twice."
    );
    assert_eq!(
        thinking.thinking_signature.as_deref(),
        Some("EqQBCkgIBRABGAIiQL0qx7Yv4Q==")
    );
    assert_eq!(result.text(), "Checking both cities.");
    let calls: Vec<&ToolCall> = result.tool_calls().collect();
    assert_eq!(calls[0].id, "toolu_01A");
    assert_eq!(Value::Object(calls[0].arguments.clone()), json!({"city": "Paris"}));
    assert_eq!(calls[1].id, "toolu_01B");
    assert_eq!(Value::Object(calls[1].arguments.clone()), json!({"city": "Tokyo"}));

    // Interleaved deltas keep their content index.
    let deltas: Vec<(usize, &str)> = events
        .iter()
        .filter_map(|event| match event {
            AssistantMessageEvent::ToolCallDelta { content_index, delta } => Some((*content_index, delta.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(
        deltas,
        [(2, "{\"city\": \"Pa"), (3, "{\"city\": \"Tokyo\"}"), (2, "ris\"}")]
    );
    let thinking_end = events.iter().find_map(|event| match event {
        AssistantMessageEvent::ThinkingEnd { signature, .. } => Some(signature.clone()),
        _ => None,
    });
    assert_eq!(thinking_end, Some(Some("EqQBCkgIBRABGAIiQL0qx7Yv4Q==".into())));

    // Usage: cache tokens from message_start survive message_delta, reasoning reported.
    let usage = &result.usage;
    assert_eq!(
        (
            usage.input,
            usage.output,
            usage.cache_read,
            usage.cache_write,
            usage.reasoning
        ),
        (2045, 112, 3000, 1500, Some(40))
    );
    assert_eq!(usage.total_tokens, 2045 + 112 + 3000 + 1500);
    let expected = (2045.0 * 3.0 + 112.0 * 15.0 + 3000.0 * 0.3 + 1500.0 * 3.75) / 1_000_000.0;
    assert!((usage.cost.total - expected).abs() < 1e-12);

    // The streaming consumer's view matches the final message.
    assert_eq!(kinds(&events)[0], "start");
}

#[tokio::test]
async fn keeps_content_from_content_block_start_events() {
    let events = vec![
        message_start(),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "Initial text"}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " plus delta"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "content_block_start", "index": 1, "content_block": {"type": "thinking", "thinking": "Initial thinking", "signature": "initial signature"}}),
        json!({"type": "content_block_delta", "index": 1, "delta": {"type": "thinking_delta", "thinking": " plus delta"}}),
        json!({"type": "content_block_delta", "index": 1, "delta": {"type": "signature_delta", "signature": " plus delta"}}),
        json!({"type": "content_block_stop", "index": 1}),
        message_delta("end_turn"),
        json!({"type": "message_stop"}),
    ];
    let server = serve(sse(&events)).await;
    let model = model(&server.uri(), json!({}));
    let mut stream = stream(&model, &transcript(&hello()), key());
    while stream.next().await.is_some() {}
    let partial = stream.partial().unwrap();
    assert_eq!(partial.text(), "Initial text plus delta");
    let thinking = partial.content[1].as_thinking().unwrap();
    assert_eq!(thinking.thinking, "Initial thinking plus delta");
    assert_eq!(
        thinking.thinking_signature.as_deref(),
        Some("initial signature plus delta")
    );
    assert_eq!(thinking.redacted, None);
}

#[tokio::test]
async fn redacted_thinking_blocks() {
    let events = vec![
        message_start(),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "redacted_thinking", "data": "opaque"}}),
        json!({"type": "content_block_stop", "index": 0}),
        message_delta("end_turn"),
        json!({"type": "message_stop"}),
    ];
    let (_, events, result) = run_events(json!({}), &events, &hello(), key()).await;
    assert_eq!(kinds(&events), ["start", "thinking_start", "thinking_end", "done"]);
    let thinking = result.content[0].as_thinking().unwrap();
    assert_eq!(thinking.thinking, "[Reasoning redacted]");
    assert_eq!(thinking.thinking_signature.as_deref(), Some("opaque"));
    assert_eq!(thinking.redacted, Some(true));
}

fn tool_events(partial_json: &[&str]) -> Vec<Value> {
    let mut events = vec![
        message_start(),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}}),
    ];
    for json in partial_json {
        events.push(json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": json}}));
    }
    events.push(json!({"type": "content_block_stop", "index": 0}));
    events.push(message_delta("tool_use"));
    events.push(json!({"type": "message_stop"}));
    events
}

#[tokio::test]
async fn repairs_malformed_sse_json_and_tool_arguments() {
    let malformed = "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\
                     \"partial_json\":\"{\\\"path\\\":\\\"A\\H\\\",\\\"text\\\":\\\"col1\tcol2\\\"}\"}}";
    let mut events = tool_events(&[]);
    events.insert(2, raw("content_block_delta", malformed));
    let (_, events, result) = run_events(json!({}), &events, &hello(), key()).await;
    assert_eq!(
        kinds(&events),
        ["start", "toolcall_start", "toolcall_delta", "toolcall_end", "done"]
    );
    assert_eq!(result.stop_reason, StopReason::ToolUse);
    let call = result.tool_calls().next().unwrap();
    assert_eq!(
        Value::Object(call.arguments.clone()),
        json!({"path": "A\\H", "text": "col1\tcol2"})
    );
    assert_eq!(call.arguments_error, None);
}

#[tokio::test]
async fn streams_partial_tool_arguments_and_reports_invalid_ones() {
    let server = serve(sse(&tool_events(&["{\"value\": \"a", "bc\"}"]))).await;
    let model = model(&server.uri(), json!({}));
    let mut stream = stream(&model, &transcript(&hello()), key());
    let mut partial_values = Vec::new();
    while let Some(event) = stream.next().await {
        if let AssistantMessageEvent::ToolCallDelta { content_index, .. } = event {
            let call = stream.partial().unwrap().content[content_index].as_tool_call().unwrap();
            partial_values.push(Value::Object(call.arguments.clone()));
        }
    }
    assert_eq!(partial_values, [json!({"value": "a"}), json!({"value": "abc"})]);

    let (_, _, result) = run_events(json!({}), &tool_events(&["[1, 2]"]), &hello(), key()).await;
    let call = result.tool_calls().next().unwrap();
    assert!(call.arguments.is_empty());
    assert!(
        call.arguments_error
            .as_deref()
            .unwrap()
            .contains("Received arguments:\n[1, 2]")
    );
}

#[tokio::test]
async fn maps_refusal_sensitive_and_unknown_stop_reasons() {
    let explanation = "This request triggered restrictions on violative cyber content.";
    let events = vec![
        message_start(),
        json!({"type": "message_delta", "delta": {"stop_reason": "refusal",
            "stop_details": {"type": "refusal", "category": "cyber", "explanation": explanation}},
            "usage": {"input_tokens": 412, "output_tokens": 0}}),
        json!({"type": "message_stop"}),
    ];
    let (_, events, result) = run_events(json!({}), &events, &hello(), key()).await;
    assert_eq!(kinds(&events), ["start", "error"]);
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(result.raw_stop_reason.as_deref(), Some("refusal"));
    assert_eq!(result.error_message.as_deref(), Some(explanation));
    assert_eq!(result.usage.input, 412);

    for (raw_reason, expected) in [
        ("sensitive", "Provider stopped with: sensitive"),
        ("new_reason", "Unhandled stop reason: new_reason"),
    ] {
        let events = vec![
            message_start(),
            json!({"type": "message_delta", "delta": {"stop_reason": raw_reason}}),
            json!({"type": "message_stop"}),
        ];
        let (_, _, result) = run_events(json!({}), &events, &hello(), key()).await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(result.raw_stop_reason.as_deref(), Some(raw_reason));
        assert_eq!(result.error_message.as_deref(), Some(expected));
    }

    for (raw_reason, expected) in [("max_tokens", StopReason::Length), ("pause_turn", StopReason::Stop)] {
        let events = vec![
            message_start(),
            json!({"type": "message_delta", "delta": {"stop_reason": raw_reason}}),
            json!({"type": "message_stop"}),
        ];
        let (_, _, result) = run_events(json!({}), &events, &hello(), key()).await;
        assert_eq!(result.stop_reason, expected);
    }
}

#[tokio::test]
async fn message_delta_without_usage_keeps_start_usage() {
    let mut events = minimal_events();
    events[4] = json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}});
    let (_, _, result) = run_events(json!({}), &events, &hello(), key()).await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!((result.usage.input, result.usage.total_tokens), (12, 12));
}

#[tokio::test]
async fn in_stream_errors() {
    // `event: error` fails the stream with the raw data; usage parsed so far is kept.
    let data = r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
    let (_, events, result) = run_events(json!({}), &[message_start(), raw("error", data)], &hello(), key()).await;
    assert_eq!(kinds(&events), ["start", "error"]);
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(result.error_message.as_deref(), Some(data));
    assert_eq!(result.usage.input, 12);

    // An unparseable known event reports the data and raw lines.
    let (_, _, result) = run_events(
        json!({}),
        &[message_start(), raw("content_block_delta", "{nope")],
        &hello(),
        key(),
    )
    .await;
    let message = result.error_message.unwrap();
    assert!(message.starts_with("Could not parse Anthropic SSE event content_block_delta: "));
    assert!(message.ends_with("; data={nope; raw=event: content_block_delta\\ndata: {nope"));

    // A stream that ends before message_stop keeps its content.
    let events = minimal_events();
    let (_, _, result) = run_events(json!({}), &events[..5], &hello(), key()).await;
    assert_eq!(
        result.error_message.as_deref(),
        Some("Anthropic stream ended before message_stop")
    );
    assert_eq!(result.text(), "Hello");

    // No events at all: no stop reason.
    let (_, _, result) = run_events(json!({}), &[], &hello(), key()).await;
    assert_eq!(
        result.error_message.as_deref(),
        Some("Anthropic stream ended without a stop reason")
    );
}

#[tokio::test]
async fn server_side_fallback() {
    // A fallback block after output began is unsupported.
    let events = vec![
        message_start(),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "partial"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "content_block_start", "index": 1, "content_block": {"type": "fallback",
            "from": {"model": "claude-test"}, "to": {"model": "claude-other"}}}),
    ];
    let (_, _, result) = run_events(json!({}), &events, &hello(), key()).await;
    assert_eq!(result.stop_reason, StopReason::Error);
    assert!(
        result
            .error_message
            .unwrap()
            .contains("unsupported mid-output model fallback")
    );

    // A fallback block before output is ignored; the fallback model's prices apply.
    let fallback = json!({"allowedFallbackModels": [{"provider": "anthropic", "model": "fallback-model",
        "cost": {"input": 3, "output": 5, "cacheRead": 0, "cacheWrite": 0}}]});
    let events = vec![
        json!({"type": "message_start", "message": {"id": "msg_fb", "model": "fallback-model",
            "usage": {"input_tokens": 100, "output_tokens": 0}}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "fallback"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": "done"}}),
        json!({"type": "content_block_stop", "index": 1}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 100, "output_tokens": 20}}),
        json!({"type": "message_stop"}),
    ];
    let (server, _, result) = run_events(json!({"compat": fallback}), &events, &hello(), key()).await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.model, "claude-test");
    assert_eq!(result.response_model.as_deref(), Some("fallback-model"));
    assert_eq!(result.text(), "done");
    assert!((result.usage.cost.input - 0.0003).abs() < 1e-10);
    assert!((result.usage.cost.output - 0.0001).abs() < 1e-10);
    let request = &requests(&server).await[0];
    assert_eq!(
        header(request, "anthropic-beta"),
        Some("server-side-fallback-2026-07-01")
    );
    let body: Value = request.body_json().unwrap();
    assert_eq!(body["fallbacks"], json!([{"model": "fallback-model"}]));
}

#[tokio::test]
async fn prices_1h_cache_writes() {
    let start = |cache_creation: Value| {
        json!({"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 100, "output_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 1_000_000, "cache_creation": cache_creation}}})
    };
    let tail = || {
        vec![
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}),
            json!({"type": "message_stop"}),
        ]
    };
    // input 5, cacheWrite (5m) 6.25 per Mtok; 1h writes cost 2x input = 10.
    let opus = json!({"cost": {"input": 5, "output": 25, "cacheRead": 0.5, "cacheWrite": 6.25}});
    let mut events = vec![start(
        json!({"ephemeral_5m_input_tokens": 600_000, "ephemeral_1h_input_tokens": 400_000}),
    )];
    events.extend(tail());
    let (_, _, result) = run_events(opus.clone(), &events, &hello(), key()).await;
    assert_eq!(
        (result.usage.cache_write, result.usage.cache_write_1h),
        (1_000_000, Some(400_000))
    );
    assert!((result.usage.cost.cache_write - 7.75).abs() < 1e-10);

    let mut events = vec![start(Value::Null)];
    events.extend(tail());
    let (_, _, result) = run_events(opus.clone(), &events, &hello(), key()).await;
    assert_eq!(result.usage.cache_write_1h, Some(0));
    assert!((result.usage.cost.cache_write - 6.25).abs() < 1e-10);

    // Vercel AI Gateway reports cache usage only in message_delta.
    let events = vec![
        json!({"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 0, "output_tokens": 0}}}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 3,
            "output_tokens": 4, "cache_creation_input_tokens": 6535,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 6535}}}),
        json!({"type": "message_stop"}),
    ];
    let (_, _, result) = run_events(opus, &events, &hello(), key()).await;
    assert_eq!(
        (result.usage.cache_write, result.usage.cache_write_1h),
        (6535, Some(6535))
    );
    assert!((result.usage.cost.cache_write - 6535.0 * 10.0 / 1_000_000.0).abs() < 1e-10);
}

#[tokio::test]
async fn reports_input_transformations_from_the_last_event() {
    let mut events = minimal_events();
    events[0]["message"]["input_transformations"] =
        json!([{"type": "thinking_dropped", "path": "messages.1.content.0", "reason": "prefix_binding_mismatch"}]);
    events[4]["input_transformations"] = json!([{"type": "thinking_dropped", "path": "messages.3.content.0", "reason": "model_binding_mismatch", "extra": null}]);
    let (_, _, result) = run_events(json!({}), &events, &hello(), key()).await;
    let diagnostics = result.diagnostics.unwrap();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].kind, "anthropic_input_transformations");
    assert_eq!(
        Value::Object(diagnostics[0].details.clone().unwrap()),
        json!({"transformations": [{"type": "thinking_dropped", "path": "messages.3.content.0", "reason": "model_binding_mismatch"}]})
    );
}

// ---------------------------------------------------------------------------
// HTTP errors, retries, abort
// ---------------------------------------------------------------------------

fn error_body(kind: &str, message: &str) -> Value {
    json!({"type": "error", "error": {"type": kind, "message": message}})
}

#[tokio::test]
async fn http_errors_carry_status_and_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(error_body("invalid_request_error", "prompt is too long")),
        )
        .mount(&server)
        .await;
    let model = model(&server.uri(), json!({}));
    let (events, result) = collect(stream(&model, &transcript(&hello()), key())).await;
    assert_eq!(kinds(&events), ["error"]);
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(result.error_message.as_deref(), Some("400 prompt is too long"));
}

#[tokio::test]
async fn retries_only_when_max_retries_is_set() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(529)
                .set_body_json(error_body("overloaded_error", "Overloaded"))
                .insert_header("retry-after-ms", "1"),
        )
        .up_to_n_times(2)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(sse_response(sse(&minimal_events())))
        .mount(&server)
        .await;
    let model = model(&server.uri(), json!({}));

    // Upstream's retry helper defaults to no retries.
    let (_, result) = collect(stream(&model, &transcript(&hello()), key())).await;
    assert_eq!(result.error_message.as_deref(), Some("529 Overloaded"));
    assert_eq!(requests(&server).await.len(), 1);

    let options = StreamOptions {
        max_retries: Some(2),
        ..key()
    };
    let (_, result) = collect(stream(&model, &transcript(&hello()), options)).await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(requests(&server).await.len(), 3);
}

#[tokio::test]
async fn abort_before_the_request_and_during_the_response() {
    let server = serve(sse(&minimal_events())).await;
    let model = model(&server.uri(), json!({}));
    let cancel = CancellationToken::new();
    cancel.cancel();
    let options = StreamOptions {
        cancel: Some(cancel),
        ..key()
    };
    let (events, result) = collect(stream(&model, &transcript(&hello()), options)).await;
    assert_eq!(kinds(&events), ["error"]);
    assert_eq!(result.stop_reason, StopReason::Aborted);
    assert_eq!(result.error_message.as_deref(), Some("Request was aborted"));
    assert!(requests(&server).await.is_empty());

    // Cancelled while waiting for the response.
    let slow = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(sse_response(sse(&minimal_events())).set_delay(Duration::from_secs(30)))
        .mount(&slow)
        .await;
    let model = self::model(&slow.uri(), json!({}));
    let cancel = CancellationToken::new();
    let options = StreamOptions {
        cancel: Some(cancel.clone()),
        ..key()
    };
    let stream = stream(&model, &transcript(&hello()), options);
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel.cancel();
    });
    let (_, result) = tokio::time::timeout(Duration::from_secs(5), collect(stream))
        .await
        .unwrap();
    assert_eq!(result.stop_reason, StopReason::Aborted);
}

#[tokio::test]
async fn abort_mid_stream_keeps_partial_content() {
    let cancel = CancellationToken::new();
    let token = cancel.clone();
    let hook: OnProviderStreamEvent = Arc::new(move |event: &Value, _: &Model| {
        if event["type"] == "content_block_delta" {
            token.cancel();
        }
        async { Ok(()) }.boxed()
    });
    let options = StreamOptions {
        cancel: Some(cancel),
        on_provider_stream_event: Some(hook),
        ..key()
    };
    let (_, events, result) = run_events(json!({}), &minimal_events(), &hello(), options).await;
    assert_eq!(kinds(&events).last(), Some(&"error"));
    assert_eq!(result.stop_reason, StopReason::Aborted);
    assert_eq!(result.error_message.as_deref(), Some("Request was aborted"));
    assert_eq!(result.text(), "Hello");
}

#[tokio::test]
async fn missing_auth_is_an_error_event() {
    let model = model(OFFLINE, json!({"provider": "test-provider"}));
    let (events, result) = collect(stream(&model, &transcript(&hello()), StreamOptions::default())).await;
    assert_eq!(kinds(&events), ["error"]);
    assert_eq!(
        result.error_message.as_deref(),
        Some("No API key for provider: test-provider")
    );
    let blank = StreamOptions {
        headers: Some(
            [("Authorization".to_owned(), Some("  ".to_owned()))]
                .into_iter()
                .collect(),
        ),
        ..StreamOptions::default()
    };
    let (_, result) = collect(stream_simple(&model, &transcript(&hello()), blank.into())).await;
    assert_eq!(
        result.error_message.as_deref(),
        Some("No API key for provider: test-provider")
    );
}

// ---------------------------------------------------------------------------
// Auth and headers
// ---------------------------------------------------------------------------

#[tokio::test]
async fn oauth_tokens_use_bearer_auth_and_claude_code_identity() {
    let context = Context::new(vec![
        user("go", 1),
        assistant(vec![tool_call("t1", "read", &json!({}))], StopReason::ToolUse),
        tool_result("t1", "read", vec![text("ok")], false),
    ])
    .with_system_prompt("System prompt.")
    .with_tools(vec![
        empty_tool("read"),
        empty_tool("todowrite"),
        empty_tool("find"),
        empty_tool("my_custom_tool"),
    ]);
    let events = vec![
        message_start(),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_9", "name": "TodoWrite", "input": {}}}),
        json!({"type": "content_block_stop", "index": 0}),
        message_delta("tool_use"),
        json!({"type": "message_stop"}),
    ];
    let options = StreamOptions {
        api_key: Some("sk-ant-oat01-token".into()),
        ..StreamOptions::default()
    };
    let (server, _, result) = run_events(json!({}), &events, &context, options).await;
    // Tool names come back in the caller's casing.
    assert_eq!(result.tool_calls().next().unwrap().name, "todowrite");

    let request = &requests(&server).await[0];
    assert_eq!(header(request, "authorization"), Some("Bearer sk-ant-oat01-token"));
    assert_eq!(header(request, "x-api-key"), None);
    assert_eq!(header(request, "user-agent"), Some("claude-cli/2.1.280"));
    assert_eq!(header(request, "x-app"), Some("cli"));
    assert_eq!(
        header(request, "anthropic-beta"),
        Some("claude-code-20250219,oauth-2025-04-20")
    );
    let body: Value = request.body_json().unwrap();
    let system: Vec<&str> = body["system"]
        .as_array()
        .unwrap()
        .iter()
        .map(|block| block["text"].as_str().unwrap())
        .collect();
    assert_eq!(
        system,
        [
            "You are Claude Code, Anthropic's official CLI for Claude.",
            "System prompt."
        ]
    );
    assert!(
        body["system"]
            .as_array()
            .unwrap()
            .iter()
            .all(|block| block["cache_control"] == json!({"type": "ephemeral"}))
    );
    let names: Vec<&str> = body["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["Read", "TodoWrite", "find", "my_custom_tool"]);
    assert_eq!(assistant_content(&body)[0]["name"], "Read");

    // No system prompt: the identity block alone.
    let model = model(OFFLINE, json!({}));
    let options = StreamOptions {
        api_key: Some("sk-ant-oat01-test".into()),
        ..no_cache()
    };
    let (payload, _) = capture(&model, &hello(), options).await;
    assert_eq!(
        payload["system"],
        json!([{"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."}])
    );
}

#[tokio::test]
async fn header_owned_auth_and_header_overrides() {
    // Authorization header without a key: no x-api-key, no OAuth shaping.
    let options = StreamOptions {
        headers: Some(
            [("Authorization".to_owned(), Some("Bearer gateway-token".to_owned()))]
                .into_iter()
                .collect(),
        ),
        ..StreamOptions::default()
    };
    let context = hello().with_system_prompt("System prompt.");
    let (server, _, result) = run_events(json!({}), &minimal_events(), &context, options).await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    let request = &requests(&server).await[0];
    assert_eq!(header(request, "authorization"), Some("Bearer gateway-token"));
    assert_eq!(header(request, "x-api-key"), None);
    assert_eq!(header(request, "anthropic-beta"), None);
    let body: Value = request.body_json().unwrap();
    assert_eq!(body["system"][0]["text"], "System prompt.");

    // `None` headers remove the SDK's auth headers (Cloudflare AI Gateway shape).
    let options = StreamOptions {
        api_key: Some("ignored-key".into()),
        headers: Some(
            [
                ("cf-aig-authorization".to_owned(), Some("Bearer cf".to_owned())),
                ("Authorization".to_owned(), None),
                ("x-api-key".to_owned(), None),
            ]
            .into_iter()
            .collect(),
        ),
        ..StreamOptions::default()
    };
    let (server, _, _) = run_events(json!({}), &minimal_events(), &hello(), options).await;
    let request = &requests(&server).await[0];
    assert_eq!(header(request, "cf-aig-authorization"), Some("Bearer cf"));
    assert_eq!(header(request, "authorization"), None);
    assert_eq!(header(request, "x-api-key"), None);

    // Explicit headers override defaults; an anthropic-beta header replaces the betas.
    let options = StreamOptions {
        headers: Some(
            [
                ("User-Agent".to_owned(), Some("custom-client".to_owned())),
                (
                    "anthropic-beta".to_owned(),
                    Some("custom-beta, custom-beta,other".to_owned()),
                ),
                ("anthropic-dangerous-direct-browser-access".to_owned(), None),
            ]
            .into_iter()
            .collect(),
        ),
        ..key()
    };
    let (server, _, _) = run_events(
        json!({"headers": {"X-Model": "m"}}),
        &minimal_events(),
        &hello(),
        options,
    )
    .await;
    let request = &requests(&server).await[0];
    assert_eq!(header(request, "user-agent"), Some("custom-client"));
    assert_eq!(header(request, "anthropic-beta"), Some("custom-beta,other"));
    assert_eq!(header(request, "x-model"), Some("m"));
    assert_eq!(header(request, "anthropic-dangerous-direct-browser-access"), None);

    // A null anthropic-beta header suppresses every beta.
    let options = StreamOptions {
        headers: Some([("anthropic-beta".to_owned(), None)].into_iter().collect()),
        ..key()
    };
    let context = hello().with_tools(vec![empty_tool("lookup")]);
    let (server, _, _) = run_events(
        json!({"compat": {"supportsEagerToolInputStreaming": false}}),
        &minimal_events(),
        &context,
        options,
    )
    .await;
    assert_eq!(header(&requests(&server).await[0], "anthropic-beta"), None);
}

#[tokio::test]
async fn session_affinity_headers() {
    let options = || StreamOptions {
        session_id: Some("session-1".into()),
        ..key()
    };
    let (server, _, _) = run_events(
        json!({"provider": "openrouter"}),
        &minimal_events(),
        &hello(),
        options(),
    )
    .await;
    assert_eq!(header(&requests(&server).await[0], "x-session-id"), Some("session-1"));

    let none = StreamOptions {
        cache_retention: Some(tau_ai::CacheRetention::None),
        ..options()
    };
    let (server, _, _) = run_events(json!({"provider": "openrouter"}), &minimal_events(), &hello(), none).await;
    assert_eq!(header(&requests(&server).await[0], "x-session-id"), None);

    let compat = json!({"compat": {"sendSessionAffinityHeaders": true}});
    let (server, _, _) = run_events(compat, &minimal_events(), &hello(), options()).await;
    assert_eq!(
        header(&requests(&server).await[0], "x-session-affinity"),
        Some("session-1")
    );

    let (server, _, _) = run_events(json!({}), &minimal_events(), &hello(), options()).await;
    assert_eq!(header(&requests(&server).await[0], "x-session-affinity"), None);
}

#[tokio::test]
async fn on_payload_replacement_is_still_streamed_and_body_fields_become_headers() {
    let hook: tau_ai::options::OnPayload = Arc::new(|payload: &Value, _: &Model| {
        let mut replaced = payload.clone();
        replaced["stream"] = json!(false);
        replaced["inference_geo"] = json!("us");
        replaced["user_profile_id"] = json!("prof_1");
        async move { Ok(Some(replaced)) }.boxed()
    });
    let options = StreamOptions {
        on_payload: Some(hook),
        metadata: Some(json!({"user_id": "u"}).as_object().cloned().unwrap()),
        ..key()
    };
    let (server, _, result) = run_events(json!({}), &minimal_events(), &hello(), options).await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    let request = &requests(&server).await[0];
    let body: Value = request.body_json().unwrap();
    assert_eq!(body["stream"], true);
    assert_eq!(body["inference_geo"], "us");
    assert_eq!(body["metadata"], json!({"user_id": "u"}));
    assert!(body.get("user_profile_id").is_none());
    assert_eq!(header(request, "anthropic-user-profile-id"), Some("prof_1"));
}

// ---------------------------------------------------------------------------
// Strict tools
// ---------------------------------------------------------------------------

fn strict_tool(name: &str, mode: StrictMode) -> Tool {
    Tool {
        constrained_sampling: Some(ConstrainedSampling::JsonSchema { strict: mode }),
        ..Tool::new(
            name,
            "Look up a value",
            json!({"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}),
        )
    }
}

const GRAMMAR_ERROR: &str = "The compiled grammar is too large, which would cause performance issues.";

#[tokio::test]
async fn strict_tool_limit_rejection_retries_with_non_strict_preferred_tools() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(400).set_body_json(error_body("invalid_request_error", GRAMMAR_ERROR)))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(sse_response(sse(&minimal_events())))
        .mount(&server)
        .await;
    let model = model(&server.uri(), json!({"compat": {"supportsStrictTools": true}}));
    let context = hello().with_tools(vec![
        strict_tool("preferred", StrictMode::Prefer),
        strict_tool("required", StrictMode::Require),
    ]);
    let responses = Arc::new(Mutex::new(0));
    let counter = responses.clone();
    let options = StreamOptions {
        on_response: Some(Arc::new(move |_: &tau_ai::options::ProviderResponse, _: &Model| {
            *counter.lock().unwrap() += 1;
            async { Ok(()) }.boxed()
        })),
        ..key()
    };
    let (events, result) = collect(stream(&model, &transcript(&context), options)).await;
    assert_eq!(
        kinds(&events),
        ["start", "text_start", "text_delta", "text_end", "done"]
    );
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(*responses.lock().unwrap(), 1);
    let requests = requests(&server).await;
    assert_eq!(requests.len(), 2);
    let first: Value = requests[0].body_json().unwrap();
    let second: Value = requests[1].body_json().unwrap();
    assert_eq!(first["tools"][0]["strict"], true);
    assert_eq!(first["tools"][0]["input_schema"]["additionalProperties"], false);
    assert!(second["tools"][0].get("strict").is_none());
    assert_eq!(
        second["tools"][0]["input_schema"],
        json!({"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]})
    );
    // Required tools stay strict.
    assert_eq!(second["tools"][1], first["tools"][1]);
    let diagnostics = result.diagnostics.unwrap();
    assert_eq!(diagnostics[0].kind, "anthropic_strict_tools_fallback");
    assert!(diagnostics[0].error.as_ref().unwrap().message.contains(GRAMMAR_ERROR));
    assert_eq!(
        Value::Object(diagnostics[0].details.clone().unwrap()),
        json!({"reason": "strict_tool_limits_exceeded"})
    );
}

#[tokio::test]
async fn strict_tool_fallback_only_for_matching_rejections() {
    for (tools, status, message) in [
        // Only required or ordinary tools: nothing to downgrade.
        (vec![strict_tool("lookup", StrictMode::Require)], 400, GRAMMAR_ERROR),
        (vec![empty_tool("lookup")], 400, GRAMMAR_ERROR),
        (
            vec![strict_tool("lookup", StrictMode::Prefer)],
            400,
            "Invalid tool schema",
        ),
        (vec![strict_tool("lookup", StrictMode::Prefer)], 500, GRAMMAR_ERROR),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(status).set_body_json(error_body("invalid_request_error", message)))
            .mount(&server)
            .await;
        let model = model(&server.uri(), json!({"compat": {"supportsStrictTools": true}}));
        let context = hello().with_tools(tools);
        let (_, result) = collect(stream(&model, &transcript(&context), key())).await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert!(result.error_message.unwrap().contains(message));
        assert_eq!(requests(&server).await.len(), 1);
    }

    // A second rejection is surfaced rather than retried again.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(400).set_body_json(error_body("invalid_request_error", GRAMMAR_ERROR)))
        .mount(&server)
        .await;
    let model = model(&server.uri(), json!({"compat": {"supportsStrictTools": true}}));
    let context = hello().with_tools(vec![strict_tool("lookup", StrictMode::Prefer)]);
    let (_, result) = collect(stream(&model, &transcript(&context), key())).await;
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(requests(&server).await.len(), 2);
}

#[tokio::test]
async fn required_strict_tool_without_support_fails() {
    let model = model(OFFLINE, json!({}));
    let context = hello().with_tools(vec![strict_tool("lookup", StrictMode::Require)]);
    let (events, result) = collect(stream(&model, &transcript(&context), key())).await;
    assert_eq!(kinds(&events), ["error"]);
    assert!(
        result
            .error_message
            .unwrap()
            .contains("requires JSON-schema constrained sampling")
    );
}

// ---------------------------------------------------------------------------
// Request payloads
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cache_control_follows_retention() {
    let model = model(OFFLINE, json!({}));
    let context = hello().with_system_prompt("You are a helpful assistant.");
    let (payload, _) = capture(&model, &context, StreamOptions::default()).await;
    assert_eq!(payload["system"][0]["cache_control"], json!({"type": "ephemeral"}));
    assert_eq!(
        payload["messages"],
        json!([{"role": "user", "content": [{"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}]}])
    );

    let long = StreamOptions {
        cache_retention: Some(tau_ai::CacheRetention::Long),
        ..StreamOptions::default()
    };
    let (payload, _) = capture(&model, &context, long.clone()).await;
    let one_hour = json!({"type": "ephemeral", "ttl": "1h"});
    assert_eq!(payload["system"][0]["cache_control"], one_hour);
    assert_eq!(payload["messages"][0]["content"][0]["cache_control"], one_hour);

    let env = StreamOptions {
        env: Some(
            [("PI_CACHE_RETENTION".to_owned(), "long".to_owned())]
                .into_iter()
                .collect(),
        ),
        ..StreamOptions::default()
    };
    let (payload, _) = capture(&model, &context, env).await;
    assert_eq!(payload["system"][0]["cache_control"], one_hour);

    let no_long = self::model(OFFLINE, json!({"compat": {"supportsLongCacheRetention": false}}));
    let (payload, _) = capture(&no_long, &context, long).await;
    assert_eq!(payload["system"][0]["cache_control"], json!({"type": "ephemeral"}));

    let (payload, _) = capture(&model, &context, no_cache()).await;
    assert!(payload["system"][0].get("cache_control").is_none());
    assert_eq!(payload["messages"], json!([{"role": "user", "content": "Hello"}]));
}

#[tokio::test]
async fn cache_control_on_last_tool_and_tool_result() {
    let context = Context::new(vec![
        user("go", 1),
        assistant(vec![tool_call("t1", "a", &json!({}))], StopReason::ToolUse),
        tool_result("t1", "a", vec![text("done")], false),
    ])
    .with_tools(vec![empty_tool("a"), empty_tool("b")]);
    let (payload, _) = capture(&model(OFFLINE, json!({})), &context, StreamOptions::default()).await;
    assert!(payload["tools"][0].get("cache_control").is_none());
    assert_eq!(payload["tools"][1]["cache_control"], json!({"type": "ephemeral"}));
    assert_eq!(
        payload["tools"][0],
        json!({"name": "a", "description": "a tool", "eager_input_streaming": true,
               "input_schema": {"type": "object", "properties": {}, "required": []}})
    );
    assert_eq!(
        payload["messages"][2],
        json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "done",
            "is_error": false, "cache_control": {"type": "ephemeral"}}]})
    );

    // No cache control on tools when the endpoint rejects it.
    let model = model(OFFLINE, json!({"compat": {"supportsCacheControlOnTools": false}}));
    let (payload, _) = capture(&model, &context, StreamOptions::default()).await;
    assert!(payload["tools"][1].get("cache_control").is_none());
}

#[tokio::test]
async fn replays_messages_for_anthropic() {
    let model = model(OFFLINE, json!({"reasoning": true, "input": ["text", "image"]}));
    let context = Context::new(vec![
        user("   ", 0),
        Message::User(UserMessage {
            content: UserContent::Blocks(vec![
                text(" "),
                UserBlock::Image(ImageContent {
                    data: "IMG".into(),
                    mime_type: "image/jpeg".into(),
                }),
                text("look"),
            ]),
            timestamp: 1,
        }),
        assistant(
            vec![
                AssistantBlock::Thinking(ThinkingContent {
                    thinking: "[Reasoning redacted]".into(),
                    thinking_signature: Some("opaque".into()),
                    redacted: Some(true),
                }),
                AssistantBlock::Thinking(ThinkingContent::new("  ")),
                AssistantBlock::Text(TextContent::new("   ")),
                AssistantBlock::Text(TextContent::new("answer")),
                tool_call("toolu_1", "lookup", &json!({"value": "x"})),
            ],
            StopReason::ToolUse,
        ),
        tool_result("toolu_1", "lookup", vec![text("result")], false),
    ]);
    let (payload, _) = capture(&model, &context, no_cache()).await;
    assert_eq!(
        payload["messages"],
        json!([
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "IMG"}},
                {"type": "text", "text": "look"}
            ]},
            {"role": "assistant", "content": [
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "text", "text": "answer"},
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"value": "x"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "result", "is_error": false}
            ]}
        ])
    );
}

#[tokio::test]
async fn cross_provider_replay_normalizes_ids_and_fills_orphans() {
    let mut foreign = AssistantMessage::empty("openai-responses", "openai", "gpt-5");
    foreign.stop_reason = StopReason::ToolUse;
    foreign.content = vec![
        AssistantBlock::Thinking(ThinkingContent {
            thinking: "foreign reasoning".into(),
            thinking_signature: Some("rs_123".into()),
            redacted: None,
        }),
        AssistantBlock::Text(TextContent {
            text: "calling".into(),
            text_signature: Some("msg_1".into()),
        }),
        tool_call("call_1|fc/abc", "lookup", &json!({})),
        tool_call("call_2|fc/def", "lookup", &json!({})),
    ];
    let context = Context::new(vec![
        user("first", 1),
        Message::Assistant(foreign),
        tool_result("call_1|fc/abc", "lookup", vec![text("ok")], false),
        user("next", 3),
    ]);
    let (payload, _) = capture(&model(OFFLINE, json!({})), &context, no_cache()).await;
    assert_eq!(
        *assistant_content(&payload),
        json!([
            {"type": "text", "text": "calling"},
            {"type": "tool_use", "id": "call_1_fc_abc", "name": "lookup", "input": {}},
            {"type": "tool_use", "id": "call_2_fc_def", "name": "lookup", "input": {}}
        ])
    );
    assert_eq!(
        payload["messages"][2],
        json!({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_1_fc_abc", "content": "ok", "is_error": false},
            {"type": "tool_result", "tool_use_id": "call_2_fc_def", "content": "No result provided", "is_error": true}
        ]})
    );
    assert_eq!(payload["messages"][3], json!({"role": "user", "content": "next"}));
}

#[tokio::test]
async fn moves_error_result_images_after_the_tool_results() {
    let model = model(OFFLINE, json!({"input": ["text", "image"]}));
    let context = Context::new(vec![
        user("Show the screenshots", 1),
        assistant(
            vec![
                tool_call("call-1", "codemode", &json!({})),
                tool_call("call-2", "read", &json!({})),
            ],
            StopReason::ToolUse,
        ),
        tool_result(
            "call-1",
            "codemode",
            vec![
                text("Script failed"),
                image("AAAA"),
                image("CCCC"),
                text("Script error"),
            ],
            true,
        ),
        tool_result("call-2", "read", vec![image("BBBB")], false),
    ]);
    let image_block =
        |data: &str| json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}});
    let (payload, _) = capture(&model, &context, no_cache()).await;
    assert_eq!(
        payload["messages"][2],
        json!({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call-1",
             "content": "Script failed\n(image attached after the tool results)\nScript error", "is_error": true},
            {"type": "tool_result", "tool_use_id": "call-2",
             "content": [{"type": "text", "text": "(see attached image)"}, image_block("BBBB")], "is_error": false},
            {"type": "text", "text": "Images from the error result of tool call call-1 (codemode):"},
            image_block("AAAA"),
            image_block("CCCC")
        ]})
    );

    // Images become placeholders for text-only models.
    let text_only = self::model(OFFLINE, json!({}));
    let (payload, _) = capture(&text_only, &context, no_cache()).await;
    assert_eq!(
        payload["messages"][2]["content"][1]["content"],
        "(tool image omitted: model does not support images)"
    );
}

#[tokio::test]
async fn empty_signature_thinking_compat() {
    let context = |signature: &str, thinking: &str| {
        Context::new(vec![
            user("first", 1),
            assistant(
                vec![AssistantBlock::Thinking(ThinkingContent {
                    thinking: thinking.into(),
                    thinking_signature: Some(signature.into()),
                    redacted: None,
                })],
                StopReason::Stop,
            ),
            user("second", 2),
        ])
    };
    let plain = model(OFFLINE, json!({"reasoning": true}));
    let (payload, _) = capture(&plain, &context("", "internal reasoning"), no_cache()).await;
    assert_eq!(
        *assistant_content(&payload),
        json!([{"type": "text", "text": "internal reasoning"}])
    );
    let (payload, _) = capture(&plain, &context("signed-thinking", ""), no_cache()).await;
    assert_eq!(
        *assistant_content(&payload),
        json!([{"type": "thinking", "thinking": "", "signature": "signed-thinking"}])
    );
    let allow = model(
        OFFLINE,
        json!({"reasoning": true, "compat": {"allowEmptySignature": true}}),
    );
    let (payload, _) = capture(&allow, &context(" ", "internal reasoning"), no_cache()).await;
    assert_eq!(
        *assistant_content(&payload),
        json!([{"type": "thinking", "thinking": "internal reasoning", "signature": ""}])
    );
}

#[tokio::test]
async fn thinking_payloads() {
    let vendor = |compat: Value| {
        model(
            OFFLINE,
            json!({"id": "vendor--claude-opus-latest", "provider": "vendor-proxy", "reasoning": true,
                   "maxTokens": 32000, "compat": compat}),
        )
    };
    // Budget-based by default; max_tokens grows by the budget.
    let options = SimpleStreamOptions {
        reasoning: Some(ThinkingLevel::High),
        base: StreamOptions {
            max_tokens: Some(4000),
            temperature: Some(0.5),
            ..StreamOptions::default()
        },
        ..SimpleStreamOptions::default()
    };
    let (payload, _) = capture_simple(&vendor(Value::Null), &hello(), options).await;
    assert_eq!(payload["max_tokens"], 20384);
    assert_eq!(
        payload["thinking"],
        json!({"type": "enabled", "budget_tokens": 16384, "display": "summarized"})
    );
    assert_eq!(payload["betas"], json!(["interleaved-thinking-2025-05-14"]));
    assert!(payload.get("temperature").is_none());
    assert!(payload.get("output_config").is_none());

    // The budget leaves room for the answer.
    let small = model(
        OFFLINE,
        json!({"id": "vendor--small", "provider": "vendor-proxy", "reasoning": true, "maxTokens": 4096}),
    );
    let (payload, _) = capture_simple(&small, &hello(), simple(Some(ThinkingLevel::High))).await;
    assert_eq!(payload["max_tokens"], 4096);
    assert_eq!(payload["thinking"]["budget_tokens"], 4096 - 1024);

    // Adaptive thinking with the mapped effort.
    let adaptive = vendor(json!({"forceAdaptiveThinking": true}));
    let (payload, _) = capture_simple(&adaptive, &hello(), simple(Some(ThinkingLevel::Medium))).await;
    assert_eq!(
        payload["thinking"],
        json!({"type": "adaptive", "display": "summarized"})
    );
    assert_eq!(payload["output_config"], json!({"effort": "medium"}));
    assert!(payload.get("betas").is_none());
    let (payload, _) = capture_simple(&adaptive, &hello(), simple(Some(ThinkingLevel::Xhigh))).await;
    assert_eq!(payload["output_config"], json!({"effort": "high"}));
    let mapped = model(
        OFFLINE,
        json!({"reasoning": true, "thinkingLevelMap": {"xhigh": "xhigh", "max": "max"},
               "compat": {"forceAdaptiveThinking": true}}),
    );
    let (payload, _) = capture_simple(&mapped, &hello(), simple(Some(ThinkingLevel::Xhigh))).await;
    assert_eq!(payload["output_config"], json!({"effort": "xhigh"}));

    // Thinking off: disabled, unless the model does not support off.
    let (payload, _) = capture_simple(&adaptive, &hello(), simple(None)).await;
    assert_eq!(payload["thinking"], json!({"type": "disabled"}));
    assert!(payload.get("output_config").is_none());
    let no_off = model(OFFLINE, json!({"reasoning": true, "thinkingLevelMap": {"off": null}}));
    let (payload, _) = capture_simple(&no_off, &hello(), simple(None)).await;
    assert!(payload.get("thinking").is_none());
    let non_reasoning = model(OFFLINE, json!({}));
    let (payload, _) = capture_simple(&non_reasoning, &hello(), simple(None)).await;
    assert!(payload.get("thinking").is_none());

    // Direct options: display, interleaved opt-out, adaptive without effort.
    let options = StreamOptions {
        api: api(AnthropicOptions {
            thinking_enabled: Some(true),
            thinking_display: Some(ThinkingDisplay::Omitted),
            interleaved_thinking: Some(false),
            ..AnthropicOptions::default()
        }),
        ..StreamOptions::default()
    };
    let (payload, _) = capture(&vendor(Value::Null), &hello(), options).await;
    assert_eq!(
        payload["thinking"],
        json!({"type": "enabled", "budget_tokens": 1024, "display": "omitted"})
    );
    assert!(payload.get("betas").is_none());
    let options = StreamOptions {
        api: api(AnthropicOptions {
            thinking_enabled: Some(true),
            ..AnthropicOptions::default()
        }),
        ..StreamOptions::default()
    };
    let (payload, _) = capture(&adaptive, &hello(), options).await;
    assert_eq!(
        payload["thinking"],
        json!({"type": "adaptive", "display": "summarized"})
    );
    assert!(payload.get("output_config").is_none());
}

fn managed_model(level_map: Value) -> Model {
    model(
        OFFLINE,
        json!({"id": "claude-fable-5-1", "reasoning": true, "maxTokens": 32000, "thinkingLevelMap": level_map,
               "compat": {"forceAdaptiveThinking": true, "supportsMidConvoEffort": true}}),
    )
}

fn managed_assistant(level: Option<&str>, provider: &str) -> Message {
    let mut message = AssistantMessage::empty("anthropic-messages", provider, "claude-fable-5-1");
    message.content = vec![
        AssistantBlock::Thinking(ThinkingContent {
            thinking: "reasoning".into(),
            thinking_signature: Some("signature".into()),
            redacted: None,
        }),
        AssistantBlock::Text(TextContent::new("answer")),
    ];
    message.stop_reason = StopReason::Stop;
    message.provider_thinking_level = level.map(str::to_owned);
    Message::Assistant(message)
}

fn effort_options(effort: Option<AnthropicEffort>, thinking_enabled: bool) -> StreamOptions {
    StreamOptions {
        api: api(AnthropicOptions {
            thinking_enabled: Some(thinking_enabled),
            effort,
            ..AnthropicOptions::default()
        }),
        temperature: Some(0.2),
        ..no_cache()
    }
}

#[tokio::test]
async fn mid_conversation_effort_markers() {
    let level_map =
        json!({"off": null, "minimal": "low", "low": "low", "medium": "medium", "high": "high", "max": "max"});
    let model = managed_model(level_map);
    let first_context = Context::new(vec![user("one", 1)]);
    let (first, first_message) =
        capture(&model, &first_context, effort_options(Some(AnthropicEffort::Low), true)).await;
    assert_eq!(
        first["messages"],
        json!([{"role": "user", "content": "one"}, {"role": "system", "content": [], "output_config": {"effort": "low"}}])
    );
    assert_eq!(first_message.provider_thinking_level.as_deref(), Some("low"));
    assert_eq!(first["output_config"], json!({"effort": "high"}));
    assert_eq!(
        first["thinking"],
        json!({"type": "adaptive", "display": "summarized", "block_binding": {"prefix_mismatch_behavior": "drop_block"}})
    );
    assert!(first.get("temperature").is_none());
    let betas = first["betas"].as_array().unwrap();
    assert!(betas.contains(&json!("mid-conversation-output-config-2026-07-01")));
    assert!(betas.contains(&json!("thinking-binding-controls-2026-08-01")));

    // A later request reproduces the earlier marker prefix exactly.
    let second_context = Context::new(vec![
        user("one", 1),
        managed_assistant(Some("low"), "anthropic"),
        user("two", 2),
        managed_assistant(None, "anthropic"),
        user("three", 3),
        managed_assistant(Some("low"), "other-provider"),
        user("four", 4),
    ]);
    let (second, _) = capture(&model, &second_context, effort_options(None, true)).await;
    let messages = second["messages"].as_array().unwrap();
    assert_eq!(messages[..2], first["messages"].as_array().unwrap()[..]);
    let markers: Vec<&Value> = messages.iter().filter(|message| message["role"] == "system").collect();
    assert_eq!(markers.len(), 2);
    assert_eq!(
        messages.last().unwrap(),
        &json!({"role": "system", "content": [], "output_config": {"effort": "high"}})
    );

    // Thinking off where off is supported: plain disabled thinking, no markers.
    let with_off =
        managed_model(json!({"minimal": null, "low": "low", "medium": "medium", "high": "high", "max": "max"}));
    let (payload, message) = capture(&with_off, &second_context, effort_options(None, false)).await;
    assert_eq!(payload["thinking"], json!({"type": "disabled"}));
    assert!(payload.get("output_config").is_none());
    assert!(
        payload["messages"]
            .as_array()
            .unwrap()
            .iter()
            .all(|m| m["role"] != "system")
    );
    assert_eq!(message.provider_thinking_level, None);

    // Off unsupported: adaptive thinking with markers stays.
    let (payload, message) = capture(&model, &first_context, effort_options(None, false)).await;
    assert_eq!(payload["thinking"]["type"], "adaptive");
    assert_eq!(message.provider_thinking_level.as_deref(), Some("high"));
}

fn tool_change_context() -> Context {
    Context::new(vec![
        Message::System(SystemMessage {
            content: "base prompt".into(),
            sections: Some(
                [
                    ("rules".to_owned(), Some("<rules>\nold rules\n</rules>".to_owned())),
                    ("docs".to_owned(), Some("<docs>\nread docs\n</docs>".to_owned())),
                ]
                .into_iter()
                .collect(),
            ),
            tools_added: Some(vec![empty_tool("base_tool")]),
            ..SystemMessage::default()
        }),
        user("before", 1),
        Message::System(SystemMessage {
            content: "updated guidance".into(),
            sections: Some(
                [
                    ("rules".to_owned(), Some("<rules>\nnew rules\n</rules>".to_owned())),
                    ("docs".to_owned(), None),
                ]
                .into_iter()
                .collect(),
            ),
            tools_removed: Some(vec![ToolReference {
                name: "base_tool".into(),
            }]),
            tools_added: Some(vec![empty_tool("late_tool")]),
            timestamp: 2,
        }),
    ])
}

fn native_model(compat: Value) -> Model {
    model(
        OFFLINE,
        json!({"id": "claude-opus-5", "reasoning": true, "compat": compat}),
    )
}

#[tokio::test]
async fn native_tool_changes_use_inline_tool_blocks() {
    let model = native_model(json!({"supportsMidConvoSystemMessages": true, "supportsMidConvoToolChanges": true}));
    let (payload, _) = capture_simple(&model, &tool_change_context(), simple(None)).await;
    assert!(
        payload["betas"]
            .as_array()
            .unwrap()
            .contains(&json!("inline-tools-2026-09-15"))
    );
    assert_eq!(
        payload["system"][0]["text"],
        "base prompt\n\n<rules>\nold rules\n</rules>\n\n<docs>\nread docs\n</docs>"
    );
    let tools = payload["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0]["name"], "base_tool");
    assert_eq!(tools[0]["cache_control"], json!({"type": "ephemeral"}));
    assert_eq!(
        tools[1],
        json!({"name": "__pi_deferred_placeholder__", "description": "Reserved placeholder. Never available. Never call this.",
               "input_schema": {"type": "object", "properties": {}, "required": []}, "defer_loading": true})
    );
    let update = payload["messages"].as_array().unwrap().last().unwrap();
    assert_eq!(update["role"], "system");
    let content = update["content"].as_array().unwrap();
    let types: Vec<&str> = content.iter().map(|block| block["type"].as_str().unwrap()).collect();
    assert_eq!(types, ["text", "tool_removal", "tool_addition"]);
    assert_eq!(
        content[0]["text"],
        "updated guidance\n\nUpdated system prompt section \"rules\":\n\n<rules>\nnew rules\n</rules>\n\nRemoved system prompt section \"docs\"."
    );
    assert_eq!(
        content[1]["tool"],
        json!({"type": "tool_reference", "name": "base_tool"})
    );
    assert_eq!(content[2]["tool"]["definition"]["name"], "late_tool");
    assert!(content[2]["tool"]["definition"].get("cache_control").is_none());
    assert_eq!(content[2]["cache_control"], json!({"type": "ephemeral"}));

    // A same-name redefinition is an addition only.
    let redefined = Context::new(vec![
        Message::System(SystemMessage {
            content: "base prompt".into(),
            tools_added: Some(vec![empty_tool("base_tool")]),
            ..SystemMessage::default()
        }),
        user("before", 1),
        Message::System(SystemMessage {
            content: "".into(),
            tools_removed: Some(vec![ToolReference {
                name: "base_tool".into(),
            }]),
            tools_added: Some(vec![Tool::new(
                "base_tool",
                "changed",
                json!({"type": "object", "properties": {}}),
            )]),
            timestamp: 2,
            ..SystemMessage::default()
        }),
    ]);
    let (payload, _) = capture_simple(&model, &redefined, simple(None)).await;
    assert_eq!(payload["tools"][0]["description"], "base_tool tool");
    let last = payload["messages"].as_array().unwrap().last().unwrap();
    assert_eq!(last["content"].as_array().unwrap().len(), 1);
    assert_eq!(last["content"][0]["tool"]["definition"]["description"], "changed");

    // Without both capabilities, updates fold into the system prompt.
    let partial = native_model(json!({"supportsMidConvoToolChanges": true}));
    let (payload, _) = capture_simple(&partial, &tool_change_context(), simple(None)).await;
    assert!(payload.get("betas").is_none());
    assert_eq!(
        payload["system"][0]["text"],
        "base prompt\n\nupdated guidance\n\n<rules>\nnew rules\n</rules>"
    );
    assert_eq!(payload["tools"][0]["name"], "late_tool");
    assert_eq!(payload["messages"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn mid_conversation_system_messages_go_before_the_next_assistant() {
    let model = native_model(json!({"supportsMidConvoSystemMessages": true}));
    let system = |text: &str, timestamp: i64| {
        Message::System(SystemMessage {
            content: text.into(),
            timestamp,
            ..SystemMessage::default()
        })
    };
    let context = Context::new(vec![
        system("base", 0),
        user("one", 1),
        assistant(vec![tool_call("t1", "lookup", &json!({}))], StopReason::ToolUse),
        system("between tool call and result", 2),
        tool_result("t1", "lookup", vec![text("ok")], false),
        system("later", 3),
        user("two", 4),
        assistant(vec![AssistantBlock::Text(TextContent::new("answer"))], StopReason::Stop),
    ]);
    let (payload, _) = capture(&model, &context, no_cache()).await;
    let roles: Vec<&str> = payload["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|message| message["role"].as_str().unwrap())
        .collect();
    assert_eq!(
        roles,
        ["user", "assistant", "user", "user", "system", "system", "assistant"]
    );
    assert_eq!(
        payload["messages"][4]["content"][0]["text"],
        "between tool call and result"
    );
    assert_eq!(payload["messages"][5]["content"][0]["text"], "later");
}

#[tokio::test]
async fn request_fields() {
    let model = model(OFFLINE, json!({}));
    let context = hello().with_tools(vec![empty_tool("lookup")]);
    let options = StreamOptions {
        metadata: Some(json!({"user_id": "u-1", "other": 1}).as_object().cloned().unwrap()),
        max_tokens: Some(77),
        temperature: Some(0.0),
        api: api(AnthropicOptions {
            tool_choice: Some(AnthropicToolChoice::Tool { name: "lookup".into() }),
            ..AnthropicOptions::default()
        }),
        ..StreamOptions::default()
    };
    let (payload, _) = capture(&model, &context, options).await;
    assert_eq!(payload["metadata"], json!({"user_id": "u-1"}));
    assert_eq!(payload["tool_choice"], json!({"type": "tool", "name": "lookup"}));
    assert_eq!(payload["max_tokens"], 77);
    assert_eq!(payload["temperature"], 0.0);
    let keys: Vec<&str> = payload
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .take(4)
        .collect();
    assert_eq!(keys, ["model", "messages", "max_tokens", "stream"]);

    // Simple tool choice; non-string user ids are ignored.
    let options = SimpleStreamOptions {
        tool_choice: Some(ToolChoice::None),
        base: StreamOptions {
            metadata: Some(json!({"user_id": 5}).as_object().cloned().unwrap()),
            ..StreamOptions::default()
        },
        ..SimpleStreamOptions::default()
    };
    let (payload, _) = capture_simple(&model, &hello(), options).await;
    assert_eq!(payload["tool_choice"], json!({"type": "none"}));
    assert_eq!(payload["max_tokens"], 1024);
    assert!(payload.get("metadata").is_none());

    // Temperature compat.
    let no_temperature = self::model(OFFLINE, json!({"compat": {"supportsTemperature": false}}));
    let options = StreamOptions {
        temperature: Some(0.0),
        ..StreamOptions::default()
    };
    let (payload, _) = capture(&no_temperature, &hello(), options).await;
    assert!(payload.get("temperature").is_none());

    // Eager tool input streaming compat falls back to the fine-grained streaming beta.
    let legacy = self::model(OFFLINE, json!({"compat": {"supportsEagerToolInputStreaming": false}}));
    let (payload, _) = capture(&legacy, &context, StreamOptions::default()).await;
    assert!(payload["tools"][0].get("eager_input_streaming").is_none());
    assert_eq!(payload["betas"], json!(["fine-grained-tool-streaming-2025-05-14"]));
    let (payload, _) = capture(&legacy, &hello(), StreamOptions::default()).await;
    assert!(payload.get("betas").is_none());
}
