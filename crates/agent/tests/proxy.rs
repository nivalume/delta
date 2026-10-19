//! Tests of the proxy stream function, ported from upstream `test/proxy.test.ts`.

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

use futures::StreamExt;
use serde_json::{Value, json};
use std::fmt::Write as _;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use delta_agent::{ProxyStreamOptions, stream_proxy};
use delta_ai::{
    AssistantBlock, AssistantMessage, AssistantMessageEvent, Context, Model, SimpleStreamOptions, StopReason,
    normalize_context,
};

fn model() -> Model {
    serde_json::from_value(json!({
        "id": "gpt-5.4", "name": "GPT-5.4", "api": "openai-responses", "provider": "openai",
        "baseUrl": "https://api.openai.com/v1", "reasoning": true, "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 400_000, "maxTokens": 128_000
    }))
    .unwrap()
}

fn usage() -> Value {
    json!({
        "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}
    })
}

fn sse(events: &[Value]) -> String {
    events.iter().fold(String::new(), |mut body, event| {
        let _ = write!(body, "data: {event}\n\n");
        body
    })
}

/// Serve `body` from a proxy server and stream through it.
async fn stream_body(body: String) -> (Vec<AssistantMessageEvent>, AssistantMessage) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/stream"))
        .and(header("authorization", "Bearer test-token"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;
    let mut stream = stream_proxy(
        &model(),
        &normalize_context(&Context::new(Vec::new()).with_system_prompt("")),
        ProxyStreamOptions {
            options: SimpleStreamOptions::default(),
            auth_token: "test-token".into(),
            proxy_url: server.uri(),
        },
    );
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    (events, stream.result().await)
}

fn kinds(events: &[AssistantMessageEvent]) -> Vec<&'static str> {
    events.iter().map(AssistantMessageEvent::kind).collect()
}

#[tokio::test]
async fn preserves_tool_call_metadata_received_only_on_toolcall_end() {
    let body = sse(&[
        json!({"type": "start"}),
        json!({"type": "toolcall_start", "contentIndex": 0, "id": "call_test|fc_test", "toolName": "lookup"}),
        json!({"type": "toolcall_delta", "contentIndex": 0, "delta": "{\"value\":\"hello\"}"}),
        json!({
            "type": "toolcall_end",
            "contentIndex": 0,
            "toolCall": {
                "type": "toolCall",
                "id": "call_test|fc_test",
                "name": "lookup",
                "arguments": {"value": "hello"},
                "namespace": "dynamic_tools",
            },
        }),
        json!({"type": "done", "reason": "toolUse", "usage": usage()}),
    ]);
    let (events, result) = stream_body(body).await;

    let Some(AssistantMessageEvent::ToolCallEnd { tool_call, .. }) =
        events.iter().find(|event| event.kind() == "toolcall_end")
    else {
        panic!("no toolcall_end event")
    };
    assert_eq!(tool_call.namespace.as_deref(), Some("dynamic_tools"));
    let AssistantBlock::ToolCall(call) = &result.content[0] else {
        panic!("expected a tool call")
    };
    assert_eq!(Value::Object(call.arguments.clone()), json!({"value": "hello"}));
    assert_eq!(call.namespace.as_deref(), Some("dynamic_tools"));
}

// Regression tests for https://github.com/earendil-works/pi/issues/8996
#[tokio::test]
async fn processes_terminal_metadata_when_the_event_is_not_newline_terminated() {
    let start = sse(&[json!({"type": "start"})]);
    let done = format!(
        "data: {}",
        json!({"type": "done", "reason": "stop", "usage": usage(), "providerThinkingLevel": "high"})
    );
    let (events, result) = stream_body(start + &done).await;

    assert_eq!(kinds(&events), ["start", "done"]);
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.provider_thinking_level.as_deref(), Some("high"));
}

#[tokio::test]
async fn emits_an_error_instead_of_hanging_when_the_stream_ends_without_a_terminal_event() {
    let (events, result) = stream_body(sse(&[json!({"type": "start"})])).await;

    assert_eq!(kinds(&events), ["start", "error"]);
    assert_eq!(result.stop_reason, StopReason::Error);
    assert!(
        result
            .error_message
            .unwrap()
            .contains("Connection closed by proxy server")
    );
}
