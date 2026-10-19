//! End-to-end tests of the `openai-responses` API against recorded SSE bodies.

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

use futures::StreamExt;
use futures::future::BoxFuture;
use serde_json::{Value, json};
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tau_ai::api::openai_responses::{self, OpenAIResponsesApi};
use tau_ai::options::{OpenAIResponsesOptions, ReasoningEffort};
use tau_ai::types::ProviderHeaders;
use tau_ai::{
    ApiOptions, AssistantBlock, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream, Context,
    ImageContent, Message, Model, ProviderStreams, SimpleStreamOptions, StopReason, StreamOptions, TextContent,
    ThinkingLevel, Tool, ToolCall, ToolChoice, ToolResultMessage, UserBlock, UserContent, UserMessage,
    normalize_context,
};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn model(base_url: &str) -> Model {
    serde_json::from_value(json!({
        "id": "gpt-5.4", "name": "GPT-5.4", "api": "openai-responses", "provider": "openai",
        "baseUrl": base_url, "reasoning": true, "input": ["text", "image"],
        "cost": {"input": 2.5, "output": 15, "cacheRead": 0.25, "cacheWrite": 0},
        "contextWindow": 400_000, "maxTokens": 128_000
    }))
    .unwrap()
}

fn sse(events: &[Value]) -> String {
    events.iter().fold(String::new(), |mut body, event| {
        let _ = write!(body, "data: {event}\n\n");
        body
    })
}

fn completed(id: &str) -> Value {
    json!({"type": "response.completed", "response": {"id": id, "status": "completed",
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}})
}

async fn server_with(body: String) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;
    server
}

fn base_url(server: &MockServer) -> String {
    format!("{}/v1", server.uri())
}

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("sk-test".into()),
        max_retries: Some(0),
        ..StreamOptions::default()
    }
}

fn hi() -> tau_ai::TranscriptContext {
    normalize_context(&Context::new(vec![Message::user("hi")]).with_system_prompt("sys"))
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
async fn streams_reasoning_text_and_usage() {
    let reasoning = json!({"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "Think"}],
                           "encrypted_content": "enc"});
    let server = server_with(sse(&[
        json!({"type": "response.created", "response": {"id": "resp_1"}}),
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
        json!({"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "Think"}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": reasoning}),
        json!({"type": "response.output_item.added", "output_index": 1, "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []}}),
        json!({"type": "response.output_text.delta", "output_index": 1, "delta": "Hello"}),
        json!({"type": "response.output_text.delta", "output_index": 1, "delta": " world"}),
        json!({"type": "response.output_item.done", "output_index": 1, "item": {"type": "message", "id": "msg_1", "role": "assistant",
               "phase": "final_answer", "content": [{"type": "output_text", "text": "Hello world", "annotations": []}]}}),
        json!({"type": "response.completed", "response": {"id": "resp_1", "status": "completed",
               "usage": {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200,
                         "input_tokens_details": {"cached_tokens": 400}, "output_tokens_details": {"reasoning_tokens": 50}}}}),
    ]))
    .await;
    let mut opts = options();
    opts.session_id = Some("session-123".into());
    opts.api = Some(ApiOptions::OpenAIResponses(OpenAIResponsesOptions {
        reasoning_effort: Some(ReasoningEffort::High),
        ..OpenAIResponsesOptions::default()
    }));
    let (events, message) = collect(OpenAIResponsesApi.stream(&model(&base_url(&server)), &hi(), opts)).await;

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
            "text_delta",
            "text_end",
            "done"
        ]
    );
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.response_id.as_deref(), Some("resp_1"));
    assert_eq!(message.text(), "Hello world");
    let AssistantBlock::Thinking(thinking) = &message.content[0] else {
        panic!()
    };
    assert_eq!(thinking.thinking, "Think");
    assert_eq!(
        thinking.thinking_signature.as_deref(),
        Some(reasoning.to_string().as_str())
    );
    let AssistantBlock::Text(text) = &message.content[1] else {
        panic!()
    };
    assert_eq!(
        text.text_signature.as_deref(),
        Some(r#"{"v":1,"id":"msg_1","phase":"final_answer"}"#)
    );
    assert_eq!(
        (
            message.usage.input,
            message.usage.output,
            message.usage.cache_read,
            message.usage.total_tokens
        ),
        (600, 200, 400, 1200)
    );
    assert_eq!(message.usage.reasoning, Some(50));
    assert!((message.usage.cost.total - (600.0 * 2.5 + 200.0 * 15.0 + 400.0 * 0.25) / 1e6).abs() < 1e-12);

    let request = last_request(&server).await;
    assert_eq!(header(&request, "authorization"), Some("Bearer sk-test"));
    assert_eq!(header(&request, "session_id"), Some("session-123"));
    assert_eq!(header(&request, "x-client-request-id"), Some("session-123"));
    assert!(header(&request, "user-agent").unwrap().starts_with("tau/"));
    let body = body(&request);
    assert_eq!(body["model"], "gpt-5.4");
    assert_eq!(body["stream"], true);
    assert_eq!(body["store"], false);
    assert_eq!(body["prompt_cache_key"], "session-123");
    assert_eq!(body["reasoning"], json!({"effort": "high", "summary": "auto"}));
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(
        body["input"],
        json!([
            {"role": "developer", "content": "sys"},
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ])
    );
}

#[tokio::test]
async fn streams_parallel_tool_calls_and_replays_them() {
    let call = |n: u32, arguments: &str| json!({"type": "function_call", "id": format!("fc_{n}"), "call_id": format!("call_{n}"), "name": "read", "arguments": arguments});
    let server = server_with(sse(&[
        json!({"type": "response.output_item.added", "output_index": 0, "item": call(1, "")}),
        json!({"type": "response.output_item.added", "output_index": 1, "item": call(2, "")}),
        json!({"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "{\"path\":"}),
        json!({"type": "response.function_call_arguments.delta", "output_index": 1, "delta": "{\"path\":\"b\"}"}),
        json!({"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "\"a\"}"}),
        json!({"type": "response.function_call_arguments.done", "output_index": 0, "arguments": "{\"path\":\"a\"}"}),
        json!({"type": "response.output_item.done", "output_index": 1, "item": call(2, "{\"path\":\"b\"}")}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": call(1, "{\"path\":\"a\"}")}),
        completed("resp_tools"),
    ]))
    .await;
    let tool = Tool::new(
        "read",
        "Read a file",
        json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
    );
    let context = normalize_context(&Context::new(vec![Message::user("read a and b")]).with_tools(vec![tool]));
    let target = model(&base_url(&server));
    let (events, message) = collect(openai_responses::stream(&target, &context, options())).await;
    assert_eq!(message.stop_reason, StopReason::ToolUse);
    let calls: Vec<&ToolCall> = message.tool_calls().collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id, "call_1|fc_1");
    assert_eq!(Value::Object(calls[0].arguments.clone()), json!({"path": "a"}));
    assert_eq!(Value::Object(calls[1].arguments.clone()), json!({"path": "b"}));
    let ends: Vec<usize> = events
        .iter()
        .filter_map(|event| match event {
            AssistantMessageEvent::ToolCallEnd { content_index, .. } => Some(*content_index),
            _ => None,
        })
        .collect();
    assert_eq!(ends, [1, 0]);
    let request = body(&last_request(&server).await);
    assert_eq!(
        request["tools"],
        json!([{"type": "function", "name": "read", "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}])
    );

    // The next request replays both calls and their results.
    let results: Vec<Message> = calls
        .iter()
        .map(|call| ToolResultMessage::text(call, format!("contents of {}", call.arguments["path"]), false).into())
        .collect();
    let mut messages = vec![Message::user("read a and b"), message.clone().into()];
    messages.extend(results);
    let context = normalize_context(&Context::new(messages));
    let input = tau_ai::api::openai_responses_shared::convert_responses_messages(
        &target,
        &context,
        &["openai"],
        &tau_ai::api::openai_responses_shared::ConvertResponsesMessagesOptions::default(),
    )
    .unwrap();
    assert_eq!(
        input[1..],
        [
            json!({"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read", "arguments": "{\"path\":\"a\"}"}),
            json!({"type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "read", "arguments": "{\"path\":\"b\"}"}),
            json!({"type": "function_call_output", "call_id": "call_1", "output": "contents of \"a\""}),
            json!({"type": "function_call_output", "call_id": "call_2", "output": "contents of \"b\""}),
        ]
    );
}

#[tokio::test]
async fn http_errors_carry_status_and_body_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(400)
                .set_body_json(json!({"error": {"message": "Invalid model", "type": "invalid_request_error"}})),
        )
        .mount(&server)
        .await;
    let (events, message) = collect(openai_responses::stream(&model(&base_url(&server)), &hi(), options())).await;
    assert_eq!(events.len(), 1);
    assert_eq!(message.stop_reason, StopReason::Error);
    // The prefix, the status and the body's `error` object (upstream `formatProviderError`).
    assert_eq!(
        message.error_message.as_deref(),
        Some(r#"OpenAI API error (400): {"message":"Invalid model","type":"invalid_request_error"}"#)
    );

    let mut proxy = model(&base_url(&server));
    proxy.provider = "opencode".into();
    let (_, message) = collect(openai_responses::stream(&proxy, &hi(), options())).await;
    assert_eq!(
        message.error_message.as_deref(),
        Some(r#"opencode API error (400): {"message":"Invalid model","type":"invalid_request_error"}"#)
    );

    // A gateway body without an `error` object keeps the SDK message after the prefix.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(403).set_body_string("blocked by gateway WAF"))
        .mount(&server)
        .await;
    let (_, message) = collect(openai_responses::stream(&model(&base_url(&server)), &hi(), options())).await;
    assert_eq!(
        message.error_message.as_deref(),
        Some("OpenAI API error (403): 403 blocked by gateway WAF")
    );
}

#[tokio::test]
async fn retries_retryable_statuses() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(503)
                .insert_header("retry-after-ms", "1")
                .set_body_json(json!({"error": {"message": "busy"}})),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse(&[completed("r")]), "text/event-stream"))
        .mount(&server)
        .await;
    let mut opts = options();
    opts.max_retries = Some(1);
    let (_, message) = collect(openai_responses::stream(&model(&base_url(&server)), &hi(), opts)).await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn in_stream_errors() {
    for (events, expected) in [
        (
            vec![json!({"type": "error", "code": "server_error", "message": "boom", "param": null})],
            "Error Code server_error: boom",
        ),
        (
            vec![json!({"error": {"message": "upstream exploded", "code": "server_error"}})],
            "upstream exploded",
        ),
        (
            vec![
                json!({"type": "response.failed", "response": {"status": "failed", "error": {"code": "rate_limit_exceeded", "message": "slow"}}}),
            ],
            "rate_limit_exceeded: slow",
        ),
        (
            vec![
                json!({"type": "response.incomplete", "response": {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}}}),
            ],
            "Response incomplete: content_filter",
        ),
    ] {
        let server = server_with(sse(&events)).await;
        let (stream_events, message) =
            collect(openai_responses::stream(&model(&base_url(&server)), &hi(), options())).await;
        assert_eq!(stream_events.first().unwrap().kind(), "start");
        assert_eq!(message.stop_reason, StopReason::Error);
        assert_eq!(message.error_message.as_deref(), Some(expected));
    }
    // `[DONE]` without a terminal response event.
    let server = server_with("data: [DONE]\n\n".into()).await;
    let (_, message) = collect(openai_responses::stream(&model(&base_url(&server)), &hi(), options())).await;
    assert_eq!(
        message.error_message.as_deref(),
        Some("OpenAI Responses stream ended before a terminal response event")
    );
}

#[tokio::test]
async fn length_stop_reason() {
    let server = server_with(sse(&[
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "m", "content": []}}),
        json!({"type": "response.output_text.delta", "output_index": 0, "delta": "cut"}),
        json!({"type": "response.incomplete", "response": {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}}),
    ]))
    .await;
    let (_, message) = collect(openai_responses::stream(&model(&base_url(&server)), &hi(), options())).await;
    assert_eq!(message.stop_reason, StopReason::Length);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("incomplete.max_output_tokens"));
    assert_eq!(message.text(), "cut");
}

#[tokio::test]
async fn abort_via_cancel() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse(&[completed("r")]), "text/event-stream")
                .set_delay(Duration::from_secs(30)),
        )
        .mount(&server)
        .await;
    let cancel = CancellationToken::new();
    let mut opts = options();
    opts.cancel = Some(cancel.clone());
    let stream = openai_responses::stream(&model(&base_url(&server)), &hi(), opts);
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel.cancel();
    });
    let (events, message) = collect(stream).await;
    assert_eq!(events.len(), 1);
    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert_eq!(message.error_message.as_deref(), Some("Request was aborted"));
}

#[tokio::test]
async fn callbacks_see_payload_response_and_events() {
    let server = server_with(sse(&[
        json!({"type": "response.created", "response": {"id": "resp_cb"}}),
        completed("resp_cb"),
    ]))
    .await;
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let mut opts = options();
    let log = seen.clone();
    opts.on_provider_stream_event = Some(Arc::new(move |event: &Value, _: &Model| -> BoxFuture<'static, _> {
        log.lock().unwrap().push(event["type"].as_str().unwrap().to_owned());
        Box::pin(async { Ok(()) })
    }));
    let log = seen.clone();
    opts.on_response = Some(Arc::new(
        move |response: &tau_ai::options::ProviderResponse, _: &Model| -> BoxFuture<'static, _> {
            log.lock().unwrap().push(format!("status {}", response.status));
            Box::pin(async { Ok(()) })
        },
    ));
    opts.on_payload = Some(Arc::new(|payload: &Value, _: &Model| -> BoxFuture<'static, _> {
        let mut replaced = payload.clone();
        replaced["metadata"] = json!({"tag": "x"});
        Box::pin(async move { Ok(Some(replaced)) })
    }));
    let (_, message) = collect(openai_responses::stream(&model(&base_url(&server)), &hi(), opts)).await;
    assert_eq!(message.response_id.as_deref(), Some("resp_cb"));
    assert_eq!(
        *seen.lock().unwrap(),
        ["status 200", "response.created", "response.completed"]
    );
    assert_eq!(body(&last_request(&server).await)["metadata"], json!({"tag": "x"}));
}

#[tokio::test]
async fn auth_requirements() {
    let target = model("http://127.0.0.1:9/v1");
    let (_, message) = collect(openai_responses::stream(&target, &hi(), StreamOptions::default())).await;
    assert_eq!(
        message.error_message.as_deref(),
        Some("No API key for provider: openai")
    );

    let server = server_with(sse(&[completed("r")])).await;
    let headers: ProviderHeaders = [("Authorization".to_owned(), Some("Bearer custom".to_owned()))]
        .into_iter()
        .collect();
    let opts = StreamOptions {
        headers: Some(headers),
        max_retries: Some(0),
        ..StreamOptions::default()
    };
    let (_, message) = collect(openai_responses::stream_simple(
        &model(&base_url(&server)),
        &hi(),
        SimpleStreamOptions::from(opts),
    ))
    .await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(
        header(&last_request(&server).await, "authorization"),
        Some("Bearer custom")
    );
}

#[tokio::test]
async fn session_affinity_formats_and_overrides() {
    let server = server_with(sse(&[completed("r")])).await;
    let run = |target: Model, opts: StreamOptions| {
        let server = &server;
        async move {
            collect(openai_responses::stream(&target, &hi(), opts)).await;
            last_request(server).await
        }
    };
    let mut openrouter = model(&base_url(&server));
    openrouter.provider = "openrouter".into();
    let mut opts = options();
    opts.session_id = Some("session-or".into());
    let request = run(openrouter, opts.clone()).await;
    assert_eq!(header(&request, "x-session-id"), Some("session-or"));
    assert_eq!(header(&request, "session_id"), None);
    assert_eq!(body(&request)["prompt_cache_key"], "session-or");

    let mut nosession = model(&base_url(&server));
    nosession.compat = Some(
        tau_ai::compat::ModelCompat::from_value(
            "openai-responses",
            json!({"sessionAffinityFormat": "openai-nosession"}),
        )
        .unwrap(),
    );
    let request = run(nosession, opts.clone()).await;
    assert_eq!(header(&request, "session_id"), None);
    assert_eq!(header(&request, "x-client-request-id"), Some("session-or"));

    let mut overridden = opts.clone();
    overridden.headers = Some(
        [
            ("session_id".to_owned(), Some("override-session".to_owned())),
            ("x-client-request-id".to_owned(), None),
        ]
        .into_iter()
        .collect(),
    );
    let request = run(model(&base_url(&server)), overridden).await;
    assert_eq!(header(&request, "session_id"), Some("override-session"));
    assert_eq!(header(&request, "x-client-request-id"), None);

    let mut no_cache = opts;
    no_cache.cache_retention = Some(tau_ai::CacheRetention::None);
    let request = run(model(&base_url(&server)), no_cache).await;
    assert_eq!(header(&request, "session_id"), None);
    assert!(body(&request).get("prompt_cache_key").is_none());
}

#[tokio::test]
async fn service_tier_is_sent_and_priced() {
    let server = server_with(sse(&[json!({"type": "response.completed", "response": {
        "status": "completed", "service_tier": "flex",
        "usage": {"input_tokens": 1_000_000, "output_tokens": 0, "total_tokens": 1_000_000}}})]))
    .await;
    let mut opts = options();
    opts.api = Some(ApiOptions::OpenAIResponses(OpenAIResponsesOptions {
        service_tier: Some("priority".into()),
        ..OpenAIResponsesOptions::default()
    }));
    let (_, message) = collect(openai_responses::stream(&model(&base_url(&server)), &hi(), opts)).await;
    assert_eq!(body(&last_request(&server).await)["service_tier"], "priority");
    // The response's tier wins over the requested one.
    assert!((message.usage.cost.total - 2.5 * 0.5).abs() < 1e-9);
}

#[tokio::test]
async fn stream_simple_maps_reasoning_and_tool_choice() {
    let server = server_with(sse(&[completed("r")])).await;
    let tool = Tool::new("ping", "Ping", json!({"type": "object", "properties": {}}));
    let context = normalize_context(&Context::new(vec![Message::user("hi")]).with_tools(vec![tool]));
    let options = SimpleStreamOptions {
        base: options(),
        reasoning: Some(ThinkingLevel::Medium),
        tool_choice: Some(ToolChoice::None),
        ..SimpleStreamOptions::default()
    };
    let (_, message) = collect(openai_responses::stream_simple(
        &model(&base_url(&server)),
        &context,
        options,
    ))
    .await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    let request = body(&last_request(&server).await);
    assert_eq!(request["reasoning"], json!({"effort": "medium", "summary": "auto"}));
    assert_eq!(request["tool_choice"], "none");
    assert_eq!(request["max_output_tokens"], 128_000);
}

#[tokio::test]
async fn replays_foreign_messages_and_images() {
    let server = server_with(sse(&[completed("r")])).await;
    let foreign = AssistantMessage {
        content: vec![
            AssistantBlock::Thinking(tau_ai::ThinkingContent {
                thinking: "secret".into(),
                thinking_signature: Some("anthropic-sig".into()),
                redacted: None,
            }),
            AssistantBlock::Text(TextContent::new("Calling a tool.")),
            AssistantBlock::ToolCall(ToolCall {
                id: "toolu_01A+B".into(),
                name: "screenshot".into(),
                ..ToolCall::default()
            }),
        ],
        stop_reason: StopReason::ToolUse,
        ..AssistantMessage::empty("anthropic-messages", "anthropic", "claude-opus-4-8")
    };
    let image = ImageContent {
        data: "iVBORw0KGgo=".into(),
        mime_type: "image/png".into(),
    };
    let messages = vec![
        Message::User(UserMessage::new(UserContent::Blocks(vec![
            UserBlock::text("what is this?"),
            UserBlock::Image(image.clone()),
        ]))),
        foreign.into(),
        // The tool call is orphaned: a synthetic error result is sent.
        Message::user("continue"),
    ];
    let context = normalize_context(&Context::new(messages));
    collect(openai_responses::stream(
        &model(&base_url(&server)),
        &context,
        options(),
    ))
    .await;
    let request = body(&last_request(&server).await);
    assert_eq!(
        request["input"],
        json!([
            {"role": "user", "content": [
                {"type": "input_text", "text": "what is this?"},
                {"type": "input_image", "detail": "auto", "image_url": "data:image/png;base64,iVBORw0KGgo="}
            ]},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "Calling a tool.", "annotations": []}],
             "status": "completed", "id": "msg_pi_1"},
            {"type": "function_call", "call_id": "toolu_01A_B", "name": "screenshot", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "toolu_01A_B", "output": "No result provided"},
            {"role": "user", "content": [{"type": "input_text", "text": "continue"}]}
        ])
    );
}

#[tokio::test]
async fn reasoning_effort_off_and_max_tokens_compat() {
    let server = server_with(sse(&[completed("r")])).await;
    let mut opts = options();
    opts.max_tokens = Some(1024);
    collect(openai_responses::stream(
        &model(&base_url(&server)),
        &hi(),
        opts.clone(),
    ))
    .await;
    let request = body(&last_request(&server).await);
    assert_eq!(request["reasoning"], json!({"effort": "none"}));
    assert_eq!(request["max_output_tokens"], 1024);

    let mut capped = model(&base_url(&server));
    capped.compat = Some(
        tau_ai::compat::ModelCompat::from_value("openai-responses", json!({"supportsMaxOutputTokens": false})).unwrap(),
    );
    capped.thinking_level_map = Some([(tau_ai::model::ModelThinkingLevel::Off, None)].into_iter().collect());
    collect(openai_responses::stream(&capped, &hi(), opts)).await;
    let request = body(&last_request(&server).await);
    assert!(request.get("max_output_tokens").is_none());
    assert!(request.get("reasoning").is_none());
}
