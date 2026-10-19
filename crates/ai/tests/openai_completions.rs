//! End-to-end tests of the `openai-completions` API against recorded SSE bodies.

// Shared helpers outside `#[test]` functions panic on setup failures too.
#![allow(clippy::unwrap_used)]

use futures::StreamExt;
use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use std::fmt::Write as _;
use std::io::{Read, Write as _};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tau_ai::api::openai_completions::{self, OpenAICompletionsApi};
use tau_ai::options::{OpenAICompletionsOptions, ProviderResponse, ReasoningEffort};
use tau_ai::types::{ConstrainedSampling, GrammarVariants};
use tau_ai::{
    ApiOptions, AssistantBlock, AssistantMessage, AssistantMessageEvent, Context, Message, Model, ProviderStreams,
    SimpleStreamOptions, StopReason, StreamOptions, TextContent, ThinkingContent, ThinkingLevel, Tool, ToolCall,
    ToolChoice, TranscriptContext, normalize_context,
};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn model(base_url: &str, provider: &str, id: &str, reasoning: bool, compat: Option<Value>) -> Model {
    let mut value = json!({
        "id": id, "name": id, "api": "openai-completions", "provider": provider, "baseUrl": base_url,
        "reasoning": reasoning, "input": ["text", "image"],
        "cost": {"input": 1, "output": 2, "cacheRead": 0.5, "cacheWrite": 0},
        "contextWindow": 128_000, "maxTokens": 4096
    });
    if let Some(compat) = compat {
        value["compat"] = compat;
    }
    serde_json::from_value(value).unwrap()
}

fn ctx(messages: Vec<Message>) -> TranscriptContext {
    normalize_context(&Context::new(messages))
}

fn hi() -> TranscriptContext {
    ctx(vec![Message::user("Hi")])
}

fn read_tool() -> Tool {
    Tool::new(
        "read",
        "Read a file",
        json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
    )
}

fn sse(chunks: &[Value]) -> String {
    let mut body = String::new();
    for chunk in chunks {
        let _ = write!(body, "data: {chunk}\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    body
}

fn sse_response(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
}

async fn serve(chunks: &[Value]) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse_response(sse(chunks)))
        .mount(&server)
        .await;
    server
}

fn client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

fn options() -> StreamOptions {
    StreamOptions {
        api_key: Some("sk-test".into()),
        http_client: Some(client()),
        max_retries: Some(0),
        ..StreamOptions::default()
    }
}

async fn collect(mut stream: tau_ai::AssistantMessageEventStream) -> (AssistantMessage, Vec<AssistantMessageEvent>) {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    let message = events
        .last()
        .and_then(AssistantMessageEvent::final_message)
        .cloned()
        .unwrap();
    (message, events)
}

async fn run(
    model: &Model,
    context: &TranscriptContext,
    options: StreamOptions,
) -> (AssistantMessage, Vec<AssistantMessageEvent>) {
    collect(openai_completions::stream(model, context, options)).await
}

fn kinds(events: &[AssistantMessageEvent]) -> Vec<&'static str> {
    events.iter().map(AssistantMessageEvent::kind).collect()
}

async fn last_request(server: &MockServer) -> Request {
    server.received_requests().await.unwrap().pop().unwrap()
}

fn stop_chunk() -> Value {
    json!({"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
}

fn tool_call(call: &AssistantBlock) -> &ToolCall {
    call.as_tool_call().unwrap()
}

#[tokio::test]
async fn streams_text_with_usage_and_sends_the_wire_format() {
    let chunks = [
        json!({"id": "chatcmpl-1", "model": "gpt-4o-mini-2024", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]}),
        json!({"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": null}]}),
        json!({"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        json!({"id": "chatcmpl-1", "choices": [], "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000,
            "prompt_tokens_details": {"cached_tokens": 500_000}, "completion_tokens_details": {"reasoning_tokens": 7}}}),
    ];
    let server = serve(&chunks).await;
    let mut model = model(&format!("{}/v1", server.uri()), "openai", "gpt-4o-mini", false, None);
    model.headers = Some([("X-Model".to_owned(), "model-header".to_owned())].into());
    let context = normalize_context(&Context::new(vec![Message::user("Hi")]).with_system_prompt("sys"));
    let mut opts = options();
    opts.max_tokens = Some(100);
    opts.headers = Some([("X-Extra".to_owned(), Some("1".to_owned()))].into());
    let (message, events) = run(&model, &context, opts).await;

    assert_eq!(
        kinds(&events),
        ["start", "text_start", "text_delta", "text_delta", "text_end", "done"]
    );
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("stop"));
    assert_eq!(message.content, vec![AssistantBlock::Text(TextContent::new("Hello"))]);
    assert_eq!(message.response_id.as_deref(), Some("chatcmpl-1"));
    assert_eq!(message.response_model.as_deref(), Some("gpt-4o-mini-2024"));
    let usage = &message.usage;
    assert_eq!(
        (usage.input, usage.output, usage.cache_read),
        (500_000, 1_000_000, 500_000)
    );
    assert_eq!(usage.reasoning, Some(7));
    assert_eq!(usage.total_tokens, 2_000_000);
    assert!((usage.cost.total - (0.5 + 2.0 + 0.25)).abs() < 1e-9);

    let request = last_request(&server).await;
    let header = |name: &str| request.headers.get(name).map(|v| v.to_str().unwrap().to_owned());
    assert_eq!(header("authorization").as_deref(), Some("Bearer sk-test"));
    assert_eq!(header("content-type").as_deref(), Some("application/json"));
    assert!(header("user-agent").unwrap().starts_with("tau/"));
    assert_eq!(header("x-model").as_deref(), Some("model-header"));
    assert_eq!(header("x-extra").as_deref(), Some("1"));
    assert_eq!(
        request.body_json::<Value>().unwrap(),
        json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "Hi"}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "store": false,
            "max_completion_tokens": 100
        })
    );
}

#[tokio::test]
async fn headers_override_defaults_and_session_affinity() {
    let server = serve(&[stop_chunk()]).await;
    let mut base = model(
        &format!("{}/v1/", server.uri()),
        "proxy",
        "m",
        false,
        Some(json!({"sendSessionAffinityHeaders": true})),
    );
    base.headers = Some([("X-Model".to_owned(), "a".to_owned())].into());
    let mut opts = options();
    opts.session_id = Some("session-1".into());
    opts.headers = Some(
        [
            ("User-Agent".to_owned(), Some("custom".to_owned())),
            ("x-model".to_owned(), None),
            ("Authorization".to_owned(), None),
            ("x-session-affinity".to_owned(), Some("override".to_owned())),
        ]
        .into(),
    );
    let (message, _) = run(&base, &hi(), opts).await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    let request = last_request(&server).await;
    let header = |name: &str| request.headers.get(name).map(|v| v.to_str().unwrap().to_owned());
    assert_eq!(header("user-agent").as_deref(), Some("custom"));
    assert_eq!(header("x-model"), None);
    assert_eq!(header("authorization"), None);
    assert_eq!(header("session_id").as_deref(), Some("session-1"));
    assert_eq!(header("x-client-request-id").as_deref(), Some("session-1"));
    assert_eq!(header("x-session-affinity").as_deref(), Some("override"));

    // OpenRouter format, and no headers when caching is disabled.
    let open_router = model(&format!("{}/v1", server.uri()), "openrouter", "auto", false, None);
    let mut opts = options();
    opts.session_id = Some("s".into());
    run(&open_router, &hi(), opts.clone()).await;
    let request = last_request(&server).await;
    assert_eq!(request.headers.get("x-session-id").unwrap(), "s");
    assert!(request.headers.get("x-session-affinity").is_none());
    opts.cache_retention = Some(tau_ai::CacheRetention::None);
    run(&open_router, &hi(), opts).await;
    assert!(last_request(&server).await.headers.get("x-session-id").is_none());
}

#[tokio::test]
async fn missing_api_key_is_an_error_event_and_auth_headers_stand_in() {
    let model_without_key = model("https://x.invalid", "test-provider", "m", false, None);
    let (message, events) = run(&model_without_key, &hi(), StreamOptions::default()).await;
    assert_eq!(kinds(&events), ["error"]);
    assert_eq!(
        message.error_message.as_deref(),
        Some("No API key for provider: test-provider")
    );

    let server = serve(&[stop_chunk()]).await;
    let gateway = model(
        &format!("{}/v1", server.uri()),
        "cloudflare-ai-gateway",
        "m",
        false,
        None,
    );
    let mut opts = options();
    opts.api_key = None;
    opts.headers = Some(
        [
            ("Authorization".to_owned(), None),
            ("cf-aig-authorization".to_owned(), Some("Bearer cf".to_owned())),
        ]
        .into(),
    );
    let (message, _) = run(&gateway, &hi(), opts).await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    let request = last_request(&server).await;
    assert!(request.headers.get("authorization").is_none());
    assert_eq!(request.headers.get("cf-aig-authorization").unwrap(), "Bearer cf");
}

#[tokio::test]
async fn thinking_precedes_text_and_keeps_its_field_signature() {
    let chunks = [
        json!({"id": "c", "choices": [{"index": 0, "delta": {"reasoning_content": "think", "content": "answer"}}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {"reasoning_content": "", "reasoning": " more"}}]}),
        stop_chunk(),
    ];
    let server = serve(&chunks).await;
    let m = model(&format!("{}/v1", server.uri()), "local", "m", true, None);
    let (message, events) = run(&m, &hi(), options()).await;
    assert_eq!(
        kinds(&events),
        [
            "start",
            "thinking_start",
            "thinking_delta",
            "text_start",
            "text_delta",
            "thinking_delta",
            "thinking_end",
            "text_end",
            "done"
        ]
    );
    assert_eq!(
        message.content,
        vec![
            AssistantBlock::Thinking(ThinkingContent {
                thinking: "think more".into(),
                thinking_signature: Some("reasoning_content".into()),
                redacted: None,
            }),
            AssistantBlock::Text(TextContent::new("answer")),
        ]
    );
    let AssistantMessageEvent::ThinkingEnd { signature, .. } = &events[6] else {
        panic!()
    };
    assert_eq!(signature.as_deref(), Some("reasoning_content"));
}

#[tokio::test]
async fn reasoning_details_become_the_signature_and_replay() {
    let detail = json!({"type": "reasoning.encrypted", "id": "call_1", "data": "encrypted-signature"});
    let chunks = [
        json!({"id": "c", "choices": [{"index": 0, "delta": {"reasoning_details": [detail, {"type": "bogus"}]}}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
            "function": {"name": "read", "arguments": "{\"path\":\"README.md\"}"}}]}}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let server = serve(&chunks).await;
    let m = model(
        &format!("{}/v1", server.uri()),
        "openrouter",
        "google/gemini",
        true,
        None,
    );
    let context = normalize_context(&Context::new(vec![Message::user("read")]).with_tools(vec![read_tool()]));
    let (message, _) = run(&m, &context, options()).await;
    assert_eq!(message.stop_reason, StopReason::ToolUse);
    let thinking = message.content[0].as_thinking().unwrap();
    assert_eq!(thinking.thinking, "");
    assert_eq!(
        thinking.thinking_signature.as_deref(),
        Some(json!([detail]).to_string().as_str())
    );
    assert_eq!(
        tool_call(&message.content[1]).arguments,
        json!({"path": "README.md"}).as_object().cloned().unwrap()
    );

    let follow_up = ctx(vec![
        Message::user("read"),
        Message::Assistant(message),
        Message::ToolResult(tau_ai::ToolResultMessage::text(
            &ToolCall {
                id: "call_1".into(),
                name: "read".into(),
                ..ToolCall::default()
            },
            "contents",
            false,
        )),
    ]);
    let body = openai_completions::build_params(&m, &follow_up, &options()).unwrap();
    assert_eq!(body["messages"][1]["reasoning_details"], json!([detail]));
}

#[tokio::test]
async fn accumulates_parallel_tool_calls_by_index_and_id() {
    fn call(index: Option<i64>, id: Option<&str>, arguments: &str, name: Option<&str>) -> Value {
        let mut item = json!({"type": "function", "function": {"arguments": arguments}});
        if let Some(index) = index {
            item["index"] = json!(index);
        }
        if let Some(id) = id {
            item["id"] = json!(id);
        }
        if let Some(name) = name {
            item["function"]["name"] = json!(name);
        }
        item
    }
    let chunks = [
        json!({"id": "c", "choices": [{"index": 0, "delta": {"content": "answer 1", "reasoning_content": "think 1",
        "tool_calls": [
            call(Some(0), Some("tc_read_initial"), "{\"path\":\"README", Some("read")),
            call(Some(1), Some("tc_grep_initial"), "{\"pattern\":\"TODO", Some("grep")),
            call(None, Some("tc_list_no_index"), "{\"path\":\"packages", Some("list")),
            call(None, Some("tc_write_no_index"), "{\"path\":\"out", Some("write")),
        ]}}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {"content": " answer 2", "tool_calls": [
            call(Some(1), Some("tc_grep_changed"), "\",\"path\":\"src", None),
            call(None, Some("tc_write_no_index"), ".txt\",\"content\":\"ok\"}", None),
            call(None, Some("tc_list_no_index"), "/ai\"}", None),
        ]}}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {"content": "\n", "reasoning_content": " think 2",
            "tool_calls": [call(Some(0), Some("tc_read_changed"), ".md\"}", None), call(Some(1), None, "\"}", None)]},
            "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "completion_tokens_details": {"reasoning_tokens": 2}}}),
    ];
    let server = serve(&chunks).await;
    let m = model(&format!("{}/v1", server.uri()), "openai", "gpt-4o-mini", false, None);
    let (message, events) = run(&m, &hi(), options()).await;
    assert_eq!(message.stop_reason, StopReason::ToolUse);
    let count = |kind: &str| events.iter().filter(|event| event.kind() == kind).count();
    for (kind, expected) in [
        ("text_start", 1),
        ("text_delta", 3),
        ("thinking_delta", 2),
        ("toolcall_start", 4),
        ("toolcall_delta", 9),
        ("toolcall_end", 4),
    ] {
        assert_eq!(count(kind), expected, "{kind}");
    }
    let obj = |value: Value| value.as_object().cloned().unwrap();
    let expected = [
        ("tc_read_initial", "read", obj(json!({"path": "README.md"}))),
        (
            "tc_grep_initial",
            "grep",
            obj(json!({"pattern": "TODO", "path": "src"})),
        ),
        ("tc_list_no_index", "list", obj(json!({"path": "packages/ai"}))),
        (
            "tc_write_no_index",
            "write",
            obj(json!({"path": "out.txt", "content": "ok"})),
        ),
    ];
    assert_eq!(message.content.len(), 6);
    assert_eq!(message.content[0].as_thinking().unwrap().thinking, "think 1 think 2");
    assert_eq!(message.content[1].as_text().unwrap().text, "answer 1 answer 2\n");
    for (block, (id, name, arguments)) in message.content[2..].iter().zip(expected) {
        let call = tool_call(block);
        assert_eq!(
            (call.id.as_str(), call.name.as_str(), &call.arguments),
            (id, name, &arguments)
        );
    }
    assert_eq!(message.usage.reasoning, Some(2));
}

#[tokio::test]
async fn coalesces_tool_call_deltas_with_changing_ids_at_one_index() {
    let delta = |id: &str, name: Option<&str>, arguments: &str, finish: Option<&str>| {
        json!({"id": "chatcmpl-kimi", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": id,
            "type": "function", "function": {"name": name, "arguments": arguments}}]}, "finish_reason": finish}]})
    };
    let chunks = [
        delta("functions.read:0", Some("read"), "", None),
        delta("chatcmpl-tool-a", None, "{\"path\":\"README", None),
        delta("chatcmpl-tool-b", None, ".md\"}", Some("tool_calls")),
    ];
    let server = serve(&chunks).await;
    let m = model(&format!("{}/v1", server.uri()), "openai", "gpt-4o-mini", false, None);
    let (message, events) = run(&m, &hi(), options()).await;
    let tool_events: Vec<_> = events
        .iter()
        .filter(|event| event.kind().starts_with("toolcall_"))
        .map(AssistantMessageEvent::kind)
        .collect();
    assert_eq!(
        tool_events,
        ["toolcall_start", "toolcall_delta", "toolcall_delta", "toolcall_end"]
    );
    assert_eq!(message.content.len(), 1);
    let call = tool_call(&message.content[0]);
    assert_eq!((call.id.as_str(), call.name.as_str()), ("functions.read:0", "read"));
    assert_eq!(
        call.arguments,
        json!({"path": "README.md"}).as_object().cloned().unwrap()
    );
}

#[tokio::test]
async fn anonymous_deltas_go_to_the_only_call_and_late_ids_are_adopted() {
    let chunks = [
        json!({"choices": [{"delta": {"tool_calls": [{"function": {"name": "read", "arguments": "{\"pa"}}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"id": "late", "function": {"arguments": "th\":\"a\"}"}}]}}]}),
        // Compat servers may send `stop` after tool calls; the calls are still emitted.
        json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
    ];
    let server = serve(&chunks).await;
    let m = model(&format!("{}/v1", server.uri()), "local", "m", false, None);
    let (message, _) = run(&m, &hi(), options()).await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.content.len(), 2, "{:?}", message.content);
    // The id-only delta has no index and an unknown id, so it starts a second call.
    let first = tool_call(&message.content[0]);
    assert_eq!(first.name, "read");
    assert_eq!(
        first
            .arguments_error
            .as_deref()
            .map(|e| e.contains("Received arguments")),
        Some(true)
    );

    let chunks = [
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "read", "arguments": "{\"pa"}}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "late", "function": {"arguments": "th\":\"a\"}"}}]}}]}),
        json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let server = serve(&chunks).await;
    let m = model(&format!("{}/v1", server.uri()), "local", "m", false, None);
    let (message, _) = run(&m, &hi(), options()).await;
    let call = tool_call(&message.content[0]);
    assert_eq!((call.id.as_str(), call.arguments_error.is_none()), ("late", true));

    let chunks = [
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "x", "arguments": "{"}}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "b", "function": {"name": "y", "arguments": "{"}}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"function": {"arguments": "}"}}]}}]}),
    ];
    let server = serve(&chunks).await;
    let m = model(&format!("{}/v1", server.uri()), "local", "m", false, None);
    let (message, _) = run(&m, &hi(), options()).await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert!(message.error_message.unwrap().contains("several tool calls"));

    let chunks = [
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "x", "arguments": "{\"k\":"}}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"function": {"arguments": "1}"}}]}, "finish_reason": "tool_calls"}]}),
    ];
    let server = serve(&chunks).await;
    let m = model(&format!("{}/v1", server.uri()), "local", "m", false, None);
    let (message, _) = run(&m, &hi(), options()).await;
    assert_eq!(
        tool_call(&message.content[0]).arguments,
        json!({"k": 1}).as_object().cloned().unwrap()
    );
}

#[tokio::test]
async fn invalid_tool_arguments_are_a_tool_level_error() {
    let chunks = [
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read", "arguments": "{\"pa"}}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "th\":\"a\""}}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": " oops"}}]}, "finish_reason": "tool_calls"}]}),
    ];
    let server = serve(&chunks).await;
    let m = model(&format!("{}/v1", server.uri()), "openai", "gpt-4o-mini", false, None);
    let mut stream = openai_completions::stream(&m, &hi(), options());
    let mut partial_arguments = Vec::new();
    while let Some(event) = stream.next().await {
        if let AssistantMessageEvent::ToolCallDelta { content_index, .. } = &event {
            let partial = stream.partial().unwrap();
            partial_arguments.push(Value::Object(
                tool_call(&partial.content[*content_index]).arguments.clone(),
            ));
        }
    }
    // The preview of the invalid tail is up to the shared partial-JSON parser.
    assert_eq!(partial_arguments[..2], [json!({}), json!({"path": "a"})]);
    assert_eq!(partial_arguments.len(), 3);
    let message = stream.partial().unwrap();
    assert_eq!(message.stop_reason, StopReason::ToolUse);
    let call = tool_call(&message.content[0]);
    assert!(call.arguments.is_empty());
    assert!(
        call.arguments_error
            .as_deref()
            .unwrap()
            .ends_with("Received arguments:\n{\"path\":\"a\" oops")
    );
}

#[tokio::test]
async fn grammar_tool_calls_stream_as_json_arguments() {
    let chunks = [
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "custom",
            "custom": {"name": "apply_patch", "input": "a\"b"}}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "type": "custom", "custom": {"input": "c"}}]},
            "finish_reason": "tool_calls"}]}),
    ];
    let server = serve(&chunks).await;
    let m = model(
        &format!("{}/v1", server.uri()),
        "openai",
        "gpt-5",
        false,
        Some(json!({"supportsOpenAIGrammarTools": true})),
    );
    let mut tool = Tool::new(
        "apply_patch",
        "Apply a patch",
        json!({"type": "object", "properties": {"patch": {"type": "string"}}, "required": ["patch"]}),
    );
    tool.constrained_sampling = Some(ConstrainedSampling::Grammar {
        variants: GrammarVariants {
            openai_lark: Some("start: /.+/".into()),
            openai_regex: None,
        },
    });
    let context = normalize_context(&Context::new(vec![Message::user("Hi")]).with_tools(vec![tool]));
    let (message, events) = run(&m, &context, options()).await;
    let deltas: String = events
        .iter()
        .filter_map(|event| match event {
            AssistantMessageEvent::ToolCallDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, r#"{"patch":"a\"bc"}"#);
    let call = tool_call(&message.content[0]);
    assert_eq!(
        (
            call.id.as_str(),
            call.name.as_str(),
            Value::Object(call.arguments.clone())
        ),
        ("call_1", "apply_patch", json!({"patch": "a\"bc"}))
    );
    assert_eq!(message.stop_reason, StopReason::ToolUse);
    let body = last_request(&server).await.body_json::<Value>().unwrap();
    assert_eq!(body["tools"][0]["type"], "custom");
}

#[tokio::test]
async fn finish_reason_rules() {
    let partial = json!({"id": "c", "choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": null}]});
    let server = serve(std::slice::from_ref(&partial)).await;
    let m = model(&format!("{}/v1", server.uri()), "openai", "gpt-4o-mini", false, None);
    let (message, events) = run(&m, &hi(), options()).await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Stream ended without finish_reason")
    );
    assert_eq!(events.last().unwrap().kind(), "error");
    assert_eq!(message.content, vec![AssistantBlock::Text(TextContent::new("partial"))]);

    let lenient = model(
        &format!("{}/v1", server.uri()),
        "local",
        "m",
        false,
        Some(json!({"supportsFinishReason": false})),
    );
    let (message, _) = run(&lenient, &hi(), options()).await;
    assert_eq!(message.stop_reason, StopReason::Stop);

    let tool_server = serve(&[
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read"}}]}}]}),
    ])
    .await;
    let lenient = model(
        &format!("{}/v1", tool_server.uri()),
        "local",
        "m",
        false,
        Some(json!({"supportsFinishReason": false})),
    );
    let (message, _) = run(&lenient, &hi(), options()).await;
    assert_eq!(message.stop_reason, StopReason::ToolUse);

    let server = serve(&[
        partial,
        json!({"choices": [{"delta": {}, "finish_reason": "network_error"}]}),
    ])
    .await;
    let m = model(&format!("{}/v1", server.uri()), "zai", "glm", false, None);
    let (message, events) = run(&m, &hi(), options()).await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("network_error"));
    assert_eq!(
        message.error_message.as_deref(),
        Some("Provider finish_reason: network_error")
    );
    assert_eq!(kinds(&events)[events.len() - 2..], ["text_end", "error"]);
}

#[tokio::test]
async fn ignores_null_chunks_and_stops_at_done() {
    let body = format!(
        "data: null\n\n: keep-alive\n\ndata: {}\n\ndata: {}\n\ndata: [DONE]\n\ndata: {}\n\n",
        json!({"id": "chatcmpl-test", "choices": [{"delta": {"content": "OK"}, "finish_reason": null}]}),
        json!({"id": "chatcmpl-test", "choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}),
        json!({"id": "x", "choices": [{"delta": {"content": "ignored"}}]}),
    );
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;
    let m = model(&format!("{}/v1", server.uri()), "openai", "gpt-4o-mini", false, None);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut opts = options();
    let sink = seen.clone();
    opts.on_provider_stream_event = Some(Arc::new(move |event: &Value, _: &Model| {
        sink.lock().unwrap().push(event.clone());
        Box::pin(async { Ok(()) })
    }));
    let (message, _) = run(&m, &hi(), opts).await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "OK");
    assert_eq!(message.usage.total_tokens, 4);
    assert_eq!(seen.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn in_stream_errors_keep_partial_content() {
    let chunks = [
        json!({"id": "c", "choices": [{"delta": {"content": "partial"}}]}),
        json!({"error": {"message": "Overloaded", "code": 529, "metadata": {"raw": "upstream overloaded"}}}),
    ];
    let server = serve(&chunks).await;
    let m = model(&format!("{}/v1", server.uri()), "openrouter", "m", false, None);
    let seen = Arc::new(Mutex::new(0));
    let mut opts = options();
    let counter = seen.clone();
    opts.on_provider_stream_event = Some(Arc::new(move |_: &Value, _: &Model| {
        *counter.lock().unwrap() += 1;
        Box::pin(async { Ok(()) })
    }));
    let (message, events) = run(&m, &hi(), opts).await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Overloaded\nupstream overloaded")
    );
    assert_eq!(message.content, vec![AssistantBlock::Text(TextContent::new("partial"))]);
    assert_eq!(*seen.lock().unwrap(), 1);
    assert_eq!(events.last().unwrap().kind(), "error");

    let server = serve(&[json!({"detail": "Not authenticated"})]).await;
    let m = model(&format!("{}/v1", server.uri()), "local", "m", false, None);
    let (message, _) = run(&m, &hi(), options()).await;
    assert_eq!(message.error_message.as_deref(), Some("Not authenticated"));
}

#[tokio::test]
async fn http_errors_and_retries() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(
                json!({"error": {"message": "Invalid value for 'model'", "type": "invalid_request_error"}}),
            ),
        )
        .mount(&server)
        .await;
    let m = model(&format!("{}/v1", server.uri()), "openai", "gpt-4o-mini", false, None);
    let (message, events) = run(&m, &hi(), options()).await;
    assert_eq!(kinds(&events), ["error"]);
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_message.as_deref(), Some("400 Invalid value for 'model'"));

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after-ms", "1")
                .set_body_json(json!({"error": {"message": "rate limited"}})),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(sse_response(sse(&[
            json!({"id": "c", "choices": [{"delta": {"content": "ok"}}]}),
            stop_chunk(),
        ])))
        .mount(&server)
        .await;
    let m = model(&format!("{}/v1", server.uri()), "opencode-go", "m", false, None);
    let mut opts = options();
    opts.max_retries = Some(2);
    let (message, _) = run(&m, &hi(), opts).await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.text(), "ok");
    assert_eq!(server.received_requests().await.unwrap().len(), 2);

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "277403")
                .set_body_json(json!({"error": {"message": "rate limited"}})),
        )
        .mount(&server)
        .await;
    let m = model(&format!("{}/v1", server.uri()), "opencode-go", "m", false, None);
    let mut opts = options();
    opts.max_retries = Some(2);
    opts.max_retry_delay = Some(Duration::from_secs(1));
    let (message, _) = run(&m, &hi(), opts).await;
    let error = message.error_message.unwrap();
    assert!(
        error.contains("Server requested 277403s retry delay (max: 1s)"),
        "{error}"
    );
    assert!(error.contains("rate limited"));
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn on_payload_replaces_the_body_and_on_response_sees_headers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(sse_response(sse(&[stop_chunk()])).insert_header("x-request-id", "req-1"))
        .mount(&server)
        .await;
    let m = model(&format!("{}/v1", server.uri()), "openai", "gpt-4o-mini", false, None);
    let seen: Arc<Mutex<Option<ProviderResponse>>> = Arc::default();
    let mut opts = options();
    opts.on_payload = Some(Arc::new(|payload: &Value, _: &Model| {
        let mut payload = payload.clone();
        payload["temperature"] = json!(0.1);
        Box::pin(async move { Ok(Some(payload)) })
    }));
    let sink = seen.clone();
    opts.on_response = Some(Arc::new(move |response: &ProviderResponse, _: &Model| {
        *sink.lock().unwrap() = Some(response.clone());
        Box::pin(async { Ok(()) })
    }));
    run(&m, &hi(), opts).await;
    let response = seen.lock().unwrap().clone().unwrap();
    assert_eq!(response.status, 200);
    assert_eq!(response.headers.get("x-request-id").map(String::as_str), Some("req-1"));
    assert_eq!(
        last_request(&server).await.body_json::<Value>().unwrap()["temperature"],
        0.1
    );
}

#[tokio::test]
async fn stream_simple_maps_reasoning_tool_choice_and_max_tokens() {
    let server = serve(&[stop_chunk()]).await;
    let m = model(&format!("{}/v1", server.uri()), "openai", "o3", true, None);
    let options = SimpleStreamOptions {
        base: options(),
        reasoning: Some(ThinkingLevel::Max),
        tool_choice: Some(ToolChoice::None),
        ..SimpleStreamOptions::default()
    };
    let (message, _) = collect(OpenAICompletionsApi.stream_simple(&m, &hi(), options)).await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    let body = last_request(&server).await.body_json::<Value>().unwrap();
    // `max` clamps to `high`, the highest level the model supports.
    assert_eq!(body["reasoning_effort"], "high");
    assert_eq!(body["tool_choice"], "none");
    assert_eq!(body["max_completion_tokens"], 4096);

    let api = OpenAICompletionsOptions {
        reasoning_effort: Some(ReasoningEffort::Low),
        ..OpenAICompletionsOptions::default()
    };
    let mut opts = self::options();
    opts.api = Some(ApiOptions::OpenAICompletions(api));
    collect(OpenAICompletionsApi.stream(&m, &hi(), opts)).await;
    let body = last_request(&server).await.body_json::<Value>().unwrap();
    assert_eq!(body["reasoning_effort"], "low");
    assert!(body.get("max_completion_tokens").is_none());
}

#[tokio::test]
async fn replays_foreign_messages_images_and_orphaned_calls() {
    let server = serve(&[stop_chunk()]).await;
    let m = model(&format!("{}/v1", server.uri()), "openai", "gpt-4o", false, None);
    let mut foreign = AssistantMessage::empty("anthropic-messages", "anthropic", "claude");
    foreign.stop_reason = StopReason::ToolUse;
    foreign.content = vec![
        AssistantBlock::Thinking(ThinkingContent {
            thinking: "secret".into(),
            thinking_signature: Some("sig".into()),
            redacted: None,
        }),
        AssistantBlock::Text(TextContent {
            text: "Let me look.".into(),
            text_signature: Some("x".into()),
        }),
        AssistantBlock::ToolCall(ToolCall {
            id: "toolu_1".into(),
            name: "read".into(),
            arguments: json!({"path": "a.png"}).as_object().cloned().unwrap(),
            ..ToolCall::default()
        }),
    ];
    let mut aborted = AssistantMessage::empty("openai-completions", "openai", "gpt-4o");
    aborted.stop_reason = StopReason::Aborted;
    aborted.content = vec![AssistantBlock::Text(TextContent::new("half"))];
    let image = tau_ai::UserBlock::Image(tau_ai::ImageContent {
        data: "AAAA".into(),
        mime_type: "image/png".into(),
    });
    let context = normalize_context(
        &Context::new(vec![
            Message::User(tau_ai::UserMessage {
                content: tau_ai::UserContent::Blocks(vec![tau_ai::UserBlock::text("look"), image]),
                timestamp: 1,
            }),
            Message::Assistant(foreign),
            Message::Assistant(aborted),
            Message::user("continue"),
        ])
        .with_tools(vec![read_tool()]),
    );
    let (message, _) = run(&m, &context, options()).await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    let body = last_request(&server).await.body_json::<Value>().unwrap();
    assert_eq!(
        body["messages"],
        json!([
            {"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]},
            {"role": "assistant", "content": "Let me look.", "tool_calls": [
                {"id": "toolu_1", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"a.png\"}"}}
            ]},
            {"role": "tool", "content": "No result provided", "tool_call_id": "toolu_1"},
            {"role": "user", "content": "continue"}
        ])
    );
}

/// A one-shot HTTP server that sends `first` as an SSE event and then holds the response
/// open until the test ends.
fn hanging_server(first: &Value) -> (String, std::sync::mpsc::Sender<()>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (release, wait) = std::sync::mpsc::channel::<()>();
    let event = format!("data: {first}\n\n");
    std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        let mut buffer = vec![0u8; 65536];
        let _ = socket.read(&mut buffer);
        let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n";
        let chunk = format!("{:x}\r\n{event}\r\n", event.len());
        socket.write_all(head.as_bytes()).unwrap();
        socket.write_all(chunk.as_bytes()).unwrap();
        socket.flush().unwrap();
        let _ = wait.recv_timeout(Duration::from_secs(10));
    });
    (format!("http://{address}/v1"), release)
}

#[tokio::test]
async fn abort_before_and_during_the_stream() {
    let m = model("http://127.0.0.1:9/v1", "openai", "gpt-4o-mini", false, None);
    let cancel = CancellationToken::new();
    cancel.cancel();
    let mut opts = options();
    opts.cancel = Some(cancel);
    let (message, events) = run(&m, &hi(), opts).await;
    assert_eq!(kinds(&events), ["error"]);
    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert_eq!(message.error_message.as_deref(), Some("Request was aborted"));

    let (base_url, release) = hanging_server(&json!({"id": "c", "choices": [{"delta": {"content": "hello"}}]}));
    let m = model(&base_url, "openai", "gpt-4o-mini", false, None);
    let cancel = CancellationToken::new();
    let mut opts = options();
    opts.cancel = Some(cancel.clone());
    let mut stream = openai_completions::stream(&m, &hi(), opts);
    let mut events = Vec::new();
    while let Some(event) = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .unwrap()
    {
        if event.kind() == "text_delta" {
            cancel.cancel();
        }
        events.push(event);
    }
    drop(release);
    assert_eq!(
        kinds(&events),
        ["start", "text_start", "text_delta", "text_end", "error"]
    );
    let message = events.last().unwrap().final_message().unwrap();
    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert_eq!(message.error_message.as_deref(), Some("Request was aborted"));
    assert_eq!(message.content, vec![AssistantBlock::Text(TextContent::new("hello"))]);
}
