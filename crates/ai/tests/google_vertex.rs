//! End-to-end tests of the `google-vertex` API through a stub `PredictionService` transport
//! (`google_cloud_aiplatform_v1::stub`), so no request leaves the process.
//!
//! Ports of upstream `test/google-raw-stop-reason.test.ts`, `google-thinking-level-map.test.ts`
//! and `google-vertex-api-key-resolution.test.ts` (which mock `@google/genai`), plus the
//! Vertex checks of pi-py's `test_google_streams.py`.
#![cfg(feature = "vertex")]
#![allow(clippy::unwrap_used, clippy::too_many_lines)]

use futures::StreamExt;
use futures::future::BoxFuture;
use google_cloud_aiplatform_v1::client::PredictionService;
use google_cloud_aiplatform_v1::model::function_calling_config::Mode;
use google_cloud_aiplatform_v1::model::generation_config::thinking_config::ThinkingLevel as ProtoThinkingLevel;
use google_cloud_aiplatform_v1::model::{GenerateContentRequest, GenerateContentResponse};
use google_cloud_gax::error::rpc::{Code, Status};
use google_cloud_gax::options::RequestOptions;
use google_cloud_gax::streaming::ResponseStream;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use delta_ai::api::google_vertex::{GoogleVertexApi, stream_simple, stream_with_client};
use delta_ai::options::{
    ApiOptions, CallbackError, GoogleOptions, GoogleThinking, GoogleThinkingLevel, GoogleToolChoice,
    GoogleVertexOptions, OnPayload, OnProviderStreamEvent, ThinkingBudgets,
};
use delta_ai::types::{AssistantBlock, ImageContent, TextContent, ThinkingContent, ToolCall, ToolResultMessage};
use delta_ai::{
    AssistantMessage, AssistantMessageEvent, Context, Message, Model, ProviderStreams, SimpleStreamOptions, StopReason,
    StreamOptions, ThinkingLevel, Tool, TranscriptContext, UserBlock, UserContent, UserMessage, normalize_context,
};

// ---------------------------------------------------------------------------
// Stub transport
// ---------------------------------------------------------------------------

type SdkResult<T> = google_cloud_aiplatform_v1::Result<T>;

fn sdk_error(code: Code, message: &str) -> google_cloud_aiplatform_v1::Error {
    google_cloud_aiplatform_v1::Error::service(Status::default().set_code(code).set_message(message))
}

/// What the stub answers to one `stream_generate_content` call.
#[derive(Debug)]
enum Reply {
    /// The call itself fails (like an HTTP error status).
    Fail(Code, &'static str),
    /// A stream of items, then the end of the stream.
    Stream(Vec<SdkResult<GenerateContentResponse>>),
    /// A stream of items that then stays open.
    Hang(Vec<GenerateContentResponse>),
}

#[derive(Debug, Default)]
struct Stub {
    requests: Mutex<Vec<GenerateContentRequest>>,
    replies: Mutex<VecDeque<Reply>>,
    open: Mutex<Vec<mpsc::Sender<SdkResult<GenerateContentResponse>>>>,
}

impl Stub {
    fn new(replies: Vec<Reply>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies.into()),
            ..Self::default()
        })
    }

    fn requests(&self) -> Vec<GenerateContentRequest> {
        self.requests.lock().unwrap().clone()
    }

    fn last_request(&self) -> GenerateContentRequest {
        self.requests().pop().unwrap()
    }
}

fn response_stream(
    items: Vec<SdkResult<GenerateContentResponse>>,
) -> (
    mpsc::Sender<SdkResult<GenerateContentResponse>>,
    ResponseStream<GenerateContentResponse>,
) {
    let (sender, receiver) = mpsc::channel(items.len().max(1));
    for item in items {
        sender.try_send(item).unwrap();
    }
    (sender, ResponseStream::from(receiver))
}

impl google_cloud_aiplatform_v1::stub::PredictionService for Stub {
    fn stream_generate_content(
        &self,
        request: GenerateContentRequest,
        _options: RequestOptions,
    ) -> impl Future<Output = SdkResult<ResponseStream<GenerateContentResponse>>> + Send {
        self.requests.lock().unwrap().push(request);
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(Reply::Fail(Code::NotFound, "no stub reply"));
        let result = match reply {
            Reply::Fail(code, message) => Err(sdk_error(code, message)),
            Reply::Stream(items) => Ok(response_stream(items).1),
            Reply::Hang(items) => {
                let (sender, stream) = response_stream(items.into_iter().map(Ok).collect());
                self.open.lock().unwrap().push(sender);
                Ok(stream)
            }
        };
        std::future::ready(result)
    }
}

fn client(stub: &Arc<Stub>) -> PredictionService {
    PredictionService::from_stub::<Stub>(stub.clone())
}

/// A proto response from its REST JSON form.
fn chunk(value: Value) -> GenerateContentResponse {
    serde_json::from_value(value).unwrap()
}

fn chunks(values: &[Value]) -> Reply {
    Reply::Stream(values.iter().cloned().map(|value| Ok(chunk(value))).collect())
}

fn stop_chunk(finish_reason: &str, function_call: bool) -> Value {
    let mut candidate = json!({"finishReason": finish_reason});
    if function_call {
        candidate["content"] = json!({"parts": [{"functionCall": {"name": "echo", "args": {"value": "truncated"}}}]});
    }
    json!({
        "responseId": "vertex-response-id",
        "candidates": [candidate],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
    })
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn model(id: &str) -> Model {
    serde_json::from_value(json!({
        "id": id, "name": id, "api": "google-vertex", "provider": "google-vertex",
        "baseUrl": "https://{location}-aiplatform.googleapis.com", "reasoning": true, "input": ["text", "image"],
        "cost": {"input": 0.3, "output": 2.5, "cacheRead": 0.03, "cacheWrite": 0},
        "contextWindow": 1_048_576, "maxTokens": 65536
    }))
    .unwrap()
}

fn custom_model(id: &str, thinking_level_map: &Value) -> Model {
    serde_json::from_value(json!({
        "id": id, "name": id, "api": "google-vertex", "provider": "test-vertex",
        "baseUrl": "https://example.invalid/v1", "reasoning": true, "input": ["text"],
        "thinkingLevelMap": thinking_level_map,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 128_000, "maxTokens": 4096
    }))
    .unwrap()
}

fn hello() -> TranscriptContext {
    normalize_context(&Context::new(vec![Message::user("hello")]))
}

/// ADC options with an explicit project and location (no environment lookups).
fn options() -> StreamOptions {
    StreamOptions {
        max_retries: Some(0),
        api: Some(ApiOptions::GoogleVertex(GoogleVertexOptions {
            project: Some("test-project".into()),
            location: Some("us-central1".into()),
            ..GoogleVertexOptions::default()
        })),
        ..StreamOptions::default()
    }
}

fn with_vertex(mut opts: StreamOptions, update: impl FnOnce(&mut GoogleVertexOptions)) -> StreamOptions {
    if let Some(ApiOptions::GoogleVertex(vertex)) = &mut opts.api {
        update(vertex);
    }
    opts
}

const MODEL_NAME: &str = "projects/test-project/locations/us-central1/publishers/google/models/gemini-2.5-flash";

async fn collect(mut events: delta_ai::AssistantMessageEventStream) -> (Vec<AssistantMessageEvent>, AssistantMessage) {
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

async fn run(
    id: &str,
    context: &TranscriptContext,
    opts: StreamOptions,
    replies: Vec<Reply>,
) -> (Arc<Stub>, AssistantMessage) {
    let stub = Stub::new(replies);
    let message = stream_with_client(&model(id), context, opts, client(&stub))
        .result()
        .await;
    (stub, message)
}

/// The proto request as JSON (REST field names, base64 bytes, numeric enums).
fn request_json(request: &GenerateContentRequest) -> Value {
    serde_json::to_value(request).unwrap()
}

// ---------------------------------------------------------------------------
// Text, thinking, tool calls, usage
// ---------------------------------------------------------------------------

#[tokio::test]
async fn text_and_thinking_stream() {
    let stub = Stub::new(vec![chunks(&[
        json!({"responseId": "resp_1", "candidates": [{"content": {"role": "model", "parts": [
            {"text": "Let me ", "thought": true, "thoughtSignature": "c2ln"}]}}]}),
        json!({"candidates": [{"content": {"parts": [{"text": "think.", "thought": true}]}}]}),
        json!({"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}),
        json!({"candidates": [{"content": {"parts": [{"text": " world", "thoughtSignature": "dGV4dA=="}]},
            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "cachedContentTokenCount": 4, "candidatesTokenCount": 5,
                "thoughtsTokenCount": 3, "totalTokenCount": 18}}),
    ])]);
    let m = model("gemini-2.5-flash");
    let (events, message) = collect(stream_with_client(&m, &hello(), options(), client(&stub))).await;
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
    assert_eq!(message.api, "google-vertex");
    assert_eq!(message.provider, "google-vertex");
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
    let expected_cost = (6.0 * 0.3 + 8.0 * 2.5 + 4.0 * 0.03) / 1_000_000.0;
    assert!((usage.cost.total - expected_cost).abs() < 1e-12);

    let requests = stub.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        request_json(&requests[0]),
        json!({
            "model": MODEL_NAME,
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "generationConfig": {}
        })
    );
}

#[tokio::test]
async fn tool_calls_generated_ids_and_request_shape() {
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
    let mut opts = with_vertex(options(), |vertex| vertex.tool_choice = Some(GoogleToolChoice::Auto));
    opts.temperature = Some(0.25);
    opts.max_tokens = Some(512);
    let stub = Stub::new(vec![chunks(&[
        json!({"candidates": [{"content": {"parts": [{"text": "Calling"}]}}]}),
        json!({"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "bash", "args": {"command": "ls"}}, "thoughtSignature": "c2ln"},
            {"functionCall": {"name": "bash", "args": {"command": "pwd"}}},
            {"functionCall": {"name": "bash"}}
        ]}, "finishReason": "STOP"}]}),
    ])]);
    let m = model("gemini-2.5-flash");
    let (events, message) = collect(stream_with_client(&m, &context, opts, client(&stub))).await;
    assert_eq!(message.stop_reason, StopReason::ToolUse);
    let calls: Vec<&ToolCall> = message.tool_calls().collect();
    assert_eq!(calls.len(), 3);
    // The v1 proto has no call ids: every call gets a generated, unique id.
    for call in &calls {
        assert!(call.id.starts_with("bash_"), "{}", call.id);
    }
    assert_ne!(calls[0].id, calls[1].id);
    assert_ne!(calls[1].id, calls[2].id);
    assert_eq!(calls[0].thought_signature.as_deref(), Some("c2ln"));
    assert_eq!(calls[1].thought_signature, None);
    assert_eq!(Value::Object(calls[0].arguments.clone()), json!({"command": "ls"}));
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

    let request = stub.last_request();
    assert_eq!(
        request
            .tool_config
            .as_ref()
            .unwrap()
            .function_calling_config
            .as_ref()
            .unwrap()
            .mode,
        Mode::Auto
    );
    let mut body = request_json(&request);
    body.as_object_mut().unwrap().remove("toolConfig");
    assert_eq!(
        body,
        json!({
            "model": MODEL_NAME,
            "contents": [{"role": "user", "parts": [{"text": "go"}]}],
            "systemInstruction": {"role": "user", "parts": [{"text": "sys"}]},
            "tools": [{"functionDeclarations": [{"name": "bash", "description": "Run a command",
                "parametersJsonSchema": {"type": "object", "properties": {"command": {"type": "string"}}}}]}],
            "generationConfig": {"temperature": 0.25, "maxOutputTokens": 512}
        })
    );
}

// ---------------------------------------------------------------------------
// Finish reasons (google-raw-stop-reason.test.ts)
// ---------------------------------------------------------------------------

async fn run_chunks(values: &[Value]) -> AssistantMessage {
    run("gemini-3-flash-preview", &hello(), options(), vec![chunks(values)])
        .await
        .1
}

#[tokio::test]
async fn raw_finish_reasons() {
    let message = run_chunks(&[stop_chunk("SAFETY", false)]).await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("SAFETY"));
    assert_eq!(message.error_message.as_deref(), Some("Provider stopped with: SAFETY"));

    let message = run_chunks(&[stop_chunk("MALFORMED_FUNCTION_CALL", false)]).await;
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
    assert_eq!(message.response_id.as_deref(), Some("vertex-response-id"));
    assert_eq!((message.usage.input, message.usage.output), (1, 1));
}

#[tokio::test]
async fn missing_finish_reason_and_empty_stream() {
    let message = run_chunks(&[json!({"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})]).await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Google Vertex stream ended without a finish reason")
    );
    assert_eq!(message.text(), "hi");

    let message = run_chunks(&[]).await;
    assert_eq!(
        message.error_message.as_deref(),
        Some("Google Vertex stream ended without a finish reason")
    );
}

#[tokio::test]
async fn usage_waits_for_prompt_token_count() {
    let message = run_chunks(&[
        json!({"candidates": [{"content": {"parts": [{"text": "a"}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1, "totalTokenCount": 6}}),
        json!({"candidates": [{"content": {"parts": [{"text": "b"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"candidatesTokenCount": 2}}),
    ])
    .await;
    assert_eq!(message.text(), "ab");
    assert_eq!(
        (message.usage.input, message.usage.output, message.usage.total_tokens),
        (5, 1, 6)
    );
}

#[tokio::test]
async fn forwards_provider_stream_events_in_order() {
    let values = [
        json!({"responseId": "resp_google", "candidates": [{"content": {"role": "model", "parts": [{"text": "hello"}]}}]}),
        json!({"candidates": [{"finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3}}),
    ];
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
    let mut opts = options();
    opts.on_provider_stream_event = Some(hook);
    let (_, message) = run("gemini-3-flash-preview", &hello(), opts, vec![chunks(&values)]).await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.response_id.as_deref(), Some("resp_google"));
    assert_eq!(message.content, vec![AssistantBlock::Text(TextContent::new("hello"))]);
    let received = received.lock().unwrap().clone();
    assert_eq!(
        received,
        values
            .iter()
            .map(|value| (value.clone(), "gemini-3-flash-preview".to_owned()))
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Errors, retries, abort
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sdk_errors_carry_status_and_message() {
    let (stub, message) = run(
        "gemini-2.5-flash",
        &hello(),
        options(),
        vec![Reply::Fail(
            Code::InvalidArgument,
            "Request contains an invalid argument.",
        )],
    )
    .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("400 Request contains an invalid argument.")
    );
    assert!(message.content.is_empty());
    assert_eq!(stub.requests().len(), 1);

    let (_, message) = run(
        "gemini-2.5-flash",
        &hello(),
        options(),
        vec![Reply::Fail(Code::ResourceExhausted, "Quota exceeded")],
    )
    .await;
    assert_eq!(message.error_message.as_deref(), Some("429 Quota exceeded"));
}

#[tokio::test(start_paused = true)]
async fn retries_retryable_status() {
    let mut opts = options();
    opts.max_retries = Some(1);
    let (stub, message) = run(
        "gemini-2.5-flash",
        &hello(),
        opts.clone(),
        vec![
            Reply::Fail(Code::Unavailable, "overloaded"),
            chunks(&[stop_chunk("STOP", false)]),
        ],
    )
    .await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    assert_eq!(stub.requests().len(), 2);

    // Non-retryable codes fail at once.
    let (stub, message) = run(
        "gemini-2.5-flash",
        &hello(),
        opts,
        vec![
            Reply::Fail(Code::PermissionDenied, "denied"),
            chunks(&[stop_chunk("STOP", false)]),
        ],
    )
    .await;
    assert_eq!(message.error_message.as_deref(), Some("403 denied"));
    assert_eq!(stub.requests().len(), 1);
}

#[tokio::test]
async fn in_stream_error_is_reported_and_not_retried() {
    let stub = Stub::new(vec![
        Reply::Stream(vec![
            Ok(chunk(json!({"candidates": [{"content": {"parts": [{"text": "x"}]}}]}))),
            Err(sdk_error(Code::ResourceExhausted, "Resource exhausted")),
        ]),
        chunks(&[stop_chunk("STOP", false)]),
    ]);
    let mut opts = options();
    opts.max_retries = Some(2);
    let (events, message) = collect(stream_with_client(
        &model("gemini-2.5-flash"),
        &hello(),
        opts,
        client(&stub),
    ))
    .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_message.as_deref(), Some("429 Resource exhausted"));
    assert_eq!(message.text(), "x");
    assert_eq!(kinds(&events), ["start", "text_start", "text_delta", "error"]);
    assert_eq!(stub.requests().len(), 1);
}

#[tokio::test]
async fn cancelled_before_the_request() {
    let cancel = CancellationToken::new();
    cancel.cancel();
    let mut opts = options();
    opts.cancel = Some(cancel);
    let stub = Stub::new(vec![chunks(&[stop_chunk("STOP", false)])]);
    let (events, message) = collect(stream_with_client(
        &model("gemini-2.5-flash"),
        &hello(),
        opts,
        client(&stub),
    ))
    .await;
    assert_eq!(kinds(&events), ["error"]);
    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert_eq!(message.error_message.as_deref(), Some("Request was aborted"));
    assert!(stub.requests().is_empty());
}

#[tokio::test]
async fn abort_mid_stream_keeps_partial_content() {
    let stub = Stub::new(vec![Reply::Hang(vec![chunk(
        json!({"candidates": [{"content": {"parts": [{"text": "partial"}]}}]}),
    )])]);
    let cancel = CancellationToken::new();
    let mut opts = options();
    opts.cancel = Some(cancel.clone());
    let mut events = stream_with_client(&model("gemini-2.5-flash"), &hello(), opts, client(&stub));
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
    let message = final_message.unwrap();
    assert_eq!(kinds_seen, ["start", "text_start", "text_delta", "error"]);
    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert_eq!(message.error_message.as_deref(), Some("Request was aborted"));
    assert_eq!(message.text(), "partial");
}

// ---------------------------------------------------------------------------
// Authentication and client configuration (google-vertex-api-key-resolution.test.ts)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn api_keys_and_adc_markers_select_the_model_name() {
    for api_key in ["<authenticated>", "gcp-vertex-credentials"] {
        let mut opts = options();
        opts.api_key = Some(api_key.into());
        let (stub, message) = run(
            "gemini-2.5-flash",
            &hello(),
            opts,
            vec![chunks(&[stop_chunk("STOP", false)])],
        )
        .await;
        assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
        assert_eq!(stub.last_request().model, MODEL_NAME);
    }
    // A real key uses express mode: no project or location in the model name.
    let opts = StreamOptions {
        api_key: Some("AIzaSyExampleRealisticLookingApiKey123456".into()),
        max_retries: Some(0),
        ..StreamOptions::default()
    };
    let (stub, message) = run(
        "gemini-2.5-flash",
        &hello(),
        opts,
        vec![chunks(&[stop_chunk("STOP", false)])],
    )
    .await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    assert_eq!(stub.last_request().model, "publishers/google/models/gemini-2.5-flash");

    // Project and location from the scoped environment.
    let opts = StreamOptions {
        env: Some(
            [
                ("GOOGLE_CLOUD_PROJECT".to_owned(), "env-project".to_owned()),
                ("GOOGLE_CLOUD_LOCATION".to_owned(), "europe-west4".to_owned()),
            ]
            .into(),
        ),
        max_retries: Some(0),
        ..StreamOptions::default()
    };
    let (stub, _) = run(
        "gemini-2.5-flash",
        &hello(),
        opts,
        vec![chunks(&[stop_chunk("STOP", false)])],
    )
    .await;
    assert_eq!(
        stub.last_request().model,
        "projects/env-project/locations/europe-west4/publishers/google/models/gemini-2.5-flash"
    );
}

#[tokio::test]
async fn missing_location_fails_before_any_request() {
    if std::env::var_os("GOOGLE_CLOUD_LOCATION").is_some() {
        return;
    }
    let opts = with_vertex(options(), |vertex| vertex.location = None);
    let (stub, message) = run("gemini-2.5-flash", &hello(), opts, vec![]).await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options.")
    );
    assert!(stub.requests().is_empty());
}

#[tokio::test]
async fn unreadable_credentials_file_fails_the_stream() {
    let mut opts = options();
    opts.env = Some(
        [(
            "GOOGLE_APPLICATION_CREDENTIALS".to_owned(),
            "/nonexistent/delta-vertex-adc.json".to_owned(),
        )]
        .into(),
    );
    let message = GoogleVertexApi
        .stream(&model("gemini-2.5-flash"), &hello(), opts)
        .result()
        .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert!(
        message
            .error_message
            .as_deref()
            .unwrap()
            .starts_with("Failed to read Google credentials file /nonexistent/delta-vertex-adc.json"),
        "{:?}",
        message.error_message
    );
}

// ---------------------------------------------------------------------------
// Callbacks and options
// ---------------------------------------------------------------------------

#[tokio::test]
async fn on_payload_sees_rest_json_and_may_replace_it() {
    let seen: Arc<Mutex<Option<Value>>> = Arc::default();
    let sink = seen.clone();
    let on_payload: OnPayload = Arc::new(move |payload: &Value, _: &Model| -> BoxFuture<'static, _> {
        *sink.lock().unwrap() = Some(payload.clone());
        let mut payload = payload.clone();
        payload["generationConfig"]["topK"] = json!(3);
        payload["labels"] = json!({"team": "delta"});
        Box::pin(async move { Ok(Some(payload)) })
    });
    let mut opts = options();
    opts.on_payload = Some(on_payload);
    let (stub, message) = run(
        "gemini-2.5-flash",
        &hello(),
        opts,
        vec![chunks(&[stop_chunk("STOP", false)])],
    )
    .await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    assert_eq!(
        seen.lock().unwrap().clone().unwrap(),
        json!({"contents": [{"role": "user", "parts": [{"text": "hello"}]}], "generationConfig": {}})
    );
    let request = stub.last_request();
    assert_eq!(request.generation_config.unwrap().top_k, Some(3.0));
    assert_eq!(request.labels.get("team").map(String::as_str), Some("delta"));
    assert_eq!(request.model, MODEL_NAME);

    // A payload the proto cannot represent fails the request before it is sent.
    let on_payload: OnPayload = Arc::new(|_: &Value, _: &Model| -> BoxFuture<'static, _> {
        Box::pin(async { Ok(Some(json!({"contents": 1}))) })
    });
    let mut opts = options();
    opts.on_payload = Some(on_payload);
    let (stub, message) = run("gemini-2.5-flash", &hello(), opts, vec![]).await;
    assert!(
        message
            .error_message
            .as_deref()
            .unwrap()
            .starts_with("Invalid Google Vertex request payload: ")
    );
    assert!(stub.requests().is_empty());
}

#[tokio::test]
async fn explicit_thinking_options() {
    let opts = with_vertex(options(), |vertex| {
        vertex.tool_choice = Some(GoogleToolChoice::Any);
        vertex.thinking = Some(GoogleThinking {
            enabled: true,
            budget_tokens: Some(-1),
            level: None,
        });
    });
    let (stub, _) = run(
        "gemini-2.5-flash",
        &hello(),
        opts,
        vec![chunks(&[stop_chunk("STOP", false)])],
    )
    .await;
    let request = stub.last_request();
    // No tools: no toolConfig even with a tool choice.
    assert!(request.tool_config.is_none());
    let thinking = request.generation_config.unwrap().thinking_config.unwrap();
    assert_eq!(
        (
            thinking.include_thoughts,
            thinking.thinking_budget,
            thinking.thinking_level
        ),
        (Some(true), Some(-1), None)
    );

    // Disabled thinking on a Gemini 3 model that cannot turn it off: the lowest level.
    let opts = with_vertex(options(), |vertex| {
        vertex.thinking = Some(GoogleThinking::default());
    });
    let stub = Stub::new(vec![chunks(&[stop_chunk("STOP", false)])]);
    let m = custom_model(
        "gemini-3.1-pro-preview",
        &json!({"off": null, "minimal": null, "low": "low", "medium": "medium", "high": "high"}),
    );
    stream_with_client(&m, &hello(), opts, client(&stub)).result().await;
    let thinking = stub.last_request().generation_config.unwrap().thinking_config.unwrap();
    assert_eq!(
        (thinking.include_thoughts, thinking.thinking_level),
        (None, Some(ProtoThinkingLevel::Low))
    );

    // Options of another API are ignored.
    let mut opts = options();
    opts.api = Some(ApiOptions::GoogleGenerativeAi(GoogleOptions {
        tool_choice: None,
        thinking: Some(GoogleThinking {
            enabled: true,
            budget_tokens: None,
            level: Some(GoogleThinkingLevel::High),
        }),
    }));
    opts.api_key = Some("AIzaKey".into());
    let (stub, _) = run(
        "gemini-2.5-flash",
        &hello(),
        opts,
        vec![chunks(&[stop_chunk("STOP", false)])],
    )
    .await;
    assert_eq!(request_json(&stub.last_request())["generationConfig"], json!({}));
}

// ---------------------------------------------------------------------------
// Thinking configuration through stream_simple (google-thinking-level-map.test.ts)
// ---------------------------------------------------------------------------

/// Capture the payload of a simple request (as upstream: an API key, then an `on_payload`
/// error stops the request before anything is sent).
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
    // Extended levels map to Gemini levels.
    let payload = capture_payload(
        &custom_model("gemini-3.7-flash", &json!({"xhigh": "high"})),
        Some(ThinkingLevel::Xhigh),
        None,
    )
    .await;
    assert_eq!(
        thinking_config(&payload),
        &json!({"includeThoughts": true, "thinkingLevel": "HIGH"})
    );
    // Mapped levels select custom token budgets.
    let payload = capture_payload(
        &custom_model("gemini-2.5-flash", &json!({"max": "high"})),
        Some(ThinkingLevel::Max),
        Some(ThinkingBudgets {
            high: Some(4321),
            ..ThinkingBudgets::default()
        }),
    )
    .await;
    assert_eq!(
        thinking_config(&payload),
        &json!({"includeThoughts": true, "thinkingBudget": 4321})
    );
    // Vertex budgets: Flash-Lite uses the 2.5 Flash table.
    let payload = capture_payload(
        &custom_model("gemini-2.5-flash-lite", &json!({})),
        Some(ThinkingLevel::Minimal),
        None,
    )
    .await;
    assert_eq!(
        thinking_config(&payload),
        &json!({"includeThoughts": true, "thinkingBudget": 128})
    );
    // No reasoning: thinking is turned off.
    let payload = capture_payload(&custom_model("gemini-2.5-flash", &json!({})), None, None).await;
    assert_eq!(thinking_config(&payload), &json!({"thinkingBudget": 0}));
}

#[tokio::test]
async fn invalid_thinking_mapping_fails_the_stream() {
    let m = custom_model("gemini-3.7-flash", &json!({"xhigh": "extreme"}));
    let options = SimpleStreamOptions {
        base: options(),
        reasoning: Some(ThinkingLevel::Xhigh),
        ..SimpleStreamOptions::default()
    };
    let (events, message) = collect(stream_simple(&m, &hello(), options)).await;
    assert_eq!(kinds(&events), ["error"]);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Unsupported Google thinking level mapping for test-vertex/gemini-3.7-flash: xhigh -> extreme")
    );
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

#[tokio::test]
async fn thought_signature_round_trip() {
    let stub = Stub::new(vec![
        chunks(&[json!({"candidates": [{"content": {"parts": [
            {"text": "a", "thoughtSignature": "AAAAAAAA/+8="},
            {"functionCall": {"name": "read", "args": {"path": "x"}}, "thoughtSignature": "c2lnMQ=="}
        ]}, "finishReason": "STOP"}]})]),
        chunks(&[stop_chunk("STOP", false)]),
    ]);
    let m = model("gemini-3-flash-preview");
    let first = stream_with_client(&m, &hello(), options(), client(&stub))
        .result()
        .await;
    assert_eq!(first.stop_reason, StopReason::ToolUse);
    let call_id = first.tool_calls().next().unwrap().id.clone();
    let result = ToolResultMessage {
        tool_call_id: call_id.clone(),
        tool_name: "read".into(),
        content: vec![UserBlock::text("contents")],
        ..ToolResultMessage::default()
    };
    let context = normalize_context(&Context::new(vec![
        Message::user("hello"),
        Message::Assistant(first),
        Message::ToolResult(result),
    ]));
    let message = stream_with_client(&m, &context, options(), client(&stub))
        .result()
        .await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    let request = stub.last_request();
    // Signatures go back as the original bytes.
    assert_eq!(
        request.contents[1].parts[0].thought_signature.as_ref(),
        [0u8, 0, 0, 0, 0, 0, 0xff, 0xef]
    );
    // Gemini 3 gets call ids in the REST body; the v1 proto has no such field, so they stay
    // unknown fields of the request (the gRPC transport does not send them).
    assert_eq!(
        request_json(&request)["contents"],
        json!([
            {"role": "user", "parts": [{"text": "hello"}]},
            {"role": "model", "parts": [
                {"text": "a", "thoughtSignature": "AAAAAAAA/+8="},
                {"functionCall": {"name": "read", "args": {"path": "x"}, "id": call_id}, "thoughtSignature": "c2lnMQ=="}
            ]},
            {"role": "user", "parts": [{"functionResponse": {"name": "read", "response": {"output": "contents"},
                "id": call_id}}]}
        ])
    );
}

#[tokio::test]
async fn cross_provider_replay_images_and_orphans() {
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
    let (stub, message) = run(
        "gemini-3-flash-preview",
        &context,
        options(),
        vec![chunks(&[stop_chunk("STOP", false)])],
    )
    .await;
    assert_eq!(message.stop_reason, StopReason::Stop, "{:?}", message.error_message);
    let body = request_json(&stub.last_request());
    assert_eq!(
        body["contents"],
        json!([
            {"role": "user", "parts": [{"text": "what is this?"},
                {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}]},
            // Foreign thinking is dropped and signatures are not replayed. Ids are normalized
            // for Gemini 3 (unknown proto fields, not sent over gRPC).
            {"role": "model", "parts": [
                {"text": "Let me look."},
                {"functionCall": {"name": "read", "args": {"path": "a.png"}, "id": "toolu_01_x"}},
                {"functionCall": {"name": "read", "args": {}, "id": "toolu_02"}}
            ]},
            // Gemini 3: the image is nested in the function response; the orphaned call gets a
            // synthetic error result in the same turn.
            {"role": "user", "parts": [
                {"functionResponse": {"name": "read", "response": {"output": "(see attached image)"},
                    "parts": [{"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}], "id": "toolu_01_x"}},
                {"functionResponse": {"name": "read", "response": {"error": "No result provided"}, "id": "toolu_02"}}
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
    let context = normalize_context(
        &Context::new(vec![
            Message::user("one"),
            Message::System(delta_ai::SystemMessage {
                content: "Also be kind.".into(),
                tools_added: Some(vec![Tool::new("late", "Late tool", json!({"type": "object"}))]),
                ..delta_ai::SystemMessage::default()
            }),
            Message::user("two"),
        ])
        .with_system_prompt("Be brief."),
    );
    let (stub, _) = run(
        "gemini-2.5-flash",
        &context,
        options(),
        vec![chunks(&[stop_chunk("STOP", false)])],
    )
    .await;
    let body = request_json(&stub.last_request());
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
