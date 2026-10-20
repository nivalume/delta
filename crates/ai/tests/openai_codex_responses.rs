//! End-to-end tests of the `openai-codex-responses` API. Ported from upstream
//! `test/openai-codex-stream.test.ts` (plus the Codex cases of `pre-generation-error.test.ts`).
//!
//! Upstream stubs `fetch` and the global `WebSocket`. Here one local server answers both on
//! the same port: WebSocket upgrades go to a tokio-tungstenite server, other requests get
//! scripted HTTP responses (SSE bodies that can stay open or arrive slowly). The retry tests
//! run on Tokio's paused clock instead of fake timers. The 55-minute connection age case is a
//! unit test in `src/api/openai_codex_responses.rs`, where the cache entry's age can be set.

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

use base64::Engine as _;
use delta_ai::api::openai_codex_responses::{
    self, close_openai_codex_websocket_sessions, get_openai_codex_websocket_debug_stats,
    reset_openai_codex_websocket_debug_stats,
};
use delta_ai::options::{OpenAICodexResponsesOptions, ReasoningEffort, Transport};
use delta_ai::session_resources::cleanup_session_resources;
use delta_ai::{
    ApiOptions, AssistantMessage, AssistantMessageEvent, CacheRetention, Context, Message, Model, SimpleStreamOptions,
    StopReason, StreamOptions, ThinkingLevel, Tool, TranscriptContext, normalize_context,
};
use futures::future::BoxFuture;
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Read as _;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::tungstenite::handshake::server::{Request as WsRequest, Response as WsResponse};
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Test server: scripted HTTP and WebSocket on one port
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct HttpRequest {
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl HttpRequest {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(&name.to_ascii_lowercase()).map(String::as_str)
    }

    /// The JSON body, zstd-decoded when the request says so.
    fn json(&self) -> Value {
        let body = if self.header("content-encoding") == Some("zstd") {
            let mut decoder = ruzstd::decoding::StreamingDecoder::new(&self.body[..]).unwrap();
            let mut decoded = Vec::new();
            decoder.read_to_end(&mut decoded).unwrap();
            decoded
        } else {
            self.body.clone()
        };
        serde_json::from_slice(&body).unwrap()
    }
}

enum HttpReply {
    /// A complete response.
    Full {
        status: u16,
        headers: Vec<(&'static str, String)>,
        body: String,
    },
    /// A chunked SSE body: each chunk after its delay; the body stays open when
    /// `keep_open` is set.
    Stream {
        chunks: Vec<(u64, String)>,
        keep_open: bool,
    },
    /// Never answer.
    Hang,
}

fn sse_reply(body: String) -> HttpReply {
    HttpReply::Stream {
        chunks: vec![(0, body)],
        keep_open: false,
    }
}

enum WsReply {
    Events(Vec<Value>),
    Silent,
    Close(u16, &'static str),
}

type HttpHandler = Box<dyn Fn(usize, &HttpRequest) -> HttpReply + Send + Sync>;
/// `(connection index, message index on that connection, message)`.
type WsHandler = Box<dyn Fn(usize, usize, &Value) -> WsReply + Send + Sync>;

#[derive(Default)]
struct Recorded {
    http: Vec<HttpRequest>,
    ws_headers: Vec<HashMap<String, String>>,
    /// `(connection index, message)`.
    ws_messages: Vec<(usize, Value)>,
}

struct ServerState {
    http_handler: HttpHandler,
    ws_handler: WsHandler,
    /// Connections whose WebSocket handshake is never answered.
    hang_handshake: bool,
    recorded: Mutex<Recorded>,
    ws_connections: AtomicUsize,
    ws_closed: AtomicUsize,
    /// HTTP connections the client closed while the body was still streaming.
    http_cancelled: AtomicUsize,
}

struct CodexServer {
    addr: SocketAddr,
    state: Arc<ServerState>,
}

impl CodexServer {
    async fn start(http_handler: HttpHandler, ws_handler: WsHandler) -> Self {
        Self::start_with(http_handler, ws_handler, false).await
    }

    async fn start_with(http_handler: HttpHandler, ws_handler: WsHandler, hang_handshake: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let state = Arc::new(ServerState {
            http_handler,
            ws_handler,
            hang_handshake,
            recorded: Mutex::new(Recorded::default()),
            ws_connections: AtomicUsize::new(0),
            ws_closed: AtomicUsize::new(0),
            http_cancelled: AtomicUsize::new(0),
        });
        let accept_state = state.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                tokio::spawn(handle_connection(stream, accept_state.clone()));
            }
        });
        Self { addr, state }
    }

    fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn http_requests(&self) -> Vec<HttpRequest> {
        self.state.recorded.lock().unwrap().http.clone()
    }

    fn ws_messages(&self) -> Vec<(usize, Value)> {
        self.state.recorded.lock().unwrap().ws_messages.clone()
    }

    fn ws_headers(&self) -> Vec<HashMap<String, String>> {
        self.state.recorded.lock().unwrap().ws_headers.clone()
    }

    fn ws_connections(&self) -> usize {
        self.state.ws_connections.load(Ordering::SeqCst)
    }

    /// Wait until the client closed `count` WebSockets (closes happen in the background).
    async fn wait_for_ws_closed(&self, count: usize) {
        for _ in 0..200 {
            if self.state.ws_closed.load(Ordering::SeqCst) >= count {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!(
            "expected {count} closed WebSockets, saw {}",
            self.state.ws_closed.load(Ordering::SeqCst)
        );
    }
}

fn no_http() -> HttpHandler {
    Box::new(|_, _| HttpReply::Full {
        status: 500,
        headers: Vec::new(),
        body: "unexpected fetch".into(),
    })
}

fn no_ws() -> WsHandler {
    Box::new(|_, _, _| WsReply::Close(1011, "unexpected websocket"))
}

/// Peek at the request head without consuming it.
async fn peek_head(stream: &TcpStream) -> Option<String> {
    let mut buffer = vec![0u8; 64 * 1024];
    for _ in 0..1000 {
        let read = stream.peek(&mut buffer).await.ok()?;
        if read == 0 {
            return None;
        }
        let text = String::from_utf8_lossy(&buffer[..read]).into_owned();
        if text.contains("\r\n\r\n") {
            return Some(text);
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    None
}

async fn handle_connection(stream: TcpStream, state: Arc<ServerState>) {
    let Some(head) = peek_head(&stream).await else {
        return;
    };
    if head.to_ascii_lowercase().contains("upgrade: websocket") {
        handle_websocket(stream, state).await;
    } else {
        handle_http(stream, state).await;
    }
}

async fn handle_websocket(stream: TcpStream, state: Arc<ServerState>) {
    let connection = state.ws_connections.fetch_add(1, Ordering::SeqCst);
    if state.hang_handshake {
        let mut stream = stream;
        let mut sink = [0u8; 1024];
        while matches!(stream.read(&mut sink).await, Ok(read) if read > 0) {}
        return;
    }
    let recorder = state.clone();
    #[allow(clippy::result_large_err)] // The handshake callback's signature.
    let callback = move |request: &WsRequest, response: WsResponse| {
        let headers = request
            .headers()
            .iter()
            .map(|(name, value)| (name.as_str().to_owned(), value.to_str().unwrap().to_owned()))
            .collect();
        recorder.recorded.lock().unwrap().ws_headers.push(headers);
        Ok(response)
    };
    let Ok(mut socket) = tokio_tungstenite::accept_hdr_async(stream, callback).await else {
        return;
    };
    let mut index = 0;
    while let Some(Ok(message)) = socket.next().await {
        match message {
            WsMessage::Text(text) => {
                let value: Value = serde_json::from_str(&text).unwrap();
                state
                    .recorded
                    .lock()
                    .unwrap()
                    .ws_messages
                    .push((connection, value.clone()));
                let reply = (state.ws_handler)(connection, index, &value);
                index += 1;
                match reply {
                    WsReply::Events(events) => {
                        for event in events {
                            if socket.send(WsMessage::text(event.to_string())).await.is_err() {
                                return;
                            }
                        }
                    }
                    WsReply::Silent => {}
                    WsReply::Close(code, reason) => {
                        let frame = tokio_tungstenite::tungstenite::protocol::CloseFrame {
                            code: code.into(),
                            reason: reason.into(),
                        };
                        let _ = socket.close(Some(frame)).await;
                    }
                }
            }
            WsMessage::Close(_) => {
                state.ws_closed.fetch_add(1, Ordering::SeqCst);
                break;
            }
            _ => {}
        }
    }
}

async fn read_http_request(stream: &mut TcpStream) -> Option<HttpRequest> {
    let mut data = Vec::new();
    let mut chunk = [0u8; 8192];
    let head_end = loop {
        if let Some(index) = data.windows(4).position(|window| window == b"\r\n\r\n") {
            break index;
        }
        let read = stream.read(&mut chunk).await.ok()?;
        if read == 0 {
            return None;
        }
        data.extend_from_slice(&chunk[..read]);
    };
    let head = String::from_utf8_lossy(&data[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let path = lines.next()?.split_whitespace().nth(1)?.to_owned();
    let headers: HashMap<String, String> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_owned()))
        .collect();
    let length: usize = headers
        .get("content-length")
        .and_then(|length| length.parse().ok())
        .unwrap_or(0);
    let mut body = data[head_end + 4..].to_vec();
    while body.len() < length {
        let read = stream.read(&mut chunk).await.ok()?;
        if read == 0 {
            return None;
        }
        body.extend_from_slice(&chunk[..read]);
    }
    Some(HttpRequest { path, headers, body })
}

async fn handle_http(mut stream: TcpStream, state: Arc<ServerState>) {
    let Some(request) = read_http_request(&mut stream).await else {
        return;
    };
    let index = {
        let mut recorded = state.recorded.lock().unwrap();
        recorded.http.push(request.clone());
        recorded.http.len() - 1
    };
    match (state.http_handler)(index, &request) {
        HttpReply::Full { status, headers, body } => {
            let mut response = format!(
                "HTTP/1.1 {status} Status\r\ncontent-length: {}\r\nconnection: close\r\n",
                body.len()
            );
            for (name, value) in headers {
                let _ = write!(response, "{name}: {value}\r\n");
            }
            response.push_str("\r\n");
            response.push_str(&body);
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.shutdown().await;
        }
        HttpReply::Stream { chunks, keep_open } => {
            let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n";
            if stream.write_all(head.as_bytes()).await.is_err() {
                return;
            }
            for (delay, chunk) in chunks {
                if delay > 0 {
                    // Watch for the client going away while waiting.
                    let mut probe = [0u8; 1];
                    tokio::select! {
                        () = tokio::time::sleep(Duration::from_millis(delay)) => {}
                        read = stream.read(&mut probe) => {
                            if matches!(read, Ok(0) | Err(_)) {
                                state.http_cancelled.fetch_add(1, Ordering::SeqCst);
                                return;
                            }
                        }
                    }
                }
                let framed = format!("{:x}\r\n{chunk}\r\n", chunk.len());
                if stream.write_all(framed.as_bytes()).await.is_err() {
                    state.http_cancelled.fetch_add(1, Ordering::SeqCst);
                    return;
                }
            }
            if keep_open {
                let mut sink = [0u8; 1024];
                while matches!(stream.read(&mut sink).await, Ok(read) if read > 0) {}
                state.http_cancelled.fetch_add(1, Ordering::SeqCst);
            } else {
                let _ = stream.write_all(b"0\r\n\r\n").await;
                let _ = stream.shutdown().await;
            }
        }
        HttpReply::Hang => {
            let mut sink = [0u8; 1024];
            while matches!(stream.read(&mut sink).await, Ok(read) if read > 0) {}
        }
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn mock_token(account_id: &str) -> String {
    let payload = base64::engine::general_purpose::STANDARD
        .encode(json!({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}).to_string());
    format!("aaa.{payload}.bbb")
}

fn model_with(base_url: &str, id: &str, extra: &Value) -> Model {
    let mut value = json!({
        "id": id, "name": id, "api": "openai-codex-responses", "provider": "openai-codex",
        "baseUrl": base_url, "reasoning": true, "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 400_000, "maxTokens": 128_000
    });
    for (key, field) in extra.as_object().unwrap() {
        value[key] = field.clone();
    }
    serde_json::from_value(value).unwrap()
}

fn model(base_url: &str) -> Model {
    model_with(base_url, "gpt-5.1-codex", &json!({}))
}

fn context(text: &str) -> TranscriptContext {
    normalize_context(&Context::new(vec![Message::user(text)]).with_system_prompt("You are a helpful assistant."))
}

fn hello_events(status: &str, end_turn: Option<bool>) -> Vec<Value> {
    let terminal = if status == "incomplete" {
        "response.incomplete"
    } else {
        "response.completed"
    };
    let mut response = json!({
        "status": status,
        "incomplete_details": if status == "incomplete" { json!({"reason": "max_output_tokens"}) } else { Value::Null },
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8, "input_tokens_details": {"cached_tokens": 0}}
    });
    if let Some(end_turn) = end_turn {
        response["end_turn"] = json!(end_turn);
    }
    vec![
        json!({"type": "response.output_item.added",
            "item": {"type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": []}}),
        json!({"type": "response.content_part.added", "part": {"type": "output_text", "text": ""}}),
        json!({"type": "response.output_text.delta", "delta": "Hello"}),
        json!({"type": "response.output_item.done",
            "item": {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "Hello"}]}}),
        json!({"type": terminal, "response": response}),
    ]
}

fn sse_payload(events: &[Value]) -> String {
    events.iter().fold(String::new(), |mut body, event| {
        let _ = write!(body, "data: {event}\n\n");
        body
    })
}

fn hello_sse() -> String {
    sse_payload(&hello_events("completed", None))
}

fn sse_server_handler() -> HttpHandler {
    Box::new(|_, _| sse_reply(hello_sse()))
}

fn completed(id: &str) -> Value {
    json!({"type": "response.completed", "response": {"id": id, "status": "completed",
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}}})
}

fn http_client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

fn options(token: &str) -> StreamOptions {
    StreamOptions {
        api_key: Some(token.into()),
        http_client: Some(http_client()),
        ..StreamOptions::default()
    }
}

fn sse_options(token: &str) -> StreamOptions {
    StreamOptions {
        transport: Some(Transport::Sse),
        ..options(token)
    }
}

fn with_api(mut options: StreamOptions, api: OpenAICodexResponsesOptions) -> StreamOptions {
    options.api = Some(ApiOptions::OpenAICodexResponses(api));
    options
}

fn text_of(message: &AssistantMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|block| block.as_text().map(|text| text.text.clone()))
        .collect()
}

async fn run(model: &Model, context: &TranscriptContext, options: StreamOptions) -> AssistantMessage {
    // Longer than the retry delays of the paused-clock tests.
    tokio::time::timeout(
        Duration::from_secs(300),
        openai_codex_responses::stream(model, context, options).result(),
    )
    .await
    .expect("stream did not finish")
}

type Captured = Arc<Mutex<Vec<Value>>>;

fn capture_provider_events(options: &mut StreamOptions) -> (Captured, Arc<Mutex<Vec<String>>>) {
    let events: Captured = Arc::new(Mutex::new(Vec::new()));
    let models = Arc::new(Mutex::new(Vec::new()));
    let (event_slot, model_slot) = (events.clone(), models.clone());
    options.on_provider_stream_event = Some(Arc::new(move |event: &Value, model: &Model| {
        event_slot.lock().unwrap().push(event.clone());
        model_slot.lock().unwrap().push(model.id.clone());
        let done: BoxFuture<'static, _> = Box::pin(async { Ok(()) });
        done
    }));
    (events, models)
}

fn event_types(events: &Captured) -> Vec<String> {
    events
        .lock()
        .unwrap()
        .iter()
        .map(|event| event["type"].as_str().unwrap().to_owned())
        .collect()
}

// ---------------------------------------------------------------------------
// SSE transport
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streams_sse_responses_and_forwards_raw_provider_events() {
    let server = CodexServer::start(sse_server_handler(), no_ws()).await;
    let token = mock_token("acc_test");
    let model = model(&server.base_url());
    let mut options = sse_options(&token);
    let (provider_events, provider_models) = capture_provider_events(&mut options);

    let mut stream = openai_codex_responses::stream(&model, &context("Say hello"), options);
    let mut saw_text_delta = false;
    let mut done = None;
    while let Some(event) = stream.next().await {
        match event {
            AssistantMessageEvent::TextDelta { .. } => saw_text_delta = true,
            AssistantMessageEvent::Done { message, .. } => done = Some(message),
            _ => {}
        }
    }
    assert!(saw_text_delta);
    assert_eq!(text_of(&done.unwrap()), "Hello");
    assert_eq!(
        event_types(&provider_events),
        [
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_item.done",
            "response.completed"
        ]
    );
    assert_eq!(*provider_models.lock().unwrap(), vec![model.id.clone(); 5]);

    let requests = server.http_requests();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.path, "/codex/responses");
    assert_eq!(
        request.header("authorization"),
        Some(format!("Bearer {token}").as_str())
    );
    assert_eq!(request.header("chatgpt-account-id"), Some("acc_test"));
    assert_eq!(request.header("openai-beta"), Some("responses=experimental"));
    assert_eq!(request.header("originator"), Some("pi"));
    assert_eq!(
        request.header("user-agent"),
        Some(delta_ai::http::user_agent().as_str())
    );
    assert_eq!(request.header("accept"), Some("text/event-stream"));
    assert_eq!(request.header("x-api-key"), None);
    let body = request.json();
    assert_eq!(body["model"], "gpt-5.1-codex");
    assert_eq!(body["instructions"], "You are a helpful assistant.");
    assert_eq!(body["store"], false);
    assert_eq!(body["parallel_tool_calls"], true);
}

#[tokio::test]
async fn processes_a_terminal_sse_event_without_a_trailing_blank_line() {
    let server = CodexServer::start(Box::new(|_, _| sse_reply(hello_sse().trim_end().to_owned())), no_ws()).await;
    let result = run(
        &model(&server.base_url()),
        &context("Say hello"),
        sse_options(&mock_token("acc")),
    )
    .await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(text_of(&result), "Hello");
}

#[tokio::test]
async fn completes_after_response_completed_even_when_the_sse_body_stays_open() {
    let mut body = sse_payload(&hello_events("completed", Some(false)));
    body.push_str("data: [DONE]\n\n");
    let server = CodexServer::start(
        Box::new(move |_, _| HttpReply::Stream {
            chunks: vec![(0, body.clone())],
            keep_open: true,
        }),
        no_ws(),
    )
    .await;
    let result = run(
        &model(&server.base_url()),
        &context("Say hello"),
        sse_options(&mock_token("acc")),
    )
    .await;
    assert_eq!(text_of(&result), "Hello");
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.end_turn, Some(false));
}

#[tokio::test]
async fn maps_response_incomplete_to_stop_reason_length_even_when_the_sse_body_stays_open() {
    let body = sse_payload(&hello_events("incomplete", None));
    let server = CodexServer::start(
        Box::new(move |_, _| HttpReply::Stream {
            chunks: vec![(0, body.clone())],
            keep_open: true,
        }),
        no_ws(),
    )
    .await;
    let result = run(
        &model(&server.base_url()),
        &context("Say hello"),
        sse_options(&mock_token("acc")),
    )
    .await;
    assert_eq!(text_of(&result), "Hello");
    assert_eq!(result.stop_reason, StopReason::Length);
}

#[tokio::test]
async fn times_out_the_sse_request_when_response_headers_do_not_arrive() {
    let server = CodexServer::start(Box::new(|_, _| HttpReply::Hang), no_ws()).await;
    let options = StreamOptions {
        timeout: Some(Duration::from_millis(10)),
        ..sse_options(&mock_token("acc"))
    };
    let result = run(&model(&server.base_url()), &context("Say hello"), options).await;
    assert_eq!(server.http_requests().len(), 1);
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("Codex SSE response headers timed out after 10ms")
    );
}

#[tokio::test]
async fn aborts_sse_body_reads_after_response_headers_arrive() {
    let first = sse_payload(&[
        json!({"type": "response.output_item.added",
            "item": {"type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": []}}),
        json!({"type": "response.content_part.added", "part": {"type": "output_text", "text": ""}}),
        json!({"type": "response.output_text.delta", "delta": "one"}),
    ]);
    let second = sse_payload(&[json!({"type": "response.output_text.delta", "delta": "two"})]);
    let rest = sse_payload(&[
        json!({"type": "response.output_item.done", "item": {"type": "message", "id": "msg_1", "role": "assistant",
            "status": "completed", "content": [{"type": "output_text", "text": "onetwo"}]}}),
        completed("resp_1"),
    ]);
    let server = CodexServer::start(
        Box::new(move |_, _| HttpReply::Stream {
            chunks: vec![(0, first.clone()), (200, second.clone()), (200, rest.clone())],
            keep_open: false,
        }),
        no_ws(),
    )
    .await;
    let cancel = CancellationToken::new();
    let options = StreamOptions {
        cancel: Some(cancel.clone()),
        ..sse_options(&mock_token("acc"))
    };
    let mut stream = openai_codex_responses::stream(&model(&server.base_url()), &context("Say hello"), options);
    let mut events = Vec::new();
    let mut result = None;
    while let Some(event) = stream.next().await {
        match &event {
            AssistantMessageEvent::TextDelta { delta, .. } => {
                events.push(format!("text_delta:{delta}"));
                if delta == "one" {
                    cancel.cancel();
                }
            }
            AssistantMessageEvent::Error { error, .. } => result = Some(error.clone()),
            other => events.push(other.kind().to_owned()),
        }
    }
    let result = result.unwrap();
    assert_eq!(result.stop_reason, StopReason::Aborted);
    assert_eq!(result.error_message.as_deref(), Some("Request was aborted"));
    assert!(events.contains(&"text_delta:one".to_owned()));
    assert!(!events.contains(&"text_delta:two".to_owned()));
    // The body read was cancelled: the server sees the connection go away.
    for _ in 0..100 {
        if server.state.http_cancelled.load(Ordering::SeqCst) > 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the SSE body was not cancelled");
}

#[tokio::test]
async fn sets_session_headers_and_prompt_cache_key_when_a_session_id_is_provided() {
    let server = CodexServer::start(sse_server_handler(), no_ws()).await;
    let options = StreamOptions {
        session_id: Some("test-session-123".into()),
        ..sse_options(&mock_token("acc"))
    };
    run(&model(&server.base_url()), &context("Say hello"), options).await;
    let request = &server.http_requests()[0];
    assert_eq!(request.header("session-id"), Some("test-session-123"));
    assert_eq!(request.header("session_id"), None);
    assert_eq!(request.header("x-client-request-id"), Some("test-session-123"));
    assert_eq!(request.json()["prompt_cache_key"], "test-session-123");
}

#[tokio::test]
async fn omits_sse_cache_affinity_when_cache_retention_is_none() {
    let server = CodexServer::start(sse_server_handler(), no_ws()).await;
    let options = StreamOptions {
        cache_retention: Some(CacheRetention::None),
        session_id: Some("one-off-summary-sse".into()),
        ..sse_options(&mock_token("acc"))
    };
    run(&model(&server.base_url()), &context("Say hello"), options).await;
    let request = &server.http_requests()[0];
    assert_eq!(request.header("session-id"), None);
    assert_eq!(request.header("x-client-request-id"), None);
    assert!(request.json().get("prompt_cache_key").is_none());
}

#[tokio::test]
async fn clamps_the_prompt_cache_key_and_session_header_to_64_characters() {
    let server = CodexServer::start(sse_server_handler(), no_ws()).await;
    let captured: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let slot = captured.clone();
    let options = StreamOptions {
        session_id: Some("x".repeat(67)),
        on_payload: Some(Arc::new(move |payload: &Value, _: &Model| {
            *slot.lock().unwrap() = Some(payload.clone());
            let keep: BoxFuture<'static, _> = Box::pin(async { Ok(None) });
            keep
        })),
        ..sse_options(&mock_token("acc"))
    };
    run(&model(&server.base_url()), &context("Say hello"), options).await;
    assert_eq!(
        captured.lock().unwrap().as_ref().unwrap()["prompt_cache_key"],
        "x".repeat(64)
    );
    let request = &server.http_requests()[0];
    assert_eq!(request.header("session-id"), Some("x".repeat(64).as_str()));
    assert_eq!(request.header("x-client-request-id"), Some("x".repeat(64).as_str()));
}

#[tokio::test]
async fn lets_model_and_caller_headers_override_originator_and_user_agent() {
    let server = CodexServer::start(sse_server_handler(), no_ws()).await;
    let token = mock_token("acc");
    let model = model_with(
        &server.base_url(),
        "gpt-5.1-codex",
        &json!({"headers": {"originator": "my-app"}}),
    );
    let options = StreamOptions {
        headers: Some(
            [
                ("user-agent".to_owned(), Some("my-app/1.0".to_owned())),
                ("Authorization".to_owned(), Some("Bearer ignored".to_owned())),
            ]
            .into_iter()
            .collect(),
        ),
        ..sse_options(&token)
    };
    run(&model, &context("Say hello"), options).await;
    let request = &server.http_requests()[0];
    assert_eq!(request.header("originator"), Some("my-app"));
    assert_eq!(request.header("user-agent"), Some("my-app/1.0"));
    assert_eq!(
        request.header("authorization"),
        Some(format!("Bearer {token}").as_str())
    );
}

#[tokio::test]
async fn preserves_gpt_5_5_xhigh_reasoning_effort_from_simple_options() {
    let server = CodexServer::start(sse_server_handler(), no_ws()).await;
    let model = model_with(
        &server.base_url(),
        "gpt-5.5",
        &json!({"thinkingLevelMap": {"xhigh": "xhigh"}}),
    );
    let options = SimpleStreamOptions {
        base: sse_options(&mock_token("acc")),
        reasoning: Some(ThinkingLevel::Xhigh),
        ..SimpleStreamOptions::default()
    };
    openai_codex_responses::stream_simple(&model, &context("Say hello"), options)
        .result()
        .await;
    assert_eq!(
        server.http_requests()[0].json()["reasoning"],
        json!({"effort": "xhigh", "summary": "auto"})
    );
}

fn tool(value: Value) -> Tool {
    serde_json::from_value(value).unwrap()
}

#[tokio::test]
async fn forwards_required_tool_choice() {
    let server = CodexServer::start(sse_server_handler(), no_ws()).await;
    let context = normalize_context(
        &Context::new(vec![Message::user("Do not call ping. Respond with text instead.")]).with_tools(vec![tool(
            json!({"name": "ping", "description": "Ping",
                "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}}),
        )]),
    );
    let options = with_api(
        sse_options(&mock_token("acc")),
        OpenAICodexResponsesOptions {
            tool_choice: Some("required".into()),
            ..OpenAICodexResponsesOptions::default()
        },
    );
    run(
        &model_with(&server.base_url(), "gpt-5.5", &json!({})),
        &context,
        options,
    )
    .await;
    assert_eq!(server.http_requests()[0].json()["tool_choice"], "required");
}

#[tokio::test]
async fn sets_codex_strict_mode_explicitly_and_honors_constrained_sampling() {
    let server = CodexServer::start(sse_server_handler(), no_ws()).await;
    let parameters = json!({"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]});
    let mut strict_parameters = parameters.clone();
    strict_parameters["additionalProperties"] = json!(false);
    let context = normalize_context(&Context::new(vec![Message::user("Use a tool")]).with_tools(vec![
        tool(
            json!({"name": "optional", "description": "Optional constrained sampling",
            "parameters": parameters, "constrainedSampling": false}),
        ),
        tool(json!({"name": "strict", "description": "Strict constrained sampling",
            "parameters": strict_parameters, "constrainedSampling": {"type": "json_schema", "strict": "prefer"}})),
    ]));
    run(
        &model_with(&server.base_url(), "gpt-5.5", &json!({})),
        &context,
        sse_options(&mock_token("acc")),
    )
    .await;
    let tools = server.http_requests()[0].json()["tools"].clone();
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["name"], "optional");
    assert_eq!(tools[0]["strict"], Value::Null);
    assert!(tools[0].as_object().unwrap().contains_key("strict"));
    assert_eq!(tools[1]["name"], "strict");
    assert_eq!(tools[1]["strict"], true);
}

#[tokio::test]
async fn clamps_minimal_reasoning_effort_to_low() {
    for id in ["gpt-5.3-codex", "gpt-5.4", "gpt-5.5"] {
        let server = CodexServer::start(sse_server_handler(), no_ws()).await;
        let model = model_with(&server.base_url(), id, &json!({"thinkingLevelMap": {"minimal": "low"}}));
        let options = with_api(
            sse_options(&mock_token("acc")),
            OpenAICodexResponsesOptions {
                reasoning_effort: Some(ReasoningEffort::Minimal),
                ..OpenAICodexResponsesOptions::default()
            },
        );
        run(&model, &context("Say hello"), options).await;
        assert_eq!(
            server.http_requests()[0].json()["reasoning"],
            json!({"effort": "low", "summary": "auto"}),
            "{id}"
        );
    }
}

#[tokio::test]
async fn uses_the_client_sent_service_tier_when_codex_echoes_default() {
    for (id, tier, multiplier) in [
        ("gpt-5.1-codex", "flex", 0.5),
        ("gpt-5.1-codex", "priority", 2.0),
        ("gpt-5.5", "flex", 0.5),
        ("gpt-5.5", "priority", 2.5),
    ] {
        let mut events = hello_events("completed", None);
        events[4]["response"]["service_tier"] = json!("default");
        events[4]["response"]["usage"] = json!({"input_tokens": 1_000_000, "output_tokens": 1_000_000,
            "total_tokens": 2_000_000, "input_tokens_details": {"cached_tokens": 0}});
        let body = sse_payload(&events);
        let server = CodexServer::start(Box::new(move |_, _| sse_reply(body.clone())), no_ws()).await;
        let model = model_with(
            &server.base_url(),
            id,
            &json!({"cost": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0}}),
        );
        let options = with_api(
            sse_options(&mock_token("acc")),
            OpenAICodexResponsesOptions {
                service_tier: Some(tier.into()),
                ..OpenAICodexResponsesOptions::default()
            },
        );
        let result = run(&model, &context("Say hello"), options).await;
        assert_eq!(server.http_requests()[0].json()["service_tier"], tier);
        assert!((result.usage.cost.input - multiplier).abs() < 1e-9, "{id} {tier}");
        assert!(
            (result.usage.cost.output - 2.0 * multiplier).abs() < 1e-9,
            "{id} {tier}"
        );
        assert!((result.usage.cost.total - 3.0 * multiplier).abs() < 1e-9, "{id} {tier}");
    }
}

#[tokio::test]
async fn does_not_set_session_headers_when_no_session_id_is_provided() {
    let server = CodexServer::start(sse_server_handler(), no_ws()).await;
    run(
        &model(&server.base_url()),
        &context("Say hello"),
        sse_options(&mock_token("acc")),
    )
    .await;
    let request = &server.http_requests()[0];
    assert_eq!(request.header("session-id"), None);
    assert_eq!(request.header("session_id"), None);
    assert_eq!(request.header("x-client-request-id"), None);
    assert!(request.json().get("prompt_cache_key").is_none());
}

#[tokio::test]
async fn zstd_compresses_sse_request_bodies() {
    let server = CodexServer::start(sse_server_handler(), no_ws()).await;
    let large_text = "compress me ".repeat(400);
    run(
        &model(&server.base_url()),
        &context(&large_text),
        sse_options(&mock_token("acc")),
    )
    .await;
    run(
        &model(&server.base_url()),
        &context("hi"),
        sse_options(&mock_token("acc")),
    )
    .await;
    let requests = server.http_requests();
    for request in &requests {
        assert_eq!(request.header("content-encoding"), Some("zstd"));
        assert_eq!(request.header("content-type"), Some("application/json"));
    }
    assert!(requests[0].body.len() < large_text.len());
    assert_eq!(requests[0].json()["input"][0]["content"][0]["text"], large_text);
}

#[tokio::test]
async fn fails_without_an_api_key_or_with_a_token_without_an_account_id() {
    let model = model("https://example.invalid");
    let result = openai_codex_responses::stream_simple(&model, &context("hi"), SimpleStreamOptions::default())
        .result()
        .await;
    assert_eq!(
        result.error_message.as_deref(),
        Some("No API key for provider: openai-codex")
    );
    let result = run(&model, &context("hi"), sse_options("not-a-jwt")).await;
    assert_eq!(
        result.error_message.as_deref(),
        Some("Failed to extract accountId from token")
    );
}

// ---------------------------------------------------------------------------
// SSE retries
// ---------------------------------------------------------------------------

fn rate_limited(headers: Vec<(&'static str, String)>) -> HttpReply {
    HttpReply::Full {
        status: 429,
        headers,
        body: json!({"error": {"code": "rate_limit_exceeded", "message": "rate limited"}}).to_string(),
    }
}

/// Serves `failures` 429 responses with `headers`, then the hello stream; records the
/// (paused) time of each request.
async fn retry_server(
    failures: usize,
    headers: impl Fn() -> Vec<(&'static str, String)> + Send + Sync + 'static,
) -> (CodexServer, Arc<Mutex<Vec<tokio::time::Instant>>>) {
    let times = Arc::new(Mutex::new(Vec::new()));
    let recorded = times.clone();
    let server = CodexServer::start(
        Box::new(move |index, _| {
            recorded.lock().unwrap().push(tokio::time::Instant::now());
            if index < failures {
                rate_limited(headers())
            } else {
                sse_reply(hello_sse())
            }
        }),
        no_ws(),
    )
    .await;
    (server, times)
}

fn gaps(times: &Mutex<Vec<tokio::time::Instant>>) -> Vec<Duration> {
    let times = times.lock().unwrap();
    times.windows(2).map(|pair| pair[1] - pair[0]).collect()
}

/// SSE options for the paused-clock tests: a client without connect or pool-idle timers, so
/// the only timer while a request is in flight is the retry delay under test. With other
/// timers pending, the paused clock can jump to them while loopback I/O is still in transit
/// (seen on macOS).
fn paused_sse_options(token: &str) -> StreamOptions {
    StreamOptions {
        http_client: Some(
            reqwest::Client::builder()
                .pool_idle_timeout(None)
                .pool_max_idle_per_host(0)
                .build()
                .unwrap(),
        ),
        ..sse_options(token)
    }
}

/// `run` without the outer timeout, whose timer the paused clock could also jump to.
async fn run_paused(model: &Model, context: &TranscriptContext, options: StreamOptions) -> AssistantMessage {
    openai_codex_responses::stream(model, context, options).result().await
}

fn approx(actual: Duration, expected: Duration) -> bool {
    actual >= expected && actual < expected + Duration::from_millis(500)
}

#[tokio::test(start_paused = true)]
async fn uses_retry_after_ms_for_sse_retries() {
    let (server, times) = retry_server(1, || vec![("retry-after-ms", "1500".into())]).await;
    let options = StreamOptions {
        max_retries: Some(1),
        ..paused_sse_options(&mock_token("acc"))
    };
    let result = run_paused(&model(&server.base_url()), &context("Say hello"), options).await;
    assert_eq!(text_of(&result), "Hello");
    assert_eq!(server.http_requests().len(), 2);
    assert!(
        approx(gaps(&times)[0], Duration::from_millis(1500)),
        "{:?}",
        gaps(&times)
    );
}

#[tokio::test(start_paused = true)]
async fn uses_retry_after_seconds_for_sse_retries() {
    let (server, times) = retry_server(1, || vec![("retry-after", "60".into())]).await;
    let options = StreamOptions {
        max_retries: Some(1),
        ..paused_sse_options(&mock_token("acc"))
    };
    let result = run_paused(&model(&server.base_url()), &context("Say hello"), options).await;
    assert_eq!(text_of(&result), "Hello");
    assert!(approx(gaps(&times)[0], Duration::from_secs(60)), "{:?}", gaps(&times));
}

#[tokio::test(start_paused = true)]
async fn uses_a_retry_after_http_date_for_sse_retries() {
    fn http_date(seconds_from_now: u64) -> String {
        const DAYS: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];
        const MONTHS: [&str; 12] = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ];
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + seconds_from_now;
        let days = (now / 86_400) as i64;
        let seconds = now % 86_400;
        // Civil from days (Howard Hinnant's algorithm).
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = doy - (153 * mp + 2) / 5 + 1;
        let month = if mp < 10 { mp + 3 } else { mp - 9 };
        let year = yoe + era * 400 + i64::from(month <= 2);
        format!(
            "{}, {day:02} {} {year} {:02}:{:02}:{:02} GMT",
            DAYS[(days % 7) as usize],
            MONTHS[(month - 1) as usize],
            seconds / 3600,
            seconds / 60 % 60,
            seconds % 60
        )
    }
    let (server, times) = retry_server(1, || vec![("retry-after", http_date(45))]).await;
    let options = StreamOptions {
        max_retries: Some(1),
        ..paused_sse_options(&mock_token("acc"))
    };
    let result = run_paused(&model(&server.base_url()), &context("Say hello"), options).await;
    assert_eq!(text_of(&result), "Hello");
    let gap = gaps(&times)[0];
    assert!(
        gap > Duration::from_secs(43) && gap <= Duration::from_secs(46),
        "{gap:?}"
    );
}

#[tokio::test]
async fn fails_immediately_when_a_retry_delay_exceeds_the_limit() {
    for status in [429, 503] {
        let server = CodexServer::start(
            Box::new(move |_, _| HttpReply::Full {
                status,
                headers: vec![("content-type", "application/json".into()), ("retry-after", "2".into())],
                body: json!({"error": {"code": "temporarily_unavailable", "message": "retry later"}}).to_string(),
            }),
            no_ws(),
        )
        .await;
        let options = StreamOptions {
            max_retries: Some(3),
            max_retry_delay: Some(Duration::from_secs(1)),
            ..paused_sse_options(&mock_token("acc"))
        };
        let result = run_paused(&model(&server.base_url()), &context("Say hello"), options).await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(
            result.error_message.as_deref(),
            Some("Server requested 2s retry delay (max: 1s)")
        );
        assert_eq!(server.http_requests().len(), 1);
    }
}

#[tokio::test(start_paused = true)]
async fn uses_exponential_backoff_across_repeated_sse_retries_without_retry_headers() {
    let (server, times) = retry_server(3, Vec::new).await;
    let options = StreamOptions {
        max_retries: Some(3),
        ..paused_sse_options(&mock_token("acc"))
    };
    let result = run_paused(&model(&server.base_url()), &context("Say hello"), options).await;
    assert_eq!(text_of(&result), "Hello");
    assert_eq!(server.http_requests().len(), 4);
    let gaps = gaps(&times);
    for (gap, expected) in gaps.iter().zip([1, 2, 4]) {
        assert!(approx(*gap, Duration::from_secs(expected)), "{gaps:?}");
    }
}

#[tokio::test]
async fn reports_the_chatgpt_usage_limit_without_retrying() {
    let server = CodexServer::start(
        Box::new(|_, _| HttpReply::Full {
            status: 429,
            headers: Vec::new(),
            body: json!({"error": {"code": "usage_limit_reached", "plan_type": "PLUS", "message": "limit"}})
                .to_string(),
        }),
        no_ws(),
    )
    .await;
    let options = StreamOptions {
        max_retries: Some(0),
        ..sse_options(&mock_token("acc"))
    };
    let result = run(&model(&server.base_url()), &context("hi"), options).await;
    assert_eq!(
        result.error_message.as_deref(),
        Some("You have hit your ChatGPT usage limit (plus plan).")
    );
    assert_eq!(server.http_requests().len(), 1);
}

// ---------------------------------------------------------------------------
// WebSocket transport
// ---------------------------------------------------------------------------

fn ws_events(events: Vec<Value>) -> WsHandler {
    Box::new(move |_, _, _| WsReply::Events(events.clone()))
}

fn ws_options(token: &str, transport: Transport, session_id: Option<&str>) -> StreamOptions {
    StreamOptions {
        transport: Some(transport),
        session_id: session_id.map(str::to_owned),
        ..options(token)
    }
}

#[tokio::test]
async fn forwards_auto_transport_and_raw_provider_events_from_stream_simple() {
    let session = "codex-session-auto";
    let mut events = hello_events("completed", None);
    events[4] = json!({"type": "response.done", "response": {"status": "completed", "end_turn": false,
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8, "input_tokens_details": {"cached_tokens": 0}}}});
    let server = CodexServer::start(no_http(), ws_events(events)).await;
    let model = model(&server.base_url());
    let mut base = ws_options(&mock_token("acc"), Transport::Auto, Some(session));
    let (provider_events, provider_models) = capture_provider_events(&mut base);
    let options = SimpleStreamOptions {
        base,
        ..SimpleStreamOptions::default()
    };
    let result = openai_codex_responses::stream_simple(&model, &context("Say hello"), options)
        .result()
        .await;

    assert_eq!(text_of(&result), "Hello");
    assert_eq!(result.end_turn, Some(false));
    let messages = server.ws_messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].1["type"], "response.create");
    assert_eq!(messages[0].1["model"], "gpt-5.1-codex");
    assert_eq!(
        event_types(&provider_events),
        [
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_item.done",
            "response.done"
        ]
    );
    assert_eq!(*provider_models.lock().unwrap(), vec![model.id.clone(); 5]);
    let headers = &server.ws_headers()[0];
    assert_eq!(headers["session-id"], session);
    assert!(!headers.contains_key("session_id"));
    assert_eq!(headers["x-client-request-id"], session);
    assert_eq!(headers["openai-beta"], "responses_websockets=2026-02-06");
    assert_eq!(headers["chatgpt-account-id"], "acc");
    assert_eq!(headers["originator"], "pi");
    assert!(!headers.contains_key("accept") && !headers.contains_key("content-type"));
    assert!(server.http_requests().is_empty());
    let stats = get_openai_codex_websocket_debug_stats(session).unwrap();
    assert_eq!(stats.cached_context_requests, 1);
    assert_eq!(stats.full_context_requests, 1);
    close_openai_codex_websocket_sessions(Some(session));
}

#[tokio::test]
async fn uses_a_fresh_request_id_for_websockets_without_a_session() {
    let server = CodexServer::start(no_http(), ws_events(vec![completed("resp_1")])).await;
    let result = run(
        &model(&server.base_url()),
        &context("hi"),
        ws_options(&mock_token("acc"), Transport::Websocket, None),
    )
    .await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    let headers = &server.ws_headers()[0];
    let request_id = uuid::Uuid::parse_str(&headers["x-client-request-id"]).unwrap();
    assert_eq!(request_id.get_version_num(), 7);
    assert_eq!(headers["session-id"], headers["x-client-request-id"]);
    assert!(server.ws_messages()[0].1.get("prompt_cache_key").is_none());
    // One-shot sockets are closed after the response.
    server.wait_for_ws_closed(1).await;
}

#[tokio::test]
async fn scopes_cached_websockets_to_the_authenticated_account() {
    let session = "codex-shared-session";
    let responses = Arc::new(AtomicUsize::new(0));
    let server = CodexServer::start(
        no_http(),
        Box::new(move |_, _, _| {
            let id = responses.fetch_add(1, Ordering::SeqCst) + 1;
            WsReply::Events(vec![completed(&format!("resp_{id}"))])
        }),
    )
    .await;
    let model = model(&server.base_url());
    let empty = normalize_context(&Context::new(Vec::new()));
    for account in ["account-a", "account-b", "account-a"] {
        let result = run(
            &model,
            &empty,
            ws_options(&mock_token(account), Transport::WebsocketCached, Some(session)),
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Stop, "{:?}", result.error_message);
    }
    let headers = server.ws_headers();
    assert_eq!(
        headers
            .iter()
            .map(|h| h["chatgpt-account-id"].clone())
            .collect::<Vec<_>>(),
        ["account-a", "account-b"]
    );
    assert_eq!(
        headers.iter().map(|h| h["authorization"].clone()).collect::<Vec<_>>(),
        [
            format!("Bearer {}", mock_token("account-a")),
            format!("Bearer {}", mock_token("account-b"))
        ]
    );
    assert!(server.http_requests().is_empty());
    let stats = get_openai_codex_websocket_debug_stats(session).unwrap();
    assert_eq!((stats.connections_created, stats.connections_reused), (2, 1));
    close_openai_codex_websocket_sessions(Some(session));
    server.wait_for_ws_closed(2).await;
}

#[tokio::test]
async fn closes_one_shot_websockets_when_cache_retention_is_none() {
    let session = "codex-one-off-summary";
    let server = CodexServer::start(no_http(), ws_events(vec![completed("resp_1")])).await;
    let model = model(&server.base_url());
    for _ in 0..2 {
        let options = StreamOptions {
            cache_retention: Some(CacheRetention::None),
            ..ws_options(&mock_token("acc"), Transport::Auto, Some(session))
        };
        let result = run(&model, &context("Say hello"), options).await;
        assert_eq!(result.stop_reason, StopReason::Stop);
    }
    assert_eq!(server.ws_connections(), 2);
    server.wait_for_ws_closed(2).await;
    let messages = server.ws_messages();
    assert_eq!(messages.len(), 2);
    assert!(messages.iter().all(|(_, body)| body.get("prompt_cache_key").is_none()));
    assert_eq!(get_openai_codex_websocket_debug_stats(session), None);
    assert!(server.http_requests().is_empty());
}

#[tokio::test]
async fn falls_back_to_sse_when_the_websocket_does_not_open_before_the_connect_timeout() {
    let session = "codex-ws-connect-timeout";
    let server = CodexServer::start_with(sse_server_handler(), no_ws(), true).await;
    let options = StreamOptions {
        timeout: Some(Duration::from_secs(300)),
        websocket_connect_timeout: Some(Duration::from_millis(50)),
        ..ws_options(&mock_token("acc"), Transport::Auto, Some(session))
    };
    let result = run(&model(&server.base_url()), &context("Say hello"), options).await;
    assert_eq!(text_of(&result), "Hello");
    assert_eq!(server.http_requests().len(), 1);
    let stats = get_openai_codex_websocket_debug_stats(session).unwrap();
    assert_eq!(stats.websocket_failures, 1);
    assert_eq!(stats.sse_fallbacks, 1);
    assert_eq!(stats.websocket_fallback_active, Some(true));
    assert_eq!(
        stats.last_websocket_error.as_deref(),
        Some("WebSocket connect timeout after 50ms")
    );
    let diagnostics = result.diagnostics.unwrap();
    assert_eq!(diagnostics[0].kind, "provider_transport_failure");
    let details = diagnostics[0].details.as_ref().unwrap();
    assert_eq!(details["configuredTransport"], "auto");
    assert_eq!(details["fallbackTransport"], "sse");
    assert_eq!(details["eventsEmitted"], false);
    assert_eq!(details["phase"], "before_message_stream_start");

    // The session stays on SSE.
    let options = ws_options(&mock_token("acc"), Transport::Auto, Some(session));
    run(&model(&server.base_url()), &context("Again"), options).await;
    assert_eq!(server.http_requests().len(), 2);
    assert_eq!(server.ws_connections(), 1);
    let stats = get_openai_codex_websocket_debug_stats(session).unwrap();
    assert_eq!((stats.websocket_failures, stats.sse_fallbacks), (1, 2));
    reset_openai_codex_websocket_debug_stats(Some(session));
    assert_eq!(get_openai_codex_websocket_debug_stats(session), None);
}

#[tokio::test]
async fn reconnects_once_when_the_websocket_connection_limit_is_reached_before_output_starts() {
    let server = CodexServer::start(
        no_http(),
        Box::new(|connection, _, _| {
            WsReply::Events(vec![if connection == 0 {
                json!({"type": "error", "error": {"code": "websocket_connection_limit_reached"}})
            } else {
                completed("resp_1")
            }])
        }),
    )
    .await;
    let empty = normalize_context(&Context::new(Vec::new()));
    let result = run(&model(&server.base_url()), &empty, options(&mock_token("acc"))).await;
    assert_eq!(result.stop_reason, StopReason::Stop, "{:?}", result.error_message);
    assert_eq!(server.ws_connections(), 2);
    assert!(server.http_requests().is_empty());
}

#[tokio::test]
async fn falls_back_to_sse_when_a_websocket_is_idle_before_the_first_event() {
    let session = "codex-ws-idle-before-start";
    let server = CodexServer::start(sse_server_handler(), Box::new(|_, _, _| WsReply::Silent)).await;
    let options = StreamOptions {
        timeout: Some(Duration::from_millis(50)),
        ..ws_options(&mock_token("acc"), Transport::Auto, Some(session))
    };
    let result = run(&model(&server.base_url()), &context("Say hello"), options).await;
    assert_eq!(text_of(&result), "Hello");
    assert_eq!(server.ws_messages().len(), 1);
    assert_eq!(server.http_requests().len(), 1);
    let stats = get_openai_codex_websocket_debug_stats(session).unwrap();
    assert_eq!(stats.websocket_failures, 1);
    assert_eq!(stats.sse_fallbacks, 1);
    assert_eq!(stats.websocket_fallback_active, Some(true));
    assert_eq!(
        stats.last_websocket_error.as_deref(),
        Some("WebSocket idle timeout after 50ms")
    );
}

#[tokio::test]
async fn errors_when_a_websocket_is_idle_after_the_stream_started() {
    let server = CodexServer::start(
        no_http(),
        ws_events(vec![json!({"type": "response.output_item.added",
            "item": {"type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": []}})]),
    )
    .await;
    let options = StreamOptions {
        timeout: Some(Duration::from_millis(50)),
        ..ws_options(&mock_token("acc"), Transport::Auto, None)
    };
    let result = run(&model(&server.base_url()), &context("Say hello"), options).await;
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("WebSocket idle timeout after 50ms")
    );
    assert!(server.http_requests().is_empty());
    let details = result.diagnostics.unwrap()[0].details.clone().unwrap();
    assert_eq!(details["phase"], "after_message_stream_start");
    assert!(details.get("fallbackTransport").is_none());
}

#[tokio::test]
async fn falls_back_to_sse_when_the_websocket_closes_before_the_first_event() {
    let server = CodexServer::start(sse_server_handler(), Box::new(|_, _, _| WsReply::Close(1009, ""))).await;
    let result = run(
        &model(&server.base_url()),
        &context("hi"),
        ws_options(&mock_token("acc"), Transport::Websocket, None),
    )
    .await;
    assert_eq!(text_of(&result), "Hello");
    assert_eq!(server.http_requests().len(), 1);
    let diagnostics = result.diagnostics.unwrap();
    let error = diagnostics[0].error.as_ref().unwrap();
    assert_eq!(error.message, "WebSocket closed 1009 message too big");
    assert_eq!(error.name.as_deref(), Some("WebSocketCloseError"));
    assert_eq!(error.code, Some(json!(1009)));
    assert_eq!(
        diagnostics[0].details.as_ref().unwrap()["configuredTransport"],
        "websocket"
    );
}

fn custom_tool_events(response_id: &str) -> Vec<Value> {
    vec![
        json!({"type": "response.created", "response": {"id": response_id}}),
        json!({"type": "response.output_item.added",
            "item": {"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "sample_tool", "input": ""}}),
        json!({"type": "response.custom_tool_call_input.delta", "item_id": "ctc_1", "delta": "abc"}),
        json!({"type": "response.custom_tool_call_input.done", "item_id": "ctc_1", "input": "abc"}),
        json!({"type": "response.output_item.done",
            "item": {"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "sample_tool", "input": "abc"}}),
    ]
}

#[tokio::test]
async fn sends_only_response_input_deltas_in_websocket_cached_mode() {
    let session = "codex-session-deltas";
    let server = CodexServer::start(
        no_http(),
        Box::new(|_, index, _| {
            let response_id = format!("resp_{}", index + 1);
            let mut events = if index == 0 {
                custom_tool_events(&response_id)
            } else {
                vec![json!({"type": "response.created", "response": {"id": response_id}})]
            };
            let mut terminal = completed(&response_id);
            terminal["response"]["usage"]["input_tokens_details"] = json!({"cached_tokens": 0});
            events.push(terminal);
            WsReply::Events(events)
        }),
    )
    .await;
    let model = model_with(
        &server.base_url(),
        "gpt-5.1-codex",
        &json!({"compat": {"supportsOpenAIGrammarTools": true}}),
    );
    let sample_tool = tool(json!({"name": "sample_tool", "description": "Sample tool",
        "parameters": {"type": "object", "properties": {"payload": {"type": "string"}}, "required": ["payload"]},
        "constrainedSampling": {"type": "grammar", "variants": {"openai_lark": "start: /[a-z]+/"}}}));
    let first_context = Context::new(vec![Message::user("Use the tool")])
        .with_system_prompt("You are a helpful assistant.")
        .with_tools(vec![sample_tool]);
    let first = run(
        &model,
        &normalize_context(&first_context),
        ws_options(&mock_token("acc"), Transport::WebsocketCached, Some(session)),
    )
    .await;
    assert_eq!(first.stop_reason, StopReason::ToolUse, "{:?}", first.error_message);

    let mut second_context = first_context.clone();
    second_context.messages.push(Message::Assistant(first));
    second_context.messages.push(
        serde_json::from_value(
            json!({"role": "toolResult", "toolCallId": "call_1|ctc_1", "toolName": "sample_tool",
            "content": [{"type": "text", "text": "real result"}], "isError": false, "timestamp": 2}),
        )
        .unwrap(),
    );
    second_context.messages.push(Message::user("Now finish"));
    run(
        &model,
        &normalize_context(&second_context),
        ws_options(&mock_token("acc"), Transport::WebsocketCached, Some(session)),
    )
    .await;

    let messages = server.ws_messages();
    assert_eq!(messages.len(), 2);
    let (first_body, second_body) = (&messages[0].1, &messages[1].1);
    assert_eq!(first_body["store"], false);
    assert!(first_body.get("previous_response_id").is_none());
    assert_eq!(
        first_body["input"],
        json!([{"role": "user", "content": [{"type": "input_text", "text": "Use the tool"}]}])
    );
    assert_eq!(second_body["store"], false);
    assert_eq!(second_body["previous_response_id"], "resp_1");
    assert_eq!(
        second_body["input"],
        json!([
            {"type": "custom_tool_call_output", "call_id": "call_1", "output": "real result"},
            {"role": "user", "content": [{"type": "input_text", "text": "Now finish"}]}
        ])
    );
    let stats = get_openai_codex_websocket_debug_stats(session).unwrap();
    assert_eq!(stats.requests, 2);
    assert_eq!(stats.connections_created, 1);
    assert_eq!(stats.connections_reused, 1);
    assert_eq!(stats.cached_context_requests, 2);
    assert_eq!(stats.store_true_requests, 0);
    assert_eq!(stats.full_context_requests, 1);
    assert_eq!(stats.delta_requests, 1);
    assert_eq!(stats.last_delta_input_items, Some(2));
    assert_eq!(stats.last_previous_response_id.as_deref(), Some("resp_1"));

    // Ending the session closes its cached socket.
    cleanup_session_resources(Some(session)).unwrap();
    server.wait_for_ws_closed(1).await;
}

fn message_events(response_id: &str, message_id: &str, text: &str) -> Vec<Value> {
    vec![
        json!({"type": "response.created", "response": {"id": response_id}}),
        json!({"type": "response.output_item.added", "output_index": 0,
            "item": {"type": "message", "id": message_id, "role": "assistant", "status": "in_progress", "content": []}}),
        json!({"type": "response.output_item.done", "output_index": 0,
            "item": {"type": "message", "id": message_id, "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": text}]}}),
        completed(response_id),
    ]
}

async fn recovers_a_missing_cached_websocket_continuation(recover_with_sse: bool) {
    let session = if recover_with_sse {
        "codex-missing-continuation-sse"
    } else {
        "codex-missing-continuation-websocket"
    };
    let requests = Arc::new(AtomicUsize::new(0));
    let server = CodexServer::start(
        sse_server_handler(),
        Box::new(move |_, _, _| {
            let request = requests.fetch_add(1, Ordering::SeqCst) + 1;
            match request {
                2 => WsReply::Events(vec![
                    json!({"type": "codex.rate_limits", "plan_type": "plus",
                        "rate_limits": {"allowed": true, "limit_reached": false,
                            "primary": {"used_percent": 7, "window_minutes": 10080, "reset_after_seconds": 556_112,
                                "reset_at": 1_785_269_351u64},
                            "secondary": null},
                        "code_review_rate_limits": null, "additional_rate_limits": null,
                        "credits": {"has_credits": false, "unlimited": false, "balance": "0"}, "promo": null}),
                    json!({"type": "error", "status": 400, "error": {"code": "previous_response_not_found",
                        "message": "Previous response with id 'resp_1' not found.", "param": "previous_response_id"}}),
                ]),
                3 if recover_with_sse => WsReply::Close(1011, "retry websocket failed"),
                1 => WsReply::Events(message_events("resp_1", "msg_1", "Hello")),
                _ => WsReply::Events(message_events("resp_2", "msg_2", "Recovered")),
            }
        }),
    )
    .await;
    let model = model(&server.base_url());
    let first_context =
        Context::new(vec![Message::user("Say hello")]).with_system_prompt("You are a helpful assistant.");
    let first = run(
        &model,
        &normalize_context(&first_context),
        ws_options(&mock_token("acc"), Transport::WebsocketCached, Some(session)),
    )
    .await;
    assert_eq!(text_of(&first), "Hello");
    let mut second_context = first_context.clone();
    second_context.messages.push(Message::Assistant(first));
    second_context.messages.push(Message::user("Now finish"));

    let mut stream = openai_codex_responses::stream(
        &model,
        &normalize_context(&second_context),
        ws_options(&mock_token("acc"), Transport::WebsocketCached, Some(session)),
    );
    let mut kinds = Vec::new();
    let mut second = None;
    while let Some(event) = stream.next().await {
        kinds.push(event.kind());
        if let AssistantMessageEvent::Done { message, .. } = event {
            second = Some(message);
        }
    }
    let second = second.unwrap_or_else(|| panic!("no done event: {kinds:?}"));
    assert_eq!(second.stop_reason, StopReason::Stop);
    assert_eq!(text_of(&second), if recover_with_sse { "Hello" } else { "Recovered" });
    assert_eq!(kinds.iter().filter(|kind| **kind == "start").count(), 1);
    assert!(!kinds.contains(&"error"));

    assert_eq!(server.ws_connections(), 2);
    let messages = server.ws_messages();
    assert_eq!(messages.len(), 3);
    assert_eq!(
        messages.iter().map(|(connection, _)| *connection).collect::<Vec<_>>(),
        [0, 0, 1]
    );
    assert_eq!(messages[1].1["previous_response_id"], "resp_1");
    assert_eq!(
        messages[1].1["input"],
        json!([{"role": "user", "content": [{"type": "input_text", "text": "Now finish"}]}])
    );
    assert!(messages[2].1.get("previous_response_id").is_none());
    let input = messages[2].1["input"].as_array().unwrap();
    assert_eq!(input.len(), 3);
    assert_eq!(
        input[2],
        json!({"role": "user", "content": [{"type": "input_text", "text": "Now finish"}]})
    );
    assert_eq!(server.http_requests().len(), usize::from(recover_with_sse));
    let stats = get_openai_codex_websocket_debug_stats(session).unwrap();
    assert_eq!(stats.requests, 3);
    assert_eq!(stats.connections_created, 2);
    assert_eq!(stats.connections_reused, 1);
    assert_eq!(stats.full_context_requests, 2);
    assert_eq!(stats.delta_requests, 1);
    assert_eq!(stats.websocket_failures, u64::from(recover_with_sse));
    assert_eq!(stats.sse_fallbacks, u64::from(recover_with_sse));
    close_openai_codex_websocket_sessions(Some(session));
}

#[tokio::test]
async fn recovers_a_missing_cached_websocket_continuation_via_websocket() {
    recovers_a_missing_cached_websocket_continuation(false).await;
}

#[tokio::test]
async fn recovers_a_missing_cached_websocket_continuation_via_sse() {
    recovers_a_missing_cached_websocket_continuation(true).await;
}

#[tokio::test]
async fn does_not_retry_codex_errors_on_another_transport() {
    let server = CodexServer::start(
        sse_server_handler(),
        ws_events(vec![json!({"type": "response.failed",
            "response": {"error": {"code": "server_error", "message": "boom"}}})]),
    )
    .await;
    let result = run(
        &model(&server.base_url()),
        &context("hi"),
        ws_options(
            &mock_token("acc"),
            Transport::Auto,
            Some("codex-no-fallback-on-api-error"),
        ),
    )
    .await;
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(result.error_message.as_deref(), Some("boom"));
    assert!(server.http_requests().is_empty());
    assert!(result.diagnostics.is_none());
    close_openai_codex_websocket_sessions(Some("codex-no-fallback-on-api-error"));
}
