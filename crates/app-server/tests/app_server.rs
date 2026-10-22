//! The app server over an in-memory pipe: a harness with the coding tools and a scripted
//! model, driven by a JSON-RPC client as Codex's clients drive Codex's app-server.

#![allow(clippy::unwrap_used, clippy::too_many_lines)] // Test helpers outside #[test] functions.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use delta_ai::Models;
use delta_ai::providers::faux::{
    FauxOptions, FauxProvider, FauxResponse, faux_assistant_message, faux_text, faux_thinking, faux_tool_call,
};
use delta_app_server::{AppServer, AppServerOptions};
use delta_chord::context::Context;
use delta_claude_hooks::{ClaudeHooks, ClaudeHooksOptions, HooksConfig};
use delta_harness::agent::LocalEnvBuilder;
use delta_harness::approval::{AllowAll, ApprovalPolicy, AskAll, PermissionDefaults, RulesPolicy};
use delta_harness::harness::{Harness, HarnessOptions};
use delta_harness::model::{ModelRef, ModelsClient};
use delta_harness::registry::{Extension, Registry};
use delta_harness::storage::MemoryStorage;
use delta_harness::tools::{BashOptions, coding_tools};
use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines, ReadHalf, WriteHalf};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const WAIT: Duration = Duration::from_secs(20);

fn ctx() -> Context {
    Context::background()
}

/// A directory under the system temp directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let name = format!(
            "delta-app-server-{}-{nanos}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let path = std::env::temp_dir().join(name);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn cwd(&self) -> String {
        self.0.to_string_lossy().into_owned()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A harness and the server over it.
struct Setup {
    harness: Harness,
    server: AppServer,
    faux: FauxProvider,
    dir: TempDir,
}

struct Build {
    policy: Arc<dyn ApprovalPolicy>,
    extensions: Vec<Extension>,
    faux: FauxOptions,
}

impl Default for Build {
    fn default() -> Self {
        Self {
            policy: Arc::new(AllowAll),
            extensions: Vec::new(),
            faux: FauxOptions::default(),
        }
    }
}

async fn setup(build: Build) -> Setup {
    let dir = TempDir::new();
    let faux = FauxProvider::new(build.faux);
    let models = Models::default();
    models.set_provider(faux.clone());
    let registry = Registry::new();
    registry.install(coding_tools(BashOptions::default())).unwrap();
    for extension in build.extensions {
        registry.install(extension).unwrap();
    }
    let mut options = HarnessOptions::new(Arc::new(ModelsClient::new(models)));
    options.registry = registry;
    options.env = Some(Arc::new(LocalEnvBuilder::new(dir.path())));
    options.approval = build.policy;
    let harness = Harness::open(Arc::new(MemoryStorage::new()), options, &ctx())
        .await
        .unwrap();
    let mut server_options = AppServerOptions::new(dir.cwd());
    server_options.model = Some(ModelRef::new("faux", "faux-1"));
    let server = AppServer::new(harness.clone(), server_options);
    Setup {
        harness,
        server,
        faux,
        dir,
    }
}

fn text(text: &str) -> FauxResponse {
    faux_assistant_message(vec![faux_text(text)]).into()
}

fn bash(command: &str) -> FauxResponse {
    faux_assistant_message(vec![faux_tool_call("bash", json!({ "command": command }))]).into()
}

/// A JSON-RPC client of one connection.
struct Client {
    writer: WriteHalf<DuplexStream>,
    lines: Lines<BufReader<ReadHalf<DuplexStream>>>,
    next_id: i64,
    backlog: VecDeque<Value>,
    /// Every message `next` returned.
    seen: Vec<Value>,
    served: tokio::task::JoinHandle<()>,
}

impl Client {
    fn connect(server: &AppServer) -> Self {
        let (client, server_side) = tokio::io::duplex(1 << 20);
        let (server_read, server_write) = tokio::io::split(server_side);
        let server = server.clone();
        let served = tokio::spawn(async move {
            server.serve(server_read, server_write, &ctx()).await.unwrap();
        });
        let (read, writer) = tokio::io::split(client);
        Self {
            writer,
            lines: BufReader::new(read).lines(),
            next_id: 0,
            backlog: VecDeque::new(),
            seen: Vec::new(),
            served,
        }
    }

    /// Connects and initializes.
    async fn open(server: &AppServer) -> Self {
        let mut client = Self::connect(server);
        let initialized = client
            .request(
                "initialize",
                json!({ "clientInfo": { "name": "test", "title": null, "version": "0" } }),
            )
            .await;
        assert!(initialized["userAgent"].as_str().unwrap().starts_with("delta/"));
        client.send(&json!({ "method": "initialized" })).await;
        client
    }

    async fn send(&mut self, message: &Value) {
        let mut line = serde_json::to_string(message).unwrap();
        line.push('\n');
        self.writer.write_all(line.as_bytes()).await.unwrap();
    }

    async fn read(&mut self) -> Value {
        let line = tokio::time::timeout(WAIT, self.lines.next_line())
            .await
            .expect("a message in time")
            .unwrap()
            .expect("the connection is open");
        let message: Value = serde_json::from_str(&line).unwrap();
        assert!(message.get("jsonrpc").is_none(), "{message}");
        message
    }

    /// The next message, from the backlog first.
    async fn next(&mut self) -> Value {
        let message = match self.backlog.pop_front() {
            Some(message) => message,
            None => self.read().await,
        };
        self.seen.push(message.clone());
        message
    }

    /// Sends a request and returns its whole response; other messages wait in the backlog.
    async fn call(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({ "method": method, "id": id, "params": params }))
            .await;
        loop {
            let message = self.read().await;
            if message.get("id") == Some(&json!(id)) && message.get("method").is_none() {
                return message;
            }
            self.backlog.push_back(message);
        }
    }

    /// Sends a request and returns its result.
    async fn request(&mut self, method: &str, params: Value) -> Value {
        let response = self.call(method, params).await;
        assert!(response.get("error").is_none(), "{method}: {response}");
        response["result"].clone()
    }

    /// The messages up to and including the first that `stop` accepts.
    async fn until(&mut self, stop: impl Fn(&Value) -> bool) -> Vec<Value> {
        let mut seen = Vec::new();
        loop {
            let message = self.next().await;
            let done = stop(&message);
            seen.push(message);
            if done {
                return seen;
            }
        }
    }

    /// The messages up to and including the first with `method`.
    async fn until_method(&mut self, method: &str) -> Vec<Value> {
        self.until(|message| message["method"] == method).await
    }

    async fn turn_completed(&mut self) -> (Vec<Value>, Value) {
        let mut seen = self.until_method("turn/completed").await;
        let completed = seen.pop().unwrap();
        (seen, completed["params"]["turn"].clone())
    }

    async fn respond(&mut self, id: &Value, result: Value) {
        self.send(&json!({ "id": id, "result": result })).await;
    }

    async fn start_thread(&mut self) -> String {
        let session = self.request("thread/start", json!({})).await;
        let started = self.until_method("thread/started").await;
        assert_eq!(
            started.last().unwrap()["params"]["thread"]["id"],
            session["thread"]["id"]
        );
        session["thread"]["id"].as_str().unwrap().to_owned()
    }

    async fn start_turn(&mut self, thread: &str, input: &str) -> String {
        let response = self
            .request(
                "turn/start",
                json!({ "threadId": thread, "input": [{ "type": "text", "text": input, "text_elements": [] }] }),
            )
            .await;
        assert_eq!(response["turn"]["status"], "inProgress");
        response["turn"]["id"].as_str().unwrap().to_owned()
    }

    /// Closes the client's side and waits for the server to finish.
    async fn close(self) {
        drop(self.writer);
        drop(self.lines);
        tokio::time::timeout(WAIT, self.served).await.unwrap().unwrap();
    }
}

fn methods(messages: &[Value]) -> Vec<&str> {
    messages
        .iter()
        .filter_map(|message| message["method"].as_str())
        .collect()
}

fn of_method<'a>(messages: &'a [Value], method: &str) -> Vec<&'a Value> {
    messages.iter().filter(|message| message["method"] == method).collect()
}

fn deltas(messages: &[Value], method: &str, item: &str) -> String {
    of_method(messages, method)
        .iter()
        .filter(|message| message["params"]["itemId"] == item)
        .map(|message| message["params"]["delta"].as_str().unwrap())
        .collect()
}

/// The `item/completed` items of `messages`.
fn completed(messages: &[Value]) -> Vec<Value> {
    of_method(messages, "item/completed")
        .iter()
        .map(|message| message["params"]["item"].clone())
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn requests_need_initialize_and_known_methods() {
    let setup = setup(Build::default()).await;
    let mut client = Client::connect(&setup.server);
    let early = client.call("thread/start", json!({})).await;
    assert_eq!(early["error"]["code"], -32600);
    client
        .request(
            "initialize",
            json!({ "clientInfo": { "name": "test", "version": "0" }, "capabilities": { "experimentalApi": false } }),
        )
        .await;
    let again = client
        .call(
            "initialize",
            json!({ "clientInfo": { "name": "test", "version": "0" } }),
        )
        .await;
    assert_eq!(again["error"]["code"], -32600);
    let unknown = client.call("account/read", json!({})).await;
    assert_eq!(unknown["error"]["code"], -32601);
    let invalid = client.call("turn/start", json!({ "threadId": 1 })).await;
    assert_eq!(invalid["error"]["code"], -32602);
    let missing = client.call("thread/read", json!({ "threadId": "999" })).await;
    assert_eq!(missing["error"]["code"], -32602);
    client.writer.write_all(b"{not json\n").await.unwrap();
    let parse = client.next().await;
    assert_eq!(parse["error"]["code"], -32700);
    assert_eq!(parse["id"], Value::Null);
    client.close().await;
    setup.harness.close(&ctx()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_turn_streams_its_answer() {
    let setup = setup(Build {
        faux: FauxOptions {
            tokens_per_second: Some(60.0),
            token_size: Some((1, 1)),
            ..FauxOptions::default()
        },
        ..Build::default()
    })
    .await;
    let answer = "Hello there, this answer streams in several pieces.";
    setup.faux.set_responses(vec![text(answer)]);
    let mut client = Client::open(&setup.server).await;
    let session = client.request("thread/start", json!({})).await;
    assert_eq!(session["model"], "faux-1");
    assert_eq!(session["modelProvider"], "faux");
    assert_eq!(session["cwd"], setup.dir.cwd());
    assert_eq!(session["approvalPolicy"], "on-request");
    assert_eq!(session["thread"]["status"], json!({ "type": "idle" }));
    let thread = session["thread"]["id"].as_str().unwrap().to_owned();
    let turn = client.start_turn(&thread, "hi").await;
    let (messages, completed_turn) = client.turn_completed().await;
    assert_eq!(completed_turn["id"], turn);
    assert_eq!(completed_turn["status"], "completed");
    let started = of_method(&messages, "turn/started");
    assert_eq!(started.len(), 1);
    assert_eq!(started[0]["params"]["turn"]["id"], turn);
    // The user message, then the answer streamed into one agent message.
    let items = completed(&messages);
    assert_eq!(items[0]["type"], "userMessage");
    assert_eq!(items[0]["id"], turn);
    assert_eq!(items[0]["content"][0]["text"], "hi");
    let message = items.iter().find(|item| item["type"] == "agentMessage").unwrap();
    assert_eq!(message["text"], answer);
    let id = message["id"].as_str().unwrap();
    assert!(id.ends_with(":0"), "{id}");
    let streamed = deltas(&messages, "item/agentMessage/delta", id);
    assert_eq!(streamed, answer);
    assert!(of_method(&messages, "item/agentMessage/delta").len() > 1);
    // Started before its deltas, completed after them.
    let position = |method: &str| messages.iter().position(|message| message["method"] == method).unwrap();
    assert!(position("item/started") < position("item/agentMessage/delta"));
    for message in &messages {
        if message["method"] == "item/started" || message["method"] == "item/completed" {
            assert_eq!(message["params"]["turnId"], turn);
            assert_eq!(message["params"]["threadId"], thread);
        }
    }
    let usage = of_method(&messages, "thread/tokenUsage/updated");
    assert!(!usage.is_empty());
    assert!(
        usage.last().unwrap()["params"]["tokenUsage"]["total"]["totalTokens"]
            .as_i64()
            .unwrap()
            > 0
    );
    // Active while the turn runs, idle after it.
    let statuses: Vec<&Value> = of_method(&messages, "thread/status/changed")
        .iter()
        .map(|message| &message["params"]["status"])
        .collect();
    assert_eq!(statuses.first().unwrap()["type"], "active");
    let idle = client.until_method("thread/status/changed").await;
    assert_eq!(idle.last().unwrap()["params"]["status"]["type"], "idle");
    // Every notification carries its emission time.
    assert!(messages.iter().all(|message| message["emittedAtMs"].as_i64().is_some()));
    client.close().await;
    setup.harness.close(&ctx()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reasoning_and_text_are_separate_items() {
    let setup = setup(Build::default()).await;
    setup.faux.set_responses(vec![
        faux_assistant_message(vec![faux_thinking("Let me think."), faux_text("Done.")]).into(),
    ]);
    let mut client = Client::open(&setup.server).await;
    let thread = client.start_thread().await;
    client.start_turn(&thread, "think").await;
    let (messages, _) = client.turn_completed().await;
    let items = completed(&messages);
    let kinds: Vec<&str> = items.iter().map(|item| item["type"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["userMessage", "reasoning", "agentMessage"]);
    let reasoning = &items[1];
    let message = &items[2];
    assert_eq!(reasoning["content"], json!(["Let me think."]));
    assert_eq!(message["text"], "Done.");
    let (task, index) = reasoning["id"].as_str().unwrap().split_once(':').unwrap();
    assert_eq!(index, "0");
    assert_eq!(message["id"], format!("{task}:1"));
    client.close().await;
    setup.harness.close(&ctx()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commands_and_file_changes_are_items() {
    let setup = setup(Build::default()).await;
    std::fs::write(setup.dir.path().join("notes.txt"), "old line\n").unwrap();
    let edit = faux_assistant_message(vec![faux_tool_call(
        "edit",
        json!({ "path": "notes.txt", "edits": [{ "oldText": "old line", "newText": "new line" }] }),
    )]);
    setup.faux.set_responses(vec![
        bash("echo hi; echo there"),
        edit.into(),
        bash("exit 3"),
        text("done"),
    ]);
    let mut client = Client::open(&setup.server).await;
    let thread = client.start_thread().await;
    client.start_turn(&thread, "work").await;
    let (messages, turn) = client.turn_completed().await;
    assert_eq!(turn["status"], "completed");
    let items = completed(&messages);
    let commands: Vec<&Value> = items.iter().filter(|item| item["type"] == "commandExecution").collect();
    assert_eq!(commands.len(), 2);
    let echo = commands[0];
    assert_eq!(echo["command"], "echo hi; echo there");
    assert_eq!(echo["cwd"], setup.dir.cwd());
    assert_eq!(echo["status"], "completed");
    assert_eq!(echo["exitCode"], 0);
    assert_eq!(echo["aggregatedOutput"], "hi\nthere\n");
    assert_eq!(
        echo["commandActions"],
        json!([{ "type": "unknown", "command": "echo hi; echo there" }])
    );
    let id = echo["id"].as_str().unwrap();
    assert_eq!(
        deltas(&messages, "item/commandExecution/outputDelta", id),
        "hi\nthere\n"
    );
    let started: Vec<&Value> = of_method(&messages, "item/started")
        .into_iter()
        .filter(|message| message["params"]["item"]["id"] == id)
        .collect();
    assert_eq!(started.len(), 1);
    assert_eq!(started[0]["params"]["item"]["status"], "inProgress");
    let failed = commands[1];
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["exitCode"], 3);
    let change = items.iter().find(|item| item["type"] == "fileChange").unwrap();
    assert_eq!(change["status"], "completed");
    assert_eq!(change["changes"][0]["path"], "notes.txt");
    assert_eq!(
        change["changes"][0]["kind"],
        json!({ "type": "update", "move_path": null })
    );
    let diff = change["changes"][0]["diff"].as_str().unwrap();
    assert!(diff.contains("-old line") && diff.contains("+new line"), "{diff}");
    client.close().await;
    setup.harness.close(&ctx()).await.unwrap();
}

/// The approval request of the next call, after its in-progress item.
async fn approval_request(client: &mut Client, method: &str) -> (Value, Value) {
    let messages = client.until_method(method).await;
    let request = messages.last().unwrap().clone();
    let item = request["params"]["itemId"].clone();
    let started = of_method(&messages, "item/started")
        .into_iter()
        .any(|message| message["params"]["item"]["id"] == item);
    assert!(started, "{messages:?}");
    (request["id"].clone(), request["params"].clone())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn approvals_are_server_requests() {
    let setup = setup(Build {
        policy: Arc::new(AskAll),
        ..Build::default()
    })
    .await;
    std::fs::write(setup.dir.path().join("a.txt"), "a\n").unwrap();
    let write = faux_assistant_message(vec![faux_tool_call(
        "write",
        json!({ "path": "b.txt", "content": "b\n" }),
    )]);
    let read = faux_assistant_message(vec![faux_tool_call("read", json!({ "path": "a.txt" }))]);
    setup.faux.set_responses(vec![
        bash("echo approved"),
        bash("echo declined"),
        write.into(),
        read.into(),
        text("done"),
    ]);
    let mut client = Client::open(&setup.server).await;
    let thread = client.start_thread().await;
    let turn = client.start_turn(&thread, "go").await;

    let (id, params) = approval_request(&mut client, "item/commandExecution/requestApproval").await;
    assert_eq!(params["threadId"], thread);
    assert_eq!(params["turnId"], turn);
    assert_eq!(params["command"], "echo approved");
    assert_eq!(params["cwd"], setup.dir.cwd());
    let status = client.until_method("thread/status/changed").await;
    assert_eq!(
        status.last().unwrap()["params"]["status"],
        json!({ "type": "active", "activeFlags": ["waitingOnApproval"] })
    );
    client.respond(&id, json!({ "decision": "accept" })).await;
    let resolved = client.until_method("serverRequest/resolved").await;
    assert_eq!(
        resolved.last().unwrap()["params"],
        json!({ "threadId": thread, "requestId": id })
    );

    let (id, _) = approval_request(&mut client, "item/commandExecution/requestApproval").await;
    client.respond(&id, json!({ "decision": "decline" })).await;

    let (id, params) = approval_request(&mut client, "item/fileChange/requestApproval").await;
    assert_eq!(params["turnId"], turn);
    client.respond(&id, json!({ "decision": "accept" })).await;

    let (id, params) = approval_request(&mut client, "delta/tool/requestApproval").await;
    assert_eq!(params["tool"], "read");
    assert_eq!(params["arguments"], json!({ "path": "a.txt" }));
    client.respond(&id, json!({ "decision": "accept" })).await;

    let (_, completed_turn) = client.turn_completed().await;
    assert_eq!(completed_turn["status"], "completed");
    let items = completed(&client.seen);
    let commands: Vec<&Value> = items.iter().filter(|item| item["type"] == "commandExecution").collect();
    // The approved command ran; the declined one did not.
    assert!(
        commands
            .iter()
            .any(|item| item["command"] == "echo approved" && item["status"] == "completed")
    );
    assert_eq!(std::fs::read_to_string(setup.dir.path().join("b.txt")).unwrap(), "b\n");
    let read = items.iter().find(|item| item["type"] == "dynamicToolCall").unwrap();
    assert_eq!(read["tool"], "read");
    assert_eq!(read["success"], true);
    assert_eq!(read["status"], "completed");
    client.close().await;
    setup.harness.close(&ctx()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn declined_commands_complete_declined() {
    let setup = setup(Build {
        policy: Arc::new(AskAll),
        ..Build::default()
    })
    .await;
    setup.faux.set_responses(vec![bash("echo no"), text("ok")]);
    let mut client = Client::open(&setup.server).await;
    let thread = client.start_thread().await;
    client.start_turn(&thread, "go").await;
    let (id, params) = approval_request(&mut client, "item/commandExecution/requestApproval").await;
    let item = params["itemId"].clone();
    client.respond(&id, json!({ "decision": "decline" })).await;
    let (messages, _) = client.turn_completed().await;
    let declined = completed(&messages)
        .into_iter()
        .find(|completed| completed["id"] == item)
        .unwrap();
    assert_eq!(declined["status"], "declined");
    assert_eq!(declined["exitCode"], Value::Null);
    client.close().await;
    setup.harness.close(&ctx()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_interrupts_the_turn() {
    let setup = setup(Build {
        policy: Arc::new(AskAll),
        ..Build::default()
    })
    .await;
    setup.faux.set_responses(vec![bash("echo never")]);
    let mut client = Client::open(&setup.server).await;
    let thread = client.start_thread().await;
    let turn = client.start_turn(&thread, "go").await;
    let (id, _) = approval_request(&mut client, "item/commandExecution/requestApproval").await;
    client.respond(&id, json!({ "decision": "cancel" })).await;
    let (_, completed_turn) = client.turn_completed().await;
    assert_eq!(completed_turn["id"], turn);
    assert_eq!(completed_turn["status"], "interrupted");
    client.close().await;
    setup.harness.close(&ctx()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn interrupt_and_steer_check_the_running_turn() {
    let setup = setup(Build {
        faux: FauxOptions {
            tokens_per_second: Some(20.0),
            token_size: Some((1, 1)),
            ..FauxOptions::default()
        },
        ..Build::default()
    })
    .await;
    setup
        .faux
        .set_responses(vec![text(&"A long and slow answer. ".repeat(40))]);
    let mut client = Client::open(&setup.server).await;
    let thread = client.start_thread().await;
    let turn = client.start_turn(&thread, "talk").await;
    client.until_method("item/agentMessage/delta").await;
    let wrong = client
        .call(
            "turn/steer",
            json!({ "threadId": thread, "input": [{ "type": "text", "text": "x" }], "expectedTurnId": "12345" }),
        )
        .await;
    assert_eq!(wrong["error"]["code"], -32600);
    let steered = client
        .request(
            "turn/steer",
            json!({ "threadId": thread, "input": [{ "type": "text", "text": "faster" }], "expectedTurnId": turn }),
        )
        .await;
    assert_eq!(steered["turnId"], turn);
    let interrupted = client
        .request("turn/interrupt", json!({ "threadId": thread, "turnId": turn }))
        .await;
    assert_eq!(interrupted, json!({}));
    let (_, completed_turn) = client.turn_completed().await;
    assert_eq!(completed_turn["id"], turn);
    assert_eq!(completed_turn["status"], "interrupted");
    client.close().await;
    setup.harness.close(&ctx()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_second_client_resumes_with_the_pending_approval() {
    let setup = setup(Build {
        policy: Arc::new(AskAll),
        ..Build::default()
    })
    .await;
    setup.faux.set_responses(vec![bash("echo resumed"), text("ok")]);
    let mut first = Client::open(&setup.server).await;
    let thread = first.start_thread().await;
    let turn = first.start_turn(&thread, "go").await;
    let (first_id, _) = approval_request(&mut first, "item/commandExecution/requestApproval").await;

    let mut second = Client::open(&setup.server).await;
    let session = second.request("thread/resume", json!({ "threadId": thread })).await;
    let turns = session["thread"]["turns"].as_array().unwrap();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0]["id"], turn);
    assert_eq!(turns[0]["status"], "inProgress");
    assert_eq!(
        session["thread"]["status"],
        json!({ "type": "active", "activeFlags": ["waitingOnApproval"] })
    );
    let (second_id, params) = approval_request(&mut second, "item/commandExecution/requestApproval").await;
    assert_eq!(params["command"], "echo resumed");
    second.respond(&second_id, json!({ "decision": "accept" })).await;
    // Both clients learn the request resolved, and the turn completes for both.
    let resolved = first.until_method("serverRequest/resolved").await;
    assert_eq!(resolved.last().unwrap()["params"]["requestId"], first_id);
    let (_, done) = first.turn_completed().await;
    assert_eq!(done["status"], "completed");
    let (_, done) = second.turn_completed().await;
    assert_eq!(done["id"], turn);
    // Resuming a followed thread sends nothing pending.
    let again = second.request("thread/resume", json!({ "threadId": thread })).await;
    assert_eq!(again["thread"]["turns"][0]["status"], "completed");
    first.close().await;
    second.close().await;
    setup.harness.close(&ctx()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_fork_and_list_rebuild_turns() {
    let setup = setup(Build::default()).await;
    let seen = Arc::new(std::sync::Mutex::new(String::new()));
    let capture = seen.clone();
    let forked = FauxResponse::Factory(Arc::new(move |context, _, _| {
        *capture.lock().unwrap() = serde_json::to_string(context.messages()).unwrap();
        faux_assistant_message(vec![faux_text("forked answer")])
    }));
    setup.faux.set_responses(vec![
        bash("echo listed"),
        text("first answer"),
        text("second answer"),
        forked,
    ]);
    let mut client = Client::open(&setup.server).await;
    let thread = client.start_thread().await;
    let first = client.start_turn(&thread, "one").await;
    let (live, _) = client.turn_completed().await;
    let second = client.start_turn(&thread, "two").await;
    client.turn_completed().await;

    let read = client
        .request("thread/read", json!({ "threadId": thread, "includeTurns": true }))
        .await;
    let turns = read["thread"]["turns"].as_array().unwrap();
    assert_eq!(turns.len(), 2);
    assert_eq!(turns[0]["id"], first);
    assert_eq!(turns[1]["id"], second);
    assert!(turns.iter().all(|turn| turn["status"] == "completed"));
    assert!(turns[0]["startedAt"].as_i64().unwrap() > 0);
    // The rebuilt items are the live items.
    let live_items = completed(&live);
    assert_eq!(turns[0]["items"].as_array().unwrap(), &live_items);
    assert_eq!(read["thread"]["preview"], "one");
    let without = client.request("thread/read", json!({ "threadId": thread })).await;
    assert_eq!(without["thread"]["turns"], json!([]));

    let fork = client
        .request("thread/fork", json!({ "threadId": thread, "lastTurnId": first }))
        .await;
    let fork_id = fork["thread"]["id"].as_str().unwrap().to_owned();
    assert_eq!(fork["thread"]["forkedFromId"], thread);
    let fork_turns = fork["thread"]["turns"].as_array().unwrap();
    assert_eq!(fork_turns.len(), 1);
    assert_eq!(fork_turns[0]["id"], first);
    let started = client.until_method("thread/started").await;
    assert_eq!(started.last().unwrap()["params"]["thread"]["id"], fork_id);
    let forked_turn = client.start_turn(&fork_id, "again").await;
    let (_, done) = client.turn_completed().await;
    assert_eq!(done["id"], forked_turn);
    // The fork's model saw the first turn only.
    let seen = seen.lock().unwrap().clone();
    assert!(
        seen.contains("first answer") && !seen.contains("second answer"),
        "{seen}"
    );

    let list = client.request("thread/list", json!({ "limit": 1 })).await;
    assert_eq!(list["data"].as_array().unwrap().len(), 1);
    assert_eq!(list["data"][0]["id"], fork_id);
    let cursor = list["nextCursor"].as_str().unwrap();
    let rest = client
        .request("thread/list", json!({ "cursor": cursor, "limit": 10 }))
        .await;
    let ids: Vec<&str> = rest["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|thread| thread["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, [thread.as_str()]);
    assert_eq!(rest["data"][0]["status"]["type"], "idle");
    client.close().await;
    setup.harness.close(&ctx()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn permissions_are_read_and_updated() {
    let setup = setup(Build::default()).await;
    let mut client = Client::open(&setup.server).await;
    let thread = client.start_thread().await;
    let read = client
        .request("delta/permissions/read", json!({ "threadId": thread }))
        .await;
    assert_eq!(
        read["permissions"],
        json!({ "mode": null, "effectiveMode": "default", "rules": { "allow": [], "ask": [], "deny": [] }, "directories": [] })
    );
    let updated = client
        .request(
            "delta/permissions/update",
            json!({ "threadId": thread, "mode": "acceptEdits", "add": { "deny": ["Bash(rm:*)"] }, "directories": ["/tmp"] }),
        )
        .await;
    assert_eq!(updated["permissions"]["mode"], "acceptEdits");
    assert_eq!(updated["permissions"]["effectiveMode"], "acceptEdits");
    assert_eq!(updated["permissions"]["rules"]["deny"], json!(["Bash(rm:*)"]));
    assert_eq!(updated["permissions"]["directories"], json!(["/tmp"]));
    let invalid = client
        .call(
            "delta/permissions/update",
            json!({ "threadId": thread, "add": { "allow": ["Bash(unbalanced"] } }),
        )
        .await;
    assert_eq!(invalid["error"]["code"], -32602);
    let cleared = client
        .request(
            "delta/permissions/update",
            json!({ "threadId": thread, "clearMode": true }),
        )
        .await;
    assert_eq!(cleared["permissions"]["mode"], Value::Null);
    let rules = client.request("delta/permissionRules/read", Value::Null).await;
    assert_eq!(rules["rules"]["allow"], json!([]));
    client.close().await;
    setup.harness.close(&ctx()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn opted_out_notifications_are_not_sent() {
    let setup = setup(Build::default()).await;
    setup.faux.set_responses(vec![text("quiet")]);
    let mut client = Client::connect(&setup.server);
    client
        .request(
            "initialize",
            json!({ "clientInfo": { "name": "test", "version": "0" }, "capabilities": { "optOutNotificationMethods": ["item/started", "thread/tokenUsage/updated"] } }),
        )
        .await;
    let thread = client.start_thread().await;
    client.start_turn(&thread, "hi").await;
    let (messages, _) = client.turn_completed().await;
    assert!(
        of_method(&messages, "item/started").is_empty(),
        "{:?}",
        methods(&messages)
    );
    assert!(of_method(&messages, "thread/tokenUsage/updated").is_empty());
    assert!(!of_method(&messages, "item/completed").is_empty());
    client.close().await;
    setup.harness.close(&ctx()).await.unwrap();
}

/// The rules policy and a Claude Code `PreToolUse` HTTP hook that asks once: the client
/// accepts with an exec policy amendment, the rule is stored for every thread, and the same
/// command later runs without asking.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_amendment_from_a_hook_ask_becomes_a_rule() {
    let hook_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/pre"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "ask",
                "permissionDecisionReason": "Check this command"
            }
        })))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&hook_server)
        .await;
    Mock::given(method("POST"))
        .and(path("/pre"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .with_priority(2)
        .mount(&hook_server)
        .await;
    let config = HooksConfig::from_value(json!({
        "PreToolUse": [{ "matcher": "Bash", "hooks": [{ "type": "http", "url": format!("{}/pre", hook_server.uri()) }] }]
    }))
    .unwrap();
    let hooks = ClaudeHooks::new(
        config,
        ClaudeHooksOptions {
            client: Some(reqwest::Client::builder().no_proxy().build().unwrap()),
            ..ClaudeHooksOptions::default()
        },
    );
    let setup = setup(Build {
        policy: Arc::new(RulesPolicy::new(PermissionDefaults::default()).unwrap()),
        extensions: vec![hooks.extension()],
        ..Build::default()
    })
    .await;
    setup.faux.set_responses(vec![
        bash("git status"),
        text("first"),
        bash("git status"),
        text("second"),
    ]);
    let mut client = Client::open(&setup.server).await;
    let thread = client.start_thread().await;
    client.start_turn(&thread, "check").await;
    let (id, params) = approval_request(&mut client, "item/commandExecution/requestApproval").await;
    assert_eq!(params["reason"], "Check this command");
    assert_eq!(params["command"], "git status");
    assert_eq!(params["proposedExecpolicyAmendment"], json!(["git", "status"]));
    client
        .respond(
            &id,
            json!({ "decision": { "acceptWithExecpolicyAmendment": { "execpolicy_amendment": ["git", "status"] } } }),
        )
        .await;
    let (_, done) = client.turn_completed().await;
    assert_eq!(done["status"], "completed");
    let rules = client.request("delta/permissionRules/read", Value::Null).await;
    assert_eq!(rules["rules"]["allow"], json!(["Bash(git status:*)"]));

    // The second turn's identical command runs without a request.
    client.start_turn(&thread, "again").await;
    let (messages, done) = client.turn_completed().await;
    assert_eq!(done["status"], "completed");
    assert!(
        of_method(&messages, "item/commandExecution/requestApproval").is_empty(),
        "{:?}",
        methods(&messages)
    );
    let command = completed(&messages)
        .into_iter()
        .find(|item| item["type"] == "commandExecution")
        .unwrap();
    assert_ne!(command["status"], "declined");
    let removed = client
        .request("delta/permissionRules/remove", json!({ "rule": "Bash(git status:*)" }))
        .await;
    assert_eq!(removed["rules"]["allow"], json!([]));
    client.close().await;
    setup.harness.close(&ctx()).await.unwrap();
}
