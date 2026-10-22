//! The agent end to end: scripted answers from the faux provider, commands in a temporary
//! directory, and scripted answers for the interactive agent.

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use delta_ai::AssistantMessageEventStream;
use delta_ai::model::Model;
use delta_ai::types::{DeferredHandle, Message};
use delta_chord::context::Context;
use delta_harness::model::{ModelClient, ModelError, ModelRequest};
use delta_mini::agent::{Agent, AgentKind, AgentOptions, Frontend, Input, RunOutput, Scripted};
use delta_mini::config;
use delta_mini::env::{LocalEnvironment, LocalEnvironmentConfig};
use delta_mini::model::{MiniModelClient, ModelSettings};
use delta_mini::trajectory::{Entry, ExitStatus, InterruptReason, Trajectory, message_text};
use futures::future::BoxFuture;
use pretty_assertions::assert_eq;
use serde_json::{Value as Json, json};

/// A script item calling bash with each command, at a cost of 1.
fn calls(commands: &[&str]) -> Json {
    let mut content = vec![json!({ "type": "text", "text": "Running." })];
    content.extend(commands.iter().enumerate().map(|(index, command)| {
        json!({
            "type": "toolCall",
            "id": format!("call_{}_{index}", uuid::Uuid::new_v4().simple()),
            "name": "bash",
            "arguments": { "command": command },
        })
    }));
    json!({ "content": content, "cost": 1.0 })
}

/// A text-only script item, at a cost of 1.
fn text(content: &str) -> Json {
    json!({ "content": [{ "type": "text", "text": content }], "cost": 1.0 })
}

const SUBMIT: &str = "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nthe patch\\n'";

/// Records each request's messages, then streams as the wrapped client does.
struct Recording {
    inner: MiniModelClient,
    requests: Mutex<Vec<Vec<Message>>>,
}

impl ModelClient for Recording {
    fn model(&self, reference: &delta_harness::model::ModelRef) -> Option<Model> {
        self.inner.model(reference)
    }

    fn stream(&self, request: ModelRequest<'_>, ctx: &Context) -> AssistantMessageEventStream {
        self.requests.lock().unwrap().push(request.messages.clone());
        self.inner.stream(request, ctx)
    }

    fn fetch_deferred<'a>(
        &'a self,
        model: &'a Model,
        handle: &'a DeferredHandle,
        ctx: &'a Context,
    ) -> BoxFuture<'a, delta_ai::types::AssistantMessage> {
        self.inner.fetch_deferred(model, handle, ctx)
    }

    fn cancel_deferred<'a>(
        &'a self,
        model: &'a Model,
        handle: &'a DeferredHandle,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<(), ModelError>> {
        self.inner.cancel_deferred(model, handle, ctx)
    }
}

struct Run {
    output: RunOutput,
    dir: tempfile::TempDir,
    requests: Vec<Vec<Message>>,
}

struct Setup {
    specs: Vec<String>,
    script: Vec<Json>,
    agent: Json,
    frontend: Option<Arc<Scripted>>,
    kind: AgentKind,
    interrupt_after: Option<Duration>,
}

impl Setup {
    fn tool_calls(script: Vec<Json>) -> Self {
        Self {
            specs: vec!["mini.yaml".to_owned(), "model.provider=faux".to_owned()],
            script,
            agent: json!({}),
            frontend: None,
            kind: AgentKind::Default,
            interrupt_after: None,
        }
    }

    fn text(script: Vec<Json>) -> Self {
        Self {
            specs: vec!["mini_textbased.yaml".to_owned(), "model.provider=faux".to_owned()],
            ..Self::tool_calls(script)
        }
    }

    fn agent(mut self, agent: Json) -> Self {
        self.agent = agent;
        self
    }

    fn interactive(mut self, answers: Vec<Input>) -> Self {
        self.frontend = Some(Arc::new(Scripted::new(answers, true)));
        self.kind = AgentKind::Interactive;
        self
    }

    async fn run(self) -> Run {
        let dir = tempfile::tempdir().unwrap();
        let mut config = config::load(&self.specs).unwrap();
        config["model"]["script"] = Json::Array(self.script);
        config::merge(&mut config, json!({ "agent": self.agent }));
        let model = ModelSettings::new(&config["model"]).unwrap();
        let (client, warning) = MiniModelClient::new(&model, &Json::Null, None).unwrap();
        assert_eq!(warning, None);
        let resolved = client.resolved().clone();
        let recording = Arc::new(Recording {
            inner: client,
            requests: Mutex::new(Vec::new()),
        });
        let env = LocalEnvironment::new(LocalEnvironmentConfig {
            cwd: dir.path().display().to_string(),
            ..LocalEnvironmentConfig::default()
        });
        let agent = Agent::new(AgentOptions {
            config,
            kind: self.kind,
            model,
            resolved,
            client: Arc::clone(&recording) as Arc<dyn ModelClient>,
            env: Arc::new(env),
            frontend: self.frontend.clone().map(|frontend| frontend as Arc<dyn Frontend>),
            output: Some(dir.path().join("run.traj.json")),
            extra: serde_json::Map::new(),
            handle_ctrl_c: false,
            on_step: None,
            batch: None,
        })
        .unwrap();
        if let Some(delay) = self.interrupt_after {
            let interrupter = agent.interrupter();
            tokio::spawn(async move {
                tokio::time::sleep(delay).await;
                interrupter.notify_one();
            });
        }
        let output = tokio::time::timeout(
            Duration::from_secs(60),
            agent.run("Fix the bug.", serde_json::Map::new()),
        )
        .await
        .expect("the run finishes")
        .unwrap();
        let requests = recording.requests.lock().unwrap().clone();
        Run { output, dir, requests }
    }
}

fn kinds(run: &Run) -> Vec<&'static str> {
    run.output.trajectory.entries.iter().map(Entry::kind).collect()
}

fn entry(run: &Run, index: usize) -> &Entry {
    &run.output.trajectory.entries[index]
}

/// An entry as its JSON.
fn entry_json(run: &Run, index: usize) -> Json {
    serde_json::to_value(entry(run, index)).unwrap()
}

fn text_of(run: &Run, index: usize) -> String {
    message_text(entry(run, index).message())
}

fn status(run: &Run) -> ExitStatus {
    run.output.exit.status
}

/// The messages of the `index`th request as `role: text` lines.
fn request(run: &Run, index: usize) -> Vec<String> {
    run.requests[index]
        .iter()
        .map(|message| match message {
            Message::System(system) => format!(
                "system: {}",
                system
                    .sections
                    .as_ref()
                    .map(|sections| sections.values().flatten().cloned().collect::<Vec<_>>().join("|"))
                    .unwrap_or_default()
            ),
            Message::User(user) => format!("user: {}", user.content.text()),
            Message::Assistant(assistant) => format!("assistant: {}", assistant.text()),
            Message::ToolResult(_) => format!("tool: {}", message_text(message)),
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn submits_and_saves_a_trajectory() {
    let run = Setup::tool_calls(vec![calls(&["echo hello"]), calls(&[SUBMIT])])
        .run()
        .await;
    assert_eq!(status(&run), ExitStatus::Submitted);
    assert_eq!(run.output.exit.submission, "the patch\n");
    assert_eq!(
        kinds(&run),
        ["system", "task", "answer", "observation", "answer", "observation"]
    );
    let trajectory = &run.output.trajectory;
    assert_eq!(trajectory.format, "delta-mini/1");
    assert_eq!((trajectory.stats.calls, trajectory.stats.cost), (2, 2.0));
    assert_eq!(trajectory.model["provider"], "faux");
    assert_eq!(trajectory.config["environment"]["type"], "local");
    assert_eq!(trajectory.config["model"]["actions"], "tool_calls");
    assert_eq!(trajectory.config["agent"]["mode"], "confirm");

    let answer = entry_json(&run, 2);
    assert_eq!(answer["message"]["role"], "assistant");
    assert_eq!(answer["actions"][0]["command"], "echo hello");
    assert_eq!(answer["cost"], 1.0);
    let call_id = answer["message"]["content"][1]["id"].clone();
    assert_eq!(answer["actions"][0]["tool_call_id"], call_id);
    let observation = entry_json(&run, 3);
    assert_eq!(observation["message"]["role"], "toolResult");
    assert_eq!(observation["message"]["toolCallId"], call_id);
    assert_eq!(
        observation["output"],
        json!({ "output": "hello\n", "returncode": 0, "exception_info": "" })
    );
    assert!(text_of(&run, 3).contains("hello"));

    let saved = Trajectory::load(&run.dir.path().join("run.traj.json")).unwrap();
    assert_eq!(saved.entries, trajectory.entries);
    assert_eq!(saved.exit, trajectory.exit);
    assert!(run.dir.path().join("run.harness.db").exists());

    // The model saw the rendered system prompt, the task, and the tool results the
    // trajectory holds.
    let system = text_of(&run, 0);
    let first = request(&run, 0);
    assert_eq!(first.len(), 2);
    assert!(first[0].contains(system.trim()), "{first:?}");
    assert_eq!(first[1], format!("user: {}", text_of(&run, 1)));
    assert!(first[1].contains("Fix the bug."));
    let second = request(&run, 1);
    assert_eq!(second.len(), 4);
    assert_eq!(second[3], format!("tool: {}", text_of(&run, 3)));
}

#[tokio::test(flavor = "multi_thread")]
async fn repeated_format_errors_end_the_run() {
    let run = Setup::tool_calls(vec![text("a"), text("b"), text("c"), text("d")])
        .run()
        .await;
    assert_eq!(status(&run), ExitStatus::FormatErrors);
    assert_eq!(
        kinds(&run),
        [
            "system",
            "task",
            "answer",
            "format_error",
            "answer",
            "format_error",
            "answer"
        ]
    );
    let error = entry_json(&run, 3);
    assert_eq!(
        error["error"],
        "No tool calls found in the response. Every response MUST include at least one tool call."
    );
    assert_eq!(error["message"]["role"], "user");
    assert_eq!(entry_json(&run, 2)["actions"], json!([]));
    assert_eq!(run.output.trajectory.stats.calls, 3);
    // The model saw each error as a continuation.
    assert_eq!(run.requests.len(), 3);
    assert_eq!(request(&run, 1)[3], format!("user: {}", text_of(&run, 3)));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_format_error_then_a_clean_step_resets_the_count() {
    let run = Setup::tool_calls(vec![
        text("a"),
        text("b"),
        calls(&["true"]),
        text("c"),
        calls(&[SUBMIT]),
    ])
    .agent(json!({ "cost_limit": 0 }))
    .run()
    .await;
    assert_eq!(status(&run), ExitStatus::Submitted);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_step_limit_ends_the_run_after_the_round() {
    let run = Setup::tool_calls(vec![calls(&["echo one"]), calls(&["echo two"])])
        .agent(json!({ "step_limit": 1 }))
        .run()
        .await;
    assert_eq!(status(&run), ExitStatus::LimitsExceeded);
    assert_eq!(kinds(&run), ["system", "task", "answer", "observation"]);
    assert_eq!(run.requests.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_cost_limit_ends_the_run() {
    let run = Setup::tool_calls(vec![calls(&["echo one"]), calls(&["echo two"]), calls(&["echo three"])])
        .agent(json!({ "cost_limit": 2.0 }))
        .run()
        .await;
    assert_eq!(status(&run), ExitStatus::LimitsExceeded);
    assert_eq!(run.requests.len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_time_limit_ends_the_run() {
    let run = Setup::tool_calls(vec![calls(&["sleep 1.2"]), calls(&["echo late"])])
        .agent(json!({ "time_limit": 1 }))
        .run()
        .await;
    assert_eq!(status(&run), ExitStatus::TimeExceeded);
    assert_eq!(run.requests.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_submission_mid_round_skips_the_rest() {
    let run = Setup::tool_calls(vec![calls(&["echo a", SUBMIT, "echo c"])])
        .run()
        .await;
    assert_eq!(status(&run), ExitStatus::Submitted);
    assert_eq!(
        kinds(&run),
        ["system", "task", "answer", "observation", "observation", "observation"]
    );
    assert_eq!(
        entry_json(&run, 5)["output"],
        json!({ "output": "", "returncode": -1, "exception_info": "action was not executed" })
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn text_actions_run_from_fenced_blocks() {
    let block = |command: &str| text(&format!("THOUGHT: go.\n\n```mswea_bash_command\n{command}\n```"));
    let run = Setup::text(vec![block("echo hello"), text("no block here"), block(SUBMIT)])
        .run()
        .await;
    assert_eq!(status(&run), ExitStatus::Submitted);
    assert_eq!(
        kinds(&run),
        [
            "system",
            "task",
            "answer",
            "observation",
            "answer",
            "format_error",
            "answer",
            "observation"
        ]
    );
    assert_eq!(entry_json(&run, 2)["actions"], json!([{ "command": "echo hello" }]));
    let observation = entry_json(&run, 3);
    assert_eq!(observation["message"]["role"], "user");
    assert_eq!(observation["output"]["output"], "hello\n");
    assert_eq!(entry_json(&run, 5)["error"], "Expected exactly 1 action, found 0.");
    // No tools are offered; the observation is the continuation.
    assert_eq!(request(&run, 1)[3], format!("user: {}", text_of(&run, 3)));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_rejection_reaches_the_model_and_the_run_goes_on() {
    let run = Setup::tool_calls(vec![calls(&["rm -rf build"]), calls(&[SUBMIT])])
        .interactive(vec![
            Input::Line("no thanks".to_owned()),
            Input::Line(String::new()),
            Input::Line(String::new()),
        ])
        .run()
        .await;
    assert_eq!(status(&run), ExitStatus::Submitted);
    assert_eq!(
        kinds(&run),
        [
            "system",
            "task",
            "answer",
            "observation",
            "interrupt",
            "answer",
            "observation"
        ]
    );
    assert_eq!(
        entry_json(&run, 3)["output"]["exception_info"],
        "action was not executed"
    );
    let rejection = "Commands not executed. The user rejected your commands with the following message: no thanks";
    assert_eq!(text_of(&run, 4), rejection);
    let Entry::Interrupt { reason, .. } = entry(&run, 4) else {
        panic!("an interrupt");
    };
    assert_eq!(*reason, InterruptReason::UserRejection);
    assert_eq!(request(&run, 1).last().unwrap(), &format!("user: {rejection}"));
}

#[tokio::test(flavor = "multi_thread")]
async fn whitelisted_and_yolo_actions_run_without_asking() {
    let frontend = Arc::new(Scripted::new(
        [Input::Line("/y".to_owned()), Input::Line(String::new())],
        true,
    ));
    let mut setup = Setup::tool_calls(vec![calls(&["ls"]), calls(&["echo two"]), calls(&[SUBMIT])])
        .agent(json!({ "whitelist": ["ls"] }));
    setup.frontend = Some(Arc::clone(&frontend));
    setup.kind = AgentKind::Interactive;
    let run = setup.run().await;
    assert_eq!(status(&run), ExitStatus::Submitted);
    let prompts: Vec<String> = frontend
        .log()
        .into_iter()
        .filter(|line| line.starts_with("prompt") || line.starts_with("notice"))
        .collect();
    assert_eq!(
        prompts,
        [
            "prompt Confirm { actions: 1 }",
            "notice Switched { mode: Yolo }",
            "prompt ConfirmExit",
        ]
    );
    assert!(
        frontend
            .log()
            .iter()
            .any(|line| line.starts_with("entry answer (step 1): Running."))
    );
    assert_eq!(run.output.trajectory.config["agent"]["mode"], "yolo");
}

#[tokio::test(flavor = "multi_thread")]
async fn human_mode_runs_the_users_commands_then_hands_back() {
    let run = Setup::tool_calls(vec![calls(&[SUBMIT])])
        .agent(json!({ "mode": "human" }))
        .interactive(vec![
            Input::Line("echo from-human".to_owned()),
            Input::Line("/y".to_owned()),
            Input::Line(String::new()),
        ])
        .run()
        .await;
    assert_eq!(status(&run), ExitStatus::Submitted);
    assert_eq!(
        kinds(&run),
        ["system", "task", "human", "observation", "answer", "observation"]
    );
    assert_eq!(text_of(&run, 2), "User command: \n```bash\necho from-human\n```");
    assert_eq!(entry_json(&run, 2)["command"], "echo from-human");
    let observation = entry_json(&run, 3);
    assert_eq!(observation["output"]["output"], "from-human\n");
    assert_eq!(observation["message"]["role"], "user");
    // The task, the command and its output reached the model before its first answer.
    let first = request(&run, 0);
    assert_eq!(first.len(), 4, "{first:?}");
    assert!(first[3].contains("from-human"));
    assert_eq!(run.output.trajectory.stats.calls, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_new_task_instead_of_finishing() {
    let run = Setup::tool_calls(vec![calls(&[SUBMIT]), calls(&[SUBMIT])])
        .agent(json!({ "mode": "yolo" }))
        .interactive(vec![
            Input::Line("also add tests".to_owned()),
            Input::Line(String::new()),
        ])
        .run()
        .await;
    assert_eq!(status(&run), ExitStatus::Submitted);
    assert_eq!(
        kinds(&run),
        [
            "system",
            "task",
            "answer",
            "observation",
            "interrupt",
            "answer",
            "observation"
        ]
    );
    assert_eq!(text_of(&run, 4), "The user added a new task: also add tests");
    assert_eq!(entry_json(&run, 4)["reason"], "user_new_task");
}

#[tokio::test(flavor = "multi_thread")]
async fn new_limits_let_the_run_go_on() {
    let run = Setup::tool_calls(vec![calls(&["echo one"]), calls(&[SUBMIT])])
        .agent(json!({ "mode": "yolo", "step_limit": 1, "confirm_exit": false }))
        .interactive(vec![Input::Line("5".to_owned()), Input::Line("10".to_owned())])
        .run()
        .await;
    assert_eq!(status(&run), ExitStatus::Submitted);
    assert_eq!(run.output.trajectory.config["agent"]["step_limit"], 5);
}

#[tokio::test(flavor = "multi_thread")]
async fn ctrl_c_stops_the_command_and_asks_for_a_comment() {
    let mut setup = Setup::tool_calls(vec![calls(&["sleep 30"]), calls(&[SUBMIT])])
        .agent(json!({ "mode": "yolo", "confirm_exit": false }))
        .interactive(vec![Input::Line("try something faster".to_owned())]);
    setup.interrupt_after = Some(Duration::from_millis(800));
    let run = setup.run().await;
    assert_eq!(status(&run), ExitStatus::Submitted);
    assert_eq!(
        kinds(&run),
        [
            "system",
            "task",
            "answer",
            "observation",
            "interrupt",
            "answer",
            "observation"
        ]
    );
    assert_eq!(
        entry_json(&run, 3)["output"]["exception_info"],
        "action was not executed"
    );
    assert_eq!(text_of(&run, 4), "Interrupted by user: try something faster");
    assert_eq!(entry_json(&run, 4)["reason"], "user_interruption");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unknown_tool_is_a_format_error_the_harness_answers() {
    let mut unknown = calls(&["ls"]);
    unknown["content"][1]["name"] = json!("python");
    let run = Setup::tool_calls(vec![unknown, calls(&[SUBMIT])]).run().await;
    assert_eq!(status(&run), ExitStatus::Submitted);
    // The answer has no actions; the harness answers the call itself, so no format error
    // entry holds what the model saw.
    assert_eq!(kinds(&run), ["system", "task", "answer", "answer", "observation"]);
    assert_eq!(entry_json(&run, 2)["actions"], json!([]));
    assert!(request(&run, 1)[3].contains("python"), "{:?}", request(&run, 1));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_malformed_bash_call_gets_the_format_error() {
    let mut malformed = calls(&["ls", "pwd"]);
    malformed["content"][2]["arguments"] = json!({ "cmd": "pwd" });
    let run = Setup::tool_calls(vec![malformed, calls(&[SUBMIT])])
        .agent(json!({ "cost_limit": 0 }))
        .run()
        .await;
    assert_eq!(status(&run), ExitStatus::Submitted);
    // The valid call gets the rendered format error; the harness refuses the other.
    assert_eq!(
        kinds(&run),
        ["system", "task", "answer", "format_error", "answer", "observation"]
    );
    let error = entry_json(&run, 3);
    assert_eq!(error["error"], "Missing 'command' argument in bash tool call.");
    assert_eq!(error["message"]["role"], "toolResult");
    assert_eq!(
        error["message"]["toolCallId"],
        entry_json(&run, 2)["message"]["content"][1]["id"]
    );
    assert!(text_of(&run, 3).contains("Missing 'command' argument"));
}
