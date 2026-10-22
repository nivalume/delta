//! The agent end to end: scripted model outputs, commands in a temporary directory, and
//! scripted answers for the interactive agent.

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
use futures::future::BoxFuture;
use pretty_assertions::assert_eq;
use serde_json::{Value as Json, json};

/// A tool-call output calling bash with each command.
fn calls(commands: &[&str]) -> Json {
    let tool_calls: Vec<Json> = commands
        .iter()
        .enumerate()
        .map(|(index, command)| {
            json!({
                "id": format!("call_{}_{index}", uuid::Uuid::new_v4().simple()),
                "type": "function",
                "function": { "name": "bash", "arguments": json!({ "command": command }).to_string() },
            })
        })
        .collect();
    json!({ "role": "assistant", "content": "Running.", "tool_calls": tool_calls })
}

/// A text-only output.
fn text(content: &str) -> Json {
    json!({ "role": "assistant", "content": content })
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
    outputs: Vec<Json>,
    agent: Json,
    frontend: Option<Arc<Scripted>>,
    kind: AgentKind,
    interrupt_after: Option<Duration>,
}

impl Setup {
    fn toolcall(outputs: Vec<Json>) -> Self {
        Self {
            specs: vec![
                "mini.yaml".to_owned(),
                "model.model_class=deterministic_toolcall".to_owned(),
            ],
            outputs,
            agent: json!({}),
            frontend: None,
            kind: AgentKind::Default,
            interrupt_after: None,
        }
    }

    fn textbased(outputs: Vec<Json>) -> Self {
        Self {
            specs: vec![
                "mini_textbased.yaml".to_owned(),
                "model.model_class=deterministic".to_owned(),
            ],
            ..Self::toolcall(outputs)
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
        config["model"]["outputs"] = Json::Array(self.outputs);
        config::merge(&mut config, json!({ "agent": self.agent }));
        config["agent"]["output_path"] = json!(dir.path().join("run.traj.json"));
        let model = ModelSettings::new(&config["model"], None).unwrap();
        let client = MiniModelClient::new(&model).unwrap();
        let model_ref = client.reference();
        let recording = Arc::new(Recording {
            inner: client,
            requests: Mutex::new(Vec::new()),
        });
        let env = LocalEnvironment::new(LocalEnvironmentConfig {
            cwd: dir.path().display().to_string(),
            ..LocalEnvironmentConfig::default()
        });
        let agent = Agent::new(AgentOptions {
            config: config["agent"].clone(),
            default_kind: self.kind,
            model,
            client: Arc::clone(&recording) as Arc<dyn ModelClient>,
            model_ref,
            env: Arc::new(env),
            frontend: self.frontend.clone().map(|frontend| frontend as Arc<dyn Frontend>),
            extra_info: Vec::new(),
            handle_ctrl_c: false,
            on_step: None,
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

fn roles(run: &Run) -> Vec<String> {
    run.output.trajectory["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|message| message["role"].as_str().unwrap().to_owned())
        .collect()
}

fn message(run: &Run, index: usize) -> &Json {
    &run.output.trajectory["messages"][index]
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
            Message::ToolResult(result) => format!(
                "tool: {}",
                result
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        delta_ai::types::UserBlock::Text(text) => Some(text.text.as_str()),
                        delta_ai::types::UserBlock::Image(_) => None,
                    })
                    .collect::<String>()
            ),
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn submits_and_saves_an_upstream_trajectory() {
    let run = Setup::toolcall(vec![calls(&["echo hello"]), calls(&[SUBMIT])])
        .run()
        .await;
    assert_eq!(run.output.exit.exit_status, "Submitted");
    assert_eq!(run.output.exit.submission, "the patch\n");
    assert_eq!(
        roles(&run),
        ["system", "user", "assistant", "tool", "assistant", "tool", "exit"]
    );
    let trajectory = &run.output.trajectory;
    assert_eq!(trajectory["trajectory_format"], "mini-swe-agent-1.1");
    assert_eq!(
        trajectory["info"]["model_stats"],
        json!({ "instance_cost": 2.0, "api_calls": 2 })
    );
    assert_eq!(trajectory["info"]["exit_status"], "Submitted");
    assert_eq!(trajectory["info"]["submission"], "the patch\n");
    assert_eq!(
        trajectory["info"]["config"]["agent_type"],
        "minisweagent.agents.default.DefaultAgent"
    );
    assert_eq!(
        trajectory["info"]["config"]["model_type"],
        "minisweagent.models.test_models.DeterministicToolcallModel"
    );
    assert_eq!(
        trajectory["info"]["config"]["environment_type"],
        "minisweagent.environments.local.LocalEnvironment"
    );
    let assistant = message(&run, 2);
    assert_eq!(assistant["extra"]["actions"][0]["command"], "echo hello");
    assert_eq!(assistant["extra"]["cost"], 1.0);
    let tool = message(&run, 3);
    assert_eq!(tool["tool_call_id"], assistant["tool_calls"][0]["id"]);
    assert_eq!(tool["extra"]["raw_output"], "hello\n");
    assert_eq!(tool["extra"]["returncode"], 0);
    assert!(tool["content"].as_str().unwrap().contains("hello"));
    assert_eq!(message(&run, 6)["extra"]["exit_status"], "Submitted");

    let saved = delta_mini::trajectory::load(&run.dir.path().join("run.traj.json")).unwrap();
    assert_eq!(&saved, trajectory);
    assert!(run.dir.path().join("run.harness.db").exists());

    // The model saw the rendered system prompt, the task, and the tool results.
    let system = message(&run, 0)["content"].as_str().unwrap();
    let first = request(&run, 0);
    assert_eq!(first.len(), 2);
    assert!(first[0].contains(system.trim()), "{first:?}");
    assert!(first[1].starts_with("user: ") && first[1].contains("Fix the bug."));
    let second = request(&run, 1);
    assert_eq!(second.len(), 4);
    assert!(
        second[3].starts_with("tool: ") && second[3].contains("hello"),
        "{second:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn repeated_format_errors_end_the_run() {
    let run = Setup::toolcall(vec![text("a"), text("b"), text("c"), text("d")])
        .run()
        .await;
    assert_eq!(run.output.exit.exit_status, "RepeatedFormatError");
    assert_eq!(roles(&run), ["system", "user", "user", "user", "user", "exit"]);
    let error = message(&run, 2);
    assert_eq!(error["extra"]["interrupt_type"], "FormatError");
    assert!(error["content"].as_str().unwrap().contains("No tool calls found"));
    assert_eq!(run.output.trajectory["info"]["model_stats"]["api_calls"], 3);
    // The model saw each error as a continuation.
    assert_eq!(run.requests.len(), 3);
    assert!(request(&run, 1)[3].contains("No tool calls found"));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_format_error_then_a_clean_step_resets_the_count() {
    let run = Setup::toolcall(vec![
        text("a"),
        text("b"),
        calls(&["true"]),
        text("c"),
        calls(&[SUBMIT]),
    ])
    .agent(json!({ "cost_limit": 0 }))
    .run()
    .await;
    assert_eq!(run.output.exit.exit_status, "Submitted");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_step_limit_ends_the_run_after_the_round() {
    let run = Setup::toolcall(vec![calls(&["echo one"]), calls(&["echo two"])])
        .agent(json!({ "step_limit": 1 }))
        .run()
        .await;
    assert_eq!(run.output.exit.exit_status, "LimitsExceeded");
    assert_eq!(roles(&run), ["system", "user", "assistant", "tool", "exit"]);
    assert_eq!(run.requests.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_cost_limit_ends_the_run() {
    let run = Setup::toolcall(vec![calls(&["echo one"]), calls(&["echo two"]), calls(&["echo three"])])
        .agent(json!({ "cost_limit": 2.0 }))
        .run()
        .await;
    assert_eq!(run.output.exit.exit_status, "LimitsExceeded");
    assert_eq!(run.requests.len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_wall_time_limit_ends_the_run() {
    let run = Setup::toolcall(vec![calls(&["sleep 1.2"]), calls(&["echo late"])])
        .agent(json!({ "wall_time_limit_seconds": 1 }))
        .run()
        .await;
    assert_eq!(run.output.exit.exit_status, "TimeExceeded");
    assert_eq!(run.requests.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_submission_mid_round_skips_the_rest() {
    let run = Setup::toolcall(vec![calls(&["echo a", SUBMIT, "echo c"])]).run().await;
    assert_eq!(run.output.exit.exit_status, "Submitted");
    assert_eq!(
        roles(&run),
        ["system", "user", "assistant", "tool", "tool", "tool", "exit"]
    );
    assert_eq!(message(&run, 5)["extra"]["exception_info"], "action was not executed");
    assert_eq!(message(&run, 5)["extra"]["returncode"], -1);
}

#[tokio::test(flavor = "multi_thread")]
async fn text_based_actions_run_from_fenced_blocks() {
    let block = |command: &str| text(&format!("THOUGHT: go.\n\n```mswea_bash_command\n{command}\n```"));
    let run = Setup::textbased(vec![block("echo hello"), text("no block here"), block(SUBMIT)])
        .run()
        .await;
    assert_eq!(run.output.exit.exit_status, "Submitted");
    assert_eq!(
        roles(&run),
        [
            "system",
            "user",
            "assistant",
            "user",
            "user",
            "assistant",
            "user",
            "exit"
        ]
    );
    assert_eq!(
        message(&run, 2)["extra"]["actions"],
        json!([{ "command": "echo hello" }])
    );
    assert_eq!(message(&run, 3)["extra"]["raw_output"], "hello\n");
    assert_eq!(message(&run, 4)["extra"]["interrupt_type"], "FormatError");
    assert_eq!(message(&run, 4)["extra"]["n_actions"], 0);
    // No tools are offered; the observation is the continuation.
    assert!(request(&run, 1)[3].starts_with("user: ") && request(&run, 1)[3].contains("hello"));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_rejection_reaches_the_model_and_the_run_goes_on() {
    let run = Setup::toolcall(vec![calls(&["rm -rf build"]), calls(&[SUBMIT])])
        .interactive(vec![
            Input::Line("no thanks".to_owned()),
            Input::Line(String::new()),
            Input::Line(String::new()),
        ])
        .run()
        .await;
    assert_eq!(run.output.exit.exit_status, "Submitted");
    assert_eq!(
        roles(&run),
        [
            "system",
            "user",
            "assistant",
            "tool",
            "user",
            "assistant",
            "tool",
            "exit"
        ]
    );
    assert_eq!(message(&run, 3)["extra"]["exception_info"], "action was not executed");
    let rejection = message(&run, 4);
    assert_eq!(
        rejection["content"],
        "Commands not executed. The user rejected your commands with the following message: no thanks"
    );
    assert_eq!(rejection["extra"]["interrupt_type"], "UserRejection");
    let second = request(&run, 1);
    assert_eq!(
        second.last().unwrap(),
        "user: Commands not executed. The user rejected your commands with the following message: no thanks"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn whitelisted_and_yolo_actions_run_without_asking() {
    let frontend = Arc::new(Scripted::new(
        [Input::Line("/y".to_owned()), Input::Line(String::new())],
        true,
    ));
    let mut setup = Setup::toolcall(vec![calls(&["ls"]), calls(&["echo two"]), calls(&[SUBMIT])])
        .agent(json!({ "whitelist_actions": ["ls"] }));
    setup.frontend = Some(Arc::clone(&frontend));
    setup.kind = AgentKind::Interactive;
    let run = setup.run().await;
    assert_eq!(run.output.exit.exit_status, "Submitted");
    let prompts: Vec<String> = frontend
        .log()
        .into_iter()
        .filter(|entry| entry.starts_with("prompt") || entry.starts_with("notice"))
        .collect();
    assert_eq!(
        prompts,
        [
            "prompt Confirm { actions: 1 }",
            "notice Switched { mode: Yolo }",
            "prompt ConfirmExit",
        ]
    );
    assert_eq!(run.output.trajectory["info"]["config"]["agent"]["mode"], "yolo");
}

#[tokio::test(flavor = "multi_thread")]
async fn human_mode_runs_the_users_commands_then_hands_back() {
    let run = Setup::toolcall(vec![calls(&[SUBMIT])])
        .agent(json!({ "mode": "human" }))
        .interactive(vec![
            Input::Line("echo from-human".to_owned()),
            Input::Line("/y".to_owned()),
            Input::Line(String::new()),
        ])
        .run()
        .await;
    assert_eq!(run.output.exit.exit_status, "Submitted");
    assert_eq!(
        roles(&run),
        ["system", "user", "user", "user", "assistant", "tool", "exit"]
    );
    assert_eq!(
        message(&run, 2)["content"],
        "User command: \n```bash\necho from-human\n```"
    );
    assert_eq!(message(&run, 3)["extra"]["raw_output"], "from-human\n");
    assert!(message(&run, 3).get("tool_call_id").is_none());
    // The task, the command and its output reached the model before its first answer.
    let first = request(&run, 0);
    assert_eq!(first.len(), 4, "{first:?}");
    assert!(first[3].contains("from-human"));
    assert_eq!(run.output.trajectory["info"]["model_stats"]["api_calls"], 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_new_task_instead_of_finishing() {
    let run = Setup::toolcall(vec![calls(&[SUBMIT]), calls(&[SUBMIT])])
        .agent(json!({ "mode": "yolo" }))
        .interactive(vec![
            Input::Line("also add tests".to_owned()),
            Input::Line(String::new()),
        ])
        .run()
        .await;
    assert_eq!(run.output.exit.exit_status, "Submitted");
    assert_eq!(
        roles(&run),
        [
            "system",
            "user",
            "assistant",
            "tool",
            "user",
            "assistant",
            "tool",
            "exit"
        ]
    );
    assert_eq!(message(&run, 4)["content"], "The user added a new task: also add tests");
    assert_eq!(message(&run, 4)["extra"]["interrupt_type"], "UserNewTask");
}

#[tokio::test(flavor = "multi_thread")]
async fn new_limits_let_the_run_go_on() {
    let run = Setup::toolcall(vec![calls(&["echo one"]), calls(&[SUBMIT])])
        .agent(json!({ "mode": "yolo", "step_limit": 1, "confirm_exit": false }))
        .interactive(vec![Input::Line("5".to_owned()), Input::Line("10".to_owned())])
        .run()
        .await;
    assert_eq!(run.output.exit.exit_status, "Submitted");
    assert_eq!(run.output.trajectory["info"]["config"]["agent"]["step_limit"], 5);
}

#[tokio::test(flavor = "multi_thread")]
async fn ctrl_c_stops_the_command_and_asks_for_a_comment() {
    let mut setup = Setup::toolcall(vec![calls(&["sleep 30"]), calls(&[SUBMIT])])
        .agent(json!({ "mode": "yolo", "confirm_exit": false }))
        .interactive(vec![Input::Line("try something faster".to_owned())]);
    setup.interrupt_after = Some(Duration::from_millis(800));
    let run = setup.run().await;
    assert_eq!(run.output.exit.exit_status, "Submitted");
    assert_eq!(
        roles(&run),
        [
            "system",
            "user",
            "assistant",
            "tool",
            "user",
            "assistant",
            "tool",
            "exit"
        ]
    );
    assert_eq!(message(&run, 3)["extra"]["exception_info"], "action was not executed");
    assert_eq!(message(&run, 4)["content"], "Interrupted by user: try something faster");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unknown_tool_is_a_format_error() {
    let mut unknown = calls(&["ls"]);
    unknown["tool_calls"][0]["function"]["name"] = json!("python");
    let run = Setup::toolcall(vec![unknown, calls(&[SUBMIT])]).run().await;
    assert_eq!(run.output.exit.exit_status, "Submitted");
    assert_eq!(roles(&run), ["system", "user", "user", "assistant", "tool", "exit"]);
    let error = message(&run, 2);
    assert_eq!(error["extra"]["interrupt_type"], "FormatError");
    assert!(
        error["content"]
            .as_str()
            .unwrap()
            .contains("<error>\nUnknown tool 'python'.\n</error>")
    );
    // The model saw the harness's own refusal of the call.
    assert!(request(&run, 1)[3].contains("python"), "{:?}", request(&run, 1));
}
