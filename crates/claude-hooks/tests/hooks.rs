//! Claude Code hooks over a harness with the coding tools and a scripted model: command
//! handlers through the local environment, HTTP handlers against a local mock server.

#![allow(clippy::unwrap_used, clippy::too_many_lines)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use delta_ai::Models;
use delta_ai::providers::faux::{FauxOptions, FauxProvider, faux_assistant_message, faux_text, faux_tool_call};
use delta_ai::types::{Message, UserBlock};
use delta_chord::context::Context;
use delta_chord::json::Json;
use delta_claude_hooks::{ClaudeHooks, ClaudeHooksOptions, HooksConfig};
use delta_harness::agent::{
    AgentChange, Change, CompactionSettings, HarnessSettings, LocalEnvBuilder, PartialCompactionPolicy, SettingsSource,
};
use delta_harness::approval::{AllowAll, ApprovalCall, ApprovalPolicy, AskAll, PolicyDecision, PolicyError};
use delta_harness::harness::{
    Conversation, ConversationCreateOptions, EntryRange, Harness, HarnessOptions, RootOptions,
};
use delta_harness::inbox::SubmissionDraft;
use delta_harness::model::{ModelRef, ModelsClient};
use delta_harness::records::{ConversationOwnership, EntryRecord, ScanOrder, SubmissionRecord, SubmissionState};
use delta_harness::registry::{Extension, HookApi, Registry};
use delta_harness::scheduler::{Next, TaskDefinition, TaskError, TaskRuntime};
use delta_harness::session::TaskOptions;
use delta_harness::storage::{MemoryStorage, SqliteOptions, SqliteStorage, Storage};
use delta_harness::tools::{BashOptions, coding_tools};
use futures::future::BoxFuture;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

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
            "delta-claude-hooks-{}-{nanos}-{}",
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

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Non-blocking handler failures, by event.
type Errors = Arc<Mutex<Vec<(String, String)>>>;

/// A harness whose root conversation works in a temp directory, with the coding tools and
/// the configured hooks.
struct Setup {
    harness: Harness,
    conversation: Conversation,
    faux: FauxProvider,
    errors: Errors,
    dir: TempDir,
}

/// Makes the approval policy from the hooks.
type MakePolicy = Box<dyn Fn(&Arc<ClaudeHooks>) -> Arc<dyn ApprovalPolicy>>;

/// How a setup is made.
struct Build {
    config: serde_json::Value,
    policy: MakePolicy,
    settings: HarnessSettings,
    extensions: Vec<Extension>,
}

impl Build {
    fn new(config: serde_json::Value) -> Self {
        Self {
            config,
            policy: Box::new(|_| Arc::new(AllowAll)),
            settings: HarnessSettings::default(),
            extensions: Vec::new(),
        }
    }
}

fn hooks(build: &Build, errors: &Errors) -> Arc<ClaudeHooks> {
    let errors = errors.clone();
    let options = ClaudeHooksOptions {
        client: Some(reqwest::Client::builder().no_proxy().build().unwrap()),
        transcript_path: None,
        on_error: Some(Arc::new(move |event, error| {
            errors.lock().unwrap().push((event.to_owned(), error.to_owned()));
        })),
    };
    ClaudeHooks::new(HooksConfig::from_value(build.config.clone()).unwrap(), options)
}

async fn open(build: &Build, storage: Arc<dyn Storage>, faux: &FauxProvider, dir: &Path, errors: &Errors) -> Harness {
    let hooks = hooks(build, errors);
    let models = Models::default();
    models.set_provider(faux.clone());
    let registry = Registry::new();
    registry.install(coding_tools(BashOptions::default())).unwrap();
    for extension in &build.extensions {
        registry.install(extension.clone()).unwrap();
    }
    registry.install(hooks.extension()).unwrap();
    let mut options = HarnessOptions::new(Arc::new(ModelsClient::new(models)));
    options.registry = registry;
    options.env = Some(Arc::new(LocalEnvBuilder::new(dir)));
    options.approval = (build.policy)(&hooks);
    options.settings = SettingsSource::Fixed(build.settings.clone());
    Harness::open(storage, options, &ctx()).await.unwrap()
}

fn agent() -> AgentChange {
    AgentChange {
        model: Change::Set(ModelRef::new("faux", "faux-1")),
        ..AgentChange::default()
    }
}

async fn root(harness: &Harness) -> Conversation {
    let options = RootOptions {
        agent: Some(agent()),
        init: None,
    };
    harness.root(options, &ctx()).await.unwrap()
}

async fn setup(build: Build) -> Setup {
    let dir = TempDir::new();
    let faux = FauxProvider::new(FauxOptions::default());
    let errors = Errors::default();
    let harness = open(&build, Arc::new(MemoryStorage::new()), &faux, dir.path(), &errors).await;
    let conversation = root(&harness).await;
    Setup {
        harness,
        conversation,
        faux,
        errors,
        dir,
    }
}

fn bash(command: &str) -> delta_ai::providers::faux::FauxResponse {
    faux_assistant_message(vec![faux_tool_call("bash", serde_json::json!({ "command": command }))]).into()
}

fn text(text: &str) -> delta_ai::providers::faux::FauxResponse {
    faux_assistant_message(vec![faux_text(text)]).into()
}

/// A command hook group for `event`.
fn command(event: &str, matcher: &str, command: &str) -> serde_json::Value {
    serde_json::json!({ event: [{ "matcher": matcher, "hooks": [{ "type": "command", "command": command }] }] })
}

async fn submit(conversation: &Conversation, input: &str) -> SubmissionRecord {
    let submission = conversation
        .submit(SubmissionDraft::input(input), &ctx())
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(20), submission.wait(&ctx()))
        .await
        .unwrap()
        .unwrap()
}

async fn entries(conversation: &Conversation) -> Vec<EntryRecord> {
    conversation
        .entries(
            EntryRange {
                order: Some(ScanOrder::Ascending),
                ..EntryRange::default()
            },
            1000,
            None,
            &ctx(),
        )
        .await
        .unwrap()
        .items
}

/// The text items of each tool result, in order.
async fn tool_texts(conversation: &Conversation) -> Vec<Vec<String>> {
    entries(conversation)
        .await
        .into_iter()
        .filter_map(|entry| match entry.model.as_deref() {
            Some([Message::ToolResult(result)]) => Some(
                result
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        UserBlock::Text(text) => Some(text.text.clone()),
                        UserBlock::Image(_) => None,
                    })
                    .collect(),
            ),
            _ => None,
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pre_tool_use_exit_2_blocks_with_stderr_and_gets_the_input() {
    let script = r#"cat > "$CLAUDE_PROJECT_DIR/input.json"; echo "no commands today" >&2; exit 2"#;
    let setup = setup(Build::new(command("PreToolUse", "Bash", script))).await;
    setup.faux.set_responses(vec![bash("echo hi"), text("ok")]);
    submit(&setup.conversation, "go").await;
    assert_eq!(
        tool_texts(&setup.conversation).await,
        [["<harness>\n[error] Tool call blocked: no commands today\n</harness>"]]
    );
    let input: serde_json::Value =
        serde_json::from_slice(&std::fs::read(setup.dir.file("input.json")).unwrap()).unwrap();
    assert_eq!(input["hook_event_name"], "PreToolUse");
    assert_eq!(input["tool_name"], "Bash");
    assert_eq!(input["tool_input"]["command"], "echo hi");
    assert_eq!(input["session_id"], "1");
    assert_eq!(input["permission_mode"], "default");
    assert_eq!(input["cwd"], setup.dir.path().to_str().unwrap());
    setup.harness.close(&ctx()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pre_tool_use_replaces_the_input_and_adds_context() {
    let output = serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "updatedInput": { "command": "echo changed" },
            "additionalContext": "the hook changed the command"
        }
    });
    let script = format!("cat > /dev/null; printf '%s' '{output}'");
    // A matcher that does not match runs nothing.
    let mut config = command("PreToolUse", "Bash", &script);
    config["PreToolUse"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({ "matcher": "Read", "hooks": [{ "type": "command", "command": "exit 2" }] }));
    let setup = setup(Build::new(config)).await;
    setup.faux.set_responses(vec![bash("echo original"), text("ok")]);
    submit(&setup.conversation, "go").await;
    assert_eq!(
        tool_texts(&setup.conversation).await,
        [["changed\n", "the hook changed the command"]]
    );
    setup.harness.close(&ctx()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn http_handlers_deny_and_fail_without_blocking() {
    let server = MockServer::start().await;
    let deny = serde_json::json!({
        "hookSpecificOutput": { "hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "remote says no" }
    });
    Mock::given(method("POST"))
        .and(path("/deny"))
        .respond_with(ResponseTemplate::new(200).set_body_json(deny))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/broken"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    let config = serde_json::json!({
        "PreToolUse": [
            { "matcher": "Bash", "hooks": [{
                "type": "http", "url": format!("{}/deny", server.uri()),
                "headers": { "X-Home": "${HOME}" }, "allowedEnvVars": ["HOME"]
            }] }
        ],
        "PostToolUse": [{ "hooks": [{ "type": "http", "url": format!("{}/broken", server.uri()) }] }]
    });
    let setup = setup(Build::new(config)).await;
    setup.faux.set_responses(vec![bash("echo hi"), text("ok")]);
    submit(&setup.conversation, "go").await;
    assert_eq!(
        tool_texts(&setup.conversation).await,
        [["<harness>\n[error] Tool call blocked: remote says no\n</harness>"]]
    );
    let requests = server.received_requests().await.unwrap();
    let request = &requests[0];
    let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(body["tool_input"]["command"], "echo hi");
    assert_eq!(
        request.headers.get("x-home").unwrap().to_str().unwrap(),
        std::env::var("HOME").unwrap_or_default()
    );

    // A failing handler does not block: the next call runs and the failure is reported.
    let read = faux_assistant_message(vec![faux_tool_call(
        "read",
        serde_json::json!({ "path": "missing.txt" }),
    )]);
    setup.faux.set_responses(vec![read.into(), text("ok")]);
    std::fs::write(setup.dir.file("missing.txt"), "here\n").unwrap();
    submit(&setup.conversation, "again").await;
    assert_eq!(tool_texts(&setup.conversation).await[1], ["here\n"]);
    let errors = setup.errors.lock().unwrap().clone();
    assert!(
        errors
            .iter()
            .any(|(event, error)| event == "PostToolUse" && error.contains("500")),
        "{errors:?}"
    );
    setup.harness.close(&ctx()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn post_tool_use_replaces_the_output() {
    let output = serde_json::json!({ "hookSpecificOutput": { "hookEventName": "PostToolUse", "updatedToolOutput": "[redacted]" } });
    let script = format!("cat > \"$CLAUDE_PROJECT_DIR/post.json\"; printf '%s' '{output}'");
    let setup = setup(Build::new(command("PostToolUse", "Bash", &script))).await;
    setup.faux.set_responses(vec![bash("echo secret"), text("ok")]);
    submit(&setup.conversation, "go").await;
    assert_eq!(tool_texts(&setup.conversation).await, [["[redacted]"]]);
    let input: serde_json::Value =
        serde_json::from_slice(&std::fs::read(setup.dir.file("post.json")).unwrap()).unwrap();
    assert_eq!(input["tool_response"]["output"], "secret\n");
    assert_eq!(input["tool_response"]["structured"]["exitCode"], 0);
    setup.harness.close(&ctx()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stop_continues_the_run_once() {
    let script = r#"input=$(cat); if printf '%s' "$input" | grep -q '"stop_hook_active":true'; then exit 0; fi; printf '{"decision":"block","reason":"run the tests too"}'"#;
    let setup = setup(Build::new(command("Stop", "", script))).await;
    setup.faux.set_responses(vec![text("first"), text("second")]);
    let record = submit(&setup.conversation, "go").await;
    assert!(matches!(record.state, SubmissionState::Done { .. }));
    let users: Vec<String> = entries(&setup.conversation)
        .await
        .into_iter()
        .filter_map(|entry| match entry.model.as_deref() {
            Some([Message::User(user)]) => Some(user.content.text()),
            _ => None,
        })
        .collect();
    assert_eq!(users, ["go", "run the tests too"]);
    assert_eq!(setup.faux.pending_response_count(), 0);
    setup.harness.close(&ctx()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn user_prompt_submit_blocks_or_adds_context() {
    let script = r#"input=$(cat); if printf '%s' "$input" | grep -q secret; then echo "no secrets" >&2; exit 2; fi; echo "extra context""#;
    let setup = setup(Build::new(command("UserPromptSubmit", "", script))).await;
    let blocked = submit(&setup.conversation, "tell me the secret").await;
    let SubmissionState::Unanswered { reason, .. } = blocked.state else {
        panic!("unanswered: {:?}", blocked.state)
    };
    assert_eq!(reason, "blocked");

    setup.faux.set_responses(vec![text("hello")]);
    submit(&setup.conversation, "hi").await;
    let contexts: Vec<String> = entries(&setup.conversation)
        .await
        .into_iter()
        .filter(|entry| entry.kind == "delta.context")
        .filter_map(|entry| match entry.model.as_deref() {
            Some([Message::System(system)]) => Some(system.content.text()),
            _ => None,
        })
        .collect();
    assert_eq!(contexts, ["extra context"]);
    setup.harness.close(&ctx()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn permission_request_answers_what_the_policy_asks() {
    let allow = serde_json::json!({ "hookSpecificOutput": { "hookEventName": "PermissionRequest", "decision": { "behavior": "allow" } } });
    let script = format!("cat > /dev/null; printf '%s' '{allow}'");
    let mut build = Build::new(command("PermissionRequest", "Bash", &script));
    build.policy = Box::new(|hooks| hooks.permission_policy(Arc::new(AskAll)));
    let setup = setup(build).await;
    setup.faux.set_responses(vec![bash("echo allowed"), text("ok")]);
    submit(&setup.conversation, "go").await;
    assert_eq!(tool_texts(&setup.conversation).await, [["allowed\n"]]);
    assert!(setup.conversation.pending_approvals(&ctx()).await.unwrap().is_empty());
    setup.harness.close(&ctx()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pre_compact_declines_a_manual_compaction() {
    let mut build = Build::new(command(
        "PreCompact",
        "manual",
        r#"cat > "$CLAUDE_PROJECT_DIR/compact.json"; exit 2"#,
    ));
    build.settings.compaction = Some(CompactionSettings {
        policy: PartialCompactionPolicy {
            keep_recent_tokens: Some(10),
            ..PartialCompactionPolicy::default()
        },
        ..CompactionSettings::default()
    });
    let setup = setup(build).await;
    setup
        .faux
        .set_responses(vec![text(&"long answer ".repeat(500)), text("second")]);
    submit(&setup.conversation, "first").await;
    submit(&setup.conversation, "second").await;
    let task = setup
        .conversation
        .compact(Some("keep the decisions".to_owned()), &ctx())
        .await
        .unwrap();
    setup.harness.wait_for_task(task, &ctx()).await.unwrap();
    let input: serde_json::Value =
        serde_json::from_slice(&std::fs::read(setup.dir.file("compact.json")).unwrap()).unwrap();
    assert_eq!(input["trigger"], "manual");
    assert_eq!(input["custom_instructions"], "keep the decisions");
    assert!(
        entries(&setup.conversation)
            .await
            .iter()
            .all(|entry| entry.kind != "delta.compaction")
    );
    setup.harness.close(&ctx()).await.unwrap();
}

/// A task that never finishes on its own: the owner of a sub-agent conversation.
#[derive(Clone, Copy, Debug, Default)]
struct Forever;

impl TaskDefinition for Forever {
    type Input = Json;
    type Checkpoint = Json;
    type Result = Json;

    fn name(&self) -> &'static str {
        "test.forever"
    }

    fn version(&self) -> u32 {
        1
    }

    fn initial(&self, _input: &Json) -> Json {
        Json::Null
    }

    async fn run<'a>(&'a self, _: Json, _: &'a TaskRuntime<Self>, ctx: &'a Context) -> Result<Next<Self>, TaskError> {
        ctx.cancelled().await;
        Ok(Next::fail("cancelled"))
    }

    async fn abort<'a>(&'a self, _: Json, _: &'a TaskRuntime<Self>, _: &'a Context) -> Result<Next<Self>, TaskError> {
        Ok(Next::abort(None))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_task_owned_conversation_runs_subagent_stop() {
    let mut config = command(
        "SubagentStop",
        "",
        r#"cat > /dev/null; touch "$CLAUDE_PROJECT_DIR/subagent""#,
    );
    config["Stop"] = command("Stop", "", r#"cat > /dev/null; touch "$CLAUDE_PROJECT_DIR/stop""#)["Stop"].clone();
    let mut build = Build::new(config);
    build
        .extensions
        .push(Extension::new("forever").with_task(Arc::new(Forever)));
    let setup = setup(build).await;
    let task = setup
        .conversation
        .commit(&ctx(), |tx| {
            Box::pin(async move { Ok(tx.create_task(&Forever, Json::Null, TaskOptions::default()).await?.id()) })
        })
        .await
        .unwrap();
    let options = ConversationCreateOptions {
        ownership: ConversationOwnership::Task(task),
        agent: Some(agent()),
        init: None,
    };
    let sub = setup.harness.create_conversation(options, &ctx()).await.unwrap();
    setup.faux.set_responses(vec![text("sub answer")]);
    submit(&sub, "work").await;
    assert!(setup.dir.file("subagent").exists());
    assert!(!setup.dir.file("stop").exists());
    setup.harness.close(&ctx()).await.unwrap();
}

/// Hangs on its first decision, so the harness can stop between the hooks and the intent.
struct HangOnce {
    calls: AtomicU32,
}

impl ApprovalPolicy for HangOnce {
    fn decide<'a>(
        &'a self,
        _call: &'a ApprovalCall<'a>,
        _api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<PolicyDecision, PolicyError>> {
        Box::pin(async move {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                ctx.cancelled().await;
            }
            Ok(PolicyDecision::Allow)
        })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rerun_does_not_run_handlers_again() {
    let dir = TempDir::new();
    let database = dir.file("harness.db");
    let faux = FauxProvider::new(FauxOptions::default());
    let errors = Errors::default();
    let policy = Arc::new(HangOnce {
        calls: AtomicU32::new(0),
    });
    let mut build = Build::new(command(
        "PreToolUse",
        "Bash",
        r#"cat > /dev/null; echo run >> "$CLAUDE_PROJECT_DIR/count""#,
    ));
    let shared = policy.clone();
    build.policy = Box::new(move |_| shared.clone());
    faux.set_responses(vec![bash("echo once"), text("ok")]);

    let storage = SqliteStorage::open(&database, SqliteOptions::default()).await.unwrap();
    let harness = open(&build, Arc::new(storage), &faux, dir.path(), &errors).await;
    let conversation = root(&harness).await;
    conversation.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    while policy.calls.load(Ordering::SeqCst) == 0 {
        assert!(tokio::time::Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let closed = harness.session().closed();
    drop((conversation, harness));
    tokio::time::timeout(Duration::from_secs(20), closed).await.unwrap();

    let storage = SqliteStorage::open(&database, SqliteOptions::default()).await.unwrap();
    let harness = open(&build, Arc::new(storage), &faux, dir.path(), &errors).await;
    harness.resume().unwrap();
    let conversation = root(&harness).await;
    tokio::time::timeout(Duration::from_secs(20), conversation.wait_for_idle(&ctx()))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(policy.calls.load(Ordering::SeqCst), 2);
    assert_eq!(tool_texts(&conversation).await, [["once\n"]]);
    assert_eq!(std::fs::read_to_string(dir.file("count")).unwrap(), "run\n");
    harness.close(&ctx()).await.unwrap();
}
