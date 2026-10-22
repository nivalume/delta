//! `RulesPolicy` over the built-in tools: rules, modes, rules added by approvals, how
//! permissions follow forks and owned conversations, and the memoized decision.

use std::sync::{Arc, Mutex};

use delta_chord::context::Context;
use delta_chord::json::Json;
use delta_harness::agent::LocalEnvBuilder;
use delta_harness::approval::{
    ApprovalCall, ApprovalError, ApprovalPolicy, ApprovalRequest, PermissionDefaults, PermissionMode, PermissionRules,
    PermissionsChange, PolicyDecision, PolicyError, ReviewDecision, RulesPolicy,
};
use delta_harness::harness::{ConversationCreateOptions, Harness, HarnessError, Submission};
use delta_harness::inbox::SubmissionDraft;
use delta_harness::observe::AgentEvent;
use delta_harness::records::{ConversationOwnership, SubmissionState};
use delta_harness::registry::{Extension, HookApi};
use delta_harness::scheduler::{Next, TaskDefinition, TaskError, TaskRuntime};
use delta_harness::session::TaskOptions;
use delta_harness::tools::{BashOptions, coding_tools};
use futures::future::BoxFuture;

use crate::support::TempDir;
use crate::support::fixture::{Fixture, PATIENCE, chat, ctx, entries, eventually, settled};
use crate::support::model::Reply;
use crate::{Backend, Called, Workspace};

cases!(
    deny_rules_beat_allow_rules,
    ask_rules_ask_without_a_proposal,
    the_default_mode_reads_inside_and_asks_for_the_rest,
    accept_edits_writes_inside,
    plan_denies_commands_and_writes,
    dont_ask_denies_what_would_ask,
    bypass_keeps_deny_and_ask_rules,
    an_approved_rule_applies_to_every_conversation,
    forks_and_owned_conversations_take_the_permissions,
    rules_are_validated,
    a_rerun_reuses_the_decision,
);

fn strings(rules: &[&str]) -> Vec<String> {
    rules.iter().map(|rule| (*rule).to_owned()).collect()
}

fn policy(mode: PermissionMode, allow: &[&str], ask: &[&str], deny: &[&str]) -> Arc<dyn ApprovalPolicy> {
    let defaults = PermissionDefaults {
        mode,
        rules: PermissionRules {
            allow: strings(allow),
            ask: strings(ask),
            deny: strings(deny),
        },
        ..PermissionDefaults::default()
    };
    Arc::new(RulesPolicy::new(defaults).unwrap())
}

async fn open(backend: Backend, policy: Arc<dyn ApprovalPolicy>) -> Workspace {
    Workspace::open(backend, BashOptions::default(), Some(policy)).await
}

/// The reason a call was blocked.
fn blocked(called: &Called) -> String {
    assert!(called.message.is_error, "{called:?}");
    called.diagnostic("blocked").message.clone()
}

/// Submits one call of `tool` and waits until its request is pending.
async fn asking(
    workspace: &Workspace,
    id: &str,
    tool: &str,
    arguments: serde_json::Value,
) -> (Submission, ApprovalRequest) {
    workspace
        .fixture
        .model
        .push([Reply::calls_with(&[(id, tool, arguments)]), Reply::text("Done")]);
    let conversation = &workspace.conversation;
    let submission = conversation.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    eventually(|| async { conversation.pending_approvals(&ctx()).await.unwrap().len() == 1 }).await;
    let request = conversation.pending_approvals(&ctx()).await.unwrap().remove(0);
    (submission, request)
}

async fn deny_rules_beat_allow_rules(backend: Backend) {
    let workspace = open(
        backend,
        policy(PermissionMode::Default, &["Bash"], &[], &["Bash(rm *)"]),
    )
    .await;
    let called = workspace
        .call("bash", serde_json::json!({ "command": "echo ok" }))
        .await;
    assert_eq!(called.text(), "ok\n");
    let called = workspace
        .call("bash", serde_json::json!({ "command": "echo ok && rm -rf build" }))
        .await;
    assert_eq!(
        blocked(&called),
        "Tool call blocked: approval denied: denied by rule Bash(rm *)"
    );
    workspace.close().await;
}

async fn ask_rules_ask_without_a_proposal(backend: Backend) {
    let workspace = open(
        backend,
        policy(PermissionMode::Default, &["Bash"], &["Bash(git push:*)"], &[]),
    )
    .await;
    let (submission, request) = asking(
        &workspace,
        "c1",
        "bash",
        serde_json::json!({ "command": "git push origin" }),
    )
    .await;
    assert_eq!(request.prompt.as_deref(), Some("Bash(git push:*) requires approval"));
    assert_eq!(request.proposed_rule, None);
    let denied = ReviewDecision::Denied { rejection: None };
    workspace.conversation.decide(request.id, denied, &ctx()).await.unwrap();
    settled(&submission).await;
    workspace.close().await;
}

async fn the_default_mode_reads_inside_and_asks_for_the_rest(backend: Backend) {
    let workspace = open(backend, policy(PermissionMode::Default, &[], &[], &[])).await;
    workspace.write("inside.txt", "in\n");
    let called = workspace
        .call("read", serde_json::json!({ "path": "inside.txt" }))
        .await;
    assert_eq!(called.text(), "in\n");

    let outside = TempDir::new();
    std::fs::write(outside.path().join("far.txt"), "far\n").unwrap();
    let far = outside.path().join("far.txt");
    let (submission, request) = asking(
        &workspace,
        "c1",
        "read",
        serde_json::json!({ "path": far.to_str().unwrap() }),
    )
    .await;
    let directory = outside.path().to_str().unwrap().trim_start_matches('/');
    assert_eq!(request.proposed_rule, Some(format!("Read(//{directory}/**)")));
    workspace
        .conversation
        .decide(request.id, ReviewDecision::Approved, &ctx())
        .await
        .unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));

    let (submission, request) = asking(
        &workspace,
        "c2",
        "write",
        serde_json::json!({ "path": "new.txt", "content": "x" }),
    )
    .await;
    let cwd = workspace.dir.path().to_str().unwrap().trim_start_matches('/');
    assert_eq!(request.proposed_rule, Some(format!("Edit(//{cwd}/**)")));
    let (_, bash) = (submission, request);
    workspace
        .conversation
        .decide(bash.id, ReviewDecision::Denied { rejection: None }, &ctx())
        .await
        .unwrap();
    assert!(!workspace.path("new.txt").exists());
    workspace.close().await;
}

async fn accept_edits_writes_inside(backend: Backend) {
    let workspace = open(backend, policy(PermissionMode::Default, &[], &[], &[])).await;
    workspace
        .conversation
        .set_permission_mode(PermissionMode::AcceptEdits, &ctx())
        .await
        .unwrap();
    let called = workspace
        .call("write", serde_json::json!({ "path": "a/b.txt", "content": "edited" }))
        .await;
    assert!(!called.message.is_error, "{:?}", called.diagnostics);
    assert_eq!(workspace.read("a/b.txt"), "edited");
    workspace.close().await;
}

async fn plan_denies_commands_and_writes(backend: Backend) {
    let workspace = open(backend, policy(PermissionMode::Plan, &["Bash"], &[], &[])).await;
    workspace.write("notes.txt", "plan\n");
    let read = workspace.call("read", serde_json::json!({ "path": "notes.txt" })).await;
    assert_eq!(read.text(), "plan\n");
    let bash = workspace
        .call("bash", serde_json::json!({ "command": "echo hi" }))
        .await;
    assert_eq!(
        blocked(&bash),
        "Tool call blocked: approval denied: plan mode: bash may not run"
    );
    let write = workspace
        .call("write", serde_json::json!({ "path": "notes.txt", "content": "x" }))
        .await;
    assert_eq!(
        blocked(&write),
        "Tool call blocked: approval denied: plan mode: write may not run"
    );
    assert_eq!(workspace.read("notes.txt"), "plan\n");
    workspace.close().await;
}

async fn dont_ask_denies_what_would_ask(backend: Backend) {
    let workspace = open(backend, policy(PermissionMode::DontAsk, &["Bash(echo *)"], &[], &[])).await;
    let echo = workspace
        .call("bash", serde_json::json!({ "command": "echo fine" }))
        .await;
    assert_eq!(echo.text(), "fine\n");
    let other = workspace.call("bash", serde_json::json!({ "command": "pwd" })).await;
    assert_eq!(
        blocked(&other),
        "Tool call blocked: approval denied: bash would need approval (dont_ask mode)"
    );
    workspace.close().await;
}

async fn bypass_keeps_deny_and_ask_rules(backend: Backend) {
    let workspace = open(
        backend,
        policy(
            PermissionMode::Bypass,
            &[],
            &["Bash(git push:*)"],
            &["Read(./secret.txt)"],
        ),
    )
    .await;
    let echo = workspace
        .call("bash", serde_json::json!({ "command": "echo free" }))
        .await;
    assert_eq!(echo.text(), "free\n");
    workspace.write("secret.txt", "hidden");
    let secret = workspace
        .call("read", serde_json::json!({ "path": "secret.txt" }))
        .await;
    assert_eq!(
        blocked(&secret),
        "Tool call blocked: approval denied: denied by rule Read(./secret.txt)"
    );
    let (submission, request) = asking(&workspace, "c1", "bash", serde_json::json!({ "command": "git push" })).await;
    workspace
        .conversation
        .decide(request.id, ReviewDecision::Abort, &ctx())
        .await
        .unwrap();
    settled(&submission).await;
    workspace.close().await;
}

async fn an_approved_rule_applies_to_every_conversation(backend: Backend) {
    let workspace = open(backend, policy(PermissionMode::Default, &[], &[], &[])).await;
    let (submission, request) = asking(&workspace, "c1", "bash", serde_json::json!({ "command": "echo one" })).await;
    let rule = request.proposed_rule.clone().unwrap();
    assert_eq!(rule, "Bash(echo one:*)");
    workspace
        .conversation
        .decide(
            request.id,
            ReviewDecision::ApprovedWithRule { rule: rule.clone() },
            &ctx(),
        )
        .await
        .unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    assert_eq!(
        workspace.harness.permission_rules(&ctx()).await.unwrap().allow,
        std::slice::from_ref(&rule)
    );

    // Another conversation runs the same command without asking.
    let other = workspace
        .harness
        .create_conversation(
            ConversationCreateOptions::ownerless().with_agent(delta_harness::agent::AgentChange {
                model: delta_harness::agent::Change::Set(crate::support::fixture::model_ref()),
                ..delta_harness::agent::AgentChange::default()
            }),
            &ctx(),
        )
        .await
        .unwrap();
    workspace.fixture.model.push([
        Reply::calls_with(&[("c2", "bash", serde_json::json!({ "command": "echo one two" }))]),
        Reply::text("Done"),
    ]);
    let submission = other.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    assert!(other.pending_approvals(&ctx()).await.unwrap().is_empty());

    assert!(workspace.harness.remove_permission_rule(&rule, &ctx()).await.unwrap());
    assert!(
        workspace
            .harness
            .permission_rules(&ctx())
            .await
            .unwrap()
            .allow
            .is_empty()
    );
    workspace.close().await;
}

/// A task that never finishes on its own: the owner of a conversation.
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

    fn abort<'a>(
        &'a self,
        _phase: Json,
        _rt: &'a TaskRuntime<Self>,
        _ctx: &'a Context,
    ) -> impl Future<Output = Result<Next<Self>, TaskError>> + Send + 'a {
        std::future::ready(Ok(Next::abort(None)))
    }
}

async fn forks_and_owned_conversations_take_the_permissions(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture
        .registry
        .install(Extension::new("forever").with_task(Arc::new(Forever)))
        .unwrap();
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    conversation
        .set_permission_mode(PermissionMode::Plan, &ctx())
        .await
        .unwrap();
    let mut events = conversation.events(&ctx()).await.unwrap();
    conversation
        .update_permissions(
            PermissionsChange {
                add: PermissionRules {
                    deny: strings(&["Read(./.env)"]),
                    ..PermissionRules::default()
                },
                ..PermissionsChange::default()
            },
            &ctx(),
        )
        .await
        .unwrap();
    let batch = tokio::time::timeout(PATIENCE, events.next()).await.unwrap().unwrap();
    assert!(
        batch.iter().any(|event| matches!(
            event,
            AgentEvent::PermissionsChanged { permissions } if permissions.rules.deny == ["Read(./.env)"]
        )),
        "{batch:?}"
    );

    fixture.model.push([Reply::text("Hello")]);
    settled(&conversation.submit(SubmissionDraft::input("hi"), &ctx()).await.unwrap()).await;
    let at = entries(&conversation).await.last().unwrap().id;
    let fork = conversation
        .fork(at, ConversationCreateOptions::ownerless(), &ctx())
        .await
        .unwrap();
    conversation
        .set_permission_mode(PermissionMode::AcceptEdits, &ctx())
        .await
        .unwrap();
    let forked = fork.permissions(&ctx()).await.unwrap();
    assert_eq!(forked.mode, Some(PermissionMode::Plan));
    assert_eq!(forked.rules.deny, ["Read(./.env)"]);

    // A conversation owned by a task of this one starts with its permissions.
    let task = conversation
        .commit(&ctx(), |tx| {
            Box::pin(async move { Ok(tx.create_task(&Forever, Json::Null, TaskOptions::default()).await?.id()) })
        })
        .await
        .unwrap();
    let owned = harness
        .create_conversation(
            ConversationCreateOptions {
                ownership: ConversationOwnership::Task(task),
                agent: None,
                init: None,
            },
            &ctx(),
        )
        .await
        .unwrap();
    assert_eq!(
        owned.permissions(&ctx()).await.unwrap().mode,
        Some(PermissionMode::AcceptEdits)
    );
    harness.close(&ctx()).await.unwrap();
}

async fn rules_are_validated(backend: Backend) {
    let workspace = open(backend, policy(PermissionMode::Default, &[], &[], &[])).await;
    let invalid = PermissionsChange {
        add: PermissionRules {
            allow: strings(&["Bash(ls"]),
            ..PermissionRules::default()
        },
        ..PermissionsChange::default()
    };
    let error = workspace
        .conversation
        .update_permissions(invalid, &ctx())
        .await
        .unwrap_err();
    assert!(
        matches!(error, HarnessError::Approval(ApprovalError::InvalidRule(_))),
        "{error:?}"
    );

    let (submission, request) = asking(&workspace, "c1", "bash", serde_json::json!({ "command": "ls" })).await;
    let bad = ReviewDecision::ApprovedWithRule {
        rule: "fetch(x)".to_owned(),
    };
    let error = workspace
        .conversation
        .decide(request.id, bad, &ctx())
        .await
        .unwrap_err();
    assert!(
        matches!(error, HarnessError::Approval(ApprovalError::InvalidRule(_))),
        "{error:?}"
    );
    assert_eq!(workspace.conversation.pending_approvals(&ctx()).await.unwrap().len(), 1);
    workspace
        .conversation
        .decide(request.id, ReviewDecision::Approved, &ctx())
        .await
        .unwrap();
    settled(&submission).await;
    workspace.close().await;
}

/// The policy under test, which hangs after its first decision so the harness can crash
/// between the decision and the request.
struct HangOnce {
    inner: Arc<dyn ApprovalPolicy>,
    decisions: Mutex<Vec<PolicyDecision>>,
}

impl ApprovalPolicy for HangOnce {
    fn decide<'a>(
        &'a self,
        call: &'a ApprovalCall<'a>,
        api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<PolicyDecision, PolicyError>> {
        Box::pin(async move {
            let decision = self.inner.decide(call, api, ctx).await?;
            let first = {
                let mut decisions = self.decisions.lock().unwrap();
                decisions.push(decision.clone());
                decisions.len() == 1
            };
            if first {
                ctx.cancelled().await;
            }
            Ok(decision)
        })
    }
}

async fn a_rerun_reuses_the_decision(backend: Backend) {
    let dir = TempDir::new();
    let mut fixture = Fixture::new(backend).await;
    fixture.env = Some(Arc::new(LocalEnvBuilder::new(dir.path())));
    fixture.registry.install(coding_tools(BashOptions::default())).unwrap();
    let hang = Arc::new(HangOnce {
        inner: policy(PermissionMode::Default, &[], &[], &[]),
        decisions: Mutex::default(),
    });
    let options = |fixture: &Fixture| {
        let mut options = fixture.options();
        options.approval = hang.clone();
        options
    };
    fixture.model.push([
        Reply::calls_with(&[("c1", "bash", serde_json::json!({ "command": "echo once" }))]),
        Reply::text("Done"),
    ]);
    let harness = Harness::open(fixture.storage.clone(), options(&fixture), &ctx())
        .await
        .unwrap();
    let conversation = chat(&harness).await;
    conversation.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    eventually(|| async { hang.decisions.lock().unwrap().len() == 1 }).await;
    let closed = harness.session().closed();
    drop((conversation, harness));
    tokio::time::timeout(PATIENCE, closed).await.unwrap();

    // Bypass would allow the command, but the rerun reuses the decision it memoized.
    let harness = Harness::open(fixture.storage.clone(), options(&fixture), &ctx())
        .await
        .unwrap();
    let conversation = chat(&harness).await;
    conversation
        .set_permission_mode(PermissionMode::Bypass, &ctx())
        .await
        .unwrap();
    harness.resume().unwrap();
    eventually(|| async { conversation.pending_approvals(&ctx()).await.unwrap().len() == 1 }).await;
    let decisions = hang.decisions.lock().unwrap().clone();
    assert_eq!(decisions.len(), 2);
    assert_eq!(decisions[0], decisions[1]);
    assert!(matches!(decisions[1], PolicyDecision::Ask { .. }));
    harness.close(&ctx()).await.unwrap();
}
