//! `before_tool`, `after_tool` and the approval policy.

use std::sync::{Arc, Mutex};

use delta_ai::types::{JsonObject, Tool};
use delta_chord::context::Context;
use delta_harness::approval::{ApprovalCall, ApprovalPolicy, DenyAll, PolicyDecision, PolicyError};
use delta_harness::harness::Harness;
use delta_harness::registry::{BeforeTool, Extension, HookApi, HookResult, ToolHooks};
use delta_harness::tool::{FnHarnessTool, HarnessTool, TOOL_TASK, ToolExecutionResult, ToolHookCall};
use futures::future::BoxFuture;

use crate::support::fixture::{Fixture, chat, ctx};
use crate::support::model::Reply;
use crate::{Backend, results, run, texts};

cases!(
    before_tool_rewrites_and_blocks,
    a_failing_before_tool_hook_blocks,
    after_tool_replaces_the_result,
    the_approval_policy_denies,
    a_failing_policy_denies,
);

/// `echo`: answers its `text` argument; counts its executions.
fn echo(runs: &Arc<Mutex<u32>>) -> Arc<dyn HarnessTool> {
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "text": { "type": "string" } },
        "required": ["text"],
    });
    let runs = runs.clone();
    Arc::new(FnHarnessTool::new(
        Tool::new("echo", "Echo", schema),
        move |arguments, _api, _ctx| {
            *runs.lock().unwrap() += 1;
            Box::pin(async move {
                let text = arguments.get("text").and_then(serde_json::Value::as_str).unwrap_or("");
                Ok(ToolExecutionResult::text(text))
            })
        },
    ))
}

/// What a test hook does.
#[derive(Clone)]
enum Act {
    Rewrite(&'static str),
    Block(&'static str),
    Fail(&'static str),
    Replace(&'static str),
}

struct Hook(Act);

impl ToolHooks for Hook {
    fn before_tool<'a>(
        &'a self,
        call: &'a ToolHookCall,
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<BeforeTool>> {
        let act = self.0.clone();
        let seen = call
            .call
            .arguments
            .get("text")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_owned();
        Box::pin(async move {
            match act {
                Act::Rewrite(suffix) => {
                    let mut arguments = JsonObject::new();
                    arguments.insert("text".to_owned(), format!("{seen}{suffix}").into());
                    Ok(Some(BeforeTool {
                        arguments: Some(arguments),
                        block: None,
                    }))
                }
                Act::Block(text) => Ok(Some(BeforeTool {
                    arguments: None,
                    block: Some(text.to_owned()),
                })),
                Act::Fail(text) => Err(text.into()),
                Act::Replace(_) => Ok(None),
            }
        })
    }

    fn after_tool<'a>(
        &'a self,
        _call: &'a ToolHookCall,
        result: &'a ToolExecutionResult,
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<ToolExecutionResult>> {
        let _ = result;
        let act = self.0.clone();
        Box::pin(async move {
            match act {
                Act::Replace(text) => Ok(Some(ToolExecutionResult::text(text))),
                _ => Ok(None),
            }
        })
    }
}

fn hooks(name: &str, act: Act) -> Extension {
    Extension::new(name).with_hooks::<dyn ToolHooks>(TOOL_TASK, Arc::new(Hook(act)))
}

async fn one_call(fixture: &Fixture, harness: &Harness) -> Vec<String> {
    fixture.model.push([
        Reply::calls_with(&[("c1", "echo", serde_json::json!({ "text": "hi" }))]),
        Reply::text("Done"),
    ]);
    let conversation = chat(harness).await;
    run(&conversation, "echo").await;
    let results = results(&conversation).await;
    texts(results.last().unwrap()).into_iter().map(str::to_owned).collect()
}

async fn before_tool_rewrites_and_blocks(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    let registry = &fixture.registry;
    registry
        .install(Extension::new("tools").with_tool(echo(&runs)))
        .unwrap();
    registry.install(hooks("first", Act::Rewrite(" there"))).unwrap();
    registry.install(hooks("second", Act::Rewrite("!"))).unwrap();
    let harness = fixture.open().await;
    assert_eq!(one_call(&fixture, &harness).await, ["hi there!"]);
    harness.close(&ctx()).await.unwrap();

    // A block ends the chain; the first block wins.
    registry.install(hooks("second", Act::Block("not today"))).unwrap();
    registry.install(hooks("third", Act::Block("never"))).unwrap();
    let harness = fixture.open().await;
    assert_eq!(
        one_call(&fixture, &harness).await,
        ["<harness>\n[error] Tool call blocked: not today\n</harness>"]
    );
    assert_eq!(*runs.lock().unwrap(), 1);
    harness.close(&ctx()).await.unwrap();
}

async fn a_failing_before_tool_hook_blocks(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    fixture
        .registry
        .install(Extension::new("tools").with_tool(echo(&runs)))
        .unwrap();
    fixture
        .registry
        .install(hooks("broken", Act::Fail("hook crashed")))
        .unwrap();
    let harness = fixture.open().await;
    assert_eq!(
        one_call(&fixture, &harness).await,
        ["<harness>\n[error] Tool call blocked: hook crashed\n</harness>"]
    );
    assert_eq!(*runs.lock().unwrap(), 0);
    assert!(fixture.reported().iter().any(|(message, _)| message == "hook crashed"));
    harness.close(&ctx()).await.unwrap();
}

async fn after_tool_replaces_the_result(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    fixture
        .registry
        .install(Extension::new("tools").with_tool(echo(&runs)))
        .unwrap();
    fixture
        .registry
        .install(hooks("redact", Act::Replace("[redacted]")))
        .unwrap();
    let harness = fixture.open().await;
    assert_eq!(one_call(&fixture, &harness).await, ["[redacted]"]);
    harness.close(&ctx()).await.unwrap();
}

/// A policy that fails.
struct Broken;

impl ApprovalPolicy for Broken {
    fn decide<'a>(
        &'a self,
        _call: &'a ApprovalCall<'a>,
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, Result<PolicyDecision, PolicyError>> {
        Box::pin(async { Err("policy store unreachable".into()) })
    }
}

async fn open_with(fixture: &Fixture, policy: Arc<dyn ApprovalPolicy>) -> Harness {
    let mut options = fixture.options();
    options.approval = policy;
    Harness::open(fixture.storage.clone(), options, &ctx()).await.unwrap()
}

async fn the_approval_policy_denies(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    fixture
        .registry
        .install(Extension::new("tools").with_tool(echo(&runs)))
        .unwrap();
    let policy = DenyAll {
        reason: "read-only session".to_owned(),
    };
    let harness = open_with(&fixture, Arc::new(policy)).await;
    assert_eq!(
        one_call(&fixture, &harness).await,
        ["<harness>\n[error] Tool call blocked: approval denied: read-only session\n</harness>"]
    );
    assert_eq!(*runs.lock().unwrap(), 0);
    harness.close(&ctx()).await.unwrap();
}

async fn a_failing_policy_denies(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    fixture
        .registry
        .install(Extension::new("tools").with_tool(echo(&runs)))
        .unwrap();
    let harness = open_with(&fixture, Arc::new(Broken)).await;
    assert_eq!(
        one_call(&fixture, &harness).await,
        ["<harness>\n[error] Tool call blocked: approval denied: policy store unreachable\n</harness>"]
    );
    assert_eq!(*runs.lock().unwrap(), 0);
    harness.close(&ctx()).await.unwrap();
}
