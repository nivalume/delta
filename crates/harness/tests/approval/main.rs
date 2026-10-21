//! Durable approvals: requests, decisions, grants, timeouts, abort, and recovery of a
//! waiting call. Every case runs against memory and SQLite in a temp file with a scripted
//! model; timed cases run on a paused clock against memory.

#![allow(clippy::unwrap_used, clippy::too_many_lines)]

#[path = "../support/mod.rs"]
mod support;

/// Runs each case against memory and SQLite in a file, on a multi-threaded runtime.
macro_rules! cases {
    ($($case:ident),* $(,)?) => {
        mod memory {
            $(#[tokio::test(flavor = "multi_thread", worker_threads = 4)] async fn $case() { super::$case(crate::Backend::Memory).await; })*
        }
        mod sqlite {
            $(#[tokio::test(flavor = "multi_thread", worker_threads = 4)] async fn $case() { super::$case(crate::Backend::SqliteFile).await; })*
        }
    };
}

/// [`cases!`] with tokio's clock paused, memory only (SQLite runs on its own thread).
macro_rules! paused_cases {
    ($($case:ident),* $(,)?) => {
        mod paused {
            $(#[tokio::test(start_paused = true)] async fn $case() { super::$case(crate::Backend::Memory).await; })*
        }
    };
}

mod decisions;
mod recovery;

use std::sync::{Arc, Mutex};

use delta_ai::types::{Message, Tool, ToolResultMessage, UserBlock};
use delta_chord::context::Context;
use delta_harness::approval::{ApprovalCall, ApprovalPolicy, ApprovalRequest, PolicyDecision, PolicyError};
use delta_harness::harness::{Conversation, Harness, Submission};
use delta_harness::records::SubmissionRecord;
use delta_harness::registry::{Extension, HookApi};
use delta_harness::tool::{FnHarnessTool, ToolExecutionResult};
use futures::future::BoxFuture;
pub(crate) use support::Backend;

use crate::support::fixture::{Fixture, ctx, entries, eventually};

/// Asks for every call of `echo`, with an optional timeout; allows any other tool.
struct AskEcho(Option<std::time::Duration>);

impl ApprovalPolicy for AskEcho {
    fn decide<'a>(
        &'a self,
        call: &'a ApprovalCall<'a>,
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, Result<PolicyDecision, PolicyError>> {
        let decision = if call.call.call.name == "echo" {
            PolicyDecision::Ask {
                timeout: self.0,
                prompt: Some(format!("Run echo {:?}?", call.call.call.arguments)),
                proposed_rule: None,
            }
        } else {
            PolicyDecision::Allow
        };
        Box::pin(async move { Ok(decision) })
    }
}

/// Installs `echo` (answers its `text`, counting runs) and opens a Harness whose policy
/// asks for it.
async fn open(fixture: &Fixture, runs: &Arc<Mutex<u32>>, timeout: Option<std::time::Duration>) -> Harness {
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "text": { "type": "string" } },
        "required": ["text"],
    });
    let counter = runs.clone();
    let echo = FnHarnessTool::new(Tool::new("echo", "Echo", schema), move |arguments, _api, _ctx| {
        *counter.lock().unwrap() += 1;
        Box::pin(async move {
            let text = arguments.get("text").and_then(serde_json::Value::as_str).unwrap_or("");
            Ok(ToolExecutionResult::text(text))
        })
    });
    fixture
        .registry
        .install(Extension::new("tools").with_tool(Arc::new(echo)))
        .unwrap();
    let mut options = fixture.options();
    options.approval = Arc::new(AskEcho(timeout));
    Harness::open(fixture.storage.clone(), options, &ctx()).await.unwrap()
}

/// Waits until the conversation has `count` requests waiting, and returns them.
async fn requests(conversation: &Conversation, count: usize) -> Vec<ApprovalRequest> {
    eventually(|| async { conversation.pending_approvals(&ctx()).await.unwrap().len() == count }).await;
    conversation.pending_approvals(&ctx()).await.unwrap()
}

/// The submission's settled record, waiting past approval deadlines (paused clock).
async fn settled_slowly(submission: &Submission) -> SubmissionRecord {
    tokio::time::timeout(std::time::Duration::from_hours(1), submission.wait(&ctx()))
        .await
        .expect("the submission settles")
        .unwrap()
}

/// The tool results of the transcript, in order.
async fn results(conversation: &Conversation) -> Vec<ToolResultMessage> {
    entries(conversation)
        .await
        .into_iter()
        .filter(|entry| entry.kind == "delta.tool-result")
        .map(|entry| match entry.model.as_deref() {
            Some([Message::ToolResult(result)]) => result.clone(),
            other => panic!("a tool result: {other:?}"),
        })
        .collect()
}

/// The text of a result.
fn text(result: &ToolResultMessage) -> String {
    result
        .content
        .iter()
        .filter_map(|block| match block {
            UserBlock::Text(text) => Some(text.text.as_str()),
            UserBlock::Image(_) => None,
        })
        .collect()
}

/// The decisions of the `delta.approval` entries, in order.
async fn decision_entries(conversation: &Conversation) -> Vec<serde_json::Value> {
    entries(conversation)
        .await
        .into_iter()
        .filter(|entry| entry.kind == "delta.approval")
        .map(|entry| {
            assert!(entry.model.is_none(), "display-only");
            serde_json::Value::from(entry.data.unwrap())["decision"].clone()
        })
        .collect()
}
