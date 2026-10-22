//! The built-in tools behind an approval policy that asks for every call: a denied command
//! never runs, an approved write lands, and a grant for the session lets the same command
//! run again without asking.

use std::sync::Arc;

use delta_harness::approval::{AskAll, ReviewDecision};
use delta_harness::inbox::SubmissionDraft;
use delta_harness::records::SubmissionState;
use delta_harness::tools::BashOptions;

use crate::support::fixture::{ctx, eventually, settled};
use crate::support::model::Reply;
use crate::{Backend, Workspace};

cases!(decisions_reach_the_files_and_commands);

async fn decisions_reach_the_files_and_commands(backend: Backend) {
    let workspace = Workspace::open(backend, BashOptions::default(), Some(Arc::new(AskAll))).await;
    let conversation = &workspace.conversation;
    workspace.fixture.model.push([
        Reply::calls_with(&[
            ("c1", "bash", serde_json::json!({ "command": "touch made" })),
            ("c2", "write", serde_json::json!({ "path": "w.txt", "content": "ok" })),
        ]),
        Reply::text("Half done"),
    ]);
    let submission = conversation.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    eventually(|| async { conversation.pending_approvals(&ctx()).await.unwrap().len() == 2 }).await;
    let pending = conversation.pending_approvals(&ctx()).await.unwrap();
    let of = |tool: &str| {
        pending
            .iter()
            .find(|request| request.name == tool)
            .unwrap_or_else(|| panic!("a request for {tool}: {pending:?}"))
            .id
    };
    let denied = ReviewDecision::Denied {
        rejection: Some("no commands".to_owned()),
    };
    conversation.decide(of("bash"), denied, &ctx()).await.unwrap();
    conversation
        .decide(of("write"), ReviewDecision::Approved, &ctx())
        .await
        .unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    assert!(!workspace.path("made").exists(), "a denied command never runs");
    assert_eq!(workspace.read("w.txt"), "ok");

    // Approved for the session: the same command runs again without a prompt.
    workspace.fixture.model.push([
        Reply::calls_with(&[("c3", "bash", serde_json::json!({ "command": "echo hi" }))]),
        Reply::text("Said hi"),
    ]);
    let submission = conversation.submit(SubmissionDraft::input("hi"), &ctx()).await.unwrap();
    eventually(|| async { conversation.pending_approvals(&ctx()).await.unwrap().len() == 1 }).await;
    let request = conversation.pending_approvals(&ctx()).await.unwrap().remove(0);
    conversation
        .decide(request.id, ReviewDecision::ApprovedForSession, &ctx())
        .await
        .unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    let again = workspace
        .call("bash", serde_json::json!({ "command": "echo hi" }))
        .await;
    assert!(!again.message.is_error, "{:?}", again.diagnostics);
    assert_eq!(again.text(), "hi\n");
    workspace.close().await;
}
