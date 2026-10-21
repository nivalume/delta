//! Asking and deciding: approve, deny, session grants, abort, timeout and refusals.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use delta_harness::approval::{ApprovalError, ApprovalId, ApprovalsDoc, ApprovalsState, ReviewDecision};
use delta_harness::documents::DocAddress;
use delta_harness::generation::SlotStatus;
use delta_harness::harness::HarnessError;
use delta_harness::inbox::SubmissionDraft;
use delta_harness::records::{SubmissionState, TaskId};

use crate::support::fixture::{Fixture, chat, ctx, live, settled};
use crate::support::model::Reply;
use crate::{Backend, decision_entries, open, requests, results, settled_slowly, text};

cases!(
    an_approved_call_runs_with_the_requested_arguments,
    a_denied_call_is_blocked,
    approval_for_the_session_skips_later_prompts,
    an_abort_decision_aborts_the_run,
    decide_refuses_what_it_cannot_decide,
);

paused_cases!(an_unanswered_request_times_out);

fn echo_call(id: &str, text: &str) -> (String, &'static str, serde_json::Value) {
    (id.to_owned(), "echo", serde_json::json!({ "text": text }))
}

fn reply(calls: &[(String, &'static str, serde_json::Value)]) -> Reply {
    let calls: Vec<(&str, &str, serde_json::Value)> = calls
        .iter()
        .map(|(id, name, arguments)| (id.as_str(), *name, arguments.clone()))
        .collect();
    Reply::calls_with(&calls)
}

async fn an_approved_call_runs_with_the_requested_arguments(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    let harness = open(&fixture, &runs, None).await;
    fixture
        .model
        .push([reply(&[echo_call("c1", "hi")]), Reply::text("Echoed")]);
    let conversation = chat(&harness).await;
    let submission = conversation
        .submit(SubmissionDraft::input("echo"), &ctx())
        .await
        .unwrap();
    let pending = requests(&conversation, 1).await;
    let request = &pending[0];
    assert_eq!((request.call_id.as_str(), request.name.as_str()), ("c1", "echo"));
    assert_eq!(request.arguments.get("text"), Some(&serde_json::json!("hi")));
    assert_eq!(request.prompt.as_deref(), Some(r#"Run echo {"text": String("hi")}?"#));
    assert_eq!(request.deadline - request.requested_at, 15 * 60 * 1000);
    assert_eq!(*runs.lock().unwrap(), 0);
    let slot = live(&harness, &conversation).await.tools.unwrap().remove(0);
    assert_eq!(slot.status, SlotStatus::Pending);
    let everywhere = harness.pending_approvals(&ctx()).await.unwrap();
    assert_eq!(everywhere, vec![(conversation.id(), request.clone())]);

    harness
        .decide(request.id, ReviewDecision::Approved, &ctx())
        .await
        .unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    assert_eq!(text(&results(&conversation).await[0]), "hi");
    assert_eq!(*runs.lock().unwrap(), 1);
    assert_eq!(
        decision_entries(&conversation).await,
        [serde_json::json!({ "type": "approved" })]
    );
    let state: ApprovalsState = delta_chord::json::from_json(
        &harness
            .snapshot(DocAddress::<ApprovalsDoc>::new(conversation.id()), &ctx())
            .await
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    assert!(state.requests.is_empty() && state.grants.is_empty());
    // The decision is not model context.
    let last = fixture.model.requests().last().unwrap().messages.clone();
    assert!(matches!(last.last(), Some(delta_ai::types::Message::ToolResult(_))));
    harness.close(&ctx()).await.unwrap();
}

async fn a_denied_call_is_blocked(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    let harness = open(&fixture, &runs, None).await;
    fixture.model.push([
        reply(&[echo_call("c1", "rm -rf"), echo_call("c2", "ls")]),
        Reply::text("Blocked"),
    ]);
    let conversation = chat(&harness).await;
    let submission = conversation
        .submit(SubmissionDraft::input("echo"), &ctx())
        .await
        .unwrap();
    let pending = requests(&conversation, 2).await;
    let denied = ReviewDecision::Denied {
        rejection: Some("too dangerous".to_owned()),
    };
    conversation.decide(pending[0].id, denied, &ctx()).await.unwrap();
    let plain = ReviewDecision::Denied { rejection: None };
    conversation.decide(pending[1].id, plain, &ctx()).await.unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    let results = results(&conversation).await;
    let of = |id: &str| text(results.iter().find(|result| result.tool_call_id == id).unwrap());
    assert_eq!(
        of("c1"),
        "<harness>\n[error] Tool call blocked: approval denied: too dangerous\n</harness>"
    );
    assert_eq!(
        of("c2"),
        "<harness>\n[error] Tool call blocked: approval denied\n</harness>"
    );
    assert!(results.iter().all(|result| result.is_error));
    assert_eq!(*runs.lock().unwrap(), 0);
    assert!(conversation.pending_approvals(&ctx()).await.unwrap().is_empty());
    harness.close(&ctx()).await.unwrap();
}

async fn approval_for_the_session_skips_later_prompts(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    let harness = open(&fixture, &runs, None).await;
    fixture.model.push([
        reply(&[echo_call("c1", "status")]),
        reply(&[echo_call("c2", "status")]),
        reply(&[echo_call("c3", "deploy")]),
        Reply::text("Done"),
    ]);
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    let first = requests(&conversation, 1).await.remove(0);
    conversation
        .decide(first.id, ReviewDecision::ApprovedForSession, &ctx())
        .await
        .unwrap();
    // `c2` has the same key and runs without a request; `c3` asks.
    let third = requests(&conversation, 1).await.remove(0);
    assert_eq!(third.call_id, "c3");
    conversation
        .decide(third.id, ReviewDecision::Approved, &ctx())
        .await
        .unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    let texts: Vec<String> = results(&conversation).await.iter().map(text).collect();
    assert_eq!(texts, ["status", "status", "deploy"]);
    let state: ApprovalsState = delta_chord::json::from_json(
        &harness
            .snapshot(DocAddress::<ApprovalsDoc>::new(conversation.id()), &ctx())
            .await
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(state.grants.len(), 1);
    assert_eq!(state.grants[0].key, r#"{"text":"status"}"#);
    assert_eq!(
        decision_entries(&conversation).await,
        [
            serde_json::json!({ "type": "approved_for_session" }),
            serde_json::json!({ "type": "approved" })
        ]
    );
    harness.close(&ctx()).await.unwrap();
}

async fn an_abort_decision_aborts_the_run(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    let harness = open(&fixture, &runs, None).await;
    fixture.model.push([reply(&[echo_call("c1", "hi")])]);
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    let request = requests(&conversation, 1).await.remove(0);
    conversation
        .decide(request.id, ReviewDecision::Abort, &ctx())
        .await
        .unwrap();
    let state = settled(&submission).await.state;
    assert!(
        matches!(&state, SubmissionState::Unanswered { reason, .. } if reason == "aborted"),
        "{state:?}"
    );
    assert_eq!(
        text(&results(&conversation).await[0]),
        "<harness>\n[error] Tool echo was aborted\n</harness>"
    );
    assert_eq!(*runs.lock().unwrap(), 0);
    let state: ApprovalsState = delta_chord::json::from_json(
        &harness
            .snapshot(DocAddress::<ApprovalsDoc>::new(conversation.id()), &ctx())
            .await
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    assert!(state.requests.is_empty());
    assert!(live(&harness, &conversation).await.run.is_none());
    harness.close(&ctx()).await.unwrap();
}

async fn decide_refuses_what_it_cannot_decide(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    let harness = open(&fixture, &runs, None).await;
    fixture
        .model
        .push([reply(&[echo_call("c1", "hi")]), Reply::text("Done")]);
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    let request = requests(&conversation, 1).await.remove(0);
    let refused = conversation.decide(request.id, ReviewDecision::TimedOut, &ctx()).await;
    assert!(matches!(
        refused,
        Err(HarnessError::Approval(ApprovalError::InvalidDecision(_)))
    ));
    let unknown = ApprovalId(TaskId(999_999));
    let refused = harness.decide(unknown, ReviewDecision::Approved, &ctx()).await;
    assert!(matches!(
        refused,
        Err(HarnessError::Approval(ApprovalError::UnknownApproval(_)))
    ));
    conversation
        .decide(request.id, ReviewDecision::Approved, &ctx())
        .await
        .unwrap();
    let again = conversation.decide(request.id, ReviewDecision::Approved, &ctx()).await;
    assert!(matches!(
        again,
        Err(HarnessError::Approval(ApprovalError::NotPending(_)))
    ));
    settled(&submission).await;
    assert_eq!(decision_entries(&conversation).await.len(), 1);
    harness.close(&ctx()).await.unwrap();
}

async fn an_unanswered_request_times_out(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    let harness = open(&fixture, &runs, Some(Duration::from_secs(30))).await;
    fixture
        .model
        .push([reply(&[echo_call("c1", "hi")]), Reply::text("Timed out")]);
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    let request = requests(&conversation, 1).await.remove(0);
    assert!(matches!(
        settled_slowly(&submission).await.state,
        SubmissionState::Done { .. }
    ));
    assert!(fixture.now() >= request.deadline);
    assert_eq!(
        text(&results(&conversation).await[0]),
        "<harness>\n[error] Tool call blocked: approval timed out after 30s\n</harness>"
    );
    assert_eq!(
        decision_entries(&conversation).await,
        [serde_json::json!({ "type": "timed_out" })]
    );
    let late = conversation.decide(request.id, ReviewDecision::Approved, &ctx()).await;
    assert!(matches!(
        late,
        Err(HarnessError::Approval(ApprovalError::NotPending(_)))
    ));
    assert_eq!(*runs.lock().unwrap(), 0);
    harness.close(&ctx()).await.unwrap();
}
