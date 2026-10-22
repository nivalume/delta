//! A call waiting for its decision survives a crash: the rerun reads a decision committed
//! meanwhile, or waits again until the stored deadline.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use delta_harness::approval::{ReviewDecision, record_decision};
use delta_harness::harness::Harness;
use delta_harness::inbox::{QueueModes, SubmissionDraft};
use delta_harness::records::{SubmissionId, SubmissionState};

use crate::support::fixture::{Fixture, PATIENCE, chat, ctx, entries, settled};
use crate::support::model::Reply;
use crate::{Backend, decision_entries, open, requests, results, settled_slowly, text};

cases!(a_decision_committed_while_down_is_read_by_the_rerun);

paused_cases!(a_crash_while_waiting_times_out_at_the_stored_deadline);

/// Submits a call of `echo` and returns the submission once its request is pending.
async fn ask(harness: &Harness) -> (SubmissionId, delta_harness::approval::ApprovalRequest) {
    let conversation = chat(harness).await;
    let submission = conversation.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    let request = requests(&conversation, 1).await.remove(0);
    (submission.id(), request)
}

async fn crash(harness: Harness) {
    let closed = harness.session().closed();
    drop(harness);
    tokio::time::timeout(PATIENCE, closed)
        .await
        .expect("the old Session ends");
}

async fn a_decision_committed_while_down_is_read_by_the_rerun(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    let harness = open(&fixture, &runs, None).await;
    fixture.model.push([Reply::calls_with(&[(
        "c1",
        "echo",
        serde_json::json!({ "text": "hi" }),
    )])]);
    let (id, request) = ask(&harness).await;
    crash(harness).await;

    // Reopened paused: nothing runs, so only the committed decision can reach the call.
    let harness = open(&fixture, &runs, None).await;
    let conversation = chat(&harness).await.id();
    harness
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let modes = QueueModes::from(&delta_harness::agent::Settings::default());
                record_decision(tx, conversation, request.id, ReviewDecision::Approved, 0, modes).await
            })
        })
        .await
        .unwrap();
    fixture.model.push([Reply::text("Echoed")]);
    let submission = harness.submission(id, &ctx()).await.unwrap().unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    let conversation = chat(&harness).await;
    assert_eq!(text(&results(&conversation).await[0]), "hi");
    assert_eq!(*runs.lock().unwrap(), 1);
    assert_eq!(decision_entries(&conversation).await.len(), 1);
    harness.close(&ctx()).await.unwrap();
}

async fn a_crash_while_waiting_times_out_at_the_stored_deadline(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    let harness = open(&fixture, &runs, Some(Duration::from_mins(1))).await;
    fixture.model.push([Reply::calls_with(&[(
        "c1",
        "echo",
        serde_json::json!({ "text": "hi" }),
    )])]);
    let (id, request) = ask(&harness).await;
    tokio::time::sleep(Duration::from_secs(20)).await;
    crash(harness).await;

    let harness = open(&fixture, &runs, Some(Duration::from_mins(1))).await;
    fixture.model.push([Reply::text("Timed out")]);
    let submission = harness.submission(id, &ctx()).await.unwrap().unwrap();
    assert!(matches!(
        settled_slowly(&submission).await.state,
        SubmissionState::Done { .. }
    ));
    let now = fixture.now();
    assert!(now >= request.deadline, "timed out early");
    assert!(
        now < request.deadline + 20_000,
        "the rerun waits until the stored deadline"
    );
    let conversation = chat(&harness).await;
    assert_eq!(
        text(&results(&conversation).await[0]),
        "<harness>\n[error] Tool call blocked: approval timed out after 60s\n</harness>"
    );
    let entry = entries(&conversation)
        .await
        .into_iter()
        .find(|entry| entry.kind == "delta.approval")
        .unwrap();
    let data = serde_json::Value::from(entry.data.unwrap());
    assert_eq!(data["approval_id"], serde_json::json!(request.id.0.0));
    assert_eq!(*runs.lock().unwrap(), 0);
    harness.close(&ctx()).await.unwrap();
}
