//! Withdrawal by a conversation abort, and what a failed run does with the inbox.

use std::sync::Arc;

use delta_harness::agent::AfterFailedRun;
use delta_harness::harness::AbortOptions;
use delta_harness::inbox::SubmissionDraft;
use delta_harness::records::{EntryDraft, SubmissionState};
use tokio::sync::Notify;

use crate::Backend;
use crate::support::fixture::{Fixture, chat, ctx, entries, eventually, settled};
use crate::support::model::Reply;

cases!(
    a_conversation_abort_withdraws_queued_inputs_and_keeps_writes,
    a_failed_run_keeps_its_queue_by_default,
    a_failed_run_continues_with_its_queue,
    a_failed_run_withdraws_its_queued_inputs,
);

fn reason(state: &SubmissionState) -> Option<&str> {
    match state {
        SubmissionState::Unanswered { reason, .. } => Some(reason),
        _ => None,
    }
}

async fn a_conversation_abort_withdraws_queued_inputs_and_keeps_writes(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let gate = Arc::new(Notify::new());
    fixture.model.push([Reply::text("never finished").gate(&gate)]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let running = conversation
        .submit(SubmissionDraft::input("one"), &ctx())
        .await
        .unwrap();
    // The run's request is in flight before anything else happens.
    eventually(|| async { fixture.model.requests().len() == 1 }).await;
    let queued = conversation
        .submit(SubmissionDraft::input("two"), &ctx())
        .await
        .unwrap();
    let summary = EntryDraft {
        model: Some(vec![delta_ai::types::Message::user("a passive note")]),
        ..EntryDraft::new("app.summary")
    };
    let write = conversation
        .submit(SubmissionDraft::write(summary), &ctx())
        .await
        .unwrap();
    conversation.abort(AbortOptions::default(), &ctx()).await.unwrap();
    assert_eq!(reason(&settled(&running).await.state), Some("aborted"));
    assert_eq!(reason(&settled(&queued).await.state), Some("aborted"));
    // The write stays queued for the next boundary.
    assert!(matches!(
        write.status(&ctx()).await.unwrap().state,
        SubmissionState::Queued
    ));
    let inspection = harness.inspect(&ctx()).await.unwrap();
    let ids: Vec<_> = inspection.submissions.iter().map(|submission| submission.id).collect();
    assert_eq!(ids, [write.id()]);
    // The next input runs a final boundary that places the write first.
    fixture.model.push([Reply::text("fresh")]);
    let next = conversation
        .submit(SubmissionDraft::input("three"), &ctx())
        .await
        .unwrap();
    settled(&next).await;
    assert!(matches!(
        write.status(&ctx()).await.unwrap().state,
        SubmissionState::Done { .. }
    ));
    let kinds: Vec<String> = entries(&conversation)
        .await
        .into_iter()
        .map(|entry| entry.kind)
        .collect();
    let summary_at = kinds.iter().position(|kind| kind == "app.summary").unwrap();
    assert!(kinds[summary_at + 1..].contains(&"delta.user".to_owned()));
    harness.close(&ctx()).await.unwrap();
}

/// One failing run with a follow-up queued behind it, under `policy`.
async fn failed_run_with_queue(backend: Backend, policy: Option<AfterFailedRun>) -> (SubmissionState, usize) {
    let fixture = Fixture::new(backend).await;
    fixture.settings(|settings| settings.after_failed_run = policy);
    let gate = Arc::new(Notify::new());
    fixture.model.push([
        Reply::error("invalid request").gate(&gate),
        Reply::text("follow-up answered"),
    ]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let failing = conversation
        .submit(SubmissionDraft::input("one"), &ctx())
        .await
        .unwrap();
    let queued = conversation
        .submit(SubmissionDraft::input("two"), &ctx())
        .await
        .unwrap();
    gate.notify_one();
    assert_eq!(reason(&settled(&failing).await.state), Some("model_error"));
    conversation.wait_for_idle(&ctx()).await.unwrap();
    let state = queued.status(&ctx()).await.unwrap().state;
    let requests = fixture.model.requests().len();
    harness.close(&ctx()).await.unwrap();
    (state, requests)
}

async fn a_failed_run_keeps_its_queue_by_default(backend: Backend) {
    let (state, requests) = failed_run_with_queue(backend, None).await;
    assert!(matches!(state, SubmissionState::Queued), "{state:?}");
    assert_eq!(requests, 1);
}

async fn a_failed_run_continues_with_its_queue(backend: Backend) {
    let (state, requests) = failed_run_with_queue(backend, Some(AfterFailedRun::Continue)).await;
    assert!(matches!(state, SubmissionState::Done { .. }), "{state:?}");
    assert_eq!(requests, 2);
}

async fn a_failed_run_withdraws_its_queued_inputs(backend: Backend) {
    let (state, requests) = failed_run_with_queue(backend, Some(AfterFailedRun::Withdraw)).await;
    assert_eq!(reason(&state), Some("run_failed"));
    assert_eq!(requests, 1);
}
