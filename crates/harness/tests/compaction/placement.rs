//! Conversation-owned compactions (manual, background) and their placement, hooks, retries,
//! abort and failure.

use std::sync::Arc;
use std::time::Duration;

use delta_chord::context::Context;
use delta_harness::agent::PartialRetryPolicy;
use delta_harness::compaction::CompactionResult;
use delta_harness::harness::AbortOptions;
use delta_harness::records::{SubmissionState, TaskId, TaskOutcome, TaskQuery};
use delta_harness::registry::{BeforeCompact, CompactDecision, CompactionHooks, Extension, HookApi, HookResult};
use futures::future::BoxFuture;
use tokio::sync::Notify;

use crate::support::fixture::{CONTEXT_WINDOW, Fixture, MODEL_ID, PATIENCE, chat, ctx, eventually, live, settled};
use crate::support::model::Reply;
use crate::{Backend, history, keep_little, run, summaries, summary_requests, user_text};

cases!(
    a_manual_compaction_places_a_write_submission,
    a_background_compaction_runs_beside_the_request,
    before_compact_supplies_or_declines,
    an_abort_removes_the_status,
    no_model_records_a_failure,
);

paused_cases!(a_retryable_summary_error_backs_off);

/// The completed result of compaction `task`.
async fn completed(harness: &delta_harness::harness::Harness, task: TaskId) -> CompactionResult {
    let settled = tokio::time::timeout(PATIENCE * 100, harness.wait_for_task(task, &ctx()))
        .await
        .unwrap()
        .unwrap();
    match settled.outcome() {
        TaskOutcome::Completed { result } => delta_chord::json::from_json(result).unwrap(),
        other => panic!("completed: {other:?}"),
    }
}

async fn a_manual_compaction_places_a_write_submission(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    keep_little(&fixture, |_| {});
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    history(&fixture, &conversation).await;
    fixture.model.push([Reply::text("Second answer")]);
    run(&conversation, "Second question").await;
    fixture.model.push_summaries([Reply::text("Two questions so far.")]);
    let task = conversation
        .compact(Some("the open questions".to_owned()), &ctx())
        .await
        .unwrap();
    let record = harness.get_task(task, &ctx()).await.unwrap().unwrap();
    assert!(
        record.owner.is_none() && !record.background,
        "manual compactions are ordinary work"
    );
    let result = completed(&harness, task).await;
    let submission = harness
        .submission(result.submission_id.unwrap(), &ctx())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    let summaries = summaries(&conversation).await;
    assert_eq!(summaries.len(), 1);
    assert!(user_text(&summaries[0]).contains("Two questions so far."));
    let requests = summary_requests(&fixture);
    let prompt = match &requests[0].messages[1] {
        delta_ai::types::Message::User(user) => user.content.text(),
        other => panic!("a prompt: {other:?}"),
    };
    assert!(prompt.ends_with("\n\nAdditional focus: the open questions"));
    // The next request starts at the summary.
    fixture.model.push([Reply::text("Third answer")]);
    run(&conversation, "Third question").await;
    let last = fixture.model.requests().last().unwrap().messages.clone();
    let delta_ai::types::Message::User(first) = &last[1] else {
        panic!("the summary")
    };
    assert!(first.content.text().contains("Two questions so far."));
    harness.close(&ctx()).await.unwrap();
}

async fn a_background_compaction_runs_beside_the_request(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    keep_little(&fixture, |policy| {
        policy.reserve_tokens = Some(0);
        policy.background_tokens = Some(CONTEXT_WINDOW - 5);
    });
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    history(&fixture, &conversation).await;
    assert!(summaries(&conversation).await.is_empty());
    fixture.model.push([Reply::text("Answered anyway")]);
    fixture.model.push_summaries([Reply::text("Background summary.")]);
    let state = run(&conversation, "Second question").await;
    assert!(matches!(state, SubmissionState::Done { .. }), "{state:?}");
    eventually(|| async { summaries(&conversation).await.len() == 1 }).await;
    // Placed by its own commit when idle, or by the run's boundary when it was busy.
    let query = TaskQuery {
        kind: Some("delta.compaction".to_owned()),
        ..TaskQuery::default()
    };
    let tasks = harness
        .commit(&ctx(), move |tx| {
            Box::pin(async move { tx.scan_tasks(query, 10, None).await })
        })
        .await
        .unwrap()
        .items;
    assert_eq!(tasks.len(), 1);
    assert!(tasks[0].owner.is_none() && tasks[0].background);
    eventually(|| async { live(&harness, &conversation).await.compactions.is_none() }).await;
    harness.close(&ctx()).await.unwrap();
}

/// A hook that supplies a summary, or declines.
struct Decide(Option<&'static str>);

impl CompactionHooks for Decide {
    fn before_compact<'a>(
        &'a self,
        compaction: &'a BeforeCompact,
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<CompactDecision>> {
        assert!(!compaction.messages.is_empty());
        let decision = match self.0 {
            Some(summary) => CompactDecision::Summary(summary.to_owned()),
            None => CompactDecision::Decline,
        };
        Box::pin(async move { Ok(Some(decision)) })
    }
}

fn hooks(decision: Option<&'static str>) -> Extension {
    Extension::new("summaries").with_hooks::<dyn CompactionHooks>("delta.compaction", Arc::new(Decide(decision)))
}

async fn before_compact_supplies_or_declines(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    keep_little(&fixture, |_| {});
    fixture.registry.install(hooks(Some("A hook's summary."))).unwrap();
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    history(&fixture, &conversation).await;
    fixture.model.push([Reply::text("Second answer")]);
    run(&conversation, "Second question").await;
    let task = conversation.compact(None, &ctx()).await.unwrap();
    assert!(completed(&harness, task).await.submission_id.is_some());
    eventually(|| async { summaries(&conversation).await.len() == 1 }).await;
    assert!(user_text(&summaries(&conversation).await[0]).contains("A hook's summary."));
    assert!(summary_requests(&fixture).is_empty());

    fixture.registry.install(hooks(None)).unwrap();
    fixture.model.push([Reply::text("Third answer")]);
    run(&conversation, "Third question").await;
    let task = conversation.compact(None, &ctx()).await.unwrap();
    assert_eq!(completed(&harness, task).await, CompactionResult::default());
    assert_eq!(summaries(&conversation).await.len(), 1);
    harness.close(&ctx()).await.unwrap();
}

async fn an_abort_removes_the_status(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    keep_little(&fixture, |_| {});
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    history(&fixture, &conversation).await;
    fixture.model.push([Reply::text("Second answer")]);
    run(&conversation, "Second question").await;
    let gate = Arc::new(Notify::new());
    fixture.model.push_summaries([Reply::text("Never placed.").gate(&gate)]);
    let task = conversation.compact(None, &ctx()).await.unwrap();
    eventually(|| async { summary_requests(&fixture).len() == 1 }).await;
    let status = live(&harness, &conversation).await.compactions.unwrap().remove(0);
    assert_eq!((status.task_id, status.attempt, status.blocking), (task, 1, false));
    conversation.abort(AbortOptions::default(), &ctx()).await.unwrap();
    let settled = harness.wait_for_task(task, &ctx()).await.unwrap();
    assert!(matches!(settled.outcome(), TaskOutcome::Aborted { .. }));
    let state = live(&harness, &conversation).await;
    assert!(state.compactions.is_none() && state.compaction_failure.is_none());
    assert!(summaries(&conversation).await.is_empty());
    harness.close(&ctx()).await.unwrap();
}

async fn no_model_records_a_failure(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    keep_little(&fixture, |_| {});
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    history(&fixture, &conversation).await;
    fixture.model.forget(MODEL_ID);
    let task = conversation.compact(None, &ctx()).await.unwrap();
    let settled = harness.wait_for_task(task, &ctx()).await.unwrap();
    let TaskOutcome::Failed { error, .. } = settled.outcome() else {
        panic!("failed: {:?}", settled.outcome())
    };
    assert_eq!(error.message, format!("Model fake/{MODEL_ID} is not available"));
    let failure = live(&harness, &conversation).await.compaction_failure.unwrap();
    assert_eq!(
        (failure.task_id, failure.error.as_str()),
        (task, error.message.as_str())
    );
    harness.close(&ctx()).await.unwrap();
}

async fn a_retryable_summary_error_backs_off(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    keep_little(&fixture, |_| {});
    fixture.settings(|settings| {
        settings.retry = Some(PartialRetryPolicy {
            base_delay: Some(Duration::from_secs(5)),
            ..PartialRetryPolicy::default()
        });
    });
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    history(&fixture, &conversation).await;
    fixture.model.push([Reply::text("Second answer")]);
    run(&conversation, "Second question").await;
    fixture
        .model
        .push_summaries([Reply::retryable(), Reply::text("Summary after a retry.")]);
    let start = fixture.now();
    let task = conversation.compact(None, &ctx()).await.unwrap();
    assert!(completed(&harness, task).await.submission_id.is_some());
    assert!(fixture.now() - start >= 5_000);
    assert_eq!(summary_requests(&fixture).len(), 2);
    eventually(|| async { summaries(&conversation).await.len() == 1 }).await;
    harness.close(&ctx()).await.unwrap();
}
