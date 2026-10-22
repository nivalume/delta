//! Failed attempts: `no_model`, model errors, retries with backoff, and a context overflow
//! with nothing to compact (compaction itself is `tests/compaction`).

use std::time::Duration;

use delta_chord::json::Json;
use delta_harness::agent::{AgentChange, Change, PartialRetryPolicy};
use delta_harness::harness::Conversation;
use delta_harness::inbox::SubmissionDraft;
use delta_harness::model::ModelRef;
use delta_harness::records::{SubmissionState, TaskOutcome, TaskOutcomeError};

use crate::Backend;
use crate::support::fixture::{Fixture, chat, ctx, entries, kinds, settled};
use crate::support::model::{PROVIDER, Reply};

cases!(
    no_model_fails_the_run,
    a_model_error_fails_the_run,
    retries_stop_at_the_policy_limit,
    an_overflow_without_a_cut_fails,
);
paused_cases!(a_retryable_error_sleeps_and_retries);

async fn submit(conversation: &Conversation, text: &str) -> SubmissionState {
    let submission = conversation.submit(SubmissionDraft::input(text), &ctx()).await.unwrap();
    settled(&submission).await.state
}

fn unanswered(state: &SubmissionState) -> (&str, Option<&Json>) {
    match state {
        SubmissionState::Unanswered { reason, detail, .. } => (reason.as_str(), detail.as_ref()),
        other => panic!("unanswered: {other:?}"),
    }
}

async fn failure_of(harness: &delta_harness::harness::Harness, conversation: &Conversation) -> TaskOutcomeError {
    let transcript = entries(conversation).await;
    let task = transcript.iter().rev().find_map(|entry| entry.by_task_id);
    let Some(task) = task else {
        panic!("no generation wrote an entry")
    };
    let record = harness.get_task(task, &ctx()).await.unwrap().unwrap();
    match record.state.outcome() {
        Some(TaskOutcome::Failed { error, .. }) => error.clone(),
        other => panic!("failed: {other:?}"),
    }
}

async fn no_model_fails_the_run(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let change = AgentChange {
        model: Change::Set(ModelRef::new(PROVIDER, "unknown")),
        ..AgentChange::default()
    };
    conversation.configure(change, &ctx()).await.unwrap();
    let state = submit(&conversation, "hi").await;
    assert_eq!(unanswered(&state), ("no_model", None));
    assert!(fixture.model.requests().is_empty());
    // Without any model at all, too.
    let change = AgentChange {
        model: Change::Clear,
        ..AgentChange::default()
    };
    conversation.configure(change, &ctx()).await.unwrap();
    let state = submit(&conversation, "again").await;
    assert_eq!(unanswered(&state).0, "no_model");
    assert_eq!(kinds(&entries(&conversation).await), ["delta.user", "delta.user"]);
    harness.close(&ctx()).await.unwrap();
}

async fn a_model_error_fails_the_run(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture
        .model
        .push([Reply::error("invalid request: bad schema").usage(3, 0)]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let state = submit(&conversation, "hi").await;
    let (reason, detail) = unanswered(&state);
    assert_eq!(reason, "model_error");
    assert_eq!(detail.and_then(Json::as_str), Some("invalid request: bad schema"));
    let error = failure_of(&harness, &conversation).await;
    assert_eq!(error.message, "invalid request: bad schema");
    assert_eq!(error.detail, Some(delta_chord::json!({ "reason": "model_error" })));
    // The failed attempt is in the transcript and its spend counts.
    let transcript = entries(&conversation).await;
    assert_eq!(kinds(&transcript), ["delta.user", "delta.system", "delta.assistant"]);
    let usage = harness.usage(&ctx()).await.unwrap();
    assert_eq!(usage.models.values().next().unwrap().input, 3);
    harness.close(&ctx()).await.unwrap();
}

async fn a_retryable_error_sleeps_and_retries(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture.model.push([Reply::retryable(), Reply::text("Recovered")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let before = fixture.now();
    let state = submit(&conversation, "hi").await;
    assert!(matches!(state, SubmissionState::Done { .. }));
    // The default backoff is 2 s on the Harness clock.
    assert!(fixture.now() - before >= 2_000);
    let transcript = entries(&conversation).await;
    assert_eq!(
        kinds(&transcript),
        ["delta.user", "delta.system", "delta.assistant", "delta.assistant"]
    );
    // One generation made both attempts; the failed one is out of the second request.
    assert_eq!(transcript[2].by_task_id, transcript[3].by_task_id);
    let requests = fixture.model.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].messages.len(), 2);
    harness.close(&ctx()).await.unwrap();
}

async fn retries_stop_at_the_policy_limit(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture.settings(|settings| {
        // One-millisecond backoffs: the sleeps fall due around the scheduler's reservation
        // pass, which once left them without a timer.
        settings.retry = Some(PartialRetryPolicy {
            max_retries: Some(2),
            base_delay: Some(Duration::from_millis(1)),
            ..PartialRetryPolicy::default()
        });
    });
    fixture.model.push([
        Reply::retryable(),
        Reply::retryable(),
        Reply::retryable(),
        Reply::text("never"),
    ]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let state = submit(&conversation, "hi").await;
    assert_eq!(unanswered(&state).0, "model_error");
    assert_eq!(fixture.model.requests().len(), 3);
    harness.close(&ctx()).await.unwrap();
}

async fn an_overflow_without_a_cut_fails(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture.model.push([Reply::overflow()]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let state = submit(&conversation, "hi").await;
    assert_eq!(unanswered(&state).0, "model_error");
    assert_eq!(fixture.model.requests().len(), 1);
    harness.close(&ctx()).await.unwrap();
}
