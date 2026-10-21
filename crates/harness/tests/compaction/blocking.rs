//! Compactions the generation owns and waits for: above the blocking threshold, and after a
//! context overflow.

use delta_ai::model::CacheRetention;
use delta_ai::types::Message;
use delta_harness::compaction::{CompactionResult, SUMMARIZATION_SYSTEM_PROMPT};
use delta_harness::records::{SubmissionState, TaskOutcome};

use crate::support::fixture::{CONTEXT_WINDOW, Fixture, chat, ctx, live};
use crate::support::model::Reply;
use crate::{Backend, history, keep_little, run, summaries, summary_requests, user_text};

cases!(
    an_overflow_compacts_and_requests_again,
    the_blocking_threshold_compacts_before_the_request,
    a_failed_summary_fails_the_overflowing_run,
    a_second_overflow_fails_the_run,
);

const WRAPPED: &str = "The conversation history before this point was compacted into the following summary:\n\n<summary>\nThe user asked a first question.\n</summary>";

async fn an_overflow_compacts_and_requests_again(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    keep_little(&fixture, |_| {});
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    history(&fixture, &conversation).await;
    fixture.model.push([Reply::overflow(), Reply::text("Fits now")]);
    fixture
        .model
        .push_summaries([Reply::text("The user asked a first question.")]);
    // Long enough to be the kept window by itself: the cut is at this question.
    let state = run(&conversation, &"Second question. ".repeat(20)).await;
    assert!(matches!(state, SubmissionState::Done { .. }), "{state:?}");

    let summaries = summaries(&conversation).await;
    assert_eq!(summaries.len(), 1);
    let summary = &summaries[0];
    assert_eq!(user_text(summary), WRAPPED);
    assert_eq!(
        summary
            .data
            .as_ref()
            .and_then(|data| data.get("reason"))
            .and_then(delta_chord::json::Json::as_str),
        Some("overflow")
    );
    // A blocking compaction is owned by the generation and appends directly.
    let task = harness
        .get_task(summary.by_task_id.unwrap(), &ctx())
        .await
        .unwrap()
        .unwrap();
    assert!(task.owner.is_some());
    let Some(TaskOutcome::Completed { result }) = task.state.outcome() else {
        panic!("completed: {:?}", task.state)
    };
    let result: CompactionResult = delta_chord::json::from_json(result).unwrap();
    assert_eq!((result.entry_id, result.submission_id), (Some(summary.id), None));

    // The summarizer read a transcript, pinned and uncached.
    let requests = summary_requests(&fixture);
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(
        request.max_tokens,
        Some(1000),
        "the model's limit is below 80% of the reserve"
    );
    assert_eq!(request.options.cache_retention, Some(CacheRetention::None));
    assert!(request.session_id.is_some());
    let [Message::System(system), Message::User(prompt)] = request.messages.as_slice() else {
        panic!("a system and a user message: {:?}", request.messages)
    };
    assert_eq!(system.content.text(), SUMMARIZATION_SYSTEM_PROMPT);
    let prompt = prompt.content.text();
    assert!(prompt.starts_with("<conversation>\n[User]: First question\n\n[Assistant]: An answer"));
    assert!(!prompt.contains("Second question"));

    // The retried request starts at the summary.
    let last = fixture.model.requests().last().unwrap().messages.clone();
    let roles: Vec<&str> = last.iter().map(Message::role).collect();
    assert_eq!(roles, ["system", "user", "user"]);
    let Message::User(first) = &last[1] else {
        panic!("the summary")
    };
    assert_eq!(first.content.text(), WRAPPED);
    let state = live(&harness, &conversation).await;
    assert!(state.compactions.is_none() && state.compaction_failure.is_none());
    harness.close(&ctx()).await.unwrap();
}

async fn the_blocking_threshold_compacts_before_the_request(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    keep_little(&fixture, |policy| policy.reserve_tokens = Some(CONTEXT_WINDOW - 2));
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    // The first run is over the threshold too, but has nothing before a cut.
    history(&fixture, &conversation).await;
    assert!(summaries(&conversation).await.is_empty());
    fixture.model.push([Reply::text("After compaction")]);
    fixture
        .model
        .push_summaries([Reply::text("The user asked a first question.")]);
    let state = run(&conversation, "Second question").await;
    assert!(matches!(state, SubmissionState::Done { .. }), "{state:?}");
    let summaries = summaries(&conversation).await;
    assert_eq!(summaries.len(), 1);
    assert_eq!(
        summaries[0]
            .data
            .as_ref()
            .and_then(|data| data.get("reason"))
            .and_then(delta_chord::json::Json::as_str),
        Some("threshold")
    );
    // Only one generation request in the second run, after the summary.
    let generation: Vec<_> = fixture
        .model
        .requests()
        .into_iter()
        .filter(|request| request.max_tokens.is_none())
        .collect();
    assert_eq!(generation.len(), 2);
    let Message::User(first) = &generation[1].messages[1] else {
        panic!("the summary")
    };
    assert_eq!(first.content.text(), WRAPPED);
    harness.close(&ctx()).await.unwrap();
}

async fn a_failed_summary_fails_the_overflowing_run(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    keep_little(&fixture, |_| {});
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    history(&fixture, &conversation).await;
    fixture.model.push([Reply::overflow()]);
    fixture.model.push_summaries([Reply::error("invalid request")]);
    let state = run(&conversation, "Second question").await;
    let SubmissionState::Unanswered { reason, detail, .. } = &state else {
        panic!("unanswered: {state:?}")
    };
    assert_eq!(reason, "model_error");
    assert!(
        detail
            .as_ref()
            .and_then(delta_chord::json::Json::as_str)
            .is_some_and(|text| text.contains("prompt is too long"))
    );
    assert!(summaries(&conversation).await.is_empty());
    let failure = live(&harness, &conversation).await.compaction_failure.unwrap();
    assert_eq!(failure.error, "Summarization failed: invalid request");
    let task = harness.get_task(failure.task_id, &ctx()).await.unwrap().unwrap();
    let Some(TaskOutcome::Failed { error, .. }) = task.state.outcome() else {
        panic!("failed")
    };
    assert_eq!(error.detail, Some(delta_chord::json!({ "reason": "model_error" })));
    harness.close(&ctx()).await.unwrap();
}

async fn a_second_overflow_fails_the_run(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    keep_little(&fixture, |_| {});
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    history(&fixture, &conversation).await;
    fixture.model.push([Reply::overflow(), Reply::overflow()]);
    fixture
        .model
        .push_summaries([Reply::text("The user asked a first question.")]);
    let state = run(&conversation, "Second question").await;
    assert!(
        matches!(&state, SubmissionState::Unanswered { reason, .. } if reason == "model_error"),
        "{state:?}"
    );
    assert_eq!(summaries(&conversation).await.len(), 1);
    assert_eq!(summary_requests(&fixture).len(), 1);
    harness.close(&ctx()).await.unwrap();
}
