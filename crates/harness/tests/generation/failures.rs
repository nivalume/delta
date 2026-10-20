//! Failed attempts: `no_model`, model errors, retries with backoff, context overflow and the
//! compaction thresholds (through the compaction seam).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use delta_ai::types::{Message, StopReason};
use delta_chord::context::Context;
use delta_chord::json::Json;
use delta_harness::agent::{AgentChange, Change, CompactionSettings, PartialCompactionPolicy, PartialRetryPolicy};
use delta_harness::compaction::{CompactionInput, CompactionReason, Compactor};
use delta_harness::context::ContextView;
use delta_harness::harness::Conversation;
use delta_harness::inbox::SubmissionDraft;
use delta_harness::model::ModelRef;
use delta_harness::records::{
    ConversationId, EntryDraft, Head, SubmissionState, TaskId, TaskOutcome, TaskOutcomeError,
};
use delta_harness::registry::Extension;
use delta_harness::scheduler::{Next, NextState, Outcome, TaskDefinition, TaskError, TaskRuntime};
use delta_harness::session::{TaskOptions, Tx, TxError};
use futures::future::BoxFuture;

use crate::Backend;
use crate::support::fixture::{Fixture, chat, ctx, entries, kinds, settled};
use crate::support::model::{PROVIDER, Reply};

cases!(
    no_model_fails_the_run,
    a_model_error_fails_the_run,
    retries_stop_at_the_policy_limit,
    an_overflow_without_a_cut_fails,
    an_overflow_compacts_and_requests_again,
    a_failed_overflow_compaction_fails_the_run,
    the_blocking_threshold_compacts_before_the_request,
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
        // Backoffs of a few milliseconds: shorter sleeps can fall due between the scheduler's
        // reservation pass and its timer computation (see the scheduler's `drive`).
        settings.retry = Some(PartialRetryPolicy {
            max_retries: Some(2),
            base_delay: Some(Duration::from_millis(20)),
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

// ─── A compaction stand-in ──────────────────────────────────────────────────

/// What the stand-in compaction task does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Summarize {
    /// Appends a summary that heads at itself.
    Summary,
    Fail,
}

/// A `delta.compaction` stand-in: summarizes the conversation into one user message.
struct TestCompaction(Summarize);

impl TaskDefinition for TestCompaction {
    type Input = CompactionInput;
    type Checkpoint = Json;
    type Result = Json;

    fn name(&self) -> &'static str {
        "delta.compaction"
    }

    fn version(&self) -> u32 {
        1
    }

    fn initial(&self, _input: &CompactionInput) -> Json {
        delta_chord::json!({ "phase": "select" })
    }

    async fn run<'a>(
        &'a self,
        _phase: Json,
        rt: &'a TaskRuntime<Self>,
        _ctx: &'a Context,
    ) -> Result<Next<Self>, TaskError> {
        if self.0 == Summarize::Fail {
            return Ok(Next::fail("the summarizer failed"));
        }
        let conversation = rt.conversation_id();
        Ok(rt.next(move |tx, _| {
            Box::pin(async move {
                let draft = EntryDraft {
                    model: Some(vec![Message::user("Summary of earlier work")]),
                    head: Some(Head::SelfRef),
                    ..EntryDraft::new("test.summary")
                };
                let entry = tx.append_entry(conversation, draft).await?;
                Ok(NextState::Terminal(Outcome::Completed(
                    delta_chord::json!({ "entry_id": entry.id.0 }),
                )))
            })
        }))
    }

    async fn abort<'a>(
        &'a self,
        _phase: Json,
        _rt: &'a TaskRuntime<Self>,
        _ctx: &'a Context,
    ) -> Result<Next<Self>, TaskError> {
        Ok(Next::abort(None))
    }
}

/// A compactor that always finds a cut and records what it was asked to create.
#[derive(Default)]
struct TestCompactor {
    created: Mutex<Vec<(CompactionReason, Option<TaskId>)>>,
}

impl Compactor for TestCompactor {
    fn select_cut(&self, view: &ContextView, _keep_recent_tokens: u64) -> Option<usize> {
        (!view.entries.is_empty()).then_some(view.entries.len() - 1)
    }

    fn create<'a>(
        &'a self,
        tx: &'a mut Tx<'_>,
        conversation: ConversationId,
        input: CompactionInput,
        owner: Option<TaskId>,
    ) -> BoxFuture<'a, Result<TaskId, TxError>> {
        self.created.lock().unwrap().push((input.reason, owner));
        Box::pin(async move {
            let options = match owner {
                Some(owner) => TaskOptions::owned_by(owner),
                None => TaskOptions {
                    background: input.reason != CompactionReason::Manual,
                    ..TaskOptions::conversation(conversation)
                },
            };
            let summarize = Summarize::Summary;
            Ok(tx.create_task(&TestCompaction(summarize), input, options).await?.id())
        })
    }
}

async fn compacting(backend: Backend, summarize: Summarize) -> (Fixture, Arc<TestCompactor>) {
    let mut fixture = Fixture::new(backend).await;
    let compactor = Arc::new(TestCompactor::default());
    fixture.compactor = compactor.clone();
    let task = Arc::new(TestCompaction(summarize));
    fixture
        .registry
        .install(Extension::new("compaction").with_task(task))
        .unwrap();
    (fixture, compactor)
}

async fn an_overflow_compacts_and_requests_again(backend: Backend) {
    let (fixture, compactor) = compacting(backend, Summarize::Summary).await;
    fixture.model.push([Reply::overflow(), Reply::text("Fits now")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let state = submit(&conversation, "hi").await;
    assert!(matches!(state, SubmissionState::Done { .. }), "{state:?}");
    let created = compactor.created.lock().unwrap().clone();
    assert_eq!(created.len(), 1);
    assert_eq!(created[0].0, CompactionReason::Overflow);
    let transcript = entries(&conversation).await;
    let generation = transcript[2].by_task_id;
    assert_eq!(created[0].1, generation);
    // The retried request starts at the summary, with a new baseline.
    let requests = fixture.model.requests();
    assert_eq!(requests.len(), 2);
    let second: Vec<&str> = requests[1].messages.iter().map(Message::role).collect();
    assert_eq!(second, ["system", "user"]);
    let Message::User(summary) = &requests[1].messages[1] else {
        panic!("a summary")
    };
    assert_eq!(summary.content.text(), "Summary of earlier work");
    let Some([Message::Assistant(failed)]) = transcript[2].model.as_deref() else {
        panic!("the overflow entry")
    };
    assert_eq!(failed.stop_reason, StopReason::Error);
    harness.close(&ctx()).await.unwrap();
}

async fn a_failed_overflow_compaction_fails_the_run(backend: Backend) {
    // The installed definition decides what the created task does.
    let (fixture, _compactor) = compacting(backend, Summarize::Fail).await;
    fixture.model.push([Reply::overflow()]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let state = submit(&conversation, "hi").await;
    let (reason, detail) = unanswered(&state);
    assert_eq!(reason, "model_error");
    assert!(
        detail
            .and_then(Json::as_str)
            .is_some_and(|text| text.contains("prompt is too long"))
    );
    assert_eq!(fixture.model.requests().len(), 1);
    harness.close(&ctx()).await.unwrap();
}

async fn the_blocking_threshold_compacts_before_the_request(backend: Backend) {
    let (fixture, compactor) = compacting(backend, Summarize::Summary).await;
    fixture.settings(|settings| {
        settings.compaction = Some(CompactionSettings {
            policy: PartialCompactionPolicy {
                reserve_tokens: Some(crate::support::fixture::CONTEXT_WINDOW - 2),
                ..PartialCompactionPolicy::default()
            },
            ..CompactionSettings::default()
        });
    });
    fixture.model.push([Reply::text("After compaction")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let state = submit(&conversation, "a long question that crosses the threshold").await;
    assert!(matches!(state, SubmissionState::Done { .. }), "{state:?}");
    let created = compactor.created.lock().unwrap().clone();
    assert_eq!(created.len(), 1);
    assert_eq!(created[0].0, CompactionReason::Threshold);
    assert!(
        created[0].1.is_some(),
        "a blocking compaction is owned by the generation"
    );
    let requests = fixture.model.requests();
    assert_eq!(requests.len(), 1);
    let Message::User(summary) = &requests[0].messages[1] else {
        panic!("a summary")
    };
    assert_eq!(summary.content.text(), "Summary of earlier work");
    harness.close(&ctx()).await.unwrap();
}
