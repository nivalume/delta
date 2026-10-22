//! A crash after the intent commit, and an abort while a call runs.

use std::sync::{Arc, Mutex};

use delta_ai::types::Tool;
use delta_harness::harness::{AbortOptions, Conversation, Harness};
use delta_harness::inbox::SubmissionDraft;
use delta_harness::records::{SubmissionState, TaskOutcome};
use delta_harness::registry::Extension;
use delta_harness::tool::{FnHarnessTool, HarnessTool, ToolExecutionResult, ToolReplay};

use crate::support::fixture::{Fixture, PATIENCE, chat, ctx, entries, eventually, live, settled};
use crate::support::model::Reply;
use crate::{Backend, results, texts};

cases!(
    a_safe_call_runs_again_after_a_crash,
    an_unsafe_call_settles_interrupted_after_a_crash,
    an_abort_settles_the_running_call_with_its_output,
);

/// A tool that reports progress, then waits until its context ends on every execution but
/// the last `finish_after`th; it counts its executions.
fn stuck(name: &str, replay: ToolReplay, runs: &Arc<Mutex<u32>>, finish_after: u32) -> Arc<dyn HarnessTool> {
    let runs = runs.clone();
    let tool = FnHarnessTool::new(
        Tool::new(name, "Works for a while", serde_json::json!({ "type": "object" })),
        move |_arguments, api, ctx| {
            let run = {
                let mut runs = runs.lock().unwrap();
                *runs += 1;
                *runs
            };
            Box::pin(async move {
                if run >= finish_after {
                    return Ok(ToolExecutionResult::text(format!("finished on run {run}")));
                }
                api.output_str(&format!("step 1 of run {run}\n"));
                api.details(serde_json::json!({ "step": 1 }), ctx).await?;
                ctx.run(std::future::pending::<()>()).await?;
                Ok(ToolExecutionResult::default())
            })
        },
    )
    .with_replay(replay);
    Arc::new(tool)
}

/// Waits until the call's first step is committed in its slot.
async fn step_published(harness: &Harness, conversation: &Conversation) {
    eventually(|| async {
        live(harness, conversation)
            .await
            .tools
            .into_iter()
            .flatten()
            .any(|slot| slot.progress.details.is_some())
    })
    .await;
}

/// Drops every handle without closing, as a crash.
async fn crash(harness: Harness) {
    let closed = harness.session().closed();
    drop(harness);
    tokio::time::timeout(PATIENCE, closed)
        .await
        .expect("the old Session ends");
}

async fn a_safe_call_runs_again_after_a_crash(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    fixture
        .registry
        .install(Extension::new("tools").with_tool(stuck("fetch", ToolReplay::Safe, &runs, 2)))
        .unwrap();
    fixture.model.push([Reply::calls(&[("c1", "fetch")])]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    let id = submission.id();
    step_published(&harness, &conversation).await;
    drop((submission, conversation));
    crash(harness).await;

    fixture.model.push([Reply::text("Fetched")]);
    let harness = fixture.open().await;
    let submission = harness.submission(id, &ctx()).await.unwrap().unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    let conversation = chat(&harness).await;
    let results = results(&conversation).await;
    assert_eq!(texts(&results[0]), ["finished on run 2"]);
    assert!(!results[0].is_error);
    // The rerun reported from scratch: the first run's details are gone.
    assert_eq!(results[0].details, None);
    assert_eq!(*runs.lock().unwrap(), 2);
    harness.close(&ctx()).await.unwrap();
}

async fn an_unsafe_call_settles_interrupted_after_a_crash(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    fixture
        .registry
        .install(Extension::new("tools").with_tool(stuck("write", ToolReplay::Unsafe, &runs, 2)))
        .unwrap();
    fixture.model.push([Reply::calls(&[("c1", "write")])]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    let id = submission.id();
    step_published(&harness, &conversation).await;
    drop((submission, conversation));
    crash(harness).await;

    fixture.model.push([Reply::text("Recovered")]);
    let harness = fixture.open().await;
    let submission = harness.submission(id, &ctx()).await.unwrap().unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    let conversation = chat(&harness).await;
    let results = results(&conversation).await;
    assert!(results[0].is_error);
    assert_eq!(
        texts(&results[0]),
        [
            "step 1 of run 1\n",
            "<harness>\n[error] Tool write was interrupted and may have partially run\n</harness>"
        ]
    );
    assert_eq!(results[0].details, Some(serde_json::json!({ "step": 1 })));
    assert_eq!(*runs.lock().unwrap(), 1, "an unsafe call never runs twice");
    let transcript = entries(&conversation).await;
    let entry = transcript
        .iter()
        .find(|entry| entry.kind == "delta.tool-result")
        .unwrap();
    let task = harness
        .get_task(entry.by_task_id.unwrap(), &ctx())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.state.outcome(), Some(TaskOutcome::Failed { .. })));
    harness.close(&ctx()).await.unwrap();
}

async fn an_abort_settles_the_running_call_with_its_output(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    fixture
        .registry
        .install(Extension::new("tools").with_tool(stuck("build", ToolReplay::Safe, &runs, 9)))
        .unwrap();
    fixture.model.push([Reply::calls(&[("c1", "build")])]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    step_published(&harness, &conversation).await;
    tokio::time::timeout(PATIENCE, conversation.abort(AbortOptions::default(), &ctx()))
        .await
        .unwrap()
        .unwrap();
    let state = settled(&submission).await.state;
    assert!(matches!(&state, SubmissionState::Unanswered { reason, .. } if reason == "aborted"));
    let results = results(&conversation).await;
    assert_eq!(
        texts(&results[0]),
        [
            "step 1 of run 1\n",
            "<harness>\n[error] Tool build was aborted\n</harness>"
        ]
    );
    let state = live(&harness, &conversation).await;
    assert!(state.tools.is_none() && state.run.is_none());
    harness.close(&ctx()).await.unwrap();
}
