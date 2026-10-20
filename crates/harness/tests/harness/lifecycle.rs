//! Lifecycle: opening paused, resuming, close, inspection, task operations and reports.

use std::sync::Arc;
use std::time::Duration;

use delta_harness::harness::HarnessError;
use delta_harness::inbox::SubmissionDraft;
use delta_harness::records::{SubmissionState, SubmissionStatus, TaskOutcome};
use delta_harness::registry::{Extension, Wrap};
use delta_harness::scheduler::{InspectState, Scheduling, TaskAbort};
use delta_harness::session::ReportInfo;
use tokio::sync::Notify;

use crate::Backend;
use crate::support::fixture::{Fixture, chat, ctx, eventually, settled};
use crate::support::model::Reply;
use crate::support::tools::tool;

cases!(
    a_reopened_harness_waits_paused_until_resumed,
    inspection_lists_live_tasks_and_open_submissions,
    task_operations_reach_the_scheduler,
    operations_after_close_fail_closed,
    a_failing_wrapper_is_reported_with_its_extension,
);

async fn a_reopened_harness_waits_paused_until_resumed(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let gate = Arc::new(Notify::new());
    fixture.model.push([Reply::text("interrupted").gate(&gate)]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("hi"), &ctx()).await.unwrap();
    let id = submission.id();
    eventually(|| async { fixture.model.requests().len() == 1 }).await;
    // Close is not a crash: the running generation is written back as pending.
    harness.close(&ctx()).await.unwrap();

    fixture.model.push([Reply::text("resumed")]);
    let harness = fixture.open().await;
    let inspection = harness.inspect(&ctx()).await.unwrap();
    assert_eq!(inspection.scheduler.scheduling, Scheduling::Paused);
    assert_eq!(inspection.scheduler.tasks.len(), 1);
    assert!(matches!(
        inspection.scheduler.tasks[0].state,
        InspectState::Ready { .. }
    ));
    assert_eq!(inspection.scheduler.tasks[0].record.restarts, None);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(fixture.model.requests().len(), 1);
    harness.resume().unwrap();
    let submission = harness.submission(id, &ctx()).await.unwrap().unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    assert_eq!(fixture.model.requests().len(), 2);
    harness.close(&ctx()).await.unwrap();
}

async fn inspection_lists_live_tasks_and_open_submissions(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let gate = Arc::new(Notify::new());
    fixture.model.push([Reply::text("one").gate(&gate), Reply::text("two")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let first = conversation
        .submit(SubmissionDraft::input("one"), &ctx())
        .await
        .unwrap();
    let second = conversation
        .submit(SubmissionDraft::input("two"), &ctx())
        .await
        .unwrap();
    let inspection = harness.inspect(&ctx()).await.unwrap();
    let open: Vec<_> = inspection
        .submissions
        .iter()
        .map(|submission| (submission.id, submission.state.status()))
        .collect();
    assert_eq!(
        open,
        [
            (first.id(), SubmissionStatus::Placed),
            (second.id(), SubmissionStatus::Queued)
        ]
    );
    assert_eq!(inspection.scheduler.scheduling, Scheduling::Running);
    assert_eq!(inspection.scheduler.tasks.len(), 1);
    assert_eq!(inspection.scheduler.tasks[0].record.kind, "delta.generation");
    gate.notify_one();
    settled(&second).await;
    harness.wait_for_idle(&ctx()).await.unwrap();
    let inspection = harness.inspect(&ctx()).await.unwrap();
    assert!(inspection.submissions.is_empty() && inspection.scheduler.tasks.is_empty());
    harness.close(&ctx()).await.unwrap();
}

async fn task_operations_reach_the_scheduler(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let gate = Arc::new(Notify::new());
    fixture.model.push([Reply::text("never").gate(&gate)]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("hi"), &ctx()).await.unwrap();
    eventually(|| async { fixture.model.requests().len() == 1 }).await;
    let task = harness.inspect(&ctx()).await.unwrap().scheduler.tasks[0].record.id;
    assert_eq!(harness.abort_task(task, &ctx()).await.unwrap(), TaskAbort::Marked);
    let settled_task = harness.wait_for_task(task, &ctx()).await.unwrap();
    assert!(matches!(settled_task.outcome(), TaskOutcome::Aborted { .. }));
    assert_eq!(harness.abort_task(task, &ctx()).await.unwrap(), TaskAbort::Terminal);
    let state = settled(&submission).await.state;
    assert!(matches!(&state, SubmissionState::Unanswered { reason, .. } if reason == "aborted"));
    harness.retry_task(task, &ctx()).await.unwrap();
    let unknown = harness.abort_task(delta_harness::records::TaskId(9_999), &ctx()).await;
    assert!(matches!(unknown, Err(HarnessError::UnknownTask(_))));
    harness.close(&ctx()).await.unwrap();
}

async fn operations_after_close_fail_closed(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    harness.close(&ctx()).await.unwrap();
    assert!(matches!(harness.resume(), Err(HarnessError::Closed)));
    let refused = conversation.submit(SubmissionDraft::input("late"), &ctx()).await;
    assert!(matches!(refused, Err(HarnessError::Closed)));
    assert!(matches!(harness.wait_for_idle(&ctx()).await, Err(HarnessError::Closed)));
    // Closing twice waits for the same close.
    harness.close(&ctx()).await.unwrap();
}

async fn a_failing_wrapper_is_reported_with_its_extension(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture
        .registry
        .install(Extension::new("tools").with_tool(tool("read")))
        .unwrap();
    fixture
        .registry
        .install(Extension::new("broken").with_wrap(Wrap::Tool {
            name: "read".to_owned(),
            wrap: Arc::new(|_| Err("cannot wrap".into())),
        }))
        .unwrap();
    fixture.model.push([Reply::text("no tools")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let agent = conversation.agent(&ctx()).await.unwrap();
    assert!(agent.tools.is_empty());
    let submission = conversation.submit(SubmissionDraft::input("hi"), &ctx()).await.unwrap();
    settled(&submission).await;
    let reports = fixture.reported();
    let extension_reports: Vec<_> = reports
        .iter()
        .filter_map(|(error, info)| match info {
            ReportInfo::ExtensionError { extension, task, .. } => Some((
                error.as_str(),
                extension.as_deref(),
                task.as_ref().map(|task| task.kind.clone()),
            )),
            _ => None,
        })
        .collect();
    // Once from the host's resolution, once from the generation's phase.
    assert!(extension_reports.contains(&("cannot wrap", Some("broken"), None)));
    assert!(extension_reports.contains(&("cannot wrap", Some("broken"), Some("delta.generation".to_owned()))));
    harness.close(&ctx()).await.unwrap();
}
