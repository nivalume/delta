//! Ownership: `completing` holds, abort cascades (bottom-up, background boundaries,
//! failed owners), orphaning, withdrawn inputs and marks a crash left unapplied.

use std::sync::Arc;
use std::time::Duration;

use delta_chord::json::Json;
use delta_harness::records::{
    ConversationOwnership, SubmissionCreate, SubmissionState, SubmissionType, TaskOutcome, TaskRecord, TaskState,
    TaskStatus,
};
use delta_harness::scheduler::{Next, NextState, TaskAbort, TaskError};
use delta_harness::session::{SessionError, TaskOptions, TxError};

use crate::{Backend, Env, Gate, Log, Probe, ctx, erased, eventually, phase};

cases!(
    holds_a_finished_owner_until_its_work_drains,
    aborts_bottom_up_across_three_levels,
    background_tasks_are_boundaries,
    cascades_a_failed_owner_but_not_a_completed_one,
    orphans_a_blocked_task_on_abort,
    cascade_withdraws_queued_inputs_below_the_owner,
    rejects_owned_work_below_a_marked_owner,
    lists_owned_tasks,
    derives_marks_a_crash_left_unapplied,
    a_finishing_commit_cannot_create_a_child,
    a_running_task_aborts_its_owned_child_and_continues,
);

/// A task that waits at `gate` (or, without one, until its context is cancelled) and
/// completes with its name; its abort handler logs and ends aborted.
fn worker(name: &str, gate: Option<Gate>, log: &Log) -> Arc<Probe> {
    let label = name.to_owned();
    let aborted = log.clone();
    Probe::new(name)
        .run(move |rt, _| {
            let gate = gate.clone();
            let label = label.clone();
            Box::pin(async move {
                match gate {
                    Some(gate) => gate.wait().await,
                    None => rt.context().cancelled().await,
                }
                Ok(Next::complete(Json::from(label.as_str())))
            })
        })
        .on_abort(move |rt, _| {
            let log = aborted.clone();
            Box::pin(async move {
                log.push(format!("abort {}", rt.task().record.kind));
                Ok(Next::abort(None))
            })
        })
        .arc()
}

/// A task that creates a child of `child` in its first phase, then behaves as `worker`
/// in its second, or ends with `end` there.
fn owner(name: &str, child: &Arc<Probe>, end: Option<NextState<Json, Json>>, log: &Log) -> Arc<Probe> {
    let child = child.clone();
    let aborted = log.clone();
    Probe::new(name)
        .run(move |rt, checkpoint| {
            let child = child.clone();
            let end = end.clone();
            Box::pin(async move {
                if checkpoint == phase("start") {
                    return Ok(rt.next(move |tx, current| {
                        Box::pin(async move {
                            tx.create_task(&*child, Json::Null, TaskOptions::owned_by(current.record.id))
                                .await?;
                            Ok(NextState::Running(phase("own")))
                        })
                    }));
                }
                match end {
                    Some(end) => Ok(Next::state(end)),
                    None => {
                        rt.context().cancelled().await;
                        Err(TaskError::new("cancelled"))
                    }
                }
            })
        })
        .on_abort(move |rt, _| {
            let log = aborted.clone();
            Box::pin(async move {
                log.push(format!("abort {}", rt.task().record.kind));
                Ok(Next::abort(None))
            })
        })
        .arc()
}

/// The live record of the first task of `kind`.
async fn live_of(env: &Env, kind: &str) -> Option<TaskRecord> {
    let query = delta_harness::records::TaskQuery {
        kind: Some(kind.to_owned()),
        ..Default::default()
    };
    env.storage
        .scan_tasks(query, 10, None, &ctx())
        .await
        .unwrap()
        .items
        .into_iter()
        .next()
}

async fn holds_a_finished_owner_until_its_work_drains(backend: Backend) {
    let log = Log::default();
    let gate = Gate::new();
    let child = worker("test.child", Some(gate.clone()), &log);
    let parent = owner(
        "test.parent",
        &child,
        Some(NextState::completed(Json::from("parent"))),
        &log,
    );
    let env = Env::new(backend, [erased(&child), erased(&parent)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&parent, conversation).await;
    let held = harness
        .until(id, |record| record.state.status() == TaskStatus::Completing)
        .await;
    assert!(matches!(
        held.state,
        TaskState::Completing {
            outcome: TaskOutcome::Completed { .. }
        }
    ));
    let background = ctx();
    let wait = harness.scheduler.wait_for_task(id, &background);
    assert!(tokio::time::timeout(Duration::from_millis(30), wait).await.is_err());
    gate.open();
    let settled = harness.settled(id).await;
    assert_eq!(settled.result::<String>().unwrap().unwrap(), "parent");
    let child_record = live_of(&env, "test.child").await.unwrap();
    assert_eq!(child_record.owner, Some(id));
    assert_eq!(child_record.state.status(), TaskStatus::Terminal);
    assert!(log.lines().is_empty());
    harness.close().await;
}

async fn aborts_bottom_up_across_three_levels(backend: Backend) {
    let log = Log::default();
    let leaf = worker("test.leaf", None, &log);
    let middle = owner("test.middle", &leaf, None, &log);
    let top = owner("test.top", &middle, None, &log);
    let env = Env::new(backend, [erased(&leaf), erased(&middle), erased(&top)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&top, conversation).await;
    for kind in ["test.middle", "test.leaf"] {
        for _ in 0..2000 {
            if live_of(&env, kind)
                .await
                .is_some_and(|record| record.state.status() == TaskStatus::Running)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }
    assert_eq!(
        harness.scheduler.abort_task(id, &ctx()).await.unwrap(),
        TaskAbort::Marked
    );
    let settled = harness.settled(id).await;
    assert!(matches!(settled.outcome(), TaskOutcome::Aborted { .. }));
    assert_eq!(log.lines(), ["abort test.leaf", "abort test.middle", "abort test.top"]);
    harness.close().await;
}

async fn background_tasks_are_boundaries(backend: Backend) {
    let log = Log::default();
    let below = worker("test.below", None, &log);
    let supervisor = owner("test.supervisor", &below, None, &log);
    let foreground = worker("test.foreground", None, &log);
    let env = Env::new(backend, [erased(&below), erased(&supervisor), erased(&foreground)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let background = harness
        .create(
            &supervisor,
            Json::Null,
            TaskOptions {
                background: true,
                ..TaskOptions::conversation(conversation)
            },
        )
        .await;
    let fore = harness.start(&foreground, conversation).await;
    for _ in 0..2000 {
        if live_of(&env, "test.below")
            .await
            .is_some_and(|record| record.state.status() == TaskStatus::Running)
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    harness
        .scheduler
        .abort_conversation(conversation, false, &ctx())
        .await
        .unwrap();
    assert!(matches!(
        harness.settled(fore).await.outcome(),
        TaskOutcome::Aborted { .. }
    ));
    assert_eq!(log.lines(), ["abort test.foreground"]);
    assert!(
        !harness.record(background).await.abort_requested,
        "a background task is a boundary"
    );
    harness
        .scheduler
        .abort_conversation(conversation, true, &ctx())
        .await
        .unwrap();
    assert!(matches!(
        harness.settled(background).await.outcome(),
        TaskOutcome::Aborted { .. }
    ));
    assert_eq!(
        log.lines(),
        ["abort test.foreground", "abort test.below", "abort test.supervisor"]
    );
    harness.close().await;
}

async fn cascades_a_failed_owner_but_not_a_completed_one(backend: Backend) {
    let log = Log::default();
    let gate = Gate::new();
    let doomed = worker("test.doomed", None, &log);
    let survivor = worker("test.survivor", Some(gate.clone()), &log);
    let failing = owner("test.failing", &doomed, Some(NextState::failed("no")), &log);
    let completing = owner(
        "test.completing",
        &survivor,
        Some(NextState::completed(Json::Null)),
        &log,
    );
    let env = Env::new(
        backend,
        [
            erased(&doomed),
            erased(&survivor),
            erased(&failing),
            erased(&completing),
        ],
    )
    .await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let failed = harness.start(&failing, conversation).await;
    let completed = harness.start(&completing, conversation).await;
    // The held failure is cancellation intent: the child is aborted, then the owner ends failed.
    match harness.settled(failed).await.outcome() {
        TaskOutcome::Failed { error, .. } => assert_eq!(error.message, "no"),
        other => panic!("expected failed, got {other:?}"),
    }
    assert_eq!(log.lines(), ["abort test.doomed"]);
    // A held completion waits for its work to finish normally.
    harness
        .until(completed, |record| record.state.status() == TaskStatus::Completing)
        .await;
    gate.open();
    harness.settled(completed).await;
    let survivor = live_of(&env, "test.survivor").await.unwrap();
    assert!(matches!(
        survivor.state,
        TaskState::Terminal {
            outcome: TaskOutcome::Completed { .. }
        }
    ));
    assert_eq!(log.lines(), ["abort test.doomed"]);
    harness.close().await;
}

async fn orphans_a_blocked_task_on_abort(backend: Backend) {
    let missing = Probe::new("test.gone").arc();
    let env = Env::new(backend, []).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&missing, conversation).await;
    assert_eq!(
        harness.scheduler.abort_task(id, &ctx()).await.unwrap(),
        TaskAbort::Marked
    );
    match harness.settled(id).await.outcome() {
        TaskOutcome::Orphaned { reason } => assert_eq!(reason, "missing_task"),
        other => panic!("expected orphaned, got {other:?}"),
    }
    harness.close().await;
}

async fn cascade_withdraws_queued_inputs_below_the_owner(backend: Backend) {
    let log = Log::default();
    let parent = Probe::new("test.subagent")
        .run(|rt, checkpoint| {
            Box::pin(async move {
                if checkpoint == phase("start") {
                    return Ok(rt.next(|tx, current| {
                        Box::pin(async move {
                            let owned = tx
                                .create_conversation(ConversationOwnership::Task(current.record.id))
                                .await?;
                            let submission = tx.create_submission(queued_input(owned.id)).await?;
                            let data = delta_chord::json!({ "submission": submission.id });
                            Ok(NextState::Running(delta_chord::json!({ "phase": "own", "data": data })))
                        })
                    }));
                }
                rt.context().cancelled().await;
                Err(TaskError::new("cancelled"))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&parent)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let own_queue = harness
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move { tx.create_submission(queued_input(conversation)).await })
        })
        .await
        .unwrap();
    let id = harness.start(&parent, conversation).await;
    let record = harness
        .until(id, |record| {
            record
                .state
                .checkpoint()
                .is_some_and(|checkpoint| checkpoint.get("data").is_some())
        })
        .await;
    let found = record
        .state
        .checkpoint()
        .and_then(|checkpoint| checkpoint.get("data")?.get("submission"));
    let below = delta_harness::records::SubmissionId(found.and_then(Json::as_u64).unwrap());
    harness.scheduler.abort_task(id, &ctx()).await.unwrap();
    harness.settled(id).await;
    let mut withdrawn = None;
    for _ in 0..2000 {
        let record = env.storage.submission(below, &ctx()).await.unwrap().unwrap();
        if !matches!(record.state, SubmissionState::Queued) {
            withdrawn = Some(record);
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    match withdrawn.expect("the input below is withdrawn").state {
        SubmissionState::Unanswered { reason, .. } => assert_eq!(reason, "aborted"),
        other => panic!("expected unanswered, got {other:?}"),
    }
    // The cancelled task's own conversation keeps its queue.
    let own = env.storage.submission(own_queue.id, &ctx()).await.unwrap().unwrap();
    assert!(matches!(own.state, SubmissionState::Queued));
    assert!(log.lines().is_empty());
    harness.close().await;
}

fn queued_input(conversation: delta_harness::records::ConversationId) -> SubmissionCreate {
    SubmissionCreate {
        conversation_id: conversation,
        request_id: None,
        kind: SubmissionType::Input,
        state: SubmissionState::Queued,
    }
}

async fn rejects_owned_work_below_a_marked_owner(backend: Backend) {
    let probe = Probe::new("test.owner").arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.open().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    assert_eq!(
        harness.scheduler.abort_task(id, &ctx()).await.unwrap(),
        TaskAbort::Marked
    );
    let child = probe.clone();
    let refused = harness
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move { tx.create_task(&*child, Json::Null, TaskOptions::owned_by(id)).await })
        })
        .await
        .unwrap_err();
    assert!(
        matches!(
            refused,
            SessionError::Tx(TxError::OwnerUnavailable {
                status: "abort-marked",
                ..
            })
        ),
        "{refused:?}"
    );
    harness.scheduler.resume().unwrap();
    assert!(matches!(
        harness.settled(id).await.outcome(),
        TaskOutcome::Aborted { .. }
    ));
    harness.close().await;
}

async fn lists_owned_tasks(backend: Backend) {
    let log = Log::default();
    let gate = Gate::new();
    let child = worker("test.listed", Some(gate.clone()), &log);
    let listed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let (spawn, seen) = (child.clone(), listed.clone());
    let parent = Probe::new("test.lister")
        .run(move |rt, checkpoint| {
            let (spawn, seen) = (spawn.clone(), seen.clone());
            Box::pin(async move {
                if checkpoint == phase("start") {
                    return Ok(rt.next(move |tx, current| {
                        Box::pin(async move {
                            for _ in 0..2 {
                                tx.create_task(&*spawn, Json::Null, TaskOptions::owned_by(current.record.id))
                                    .await?;
                            }
                            Ok(NextState::Running(phase("list")))
                        })
                    }));
                }
                let owned = rt.owned_tasks(&crate::ctx()).await?;
                seen.lock().unwrap().extend(owned.iter().map(|record| record.owner));
                Ok(Next::complete(Json::Null))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&child), erased(&parent)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&parent, conversation).await;
    eventually(|| listed.lock().unwrap().len() == 2).await;
    assert_eq!(*listed.lock().unwrap(), [Some(id), Some(id)]);
    gate.open();
    harness.settled(id).await;
    harness.close().await;
}

async fn derives_marks_a_crash_left_unapplied(backend: Backend) {
    let log = Log::default();
    let child = worker("test.unmarked", None, &log);
    let parent = worker("test.marked", None, &log);
    let env = Env::new(backend, [erased(&child), erased(&parent)]).await;
    let harness = env.open().await;
    let conversation = harness.conversation().await;
    let parent_id = harness.start(&parent, conversation).await;
    let child_id = harness
        .create(&child, Json::Null, TaskOptions::owned_by(parent_id))
        .await;
    harness.crash().await;
    // A crash after the owner's mark, before the cascade commit.
    let mut marked = env.stored(parent_id).await;
    marked.abort_requested = true;
    env.write(marked).await;
    let harness = env.open().await;
    // Open re-derives the cascade without resume.
    harness.until(child_id, |record| record.abort_requested).await;
    harness.scheduler.resume().unwrap();
    harness.settled(parent_id).await;
    assert_eq!(log.lines(), ["abort test.unmarked", "abort test.marked"]);
    harness.close().await;
}

async fn a_finishing_commit_cannot_create_a_child(backend: Backend) {
    let log = Log::default();
    let child = worker("test.unborn", None, &log);
    let spawn = child.clone();
    let parent = Probe::new("test.finisher")
        .run(move |rt, _| {
            let spawn = spawn.clone();
            Box::pin(async move {
                Ok(rt.next(move |tx, current| {
                    Box::pin(async move {
                        tx.create_task(&*spawn, Json::Null, TaskOptions::owned_by(current.record.id))
                            .await?;
                        Ok(NextState::completed(Json::Null))
                    })
                }))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&child), erased(&parent)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&parent, conversation).await;
    match harness.settled(id).await.outcome() {
        TaskOutcome::Faulted { error } => assert_eq!(error.message, format!("Task owner {id} is completing")),
        other => panic!("expected a fault, got {other:?}"),
    }
    assert!(
        live_of(&env, "test.unborn").await.is_none(),
        "the refused commit rolled back"
    );
    assert!(harness.session.failure().is_none());
    harness.close().await;
}

async fn a_running_task_aborts_its_owned_child_and_continues(backend: Backend) {
    let log = Log::default();
    let child = worker("test.helper", None, &log);
    let spawn = child.clone();
    let parent = Probe::new("test.manager")
        .run(move |rt, checkpoint| {
            let spawn = spawn.clone();
            Box::pin(async move {
                if checkpoint == phase("start") {
                    return Ok(rt.next(move |tx, current| {
                        Box::pin(async move {
                            let id = tx
                                .create_task(&*spawn, Json::Null, TaskOptions::owned_by(current.record.id))
                                .await?;
                            Ok(NextState::Running(
                                delta_chord::json!({ "phase": "cancel", "child": id.id().0 }),
                            ))
                        })
                    }));
                }
                let child = delta_harness::records::TaskId(checkpoint.get("child").and_then(Json::as_u64).unwrap_or(0));
                rt.abort_owned(child, &crate::ctx()).await?;
                let outcomes = rt.outcomes(&[child], &crate::ctx()).await?;
                let aborted = matches!(outcomes.first(), Some(TaskOutcome::Aborted { .. }));
                Ok(Next::complete(Json::Bool(aborted)))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&child), erased(&parent)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&parent, conversation).await;
    let settled = harness.settled(id).await;
    assert!(
        settled.result::<bool>().unwrap().unwrap(),
        "the child ended aborted first"
    );
    assert!(!settled.record().abort_requested);
    assert_eq!(log.lines(), ["abort test.helper"]);
    harness.close().await;
}
