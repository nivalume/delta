//! Structured waits: `all_settled` and `fail_fast`, the wait-for cycle check, waits held in
//! memory and their end at close.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use delta_chord::json::Json;
use delta_harness::records::{JoinPolicy, TaskId, TaskOutcome};
use delta_harness::scheduler::{Next, NextState, RuntimeError, SchedulerError, StateError, TaskError};
use delta_harness::session::TaskOptions;

use crate::{Backend, Env, Gate, Probe, ctx, erased, eventually, phase};

cases!(
    all_settled_resumes_once_every_task_is_terminal,
    fail_fast_aborts_the_live_siblings_not_the_waiter,
    rejects_waits_that_cannot_be_kept,
    runtime_wait_for_task_reads_a_receipt,
    caller_cancellation_ends_only_the_wait,
    waits_end_at_close,
);

/// A child that ends with the outcome its input names: `complete`, `fail`, or waits for its
/// context to be cancelled.
fn child() -> Arc<Probe> {
    Probe::new("test.child")
        .run(|rt, _| {
            Box::pin(async move {
                match rt.input().as_str() {
                    Some("complete") => Ok(Next::complete(Json::from("done"))),
                    Some("fail") => Ok(Next::fail("expired")),
                    _ => {
                        rt.context().cancelled().await;
                        Err(TaskError::new("cancelled"))
                    }
                }
            })
        })
        .arc()
}

fn status(outcome: &TaskOutcome) -> &'static str {
    match outcome {
        TaskOutcome::Completed { .. } => "completed",
        TaskOutcome::Failed { .. } => "failed",
        TaskOutcome::Aborted { .. } => "aborted",
        TaskOutcome::Orphaned { .. } => "orphaned",
        TaskOutcome::Faulted { .. } => "faulted",
    }
}

/// A parent that creates one child per input entry, waits on them under `policy` and
/// completes with their outcomes' statuses.
fn parent(child: &Arc<Probe>, policy: JoinPolicy) -> Arc<Probe> {
    let child = child.clone();
    Probe::new("test.parent")
        .run(move |rt, checkpoint| {
            let child = child.clone();
            Box::pin(async move {
                if checkpoint == phase("start") {
                    let inputs: Vec<Json> = rt.input().as_array().map(|items| items.to_vec()).unwrap_or_default();
                    return Ok(rt.next(move |tx, current| {
                        Box::pin(async move {
                            let mut on = Vec::new();
                            for input in inputs {
                                let id = tx
                                    .create_task(&*child, input, TaskOptions::owned_by(current.record.id))
                                    .await?;
                                on.push(id.id());
                            }
                            let ids: Vec<Json> = on.iter().map(|id| Json::from(id.0)).collect();
                            Ok(NextState::Waiting {
                                checkpoint: delta_chord::json!({ "phase": "read", "on": ids }),
                                on,
                                policy,
                            })
                        })
                    }));
                }
                let on: Vec<TaskId> = checkpoint
                    .get("on")
                    .and_then(Json::as_array)
                    .map(|ids| ids.iter().filter_map(Json::as_u64).map(TaskId).collect())
                    .unwrap_or_default();
                let outcomes = rt.outcomes(&on, &ctx()).await?;
                let statuses: Vec<Json> = outcomes.iter().map(|outcome| Json::from(status(outcome))).collect();
                Ok(Next::complete(Json::from(statuses)))
            })
        })
        .arc()
}

async fn all_settled_resumes_once_every_task_is_terminal(backend: Backend) {
    let child = child();
    let parent = parent(&child, JoinPolicy::AllSettled);
    let env = Env::new(backend, [erased(&child), erased(&parent)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let input = delta_chord::json!(["complete", "fail", "complete"]);
    let id = harness
        .create(&parent, input, TaskOptions::conversation(conversation))
        .await;
    let settled = harness.settled(id).await;
    assert_eq!(
        settled.result::<Vec<String>>().unwrap().unwrap(),
        ["completed", "failed", "completed"]
    );
    harness.close().await;
}

async fn fail_fast_aborts_the_live_siblings_not_the_waiter(backend: Backend) {
    let child = child();
    let parent = parent(&child, JoinPolicy::FailFast);
    let env = Env::new(backend, [erased(&child), erased(&parent)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let input = delta_chord::json!(["wait", "fail", "wait", "wait"]);
    let id = harness
        .create(&parent, input, TaskOptions::conversation(conversation))
        .await;
    let settled = harness.settled(id).await;
    assert_eq!(
        settled.result::<Vec<String>>().unwrap().unwrap(),
        ["aborted", "failed", "aborted", "aborted"]
    );
    assert!(!settled.record().abort_requested, "the waiter itself is not marked");
    harness.close().await;
}

async fn rejects_waits_that_cannot_be_kept(backend: Backend) {
    let errors = Arc::new(Mutex::new(Vec::<StateError>::new()));
    let gate = Gate::new();
    let (kept, held) = (errors.clone(), gate.clone());
    // The child tries waits from its own phase: on itself, its owner (which holds it), a
    // missing task, and fail_fast on a task it does not own.
    let child = Probe::new("test.waiter")
        .run(move |rt, _| {
            let kept = kept.clone();
            Box::pin(async move {
                let me = rt.task_id();
                let owner = rt.task().record.owner.unwrap();
                let attempts = [
                    (vec![me], JoinPolicy::AllSettled),
                    (vec![owner], JoinPolicy::AllSettled),
                    (vec![TaskId(9_999)], JoinPolicy::AllSettled),
                    (vec![owner], JoinPolicy::FailFast),
                ];
                for (on, policy) in attempts {
                    let wait = NextState::Waiting {
                        checkpoint: phase("after"),
                        on,
                        policy,
                    };
                    let refused = rt
                        .commit(&ctx(), move |_, _| Box::pin(async move { Ok(((), Some(wait))) }))
                        .await
                        .unwrap_err();
                    match refused {
                        RuntimeError::State(error) => kept.lock().unwrap().push(error),
                        other => panic!("expected a refused state, got {other:?}"),
                    }
                }
                Ok(Next::complete(Json::Null))
            })
        })
        .arc();
    let spawn = child.clone();
    let owner = Probe::new("test.holder")
        .run(move |rt, checkpoint| {
            let (spawn, held) = (spawn.clone(), held.clone());
            Box::pin(async move {
                if checkpoint == phase("start") {
                    return Ok(rt.next(move |tx, current| {
                        Box::pin(async move {
                            tx.create_task(&*spawn, Json::Null, TaskOptions::owned_by(current.record.id))
                                .await?;
                            Ok(NextState::Running(phase("hold")))
                        })
                    }));
                }
                held.wait().await;
                Ok(Next::complete(Json::Null))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&child), erased(&owner)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&owner, conversation).await;
    eventually(|| errors.lock().unwrap().len() == 4).await;
    let errors = errors.lock().unwrap().clone();
    let child_id = match &errors[0] {
        StateError::Cycle { task, on, path } => {
            assert_eq!(task, on);
            assert_eq!(path, &[*task, *task]);
            *task
        }
        other => panic!("expected a cycle, got {other:?}"),
    };
    match &errors[1] {
        StateError::Cycle { task, on, path } => {
            assert_eq!((*task, *on), (child_id, id));
            assert_eq!(path, &[child_id, id, child_id]);
            assert_eq!(
                errors[1].to_string(),
                format!("Task {child_id} cannot wait on {id}: {child_id} -> {id} -> {child_id} is a wait cycle")
            );
        }
        other => panic!("expected a cycle through the holding owner, got {other:?}"),
    }
    assert_eq!(errors[2], StateError::UnknownTask(TaskId(9_999)));
    assert_eq!(errors[3], StateError::FailFastNotOwned { task: child_id, on: id });
    gate.open();
    harness.settled(id).await;
    harness.close().await;
}

async fn runtime_wait_for_task_reads_a_receipt(backend: Backend) {
    let worker = Probe::new("test.worker")
        .run(|_, _| Box::pin(async { Ok(Next::complete(Json::from(7))) }))
        .arc();
    let spawn = worker.clone();
    // An abort-free supervisor: starts background work in its conversation and awaits it in memory.
    let supervisor = Probe::new("test.supervisor")
        .run(move |rt, checkpoint| {
            let spawn = spawn.clone();
            Box::pin(async move {
                if checkpoint == phase("start") {
                    return Ok(rt.next(move |tx, _| {
                        Box::pin(async move {
                            let options = TaskOptions {
                                background: true,
                                ..TaskOptions::default()
                            };
                            let id = tx.create_task(&*spawn, Json::Null, options).await?;
                            Ok(NextState::Running(
                                delta_chord::json!({ "phase": "await", "id": id.id().0 }),
                            ))
                        })
                    }));
                }
                let id = TaskId(checkpoint.get("id").and_then(Json::as_u64).unwrap_or_default());
                let receipt = rt.wait_for_task(id, &ctx()).await?;
                let value: i64 = receipt.result().unwrap_or(Ok(0))?;
                Ok(Next::complete(Json::from(value * 6)))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&worker), erased(&supervisor)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&supervisor, conversation).await;
    assert_eq!(harness.settled(id).await.result::<i64>().unwrap().unwrap(), 42);
    harness.close().await;
}

async fn caller_cancellation_ends_only_the_wait(backend: Backend) {
    let gate = Gate::new();
    let held = gate.clone();
    let slow = Probe::new("test.slow")
        .run(move |_, _| {
            let gate = held.clone();
            Box::pin(async move {
                gate.wait().await;
                Ok(Next::complete(Json::Null))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&slow)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&slow, conversation).await;
    let (cancelled, canceller) = ctx().with_cancel();
    let wait = harness.scheduler.wait_for_task(id, &cancelled);
    canceller.cancel();
    assert!(matches!(wait.await, Err(SchedulerError::Cancelled(_))));
    gate.open();
    assert!(matches!(
        harness.settled(id).await.outcome(),
        TaskOutcome::Completed { .. }
    ));
    harness.close().await;
}

async fn waits_end_at_close(backend: Backend) {
    let blocked = Probe::new("test.absent").arc();
    let env = Env::new(backend, []).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&blocked, conversation).await;
    let scheduler = harness.scheduler.clone();
    let task_wait = tokio::spawn(async move { scheduler.wait_for_task(id, &ctx()).await });
    let scheduler = harness.scheduler.clone();
    let idle_wait = tokio::spawn(async move { scheduler.wait_for_idle(None, &ctx()).await });
    tokio::time::sleep(Duration::from_millis(20)).await;
    harness.close().await;
    assert!(matches!(task_wait.await.unwrap(), Err(SchedulerError::Closed)));
    assert!(matches!(idle_wait.await.unwrap(), Err(SchedulerError::Closed)));
}
