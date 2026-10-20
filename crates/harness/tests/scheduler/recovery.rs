//! Recovery and close: reruns after a crash, the restart count and parking, migration,
//! blocked tasks, handover, abandonment, and close's cancel, join, grace and write-back.

use std::time::Duration;

use delta_chord::json::Json;
use delta_harness::documents::MigrateError;
use delta_harness::records::{AbortReason, TaskOutcome, TaskState, TaskStatus};
use delta_harness::scheduler::{BlockedReason, InspectState, Next, SchedulerSettings, TaskError};
use delta_harness::session::{FaultReason, ReportInfo, TaskOptions};

use crate::{Backend, Count, Env, Gate, Harness, Probe, ctx, erased, eventually, phase};

cases!(
    crash_reruns_the_current_phase,
    close_writes_back_pending_without_a_count,
    close_drops_handlers_that_ignore_their_context,
    parks_at_the_restart_limit_until_retry,
    faults_an_abort_handler_cut_off_too_often,
    progress_clears_the_restart_count,
    migrates_older_versions_at_reservation,
    failed_migration_blocks_until_another_definition,
    blocks_a_record_newer_than_its_definition,
    hands_over_to_a_replacement_definition,
    abandons_marked_work_after_a_restart,
    a_sleep_survives_a_crash,
);

async fn inspect_state(harness: &Harness, id: delta_harness::records::TaskId) -> InspectState {
    let inspection = harness.scheduler.inspect(&ctx()).await.unwrap();
    let task = inspection.tasks.into_iter().find(|task| task.record.id == id).unwrap();
    task.state
}

/// An effect phase that waits forever on its first run (a crash cuts it off) and completes
/// on its next.
fn effect(runs: &Count) -> std::sync::Arc<Probe> {
    let runs = runs.clone();
    Probe::new("test.effect")
        .run(move |_, checkpoint| {
            let runs = runs.clone();
            Box::pin(async move {
                if checkpoint == phase("start") {
                    return Ok(Next::running(phase("effect")));
                }
                if runs.hit() == 1 {
                    std::future::pending::<()>().await;
                }
                Ok(Next::complete(Json::from("done")))
            })
        })
        .arc()
}

async fn crash_reruns_the_current_phase(backend: Backend) {
    let runs = Count::default();
    let probe = effect(&runs);
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    eventually(|| runs.get() == 1).await;
    harness.crash().await;
    let crashed = env.stored(id).await;
    assert_eq!(
        crashed.state,
        TaskState::Running {
            checkpoint: phase("effect")
        }
    );
    let harness = env.open().await;
    let reopened = harness.record(id).await;
    assert_eq!(
        reopened.state,
        TaskState::Pending {
            checkpoint: phase("effect")
        }
    );
    assert_eq!(reopened.restarts, Some(1));
    assert_eq!(runs.get(), 1, "open runs no handler");
    harness.scheduler.resume().unwrap();
    let settled = harness.settled(id).await;
    assert_eq!(settled.result::<String>().unwrap().unwrap(), "done");
    assert_eq!(runs.get(), 2);
    harness.close().await;
}

async fn close_writes_back_pending_without_a_count(backend: Backend) {
    let started = Count::default();
    let seen = started.clone();
    let probe = Probe::new("test.cooperative")
        .run(move |rt, _| {
            let seen = seen.clone();
            Box::pin(async move {
                seen.hit();
                rt.context().cancelled().await;
                Err(TaskError::new("stopped"))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    eventually(|| started.get() == 1).await;
    harness.close().await;
    let stored = env.stored(id).await;
    assert_eq!(
        stored.state,
        TaskState::Pending {
            checkpoint: phase("start")
        }
    );
    assert_eq!(stored.restarts, None, "a close is not a crash");
    assert!(env.reported().is_empty(), "close writes no fault: {:?}", env.reported());
    let harness = env.run().await;
    eventually(|| started.get() == 2).await;
    harness.close().await;
}

async fn close_drops_handlers_that_ignore_their_context(backend: Backend) {
    let started = Count::default();
    let finished = Count::default();
    let (seen, done) = (started.clone(), finished.clone());
    let probe = Probe::new("test.stubborn")
        .run(move |_, _| {
            let (seen, done) = (seen.clone(), done.clone());
            Box::pin(async move {
                seen.hit();
                tokio::time::sleep(Duration::from_hours(1)).await;
                done.hit();
                Ok(Next::complete(Json::Null))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    env.set_settings(SchedulerSettings {
        close_grace: Duration::from_millis(50),
        ..SchedulerSettings::default()
    });
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    eventually(|| started.get() == 1).await;
    tokio::time::timeout(Duration::from_secs(5), harness.close())
        .await
        .expect("close ends after the grace");
    assert_eq!(finished.get(), 0);
    let stored = env.stored(id).await;
    assert_eq!(
        stored.state,
        TaskState::Pending {
            checkpoint: phase("start")
        }
    );
    assert_eq!(stored.restarts, None);
}

async fn parks_at_the_restart_limit_until_retry(backend: Backend) {
    let probe = Probe::new("test.parked").arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.open().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    harness.crash().await;
    let mut cut_off = env.stored(id).await;
    cut_off.state = TaskState::Running {
        checkpoint: phase("start"),
    };
    cut_off.restarts = Some(4);
    env.write(cut_off).await;
    let harness = env.run().await;
    assert_eq!(harness.record(id).await.restarts, Some(5));
    let parked = env
        .reported()
        .into_iter()
        .filter(|(_, info)| matches!(info, ReportInfo::TaskParked { task } if task.id == id))
        .count();
    assert_eq!(parked, 1);
    assert_eq!(
        inspect_state(&harness, id).await,
        InspectState::Blocked {
            reason: BlockedReason::RestartLimit
        }
    );
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(harness.record(id).await.state.status(), TaskStatus::Pending);
    harness.scheduler.retry_task(id, &ctx()).await.unwrap();
    assert!(matches!(
        harness.settled(id).await.outcome(),
        TaskOutcome::Completed { .. }
    ));
    harness.close().await;
}

async fn faults_an_abort_handler_cut_off_too_often(backend: Backend) {
    let probe = Probe::new("test.cut-off").arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.open().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    harness.crash().await;
    let mut cut_off = env.stored(id).await;
    cut_off.state = TaskState::Running {
        checkpoint: phase("start"),
    };
    cut_off.abort_requested = true;
    cut_off.restarts = Some(4);
    env.write(cut_off).await;
    let harness = env.run().await;
    match harness.settled(id).await.outcome() {
        TaskOutcome::Faulted { error } => {
            assert_eq!(error.detail, Some(delta_chord::json!({ "reason": "restart_limit" })));
        }
        other => panic!("expected a fault, got {other:?}"),
    }
    eventually(|| {
        env.reported().iter().any(|(_, info)| {
            matches!(
                info,
                ReportInfo::TaskFaulted {
                    reason: FaultReason::RestartLimit,
                    ..
                }
            )
        })
    })
    .await;
    harness.close().await;
}

async fn progress_clears_the_restart_count(backend: Backend) {
    let gate = Gate::new();
    let held = gate.clone();
    let probe = Probe::new("test.progress")
        .run(move |_, checkpoint| {
            let gate = held.clone();
            Box::pin(async move {
                if checkpoint == phase("start") {
                    return Ok(Next::running(phase("next")));
                }
                gate.wait().await;
                Ok(Next::complete(Json::Null))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.open().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    harness.crash().await;
    let mut cut_off = env.stored(id).await;
    cut_off.state = TaskState::Running {
        checkpoint: phase("start"),
    };
    cut_off.restarts = Some(2);
    env.write(cut_off).await;
    let harness = env.run().await;
    let progressed = harness
        .until(id, |record| record.state.checkpoint() == Some(&phase("next")))
        .await;
    assert_eq!(progressed.restarts, None);
    gate.open();
    harness.settled(id).await;
    harness.close().await;
}

async fn migrates_older_versions_at_reservation(backend: Backend) {
    let old = Probe::new("test.versioned").arc();
    let new = Probe::new("test.versioned")
        .version(2)
        .migrate(|input, checkpoint, from| {
            assert_eq!(from, 1);
            let n = input.get("n").and_then(Json::as_i64).unwrap_or_default();
            Ok((delta_chord::json!({ "n": n + 1 }), checkpoint))
        })
        .run(|rt, _| {
            let n = rt.input().get("n").and_then(Json::as_i64).unwrap_or_default();
            Box::pin(async move { Ok(Next::complete(Json::from(n))) })
        })
        .arc();
    let env = Env::new(backend, [erased(&new)]).await;
    let harness = env.open().await;
    let conversation = harness.conversation().await;
    let id = harness
        .create(
            &old,
            delta_chord::json!({ "n": 1 }),
            TaskOptions::conversation(conversation),
        )
        .await;
    assert_eq!(
        inspect_state(&harness, id).await,
        InspectState::Ready { migrates: true }
    );
    harness.scheduler.resume().unwrap();
    let settled = harness.settled(id).await;
    assert_eq!(settled.result::<i64>().unwrap().unwrap(), 2);
    assert_eq!(settled.record().version, 2);
    assert_eq!(settled.record().input, delta_chord::json!({ "n": 2 }));
    harness.close().await;
}

async fn failed_migration_blocks_until_another_definition(backend: Backend) {
    let old = Probe::new("test.migrating").arc();
    let failing = Probe::new("test.migrating")
        .version(2)
        .migrate(|_, _, _| Err(MigrateError::failed("cannot")))
        .arc();
    let working = Probe::new("test.migrating")
        .version(2)
        .migrate(|input, checkpoint, _| Ok((input, checkpoint)))
        .arc();
    let env = Env::new(backend, [erased(&failing)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness
        .create(&old, Json::Null, TaskOptions::conversation(conversation))
        .await;
    harness
        .until(id, |_| {
            env.reported()
                .iter()
                .any(|(_, info)| matches!(info, ReportInfo::MigrationFailed { .. }))
        })
        .await;
    assert_eq!(
        inspect_state(&harness, id).await,
        InspectState::Blocked {
            reason: BlockedReason::MigrationFailed
        }
    );
    // Retried only for another definition object: more passes report nothing new.
    harness.scheduler.resume().unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    let failures = env
        .reported()
        .iter()
        .filter(|(_, info)| matches!(info, ReportInfo::MigrationFailed { .. }))
        .count();
    assert_eq!(failures, 1);
    assert_eq!(
        env.stored(id).await.version,
        1,
        "a failed migration leaves the record unchanged"
    );
    env.catalog.install(erased(&working));
    let settled = harness.settled(id).await;
    assert!(matches!(settled.outcome(), TaskOutcome::Completed { .. }));
    assert_eq!(settled.record().version, 2);
    harness.close().await;
}

async fn blocks_a_record_newer_than_its_definition(backend: Backend) {
    let newer = Probe::new("test.newer").version(3).arc();
    let older = Probe::new("test.newer").version(2).arc();
    let env = Env::new(backend, [erased(&older)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness
        .create(&newer, Json::Null, TaskOptions::conversation(conversation))
        .await;
    assert_eq!(
        inspect_state(&harness, id).await,
        InspectState::Blocked {
            reason: BlockedReason::TaskTooOld
        }
    );
    env.catalog.install(erased(&newer));
    assert!(matches!(
        harness.settled(id).await.outcome(),
        TaskOutcome::Completed { .. }
    ));
    harness.close().await;
}

async fn hands_over_to_a_replacement_definition(backend: Backend) {
    let gate = Gate::new();
    let held = gate.clone();
    let original = Probe::new("test.handover")
        .run(move |_, checkpoint| {
            let gate = held.clone();
            Box::pin(async move {
                if checkpoint == phase("start") {
                    gate.wait().await;
                    return Ok(Next::running(phase("next")));
                }
                Ok(Next::complete(Json::from("original")))
            })
        })
        .arc();
    let replacement = Probe::new("test.handover")
        .run(|_, _| Box::pin(async { Ok(Next::complete(Json::from("replacement"))) }))
        .arc();
    let env = Env::new(backend, [erased(&original)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&original, conversation).await;
    harness
        .until(id, |record| record.state.status() == TaskStatus::Running)
        .await;
    env.catalog.install(erased(&replacement));
    gate.open();
    let settled = harness.settled(id).await;
    assert_eq!(settled.result::<String>().unwrap().unwrap(), "replacement");
    harness.close().await;
}

async fn abandons_marked_work_after_a_restart(backend: Backend) {
    let reasons = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let kept = reasons.clone();
    let probe = Probe::new("test.abandoned")
        .run(|_, _| Box::pin(async { std::future::pending().await }))
        .on_abort(move |rt, _| {
            let kept = kept.clone();
            Box::pin(async move {
                kept.lock().unwrap().push(rt.task().record.abort_reason);
                Ok(Next::abort(Some("abandoned".to_owned())))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let options = TaskOptions {
        abandon_on_restart: true,
        ..TaskOptions::conversation(conversation)
    };
    let id = harness.create(&probe, Json::Null, options).await;
    harness
        .until(id, |record| record.state.status() == TaskStatus::Running)
        .await;
    harness.crash().await;
    // Opening alone marks nothing.
    let harness = env.open().await;
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(!harness.record(id).await.abort_requested);
    harness.scheduler.resume().unwrap();
    assert!(matches!(
        harness.settled(id).await.outcome(),
        TaskOutcome::Aborted { .. }
    ));
    assert_eq!(*reasons.lock().unwrap(), [Some(AbortReason::Restart)]);
    harness.close().await;
}

async fn a_sleep_survives_a_crash(backend: Backend) {
    let probe = Probe::new("test.napper")
        .run(|rt, checkpoint| {
            Box::pin(async move {
                if checkpoint == phase("start") {
                    return Ok(Next::sleep(rt.now() + 150, phase("after")));
                }
                Ok(Next::complete(Json::from(rt.now())))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    let asleep = harness
        .until(id, |record| matches!(record.state, TaskState::Sleeping { .. }))
        .await;
    harness.crash().await;
    let harness = env.open().await;
    assert_eq!(harness.record(id).await.state, asleep.state, "open keeps a sleep");
    harness.scheduler.resume().unwrap();
    let TaskState::Sleeping { until, .. } = asleep.state else {
        unreachable!()
    };
    let woke = harness.settled(id).await.result::<i64>().unwrap().unwrap();
    assert!(woke >= until);
    harness.close().await;
}
