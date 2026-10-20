//! Inspection derives every live task's state without running task code.

use delta_chord::json::Json;
use delta_harness::records::{JoinPolicy, TaskId, TaskStatus};
use delta_harness::scheduler::{BlockedReason, InspectState, Inspection, Next, NextState, Scheduling};
use delta_harness::session::TaskOptions;

use crate::{Backend, Count, Env, Gate, Probe, ctx, erased, phase};

cases!(derives_every_live_task_state, keeps_index_sizes_bounded_by_live_work);

fn state_of(inspection: &Inspection, id: TaskId) -> InspectState {
    inspection
        .tasks
        .iter()
        .find(|task| task.record.id == id)
        .map(|task| task.state.clone())
        .unwrap()
}

async fn derives_every_live_task_state(backend: Backend) {
    let gate = Gate::new();
    let held = gate.clone();
    let blocking = Probe::new("test.blocking")
        .run(move |_, _| {
            let gate = held.clone();
            Box::pin(async move {
                gate.wait().await;
                Ok(Next::complete(Json::Null))
            })
        })
        .arc();
    let calls = Count::default();
    let migrations = calls.clone();
    let migrating_old = Probe::new("test.migrates").arc();
    let migrating = Probe::new("test.migrates")
        .version(2)
        .migrate(move |input, checkpoint, _| {
            migrations.hit();
            Ok((input, checkpoint))
        })
        .arc();
    let missing = Probe::new("test.missing").arc();
    let sleeper = Probe::new("test.sleeps")
        .run(|rt, _| Box::pin(async move { Ok(Next::sleep(rt.now() + 3_600_000, phase("after"))) }))
        .arc();
    let waiter = Probe::new("test.waits")
        .run(|rt, checkpoint| {
            Box::pin(async move {
                if checkpoint == phase("start") {
                    let on = rt.input().as_u64().map(TaskId).into_iter().collect();
                    return Ok(Next::wait(on, JoinPolicy::AllSettled, phase("after")));
                }
                Ok(Next::complete(Json::Null))
            })
        })
        .arc();
    let spawn = blocking.clone();
    // Creates a child, then completes in its next phase (not in the creating commit: an
    // owner that finishes in a commit cannot own what that commit creates).
    let holder = Probe::new("test.holds")
        .run(move |rt, checkpoint| {
            let spawn = spawn.clone();
            Box::pin(async move {
                if checkpoint != phase("start") {
                    return Ok(Next::complete(Json::Null));
                }
                Ok(rt.next(move |tx, current| {
                    Box::pin(async move {
                        tx.create_task(&*spawn, Json::Null, TaskOptions::owned_by(current.record.id))
                            .await?;
                        Ok(NextState::Running(phase("finish")))
                    })
                }))
            })
        })
        .arc();
    let env = Env::new(
        backend,
        [
            erased(&blocking),
            erased(&migrating),
            erased(&sleeper),
            erased(&waiter),
            erased(&holder),
        ],
    )
    .await;
    let harness = env.open().await;
    let conversation = harness.conversation().await;
    let ready = harness.start(&blocking, conversation).await;
    let migrates = harness.start(&migrating_old, conversation).await;
    let blocked = harness.start(&missing, conversation).await;
    let paused = harness.scheduler.inspect(&ctx()).await.unwrap();
    assert_eq!(paused.scheduling, Scheduling::Paused);
    assert_eq!(state_of(&paused, ready), InspectState::Ready { migrates: false });
    assert_eq!(state_of(&paused, migrates), InspectState::Ready { migrates: true });
    assert_eq!(
        state_of(&paused, blocked),
        InspectState::Blocked {
            reason: BlockedReason::MissingTask
        }
    );
    assert_eq!(calls.get(), 0, "inspection runs no task code");
    harness.scheduler.resume().unwrap();
    let sleeping = harness.start(&sleeper, conversation).await;
    let waiting = harness
        .create(&waiter, Json::from(ready.0), TaskOptions::conversation(conversation))
        .await;
    let holding = harness.start(&holder, conversation).await;
    harness
        .until(holding, |record| record.state.status() == TaskStatus::Completing)
        .await;
    harness
        .until(waiting, |record| record.state.status() == TaskStatus::Waiting)
        .await;
    harness
        .until(sleeping, |record| record.state.status() == TaskStatus::Sleeping)
        .await;
    let running = harness.scheduler.inspect(&ctx()).await.unwrap();
    assert_eq!(running.scheduling, Scheduling::Running);
    assert_eq!(state_of(&running, ready), InspectState::Running);
    assert_eq!(state_of(&running, waiting), InspectState::Waiting { on: vec![ready] });
    assert!(matches!(state_of(&running, sleeping), InspectState::Sleeping { .. }));
    assert_eq!(state_of(&running, holding), InspectState::Completing);
    assert!(matches!(
        state_of(&running, blocked),
        InspectState::Blocked {
            reason: BlockedReason::MissingTask
        }
    ));
    gate.open();
    harness.settled(holding).await;
    harness.settled(waiting).await;
    harness.close().await;
}

async fn keeps_index_sizes_bounded_by_live_work(backend: Backend) {
    let leaf = Probe::new("test.leaf").arc();
    let spawn = leaf.clone();
    let parent = Probe::new("test.branch")
        .run(move |rt, checkpoint| {
            let spawn = spawn.clone();
            Box::pin(async move {
                if checkpoint != phase("start") {
                    return Ok(Next::complete(Json::Null));
                }
                Ok(rt.next(move |tx, current| {
                    Box::pin(async move {
                        for _ in 0..3 {
                            tx.create_task(&*spawn, Json::Null, TaskOptions::owned_by(current.record.id))
                                .await?;
                        }
                        Ok(NextState::Running(phase("finish")))
                    })
                }))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&leaf), erased(&parent)]).await;
    let harness = env.run().await;
    for _ in 0..3 {
        let conversation = harness.conversation().await;
        let id = harness.start(&parent, conversation).await;
        harness.settled(id).await;
    }
    harness.scheduler.wait_for_idle(None, &ctx()).await.unwrap();
    // Nothing live: the indexes keep nothing but what a later sweep drops.
    harness.conversation().await;
    let sizes = harness.scheduler.index_sizes();
    assert_eq!(
        (sizes.live, sizes.below, sizes.roots, sizes.unloaded),
        (0, 0, 0, 0),
        "{sizes:?}"
    );
    assert!(sizes.settled <= 1 && sizes.edges <= 1, "{sizes:?}");
    harness.close().await;
}
