//! Durable sleeps, the sleep timer, wakes and runtime timers, on tokio's paused clock.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use delta_chord::json::Json;
use delta_harness::records::{TaskOutcome, TaskState};
use delta_harness::scheduler::{Next, NextState, RuntimeError, StateError, Wake};

use crate::{Backend, Env, Probe, ctx, erased, phase};

paused_cases!(
    sleeps_without_an_invocation_until_due,
    keeps_a_sleep_across_reopen,
    an_abort_mark_ends_a_sleep,
    rejects_sleeps_that_are_not_times_and_sleeps_in_abort_handlers,
    on_wake_replaces_the_sleep_timer,
    runtime_timers_pace_a_phase,
);

const TEN_MINUTES: i64 = 600_000;

/// Sleeps ten minutes from its first phase, then completes with the time it woke.
fn sleeper() -> Arc<Probe> {
    Probe::new("test.sleeper")
        .run(|rt, checkpoint| {
            Box::pin(async move {
                if checkpoint == phase("start") {
                    return Ok(Next::sleep(rt.now() + TEN_MINUTES, phase("after")));
                }
                Ok(Next::complete(Json::from(rt.now())))
            })
        })
        .arc()
}

async fn sleeps_without_an_invocation_until_due(backend: Backend) {
    let probe = sleeper();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let start = env.now();
    let id = harness.start(&probe, conversation).await;
    let asleep = harness
        .until(id, |record| matches!(record.state, TaskState::Sleeping { .. }))
        .await;
    let TaskState::Sleeping { until, checkpoint } = asleep.state else {
        unreachable!()
    };
    assert_eq!(checkpoint, phase("after"));
    assert!(until >= start + TEN_MINUTES);
    assert_eq!(
        harness.scheduler.wake(),
        Wake {
            active: false,
            at: Some(until)
        }
    );
    tokio::time::advance(Duration::from_millis(u64::try_from(TEN_MINUTES).unwrap())).await;
    let settled = harness.settled(id).await;
    assert!(settled.result::<i64>().unwrap().unwrap() >= until);
    harness.close().await;
}

async fn keeps_a_sleep_across_reopen(backend: Backend) {
    let probe = sleeper();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    let asleep = harness
        .until(id, |record| matches!(record.state, TaskState::Sleeping { .. }))
        .await;
    harness.close().await;
    let harness = env.run().await;
    assert_eq!(harness.record(id).await.state, asleep.state, "open keeps a sleep");
    tokio::time::advance(Duration::from_millis(u64::try_from(TEN_MINUTES).unwrap())).await;
    assert!(matches!(
        harness.settled(id).await.outcome(),
        TaskOutcome::Completed { .. }
    ));
    harness.close().await;
}

async fn an_abort_mark_ends_a_sleep(backend: Backend) {
    let probe = sleeper();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    harness
        .until(id, |record| matches!(record.state, TaskState::Sleeping { .. }))
        .await;
    harness.scheduler.abort_task(id, &ctx()).await.unwrap();
    assert!(matches!(
        harness.settled(id).await.outcome(),
        TaskOutcome::Aborted { .. }
    ));
    harness.close().await;
}

async fn rejects_sleeps_that_are_not_times_and_sleeps_in_abort_handlers(backend: Backend) {
    let refused = Arc::new(Mutex::new(None));
    let kept = refused.clone();
    let probe = Probe::new("test.bad-sleep")
        .run(move |rt, checkpoint| {
            let kept = kept.clone();
            Box::pin(async move {
                let sleep = NextState::Sleeping {
                    checkpoint: checkpoint.clone(),
                    until: -1,
                };
                let error = rt
                    .commit(&ctx(), move |_, _| Box::pin(async move { Ok(((), Some(sleep))) }))
                    .await
                    .unwrap_err();
                *kept.lock().unwrap() = Some(error);
                rt.context().cancelled().await;
                Ok(Next::running(checkpoint))
            })
        })
        .on_abort(|rt, checkpoint| Box::pin(async move { Ok(Next::sleep(rt.now() + 1_000, checkpoint)) }))
        .arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    harness.until(id, |_| refused.lock().unwrap().is_some()).await;
    let error = refused.lock().unwrap().take().unwrap();
    assert!(
        matches!(error, RuntimeError::State(StateError::InvalidSleep { until: -1, .. })),
        "{error:?}"
    );
    harness.scheduler.abort_task(id, &ctx()).await.unwrap();
    match harness.settled(id).await.outcome() {
        TaskOutcome::Faulted { error } => assert_eq!(error.message, format!("Abort handler of task {id} cannot sleep")),
        other => panic!("expected a fault, got {other:?}"),
    }
    harness.close().await;
}

async fn on_wake_replaces_the_sleep_timer(backend: Backend) {
    let probe = sleeper();
    let mut env = Env::new(backend, [erased(&probe)]).await;
    let wakes = Arc::new(Mutex::new(Vec::new()));
    env.wakes = Some(wakes.clone());
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    let asleep = harness
        .until(id, |record| matches!(record.state, TaskState::Sleeping { .. }))
        .await;
    let TaskState::Sleeping { until, .. } = asleep.state else {
        unreachable!()
    };
    let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
    while wakes.lock().unwrap().last()
        != Some(&Wake {
            active: false,
            at: Some(until),
        })
    {
        assert!(
            tokio::time::Instant::now() < deadline,
            "on_wake reports the sleep: {:?}",
            wakes.lock().unwrap()
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    // No timer is armed: the host wakes the scheduler.
    tokio::time::advance(Duration::from_millis(u64::try_from(TEN_MINUTES).unwrap())).await;
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(matches!(harness.record(id).await.state, TaskState::Sleeping { .. }));
    harness.scheduler.resume().unwrap();
    assert!(matches!(
        harness.settled(id).await.outcome(),
        TaskOutcome::Completed { .. }
    ));
    harness.close().await;
}

async fn runtime_timers_pace_a_phase(backend: Backend) {
    let probe = Probe::new("test.paced")
        .run(|rt, _| {
            Box::pin(async move {
                let before = rt.now();
                rt.timer(Duration::from_secs(5)).await?;
                Ok(Next::complete(Json::from(rt.now() - before)))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    let settled = harness.settled(id).await;
    assert!(settled.result::<i64>().unwrap().unwrap() >= 5_000);
    harness.close().await;
}
