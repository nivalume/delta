//! Phases, checkpoints, closing commits, memos, the runtime's gate and abort handlers.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use delta_chord::context::Context;
use delta_chord::json::Json;
use delta_harness::records::{EntryDraft, TaskOutcome, TaskState, TaskStatus};
use delta_harness::scheduler::{
    Checkpoint, Next, NextState, RuntimeError, SchedulerError, TaskAbort, TaskDefinition, TaskError, TaskRuntime,
};
use delta_harness::session::{FaultReason, ReportInfo, TaskOptions, TxError};
use serde::{Deserialize, Serialize};

use crate::{Backend, Count, Env, Gate, Log, Probe, ctx, erased, eventually, phase};

cases!(
    phases_progress_through_typed_checkpoints,
    faults_without_progress_errors_and_panics,
    closing_commit_is_atomic_with_its_writes,
    failed_closing_commit_rolls_back_and_faults,
    runtime_commit_ends_the_phase_early,
    memos_keep_the_first_writer_until_terminal,
    stamps_lifecycle_times,
    runtime_rejects_once_the_task_is_not_running,
    abort_mark_stops_the_run_then_runs_the_abort_handler,
    abort_handler_without_an_outcome_faults,
    abort_of_unknown_and_terminal_tasks,
    idle_waits_count_blocked_work_and_skip_background,
);

/// The typed counter's checkpoint, stored as `{"phase":"count","n":…}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
enum Counting {
    Count { n: u32 },
}

impl Checkpoint for Counting {
    fn phase(&self) -> &'static str {
        "count"
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CountTo {
    to: u32,
}

/// Counts to its input, one phase per step; waits at `hold` once.
struct Counter {
    seen: Arc<Mutex<Vec<u32>>>,
    hold: Option<(u32, Gate)>,
}

impl TaskDefinition for Counter {
    type Input = CountTo;
    type Checkpoint = Counting;
    type Result = u32;

    fn name(&self) -> &'static str {
        "test.counter"
    }

    fn version(&self) -> u32 {
        1
    }

    fn initial(&self, _input: &CountTo) -> Counting {
        Counting::Count { n: 0 }
    }

    async fn run(&self, phase: Counting, rt: &TaskRuntime<Self>, _ctx: &Context) -> Result<Next<Self>, TaskError> {
        let Counting::Count { n } = phase;
        self.seen.lock().unwrap().push(n);
        if let Some((at, gate)) = &self.hold
            && *at == n
        {
            gate.wait().await;
        }
        let next = n + 1;
        Ok(if next == rt.input().to {
            Next::complete(next)
        } else {
            Next::running(Counting::Count { n: next })
        })
    }

    async fn abort(&self, _phase: Counting, _rt: &TaskRuntime<Self>, _ctx: &Context) -> Result<Next<Self>, TaskError> {
        Ok(Next::abort(None))
    }
}

async fn phases_progress_through_typed_checkpoints(backend: Backend) {
    let gate = Gate::new();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let counter = Arc::new(Counter {
        seen: seen.clone(),
        hold: Some((2, gate.clone())),
    });
    let env = Env::new(backend, [counter.clone() as Arc<dyn delta_harness::scheduler::AnyTask>]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness
        .create(&counter, CountTo { to: 3 }, TaskOptions::conversation(conversation))
        .await;
    eventually(|| seen.lock().unwrap().len() == 3).await;
    // The typed checkpoint is stored serde-tagged.
    let running = env.stored(id).await;
    assert_eq!(
        running.state,
        TaskState::Running {
            checkpoint: delta_chord::json!({ "phase": "count", "n": 2 })
        }
    );
    gate.open();
    let settled = harness.settled(id).await;
    assert_eq!(settled.result::<u32>().unwrap().unwrap(), 3);
    assert_eq!(*seen.lock().unwrap(), vec![0, 1, 2]);
    harness.close().await;
}

async fn faults_without_progress_errors_and_panics(backend: Backend) {
    let idle = Probe::new("test.idle")
        .run(|_, checkpoint| Box::pin(async move { Ok(Next::running(checkpoint)) }))
        .arc();
    let throws = Probe::new("test.throws")
        .run(|_, _| Box::pin(async { Err(TaskError::new("boom")) }))
        .arc();
    let panics = Probe::new("test.panics")
        .run(|_, _| Box::pin(async { panic!("kaboom") }))
        .arc();
    let env = Env::new(backend, [erased(&idle), erased(&throws), erased(&panics)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let ids = [
        harness.start(&idle, conversation).await,
        harness.start(&throws, conversation).await,
        harness.start(&panics, conversation).await,
    ];
    let mut messages = Vec::new();
    for id in ids {
        match harness.settled(id).await.outcome() {
            TaskOutcome::Faulted { error } => messages.push(error.message.clone()),
            other => panic!("expected a fault, got {other:?}"),
        }
    }
    assert_eq!(
        messages[0],
        "Task test.idle phase start returned without durable progress"
    );
    assert_eq!(messages[1], "boom");
    assert!(messages[2].contains("kaboom"), "{}", messages[2]);
    // Reported once each fault is durable, which a waiter may see first.
    eventually(|| {
        let faults = env
            .reported()
            .into_iter()
            .filter(|(_, info)| {
                matches!(
                    info,
                    ReportInfo::TaskFaulted {
                        reason: FaultReason::Threw,
                        ..
                    }
                )
            })
            .count();
        faults == 3
    })
    .await;
    harness.close().await;
}

async fn closing_commit_is_atomic_with_its_writes(backend: Backend) {
    let probe = Probe::new("test.result")
        .run(|rt, _| {
            Box::pin(async move {
                Ok(rt.next(|tx, current| {
                    Box::pin(async move {
                        let entry = tx
                            .append_entry(current.record.conversation_id, EntryDraft::new("test.result"))
                            .await?;
                        Ok(NextState::completed(delta_chord::json!({ "entry": entry.id })))
                    })
                }))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    let settled = harness.settled(id).await;
    let result: Json = settled.result().unwrap().unwrap();
    let entry = delta_harness::records::EntryId(result.get("entry").and_then(Json::as_u64).unwrap());
    let stored = env.storage.entry(entry, &ctx()).await.unwrap().unwrap();
    assert_eq!(
        stored.entry.by_task_id,
        Some(id),
        "the step commit is scoped to the task"
    );
    assert!(settled.record().memos.is_none());
    harness.close().await;
}

async fn failed_closing_commit_rolls_back_and_faults(backend: Backend) {
    let probe = Probe::new("test.rollback")
        .run(|rt, _| {
            Box::pin(async move {
                Ok(rt.next(|tx, current| {
                    Box::pin(async move {
                        tx.append_entry(current.record.conversation_id, EntryDraft::new("test.lost"))
                            .await?;
                        Err(TxError::callback("refused"))
                    })
                }))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    match harness.settled(id).await.outcome() {
        TaskOutcome::Faulted { error } => assert_eq!(error.message, "refused"),
        other => panic!("expected a fault, got {other:?}"),
    }
    let query = delta_harness::records::EntryQuery {
        conversation_id: conversation,
        min_entry_id: None,
        max_entry_id: None,
        order: None,
    };
    let entries = env.storage.scan_entries(query, 10, None, &ctx()).await.unwrap();
    assert!(entries.items.is_empty(), "the closing commit rolled back");
    harness.close().await;
}

async fn runtime_commit_ends_the_phase_early(backend: Backend) {
    let discarded = Count::default();
    let ran = discarded.clone();
    let probe = Probe::new("test.early")
        .run(move |rt, _| {
            let ran = ran.clone();
            Box::pin(async move {
                rt.commit(&ctx(), |_, _| {
                    Box::pin(async { Ok(((), Some(NextState::completed(delta_chord::json!("early"))))) })
                })
                .await?;
                // The phase's own closing commit is no longer admitted: the task is terminal.
                Ok(rt.next(move |_, _| {
                    ran.hit();
                    Box::pin(async { Ok(NextState::completed(delta_chord::json!("late"))) })
                }))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    let settled = harness.settled(id).await;
    assert_eq!(settled.result::<String>().unwrap().unwrap(), "early");
    harness
        .scheduler
        .wait_for_idle(Some(conversation), &ctx())
        .await
        .unwrap();
    assert_eq!(discarded.get(), 0);
    harness.close().await;
}

async fn memos_keep_the_first_writer_until_terminal(backend: Backend) {
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let record = seen.clone();
    let probe = Probe::new("test.memo")
        .run(move |rt, checkpoint| {
            let record = record.clone();
            Box::pin(async move {
                if checkpoint.get("phase").and_then(Json::as_str) == Some("start") {
                    let first: String = rt.memo_or_insert("key", "first".to_owned(), &ctx()).await?;
                    let second: String = rt.memo_or_insert("key", "second".to_owned(), &ctx()).await?;
                    record.lock().unwrap().extend([first, second]);
                    return Ok(Next::running(phase("next")));
                }
                // Memos survive checkpoints.
                let kept: Option<String> = rt.memo("key")?;
                record.lock().unwrap().push(kept.unwrap_or_default());
                Ok(Next::complete(Json::Null))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    let settled = harness.settled(id).await;
    assert_eq!(*seen.lock().unwrap(), ["first", "first", "first"]);
    assert!(settled.record().memos.is_none(), "terminal drops the memos");
    harness.close().await;
}

async fn stamps_lifecycle_times(backend: Backend) {
    let probe = Probe::new("test.times")
        .run(|_, checkpoint| {
            Box::pin(async move {
                if checkpoint == phase("start") {
                    return Ok(Next::wait(
                        Vec::new(),
                        delta_harness::records::JoinPolicy::AllSettled,
                        phase("after"),
                    ));
                }
                Ok(Next::complete(Json::Null))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.open().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    let pending = harness.record(id).await;
    assert!(pending.started_at.is_none() && pending.ended_at.is_none());
    harness.scheduler.resume().unwrap();
    let settled = harness.settled(id).await;
    let started = settled.record().started_at.unwrap();
    let ended = settled.record().ended_at.unwrap();
    assert!(started >= crate::EPOCH && ended >= started);
    harness.close().await;
}

async fn runtime_rejects_once_the_task_is_not_running(backend: Backend) {
    let refused = Arc::new(Mutex::new(None));
    let contexts = Arc::new(Mutex::new(Vec::<Context>::new()));
    let (slot, kept) = (refused.clone(), contexts.clone());
    let probe = Probe::new("test.gate")
        .run(move |rt, checkpoint| {
            let (slot, kept) = (slot.clone(), kept.clone());
            Box::pin(async move {
                kept.lock().unwrap().push(rt.context().clone());
                if checkpoint == phase("start") {
                    rt.commit(&ctx(), |_, _| {
                        let wait = NextState::Waiting {
                            checkpoint: phase("after"),
                            on: Vec::new(),
                            policy: delta_harness::records::JoinPolicy::AllSettled,
                        };
                        Box::pin(async move { Ok(((), Some(wait))) })
                    })
                    .await?;
                    let error = rt.memo_or_insert("late", 1, &ctx()).await.unwrap_err();
                    *slot.lock().unwrap() = Some(error);
                    return Ok(Next::running(phase("ignored")));
                }
                Ok(Next::complete(Json::Null))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    harness.settled(id).await;
    let error = refused.lock().unwrap().take().unwrap();
    assert!(
        matches!(
            error,
            RuntimeError::NotRunning {
                status: TaskStatus::Waiting,
                ..
            }
        ),
        "{error:?}"
    );
    harness
        .scheduler
        .wait_for_idle(Some(conversation), &ctx())
        .await
        .unwrap();
    // Each invocation's context is cancelled once it ended.
    let contexts = contexts.lock().unwrap().clone();
    assert_eq!(contexts.len(), 2);
    eventually(|| contexts.iter().all(Context::is_cancelled)).await;
    harness.close().await;
}

async fn abort_mark_stops_the_run_then_runs_the_abort_handler(backend: Backend) {
    let log = Log::default();
    let (running, aborting) = (log.clone(), log.clone());
    let probe = Probe::new("test.abortable")
        .run(move |rt, _| {
            let log = running.clone();
            Box::pin(async move {
                log.push("run");
                rt.context().cancelled().await;
                let error = rt.memo_or_insert("late", 1, &ctx()).await.unwrap_err();
                log.push(format!("refused: {}", matches!(error, RuntimeError::Aborted(_))));
                Err(TaskError::new("cancelled"))
            })
        })
        .on_abort(move |rt, _| {
            let log = aborting.clone();
            Box::pin(async move {
                log.push(format!("abort marked={}", rt.task().record.abort_requested));
                Ok(Next::abort(Some("user".to_owned())))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    eventually(|| log.lines().len() == 1).await;
    let result = harness.scheduler.abort_task(id, &ctx()).await.unwrap();
    assert_eq!(result, TaskAbort::Marked);
    // abort_task joined the run before returning; the abort handler may have started since.
    assert_eq!(log.lines()[..2], ["run", "refused: true"]);
    match harness.settled(id).await.outcome() {
        TaskOutcome::Aborted { reason, .. } => assert_eq!(reason.as_deref(), Some("user")),
        other => panic!("expected aborted, got {other:?}"),
    }
    assert_eq!(log.lines(), ["run", "refused: true", "abort marked=true"]);
    // The run's error raced the mark and lost: no fault was written or reported.
    assert!(env.reported().is_empty(), "{:?}", env.reported());
    harness.close().await;
}

async fn abort_handler_without_an_outcome_faults(backend: Backend) {
    let gate = Gate::new();
    let held = gate.clone();
    let probe = Probe::new("test.no-outcome")
        .run(move |_, _| {
            let gate = held.clone();
            Box::pin(async move {
                gate.wait().await;
                Ok(Next::complete(Json::Null))
            })
        })
        .on_abort(|_, checkpoint| Box::pin(async move { Ok(Next::running(checkpoint)) }))
        .arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.open().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    harness.scheduler.abort_task(id, &ctx()).await.unwrap();
    harness.scheduler.resume().unwrap();
    match harness.settled(id).await.outcome() {
        TaskOutcome::Faulted { error } => {
            assert_eq!(
                error.message,
                format!("Abort handler of task {id} returned without a terminal outcome")
            );
        }
        other => panic!("expected a fault, got {other:?}"),
    }
    harness.close().await;
}

async fn abort_of_unknown_and_terminal_tasks(backend: Backend) {
    let probe = Probe::new("test.done").arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    harness.settled(id).await;
    assert_eq!(
        harness.scheduler.abort_task(id, &ctx()).await.unwrap(),
        TaskAbort::Terminal
    );
    let unknown = delta_harness::records::TaskId(9_999);
    assert!(matches!(
        harness.scheduler.abort_task(unknown, &ctx()).await,
        Err(SchedulerError::UnknownTask(found)) if found == unknown
    ));
    assert!(matches!(
        harness.scheduler.wait_for_task(unknown, &ctx()).await,
        Err(SchedulerError::UnknownTask(_))
    ));
    harness.close().await;
}

async fn idle_waits_count_blocked_work_and_skip_background(backend: Backend) {
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
    let missing = Probe::new("test.missing").arc();
    let env = Env::new(backend, [erased(&slow)]).await;
    let harness = env.run().await;
    let busy = harness.conversation().await;
    let background = harness.conversation().await;
    let blocked = harness.conversation().await;
    let fast = harness.start(&slow, busy).await;
    harness
        .create(
            &slow,
            Json::Null,
            TaskOptions {
                background: true,
                ..TaskOptions::conversation(background)
            },
        )
        .await;
    harness.start(&missing, blocked).await;
    // Background work does not keep its conversation busy.
    harness.scheduler.wait_for_idle(Some(background), &ctx()).await.unwrap();
    let background = ctx();
    let idle = harness.scheduler.wait_for_idle(Some(busy), &background);
    assert!(tokio::time::timeout(Duration::from_millis(50), idle).await.is_err());
    gate.open();
    harness.settled(fast).await;
    harness.scheduler.wait_for_idle(Some(busy), &ctx()).await.unwrap();
    // A task without a definition stays live and blocks idleness.
    let blocked_idle = harness.scheduler.wait_for_idle(Some(blocked), &background);
    assert!(
        tokio::time::timeout(Duration::from_millis(50), blocked_idle)
            .await
            .is_err()
    );
    let all = harness.scheduler.wait_for_idle(None, &background);
    assert!(tokio::time::timeout(Duration::from_millis(20), all).await.is_err());
    env.catalog.install(erased(&missing));
    harness.scheduler.wait_for_idle(None, &ctx()).await.unwrap();
    harness.close().await;
}
