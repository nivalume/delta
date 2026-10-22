//! Open and close (spec §5.1, §5.4). Open is crash recovery: every surviving `running` task
//! goes back to `pending` with its restart count raised, and the first reconcile pass
//! re-derives cascades and finalizes held outcomes; no handler runs. Close is not a crash:
//! it stops every invocation, joins them and writes the tasks it stopped back as `pending`
//! without a count.

use std::sync::{Arc, Weak};

use delta_chord::context::Context;
use futures::future::join_all;

use super::Inner;
use super::error::{Failure, SchedulerError};
use super::invocation::Invocation;
use super::reserve::parked;
use super::transition::{stage, with_state};
use crate::records::{JoinPolicy, StorageWrite, TaskQuery, TaskRecord, TaskState, TaskStatus};
use crate::session::{ReportInfo, ReportTask};

const SCAN_PAGE: usize = 256;
const LIVE: [TaskStatus; 5] = [
    TaskStatus::Pending,
    TaskStatus::Running,
    TaskStatus::Waiting,
    TaskStatus::Sleeping,
    TaskStatus::Completing,
];

/// Loads every live task and writes surviving `running` tasks back as `pending`, counting
/// the restart. A task that reaches the restart limit this way is parked and reported.
pub(super) async fn open(inner: &Arc<Inner>, ctx: &Context) -> Result<(), SchedulerError> {
    let limit = inner.settings().restart_limit;
    let shared = inner.clone();
    let parked_now = inner
        .session
        .commit(ctx, move |tx| {
            Box::pin(async move {
                // Every table read before the first write.
                let mut records = Vec::new();
                for status in LIVE {
                    let query = TaskQuery {
                        status: Some(status),
                        ..TaskQuery::default()
                    };
                    let mut cursor = None;
                    loop {
                        let page = tx.scan_tasks(query.clone(), SCAN_PAGE, cursor).await?;
                        records.extend(page.items);
                        match page.next {
                            Some(next) => cursor = Some(next),
                            None => break,
                        }
                    }
                }
                let mut restarted = Vec::new();
                let mut parked_now = Vec::new();
                {
                    let mut state = shared.shared.lock();
                    for record in records {
                        if record.abandon_on_restart {
                            state.abandoned.insert(record.id);
                        }
                        if let TaskState::Waiting {
                            policy: JoinPolicy::FailFast,
                            ..
                        } = record.state
                        {
                            state.fail_fast_checks.insert(record.id);
                        }
                        if let TaskState::Running { checkpoint } = &record.state {
                            let mut pending = with_state(
                                &record,
                                TaskState::Pending {
                                    checkpoint: checkpoint.clone(),
                                },
                            );
                            pending.restarts = Some(record.restarts.unwrap_or(0).saturating_add(1));
                            if parked(&pending, limit) {
                                parked_now.push(pending.clone());
                            }
                            restarted.push(pending);
                        }
                        state.track(None, record);
                    }
                }
                for record in restarted {
                    stage(tx, record)?;
                }
                Ok(parked_now)
            })
        })
        .await?;
    for record in parked_now {
        report_parked(inner, &record);
    }
    // Derive the marks a crash left unapplied below cancelled owners; finalize held outcomes.
    {
        let mut state = inner.shared.lock();
        state.cascade_pending = true;
        state.reconcile_scheduled = true;
    }
    inner.shared.wake_driver();
    Ok(())
}

fn report_parked(inner: &Inner, record: &TaskRecord) {
    let restarts = record.restarts.unwrap_or(0);
    let error = Failure(format!(
        "Task {} ({}) was cut off {restarts} times without progress; it is parked until retry_task()",
        record.id, record.kind
    ));
    let snapshot = inner.catalog.snapshot();
    let task = ReportTask {
        id: record.id,
        kind: record.kind.clone(),
        conversation_id: record.conversation_id,
        extension: snapshot.task(&record.kind).and_then(|task| snapshot.extension_of(task)),
    };
    inner.session.report(&error, &ReportInfo::TaskParked { task });
}

/// The close barrier `Session::close` runs after its seal: stops every invocation, joins
/// their code (aborting stragglers after `close_grace`) and writes the tasks it stopped
/// back as `pending`, on the line after the commits admitted before the seal settled.
pub(super) async fn close(inner: Weak<Inner>) {
    let Some(inner) = inner.upgrade() else {
        return;
    };
    let stopped = inner.seal();
    let spawned: Vec<Arc<Invocation>> = inner.shared.lock().spawned.values().cloned().collect();
    let grace = inner.settings().close_grace;
    let joined = tokio::time::timeout(grace, join_all(spawned.iter().map(|invocation| invocation.finished()))).await;
    if joined.is_err() {
        // A handler that ignores its context: dropped at its next yield.
        for invocation in &spawned {
            invocation.abort_code();
        }
        join_all(spawned.iter().map(|invocation| invocation.finished())).await;
    }
    let shared = inner.shared.clone();
    let ids: Vec<_> = stopped.iter().map(|invocation| invocation.task).collect();
    // A failure leaves them `running`, counted at the next open as if cut off.
    let _ = inner
        .session
        .write_at_close(move || {
            let state = shared.lock();
            ids.iter()
                .filter_map(|id| state.tree.live.get(id))
                .filter_map(|record| match &record.state {
                    TaskState::Running { checkpoint } => Some(StorageWrite::Task(with_state(
                        record,
                        TaskState::Pending {
                            checkpoint: checkpoint.clone(),
                        },
                    ))),
                    _ => None,
                })
                .collect()
        })
        .await;
}
