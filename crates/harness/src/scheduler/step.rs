//! The step: one commit on the line before every phase that applies the precedence rules
//! to the record as committed (spec §5.1).
//!
//! | # | Condition | Action |
//! | --- | --- | --- |
//! | 1 | not `running` (terminal, `completing`, waiting, sleeping) | stop |
//! | 2 | closing | stop; keep checkpoint and mark |
//! | 3 | run mode with a mark, or below a live owner with cancellation intent | stop; reconcile |
//! | 4 | the handler failed, or its closing commit did | write `faulted` |
//! | 5 | checkpoint changed | refresh the snapshot; hand over, or continue |
//! | 6 | checkpoint unchanged | write `faulted` (no durable progress) |
//!
//! A phase's closing [`Next`](super::Next) is run inside the step commit, so its writes, the
//! next state and the decision are one commit. A closure that fails rolls back; a second
//! step then writes the fault.

use std::sync::Arc;

use delta_chord::json::Json;
use futures::future::BoxFuture;

use super::Inner;
use super::catalog::CatalogSnapshot;
use super::definition::Checkpoint;
use super::erased::{AnyTask, same_definition};
use super::error::{Failure, StateError};
use super::invocation::{Invocation, Mode, end};
use super::next::ErasedNext;
use super::ownership::TaskNode;
use super::transition::{can_reserve, commit_state, stage, terminate, with_state};
use crate::records::{TaskOutcome, TaskOutcomeError, TaskRecord, TaskState, TaskStatus};
use crate::session::{FaultReason, OutdatedReason, ReportInfo, SessionError, TransactionScope, Tx, TxError};

/// What a run invocation carries from phase to phase: its definition, the phase's snapshot
/// and the replacement definition it already reported as unable to take over.
pub(super) struct Phase {
    pub(super) definition: Arc<dyn AnyTask>,
    pub(super) snapshot: Arc<dyn CatalogSnapshot>,
    reported: Reported,
}

/// The replacement a run invocation already reported as unable to take over.
enum Reported {
    Nothing,
    Missing,
    Definition(Arc<dyn AnyTask>),
}

impl Reported {
    fn is(&self, replacement: Option<&Arc<dyn AnyTask>>) -> bool {
        match (self, replacement) {
            (Self::Missing, None) => true,
            (Self::Definition(reported), Some(replacement)) => same_definition(reported, replacement),
            _ => false,
        }
    }
}

impl Phase {
    pub(super) fn new(definition: Arc<dyn AnyTask>, snapshot: Arc<dyn CatalogSnapshot>) -> Self {
        Self {
            definition,
            snapshot,
            reported: Reported::Nothing,
        }
    }
}

/// A report the step makes once its commit is durable.
struct Report {
    error: Failure,
    info: ReportInfo,
}

enum Decision {
    Continue(TaskRecord, Phase, Vec<Report>),
    Stop(Vec<Report>),
}

/// The phase's closing commit failed with this message; it rolled back and the task
/// faults in a second step.
struct NextFailed(String);

fn scope_of(invocation: &Invocation) -> TransactionScope {
    TransactionScope {
        conversation_id: Some(invocation.conversation),
        task_id: Some(invocation.task),
    }
}

fn faulted(message: &str) -> TaskOutcome {
    TaskOutcome::Faulted {
        error: TaskOutcomeError::new(message),
    }
}

fn fault_report(invocation: &Invocation, message: String) -> Report {
    Report {
        error: Failure(message),
        info: ReportInfo::TaskFaulted {
            task: invocation.report_task(),
            reason: FaultReason::Threw,
        },
    }
}

/// Rules 1–3 against the committed record; `None` when the invocation stops, which ends it
/// on the line.
fn admit(inner: &Inner, invocation: &Arc<Invocation>, current: Option<TaskRecord>) -> Option<TaskRecord> {
    let stop = {
        let mut state = inner.shared.lock();
        match &current {
            Some(record) if record.state.status() == TaskStatus::Running && !state.closing => {
                if invocation.mode == Mode::Abort {
                    false
                } else if record.abort_requested {
                    true
                } else if !record.background && state.tree.below_cancelled(TaskNode::of(record).parent()) {
                    // An owner's intent ends it before its cascade marks it.
                    state.cascade_pending = true;
                    state.reconcile_scheduled = true;
                    true
                } else {
                    false
                }
            }
            _ => true,
        }
    };
    if stop {
        end(inner, invocation);
        None
    } else {
        current
    }
}

/// Runs one step commit. `Err` when the phase's closing commit failed and rolled back:
/// its closure failed, or (`closing` commits) settlement refused what it wrote. `Ok(None)`
/// for any other failure, which ends the invocation and, unless it is the Session's own,
/// fails the Session (a storage error, a cleanup hook's error).
async fn commit<T, F>(
    inner: &Arc<Inner>,
    invocation: &Arc<Invocation>,
    closing: bool,
    f: F,
) -> Result<Option<T>, NextFailed>
where
    F: for<'a, 'l> FnOnce(&'a mut Tx<'l>) -> BoxFuture<'a, Result<T, TxError>> + Send + 'static,
    T: Send + 'static,
{
    let ctx = invocation.ctx.without_cancel();
    let error = match inner.session.commit_scoped(&ctx, scope_of(invocation), f).await {
        Ok(value) => return Ok(Some(value)),
        // The closing closure failed or panicked, its state was refused, or settlement
        // refused its writes (an owner that is finishing, a document rule): rolled back, and
        // the task's fault, not the Session's. A storage error or a failed Session is lifted
        // out of `SessionError::Tx` and fails the Session below.
        Err(SessionError::Tx(error)) if closing => return Err(NextFailed(error.to_string())),
        Err(error) => error,
    };
    end(inner, invocation);
    inner.fail(error);
    Ok(None)
}

/// Makes a decided step's reports and returns what continues.
fn settle(inner: &Inner, decision: Decision) -> Option<(TaskRecord, Phase)> {
    let (next, reports) = match decision {
        Decision::Continue(record, phase, reports) => (Some((record, phase)), reports),
        Decision::Stop(reports) => (None, reports),
    };
    for report in reports {
        inner.session.report(&report.error, &report.info);
    }
    next
}

/// The step before the first phase: rules 1–3, no writes.
pub(super) async fn before(inner: &Arc<Inner>, invocation: &Arc<Invocation>) -> Option<TaskRecord> {
    let (shared, held) = (inner.clone(), invocation.clone());
    let admitted = commit(inner, invocation, false, move |tx| {
        Box::pin(async move {
            let current = tx.task(held.task).await?;
            Ok(admit(&shared, &held, current))
        })
    })
    .await;
    admitted.ok().flatten().flatten()
}

/// The step after a run phase: rules 1–6.
pub(super) async fn after_phase(
    inner: &Arc<Inner>,
    invocation: &Arc<Invocation>,
    phase: Phase,
    start: Json,
    result: Result<ErasedNext, String>,
) -> Option<(TaskRecord, Phase)> {
    let next = match result {
        Ok(next) => next,
        Err(message) => {
            fault(inner, invocation, message).await;
            return None;
        }
    };
    let (shared, held) = (inner.clone(), invocation.clone());
    let decided = commit(inner, invocation, true, move |tx| {
        Box::pin(async move {
            let current = tx.task(held.task).await?;
            let Some(current) = admit(&shared, &held, current) else {
                return Ok(Decision::Stop(Vec::new()));
            };
            let state = next(&mut *tx, &current).await?;
            let staged = commit_state(&shared, tx, Mode::Run, &current, state).await?;
            let decision = if staged.state.status() == TaskStatus::Running && staged.state.checkpoint() == Some(&start)
            {
                // Rule 6: the closing commit's writes land with the fault.
                let message = format!(
                    "Task {} phase {} returned without durable progress",
                    staged.kind,
                    Checkpoint::phase(&start)
                );
                terminate(&shared, tx, &staged, faulted(&message)).await?;
                Decision::Stop(vec![fault_report(&held, message)])
            } else {
                decide(&shared, tx, &held, phase, staged)?
            };
            if matches!(decision, Decision::Stop(_)) {
                stop_when_durable(&shared, &held);
            }
            Ok(decision)
        })
    })
    .await;
    match decided {
        Ok(decision) => {
            let next = settle(inner, decision?);
            if next.is_none() {
                end(inner, invocation);
            }
            next
        }
        Err(NextFailed(message)) => {
            fault(inner, invocation, message).await;
            None
        }
    }
}

/// A closing commit that stops its invocation ends it when the commit publishes the task's
/// record, on the line, so a runtime commit queued behind it is refused and the scheduler's
/// memory never shows it running. Settlement may still refuse the commit: then nothing is
/// published, the invocation stays, and the fault step ends it.
fn stop_when_durable(inner: &Inner, invocation: &Arc<Invocation>) {
    inner.shared.lock().stopping.insert(invocation.task, invocation.clone());
}

/// Rules 1 and 5 for the record a phase staged.
fn decide(
    inner: &Inner,
    tx: &mut Tx<'_>,
    invocation: &Arc<Invocation>,
    mut phase: Phase,
    staged: TaskRecord,
) -> Result<Decision, TxError> {
    let Some(checkpoint) = staged
        .state
        .checkpoint()
        .filter(|_| staged.state.status() == TaskStatus::Running)
    else {
        // Waiting, sleeping or an outcome: the invocation ends here.
        return Ok(Decision::Stop(Vec::new()));
    };
    let checkpoint = checkpoint.clone();
    let mut reports = Vec::new();
    // Rule 5: refresh the snapshot; hand over to a replacement that can take the task.
    let snapshot = inner.catalog.snapshot();
    if snapshot.generation() != phase.snapshot.generation() {
        let replacement = snapshot.task(&staged.kind).cloned();
        let same = replacement
            .as_ref()
            .is_some_and(|replacement| same_definition(replacement, &phase.definition));
        if !same {
            if let Some(replacement) = &replacement
                && can_reserve(&**replacement, &staged)
            {
                stage(tx, with_state(&staged, TaskState::Pending { checkpoint }))?;
                return Ok(Decision::Stop(reports));
            }
            if !phase.reported.is(replacement.as_ref()) {
                let reason = if replacement.is_some() {
                    OutdatedReason::IncompatibleTask
                } else {
                    OutdatedReason::MissingTask
                };
                reports.push(Report {
                    error: Failure(format!(
                        "Task {} keeps running under its old {} definition",
                        staged.id, staged.kind
                    )),
                    info: ReportInfo::TaskOutdated {
                        task: invocation.report_task(),
                        reason,
                    },
                });
                phase.reported = replacement.map_or(Reported::Missing, Reported::Definition);
            }
        }
        phase.snapshot = snapshot;
    }
    Ok(Decision::Continue(staged, phase, reports))
}

/// The step after the abort handler: rules 1, 2 and 4; a handler that ends without a
/// terminal outcome faults.
pub(super) async fn after_abort(inner: &Arc<Inner>, invocation: &Arc<Invocation>, result: Result<ErasedNext, String>) {
    let next = match result {
        Ok(next) => next,
        Err(message) => return fault(inner, invocation, message).await,
    };
    let (shared, held) = (inner.clone(), invocation.clone());
    let decided = commit(inner, invocation, true, move |tx| {
        Box::pin(async move {
            let current = tx.task(held.task).await?;
            let Some(current) = admit(&shared, &held, current) else {
                return Ok(Decision::Stop(Vec::new()));
            };
            let state = next(&mut *tx, &current).await?;
            let refused = match state {
                TaskState::Terminal { .. } => None,
                TaskState::Waiting { .. } => Some(StateError::AbortWait(held.task)),
                TaskState::Sleeping { .. } => Some(StateError::AbortSleep(held.task)),
                _ => Some(StateError::AbortWithoutOutcome(held.task)),
            };
            if let Some(refused) = refused {
                return Err(TxError::callback(refused));
            }
            commit_state(&shared, tx, Mode::Abort, &current, state).await?;
            stop_when_durable(&shared, &held);
            Ok(Decision::Stop(Vec::new()))
        })
    })
    .await;
    match decided {
        Ok(decision) => {
            if let Some(decision) = decision {
                settle(inner, decision);
                end(inner, invocation);
            }
        }
        Err(NextFailed(message)) => fault(inner, invocation, message).await,
    }
}

/// Rule 4: the handler or its closing commit failed. Rules 1–3 still win: an abort mark
/// that raced the failure stops the run instead.
async fn fault(inner: &Arc<Inner>, invocation: &Arc<Invocation>, message: String) {
    // A refused closing commit published nothing: its stop no longer applies.
    inner.shared.lock().stopping.remove(&invocation.task);
    let (shared, held) = (inner.clone(), invocation.clone());
    let decided = commit(inner, invocation, false, move |tx| {
        Box::pin(async move {
            let current = tx.task(held.task).await?;
            let Some(current) = admit(&shared, &held, current) else {
                return Ok(Decision::Stop(Vec::new()));
            };
            end(&shared, &held);
            terminate(&shared, tx, &current, faulted(&message)).await?;
            Ok(Decision::Stop(vec![fault_report(&held, message)]))
        })
    })
    .await;
    match decided {
        Ok(Some(decision)) => {
            settle(inner, decision);
        }
        // A fault write has no closing commit of its own to fail.
        Ok(None) | Err(NextFailed(_)) => end(inner, invocation),
    }
}
