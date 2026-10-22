//! Abort marks and what they imply (spec §5.4, §5.5): `abort_task`, conversation abort, and
//! the reconcile pass that derives cascade marks, `fail_fast` marks and withdrawals and
//! writes the final record of every `completing` task whose owned work drained.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::ops::ControlFlow;
use std::sync::Arc;

use delta_chord::context::Context;

use super::Inner;
use super::error::SchedulerError;
use super::invocation::Mode;
use super::ownership::{Node, Overlay, Scope, TaskNode, Walk, failed_outcome};
use super::scopes::load_scopes;
use super::transition::{stage, terminate, with_abort_mark, with_state};
use super::waits;
use crate::records::{AbortReason, ConversationId, TaskId, TaskOutcome, TaskRecord, TaskState, TaskStatus};
use crate::session::{SessionError, Tx, TxError};

/// What [`Scheduler::abort_task`](super::Scheduler::abort_task) found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskAbort {
    /// The mark is durable (or the task was orphaned at once); its abort handler settles it.
    Marked,
    /// The task was already terminal.
    Terminal,
}

/// Commits the abort mark, or settles a task no definition can take as `orphaned` when
/// nothing it owns is live, then joins the run invocation seen on the line (the observer
/// signalled it). A `completing` task is only marked. A request replaces a `restart` mark;
/// with `keep_restart` (an owner's own cleanup) such a task keeps its mark and waits on.
pub(super) async fn abort(
    inner: &Arc<Inner>,
    id: TaskId,
    ctx: &Context,
    keep_restart: bool,
) -> Result<TaskAbort, SchedulerError> {
    let shared = inner.clone();
    let marked = inner
        .session
        .commit(ctx, move |tx| {
            Box::pin(async move {
                let inner = shared;
                let current = tx.task(id).await?.ok_or(TxError::UnknownTask(id))?;
                if current.state.status() == TaskStatus::Terminal {
                    return Ok((TaskAbort::Terminal, None));
                }
                if keep_restart && current.abort_reason == Some(AbortReason::Restart) {
                    return Ok((TaskAbort::Marked, None));
                }
                let invocation = inner.shared.lock().invocations.get(&id).cloned();
                if invocation.is_none() && current.state.status() != TaskStatus::Completing {
                    load_scopes(&inner.shared, tx, false).await?;
                    let snapshot = inner.catalog.snapshot();
                    let blocked = {
                        let mut state = inner.shared.lock();
                        if state.tree.has_owned_live(id, None, None) {
                            None
                        } else {
                            let mut reports = Vec::new();
                            let blocked = state.resolve(&current, &*snapshot, &mut reports).err();
                            drop(state);
                            for (error, info) in &reports {
                                inner.session.report(error, info);
                            }
                            blocked
                        }
                    };
                    if let Some(reason) = blocked {
                        let outcome = TaskOutcome::Orphaned {
                            reason: reason.as_str().to_owned(),
                        };
                        terminate(&inner, tx, &current, outcome).await?;
                        return Ok((TaskAbort::Marked, None));
                    }
                }
                if !current.abort_requested || current.abort_reason.is_some() {
                    stage(tx, with_abort_mark(&current, None))?;
                }
                let run = invocation.filter(|invocation| invocation.mode == Mode::Run);
                Ok((TaskAbort::Marked, run))
            })
        })
        .await
        .map_err(|error| unknown_task(error, id))?;
    let (result, run) = marked;
    if let Some(run) = run {
        // A run that ignores its signal can outlive a failed Session, which ends the wait.
        let closed = inner.session.closed();
        ctx.run(async {
            tokio::select! {
                () = run.finished() => {}
                _ = closed => {}
            }
        })
        .await?;
    }
    Ok(result)
}

/// `TxError::UnknownTask` as the scheduler's own error.
pub(super) fn unknown_task(error: SessionError, id: TaskId) -> SchedulerError {
    match error {
        SessionError::Tx(TxError::UnknownTask(unknown)) if unknown == id => SchedulerError::UnknownTask(id),
        other => SchedulerError::from(other),
    }
}

/// `Conversation::abort`: in one commit, withdraws the queued inputs of the scope and marks
/// every live non-background task ordinary traversal reaches from it; resolves once the
/// scope is idle. With `background`, traversal crosses background boundaries, and the wait
/// also covers every task it marked.
pub(super) async fn abort_conversation(
    inner: &Arc<Inner>,
    conversation: ConversationId,
    background: bool,
    ctx: &Context,
) -> Result<(), SchedulerError> {
    let shared = inner.clone();
    let reached = inner
        .session
        .commit(ctx, move |tx| {
            Box::pin(async move {
                let inner = shared;
                let queued = load_scopes(&inner.shared, tx, true).await?;
                let mut reached = Vec::new();
                {
                    let state = inner.shared.lock();
                    let walk = Walk {
                        cross_background: background,
                        ..Walk::default()
                    };
                    let mut marks = Vec::new();
                    state.tree.walk_down(Node::Conversation(conversation), walk, |found| {
                        if let Some(live) = found.live
                            && (background || !found.node.background)
                        {
                            reached.push(found.id);
                            if !live.abort_requested || live.abort_reason.is_some() {
                                marks.push(with_abort_mark(live, None));
                            }
                        }
                        ControlFlow::Continue(())
                    });
                    for mark in marks {
                        tx.set_task(mark)?;
                    }
                }
                for (id, inputs) in queued {
                    let scope = Scope::Conversation(conversation);
                    let inside = inner
                        .shared
                        .lock()
                        .tree
                        .in_scope(Node::Conversation(id), scope, background);
                    if inside == Some(true) {
                        inner.hooks.withdraw_inputs(tx, id, &inputs).await?;
                    }
                }
                Ok(reached)
            })
        })
        .await?;
    if background {
        for id in reached {
            waits::wait_for_task(inner, id, ctx).await?;
        }
    }
    waits::wait_for_idle(inner, Some(conversation), ctx).await
}

/// One commit that applies what committed records imply: abort marks below live owners with
/// cancellation intent, `fail_fast` marks, withdrawn queued inputs below cancelled owners,
/// and the final record of every `completing` task without live owned work. The durable
/// records are the intent, so this also repairs what a crash left unapplied.
pub(super) async fn reconcile(inner: &Arc<Inner>) {
    let (work, cascade, checks) = {
        let mut state = inner.shared.lock();
        state.reconcile_scheduled = false;
        let work = state.reconcile_work();
        if work {
            state.reconciling += 1;
        }
        let cascade = std::mem::take(&mut state.cascade_pending);
        let checks = std::mem::take(&mut state.fail_fast_checks);
        (work, cascade, checks)
    };
    let shared = inner.clone();
    let result = inner
        .session
        .commit(&inner.ctx, move |tx| {
            Box::pin(async move {
                let inner = shared;
                {
                    let mut state = inner.shared.lock();
                    state.tree.sweep();
                    if state.closing {
                        return Ok(());
                    }
                }
                let queued = load_scopes(&inner.shared, tx, cascade).await?;
                let failing = fail_fast_members(&inner, tx, checks).await?;
                {
                    let state = inner.shared.lock();
                    for (record, reason) in cascade_marks(&state.tree, &failing) {
                        tx.set_task(with_abort_mark(&record, reason))?;
                    }
                }
                for (id, inputs) in queued {
                    if inner.shared.lock().tree.below_cancelled(Node::Conversation(id)) {
                        inner.hooks.withdraw_inputs(tx, id, &inputs).await?;
                    }
                }
                finalize(&inner, tx).await
            })
        })
        .await;
    if let Err(error) = result {
        // No task code runs here: an error is a storage failure, a hook's or a bug.
        inner.fail(error);
    }
    if work {
        inner.shared.lock().reconciling -= 1;
    }
    inner.shared.bump_changes();
}

/// The live members of every checked `fail_fast` wait whose `on` holds a failure: they get
/// marks (the failed member keeps its own outcome).
async fn fail_fast_members(
    inner: &Inner,
    tx: &mut Tx<'_>,
    checks: BTreeSet<TaskId>,
) -> Result<Vec<TaskRecord>, TxError> {
    let waits: Vec<Vec<TaskId>> = {
        let state = inner.shared.lock();
        checks
            .iter()
            .filter_map(|id| match state.tree.live.get(id).map(|record| &record.state) {
                Some(TaskState::Waiting { on, .. }) => Some(on.clone()),
                _ => None,
            })
            .collect()
    };
    let mut members = Vec::new();
    for on in waits {
        let mut failed = false;
        for id in &on {
            let live = inner.shared.lock().tree.live.get(id).cloned();
            let record = match live {
                Some(record) => Some(record),
                None => tx.stored_task(*id).await?,
            };
            if record.is_some_and(|record| failed_outcome(&record)) {
                failed = true;
                break;
            }
        }
        if failed {
            let state = inner.shared.lock();
            members.extend(
                on.iter()
                    .filter_map(|id| state.tree.live.get(id))
                    .filter(|record| !failed_outcome(record))
                    .cloned(),
            );
        }
    }
    Ok(members)
}

/// The marks a pass stages: every live foreground task below a live owner with intent, its
/// nearest one (a walk down does not enter another owner with intent, whose own walk covers
/// what is below it), and the `fail_fast` members. Only an owner abandoned after a restart
/// passes `Restart` on; any other intent marks as a request and replaces a `Restart` mark.
fn cascade_marks(tree: &super::ownership::Tree, failing: &[TaskRecord]) -> Vec<(TaskRecord, Option<AbortReason>)> {
    let mut marks: BTreeMap<TaskId, (TaskRecord, Option<AbortReason>)> = BTreeMap::new();
    let mut mark = |record: &TaskRecord, reason: Option<AbortReason>| {
        if marks.get(&record.id).is_some_and(|(_, staged)| staged.is_none()) {
            return;
        }
        if record.abort_requested && (record.abort_reason.is_none() || reason.is_some()) {
            return;
        }
        marks.insert(record.id, (record.clone(), reason));
    };
    for id in &tree.intent {
        let Some(owner) = tree.live.get(id) else {
            continue;
        };
        let restart = owner.abort_reason == Some(AbortReason::Restart) && !failed_outcome(owner);
        let reason = restart.then_some(AbortReason::Restart);
        let enter = |below: TaskId| !tree.intent.contains(&below);
        let walk = Walk {
            enter: Some(&enter),
            ..Walk::default()
        };
        tree.walk_down(Node::Task(*id), walk, |found| {
            if let Some(live) = found.live
                && !found.node.background
            {
                mark(live, reason);
            }
            ControlFlow::Continue(())
        });
    }
    for record in failing {
        mark(record, None);
    }
    marks.into_values().collect()
}

/// Writes the terminal record of every `completing` task without live ordinary owned work.
/// Finalizing one can free its owner, so this repeats in rounds over the commit's
/// candidates. A held scheduler outcome gets its cleanup hook here.
async fn finalize(inner: &Inner, tx: &mut Tx<'_>) -> Result<(), TxError> {
    let mut round: Vec<TaskId> = std::mem::take(&mut inner.shared.lock().finalize_checks)
        .into_iter()
        .collect();
    let mut done: HashSet<TaskId> = HashSet::new();
    let mut overlay = Overlay::of(tx);
    while !round.is_empty() {
        let mut next = BTreeSet::new();
        for id in round {
            let ready = {
                let state = inner.shared.lock();
                let record = overlay.tasks.get(&id).or_else(|| state.tree.live.get(&id));
                record
                    .filter(|record| record.state.status() == TaskStatus::Completing && !done.contains(&id))
                    .filter(|_| !state.tree.has_owned_live(id, Some(&overlay), Some(&done)))
                    .cloned()
            };
            let Some(record) = ready else {
                continue;
            };
            let Some(outcome) = record.state.outcome().cloned() else {
                continue;
            };
            let scheduler_written = matches!(outcome, TaskOutcome::Faulted { .. } | TaskOutcome::Orphaned { .. });
            let terminal = stage(tx, with_state(&record, TaskState::Terminal { outcome }))?;
            overlay.tasks.insert(id, terminal.clone());
            done.insert(id);
            // Only the scheduler writes `faulted` and `orphaned`; their cleanup waits for this commit.
            if scheduler_written {
                inner.hooks.settle_scheduler_outcome(tx, &terminal).await?;
            }
            let state = inner.shared.lock();
            if let Some(above) = state.tree.nearest_live_above(TaskNode::of(&record), Some(&overlay))
                && above.state.status() == TaskStatus::Completing
            {
                next.insert(above.id);
            }
        }
        round = next.into_iter().collect();
    }
    Ok(())
}

/// Clears the restart count of a parked task; nothing for one that is not parked.
pub(super) async fn retry(inner: &Arc<Inner>, id: TaskId, ctx: &Context) -> Result<(), SchedulerError> {
    let limit = inner.settings().restart_limit;
    inner
        .session
        .commit(ctx, move |tx| {
            Box::pin(async move {
                let current = tx.task(id).await?.ok_or(TxError::UnknownTask(id))?;
                let status = current.state.status();
                if matches!(status, TaskStatus::Terminal | TaskStatus::Completing)
                    || !super::reserve::parked(&current, limit)
                {
                    return Ok(());
                }
                stage(tx, super::transition::progressed(&current))?;
                Ok(())
            })
        })
        .await
        .map_err(|error| unknown_task(error, id))
}
