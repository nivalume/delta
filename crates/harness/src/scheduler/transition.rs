//! Task transitions inside a scheduler commit: the record helpers, a state a task commits
//! for itself (with its validation and the `completing` hold), and an outcome the scheduler
//! writes (`terminate`).

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;

use super::Inner;
use super::erased::AnyTask;
use super::error::StateError;
use super::invocation::Mode;
use super::ownership::{Node, Overlay, TaskNode, Tree, Walk};
use super::scopes::{load_chain, load_scopes};
use crate::records::{AbortReason, JoinPolicy, TaskId, TaskOutcome, TaskRecord, TaskState, TaskStatus};
use crate::session::{Tx, TxError};

/// The record with `state`; memos and the restart count go once an outcome is decided.
pub(super) fn with_state(record: &TaskRecord, state: TaskState) -> TaskRecord {
    let mut next = record.clone();
    if matches!(state, TaskState::Completing { .. } | TaskState::Terminal { .. }) {
        next.memos = None;
        next.restarts = None;
    }
    next.state = state;
    next
}

/// The record without its restart count, as after a state the task committed.
pub(super) fn progressed(record: &TaskRecord) -> TaskRecord {
    let mut next = record.clone();
    next.restarts = None;
    next
}

/// The record with an abort mark: `Restart` for an abandonment; without a reason, a
/// request, which replaces `Restart`. The restart count starts over: it now counts cut-offs
/// of the abort handler.
pub(super) fn with_abort_mark(record: &TaskRecord, reason: Option<AbortReason>) -> TaskRecord {
    let mut next = record.clone();
    next.abort_requested = true;
    next.abort_reason = reason;
    next.restarts = None;
    next
}

/// Whether a definition can take the task at reservation: same version, or newer with a
/// migration.
pub(super) fn can_reserve(task: &dyn AnyTask, record: &TaskRecord) -> bool {
    task.version() == record.version || (task.version() > record.version && task.can_migrate())
}

/// Stages `record` and returns the candidate as stamped (`started_at`, `ended_at`).
pub(super) fn stage(tx: &mut Tx<'_>, record: TaskRecord) -> Result<TaskRecord, TxError> {
    let id = record.id;
    tx.set_task(record.clone())?;
    Ok(tx
        .staged_tasks()
        .find(|staged| staged.id == id)
        .cloned()
        .unwrap_or(record))
}

fn refuse(error: StateError) -> TxError {
    TxError::callback(error)
}

/// Replaces a running task's state with what it committed. A terminal state holds as
/// `completing` while ordinary owned work is live, judged on the commit's candidates; a wait
/// and a sleep are validated first. A state that differs from the current one is progress
/// and clears the restart count. Returns the staged record.
pub(super) async fn commit_state(
    inner: &Inner,
    tx: &mut Tx<'_>,
    mode: Mode,
    current: &TaskRecord,
    next: TaskState,
) -> Result<TaskRecord, TxError> {
    match &next {
        TaskState::Waiting { on, policy, .. } => validate_wait(inner, tx, mode, current, on, *policy).await?,
        TaskState::Sleeping { until, .. } => validate_sleep(mode, current.id, *until)?,
        TaskState::Terminal { outcome } => {
            let shared = &inner.shared;
            load_scopes(shared, tx, false).await?;
            let overlay = Overlay::of(tx);
            for record in overlay.tasks.values() {
                load_chain(shared, tx, TaskNode::of(record).parent(), Some(&overlay)).await?;
            }
            let holds = shared.lock().tree.has_owned_live(current.id, Some(&overlay), None);
            if holds {
                let held = TaskState::Completing {
                    outcome: outcome.clone(),
                };
                return stage(tx, with_state(&progressed(current), held));
            }
        }
        TaskState::Pending { .. } | TaskState::Running { .. } | TaskState::Completing { .. } => {}
    }
    // The same checkpoint committed again (intent before an effect, say) is not progress.
    let base = if current.state == next {
        current.clone()
    } else {
        progressed(current)
    };
    stage(tx, with_state(&base, next))
}

/// Writes an outcome the scheduler decided (`faulted`, `orphaned`). While the task's
/// ordinary owned work is live it holds as `completing` and the cleanup hook waits for the
/// final commit (§5.5, rule 4); otherwise it is terminal and the hook runs now.
pub(super) async fn terminate(
    inner: &Inner,
    tx: &mut Tx<'_>,
    record: &TaskRecord,
    outcome: TaskOutcome,
) -> Result<(), TxError> {
    load_scopes(&inner.shared, tx, false).await?;
    let overlay = Overlay::of(tx);
    let holds = inner.shared.lock().tree.has_owned_live(record.id, Some(&overlay), None);
    if holds {
        stage(tx, with_state(record, TaskState::Completing { outcome }))?;
        return Ok(());
    }
    let terminal = stage(tx, with_state(record, TaskState::Terminal { outcome }))?;
    inner.hooks.settle_scheduler_outcome(tx, &terminal).await
}

/// A sleep's `until` is a time; an abort handler cannot sleep.
fn validate_sleep(mode: Mode, task: TaskId, until: i64) -> Result<(), TxError> {
    if mode == Mode::Abort {
        return Err(refuse(StateError::AbortSleep(task)));
    }
    if until < 0 {
        return Err(refuse(StateError::InvalidSleep { task, until }));
    }
    Ok(())
}

/// A wait names existing tasks, and `fail_fast` only tasks the waiter owns; it may not
/// close a cycle in the wait-for graph (§5.5). An abort handler cannot wait.
///
/// Invariant: the wait-for graph is acyclic. Only a wait adds an edge between existing
/// tasks (a new task has only incoming edges), so checking each wait maintains it.
async fn validate_wait(
    inner: &Inner,
    tx: &mut Tx<'_>,
    mode: Mode,
    current: &TaskRecord,
    on: &[TaskId],
    policy: JoinPolicy,
) -> Result<(), TxError> {
    if mode == Mode::Abort {
        return Err(refuse(StateError::AbortWait(current.id)));
    }
    let overlay = Overlay::of(tx);
    let mut targets = Vec::new();
    for &id in on {
        let known = overlay
            .tasks
            .get(&id)
            .cloned()
            .or_else(|| inner.shared.lock().tree.live.get(&id).cloned());
        let member = match known {
            Some(member) => Some(member),
            None => tx.stored_task(id).await?,
        };
        let Some(member) = member else {
            return Err(refuse(StateError::UnknownTask(id)));
        };
        let owned = member.owner == Some(current.id);
        if policy == JoinPolicy::FailFast && !owned {
            return Err(refuse(StateError::FailFastNotOwned {
                task: current.id,
                on: id,
            }));
        }
        // A live child already holds the waiter: waiting on it adds no edge. A terminal task
        // has no edges.
        if member.state.status() != TaskStatus::Terminal && !owned {
            targets.push(id);
        }
    }
    if targets.is_empty() {
        return Ok(());
    }
    // Owned work is found by walks down, which need the chains of every live task loaded.
    load_scopes(&inner.shared, tx, false).await?;
    let cycle = wait_cycle(&inner.shared.lock().tree, current.id, &targets, &overlay);
    match cycle {
        Some(path) => {
            let on = path.first().copied().unwrap_or(current.id);
            let mut full = vec![current.id];
            full.extend(path);
            Err(refuse(StateError::Cycle {
                task: current.id,
                on,
                path: full,
            }))
        }
        None => Ok(()),
    }
}

/// The path from a member of `on` back to `task` in the wait-for graph, if the wait would
/// close a cycle. An edge X → Y means X cannot settle before Y: Y is in the `on` of a
/// waiting X without an abort mark, or Y is live ordinary owned work of X. Each task is
/// visited once and each subtree walked once.
fn wait_cycle(tree: &Tree, task: TaskId, on: &[TaskId], overlay: &Overlay) -> Option<Vec<TaskId>> {
    fn reach(id: TaskId, via: Option<TaskId>, pending: &mut Vec<TaskId>, from: &mut HashMap<TaskId, Option<TaskId>>) {
        if let Entry::Vacant(entry) = from.entry(id) {
            entry.insert(via);
            pending.push(id);
        }
    }

    let mut from: HashMap<TaskId, Option<TaskId>> = HashMap::new();
    let mut pending = Vec::new();
    let mut walked: HashSet<TaskId> = HashSet::new();
    for &id in on {
        reach(id, None, &mut pending, &mut from);
    }
    while let Some(id) = pending.pop() {
        if id == task {
            let mut path = vec![id];
            let mut at = from.get(&id).copied().flatten();
            while let Some(step) = at {
                path.push(step);
                at = from.get(&step).copied().flatten();
            }
            path.reverse();
            return Some(path);
        }
        let record = overlay.tasks.get(&id).or_else(|| tree.live.get(&id));
        let Some(record) = record.filter(|record| record.state.status() != TaskStatus::Terminal) else {
            continue;
        };
        // An abort-marked task waits only for its owned work, as abort runs bottom-up.
        if let TaskState::Waiting { on, .. } = &record.state
            && !record.abort_requested
        {
            for &member in on {
                reach(member, Some(id), &mut pending, &mut from);
            }
        }
        if !walked.insert(id) {
            continue;
        }
        // Below a walked task the walk does not go again: its own walk covered it.
        let mut found = Vec::new();
        {
            let enter = |below: TaskId| !walked.contains(&below);
            let walk = Walk {
                overlay: Some(overlay),
                cross_background: false,
                enter: Some(&enter),
            };
            tree.walk_down(Node::Task(id), walk, |reached| {
                if reached.live.is_some() && !reached.node.background {
                    found.push(reached.id);
                }
                ControlFlow::Continue(())
            });
        }
        for below in found {
            reach(below, Some(id), &mut pending, &mut from);
            walked.insert(below);
        }
    }
    None
}
