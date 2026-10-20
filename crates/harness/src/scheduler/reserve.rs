//! Reservation: one commit that turns every eligible task `running` and registers its
//! invocation, resolving definitions and migrating older versions (spec §5.4).
//!
//! ```text
//! restarts >= restart_limit          -> pending, blocked: restart_limit
//! no definition                      -> pending, blocked: missing_task
//! stored version > definition's      -> pending, blocked: task_too_old
//! stored version < definition's      -> migrate (failure: blocked: migration_failed)
//! equal, or migrated                 -> running, invocation registered
//! ```

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use super::Inner;
use super::catalog::CatalogSnapshot;
use super::erased::{AnyTask, same_definition};
use super::error::Failure;
use super::invocation::{Invocation, Mode, Reservation};
use super::ownership::TaskNode;
use super::scopes::load_scopes;
use super::state::State;
use super::transition::{stage, terminate, with_abort_mark, with_state};
use crate::records::{AbortReason, TaskId, TaskOutcome, TaskOutcomeError, TaskRecord, TaskState, TaskStatus};
use crate::session::{FaultReason, ReportInfo, ReportTask, SessionError};

/// Why a live task cannot be reserved. Derived from the catalog, the settings and the
/// record's restart count; never stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlockedReason {
    /// No definition of the task's kind is installed.
    MissingTask,
    /// The stored version is newer than the installed definition.
    TaskTooOld,
    /// The installed definition is newer and cannot migrate the record.
    MigrationFailed,
    /// Cut off `restart_limit` times in a row; `retry_task` runs it again.
    RestartLimit,
}

impl BlockedReason {
    /// The snake_case name, also the reason of an `orphaned` outcome.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MissingTask => "missing_task",
            Self::TaskTooOld => "task_too_old",
            Self::MigrationFailed => "migration_failed",
            Self::RestartLimit => "restart_limit",
        }
    }
}

/// A definition that can take a record, or why none can. Deciding it runs no task code.
pub(super) enum Fit {
    Ready { task: Arc<dyn AnyTask>, migrates: bool },
    Blocked(BlockedReason),
}

/// A record whose restart count reached the limit; an abort mark still runs its handler.
pub(super) fn parked(record: &TaskRecord, limit: Option<u32>) -> bool {
    !record.abort_requested && limit.is_some_and(|limit| record.restarts.is_some_and(|restarts| restarts >= limit))
}

impl State {
    pub(super) fn fit(&self, record: &TaskRecord, task: Option<&Arc<dyn AnyTask>>) -> Fit {
        let Some(task) = task else {
            return Fit::Blocked(BlockedReason::MissingTask);
        };
        let version = task.version();
        if version == record.version {
            return Fit::Ready {
                task: task.clone(),
                migrates: false,
            };
        }
        if version < record.version {
            return Fit::Blocked(BlockedReason::TaskTooOld);
        }
        if self
            .failed_migrations
            .get(&record.id)
            .is_some_and(|failed| same_definition(failed, task))
        {
            return Fit::Blocked(BlockedReason::MigrationFailed);
        }
        Fit::Ready {
            task: task.clone(),
            migrates: true,
        }
    }

    /// The definition that takes `record`, with the record migrated to its version. A
    /// failed migration is remembered for that definition and reported once.
    pub(super) fn resolve(
        &mut self,
        record: &TaskRecord,
        snapshot: &dyn CatalogSnapshot,
        reports: &mut Vec<(Failure, ReportInfo)>,
    ) -> Result<(Arc<dyn AnyTask>, TaskRecord), BlockedReason> {
        let (task, migrates) = match self.fit(record, snapshot.task(&record.kind)) {
            Fit::Blocked(reason) => return Err(reason),
            Fit::Ready { task, migrates } => (task, migrates),
        };
        if !migrates {
            return Ok((task, record.clone()));
        }
        match migrate(&*task, record) {
            Ok(migrated) => Ok((task, migrated)),
            Err(message) => {
                self.failed_migrations.insert(record.id, task.clone());
                let report = ReportTask {
                    id: record.id,
                    kind: record.kind.clone(),
                    conversation_id: record.conversation_id,
                    extension: snapshot.extension_of(&task),
                };
                reports.push((Failure(message), ReportInfo::MigrationFailed { task: report }));
                Err(BlockedReason::MigrationFailed)
            }
        }
    }
}

/// The record migrated to `task`'s version; a panic in the definition is a failure.
fn migrate(task: &dyn AnyTask, record: &TaskRecord) -> Result<TaskRecord, String> {
    if !task.can_migrate() {
        return Err(format!(
            "Task {} version {} has no migration from {}",
            record.kind,
            task.version(),
            record.version
        ));
    }
    let checkpoint = record.state.checkpoint().cloned().unwrap_or_default();
    let call = || task.migrate_json(record.input.clone(), checkpoint, record.version);
    match std::panic::catch_unwind(AssertUnwindSafe(call)) {
        Ok(Ok((input, checkpoint))) => {
            let mut migrated = record.clone();
            migrated.version = task.version();
            migrated.input = input;
            migrated.state = with_checkpoint(&record.state, checkpoint);
            Ok(migrated)
        }
        Ok(Err(error)) => Err(format!(
            "Task {} migration from {} failed: {error}",
            record.kind, record.version
        )),
        Err(_) => Err(format!(
            "Task {} migration from {} panicked",
            record.kind, record.version
        )),
    }
}

/// A live state with another checkpoint.
fn with_checkpoint(state: &TaskState, checkpoint: delta_chord::json::Json) -> TaskState {
    match state {
        TaskState::Waiting { on, policy, .. } => TaskState::Waiting {
            checkpoint,
            on: on.clone(),
            policy: *policy,
        },
        TaskState::Sleeping { until, .. } => TaskState::Sleeping {
            checkpoint,
            until: *until,
        },
        TaskState::Running { .. } => TaskState::Running { checkpoint },
        _ => TaskState::Pending { checkpoint },
    }
}

/// What one candidate of a pass asks for.
enum Action {
    /// An abort handler cut off `restart_limit` times: faulted.
    Fault(TaskRecord, String),
    /// An abort request on a task no definition can take: orphaned.
    Orphan(TaskRecord, BlockedReason),
}

/// Reserves every eligible task in one commit; orphans abort-marked tasks no definition
/// can take, unless abandoned after a restart. The first pass after open only marks the
/// abandoned tasks, so later passes see the marks.
pub(super) async fn reserve(inner: &Arc<Inner>) -> Result<Vec<Reservation>, SessionError> {
    let shared = inner.clone();
    let limit = inner.settings().restart_limit;
    let pass = inner
        .session
        .commit(&inner.ctx, move |tx| {
            Box::pin(async move {
                let inner = shared;
                let mut pass = Pass::default();
                {
                    let mut state = inner.shared.lock();
                    state.tree.sweep();
                    if !state.enabled || state.closing {
                        return Ok(pass);
                    }
                    if !state.abandoned.is_empty() {
                        pass.abandoned = std::mem::take(&mut state.abandoned).into_iter().collect();
                        for id in &pass.abandoned {
                            if let Some(record) = state.tree.live.get(id).filter(|record| !record.abort_requested) {
                                tx.set_task(with_abort_mark(record, Some(AbortReason::Restart)))?;
                            }
                        }
                        return Ok(pass);
                    }
                }
                load_scopes(&inner.shared, tx, false).await?;
                let snapshot = inner.catalog.snapshot();
                let now = inner.session.now();
                let mut actions = Vec::new();
                {
                    let mut state = inner.shared.lock();
                    state.wake_due(now);
                    state.unpark(limit, &inner.shared);
                    let candidates: Vec<TaskId> = state.runnable.iter().copied().collect();
                    for id in candidates {
                        if let Some(action) = state.consider(id, limit, now, &*snapshot, tx, &mut pass)? {
                            actions.push(action);
                        }
                    }
                }
                for action in actions {
                    match action {
                        Action::Fault(record, message) => {
                            let outcome = TaskOutcome::Faulted {
                                error: TaskOutcomeError {
                                    message: message.clone(),
                                    detail: Some(delta_chord::json!({ "reason": "restart_limit" })),
                                },
                            };
                            terminate(&inner, tx, &record, outcome).await?;
                            let task = report_task(&record, &*snapshot);
                            let info = ReportInfo::TaskFaulted {
                                task,
                                reason: FaultReason::RestartLimit,
                            };
                            pass.reports.push((Failure(message), info));
                        }
                        Action::Orphan(record, reason) => {
                            let outcome = TaskOutcome::Orphaned {
                                reason: reason.as_str().to_owned(),
                            };
                            terminate(&inner, tx, &record, outcome).await?;
                        }
                    }
                }
                pass.snapshot = Some(snapshot);
                Ok(pass)
            })
        })
        .await;
    let pass = match pass {
        Ok(pass) => pass,
        Err(error) => {
            // Nothing was reserved: the invocations the pass registered go.
            let mut state = inner.shared.lock();
            for invocation in std::mem::take(&mut state.reserving) {
                state.invocations.remove(&invocation.task);
                state.spawned.remove(&invocation.id);
                state.runnable.insert(invocation.task);
                invocation.finish();
            }
            return Err(error);
        }
    };
    inner.shared.lock().reserving.clear();
    for (error, info) in &pass.reports {
        inner.session.report(error, info);
    }
    if pass.blocked_changed {
        inner.shared.bump_blocked();
    }
    if !pass.abandoned.is_empty() {
        // Marked, or already marked or gone: reserve again, now seeing the marks.
        inner.shared.kick();
    }
    let Some(snapshot) = pass.snapshot else {
        return Ok(Vec::new());
    };
    Ok(pass
        .reserved
        .into_iter()
        .map(|(invocation, definition)| Reservation {
            invocation,
            definition,
            snapshot: snapshot.clone(),
        })
        .collect())
}

fn report_task(record: &TaskRecord, snapshot: &dyn CatalogSnapshot) -> ReportTask {
    ReportTask {
        id: record.id,
        kind: record.kind.clone(),
        conversation_id: record.conversation_id,
        extension: snapshot.task(&record.kind).and_then(|task| snapshot.extension_of(task)),
    }
}

/// What a reservation pass did.
#[derive(Default)]
struct Pass {
    reserved: Vec<(Arc<Invocation>, Arc<dyn AnyTask>)>,
    abandoned: Vec<TaskId>,
    reports: Vec<(Failure, ReportInfo)>,
    blocked_changed: bool,
    snapshot: Option<Arc<dyn CatalogSnapshot>>,
}

impl State {
    /// Makes due sleeps reservation candidates.
    pub(super) fn wake_due(&mut self, now: i64) {
        let due: Vec<TaskId> = self
            .sleeping
            .iter()
            .copied()
            .filter(|id| self.tree.live.get(id).is_some_and(|record| !not_due(record, now)))
            .collect();
        self.runnable.extend(due);
    }

    /// Makes parked tasks a raised limit no longer parks candidates again.
    fn unpark(&mut self, limit: Option<u32>, shared: &super::state::Shared) {
        let candidates: Vec<TaskId> = self.parked.iter().copied().collect();
        let mut unparked = false;
        for id in candidates {
            match self.tree.live.get(&id) {
                Some(record) if parked(record, limit) => {}
                Some(_) => {
                    self.parked.remove(&id);
                    self.runnable.insert(id);
                    unparked = true;
                }
                None => {
                    self.parked.remove(&id);
                }
            }
        }
        if unparked {
            shared.bump_blocked();
        }
    }

    /// Decides one candidate: leave it, reserve it (staging `running`), or an outcome.
    fn consider(
        &mut self,
        id: TaskId,
        limit: Option<u32>,
        now: i64,
        snapshot: &dyn CatalogSnapshot,
        tx: &mut crate::session::Tx<'_>,
        pass: &mut Pass,
    ) -> Result<Option<Action>, crate::session::TxError> {
        let Some(record) = self.tree.live.get(&id).cloned() else {
            self.runnable.remove(&id);
            return Ok(None);
        };
        if record.state.status() == TaskStatus::Completing || self.invocations.contains_key(&id) {
            self.runnable.remove(&id);
            return Ok(None);
        }
        // It returns once what it waits for ends, its record changes, or the work below ends.
        if !self.waiting_on(&record, true).is_empty() || not_due(&record, now) {
            self.runnable.remove(&id);
            return Ok(None);
        }
        if parked(&record, limit) {
            self.runnable.remove(&id);
            if self.parked.insert(id) {
                pass.blocked_changed = true;
            }
            return Ok(None);
        }
        let mode = if record.abort_requested { Mode::Abort } else { Mode::Run };
        // Work below an owner with cancellation intent waits for its cascade mark.
        if mode == Mode::Run && !record.background && self.tree.below_cancelled(TaskNode::of(&record).parent()) {
            self.cascade_pending = true;
            self.reconcile_scheduled = true;
            return Ok(None);
        }
        if mode == Mode::Abort && limit.is_some_and(|limit| record.restarts.unwrap_or(0) >= limit) {
            self.runnable.remove(&id);
            let message = format!(
                "Abort handler of task {id} ({}) was cut off {} times",
                record.kind,
                record.restarts.unwrap_or(0)
            );
            return Ok(Some(Action::Fault(record, message)));
        }
        let (task, resolved) = match self.resolve(&record, snapshot, &mut pass.reports) {
            Ok(resolved) => resolved,
            Err(reason) => {
                if reason == BlockedReason::MigrationFailed {
                    pass.blocked_changed = true;
                }
                // An abandoned task waits for its definition, so its handler can clean up.
                if mode == Mode::Abort && record.abort_reason.is_none() {
                    self.runnable.remove(&id);
                    return Ok(Some(Action::Orphan(record, reason)));
                }
                return Ok(None);
            }
        };
        if resolved != record || record.state.status() != TaskStatus::Running {
            let checkpoint = resolved.state.checkpoint().cloned().unwrap_or_default();
            stage(tx, with_state(&resolved, TaskState::Running { checkpoint }))?;
        }
        self.runnable.remove(&id);
        self.next_invocation += 1;
        let extension = snapshot.extension_of(&task);
        let invocation = Arc::new(Invocation::new(self.next_invocation, &resolved, mode, extension));
        self.invocations.insert(id, invocation.clone());
        self.spawned.insert(invocation.id, invocation.clone());
        self.reserving.push(invocation.clone());
        pass.reserved.push((invocation, task));
        Ok(None)
    }

    /// Live tasks a task waits for before its next invocation: its live ordinary owned work
    /// when abort-marked (abort runs bottom-up), otherwise the live part of its wait's `on`.
    pub(super) fn waiting_on(&self, record: &TaskRecord, first: bool) -> Vec<TaskId> {
        if record.abort_requested {
            return self.tree.owned_live(record.id, None, first, None);
        }
        let TaskState::Waiting { on, .. } = &record.state else {
            return Vec::new();
        };
        let mut live = on.iter().copied().filter(|id| self.tree.live.contains_key(id));
        if first {
            live.next().into_iter().collect()
        } else {
            live.collect()
        }
    }
}

/// A sleep whose `until` the clock has not reached; an abort mark ends a sleep.
pub(super) fn not_due(record: &TaskRecord, now: i64) -> bool {
    matches!(record.state, TaskState::Sleeping { until, .. } if until > now) && !record.abort_requested
}
