//! Inspection: every live task with its derived state, read on the line. No writes, no
//! scheduling, no task code: a pending migration shows as `Ready { migrates }`, and only a
//! migration already tried, or one that cannot exist, as blocked.

use std::fmt;
use std::sync::{Arc, Weak};

use delta_chord::context::Context;

use super::Inner;
use super::catalog::CatalogSnapshot;
use super::error::SchedulerError;
use super::reserve::{BlockedReason, Fit, not_due, parked};
use super::scopes::load_scopes;
use super::state::State;
use crate::records::{TaskId, TaskRecord, TaskState, TaskStatus};

/// Whether the scheduler starts work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheduling {
    /// Opened, not resumed: nothing runs.
    Paused,
    Running,
    Closing,
}

/// A live task's derived state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InspectState {
    /// An invocation runs it.
    Running,
    /// The next pass may reserve it; `migrates` when its record is of an older version.
    Ready {
        migrates: bool,
    },
    /// Waits for the live part of its `on`, or (abort-marked) for its live owned work.
    Waiting {
        on: Vec<TaskId>,
    },
    Sleeping {
        until: i64,
    },
    /// Holds its outcome until its owned work drained.
    Completing,
    Blocked {
        reason: BlockedReason,
    },
}

/// One live task of an [`Inspection`].
#[derive(Clone, Debug, PartialEq)]
pub struct TaskInspection {
    pub record: TaskRecord,
    pub state: InspectState,
}

/// What [`Scheduler::inspect`](super::Scheduler::inspect) returns.
#[derive(Clone, Debug, PartialEq)]
pub struct Inspection {
    pub scheduling: Scheduling,
    /// Every live task, in id order.
    pub tasks: Vec<TaskInspection>,
}

impl State {
    fn inspect_task(
        &self,
        record: &TaskRecord,
        snapshot: &dyn CatalogSnapshot,
        now: i64,
        limit: Option<u32>,
    ) -> InspectState {
        if self.invocations.contains_key(&record.id) {
            return InspectState::Running;
        }
        if record.state.status() == TaskStatus::Completing {
            return InspectState::Completing;
        }
        let on = self.waiting_on(record, false);
        if !on.is_empty() {
            return InspectState::Waiting { on };
        }
        if let TaskState::Sleeping { until, .. } = record.state
            && not_due(record, now)
        {
            return InspectState::Sleeping { until };
        }
        match self.blocked_reason(record, snapshot, limit) {
            Some(reason) => InspectState::Blocked { reason },
            None => InspectState::Ready {
                migrates: snapshot
                    .task(&record.kind)
                    .is_some_and(|task| task.version() != record.version),
            },
        }
    }

    /// Why a live task cannot run with these definitions, if it cannot. A running task, an
    /// abort request and a held outcome are never blocked; a `restart` mark waits for its
    /// definition, blocked.
    pub(super) fn blocked_reason(
        &self,
        record: &TaskRecord,
        snapshot: &dyn CatalogSnapshot,
        limit: Option<u32>,
    ) -> Option<BlockedReason> {
        let status = record.state.status();
        if matches!(
            status,
            TaskStatus::Running | TaskStatus::Completing | TaskStatus::Terminal
        ) {
            return None;
        }
        if record.abort_requested && record.abort_reason.is_none() {
            return None;
        }
        if parked(record, limit) {
            return Some(BlockedReason::RestartLimit);
        }
        match self.fit(record, snapshot.task(&record.kind)) {
            Fit::Blocked(reason) => Some(reason),
            Fit::Ready { task, migrates } => {
                (migrates && !task.can_migrate()).then_some(BlockedReason::MigrationFailed)
            }
        }
    }
}

/// Every live task with its derived state; loads owner chains first, writes nothing.
pub(super) async fn inspect(inner: &Arc<Inner>, ctx: &Context) -> Result<Inspection, SchedulerError> {
    let shared = inner.clone();
    let limit = inner.settings().restart_limit;
    let inspection = inner
        .session
        .commit(ctx, move |tx| {
            Box::pin(async move {
                load_scopes(&shared.shared, tx, false).await?;
                let snapshot = shared.catalog.snapshot();
                let now = shared.session.now();
                let state = shared.shared.lock();
                let scheduling = if state.closing {
                    Scheduling::Closing
                } else if state.enabled {
                    Scheduling::Running
                } else {
                    Scheduling::Paused
                };
                let tasks = state
                    .tree
                    .live
                    .values()
                    .map(|record| TaskInspection {
                        record: record.clone(),
                        state: state.inspect_task(record, &*snapshot, now, limit),
                    })
                    .collect();
                Ok(Inspection { scheduling, tasks })
            })
        })
        .await?;
    Ok(inspection)
}

/// Why the live task `record` cannot run with the installed definitions and the settings,
/// if it cannot; reads the scheduler's memory only, so it may run on the line.
pub(super) fn blocked(inner: &Inner, record: &TaskRecord) -> Option<BlockedReason> {
    let snapshot = inner.catalog.snapshot();
    let limit = inner.settings().restart_limit;
    inner.shared.lock().blocked_reason(record, &*snapshot, limit)
}

/// Answers [`Scheduler::blocked_reason`](super::Scheduler::blocked_reason) without keeping the scheduler
/// alive: an observer of the commit line holds one.
#[derive(Clone)]
pub struct BlockedProbe(pub(super) Weak<Inner>);

impl BlockedProbe {
    /// Why `record` cannot run, if it cannot; `None` once the scheduler is gone.
    pub fn blocked(&self, record: &TaskRecord) -> Option<BlockedReason> {
        self.0.upgrade().and_then(|inner| blocked(&inner, record))
    }
}

impl fmt::Debug for BlockedProbe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlockedProbe").finish_non_exhaustive()
    }
}
