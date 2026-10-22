//! The scheduler: the runtime of durable tasks (spec §5).
//!
//! A task is a state machine whose record holds a phase checkpoint. The scheduler reserves
//! runnable tasks in commits on the Session line, runs their phase handlers off it on tokio
//! tasks, and every phase ends with the [`Next`] commit its handler returns, which is also
//! the step that applies the precedence rules; a reopened storage reruns the current phase
//! from its checkpoint. On top sit ownership (tasks own tasks and conversations), abort
//! cascades, `completing` holds, structured waits, durable sleeps, restarts and parking.
//!
//! The scheduler knows no task kind: every task, the agent layer's included, is a
//! [`TaskDefinition`] found by name in the [`TaskCatalog`] the layer above supplies, and
//! that layer gives it [`SchedulerHooks`] for the outcomes the scheduler writes itself.
//!
//! Corresponds to upstream `src/harness/scheduler.ts` and `src/tasks.ts`. See
//! `docs/design/scheduler.md`.
//!
//! ```text
//! open ──► reconcile (marks, fail_fast, withdrawals, finalize) ──┐
//!            ▲  observer: every adopted commit                    │
//! resume ─► reserve (pending/waiting/sleeping -> running) ──► invocation (tokio task)
//!                                                                 │ step ─ phase ─ step …
//! close ──► seal, cancel, join (close_grace), write back pending ◄┘
//! ```

mod abort;
mod catalog;
mod definition;
mod erased;
mod error;
mod inspect;
mod invocation;
mod next;
mod ownership;
mod recovery;
mod reserve;
mod runtime;
mod scopes;
mod state;
mod step;
mod timers;
mod transition;
mod waits;

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use delta_chord::context::Context;
use futures::future::BoxFuture;
use tokio::task::JoinHandle;

pub use abort::TaskAbort;
pub use catalog::{CatalogSnapshot, MemoryCatalog, TaskCatalog};
pub use definition::{Checkpoint, RunningTask, TaskDefinition};
pub use erased::AnyTask;
pub use error::{RuntimeError, SchedulerError, StateError, TaskError};
pub use inspect::{BlockedProbe, InspectState, Inspection, Scheduling, TaskInspection};
pub use next::{Next, NextOf, NextState, Outcome};
pub use ownership::IndexSizes;
pub use reserve::BlockedReason;
pub use runtime::{PhaseData, ReportFrom, TaskRuntime, Timer};
pub use timers::Wake;
pub use waits::SettledTask;

use invocation::Invocation;
use state::Shared;

use crate::records::{ConversationId, SubmissionRecord, SubmissionSettlement, TaskId, TaskRecord};
use crate::session::{ObserverGuard, Session, SessionError, Tx, TxError};

/// The default restart limit: a task cut off this many times in a row is parked.
pub const DEFAULT_RESTART_LIMIT: u32 = 5;

/// How long close waits for handlers that ignore their context before dropping them.
pub const DEFAULT_CLOSE_GRACE: Duration = Duration::from_secs(30);

/// The scheduler's settings, read at each use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchedulerSettings {
    /// Restarts in a row that park a task; `None` never parks.
    pub restart_limit: Option<u32>,
    /// How long close waits for invocations before aborting their code.
    pub close_grace: Duration,
}

impl Default for SchedulerSettings {
    fn default() -> Self {
        Self {
            restart_limit: Some(DEFAULT_RESTART_LIMIT),
            close_grace: DEFAULT_CLOSE_GRACE,
        }
    }
}

/// The cleanup the layer above performs in the scheduler's own commits. Every method has a
/// default; the kernel calls them for every task alike.
pub trait SchedulerHooks: Send + Sync + 'static {
    /// Runs in the commit that makes an outcome the scheduler wrote (`faulted`, `orphaned`)
    /// terminal, which is the final commit after a hold; `record` is the terminal record.
    /// The commit has no task scope. The agent layer cleans up what the task's own handlers
    /// would have (a run's marker and inputs, a tool's slot).
    fn settle_scheduler_outcome<'a>(
        &'a self,
        tx: &'a mut Tx<'_>,
        record: &'a TaskRecord,
    ) -> BoxFuture<'a, Result<(), TxError>> {
        let _ = (tx, record);
        Box::pin(async { Ok(()) })
    }

    /// Withdraws the queued inputs of a conversation a cascade or a conversation abort
    /// reached. By default each settles `unanswered` with reason `aborted`; the agent layer
    /// also removes them from its inbox.
    fn withdraw_inputs<'a>(
        &'a self,
        tx: &'a mut Tx<'_>,
        conversation: ConversationId,
        queued: &'a [SubmissionRecord],
    ) -> BoxFuture<'a, Result<(), TxError>> {
        let _ = conversation;
        for submission in queued {
            tx.settle_submission(
                submission.id,
                SubmissionSettlement::Unanswered {
                    reason: "aborted".to_owned(),
                    detail: None,
                },
            );
        }
        Box::pin(async { Ok(()) })
    }
}

/// [`SchedulerHooks`] with every default.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultHooks;

impl SchedulerHooks for DefaultHooks {}

/// Reads the settings at each use.
pub type SettingsSource = Arc<dyn Fn() -> SchedulerSettings + Send + Sync>;

/// Called off the line with the [`Wake`] after open and whenever it changes.
pub type WakeListener = Arc<dyn Fn(Wake) + Send + Sync>;

/// How a scheduler runs.
pub struct SchedulerOptions {
    pub catalog: Arc<dyn TaskCatalog>,
    pub hooks: Arc<dyn SchedulerHooks>,
    pub settings: SettingsSource,
    /// With a listener the scheduler arms no timer for sleeps; the host wakes it with
    /// [`Scheduler::resume`].
    pub on_wake: Option<WakeListener>,
}

impl SchedulerOptions {
    /// Default hooks and settings over `catalog`.
    pub fn new(catalog: Arc<dyn TaskCatalog>) -> Self {
        Self {
            catalog,
            hooks: Arc::new(DefaultHooks),
            settings: Arc::new(SchedulerSettings::default),
            on_wake: None,
        }
    }
}

impl fmt::Debug for SchedulerOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SchedulerOptions")
            .field("settings", &(self.settings)())
            .field("on_wake", &self.on_wake.is_some())
            .finish_non_exhaustive()
    }
}

/// What the scheduler's parts share.
pub(crate) struct Inner {
    session: Session,
    shared: Arc<Shared>,
    catalog: Arc<dyn TaskCatalog>,
    hooks: Arc<dyn SchedulerHooks>,
    settings: SettingsSource,
    on_wake: Option<WakeListener>,
    /// The context of the scheduler's own commits: never cancelled.
    ctx: Context,
    _observer: ObserverGuard,
}

impl Inner {
    fn settings(&self) -> SchedulerSettings {
        (self.settings)()
    }

    fn wake(&self) -> Wake {
        let snapshot = self.catalog.snapshot();
        let limit = self.settings().restart_limit;
        self.shared.lock().wake(self.session.now(), &*snapshot, limit)
    }

    /// Fails the Session for an error of one of the scheduler's own commits: memory would
    /// be behind storage. Nothing when the Session is closing or already failed.
    fn fail(&self, error: SessionError) {
        if matches!(error, SessionError::Closed | SessionError::Failed(_)) || self.shared.lock().closing {
            return;
        }
        self.session.fail(Arc::new(error));
    }

    /// Seals reservation, once: later runtime operations reject, waits end, and every
    /// active invocation is signalled. Returns the invocations active at the seal.
    fn seal(&self) -> Vec<Arc<Invocation>> {
        let (stopped, watches) = {
            let mut state = self.shared.lock();
            if !state.closing {
                state.closing = true;
                state.stopped = state.invocations.values().cloned().collect();
            }
            (state.stopped.clone(), std::mem::take(&mut state.task_watches))
        };
        // Dropping the senders ends the task waits; idle waits see `closing`.
        drop(watches);
        self.shared.bump_changes();
        self.shared.wake_driver();
        for invocation in &stopped {
            invocation.signal("is stopped: the scheduler is closing");
        }
        stopped
    }
}

/// Stops the driver and drops the invocations' code when the last handle goes without a
/// close, as a crash would.
struct Owner {
    inner: Arc<Inner>,
    driver: JoinHandle<()>,
}

impl Drop for Owner {
    fn drop(&mut self) {
        self.driver.abort();
        let spawned: Vec<Arc<Invocation>> = {
            let mut state = self.inner.shared.lock();
            state.closing = true;
            state.spawned.values().cloned().collect()
        };
        for invocation in spawned {
            invocation.abort_code();
        }
    }
}

/// The scheduler of one Session. Cheap to clone; dropping the last handle without
/// [`close`](Self::close) stops everything where it is, as a crash: the next open counts
/// a restart for every task that was running.
#[derive(Clone)]
pub struct Scheduler {
    owner: Arc<Owner>,
}

impl fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.inner().shared.lock();
        f.debug_struct("Scheduler")
            .field("enabled", &state.enabled)
            .field("closing", &state.closing)
            .field("live", &state.tree.live.len())
            .field("invocations", &state.invocations.len())
            .finish_non_exhaustive()
    }
}

impl Scheduler {
    /// Opens the scheduler of `session`: loads the live tasks, writes surviving `running`
    /// tasks back as `pending` with a restart counted, and schedules the reconcile pass
    /// that re-derives cascades. Nothing runs until [`resume`](Self::resume).
    pub async fn open(session: Session, options: SchedulerOptions, ctx: &Context) -> Result<Scheduler, SchedulerError> {
        let SchedulerOptions {
            catalog,
            hooks,
            settings,
            on_wake,
        } = options;
        let shared = Arc::new(Shared::new());
        let observer = session.observe_commits(Box::new(state::Observer::new(shared.clone())));
        let catalog_changes = catalog.subscribe();
        let close = session.subscribe_close();
        let inner = Arc::new(Inner {
            session: session.clone(),
            shared: shared.clone(),
            catalog,
            hooks,
            settings,
            on_wake,
            ctx: Context::background(),
            _observer: observer,
        });
        let barrier = Arc::downgrade(&inner);
        session.set_close_barrier(Box::new(move || Box::pin(recovery::close(barrier))));
        let driver = tokio::spawn(timers::drive(Arc::downgrade(&inner), shared, catalog_changes, close));
        let scheduler = Scheduler {
            owner: Arc::new(Owner {
                inner: inner.clone(),
                driver,
            }),
        };
        recovery::open(&inner, ctx).await?;
        Ok(scheduler)
    }

    fn inner(&self) -> &Arc<Inner> {
        &self.owner.inner
    }

    /// The Session the scheduler runs on.
    pub fn session(&self) -> &Session {
        &self.inner().session
    }

    /// Enables scheduling and makes due sleeps runnable. Idempotent; a host wakeup calls it.
    pub fn resume(&self) -> Result<(), SchedulerError> {
        let shared = &self.inner().shared;
        {
            let mut state = shared.lock();
            if state.closing {
                return Err(SchedulerError::Closed);
            }
            state.enabled = true;
            state.wake_due(self.session().now());
            state.dirty = true;
        }
        shared.wake_driver();
        Ok(())
    }

    /// When the Session next needs a running process, from memory.
    pub fn wake(&self) -> Wake {
        self.inner().wake()
    }

    /// Commits an abort mark, or settles a blocked task `orphaned` at once, then joins the
    /// active run invocation. `Terminal` for a terminal task; a `completing` task is only
    /// marked. Cancelling `ctx` stops the wait, not the abort.
    pub async fn abort_task(&self, id: TaskId, ctx: &Context) -> Result<TaskAbort, SchedulerError> {
        abort::abort(self.inner(), id, ctx, false).await
    }

    /// Withdraws a conversation's queued inputs and marks the live non-background tasks
    /// ordinary traversal reaches from it; with `background`, across background boundaries,
    /// waiting for every task it marked. Resolves once the conversation is idle.
    pub async fn abort_conversation(
        &self,
        conversation: ConversationId,
        background: bool,
        ctx: &Context,
    ) -> Result<(), SchedulerError> {
        abort::abort_conversation(self.inner(), conversation, background, ctx).await
    }

    /// Clears the restart count of a parked task and resumes scheduling; nothing for a task
    /// that is not parked.
    pub async fn retry_task(&self, id: TaskId, ctx: &Context) -> Result<(), SchedulerError> {
        abort::retry(self.inner(), id, ctx).await?;
        self.resume()
    }

    /// Resolves with the task's terminal record once it is terminal.
    pub async fn wait_for_task(&self, id: impl Into<TaskId>, ctx: &Context) -> Result<SettledTask, SchedulerError> {
        waits::wait_for_task(self.inner(), id.into(), ctx).await
    }

    /// Resolves once ordinary traversal from `conversation` (every ownerless conversation
    /// with `None`) reaches no live non-background task.
    pub async fn wait_for_idle(
        &self,
        conversation: Option<ConversationId>,
        ctx: &Context,
    ) -> Result<(), SchedulerError> {
        waits::wait_for_idle(self.inner(), conversation, ctx).await
    }

    /// Every live task with its derived state, and whether scheduling runs.
    pub async fn inspect(&self, ctx: &Context) -> Result<Inspection, SchedulerError> {
        inspect::inspect(self.inner(), ctx).await
    }

    /// Why a live task cannot run with the current definitions, if it cannot (the task
    /// graph shows it).
    pub fn blocked_reason(&self, record: &TaskRecord) -> Option<BlockedReason> {
        inspect::blocked(self.inner(), record)
    }

    /// A handle answering [`blocked_reason`](Self::blocked_reason) that does not keep the
    /// scheduler alive, for an observer on the commit line.
    pub fn blocked_probe(&self) -> BlockedProbe {
        BlockedProbe(Arc::downgrade(self.inner()))
    }

    /// Changes when blocked reasons may have changed without a commit.
    pub fn subscribe_blocked(&self) -> tokio::sync::watch::Receiver<u64> {
        self.inner().shared.blocked.subscribe()
    }

    /// Sizes of the ownership indexes, for tests that bound the scheduler's memory.
    #[doc(hidden)]
    pub fn index_sizes(&self) -> IndexSizes {
        self.inner().shared.lock().tree.sizes()
    }

    /// Closes the Session: seals admission, stops and joins every invocation, writes the
    /// tasks it stopped back as `pending`, then closes storage. Writes no outcome and sets
    /// no mark. Once it returns no invocation code of this scheduler runs.
    pub async fn close(&self, ctx: &Context) -> Result<(), SchedulerError> {
        self.session().close(ctx).await?;
        Ok(())
    }
}
