//! Invocations: one in-memory execution of a task in run or abort mode, on its own tokio
//! task, with its own [`Context`]. A run invocation runs phases in sequence, each preceded
//! by a step on the line; an abort invocation runs the abort handler once.

use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use delta_chord::context::{CancelReason, Canceller, Context};
use delta_chord::json::Json;
use futures::FutureExt;
use tokio::sync::watch;
use tokio::task::AbortHandle;

use super::Inner;
use super::catalog::CatalogSnapshot;
use super::erased::{AnyTask, PhaseCall};
use super::next::ErasedNext;
use super::runtime::RuntimeCore;
use super::step::{self, Phase};
use crate::records::{ConversationId, TaskId, TaskRecord};
use crate::session::ReportTask;

/// What an invocation runs: phases, or the abort handler once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    Run,
    Abort,
}

/// One execution of a task. Registered on the line at reservation; ended inside the step
/// that decides its end, so a runtime commit it queued either lands before that decision
/// or is refused.
pub(crate) struct Invocation {
    pub(super) id: u64,
    pub(super) task: TaskId,
    pub(super) kind: String,
    pub(super) conversation: ConversationId,
    pub(super) mode: Mode,
    /// The extension that installed the definition it runs, for reports.
    pub(super) extension: Option<String>,
    /// Handed to its handlers; cancelled by an abort mark (run mode), close and its end.
    pub(super) ctx: Context,
    canceller: Canceller,
    ended: AtomicBool,
    /// `true` once its code finished (or was dropped).
    done: watch::Sender<bool>,
    join: Mutex<Option<AbortHandle>>,
}

impl Invocation {
    pub(super) fn new(id: u64, record: &TaskRecord, mode: Mode, extension: Option<String>) -> Self {
        let (ctx, canceller) = Context::background().with_cancel();
        Self {
            id,
            task: record.id,
            kind: record.kind.clone(),
            conversation: record.conversation_id,
            mode,
            extension,
            ctx,
            canceller,
            ended: AtomicBool::new(false),
            done: watch::Sender::new(false),
            join: Mutex::new(None),
        }
    }

    pub(super) fn is_ended(&self) -> bool {
        self.ended.load(Ordering::Acquire)
    }

    /// Cancels the handlers' context: an abort mark or close.
    pub(super) fn signal(&self, reason: &str) {
        self.canceller
            .cancel_with(CancelReason::message(format!("Task {} {reason}", self.task)));
    }

    /// Resolves once the invocation's code finished.
    pub(super) async fn finished(&self) {
        let mut done = self.done.subscribe();
        let _ = done.wait_for(|done| *done).await;
    }

    /// Drops the invocation's code at its next yield (close after the grace, a crash).
    pub(super) fn abort_code(&self) {
        if let Some(join) = self.join.lock().unwrap_or_else(PoisonError::into_inner).as_ref() {
            join.abort();
        }
    }

    pub(super) fn report_task(&self) -> ReportTask {
        ReportTask {
            id: self.task,
            kind: self.kind.clone(),
            conversation_id: self.conversation,
            extension: self.extension.clone(),
        }
    }

    /// Marks the code finished (it never runs, or it ended).
    pub(super) fn finish(&self) {
        self.done.send_replace(true);
    }
}

impl fmt::Debug for Invocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Invocation")
            .field("id", &self.id)
            .field("task", &self.task)
            .field("mode", &self.mode)
            .field("ended", &self.is_ended())
            .finish_non_exhaustive()
    }
}

/// Ends an invocation: its runtime operations reject from now on, its context is cancelled
/// and its task becomes a reservation candidate again. Idempotent.
pub(super) fn end(inner: &Inner, invocation: &Arc<Invocation>) {
    let ended = {
        let mut state = inner.shared.lock();
        state.stopping.remove(&invocation.task);
        invocation.mark_ended() && {
            state.release_invocation(invocation);
            true
        }
    };
    if ended {
        inner.shared.wake_driver();
        invocation.signal("invocation has ended");
    }
}

impl Invocation {
    /// Marks the invocation ended; `false` when it already was.
    pub(super) fn mark_ended(&self) -> bool {
        !self.ended.swap(true, Ordering::AcqRel)
    }
}

impl super::state::State {
    /// Unregisters an ended invocation; its task, if live, is a candidate again.
    pub(super) fn release_invocation(&mut self, invocation: &Arc<Invocation>) {
        let task = invocation.task;
        if self
            .invocations
            .get(&task)
            .is_some_and(|active| Arc::ptr_eq(active, invocation))
        {
            self.invocations.remove(&task);
            if self.tree.live.contains_key(&task) {
                self.runnable.insert(task);
            }
        }
        self.dirty = true;
    }
}

/// A reserved invocation, ready to start.
pub(super) struct Reservation {
    pub(super) invocation: Arc<Invocation>,
    pub(super) definition: Arc<dyn AnyTask>,
    pub(super) snapshot: Arc<dyn CatalogSnapshot>,
}

/// Spawns the invocation's code.
pub(super) fn start(inner: &Arc<Inner>, reservation: Reservation) {
    let invocation = reservation.invocation.clone();
    let task = tokio::spawn(run(inner.clone(), reservation));
    *invocation.join.lock().unwrap_or_else(PoisonError::into_inner) = Some(task.abort_handle());
}

/// Ends the invocation and marks its code finished when the code returns or is dropped.
struct Finish {
    inner: Arc<Inner>,
    invocation: Arc<Invocation>,
}

impl Drop for Finish {
    fn drop(&mut self) {
        end(&self.inner, &self.invocation);
        self.inner.shared.lock().spawned.remove(&self.invocation.id);
        self.invocation.finish();
    }
}

async fn run(inner: Arc<Inner>, reservation: Reservation) {
    let finish = Finish {
        inner: inner.clone(),
        invocation: reservation.invocation.clone(),
    };
    match reservation.invocation.mode {
        Mode::Run => run_phases(&inner, reservation).await,
        Mode::Abort => run_abort(&inner, reservation).await,
    }
    drop(finish);
}

/// Phases in sequence, each preceded by a step that decides on the line whether the
/// invocation continues.
async fn run_phases(inner: &Arc<Inner>, reservation: Reservation) {
    let Reservation {
        invocation,
        definition,
        snapshot,
    } = reservation;
    let mut phase = Phase::new(definition, snapshot);
    let mut current = step::before(inner, &invocation).await;
    while let Some(record) = current.take() {
        if inner.shared.lock().closing {
            break;
        }
        let start = record.state.checkpoint().cloned().unwrap_or(Json::Null);
        let core = Arc::new(RuntimeCore::new(
            inner.clone(),
            invocation.clone(),
            phase.snapshot.clone(),
        ));
        let call = PhaseCall { core, record };
        let result = guarded(phase.definition.run_phase(call)).await;
        if let Some((record, next)) = step::after_phase(inner, &invocation, phase, start, result).await {
            current = Some(record);
            phase = next;
        } else {
            break;
        }
    }
}

/// The abort handler once; then a step under rules 1, 2 and 4.
async fn run_abort(inner: &Arc<Inner>, reservation: Reservation) {
    let Reservation {
        invocation,
        definition,
        snapshot,
    } = reservation;
    let current = {
        let state = inner.shared.lock();
        if state.closing {
            return;
        }
        state.tree.live.get(&invocation.task).cloned()
    };
    let Some(record) = current else {
        return;
    };
    let core = Arc::new(RuntimeCore::new(inner.clone(), invocation.clone(), snapshot));
    let result = guarded(definition.run_abort(PhaseCall { core, record })).await;
    step::after_abort(inner, &invocation, result).await;
}

/// A handler's future with its panic as an error message.
async fn guarded(handler: impl Future<Output = Result<ErasedNext, String>>) -> Result<ErasedNext, String> {
    match AssertUnwindSafe(handler).catch_unwind().await {
        Ok(result) => result,
        Err(panic) => Err(panic
            .downcast_ref::<&str>()
            .map(|message| (*message).to_owned())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .map_or_else(|| "panicked".to_owned(), |message| format!("panicked: {message}"))),
    }
}
