//! The driver and wakeups (spec §5.6): one tokio task per scheduler runs the reconcile and
//! reservation passes when asked, arms one timer for the earliest sleep while scheduling is
//! enabled (none with `on_wake`), follows catalog publications, and computes [`Wake`].

use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Weak};
use std::time::Duration;

use tokio::sync::watch;
use tokio::time::Instant;

use super::Inner;
use super::abort::reconcile;
use super::catalog::CatalogSnapshot;
use super::error::Failure;
use super::invocation::start;
use super::reserve::{Fit, not_due, parked, reserve};
use super::state::{Shared, State};
use crate::records::TaskState;
use crate::session::{CloseSubscription, HostCallback, ReportInfo};

/// When the Session next needs a running process, from memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Wake {
    /// Some task can run now or is running: a pending task that is not blocked, a running
    /// invocation, a waiting task whose `on` settled, a due sleep, an abort-marked task
    /// whose work drained, or a reconcile pass that is scheduled or underway.
    pub active: bool,
    /// The earliest sleep not yet due (epoch ms on the Session clock).
    pub at: Option<i64>,
}

impl State {
    /// The wake (blocked, `completing` and waiting work counts for neither; the work it
    /// waits for does).
    pub(super) fn wake(&self, now: i64, snapshot: &dyn CatalogSnapshot, limit: Option<u32>) -> Wake {
        let mut active = !self.invocations.is_empty() || self.reconciling > 0 || self.reconcile_work();
        let mut at: Option<i64> = None;
        for record in self.tree.live.values() {
            match &record.state {
                TaskState::Completing { .. } => continue,
                TaskState::Waiting { on, .. }
                    if !record.abort_requested && on.iter().any(|id| self.tree.live.contains_key(id)) =>
                {
                    continue;
                }
                TaskState::Sleeping { until, .. } if not_due(record, now) => {
                    at = Some(at.map_or(*until, |at| at.min(*until)));
                    continue;
                }
                _ => {}
            }
            if active || parked(record, limit) {
                continue;
            }
            // An abort invocation waits for the live work below it, which counts itself.
            if record.abort_requested && !self.waiting_on(record, true).is_empty() {
                continue;
            }
            // A task no definition can take waits for a publication, except an abort
            // request, which orphans it.
            let request = record.abort_requested && record.abort_reason.is_none();
            if !request && matches!(self.fit(record, snapshot.task(&record.kind)), Fit::Blocked(_)) {
                continue;
            }
            active = true;
        }
        Wake { active, at }
    }
}

/// The driver: waits for a reason to run a pass, runs it, and follows the wake.
pub(super) async fn drive(
    inner: Weak<Inner>,
    shared: Arc<Shared>,
    mut catalog: watch::Receiver<u64>,
    mut close: CloseSubscription,
) {
    let mut catalog_open = true;
    let mut timer: Option<Instant> = None;
    let mut last_wake: Option<Wake> = None;
    loop {
        let sleep = async {
            match timer {
                Some(deadline) => tokio::time::sleep_until(deadline).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            () = shared.wakeup.notified() => {}
            () = sleep => {
                timer = None;
                shared.lock().dirty = true;
            }
            changed = catalog.changed(), if catalog_open => {
                if changed.is_err() {
                    catalog_open = false;
                } else {
                    shared.bump_blocked();
                    shared.lock().dirty = true;
                }
            }
            _ = close.wait() => {
                if let Some(inner) = inner.upgrade() {
                    inner.seal();
                }
                return;
            }
        }
        let Some(inner) = inner.upgrade() else {
            return;
        };
        if shared.lock().closing {
            return;
        }
        pass(&inner).await;
        let wake = inner.wake();
        match &inner.on_wake {
            Some(on_wake) => {
                if last_wake != Some(wake) {
                    match std::panic::catch_unwind(AssertUnwindSafe(|| on_wake(wake))) {
                        // Only a delivered wake counts; after a panic the next pass offers it again.
                        Ok(()) => last_wake = Some(wake),
                        Err(_) => {
                            let info = ReportInfo::HostError {
                                callback: HostCallback::OnWake,
                                task: None,
                            };
                            inner.session.report(&Failure("on_wake panicked".to_owned()), &info);
                        }
                    }
                }
            }
            None => {
                let enabled = shared.lock().enabled;
                timer = wake.at.filter(|_| enabled).map(|at| {
                    let delay = u64::try_from(at.saturating_sub(inner.session.now())).unwrap_or(0);
                    Instant::now() + Duration::from_millis(delay)
                });
            }
        }
    }
}

/// Reconcile and reservation passes until neither is asked for.
async fn pass(inner: &Arc<Inner>) {
    loop {
        let (reconciling, reserving) = {
            let mut state = inner.shared.lock();
            if state.closing {
                return;
            }
            let reserving = state.enabled && std::mem::take(&mut state.dirty);
            (state.reconcile_scheduled, reserving)
        };
        if !reconciling && !reserving {
            return;
        }
        if reconciling {
            reconcile(inner).await;
        }
        if reserving {
            match reserve(inner).await {
                Ok(reservations) => {
                    for reservation in reservations {
                        start(inner, reservation);
                    }
                }
                Err(error) => inner.fail(error),
            }
        }
    }
}
