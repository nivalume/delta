//! State shared between the Session handles, the line and the observers: the phase
//! (open, closing, ended), the failure latch, the attached document states, the observer
//! registry and the publication channel.

use std::error::Error as StdError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use tokio::sync::{Notify, broadcast, watch};

use super::cache::{DocHandle, DocIndex};
use super::error::{SessionError, SessionFailed};
use super::observation::WatchEnd;
use super::publish::{COMMIT_BACKLOG, CommitPublication, Observers, SharedObservers};
use super::report::{ReportInfo, Reporter};
use super::services::Services;

/// How a Session ended.
#[derive(Clone, Debug)]
pub enum SessionEnd {
    /// Closed by [`Session::close`](super::Session::close) or by dropping its last handle.
    Closed,
    /// Closed by a failure; reopen the storage.
    Failed(SessionFailed),
}

/// Where the Session is in its life. `Closing` and `Ended` carry why it closes.
#[derive(Clone, Debug)]
pub(super) enum Phase {
    Open,
    /// Admission is sealed; admitted work settles, then storage closes.
    Closing(SessionEnd),
    Ended(SessionEnd),
}

/// What every handle, the line and every attached state share.
pub(super) struct Shared {
    phase: watch::Sender<Phase>,
    failure: OnceLock<SessionFailed>,
    /// The error of a storage `close` that failed.
    close_error: OnceLock<SessionFailed>,
    reporter: Option<super::report::Reporter>,
    pub(super) services: Services,
    /// Cached current incarnations by address, for snapshots answered without a line job.
    pub(super) documents: DocIndex,
    /// Handles with an attached [`DocumentState`](super::DocumentState), ended at the seal.
    attached: Mutex<Vec<Weak<DocHandle>>>,
    pub(super) observers: SharedObservers,
    commits: Mutex<Option<broadcast::Sender<Arc<CommitPublication>>>>,
    /// Wakes the line when a failure off the line asks it to finish closing.
    pub(super) wake: Notify,
    close_requested: AtomicBool,
}

impl Shared {
    pub(super) fn new(reporter: Option<Reporter>, services: Services) -> Self {
        Self {
            phase: watch::Sender::new(Phase::Open),
            failure: OnceLock::new(),
            close_error: OnceLock::new(),
            reporter,
            services,
            documents: DocIndex::default(),
            attached: Mutex::new(Vec::new()),
            observers: SharedObservers::default(),
            commits: Mutex::new(Some(broadcast::Sender::new(COMMIT_BACKLOG))),
            wake: Notify::new(),
            close_requested: AtomicBool::new(false),
        }
    }

    pub(super) fn failure(&self) -> Option<SessionFailed> {
        self.failure.get().cloned()
    }

    /// `SessionFailed` once the Session failed.
    pub(super) fn check_healthy(&self) -> Result<(), SessionFailed> {
        self.failure().map_or(Ok(()), Err)
    }

    /// Admission: healthy and not closing.
    pub(super) fn check_usable(&self) -> Result<(), SessionError> {
        self.check_healthy()?;
        if self.is_sealed() {
            return Err(SessionError::Closed);
        }
        Ok(())
    }

    pub(super) fn is_sealed(&self) -> bool {
        !matches!(*self.phase.borrow(), Phase::Open)
    }

    pub(super) fn phase(&self) -> watch::Receiver<Phase> {
        self.phase.subscribe()
    }

    /// The error a call gets when the line is gone or refused it.
    pub(super) fn terminal_error(&self) -> SessionError {
        self.failure().map_or(SessionError::Closed, SessionError::Failed)
    }

    /// Seals admission, once: later calls are refused and every attached document state
    /// ends with `end`. Returns whether this call sealed.
    pub(super) fn seal(&self, end: SessionEnd) -> bool {
        let reason = match &end {
            SessionEnd::Closed => WatchEnd::SessionClosed,
            SessionEnd::Failed(failed) => WatchEnd::SessionFailed(failed.clone()),
        };
        let sealed = self.phase.send_if_modified(|phase| {
            if matches!(phase, Phase::Open) {
                *phase = Phase::Closing(end);
                true
            } else {
                false
            }
        });
        if sealed {
            let attached = std::mem::take(&mut *self.attached.lock().unwrap_or_else(PoisonError::into_inner));
            for handle in attached.iter().filter_map(Weak::upgrade) {
                handle.end(reason.clone());
            }
        }
        sealed
    }

    /// Fails the Session with `cause`; the first failure wins and is returned to every later
    /// caller. Seals admission, ends attached states, asks the line to close storage and
    /// reports the failure once.
    pub(super) fn fail(&self, cause: Arc<dyn StdError + Send + Sync>) -> SessionFailed {
        let failed = SessionFailed { cause };
        if self.failure.set(failed.clone()).is_err() {
            return self.failure().unwrap_or(failed);
        }
        // Seal first, so a reporter that calls back finds the Session failed.
        self.seal(SessionEnd::Failed(failed.clone()));
        self.request_close();
        self.report(&failed);
        failed
    }

    /// Records that storage failed to close: the Session fails unless it already had.
    pub(super) fn close_failed(&self, cause: Arc<dyn StdError + Send + Sync>) {
        let failed = SessionFailed { cause };
        let _ = self.close_error.set(failed.clone());
        if self.failure.set(failed.clone()).is_ok() {
            self.report(&failed);
        }
    }

    pub(super) fn close_error(&self) -> Option<SessionFailed> {
        self.close_error.get().cloned()
    }

    fn report(&self, failed: &SessionFailed) {
        if let Some(reporter) = &self.reporter {
            reporter(&*failed.cause, &ReportInfo::SessionFailed);
        }
    }

    /// Asks the line to finish closing after its current job.
    pub(super) fn request_close(&self) {
        self.close_requested.store(true, Ordering::Release);
        self.wake.notify_one();
    }

    pub(super) fn close_requested(&self) -> bool {
        self.close_requested.load(Ordering::Acquire)
    }

    /// Marks the Session ended; `closed()` resolves.
    pub(super) fn finish(&self) {
        let end = self.failure().map_or(SessionEnd::Closed, SessionEnd::Failed);
        self.phase.send_replace(Phase::Ended(end));
    }

    /// Waits until the Session ended.
    pub(super) async fn ended(&self) -> SessionEnd {
        let mut phase = self.phase.subscribe();
        let ended = phase.wait_for(|phase| matches!(phase, Phase::Ended(_))).await;
        match ended.as_deref() {
            Ok(Phase::Ended(end)) => end.clone(),
            // The sender lives in `self`, so the channel cannot close while we wait.
            _ => SessionEnd::Closed,
        }
    }

    /// Registers an attached document state's handle, so the seal ends it.
    pub(super) fn attach(&self, handle: &Arc<DocHandle>) {
        let mut attached = self.attached.lock().unwrap_or_else(PoisonError::into_inner);
        attached.retain(|weak| weak.strong_count() > 0);
        attached.push(Arc::downgrade(handle));
    }

    /// Whether anyone receives publications: an observer or a subscriber.
    pub(super) fn has_listeners(&self) -> bool {
        let subscribers = self
            .commits
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .is_some_and(|sender| sender.receiver_count() > 0);
        subscribers || !Observers::is_empty(&self.observers)
    }

    pub(super) fn subscribe_commits(&self) -> Option<broadcast::Receiver<Arc<CommitPublication>>> {
        let commits = self.commits.lock().unwrap_or_else(PoisonError::into_inner);
        commits.as_ref().map(broadcast::Sender::subscribe)
    }

    /// Feeds the subscribers; a send without receivers is not an error.
    pub(super) fn send_publication(&self, publication: Arc<CommitPublication>) {
        let commits = self.commits.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(sender) = commits.as_ref() {
            let _ = sender.send(publication);
        }
    }

    /// Ends every subscription once the queued publications are received (at close).
    pub(super) fn close_commits(&self) {
        let sender = self.commits.lock().unwrap_or_else(PoisonError::into_inner).take();
        drop(sender);
    }
}
