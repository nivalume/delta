//! [`PendingApprovals`]: the in-memory fast path from a deciding commit to the phase that
//! waits for it (Codex's `pending_approvals` map), and the [`RunAbort`] seam an `Abort`
//! decision goes through.

use std::collections::HashMap;
use std::error::Error as StdError;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use delta_chord::context::Context;
use futures::future::BoxFuture;
use tokio::sync::oneshot;

use super::document::{ApprovalId, ReviewDecision};
use crate::records::TaskId;

/// Aborts a conversation's run task: what an `Abort` decision asks. The surface implements
/// it over the scheduler, which the agent layer cannot name.
pub trait RunAbort: Send + Sync {
    fn abort_run<'a>(
        &'a self,
        task: TaskId,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<(), Box<dyn StdError + Send + Sync>>>;
}

type Waiters = HashMap<ApprovalId, (u64, oneshot::Sender<ReviewDecision>)>;

/// The waiters of the session's approval phases, keyed by approval id. A session service,
/// never persisted: losing it loses nothing, because a waiting phase reads the committed
/// decision first.
#[derive(Default)]
pub struct PendingApprovals {
    waiters: Mutex<Waiters>,
    next: AtomicU64,
    run_abort: OnceLock<Arc<dyn RunAbort>>,
}

impl PendingApprovals {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, Waiters> {
        self.waiters.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A waiter for `id`; replaces an older one (a rerun of the phase).
    pub fn register(self: &Arc<Self>, id: ApprovalId) -> ApprovalWaiter {
        let (sender, receiver) = oneshot::channel();
        let token = self.next.fetch_add(1, Ordering::Relaxed);
        self.lock().insert(id, (token, sender));
        ApprovalWaiter {
            pending: self.clone(),
            id,
            token,
            receiver,
        }
    }

    /// Hands `decision` to the waiter of `id`, after the deciding commit settled; `false`
    /// when nobody waits.
    pub fn notify(&self, id: ApprovalId, decision: ReviewDecision) -> bool {
        let waiter = self.lock().remove(&id);
        waiter.is_some_and(|(_, sender)| sender.send(decision).is_ok())
    }

    /// Installs the run abort; the first installation wins.
    pub fn set_run_abort(&self, run_abort: Arc<dyn RunAbort>) {
        let _ = self.run_abort.set(run_abort);
    }

    pub fn run_abort(&self) -> Option<&Arc<dyn RunAbort>> {
        self.run_abort.get()
    }
}

impl fmt::Debug for PendingApprovals {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let waiting: Vec<ApprovalId> = self.lock().keys().copied().collect();
        f.debug_struct("PendingApprovals")
            .field("waiting", &waiting)
            .finish_non_exhaustive()
    }
}

/// A registered waiter; removes its entry when dropped, unless a newer one replaced it.
pub struct ApprovalWaiter {
    pending: Arc<PendingApprovals>,
    id: ApprovalId,
    token: u64,
    receiver: oneshot::Receiver<ReviewDecision>,
}

impl ApprovalWaiter {
    /// The notified decision; `None` when the waiter was replaced.
    pub async fn decision(&mut self) -> Option<ReviewDecision> {
        (&mut self.receiver).await.ok()
    }
}

impl Drop for ApprovalWaiter {
    fn drop(&mut self) {
        let mut waiters = self.pending.lock();
        if waiters.get(&self.id).is_some_and(|(token, _)| *token == self.token) {
            waiters.remove(&self.id);
        }
    }
}

impl fmt::Debug for ApprovalWaiter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApprovalWaiter")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn notify_reaches_the_latest_waiter_only() {
        let pending = Arc::new(PendingApprovals::new());
        let id = ApprovalId(TaskId(3));
        assert!(!pending.notify(id, ReviewDecision::Approved));
        let mut first = pending.register(id);
        let mut second = pending.register(id);
        // The replaced waiter's sender is gone.
        assert_eq!(first.decision().await, None);
        drop(first);
        assert!(pending.notify(id, ReviewDecision::Abort));
        assert_eq!(second.decision().await, Some(ReviewDecision::Abort));
        let third = pending.register(id);
        drop(third);
        assert!(!pending.notify(id, ReviewDecision::Approved));
    }
}
