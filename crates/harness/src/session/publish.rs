//! Commit publications and who receives them: on-line observers, off-line subscribers and
//! close subscribers.

use std::error::Error as StdError;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError, Weak};

use delta_chord::context::Context;
use delta_chord::delta::Op;
use delta_chord::json::Json;
use tokio::sync::{broadcast, watch};

use super::lifecycle::{Phase, SessionEnd};
use crate::records::{
    ConversationId, ConversationRecord, DocumentCopySource, DocumentRecord, EntryRecord, Seq, SubmissionRecord,
    TaskRecord,
};

/// Publications a [`CommitSubscription`] keeps before it reports a [`CommitGap`].
pub const COMMIT_BACKLOG: usize = 1024;

/// What one commit changed, published after storage accepted it and memory adopted it.
#[derive(Clone, Debug)]
pub struct CommitPublication {
    pub seq: Seq,
    /// Table records in write order, then document changes in staging order.
    pub changes: Vec<CommitChange>,
    /// The commit's context, without its cancellation.
    pub context: Context,
}

/// One change of a [`CommitPublication`].
#[derive(Clone, Debug)]
pub enum CommitChange {
    Conversation(ConversationRecord),
    Entry(EntryRecord),
    Task(TaskRecord),
    Submission(SubmissionRecord),
    /// A created, changed or retired incarnation. Retired: `value` and `version` are `None`.
    /// `ops` is empty for a creation and for a base that changed nothing (a migration).
    Document {
        record: DocumentRecord,
        /// The owning conversation: the scope's, or the owning task's for a task document.
        conversation_id: Option<ConversationId>,
        version: Option<u32>,
        value: Option<Json>,
        ops: Arc<[Op]>,
    },
    /// A fork copy that no definition read in the fork commit; its value is the source's.
    DocumentCopy {
        record: DocumentRecord,
        conversation_id: ConversationId,
        source: DocumentCopySource,
    },
}

/// A Session component that keeps memory in step with storage: the scheduler and the
/// indexes of the layers above.
///
/// It runs on the commit line, synchronously after adoption and before subscribers are fed,
/// so it sees every commit in order and nothing else runs meanwhile. It must not block or
/// call Session operations. An error fails the Session: the component's memory would be
/// behind storage.
pub trait CommitObserver: Send {
    fn committed(&mut self, publication: &CommitPublication) -> Result<(), Box<dyn StdError + Send + Sync>>;
}

impl<F> CommitObserver for F
where
    F: FnMut(&CommitPublication) -> Result<(), Box<dyn StdError + Send + Sync>> + Send,
{
    fn committed(&mut self, publication: &CommitPublication) -> Result<(), Box<dyn StdError + Send + Sync>> {
        self(publication)
    }
}

/// The registered on-line observers.
#[derive(Default)]
pub(super) struct Observers {
    next_id: u64,
    list: Vec<(u64, Box<dyn CommitObserver>)>,
    /// Set while the list is taken out to run; removals then wait in `removed`.
    running: bool,
    removed: Vec<u64>,
}

pub(super) type SharedObservers = Arc<Mutex<Observers>>;

fn lock(observers: &Mutex<Observers>) -> std::sync::MutexGuard<'_, Observers> {
    observers.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Observers {
    pub(super) fn register(observers: &SharedObservers, observer: Box<dyn CommitObserver>) -> ObserverGuard {
        let mut guard = lock(observers);
        guard.next_id += 1;
        let id = guard.next_id;
        guard.list.push((id, observer));
        ObserverGuard {
            observers: Arc::downgrade(observers),
            id,
        }
    }

    /// Runs every observer once, in registration order, outside the lock (an observer may
    /// register or drop observers). Every observer runs; the first error is returned.
    pub(super) fn run(
        observers: &Mutex<Observers>,
        publication: &CommitPublication,
    ) -> Result<(), Box<dyn StdError + Send + Sync>> {
        let mut taken = {
            let mut guard = lock(observers);
            if guard.list.is_empty() {
                return Ok(());
            }
            guard.running = true;
            std::mem::take(&mut guard.list)
        };
        let mut result = Ok(());
        for (_, observer) in &mut taken {
            if let Err(error) = observer.committed(publication) {
                result = result.and(Err(error));
            }
        }
        let mut guard = lock(observers);
        guard.running = false;
        let removed = std::mem::take(&mut guard.removed);
        taken.retain(|(id, _)| !removed.contains(id));
        taken.append(&mut guard.list);
        guard.list = taken;
        result
    }

    pub(super) fn is_empty(observers: &Mutex<Observers>) -> bool {
        lock(observers).list.is_empty()
    }

    /// Drops every observer (at close).
    pub(super) fn clear(observers: &Mutex<Observers>) {
        let list = std::mem::take(&mut lock(observers).list);
        drop(list);
    }
}

impl fmt::Debug for Observers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Observers")
            .field("count", &self.list.len())
            .finish_non_exhaustive()
    }
}

/// Keeps an on-line observer registered; dropping it removes the observer.
#[derive(Debug)]
#[must_use = "dropping the guard removes the observer"]
pub struct ObserverGuard {
    observers: Weak<Mutex<Observers>>,
    id: u64,
}

impl Drop for ObserverGuard {
    fn drop(&mut self) {
        let Some(observers) = self.observers.upgrade() else {
            return;
        };
        let removed = {
            let mut guard = lock(&observers);
            let index = guard.list.iter().position(|(id, _)| *id == self.id);
            if guard.running && index.is_none() {
                guard.removed.push(self.id);
            }
            index.map(|index| guard.list.remove(index))
        };
        drop(removed);
    }
}

/// Publications a subscriber missed because it fell behind by more than [`COMMIT_BACKLOG`];
/// it resynchronizes from storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("missed {missed} commit publications")]
pub struct CommitGap {
    pub missed: u64,
}

/// Every commit publication from the moment of subscription, off the line. A slow
/// subscriber delays only itself.
#[derive(Debug)]
pub struct CommitSubscription {
    receiver: Option<broadcast::Receiver<Arc<CommitPublication>>>,
}

impl CommitSubscription {
    pub(super) fn new(receiver: Option<broadcast::Receiver<Arc<CommitPublication>>>) -> Self {
        Self { receiver }
    }

    /// The next publication, a [`CommitGap`] after an overflow, or `None` once the Session
    /// closed and every publication was received.
    pub async fn next(&mut self) -> Option<Result<Arc<CommitPublication>, CommitGap>> {
        let receiver = self.receiver.as_mut()?;
        match receiver.recv().await {
            Ok(publication) => Some(Ok(publication)),
            Err(broadcast::error::RecvError::Lagged(missed)) => Some(Err(CommitGap { missed })),
            Err(broadcast::error::RecvError::Closed) => {
                self.receiver = None;
                None
            }
        }
    }
}

/// Resolves when the Session begins to close, a failure included, after admission is
/// sealed.
#[derive(Debug)]
pub struct CloseSubscription {
    phase: watch::Receiver<Phase>,
}

impl CloseSubscription {
    pub(super) fn new(phase: watch::Receiver<Phase>) -> Self {
        Self { phase }
    }

    /// Whether close has begun.
    pub fn is_closing(&self) -> bool {
        !matches!(*self.phase.borrow(), Phase::Open)
    }

    /// Waits until close begins and returns why: [`SessionEnd::Closed`] for a requested
    /// close, [`SessionEnd::Failed`] for a failure.
    pub async fn wait(&mut self) -> SessionEnd {
        let phase = self.phase.wait_for(|phase| !matches!(phase, Phase::Open)).await;
        match phase.as_deref() {
            Ok(Phase::Closing(end) | Phase::Ended(end)) => end.clone(),
            // The Session is gone without a recorded close: it was dropped.
            Ok(Phase::Open) | Err(_) => SessionEnd::Closed,
        }
    }
}
