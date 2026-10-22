//! [`AgentEventStream`]: a conversation's agent events, one batch per commit, relative to the
//! snapshot taken at attachment.

use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use tokio::sync::Notify;

use super::events::{AgentEvent, Snapshot};
use super::translate::{Translator, snapshot};
use super::view::{Mount, MountListener, Revision};
use crate::generation::GENERATION_TASK;
use crate::records::{ConversationId, TaskId, TaskQuery, TaskStatus};
use crate::session::{CloseSubscription, Tx, TxError};

/// Undelivered batches a stream keeps; one more replaces them with a snapshot.
pub const EVENT_BACKLOG: usize = 100;

struct Queued {
    batches: VecDeque<Vec<AgentEvent>>,
    translator: Translator,
}

/// The listening side: translates each revision on the line and queues its batch.
struct EventQueue {
    queued: Mutex<Queued>,
    notify: Notify,
}

impl EventQueue {
    fn lock(&self) -> MutexGuard<'_, Queued> {
        self.queued.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl MountListener for EventQueue {
    fn advanced(&self, revision: &Revision<'_>) {
        let mut queued = self.lock();
        let events = queued.translator.translate(revision);
        if events.is_empty() {
            return;
        }
        if queued.batches.len() >= EVENT_BACKLOG {
            queued.batches.clear();
            queued
                .batches
                .push_back(vec![AgentEvent::Snapshot(Box::new(snapshot(revision.after)))]);
        } else {
            queued.batches.push_back(events);
        }
        drop(queued);
        self.notify.notify_one();
    }
}

/// A conversation's agent events from attachment on. Nothing is replayed: a consumer that
/// reconnects attaches again. Dropping the stream detaches it.
pub struct AgentEventStream {
    snapshot: Snapshot,
    queue: Arc<EventQueue>,
    close: CloseSubscription,
    _mount: Arc<Mount>,
}

impl AgentEventStream {
    /// The state at attachment; every event is relative to it and the events before.
    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    /// The next commit's events; after an overflow, one `Snapshot` of the newest state.
    /// `None` once the Session closes and every queued batch was taken.
    pub async fn next(&mut self) -> Option<Vec<AgentEvent>> {
        loop {
            let notified = self.queue.notify.notified();
            if let Some(batch) = self.queue.lock().batches.pop_front() {
                return Some(batch);
            }
            if self.close.is_closing() {
                return None;
            }
            tokio::select! {
                () = notified => {}
                _ = self.close.wait() => {}
            }
        }
    }

    /// The next batch if one is queued.
    pub fn try_next(&mut self) -> Option<Vec<AgentEvent>> {
        self.queue.lock().batches.pop_front()
    }
}

impl fmt::Debug for AgentEventStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let queued = self.queue.lock().batches.len();
        f.debug_struct("AgentEventStream")
            .field("queued", &queued)
            .finish_non_exhaustive()
    }
}

/// A stream attached on the line, before its handle exists.
pub(crate) struct Attached {
    snapshot: Snapshot,
    queue: Arc<EventQueue>,
}

impl Attached {
    pub(crate) fn into_stream(self, close: CloseSubscription, mount: Arc<Mount>) -> AgentEventStream {
        AgentEventStream {
            snapshot: self.snapshot,
            queue: self.queue,
            close,
            _mount: mount,
        }
    }
}

/// Attaches a stream to `mount` on the line: the snapshot of the mount's current value and
/// the generations of `conversation` that already hold their outcome, registered before
/// any later commit runs.
pub(crate) async fn attach(
    tx: &mut Tx<'_>,
    mount: &Arc<Mount>,
    conversation: ConversationId,
) -> Result<Attached, TxError> {
    let query = TaskQuery {
        conversation_id: Some(conversation),
        kind: Some(GENERATION_TASK.to_owned()),
        status: Some(TaskStatus::Completing),
        ..TaskQuery::default()
    };
    let mut held: HashSet<TaskId> = HashSet::new();
    let mut cursor = None;
    loop {
        let page = tx.scan_tasks(query.clone(), 256, cursor).await?;
        held.extend(page.items.iter().map(|task| task.id));
        match page.next {
            Some(next) => cursor = Some(next),
            None => break,
        }
    }
    let queue = Arc::new(EventQueue {
        queued: Mutex::new(Queued {
            batches: VecDeque::new(),
            translator: Translator::new(conversation, held),
        }),
        notify: Notify::new(),
    });
    let listener: Weak<dyn MountListener> = Arc::downgrade(&queue) as Weak<dyn MountListener>;
    mount.listen(listener);
    Ok(Attached {
        snapshot: snapshot(&mount.state().value()),
        queue,
    })
}
