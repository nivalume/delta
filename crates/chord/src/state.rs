//! Observable revision sequences: [`State`] adopts changes and delivers each one, value and
//! ops, to every [`Subscription`] in commit order. See `docs/design/state.md`.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use futures::Stream;
use tokio::sync::Notify;

use crate::context::Context;
use crate::delta::{AdoptError, Change, Draft, MAX_OPS, Op, Prepared, Tracker};
use crate::json::Json;

/// Deliveries a subscription keeps before merging them, by default.
pub const DEFAULT_MAX_PENDING: usize = 100;

/// An observable JSON value. Clones are handles on the same state.
#[derive(Clone)]
pub struct State {
    shared: Arc<Shared>,
}

struct Shared {
    inner: Mutex<Inner>,
}

struct Inner {
    tracker: Tracker,
    sequence: u64,
    subscribers: Vec<Weak<Queue>>,
}

thread_local! {
    /// States whose `change` closure runs on this thread, to turn reentrancy into a panic
    /// instead of a deadlock.
    static CHANGING: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

struct ChangingGuard(usize);

impl ChangingGuard {
    fn enter(id: usize) -> Self {
        CHANGING.with_borrow_mut(|changing| {
            assert!(
                !changing.contains(&id),
                "State::change called from inside a change of the same state"
            );
            changing.push(id);
        });
        Self(id)
    }
}

impl Drop for ChangingGuard {
    fn drop(&mut self) {
        CHANGING.with_borrow_mut(|changing| changing.retain(|&id| id != self.0));
    }
}

impl State {
    pub fn new(initial: Json) -> Self {
        let inner = Inner {
            tracker: Tracker::new(initial),
            sequence: 0,
            subscribers: Vec::new(),
        };
        Self {
            shared: Arc::new(Shared {
                inner: Mutex::new(inner),
            }),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        // A panic inside `change` leaves the tracker as it was (nothing was adopted).
        self.shared.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The current value. O(1).
    pub fn value(&self) -> Json {
        self.lock().tracker.value().clone()
    }

    /// The number of published changes.
    pub fn sequence(&self) -> u64 {
        self.lock().sequence
    }

    /// Edit the value through a draft, then adopt and publish the change.
    ///
    /// `edit` runs while the state is locked: changes from other threads wait. If it panics,
    /// the state keeps its value.
    ///
    /// # Panics
    ///
    /// When called from inside `edit` on the same state.
    pub fn change<R>(&self, context: &Context, edit: impl FnOnce(Draft<'_>) -> R) -> R {
        let _guard = ChangingGuard::enter(Arc::as_ptr(&self.shared) as usize);
        let mut inner = self.lock();
        let mut change = inner.tracker.begin();
        let result = edit(change.draft());
        let ops = inner
            .tracker
            .adopt(change.prepare())
            .expect("no competing change while locked");
        inner.publish(context, ops);
        result
    }

    /// Change to `value`, publishing the [`diff`](crate::delta::diff) from the current value.
    pub fn replace(&self, context: &Context, value: Json) {
        let mut inner = self.lock();
        let prepared = inner.tracker.prepare_diff(value);
        let ops = inner.tracker.adopt(prepared).expect("no competing change while locked");
        inner.publish(context, ops);
    }

    /// Start a change for a writer that persists its batch before committing it.
    pub fn begin(&self) -> Change {
        self.lock().tracker.begin()
    }

    /// Adopt and publish a prepared change. Fails when another change was committed since it
    /// began.
    pub fn commit(&self, context: &Context, prepared: Prepared) -> Result<(), AdoptError> {
        let mut inner = self.lock();
        let ops = inner.tracker.adopt(prepared)?;
        inner.publish(context, ops);
        Ok(())
    }

    /// Subscribe with the default backlog.
    pub fn subscribe(&self) -> Subscription {
        self.subscribe_with(SubscribeOptions::default())
    }

    pub fn subscribe_with(&self, options: SubscribeOptions) -> Subscription {
        let mut inner = self.lock();
        let hydrate = Delivery {
            kind: DeliveryKind::Hydrate,
            value: inner.tracker.value().clone(),
            ops: Arc::from([]),
            sequence: inner.sequence,
            skipped: 0,
            context: Context::background(),
        };
        let queue = Arc::new(Queue {
            pending: Mutex::new(Pending {
                deliveries: VecDeque::from([hydrate]),
                max: options.max_pending.max(1),
                closed: false,
            }),
            notify: Notify::new(),
        });
        inner.subscribers.push(Arc::downgrade(&queue));
        Subscription { queue }
    }
}

impl Inner {
    fn publish(&mut self, context: &Context, ops: Vec<Op>) {
        if ops.is_empty() {
            return;
        }
        self.sequence += 1;
        let ops: Arc<[Op]> = ops.into();
        let value = self.tracker.value();
        let sequence = self.sequence;
        self.subscribers.retain(|subscriber| {
            let Some(queue) = subscriber.upgrade() else {
                return false;
            };
            queue.push(Delivery {
                kind: DeliveryKind::Update,
                value: value.clone(),
                ops: ops.clone(),
                sequence,
                skipped: 0,
                context: context.clone(),
            });
            true
        });
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        let inner = self.inner.get_mut().unwrap_or_else(PoisonError::into_inner);
        for queue in inner
            .subscribers
            .drain(..)
            .filter_map(|subscriber| subscriber.upgrade())
        {
            queue.lock().closed = true;
            queue.notify.notify_one();
        }
    }
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.lock();
        f.debug_struct("State")
            .field("sequence", &inner.sequence)
            .field("subscribers", &inner.subscribers.len())
            .field("value", inner.tracker.value())
            .finish()
    }
}

// ─── Subscriptions ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubscribeOptions {
    /// Deliveries kept before they merge into one. At least 1.
    pub max_pending: usize,
}

impl Default for SubscribeOptions {
    fn default() -> Self {
        Self {
            max_pending: DEFAULT_MAX_PENDING,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryKind {
    /// The first delivery: the value when the subscription started (or later, when it merged
    /// with updates), and no ops.
    Hydrate,
    /// A change: `ops` turn the previous delivery's value into this one.
    Update,
}

/// One step of a subscription.
#[derive(Debug, Clone)]
pub struct Delivery {
    pub kind: DeliveryKind,
    pub value: Json,
    /// From the previous delivery's value to `value`; empty for [`DeliveryKind::Hydrate`].
    pub ops: Arc<[Op]>,
    /// The state's [`sequence`](State::sequence) after this change.
    pub sequence: u64,
    /// Changes merged into this delivery because the subscriber fell behind.
    pub skipped: u64,
    /// The context of the (last) change; the background context for a hydration.
    pub context: Context,
}

/// Receives every change of a [`State`] in commit order. Dropping it unsubscribes.
#[derive(Debug)]
pub struct Subscription {
    queue: Arc<Queue>,
}

struct Queue {
    pending: Mutex<Pending>,
    notify: Notify,
}

struct Pending {
    deliveries: VecDeque<Delivery>,
    max: usize,
    closed: bool,
}

impl fmt::Debug for Queue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pending = self.lock();
        f.debug_struct("Queue")
            .field("pending", &pending.deliveries.len())
            .field("closed", &pending.closed)
            .finish()
    }
}

impl Queue {
    fn lock(&self) -> MutexGuard<'_, Pending> {
        self.pending.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn push(&self, delivery: Delivery) {
        let mut pending = self.lock();
        if pending.deliveries.len() >= pending.max {
            let backlog = std::mem::take(&mut pending.deliveries);
            pending.deliveries.push_back(merge(&backlog, delivery));
        } else {
            pending.deliveries.push_back(delivery);
        }
        drop(pending);
        self.notify.notify_one();
    }
}

/// One delivery for a backlog and the newest change: the newest value with the batches
/// concatenated (a replacement when that is too long).
fn merge(backlog: &VecDeque<Delivery>, newest: Delivery) -> Delivery {
    let hydrate = backlog.front().is_some_and(|first| first.kind == DeliveryKind::Hydrate);
    // Updates absorbed into the result: every queued update, and the newest one too when the
    // result is a hydration (its value already includes it).
    let skipped = backlog
        .iter()
        .map(|delivery| delivery.skipped + u64::from(delivery.kind == DeliveryKind::Update))
        .sum::<u64>()
        + newest.skipped
        + u64::from(hydrate);
    let ops: Arc<[Op]> = if hydrate {
        Arc::from([])
    } else {
        let total = backlog
            .iter()
            .chain([&newest])
            .map(|delivery| delivery.ops.len())
            .sum::<usize>();
        if total > MAX_OPS {
            Arc::from([Op::Replace(newest.value.clone())])
        } else {
            backlog
                .iter()
                .chain([&newest])
                .flat_map(|delivery| delivery.ops.iter().cloned())
                .collect()
        }
    };
    Delivery {
        kind: if hydrate {
            DeliveryKind::Hydrate
        } else {
            DeliveryKind::Update
        },
        ops,
        skipped,
        ..newest
    }
}

impl Subscription {
    /// The next delivery; `None` once every handle of the state is dropped and the queue is
    /// drained.
    pub async fn next(&mut self) -> Option<Delivery> {
        loop {
            {
                let mut pending = self.queue.lock();
                if let Some(delivery) = pending.deliveries.pop_front() {
                    return Some(delivery);
                }
                if pending.closed {
                    return None;
                }
            }
            self.queue.notify.notified().await;
        }
    }

    /// The next delivery if one is queued.
    pub fn try_next(&mut self) -> Option<Delivery> {
        self.queue.lock().deliveries.pop_front()
    }

    /// The deliveries as a stream.
    pub fn into_stream(self) -> impl Stream<Item = Delivery> + Send + 'static {
        futures::stream::unfold(self, |mut subscription| async move {
            let delivery = subscription.next().await?;
            Some((delivery, subscription))
        })
    }
}
