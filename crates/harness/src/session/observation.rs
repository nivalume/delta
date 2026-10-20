//! Read-only document observation: [`DocumentState`] over one incarnation's chord state,
//! and its subscriptions.

use std::fmt;
use std::sync::{Arc, Weak};

use delta_chord::json::Json;
use delta_chord::state::{Delivery, Subscription};
use futures::Stream;

use super::cache::DocHandle;
use super::error::SessionFailed;
use super::lifecycle::Shared;
use crate::records::{DocumentId, DocumentRecord};

/// Why a [`DocumentState`] stopped following its incarnation.
#[derive(Clone, Debug)]
pub enum WatchEnd {
    /// The incarnation was retired (`["r", null]` was its last delivery).
    Retired,
    SessionClosed,
    SessionFailed(SessionFailed),
}

/// A read-only view of one committed document incarnation (upstream `documentState` and
/// `watchDoc`).
///
/// It is bound to the incarnation it was acquired for: retirement ends it and a later
/// recreation at the address needs a new acquisition. While it or one of its subscriptions
/// lives, the incarnation stays cached. Session close ends it at the seal, a failure with
/// [`WatchEnd::SessionFailed`]; a commit settling during close delivers nothing.
pub struct DocumentState {
    pin: Pin,
}

impl DocumentState {
    /// Pins `handle` (on the line) and registers it so the seal ends it.
    pub(super) fn attach(handle: Arc<DocHandle>, shared: &Arc<Shared>) -> Self {
        shared.attach(&handle);
        Self {
            pin: Pin::new(handle, shared),
        }
    }

    /// The incarnation's record as acquired.
    pub fn record(&self) -> &DocumentRecord {
        self.pin.handle.record()
    }

    /// The incarnation's id.
    pub fn id(&self) -> DocumentId {
        self.pin.handle.record().id
    }

    /// The last published revision; `Json::Null` after retirement, the last value after the
    /// Session closed.
    pub fn value(&self) -> Json {
        self.pin.handle.value()
    }

    /// Every committed revision from now on: a `Hydrate` with the current value, then an
    /// `Update` per commit with the exact adopted ops, in commit order, off the line. A
    /// subscriber that falls behind gets merged batches (chord's rule). Retirement delivers
    /// `["r", null]` with value `Null`, then the subscription ends; so does close.
    pub fn subscribe(&self) -> DocumentSubscription {
        DocumentSubscription {
            inner: self.pin.handle.subscribe(),
            _pin: self.pin.clone(),
        }
    }

    /// Why the state stopped following its incarnation; `None` while it follows it.
    pub fn end(&self) -> Option<WatchEnd> {
        self.pin.handle.end_reason()
    }

    /// Waits until the state stops following its incarnation.
    pub async fn closed(&self) -> WatchEnd {
        let mut end = self.pin.handle.watch_end();
        let ended = end.wait_for(Option::is_some).await;
        match ended.as_deref() {
            Ok(Some(end)) => end.clone(),
            // The sender lives in the handle this state holds.
            Ok(None) | Err(_) => WatchEnd::SessionClosed,
        }
    }
}

impl Clone for DocumentState {
    fn clone(&self) -> Self {
        Self { pin: self.pin.clone() }
    }
}

impl fmt::Debug for DocumentState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DocumentState")
            .field("id", &self.id())
            .field("kind", &self.record().kind)
            .field("end", &self.end())
            .finish_non_exhaustive()
    }
}

/// The deliveries of a [`DocumentState`]: chord [`Delivery`]s. It keeps the incarnation
/// cached like the state; dropping it unsubscribes.
pub struct DocumentSubscription {
    inner: Subscription,
    _pin: Pin,
}

impl DocumentSubscription {
    /// The next delivery; `None` once the state ended and every queued delivery was taken.
    pub async fn next(&mut self) -> Option<Delivery> {
        self.inner.next().await
    }

    /// The next delivery if one is queued.
    pub fn try_next(&mut self) -> Option<Delivery> {
        self.inner.try_next()
    }

    /// The deliveries as a stream.
    pub fn into_stream(self) -> impl Stream<Item = Delivery> + Send + 'static {
        futures::stream::unfold(self, |mut subscription| async move {
            let delivery = subscription.next().await?;
            Some((delivery, subscription))
        })
    }
}

impl fmt::Debug for DocumentSubscription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DocumentSubscription")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

/// One pin of a cached handle: a pinned incarnation is never evicted. Unpinning stamps a
/// use and wakes the line, which sweeps.
struct Pin {
    handle: Arc<DocHandle>,
    shared: Weak<Shared>,
}

impl Pin {
    fn new(handle: Arc<DocHandle>, shared: &Arc<Shared>) -> Self {
        handle.pin();
        Self {
            handle,
            shared: Arc::downgrade(shared),
        }
    }
}

impl Clone for Pin {
    fn clone(&self) -> Self {
        self.handle.pin();
        Self {
            handle: self.handle.clone(),
            shared: self.shared.clone(),
        }
    }
}

impl Drop for Pin {
    fn drop(&mut self) {
        if self.handle.unpin()
            && let Some(shared) = self.shared.upgrade()
        {
            shared.wake.notify_one();
        }
    }
}
