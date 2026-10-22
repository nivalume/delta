//! The document cache: current incarnations loaded on the line, each a chord [`State`]
//! behind a [`DocHandle`], kept while used or observed and evicted after the retention.
//!
//! Every cached address is in the idle index once, keyed by the time the line last
//! recorded a use. Uses only stamp the handle (an atomic), so a hot document costs no
//! reordering; the sweep re-keys an entry it finds used since, pinned, or observed by an
//! older-version state, and evicts the rest. A pinned entry is re-checked once per
//! retention period.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock};
use std::time::Duration;

use delta_chord::context::Context;
use delta_chord::delta::{AdoptError, Change, Prepared};
use delta_chord::json::Json;
use delta_chord::state::{State, Subscription};
use tokio::sync::watch;
use tokio::time::Instant;

use super::error::TxError;
use super::guard::Guard;
use super::observation::WatchEnd;
use crate::documents::{DocDefinition, check_record, upgrade};
use crate::records::{DocumentAddress, DocumentPoint, DocumentRecord};

/// One cached incarnation's observable value: shared by the cache, the transactions that
/// edit it and the [`DocumentState`](super::DocumentState)s attached to it.
pub(super) struct DocHandle {
    record: DocumentRecord,
    /// The definition version the value is shaped for.
    value_version: u32,
    slot: Mutex<Slot>,
    end: watch::Sender<Option<WatchEnd>>,
    pins: AtomicUsize,
    /// Milliseconds after `epoch` of the last use.
    last_use: AtomicU64,
    epoch: Instant,
}

/// The value: a live chord state (the Session holds the only writable handle), or the
/// final value once ended. Dropping the state ends its subscriptions.
enum Slot {
    Live(State),
    Ended(Json),
}

impl DocHandle {
    fn new(record: DocumentRecord, value_version: u32, state: State, epoch: Instant) -> Arc<Self> {
        let handle = Arc::new(Self {
            record,
            value_version,
            slot: Mutex::new(Slot::Live(state)),
            end: watch::Sender::new(None),
            pins: AtomicUsize::new(0),
            last_use: AtomicU64::new(0),
            epoch,
        });
        handle.touch();
        handle
    }

    fn slot(&self) -> MutexGuard<'_, Slot> {
        self.slot.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(super) fn record(&self) -> &DocumentRecord {
        &self.record
    }

    pub(super) fn value_version(&self) -> u32 {
        self.value_version
    }

    /// The last published revision, or the final value once ended.
    pub(super) fn value(&self) -> Json {
        match &*self.slot() {
            Slot::Live(state) => state.value(),
            Slot::Ended(value) => value.clone(),
        }
    }

    /// The current value while live.
    pub(super) fn live_value(&self) -> Option<Json> {
        match &*self.slot() {
            Slot::Live(state) => Some(state.value()),
            Slot::Ended(_) => None,
        }
    }

    pub(super) fn is_ended(&self) -> bool {
        matches!(&*self.slot(), Slot::Ended(_))
    }

    /// A change of the current revision; `None` once ended.
    pub(super) fn begin(&self) -> Option<Change> {
        match &*self.slot() {
            Slot::Live(state) => Some(state.begin()),
            Slot::Ended(_) => None,
        }
    }

    /// Adopts and publishes a prepared change; an ended handle delivers nothing.
    pub(super) fn commit(&self, context: &Context, prepared: Prepared) -> Result<(), AdoptError> {
        match &*self.slot() {
            Slot::Live(state) => state.commit(context, prepared),
            Slot::Ended(_) => Ok(()),
        }
    }

    /// Publishes `value` as a diff from the current one (observers of an older shape).
    pub(super) fn replace(&self, context: &Context, value: Json) {
        if let Slot::Live(state) = &*self.slot() {
            state.replace(context, value);
        }
    }

    /// Publishes `["r", null]` and ends with [`WatchEnd::Retired`].
    pub(super) fn retire(&self, context: &Context) {
        self.replace(context, Json::Null);
        self.end(WatchEnd::Retired);
    }

    /// Ends the handle, once: the value freezes and every subscription ends after its queued
    /// deliveries.
    pub(super) fn end(&self, reason: WatchEnd) {
        let state = {
            let mut slot = self.slot();
            match &*slot {
                Slot::Live(state) => {
                    let state = state.clone();
                    *slot = Slot::Ended(state.value());
                    Some(state)
                }
                Slot::Ended(_) => None,
            }
        };
        drop(state);
        self.end.send_if_modified(|end| {
            if end.is_none() {
                *end = Some(reason);
                true
            } else {
                false
            }
        });
    }

    pub(super) fn end_reason(&self) -> Option<WatchEnd> {
        self.end.borrow().clone()
    }

    pub(super) fn watch_end(&self) -> watch::Receiver<Option<WatchEnd>> {
        self.end.subscribe()
    }

    /// A subscription to the live state; once ended, one that hydrates with the final value
    /// and ends.
    pub(super) fn subscribe(&self) -> Subscription {
        match &*self.slot() {
            Slot::Live(state) => state.subscribe(),
            Slot::Ended(value) => State::new(value.clone()).subscribe(),
        }
    }

    pub(super) fn pin(&self) {
        self.pins.fetch_add(1, Ordering::AcqRel);
    }

    /// Returns whether this was the last pin.
    pub(super) fn unpin(&self) -> bool {
        self.touch();
        self.pins.fetch_sub(1, Ordering::AcqRel) == 1
    }

    pub(super) fn is_pinned(&self) -> bool {
        self.pins.load(Ordering::Acquire) > 0
    }

    pub(super) fn touch(&self) {
        self.last_use.store(millis(self.epoch), Ordering::Release);
    }

    fn last_use(&self) -> u64 {
        self.last_use.load(Ordering::Acquire)
    }
}

fn millis(epoch: Instant) -> u64 {
    u64::try_from(epoch.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// The cached current incarnations by address, readable off the line (snapshot cache hits).
#[derive(Default)]
pub(super) struct DocIndex(RwLock<HashMap<DocumentAddress, Arc<DocHandle>>>);

impl DocIndex {
    pub(super) fn get(&self, address: &DocumentAddress) -> Option<Arc<DocHandle>> {
        self.0
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(address)
            .cloned()
    }

    fn insert(&self, address: DocumentAddress, handle: Arc<DocHandle>) {
        self.0
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(address, handle);
    }

    fn remove(&self, address: &DocumentAddress) {
        self.0.write().unwrap_or_else(PoisonError::into_inner).remove(address);
    }

    fn clear(&self) {
        self.0.write().unwrap_or_else(PoisonError::into_inner).clear();
    }
}

/// A cached incarnation as a transaction sees it at acquisition.
pub(super) struct Loaded {
    pub(super) handle: Arc<DocHandle>,
    /// The persisted version; older while the value is migrated only in memory.
    pub(super) stored_version: u32,
    pub(super) deltas_since_base: u64,
}

struct CachedDocument {
    handle: Arc<DocHandle>,
    stored_version: u32,
    deltas_since_base: u64,
    /// Attached states of this incarnation hydrated under another definition version.
    stale: Vec<Arc<DocHandle>>,
    idle_key: Option<(u64, u64)>,
}

/// The line's document cache.
pub(super) struct Cache {
    entries: HashMap<DocumentAddress, CachedDocument>,
    idle: BTreeMap<(u64, u64), DocumentAddress>,
    next_tie: u64,
    /// `None`: keep forever.
    retention: Option<Duration>,
    epoch: Instant,
    shared: Arc<super::lifecycle::Shared>,
}

impl Cache {
    pub(super) fn new(retention: Option<Duration>, shared: Arc<super::lifecycle::Shared>) -> Self {
        Self {
            entries: HashMap::new(),
            idle: BTreeMap::new(),
            next_tie: 0,
            retention,
            epoch: Instant::now(),
            shared,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    /// The current incarnation at `address` for `D`, from the cache or loaded from storage
    /// and migrated in memory; `None` when absent. A cached value shaped for another version
    /// is reloaded from the stored bytes; its attached states stay subscribed as observers of
    /// the older shape.
    pub(super) async fn load<D: DocDefinition>(
        &mut self,
        guard: &Guard,
        address: &DocumentAddress,
        ctx: &Context,
    ) -> Result<Option<Loaded>, TxError> {
        if let Some(entry) = self.entries.get(address)
            && entry.handle.value_version == D::VERSION
            && !entry.handle.is_ended()
        {
            check_record::<D>(&entry.handle.record)?;
            entry.handle.touch();
            return Ok(Some(entry.loaded()));
        }
        let previous = self.take(address);
        match fetch::<D>(guard, address, ctx).await {
            Ok(Some((record, stored_version, deltas_since_base, value))) => {
                let mut stale = Vec::new();
                if let Some(previous) = previous.filter(|previous| previous.handle.record.id == record.id) {
                    stale = previous.stale;
                    stale.push(previous.handle);
                    stale.retain(|handle| handle.is_pinned() && !handle.is_ended());
                }
                let handle = DocHandle::new(record, D::VERSION, State::new(value), self.epoch);
                self.install(address.clone(), handle, stored_version, deltas_since_base, stale);
                Ok(self.entries.get(address).map(CachedDocument::loaded))
            }
            Ok(None) => Ok(None),
            Err(error) => {
                if let Some(previous) = previous {
                    self.put(address.clone(), previous);
                }
                Err(error)
            }
        }
    }

    /// Caches an incarnation created by a commit.
    pub(super) fn install_created(
        &mut self,
        address: DocumentAddress,
        record: DocumentRecord,
        version: u32,
        state: State,
    ) {
        let handle = DocHandle::new(record, version, state, self.epoch);
        self.take(&address);
        self.install(address, handle, version, 0, Vec::new());
    }

    /// Records an adopted change of `handle`: the stored version, the delta count, and the
    /// new value for observers of an older shape.
    pub(super) fn changed(&mut self, handle: &Arc<DocHandle>, change: AdoptedChange<'_>, ctx: &Context) {
        let Some(entry) = self.entries.get_mut(&handle.record.address()) else {
            return;
        };
        if !Arc::ptr_eq(&entry.handle, handle) {
            return;
        }
        entry.stored_version = entry.stored_version.max(change.version);
        match change.base {
            Some(true) => entry.deltas_since_base = 0,
            Some(false) => entry.deltas_since_base += 1,
            None => {}
        }
        entry.stale.retain(|stale| stale.is_pinned() && !stale.is_ended());
        for stale in &entry.stale {
            stale.replace(ctx, change.value.clone());
        }
        handle.touch();
    }

    /// Ends and evicts the cached incarnation `record` at its address, if cached: its
    /// states receive `["r", null]` and end with [`WatchEnd::Retired`].
    pub(super) fn retire(&mut self, record: &DocumentRecord, ctx: &Context) {
        let address = record.address();
        if self
            .entries
            .get(&address)
            .is_some_and(|entry| entry.handle.record.id == record.id)
            && let Some(entry) = self.take(&address)
        {
            entry.handle.retire(ctx);
            for stale in &entry.stale {
                stale.retire(ctx);
            }
        }
    }

    /// Ends every cached handle with `reason` and empties the cache (at close).
    pub(super) fn clear(&mut self, reason: &WatchEnd) {
        for entry in self.entries.values() {
            entry.handle.end(reason.clone());
            for stale in &entry.stale {
                stale.end(reason.clone());
            }
        }
        self.entries.clear();
        self.idle.clear();
        self.shared.documents.clear();
    }

    /// Drops the cached handle of `address` when it is `handle` (an ended handle met by a
    /// transaction during close).
    pub(super) fn forget(&mut self, address: &DocumentAddress, handle: &Arc<DocHandle>) {
        if self
            .entries
            .get(address)
            .is_some_and(|entry| Arc::ptr_eq(&entry.handle, handle))
        {
            self.take(address);
        }
    }

    fn install(
        &mut self,
        address: DocumentAddress,
        handle: Arc<DocHandle>,
        stored_version: u32,
        deltas_since_base: u64,
        stale: Vec<Arc<DocHandle>>,
    ) {
        let entry = CachedDocument {
            handle,
            stored_version,
            deltas_since_base,
            stale,
            idle_key: None,
        };
        self.put(address, entry);
    }

    fn put(&mut self, address: DocumentAddress, mut entry: CachedDocument) {
        entry.idle_key = None;
        self.shared.documents.insert(address.clone(), entry.handle.clone());
        self.entries.insert(address.clone(), entry);
        let now = self.now();
        self.mark_idle(address, now);
    }

    fn take(&mut self, address: &DocumentAddress) -> Option<CachedDocument> {
        let entry = self.entries.remove(address)?;
        if let Some(key) = entry.idle_key {
            self.idle.remove(&key);
        }
        self.shared.documents.remove(address);
        Some(entry)
    }

    fn mark_idle(&mut self, address: DocumentAddress, at: u64) {
        if self.retention.is_none() {
            return;
        }
        let Some(entry) = self.entries.get_mut(&address) else {
            return;
        };
        if let Some(key) = entry.idle_key.take() {
            self.idle.remove(&key);
        }
        self.next_tie += 1;
        let key = (at, self.next_tie);
        entry.idle_key = Some(key);
        self.idle.insert(key, address);
    }

    fn now(&self) -> u64 {
        millis(self.epoch)
    }

    fn retention_millis(&self) -> Option<u64> {
        self.retention
            .map(|retention| u64::try_from(retention.as_millis()).unwrap_or(u64::MAX))
    }

    /// When the oldest idle entry expires; `None` without a positive retention (a zero
    /// retention sweeps after every job instead).
    pub(super) fn next_sweep(&self) -> Option<Instant> {
        let retention = self.retention_millis().filter(|&retention| retention > 0)?;
        let ((used, _), _) = self.idle.first_key_value()?;
        let at = used.saturating_add(retention);
        Some(self.epoch + Duration::from_millis(at))
    }

    /// Evicts the idle entries whose retention passed, oldest first. An entry used since it
    /// was keyed is re-keyed at that use; a pinned one (or one with older-version observers)
    /// at now.
    pub(super) fn sweep(&mut self) {
        let Some(retention) = self.retention_millis() else {
            return;
        };
        let now = self.now();
        let mut rekey = Vec::new();
        while let Some(first) = self.idle.first_entry() {
            let (used, _) = *first.key();
            if used.saturating_add(retention) > now {
                break;
            }
            let address = first.remove();
            let Some(entry) = self.entries.get_mut(&address) else {
                continue;
            };
            entry.idle_key = None;
            entry.stale.retain(|stale| stale.is_pinned() && !stale.is_ended());
            let last = entry.handle.last_use();
            if entry.handle.is_pinned() || !entry.stale.is_empty() {
                rekey.push((now, address));
            } else if last > used && last.saturating_add(retention) > now {
                rekey.push((last, address));
            } else {
                self.take(&address);
            }
        }
        for (at, address) in rekey {
            self.mark_idle(address, at);
        }
    }
}

impl CachedDocument {
    fn loaded(&self) -> Loaded {
        Loaded {
            handle: self.handle.clone(),
            stored_version: self.stored_version,
            deltas_since_base: self.deltas_since_base,
        }
    }
}

/// What adoption tells the cache about a loaded incarnation's change.
#[derive(Clone, Copy)]
pub(super) struct AdoptedChange<'a> {
    pub(super) version: u32,
    /// `Some(true)`: a base was stored; `Some(false)`: a delta; `None`: nothing.
    pub(super) base: Option<bool>,
    pub(super) value: &'a Json,
}

/// Reads the current incarnation at `address` and brings its value to `D::VERSION`.
async fn fetch<D: DocDefinition>(
    guard: &Guard,
    address: &DocumentAddress,
    ctx: &Context,
) -> Result<Option<(DocumentRecord, u32, u64, Json)>, TxError> {
    let found = guard
        .read(ctx, |storage| {
            storage.find_document(address, DocumentPoint::Current, ctx)
        })
        .await?;
    let Some(record) = found else {
        return Ok(None);
    };
    let stored = guard
        .read(ctx, |storage| storage.document(record.id, DocumentPoint::Current, ctx))
        .await?;
    let Some(stored) = stored else {
        return Err(guard.corrupt(format!(
            "Current document {} ({}) cannot be read",
            record.id, record.kind
        )));
    };
    check_record::<D>(&stored.record)?;
    let upgraded = upgrade::<D>(stored.record.id, stored.version, stored.value)?;
    Ok(Some((
        stored.record,
        stored.version,
        stored.deltas_since_base,
        upgraded.value,
    )))
}
