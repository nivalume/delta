# Replicated state (`state`)

Corresponds to upstream `src/state.ts` (`replicatedState`, `MutableReplicatedState`,
`AttachedReplicatedState`, `ReplicatedStateSource`) and to the delivery rules durable's
`watchDoc` adds on top of it.

## Problem

A document's revisions have observers: a UI showing a conversation, a view that re-roots
a document under a prefix, a consumer deriving agent events from ops. Each observer needs
the current value once, then every change in order, without one slow observer holding up
the writer or the others, and with bounded memory when an observer falls behind.

Upstream has two pieces for this. `replicatedState` delivers complete values to
callbacks, serialized per subscription, keeping at most 100 pending values and dropping to
the newest on overflow. Durable's `watchDoc` delivers `(value, ops)` and, past 100 pending
frames, turns the backlog into one `["r", newest]`. Values from storage-driven documents
enter through a separate "attached" form that checks frame cursors.

delta has one type that delivers both the value and the ops, and pull-based subscriptions
instead of callbacks.

## `State`

```rust
#[derive(Clone)]
pub struct State { /* Arc<Mutex<Inner>>: tracker, sequence, subscriber queues */ }

impl State {
    pub fn new(initial: Json) -> Self;
    pub fn value(&self) -> Json;                  // O(1)
    pub fn sequence(&self) -> u64;                // number of published changes

    // One-step change: run the closure on a draft, adopt, publish.
    pub fn change<R>(&self, ctx: &Context, edit: impl FnOnce(Draft<'_>) -> R) -> R;
    pub fn replace(&self, ctx: &Context, value: Json);           // diff against the current value

    // Two-step change, for a writer that persists the batch first.
    pub fn begin(&self) -> Change;
    pub fn commit(&self, ctx: &Context, prepared: Prepared) -> Result<(), AdoptError>;

    // A batch made elsewhere, relayed as it is (a mirror of another state).
    pub fn apply(&self, ctx: &Context, ops: Vec<Op>) -> Result<(), ApplyError>;

    pub fn subscribe(&self) -> Subscription;
    pub fn subscribe_with(&self, options: SubscribeOptions) -> Subscription;
}
```

- `State` is a cheap handle (`Clone`, `Send`, `Sync`). Every clone is the same state.
- `change` runs `edit` while holding the state's lock, so concurrent changes from other
  threads wait and never go stale. `edit` is synchronous, as upstream's. A change from
  inside `edit` on the same state is a programming error and panics with a message (it
  would otherwise deadlock). If `edit` panics, the change is dropped and the state keeps
  its value.
- `replace` diffs the new value against the current one (see [delta.md](delta.md)). A value
  built from the current one shares its unchanged containers, so the diff costs time
  proportional to the change. Upstream's `replace` publishes `["r", value]` instead.
- `begin` / `commit` is the durable path: begin a change, edit it (across `.await` if
  needed), `prepare` it, write the ops to storage, then `commit`. `commit` adopts and
  publishes; it fails with `AdoptError::Stale` when another change was committed since
  `begin`.
- `apply` applies a batch to the current value and publishes exactly that batch
  (`Tracker::prepare_ops`). A harness view mounts several documents into one state and
  relays each commit's ops re-rooted under the document's key, so its subscribers see the
  writer's ops, not a diff. A batch that does not apply changes nothing.
- Empty batches are adopted but not published. `sequence()` counts published changes.

## Subscriptions

```rust
pub struct Subscription;

impl Subscription {
    pub async fn next(&mut self) -> Option<Delivery>;
    pub fn try_next(&mut self) -> Option<Delivery>;        // a queued delivery, without waiting
    pub fn into_stream(self) -> impl Stream<Item = Delivery>;
}

pub struct Delivery {
    pub kind: DeliveryKind,        // Hydrate | Update
    pub value: Json,
    pub ops: Arc<[Op]>,            // from the previous delivery's value to `value`
    pub sequence: u64,
    pub skipped: u64,              // updates merged into this one
    pub context: Context,          // the context of the (last) change
}
```

- The first delivery is `Hydrate` with the current value and no ops. Every later delivery
  is an `Update` whose ops turn the previous delivery's value into this one, so a
  subscriber can maintain a replica or derive events from either.
- Deliveries are queued per subscription while the state's lock is held, so every
  subscriber sees changes in commit order. There is no process-wide publication queue, and
  a slow subscriber delays only itself (upstream's queue is process-wide and synchronous,
  which its spec lists as an open problem).
- `next()` returns `None` once every `State` handle is dropped and the queue is drained.
  Dropping a `Subscription` unsubscribes.

### Backlog

A subscription keeps at most `max_pending` deliveries (default 100). When a new change
would exceed it, the pending deliveries merge into one:

- `value`, `sequence` and `context` are the newest change's.
- `ops` are the pending batches concatenated in order; a batch sequence applied in order
  is itself a valid batch. If the result would exceed `MAX_OPS`, it is `["r", value]`
  instead.
- A pending `Hydrate` stays a `Hydrate` with the newest value.
- `skipped` counts the updates absorbed into the result: every queued update, plus the
  newest one when the result is a hydration.

So memory per subscription is bounded, and a subscriber that falls behind still receives
ops it can apply. Upstream's `replicatedState` instead drops the backlog to the newest
complete value and skips sequence numbers; `watchDoc` does what delta does, with an `r`.

## Differences from upstream

- Pull-based subscriptions instead of callbacks. Upstream's callback model needs rules
  for synchronous vs. asynchronous listeners, for errors thrown by listeners (reported
  through `onError` or a microtask), and for unsubscribing while a callback runs; a stream
  that the subscriber polls needs none of them. A subscriber that wants a callback spawns
  a task that loops over `next()`.
- Deliveries carry ops. Upstream's `replicatedState` delivers values only.
- No attached form. Upstream's `AttachedReplicatedState` wraps an external source that
  emits `{cursor, value, ops}` frames and checks that cursors are consecutive. In delta,
  a producer with its own storage owns the `State` and calls `commit` after its write, so
  ordering is guaranteed by the tracker and there is nothing to check.
- `replace` diffs (upstream publishes a replacement).
- Hydration is a delivery like any other, not a synchronous callback during `subscribe`.
