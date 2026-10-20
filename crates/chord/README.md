# delta-chord

JSON revisions with op batches, replicated state, explicit contexts, bounded output and
SQLite storage helpers. Part of [delta](https://github.com/nivalume/pi-rs); import it as
`delta_chord`.

delta's counterpart of the base library of
[`@earendil-works/chord`](https://github.com/earendil-works/pi/tree/main/packages/chord). It
solves the same problems with its own API: values are an `Arc` copy-on-write tree, drafts
record edits as they happen, and subscriptions are pulled rather than called back. Ops keep
upstream's tuple form, so stored batches move between pi and delta. The design notes in
[`docs/design/`](docs/design/README.md) list every difference from upstream.

- `json`: `Json`, shared between revisions; cloning is O(1), and mutating copies only the
  touched path
- `delta`: `Tracker`, `Change`, drafts, `Op`, `apply`, `diff`
- `state`: `State`, an observable revision sequence, and its subscriptions
- `context`: `Context`, cancellation with reasons, typed values
- `output`: `OutputBuffer`, `bound_output`, `Progress`
- `sqlite` (feature `sqlite`): `Database` on its own thread, transactions, migrations

## Installation

```bash
cargo add delta-chord
cargo add delta-chord --features sqlite   # with the SQLite helpers
```

## Changes and op batches

A change edits a copy of the current revision through drafts. `prepare()` returns the new
revision and the ops from the old one; a writer stores the ops, then adopts the preparation.

```rust
use delta_chord::delta::{Tracker, applied};
use delta_chord::json;

let mut tracker = Tracker::new(json!({"output": "", "entries": []}));
let mut change = tracker.begin();
{
    let mut draft = change.draft();
    draft.field("output").unwrap().into_string().unwrap().push_str("done\n");
    draft.field("entries").unwrap().into_array().unwrap().push(json!({"id": 1}));
}
let prepared = change.prepare();
assert_eq!(
    serde_json::to_string(prepared.ops()).unwrap(),
    r#"[["a",["output"],"done\n"],["p",["entries"],0,0,[{"id":1}]]]"#,
);

// A replica applies the same ops; the base revision is untouched.
let replica = applied(prepared.base(), prepared.ops()).unwrap();
assert_eq!(&replica, prepared.value());

tracker.adopt(prepared).unwrap();
assert_eq!(tracker.value(), &replica);
```

Assigning a value records the diff against the value it replaces, so a tool's output window
assigned as one string becomes an append, or a trim plus an append when the window slides:

```rust
use delta_chord::delta::Tracker;
use delta_chord::json;

let tracker = Tracker::new(json!({"output": "line 1\nline 2\n"}));
let mut change = tracker.begin();
change.draft().field("output").unwrap().set("line 2\nline 3\n");
assert_eq!(
    serde_json::to_string(change.prepare().ops()).unwrap(),
    r#"[["t",["output"],7],["a",["output"],"line 3\n"]]"#,
);
```

## Replicated state

```rust
use delta_chord::context::Context;
use delta_chord::json;
use delta_chord::state::{DeliveryKind, State};

# tokio_test::block_on(async {
let state = State::new(json!({"count": 0}));
let mut subscription = state.subscribe();
state.change(&Context::background(), |mut draft| draft.field("count").unwrap().set(1));

let first = subscription.next().await.unwrap();
assert_eq!(first.kind, DeliveryKind::Hydrate);
assert_eq!(first.value, json!({"count": 0}));

let update = subscription.next().await.unwrap();
assert_eq!(update.value, json!({"count": 1}));
assert_eq!(serde_json::to_string(&*update.ops).unwrap(), r#"[["s",["count"],1]]"#);
# });
```

## Contexts

```rust
use delta_chord::context::{CancelReason, Context, ContextKey};

static USER: ContextKey<String> = ContextKey::new("user");

# tokio_test::block_on(async {
let (context, canceller) = Context::background().with_value(&USER, "ann".to_owned()).with_cancel();
assert_eq!(context.value(&USER).unwrap(), "ann");

canceller.cancel_with(CancelReason::message("user abort"));
let error = context.run(std::future::pending::<()>()).await.unwrap_err();
assert_eq!(error.to_string(), "user abort");
// `context.token()` is a CancellationToken for delta-ai and delta-agent.
assert!(context.token().is_cancelled());
# });
```
