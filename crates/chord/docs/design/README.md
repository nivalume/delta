# delta-chord (`delta_chord`) design notes

`delta-chord` is delta's counterpart of the base library of
[`@earendil-works/chord`](https://github.com/earendil-works/pi/tree/main/packages/chord):
explicit contexts, JSON revisions with op batches, replicated state, bounded output and
SQLite storage helpers. Code imports it as `use delta_chord::...`. Its first consumer will
be delta's durable harness.

Upstream is the reference for which problems this crate solves and how the pieces fit
together. The API and the algorithms are delta's own wherever Rust offers a better fit;
each document lists the differences from upstream so they stay visible. Upstream chord's
host runtime (objects, contracts, links, hot reload) is out of scope; see
[docs/research/chord.md](../../../../docs/research/chord.md).

## Document index

| Document | Module | Contents |
| --- | --- | --- |
| [json.md](json.md) | `json` | `Json`, the `Arc` copy-on-write JSON tree; paths; serde bridges |
| [delta.md](delta.md) | `delta` | `Op` and its wire shape, `apply`, `Tracker` / `Change` / `Prepared`, drafts, `diff` |
| [state.md](state.md) | `state` | `State`: an observable revision sequence; subscriptions that deliver values and ops |
| [context.md](context.md) | `context` | `Context`: cancellation with reasons, typed values |
| [output.md](output.md) | `output` | `OutputBuffer`, `bound_output`, `sanitize_output`, `Progress` |
| [sqlite.md](sqlite.md) | `sqlite` | `Database` on a dedicated thread, transactions, migrations |

## How the pieces fit

```
             Context ─────────────── passed to every change and delivered with it
                │
  Tracker ── Change ── Draft         record edits; prepare() yields value + ops
     │          │
     │       Prepared ──► storage writes ops (durable), then adopt / commit
     │          │
   State ◄──────┘                    adopts and fans out to subscribers
     │
  Subscription ──► Delivery { value, ops, sequence, context }

  Json ── shared by every revision; clone is O(1), mutation copies only the touched path
```

- A **revision** is a `Json` value. Revisions share every container they did not change.
- A **change** starts from the current revision, records edits through a draft, and
  `prepare()` yields the new revision and the op batch from the old one to the new one.
- **Adopting** a prepared change makes it current. Durable writes the batch to storage
  first and adopts only once the write succeeded.
- A **state** wraps a tracker and publishes every adopted revision, with its batch, to
  subscribers.

## Crate layout

```
crates/chord/
  docs/design/      # this directory
  src/
    lib.rs
    json.rs         # Json, Path, serde bridges
    delta/
      mod.rs        # Op and its tuple form
      apply.rs      # apply, applied, ApplyError
      tracker.rs    # Tracker, Change, Prepared
      draft.rs      # Draft handles and the op log
      diff.rs       # diff of two revisions
    state.rs        # State, Subscription, Delivery
    context.rs      # Context, Canceller, ContextKey
    output.rs       # OutputBuffer, bound_output, Progress
    sqlite.rs       # Database, migrations (feature "sqlite")
  tests/
```

| feature | Default | Contents |
| --- | --- | --- |
| `sqlite` | off | The `sqlite` module, on `rusqlite` with the bundled SQLite |

Dependencies: `serde`, `serde_json`, `indexmap`, `thiserror`, `tokio`, `tokio-util`,
`futures`; `rusqlite` with `sqlite`.

## Confirmed design decisions

| Question | Decision |
| --- | --- |
| Scope | The base library only. The host runtime waits for a consumer in delta and for upstream's design to settle |
| JSON representation | An `Arc`-shared copy-on-write tree (`Json`), not `serde_json::Value`. See [json.md](json.md) |
| Op sequences | delta's tracker does not reproduce upstream's op sequences. Batches must be correct and as fine-grained as the edit allows (an append is an `a`, a push is an insert-only `p`) |
| Op wire shape | Upstream's tuples and their meaning, `t` counted in UTF-16 code units included, so stored batches and transcripts move between pi and delta. See [delta.md](delta.md) |
| Immutability | No ownership rules: every `Json` is a value, and mutating one never affects another holder. Upstream's "never mutate what you transferred" contract and its footguns go away |
| Drafts | Explicit handles. Structural edits (push, insert, remove, splice, append) are recorded as they happen; assignments are diffed against the value they replace. Upstream reconstructs every op from base vs. final at `prepare()` |
| Change lifecycle | Ownership replaces runtime checks: `prepare(self)` consumes the change, `adopt` consumes the preparation, dropping either aborts it. A stale preparation is refused by revision number |
| Subscriptions | Pull-based (`Subscription::next().await`). A delivery carries the value and the ops since the subscriber's previous delivery; on overflow, pending deliveries merge into one with concatenated ops, or a replacement past a size cap |
| Attached replicated state | Not a separate type. A producer with its own storage owns a `State` and commits prepared changes into it after its write |
| Cancellation | Built on `CancellationToken`, like `delta_ai`, plus an optional cancellation reason. Waiting is `ctx.run(fut)`: dropping the future on cancellation, as everywhere in delta |
| Telemetry | Not in `Context` yet. It waits for delta's telemetry design, as in `delta_ai` |
| SQLite | One concrete driver (`rusqlite`) on a dedicated thread. Transaction bodies are synchronous closures on that thread; no portable async trait |
| Encoder/decoder (path interning) | Not ported. It only serves the host runtime's links |
| Package name | crates.io package `delta-chord`, library `delta_chord` |

## Not ported

- The host runtime (see the research note).
- `telemetry-sqlite`: waits for delta's telemetry crate.
- The wire `encoder`/`decoder`, `JsonRevisionValidator` and `copyJson`: the first two serve
  links and untrusted roots in JS; `Json` is strict by construction, so `copyJson` has
  nothing left to do.
- The Cloudflare Durable Object SQLite driver.
