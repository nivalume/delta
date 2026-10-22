# Upstream chord: what it is and what delta takes from it

Notes from reading `@earendil-works/chord` and its consumers in the pi monorepo, used to
scope `delta-chord`.

Read at: pi `73f8de8` (2026-10-19), chord 1.3.0 plus `[Unreleased]`.

## Two packages in one

Chord ships two layers that share a name but little else.

| Layer | Modules | Size (src) | Used by |
| --- | --- | --- | --- |
| Base library | `context`, `json`, `delta` (tracker, ops, apply, diff, wire encoder), `state` (`replicatedState`), `output`, `sqlite`, `telemetry-sqlite` | ≈5k lines, 2.2k of them the draft tracker | `pi-durable` (≈50 files), `pi-env` |
| Host runtime | contracts, objects, cells, references, principals, hosts, links, wire protocol, hot reload, cluster reload, process manager | ≈8.7k lines | nothing in the monorepo |

### Base library

- **Context**: an immutable value passed explicitly to every operation. It carries
  cancellation (`AbortSignal`), a telemetry span and typed values. Deriving one never
  changes its parent.
- **Delta**: a transactional JSON tracker. A change hands out a draft (a `Proxy`), and
  `prepare()` produces the next immutable revision plus the op batch that turns the base
  into it. `adopt()` makes the revision current. The two phases exist so storage can
  write the batch between them.
- **Ops** are JSON tuples: `["r", v]`, `["s", path, v]`, `["d", path]`, `["a", path, text]`,
  `["t", path, utf16Count]`, `["p", path, index, remove, items]`, `["m", path, perm]`. The
  same tuples are used in memory, on the wire and on disk.
- **Replicated state**: an observable JSON value. A subscriber gets the current value, then
  every new one, with per-subscriber serialization and a 100-frame backlog that coalesces
  to the newest value.
- **Output**: `OutputBuffer` keeps a head or tail window of a stream (tool or process
  output); `Progress` paces commits of a changing value by their size.
- **SQLite**: a portable async facade (Node and Cloudflare Durable Object drivers), an
  atomic migration runner and schema-prefix validation.

What `pi-durable` depends on:

- Persisted documents are a base value plus delta rows whose content is `JSON(Op[])`
  (`document_revisions`). The op tuple shape is therefore an on-disk format.
- `harness/events.ts` reads op shapes to derive agent events: an `a` on a text block
  becomes `text_delta`, a `p` that only inserts becomes `*_start`. Other shapes still
  produce correct events, but coarser ones (whole message or block).
- Tool progress assigns the whole output window as one string and relies on the
  tracker turning that into `a`, or `t` plus `a` for a sliding tail.
- Drafts stay open across `await`, and several documents are open in one transaction.

Not used by durable: the wire encoder/decoder (path interning), `diffRevisions`, the
revision validator, and everything in the host runtime.

### Host runtime

A system of stateful objects that live in several processes (server, workers, browser
tabs) and can be hot reloaded while they run:

- An object is addressed by (type, normalized input, principal), created on demand and
  disposed when the last reference is released. Callers only hold references, which work
  the same locally and across links and survive reloads.
- Contracts are runtime values built from TypeBox schemas, with versions, a fingerprint, a
  structural coverage check and `accepts` adapters for old consumers. The contract also
  says which host the objects live on.
- Principals select variants: an owner and a guest get different members.
- State is replicated as a snapshot followed by op batches, with end-to-end flow control
  (ack windows, per-stream slots, coalescing to resets) and deflate for large payloads.
- Hot reload runs in phases: load new modules with a cache-busted `import()`, re-run every
  live object's factory in a recording mode, gate objects whose state version rose
  (migrate, swap exclusive resources), commit in one synchronous step, and retire the
  previous code version once its roots end. Calls in flight finish on the code they
  started with.

Status:

- The current runtime replaced the previous one on 2026-10-12. Release 1.3.0, three days
  later, carried about 15 breaking changes; the link protocol went from version 1 to 4
  within those three days.
- The spec lists twelve open problems, among them: the commit decision is not persisted,
  links do not resume, the module loader leaks every reloaded copy, and the publication
  queue is process-wide and synchronous.
- `pi-coding-agent` removed its experimental server/client/plugin stack, the runtime's
  only consumer, in 1.2.0.

## Decisions for delta

Confirmed with the maintainer:

1. `delta-chord` covers the base library only. The host runtime waits until delta has a
   consumer for it and upstream's design settles.
2. JSON values are an `Arc`-shared copy-on-write tree.
3. delta's tracker does not have to emit the same op sequences as upstream. Upstream is a
   reference for the problem and the high-level design; delta is free to design its own
   API.

Design notes: [crates/chord/docs/design](../../crates/chord/docs/design/README.md).

## What does not carry over to Rust

These shape the base library design too:

| Upstream mechanism | Relies on | delta |
| --- | --- | --- |
| Immutability "by contract": nothing is frozen, a mutation of a shared container corrupts other revisions | JS object sharing | `Arc` copy-on-write: any `Json` may be mutated, and a mutation never reaches another holder |
| Drafts | `Proxy` over an overlay, a treap piece table so held handles follow array elements through reindexing | Explicit draft handles that borrow the change; the borrow checker rules out stale handles, so the reindexing machinery is not needed |
| Ops derived from base vs. final value at `prepare()` | Draft records only which nodes were touched | Structural edits are recorded as they happen; assignments are diffed against the value they replace |
| `undefined` (delete on assignment, absent snapshot value) | JS | `Option` / explicit `remove` |
| Reserved path segments `__proto__`, `constructor`, `prototype` | Prototype pollution | Not a hazard for Rust appliers (see the delta design note) |
| Subscriber callbacks, sync or async, with error reporting | Event loop | Pull-based subscription streams |
| Portable SQLite facade with async transaction callbacks | Node and Cloudflare drivers | One driver (`rusqlite`) on its own thread; transaction bodies are synchronous closures |

## Notes for a later host runtime

Kept here so the analysis is not lost; nothing below is planned.

- **Protocol first.** The frame set, `Description` and its coverage check (a structural
  subtype check over a JSON Schema subset), and flow control are plain data and port to
  any language. A Rust host speaking protocol 4 could interoperate with TS hosts and
  browsers.
- **Fingerprints will differ.** The fingerprint hashes TypeBox's schema JSON in UTF-16
  code units. Schemas generated by `schemars` will not match it byte for byte. The
  provider recomputes the consumer's fingerprint from the token it receives, so a mismatch
  only loses the equal-fingerprint fast path.
- **Reload at the process (or WASM) level.** Rust cannot unload code in-process safely, and
  the workspace forbids `unsafe`. A code version would be a worker process: new roots go
  to the new worker while the old one drains, which is the two-version rule. Cells are
  schema-described JSON, so they hand over as a snapshot plus `migrate`. Native resources
  cannot cross processes and would reopen, which is a semantic difference from upstream.
- **Single-threaded hosts.** The runtime depends on JS's run-to-completion for
  correctness: commit in one synchronous step, the publication queue, check-then-act on
  the object table. A host would run on one `current_thread` executor (`LocalSet`), not on
  a multi-threaded runtime with locks.
