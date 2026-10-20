# Revisions and op batches (`delta`)

Corresponds to upstream `src/delta` (`index.ts`, `tracker.ts`, `diff.ts`, `draft.ts`).

## Problem

A document (a conversation, a tool's live output, a task graph) changes in small steps.
Three parties need each step:

- **Storage** persists the step, not the whole document, and must be able to refuse it
  (a failed write leaves the document unchanged).
- **Observers** (UI, views, other processes) keep a replica up to date from the step.
- **Event derivation** reads what kind of edit happened: an append to a text block is a
  `text_delta`, an insert into a content array starts a block.

So a change must produce two things: the next immutable revision and a batch of ops that
turns the previous revision into it. Upstream gets there with a `Proxy` draft over an
overlay that records which nodes were touched, and at `prepare()` it reconstructs the ops
by comparing each touched node's base and final value (≈2.2k lines including a treap that
lets held handles follow array elements through reindexing).

delta records structural edits as they happen and diffs only assignments. Rust's borrow
rules make the reindexing problem disappear: a draft handle cannot outlive the edit that
moves its element.

## Ops

```rust
pub enum Op {
    Replace(Json),                                             // ["r", value]
    Set(Path, Json),                                           // ["s", path, value]
    Delete(Path),                                              // ["d", path]
    Append(Path, Str),                                         // ["a", path, text]
    TrimStart(Path, u64),                                      // ["t", path, utf16_units]
    Splice { path: Path, index: usize, remove: usize, items: Vec<Json> }, // ["p", path, index, remove, items]
    Permute(Path, Vec<usize>),                                 // ["m", path, permutation]
}
```

| Tuple | Meaning |
| --- | --- |
| `["r", value]` | Replace the whole value. The only op that targets the root as a value |
| `["s", path, value]` | Set an object key (insert or overwrite) or an array element (`index == len` appends) |
| `["d", path]` | Delete an object key, or remove an array element |
| `["a", path, text]` | Append to a string |
| `["t", path, count]` | Remove `count` UTF-16 code units from the front of a string |
| `["p", path, index, remove, items]` | Splice an array; the path may be empty (the root is an array) |
| `["m", path, perm]` | Reorder an array: `new[i] = old[perm[i]]`; the path may be empty |

- `Serialize` / `Deserialize` use exactly these tuples. Deserialization checks arity, that
  `s`/`d`/`a`/`t` paths are non-empty, that numbers are non-negative integers, and that a
  permutation is a bijection.
- An `r` is always the first op of a batch when present. `Op::is_base(&[Op])` checks it.
- Ops can be built by hand. Durable builds view batches itself (re-rooting a document's
  ops under a prefix, inserting entries); `Op::prefixed(prefix)` re-roots one op.

### Why keep upstream's tuples

The tuples are what durable stores in its revision rows and what transcripts and views
carry. Keeping their shape and meaning lets stored documents move between pi and delta in
both directions, as `CLAUDE.md` requires for serialized types. It costs one thing: `t`
counts UTF-16 code units, so applying it converts the count to a byte offset by walking
the removed prefix (O(removed text)), and emitting it counts the units of the dropped
prefix.

### Reserved segments

Upstream's appliers refuse the path segments `__proto__`, `constructor` and `prototype`
(prototype pollution in JS), and its tracker never emits them: an edit at or below such a
key is folded into a set of the nearest ancestor above it. A Rust applier has no such
hazard and accepts them. delta's tracker still folds them, so batches it writes stay
readable by pi:

- A recorded edit whose path contains a reserved segment becomes `s` of the ancestor above
  the first reserved segment, with that ancestor's value right after the edit (an O(1)
  clone), or `r` at the root.
- `diff` emits `s` for a whole object node when one of its changed keys is reserved,
  instead of descending into it.

## `apply`

```rust
pub fn apply(value: &mut Json, ops: &[Op]) -> Result<(), ApplyError>;
pub fn applied(base: &Json, ops: &[Op]) -> Result<Json, ApplyError>;   // clone + apply
```

- Copy-on-write makes upstream's three appliers one function. `apply` on a clone of a
  revision leaves the revision untouched (upstream `applyImmutable`). Containers copied by
  one op are unique afterwards, so later ops in the same call, or later batches applied to
  the same value, change them in place (upstream `applyImmutableBatches`, which tracks the
  copies it owns in a `WeakSet`).
- On error, `apply` may have applied a prefix of the batch. `applied` leaves `base` intact
  and returns the error. `ApplyError` names the op index, its path and the reason.
- Strict: every target must exist and have the right kind. Upstream clamps where JS does
  (`splice` past the end, `t` longer than the string) and ignores `d` of a missing key.
  delta refuses those, because they only happen when a replica has diverged, and a replica
  that has diverged should recover from a later `r` instead of carrying on. Batches that
  upstream's tracker emits never hit these cases.
- A `t` count that ends inside a surrogate pair is an error (Rust strings cannot hold half
  of one).

## Tracker, change, preparation

```rust
pub struct Tracker { /* value: Json, revision: u64, id */ }

impl Tracker {
    pub fn new(initial: Json) -> Self;
    pub fn value(&self) -> &Json;
    pub fn revision(&self) -> u64;
    pub fn begin(&self) -> Change;                       // O(1)
    pub fn prepare_replace(&self, value: Json) -> Prepared;   // ["r", value], or empty when equal
    pub fn prepare_diff(&self, value: Json) -> Prepared;      // ops from diff(current, value)
    pub fn adopt(&mut self, prepared: Prepared) -> Result<Vec<Op>, AdoptError>;  // returns the batch
}

pub struct Change { /* base, working: Json, log, revision, tracker id */ }

impl Change {
    pub fn draft(&mut self) -> Draft<'_>;
    pub fn value(&self) -> &Json;                        // the working value so far
    pub fn prepare(self) -> Prepared;
}

pub struct Prepared { /* base, value, ops, revision, tracker id */ }

impl Prepared {
    pub fn base(&self) -> &Json;
    pub fn value(&self) -> &Json;
    pub fn ops(&self) -> &[Op];
    pub fn is_empty(&self) -> bool;
}
```

Lifecycle:

1. `begin()` clones the current revision into the change's working value (O(1)).
2. Edits go through `change.draft()`. They change the working value (copy on write) and
   append ops to the change's log.
3. `prepare(self)` consumes the change. The working value *is* the new revision, so there
   is no materialization step (upstream applies the ops to the base to build it).
4. The caller writes `prepared.ops()` to storage, then calls `adopt(prepared)`. If the write
   fails, it drops `prepared`.

Rules:

- `Change` and `Prepared` are owned, `Send` values. A change may stay open across `.await`,
  and several changes may be open at once, on one tracker or several.
- `adopt` refuses a preparation from another tracker (`AdoptError::Foreign`) and one whose
  base is no longer current (`AdoptError::Stale`): adopting one change makes every other
  change begun from the same revision stale.
- Aborting is dropping. Ownership rules out upstream's runtime checks for consumed, aborted
  or settled drafts and preparations.
- Adopting an empty preparation still advances `revision()`, so it stales competitors, as
  upstream.

### No-op batches

`prepare` returns an empty batch, with `value()` the same allocation as `base()`, when the
log is empty or the working value equals the base. Equality is cheap because unchanged
subtrees are shared. So a write that restores the original value, or an assignment of an
equal value, is a no-op.

### Folding

A batch longer than `MAX_OPS` (4096) is replaced by `[["r", value]]` at `prepare`, so replay
cost stays bounded. Upstream folds at the same count.

## Drafts

A draft is a handle on one place in the working value. It borrows the change mutably, so
at most one handle is usable at a time and none outlives the change.

```rust
pub struct Draft<'a>;          // any value
pub struct ObjectDraft<'a>;
pub struct ArrayDraft<'a>;
pub struct StrDraft<'a>;
```

| Handle | Methods |
| --- | --- |
| `Draft` | `value() -> &Json`, `set(Json)`, `update(FnOnce(&mut Json))`, `update_as::<T>(FnOnce(&mut T))`, `field(key) -> Option<Draft>`, `index(i) -> Option<Draft>`, `at(&[Seg]) -> Option<Draft>`; `object()`, `array()`, `string()` (borrowing) and `into_object()`, `into_array()`, `into_string()` (consuming), each `None` for another kind |
| `ObjectDraft` | `get(key) -> Option<&Json>`, `field(key) -> Option<Draft>`, `field_or_insert(key, default) -> Draft`, `insert(key, Json) -> Option<Json>`, `remove(key) -> Option<Json>`, `len`, `keys` |
| `ArrayDraft` | `get(i)`, `index(i) -> Option<Draft>`, `len`, `push`, `extend`, `insert(i, Json)`, `remove(i) -> Json`, `pop`, `splice(range, items) -> Vec<Json>`, `truncate`, `clear`, `set(i, Json)`, `reverse`, `sort_by(cmp)`, `retain(pred)` |
| `StrDraft` | `as_str()`, `push_str(&str)`, `remove_prefix(byte_len)`, `set(String)` |

Index methods panic out of bounds, as `Vec`'s do. `remove_prefix` panics when the length is
not on a character boundary, as `String::drain` does.

### What each edit records

Structural edits record their op directly; assignments are diffed against the value they
replace.

| Edit | Op |
| --- | --- |
| `ObjectDraft::insert` of a new key | `s` |
| `ObjectDraft::insert` over an existing key, `Draft::set`, `ArrayDraft::set`, `StrDraft::set` | `diff(old, new)` at that path |
| `Draft::update`, `update_as` | `diff(before, after)` at that path |
| `ObjectDraft::remove` | `d` |
| `push`, `insert` | insert-only `p` |
| `remove`, `pop`, `splice`, `truncate`, `clear`, `retain` | `p` (`retain` emits one `p` per removed run, from the end) |
| `reverse`, `sort_by` | `m` with the permutation |
| `StrDraft::push_str` | `a` |
| `StrDraft::remove_prefix` | `t` |
| Any value or string edit at the root (`s`, `d`, `a`, `t` have no root form) | `r` |

`update_as::<T>` deserializes the value into `T`, runs the closure, serializes it back and
diffs. It is the typed escape hatch for code that would rather mutate a struct than walk a
draft; its cost is a serde round trip of that subtree.

### Log coalescing

The log merges an op into the previous one when they are adjacent:

- `a` after `a` on the same path: one `a` with both texts. A stream of tokens appended in
  one change is one op.
- An insert-only `p` that continues the previous insert-only `p` on the same array: one
  `p`. Consecutive pushes are one op.
- An assignment or deletion at a path drops the ops directly before it whose paths are
  below that path, and those at it, except a deletion (an assignment does not repeat its
  effect on key order) and, before a deletion, a set (it may have created the key). Only
  the adjacent run: anything earlier may address elements whose indices moved since.
- `r` clears the log.

Because the log is replayed in order, every recorded op is correct for the value as it was
when it was recorded; coalescing only shortens the batch.

## `diff`

```rust
pub fn diff(old: &Json, new: &Json) -> Vec<Op>;
```

Used for assignments in drafts, `Tracker::prepare_diff` and `State::replace`.

- `Json::same(old, new)`: nothing. Unchanged shared subtrees cost O(1), so diffing a value
  derived from the previous revision costs time proportional to what changed.
- Two strings: `new` starts with `old`: `a` of the rest. Otherwise, when a suffix of `old` is
  a prefix of `new` (a sliding window): `t` of what precedes the suffix, then `a` of what
  follows it in `new`. Otherwise `s`. The overlap search is bounded (the last 64 KiB of
  `old`, a few candidate positions, as upstream's `overlap`); giving up yields an `s`, which
  is larger but never wrong.
- Two objects: `d` for removed keys, `s` for added keys, recursion for keys in both. A
  changed reserved key makes the whole object one `s`.
- Two arrays: trim the common prefix and suffix. If the remaining middles have the same
  length, recurse pairwise. Otherwise pair leading elements while both are containers of
  the same kind (an element edited in place, followed by inserts), then one `p` for the
  rest. An insert or removal between unchanged elements is therefore exactly one `p`.
- Different kinds: `s` (`r` at the root).

## Testing

- Invariant for every batch: `applied(base, ops) == value` and the batch survives a JSON
  round trip. A seeded random generator of drafts and edits checks it over many runs
  (upstream `state-fuzz.test.ts` and the tracker tests' convergence cases).
- Granularity tests for the cases consumers depend on: appended text is `a`, a sliding
  window is `t` + `a`, a push is one insert-only `p`, an equal assignment is empty.
- Wire tests: each tuple form round-trips, and batches captured from upstream apply with
  the expected result.

## Differences from upstream

- Op sequences differ. Upstream emits per node, shallowest first, objects as
  writes-then-deletes in first-write order, arrays as removal runs, a permutation, insert
  runs and overrides. delta emits edits in the order they were made. Upstream itself says
  batches are "exact but not canonical"; only the resulting value is contractual.
- Array removals are `p`, not `d`.
- `apply` is strict where upstream clamps (see above). Reserved segments are accepted by
  `apply`.
- No `undefined`: deleting is `remove`, not an assignment of `undefined`. Arrays cannot
  grow by assigning `length`.
- No draft validation of placed values: a `Json` is always strict JSON, so upstream's
  placement walk (≈19 ms per 100k objects in its benchmark) has nothing to check, and
  placing a value never clones it.
- Not ported: the wire `encoder` / `decoder` (path interning; it serves the host runtime's
  links) and `JsonRevisionValidator`.

`overlap(a, b, scan) -> usize` (the longest suffix of `a`, within the last `scan` bytes, that
is a prefix of `b`) stays public: durable uses it to estimate what a progress commit
writes. It counts bytes; upstream counts UTF-16 code units.
