# JSON values (`json`)

Corresponds to upstream `src/json.ts` and to the ownership rules in `src/delta/README.md`.

## Problem

Every revision of a tracked document is a complete JSON value, and consecutive revisions
differ in a few paths. Copying the whole value per revision is too slow for documents that
change on every streamed token (a conversation transcript, a tool's output window).
Upstream shares unchanged containers between revisions and makes immutability a contract:
nothing is frozen, and mutating a shared container silently corrupts other revisions,
subscribers and replicas. Its README needs a table of who may mutate what.

delta shares containers through `Arc` and copies on write, so sharing is safe without any
contract.

## `Json`

```rust
pub enum Json {
    Null,
    Bool(bool),
    Number(Number),   // serde_json::Number: i64, u64 or finite f64
    String(Str),      // Arc<String>
    Array(Array),     // Arc<Vec<Json>>
    Object(Object),   // Arc<IndexMap<Key, Json>>, Key = Arc<str>
}
```

- `Str`, `Array` and `Object` are newtypes over the `Arc`. They `Deref` to `str`, `[Json]`
  and `IndexMap<Key, Json>`, and `make_mut()` returns the mutable inner value, copying it
  first when it is shared. The newtypes keep the representation private, so it can change
  later (for example an append-friendly string buffer) without an API break.
- `Clone` is O(1) for every variant.
- Mutation goes through `make_mut`: a container that is shared is copied (its spine of
  pointers, not its children) and the copy is changed. A value mutated at depth `d` copies
  at most `d` containers. Another holder of the old value never sees the change.
- Strict by construction: numbers are finite, strings are valid UTF-8, and there is no
  `undefined`. Upstream's `copyJson` / `isJsonValue` (validate and detach untrusted values)
  have nothing to do and are not ported.
- Objects keep insertion order (`IndexMap`, removal with `shift_remove`).

### Equality and identity

- `PartialEq` is deep equality: numbers compare by value (`1 == 1.0`), objects ignore key
  order. When both sides share one allocation the comparison returns at once, so comparing
  two revisions costs time proportional to what differs between them.
- `Json::same(a, b)` is the identity test that the tracker and `diff` use to skip unchanged
  subtrees: true when both are the same shared string or container, or equal scalars.

### Construction and conversion

| API | Notes |
| --- | --- |
| `json!(...)` | `serde_json::json!` syntax, converted to `Json` |
| `From<serde_json::Value>`, `From<Json> for serde_json::Value` | Boundary with `delta_ai` and other crates that use `serde_json::Value` |
| `Serialize`, `Deserialize` | Direct; deserializing builds the `Arc` tree without going through `serde_json::Value` |
| `to_json(&T)`, `from_json::<T>(&Json)` | Typed values. They go through `serde_json::Value` at first; a direct `Serializer` / `Deserializer` can replace that later without an API change |
| `From<bool>`, `From<&str>`, `From<String>`, `From<i64>`, `From<u64>`, `Json::from_f64(f64) -> Option<Json>` | `from_f64` returns `None` for non-finite numbers |

### Access

- `get(key)`, `get_index(i)`, `pointer(&[Seg])`, `as_str`, `as_array`, `as_object`, `as_f64`,
  `as_i64`, `as_bool`, `is_null`, `kind()`.
- `pointer_mut(&[Seg]) -> Option<&mut Json>` makes every container on the path unique first.

## Paths

```rust
pub enum Seg {
    Key(Key),       // serialized as a JSON string
    Index(usize),   // serialized as a JSON number
}
pub struct Path(Vec<Seg>);
```

- An object key is always a `Key`, even when it looks numeric (`["tasks", "123"]`); an array
  index is always an `Index`. This matches upstream's paths.
- `Path` implements `Display` as a JSON array for errors and logs, and `From` for arrays of
  `&str` / `usize` segments for tests.

## Costs

| Operation | Cost |
| --- | --- |
| Clone a revision | O(1) |
| Change one value at depth `d` | Copies `d` containers' pointer storage, once per change |
| Append to a string in a change | The first append copies the string (the base revision still holds it); later appends in the same change are amortized O(1) |
| Change an element of a flat array of `n` elements | Copies `n` pointers once per change (same as upstream) |
| Compare or diff two revisions | Proportional to the paths that differ |

## Differences from upstream

- No ownership rules. Upstream requires that roots passed to `track()`, `replace()` and
  `replicatedState()` are alias-free and never mutated again, that op payloads and
  published values are never mutated, and that each mutable replica gets its own detached
  copy of every batch. None of this applies: any `Json` may be mutated by whoever holds it.
- No `undefined`. Upstream distinguishes `undefined` from `null` in several places (a draft
  assignment of `undefined` deletes a key; a snapshot without a value). delta uses explicit
  removal and `Option`.
- Lone UTF-16 surrogates cannot be represented. `serde_json` refuses a `\ud800` escape
  without its pair; JS accepts it.
- Integer-like object keys keep insertion order. JS orders them first. Values are equal
  either way; only the serialized key order can differ.
- Floats that hold an integer keep their source representation: a number parsed as `1.0`
  serializes as `1.0`, where JS writes `1`. The parsed value is the same.
