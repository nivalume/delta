# Documents (`documents`)

Follows spec §3.1 (definitions), §3.2 (records and lifetimes), §3.5 (bases and checkpoints),
§3.6 (versions and migrations), §3.7 (fork policy) and the document items of §12.
Corresponds to upstream `src/documents.ts`. The transaction side (`tx.doc`, planning, copies,
observation) is [session.md](session.md); the storage side (incarnations, materialization) is
[storage.md](storage.md).

## Problem

Mutable state that must survive a crash (the live turn, the inbox, usage counters, an agent's
resolved selection) is a document: a JSON object whose changes are committed as chord op
batches with occasional complete bases. A definition says what a document is: its kind, its
version and how older values migrate, its initial value, which scope owns it and so when it
dies, whether a conversation document keeps history and what a fork copies, and when a change
is worth a complete base. Upstream expresses this as a definition object behind eight
`defineDoc` overloads whose return types encode scope and family; delta expresses it as a
trait whose associated types do the same, so a session document cannot be asked for with a
conversation id and `snapshot_as_of` cannot be called on a `latest` document.

## Definitions

```rust
pub trait DocDefinition: Send + Sync + 'static {
    /// Stable persisted kind; `delta.*` belongs to the crate's own definitions by convention.
    const KIND: &'static str;
    /// Positive. Versions belong to stored bases and deltas.
    const VERSION: u32;
    /// Owner, lifetime and (for conversations) history and fork:
    /// `SessionScope`, `TaskScope` or `ConversationScope<H, F>`.
    type Scope: DocScope;
    /// `()` for a singleton. A family's seed, given to `initial` on the first acquisition of a member.
    type Seed: Serialize + Send + 'static;
    const FAMILY: bool = false;

    /// The value of a new instance. Must be an object.
    fn initial(seed: &Self::Seed) -> Json;
    /// A complete current-version value from one stored at `from < VERSION`. Pure. Default: unsupported.
    fn migrate(value: Json, from: u32) -> Result<Json, MigrateError> { Err(MigrateError::Unsupported { from }) }
    /// Whether this change is stored as a complete base (below). Required; `|_, _, _| false` keeps only required bases.
    fn checkpoint_when(value: &Json, ops: &[Op], info: CheckpointInfo) -> bool;
}

pub struct CheckpointInfo {
    /// Deltas already stored after the newest base, excluding this change.
    pub deltas_since_base: u64,
}
#[derive(Debug, thiserror::Error)]
pub enum MigrateError {
    #[error("no migration from version {from}")] Unsupported { from: u32 },
    #[error(transparent)] Failed(Box<dyn Error + Send + Sync>),   // MigrateError::failed(e)
}

/// Sealed. The scope types are uninhabited markers.
pub trait DocScope: sealed::Sealed {
    type Owner: Copy + Debug + Eq + Hash + Send + Sync + 'static;   // (), ConversationId, TaskId
    const KIND: DocumentScopeKind;
    const POLICY: Option<ConversationPolicy>;                         // Some exactly for conversations
    fn scope(owner: Self::Owner) -> DocumentScope;
}
pub enum SessionScope {}
pub enum TaskScope {}
pub struct ConversationScope<H: HistoryKind, F: ForkKind<H>>(/* uninhabited */);

pub trait HistoryKind: sealed::Sealed { const HISTORY: History; }
pub enum Latest {}
pub enum Rewindable {}
/// A fork policy allowed with history `H`: `ForkAsOf` implements it only for `Rewindable`.
pub trait ForkKind<H>: sealed::Sealed { const FORK: Fork; }
pub enum ForkAsOf {}
pub enum ForkCurrent {}
pub enum ForkInitial {}

/// Implemented for every definition whose scope is a `ConversationScope`; bounds such as
/// `D: ConversationDoc<History = Rewindable>` select rewindable documents (`snapshot_as_of`).
pub trait ConversationDoc: DocDefinition { type History: HistoryKind; type Fork: ForkKind<Self::History>; }
impl<D, H, F> ConversationDoc for D where D: DocDefinition<Scope = ConversationScope<H, F>>, H: HistoryKind, F: ForkKind<H> { .. }

/// What a definition claims about every incarnation it touches; compared with the persisted record.
pub struct Semantics { pub kind: &'static str, pub version: u32, pub scope: DocumentScopeKind, pub policy: Option<ConversationPolicy>, pub family: bool }
impl Semantics { pub fn matches(&self, record: &DocumentRecord) -> bool; }
pub fn semantics<D: DocDefinition>() -> Semantics;
```

A definition is a unit struct; `KIND`, `VERSION` and the scope are read without an instance.
History and fork are parameters of the conversation scope, so the scope type alone carries
the persisted semantics: `semantics::<D>()` reads them from `D::Scope` without a second trait
or specialization, and a conversation document declares them in one place
(`type Scope = ConversationScope<Latest, ForkInitial>`).

A logical address is `(kind, scope owner, key?)`:

```rust
pub type OwnerOf<D> = <<D as DocDefinition>::Scope as DocScope>::Owner;
pub struct DocAddress<D: DocDefinition> { owner: OwnerOf<D>, key: Option<String>, _d: PhantomData<fn() -> D> }
impl<D: DocDefinition<Seed = ()>> DocAddress<D> { pub fn new(owner: OwnerOf<D>) -> Self }   // singleton
impl<D: DocDefinition> DocAddress<D> {
    pub fn member(owner: OwnerOf<D>, key: impl Into<String>) -> Self;   // family
    pub fn owner(&self) -> OwnerOf<D>;
    pub fn key(&self) -> Option<&str>;
    pub fn scope(&self) -> DocumentScope;
    /// The untyped address storage looks up; also the Session's cache key.
    pub fn to_address(&self) -> DocumentAddress;
}
```

`tx.doc(DocAddress::<Live>::new(conversation))`, `tx.doc_member(DocAddress::<NestedCall>::member(task, "c7"), seed)`,
`session.snapshot(DocAddress::<Live>::new(conversation), &ctx)` take these. The singleton
constructor exists only when `Seed = ()`; a family acquisition takes its seed as a separate
argument (session.md), and the seed is ignored whenever the member exists.

Validation. Everything about the definition itself is checked at compile time; what depends
on stored data is checked on access by functions the Session calls:

| Rule | Check |
| --- | --- |
| `VERSION >= 1` | an inline `const` assertion in `semantics::<D>()`: a compile error where the definition is used |
| `FORK == AsOf` requires `History = Rewindable` | the types: `ForkAsOf` implements `ForkKind<Rewindable>` only |
| `FAMILY` against addressing | inline `const` assertions: `DocAddress::new` refuses a family, `member` a singleton |
| `initial()` returns an object | `initial_value::<D>(seed)`, else `DocumentError::NotAnObject` |
| typed access against the persisted record | `check_record::<D>(record)`: kind, scope kind, history, fork, and a key exactly for a family; else `DocumentError::DefinitionMismatch`. A migration cannot reinterpret them |
| stored version against `VERSION` | `upgrade::<D>(id, stored_version, value)` (below) |

```rust
pub fn check_record<D: DocDefinition>(record: &DocumentRecord) -> Result<(), DocumentError>;
pub fn initial_value<D: DocDefinition>(seed: &D::Seed) -> Result<Json, DocumentError>;
/// Equal: unchanged; older: `migrate` (result must be an object); newer: refused.
pub fn upgrade<D: DocDefinition>(id: DocumentId, stored_version: u32, value: Json) -> Result<Upgraded, DocumentError>;
pub struct Upgraded { pub value: Json, pub migrated: bool }   // `migrated`: the first `tx.doc` stores a base

#[derive(Debug, thiserror::Error)]
pub enum DocumentError {
    #[error("Document {id} ({kind}) does not match the supplied definition semantics")] DefinitionMismatch { id: DocumentId, kind: String },
    #[error("Document {id} ({kind}) has newer version {stored} than {definition}")] VersionTooNew { id: DocumentId, kind: String, stored: u32, definition: u32 },
    #[error("Document {id} ({kind}) requires migration from version {stored}")] MigrationRequired { id: DocumentId, kind: String, stored: u32 },
    #[error("Document {kind} migration from version {from} failed")] Migrate { kind: String, from: u32, #[source] source: MigrateError },
    #[error("Document {kind} {origin} value is not an object")] NotAnObject { kind: String, origin: ValueSource },   // Initial | Migrated
}
```

The Session wraps `DocumentError` in `TxError::Document` (a rejected commit) or returns it
from a read; nothing is written.

Scope decides lifetime:

| Scope | Lifetime | History | Fork |
| --- | --- | --- | --- |
| session | current only; survives close and reopen; retired only by `retire_doc` | none | shared, never copied |
| conversation | current (`latest`) or rewindable; retired by `retire_doc` | declared | declared |
| task | current only; retired by `retire_doc` or atomically when its task becomes terminal | none | never copied |

`checkpoint_when` selects complete bases only; it changes no scope, lifetime, history or fork
semantics.

## Records and lifetimes (§3.2)

`DocumentRecord`, `DocumentCreate` and `DocumentAddress` are in [records.md](records.md).
What the definition layer relies on:

- An address has at most one current incarnation. Retire then recreate makes a new
  incarnation with a new id; a storage lookup at a historical point selects the incarnation
  whose `[created_at, retired_at)` contains it.
- The record persists scope, history and fork, so data outlives the code that defined it: a
  document whose definition is missing keeps its bytes and its semantics.
- Version lives on content. `StoredDocument::version` is the version of the selected base,
  which every delta after it shares (storage.md).

## Bases and checkpoints (§3.5)

Creation always stores a base. For a later change of a loaded incarnation the Session
(session.md, "Planning a document write") decides:

```text
stored version < VERSION            -> base (required; checkpoint_when not called)
prepared batch empty                -> nothing written, nothing published
checkpoint_when(value, ops, info)   -> true:  base with the complete prepared value
                                       false: delta with the prepared batch
```

- `info.deltas_since_base` counts the deltas stored after the incarnation's newest base,
  excluding this change: storage reports it at load, the Session increments it after each
  adopted delta and resets it after each base. The predicate never reads storage.
- It runs exactly once per change, after preparation and after every other validation. A
  panic in it rolls the commit back before storage admission.
- A root replacement op (`["r", value]`) in a batch is still a delta; only the predicate
  selects a base, and only a base authorizes reclamation.
- Storage never counts bytes, compares against `initial()` or invents a base. A definition
  that never checkpoints grows an unbounded replay tail: a definition bug, not a backend
  heuristic.
- A base permits reclamation of earlier records for session, task and `latest` conversation
  documents; for `rewindable` ones it bounds replay only.

Idioms: `|_, _, info| info.deltas_since_base >= 31`;
`|v, _, _| v.get("generation").is_none() && v.get("tools").is_none()`.

## Versions and migrations (§3.6)

| stored vs. `VERSION` | typed access (`upgrade::<D>`) |
| --- | --- |
| equal | use the value |
| older, `migrate` succeeds | migrated value; `tx.doc` stages a required base, read-only access migrates in memory only |
| older, `migrate` unsupported or failed | `DocumentError::MigrationRequired` / `DocumentError::Migrate` |
| newer | `DocumentError::VersionTooNew` |

- Migration is access-driven, through the definition supplied to that call: `tx.doc`,
  `snapshot`, `snapshot_as_of`, `document_state`. Open sweeps nothing; unaccessed documents
  keep their stored version and bytes.
- `migrate` always starts from the detached stored value, never from a cached revision.
- The first `tx.doc` after a migration writes a current-version base even when the migrated
  JSON equals the stored one; such an equal-value version base publishes nothing to observers
  of the new version.
- Rewindable history is never rewritten; historical reads migrate after replay.
- A cached document serves only definitions of the version its value was materialized for;
  another version drops it and reloads from storage (session.md, "Document cache").
- A fork copies the stored value and version untyped; the child migrates on its first typed
  access.

## Forks (§3.7)

A fork of conversation `A` at entry `E` creates child `B` with `parent = {A, E}`. For each
conversation document instance logically present at the selected point, by the policy
persisted in its record:

| `fork` | child instance | source |
| --- | --- | --- |
| `as_of` | copy of the value at `E`'s commit | `At(commit_seq(E))` on the ancestor conversation that owns `E` |
| `current` | copy of the committed value when the fork commit runs | `Current` on `A` |
| `initial` | none; `initial()` on first access in `B` | — |

- Copies are new incarnations with new ids and their own initial base at the source's stored
  version; no definition is needed (`DocumentCopy` write, storage.md).
- Task documents and tasks are never copied; session documents are shared and not rewindable.
- Document state at `E` is the final state of the commit that wrote `E`; different states
  need different commits.
- One fork may not select two sources for one child address (an `as_of` instance on the
  ancestor and a `current` one on `A`): `TxError::Fork`.
- The fork transaction may write neither a `fork: current` document of `A` nor a selected
  source incarnation: a copy has one unambiguous committed source.

Selection and staging run in the Session (session.md, "Forks").

## The agent layer's documents

The agent layer defines its documents with this trait, as an extension would; the kernel
has no special case for any of them. They are listed here so their semantics are in one
place; each is specified in its own note.

| Kind | Scope | History | Fork | `checkpoint_when` | Note |
| --- | --- | --- | --- | --- | --- |
| `delta.live` | conversation | latest | initial | nothing runs: no generation and no running tool slot | generation.md, tool.md |
| `delta.inbox` | conversation | latest | initial | `items` is empty | inbox.md |
| `delta.agent` | conversation | rewindable | as_of | always | harness.md |
| `delta.provider` | conversation | latest | initial | always | generation.md |
| `delta.usage` | conversation | latest | initial | always | generation.md |
| `delta.approvals` | conversation | latest | initial | `requests` is empty | approval.md |
| `delta.tool.nested-call` | task, family (seed `null`) | — | — | never | tool.md |
| `delta.codemode.run` | task | — | — | — | deferred with code mode |

All are version 1. `delta.live` is the one high-churn document: its rule keeps every base
small and bounds a delta chain to one generation or one round of tools.

## Example

```rust
pub struct Inbox;
impl DocDefinition for Inbox {
    const KIND: &'static str = "delta.inbox";
    const VERSION: u32 = 1;
    type Scope = ConversationScope<Latest, ForkInitial>;
    type Seed = ();
    fn initial(_: &()) -> Json { json!({ "items": [] }) }
    fn checkpoint_when(value: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        value.get("items").and_then(Json::as_array).is_some_and(|items| items.is_empty())
    }
}

// inside a commit closure
let mut inbox = tx.doc(DocAddress::<Inbox>::new(conversation)).await?;
inbox.draft().field("items").unwrap().into_array().unwrap().push(item);
```

A typed value is one `update_as` away:
`inbox.update_as::<InboxState, _>(|s| s.items.push(item))?` costs a serde round trip of
the document and yields a diffed batch.

## Differences from upstream

- A trait with associated types instead of a definition object, eight `defineDoc` overloads
  and twelve token aliases; scope, history and fork are types and the version and family
  rules are `const` assertions, so misuse is a compile error where upstream throws
  `TypeError` at run time.
- Definitions are stateless: `initial`, `migrate` and `checkpoint_when` take no `self`, and a
  definition is a type name, not a value to pass around.
- Values are `delta_chord::Json`, not a generic `T`. Typing is at the edges (`update_as`,
  `from_json`), which keeps drafts, ops and storage one code path. Upstream's typed drafts
  come free from `Proxy`; Rust has no equivalent short of a derive-based draft generator.
- No `copyJson` of `initial`/`migrate` results: a `Json` is strict and owned. A non-object
  result is still refused.
- `migrate` returns `Result` instead of throwing; `Unsupported` replaces "no `migrate`".

## Decisions

- One document is edited at a time through `Doc<'tx, D>`, which borrows the `Tx`
  ([session.md](session.md)).
- Observers get a read-only `DocumentState` over the incarnation's `delta_chord::State`.
- The crate's own kinds start with `delta.` and are version 1 (README).
- Scope, history and fork are types: `ConversationScope<H, F>` with marker types, chosen over
  a separate `ConversationDoc` impl with associated items (which `semantics::<D>()` could not
  read from a plain `D: DocDefinition` bound). `ConversationDoc` remains as a blanket helper
  trait for bounds.
- `DocAddress<D>` is the single argument shape of `tx.doc`, `snapshot`, `document_state` and
  `retire_doc`; its `to_address()` (a `DocumentAddress`) is the Session's cache key, so there
  is no separate `AddressId`.
- No `define_doc!` macro: inline `const` assertions give the compile-time checks.
- Documents stay untyped `Json` at the API; a typed convenience wrapper would be additive.
- Module layout: `documents/` with `definition.rs` (trait and markers), `address.rs` and
  `validate.rs` (semantics, access checks, `DocumentError`).
