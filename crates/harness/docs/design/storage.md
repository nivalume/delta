# Storage (`storage`)

Follows spec §4 (storage ownership), §10 (storage contract), §10.1 (failures, the storage
side), §11.1 (memory), §11.2 (SQLite) and §11.3 (JSONL). Corresponds to upstream
`src/types.ts` (`Storage`), `src/storage/scan.ts`, `src/storage/memory.ts`,
`src/storage/sqlite/{schema,migrations,storage}.ts`, `src/storage/jsonl/{storage,node}.ts`
and `src/testing/storage-conformance.ts`.

## Problem

The Session commits one batch per transaction: records of four tables plus document
creations, changes, copies and retirements. Storage makes the batch atomic, stamps it with a
sequence, keeps one id namespace, materializes documents from a base and its delta tail, and
answers keyed lookups and cursor scans in id order. Upstream defines this as an async
interface with a reference implementation (memory) and two persistent ones (SQLite, JSONL)
checked by one conformance suite. delta keeps the contract and implements all three, with
its own SQLite schema and its own JSONL line format (version 1 each, below; a pi database or
JSONL directory does not open in delta).

## `Storage`

```rust
use futures::future::BoxFuture;
pub type StorageFuture<'a, T> = BoxFuture<'a, Result<T, StorageError>>;   // `Fut` below

pub trait Storage: Send + Sync + 'static {
    /// Persist one batch atomically and return its sequence; once it returns, later reads observe it.
    fn commit(&self, writes: Vec<StorageWrite>, ctx: &Context) -> Fut<'_, Seq>;
    /// An id above every id minted or stored before, also across reopen.
    fn mint_id(&self) -> Fut<'_, RawId>;

    fn conversation(&self, id: ConversationId, ctx: &Context) -> Fut<'_, Option<ConversationRecord>>;
    fn scan_conversations(&self, query: ConversationQuery, limit: usize, cursor: Option<Cursor>, ctx: &Context) -> Fut<'_, Page<ConversationRecord>>;

    fn entry(&self, id: EntryId, ctx: &Context) -> Fut<'_, Option<StoredEntry>>;
    /// Only when the entry is visible through the conversation's ancestry (upstream's two-argument `entry`).
    fn visible_entry(&self, conversation: ConversationId, id: EntryId, ctx: &Context) -> Fut<'_, Option<StoredEntry>>;
    fn find_latest_head_marker(&self, conversation: ConversationId, at_or_before: Option<EntryId>, ctx: &Context) -> Fut<'_, Option<EntryRecord>>;
    fn scan_entries(&self, query: EntryQuery, limit: usize, cursor: Option<Cursor>, ctx: &Context) -> Fut<'_, Page<EntryRecord>>;

    fn task(&self, id: TaskId, ctx: &Context) -> Fut<'_, Option<TaskRecord>>;
    fn scan_tasks(&self, query: TaskQuery, limit: usize, cursor: Option<Cursor>, ctx: &Context) -> Fut<'_, Page<TaskRecord>>;

    fn submission(&self, id: SubmissionId, ctx: &Context) -> Fut<'_, Option<SubmissionRecord>>;
    fn scan_submissions(&self, query: SubmissionQuery, limit: usize, cursor: Option<Cursor>, ctx: &Context) -> Fut<'_, Page<SubmissionRecord>>;
    fn submission_by_request(&self, conversation: ConversationId, request_id: &str, ctx: &Context) -> Fut<'_, Option<SubmissionRecord>>;

    fn find_document(&self, address: &DocumentAddress, at: DocumentPoint, ctx: &Context) -> Fut<'_, Option<DocumentRecord>>;
    fn document(&self, id: DocumentId, at: DocumentPoint, ctx: &Context) -> Fut<'_, Option<StoredDocument>>;
    fn scan_documents(&self, query: DocumentQuery, limit: usize, cursor: Option<Cursor>, ctx: &Context) -> Fut<'_, Page<DocumentRecord>>;

    fn close(&self, ctx: &Context) -> Fut<'_, ()>;
}
```

- `BoxFuture`s keep the trait object-safe (`Arc<dyn Storage>`); the workspace has no
  `async-trait`. Backends write `async move { .. }.boxed()`.
- The trait trusts the Session for semantic validity (references, ancestry, transitions) and
  enforces atomicity, id ownership, immutable conversation and entry creation, and document
  record consistency. The Session serializes commits; a backend adds no commit mutex of its
  own, but each admitted batch is still atomic.
- Every value crossing the boundary is owned (`Json` is copy-on-write, records are values),
  so upstream's detach rules (clone on retain and on read) hold by construction.
- `ctx` carries cancellation for reads and feeds the Session's exemption rule (below); the
  built-in backends finish a call once it started and do not read `ctx`.

```rust
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("{0}")] Request(String),          // invalid read, no durable effect (upstream `StorageRequestError`)
    #[error("storage is closed")] Closed,
    #[error("ID space is exhausted")] IdSpaceExhausted,
    #[error("{0}")] Rejected(String),         // a refused batch, no effect
    #[error("{0}")] Corrupt(String),          // missing base, version change in a tail, op that does not apply, unreadable row
    #[error(transparent)] Sqlite(#[from] delta_chord::sqlite::SqliteError),
    #[error(transparent)] Json(#[from] serde_json::Error),   // a record that does not encode
    #[error("JSONL storage is poisoned and must be reopened: {0}")] Poisoned(String),
    #[error(transparent)] File(#[from] crate::env::FileError),   // JSONL: a file operation at open, or a path before a commit writes
}
impl StorageError { pub fn is_request(&self) -> bool; }
```

No `Io` variant: SQLite reports file errors through `SqliteError::Io`, JSONL through
`File` (its file system's `FileError`).

### Shared algorithms

The rules below are written once and the backends only map them onto their tables:

| Module | Contents |
| --- | --- |
| `storage/batch.rs` | `plan(writes, view) -> Batch`: id claims, request-id uniqueness, document command normalization, copy resolution, incarnation checks. A backend implements the read-only `CommitView` of its committed state, plans, and applies the `Batch` only on success, so a refused batch has no effect |
| `storage/ancestry.rs` | Visibility chains, `visible_entry`, `latest_head`, entry scans over an `EntryReader` (lineage, entry, per-conversation range read, latest head) |
| `storage/replay.rs` | Materialization of a base and its delta tail |
| `storage/scan.rs` | `ScanStart` (order, cursor, `limit + 1`) and pages |

## Commit rules

1. The batch sequence is `next_seq`; `next_seq` advances to `seq + 1` in the same
   transaction. Sequences strictly increase; gaps are allowed.
2. Id ownership: every create (`Conversation`, `Entry`, `DocumentCreate`, `DocumentCopy`)
   claims its id in the global namespace; an id already owned, or claimed twice in one batch,
   rejects (`"ID {id} already belongs to {table}"`, `"ID {id} is written more than once"`).
   `Task` and `Submission` writes replace: the id must be free or already of that kind
   (`"ID {id} is written as two record types"` within a batch). `next_id` becomes
   `max(next_id, highest written id + 1)`, saturating at the backend's maximum. A request id
   belongs to one submission per conversation, also within the batch
   (`"Request ID {r:?} already belongs to submission {id} in conversation {c}"`); SQLite's
   unique index is the backstop.
3. Document writes are normalized per incarnation id: at most one content command (`create`,
   `copy` or `change`) and at most one `retire`; content applies before retirement whatever
   the array order. A create or copy record must pass `DocumentCreate::check()`; a content
   version must be positive. Rejections: unknown id without a create; create of an existing id; any
   command on a retired incarnation; a delta on an incarnation with no revision, or whose
   newest revision has another version (`"Document {id} version transition requires a base"`);
   more than one current incarnation at an address after the batch
   (`"Document address already has a current incarnation"`), counted as current (0 or 1)
   − retired + created per address.
4. `DocumentCopy`: read the source with `document(source.id, source.at)` from committed
   pre-batch state. It must be an alive conversation document whose kind, key, history and
   fork equal the child record's, and no command in the batch may target it. Persist one
   child base `{version: source.version, value}`; later source changes, reclamation,
   retirement or reopen cannot affect the child. A failing copy rejects the batch
   (`"Document copy {id} was rejected"`).
5. Stamps: a create gets `created_at = seq`; a retire sets `retired_at = seq`; create plus
   retire gives both (empty lifetime); retire plus create at one address makes the new
   incarnation current at `seq`.
6. Reclamation for current-only incarnations (session, task, `latest` conversation): a base
   deletes every earlier revision; retirement deletes all revisions. Rewindable incarnations
   keep every revision.
7. Entries record `commit_seq = seq`, which `snapshot_as_of` needs.
8. Either the whole batch is visible to later reads or none of it.

## Reads

Keyed lookups return `None` for an unknown id. Invalid requests are `StorageError::Request`
with no durable effect: an unknown conversation in an entry read or scan, a malformed cursor,
a cursor from a scan in the other order, a historical read of a current-only document.

### Scans and cursors

- Conversation, task and submission scans run by id in `query.order`, default ascending;
  entry scans default descending; document scans are ascending by incarnation id and take no
  order.
- A page holds at most `limit` items; `next` is present only when more remain (the backend
  reads `limit + 1`).
- `limit` must be positive; `0` is a `Request` error.
- Built-in cursors are `{"after": <last id>, "order": "ascending" | "descending"}`, document
  scans included. A scan
  given a cursor continues in the cursor's order whether the query repeats or omits `order`,
  and rejects the other one (`"The cursor continues a {order} scan; the query asks for
  {requested}"`). A cursor without `order` continues in the scan's default order.
- A cursor is valid only for the same query on the same storage. Items committed after a
  page was read appear in later pages when their ids fall in the remaining range.
- Filters: conversations by `owner_conversation_id` and `owner_task_id` (indexed,
  conjunctive); tasks by conversation, kind, status, abort mark and background; submissions by
  conversation and status; documents by exact scope, point and optional kind.

### Entries and ancestry

Entry `e` is visible from conversation `c` when `e.conversation_id == c`, or `c` has a parent,
`e.id <= c.parent.at` and `e` is visible from `c.parent.conversation_id`. The cap tightens
along the chain: `upper = min(upper, parent.at)`.

- `visible_entry(c, id)`: the entry with its `commit_seq`, when visible from `c`.
- `find_latest_head_marker(c, at_or_before)`: the newest visible entry with a `head` whose id
  is at or below the cutoff; walks `c`, then the parent under the cap, and so on. The marker
  is returned; its `head` is the context's lower bound.
- `scan_entries` descending: `c`'s entries from `max_entry_id` down to `min_entry_id`, then
  the parent's up to its cap, and so on. Ascending: the segments are collected root-first and
  read oldest first, each ending at its cap. A cursor narrows the bound on the side the scan
  moves away from (`max = after − 1` descending, `min = after + 1` ascending).

### Documents

- `find_document(address, at)`: the incarnation at one exact logical address alive at `at`; a
  missing `key` means the singleton. `Current`: the unretired one; `At(seq)`:
  `created_at <= seq < retired_at`.
- `document(id, at)`: one specific incarnation, never a replacement at its address. Unknown
  id: `None`. `At(seq)` on a current-only incarnation:
  `Request("Document {id} does not retain historical content")`. Not alive at `at`: `None`.
  Otherwise select the newest base with revision `seq <= at` (`Current`: unbounded), apply the
  deltas after it in sequence order with `delta_chord::delta::apply`, and return
  `{record, version: base.version, value, deltas_since_base: tail.len()}`. A missing base, a
  delta of another version in the tail, or an op that fails to apply is `Corrupt`, not
  absence. The record and revision reads observe one committed state.
- `scan_documents({scope, at, kind?})`: incarnations alive in one exact scope at the point,
  ascending by id; its cursor is the same `{"after", "order": "ascending"}` shape, and a
  descending cursor is refused like any cursor from another scan.

### Id minting

`mint_id` returns `next_id` and increments it in memory; `commit` persists
`max(stored next_id, highest id in the batch + 1)`. Minted but uncommitted ids are lost on
reopen; nothing stored references them. `next_id` never wraps: the backend's maximum
(`u64::MAX` in memory, `i64::MAX` in SQLite) is never minted and serves as the exhausted
mark, so writing that id (or minting up to it) makes every later `mint_id` fail with
`IdSpaceExhausted`, which no real workload reaches. A written id past SQLite's range is a
`Rejected` batch (checked conversion).

## Failures (§10.1, storage side)

- A backend retries its own transient failures inside the method; what it returns is final
  for the Session: any error from `commit` or `mint_id`, and any read error but `Request`,
  fails the Session ([session.md](session.md)). A read cancelled through `ctx` fails only that
  read.
- A `commit` that returned an error may or may not have committed; the Session reopens and
  finds admitted submissions by `request_id`.
- `close` refuses new calls (`Closed`), waits for calls underway, then releases the backend;
  closing twice succeeds.

## Memory backend

`MemoryStorage::new()`. Reference semantics, in `storage/memory/{mod,tables}.rs`: tables are
`BTreeMap`s by id (ordered scans fall out), with the secondary indexes upstream keeps (conversations by owner, entry ids and head
markers per conversation, tasks and submissions by status, submissions by
`(conversation, request_id)`, documents by address and by scope). Revisions are
`Vec<(Seq, DocumentContent)>` per incarnation; a base on a current-only incarnation replaces
the vector. `next_id` starts at 2 (id 1 is the root conversation), `next_seq` at 1.
`commit` validates the whole batch, including copies, before it changes anything.

Upstream clones every retained write and every read result to simulate the ownership boundary
of a serializing backend; in delta the values are owned, so the same cases pass without
copying.

## SQLite backend

```rust
#[derive(Default)]
pub struct SqliteOptions { pub prefix: Option<String>, pub open: delta_chord::sqlite::OpenOptions }

impl SqliteStorage {
    pub async fn open(path: impl AsRef<Path>, options: SqliteOptions) -> Result<SqliteStorage, StorageError>;
    pub async fn open_database(db: Database, prefix: Option<String>) -> Result<SqliteStorage, StorageError>;
}
pub const CURRENT_SCHEMA_VERSION: u32 = 1;
/// Prefixed table names; `Schema::new(prefix)` validates the prefix.
pub struct Schema { .. }
pub fn migrations(schema: &Schema) -> Vec<Migration>;        // the DDL below, rendered for one prefix
```

`delta_chord::sqlite::Migration` owns its statements (`Vec<String>`), so the DDL is rendered
per prefix at open. Files: `storage/sqlite/{mod,schema,codec,read,write}.rs`; `read.rs`'s
`Reader` serves the reads and is the `CommitView` inside the commit transaction.

- Opens through `delta_chord::sqlite::Database` (WAL, `synchronous = NORMAL`, busy timeout),
  runs `apply_migrations(db, &migrations, "{prefix}schema")`, reads `next_id` from the
  metadata row (missing: `Corrupt("SQLite storage metadata is missing")`). A failed open
  closes the database and returns the first error. An invalid prefix fails before the file
  or its directory is created.
- `delta_chord::sqlite` re-exports `rusqlite`, so closures name `ToSql`, `FromSql` and
  `params!` through it; the harness has no direct `rusqlite` dependency.
- `prefix` passes `validate_schema_prefix`; every table and index name is `{prefix}{name}`.
  Data is not moved when the prefix changes.
- One Session commit is one `BEGIN IMMEDIATE` transaction: read metadata, check ids, check
  document actions, table writes in batch order, document actions, update metadata. The
  closure is synchronous on the database thread (chord's design); the Session line awaits it.
- Every read runs in one deferred transaction, so `document(id, at)`'s two queries and an
  ancestry walk observe one committed state.
- The `Database` sits behind a `tokio::sync::RwLock<Option<Database>>`: calls hold the read
  guard while they run; `close` takes the write guard (waiting for calls underway), takes the
  database and runs `Database::close` (checkpoint and close); later calls see `None` and fail
  with `Closed`.

### Schema (version 1; `{t.x}` and `{i.x}` are the prefixed names)

delta's own schema. Ids, sequences and times are `INTEGER`; kinds, keys and request ids are
plain `TEXT`; every record keeps its JSON in `record`, and the other columns are the
indexed or joined fields copied out of it, so reads decode `record` alone.

```sql
CREATE TABLE {t.metadata} (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    next_id INTEGER NOT NULL CHECK (next_id >= 2),
    next_seq INTEGER NOT NULL CHECK (next_seq >= 1)
) STRICT;
INSERT INTO {t.metadata} (singleton, next_id, next_seq) VALUES (1, 2, 1);

-- One global id namespace: every created id, with the table that owns it.
CREATE TABLE {t.record_ids} (
    id INTEGER PRIMARY KEY,
    record_type TEXT NOT NULL
        CHECK (record_type IN ('conversation', 'entry', 'task', 'submission', 'document'))
) STRICT;

CREATE TABLE {t.conversations} (
    id INTEGER PRIMARY KEY,
    parent_id INTEGER,                  -- parent.conversation_id; NULL without a parent
    parent_at INTEGER,                  -- parent.at
    owner_conversation_id INTEGER,
    owner_task_id INTEGER,
    record TEXT NOT NULL CHECK (json_valid(record)),
    CHECK ((parent_id IS NULL) = (parent_at IS NULL)),
    CHECK ((owner_conversation_id IS NULL) = (owner_task_id IS NULL))
) STRICT;
CREATE INDEX {i.conversations_by_owner_conversation} ON {t.conversations} (owner_conversation_id, id);
CREATE INDEX {i.conversations_by_owner_task} ON {t.conversations} (owner_task_id, id);

CREATE TABLE {t.entries} (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL,
    head INTEGER,                       -- NULL unless the entry is a head marker
    commit_seq INTEGER NOT NULL,
    record TEXT NOT NULL CHECK (json_valid(record))
) STRICT;
CREATE INDEX {i.entries_by_conversation} ON {t.entries} (conversation_id, id);
CREATE INDEX {i.entry_heads_by_conversation} ON {t.entries} (conversation_id, id) WHERE head IS NOT NULL;

CREATE TABLE {t.tasks} (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('pending', 'running', 'waiting', 'sleeping', 'completing', 'terminal')),
    abort_requested INTEGER NOT NULL CHECK (abort_requested IN (0, 1)),
    background INTEGER NOT NULL CHECK (background IN (0, 1)),
    record TEXT NOT NULL CHECK (json_valid(record))
) STRICT;
CREATE INDEX {i.tasks_by_status} ON {t.tasks} (status, id);
CREATE INDEX {i.tasks_by_conversation} ON {t.tasks} (conversation_id, id);
CREATE INDEX {i.tasks_by_kind} ON {t.tasks} (kind, id);
CREATE INDEX {i.tasks_by_abort_requested} ON {t.tasks} (abort_requested, id);
CREATE INDEX {i.tasks_by_background} ON {t.tasks} (background, id);

CREATE TABLE {t.submissions} (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL,
    request_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('queued', 'placed', 'done', 'unanswered')),
    record TEXT NOT NULL CHECK (json_valid(record))
) STRICT;
CREATE UNIQUE INDEX {i.submissions_by_request}
    ON {t.submissions} (conversation_id, request_id) WHERE request_id IS NOT NULL;
CREATE INDEX {i.submissions_by_conversation} ON {t.submissions} (conversation_id, id);
CREATE INDEX {i.submissions_by_status} ON {t.submissions} (status, id);

CREATE TABLE {t.documents} (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    key TEXT,                           -- family member key; NULL for a singleton
    scope_kind TEXT NOT NULL CHECK (scope_kind IN ('session', 'conversation', 'task')),
    owner_id INTEGER NOT NULL,          -- 0 for session scope
    created_at INTEGER NOT NULL,
    retired_at INTEGER,
    record TEXT NOT NULL CHECK (json_valid(record)),
    CHECK (retired_at IS NULL OR retired_at >= created_at)
) STRICT;
CREATE INDEX {i.documents_by_address}
    ON {t.documents} (kind, scope_kind, owner_id, key, created_at);
CREATE INDEX {i.documents_by_scope} ON {t.documents} (scope_kind, owner_id, id);
CREATE INDEX {i.documents_by_scope_kind} ON {t.documents} (scope_kind, owner_id, kind, id);

CREATE TABLE {t.document_revisions} (
    document_id INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('base', 'delta')),
    version INTEGER NOT NULL CHECK (version >= 1),
    content TEXT NOT NULL CHECK (json_valid(content)),
    PRIMARY KEY (document_id, seq)
) STRICT, WITHOUT ROWID;
CREATE INDEX {i.document_bases} ON {t.document_revisions} (document_id, seq) WHERE kind = 'base';
```

- Address lookups match the key with `key IS ?` (SQLite uses the index for `IS`), so a
  singleton (`NULL`) and a family member keyed `""` are different addresses without a
  separate `family` column. At most one current incarnation per address is enforced by
  commit rule 3, not by an index (`UNIQUE` treats `NULL`s as distinct).
- `document_bases` serves "newest base at or before `at`"; the delta tail is the primary
  key range after it. On an unanalyzed database SQLite may instead walk the primary key back
  from `at`, which reads the same tail; either way it is a search, never a scan
  (`tests/sqlite_storage.rs` checks the plans).
- The schema version lives in chord's `{prefix}schema` table (`singleton`, `version`); a
  database at a later version is refused with `NewerSchema`. Later versions append
  migrations to `migrations()`; version 1 is the first and has no predecessor.

### Column encoding

| Column | Content |
| --- | --- |
| `record` | the record's JSON (records.md shapes) |
| `kind`, `key`, `request_id` | plain text, bound as parameters; Rust strings are valid UTF-8, so nothing is escaped or encoded |
| `scope_kind`, `owner_id` | `session`/0, `conversation`/conversation id, `task`/task id |
| ids, `seq`, `commit_seq`, `created_at`, `retired_at`, `next_id`, `next_seq` | `INTEGER`; a `u64` is written as `i64` through a checked conversion (a value past `i64::MAX` is never minted, see "Id minting") |
| `content` | base: the value's JSON; delta: the op batch's JSON (chord tuples) |
| `head`, `parent_*`, `owner_*`, `retired_at`, `key`, `request_id` | `NULL` when absent |
| `status`, `record_type`, revision `kind` | plain text |

## JSONL backend

```rust
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JsonlOptions { pub fsync: bool }
pub const JSONL_FORMAT_VERSION: u32 = 1;

impl JsonlStorage {
    /// Opens or creates `dir` on `fs` and recovers it.
    pub async fn open(dir: &str, fs: Arc<dyn FileSystem>, options: JsonlOptions, ctx: &Context) -> Result<JsonlStorage, StorageError>;
    /// The same on this machine (`LocalEnv`); a relative `dir` is below the current directory.
    pub async fn open_local(dir: impl AsRef<Path>, options: JsonlOptions) -> Result<JsonlStorage, StorageError>;
    pub fn dir(&self) -> &str;
}
```

Files: `storage/jsonl/{mod,format,recover}.rs`. The backend writes through `env::FileSystem`
(`append_file`, `flush_file`, `truncate_file`, `rename_file`, `write_file`, `remove`,
`list_dir`, `read_binary_file`), so a remote environment can host it. It serves every read
from an inner `MemoryStorage` that recovery rebuilds; the memory backend lends three
crate-internal steps for it: `prepare(writes)` plans a batch against the committed tables
without changing them, `publish(prepared)` applies it, and `replay(writes, seq)` applies a
recovered commit at its original sequence.

### Layout and lines

| File | Lines |
| --- | --- |
| `main.jsonl` | one commit marker per commit |
| `doc-<id>.jsonl` | the bases and deltas of one document incarnation |
| `task-<id>.jsonl` | whole records of one task while it is live |
| `<sidecar>.reclaim` | a reclamation's temporary replacement; removed at open |

Every line is one JSON object and a `\n`. A marker is
`{"format":1,"type":"commit","seq":N,"writes":[...]}`; its writes use `StorageWrite`'s
adjacent tagging (`{"type": ..., "value": ...}`) over delta's record JSON:

| Write | Value |
| --- | --- |
| `conversation`, `entry`, `submission` | the record |
| `task` | a terminal task's record; a live one is a `task_sidecar` |
| `task_sidecar` | `{id, ordinal}` |
| `document_create` | `{record, ordinal}` (the `DocumentCreate` record) |
| `document_change` | `{id, ordinal}` |
| `document_retire` | `{id}` |

A sidecar line is `{"format":1,"type":"record","seq":N,"ordinal":k,"payload":P}` with `P`
either `{"type":"task","value":TaskRecord}` (live) or
`{"type":"document","id":D,"content":DocumentContent}`. The ordinal numbers the sidecar
records of one commit in batch order, so `(seq, ordinal)` names a record across files.
Copies are not written as such: `plan()` already resolved each `DocumentCopy` into a create
with the source's base, which is what the sidecar keeps.

### Commit

1. One `tokio::sync::Mutex` serializes commits and `close`; it also holds the set of tasks
   that have a sidecar. Closed or poisoned: fail.
2. `memory.prepare(writes)` plans the batch (a `Rejected` batch returns here), and the lines
   are encoded; a record that does not encode is `Json`. Nothing has been written yet, and
   the storage stays usable.
3. Each affected sidecar gets one append with its records, in the order of their first
   ordinal; with `fsync`, each is then flushed; then the marker is appended to
   `main.jsonl`. A failure at any of these steps poisons the storage: the write may or may
   not have reached the file, so every later call (reads, `mint_id` and commits) fails with
   `Poisoned` until the directory is reopened.
4. `memory.publish(prepared)`: reads observe the batch only after its marker is written.
5. Reclamation, best effort. With `fsync`, `main.jsonl` is flushed first (a failure skips
   reclamation, so an undurable marker never authorizes it). A current-only incarnation's
   sidecar is replaced by the new base alone (written to `.reclaim`, flushed with `fsync`,
   renamed over the sidecar); a retired current-only incarnation's sidecar and a task's
   sidecar once the task is terminal are removed. A failure is ignored: the committed state
   does not depend on it, and the next open redoes it. Rewindable incarnations are never
   reclaimed.

The file operations run on `Context::background()`: a commit that started finishes, as the
other backends do, since a cancelled append would be an uncertain write.

Without `fsync`, a process crash leaves a directory that recovers to a prefix of the
acknowledged commits; a power, host, kernel or file-system failure may lose acknowledged
ones. With it, an acknowledged commit survives those too.

### Recovery

`open` creates the directory and recovers it before it returns:

1. A final line without `\n` is torn: the file is truncated to the last line break, at the
   byte. Every complete line must be UTF-8 and parse (`Corrupt`, naming the file and line).
   Marker sequences must strictly increase.
2. `.reclaim` files are removed; every `doc-<id>.jsonl` and `task-<id>.jsonl` (canonical
   decimal ids) is read the same way, and its `(seq, ordinal)` keys must increase.
3. From the markers alone: the current-only incarnations, the retired ones among them, each
   one's newest base, and each task's final state (terminal in the marker, or live in a
   sidecar).
4. Each marker's writes are rebuilt with the sidecar records it names and replayed with
   `memory.replay(writes, seq)`; a replay that fails is `Corrupt`. A named record must exist
   (`Missing confirmed sidecar record`) unless reclamation may have removed it: it precedes
   the newest base of its current-only incarnation, the incarnation is retired and
   current-only, or the task is terminal later. Such a record is skipped even when present;
   a reclaimed creation is replayed with the placeholder base `{version: 1, value: {}}`,
   which a later base or the retirement supersedes.
5. Sidecar records no marker names belong to commits whose marker was never written: the
   tail they start is truncated. A named record after an unnamed one is `Corrupt`, as is a
   record named twice.
6. The reclamations the committed state allows are redone (step 5 of the commit), so an
   interrupted one finishes and a task that ended stays without a sidecar.

`next_seq` follows the last marker; `next_id` follows the highest id the replay wrote. The
sequence of a commit whose marker was lost is reused by the next commit.

### Close

`close` refuses later calls, waits for a commit underway (it takes the commit mutex) and
releases the memory tables; closing twice succeeds. The backend keeps no file open between
operations, so nothing else needs closing.

## Conformance suite

`tests/storage_conformance/` writes every case once as an `async fn(Fixture)` over
`Arc<dyn Storage>` (in `tables.rs`, `documents.rs` and `general.rs`; `main.rs` holds the
helpers) and runs it against memory, SQLite in memory, SQLite in a file under a unique temp
directory (removed afterwards), SQLite in a file under a table prefix, JSONL, and JSONL closed
and reopened after every commit (upstream's "across reopen" run, so every case also checks
what recovery rebuilds), as upstream's `createStorageConformance`:

| # | Upstream case | Rust test | Notes |
| --- | --- | --- | --- |
| 1 | reserves ID 1 for the immutable root conversation | `reserves_root_id` | first minted id is 2; id 1 twice rejects |
| 2 | commits mixed table writes atomically and rolls all of them back on failure | `mixed_writes_are_atomic` | |
| 3 | detaches retained writes and every returned record | `returned_records_are_owned` | mutating a result or a committed write changes nothing stored |
| 4 | detaches prototype-like JSON keys without changing object prototypes | `prototype_like_keys_round_trip` | `__proto__`, `constructor` as data keys round-trip; nothing else applies in Rust |
| 5 | indexes entries committed out of ID order | `entries_out_of_id_order` | |
| 6 | continues an entry cursor below its last item after a newer commit | `entry_cursor_after_newer_commit` | |
| 7 | paginates conversations by opaque cursor in ascending ID order | `conversation_pages` | |
| 8 | filters and pages conversations by durable owner edges | `conversation_owner_filters` | |
| 9 | scans deep fork history newest-first through every ancestor cap | `deep_fork_history` | |
| 10 | scans tables in either ID order and continues a cursor in its order | `scan_orders_and_cursors` | the other order with a cursor is `Request` |
| 11 | replaces complete task records and pages filtered task scans | `task_replacement_and_filters` | |
| 12 | stores owners and scans waiting and completing tasks by status | `task_owner_and_status_scans` | |
| 13 | stores and scans sleeping tasks by status | `sleeping_tasks` | `sleeping` is in the v1 status CHECK |
| 14 | indexes request IDs per conversation and replaces complete submission records | `request_ids_and_submission_replacement` | |
| 15 | stores passive write submissions without input-only lifecycle states | `write_submissions` | |
| 16 | reconstructs rewindable documents and preserves half-open incarnations | `rewindable_documents` | historical points; create+retire; retire+create |
| 17 | streams long document tails across root replacement deltas | `long_delta_tails` | `["r", v]` deltas stay deltas |
| 18 | copies stored document bases independently and rejects ambiguous sources | `copies_and_ambiguous_sources` | |
| 19 | uses bases for version transitions and rejects historical reads of current-only documents | `version_transitions_and_current_only_history` | |
| 20 | indexes logical addresses and exact-scope scans independently | `addresses_and_scope_scans` | |
| 21 | keeps document lifecycle failures atomic and gives create-plus-retire an empty lifetime | `document_lifecycle_and_empty_lifetimes` | |
| 22 | rolls back record tables and secondary indexes when a document command fails | `document_failures_roll_back_tables` | |
| 23 | keeps indexed string identities lossless | `indexed_strings_lossless` | kinds, keys and request ids with quotes, escapes, non-ASCII, LIKE metacharacters and `""` round-trip through plain text; Rust has no lone surrogates |
| 24 | keeps one global record ID namespace and rejects exhausted ID minting | `global_id_namespace` | exhaustion by writing the backend's maximum id (`i64::MAX` in SQLite) |
| 25 | rejects every operation after close | `rejects_after_close` | `Closed`; closing twice succeeds |

Added for delta: `copy_source_rules` (the `DocumentCopy` rejections of commit rule 4),
`singleton_and_empty_key` (a singleton and a member keyed `""` are separate addresses),
`duplicate_request_id_rejected` (request-id uniqueness, in a batch and across commits) and
`invalid_reads_are_requests` (zero limits, malformed and foreign cursors, unknown
conversations).

`tests/sqlite_storage.rs` covers what only SQLite has, after upstream's
`test/sqlite-storage.test.ts`: `reopen_continues_ids_and_seqs`,
`prefixed_schemas_are_independent` (two prefixes and a foreign table in one file),
`invalid_prefix_fails_before_creating_the_database`, `open_database_uses_an_open_connection`,
`missing_metadata_and_newer_schemas_are_refused`, `a_missing_base_is_corruption`,
`replays_root_replacements_and_refuses_corrupt_ops`,
`refused_commits_leave_rows_and_sequences_untouched`, `rewindable_points_survive_reopen`,
`current_only_revisions_are_reclaimed_by_bases_and_retirement`, `queries_use_their_indexes`,
`close_waits_and_refuses_later_calls`.

`tests/jsonl_storage.rs` covers what only JSONL has, after upstream's
`test/jsonl-storage.test.ts`. It wraps `LocalEnv` in a file system that logs the appends,
flushes, writes, renames and removes and fails one of them on demand, before it acts, after
it, or after writing half the bytes:

| Upstream case | Rust test |
| --- | --- |
| opens through the Node adapter | `a_new_directory_opens_empty_and_files_carry_delta_s_own_records` (`open_local`, a nested new directory, the marker shape, close twice) |
| publishes every commit before reclaiming current-only document and terminal-task sidecars | `every_commit_publishes_before_current_only_documents_and_finished_tasks_are_reclaimed` |
| orders multiple live-task replacements in one sidecar by commit ordinal | `live_task_replacements_in_one_commit_keep_their_order` |
| removes a complete-plus-torn unconfirmed multi-record sidecar append | `a_torn_multi_record_sidecar_append_is_removed` |
| serializes the complete candidate before I/O and leaves preparation failures usable | `a_refused_batch_writes_nothing_and_leaves_the_storage_usable` (a Rust record always encodes, so only the preparation failure applies) |
| poisons after {mode} failure at append {call} and recovers only confirmed state | `an_append_failure_poisons_and_reopening_recovers_only_confirmed_commits` |
| poisons after {mode} failure at flush {call} and never writes a marker | `a_sidecar_flush_failure_poisons_before_any_marker` |
| orders publication flushes exactly and flushes main only to authorize reclamation | `flushes_order_publication_and_flush_main_only_to_authorize_reclamation` |
| recovers a committed base across reclaim {operation} {mode} | `a_failed_base_reclamation_is_redone_by_the_next_open` |
| recovers document-retirement reclamation across remove {mode} | `a_failed_retirement_reclamation_is_redone_by_the_next_open` |
| defers reclamation after authorizing-main flush {mode} failure | `a_failed_authorizing_main_flush_defers_reclamation` (also the flush recovery makes) |
| keeps a committed base usable after reclaim-temp flush {mode} failure | `a_failed_replacement_flush_keeps_the_committed_base` |
| recovers terminal-task reclamation across remove {mode} | `a_failed_task_reclamation_is_redone_by_the_next_open` |
| appends later deltas to the replacement sidecar after a current-only base | `deltas_after_a_current_only_base_append_to_the_replacement` |
| never reclaims rewindable document history, including after a base and retirement | `rewindable_history_is_never_reclaimed` |
| reclaims retired task-, session-, and latest-conversation document sidecars | `retired_task_session_and_latest_documents_are_reclaimed` |
| truncates torn UTF-8 tails at exact byte offsets and reuses the unconfirmed sequence | `torn_utf8_tails_are_truncated_at_their_byte_and_their_sequence_reused` |
| removes complete unconfirmed sidecar tails without resurrecting terminal tasks | `unconfirmed_sidecar_tails_are_removed_without_resurrecting_finished_tasks` |
| fails open when confirmed sidecar data is missing | `missing_confirmed_data_fails_the_open` |
| rejects a confirmed record after an unconfirmed sidecar record | `a_confirmed_record_after_an_unconfirmed_one_fails_the_open` |
| rejects non-increasing main commit sequences | `sequences_that_do_not_increase_fail_the_open` |
| rejects structurally invalid confirmed document content | `invalid_document_content_fails_the_open` |
| rejects malformed complete main and sidecar lines | `malformed_complete_lines_fail_the_open` |

The unit tests in `jsonl/format.rs` check canonical sidecar names and the marker and record
round trip. `tests/resume.rs` runs its restart case on JSONL and reopens the directory at
each restart.

## Differences from upstream

- A `BoxFuture` trait instead of an async interface; `visible_entry` instead of an overloaded
  `entry`.
- `mint_id` returns `RawId`; the Session types it.
- One `StorageError` enum; `Request` replaces the `StorageRequestError` subclass.
- No detach-by-cloning: owned values make the ownership boundary structural. The conformance
  cases for it still run.
- `limit` is `usize`; a cursor is `Cursor(Json)`.
- SQLite goes through `delta_chord::sqlite::Database` with synchronous transaction closures;
  no portable `SqliteDatabase` facade and no Cloudflare driver.
- Memory tables are `BTreeMap`s rather than sorted id vectors with binary search.
- JSONL lines are delta's own: the marker writes and sidecar payloads carry delta's records
  in `StorageWrite`'s adjacent tagging and snake_case names, under
  `JSONL_FORMAT_VERSION = 1`. The layout, the publication protocol, `fsync`, reclamation and
  recovery follow upstream, including its corruption messages.
- JSONL errors are `StorageError::Poisoned` and `Corrupt` instead of
  `JsonlStoragePoisonedError` and `JsonlCorruptionError`; file failures at open are `File`.
- `JsonlStorage::open(dir, fs, options, ctx)` takes `ctx` last; `open_local` is upstream's
  `openNodeJsonlStorage`.
- A JSONL commit holds a mutex of its own, so `close` waits for one underway (SQLite's
  `RwLock` does the same), and its file operations ignore the caller's `ctx`; upstream
  passes the context to them.
- The SQLite schema is delta's own version 1: `INTEGER` ids and `next_id` (no decimal text,
  no 2^53 limit), plain-text kinds, keys and request ids (no JSON encoding), `key IS NULL`
  for singletons instead of a `family` flag and `""` key, conversation parent columns for
  ancestry walks, `sleeping` in the task status CHECK from the start, a unique request id
  per conversation. Upstream's schema and its v2 task rebuild are not carried.

## Decisions

- delta's own SQLite schema, version 1 as above; no compatibility with pi's databases.
- `delta_chord::sqlite` re-exports `rusqlite`.
- The optional table prefix stays.
- delta's own JSONL line format (above); no compatibility with pi's JSONL directories.
- The kernel's `storage` writes JSONL through `env::FileSystem`; the layering rule lets the
  kernel name `env`, which names nothing else in the crate.

- `scan_documents` cursors carry `"order": "ascending"` like every other scan; a descending
  cursor is refused rather than ignored (upstream ignores it).

## Open (minor)

- Async `mint_id` (kept for a later remote backend) versus a synchronous method, which all
  three built-in backends could offer.
