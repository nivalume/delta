# Storage (`storage`)

Follows spec §4 (storage ownership), §10 (storage contract), §10.1 (failures, the storage
side), §11.1 (memory) and §11.2 (SQLite); §11.3 (JSONL) is deferred. Corresponds to upstream
`src/types.ts` (`Storage`), `src/storage/scan.ts`, `src/storage/memory.ts`,
`src/storage/sqlite/{schema,migrations,storage}.ts` and
`src/testing/storage-conformance.ts`.

## Problem

The Session commits one batch per transaction: records of four tables plus document
creations, changes, copies and retirements. Storage makes the batch atomic, stamps it with a
sequence, keeps one id namespace, materializes documents from a base and its delta tail, and
answers keyed lookups and cursor scans in id order. Upstream defines this as an async
interface with a reference implementation (memory) and two persistent ones (SQLite, JSONL)
checked by one conformance suite. delta keeps the contract and the SQLite schema, so a
database written by pi opens in delta, and implements memory and SQLite in phase 1.

## `Storage`

```rust
use futures::future::BoxFuture;
type Fut<'a, T> = BoxFuture<'a, Result<T, StorageError>>;

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
  built-in backends do not cancel a query already sent to SQLite.

## Commit rules

1. The batch sequence is `next_seq`; `next_seq` advances to `seq + 1` in the same
   transaction. Sequences strictly increase; gaps are allowed.
2. Id ownership: every create (`Conversation`, `Entry`, `DocumentCreate`, `DocumentCopy`)
   claims its id in the global namespace; an id already owned, or claimed twice in one batch,
   rejects (`Rejected("ID {id} already belongs to {table}")`). `Task` and `Submission` writes
   replace: the id must be free or already of that kind. `next_id` becomes
   `max(next_id, highest written id + 1)`.
3. Document writes are normalized per incarnation id: at most one content command (`create`,
   `copy` or `change`) and at most one `retire`; content applies before retirement whatever
   the array order. Rejections: unknown id without a create; create of an existing id; any
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
- Built-in cursors are `{"after": <last id>, "order": "ascending" | "descending"}`. A scan
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
  ascending by id; the cursor is `{"after": id}`.

### Id minting

`mint_id` returns `next_id` and increments it in memory; `commit` persists
`max(stored next_id, highest id in the batch + 1)`. Minted but uncommitted ids are lost on
reopen; nothing stored references them. Minting at `MAX_ID` fails with `IdSpaceExhausted`.

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

`MemoryStorage::new()`. Reference semantics: tables are `BTreeMap`s by id (ordered scans fall
out), with the secondary indexes upstream keeps (conversations by owner, entry ids and head
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
pub struct SqliteOptions { pub prefix: Option<String>, pub open: delta_chord::sqlite::OpenOptions }

impl SqliteStorage {
    pub async fn open(path: impl AsRef<Path>, options: SqliteOptions) -> Result<SqliteStorage, StorageError>;
    pub async fn open_database(db: Database, prefix: Option<String>) -> Result<SqliteStorage, StorageError>;
}
pub const CURRENT_SCHEMA_VERSION: u32 = 2;
pub fn migrations(schema: &Schema) -> Vec<Migration>;        // the DDL below, rendered for one prefix
```

- Opens through `delta_chord::sqlite::Database` (WAL, `synchronous = NORMAL`, busy timeout),
  runs `apply_migrations(db, &migrations, "{prefix}durable_schema")`, reads `next_id` from the
  metadata row. A failed open closes the database and returns the first error.
- `prefix` passes `validate_schema_prefix`; every table and index name is `{prefix}{name}`.
  Data is not moved when the prefix changes.
- One Session commit is one `BEGIN IMMEDIATE` transaction: read metadata, check ids, check
  document actions, table writes in batch order, document actions, update metadata. The
  closure is synchronous on the database thread (chord's design); the Session line awaits it.
- `document(id, at)` runs its two queries in one transaction so a concurrent commit cannot
  replace the base between them.
- `close` waits for multi-query reads underway, then `Database::close` (checkpoint and close).

### Schema (version 1, verbatim; `{t.x}` and `{i.x}` are the prefixed names)

```sql
CREATE TABLE {t.durable_metadata} (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    next_id TEXT NOT NULL,
    next_seq INTEGER NOT NULL
) STRICT;
INSERT INTO {t.durable_metadata} (singleton, next_id, next_seq) VALUES (1, '2', 1);
CREATE TABLE {t.record_ids} (
    id INTEGER PRIMARY KEY,
    record_type TEXT NOT NULL CHECK (record_type IN ('conversation', 'entry', 'task', 'submission', 'document'))
) STRICT;
CREATE TABLE {t.conversations} (
    id INTEGER PRIMARY KEY,
    owner_conversation_id INTEGER,
    owner_task_id INTEGER,
    record TEXT NOT NULL CHECK (json_valid(record))
) STRICT;
CREATE INDEX {i.conversations_by_owner_conversation} ON {t.conversations} (owner_conversation_id, id);
CREATE INDEX {i.conversations_by_owner_task} ON {t.conversations} (owner_task_id, id);
CREATE TABLE {t.entries} (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL,
    head INTEGER,
    commit_seq INTEGER NOT NULL,
    record TEXT NOT NULL CHECK (json_valid(record))
) STRICT;
CREATE INDEX {i.entries_by_conversation} ON {t.entries} (conversation_id, id DESC);
CREATE INDEX {i.entry_heads_by_conversation} ON {t.entries} (conversation_id, id DESC) WHERE head IS NOT NULL;
CREATE TABLE {t.tasks} (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'waiting', 'completing', 'terminal')),
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
CREATE INDEX {i.submissions_by_request} ON {t.submissions} (conversation_id, request_id);
CREATE INDEX {i.submissions_by_conversation} ON {t.submissions} (conversation_id, id);
CREATE INDEX {i.submissions_by_status} ON {t.submissions} (status, id);
CREATE TABLE {t.documents} (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    family INTEGER NOT NULL CHECK (family IN (0, 1)),
    key_value TEXT NOT NULL,
    scope_kind TEXT NOT NULL CHECK (scope_kind IN ('session', 'conversation', 'task')),
    owner_id INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    retired_at INTEGER,
    record TEXT NOT NULL CHECK (json_valid(record))
) STRICT;
CREATE INDEX {i.documents_by_address}
    ON {t.documents} (kind, scope_kind, owner_id, family, key_value, created_at DESC, retired_at);
CREATE INDEX {i.documents_by_scope} ON {t.documents} (scope_kind, owner_id, id);
CREATE INDEX {i.documents_by_scope_kind} ON {t.documents} (scope_kind, owner_id, kind, id);
CREATE TABLE {t.document_revisions} (
    document_id INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('base', 'delta')),
    version INTEGER NOT NULL,
    content TEXT NOT NULL CHECK (json_valid(content)),
    PRIMARY KEY (document_id, seq)
) STRICT;
CREATE INDEX {i.document_revisions_by_kind} ON {t.document_revisions} (document_id, kind, seq DESC);
```

Version 2 rebuilds `tasks` to admit `sleeping` (SQLite cannot alter a CHECK): create
`{t.tasks}_v2` with `status IN ('pending', 'running', 'waiting', 'sleeping', 'completing',
'terminal')` and the same other columns, `INSERT INTO {t.tasks}_v2 (...) SELECT ... FROM
{t.tasks}`, `DROP TABLE {t.tasks}`, `ALTER TABLE {t.tasks}_v2 RENAME TO {t.tasks}`, then
recreate the five task indexes. The schema metadata table is chord's `{prefix}durable_schema`
(`singleton`, `version`); a database at a later version is refused with `NewerSchema`.

### Column encoding

| Column | Content |
| --- | --- |
| `record` | the record's JSON (records.md shapes) |
| `kind` (tasks, documents), `key_value`, `request_id` | the string **JSON-encoded** (`"\"pi.live\""`), so lone surrogates survive bindings that replace them; a singleton's `key_value` is `"\"\""`. Query parameters for these columns are encoded the same way |
| `family` | 1 when `key` is present |
| `scope_kind`, `owner_id` | `session`/0, `conversation`/conversation id, `task`/task id |
| `next_id` | decimal text (upstream: `node:sqlite` refuses integers outside the safe range) |
| `content` | base: the value's JSON; delta: the op batch's JSON (chord tuples) |
| `head`, `owner_*`, `retired_at`, `request_id` | `NULL` when absent |
| `status`, `record_type`, revision `kind` | plain text |

Integers are `i64`; ids above `MAX_ID` are never written.

## JSONL

Deferred (README). The contract above is what it must meet; its sidecar layout, publication
protocol and recovery are spec §11.3.

## Conformance suite

`tests/storage_conformance.rs` runs every case against memory and against SQLite in a temp
directory, as upstream's `createStorageConformance`:

| # | Upstream case | Rust test | Notes |
| --- | --- | --- | --- |
| 1 | reserves ID 1 for the immutable root conversation | `reserves_root_id` | first minted id is 2; id 1 twice rejects |
| 2 | detaches retained writes and every returned record | `returned_records_are_owned` | mutating a result or a committed write changes nothing stored |
| 3 | detaches prototype-like JSON keys without changing object prototypes | `prototype_like_keys_round_trip` | `__proto__`, `constructor` as data keys round-trip; nothing else applies in Rust |
| 4 | indexes entries committed out of ID order | `entries_out_of_id_order` | |
| 5 | continues an entry cursor below its last item after a newer commit | `entry_cursor_after_newer_commit` | |
| 6 | paginates conversations by opaque cursor in ascending ID order | `conversation_pages` | |
| 7 | filters and pages conversations by durable owner edges | `conversation_owner_filters` | |
| 8 | scans deep fork history newest-first through every ancestor cap | `deep_fork_history` | |
| 9 | scans tables in either ID order and continues a cursor in its order | `scan_orders_and_cursors` | the other order with a cursor is `Request` |
| 10 | replaces complete task records and pages filtered task scans | `task_replacement_and_filters` | |
| 11 | stores owners and scans waiting and completing tasks by status | `task_owner_and_status_scans` | |
| 12 | stores and scans sleeping tasks by status | `sleeping_tasks` | schema v2 |
| 13 | stores passive write submissions without input-only lifecycle states | `write_submissions` | |
| 14 | reconstructs rewindable documents and preserves half-open incarnations | `rewindable_documents` | historical points; create+retire; retire+create |
| 15 | streams long document tails across root replacement deltas | `long_delta_tails` | `["r", v]` deltas stay deltas |
| 16 | indexes logical addresses and exact-scope scans independently | `addresses_and_scope_scans` | |
| 17 | keeps indexed string identities lossless | `indexed_strings_lossless` | kinds, keys and request ids with quotes and non-ASCII; Rust has no lone surrogates |
| 18 | keeps one global record ID namespace and rejects exhausted ID minting | `global_id_namespace` | `next_id` at `MAX_ID` |
| 19 | rejects every operation after close | `rejects_after_close` | `Closed` |

Added for delta: `reopen_continues_ids_and_seqs` (SQLite file closed and reopened),
`pi_database_opens` (a fixture written by pi at schema v2, with and without a prefix) and
`copy_source_rules` (the `DocumentCopy` rejections of commit rule 4).

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
- JSONL is not implemented.

## Pending confirmation

- Async `mint_id` (kept for a later remote or JSONL backend) versus a synchronous method,
  which both phase-1 backends could offer.
- `rusqlite` must be reachable from the harness crate for `ToSql`/`FromSql` in closures: add
  it to the harness dependencies, or have `delta_chord::sqlite` re-export it.
- Whether `scan_documents` ignores an `order` field in its cursor, as upstream does, or
  rejects it.
- Keeping upstream's JSON encoding of the indexed string columns unconditionally; it is
  required for reading pi's databases.
