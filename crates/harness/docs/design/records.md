# Records (`records`)

Follows spec §1 (terms and invariants), §2 (core records), §2.1 (the record side of entries;
context derivation is [context.md](context.md)'s), the task records of §5.1, the document
records of §3.2 and the storage types of §10. Corresponds to upstream `src/types.ts`,
`src/ids.ts` and `src/errors.ts`.

## Problem

Five record kinds make up everything the harness persists: conversations, entries, tasks,
submissions and document incarnations. Upstream brands their numeric ids at compile time
only (`Id<"task">` is a `number` at run time), writes the records as TypeScript unions whose
`never`-typed fields say which fields a status may carry, and stores them as JSON. delta
keeps the record model and represents it as Rust types that carry the same invariants:
newtype ids, tagged enums for the status unions, `Option` where upstream has `undefined`.
The JSON is delta's own (serde-native, snake_case, `delta.*` kinds); a storage written by pi
does not open in delta, and nothing here is shaped for it.

## Ids and sequences

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ConversationId(pub u64);
pub struct EntryId(pub u64);        // same derives
pub struct TaskId(pub u64);
pub struct SubmissionId(pub u64);
pub struct DocumentId(pub u64);
/// Commit sequence: strictly increases between commits; gaps are permitted.
pub struct Seq(pub u64);

pub const ROOT_CONVERSATION_ID: ConversationId = ConversationId(1);

/// A freshly minted id before it has a kind. `Storage::mint_id` returns one; the Session types it.
pub struct RawId(pub u64);
/// The five id newtypes. Sealed.
pub trait RecordId:
    Copy + Ord + Hash + Debug + Display + Serialize + DeserializeOwned + From<RawId> + Into<u64> + Send + Sync + sealed::Sealed
{
    fn from_raw(raw: u64) -> Self;   // provided
    fn get(self) -> u64;             // provided
}
```

- Ids and sequences are `u64` and serialize as JSON numbers; there is no cap below the
  type's range (and SQLite's `INTEGER`, storage.md). `Seq` is a separate type, so an entity
  id cannot be passed as a document point.
- One global numeric namespace for all five kinds. Ids never repeat, also across reopen, and
  scans order by id, which is creation order ([storage.md](storage.md)).
- Upstream's `TaskId<R>` carries an erased result type. delta's `TaskId` is untyped; a
  result-typed `TaskRef<R>` (`TaskId` plus `PhantomData<R>`, in `records::ids`) is what
  [scheduler.md](scheduler.md)'s typed waits take.

## Conversations

```rust
pub struct ConversationRecord {
    pub id: ConversationId,
    pub parent: Option<ConversationParent>,   // history: inherited entries and historical documents
    pub owner: Option<ConversationOwner>,     // task attribution, abort and idle traversal; not access control
}
pub struct ConversationParent { pub conversation_id: ConversationId, pub at: EntryId }
pub struct ConversationOwner { pub conversation_id: ConversationId, pub task_id: TaskId }
/// Creation input. The Session derives the persisted owner pair from the task; callers never build it.
pub enum ConversationOwnership { Ownerless, Task(TaskId) }
```

JSON: `{"id":1}`; `{"id":7,"parent":{"conversation_id":1,"at":42},"owner":{"conversation_id":1,"task_id":40}}`.

- The root conversation is id 1, ownerless, without parent; storage reserves the id.
- `parent.at` is an entry visible in the parent; the child sees parent entries through it.
- `owner` stays recorded after the owning task is terminal.

## Entries

```rust
pub struct EntryRecord {
    pub id: EntryId,
    pub conversation_id: ConversationId,
    pub kind: String,                           // `delta.*` kinds belong to built-ins by convention
    pub model: Option<Vec<delta_ai::Message>>,  // absent: contributes no model messages
    pub data: Option<Json>,
    pub head: Option<EntryId>,                  // head marker: context starts at `head`
    pub edits: Option<Vec<ContextEdit>>,
    pub by_task_id: Option<TaskId>,             // stamped by a task runtime commit
}
pub struct StoredEntry { pub entry: EntryRecord, pub commit_seq: Seq }
pub struct ContextEdit { pub target: EntryId, pub action: EditAction }
pub enum EditAction { Omit, Replace(Vec<delta_ai::Message>) }
// {"target":5,"action":"omit"} | {"target":5,"action":"replace","messages":[...]}

pub struct EntryDraft {
    pub kind: String,
    pub model: Option<Vec<delta_ai::Message>>,
    pub data: Option<Json>,
    pub head: Option<Head>,
    pub edits: Option<Vec<ContextEdit>>,
}
pub enum Head { At(EntryId), SelfRef }        // upstream `head: EntryId | "self"`: the entry's own id

/// A typed entry kind: `kind` plus the type of `data` (upstream `Entry<D>`).
pub struct EntryKind<D> { pub kind: &'static str, _data: PhantomData<fn() -> D> }
impl<D> EntryKind<D> { pub const fn new(kind: &'static str) -> Self; }
impl<D: DeserializeOwned> EntryKind<D> {
    /// `None` when the record is of another kind; missing `data` decodes from `null`.
    pub fn decode(&self, record: EntryRecord) -> Result<Option<TypedEntry<D>>, RecordError>;
}
impl<D: Serialize> EntryKind<D> { pub fn encode(&self, draft: TypedEntryDraft<D>) -> Result<EntryDraft, RecordError>; }
pub struct TypedEntry<D> { pub record: EntryRecord, pub data: D }
pub struct TypedEntryDraft<D> { pub data: D, pub model, pub head, pub edits }   // EntryDraft without `kind`
```

The Session's `typed_entry` and `append_typed_entry` are `decode` and `encode` around the
untyped calls, so the conversion lives with the records.

- Entries are immutable; ids are Session-global and ordered.
- `model` holds `delta_ai::Message`, whose JSON keeps upstream's (pi-ai's) shape as every
  `delta_ai` message does, including the `SystemMessage { content: "", sections,
  toolsAdded, toolsRemoved }` written for `delta.system` entries. It is the one camelCase
  island inside a record.
- `head` and `edits` change later context, never storage; the derivation rules (§2.1, steps
  1–10) are context.md's.
- `EditAction` serializes flat beside `target` (`action` tag and `messages` payload in one
  object): `ContextEdit` converts through a private flat wire struct
  (`#[serde(try_from, into)]`), which refuses `omit` with messages and `replace` without.

## Submissions

```rust
pub struct SubmissionRecord {
    pub id: SubmissionId,
    pub conversation_id: ConversationId,
    pub request_id: Option<String>,             // host deduplication key, unique per conversation
    #[serde(rename = "type")]
    pub kind: SubmissionType,
    #[serde(flatten)]
    pub state: SubmissionState,
}
pub enum SubmissionType { Input, Write }
#[serde(tag = "status", rename_all = "lowercase")]
pub enum SubmissionState {
    Queued,
    Placed { entry: EntryId },                                                   // input only
    Done { entry: EntryId, answer: Option<EntryId> },                            // `answer`: input only
    Unanswered { entry: Option<EntryId>, reason: String, detail: Option<Json> }, // `entry`: input only
}
pub enum SubmissionStatus { Queued, Placed, Done, Unanswered }                   // queries, SQLite column
pub enum SubmissionSettlement { Done { answer: EntryId }, Unanswered { reason: String, detail: Option<Json> } }
pub struct SubmissionCreate { /* SubmissionRecord without `id` */ }
```

| type | legal states |
| --- | --- |
| `input` | `queued` → `placed{entry}` → `done{entry, answer}` or `unanswered{entry?, reason, detail?}` |
| `write` | `queued` → `done{entry}` or `unanswered{reason, detail?}` |

`SubmissionRecord::check()` refuses the other combinations (a `placed` write, a write with
`answer`, an unanswered write with `entry`, a `done` input without `answer`);
`Tx::create_submission` and settlement call it, so storage never sees them.
`SubmissionCreate::into_record(id)` adds the id. Settlement rules (only a placed input becomes `done`; a settled
record stays) are [session.md](session.md)'s.

## Tasks

```rust
pub struct TaskRecord {
    pub id: TaskId,
    pub conversation_id: ConversationId,        // immutable
    pub kind: String,                           // task definition name
    pub version: u32,
    pub input: Json,
    pub owner: Option<TaskId>,                  // owning task; absent for a conversation-owned task
    pub background: bool,
    pub abort_requested: bool,
    pub abort_reason: Option<AbortReason>,      // `Restart`: the mark comes from abandonment after a restart
    #[serde(skip_serializing_if = "is_false")]
    pub abandon_on_restart: bool,               // JSON: present only as `true`
    pub started_at: Option<i64>,                // ms wall clock; first change to `running`
    pub ended_at: Option<i64>,                  // change to `terminal`
    pub restarts: Option<u32>,
    pub state: TaskState,
    pub memos: Option<IndexMap<String, Json>>,  // live states only
    pub trace: Option<Json>,                    // telemetry carriers, carried unread (telemetry is later)
    pub links: Option<Vec<Json>>,
}
#[serde(tag = "status", rename_all = "lowercase")]
pub enum TaskState {
    Pending { checkpoint: Json },
    Running { checkpoint: Json },
    Waiting { checkpoint: Json, on: Vec<TaskId>, policy: JoinPolicy },
    Sleeping { checkpoint: Json, until: i64 },
    Completing { outcome: TaskOutcome },
    Terminal { outcome: TaskOutcome },
}
#[serde(tag = "status", rename_all = "lowercase")]
pub enum TaskOutcome {
    Completed { result: Json },
    Failed { error: TaskOutcomeError, result: Option<Json> },
    Aborted { reason: Option<String>, result: Option<Json> },
    Orphaned { reason: String },
    Faulted { error: TaskOutcomeError },
}
pub struct TaskOutcomeError { pub message: String, pub detail: Option<Json> }
pub enum JoinPolicy { FailFast, AllSettled }          // "fail_fast" | "all_settled"
pub enum AbortReason { Restart }
pub enum TaskStatus { Pending, Running, Waiting, Sleeping, Completing, Terminal }

impl TaskState {
    pub fn status(&self) -> TaskStatus;
    pub fn checkpoint(&self) -> Option<&Json>;    // live states
    pub fn outcome(&self) -> Option<&TaskOutcome>; // completing, terminal
}
```

- A task record is replaced whole on every transition; terminal records stay as small rows.
- `memos` with a `completing` or `terminal` state is invalid; `TaskRecord::check()` refuses
  it, as it refuses a `Completed` outcome without `result` (the enum already does).
- `input`, `checkpoint` and `result` are `Json` here; built-in tasks convert them to their
  serde-tagged enums at the definition boundary (scheduler.md), user tasks work on `Json`.
- Times are `i64` Unix epoch milliseconds; the Session stamps them.

## Documents

```rust
pub struct DocumentRecord {
    pub id: DocumentId,                         // one incarnation; never reused
    pub kind: String,
    pub key: Option<String>,                    // family member; absent for a singleton
    pub scope: DocumentScope,
    #[serde(flatten)]
    pub policy: Option<ConversationPolicy>,     // present iff `scope` is a conversation
    pub created_at: Seq,                        // stamped by storage
    pub retired_at: Option<Seq>,
}
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DocumentScope { Session, Conversation { conversation_id: ConversationId }, Task { task_id: TaskId } }
pub enum DocumentScopeKind { Session, Conversation, Task }   // DocumentScope::kind(); owner_id() is 0 for session
pub struct ConversationPolicy { pub history: History, pub fork: Fork }
pub enum History { Latest, Rewindable }
pub enum Fork { AsOf, Current, Initial }       // "as_of" | "current" | "initial"
pub struct DocumentCreate { pub id, pub kind, pub key, pub scope, pub policy }   // no lifetime fields
pub struct DocumentAddress { pub kind: String, pub scope: DocumentScope, pub key: Option<String> }
pub enum DocumentPoint { Current, At(Seq) }
pub enum DocumentContent { Base { version: u32, value: Json }, Delta { version: u32, ops: Vec<Op> } }
pub struct BaseContent { pub version: u32, pub value: Json }
pub struct StoredDocument { pub record: DocumentRecord, pub version: u32, pub value: Json, pub deltas_since_base: u64 }
pub struct DocumentCopySource { pub id: DocumentId, pub at: DocumentPoint }
```

JSON: `{"id":12,"kind":"delta.live","scope":{"kind":"conversation","conversation_id":1},"history":"latest","fork":"initial","created_at":3}`;
a family member adds `"key":"call-7"`; retirement adds `"retired_at":9`.

- A singleton is identified by kind and scope, a family member by kind, scope and key: the
  address is the logical identity, the id one incarnation of it.
- Lifetime is the half-open interval `created_at <= at < retired_at`, unbounded while current;
  create and retire in one commit gives an empty lifetime.
- `policy` is present iff `scope` is `Conversation`; `Fork::AsOf` requires
  `History::Rewindable`. `DocumentRecord::check()` and `DocumentCreate::check()` enforce
  both; storage runs the latter on every create and copy.
- `DocumentRecord::is_alive_at(point)`, `keeps_history()` (rewindable conversation documents
  only) and `address()` are the record-level predicates both backends share.
- Definition versions live on bases and deltas, not on the record: one incarnation may hold
  records written by several definition versions.

## Storage writes, queries and pages

```rust
pub enum StorageWrite {
    Conversation(ConversationRecord),
    Entry(EntryRecord),
    Task(TaskRecord),
    Submission(SubmissionRecord),
    DocumentCreate { record: DocumentCreate, content: BaseContent },
    DocumentCopy { record: DocumentCreate, source: DocumentCopySource },
    DocumentChange { id: DocumentId, content: DocumentContent },
    DocumentRetire { id: DocumentId },
}
pub enum ScanOrder { Ascending, Descending }
pub struct Page<T> { pub items: Vec<T>, pub next: Option<Cursor> }
/// Backend-owned continuation, round-tripped unread. Built-in backends use `{"after": <id>, "order": "ascending"}`.
pub struct Cursor(pub Json);

pub struct ConversationQuery { pub owner_conversation_id: Option<ConversationId>, pub owner_task_id: Option<TaskId>, pub order: Option<ScanOrder> }        // default ascending
pub struct EntryQuery { pub conversation_id: ConversationId, pub min_entry_id: Option<EntryId>, pub max_entry_id: Option<EntryId>, pub order: Option<ScanOrder> } // inclusive; default descending
pub struct TaskQuery { pub conversation_id: Option<ConversationId>, pub kind: Option<String>, pub status: Option<TaskStatus>, pub abort_requested: Option<bool>, pub background: Option<bool>, pub order: Option<ScanOrder> }
pub struct SubmissionQuery { pub conversation_id: Option<ConversationId>, pub status: Option<SubmissionStatus>, pub order: Option<ScanOrder> }
pub struct DocumentQuery { pub scope: DocumentScope, pub at: DocumentPoint, pub kind: Option<String> }  // ascending incarnation id only
pub struct StoredEntry { pub entry: EntryRecord, pub commit_seq: Seq }
```

Memory and SQLite do not persist `StorageWrite` itself; the JSONL backend's commit markers
use its serde form for the writes they carry ([storage.md](storage.md#jsonl-backend)). It is
adjacently tagged, `{"type":"document_create","value":{...}}`, because an internally tagged
`submission` would collide with the record's own `type` field. `limit` must be positive (`0` is a `Request` error).

## Errors and reports

Each module owns its error enum; errors convert only at layer boundaries.

```rust
/// A record that breaks a rule its type cannot express, or typed entry data that does not convert.
#[derive(Debug, thiserror::Error)]
pub enum RecordError {
    #[error("Submission {id}: {problem}")] Submission { id: SubmissionId, problem: String },
    #[error("Task {id}: {problem}")] Task { id: TaskId, problem: String },
    #[error("Document {id}: {problem}")] Document { id: DocumentId, problem: String },
    #[error("Entry data of kind {kind} does not convert")] EntryData { kind: String, #[source] source: serde_json::Error },
}
```

`StorageError` is the `storage` module's ([storage.md](storage.md)); `DocumentError` the
`documents` module's ([documents.md](documents.md)). The Session's errors and the report
types below belong to the session module (`session::{error, report}`) and are listed here for
the record model they name (`ConversationBusy` arrives with the inbox):

```rust
/// The Session failed; `cause` is the first error. Reopen it.
#[derive(Clone, Debug, thiserror::Error)]
#[error("Session failed after a storage error; close and reopen it")]
pub struct SessionFailed { pub cause: Arc<dyn Error + Send + Sync> }

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error(transparent)] Failed(SessionFailed),
    #[error("Session is closed")] Closed,
    #[error(transparent)] Cancelled(delta_chord::context::Cancelled),
    #[error(transparent)] Storage(StorageError),   // only `Request` reaches a caller; everything else becomes `Failed`
    #[error(transparent)] Tx(TxError),             // a rejected commit; rolled back
}

#[derive(Debug, thiserror::Error)]
pub enum TxError {
    #[error("Tx::{method}() cannot read tables after the first table write")] ReadAfterWrite { method: &'static str },
    #[error("Conversation {0} does not exist")] UnknownConversation(ConversationId),
    #[error("Task {0} does not exist")] UnknownTask(TaskId),
    #[error("Submission {0} does not exist")] UnknownSubmission(SubmissionId),
    #[error("{what} {task} is {status}")] OwnerUnavailable { what: &'static str, task: TaskId, status: &'static str },
    #[error("Task {0} is terminal")] TerminalTask(TaskId),
    #[error("Task {0} cannot change conversations")] TaskConversationChange(TaskId),
    /// Definition mismatch, version too new, migration required or failed, non-object value.
    #[error(transparent)] Document(DocumentError),
    #[error(transparent)] Record(RecordError),     // a `check()` refusal
    #[error("{0}")] Submission(String),            // illegal settlement or placement
    #[error("{0}")] Fork(String),                  // invisible fork entry, duplicate copies, writes to fork sources
    #[error("{0}")] InvalidTask(String),           // a background child, a child elsewhere, no conversation, a second terminal candidate
    #[error(transparent)] Storage(StorageError),   // a `Request` error seen by the closure; anything else failed the Session
    #[error(transparent)] Cancelled(Cancelled),    // the transaction's context was cancelled during a read
    #[error(transparent)] Failed(SessionFailed),   // the Session failed while the transaction ran
    #[error("panicked: {0}")] Panic(String),       // the closure or a definition function panicked; rolled back
    #[error(transparent)] Callback(Box<dyn Error + Send + Sync>),   // the commit closure's own error
}

#[derive(Debug, thiserror::Error)]
#[error("Conversation {0} is busy")]
pub struct ConversationBusy(pub ConversationId);

/// `onReport`'s `info`. Every task variant carries the task's id, kind and conversation.
pub enum ReportInfo {
    SessionFailed,
    TaskFaulted { task: ReportTask, reason: FaultReason },         // Threw | RestartLimit
    TaskParked { task: ReportTask },
    MigrationFailed { task: ReportTask },
    TaskOutdated { task: ReportTask, reason: OutdatedReason },     // MissingTask | IncompatibleTask
    TaskError { task: ReportTask },
    ExtensionError { task: Option<ReportTask>, conversation: Option<ConversationId>, extension: Option<String> },
    HostError { callback: HostCallback, task: Option<ReportTask> },  // Now | Settings | Registry | Env | Models | OnWake
    ListenerError,
}
pub struct ReportTask { pub id: TaskId, pub kind: String, pub conversation_id: ConversationId, pub extension: Option<String> }
pub type Reporter = Arc<dyn Fn(&(dyn Error + Send + Sync), &ReportInfo) + Send + Sync>;
```

Messages keep upstream's wording where its tests match on them (`cannot read tables after
the first table write`, `does not retain historical content`, `has newer version`,
`requires migration`, `already belongs to`).

## JSON encoding

Records are serde-native; nothing is shaped for another implementation.

| Rule | Detail |
| --- | --- |
| Field names | Rust field names, snake_case; no `rename_all` on structs |
| Tags | Enum tags and unit variants `#[serde(rename_all = "snake_case")]` (`"status":"completing"`, `"fail_fast"`, `"as_of"`) |
| Absent fields | Every `Option` is `default` and `skip_serializing_if = "Option::is_none"`. `null` is a value only inside `data`, `detail`, `input`, `checkpoint`, `result`, `trace` and document values; the `Option<Json>` fields among them read a present `null` as `Some(Null)` (a `deserialize_with` helper), so it round-trips |
| Booleans | `background` and `abort_requested` always written; `abandon_on_restart` only when `true` |
| Numbers | Ids and sequences are `u64`, timestamps `i64`; floats never appear in records |
| Unknown fields | Accepted on read; `trace` and `links` are kept explicitly, other unknown fields are dropped |
| Key order | Not significant for readers; delta writes records in field order |
| Messages | `delta_ai::Message` JSON, which keeps upstream's shape (the workspace rule for messages) |
| Ops | `delta_chord::delta::Op` tuples |

## Differences from upstream

- Newtype ids and `Seq` over `u64` instead of branded numbers, without upstream's 2^53
  limit; `RawId` marks the one untyped point.
- Record JSON, kinds and tags are delta's (snake_case, `delta.*`); upstream's camelCase
  records and `pi.*` kinds are not read.
- Status unions are tagged enums: a `placed` write or a terminal task with memos is either
  unrepresentable or refused by `check()`, where upstream relies on `never`-typed fields.
- `Option` instead of `undefined`; `Head::SelfRef` instead of a string sharing a field with ids.
- `TaskId` carries no result type; `TaskRef<R>` is added where typed waits need it.
- `trace` and `links` are opaque `Json` until delta has a telemetry crate.
- One `StorageError` enum with `is_request()` instead of an error subclass; the Session's
  exemption rule reads the variant.
- Errors are `thiserror` enums with `source()` chains instead of `Error` subclasses.

## Decisions

- Ids are `u64` newtypes with no minting cap of their own (README).
- Record JSON is serde-native snake_case; built-in kinds are `delta.*` (README); no
  compatibility with pi's records.

- `RawId` plus a sealed `RecordId` for typed minting (phase 1): `Storage` stays object-safe.
- `TaskRecord` keeps `trace`/`links` as `Json` (phase 1).
- Module layout: `records/` with one file per record kind (`ids`, `conversation`, `entry`,
  `submission`, `task`, `document`, `write`, `query`, `error`).

## Open (minor)

- `TxError::Callback(Box<dyn Error>)` as the carrier of a commit closure's own error, versus
  an error type parameter on `commit`.
