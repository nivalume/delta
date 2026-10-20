# Session (`session`)

Follows spec §3.3 (access and creation), §3.4 (mutation ownership), §3.7 (fork staging),
§4 (transactions and storage ownership), §9.1 (document state) and §9.2 (`watchDoc`, as
subscriptions), §10.1 (failures, the Session side) and the transaction items of §12.
Corresponds to upstream `src/session/{session,transaction,forks,observation}.ts`.

## Problem

One mutation line commits entries, task records and document changes atomically and
publishes only what storage accepted. Upstream serializes commits on a promise chain: an async
callback holds the line, reads tables through `tx`, edits documents through revocable proxy
drafts, and every other read (`snapshot()`, `watchDoc()`, a cold document load) also queues
on the line, so calling one inside a callback deadlocks; drafts are revoked by runtime checks
when the callback settles, and a `tx.doc()` still pending at settlement is aborted and
drained. delta puts the line in one tokio task that owns the session state. A commit is a
closure that receives `&mut Tx`; `Tx` borrows the line's state, so a cold load inside a
commit is an ordinary `await` on the same task, a draft cannot outlive the closure (it borrows
the `Tx`), and nothing is pending at settlement because the closure's future runs to
completion before settlement starts.

## Architecture

```text
 Session (Clone handle) ──mpsc──► line task: Line { storage: Arc<dyn Storage>, cache, observers, failure, clock }
      │                                 │
      │ commit(ctx, |tx| ..)            ├── one Job at a time: commit | read_on_line | sweep | close
      │ snapshot / document_state       │
      │ subscribe_commits / closed      ▼
      ▼                            Tx<'line> { &mut Line, ctx, staged writes, open Changes, read state }
 oneshot result ◄─────────────────────────── settle → prepare → storage.commit → adopt → publish
```

- `Session` is a cheap handle (`Clone + Send + Sync`); every clone is the same session.
  Dropping the last handle closes it.
- A job is `Box<dyn FnOnce(&mut Line) -> BoxFuture<'_, ()> + Send>`; jobs run strictly in
  arrival order, one at a time, and a job's future is polled to completion (never dropped on
  caller cancellation), so the line is never left mid-transaction.
- A `snapshot` cache hit is answered without a job from a shared read of the cache; anything
  that loads or registers is a job, as upstream.

## API

```rust
#[derive(Clone)]
pub struct Session;

pub struct SessionOptions {
    pub document_retention: Option<Duration>,     // Some(0): drop after use; None: keep; default 10 min
    pub now: Arc<dyn Fn() -> i64 + Send + Sync>,  // ms wall clock for task times; default system time
    pub reporter: Option<Reporter>,
    pub services: Services,                       // in-memory services of the layers above
}

/// A type map of per-session, in-memory services (`http::Extensions`-like). Never persisted,
/// dropped at close; the kernel stores them and never reads them.
pub struct Services { /* RwLock<HashMap<TypeId, Arc<dyn Any + Send + Sync>>> */ }
impl Services {
    pub fn insert<T: Send + Sync + 'static>(&self, service: Arc<T>) -> Option<Arc<T>>;
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<Arc<T>>;
}

impl Session {
    pub fn open(storage: Arc<dyn Storage>, options: SessionOptions) -> Session;

    pub async fn commit<T, F>(&self, ctx: &Context, f: F) -> Result<T, SessionError>
    where
        F: for<'a, 'l> FnOnce(&'a mut Tx<'l>) -> BoxFuture<'a, Result<T, TxError>> + Send + 'static,
        T: Send + 'static;

    pub async fn snapshot<D: DocDefinition>(&self, at: DocAddress<D>, ctx: &Context) -> Result<Option<Json>, SessionError>;
    pub async fn snapshot_as_of<D: ConversationDoc<History = Rewindable>>(&self, at: DocAddress<D>, entry: EntryId, ctx: &Context) -> Result<Option<Json>, SessionError>;
    pub async fn document_state<D: DocDefinition>(&self, at: DocAddress<D>, ctx: &Context) -> Result<Option<DocumentState>, SessionError>;

    pub fn subscribe_commits(&self) -> CommitSubscription;                           // off the line
    pub fn observe_commits(&self, observer: Box<dyn CommitObserver>) -> ObserverGuard; // on the line; Session components only
    pub fn subscribe_close(&self) -> CloseSubscription;
    pub async fn close(&self, ctx: &Context) -> Result<(), SessionError>;
    pub fn closed(&self) -> impl Future<Output = SessionEnd> + Send;
    pub fn failure(&self) -> Option<SessionFailed>;
    pub fn services(&self) -> &Services;
}
pub enum SessionEnd { Closed, Failed(SessionFailed) }
```

Services let an upper layer keep process-local state per session without a kernel type
for it: the approval waiters of [approval.md](approval.md), for instance, are a service the
agent layer inserts at open and reads from its tasks and its `decide` call.

A commit (`Live` and `Follow` stand for any document and task definitions):

```rust
let entry_id = session.commit(&ctx, |tx| Box::pin(async move {
    let conversation = tx.conversation(id).await?;                  // table read
    let mut live = tx.doc(DocAddress::<Live>::new(id)).await?;      // draft; loads from storage if needed
    live.draft().into_object().unwrap().remove("generation");
    drop(live);                                                     // the borrow ends; the Change stays open
    let entry = tx.append_entry(id, draft).await?;                  // first table write
    tx.create_task(&Follow, input, TaskOptions::conversation(id)).await?;
    Ok(entry.id)
})).await?;
```

### `Tx`

```rust
pub struct Tx<'line> { /* &'line mut Line, ctx, scope, writes, tasks, submissions, documents, has_table_write */ }

impl Tx<'_> {
    // Table reads: allowed until the first table write, then `TxError::ReadAfterWrite`.
    pub async fn conversation(&mut self, id: ConversationId) -> Result<Option<ConversationRecord>, TxError>;
    pub async fn entry(&mut self, id: EntryId) -> Result<Option<EntryRecord>, TxError>;
    /// `None` when absent or of another kind.
    pub async fn typed_entry<D: DeserializeOwned>(&mut self, kind: &EntryKind<D>, id: EntryId) -> Result<Option<TypedEntry<D>>, TxError>;
    pub async fn task(&mut self, id: TaskId) -> Result<Option<TaskRecord>, TxError>;
    pub async fn scan_conversations(&mut self, q: ConversationQuery, limit: usize, cursor: Option<Cursor>) -> Result<Page<ConversationRecord>, TxError>;
    pub async fn scan_entries(&mut self, q: EntryQuery, limit: usize, cursor: Option<Cursor>) -> Result<Page<EntryRecord>, TxError>;
    pub async fn scan_tasks(&mut self, q: TaskQuery, limit: usize, cursor: Option<Cursor>) -> Result<Page<TaskRecord>, TxError>;
    pub async fn latest_head_marker(&mut self, conversation: ConversationId) -> Result<Option<EntryRecord>, TxError>;
    pub async fn submission_by_request(&mut self, conversation: ConversationId, request_id: &str) -> Result<Option<SubmissionRecord>, TxError>;

    // Table writes.
    pub async fn create_conversation(&mut self, ownership: ConversationOwnership) -> Result<ConversationRecord, TxError>;
    pub async fn fork_conversation(&mut self, parent: ConversationId, at: EntryId, ownership: ConversationOwnership) -> Result<ConversationRecord, TxError>;
    pub async fn append_entry(&mut self, conversation: ConversationId, draft: EntryDraft) -> Result<EntryRecord, TxError>;
    pub async fn append_typed_entry<D: Serialize>(&mut self, kind: &EntryKind<D>, conversation: ConversationId, draft: TypedEntryDraft<D>) -> Result<TypedEntry<D>, TxError>;
    pub async fn create_task<T: TaskDefinition>(&mut self, task: &T, input: T::Input, options: TaskOptions) -> Result<TaskId, TxError>;
    /// Raw record, no admission rules; hosts use `Conversation::submit`.
    pub async fn create_submission(&mut self, create: SubmissionCreate) -> Result<SubmissionRecord, TxError>;
    pub fn settle_submission(&mut self, id: SubmissionId, settlement: SubmissionSettlement);
    pub fn place_submission(&mut self, id: SubmissionId, entry: EntryId);

    // Documents: get-or-create, memoized per address; allowed after table writes.
    pub async fn doc<D: DocDefinition<Seed = ()>>(&mut self, at: DocAddress<D>) -> Result<Doc<'_, D>, TxError>;
    pub async fn doc_member<D: DocDefinition>(&mut self, at: DocAddress<D>, seed: D::Seed) -> Result<Doc<'_, D>, TxError>;
    pub async fn retire_doc<D: DocDefinition>(&mut self, at: DocAddress<D>) -> Result<(), TxError>;

    pub fn context(&self) -> &Context;
}

/// A borrowed handle on the transaction's `Change` of one incarnation.
pub struct Doc<'tx, D> { /* &'tx mut Change, PhantomData<D> */ }
impl<D> Doc<'_, D> {
    pub fn value(&self) -> &Json;                                   // the working value so far
    pub fn draft(&mut self) -> delta_chord::delta::Draft<'_>;
    pub fn set(&mut self, value: Json);                             // whole-value replacement, diffed
    pub fn update_as<T: Serialize + DeserializeOwned, R>(&mut self, f: impl FnOnce(&mut T) -> R) -> Result<R, serde_json::Error>;
}
```

- `Doc` borrows the `Tx` mutably: while held, no other `Tx` method can run, and it cannot
  escape the closure. Reacquiring an address returns a handle on the same open `Change`, so
  "edit, table write, edit again" is `doc`, `drop`, `append_entry`, `doc`.
- `settle_submission` and `place_submission` resolve at assembly against the transaction's
  latest candidate record, falling back to committed state; they count as table writes and
  are not caller reads. Placement turns a queued input `placed` and a queued write `done`;
  only a placed input takes `done { answer }`; a settled record stays unchanged.
- Task-scoped `doc` validates against the latest candidate task record (not a caller read):
  a missing task rejects; a terminal candidate rejects later task-document access.
- `pub(crate)` for the scheduler and the layers above: `set_task` (whole-record replacement, stamps `started_at` on
  the first `running` and `ended_at` on `terminal`, refuses a conversation change or a second
  terminal candidate), `create_root_conversation`, `staged_tasks`, `staged_conversations`,
  and a `TransactionScope` binding the default conversation of `create_task` and the task
  stamped as `by_task_id` on appended entries.

## Commit lifecycle

```text
admission      on the line, before the closure: healthy, not closing, ctx not cancelled
closure        runs to completion with &mut Tx; reads and writes stage; documents load and open chord Changes
health check   a closure that caught a failed read must not commit: SessionFailed
seal           every Doc is gone (borrow); prepare() each open Change -> Prepared (value + ops)
assemble       plan each document (below); reject fork-source writes; validate owners; validate task
               replacements (exists, not terminal, same conversation); retire the task documents of tasks
               turning terminal (scan at Current for committed tasks); resolve publication conversation ids;
               resolve submission changes; run checkpoint_when last
no writes      drop the Prepareds; return the closure's value; nothing is published
storage.commit with ctx.without_cancel(): once admitted, caller cancellation does not interrupt settlement
error          the guard has failed the Session; drop the Prepareds; return SessionError::Failed
adopt          State::commit(prepared) per plan; update cache bookkeeping; a failure here fails the Session
publish        on-line observers synchronously; the CommitPublication to subscribers' channels
evict          sweep expired idle documents; release the line
```

A closure error, or a `TxError` from assembly, rolls back: every `Change` is dropped and
nothing is written. Table rules: reads before the first table write only; document access and
read-your-writes stay available after it; creation methods return their record, so nothing is
read back. Owner validation judges the owner's final candidate in the transaction: a missing,
`completing`, terminal or abort-marked owner rejects a new owned task or conversation; a child
task with `background` or with another conversation than its owner rejects. The persisted
`ConversationOwner` is derived from the owner task's candidate.

### Planning a document write

| Acquired as | Content write | Publication |
| --- | --- | --- |
| created (absent address; `initial`) | `DocumentCreate` with base `{VERSION, prepared.value}`, also when the batch is empty | value, empty ops |
| loaded, stored version < `VERSION` | `DocumentChange` base (required migration base) | value, ops; observers of the old version get `["r", value]` |
| loaded, non-empty batch, `checkpoint_when` true | `DocumentChange` base | value, ops |
| loaded, non-empty batch, false | `DocumentChange` delta `{VERSION, ops}` | value, ops |
| loaded, empty batch | nothing | nothing |
| fork copy, not accessed | `DocumentCopy` | `document.copy` descriptor, no value |
| fork copy, accessed through a definition | `DocumentCreate` with the migrated, edited value | value, empty ops |
| retire (`retire_doc`, or its task turned terminal) | `DocumentRetire`; an acquired draft's final content is written first | `Null`, `["r", null]` |

`checkpoint_when` runs once per change with `deltas_since_base` from the cache; at adoption a
base resets the count and a delta increments it. An incarnation created and retired in one
commit is written (base, then retire) but never enters the cache.

## Document cache

```rust
struct LoadedDocument {
    address: AddressId,
    record: DocumentRecord,
    stored_version: u32,     // persisted; older while the value is migrated only in memory
    value_version: u32,      // the definition version the value is shaped for
    deltas_since_base: u64,
    state: delta_chord::state::State,   // the tracker and its subscribers
    pinned: usize,           // attached DocumentStates
    last_use: Instant,
}
```

- Load: `find_document(address, Current)`, `document(id, Current)`, check scope, history, fork
  and version, migrate in memory, `State::new(value)`. A cached document serves only
  definitions of its `value_version`; another version drops it and reloads from storage, so a
  reloaded extension migrates from the stored bytes.
- Change: `state.begin()` is the `Change` behind `Doc`; `state.commit(ctx, prepared)` adopts
  after storage success and publishes to every subscription (chord `state.md`). The Session
  holds the only writable `State` handle; observers get a read-only wrapper.
- Retention: a document with no attached state leaves the cache once unused for
  `document_retention` (default 10 min; `Some(0)`: right after use; `None`: never). Eviction
  runs on the line after each commit and from a tokio timer armed for the oldest idle
  document's expiry; a hot document moves to the back of the idle order at most once per
  granule (a tenth of the retention, at most 1 s). An attached document never leaves;
  retirement evicts it and unpins its observers.

## Observation (§9.1, §9.2)

```rust
pub struct DocumentState { /* read-only view of the State, pin guard, end */ }
impl DocumentState {
    pub fn value(&self) -> Json;                   // last published revision; `Json::Null` after retirement
    pub fn subscribe(&self) -> Subscription;       // chord: `Hydrate`, then `Update { value, ops, context }`
    pub fn end(&self) -> Option<WatchEnd>;
    pub async fn closed(&self) -> WatchEnd;
}
pub enum WatchEnd { Retired, SessionClosed, SessionFailed(SessionFailed) }
```

- Acquisition is a line job: resolve the current incarnation (load if needed), check the
  definition, capture the current revision and register, atomically with respect to commits.
  Absent returns `None` and never creates. The state is bound to that incarnation; a later
  recreation at the address needs a new acquisition.
- Deliveries are chord `Delivery`s: every committed revision with the exact adopted ops, in
  commit order, off the line. A subscriber that falls behind gets merged batches (chord's
  rule) rather than upstream's `["r", newest]` past 100 frames. Retirement delivers
  `["r", null]` with value `Null` and ends the stream (`Retired`). A migration-only base
  delivers nothing to observers of the new version.
- Dropping the `DocumentState` unpins the document; dropping a `Subscription` unsubscribes.
  Session close ends every state at its seal (`SessionClosed`), a failure with
  `SessionFailed`; a commit settling during close delivers nothing. Delivered values are
  `Json`: a holder may mutate its copy freely.
- Upstream's `watchDoc` (one serialized async listener, `stop()`, `listener_error`) is a loop
  over `subscribe().next()` in a spawned task whose context cancellation drops it. There is no
  separate watch type.
- `snapshot`: the current cached revision when the cache holds the definition's version,
  otherwise a load on the line; migrates in memory, writes nothing, `None` when absent.
- `snapshot_as_of(D, conversation, entry)`: on the line, `visible_entry(conversation, entry)`
  (invisible: `TxError::Fork`), then `find_document` for the address re-scoped to the entry's
  own conversation at `At(commit_seq)`, then `document(id, At(commit_seq))`, then migrate in
  memory. It never starts from today's incarnation and never creates; `None` when no instance
  was alive at that commit.

## Forks

`fork_conversation(parent, at, ownership)`:

1. `visible_entry(parent, at)` must exist (`TxError::Fork`); keep its `conversation_id` and
   `commit_seq`.
2. Scan the entry's own conversation at `At(commit_seq)` and take the documents with
   `fork: as_of`; scan `parent` at `Current` and take those with `fork: current`. Each becomes
   a staged copy `{record: DocumentCreate with a fresh id and the child's scope, source: {id, at}}`;
   a second source for one child address rejects (`"Fork selects multiple source documents"`).
3. Stage the conversation record (owner derived from the owning task's final candidate), then
   run the conversation-creation hook the layer above installs (the agent layer stages its
   documents there, [harness.md](harness.md)).
4. At assembly, reject if the transaction writes any source incarnation or any
   `fork: current` document of `parent`
   (`"Cannot fork conversation {id} while changing its current-policy documents"`).
5. A copy acquired through `tx.doc` in the same transaction is read from storage
   (`document(source)`), checked against the copy's record, migrated and edited, and becomes a
   `created` plan with the final value; an unaccessed copy stays a `DocumentCopy` write.

## Publications and listeners

```rust
pub struct CommitPublication { pub seq: Seq, pub changes: Vec<CommitChange>, pub context: Context }
pub enum CommitChange {
    Conversation(ConversationRecord), Entry(EntryRecord), Task(TaskRecord), Submission(SubmissionRecord),
    /// `value: None` and `version: None`: retired. `ops` empty for creation.
    Document { record: DocumentRecord, conversation_id: Option<ConversationId>, version: Option<u32>, value: Option<Json>, ops: Arc<[Op]> },
    DocumentCopy { record: DocumentRecord, conversation_id: ConversationId, source: DocumentCopySource },
}
pub trait CommitObserver: Send {
    fn committed(&mut self, publication: &CommitPublication) -> Result<(), Box<dyn Error + Send + Sync>>;
}
```

- On-line observers (`observe_commits`) are components registered at open (the scheduler,
  and the indexes of the layers above: submission waiters, view mounts, the task graph):
  they run synchronously after adoption to keep memory in
  step with storage, before subscribers are fed, and one that fails fails the Session.
- `subscribe_commits` returns a bounded channel of `Arc<CommitPublication>`; a slow subscriber
  delays only itself, and on overflow the subscription records a gap so the subscriber can
  resynchronize from storage. Host listeners never run on the line and may call Session APIs.
- `subscribe_close` fires when close begins, failure included, after admission is sealed; the
  Harness ends waits and streams there and signals invocations after every close listener ran.

## Failures (§10.1)

Every storage call goes through one guard on the line:

| Error | Effect |
| --- | --- |
| any error from `commit` or `mint_id` | fails the Session |
| a read error other than `Request`, not cancelled by the caller's `ctx` | fails the Session |
| `StorageError::Request`; a read whose `ctx` was cancelled | fails that call only |
| `AdoptError` after a successful commit | fails the Session (memory would be behind storage) |
| an on-line observer error; an error in a scheduler bookkeeping commit | fails the Session |
| the closure's own error, `initial`, `migrate`, `checkpoint_when`, owner validation | rolls that commit back only |

On the first failure the failing call returns its error; every later call, and every call
underway, returns `SessionFailed { cause }`; states and subscriptions end with
`SessionFailed`; the reporter gets `ReportInfo::SessionFailed` once; the Session closes
itself, storage included, and `closed()` resolves `Failed`. `close()` seals admission, runs
close listeners, lets admitted commits settle, drops the cache, then closes storage after
calls underway; a failing storage close fails the Session unless it already failed. Retrying a
failed call cannot succeed; the host reopens.

## Differences from upstream

- An actor task instead of a promise chain. `Tx` borrows the line's state, so a cold load
  inside a commit is a plain await and the "read inside a commit never resolves" footgun does
  not exist; `snapshot`, `document_state` and `snapshot_as_of` from outside still queue on the
  line.
- Drafts are borrowed handles (`Doc<'tx>`), revoked by the borrow checker: no pending
  acquisition drain, no revocation checks, no fire-and-forget mutation hazard.
- `commit` takes a closure returning a boxed future; the closure's error type is `TxError`,
  with `Callback` for the caller's own errors.
- Documents are `delta_chord::State`s and observation is pull-based: `watchDoc`'s listener
  machinery and `AttachedReplicatedState` collapse into `DocumentState::subscribe()`.
  Backlog overflow merges ops instead of sending `["r", newest]`.
- An empty prepared batch is a no-op even when the edits were structural (chord's tracker
  proves equality cheaply); upstream keeps such a batch as a durable change and publication.
- Commit subscribers run off the line and may call the Session; only internal observers run
  on it.
- `ctx.without_cancel()` replaces `withoutAbortSignal`; `document_retention` is
  `Option<Duration>`; eviction timers are tokio's, with no host `setTimer`.
- No telemetry events on publish until delta has a telemetry crate.

## Decisions

- No-op batches: chord's rule. A change that ends equal to its base, structural edits
  included, writes and publishes nothing.
- Overflow: chord's merge. A `DocumentState` subscriber that falls behind gets merged
  batches (`["r", value]` past `MAX_OPS`), not upstream's root replacement.
- `DocumentState` is a read-only wrapper over the incarnation's `delta_chord::State`; the
  Session holds the only writable handle.
- One document is edited at a time: `Doc<'tx, D>` borrows the `Tx`; "edit, table write,
  edit again" re-acquires.

## Open (minor)

- The `commit` signature (`FnOnce(&mut Tx) -> BoxFuture<Result<T, TxError>>`) versus a
  generic error type; boxing costs one allocation per commit.
- A gap marker rather than merged publications when a `subscribe_commits` channel
  overflows.
- `document_retention: Option<Duration>` with `None` meaning forever, versus an enum.
- `Session::open(Arc<dyn Storage>)` rather than a generic `S: Storage`, so the Harness stays
  non-generic.
