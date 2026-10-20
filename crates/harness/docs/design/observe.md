# Observation: view, agent events, task graph (`observe`)

Follows spec §9.2 (`watchDoc` delivery rules), §9.3 (conversation view), §9.4 (agent
events) and §9.5 (task graph view). Corresponds to upstream `src/harness/view.ts`,
`events.ts` and `task-graph.ts`. The document-level primitive (`DocumentState`, the commit
publication stream) is [session.md](session.md); this note builds on it.

## Problem

A UI needs the conversation as one value that changes in small steps (a TUI rendering a
transcript, a web client replaying ops), or as events shaped like the coding agent's
(`tool_execution_start`, `message_update`, ...). A debugger needs every live task and what
it waits for. All three must show only committed state and never block a writer. Upstream
mounts documents into one Chord value per conversation and derives events from the ops of
each commit; delta does the same on `delta_chord::State`, whose subscriptions already
deliver values with ops and merge backlog ([chord state.md](../../../chord/docs/design/state.md)).

## Document subscriptions (upstream `watchDoc`)

```rust
impl Session {   // session.md
    pub async fn document_state<D: DocDefinition>(&self, at: DocAddress<D>, ctx: &Context) -> Result<Option<DocumentState>, SessionError>;
}
// DocumentState: a read-only wrapper over the incarnation's `delta_chord::State`.
// value() -> Json; subscribe() -> DocumentSubscription: chord `Delivery`s (`Hydrate`, then
// `Update { value, ops, context }`), pinning the incarnation in the cache while it lives.
```

Rules kept from §9.2: acquisition on the session line captures the current revision in
O(1) and registers for every later committed revision; retirement delivers `["r", null]`
and ends the subscription, never following a replacement incarnation; the delivery's
context carries the commit's values without its cancellation. Two rules are chord's
(decided): a change that ends equal to its base publishes nothing, and a subscriber that
falls behind gets merged batches (ops concatenated, `["r", value]` past `MAX_OPS`) instead
of upstream's root replacement. Pull-based, so upstream's listener rules (`start`, `stop`,
`listener_error`) have no counterpart: dropping the subscription stops it.

## Conversation view

```rust
pub struct ConversationView {
    pub conversation: ConversationRecord,
    /// The head marker, then the non-head entries from its head (ContextView.entries).
    pub entries: Vec<EntryRecord>,
    /// delta.agent, delta.live, delta.inbox, delta.provider, delta.usage, and delta.approvals, keyed by kind; absent documents absent.
    pub docs: IndexMap<String, Json>,
}

impl Conversation {
    pub async fn view_state(&self, ctx: &Context) -> Result<delta_chord::State, HarnessError>;   // read-only
    pub async fn watch(&self, ctx: &Context) -> Result<Subscription, HarnessError>;
}
```

- The harness keeps at most one **mount** per conversation. The first `view_state`,
  `watch` or event attachment builds it on the session line from the committed active
  transcript (`activeEntries` over the captured context bounds) and the mounted
  documents; it is dropped with its last observer. Its value is a `delta_chord::State`.
- The mount advances from the session's commit publications (`Session::subscribe_commits`).
  One publication yields one batch:

  | change in the publication | view op |
  | --- | --- |
  | document op on a mounted kind, same incarnation and definition version | the op re-rooted: `["s", ["docs", kind, ..path], v]` (`Op::prefixed`); an `r` becomes `s` of `["docs", kind]` |
  | document created, or another incarnation / version | `["s", ["docs", kind], value]` |
  | document retired (value null) of the mounted incarnation | `["d", ["docs", kind]]` |
  | entry appended without `head` | `["p", ["entries"], len, 0, [entry]]` |
  | entry appended with `head` H | `["p", ["entries"], 0, k, [entry]]` where `k` is the index of the first non-head entry with id ≥ `H.head` (all kept entries are a suffix) |

  Document ops come first, then entry ops in id order. A publication touching nothing
  mounted makes no revision; a nonempty redundant batch is still a publication.
- Entries through `parent.at` are immutable, so a child's mount follows only its own
  conversation. A raw head write that targets before the mounted range shows only the
  mounted entries (the §12 footgun; write submissions reject such heads as stale).
- `view_state` returns the mount's `State` clone (cheap; `value()` is O(1)); `watch`
  returns `state.subscribe()`, whose first delivery hydrates. Both end at session close.
- The mount is structural: no reduction, no persistence. UI reduction (display-only
  entries, model filtering) and model-context reduction ([context.md](context.md)) are
  separate consumers.

## Agent events

```rust
pub async fn watch_events(harness: &Harness, id: ConversationId, ctx: &Context) -> Result<AgentEventStream, HarnessError>;

pub struct AgentEventStream;
impl AgentEventStream {
    pub fn snapshot(&self) -> &Snapshot;                        // captured at attachment
    pub async fn next(&mut self) -> Option<Vec<AgentEvent>>;    // one batch per commit; a snapshot batch after overflow
}

#[derive(Serialize)] #[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    Snapshot(Snapshot),
    RunStart { inputs: Vec<SubmissionId> }, RunEnd { inputs: Vec<SubmissionId> },
    TurnStart, TurnEnd,
    MessageStart { message: Message },
    MessageUpdate { usage: Usage, changes: Vec<MessageChange> },
    MessageEnd { entry: EntryRecord },
    ToolExecutionStart { #[serde(flatten)] call: ToolEventCall, args: JsonObject },
    ToolExecutionUpdate { #[serde(flatten)] call: ToolEventCall, output: Option<OutputChange>, details: Option<Json>, diagnostics: Option<Vec<ToolDiagnostic>> },
    ToolExecutionEnd { #[serde(flatten)] call: ToolEventCall, entry: Option<EntryRecord>, result: Option<NestedToolExecutionResult> },
    InboxUpdate { items: Vec<QueuedItem> },
    Submission { record: SubmissionRecord },
    AutoRetryStart { attempt: u32, at: u64, error_message: String }, AutoRetryEnd { attempt: u32 },
    DeferredPoll { poll_at: u64 },
    EntryAppended { entry: EntryRecord },
    AgentChanged { agent: AgentState }, UsageChanged { usage: UsageState },
    TaskFailed { task_id: TaskId, kind: String, message: String },
    CompactionStart { task_id: TaskId, reason: CompactionReason, blocking: bool },
    CompactionEnd { task_id: TaskId, reason: CompactionReason, error: Option<String> },
    ApprovalUpdate { requests: Vec<ApprovalRequest> },          // delta (approval.md)
}

pub struct ToolEventCall { tool_call_id: String, tool_name: String, task_id: Option<TaskId>, parent_tool_call_id: Option<String>, parent_task_id: Option<TaskId> }
pub enum OutputChange { Delta { trim_start: Option<u64>, append: Option<String> }, Set(String) }
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageChange {
    TextStart | ThinkingStart | ToolcallStart { content_index: usize, block: AssistantBlock },
    TextDelta | ThinkingDelta { content_index: usize, delta: String },
    ToolcallDelta { content_index: usize, path: Vec<Seg>, delta: String },
    Block { content_index: usize, block: AssistantBlock },
    Message { message: AssistantMessage },
}
```

The `Snapshot` holds `entries`, `run`, `generation { attempt, message, retry, deferred }`,
`tools`, `nested_tools`, `compactions`, `compaction_failure`, `inbox` (id and mode per
item), `agent` (`{}` when absent), `usage`, and delta's `approvals`. Names and JSON tags
are upstream's `AgentSessionEvent` names, the same `kind()` strings
`delta_agent::AgentEvent` reports (`message_start`, `tool_execution_update`, ...), so a UI
built on `delta_agent` events reads these with the known vocabulary; `run_start`/`run_end`
stand where `agent_start`/`agent_end` do there. Unlike `delta_agent`'s, these events
serialize: they are the base of the later UI protocol crate.

Attachment captures the snapshot and registers on the session line atomically. Every event
is relative to the snapshot and the events before it, never to document paths. The stream
buffers at most 100 undelivered batches; one more replaces them all with a single
`Snapshot` of the newest state. Nothing is replayed; a reconnecting consumer attaches
again. `Submission::wait` reports exact outcomes.

### Derivation

The translator receives, per publication, the view before and after, the view ops, and the
publication's changes (entries, task records, submissions, and `NestedCallDoc` values of
this conversation). `before`/`after` comparisons use the mounted values; ops decide deltas.

| op pattern / record change | event |
| --- | --- |
| `delta.live.run` appears, or its first input differs from before | `run_end` of the old (if any), then `run_start` |
| `run.task_id` changes to a `delta.generation` task created in this commit | `turn_start` |
| a `delta.generation` record becomes `completing` (first time) or terminal without having held | `turn_end` (once per commit) |
| `delta.live.generation.message` appears | `message_start` with the partial |
| ops below `docs.delta.live.generation.message` on an existing partial | `message_update` with the partial's usage and the translated changes (below); only-`usage` changes send no event |
| a `tools` or `nested_tools` slot becomes `running` | `tool_execution_start`; `args` from the task's intent checkpoint published in the same commit (nested: the slot's arguments) |
| ops below a running slot's `output` | `tool_execution_update.output`: `t` ops sum into `trim_start`, `a` ops concatenate into `append`; any other op, or a changed output without `t`/`a`, is `Set(slot.output)` |
| a running slot's `details` / `diagnostics` differ | the same event's `details` (`null` when removed) / `diagnostics` (`[]` when removed) |
| `generation.retry` appears / disappears | `auto_retry_start { attempt, at, error }` / `auto_retry_end { attempt }` |
| `generation.deferred.poll_at` appears or moves | `deferred_poll` |
| a slot becomes `done`, or is created `done` (a call not offered) | `tool_execution_end` with its result entry when the commit appended it |
| an unfinished slot disappears (its run ended) | `tool_execution_end` with the result entry appended for its call id in the same commit, else none |
| a nested slot ends or disappears unfinished | `tool_execution_end` with the `NestedCallDoc` result published in this commit, else none; children before parents (the lists hold parents first; walk backwards) |
| entry appended with `model` | `message_start` (unless it is the streamed assistant answer whose partial already started) then `message_end`; a tool result's `tool_execution_end` directly precedes its `message_start` |
| entry appended without `model` | `entry_appended` |
| a status leaves `delta.live.compactions` | `compaction_end`, with `error` when `compaction_failure` names that task in the same commit |
| a task record settles `faulted` / `orphaned` | `task_failed { task_id, kind, message }` |
| submission records in the publication | `submission { record }` in id order |
| `delta.inbox` value differs | `inbox_update` (retired reads as initial) |
| `delta.agent` / `delta.usage` value differs | `agent_changed` / `usage_changed` |
| `delta.approvals` value differs | `approval_update` |
| a status joins `delta.live.compactions` | `compaction_start { task_id, reason, blocking }` |

`message_update` translation of the ops under `generation.message`: an insert-only `p`
into `content` at index `i` starts one block per inserted item (`text_start`,
`thinking_start`, `toolcall_start` with `content_index = i + offset`); `a` on
`content[i].text` / `.thinking` is a delta; `a` anywhere under `content[i].arguments` is
`toolcall_delta` with the path relative to `arguments`; any other op inside `content[i]`
sends that `block` once (later ops on it in the batch are folded, since the block already
holds them); any other op on `content`, on the message, or on the generation sends the
whole `message`. Ops on `usage` are skipped. Only deltas travel, so throttled partials
cause no write amplification.

One commit's batch is ordered as upstream: `tool_execution_start`, `message_start` of a
first partial, `message_update`, `tool_execution_update`, retry/deferred; entries in
append order with their `tool_execution_end` / `message_start` / `message_end` /
`entry_appended`, nested ends first; tool ends without an entry; `compaction_end`,
`task_failed`, `turn_end`, `run_end`; `submission`s; `inbox_update`, `agent_changed`,
`usage_changed`, `approval_update`; last `compaction_start`, `run_start`, `turn_start`.

The attachment also scans the conversation's `completing` generations on the line, so a
turn already ended by a held outcome does not end again when the task goes terminal.

## Task graph

```rust
#[serde(tag = "status", rename_all = "lowercase")]
pub enum TaskGraphState {
    Pending { phase: String }, Running { phase: String },
    Waiting { phase: String, on: Vec<TaskId>, policy: JoinPolicy },
    Sleeping { phase: String, until: u64 },
    Completing { outcome: OutcomeStatus },
}
pub struct TaskGraphNode {
    pub id: TaskId, pub kind: String, pub conversation_id: ConversationId,
    pub owner: Option<TaskId>, pub background: bool, pub abort_requested: bool,
    pub state: TaskGraphState,
    pub conversations: Vec<ConversationId>,            // owned, in id order; only grows
    pub blocked: Option<BlockedReason>,                 // missing_task | task_too_old | migration_failed | restart_limit
}
pub struct TaskGraph { pub tasks: BTreeMap<String, TaskGraphNode> }   // keyed by decimal id

impl Harness {
    pub async fn task_graph(&self, ctx: &Context) -> Result<State, HarnessError>;
    pub async fn watch_task_graph(&self, ctx: &Context) -> Result<Subscription, HarnessError>;
}
```

- One mount for the session, built on the line by the first observer: scan every live task
  (`pending`, `running`, `waiting`, `sleeping`, `completing`), sorted by id, and one
  `scan_conversations(owner_task_id)` per task. Dropped with its last observer.
- A node is the committed record without input, checkpoint payload, outcome payload and
  memos. `running` is the durable status: open rewrites surviving `running` tasks to
  `pending`, so they show `pending` until reserved again.
- Advance per publication: a task record that is terminal deletes `tasks[id]`; another
  record sets `tasks[id]` when the node differs; a conversation created with an owner
  whose node exists sets that node's `conversations` (sorted again, since change order
  within a publication is unspecified). Tasks are processed before conversations so a
  task and its conversation created in one commit find each other.
- `blocked` derives from the registry, the settings and the record's `restarts`; the
  scheduler exposes `blocked(record) -> Option<BlockedReason>` and a change notification
  ([scheduler.md](scheduler.md)); on notification the mount recomputes every node's
  `blocked` on the line and publishes `s`/`d` ops for the ones that changed, with a
  background context.
- Live tasks only: a conversation whose owner task finished has no node listing it; a UI
  that nests it reads `ConversationRecord.owner`.
- Read-only: observing never schedules. Session close ends both.

## Differences from upstream

- Mounts are `delta_chord::State`s; `view_state`/`task_graph` return the state and
  `watch`/`watch_task_graph` return its subscription, so there is no separate
  `AttachedReplicatedState` / `WatchHandle` pair and no listener protocol. Overflow merges
  ops instead of replacing with `r` (same convergence, more deltas survive).
- `AgentEventStream` is pull-based with batches; upstream's `start(listener)` is a loop
  over `next()`.
- Events serialize (serde-tagged), where `delta_agent::AgentEvent` does not.
- `approval_update` and `docs["delta.approvals"]` are delta additions.
- Op translation reads `delta_chord::delta::Op` variants instead of tuple positions.

## Decisions

- `delta.approvals` is mounted in the view and drives the `approval_update` event
  ([approval.md](approval.md)).
- chord's rules for no-op batches and overflow (merged batches) apply to the mounts and to
  document subscriptions.
- `DocumentState` is a read-only wrapper; the view and the task graph hand out their
  `State`'s read-only side the same way.

## Open (minor)

- `Snapshot.generation.message` as `AssistantMessage` (typed) versus `Json`; typed costs a
  deserialization per batch, `Json` keeps the mount's sharing.
- Whether the events module lands in phase 1 or waits for the protocol crate; the view and
  the task graph are needed by the TUI either way.
