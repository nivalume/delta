# Harness, conversations, registry (`harness`, `registry`)

Corresponds to upstream `src/harness/harness.ts`, `agent.ts`, `registry.ts`, `provider.ts`,
`define.ts` and the host half of `src/harness/types.ts`. Follows spec §2.2 (the public
surface), §7.1 (extensions, registry, agent resolution), §7.2 (hooks) and §7.5 (extension
reload). Submissions are in [inbox.md](inbox.md), tasks in [scheduler.md](scheduler.md),
the built-in documents' definitions in [documents.md](documents.md), the session commit
API in [session.md](session.md).

## Problem

A host opens a storage and gets conversations it can submit to, configure, fork, abort and
observe. The code that runs inside (tools, prompt sections, hooks, task definitions) is not
persisted: it comes from an application-owned registry of extensions, selected per
conversation by name in a stored document, so a process restart or an in-process reload
finds the same names and resolves them against whatever is installed now. Settings and
models are read live at each use and never copied into storage.

## Opening

```rust
pub struct HarnessOptions {
    pub models: ModelsSource,                       // Models, or a closure called at each use
    pub registry: Registry,                         // must hold the built-in tasks
    pub settings: SettingsSource,                   // HarnessSettings, or a closure; read at each resolution
    pub env: Option<Arc<dyn EnvBuilder>>,           // builds a conversation's ExecutionEnv (env.md); never on the line
    pub conversation_created: Option<Arc<dyn Fn(&mut Tx, &ConversationRecord) -> Result<(), TxError> + Send + Sync>>,
    pub clock: Arc<dyn Clock>,                      // now() -> i64 epoch ms; default system time
    pub on_wake: Option<Arc<dyn Fn(Wake) + Send + Sync>>,
    pub on_report: Option<Arc<dyn Fn(ReportError, ReportInfo) + Send + Sync>>,
}

pub enum ModelsSource { Fixed(Models), Dynamic(Arc<dyn Fn() -> Models + Send + Sync>) }
pub enum SettingsSource { Fixed(HarnessSettings), Dynamic(Arc<dyn Fn() -> HarnessSettings + Send + Sync>) }

impl Harness {
    pub async fn open(storage: Arc<dyn Storage>, options: HarnessOptions, ctx: &Context) -> Result<Harness, OpenError>;
    pub fn resume(&self) -> Result<(), ClosedError>;
    pub fn wake(&self) -> Wake;
    pub async fn root(&self, ctx: &Context, options: RootOptions) -> Result<Conversation, HarnessError>;
    pub async fn conversation(&self, id: ConversationId, ctx: &Context) -> Result<Option<Conversation>, HarnessError>;
    pub async fn create_conversation(&self, options: ConversationCreateOptions, ctx: &Context) -> Result<Conversation, HarnessError>;
    pub async fn get_task(&self, id: TaskId, ctx: &Context) -> Result<Option<TaskRecord>, HarnessError>;
    pub async fn inspect(&self, ctx: &Context) -> Result<HarnessInspection, HarnessError>;
    pub async fn submission(&self, id: SubmissionId, ctx: &Context) -> Result<Option<Submission>, HarnessError>;
    pub async fn abort_submission(&self, id: SubmissionId, conversation: Option<ConversationId>, ctx: &Context) -> Result<SubmissionAbort, HarnessError>;
    pub async fn abort_task(&self, id: TaskId, ctx: &Context) -> Result<TaskAbort, HarnessError>;   // Marked | Terminal
    pub async fn retry_task(&self, id: TaskId, ctx: &Context) -> Result<(), HarnessError>;
    pub async fn wait_for_task(&self, id: TaskId, ctx: &Context) -> Result<SettledTask, HarnessError>;
    pub async fn wait_for_idle(&self, ctx: &Context) -> Result<(), HarnessError>;
    pub async fn usage(&self, ctx: &Context) -> Result<UsageState, HarnessError>;
    pub async fn task_graph(&self, ctx: &Context) -> ..;                       // observe.md
    pub fn session(&self) -> &Session;                                         // Harness derefs to Session
    pub async fn close(&self, ctx: &Context) -> Result<(), CloseError>;
}
```

- `Harness` is a cheap handle (`Clone`, `Arc` inside); `Deref<Target = Session>` gives the
  generic document operations.
- `open` runs reconciliation ([scheduler.md](scheduler.md)), rejects a registry snapshot
  that lacks a built-in task, and starts paused: nothing runs until `resume()` or a
  progress-asking call. Install extensions before open so recovered work resumes at once;
  an install after open also unblocks it.
- `root` creates conversation `1` lazily in one commit (ownerless, creation hook, `agent`
  change, `init`); when it exists, `agent` and `init` are ignored.
- `create_conversation` / `fork` commit atomically: the record with its explicit
  `ConversationOwnership`, the creation hook's documents, the `agent` change and `init`'s
  writes. Hosts must choose ownership; nothing infers it.
- `close()`: see [scheduler.md](scheduler.md#harness-level-operations). After it begins
  every new operation fails with `Closed`; queued commits settle; waits, states and watches
  end with the close reason, except `Submission::wait` of a settled submission and the
  acquisition of an absent document.

**Creation hook.** Every commit that creates or forks a conversation, through the
conveniences or raw `tx.create_conversation()` / `tx.fork_conversation()` (inside a tool
commit, say), runs the built-in hook inside those calls, before they return: empty
`pi.live`, `pi.inbox`, `pi.usage`; `pi.provider` with a fresh UUIDv7; and `pi.agent`:

| Path | `pi.agent` |
| --- | --- |
| Fork | The `as_of` copy of the parent's (documents.md, fork policy) |
| New task-owned conversation | A copy of the owner task's conversation's stored document, every field |
| New ownerless conversation | `{}` |

`conversation_created`, if given, then runs in the same commit with the record (`parent` and
`owner` tell the paths apart); table reads there and in `init` fail with `ReadAfterWrite`;
an error fails the creating commit. The conveniences then apply `agent` and run `init`.

## Conversations and submissions

```rust
#[derive(Clone)]
pub struct Conversation { /* Arc<HarnessInner>, id */ }   // stateless; compare by id

impl Conversation {
    pub fn id(&self) -> ConversationId;
    pub async fn submit(&self, draft: SubmissionDraft, ctx: &Context) -> Result<Submission, SubmitError>;
    pub async fn agent(&self, ctx: &Context) -> Result<Arc<Agent>, HarnessError>;       // current snapshot + settings; never writes
    pub async fn configure(&self, change: AgentChange, ctx: &Context) -> Result<(), HarnessError>;   // its own commit
    pub async fn commit<R>(&self, ctx: &Context, f: impl FnOnce(&mut Tx) -> Result<R, TxError> + Send) -> Result<R, HarnessError>;
    pub async fn context(&self, at: Option<EntryId>, ctx: &Context) -> Result<ContextView, HarnessError>;
    pub async fn entries(&self, query: EntryQuery, limit: usize, cursor: Option<Cursor>, ctx: &Context) -> Result<Page<EntryRecord>, HarnessError>;
    pub async fn fork(&self, at: EntryId, options: ConversationCreateOptions, ctx: &Context) -> Result<Conversation, HarnessError>;
    pub async fn compact(&self, instructions: Option<String>, ctx: &Context) -> Result<TaskId, HarnessError>;
    pub async fn reset(&self, handoff: Option<String>, ctx: &Context) -> Result<(), HarnessError>;
    pub async fn abort(&self, options: AbortOptions, ctx: &Context) -> Result<(), HarnessError>;   // { background: bool }
    pub async fn wait_for_idle(&self, ctx: &Context) -> Result<(), HarnessError>;
    pub async fn view(&self, ctx: &Context) -> ..;      // observe.md: State<ConversationView> and watch
}

pub struct Submission { /* Arc<HarnessInner>, id */ }
impl Submission {
    pub fn id(&self) -> SubmissionId;
    pub async fn status(&self, ctx: &Context) -> Result<SubmissionRecord, HarnessError>;
    pub async fn wait(&self, ctx: &Context) -> Result<SettledSubmissionRecord, HarnessError>;
    pub async fn abort(&self, ctx: &Context) -> Result<SubmissionAbort, HarnessError>;   // Aborted | AlreadyPlaced | Settled
}
```

- A `Conversation::commit` is a session commit bound to the conversation: `tx.create_task`
  defaults to it; `entries` binds the query to it and paginates fork-aware history.
- `compact` admits a manual, conversation-owned, non-background `pi.compaction` task and
  returns its id; it does not take run control ([compaction.md](compaction.md)).
- `reset` admits a write submission of a `pi.reset` entry with `head: "self"` and the
  handoff as a user message; placement follows [inbox.md](inbox.md) and may happen later.
- `abort` withdraws queued inputs, marks the tasks ordinary traversal selects, signals them,
  and returns once the scope is ordinarily idle; `background: true` crosses boundaries.
- `submit` returns after durable admission; `Submission::wait` after settlement. Cancelling
  a wait's context cancels only the wait.

## Settings

```rust
pub struct HarnessSettings {                       // every field optional; read live, never stored
    pub extensions: Option<Vec<ExtensionName>>,   // default selection; None: every installed, install order
    pub stream: Option<ConversationStreamOptions>,
    pub retry: Option<PartialRetryPolicy>,
    pub compaction: Option<CompactionSettings>,   // Partial<CompactionPolicy> + model_overrides
    pub progress: Option<PartialProgressPolicy>,
    pub tool_execution: Option<ToolExecutionMode>,
    pub steering_mode: Option<QueueMode>,
    pub follow_up_mode: Option<QueueMode>,
    pub context_retention: Option<Duration>,
    pub document_retention: Option<Duration>,
    pub after_failed_run: Option<AfterFailedRun>,
    pub restart_limit: Option<Option<u32>>,       // Some(None): never park
    pub close_grace: Option<Duration>,
}

pub struct Settings {                               // resolved: defaults filled, object fields merged key by key
    pub extensions: Option<Vec<ExtensionName>>,
    pub stream: ConversationStreamOptions,        // {}
    pub retry: RetryPolicy,                        // enabled, max_retries 3, base_delay 2 s, max_delay 60 s
    pub compaction: CompactionPolicy,              // enabled, reserve 16384, keep_recent 20000, background 32768; + model_overrides
    pub progress: ProgressPolicy,                  // partial_interval 100 ms, output_interval 100 ms
    pub tool_execution: ToolExecutionMode,         // Parallel
    pub steering_mode: QueueMode,                  // OneAtATime
    pub follow_up_mode: QueueMode,                 // OneAtATime
    pub context_retention: Duration,               // 10 min
    pub document_retention: Duration,              // 10 min
    pub after_failed_run: AfterFailedRun,          // Keep
    pub restart_limit: Option<u32>,                // Some(5)
    pub close_grace: Duration,                     // 30 s
}
```

`ConversationStreamOptions` is the curated subset of `delta_ai::StreamOptions` the spec
lists (transport, timeout, provider `max_retries`, `max_retry_delay`, headers, metadata,
`cache_retention`, `deferred`); it is JSON (it is pinned in the generation checkpoint).
Settings changes make no commit and emit nothing. `models` and `settings` are resolved at
each use (`SettingsSource::Dynamic` replaces upstream's getters), so a host follows user
preferences without a session write. Who resolves what, and when (spec §7.1):

| Reader | Resolves | When |
| --- | --- | --- |
| Generation `prepare` | Agent, stream options, compaction thresholds for the model | Once per request; model, options, offered tools fixed in the `request` checkpoint |
| Every task phase | Extension selection for hooks; `rt.agent()` | At most once per phase, at the first hook dispatch or `agent()` call, from the phase snapshot and committed `pi.agent` |
| Tool task | The implementation of a call | At `call` and `execute`, by name among the phase agent's `tools`; none: `tool_unavailable` |
| Generation, round start | Tool execution mode | Once per round |
| Generation, compaction | Retry policy | At each attempt's classification |
| Generation, overflow | Compaction policy for the request's model | At classification |
| Compaction `select` | Model, thinking level, stream options, thresholds | Once; pinned across retries |
| Boundaries and idle admission (on the line) | Queue modes | Per boundary |
| Tool task, `prepare`, `rt.env()` | Environment | At each call |

## The agent document and resolution

```rust
pub struct AgentState {                            // stored `pi.agent`; names, never code
    pub model: Option<ModelRef>,                   // { provider, model_id }
    pub thinking_level: Option<ModelThinkingLevel>,
    pub extensions: Option<ExtensionSelection>,    // Exactly(Vec<String>) | Edit { add, remove }
    pub tools: Option<ToolFilter>,                 // Exactly(Vec<String>) | Remove(Vec<String>)
    pub model_tools: Option<ModelToolFilter>,      // Exactly(Vec<String>) | Edit { remove, add }
    pub instructions: Option<String>,
    pub cwd: Option<String>,
}

pub enum Change<T> { Keep, Clear, Set(T) }        // upstream undefined | null | value
pub struct AgentChange {
    pub model: Change<ModelRef>, pub thinking_level: Change<ModelThinkingLevel>,
    pub extensions: Change<ExtensionSelection>, pub tools: Change<ToolFilter>,
    pub model_tools: Change<ModelToolFilter>, pub instructions: Change<String>, pub cwd: Change<String>,
}
pub async fn configure(tx: &mut Tx, conversation: ConversationId, change: &AgentChange) -> Result<(), TxError>;

pub struct Agent {                                 // resolved; `Arc<Agent>` per phase
    pub model: Option<ModelRef>, pub thinking_level: ModelThinkingLevel,
    pub extensions: Vec<Arc<Extension>>,
    pub tools: Vec<Arc<dyn HarnessTool>>,          // offered to the model
    pub callable: Vec<Arc<dyn HarnessTool>>,       // nested calls resolve among these
    pub deferred: Vec<Arc<dyn HarnessTool>>,
    pub sections: Vec<Arc<dyn PromptSection>>,     // extension sections, then `instructions`
    pub instructions: Option<String>, pub cwd: Option<String>,
}
```

`configure` replaces whole fields (`Set`), clears (`Clear`) or leaves (`Keep`); extension and
tool objects given by a host stand for their names. `pi.agent` is rewindable with fork
policy `as_of`, complete base on every change, `initial = {}`. Nothing checks that a stored
name is installed: resolution skips unknown names and the name takes effect when installed
again. A change neither starts generation nor appends a system entry; the next preparation
does ([context.md](context.md)). `pi.provider` holds the provider-facing `session_id`
(UUIDv7, never copied by a fork, created on the line for a legacy conversation before its
first request).

Resolution (`resolve_agent(state, snapshot, settings, report) -> Agent`):

```text
extensions  base = settings.extensions or every installed, in order
            Exactly: exactly these in array order; Edit: base + add appended − remove
            duplicates keep their first position; uninstalled names skipped
tools       selected extensions' tools in extension order; a later same-name tool replaces in place
            then every selected extension's tool wrappers (extension order, then declaration order)
            then the `tools` filter → enabled: Exactly keeps these names in its order, Remove drops
            tools:    enabled with callers ∋ model, through `model_tools` (Exactly: these, deferred too;
                      Edit: all non-deferred minus remove, then deferred `add` not in remove)
            callable: enabled with callers ∋ tools;  deferred: model-callable deferred tools not offered
sections    selected extensions' sections, same-key replacement in place, then section wrappers,
            then `instructions` (tagged, key reserved, no wrappers)
hooks       selected extensions' hooks for the task name, in extension order
model       stored or none (request fails no_model);  thinking: stored or Off;  cwd: stored or none
```

A wrapper that fails or returns a different name drops its target for that resolution
(reported `extension_error`; calls produce `tool_unavailable`); one that finds no target
does nothing.

## Registry and extensions

```rust
pub struct Extension {
    pub name: String,
    pub tools: Vec<Arc<dyn HarnessTool>>,          // tool.md
    pub sections: Vec<Arc<dyn PromptSection>>,
    pub hooks: Vec<HookRegistration>,              // (task name, Arc<dyn Any + Send + Sync> hook object)
    pub wraps: Vec<Wrap>,                          // Tool(name, Arc<dyn Fn(Arc<dyn HarnessTool>) -> Arc<dyn HarnessTool>>) | Section(key, ..)
    pub tasks: Vec<Arc<dyn AnyTask>>,
}

pub trait PromptSection: Send + Sync {
    fn key(&self) -> &str;
    fn tag(&self) -> bool { true }
    async fn render(&self, input: &PromptInput<'_>, ctx: &Context) -> Result<Option<String>, SectionError>;
}
pub struct PromptInput<'a> { pub conversation: ConversationId, pub agent: &'a Agent, pub env: Option<&'a dyn ExecutionEnv>,
                             pub shown: &'a IndexMap<String, String>, pub read: &'a dyn DocumentReader }

#[derive(Clone)]
pub struct Registry { /* Arc<RwLock<Inner>>, tokio::sync::watch<u64> generation */ }
impl Registry {
    pub fn new() -> Registry;                                       // holds the built-in tasks
    pub fn install(&self, extension: Extension) -> Result<(), RegistryError>;   // replace in place by name; validates first
    pub fn uninstall(&self, name: &str);
    pub fn snapshot(&self) -> RegistrySnapshot;                    // Arc, immutable, with `generation`
    pub fn subscribe(&self) -> watch::Receiver<u64>;               // the scheduler reconsiders blocked tasks on change
}
impl RegistrySnapshot {
    pub fn generation(&self) -> u64;
    pub fn installed(&self) -> &[Arc<Extension>];
    pub fn extension(&self, name: &str) -> Option<&Arc<Extension>>;
    pub fn tools(&self) -> impl Iterator<Item = (&Arc<Extension>, &Arc<dyn HarnessTool>)>;   // install order; names may repeat
    pub fn sections(&self) -> ..;
    pub fn tasks(&self) -> &[Arc<dyn AnyTask>];                    // built-ins first
    pub fn task(&self, name: &str) -> Option<&Arc<dyn AnyTask>>;
}
```

Rules (§7.1, §7.5):

- The built-in tasks are always present and not an extension; nothing replaces them.
- Tool names and section keys are unique within an extension; across extensions they may
  repeat. Keys match `^[a-z][a-z0-9_-]*$`; `instructions` is reserved. Task names are unique
  across built-ins and every installed extension. `install` validates the registry as it
  would be after the replacement and publishes nothing when invalid.
- Publication is synchronous (a write-locked swap and a `watch` bump), invokes no extension
  code, and is per extension: reinstalling replaces tools, sections, hooks, wrappers and
  tasks in one publication at the same install position.
- Extensions, tools and definitions are immutable (`Arc`); wrappers are pure.
- A snapshot is an immutable value; the scheduler takes one per phase invocation and hands
  it down; handlers and hooks never take their own. Work already running keeps its snapshot
  until the phase settles; a task definition replaced by name hands over at the next phase
  boundary ([scheduler.md](scheduler.md)); a pinned tool keeps its implementation until the
  execution settles. The registry tracks no use; an extension that frees resources at
  uninstall makes still-running calls fail with an ordinary error.
- Application metadata on tools (a prompt snippet, say) is reached through
  `HarnessTool::extra() -> Option<&dyn Any>` and downcast by the application's own sections;
  `delta_ai::Tool` declarations derived from a tool carry only `delta_ai` fields.

## Hooks (§7.2)

One trait per built-in task's hook set, every method with a default that does nothing, as
`delta_agent::AgentHooks`; a registration is `hook::<GenerationHooks>(Arc<dyn GenerationHooks>)`
stored as `(task name, Arc<dyn Any>)` and downcast at dispatch by the task's `Hooks` type.

```rust
pub trait GenerationHooks: Send + Sync {
    fn before_request<'a>(&'a self, request: &'a RequestMessages, api: &'a HookApi, ctx: &'a Context) -> BoxFuture<'a, HookResult<Vec<Message>>> { none() }
    fn after_response<'a>(&'a self, message: &'a AssistantMessage, api: &'a HookApi, ctx: &'a Context) -> BoxFuture<'a, HookResult<()>> { none() }
    fn on_yield<'a>(&'a self, answer: &'a AssistantMessage, api: &'a HookApi, ctx: &'a Context) -> BoxFuture<'a, HookResult<UserContent>> { none() }
    fn after_tools<'a>(&'a self, assistant: EntryId, results: &'a [EntryId], api: &'a HookApi, ctx: &'a Context) -> BoxFuture<'a, HookResult<()>> { none() }
}
pub trait ToolHooks: Send + Sync {
    fn before_tool<'a>(&'a self, call: &'a ToolHookCall, api: &'a HookApi, ctx: &'a Context) -> BoxFuture<'a, HookResult<BeforeTool>> { none() }   // { arguments?, block? }
    fn after_tool<'a>(&'a self, call: &'a ToolHookCall, result: &'a ToolExecutionResult, api: &'a HookApi, ctx: &'a Context) -> BoxFuture<'a, HookResult<ToolExecutionResult>> { none() }
}
pub trait CompactionHooks: Send + Sync {
    fn before_compact<'a>(&'a self, compaction: &'a BeforeCompact, api: &'a HookApi, ctx: &'a Context) -> BoxFuture<'a, HookResult<CompactDecision>> { none() }   // Decline | Summary(String)
}
pub type HookResult<T> = Result<Option<T>, HookError>;

pub struct HookApi<'a> { pub task_id: TaskId, pub conversation_id: ConversationId, /* models(), memo(), snapshot(), snapshot_as_of() */ }
pub struct HookRunner<H: ?Sized> { /* the phase's resolved hooks, extension order */ }
impl<H: ?Sized> HookRunner<H> {
    /// Calls `invoke` with every handler; an `Err` is reported and the next handler runs; a
    /// cancelled invocation context propagates.
    pub async fn each<F>(&self, invoke: impl FnMut(&H, &str /* extension */) -> F) -> Result<(), Cancelled> where F: Future<Output = Result<(), HookError>>;
}
```

| Hook | Composition | Ordinary error | Claude Code hook | Codex hook |
| --- | --- | --- | --- | --- |
| `before_request` | Replacement chain on the messages, this request only | Report, continue | `UserPromptSubmit` (first attempt of a run; later attempts have no equivalent) | `UserPromptSubmit` |
| `after_response` | All observers | Report, continue | — (observer only) | — |
| `on_yield` | First continuation wins | Report, continue | `Stop` (a block with reason continues the run) | `Stop` |
| `after_tools` | All observers | Report, continue | — | — |
| `before_tool` | Argument replacement chain; first block wins | Block the tool with the error text | `PreToolUse` (`deny` → block, `updatedInput` → arguments) | `PreToolUse` |
| `after_tool` | Result replacement chain | Report, continue | `PostToolUse` / `PostToolUseFailure` | `PostToolUse` |
| `before_compact` | First decision wins | Report, continue | `PreCompact` | — |

Dispatch matches the task's *name*, so hooks survive a reload of their task; keeping
signatures compatible across versions is the task author's job. Hooks run in extension
order off the line; a crash before the consuming commit may rerun them; they use memos
(prefixed by hook authors) for first-writer-wins decisions. A command / HTTP hook bridge
later maps the column names onto these methods mechanically; it is out of phase 1
([README.md](README.md#later)).

## Environment

`EnvBuilder::build(&self, target: EnvTarget { conversation_id, cwd, read: &dyn DocumentReader }, ctx)
-> Result<Option<Arc<dyn ExecutionEnv>>, EnvError>` is called at each use, never on the
line: by the tool task per execution (an error becomes an error result), by generation
`prepare` once for the section renderers (an error is reported; sections render with no
env), and by `rt.env()`. Without a builder there is no environment. The host caches what is
expensive (one env per directory).

## Reports

`on_report(error, info)` receives failures that do not fail the calling operation and, once,
the session's failure: `session_failed`, `task_faulted { reason: Threw | RestartLimit }`,
`task_parked`, `migration_failed`, `task_outdated { reason }`, `task_error`,
`extension_error { extension, task?, conversation? }`, `host_error { callback }`,
`listener_error`. `ReportInfo` keeps upstream's shape; `HostCallback` loses `setTimer`.

## Testing

Ports of `harness-conversations`, `harness-registry`, `harness-host-inputs`,
`harness-callers`, `harness-lifecycle`, `harness-inspect`, `harness-reports`,
`harness-events` (the host-facing half), `provider-session-cache-e2e` (recorded streams)
and the `examples/` under `test/`. Registry tests cover validation, replacement in place,
snapshot immutability, and handover after reinstall.

## Differences from upstream

- `AgentChange` fields are `Change<T>` (`Keep | Clear | Set`), since Rust cannot carry
  `undefined` vs `null` in one `Option`.
- The registry is not generic over a tool type; application metadata is reached through
  `HarnessTool::extra()` and downcast. `RegistryReader::subscribe` is a `watch` channel.
- Hooks are trait objects with default methods and a typed `HookRunner::each`, not
  `Partial<HooksOf<K>>` records; registrations downcast at dispatch.
- `settings` and `models` are enums (`Fixed` / `Dynamic`) instead of value-or-function
  unions; `HarnessSettings` has no getters, so a live-updating host passes a closure.
- No `setTimer`; `Clock` replaces `now`; `close_grace` is a new setting
  ([scheduler.md](scheduler.md)).
- The surface returns `Result`s with typed errors (`HarnessError`, `SubmitError` with
  `ConversationBusy`, `Closed`, `NotFound`) instead of thrown `Error`s.

## Pending confirmation

- `Change<T>` for `AgentChange`, and keeping `AgentState`'s JSON exactly as upstream (wire
  compatibility of `pi.agent`). Position: adopt both.
- Tool metadata through `extra() -> Option<&dyn Any>` rather than a generic registry.
  Position: adopt; a generic parameter would spread through every task and hook type.
- The Codex column of the hook table is a mapping onto Codex's hook event names for the
  events it supports; the bridge is deferred, so the column is informative only.
