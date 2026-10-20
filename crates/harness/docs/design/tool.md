# The tool task (`tool`)

Follows spec §7.3 (tools), §8.4 (upstream's `pi.tool` task, here `delta.tool`), the parts
of §8.5 a tool task sees, and §8.2 (the `delta.live.tools` and `nested_tools` slots). Corresponds to upstream
`src/harness/tool.ts`, `src/harness/define.ts` and `src/truncate.ts`. The approval seam
this note slots in between `call` and `execute` is [approval.md](approval.md).

## Problem

A tool call is an external effect. The harness must record what it is about to run before
it runs it, show its output while it runs, commit its result once, and on reopen know
whether the call may run again. Upstream's answer is one durable task per call with two
phases, the effect sandwich around `execute()`, a bounded output window committed on a
throttle, and a replay policy per tool. delta keeps that and adds one phase: a durable
approval request between validation and intent.

## `HarnessTool`

```rust
pub trait HarnessTool: Send + Sync {
    fn definition(&self) -> &delta_ai::Tool;                       // name, description, parameters (JSON Schema)
    fn replay(&self) -> ToolReplay { ToolReplay::Unsafe }
    fn execution_mode(&self) -> Option<ToolExecutionMode> { None } // None: the settings' `tool_execution`
    fn callers(&self) -> Callers { Callers::BOTH }                  // MODEL | TOOLS
    fn output_limits(&self) -> OutputLimits { DEFAULT_OUTPUT_LIMITS } // 50 KiB, 2000 lines, Retain::Head
    fn structured_output_schema(&self) -> Option<&Value> { None }
    /// Pure repair of commonly malformed arguments, before validation; an error makes them invalid.
    fn prepare_arguments(&self, arguments: JsonObject) -> Result<JsonObject, ToolError> { Ok(arguments) }
    /// The key a session grant matches (approval.md); `None`: never granted for the session.
    fn approval_key(&self, arguments: &JsonObject) -> Option<String> { Some(canonical_json(arguments)) }
    fn execute<'a>(&'a self, arguments: JsonObject, api: &'a ToolApi, ctx: &'a Context)
        -> BoxFuture<'a, Result<ToolExecutionResult, ToolError>>;
}

#[derive(Serialize, Deserialize)] #[serde(rename_all = "lowercase")]
pub enum ToolReplay { Safe, Unsafe }
pub type ToolError = Box<dyn std::error::Error + Send + Sync + 'static>;
```

The trait mirrors `delta_agent::AgentTool` (`definition`, `execution_mode`, `replay`,
`prepare_arguments`, boxed `execute`) so a tool written for one reads like one for the
other; the harness adds what durability needs (`callers`, `output_limits`, the structured
output schema) and what approval needs (`approval_key`, as Codex's serialized approval
key: `bash` returns its command) and hands the tool a `ToolApi` instead of `ToolContext`. `FnHarnessTool`
and `typed_harness_tool::<P>(name, description, |args: P, api, ctx| ...)` (schema from
`schemars`, as `delta_agent::typed_tool`) are the convenience forms; builder methods set
`replay`, `callers`, `output_limits` and `structured_output_schema`. Tools come with
extensions and are registered in the registry ([harness.md](harness.md)).

### Results and the three channels

```rust
pub struct ToolExecutionResult {
    pub output: Option<Vec<UserBlock>>,     // for the model; None: the retained output
    pub structured_output: Option<Value>,   // for programs (nested callers)
    pub is_error: bool,
    pub details: Option<Value>,             // for UIs; None: the last `details()` value
    pub diagnostics: Vec<ToolDiagnostic>,
    pub usage: Option<Usage>,               // the execution's own spend; goes to `delta.usage.tools`
    pub control: Option<ToolControl>,
}
pub struct ToolDiagnostic { pub severity: Severity /* Info | Warn | Error */, pub message: String, pub code: Option<String> }
pub struct ToolControl { pub add_tools: Vec<String>, pub terminate: bool, pub handoff: Option<String> }
```

- `Err(ToolError)` from `execute` is upstream's throw: a `tool_error` diagnostic with the
  message, `is_error`, and the task ends `failed` (which cancels work the call owns).
  A returned `is_error` result completes the task and keeps owned work.
- Diagnostics are a channel, never text in the data: truncation, spill paths, corrected
  paths, exit codes. Every diagnostic is model-visible; UI-only data goes in `details`.
- `control` applies to model-issued calls only; the generation's `tools` phase reads it
  ([generation.md](generation.md)).

### `ToolApi`

```rust
pub struct ToolApi { /* runtime handle, call, reported output, progress */ }

impl ToolApi {
    pub fn task_id(&self) -> TaskId;  pub fn conversation_id(&self) -> ConversationId;  pub fn call_id(&self) -> &str;
    pub fn registry(&self) -> &RegistrySnapshot;
    pub async fn agent(&self, ctx: &Context) -> Result<Agent, HarnessError>;
    pub fn models(&self) -> Models;                               // `HarnessOptions.models`, read at each access
    pub fn env(&self) -> Option<&Arc<dyn ExecutionEnv>>;          // built for this call; see env.md
    pub fn now(&self) -> u64;                                     // the harness clock, epoch ms
    pub fn output(&self, chunk: &[u8]);  pub fn output_str(&self, chunk: &str);
    pub fn output_skipped(&self, skip: OutputSkip);               // an omission the env reported (tail only)
    pub fn output_window(&self) -> Option<ShellOutputWindow>;     // Some for Retain::Tail
    pub fn retained_output(&self) -> RetainedOutput;              // { text, truncated }
    pub fn diagnostic(&self, diagnostic: ToolDiagnostic);
    pub async fn details(&self, value: Value, ctx: &Context) -> Result<(), HarnessError>;
    pub async fn commit<T>(&self, change: impl FnOnce(&mut Tx) -> T, ctx: &Context) -> Result<T, HarnessError>;
    pub async fn memo<T>(&self, name: &str, candidate: Option<T>, ctx: &Context) -> Result<Option<T>, HarnessError>;
    pub async fn create_task(..) / get_task(..) / wait_for_task(..) / conversation(..) / snapshot(..) / watch_doc(..);
    pub async fn execute_tool(&self, name: &str, args: JsonObject, options: NestedCallOptions, ctx: &Context)
        -> Result<NestedToolExecutionResult, HarnessError>;
}
```

- `output` and `diagnostic` are synchronous: they push into the invocation's
  `delta_chord::output::OutputBuffer` (built from `output_limits`) and `mark()` a
  `Progress`. After the invocation ends they panic, as upstream throws; a tool must not
  keep reporting from a detached task.
- `details(value)` replaces the running details whole and resolves after the commit that
  carries it (`Progress::mark_and_wait`); cancelling the wait leaves the value in place.
- `env()` is built right before `execute` through `HarnessOptions.env` with the
  conversation's id and `cwd`; an error from the builder is a `tool_error` result. A
  rerun after recovery builds it again, with the conversation's `cwd` at that time.

## Checkpoints and records

```rust
#[derive(Serialize, Deserialize)] #[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolTaskInput {
    Model  { assistant: EntryId, call_id: String },
    Nested { parent: TaskId, parent_call_id: String, key: String, call: ToolCall,
             #[serde(skip_serializing_if = "Option::is_none")] progress: Option<bool> },  // Some(false) only
}

#[derive(Serialize, Deserialize)] #[serde(tag = "phase", rename_all = "lowercase")]
pub enum ToolTaskCheckpoint {
    Call,
    /// Waiting for the decision on a committed approval request until `deadline` (approval.md).
    Approval { id: ApprovalId, arguments: JsonObject, replay: ToolReplay, deadline: i64 },
    /// Durable intent: the final arguments and the replay policy, recorded before execution.
    Execute  { arguments: JsonObject, replay: ToolReplay },
}

#[derive(Serialize, Deserialize)] #[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolTaskResult { Model { entry_id: EntryId, control: Option<ToolControl> }, Nested }
```

`delta.tool` is version 1 with three phases; there is nothing older to migrate. The
`kind` tags on input and result are present from the start (upstream added them in its
version 2).

## The pipeline (`call`)

One handler runs from resolution to settlement, so no phase boundary separates them:

1. **Read the call**: from the assistant entry (`Model`) or the input (`Nested`, which
   hooks see with `parent`). A missing call is a fault.
2. **Resolve** the name among the phase agent's `tools` (`Model`) or `callable` tools
   (`Nested`: the enabled tools whose `callers` include `TOOLS`). None:
   `tool_unavailable` result, no intent.
3. **Repair**: `prepare_arguments` on a copy; the stored call keeps what the model sent.
   An error is `invalid_arguments`.
4. **Validate** with `delta_ai::validation::validate_tool_arguments` (coercion included).
   Failure: `invalid_arguments`.
5. **`before_tool` chain** (hooks of the selected extensions, in extension order): each may
   replace the arguments or block with text; the first block wins; an ordinary error from
   a hook blocks with its message; a cancellation propagates. A block: `blocked` result
   with `Tool call blocked: <text>`, no intent.
6. **Validate again** the hooks' arguments; failure: `invalid_arguments`.
7. **Approval**: the `ApprovalPolicy` decides on the final arguments. `Allow` continues;
   `Deny` settles `blocked` (`Tool call blocked: approval denied: <reason>`); `Ask`
   continues when a session grant matches `approval_key`, otherwise commits the request
   into `delta.approvals` and `Running(Approval { id, arguments, replay, deadline })`. The
   `approval` phase then waits in the same invocation for the decision (a oneshot from
   `PendingApprovals`, or the committed one on a rerun) until `deadline`, and continues at
   step 8 on `Approved` / `ApprovedForSession`, settles `blocked` on `Denied` or timeout,
   and leaves `Abort` to the abort handler. See [approval.md](approval.md).
8. **Intent commit**: `Execute { arguments, replay: tool.replay() }`, status `running`;
   the slot becomes `running`; a nested slot gets the final arguments when they differ
   from the ones it was created with.
9. **Execute** with the resolved implementation, `env` built just before; `duration_ms`
   is measured around `execute` with a monotonic clock, also when it fails.
10. **Final result** (below), **`after_tool` chain** (a replacement chain; an error is
    reported and skipped), **settle** (below).

Steps 1–7 end without intent when they fail; their results have no `duration_ms`.

### `execute` (recovery only)

Reached only by reopen. The tool is resolved again as in `call`. When the stored policy
and the current declaration are both `Safe`, the slot's progress is cleared (a rerun
reports from scratch) and execution runs again with the stored arguments, without
`before_tool` and without asking approval again. Otherwise the task settles an
`interrupted` error result built from the slot's durable partial output, details and
diagnostics (`Tool <name> was interrupted and may have partially run`) and ends `failed`.
A tool that no longer resolves counts as `Unsafe`. A current `Unsafe` vetoes a stored
`Safe`; a current `Safe` never upgrades a stored `Unsafe`.

### Abort

The abort handler builds an error result from the slot and ends `aborted`: code `aborted`
(`Tool <name> was aborted`); after a restart that abandoned the task
(`abandon_on_restart`), `interrupted` when intent was recorded, else `abandoned`
(`Tool <name> was not started: its caller ended with a restart`).

## Output windows and progress

- The invocation owns one `OutputBuffer` with the tool's limits. `output()` pushes bytes
  (decoded as a stream) or text; `output_skipped` records an environment's omission so the
  dropped counts stay exact (tail only; it panics for `Head`, as chord's `skip`).
- A `delta_chord::output::Progress` paces commits: the first mark after idle commits at
  once; each commit delays the next by `settings.progress.output_interval_ms` (100 ms) and
  by its written size at 100 KiB/s. The write closure snapshots the buffer, details and
  the diagnostics count synchronously, then commits only what changed into the slot:
  `output` assigned as one string (chord diffs it into an `a`, or `t` + `a` for a sliding
  tail, or an `s` when its bounded overlap search finds nothing), `dropped_bytes`,
  `dropped_lines`, details diffed leaf by leaf, new diagnostics pushed. The written byte
  estimate uses `delta_chord::delta::overlap` on the previous and current window.
- Commits apply only while a slot with the task's id exists; otherwise they write nothing.
- With `Nested { progress: Some(false) }` nothing is published; the result still carries
  details and diagnostics.
- `output_window()` is `Some` for tail-retaining tools: `{ max_bytes, max_lines,
  min_interval_ms, bytes_per_second }`, which `bash` passes to `exec` so a remote
  environment may omit output outside the tail and report the omission.
- At settlement: `OutputBuffer::end()`, `Progress::stop()`; the terminal commit is the final
  flush and settles pending `details()` waiters (or rejects them when it fails). A commit
  error after an abort mark is expected and not reported.

## Final result and settlement

`final_result` builds what is committed:

1. `output`: the result's, or the retained window as one text item (none when empty).
2. `details`: the result's, or the last reported value. `diagnostics`: those recorded
   through `api`, then the result's.
3. `after_tool` chain runs over that (hooks may redact; what they return is what programs
   and the model get).
4. Harness diagnostics, appended last: `truncated` (warn, `Output truncated to its
   beginning|end: N lines, M bytes dropped`) when the retained window dropped output and
   the hooks kept it, and again when bounding explicit text content (below) drops text.
5. Explicit text content is bounded with `bound_output` over the joined text items: on
   overflow the text items collapse into one bounded item at the first (`Head`) or last
   (`Tail`) text item's position; non-text content is kept.
6. `structured_output` contract (validated with `jsonschema`): a schema tool must return a
   matching value except on an error result; a schema-less tool must not return one. A
   broken nested result becomes `is_error` with an `invalid_structured_output` diagnostic
   and no value; a model-issued one only loses the value and the break is reported
   through `on_report`. A schema-less nested result gets its bounded output collapsed as
   its `structured_output`: one text item as its string, one image as itself, none as
   `""`, else the list. A schema tool's nested error result without a value and without an
   error diagnostic turns its output text into a `tool_error` diagnostic.

The settle commit, for a model-issued call: render the diagnostics as one trailing text
item (`<harness>\n[severity] message\n...\n</harness>`), append the `delta.tool-result` entry
(`model: [ToolResultMessage { tool_call_id, tool_name, content, details, usage, is_error,
duration_ms, timestamp }]`, `data: { diagnostics }`, possibly empty), add `usage` to
`delta.usage.tools[name]`, mark the slot `done` with `entry` and without progress, remove the
nested slots below the task (transitively, only when the `delta.tool.made_nested_calls` memo
says it made any), and end with `Model { entry_id, control }` (`control` only on
`completed`). Outcome `completed` for a returned result (also `is_error`), `failed` for an
error from `execute` or `env`, or an interruption; `aborted` from the abort handler. A
nested call instead writes its result to the caller's `NestedCallDoc` member, records its
usage, sets its slot's `summary` (`is_error`, `duration_ms`, `usage`, up to 500 chars of
error text from its error diagnostics or else its output), and ends `Nested`.

Harness-written error results have `output: []`, `is_error`, and one `error` diagnostic
with code `tool_unavailable`, `invalid_arguments`, `blocked`, `interrupted`, `aborted`,
`abandoned` or `tool_error`; those built from a slot keep its partial output, details and
diagnostics and add `truncated` when the slot dropped output.

## Nested calls

`api.execute_tool(name, args, { key, progress })`:

- Keys: default the call's position among this invocation's calls (`1`, `2`, ...); an
  explicit key is non-empty, without `/`, not `__proto__`, not a positive integer. The
  nested call id is `<call_id>/<key>`.
- First the memo `delta.tool.made_nested_calls = true` (once per attempt), then one commit: open
  or create the `NestedCallDoc` member (`delta.tool.nested-call`, a task-scoped family of the
  caller keyed by the key, `{ task_id?, result? }`, retires with the caller); a member with
  a task already is reused when the name and arguments are structurally equal, otherwise
  the call fails; else create the `delta.tool` task (`Nested` input, owned by the caller,
  `abandon_on_restart` unless the caller is `Safe`), store its id, push a `pending` slot to
  `delta.live.nested_tools` with the arguments as passed.
- Wait for the task, then read the member's `result`; without one (faulted, orphaned) a
  synthesized `is_error` result with a `faulted` / `orphaned` diagnostic.
- `execute_tool` after `execute` returned panics; cancelling its context stops only the
  wait. Child tasks the tool creates through `create_task` with task ownership inherit
  `abandon_on_restart` the same way.
- Before settling, by any path: let admissions in flight commit, abort every live nested
  call the task owns (`owned_tasks` filtered to nested `delta.tool`), all at once, and wait,
  so they settle into their slots before the caller's commit removes them.
- `NestedToolExecutionResult { task_id, structured_output, is_error, details, diagnostics,
  usage, duration_ms }`: no `output`, no `control`.

Nothing in phase 1 calls `execute_tool` except tests; code mode and an MCP bridge are the
later callers (the seam is `callers` plus this method).

## Hooks

| delta `Hooks` method | Composition | Claude Code / Codex name |
| --- | --- | --- |
| `before_tool(call, api, ctx) -> Option<BeforeTool { arguments, block }>` | argument replacement chain; first block wins; an error blocks | `PreToolUse` (rewrite input, deny) |
| `ApprovalPolicy::decide` ([approval.md](approval.md)) | one policy | `PermissionRequest` |
| `after_tool(call, result, api, ctx) -> Option<ToolExecutionResult>` | result replacement chain; an error is reported | `PostToolUse` (replace output) |

Hooks run off the session line with the task's `HookApi` (committed reads, memos). They
are not rerun on recovery: the arguments they produced are in the checkpoint.

## Replay safety on recovery

- Intent is durable before any effect: a crash before the intent commit reruns `call`
  from the top (hooks and approval included; an approval already decided is reused).
- A crash after intent reaches `execute`; only a tool that is `Safe` both in storage and
  now reruns. A `Safe` tool must make its own effects idempotent from the stored arguments
  (random values, timestamps and mutable reads belong in memos, checkpoints or task
  inputs); it reattaches to its nested calls and owned conversations by key and by owner
  scan. Any other tool settles `interrupted` with the durable partial output.
- The model never sees a result twice: completed, failed and aborted outcomes all keep
  one `delta.tool-result` entry for the generation's `tools` phase.

## Differences from upstream

- An `approval` phase between `call` and `execute` that awaits a decision inside its
  invocation, codex-rs style, with the request and the deadline committed
  ([approval.md](approval.md)).
- Version 1 with snake_case JSON; upstream's version 1 → 2 input migration has no
  counterpart.
- A trait instead of an object literal; `callers` is a bitflag; `prepare_arguments` returns
  `Result`; `ToolApi` is a struct with methods, not a spreadable object.
- Output is bytes-native (`OutputBuffer::push_bytes`), skips are a separate call.
- The throttle uses tokio timers through chord's `Progress`, owned by the invocation and
  dropped with it (scheduler.md's `rt.timer`); the harness clock only stamps results.
- `typed_harness_tool` replaces `defineTool` type inference.
- Not ported: PowerShell variants, `wrapTool` as a separate helper (wrapping is a
  `HarnessTool` that holds an `Arc<dyn HarnessTool>`; the registry composes wrappers per
  resolution as [harness.md](harness.md) describes), images in `read` (tools.md).

## Decisions

- `delta.tool` is version 1, with the `approval` phase from the start.
- A denied or timed-out approval reuses code `blocked` with an `approval denied:` /
  `approval timed out` message; the structured decision is in the `delta.approval` entry.
- Hooks run before the approval policy; `ToolHooks` is the tool task's hook trait.

- Phase 4 defined the declaration side of `HarnessTool` (`definition`, `name`,
  `execution_mode`, `callers`, `deferred`, `extra`), `ToolExecutionMode`, `Callers`,
  `ToolExecutionResult`, `ToolDiagnostic`, `ToolControl`, `ToolTaskInput::Model`,
  `ToolTaskResult`, `harness_error`, `append_tool_result` and the generation's seam
  `create_model_call_task` (kind `delta.tool`, version 1, first checkpoint
  `{"phase":"call"}`). Phase 5 adds `execute` and the rest of the trait, the `Nested`
  input, `ToolApi` and the task itself, registered as a built-in beside the generation
  (`generation::builtin_tasks`). `deferred()` is new: resolution offers a deferred tool only once
  `model_tools` loads it (harness.md).

## Open (minor)

- `output()` panicking after the invocation ends (upstream throws). Alternative: ignore
  late output and report once.
- Nested calls land in phase 1 behind tests only; they could wait for code mode.
