# delta-harness (`delta_harness`) design notes

`delta-harness` is delta's counterpart of
[`@earendil-works/pi-durable`](https://github.com/earendil-works/pi/tree/main/packages/durable):
a durable agent harness. Conversations, model turns, tool calls and application state are
committed to storage before anything is shown; a reopened storage resumes a turn where it
stopped. Code imports it as `use delta_harness::...`.

Upstream's specification (`packages/durable/docs/spec.md` at pi `73f8de8`) is normative for
the behaviour: each note names the spec sections it follows. The API is delta's own where
Rust offers a better fit; each note lists its differences from upstream. The survey that
scoped this crate is [docs/research/harnesses.md](../../../../docs/research/harnesses.md).

## Document index

| Document | Module | Contents |
| --- | --- | --- |
| [records.md](records.md) | `records` | IDs, conversation / entry / task / submission / document records, task states and outcomes, storage writes, errors, reports |
| [documents.md](documents.md) | `documents` | Document definitions and families, scope, history, fork policy, checkpoints, migration, the built-in documents |
| [storage.md](storage.md) | `storage` | The `Storage` trait, memory and SQLite backends, schema, scans, the fatal-error model, the conformance suite |
| [session.md](session.md) | `session` | The commit actor, `Tx`, drafts over chord, base/delta planning, document cache, forks, snapshots and subscriptions |
| [scheduler.md](scheduler.md) | `scheduler` | Reservation, the step rules, invocations, ownership and abort cascades, waiting / sleeping / completing, restarts and parking |
| [harness.md](harness.md) | `harness`, `registry` | `Harness::open`, conversations, agent resolution, settings, extensions, `inspect` |
| [inbox.md](inbox.md) | `inbox` | Submissions, request ids, steer / follow-up / write placement, turn boundaries |
| [context.md](context.md) | `context`, `prompt` | Model context from entries, head markers and edits, positional system sections, prompt-cache stability |
| [generation.md](generation.md) | `generation` | The `delta.generation` task: prepare, request, retry, poll, tool rounds, handover, usage |
| [tool.md](tool.md) | `tool` | The `delta.tool` task: validation, hooks, approval, execution with output windows, results, nested calls, replay safety |
| [approval.md](approval.md) | `approval` | The approval seam: policy, request and decision records, timeout, later rules and modes |
| [compaction.md](compaction.md) | `compaction` | The `delta.compaction` task: thresholds, range selection, overflow retry, stale summaries |
| [observe.md](observe.md) | `observe` | Conversation view, agent events derived from ops, task graph |
| [env.md](env.md) | `env` | `ExecutionEnv`: file system and shell, the local implementation |
| [tools.md](tools.md) | `tools` | `read`, `write`, `edit`, `bash`; the file mutation queue; output limits |

## How the pieces fit

```
  Harness ── conversations ── submit() ──► inbox ──► run (delta.live.run)
     │                                                  │
  Registry (extensions: tools, sections, hooks, tasks)  │
     │                                                  ▼
  Scheduler ── reserves tasks ──► delta.generation ──► model (delta_ai)
     │                               │  owns
     │                               ▼
     │                           delta.tool ×n ──► approval ──► env / tool body
     │                                                  │
  Session (one commit line) ◄── every checkpoint, entry, document change ──┘
     │
  Storage (memory | sqlite)        Observe: view, agent events, task graph
```

- A **submission** is admitted in one commit: the `delta.user` entry, the run marker in
  `delta.live`, and the first `delta.generation` task.
- The **scheduler** reserves runnable tasks in commits on the session line and runs their
  phase handlers off it. Every phase ends by committing the next state; a reopened storage
  reruns the current phase from its checkpoint.
- A **generation** streams the model's answer into `delta.live` on a throttle, commits the
  assistant entry, and either ends the run or creates the round's `delta.tool` tasks, waits
  for them, and hands the run to the next generation.
- Everything observable is a committed revision: documents are `delta_chord::State`s whose
  subscriptions deliver values with ops; the conversation view and agent events derive
  from those ops.

## Crate layout

```
crates/harness/
  docs/design/
  src/
    lib.rs
    records/         # ids, one file per record kind, writes, queries, errors
    documents/       # definitions, addresses, validation
    storage/         # trait, shared commit/scan/ancestry/replay rules, memory, sqlite
    session/         # line (actor), tx, drafts, plan, cache, observation, forks, snapshot
    scheduler/       # reservation, invocations, ownership
    harness.rs       # public surface, settings
    registry.rs      # extensions
    inbox.rs
    context.rs       # model context derivation
    prompt.rs        # sections, delta.system planning
    generation.rs
    tool.rs
    approval.rs
    compaction.rs
    observe/         # view, events, task_graph
    env/             # trait, local
    tools/           # read, write, edit, bash
  tests/             # ported upstream tests; storage and env conformance
```

Dependencies: `delta-chord` (with `sqlite`), `delta-ai`, `serde`, `serde_json`, `indexmap`,
`tokio`, `tokio-util`, `futures`, `thiserror`; `jsonschema` and `schemars` (tool argument
validation and typed tools); `similar` for `edit` when that phase lands. The crate does not
depend on `delta-agent`: upstream durable does not use agent-core either; it is a rewrite
of that loop whose steps are durable.

## Invariants

From spec §1, kept as stated:

1. One session commit is atomic across every record and document write.
2. A document revision is published only after its storage commit succeeded. All visible
   progress is durable; there is no volatile publication path.
3. External effects never run inside the commit (the effect sandwich: commit intent,
   perform, commit outcome).
4. Entries and ids are immutable and never reused.
5. Drafts are settled when the commit closure returns.
6. The commit line stays held through storage settlement and adoption; observers run on it
   only to capture immutable state, subscribers run off it.
7. A storage error is fatal to the open session: it publishes nothing and must be reopened.

## Architecture and seams

One crate, as upstream's package: decoupling comes from module layers and trait seams, not
from crate boundaries. Each layer depends only on the layers below it.

```
surface   harness (Harness, Conversation, Submission, Settings), observe (view, events, task graph)
agent     inbox, context, prompt, generation, tool, approval, compaction, usage,
          registry (Extension and the hook traits)
kernel    records, documents, storage (trait; memory, sqlite), session (actor, Tx),
          scheduler (the TaskDefinition runtime)
side      env (ExecutionEnv trait; local), tools (read, write, edit, bash)
```

New features plug in through these traits; none needs a kernel change:

| Trait | For |
| --- | --- |
| `Storage` | Another backend (JSONL later) |
| `TaskDefinition` | New durable work. The built-in tasks register through the same registry user tasks use; the scheduler has no special cases |
| `DocDefinition` | New durable state |
| `HarnessTool` | New tools; the built-in tools are ordinary implementations |
| `ApprovalPolicy` | Rules engine and permission modes later |
| `GenerationHooks`, `ToolHooks`, `CompactionHooks` | Extension behaviour; a command-hook bridge later |
| `ModelClient` | Model access; wraps `delta_ai::Models`, tests inject a fake |
| `ExecutionEnv` | Remote environments later |

Rules:

- The kernel modules (`records`, `documents`, `storage`, `session`, `scheduler`) never name
  an agent, surface or side module. `tests/layering.rs` scans their sources and fails on a
  `crate::` path into an upper layer.
- Built-in tasks, documents and tools use only the public traits, so an extension can do
  anything a built-in can.
- No module-level mutable state. The clock, models, environment builder and approval policy
  come through `HarnessOptions`.
- Each module has its own error enum; errors convert only at layer boundaries.

## Design decisions

| Question | Decision |
| --- | --- |
| Kernel | pi-durable's model (storage-first commits, task state machines, documents, inbox, ownership tree), not pi-coding-agent's in-memory loop |
| Compatibility with pi | None for the harness's own records and storage: kinds, record JSON and the SQLite schema are delta's. Messages inside entries are `delta_ai` types, which keep upstream's shape |
| Namespace | Built-in entry, document and task kinds start with `delta.` (`delta.user`, `delta.assistant`, `delta.tool-result`, `delta.system`, `delta.reset`, `delta.compaction`, `delta.approval`; documents `delta.agent`, `delta.provider`, `delta.live`, `delta.inbox`, `delta.usage`, `delta.approvals`; tasks `delta.generation`, `delta.tool`, `delta.compaction`, each at version 1) |
| Phase 1 scope | Kernel, built-in tools and a local execution env. Deferred: code mode, remote env, JSONL storage, telemetry, Durable Object hosting, images |
| The session line | One tokio task owns the session state. `commit` sends a closure that receives `&mut Tx`; a document that is not loaded is read through `tx`, so a read inside a commit cannot hang |
| Documents | `tx.doc::<D>()` borrows the `Tx` and returns a draft over `delta_chord::delta::Change` (one document edited at a time); prepared ops become `base` or `delta` writes under `checkpoint_when`; observers get a read-only `DocumentState` over a `delta_chord::State` |
| No-op batches and overflow | chord's rules: a change that ends equal to its base writes and publishes nothing; a subscriber that falls behind gets merged batches (`r` past `MAX_OPS`) |
| Tasks | `TaskDefinition` trait (`name`, `version`, `initial`, `run`, `abort`, `migrate`); built-in checkpoints are serde-tagged enums, user tasks may use `Json`. A phase returns `Next`, a closure run on the session line, so its last writes, its next state and the step decision are one commit. A repeated checkpoint is detected by `Json` equality |
| `close()` | Cancel every invocation, join them, abort stragglers after `close_grace` (30 s), write them back as `pending` |
| Approval | codex-rs's flow made durable ([approval.md](approval.md)): the request is committed, the tool phase awaits an in-memory waiter keyed by approval id, `Conversation::decide` commits the decision and notifies the waiter; a rerun reads a committed decision first; `ReviewDecision { Approved, ApprovedForSession, Denied, TimedOut, Abort }`; session grants skip later prompts with the same approval key |
| Approval visibility | A `delta.approvals` conversation document in the view, an `approval_update` agent event, denials reported as `blocked` |
| Hooks | Per-task traits with default methods, as `delta_agent::AgentHooks`: `GenerationHooks` (`before_request`, `after_response`, `on_yield`, `after_tools`), `ToolHooks` (`before_tool`, `after_tool`), `CompactionHooks` (`before_compact`); [harness.md](harness.md) maps them to the Claude Code / Codex names |
| Ids | Newtypes over `u64` |
| `delta_ai` additions | `is_context_overflow`, `is_retryable_assistant_error`, `retry_delay`, `to_tool_declaration`, as upstream has them in pi-ai |
| SQLite types | `delta_chord::sqlite` re-exports `rusqlite` |
| Env `watch` and `list_files` | Optional methods that return `not_supported` until `notify` / `ignore` are dependencies |
| Cancellation | `delta_chord::context::Context` wherever upstream passes a chord `Context` |
| Tests | Never the network: fake models replay recorded stream events; storage in memory or in SQLite files under a temp dir |
| Package name | crates.io package `delta-harness`, library `delta_harness`; not published until phase 1 is complete |

## Later

Seams kept visible for work after phase 1, in the order the survey suggests:

- A rules engine and permission modes over the approval seam (deny > ask > allow, rules
  that approvals can amend, a per-turn policy record as Codex's).
- A command / HTTP hook bridge speaking the Claude Code hook protocol.
- A protocol crate for UIs (thread / turn / item lifecycle, approval requests) derived from
  agent events and submissions.
- Code mode on an embedded JavaScript VM with the journal and replay rules.
- JSONL storage; remote execution environments on pi-env's daemon protocol; telemetry
  once delta has a telemetry crate; sub-agent roles and limits.
