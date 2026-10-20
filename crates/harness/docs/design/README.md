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
| [generation.md](generation.md) | `generation` | The `pi.generation` task: prepare, request, retry, poll, tool rounds, handover, usage |
| [tool.md](tool.md) | `tool` | The `pi.tool` task: validation, hooks, approval, execution with output windows, results, nested calls, replay safety |
| [approval.md](approval.md) | `approval` | The approval seam: policy, request and decision records, timeout, later rules and modes |
| [compaction.md](compaction.md) | `compaction` | The `pi.compaction` task: thresholds, range selection, overflow retry, stale summaries |
| [observe.md](observe.md) | `observe` | Conversation view, agent events derived from ops, task graph |
| [env.md](env.md) | `env` | `ExecutionEnv`: file system and shell, the local implementation |
| [tools.md](tools.md) | `tools` | `read`, `write`, `edit`, `bash`; the file mutation queue; output limits |

## How the pieces fit

```
  Harness ── conversations ── submit() ──► inbox ──► run (pi.live.run)
     │                                                  │
  Registry (extensions: tools, sections, hooks, tasks)  │
     │                                                  ▼
  Scheduler ── reserves tasks ──► pi.generation ──► model (delta_ai)
     │                               │  owns
     │                               ▼
     │                           pi.tool ×n ──► approval ──► env / tool body
     │                                                  │
  Session (one commit line) ◄── every checkpoint, entry, document change ──┘
     │
  Storage (memory | sqlite)        Observe: view, agent events, task graph
```

- A **submission** is admitted in one commit: the `pi.user` entry, the run marker in
  `pi.live`, and the first `pi.generation` task.
- The **scheduler** reserves runnable tasks in commits on the session line and runs their
  phase handlers off it. Every phase ends by committing the next state; a reopened storage
  reruns the current phase from its checkpoint.
- A **generation** streams the model's answer into `pi.live` on a throttle, commits the
  assistant entry, and either ends the run or creates the round's `pi.tool` tasks, waits
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
    records.rs       # ids, records, states, outcomes, writes, errors
    documents.rs     # definitions, built-in documents
    storage/         # trait, memory, sqlite
    session/         # actor, tx, forks, observation
    scheduler/       # reservation, invocations, ownership
    harness.rs       # public surface, settings
    registry.rs      # extensions
    inbox.rs
    context.rs       # model context derivation
    prompt.rs        # sections, pi.system planning
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

## Confirmed design decisions

| Question | Decision |
| --- | --- |
| Kernel | pi-durable's model (storage-first commits, task state machines, documents, inbox, ownership tree), not pi-coding-agent's in-memory loop |
| Phase 1 scope | Kernel, built-in tools and a local execution env. Deferred: code mode, remote env, JSONL storage, telemetry, Durable Object hosting, images |
| Approval | A seam with a minimal implementation (see [approval.md](approval.md)); rules and permission modes later |
| Package name | crates.io package `delta-harness`, library `delta_harness`; not published until phase 1 is complete |

## Proposed design decisions

| Question | Proposal |
| --- | --- |
| The session line | One tokio task owns the session state. `commit` sends a closure that receives `&mut Tx`; a document that is not loaded is read through `tx`, so upstream's "a read inside a commit never resolves" footgun cannot be written |
| Documents | `tx.doc::<D>()` returns a draft handle from `delta_chord::delta::Change`; the prepared ops become `base` or `delta` writes under `checkpoint_when`; committed revisions publish through `delta_chord::State` subscriptions (pull-based, values with ops) |
| Tasks | A `TaskDefinition` trait: `name`, `version`, `run(phase)`, `abort`, `migrate`; built-in checkpoints are serde-tagged enums, user tasks work on `Json`. Hand-over after a reload is detected by a registry generation counter, not object identity |
| Ids | Newtypes over `u64`, serialized as JSON numbers; `Seq` likewise |
| Wire compatibility | Entry kinds, document kinds, record JSON shapes and the SQLite schema (v2, optional prefix) stay as upstream's, so a storage written by pi opens in delta |
| Hooks | One `Hooks` trait per extension with default methods (`before_request`, `after_response`, `on_yield`, `after_tools`, `before_tool`, `after_tool`, `before_compact`), as `delta_agent::AgentHooks`; each note maps its hooks onto the Claude Code / Codex names so a command-hook bridge later is mechanical |
| Cancellation | `delta_chord::context::Context` everywhere upstream passes a chord `Context` |
| Tests | Never the network: fake models replay recorded stream events; storage in memory or in SQLite files under a temp dir |

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
