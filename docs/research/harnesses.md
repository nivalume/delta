# Agent harnesses: Codex CLI, Claude Code, DeepSeek Harness, pi

Notes from a survey of four coding-agent harnesses, used to scope `delta-harness`. Read on
2026-10-20. Some OpenAI and GitHub pages refused direct reads; details taken from secondary
sources are marked.

A harness, for delta, is the layer between the agent loop (`delta-agent`) and a product
(a CLI, an IDE server, a hosted agent): persistent sessions, a turn loop whose progress
survives a crash, tools, approvals and sandboxing, context compaction, extensions and
sub-agents, and a protocol UIs consume.

## Codex CLI (`openai/codex`, `codex-rs`)

A Cargo workspace of about 100 crates. The engine crates are `core`, `protocol`, `rollout`,
`history`, `state`; the surfaces `app-server*`, `exec`, `cli`, `tui`; tools `apply-patch`,
`exec-server`, `shell-command`, `codex-mcp`, `rmcp-client`, `code-mode-*`; sandboxing
`sandboxing`, `linux-sandbox`, `bwrap`, `execpolicy`, `network-proxy`; extensions `hooks`,
`skills`, `plugin`, `agent-roles`, `memories`, `context-fragments`.

- **Loop and state.** The engine and its UI talk through two queues: submissions carry `Op`
  values (`TurnInput`, `Interrupt`, `ExecApproval`, `PatchApproval`, `Compact`, `Review`,
  `Shutdown`, ...), events carry `EventMsg` (about 90 variants: turn lifecycle, item
  started/completed, exec begin/output/end, approval requests, `ContextCompacted`,
  `TokenCount`, hook and sub-agent events). One task (turn) runs per session at a time.
  Requests are stateless: the full history is resent every call, and cost is controlled by
  keeping each request an exact prefix of the next so the prompt cache hits; reordering
  tools mid-session once broke that.
- **Persistence.** A session is a "rollout": JSONL under `~/.codex/sessions/`, each line a
  timestamped `RolloutItem`. `SessionMeta` records `forked_from_id`, `parent_thread_id`,
  `agent_role`, `base_instructions`. A `TurnContextItem` per turn records cwd, approval and
  sandbox policy, model and effort, so a resumed or forked turn runs under the configuration
  it was recorded with. A SQLite state database indexes sessions.
- **Tools, approval, sandbox.** `shell`/`unified_exec` (PTY sessions with `write_stdin`),
  `apply_patch` (a grammar-constrained freeform tool), MCP, "code mode", client-defined
  dynamic tools. Sandbox mode (`read-only`, `workspace-write`, `danger-full-access`) and
  approval policy (`AskForApproval::{UnlessTrusted, OnRequest, Granular, Never}`) are two
  independent settings; OS enforcement is Seatbelt on macOS, bubblewrap on Linux, a native
  Windows sandbox. A command that must leave the sandbox raises an approval whose decision
  can be `Approved`, `ApprovedForSession`, `ApprovedExecpolicyAmendment` (persist a rule),
  `NetworkPolicyAmendment`, `Denied`, `Abort`. A "guardian" reviewer agent can answer
  approvals instead of the user.
- **Compaction.** Automatic before and during a turn at `min(model limit, 90% of the
  window)`, or manual. OpenAI models use a remote `/responses/compact` that returns an
  opaque compaction item; other providers run a local summary prompt and keep recent user
  messages. Both paths reinsert the initial context. (Third-party write-up.)
- **Extensions.** Hooks deliberately mirror Claude Code's (`PreToolUse`, `PermissionRequest`,
  `PostToolUse`, `Pre/PostCompact`, `UserPromptSubmit`, `Stop`, `SubagentStart/Stop`,
  `SessionStart/End`), handlers `command` or `mcp_tool`, trusted by hash. Also skills,
  plugins, `execpolicy` rules, AGENTS.md.
- **Sub-agents.** Roles (`default`, `worker`, `explorer`, custom TOML) run as full threads
  with their own rollout and `parent_thread_id`; they inherit the parent's approval and
  sandbox policy and cannot override it; concurrency is capped.
- **UI protocol.** The App Server: bidirectional JSON-RPC over stdio, WebSocket or in-process,
  with Thread, Turn and Item primitives (`thread/start|resume|fork|list|read`, `turn/start`,
  `item/started` → deltas → `item/completed`, server-to-client approval requests).
- **Worth borrowing.** A typed protocol crate separate from the engine; a per-turn context
  record; approvals that can amend policy.

Sources: codex-rs tree, `protocol/src/protocol.rs`, `rollout/src/lib.rs`, `app-server/README.md`
(github.com/openai/codex); learn.chatgpt.com/docs/sandboxing, /docs/hooks,
/docs/agent-configuration/subagents; InfoQ "Unrolling the Codex agent loop" (2026-02).

## Claude Code and the Claude Agent SDK

Closed source; the SDK bundles the CLI binary as a subprocess. Public documentation is
extensive.

- **Loop and state.** Model output → run tools → feed results back, until a response has no
  tool calls; each round trip is a turn. The SDK streams `SystemMessage` (`init`,
  `compact_boundary`), `AssistantMessage`, `UserMessage` (tool results), `StreamEvent`,
  `ResultMessage` (subtype, cost, usage, session id). Read-only tools run concurrently,
  mutating ones one at a time. `maxTurns` and `maxBudgetUsd` bound a run; queued user
  messages are delivered after the current tool batch.
- **Persistence.** JSONL under `~/.claude/projects/<project>/`; entries form a uuid chain;
  forking rewrites the session id and remaps uuids. Files are snapshotted before each edit
  for rewind. Sub-agent transcripts live in `<session>/subagents/`. The SDK's `SessionStore`
  is two methods, `append(key, entries)` and `load(key)`, written local-first with a
  best-effort mirror.
- **Tools, approval, sandbox.** Permission modes (`default`, `acceptEdits`, `plan`, `auto`,
  `dontAsk`, `bypassPermissions`); rules `Tool(specifier)` evaluated deny, then ask, then
  allow, first match wins; `PreToolUse` hooks run before the prompt but cannot override deny
  or ask; an undecided call goes to the SDK's `canUseTool`, and with no callback is denied.
  The Bash sandbox is `@anthropic-ai/sandbox-runtime` (Seatbelt / bubblewrap plus a domain
  allowlist proxy); file tools, MCP servers and hooks run outside it.
- **Compaction.** Clear older tool outputs first, then summarize; a `compact_boundary` entry
  with `preTokens` is written; `/compact <focus>` and `Pre/PostCompact` hooks steer it; a
  thrash guard stops after a few consecutive failures; CLAUDE.md is re-injected every
  request. Sub-agents compact their own context.
- **Extensions.** About 30 hook events with `command`, `http`, `mcp_tool`, `prompt` and
  `agent` handlers; `PreToolUse` may allow, deny, ask or defer and rewrite the input,
  `PostToolUse` may replace the output. Skills (description in context, body on demand),
  MCP with deferred tool schemas, plugins, output styles.
- **Sub-agents.** Markdown definitions in `.claude/agents/` (tools, model, permission mode,
  hooks, skills, worktree isolation, background, max turns); built-in Explore, Plan,
  general-purpose; a "fork" sub-agent shares the parent's prefix; depth and concurrency are
  limited; a finished sub-agent can be resumed.
- **UI protocol.** The SDK's stream-JSON message protocol plus control requests for
  permissions and hooks; no public JSON-RPC specification comparable to Codex's.
- **Worth borrowing.** The two-method store with a conformance suite; deny-first rules that
  hooks cannot bypass, with a `defer` decision; fork sub-agents next to fresh-context ones.

Sources: code.claude.com/docs/en/how-claude-code-works, /permissions, /sandboxing, /hooks,
/sub-agents, /agent-sdk/agent-loop, /agent-sdk/session-storage.

## DeepSeek Harness (`deepseek-ai/deepseek-harness`, `dsh`)

Open-sourced 2026-08-13 with DeepSeek-V4-Pro, MIT, TypeScript on Node 22, developer
preview. Built on the Cordis plugin kernel: the model adapter, tool registry, session log and
the agent loop itself are plugins that can be replaced; unloading a plugin removes everything
it registered.

- **Loop and state.** Services on a shared context (`ctx.sessions`, `ctx.tools`,
  `ctx.agentLoop`, `ctx.llm`, `ctx.systemPrompt`). A step is one model request plus its tool
  calls; a turn is zero or more steps. Hook points (`turn/start`, `agent/pre-step`,
  `step/start`, `llm/stream`, `tools/pre-execute`, `tools/execute`, `tools/post-execute`,
  `step/end`, `agent/turn-stopping`) are waterfalls where each listener calls `next()`.
- **Persistence.** An append-only `SessionEvent` log is the source of model context
  (`deriveMessages()`); "anything model-visible must be logged". `assistant/attempt`
  records failed attempts outside model history. Files are `session.vN.jsonl[.zstd]`; old
  versions are never rewritten and each `vN → vN+1` migration is its own package. Fork
  happens at a turn boundary with a seed length.
- **Tools, approval, sandbox.** The pipeline logs `tool/call`, runs the pre-execute
  waterfall (hooks, permission, sandbox), then guards that may only deny or abstain, then
  execute (timeout, retry, metrics), the tool body, post-execute (accept, block, replace,
  add context), and logs one `tool/result`. `ctx.approval` fails closed: no client to
  answer means denied. Sandbox backends are swappable (`sandbox-local`, `sandbox-ssh`,
  `bash-*`, `fs-*`) under one policy root. Agent presets include a minimal one (persistent
  bash plus `str_replace_editor`, the setup of DeepSeek's benchmarks) and a code one (a
  single `run_code` tool over generated bindings).
- **Compaction.** `ctx.compaction` with a basic plugin; a third-party plugin lets the model
  decide and logs each decision as a replacement event.
- **Extensions.** Everything; also skills, MCP resources, slash commands, webhooks, jobs,
  profiles layered from bundles and patches; bridges for Claude Code and Codex hooks.
- **Sub-agents.** Providers: in-process spawn and fork, ACP, Codex and Claude Code out of
  process, the dsh SDK; experimental agent teams with a durable roster and inbox messaging.
- **UI protocol.** A JSON-RPC SDK server (TypeScript and Python SDKs), an Agent Client
  Protocol server, a web app with a trajectory replay view.
- **Worth borrowing.** Each capability as a small trait with interchangeable backends;
  immutable session-format versions with one migration unit per step; approval that fails
  closed, and guards that run after extensions and can only deny.

Sources: github.com/deepseek-ai/deepseek-harness `docs/architecture.md`,
`docs/tool-execution-pipeline.md`, `docs/capability-seams.md`, `docs/session-format-status.md`.

## pi (`earendil-works/pi`)

Two harnesses in one monorepo.

### pi-coding-agent

The shipped CLI (about 76k lines) on pi-agent-core.

- **Loop and state.** The agent-core loop (`delta-agent` is its counterpart): one turn is an
  assistant reply and its tool calls; steering joins after the current turn, follow-ups
  after the run. Sessions are JSONL trees: every entry has `id` and `parentId`, the current
  entry names the active branch, continuing from an earlier entry branches in place, fork
  and clone copy history into a new file. Model context is rebuilt from the active branch.
- **Entries.** `message` (with system messages that patch prompt sections and tool
  declarations by name), `model_change`, `thinking_level_change`, `usage`, `compaction`
  (summary, `firstKeptEntryId`, the system prompt checkpoint), `context_edit`, branch
  summaries. Format versions are migrated on load.
- **Compaction.** Threshold-triggered with cut-point rules (never split a tool round),
  overflow and length recovery, branch summarization, file tracking across compactions,
  per-model settings, custom summarization through extensions.
- **Extensions.** TypeScript modules loaded in-process (`jiti`) that register tools,
  commands, shortcuts, providers, event handlers (a hook sequence from `input` through
  `before_provider_request`, `tool_call`, `tool_result` to `turn_end`), renderers and UI.
  Skills, prompt templates, themes, packages via npm or git. Project trust gates project
  resources.
- **Interfaces.** Interactive TUI, print mode, JSON mode (agent events as JSONL), RPC mode
  (JSONL commands on stdin), a TypeScript SDK. No built-in sub-agents or plan mode: "ask Pi
  to build what you want, or install a package".

### pi-durable

Experimental (the API "changes without notice"), about 26k lines, on chord and pi-ai; not
used by coding-agent. It rewrites the loop so that every step is durable:

- **Rule.** A session atomically commits immutable entries, full task records and
  chord-tracked documents; only committed state is observable; effects never run inside
  the commit; a storage error is fatal to the session.
- **Records.** Conversations (forkable, ownable by a task), entries (`pi.user`,
  `pi.assistant`, `pi.tool-result`, `pi.system` with positional section deltas, `pi.reset`,
  `pi.compaction`, user kinds; `head` markers and `edits` shape the model context),
  submissions (input or write; queued → placed → done/unanswered; `requestId` dedupe),
  tasks, documents (typed JSON, scoped to session, conversation or task, stored as a base
  plus op deltas with a required `checkpointWhen`).
- **Tasks.** Durable state machines with an exhaustive phase map; the effect sandwich
  (commit intent, perform the effect, commit the outcome); `waiting` on other tasks,
  `sleeping` until a time, a `completing` hold until owned work drains; outcomes
  `completed`, `failed`, `aborted`, `orphaned`, `faulted`. Every task has an owner (its
  conversation or another task); abort cascades down the ownership tree and runs bottom-up;
  `background` tasks are boundaries. The scheduler rewrites surviving `running` tasks to
  `pending` on open with a restart count and parks them after a limit; a resumed task reruns
  its current phase from the stored checkpoint, so built-in phases are idempotent (a
  partial assistant message becomes an aborted entry and the request is resent; a tool
  reruns only when its `replay` is `safe`, otherwise the model sees `interrupted`).
- **Built-in tasks.** `pi.generation` (prepare → request → tools, with retry and deferred
  poll phases; streams partials into `pi.live` on a throttle; starts a tool round of owned
  `pi.tool` tasks and waits; hands the run to the next generation), `pi.tool` (validate,
  hooks, intent commit, execute with chord's output window and progress, result entry),
  `pi.compaction` (blocking at a threshold, background, manual, or after an overflow).
- **Inbox.** Steer, follow-up and write submissions placed at turn boundaries; display-only
  writes land at once; `whenBusy` and `afterFailedRun` policies.
- **Extensions.** Plain objects `{name, tools, sections, hooks, wraps, tasks}` installed in
  a registry; conversations store selections by name and resolve them at each phase, so
  reload is `install` with the same name.
- **Observation.** Documents are chord replicated states; a conversation view mounts the
  built-in documents; agent events (coding-agent compatible) are derived from the ops of
  each commit; a task graph view; content-free telemetry spans.
- **Hosting.** Memory, SQLite and JSONL storage; one process owns a storage; sleeping tasks
  and host wake-ups for Cloudflare Durable Objects; a Rust daemon in pi-env for remote
  execution over SSH.
- **Code mode.** A tool whose argument is JavaScript run in a deterministic QuickJS VM; every
  tool result is journaled before delivery so a rerun replays the journal.
- **Churn.** First commit 2026-09-18; seven releases in fifteen days; 71 commits in the two
  weeks before this note; six unreleased breaking changes (sleeping state, SQLite schema
  v2, required `checkpointWhen`, restart parking).

Sources: `packages/durable/docs/spec.md`, `packages/durable/README.md`,
`packages/coding-agent/docs/{how-pi-works,session-format,compaction,extensions}.md` at pi
`73f8de8`.

## Comparison

| | Codex CLI | Claude Code | dsh | pi coding-agent | pi-durable |
| --- | --- | --- | --- | --- | --- |
| Language | Rust | closed (TS) | TS on Cordis | TS | TS on chord |
| Loop unit | session → turn, `Op`/`EventMsg` | turn = tool round trip | turn → steps | turn, steering queues | run → generation and tool tasks |
| Source of truth | rollout JSONL + SQLite index | session JSONL chain | versioned event JSONL | JSONL entry tree | commits: entries, tasks, documents |
| Resume | `thread/resume`, per-turn context | `--resume`, store adapter | replay | reload branch | rerun the current task phase |
| Fork | `thread/fork` | `--fork-session`, `/branch` | seed at turn boundary | in-file branches, clone | `fork(at)` with per-document policy |
| Sandbox | Seatbelt / bwrap / Windows, modes | sandbox-runtime + proxy | swappable backends | none built in | host-provided env |
| Approval | policy + execpolicy + guardian | modes + deny>ask>allow + `canUseTool` | fails closed | extension hooks | extension hooks |
| Compaction | ≤90%, remote or local summary | clear outputs then summarize, boundary entry | plugin | cut rules, branch summaries | task at threshold, background or overflow |
| Hooks | Claude Code vocabulary | ~30 events, 5 handler types | every event | event handlers in process | typed hooks per extension |
| Sub-agents | threads with roles, inherited policy | Agent tool, definitions, forks | provider seam, teams | none built in | task-owned conversations |
| UI protocol | App Server JSON-RPC | stream-JSON | JSON-RPC, ACP | RPC JSONL, SDK | events + submit API |

## Table stakes

Ideas present in two or more harnesses:

1. An append-only log is the source of truth; model context is derived from it; resume,
   fork, search and replay operate on it.
2. Fork as a first-class operation with parent and fork-point metadata.
3. Tool call and tool result as separate records, with a policy pipeline between them
   (pre-check → approval → sandboxed execution → post-check).
4. Sandboxing separate from approval; approval is the escalation path out of the sandbox.
5. Deny wins; fail closed when no one can answer.
6. One hook vocabulary (`PreToolUse`, `PostToolUse`, `Stop`, `SessionStart`, `PreCompact`,
   `SubagentStart/Stop`, JSON on stdin/stdout, exit code 2 blocks): Codex copied Claude
   Code's, dsh bridges both.
7. Compaction is logged as a boundary or replacement entry, can be triggered manually with
   a focus, and has hooks before and after.
8. Sub-agents get their own transcript and context, inherit the parent's policy, and are
   limited in depth and concurrency.
9. A versioned protocol UIs consume, separate from the engine, with an item lifecycle of
   started → deltas → completed.
10. Steering: queued input joins after the current tool batch.
11. Lazily loaded capabilities: skills show only a description until used; MCP tool schemas
    are deferred.
12. A stable prompt prefix for caching: no reordering of tools or rewriting of the system
    prompt mid-session (pi-durable's positional `pi.system` deltas exist for this).

Rarer ideas worth keeping in view: storage-first admission and per-tool replay safety
(pi-durable); a per-turn context record and approvals that write policy (Codex); immutable
session-format versions and capability traits (dsh); a two-method store with a conformance
suite (Claude Code).

## Decisions for delta

Confirmed with the maintainer:

1. The crate is `crates/harness` (`delta-harness`). Its kernel is pi-durable's model:
   storage-first commits, task state machines with checkpoints, documents, inbox, the
   ownership tree. Crash resume is the capability the other harnesses lack, and
   `delta-chord` was built for it.
2. Phase 1 covers the kernel plus the built-in tools (`read`, `write`, `edit`, `bash`) and a
   local execution environment, so an end-to-end coding agent runs. Not in phase 1: code
   mode, remote environments, JSONL storage, telemetry, Durable Object hosting, images.
3. Approval is a seam in the design with a minimal implementation: a policy decides allow,
   deny or ask; `ask` parks the tool task on a durable approval request; no answer means
   denied. A rules engine, permission modes and approvals that amend policy come later.
4. Upstream is pinned at pi `73f8de8`. The spec is normative and the port follows it, not
   the code; upstream's tests are the behavioural oracle. The spec is still moving, so each
   design note names the spec section it follows.
5. Kept wire-compatible: entry and document kinds, record JSON shapes, the SQLite schema, so
   a storage written by pi opens in delta.

Design notes: [crates/harness/docs/design](../../crates/harness/docs/design/README.md).
