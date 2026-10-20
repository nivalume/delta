# The approval seam (`approval`)

Follows no upstream section: pi-durable has no approval of its own and leaves permission
to `beforeTool` hooks (spec §7.2, the `PlanMode` example). This note implements the
confirmed decision in [README.md](README.md): a seam with a minimal implementation,
rules and permission modes later. The ideas come from the survey
([docs/research/harnesses.md](../../../../docs/research/harnesses.md)): Claude Code's
deny > ask > allow with hooks that cannot override it, Codex's approval decisions that can
amend policy, dsh's approval that fails closed.

## Problem

A tool call may need a human (or a guardian agent) to say yes before it runs. The answer
can take minutes, the process may restart meanwhile, and the model must see one durable
outcome either way. A `beforeTool` hook cannot wait: hooks run inside the tool task's
invocation, off the line, and a crash before the consuming commit reruns them. So the wait
must be a task state, the request a committed record, and the answer a commit.

## Policy

```rust
pub trait ApprovalPolicy: Send + Sync {
    fn decide<'a>(&'a self, call: &'a ApprovalCall, api: &'a HookApi, ctx: &'a Context)
        -> BoxFuture<'a, Result<ApprovalDecision, HarnessError>>;
}

pub struct ApprovalCall<'a> {
    pub conversation_id: ConversationId,
    pub task_id: TaskId,
    pub call: &'a ToolHookCall,          // name, id, final arguments, `parent` for a nested call
    pub tool: &'a dyn HarnessTool,       // declaration, replay policy, callers
    pub cwd: Option<&'a str>,
}

pub enum ApprovalDecision {
    Allow,
    Deny { reason: String },
    Ask { timeout: Option<Duration>, prompt: Option<String> },
}

pub struct AllowAll;   // the default policy: phase 1 without a policy behaves as upstream
pub struct DenyAll { pub reason: String };
pub struct AskAll;
```

`HarnessOptions.approval: Arc<dyn ApprovalPolicy>` (default `AllowAll`; a field to add to
the struct in [harness.md](harness.md)). The policy is
asked once per call, after the `before_tool` chain and the second validation, on the final
arguments ([tool.md](tool.md) step 7). It runs off the line with the task's `HookApi`, so
it may read committed documents (a per-conversation mode, a rules document) and memos.
An error from the policy denies (fail closed), with the error text as the reason.

Hooks run first and the policy second, so a hook block always wins and a hook cannot
grant what the policy would ask or deny: Claude Code's order. A hook that wants to say
"allow" for the policy's benefit writes a memo the policy reads; phase 1 defines no such
protocol.

## Request and decision records

A request is a member of the delta-owned conversation document `delta.approvals`, and a
decision is an entry:

```rust
/// kind "delta.approvals", version 1, scope conversation, history latest, fork initial,
/// initial {}, checkpoint: complete base whenever `requests` is empty, mounted in the view as docs["delta.approvals"].
pub struct ApprovalsState { pub requests: Vec<ApprovalRequest> }   // in task id order

#[serde(rename_all = "camelCase")]
pub struct ApprovalRequest {
    pub task_id: TaskId,            // the tool task
    pub call_id: String,
    pub name: String,
    pub arguments: JsonObject,      // the final arguments, as the tool would run them
    pub prompt: Option<String>,
    pub requested_at: u64,
    pub until: u64,                 // deadline on the harness clock, epoch ms
    pub decision: Option<ApprovalOutcome>,   // set by the answering commit; the task consumes it
}

#[serde(rename_all = "camelCase")]
pub struct ApprovalOutcome { pub decision: Decision /* allow | deny */, pub reason: Option<String>, pub source: Source /* user | timeout | policy */ }
```

The decision entry has kind `delta.approval`, no `model` (display-only: it changes no
model context, is placed at once even while the conversation is busy, and is excluded from
requests by context rule 9), and `data: { taskId, callId, name, decision, reason?, source }`.
It is the audit record and what observers see; the document member is what the task
reads. `delta.approval` and `delta.approvals` are delta kinds; pi ignores entry kinds it
does not know and never mounts the document.

## The `approval` phase

```rust
ToolTaskCheckpoint::Approval { arguments, replay, until }   // committed as Next::sleep(until)
```

- **Ask** (in `call`): one commit pushes the request (`decision: None`, `until = now +
  timeout`, the policy's timeout or `settings.approval.timeout_ms`, default 900 000) and
  moves to `Approval` as a sleep until `until`. The slot stays `pending`: it is not running.
- **Answer**: `Conversation::decide(task_id, decision, reason, ctx)` (also reachable as
  `Harness::decide`) runs one commit: the request must exist with `decision: None`,
  otherwise it fails with `ApprovalError::NotPending`; it sets the member's `decision`
  (`source: user`), admits the `delta.approval` entry as a display-only write submission
  with request id `approval:<task id>:user`, and wakes the tool task (`tx.wake_task`).
- **Resume** (the `approval` phase handler, run when woken or when due): resolve the tool
  as `execute` does (`tool_unavailable` when gone, request removed). Then:
  - `decision: allow`: continue at [tool.md](tool.md) step 8; the intent commit also
    removes the request from the document.
  - `decision: deny`: settle `blocked` (`Tool call blocked: approval denied: <reason>`),
    the settle commit removes the request.
  - no decision and `now >= until`: timeout. One commit sets `decision: deny` with
    `source: timeout`, admits the `delta.approval` entry (request id
    `approval:<task id>:timeout`), removes the request, and settles `blocked`
    (`Tool call blocked: approval timed out after <n>s`).
  - no decision, woken early: sleep again with the same checkpoint.
- **Abort**: the abort handler removes the request (no decision entry) and settles
  `aborted` as usual. A run abort cascades through the generation to the tool task, so a
  user who aborts the turn also withdraws its pending approvals.
- **Recovery**: the sleep survives reopen; a crash between the answering commit and the
  phase run is harmless, the decision is in the document. The policy is never asked twice
  for one call: a rerun of `call` from before the ask commit asks again only because no
  request exists yet, and a rerun of `approval` reads the stored decision.
- **Generation**: nothing changes. The generation waits on the tool task; a sequential
  round waits for the decision before the next call starts, a parallel round lets the
  other calls run. `pi.live.tools` keeps the slot `pending` and the view shows the request
  under `docs["delta.approvals"]`.

A request's `arguments` are exactly the ones the intent commit records, so what the user
approved is what runs; a replay-safe rerun after recovery uses the same stored arguments
and asks nobody.

## Observing requests

- `ConversationView.docs["delta.approvals"]` (a delta addition to the mount,
  [observe.md](observe.md)).
- Agent event `approval_update { requests }` whenever the document changes, like
  `inbox_update`; the decision entry arrives as `entry_appended`.
- `Harness::pending_approvals(ctx)` scans every conversation's document for a UI that
  shows one queue.

## Later: rules, modes, amendments

The seam is the policy trait plus the request and decision records. What comes later
plugs into it without changing the tool task:

- **Rules and modes.** A `RulesPolicy` reads a per-conversation `delta.permissions`
  document (`mode`, and rules `Tool(specifier)` evaluated deny, then ask, then allow,
  first match wins, as Claude Code) and a harness-wide default. Permission modes are
  policies over the same rules (`default`, `accept_edits`, `plan`, `dont_ask`,
  `bypass`), selected per conversation like Codex's per-turn policy record.
- **Amendments.** `ApprovalOutcome` gains `amend: Option<Rule>` (`ApprovedForSession`,
  `ApprovedExecpolicyAmendment` in Codex's terms); the answering commit writes the rule
  into `delta.permissions` in the same commit as the decision, so the rule and the
  decision that created it are one durable fact.
- **Hook bridge.** A command / HTTP hook speaking Claude Code's `PermissionRequest`
  protocol is an `ApprovalPolicy`; `PreToolUse` and `PostToolUse` map to `before_tool`
  and `after_tool` ([tool.md](tool.md)).
- **Guardian.** A policy that submits the request to an owned reviewer conversation and
  answers from its result is `Ask` with a longer timeout plus a task that calls
  `Conversation::decide`; no new primitive.
- **Sandbox escalation.** When an execution environment can run a command confined, the
  policy decides between confined and unconfined runs; the decision and its sandbox
  policy become fields of the request record then. Phase 1 has a local environment and
  no sandbox.

## Differences from upstream

- Everything here is delta's: upstream has no approval records, phase or policy.
- The `approval` phase is a sleep that an answer wakes early. Upstream says a sleep is
  woken only by an abort; delta adds `Tx::wake_task(id)`, which marks a sleeping task due
  now (the scheduler already handles early wakes: a phase run before its time sleeps
  again). This is a [scheduler.md](scheduler.md) primitive and is listed there.
- Two new kinds, entry `delta.approval` and document `delta.approvals`, both
  conversation-scoped and ignored by pi.

## Pending confirmation

- `Tx::wake_task` as the wake path. The fallback that needs no new primitive is an
  approval task owned by the tool task, sleeping until the deadline, whose abort handler
  completes with the decision the answering commit stored; the tool task then `waits` on
  it. It works with upstream primitives but overloads the abort mark, so it is second.
- Default timeout 15 minutes, overridable per `Ask` and in settings; a timed-out request
  is denied, never resumed.
- The decision as a display-only entry placed at once (the transcript records it in
  order) versus document state only (no entry).
- Mounting `delta.approvals` in the conversation view and the `approval_update` event
  (spec §9.3 mounts built-in kinds only).
- `AllowAll` as the default: a host that wants fail-closed sets `AskAll` or `DenyAll`.
