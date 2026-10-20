# The approval seam (`approval`)

Follows no upstream section: pi-durable has no approval of its own and leaves permission
to `beforeTool` hooks (spec §7.2, the `PlanMode` example). The flow is codex-rs's, made
durable, as decided in [README.md](README.md); rules and permission modes come later. Other
ideas come from the survey ([docs/research/harnesses.md](../../../../docs/research/harnesses.md)):
Claude Code's deny > ask > allow with hooks that cannot override it, dsh's approval that
fails closed.

## Problem

A tool call may need a human (or a guardian agent) to say yes before it runs. The answer
can take minutes, the process may restart meanwhile, and the model must see one durable
outcome either way. A `before_tool` hook cannot wait for it: hooks run inside the tool
task's invocation, off the line, and a crash before the consuming commit reruns them. So
the request must be a committed record, the answer a commit, and the wait something a
rerun resumes.

## Codex's flow

Codex keeps, per turn, `TurnState::pending_approvals: HashMap<String,
oneshot::Sender<ReviewDecision>>`. `request_command_approval` registers a oneshot under the
approval id, emits an approval-request event and awaits the receiver; the client answers
with `Op::ExecApproval { id, decision }`, and `notify_approval(id, decision)` sends on the
stored sender. `ReviewDecision` is `Approved`, `ApprovedForSession`, `Denied`, `TimedOut`
or `Abort` (plus policy amendments, see "Later"). `ApprovedForSession` puts the request's
serialized approval key into the session's `ApprovalStore`, so a later request with the
same key is approved without asking; `Abort` aborts the turn. All of it is in memory: a
restart loses the pending request.

delta keeps the shape and commits three parts of it: the request, the decision and the
session grants. The oneshot map stays in memory and is only the fast path from the deciding
commit to the waiting phase.

## Policy

```rust
pub trait ApprovalPolicy: Send + Sync {
    fn decide<'a>(&'a self, call: &'a ApprovalCall<'a>, api: &'a HookApi, ctx: &'a Context)
        -> BoxFuture<'a, Result<PolicyDecision, HarnessError>>;
}

pub struct ApprovalCall<'a> {
    pub conversation_id: ConversationId,
    pub task_id: TaskId,
    pub call: &'a ToolHookCall,          // name, id, final arguments, `parent` for a nested call
    pub tool: &'a dyn HarnessTool,       // declaration, replay policy, callers, approval key
    pub cwd: Option<&'a str>,
}

/// Codex's `ExecApprovalRequirement` (`Skip`, `NeedsApproval`, `Forbidden`).
pub enum PolicyDecision {
    Allow,
    Deny { reason: String },
    Ask { timeout: Option<Duration>, prompt: Option<String> },
}

pub struct AllowAll;   // the default policy: phase 1 without a policy behaves as upstream
pub struct DenyAll { pub reason: String };
pub struct AskAll;
```

`HarnessOptions.approval: Arc<dyn ApprovalPolicy>` (default `AllowAll`,
[harness.md](harness.md)). The policy is asked once per call, after the `before_tool`
chain and the second validation, on the final arguments ([tool.md](tool.md) step 7). It
runs off the line with the task's `HookApi`, so it may read committed documents (a
per-conversation mode, a rules document) and memos. An error from the policy denies (fail
closed), with the error text as the reason.

Hooks run first and the policy second, so a hook block always wins and a hook cannot
grant what the policy would ask or deny: Claude Code's order. A hook that wants to say
"allow" for the policy's benefit writes a memo the policy reads; phase 1 defines no such
protocol.

**Session grants.** On `Ask`, before any request is committed, the call's key
`HarnessTool::approval_key(&arguments)` ([tool.md](tool.md); default: the canonical JSON of
the arguments, `None` for a call that may never be granted) is looked up with the tool name
in `delta.approvals.grants`. A match proceeds as `Approved` without a request, as Codex's
`ApprovalStore` check does. `Allow` and `Deny` never consult the grants.

## Records

A request and the session grants live in the conversation document `delta.approvals`; a
decision is written into its request and recorded as an entry:

```rust
/// One request per tool task, so the task id names it.
#[serde(transparent)]
pub struct ApprovalId(pub TaskId);

/// kind "delta.approvals", version 1, scope conversation, history latest, fork initial
/// (a fork starts without requests or grants), initial {"requests": [], "grants": []},
/// complete base whenever `requests` is empty, mounted in the view as docs["delta.approvals"].
pub struct ApprovalsState {
    pub requests: Vec<ApprovalRequest>,      // in id order
    pub grants: Vec<ApprovalGrant>,          // in grant order
}

pub struct ApprovalRequest {
    pub id: ApprovalId,
    pub call_id: String,
    pub name: String,
    pub arguments: JsonObject,               // the final arguments, as the tool would run them
    pub approval_key: Option<String>,
    pub prompt: Option<String>,
    pub requested_at: i64,
    pub deadline: i64,                       // harness clock, epoch ms; also in the task checkpoint
    pub decision: Option<ReviewDecision>,    // set by the deciding commit; the task consumes it
}

pub struct ApprovalGrant { pub tool: String, pub key: String, pub granted_by: ApprovalId }

#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReviewDecision {
    Approved,
    ApprovedForSession,
    Denied { rejection: Option<String> },
    TimedOut,                                // written by the harness only
    Abort,
}
```

The decision entry has kind `delta.approval`, no `model` (display-only: it changes no
model context, is placed at once even while the conversation is busy, and is excluded from
requests by context rule 9), and `data: { approval_id, call_id, name, decision }`. It is
the audit record and what observers see; the request's `decision` is what the task reads.
It is admitted as a display-only write submission with request id `approval:<id>`, so a
retried `decide` writes it once.

## Waiters

```rust
/// An agent-layer session service (session.md `Services`), inserted when the Harness opens.
pub struct PendingApprovals { /* Mutex<HashMap<ApprovalId, oneshot::Sender<ReviewDecision>>> */ }
impl PendingApprovals {
    /// Replaces an older sender for the same id (a rerun of the phase).
    pub fn register(&self, id: ApprovalId) -> ApprovalWaiter;
    /// After the deciding commit settled; `false` when nobody waits.
    pub fn notify(&self, id: ApprovalId, decision: ReviewDecision) -> bool;
}
/// The receiver; removes its map entry when dropped.
pub struct ApprovalWaiter { /* oneshot::Receiver<ReviewDecision> */ }
```

Codex's map, held per session instead of per turn and keyed by `ApprovalId`. It is never
persisted and the kernel never reads it: losing it (crash, close) loses nothing, because
the phase reads the committed decision before it waits.

## The flow

```rust
ToolTaskCheckpoint::Approval { id: ApprovalId, arguments, replay, deadline: i64 }
```

1. **Ask** (in `call`, [tool.md](tool.md) step 7): the policy says `Ask` and no grant
   matches. One commit pushes the request (`decision: None`, `deadline = now + timeout`,
   the policy's timeout or `settings.approval.timeout`, default 15 min) and moves to
   `Running(Approval { .. })`. The slot stays `pending`. Step rule 5 runs the `approval`
   phase next in the same invocation.
2. **Wait** (the `approval` phase): register a waiter in `PendingApprovals`, then read the
   request from the committed document. Registering first closes the race with a decision
   committed between the read and the registration.
   - A decision is there: consume it (below).
   - No decision: await the waiter under `tokio::time::timeout_at(deadline)` through the
     invocation context. The task stays `running` and the invocation holds it; an abort
     mark or close cancels the await.
   - No request (a bug: only this task's own commits remove it): the phase fails and the
     task faults.
3. **Decide**: `Conversation::decide(id, decision, ctx)` (also `Harness::decide`) runs one
   commit: the request must exist with `decision: None`, otherwise
   `ApprovalError::NotPending`; `TimedOut` is refused from callers. It sets the request's
   `decision`, pushes a grant for `ApprovedForSession` when the request has an
   `approval_key`, admits the `delta.approval` entry, and for `Abort` marks the
   conversation's run task for abort (the cascade reaches this task). After the commit
   settles it calls
   `PendingApprovals::notify`.
4. **Consume**:
   - `Approved`, `ApprovedForSession`: continue at [tool.md](tool.md) step 8; the intent
     commit removes the request.
   - `Denied { rejection }`: settle `blocked` (`Tool call blocked: approval denied:
     <rejection>`, or `approval denied` alone); the settle commit removes the request.
   - `Abort`: the deciding commit already marked the run, so the cascade marks this task
     and its invocation ends at the next await or step; the abort handler settles it.
   - The deadline passes without a decision: one commit sets `TimedOut`, admits the
     `delta.approval` entry, removes the request and settles `blocked` (`Tool call
     blocked: approval timed out after <n>s`). A `decide` racing it finds the request
     decided or gone (`NotPending`).
5. **Abort**: the abort handler removes the request (an undecided one gets no decision
   entry) and settles `aborted` as usual. A run abort cascades through the generation to
   the tool task, so a user who aborts the turn also withdraws its pending approvals.
6. **Recovery**: a crash or close while waiting leaves the task `running`; open (or close)
   writes it back to `pending`. The next reservation reruns `approval`: it reads a
   committed decision first, otherwise registers and waits again until the stored
   deadline, timing out at once when it already passed. The policy is never asked twice
   for one call: a rerun of `call` from before the ask commit asks again only because no
   request exists yet, and a rerun of `approval` reads the stored decision.
7. **Generation**: nothing changes. The generation waits on the tool task; a sequential
   round waits for the decision before the next call starts, a parallel round lets the
   other calls run. `delta.live.tools` keeps the slot `pending` and the view shows the
   request under `docs["delta.approvals"]`.

A request's `arguments` are exactly the ones the intent commit records, so what the user
approved is what runs; a replay-safe rerun after recovery uses the same stored arguments
and asks nobody.

## Observing requests

- `ConversationView.docs["delta.approvals"]` ([observe.md](observe.md)).
- Agent event `approval_update { requests }` whenever the document changes, like
  `inbox_update`; the decision entry arrives as `entry_appended`.
- `Harness::pending_approvals(ctx)` scans every conversation's document for a UI that
  shows one queue.

## Later: rules, modes, amendments

The seam is the policy trait plus the request, decision and grant records. What comes
later plugs into it without changing the tool task:

- **Rules and modes.** A `RulesPolicy` reads a per-conversation `delta.permissions`
  document (`mode`, and rules `Tool(specifier)` evaluated deny, then ask, then allow,
  first match wins, as Claude Code) and a harness-wide default. Permission modes are
  policies over the same rules (`default`, `accept_edits`, `plan`, `dont_ask`,
  `bypass`), selected per conversation like Codex's per-turn policy record.
- **Amendments.** Codex's `ApprovedExecpolicyAmendment` (and its network policy
  amendment) lets a decision add a rule. delta adds a `ReviewDecision` variant carrying
  the rule; the deciding commit writes it into `delta.permissions` with the decision, so
  the rule and the decision that created it are one durable fact.
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
- Two kinds, entry `delta.approval` and document `delta.approvals`, both
  conversation-scoped.

## Differences from codex-rs

- The request, the decision and the session grants are committed; Codex keeps them in
  memory. The oneshot map is a fast path, and a rerun reads the committed decision.
- The pending map is a session service keyed by `ApprovalId` (the tool task), not a
  per-turn map keyed by a string.
- Grants are per conversation, in its document; a fork starts without them.
- A pluggable `ApprovalPolicy` decides whether to ask, where Codex derives the requirement
  from its approval and sandbox policies.

## Decisions

- Codex's flow made durable, as above: committed request, in-memory waiter keyed by
  approval id, `Conversation::decide` commits then notifies, a rerun reads the committed
  decision first. No wake primitive in the scheduler.
- `ReviewDecision { Approved, ApprovedForSession, Denied { rejection }, TimedOut, Abort }`;
  `ApprovedForSession` stores `HarnessTool::approval_key(args)` as a grant; `Abort` aborts
  the run; `Denied` and `TimedOut` settle `blocked` with an "approval denied:" or
  "approval timed out" message.
- Hooks run before the policy; a hook block wins; hooks cannot grant; a policy error
  denies.
- Visibility: `delta.approvals` mounted in the view, the `approval_update` event, the
  decision as a display-only `delta.approval` entry.

- Phase 4 defined `ApprovalPolicy::decide(&ApprovalCall, &Context)`, `PolicyDecision` and
  `AllowAll`, passed through `HarnessOptions.approval` and held by the agent services.
  `ApprovalCall` carries the delta_ai `ToolCall` with its final arguments; phase 5 adds the
  `HookApi` argument (once hooks read documents), `ToolHookCall`, the records, the waiting
  phase and `decide` on the surface.

## Open (minor)

- `AllowAll` as the default: a host that wants fail-closed sets `AskAll` or `DenyAll`.
- Default timeout 15 minutes, overridable per `Ask` and in settings.
- Whether an `Abort` decision also withdraws the conversation's queued inputs, as
  `Conversation::abort` does, or only marks the run.
