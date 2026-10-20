# Submissions and the inbox (`inbox`)

Corresponds to upstream `src/harness/inbox.ts` and `src/harness/submissions.ts`. Follows
spec §6 (submissions and inbox) and the submission parts of §2.2 (`submit`, `Submission`,
`reset`, `abort`, request ids) and §8.1 (`pi.user`, `pi.reset`). Record shapes
(`SubmissionRecord`, `SubmissionSettlement`, `EntryDraft`) are in [records.md](records.md);
run control (`pi.live.run`) and the boundaries' callers in [generation.md](generation.md).

## Problem

Input arrives while a conversation is busy, and passive entry writes (a note for the UI, a
compaction summary, a reset) arrive at any time. Each must be admitted durably in one commit,
be awaitable, and be placed into the transcript at a point where it changes the model's
context predictably: between a tool round and the next request, or after an answer. The
inbox is a built-in document holding the queued items in order; a *boundary* places the
items it selects; the submission record tracks the result.

## Records and documents

```rust
pub enum SubmissionDraft {
    Input { content: UserContent, when_busy: WhenBusy /* Steer | FollowUp | Reject; default FollowUp */,
            entry: Option<InputEntry> /* { kind, data? } */, request_id: Option<String> },
    Write { entry: EntryDraft, request_id: Option<String> },
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "camelCase")]
pub enum InboxItem {
    Steer    { id: SubmissionId, content: UserContent, #[serde(skip_serializing_if = "Option::is_none")] entry: Option<InputEntry> },
    FollowUp { id: SubmissionId, content: UserContent, #[serde(skip_serializing_if = "Option::is_none")] entry: Option<InputEntry> },
    Write    { id: SubmissionId, entry: Json },     // the EntryDraft as plain JSON
}
pub struct InboxState { pub items: Vec<InboxItem> }   // ID order
```

| field | value |
| --- | --- |
| kind / version | `pi.inbox` / 1 |
| scope / history / fork | conversation, `latest`, `initial` (a fork starts empty) |
| `initial()` | `{ items: [] }` |
| checkpoint | complete base whenever `items` is empty |
| created | with every Harness conversation (creation hook, [harness.md](harness.md)) |

The definition type `Inbox` is in [documents.md](documents.md); this module owns the contents, read and written through `Doc::update_as::<InboxState, _>` and the draft handle. A
queued input stores its content and its entry's `kind` and `data` (nothing else from the
host's draft); its entry gets the Harness clock's timestamp at placement. A queued write
stores the `EntryDraft` as JSON.

An input is placed as one entry whose `model` is `[UserMessage { content, timestamp }]`: a
`pi.user` entry, or an entry of `entry.kind` carrying `entry.data`. Admission rejects a kind
that is empty or starts with `pi.`. Nothing reads input entries by kind: context, compaction,
events and run accounting go by `model` and the submission records.

## API

```rust
pub struct Boundary<'tx> { conversation: ConversationId, inbox: Doc<'tx, Inbox>,
                           steering_mode: QueueMode, follow_up_mode: QueueMode, head: Option<EntryId> }
pub enum At { PostTools, Final }
pub struct BoundaryResult { pub users: Vec<SubmissionId>, pub reset: bool }

/// Table reads precede the commit's first table write: callers prepare the boundary first.
pub async fn prepare_boundary(tx: &mut Tx, conversation: ConversationId, modes: &QueueModes) -> Result<Boundary, TxError>;
pub async fn apply_boundary(tx: &mut Tx, boundary: Boundary, at: At, now: i64) -> Result<BoundaryResult, TxError>;

pub async fn admit_submission(tx: &mut Tx, conversation: ConversationId, draft: SubmissionDraft, now: i64, modes: &QueueModes)
    -> Result<Admission /* { id, status: Queued | Placed | Done } */, SubmitError>;
pub async fn append_input(tx: &mut Tx, conversation: ConversationId, content: UserContent, entry: Option<&InputEntry>, now: i64) -> Result<EntryRecord, TxError>;
pub async fn withdraw_queued_inputs(tx: &mut Tx, conversation: ConversationId, reason: &str /* "aborted" */) -> Result<(), TxError>;
pub async fn remove_inbox_item(tx: &mut Tx, conversation: ConversationId, id: SubmissionId) -> Result<(), TxError>;
pub fn is_display_only(entry: &EntryDraft) -> bool;   // no model, no head, no edits (present counts, even empty)
pub fn has_queued_reset(items: &[InboxItem]) -> bool; // a Write whose entry has head: "self"
pub fn is_stale(boundary: &Boundary, entry: &EntryDraft) -> bool;   // numeric head < active range start
```

The boundary's `head` is the newest head marker's `head` as of the commit's start, advanced
by every head the boundary places (a `head: "self"` write advances it to the entry's own
id). Queue modes are read on the line per boundary from the resolved settings
([harness.md](harness.md)).

## Admission

`Conversation::submit` runs `admit_submission` in one commit and returns a `Submission`
handle after that commit settled.

1. `request_id`: a submission with the same id in this conversation is returned instead of a
   new write; a different `type` under the same id is an error. The retry of a crashed
   `submit` is therefore exactly-once.
2. A display-only write with no queued reset is appended and settled `done` at once, busy or
   idle, ahead of earlier queued submissions: it changes no model context. Between a tool
   call and its result it changes nothing either, because results are ordered by message.
3. `busy = pi.live.run.is_some()`. A busy input with `WhenBusy::Reject` writes nothing and
   fails with `SubmitError::ConversationBusy`.
4. Busy, or idle with a non-empty inbox: create the record `queued` and push the item
   (`Steer` when `when_busy` is `Steer`, else `FollowUp`; `Write` for a write). Idle with a
   non-empty inbox then runs a `Final` boundary in the same commit, so order is preserved: a
   follow-up queued before a failed run is selected before the new input. A display-only
   write with a queued reset queues too (rule 2 does not apply), so it lands after the reset.
5. Idle with an empty inbox: a write is appended and settled `done` (or `unanswered` / `stale`
   when its head is stale); an input is placed as its entry, the record is `placed`, and
   `start_run` creates `pi.live.run { task_id, inputs: [id] }` with the first `pi.generation`
   task ([generation.md](generation.md)).

| action | submission state | other writes |
| --- | --- | --- |
| input, idle, empty inbox | `placed` + user entry | run control + generation task |
| display-only write, no queued reset | `done` + entry | append entry; no run |
| write, idle, empty inbox | `done` + entry (or `unanswered` / `stale`) | append entry; no run |
| input or other write, busy or non-empty inbox | `queued` | inbox item (+ `Final` boundary when idle) |
| boundary places a user item | `placed` + entry | id added to the current or successor run |
| boundary places a write | `done` + entry | append entry |
| run answers | input `done`, `answer` = the answer entry | run control removed or handed to a successor |
| run fails or its task aborts | input `unanswered` + reason | run control removed; inbox per `after_failed_run` |
| withdraw a queued item | `unanswered` / `aborted` | item removed |
| stale head write | `unanswered` / `stale` | item removed |

## Boundaries

A boundary runs inside the commit that ends a generation attempt or a tool round. Selection
is deterministic by item id:

| boundary | writes | steers | follow-ups |
| --- | --- | --- | --- |
| `PostTools` | all | first (`OneAtATime`) or all | none |
| `Final` | all | first or all | first or all |

1. A queued reset (`head: "self"` write) turns `PostTools` into `Final`.
2. Writes are placed first, in id order. A write whose numeric `head` is older than the
   boundary's `head` is stale: it would bring back history a head cut. Heads placed earlier
   in the same boundary count, so after a queued reset a queued summary targeting an older
   entry is stale. `head: "self"` is never stale.
3. Selected user items are placed next, in id order, so a user item queued before a reset or
   a compaction summary lands after it and runs in the new context. Queued user items never
   become stale.
4. Selected and stale items are removed positionally (a `p` op per removal, from the end);
   retained items keep their order. The draft handle records those removals as ops without
   carrying retained values ([chord delta.md](../../../chord/docs/design/delta.md)).
5. The result lists the placed user ids and whether a reset was placed.

What the caller does with the result (spec §6, §8.3, §8.5):

- Ordinary `PostTools`: generation continues even with no selected item; the selected steer
  ids join `pi.live.run.inputs`. A placed reset ends the current run with its inputs
  `unanswered` / `reset`, and the selected user ids start a successor run. A terminating or
  handoff tool round uses `Final` behaviour.
- `Final`: the run's placed inputs settle `done` first (unless an `on_yield` continuation
  keeps them open); the selected user ids start one successor generation with those ids as
  the new run's inputs. A reset at `Final` follows the answer, so the inputs are already
  `done`. Writes never trigger generation. Without user triggers the conversation goes idle,
  except for an `on_yield` continuation, which applies only when no user item and no reset
  was selected.
- Only successful run ends (an answer, `terminate`, `handoff`) apply the final boundary.
  Failure, an abort handler, a fault and an orphan settle the run's inputs `unanswered` and
  leave the inbox alone: the items stay visible in `pi.inbox` until the next input or queued
  write runs a boundary, or until withdrawn.

**After a failed run.** `settings.after_failed_run` decides, in the commit that settles a
`model_error` or `no_model` failure, what happens to the queued items: `Keep` (default)
leaves them; `Continue` applies `Final`, so the next run starts with them; `Withdraw`
settles queued inputs `unanswered` / `run_failed` and keeps queued writes. A run whose task
is aborted, faults or is orphaned keeps its inbox. An idle conversation with a non-empty
inbox queues every new input and every non-display-only write behind the waiting items and
runs a `Final` boundary in the same commit (rule 4 above); a display-only write is appended
without one, so a chat message never restarts failed work.

**Compaction summaries** are head writes, so the stale rule alone orders compactions by cut
position: a summary cutting before the active range start is stale, one cutting at or after
it is placed, whenever it was selected ([compaction.md](compaction.md)). An application head
placed while a compaction summarizes can be lost, and an application head can be undone;
both are documented footguns, not bugs.

## Run control

`pi.live.run: Option<{ task_id: TaskId, inputs: Vec<SubmissionId> }>` defines busy. It names
the generation responsible for the run, including while its tool round runs; tool tasks
never own it. `inputs` is mutable because a boundary adds placed steering ids to an active
run; every terminal path settles exactly the listed inputs. `tx.settle_submission(id,
settlement)` stages the status change and resolves against the transaction's latest
candidate record during assembly, so it works after the commit's first table write.

## Submission handle and withdrawal

- `Submission::wait` resolves on `done` or `unanswered`: an input after its answer or
  terminal failure, a write when its entry is placed or it became unplaceable. It is a
  per-record `watch`; cancelling the wait's context cancels only the wait.
- `Submission::abort` withdraws a still-queued submission (`Aborted`: record `unanswered` /
  `aborted`, item removed), reports `AlreadyPlaced` for a placed input and `Settled` for a
  terminal one. `Harness::abort_submission` adds `NotFound`.
- `Conversation::abort` and abort cascades call `withdraw_queued_inputs`: steers and
  follow-ups settle `unanswered` / `aborted` and leave the inbox; writes stay for later
  placement.
- `Conversation::reset(handoff)` is `submit(Write { entry: pi.reset, head: "self", model:
  handoff.map(user message) })`; it resolves after admission, and placement follows the
  rules above. An idle wait does not guarantee placement of queued passive writes.

## Testing

Ports of `harness-inbox`, `harness-submissions`, `harness-failed-run-inbox`,
`harness-input-entries` and the reset cases of `harness-conversations`: queue modes,
display-only writes before and after a queued reset, stale heads inside one boundary,
request-id idempotency, `WhenBusy::Reject`, `after_failed_run` in all three modes,
withdrawal while placed, and the positional ops of item removal (asserted on the
`pi.inbox` deltas).

## Differences from upstream

- `InboxItem` is a serde-tagged enum; the stored JSON (`mode`, `id`, `content`, `entry`) is
  unchanged.
- Admission failures are `SubmitError` variants (`ConversationBusy`, `InvalidEntryKind`,
  `RequestIdTypeMismatch`, `Closed`), not thrown errors.
- The boundary borrows its inbox draft from the `Tx` (`Doc<'tx, Inbox>`), so it cannot outlive
  the commit; upstream's "prepare before the first table write" rule becomes a borrow rule
  plus the `ReadAfterWrite` check the session enforces.
- Telemetry links between a steer and the run's trace are not ported (no telemetry crate).

## Pending confirmation

- Nothing beyond [README.md](README.md). The module is a direct port of §6 on the session
  API; the only choices are the enum shapes above.
