# The compaction task (`compaction`)

Follows spec §8.7 (compaction), the threshold rules of §8.3 (`prepare`), the
`beforeCompact` hook of §7.2, the `pi.compaction` entry of §8.1 and the
`compactions` / `compactionFailure` fields of `pi.live` (§8.2). Corresponds to upstream
`src/harness/compaction.ts`.

## Problem

A long conversation outgrows the model's context window. The harness replaces an old
prefix of the model context with a summary entry whose `head` is the first kept entry;
raw history stays in storage ([context.md](context.md) derives the context from the head
marker). Summarizing is a model call that can fail, retry, or be overtaken by the
conversation it summarizes, so it is a durable task with pinned request parameters and a
placement rule for its summary.

## Records

```rust
#[derive(Serialize, Deserialize)] #[serde(rename_all = "lowercase")]
pub enum CompactionReason { Manual, Threshold, Overflow }

#[serde(rename_all = "camelCase")]
pub struct CompactionInput { pub reason: CompactionReason, pub instructions: Option<String> }

/// The pinned summarization request; fixed across retries.
#[serde(rename_all = "camelCase")]
pub struct SummaryRequest {
    pub attempt: u32,
    pub model: ModelRef,
    pub thinking_level: ModelThinkingLevel,
    pub stream_options: ConversationStreamOptions,
    pub max_tokens: u64,
    pub tail: EntryId,          // newest entry of the context the range was selected from
    pub first_kept: EntryId,    // first entry kept verbatim; the summary's `head`
}

#[derive(Serialize, Deserialize)] #[serde(tag = "phase", rename_all = "lowercase")]
pub enum CompactionCheckpoint {
    Select,
    Summarize { #[serde(flatten)] request: SummaryRequest },
    Retry     { #[serde(flatten)] request: SummaryRequest, until: u64 },
}

/// `entry_id` for a blocking compaction's summary, `submission_id` for a conversation-owned one's write; both absent when nothing was compacted.
#[serde(rename_all = "camelCase")]
pub struct CompactionResult { pub entry_id: Option<EntryId>, pub submission_id: Option<SubmissionId> }
```

`pi.compaction` is version 1 and starts at `Select`. Field names are upstream's, so a
compaction in flight when pi wrote the storage resumes in delta.

```rust
pub fn create_compaction(tx: &mut Tx, conversation: ConversationId, input: CompactionInput, owner: Option<TaskId>)
    -> Result<TaskId, HarnessError>;
```

creates the task (owned by `owner`, the generation, or by the conversation; `background`
when conversation-owned and not `Manual`) and adds its `CompactionStatus { task_id,
reason, blocking: owner.is_some(), attempt: 1, retry: None }` to `pi.live.compactions` in
the same commit.

## Who starts one

| started by | owner | background | generation waits | placement |
| --- | --- | --- | --- | --- |
| `Conversation::compact(instructions)` | conversation | no | no | write submission |
| generation above the background threshold | conversation | yes | no | write submission |
| generation above the blocking threshold, or on overflow | the generation | no | yes | direct append |

Only a blocking compaction runs while the run waits for it. Compactions do not coordinate:
several may run at once, and the inbox's stale check decides between their summaries.

## Thresholds (read by generation `prepare`)

```rust
pub struct CompactionPolicy { pub enabled: bool, pub reserve_tokens: u64, pub keep_recent_tokens: u64, pub background_tokens: u64 }
pub fn compaction_policy(settings: &Settings, model: &ModelRef) -> CompactionPolicy;   // model override, field by field
pub fn estimate_context(view: &ContextView, extra: &[Message]) -> u64;
```

Defaults: `enabled`, `reserve_tokens 16384`, `keep_recent_tokens 20000`,
`background_tokens 32768`; `settings.compaction.model_overrides["provider/modelId"]`
replaces fields for that model.

`estimate_context`: find the newest assistant message in the committed context whose
entry is after the head marker (every entry without one) and whose usage is nonzero;
start from `delta_ai::estimate::calculate_context_tokens(usage)` and add
`estimate_message_tokens` of every context message after it and of `extra` (the planned
system messages). Without such a message, estimate every message. Entry order, not
timestamps, decides "after".

Generation checks, when the policy is enabled, the model's `context_window` is positive,
and the generation has not already compacted (`compacted` absent), and only when
`select_cut` finds a cut:

- estimate > `context_window - reserve_tokens`: blocking; a generation-owned task with
  reason `Threshold`, the generation waits (`all_settled`) and sends its request when the
  compaction completed (with or without a summary), fails `model_error` otherwise.
- estimate > `context_window - reserve_tokens - background_tokens` with
  `background_tokens > 0` and no status in `pi.live.compactions`: a conversation-owned
  background task is created in the commit that moves to `request`.
- a context-overflow error with `compacted` absent and a cut: an `Overflow` compaction the
  generation owns; the recovery request does not count against retries; a second overflow
  fails. Details in [generation.md](generation.md).

## Range selection

```rust
pub fn select_cut(view: &ContextView, keep_recent_tokens: u64) -> Option<usize>;    // index into view.entries
pub fn summarized_messages(view: &ContextView, cut: usize) -> Vec<Message>;
```

A pure function of a `ContextView` (`entries`, `head`, `contributions`: each active
entry's model messages after every edit in the range).

1. Candidates are the non-marker entries whose contribution begins with a user or
   assistant message. Tool results and system entries never are, so a kept assistant keeps
   its results. A user entry is not a candidate while a result for a call of the newest
   assistant before it still follows it, before the next assistant message.
2. Walk the non-marker entries from newest to oldest, summing `estimate_message_tokens`
   of each contribution. At the first entry where the sum reaches `keep_recent_tokens`,
   the cut is the first candidate at or after that entry in transcript order, or the
   newest candidate when none follows.
3. `None` when the sum never reaches `keep_recent_tokens`, or when no non-marker entry
   before the cut contributes a message.

The summarized entries are the head marker, if any, then the non-marker entries before
the cut; their messages are their contributions in context order (tool results moved
after their calls, as [context.md](context.md) orders them), so an earlier summary is
summarized again with the history after it.

```text
1 user   2 assistant (read)   3 tool result, 30k tokens   4 assistant   5 user   6 assistant
keep_recent_tokens 20k: the walk from 6 reaches 20k at 3; the first candidate at or after 3 is 4.
The summary covers 1-3; the model context becomes [summary, 4, 5, 6].
```

## Phases

- **`select`**: resolve the phase's agent and settings once; capture the committed
  context. No model, or `models.get_model` does not know it: fail `no_model` (status
  removed, failure recorded). No cut: complete with `{}` (status removed). Otherwise run
  the `before_compact` chain with `{ reason, entries (marker first), messages, first_kept,
  instructions }`; the first decision wins: `Decline` completes with `{}`, `Summary(text)`
  is placed as below in its own commit. Without a decision, one commit moves to
  `Summarize` with attempt 1, the agent's model and thinking level, the settings' stream
  options, `max_tokens = min(floor(0.8 * reserve_tokens), model.max_tokens)` (the
  model's value only when positive), `tail` = the newest entry id of the captured
  context, `first_kept` = the cut entry's id.
- **`summarize`**: derive the summarized messages again from the context at `tail`
  (immutable), serialize them, and send one request with no tools: a system message with
  the built-in summarization system prompt and a user message with
  `<conversation>\n…\n</conversation>\n\n<prompt>` plus `\n\nAdditional focus:
  <instructions>` when given. Options: the pinned thinking level as `reasoning` (omitted
  for `off`), the conversation's persisted provider `session_id`, the pinned stream
  options without `deferred`, `cache_retention: none`, the pinned `max_tokens`, the
  invocation's cancellation. Recovery resends the same request. After the response,
  check the abort mark, then classify in one commit that adds the response's usage to
  `pi.usage.models["provider/model"]`:
  - `stop`, non-empty text, no tool call: the text is the summary; place it.
  - `error` the retry classification accepts (the same predicate generation uses) while
    `settings.retry` allows another attempt (`enabled && attempt <= max_retries`): set
    the status's `retry { at, error }` and sleep until `until = now + retry_delay(policy,
    attempt)` at `Retry`.
  - anything else (`length`, a tool call, empty text, non-retryable error, `aborted`):
    fail `model_error` with a message (`Summarization failed: <error>`, `Summarization
    hit the token limit; the summary is incomplete`, `Summarization attempted to call a
    tool`, `Summarization produced no text`).
- **`retry`**: run before `until`, sleep again with the same checkpoint. Otherwise one
  commit sets the status's `attempt` to `attempt + 1`, removes its `retry`, and returns to
  `Summarize` with the same pinned request and the next attempt.

Serialization (`serialize_conversation`): `[User]: <text>`, `[Assistant]: <text>`,
`[Assistant tool calls]: name(key=json, ...)` joined with `; `, `[Tool result]: <text>`
cut to 2000 characters with `[... n more characters truncated]`; parts joined by blank
lines; system messages and thinking omitted; empty texts skipped. The prompts are the
coding agent's structured checkpoint format (Goal, Constraints & Preferences, Progress,
Key Decisions, Next Steps, Critical Context) and ask the model to fold an earlier summary
at the start of the conversation into the new one.

## Placement

The summary entry:

```json
{ "kind": "pi.compaction", "head": <first_kept>,
  "model": [{ "role": "user", "content": [{ "type": "text", "text": <wrapped> }], "timestamp": <now> }],
  "data": { "reason": <reason> } }
```

`wrapped` = `"The conversation history before this point was compacted into the
following summary:\n\n<summary>\n"` + summary + `"\n</summary>"`. The placing commit also
removes the task's status, removes `pi.live.compactionFailure`, and commits `completed`:

- blocking (task-owned): append the entry directly; result `{ entry_id }`. The generation
  holds the run, so nothing else writes.
- conversation-owned: admit the entry as a write submission with request id
  `compaction:<task id>` through [inbox.md](inbox.md): placed at once in an idle
  conversation with an empty inbox, or settled `stale` when its head no longer targets
  the current range; queued behind waiting items otherwise. Result `{ submission_id }`;
  the task never waits for placement, `Harness::submission` observes it.

The next generation's `prepare` finds a head marker without a later `pi.system` entry and
writes a complete system baseline ([context.md](context.md)).

## Cancellation and failure

- Abort handler: remove the status, end `aborted`, write nothing. A cut-off attempt has
  no committed response, so no usage is counted. A manual compaction is ordinary work:
  `Conversation::abort` aborts it. A background one survives it and stops only through
  `abort_task` or `abort(background: true)`. A blocking one is aborted with its
  generation, before the generation's abort handler runs. A queued summary is a write
  submission: conversation abort keeps it, `Submission::abort` withdraws it.
- Failure (`model_error`, `no_model`): the failing commit removes the status and sets
  `pi.live.compactionFailure { task_id, reason, error }`, replacing an older one; the
  scheduler cleanup does the same for a faulted or orphaned compaction with the fault's
  message or the orphan reason. Only a placed summary removes the failure; a compaction
  that is aborted or finds nothing to compact leaves it.
- Stale summaries: a summary reflects the context at `select`. A conversation-owned one
  whose `head` targets an entry no longer in the current range (another compaction cut
  further) settles `stale` at admission; its spend was still counted. Application edits
  placed meanwhile whose target leaves the range are lost in the summary; this is the
  documented footgun, not something compaction defends against.

## Differences from upstream

- Serde-tagged checkpoint enum; `SummaryRequest` flattened into `Summarize` and `Retry`,
  so the JSON equals upstream's.
- `select_cut`, `summarized_messages`, `estimate_context`, `serialize_conversation` and
  `compaction_policy` are public functions tested against upstream's `compaction.test.ts`
  cases with recorded responses; the retry predicate and delay are shared with generation
  (defined there, since `delta_ai` has no `isRetryableAssistantError` / `retryDelayMs`
  port yet).
- The summarization request goes through `delta_ai::Models::stream_simple` collected to a
  message (upstream `completeSimple`); the stream is cancelled by dropping it on abort.

## Pending confirmation

- Where the retry predicate lives (`generation` or a small `retry` module both tasks use).
- Whether `before_compact` receives `&[EntryRecord]` and `&[Message]` borrowed from the
  phase (cheap) or owned clones (simpler hook signatures).
