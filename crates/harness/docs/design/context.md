# Model context and the system prompt (`context`, `prompt`)

Corresponds to upstream `src/harness/context.ts` and `src/harness/prompt.ts`. Follows spec
§2.1 (entries and context derivation), §7.4 (system prompt and dynamic tools) and the
`delta.system` row of §8.1. Entry records, `ContextEdit` and head markers are in
[records.md](records.md); the extension sections and agent resolution the prompt consumes
in [harness.md](harness.md); the generation phase that calls both in
[generation.md](generation.md).

## Problem

The transcript is an immutable, append-only list of entries; what the model sees is derived
from it. Two things make the derivation non-trivial. Head markers (resets, compaction
summaries) cut the active range without deleting history, and context edits omit or replace
older entries. Prompt sections and tool declarations are positional `SystemMessage`s in the
transcript, so the request a provider gets is reproducible from stored entries alone, and
preparation must append the smallest patch that makes the replayed state equal the desired
one, keeping provider prompt caches stable.

## Context derivation (§2.1)

```rust
pub struct ContextView {
    pub head: Option<EntryRecord>,            // H: newest head marker in range
    pub entries: Vec<EntryRecord>,            // active entries: H, then the non-head entries after it
    pub contributions: Vec<Vec<Message>>,     // per entry: its messages after edits and rule 9, before rules 7, 8, 10
    pub messages: Vec<Message>,               // the model context
}

pub struct ContextBounds { head: Option<EntryRecord /* with head */>, tail: EntryId }

/// Two O(1) storage reads: the newest visible entry (or `at`, which must be visible) and the
/// newest head marker at or before it. Entries at or below `tail` never change.
pub async fn capture_bounds(storage: &dyn Storage, conversation: ConversationId, at: Option<EntryId>, ctx: &Context) -> Result<Option<ContextBounds>, ContextError>;
/// Scan the range and derive the view.
pub async fn derive_context(storage: &dyn Storage, conversation: ConversationId, bounds: Option<&ContextBounds>, ctx: &Context) -> Result<ContextView, ContextError>;
pub fn derive_view(head: Option<&EntryRecord>, entries: &[EntryRecord]) -> ContextView;   // the rules over scanned entries
pub fn order_tool_results(messages: &[Message]) -> Vec<Message>;    // rules 7 and 8
```

The algorithm, as a table over the captured bounds:

| # | Step |
| --- | --- |
| 1 | `H` = the newest visible entry at or before the tail with `head` |
| 2 | `from` = `H.head`, or the transcript start without `H` |
| 3 | Scan visible entries `from ..= tail` (fork traversal: child entries, then parent entries through each `parent.at` cap), pages of 256 |
| 4 | For each edit target, the newest edit in the range wins: `omit` contributes nothing, `replace` contributes its messages |
| 5 | Context entries are `H` followed by the non-head entries of the range; without `H`, the range |
| 6 | Every positional system message, with its section and tool changes, is kept |
| 9 | Model-less entries and assistant messages with stop reason `aborted`, `error` or `deferred` contribute nothing (`contributions` is the state after this step) |
| 10 | Move the first system message to the front when only user messages precede it, or when it is the baseline's (below) |
| 7 | Move each assistant's tool results, found before the next assistant message, directly after it in tool-call order |
| 8 | Synthesize an error result (`isError`, `details: { reason: "missing_result" }`, text "Tool result unavailable: history ends before this call completed.") for every call without one; drop results without a preceding call |

Steps are numbered as the spec numbers them; delta applies 10 before 7 and 8 because
moving the baseline never crosses an assistant message (so results stay ordered) and the
baseline search needs the per-entry contributions, not the reordered list. The stored
entries and `contributions` keep the committed order.

**Baseline (rule 10).** The baseline is the system message generation wrote with every
section and tool in effect: with `H`, the first `delta.system` entry after `H` (unless an edit
targets it); without `H`, the first `delta.system` entry, unless an earlier entry carries a
system message in its own messages or in an edit's replacement (even one an edit later
omits), in which case there is no baseline. A run's input is committed before generation
renders the prompt, so a context otherwise starts with the input, or with kept history. Only
the first system message moves, so replay yields the same sections and tools; any other
system message with an assistant message before it stays in place.

```rust
fn find_baseline(head: Option<&EntryRecord>, entries: &[EntryRecord]) -> Baseline;   // Some(msg) | None(impossible) | Undecided
fn lead_with_system(messages: Vec<Message>, baseline: &Baseline) -> Vec<Message>;
```

Views carry raw active entries; UI reduction ([observe.md](observe.md)) and model-context
reduction are separate consumers. Older history is read through `Conversation::entries`.

## Kept ranges

```rust
pub struct ContextRanges { /* Mutex<HashMap<ConversationId, Kept { bounds, entries: Arc<[EntryRecord]>, view, used }>> */ }
impl ContextRanges {
    pub async fn read(&self, storage: &dyn Storage, conversation: ConversationId, at: Option<EntryId>,
                      retention: Duration, ctx: &Context) -> Result<ContextView, ContextError>;
}
```

- A session service of the agent layer (inside `AgentServices`) keeps each conversation's
  last range in memory; agent-layer tasks read context through it (`rt.context_view`), hosts
  through `Conversation::context` without it. A later read with the same head marker and a
  newer tail scans only `previous.tail + 1 ..= tail`; an equal tail returns the kept view; a
  shorter tail (a `fork(at)` preview, a request's `cutoff`) filters the kept entries without
  replacing the range; a different head marker rescans.
- The view is derived again from the kept entries plus the added ones, in memory: the scan
  is what the range saves. Upstream's incremental split (`settled` / `open`, a baseline
  searched only among added entries) is an optimization of that derivation and can be added
  behind the same function when profiles ask for it.
- A range unused for `settings.context_retention` (default 10 min) is dropped at the next
  read of any conversation; there is no background sweep. A busy conversation reads its
  context at every request, so its range stays.
- Kept entries are shared (`Arc`) and never mutated; each read returns its own view.
- Bounds are captured from storage directly, not on the session line: a storage commit is
  durable before it is published, and reading the head marker at or before the captured tail
  makes the two reads consistent.

## System prompt and dynamic tools (§7.4)

Durable has no stored prompt sections. The resolved agent produces the desired sections
for each request; the transcript's `delta.system` entries are the only durable record of what
the model saw. Every `delta.system` entry is `model: [SystemMessage { content: "", sections,
tools_added, tools_removed, timestamp }]` (`delta_ai::SystemMessage`; `sections` is an
`IndexMap<String, Option<String>>`, `None` removes).

```rust
/// Sections in effect after replaying system messages: set in place, `None` deletes, re-adding appends.
pub fn replay_sections(messages: &[Message]) -> IndexMap<String, String>;
/// `delta_ai::get_current_tools` replays the offered tools the same way.

/// Render the agent's sections in order. `None` omits one; tagged text is `<key>\n...\n</key>`.
/// A section that fails keeps its shown text, if any, and is reported; a failure after the
/// invocation context is cancelled propagates.
pub async fn render_sections(sections: &[Arc<dyn PromptSection>], input: &PromptInput<'_>, shown: &IndexMap<String, String>,
                             report: impl FnMut(SectionError, &dyn PromptSection), ctx: &Context) -> Result<IndexMap<String, String>, Cancelled>;

/// The `delta.system` entries that make the replayed sections and tools of `view` equal `desired`
/// in values and order. Zero, one or two drafts.
pub fn plan_system_entries(view: &ContextView, desired: &IndexMap<String, String>, tools: &[Tool], now: i64) -> Vec<EntryDraft>;
```

Generation `prepare` ([generation.md](generation.md)) takes these steps against its phase
snapshot: resolve the agent and replay the active transcript into shown sections and offered
tools; the desired tools are `agent.tools` in order; build the environment and render
`agent.sections` with `PromptInput { conversation, agent, env, shown, read }`; compare and
append. Preparation does not recheck the transcript before appending: only the Harness
writes model context to a busy conversation (submissions, run tasks, boundaries, a blocking
compaction that appends only while its generation waits), so the context it read is still
current; display-only writes may land meanwhile and contribute nothing.

Planning rules, in precedence order:

1. **Head rebaseline.** When the active context has a head marker and no `delta.system` entry
   after it (`id > H.id`), the new entry is a complete baseline: every desired section in
   order, every desired tool in `tools_added`, plus `ContextEdit::Omit` for every earlier
   `delta.system` entry still retained after the cut. It is written even when it restates the
   replayed values, so every later preparation finds a `delta.system` entry after the marker. A
   `SystemMessage` is a patch, not a reset: it cannot remove previous `content` or restore
   order by restating values, hence the omissions.
2. **Order mismatch.** If patching values would leave the shown order different from the
   desired order (patched order = shown keys that remain, in shown order, then new keys in
   desired order), one commit appends two entries: the first removes every shown section
   with `None`, the second re-adds every desired section in desired order.
3. **Minimal patch.** Otherwise one entry with the changed values and `None` removals, or
   nothing.
4. **Tools** are planned separately and ride on the last planned entry, or on one entry of
   their own when sections are unchanged. Declarations are compared with
   `delta_ai::declarations_equal` and written as plain `delta_ai::Tool` declarations (no
   application metadata). A changed declaration is removed and re-added in the same message
   (removals apply before additions). Replay keeps retained tools in place and appends
   additions; when that would not yield the desired order, the message removes every offered
   tool and re-adds the desired tools in order. A head rebaseline adds every desired tool.

Replay semantics (what the planner assumes of `delta_ai`'s transcript replay): messages
apply in order; non-empty `content` appends instructions (durable always writes `""`); a
section string adds or replaces in place without moving; `None` removes and a later
re-addition appends; within one message tool removals precede additions, so a same-name
replacement gets the new declaration and position.

**Cache stability.** Renderers must be deterministic for equal inputs: any change in
rendered text (a timestamp, say) appends a delta and invalidates provider prompt caches.
The stored rendered strings stay authoritative even when the renderer changes; delta never
rewrites stored entries for provider compatibility. `delta_ai` decides whether a provider
gets the messages positionally or folded into one leading system message. Model and
thinking level are request options, not prompt state.

## Testing

Ports of `harness-context` (edits, heads, fork caps, tool result ordering, synthesized
results, excluded stop reasons, baseline placement), `harness-prompt` (minimal patch, order
mismatch pair, head rebaseline with omissions, tool add/remove/reorder, failing sections
keeping shown text, `instructions` last), `system-order-cache-e2e` and the kept-range half of
`harness-scheduler-retention` (extend, shrink, head change, retention expiry). The planner
is pure and unit-tested on `ContextView` fixtures; derivation is tested on `MemoryStorage`.

## Differences from upstream

- Derivation applies rule 10 before rules 7 and 8 (same result, see above).
- `replay_sections` and `plan_system_entries` are pure functions over `IndexMap`s, which
  carry insertion order natively; upstream relies on `Map` iteration order.
- The kept range is `Arc`-shared and immutable by construction, so upstream's deep
  `freezeJson` is not needed; the view is re-derived from kept entries instead of extended
  incrementally (see "Kept ranges").
- Section failures are `Result`s, not thrown errors; cancellation is `delta_chord::Context`.
- `delta_ai` provides `declarations_equal` and `get_current_tools` (present) and
  `to_tool_declaration`, the plain-declaration conversion, which is added there.

## Decisions

- System entries are `delta.system`; their `SystemMessage` is `delta_ai`'s, in upstream's
  message shape.
- `to_tool_declaration` is added to `delta_ai`.
- Kept context ranges are an agent-layer session service, not scheduler state.
- `PromptInput` has no document reader in phase 4 (harness.md, "Phase 4").
- Otherwise the module ports §2.1 and §7.4 as specified, including upstream's kept-range
  extension rules; nothing is open.
