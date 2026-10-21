//! Derives the agent events of one view revision (observe.md, "Derivation"): the view before
//! and after, and the publication's records, become one batch in upstream's order.

use std::collections::{HashMap, HashSet};

use delta_ai::types::{AssistantBlock, AssistantMessage, JsonObject, Message};
use delta_chord::delta::{Op, diff, overlap};
use delta_chord::json::{self, Json, Seg};
use serde::de::DeserializeOwned;

use super::events::{AgentEvent, MessageChange, OutputChange, QueuedItem, Snapshot, ToolEventCall};
use super::view::{ConversationView, Revision};
use crate::agent::{AgentDoc, AgentState};
use crate::approval::{ApprovalsDoc, ApprovalsState};
use crate::documents::DocDefinition;
use crate::generation::{
    ASSISTANT_ENTRY, GENERATION_TASK, LiveDoc, LiveState, NestedToolSlot, SlotProgress, SlotStatus, ToolSlot,
};
use crate::inbox::{InboxDoc, InboxState};
use crate::records::{
    ConversationId, EntryRecord, SubmissionRecord, TaskId, TaskOutcome, TaskRecord, TaskState, TaskStatus,
};
use crate::session::CommitChange;
use crate::tool::{NestedCallDoc, NestedCallState, NestedToolExecutionResult, TOOL_RESULT_ENTRY};
use crate::usage::{UsageDoc, UsageState};

/// The documents events read, typed from a view value.
#[derive(Default)]
struct Docs {
    live: LiveState,
    inbox: InboxState,
    agent: Option<Json>,
    usage: Option<Json>,
    approvals: ApprovalsState,
}

/// A mounted document of `view`.
fn doc(view: &Json, kind: &str) -> Option<Json> {
    view.get("docs").and_then(|docs| docs.get(kind)).cloned()
}

/// A value as `T`; the default when absent or not a `T`.
fn typed<T: DeserializeOwned + Default>(value: Option<&Json>) -> T {
    value.and_then(|value| json::from_json(value).ok()).unwrap_or_default()
}

impl Docs {
    fn of(view: &Json) -> Self {
        Self {
            live: typed(doc(view, LiveDoc::KIND).as_ref()),
            inbox: typed(doc(view, InboxDoc::KIND).as_ref()),
            agent: doc(view, AgentDoc::KIND),
            usage: doc(view, UsageDoc::KIND),
            approvals: typed(doc(view, ApprovalsDoc::KIND).as_ref()),
        }
    }
}

/// The snapshot of a view value.
pub(crate) fn snapshot(view: &Json) -> Snapshot {
    let docs = Docs::of(view);
    let entries = json::from_json::<ConversationView>(view)
        .map(|view| view.entries)
        .unwrap_or_default();
    let live = docs.live;
    Snapshot {
        entries,
        run: live.run,
        generation: live.generation,
        tools: live.tools,
        nested_tools: live.nested_tools,
        compactions: live.compactions,
        compaction_failure: live.compaction_failure,
        inbox: docs.inbox.items.iter().map(QueuedItem::from).collect(),
        agent: typed(docs.agent.as_ref()),
        usage: typed(docs.usage.as_ref()),
        approvals: pending_requests(&docs.approvals),
    }
}

fn pending_requests(state: &ApprovalsState) -> Vec<crate::approval::ApprovalRequest> {
    state
        .requests
        .iter()
        .filter(|request| request.decision.is_none())
        .cloned()
        .collect()
}

/// What one publication carried for this conversation.
struct Records<'a> {
    entries: Vec<&'a EntryRecord>,
    tasks: Vec<&'a TaskRecord>,
    submissions: Vec<&'a SubmissionRecord>,
    nested: HashMap<TaskId, NestedToolExecutionResult>,
}

impl<'a> Records<'a> {
    fn of(revision: &'a Revision<'_>, conversation: ConversationId) -> Self {
        let mut records = Self {
            entries: Vec::new(),
            tasks: Vec::new(),
            submissions: Vec::new(),
            nested: HashMap::new(),
        };
        for change in &revision.publication.changes {
            match change {
                CommitChange::Entry(entry) if entry.conversation_id == conversation => records.entries.push(entry),
                CommitChange::Task(task) if task.conversation_id == conversation => records.tasks.push(task),
                CommitChange::Submission(submission) if submission.conversation_id == conversation => {
                    records.submissions.push(submission);
                }
                CommitChange::Document {
                    record,
                    conversation_id: Some(id),
                    value: Some(value),
                    ..
                } if *id == conversation && record.kind == NestedCallDoc::KIND => {
                    if let Ok(NestedCallState {
                        task_id: Some(task),
                        result: Some(result),
                        ..
                    }) = json::from_json::<NestedCallState>(value)
                    {
                        records.nested.insert(task, result);
                    }
                }
                _ => {}
            }
        }
        records.entries.sort_by_key(|entry| entry.id);
        records.submissions.sort_by_key(|submission| submission.id);
        records
    }

    /// The result entry this commit appended for `call_id`.
    fn result_entry(&self, call_id: &str) -> Option<&'a EntryRecord> {
        self.entries.iter().copied().find(|entry| {
            entry.kind == TOOL_RESULT_ENTRY
                && entry.model.iter().flatten().any(|message| match message {
                    Message::ToolResult(result) => result.tool_call_id == call_id,
                    _ => false,
                })
        })
    }

    /// The final arguments of tool task `task`, from its intent checkpoint in this commit.
    fn intent_arguments(&self, task: Option<TaskId>) -> JsonObject {
        let checkpoint = self
            .tasks
            .iter()
            .find(|record| Some(record.id) == task)
            .and_then(|record| record.state.checkpoint());
        checkpoint
            .filter(|checkpoint| checkpoint.get("phase").and_then(Json::as_str) == Some("execute"))
            .and_then(|checkpoint| checkpoint.get("arguments"))
            .and_then(|arguments| json::from_json(arguments).ok())
            .unwrap_or_default()
    }
}

/// The event state of one stream: generations whose turn already ended while they held
/// their outcome (`completing`), so their terminal state does not end it again.
pub(crate) struct Translator {
    conversation: ConversationId,
    held: HashSet<TaskId>,
}

/// A slot of the round or a nested one, as events see it.
struct Slot<'a> {
    call: ToolEventCall,
    status: SlotStatus,
    progress: &'a SlotProgress,
}

impl Translator {
    pub(crate) fn new(conversation: ConversationId, held: HashSet<TaskId>) -> Self {
        Self { conversation, held }
    }

    /// The events of `revision`, in upstream's batch order.
    pub(crate) fn translate(&mut self, revision: &Revision<'_>) -> Vec<AgentEvent> {
        let before = Docs::of(revision.before);
        let after = Docs::of(revision.after);
        let records = Records::of(revision, self.conversation);
        let before_slots = slots(&before.live);
        let after_slots = slots(&after.live);
        let mut out = Vec::new();

        // Tool starts.
        for (key, slot) in &after_slots {
            let started = before_slots
                .iter()
                .find(|(before, _)| before == key)
                .is_none_or(|(_, before)| before.status != SlotStatus::Running);
            if slot.status == SlotStatus::Running && started {
                let args = match key {
                    SlotKey::Round(_) => records.intent_arguments(slot.call.task_id),
                    SlotKey::Nested(task) => nested_arguments(&after.live, *task),
                };
                out.push(AgentEvent::ToolExecutionStart {
                    call: slot.call.clone(),
                    args,
                });
            }
        }

        // The streamed partial.
        let before_generation = before.live.generation.as_ref();
        let after_generation = after.live.generation.as_ref();
        let before_partial = before_generation.and_then(|generation| generation.message.as_ref());
        let after_partial = after_generation.and_then(|generation| generation.message.as_ref());
        let same_attempt = before_generation.map(|generation| generation.attempt)
            == after_generation.map(|generation| generation.attempt);
        match (before_partial.filter(|_| same_attempt), after_partial) {
            (None, Some(partial)) => {
                if let Ok(message) = json::from_json::<AssistantMessage>(partial) {
                    out.push(AgentEvent::MessageStart {
                        message: Message::Assistant(message),
                    });
                }
            }
            (Some(before), Some(after)) if before != after => {
                if let Some(event) = message_update(before, after) {
                    out.push(event);
                }
            }
            _ => {}
        }

        // Tool updates.
        for (key, slot) in &after_slots {
            let Some((_, before)) = before_slots.iter().find(|(before, _)| before == key) else {
                continue;
            };
            if slot.status != SlotStatus::Running || before.status != SlotStatus::Running {
                continue;
            }
            if let Some(event) = tool_update(slot, before.progress) {
                out.push(event);
            }
        }

        // Retry and deferred polls.
        let before_retry = before_generation.and_then(|generation| generation.retry.as_ref());
        let after_retry = after_generation.and_then(|generation| generation.retry.as_ref());
        match (before_retry, after_retry) {
            (None, Some(retry)) => out.push(AgentEvent::AutoRetryStart {
                attempt: after_generation.map_or(0, |generation| generation.attempt),
                at: retry.at,
                error_message: retry.error.clone(),
            }),
            (Some(_), None) => out.push(AgentEvent::AutoRetryEnd {
                attempt: before_generation.map_or(0, |generation| generation.attempt),
            }),
            _ => {}
        }
        let poll_at = |generation: Option<&crate::generation::LiveGeneration>| {
            generation
                .and_then(|generation| generation.deferred)
                .map(|deferred| deferred.poll_at)
        };
        if let Some(at) = poll_at(after_generation)
            && poll_at(before_generation) != Some(at)
        {
            out.push(AgentEvent::DeferredPoll { poll_at: at });
        }

        // Tool ends: nested ones first (children before parents), then the round's with
        // their result entries, then those without one.
        let mut ended = Vec::new();
        for (key, slot) in before_slots.iter().chain(after_slots.iter()).rev() {
            if ended.iter().any(|(done, _): &(SlotKey, ToolEventCall)| done == key) {
                continue;
            }
            let before = before_slots
                .iter()
                .find(|(before, _)| before == key)
                .map(|(_, slot)| slot.status);
            let after = after_slots
                .iter()
                .find(|(after, _)| after == key)
                .map(|(_, slot)| slot.status);
            let ends = match (before, after) {
                (Some(before), Some(SlotStatus::Done) | None) => before != SlotStatus::Done,
                (None, Some(SlotStatus::Done)) => true,
                _ => false,
            };
            if ends {
                ended.push((key.clone(), slot.call.clone()));
            }
        }
        let mut round_ends = Vec::new();
        for (key, call) in ended {
            match key {
                SlotKey::Nested(task) => out.push(AgentEvent::ToolExecutionEnd {
                    call,
                    entry: None,
                    result: records.nested.get(&task).cloned(),
                }),
                SlotKey::Round(_) => round_ends.push(call),
            }
        }
        round_ends.reverse();
        let streamed = before_partial.is_some();
        for entry in &records.entries {
            if entry.kind == TOOL_RESULT_ENTRY
                && let Some(index) = round_ends.iter().position(|call| {
                    records
                        .result_entry(&call.tool_call_id)
                        .is_some_and(|found| found.id == entry.id)
                })
            {
                let call = round_ends.remove(index);
                out.push(AgentEvent::ToolExecutionEnd {
                    call,
                    entry: Some((*entry).clone()),
                    result: None,
                });
            }
            match entry.model.as_deref() {
                Some([first, ..]) => {
                    if !(entry.kind == ASSISTANT_ENTRY && streamed) {
                        out.push(AgentEvent::MessageStart { message: first.clone() });
                    }
                    out.push(AgentEvent::MessageEnd {
                        entry: (*entry).clone(),
                    });
                }
                _ => out.push(AgentEvent::EntryAppended {
                    entry: (*entry).clone(),
                }),
            }
        }
        for call in round_ends {
            out.push(AgentEvent::ToolExecutionEnd {
                call,
                entry: None,
                result: None,
            });
        }

        // Compaction ends, failures, turn and run ends.
        let failure = after.live.compaction_failure.as_ref();
        for status in before.live.compactions.iter().flatten() {
            let left = !after
                .live
                .compactions
                .iter()
                .flatten()
                .any(|after| after.task_id == status.task_id);
            if left {
                out.push(AgentEvent::CompactionEnd {
                    task_id: status.task_id,
                    reason: status.reason,
                    error: failure
                        .filter(|failure| failure.task_id == status.task_id)
                        .map(|failure| failure.error.clone()),
                });
            }
        }
        for task in &records.tasks {
            let failed = match &task.state {
                TaskState::Terminal {
                    outcome: TaskOutcome::Faulted { error },
                } => Some(error.message.clone()),
                TaskState::Terminal {
                    outcome: TaskOutcome::Orphaned { reason },
                } => Some(reason.clone()),
                _ => None,
            };
            if let Some(message) = failed {
                out.push(AgentEvent::TaskFailed {
                    task_id: task.id,
                    kind: task.kind.clone(),
                    message,
                });
            }
        }
        if self.turn_ended(&records) {
            out.push(AgentEvent::TurnEnd);
        }
        let before_run = before.live.run.as_ref();
        let after_run = after.live.run.as_ref();
        let run_changed = match (before_run, after_run) {
            (Some(before), Some(after)) => before.inputs.first() != after.inputs.first(),
            (None, None) => false,
            _ => true,
        };
        if run_changed && let Some(run) = before_run {
            out.push(AgentEvent::RunEnd {
                inputs: run.inputs.clone(),
            });
        }

        // Submissions.
        for record in &records.submissions {
            out.push(AgentEvent::Submission {
                record: (*record).clone(),
            });
        }

        // Documents.
        if before.inbox != after.inbox {
            out.push(AgentEvent::InboxUpdate {
                items: after.inbox.items.iter().map(QueuedItem::from).collect(),
            });
        }
        if before.agent != after.agent {
            let agent: AgentState = typed(after.agent.as_ref());
            out.push(AgentEvent::AgentChanged { agent });
        }
        if before.usage != after.usage {
            let usage: UsageState = typed(after.usage.as_ref());
            out.push(AgentEvent::UsageChanged { usage });
        }
        if before.approvals != after.approvals {
            out.push(AgentEvent::ApprovalUpdate {
                requests: pending_requests(&after.approvals),
            });
        }

        // Starts.
        for status in after.live.compactions.iter().flatten() {
            let joined = !before
                .live
                .compactions
                .iter()
                .flatten()
                .any(|before| before.task_id == status.task_id);
            if joined {
                out.push(AgentEvent::CompactionStart {
                    task_id: status.task_id,
                    reason: status.reason,
                    blocking: status.blocking,
                });
            }
        }
        if run_changed && let Some(run) = after_run {
            out.push(AgentEvent::RunStart {
                inputs: run.inputs.clone(),
            });
        }
        let successor = after_run
            .map(|run| run.task_id)
            .filter(|task| before_run.map(|run| run.task_id) != Some(*task));
        let created = successor.is_some_and(|task| {
            records.tasks.iter().any(|record| {
                record.id == task && record.kind == GENERATION_TASK && record.state.status() == TaskStatus::Pending
            })
        });
        if created {
            out.push(AgentEvent::TurnStart);
        }
        out
    }

    /// Whether a generation's turn ended in this commit: one that holds its outcome for the
    /// first time, or one that went terminal without having held it. Once per commit.
    fn turn_ended(&mut self, records: &Records<'_>) -> bool {
        let mut ended = false;
        for task in records.tasks.iter().filter(|task| task.kind == GENERATION_TASK) {
            match task.state.status() {
                TaskStatus::Completing => ended |= self.held.insert(task.id),
                TaskStatus::Terminal => ended |= !self.held.remove(&task.id),
                _ => {}
            }
        }
        ended
    }
}

/// A slot's identity across revisions: a round slot by call id, a nested slot by task.
#[derive(Clone, Debug, PartialEq, Eq)]
enum SlotKey {
    Round(String),
    Nested(TaskId),
}

/// Every slot of `live`: the round's in call order, then the nested ones (parents first).
fn slots(live: &LiveState) -> Vec<(SlotKey, Slot<'_>)> {
    let round = live.tools.iter().flatten().map(|slot: &ToolSlot| {
        let call = ToolEventCall {
            tool_call_id: slot.call_id.clone(),
            tool_name: slot.name.clone(),
            task_id: slot.task_id,
            parent_tool_call_id: None,
            parent_task_id: None,
        };
        (
            SlotKey::Round(slot.call_id.clone()),
            Slot {
                call,
                status: slot.status,
                progress: &slot.progress,
            },
        )
    });
    let nested = live.nested_tools.iter().flatten().map(|slot: &NestedToolSlot| {
        let call = ToolEventCall {
            tool_call_id: slot.call_id.clone(),
            tool_name: slot.name.clone(),
            task_id: Some(slot.task_id),
            parent_tool_call_id: parent_call_id(live, slot.parent_task_id),
            parent_task_id: Some(slot.parent_task_id),
        };
        (
            SlotKey::Nested(slot.task_id),
            Slot {
                call,
                status: slot.status,
                progress: &slot.progress,
            },
        )
    });
    round.chain(nested).collect()
}

fn parent_call_id(live: &LiveState, parent: TaskId) -> Option<String> {
    let round = live.tools.iter().flatten().find(|slot| slot.task_id == Some(parent));
    let nested = live.nested_tools.iter().flatten().find(|slot| slot.task_id == parent);
    round
        .map(|slot| slot.call_id.clone())
        .or_else(|| nested.map(|slot| slot.call_id.clone()))
}

fn nested_arguments(live: &LiveState, task: TaskId) -> JsonObject {
    live.nested_tools
        .iter()
        .flatten()
        .find(|slot| slot.task_id == task)
        .map(|slot| slot.arguments.clone())
        .unwrap_or_default()
}

/// A running slot's update: its output change, details and diagnostics when they differ.
fn tool_update(slot: &Slot<'_>, before: &SlotProgress) -> Option<AgentEvent> {
    let after = slot.progress;
    let output = (before.output != after.output).then(|| {
        output_change(
            before.output.as_deref().unwrap_or_default(),
            after.output.as_deref().unwrap_or_default(),
        )
    });
    let details = (before.details != after.details).then(|| after.details.clone().unwrap_or(Json::Null));
    let diagnostics = (before.diagnostics != after.diagnostics).then(|| after.diagnostics.clone().unwrap_or_default());
    if output.is_none() && details.is_none() && diagnostics.is_none() {
        return None;
    }
    Some(AgentEvent::ToolExecutionUpdate {
        call: slot.call.clone(),
        output,
        details,
        diagnostics,
    })
}

/// The window's change: a trim and an append when the new window continues the old one (a
/// growing head, a sliding tail), else the whole window.
fn output_change(before: &str, after: &str) -> OutputChange {
    if before.is_empty() {
        return OutputChange::Delta {
            trim_start: None,
            append: Some(after.to_owned()),
        };
    }
    let shared = overlap(before, after, before.len());
    if shared == 0 {
        return OutputChange::Set {
            output: after.to_owned(),
        };
    }
    let trimmed = &before[..before.len() - shared];
    let units = trimmed.encode_utf16().count();
    let appended = &after[shared..];
    OutputChange::Delta {
        trim_start: (units > 0).then(|| u64::try_from(units).unwrap_or(u64::MAX)),
        append: (!appended.is_empty()).then(|| appended.to_owned()),
    }
}

/// `message_update` for a partial that changed: the diff of the two partials translated into
/// block starts, text and thinking deltas, tool call argument deltas, whole blocks, or the
/// whole message. Only usage changed: no event.
fn message_update(before: &Json, after: &Json) -> Option<AgentEvent> {
    let message: AssistantMessage = json::from_json(after).ok()?;
    let mut changes = Vec::new();
    let mut whole_blocks: Vec<usize> = Vec::new();
    let mut whole_message = false;
    let block = |index: usize| -> Option<AssistantBlock> { message.content.get(index).cloned() };
    for op in diff(before, after) {
        let path: Vec<Seg> = op.path().to_vec();
        match path.as_slice() {
            [Seg::Key(key), ..] if &**key == "usage" => {}
            [Seg::Key(key)] if &**key == "content" => match &op {
                Op::Splice {
                    index,
                    remove: 0,
                    items,
                    ..
                } => {
                    for offset in 0..items.len() {
                        let content_index = index + offset;
                        if let Some(block) = block(content_index) {
                            changes.push(start(content_index, block));
                        }
                    }
                }
                _ => whole_message = true,
            },
            [Seg::Key(key), Seg::Index(index), rest @ ..] if &**key == "content" => {
                let index = *index;
                if whole_blocks.contains(&index) {
                    continue;
                }
                match (&op, rest) {
                    (Op::Append(_, text), [Seg::Key(field)]) if &**field == "text" => {
                        changes.push(MessageChange::TextDelta {
                            content_index: index,
                            delta: text.to_string(),
                        });
                    }
                    (Op::Append(_, text), [Seg::Key(field)]) if &**field == "thinking" => {
                        changes.push(MessageChange::ThinkingDelta {
                            content_index: index,
                            delta: text.to_string(),
                        });
                    }
                    (Op::Append(_, text), [Seg::Key(field), below @ ..]) if &**field == "arguments" => {
                        changes.push(MessageChange::ToolcallDelta {
                            content_index: index,
                            path: below.iter().map(seg_json).collect(),
                            delta: text.to_string(),
                        });
                    }
                    _ => {
                        whole_blocks.push(index);
                        if let Some(block) = block(index) {
                            changes.push(MessageChange::Block {
                                content_index: index,
                                block,
                            });
                        }
                    }
                }
            }
            _ => whole_message = true,
        }
    }
    if whole_message {
        changes = vec![MessageChange::Message {
            message: message.clone(),
        }];
    }
    if changes.is_empty() {
        return None;
    }
    Some(AgentEvent::MessageUpdate {
        usage: message.usage,
        changes,
    })
}

fn start(content_index: usize, block: AssistantBlock) -> MessageChange {
    match block {
        AssistantBlock::Text(_) => MessageChange::TextStart { content_index, block },
        AssistantBlock::Thinking(_) => MessageChange::ThinkingStart { content_index, block },
        AssistantBlock::ToolCall(_) => MessageChange::ToolcallStart { content_index, block },
    }
}

fn seg_json(seg: &Seg) -> Json {
    match seg {
        Seg::Key(key) => Json::from(&**key),
        Seg::Index(index) => Json::from(u64::try_from(*index).unwrap_or(u64::MAX)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_changes_follow_the_window() {
        assert_eq!(
            output_change("", "ab"),
            OutputChange::Delta {
                trim_start: None,
                append: Some("ab".to_owned())
            }
        );
        assert_eq!(
            output_change("abc", "abcde"),
            OutputChange::Delta {
                trim_start: None,
                append: Some("de".to_owned())
            }
        );
        assert_eq!(
            output_change("abcdef", "cdefgh"),
            OutputChange::Delta {
                trim_start: Some(2),
                append: Some("gh".to_owned())
            }
        );
        assert_eq!(
            output_change("abc", "xyz"),
            OutputChange::Set {
                output: "xyz".to_owned()
            }
        );
    }

    #[test]
    fn partials_become_deltas() {
        let empty = AssistantMessage::empty("fake", "fake", "m");
        let partial = |text: &str| {
            let mut message = empty.clone();
            message
                .content
                .push(AssistantBlock::Text(delta_ai::types::TextContent::new(text)));
            json::to_json(&message).unwrap()
        };
        let Some(AgentEvent::MessageUpdate { changes, .. }) = message_update(&partial("Hel"), &partial("Hello")) else {
            panic!("an update")
        };
        assert_eq!(
            changes,
            [MessageChange::TextDelta {
                content_index: 0,
                delta: "lo".to_owned()
            }]
        );
    }
}
