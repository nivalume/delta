//! The derivation rules over scanned entries (spec §2.1, steps 1–10).

use std::collections::HashMap;

use delta_ai::types::{Message, StopReason};

use super::ContextView;
use super::order::order_tool_results;
use crate::prompt::SYSTEM_ENTRY;
use crate::records::{ContextEdit, EditAction, EntryId, EntryRecord};

/// The baseline's system message: the one generation wrote with every section and tool in
/// effect, which leads the context.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Baseline {
    Found(Message),
    /// An earlier entry carries a system message of its own: there is no baseline.
    Impossible,
    /// No entry decided it yet.
    Undecided,
}

/// The view of `entries` (the visible range from `head.head`, or the transcript start,
/// through the tail, oldest first) under the head marker `head`.
pub fn derive_view(head: Option<&EntryRecord>, entries: &[EntryRecord]) -> ContextView {
    // Rule 4: per target the newest edit in the range wins, including the edits of older
    // head markers the active selection drops.
    let mut edits: HashMap<EntryId, &ContextEdit> = HashMap::new();
    for entry in entries {
        for edit in entry.edits.iter().flatten() {
            edits.insert(edit.target, edit);
        }
    }
    let active = select_active(head, entries);
    let contributions: Vec<Vec<Message>> = active
        .iter()
        .map(|entry| contribute(entry, edits.get(&entry.id).copied()))
        .collect();
    let baseline = find_baseline(head, &active);
    let led = lead_with_system(contributions.concat(), &baseline);
    ContextView {
        head: head.cloned(),
        entries: active,
        contributions,
        messages: order_tool_results(&led),
    }
}

/// Rule 5: the head marker followed by the range's non-head entries; without one, the range.
fn select_active(head: Option<&EntryRecord>, entries: &[EntryRecord]) -> Vec<EntryRecord> {
    match head {
        None => entries.to_vec(),
        Some(head) => std::iter::once(head.clone())
            .chain(entries.iter().filter(|entry| entry.head.is_none()).cloned())
            .collect(),
    }
}

/// Rules 4 and 9: one active entry's messages after its edit, without assistant messages
/// that ended `aborted`, `error` or `deferred`.
fn contribute(entry: &EntryRecord, edit: Option<&ContextEdit>) -> Vec<Message> {
    let messages = match edit.map(|edit| &edit.action) {
        Some(EditAction::Omit) => return Vec::new(),
        Some(EditAction::Replace(messages)) => messages.as_slice(),
        None => entry.model.as_deref().unwrap_or_default(),
    };
    messages
        .iter()
        .filter(|message| match message {
            Message::Assistant(assistant) => !matches!(
                assistant.stop_reason,
                StopReason::Aborted | StopReason::Error | StopReason::Deferred
            ),
            _ => true,
        })
        .cloned()
        .collect()
}

/// The baseline among `entries`, oldest first: with a head marker, the system message of the
/// first `delta.system` entry after it; without one, of the first `delta.system` entry,
/// unless an earlier entry carries a system message in its messages or in an edit's
/// replacement (even one an edit later omits).
pub(super) fn find_baseline(head: Option<&EntryRecord>, entries: &[EntryRecord]) -> Baseline {
    for entry in entries {
        if head.is_some_and(|head| entry.id <= head.id) {
            continue;
        }
        if entry.kind == SYSTEM_ENTRY {
            let system = entry
                .model
                .iter()
                .flatten()
                .find(|message| matches!(message, Message::System(_)));
            return system.map_or(Baseline::Impossible, |message| Baseline::Found(message.clone()));
        }
        if head.is_none() && carries_system_message(entry) {
            return Baseline::Impossible;
        }
    }
    Baseline::Undecided
}

fn carries_system_message(entry: &EntryRecord) -> bool {
    let is_system = |message: &Message| matches!(message, Message::System(_));
    entry.model.iter().flatten().any(is_system)
        || entry.edits.iter().flatten().any(|edit| match &edit.action {
            EditAction::Replace(messages) => messages.iter().any(is_system),
            EditAction::Omit => false,
        })
}

/// Rule 10: moves the first system message to the front when only user messages precede
/// it, or when it is the baseline's. Only the first moves, so replay yields the same
/// sections and tools.
pub(super) fn lead_with_system(mut messages: Vec<Message>, baseline: &Baseline) -> Vec<Message> {
    let Some(index) = messages
        .iter()
        .position(|message| matches!(message, Message::System(_)))
    else {
        return messages;
    };
    if index == 0 {
        return messages;
    }
    let is_baseline = matches!(baseline, Baseline::Found(found) if *found == messages[index]);
    let only_users = messages[..index]
        .iter()
        .all(|message| matches!(message, Message::User(_)));
    if is_baseline || only_users {
        let system = messages.remove(index);
        messages.insert(0, system);
    }
    messages
}
