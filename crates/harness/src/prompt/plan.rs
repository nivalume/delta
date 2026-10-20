//! Replaying and planning positional system messages.

use std::collections::HashSet;

use delta_ai::transcript::{declarations_equal, get_current_tools};
use delta_ai::types::{Message, SystemContent, SystemMessage, Tool, ToolReference};
use indexmap::IndexMap;

use crate::context::ContextView;
use crate::records::{ContextEdit, EditAction, EntryDraft};

/// The entry kind of a positional system message.
pub const SYSTEM_ENTRY: &str = "delta.system";

/// The sections in effect after replaying the system messages of `messages`: a string sets
/// in place, `None` deletes, a re-addition appends.
pub fn replay_sections(messages: &[Message]) -> IndexMap<String, String> {
    let mut shown = IndexMap::new();
    for message in messages {
        let Message::System(system) = message else { continue };
        for (key, value) in system.sections.iter().flatten() {
            match value {
                Some(text) => {
                    shown.insert(key.clone(), text.clone());
                }
                None => {
                    shown.shift_remove(key);
                }
            }
        }
    }
    shown
}

/// Tool changes of one system message; removals apply before additions.
#[derive(Default)]
struct ToolChanges {
    removed: Vec<ToolReference>,
    added: Vec<Tool>,
}

impl ToolChanges {
    fn is_empty(&self) -> bool {
        self.removed.is_empty() && self.added.is_empty()
    }
}

type Patch = IndexMap<String, Option<String>>;

/// The `delta.system` entries that make the replayed sections and tools of `view` equal
/// `desired` and `tools`, in values and order: zero, one or two drafts.
///
/// 1. A head marker without a later `delta.system` entry: one complete baseline (every
///    section, every tool) that omits every retained earlier `delta.system` entry, written
///    even when it restates the replayed values.
/// 2. A minimal patch that would leave another order: remove every shown section, then
///    re-add every desired section in order.
/// 3. Otherwise the minimal patch of changed values and `None` removals, or nothing.
///
/// Tool changes ride on the last planned entry, or on one entry of their own.
pub fn plan_system_entries(
    view: &ContextView,
    desired: &IndexMap<String, String>,
    tools: &[Tool],
    timestamp: i64,
) -> Vec<EntryDraft> {
    if let Some(head) = &view.head
        && !view
            .entries
            .iter()
            .any(|entry| entry.kind == SYSTEM_ENTRY && entry.id > head.id)
    {
        let edits: Vec<ContextEdit> = view
            .entries
            .iter()
            .filter(|entry| entry.kind == SYSTEM_ENTRY)
            .map(|entry| ContextEdit {
                target: entry.id,
                action: EditAction::Omit,
            })
            .collect();
        let sections: Patch = desired
            .iter()
            .map(|(key, text)| (key.clone(), Some(text.clone())))
            .collect();
        let changes = ToolChanges {
            removed: Vec::new(),
            added: tools.to_vec(),
        };
        let mut baseline = system_entry(Some(sections), Some(changes), timestamp);
        if !edits.is_empty() {
            baseline.edits = Some(edits);
        }
        return vec![baseline];
    }
    let sections = plan_sections(&replay_sections(&view.messages), desired);
    let changes = plan_tools(&get_current_tools(&view.messages), tools);
    if changes.is_empty() {
        return sections
            .into_iter()
            .map(|patch| system_entry(Some(patch), None, timestamp))
            .collect();
    }
    if sections.is_empty() {
        return vec![system_entry(None, Some(changes), timestamp)];
    }
    let last = sections.len() - 1;
    let mut changes = Some(changes);
    sections
        .into_iter()
        .enumerate()
        .map(|(index, patch)| {
            let tools = if index == last { changes.take() } else { None };
            system_entry(Some(patch), tools, timestamp)
        })
        .collect()
}

/// Tool changes from `offered` to `desired`. A changed declaration is removed and re-added;
/// when replay would not yield the desired order, every offered tool is removed and every
/// desired tool re-added in order.
fn plan_tools(offered: &[Tool], desired: &[Tool]) -> ToolChanges {
    let wanted: IndexMap<&str, &Tool> = desired.iter().map(|tool| (tool.name.as_str(), tool)).collect();
    let kept: Vec<&Tool> = offered
        .iter()
        .filter(|tool| {
            wanted
                .get(tool.name.as_str())
                .is_some_and(|next| declarations_equal(tool, next))
        })
        .collect();
    let kept_names: HashSet<&str> = kept.iter().map(|tool| tool.name.as_str()).collect();
    let added: Vec<&Tool> = desired
        .iter()
        .filter(|tool| !kept_names.contains(tool.name.as_str()))
        .collect();
    let replayed = kept.iter().chain(added.iter()).map(|tool| tool.name.as_str());
    let in_order = replayed.eq(desired.iter().map(|tool| tool.name.as_str()));
    let reference = |tool: &Tool| ToolReference {
        name: tool.name.clone(),
    };
    if !in_order {
        return ToolChanges {
            removed: offered.iter().map(reference).collect(),
            added: desired.to_vec(),
        };
    }
    ToolChanges {
        removed: offered
            .iter()
            .filter(|tool| !kept_names.contains(tool.name.as_str()))
            .map(reference)
            .collect(),
        added: added.into_iter().cloned().collect(),
    }
}

/// Section patches: none, the minimal patch, or the remove-all / re-add-all pair when the
/// patched order would differ from the desired one.
fn plan_sections(shown: &IndexMap<String, String>, desired: &IndexMap<String, String>) -> Vec<Patch> {
    let patched = shown
        .keys()
        .filter(|key| desired.contains_key(*key))
        .chain(desired.keys().filter(|key| !shown.contains_key(*key)));
    if !patched.eq(desired.keys()) {
        let removals: Patch = shown.keys().map(|key| (key.clone(), None)).collect();
        let additions: Patch = desired
            .iter()
            .map(|(key, text)| (key.clone(), Some(text.clone())))
            .collect();
        return vec![removals, additions];
    }
    let mut patch = Patch::new();
    for (key, text) in shown {
        let next = desired.get(key);
        if next != Some(text) {
            patch.insert(key.clone(), next.cloned());
        }
    }
    for (key, text) in desired {
        if !shown.contains_key(key) {
            patch.insert(key.clone(), Some(text.clone()));
        }
    }
    if patch.is_empty() { Vec::new() } else { vec![patch] }
}

fn system_entry(sections: Option<Patch>, tools: Option<ToolChanges>, timestamp: i64) -> EntryDraft {
    let tools = tools.unwrap_or_default();
    let message = SystemMessage {
        content: SystemContent::Text(String::new()),
        sections,
        tools_added: (!tools.added.is_empty()).then_some(tools.added),
        tools_removed: (!tools.removed.is_empty()).then_some(tools.removed),
        timestamp,
    };
    EntryDraft {
        model: Some(vec![Message::System(message)]),
        ..EntryDraft::new(SYSTEM_ENTRY)
    }
}
