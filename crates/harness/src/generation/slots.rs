//! The presentation of the round's calls in `delta.live`: one [`ToolSlot`] per model-issued
//! call in `tools`, one [`NestedToolSlot`] per nested call in `nested_tools`, both carrying the
//! running [`SlotProgress`] their tool task publishes.

use delta_ai::types::{JsonObject, Usage};
use delta_chord::json::Json;
use serde::{Deserialize, Serialize};

use super::live::LiveState;
use crate::records::{EntryId, TaskId};
use crate::tool::ToolDiagnostic;

/// Where a tool call stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotStatus {
    Pending,
    Running,
    Done,
}

/// What a running call published: its retained output window, what the window dropped,
/// its details and its diagnostics so far. Empty once the call is done.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SlotProgress {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dropped_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dropped_lines: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Json>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<ToolDiagnostic>>,
}

/// The presentation of one model-issued call of the current round.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSlot {
    pub call_id: String,
    pub name: String,
    /// Absent for a call not started yet (sequential round) and for a call the request did
    /// not offer, which starts `done`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    pub status: SlotStatus,
    /// Only while `running`.
    #[serde(flatten)]
    pub progress: SlotProgress,
    /// The result entry once done; absent when the tool task faulted or was orphaned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry: Option<EntryId>,
}

impl ToolSlot {
    /// A slot of call `call_id` in status `status`.
    pub fn new(call_id: &str, name: &str, status: SlotStatus) -> Self {
        Self {
            call_id: call_id.to_owned(),
            name: name.to_owned(),
            task_id: None,
            status,
            progress: SlotProgress::default(),
            entry: None,
        }
    }

    /// Done, without the progress it published while running.
    pub fn finish(&mut self) {
        self.status = SlotStatus::Done;
        self.progress = SlotProgress::default();
    }
}

/// How a nested call ended, for presentation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NestedToolSummary {
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Up to 500 characters of its error diagnostics' text, or else of its output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// The presentation of one nested call, below its caller.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NestedToolSlot {
    /// `<parent call id>/<key>`.
    pub call_id: String,
    pub name: String,
    pub task_id: TaskId,
    pub parent_task_id: TaskId,
    pub status: SlotStatus,
    /// The arguments as passed, replaced by the final ones at the intent commit.
    pub arguments: JsonObject,
    #[serde(flatten)]
    pub progress: SlotProgress,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<NestedToolSummary>,
}

impl LiveState {
    /// The round slot of tool task `task`.
    pub fn tool_slot_mut(&mut self, task: TaskId) -> Option<&mut ToolSlot> {
        self.tools.iter_mut().flatten().find(|slot| slot.task_id == Some(task))
    }

    /// The nested slot of tool task `task`.
    pub fn nested_slot_mut(&mut self, task: TaskId) -> Option<&mut NestedToolSlot> {
        self.nested_tools.iter_mut().flatten().find(|slot| slot.task_id == task)
    }

    /// The published progress of tool task `task`, in its round or nested slot.
    pub fn slot_progress_mut(&mut self, task: TaskId) -> Option<&mut SlotProgress> {
        if self.tools.iter().flatten().any(|slot| slot.task_id == Some(task)) {
            return self.tool_slot_mut(task).map(|slot| &mut slot.progress);
        }
        self.nested_slot_mut(task).map(|slot| &mut slot.progress)
    }

    /// Removes the nested slots below `task`, transitively; parents precede their children.
    pub fn remove_nested_below(&mut self, task: TaskId) {
        let Some(nested) = &mut self.nested_tools else {
            return;
        };
        let mut removed = vec![task];
        for slot in nested.iter() {
            if removed.contains(&slot.parent_task_id) {
                removed.push(slot.task_id);
            }
        }
        nested.retain(|slot| !removed.contains(&slot.parent_task_id));
        if nested.is_empty() {
            self.nested_tools = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nested(task: u64, parent: u64) -> NestedToolSlot {
        NestedToolSlot {
            call_id: format!("c/{task}"),
            name: "t".to_owned(),
            task_id: TaskId(task),
            parent_task_id: TaskId(parent),
            status: SlotStatus::Running,
            arguments: JsonObject::new(),
            progress: SlotProgress::default(),
            summary: None,
        }
    }

    #[test]
    fn nested_slots_are_removed_transitively() {
        let mut live = LiveState {
            nested_tools: Some(vec![nested(2, 1), nested(3, 2), nested(4, 9), nested(5, 3)]),
            ..LiveState::default()
        };
        live.remove_nested_below(TaskId(1));
        let left: Vec<TaskId> = live.nested_tools.iter().flatten().map(|slot| slot.task_id).collect();
        assert_eq!(left, vec![TaskId(4)]);
        live.remove_nested_below(TaskId(9));
        assert_eq!(live.nested_tools, None);
    }

    #[test]
    fn slot_progress_is_flat() {
        let mut slot = ToolSlot::new("c1", "read", SlotStatus::Running);
        slot.progress.output = Some("hi".to_owned());
        assert_eq!(
            serde_json::to_value(&slot).unwrap(),
            serde_json::json!({ "call_id": "c1", "name": "read", "status": "running", "output": "hi" })
        );
        let back: ToolSlot = serde_json::from_value(serde_json::to_value(&slot).unwrap()).unwrap();
        assert_eq!(back, slot);
    }
}
