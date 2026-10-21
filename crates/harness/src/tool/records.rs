//! The `delta.tool` task's stored shapes: input, checkpoint and result, the call hooks see, and
//! the `delta.tool.nested-call` documents a caller keeps its nested calls in.

use delta_ai::types::{JsonObject, ToolCall};
use delta_chord::delta::Op;
use delta_chord::json::Json;
use serde::{Deserialize, Serialize};

use super::definition::ToolReplay;
use super::result::{NestedToolExecutionResult, ToolControl};
use crate::approval::ApprovalId;
use crate::documents::{CheckpointInfo, DocDefinition, TaskScope};
use crate::records::{EntryId, TaskId};

/// The input of a `delta.tool` task.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolTaskInput {
    /// The call `call_id` of the assistant entry `assistant`.
    Model { assistant: EntryId, call_id: String },
    /// A call another tool made through `ToolApi::execute_tool`.
    Nested {
        /// The calling tool task.
        parent: TaskId,
        parent_call_id: String,
        /// The key of the call among the caller's nested calls.
        key: String,
        /// The call as made: id `<parent_call_id>/<key>`, the arguments as passed.
        call: ToolCall,
        /// `Some(false)`: nothing is published while it runs.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        progress: Option<bool>,
    },
}

/// The phases of a `delta.tool` task.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "lowercase")]
pub enum ToolTaskCheckpoint {
    /// Resolution, repair, validation, hooks and approval, then intent and execution.
    Call,
    /// Waiting for the decision on a committed approval request until `deadline`.
    Approval {
        id: ApprovalId,
        arguments: JsonObject,
        replay: ToolReplay,
        deadline: i64,
    },
    /// Durable intent: the final arguments and the replay policy, recorded before execution.
    /// Reached as a phase only by recovery.
    Execute { arguments: JsonObject, replay: ToolReplay },
}

impl crate::scheduler::Checkpoint for ToolTaskCheckpoint {
    fn phase(&self) -> &str {
        match self {
            Self::Call => "call",
            Self::Approval { .. } => "approval",
            Self::Execute { .. } => "execute",
        }
    }
}

/// The result of a `delta.tool` task.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolTaskResult {
    /// A model-issued call: its result entry and, when it completed, its control.
    Model {
        entry_id: EntryId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        control: Option<ToolControl>,
    },
    /// A nested call; its result is in the caller's `delta.tool.nested-call` member.
    Nested,
}

/// The tool call that issued a nested call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParentCall {
    pub task_id: TaskId,
    pub call_id: String,
}

/// A call as hooks and the approval policy see it: the call with its current arguments and,
/// for a nested call, the call that made it.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolHookCall {
    pub task_id: TaskId,
    pub call: ToolCall,
    pub parent: Option<ParentCall>,
}

/// One nested call of a tool task: a task-scoped family of the caller keyed by the call's
/// key, retired with the caller.
#[derive(Clone, Copy, Debug, Default)]
pub struct NestedCallDoc;

impl DocDefinition for NestedCallDoc {
    const KIND: &'static str = "delta.tool.nested-call";
    const VERSION: u32 = 1;
    const FAMILY: bool = true;
    type Scope = TaskScope;
    type Seed = ();

    fn initial((): &()) -> Json {
        delta_chord::json!({})
    }

    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        false
    }
}

/// The value of a [`NestedCallDoc`] member.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NestedCallState {
    /// The nested `delta.tool` task, once created.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    /// The tool name and arguments the call was made with; a retry must match them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<JsonObject>,
    /// Written by the nested task as it settles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<NestedToolExecutionResult>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_are_tagged() {
        let input = ToolTaskInput::Model {
            assistant: EntryId(4),
            call_id: "c1".to_owned(),
        };
        assert_eq!(
            serde_json::to_value(&input).unwrap(),
            serde_json::json!({ "kind": "model", "assistant": 4, "call_id": "c1" })
        );
        let result = ToolTaskResult::Model {
            entry_id: EntryId(9),
            control: Some(ToolControl {
                terminate: true,
                ..ToolControl::default()
            }),
        };
        assert_eq!(
            serde_json::to_value(&result).unwrap(),
            serde_json::json!({ "kind": "model", "entry_id": 9, "control": { "terminate": true } })
        );
        assert_eq!(
            serde_json::to_value(ToolTaskCheckpoint::Call).unwrap(),
            serde_json::json!({ "phase": "call" })
        );
        let execute = ToolTaskCheckpoint::Execute {
            arguments: JsonObject::new(),
            replay: ToolReplay::Safe,
        };
        assert_eq!(
            serde_json::to_value(&execute).unwrap(),
            serde_json::json!({ "phase": "execute", "arguments": {}, "replay": "safe" })
        );
    }
}
