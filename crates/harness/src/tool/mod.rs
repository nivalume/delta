//! Tools and the `delta.tool` task (spec §7.3, §8.4): one durable task per call.
//!
//! A [`HarnessTool`] declares a tool and executes it with a [`ToolApi`]. The [`ToolTask`]
//! runs one call, model-issued (created by the generation's tool round through
//! [`create_model_call_task`]) or nested (made by another tool through
//! [`ToolApi::execute_tool`]): it resolves the tool, repairs and validates the arguments,
//! runs the `before_tool` hooks and the approval policy, records its intent (the final
//! arguments and the replay policy) before anything runs, executes with a bounded output
//! window published to its slot in `delta.live` on a throttle, and settles exactly one
//! result. A restart after the intent reruns only a tool that is replay-safe; any other
//! call settles `interrupted` with its durable partial output.
//!
//! Corresponds to upstream `src/harness/tool.ts`, `define.ts` and `truncate.ts`. See
//! `docs/design/tool.md`.

mod api;
mod approval;
mod definition;
mod execute;
mod function;
mod nested;
mod output;
mod pipeline;
mod records;
mod result;
mod settle;
mod task;
mod validate;

use delta_chord::json::{self, Json};

pub use api::{NestedCallOptions, RetainedOutput, ToolApi, ToolApiError};
pub use definition::{
    Callers, DEFAULT_OUTPUT_LIMITS, HarnessTool, ToolError, ToolExecutionMode, ToolReplay, canonical_json,
};
pub use function::FnHarnessTool;
#[cfg(feature = "schemars")]
pub use function::{schema_for, typed_harness_tool};
pub use records::{
    NestedCallDoc, NestedCallState, ParentCall, ToolHookCall, ToolTaskCheckpoint, ToolTaskInput, ToolTaskResult,
};
pub use result::{
    NestedToolExecutionResult, Severity, ToolControl, ToolDiagnostic, ToolExecutionResult, append_tool_result,
    harness_error,
};
pub use task::ToolTask;

use crate::records::{EntryId, TaskId};
use crate::session::{TaskOptions, Tx, TxError};

/// The kind of the tool task, one per call.
pub const TOOL_TASK: &str = "delta.tool";
/// The version of the tool task's records.
pub const TOOL_TASK_VERSION: u32 = 1;
/// The entry kind of a tool result.
pub const TOOL_RESULT_ENTRY: &str = "delta.tool-result";

/// Creates the `delta.tool` task of a model-issued call, owned by the generation `owner`.
/// The seam between a tool round and the tool task.
pub async fn create_model_call_task(
    tx: &mut Tx<'_>,
    owner: TaskId,
    assistant: EntryId,
    call_id: &str,
) -> Result<TaskId, TxError> {
    let input = ToolTaskInput::Model {
        assistant,
        call_id: call_id.to_owned(),
    };
    let task = tx.create_task(&ToolTask, input, TaskOptions::owned_by(owner)).await?;
    Ok(task.id())
}

/// Decodes the stored result of a tool task.
pub fn decode_result(result: &Json) -> Option<ToolTaskResult> {
    json::from_json(result).ok()
}
