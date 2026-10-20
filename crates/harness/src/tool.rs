//! Tools as the generation sees them: the [`HarnessTool`] declaration side, the results and
//! controls a tool round reads, and the `delta.tool` task's input and result shapes.
//!
//! Phase 4 of the crate holds what the generation's tool rounds need: resolving and
//! offering tools, creating one `delta.tool` task per model-issued call through
//! [`create_model_call_task`], writing harness error results, and reading each call's
//! [`ToolTaskResult`]. The `delta.tool` task itself (validation, hooks, approval, execution
//! with output windows, nested calls) and `HarnessTool::execute` arrive with phase 5; until
//! then a host installs its own `delta.tool` definition. See `docs/design/tool.md`.

use std::any::Any;
use std::fmt;

use delta_ai::types::{ToolCall, ToolResultMessage, Usage, UserBlock};
use delta_chord::json::{self, Json};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::records::ConversationId;
use crate::records::{EntryDraft, EntryId, EntryRecord, TaskId};
use crate::session::{TaskCreate, TaskOptions, Tx, TxError};
use crate::usage::{UsageBucket, record_usage};

/// The kind of the tool task, one per call.
pub const TOOL_TASK: &str = "delta.tool";
/// The version of the tool task's records.
pub const TOOL_TASK_VERSION: u32 = 1;
/// The entry kind of a tool result.
pub const TOOL_RESULT_ENTRY: &str = "delta.tool-result";

/// Whether a round's calls run at once or one after the other.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionMode {
    #[default]
    Parallel,
    Sequential,
}

/// Who may call a tool: the model, other tools (nested calls), or both.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Callers(u8);

impl Callers {
    pub const MODEL: Self = Self(1);
    pub const TOOLS: Self = Self(2);
    pub const BOTH: Self = Self(3);

    /// Whether every caller of `other` is one of these.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

/// A tool an extension provides. Phase 4 declares what resolution and the generation read;
/// phase 5 adds execution (`execute`, replay policy, output limits, approval key).
pub trait HarnessTool: Send + Sync {
    /// The declaration offered to the model: name, description, JSON Schema parameters.
    fn definition(&self) -> &delta_ai::types::Tool;

    /// The tool's name, its declaration's.
    fn name(&self) -> &str {
        &self.definition().name
    }

    /// `None`: the settings' `tool_execution`. `Sequential` makes a round that calls the
    /// tool sequential.
    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        None
    }

    fn callers(&self) -> Callers {
        Callers::BOTH
    }

    /// A deferred tool is offered only once loaded (`model_tools` names it in `add`).
    fn deferred(&self) -> bool {
        false
    }

    /// Application metadata, downcast by the application's own sections.
    fn extra(&self) -> Option<&dyn Any> {
        None
    }
}

impl fmt::Debug for dyn HarnessTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HarnessTool")
            .field("name", &self.name())
            .finish_non_exhaustive()
    }
}

/// How serious a diagnostic is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Warn,
    Error,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
        }
    }
}

/// A model-visible note about a call: truncation, a corrected path, an exit code, a
/// harness error.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDiagnostic {
    pub severity: Severity,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// What a model-issued call asks of its round (spec §8.5).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolControl {
    /// Tools to enable (and load, when deferred) for the next request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub add_tools: Vec<String>,
    /// End the run once every call of the round asks to.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub terminate: bool,
    /// End the run with a reset whose user message is this text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handoff: Option<String>,
}

/// A call's result through its three channels: output for the model, structured output for
/// programs, details for UIs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolExecutionResult {
    pub output: Option<Vec<UserBlock>>,
    pub structured_output: Option<Value>,
    pub is_error: bool,
    pub details: Option<Value>,
    pub diagnostics: Vec<ToolDiagnostic>,
    pub usage: Option<Usage>,
    pub control: Option<ToolControl>,
}

/// A harness-written error result: no output, one `error` diagnostic with `code`
/// (`tool_unavailable`, `aborted`, ...).
pub fn harness_error(code: &str, message: impl Into<String>) -> ToolExecutionResult {
    ToolExecutionResult {
        output: Some(Vec::new()),
        is_error: true,
        diagnostics: vec![ToolDiagnostic {
            severity: Severity::Error,
            message: message.into(),
            code: Some(code.to_owned()),
        }],
        ..ToolExecutionResult::default()
    }
}

/// The input of a `delta.tool` task. Phase 4 creates model-issued calls only.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolTaskInput {
    /// The call `call_id` of the assistant entry `assistant`.
    Model { assistant: EntryId, call_id: String },
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
    /// A nested call; its result is in the caller's task documents.
    Nested,
}

/// Creates the `delta.tool` task of a model-issued call, owned by the generation `owner`,
/// at its first phase `{"phase":"call"}` (tool.md). The seam between a tool round and the
/// tool task: the round knows the task only by kind, version and input.
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
    let create = TaskCreate {
        kind: TOOL_TASK.to_owned(),
        version: TOOL_TASK_VERSION,
        input: json::to_json(&input).map_err(TxError::callback)?,
        checkpoint: delta_chord::json!({ "phase": "call" }),
    };
    tx.create_task_record(create, TaskOptions::owned_by(owner)).await
}

/// Appends a call's `delta.tool-result` entry: the result's output with its diagnostics
/// rendered as one trailing text item, `data: { diagnostics }`; its usage goes to
/// `delta.usage.tools[name]` in the same commit.
pub async fn append_tool_result(
    tx: &mut Tx<'_>,
    conversation: ConversationId,
    call: &ToolCall,
    result: &ToolExecutionResult,
    timestamp: i64,
    duration_ms: Option<u64>,
) -> Result<EntryRecord, TxError> {
    let mut content = result.output.clone().unwrap_or_default();
    if !result.diagnostics.is_empty() {
        content.push(UserBlock::text(render_diagnostics(&result.diagnostics)));
    }
    let message = ToolResultMessage {
        tool_call_id: call.id.clone(),
        tool_name: call.name.clone(),
        content,
        details: result.details.clone(),
        usage: result.usage.clone(),
        nested_calls: None,
        is_error: result.is_error,
        timestamp,
        duration_ms,
    };
    if let Some(usage) = &result.usage {
        record_usage(tx, conversation, UsageBucket::Tools, &call.name, usage).await?;
    }
    let data = delta_chord::json!({ "diagnostics": json::to_json(&result.diagnostics).map_err(TxError::callback)? });
    let draft = EntryDraft {
        kind: TOOL_RESULT_ENTRY.to_owned(),
        model: Some(vec![message.into()]),
        data: Some(data),
        head: None,
        edits: None,
    };
    tx.append_entry(conversation, draft).await
}

/// `<harness>\n[severity] message\n...\n</harness>`.
fn render_diagnostics(diagnostics: &[ToolDiagnostic]) -> String {
    let lines: Vec<String> = diagnostics
        .iter()
        .map(|diagnostic| format!("[{}] {}", diagnostic.severity.as_str(), diagnostic.message))
        .collect();
    format!("<harness>\n{}\n</harness>", lines.join("\n"))
}

/// Decodes the stored result of a tool task.
pub fn decode_result(result: &Json) -> Option<ToolTaskResult> {
    json::from_json(result).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_input_and_result_are_kind_tagged() {
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
    }

    #[test]
    fn diagnostics_render_as_one_harness_block() {
        let result = harness_error("aborted", "Tool read was aborted");
        assert_eq!(
            render_diagnostics(&result.diagnostics),
            "<harness>\n[error] Tool read was aborted\n</harness>"
        );
        assert!(Callers::BOTH.contains(Callers::MODEL));
        assert!(!Callers::TOOLS.contains(Callers::MODEL));
    }
}
