//! A call's result through its three channels (output for the model, structured output for
//! programs, details for UIs), its diagnostics and controls, and the `delta.tool-result` entry.

use delta_ai::types::{ToolCall, ToolResultMessage, Usage, UserBlock};
use delta_chord::json;
use delta_chord::output::Retain;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::TOOL_RESULT_ENTRY;
use crate::records::{ConversationId, EntryDraft, EntryRecord, TaskId};
use crate::session::{Tx, TxError};
use crate::usage::{UsageBucket, record_usage};

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

impl ToolDiagnostic {
    pub fn new(severity: Severity, message: impl Into<String>) -> Self {
        Self {
            severity,
            message: message.into(),
            code: None,
        }
    }

    /// With the machine-readable `code`.
    #[must_use]
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// An `error` diagnostic with `code`.
    pub fn error(code: &str, message: impl Into<String>) -> Self {
        Self::new(Severity::Error, message).with_code(code)
    }
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
    /// For the model; `None`: the retained output window.
    pub output: Option<Vec<UserBlock>>,
    /// For programs (nested callers), matching the tool's structured output schema.
    pub structured_output: Option<Value>,
    pub is_error: bool,
    /// For UIs; `None`: the last value reported through `ToolApi::details`.
    pub details: Option<Value>,
    pub diagnostics: Vec<ToolDiagnostic>,
    /// The execution's own spend; recorded in `delta.usage.tools`.
    pub usage: Option<Usage>,
    /// Applies to model-issued calls only.
    pub control: Option<ToolControl>,
}

impl ToolExecutionResult {
    /// A result whose output is `text`.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            output: Some(vec![UserBlock::text(text.into())]),
            ..Self::default()
        }
    }

    /// An error result whose output is `text`.
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            is_error: true,
            ..Self::text(text)
        }
    }
}

/// A nested call's result as its caller gets it: no output, no control.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NestedToolExecutionResult {
    pub task_id: TaskId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_output: Option<Value>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<ToolDiagnostic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

impl NestedToolExecutionResult {
    /// The empty result of task `task_id`.
    pub fn new(task_id: TaskId) -> Self {
        Self {
            task_id,
            structured_output: None,
            is_error: false,
            details: None,
            diagnostics: Vec::new(),
            usage: None,
            duration_ms: None,
        }
    }
}

/// A harness-written error result: no output, one `error` diagnostic with `code`
/// (`tool_unavailable`, `invalid_arguments`, `blocked`, `interrupted`, `aborted`,
/// `abandoned`, `tool_error`).
pub fn harness_error(code: &str, message: impl Into<String>) -> ToolExecutionResult {
    ToolExecutionResult {
        output: Some(Vec::new()),
        is_error: true,
        diagnostics: vec![ToolDiagnostic::error(code, message)],
        ..ToolExecutionResult::default()
    }
}

/// The `truncated` warning of output a window dropped.
pub(super) fn truncated(retain: Retain, dropped_lines: u64, dropped_bytes: u64) -> ToolDiagnostic {
    let end = match retain {
        Retain::Head => "beginning",
        Retain::Tail => "end",
    };
    ToolDiagnostic::new(
        Severity::Warn,
        format!("Output truncated to its {end}: {dropped_lines} lines, {dropped_bytes} bytes dropped"),
    )
    .with_code("truncated")
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
pub(super) fn render_diagnostics(diagnostics: &[ToolDiagnostic]) -> String {
    let lines: Vec<String> = diagnostics
        .iter()
        .map(|diagnostic| format!("[{}] {}", diagnostic.severity.as_str(), diagnostic.message))
        .collect();
    format!("<harness>\n{}\n</harness>", lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_render_as_one_harness_block() {
        let result = harness_error("aborted", "Tool read was aborted");
        assert_eq!(
            render_diagnostics(&result.diagnostics),
            "<harness>\n[error] Tool read was aborted\n</harness>"
        );
        assert_eq!(
            truncated(Retain::Tail, 3, 120).message,
            "Output truncated to its end: 3 lines, 120 bytes dropped"
        );
    }

    #[test]
    fn nested_results_omit_empty_fields() {
        let result = NestedToolExecutionResult {
            structured_output: Some(Value::from("ok")),
            ..NestedToolExecutionResult::new(TaskId(7))
        };
        assert_eq!(
            serde_json::to_value(&result).unwrap(),
            serde_json::json!({ "task_id": 7, "structured_output": "ok" })
        );
    }
}
