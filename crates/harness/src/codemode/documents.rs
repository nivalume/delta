//! The run, journal and store documents of code mode, and their stored shapes.

use delta_ai::types::UserBlock;
use delta_chord::delta::Op;
use delta_chord::json::Json;
use delta_codemode::{DeclaredGlobal, Outcome};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::documents::{CheckpointInfo, ConversationScope, DocDefinition, ForkAsOf, Rewindable, TaskScope};

/// One tool as the script sees it: `tools.<js_name>`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartTool {
    pub name: String,
    pub js_name: String,
    /// Its description and declaration, as `describeTool()` answers.
    pub description: String,
    /// The tool declares a structured output schema, so an error result with a value
    /// resolves with it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub structured: bool,
    /// An internal tool of the extension (`delta.codemode.*`), reached only through a global;
    /// never in `tools`, `ALL_TOOLS` or search.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hidden: bool,
}

/// A namespace of tools as `describeNamespace()` and `searchTools()` see it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartNamespace {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// The names of its tools.
    pub tools: Vec<String>,
}

/// What a call starts with; a rerun starts the VM from exactly this.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodemodeStart {
    /// [`CODEMODE_MODULE_VERSION`](super::CODEMODE_MODULE_VERSION) of the first run.
    pub version: u32,
    /// The VM clock's start, Harness clock ms.
    pub started_at: i64,
    pub seed: u32,
    /// The VM's fingerprint, recorded by the first run before any call is admitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<String>,
    /// `started_at + timeout_ms` when the script set one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline: Option<i64>,
    pub tools: Vec<StartTool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub namespaces: Vec<StartNamespace>,
    pub globals: Vec<DeclaredGlobal>,
    /// `max_output_tokens` of the options line, or the default.
    pub max_output_tokens: u64,
    /// The conversation's store as the script found it.
    pub store: Map<String, Value>,
}

/// A call's outcome as stored: what the transcript gets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredResult {
    pub output: Vec<UserBlock>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

/// The value of a [`CodemodeRunDoc`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RunState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<CodemodeStart>,
    /// The directory of the result's files, created once: a rerun writes the same files again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_dir: Option<String>,
    /// The outcome, decided once.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<StoredResult>,
}

/// The `delta.codemode.run` document: task scope, the start data and the outcome.
#[derive(Clone, Copy, Debug, Default)]
pub struct CodemodeRunDoc;

impl DocDefinition for CodemodeRunDoc {
    const KIND: &'static str = "delta.codemode.run";
    const VERSION: u32 = 1;
    type Scope = TaskScope;
    type Seed = ();

    fn initial((): &()) -> Json {
        delta_chord::json!({})
    }

    // The document retires with its task.
    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        false
    }
}

/// A global's call and outcome; a tool's result is its nested call's.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JournaledGlobal {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Value>,
    pub outcome: Outcome,
}

/// One result delivered to the VM.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JournalEntry {
    /// The VM's call id. Its result is the entry's position + 1 (`seq`).
    pub id: u64,
    /// The VM clock while it processes this result.
    pub at: i64,
    /// Calls the VM had issued after processing this result; absent until acknowledged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub issued_after: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global: Option<JournaledGlobal>,
}

/// The value of a [`CodemodeJournalDoc`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JournalState {
    #[serde(default)]
    pub entries: Vec<JournalEntry>,
    /// Calls the VM had issued after the script's first run, before any result; absent until
    /// acknowledged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_issued: Option<u64>,
}

/// The `delta.codemode.journal` document: task scope, one entry per delivered result.
#[derive(Clone, Copy, Debug, Default)]
pub struct CodemodeJournalDoc;

impl DocDefinition for CodemodeJournalDoc {
    const KIND: &'static str = "delta.codemode.journal";
    const VERSION: u32 = 1;
    type Scope = TaskScope;
    type Seed = ();

    fn initial((): &()) -> Json {
        delta_chord::json!({ "entries": [] })
    }

    // The document retires with its task.
    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        false
    }
}

/// The value of a [`CodemodeStoreDoc`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StoreState {
    #[serde(default)]
    pub values: Map<String, Value>,
}

/// The `delta.codemode.store` document: what scripts keep with `store()`, per conversation.
/// Rewindable and copied as of the fork point, so a fork sees the values as they were when its
/// entry was committed.
#[derive(Clone, Copy, Debug, Default)]
pub struct CodemodeStoreDoc;

impl DocDefinition for CodemodeStoreDoc {
    const KIND: &'static str = "delta.codemode.store";
    const VERSION: u32 = 1;
    type Scope = ConversationScope<Rewindable, ForkAsOf>;
    type Seed = ();

    fn initial((): &()) -> Json {
        delta_chord::json!({ "values": {} })
    }

    fn checkpoint_when(_: &Json, _: &[Op], info: CheckpointInfo) -> bool {
        info.deltas_since_base >= 64
    }
}

/// How a tool call of a script ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallStatus {
    Running,
    Ok,
    Error,
    /// Still running when the script ended.
    Cancelled,
}

impl CallStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Ok => "ok",
            Self::Error => "error",
            Self::Cancelled => "cancelled",
        }
    }
}

/// One tool call a script made, as the call's details show it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodemodeNestedCall {
    /// The VM's number for the call; its nested call id is `<call id>/c<id>`.
    pub id: u64,
    pub name: String,
    /// Compact JSON of the arguments, cut to 200 characters.
    pub args: String,
    pub status: CallStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// Error text, cut to 500 characters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// The `codemode` call's details: its tool calls in the order it made them. Calls to globals
/// are not listed. While the script runs, UIs show its nested calls from the live document's
/// nested tool slots, so the list is written once, with the result.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodemodeToolDetails {
    pub calls: Vec<CodemodeNestedCall>,
    /// The file with the complete text output, when the output was cut to its token budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_shapes_are_snake_case_and_omit_empty_fields() {
        let entry = JournalEntry {
            id: 2,
            at: 10,
            issued_after: Some(3),
            global: Some(JournaledGlobal {
                name: "describeTool".to_owned(),
                args: None,
                outcome: Outcome::ok(None),
            }),
        };
        assert_eq!(
            serde_json::to_value(&entry).unwrap(),
            serde_json::json!({ "id": 2, "at": 10, "issued_after": 3, "global": { "name": "describeTool", "outcome": { "ok": true } } })
        );
        let tool = StartTool {
            name: "my-tool".to_owned(),
            js_name: "my_tool".to_owned(),
            description: String::new(),
            structured: true,
            hidden: false,
        };
        assert_eq!(
            serde_json::to_value(&tool).unwrap(),
            serde_json::json!({ "name": "my-tool", "js_name": "my_tool", "description": "", "structured": true })
        );
    }
}
