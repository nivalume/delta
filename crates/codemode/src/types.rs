//! What a run reports and what a host answers: output items, call records, errors, results,
//! and the calls and outcomes of deterministic runs. Serialized as upstream's
//! `CodemodeResult` (camelCase).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Characters of JSON one stored value may have.
pub const MAX_STORE_VALUE_CHARS: usize = 256 * 1024;
/// Characters of keys and JSON all stored values may have together.
pub const MAX_STORE_TOTAL_CHARS: usize = 1024 * 1024;
/// Characters of text and base64 image data one script may output.
pub const MAX_OUTPUT_CHARS: usize = 16 * 1024 * 1024;
/// `text()`, `image()` and `console.*` calls one script may make.
pub const MAX_OUTPUT_ITEMS: usize = 100_000;

/// One item of the script's output, in the order the script produced it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum OutputItem {
    /// From `text()`, or from `console.*` when `console` is set.
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        console: bool,
    },
    /// From `image()`; `data` is base64.
    #[serde(rename_all = "camelCase")]
    Image { data: String, mime_type: String },
}

/// How a tool call of the script ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CallStatus {
    Ok,
    Error,
    /// Still running when the script ended, or never answered.
    Cancelled,
}

/// One tool call of the script; globals are not recorded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallRecord {
    pub name: String,
    pub status: CallStatus,
    pub duration_ms: f64,
}

/// Why a run failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ErrorKind {
    /// The script threw or failed to parse; `name` and `stack` come from its error.
    Script,
    /// The deadline expired or the interrupt budget ran out.
    Timeout,
    /// The caller aborted the run.
    Aborted,
    /// The VM or the bridge failed outside the script's control.
    Sandbox,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodemodeError {
    pub kind: ErrorKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
}

impl CodemodeError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            name: None,
            message: message.into(),
            stack: None,
        }
    }
}

/// Keys a successful script changed with `store()`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StoreWrites {
    pub set: Map<String, Value>,
    /// Keys stored as `undefined`.
    pub delete: Vec<String>,
}

/// In a deterministic run the script ended itself (returned, threw or exited): the result being
/// processed when it did (0: its first run) and the calls issued by then.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct During {
    pub seq: u64,
    pub issued: u64,
}

/// How a run ended. `output` and `calls` are kept for failed runs too; `exit()` succeeds with no
/// value. The default is a successful run without output.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodemodeResult {
    /// `None`: it succeeded. The returned value is `value`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<CodemodeError>,
    /// What the script returned; `None` for `undefined`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
    pub output: Vec<OutputItem>,
    pub calls: Vec<CallRecord>,
    /// Only a successful run reports writes.
    #[serde(default)]
    pub store_writes: StoreWrites,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub during: Option<During>,
}

impl CodemodeResult {
    pub fn is_ok(&self) -> bool {
        self.error.is_none()
    }

    pub(crate) fn failed(error: CodemodeError) -> Self {
        Self {
            error: Some(error),
            value: None,
            output: Vec::new(),
            calls: Vec::new(),
            store_writes: StoreWrites::default(),
            during: None,
        }
    }
}

/// Whether a call is of `tools.<name>` or of a global.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CallTarget {
    Tool,
    Global,
}

/// A call the script made in a deterministic run, answered with `Run::deliver`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HostCall {
    /// The VM's own number for the call, 1, 2, … in the order the script made its calls.
    pub id: u64,
    pub target: CallTarget,
    pub name: String,
    /// What the script passed, after a JSON round trip; for a spread global the argument
    /// list; `None` for `undefined`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Value>,
}

/// A call's answer. A rejection becomes an `Error` with `message` in the script, carrying the
/// properties (except `name`, `message` and `stack`), for example `{ structuredOutput }`.
/// Serialized as upstream's `{ ok: true, value? }` / `{ ok: false, message, properties? }`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "OutcomeJson", into = "OutcomeJson")]
pub enum Outcome {
    Ok {
        /// `None` resolves with `undefined`.
        value: Option<Value>,
    },
    Err {
        message: String,
        properties: Option<Map<String, Value>>,
    },
}

#[derive(Serialize, Deserialize)]
struct OutcomeJson {
    ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    value: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    properties: Option<Map<String, Value>>,
}

impl TryFrom<OutcomeJson> for Outcome {
    type Error = String;

    fn try_from(json: OutcomeJson) -> Result<Self, String> {
        if json.ok {
            return Ok(Self::Ok { value: json.value });
        }
        let message = json.message.ok_or("a failed outcome needs a message")?;
        Ok(Self::Err {
            message,
            properties: json.properties,
        })
    }
}

impl From<Outcome> for OutcomeJson {
    fn from(outcome: Outcome) -> Self {
        match outcome {
            Outcome::Ok { value } => Self {
                ok: true,
                value,
                message: None,
                properties: None,
            },
            Outcome::Err { message, properties } => Self {
                ok: false,
                value: None,
                message: Some(message),
                properties,
            },
        }
    }
}

impl Outcome {
    pub fn ok(value: impl Into<Option<Value>>) -> Self {
        Self::Ok { value: value.into() }
    }

    pub fn err(message: impl Into<String>) -> Self {
        Self::Err {
            message: message.into(),
            properties: None,
        }
    }

    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok { .. })
    }
}

/// A tool as a deterministic run's script sees it: `tools.<js_name>` and its `ALL_TOOLS` entry.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeclaredTool {
    pub name: String,
    /// The identifier the script uses; `to_codemode_identifier(name)` when `None`. A host that
    /// replays a run passes the one the run started with.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub js_name: Option<String>,
    #[serde(default)]
    pub description: String,
}

/// A global as a script sees it: an identifier, or `namespace.member`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclaredGlobal {
    pub name: String,
    /// The call receives the whole argument list instead of the first argument.
    #[serde(default)]
    pub spread: bool,
}
