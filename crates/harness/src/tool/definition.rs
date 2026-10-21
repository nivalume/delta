//! The [`HarnessTool`] trait and what it declares: callers, execution mode, replay policy and
//! output limits.

use std::any::Any;
use std::error::Error as StdError;
use std::fmt;

use delta_ai::types::{JsonObject, Tool};
use delta_chord::context::Context;
use delta_chord::output::{OutputLimits, Retain};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::api::ToolApi;
use super::result::ToolExecutionResult;
use crate::env::ExecutionEnv;

/// An error a tool returns: the call ends `failed` with a `tool_error` result.
pub type ToolError = Box<dyn StdError + Send + Sync + 'static>;

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

/// Whether a call whose intent was recorded may run again after a restart.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolReplay {
    /// Running it again from the stored arguments is harmless.
    Safe,
    /// A restart settles the call `interrupted` with its durable partial output.
    #[default]
    Unsafe,
}

/// How a call reaches a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathAccess {
    Read,
    Write,
}

/// What permission rules match a call on (`docs/design/permissions.md`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PermissionSubject {
    /// A shell command, as the model wrote it.
    Command { command: String },
    /// An absolute path the call reads or writes.
    Path { path: String, access: PathAccess },
}

/// The output window a tool keeps when it declares none: the first 2000 lines, at most
/// 50 KiB.
pub const DEFAULT_OUTPUT_LIMITS: OutputLimits = OutputLimits {
    max_bytes: 50 * 1024,
    max_lines: 2000,
    retain: Retain::Head,
};

/// A tool an extension provides.
///
/// Mirrors `delta_agent::AgentTool` (`definition`, `execution_mode`, `prepare_arguments`, a
/// boxed `execute`) and adds what durability and approval need: `callers`, `replay`,
/// `output_limits`, the structured output schema and `approval_key`. [`FnHarnessTool`] and
/// [`typed_harness_tool`] are the convenience forms.
///
/// [`FnHarnessTool`]: super::FnHarnessTool
/// [`typed_harness_tool`]: super::typed_harness_tool
pub trait HarnessTool: Send + Sync {
    /// The declaration offered to the model: name, description, JSON Schema parameters.
    fn definition(&self) -> &Tool;

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

    /// Whether a restart after the intent commit runs the call again.
    fn replay(&self) -> ToolReplay {
        ToolReplay::Unsafe
    }

    /// The running output window.
    fn output_limits(&self) -> OutputLimits {
        DEFAULT_OUTPUT_LIMITS
    }

    /// The JSON Schema `structured_output` must match; a tool without one returns none.
    fn structured_output_schema(&self) -> Option<&Value> {
        None
    }

    /// Repairs commonly malformed arguments before validation; an error makes them invalid.
    fn prepare_arguments(&self, arguments: JsonObject) -> Result<JsonObject, ToolError> {
        Ok(arguments)
    }

    /// What permission rules match the call on: a command, or a path it reads or writes,
    /// made absolute with `env` when there is one. `None`: only rules naming the tool.
    fn permission_subject(&self, arguments: &JsonObject, env: Option<&dyn ExecutionEnv>) -> Option<PermissionSubject> {
        let _ = (arguments, env);
        None
    }

    /// The key a session approval grant matches; `None`: never granted for the session.
    fn approval_key(&self, arguments: &JsonObject) -> Option<String> {
        Some(canonical_json(arguments))
    }

    /// Runs the call with its final arguments.
    fn execute<'a>(
        &'a self,
        arguments: JsonObject,
        api: &'a ToolApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<ToolExecutionResult, ToolError>>;

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

/// `value` as JSON with every object's keys sorted, so equal values print equally.
pub fn canonical_json(value: &JsonObject) -> String {
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(object) => {
                let mut keys: Vec<&String> = object.keys().collect();
                keys.sort();
                Value::Object(
                    keys.into_iter()
                        .map(|key| (key.clone(), sorted(&object[key])))
                        .collect(),
                )
            }
            Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
            other => other.clone(),
        }
    }
    sorted(&Value::Object(value.clone())).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_sorts_nested_keys() {
        let value = |json: Value| match json {
            Value::Object(object) => object,
            _ => JsonObject::new(),
        };
        let a = value(serde_json::json!({ "b": 1, "a": { "y": [1, { "d": 2, "c": 3 }], "x": null } }));
        let b = value(serde_json::json!({ "a": { "x": null, "y": [1, { "c": 3, "d": 2 }] }, "b": 1 }));
        assert_eq!(canonical_json(&a), canonical_json(&b));
        assert_eq!(canonical_json(&b), r#"{"a":{"x":null,"y":[1,{"c":3,"d":2}]},"b":1}"#);
        assert!(Callers::BOTH.contains(Callers::MODEL));
        assert!(!Callers::TOOLS.contains(Callers::MODEL));
    }
}
