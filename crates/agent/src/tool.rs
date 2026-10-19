//! Tools the agent can execute.

use futures::future::BoxFuture;
use serde_json::Value;
use std::fmt;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use tau_ai::{JsonObject, Tool};

use crate::types::{AgentToolResult, ToolExecutionMode};

/// Error a tool returns when it fails; the model receives the message as an error result.
pub type ToolError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Streams partial results while a tool runs. Calls after `execute` returns are ignored.
#[derive(Clone)]
pub struct ToolUpdates {
    sender: Option<tokio::sync::mpsc::UnboundedSender<AgentToolResult>>,
}

impl fmt::Debug for ToolUpdates {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolUpdates").finish_non_exhaustive()
    }
}

impl ToolUpdates {
    pub(crate) fn new(sender: tokio::sync::mpsc::UnboundedSender<AgentToolResult>) -> Self {
        Self { sender: Some(sender) }
    }

    /// Updates that go nowhere, for calling a tool directly.
    pub fn none() -> Self {
        Self { sender: None }
    }

    /// Report a partial result.
    pub fn send(&self, partial: AgentToolResult) {
        if let Some(sender) = &self.sender {
            let _ = sender.send(partial);
        }
    }
}

/// What a tool's `execute` receives besides its arguments.
#[derive(Clone, Debug)]
pub struct ToolContext {
    /// Id of the tool call.
    pub tool_call_id: String,
    /// The run's cancellation; tools honor it for blocking work.
    pub cancel: CancellationToken,
    /// Partial result channel.
    pub updates: ToolUpdates,
}

/// Recovery policy for an effect whose durable intent exists but whose outcome is unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolReplay {
    Never,
    Safe,
}

/// A tool the agent runtime executes.
pub trait AgentTool: Send + Sync {
    /// Declaration sent to the model: name, description, JSON Schema of the arguments,
    /// constrained sampling.
    fn definition(&self) -> &Tool;

    /// Human-readable label for UIs. Default: the tool name.
    fn label(&self) -> &str {
        &self.definition().name
    }

    /// Per-tool execution mode. `Sequential` makes the whole batch sequential.
    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        None
    }

    /// JSON Schema of `structured_content` in successful results.
    fn output_schema(&self) -> Option<&Value> {
        None
    }

    fn replay(&self) -> Option<ToolReplay> {
        None
    }

    /// Compatibility shim for raw arguments, applied before schema validation.
    fn prepare_arguments(&self, arguments: JsonObject) -> JsonObject {
        arguments
    }

    /// Run the tool with validated arguments. Return `Err` on failure, or a result with
    /// `is_error`; do not only describe a failure in the content.
    fn execute(&self, arguments: Value, context: ToolContext) -> BoxFuture<'_, Result<AgentToolResult, ToolError>>;
}

/// Closure behind a [`FnTool`].
pub type ToolFn =
    Arc<dyn Fn(Value, ToolContext) -> BoxFuture<'static, Result<AgentToolResult, ToolError>> + Send + Sync>;

/// A tool from a declaration and a closure.
#[derive(Clone)]
pub struct FnTool {
    definition: Tool,
    label: String,
    execution_mode: Option<ToolExecutionMode>,
    execute: ToolFn,
}

impl fmt::Debug for FnTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FnTool")
            .field("name", &self.definition.name)
            .finish_non_exhaustive()
    }
}

impl FnTool {
    pub fn new<F, Fut>(definition: Tool, execute: F) -> Self
    where
        F: Fn(Value, ToolContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<AgentToolResult, ToolError>> + Send + 'static,
    {
        Self {
            label: definition.name.clone(),
            definition,
            execution_mode: None,
            execute: Arc::new(move |arguments, context| Box::pin(execute(arguments, context))),
        }
    }

    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    #[must_use]
    pub fn with_execution_mode(mut self, mode: ToolExecutionMode) -> Self {
        self.execution_mode = Some(mode);
        self
    }
}

impl AgentTool for FnTool {
    fn definition(&self) -> &Tool {
        &self.definition
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        self.execution_mode
    }

    fn execute(&self, arguments: Value, context: ToolContext) -> BoxFuture<'_, Result<AgentToolResult, ToolError>> {
        (self.execute)(arguments, context)
    }
}

/// JSON Schema of `T`, in the shape tool schemas take: inline subschemas, no `$schema` or
/// root `title`.
#[cfg(feature = "schemars")]
pub fn schema_for<T: schemars::JsonSchema>() -> Value {
    let generator = schemars::generate::SchemaSettings::draft07()
        .with(|settings| settings.inline_subschemas = true)
        .into_generator();
    let mut schema = generator.into_root_schema_for::<T>().to_value();
    if let Value::Object(object) = &mut schema {
        object.remove("$schema");
        object.remove("title");
        object.remove("definitions");
    }
    schema
}

/// A tool whose arguments are a Rust type: the schema is generated from `P` and arguments
/// are deserialized into it after validation.
///
/// ```
/// use schemars::JsonSchema;
/// use serde::Deserialize;
/// use tau_agent::{AgentToolResult, typed_tool};
///
/// /// Arguments of `get_time`.
/// #[derive(Deserialize, JsonSchema)]
/// struct GetTime {
///     /// IANA timezone name.
///     timezone: Option<String>,
/// }
///
/// let tool = typed_tool("get_time", "Get the current time", |args: GetTime, _ctx| async move {
///     Ok(AgentToolResult::text(format!("12:00 in {}", args.timezone.unwrap_or_else(|| "UTC".into()))))
/// });
/// ```
#[cfg(feature = "schemars")]
pub fn typed_tool<P, F, Fut>(name: impl Into<String>, description: impl Into<String>, execute: F) -> FnTool
where
    P: serde::de::DeserializeOwned + schemars::JsonSchema + Send + 'static,
    F: Fn(P, ToolContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<AgentToolResult, ToolError>> + Send + 'static,
{
    let execute = Arc::new(execute);
    FnTool::new(
        Tool::new(name, description, schema_for::<P>()),
        move |arguments, context| {
            let execute = execute.clone();
            async move {
                let params: P = serde_json::from_value(arguments)?;
                execute(params, context).await
            }
        },
    )
}

#[cfg(all(test, feature = "schemars"))]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serde_json::json;

    #[allow(dead_code)]
    #[derive(Deserialize, schemars::JsonSchema)]
    struct Inner {
        path: String,
    }

    #[allow(dead_code)]
    #[derive(Deserialize, schemars::JsonSchema)]
    struct Args {
        /// How many.
        count: u32,
        inner: Inner,
        note: Option<String>,
    }

    #[test]
    fn schema_is_inline_object() {
        let schema = schema_for::<Args>();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["count"]["description"], "How many.");
        assert_eq!(schema["properties"]["inner"]["properties"]["path"]["type"], "string");
        assert!(schema.get("$schema").is_none());
        assert!(schema.get("definitions").is_none());
        assert_eq!(schema["required"], json!(["count", "inner"]));
    }
}
