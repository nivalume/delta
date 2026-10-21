//! The convenience forms of [`HarnessTool`]: [`FnHarnessTool`], a declaration and a closure
//! with builder methods, and [`typed_harness_tool`], whose arguments are a Rust type.

use std::fmt;
use std::sync::Arc;

use delta_ai::types::{JsonObject, Tool};
use delta_chord::context::Context;
use delta_chord::output::OutputLimits;
use futures::future::BoxFuture;
use serde_json::Value;

use super::api::ToolApi;
use super::definition::{Callers, DEFAULT_OUTPUT_LIMITS, HarnessTool, ToolError, ToolExecutionMode, ToolReplay};
use super::result::ToolExecutionResult;

type ExecuteFn = dyn for<'a> Fn(JsonObject, &'a ToolApi<'a>, &'a Context) -> BoxFuture<'a, Result<ToolExecutionResult, ToolError>>
    + Send
    + Sync;
type PrepareFn = dyn Fn(JsonObject) -> Result<JsonObject, ToolError> + Send + Sync;
type KeyFn = dyn Fn(&JsonObject) -> Option<String> + Send + Sync;

/// A tool from a declaration and an `execute` closure; the builder methods set the rest of
/// [`HarnessTool`].
///
/// ```
/// use delta_ai::types::Tool;
/// use delta_harness::tool::{FnHarnessTool, ToolExecutionResult, ToolReplay};
///
/// let echo = FnHarnessTool::new(
///     Tool::new("echo", "Echo the text", serde_json::json!({ "type": "object" })),
///     |arguments, _api, _ctx| Box::pin(async move { Ok(ToolExecutionResult::text(format!("{arguments:?}"))) }),
/// )
/// .with_replay(ToolReplay::Safe);
/// ```
#[derive(Clone)]
pub struct FnHarnessTool {
    definition: Tool,
    execute: Arc<ExecuteFn>,
    replay: ToolReplay,
    callers: Callers,
    execution_mode: Option<ToolExecutionMode>,
    output_limits: OutputLimits,
    structured_output_schema: Option<Value>,
    deferred: bool,
    prepare: Option<Arc<PrepareFn>>,
    approval_key: Option<Arc<KeyFn>>,
}

impl FnHarnessTool {
    pub fn new<F>(definition: Tool, execute: F) -> Self
    where
        F: for<'a> Fn(
                JsonObject,
                &'a ToolApi<'a>,
                &'a Context,
            ) -> BoxFuture<'a, Result<ToolExecutionResult, ToolError>>
            + Send
            + Sync
            + 'static,
    {
        Self {
            definition,
            execute: Arc::new(execute),
            replay: ToolReplay::Unsafe,
            callers: Callers::BOTH,
            execution_mode: None,
            output_limits: DEFAULT_OUTPUT_LIMITS,
            structured_output_schema: None,
            deferred: false,
            prepare: None,
            approval_key: None,
        }
    }

    #[must_use]
    pub fn with_replay(mut self, replay: ToolReplay) -> Self {
        self.replay = replay;
        self
    }

    #[must_use]
    pub fn with_callers(mut self, callers: Callers) -> Self {
        self.callers = callers;
        self
    }

    #[must_use]
    pub fn with_execution_mode(mut self, mode: ToolExecutionMode) -> Self {
        self.execution_mode = Some(mode);
        self
    }

    #[must_use]
    pub fn with_output_limits(mut self, limits: OutputLimits) -> Self {
        self.output_limits = limits;
        self
    }

    #[must_use]
    pub fn with_structured_output_schema(mut self, schema: Value) -> Self {
        self.structured_output_schema = Some(schema);
        self
    }

    #[must_use]
    pub fn with_deferred(mut self, deferred: bool) -> Self {
        self.deferred = deferred;
        self
    }

    /// Repairs arguments before validation.
    #[must_use]
    pub fn with_prepare_arguments(
        mut self,
        prepare: impl Fn(JsonObject) -> Result<JsonObject, ToolError> + Send + Sync + 'static,
    ) -> Self {
        self.prepare = Some(Arc::new(prepare));
        self
    }

    /// The key a session approval grant matches; default the canonical arguments.
    #[must_use]
    pub fn with_approval_key(mut self, key: impl Fn(&JsonObject) -> Option<String> + Send + Sync + 'static) -> Self {
        self.approval_key = Some(Arc::new(key));
        self
    }
}

impl HarnessTool for FnHarnessTool {
    fn definition(&self) -> &Tool {
        &self.definition
    }

    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        self.execution_mode
    }

    fn callers(&self) -> Callers {
        self.callers
    }

    fn deferred(&self) -> bool {
        self.deferred
    }

    fn replay(&self) -> ToolReplay {
        self.replay
    }

    fn output_limits(&self) -> OutputLimits {
        self.output_limits
    }

    fn structured_output_schema(&self) -> Option<&Value> {
        self.structured_output_schema.as_ref()
    }

    fn prepare_arguments(&self, arguments: JsonObject) -> Result<JsonObject, ToolError> {
        match &self.prepare {
            Some(prepare) => prepare(arguments),
            None => Ok(arguments),
        }
    }

    fn approval_key(&self, arguments: &JsonObject) -> Option<String> {
        match &self.approval_key {
            Some(key) => key(arguments),
            None => Some(super::canonical_json(arguments)),
        }
    }

    fn execute<'a>(
        &'a self,
        arguments: JsonObject,
        api: &'a ToolApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<ToolExecutionResult, ToolError>> {
        (self.execute)(arguments, api, ctx)
    }
}

impl fmt::Debug for FnHarnessTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FnHarnessTool")
            .field("name", &self.definition.name)
            .field("replay", &self.replay)
            .field("callers", &self.callers)
            .finish_non_exhaustive()
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

/// A tool whose arguments are a Rust type: the schema is generated from `P` and validated
/// arguments are deserialized into it.
///
/// ```
/// use delta_harness::tool::{ToolExecutionResult, typed_harness_tool};
/// use schemars::JsonSchema;
/// use serde::Deserialize;
///
/// /// Arguments of `greet`.
/// #[derive(Deserialize, JsonSchema)]
/// struct Greet {
///     /// Who to greet.
///     name: String,
/// }
///
/// let tool = typed_harness_tool("greet", "Greet someone", |args: Greet, _api, _ctx| {
///     Box::pin(async move { Ok(ToolExecutionResult::text(format!("Hello, {}", args.name))) })
/// });
/// ```
#[cfg(feature = "schemars")]
pub fn typed_harness_tool<P, F>(name: impl Into<String>, description: impl Into<String>, execute: F) -> FnHarnessTool
where
    P: serde::de::DeserializeOwned + schemars::JsonSchema + Send + 'static,
    F: for<'a> Fn(P, &'a ToolApi<'a>, &'a Context) -> BoxFuture<'a, Result<ToolExecutionResult, ToolError>>
        + Send
        + Sync
        + 'static,
{
    FnHarnessTool::new(
        Tool::new(name, description, schema_for::<P>()),
        move |arguments, api, ctx| match serde_json::from_value::<P>(Value::Object(arguments)) {
            Ok(arguments) => execute(arguments, api, ctx),
            Err(error) => Box::pin(async move { Err(error.into()) }),
        },
    )
}
