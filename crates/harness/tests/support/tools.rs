//! Test tools: real `HarnessTool`s answering `<name> done`, and a scripted wrapper that
//! records which calls started, holds them at a gate and returns controls.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use delta_ai::types::{JsonObject, Tool};
use delta_chord::context::Context;
use delta_harness::registry::Extension;
use delta_harness::tool::{
    Callers, FnHarnessTool, HarnessTool, ToolApi, ToolControl, ToolError, ToolExecutionMode, ToolExecutionResult,
    ToolReplay,
};
use futures::future::BoxFuture;
use tokio::sync::Notify;

/// A declaration of `name` taking any object.
pub fn declaration(name: &str) -> Tool {
    Tool::new(
        name,
        format!("The {name} tool"),
        serde_json::json!({ "type": "object" }),
    )
}

/// A tool named `name` answering `<name> done`.
pub fn done_tool(name: &str) -> FnHarnessTool {
    let answer = format!("{name} done");
    FnHarnessTool::new(declaration(name), move |_arguments, _api, _ctx| {
        let answer = answer.clone();
        Box::pin(async move { Ok(ToolExecutionResult::text(answer)) })
    })
}

/// A tool named `name` answering `<name> done`.
pub fn tool(name: &str) -> Arc<dyn HarnessTool> {
    Arc::new(done_tool(name))
}

/// A tool that makes its round sequential.
pub fn sequential_tool(name: &str) -> Arc<dyn HarnessTool> {
    Arc::new(done_tool(name).with_execution_mode(ToolExecutionMode::Sequential))
}

/// What the scripted tools do, shared with the case.
#[derive(Clone, Default)]
pub struct ToolScript {
    /// The control a call of each tool name returns.
    pub controls: Arc<Mutex<HashMap<String, ToolControl>>>,
    /// Call ids in the order their executions started.
    pub started: Arc<Mutex<Vec<String>>>,
    /// When set, every call waits for a permit before it runs.
    pub gate: Option<Arc<Notify>>,
}

impl ToolScript {
    pub fn control(&self, tool: &str, control: ToolControl) {
        self.controls.lock().unwrap().insert(tool.to_owned(), control);
    }

    pub fn started(&self) -> Vec<String> {
        self.started.lock().unwrap().clone()
    }
}

/// A tool run under a [`ToolScript`].
pub struct Scripted {
    inner: Arc<dyn HarnessTool>,
    script: ToolScript,
}

impl Scripted {
    pub fn wrap(inner: Arc<dyn HarnessTool>, script: &ToolScript) -> Arc<dyn HarnessTool> {
        Arc::new(Self {
            inner,
            script: script.clone(),
        })
    }
}

impl HarnessTool for Scripted {
    fn definition(&self) -> &Tool {
        self.inner.definition()
    }

    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        self.inner.execution_mode()
    }

    fn callers(&self) -> Callers {
        self.inner.callers()
    }

    fn deferred(&self) -> bool {
        self.inner.deferred()
    }

    fn replay(&self) -> ToolReplay {
        self.inner.replay()
    }

    fn execute<'a>(
        &'a self,
        arguments: JsonObject,
        api: &'a ToolApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<ToolExecutionResult, ToolError>> {
        Box::pin(async move {
            self.script.started.lock().unwrap().push(api.call_id().to_owned());
            if let Some(gate) = &self.script.gate {
                ctx.run(gate.notified()).await?;
            }
            let mut result = self.inner.execute(arguments, api, ctx).await?;
            let control = self.script.controls.lock().unwrap().get(self.name()).cloned();
            if control.is_some() {
                result.control = control;
            }
            Ok(result)
        })
    }
}

/// An extension with `tools`, each run under `script`.
pub fn extension(script: &ToolScript, tools: Vec<Arc<dyn HarnessTool>>) -> Extension {
    let mut extension = Extension::new("test-tools");
    extension.tools = tools.into_iter().map(|tool| Scripted::wrap(tool, script)).collect();
    extension
}
