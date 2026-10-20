//! A test tool and a minimal `delta.tool` task that answers model-issued calls, standing in
//! for the tool task until it lands.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use delta_ai::types::{Message, Tool, UserBlock};
use delta_chord::context::Context;
use delta_chord::json::Json;
use delta_harness::generation::edit_live;
use delta_harness::registry::Extension;
use delta_harness::scheduler::{Next, NextState, Outcome, TaskDefinition, TaskError, TaskRuntime};
use delta_harness::tool::{
    HarnessTool, ToolControl, ToolExecutionMode, ToolExecutionResult, ToolTaskInput, ToolTaskResult, append_tool_result,
};
use tokio::sync::Notify;

/// A tool offered to the model; the test task executes it.
pub struct TestTool {
    declaration: Tool,
    mode: Option<ToolExecutionMode>,
}

impl HarnessTool for TestTool {
    fn definition(&self) -> &Tool {
        &self.declaration
    }

    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        self.mode
    }
}

/// A tool named `name`.
pub fn tool(name: &str) -> Arc<dyn HarnessTool> {
    Arc::new(TestTool {
        declaration: Tool::new(
            name,
            format!("The {name} tool"),
            serde_json::json!({ "type": "object" }),
        ),
        mode: None,
    })
}

/// A tool that makes its round sequential.
pub fn sequential_tool(name: &str) -> Arc<dyn HarnessTool> {
    Arc::new(TestTool {
        declaration: Tool::new(
            name,
            format!("The {name} tool"),
            serde_json::json!({ "type": "object" }),
        ),
        mode: Some(ToolExecutionMode::Sequential),
    })
}

/// What the test task does, shared with the case.
#[derive(Clone, Default)]
pub struct ToolScript {
    /// The control a call of each tool name returns.
    pub controls: Arc<Mutex<HashMap<String, ToolControl>>>,
    /// Call ids in the order their tasks started.
    pub started: Arc<Mutex<Vec<String>>>,
    /// When set, every call waits for a permit before it answers.
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

/// The minimal `delta.tool` task: answers each call with `<name> done` in one phase.
pub struct TestToolTask {
    pub script: ToolScript,
}

impl TaskDefinition for TestToolTask {
    type Input = ToolTaskInput;
    type Checkpoint = Json;
    type Result = ToolTaskResult;

    fn name(&self) -> &'static str {
        "delta.tool"
    }

    fn version(&self) -> u32 {
        1
    }

    fn initial(&self, _input: &ToolTaskInput) -> Json {
        delta_chord::json!({ "phase": "call" })
    }

    async fn run<'a>(
        &'a self,
        _phase: Json,
        rt: &'a TaskRuntime<Self>,
        ctx: &'a Context,
    ) -> Result<Next<Self>, TaskError> {
        let ToolTaskInput::Model { assistant, call_id } = rt.input().clone();
        self.script.started.lock().unwrap().push(call_id.clone());
        if let Some(gate) = &self.script.gate {
            ctx.run(gate.notified()).await?;
        }
        let entry = rt.entry(assistant, ctx).await?.ok_or("no assistant entry")?;
        let call = entry
            .model
            .into_iter()
            .flatten()
            .find_map(|message| match message {
                Message::Assistant(message) => message.tool_calls().find(|call| call.id == call_id).cloned(),
                _ => None,
            })
            .ok_or("no such call")?;
        let control = self.script.controls.lock().unwrap().get(&call.name).cloned();
        let conversation = rt.conversation_id();
        let task = rt.task_id();
        let now = rt.now();
        Ok(rt.next(move |tx, _| {
            Box::pin(async move {
                let result = ToolExecutionResult {
                    output: Some(vec![UserBlock::text(format!("{} done", call.name))]),
                    ..ToolExecutionResult::default()
                };
                let record = append_tool_result(tx, conversation, &call, &result, now, Some(1)).await?;
                edit_live(tx, conversation, |live| {
                    if let Some(slot) = live.tools.iter_mut().flatten().find(|slot| slot.task_id == Some(task)) {
                        slot.finish();
                        slot.entry = Some(record.id);
                    }
                })
                .await?;
                Ok(NextState::Terminal(Outcome::Completed(ToolTaskResult::Model {
                    entry_id: record.id,
                    control,
                })))
            })
        }))
    }

    async fn abort<'a>(
        &'a self,
        _phase: Json,
        _rt: &'a TaskRuntime<Self>,
        _ctx: &'a Context,
    ) -> Result<Next<Self>, TaskError> {
        Ok(Next::abort(None))
    }
}

/// An extension with `tools` and the test task.
pub fn extension(script: &ToolScript, tools: Vec<Arc<dyn HarnessTool>>) -> Extension {
    let mut extension = Extension::new("test-tools").with_task(Arc::new(TestToolTask { script: script.clone() }));
    extension.tools = tools;
    extension
}
