//! The agent's pieces in the harness: the `bash` tool and the generation hooks.

use std::sync::{Arc, LazyLock};

use delta_ai::types::{AssistantMessage, JsonObject, Tool, UserBlock, UserContent};
use delta_chord::context::Context;
use delta_harness::registry::{GenerationHooks, HookApi, HookResult, RequestMessages, Yielded};
use delta_harness::tool::{HarnessTool, ToolApi, ToolControl, ToolError, ToolExecutionMode, ToolExecutionResult};
use futures::future::BoxFuture;

use super::frontend::Notice;
use super::state::Shared;

/// The tool's name.
pub const BASH_TOOL_NAME: &str = "bash";

/// The tool the model runs commands with.
static BASH_TOOL: LazyLock<Tool> = LazyLock::new(|| Tool {
    name: BASH_TOOL_NAME.to_owned(),
    description: "Execute a bash command".to_owned(),
    parameters: serde_json::json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute",
            }
        },
        "required": ["command"],
    }),
    constrained_sampling: None,
});

/// Runs the model's commands in the agent's environment. The first call of a round runs
/// every action of the round, so each call knows whether the round ended the run.
#[derive(Debug)]
pub(super) struct BashTool {
    pub(super) shared: Arc<Shared>,
}

impl HarnessTool for BashTool {
    fn definition(&self) -> &Tool {
        &BASH_TOOL
    }

    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        Some(ToolExecutionMode::Sequential)
    }

    fn execute<'a>(
        &'a self,
        _arguments: JsonObject,
        api: &'a ToolApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<ToolExecutionResult, ToolError>> {
        Box::pin(async move {
            let (observation, stop) = self.shared.tool_call(api.call_id(), ctx).await;
            Ok(ToolExecutionResult {
                output: Some(vec![UserBlock::text(observation)]),
                control: stop.then(|| ToolControl {
                    terminate: true,
                    ..ToolControl::default()
                }),
                ..ToolExecutionResult::default()
            })
        })
    }
}

/// Counts and records answers, and continues the run when an answer called no tool.
#[derive(Debug)]
pub(super) struct Hooks {
    pub(super) shared: Arc<Shared>,
}

impl GenerationHooks for Hooks {
    fn before_request<'a>(
        &'a self,
        _request: &'a RequestMessages,
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<Vec<delta_ai::types::Message>>> {
        if let Some(frontend) = &self.shared.frontend {
            frontend.notice(Notice::Rule);
        }
        if let Some(on_step) = &self.shared.on_step {
            let (step, cost) = {
                let state = self.shared.lock();
                (state.n_calls + 1, state.cost)
            };
            on_step(step, cost);
        }
        Box::pin(async { Ok(None) })
    }

    fn after_response<'a>(
        &'a self,
        message: &'a AssistantMessage,
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<()>> {
        self.shared.record_answer(message);
        Box::pin(async { Ok(None) })
    }

    fn on_yield<'a>(
        &'a self,
        _yielded: &'a Yielded,
        _api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<UserContent>> {
        Box::pin(async move { Ok(self.shared.continue_after_answer(ctx).await.map(UserContent::from)) })
    }
}
