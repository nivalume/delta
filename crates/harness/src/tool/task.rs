//! [`ToolTask`]: the `delta.tool` task definition, its phases and its abort handler.

use delta_chord::context::Context;

use super::definition::DEFAULT_OUTPUT_LIMITS;
use super::output::Waiters;
use super::pipeline::{self, read_call, resolve};
use super::records::{ToolTaskCheckpoint, ToolTaskInput, ToolTaskResult};
use super::settle::{Ending, Settlement, from_slot, settle, slot_progress};
use super::{TOOL_TASK, TOOL_TASK_VERSION, approval, execute};
use crate::agent::AgentRuntime;
use crate::records::AbortReason;
use crate::registry::{TaskHooks, ToolHooks};
use crate::scheduler::{Next, TaskDefinition, TaskError, TaskRuntime};

/// The `delta.tool` task definition, version 1: one task per call, model-issued or nested.
#[derive(Clone, Copy, Debug, Default)]
pub struct ToolTask;

pub(super) type Rt = TaskRuntime<ToolTask>;

impl TaskDefinition for ToolTask {
    type Input = ToolTaskInput;
    type Checkpoint = ToolTaskCheckpoint;
    type Result = ToolTaskResult;

    fn name(&self) -> &str {
        TOOL_TASK
    }

    fn version(&self) -> u32 {
        TOOL_TASK_VERSION
    }

    fn initial(&self, _input: &ToolTaskInput) -> ToolTaskCheckpoint {
        ToolTaskCheckpoint::Call
    }

    async fn run<'a>(
        &'a self,
        phase: ToolTaskCheckpoint,
        rt: &'a Rt,
        ctx: &'a Context,
    ) -> Result<Next<Self>, TaskError> {
        match phase {
            ToolTaskCheckpoint::Call => pipeline::call(rt, ctx).await,
            ToolTaskCheckpoint::Approval {
                arguments,
                replay,
                deadline,
                ..
            } => approval::wait(rt, ctx, arguments, replay, deadline).await,
            ToolTaskCheckpoint::Execute { arguments, replay } => execute::recover(rt, ctx, arguments, replay).await,
        }
    }

    /// Settles an error result from what the slot published: `aborted`, or after a restart
    /// that abandoned the task, `interrupted` once intent was recorded and `abandoned`
    /// before. A pending approval request is withdrawn, without a decision entry.
    async fn abort<'a>(
        &'a self,
        phase: ToolTaskCheckpoint,
        rt: &'a Rt,
        ctx: &'a Context,
    ) -> Result<Next<Self>, TaskError> {
        let call = read_call(rt, ctx).await?;
        let name = &call.call.name;
        let restart = rt.task().record.abort_reason == Some(AbortReason::Restart);
        let (code, message) = match (restart, &phase) {
            (true, ToolTaskCheckpoint::Execute { .. }) => (
                "interrupted",
                format!("Tool {name} was interrupted and may have partially run"),
            ),
            (true, _) => (
                "abandoned",
                format!("Tool {name} was not started: its caller ended with a restart"),
            ),
            (false, _) => ("aborted", format!("Tool {name} was aborted")),
        };
        let agent = rt.agent(ctx).await?;
        let retain =
            resolve(&agent, rt.input(), name).map_or(DEFAULT_OUTPUT_LIMITS.retain, |tool| tool.output_limits().retain);
        let result = from_slot(slot_progress(rt, ctx).await?, retain, code, message);
        let settlement = Settlement {
            call: call.call,
            result,
            duration_ms: None,
            ending: Ending::Aborted,
            waiters: Waiters::none(rt.task_id()),
            withdraw: match phase {
                ToolTaskCheckpoint::Approval { id, .. } => Some(id),
                _ => None,
            },
        };
        settle(rt, settlement)
    }
}

impl TaskHooks for ToolTask {
    type Hooks = dyn ToolHooks;
}
