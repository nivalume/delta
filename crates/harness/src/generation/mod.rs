//! The `delta.generation` task (spec §8.3): one model request per attempt as durable phases.
//!
//! `prepare` renders the prompt and plans the positional `delta.system` entries; `request`
//! streams the answer into `delta.live` on a throttle and classifies it in one commit; a
//! retryable error sleeps in `retry`, a deferred response in `poll`; a tool-calling answer
//! starts a round of owned `delta.tool` tasks and waits in `tools`, then hands the run to the
//! next generation. Every observable step is a committed revision, so a crash loses at most
//! one throttle window of partial output and a reopened storage resumes from the checkpoint.
//!
//! The run's inputs live in `delta.live.run`, not in the task. Corresponds to upstream
//! `src/harness/generation.ts`, `live.ts` and `provider.ts`. See `docs/design/generation.md`.

mod abort;
mod checkpoint;
mod classify;
mod cleanup;
mod failure;
mod live;
mod prepare;
mod provider;
mod request;
mod round;
mod slots;

use std::sync::Arc;

use delta_chord::context::{Cancelled, Context};
use serde::{Deserialize, Serialize};

pub use checkpoint::GenerationCheckpoint;
pub use cleanup::AgentSchedulerHooks;
pub use live::{
    LiveDeferred, LiveDoc, LiveGeneration, LiveRetry, LiveState, RunControl, append_assistant, convert_partial,
    create_generation, edit_live, end_run, live_state, start_run,
};
pub use provider::ProviderDoc;
pub use slots::{NestedToolSlot, NestedToolSummary, SlotProgress, SlotStatus, ToolSlot};

use crate::agent::HookRuntime;
use crate::records::EntryId;
use crate::registry::{GenerationHooks, TaskHooks};
use crate::scheduler::{AnyTask, Next, TaskDefinition, TaskError, TaskRuntime};
use crate::tool::ToolTask;

/// The task kind.
pub const GENERATION_TASK: &str = "delta.generation";
/// The entry kind of a model response.
pub const ASSISTANT_ENTRY: &str = "delta.assistant";

/// The built-in task definitions every registry holds.
pub fn builtin_tasks() -> Vec<Arc<dyn AnyTask>> {
    vec![Arc::new(GenerationTask), Arc::new(ToolTask)]
}

/// The input of a generation: `{}`. The run's inputs are in `delta.live.run`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationInput {}

/// A completed generation's answer, or the tool-calling answer of its round.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationResult {
    pub entry_id: EntryId,
}

/// The `delta.generation` task definition, version 1.
#[derive(Clone, Copy, Debug, Default)]
pub struct GenerationTask;

type Rt = TaskRuntime<GenerationTask>;

impl TaskDefinition for GenerationTask {
    type Input = GenerationInput;
    type Checkpoint = GenerationCheckpoint;
    type Result = GenerationResult;

    fn name(&self) -> &str {
        GENERATION_TASK
    }

    fn version(&self) -> u32 {
        1
    }

    fn initial(&self, _input: &GenerationInput) -> GenerationCheckpoint {
        GenerationCheckpoint::prepare(1, None)
    }

    async fn run<'a>(
        &'a self,
        phase: GenerationCheckpoint,
        rt: &'a Rt,
        ctx: &'a Context,
    ) -> Result<Next<Self>, TaskError> {
        match phase {
            GenerationCheckpoint::Prepare {
                attempt,
                compacted,
                overflow,
                threshold,
            } => prepare::run(rt, ctx, attempt, compacted, overflow, threshold == Some(true)).await,
            GenerationCheckpoint::Request {
                attempt,
                compacted,
                model,
                thinking_level,
                stream_options,
                cutoff,
            } => {
                let request = request::Pinned {
                    attempt,
                    compacted,
                    model,
                    thinking_level,
                    stream_options,
                    cutoff,
                };
                request::run(rt, ctx, request).await
            }
            GenerationCheckpoint::Retry {
                attempt,
                compacted,
                until,
            } => Ok(request::retry(rt, attempt, compacted, until)),
            GenerationCheckpoint::Poll {
                attempt,
                compacted,
                model,
                cutoff,
                handle,
                poll_at,
            } => request::poll(rt, ctx, attempt, compacted, model, cutoff, handle, poll_at).await,
            GenerationCheckpoint::Tools {
                assistant,
                tools,
                pending,
            } => round::run(rt, ctx, assistant, tools, pending).await,
        }
    }

    fn abort<'a>(
        &'a self,
        phase: GenerationCheckpoint,
        rt: &'a Rt,
        ctx: &'a Context,
    ) -> impl Future<Output = Result<Next<Self>, TaskError>> + Send + 'a {
        abort::run(rt, phase, ctx)
    }
}

impl TaskHooks for GenerationTask {
    type Hooks = dyn GenerationHooks;
}

/// A hook's ordinary failure is reported and the chain continues; one after the
/// invocation's context was cancelled stops the phase.
fn hook_failed(
    rt: &Rt,
    error: &(dyn std::error::Error + Send + Sync),
    extension: &str,
    ctx: &Context,
) -> Result<(), Cancelled> {
    if let Some(reason) = ctx.reason() {
        return Err(Cancelled { reason });
    }
    rt.report_hook(error, extension);
    Ok(())
}
