//! Execution after the intent commit, and the `execute` phase recovery reaches.

use delta_ai::types::JsonObject;
use delta_chord::context::Context;
use tokio::time::Instant;

use super::api::{NestedCalls, ToolApi};
use super::definition::{DEFAULT_OUTPUT_LIMITS, HarnessTool, ToolReplay};
use super::nested::{MADE_NESTED_CALLS, abort_nested};
use super::output::{self, Reporter, Waiters};
use super::pipeline::{read_call, resolve};
use super::records::{ToolHookCall, ToolTaskInput};
use super::result::{ToolDiagnostic, ToolExecutionResult, harness_error};
use super::settle::{Ending, Settlement, final_result, from_slot, settle, slot_progress};
use super::task::{Rt, ToolTask};
use crate::agent::{AgentError, AgentRuntime, report_host};
use crate::generation::{SlotProgress, edit_live};
use crate::scheduler::{Next, TaskError};
use crate::session::HostCallback;

/// Executes the call with its final `arguments`: builds the environment, runs `execute`
/// beside the progress commits, aborts nested calls left running, and settles the final
/// result. `duration_ms` is measured from here, also when execution fails.
pub(super) async fn run(
    rt: &Rt,
    ctx: &Context,
    tool: &dyn HarnessTool,
    call: &ToolHookCall,
    arguments: JsonObject,
    replay: ToolReplay,
) -> Result<Next<ToolTask>, TaskError> {
    let task = rt.task_id();
    let agent = rt.agent(ctx).await?;
    let settings = rt.agent_services()?.settings();
    let started = Instant::now();
    let elapsed = || u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);

    let env = match rt.env(ctx).await {
        Ok(env) => env,
        Err(AgentError::Cancelled(cancelled)) => return Err(cancelled.into()),
        Err(error) => {
            report_host(rt, &error, HostCallback::Env);
            let message = error.to_string();
            let settlement = Settlement {
                call: call.call.clone(),
                result: harness_error("tool_error", message.clone()),
                duration_ms: Some(elapsed()),
                ending: Ending::Failed(message),
                waiters: Waiters::none(task),
            };
            return settle(rt, settlement);
        }
    };

    let publish = !matches!(
        rt.input(),
        ToolTaskInput::Nested {
            progress: Some(false),
            ..
        }
    );
    let interval = settings.progress.output_interval;
    let (reporter, receiver) = Reporter::new(tool.output_limits(), publish.then_some(interval), task);
    let nested = NestedCalls::default();
    let work = async {
        let api = ToolApi {
            rt,
            call_id: &call.call.id,
            agent,
            env,
            reporter: &reporter,
            replay,
            nested: &nested,
            output_interval: interval,
        };
        let returned = ctx.run(tool.execute(arguments, &api, ctx)).await;
        drop(api);
        (returned, reporter.finish().await)
    };
    let ((returned, reports), ()) = futures::future::join(work, output::serve(rt, receiver, ctx)).await;
    let duration_ms = elapsed();
    let returned = returned?;

    if rt.memo::<bool>(MADE_NESTED_CALLS)?.unwrap_or(false) {
        abort_nested(rt, ctx).await?;
    }
    let (ending, returned) = match returned {
        Ok(result) => (Ending::Completed, result),
        Err(error) => {
            let message = error.to_string();
            let result = ToolExecutionResult {
                is_error: true,
                diagnostics: vec![ToolDiagnostic::error("tool_error", message.clone())],
                ..ToolExecutionResult::default()
            };
            (Ending::Failed(message), result)
        }
    };
    let (result, waiters) = final_result(rt, ctx, tool, call, returned, reports).await?;
    let settlement = Settlement {
        call: call.call.clone(),
        result,
        duration_ms: Some(duration_ms),
        ending,
        waiters,
    };
    settle(rt, settlement)
}

/// The `execute` phase, reached only by recovery: a call whose stored policy and current
/// declaration are both `Safe` runs again from the stored arguments, without hooks or
/// approval and with its slot's progress cleared; any other settles `interrupted` from what
/// its slot published and ends `failed`.
pub(super) async fn recover(
    rt: &Rt,
    ctx: &Context,
    arguments: JsonObject,
    replay: ToolReplay,
) -> Result<Next<ToolTask>, TaskError> {
    let mut call = read_call(rt, ctx).await?;
    let agent = rt.agent(ctx).await?;
    let tool = resolve(&agent, rt.input(), &call.call.name);
    if let Some(tool) = tool
        .as_ref()
        .filter(|tool| replay == ToolReplay::Safe && tool.replay() == ToolReplay::Safe)
    {
        clear_progress(rt, ctx).await?;
        call.call.arguments = arguments.clone();
        return run(rt, ctx, &**tool, &call, arguments, ToolReplay::Safe).await;
    }
    if rt.memo::<bool>(MADE_NESTED_CALLS)?.unwrap_or(false) {
        abort_nested(rt, ctx).await?;
    }
    let retain = tool.map_or(DEFAULT_OUTPUT_LIMITS.retain, |tool| tool.output_limits().retain);
    let message = format!("Tool {} was interrupted and may have partially run", call.call.name);
    let result = from_slot(slot_progress(rt, ctx).await?, retain, "interrupted", message.clone());
    let settlement = Settlement {
        call: call.call,
        result,
        duration_ms: None,
        ending: Ending::Failed(message),
        waiters: Waiters::none(rt.task_id()),
    };
    settle(rt, settlement)
}

/// A rerun reports from scratch.
async fn clear_progress(rt: &Rt, ctx: &Context) -> Result<(), TaskError> {
    let conversation = rt.conversation_id();
    let task = rt.task_id();
    rt.commit(ctx, move |tx, _| {
        Box::pin(async move {
            edit_live(tx, conversation, |live| {
                if let Some(progress) = live.slot_progress_mut(task) {
                    *progress = SlotProgress::default();
                }
            })
            .await?;
            Ok(((), None))
        })
    })
    .await?;
    Ok(())
}
