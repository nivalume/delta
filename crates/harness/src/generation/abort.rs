//! The generation's abort handler: it runs once the round's tool tasks are terminal.

use delta_ai::types::{Message, ToolCall};
use delta_chord::context::Context;

use super::live::{convert_partial, end_run};
use super::{GenerationCheckpoint, GenerationTask, Rt};
use crate::agent::{AgentRuntime, report_host};
use crate::records::{EntryId, SubmissionSettlement};
use crate::scheduler::{Next, NextState, TaskError};
use crate::session::HostCallback;
use crate::tool::{append_tool_result, harness_error};

/// Cancels a polled deferred response (best effort), converts a committed partial into an
/// aborted entry, gives every call of the round that never started an `aborted` result,
/// settles the inputs `unanswered` / `aborted` and ends `aborted`.
pub(super) async fn run(
    rt: &Rt,
    phase: GenerationCheckpoint,
    ctx: &Context,
) -> Result<Next<GenerationTask>, TaskError> {
    let mut unstarted = Vec::new();
    match &phase {
        GenerationCheckpoint::Poll { model, handle, .. } => {
            let models = rt.agent_services()?.models().clone();
            if let Some(model) = models.model(model)
                && let Err(error) = models.cancel_deferred(&model, handle, ctx).await
            {
                report_host(rt, &error, HostCallback::Models);
            }
        }
        GenerationCheckpoint::Tools { assistant, pending, .. } => {
            unstarted = read_calls(rt, *assistant, pending, ctx).await?;
        }
        _ => {}
    }
    let conversation = rt.conversation_id();
    let task = rt.task_id();
    let now = rt.now();
    Ok(rt.next(move |tx, _| {
        Box::pin(async move {
            convert_partial(tx, conversation).await?;
            for call in &unstarted {
                let result = harness_error("aborted", format!("Tool {} was aborted", call.name));
                append_tool_result(tx, conversation, call, &result, now, None).await?;
            }
            let aborted = SubmissionSettlement::Unanswered {
                reason: "aborted".to_owned(),
                detail: None,
            };
            end_run(tx, conversation, task, &aborted).await?;
            Ok(NextState::aborted(None))
        })
    }))
}

/// The calls `ids` of the assistant entry `assistant`, in the given order.
async fn read_calls(rt: &Rt, assistant: EntryId, ids: &[String], ctx: &Context) -> Result<Vec<ToolCall>, TaskError> {
    let entry = rt.entry(assistant, ctx).await?;
    let calls: Vec<ToolCall> = entry
        .and_then(|entry| entry.model)
        .into_iter()
        .flatten()
        .filter_map(|message| match message {
            Message::Assistant(message) => Some(message),
            _ => None,
        })
        .flat_map(|message| {
            message
                .content
                .into_iter()
                .filter_map(|block| block.as_tool_call().cloned())
        })
        .collect();
    Ok(ids
        .iter()
        .filter_map(|id| calls.iter().find(|call| &call.id == id).cloned())
        .collect())
}
