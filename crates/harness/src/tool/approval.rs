//! The tool task's side of an approval (approval.md, "The flow"): the request commit after
//! the policy asks, the `approval` phase that waits for the decision, and its consumption.

use std::time::Duration;

use delta_ai::types::JsonObject;
use delta_chord::context::{CancelReason, Cancelled, Context};
use delta_chord::json;

use super::definition::{HarnessTool, ToolReplay};
use super::execute;
use super::pipeline::{commit_intent, read_call, resolve};
use super::records::{ToolHookCall, ToolTaskCheckpoint};
use super::result::harness_error;
use super::settle::{Ending, Settlement, settle};
use super::task::{Rt, ToolTask};
use crate::agent::AgentRuntime;
use crate::approval::{
    ApprovalId, ApprovalRequest, ApprovalsDoc, ApprovalsState, PendingApprovals, Recorded, ReviewDecision,
    edit_approvals, record_decision,
};
use crate::documents::DocAddress;
use crate::generation::{LiveDoc, LiveState};
use crate::inbox::QueueModes;
use crate::scheduler::{Next, NextState, TaskError};
use crate::session::TxError;

/// What the policy asked for.
pub(super) struct Ask {
    pub(super) timeout: Option<Duration>,
    pub(super) prompt: Option<String>,
}

/// The committed `delta.approvals` of the task's conversation.
async fn approvals(rt: &Rt, ctx: &Context) -> Result<ApprovalsState, TaskError> {
    let address = DocAddress::<ApprovalsDoc>::new(rt.conversation_id());
    Ok(match rt.snapshot(address, ctx).await? {
        Some(value) => json::from_json(&value)?,
        None => ApprovalsState::default(),
    })
}

/// The policy asked: a matching session grant lets the call run (`None`); otherwise the
/// request is committed with its deadline and the task waits in its `approval` phase.
pub(super) async fn ask(
    rt: &Rt,
    ctx: &Context,
    tool: &dyn HarnessTool,
    call: &ToolHookCall,
    ask: Ask,
) -> Result<Option<Next<ToolTask>>, TaskError> {
    let key = tool.approval_key(&call.call.arguments);
    if let Some(key) = &key
        && approvals(rt, ctx).await?.granted(tool.name(), key)
    {
        return Ok(None);
    }
    let settings = rt.agent_services()?.settings();
    let timeout = ask.timeout.unwrap_or(settings.approval_timeout);
    let now = rt.now();
    let deadline = now.saturating_add(i64::try_from(timeout.as_millis()).unwrap_or(i64::MAX));
    let id = ApprovalId(rt.task_id());
    let arguments = call.call.arguments.clone();
    let replay = tool.replay();
    let request = ApprovalRequest {
        id,
        call_id: call.call.id.clone(),
        name: call.call.name.clone(),
        arguments: arguments.clone(),
        approval_key: key,
        prompt: ask.prompt,
        requested_at: now,
        deadline,
        decision: None,
    };
    let conversation = rt.conversation_id();
    let checkpoint = ToolTaskCheckpoint::Approval {
        id,
        arguments: arguments.clone(),
        replay,
        deadline,
    };
    rt.commit(ctx, move |tx, _| {
        Box::pin(async move {
            edit_approvals(tx, conversation, |state| {
                state.requests.retain(|pending| pending.id != id);
                state.requests.push(request);
                state.requests.sort_by_key(|pending| pending.id);
            })
            .await?;
            Ok(((), Some(NextState::Running(checkpoint))))
        })
    })
    .await?;
    wait(rt, ctx, arguments, replay, deadline).await.map(Some)
}

/// The `approval` phase: registers a waiter, then reads the committed request (registering
/// first closes the race with a decision committed in between); without a decision it waits
/// for the notification until `deadline`, then commits `TimedOut` unless a decision landed.
pub(super) async fn wait(
    rt: &Rt,
    ctx: &Context,
    arguments: JsonObject,
    replay: ToolReplay,
    deadline: i64,
) -> Result<Next<ToolTask>, TaskError> {
    let id = ApprovalId(rt.task_id());
    let pending = pending(rt)?;
    let mut waiter = pending.register(id);
    let request = approvals(rt, ctx)
        .await?
        .request(id)
        .cloned()
        .ok_or_else(|| TaskError::new(format!("Approval request {id} is missing")))?;
    let decision = match request.decision.clone() {
        Some(decision) => decision,
        None => {
            let remaining = u64::try_from(deadline.saturating_sub(rt.now())).unwrap_or(0);
            let notified = ctx
                .run(tokio::time::timeout(
                    Duration::from_millis(remaining),
                    waiter.decision(),
                ))
                .await?;
            match notified {
                Ok(Some(decision)) => decision,
                // Only a rerun replaces the waiter, never while this invocation lives.
                Ok(None) | Err(_) => time_out(rt, id, ctx).await?,
            }
        }
    };
    drop(waiter);
    consume(rt, ctx, &request, decision, arguments, replay).await
}

fn pending(rt: &Rt) -> Result<std::sync::Arc<PendingApprovals>, TaskError> {
    rt.services()
        .get::<PendingApprovals>()
        .ok_or_else(|| TaskError::new("The session has no approval waiters"))
}

/// Commits `TimedOut` unless a decision landed first; returns the request's decision.
async fn time_out(rt: &Rt, id: ApprovalId, ctx: &Context) -> Result<ReviewDecision, TaskError> {
    let conversation = rt.conversation_id();
    let modes = QueueModes::from(&rt.agent_services()?.settings());
    let now = rt.now();
    let decision = rt
        .commit(ctx, move |tx, _| {
            Box::pin(async move {
                match record_decision(tx, conversation, id, ReviewDecision::TimedOut, now, modes).await? {
                    Recorded::Written => Ok((ReviewDecision::TimedOut, None)),
                    Recorded::Already(decision) => Ok((decision, None)),
                    Recorded::Missing => Err(TxError::callback(format!("Approval request {id} is missing"))),
                }
            })
        })
        .await?;
    Ok(decision)
}

/// Acts on the decision: an approval runs the call with the requested arguments, a denial
/// or timeout settles `blocked`, an abort aborts the conversation's run.
async fn consume(
    rt: &Rt,
    ctx: &Context,
    request: &ApprovalRequest,
    decision: ReviewDecision,
    arguments: JsonObject,
    replay: ToolReplay,
) -> Result<Next<ToolTask>, TaskError> {
    let mut call = read_call(rt, ctx).await?;
    let stored = call.call.clone();
    let blocked = |reason: String| {
        let message = format!("Tool call blocked: {reason}");
        settle(
            rt,
            Settlement::early(rt, stored.clone(), "blocked", message).withdrawing(request.id),
        )
    };
    match decision {
        ReviewDecision::Approved | ReviewDecision::ApprovedForSession => {
            let agent = rt.agent(ctx).await?;
            let Some(tool) = resolve(&agent, rt.input(), &stored.name) else {
                let message = format!("Tool {} is not available", stored.name);
                let settlement = Settlement::early(rt, stored, "tool_unavailable", message).withdrawing(request.id);
                return settle(rt, settlement);
            };
            call.call.arguments.clone_from(&arguments);
            commit_intent(rt, arguments.clone(), replay, Some(request.id), ctx).await?;
            execute::run(rt, ctx, &*tool, &call, arguments, replay).await
        }
        ReviewDecision::Denied { rejection } => blocked(match rejection {
            Some(rejection) => format!("approval denied: {rejection}"),
            None => "approval denied".to_owned(),
        }),
        ReviewDecision::TimedOut => {
            let seconds = request.deadline.saturating_sub(request.requested_at) / 1000;
            blocked(format!("approval timed out after {seconds}s"))
        }
        ReviewDecision::Abort => abort_run(rt, ctx, request.id, stored).await,
    }
}

/// An `Abort` decision: marks the conversation's run for abort, whose cascade reaches this
/// task, and waits for the cancellation; the abort handler settles. Without a run (or a
/// run abort), the task settles itself `aborted`.
async fn abort_run(
    rt: &Rt,
    ctx: &Context,
    id: ApprovalId,
    stored: delta_ai::types::ToolCall,
) -> Result<Next<ToolTask>, TaskError> {
    let live: LiveState = match rt
        .snapshot(DocAddress::<LiveDoc>::new(rt.conversation_id()), ctx)
        .await?
    {
        Some(value) => json::from_json(&value)?,
        None => LiveState::default(),
    };
    let pending = pending(rt)?;
    if let (Some(run), Some(run_abort)) = (live.run, pending.run_abort()) {
        match run_abort.abort_run(run.task_id, ctx).await {
            Ok(()) => {
                ctx.cancelled().await;
                let reason = ctx.reason().unwrap_or_else(CancelReason::aborted);
                return Err(Cancelled { reason }.into());
            }
            Err(error) => rt.report(&*error, None),
        }
    }
    let message = format!("Tool {} was aborted", stored.name);
    let settlement = Settlement {
        call: stored,
        result: harness_error("aborted", message),
        duration_ms: None,
        ending: Ending::Aborted,
        waiters: super::output::Waiters::none(rt.task_id()),
        withdraw: Some(id),
    };
    settle(rt, settlement)
}
