//! The `call` phase (tool.md, "The pipeline"): read the call, resolve the tool, repair and
//! validate the arguments, run the `before_tool` chain, ask the approval policy, commit the
//! intent and execute. Every step before the intent commit that fails settles a harness
//! error result without having run anything.

use std::sync::Arc;

use delta_ai::types::{JsonObject, Message};
use delta_chord::context::{Cancelled, Context};

use super::definition::HarnessTool;
use super::records::{ParentCall, ToolHookCall, ToolTaskCheckpoint, ToolTaskInput};
use super::settle::{Settlement, settle};
use super::task::{Rt, ToolTask};
use super::validate;
use super::{approval, execute};
use crate::agent::{Agent, AgentError, AgentRuntime, HookRuntime, report_host};
use crate::approval::{ApprovalCall, ApprovalId, PolicyDecision, remove_request};
use crate::generation::{SlotStatus, edit_live};
use crate::registry::BeforeTool;
use crate::scheduler::{Next, NextState, TaskError};
use crate::session::HostCallback;

/// A hook's ordinary failure is reported; one after the invocation's context was cancelled
/// stops the phase.
pub(super) fn hook_failed(
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

/// The call of the task: from the assistant entry (a model-issued call) or the input (a
/// nested call). A missing call is a fault.
pub(super) async fn read_call(rt: &Rt, ctx: &Context) -> Result<ToolHookCall, TaskError> {
    let task_id = rt.task_id();
    match rt.input() {
        ToolTaskInput::Model { assistant, call_id } => {
            let entry = rt
                .entry(*assistant, ctx)
                .await?
                .ok_or_else(|| TaskError::new(format!("Assistant entry {assistant} does not exist")))?;
            let call = entry
                .model
                .into_iter()
                .flatten()
                .find_map(|message| match message {
                    Message::Assistant(message) => message.tool_calls().find(|call| &call.id == call_id).cloned(),
                    _ => None,
                })
                .ok_or_else(|| TaskError::new(format!("Assistant entry {assistant} has no call {call_id}")))?;
            Ok(ToolHookCall {
                task_id,
                call,
                parent: None,
            })
        }
        ToolTaskInput::Nested {
            parent,
            parent_call_id,
            call,
            ..
        } => Ok(ToolHookCall {
            task_id,
            call: call.clone(),
            parent: Some(ParentCall {
                task_id: *parent,
                call_id: parent_call_id.clone(),
            }),
        }),
    }
}

/// The tool a call resolves to: among the offered tools for a model-issued call, among the
/// tools other tools may call for a nested one.
pub(super) fn resolve(agent: &Agent, input: &ToolTaskInput, name: &str) -> Option<Arc<dyn HarnessTool>> {
    let tools = match input {
        ToolTaskInput::Model { .. } => &agent.tools,
        ToolTaskInput::Nested { .. } => &agent.callable,
    };
    tools.iter().find(|tool| tool.name() == name).cloned()
}

/// The `call` phase.
pub(super) async fn call(rt: &Rt, ctx: &Context) -> Result<Next<ToolTask>, TaskError> {
    let mut call = read_call(rt, ctx).await?;
    let stored = call.call.clone();
    let agent = rt.agent(ctx).await?;
    let Some(tool) = resolve(&agent, rt.input(), &stored.name) else {
        let message = format!("Tool {} is not available", stored.name);
        return settle(rt, Settlement::early(rt, stored, "tool_unavailable", message));
    };
    let invalid = |message: String| settle(rt, Settlement::early(rt, stored.clone(), "invalid_arguments", message));

    match tool.prepare_arguments(call.call.arguments.clone()) {
        Ok(prepared) => call.call.arguments = prepared,
        Err(error) => return invalid(format!("Invalid arguments for tool \"{}\": {error}", stored.name)),
    }
    match validate::arguments(&*tool, &call.call) {
        Ok(arguments) => call.call.arguments = arguments,
        Err(message) => return invalid(message),
    }
    let rewritten = match before_tool(rt, &call, ctx).await? {
        Chain::Blocked(text) => {
            let message = format!("Tool call blocked: {text}");
            return settle(rt, Settlement::early(rt, stored, "blocked", message));
        }
        Chain::Arguments(arguments) => arguments,
    };
    if let Some(arguments) = rewritten {
        call.call.arguments = arguments;
        match validate::arguments(&*tool, &call.call) {
            Ok(arguments) => call.call.arguments = arguments,
            Err(message) => return invalid(message),
        }
    }
    match policy(rt, &agent, &*tool, &call, ctx).await? {
        Verdict::Allow => {}
        Verdict::Block(reason) => {
            let message = format!("Tool call blocked: {reason}");
            return settle(rt, Settlement::early(rt, stored, "blocked", message));
        }
        Verdict::Ask(ask) => {
            if let Some(next) = approval::ask(rt, ctx, &*tool, &call, ask).await? {
                return Ok(next);
            }
        }
    }

    let arguments = call.call.arguments.clone();
    let replay = tool.replay();
    commit_intent(rt, arguments.clone(), replay, None, ctx).await?;
    execute::run(rt, ctx, &*tool, &call, arguments, replay).await
}

/// How the `before_tool` chain ended.
enum Chain {
    Blocked(String),
    /// The replacement arguments, if any hook replaced them.
    Arguments(Option<JsonObject>),
}

/// The `before_tool` chain: each hook sees the arguments as the hooks before it left them;
/// the first block wins; an ordinary error blocks with its text.
async fn before_tool(rt: &Rt, call: &ToolHookCall, ctx: &Context) -> Result<Chain, TaskError> {
    let hooks = rt.hooks(ctx).await?;
    if hooks.is_empty() {
        return Ok(Chain::Arguments(None));
    }
    let api = rt.hook_api()?;
    let mut current = call.clone();
    let mut replaced = None;
    for (extension, hook) in hooks.iter() {
        match hook.before_tool(&current, &api, ctx).await {
            Ok(None) => {}
            Ok(Some(BeforeTool { block: Some(text), .. })) => return Ok(Chain::Blocked(text)),
            Ok(Some(BeforeTool { arguments, block: None })) => {
                if let Some(arguments) = arguments {
                    current.call.arguments.clone_from(&arguments);
                    replaced = Some(arguments);
                }
            }
            Err(error) => {
                hook_failed(rt, &*error, extension, ctx)?;
                return Ok(Chain::Blocked(error.to_string()));
            }
        }
    }
    Ok(Chain::Arguments(replaced))
}

/// What the approval policy decided.
enum Verdict {
    Allow,
    /// Blocked with the reason.
    Block(String),
    Ask(approval::Ask),
}

/// The approval policy on the final arguments. A policy failure denies (fail closed).
async fn policy(
    rt: &Rt,
    agent: &Agent,
    tool: &dyn HarnessTool,
    call: &ToolHookCall,
    ctx: &Context,
) -> Result<Verdict, TaskError> {
    let policy = rt.agent_services()?.approval().clone();
    let api = rt.hook_api()?;
    // The environment makes the subject's paths absolute; without one the policy decides on
    // what the call names.
    let env = match rt.env(ctx).await {
        Ok(env) => env,
        Err(AgentError::Cancelled(cancelled)) => return Err(cancelled.into()),
        Err(error) => {
            report_host(rt, &error, HostCallback::Env);
            None
        }
    };
    let request = ApprovalCall {
        conversation_id: rt.conversation_id(),
        task_id: rt.task_id(),
        call,
        tool,
        subject: tool.permission_subject(&call.call.arguments, env.as_deref()),
        cwd: match env.as_deref() {
            Some(env) => Some(env.cwd()),
            None => agent.cwd.as_deref(),
        },
    };
    match policy.decide(&request, &api, ctx).await {
        Ok(PolicyDecision::Allow) => Ok(Verdict::Allow),
        Ok(PolicyDecision::Deny { reason }) => Ok(Verdict::Block(format!("approval denied: {reason}"))),
        Ok(PolicyDecision::Ask {
            timeout,
            prompt,
            proposed_rule,
        }) => Ok(Verdict::Ask(approval::Ask {
            timeout,
            prompt,
            proposed_rule,
        })),
        Err(error) => {
            if let Some(reason) = ctx.reason() {
                return Err(Cancelled { reason }.into());
            }
            Ok(Verdict::Block(format!("approval denied: {error}")))
        }
    }
}

/// The intent commit: the final arguments and the replay policy become the checkpoint, and
/// the slot shows the call running (a nested slot with the final arguments); an approved
/// request is removed.
pub(super) async fn commit_intent(
    rt: &Rt,
    arguments: JsonObject,
    replay: super::ToolReplay,
    approved: Option<ApprovalId>,
    ctx: &Context,
) -> Result<(), TaskError> {
    let conversation = rt.conversation_id();
    let task = rt.task_id();
    rt.commit(ctx, move |tx, _| {
        Box::pin(async move {
            if let Some(id) = approved {
                remove_request(tx, conversation, id).await?;
            }
            edit_live(tx, conversation, |live| {
                if let Some(slot) = live.tool_slot_mut(task) {
                    slot.status = SlotStatus::Running;
                }
                if let Some(slot) = live.nested_slot_mut(task) {
                    slot.status = SlotStatus::Running;
                    if slot.arguments != arguments {
                        slot.arguments.clone_from(&arguments);
                    }
                }
            })
            .await?;
            let next = NextState::Running(ToolTaskCheckpoint::Execute { arguments, replay });
            Ok(((), Some(next)))
        })
    })
    .await?;
    Ok(())
}
