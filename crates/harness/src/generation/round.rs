//! Tool rounds (spec §8.5): the generation starts one owned `delta.tool` task per offered
//! call, waits for them all, and hands the run over to the next generation.

use std::collections::HashSet;

use delta_ai::transcript::get_current_tools;
use delta_ai::types::{AssistantMessage, Message, ToolCall, UserMessage};
use delta_chord::context::Context;
use delta_chord::json;

use super::live::{
    LiveDoc, LiveState, SlotStatus, ToolSlot, append_assistant, create_generation, edit_live, end_run, start_run,
};
use super::{GenerationCheckpoint, GenerationResult, GenerationTask, Rt, hook_failed};
use crate::agent::{AgentRuntime, HookRuntime, add_tools};
use crate::documents::DocAddress;
use crate::inbox::{At, QueueModes, RESET_ENTRY, apply_boundary, prepare_boundary};
use crate::records::{EntryDraft, EntryId, Head, JoinPolicy, SubmissionSettlement, TaskId, TaskOutcome};
use crate::scheduler::{Next, NextState, Outcome, TaskError};
use crate::tool::{
    ToolControl, ToolExecutionMode, ToolTaskResult, append_tool_result, create_model_call_task, harness_error,
};

/// What a round needs of the request that produced the tool-calling answer.
pub(super) struct RoundRequest {
    pub(super) cutoff: EntryId,
    pub(super) messages: Option<Vec<Message>>,
}

/// Appends the tool-calling answer and starts its round in one commit. A call to a tool the
/// request did not offer gets its `tool_unavailable` result here; every other call gets a
/// tool task owned by the generation, only the first now when the round is sequential.
pub(super) async fn start(
    rt: &Rt,
    request: &RoundRequest,
    message: AssistantMessage,
    calls: Vec<ToolCall>,
    ctx: &Context,
) -> Result<Next<GenerationTask>, TaskError> {
    let conversation = rt.conversation_id();
    let task = rt.task_id();
    let messages = match &request.messages {
        Some(messages) => messages.clone(),
        None => rt.context_view(conversation, Some(request.cutoff), ctx).await?.messages,
    };
    let offered: HashSet<String> = get_current_tools(&messages).into_iter().map(|tool| tool.name).collect();
    // Read as the round starts; a tool resolves as its tool task resolves it.
    let agent = rt.agent(ctx).await?;
    let settings = rt.agent_services()?.settings();
    let sequential = settings.tool_execution == ToolExecutionMode::Sequential
        || calls.iter().any(|call| {
            offered.contains(&call.name)
                && agent
                    .tool(&call.name)
                    .is_some_and(|tool| tool.execution_mode() == Some(ToolExecutionMode::Sequential))
        });
    let now = rt.now();
    Ok(rt.next(move |tx, _| {
        Box::pin(async move {
            let entry = append_assistant(tx, conversation, message).await?;
            let mut slots = Vec::with_capacity(calls.len());
            let mut tools = Vec::new();
            let mut pending = Vec::new();
            for call in &calls {
                if !offered.contains(&call.name) {
                    let unavailable = harness_error("tool_unavailable", format!("Tool {} is not available", call.name));
                    let result = append_tool_result(tx, conversation, call, &unavailable, now, None).await?;
                    let mut slot = ToolSlot::new(&call.id, &call.name, SlotStatus::Done);
                    slot.entry = Some(result.id);
                    slots.push(slot);
                    continue;
                }
                let mut slot = ToolSlot::new(&call.id, &call.name, SlotStatus::Pending);
                if sequential && !tools.is_empty() {
                    pending.push(call.id.clone());
                } else {
                    let tool = create_model_call_task(tx, task, entry.id, &call.id).await?;
                    tools.push(tool);
                    slot.task_id = Some(tool);
                }
                slots.push(slot);
            }
            edit_live(tx, conversation, |live| {
                live.generation = None;
                live.tools = Some(slots);
            })
            .await?;
            let checkpoint = GenerationCheckpoint::Tools {
                assistant: entry.id,
                tools: tools.clone(),
                pending,
            };
            Ok(NextState::Waiting {
                checkpoint,
                on: tools,
                policy: JoinPolicy::AllSettled,
            })
        })
    }))
}

/// The `tools` phase, resumed once every task in the wait is terminal: start the next call
/// of a sequential round, or finish the round.
pub(super) async fn run(
    rt: &Rt,
    ctx: &Context,
    assistant: EntryId,
    tools: Vec<TaskId>,
    pending: Vec<String>,
) -> Result<Next<GenerationTask>, TaskError> {
    let Some((next, rest)) = pending.split_first() else {
        return finish(rt, ctx, assistant, tools).await;
    };
    let conversation = rt.conversation_id();
    let task = rt.task_id();
    let next = next.clone();
    let rest = rest.to_vec();
    Ok(rt.next(move |tx, _| {
        Box::pin(async move {
            let tool = create_model_call_task(tx, task, assistant, &next).await?;
            edit_live(tx, conversation, |live| {
                let slot = live
                    .tools
                    .iter_mut()
                    .flatten()
                    .find(|slot| slot.call_id == next && slot.task_id.is_none());
                if let Some(slot) = slot {
                    slot.task_id = Some(tool);
                }
            })
            .await?;
            let mut started = tools;
            started.push(tool);
            let checkpoint = GenerationCheckpoint::Tools {
                assistant,
                tools: started,
                pending: rest,
            };
            Ok(NextState::Waiting {
                checkpoint,
                on: vec![tool],
                policy: JoinPolicy::AllSettled,
            })
        })
    }))
}

/// The control of a tool task that completed a model-issued call.
fn control_of(outcome: &TaskOutcome) -> Option<ToolControl> {
    let TaskOutcome::Completed { result } = outcome else {
        return None;
    };
    match crate::tool::decode_result(result)? {
        ToolTaskResult::Model { control, .. } => control,
        ToolTaskResult::Nested => None,
    }
}

/// The round is over: apply the calls' controls, then end the run at the `Final` boundary
/// (`terminate`, `handoff`, or a placed reset) or hand it to the next generation at the
/// `PostTools` boundary.
async fn finish(
    rt: &Rt,
    ctx: &Context,
    assistant: EntryId,
    tools: Vec<TaskId>,
) -> Result<Next<GenerationTask>, TaskError> {
    let conversation = rt.conversation_id();
    let task = rt.task_id();
    let outcomes = rt.outcomes(&tools, ctx).await?;
    let controls: Vec<(TaskId, Option<ToolControl>)> =
        tools.iter().copied().zip(outcomes.iter().map(control_of)).collect();
    let live: LiveState = match rt.snapshot(DocAddress::<LiveDoc>::new(conversation), ctx).await? {
        Some(value) => json::from_json(&value)?,
        None => LiveState::default(),
    };
    let slots = live.tools.unwrap_or_default();
    let results: Vec<EntryId> = slots.iter().filter_map(|slot| slot.entry).collect();
    after_tools(rt, assistant, &results, ctx).await?;
    let control = |id: TaskId| {
        controls
            .iter()
            .find(|(task, _)| *task == id)
            .and_then(|(_, control)| control.as_ref())
    };
    // Every call of the round, those answered without a task too, must ask to terminate.
    let terminate = !slots.is_empty()
        && slots
            .iter()
            .all(|slot| slot.task_id.and_then(control).is_some_and(|control| control.terminate));
    let added: Vec<String> = controls
        .iter()
        .filter_map(|(_, control)| control.as_ref())
        .flat_map(|control| control.add_tools.iter().cloned())
        .collect();
    // The last handoff in call order wins.
    let handoff = controls
        .iter()
        .rev()
        .find_map(|(_, control)| control.as_ref()?.handoff.clone());
    let modes = QueueModes::from(&rt.agent_services()?.settings());
    let now = rt.now();
    Ok(rt.next(move |tx, _| {
        Box::pin(async move {
            let mut boundary = prepare_boundary(tx, conversation, modes).await?;
            if !added.is_empty() {
                add_tools(tx, conversation, &added).await?;
            }
            if terminate || handoff.is_some() {
                if let Some(handoff) = handoff {
                    let user = Message::User(UserMessage {
                        content: handoff.into(),
                        timestamp: now,
                    });
                    let draft = EntryDraft {
                        model: Some(vec![user]),
                        head: Some(Head::SelfRef),
                        ..EntryDraft::new(RESET_ENTRY)
                    };
                    let reset = tx.append_entry(conversation, draft).await?;
                    boundary.head = Some(reset.id);
                }
                let placed = apply_boundary(tx, &mut boundary, At::Final, now).await?;
                end_run(
                    tx,
                    conversation,
                    task,
                    &SubmissionSettlement::Done { answer: assistant },
                )
                .await?;
                if !placed.users.is_empty() {
                    start_run(tx, conversation, placed.users).await?;
                }
            } else {
                let placed = apply_boundary(tx, &mut boundary, At::PostTools, now).await?;
                if placed.reset {
                    // The queued reset cut the run's context before an answer.
                    let reset = SubmissionSettlement::Unanswered {
                        reason: "reset".to_owned(),
                        detail: None,
                    };
                    end_run(tx, conversation, task, &reset).await?;
                    if !placed.users.is_empty() {
                        start_run(tx, conversation, placed.users).await?;
                    }
                } else {
                    let successor = create_generation(tx, conversation).await?;
                    edit_live(tx, conversation, |live| {
                        live.tools = None;
                        live.nested_tools = None;
                        if let Some(run) = live.run.as_mut().filter(|run| run.task_id == task) {
                            run.inputs.extend(placed.users);
                            run.task_id = successor;
                        }
                    })
                    .await?;
                }
            }
            Ok(NextState::Terminal(Outcome::Completed(GenerationResult {
                entry_id: assistant,
            })))
        })
    }))
}

/// The `after_tools` observers.
async fn after_tools(rt: &Rt, assistant: EntryId, results: &[EntryId], ctx: &Context) -> Result<(), TaskError> {
    let hooks = rt.hooks(ctx).await?;
    if hooks.is_empty() {
        return Ok(());
    }
    let api = rt.hook_api()?;
    for (extension, hook) in hooks.iter() {
        if let Err(error) = hook.after_tools(assistant, results, &api, ctx).await {
            hook_failed(rt, &*error, extension, ctx)?;
        }
    }
    Ok(())
}
