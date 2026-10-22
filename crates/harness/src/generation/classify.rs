//! Classification: one commit per terminal response (spec §8.3).
//!
//! | Stop reason | Commit |
//! | --- | --- |
//! | `stop` / `length`, `tool_use` without calls | the answer; the `Final` boundary; settle or continue |
//! | `tool_use` with calls | the answer; start the tool round; wait |
//! | `error`, context overflow with a cut | the error entry; a blocking compaction; wait |
//! | `error`, retryable within the policy | the error entry; sleep until the backoff ends |
//! | other `error`, `aborted` without a mark | the entry; fail `model_error` |
//! | `deferred` | sleep until `poll_at` |

use delta_ai::overflow::is_context_overflow;
use delta_ai::retry::is_retryable_assistant_error;
use delta_ai::types::{AssistantMessage, Message, StopReason, ToolCall, UserContent, UserMessage};
use delta_chord::context::{Cancelled, Context};

use super::failure::model_error;
use super::live::{
    LiveDeferred, LiveGeneration, LiveRetry, append_assistant, create_generation, edit_live, end_run, live_snapshot,
    start_run,
};
use super::{GenerationCheckpoint, GenerationResult, GenerationTask, Rt, hook_failed, round};
use crate::agent::{AgentRuntime, HookRuntime};
use crate::compaction::{CompactionInput, CompactionReason};
use crate::inbox::{At, QueueModes, USER_ENTRY, apply_boundary, prepare_boundary};
use crate::model::ModelRef;
use crate::records::{EntryDraft, EntryId, JoinPolicy, SubmissionSettlement, TaskId};
use crate::registry::Yielded;
use crate::scheduler::{Next, NextState, Outcome, ReportFrom, TaskError};

/// How long a deferred response waits before its first poll when the provider gives no hint.
const DEFAULT_POLL_AFTER_MS: i64 = 5_000;

/// What classification needs from the request that produced a message.
pub(super) struct Request {
    pub(super) attempt: u32,
    pub(super) compacted: Option<TaskId>,
    pub(super) model: ModelRef,
    pub(super) cutoff: EntryId,
    /// The committed context through `cutoff`, when the phase derived it.
    pub(super) messages: Option<Vec<Message>>,
    /// Set when the message came from polling, so a still-deferred result polls later.
    pub(super) poll_at: Option<i64>,
}

pub(super) async fn classify(
    rt: &Rt,
    request: Request,
    message: AssistantMessage,
    ctx: &Context,
) -> Result<Next<GenerationTask>, TaskError> {
    // An abort mark or close: the abort invocation or the reopened run handles the state.
    if let Some(reason) = ctx.reason() {
        return Err(Cancelled { reason }.into());
    }
    let conversation = rt.conversation_id();
    let task = rt.task_id();
    let Request {
        attempt,
        compacted,
        cutoff,
        ..
    } = request;
    if message.stop_reason == StopReason::Deferred
        && let Some(handle) = message.deferred.clone()
    {
        let after = handle
            .poll_after_ms
            .map_or(DEFAULT_POLL_AFTER_MS, |ms| i64::try_from(ms).unwrap_or(i64::MAX));
        let poll_at = rt
            .now()
            .saturating_add(after)
            .max(request.poll_at.map_or(i64::MIN, |previous| previous + 1));
        let model = request.model;
        return Ok(rt.next(move |tx, _| {
            Box::pin(async move {
                edit_live(tx, conversation, |live| {
                    live.generation = Some(LiveGeneration {
                        deferred: Some(LiveDeferred { poll_at }),
                        ..LiveGeneration::new(attempt)
                    });
                })
                .await?;
                let checkpoint = GenerationCheckpoint::Poll {
                    attempt,
                    compacted,
                    model,
                    cutoff,
                    handle,
                    poll_at,
                };
                Ok(NextState::Sleeping {
                    checkpoint,
                    until: poll_at,
                })
            })
        }));
    }
    after_response(rt, &message, ctx).await?;
    let calls: Vec<ToolCall> = message.tool_calls().cloned().collect();
    match message.stop_reason {
        StopReason::ToolUse if !calls.is_empty() => {
            return round::start(rt, &request_context(&request), message, calls, ctx).await;
        }
        StopReason::Stop | StopReason::Length | StopReason::ToolUse => return answer(rt, message, ctx).await,
        _ => {}
    }
    // The retry and compaction policies govern the next attempt: read now, not pinned.
    let services = rt.agent_services()?;
    let settings = services.settings();
    let overflow = message.stop_reason == StopReason::Error && is_context_overflow(&message, None);
    let policy = settings.compaction.policy_for(&request.model);
    if overflow && compacted.is_none() && policy.enabled {
        let view = rt.context_view(conversation, Some(cutoff), ctx).await?;
        let compactor = services.compactor().clone();
        if compactor.select_cut(&view, policy.keep_recent_tokens).is_some() {
            let text = message
                .error_message
                .clone()
                .unwrap_or_else(|| "Context overflow".to_owned());
            return Ok(rt.next(move |tx, _| {
                Box::pin(async move {
                    append_assistant(tx, conversation, message).await?;
                    edit_live(tx, conversation, |live| live.generation = None).await?;
                    let input = CompactionInput {
                        reason: CompactionReason::Overflow,
                        instructions: None,
                    };
                    let child = compactor.create(tx, conversation, input, Some(task)).await?;
                    let checkpoint = GenerationCheckpoint::Prepare {
                        attempt,
                        compacted: Some(child),
                        overflow: Some(text),
                        threshold: None,
                    };
                    Ok(NextState::Waiting {
                        checkpoint,
                        on: vec![child],
                        policy: JoinPolicy::AllSettled,
                    })
                })
            }));
        }
    }
    // An overflow is never retried: only a compaction can make the next request fit.
    let retry = settings.retry;
    if message.stop_reason == StopReason::Error
        && !overflow
        && is_retryable_assistant_error(&message)
        && retry.enabled
        && attempt <= retry.max_retries
    {
        let delay = i64::try_from(retry.delay(attempt).as_millis()).unwrap_or(i64::MAX);
        let until = rt.now().saturating_add(delay);
        let error = message.error_message.clone().unwrap_or_default();
        return Ok(rt.next(move |tx, _| {
            Box::pin(async move {
                append_assistant(tx, conversation, message).await?;
                edit_live(tx, conversation, |live| {
                    live.generation = Some(LiveGeneration {
                        retry: Some(LiveRetry { at: until, error }),
                        ..LiveGeneration::new(attempt)
                    });
                })
                .await?;
                let checkpoint = GenerationCheckpoint::Retry {
                    attempt,
                    compacted,
                    until,
                };
                Ok(NextState::Sleeping { checkpoint, until })
            })
        }));
    }
    let text = message
        .error_message
        .clone()
        .unwrap_or_else(|| format!("Model response ended with stop reason {}", message.stop_reason.as_str()));
    model_error(rt, text, Some(message))
}

/// What the tool round needs of the request.
fn request_context(request: &Request) -> round::RoundRequest {
    round::RoundRequest {
        cutoff: request.cutoff,
        messages: request.messages.clone(),
    }
}

/// The `after_response` observers.
async fn after_response(rt: &Rt, message: &AssistantMessage, ctx: &Context) -> Result<(), TaskError> {
    let hooks = rt.hooks(ctx).await?;
    if hooks.is_empty() {
        return Ok(());
    }
    let api = rt.hook_api()?;
    for (extension, hook) in hooks.iter() {
        if let Err(error) = hook.after_response(message, &api, ctx).await {
            hook_failed(rt, &*error, extension, ctx)?;
        }
    }
    Ok(())
}

/// The first `on_yield` continuation, unless the run took `max_continuations` already
/// (then it is reported and ignored).
async fn on_yield(rt: &Rt, message: &AssistantMessage, ctx: &Context) -> Result<Option<UserContent>, TaskError> {
    let hooks = rt.hooks(ctx).await?;
    if hooks.is_empty() {
        return Ok(None);
    }
    let continuations = live_snapshot(rt, ctx)
        .await?
        .run
        .filter(|run| run.task_id == rt.task_id())
        .map_or(0, |run| run.continuations);
    let yielded = Yielded {
        message: message.clone(),
        continuations,
    };
    let api = rt.hook_api()?;
    for (extension, hook) in hooks.iter() {
        match hook.on_yield(&yielded, &api, ctx).await {
            Ok(Some(continuation)) => {
                let limit = rt.agent_services()?.settings().max_continuations;
                if continuations >= limit {
                    let error =
                        std::io::Error::other(format!("The run took {limit} continuations; the next one is ignored"));
                    rt.report(&error, Some(ReportFrom::Extension(Some(extension.to_owned()))));
                    return Ok(None);
                }
                return Ok(Some(continuation));
            }
            Ok(None) => {}
            Err(error) => hook_failed(rt, &*error, extension, ctx)?,
        }
    }
    Ok(None)
}

/// A final answer. The `Final` boundary places queued items; the first `on_yield`
/// continuation appends a user message and hands the run to a successor generation, but
/// only when the boundary placed no user item and no reset. Otherwise the run's inputs
/// settle `done` and the placed user items start the next run.
async fn answer(rt: &Rt, message: AssistantMessage, ctx: &Context) -> Result<Next<GenerationTask>, TaskError> {
    let continuation = on_yield(rt, &message, ctx).await?;
    let conversation = rt.conversation_id();
    let task = rt.task_id();
    let modes = QueueModes::from(&rt.agent_services()?.settings());
    let now = rt.now();
    Ok(rt.next(move |tx, _| {
        Box::pin(async move {
            let mut boundary = prepare_boundary(tx, conversation, modes).await?;
            let entry = append_assistant(tx, conversation, message).await?;
            let placed = apply_boundary(tx, &mut boundary, At::Final, now).await?;
            match continuation {
                Some(content) if placed.users.is_empty() && !placed.reset => {
                    let user = Message::User(UserMessage {
                        content,
                        timestamp: now,
                    });
                    let draft = EntryDraft {
                        model: Some(vec![user]),
                        data: Some(delta_chord::json!({ "continuation": true })),
                        ..EntryDraft::new(USER_ENTRY)
                    };
                    tx.append_entry(conversation, draft).await?;
                    let successor = create_generation(tx, conversation).await?;
                    edit_live(tx, conversation, |live| {
                        live.generation = None;
                        if let Some(run) = live.run.as_mut().filter(|run| run.task_id == task) {
                            run.task_id = successor;
                            run.continuations += 1;
                        }
                    })
                    .await?;
                }
                _ => {
                    end_run(tx, conversation, task, &SubmissionSettlement::Done { answer: entry.id }).await?;
                    if !placed.users.is_empty() {
                        start_run(tx, conversation, placed.users).await?;
                    }
                }
            }
            Ok(NextState::Terminal(Outcome::Completed(GenerationResult {
                entry_id: entry.id,
            })))
        })
    }))
}
