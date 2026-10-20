//! The `request`, `retry` and `poll` phases: stream one request with throttled partials,
//! sleep through a backoff, poll a deferred response.

use std::time::Duration;

use delta_ai::model::ModelThinkingLevel;
use delta_ai::types::{AssistantMessage, DeferredHandle, Message, StopReason};
use delta_chord::context::{Cancelled, Context};
use delta_chord::json;
use futures::StreamExt;
use futures::future::BoxFuture;

use super::classify::{Request, classify};
use super::failure::no_model;
use super::live::{LiveGeneration, convert_partial, edit_live};
use super::provider::ensure_session_id;
use super::{GenerationCheckpoint, GenerationTask, Rt, hook_failed};
use crate::agent::{AgentRuntime, HookRuntime};
use crate::model::{ConversationStreamOptions, ModelClient, ModelRef, ModelRequest};
use crate::records::{EntryId, TaskId};
use crate::registry::RequestMessages;
use crate::scheduler::{Next, NextState, RuntimeError, TaskError};

/// What the `request` checkpoint pins.
pub(super) struct Pinned {
    pub(super) attempt: u32,
    pub(super) compacted: Option<TaskId>,
    pub(super) model: ModelRef,
    pub(super) thinking_level: ModelThinkingLevel,
    pub(super) stream_options: ConversationStreamOptions,
    pub(super) cutoff: EntryId,
}

pub(super) async fn run(rt: &Rt, ctx: &Context, pinned: Pinned) -> Result<Next<GenerationTask>, TaskError> {
    let conversation = rt.conversation_id();
    let attempt = pinned.attempt;
    // A partial an interrupted attempt committed becomes an aborted entry first.
    rt.commit(ctx, move |tx, _| {
        Box::pin(async move {
            convert_partial(tx, conversation).await?;
            edit_live(tx, conversation, |live| {
                live.generation = Some(LiveGeneration::new(attempt));
            })
            .await?;
            Ok(((), None))
        })
    })
    .await?;
    let services = rt.agent_services()?;
    let models = services.models().clone();
    let Some(model) = models.model(&pinned.model) else {
        return no_model(rt, Some(&pinned.model));
    };
    let view = rt.context_view(conversation, Some(pinned.cutoff), ctx).await?;
    let messages = before_request(rt, attempt, view.messages.clone(), ctx).await?;
    let session_id = ensure_session_id(rt, ctx).await?;
    let request = ModelRequest {
        model: &model,
        messages,
        thinking_level: pinned.thinking_level,
        options: &pinned.stream_options,
        session_id: Some(&session_id),
    };
    let interval = services.settings().progress.partial_interval;
    let message = stream_response(rt, &*models, request, attempt, interval, ctx).await?;
    let request = Request {
        attempt,
        compacted: pinned.compacted,
        model: pinned.model,
        cutoff: pinned.cutoff,
        messages: Some(view.messages),
        poll_at: None,
    };
    classify(rt, request, message, ctx).await
}

/// The `before_request` chain: each handler may replace the messages of this request.
async fn before_request(
    rt: &Rt,
    attempt: u32,
    messages: Vec<Message>,
    ctx: &Context,
) -> Result<Vec<Message>, TaskError> {
    let hooks = rt.hooks(ctx).await?;
    if hooks.is_empty() {
        return Ok(messages);
    }
    let api = rt.hook_api()?;
    let mut request = RequestMessages { attempt, messages };
    for (extension, hook) in hooks.iter() {
        match hook.before_request(&request, &api, ctx).await {
            Ok(Some(replaced)) => request.messages = replaced,
            Ok(None) => {}
            Err(error) => hook_failed(rt, &*error, extension, ctx)?,
        }
    }
    Ok(request.messages)
}

/// Resolves when `slot` holds a future that resolves; never otherwise.
async fn armed<F: Future + Unpin>(slot: &mut Option<F>) -> F::Output {
    match slot {
        Some(future) => future.await,
        None => std::future::pending().await,
    }
}

/// Streams one request and returns the terminal message. Partials commit as trailing
/// writes, at most one per `interval` and one in flight; the throttle stops and the commit
/// in flight is awaited before returning, so no stale partial lands after the outcome.
async fn stream_response(
    rt: &Rt,
    models: &dyn ModelClient,
    request: ModelRequest<'_>,
    attempt: u32,
    interval: Duration,
    ctx: &Context,
) -> Result<AssistantMessage, TaskError> {
    let fallback = AssistantMessage::empty(&request.model.api, &request.model.provider, &request.model.id);
    let mut stream = models.stream(request, ctx);
    let mut pending: Option<AssistantMessage> = None;
    let mut timer = None;
    let mut in_flight: Option<BoxFuture<'_, Result<(), RuntimeError>>> = None;
    let outcome = loop {
        tokio::select! {
            event = stream.next() => {
                let Some(event) = event else {
                    // A producer that ends without a terminal event fails the attempt.
                    let mut failed = stream.partial().cloned().unwrap_or_else(|| fallback.clone());
                    failed.stop_reason = StopReason::Error;
                    failed.error_message = Some("The response stream ended without a result".to_owned());
                    break Ok(failed);
                };
                if let Some(message) = event.final_message() {
                    break Ok(message.clone());
                }
                // An empty partial (the opening event) shows nothing; a deferred response
                // never gets past it, so it never leaves a partial.
                if let Some(partial) = stream.partial().filter(|partial| !partial.content.is_empty()) {
                    pending = Some(partial.clone());
                    if timer.is_none() && in_flight.is_none() {
                        timer = Some(rt.timer(interval));
                    }
                }
            }
            fired = armed(&mut timer) => {
                timer = None;
                if fired.is_ok() && let Some(partial) = pending.take() {
                    in_flight = Some(Box::pin(commit_partial(rt, attempt, partial, ctx)));
                }
            }
            committed = armed(&mut in_flight) => {
                in_flight = None;
                if let Err(error) = committed {
                    report_partial_failure(rt, &error, ctx);
                }
                if pending.is_some() {
                    timer = Some(rt.timer(interval));
                }
            }
            () = ctx.cancelled() => {
                break Err(Cancelled { reason: ctx.reason().unwrap_or_else(delta_chord::context::CancelReason::aborted) });
            }
        }
    };
    drop(timer);
    if let Some(commit) = in_flight
        && let Err(error) = commit.await
    {
        report_partial_failure(rt, &error, ctx);
    }
    Ok(outcome?)
}

/// Rejections after an abort mark or close are expected; anything else is reported.
fn report_partial_failure(rt: &Rt, error: &RuntimeError, ctx: &Context) {
    let expected = matches!(
        error,
        RuntimeError::Aborted(_) | RuntimeError::Ended(_) | RuntimeError::Closing | RuntimeError::Cancelled(_)
    );
    if !expected && !ctx.is_cancelled() {
        rt.report(error, None);
    }
}

/// One partial commit: `generation.message` replaced with `partial` as strict JSON.
async fn commit_partial(rt: &Rt, attempt: u32, partial: AssistantMessage, ctx: &Context) -> Result<(), RuntimeError> {
    let conversation = rt.conversation_id();
    let message = json::to_json(&partial).map_err(RuntimeError::Codec)?;
    rt.commit(ctx, move |tx, _| {
        Box::pin(async move {
            edit_live(tx, conversation, |live| {
                live.generation
                    .get_or_insert_with(|| LiveGeneration::new(attempt))
                    .message = Some(message);
            })
            .await?;
            Ok(((), None))
        })
    })
    .await
}

/// After the backoff's sleep: the next attempt prepares again, so agent and settings
/// changes during the backoff apply. Run early (a host's early resume), it sleeps again.
pub(super) fn retry(rt: &Rt, attempt: u32, compacted: Option<TaskId>, until: i64) -> Next<GenerationTask> {
    if until > rt.now() {
        return Next::sleep(
            until,
            GenerationCheckpoint::Retry {
                attempt,
                compacted,
                until,
            },
        );
    }
    let conversation = rt.conversation_id();
    rt.next(move |tx, _| {
        Box::pin(async move {
            edit_live(tx, conversation, |live| {
                live.generation = Some(LiveGeneration::new(attempt + 1));
            })
            .await?;
            Ok(NextState::Running(GenerationCheckpoint::prepare(
                attempt + 1,
                compacted,
            )))
        })
    })
}

/// After the sleep until `poll_at`: polls the deferred response and classifies the result.
#[allow(clippy::too_many_arguments, reason = "the fields of the poll checkpoint")]
pub(super) async fn poll(
    rt: &Rt,
    ctx: &Context,
    attempt: u32,
    compacted: Option<TaskId>,
    model_ref: ModelRef,
    cutoff: EntryId,
    handle: DeferredHandle,
    poll_at: i64,
) -> Result<Next<GenerationTask>, TaskError> {
    if poll_at > rt.now() {
        let checkpoint = GenerationCheckpoint::Poll {
            attempt,
            compacted,
            model: model_ref,
            cutoff,
            handle,
            poll_at,
        };
        return Ok(Next::sleep(poll_at, checkpoint));
    }
    let models = rt.agent_services()?.models().clone();
    let Some(model) = models.model(&model_ref) else {
        return no_model(rt, Some(&model_ref));
    };
    let message = ctx.run(models.fetch_deferred(&model, &handle, ctx)).await?;
    let request = Request {
        attempt,
        compacted,
        model: model_ref,
        cutoff,
        messages: None,
        poll_at: Some(poll_at),
    };
    classify(rt, request, message, ctx).await
}
