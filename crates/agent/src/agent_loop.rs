//! The low-level agent loop. It works with agent messages throughout and converts them to
//! LLM messages only at the provider boundary.

use futures::future::{BoxFuture, join_all};
use futures::{Stream, StreamExt};
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use tau_ai::transcript::{ToolStateChanges, get_current_tools, get_tool_state_changes};
use tau_ai::validation::validate_tool_arguments;
use tau_ai::{
    AssistantMessage, AssistantMessageEvent, Context, Message, StopReason, SystemMessage, ToolCall, ToolResultMessage,
    normalize_context,
};

use crate::hooks::{
    AfterToolCallContext, AgentContext, AgentLoopConfig, AgentTurnContext, BeforeToolCallContext,
    PrepareRequestContext, StreamFn,
};
use crate::tool::{AgentTool, ToolContext, ToolUpdates};
use crate::types::{
    AgentEvent, AgentMessage, AgentToolCallOutcome, AgentToolResult, AgentTurnDecision, CustomMessage,
    ToolExecutionMode,
};

/// Receives the loop's events, in order. The loop awaits each `emit`.
pub trait AgentEventSink<C>: Send + Sync {
    fn emit(&self, event: AgentEvent<C>) -> BoxFuture<'_, ()>;
}

/// Errors of starting a run.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AgentLoopError {
    #[error("Cannot continue: no messages in context")]
    NoMessages,
    #[error("Cannot continue from message role: assistant")]
    AssistantTail,
}

/// Start a loop with new prompt messages: they are added to the context and emitted.
/// Returns the messages the run added.
pub async fn run_agent_loop<C: CustomMessage>(
    prompts: Vec<AgentMessage<C>>,
    mut context: AgentContext<C>,
    mut config: AgentLoopConfig<C>,
    sink: &dyn AgentEventSink<C>,
    cancel: CancellationToken,
    stream_fn: &StreamFn,
) -> Vec<AgentMessage<C>> {
    let initial = declare_tool_changes(&context, prompts);
    let mut new_messages = initial.clone();
    context.messages.extend(initial.iter().cloned());
    sink.emit(AgentEvent::AgentStart).await;
    sink.emit(AgentEvent::TurnStart).await;
    for message in initial {
        sink.emit(AgentEvent::MessageStart {
            message: message.clone(),
        })
        .await;
        sink.emit(AgentEvent::MessageEnd { message }).await;
    }
    run_loop(&mut context, &mut new_messages, &mut config, &cancel, sink, stream_fn).await;
    new_messages
}

/// Continue from the current context without adding a message (retries). The last message
/// must convert to a user or tool result message.
pub async fn run_agent_loop_continue<C: CustomMessage>(
    mut context: AgentContext<C>,
    mut config: AgentLoopConfig<C>,
    sink: &dyn AgentEventSink<C>,
    cancel: CancellationToken,
    stream_fn: &StreamFn,
) -> Result<Vec<AgentMessage<C>>, AgentLoopError> {
    check_continue(&context)?;
    let mut new_messages = Vec::new();
    sink.emit(AgentEvent::AgentStart).await;
    sink.emit(AgentEvent::TurnStart).await;
    run_loop(&mut context, &mut new_messages, &mut config, &cancel, sink, stream_fn).await;
    Ok(new_messages)
}

fn check_continue<C: CustomMessage>(context: &AgentContext<C>) -> Result<(), AgentLoopError> {
    match context.messages.last() {
        None => Err(AgentLoopError::NoMessages),
        Some(message) if message.role() == "assistant" => Err(AgentLoopError::AssistantTail),
        Some(_) => Ok(()),
    }
}

/// Events of a loop started by [`agent_loop`] or [`agent_loop_continue`]. The stream is
/// observational: the loop does not wait for the consumer between events.
#[derive(Debug)]
pub struct AgentEventStream<C> {
    rx: mpsc::UnboundedReceiver<AgentEvent<C>>,
    task: Option<tokio::task::JoinHandle<Vec<AgentMessage<C>>>>,
}

impl<C: CustomMessage> AgentEventStream<C> {
    /// Drain the events and return the messages the run added.
    pub async fn result(mut self) -> Vec<AgentMessage<C>> {
        while self.rx.recv().await.is_some() {}
        match self.task.take() {
            Some(task) => task.await.unwrap_or_default(),
            None => Vec::new(),
        }
    }
}

impl<C> Stream for AgentEventStream<C> {
    type Item = AgentEvent<C>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

struct ChannelSink<C>(mpsc::UnboundedSender<AgentEvent<C>>);

impl<C: Send + 'static> AgentEventSink<C> for ChannelSink<C> {
    fn emit(&self, event: AgentEvent<C>) -> BoxFuture<'_, ()> {
        let _ = self.0.send(event);
        Box::pin(async {})
    }
}

/// [`run_agent_loop`] on a Tokio task, as a stream of events.
pub fn agent_loop<C: CustomMessage>(
    prompts: Vec<AgentMessage<C>>,
    context: AgentContext<C>,
    config: AgentLoopConfig<C>,
    cancel: CancellationToken,
    stream_fn: StreamFn,
) -> AgentEventStream<C> {
    let (tx, rx) = mpsc::unbounded_channel();
    let task = tokio::spawn(async move {
        let sink = ChannelSink(tx);
        run_agent_loop(prompts, context, config, &sink, cancel, &stream_fn).await
    });
    AgentEventStream { rx, task: Some(task) }
}

/// [`run_agent_loop_continue`] on a Tokio task, as a stream of events.
pub fn agent_loop_continue<C: CustomMessage>(
    context: AgentContext<C>,
    config: AgentLoopConfig<C>,
    cancel: CancellationToken,
    stream_fn: StreamFn,
) -> Result<AgentEventStream<C>, AgentLoopError> {
    check_continue(&context)?;
    let (tx, rx) = mpsc::unbounded_channel();
    let task = tokio::spawn(async move {
        let sink = ChannelSink(tx);
        run_agent_loop_continue(context, config, &sink, cancel, &stream_fn)
            .await
            .unwrap_or_default()
    });
    Ok(AgentEventStream { rx, task: Some(task) })
}

async fn poll_source<C>(source: Option<&crate::hooks::MessageSource<C>>) -> Vec<AgentMessage<C>> {
    match source {
        Some(source) => source().await,
        None => Vec::new(),
    }
}

fn apply_request_update<C: CustomMessage>(
    context: &mut AgentContext<C>,
    config: &mut AgentLoopConfig<C>,
    update: crate::hooks::AgentRequestUpdate<C>,
) {
    if let Some(next) = update.context {
        *context = next;
    }
    if let Some(model) = update.model {
        config.model = model;
    }
    if let Some(level) = update.thinking_level {
        config.set_thinking_level(level);
    }
}

/// Main loop shared by both entry points.
async fn run_loop<C: CustomMessage>(
    context: &mut AgentContext<C>,
    new_messages: &mut Vec<AgentMessage<C>>,
    config: &mut AgentLoopConfig<C>,
    cancel: &CancellationToken,
    sink: &dyn AgentEventSink<C>,
    stream_fn: &StreamFn,
) {
    let mut last_turn: Option<(AssistantMessage, Vec<ToolResultMessage>)> = None;
    let mut explicit_continuation = false;
    // Steering typed while waiting for the run to start.
    let mut pending = poll_source(config.get_steering_messages.as_ref()).await;

    loop {
        let mut has_more_tool_calls = true;
        while has_more_tool_calls || !pending.is_empty() {
            let mut prepared = Vec::new();
            if let Some((message, tool_results)) = &last_turn {
                let hooks = config.hooks.clone();
                let turn = AgentTurnContext {
                    message,
                    tool_results,
                    context,
                    new_messages,
                };
                if let Some(update) = hooks.prepare_next_turn(turn, cancel).await {
                    prepared = update.messages.unwrap_or_default();
                    apply_request_update(
                        context,
                        config,
                        crate::hooks::AgentRequestUpdate {
                            context: update.context,
                            model: update.model,
                            thinking_level: update.thinking_level,
                        },
                    );
                }
                // Preparation can be slow (compaction); pick up steering queued meanwhile,
                // unless a batch is already pending (one-at-a-time would deliver two).
                if pending.is_empty() {
                    pending = poll_source(config.get_steering_messages.as_ref()).await;
                }
                sink.emit(AgentEvent::TurnStart).await;
            }

            prepared.append(&mut pending);
            for message in declare_tool_changes(context, prepared) {
                sink.emit(AgentEvent::MessageStart {
                    message: message.clone(),
                })
                .await;
                sink.emit(AgentEvent::MessageEnd {
                    message: message.clone(),
                })
                .await;
                context.messages.push(message.clone());
                new_messages.push(message);
            }

            let hooks = config.hooks.clone();
            let request = PrepareRequestContext {
                context,
                model: &config.model,
                thinking_level: config.thinking_level(),
            };
            if let Some(update) = hooks.prepare_request(request, cancel).await {
                apply_request_update(context, config, update);
            }

            let message = stream_assistant_response(context, config, cancel, sink, stream_fn).await;
            new_messages.push(message.clone().into());

            if message.stop_reason.is_failure() {
                let turn = AgentTurnContext {
                    message: &message,
                    tool_results: &[],
                    context,
                    new_messages,
                };
                let _ = hooks.finish_turn(turn, cancel).await;
                sink.emit(AgentEvent::TurnEnd {
                    message: message.into(),
                    tool_results: Vec::new(),
                })
                .await;
                sink.emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                })
                .await;
                return;
            }

            let tool_calls: Vec<ToolCall> = message.tool_calls().cloned().collect();
            let mut tool_results = Vec::new();
            has_more_tool_calls = false;
            if !tool_calls.is_empty() {
                // A length stop may have truncated every call's arguments; fail them all.
                let batch = if message.stop_reason == StopReason::Length {
                    fail_truncated_tool_calls(&tool_calls, sink).await
                } else {
                    execute_tool_calls(context, &message, &tool_calls, config, cancel, sink).await
                };
                tool_results = batch.messages;
                has_more_tool_calls = !batch.terminate;
                for result in &tool_results {
                    context.messages.push(result.clone().into());
                    new_messages.push(result.clone().into());
                }
            }

            let turn = AgentTurnContext {
                message: &message,
                tool_results: &tool_results,
                context,
                new_messages,
            };
            let decision = hooks.finish_turn(turn, cancel).await;
            sink.emit(AgentEvent::TurnEnd {
                message: message.clone().into(),
                tool_results: tool_results.clone(),
            })
            .await;
            last_turn = Some((message, tool_results));

            if decision == Some(AgentTurnDecision::End) {
                sink.emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                })
                .await;
                return;
            }
            explicit_continuation = decision == Some(AgentTurnDecision::Continue);
            pending = poll_source(config.get_steering_messages.as_ref()).await;
            if has_more_tool_calls || !pending.is_empty() {
                explicit_continuation = false;
            }
        }

        // The agent would stop here; follow-up messages start another turn.
        let follow_ups = poll_source(config.get_follow_up_messages.as_ref()).await;
        if !follow_ups.is_empty() {
            explicit_continuation = false;
            pending = follow_ups;
            continue;
        }
        // No natural request was selected: fulfill a continue decision with one context-only turn.
        if explicit_continuation {
            explicit_continuation = false;
            continue;
        }
        break;
    }
    sink.emit(AgentEvent::AgentEnd {
        messages: new_messages.clone(),
    })
    .await;
}

fn system_messages<C: CustomMessage>(messages: &[AgentMessage<C>]) -> impl Iterator<Item = Message> + '_ {
    messages.iter().filter_map(|message| match message.as_llm() {
        Some(system @ Message::System(_)) => Some(system.clone()),
        _ => None,
    })
}

fn with_tool_changes(message: &SystemMessage, changes: &ToolStateChanges) -> SystemMessage {
    SystemMessage {
        tools_added: (!changes.tools_added.is_empty()).then(|| changes.tools_added.clone()),
        tools_removed: (!changes.tools_removed.is_empty()).then(|| changes.tools_removed.clone()),
        ..message.clone()
    }
}

/// Declare tool loadout changes to the model.
///
/// `context.tools` is what the runtime can execute; the transcript's system messages declare
/// what the model may call. The difference becomes `tools_added`/`tools_removed` on a system
/// message: the last pending system message when there is one (its tool fields are replaced
/// by the delta), otherwise a new one inserted before the first non-system pending message.
pub fn declare_tool_changes<C: CustomMessage>(
    context: &AgentContext<C>,
    pending: Vec<AgentMessage<C>>,
) -> Vec<AgentMessage<C>> {
    let system_index = pending.iter().rposition(AgentMessage::is_system);
    let no_changes = ToolStateChanges::default();
    let baseline: Vec<AgentMessage<C>> = match system_index {
        Some(index) => pending
            .iter()
            .enumerate()
            .map(|(i, message)| match message {
                AgentMessage::Llm(Message::System(system)) if i == index => {
                    AgentMessage::Llm(Message::System(with_tool_changes(system, &no_changes)))
                }
                other => other.clone(),
            })
            .collect(),
        None => pending.clone(),
    };
    let declared: Vec<Message> = system_messages(&context.messages)
        .chain(system_messages(&baseline))
        .collect();
    let executable: Vec<_> = context.tools.iter().map(|tool| tool.definition().clone()).collect();
    let changes = get_tool_state_changes(&get_current_tools(&declared), &executable);
    let unchanged = changes.tools_added.is_empty() && changes.tools_removed.is_empty();

    if let Some(index) = system_index {
        let AgentMessage::Llm(Message::System(system)) = &pending[index] else {
            unreachable!()
        };
        let declares = system.tools_added.as_ref().is_some_and(|tools| !tools.is_empty())
            || system.tools_removed.as_ref().is_some_and(|tools| !tools.is_empty());
        // Keep the caller's message when it declares no tool changes and none are needed.
        if unchanged && !declares {
            return pending;
        }
        let mut result = baseline;
        result[index] = AgentMessage::Llm(Message::System(with_tool_changes(system, &changes)));
        return result;
    }
    if unchanged {
        return pending;
    }
    let update = with_tool_changes(
        &SystemMessage {
            timestamp: tau_ai::types::now_ms(),
            ..SystemMessage::default()
        },
        &changes,
    );
    let insert_at = pending
        .iter()
        .position(|message| !message.is_system())
        .unwrap_or(pending.len());
    let mut result = pending;
    result.insert(insert_at, AgentMessage::Llm(Message::System(update)));
    result
}

/// Stream one assistant response, converting the transcript at the provider boundary.
async fn stream_assistant_response<C: CustomMessage>(
    context: &mut AgentContext<C>,
    config: &AgentLoopConfig<C>,
    cancel: &CancellationToken,
    sink: &dyn AgentEventSink<C>,
    stream_fn: &StreamFn,
) -> AssistantMessage {
    let hooks = config.hooks.clone();
    let messages = hooks.transform_context(context.messages.clone(), cancel).await;
    let llm_messages = hooks.convert_to_llm(&messages).await;
    let transcript = normalize_context(&Context::new(llm_messages));
    let api_key = hooks
        .get_api_key(&config.model.provider)
        .await
        .or_else(|| config.stream_options.base.api_key.clone());
    let mut options = config.stream_options.clone();
    options.base.api_key = api_key;
    options.base.cancel = Some(cancel.clone());
    let thinking_level = config.thinking_level();

    let mut stream = stream_fn(config.model.clone(), transcript, options).await;
    let mut started = false;
    let mut final_message = None;
    while let Some(event) = stream.next().await {
        match &event {
            AssistantMessageEvent::Start { message } => {
                started = true;
                sink.emit(AgentEvent::MessageStart {
                    message: message.clone().into(),
                })
                .await;
            }
            AssistantMessageEvent::Done { message, .. } | AssistantMessageEvent::Error { error: message, .. } => {
                final_message = Some(message.clone());
                break;
            }
            _ if started => {
                if let Some(partial) = stream.partial() {
                    let snapshot = Arc::new(partial.clone());
                    sink.emit(AgentEvent::MessageUpdate {
                        message: snapshot,
                        assistant_message_event: event,
                    })
                    .await;
                }
            }
            _ => {}
        }
    }
    let mut message = final_message.unwrap_or_else(|| {
        let mut message = stream
            .partial()
            .cloned()
            .unwrap_or_else(|| AssistantMessage::empty(&config.model.api, &config.model.provider, &config.model.id));
        message.stop_reason = StopReason::Error;
        message.error_message = Some("Stream ended without a final message".into());
        message
    });
    // Record the requested level, whichever stream function answered.
    message.thinking_level = Some(thinking_level);
    if !started {
        sink.emit(AgentEvent::MessageStart {
            message: message.clone().into(),
        })
        .await;
    }
    sink.emit(AgentEvent::MessageEnd {
        message: message.clone().into(),
    })
    .await;
    context.messages.push(message.clone().into());
    message
}

struct ExecutedBatch {
    messages: Vec<ToolResultMessage>,
    terminate: bool,
}

fn should_terminate(outcomes: &[AgentToolCallOutcome]) -> bool {
    !outcomes.is_empty() && outcomes.iter().all(|outcome| outcome.result.terminate)
}

fn args_value(call: &ToolCall) -> Value {
    Value::Object(call.arguments.clone())
}

async fn fail_truncated_tool_calls<C: CustomMessage>(
    calls: &[ToolCall],
    sink: &dyn AgentEventSink<C>,
) -> ExecutedBatch {
    let mut messages = Vec::new();
    for call in calls {
        sink.emit(AgentEvent::ToolExecutionStart {
            tool_call_id: call.id.clone(),
            tool_name: call.name.clone(),
            args: args_value(call),
        })
        .await;
        let outcome = AgentToolCallOutcome {
            tool_call: call.clone(),
            result: AgentToolResult::error(format!(
                "Tool call \"{}\" was not executed: the response hit the output token limit, so its arguments may be \
                 truncated. Re-issue the tool call with complete arguments.",
                call.name
            )),
            is_error: true,
            duration_ms: None,
        };
        emit_tool_execution_end(&outcome, sink).await;
        let message = tool_result_message(&outcome);
        emit_tool_result_message(&message, sink).await;
        messages.push(message);
    }
    ExecutedBatch {
        messages,
        terminate: false,
    }
}

async fn execute_tool_calls<C: CustomMessage>(
    context: &AgentContext<C>,
    assistant: &AssistantMessage,
    calls: &[ToolCall],
    config: &AgentLoopConfig<C>,
    cancel: &CancellationToken,
    sink: &dyn AgentEventSink<C>,
) -> ExecutedBatch {
    let has_sequential = calls.iter().any(|call| {
        find_tool(&context.tools, &call.name)
            .is_some_and(|tool| tool.execution_mode() == Some(ToolExecutionMode::Sequential))
    });
    if config.tool_execution == ToolExecutionMode::Sequential || has_sequential {
        execute_sequential(context, assistant, calls, config, cancel, sink).await
    } else {
        execute_parallel(context, assistant, calls, config, cancel, sink).await
    }
}

fn find_tool<'a>(tools: &'a [Arc<dyn AgentTool>], name: &str) -> Option<&'a Arc<dyn AgentTool>> {
    tools.iter().find(|tool| tool.definition().name == name)
}

async fn execute_sequential<C: CustomMessage>(
    context: &AgentContext<C>,
    assistant: &AssistantMessage,
    calls: &[ToolCall],
    config: &AgentLoopConfig<C>,
    cancel: &CancellationToken,
    sink: &dyn AgentEventSink<C>,
) -> ExecutedBatch {
    let mut outcomes = Vec::new();
    let mut messages = Vec::new();
    for call in calls {
        sink.emit(AgentEvent::ToolExecutionStart {
            tool_call_id: call.id.clone(),
            tool_name: call.name.clone(),
            args: args_value(call),
        })
        .await;
        let outcome = match prepare_tool_call(context, assistant, call, &context.tools, config, cancel).await {
            Prepared::Immediate(outcome) => outcome,
            Prepared::Ready(prepared) => {
                let executed = execute_prepared(&prepared, cancel, Some(sink)).await;
                finalize(context, assistant, &prepared, executed, config, cancel).await
            }
        };
        emit_tool_execution_end(&outcome, sink).await;
        let message = tool_result_message(&outcome);
        emit_tool_result_message(&message, sink).await;
        outcomes.push(outcome);
        messages.push(message);
        if cancel.is_cancelled() {
            break;
        }
    }
    ExecutedBatch {
        messages,
        terminate: should_terminate(&outcomes),
    }
}

async fn execute_parallel<C: CustomMessage>(
    context: &AgentContext<C>,
    assistant: &AssistantMessage,
    calls: &[ToolCall],
    config: &AgentLoopConfig<C>,
    cancel: &CancellationToken,
    sink: &dyn AgentEventSink<C>,
) -> ExecutedBatch {
    let mut entries: Vec<BoxFuture<'_, AgentToolCallOutcome>> = Vec::new();
    for call in calls {
        sink.emit(AgentEvent::ToolExecutionStart {
            tool_call_id: call.id.clone(),
            tool_name: call.name.clone(),
            args: args_value(call),
        })
        .await;
        match prepare_tool_call(context, assistant, call, &context.tools, config, cancel).await {
            Prepared::Immediate(outcome) => {
                emit_tool_execution_end(&outcome, sink).await;
                entries.push(Box::pin(async move { outcome }));
            }
            Prepared::Ready(prepared) => {
                entries.push(Box::pin(async move {
                    let outcome = if cancel.is_cancelled() {
                        AgentToolCallOutcome {
                            tool_call: prepared.call.clone(),
                            result: AgentToolResult::error("Operation aborted"),
                            is_error: true,
                            duration_ms: None,
                        }
                    } else {
                        let executed = execute_prepared(&prepared, cancel, Some(sink)).await;
                        finalize(context, assistant, &prepared, executed, config, cancel).await
                    };
                    emit_tool_execution_end(&outcome, sink).await;
                    outcome
                }));
            }
        }
        if cancel.is_cancelled() {
            break;
        }
    }
    let outcomes = join_all(entries).await;
    let mut messages = Vec::new();
    for outcome in &outcomes {
        let message = tool_result_message(outcome);
        emit_tool_result_message(&message, sink).await;
        messages.push(message);
    }
    ExecutedBatch {
        messages,
        terminate: should_terminate(&outcomes),
    }
}

struct PreparedCall {
    call: ToolCall,
    tool: Arc<dyn AgentTool>,
    args: Value,
}

enum Prepared {
    Ready(PreparedCall),
    Immediate(AgentToolCallOutcome),
}

fn immediate(call: &ToolCall, result: AgentToolResult) -> Prepared {
    Prepared::Immediate(AgentToolCallOutcome {
        tool_call: call.clone(),
        result,
        is_error: true,
        duration_ms: None,
    })
}

async fn prepare_tool_call<C: CustomMessage>(
    context: &AgentContext<C>,
    assistant: &AssistantMessage,
    call: &ToolCall,
    tools: &[Arc<dyn AgentTool>],
    config: &AgentLoopConfig<C>,
    cancel: &CancellationToken,
) -> Prepared {
    let Some(tool) = find_tool(tools, &call.name) else {
        return immediate(call, AgentToolResult::error(format!("Tool {} not found", call.name)));
    };
    let mut prepared_call = call.clone();
    prepared_call.arguments = tool.prepare_arguments(call.arguments.clone());
    let args = match validate_tool_arguments(tool.definition(), &prepared_call) {
        Ok(args) => args,
        Err(error) => return immediate(call, AgentToolResult::error(error.0)),
    };
    let before = BeforeToolCallContext {
        assistant_message: assistant,
        tool_call: call,
        args: &args,
        context,
    };
    let decision = config.hooks.before_tool_call(before, cancel).await;
    if cancel.is_cancelled() {
        return immediate(call, AgentToolResult::error("Operation aborted"));
    }
    if let Some(decision) = decision
        && decision.block
    {
        let mut result = AgentToolResult::error(decision.reason.unwrap_or_else(|| "Tool execution was blocked".into()));
        result.terminate = decision.terminate;
        return immediate(call, result);
    }
    Prepared::Ready(PreparedCall {
        call: call.clone(),
        tool: tool.clone(),
        args,
    })
}

struct Executed {
    result: AgentToolResult,
    is_error: bool,
    duration_ms: u64,
}

async fn execute_prepared<C: CustomMessage>(
    prepared: &PreparedCall,
    cancel: &CancellationToken,
    sink: Option<&dyn AgentEventSink<C>>,
) -> Executed {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let context = ToolContext {
        tool_call_id: prepared.call.id.clone(),
        cancel: cancel.clone(),
        updates: ToolUpdates::new(tx),
    };
    let started = Instant::now();
    let execution = prepared.tool.execute(prepared.args.clone(), context);
    tokio::pin!(execution);
    let emit_update = |partial: AgentToolResult| async move {
        if let Some(sink) = sink {
            sink.emit(AgentEvent::ToolExecutionUpdate {
                tool_call_id: prepared.call.id.clone(),
                tool_name: prepared.call.name.clone(),
                args: args_value(&prepared.call),
                partial_result: partial,
            })
            .await;
        }
    };
    let outcome = loop {
        tokio::select! {
            biased;
            Some(partial) = rx.recv() => emit_update(partial).await,
            outcome = &mut execution => break outcome,
        }
    };
    let duration_ms = started.elapsed().as_millis() as u64;
    // Updates sent before `execute` returned are still delivered; later ones are dropped.
    rx.close();
    while let Ok(partial) = rx.try_recv() {
        emit_update(partial).await;
    }
    match outcome {
        Ok(result) => Executed {
            is_error: result.is_error,
            result,
            duration_ms,
        },
        Err(error) => Executed {
            result: AgentToolResult::error(error.to_string()),
            is_error: true,
            duration_ms,
        },
    }
}

async fn finalize<C: CustomMessage>(
    context: &AgentContext<C>,
    assistant: &AssistantMessage,
    prepared: &PreparedCall,
    executed: Executed,
    config: &AgentLoopConfig<C>,
    cancel: &CancellationToken,
) -> AgentToolCallOutcome {
    let Executed {
        mut result,
        mut is_error,
        duration_ms,
    } = executed;
    let after = AfterToolCallContext {
        assistant_message: assistant,
        tool_call: &prepared.call,
        args: &prepared.args,
        result: &result,
        is_error,
        context,
    };
    if let Some(update) = config.hooks.after_tool_call(after, cancel).await {
        // Structured content not replaced along with the content may no longer match it.
        let structured = update.structured_content.or_else(|| {
            if update.content.is_some() {
                None
            } else {
                result.structured_content.take()
            }
        });
        if let Some(content) = update.content {
            result.content = content;
        }
        if let Some(details) = update.details {
            result.details = Some(details);
        }
        if let Some(usage) = update.usage {
            result.usage = Some(usage);
        }
        if let Some(terminate) = update.terminate {
            result.terminate = terminate;
        }
        result.structured_content = structured;
        if let Some(flag) = update.is_error {
            is_error = flag;
        }
    }
    AgentToolCallOutcome {
        tool_call: prepared.call.clone(),
        result,
        is_error,
        duration_ms: Some(duration_ms),
    }
}

/// Options of [`run_tool_call`].
pub struct RunToolCallOptions<'a, C: CustomMessage> {
    /// Tools the call resolves against.
    pub tools: &'a [Arc<dyn AgentTool>],
    /// Passed to the hooks as the message that issued the call.
    pub assistant_message: &'a AssistantMessage,
    /// Passed to the hooks as the current context.
    pub context: &'a AgentContext<C>,
    pub config: &'a AgentLoopConfig<C>,
    pub cancel: CancellationToken,
}

impl<C: CustomMessage> std::fmt::Debug for RunToolCallOptions<'_, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunToolCallOptions")
            .field("tools", &self.tools.len())
            .finish_non_exhaustive()
    }
}

/// Run one tool call through the same steps as a model-issued call (argument preparation,
/// validation, `before_tool_call`, execution, `after_tool_call`) without events or messages.
/// Tools that call other tools use this so the hooks apply to those calls too. Failures come
/// back as `is_error` outcomes.
pub async fn run_tool_call<C: CustomMessage>(
    call: &ToolCall,
    options: RunToolCallOptions<'_, C>,
) -> AgentToolCallOutcome {
    let RunToolCallOptions {
        tools,
        assistant_message,
        context,
        config,
        cancel,
    } = options;
    match prepare_tool_call(context, assistant_message, call, tools, config, &cancel).await {
        Prepared::Immediate(outcome) => outcome,
        Prepared::Ready(prepared) => {
            let executed = execute_prepared::<C>(&prepared, &cancel, None).await;
            finalize(context, assistant_message, &prepared, executed, config, &cancel).await
        }
    }
}

async fn emit_tool_execution_end<C>(outcome: &AgentToolCallOutcome, sink: &dyn AgentEventSink<C>) {
    sink.emit(AgentEvent::ToolExecutionEnd {
        tool_call_id: outcome.tool_call.id.clone(),
        tool_name: outcome.tool_call.name.clone(),
        result: outcome.result.clone(),
        is_error: outcome.is_error,
        duration_ms: outcome.duration_ms,
    })
    .await;
}

fn tool_result_message(outcome: &AgentToolCallOutcome) -> ToolResultMessage {
    ToolResultMessage {
        tool_call_id: outcome.tool_call.id.clone(),
        tool_name: outcome.tool_call.name.clone(),
        content: outcome.result.content.clone(),
        details: outcome.result.details.clone(),
        usage: outcome.result.usage.clone(),
        nested_calls: None,
        is_error: outcome.is_error,
        timestamp: tau_ai::types::now_ms(),
        duration_ms: outcome.duration_ms,
    }
}

async fn emit_tool_result_message<C: CustomMessage>(message: &ToolResultMessage, sink: &dyn AgentEventSink<C>) {
    sink.emit(AgentEvent::MessageStart {
        message: message.clone().into(),
    })
    .await;
    sink.emit(AgentEvent::MessageEnd {
        message: message.clone().into(),
    })
    .await;
}
