//! The stateful [`Agent`]: owns the transcript, emits lifecycle events to listeners, runs
//! the loop and exposes the steering and follow-up queues.

use futures::future::BoxFuture;
use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use delta_ai::options::{OnPayload, OnProviderStreamEvent, OnResponse, ThinkingBudgets, Transport};
use delta_ai::transcript::{create_initial_system_message, get_current_system_message, get_current_system_prompt};
use delta_ai::{
    AssistantMessage, ImageContent, InputModality, Message, Model, ModelCost, ModelThinkingLevel, SimpleStreamOptions,
    StreamOptions, UserBlock, UserContent, UserMessage,
};

use crate::agent_loop::{AgentEventSink, run_agent_loop, run_agent_loop_continue};
use crate::hooks::{AgentContext, AgentHooks, AgentLoopConfig, DefaultHooks, MessageSource, StreamFn};
use crate::tool::AgentTool;
use crate::types::{AgentEvent, AgentMessage, CustomMessage, NoCustom, QueueMode, ToolExecutionMode};

/// Errors of starting a run or resetting.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AgentError {
    #[error(
        "Agent is already processing a prompt. Use steer() or follow_up() to queue messages, or wait for completion."
    )]
    AlreadyRunning,
    #[error("Agent is already processing. Wait for completion before resetting.")]
    ResetWhileRunning,
    #[error("Agent is already processing. Wait for completion before continuing.")]
    ContinueWhileRunning,
    #[error("No messages to continue from")]
    NoMessages,
    #[error("Cannot continue from message role: assistant")]
    AssistantTail,
}

/// The message being streamed: any message between its start and end events, or the
/// assistant response as of the last update.
#[derive(Clone, Debug)]
pub enum StreamingMessage<C> {
    Message(AgentMessage<C>),
    Assistant(Arc<AssistantMessage>),
}

/// Agent state. Read with [`Agent::with_state`], change with the setters.
pub struct AgentState<C = NoCustom> {
    /// Model of future turns.
    pub model: Model,
    /// Requested reasoning level of future turns.
    pub thinking_level: ModelThinkingLevel,
    /// Executable tools. Differences from the tools the transcript declares are announced
    /// to the model with a system message before the next request.
    pub tools: Vec<Arc<dyn AgentTool>>,
    /// Transcript. System messages carry the prompt and tool declarations.
    pub messages: Vec<AgentMessage<C>>,
    /// True while a run is in progress, until the `AgentEnd` listeners finish.
    pub is_streaming: bool,
    pub streaming_message: Option<StreamingMessage<C>>,
    /// Tool call ids currently executing.
    pub pending_tool_calls: HashSet<String>,
    /// Error of the most recent failed or aborted turn.
    pub error_message: Option<String>,
}

impl<C: CustomMessage> AgentState<C> {
    /// Current system prompt, replayed from the transcript's system messages.
    pub fn system_prompt(&self) -> String {
        get_current_system_prompt(&llm_system_messages(&self.messages))
    }
}

impl<C: fmt::Debug> fmt::Debug for AgentState<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentState")
            .field("model", &self.model.id)
            .field("thinking_level", &self.thinking_level)
            .field(
                "tools",
                &self
                    .tools
                    .iter()
                    .map(|tool| tool.definition().name.clone())
                    .collect::<Vec<_>>(),
            )
            .field("messages", &self.messages.len())
            .field("is_streaming", &self.is_streaming)
            .field("error_message", &self.error_message)
            .finish_non_exhaustive()
    }
}

fn llm_system_messages<C: CustomMessage>(messages: &[AgentMessage<C>]) -> Vec<Message> {
    messages
        .iter()
        .filter(|message| message.is_system())
        .filter_map(|message| message.as_llm().cloned())
        .collect()
}

/// Initial state. `system_prompt` and `tools` become the leading system message unless
/// `messages` already starts with one.
pub struct AgentInitialState<C = NoCustom> {
    pub system_prompt: Option<String>,
    pub model: Option<Model>,
    pub thinking_level: ModelThinkingLevel,
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub messages: Vec<AgentMessage<C>>,
}

impl<C> Default for AgentInitialState<C> {
    fn default() -> Self {
        Self {
            system_prompt: None,
            model: None,
            thinking_level: ModelThinkingLevel::Off,
            tools: Vec::new(),
            messages: Vec::new(),
        }
    }
}

impl<C: fmt::Debug> fmt::Debug for AgentInitialState<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentInitialState")
            .field("model", &self.model.as_ref().map(|model| &model.id))
            .field("messages", &self.messages.len())
            .finish_non_exhaustive()
    }
}

/// Options of [`Agent::new`].
pub struct AgentOptions<C: CustomMessage = NoCustom> {
    pub initial_state: AgentInitialState<C>,
    pub stream_fn: StreamFn,
    pub hooks: Arc<dyn AgentHooks<C>>,
    pub steering_mode: QueueMode,
    pub follow_up_mode: QueueMode,
    pub session_id: Option<String>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    /// Default: auto.
    pub transport: Transport,
    pub max_retry_delay: Option<Duration>,
    pub tool_execution: ToolExecutionMode,
    pub on_payload: Option<OnPayload>,
    pub on_response: Option<OnResponse>,
    pub on_provider_stream_event: Option<OnProviderStreamEvent>,
}

impl<C: CustomMessage> AgentOptions<C> {
    pub fn new(stream_fn: StreamFn) -> Self {
        Self {
            initial_state: AgentInitialState::default(),
            stream_fn,
            hooks: Arc::new(DefaultHooks),
            steering_mode: QueueMode::OneAtATime,
            follow_up_mode: QueueMode::OneAtATime,
            session_id: None,
            thinking_budgets: None,
            transport: Transport::Auto,
            max_retry_delay: None,
            tool_execution: ToolExecutionMode::Parallel,
            on_payload: None,
            on_response: None,
            on_provider_stream_event: None,
        }
    }
}

impl<C: CustomMessage> fmt::Debug for AgentOptions<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentOptions")
            .field("initial_state", &self.initial_state)
            .finish_non_exhaustive()
    }
}

/// Listener of agent events. Listeners run in subscription order and are awaited; they
/// receive the run's cancellation token.
pub type AgentListener<C> = Arc<dyn Fn(Arc<AgentEvent<C>>, CancellationToken) -> BoxFuture<'static, ()> + Send + Sync>;

/// Handle of a subscription, for [`Agent::unsubscribe`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubscriptionId(u64);

#[derive(Debug, Default)]
struct PendingQueue<C> {
    messages: VecDeque<AgentMessage<C>>,
    mode: QueueMode,
}

impl<C: Clone> PendingQueue<C> {
    fn peek(&self) -> Vec<AgentMessage<C>> {
        match self.mode {
            QueueMode::All => self.messages.iter().cloned().collect(),
            QueueMode::OneAtATime => self.messages.front().cloned().into_iter().collect(),
        }
    }

    fn drain(&mut self) -> Vec<AgentMessage<C>> {
        let count = match self.mode {
            QueueMode::All => self.messages.len(),
            QueueMode::OneAtATime => self.messages.len().min(1),
        };
        self.messages.drain(..count).collect()
    }
}

/// Next subscription number and the listeners in subscription order.
type Listeners<C> = (u64, Vec<(SubscriptionId, AgentListener<C>)>);

struct Settings<C: CustomMessage> {
    stream_fn: StreamFn,
    hooks: Arc<dyn AgentHooks<C>>,
    session_id: Option<String>,
    thinking_budgets: Option<ThinkingBudgets>,
    transport: Transport,
    max_retry_delay: Option<Duration>,
    tool_execution: ToolExecutionMode,
    on_payload: Option<OnPayload>,
    on_response: Option<OnResponse>,
    on_provider_stream_event: Option<OnProviderStreamEvent>,
}

struct Inner<C: CustomMessage> {
    state: Mutex<AgentState<C>>,
    settings: Mutex<Settings<C>>,
    listeners: Mutex<Listeners<C>>,
    steering: Mutex<PendingQueue<C>>,
    follow_up: Mutex<PendingQueue<C>>,
    /// Cancellation of the active run; `None` when idle.
    active: Mutex<Option<CancellationToken>>,
    /// True while idle. Update it with `send_replace`: the agent holds no receiver, so
    /// `send` would fail and leave the value unchanged.
    idle: watch::Sender<bool>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Stateful wrapper around the agent loop. Cheap to clone; clones share the agent, so one
/// task can `prompt` while another steers or aborts.
pub struct Agent<C: CustomMessage = NoCustom> {
    inner: Arc<Inner<C>>,
}

impl<C: CustomMessage> Clone for Agent<C> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<C: CustomMessage> fmt::Debug for Agent<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("state", &*lock(&self.inner.state))
            .finish_non_exhaustive()
    }
}

fn unknown_model() -> Model {
    Model {
        id: "unknown".into(),
        name: "unknown".into(),
        api: "unknown".into(),
        provider: "unknown".into(),
        base_url: String::new(),
        input: Vec::<InputModality>::new(),
        input_limits: None,
        cost: ModelCost::default(),
        headers: None,
        reasoning: false,
        thinking_level_map: None,
        prompt_cache: None,
        context_window: 0,
        max_tokens: 0,
        sampling_params: None,
        sampling_params_by_thinking_level: None,
        compat: None,
    }
}

/// Clears run state when a run ends, including when the run's future is dropped. A dropped
/// run is also cancelled, so tools still holding its token stop.
struct RunGuard<C: CustomMessage> {
    inner: Arc<Inner<C>>,
    finished: bool,
}

impl<C: CustomMessage> RunGuard<C> {
    fn finish(mut self) {
        self.finished = true;
    }
}

impl<C: CustomMessage> Drop for RunGuard<C> {
    fn drop(&mut self) {
        if !self.finished
            && let Some(cancel) = lock(&self.inner.active).as_ref()
        {
            cancel.cancel();
        }
        {
            let mut state = lock(&self.inner.state);
            state.is_streaming = false;
            state.streaming_message = None;
            state.pending_tool_calls.clear();
        }
        *lock(&self.inner.active) = None;
        self.inner.idle.send_replace(true);
    }
}

impl<C: CustomMessage> Agent<C> {
    pub fn new(options: AgentOptions<C>) -> Self {
        let AgentOptions {
            initial_state,
            stream_fn,
            hooks,
            steering_mode,
            follow_up_mode,
            session_id,
            thinking_budgets,
            transport,
            max_retry_delay,
            tool_execution,
            on_payload,
            on_response,
            on_provider_stream_event,
        } = options;
        let mut messages = initial_state.messages;
        let declarations: Vec<_> = initial_state
            .tools
            .iter()
            .map(|tool| tool.definition().clone())
            .collect();
        if !messages.first().is_some_and(AgentMessage::is_system)
            && let Some(initial) =
                create_initial_system_message(initial_state.system_prompt.as_deref(), Some(&declarations))
        {
            messages.insert(0, initial.into());
        }
        let state = AgentState {
            model: initial_state.model.unwrap_or_else(unknown_model),
            thinking_level: initial_state.thinking_level,
            tools: initial_state.tools,
            messages,
            is_streaming: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
        };
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(state),
                settings: Mutex::new(Settings {
                    stream_fn,
                    hooks,
                    session_id,
                    thinking_budgets,
                    transport,
                    max_retry_delay,
                    tool_execution,
                    on_payload,
                    on_response,
                    on_provider_stream_event,
                }),
                listeners: Mutex::new((0, Vec::new())),
                steering: Mutex::new(PendingQueue {
                    messages: VecDeque::new(),
                    mode: steering_mode,
                }),
                follow_up: Mutex::new(PendingQueue {
                    messages: VecDeque::new(),
                    mode: follow_up_mode,
                }),
                active: Mutex::new(None),
                idle: watch::channel(true).0,
            }),
        }
    }

    /// Subscribe to events. Listeners are awaited in subscription order and count toward
    /// the run's settlement.
    pub fn subscribe<F, Fut>(&self, listener: F) -> SubscriptionId
    where
        F: Fn(Arc<AgentEvent<C>>, CancellationToken) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let listener: AgentListener<C> = Arc::new(move |event, cancel| Box::pin(listener(event, cancel)));
        let mut listeners = lock(&self.inner.listeners);
        listeners.0 += 1;
        let id = SubscriptionId(listeners.0);
        listeners.1.push((id, listener));
        id
    }

    pub fn unsubscribe(&self, id: SubscriptionId) {
        lock(&self.inner.listeners).1.retain(|(existing, _)| *existing != id);
    }

    /// Read the state.
    pub fn with_state<R>(&self, f: impl FnOnce(&AgentState<C>) -> R) -> R {
        f(&lock(&self.inner.state))
    }

    /// Change the state. Changes to `is_streaming`, `streaming_message` and
    /// `pending_tool_calls` are overwritten by the next run event.
    pub fn with_state_mut<R>(&self, f: impl FnOnce(&mut AgentState<C>) -> R) -> R {
        f(&mut lock(&self.inner.state))
    }

    /// A copy of the transcript.
    pub fn messages(&self) -> Vec<AgentMessage<C>> {
        self.with_state(|state| state.messages.clone())
    }

    pub fn set_messages(&self, messages: Vec<AgentMessage<C>>) {
        self.with_state_mut(|state| state.messages = messages);
    }

    pub fn model(&self) -> Model {
        self.with_state(|state| state.model.clone())
    }

    pub fn set_model(&self, model: Model) {
        self.with_state_mut(|state| state.model = model);
    }

    pub fn set_thinking_level(&self, level: ModelThinkingLevel) {
        self.with_state_mut(|state| state.thinking_level = level);
    }

    pub fn set_tools(&self, tools: Vec<Arc<dyn AgentTool>>) {
        self.with_state_mut(|state| state.tools = tools);
    }

    pub fn system_prompt(&self) -> String {
        self.with_state(AgentState::system_prompt)
    }

    pub fn is_streaming(&self) -> bool {
        self.with_state(|state| state.is_streaming)
    }

    pub fn error_message(&self) -> Option<String> {
        self.with_state(|state| state.error_message.clone())
    }

    pub fn set_hooks(&self, hooks: Arc<dyn AgentHooks<C>>) {
        lock(&self.inner.settings).hooks = hooks;
    }

    pub fn set_stream_fn(&self, stream_fn: StreamFn) {
        lock(&self.inner.settings).stream_fn = stream_fn;
    }

    pub fn set_tool_execution(&self, mode: ToolExecutionMode) {
        lock(&self.inner.settings).tool_execution = mode;
    }

    pub fn set_session_id(&self, session_id: Option<String>) {
        lock(&self.inner.settings).session_id = session_id;
    }

    pub fn set_thinking_budgets(&self, budgets: Option<ThinkingBudgets>) {
        lock(&self.inner.settings).thinking_budgets = budgets;
    }

    pub fn set_steering_mode(&self, mode: QueueMode) {
        lock(&self.inner.steering).mode = mode;
    }

    pub fn steering_mode(&self) -> QueueMode {
        lock(&self.inner.steering).mode
    }

    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        lock(&self.inner.follow_up).mode = mode;
    }

    pub fn follow_up_mode(&self) -> QueueMode {
        lock(&self.inner.follow_up).mode
    }

    /// Queue a message to inject after the current assistant turn finishes.
    pub fn steer(&self, message: impl Into<AgentMessage<C>>) {
        lock(&self.inner.steering).messages.push_back(message.into());
    }

    /// Queue a message to run only after the agent would otherwise stop.
    pub fn follow_up(&self, message: impl Into<AgentMessage<C>>) {
        lock(&self.inner.follow_up).messages.push_back(message.into());
    }

    pub fn clear_steering_queue(&self) {
        lock(&self.inner.steering).messages.clear();
    }

    pub fn clear_follow_up_queue(&self) {
        lock(&self.inner.follow_up).messages.clear();
    }

    pub fn clear_all_queues(&self) {
        self.clear_steering_queue();
        self.clear_follow_up_queue();
    }

    pub fn has_queued_messages(&self) -> bool {
        !lock(&self.inner.steering).messages.is_empty() || !lock(&self.inner.follow_up).messages.is_empty()
    }

    /// The messages the next turn would take, without consuming them.
    pub fn peek_queued_messages(&self) -> Vec<AgentMessage<C>> {
        let steering = lock(&self.inner.steering).peek();
        if steering.is_empty() {
            lock(&self.inner.follow_up).peek()
        } else {
            steering
        }
    }

    /// Cancellation token of the active run.
    pub fn cancel_token(&self) -> Option<CancellationToken> {
        lock(&self.inner.active).clone()
    }

    /// Abort the active run, if any.
    pub fn abort(&self) {
        if let Some(cancel) = lock(&self.inner.active).as_ref() {
            cancel.cancel();
        }
    }

    /// Resolve when the active run and its awaited listeners have finished.
    pub async fn wait_for_idle(&self) {
        let mut idle = self.inner.idle.subscribe();
        let _ = idle.wait_for(|idle| *idle).await;
    }

    /// Clear the conversation and queues, keeping the replayed prompt and tool baseline.
    pub fn reset(&self) -> Result<(), AgentError> {
        if lock(&self.inner.active).is_some() {
            return Err(AgentError::ResetWhileRunning);
        }
        {
            let mut state = lock(&self.inner.state);
            let baseline = get_current_system_message(&llm_system_messages(&state.messages));
            state.messages = baseline.into_iter().map(AgentMessage::from).collect();
            state.is_streaming = false;
            state.streaming_message = None;
            state.pending_tool_calls.clear();
            state.error_message = None;
        }
        self.clear_all_queues();
        Ok(())
    }

    /// Prompt with text and optional images.
    pub async fn prompt_text(&self, text: impl Into<String>, images: Vec<ImageContent>) -> Result<(), AgentError> {
        let mut content = vec![UserBlock::text(text)];
        content.extend(images.into_iter().map(UserBlock::Image));
        self.prompt(vec![UserMessage::new(UserContent::Blocks(content)).into()])
            .await
    }

    /// Start a run with a batch of messages.
    pub async fn prompt(&self, messages: Vec<AgentMessage<C>>) -> Result<(), AgentError> {
        let guard = self.begin_run()?;
        self.run_prompt(messages, false, guard).await;
        Ok(())
    }

    /// Continue from the current transcript. A non-assistant tail is retried; an assistant
    /// tail falls back to one queued steering batch, then one follow-up batch.
    pub async fn continue_run(&self) -> Result<(), AgentError> {
        if lock(&self.inner.active).is_some() {
            return Err(AgentError::ContinueWhileRunning);
        }
        let (last_role, all_system) = self.with_state(|state| {
            (
                state.messages.last().map(|message| message.role().to_owned()),
                state.messages.iter().all(AgentMessage::is_system),
            )
        });
        let Some(last_role) = last_role.filter(|_| !all_system) else {
            return Err(AgentError::NoMessages);
        };
        if last_role == "assistant" {
            let steering = lock(&self.inner.steering).drain();
            if !steering.is_empty() {
                let guard = self.begin_run()?;
                self.run_prompt(steering, true, guard).await;
                return Ok(());
            }
            let follow_ups = lock(&self.inner.follow_up).drain();
            if !follow_ups.is_empty() {
                let guard = self.begin_run()?;
                self.run_prompt(follow_ups, false, guard).await;
                return Ok(());
            }
            return Err(AgentError::AssistantTail);
        }
        let guard = self.begin_run()?;
        let (context, config, stream_fn, cancel) = self.prepare_run(false);
        let sink = AgentSink {
            agent: self.clone(),
            cancel: cancel.clone(),
        };
        let _ = run_agent_loop_continue(context, config, &sink, cancel, &stream_fn).await;
        guard.finish();
        Ok(())
    }

    fn begin_run(&self) -> Result<RunGuard<C>, AgentError> {
        let mut active = lock(&self.inner.active);
        if active.is_some() {
            return Err(AgentError::AlreadyRunning);
        }
        *active = Some(CancellationToken::new());
        drop(active);
        self.inner.idle.send_replace(false);
        let mut state = lock(&self.inner.state);
        state.is_streaming = true;
        state.streaming_message = None;
        state.error_message = None;
        Ok(RunGuard {
            inner: self.inner.clone(),
            finished: false,
        })
    }

    async fn run_prompt(&self, messages: Vec<AgentMessage<C>>, skip_initial_steering_poll: bool, guard: RunGuard<C>) {
        let (context, config, stream_fn, cancel) = self.prepare_run(skip_initial_steering_poll);
        let sink = AgentSink {
            agent: self.clone(),
            cancel: cancel.clone(),
        };
        run_agent_loop(messages, context, config, &sink, cancel, &stream_fn).await;
        guard.finish();
    }

    fn prepare_run(
        &self,
        skip_initial_steering_poll: bool,
    ) -> (AgentContext<C>, AgentLoopConfig<C>, StreamFn, CancellationToken) {
        let cancel = lock(&self.inner.active).clone().unwrap_or_default();
        let (context, model, thinking_level) = self.with_state(|state| {
            (
                AgentContext {
                    messages: state.messages.clone(),
                    tools: state.tools.clone(),
                },
                state.model.clone(),
                state.thinking_level,
            )
        });
        let settings = lock(&self.inner.settings);
        let inner = self.inner.clone();
        let skip = Arc::new(std::sync::atomic::AtomicBool::new(skip_initial_steering_poll));
        let get_steering: MessageSource<C> = Arc::new(move || {
            let messages = if skip.swap(false, std::sync::atomic::Ordering::SeqCst) {
                Vec::new()
            } else {
                lock(&inner.steering).drain()
            };
            Box::pin(async move { messages })
        });
        let inner = self.inner.clone();
        let get_follow_up: MessageSource<C> = Arc::new(move || {
            let messages = lock(&inner.follow_up).drain();
            Box::pin(async move { messages })
        });
        let config = AgentLoopConfig {
            model,
            stream_options: SimpleStreamOptions {
                base: StreamOptions {
                    session_id: settings.session_id.clone(),
                    transport: Some(settings.transport),
                    max_retry_delay: settings.max_retry_delay,
                    on_payload: settings.on_payload.clone(),
                    on_response: settings.on_response.clone(),
                    on_provider_stream_event: settings.on_provider_stream_event.clone(),
                    ..StreamOptions::default()
                },
                reasoning: thinking_level.reasoning(),
                thinking_budgets: settings.thinking_budgets,
                tool_choice: None,
            },
            tool_execution: settings.tool_execution,
            hooks: settings.hooks.clone(),
            get_steering_messages: Some(get_steering),
            get_follow_up_messages: Some(get_follow_up),
        };
        (context, config, settings.stream_fn.clone(), cancel)
    }

    /// Reduce the state for an event, then await the listeners.
    async fn process_event(&self, event: AgentEvent<C>, cancel: &CancellationToken) {
        {
            let mut state = lock(&self.inner.state);
            match &event {
                AgentEvent::MessageStart { message } => {
                    state.streaming_message = Some(StreamingMessage::Message(message.clone()));
                }
                AgentEvent::MessageUpdate { message, .. } => {
                    state.streaming_message = Some(StreamingMessage::Assistant(message.clone()));
                }
                AgentEvent::MessageEnd { message } => {
                    state.streaming_message = None;
                    state.messages.push(message.clone());
                }
                AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
                    state.pending_tool_calls.insert(tool_call_id.clone());
                }
                AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                    state.pending_tool_calls.remove(tool_call_id);
                }
                AgentEvent::TurnEnd { message, .. } => {
                    if let Some(error) = message.as_assistant().and_then(|message| message.error_message.clone()) {
                        state.error_message = Some(error);
                    }
                }
                AgentEvent::AgentEnd { .. } => state.streaming_message = None,
                _ => {}
            }
        }
        let listeners: Vec<AgentListener<C>> = lock(&self.inner.listeners)
            .1
            .iter()
            .map(|(_, listener)| listener.clone())
            .collect();
        if listeners.is_empty() {
            return;
        }
        let event = Arc::new(event);
        for listener in listeners {
            listener(event.clone(), cancel.clone()).await;
        }
    }
}

struct AgentSink<C: CustomMessage> {
    agent: Agent<C>,
    cancel: CancellationToken,
}

impl<C: CustomMessage> AgentEventSink<C> for AgentSink<C> {
    fn emit(&self, event: AgentEvent<C>) -> BoxFuture<'_, ()> {
        Box::pin(self.agent.process_event(event, &self.cancel))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use delta_ai::events::MessageBuilder;
    use delta_ai::{AssistantMessageEventStream, Message};

    /// Starts a response and waits for cancellation; `started` is notified once the request
    /// is in flight.
    fn hanging_stream_fn(started: Arc<tokio::sync::Notify>, cancelled: Arc<tokio::sync::Notify>) -> StreamFn {
        Arc::new(move |model, _context, options| {
            let started = started.clone();
            let cancelled = cancelled.clone();
            Box::pin(async move {
                let (sender, stream) = AssistantMessageEventStream::channel();
                let cancel = options.base.cancel.expect("loop sets the cancel token");
                tokio::spawn(async move {
                    let mut builder =
                        MessageBuilder::new(sender, AssistantMessage::empty(&model.api, &model.provider, &model.id));
                    builder.start();
                    started.notify_one();
                    cancel.cancelled().await;
                    cancelled.notify_one();
                    builder.fail("Request aborted by user", true);
                });
                stream
            })
        })
    }

    #[tokio::test]
    async fn dropping_the_prompt_future_cancels_the_run() {
        let started = Arc::new(tokio::sync::Notify::new());
        let cancelled = Arc::new(tokio::sync::Notify::new());
        let agent: Agent = Agent::new(AgentOptions::new(hanging_stream_fn(started.clone(), cancelled.clone())));
        let run = agent.prompt_text("hi", Vec::new());
        tokio::select! {
            _ = run => panic!("the run cannot finish before it is cancelled"),
            () = started.notified() => {}
        }
        // `run` was dropped at the end of the select.
        tokio::time::timeout(Duration::from_secs(5), cancelled.notified())
            .await
            .expect("run cancelled");
        assert!(!agent.is_streaming());
        agent.wait_for_idle().await;
        assert!(matches!(
            agent.messages().last(),
            Some(AgentMessage::Llm(Message::User(_)))
        ));
    }
}
