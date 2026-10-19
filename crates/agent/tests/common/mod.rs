//! Helpers shared by the agent integration tests: a mock model, scripted stream functions,
//! closure-based hooks and event recorders.
#![allow(dead_code, unreachable_pub, clippy::unwrap_used)]

use futures::StreamExt;
use futures::future::BoxFuture;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use tau_agent::{
    AfterToolCallContext, AfterToolCallResult, AgentEvent, AgentEventSink, AgentEventStream, AgentHooks,
    AgentLoopConfig, AgentLoopTurnUpdate, AgentMessage, AgentRequestUpdate, AgentToolResult, AgentTurnContext,
    AgentTurnDecision, BeforeToolCallContext, BeforeToolCallResult, CustomMessage, FnTool, MessageSource, NoCustom,
    PrepareRequestContext, StreamFn, ToolContext, ToolError,
};
use tau_ai::{
    AssistantBlock, AssistantMessage, AssistantMessageEventStream, InputModality, Message, MessageBuilder, Model,
    ModelCost, SimpleStreamOptions, StopReason, TextContent, Tool, ToolCall, TranscriptContext, UserContent,
    UserMessage,
};

/// Lock a test mutex.
pub fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Shared, lockable test state.
pub type Shared<T> = Arc<Mutex<T>>;

pub fn shared<T>(value: T) -> Shared<T> {
    Arc::new(Mutex::new(value))
}

pub fn model() -> Model {
    Model {
        id: "mock".into(),
        name: "mock".into(),
        api: "openai-responses".into(),
        provider: "openai".into(),
        base_url: "https://example.invalid".into(),
        input: vec![InputModality::Text],
        input_limits: None,
        cost: ModelCost::default(),
        headers: None,
        reasoning: false,
        thinking_level_map: None,
        prompt_cache: None,
        context_window: 8192,
        max_tokens: 2048,
        sampling_params: None,
        sampling_params_by_thinking_level: None,
        compat: None,
    }
}

pub fn user_message(text: &str) -> UserMessage {
    UserMessage::new(text)
}

pub fn user<C>(text: &str) -> AgentMessage<C> {
    user_message(text).into()
}

pub fn text(text: &str) -> AssistantBlock {
    AssistantBlock::Text(TextContent::new(text))
}

pub fn tool_call(id: &str, name: &str, arguments: Value) -> AssistantBlock {
    AssistantBlock::ToolCall(call(id, name, arguments))
}

pub fn call(id: &str, name: &str, arguments: Value) -> ToolCall {
    let Value::Object(arguments) = arguments else {
        panic!("tool call arguments must be an object")
    };
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments,
        ..ToolCall::default()
    }
}

pub fn assistant(content: Vec<AssistantBlock>, stop_reason: StopReason) -> AssistantMessage {
    AssistantMessage {
        content,
        stop_reason,
        ..AssistantMessage::empty("openai-responses", "openai", "mock")
    }
}

pub fn assistant_text(value: &str) -> AssistantMessage {
    assistant(vec![text(value)], StopReason::Stop)
}

/// A response stream holding only the terminal event for `message`.
pub fn respond(message: AssistantMessage) -> AssistantMessageEventStream {
    AssistantMessageEventStream::from_final(message)
}

/// A failed response with `reason` (`Error` or `Aborted`) and `error_message`.
pub fn failed(reason: StopReason, error_message: &str) -> AssistantMessage {
    AssistantMessage {
        error_message: Some(error_message.into()),
        ..assistant(Vec::new(), reason)
    }
}

/// Stream function from a closure that gets the 0-based request index.
pub fn stream_fn<F>(respond: F) -> StreamFn
where
    F: Fn(usize, &Model, &TranscriptContext, &SimpleStreamOptions) -> AssistantMessageEventStream
        + Send
        + Sync
        + 'static,
{
    let calls = AtomicUsize::new(0);
    Arc::new(move |model, context, options| {
        let index = calls.fetch_add(1, Ordering::SeqCst);
        let stream = respond(index, &model, &context, &options);
        Box::pin(async move { stream })
    })
}

/// Stream function that answers each request with the next message; the last one repeats.
pub fn scripted(messages: Vec<AssistantMessage>) -> StreamFn {
    stream_fn(move |index, _, _, _| respond(messages[index.min(messages.len() - 1)].clone()))
}

/// Stream function that starts a response and fails it as aborted once the request's
/// cancellation fires.
pub fn abortable() -> StreamFn {
    Arc::new(|model, _context, options| {
        let (sender, stream) = AssistantMessageEventStream::channel();
        let cancel = options.base.cancel.clone().expect("the loop sets a cancellation token");
        tokio::spawn(async move {
            let mut builder =
                MessageBuilder::new(sender, AssistantMessage::empty(&model.api, &model.provider, &model.id));
            builder.start();
            cancel.cancelled().await;
            builder.fail("Request aborted by user", true);
        });
        Box::pin(async move { stream })
    })
}

/// User texts with plain string content, in request order.
pub fn user_texts(context: &TranscriptContext) -> Vec<String> {
    context
        .messages()
        .iter()
        .filter_map(|message| match message {
            Message::User(UserMessage {
                content: UserContent::Text(text),
                ..
            }) => Some(text.clone()),
            _ => None,
        })
        .collect()
}

/// Text of a user message, string or blocks.
pub fn user_text<C>(message: &AgentMessage<C>) -> Option<String> {
    match message {
        AgentMessage::Llm(Message::User(user)) => Some(user.content.text()),
        _ => None,
    }
}

pub fn roles<C: CustomMessage>(messages: &[AgentMessage<C>]) -> Vec<String> {
    messages.iter().map(|message| message.role().to_owned()).collect()
}

pub fn kinds<C>(events: &[AgentEvent<C>]) -> Vec<&'static str> {
    events.iter().map(AgentEvent::kind).collect()
}

/// Every event of a loop and the messages it returned.
pub async fn collect<C: CustomMessage>(mut stream: AgentEventStream<C>) -> (Vec<AgentEvent<C>>, Vec<AgentMessage<C>>) {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    (events, stream.result().await)
}

/// The tool result messages among `messages`.
pub fn tool_results<C>(messages: &[AgentMessage<C>]) -> Vec<tau_ai::ToolResultMessage> {
    messages
        .iter()
        .filter_map(|message| match message {
            AgentMessage::Llm(Message::ToolResult(result)) => Some(result.clone()),
            _ => None,
        })
        .collect()
}

/// Text of the first text block.
pub fn first_text(content: &[tau_ai::UserBlock]) -> String {
    content
        .iter()
        .find_map(|block| match block {
            tau_ai::UserBlock::Text(text) => Some(text.text.clone()),
            tau_ai::UserBlock::Image(_) => None,
        })
        .unwrap_or_default()
}

/// `{ value: string }`.
pub fn value_schema() -> Value {
    json!({
        "type": "object",
        "properties": { "value": { "type": "string" } },
        "required": ["value"],
    })
}

pub fn empty_schema() -> Value {
    json!({ "type": "object", "properties": {} })
}

/// A tool from a name, schema and closure.
pub fn tool<F, Fut>(name: &str, parameters: Value, execute: F) -> Arc<dyn tau_agent::AgentTool>
where
    F: Fn(Value, ToolContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<AgentToolResult, ToolError>> + Send + 'static,
{
    Arc::new(FnTool::new(
        Tool::new(name, format!("{name} tool"), parameters),
        execute,
    ))
}

/// A `{ value: string }` tool that records the values it ran with and echoes them.
pub fn echo_tool(executed: Shared<Vec<Value>>) -> Arc<dyn tau_agent::AgentTool> {
    tool("echo", value_schema(), move |args, _| {
        let executed = executed.clone();
        async move {
            executed.lock().unwrap().push(args["value"].clone());
            Ok(AgentToolResult {
                details: Some(json!({ "value": args["value"] })),
                ..AgentToolResult::text(format!("echoed: {}", args["value"].as_str().unwrap_or_default()))
            })
        }
    })
}

/// A tool without parameters that returns its name.
pub fn noop_tool(name: &str) -> Arc<dyn tau_agent::AgentTool> {
    let label = name.to_owned();
    tool(name, empty_schema(), move |_, _| {
        let label = label.clone();
        async move {
            Ok(AgentToolResult {
                details: Some(json!({})),
                ..AgentToolResult::text(label)
            })
        }
    })
}

/// A queue source from a closure.
pub fn source<C, F>(poll: F) -> MessageSource<C>
where
    C: CustomMessage,
    F: Fn() -> Vec<AgentMessage<C>> + Send + Sync + 'static,
{
    Arc::new(move || {
        let messages = poll();
        Box::pin(async move { messages })
    })
}

pub fn config<C: CustomMessage>(hooks: TestHooks<C>) -> AgentLoopConfig<C> {
    AgentLoopConfig {
        hooks: Arc::new(hooks),
        ..AgentLoopConfig::new(model())
    }
}

/// An event sink from a closure, for `run_agent_loop`.
pub struct FnSink<F>(pub F);

impl<C, F> AgentEventSink<C> for FnSink<F>
where
    F: Fn(AgentEvent<C>) + Send + Sync,
{
    fn emit(&self, event: AgentEvent<C>) -> BoxFuture<'_, ()> {
        (self.0)(event);
        Box::pin(async {})
    }
}

/// A one-shot latch: `wait` resolves once `open` was called, before or after.
#[derive(Clone)]
pub struct Gate(Arc<watch::Sender<bool>>);

impl Default for Gate {
    fn default() -> Self {
        Self(Arc::new(watch::channel(false).0))
    }
}

impl Gate {
    pub fn open(&self) {
        self.0.send_replace(true);
    }

    pub fn is_open(&self) -> bool {
        *self.0.borrow()
    }

    pub async fn wait(&self) {
        let mut receiver = self.0.subscribe();
        let _ = receiver.wait_for(|open| *open).await;
    }
}

type ConvertFn<C> = Box<dyn Fn(&[AgentMessage<C>]) -> Vec<Message> + Send + Sync>;
type TransformFn<C> = Box<dyn Fn(Vec<AgentMessage<C>>) -> Vec<AgentMessage<C>> + Send + Sync>;
type BeforeFn<C> = Box<dyn Fn(&BeforeToolCallContext<'_, C>) -> Option<BeforeToolCallResult> + Send + Sync>;
type AfterFn<C> = Box<dyn Fn(&AfterToolCallContext<'_, C>) -> Option<AfterToolCallResult> + Send + Sync>;
type FinishFn<C> = Box<dyn Fn(&AgentTurnContext<'_, C>, &CancellationToken) -> Option<AgentTurnDecision> + Send + Sync>;
type PrepareRequestFn<C> = Box<dyn Fn(&PrepareRequestContext<'_, C>) -> Option<AgentRequestUpdate<C>> + Send + Sync>;
type PrepareNextFn<C> =
    Box<dyn Fn(&AgentTurnContext<'_, C>, &CancellationToken) -> Option<AgentLoopTurnUpdate<C>> + Send + Sync>;

/// Hooks from synchronous closures; unset hooks keep the defaults.
pub struct TestHooks<C: CustomMessage = NoCustom> {
    convert_to_llm: Option<ConvertFn<C>>,
    transform_context: Option<TransformFn<C>>,
    before_tool_call: Option<BeforeFn<C>>,
    after_tool_call: Option<AfterFn<C>>,
    finish_turn: Option<FinishFn<C>>,
    prepare_request: Option<PrepareRequestFn<C>>,
    prepare_next_turn: Option<PrepareNextFn<C>>,
}

impl<C: CustomMessage> Default for TestHooks<C> {
    fn default() -> Self {
        Self {
            convert_to_llm: None,
            transform_context: None,
            before_tool_call: None,
            after_tool_call: None,
            finish_turn: None,
            prepare_request: None,
            prepare_next_turn: None,
        }
    }
}

impl TestHooks {
    /// Hooks without custom messages; use `TestHooks::<C>::default()` for custom ones.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<C: CustomMessage> TestHooks<C> {
    pub fn convert_to_llm(mut self, f: impl Fn(&[AgentMessage<C>]) -> Vec<Message> + Send + Sync + 'static) -> Self {
        self.convert_to_llm = Some(Box::new(f));
        self
    }

    pub fn transform_context(
        mut self,
        f: impl Fn(Vec<AgentMessage<C>>) -> Vec<AgentMessage<C>> + Send + Sync + 'static,
    ) -> Self {
        self.transform_context = Some(Box::new(f));
        self
    }

    pub fn before_tool_call(
        mut self,
        f: impl Fn(&BeforeToolCallContext<'_, C>) -> Option<BeforeToolCallResult> + Send + Sync + 'static,
    ) -> Self {
        self.before_tool_call = Some(Box::new(f));
        self
    }

    pub fn after_tool_call(
        mut self,
        f: impl Fn(&AfterToolCallContext<'_, C>) -> Option<AfterToolCallResult> + Send + Sync + 'static,
    ) -> Self {
        self.after_tool_call = Some(Box::new(f));
        self
    }

    pub fn finish_turn(
        mut self,
        f: impl Fn(&AgentTurnContext<'_, C>, &CancellationToken) -> Option<AgentTurnDecision> + Send + Sync + 'static,
    ) -> Self {
        self.finish_turn = Some(Box::new(f));
        self
    }

    pub fn prepare_request(
        mut self,
        f: impl Fn(&PrepareRequestContext<'_, C>) -> Option<AgentRequestUpdate<C>> + Send + Sync + 'static,
    ) -> Self {
        self.prepare_request = Some(Box::new(f));
        self
    }

    pub fn prepare_next_turn(
        mut self,
        f: impl Fn(&AgentTurnContext<'_, C>, &CancellationToken) -> Option<AgentLoopTurnUpdate<C>> + Send + Sync + 'static,
    ) -> Self {
        self.prepare_next_turn = Some(Box::new(f));
        self
    }
}

impl<C: CustomMessage> AgentHooks<C> for TestHooks<C> {
    fn convert_to_llm<'a>(&'a self, messages: &'a [AgentMessage<C>]) -> BoxFuture<'a, Vec<Message>> {
        let converted = match &self.convert_to_llm {
            Some(f) => f(messages),
            None => messages
                .iter()
                .filter_map(|message| message.as_llm().cloned())
                .collect(),
        };
        Box::pin(async move { converted })
    }

    fn transform_context<'a>(
        &'a self,
        messages: Vec<AgentMessage<C>>,
        _cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Vec<AgentMessage<C>>> {
        let messages = match &self.transform_context {
            Some(f) => f(messages),
            None => messages,
        };
        Box::pin(async move { messages })
    }

    fn before_tool_call<'a>(
        &'a self,
        context: BeforeToolCallContext<'a, C>,
        _cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Option<BeforeToolCallResult>> {
        let result = self.before_tool_call.as_ref().and_then(|f| f(&context));
        Box::pin(async move { result })
    }

    fn after_tool_call<'a>(
        &'a self,
        context: AfterToolCallContext<'a, C>,
        _cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Option<AfterToolCallResult>> {
        let result = self.after_tool_call.as_ref().and_then(|f| f(&context));
        Box::pin(async move { result })
    }

    fn finish_turn<'a>(
        &'a self,
        turn: AgentTurnContext<'a, C>,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Option<AgentTurnDecision>> {
        let result = self.finish_turn.as_ref().and_then(|f| f(&turn, cancel));
        Box::pin(async move { result })
    }

    fn prepare_request<'a>(
        &'a self,
        request: PrepareRequestContext<'a, C>,
        _cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Option<AgentRequestUpdate<C>>> {
        let result = self.prepare_request.as_ref().and_then(|f| f(&request));
        Box::pin(async move { result })
    }

    fn prepare_next_turn<'a>(
        &'a self,
        turn: AgentTurnContext<'a, C>,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Option<AgentLoopTurnUpdate<C>>> {
        let result = self.prepare_next_turn.as_ref().and_then(|f| f(&turn, cancel));
        Box::pin(async move { result })
    }
}
