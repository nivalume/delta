//! A [`ModelClient`] that replays scripted responses: text streamed in chunks, tool calls,
//! errors (overflow, retryable), deferred handles; no network.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use delta_ai::model::{Model, ModelThinkingLevel};
use delta_ai::types::{AssistantMessage, DeferredHandle, JsonObject, Message, StopReason, Usage};
use delta_ai::{AssistantMessageEventStream, MessageBuilder};
use delta_chord::context::Context;
use delta_harness::model::{ConversationStreamOptions, ModelClient, ModelError, ModelRef, ModelRequest};
use futures::future::BoxFuture;
use tokio::sync::Notify;

/// The provider of every fake model.
pub const PROVIDER: &str = "fake";

/// One block of a scripted response.
#[derive(Clone, Debug)]
pub enum Block {
    /// Streamed word by word.
    Text(String),
    ToolCall {
        id: String,
        name: String,
        arguments: JsonObject,
    },
}

/// One scripted response.
#[derive(Clone, Debug)]
pub struct Reply {
    pub blocks: Vec<Block>,
    pub stop: StopReason,
    pub error: Option<String>,
    pub usage: Usage,
    /// Paused before every streamed event.
    pub step: Duration,
    /// When set, the response waits for a permit before it ends.
    pub gate: Option<Arc<Notify>>,
    pub deferred: Option<DeferredHandle>,
}

impl Reply {
    fn new(stop: StopReason) -> Self {
        Self {
            blocks: Vec::new(),
            stop,
            error: None,
            usage: Usage::default(),
            step: Duration::ZERO,
            gate: None,
            deferred: None,
        }
    }

    /// A final answer with `text`.
    pub fn text(text: &str) -> Self {
        Self {
            blocks: vec![Block::Text(text.to_owned())],
            ..Self::new(StopReason::Stop)
        }
    }

    /// A tool-calling answer.
    pub fn calls(calls: &[(&str, &str)]) -> Self {
        let blocks = calls
            .iter()
            .map(|(id, name)| Block::ToolCall {
                id: (*id).to_owned(),
                name: (*name).to_owned(),
                arguments: JsonObject::new(),
            })
            .collect();
        Self {
            blocks,
            ..Self::new(StopReason::ToolUse)
        }
    }

    /// A failed response with `message`.
    pub fn error(message: &str) -> Self {
        Self {
            error: Some(message.to_owned()),
            ..Self::new(StopReason::Error)
        }
    }

    /// A context overflow.
    pub fn overflow() -> Self {
        Self::error("prompt is too long: 250000 tokens > 200000 maximum")
    }

    /// A transient provider failure.
    pub fn retryable() -> Self {
        Self::error("503 Service Unavailable: the model is overloaded")
    }

    /// A deferred response polled after `poll_after_ms`.
    pub fn deferred(id: &str, poll_after_ms: u64) -> Self {
        let handle = DeferredHandle {
            provider: PROVIDER.to_owned(),
            model_id: String::new(),
            api: "fake".to_owned(),
            id: id.to_owned(),
            expires_at: None,
            poll_after_ms: Some(poll_after_ms),
            data: None,
        };
        Self {
            deferred: Some(handle),
            ..Self::new(StopReason::Deferred)
        }
    }

    #[must_use]
    pub fn usage(mut self, input: u64, output: u64) -> Self {
        self.usage = Usage {
            input,
            output,
            total_tokens: input + output,
            ..Usage::default()
        };
        self
    }

    #[must_use]
    pub fn step(mut self, step: Duration) -> Self {
        self.step = step;
        self
    }

    #[must_use]
    pub fn gate(mut self, gate: &Arc<Notify>) -> Self {
        self.gate = Some(gate.clone());
        self
    }
}

/// A request the fake received.
#[derive(Clone, Debug)]
pub struct Recorded {
    pub model: String,
    pub messages: Vec<Message>,
    pub thinking_level: ModelThinkingLevel,
    pub session_id: Option<String>,
    pub options: ConversationStreamOptions,
}

#[derive(Default)]
struct State {
    script: VecDeque<Reply>,
    polls: VecDeque<Reply>,
    requests: Vec<Recorded>,
    cancelled: Vec<String>,
    /// Model ids it knows, with their context window.
    models: Vec<(String, u64)>,
}

/// The scripted model client. Clones share the script.
#[derive(Clone, Default)]
pub struct FakeModel {
    state: Arc<Mutex<State>>,
}

impl std::fmt::Debug for FakeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeModel").finish_non_exhaustive()
    }
}

impl FakeModel {
    /// A fake that knows `model_id` with `context_window`.
    pub fn new(model_id: &str, context_window: u64) -> Self {
        let fake = Self::default();
        fake.state
            .lock()
            .unwrap()
            .models
            .push((model_id.to_owned(), context_window));
        fake
    }

    /// Queues responses for the next requests.
    pub fn push(&self, replies: impl IntoIterator<Item = Reply>) {
        self.state.lock().unwrap().script.extend(replies);
    }

    /// Queues the results of the next deferred polls.
    pub fn push_polls(&self, replies: impl IntoIterator<Item = Reply>) {
        self.state.lock().unwrap().polls.extend(replies);
    }

    /// Every request so far.
    pub fn requests(&self) -> Vec<Recorded> {
        self.state.lock().unwrap().requests.clone()
    }

    /// Ids of the deferred responses cancelled so far.
    pub fn cancelled(&self) -> Vec<String> {
        self.state.lock().unwrap().cancelled.clone()
    }

    /// Forgets `model_id`, so requests for it fail `no_model`.
    pub fn forget(&self, model_id: &str) {
        self.state.lock().unwrap().models.retain(|(id, _)| id != model_id);
    }

    fn reply(&self, request: Recorded) -> Reply {
        let mut state = self.state.lock().unwrap();
        state.requests.push(request);
        state
            .script
            .pop_front()
            .unwrap_or_else(|| Reply::error("the fake model has no scripted response"))
    }
}

/// Streams `reply` through a builder, as a provider would.
fn play(reply: Reply, model: &Model, ctx: Context) -> AssistantMessageEventStream {
    let (sender, stream) = AssistantMessageEventStream::channel();
    let mut builder = MessageBuilder::new(sender, AssistantMessage::empty(&model.api, &model.provider, &model.id));
    tokio::spawn(async move {
        let pause = |step: Duration| {
            let ctx = ctx.clone();
            async move { ctx.run(tokio::time::sleep(step)).await.is_ok() }
        };
        builder.start();
        for block in &reply.blocks {
            match block {
                Block::Text(text) => {
                    let index = builder.start_text();
                    for word in text.split_inclusive(' ') {
                        if !pause(reply.step).await {
                            builder.fail("aborted", true);
                            return;
                        }
                        builder.text_delta(index, word);
                    }
                    builder.end_text(index);
                }
                Block::ToolCall { id, name, arguments } => {
                    let index = builder.start_tool_call(id.clone(), name.clone(), arguments.clone());
                    builder.end_tool_call(index);
                }
            }
        }
        if let Some(gate) = &reply.gate
            && ctx.run(gate.notified()).await.is_err()
        {
            builder.fail("aborted", true);
            return;
        }
        builder.output.usage = reply.usage.clone();
        builder.output.stop_reason = reply.stop;
        builder.output.error_message = reply.error.clone();
        builder.output.deferred = reply.deferred.clone();
        builder.finish();
    });
    stream
}

impl ModelClient for FakeModel {
    fn model(&self, reference: &ModelRef) -> Option<Model> {
        if reference.provider != PROVIDER {
            return None;
        }
        let state = self.state.lock().unwrap();
        let (id, window) = state.models.iter().find(|(id, _)| *id == reference.model_id)?;
        let model = serde_json::json!({
            "id": id, "name": id, "api": "fake", "provider": PROVIDER, "baseUrl": "",
            "input": ["text"], "cost": { "input": 1.0, "output": 2.0, "cacheRead": 0.0, "cacheWrite": 0.0 },
            "contextWindow": window, "maxTokens": 1000
        });
        Some(serde_json::from_value(model).unwrap())
    }

    fn stream(&self, request: ModelRequest<'_>, ctx: &Context) -> AssistantMessageEventStream {
        let recorded = Recorded {
            model: request.model.id.clone(),
            messages: request.messages,
            thinking_level: request.thinking_level,
            session_id: request.session_id.map(str::to_owned),
            options: request.options.clone(),
        };
        let reply = self.reply(recorded);
        play(reply, request.model, ctx.clone())
    }

    fn fetch_deferred<'a>(
        &'a self,
        model: &'a Model,
        _handle: &'a DeferredHandle,
        ctx: &'a Context,
    ) -> BoxFuture<'a, AssistantMessage> {
        let reply = self
            .state
            .lock()
            .unwrap()
            .polls
            .pop_front()
            .unwrap_or_else(|| Reply::error("the fake model has no scripted poll"));
        let stream = play(reply, model, ctx.clone());
        Box::pin(stream.result())
    }

    fn cancel_deferred<'a>(
        &'a self,
        _model: &'a Model,
        handle: &'a DeferredHandle,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, Result<(), ModelError>> {
        self.state.lock().unwrap().cancelled.push(handle.id.clone());
        Box::pin(async { Ok(()) })
    }
}
