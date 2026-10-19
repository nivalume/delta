//! In-memory provider with scripted responses, for tests and demos.
//!
//! ```
//! # #[tokio::main(flavor = "current_thread")] async fn main() {
//! use tau_ai::providers::faux::{FauxOptions, FauxProvider, faux_assistant_message, faux_text};
//! use tau_ai::{Context, Message, Models, StreamOptions};
//!
//! let faux = FauxProvider::new(FauxOptions::default());
//! let models = Models::default();
//! models.set_provider(faux.clone());
//! faux.set_responses(vec![faux_assistant_message(vec![faux_text("Hello!")]).into()]);
//! let reply = models.complete(&faux.model(), &Context::new(vec![Message::user("hi")]), StreamOptions::default()).await;
//! assert_eq!(reply.text(), "Hello!");
//! # }
//! ```

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::auth::{Credential, ProviderAuth};
use crate::events::{AssistantMessageEventStream, MessageBuilder};
use crate::model::{AnyModel, InputModality, Model, ModelCost};
use crate::options::{SimpleStreamOptions, StreamOptions};
use crate::provider::Provider;
use crate::transcript::get_system_message_text;
use crate::types::{
    AssistantBlock, AssistantMessage, Message, StopReason, TextContent, ThinkingContent, ToolCall, TranscriptContext,
    Usage, UserBlock, UserContent, now_ms,
};

const DEFAULT_API: &str = "faux";
const DEFAULT_PROVIDER: &str = "faux";
const DEFAULT_MODEL_ID: &str = "faux-1";

/// A text block.
pub fn faux_text(text: impl Into<String>) -> AssistantBlock {
    AssistantBlock::Text(TextContent::new(text))
}

/// A thinking block.
pub fn faux_thinking(thinking: impl Into<String>) -> AssistantBlock {
    AssistantBlock::Thinking(ThinkingContent::new(thinking))
}

/// A tool call block with a generated id.
pub fn faux_tool_call(name: impl Into<String>, arguments: serde_json::Value) -> AssistantBlock {
    let arguments = match arguments {
        serde_json::Value::Object(object) => object,
        _ => serde_json::Map::new(),
    };
    AssistantBlock::ToolCall(ToolCall {
        id: random_id("tool"),
        name: name.into(),
        arguments,
        ..ToolCall::default()
    })
}

/// A scripted assistant reply with stop reason `stop` (`tool_use` when it calls tools).
pub fn faux_assistant_message(content: Vec<AssistantBlock>) -> AssistantMessage {
    let has_tool_calls = content.iter().any(|block| matches!(block, AssistantBlock::ToolCall(_)));
    AssistantMessage {
        content,
        stop_reason: if has_tool_calls {
            StopReason::ToolUse
        } else {
            StopReason::Stop
        },
        ..AssistantMessage::empty(DEFAULT_API, DEFAULT_PROVIDER, DEFAULT_MODEL_ID)
    }
}

/// Mutable call counters of a faux provider.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FauxState {
    pub call_count: usize,
}

/// Builds a reply from the request.
pub type FauxResponseFactory = Arc<dyn Fn(&TranscriptContext, &FauxState, &Model) -> AssistantMessage + Send + Sync>;

/// One queued reply.
#[derive(Clone)]
pub enum FauxResponse {
    Message(AssistantMessage),
    Factory(FauxResponseFactory),
}

impl fmt::Debug for FauxResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(message) => f.debug_tuple("Message").field(message).finish(),
            Self::Factory(_) => f.write_str("Factory"),
        }
    }
}

impl From<AssistantMessage> for FauxResponse {
    fn from(message: AssistantMessage) -> Self {
        Self::Message(message)
    }
}

/// A faux model.
#[derive(Clone, Debug)]
pub struct FauxModelDefinition {
    pub id: String,
    pub name: Option<String>,
    pub reasoning: bool,
    pub input: Vec<InputModality>,
    pub cost: ModelCost,
    pub context_window: u64,
    pub max_tokens: u64,
}

impl FauxModelDefinition {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: None,
            reasoning: false,
            input: vec![InputModality::Text, InputModality::Image],
            cost: ModelCost::default(),
            context_window: 128_000,
            max_tokens: 16_384,
        }
    }
}

/// Options of [`FauxProvider::new`].
#[derive(Clone, Debug, Default)]
pub struct FauxOptions {
    /// API id of the models. Default: `faux`.
    pub api: Option<String>,
    /// Provider id. Default: `faux`.
    pub provider: Option<String>,
    /// Default: one model `faux-1`.
    pub models: Vec<FauxModelDefinition>,
    /// Pace chunk delivery in real time; unset emits chunks as fast as possible.
    pub tokens_per_second: Option<f64>,
    /// Chunk size range in tokens (4 characters each). Default: 3 to 5.
    pub token_size: Option<(usize, usize)>,
}

struct Core {
    api: String,
    provider: String,
    models: Vec<Model>,
    tokens_per_second: Option<f64>,
    token_size: (usize, usize),
    pending: Mutex<VecDeque<FauxResponse>>,
    state: Mutex<FauxState>,
    prompt_cache: Mutex<HashMap<String, Vec<String>>>,
    auth: ProviderAuth,
}

/// Scripted provider. Clones share the response queue. Replies are consumed in request
/// start order; an empty queue yields an error reply ("No more faux responses queued").
/// Usage is estimated at 1 token per 4 characters, with prompt caching simulated per
/// `session_id` unless cache retention is `none`.
#[derive(Clone)]
pub struct FauxProvider {
    core: Arc<Core>,
}

impl fmt::Debug for FauxProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FauxProvider")
            .field("provider", &self.core.provider)
            .finish_non_exhaustive()
    }
}

impl FauxProvider {
    pub fn new(options: FauxOptions) -> Self {
        let api = options.api.unwrap_or_else(|| DEFAULT_API.into());
        let provider = options.provider.unwrap_or_else(|| DEFAULT_PROVIDER.into());
        let definitions = if options.models.is_empty() {
            vec![FauxModelDefinition {
                name: Some("Faux Model".into()),
                ..FauxModelDefinition::new(DEFAULT_MODEL_ID)
            }]
        } else {
            options.models
        };
        let models = definitions
            .into_iter()
            .map(|definition| Model {
                name: definition.name.unwrap_or_else(|| definition.id.clone()),
                id: definition.id,
                api: api.clone(),
                provider: provider.clone(),
                base_url: "http://localhost:0".into(),
                input: definition.input,
                input_limits: None,
                cost: definition.cost,
                headers: None,
                reasoning: definition.reasoning,
                thinking_level_map: None,
                prompt_cache: None,
                context_window: definition.context_window,
                max_tokens: definition.max_tokens,
                sampling_params: None,
                sampling_params_by_thinking_level: None,
                compat: None,
            })
            .collect();
        let (min, max) = options.token_size.unwrap_or((3, 5));
        let min = min.max(1);
        Self {
            core: Arc::new(Core {
                auth: ProviderAuth::keyless(format!("{provider} (faux)")),
                api,
                provider,
                models,
                tokens_per_second: options.tokens_per_second,
                token_size: (min, max.max(min)),
                pending: Mutex::default(),
                state: Mutex::default(),
                prompt_cache: Mutex::default(),
            }),
        }
    }

    /// The first model.
    pub fn model(&self) -> Model {
        self.core.models[0].clone()
    }

    /// A model by id.
    pub fn model_by_id(&self, id: &str) -> Option<Model> {
        self.core.models.iter().find(|model| model.id == id).cloned()
    }

    /// Replace the remaining queue.
    pub fn set_responses(&self, responses: Vec<FauxResponse>) {
        *self.core.pending.lock().unwrap_or_else(PoisonError::into_inner) = responses.into();
    }

    /// Append to the queue.
    pub fn append_responses(&self, responses: Vec<FauxResponse>) {
        self.core
            .pending
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend(responses);
    }

    pub fn pending_response_count(&self) -> usize {
        self.core.pending.lock().unwrap_or_else(PoisonError::into_inner).len()
    }

    pub fn state(&self) -> FauxState {
        *self.core.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn run(&self, model: &Model, context: &TranscriptContext, options: &StreamOptions) -> AssistantMessageEventStream {
        let core = self.core.clone();
        let step = core.pending.lock().unwrap_or_else(PoisonError::into_inner).pop_front();
        let state = {
            let mut state = core.state.lock().unwrap_or_else(PoisonError::into_inner);
            state.call_count += 1;
            *state
        };
        let (sender, stream) = AssistantMessageEventStream::channel();
        let model = model.clone();
        let context = context.clone();
        let session = options
            .session_id
            .clone()
            .filter(|_| options.cache_retention != Some(crate::model::CacheRetention::None));
        let cancel = options.cancel.clone();
        tokio::spawn(async move {
            let mut message = match step {
                Some(FauxResponse::Message(message)) => message,
                Some(FauxResponse::Factory(factory)) => factory(&context, &state, &model),
                None => AssistantMessage {
                    stop_reason: StopReason::Error,
                    error_message: Some("No more faux responses queued".into()),
                    ..AssistantMessage::empty(&model.api, &model.provider, &model.id)
                },
            };
            message.api.clone_from(&core.api);
            message.provider.clone_from(&core.provider);
            message.model.clone_from(&model.id);
            message.usage = core.estimate_usage(&message, &context, session.as_deref());
            core.play(
                MessageBuilder::new(sender, AssistantMessage::default()),
                message,
                cancel,
            )
            .await;
        });
        stream
    }
}

impl Core {
    fn estimate_usage(&self, message: &AssistantMessage, context: &TranscriptContext, session: Option<&str>) -> Usage {
        let prompt: Vec<String> = context
            .messages()
            .iter()
            .map(|message| format!("{}:{}", message.role(), message_text(message)))
            .collect();
        let prompt_length = joined_length(&prompt, prompt.len());
        let prompt_tokens = prompt_length.div_ceil(4) as u64;
        let output = assistant_text(&message.content).encode_utf16().count().div_ceil(4) as u64;
        let (mut input, mut cache_read, mut cache_write) = (prompt_tokens, 0, 0);
        if let Some(session) = session {
            let mut cache = self.prompt_cache.lock().unwrap_or_else(PoisonError::into_inner);
            match cache.get(session) {
                Some(previous) => {
                    let cached = common_prompt_prefix_length(previous, &prompt);
                    cache_read = cached.div_ceil(4) as u64;
                    cache_write = (prompt_length - cached).div_ceil(4) as u64;
                    input = prompt_tokens.saturating_sub(cache_read);
                }
                None => cache_write = prompt_tokens,
            }
            cache.insert(session.to_owned(), prompt);
        }
        Usage {
            input,
            output,
            cache_read,
            cache_write,
            total_tokens: input + output + cache_read + cache_write,
            ..Usage::default()
        }
    }

    fn chunks(&self, text: &str, rng: &mut u64) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let (min, max) = self.token_size;
        let mut chunks = Vec::new();
        let mut index = 0;
        while index < chars.len() {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            let size = (min + (*rng as usize) % (max - min + 1)) * 4;
            let end = (index + size).min(chars.len());
            chunks.push(chars[index..end].iter().collect());
            index = end;
        }
        if chunks.is_empty() {
            chunks.push(String::new());
        }
        chunks
    }

    async fn pace(&self, chunk: &str, cancel: Option<&CancellationToken>) -> bool {
        match self.tokens_per_second.filter(|tps| *tps > 0.0) {
            Some(tps) => {
                let delay = Duration::from_secs_f64(chunk.len().div_ceil(4) as f64 / tps);
                tokio::time::sleep(delay).await;
            }
            None => tokio::task::yield_now().await,
        }
        cancel.is_some_and(CancellationToken::is_cancelled)
    }

    async fn play(&self, mut builder: MessageBuilder, message: AssistantMessage, cancel: Option<CancellationToken>) {
        let cancel = cancel.as_ref();
        let aborted = |mut builder: MessageBuilder, message: &AssistantMessage| {
            builder.output.usage = message.usage.clone();
            builder.fail("Request was aborted", true);
        };
        builder.output = AssistantMessage {
            content: Vec::new(),
            stop_reason: StopReason::Pending,
            ..message.clone()
        };
        if message.stop_reason.is_failure() && message.content.is_empty() {
            builder.output = message.clone();
            let error = message.error_message.clone().unwrap_or_default();
            builder.fail(error, message.stop_reason == StopReason::Aborted);
            return;
        }
        if cancel.is_some_and(CancellationToken::is_cancelled) {
            return aborted(builder, &message);
        }
        builder.start();
        let mut rng = (now_ms() as u64) | 1;
        for block in &message.content {
            match block {
                AssistantBlock::Thinking(thinking) => {
                    let index = builder.start_thinking(String::new(), false, None);
                    for chunk in self.chunks(&thinking.thinking, &mut rng) {
                        if self.pace(&chunk, cancel).await {
                            return aborted(builder, &message);
                        }
                        builder.thinking_delta(index, &chunk);
                    }
                    builder.set_thinking_signature(index, thinking.thinking_signature.clone());
                    builder.end_thinking(index);
                }
                AssistantBlock::Text(text) => {
                    let index = builder.start_text();
                    for chunk in self.chunks(&text.text, &mut rng) {
                        if self.pace(&chunk, cancel).await {
                            return aborted(builder, &message);
                        }
                        builder.text_delta(index, &chunk);
                    }
                    builder.set_text_signature(index, text.text_signature.clone());
                    builder.end_text(index);
                }
                AssistantBlock::ToolCall(call) => {
                    let index = builder.start_tool_call(call.id.clone(), call.name.clone(), serde_json::Map::new());
                    let json = serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".into());
                    for chunk in self.chunks(&json, &mut rng) {
                        if self.pace(&chunk, cancel).await {
                            return aborted(builder, &message);
                        }
                        builder.tool_call_delta(index, &chunk);
                    }
                    if let Some(target) = builder.tool_call_mut(index) {
                        target.thought_signature.clone_from(&call.thought_signature);
                        target.namespace.clone_from(&call.namespace);
                    }
                    builder.end_tool_call(index);
                }
            }
        }
        builder.output = message;
        if builder.output.stop_reason == StopReason::Pending {
            builder.fail("Faux response ended without a stop reason", false);
        } else if builder.output.stop_reason.is_failure() {
            let error = builder.output.error_message.clone().unwrap_or_default();
            let aborted = builder.output.stop_reason == StopReason::Aborted;
            builder.fail(error, aborted);
        } else {
            builder.finish();
        }
    }
}

fn random_id(prefix: &str) -> String {
    format!("{prefix}:{}:{}", now_ms(), uuid::Uuid::new_v4().simple())
}

fn user_text(content: &[UserBlock]) -> String {
    content
        .iter()
        .map(|block| match block {
            UserBlock::Text(text) => text.text.clone(),
            UserBlock::Image(image) => format!("[image:{}:{}]", image.mime_type, image.data.len()),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn assistant_text(content: &[AssistantBlock]) -> String {
    content
        .iter()
        .map(|block| match block {
            AssistantBlock::Text(text) => text.text.clone(),
            AssistantBlock::Thinking(thinking) => thinking.thinking.clone(),
            AssistantBlock::ToolCall(call) => {
                format!(
                    "{}:{}",
                    call.name,
                    serde_json::to_string(&call.arguments).unwrap_or_default()
                )
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn message_text(message: &Message) -> String {
    match message {
        Message::System(system) => {
            let mut parts = vec![get_system_message_text(system)];
            for tool in system.tools_removed.iter().flatten() {
                parts.push(format!("tool-:{}", serde_json::to_string(tool).unwrap_or_default()));
            }
            for tool in system.tools_added.iter().flatten() {
                parts.push(format!("tool+:{}", serde_json::to_string(tool).unwrap_or_default()));
            }
            parts.retain(|part| !part.is_empty());
            parts.join("\n")
        }
        Message::User(user) => match &user.content {
            UserContent::Text(text) => text.clone(),
            UserContent::Blocks(blocks) => user_text(blocks),
        },
        Message::Assistant(assistant) => assistant_text(&assistant.content),
        Message::ToolResult(result) => format!("{}\n{}", result.tool_name, user_text(&result.content)),
    }
}

fn js_len(text: &str) -> usize {
    text.encode_utf16().count()
}

fn joined_length(messages: &[String], count: usize) -> usize {
    let separators = if count > 0 { (count - 1) * 2 } else { 0 };
    separators + messages[..count].iter().map(|message| js_len(message)).sum::<usize>()
}

fn common_prompt_prefix_length(previous: &[String], current: &[String]) -> usize {
    let mut index = 0;
    while index < previous.len() && index < current.len() && previous[index] == current[index] {
        index += 1;
    }
    let rest = |messages: &[String]| {
        if index == messages.len() {
            String::new()
        } else {
            format!(
                "{}{}",
                if index > 0 { "\n\n" } else { "" },
                messages[index..].join("\n\n")
            )
        }
    };
    let (a, b) = (rest(previous), rest(current));
    let common = a
        .encode_utf16()
        .zip(b.encode_utf16())
        .take_while(|(x, y)| x == y)
        .count();
    joined_length(previous, index) + common
}

impl Provider for FauxProvider {
    fn id(&self) -> &str {
        &self.core.provider
    }

    fn name(&self) -> &str {
        &self.core.provider
    }

    fn auth(&self) -> &ProviderAuth {
        &self.core.auth
    }

    fn models(&self) -> Vec<Model> {
        self.core.models.clone()
    }

    fn all_models(&self) -> Vec<AnyModel> {
        self.core.models.iter().cloned().map(AnyModel::Chat).collect()
    }

    fn filter_models(&self, models: Vec<Model>, _credential: Option<&Credential>) -> Vec<Model> {
        models
    }

    fn stream(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        self.run(model, context, &options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.run(model, context, &options.base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::AssistantMessageEvent;
    use crate::models::Models;
    use crate::types::Context;
    use futures::StreamExt;
    use serde_json::json;

    #[tokio::test]
    async fn streams_scripted_replies_through_models() {
        let faux = FauxProvider::new(FauxOptions::default());
        let models = Models::default();
        models.set_provider(faux.clone());
        let model = faux.model();
        faux.set_responses(vec![
            faux_assistant_message(vec![
                faux_thinking("Need to look."),
                faux_tool_call("echo", json!({"text": "hi"})),
            ])
            .into(),
            faux_assistant_message(vec![faux_text("done")]).into(),
        ]);
        let mut context = Context::new(vec![Message::user("Summarize")]);
        let mut stream = models.stream(&model, &context, StreamOptions::default());
        let mut kinds = Vec::new();
        while let Some(event) = stream.next().await {
            kinds.push(event.kind());
            if let AssistantMessageEvent::Done { message, .. } = &event {
                assert_eq!(message.stop_reason, StopReason::ToolUse);
                let call = message.tool_calls().next().unwrap();
                assert_eq!(serde_json::Value::Object(call.arguments.clone()), json!({"text": "hi"}));
                assert!(message.usage.input > 0);
                context.messages.push(Message::Assistant(message.clone()));
            }
        }
        assert_eq!(kinds.first(), Some(&"start"));
        assert!(kinds.contains(&"toolcall_delta"));
        assert_eq!(kinds.last(), Some(&"done"));
        let second = models.complete(&model, &context, StreamOptions::default()).await;
        assert_eq!(second.text(), "done");
        let third = models.complete(&model, &context, StreamOptions::default()).await;
        assert_eq!(third.stop_reason, StopReason::Error);
        assert_eq!(third.error_message.as_deref(), Some("No more faux responses queued"));
        assert_eq!(faux.state().call_count, 3);
    }

    #[tokio::test]
    async fn simulates_prompt_caching_per_session() {
        let faux = FauxProvider::new(FauxOptions::default());
        let models = Models::default();
        models.set_provider(faux.clone());
        faux.set_responses(vec![
            faux_assistant_message(vec![faux_text("a")]).into(),
            faux_assistant_message(vec![faux_text("b")]).into(),
        ]);
        let options = StreamOptions {
            session_id: Some("s".into()),
            ..StreamOptions::default()
        };
        let context = Context::new(vec![Message::user("a long enough prompt to cache")]);
        let first = models.complete(&faux.model(), &context, options.clone()).await;
        assert_eq!(first.usage.cache_read, 0);
        assert!(first.usage.cache_write > 0);
        let second = models.complete(&faux.model(), &context, options).await;
        assert!(second.usage.cache_read > 0);
        assert_eq!(second.usage.input, 0);
    }

    #[tokio::test]
    async fn aborts_mid_stream() {
        let faux = FauxProvider::new(FauxOptions {
            tokens_per_second: Some(20.0),
            ..FauxOptions::default()
        });
        let models = Models::default();
        models.set_provider(faux.clone());
        faux.set_responses(vec![faux_assistant_message(vec![faux_text("x".repeat(400))]).into()]);
        let cancel = CancellationToken::new();
        let options = StreamOptions {
            cancel: Some(cancel.clone()),
            ..StreamOptions::default()
        };
        let mut stream = models.stream(&faux.model(), &Context::new(vec![Message::user("go")]), options);
        while let Some(event) = stream.next().await {
            if event.kind() == "text_delta" {
                cancel.cancel();
            }
            if let AssistantMessageEvent::Error { error, .. } = &event {
                assert_eq!(error.stop_reason, StopReason::Aborted);
                return;
            }
        }
        panic!("expected an aborted error event");
    }
}
