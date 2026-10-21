//! Hooks: one trait per built-in task's hook set, every method with a default that does
//! nothing (spec §7.2). A task declares its hook trait with [`TaskHooks`]; a phase resolves
//! the selected extensions' hooks for the task's name into a [`HookRunner`] and calls them in
//! extension order with a [`HookApi`].

use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;

use delta_ai::types::{AssistantMessage, Message, UserContent};
use delta_chord::context::Context;
use delta_chord::json::{self, Json};
use futures::future::BoxFuture;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::agent::{Agent, AgentError, AgentRuntime};
use crate::documents::{DocAddress, DocDefinition};
use crate::env::ExecutionEnv;
use crate::model::ModelClient;
use crate::records::{ConversationId, ConversationRecord, EntryId, EntryRecord, TaskId};
use crate::scheduler::{RuntimeError, TaskDefinition, TaskRuntime};
use crate::session::Session;

/// An ordinary hook failure: reported, and the next handler runs.
pub type HookError = Box<dyn StdError + Send + Sync>;

/// What a hook returns: `None` changes nothing.
pub type HookResult<T> = Result<Option<T>, HookError>;

/// Beside [`TaskDefinition`]: the hook trait a task dispatches to. The kernel's definitions
/// have no hook type; a task without hooks does not implement this.
pub trait TaskHooks: TaskDefinition {
    type Hooks: ?Sized + Send + Sync + 'static;
}

/// The phase's resolved hooks for one task name, in extension order, with the name of the
/// extension each came from (for reports).
pub struct HookRunner<H: ?Sized> {
    handlers: Vec<(String, Arc<H>)>,
}

impl<H: ?Sized> HookRunner<H> {
    pub fn new(handlers: Vec<(String, Arc<H>)>) -> Self {
        Self { handlers }
    }

    /// The handlers with their extension's name, in extension order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &H)> {
        self.handlers
            .iter()
            .map(|(extension, handler)| (extension.as_str(), &**handler))
    }

    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    pub fn len(&self) -> usize {
        self.handlers.len()
    }
}

impl<H: ?Sized> fmt::Debug for HookRunner<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let extensions: Vec<&str> = self.handlers.iter().map(|(extension, _)| extension.as_str()).collect();
        f.debug_struct("HookRunner").field("extensions", &extensions).finish()
    }
}

/// What a hook reaches of its task, type-erased so hook traits stay object-safe:
/// first-writer-wins memos in the task's namespace (prefix them with the extension's name).
pub trait HookHost: Send + Sync {
    /// The committed memo `name`, from memory.
    fn memo_value(&self, name: &str) -> Result<Option<Json>, RuntimeError>;

    /// The memo `name`, inserting `candidate` when there is none; returns the winner.
    fn memo_or_insert_value<'a>(
        &'a self,
        name: &'a str,
        candidate: Json,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<Json, RuntimeError>>;

    /// The Session, for committed document reads; refused once the invocation ended.
    fn session(&self) -> Result<&Session, RuntimeError>;

    /// The conversation's agent as the phase resolved it.
    fn agent<'a>(&'a self, ctx: &'a Context) -> BoxFuture<'a, Result<Arc<Agent>, AgentError>>;

    /// The conversation's environment, built for this use; `None` without a builder.
    fn env<'a>(&'a self, ctx: &'a Context) -> BoxFuture<'a, Result<Option<Arc<dyn ExecutionEnv>>, AgentError>>;
}

impl<T: TaskDefinition> HookHost for TaskRuntime<T> {
    fn memo_value(&self, name: &str) -> Result<Option<Json>, RuntimeError> {
        self.memo(name)
    }

    fn memo_or_insert_value<'a>(
        &'a self,
        name: &'a str,
        candidate: Json,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<Json, RuntimeError>> {
        Box::pin(self.memo_or_insert(name, candidate, ctx))
    }

    fn session(&self) -> Result<&Session, RuntimeError> {
        self.checked_session()
    }

    fn agent<'a>(&'a self, ctx: &'a Context) -> BoxFuture<'a, Result<Arc<Agent>, AgentError>> {
        Box::pin(AgentRuntime::agent(self, ctx))
    }

    fn env<'a>(&'a self, ctx: &'a Context) -> BoxFuture<'a, Result<Option<Arc<dyn ExecutionEnv>>, AgentError>> {
        Box::pin(AgentRuntime::env(self, ctx))
    }
}

/// The task a hook runs for: its ids, the models and its memos. Hooks run off the session
/// line; a crash before the consuming commit may run them again, so first-writer-wins
/// decisions belong in memos.
pub struct HookApi<'a> {
    pub task_id: TaskId,
    pub conversation_id: ConversationId,
    models: Arc<dyn ModelClient>,
    host: &'a dyn HookHost,
}

impl<'a> HookApi<'a> {
    pub fn new(
        task_id: TaskId,
        conversation_id: ConversationId,
        models: Arc<dyn ModelClient>,
        host: &'a dyn HookHost,
    ) -> Self {
        Self {
            task_id,
            conversation_id,
            models,
            host,
        }
    }

    /// The Harness's models, read when the phase resolved them.
    pub fn models(&self) -> &Arc<dyn ModelClient> {
        &self.models
    }

    /// The committed memo `name`.
    pub fn memo<V: DeserializeOwned>(&self, name: &str) -> Result<Option<V>, RuntimeError> {
        self.host
            .memo_value(name)?
            .map(|value| json::from_json(&value))
            .transpose()
            .map_err(RuntimeError::Codec)
    }

    /// The conversation's agent as the phase resolved it.
    pub async fn agent(&self, ctx: &Context) -> Result<Arc<Agent>, AgentError> {
        self.host.agent(ctx).await
    }

    /// The conversation's environment, built for this use; `None` without an environment
    /// builder. Commands a hook runs go through it.
    pub async fn env(&self, ctx: &Context) -> Result<Option<Arc<dyn ExecutionEnv>>, AgentError> {
        self.host.env(ctx).await
    }

    /// The conversation's working directory: its environment's, else the agent's.
    pub async fn cwd(&self, ctx: &Context) -> Result<Option<String>, AgentError> {
        if let Some(env) = self.env(ctx).await? {
            return Ok(Some(env.cwd().to_owned()));
        }
        Ok(self.agent(ctx).await?.cwd.clone())
    }

    /// The committed record of the hook's conversation (its owner, its parent).
    pub async fn conversation(&self, ctx: &Context) -> Result<Option<ConversationRecord>, RuntimeError> {
        let id = self.conversation_id;
        let session = self.host.session()?;
        Ok(session
            .commit(ctx, move |tx| Box::pin(async move { tx.conversation(id).await }))
            .await?)
    }

    /// The current value of a committed document.
    pub async fn snapshot<D: DocDefinition>(
        &self,
        at: DocAddress<D>,
        ctx: &Context,
    ) -> Result<Option<Json>, RuntimeError> {
        Ok(self.host.session()?.snapshot(at, ctx).await?)
    }

    /// The memo `name`, inserting `candidate` when there is none; returns the winner.
    pub async fn memo_or_insert<V: Serialize + DeserializeOwned>(
        &self,
        name: &str,
        candidate: V,
        ctx: &Context,
    ) -> Result<V, RuntimeError> {
        let candidate = json::to_json(&candidate).map_err(RuntimeError::Codec)?;
        let winner = self.host.memo_or_insert_value(name, candidate, ctx).await?;
        json::from_json(&winner).map_err(RuntimeError::Codec)
    }
}

impl fmt::Debug for HookApi<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HookApi")
            .field("task_id", &self.task_id)
            .field("conversation_id", &self.conversation_id)
            .finish_non_exhaustive()
    }
}

/// A final answer, as `on_yield` sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct Yielded {
    pub message: AssistantMessage,
    /// `on_yield` continuations the run took so far (Claude Code's `stop_hook_active`).
    pub continuations: u32,
}

/// The user input a run starts with, as `on_input` sees it: the entries its inputs placed.
#[derive(Clone, Debug, PartialEq)]
pub struct RunInput {
    pub entries: Vec<EntryRecord>,
}

/// What an `on_input` hook decides about a run's input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputDecision {
    /// The run ends `unanswered` with reason `blocked` and this detail before any request;
    /// the input stays in the transcript.
    Block { reason: String },
    /// Context for the model, appended as a `delta.context` entry before the first request.
    Context(String),
}

/// The messages of one request, as `before_request` sees them.
#[derive(Clone, Debug, PartialEq)]
pub struct RequestMessages {
    /// The attempt, from 1; a retry or a resend after recovery runs the chain again.
    pub attempt: u32,
    pub messages: Vec<Message>,
}

/// Hooks of `delta.generation`. Every method defaults to doing nothing.
///
/// | Hook | Composition | Ordinary error |
/// | --- | --- | --- |
/// | `before_request` | replacement chain on the messages, this request only | reported, chain continues |
/// | `after_response` | every observer, each terminal message | reported |
/// | `on_input` | the first block wins; every context, once per run | reported |
/// | `on_yield` | the first continuation wins | reported |
/// | `after_tools` | every observer, once per tool round | reported |
pub trait GenerationHooks: Send + Sync {
    /// Replaces the messages of this request; later handlers see the replacement.
    fn before_request<'a>(
        &'a self,
        request: &'a RequestMessages,
        api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<Vec<Message>>> {
        let _ = (request, api, ctx);
        Box::pin(async { Ok(None) })
    }

    /// Observes every terminal response before it is classified.
    fn after_response<'a>(
        &'a self,
        message: &'a AssistantMessage,
        api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<()>> {
        let _ = (message, api, ctx);
        Box::pin(async { Ok(None) })
    }

    /// Sees the input a run starts with, once per run before its first request; may block
    /// the run or add context. The first block wins; every context is appended.
    fn on_input<'a>(
        &'a self,
        input: &'a RunInput,
        api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<InputDecision>> {
        let _ = (input, api, ctx);
        Box::pin(async { Ok(None) })
    }

    /// Continues a run that would end with `yielded.message`: the continuation is appended
    /// as a user message (its entry's data `{ "continuation": true }`) and a successor
    /// generation runs, unless the final boundary placed user input or a reset, or the run
    /// took `max_continuations` already.
    fn on_yield<'a>(
        &'a self,
        yielded: &'a Yielded,
        api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<UserContent>> {
        let _ = (yielded, api, ctx);
        Box::pin(async { Ok(None) })
    }

    /// Observes a finished tool round: the tool-calling answer and the round's result
    /// entries in call order.
    fn after_tools<'a>(
        &'a self,
        assistant: EntryId,
        results: &'a [EntryId],
        api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<()>> {
        let _ = (assistant, results, api, ctx);
        Box::pin(async { Ok(None) })
    }
}
