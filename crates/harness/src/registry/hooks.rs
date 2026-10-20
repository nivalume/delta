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

use crate::model::ModelClient;
use crate::records::{ConversationId, EntryId, TaskId};
use crate::scheduler::{RuntimeError, TaskDefinition, TaskRuntime};

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

    /// Continues a run that would end with `answer`: the continuation is appended as a user
    /// message and a successor generation runs, unless the final boundary placed user input
    /// or a reset.
    fn on_yield<'a>(
        &'a self,
        answer: &'a AssistantMessage,
        api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<UserContent>> {
        let _ = (answer, api, ctx);
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
