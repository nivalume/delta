//! [`ToolApi`]: what a tool's `execute` reaches of its call and its task.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64};

use delta_ai::types::JsonObject;
use delta_chord::context::Context;
use delta_chord::json::Json;
use delta_chord::output::{OutputSkip, PROGRESS_BYTES_PER_SECOND, Retain};
use futures::future::BoxFuture;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::definition::ToolReplay;
use super::nested;
use super::output::Reporter;
use super::result::{NestedToolExecutionResult, ToolDiagnostic};
use super::task::ToolTask;
use crate::agent::{Agent, AgentError, AgentRuntime};
use crate::documents::{DocAddress, DocDefinition};
use crate::env::{ExecutionEnv, ShellOutputWindow};
use crate::model::ModelClient;
use crate::records::{ConversationId, TaskId, TaskRecord, TaskRef};
use crate::registry::RegistrySnapshot;
use crate::scheduler::{RuntimeError, SettledTask, TaskDefinition, TaskRuntime};
use crate::session::{TaskOptions, Tx, TxError};

/// Why a [`ToolApi`] operation failed.
#[derive(Debug, thiserror::Error)]
pub enum ToolApiError {
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Agent(#[from] AgentError),
    /// The progress commit carrying `details` failed.
    #[error("Progress commit failed: {0}")]
    Progress(Arc<RuntimeError>),
    /// An explicit nested call key that may not be used.
    #[error("Invalid nested call key {key:?}: {reason}")]
    InvalidKey { key: String, reason: &'static str },
    /// A nested call key reused with another tool or other arguments.
    #[error("Nested call {key} was made with another tool or other arguments")]
    KeyMismatch { key: String },
    #[error("Tool {name} is not callable from tools")]
    NotCallable { name: String },
}

/// The retained output window.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RetainedOutput {
    pub text: String,
    /// Whether the window dropped output.
    pub truncated: bool,
}

/// Options of a nested call.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NestedCallOptions {
    /// The call's key among this call's nested calls: non-empty, without `/`, not
    /// `__proto__`, not a positive integer. Default: its position (`1`, `2`, ...).
    pub key: Option<String>,
    /// `Some(false)`: the nested call publishes no progress.
    pub progress: Option<bool>,
}

/// The nested-call bookkeeping of one invocation.
#[derive(Debug, Default)]
pub(super) struct NestedCalls {
    /// Calls made so far; the default key of the next is one more.
    pub(super) count: AtomicU64,
    /// Whether this invocation recorded the `delta.tool.made_nested_calls` memo.
    pub(super) recorded: AtomicBool,
}

/// What a tool's `execute` reaches: its call, the agent and models, the environment, the
/// output window and progress, commits and memos in its task, child tasks and nested calls.
///
/// It borrows the invocation, so it cannot outlive `execute`: nothing reports into a call
/// after its result was settled.
pub struct ToolApi<'r> {
    pub(super) rt: &'r TaskRuntime<ToolTask>,
    pub(super) call_id: &'r str,
    pub(super) agent: Arc<Agent>,
    pub(super) env: Option<Arc<dyn ExecutionEnv>>,
    pub(super) reporter: &'r Reporter,
    pub(super) replay: ToolReplay,
    pub(super) nested: &'r NestedCalls,
    pub(super) output_interval: std::time::Duration,
}

impl ToolApi<'_> {
    pub fn task_id(&self) -> TaskId {
        self.rt.task_id()
    }

    pub fn conversation_id(&self) -> ConversationId {
        self.rt.conversation_id()
    }

    /// The id of the call: the model's, or `<caller's call id>/<key>` for a nested call.
    pub fn call_id(&self) -> &str {
        self.call_id
    }

    /// The registry snapshot of the phase.
    pub fn registry(&self) -> Result<RegistrySnapshot, AgentError> {
        self.rt.registry()
    }

    /// The conversation's agent as the call's phase resolved it.
    pub fn agent(&self) -> &Arc<Agent> {
        &self.agent
    }

    /// The Harness's models.
    pub fn models(&self) -> Result<Arc<dyn ModelClient>, AgentError> {
        Ok(self.rt.agent_services()?.models().clone())
    }

    /// The environment built for this call, in the agent's `cwd`; `None` without an
    /// environment builder.
    pub fn env(&self) -> Option<&Arc<dyn ExecutionEnv>> {
        self.env.as_ref()
    }

    /// The Harness clock, Unix epoch milliseconds.
    pub fn now(&self) -> i64 {
        self.rt.now()
    }

    /// Appends raw output, decoded as a UTF-8 stream.
    pub fn output(&self, chunk: &[u8]) {
        self.reporter.push_bytes(chunk);
    }

    /// Appends text output.
    pub fn output_str(&self, chunk: &str) {
        self.reporter.push_str(chunk);
    }

    /// Records output an environment omitted before the next chunk (see
    /// [`ShellOutputWindow`]); only a tail window can; a head window reports the misuse and
    /// ignores it.
    pub fn output_skipped(&self, skip: OutputSkip) {
        if !self.reporter.skip(skip) {
            let error = std::io::Error::other("output_skipped needs a tail-retaining tool");
            self.rt.report(&error, None);
        }
    }

    /// The window a tail-retaining tool passes to `exec`, so an environment may omit
    /// output outside it.
    pub fn output_window(&self) -> Option<ShellOutputWindow> {
        let limits = self.reporter.limits();
        (limits.retain == Retain::Tail).then_some(ShellOutputWindow {
            max_bytes: limits.max_bytes,
            max_lines: limits.max_lines,
            min_interval: self.output_interval,
            bytes_per_second: PROGRESS_BYTES_PER_SECOND,
        })
    }

    /// The retained output so far.
    pub fn retained_output(&self) -> RetainedOutput {
        let output = self.reporter.retained();
        RetainedOutput {
            truncated: output.dropped_bytes > 0 || output.dropped_lines > 0,
            text: output.text,
        }
    }

    /// Adds a model-visible diagnostic to the call's result.
    pub fn diagnostic(&self, diagnostic: ToolDiagnostic) {
        self.reporter.diagnostic(diagnostic);
    }

    /// Replaces the running details; resolves after the commit that carries them.
    /// Cancelling the wait leaves the value in place.
    pub async fn details(&self, value: Value, ctx: &Context) -> Result<(), ToolApiError> {
        let committed = ctx
            .run(self.reporter.details(value))
            .await
            .map_err(RuntimeError::Cancelled)?;
        committed.map_err(ToolApiError::Progress)
    }

    /// A commit in the call's task; refused once the task ends or is aborted.
    pub async fn commit<R, F>(&self, ctx: &Context, f: F) -> Result<R, ToolApiError>
    where
        F: for<'a, 'l> FnOnce(&'a mut Tx<'l>) -> BoxFuture<'a, Result<R, TxError>> + Send + 'static,
        R: Send + 'static,
    {
        let value = self
            .rt
            .commit(ctx, move |tx, _| {
                Box::pin(async move {
                    let value = f(tx).await?;
                    Ok((value, None))
                })
            })
            .await?;
        Ok(value)
    }

    /// The committed memo `name` of the call's task.
    pub fn memo<V: DeserializeOwned>(&self, name: &str) -> Result<Option<V>, ToolApiError> {
        Ok(self.rt.memo(name)?)
    }

    /// The memo `name`, inserting `candidate` when there is none; returns the winner.
    pub async fn memo_or_insert<V: Serialize + DeserializeOwned>(
        &self,
        name: &str,
        candidate: V,
        ctx: &Context,
    ) -> Result<V, ToolApiError> {
        Ok(self.rt.memo_or_insert(name, candidate, ctx).await?)
    }

    /// Creates a task of `definition`. A child of this task (`options.owner` is this task)
    /// of a call that is not replay-safe is abandoned after a restart, as the call is.
    pub async fn create_task<T>(
        &self,
        definition: &T,
        input: T::Input,
        mut options: TaskOptions,
        ctx: &Context,
    ) -> Result<TaskRef<T::Result>, ToolApiError>
    where
        T: TaskDefinition + Clone,
    {
        if options.owner == Some(self.task_id()) && self.replay != ToolReplay::Safe {
            options.abandon_on_restart = true;
        }
        let definition = definition.clone();
        self.commit(ctx, move |tx| {
            Box::pin(async move { tx.create_task(&definition, input, options).await })
        })
        .await
    }

    /// The committed record of task `id`.
    pub async fn get_task(&self, id: TaskId, ctx: &Context) -> Result<Option<TaskRecord>, ToolApiError> {
        Ok(self.rt.get_task(id, ctx).await?)
    }

    /// The terminal record of task `id` once it is terminal.
    pub async fn wait_for_task(&self, id: impl Into<TaskId>, ctx: &Context) -> Result<SettledTask, ToolApiError> {
        Ok(self.rt.wait_for_task(id, ctx).await?)
    }

    /// The current value of a committed document.
    pub async fn snapshot<D: DocDefinition>(
        &self,
        at: DocAddress<D>,
        ctx: &Context,
    ) -> Result<Option<Json>, ToolApiError> {
        Ok(self.rt.snapshot(at, ctx).await?)
    }

    /// Calls the tool `name` as a nested call: a `delta.tool` task owned by this one, whose
    /// result comes back here instead of into the transcript. Calls with the same key are
    /// the same call: a rerun reattaches to it, with the same tool and arguments only.
    /// Cancelling `ctx` stops only the wait.
    pub async fn execute_tool(
        &self,
        name: &str,
        arguments: JsonObject,
        options: NestedCallOptions,
        ctx: &Context,
    ) -> Result<NestedToolExecutionResult, ToolApiError> {
        nested::execute_tool(self, name, arguments, options, ctx).await
    }
}

impl fmt::Debug for ToolApi<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolApi")
            .field("task_id", &self.task_id())
            .field("call_id", &self.call_id)
            .finish_non_exhaustive()
    }
}
