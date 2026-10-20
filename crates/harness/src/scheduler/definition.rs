//! Task definitions: the [`TaskDefinition`] trait a durable task implements, its typed
//! [`Checkpoint`], the [`RunningTask`] a phase sees and the typed `Tx::create_task`.

use std::fmt;
use std::future::Future;

use delta_chord::context::Context;
use delta_chord::json::{self, Json};
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::error::TaskError;
use super::next::Next;
use super::runtime::TaskRuntime;
use crate::documents::MigrateError;
use crate::records::{RecordError, TaskRecord, TaskRef};
use crate::session::{TaskCreate, TaskOptions, Tx, TxError};

/// A task's phase checkpoint: what the next phase resumes from.
///
/// Built-in tasks use serde-tagged enums, `#[serde(tag = "phase", rename_all =
/// "snake_case")]`, stored as `{"phase":"request","attempt":1,...}`; a user task may use
/// [`Json`] and read its own fields.
pub trait Checkpoint: Serialize + DeserializeOwned + Clone + Send + Sync + 'static {
    /// The phase's name, for reports and telemetry.
    fn phase(&self) -> &str;
}

/// A `Json` checkpoint names its phase with a `phase` string key; `""` without one.
impl Checkpoint for Json {
    fn phase(&self) -> &str {
        self.get("phase").and_then(Json::as_str).unwrap_or("")
    }
}

/// Durable work: a state machine whose every phase ends by committing its next state.
///
/// The scheduler finds definitions by [`name`](Self::name) in a
/// [`TaskCatalog`](super::TaskCatalog); built-in tasks and user tasks are alike. A phase
/// performs its effect off the commit line and returns the [`Next`] commit that ends it; a
/// reopened storage reruns the current phase from its checkpoint, so a phase is written as
/// an effect sandwich (commit intent, perform, commit outcome) and retries idempotently.
///
/// `run` and `abort` may be written as `async fn`; their futures must be `Send`.
pub trait TaskDefinition: Send + Sync + Sized + 'static {
    type Input: Serialize + DeserializeOwned + Send + Sync + 'static;
    type Checkpoint: Checkpoint;
    type Result: Serialize + DeserializeOwned + Send + Sync + 'static;

    /// The name tasks of this definition are stored under (`TaskRecord::kind`).
    fn name(&self) -> &str;

    /// The version that writes new records; an older stored version is migrated at
    /// reservation, a newer one blocks the task.
    fn version(&self) -> u32;

    /// The first checkpoint of a task created with `input`.
    fn initial(&self, input: &Self::Input) -> Self::Checkpoint;

    /// One phase: perform the effect for `phase` and return the commit that ends it.
    fn run<'a>(
        &'a self,
        phase: Self::Checkpoint,
        rt: &'a TaskRuntime<Self>,
        ctx: &'a Context,
    ) -> impl Future<Output = Result<Next<Self>, TaskError>> + Send + 'a;

    /// Runs once after an abort mark, when the task's ordinary owned work drained. Its
    /// [`Next`] must be terminal; an abort handler cannot wait, sleep or create owned work.
    fn abort<'a>(
        &'a self,
        phase: Self::Checkpoint,
        rt: &'a TaskRuntime<Self>,
        ctx: &'a Context,
    ) -> impl Future<Output = Result<Next<Self>, TaskError>> + Send + 'a;

    /// Whether [`migrate`](Self::migrate) takes older stored versions. Without it an older
    /// record stays blocked (`migration_failed`) until a definition that can take it is
    /// installed.
    fn can_migrate(&self) -> bool {
        false
    }

    /// The input and checkpoint of a record stored by version `from` (older than this
    /// one), converted at reservation in the commit that starts the task.
    fn migrate(
        &self,
        input: Json,
        checkpoint: Json,
        from: u32,
    ) -> Result<(Self::Input, Self::Checkpoint), MigrateError> {
        drop((input, checkpoint));
        Err(MigrateError::Unsupported { from })
    }
}

/// A running task as its phase sees it: the record with its input and checkpoint decoded.
pub struct RunningTask<T: TaskDefinition> {
    /// The record as committed when the phase (or the commit) started.
    pub record: TaskRecord,
    pub input: T::Input,
    pub checkpoint: T::Checkpoint,
}

impl<T: TaskDefinition> RunningTask<T> {
    /// Decodes a live record's input and checkpoint.
    pub(super) fn decode(record: TaskRecord) -> Result<Self, RecordError> {
        let problem = |what: &str, error: serde_json::Error| RecordError::Task {
            id: record.id,
            problem: format!("its {what} does not convert: {error}"),
        };
        let input = json::from_json(&record.input).map_err(|error| problem("input", error))?;
        let stored = record.state.checkpoint().ok_or_else(|| RecordError::Task {
            id: record.id,
            problem: format!("a {} task has no checkpoint", record.state.status().as_str()),
        })?;
        let checkpoint = json::from_json(stored).map_err(|error| problem("checkpoint", error))?;
        Ok(Self {
            record,
            input,
            checkpoint,
        })
    }
}

impl<T: TaskDefinition> fmt::Debug for RunningTask<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunningTask")
            .field("record", &self.record)
            .field("phase", &self.checkpoint.phase())
            .finish_non_exhaustive()
    }
}

impl Tx<'_> {
    /// A pending task of `definition` with `input` and its initial checkpoint. A scoped
    /// commit (a task runtime's) defaults the conversation to its task's; the owner is
    /// validated against its final candidate at settlement.
    pub async fn create_task<T: TaskDefinition>(
        &mut self,
        definition: &T,
        input: T::Input,
        options: TaskOptions,
    ) -> Result<TaskRef<T::Result>, TxError> {
        let checkpoint = definition.initial(&input);
        let encode = |error: serde_json::Error| {
            TxError::callback(format!(
                "Task {} input or checkpoint does not convert: {error}",
                definition.name()
            ))
        };
        let create = TaskCreate {
            kind: definition.name().to_owned(),
            version: definition.version(),
            input: json::to_json(&input).map_err(encode)?,
            checkpoint: json::to_json(&checkpoint).map_err(encode)?,
        };
        drop(input);
        let id = self.create_task_record(create, options).await?;
        Ok(TaskRef::new(id))
    }
}
