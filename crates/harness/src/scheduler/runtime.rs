//! [`TaskRuntime`]: what a phase handler reaches of the scheduler and the Session, bound to
//! its invocation. Every operation rejects once the invocation ended.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::error::Error as StdError;
use std::fmt;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context as TaskContext, Poll};
use std::time::Duration;

use delta_chord::context::Context;
use delta_chord::json::{self, Json};
use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::Inner;
use super::catalog::CatalogSnapshot;
use super::definition::{RunningTask, TaskDefinition};
use super::error::{Gate, RuntimeError};
use super::invocation::{Invocation, Mode};
use super::next::{Next, NextOf};
use super::transition::commit_state;
use super::waits::{self, SettledTask};
use crate::documents::{ConversationDoc, DocAddress, DocDefinition, Rewindable};
use crate::records::{
    ConversationId, EntryId, EntryRecord, SubmissionId, SubmissionRecord, TaskId, TaskOutcome, TaskRecord, TaskState,
    TaskStatus,
};
use crate::session::{HostCallback, ReportInfo, ReportTask, Services, Session, TransactionScope, Tx, TxError};

/// What every runtime of one phase shares: the scheduler, the invocation, the phase's
/// catalog snapshot and its per-phase data.
pub(crate) struct RuntimeCore {
    pub(super) inner: Arc<Inner>,
    pub(super) invocation: Arc<Invocation>,
    pub(super) snapshot: Arc<dyn CatalogSnapshot>,
    pub(super) phase_data: PhaseData,
}

impl RuntimeCore {
    pub(super) fn new(inner: Arc<Inner>, invocation: Arc<Invocation>, snapshot: Arc<dyn CatalogSnapshot>) -> Self {
        Self {
            inner,
            invocation,
            snapshot,
            phase_data: PhaseData::default(),
        }
    }

    fn check(&self) -> Result<(), RuntimeError> {
        if self.invocation.is_ended() {
            Err(RuntimeError::Ended(self.invocation.task))
        } else {
            Ok(())
        }
    }

    /// A commit gated on the line: the invocation is live, the scheduler open, the task
    /// running and, for a run invocation, unmarked. `f` gets the committed record.
    async fn gated<R, F>(&self, ctx: &Context, f: F) -> Result<R, RuntimeError>
    where
        F: for<'a, 'l> FnOnce(&'a mut Tx<'l>, TaskRecord) -> BoxFuture<'a, Result<R, TxError>> + Send + 'static,
        R: Send + 'static,
    {
        self.check()?;
        let inner = self.inner.clone();
        let invocation = self.invocation.clone();
        let scope = TransactionScope {
            conversation_id: Some(invocation.conversation),
            task_id: Some(invocation.task),
        };
        let result = self
            .inner
            .session
            .commit_scoped(ctx, scope, move |tx| {
                Box::pin(async move {
                    let current = gate(&inner, &invocation)?;
                    f(tx, current).await
                })
            })
            .await;
        result.map_err(RuntimeError::from)
    }

    /// A read on the line (a commit that writes nothing), refused once the invocation ended.
    async fn read<R, F>(&self, ctx: &Context, f: F) -> Result<R, RuntimeError>
    where
        F: for<'a, 'l> FnOnce(&'a mut Tx<'l>) -> BoxFuture<'a, Result<R, TxError>> + Send + 'static,
        R: Send + 'static,
    {
        self.check()?;
        let invocation = self.invocation.clone();
        let result = self
            .inner
            .session
            .commit(ctx, move |tx| {
                Box::pin(async move {
                    if invocation.is_ended() {
                        return Err(Gate::reject(RuntimeError::Ended(invocation.task)));
                    }
                    f(tx).await
                })
            })
            .await;
        result.map_err(RuntimeError::from)
    }

    /// `ctx` bound to the invocation: cancelled with it as well.
    fn bind(&self, ctx: &Context) -> Context {
        ctx.with_token(self.invocation.ctx.token())
    }

    /// An error of an operation the invocation's end cut short is `Ended`.
    fn ended_or(&self, error: RuntimeError) -> RuntimeError {
        match error {
            RuntimeError::Cancelled(_) if self.invocation.is_ended() => RuntimeError::Ended(self.invocation.task),
            other => other,
        }
    }
}

/// The gate of a runtime commit, decided on the line against the committed mirror.
fn gate(inner: &Inner, invocation: &Invocation) -> Result<TaskRecord, TxError> {
    let state = inner.shared.lock();
    let task = invocation.task;
    let refuse = |error| Err(Gate::reject(error));
    if invocation.is_ended() {
        return refuse(RuntimeError::Ended(task));
    }
    if state.closing {
        return refuse(RuntimeError::Closing);
    }
    let Some(found) = state.tree.live.get(&task) else {
        return refuse(RuntimeError::Terminal(task));
    };
    let status = found.state.status();
    if status != TaskStatus::Running {
        return refuse(RuntimeError::NotRunning { task, status });
    }
    if invocation.mode == Mode::Run && found.abort_requested {
        return refuse(RuntimeError::Aborted(task));
    }
    Ok(found.clone())
}

/// Where a reported error came from, when not from the task's own code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReportFrom {
    /// An extension's code (a hook), by name when known.
    Extension(Option<String>),
    /// A host callback.
    Host(HostCallback),
}

/// What one phase or abort handler reaches: commits gated on its invocation, memos, owned
/// work, committed reads, the clock and reports.
///
/// The layers above add what their tasks need (agent resolution, settings, models, the
/// environment) as extension traits built from [`services`](Self::services),
/// [`catalog`](Self::catalog) and [`phase_data`](Self::phase_data); the kernel defines none
/// of them.
pub struct TaskRuntime<T: TaskDefinition> {
    core: Arc<RuntimeCore>,
    task: RunningTask<T>,
}

impl<T: TaskDefinition> TaskRuntime<T> {
    pub(super) fn new(core: Arc<RuntimeCore>, task: RunningTask<T>) -> Self {
        Self { core, task }
    }

    pub fn task_id(&self) -> TaskId {
        self.core.invocation.task
    }

    pub fn conversation_id(&self) -> ConversationId {
        self.core.invocation.conversation
    }

    /// The task as committed when this phase started.
    pub fn task(&self) -> &RunningTask<T> {
        &self.task
    }

    pub fn input(&self) -> &T::Input {
        &self.task.input
    }

    /// The invocation's context: cancelled by an abort mark (run mode), close and the
    /// invocation's end.
    pub fn context(&self) -> &Context {
        &self.core.invocation.ctx
    }

    /// The catalog snapshot of this phase; refreshed at each phase boundary.
    pub fn catalog(&self) -> &Arc<dyn CatalogSnapshot> {
        &self.core.snapshot
    }

    /// The Session's services.
    pub fn services(&self) -> &Services {
        self.core.inner.session.services()
    }

    /// Values that live for this phase only (the agent layer keeps its per-phase agent
    /// resolution here).
    pub fn phase_data(&self) -> &PhaseData {
        &self.core.phase_data
    }

    /// Whether the invocation ended; every operation rejects from then on.
    pub fn is_ended(&self) -> bool {
        self.core.invocation.is_ended()
    }

    /// An intermediate commit of this phase (intent, progress): `f` sees the task as
    /// committed and returns a value and, optionally, the next state; `None` keeps the
    /// state. A `Waiting`, `Sleeping` or terminal state ends the invocation at its next step.
    pub async fn commit<R, F>(&self, ctx: &Context, f: F) -> Result<R, RuntimeError>
    where
        F: for<'a, 'l> FnOnce(
                &'a mut Tx<'l>,
                &'a RunningTask<T>,
            ) -> BoxFuture<'a, Result<(R, Option<NextOf<T>>), TxError>>
            + Send
            + 'static,
        R: Send + 'static,
    {
        let inner = self.core.inner.clone();
        let mode = self.core.invocation.mode;
        self.core
            .gated(ctx, move |tx, current| {
                Box::pin(async move {
                    let running = RunningTask::<T>::decode(current.clone())?;
                    let (value, next) = f(&mut *tx, &running).await?;
                    drop(running);
                    if let Some(next) = next {
                        let state = next.encode().map_err(TxError::callback)?;
                        commit_state(&inner, tx, mode, &current, state).await?;
                    }
                    Ok(value)
                })
            })
            .await
    }

    /// The closing commit of the phase, returned from `run` or `abort`.
    pub fn next<F>(&self, f: F) -> Next<T>
    where
        F: for<'a, 'l> FnOnce(&'a mut Tx<'l>, &'a RunningTask<T>) -> BoxFuture<'a, Result<NextOf<T>, TxError>>
            + Send
            + 'static,
    {
        Next::commit(f)
    }

    /// The committed memo `name`, from memory.
    pub fn memo<V: DeserializeOwned>(&self, name: &str) -> Result<Option<V>, RuntimeError> {
        self.core.check()?;
        let value = {
            let state = self.core.inner.shared.lock();
            let record = state.tree.live.get(&self.task_id());
            record.and_then(|record| record.memos.as_ref()?.get(name).cloned())
        };
        value
            .map(|value| json::from_json(&value))
            .transpose()
            .map_err(RuntimeError::Codec)
    }

    /// The memo `name`, inserting `candidate` when there is none: one gated commit that
    /// returns the stored winner (first writer wins).
    pub async fn memo_or_insert<V>(&self, name: &str, candidate: V, ctx: &Context) -> Result<V, RuntimeError>
    where
        V: Serialize + DeserializeOwned,
    {
        let value = json::to_json(&candidate).map_err(RuntimeError::Codec)?;
        let name = name.to_owned();
        let winner = self
            .core
            .gated(ctx, move |tx, mut current| {
                Box::pin(async move {
                    if let Some(winner) = current.memos.as_ref().and_then(|memos| memos.get(&name)) {
                        return Ok(winner.clone());
                    }
                    current
                        .memos
                        .get_or_insert_with(IndexMap::new)
                        .insert(name, value.clone());
                    tx.set_task(current)?;
                    Ok(value)
                })
            })
            .await?;
        json::from_json(&winner).map_err(RuntimeError::Codec)
    }

    /// The committed record of task `id`.
    pub async fn get_task(&self, id: TaskId, ctx: &Context) -> Result<Option<TaskRecord>, RuntimeError> {
        self.core
            .read(ctx, move |tx| Box::pin(async move { tx.task(id).await }))
            .await
    }

    /// The terminal record of task `id` once it is terminal; rejects when the invocation
    /// ends first (a wait held in memory, not a durable wait).
    pub async fn wait_for_task(&self, id: impl Into<TaskId>, ctx: &Context) -> Result<SettledTask, RuntimeError> {
        self.core.check()?;
        let bound = self.core.bind(ctx);
        waits::wait_for_task(&self.core.inner, id.into(), &bound)
            .await
            .map_err(|error| self.core.ended_or(error.into()))
    }

    /// Aborts a task this task owns and resolves once it is terminal; keeps a `restart`
    /// mark (an owner's own cleanup).
    pub async fn abort_owned(&self, id: TaskId, ctx: &Context) -> Result<(), RuntimeError> {
        let bound = self.core.bind(ctx);
        let owner = self.task_id();
        let record = self.get_task(id, &bound).await?;
        let Some(record) = record.filter(|record| record.owner == Some(owner)) else {
            return Err(RuntimeError::NotOwned { task: id, owner });
        };
        if record.state.status() == TaskStatus::Terminal {
            return Ok(());
        }
        let inner = &self.core.inner;
        super::abort::abort(inner, id, &bound, true)
            .await
            .map_err(|error| self.core.ended_or(error.into()))?;
        waits::wait_for_task(inner, id, &bound)
            .await
            .map_err(|error| self.core.ended_or(error.into()))?;
        Ok(())
    }

    /// The live tasks this task owns directly, in id order; not those in its conversations.
    pub async fn owned_tasks(&self, ctx: &Context) -> Result<Vec<TaskRecord>, RuntimeError> {
        let inner = self.core.inner.clone();
        let owner = self.task_id();
        self.core
            .read(ctx, move |_tx| {
                Box::pin(async move {
                    let state = inner.shared.lock();
                    Ok(state.tree.owned_children(owner))
                })
            })
            .await
    }

    /// The outcomes of terminal tasks, in order; rejects when one is not terminal. Used
    /// after a wait.
    pub async fn outcomes(&self, ids: &[TaskId], ctx: &Context) -> Result<Vec<TaskOutcome>, RuntimeError> {
        let ids = ids.to_vec();
        self.core
            .read(ctx, move |tx| {
                Box::pin(async move {
                    let mut outcomes = Vec::with_capacity(ids.len());
                    for id in ids {
                        match tx.task(id).await?.map(|record| record.state) {
                            Some(TaskState::Terminal { outcome }) => outcomes.push(outcome),
                            Some(_) => return Err(Gate::reject(RuntimeError::NotTerminal(id))),
                            None => return Err(Gate::reject(RuntimeError::UnknownTask(id))),
                        }
                    }
                    Ok(outcomes)
                })
            })
            .await
    }

    /// A committed entry by id.
    pub async fn entry(&self, id: EntryId, ctx: &Context) -> Result<Option<EntryRecord>, RuntimeError> {
        self.core
            .read(ctx, move |tx| Box::pin(async move { tx.entry(id).await }))
            .await
    }

    /// The Session, while the invocation lasts.
    pub fn checked_session(&self) -> Result<&Session, RuntimeError> {
        self.core.check()?;
        Ok(&self.core.inner.session)
    }

    /// A committed submission by id.
    pub async fn submission(&self, id: SubmissionId, ctx: &Context) -> Result<Option<SubmissionRecord>, RuntimeError> {
        self.core
            .read(ctx, move |tx| Box::pin(async move { tx.submission(id).await }))
            .await
    }

    /// The current value of a committed document.
    pub async fn snapshot<D: DocDefinition>(
        &self,
        at: DocAddress<D>,
        ctx: &Context,
    ) -> Result<Option<Json>, RuntimeError> {
        self.core.check()?;
        Ok(self.core.inner.session.snapshot(at, ctx).await?)
    }

    /// A rewindable conversation document as of the commit that wrote `entry`.
    pub async fn snapshot_as_of<D: ConversationDoc<History = Rewindable>>(
        &self,
        at: DocAddress<D>,
        entry: EntryId,
        ctx: &Context,
    ) -> Result<Option<Json>, RuntimeError> {
        self.core.check()?;
        Ok(self.core.inner.session.snapshot_as_of(at, entry, ctx).await?)
    }

    /// The Session clock, Unix epoch milliseconds.
    pub fn now(&self) -> i64 {
        self.core.inner.session.now()
    }

    /// Reports a failure that does not fail the phase: a `task_error`, or with `from` an
    /// `extension_error` or `host_error` naming this task. Nothing once the invocation ended.
    pub fn report(&self, error: &(dyn StdError + Send + Sync), from: Option<ReportFrom>) {
        if self.is_ended() {
            return;
        }
        let task = ReportTask {
            id: self.task_id(),
            kind: self.core.invocation.kind.clone(),
            conversation_id: self.conversation_id(),
            extension: self.core.invocation.extension.clone(),
        };
        let info = match from {
            None => ReportInfo::TaskError { task },
            Some(ReportFrom::Extension(extension)) => ReportInfo::ExtensionError {
                task: Some(task),
                conversation: None,
                extension,
            },
            Some(ReportFrom::Host(callback)) => ReportInfo::HostError {
                callback,
                task: Some(task),
            },
        };
        self.core.inner.session.report(error, &info);
    }

    /// A timer that paces work inside the phase (progress commits). It fails once the
    /// invocation's context is cancelled; it is no durable sleep (see [`Next::sleep`]).
    pub fn timer(&self, delay: Duration) -> Timer {
        let invocation = self.core.invocation.clone();
        Timer {
            inner: Box::pin(async move {
                match invocation.ctx.run(tokio::time::sleep(delay)).await {
                    Ok(()) => Ok(()),
                    Err(_) if invocation.is_ended() => Err(RuntimeError::Ended(invocation.task)),
                    Err(cancelled) => Err(RuntimeError::Cancelled(cancelled)),
                }
            }),
        }
    }
}

impl<T: TaskDefinition> fmt::Debug for TaskRuntime<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskRuntime")
            .field("task", &self.task_id())
            .field("mode", &self.core.invocation.mode)
            .field("ended", &self.is_ended())
            .finish_non_exhaustive()
    }
}

/// A runtime timer ([`TaskRuntime::timer`]).
pub struct Timer {
    inner: Pin<Box<dyn Future<Output = Result<(), RuntimeError>> + Send>>,
}

impl Future for Timer {
    type Output = Result<(), RuntimeError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        self.inner.as_mut().poll(cx)
    }
}

impl fmt::Debug for Timer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Timer").finish_non_exhaustive()
    }
}

/// Values of one phase, one per type, dropped when the phase's runtime is.
#[derive(Default)]
pub struct PhaseData {
    values: Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>,
}

impl PhaseData {
    /// The phase's `V`, made by `init` at first use (outside the lock, so `init` may use
    /// the phase's data too).
    pub fn get_or_insert_with<V: Send + Sync + 'static>(&self, init: impl FnOnce() -> V) -> Arc<V> {
        if let Some(value) = self.get::<V>() {
            return value;
        }
        let made = Arc::new(init());
        let mut values = self.values.lock().unwrap_or_else(PoisonError::into_inner);
        let stored = values
            .entry(TypeId::of::<V>())
            .or_insert_with(|| made.clone() as Arc<dyn Any + Send + Sync>)
            .clone();
        drop(values);
        // Keyed by its type: the stored value is a `V`.
        stored.downcast::<V>().unwrap_or(made)
    }

    /// The phase's `V`, if one was inserted.
    pub fn get<V: Send + Sync + 'static>(&self) -> Option<Arc<V>> {
        let values = self.values.lock().unwrap_or_else(PoisonError::into_inner);
        values.get(&TypeId::of::<V>()).cloned()?.downcast::<V>().ok()
    }
}

impl fmt::Debug for PhaseData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = self.values.lock().unwrap_or_else(PoisonError::into_inner).len();
        f.debug_struct("PhaseData").field("values", &count).finish()
    }
}
