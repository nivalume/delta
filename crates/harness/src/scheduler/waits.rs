//! Waits held in memory: a task's terminal record (`wait_for_task`) and idleness
//! (`wait_for_idle`). Both check and register on the line, so no publication falls between
//! them, and end with the scheduler's close.

use std::sync::Arc;

use delta_chord::context::Context;
use delta_chord::json;
use serde::de::DeserializeOwned;
use tokio::sync::watch;

use super::Inner;
use super::error::SchedulerError;
use super::ownership::Scope;
use crate::records::{ConversationId, TaskId, TaskOutcome, TaskRecord, TaskState};
use crate::session::TxError;

/// A task's terminal record: the durable result receipt.
#[derive(Clone, Debug, PartialEq)]
pub struct SettledTask {
    record: TaskRecord,
    outcome: TaskOutcome,
}

impl SettledTask {
    /// `record` when it is terminal.
    pub fn new(record: TaskRecord) -> Option<Self> {
        let TaskState::Terminal { outcome } = &record.state else {
            return None;
        };
        let outcome = outcome.clone();
        Some(Self { record, outcome })
    }

    pub fn record(&self) -> &TaskRecord {
        &self.record
    }

    pub fn into_record(self) -> TaskRecord {
        self.record
    }

    pub fn id(&self) -> TaskId {
        self.record.id
    }

    pub fn outcome(&self) -> &TaskOutcome {
        &self.outcome
    }

    /// The result of a `completed` outcome, decoded; `None` for any other outcome.
    pub fn result<R: DeserializeOwned>(&self) -> Option<Result<R, serde_json::Error>> {
        match self.outcome() {
            TaskOutcome::Completed { result } => Some(json::from_json(result)),
            _ => None,
        }
    }
}

/// What a check on the line found.
enum Found {
    Settled(SettledTask),
    Live(watch::Receiver<()>),
}

/// Resolves with the task's terminal record once it is terminal; rejects at close and for
/// an unknown task. Cancelling `ctx` cancels only this wait.
pub(super) async fn wait_for_task(
    inner: &Arc<Inner>,
    id: TaskId,
    ctx: &Context,
) -> Result<SettledTask, SchedulerError> {
    loop {
        let shared = inner.clone();
        let found = inner
            .session
            .commit(ctx, move |tx| {
                Box::pin(async move {
                    {
                        let mut state = shared.shared.lock();
                        if state.closing {
                            return Ok(None);
                        }
                        if state.tree.live.contains_key(&id) {
                            let watch = state.task_watches.entry(id).or_insert_with(|| watch::Sender::new(()));
                            return Ok(Some(Found::Live(watch.subscribe())));
                        }
                    }
                    let record = tx.task(id).await?.ok_or(TxError::UnknownTask(id))?;
                    Ok(SettledTask::new(record).map(Found::Settled))
                })
            })
            .await
            .map_err(|error| super::abort::unknown_task(error, id))?;
        match found {
            None => return Err(SchedulerError::Closed),
            Some(Found::Settled(settled)) => return Ok(settled),
            // The sender is dropped when the task turns terminal or at close.
            Some(Found::Live(mut receiver)) => {
                let _ = ctx.run(receiver.changed()).await?;
            }
        }
    }
}

/// Resolves once ordinary traversal from `conversation`, or from every ownerless
/// conversation, reaches no live non-background task. `completing`, waiting, blocked and
/// sleeping work is live.
pub(super) async fn wait_for_idle(
    inner: &Arc<Inner>,
    conversation: Option<ConversationId>,
    ctx: &Context,
) -> Result<(), SchedulerError> {
    let scope = conversation.map_or(Scope::Roots, Scope::Conversation);
    let mut changes = inner.shared.changes.subscribe();
    let mut first = true;
    loop {
        changes.borrow_and_update();
        {
            let mut state = inner.shared.lock();
            if state.closing {
                return Err(SchedulerError::Closed);
            }
            if state.tree.idle(scope) {
                return Ok(());
            }
            // Tasks whose chain is not loaded count as inside until a pass loads them.
            if first && !state.tree.unloaded.is_empty() {
                state.reconcile_scheduled = true;
                inner.shared.wake_driver();
            }
        }
        first = false;
        if ctx.run(changes.changed()).await?.is_err() {
            return Err(SchedulerError::Closed);
        }
    }
}
