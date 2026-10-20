//! The commit that ends a phase: [`Next`], the [`NextState`] it returns and the typed
//! [`Outcome`] of a task that ends itself.

use std::fmt;

use delta_chord::json;
use futures::future::BoxFuture;
use serde::Serialize;

use super::definition::{RunningTask, TaskDefinition};
use crate::records::{JoinPolicy, TaskId, TaskOutcome, TaskOutcomeError, TaskRecord, TaskState};
use crate::session::{Tx, TxError};

/// How a task ends itself. `orphaned` and `faulted` are written only by the scheduler.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome<R> {
    Completed(R),
    Failed { error: TaskOutcomeError, result: Option<R> },
    Aborted { reason: Option<String>, result: Option<R> },
}

impl<R: Serialize> Outcome<R> {
    /// The stored outcome.
    pub fn encode(self) -> Result<TaskOutcome, serde_json::Error> {
        let encode = |result: Option<R>| result.map(|result| json::to_json(&result)).transpose();
        Ok(match self {
            Self::Completed(result) => TaskOutcome::Completed {
                result: json::to_json(&result)?,
            },
            Self::Failed { error, result } => TaskOutcome::Failed {
                error,
                result: encode(result)?,
            },
            Self::Aborted { reason, result } => TaskOutcome::Aborted {
                reason,
                result: encode(result)?,
            },
        })
    }
}

/// The state a task commits for itself. A returned terminal outcome is held as
/// `completing` while ordinary owned work below the task is live (spec §5.5); `pending` is
/// never returned.
#[derive(Clone, Debug, PartialEq)]
pub enum NextState<S, R> {
    /// A replacement checkpoint; the invocation continues with its phase.
    Running(S),
    /// Parks the task until every task in `on` is terminal; it resumes at `checkpoint`.
    Waiting {
        checkpoint: S,
        on: Vec<TaskId>,
        policy: JoinPolicy,
    },
    /// Parks the task until the clock reaches `until` (Unix epoch milliseconds).
    Sleeping {
        checkpoint: S,
        until: i64,
    },
    Terminal(Outcome<R>),
}

impl<S, R> NextState<S, R> {
    /// Completes with `result`.
    pub fn completed(result: R) -> Self {
        Self::Terminal(Outcome::Completed(result))
    }

    /// Fails with `message`.
    pub fn failed(message: impl Into<String>) -> Self {
        Self::Terminal(Outcome::Failed {
            error: TaskOutcomeError::new(message),
            result: None,
        })
    }

    /// Ends aborted, as an abort handler does.
    pub fn aborted(reason: Option<String>) -> Self {
        Self::Terminal(Outcome::Aborted { reason, result: None })
    }
}

impl<S: Serialize, R: Serialize> NextState<S, R> {
    /// The stored state.
    pub fn encode(self) -> Result<TaskState, serde_json::Error> {
        Ok(match self {
            Self::Running(checkpoint) => TaskState::Running {
                checkpoint: json::to_json(&checkpoint)?,
            },
            Self::Waiting { checkpoint, on, policy } => TaskState::Waiting {
                checkpoint: json::to_json(&checkpoint)?,
                on,
                policy,
            },
            Self::Sleeping { checkpoint, until } => TaskState::Sleeping {
                checkpoint: json::to_json(&checkpoint)?,
                until,
            },
            Self::Terminal(outcome) => TaskState::Terminal {
                outcome: outcome.encode()?,
            },
        })
    }
}

/// The typed next state of `T`.
pub type NextOf<T> = NextState<<T as TaskDefinition>::Checkpoint, <T as TaskDefinition>::Result>;

type NextFn<T> =
    Box<dyn for<'a, 'l> FnOnce(&'a mut Tx<'l>, &'a RunningTask<T>) -> BoxFuture<'a, Result<NextOf<T>, TxError>> + Send>;

/// The type-erased closing commit the scheduler runs in its step.
pub(super) type ErasedNext =
    Box<dyn for<'a, 'l> FnOnce(&'a mut Tx<'l>, &'a TaskRecord) -> BoxFuture<'a, Result<TaskState, TxError>> + Send>;

/// The closing commit of a phase, returned by [`TaskDefinition::run`] and
/// [`TaskDefinition::abort`].
///
/// The closure runs on the commit line inside the scheduler's step commit, with the task as
/// it is committed (`current`): its entries, documents and child tasks land in the same
/// commit as the next state and the step's decision, so a result entry is atomic with the
/// terminal state (§5.3). An error from the closure rolls its writes back and faults the
/// task; an abort mark that landed during the phase discards the commit unrun.
pub struct Next<T: TaskDefinition> {
    commit: NextFn<T>,
}

impl<T: TaskDefinition> Next<T> {
    /// A closing commit that writes with `f` and returns the next state.
    pub fn commit<F>(f: F) -> Self
    where
        F: for<'a, 'l> FnOnce(&'a mut Tx<'l>, &'a RunningTask<T>) -> BoxFuture<'a, Result<NextOf<T>, TxError>>
            + Send
            + 'static,
    {
        Self { commit: Box::new(f) }
    }

    /// A closing commit that writes nothing but `state`.
    pub fn state(state: NextOf<T>) -> Self {
        Self::commit(move |_, _| Box::pin(async move { Ok(state) }))
    }

    /// Continues with `checkpoint`.
    pub fn running(checkpoint: T::Checkpoint) -> Self {
        Self::state(NextState::Running(checkpoint))
    }

    /// Sleeps until `until` (epoch ms on the Session clock), then resumes at `checkpoint`.
    pub fn sleep(until: i64, checkpoint: T::Checkpoint) -> Self {
        Self::state(NextState::Sleeping { checkpoint, until })
    }

    /// Waits until every task in `on` is terminal, then resumes at `checkpoint`.
    pub fn wait(on: Vec<TaskId>, policy: JoinPolicy, checkpoint: T::Checkpoint) -> Self {
        Self::state(NextState::Waiting { checkpoint, on, policy })
    }

    /// Completes with `result`.
    pub fn complete(result: T::Result) -> Self {
        Self::state(NextState::completed(result))
    }

    /// Fails with `message`.
    pub fn fail(message: impl Into<String>) -> Self {
        Self::state(NextState::failed(message))
    }

    /// Ends aborted with `reason`.
    pub fn abort(reason: Option<String>) -> Self {
        Self::state(NextState::aborted(reason))
    }

    /// The commit over stored records: decodes `current`, runs the closure, encodes the state.
    pub(super) fn erase(self) -> ErasedNext {
        let commit = self.commit;
        erased(move |tx, record| {
            Box::pin(async move {
                let current = RunningTask::<T>::decode(record.clone())?;
                let state = commit(tx, &current).await?;
                state
                    .encode()
                    .map_err(|error| TxError::callback(format!("Task {} state does not convert: {error}", record.id)))
            })
        })
    }
}

/// Boxes an erased commit; the bound makes the closure higher-ranked.
fn erased<F>(f: F) -> ErasedNext
where
    F: for<'a, 'l> FnOnce(&'a mut Tx<'l>, &'a TaskRecord) -> BoxFuture<'a, Result<TaskState, TxError>> + Send + 'static,
{
    Box::new(f)
}

impl<T: TaskDefinition> fmt::Debug for Next<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Next").finish_non_exhaustive()
    }
}
