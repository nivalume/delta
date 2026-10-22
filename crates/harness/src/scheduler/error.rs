//! The scheduler's errors: refused scheduler calls, refused runtime operations, states a
//! task may not commit, and the error a phase handler returns.

use std::error::Error as StdError;
use std::fmt;

use delta_chord::context::Cancelled;

use crate::records::{TaskId, TaskStatus};
use crate::session::{SessionError, TxError};

/// Why a [`Scheduler`](super::Scheduler) call did not succeed.
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    /// The scheduler is closing or closed; waits end with it.
    #[error("The scheduler is closed")]
    Closed,
    #[error("Task {0} does not exist")]
    UnknownTask(TaskId),
    /// The caller's context was cancelled; shared work is unaffected.
    #[error(transparent)]
    Cancelled(Cancelled),
    #[error(transparent)]
    Session(SessionError),
}

impl From<SessionError> for SchedulerError {
    fn from(error: SessionError) -> Self {
        match error {
            SessionError::Closed => Self::Closed,
            SessionError::Cancelled(cancelled) => Self::Cancelled(cancelled),
            other => Self::Session(other),
        }
    }
}

impl From<Cancelled> for SchedulerError {
    fn from(cancelled: Cancelled) -> Self {
        Self::Cancelled(cancelled)
    }
}

/// Why a [`TaskRuntime`](super::TaskRuntime) operation was refused.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// The invocation that owns the runtime ended; detached work holding it stops here.
    #[error("Task {0} invocation has ended")]
    Ended(TaskId),
    /// The scheduler is closing.
    #[error("The scheduler is closing")]
    Closing,
    #[error("Task {0} is terminal")]
    Terminal(TaskId),
    /// The task holds its outcome, waits or sleeps: its invocation is about to end.
    #[error("Task {task} is {}", status.as_str())]
    NotRunning { task: TaskId, status: TaskStatus },
    /// A run invocation's task carries an abort mark; only its abort handler commits now.
    #[error("Task {0} has a durable abort mark")]
    Aborted(TaskId),
    #[error("Task {task} is not owned by task {owner}")]
    NotOwned { task: TaskId, owner: TaskId },
    #[error("Task {0} is not terminal")]
    NotTerminal(TaskId),
    #[error("Task {0} does not exist")]
    UnknownTask(TaskId),
    /// The state the commit returned may not be committed.
    #[error(transparent)]
    State(StateError),
    /// A memo, input, checkpoint or result that does not convert.
    #[error(transparent)]
    Codec(serde_json::Error),
    #[error(transparent)]
    Cancelled(Cancelled),
    #[error(transparent)]
    Session(SessionError),
}

impl From<SessionError> for RuntimeError {
    fn from(error: SessionError) -> Self {
        match error {
            SessionError::Closed => Self::Closing,
            SessionError::Cancelled(cancelled) => Self::Cancelled(cancelled),
            SessionError::Tx(TxError::Callback(source)) => match source.downcast::<Gate>() {
                Ok(gate) => gate.0,
                Err(source) => match source.downcast::<StateError>() {
                    Ok(state) => Self::State(*state),
                    Err(source) => Self::Session(SessionError::Tx(TxError::Callback(source))),
                },
            },
            other => Self::Session(other),
        }
    }
}

impl From<SchedulerError> for RuntimeError {
    fn from(error: SchedulerError) -> Self {
        match error {
            SchedulerError::Closed => Self::Closing,
            SchedulerError::UnknownTask(id) => Self::UnknownTask(id),
            SchedulerError::Cancelled(cancelled) => Self::Cancelled(cancelled),
            SchedulerError::Session(error) => Self::from(error),
        }
    }
}

/// A runtime refusal decided on the line, carried out of the commit as its closure error.
#[derive(Debug)]
pub(super) struct Gate(pub(super) RuntimeError);

impl fmt::Display for Gate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl StdError for Gate {}

impl Gate {
    pub(super) fn reject(error: RuntimeError) -> TxError {
        TxError::callback(Self(error))
    }
}

/// A state a task may not commit (spec §5.1): a wait on a missing task or one that closes a
/// wait cycle, a `fail_fast` wait on a task it does not own, a sleep until a negative time,
/// a wait or sleep from an abort handler, an abort handler that ends without an outcome.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StateError {
    #[error("Task {0} does not exist")]
    UnknownTask(TaskId),
    #[error("Task {task} can wait fail_fast only on tasks it owns; {on} is not one")]
    FailFastNotOwned { task: TaskId, on: TaskId },
    /// `path` runs from the waiter through `on` back to the waiter.
    #[error("Task {task} cannot wait on {on}: {} is a wait cycle", cycle_path(path))]
    Cycle {
        task: TaskId,
        on: TaskId,
        path: Vec<TaskId>,
    },
    #[error("Task {task} cannot sleep until {until}: not a time in milliseconds")]
    InvalidSleep { task: TaskId, until: i64 },
    #[error("Abort handler of task {0} cannot wait")]
    AbortWait(TaskId),
    #[error("Abort handler of task {0} cannot sleep")]
    AbortSleep(TaskId),
    #[error("Abort handler of task {0} returned without a terminal outcome")]
    AbortWithoutOutcome(TaskId),
}

fn cycle_path(cycle: &[TaskId]) -> String {
    cycle.iter().map(ToString::to_string).collect::<Vec<_>>().join(" -> ")
}

/// The error a phase or abort handler returns: any error, or a message. The step writes the
/// task `faulted` with its message (rule 4).
pub struct TaskError {
    source: Box<dyn StdError + Send + Sync>,
}

impl TaskError {
    /// An error with `message`.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            source: message.into().into(),
        }
    }

    /// The error the handler returned.
    pub fn source(&self) -> &(dyn StdError + Send + Sync + 'static) {
        &*self.source
    }

    /// The error, boxed.
    pub fn into_source(self) -> Box<dyn StdError + Send + Sync> {
        self.source
    }
}

/// Every error converts, so `?` works in handlers.
impl<E: Into<Box<dyn StdError + Send + Sync>>> From<E> for TaskError {
    fn from(error: E) -> Self {
        Self { source: error.into() }
    }
}

impl fmt::Display for TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.source, f)
    }
}

impl fmt::Debug for TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.source, f)
    }
}

/// A failure the scheduler reports with only a message (a fault, a parked task).
#[derive(Clone, Debug, thiserror::Error)]
#[error("{0}")]
pub(super) struct Failure(pub(super) String);
