//! Task records: durable work with its state machine, outcome and bookkeeping.

use delta_chord::json::Json;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::error::RecordError;
use super::ids::{ConversationId, TaskId};

/// A durable task, replaced whole on every transition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskRecord {
    pub id: TaskId,
    /// The conversation the task belongs to; immutable.
    pub conversation_id: ConversationId,
    /// The task definition's name.
    pub kind: String,
    /// The task definition's version that wrote `input` and the checkpoint.
    pub version: u32,
    pub input: Json,
    /// The owning task; absent for a conversation-owned task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<TaskId>,
    pub background: bool,
    pub abort_requested: bool,
    /// Why the abort mark was set when the reason is not the caller's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abort_reason: Option<AbortReason>,
    /// JSON: present only as `true`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub abandon_on_restart: bool,
    /// Unix epoch milliseconds of the first change to `running`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<i64>,
    /// Unix epoch milliseconds of the change to `terminal`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restarts: Option<u32>,
    pub state: TaskState,
    /// Per-task memo values; live states only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memos: Option<IndexMap<String, Json>>,
    /// Telemetry carrier, kept unread until delta has a telemetry crate.
    #[serde(
        default,
        deserialize_with = "super::present",
        skip_serializing_if = "Option::is_none"
    )]
    pub trace: Option<Json>,
    /// Telemetry links, kept unread.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub links: Option<Vec<Json>>,
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde's `skip_serializing_if` passes a reference.
fn is_false(value: &bool) -> bool {
    !*value
}

impl TaskRecord {
    /// A pending, foreground, unmarked task with nothing optional set.
    pub fn new(
        id: TaskId,
        conversation_id: ConversationId,
        kind: impl Into<String>,
        version: u32,
        input: Json,
        state: TaskState,
    ) -> Self {
        Self {
            id,
            conversation_id,
            kind: kind.into(),
            version,
            input,
            owner: None,
            background: false,
            abort_requested: false,
            abort_reason: None,
            abandon_on_restart: false,
            started_at: None,
            ended_at: None,
            restarts: None,
            state,
            memos: None,
            trace: None,
            links: None,
        }
    }

    /// Refuses memos on a `completing` or `terminal` task.
    pub fn check(&self) -> Result<(), RecordError> {
        if self.memos.is_some() && self.state.outcome().is_some() {
            return Err(RecordError::Task {
                id: self.id,
                problem: format!("a {} task keeps no memos", self.state.status().as_str()),
            });
        }
        Ok(())
    }
}

/// A task's state, tagged by `status`. Live states carry the checkpoint the next phase
/// resumes from; `completing` and `terminal` carry the outcome.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TaskState {
    Pending {
        checkpoint: Json,
    },
    Running {
        checkpoint: Json,
    },
    /// Waits for `on` under `policy`.
    Waiting {
        checkpoint: Json,
        on: Vec<TaskId>,
        policy: JoinPolicy,
    },
    /// Sleeps until `until` (Unix epoch milliseconds).
    Sleeping {
        checkpoint: Json,
        until: i64,
    },
    /// The outcome is decided; cleanup is pending.
    Completing {
        outcome: TaskOutcome,
    },
    Terminal {
        outcome: TaskOutcome,
    },
}

impl TaskState {
    pub fn status(&self) -> TaskStatus {
        match self {
            Self::Pending { .. } => TaskStatus::Pending,
            Self::Running { .. } => TaskStatus::Running,
            Self::Waiting { .. } => TaskStatus::Waiting,
            Self::Sleeping { .. } => TaskStatus::Sleeping,
            Self::Completing { .. } => TaskStatus::Completing,
            Self::Terminal { .. } => TaskStatus::Terminal,
        }
    }

    /// The checkpoint of a live state.
    pub fn checkpoint(&self) -> Option<&Json> {
        match self {
            Self::Pending { checkpoint }
            | Self::Running { checkpoint }
            | Self::Waiting { checkpoint, .. }
            | Self::Sleeping { checkpoint, .. } => Some(checkpoint),
            Self::Completing { .. } | Self::Terminal { .. } => None,
        }
    }

    /// The outcome of a `completing` or `terminal` state.
    pub fn outcome(&self) -> Option<&TaskOutcome> {
        match self {
            Self::Completing { outcome } | Self::Terminal { outcome } => Some(outcome),
            _ => None,
        }
    }
}

/// How a task ended, tagged by `status`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TaskOutcome {
    Completed {
        result: Json,
    },
    Failed {
        error: TaskOutcomeError,
        #[serde(
            default,
            deserialize_with = "super::present",
            skip_serializing_if = "Option::is_none"
        )]
        result: Option<Json>,
    },
    Aborted {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        #[serde(
            default,
            deserialize_with = "super::present",
            skip_serializing_if = "Option::is_none"
        )]
        result: Option<Json>,
    },
    /// The owner went away before the task finished.
    Orphaned {
        reason: String,
    },
    /// The task's own code failed (threw, or exceeded the restart limit).
    Faulted {
        error: TaskOutcomeError,
    },
}

/// The error of a failed or faulted outcome.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskOutcomeError {
    pub message: String,
    #[serde(
        default,
        deserialize_with = "super::present",
        skip_serializing_if = "Option::is_none"
    )]
    pub detail: Option<Json>,
}

impl TaskOutcomeError {
    /// An error with `message` and no detail.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            detail: None,
        }
    }
}

/// How a waiting task treats failures among the tasks it waits for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinPolicy {
    /// Wake on the first failure.
    FailFast,
    /// Wake once every task settled.
    AllSettled,
}

/// Why an abort mark was set by the runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbortReason {
    /// The task was abandoned after a restart (`abandon_on_restart`).
    Restart,
}

/// The status alone: query filters and the SQLite `status` column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Running,
    Waiting,
    Sleeping,
    Completing,
    Terminal,
}

impl TaskStatus {
    /// The serialized name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Waiting => "waiting",
            Self::Sleeping => "sleeping",
            Self::Completing => "completing",
            Self::Terminal => "terminal",
        }
    }
}
