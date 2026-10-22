//! How compactions show in `delta.live`: one [`CompactionStatus`] per live compaction task, and
//! the newest [`CompactionFailure`] until a summary is placed.

use serde::{Deserialize, Serialize};

use super::CompactionReason;
use crate::generation::{LiveRetry, LiveState};
use crate::records::TaskId;

/// The presentation of one live compaction task.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionStatus {
    pub task_id: TaskId,
    pub reason: CompactionReason,
    /// Whether a generation waits for it: a compaction the generation owns.
    pub blocking: bool,
    pub attempt: u32,
    /// A durable backoff before the next attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry: Option<LiveRetry>,
}

/// The newest compaction that failed since a compaction last placed a summary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionFailure {
    pub task_id: TaskId,
    pub reason: CompactionReason,
    pub error: String,
}

impl LiveState {
    /// Adds the status of a compaction created in this commit; statuses stay in task order.
    pub fn add_compaction(&mut self, status: CompactionStatus) {
        let statuses = self.compactions.get_or_insert_with(Vec::new);
        statuses.push(status);
        statuses.sort_by_key(|status| status.task_id);
    }

    /// The status of compaction task `task`.
    pub fn compaction_mut(&mut self, task: TaskId) -> Option<&mut CompactionStatus> {
        self.compactions
            .iter_mut()
            .flatten()
            .find(|status| status.task_id == task)
    }

    /// Removes the status of `task`, and the list once empty; returns it.
    pub fn remove_compaction(&mut self, task: TaskId) -> Option<CompactionStatus> {
        let statuses = self.compactions.as_mut()?;
        let index = statuses.iter().position(|status| status.task_id == task)?;
        let status = statuses.remove(index);
        if statuses.is_empty() {
            self.compactions = None;
        }
        Some(status)
    }

    /// Removes the status of `task` and records that it failed with `error`.
    pub fn fail_compaction(&mut self, task: TaskId, error: String) {
        if let Some(status) = self.remove_compaction(task) {
            self.compaction_failure = Some(CompactionFailure {
                task_id: task,
                reason: status.reason,
                error,
            });
        }
    }
}
