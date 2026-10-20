//! The generation's checkpoint: one variant per phase.

use delta_ai::model::ModelThinkingLevel;
use delta_ai::types::DeferredHandle;
use serde::{Deserialize, Serialize};

use crate::model::{ConversationStreamOptions, ModelRef};
use crate::records::{EntryId, TaskId};
use crate::scheduler::Checkpoint;

/// Where a generation resumes, stored as `{"phase":"request","attempt":1,...}`; absent
/// options are absent in the JSON.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum GenerationCheckpoint {
    /// Render the prompt, plan `delta.system` entries, check the compaction thresholds.
    Prepare {
        attempt: u32,
        /// The blocking compaction this generation waited for; it starts no other.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        compacted: Option<TaskId>,
        /// The error text of the overflow that started `compacted`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        overflow: Option<String>,
        /// Set when the blocking threshold started `compacted`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        threshold: Option<bool>,
    },
    /// Send the request prepared through `cutoff` with the pinned model and options.
    Request {
        attempt: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        compacted: Option<TaskId>,
        model: ModelRef,
        thinking_level: ModelThinkingLevel,
        stream_options: ConversationStreamOptions,
        /// The newest entry the request includes.
        cutoff: EntryId,
    },
    /// The backoff before attempt `attempt + 1`.
    Retry {
        attempt: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        compacted: Option<TaskId>,
        until: i64,
    },
    /// Poll a deferred response at `poll_at`.
    Poll {
        attempt: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        compacted: Option<TaskId>,
        model: ModelRef,
        cutoff: EntryId,
        handle: DeferredHandle,
        poll_at: i64,
    },
    /// Waiting on the round's tool tasks, which the generation owns.
    Tools {
        /// The tool-calling answer.
        assistant: EntryId,
        /// The tool tasks created so far, in call order.
        tools: Vec<TaskId>,
        /// Calls of a sequential round not started yet, in call order.
        pending: Vec<String>,
    },
}

impl GenerationCheckpoint {
    /// The first phase of attempt `attempt`.
    pub fn prepare(attempt: u32, compacted: Option<TaskId>) -> Self {
        Self::Prepare {
            attempt,
            compacted,
            overflow: None,
            threshold: None,
        }
    }
}

impl Checkpoint for GenerationCheckpoint {
    fn phase(&self) -> &str {
        match self {
            Self::Prepare { .. } => "prepare",
            Self::Request { .. } => "request",
            Self::Retry { .. } => "retry",
            Self::Poll { .. } => "poll",
            Self::Tools { .. } => "tools",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_options_are_absent_in_json() {
        let json = serde_json::to_value(GenerationCheckpoint::prepare(1, None)).unwrap();
        assert_eq!(json, serde_json::json!({ "phase": "prepare", "attempt": 1 }));
        let retry = GenerationCheckpoint::Retry {
            attempt: 2,
            compacted: Some(TaskId(5)),
            until: 10,
        };
        let json = serde_json::to_value(&retry).unwrap();
        assert_eq!(
            json,
            serde_json::json!({ "phase": "retry", "attempt": 2, "compacted": 5, "until": 10 })
        );
        assert_eq!(serde_json::from_value::<GenerationCheckpoint>(json).unwrap(), retry);
    }
}
