//! Compaction (spec §8.7): the `delta.compaction` task that replaces an old prefix of the
//! model context with a summary entry whose `head` is the first kept entry, and the
//! thresholds the generation checks before a request (§8.3).
//!
//! The generation reaches compaction through the [`Compactor`] seam; the default
//! [`TaskCompactor`] selects a cut with [`select_cut`] and starts a [`CompactionTask`]. A
//! compaction the generation owns blocks it and appends its summary directly; a
//! conversation-owned one (background above the background threshold, or manual through
//! `Conversation::compact`) places its summary as a write submission. Summarizing is one
//! pinned model request with retries; raw history stays in storage. Corresponds to
//! upstream `src/harness/compaction.ts`. See `docs/design/compaction.md`.

mod select;
mod status;
mod summary;
mod task;

use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

pub use select::{Threshold, estimate_context, select_cut, summarized_messages, threshold};
pub use status::{CompactionFailure, CompactionStatus};
pub use summary::{SUMMARIZATION_PROMPT, SUMMARIZATION_SYSTEM_PROMPT, serialize_conversation};
pub use task::{CompactionCheckpoint, CompactionTask, SummaryRequest};

use crate::context::ContextView;
use crate::generation::edit_live;
use crate::records::{ConversationId, EntryId, SubmissionId, TaskId};
use crate::session::{TaskOptions, Tx, TxError};

/// The task kind.
pub const COMPACTION_TASK: &str = "delta.compaction";
/// The entry kind of a summary.
pub const COMPACTION_ENTRY: &str = "delta.compaction";

/// Why a compaction runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CompactionReason {
    Manual,
    Threshold,
    Overflow,
}

/// The input of a `delta.compaction` task.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionInput {
    pub reason: CompactionReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// What a compaction placed: the summary entry of a blocking one, the write submission of a
/// conversation-owned one; neither when there was nothing to compact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionResult {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_id: Option<EntryId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub submission_id: Option<SubmissionId>,
}

/// The seam between generation and compaction.
pub trait Compactor: Send + Sync {
    /// Range selection over `view`: the index into `view.entries` of the first entry a
    /// summary keeps, or `None` when there is nothing to compact.
    fn select_cut(&self, view: &ContextView, keep_recent_tokens: u64) -> Option<usize>;

    /// Creates a `delta.compaction` task in `tx`: owned by the generation `owner` (blocking,
    /// the generation waits for it), or by the conversation (background unless manual), and
    /// adds its status to `delta.live.compactions` in the same commit.
    fn create<'a>(
        &'a self,
        tx: &'a mut Tx<'_>,
        conversation: ConversationId,
        input: CompactionInput,
        owner: Option<TaskId>,
    ) -> BoxFuture<'a, Result<TaskId, TxError>>;
}

/// The default [`Compactor`]: [`select_cut`] and the [`CompactionTask`].
#[derive(Clone, Copy, Debug, Default)]
pub struct TaskCompactor;

impl Compactor for TaskCompactor {
    fn select_cut(&self, view: &ContextView, keep_recent_tokens: u64) -> Option<usize> {
        select_cut(view, keep_recent_tokens)
    }

    fn create<'a>(
        &'a self,
        tx: &'a mut Tx<'_>,
        conversation: ConversationId,
        input: CompactionInput,
        owner: Option<TaskId>,
    ) -> BoxFuture<'a, Result<TaskId, TxError>> {
        Box::pin(create_compaction(tx, conversation, input, owner))
    }
}

/// A [`Compactor`] that never compacts: no threshold fires and an overflow fails the run.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoCompaction;

impl Compactor for NoCompaction {
    fn select_cut(&self, _view: &ContextView, _keep_recent_tokens: u64) -> Option<usize> {
        None
    }

    fn create<'a>(
        &'a self,
        _tx: &'a mut Tx<'_>,
        _conversation: ConversationId,
        _input: CompactionInput,
        _owner: Option<TaskId>,
    ) -> BoxFuture<'a, Result<TaskId, TxError>> {
        Box::pin(async { Err(TxError::callback("Compaction is disabled")) })
    }
}

/// Creates a compaction with its status in this commit: owned by `owner` (the generation
/// that waits for it) or by the conversation, `background` unless manual.
pub async fn create_compaction(
    tx: &mut Tx<'_>,
    conversation: ConversationId,
    input: CompactionInput,
    owner: Option<TaskId>,
) -> Result<TaskId, TxError> {
    let options = match owner {
        Some(owner) => TaskOptions::owned_by(owner),
        None => TaskOptions {
            background: input.reason != CompactionReason::Manual,
            ..TaskOptions::conversation(conversation)
        },
    };
    let reason = input.reason;
    let task_id = tx.create_task(&CompactionTask, input, options).await?.id();
    let status = CompactionStatus {
        task_id,
        reason,
        blocking: owner.is_some(),
        attempt: 1,
        retry: None,
    };
    edit_live(tx, conversation, |live| live.add_compaction(status)).await?;
    Ok(task_id)
}

#[cfg(test)]
mod tests;
