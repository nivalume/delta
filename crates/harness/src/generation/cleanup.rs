//! The agent layer's cleanup in the scheduler's own commits: outcomes the scheduler writes
//! (`faulted`, `orphaned`) and withdrawals of queued inputs (spec §5.4).

use delta_chord::json::Json;
use futures::future::BoxFuture;

use super::GENERATION_TASK;
use super::live::{convert_partial, edit_live, end_run, live_state};
use crate::inbox::withdraw_inputs;
use crate::records::{
    ConversationId, SubmissionRecord, SubmissionSettlement, TaskId, TaskOutcome, TaskRecord, TaskState,
};
use crate::scheduler::SchedulerHooks;
use crate::session::{Tx, TxError};
use crate::tool::TOOL_TASK;

/// The [`SchedulerHooks`] of a Harness. It handles the agent layer's task kinds and ignores
/// every other, so it never creates `delta.live` for a task it does not know:
///
/// - a run generation: a committed partial becomes an aborted entry, the run's inputs
///   settle `unanswered` (`faulted` with the message, or the orphaning reason), and the run
///   marker is removed;
/// - a tool task: its slot is marked done without an entry (context derivation synthesizes
///   the missing result) and its nested slots are removed.
///
/// Queued inputs the scheduler withdraws also leave `delta.inbox`.
#[derive(Clone, Copy, Debug, Default)]
pub struct AgentSchedulerHooks;

impl SchedulerHooks for AgentSchedulerHooks {
    fn settle_scheduler_outcome<'a>(
        &'a self,
        tx: &'a mut Tx<'_>,
        record: &'a TaskRecord,
    ) -> BoxFuture<'a, Result<(), TxError>> {
        Box::pin(async move {
            match record.kind.as_str() {
                GENERATION_TASK => settle_generation(tx, record).await,
                TOOL_TASK => settle_tool(tx, record.conversation_id, record.id).await,
                _ => Ok(()),
            }
        })
    }

    fn withdraw_inputs<'a>(
        &'a self,
        tx: &'a mut Tx<'_>,
        conversation: ConversationId,
        queued: &'a [SubmissionRecord],
    ) -> BoxFuture<'a, Result<(), TxError>> {
        Box::pin(withdraw_inputs(tx, conversation, queued))
    }
}

async fn settle_generation(tx: &mut Tx<'_>, record: &TaskRecord) -> Result<(), TxError> {
    let conversation = record.conversation_id;
    let owns_run = live_state(tx, conversation)
        .await?
        .run
        .is_some_and(|run| run.task_id == record.id);
    if !owns_run {
        return Ok(());
    }
    let settlement = match &record.state {
        TaskState::Terminal {
            outcome: TaskOutcome::Faulted { error },
        } => SubmissionSettlement::Unanswered {
            reason: "faulted".to_owned(),
            detail: Some(Json::from(error.message.clone())),
        },
        TaskState::Terminal {
            outcome: TaskOutcome::Orphaned { reason },
        } => SubmissionSettlement::Unanswered {
            reason: reason.clone(),
            detail: None,
        },
        _ => return Ok(()),
    };
    convert_partial(tx, conversation).await?;
    end_run(tx, conversation, record.id, &settlement).await
}

async fn settle_tool(tx: &mut Tx<'_>, conversation: ConversationId, task: TaskId) -> Result<(), TxError> {
    edit_live(tx, conversation, |live| {
        if let Some(slot) = live.tools.iter_mut().flatten().find(|slot| slot.task_id == Some(task)) {
            slot.finish();
        }
        if let Some(nested) = &mut live.nested_tools {
            remove_nested(nested, task);
            if nested.is_empty() {
                live.nested_tools = None;
            }
        }
    })
    .await
}

/// Removes the nested slots below `task`, transitively; parents precede their children.
fn remove_nested(nested: &mut Vec<Json>, task: TaskId) {
    let id = |slot: &Json, field: &str| slot.get(field).and_then(Json::as_u64).map(TaskId);
    let mut removed = vec![task];
    for slot in nested.iter() {
        if id(slot, "parent_task_id").is_some_and(|parent| removed.contains(&parent))
            && let Some(child) = id(slot, "task_id")
        {
            removed.push(child);
        }
    }
    nested.retain(|slot| !id(slot, "parent_task_id").is_some_and(|parent| removed.contains(&parent)));
}
