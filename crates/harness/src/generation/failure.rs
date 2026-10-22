//! Failed runs: settling the inputs `unanswered` and what `after_failed_run` does with the
//! items queued behind them.

use delta_ai::types::AssistantMessage;
use delta_chord::json::Json;

use super::live::{append_assistant, end_run, start_run};
use super::{GenerationTask, Rt};
use crate::agent::{AfterFailedRun, AgentRuntime};
use crate::inbox::{At, Boundary, QueueModes, apply_boundary, prepare_boundary, withdraw_queued_inputs};
use crate::model::ModelRef;
use crate::records::{ConversationId, SubmissionSettlement, TaskId, TaskOutcomeError};
use crate::scheduler::{Next, NextState, Outcome, TaskError};
use crate::session::{Tx, TxError};

/// What a failed run's end needs from the tables, read before the commit's first table
/// write.
pub(super) struct Failure {
    policy: AfterFailedRun,
    boundary: Option<Boundary>,
}

/// The policy and the queue modes a failure reads, from the settings now.
#[derive(Clone, Copy)]
pub(super) struct FailurePolicy {
    policy: AfterFailedRun,
    modes: QueueModes,
}

impl FailurePolicy {
    pub(super) fn of(rt: &Rt) -> Result<Self, TaskError> {
        let settings = rt.agent_services()?.settings();
        Ok(Self {
            policy: settings.after_failed_run,
            modes: QueueModes::from(&settings),
        })
    }

    /// Reads what the failure needs; call before the commit's first table write.
    pub(super) async fn prepare(self, tx: &mut Tx<'_>, conversation: ConversationId) -> Result<Failure, TxError> {
        let boundary = match self.policy {
            AfterFailedRun::Continue => Some(prepare_boundary(tx, conversation, self.modes).await?),
            AfterFailedRun::Keep | AfterFailedRun::Withdraw => None,
        };
        Ok(Failure {
            policy: self.policy,
            boundary,
        })
    }
}

/// Ends a failed run with `settlement` and applies `after_failed_run`: keep the queued items,
/// start the next run with them, or withdraw the queued inputs.
pub(super) async fn end_failed_run(
    tx: &mut Tx<'_>,
    conversation: ConversationId,
    task: TaskId,
    failure: Failure,
    settlement: &SubmissionSettlement,
    now: i64,
) -> Result<(), TxError> {
    end_run(tx, conversation, task, settlement).await?;
    if failure.policy == AfterFailedRun::Withdraw {
        withdraw_queued_inputs(tx, conversation, "run_failed").await?;
    }
    if let Some(mut boundary) = failure.boundary {
        let placed = apply_boundary(tx, &mut boundary, At::Final, now).await?;
        if !placed.users.is_empty() {
            start_run(tx, conversation, placed.users).await?;
        }
    }
    Ok(())
}

/// The failed outcome with `message` and `{ "reason": reason }`.
fn failed(message: String, reason: &str) -> NextState<super::GenerationCheckpoint, super::GenerationResult> {
    NextState::Terminal(Outcome::Failed {
        error: TaskOutcomeError {
            message,
            detail: Some(delta_chord::json!({ "reason": reason })),
        },
        result: None,
    })
}

/// Fails the run: appends `message`'s entry when given, settles the inputs `unanswered`
/// with `reason` (and `detail`), applies `after_failed_run` and ends `failed`.
pub(super) fn fail_run(
    rt: &Rt,
    reason: &'static str,
    text: String,
    detail: Option<Json>,
    message: Option<AssistantMessage>,
) -> Result<Next<GenerationTask>, TaskError> {
    let policy = FailurePolicy::of(rt)?;
    let conversation = rt.conversation_id();
    let task = rt.task_id();
    let now = rt.now();
    Ok(rt.next(move |tx, _| {
        Box::pin(async move {
            let failure = policy.prepare(tx, conversation).await?;
            if let Some(message) = message {
                append_assistant(tx, conversation, message).await?;
            }
            let settlement = SubmissionSettlement::Unanswered {
                reason: reason.to_owned(),
                detail,
            };
            end_failed_run(tx, conversation, task, failure, &settlement, now).await?;
            Ok(failed(text, reason))
        })
    }))
}

/// Fails `no_model`: the agent has no model, or no provider knows it.
pub(super) fn no_model(rt: &Rt, model: Option<&ModelRef>) -> Result<Next<GenerationTask>, TaskError> {
    let text = match model {
        None => "No model is configured".to_owned(),
        Some(model) => format!("Model {model} is not available"),
    };
    fail_run(rt, "no_model", text, None, None)
}

/// Fails `model_error` with `text` as the detail.
pub(super) fn model_error(
    rt: &Rt,
    text: String,
    message: Option<AssistantMessage>,
) -> Result<Next<GenerationTask>, TaskError> {
    let detail = Some(Json::from(text.clone()));
    fail_run(rt, "model_error", text, detail, message)
}
