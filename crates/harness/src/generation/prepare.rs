//! The `prepare` phase: resolve the agent once, render the prompt, plan the positional
//! `delta.system` entries, check the compaction thresholds, then pin the request.

use delta_ai::types::{Message, Tool};
use delta_chord::context::Context;

use super::failure::{model_error, no_model};
use super::live::live_state;
use super::{GenerationCheckpoint, GenerationTask, Rt};
use crate::agent::AgentRuntime;
use crate::agent::report_host;
use crate::compaction::{CompactionInput, CompactionReason, Threshold, threshold};
use crate::prompt::{PromptInput, plan_system_entries, render_sections, replay_sections};
use crate::records::{EntryQuery, JoinPolicy, ScanOrder, TaskId, TaskOutcome};
use crate::scheduler::{Next, NextState, ReportFrom, TaskError};
use crate::session::{HostCallback, TxError};

/// Error text for a blocking compaction that ended without completing.
fn compaction_failure(outcome: &TaskOutcome) -> Option<String> {
    match outcome {
        TaskOutcome::Completed { .. } => None,
        TaskOutcome::Failed { error, .. } | TaskOutcome::Faulted { error } => {
            Some(format!("Compaction failed: {}", error.message))
        }
        TaskOutcome::Orphaned { reason } => Some(format!("Compaction failed: {reason}")),
        TaskOutcome::Aborted { .. } => Some("Compaction was aborted".to_owned()),
    }
}

/// Whether a completed compaction produced a summary entry.
fn summarized(outcome: &TaskOutcome) -> bool {
    match outcome {
        TaskOutcome::Completed { result } => result.get("entry_id").is_some_and(|entry| !entry.is_null()),
        _ => false,
    }
}

pub(super) async fn run(
    rt: &Rt,
    ctx: &Context,
    attempt: u32,
    compacted: Option<TaskId>,
    overflow: Option<String>,
    blocked: bool,
) -> Result<Next<GenerationTask>, TaskError> {
    let conversation = rt.conversation_id();
    let task = rt.task_id();
    let agent = rt.agent(ctx).await?;
    let services = rt.agent_services()?;
    let settings = services.settings();
    let Some(model_ref) = agent.model.clone() else {
        return no_model(rt, None);
    };
    let Some(model) = services.models().model(&model_ref) else {
        return no_model(rt, Some(&model_ref));
    };
    if let Some(compaction) = compacted {
        let outcome = rt.outcomes(&[compaction], ctx).await?.into_iter().next();
        if let Some(text) = overflow
            && !outcome.as_ref().is_some_and(summarized)
        {
            return model_error(rt, text, None);
        }
        // A threshold compaction that found nothing to compact leaves a request that may
        // still fit; one that failed left a context it judged too large.
        if blocked && let Some(text) = outcome.as_ref().and_then(compaction_failure) {
            return model_error(rt, text, None);
        }
    }
    let view = rt.context_view(conversation, None, ctx).await?;
    let shown = replay_sections(&view.messages);
    let env = match rt.env(ctx).await {
        Ok(env) => env,
        Err(error) => {
            if let Some(reason) = ctx.reason() {
                return Err(delta_chord::context::Cancelled { reason }.into());
            }
            report_host(rt, &error, HostCallback::Env);
            None
        }
    };
    let input = PromptInput {
        conversation,
        agent: &agent,
        env: env.as_deref(),
        shown: &shown,
    };
    let desired = render_sections(
        &agent.sections,
        &input,
        |error, section| {
            let extension = agent.section_extension(section.key()).map(str::to_owned);
            rt.report(&**error, Some(ReportFrom::Extension(extension)));
        },
        ctx,
    )
    .await?;
    let tools: Vec<Tool> = agent.tools.iter().map(|tool| tool.definition().clone()).collect();
    let entries = plan_system_entries(&view, &desired, &tools, rt.now());
    let policy = settings.compaction.policy_for(&model_ref);
    let crossed = if compacted.is_none() {
        let planned: Vec<Message> = entries
            .iter()
            .flat_map(|entry| entry.model.clone().unwrap_or_default())
            .collect();
        threshold(&view, &planned, model.context_window, &policy, &**services.compactor())
    } else {
        None
    };
    let compactor = services.compactor().clone();
    if crossed == Some(Threshold::Blocking) {
        // Compact first and prepare again; the transcript is unchanged until it appends.
        return Ok(rt.next(move |tx, _| {
            Box::pin(async move {
                let input = CompactionInput {
                    reason: CompactionReason::Threshold,
                    instructions: None,
                };
                let child = compactor.create(tx, conversation, input, Some(task)).await?;
                Ok(NextState::Waiting {
                    checkpoint: GenerationCheckpoint::Prepare {
                        attempt,
                        compacted: Some(child),
                        overflow: None,
                        threshold: Some(true),
                    },
                    on: vec![child],
                    policy: JoinPolicy::AllSettled,
                })
            })
        }));
    }
    let thinking_level = agent.thinking_level;
    let stream_options = settings.stream;
    Ok(rt.next(move |tx, _| {
        Box::pin(async move {
            let newest = EntryQuery {
                order: Some(ScanOrder::Descending),
                ..EntryQuery::new(conversation)
            };
            let mut cutoff = tx
                .scan_entries(newest, 1, None)
                .await?
                .items
                .first()
                .map(|entry| entry.id);
            for entry in entries {
                cutoff = Some(tx.append_entry(conversation, entry).await?.id);
            }
            let cutoff = cutoff
                .ok_or_else(|| TxError::callback(format!("Conversation {conversation} has no entries to send")))?;
            // Checked in this commit, so a compaction admitted during preparation counts.
            if crossed == Some(Threshold::Background) && live_state(tx, conversation).await?.compactions.is_none() {
                let input = CompactionInput {
                    reason: CompactionReason::Threshold,
                    instructions: None,
                };
                compactor.create(tx, conversation, input, None).await?;
            }
            Ok(NextState::Running(GenerationCheckpoint::Request {
                attempt,
                compacted,
                model: model_ref,
                thinking_level,
                stream_options,
                cutoff,
            }))
        })
    }))
}
