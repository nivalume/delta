//! [`CompactionTask`]: the `delta.compaction` task definition, its phases and placement.

use delta_ai::model::{CacheRetention, ModelThinkingLevel};
use delta_ai::retry::is_retryable_assistant_error;
use delta_ai::types::{Message, StopReason, SystemMessage, UserMessage};
use delta_chord::context::{Cancelled, Context};
use serde::{Deserialize, Serialize};

use super::select::{select_cut, summarized_messages};
use super::summary::{SUMMARIZATION_SYSTEM_PROMPT, summary_failure, summary_prompt, summary_text, wrap};
use super::{COMPACTION_ENTRY, COMPACTION_TASK, CompactionInput, CompactionReason, CompactionResult};
use crate::agent::{AgentRuntime, HookRuntime};
use crate::generation::{LiveRetry, edit_live, ensure_session_id};
use crate::inbox::{QueueModes, SubmissionDraft, SubmitError, admit_submission};
use crate::model::{ConversationStreamOptions, ModelRef, ModelRequest};
use crate::records::{ConversationId, EntryDraft, EntryId, Head, TaskId, TaskOutcomeError};
use crate::registry::{BeforeCompact, CompactDecision, CompactionHooks, TaskHooks};
use crate::scheduler::{Next, NextState, Outcome, TaskDefinition, TaskError, TaskRuntime};
use crate::session::{Tx, TxError};
use crate::usage::{UsageBucket, record_usage};

/// The pinned summarization request, fixed across retries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SummaryRequest {
    pub attempt: u32,
    pub model: ModelRef,
    pub thinking_level: ModelThinkingLevel,
    pub stream_options: ConversationStreamOptions,
    pub max_tokens: u64,
    /// The newest entry of the context the range was selected from.
    pub tail: EntryId,
    /// The first entry kept verbatim; the summary's `head`.
    pub first_kept: EntryId,
}

/// The phases of a `delta.compaction` task; the request is flattened, so a phase's JSON is
/// flat: `{"phase":"retry","attempt":2,...,"until":...}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "lowercase")]
pub enum CompactionCheckpoint {
    Select,
    Summarize {
        #[serde(flatten)]
        request: SummaryRequest,
    },
    Retry {
        #[serde(flatten)]
        request: SummaryRequest,
        until: i64,
    },
}

impl crate::scheduler::Checkpoint for CompactionCheckpoint {
    fn phase(&self) -> &str {
        match self {
            Self::Select => "select",
            Self::Summarize { .. } => "summarize",
            Self::Retry { .. } => "retry",
        }
    }
}

/// The `delta.compaction` task definition, version 1.
#[derive(Clone, Copy, Debug, Default)]
pub struct CompactionTask;

type Rt = TaskRuntime<CompactionTask>;

impl TaskDefinition for CompactionTask {
    type Input = CompactionInput;
    type Checkpoint = CompactionCheckpoint;
    type Result = CompactionResult;

    fn name(&self) -> &str {
        COMPACTION_TASK
    }

    fn version(&self) -> u32 {
        1
    }

    fn initial(&self, _input: &CompactionInput) -> CompactionCheckpoint {
        CompactionCheckpoint::Select
    }

    async fn run<'a>(
        &'a self,
        phase: CompactionCheckpoint,
        rt: &'a Rt,
        ctx: &'a Context,
    ) -> Result<Next<Self>, TaskError> {
        match phase {
            CompactionCheckpoint::Select => select(rt, ctx).await,
            CompactionCheckpoint::Summarize { request } => summarize(rt, ctx, request).await,
            CompactionCheckpoint::Retry { request, until } => Ok(retry(rt, request, until)),
        }
    }

    /// Removes the status and ends `aborted`, writing nothing else: a cut-off attempt has
    /// no committed response.
    async fn abort<'a>(
        &'a self,
        _phase: CompactionCheckpoint,
        rt: &'a Rt,
        _ctx: &'a Context,
    ) -> Result<Next<Self>, TaskError> {
        let conversation = rt.conversation_id();
        let task = rt.task_id();
        Ok(rt.next(move |tx, _| {
            Box::pin(async move {
                edit_live(tx, conversation, |live| live.remove_compaction(task)).await?;
                Ok(NextState::aborted(None))
            })
        }))
    }
}

impl TaskHooks for CompactionTask {
    type Hooks = dyn CompactionHooks;
}

/// `select`: the cut over the committed context; the `before_compact` chain; then the
/// pinned request.
async fn select(rt: &Rt, ctx: &Context) -> Result<Next<CompactionTask>, TaskError> {
    let agent = rt.agent(ctx).await?;
    let services = rt.agent_services()?;
    let settings = services.settings();
    let Some(model_ref) = agent.model.clone() else {
        return Ok(fail(rt, "no_model", "No model is configured".to_owned()));
    };
    let Some(model) = services.models().model(&model_ref) else {
        return Ok(fail(
            rt,
            "no_model",
            format!("Model {} is not available", model_ref.key()),
        ));
    };
    let policy = settings.compaction.policy_for(&model_ref);
    let view = rt.context_view(rt.conversation_id(), None, ctx).await?;
    let Some(cut) = select_cut(&view, policy.keep_recent_tokens) else {
        return Ok(complete(rt));
    };
    let first_kept = view.entries[cut].id;
    let input = rt.input().clone();
    let before = BeforeCompact {
        reason: input.reason,
        entries: view.entries[..cut].to_vec(),
        messages: summarized_messages(&view, cut),
        first_kept,
        instructions: input.instructions,
    };
    match before_compact(rt, &before, ctx).await? {
        Some(CompactDecision::Decline) => return Ok(complete(rt)),
        Some(CompactDecision::Summary(summary)) => return Ok(place(rt, first_kept, summary)),
        None => {}
    }
    let model_limit = if model.max_tokens > 0 {
        model.max_tokens
    } else {
        u64::MAX
    };
    let request = SummaryRequest {
        attempt: 1,
        model: model_ref,
        thinking_level: agent.thinking_level,
        stream_options: settings.stream,
        max_tokens: (policy.reserve_tokens * 4 / 5).min(model_limit),
        tail: view.entries.iter().map(|entry| entry.id).fold(first_kept, EntryId::max),
        first_kept,
    };
    Ok(Next::running(CompactionCheckpoint::Summarize { request }))
}

/// The `before_compact` chain: the first decision wins; an ordinary error is reported.
async fn before_compact(
    rt: &Rt,
    compaction: &BeforeCompact,
    ctx: &Context,
) -> Result<Option<CompactDecision>, TaskError> {
    let hooks = rt.hooks(ctx).await?;
    if hooks.is_empty() {
        return Ok(None);
    }
    let api = rt.hook_api()?;
    for (extension, hook) in hooks.iter() {
        match hook.before_compact(compaction, &api, ctx).await {
            Ok(Some(decision)) => return Ok(Some(decision)),
            Ok(None) => {}
            Err(error) => {
                if let Some(reason) = ctx.reason() {
                    return Err(Cancelled { reason }.into());
                }
                rt.report_hook(&*error, extension);
            }
        }
    }
    Ok(None)
}

/// `summarize`: one request over the range `select` chose (the context at `tail` is
/// immutable), classified in one commit that also records the spend.
async fn summarize(rt: &Rt, ctx: &Context, request: SummaryRequest) -> Result<Next<CompactionTask>, TaskError> {
    let services = rt.agent_services()?;
    let models = services.models().clone();
    let Some(model) = models.model(&request.model) else {
        return Ok(fail(
            rt,
            "no_model",
            format!("Model {} is not available", request.model.key()),
        ));
    };
    let conversation = rt.conversation_id();
    let view = rt.context_view(conversation, Some(request.tail), ctx).await?;
    let cut = view
        .entries
        .iter()
        .position(|entry| entry.id == request.first_kept)
        .ok_or_else(|| TaskError::new(format!("Entry {} is not in the summarized range", request.first_kept)))?;
    let now = rt.now();
    let prompt = summary_prompt(&summarized_messages(&view, cut), rt.input().instructions.as_deref());
    let messages = vec![
        Message::System(SystemMessage {
            content: SUMMARIZATION_SYSTEM_PROMPT.into(),
            timestamp: now,
            ..SystemMessage::default()
        }),
        Message::User(UserMessage {
            content: prompt.into(),
            timestamp: now,
        }),
    ];
    let mut options = request.stream_options.clone();
    options.deferred = None;
    options.cache_retention = Some(CacheRetention::None);
    let session_id = ensure_session_id(rt, ctx).await?;
    let model_request = ModelRequest {
        model: &model,
        messages,
        thinking_level: request.thinking_level,
        options: &options,
        session_id: Some(&session_id),
        max_tokens: Some(request.max_tokens),
    };
    let message = models.stream(model_request, ctx).result().await;
    // An abort mark or close: the abort handler or the reopened task handles what is committed.
    if let Some(reason) = ctx.reason() {
        return Err(Cancelled { reason }.into());
    }
    let summary = summary_text(&message);
    let policy = services.settings().retry;
    let retry = summary.is_none()
        && message.stop_reason == StopReason::Error
        && is_retryable_assistant_error(&message)
        && policy.enabled
        && request.attempt <= policy.max_retries;
    let until = rt
        .now()
        .saturating_add(i64::try_from(policy.delay(request.attempt).as_millis()).unwrap_or(i64::MAX));
    let failure = summary_failure(&message);
    let placement = Placement::of(rt, request.first_kept)?;
    let task = rt.task_id();
    Ok(rt.next(move |tx, _| {
        Box::pin(async move {
            let key = format!("{}/{}", message.provider, message.model);
            record_usage(tx, conversation, UsageBucket::Models, &key, &message.usage).await?;
            if let Some(summary) = summary {
                return placement.place(tx, summary).await;
            }
            if retry {
                let error = message.error_message.clone().unwrap_or_default();
                edit_live(tx, conversation, |live| {
                    if let Some(status) = live.compaction_mut(task) {
                        status.retry = Some(LiveRetry { at: until, error });
                    }
                })
                .await?;
                return Ok(NextState::Sleeping {
                    checkpoint: CompactionCheckpoint::Retry { request, until },
                    until,
                });
            }
            edit_live(tx, conversation, |live| live.fail_compaction(task, failure.clone())).await?;
            Ok(failed(failure, "model_error"))
        })
    }))
}

/// `retry`: sleeps again before `until`; then the next attempt of the same request.
fn retry(rt: &Rt, request: SummaryRequest, until: i64) -> Next<CompactionTask> {
    if until > rt.now() {
        return Next::sleep(until, CompactionCheckpoint::Retry { request, until });
    }
    let conversation = rt.conversation_id();
    let task = rt.task_id();
    let attempt = request.attempt + 1;
    rt.next(move |tx, _| {
        Box::pin(async move {
            edit_live(tx, conversation, |live| {
                if let Some(status) = live.compaction_mut(task) {
                    status.attempt = attempt;
                    status.retry = None;
                }
            })
            .await?;
            let request = SummaryRequest { attempt, ..request };
            Ok(NextState::Running(CompactionCheckpoint::Summarize { request }))
        })
    })
}

/// Where and how a summary lands.
struct Placement {
    conversation: ConversationId,
    task: TaskId,
    /// A blocking compaction, owned by the generation that waits for it.
    owned: bool,
    reason: CompactionReason,
    first_kept: EntryId,
    now: i64,
    modes: QueueModes,
}

impl Placement {
    fn of(rt: &Rt, first_kept: EntryId) -> Result<Self, TaskError> {
        Ok(Self {
            conversation: rt.conversation_id(),
            task: rt.task_id(),
            owned: rt.task().record.owner.is_some(),
            reason: rt.input().reason,
            first_kept,
            now: rt.now(),
            modes: QueueModes::from(&rt.agent_services()?.settings()),
        })
    }

    /// Places the summary entry, removes the status and the last failure, and completes. A
    /// blocking compaction appends (its generation holds the run); a conversation-owned one
    /// admits a write submission, placed at once when idle, at the next boundary when busy,
    /// or settled `stale` when its head left the range.
    async fn place(
        self,
        tx: &mut Tx<'_>,
        summary: String,
    ) -> Result<NextState<CompactionCheckpoint, CompactionResult>, TxError> {
        let entry = EntryDraft {
            head: Some(Head::At(self.first_kept)),
            model: Some(vec![Message::User(UserMessage {
                content: wrap(&summary).into(),
                timestamp: self.now,
            })]),
            data: Some(
                delta_chord::json!({ "reason": delta_chord::json::to_json(&self.reason).map_err(TxError::callback)? }),
            ),
            ..EntryDraft::new(COMPACTION_ENTRY)
        };
        let result = if self.owned {
            let entry = tx.append_entry(self.conversation, entry).await?;
            CompactionResult {
                entry_id: Some(entry.id),
                submission_id: None,
            }
        } else {
            let draft = SubmissionDraft::write(entry).request_id(format!("compaction:{}", self.task));
            let admission = admit_submission(tx, self.conversation, draft, self.now, self.modes)
                .await
                .map_err(SubmitError::into_tx)?;
            CompactionResult {
                entry_id: None,
                submission_id: Some(admission.id),
            }
        };
        let task = self.task;
        edit_live(tx, self.conversation, |live| {
            live.remove_compaction(task);
            live.compaction_failure = None;
        })
        .await?;
        Ok(NextState::completed(result))
    }
}

/// Places a summary a hook supplied.
fn place(rt: &Rt, first_kept: EntryId, summary: String) -> Next<CompactionTask> {
    match Placement::of(rt, first_kept) {
        Ok(placement) => rt.next(move |tx, _| Box::pin(placement.place(tx, summary))),
        Err(error) => Next::fail(error.to_string()),
    }
}

/// Removes the status and completes without a summary.
fn complete(rt: &Rt) -> Next<CompactionTask> {
    let conversation = rt.conversation_id();
    let task = rt.task_id();
    rt.next(move |tx, _| {
        Box::pin(async move {
            edit_live(tx, conversation, |live| live.remove_compaction(task)).await?;
            Ok(NextState::completed(CompactionResult::default()))
        })
    })
}

/// Removes the status, records the failure and fails with `{ reason }` as the detail.
fn fail(rt: &Rt, reason: &'static str, message: String) -> Next<CompactionTask> {
    let conversation = rt.conversation_id();
    let task = rt.task_id();
    rt.next(move |tx, _| {
        Box::pin(async move {
            edit_live(tx, conversation, |live| live.fail_compaction(task, message.clone())).await?;
            Ok(failed(message, reason))
        })
    })
}

fn failed(message: String, reason: &str) -> NextState<CompactionCheckpoint, CompactionResult> {
    NextState::Terminal(Outcome::Failed {
        error: TaskOutcomeError {
            message,
            detail: Some(delta_chord::json!({ "reason": reason })),
        },
        result: None,
    })
}
