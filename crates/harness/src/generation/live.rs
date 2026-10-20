//! `delta.live`: run control and the presentation of the current generation and tool round
//! (spec §8.2), with the run transitions every terminal path shares.

use delta_ai::types::{AssistantMessage, Message, StopReason};
use delta_chord::delta::Op;
use delta_chord::json::{self, Json};
use serde::{Deserialize, Serialize};

use super::{ASSISTANT_ENTRY, GenerationInput, GenerationTask};
use crate::documents::{CheckpointInfo, ConversationScope, DocAddress, DocDefinition, ForkInitial, Latest};
use crate::records::{ConversationId, EntryDraft, EntryId, EntryRecord, SubmissionId, SubmissionSettlement, TaskId};
use crate::session::{TaskOptions, Tx, TxError};
use crate::tool::ToolDiagnostic;
use crate::usage::{UsageBucket, record_usage};

/// The `delta.live` document: conversation scope, `latest`, a fork starts empty, a complete
/// base whenever nothing runs (no generation and no running tool or nested slot), so every
/// base is small and a delta chain spans at most one generation or one round of tools.
#[derive(Clone, Copy, Debug, Default)]
pub struct LiveDoc;

impl DocDefinition for LiveDoc {
    const KIND: &'static str = "delta.live";
    const VERSION: u32 = 1;
    type Scope = ConversationScope<Latest, ForkInitial>;
    type Seed = ();

    fn initial((): &()) -> Json {
        delta_chord::json!({})
    }

    fn checkpoint_when(value: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        let running = |field: &str| {
            value.get(field).and_then(Json::as_array).is_some_and(|slots| {
                slots
                    .iter()
                    .any(|slot| slot.get("status").and_then(Json::as_str) == Some("running"))
            })
        };
        value.get("generation").is_none() && !running("tools") && !running("nested_tools")
    }
}

/// Run control: the task that settles the run's inputs, and those inputs. Present exactly
/// while the conversation is busy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunControl {
    pub task_id: TaskId,
    pub inputs: Vec<SubmissionId>,
}

/// A durable backoff before the next attempt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveRetry {
    pub at: i64,
    pub error: String,
}

/// A provider-side deferred response being polled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveDeferred {
    pub poll_at: i64,
}

/// The presentation of the current generation attempt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LiveGeneration {
    pub attempt: u32,
    /// The committed, throttled partial of the response in flight (an `AssistantMessage`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<Json>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry: Option<LiveRetry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deferred: Option<LiveDeferred>,
}

impl LiveGeneration {
    pub fn new(attempt: u32) -> Self {
        Self {
            attempt,
            message: None,
            retry: None,
            deferred: None,
        }
    }
}

/// Where a tool call of the round stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotStatus {
    Pending,
    Running,
    Done,
}

/// The presentation of one tool call of the current round.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSlot {
    pub call_id: String,
    pub name: String,
    /// Absent for a call not started yet (sequential round) and for a call the request did
    /// not offer, which starts `done`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    pub status: SlotStatus,
    /// Retained running output; only while `running`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dropped_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dropped_lines: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Json>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<ToolDiagnostic>>,
    /// The result entry once done; absent when the tool task faulted or was orphaned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry: Option<EntryId>,
}

impl ToolSlot {
    /// A slot of call `call_id` in status `status`.
    pub fn new(call_id: &str, name: &str, status: SlotStatus) -> Self {
        Self {
            call_id: call_id.to_owned(),
            name: name.to_owned(),
            task_id: None,
            status,
            output: None,
            dropped_bytes: None,
            dropped_lines: None,
            details: None,
            diagnostics: None,
            entry: None,
        }
    }

    /// Done, without the progress it published while running.
    pub fn finish(&mut self) {
        self.status = SlotStatus::Done;
        self.output = None;
        self.dropped_bytes = None;
        self.dropped_lines = None;
        self.details = None;
        self.diagnostics = None;
    }
}

/// The value of a `delta.live` document. The nested-call and compaction fields are carried
/// as JSON; their owners (the tool and compaction tasks) type them.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LiveState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<RunControl>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<LiveGeneration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSlot>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nested_tools: Option<Vec<Json>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compactions: Option<Vec<Json>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction_failure: Option<Json>,
}

/// The `delta.live` of `conversation`, as of this transaction.
pub async fn live_state(tx: &mut Tx<'_>, conversation: ConversationId) -> Result<LiveState, TxError> {
    let doc = tx.doc(DocAddress::<LiveDoc>::new(conversation)).await?;
    json::from_json(doc.value()).map_err(TxError::callback)
}

/// Edits `conversation`'s `delta.live` as a typed state; the diff is the change.
pub async fn edit_live<R>(
    tx: &mut Tx<'_>,
    conversation: ConversationId,
    edit: impl FnOnce(&mut LiveState) -> R,
) -> Result<R, TxError> {
    let mut doc = tx.doc(DocAddress::<LiveDoc>::new(conversation)).await?;
    doc.update_as::<LiveState, R>(edit).map_err(TxError::callback)
}

/// A generation owned by its conversation.
pub async fn create_generation(tx: &mut Tx<'_>, conversation: ConversationId) -> Result<TaskId, TxError> {
    let task = tx
        .create_task(
            &GenerationTask,
            GenerationInput::default(),
            TaskOptions::conversation(conversation),
        )
        .await?;
    Ok(task.id())
}

/// Starts a run for the placed inputs `inputs`: a new generation takes `delta.live.run`.
pub async fn start_run(
    tx: &mut Tx<'_>,
    conversation: ConversationId,
    inputs: Vec<SubmissionId>,
) -> Result<TaskId, TxError> {
    let task_id = create_generation(tx, conversation).await?;
    edit_live(tx, conversation, |live| live.run = Some(RunControl { task_id, inputs })).await?;
    Ok(task_id)
}

/// Ends the run `task` owns: settles each of its inputs and removes `run`. Always removes
/// `generation`, `tools` and `nested_tools`, whose presentation belongs to the ending run.
pub async fn end_run(
    tx: &mut Tx<'_>,
    conversation: ConversationId,
    task: TaskId,
    settlement: &SubmissionSettlement,
) -> Result<(), TxError> {
    let inputs = edit_live(tx, conversation, |live| {
        let inputs = match &live.run {
            Some(run) if run.task_id == task => live.run.take().map(|run| run.inputs).unwrap_or_default(),
            _ => Vec::new(),
        };
        live.generation = None;
        live.tools = None;
        live.nested_tools = None;
        inputs
    })
    .await?;
    for input in inputs {
        tx.settle_submission(input, settlement.clone());
    }
    Ok(())
}

/// Appends a provider message as a `delta.assistant` entry and adds its usage to
/// `delta.usage.models[provider/model]` in the same commit. Every writer of assistant
/// entries goes through here, so the ledger stays complete.
pub async fn append_assistant(
    tx: &mut Tx<'_>,
    conversation: ConversationId,
    message: AssistantMessage,
) -> Result<EntryRecord, TxError> {
    let key = format!("{}/{}", message.provider, message.model);
    record_usage(tx, conversation, UsageBucket::Models, &key, &message.usage).await?;
    let draft = EntryDraft {
        model: Some(vec![Message::Assistant(message)]),
        ..EntryDraft::new(ASSISTANT_ENTRY)
    };
    tx.append_entry(conversation, draft).await
}

/// Appends a committed partial left by an interrupted, aborted, faulted or orphaned attempt
/// as an aborted `delta.assistant` entry; the caller replaces or removes `generation`.
pub async fn convert_partial(tx: &mut Tx<'_>, conversation: ConversationId) -> Result<(), TxError> {
    let live = live_state(tx, conversation).await?;
    let Some(partial) = live.generation.and_then(|generation| generation.message) else {
        return Ok(());
    };
    let mut message: AssistantMessage = json::from_json(&partial).map_err(TxError::callback)?;
    message.stop_reason = StopReason::Aborted;
    append_assistant(tx, conversation, message).await?;
    Ok(())
}
