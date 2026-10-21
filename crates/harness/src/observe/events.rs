//! Agent events (spec §9.4): what a UI reads instead of document ops, with upstream's
//! `AgentSessionEvent` names, serialized with a `type` tag.

use delta_ai::types::{AssistantBlock, AssistantMessage, JsonObject, Message, Usage};
use delta_chord::json::Json;
use serde::{Deserialize, Serialize};

use crate::agent::AgentState;
use crate::approval::ApprovalRequest;
use crate::compaction::{CompactionFailure, CompactionReason, CompactionStatus};
use crate::generation::{LiveGeneration, NestedToolSlot, RunControl, ToolSlot};
use crate::inbox::InboxItem;
use crate::records::{EntryRecord, SubmissionId, SubmissionRecord, TaskId};
use crate::tool::{NestedToolExecutionResult, ToolDiagnostic};
use crate::usage::UsageState;

/// The tool call an event is about.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolEventCall {
    pub tool_call_id: String,
    pub tool_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    /// For a nested call: the call that made it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_task_id: Option<TaskId>,
}

/// How a running call's output window changed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputChange {
    /// Remove `trim_start` UTF-16 code units from the front, then append `append`.
    Delta {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trim_start: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        append: Option<String>,
    },
    /// The whole window.
    Set { output: String },
}

/// One change of a streamed partial answer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageChange {
    TextStart {
        content_index: usize,
        block: AssistantBlock,
    },
    ThinkingStart {
        content_index: usize,
        block: AssistantBlock,
    },
    ToolcallStart {
        content_index: usize,
        block: AssistantBlock,
    },
    TextDelta {
        content_index: usize,
        delta: String,
    },
    ThinkingDelta {
        content_index: usize,
        delta: String,
    },
    /// Text appended to a string inside a tool call's arguments, at `path` below them.
    ToolcallDelta {
        content_index: usize,
        path: Vec<Json>,
        delta: String,
    },
    /// The block, whole, when it changed otherwise.
    Block {
        content_index: usize,
        block: AssistantBlock,
    },
    /// The message, whole, when anything else changed.
    Message {
        message: AssistantMessage,
    },
}

/// How a queued item waits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueuedMode {
    Steer,
    FollowUp,
    Write,
}

/// One queued item of `delta.inbox`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueuedItem {
    pub id: SubmissionId,
    pub mode: QueuedMode,
}

impl From<&InboxItem> for QueuedItem {
    fn from(item: &InboxItem) -> Self {
        let mode = match item {
            InboxItem::Steer { .. } => QueuedMode::Steer,
            InboxItem::FollowUp { .. } => QueuedMode::FollowUp,
            InboxItem::Write { .. } => QueuedMode::Write,
        };
        Self { id: item.id(), mode }
    }
}

/// The state every later event is relative to.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    /// The active transcript.
    pub entries: Vec<EntryRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<RunControl>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<LiveGeneration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSlot>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nested_tools: Option<Vec<NestedToolSlot>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compactions: Option<Vec<CompactionStatus>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction_failure: Option<CompactionFailure>,
    pub inbox: Vec<QueuedItem>,
    pub agent: AgentState,
    pub usage: UsageState,
    pub approvals: Vec<ApprovalRequest>,
}

/// One event; a commit yields a batch of them, in upstream's order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// The newest state, after a consumer fell behind.
    Snapshot(Box<Snapshot>),
    RunStart {
        inputs: Vec<SubmissionId>,
    },
    RunEnd {
        inputs: Vec<SubmissionId>,
    },
    TurnStart,
    TurnEnd,
    MessageStart {
        message: Message,
    },
    MessageUpdate {
        usage: Usage,
        changes: Vec<MessageChange>,
    },
    MessageEnd {
        entry: EntryRecord,
    },
    ToolExecutionStart {
        #[serde(flatten)]
        call: ToolEventCall,
        args: JsonObject,
    },
    ToolExecutionUpdate {
        #[serde(flatten)]
        call: ToolEventCall,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<OutputChange>,
        /// `Null` when removed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<Json>,
        /// `[]` when removed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        diagnostics: Option<Vec<ToolDiagnostic>>,
    },
    ToolExecutionEnd {
        #[serde(flatten)]
        call: ToolEventCall,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        entry: Option<EntryRecord>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<NestedToolExecutionResult>,
    },
    InboxUpdate {
        items: Vec<QueuedItem>,
    },
    Submission {
        record: SubmissionRecord,
    },
    AutoRetryStart {
        attempt: u32,
        at: i64,
        error_message: String,
    },
    AutoRetryEnd {
        attempt: u32,
    },
    DeferredPoll {
        poll_at: i64,
    },
    EntryAppended {
        entry: EntryRecord,
    },
    AgentChanged {
        agent: AgentState,
    },
    UsageChanged {
        usage: UsageState,
    },
    TaskFailed {
        task_id: TaskId,
        kind: String,
        message: String,
    },
    CompactionStart {
        task_id: TaskId,
        reason: CompactionReason,
        blocking: bool,
    },
    CompactionEnd {
        task_id: TaskId,
        reason: CompactionReason,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// delta: the conversation's approval requests waiting for a decision.
    ApprovalUpdate {
        requests: Vec<ApprovalRequest>,
    },
}

impl AgentEvent {
    /// The serialized `type` tag (`message_start`, `tool_execution_update`, ...).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Snapshot(_) => "snapshot",
            Self::RunStart { .. } => "run_start",
            Self::RunEnd { .. } => "run_end",
            Self::TurnStart => "turn_start",
            Self::TurnEnd => "turn_end",
            Self::MessageStart { .. } => "message_start",
            Self::MessageUpdate { .. } => "message_update",
            Self::MessageEnd { .. } => "message_end",
            Self::ToolExecutionStart { .. } => "tool_execution_start",
            Self::ToolExecutionUpdate { .. } => "tool_execution_update",
            Self::ToolExecutionEnd { .. } => "tool_execution_end",
            Self::InboxUpdate { .. } => "inbox_update",
            Self::Submission { .. } => "submission",
            Self::AutoRetryStart { .. } => "auto_retry_start",
            Self::AutoRetryEnd { .. } => "auto_retry_end",
            Self::DeferredPoll { .. } => "deferred_poll",
            Self::EntryAppended { .. } => "entry_appended",
            Self::AgentChanged { .. } => "agent_changed",
            Self::UsageChanged { .. } => "usage_changed",
            Self::TaskFailed { .. } => "task_failed",
            Self::CompactionStart { .. } => "compaction_start",
            Self::CompactionEnd { .. } => "compaction_end",
            Self::ApprovalUpdate { .. } => "approval_update",
        }
    }
}
