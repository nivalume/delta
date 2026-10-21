//! Agent events to protocol messages. A turn opens at the entry of an input placed while no
//! turn is open, takes the inputs placed before its first input's answer (steering), and
//! closes at that answer or when the first input settles otherwise. Items follow the
//! streamed partial, the tool slots and compactions; pending approvals become requests.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use delta_ai::types::{AssistantBlock, AssistantMessage, Message, Usage};
use delta_harness::approval::{ApprovalId, ApprovalRequest};
use delta_harness::generation::{ASSISTANT_ENTRY, SlotStatus};
use delta_harness::observe::{AgentEvent, MessageChange, OutputChange, Snapshot, ToolEventCall};
use delta_harness::records::{
    EntryId, EntryRecord, SubmissionId, SubmissionRecord, SubmissionState, SubmissionType, TaskId,
};
use delta_harness::tool::TOOL_RESULT_ENTRY;
use delta_harness::usage::UsageState;
use delta_protocol::delta::ToolRequestApprovalParams;
use delta_protocol::v2::{
    CommandExecutionRequestApprovalParams, ErrorNotification, ExecPolicyAmendment, FileChangeRequestApprovalParams,
    ItemCompletedNotification, ItemDeltaNotification, ItemStartedNotification, ReasoningTextDeltaNotification,
    ThreadActiveFlag, ThreadItem, ThreadStatus, ThreadStatusChangedNotification, ThreadTokenUsage,
    ThreadTokenUsageUpdatedNotification, TokenUsageBreakdown, Turn, TurnError, TurnNotification, TurnStatus,
};
use delta_protocol::{RequestId, ServerNotification, ServerRequest};
use serde_json::Value;

use crate::items::{ApprovalKind, Call, Outcome, block_id, tool_item, user_message};

fn notify(out: &mut Vec<Output>, notification: ServerNotification) {
    out.push(Output::Notify(notification));
}

pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX))
}

/// What the translator asks the connection to send.
#[derive(Debug)]
pub(crate) enum Output {
    Notify(ServerNotification),
    /// A pending approval, to send as a server request.
    Ask(Ask),
    /// An approval no longer pending.
    Resolved(ApprovalId),
}

/// The params of an approval request, without its request id.
#[derive(Debug)]
pub(crate) enum Ask {
    CommandExecution(ApprovalId, CommandExecutionRequestApprovalParams),
    FileChange(ApprovalId, FileChangeRequestApprovalParams),
    Tool(ApprovalId, ToolRequestApprovalParams),
}

impl Ask {
    pub(crate) fn approval(&self) -> ApprovalId {
        match self {
            Self::CommandExecution(id, _) | Self::FileChange(id, _) | Self::Tool(id, _) => *id,
        }
    }

    pub(crate) fn kind(&self) -> ApprovalKind {
        match self {
            Self::CommandExecution(..) => ApprovalKind::CommandExecution,
            Self::FileChange(..) => ApprovalKind::FileChange,
            Self::Tool(..) => ApprovalKind::Tool,
        }
    }

    /// The server request with id `request_id`.
    pub(crate) fn into_request(self, request_id: RequestId) -> ServerRequest {
        match self {
            Self::CommandExecution(_, params) => ServerRequest::CommandExecutionRequestApproval { request_id, params },
            Self::FileChange(_, params) => ServerRequest::FileChangeRequestApproval { request_id, params },
            Self::Tool(_, params) => ServerRequest::ToolRequestApproval { request_id, params },
        }
    }
}

/// The command prefix of a proposed `Bash(<prefix>:*)` rule.
fn amendment(rule: Option<&str>) -> Option<ExecPolicyAmendment> {
    let prefix = rule?.strip_prefix("Bash(")?.strip_suffix(":*)")?;
    let command: Vec<String> = prefix.split_whitespace().map(str::to_owned).collect();
    (!command.is_empty()).then_some(ExecPolicyAmendment { command })
}

/// The settled status of a turn whose first input has `record`.
fn turn_status(record: Option<&SubmissionRecord>) -> (TurnStatus, Option<TurnError>) {
    match record.map(|record| &record.state) {
        Some(SubmissionState::Unanswered { reason, detail, .. }) => {
            if reason == "aborted" {
                return (TurnStatus::Interrupted, None);
            }
            let error = TurnError {
                message: reason.clone(),
                codex_error_info: Some(Value::from("other")),
                additional_details: detail.as_ref().map(ToString::to_string),
            };
            (TurnStatus::Failed, Some(error))
        }
        _ => (TurnStatus::Completed, None),
    }
}

/// The entry an input submission was placed as.
pub(crate) fn input_entry(record: &SubmissionRecord) -> Option<EntryId> {
    if record.kind != SubmissionType::Input {
        return None;
    }
    match &record.state {
        SubmissionState::Placed { entry } | SubmissionState::Done { entry, .. } => Some(*entry),
        SubmissionState::Unanswered { entry, .. } => *entry,
        SubmissionState::Queued => None,
    }
}

fn breakdown(usage: &Usage) -> TokenUsageBreakdown {
    let tokens = |value: u64| i64::try_from(value).unwrap_or(i64::MAX);
    TokenUsageBreakdown {
        total_tokens: tokens(usage.total_tokens),
        input_tokens: tokens(usage.input + usage.cache_read + usage.cache_write),
        cached_input_tokens: tokens(usage.cache_read),
        cache_write_input_tokens: tokens(usage.cache_write),
        output_tokens: tokens(usage.output),
        reasoning_output_tokens: tokens(usage.reasoning.unwrap_or(0)),
    }
}

fn add(total: &mut Usage, usage: &Usage) {
    total.input += usage.input;
    total.output += usage.output;
    total.cache_read += usage.cache_read;
    total.cache_write += usage.cache_write;
    total.reasoning = Some(total.reasoning.unwrap_or(0) + usage.reasoning.unwrap_or(0));
    total.total_tokens += usage.total_tokens;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockKind {
    Text,
    Thinking,
    Other,
}

/// One block of the streamed partial and what its item has been sent.
#[derive(Clone, Debug)]
struct PartialBlock {
    kind: BlockKind,
    text: String,
    /// `None` until the item started.
    sent: Option<String>,
}

impl PartialBlock {
    fn of(block: &AssistantBlock) -> Self {
        let (kind, text) = match block {
            AssistantBlock::Text(text) => (BlockKind::Text, text.text.clone()),
            AssistantBlock::Thinking(thinking) => (BlockKind::Thinking, thinking.thinking.clone()),
            AssistantBlock::ToolCall(_) => (BlockKind::Other, String::new()),
        };
        Self { kind, text, sent: None }
    }
}

/// The streamed answer of a generation task.
#[derive(Clone, Debug)]
struct Partial {
    task: Option<TaskId>,
    blocks: Vec<PartialBlock>,
}

impl Partial {
    fn new(task: Option<TaskId>, content: &[AssistantBlock]) -> Self {
        Self {
            task,
            blocks: content.iter().map(PartialBlock::of).collect(),
        }
    }

    fn set(&mut self, index: usize, block: &AssistantBlock) {
        let new = PartialBlock::of(block);
        if let Some(current) = self.blocks.get_mut(index) {
            current.kind = new.kind;
            current.text = new.text;
            return;
        }
        while self.blocks.len() < index {
            self.blocks.push(PartialBlock {
                kind: BlockKind::Other,
                text: String::new(),
                sent: None,
            });
        }
        self.blocks.push(new);
    }

    fn apply(&mut self, change: &MessageChange) {
        match change {
            MessageChange::TextStart { content_index, block }
            | MessageChange::ThinkingStart { content_index, block }
            | MessageChange::ToolcallStart { content_index, block }
            | MessageChange::Block { content_index, block } => self.set(*content_index, block),
            MessageChange::TextDelta { content_index, delta }
            | MessageChange::ThinkingDelta { content_index, delta } => {
                if let Some(block) = self.blocks.get_mut(*content_index) {
                    block.text.push_str(delta);
                }
            }
            MessageChange::ToolcallDelta { .. } => {}
            MessageChange::Message { message } => {
                for (index, block) in message.content.iter().enumerate() {
                    self.set(index, block);
                }
                self.blocks.truncate(message.content.len());
            }
        }
    }
}

/// The turn in progress.
#[derive(Clone, Debug)]
struct OpenTurn {
    first: SubmissionId,
    inputs: Vec<SubmissionId>,
    started_at_ms: i64,
}

/// One thread's translation state.
pub(crate) struct Translator {
    thread_id: String,
    cwd: String,
    turn: Option<OpenTurn>,
    last_turn: Option<String>,
    submissions: HashMap<SubmissionId, SubmissionRecord>,
    partial: Option<Partial>,
    /// Tool calls of the transcript's assistant messages.
    calls: HashMap<String, Call>,
    /// Tool items started and not completed.
    started: HashMap<String, Call>,
    /// The output deltas sent for each started command.
    streamed: HashMap<String, String>,
    approvals: BTreeMap<ApprovalId, ApprovalRequest>,
    compactions: HashSet<TaskId>,
    last_usage: Usage,
    run_active: bool,
    status: ThreadStatus,
    /// The time items and turns are stamped with; the wall clock when `None`.
    clock: Option<i64>,
    /// Inputs this connection started turns with (`turn/start`), until placed.
    requested: HashSet<SubmissionId>,
    /// The newest entry seen.
    last_entry: Option<EntryId>,
}

impl Translator {
    /// A thread without events yet.
    pub(crate) fn new(thread_id: String, cwd: String) -> Self {
        Self {
            thread_id,
            cwd,
            turn: None,
            last_turn: None,
            submissions: HashMap::new(),
            partial: None,
            calls: HashMap::new(),
            started: HashMap::new(),
            streamed: HashMap::new(),
            approvals: BTreeMap::new(),
            compactions: HashSet::new(),
            last_usage: Usage::default(),
            run_active: false,
            status: ThreadStatus::Idle,
            clock: None,
            requested: HashSet::new(),
            last_entry: None,
        }
    }

    /// Input `id` started a turn (`turn/start`): if it is withdrawn before it is placed,
    /// its turn starts and completes then.
    pub(crate) fn requested(&mut self, id: SubmissionId) {
        self.requested.insert(id);
    }

    /// Known submissions, so entries placed for them read as inputs.
    pub(crate) fn know(&mut self, records: impl IntoIterator<Item = SubmissionRecord>) {
        for record in records {
            self.submissions.insert(record.id, record);
        }
    }

    /// Stamps what follows with `at` (epoch ms) instead of the wall clock.
    pub(crate) fn set_clock(&mut self, at: Option<i64>) {
        self.clock = at;
    }

    fn now(&self) -> i64 {
        self.clock.unwrap_or_else(now_ms)
    }

    /// The state of a thread whose stream attached with `snapshot`, `records` holding the
    /// submissions of its run; returns the items and requests in flight.
    pub(crate) fn attach(
        thread_id: String,
        cwd: String,
        snapshot: &Snapshot,
        records: Vec<SubmissionRecord>,
    ) -> (Self, Vec<Output>) {
        let mut translator = Self::new(thread_id, cwd);
        let mut out = Vec::new();
        translator.sync(snapshot, records, &mut out);
        translator.status = translator.current_status();
        (translator, out)
    }

    /// The status the thread has now.
    pub(crate) fn status(&self) -> ThreadStatus {
        self.status.clone()
    }

    /// The turn in progress.
    pub(crate) fn current_turn(&self) -> Option<String> {
        self.turn.as_ref().map(|turn| turn.first.to_string())
    }

    /// The pending approvals known to the client.
    pub(crate) fn pending(&self) -> impl Iterator<Item = &ApprovalRequest> {
        self.approvals.values()
    }

    /// After a lag: the newest state, with `records` holding the submissions of its run.
    pub(crate) fn resync(&mut self, snapshot: &Snapshot, records: Vec<SubmissionRecord>) -> Vec<Output> {
        let mut out = Vec::new();
        self.sync(snapshot, records, &mut out);
        self.update_status(&mut out);
        out
    }

    /// One commit's events.
    pub(crate) fn batch(&mut self, events: &[AgentEvent]) -> Vec<Output> {
        let mut settled_now = HashSet::new();
        for event in events {
            if let AgentEvent::Submission { record } = event {
                self.submissions.insert(record.id, record.clone());
                settled_now.insert(record.id);
            }
        }
        let mut out = Vec::new();
        for event in events {
            self.event(event, &mut out);
        }
        // After the batch's items and usage: the turn whose answer arrived, or whose first
        // input settled in this batch.
        let close = self.turn.as_ref().is_some_and(|turn| {
            match self.submissions.get(&turn.first).map(|record| &record.state) {
                Some(SubmissionState::Done {
                    answer: Some(answer), ..
                }) if self.last_entry >= Some(*answer) => true,
                Some(SubmissionState::Done { .. } | SubmissionState::Unanswered { .. }) => {
                    settled_now.contains(&turn.first)
                }
                _ => false,
            }
        });
        if close {
            self.close_turn(&mut out);
        }
        self.update_status(&mut out);
        out
    }

    /// The end of a replay: closes the open turn if its first input settled.
    pub(crate) fn finish(&mut self) -> Vec<Output> {
        let mut out = Vec::new();
        let settled = self.turn.as_ref().is_some_and(|turn| {
            matches!(
                self.submissions.get(&turn.first).map(|record| &record.state),
                Some(SubmissionState::Done { .. } | SubmissionState::Unanswered { .. })
            )
        });
        if settled {
            self.close_turn(&mut out);
        }
        out
    }

    fn sync(&mut self, snapshot: &Snapshot, records: Vec<SubmissionRecord>, out: &mut Vec<Output>) {
        for record in records {
            self.submissions.insert(record.id, record);
        }
        self.run_active = snapshot.run.is_some();
        for entry in &snapshot.entries {
            if let Some([Message::Assistant(message), ..]) = entry.model.as_deref() {
                self.record_calls(message);
                self.last_usage = message.usage.clone();
            }
        }
        // The open turn: the run's first input still waiting for its answer.
        let open = snapshot.run.as_ref().and_then(|run| {
            run.inputs.iter().copied().find(|id| {
                matches!(
                    self.submissions.get(id).map(|record| &record.state),
                    Some(SubmissionState::Placed { .. })
                )
            })
        });
        let current = self.turn.as_ref().map(|turn| turn.first);
        // A turn the client learns of from the thread's turns or the snapshot: no
        // `turn/started`. One that ended closes after its items below.
        if current.is_none()
            && let Some(first) = open
        {
            self.resume_turn(first);
        }
        // The streamed partial.
        let partial = snapshot
            .generation
            .as_ref()
            .and_then(|generation| generation.message.as_ref())
            .and_then(|message| delta_chord::json::from_json::<AssistantMessage>(message).ok());
        let task = snapshot.run.as_ref().map(|run| run.task_id);
        match (partial, self.partial.as_mut()) {
            (Some(message), Some(current)) if current.task == task => {
                current.apply(&MessageChange::Message { message });
            }
            (Some(message), _) => self.partial = Some(Partial::new(task, &message.content)),
            (None, _) => self.partial = None,
        }
        self.sync_partial(out);
        // Tool calls: start the running ones, complete the ones that ended meanwhile.
        let slots = snapshot.tools.iter().flatten();
        let running: Vec<(String, String)> = slots
            .clone()
            .filter(|slot| slot.status == SlotStatus::Running)
            .map(|slot| (slot.call_id.clone(), slot.name.clone()))
            .collect();
        let active: HashSet<&str> = slots
            .filter(|slot| slot.status != SlotStatus::Done)
            .map(|slot| slot.call_id.as_str())
            .chain(
                snapshot
                    .approvals
                    .iter()
                    .filter(|request| request.decision.is_none())
                    .map(|request| request.call_id.as_str()),
            )
            .collect();
        let ended: Vec<String> = self
            .started
            .keys()
            .filter(|id| !active.contains(id.as_str()))
            .cloned()
            .collect();
        for id in ended {
            let entry = snapshot.entries.iter().find(|entry| {
                entry.kind == TOOL_RESULT_ENTRY
                    && Outcome::of(entry).is_some_and(|outcome| outcome.message.tool_call_id == id)
            });
            if let Some(call) = self.started.remove(&id) {
                self.complete_tool(&call, entry, out);
            }
        }
        for (id, name) in running {
            let call = self.calls.get(&id).cloned().unwrap_or(Call {
                id,
                name,
                arguments: delta_ai::types::JsonObject::default(),
            });
            self.start_tool(call, out);
        }
        // Compactions.
        let compacting: HashSet<TaskId> = snapshot
            .compactions
            .iter()
            .flatten()
            .map(|status| status.task_id)
            .collect();
        for task in self.compactions.clone() {
            if !compacting.contains(&task) {
                self.end_compaction(task, out);
            }
        }
        for task in compacting {
            self.start_compaction(task, out);
        }
        self.approvals_changed(&snapshot.approvals, out);
        if current.is_some() && current != open {
            self.close_turn(out);
            if let Some(first) = open {
                self.resume_turn(first);
            }
        }
    }

    /// A turn already in progress.
    fn resume_turn(&mut self, first: SubmissionId) {
        self.turn = Some(OpenTurn {
            first,
            inputs: vec![first],
            started_at_ms: self.now(),
        });
        self.last_turn = Some(first.to_string());
    }

    fn event(&mut self, event: &AgentEvent, out: &mut Vec<Output>) {
        match event {
            AgentEvent::RunStart { .. } => self.run_active = true,
            AgentEvent::RunEnd { .. } => self.run_active = false,
            AgentEvent::MessageStart {
                message: Message::Assistant(message),
                task_id,
            } => {
                self.partial = Some(Partial::new(*task_id, &message.content));
                self.sync_partial(out);
            }
            AgentEvent::MessageUpdate { changes, .. } => {
                if let Some(partial) = &mut self.partial {
                    for change in changes {
                        partial.apply(change);
                    }
                }
                self.sync_partial(out);
            }
            AgentEvent::MessageEnd { entry } => self.entry(entry, out),
            AgentEvent::ToolExecutionStart { call, args } if call.parent_tool_call_id.is_none() => {
                let call = Call {
                    id: call.tool_call_id.clone(),
                    name: call.tool_name.clone(),
                    arguments: args.clone(),
                };
                self.calls.insert(call.id.clone(), call.clone());
                self.start_tool(call, out);
            }
            AgentEvent::ToolExecutionUpdate {
                call,
                output: Some(OutputChange::Delta {
                    append: Some(append), ..
                }),
                ..
            } if call.parent_tool_call_id.is_none() && call.tool_name == "bash" => {
                if let (Some(turn_id), true) = (self.turn_id(), self.started.contains_key(&call.tool_call_id)) {
                    self.streamed
                        .entry(call.tool_call_id.clone())
                        .or_default()
                        .push_str(append);
                    out.push(Output::Notify(ServerNotification::CommandExecutionOutputDelta(
                        ItemDeltaNotification {
                            thread_id: self.thread_id.clone(),
                            turn_id,
                            item_id: call.tool_call_id.clone(),
                            delta: append.clone(),
                        },
                    )));
                }
            }
            AgentEvent::ToolExecutionEnd { call, entry, .. } if call.parent_tool_call_id.is_none() => {
                self.end_tool(call, entry.as_ref(), out);
            }
            AgentEvent::Submission { record } => self.settled(record, out),
            AgentEvent::AutoRetryStart { error_message, .. } => self.error(error_message, true, out),
            AgentEvent::TaskFailed { message, .. } => self.error(message, false, out),
            AgentEvent::CompactionStart { task_id, .. } => self.start_compaction(*task_id, out),
            AgentEvent::CompactionEnd { task_id, .. } => self.end_compaction(*task_id, out),
            AgentEvent::UsageChanged { usage } => self.usage(usage, out),
            AgentEvent::ApprovalUpdate { requests } => self.approvals_changed(requests, out),
            _ => {}
        }
    }

    fn turn_id(&self) -> Option<String> {
        self.current_turn()
    }

    fn item_started(&self, item: ThreadItem, out: &mut Vec<Output>) {
        if let Some(turn_id) = self.turn_id() {
            let notification = ItemStartedNotification {
                item,
                thread_id: self.thread_id.clone(),
                turn_id,
                started_at_ms: self.now(),
            };
            notify(out, ServerNotification::ItemStarted(notification));
        }
    }

    fn item_completed(&self, item: ThreadItem, out: &mut Vec<Output>) {
        if let Some(turn_id) = self.turn_id() {
            let notification = ItemCompletedNotification {
                item,
                thread_id: self.thread_id.clone(),
                turn_id,
                completed_at_ms: self.now(),
            };
            notify(out, ServerNotification::ItemCompleted(notification));
        }
    }

    fn open_turn(&mut self, first: SubmissionId, out: &mut Vec<Output>) {
        let started_at_ms = self.now();
        self.turn = Some(OpenTurn {
            first,
            inputs: Vec::new(),
            started_at_ms,
        });
        self.last_turn = Some(first.to_string());
        let turn = Turn {
            id: first.to_string(),
            items: Vec::new(),
            status: TurnStatus::InProgress,
            error: None,
            started_at: Some(started_at_ms / 1000),
            completed_at: None,
            duration_ms: None,
        };
        let notification = TurnNotification {
            thread_id: self.thread_id.clone(),
            turn,
        };
        notify(out, ServerNotification::TurnStarted(notification));
    }

    fn close_turn(&mut self, out: &mut Vec<Output>) {
        let Some(open) = self.turn.take() else {
            return;
        };
        self.partial = None;
        let (status, error) = turn_status(self.submissions.get(&open.first));
        let completed = self.now();
        let turn = Turn {
            id: open.first.to_string(),
            items: Vec::new(),
            status,
            error,
            started_at: Some(open.started_at_ms / 1000),
            completed_at: Some(completed / 1000),
            duration_ms: Some(completed - open.started_at_ms),
        };
        let notification = TurnNotification {
            thread_id: self.thread_id.clone(),
            turn,
        };
        notify(out, ServerNotification::TurnCompleted(notification));
    }

    fn entry(&mut self, entry: &EntryRecord, out: &mut Vec<Output>) {
        self.last_entry = self.last_entry.max(Some(entry.id));
        let input = self
            .submissions
            .values()
            .find(|record| input_entry(record) == Some(entry.id))
            .map(|record| record.id);
        if let Some(input) = input {
            self.input(input, entry, out);
        } else if entry.kind == ASSISTANT_ENTRY {
            self.answer(entry, out);
        }
    }

    fn input(&mut self, input: SubmissionId, entry: &EntryRecord, out: &mut Vec<Output>) {
        let joins =
            self.turn.as_ref().map(
                |turn| match self.submissions.get(&turn.first).map(|record| &record.state) {
                    None | Some(SubmissionState::Queued | SubmissionState::Placed { .. }) => true,
                    Some(SubmissionState::Done {
                        answer: Some(answer), ..
                    }) => entry.id < *answer,
                    Some(_) => false,
                },
            );
        if joins == Some(false) {
            self.close_turn(out);
        }
        if self.turn.is_none() {
            self.open_turn(input, out);
        }
        if let Some(turn) = &mut self.turn {
            turn.inputs.push(input);
        }
        self.requested.remove(&input);
        if let Some([Message::User(message), ..]) = entry.model.as_deref() {
            let item = user_message(input.to_string(), &message.content);
            self.item_started(item.clone(), out);
            self.item_completed(item, out);
        }
    }

    fn record_calls(&mut self, message: &AssistantMessage) {
        for block in &message.content {
            if let AssistantBlock::ToolCall(call) = block {
                self.calls.insert(
                    call.id.clone(),
                    Call {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                    },
                );
            }
        }
    }

    fn answer(&mut self, entry: &EntryRecord, out: &mut Vec<Output>) {
        let Some([Message::Assistant(message), ..]) = entry.model.as_deref() else {
            return;
        };
        let partial = self.partial.take().filter(|partial| partial.task == entry.by_task_id);
        for (index, block) in message.content.iter().enumerate() {
            let sent = partial
                .as_ref()
                .and_then(|partial| partial.blocks.get(index))
                .and_then(|block| block.sent.clone());
            let id = block_id(entry.by_task_id, index);
            // The tail the last partial did not carry, so the deltas add up to the text.
            if let Some(sent) = &sent {
                let tail = match block {
                    AssistantBlock::Text(text) => text.text.strip_prefix(sent.as_str()),
                    AssistantBlock::Thinking(thinking) => thinking.thinking.strip_prefix(sent.as_str()),
                    AssistantBlock::ToolCall(_) => None,
                };
                if let (Some(tail), Some(turn_id)) = (tail.filter(|tail| !tail.is_empty()), self.turn_id()) {
                    let thread_id = self.thread_id.clone();
                    let (item_id, delta) = (id.clone(), tail.to_owned());
                    let notification = if matches!(block, AssistantBlock::Text(_)) {
                        ServerNotification::AgentMessageDelta(ItemDeltaNotification {
                            thread_id,
                            turn_id,
                            item_id,
                            delta,
                        })
                    } else {
                        ServerNotification::ReasoningTextDelta(ReasoningTextDeltaNotification {
                            thread_id,
                            turn_id,
                            item_id,
                            delta,
                            content_index: 0,
                        })
                    };
                    notify(out, notification);
                }
            }
            let started = sent.is_some();
            let (start, end) = match block {
                AssistantBlock::Text(text) => (
                    ThreadItem::AgentMessage {
                        id: id.clone(),
                        text: String::new(),
                    },
                    ThreadItem::AgentMessage {
                        id,
                        text: text.text.clone(),
                    },
                ),
                AssistantBlock::Thinking(thinking) => (
                    ThreadItem::Reasoning {
                        id: id.clone(),
                        summary: Vec::new(),
                        content: Vec::new(),
                    },
                    ThreadItem::Reasoning {
                        id,
                        summary: Vec::new(),
                        content: vec![thinking.thinking.clone()],
                    },
                ),
                AssistantBlock::ToolCall(_) => continue,
            };
            if !started {
                self.item_started(start, out);
            }
            self.item_completed(end, out);
        }
        self.record_calls(message);
        self.last_usage = message.usage.clone();
    }

    fn sync_partial(&mut self, out: &mut Vec<Output>) {
        let Some(turn_id) = self.turn_id() else {
            return;
        };
        let now = self.now();
        let Some(partial) = &mut self.partial else {
            return;
        };
        let mut notifications = Vec::new();
        for (index, block) in partial.blocks.iter_mut().enumerate() {
            if block.kind == BlockKind::Other {
                continue;
            }
            let id = block_id(partial.task, index);
            if block.sent.is_none() {
                let item = match block.kind {
                    BlockKind::Text => ThreadItem::AgentMessage {
                        id: id.clone(),
                        text: String::new(),
                    },
                    _ => ThreadItem::Reasoning {
                        id: id.clone(),
                        summary: Vec::new(),
                        content: Vec::new(),
                    },
                };
                notifications.push(ServerNotification::ItemStarted(ItemStartedNotification {
                    item,
                    thread_id: self.thread_id.clone(),
                    turn_id: turn_id.clone(),
                    started_at_ms: now,
                }));
                block.sent = Some(String::new());
            }
            let sent = block.sent.get_or_insert_with(String::new);
            if block.text.len() > sent.len() && block.text.starts_with(sent.as_str()) {
                let delta = block.text[sent.len()..].to_owned();
                sent.clone_from(&block.text);
                notifications.push(match block.kind {
                    BlockKind::Text => ServerNotification::AgentMessageDelta(ItemDeltaNotification {
                        thread_id: self.thread_id.clone(),
                        turn_id: turn_id.clone(),
                        item_id: id,
                        delta,
                    }),
                    _ => ServerNotification::ReasoningTextDelta(ReasoningTextDeltaNotification {
                        thread_id: self.thread_id.clone(),
                        turn_id: turn_id.clone(),
                        item_id: id,
                        delta,
                        content_index: 0,
                    }),
                });
            }
        }
        out.extend(notifications.into_iter().map(Output::Notify));
    }

    fn start_tool(&mut self, call: Call, out: &mut Vec<Output>) {
        if self.started.contains_key(&call.id) {
            return;
        }
        self.item_started(tool_item(&call, &self.cwd, None), out);
        self.started.insert(call.id.clone(), call);
    }

    fn complete_tool(&mut self, call: &Call, entry: Option<&EntryRecord>, out: &mut Vec<Output>) {
        let outcome = entry
            .and_then(Outcome::of)
            .unwrap_or_else(|| Outcome::interrupted(call));
        let item = tool_item(call, &self.cwd, Some(&outcome));
        let streamed = self.streamed.remove(&call.id).unwrap_or_default();
        // The output the deltas did not carry, when they hold the start of it.
        if let ThreadItem::CommandExecution {
            aggregated_output: Some(output),
            ..
        } = &item
            && let Some(tail) = output.strip_prefix(streamed.as_str()).filter(|tail| !tail.is_empty())
            && let Some(turn_id) = self.turn_id()
        {
            let notification = ItemDeltaNotification {
                thread_id: self.thread_id.clone(),
                turn_id,
                item_id: call.id.clone(),
                delta: tail.to_owned(),
            };
            notify(out, ServerNotification::CommandExecutionOutputDelta(notification));
        }
        self.item_completed(item, out);
    }

    fn end_tool(&mut self, event: &ToolEventCall, entry: Option<&EntryRecord>, out: &mut Vec<Output>) {
        self.last_entry = self.last_entry.max(entry.map(|entry| entry.id));
        let call = match self.started.remove(&event.tool_call_id) {
            Some(call) => call,
            None => {
                let call = self.calls.get(&event.tool_call_id).cloned().unwrap_or(Call {
                    id: event.tool_call_id.clone(),
                    name: event.tool_name.clone(),
                    arguments: delta_ai::types::JsonObject::default(),
                });
                self.item_started(tool_item(&call, &self.cwd, None), out);
                call
            }
        };
        self.complete_tool(&call, entry, out);
    }

    fn start_compaction(&mut self, task: TaskId, out: &mut Vec<Output>) {
        if self.turn.is_some() && self.compactions.insert(task) {
            self.item_started(ThreadItem::ContextCompaction { id: task.to_string() }, out);
        }
    }

    fn end_compaction(&mut self, task: TaskId, out: &mut Vec<Output>) {
        if self.compactions.remove(&task) {
            self.item_completed(ThreadItem::ContextCompaction { id: task.to_string() }, out);
        }
    }

    fn settled(&mut self, record: &SubmissionRecord, out: &mut Vec<Output>) {
        if record.kind != SubmissionType::Input {
            return;
        }
        if !matches!(
            record.state,
            SubmissionState::Done { .. } | SubmissionState::Unanswered { .. }
        ) {
            return;
        }
        // The open turn's inputs close it at the end of the batch.
        if self
            .turn
            .as_ref()
            .is_some_and(|turn| turn.first == record.id || turn.inputs.contains(&record.id))
        {
            return;
        }
        // A requested turn's input that never reached the conversation (withdrawn while
        // queued): a turn that ends as it starts, so the client waiting for it is answered.
        if let SubmissionState::Unanswered { entry: None, .. } = record.state
            && self.requested.remove(&record.id)
        {
            let current = self.turn.take();
            self.open_turn(record.id, out);
            self.close_turn(out);
            self.turn = current;
        }
    }

    fn error(&self, message: &str, will_retry: bool, out: &mut Vec<Output>) {
        let notification = ErrorNotification {
            error: TurnError {
                message: message.to_owned(),
                codex_error_info: Some(Value::from("other")),
                additional_details: None,
            },
            will_retry,
            thread_id: self.thread_id.clone(),
            turn_id: self.turn_id().or_else(|| self.last_turn.clone()).unwrap_or_default(),
        };
        notify(out, ServerNotification::Error(notification));
    }

    fn usage(&self, usage: &UsageState, out: &mut Vec<Output>) {
        let Some(turn_id) = self.turn_id().or_else(|| self.last_turn.clone()) else {
            return;
        };
        let mut total = Usage::default();
        for usage in usage.models.values() {
            add(&mut total, usage);
        }
        let notification = ThreadTokenUsageUpdatedNotification {
            thread_id: self.thread_id.clone(),
            turn_id,
            token_usage: ThreadTokenUsage {
                total: breakdown(&total),
                last: breakdown(&self.last_usage),
                model_context_window: None,
            },
        };
        notify(out, ServerNotification::ThreadTokenUsageUpdated(notification));
    }

    fn approvals_changed(&mut self, requests: &[ApprovalRequest], out: &mut Vec<Output>) {
        let pending: BTreeMap<ApprovalId, &ApprovalRequest> = requests
            .iter()
            .filter(|request| request.decision.is_none())
            .map(|request| (request.id, request))
            .collect();
        let resolved: Vec<ApprovalId> = self
            .approvals
            .keys()
            .filter(|id| !pending.contains_key(id))
            .copied()
            .collect();
        for id in resolved {
            self.approvals.remove(&id);
            out.push(Output::Resolved(id));
        }
        for (id, request) in pending {
            if self.approvals.contains_key(&id) {
                continue;
            }
            let call = self.calls.get(&request.call_id).cloned().unwrap_or_else(|| Call {
                id: request.call_id.clone(),
                name: request.name.clone(),
                arguments: request.arguments.clone(),
            });
            self.start_tool(call, out);
            out.push(Output::Ask(self.ask(request)));
            self.approvals.insert(id, request.clone());
        }
    }

    /// The request of a pending approval.
    pub(crate) fn ask(&self, request: &ApprovalRequest) -> Ask {
        let thread_id = self.thread_id.clone();
        let turn_id = self.turn_id().or_else(|| self.last_turn.clone()).unwrap_or_default();
        let item_id = request.call_id.clone();
        let started_at_ms = request.requested_at;
        match ApprovalKind::of(&request.name) {
            ApprovalKind::CommandExecution => Ask::CommandExecution(
                request.id,
                CommandExecutionRequestApprovalParams {
                    thread_id,
                    turn_id,
                    item_id,
                    started_at_ms,
                    approval_id: Some(request.id.to_string()),
                    reason: request.prompt.clone(),
                    command: request
                        .arguments
                        .get("command")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    cwd: Some(self.cwd.clone()),
                    proposed_execpolicy_amendment: amendment(request.proposed_rule.as_deref()),
                },
            ),
            ApprovalKind::FileChange => Ask::FileChange(
                request.id,
                FileChangeRequestApprovalParams {
                    thread_id,
                    turn_id,
                    item_id,
                    started_at_ms,
                    reason: request.prompt.clone(),
                    grant_root: None,
                },
            ),
            ApprovalKind::Tool => Ask::Tool(
                request.id,
                ToolRequestApprovalParams {
                    thread_id,
                    turn_id,
                    item_id,
                    started_at_ms,
                    tool: request.name.clone(),
                    arguments: Value::Object(request.arguments.clone()),
                    reason: request.prompt.clone(),
                    proposed_rule: request.proposed_rule.clone(),
                },
            ),
        }
    }

    fn current_status(&self) -> ThreadStatus {
        if !self.approvals.is_empty() {
            ThreadStatus::Active {
                active_flags: vec![ThreadActiveFlag::WaitingOnApproval],
            }
        } else if self.run_active || self.turn.is_some() {
            ThreadStatus::Active {
                active_flags: Vec::new(),
            }
        } else {
            ThreadStatus::Idle
        }
    }

    fn update_status(&mut self, out: &mut Vec<Output>) {
        let status = self.current_status();
        if status != self.status {
            self.status = status.clone();
            let notification = ThreadStatusChangedNotification {
                thread_id: self.thread_id.clone(),
                status,
            };
            notify(out, ServerNotification::ThreadStatusChanged(notification));
        }
    }
}

#[cfg(test)]
mod tests {
    use delta_ai::providers::faux::{faux_assistant_message, faux_text};
    use delta_harness::generation::{LiveGeneration, RunControl, ToolSlot};
    use delta_harness::records::ConversationId;

    use super::*;

    fn notifications(outputs: &[Output]) -> Vec<&'static str> {
        outputs
            .iter()
            .map(|output| match output {
                Output::Notify(notification) => notification.method(),
                Output::Ask(_) => "ask",
                Output::Resolved(_) => "resolved",
            })
            .collect()
    }

    /// A snapshot of a run whose input 5 waits for its answer from generation task 7,
    /// streaming `partial`, with `approvals` pending.
    fn running(
        partial: &str,
        tools: Vec<ToolSlot>,
        approvals: Vec<ApprovalRequest>,
    ) -> (Snapshot, Vec<SubmissionRecord>) {
        let message = faux_assistant_message(vec![faux_text(partial)]);
        let mut generation = LiveGeneration::new(1);
        generation.message = Some(delta_chord::json::to_json(&message).unwrap());
        let snapshot = Snapshot {
            run: Some(RunControl {
                task_id: TaskId(7),
                inputs: vec![SubmissionId(5)],
                continuations: 0,
                input_checked: true,
            }),
            generation: Some(generation),
            tools: Some(tools),
            approvals,
            ..Snapshot::default()
        };
        let record = SubmissionRecord {
            id: SubmissionId(5),
            conversation_id: ConversationId(2),
            request_id: None,
            kind: SubmissionType::Input,
            state: SubmissionState::Placed { entry: EntryId(6) },
        };
        (snapshot, vec![record])
    }

    fn approval(id: u64) -> ApprovalRequest {
        ApprovalRequest {
            id: ApprovalId(TaskId(id)),
            call_id: format!("call-{id}"),
            name: "bash".to_owned(),
            arguments: serde_json::json!({ "command": "ls" }).as_object().cloned().unwrap(),
            approval_key: None,
            prompt: None,
            proposed_rule: Some("Bash(ls:*)".to_owned()),
            requested_at: 1,
            deadline: 2,
            decision: None,
        }
    }

    #[test]
    fn attaching_and_resyncing_send_what_is_in_flight_once() {
        let slot = |id: u64, status| ToolSlot::new(&format!("call-{id}"), "bash", status);
        let (snapshot, records) = running("Hel", vec![slot(9, SlotStatus::Pending)], vec![approval(9)]);
        let (mut translator, outputs) = Translator::attach("2".to_owned(), "/w".to_owned(), &snapshot, records);
        assert_eq!(translator.current_turn().as_deref(), Some("5"));
        assert_eq!(
            notifications(&outputs),
            ["item/started", "item/agentMessage/delta", "item/started", "ask"]
        );
        let Output::Ask(Ask::CommandExecution(_, params)) = &outputs[3] else {
            panic!("not a command approval")
        };
        assert_eq!(params.turn_id, "5");
        assert_eq!(params.proposed_execpolicy_amendment.as_ref().unwrap().command, ["ls"]);
        assert_eq!(
            translator.status(),
            ThreadStatus::Active {
                active_flags: vec![ThreadActiveFlag::WaitingOnApproval]
            }
        );
        // After a lag: only the text the client lacks, the new approval and the resolved one.
        let (snapshot, records) = running(
            "Hello",
            vec![slot(9, SlotStatus::Running), slot(10, SlotStatus::Pending)],
            vec![approval(10)],
        );
        let outputs = translator.resync(&snapshot, records.clone());
        assert_eq!(
            notifications(&outputs),
            ["item/agentMessage/delta", "resolved", "item/started", "ask"]
        );
        let Output::Notify(ServerNotification::AgentMessageDelta(delta)) = &outputs[0] else {
            panic!("not a delta")
        };
        assert_eq!((delta.item_id.as_str(), delta.delta.as_str()), ("7:0", "lo"));
        // The run ended meanwhile: its calls and the turn complete, the thread is idle.
        let mut records = records;
        records[0].state = SubmissionState::Done {
            entry: EntryId(6),
            answer: Some(EntryId(8)),
        };
        let outputs = translator.resync(&Snapshot::default(), records);
        assert_eq!(
            notifications(&outputs),
            [
                "item/completed",
                "item/completed",
                "resolved",
                "turn/completed",
                "thread/status/changed"
            ]
        );
        assert_eq!(translator.status(), ThreadStatus::Idle);
    }

    #[test]
    fn amendments_come_from_bash_prefix_rules() {
        assert_eq!(
            amendment(Some("Bash(git push:*)")),
            Some(ExecPolicyAmendment {
                command: vec!["git".to_owned(), "push".to_owned()]
            })
        );
        assert_eq!(amendment(Some("Edit(src/**)")), None);
        assert_eq!(amendment(Some("Bash(ls)")), None);
        assert_eq!(amendment(None), None);
    }
}
