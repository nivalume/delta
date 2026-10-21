//! A connection's threads: the outgoing line queue, the subscription and translator of each
//! thread it follows, and the `Thread` objects responses carry.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use delta_ai::types::Message;
use delta_chord::context::Context;
use delta_harness::approval::ApprovalId;
use delta_harness::harness::{Conversation, EntryRange, Harness, HarnessError};
use delta_harness::inbox::USER_ENTRY;
use delta_harness::model::ModelRef;
use delta_harness::observe::{AgentEvent, AgentEventStream, Snapshot};
use delta_harness::records::{ConversationId, EntryRecord, ScanOrder, SubmissionRecord};
use delta_protocol::delta::SnapshotNotification;
use delta_protocol::v2::{
    ApprovalsReviewer, AskForApproval, SandboxPolicy, ServerRequestResolvedNotification, SessionSource, Thread,
    ThreadSession, ThreadStatus,
};
use delta_protocol::{
    JsonRpcErrorBody, JsonRpcResponse, RequestId, ServerNotification, ServerNotificationEnvelope, ServerRequest,
};
use serde::Serialize;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::history::{self, timestamp};
use crate::items::ApprovalKind;
use crate::translate::{Output, Translator, now_ms};

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The connection's outgoing lines.
#[derive(Clone)]
pub(crate) struct Outgoing {
    lines: mpsc::UnboundedSender<String>,
    opt_out: Arc<RwLock<HashSet<String>>>,
}

impl Outgoing {
    pub(crate) fn new(lines: mpsc::UnboundedSender<String>) -> Self {
        Self {
            lines,
            opt_out: Arc::default(),
        }
    }

    /// Notification methods not to send.
    pub(crate) fn opt_out(&self, methods: Vec<String>) {
        *self.opt_out.write().unwrap_or_else(PoisonError::into_inner) = methods.into_iter().collect();
    }

    fn send(&self, message: &impl Serialize) {
        if let Ok(line) = serde_json::to_string(message) {
            // A closed connection drops what is left to say.
            let _ = self.lines.send(line);
        }
    }

    pub(crate) fn respond(&self, id: RequestId, result: serde_json::Value) {
        self.send(&JsonRpcResponse { id, result });
    }

    /// An error response; `id` is `null` when the request's id is unknown.
    pub(crate) fn fail(&self, id: Option<&RequestId>, code: i64, message: String) {
        let error = JsonRpcErrorBody {
            code,
            message,
            data: None,
        };
        self.send(&serde_json::json!({ "id": id, "error": error }));
    }

    pub(crate) fn notify(&self, notification: ServerNotification) {
        let opted_out = self
            .opt_out
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .contains(notification.method());
        if !opted_out {
            self.send(&ServerNotificationEnvelope {
                notification,
                emitted_at_ms: Some(now_ms()),
            });
        }
    }

    pub(crate) fn request(&self, request: &ServerRequest) {
        self.send(request);
    }
}

/// A server request waiting for the client's answer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PendingRequest {
    pub(crate) conversation: ConversationId,
    pub(crate) approval: ApprovalId,
    pub(crate) kind: ApprovalKind,
}

/// What every thread of a connection shares.
pub(crate) struct Shared {
    pub(crate) out: Outgoing,
    pub(crate) requests: Mutex<HashMap<RequestId, PendingRequest>>,
    next_request: AtomicI64,
}

impl Shared {
    pub(crate) fn new(out: Outgoing) -> Self {
        Self {
            out,
            requests: Mutex::default(),
            next_request: AtomicI64::new(0),
        }
    }

    pub(crate) fn take_request(&self, id: &RequestId) -> Option<PendingRequest> {
        lock(&self.requests).remove(id)
    }
}

/// A followed thread's translator and the request ids of its pending approvals.
pub(crate) struct ThreadState {
    pub(crate) translator: Translator,
    requests: HashMap<ApprovalId, RequestId>,
}

/// The submissions of the run `snapshot` shows.
async fn run_records(harness: &Harness, snapshot: &Snapshot, ctx: &Context) -> Vec<SubmissionRecord> {
    let mut records = Vec::new();
    for id in snapshot.run.iter().flat_map(|run| run.inputs.iter().copied()) {
        if let Ok(Some(submission)) = harness.submission(id, ctx).await
            && let Ok(record) = submission.status(ctx).await
        {
            records.push(record);
        }
    }
    records
}

/// A thread's working directory: its agent's, else `default`.
pub(crate) async fn thread_cwd(
    conversation: &Conversation,
    default: &str,
    ctx: &Context,
) -> Result<String, HarnessError> {
    Ok(conversation
        .agent(ctx)
        .await?
        .cwd
        .clone()
        .unwrap_or_else(|| default.to_owned()))
}

/// A thread attached to its events, not yet followed.
pub(crate) struct Attached {
    conversation: Conversation,
    stream: AgentEventStream,
    state: Arc<Mutex<ThreadState>>,
    outputs: Vec<Output>,
}

impl Attached {
    pub(crate) fn status(&self) -> ThreadStatus {
        lock(&self.state).translator.status()
    }

    pub(crate) fn state(&self) -> &Arc<Mutex<ThreadState>> {
        &self.state
    }
}

/// Attaches to `conversation`'s events.
pub(crate) async fn attach(
    harness: &Harness,
    conversation: Conversation,
    cwd: String,
    ctx: &Context,
) -> Result<Attached, HarnessError> {
    let stream = conversation.events(ctx).await?;
    let records = run_records(harness, stream.snapshot(), ctx).await;
    let (translator, outputs) = Translator::attach(conversation.id().to_string(), cwd, stream.snapshot(), records);
    let state = Arc::new(Mutex::new(ThreadState {
        translator,
        requests: HashMap::new(),
    }));
    Ok(Attached {
        conversation,
        stream,
        state,
        outputs,
    })
}

/// A thread the connection follows.
pub(crate) struct Followed {
    pub(crate) state: Arc<Mutex<ThreadState>>,
    task: JoinHandle<()>,
}

impl Followed {
    /// Sends what the attachment found in flight, then follows the events.
    pub(crate) fn start(attached: Attached, harness: Harness, shared: Arc<Shared>, ctx: Context) -> Self {
        let Attached {
            conversation,
            mut stream,
            state,
            outputs,
        } = attached;
        let id = conversation.id();
        emit(&shared, &state, id, outputs);
        let task_state = state.clone();
        let task = tokio::spawn(async move {
            while let Some(batch) = stream.next().await {
                let outputs = if let [AgentEvent::Snapshot(snapshot)] = batch.as_slice() {
                    shared.out.notify(ServerNotification::Snapshot(SnapshotNotification {
                        thread_id: id.to_string(),
                        snapshot: serde_json::to_value(snapshot).unwrap_or_default(),
                    }));
                    let records = run_records(&harness, snapshot, &ctx).await;
                    lock(&task_state).translator.resync(snapshot, records)
                } else {
                    lock(&task_state).translator.batch(&batch)
                };
                emit(&shared, &task_state, id, outputs);
            }
        });
        Self { state, task }
    }

    pub(crate) fn status(&self) -> ThreadStatus {
        lock(&self.state).translator.status()
    }

    pub(crate) fn current_turn(&self) -> Option<String> {
        lock(&self.state).translator.current_turn()
    }

    /// Sends the pending approvals again, under their request ids.
    pub(crate) fn resend(&self, shared: &Shared, conversation: ConversationId) {
        let state = lock(&self.state);
        for request in state.translator.pending() {
            if let Some(id) = state.requests.get(&request.id) {
                let ask = state.translator.ask(request);
                lock(&shared.requests).insert(
                    id.clone(),
                    PendingRequest {
                        conversation,
                        approval: request.id,
                        kind: ask.kind(),
                    },
                );
                shared.out.request(&ask.into_request(id.clone()));
            }
        }
    }

    /// Stops following and waits until the subscription is gone.
    pub(crate) async fn stop(self) {
        self.task.abort();
        let _ = self.task.await;
    }
}

/// Sends `outputs` of thread `conversation`: notifications, approval requests under new ids,
/// and the resolution of requests no longer pending.
fn emit(shared: &Shared, state: &Mutex<ThreadState>, conversation: ConversationId, outputs: Vec<Output>) {
    for output in outputs {
        match output {
            Output::Notify(notification) => shared.out.notify(notification),
            Output::Ask(ask) => {
                let id = RequestId::Integer(shared.next_request.fetch_add(1, Ordering::Relaxed));
                lock(&shared.requests).insert(
                    id.clone(),
                    PendingRequest {
                        conversation,
                        approval: ask.approval(),
                        kind: ask.kind(),
                    },
                );
                lock(state).requests.insert(ask.approval(), id.clone());
                shared.out.request(&ask.into_request(id));
            }
            Output::Resolved(approval) => {
                let Some(id) = lock(state).requests.remove(&approval) else {
                    continue;
                };
                lock(&shared.requests).remove(&id);
                let notification = ServerRequestResolvedNotification {
                    thread_id: conversation.to_string(),
                    request_id: id,
                };
                shared
                    .out
                    .notify(ServerNotification::ServerRequestResolved(notification));
            }
        }
    }
}

/// What a thread's session reports beside the thread.
pub(crate) struct ThreadInfo {
    pub(crate) model: Option<ModelRef>,
    pub(crate) cwd: String,
    pub(crate) reasoning_effort: Option<String>,
}

const PREVIEW_CHARS: usize = 200;

fn preview(entries: &[EntryRecord]) -> String {
    let text = entries
        .iter()
        .filter(|entry| entry.kind == USER_ENTRY)
        .find_map(|entry| match entry.model.as_deref() {
            Some([Message::User(message), ..]) => Some(message.content.text()),
            _ => None,
        })
        .unwrap_or_default();
    text.chars().take(PREVIEW_CHARS).collect()
}

fn seconds(entry: Option<&EntryRecord>) -> i64 {
    entry
        .and_then(|entry| entry.model.as_deref())
        .and_then(<[Message]>::first)
        .map_or(0, |message| timestamp(message) / 1000)
}

/// `conversation` as a thread with `status`, and with its turns when `turns`.
pub(crate) async fn thread(
    harness: &Harness,
    conversation: &Conversation,
    status: ThreadStatus,
    default_cwd: &str,
    turns: bool,
    ctx: &Context,
) -> Result<(Thread, ThreadInfo), HarnessError> {
    let record = conversation.record(ctx).await?;
    let agent = conversation.agent(ctx).await?;
    let cwd = agent.cwd.clone().unwrap_or_else(|| default_cwd.to_owned());
    let id = conversation.id().to_string();
    let (entries, last, turns) = if turns {
        let entries = history::entries(conversation, ctx).await?;
        let records = history::submissions(harness, conversation, ctx).await?;
        let turns = history::turns(&id, &cwd, &entries, records);
        let last = entries.last().cloned();
        (entries, last, turns)
    } else {
        let first = EntryRange {
            order: Some(ScanOrder::Ascending),
            ..EntryRange::default()
        };
        let entries = conversation.entries(first, 64, None, ctx).await?.items;
        let last = conversation
            .entries(EntryRange::default(), 1, None, ctx)
            .await?
            .items
            .pop();
        (entries, last, Vec::new())
    };
    let source = match record.owner {
        Some(owner) => SessionSource::SubAgent(serde_json::json!({
            "thread_spawn": { "parent_thread_id": owner.conversation_id.to_string() }
        })),
        None => SessionSource::AppServer,
    };
    let created_at = seconds(entries.first());
    let thread = Thread {
        id: id.clone(),
        session_id: id,
        forked_from_id: record.parent.map(|parent| parent.conversation_id.to_string()),
        parent_thread_id: record.owner.map(|owner| owner.conversation_id.to_string()),
        preview: preview(&entries),
        ephemeral: false,
        model_provider: agent
            .model
            .as_ref()
            .map(|model| model.provider.clone())
            .unwrap_or_default(),
        model: agent.model.as_ref().map(|model| model.model_id.clone()),
        created_at,
        updated_at: seconds(last.as_ref()).max(created_at),
        status,
        path: None,
        cwd: cwd.clone(),
        cli_version: env!("CARGO_PKG_VERSION").to_owned(),
        source,
        name: None,
        turns,
    };
    let reasoning_effort = serde_json::to_value(agent.thinking_level)
        .ok()
        .and_then(|level| level.as_str().map(str::to_owned));
    let info = ThreadInfo {
        model: agent.model.clone(),
        cwd,
        reasoning_effort,
    };
    Ok((thread, info))
}

/// The answer of `thread/start`, `thread/resume` and `thread/fork`.
pub(crate) fn session(thread: Thread, info: ThreadInfo) -> ThreadSession {
    let (provider, model) = info
        .model
        .map(|model| (model.provider, model.model_id))
        .unwrap_or_default();
    ThreadSession {
        thread,
        model,
        model_provider: provider,
        service_tier: None,
        cwd: info.cwd,
        approval_policy: AskForApproval::OnRequest,
        approvals_reviewer: ApprovalsReviewer::User,
        sandbox: SandboxPolicy::DangerFullAccess,
        reasoning_effort: info.reasoning_effort,
    }
}
