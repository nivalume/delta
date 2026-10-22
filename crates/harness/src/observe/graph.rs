//! The task graph (spec §9.5): every live task of the Session with what it waits for and
//! the conversations it owns, as one `delta_chord::State` advanced from the commits.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use delta_chord::context::Context;
use delta_chord::delta::Op;
use delta_chord::json::{self, Json, Seg};
use delta_chord::state::{Delivery, State, Subscription};
use serde::{Deserialize, Serialize};

use crate::records::{
    ConversationId, ConversationQuery, JoinPolicy, TaskId, TaskOutcome, TaskQuery, TaskRecord, TaskState, TaskStatus,
};
use crate::scheduler::{BlockedProbe, BlockedReason};
use crate::session::{CommitChange, CommitPublication, ObserverGuard, Session, SessionError, Tx, TxError};

/// Scans read pages of this many records.
const PAGE: usize = 256;

/// How a task's outcome ended, while it is held.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutcomeStatus {
    Completed,
    Failed,
    Aborted,
    Orphaned,
    Faulted,
}

impl From<&TaskOutcome> for OutcomeStatus {
    fn from(outcome: &TaskOutcome) -> Self {
        match outcome {
            TaskOutcome::Completed { .. } => Self::Completed,
            TaskOutcome::Failed { .. } => Self::Failed,
            TaskOutcome::Aborted { .. } => Self::Aborted,
            TaskOutcome::Orphaned { .. } => Self::Orphaned,
            TaskOutcome::Faulted { .. } => Self::Faulted,
        }
    }
}

/// A live task's durable status with what it needs (`running` is the committed status).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum TaskGraphState {
    Pending {
        phase: String,
    },
    Running {
        phase: String,
    },
    Waiting {
        phase: String,
        on: Vec<TaskId>,
        policy: JoinPolicy,
    },
    Sleeping {
        phase: String,
        until: i64,
    },
    Completing {
        outcome: OutcomeStatus,
    },
}

/// One live task: its record without input, checkpoint payload, outcome payload and memos.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskGraphNode {
    pub id: TaskId,
    pub kind: String,
    pub conversation_id: ConversationId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<TaskId>,
    pub background: bool,
    pub abort_requested: bool,
    pub state: TaskGraphState,
    /// The conversations it owns, in id order.
    pub conversations: Vec<ConversationId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked: Option<BlockedReason>,
}

/// The value of the task graph: live tasks keyed by decimal id.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskGraph {
    pub tasks: BTreeMap<String, TaskGraphNode>,
}

fn phase(checkpoint: &Json) -> String {
    checkpoint
        .get("phase")
        .and_then(Json::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// The node of a live record; `None` for a terminal one.
fn node(
    record: &TaskRecord,
    conversations: Vec<ConversationId>,
    blocked: Option<BlockedReason>,
) -> Option<TaskGraphNode> {
    let state = match &record.state {
        TaskState::Pending { checkpoint } => TaskGraphState::Pending {
            phase: phase(checkpoint),
        },
        TaskState::Running { checkpoint } => TaskGraphState::Running {
            phase: phase(checkpoint),
        },
        TaskState::Waiting { checkpoint, on, policy } => TaskGraphState::Waiting {
            phase: phase(checkpoint),
            on: on.clone(),
            policy: *policy,
        },
        TaskState::Sleeping { checkpoint, until } => TaskGraphState::Sleeping {
            phase: phase(checkpoint),
            until: *until,
        },
        TaskState::Completing { outcome } => TaskGraphState::Completing {
            outcome: outcome.into(),
        },
        TaskState::Terminal { .. } => return None,
    };
    Some(TaskGraphNode {
        id: record.id,
        kind: record.kind.clone(),
        conversation_id: record.conversation_id,
        owner: record.owner,
        background: record.background,
        abort_requested: record.abort_requested,
        state,
        conversations,
        blocked,
    })
}

fn key(id: TaskId) -> String {
    id.0.to_string()
}

fn path(segments: Vec<Seg>) -> delta_chord::json::Path {
    segments.into()
}

/// What the graph mount follows.
struct Core {
    state: State,
    records: HashMap<TaskId, TaskRecord>,
    nodes: HashMap<TaskId, TaskGraphNode>,
    probe: BlockedProbe,
}

impl Core {
    fn set(&mut self, node: TaskGraphNode, ops: &mut Vec<Op>) {
        if self.nodes.get(&node.id) == Some(&node) {
            return;
        }
        let at = path(vec!["tasks".into(), key(node.id).as_str().into()]);
        ops.push(Op::Set(at, json::to_json(&node).unwrap_or(Json::Null)));
        self.nodes.insert(node.id, node);
    }

    fn advance(&mut self, publication: &CommitPublication) {
        let mut ops = Vec::new();
        for change in &publication.changes {
            let CommitChange::Task(record) = change else { continue };
            let conversations = self
                .nodes
                .get(&record.id)
                .map(|node| node.conversations.clone())
                .unwrap_or_default();
            match node(record, conversations, self.probe.blocked(record)) {
                Some(node) => {
                    self.records.insert(record.id, record.clone());
                    self.set(node, &mut ops);
                }
                None => {
                    self.records.remove(&record.id);
                    if self.nodes.remove(&record.id).is_some() {
                        ops.push(Op::Delete(path(vec!["tasks".into(), key(record.id).as_str().into()])));
                    }
                }
            }
        }
        for change in &publication.changes {
            let CommitChange::Conversation(conversation) = change else {
                continue;
            };
            let Some(owner) = conversation.owner.map(|owner| owner.task_id) else {
                continue;
            };
            let Some(mut node) = self.nodes.get(&owner).cloned() else {
                continue;
            };
            if !node.conversations.contains(&conversation.id) {
                node.conversations.push(conversation.id);
                node.conversations.sort();
                self.set(node, &mut ops);
            }
        }
        // Applies: every op targets a key this mount wrote.
        let _ = self.state.apply(&publication.context, ops);
    }

    /// Recomputes every node's `blocked`; call it on the line.
    fn recompute_blocked(&mut self) {
        let mut ops = Vec::new();
        let changed: Vec<TaskGraphNode> = self
            .nodes
            .values()
            .filter_map(|node| {
                let record = self.records.get(&node.id)?;
                let blocked = self.probe.blocked(record);
                (blocked != node.blocked).then(|| TaskGraphNode {
                    blocked,
                    ..node.clone()
                })
            })
            .collect();
        for node in changed {
            self.set(node, &mut ops);
        }
        let _ = self.state.apply(&Context::background(), ops);
    }
}

/// The Session's task graph mount, at most one at a time.
pub(crate) struct GraphMount {
    core: Arc<Mutex<Core>>,
    state: State,
    _observer: ObserverGuard,
}

impl GraphMount {
    fn lock(&self) -> MutexGuard<'_, Core> {
        self.core.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Recomputes every node's `blocked` reason in a commit (on the line), after the
    /// scheduler signalled that blocked reasons may have changed.
    pub(crate) async fn refresh_blocked(self: Arc<Self>, session: &Session, ctx: &Context) -> Result<(), SessionError> {
        session
            .commit(ctx, move |_tx| {
                Box::pin(async move {
                    self.lock().recompute_blocked();
                    Ok(())
                })
            })
            .await
    }
}

/// Every live task of `tx`'s storage, in id order.
async fn live_tasks(tx: &mut Tx<'_>) -> Result<Vec<TaskRecord>, TxError> {
    let mut tasks = Vec::new();
    for status in [
        TaskStatus::Pending,
        TaskStatus::Running,
        TaskStatus::Waiting,
        TaskStatus::Sleeping,
        TaskStatus::Completing,
    ] {
        let query = TaskQuery {
            status: Some(status),
            ..TaskQuery::default()
        };
        let mut cursor = None;
        loop {
            let page = tx.scan_tasks(query.clone(), PAGE, cursor).await?;
            tasks.extend(page.items);
            match page.next {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }
    }
    tasks.sort_by_key(|task| task.id);
    Ok(tasks)
}

async fn owned_conversations(tx: &mut Tx<'_>, task: TaskId) -> Result<Vec<ConversationId>, TxError> {
    let query = ConversationQuery {
        owner_task_id: Some(task),
        ..ConversationQuery::default()
    };
    let mut ids = Vec::new();
    let mut cursor = None;
    loop {
        let page = tx.scan_conversations(query.clone(), PAGE, cursor).await?;
        ids.extend(page.items.iter().map(|conversation| conversation.id));
        match page.next {
            Some(next) => cursor = Some(next),
            None => break,
        }
    }
    ids.sort();
    Ok(ids)
}

/// The live graph mount, built on the line when none lives.
#[derive(Default)]
pub(crate) struct GraphMounts {
    mount: Mutex<Weak<GraphMount>>,
}

impl GraphMounts {
    /// The mount and whether it was built now.
    pub(crate) async fn mount(
        &self,
        session: &Session,
        probe: BlockedProbe,
        ctx: &Context,
    ) -> Result<(Arc<GraphMount>, bool), SessionError> {
        if let Some(mount) = self.mount.lock().unwrap_or_else(PoisonError::into_inner).upgrade() {
            return Ok((mount, false));
        }
        let registrar = session.clone();
        let mount = session
            .commit(ctx, move |tx| {
                Box::pin(async move {
                    let mut records = HashMap::new();
                    let mut nodes = HashMap::new();
                    let mut graph = TaskGraph::default();
                    for record in live_tasks(tx).await? {
                        let conversations = owned_conversations(tx, record.id).await?;
                        if let Some(node) = node(&record, conversations, probe.blocked(&record)) {
                            graph.tasks.insert(key(record.id), node.clone());
                            nodes.insert(record.id, node);
                            records.insert(record.id, record);
                        }
                    }
                    let state = State::new(json::to_json(&graph).map_err(TxError::callback)?);
                    let core = Arc::new(Mutex::new(Core {
                        state: state.clone(),
                        records,
                        nodes,
                        probe,
                    }));
                    let following = core.clone();
                    let observer = registrar.observe_commits(Box::new(move |publication: &CommitPublication| {
                        following
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .advance(publication);
                        Ok(())
                    }));
                    Ok(Arc::new(GraphMount {
                        core,
                        state,
                        _observer: observer,
                    }))
                })
            })
            .await?;
        *self.mount.lock().unwrap_or_else(PoisonError::into_inner) = Arc::downgrade(&mount);
        Ok((mount, true))
    }
}

impl fmt::Debug for GraphMounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let live = self.mount.lock().unwrap_or_else(PoisonError::into_inner).strong_count() > 0;
        f.debug_struct("GraphMounts").field("live", &live).finish()
    }
}

/// The read-only task graph; it follows every commit while it, or one of its
/// subscriptions, lives.
#[derive(Clone)]
pub struct TaskGraphView {
    mount: Arc<GraphMount>,
}

impl TaskGraphView {
    pub(crate) fn new(mount: Arc<GraphMount>) -> Self {
        Self { mount }
    }

    /// The current value. O(1).
    pub fn value(&self) -> Json {
        self.mount.state.value()
    }

    /// The current value, typed.
    pub fn graph(&self) -> Result<TaskGraph, serde_json::Error> {
        json::from_json(&self.value())
    }

    /// Every revision from now on: a `Hydrate`, then one `Update` per commit that changed
    /// the graph.
    pub fn subscribe(&self) -> TaskGraphSubscription {
        TaskGraphSubscription {
            inner: self.mount.state.subscribe(),
            _mount: self.mount.clone(),
        }
    }
}

impl fmt::Debug for TaskGraphView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskGraphView")
            .field("sequence", &self.mount.state.sequence())
            .finish_non_exhaustive()
    }
}

/// The deliveries of a [`TaskGraphView`].
pub struct TaskGraphSubscription {
    inner: Subscription,
    _mount: Arc<GraphMount>,
}

impl TaskGraphSubscription {
    pub async fn next(&mut self) -> Option<Delivery> {
        self.inner.next().await
    }

    pub fn try_next(&mut self) -> Option<Delivery> {
        self.inner.try_next()
    }
}

impl fmt::Debug for TaskGraphSubscription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskGraphSubscription").finish_non_exhaustive()
    }
}
