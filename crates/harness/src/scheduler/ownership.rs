//! The ownership tree around live work (spec §5.4, §5.5) and the walks over it.
//!
//! Tasks and conversations form one tree: a task hangs from its owner task or, without one,
//! from its conversation; a conversation hangs from its owner task or is a root. `live`
//! mirrors every committed non-terminal task. `below` lists the tree downward around live
//! work: each node's children that are live tasks or have nodes listed below them, so what
//! lies below a node (a task's ordinary owned work, a scope's tasks, a cascade's reach) is a
//! walk down from it, and no pass visits every live task.
//!
//! The indexes are bounded by live work: an owner edge and an ended task's ownership fields
//! are kept only while something is listed below them. Work that appears later below a
//! dropped node finds it unknown (`unloaded`) and its chain is loaded from storage again on
//! the line (`scopes`).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ops::ControlFlow;

use crate::records::{ConversationId, TaskId, TaskOutcome, TaskRecord, TaskStatus};
use crate::session::Tx;

/// The immutable ownership fields of a task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TaskNode {
    pub(super) conversation_id: ConversationId,
    pub(super) owner: Option<TaskId>,
    pub(super) background: bool,
}

impl TaskNode {
    pub(super) fn of(record: &TaskRecord) -> Self {
        Self {
            conversation_id: record.conversation_id,
            owner: record.owner,
            background: record.background,
        }
    }

    /// The node it hangs from: its owner task, or its conversation.
    pub(super) fn parent(self) -> Node {
        self.owner.map_or(Node::Conversation(self.conversation_id), Node::Task)
    }
}

/// A node of the tree: a task or a conversation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum Node {
    Task(TaskId),
    Conversation(ConversationId),
}

/// One step of a walk up.
#[derive(Clone, Copy, Debug)]
pub(super) enum Step {
    Task(TaskId, TaskNode),
    Conversation(ConversationId),
    /// An owner edge or task that is not loaded yet.
    Unknown,
}

/// A conversation's owner edge as the tree knows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Edge {
    /// Not loaded yet.
    Unknown,
    /// Ownerless.
    Root,
    Owner(TaskId),
}

impl Edge {
    pub(super) fn owner(self) -> Option<TaskId> {
        match self {
            Self::Owner(owner) => Some(owner),
            Self::Unknown | Self::Root => None,
        }
    }
}

/// Where ordinary traversal starts: one conversation, or every ownerless conversation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Scope {
    Conversation(ConversationId),
    Roots,
}

/// Candidate records a commit staged; they override committed records in walks.
#[derive(Debug, Default)]
pub(super) struct Overlay {
    pub(super) tasks: HashMap<TaskId, TaskRecord>,
    pub(super) edges: HashMap<ConversationId, Option<TaskId>>,
}

impl Overlay {
    /// The tasks and conversations `tx` staged so far.
    pub(super) fn of(tx: &Tx<'_>) -> Self {
        Self {
            tasks: tx.staged_tasks().map(|record| (record.id, record.clone())).collect(),
            edges: tx
                .staged_conversations()
                .map(|record| (record.id, record.owner.map(|owner| owner.task_id)))
                .collect(),
        }
    }
}

/// A task a walk down reached; `live` unless it is terminal.
#[derive(Clone, Copy, Debug)]
pub(super) struct Reached<'a> {
    pub(super) id: TaskId,
    pub(super) node: TaskNode,
    pub(super) live: Option<&'a TaskRecord>,
}

/// How a walk down goes.
#[derive(Clone, Copy, Default)]
pub(super) struct Walk<'a> {
    pub(super) overlay: Option<&'a Overlay>,
    /// Enter background tasks too (they are always reached).
    pub(super) cross_background: bool,
    /// Enter only the tasks this accepts.
    pub(super) enter: Option<&'a dyn Fn(TaskId) -> bool>,
}

/// A live owner's durable cancellation intent: its abort mark, or a held outcome other than
/// `completed`.
pub(super) fn cancellation_intent(record: &TaskRecord) -> bool {
    record.state.status() != TaskStatus::Terminal && (record.abort_requested || failed_outcome(record))
}

/// Whether the record holds or ended with an outcome other than `completed`.
pub(super) fn failed_outcome(record: &TaskRecord) -> bool {
    record
        .state
        .outcome()
        .is_some_and(|outcome| !matches!(outcome, TaskOutcome::Completed { .. }))
}

/// The ownership indexes.
#[derive(Debug, Default)]
pub(super) struct Tree {
    /// Every committed non-terminal task.
    pub(super) live: BTreeMap<TaskId, TaskRecord>,
    /// Ownership fields of known terminal tasks, which walks pass through.
    pub(super) settled: HashMap<TaskId, TaskNode>,
    /// Owner task of each known conversation, `None` when ownerless.
    pub(super) edges: HashMap<ConversationId, Option<TaskId>>,
    below: HashMap<Node, BTreeSet<Node>>,
    roots: BTreeSet<ConversationId>,
    /// Known tasks whose chain may not be loaded.
    pub(super) unloaded: BTreeSet<TaskId>,
    /// Live tasks with cancellation intent, the sources of cascades.
    pub(super) intent: BTreeSet<TaskId>,
    /// Ended tasks and conversations found unneeded, dropped by the next sweep unless
    /// needed again by then.
    drop_queue: HashSet<Node>,
}

impl Tree {
    fn node(&self, id: TaskId, overlay: Option<&Overlay>) -> Option<TaskNode> {
        overlay
            .and_then(|overlay| overlay.tasks.get(&id))
            .or_else(|| self.live.get(&id))
            .map(TaskNode::of)
            .or_else(|| self.settled.get(&id).copied())
    }

    /// The owner edge of a conversation.
    pub(super) fn edge(&self, id: ConversationId, overlay: Option<&Overlay>) -> Edge {
        let owner = overlay
            .and_then(|overlay| overlay.edges.get(&id))
            .or_else(|| self.edges.get(&id));
        match owner {
            None => Edge::Unknown,
            Some(None) => Edge::Root,
            Some(Some(owner)) => Edge::Owner(*owner),
        }
    }

    /// Whether the chain above `node` is loaded, without walking: known tasks outside
    /// `unloaded` have a loaded chain.
    pub(super) fn known(&self, node: Node) -> bool {
        let owner = match node {
            Node::Task(id) => {
                return (self.live.contains_key(&id) || self.settled.contains_key(&id)) && !self.unloaded.contains(&id);
            }
            Node::Conversation(id) => self.edges.get(&id),
        };
        match owner {
            None => false,
            Some(None) => true,
            Some(Some(owner)) => self.known(Node::Task(*owner)),
        }
    }

    /// Records the owner edge of a conversation and lists it.
    pub(super) fn set_edge(&mut self, id: ConversationId, owner: Option<TaskId>) {
        self.edges.insert(id, owner);
        self.relist(Node::Conversation(id));
        self.drop_if_unneeded(Node::Conversation(id));
    }

    /// Knows a terminal task a chain load read, so walks pass through it; its own chain is
    /// not loaded yet.
    pub(super) fn settle_loaded(&mut self, id: TaskId, node: TaskNode) {
        self.settled.insert(id, node);
        self.unloaded.insert(id);
        self.relist(Node::Task(id));
        self.drop_if_unneeded(Node::Task(id));
    }

    /// Records a task that just turned terminal: it stays known while work is listed below it.
    pub(super) fn settle(&mut self, id: TaskId, node: TaskNode) {
        self.live.remove(&id);
        self.intent.remove(&id);
        self.settled.insert(id, node);
    }

    /// Queues an ended task or a conversation for the sweep when nothing is listed below it.
    pub(super) fn drop_if_unneeded(&mut self, node: Node) {
        if let Node::Task(id) = node
            && self.live.contains_key(&id)
        {
            return;
        }
        if !self.below.contains_key(&node) {
            self.drop_queue.insert(node);
        }
    }

    /// Drops the queued nodes that are still unneeded. Runs once a publication is tracked
    /// and at the start of the scheduler's commits, never inside one: work a commit creates
    /// below a chain its own closure loaded keeps that chain.
    pub(super) fn sweep(&mut self) {
        for node in std::mem::take(&mut self.drop_queue) {
            if self.below.contains_key(&node) {
                continue;
            }
            match node {
                Node::Conversation(id) => {
                    self.edges.remove(&id);
                    self.roots.remove(&id);
                }
                Node::Task(id) if !self.live.contains_key(&id) => {
                    self.settled.remove(&id);
                    self.unloaded.remove(&id);
                }
                Node::Task(_) => {}
            }
        }
    }

    /// Lists `start` under its parent while it is a live task or has nodes listed below it,
    /// unlists it otherwise; then its parent, as far up as that changes. A loop, as chains
    /// can be deeper than the call stack.
    pub(super) fn relist(&mut self, start: Node) {
        let mut node = start;
        loop {
            let needed =
                matches!(node, Node::Task(id) if self.live.contains_key(&id)) || self.below.contains_key(&node);
            let parent = match node {
                Node::Conversation(id) => match self.edges.get(&id) {
                    None => return,
                    Some(None) => {
                        if needed {
                            self.roots.insert(id);
                        } else {
                            self.roots.remove(&id);
                        }
                        return;
                    }
                    Some(Some(owner)) => Node::Task(*owner),
                },
                Node::Task(id) => match self.node(id, None) {
                    None => return,
                    Some(known) => known.parent(),
                },
            };
            let listed = self.below.get(&parent).is_some_and(|children| children.contains(&node));
            if listed == needed {
                return;
            }
            if needed {
                let first = !self.below.contains_key(&parent);
                self.below.entry(parent).or_default().insert(node);
                if !first {
                    return;
                }
            } else {
                if let Some(children) = self.below.get_mut(&parent) {
                    children.remove(&node);
                    if !children.is_empty() {
                        return;
                    }
                }
                self.below.remove(&parent);
                self.drop_if_unneeded(parent);
            }
            node = parent;
        }
    }

    /// Walks up from `start`: owner tasks and conversations, ending at an ownerless root or
    /// at an edge not loaded yet (`Step::Unknown`).
    pub(super) fn above(&self, start: Node, overlay: Option<&Overlay>) -> Vec<Step> {
        let mut steps = Vec::new();
        let mut at = Some(start);
        while let Some(node) = at {
            match node {
                Node::Task(id) => match self.node(id, overlay) {
                    None => {
                        steps.push(Step::Unknown);
                        return steps;
                    }
                    Some(known) => {
                        steps.push(Step::Task(id, known));
                        at = Some(known.parent());
                    }
                },
                Node::Conversation(id) => {
                    steps.push(Step::Conversation(id));
                    match self.edge(id, overlay) {
                        Edge::Unknown => {
                            steps.push(Step::Unknown);
                            return steps;
                        }
                        edge => at = edge.owner().map(Node::Task),
                    }
                }
            }
        }
        steps
    }

    /// Visits the tasks below `start`: those `below` lists and, under an overlay, the
    /// commit's new tasks, with its candidates in place of committed records. Background
    /// tasks are reached but not entered unless `cross_background`; neither are the tasks
    /// `enter` refuses.
    pub(super) fn walk_down<'a>(
        &'a self,
        start: Node,
        walk: Walk<'a>,
        mut visit: impl FnMut(Reached<'a>) -> ControlFlow<()>,
    ) {
        let links = walk.overlay.map(|overlay| self.links(overlay));
        let mut pending = vec![start];
        while let Some(at) = pending.pop() {
            let listed = self.below.get(&at).into_iter().flatten();
            let linked = links.as_ref().and_then(|links| links.get(&at)).into_iter().flatten();
            for child in listed.chain(linked) {
                let id = match *child {
                    Node::Conversation(_) => {
                        pending.push(*child);
                        continue;
                    }
                    Node::Task(id) => id,
                };
                let record = walk
                    .overlay
                    .and_then(|overlay| overlay.tasks.get(&id))
                    .or_else(|| self.live.get(&id));
                let Some(node) = record.map(TaskNode::of).or_else(|| self.settled.get(&id).copied()) else {
                    continue;
                };
                let live = record.filter(|record| record.state.status() != TaskStatus::Terminal);
                if visit(Reached { id, node, live }).is_break() {
                    return;
                }
                if (walk.cross_background || !node.background) && walk.enter.is_none_or(|enter| enter(id)) {
                    pending.push(*child);
                }
            }
        }
    }

    /// The links walks down need for a commit's new tasks: each under its parent, then each
    /// node above it that `below` does not list yet, up to one it lists, a live task or the
    /// chain's end.
    fn links(&self, overlay: &Overlay) -> HashMap<Node, BTreeSet<Node>> {
        let mut links: HashMap<Node, BTreeSet<Node>> = HashMap::new();
        for record in overlay.tasks.values() {
            if self.live.contains_key(&record.id) || self.settled.contains_key(&record.id) {
                continue;
            }
            let mut child = Node::Task(record.id);
            let mut parent = Some(TaskNode::of(record).parent());
            while let Some(at) = parent {
                if links.get(&at).is_some_and(|children| children.contains(&child)) {
                    break;
                }
                links.entry(at).or_default().insert(child);
                if let Node::Task(id) = at
                    && (self.live.contains_key(&id) || overlay.tasks.contains_key(&id))
                {
                    break;
                }
                let above = match at {
                    Node::Conversation(id) => self.edge(id, Some(overlay)).owner().map(Node::Task),
                    Node::Task(id) => self.settled.get(&id).map(|node| node.parent()),
                };
                if let Some(above) = above
                    && self.below.get(&above).is_some_and(|children| children.contains(&at))
                {
                    break;
                }
                child = at;
                parent = above;
            }
        }
        links
    }

    /// The live ordinary owned work of `owner` (§5.5): live non-background tasks below it
    /// that a walk down reaches without entering a background task; with `first`, only the
    /// first. The walk does not enter tasks in `done`, which have none.
    pub(super) fn owned_live(
        &self,
        owner: TaskId,
        overlay: Option<&Overlay>,
        first: bool,
        done: Option<&HashSet<TaskId>>,
    ) -> Vec<TaskId> {
        let skip_done = |id: TaskId| done.is_none_or(|done| !done.contains(&id));
        let walk = Walk {
            overlay,
            cross_background: false,
            enter: Some(&skip_done),
        };
        let mut found = Vec::new();
        self.walk_down(Node::Task(owner), walk, |reached| {
            if reached.live.is_some() && !reached.node.background {
                found.push(reached.id);
                if first {
                    return ControlFlow::Break(());
                }
            }
            ControlFlow::Continue(())
        });
        found
    }

    pub(super) fn has_owned_live(
        &self,
        owner: TaskId,
        overlay: Option<&Overlay>,
        done: Option<&HashSet<TaskId>>,
    ) -> bool {
        !self.owned_live(owner, overlay, true, done).is_empty()
    }

    /// The live tasks `owner` owns directly, in id order.
    pub(super) fn owned_children(&self, owner: TaskId) -> Vec<TaskRecord> {
        let children = self.below.get(&Node::Task(owner)).into_iter().flatten();
        children
            .filter_map(|node| match node {
                Node::Task(id) => self.live.get(id).filter(|record| record.owner == Some(owner)).cloned(),
                Node::Conversation(_) => None,
            })
            .collect()
    }

    /// The nearest live task above ended ordinary work `node`, in the overlay's view: the
    /// one that work held. Work at or below a background task holds nothing above it.
    pub(super) fn nearest_live_above<'a>(
        &'a self,
        node: TaskNode,
        overlay: Option<&'a Overlay>,
    ) -> Option<&'a TaskRecord> {
        let mut current = node;
        while !current.background {
            let owner = match current.owner {
                Some(owner) => owner,
                None => self.edge(current.conversation_id, overlay).owner()?,
            };
            let record = overlay
                .and_then(|overlay| overlay.tasks.get(&owner))
                .or_else(|| self.live.get(&owner));
            if let Some(record) = record.filter(|record| record.state.status() != TaskStatus::Terminal) {
                return Some(record);
            }
            current = record.map(TaskNode::of).or_else(|| self.settled.get(&owner).copied())?;
        }
        None
    }

    /// Whether ordinary traversal from `scope` reaches `start`: walking up reaches the
    /// scope's conversation (an ownerless one for `Roots`) without crossing a background
    /// owner, unless `cross_background`. `None` while an edge is not loaded.
    pub(super) fn in_scope(&self, start: Node, scope: Scope, cross_background: bool) -> Option<bool> {
        for step in self.above(start, None) {
            match step {
                Step::Unknown => return None,
                Step::Conversation(id) => {
                    if scope == Scope::Conversation(id) {
                        return Some(true);
                    }
                }
                Step::Task(_, node) => {
                    if node.background && !cross_background {
                        return Some(false);
                    }
                }
            }
        }
        Some(scope == Scope::Roots)
    }

    /// The nearest live owner above `start` whose cancellation intent reaches it: walking up
    /// finds an owner with intent before a background owner without it. Terminal owners
    /// never cascade (§5.4).
    pub(super) fn cancelling_owner(&self, start: Node) -> Option<&TaskRecord> {
        if self.intent.is_empty() {
            return None;
        }
        for step in self.above(start, None) {
            match step {
                Step::Unknown => return None,
                Step::Conversation(_) => {}
                Step::Task(id, node) => {
                    if let Some(live) = self.live.get(&id).filter(|record| cancellation_intent(record)) {
                        return Some(live);
                    }
                    if node.background {
                        return None;
                    }
                }
            }
        }
        None
    }

    pub(super) fn below_cancelled(&self, start: Node) -> bool {
        self.cancelling_owner(start).is_some()
    }

    /// No live non-background task in the scope; a task whose chain is not loaded counts as
    /// inside unless its walk up decides otherwise.
    pub(super) fn idle(&self, scope: Scope) -> bool {
        for id in &self.unloaded {
            if let Some(record) = self.live.get(id)
                && !record.background
                && self.in_scope(TaskNode::of(record).parent(), scope, false) != Some(false)
            {
                return false;
            }
        }
        let starts: Vec<ConversationId> = match scope {
            Scope::Conversation(id) => vec![id],
            Scope::Roots => self.roots.iter().copied().collect(),
        };
        let mut busy = false;
        for start in starts {
            self.walk_down(Node::Conversation(start), Walk::default(), |reached| {
                if reached.live.is_some() && !reached.node.background {
                    busy = true;
                    return ControlFlow::Break(());
                }
                ControlFlow::Continue(())
            });
            if busy {
                return false;
            }
        }
        true
    }

    /// Sizes of the indexes, for tests that bound the scheduler's memory.
    pub(super) fn sizes(&self) -> IndexSizes {
        IndexSizes {
            live: self.live.len(),
            settled: self.settled.len(),
            edges: self.edges.len(),
            below: self.below.values().map(BTreeSet::len).sum(),
            roots: self.roots.len(),
            unloaded: self.unloaded.len(),
        }
    }
}

/// Sizes of the scheduler's ownership indexes ([`Scheduler::index_sizes`](super::Scheduler::index_sizes)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IndexSizes {
    pub live: usize,
    pub settled: usize,
    pub edges: usize,
    pub below: usize,
    pub roots: usize,
    pub unloaded: usize,
}
