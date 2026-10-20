//! The scheduler's memory: the ownership tree, the candidate sets of its passes and its
//! flags, kept in step with storage by an on-line commit observer.
//!
//! Everything here is touched on the commit line (the observer, the scheduler's commit
//! closures) or briefly from outside (`wake`, runtime memos); the mutex is uncontended.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use tokio::sync::{Notify, watch};

use super::erased::AnyTask;
use super::invocation::{Invocation, Mode};
use super::ownership::{Node, TaskNode, Tree, cancellation_intent, failed_outcome};
use crate::records::{JoinPolicy, SubmissionState, SubmissionType, TaskId, TaskRecord, TaskState, TaskStatus};
use crate::session::{CommitChange, CommitObserver, CommitPublication};

/// What the observer, the driver, the invocations and the waiters share.
#[derive(Debug)]
pub(super) struct Shared {
    state: Mutex<State>,
    /// Wakes the driver: a pass may have work.
    pub(super) wakeup: Notify,
    /// Bumped after every commit that changed tasks, every reconcile pass and at close:
    /// idle waits recheck.
    pub(super) changes: watch::Sender<u64>,
    /// Bumped when blocked reasons may have changed without a commit (a catalog
    /// publication, a failed migration, a newly parked task).
    pub(super) blocked: watch::Sender<u64>,
}

impl Shared {
    pub(super) fn new() -> Self {
        Self {
            state: Mutex::new(State::default()),
            wakeup: Notify::new(),
            changes: watch::Sender::new(0),
            blocked: watch::Sender::new(0),
        }
    }

    pub(super) fn lock(&self) -> MutexGuard<'_, State> {
        // Nothing panics while the state is inconsistent, so a poisoned lock is usable.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Wakes the driver for a pass.
    pub(super) fn wake_driver(&self) {
        self.wakeup.notify_one();
    }

    /// Asks for a reservation pass.
    pub(super) fn kick(&self) {
        self.lock().dirty = true;
        self.wake_driver();
    }

    pub(super) fn bump_changes(&self) {
        self.changes.send_modify(|count| *count = count.wrapping_add(1));
    }

    pub(super) fn bump_blocked(&self) {
        self.blocked.send_modify(|count| *count = count.wrapping_add(1));
    }
}

/// The on-line commit observer that keeps the memory in step with storage and wakes the
/// driver when a pass has work.
pub(super) struct Observer {
    shared: Arc<Shared>,
}

impl Observer {
    pub(super) fn new(shared: Arc<Shared>) -> Self {
        Self { shared }
    }
}

impl CommitObserver for Observer {
    fn committed(&mut self, publication: &CommitPublication) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (changed, wake, ended) = {
            let mut state = self.shared.lock();
            let changed = state.observe(publication);
            let wake = changed || state.reconcile_scheduled;
            (changed, wake, std::mem::take(&mut state.ended))
        };
        for invocation in ended {
            invocation.signal("invocation has ended");
        }
        if changed {
            self.shared.bump_changes();
        }
        if wake {
            self.shared.wake_driver();
        }
        Ok(())
    }
}

/// The scheduler's memory.
#[derive(Debug, Default)]
pub(super) struct State {
    pub(super) tree: Tree,
    /// Live tasks that are not `completing`: the candidates of a reservation pass.
    pub(super) runnable: BTreeSet<TaskId>,
    /// `completing` tasks the next finalize pass checks.
    pub(super) finalize_checks: BTreeSet<TaskId>,
    /// Live waiting tasks by each task in their `on`.
    pub(super) waiters: HashMap<TaskId, BTreeSet<TaskId>>,
    /// `fail_fast` waiters the next reconcile checks for a failed task in `on`.
    pub(super) fail_fast_checks: BTreeSet<TaskId>,
    /// Live `abandon_on_restart` tasks found at open; the first reservation pass marks them.
    pub(super) abandoned: BTreeSet<TaskId>,
    /// Sleeping live tasks, due or not.
    pub(super) sleeping: BTreeSet<TaskId>,
    /// Tasks a pass left out as parked; each pass checks them again.
    pub(super) parked: BTreeSet<TaskId>,
    /// The definition whose migration failed, per task; retried only for another one.
    pub(super) failed_migrations: HashMap<TaskId, Arc<dyn AnyTask>>,
    /// The active invocation of each task, registered on the line at reservation.
    pub(super) invocations: HashMap<TaskId, Arc<Invocation>>,
    /// Every spawned invocation until its code finished, by invocation id: close joins them.
    pub(super) spawned: HashMap<u64, Arc<Invocation>>,
    /// Invocations whose closing commit decided to stop: the observer ends each when that
    /// commit publishes its task, on the line; a refused commit publishes nothing.
    pub(super) stopping: HashMap<TaskId, Arc<Invocation>>,
    /// Invocations the observer ended, signalled once it released the lock.
    pub(super) ended: Vec<Arc<Invocation>>,
    /// The invocations active when the scheduler sealed: close writes them back.
    pub(super) stopped: Vec<Arc<Invocation>>,
    /// The invocations the reservation commit underway registered.
    pub(super) reserving: Vec<Arc<Invocation>>,
    /// Task waits: dropping a sender (terminal, close) wakes its receivers.
    pub(super) task_watches: HashMap<TaskId, watch::Sender<()>>,
    pub(super) next_invocation: u64,
    pub(super) reconcile_scheduled: bool,
    /// Reconcile passes underway with work: `wake()` counts them active.
    pub(super) reconciling: usize,
    pub(super) cascade_pending: bool,
    pub(super) dirty: bool,
    pub(super) enabled: bool,
    pub(super) closing: bool,
}

impl State {
    /// Whether a reconcile pass would have work.
    pub(super) fn reconcile_work(&self) -> bool {
        self.cascade_pending || !self.finalize_checks.is_empty() || !self.fail_fast_checks.is_empty()
    }

    /// Brings the indexes in line with a task's committed record. A task that ended may free
    /// the nearest live task above it, and its waiters may run.
    pub(super) fn track(&mut self, previous: Option<&TaskRecord>, record: TaskRecord) {
        let id = record.id;
        if let Some(TaskState::Waiting { on, .. }) = previous.map(|previous| &previous.state) {
            for member in on {
                if let Some(waiting) = self.waiters.get_mut(member) {
                    waiting.remove(&id);
                    if waiting.is_empty() {
                        self.waiters.remove(member);
                    }
                }
            }
        }
        if record.state.status() == TaskStatus::Terminal {
            let node = TaskNode::of(&record);
            self.tree.settle(id, node);
            self.sleeping.remove(&id);
            self.runnable.remove(&id);
            self.finalize_checks.remove(&id);
            // Before anything is forgotten: the walk up passes the ended tasks between them.
            self.release(node);
            if let Some(waiting) = self.waiters.get(&id) {
                self.runnable.extend(waiting.iter().copied());
            }
            self.tree.relist(Node::Task(id));
            self.tree.drop_if_unneeded(Node::Task(id));
            return;
        }
        let parent = TaskNode::of(&record).parent();
        if cancellation_intent(&record) {
            self.tree.intent.insert(id);
        } else {
            self.tree.intent.remove(&id);
        }
        match &record.state {
            TaskState::Completing { .. } => {
                self.runnable.remove(&id);
                self.finalize_checks.insert(id);
            }
            _ => {
                self.runnable.insert(id);
            }
        }
        if let TaskState::Waiting { on, .. } = &record.state {
            for member in on {
                self.waiters.entry(*member).or_default().insert(id);
            }
        }
        if matches!(record.state, TaskState::Sleeping { .. }) {
            self.sleeping.insert(id);
        } else {
            self.sleeping.remove(&id);
        }
        self.tree.live.insert(id, record);
        if previous.is_none() {
            if !self.tree.known(parent) {
                self.tree.unloaded.insert(id);
            }
            self.tree.relist(Node::Task(id));
        }
    }

    /// The nearest live task above ended ordinary work may be free now: one holding its
    /// outcome finalizes, an abort-marked one runs its abort handler.
    fn release(&mut self, node: TaskNode) {
        let Some(above) = self.tree.nearest_live_above(node, None) else {
            return;
        };
        let id = above.id;
        if above.state.status() == TaskStatus::Completing {
            self.finalize_checks.insert(id);
        } else if above.abort_requested {
            self.runnable.insert(id);
        }
    }

    /// Follows one publication (spec §5.4, §5.5). Returns whether it changed tasks.
    pub(super) fn observe(&mut self, publication: &CommitPublication) -> bool {
        // Edges first, so the tasks a commit creates in a new conversation hang from it.
        for change in &publication.changes {
            if let CommitChange::Conversation(record) = change
                && !self.tree.edges.contains_key(&record.id)
            {
                self.tree.set_edge(record.id, record.owner.map(|owner| owner.task_id));
            }
        }
        let mut changed = false;
        let mut created = Vec::new();
        let mut failed = Vec::new();
        for change in &publication.changes {
            let CommitChange::Task(record) = change else {
                continue;
            };
            changed = true;
            if let Some(invocation) = self.stopping.remove(&record.id)
                && invocation.mark_ended()
            {
                self.release_invocation(&invocation);
                self.ended.push(invocation);
            }
            let previous = self.tree.live.get(&record.id).cloned();
            if failed_outcome(record) && previous.as_ref().is_none_or(|previous| !failed_outcome(previous)) {
                failed.push(record.id);
            }
            self.track(previous.as_ref(), record.clone());
            self.follow(previous.as_ref(), record);
            if previous.is_none() && record.state.status() != TaskStatus::Terminal {
                created.push(record.id);
            }
        }
        for id in failed {
            let waiting: Vec<TaskId> = self.waiters.get(&id).into_iter().flatten().copied().collect();
            for waiter in waiting {
                if let Some(TaskState::Waiting {
                    policy: JoinPolicy::FailFast,
                    ..
                }) = self.tree.live.get(&waiter).map(|record| &record.state)
                {
                    self.fail_fast_checks.insert(waiter);
                    self.reconcile_scheduled = true;
                }
            }
        }
        for change in &publication.changes {
            // A queued input below a cancelled owner is withdrawn, even after its cascade.
            if let CommitChange::Submission(record) = change
                && record.kind == SubmissionType::Input
                && matches!(record.state, SubmissionState::Queued)
                && !self.tree.intent.is_empty()
            {
                let node = Node::Conversation(record.conversation_id);
                if !self.tree.known(node) || self.tree.below_cancelled(node) {
                    self.cascade_pending = true;
                }
            }
        }
        for id in created {
            // Work created below a cancelled owner, even after its cascade, is aborted too.
            let Some(record) = self.tree.live.get(&id) else {
                continue;
            };
            if self.tree.unloaded.contains(&id) {
                self.reconcile_scheduled = true;
            } else if !record.background
                && !record.abort_requested
                && self.tree.below_cancelled(TaskNode::of(record).parent())
            {
                self.cascade_pending = true;
            }
        }
        if self.cascade_pending {
            self.reconcile_scheduled = true;
        }
        self.tree.sweep();
        if changed {
            self.dirty = true;
        }
        changed
    }

    /// What a task's new record asks of the passes.
    fn follow(&mut self, previous: Option<&TaskRecord>, record: &TaskRecord) {
        let id = record.id;
        if record.state.status() == TaskStatus::Terminal {
            self.failed_migrations.remove(&id);
            self.fail_fast_checks.remove(&id);
            // Dropping the sender resolves the task's waiters.
            self.task_watches.remove(&id);
            // Its owner may finalize now.
            self.reconcile_scheduled = true;
            return;
        }
        let was_marked = previous.is_some_and(|previous| previous.abort_requested);
        if record.abort_requested && !was_marked {
            self.cascade_pending = true;
            // Signal a run invocation of the newly marked task; its next step ends it.
            if let Some(invocation) = self.invocations.get(&id)
                && invocation.mode == Mode::Run
            {
                invocation.signal("was aborted");
            }
        }
        // A request replacing a `restart` mark upgrades the marks below.
        if previous.is_some_and(|previous| previous.abort_reason.is_some()) && record.abort_reason.is_none() {
            self.cascade_pending = true;
        }
        let status = record.state.status();
        let was = previous.map(|previous| previous.state.status());
        if status == TaskStatus::Completing && was != Some(TaskStatus::Completing) {
            if cancellation_intent(record) {
                self.cascade_pending = true;
            }
            self.reconcile_scheduled = true;
        }
        if let TaskState::Waiting {
            policy: JoinPolicy::FailFast,
            ..
        } = record.state
            && was != Some(TaskStatus::Waiting)
        {
            self.fail_fast_checks.insert(id);
            self.reconcile_scheduled = true;
        }
    }
}
