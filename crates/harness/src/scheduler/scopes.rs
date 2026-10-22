//! Loading owner chains on the line: the edges and ended tasks above live work that the
//! ownership tree does not hold yet, read from committed storage inside a scheduler commit.

use std::collections::BTreeMap;

use super::ownership::{Edge, Node, Overlay, TaskNode};
use super::state::Shared;
use crate::records::{ConversationId, SubmissionRecord, TaskStatus};
use crate::session::{Tx, TxError};

/// Queued input submissions by conversation.
pub(super) type QueuedInputs = BTreeMap<ConversationId, Vec<SubmissionRecord>>;

/// Loads the owner chains of the tasks in `unloaded` and, with `queued`, of every
/// conversation with queued inputs, which it returns.
pub(super) async fn load_scopes(shared: &Shared, tx: &mut Tx<'_>, queued: bool) -> Result<QueuedInputs, TxError> {
    let unloaded: Vec<_> = shared.lock().tree.unloaded.iter().copied().collect();
    for id in unloaded {
        // An earlier load in this loop may have passed it.
        if shared.lock().tree.unloaded.contains(&id) {
            load_chain(shared, tx, Node::Task(id), None).await?;
        }
    }
    let mut inputs = QueuedInputs::new();
    if !queued {
        return Ok(inputs);
    }
    for submission in tx.queued_inputs().await? {
        inputs.entry(submission.conversation_id).or_default().push(submission);
    }
    for id in inputs.keys() {
        load_chain(shared, tx, Node::Conversation(*id), None).await?;
    }
    Ok(inputs)
}

/// Loads the owner edges and task nodes from `start` up to its ownerless root. A known task
/// outside `unloaded` ends the walk, as its chain is loaded; the `unloaded` tasks passed
/// have a loaded chain afterwards. A loaded node can reveal a cancelling owner, so it asks
/// for a reconcile pass.
pub(super) async fn load_chain(
    shared: &Shared,
    tx: &mut Tx<'_>,
    start: Node,
    overlay: Option<&Overlay>,
) -> Result<(), TxError> {
    let mut passed = Vec::new();
    let mut at = Some(start);
    while let Some(node) = at {
        match node {
            Node::Task(id) => {
                let known = {
                    let state = shared.lock();
                    let known = state
                        .tree
                        .live
                        .get(&id)
                        .map(TaskNode::of)
                        .or_else(|| state.tree.settled.get(&id).copied());
                    // Created in this commit: the load from its own record covers what is above.
                    let staged = known.is_none() && overlay.is_some_and(|overlay| overlay.tasks.contains_key(&id));
                    if staged || known.is_some_and(|_| !state.tree.unloaded.contains(&id)) {
                        break;
                    }
                    known
                };
                let node = match known {
                    Some(node) => node,
                    None => {
                        let Some(record) = tx.stored_task(id).await? else {
                            return Ok(());
                        };
                        let node = TaskNode::of(&record);
                        let mut state = shared.lock();
                        if record.state.status() == TaskStatus::Terminal {
                            state.tree.settle_loaded(id, node);
                        }
                        state.reconcile_scheduled = true;
                        drop(state);
                        shared.wake_driver();
                        node
                    }
                };
                passed.push(id);
                at = Some(node.parent());
            }
            Node::Conversation(id) => {
                let edge = shared.lock().tree.edge(id, overlay);
                let owner = match edge {
                    Edge::Root => None,
                    Edge::Owner(owner) => Some(owner),
                    Edge::Unknown => {
                        let record = tx.stored_conversation(id).await?;
                        let owner = record.and_then(|record| record.owner).map(|owner| owner.task_id);
                        let mut state = shared.lock();
                        state.tree.set_edge(id, owner);
                        state.reconcile_scheduled = true;
                        drop(state);
                        shared.wake_driver();
                        owner
                    }
                };
                at = owner.map(Node::Task);
            }
        }
    }
    let mut state = shared.lock();
    for id in passed {
        state.tree.unloaded.remove(&id);
    }
    Ok(())
}
