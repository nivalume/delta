//! Settlement: once the commit closure returned, every open change is prepared and the
//! transaction is assembled into one storage batch and the plans adoption applies.
//!
//! | Acquired as | Content write |
//! | --- | --- |
//! | created (absent address; `initial`) | `DocumentCreate` with base `{VERSION, value}`, also when the batch is empty |
//! | loaded, stored version < `VERSION` | `DocumentChange` base (required migration base) |
//! | loaded, non-empty batch, `checkpoint_when` true | `DocumentChange` base |
//! | loaded, non-empty batch, `checkpoint_when` false | `DocumentChange` delta `{VERSION, ops}` |
//! | loaded, empty batch | nothing |
//! | fork copy, not accessed | `DocumentCopy` |
//! | fork copy, accessed through a definition | `DocumentCreate` with the migrated, edited value |
//! | retired | `DocumentRetire`, after the final content |

use std::collections::HashSet;
use std::sync::Arc;

use delta_chord::delta::Prepared;
use delta_chord::state::State;

use super::cache::DocHandle;
use super::drafts::{CheckpointWhen, DocEntry, Target};
use super::error::TxError;
use super::forks::SCAN_PAGE;
use super::tx::{SubmissionChange, Tx};
use crate::documents::CheckpointInfo;
use crate::records::{
    BaseContent, ConversationId, DocumentContent, DocumentCreate, DocumentId, DocumentPoint, DocumentQuery,
    DocumentRecord, DocumentScope, Fork, Seq, StorageWrite, SubmissionRecord, SubmissionSettlement, SubmissionState,
    SubmissionType, TaskId, TaskStatus,
};

/// What a settled transaction writes and what adoption applies after storage accepted it.
pub(super) struct Settlement {
    pub(super) writes: Vec<StorageWrite>,
    pub(super) plans: Vec<Plan>,
}

/// One staged incarnation's write and publication, decided before storage admission so
/// adoption only applies it.
pub(super) struct Plan {
    pub(super) record: PlanRecord,
    pub(super) retire: bool,
    /// The create, copy or change write; `None` when only retirement (or nothing) is written.
    pub(super) content: Option<StorageWrite>,
    pub(super) change: Option<PlanChange>,
    /// The predicate and delta count of a loaded change that would store a delta.
    checkpoint: Option<(CheckpointWhen, u64)>,
    /// The owning conversation, resolved before admission so adoption performs no reads.
    pub(super) conversation_id: Option<ConversationId>,
}

/// A committed incarnation's record, or a new one's creation input.
pub(super) enum PlanRecord {
    Committed(DocumentRecord),
    New(DocumentCreate),
}

impl PlanRecord {
    fn id(&self) -> DocumentId {
        match self {
            Self::Committed(record) => record.id,
            Self::New(record) => record.id,
        }
    }

    fn scope(&self) -> DocumentScope {
        match self {
            Self::Committed(record) => record.scope,
            Self::New(record) => record.scope,
        }
    }

    fn fork(&self) -> Option<Fork> {
        match self {
            Self::Committed(record) => record.policy.map(|policy| policy.fork),
            Self::New(record) => record.policy.map(|policy| policy.fork),
        }
    }

    /// The record as committed at `seq`: a new incarnation is created there, a retired one
    /// ends there.
    pub(super) fn at(&self, seq: Seq, retire: bool) -> DocumentRecord {
        let mut record = match self {
            Self::Committed(record) => record.clone(),
            Self::New(record) => DocumentRecord::created(record.clone(), seq),
        };
        if retire {
            record.retired_at = Some(seq);
        }
        record
    }
}

/// The prepared revision of a tracked incarnation.
pub(super) enum PlanChange {
    Loaded {
        handle: Arc<DocHandle>,
        prepared: Prepared,
        version: u32,
    },
    Created {
        state: State,
        prepared: Prepared,
        version: u32,
    },
}

impl Plan {
    fn new(record: PlanRecord, retire: bool) -> Self {
        Self {
            record,
            retire,
            content: None,
            change: None,
            checkpoint: None,
            conversation_id: None,
        }
    }

    /// Whether adoption publishes it: every creation, copy and retirement, and a loaded
    /// incarnation that writes content (a migration-only base included).
    pub(super) fn publishes(&self) -> bool {
        self.retire || self.content.is_some()
    }

    /// For a loaded change: `Some(true)` when a base is written, `Some(false)` for a delta.
    pub(super) fn stored_base(&self) -> Option<bool> {
        match &self.content {
            Some(StorageWrite::DocumentChange { content, .. }) => Some(matches!(content, DocumentContent::Base { .. })),
            _ => None,
        }
    }
}

/// The plan of one staged entry; `None` for a retirement of an absent address. Prepares
/// the entry's open change.
fn plan_document(entry: DocEntry) -> Option<Plan> {
    let DocEntry { target, retire, .. } = entry;
    let plan = match target? {
        Target::Created { record, state, open } => {
            let prepared = open.change.prepare();
            let version = open.acquired.semantics.version;
            let content = BaseContent {
                version,
                value: prepared.value().clone(),
            };
            Plan {
                content: Some(StorageWrite::DocumentCreate {
                    record: record.clone(),
                    content,
                }),
                change: Some(PlanChange::Created {
                    state,
                    prepared,
                    version,
                }),
                ..Plan::new(PlanRecord::New(record), retire)
            }
        }
        Target::ForkCopy { record, source } => Plan {
            content: Some(StorageWrite::DocumentCopy {
                record: record.clone(),
                source,
            }),
            ..Plan::new(PlanRecord::New(record), retire)
        },
        Target::Retiring { record } => Plan::new(PlanRecord::Committed(record), retire),
        Target::Loaded {
            handle,
            stored_version,
            deltas_since_base,
            open,
        } => {
            let prepared = open.change.prepare();
            let version = open.acquired.semantics.version;
            let id = handle.record().id;
            let mut plan = Plan::new(PlanRecord::Committed(handle.record().clone()), retire);
            if stored_version < version {
                // A version change stores a base even when nothing else changed.
                let content = DocumentContent::Base {
                    version,
                    value: prepared.value().clone(),
                };
                plan.content = Some(StorageWrite::DocumentChange { id, content });
            } else if !prepared.is_empty() {
                let content = DocumentContent::Delta {
                    version,
                    ops: prepared.ops().to_vec(),
                };
                plan.content = Some(StorageWrite::DocumentChange { id, content });
                plan.checkpoint = Some((open.acquired.checkpoint_when, deltas_since_base));
            }
            plan.change = Some(PlanChange::Loaded {
                handle,
                prepared,
                version,
            });
            plan
        }
    };
    Some(plan)
}

impl Tx<'_> {
    /// Prepares every open change and assembles the batch. Any error rolls everything back;
    /// `checkpoint_when` runs last, after every other validation.
    pub(super) async fn settle(mut self) -> Result<Settlement, TxError> {
        let entries = std::mem::take(&mut self.documents);
        let mut plans: Vec<Plan> = entries.into_iter().filter_map(plan_document).collect();
        self.reject_fork_source_writes(&plans)?;
        self.validate_owners().await?;
        self.validate_replacements().await?;
        self.retire_terminal_task_documents(&mut plans).await?;
        self.resolve_publication_conversations(&mut plans).await?;
        self.resolve_submissions().await?;

        let mut writes = std::mem::take(&mut self.writes);
        writes.extend(self.submissions.values().cloned().map(StorageWrite::Submission));
        for task in self.tasks.values_mut() {
            if let Some(write) = task.write.take() {
                writes.push(StorageWrite::Task(write.record));
            }
        }
        for plan in &mut plans {
            if let Some((checkpoint_when, deltas_since_base)) = plan.checkpoint
                && let Some(PlanChange::Loaded { prepared, version, .. }) = &plan.change
                && checkpoint_when(prepared.value(), prepared.ops(), CheckpointInfo { deltas_since_base })
            {
                let id = plan.record.id();
                let content = DocumentContent::Base {
                    version: *version,
                    value: prepared.value().clone(),
                };
                plan.content = Some(StorageWrite::DocumentChange { id, content });
            }
            if let Some(content) = &plan.content {
                writes.push(content.clone());
            }
            if plan.retire {
                writes.push(StorageWrite::DocumentRetire { id: plan.record.id() });
            }
        }
        Ok(Settlement { writes, plans })
    }

    /// A copy has one unambiguous committed source: the fork transaction may write neither a
    /// copied source nor a `fork: current` document of the parent.
    fn reject_fork_source_writes(&self, plans: &[Plan]) -> Result<(), TxError> {
        for plan in plans.iter().filter(|plan| plan.publishes()) {
            let id = plan.record.id();
            if self.fork_source_documents.contains(&id) {
                return Err(TxError::Fork(format!(
                    "Cannot change fork source document {id} in the fork transaction"
                )));
            }
            if let DocumentScope::Conversation { conversation_id } = plan.record.scope()
                && self.fork_source_conversations.contains(&conversation_id)
                && plan.record.fork() == Some(Fork::Current)
            {
                return Err(TxError::Fork(format!(
                    "Cannot fork conversation {conversation_id} while changing its current-policy documents"
                )));
            }
        }
        Ok(())
    }

    /// New owned work needs a live owner, judged on the owner's final candidate: not
    /// `completing`, terminal or abort-marked.
    async fn validate_owners(&mut self) -> Result<(), TxError> {
        let mut owners: Vec<(&'static str, TaskId)> = Vec::new();
        for write in &self.writes {
            if let StorageWrite::Conversation(record) = write
                && let Some(owner) = record.owner
            {
                owners.push(("Conversation owner task", owner.task_id));
            }
        }
        for task in self.tasks.values() {
            if let Some(write) = task.write.as_ref().filter(|write| write.created)
                && let Some(owner) = write.record.owner
            {
                owners.push(("Task owner", owner));
            }
        }
        for (what, task_id) in owners {
            let task = self.current_task(task_id).await?.ok_or(TxError::UnknownTask(task_id))?;
            let status = task.state.status();
            if matches!(status, TaskStatus::Terminal | TaskStatus::Completing) {
                return Err(TxError::OwnerUnavailable {
                    what,
                    task: task_id,
                    status: status.as_str(),
                });
            }
            if task.abort_requested {
                return Err(TxError::OwnerUnavailable {
                    what,
                    task: task_id,
                    status: "abort-marked",
                });
            }
        }
        Ok(())
    }

    /// A replaced task exists, is not terminal and keeps its conversation.
    async fn validate_replacements(&mut self) -> Result<(), TxError> {
        let replaced: Vec<(TaskId, ConversationId)> = self
            .tasks
            .iter()
            .filter_map(|(id, task)| {
                let write = task.write.as_ref().filter(|write| !write.created)?;
                Some((*id, write.record.conversation_id))
            })
            .collect();
        for (id, conversation) in replaced {
            let committed = self.committed_task(id).await?.ok_or(TxError::UnknownTask(id))?;
            if committed.state.status() == TaskStatus::Terminal {
                return Err(TxError::TerminalTask(id));
            }
            if committed.conversation_id != conversation {
                return Err(TxError::TaskConversationChange(id));
            }
        }
        Ok(())
    }

    /// A task turning terminal retires every document of its scope, those staged here
    /// included.
    async fn retire_terminal_task_documents(&mut self, plans: &mut Vec<Plan>) -> Result<(), TxError> {
        let terminal: Vec<(TaskId, bool)> = self
            .tasks
            .iter()
            .filter_map(|(id, task)| {
                let write = task.write.as_ref()?;
                (write.record.state.status() == TaskStatus::Terminal).then_some((*id, write.created))
            })
            .collect();
        if terminal.is_empty() {
            return Ok(());
        }
        let mut retiring = HashSet::new();
        for plan in plans.iter_mut() {
            if let DocumentScope::Task { task_id } = plan.record.scope()
                && terminal.iter().any(|(id, _)| *id == task_id)
            {
                plan.retire = true;
                retiring.insert(plan.record.id());
            }
        }
        for (task_id, created) in terminal {
            if created {
                continue;
            }
            let mut cursor = None;
            loop {
                let query = DocumentQuery {
                    scope: DocumentScope::Task { task_id },
                    at: DocumentPoint::Current,
                    kind: None,
                };
                let ctx = &self.ctx;
                let page = self
                    .line
                    .guard
                    .read(ctx, |storage| storage.scan_documents(query, SCAN_PAGE, cursor, ctx))
                    .await?;
                for record in page.items {
                    if retiring.insert(record.id) {
                        plans.push(Plan::new(PlanRecord::Committed(record), true));
                    }
                }
                match page.next {
                    Some(next) => cursor = Some(next),
                    None => break,
                }
            }
        }
        Ok(())
    }

    /// The conversation each published document belongs to: its scope's, or its task's.
    async fn resolve_publication_conversations(&mut self, plans: &mut [Plan]) -> Result<(), TxError> {
        for plan in plans.iter_mut().filter(|plan| plan.publishes()) {
            plan.conversation_id = match plan.record.scope() {
                DocumentScope::Session => None,
                DocumentScope::Conversation { conversation_id } => Some(conversation_id),
                DocumentScope::Task { task_id } => {
                    let known = self.tasks.get(&task_id).and_then(|task| task.publication_conversation);
                    match known {
                        Some(conversation) => Some(conversation),
                        None => {
                            let conversation = self.current_task(task_id).await?.map(|task| task.conversation_id);
                            if let Some(conversation) = conversation {
                                self.tasks.entry(task_id).or_default().publication_conversation = Some(conversation);
                            }
                            conversation
                        }
                    }
                }
            };
        }
        Ok(())
    }

    /// Applies the staged settlements and placements in order, against the latest
    /// candidate record or the committed one.
    async fn resolve_submissions(&mut self) -> Result<(), TxError> {
        for (id, change) in std::mem::take(&mut self.submission_changes) {
            let current = match self.submissions.get(&id) {
                Some(record) => record.clone(),
                None => {
                    let ctx = &self.ctx;
                    let committed = self.line.guard.read(ctx, |storage| storage.submission(id, ctx)).await?;
                    committed.ok_or(TxError::UnknownSubmission(id))?
                }
            };
            if let Some(next) = apply_submission_change(&current, change)? {
                next.check()?;
                self.submissions.insert(id, next);
            }
        }
        Ok(())
    }
}

/// The record after one change, or `None` when it stays as it is. Placement turns a
/// queued input `placed` and a queued write `done`; only a placed input takes
/// `done { answer }`; a settled record stays.
fn apply_submission_change(
    current: &SubmissionRecord,
    change: SubmissionChange,
) -> Result<Option<SubmissionRecord>, TxError> {
    let state = match (&current.state, change) {
        (SubmissionState::Done { .. } | SubmissionState::Unanswered { .. }, _) => return Ok(None),
        (SubmissionState::Queued, SubmissionChange::Place(entry)) => match current.kind {
            SubmissionType::Input => SubmissionState::Placed { entry },
            SubmissionType::Write => SubmissionState::Done { entry, answer: None },
        },
        (SubmissionState::Placed { .. }, SubmissionChange::Place(_)) => {
            return Err(TxError::Submission(format!("Submission {} is not queued", current.id)));
        }
        (SubmissionState::Placed { entry }, SubmissionChange::Settle(SubmissionSettlement::Done { answer })) => {
            SubmissionState::Done {
                entry: *entry,
                answer: Some(answer),
            }
        }
        (SubmissionState::Queued, SubmissionChange::Settle(SubmissionSettlement::Done { .. })) => {
            return Err(TxError::Submission(format!(
                "Submission {} is not a placed input",
                current.id
            )));
        }
        (state, SubmissionChange::Settle(SubmissionSettlement::Unanswered { reason, detail })) => {
            let entry = match state {
                SubmissionState::Placed { entry } => Some(*entry),
                _ => None,
            };
            SubmissionState::Unanswered { entry, reason, detail }
        }
    };
    Ok(Some(SubmissionRecord {
        state,
        ..current.clone()
    }))
}
