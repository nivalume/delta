//! [`Tx`]: one commit's view of the Session. Table reads, table writes and the staging
//! state; documents are in `drafts`, settlement in `plan`.

use std::collections::{HashMap, HashSet};
use std::fmt;

use delta_chord::context::Context;
use delta_chord::json::{self, Json};
use indexmap::IndexMap;
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::drafts::DocEntry;
use super::error::TxError;
use super::forks;
use super::line::Line;
use crate::records::{
    ConversationId, ConversationOwner, ConversationOwnership, ConversationParent, ConversationQuery,
    ConversationRecord, Cursor, DocumentAddress, DocumentId, EntryDraft, EntryId, EntryKind, EntryQuery, EntryRecord,
    Head, Page, ROOT_CONVERSATION_ID, RecordError, StorageWrite, SubmissionCreate, SubmissionId, SubmissionQuery,
    SubmissionRecord, SubmissionSettlement, SubmissionStatus, SubmissionType, TaskId, TaskQuery, TaskRecord, TaskState,
    TaskStatus, TypedEntry, TypedEntryDraft,
};

/// The defaults one commit binds to: the conversation `create_task_record` uses when its
/// options name none, and the task stamped as `by_task_id` on appended entries (a task
/// runtime commit).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TransactionScope {
    pub(crate) conversation_id: Option<ConversationId>,
    pub(crate) task_id: Option<TaskId>,
}

/// Where and how a new task lives.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TaskOptions {
    /// The task's conversation; a child task lives in its owner's.
    pub conversation_id: Option<ConversationId>,
    /// The owning task; absent for a conversation-owned task.
    pub owner: Option<TaskId>,
    /// Not waited for by idle traversal; refused for a child task.
    pub background: bool,
    /// Abort instead of rerunning after a restart.
    pub abandon_on_restart: bool,
}

impl TaskOptions {
    /// A conversation-owned task of `conversation_id`.
    pub fn conversation(conversation_id: ConversationId) -> Self {
        Self {
            conversation_id: Some(conversation_id),
            ..Self::default()
        }
    }

    /// A child task of `owner`, in the owner's conversation.
    pub fn owned_by(owner: TaskId) -> Self {
        Self {
            owner: Some(owner),
            ..Self::default()
        }
    }
}

/// An untyped task to create: its definition's name and version, its input and first
/// checkpoint. The scheduler's typed `create_task` builds one from a definition.
#[derive(Clone, Debug, PartialEq)]
pub struct TaskCreate {
    pub kind: String,
    pub version: u32,
    pub input: Json,
    pub checkpoint: Json,
}

/// The committed and candidate state of one task the transaction touched.
#[derive(Default)]
pub(super) struct TxTask {
    /// The committed record, read once.
    pub(super) committed: Committed,
    pub(super) write: Option<TaskWrite>,
    pub(super) publication_conversation: Option<ConversationId>,
}

/// A committed task record as the transaction read it.
#[derive(Default)]
pub(super) enum Committed {
    #[default]
    Unread,
    Read(Option<TaskRecord>),
}

pub(super) struct TaskWrite {
    pub(super) created: bool,
    pub(super) record: TaskRecord,
}

/// A staged submission change, resolved at assembly against the latest candidate.
pub(super) enum SubmissionChange {
    Place(EntryId),
    Settle(SubmissionSettlement),
}

/// One commit's transaction. Reads and writes stage; nothing is visible until the commit
/// settles, and a closure error rolls everything back.
///
/// Table reads are allowed until the first table write ([`TxError::ReadAfterWrite`] after
/// it); document access stays available. Creation methods return their record, so nothing
/// needs reading back. `Tx` borrows the commit line: a document that is not cached is
/// loaded with a plain `await`, and a [`Doc`](super::Doc) cannot outlive the closure.
pub struct Tx<'line> {
    pub(super) line: &'line mut Line,
    pub(super) ctx: Context,
    pub(super) scope: TransactionScope,
    pub(super) has_table_write: bool,
    /// Conversation and entry writes, in staging order.
    pub(super) writes: Vec<StorageWrite>,
    pub(super) created_conversations: HashSet<ConversationId>,
    pub(super) fork_source_conversations: HashSet<ConversationId>,
    pub(super) fork_source_documents: HashSet<DocumentId>,
    pub(super) tasks: IndexMap<TaskId, TxTask>,
    pub(super) submissions: IndexMap<SubmissionId, SubmissionRecord>,
    pub(super) submission_changes: Vec<(SubmissionId, SubmissionChange)>,
    /// Every document acquisition, copy and retirement, in staging order.
    pub(super) documents: Vec<DocEntry>,
    /// The latest entry of `documents` at each address.
    pub(super) latest: HashMap<DocumentAddress, usize>,
}

impl fmt::Debug for Tx<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tx")
            .field("scope", &self.scope)
            .field("has_table_write", &self.has_table_write)
            .field("writes", &self.writes.len())
            .field("tasks", &self.tasks.len())
            .field("submissions", &self.submissions.len())
            .field("documents", &self.documents.len())
            .finish_non_exhaustive()
    }
}

impl<'line> Tx<'line> {
    pub(super) fn new(line: &'line mut Line, ctx: Context, scope: TransactionScope) -> Self {
        Self {
            line,
            ctx,
            scope,
            has_table_write: false,
            writes: Vec::new(),
            created_conversations: HashSet::new(),
            fork_source_conversations: HashSet::new(),
            fork_source_documents: HashSet::new(),
            tasks: IndexMap::new(),
            submissions: IndexMap::new(),
            submission_changes: Vec::new(),
            documents: Vec::new(),
            latest: HashMap::new(),
        }
    }
}

impl Tx<'_> {
    /// The commit's context.
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    fn check_read(&self, method: &'static str) -> Result<(), TxError> {
        if self.has_table_write {
            Err(TxError::ReadAfterWrite { method })
        } else {
            Ok(())
        }
    }

    // ─── Table reads ────────────────────────────────────────────────────────

    /// The committed conversation record.
    pub async fn conversation(&mut self, id: ConversationId) -> Result<Option<ConversationRecord>, TxError> {
        self.check_read("conversation")?;
        let ctx = &self.ctx;
        self.line.guard.read(ctx, |storage| storage.conversation(id, ctx)).await
    }

    /// An entry by id, whatever its conversation.
    pub async fn entry(&mut self, id: EntryId) -> Result<Option<EntryRecord>, TxError> {
        self.check_read("entry")?;
        let ctx = &self.ctx;
        let stored = self.line.guard.read(ctx, |storage| storage.entry(id, ctx)).await?;
        Ok(stored.map(|stored| stored.entry))
    }

    /// An entry decoded by `kind`; `None` when absent or of another kind.
    pub async fn typed_entry<D: DeserializeOwned>(
        &mut self,
        kind: &EntryKind<D>,
        id: EntryId,
    ) -> Result<Option<TypedEntry<D>>, TxError> {
        self.check_read("typed_entry")?;
        let ctx = &self.ctx;
        let stored = self.line.guard.read(ctx, |storage| storage.entry(id, ctx)).await?;
        match stored {
            Some(stored) => Ok(kind.decode(stored.entry)?),
            None => Ok(None),
        }
    }

    /// The committed task record.
    pub async fn task(&mut self, id: TaskId) -> Result<Option<TaskRecord>, TxError> {
        self.check_read("task")?;
        self.committed_task(id).await
    }

    /// The committed submission record.
    pub async fn submission(&mut self, id: SubmissionId) -> Result<Option<SubmissionRecord>, TxError> {
        self.check_read("submission")?;
        let ctx = &self.ctx;
        self.line.guard.read(ctx, |storage| storage.submission(id, ctx)).await
    }

    /// One page of committed conversations; `limit` and `cursor` pass through to storage.
    pub async fn scan_conversations(
        &mut self,
        query: ConversationQuery,
        limit: usize,
        cursor: Option<Cursor>,
    ) -> Result<Page<ConversationRecord>, TxError> {
        self.check_read("scan_conversations")?;
        let ctx = &self.ctx;
        self.line
            .guard
            .read(ctx, |storage| storage.scan_conversations(query, limit, cursor, ctx))
            .await
    }

    /// One page of the entries visible from `query.conversation_id`, through its ancestry.
    pub async fn scan_entries(
        &mut self,
        query: EntryQuery,
        limit: usize,
        cursor: Option<Cursor>,
    ) -> Result<Page<EntryRecord>, TxError> {
        self.check_read("scan_entries")?;
        let ctx = &self.ctx;
        self.line
            .guard
            .read(ctx, |storage| storage.scan_entries(query, limit, cursor, ctx))
            .await
    }

    /// One page of committed task records.
    pub async fn scan_tasks(
        &mut self,
        query: TaskQuery,
        limit: usize,
        cursor: Option<Cursor>,
    ) -> Result<Page<TaskRecord>, TxError> {
        self.check_read("scan_tasks")?;
        let ctx = &self.ctx;
        self.line
            .guard
            .read(ctx, |storage| storage.scan_tasks(query, limit, cursor, ctx))
            .await
    }

    /// The newest visible head marker of `conversation`.
    pub async fn latest_head_marker(&mut self, conversation: ConversationId) -> Result<Option<EntryRecord>, TxError> {
        self.check_read("latest_head_marker")?;
        let ctx = &self.ctx;
        self.line
            .guard
            .read(ctx, |storage| storage.find_latest_head_marker(conversation, None, ctx))
            .await
    }

    /// The committed submission with `request_id` in `conversation`.
    pub async fn submission_by_request(
        &mut self,
        conversation: ConversationId,
        request_id: &str,
    ) -> Result<Option<SubmissionRecord>, TxError> {
        self.check_read("submission_by_request")?;
        let ctx = &self.ctx;
        self.line
            .guard
            .read(ctx, |storage| {
                storage.submission_by_request(conversation, request_id, ctx)
            })
            .await
    }

    // ─── Table writes ───────────────────────────────────────────────────────

    /// A new conversation; `ownership` names the owning task, whose candidate gives the
    /// persisted owner edge (validated again at settlement).
    pub async fn create_conversation(
        &mut self,
        ownership: ConversationOwnership,
    ) -> Result<ConversationRecord, TxError> {
        self.stage_conversation(None, ownership, None).await
    }

    /// The reserved root conversation (id 1), ownerless; the layer above bootstraps it once.
    pub(crate) async fn create_root_conversation(&mut self) -> Result<ConversationRecord, TxError> {
        self.stage_conversation(None, ConversationOwnership::Ownerless, Some(ROOT_CONVERSATION_ID))
            .await
    }

    /// A child of `parent` that sees its entries up to `at` and gets copies of its documents
    /// by their fork policy. Refused when `at` is not visible from `parent`, when one child
    /// address would get two copies, and (at settlement) when this transaction writes a
    /// copied source or a `fork: current` document of `parent`.
    pub async fn fork_conversation(
        &mut self,
        parent: ConversationId,
        at: EntryId,
        ownership: ConversationOwnership,
    ) -> Result<ConversationRecord, TxError> {
        let parent = ConversationParent {
            conversation_id: parent,
            at,
        };
        self.stage_conversation(Some(parent), ownership, None).await
    }

    async fn stage_conversation(
        &mut self,
        parent: Option<ConversationParent>,
        ownership: ConversationOwnership,
        reserved: Option<ConversationId>,
    ) -> Result<ConversationRecord, TxError> {
        self.has_table_write = true;
        let id = match reserved {
            Some(id) => id,
            None => self.line.guard.mint().await?,
        };
        let owner = match ownership {
            ConversationOwnership::Ownerless => None,
            ConversationOwnership::Task(task_id) => {
                let task = self.current_task(task_id).await?.ok_or(TxError::UnknownTask(task_id))?;
                Some(ConversationOwner {
                    conversation_id: task.conversation_id,
                    task_id,
                })
            }
        };
        let record = ConversationRecord { id, parent, owner };
        if let Some(parent) = parent {
            let copies = forks::select_copies(&self.line.guard, parent, id, &self.ctx).await?;
            for copy in copies {
                self.fork_source_documents.insert(copy.source.id);
                self.push_document(DocEntry::fork_copy(copy.record, copy.source));
            }
            self.fork_source_conversations.insert(parent.conversation_id);
        }
        self.created_conversations.insert(id);
        self.writes.push(StorageWrite::Conversation(record.clone()));
        if let Some(hook) = self.line.hook.clone() {
            hook.conversation_created(self, &record).await?;
        }
        Ok(record)
    }

    /// Appends an immutable entry with a fresh id; `Head::SelfRef` points at the entry
    /// itself. A scoped commit stamps its task as `by_task_id`.
    pub async fn append_entry(
        &mut self,
        conversation: ConversationId,
        draft: EntryDraft,
    ) -> Result<EntryRecord, TxError> {
        self.has_table_write = true;
        self.require_conversation(conversation).await?;
        let id: EntryId = self.line.guard.mint().await?;
        let record = EntryRecord {
            id,
            conversation_id: conversation,
            kind: draft.kind,
            model: draft.model,
            data: draft.data,
            head: draft.head.map(|head| match head {
                Head::At(entry) => entry,
                Head::SelfRef => id,
            }),
            edits: draft.edits,
            by_task_id: self.scope.task_id,
        };
        self.writes.push(StorageWrite::Entry(record.clone()));
        Ok(record)
    }

    /// [`append_entry`](Self::append_entry) with data encoded by `kind`.
    pub async fn append_typed_entry<D: Serialize>(
        &mut self,
        kind: &EntryKind<D>,
        conversation: ConversationId,
        draft: TypedEntryDraft<D>,
    ) -> Result<TypedEntry<D>, TxError> {
        let data = json::to_json(&draft.data).map_err(|source| RecordError::EntryData {
            kind: kind.kind.to_owned(),
            source,
        })?;
        let untyped = EntryDraft {
            kind: kind.kind.to_owned(),
            model: draft.model,
            data: Some(data),
            head: draft.head,
            edits: draft.edits,
        };
        let record = self.append_entry(conversation, untyped).await?;
        Ok(TypedEntry {
            record,
            data: draft.data,
        })
    }

    /// A pending task with `create`'s definition, input and first checkpoint. The owner is
    /// validated again against its final candidate at settlement.
    pub async fn create_task_record(&mut self, create: TaskCreate, options: TaskOptions) -> Result<TaskId, TxError> {
        self.has_table_write = true;
        let owner = match options.owner {
            Some(owner_id) => {
                let owner = self
                    .current_task(owner_id)
                    .await?
                    .ok_or(TxError::UnknownTask(owner_id))?;
                if options.background {
                    return Err(TxError::InvalidTask("A child task cannot be background".to_owned()));
                }
                if options
                    .conversation_id
                    .is_some_and(|conversation| conversation != owner.conversation_id)
                {
                    return Err(TxError::InvalidTask(format!(
                        "A child task lives in its owner's conversation {}",
                        owner.conversation_id
                    )));
                }
                Some(owner)
            }
            None => None,
        };
        let conversation_id = owner
            .as_ref()
            .map(|owner| owner.conversation_id)
            .or(options.conversation_id)
            .or(self.scope.conversation_id)
            .ok_or_else(|| TxError::InvalidTask("Tx::create_task() requires a conversation".to_owned()))?;
        self.require_conversation(conversation_id).await?;
        let id: TaskId = self.line.guard.mint().await?;
        let mut record = TaskRecord::new(
            id,
            conversation_id,
            create.kind,
            create.version,
            create.input,
            TaskState::Pending {
                checkpoint: create.checkpoint,
            },
        );
        record.owner = owner.map(|owner| owner.id);
        record.background = options.background;
        record.abandon_on_restart = options.abandon_on_restart;
        record.check()?;
        self.tasks.entry(id).or_default().write = Some(TaskWrite { created: true, record });
        Ok(id)
    }

    /// A raw submission record with a fresh id; no admission rules (hosts submit through the
    /// layer above).
    pub async fn create_submission(&mut self, create: SubmissionCreate) -> Result<SubmissionRecord, TxError> {
        self.has_table_write = true;
        self.require_conversation(create.conversation_id).await?;
        let id: SubmissionId = self.line.guard.mint().await?;
        let record = create.into_record(id);
        record.check()?;
        self.submissions.insert(id, record.clone());
        Ok(record)
    }

    /// Settles a submission at settlement, against the transaction's latest candidate record
    /// or the committed one. Only a placed input becomes `done`; a settled record stays.
    /// Counts as a table write; not a caller read.
    pub fn settle_submission(&mut self, id: SubmissionId, settlement: SubmissionSettlement) {
        self.has_table_write = true;
        self.submission_changes.push((id, SubmissionChange::Settle(settlement)));
    }

    /// Places a queued submission at `entry`, resolved like
    /// [`settle_submission`](Self::settle_submission): an input becomes `placed`, a write
    /// `done`.
    pub fn place_submission(&mut self, id: SubmissionId, entry: EntryId) {
        self.has_table_write = true;
        self.submission_changes.push((id, SubmissionChange::Place(entry)));
    }

    /// Replaces a task record whole (the scheduler's transitions). Stamps `started_at` on the
    /// first `running` and `ended_at` on `terminal`; refuses a conversation change and a
    /// second terminal candidate. Validated against the committed record at settlement.
    pub(crate) fn set_task(&mut self, value: TaskRecord) -> Result<(), TxError> {
        self.has_table_write = true;
        let now = (self.line.now)();
        let task = self.tasks.entry(value.id).or_default();
        let candidate = task.write.as_ref().map(|write| &write.record);
        if candidate.is_some_and(|candidate| candidate.state.status() == TaskStatus::Terminal) {
            return Err(TxError::InvalidTask(format!(
                "Task {} already has a terminal candidate",
                value.id
            )));
        }
        if candidate.is_some_and(|candidate| candidate.conversation_id != value.conversation_id) {
            return Err(TxError::TaskConversationChange(value.id));
        }
        let record = stamp_times(value, candidate, now);
        record.check()?;
        let created = task.write.as_ref().is_some_and(|write| write.created);
        task.write = Some(TaskWrite { created, record });
        Ok(())
    }

    /// The candidate records of the tasks this transaction created or replaced so far.
    pub(crate) fn staged_tasks(&self) -> impl Iterator<Item = &TaskRecord> {
        self.tasks
            .values()
            .filter_map(|task| task.write.as_ref().map(|write| &write.record))
    }

    /// The conversations this transaction created or forked so far.
    pub(crate) fn staged_conversations(&self) -> impl Iterator<Item = &ConversationRecord> {
        self.writes.iter().filter_map(|write| match write {
            StorageWrite::Conversation(record) => Some(record),
            _ => None,
        })
    }

    // ─── Committed reads for the scheduler ─────────────────────────────────
    //
    // Not caller reads: allowed after table writes, as the scheduler's bookkeeping reads
    // what is committed while it assembles its own writes.

    /// The committed task record, read once per transaction.
    pub(crate) async fn stored_task(&mut self, id: TaskId) -> Result<Option<TaskRecord>, TxError> {
        self.committed_task(id).await
    }

    /// The committed conversation record.
    pub(crate) async fn stored_conversation(
        &mut self,
        id: ConversationId,
    ) -> Result<Option<ConversationRecord>, TxError> {
        let ctx = &self.ctx;
        self.line.guard.read(ctx, |storage| storage.conversation(id, ctx)).await
    }

    /// Every committed queued input submission, in id order.
    pub(crate) async fn queued_inputs(&mut self) -> Result<Vec<SubmissionRecord>, TxError> {
        const PAGE: usize = 256;
        let query = SubmissionQuery {
            status: Some(SubmissionStatus::Queued),
            ..SubmissionQuery::default()
        };
        let mut inputs = Vec::new();
        let mut cursor = None;
        loop {
            let ctx = &self.ctx;
            let page = self
                .line
                .guard
                .read(ctx, |storage| {
                    storage.scan_submissions(query.clone(), PAGE, cursor, ctx)
                })
                .await?;
            inputs.extend(
                page.items
                    .into_iter()
                    .filter(|record| record.kind == SubmissionType::Input),
            );
            match page.next {
                Some(next) => cursor = Some(next),
                None => return Ok(inputs),
            }
        }
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    pub(super) async fn require_conversation(&mut self, id: ConversationId) -> Result<(), TxError> {
        if self.created_conversations.contains(&id) {
            return Ok(());
        }
        let ctx = &self.ctx;
        let found = self
            .line
            .guard
            .read(ctx, |storage| storage.conversation(id, ctx))
            .await?;
        found.map(|_| ()).ok_or(TxError::UnknownConversation(id))
    }

    /// The latest candidate record, falling back to committed state; not a caller read.
    pub(super) async fn current_task(&mut self, id: TaskId) -> Result<Option<TaskRecord>, TxError> {
        if let Some(write) = self.tasks.get(&id).and_then(|task| task.write.as_ref()) {
            return Ok(Some(write.record.clone()));
        }
        self.committed_task(id).await
    }

    /// The committed record, read once per transaction.
    pub(super) async fn committed_task(&mut self, id: TaskId) -> Result<Option<TaskRecord>, TxError> {
        if let Some(Committed::Read(committed)) = self.tasks.get(&id).map(|task| &task.committed) {
            return Ok(committed.clone());
        }
        let ctx = &self.ctx;
        let record = self.line.guard.read(ctx, |storage| storage.task(id, ctx)).await?;
        self.tasks.entry(id).or_default().committed = Committed::Read(record.clone());
        Ok(record)
    }

    pub(super) fn push_document(&mut self, entry: DocEntry) -> usize {
        let index = self.documents.len();
        self.latest.insert(entry.address.clone(), index);
        self.documents.push(entry);
        index
    }
}

/// Lifecycle times: `started_at` on the first change to `running`, `ended_at` on the change
/// to `terminal`; once set they carry over from the replaced candidate.
fn stamp_times(mut value: TaskRecord, candidate: Option<&TaskRecord>, now: i64) -> TaskRecord {
    let status = value.state.status();
    value.started_at = candidate
        .and_then(|candidate| candidate.started_at)
        .or(value.started_at)
        .or((status == TaskStatus::Running).then_some(now));
    value.ended_at = candidate
        .and_then(|candidate| candidate.ended_at)
        .or(value.ended_at)
        .or((status == TaskStatus::Terminal).then_some(now));
    value
}
