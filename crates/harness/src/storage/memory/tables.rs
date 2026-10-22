//! The memory backend's tables and secondary indexes, their reads and batch application.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Bound;

use crate::records::{
    ConversationId, ConversationQuery, ConversationRecord, DocumentAddress, DocumentContent, DocumentCopySource,
    DocumentId, DocumentPoint, DocumentQuery, DocumentRecord, DocumentScope, EntryId, EntryRecord, RecordId, ScanOrder,
    Seq, StoredDocument, StoredEntry, SubmissionId, SubmissionQuery, SubmissionRecord, SubmissionStatus, TaskId,
    TaskQuery, TaskRecord, TaskStatus,
};
use crate::storage::ancestry::{EntryReader, Lineage};
use crate::storage::batch::{Batch, CommitView, DocumentAction, ExistingDocument, RecordType, RecordWrite, Target};
use crate::storage::error::StorageError;
use crate::storage::replay::{self, materialize};
use crate::storage::scan::ScanStart;

/// One incarnation with its revisions in sequence order.
#[derive(Debug)]
struct Incarnation {
    record: DocumentRecord,
    revisions: Vec<(Seq, DocumentContent)>,
}

/// Every incarnation of one address and the current one.
#[derive(Debug, Default)]
struct AddressIndex {
    ids: BTreeSet<DocumentId>,
    current: Option<DocumentId>,
}

/// The committed state: tables by id (ordered scans fall out) and the indexes upstream keeps.
#[derive(Debug, Default)]
pub(super) struct Tables {
    record_types: HashMap<u64, RecordType>,
    conversations: BTreeMap<ConversationId, ConversationRecord>,
    conversations_by_owner_conversation: HashMap<ConversationId, BTreeSet<ConversationId>>,
    conversations_by_owner_task: HashMap<TaskId, BTreeSet<ConversationId>>,
    entries: HashMap<EntryId, StoredEntry>,
    entries_by_conversation: HashMap<ConversationId, BTreeSet<EntryId>>,
    heads_by_conversation: HashMap<ConversationId, BTreeSet<EntryId>>,
    tasks: BTreeMap<TaskId, TaskRecord>,
    tasks_by_status: HashMap<TaskStatus, BTreeSet<TaskId>>,
    submissions: BTreeMap<SubmissionId, SubmissionRecord>,
    submissions_by_status: HashMap<SubmissionStatus, BTreeSet<SubmissionId>>,
    submissions_by_request: HashMap<(ConversationId, String), SubmissionId>,
    documents: HashMap<DocumentId, Incarnation>,
    documents_by_address: HashMap<DocumentAddress, AddressIndex>,
    documents_by_scope: HashMap<DocumentScope, BTreeSet<DocumentId>>,
}

/// The bound a cursor puts on the side the scan moves toward: the last id it returned,
/// excluded.
fn cursor_bound<I: RecordId>(start: &ScanStart) -> Bound<I> {
    start
        .after
        .map_or(Bound::Unbounded, |after| Bound::Excluded(I::from_raw(after)))
}

/// Ids of `ids` in scan order after the start's cursor.
fn scan_ids<'a, I: RecordId>(ids: Option<&'a BTreeSet<I>>, start: &ScanStart) -> Ids<'a, I> {
    let Some(ids) = ids else {
        return Box::new(std::iter::empty());
    };
    let after = cursor_bound::<I>(start);
    match start.order {
        ScanOrder::Ascending => Box::new(ids.range((after, Bound::Unbounded)).copied()),
        ScanOrder::Descending => Box::new(ids.range((Bound::Unbounded, after)).rev().copied()),
    }
}

/// A lazy id sequence in scan order.
type Ids<'a, I> = Box<dyn Iterator<Item = I> + 'a>;

/// Values of `table` in scan order after the start's cursor (or of `ids`, an index already
/// positioned), filtered, at most `fetch`.
fn scan_table<I: RecordId, V: Clone>(
    table: &BTreeMap<I, V>,
    ids: Option<Ids<'_, I>>,
    start: &ScanStart,
    keep: impl Fn(&V) -> bool,
) -> Vec<V> {
    let ids = ids.unwrap_or_else(|| {
        let after = cursor_bound::<I>(start);
        match start.order {
            ScanOrder::Ascending => Box::new(table.range((after, Bound::Unbounded)).map(|(id, _)| *id)),
            ScanOrder::Descending => Box::new(table.range((Bound::Unbounded, after)).rev().map(|(id, _)| *id)),
        }
    });
    ids.filter_map(|id| table.get(&id))
        .filter(|value| keep(value))
        .take(start.fetch)
        .cloned()
        .collect()
}

fn insert_index<K: std::hash::Hash + Eq, I: Ord>(index: &mut HashMap<K, BTreeSet<I>>, key: K, id: I) {
    index.entry(key).or_default().insert(id);
}

fn remove_index<K: std::hash::Hash + Eq, I: Ord>(index: &mut HashMap<K, BTreeSet<I>>, key: &K, id: &I) {
    if let Some(ids) = index.get_mut(key) {
        ids.remove(id);
        if ids.is_empty() {
            index.remove(key);
        }
    }
}

impl Tables {
    pub(super) fn conversation(&self, id: ConversationId) -> Option<ConversationRecord> {
        self.conversations.get(&id).cloned()
    }

    pub(super) fn scan_conversations(&self, query: &ConversationQuery, start: &ScanStart) -> Vec<ConversationRecord> {
        let ids = match (query.owner_task_id, query.owner_conversation_id) {
            (Some(task), _) => Some(scan_ids(self.conversations_by_owner_task.get(&task), start)),
            (None, Some(conversation)) => Some(scan_ids(
                self.conversations_by_owner_conversation.get(&conversation),
                start,
            )),
            (None, None) => None,
        };
        scan_table(&self.conversations, ids, start, |record| {
            query
                .owner_conversation_id
                .is_none_or(|owner| record.owner.is_some_and(|edge| edge.conversation_id == owner))
                && query
                    .owner_task_id
                    .is_none_or(|owner| record.owner.is_some_and(|edge| edge.task_id == owner))
        })
    }

    pub(super) fn entry(&self, id: EntryId) -> Option<StoredEntry> {
        self.entries.get(&id).cloned()
    }

    pub(super) fn task(&self, id: TaskId) -> Option<TaskRecord> {
        self.tasks.get(&id).cloned()
    }

    pub(super) fn scan_tasks(&self, query: &TaskQuery, start: &ScanStart) -> Vec<TaskRecord> {
        let ids = query
            .status
            .map(|status| scan_ids(self.tasks_by_status.get(&status), start));
        scan_table(&self.tasks, ids, start, |task| {
            query.conversation_id.is_none_or(|id| task.conversation_id == id)
                && query.kind.as_ref().is_none_or(|kind| &task.kind == kind)
                && query
                    .abort_requested
                    .is_none_or(|marked| task.abort_requested == marked)
                && query.background.is_none_or(|background| task.background == background)
        })
    }

    pub(super) fn submission(&self, id: SubmissionId) -> Option<SubmissionRecord> {
        self.submissions.get(&id).cloned()
    }

    pub(super) fn scan_submissions(&self, query: &SubmissionQuery, start: &ScanStart) -> Vec<SubmissionRecord> {
        let ids = query
            .status
            .map(|status| scan_ids(self.submissions_by_status.get(&status), start));
        scan_table(&self.submissions, ids, start, |submission| {
            query.conversation_id.is_none_or(|id| submission.conversation_id == id)
        })
    }

    pub(super) fn submission_by_request(
        &self,
        conversation: ConversationId,
        request_id: &str,
    ) -> Option<SubmissionRecord> {
        let id = self
            .submissions_by_request
            .get(&(conversation, request_id.to_owned()))?;
        self.submissions.get(id).cloned()
    }

    pub(super) fn find_document(&self, address: &DocumentAddress, at: DocumentPoint) -> Option<DocumentRecord> {
        let index = self.documents_by_address.get(address)?;
        let id = match at {
            DocumentPoint::Current => index.current?,
            DocumentPoint::At(_) => *index.ids.iter().find(|id| {
                self.documents
                    .get(id)
                    .is_some_and(|stored| stored.record.is_alive_at(at))
            })?,
        };
        self.documents.get(&id).map(|stored| stored.record.clone())
    }

    pub(super) fn document(&self, id: DocumentId, at: DocumentPoint) -> Result<Option<StoredDocument>, StorageError> {
        let Some(stored) = self.documents.get(&id) else {
            return Ok(None);
        };
        if matches!(at, DocumentPoint::At(_)) && !stored.record.keeps_history() {
            return Err(replay::no_history(id));
        }
        if !stored.record.is_alive_at(at) {
            return Ok(None);
        }
        let visible = match at {
            DocumentPoint::Current => stored.revisions.len(),
            DocumentPoint::At(at) => stored.revisions.partition_point(|(seq, _)| *seq <= at),
        };
        let revisions = &stored.revisions[..visible];
        let Some(base) = revisions
            .iter()
            .rposition(|(_, content)| matches!(content, DocumentContent::Base { .. }))
        else {
            return Err(replay::missing_base(id));
        };
        let DocumentContent::Base { version, value } = &revisions[base].1 else {
            return Err(replay::missing_base(id));
        };
        let tail = revisions[base + 1..].iter().map(|(seq, content)| Ok((*seq, content)));
        materialize(stored.record.clone(), *version, value.clone(), tail).map(Some)
    }

    pub(super) fn scan_documents(&self, query: &DocumentQuery, start: &ScanStart) -> Vec<DocumentRecord> {
        scan_ids(self.documents_by_scope.get(&query.scope), start)
            .filter_map(|id| self.documents.get(&id))
            .map(|stored| &stored.record)
            .filter(|record| {
                query.kind.as_ref().is_none_or(|kind| &record.kind == kind) && record.is_alive_at(query.at)
            })
            .take(start.fetch)
            .cloned()
            .collect()
    }

    /// Applies a planned batch at `seq`. Infallible: planning checked everything.
    pub(super) fn apply(&mut self, batch: Batch, seq: Seq) {
        for write in batch.records {
            match write {
                RecordWrite::Conversation(record) => self.put_conversation(record),
                RecordWrite::Entry(entry) => self.put_entry(entry, seq),
                RecordWrite::Task(task) => self.put_task(task),
                RecordWrite::Submission(submission) => self.put_submission(submission),
            }
        }
        for action in batch.documents {
            self.apply_document(action, seq);
        }
    }

    fn put_conversation(&mut self, record: ConversationRecord) {
        self.record_types.insert(record.id.0, RecordType::Conversation);
        if let Some(owner) = record.owner {
            insert_index(
                &mut self.conversations_by_owner_conversation,
                owner.conversation_id,
                record.id,
            );
            insert_index(&mut self.conversations_by_owner_task, owner.task_id, record.id);
        }
        self.conversations.insert(record.id, record);
    }

    fn put_entry(&mut self, entry: EntryRecord, seq: Seq) {
        self.record_types.insert(entry.id.0, RecordType::Entry);
        insert_index(&mut self.entries_by_conversation, entry.conversation_id, entry.id);
        if entry.head.is_some() {
            insert_index(&mut self.heads_by_conversation, entry.conversation_id, entry.id);
        }
        self.entries.insert(entry.id, StoredEntry { entry, commit_seq: seq });
    }

    fn put_task(&mut self, task: TaskRecord) {
        self.record_types.insert(task.id.0, RecordType::Task);
        if let Some(previous) = self.tasks.get(&task.id) {
            remove_index(&mut self.tasks_by_status, &previous.state.status(), &task.id);
        }
        insert_index(&mut self.tasks_by_status, task.state.status(), task.id);
        self.tasks.insert(task.id, task);
    }

    fn put_submission(&mut self, submission: SubmissionRecord) {
        self.record_types.insert(submission.id.0, RecordType::Submission);
        if let Some(previous) = self.submissions.get(&submission.id) {
            remove_index(
                &mut self.submissions_by_status,
                &previous.state.status(),
                &submission.id,
            );
            if let Some(request_id) = &previous.request_id {
                self.submissions_by_request
                    .remove(&(previous.conversation_id, request_id.clone()));
            }
        }
        insert_index(
            &mut self.submissions_by_status,
            submission.state.status(),
            submission.id,
        );
        if let Some(request_id) = &submission.request_id {
            self.submissions_by_request
                .insert((submission.conversation_id, request_id.clone()), submission.id);
        }
        self.submissions.insert(submission.id, submission);
    }

    fn apply_document(&mut self, action: DocumentAction, seq: Seq) {
        let id = action.id;
        let record = action.record_at(seq);
        let address = record.address();
        let keeps_history = record.keeps_history();
        match action.target {
            Target::Created(_) => {
                self.record_types.insert(id.0, RecordType::Document);
                insert_index(&mut self.documents_by_scope, record.scope, id);
                let index = self.documents_by_address.entry(address).or_default();
                index.ids.insert(id);
                if !action.retire {
                    index.current = Some(id);
                }
                let revisions = action.content.map(|content| vec![(seq, content)]).unwrap_or_default();
                self.documents.insert(id, Incarnation { record, revisions });
            }
            Target::Existing(_) => {
                if action.retire
                    && let Some(index) = self.documents_by_address.get_mut(&address)
                    && index.current == Some(id)
                {
                    index.current = None;
                }
                let Some(stored) = self.documents.get_mut(&id) else {
                    return;
                };
                if let Some(content) = action.content {
                    if matches!(content, DocumentContent::Base { .. }) && !keeps_history {
                        stored.revisions.clear();
                    }
                    stored.revisions.push((seq, content));
                }
                stored.record = record;
            }
        }
        if action.retire
            && !keeps_history
            && let Some(stored) = self.documents.get_mut(&id)
        {
            stored.revisions.clear();
        }
    }
}

impl EntryReader for &Tables {
    fn lineage(&mut self, conversation: ConversationId) -> Result<Lineage, StorageError> {
        Ok(Lineage::of(self.conversations.get(&conversation)))
    }

    fn entry(&mut self, id: EntryId) -> Result<Option<StoredEntry>, StorageError> {
        Ok(Tables::entry(self, id))
    }

    fn entries(
        &mut self,
        conversation: ConversationId,
        min: Option<EntryId>,
        max: Option<EntryId>,
        order: ScanOrder,
        limit: usize,
    ) -> Result<Vec<EntryRecord>, StorageError> {
        let Some(ids) = self.entries_by_conversation.get(&conversation) else {
            return Ok(Vec::new());
        };
        let range = (
            min.map_or(Bound::Unbounded, Bound::Included),
            max.map_or(Bound::Unbounded, Bound::Included),
        );
        let ids = ids.range(range);
        let ids: Box<dyn Iterator<Item = &EntryId>> = match order {
            ScanOrder::Ascending => Box::new(ids),
            ScanOrder::Descending => Box::new(ids.rev()),
        };
        Ok(ids
            .filter_map(|id| self.entries.get(id))
            .take(limit)
            .map(|stored| stored.entry.clone())
            .collect())
    }

    fn latest_head(
        &mut self,
        conversation: ConversationId,
        max: Option<EntryId>,
    ) -> Result<Option<EntryRecord>, StorageError> {
        let Some(ids) = self.heads_by_conversation.get(&conversation) else {
            return Ok(None);
        };
        let range = (Bound::Unbounded, max.map_or(Bound::Unbounded, Bound::Included));
        Ok(ids
            .range(range)
            .next_back()
            .and_then(|id| self.entries.get(id))
            .map(|stored| stored.entry.clone()))
    }
}

impl CommitView for &Tables {
    fn record_type(&mut self, id: u64) -> Result<Option<RecordType>, StorageError> {
        Ok(self.record_types.get(&id).copied())
    }

    fn document(&mut self, id: DocumentId) -> Result<Option<ExistingDocument>, StorageError> {
        Ok(self.documents.get(&id).map(|stored| ExistingDocument {
            record: stored.record.clone(),
            newest_version: stored.revisions.last().map(|(_, content)| content.version()),
        }))
    }

    fn current_document(&mut self, address: &DocumentAddress) -> Result<Option<DocumentId>, StorageError> {
        Ok(self.documents_by_address.get(address).and_then(|index| index.current))
    }

    fn materialize(&mut self, source: DocumentCopySource) -> Result<Option<StoredDocument>, StorageError> {
        Tables::document(self, source.id, source.at)
    }

    fn submission_by_request(
        &mut self,
        conversation: ConversationId,
        request_id: &str,
    ) -> Result<Option<SubmissionId>, StorageError> {
        Ok(self
            .submissions_by_request
            .get(&(conversation, request_id.to_owned()))
            .copied())
    }
}
