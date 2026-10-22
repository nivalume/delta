//! The memory backend: the reference implementation of the storage contract.

mod tables;

use std::fmt;
use std::future;
use std::sync::{Mutex, MutexGuard, PoisonError};

use delta_chord::context::Context;
use futures::FutureExt;

use self::tables::Tables;
use super::ancestry;
use super::batch::plan;
use super::error::StorageError;
use super::scan::ScanStart;
use super::{Storage, StorageFuture};
use crate::records::{
    ConversationId, ConversationQuery, ConversationRecord, Cursor, DocumentAddress, DocumentId, DocumentPoint,
    DocumentQuery, DocumentRecord, EntryId, EntryQuery, EntryRecord, Page, RawId, RecordId, ScanOrder, Seq,
    StorageWrite, StoredDocument, StoredEntry, SubmissionId, SubmissionQuery, SubmissionRecord, TaskId, TaskQuery,
    TaskRecord,
};

/// Storage held in memory, lost on drop.
///
/// Every call runs synchronously under one lock, so a commit is atomic by construction:
/// it plans the whole batch, copies included, before it changes anything. `next_id` starts
/// at 2 (id 1 is the root conversation), `next_seq` at 1; `u64::MAX` is never minted.
pub struct MemoryStorage {
    inner: Mutex<Inner>,
}

struct Inner {
    tables: Tables,
    next_id: u64,
    next_seq: u64,
    closed: bool,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                tables: Tables::default(),
                next_id: 2,
                next_seq: 1,
                closed: false,
            }),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        // Nothing panics between planning and the end of application, so a poisoned lock
        // still guards consistent tables.
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Runs `f` on the open tables and returns its result as a ready future.
    fn read<T: Send + 'static>(&self, f: impl FnOnce(&Tables) -> Result<T, StorageError>) -> StorageFuture<'_, T> {
        let inner = self.lock();
        let result = if inner.closed {
            Err(StorageError::Closed)
        } else {
            f(&inner.tables)
        };
        future::ready(result).boxed()
    }

    fn commit_now(&self, writes: Vec<StorageWrite>) -> Result<Seq, StorageError> {
        let mut inner = self.lock();
        if inner.closed {
            return Err(StorageError::Closed);
        }
        let batch = plan(writes, &mut &inner.tables)?;
        let seq = Seq(inner.next_seq);
        if let Some(highest) = batch.highest_id {
            inner.next_id = inner.next_id.max(highest.saturating_add(1));
        }
        inner.tables.apply(batch, seq);
        inner.next_seq += 1;
        Ok(seq)
    }

    fn mint_now(&self) -> Result<RawId, StorageError> {
        let mut inner = self.lock();
        if inner.closed {
            return Err(StorageError::Closed);
        }
        if inner.next_id == u64::MAX {
            return Err(StorageError::IdSpaceExhausted);
        }
        let id = inner.next_id;
        inner.next_id += 1;
        Ok(RawId(id))
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for MemoryStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.lock();
        f.debug_struct("MemoryStorage")
            .field("next_id", &inner.next_id)
            .field("next_seq", &inner.next_seq)
            .field("closed", &inner.closed)
            .finish_non_exhaustive()
    }
}

impl Storage for MemoryStorage {
    fn commit(&self, writes: Vec<StorageWrite>, _ctx: &Context) -> StorageFuture<'_, Seq> {
        future::ready(self.commit_now(writes)).boxed()
    }

    fn mint_id(&self) -> StorageFuture<'_, RawId> {
        future::ready(self.mint_now()).boxed()
    }

    fn conversation(&self, id: ConversationId, _ctx: &Context) -> StorageFuture<'_, Option<ConversationRecord>> {
        self.read(|tables| Ok(tables.conversation(id)))
    }

    fn scan_conversations(
        &self,
        query: ConversationQuery,
        limit: usize,
        cursor: Option<Cursor>,
        _ctx: &Context,
    ) -> StorageFuture<'_, Page<ConversationRecord>> {
        self.read(move |tables| {
            let start = ScanStart::new(query.order, cursor.as_ref(), ScanOrder::Ascending, limit)?;
            Ok(start.page(tables.scan_conversations(&query, &start), |record| record.id.get()))
        })
    }

    fn entry(&self, id: EntryId, _ctx: &Context) -> StorageFuture<'_, Option<StoredEntry>> {
        self.read(|tables| Ok(tables.entry(id)))
    }

    fn visible_entry(
        &self,
        conversation: ConversationId,
        id: EntryId,
        _ctx: &Context,
    ) -> StorageFuture<'_, Option<StoredEntry>> {
        self.read(move |tables| ancestry::visible_entry(&mut &*tables, conversation, id))
    }

    fn find_latest_head_marker(
        &self,
        conversation: ConversationId,
        at_or_before: Option<EntryId>,
        _ctx: &Context,
    ) -> StorageFuture<'_, Option<EntryRecord>> {
        self.read(move |tables| ancestry::latest_head(&mut &*tables, conversation, at_or_before))
    }

    fn scan_entries(
        &self,
        query: EntryQuery,
        limit: usize,
        cursor: Option<Cursor>,
        _ctx: &Context,
    ) -> StorageFuture<'_, Page<EntryRecord>> {
        self.read(move |tables| {
            let start = ScanStart::new(query.order, cursor.as_ref(), ScanOrder::Descending, limit)?;
            let entries = ancestry::scan(
                &mut &*tables,
                query.conversation_id,
                query.min_entry_id,
                query.max_entry_id,
                &start,
            )?;
            Ok(start.page(entries, |entry| entry.id.get()))
        })
    }

    fn task(&self, id: TaskId, _ctx: &Context) -> StorageFuture<'_, Option<TaskRecord>> {
        self.read(|tables| Ok(tables.task(id)))
    }

    fn scan_tasks(
        &self,
        query: TaskQuery,
        limit: usize,
        cursor: Option<Cursor>,
        _ctx: &Context,
    ) -> StorageFuture<'_, Page<TaskRecord>> {
        self.read(move |tables| {
            let start = ScanStart::new(query.order, cursor.as_ref(), ScanOrder::Ascending, limit)?;
            Ok(start.page(tables.scan_tasks(&query, &start), |task| task.id.get()))
        })
    }

    fn submission(&self, id: SubmissionId, _ctx: &Context) -> StorageFuture<'_, Option<SubmissionRecord>> {
        self.read(|tables| Ok(tables.submission(id)))
    }

    fn scan_submissions(
        &self,
        query: SubmissionQuery,
        limit: usize,
        cursor: Option<Cursor>,
        _ctx: &Context,
    ) -> StorageFuture<'_, Page<SubmissionRecord>> {
        self.read(move |tables| {
            let start = ScanStart::new(query.order, cursor.as_ref(), ScanOrder::Ascending, limit)?;
            Ok(start.page(tables.scan_submissions(&query, &start), |submission| {
                submission.id.get()
            }))
        })
    }

    fn submission_by_request(
        &self,
        conversation: ConversationId,
        request_id: &str,
        _ctx: &Context,
    ) -> StorageFuture<'_, Option<SubmissionRecord>> {
        self.read(|tables| Ok(tables.submission_by_request(conversation, request_id)))
    }

    fn find_document(
        &self,
        address: &DocumentAddress,
        at: DocumentPoint,
        _ctx: &Context,
    ) -> StorageFuture<'_, Option<DocumentRecord>> {
        self.read(|tables| Ok(tables.find_document(address, at)))
    }

    fn document(&self, id: DocumentId, at: DocumentPoint, _ctx: &Context) -> StorageFuture<'_, Option<StoredDocument>> {
        self.read(|tables| tables.document(id, at))
    }

    fn scan_documents(
        &self,
        query: DocumentQuery,
        limit: usize,
        cursor: Option<Cursor>,
        _ctx: &Context,
    ) -> StorageFuture<'_, Page<DocumentRecord>> {
        self.read(move |tables| {
            let start = ScanStart::new(Some(ScanOrder::Ascending), cursor.as_ref(), ScanOrder::Ascending, limit)?;
            Ok(start.page(tables.scan_documents(&query, &start), |record| record.id.get()))
        })
    }

    fn close(&self, _ctx: &Context) -> StorageFuture<'_, ()> {
        self.lock().closed = true;
        future::ready(Ok(())).boxed()
    }
}
