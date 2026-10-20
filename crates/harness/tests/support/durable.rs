//! A storage that outlives its sessions.

use std::sync::Arc;

use delta_chord::context::Context;
use delta_harness::records::{
    ConversationId, ConversationQuery, ConversationRecord, Cursor, DocumentAddress, DocumentId, DocumentPoint,
    DocumentQuery, DocumentRecord, EntryId, EntryQuery, EntryRecord, Page, RawId, Seq, StorageWrite, StoredDocument,
    StoredEntry, SubmissionId, SubmissionQuery, SubmissionRecord, TaskId, TaskQuery, TaskRecord,
};
use delta_harness::storage::{Storage, StorageFuture};
use futures::FutureExt;

/// Storage that outlives its sessions: `close` does nothing, so a case reopens it after a
/// close or a simulated crash.
pub struct Durable(pub Arc<dyn Storage>);

impl Storage for Durable {
    fn commit(&self, writes: Vec<StorageWrite>, ctx: &Context) -> StorageFuture<'_, Seq> {
        self.0.commit(writes, ctx)
    }
    fn mint_id(&self) -> StorageFuture<'_, RawId> {
        self.0.mint_id()
    }
    fn conversation(&self, id: ConversationId, ctx: &Context) -> StorageFuture<'_, Option<ConversationRecord>> {
        self.0.conversation(id, ctx)
    }
    fn scan_conversations(
        &self,
        query: ConversationQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<ConversationRecord>> {
        self.0.scan_conversations(query, limit, cursor, ctx)
    }
    fn entry(&self, id: EntryId, ctx: &Context) -> StorageFuture<'_, Option<StoredEntry>> {
        self.0.entry(id, ctx)
    }
    fn visible_entry(
        &self,
        conversation: ConversationId,
        id: EntryId,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<StoredEntry>> {
        self.0.visible_entry(conversation, id, ctx)
    }
    fn find_latest_head_marker(
        &self,
        conversation: ConversationId,
        at_or_before: Option<EntryId>,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<EntryRecord>> {
        self.0.find_latest_head_marker(conversation, at_or_before, ctx)
    }
    fn scan_entries(
        &self,
        query: EntryQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<EntryRecord>> {
        self.0.scan_entries(query, limit, cursor, ctx)
    }
    fn task(&self, id: TaskId, ctx: &Context) -> StorageFuture<'_, Option<TaskRecord>> {
        self.0.task(id, ctx)
    }
    fn scan_tasks(
        &self,
        query: TaskQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<TaskRecord>> {
        self.0.scan_tasks(query, limit, cursor, ctx)
    }
    fn submission(&self, id: SubmissionId, ctx: &Context) -> StorageFuture<'_, Option<SubmissionRecord>> {
        self.0.submission(id, ctx)
    }
    fn scan_submissions(
        &self,
        query: SubmissionQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<SubmissionRecord>> {
        self.0.scan_submissions(query, limit, cursor, ctx)
    }
    fn submission_by_request(
        &self,
        conversation: ConversationId,
        request_id: &str,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<SubmissionRecord>> {
        self.0.submission_by_request(conversation, request_id, ctx)
    }
    fn find_document(
        &self,
        address: &DocumentAddress,
        at: DocumentPoint,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<DocumentRecord>> {
        self.0.find_document(address, at, ctx)
    }
    fn document(&self, id: DocumentId, at: DocumentPoint, ctx: &Context) -> StorageFuture<'_, Option<StoredDocument>> {
        self.0.document(id, at, ctx)
    }
    fn scan_documents(
        &self,
        query: DocumentQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<DocumentRecord>> {
        self.0.scan_documents(query, limit, cursor, ctx)
    }
    fn close(&self, _ctx: &Context) -> StorageFuture<'_, ()> {
        futures::future::ready(Ok(())).boxed()
    }
}
