//! A JSONL storage that reopens its directory after every commit, so the conformance cases
//! also check what recovery rebuilds (upstream's "JsonlStorage across reopen").

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use delta_chord::context::Context;
use delta_harness::records::{
    ConversationId, ConversationQuery, ConversationRecord, Cursor, DocumentAddress, DocumentId, DocumentPoint,
    DocumentQuery, DocumentRecord, EntryId, EntryQuery, EntryRecord, Page, RawId, Seq, StorageWrite, StoredDocument,
    StoredEntry, SubmissionId, SubmissionQuery, SubmissionRecord, TaskId, TaskQuery, TaskRecord,
};
use delta_harness::storage::{JsonlOptions, JsonlStorage, Storage, StorageFuture};
use futures::FutureExt;

pub struct ReopeningJsonl {
    dir: std::path::PathBuf,
    current: Mutex<Arc<JsonlStorage>>,
    closed: AtomicBool,
}

impl ReopeningJsonl {
    pub async fn open(dir: &std::path::Path) -> Self {
        let storage = JsonlStorage::open_local(dir, JsonlOptions::default()).await.unwrap();
        Self {
            dir: dir.to_owned(),
            current: Mutex::new(Arc::new(storage)),
            closed: AtomicBool::new(false),
        }
    }

    fn current(&self) -> Arc<JsonlStorage> {
        Arc::clone(&self.current.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

impl Storage for ReopeningJsonl {
    fn commit(&self, writes: Vec<StorageWrite>, ctx: &Context) -> StorageFuture<'_, Seq> {
        let ctx = ctx.clone();
        async move {
            let storage = self.current();
            let result = storage.commit(writes, &ctx).await;
            if !self.closed.load(Ordering::SeqCst) {
                storage.close(&ctx).await?;
                let reopened = JsonlStorage::open_local(&self.dir, JsonlOptions::default()).await?;
                *self.current.lock().unwrap_or_else(PoisonError::into_inner) = Arc::new(reopened);
            }
            result
        }
        .boxed()
    }

    fn mint_id(&self) -> StorageFuture<'_, RawId> {
        let storage = self.current();
        async move { storage.mint_id().await }.boxed()
    }

    fn conversation(&self, id: ConversationId, ctx: &Context) -> StorageFuture<'_, Option<ConversationRecord>> {
        let (storage, ctx) = (self.current(), ctx.clone());
        async move { storage.conversation(id, &ctx).await }.boxed()
    }

    fn scan_conversations(
        &self,
        query: ConversationQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<ConversationRecord>> {
        let (storage, ctx) = (self.current(), ctx.clone());
        async move { storage.scan_conversations(query, limit, cursor, &ctx).await }.boxed()
    }

    fn entry(&self, id: EntryId, ctx: &Context) -> StorageFuture<'_, Option<StoredEntry>> {
        let (storage, ctx) = (self.current(), ctx.clone());
        async move { storage.entry(id, &ctx).await }.boxed()
    }

    fn visible_entry(
        &self,
        conversation: ConversationId,
        id: EntryId,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<StoredEntry>> {
        let (storage, ctx) = (self.current(), ctx.clone());
        async move { storage.visible_entry(conversation, id, &ctx).await }.boxed()
    }

    fn find_latest_head_marker(
        &self,
        conversation: ConversationId,
        at_or_before: Option<EntryId>,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<EntryRecord>> {
        let (storage, ctx) = (self.current(), ctx.clone());
        async move { storage.find_latest_head_marker(conversation, at_or_before, &ctx).await }.boxed()
    }

    fn scan_entries(
        &self,
        query: EntryQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<EntryRecord>> {
        let (storage, ctx) = (self.current(), ctx.clone());
        async move { storage.scan_entries(query, limit, cursor, &ctx).await }.boxed()
    }

    fn task(&self, id: TaskId, ctx: &Context) -> StorageFuture<'_, Option<TaskRecord>> {
        let (storage, ctx) = (self.current(), ctx.clone());
        async move { storage.task(id, &ctx).await }.boxed()
    }

    fn scan_tasks(
        &self,
        query: TaskQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<TaskRecord>> {
        let (storage, ctx) = (self.current(), ctx.clone());
        async move { storage.scan_tasks(query, limit, cursor, &ctx).await }.boxed()
    }

    fn submission(&self, id: SubmissionId, ctx: &Context) -> StorageFuture<'_, Option<SubmissionRecord>> {
        let (storage, ctx) = (self.current(), ctx.clone());
        async move { storage.submission(id, &ctx).await }.boxed()
    }

    fn scan_submissions(
        &self,
        query: SubmissionQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<SubmissionRecord>> {
        let (storage, ctx) = (self.current(), ctx.clone());
        async move { storage.scan_submissions(query, limit, cursor, &ctx).await }.boxed()
    }

    fn submission_by_request(
        &self,
        conversation: ConversationId,
        request_id: &str,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<SubmissionRecord>> {
        let (storage, ctx, request_id) = (self.current(), ctx.clone(), request_id.to_owned());
        async move { storage.submission_by_request(conversation, &request_id, &ctx).await }.boxed()
    }

    fn find_document(
        &self,
        address: &DocumentAddress,
        at: DocumentPoint,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<DocumentRecord>> {
        let (storage, ctx, address) = (self.current(), ctx.clone(), address.clone());
        async move { storage.find_document(&address, at, &ctx).await }.boxed()
    }

    fn document(&self, id: DocumentId, at: DocumentPoint, ctx: &Context) -> StorageFuture<'_, Option<StoredDocument>> {
        let (storage, ctx) = (self.current(), ctx.clone());
        async move { storage.document(id, at, &ctx).await }.boxed()
    }

    fn scan_documents(
        &self,
        query: DocumentQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<DocumentRecord>> {
        let (storage, ctx) = (self.current(), ctx.clone());
        async move { storage.scan_documents(query, limit, cursor, &ctx).await }.boxed()
    }

    fn close(&self, ctx: &Context) -> StorageFuture<'_, ()> {
        self.closed.store(true, Ordering::SeqCst);
        let (storage, ctx) = (self.current(), ctx.clone());
        async move { storage.close(&ctx).await }.boxed()
    }
}
