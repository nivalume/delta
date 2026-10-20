//! Kept context ranges: extend, shrink, head change and retention expiry, counted on the
//! entries storage scans.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use delta_ai::types::Message;
use delta_chord::context::Context;
use delta_harness::context::ContextRanges;
use delta_harness::records::{
    ConversationId, ConversationOwnership, ConversationQuery, ConversationRecord, Cursor, DocumentAddress, DocumentId,
    DocumentPoint, DocumentQuery, DocumentRecord, EntryDraft, EntryId, EntryQuery, EntryRecord, Head, Page, RawId, Seq,
    StorageWrite, StoredDocument, StoredEntry, SubmissionId, SubmissionQuery, SubmissionRecord, TaskId, TaskQuery,
    TaskRecord,
};
use delta_harness::session::{Session, SessionOptions};
use delta_harness::storage::{Storage, StorageFuture};
use futures::FutureExt;

use crate::Backend;
use crate::support::fixture::ctx;

cases!(a_later_read_scans_only_the_added_entries, retention_drops_unused_ranges);

/// Storage that counts the entries its range scans return.
struct Counting {
    inner: Arc<dyn Storage>,
    scanned: AtomicUsize,
}

impl Counting {
    fn take(&self) -> usize {
        self.scanned.swap(0, Ordering::SeqCst)
    }
}

impl Storage for Counting {
    fn commit(&self, writes: Vec<StorageWrite>, ctx: &Context) -> StorageFuture<'_, Seq> {
        self.inner.commit(writes, ctx)
    }
    fn mint_id(&self) -> StorageFuture<'_, RawId> {
        self.inner.mint_id()
    }
    fn conversation(&self, id: ConversationId, ctx: &Context) -> StorageFuture<'_, Option<ConversationRecord>> {
        self.inner.conversation(id, ctx)
    }
    fn scan_conversations(
        &self,
        query: ConversationQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<ConversationRecord>> {
        self.inner.scan_conversations(query, limit, cursor, ctx)
    }
    fn entry(&self, id: EntryId, ctx: &Context) -> StorageFuture<'_, Option<StoredEntry>> {
        self.inner.entry(id, ctx)
    }
    fn visible_entry(
        &self,
        conversation: ConversationId,
        id: EntryId,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<StoredEntry>> {
        self.inner.visible_entry(conversation, id, ctx)
    }
    fn find_latest_head_marker(
        &self,
        conversation: ConversationId,
        at_or_before: Option<EntryId>,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<EntryRecord>> {
        self.inner.find_latest_head_marker(conversation, at_or_before, ctx)
    }
    fn scan_entries(
        &self,
        query: EntryQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<EntryRecord>> {
        // The tail capture reads one entry; only range scans count.
        let counted = limit > 1;
        let scan = self.inner.scan_entries(query, limit, cursor, ctx);
        async move {
            let page = scan.await?;
            if counted {
                self.scanned.fetch_add(page.items.len(), Ordering::SeqCst);
            }
            Ok(page)
        }
        .boxed()
    }
    fn task(&self, id: TaskId, ctx: &Context) -> StorageFuture<'_, Option<TaskRecord>> {
        self.inner.task(id, ctx)
    }
    fn scan_tasks(
        &self,
        query: TaskQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<TaskRecord>> {
        self.inner.scan_tasks(query, limit, cursor, ctx)
    }
    fn submission(&self, id: SubmissionId, ctx: &Context) -> StorageFuture<'_, Option<SubmissionRecord>> {
        self.inner.submission(id, ctx)
    }
    fn scan_submissions(
        &self,
        query: SubmissionQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<SubmissionRecord>> {
        self.inner.scan_submissions(query, limit, cursor, ctx)
    }
    fn submission_by_request(
        &self,
        conversation: ConversationId,
        request_id: &str,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<SubmissionRecord>> {
        self.inner.submission_by_request(conversation, request_id, ctx)
    }
    fn find_document(
        &self,
        address: &DocumentAddress,
        at: DocumentPoint,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<DocumentRecord>> {
        self.inner.find_document(address, at, ctx)
    }
    fn document(&self, id: DocumentId, at: DocumentPoint, ctx: &Context) -> StorageFuture<'_, Option<StoredDocument>> {
        self.inner.document(id, at, ctx)
    }
    fn scan_documents(
        &self,
        query: DocumentQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<DocumentRecord>> {
        self.inner.scan_documents(query, limit, cursor, ctx)
    }
    fn close(&self, ctx: &Context) -> StorageFuture<'_, ()> {
        self.inner.close(ctx)
    }
}

struct Setup {
    session: Session,
    storage: Arc<Counting>,
    conversation: ConversationId,
    _fixture: crate::support::Fixture,
}

async fn setup(backend: Backend) -> Setup {
    let fixture = backend.open().await;
    let storage = Arc::new(Counting {
        inner: fixture.storage.clone(),
        scanned: AtomicUsize::new(0),
    });
    let session = Session::open(storage.clone(), SessionOptions::default());
    let conversation = session
        .commit(&ctx(), |tx| {
            Box::pin(async move { Ok(tx.create_conversation(ConversationOwnership::Ownerless).await?.id) })
        })
        .await
        .unwrap();
    Setup {
        session,
        storage,
        conversation,
        _fixture: fixture,
    }
}

impl Setup {
    async fn append(&self, draft: EntryDraft) -> EntryId {
        let conversation = self.conversation;
        self.session
            .commit(&ctx(), move |tx| {
                Box::pin(async move { Ok(tx.append_entry(conversation, draft).await?.id) })
            })
            .await
            .unwrap()
    }

    async fn say(&self, text: &str) -> EntryId {
        self.append(EntryDraft {
            model: Some(vec![Message::user(text)]),
            ..EntryDraft::new("delta.user")
        })
        .await
    }

    async fn read(&self, ranges: &ContextRanges, at: Option<EntryId>, retention: Duration) -> Vec<String> {
        let view = ranges
            .read(&*self.storage, self.conversation, at, retention, &ctx())
            .await
            .unwrap();
        view.messages
            .iter()
            .filter_map(|message| match message {
                Message::User(user) => Some(user.content.text()),
                _ => None,
            })
            .collect()
    }
}

const KEEP: Duration = Duration::from_mins(10);

async fn a_later_read_scans_only_the_added_entries(backend: Backend) {
    let setup = setup(backend).await;
    let ranges = ContextRanges::new();
    let first = setup.say("one").await;
    setup.say("two").await;
    assert_eq!(setup.read(&ranges, None, KEEP).await, ["one", "two"]);
    assert_eq!(setup.storage.take(), 2);
    assert_eq!(ranges.len(), 1);
    // Unchanged: nothing is scanned.
    assert_eq!(setup.read(&ranges, None, KEEP).await, ["one", "two"]);
    assert_eq!(setup.storage.take(), 0);
    // Extended: only the added entry.
    setup.say("three").await;
    assert_eq!(setup.read(&ranges, None, KEEP).await, ["one", "two", "three"]);
    assert_eq!(setup.storage.take(), 1);
    // An added edit changes an earlier contribution; the kept entries are derived again.
    setup
        .append(EntryDraft {
            edits: Some(vec![delta_harness::records::ContextEdit {
                target: first,
                action: delta_harness::records::EditAction::Omit,
            }]),
            ..EntryDraft::new("app.edit")
        })
        .await;
    assert_eq!(setup.read(&ranges, None, KEEP).await, ["two", "three"]);
    assert_eq!(setup.storage.take(), 1);
    // A shorter tail filters the kept entries without a scan.
    assert_eq!(setup.read(&ranges, Some(first), KEEP).await, ["one"]);
    assert_eq!(setup.storage.take(), 0);
    // A new head marker rescans from its head.
    setup
        .append(EntryDraft {
            model: Some(vec![Message::user("summary")]),
            head: Some(Head::SelfRef),
            ..EntryDraft::new("app.summary")
        })
        .await;
    assert_eq!(setup.read(&ranges, None, KEEP).await, ["summary"]);
    assert_eq!(setup.storage.take(), 1);
    setup.session.close(&ctx()).await.unwrap();
}

async fn retention_drops_unused_ranges(backend: Backend) {
    let setup = setup(backend).await;
    let ranges = ContextRanges::new();
    setup.say("one").await;
    setup.read(&ranges, None, KEEP).await;
    assert_eq!(setup.storage.take(), 1);
    // With no retention, the kept range has expired by the next read.
    setup.read(&ranges, None, Duration::ZERO).await;
    assert_eq!(setup.storage.take(), 1);
    setup.session.close(&ctx()).await.unwrap();
}
