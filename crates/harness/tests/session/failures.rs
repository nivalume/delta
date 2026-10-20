//! The fatal-error model and close (upstream `harness-storage-failure.test.ts`, the
//! failure and close cases of `session-documents.test.ts`).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use delta_chord::json;
use delta_harness::documents::DocAddress;
use delta_harness::records::{
    ConversationQuery, ConversationRecord, Cursor, DocumentAddress, DocumentId, DocumentPoint, DocumentQuery,
    DocumentRecord, EntryQuery, Page, RawId, Seq, StoredDocument, StoredEntry, SubmissionId, SubmissionQuery,
    SubmissionRecord, TaskQuery,
};
use delta_harness::session::{ReportInfo, SessionEnd, TxError, WatchEnd};
use delta_harness::storage::{MemoryStorage, StorageError, StorageFuture};
use futures::FutureExt;

use super::*;

/// Memory storage whose commits, reads or close fail on demand.
#[derive(Default)]
struct Failing {
    inner: MemoryStorage,
    commits: AtomicBool,
    reads: AtomicBool,
    close_fails: AtomicBool,
    closed: AtomicBool,
}

impl Failing {
    fn read_error(&self) -> Option<StorageError> {
        self.reads
            .load(Ordering::SeqCst)
            .then(|| StorageError::Corrupt("injected read failure".to_owned()))
    }

    fn guard<'a, T: Send + 'a>(&'a self, call: StorageFuture<'a, T>) -> StorageFuture<'a, T> {
        match self.read_error() {
            Some(error) => async move { Err(error) }.boxed(),
            None => call,
        }
    }
}

impl Storage for Failing {
    fn commit(&self, writes: Vec<StorageWrite>, ctx: &Context) -> StorageFuture<'_, Seq> {
        if self.commits.load(Ordering::SeqCst) {
            return async { Err(StorageError::Corrupt("injected commit failure".to_owned())) }.boxed();
        }
        self.inner.commit(writes, ctx)
    }
    fn mint_id(&self) -> StorageFuture<'_, RawId> {
        self.inner.mint_id()
    }
    fn conversation(&self, id: ConversationId, ctx: &Context) -> StorageFuture<'_, Option<ConversationRecord>> {
        self.guard(self.inner.conversation(id, ctx))
    }
    fn scan_conversations(
        &self,
        query: ConversationQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<ConversationRecord>> {
        self.guard(self.inner.scan_conversations(query, limit, cursor, ctx))
    }
    fn entry(&self, id: delta_harness::records::EntryId, ctx: &Context) -> StorageFuture<'_, Option<StoredEntry>> {
        self.guard(self.inner.entry(id, ctx))
    }
    fn visible_entry(
        &self,
        conversation: ConversationId,
        id: delta_harness::records::EntryId,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<StoredEntry>> {
        self.guard(self.inner.visible_entry(conversation, id, ctx))
    }
    fn find_latest_head_marker(
        &self,
        conversation: ConversationId,
        at_or_before: Option<delta_harness::records::EntryId>,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<EntryRecord>> {
        self.guard(self.inner.find_latest_head_marker(conversation, at_or_before, ctx))
    }
    fn scan_entries(
        &self,
        query: EntryQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<EntryRecord>> {
        self.guard(self.inner.scan_entries(query, limit, cursor, ctx))
    }
    fn task(&self, id: TaskId, ctx: &Context) -> StorageFuture<'_, Option<TaskRecord>> {
        self.guard(self.inner.task(id, ctx))
    }
    fn scan_tasks(
        &self,
        query: TaskQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<TaskRecord>> {
        self.guard(self.inner.scan_tasks(query, limit, cursor, ctx))
    }
    fn submission(&self, id: SubmissionId, ctx: &Context) -> StorageFuture<'_, Option<SubmissionRecord>> {
        self.guard(self.inner.submission(id, ctx))
    }
    fn scan_submissions(
        &self,
        query: SubmissionQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<SubmissionRecord>> {
        self.guard(self.inner.scan_submissions(query, limit, cursor, ctx))
    }
    fn submission_by_request(
        &self,
        conversation: ConversationId,
        request_id: &str,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<SubmissionRecord>> {
        self.guard(self.inner.submission_by_request(conversation, request_id, ctx))
    }
    fn find_document(
        &self,
        address: &DocumentAddress,
        at: DocumentPoint,
        ctx: &Context,
    ) -> StorageFuture<'_, Option<DocumentRecord>> {
        self.guard(self.inner.find_document(address, at, ctx))
    }
    fn document(&self, id: DocumentId, at: DocumentPoint, ctx: &Context) -> StorageFuture<'_, Option<StoredDocument>> {
        self.guard(self.inner.document(id, at, ctx))
    }
    fn scan_documents(
        &self,
        query: DocumentQuery,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> StorageFuture<'_, Page<DocumentRecord>> {
        self.guard(self.inner.scan_documents(query, limit, cursor, ctx))
    }
    fn close(&self, ctx: &Context) -> StorageFuture<'_, ()> {
        self.closed.store(true, Ordering::SeqCst);
        if self.close_fails.load(Ordering::SeqCst) {
            return async { Err(StorageError::Corrupt("injected close failure".to_owned())) }.boxed();
        }
        self.inner.close(ctx)
    }
}

struct Failable {
    session: Session,
    storage: Arc<Failing>,
    reports: Arc<AtomicUsize>,
}

fn open_failing() -> Failable {
    let storage = Arc::new(Failing::default());
    let reports = Arc::new(AtomicUsize::new(0));
    let counted = reports.clone();
    let options = SessionOptions {
        reporter: Some(Arc::new(move |_, info| {
            assert_eq!(info, &ReportInfo::SessionFailed);
            counted.fetch_add(1, Ordering::SeqCst);
        })),
        ..SessionOptions::default()
    };
    let session = Session::open(storage.clone(), options);
    Failable {
        session,
        storage,
        reports,
    }
}

async fn create_notes(session: &Session) -> ConversationId {
    session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                let conversation = tx.create_conversation(ConversationOwnership::Ownerless).await?;
                tx.doc(DocAddress::<Notes>::new(conversation.id)).await?;
                Ok(conversation.id)
            })
        })
        .await
        .unwrap()
}

#[tokio::test]
async fn a_commit_error_fails_the_session() {
    let Failable {
        session,
        storage,
        reports,
    } = open_failing();
    let conversation = create_notes(&session).await;
    let state = session
        .document_state(DocAddress::<Notes>::new(conversation), &ctx())
        .await
        .unwrap()
        .unwrap();
    let mut publications = session.subscribe_commits();
    storage.commits.store(true, Ordering::SeqCst);
    let error = session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Notes>::new(conversation))
                    .await?
                    .set(json!({ "items": [1] }));
                tx.append_entry(conversation, EntryDraft::new("lost")).await
            })
        })
        .await
        .unwrap_err();
    assert!(matches!(error, SessionError::Failed(_)), "{error:?}");
    // Nothing was published and the state kept its committed value, then ended.
    assert_eq!(state.value(), json!({ "items": [] }));
    assert!(matches!(state.closed().await, WatchEnd::SessionFailed(_)));
    assert!(publications.next().await.is_none());
    assert!(matches!(session.closed().await, SessionEnd::Failed(_)));
    assert!(
        storage.closed.load(Ordering::SeqCst),
        "a failed Session closes its storage"
    );
    assert_eq!(reports.load(Ordering::SeqCst), 1);
    // Every later call gets the failure; retrying cannot succeed.
    storage.commits.store(false, Ordering::SeqCst);
    let later = session
        .commit(&ctx(), |tx| Box::pin(async move { Ok(tx.context().is_cancelled()) }))
        .await
        .unwrap_err();
    assert!(matches!(later, SessionError::Failed(_)));
    let snapshot = session.snapshot(DocAddress::<Notes>::new(conversation), &ctx()).await;
    assert!(matches!(snapshot, Err(SessionError::Failed(_))));
    assert!(session.failure().is_some());
    session.close(&ctx()).await.unwrap();
    assert_eq!(reports.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_read_error_fails_the_session_even_when_caught() {
    let Failable { session, storage, .. } = open_failing();
    let conversation = create_notes(&session).await;
    storage.reads.store(true, Ordering::SeqCst);
    let error = session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                // The closure swallows the error; the commit must still not succeed.
                let read = tx.conversation(conversation).await;
                assert!(matches!(read, Err(TxError::Failed(_))));
                Ok(())
            })
        })
        .await
        .unwrap_err();
    assert!(matches!(error, SessionError::Failed(_)));
    assert!(matches!(session.closed().await, SessionEnd::Failed(_)));
}

#[tokio::test]
async fn a_cold_load_error_fails_the_session() {
    let Failable { session, storage, .. } = open_failing();
    let conversation = create_notes(&session).await;
    let other = session
        .commit(&ctx(), |tx| {
            Box::pin(async move { Ok(tx.create_conversation(ConversationOwnership::Ownerless).await?.id) })
        })
        .await
        .unwrap();
    // A cached document is still served without storage; a cold one is not.
    storage.reads.store(true, Ordering::SeqCst);
    let cached = session.snapshot(DocAddress::<Notes>::new(conversation), &ctx()).await;
    assert!(cached.unwrap().is_some());
    let cold = session
        .snapshot(DocAddress::<Notes>::new(other), &ctx())
        .await
        .unwrap_err();
    assert!(matches!(cold, SessionError::Failed(_)));
}

#[tokio::test]
async fn cancellation_fails_only_its_call() {
    let Failable { session, .. } = open_failing();
    let conversation = create_notes(&session).await;
    let (cancelled, canceller) = ctx().with_cancel();
    canceller.cancel();
    let error = session
        .commit(&cancelled, |tx| {
            Box::pin(async move { Ok(tx.context().is_cancelled()) })
        })
        .await
        .unwrap_err();
    assert!(matches!(error, SessionError::Cancelled(_)));
    let error = session
        .snapshot_as_of(
            DocAddress::<Agent>::new(conversation),
            delta_harness::records::EntryId(1),
            &cancelled,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, SessionError::Cancelled(_)), "{error:?}");
    assert!(session.failure().is_none());
}

#[tokio::test]
async fn a_failing_storage_close_fails_close() {
    let Failable {
        session,
        storage,
        reports,
    } = open_failing();
    create_notes(&session).await;
    storage.close_fails.store(true, Ordering::SeqCst);
    let error = session.close(&ctx()).await.unwrap_err();
    assert!(matches!(error, SessionError::Failed(_)));
    assert!(matches!(session.closed().await, SessionEnd::Failed(_)));
    assert_eq!(reports.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_panicking_closure_rolls_back() {
    let Failable { session, .. } = open_failing();
    let conversation = create_notes(&session).await;
    let error = session
        .commit::<(), _>(&ctx(), move |tx| {
            Box::pin(async move {
                tx.append_entry(conversation, EntryDraft::new("lost")).await?;
                panic!("the closure panicked");
            })
        })
        .await
        .unwrap_err();
    assert!(matches!(tx_error(error), TxError::Panic(message) if message == "the closure panicked"));
    assert!(session.failure().is_none());
    create_notes(&session).await;
}

#[tokio::test]
async fn close_settles_admitted_commits_and_refuses_later_ones() {
    let Failable { session, storage, .. } = open_failing();
    let conversation = create_notes(&session).await;
    let state = session
        .document_state(DocAddress::<Notes>::new(conversation), &ctx())
        .await
        .unwrap()
        .unwrap();
    let mut subscription = state.subscribe();
    subscription.next().await.unwrap();
    let (started, started_rx) = tokio::sync::oneshot::channel();
    let (release, release_rx) = tokio::sync::oneshot::channel::<()>();
    let admitted = tokio::spawn({
        let session = session.clone();
        async move {
            session
                .commit(&ctx(), move |tx| {
                    Box::pin(async move {
                        started.send(()).unwrap();
                        release_rx.await.unwrap();
                        tx.doc(DocAddress::<Notes>::new(conversation))
                            .await?
                            .set(json!({ "items": [2] }));
                        tx.append_entry(conversation, EntryDraft::new("admitted")).await
                    })
                })
                .await
        }
    });
    started_rx.await.unwrap();
    let closing = tokio::spawn({
        let session = session.clone();
        async move { session.close(&ctx()).await }
    });
    let mut close = session.subscribe_close();
    close.wait().await;
    let refused = session
        .commit(&ctx(), |tx| Box::pin(async move { Ok(tx.context().is_cancelled()) }))
        .await
        .unwrap_err();
    assert!(matches!(refused, SessionError::Closed));
    assert!(matches!(
        session.snapshot(DocAddress::<Notes>::new(conversation), &ctx()).await,
        Err(SessionError::Closed)
    ));
    assert!(matches!(state.end(), Some(WatchEnd::SessionClosed)));
    release.send(()).unwrap();
    let entry = admitted.await.unwrap().unwrap();
    assert_eq!(entry.kind, "admitted");
    closing.await.unwrap().unwrap();
    assert!(storage.closed.load(Ordering::SeqCst));
    // The admitted commit settled after the seal: it delivered nothing.
    assert!(subscription.next().await.is_none());
    assert!(matches!(session.closed().await, SessionEnd::Closed));
    session.close(&ctx()).await.unwrap();
}

#[tokio::test]
async fn dropping_the_last_handle_closes_storage() {
    let Failable { session, storage, .. } = open_failing();
    create_notes(&session).await;
    let closed = session.closed();
    let services = Arc::new(7_u32);
    assert!(session.services().insert(services.clone()).is_none());
    assert_eq!(session.services().get::<u32>().as_deref(), Some(&7));
    drop(session);
    assert!(matches!(closed.await, SessionEnd::Closed));
    assert!(storage.closed.load(Ordering::SeqCst));
    assert_eq!(Arc::strong_count(&services), 1, "services are dropped at close");
}
