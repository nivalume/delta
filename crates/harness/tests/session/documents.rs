//! Document transactions, planning, checkpoints and migrations (upstream
//! `session-documents.test.ts`, `session-checkpoints-migrations.test.ts`).

use std::cell::RefCell;

use delta_chord::json;
use delta_harness::documents::{CheckpointInfo, DocAddress, DocDefinition, DocumentError, SessionScope};
use delta_harness::records::{DocumentAddress, DocumentPoint, DocumentScope, StoredDocument};
use delta_harness::session::TxError;

use super::*;

cases!(
    creates_an_initial_base_on_first_access,
    snapshots_never_create,
    empty_batches_write_nothing,
    plans_deltas_and_checkpoint_bases,
    passes_the_prepared_change_to_checkpoint_when,
    a_panicking_predicate_rolls_back,
    migrates_in_memory_then_writes_a_required_base,
    refuses_versions_it_cannot_read,
    refuses_non_object_values,
    refuses_other_semantics,
    reacquiring_with_another_version_is_refused,
    reloads_after_eviction_and_continues_counting,
);

/// The current incarnation at `address` as storage has it.
async fn stored(env: &Env, address: &DocumentAddress) -> Option<StoredDocument> {
    let record = env
        .storage
        .find_document(address, DocumentPoint::Current, &ctx())
        .await
        .unwrap()?;
    env.storage
        .document(record.id, DocumentPoint::Current, &ctx())
        .await
        .unwrap()
}

fn counter() -> DocumentAddress {
    DocAddress::<Counter>::new(()).to_address()
}

async fn bump<D: DocDefinition<Seed = (), Scope = SessionScope>>(env: &Env) {
    env.session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                let mut doc = tx.doc(DocAddress::<D>::new(())).await?;
                let count = doc.value().get("count").and_then(Json::as_i64).unwrap_or(0);
                doc.draft().field("count").unwrap().set(count + 1);
                Ok(())
            })
        })
        .await
        .unwrap();
}

async fn creates_an_initial_base_on_first_access(backend: Backend) {
    let env = Env::open(backend).await;
    // Created with an empty batch: still written.
    env.session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Counter>::new(())).await?;
                Ok(())
            })
        })
        .await
        .unwrap();
    let document = stored(&env, &counter()).await.unwrap();
    assert_eq!((document.version, document.deltas_since_base), (1, 0));
    assert_eq!(document.value, json!({ "count": 0 }));
    assert_eq!(document.record.scope, DocumentScope::Session);
    let snapshot = env
        .session
        .snapshot(DocAddress::<Counter>::new(()), &ctx())
        .await
        .unwrap();
    assert_eq!(snapshot, Some(json!({ "count": 0 })));
}

async fn snapshots_never_create(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    assert_eq!(
        env.session
            .snapshot(DocAddress::<Counter>::new(()), &ctx())
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        env.session
            .snapshot(DocAddress::<Notes>::new(conversation), &ctx())
            .await
            .unwrap(),
        None
    );
    assert!(
        env.session
            .document_state(DocAddress::<Notes>::new(conversation), &ctx())
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        stored(&env, &DocAddress::<Notes>::new(conversation).to_address())
            .await
            .is_none()
    );
}

async fn empty_batches_write_nothing(backend: Backend) {
    let env = Env::open(backend).await;
    bump::<Counter>(&env).await;
    let before = stored(&env, &counter()).await.unwrap();
    let mut publications = env.session.subscribe_commits();
    // An edit undone, and a structural rewrite of the same value: chord's tracker proves
    // equality, so nothing is written or published.
    env.session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                let mut doc = tx.doc(DocAddress::<Counter>::new(())).await?;
                doc.draft().field("count").unwrap().set(5);
                doc.draft().field("count").unwrap().set(1);
                let same = doc.value().clone();
                doc.set(json!({ "count": 1 }));
                doc.set(same);
                Ok(())
            })
        })
        .await
        .unwrap();
    assert_eq!(stored(&env, &counter()).await.unwrap(), before);
    bump::<Counter>(&env).await;
    let publication = publications.next().await.unwrap().unwrap();
    // The first publication received is the later, real change.
    assert!(publication.changes.iter().any(|change| matches!(
        change,
        delta_harness::session::CommitChange::Document { value: Some(value), .. } if *value == json!({ "count": 2 })
    )));
}

async fn plans_deltas_and_checkpoint_bases(backend: Backend) {
    let env = Env::open(backend).await;
    bump::<Counter>(&env).await; // creation base: count 1
    let mut tails = Vec::new();
    for _ in 0..5 {
        bump::<Counter>(&env).await;
        tails.push(stored(&env, &counter()).await.unwrap().deltas_since_base);
    }
    // Counter stores a base once two deltas follow the newest one.
    assert_eq!(tails, [1, 2, 0, 1, 2]);
    assert_eq!(stored(&env, &counter()).await.unwrap().value, json!({ "count": 6 }));
}

thread_local! {
    /// The line runs on the test's current-thread runtime, so the predicate records here.
    static RECORDED: RefCell<Vec<(Json, Vec<delta_chord::delta::Op>, u64)>> = const { RefCell::new(Vec::new()) };
}

/// Records every predicate call; checkpoints when the value says so.
struct Recorded;
impl DocDefinition for Recorded {
    const KIND: &'static str = "test.recorded";
    const VERSION: u32 = 1;
    type Scope = SessionScope;
    type Seed = ();
    fn initial((): &()) -> Json {
        json!({ "n": 0, "base": false })
    }
    fn checkpoint_when(value: &Json, ops: &[delta_chord::delta::Op], info: CheckpointInfo) -> bool {
        RECORDED.with_borrow_mut(|calls| calls.push((value.clone(), ops.to_vec(), info.deltas_since_base)));
        value.get("base") == Some(&Json::Bool(true))
    }
}

async fn passes_the_prepared_change_to_checkpoint_when(backend: Backend) {
    let env = Env::open(backend).await;
    let edit = |n: i64, base: bool| {
        let env = &env;
        async move {
            env.session
                .commit(&ctx(), move |tx| {
                    Box::pin(async move {
                        let mut doc = tx.doc(DocAddress::<Recorded>::new(())).await?;
                        doc.set(json!({ "n": n, "base": base }));
                        Ok(())
                    })
                })
                .await
                .unwrap();
        }
    };
    edit(0, false).await; // creation: no predicate call
    edit(0, false).await; // empty batch: no predicate call
    edit(1, false).await;
    edit(2, true).await;
    edit(3, false).await;
    let calls = RECORDED.take();
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[0].0, json!({ "n": 1, "base": false }));
    assert_eq!(
        calls[0].1,
        [delta_chord::delta::Op::Set(delta_chord::path!["n"], Json::from(1))]
    );
    assert_eq!(
        calls.iter().map(|(_, _, deltas)| *deltas).collect::<Vec<_>>(),
        [0, 1, 0],
        "the delta count excludes this change and resets after a base"
    );
}

async fn a_panicking_predicate_rolls_back(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    env.session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Panicky>::new(())).await?;
                Ok(())
            })
        })
        .await
        .unwrap();
    let error = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.append_entry(conversation, EntryDraft::new("lost")).await?;
                tx.doc(DocAddress::<Panicky>::new(()))
                    .await?
                    .set(json!({ "panic": true }));
                Ok(())
            })
        })
        .await
        .unwrap_err();
    assert!(matches!(tx_error(error), TxError::Panic(_)));
    assert!(env.session.failure().is_none());
    let value = env
        .session
        .snapshot(DocAddress::<Panicky>::new(()), &ctx())
        .await
        .unwrap();
    assert_eq!(value, Some(json!({ "panic": false })));
    // The rolled-back change left the cached state as it was: later edits apply.
    env.session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Panicky>::new(()))
                    .await?
                    .set(json!({ "panic": false, "ok": 1 }));
                Ok(())
            })
        })
        .await
        .unwrap();
}

async fn migrates_in_memory_then_writes_a_required_base(backend: Backend) {
    let env = Env::open(backend).await;
    bump::<Counter>(&env).await;
    bump::<Counter>(&env).await;
    // A read through version 2 migrates in memory only.
    let migrated = env
        .session
        .snapshot(DocAddress::<CounterV2>::new(()), &ctx())
        .await
        .unwrap();
    assert_eq!(migrated, Some(json!({ "count": 2, "migrated": true })));
    let document = stored(&env, &counter()).await.unwrap();
    assert_eq!((document.version, document.deltas_since_base), (1, 1));
    // The first transaction writes the version base, also without edits.
    env.session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<CounterV2>::new(())).await?;
                Ok(())
            })
        })
        .await
        .unwrap();
    let document = stored(&env, &counter()).await.unwrap();
    assert_eq!((document.version, document.deltas_since_base), (2, 0));
    assert_eq!(document.value, json!({ "count": 2, "migrated": true }));
    // Then deltas at the new version.
    bump::<CounterV2>(&env).await;
    let document = stored(&env, &counter()).await.unwrap();
    assert_eq!((document.version, document.deltas_since_base), (2, 1));
    assert_eq!(document.value, json!({ "count": 3, "migrated": true }));
    // Version 1 can no longer read it.
    let error = env
        .session
        .snapshot(DocAddress::<Counter>::new(()), &ctx())
        .await
        .unwrap_err();
    assert!(matches!(
        tx_error(error),
        TxError::Document(DocumentError::VersionTooNew {
            stored: 2,
            definition: 1,
            ..
        })
    ));
}

async fn refuses_versions_it_cannot_read(backend: Backend) {
    let env = Env::open(backend).await;
    bump::<Counter>(&env).await;
    let error = env
        .session
        .snapshot(DocAddress::<CounterV3>::new(()), &ctx())
        .await
        .unwrap_err();
    assert!(matches!(
        tx_error(error),
        TxError::Document(DocumentError::MigrationRequired { stored: 1, .. })
    ));
    let error = env
        .session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<CounterV3>::new(())).await?;
                Ok(())
            })
        })
        .await
        .unwrap_err();
    assert!(matches!(
        tx_error(error),
        TxError::Document(DocumentError::MigrationRequired { .. })
    ));
    // The stored bytes are untouched and version 1 still reads them.
    let value = env
        .session
        .snapshot(DocAddress::<Counter>::new(()), &ctx())
        .await
        .unwrap();
    assert_eq!(value, Some(json!({ "count": 1 })));
    assert_eq!(stored(&env, &counter()).await.unwrap().version, 1);
}

async fn refuses_non_object_values(backend: Backend) {
    let env = Env::open(backend).await;
    let error = env
        .session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<BadInitial>::new(())).await?;
                Ok(())
            })
        })
        .await
        .unwrap_err();
    assert!(matches!(
        tx_error(error),
        TxError::Document(DocumentError::NotAnObject { .. })
    ));
    bump::<Counter>(&env).await;
    let error = env
        .session
        .snapshot(DocAddress::<CounterBadMigration>::new(()), &ctx())
        .await
        .unwrap_err();
    assert!(matches!(
        tx_error(error),
        TxError::Document(DocumentError::NotAnObject { .. })
    ));
    // A refused migration caches nothing: version 1 is served as before.
    let value = env
        .session
        .snapshot(DocAddress::<Counter>::new(()), &ctx())
        .await
        .unwrap();
    assert_eq!(value, Some(json!({ "count": 1 })));
}

async fn refuses_other_semantics(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    env.session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Notes>::new(conversation)).await?;
                Ok(())
            })
        })
        .await
        .unwrap();
    let error = env
        .session
        .snapshot(DocAddress::<NotesRewindable>::new(conversation), &ctx())
        .await
        .unwrap_err();
    assert!(matches!(
        tx_error(error),
        TxError::Document(DocumentError::DefinitionMismatch { .. })
    ));
    let error = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<NotesRewindable>::new(conversation)).await?;
                Ok(())
            })
        })
        .await
        .unwrap_err();
    assert!(matches!(
        tx_error(error),
        TxError::Document(DocumentError::DefinitionMismatch { .. })
    ));
}

async fn reacquiring_with_another_version_is_refused(backend: Backend) {
    let env = Env::open(backend).await;
    bump::<Counter>(&env).await;
    let error = env
        .session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Counter>::new(())).await?;
                tx.doc(DocAddress::<CounterV2>::new(())).await?;
                Ok(())
            })
        })
        .await
        .unwrap_err();
    assert!(matches!(
        tx_error(error),
        TxError::Document(DocumentError::DefinitionMismatch { .. })
    ));
}

async fn reloads_after_eviction_and_continues_counting(backend: Backend) {
    let options = SessionOptions {
        document_retention: Some(std::time::Duration::ZERO),
        ..SessionOptions::default()
    };
    let env = Env::open_with(backend, options).await;
    bump::<Counter>(&env).await;
    let mut tails = Vec::new();
    for _ in 0..4 {
        bump::<Counter>(&env).await;
        assert_eq!(env.session.cached_documents().await.unwrap(), 0);
        tails.push(stored(&env, &counter()).await.unwrap().deltas_since_base);
    }
    // The count comes from storage at each reload.
    assert_eq!(tails, [1, 2, 0, 1]);
    assert_eq!(
        env.session
            .snapshot(DocAddress::<Counter>::new(()), &ctx())
            .await
            .unwrap(),
        Some(json!({ "count": 5 }))
    );
}
