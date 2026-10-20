//! Incarnations: retirement, recreation and family members (upstream
//! `session-documents.test.ts`).

use delta_chord::json;
use delta_harness::documents::DocAddress;
use delta_harness::records::{DocumentAddress, DocumentPoint, StoredDocument};

use super::*;

cases!(
    retires_and_recreates_in_one_commit,
    retires_without_acquisition,
    retires_after_writing_the_final_content,
    family_seeds_apply_once,
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

async fn bump(env: &Env) {
    env.session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                let mut doc = tx.doc(DocAddress::<Counter>::new(())).await?;
                let count = doc.value().get("count").and_then(Json::as_i64).unwrap_or(0);
                doc.draft().field("count").unwrap().set(count + 1);
                Ok(())
            })
        })
        .await
        .unwrap();
}

async fn retires_and_recreates_in_one_commit(backend: Backend) {
    let env = Env::open(backend).await;
    bump(&env).await;
    let old = stored(&env, &counter()).await.unwrap().record;
    env.session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                tx.retire_doc(DocAddress::<Counter>::new(())).await?;
                let doc = tx.doc(DocAddress::<Counter>::new(())).await?;
                assert_eq!(
                    doc.value(),
                    &json!({ "count": 0 }),
                    "a new incarnation starts from initial"
                );
                Ok(())
            })
        })
        .await
        .unwrap();
    let new = stored(&env, &counter()).await.unwrap();
    assert_ne!(new.record.id, old.id);
    assert_eq!(new.value, json!({ "count": 0 }));
    let old = env
        .storage
        .document(old.id, DocumentPoint::Current, &ctx())
        .await
        .unwrap();
    assert!(old.is_none(), "the old incarnation is retired");
    assert_eq!(
        env.session
            .snapshot(DocAddress::<Counter>::new(()), &ctx())
            .await
            .unwrap(),
        Some(json!({ "count": 0 }))
    );
}

async fn retires_without_acquisition(backend: Backend) {
    let env = Env::open(backend).await;
    // Retiring an absent address does nothing.
    env.session
        .commit(&ctx(), |tx| {
            Box::pin(async move { tx.retire_doc(DocAddress::<Counter>::new(())).await })
        })
        .await
        .unwrap();
    bump(&env).await;
    env.session
        .commit(&ctx(), |tx| {
            Box::pin(async move { tx.retire_doc(DocAddress::<Counter>::new(())).await })
        })
        .await
        .unwrap();
    assert!(stored(&env, &counter()).await.is_none());
    assert_eq!(
        env.session
            .snapshot(DocAddress::<Counter>::new(()), &ctx())
            .await
            .unwrap(),
        None
    );
    // Created and retired in one commit: written, never cached.
    env.session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Counter>::new(())).await?.set(json!({ "count": 5 }));
                tx.retire_doc(DocAddress::<Counter>::new(())).await
            })
        })
        .await
        .unwrap();
    assert!(stored(&env, &counter()).await.is_none());
}

async fn retires_after_writing_the_final_content(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    let address = DocAddress::<Agent>::new(conversation);
    env.session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Agent>::new(conversation)).await?;
                Ok(())
            })
        })
        .await
        .unwrap();
    let record = stored(&env, &address.to_address()).await.unwrap().record;
    let marker = env.append(conversation, "before").await;
    env.session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Agent>::new(conversation))
                    .await?
                    .set(json!({ "turn": 9 }));
                let entry = tx.append_entry(conversation, EntryDraft::new("final")).await?;
                tx.retire_doc(DocAddress::<Agent>::new(conversation)).await?;
                Ok(entry)
            })
        })
        .await
        .unwrap();
    let stored_entry = env.storage.entry(marker.id, &ctx()).await.unwrap().unwrap();
    let retired = env
        .storage
        .find_document(&address.to_address(), DocumentPoint::Current, &ctx())
        .await
        .unwrap();
    assert!(retired.is_none());
    // The rewindable history keeps the value before the last commit.
    let before = env
        .storage
        .document(record.id, DocumentPoint::At(stored_entry.commit_seq), &ctx())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(before.value, json!({ "turn": 0 }));
}

async fn family_seeds_apply_once(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    for seed in ["first", "second"] {
        env.session
            .commit(&ctx(), move |tx| {
                Box::pin(async move {
                    tx.doc_member(DocAddress::<Member>::member(conversation, "a"), seed.to_owned())
                        .await?;
                    tx.doc_member(DocAddress::<Member>::member(conversation, ""), seed.to_owned())
                        .await?;
                    Ok(())
                })
            })
            .await
            .unwrap();
    }
    let a = env
        .session
        .snapshot(DocAddress::<Member>::member(conversation, "a"), &ctx())
        .await
        .unwrap();
    assert_eq!(a, Some(json!({ "name": "first" })));
    let empty = env
        .session
        .snapshot(DocAddress::<Member>::member(conversation, ""), &ctx())
        .await
        .unwrap();
    assert_eq!(empty, Some(json!({ "name": "first" })));
    let b = env
        .session
        .snapshot(DocAddress::<Member>::member(conversation, "b"), &ctx())
        .await
        .unwrap();
    assert_eq!(b, None);
}
