//! Document states, subscriptions and commit publications (upstream
//! `session-states.test.ts`, `session-watches.test.ts`, publication cases of
//! `session-documents.test.ts`).

use std::sync::atomic::{AtomicUsize, Ordering};

use delta_chord::context::ContextKey;
use delta_chord::delta::{Op, applied};
use delta_chord::json;
use delta_chord::state::DeliveryKind;
use delta_harness::documents::DocAddress;
use delta_harness::records::DocumentAddress;
use delta_harness::session::{CommitChange, DocumentState, SessionEnd, WatchEnd};

use super::*;

cases!(
    hydrates_then_delivers_exact_ops,
    states_are_independent_and_read_only,
    retirement_ends_the_state,
    falling_behind_merges_batches,
    migration_bases_deliver_nothing_to_the_new_version,
    older_shapes_receive_the_new_value,
    close_ends_states_and_subscriptions,
    deliveries_carry_the_commit_context,
    publications_follow_adoption,
    observers_run_on_the_line_before_subscribers,
    subscribers_may_call_the_session,
);

async fn set_notes(env: &Env, conversation: ConversationId, items: Json) {
    env.session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Notes>::new(conversation))
                    .await?
                    .draft()
                    .field("items")
                    .unwrap()
                    .set(items);
                Ok(())
            })
        })
        .await
        .unwrap();
}

async fn push_note(env: &Env, conversation: ConversationId, note: &str) {
    let note = note.to_owned();
    env.session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let mut notes = tx.doc(DocAddress::<Notes>::new(conversation)).await?;
                notes.draft().field("items").unwrap().into_array().unwrap().push(note);
                Ok(())
            })
        })
        .await
        .unwrap();
}

async fn notes_state(env: &Env, conversation: ConversationId) -> DocumentState {
    env.session
        .document_state(DocAddress::<Notes>::new(conversation), &ctx())
        .await
        .unwrap()
        .unwrap()
}

async fn hydrates_then_delivers_exact_ops(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    set_notes(&env, conversation, json!([])).await;
    let state = notes_state(&env, conversation).await;
    let mut subscription = state.subscribe();
    let hydrate = subscription.next().await.unwrap();
    assert_eq!(hydrate.kind, DeliveryKind::Hydrate);
    assert_eq!(hydrate.value, json!({ "items": [] }));
    push_note(&env, conversation, "a").await;
    push_note(&env, conversation, "b").await;
    let mut replica = hydrate.value;
    for expected in [json!({ "items": ["a"] }), json!({ "items": ["a", "b"] })] {
        let update = subscription.next().await.unwrap();
        assert_eq!(update.kind, DeliveryKind::Update);
        replica = applied(&replica, &update.ops).unwrap();
        assert_eq!(update.value, expected);
        assert_eq!(replica, expected);
    }
    assert_eq!(state.value(), json!({ "items": ["a", "b"] }));
    assert!(subscription.try_next().is_none());
    assert!(state.end().is_none());
}

async fn states_are_independent_and_read_only(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    set_notes(&env, conversation, json!(["x"])).await;
    let first = notes_state(&env, conversation).await;
    let second = notes_state(&env, conversation).await;
    assert_eq!(first.id(), second.id());
    let mut early = first.subscribe();
    push_note(&env, conversation, "y").await;
    let mut late = second.subscribe();
    drop(first);
    assert_eq!(early.next().await.unwrap().value, json!({ "items": ["x"] }));
    assert_eq!(early.next().await.unwrap().value, json!({ "items": ["x", "y"] }));
    // A late subscription hydrates at the current revision without redelivering it.
    let hydrate = late.next().await.unwrap();
    assert_eq!(
        (hydrate.kind, hydrate.value),
        (DeliveryKind::Hydrate, json!({ "items": ["x", "y"] }))
    );
    assert!(late.try_next().is_none());
    // A delivered value is the holder's own copy.
    let mut copy = second.value();
    copy = applied(&copy, &[Op::Set(delta_chord::path!["items"], json!([]))]).unwrap();
    assert_ne!(copy, second.value());
}

async fn retirement_ends_the_state(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    set_notes(&env, conversation, json!(["old"])).await;
    let state = notes_state(&env, conversation).await;
    let mut subscription = state.subscribe();
    subscription.next().await.unwrap();
    env.session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.retire_doc(DocAddress::<Notes>::new(conversation)).await?;
                tx.doc(DocAddress::<Notes>::new(conversation)).await?;
                Ok(())
            })
        })
        .await
        .unwrap();
    let retired = subscription.next().await.unwrap();
    assert_eq!(retired.value, Json::Null);
    assert_eq!(&*retired.ops, &[Op::Replace(Json::Null)]);
    assert!(
        subscription.next().await.is_none(),
        "a retired state never follows a replacement"
    );
    assert!(matches!(state.end(), Some(WatchEnd::Retired)));
    assert!(matches!(state.closed().await, WatchEnd::Retired));
    assert_eq!(state.value(), Json::Null);
    // A subscription after the end hydrates with the final value and ends.
    let mut after = state.subscribe();
    assert_eq!(after.next().await.unwrap().value, Json::Null);
    assert!(after.next().await.is_none());
    // The replacement incarnation is a new acquisition.
    let replacement = notes_state(&env, conversation).await;
    assert_ne!(replacement.id(), state.id());
    assert_eq!(replacement.value(), json!({ "items": [] }));
}

async fn falling_behind_merges_batches(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    set_notes(&env, conversation, json!([])).await;
    let state = notes_state(&env, conversation).await;
    let mut subscription = state.subscribe();
    let mut replica = subscription.next().await.unwrap().value;
    for n in 0..150 {
        push_note(&env, conversation, &n.to_string()).await;
    }
    let mut deliveries = 0;
    let mut skipped = 0;
    while let Some(delivery) = subscription.try_next() {
        deliveries += 1;
        skipped += delivery.skipped;
        replica = applied(&replica, &delivery.ops).unwrap();
    }
    assert!(deliveries < 150, "{deliveries} deliveries");
    assert_eq!(deliveries + skipped, 150);
    assert_eq!(replica, state.value());
    assert_eq!(
        replica.get("items").and_then(Json::as_array).map(|items| items.len()),
        Some(150)
    );
}

async fn migration_bases_deliver_nothing_to_the_new_version(backend: Backend) {
    let env = Env::open(backend).await;
    env.session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Counter>::new(())).await?.set(json!({ "count": 4 }));
                Ok(())
            })
        })
        .await
        .unwrap();
    let state = env
        .session
        .document_state(DocAddress::<CounterV2>::new(()), &ctx())
        .await
        .unwrap()
        .unwrap();
    let mut subscription = state.subscribe();
    assert_eq!(
        subscription.next().await.unwrap().value,
        json!({ "count": 4, "migrated": true })
    );
    env.session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<CounterV2>::new(())).await?;
                Ok(())
            })
        })
        .await
        .unwrap();
    assert!(
        subscription.try_next().is_none(),
        "an equal-value version base publishes nothing"
    );
    let stored = env
        .storage
        .find_document(
            &DocAddress::<Counter>::new(()).to_address(),
            delta_harness::records::DocumentPoint::Current,
            &ctx(),
        )
        .await
        .unwrap()
        .unwrap();
    let stored = env
        .storage
        .document(stored.id, delta_harness::records::DocumentPoint::Current, &ctx())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored.version, 2);
}

async fn older_shapes_receive_the_new_value(backend: Backend) {
    let env = Env::open(backend).await;
    env.session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Counter>::new(())).await?;
                Ok(())
            })
        })
        .await
        .unwrap();
    let old = env
        .session
        .document_state(DocAddress::<Counter>::new(()), &ctx())
        .await
        .unwrap()
        .unwrap();
    let mut subscription = old.subscribe();
    let mut replica = subscription.next().await.unwrap().value;
    // A newer definition reloads from storage and writes; the older observer follows.
    env.session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<CounterV2>::new(()))
                    .await?
                    .draft()
                    .field("count")
                    .unwrap()
                    .set(8);
                Ok(())
            })
        })
        .await
        .unwrap();
    let update = subscription.next().await.unwrap();
    replica = applied(&replica, &update.ops).unwrap();
    assert_eq!(replica, json!({ "count": 8, "migrated": true }));
    assert_eq!(old.value(), json!({ "count": 8, "migrated": true }));
}

async fn close_ends_states_and_subscriptions(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    set_notes(&env, conversation, json!(["kept"])).await;
    let state = notes_state(&env, conversation).await;
    let mut subscription = state.subscribe();
    subscription.next().await.unwrap();
    let mut publications = env.session.subscribe_commits();
    let mut closing = env.session.subscribe_close();
    assert!(!closing.is_closing());
    env.session.close(&ctx()).await.unwrap();
    assert!(matches!(closing.wait().await, SessionEnd::Closed));
    assert!(matches!(state.end(), Some(WatchEnd::SessionClosed)));
    assert!(subscription.next().await.is_none());
    assert!(publications.next().await.is_none());
    assert_eq!(
        state.value(),
        json!({ "items": ["kept"] }),
        "the value stays the last revision"
    );
}

static TRACE: ContextKey<&'static str> = ContextKey::new("test.trace");

async fn deliveries_carry_the_commit_context(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    set_notes(&env, conversation, json!([])).await;
    let state = notes_state(&env, conversation).await;
    let mut subscription = state.subscribe();
    subscription.next().await.unwrap();
    let mut publications = env.session.subscribe_commits();
    let (traced, canceller) = ctx().with_value(&TRACE, "commit-7").with_cancel();
    env.session
        .commit(&traced, move |tx| {
            Box::pin(async move {
                let mut notes = tx.doc(DocAddress::<Notes>::new(conversation)).await?;
                notes.draft().field("items").unwrap().into_array().unwrap().push(1);
                Ok(())
            })
        })
        .await
        .unwrap();
    canceller.cancel();
    let delivery = subscription.next().await.unwrap();
    assert_eq!(delivery.context.value(&TRACE), Some(&"commit-7"));
    assert!(
        !delivery.context.is_cancelled(),
        "deliveries carry no producer cancellation"
    );
    let publication = publications.next().await.unwrap().unwrap();
    assert_eq!(publication.context.value(&TRACE), Some(&"commit-7"));
}

async fn publications_follow_adoption(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    let mut publications = env.session.subscribe_commits();
    let entry = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let entry = tx.append_entry(conversation, EntryDraft::new("e")).await?;
                tx.doc(DocAddress::<Notes>::new(conversation)).await?;
                Ok(entry)
            })
        })
        .await
        .unwrap();
    push_note(&env, conversation, "n").await;
    env.session
        .commit(&ctx(), move |tx| {
            Box::pin(async move { tx.retire_doc(DocAddress::<Notes>::new(conversation)).await })
        })
        .await
        .unwrap();
    let created = publications.next().await.unwrap().unwrap();
    assert!(matches!(&created.changes[0], CommitChange::Entry(record) if *record == entry));
    let CommitChange::Document {
        record,
        conversation_id,
        version,
        value,
        ops,
    } = &created.changes[1]
    else {
        panic!("expected a document change: {:?}", created.changes);
    };
    assert_eq!(record.created_at, created.seq);
    assert_eq!((*conversation_id, *version), (Some(conversation), Some(1)));
    assert_eq!(value.as_ref(), Some(&json!({ "items": [] })));
    assert!(ops.is_empty(), "a creation publishes no ops");
    let changed = publications.next().await.unwrap().unwrap();
    let CommitChange::Document { ops, .. } = &changed.changes[0] else {
        panic!("expected a document change");
    };
    assert_eq!(ops.len(), 1);
    let retired = publications.next().await.unwrap().unwrap();
    let CommitChange::Document {
        record, value, version, ..
    } = &retired.changes[0]
    else {
        panic!("expected a document change");
    };
    assert_eq!((record.retired_at, value, version), (Some(retired.seq), &None, &None));
    assert!(created.seq < changed.seq && changed.seq < retired.seq);
    let address: DocumentAddress = DocAddress::<Notes>::new(conversation).to_address();
    assert_eq!(record.address(), address);
}

async fn observers_run_on_the_line_before_subscribers(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    let seen = Arc::new(AtomicUsize::new(0));
    let observed = seen.clone();
    let guard = env.session.observe_commits(Box::new(
        move |publication: &delta_harness::session::CommitPublication| {
            observed.fetch_add(publication.changes.len(), Ordering::SeqCst);
            Ok(())
        },
    ));
    let mut publications = env.session.subscribe_commits();
    env.append(conversation, "one").await;
    // The observer ran before the commit returned.
    assert_eq!(seen.load(Ordering::SeqCst), 1);
    assert!(publications.next().await.unwrap().is_ok());
    drop(guard);
    env.append(conversation, "two").await;
    assert_eq!(seen.load(Ordering::SeqCst), 1);
    // A failing observer fails the Session: its memory would be behind storage.
    let _failing = env
        .session
        .observe_commits(Box::new(|_: &delta_harness::session::CommitPublication| {
            Err("index out of step".into())
        }));
    env.append(conversation, "three").await;
    assert!(env.session.failure().is_some());
    assert!(matches!(env.session.closed().await, SessionEnd::Failed(_)));
}

async fn subscribers_may_call_the_session(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    let mut publications = env.session.subscribe_commits();
    let session = env.session.clone();
    let listener = tokio::spawn(async move {
        let publication = publications.next().await.unwrap().unwrap();
        let CommitChange::Entry(entry) = &publication.changes[0] else {
            panic!("expected an entry");
        };
        let reply = EntryDraft::new("reply");
        let conversation = entry.conversation_id;
        session
            .commit(&ctx(), move |tx| {
                Box::pin(async move { tx.append_entry(conversation, reply).await })
            })
            .await
            .unwrap()
    });
    env.append(conversation, "question").await;
    let reply = listener.await.unwrap();
    assert_eq!(reply.kind, "reply");
}
