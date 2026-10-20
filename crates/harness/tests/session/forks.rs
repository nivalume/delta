//! Forks and historical snapshots (upstream `session-forks.test.ts` and the historical
//! cases of `session-checkpoints-migrations.test.ts`).

use delta_chord::json;
use delta_harness::documents::{CheckpointInfo, DocAddress, DocDefinition};
use delta_harness::records::{DocumentPoint, DocumentScope, EntryId};
use delta_harness::session::{CommitChange, TxError};

use super::*;

cases!(
    copies_as_of_and_current_and_leaves_initial_absent,
    as_of_copies_the_final_state_of_the_entry_commit,
    as_of_copies_come_from_the_entry_owning_ancestor,
    invisible_fork_points_reject,
    duplicate_sources_reject,
    fork_transactions_cannot_write_sources,
    copies_read_through_a_definition_become_creations,
    copies_can_be_retired_and_recreated,
    family_members_are_copied,
    snapshots_as_of_entries,
    snapshots_as_of_select_the_live_incarnation,
);

async fn set_turn(env: &Env, conversation: ConversationId, turn: i64) {
    env.session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Agent>::new(conversation))
                    .await?
                    .set(json!({ "turn": turn }));
                Ok(())
            })
        })
        .await
        .unwrap();
}

async fn set_current(env: &Env, conversation: ConversationId, value: i64) {
    env.session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Current>::new(conversation))
                    .await?
                    .set(json!({ "value": value }));
                Ok(())
            })
        })
        .await
        .unwrap();
}

async fn fork(env: &Env, parent: ConversationId, at: EntryId) -> Result<ConversationId, SessionError> {
    env.session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let child = tx
                    .fork_conversation(parent, at, ConversationOwnership::Ownerless)
                    .await?;
                Ok(child.id)
            })
        })
        .await
}

async fn snapshot<D: DocDefinition<Seed = ()>>(env: &Env, owner: delta_harness::documents::OwnerOf<D>) -> Option<Json> {
    env.session.snapshot(DocAddress::<D>::new(owner), &ctx()).await.unwrap()
}

async fn copies_as_of_and_current_and_leaves_initial_absent(backend: Backend) {
    let env = Env::open(backend).await;
    let parent = env.conversation().await;
    set_turn(&env, parent, 1).await;
    let at = env.append(parent, "fork here").await;
    set_turn(&env, parent, 2).await;
    set_current(&env, parent, 5).await;
    env.session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Notes>::new(parent))
                    .await?
                    .set(json!({ "items": ["parent"] }));
                Ok(())
            })
        })
        .await
        .unwrap();
    let mut publications = env.session.subscribe_commits();
    let child = fork(&env, parent, at.id).await.unwrap();
    assert_eq!(snapshot::<Agent>(&env, child).await, Some(json!({ "turn": 1 })));
    assert_eq!(snapshot::<Current>(&env, child).await, Some(json!({ "value": 5 })));
    assert_eq!(
        snapshot::<Notes>(&env, child).await,
        None,
        "initial documents are not copied"
    );
    let record = env.storage.conversation(child, &ctx()).await.unwrap().unwrap();
    let parent_edge = record.parent.unwrap();
    assert_eq!((parent_edge.conversation_id, parent_edge.at), (parent, at.id));
    // Unaccessed copies publish copy descriptors.
    let publication = publications.next().await.unwrap().unwrap();
    let copies = publication
        .changes
        .iter()
        .filter(
            |change| matches!(change, CommitChange::DocumentCopy { conversation_id, .. } if *conversation_id == child),
        )
        .count();
    assert_eq!(copies, 2);
    // Copies are independent of later source changes.
    set_current(&env, parent, 6).await;
    assert_eq!(snapshot::<Current>(&env, child).await, Some(json!({ "value": 5 })));
}

async fn as_of_copies_the_final_state_of_the_entry_commit(backend: Backend) {
    let env = Env::open(backend).await;
    let parent = env.conversation().await;
    let (first, _) = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let first = tx.append_entry(parent, EntryDraft::new("first")).await?;
                tx.doc(DocAddress::<Agent>::new(parent))
                    .await?
                    .set(json!({ "turn": 3 }));
                let second = tx.append_entry(parent, EntryDraft::new("second")).await?;
                Ok((first, second))
            })
        })
        .await
        .unwrap();
    let child = fork(&env, parent, first.id).await.unwrap();
    assert_eq!(snapshot::<Agent>(&env, child).await, Some(json!({ "turn": 3 })));
    // The child sees entries up to the fork point only.
    let page = env
        .storage
        .scan_entries(delta_harness::records::EntryQuery::new(child), 10, None, &ctx())
        .await
        .unwrap();
    assert_eq!(
        page.items.iter().map(|entry| entry.kind.as_str()).collect::<Vec<_>>(),
        ["first"]
    );
}

async fn as_of_copies_come_from_the_entry_owning_ancestor(backend: Backend) {
    let env = Env::open(backend).await;
    let root = env.conversation().await;
    set_turn(&env, root, 1).await;
    let early = env.append(root, "early").await;
    set_turn(&env, root, 2).await;
    set_current(&env, root, 1).await;
    let middle = fork(&env, root, early.id).await.unwrap();
    set_turn(&env, middle, 7).await;
    set_current(&env, middle, 70).await;
    // Forking the middle at the root's entry: as_of from the root at that entry, current
    // from the middle.
    let child = fork(&env, middle, early.id).await.unwrap();
    assert_eq!(snapshot::<Agent>(&env, child).await, Some(json!({ "turn": 1 })));
    assert_eq!(snapshot::<Current>(&env, child).await, Some(json!({ "value": 70 })));
}

async fn invisible_fork_points_reject(backend: Backend) {
    let env = Env::open(backend).await;
    let a = env.conversation().await;
    let b = env.conversation().await;
    let entry = env.append(a, "a only").await;
    let error = fork(&env, b, entry.id).await.unwrap_err();
    match tx_error(error) {
        TxError::Fork(message) => assert!(message.contains("is not visible from conversation"), "{message}"),
        other => panic!("unexpected {other:?}"),
    }
    assert!(env.session.failure().is_none());
    fork(&env, a, entry.id).await.unwrap();
}

/// [`Agent`]'s kind as a latest document copied as current.
struct AgentCurrent;
impl DocDefinition for AgentCurrent {
    const KIND: &'static str = "test.agent";
    const VERSION: u32 = 1;
    type Scope = delta_harness::documents::ConversationScope<
        delta_harness::documents::Latest,
        delta_harness::documents::ForkCurrent,
    >;
    type Seed = ();
    fn initial((): &()) -> Json {
        json!({ "turn": 0 })
    }
    fn checkpoint_when(_: &Json, _: &[delta_chord::delta::Op], _: CheckpointInfo) -> bool {
        false
    }
}

async fn duplicate_sources_reject(backend: Backend) {
    let env = Env::open(backend).await;
    let root = env.conversation().await;
    set_turn(&env, root, 1).await;
    let at = env.append(root, "at").await;
    let middle = fork(&env, root, at.id).await.unwrap();
    // The middle's copy of the as_of document is replaced by a current one of the same kind.
    env.session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.retire_doc(DocAddress::<Agent>::new(middle)).await?;
                tx.doc(DocAddress::<AgentCurrent>::new(middle)).await?;
                Ok(())
            })
        })
        .await
        .unwrap();
    let error = fork(&env, middle, at.id).await.unwrap_err();
    match tx_error(error) {
        TxError::Fork(message) => assert_eq!(message, "Fork selects multiple source documents for test.agent"),
        other => panic!("unexpected {other:?}"),
    }
}

async fn fork_transactions_cannot_write_sources(backend: Backend) {
    let env = Env::open(backend).await;
    let parent = env.conversation().await;
    set_turn(&env, parent, 1).await;
    set_current(&env, parent, 1).await;
    let at = env.append(parent, "at").await;
    let error = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.fork_conversation(parent, at.id, ConversationOwnership::Ownerless)
                    .await?;
                tx.doc(DocAddress::<Current>::new(parent))
                    .await?
                    .set(json!({ "value": 2 }));
                Ok(())
            })
        })
        .await
        .unwrap_err();
    match tx_error(error) {
        TxError::Fork(message) => assert!(message.contains("fork source document"), "{message}"),
        other => panic!("unexpected {other:?}"),
    }
    // Not even a current-policy document that is not a source yet.
    let error = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.fork_conversation(parent, at.id, ConversationOwnership::Ownerless)
                    .await?;
                tx.doc_member(DocAddress::<Member>::member(parent, "new"), "new".to_owned())
                    .await?;
                Ok(())
            })
        })
        .await
        .unwrap_err();
    match tx_error(error) {
        TxError::Fork(message) => assert!(message.contains("current-policy documents"), "{message}"),
        other => panic!("unexpected {other:?}"),
    }
    let error = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.fork_conversation(parent, at.id, ConversationOwnership::Ownerless)
                    .await?;
                tx.doc(DocAddress::<Agent>::new(parent))
                    .await?
                    .set(json!({ "turn": 2 }));
                Ok(())
            })
        })
        .await
        .unwrap_err();
    match tx_error(error) {
        TxError::Fork(message) => assert!(message.contains("fork source document"), "{message}"),
        other => panic!("unexpected {other:?}"),
    }
    // Reading a source in the fork transaction is fine.
    env.session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.fork_conversation(parent, at.id, ConversationOwnership::Ownerless)
                    .await?;
                tx.doc(DocAddress::<Current>::new(parent)).await?;
                Ok(())
            })
        })
        .await
        .unwrap();
}

async fn copies_read_through_a_definition_become_creations(backend: Backend) {
    let env = Env::open(backend).await;
    let parent = env.conversation().await;
    set_current(&env, parent, 3).await;
    let at = env.append(parent, "at").await;
    let mut publications = env.session.subscribe_commits();
    let child = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let child = tx
                    .fork_conversation(parent, at.id, ConversationOwnership::Ownerless)
                    .await?;
                let mut copy = tx.doc(DocAddress::<Current>::new(child.id)).await?;
                assert_eq!(copy.value(), &json!({ "value": 3 }));
                copy.draft().field("value").unwrap().set(4);
                Ok(child.id)
            })
        })
        .await
        .unwrap();
    assert_eq!(snapshot::<Current>(&env, child).await, Some(json!({ "value": 4 })));
    assert_eq!(snapshot::<Current>(&env, parent).await, Some(json!({ "value": 3 })));
    let publication = publications.next().await.unwrap().unwrap();
    assert!(publication.changes.iter().any(|change| matches!(
        change,
        CommitChange::Document { record, value: Some(value), .. }
            if record.scope == DocumentScope::Conversation { conversation_id: child } && *value == json!({ "value": 4 })
    )));
}

async fn copies_can_be_retired_and_recreated(backend: Backend) {
    let env = Env::open(backend).await;
    let parent = env.conversation().await;
    set_current(&env, parent, 3).await;
    let at = env.append(parent, "at").await;
    let child = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let child = tx
                    .fork_conversation(parent, at.id, ConversationOwnership::Ownerless)
                    .await?;
                tx.retire_doc(DocAddress::<Current>::new(child.id)).await?;
                let fresh = tx.doc(DocAddress::<Current>::new(child.id)).await?;
                assert_eq!(fresh.value(), &json!({ "value": 0 }));
                Ok(child.id)
            })
        })
        .await
        .unwrap();
    assert_eq!(snapshot::<Current>(&env, child).await, Some(json!({ "value": 0 })));
}

async fn family_members_are_copied(backend: Backend) {
    let env = Env::open(backend).await;
    let parent = env.conversation().await;
    let names: Vec<String> = (0..300).map(|n| format!("m{n}")).collect();
    let keys = names.clone();
    env.session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                for key in keys {
                    tx.doc_member(DocAddress::<Member>::member(parent, key.clone()), key)
                        .await?;
                }
                Ok(())
            })
        })
        .await
        .unwrap();
    let at = env.append(parent, "at").await;
    let child = fork(&env, parent, at.id).await.unwrap();
    for name in [&names[0], &names[255], &names[299]] {
        let copy = env
            .session
            .snapshot(DocAddress::<Member>::member(child, name.clone()), &ctx())
            .await
            .unwrap();
        assert_eq!(copy, Some(json!({ "name": name.as_str() })));
    }
}

async fn snapshots_as_of_entries(backend: Backend) {
    let env = Env::open(backend).await;
    let parent = env.conversation().await;
    let before = env.append(parent, "before").await;
    set_turn(&env, parent, 1).await;
    let one = env.append(parent, "one").await;
    set_turn(&env, parent, 2).await;
    let two = env.append(parent, "two").await;
    let address = DocAddress::<Agent>::new(parent);
    let as_of = |entry: EntryId| {
        let env = &env;
        let address = address.clone();
        async move { env.session.snapshot_as_of(address, entry, &ctx()).await.unwrap() }
    };
    assert_eq!(as_of(before.id).await, None, "no instance was alive then");
    assert_eq!(as_of(one.id).await, Some(json!({ "turn": 1 })));
    assert_eq!(as_of(two.id).await, Some(json!({ "turn": 2 })));
    // Through fork ancestry: the child's view of a parent entry reads the parent's history.
    let child = fork(&env, parent, two.id).await.unwrap();
    set_turn(&env, child, 9).await;
    let historical = env
        .session
        .snapshot_as_of(DocAddress::<Agent>::new(child), one.id, &ctx())
        .await
        .unwrap();
    assert_eq!(historical, Some(json!({ "turn": 1 })));
    let other = env.conversation().await;
    let error = env
        .session
        .snapshot_as_of(DocAddress::<Agent>::new(other), one.id, &ctx())
        .await
        .unwrap_err();
    assert!(matches!(tx_error(error), TxError::Fork(_)));
}

async fn snapshots_as_of_select_the_live_incarnation(backend: Backend) {
    let env = Env::open(backend).await;
    let parent = env.conversation().await;
    set_turn(&env, parent, 1).await;
    let first = env.append(parent, "first").await;
    env.session
        .commit(&ctx(), move |tx| {
            Box::pin(async move { tx.retire_doc(DocAddress::<Agent>::new(parent)).await })
        })
        .await
        .unwrap();
    let gap = env.append(parent, "gap").await;
    set_turn(&env, parent, 5).await;
    let second = env.append(parent, "second").await;
    let address = DocAddress::<Agent>::new(parent);
    let read = |entry: EntryId| {
        let env = &env;
        let address = address.clone();
        async move { env.session.snapshot_as_of(address, entry, &ctx()).await.unwrap() }
    };
    assert_eq!(read(first.id).await, Some(json!({ "turn": 1 })));
    assert_eq!(read(gap.id).await, None);
    assert_eq!(read(second.id).await, Some(json!({ "turn": 5 })));
    // The current value never feeds a historical read.
    let current = env
        .storage
        .find_document(&address.to_address(), DocumentPoint::Current, &ctx())
        .await
        .unwrap();
    assert!(current.is_some());
}
