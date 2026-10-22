//! Document incarnations: revisions, history, copies, addresses and lifetimes.

use pretty_assertions::assert_eq;

use super::*;

pub(crate) async fn rewindable_documents(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let first_id: DocumentId = mint(storage).await;
    let record = create(
        first_id,
        "conversation.notes",
        conversation_scope(root),
        policy(History::Rewindable, Fork::AsOf),
    );
    let created_at = commit(
        storage,
        vec![create_write(
            record.clone(),
            1,
            value(json!({ "items": ["a"], "nested": { "count": 1 } })),
        )],
    )
    .await;
    let changed_at = commit(
        storage,
        vec![delta(
            first_id,
            1,
            json!([["p", ["items"], 1, 0, ["b"]], ["s", ["nested", "count"], 2]]),
        )],
    )
    .await;
    let at_created = storage
        .document(first_id, DocumentPoint::At(created_at), &ctx())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (at_created.version, at_created.value, at_created.deltas_since_base),
        (1, value(json!({ "items": ["a"], "nested": { "count": 1 } })), 0)
    );
    let at_changed = storage
        .document(first_id, DocumentPoint::At(changed_at), &ctx())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        at_changed.value,
        value(json!({ "items": ["a", "b"], "nested": { "count": 2 } }))
    );
    assert_eq!(at_changed.deltas_since_base, 1);
    assert_eq!(
        document_value(storage, first_id, DocumentPoint::Current).await,
        Some(value(json!({ "items": ["a", "b"], "nested": { "count": 2 } })))
    );

    let checkpoint_at = commit(
        storage,
        vec![base(
            first_id,
            2,
            value(json!({ "items": ["checkpoint"], "nested": { "count": 3 } })),
        )],
    )
    .await;
    let replaced_at = commit(
        storage,
        vec![delta(
            first_id,
            2,
            json!([["r", { "items": ["replacement"], "nested": { "count": 4 } }]]),
        )],
    )
    .await;
    let read = |at| async move { storage.document(first_id, at, &ctx()).await.unwrap().unwrap() };
    let changed = read(DocumentPoint::At(changed_at)).await;
    assert_eq!(changed.version, 1);
    assert_eq!(
        changed.value,
        value(json!({ "items": ["a", "b"], "nested": { "count": 2 } }))
    );
    let checkpoint = read(DocumentPoint::At(checkpoint_at)).await;
    assert_eq!(
        (checkpoint.version, checkpoint.value, checkpoint.deltas_since_base),
        (
            2,
            value(json!({ "items": ["checkpoint"], "nested": { "count": 3 } })),
            0
        )
    );
    let replaced = read(DocumentPoint::At(replaced_at)).await;
    assert_eq!(
        (replaced.value, replaced.deltas_since_base),
        (value(json!({ "items": ["replacement"], "nested": { "count": 4 } })), 1)
    );
    assert_eq!(read(DocumentPoint::Current).await.deltas_since_base, 1);

    let second_id: DocumentId = mint(storage).await;
    let retired_at = commit(
        storage,
        vec![
            create_write(
                DocumentCreate {
                    id: second_id,
                    ..record.clone()
                },
                1,
                value(json!({ "items": ["new"] })),
            ),
            StorageWrite::DocumentRetire { id: first_id },
            delta(first_id, 2, json!([["s", ["retiring"], true]])),
        ],
    )
    .await;
    let address = record.address();
    let found = |at| {
        let address = address.clone();
        async move { storage.find_document(&address, at, &ctx()).await.unwrap() }
    };
    assert_eq!(found(DocumentPoint::At(changed_at)).await.unwrap().id, first_id);
    let current = found(DocumentPoint::At(retired_at)).await.unwrap();
    assert_eq!((current.id, current.created_at), (second_id, retired_at));
    let first_retired = storage
        .document(first_id, DocumentPoint::At(Seq(retired_at.0 - 1)), &ctx())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(first_retired.record.retired_at, Some(retired_at));
    let scan = |at| async move {
        let query = DocumentQuery {
            scope: conversation_scope(root),
            at,
            kind: None,
        };
        ids(
            &storage.scan_documents(query, 10, None, &ctx()).await.unwrap(),
            |record| record.id,
        )
    };
    assert_eq!(scan(DocumentPoint::At(changed_at)).await, [first_id]);
    assert_eq!(scan(DocumentPoint::At(retired_at)).await, [second_id]);
    assert_eq!(
        storage
            .document(first_id, DocumentPoint::At(retired_at), &ctx())
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        document_value(storage, second_id, DocumentPoint::Current).await,
        Some(value(json!({ "items": ["new"] })))
    );
}

pub(crate) async fn long_delta_tails(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let id: DocumentId = mint(storage).await;
    let record = create(
        id,
        "conversation.long-tail",
        conversation_scope(root),
        policy(History::Rewindable, Fork::AsOf),
    );
    let rows = |prefix: &str, offset: i64| -> serde_json::Value {
        (0..512)
            .map(|index| json!({ "value": offset + index, "stable": format!("{prefix}-{index}") }))
            .collect()
    };
    let initial = json!({ "revision": 0, "rows": rows("row", 0) });
    let created_at = commit(storage, vec![create_write(record, 1, value(initial.clone()))]).await;
    let mut before = initial.clone();
    let mut before_at = created_at;
    for revision in 1..=24_i64 {
        let index = (revision * 17) % 512;
        before["rows"][index as usize]["value"] = json!(-revision);
        before["revision"] = json!(revision);
        before_at = commit(
            storage,
            vec![delta(
                id,
                1,
                json!([
                    ["s", ["rows", index, "value"], -revision],
                    ["s", ["revision"], revision]
                ]),
            )],
        )
        .await;
    }
    let replacement = json!({ "revision": 100, "rows": rows("new", 10_000) });
    let replaced_at = commit(storage, vec![delta(id, 1, json!([["r", replacement.clone()]]))]).await;
    let mut current = replacement.clone();
    for revision in 101..=124_i64 {
        let index = (revision * 19) % 512;
        current["rows"][index as usize]["value"] = json!(-revision);
        current["revision"] = json!(revision);
        commit(
            storage,
            vec![delta(
                id,
                1,
                json!([
                    ["s", ["rows", index, "value"], -revision],
                    ["s", ["revision"], revision]
                ]),
            )],
        )
        .await;
    }
    assert_eq!(
        document_value(storage, id, DocumentPoint::At(created_at)).await,
        Some(value(initial))
    );
    assert_eq!(
        document_value(storage, id, DocumentPoint::At(before_at)).await,
        Some(value(before))
    );
    assert_eq!(
        document_value(storage, id, DocumentPoint::At(replaced_at)).await,
        Some(value(replacement))
    );
    let read = storage
        .document(id, DocumentPoint::Current, &ctx())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(read.value, value(current));
    // The root replacement stays a delta: the tail runs back to the creation base.
    assert_eq!(read.deltas_since_base, 49);
}

pub(crate) async fn copies_and_ambiguous_sources(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let child: ConversationId = mint(storage).await;
    let second_child: ConversationId = mint(storage).await;
    commit(
        storage,
        vec![
            StorageWrite::Conversation(ConversationRecord::new(child)),
            StorageWrite::Conversation(ConversationRecord::new(second_child)),
        ],
    )
    .await;
    let source_id: DocumentId = mint(storage).await;
    let rewindable = policy(History::Rewindable, Fork::AsOf);
    let source = create(source_id, "copy.source", conversation_scope(root), rewindable);
    let created_at = commit(
        storage,
        vec![create_write(
            source,
            2,
            value(json!({ "count": 1, "rows": [{ "value": "base" }] })),
        )],
    )
    .await;
    commit(
        storage,
        vec![delta(
            source_id,
            2,
            json!([["s", ["count"], 2], ["p", ["rows"], 1, 0, [{ "value": "current" }]]]),
        )],
    )
    .await;
    let historical_id: DocumentId = mint(storage).await;
    let current_id: DocumentId = mint(storage).await;
    let retired_id: DocumentId = mint(storage).await;
    let child_record = |id, conversation| create(id, "copy.source", conversation_scope(conversation), rewindable);
    commit(
        storage,
        vec![
            copy(
                child_record(historical_id, child),
                source_id,
                DocumentPoint::At(created_at),
            ),
            copy(
                child_record(current_id, second_child),
                source_id,
                DocumentPoint::Current,
            ),
            copy(child_record(retired_id, root), source_id, DocumentPoint::Current),
            StorageWrite::DocumentRetire { id: retired_id },
        ],
    )
    .await;
    let historical = storage
        .document(historical_id, DocumentPoint::Current, &ctx())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (historical.version, historical.value),
        (2, value(json!({ "count": 1, "rows": [{ "value": "base" }] })))
    );
    let copied = value(json!({ "count": 2, "rows": [{ "value": "base" }, { "value": "current" }] }));
    let current = storage
        .document(current_id, DocumentPoint::Current, &ctx())
        .await
        .unwrap()
        .unwrap();
    assert_eq!((current.version, current.value), (2, copied.clone()));
    assert_eq!(document_value(storage, retired_id, DocumentPoint::Current).await, None);

    commit(
        storage,
        vec![
            base(source_id, 2, value(json!({ "count": 99, "rows": [] }))),
            StorageWrite::DocumentRetire { id: source_id },
        ],
    )
    .await;
    assert_eq!(
        document_value(storage, current_id, DocumentPoint::Current).await,
        Some(copied.clone())
    );

    let latest_source_id: DocumentId = mint(storage).await;
    let latest_copy_id: DocumentId = mint(storage).await;
    let latest = create(
        latest_source_id,
        "copy.latest",
        conversation_scope(root),
        policy(History::Latest, Fork::Current),
    );
    commit(
        storage,
        vec![create_write(latest.clone(), 4, value(json!({ "retained": "copy" })))],
    )
    .await;
    commit(
        storage,
        vec![copy(
            DocumentCreate {
                id: latest_copy_id,
                scope: conversation_scope(child),
                ..latest
            },
            latest_source_id,
            DocumentPoint::Current,
        )],
    )
    .await;
    commit(
        storage,
        vec![
            base(latest_source_id, 4, value(json!({ "retained": "source-only" }))),
            StorageWrite::DocumentRetire { id: latest_source_id },
        ],
    )
    .await;
    let latest_copy = storage
        .document(latest_copy_id, DocumentPoint::Current, &ctx())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (latest_copy.version, latest_copy.value),
        (4, value(json!({ "retained": "copy" })))
    );

    // Rejected without effect: the copy and the retirement in the same batch are both absent.
    let conflict_id: DocumentId = mint(storage).await;
    reject(
        storage,
        vec![
            copy(child_record(conflict_id, child), current_id, DocumentPoint::Current),
            StorageWrite::DocumentRetire { id: current_id },
        ],
        "was rejected",
    )
    .await;
    assert_eq!(document_value(storage, conflict_id, DocumentPoint::Current).await, None);
    assert_eq!(
        document_value(storage, current_id, DocumentPoint::Current).await,
        Some(copied)
    );

    let mismatch_id: DocumentId = mint(storage).await;
    let mismatch = DocumentCreate {
        kind: "copy.mismatch".into(),
        ..child_record(mismatch_id, child)
    };
    reject(
        storage,
        vec![copy(mismatch, current_id, DocumentPoint::Current)],
        "was rejected",
    )
    .await;
    assert_eq!(document_value(storage, mismatch_id, DocumentPoint::Current).await, None);
}

pub(crate) async fn version_transitions_and_current_only_history(fx: Fixture) {
    let storage = &*fx.storage;
    create_root(storage).await;
    let id: DocumentId = mint(storage).await;
    commit(
        storage,
        vec![create_write(
            create(id, "session.settings", DocumentScope::Session, None),
            1,
            value(json!({ "count": 1 })),
        )],
    )
    .await;
    commit(storage, vec![delta(id, 1, json!([["s", ["count"], 2]]))]).await;
    let migrated_at = commit(storage, vec![base(id, 2, value(json!({ "count": 3 })))]).await;
    let current = storage
        .document(id, DocumentPoint::Current, &ctx())
        .await
        .unwrap()
        .unwrap();
    assert_eq!((current.version, current.value), (2, value(json!({ "count": 3 }))));
    let error = storage
        .document(id, DocumentPoint::At(migrated_at), &ctx())
        .await
        .unwrap_err();
    assert!(
        error.is_request() && error.to_string().contains("does not retain historical content"),
        "{error}"
    );
    reject(
        storage,
        vec![delta(id, 1, json!([["s", ["count"], 4]]))],
        "version transition requires a base",
    )
    .await;
    assert_eq!(
        document_value(storage, id, DocumentPoint::Current).await,
        Some(value(json!({ "count": 3 })))
    );
    commit(storage, vec![StorageWrite::DocumentRetire { id }]).await;
    assert_eq!(document_value(storage, id, DocumentPoint::Current).await, None);
    reject(storage, vec![delta(id, 2, json!([]))], "is retired").await;
}

pub(crate) async fn addresses_and_scope_scans(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let first_id: DocumentId = mint(storage).await;
    let second_id: DocumentId = mint(storage).await;
    let conversation_id: DocumentId = mint(storage).await;
    let task_id: TaskId = mint(storage).await;
    let task_singleton: DocumentId = mint(storage).await;
    let task_family: DocumentId = mint(storage).await;
    let task_other: DocumentId = mint(storage).await;
    let keyed = |record: DocumentCreate, key: &str| DocumentCreate {
        key: Some(key.to_owned()),
        ..record
    };
    let task_scope = DocumentScope::Task { task_id };
    let owner = |name: &str| value(json!({ "owner": name }));
    let created_at = commit(
        storage,
        vec![
            StorageWrite::Task(pending_task(task_id, root)),
            create_write(
                keyed(create(first_id, "cache", DocumentScope::Session, None), "__proto__"),
                1,
                owner("first"),
            ),
            create_write(
                keyed(create(second_id, "cache", DocumentScope::Session, None), "constructor"),
                1,
                owner("second"),
            ),
            create_write(
                keyed(
                    create(
                        conversation_id,
                        "cache",
                        conversation_scope(root),
                        policy(History::Latest, Fork::Current),
                    ),
                    "__proto__",
                ),
                1,
                owner("conversation"),
            ),
            create_write(
                create(task_singleton, "task.cache", task_scope, None),
                1,
                owner("singleton"),
            ),
            create_write(
                keyed(create(task_family, "task.cache", task_scope, None), "member"),
                1,
                owner("family"),
            ),
            create_write(create(task_other, "task.other", task_scope, None), 1, owner("other")),
        ],
    )
    .await;
    let find = |kind: &str, scope, key: Option<&str>| {
        let address = DocumentAddress {
            kind: kind.to_owned(),
            scope,
            key: key.map(str::to_owned),
        };
        async move {
            storage
                .find_document(&address, DocumentPoint::Current, &ctx())
                .await
                .unwrap()
                .map(|record| record.id)
        }
    };
    assert_eq!(
        find("cache", DocumentScope::Session, Some("__proto__")).await,
        Some(first_id)
    );
    let session = DocumentQuery::current(DocumentScope::Session);
    let first = storage.scan_documents(session.clone(), 1, None, &ctx()).await.unwrap();
    assert_eq!(first.items.len(), 1);
    let second = storage
        .scan_documents(session, 1, first.next.clone(), &ctx())
        .await
        .unwrap();
    let both: Vec<DocumentId> = first
        .items
        .iter()
        .chain(&second.items)
        .map(|record| record.id)
        .collect();
    assert_eq!(both, [first_id, second_id]);
    let conversation = storage
        .scan_documents(DocumentQuery::current(conversation_scope(root)), 10, None, &ctx())
        .await
        .unwrap();
    assert_eq!(ids(&conversation, |record| record.id), [conversation_id]);
    assert_eq!(find("task.cache", task_scope, None).await, Some(task_singleton));
    assert_eq!(find("task.cache", task_scope, Some("member")).await, Some(task_family));
    let of_kind = DocumentQuery {
        kind: Some("task.cache".into()),
        ..DocumentQuery::current(task_scope)
    };
    let page = storage.scan_documents(of_kind, 10, None, &ctx()).await.unwrap();
    assert_eq!(ids(&page, |record| record.id), [task_singleton, task_family]);
    let error = storage
        .document(task_singleton, DocumentPoint::At(created_at), &ctx())
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("does not retain historical content"),
        "{error}"
    );
}

pub(crate) async fn document_lifecycle_and_empty_lifetimes(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let first_id: DocumentId = mint(storage).await;
    let second_id: DocumentId = mint(storage).await;
    let record = create(first_id, "singleton", DocumentScope::Session, None);
    commit(
        storage,
        vec![create_write(record.clone(), 1, value(json!({ "value": 1 })))],
    )
    .await;
    reject(
        storage,
        vec![
            create_write(
                DocumentCreate {
                    id: second_id,
                    ..record.clone()
                },
                1,
                value(json!({ "value": 2 })),
            ),
            delta(first_id, 1, json!([])),
        ],
        "already has a current incarnation",
    )
    .await;
    assert_eq!(
        document_value(storage, first_id, DocumentPoint::Current).await,
        Some(value(json!({ "value": 1 })))
    );
    assert_eq!(document_value(storage, second_id, DocumentPoint::Current).await, None);

    let empty_id: DocumentId = mint(storage).await;
    let empty = DocumentCreate {
        key: Some("empty".into()),
        ..create(
            empty_id,
            "singleton",
            conversation_scope(root),
            policy(History::Rewindable, Fork::Initial),
        )
    };
    let address = empty.address();
    let empty_at = commit(
        storage,
        vec![
            create_write(empty, 1, value(json!({}))),
            StorageWrite::DocumentRetire { id: empty_id },
        ],
    )
    .await;
    assert_eq!(document_value(storage, empty_id, DocumentPoint::Current).await, None);
    assert_eq!(
        document_value(storage, empty_id, DocumentPoint::At(empty_at)).await,
        None
    );
    assert_eq!(
        storage
            .find_document(&address, DocumentPoint::At(empty_at), &ctx())
            .await
            .unwrap(),
        None
    );
}

pub(crate) async fn document_failures_roll_back_tables(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let task_id: TaskId = mint(storage).await;
    let submission_id: SubmissionId = mint(storage).await;
    let document_id: DocumentId = mint(storage).await;
    let task = pending_task(task_id, root);
    let submission = input(submission_id, root, Some("atomic"));
    let record = create(document_id, "atomic", DocumentScope::Session, None);
    let baseline = commit(
        storage,
        vec![
            StorageWrite::Task(task.clone()),
            StorageWrite::Submission(submission.clone()),
            create_write(record.clone(), 1, value(json!({ "count": 1 }))),
        ],
    )
    .await;
    let entry_id: EntryId = mint(storage).await;
    let conflicting: DocumentId = mint(storage).await;
    let mut running = task.clone();
    running.state = TaskState::Running {
        checkpoint: value(json!({ "phase": "effect" })),
    };
    let mut failed = submission.clone();
    failed.state = SubmissionState::Unanswered {
        entry: None,
        reason: "failed".into(),
        detail: None,
    };
    reject(
        storage,
        vec![
            StorageWrite::Task(running),
            StorageWrite::Submission(failed),
            StorageWrite::Entry(EntryRecord::new(entry_id, root, "transient")),
            create_write(
                DocumentCreate {
                    id: conflicting,
                    ..record.clone()
                },
                1,
                value(json!({ "count": 2 })),
            ),
        ],
        "already has a current incarnation",
    )
    .await;
    assert_eq!(storage.task(task_id, &ctx()).await.unwrap(), Some(task.clone()));
    assert_eq!(scan_status(storage, TaskStatus::Pending).await, [task]);
    assert_eq!(
        storage.submission_by_request(root, "atomic", &ctx()).await.unwrap(),
        Some(submission)
    );
    assert_eq!(storage.entry(entry_id, &ctx()).await.unwrap(), None);
    assert_eq!(document_value(storage, conflicting, DocumentPoint::Current).await, None);
    let found = storage
        .find_document(&record.address(), DocumentPoint::Current, &ctx())
        .await
        .unwrap();
    assert_eq!(found.map(|record| record.id), Some(document_id));
    let after = commit(storage, vec![delta(document_id, 1, json!([["s", ["count"], 3]]))]).await;
    assert!(after > baseline);
}

pub(crate) async fn copy_source_rules(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let child: ConversationId = mint(storage).await;
    commit(
        storage,
        vec![StorageWrite::Conversation(ConversationRecord::new(child))],
    )
    .await;
    let session_id: DocumentId = mint(storage).await;
    let latest_id: DocumentId = mint(storage).await;
    let retired_id: DocumentId = mint(storage).await;
    let latest_policy = policy(History::Latest, Fork::Current);
    let created_at = commit(
        storage,
        vec![
            create_write(
                create(session_id, "shared", DocumentScope::Session, None),
                1,
                value(json!({})),
            ),
            create_write(
                create(latest_id, "latest", conversation_scope(root), latest_policy),
                1,
                value(json!({})),
            ),
            create_write(
                create(retired_id, "retired", conversation_scope(root), latest_policy),
                1,
                value(json!({})),
            ),
            StorageWrite::DocumentRetire { id: retired_id },
        ],
    )
    .await;
    let copy_id: DocumentId = mint(storage).await;
    let into_child = |kind: &str, policy| create(copy_id, kind, conversation_scope(child), policy);
    let cases = [
        // A session document is shared, never copied.
        (into_child("shared", latest_policy), session_id, DocumentPoint::Current),
        // A latest document has no historical content.
        (
            into_child("latest", latest_policy),
            latest_id,
            DocumentPoint::At(created_at),
        ),
        // A retired source is not alive at the point.
        (into_child("retired", latest_policy), retired_id, DocumentPoint::Current),
        // The child's policy must equal the source's.
        (
            into_child("latest", policy(History::Latest, Fork::Initial)),
            latest_id,
            DocumentPoint::Current,
        ),
        // An unknown source.
        (
            into_child("latest", latest_policy),
            DocumentId(999_999),
            DocumentPoint::Current,
        ),
    ];
    for (record, source, at) in cases {
        reject(
            storage,
            vec![copy(record, source, at)],
            &format!("Document copy {copy_id} was rejected"),
        )
        .await;
        assert_eq!(document_value(storage, copy_id, DocumentPoint::Current).await, None);
    }
    commit(
        storage,
        vec![copy(
            into_child("latest", latest_policy),
            latest_id,
            DocumentPoint::Current,
        )],
    )
    .await;
    assert_eq!(
        document_value(storage, copy_id, DocumentPoint::Current).await,
        Some(value(json!({})))
    );
}

pub(crate) async fn singleton_and_empty_key(fx: Fixture) {
    let storage = &*fx.storage;
    create_root(storage).await;
    let singleton: DocumentId = mint(storage).await;
    let empty_key: DocumentId = mint(storage).await;
    let record = create(singleton, "shared", DocumentScope::Session, None);
    commit(
        storage,
        vec![
            create_write(record.clone(), 1, value(json!({ "singleton": true }))),
            create_write(
                DocumentCreate {
                    id: empty_key,
                    key: Some(String::new()),
                    ..record.clone()
                },
                1,
                value(json!({ "singleton": false })),
            ),
        ],
    )
    .await;
    let find = |key: Option<&str>| {
        let address = DocumentAddress {
            key: key.map(str::to_owned),
            ..record.address()
        };
        async move {
            storage
                .find_document(&address, DocumentPoint::Current, &ctx())
                .await
                .unwrap()
                .map(|record| record.id)
        }
    };
    assert_eq!(find(None).await, Some(singleton));
    assert_eq!(find(Some("")).await, Some(empty_key));
    assert_eq!(find(Some("other")).await, None);
    let page = storage
        .scan_documents(DocumentQuery::current(DocumentScope::Session), 10, None, &ctx())
        .await
        .unwrap();
    assert_eq!(ids(&page, |record| record.id), [singleton, empty_key]);
}
