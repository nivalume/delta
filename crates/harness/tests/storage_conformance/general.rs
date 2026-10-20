//! The id namespace, indexed strings, close, and invalid requests.

use pretty_assertions::assert_eq;

use super::*;

pub(crate) async fn indexed_strings_lossless(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    // Rust strings have no lone surrogates; these exercise quoting, escapes, non-ASCII,
    // LIKE metacharacters and the empty string instead.
    let names = ["quote\"d 'kind'", "ünïcødé ☃ 🦀", "", "line\nbreak\\", "%_"];
    let mut writes = Vec::new();
    let mut expected = Vec::new();
    for name in names {
        let task: TaskId = mint(storage).await;
        let submission: SubmissionId = mint(storage).await;
        let kind_document: DocumentId = mint(storage).await;
        let key_document: DocumentId = mint(storage).await;
        let mut task_record = pending_task(task, root);
        name.clone_into(&mut task_record.kind);
        writes.push(StorageWrite::Task(task_record));
        writes.push(StorageWrite::Submission(input(submission, root, Some(name))));
        writes.push(create_write(
            create(kind_document, name, DocumentScope::Session, None),
            1,
            value(json!({})),
        ));
        writes.push(create_write(
            DocumentCreate {
                key: Some(name.to_owned()),
                ..create(key_document, "family", DocumentScope::Session, None)
            },
            1,
            value(json!({})),
        ));
        expected.push((name, task, submission, kind_document, key_document));
    }
    commit(storage, writes).await;
    for (name, task, submission, kind_document, key_document) in expected {
        let by_kind = TaskQuery {
            kind: Some(name.to_owned()),
            ..TaskQuery::default()
        };
        let page = storage.scan_tasks(by_kind, 10, None, &ctx()).await.unwrap();
        assert_eq!(ids(&page, |task| task.id), [task], "{name:?}");
        assert_eq!(storage.task(task, &ctx()).await.unwrap().unwrap().kind, name);
        let by_request = storage
            .submission_by_request(root, name, &ctx())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            (by_request.id, by_request.request_id.as_deref()),
            (submission, Some(name))
        );
        let address = |kind: &str, key: Option<&str>| DocumentAddress {
            kind: kind.to_owned(),
            scope: DocumentScope::Session,
            key: key.map(str::to_owned),
        };
        let found = storage
            .find_document(&address(name, None), DocumentPoint::Current, &ctx())
            .await
            .unwrap();
        assert_eq!(found.map(|record| record.id), Some(kind_document), "{name:?}");
        let found = storage
            .find_document(&address("family", Some(name)), DocumentPoint::Current, &ctx())
            .await
            .unwrap();
        assert_eq!(found.map(|record| record.id), Some(key_document), "{name:?}");
        let of_kind = DocumentQuery {
            kind: Some(name.to_owned()),
            ..DocumentQuery::current(DocumentScope::Session)
        };
        let page = storage.scan_documents(of_kind, 10, None, &ctx()).await.unwrap();
        assert_eq!(ids(&page, |record| record.id), [kind_document], "{name:?}");
    }
}

pub(crate) async fn global_id_namespace(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    commit(storage, vec![StorageWrite::Entry(entry(EntryId(100), root))]).await;
    assert_eq!(storage.mint_id().await.unwrap(), RawId(101));
    reject(
        storage,
        vec![StorageWrite::Task(pending_task(TaskId(100), root))],
        "ID 100 already belongs to entry",
    )
    .await;
    reject(
        storage,
        vec![
            StorageWrite::Task(pending_task(TaskId(102), root)),
            StorageWrite::Submission(input(SubmissionId(102), root, None)),
        ],
        "ID 102 is written as two record types",
    )
    .await;
    reject(
        storage,
        vec![
            StorageWrite::Entry(entry(EntryId(103), root)),
            StorageWrite::Entry(entry(EntryId(103), root)),
        ],
        "ID 103 is written more than once",
    )
    .await;

    commit(
        storage,
        vec![StorageWrite::Entry(EntryRecord::new(
            EntryId(fx.max_id),
            root,
            "last-id",
        ))],
    )
    .await;
    for _ in 0..2 {
        let error = storage.mint_id().await.unwrap_err();
        assert!(matches!(error, StorageError::IdSpaceExhausted), "{error:?}");
        assert_eq!(error.to_string(), "ID space is exhausted");
    }
}

pub(crate) async fn rejects_after_close(fx: Fixture) {
    let storage = &*fx.storage;
    create_root(storage).await;
    storage.close(&ctx()).await.unwrap();
    let closed = |error: StorageError| assert!(matches!(error, StorageError::Closed), "{error:?}");
    closed(storage.conversation(ROOT_CONVERSATION_ID, &ctx()).await.unwrap_err());
    closed(storage.commit(Vec::new(), &ctx()).await.unwrap_err());
    closed(storage.mint_id().await.unwrap_err());
    closed(
        storage
            .scan_entries(EntryQuery::new(ROOT_CONVERSATION_ID), 1, None, &ctx())
            .await
            .unwrap_err(),
    );
    storage.close(&ctx()).await.unwrap();
}

pub(crate) async fn invalid_reads_are_requests(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let request = |error: StorageError| assert!(error.is_request(), "{error:?}");
    request(
        storage
            .scan_conversations(ConversationQuery::default(), 0, None, &ctx())
            .await
            .unwrap_err(),
    );
    for cursor in [json!({ "after": "x" }), json!({ "order": "ascending" }), json!(7)] {
        request(
            storage
                .scan_tasks(TaskQuery::default(), 1, Some(Cursor(value(cursor))), &ctx())
                .await
                .unwrap_err(),
        );
    }
    let descending = Cursor(value(json!({ "after": 1, "order": "descending" })));
    request(
        storage
            .scan_documents(
                DocumentQuery::current(DocumentScope::Session),
                1,
                Some(descending),
                &ctx(),
            )
            .await
            .unwrap_err(),
    );
    request(
        storage
            .find_latest_head_marker(ConversationId(999_999), None, &ctx())
            .await
            .unwrap_err(),
    );
    assert_eq!(
        storage
            .document(DocumentId(999_999), DocumentPoint::At(Seq(1)), &ctx())
            .await
            .unwrap(),
        None
    );
    // Unknown ids are absence, and the storage stays usable.
    assert_eq!(storage.task(TaskId(999_999), &ctx()).await.unwrap(), None);
    commit(storage, vec![StorageWrite::Entry(entry(mint(storage).await, root))]).await;
    let ops: Vec<Op> = vec![Op::Set(Path::from_iter(["a"]), Json::Null)];
    reject(
        storage,
        vec![StorageWrite::DocumentChange {
            id: DocumentId(999_999),
            content: DocumentContent::Delta { version: 1, ops },
        }],
        "does not exist",
    )
    .await;
}
