//! Conversations, entries and fork history, tasks and submissions.

use pretty_assertions::assert_eq;

use super::*;

pub(crate) async fn reserves_root_id(fx: Fixture) {
    let storage = &*fx.storage;
    assert_eq!(storage.mint_id().await.unwrap(), RawId(2));
    assert_eq!(create_root(storage).await, ROOT_CONVERSATION_ID);
    assert_eq!(
        storage.conversation(ROOT_CONVERSATION_ID, &ctx()).await.unwrap(),
        Some(ConversationRecord::new(ROOT_CONVERSATION_ID))
    );
    reject(
        storage,
        vec![StorageWrite::Conversation(ConversationRecord::new(
            ROOT_CONVERSATION_ID,
        ))],
        "ID 1 already belongs to conversation",
    )
    .await;
}

pub(crate) async fn mixed_writes_are_atomic(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let entry_id: EntryId = mint(storage).await;
    let task_id: TaskId = mint(storage).await;
    let submission_id: SubmissionId = mint(storage).await;
    let task = pending_task(task_id, root);
    let mut placed = input(submission_id, root, Some("request-1"));
    placed.state = SubmissionState::Placed { entry: entry_id };
    let mut user = EntryRecord::new(entry_id, root, "user");
    user.data = Some(value(json!({ "text": "hello" })));
    let initial = commit(
        storage,
        vec![
            StorageWrite::Entry(user.clone()),
            StorageWrite::Task(task.clone()),
            StorageWrite::Submission(placed.clone()),
        ],
    )
    .await;
    assert_eq!(
        storage.entry(entry_id, &ctx()).await.unwrap(),
        Some(StoredEntry {
            entry: user,
            commit_seq: initial
        })
    );
    assert_eq!(storage.task(task_id, &ctx()).await.unwrap(), Some(task.clone()));
    assert_eq!(
        storage.submission(submission_id, &ctx()).await.unwrap(),
        Some(placed.clone())
    );

    let transient: EntryId = mint(storage).await;
    let mut running = task.clone();
    running.state = TaskState::Running {
        checkpoint: value(json!({ "phase": "effect" })),
    };
    let mut done = placed.clone();
    done.state = SubmissionState::Done {
        entry: entry_id,
        answer: Some(transient),
    };
    reject(
        storage,
        vec![
            StorageWrite::Task(running),
            StorageWrite::Submission(done),
            StorageWrite::Entry(EntryRecord::new(transient, root, "assistant")),
            StorageWrite::Conversation(ConversationRecord::new(root)),
        ],
        "ID 1 already belongs to conversation",
    )
    .await;
    assert_eq!(storage.task(task_id, &ctx()).await.unwrap(), Some(task));
    assert_eq!(storage.submission(submission_id, &ctx()).await.unwrap(), Some(placed));
    assert_eq!(storage.entry(transient, &ctx()).await.unwrap(), None);
    let after = commit(storage, vec![StorageWrite::Entry(entry(mint(storage).await, root))]).await;
    assert!(after > initial);
}

pub(crate) async fn returned_records_are_owned(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let entry_id: EntryId = mint(storage).await;
    let task_id: TaskId = mint(storage).await;
    let submission_id: SubmissionId = mint(storage).await;
    let mut note = EntryRecord::new(entry_id, root, "note");
    note.data = Some(value(json!({ "nested": [1, 2] })));
    let mut task = pending_task(task_id, root);
    task.state = TaskState::Pending {
        checkpoint: value(json!({ "phase": "ready", "nested": { "count": 1 } })),
    };
    let mut failed = input(submission_id, root, None);
    failed.state = SubmissionState::Unanswered {
        entry: None,
        reason: "failed".into(),
        detail: Some(value(json!({ "codes": ["initial"] }))),
    };
    let writes = vec![
        StorageWrite::Entry(note.clone()),
        StorageWrite::Task(task.clone()),
        StorageWrite::Submission(failed.clone()),
    ];
    // The caller keeps and mutates its copy of the written values.
    let mut retained = writes.clone();
    commit(storage, writes).await;
    if let StorageWrite::Entry(written) = &mut retained[0] {
        written.data = Some(value(json!({ "nested": [1, 2, 3] })));
    }

    let mut read = storage.entry(entry_id, &ctx()).await.unwrap().unwrap();
    assert_eq!(read.entry, note);
    read.entry.data = Some(value(json!({ "nested": [9] })));
    let mut read_task = storage.task(task_id, &ctx()).await.unwrap().unwrap();
    read_task.state = TaskState::Pending { checkpoint: Json::Null };
    let mut read_submission = storage.submission(submission_id, &ctx()).await.unwrap().unwrap();
    read_submission.state = SubmissionState::Queued;

    assert_eq!(storage.entry(entry_id, &ctx()).await.unwrap().unwrap().entry, note);
    assert_eq!(storage.task(task_id, &ctx()).await.unwrap(), Some(task));
    assert_eq!(storage.submission(submission_id, &ctx()).await.unwrap(), Some(failed));
}

pub(crate) async fn prototype_like_keys_round_trip(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let entry_id: EntryId = mint(storage).await;
    let data = delta_chord::json::from_str(
        r#"{"__proto__":{"polluted":false},"constructor":{"label":"stored"},"toString":"value"}"#,
    )
    .unwrap();
    let mut note = EntryRecord::new(entry_id, root, "note");
    note.data = Some(data.clone());
    commit(storage, vec![StorageWrite::Entry(note)]).await;
    let read = storage
        .entry(entry_id, &ctx())
        .await
        .unwrap()
        .unwrap()
        .entry
        .data
        .unwrap();
    assert_eq!(read, data);
    assert_eq!(read.get("__proto__"), Some(&value(json!({ "polluted": false }))));
    assert_eq!(read.get("constructor"), Some(&value(json!({ "label": "stored" }))));
    assert_eq!(read.get("toString"), Some(&Json::from("value")));
}

pub(crate) async fn entries_out_of_id_order(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let mut marker = EntryRecord::new(EntryId(20), root, "marker");
    marker.head = Some(EntryId(10));
    commit(
        storage,
        vec![
            StorageWrite::Entry(entry(EntryId(30), root)),
            StorageWrite::Entry(entry(EntryId(10), root)),
            StorageWrite::Entry(marker),
        ],
    )
    .await;
    let (found, _) = entry_ids(storage, EntryQuery::new(root), 10, None).await;
    assert_eq!(found, [EntryId(30), EntryId(20), EntryId(10)]);
    let head = storage
        .find_latest_head_marker(root, None, &ctx())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(head.id, EntryId(20));
}

pub(crate) async fn entry_cursor_after_newer_commit(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let oldest: EntryId = mint(storage).await;
    let middle: EntryId = mint(storage).await;
    let newest: EntryId = mint(storage).await;
    commit(
        storage,
        [oldest, middle, newest]
            .map(|id| StorageWrite::Entry(entry(id, root)))
            .to_vec(),
    )
    .await;
    let (first, next) = entry_ids(storage, EntryQuery::new(root), 2, None).await;
    assert_eq!(first, [newest, middle]);
    commit(storage, vec![StorageWrite::Entry(entry(mint(storage).await, root))]).await;
    let (second, next) = entry_ids(storage, EntryQuery::new(root), 2, next).await;
    assert_eq!(second, [oldest]);
    assert_eq!(next, None);
}

pub(crate) async fn conversation_pages(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let second: ConversationId = mint(storage).await;
    let third: ConversationId = mint(storage).await;
    commit(
        storage,
        vec![
            StorageWrite::Conversation(ConversationRecord::new(third)),
            StorageWrite::Conversation(ConversationRecord::new(second)),
        ],
    )
    .await;
    let first = storage
        .scan_conversations(ConversationQuery::default(), 2, None, &ctx())
        .await
        .unwrap();
    assert_eq!(ids(&first, |record| record.id), [root, second]);
    let text = serde_json::to_string(&first.next.unwrap()).unwrap();
    let cursor: Cursor = serde_json::from_str(&text).unwrap();
    let page = storage
        .scan_conversations(ConversationQuery::default(), 2, Some(cursor), &ctx())
        .await
        .unwrap();
    assert_eq!(ids(&page, |record| record.id), [third]);
    assert_eq!(page.next, None);
}

pub(crate) async fn conversation_owner_filters(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let other_owner: ConversationId = mint(storage).await;
    let first_task: TaskId = mint(storage).await;
    let second_task: TaskId = mint(storage).await;
    let first: ConversationId = mint(storage).await;
    let second: ConversationId = mint(storage).await;
    let third: ConversationId = mint(storage).await;
    let owned = |id, conversation_id, task_id| {
        StorageWrite::Conversation(ConversationRecord {
            id,
            parent: None,
            owner: Some(ConversationOwner {
                conversation_id,
                task_id,
            }),
        })
    };
    commit(
        storage,
        vec![
            StorageWrite::Conversation(ConversationRecord::new(other_owner)),
            owned(first, root, first_task),
            owned(second, root, second_task),
            owned(third, other_owner, first_task),
        ],
    )
    .await;
    let by_root = ConversationQuery {
        owner_conversation_id: Some(root),
        ..ConversationQuery::default()
    };
    let page = storage
        .scan_conversations(by_root.clone(), 1, None, &ctx())
        .await
        .unwrap();
    assert_eq!(ids(&page, |record| record.id), [first]);
    let page = storage.scan_conversations(by_root, 1, page.next, &ctx()).await.unwrap();
    assert_eq!(ids(&page, |record| record.id), [second]);
    assert_eq!(page.next, None);
    let by_task = ConversationQuery {
        owner_task_id: Some(first_task),
        ..ConversationQuery::default()
    };
    let page = storage.scan_conversations(by_task, 10, None, &ctx()).await.unwrap();
    assert_eq!(ids(&page, |record| record.id), [first, third]);
    let both = ConversationQuery {
        owner_conversation_id: Some(root),
        owner_task_id: Some(first_task),
        order: None,
    };
    let page = storage.scan_conversations(both, 10, None, &ctx()).await.unwrap();
    assert_eq!(ids(&page, |record| record.id), [first]);
}

pub(crate) async fn deep_fork_history(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let root_first: EntryId = mint(storage).await;
    let root_fork_point: EntryId = mint(storage).await;
    let root_excluded_same_commit: EntryId = mint(storage).await;
    let mut root_marker = EntryRecord::new(root_fork_point, root, "marker");
    root_marker.head = Some(root_first);
    let root_seq = commit(
        storage,
        vec![
            StorageWrite::Entry(entry(root_first, root)),
            StorageWrite::Entry(root_marker),
            StorageWrite::Entry(entry(root_excluded_same_commit, root)),
        ],
    )
    .await;
    let fork = |id, conversation_id, at| {
        StorageWrite::Conversation(ConversationRecord {
            id,
            parent: Some(ConversationParent { conversation_id, at }),
            owner: None,
        })
    };
    let child: ConversationId = mint(storage).await;
    commit(storage, vec![fork(child, root, root_fork_point)]).await;
    let child_fork_point: EntryId = mint(storage).await;
    let child_excluded: EntryId = mint(storage).await;
    commit(
        storage,
        vec![
            StorageWrite::Entry(EntryRecord::new(child_fork_point, child, "note")),
            StorageWrite::Entry(entry(child_excluded, child)),
        ],
    )
    .await;
    let root_excluded_later: EntryId = mint(storage).await;
    commit(storage, vec![StorageWrite::Entry(entry(root_excluded_later, root))]).await;
    let grandchild: ConversationId = mint(storage).await;
    commit(storage, vec![fork(grandchild, child, child_fork_point)]).await;
    let grandchild_head: EntryId = mint(storage).await;
    let grandchild_tail: EntryId = mint(storage).await;
    let mut grandchild_marker = EntryRecord::new(grandchild_head, grandchild, "marker");
    grandchild_marker.head = Some(grandchild_head);
    let grandchild_seq = commit(
        storage,
        vec![
            StorageWrite::Entry(grandchild_marker),
            StorageWrite::Entry(entry(grandchild_tail, grandchild)),
        ],
    )
    .await;
    let child_excluded_later: EntryId = mint(storage).await;
    commit(storage, vec![StorageWrite::Entry(entry(child_excluded_later, child))]).await;

    let newest = EntryQuery::new(grandchild);
    let (page, next) = entry_ids(storage, newest.clone(), 2, None).await;
    assert_eq!(page, [grandchild_tail, grandchild_head]);
    let (page, next) = entry_ids(storage, newest.clone(), 2, next).await;
    assert_eq!(page, [child_fork_point, root_fork_point]);
    let (page, next) = entry_ids(storage, newest.clone(), 2, next).await;
    assert_eq!(page, [root_first]);
    assert_eq!(next, None);

    // Oldest first: the root's segment, then each fork's, each capped at the next fork point.
    let ascending = EntryQuery {
        order: Some(ScanOrder::Ascending),
        ..newest.clone()
    };
    let (page, first_up) = entry_ids(storage, ascending.clone(), 2, None).await;
    assert_eq!(page, [root_first, root_fork_point]);
    let (page, second_up) = entry_ids(storage, ascending.clone(), 2, first_up.clone()).await;
    assert_eq!(page, [child_fork_point, grandchild_head]);
    let (page, next) = entry_ids(storage, newest.clone(), 2, second_up).await;
    assert_eq!(page, [grandchild_tail]);
    assert_eq!(next, None);
    let bounded = EntryQuery {
        min_entry_id: Some(root_fork_point),
        max_entry_id: Some(child_fork_point),
        ..ascending
    };
    assert_eq!(
        entry_ids(storage, bounded, 10, None).await.0,
        [root_fork_point, child_fork_point]
    );
    let descending = EntryQuery {
        order: Some(ScanOrder::Descending),
        ..newest.clone()
    };
    let error = storage.scan_entries(descending, 2, first_up, &ctx()).await.unwrap_err();
    assert!(error.is_request() && error.to_string().contains("cursor"), "{error}");

    let current = storage
        .find_latest_head_marker(grandchild, None, &ctx())
        .await
        .unwrap()
        .unwrap();
    assert_eq!((current.id, current.head), (grandchild_head, Some(grandchild_head)));
    let historical = storage
        .find_latest_head_marker(grandchild, Some(child_fork_point), &ctx())
        .await
        .unwrap()
        .unwrap();
    assert_eq!((historical.id, historical.head), (root_fork_point, Some(root_first)));
    assert_eq!(
        storage
            .find_latest_head_marker(grandchild, Some(root_first), &ctx())
            .await
            .unwrap(),
        None
    );

    let active = EntryQuery {
        min_entry_id: current.head,
        ..newest.clone()
    };
    let (page, next) = entry_ids(storage, active.clone(), 1, None).await;
    assert_eq!(page, [grandchild_tail]);
    assert!(next.is_some());
    let (page, next) = entry_ids(storage, active, 1, next).await;
    assert_eq!(page, [grandchild_head]);
    assert_eq!(next, None);
    let historical_range = EntryQuery {
        min_entry_id: historical.head,
        max_entry_id: Some(child_fork_point),
        ..newest
    };
    assert_eq!(
        entry_ids(storage, historical_range, 10, None).await.0,
        [child_fork_point, root_fork_point, root_first]
    );

    let stored = |id| async move { storage.entry(id, &ctx()).await.unwrap() };
    assert_eq!(
        stored(root_first).await,
        Some(StoredEntry {
            entry: entry(root_first, root),
            commit_seq: root_seq
        })
    );
    assert_eq!(stored(root_fork_point).await.unwrap().commit_seq, root_seq);
    assert_eq!(stored(grandchild_head).await.unwrap().commit_seq, grandchild_seq);
    assert_eq!(stored(grandchild_tail).await.unwrap().commit_seq, grandchild_seq);
    assert_eq!(stored(EntryId(999_999)).await, None);

    let visible = |conversation, id| async move { storage.visible_entry(conversation, id, &ctx()).await.unwrap() };
    assert_eq!(
        visible(grandchild, root_first).await,
        Some(StoredEntry {
            entry: entry(root_first, root),
            commit_seq: root_seq
        })
    );
    assert_eq!(
        visible(grandchild, child_fork_point)
            .await
            .unwrap()
            .entry
            .conversation_id,
        child
    );
    assert_eq!(
        visible(grandchild, grandchild_tail).await.unwrap().commit_seq,
        grandchild_seq
    );
    for hidden in [
        root_excluded_same_commit,
        root_excluded_later,
        child_excluded,
        child_excluded_later,
    ] {
        assert_eq!(visible(grandchild, hidden).await, None, "{hidden}");
    }
    assert_eq!(visible(root, grandchild_head).await, None);
    assert_eq!(visible(grandchild, EntryId(999_999)).await, None);
    let unknown = ConversationId(999_999);
    let error = storage.visible_entry(unknown, root_first, &ctx()).await.unwrap_err();
    assert!(
        error.is_request() && error.to_string().contains("Unknown conversation"),
        "{error}"
    );
    let error = storage
        .scan_entries(EntryQuery::new(unknown), 10, None, &ctx())
        .await
        .unwrap_err();
    assert!(
        error.is_request() && error.to_string().contains("Unknown conversation"),
        "{error}"
    );
}

#[derive(Clone, Copy)]
enum Table {
    Conversations,
    Tasks,
    Submissions,
}

async fn scan_ids(
    storage: &dyn Storage,
    table: Table,
    order: Option<ScanOrder>,
    cursor: Option<Cursor>,
) -> Result<(Vec<u64>, Option<Cursor>), StorageError> {
    Ok(match table {
        Table::Conversations => {
            let query = ConversationQuery {
                order,
                ..ConversationQuery::default()
            };
            let page = storage.scan_conversations(query, 2, cursor, &ctx()).await?;
            (
                ids(&page, |record| record.id).into_iter().map(RecordId::get).collect(),
                page.next,
            )
        }
        Table::Tasks => {
            let query = TaskQuery {
                order,
                ..TaskQuery::default()
            };
            let page = storage.scan_tasks(query, 2, cursor, &ctx()).await?;
            (
                ids(&page, |record| record.id).into_iter().map(RecordId::get).collect(),
                page.next,
            )
        }
        Table::Submissions => {
            let query = SubmissionQuery {
                order,
                ..SubmissionQuery::default()
            };
            let page = storage.scan_submissions(query, 2, cursor, &ctx()).await?;
            (
                ids(&page, |record| record.id).into_iter().map(RecordId::get).collect(),
                page.next,
            )
        }
    })
}

async fn scan_all(storage: &dyn Storage, table: Table, order: Option<ScanOrder>) -> Vec<u64> {
    let (mut found, mut next) = scan_ids(storage, table, order, None).await.unwrap();
    while let Some(cursor) = next {
        // A cursor carries its order; the query may omit it.
        let cursor: Cursor = serde_json::from_str(&serde_json::to_string(&cursor).unwrap()).unwrap();
        let (page, following) = scan_ids(storage, table, None, Some(cursor)).await.unwrap();
        found.extend(page);
        next = following;
    }
    found
}

pub(crate) async fn scan_orders_and_cursors(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let mut conversations = vec![root.get()];
    let mut tasks = Vec::new();
    let mut submissions = Vec::new();
    for _ in 0..3 {
        let conversation: ConversationId = mint(storage).await;
        let task: TaskId = mint(storage).await;
        let submission: SubmissionId = mint(storage).await;
        commit(
            storage,
            vec![
                StorageWrite::Conversation(ConversationRecord::new(conversation)),
                StorageWrite::Task(pending_task(task, root)),
                StorageWrite::Submission(input(submission, root, None)),
            ],
        )
        .await;
        conversations.push(conversation.get());
        tasks.push(task.get());
        submissions.push(submission.get());
    }
    for (table, ids) in [
        (Table::Conversations, conversations),
        (Table::Tasks, tasks),
        (Table::Submissions, submissions),
    ] {
        let reversed: Vec<u64> = ids.iter().rev().copied().collect();
        assert_eq!(scan_all(storage, table, None).await, ids);
        assert_eq!(scan_all(storage, table, Some(ScanOrder::Ascending)).await, ids);
        assert_eq!(scan_all(storage, table, Some(ScanOrder::Descending)).await, reversed);
        let (_, next) = scan_ids(storage, table, Some(ScanOrder::Descending), None)
            .await
            .unwrap();
        let (page, _) = scan_ids(storage, table, Some(ScanOrder::Descending), next.clone())
            .await
            .unwrap();
        assert_eq!(page, reversed[2..reversed.len().min(4)]);
        let error = scan_ids(storage, table, Some(ScanOrder::Ascending), next)
            .await
            .unwrap_err();
        assert!(error.is_request() && error.to_string().contains("cursor"), "{error}");
    }
}

pub(crate) async fn task_replacement_and_filters(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let first_id: TaskId = mint(storage).await;
    let second_id: TaskId = mint(storage).await;
    let third_id: TaskId = mint(storage).await;
    let mut first = pending_task(first_id, root);
    first.memos = Some([("winner".to_owned(), Json::from("first"))].into_iter().collect());
    let mut second = pending_task(second_id, root);
    second.background = true;
    let mut third = pending_task(third_id, root);
    third.abort_requested = true;
    commit(
        storage,
        [&first, &second, &third]
            .map(|task| StorageWrite::Task(task.clone()))
            .to_vec(),
    )
    .await;

    let mut running = first.clone();
    running.state = TaskState::Running {
        checkpoint: value(json!({ "phase": "effect", "attempt": 1 })),
    };
    running.abort_requested = true;
    commit(storage, vec![StorageWrite::Task(running.clone())]).await;
    assert_eq!(storage.task(first_id, &ctx()).await.unwrap(), Some(running));
    let mut terminal = pending_task(first_id, root);
    terminal.abort_requested = true;
    terminal.state = TaskState::Terminal {
        outcome: TaskOutcome::Completed {
            result: value(json!({ "entry_id": 99 })),
        },
    };
    commit(storage, vec![StorageWrite::Task(terminal.clone())]).await;
    assert_eq!(storage.task(first_id, &ctx()).await.unwrap(), Some(terminal.clone()));

    let pending = TaskQuery {
        status: Some(TaskStatus::Pending),
        ..TaskQuery::default()
    };
    let page = storage.scan_tasks(pending.clone(), 1, None, &ctx()).await.unwrap();
    assert_eq!(ids(&page, |task| task.id), [second_id]);
    let page = storage.scan_tasks(pending, 1, page.next, &ctx()).await.unwrap();
    assert_eq!(ids(&page, |task| task.id), [third_id]);
    let marked = TaskQuery {
        status: Some(TaskStatus::Terminal),
        abort_requested: Some(true),
        ..TaskQuery::default()
    };
    assert_eq!(
        storage.scan_tasks(marked, 10, None, &ctx()).await.unwrap().items,
        [terminal]
    );
    let background = TaskQuery {
        background: Some(true),
        ..TaskQuery::default()
    };
    let page = storage.scan_tasks(background, 10, None, &ctx()).await.unwrap();
    assert_eq!(ids(&page, |task| task.id), [second_id]);
    let by_kind = TaskQuery {
        conversation_id: Some(root),
        kind: Some("test.task".into()),
        ..TaskQuery::default()
    };
    assert_eq!(
        storage.scan_tasks(by_kind, 10, None, &ctx()).await.unwrap().items.len(),
        3
    );
}

pub(crate) async fn task_owner_and_status_scans(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let owner_id: TaskId = mint(storage).await;
    let waiting_id: TaskId = mint(storage).await;
    let completing_id: TaskId = mint(storage).await;
    let owner = pending_task(owner_id, root);
    let mut waiting = pending_task(waiting_id, root);
    waiting.owner = Some(owner_id);
    waiting.state = TaskState::Waiting {
        checkpoint: value(json!({ "phase": "next" })),
        on: vec![owner_id],
        policy: JoinPolicy::AllSettled,
    };
    waiting.memos = Some([("kept".to_owned(), Json::Bool(true))].into_iter().collect());
    let mut completing = pending_task(completing_id, root);
    completing.owner = Some(owner_id);
    let outcome = TaskOutcome::Failed {
        error: TaskOutcomeError::new("held"),
        result: None,
    };
    completing.state = TaskState::Completing {
        outcome: outcome.clone(),
    };
    commit(
        storage,
        [&owner, &waiting, &completing]
            .map(|task| StorageWrite::Task(task.clone()))
            .to_vec(),
    )
    .await;
    assert_eq!(storage.task(waiting_id, &ctx()).await.unwrap(), Some(waiting.clone()));
    assert_eq!(
        storage.task(completing_id, &ctx()).await.unwrap(),
        Some(completing.clone())
    );
    assert_eq!(scan_status(storage, TaskStatus::Waiting).await, [waiting]);
    assert_eq!(scan_status(storage, TaskStatus::Completing).await, [completing.clone()]);
    let pending: Vec<TaskId> = scan_status(storage, TaskStatus::Pending)
        .await
        .iter()
        .map(|task| task.id)
        .collect();
    assert_eq!(pending, [owner_id]);
    let mut terminal = completing;
    terminal.state = TaskState::Terminal { outcome };
    commit(storage, vec![StorageWrite::Task(terminal.clone())]).await;
    assert_eq!(scan_status(storage, TaskStatus::Completing).await, []);
    assert_eq!(scan_status(storage, TaskStatus::Terminal).await, [terminal]);
}

pub(crate) async fn sleeping_tasks(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let id: TaskId = mint(storage).await;
    let mut sleeping = pending_task(id, root);
    sleeping.state = TaskState::Sleeping {
        checkpoint: value(json!({ "phase": "next" })),
        until: 61_000,
    };
    sleeping.memos = Some([("kept".to_owned(), Json::Bool(true))].into_iter().collect());
    commit(storage, vec![StorageWrite::Task(sleeping.clone())]).await;
    assert_eq!(storage.task(id, &ctx()).await.unwrap(), Some(sleeping.clone()));
    assert_eq!(scan_status(storage, TaskStatus::Sleeping).await, [sleeping.clone()]);
    let mut running = sleeping;
    running.state = TaskState::Running {
        checkpoint: value(json!({ "phase": "next" })),
    };
    commit(storage, vec![StorageWrite::Task(running.clone())]).await;
    assert_eq!(scan_status(storage, TaskStatus::Sleeping).await, []);
    assert_eq!(scan_status(storage, TaskStatus::Running).await, [running]);
}

async fn submission_ids(storage: &dyn Storage, query: SubmissionQuery) -> Vec<SubmissionId> {
    let mut found = Vec::new();
    let mut cursor = None;
    loop {
        let page = storage
            .scan_submissions(query.clone(), 1, cursor, &ctx())
            .await
            .unwrap();
        found.extend(ids(&page, |record| record.id));
        cursor = page.next;
        if cursor.is_none() {
            return found;
        }
    }
}

pub(crate) async fn request_ids_and_submission_replacement(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let second_conversation: ConversationId = mint(storage).await;
    commit(
        storage,
        vec![StorageWrite::Conversation(ConversationRecord::new(second_conversation))],
    )
    .await;
    let first_id: SubmissionId = mint(storage).await;
    let second_id: SubmissionId = mint(storage).await;
    let other_id: SubmissionId = mint(storage).await;
    let first = input(first_id, root, Some("same"));
    let second = input(second_id, root, Some("other"));
    let other = input(other_id, second_conversation, Some("same"));
    commit(
        storage,
        [&first, &second, &other]
            .map(|record| StorageWrite::Submission(record.clone()))
            .to_vec(),
    )
    .await;
    let by_request = |conversation, request: &'static str| async move {
        storage
            .submission_by_request(conversation, request, &ctx())
            .await
            .unwrap()
    };
    assert_eq!(by_request(root, "same").await, Some(first.clone()));
    assert_eq!(by_request(second_conversation, "same").await, Some(other));

    let mut placed = second;
    placed.state = SubmissionState::Placed {
        entry: mint(storage).await,
    };
    commit(storage, vec![StorageWrite::Submission(placed.clone())]).await;
    assert_eq!(
        storage.submission(second_id, &ctx()).await.unwrap(),
        Some(placed.clone())
    );
    assert_eq!(by_request(root, "other").await, Some(placed.clone()));

    let query = |conversation_id, status| SubmissionQuery {
        conversation_id,
        status,
        order: None,
    };
    assert_eq!(
        submission_ids(storage, query(None, None)).await,
        [first_id, second_id, other_id]
    );
    assert_eq!(
        submission_ids(storage, query(Some(root), None)).await,
        [first_id, second_id]
    );
    // A status change moves the record between status scans.
    assert_eq!(
        submission_ids(storage, query(None, Some(SubmissionStatus::Queued))).await,
        [first_id, other_id]
    );
    assert_eq!(
        submission_ids(storage, query(None, Some(SubmissionStatus::Placed))).await,
        [second_id]
    );
    assert_eq!(
        submission_ids(
            storage,
            query(Some(second_conversation), Some(SubmissionStatus::Queued))
        )
        .await,
        [other_id]
    );
    assert_eq!(
        submission_ids(
            storage,
            query(Some(second_conversation), Some(SubmissionStatus::Placed))
        )
        .await,
        []
    );
    let page = storage
        .scan_submissions(query(None, Some(SubmissionStatus::Placed)), 10, None, &ctx())
        .await
        .unwrap();
    assert_eq!(page.items, [placed]);
}

pub(crate) async fn write_submissions(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let done_id: SubmissionId = mint(storage).await;
    let failed_id: SubmissionId = mint(storage).await;
    let write = |id, request: &str| SubmissionRecord {
        kind: SubmissionType::Write,
        ..input(id, root, Some(request))
    };
    commit(
        storage,
        vec![
            StorageWrite::Submission(write(done_id, "passive-done")),
            StorageWrite::Submission(write(failed_id, "passive-failed")),
        ],
    )
    .await;
    let mut done = write(done_id, "passive-done");
    done.state = SubmissionState::Done {
        entry: mint(storage).await,
        answer: None,
    };
    let mut unanswered = write(failed_id, "passive-failed");
    unanswered.state = SubmissionState::Unanswered {
        entry: None,
        reason: "closed".into(),
        detail: Some(value(json!({ "retryable": false }))),
    };
    commit(
        storage,
        vec![
            StorageWrite::Submission(done.clone()),
            StorageWrite::Submission(unanswered.clone()),
        ],
    )
    .await;
    assert_eq!(storage.submission(done_id, &ctx()).await.unwrap(), Some(done.clone()));
    assert_eq!(
        storage
            .submission_by_request(root, "passive-done", &ctx())
            .await
            .unwrap(),
        Some(done)
    );
    assert_eq!(
        storage.submission(failed_id, &ctx()).await.unwrap(),
        Some(unanswered.clone())
    );
    assert_eq!(
        storage
            .submission_by_request(root, "passive-failed", &ctx())
            .await
            .unwrap(),
        Some(unanswered)
    );
}

pub(crate) async fn duplicate_request_id_rejected(fx: Fixture) {
    let storage = &*fx.storage;
    let root = create_root(storage).await;
    let other: ConversationId = mint(storage).await;
    commit(
        storage,
        vec![StorageWrite::Conversation(ConversationRecord::new(other))],
    )
    .await;
    let first: SubmissionId = mint(storage).await;
    let second: SubmissionId = mint(storage).await;
    reject(
        storage,
        vec![
            StorageWrite::Submission(input(first, root, Some("same"))),
            StorageWrite::Submission(input(second, root, Some("same"))),
        ],
        "already belongs to submission",
    )
    .await;
    commit(
        storage,
        vec![StorageWrite::Submission(input(first, root, Some("same")))],
    )
    .await;
    reject(
        storage,
        vec![StorageWrite::Submission(input(second, root, Some("same")))],
        "already belongs to submission",
    )
    .await;
    // Replacing the owner and reusing the request id in another conversation are fine.
    let mut placed = input(first, root, Some("same"));
    placed.state = SubmissionState::Placed {
        entry: mint(storage).await,
    };
    commit(
        storage,
        vec![
            StorageWrite::Submission(placed.clone()),
            StorageWrite::Submission(input(second, other, Some("same"))),
        ],
    )
    .await;
    assert_eq!(
        storage.submission_by_request(root, "same", &ctx()).await.unwrap(),
        Some(placed)
    );
    assert_eq!(
        storage
            .submission_by_request(other, "same", &ctx())
            .await
            .unwrap()
            .map(|record| record.id),
        Some(second)
    );
}
