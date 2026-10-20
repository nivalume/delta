//! SQLite-only behaviour: reopen, prefixes, schema and metadata checks, corruption,
//! reclamation and index use. Ported in intent from upstream `test/sqlite-storage.test.ts`.

#![allow(clippy::unwrap_used)] // Helpers outside `#[test]` functions.

mod support;

use std::path::{Path, PathBuf};

use delta_chord::json::Json;
use delta_chord::sqlite::rusqlite::{Connection, params};
use delta_chord::sqlite::{Database, OpenOptions, SqliteError};
use delta_harness::records::{
    BaseContent, ConversationPolicy, ConversationRecord, DocumentContent, DocumentCreate, DocumentId, DocumentPoint,
    DocumentScope, EntryId, Fork, History, ROOT_CONVERSATION_ID, RawId, Seq, StorageWrite, StoredEntry, TaskId,
};
use delta_harness::storage::{CURRENT_SCHEMA_VERSION, SqliteOptions, SqliteStorage, Storage, StorageError};
use pretty_assertions::assert_eq;
use serde_json::json;

use support::{TempDir, commit, create_root, ctx, entry, mint, pending_task, reject, value};

fn path_in(dir: &TempDir) -> PathBuf {
    dir.path().join("storage.sqlite")
}

async fn open(path: &Path) -> SqliteStorage {
    SqliteStorage::open(path, SqliteOptions::default()).await.unwrap()
}

async fn open_prefixed(path: &Path, prefix: &str) -> SqliteStorage {
    let options = SqliteOptions {
        prefix: Some(prefix.to_owned()),
        ..SqliteOptions::default()
    };
    SqliteStorage::open(path, options).await.unwrap()
}

fn scalar(path: &Path, sql: &str) -> i64 {
    Connection::open(path)
        .unwrap()
        .query_row(sql, [], |row| row.get(0))
        .unwrap()
}

fn execute(path: &Path, sql: &str, id: u64) {
    let connection = Connection::open(path).unwrap();
    connection.execute(sql, params![i64::try_from(id).unwrap()]).unwrap();
}

fn session_document(id: DocumentId, kind: &str, value: Json) -> StorageWrite {
    StorageWrite::DocumentCreate {
        record: DocumentCreate {
            id,
            kind: kind.to_owned(),
            key: None,
            scope: DocumentScope::Session,
            policy: None,
        },
        content: BaseContent { version: 1, value },
    }
}

fn delta(id: DocumentId, ops: serde_json::Value) -> StorageWrite {
    StorageWrite::DocumentChange {
        id,
        content: DocumentContent::Delta {
            version: 1,
            ops: serde_json::from_value(ops).unwrap(),
        },
    }
}

fn revisions(path: &Path, id: DocumentId) -> i64 {
    let connection = Connection::open(path).unwrap();
    connection
        .query_row(
            "SELECT count(*) FROM document_revisions WHERE document_id = ?",
            params![i64::try_from(id.0).unwrap()],
            |row| row.get(0),
        )
        .unwrap()
}

#[tokio::test]
async fn reopen_continues_ids_and_seqs() {
    let dir = TempDir::new();
    let path = path_in(&dir);
    let storage = open(&path).await;
    assert_eq!(
        storage
            .commit(
                vec![StorageWrite::Conversation(ConversationRecord::new(
                    ROOT_CONVERSATION_ID
                ))],
                &ctx()
            )
            .await
            .unwrap(),
        Seq(1)
    );
    let entry_id: EntryId = mint(&storage).await;
    assert_eq!(
        commit(
            &storage,
            vec![StorageWrite::Entry(entry(entry_id, ROOT_CONVERSATION_ID))]
        )
        .await,
        Seq(2)
    );
    storage.close(&ctx()).await.unwrap();

    let reopened = open(&path).await;
    assert_eq!(
        reopened.entry(entry_id, &ctx()).await.unwrap(),
        Some(StoredEntry {
            entry: entry(entry_id, ROOT_CONVERSATION_ID),
            commit_seq: Seq(2)
        })
    );
    assert_eq!(reopened.mint_id().await.unwrap(), RawId(entry_id.0 + 1));
    let task: TaskId = mint(&reopened).await;
    assert_eq!(
        commit(
            &reopened,
            vec![StorageWrite::Task(pending_task(task, ROOT_CONVERSATION_ID))]
        )
        .await,
        Seq(3)
    );
    reopened.close(&ctx()).await.unwrap();
}

#[tokio::test]
async fn prefixed_schemas_are_independent() {
    let dir = TempDir::new();
    let path = path_in(&dir);
    Connection::open(&path)
        .unwrap()
        .execute_batch("CREATE TABLE tasks (text TEXT)")
        .unwrap();

    let first = open_prefixed(&path, "first_").await;
    let second = open_prefixed(&path, "second_").await;
    create_root(&first).await;
    let entry_id: EntryId = mint(&first).await;
    commit(&first, vec![StorageWrite::Entry(entry(entry_id, ROOT_CONVERSATION_ID))]).await;
    assert_eq!(second.conversation(ROOT_CONVERSATION_ID, &ctx()).await.unwrap(), None);
    assert_eq!(second.mint_id().await.unwrap(), RawId(2));
    create_root(&second).await;
    assert_eq!(second.entry(entry_id, &ctx()).await.unwrap(), None);
    first.close(&ctx()).await.unwrap();
    second.close(&ctx()).await.unwrap();

    let reopened = open_prefixed(&path, "first_").await;
    assert!(reopened.entry(entry_id, &ctx()).await.unwrap().is_some());
    reopened.close(&ctx()).await.unwrap();

    let connection = Connection::open(&path).unwrap();
    let mut statement = connection
        .prepare("SELECT name FROM sqlite_master WHERE type IN ('table', 'index') AND name NOT LIKE 'sqlite_%'")
        .unwrap();
    let names: Vec<String> = statement
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    let foreign: Vec<&String> = names
        .iter()
        .filter(|name| !name.starts_with("first_") && !name.starts_with("second_"))
        .collect();
    assert_eq!(foreign, ["tasks"]);
    assert!(names.contains(&"first_tasks".to_owned()));
    assert!(names.contains(&"second_document_bases".to_owned()));
    assert!(names.contains(&"first_schema".to_owned()));
}

#[tokio::test]
async fn invalid_prefix_fails_before_creating_the_database() {
    let dir = TempDir::new();
    let nested = dir.path().join("nested");
    let options = SqliteOptions {
        prefix: Some("bad-prefix".into()),
        ..SqliteOptions::default()
    };
    let error = SqliteStorage::open(nested.join("storage.sqlite"), options)
        .await
        .unwrap_err();
    assert!(
        matches!(error, StorageError::Sqlite(SqliteError::InvalidSchemaPrefix(_))),
        "{error:?}"
    );
    assert!(!nested.exists());

    let db = Database::open_in_memory(OpenOptions::default()).await.unwrap();
    let error = SqliteStorage::open_database(db, Some("sqlite_x".into()))
        .await
        .unwrap_err();
    assert!(
        matches!(error, StorageError::Sqlite(SqliteError::ReservedSchemaPrefix(_))),
        "{error:?}"
    );
}

#[tokio::test]
async fn open_database_uses_an_open_connection() {
    let db = Database::open_in_memory(OpenOptions::default()).await.unwrap();
    let storage = SqliteStorage::open_database(db, Some("app_".into())).await.unwrap();
    create_root(&storage).await;
    assert!(
        storage
            .conversation(ROOT_CONVERSATION_ID, &ctx())
            .await
            .unwrap()
            .is_some()
    );
    storage.close(&ctx()).await.unwrap();
}

#[tokio::test]
async fn missing_metadata_and_newer_schemas_are_refused() {
    let dir = TempDir::new();
    let path = path_in(&dir);
    open(&path).await.close(&ctx()).await.unwrap();
    assert_eq!(
        scalar(&path, "SELECT version FROM schema"),
        i64::from(CURRENT_SCHEMA_VERSION)
    );
    Connection::open(&path)
        .unwrap()
        .execute_batch("DELETE FROM metadata")
        .unwrap();
    let error = SqliteStorage::open(&path, SqliteOptions::default()).await.unwrap_err();
    assert!(
        matches!(&error, StorageError::Corrupt(message) if message.contains("metadata is missing")),
        "{error:?}"
    );

    Connection::open(&path)
        .unwrap()
        .execute_batch("UPDATE schema SET version = 2")
        .unwrap();
    let error = SqliteStorage::open(&path, SqliteOptions::default()).await.unwrap_err();
    assert!(
        matches!(
            error,
            StorageError::Sqlite(SqliteError::NewerSchema {
                version: 2,
                supported: 1
            })
        ),
        "{error:?}"
    );
}

#[tokio::test]
async fn a_missing_base_is_corruption() {
    let dir = TempDir::new();
    let path = path_in(&dir);
    let storage = open(&path).await;
    create_root(&storage).await;
    let id: DocumentId = mint(&storage).await;
    commit(
        &storage,
        vec![session_document(id, "corrupt", value(json!({ "retained": true })))],
    )
    .await;
    execute(&path, "DELETE FROM document_revisions WHERE document_id = ?", id.0);
    let error = storage.document(id, DocumentPoint::Current, &ctx()).await.unwrap_err();
    assert_eq!(error.to_string(), format!("Document {id} is missing a required base"));
    assert!(matches!(error, StorageError::Corrupt(_)));
}

#[tokio::test]
async fn replays_root_replacements_and_refuses_corrupt_ops() {
    let dir = TempDir::new();
    let path = path_in(&dir);
    let storage = open(&path).await;
    create_root(&storage).await;
    let id: DocumentId = mint(&storage).await;
    commit(
        &storage,
        vec![session_document(
            id,
            "replay",
            value(json!({ "nested": { "value": 1 }, "rows": [] })),
        )],
    )
    .await;
    commit(
        &storage,
        vec![delta(
            id,
            json!([["r", { "nested": { "value": 2 }, "rows": [{ "id": 1 }] }]]),
        )],
    )
    .await;
    commit(
        &storage,
        vec![delta(
            id,
            json!([["s", ["nested", "value"], 3], ["p", ["rows"], 1, 0, [{ "id": 2 }]], ["m", ["rows"], [1, 0]]]),
        )],
    )
    .await;
    commit(&storage, vec![delta(id, json!([["s", ["nested", "value"], 4]]))]).await;
    let expected = value(json!({ "nested": { "value": 4 }, "rows": [{ "id": 2 }, { "id": 1 }] }));
    let read = storage
        .document(id, DocumentPoint::Current, &ctx())
        .await
        .unwrap()
        .unwrap();
    assert_eq!((read.value, read.deltas_since_base), (expected, 3));

    execute(
        &path,
        "UPDATE document_revisions SET content = '[[\"unknown\"]]' WHERE document_id = ?1 AND seq = \
         (SELECT max(seq) FROM document_revisions WHERE document_id = ?1)",
        id.0,
    );
    let error = storage.document(id, DocumentPoint::Current, &ctx()).await.unwrap_err();
    assert!(
        matches!(&error, StorageError::Corrupt(message) if message.contains("unreadable")),
        "{error:?}"
    );

    execute(
        &path,
        "UPDATE document_revisions SET content = '[[\"s\", [\"missing\", \"deep\"], 1]]' WHERE document_id = ?1 \
         AND seq = (SELECT max(seq) FROM document_revisions WHERE document_id = ?1)",
        id.0,
    );
    let error = storage.document(id, DocumentPoint::Current, &ctx()).await.unwrap_err();
    assert!(
        matches!(&error, StorageError::Corrupt(message) if message.contains("does not apply")),
        "{error:?}"
    );
}

#[tokio::test]
async fn refused_commits_leave_rows_and_sequences_untouched() {
    let dir = TempDir::new();
    let path = path_in(&dir);
    let storage = open(&path).await;
    create_root(&storage).await;
    let transient: EntryId = mint(&storage).await;
    reject(
        &storage,
        vec![
            StorageWrite::Entry(entry(transient, ROOT_CONVERSATION_ID)),
            StorageWrite::Task(pending_task(TaskId(transient.0), ROOT_CONVERSATION_ID)),
        ],
        "is written more than once",
    )
    .await;
    // An id past SQLite's range is refused by the checked conversion, inside the transaction.
    let error = storage
        .commit(
            vec![
                StorageWrite::Entry(entry(mint(&storage).await, ROOT_CONVERSATION_ID)),
                StorageWrite::Entry(entry(EntryId(u64::MAX), ROOT_CONVERSATION_ID)),
            ],
            &ctx(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, StorageError::Rejected(_)), "{error:?}");
    assert_eq!(storage.entry(transient, &ctx()).await.unwrap(), None);
    let committed: EntryId = mint(&storage).await;
    assert_eq!(
        commit(
            &storage,
            vec![StorageWrite::Entry(entry(committed, ROOT_CONVERSATION_ID))]
        )
        .await,
        Seq(2)
    );
    assert_eq!(scalar(&path, "SELECT count(*) FROM entries"), 1);
    assert_eq!(scalar(&path, "SELECT next_seq FROM metadata WHERE singleton = 1"), 3);
}

#[tokio::test]
async fn rewindable_points_survive_reopen() {
    let dir = TempDir::new();
    let path = path_in(&dir);
    let storage = open(&path).await;
    create_root(&storage).await;
    let id: DocumentId = mint(&storage).await;
    let created = StorageWrite::DocumentCreate {
        record: DocumentCreate {
            id,
            kind: "history".into(),
            key: None,
            scope: DocumentScope::Conversation {
                conversation_id: ROOT_CONVERSATION_ID,
            },
            policy: Some(ConversationPolicy {
                history: History::Rewindable,
                fork: Fork::AsOf,
            }),
        },
        content: BaseContent {
            version: 1,
            value: value(json!({ "count": 0 })),
        },
    };
    let mut ancient = commit(&storage, vec![created]).await;
    let mut recent = ancient;
    for count in 1..=40 {
        let write = if count == 20 {
            StorageWrite::DocumentChange {
                id,
                content: DocumentContent::Base {
                    version: 1,
                    value: value(json!({ "count": count })),
                },
            }
        } else {
            delta(id, json!([["s", ["count"], count]]))
        };
        recent = commit(&storage, vec![write]).await;
        if count == 5 {
            ancient = recent;
        }
    }
    storage.close(&ctx()).await.unwrap();
    let reopened = &open(&path).await;
    let at = |seq| async move {
        reopened
            .document(id, DocumentPoint::At(seq), &ctx())
            .await
            .unwrap()
            .unwrap()
    };
    let old = at(ancient).await;
    assert_eq!((old.value, old.deltas_since_base), (value(json!({ "count": 5 })), 5));
    let new = at(recent).await;
    assert_eq!((new.value, new.deltas_since_base), (value(json!({ "count": 40 })), 20));
    // Rewindable incarnations keep every revision.
    assert_eq!(revisions(&path, id), 41);
}

#[tokio::test]
async fn current_only_revisions_are_reclaimed_by_bases_and_retirement() {
    let dir = TempDir::new();
    let path = path_in(&dir);
    let storage = open(&path).await;
    create_root(&storage).await;
    let id: DocumentId = mint(&storage).await;
    commit(
        &storage,
        vec![session_document(id, "latest", value(json!({ "count": 0 })))],
    )
    .await;
    for count in 1..=10 {
        commit(&storage, vec![delta(id, json!([["s", ["count"], count]]))]).await;
    }
    assert_eq!(revisions(&path, id), 11);
    let newest: String = Connection::open(&path)
        .unwrap()
        .query_row(
            "SELECT content FROM document_revisions WHERE document_id = ? AND kind = 'delta' ORDER BY seq DESC LIMIT 1",
            params![i64::try_from(id.0).unwrap()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&newest).unwrap(),
        json!([["s", ["count"], 10]])
    );
    commit(
        &storage,
        vec![StorageWrite::DocumentChange {
            id,
            content: DocumentContent::Base {
                version: 1,
                value: value(json!({ "count": 11 })),
            },
        }],
    )
    .await;
    assert_eq!(revisions(&path, id), 1);
    commit(&storage, vec![delta(id, json!([["r", { "count": 12 }]]))]).await;
    assert_eq!(revisions(&path, id), 2);
    commit(&storage, vec![StorageWrite::DocumentRetire { id }]).await;
    assert_eq!(revisions(&path, id), 0);
}

#[tokio::test]
async fn queries_use_their_indexes() {
    let dir = TempDir::new();
    let path = path_in(&dir);
    open(&path).await.close(&ctx()).await.unwrap();
    let connection = Connection::open(&path).unwrap();
    let plan = |sql: &str| -> String {
        let mut statement = connection.prepare(&format!("EXPLAIN QUERY PLAN {sql}")).unwrap();
        let details: Vec<String> = statement
            .query_map([], |row| row.get::<_, String>(3))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        details.join("\n")
    };
    let cases = [
        (
            "SELECT record FROM documents WHERE kind = 'k' AND scope_kind = 'session' AND owner_id = 0 \
             AND key IS NULL AND retired_at IS NULL ORDER BY created_at DESC LIMIT 1",
            "documents_by_address",
        ),
        (
            "SELECT record FROM documents WHERE kind = 'k' AND scope_kind = 'conversation' AND owner_id = 1 \
             AND key IS 'x' AND created_at <= 10 AND (retired_at IS NULL OR retired_at > 10) \
             ORDER BY created_at DESC LIMIT 1",
            "documents_by_address",
        ),
        (
            "SELECT record FROM documents WHERE scope_kind = 'task' AND owner_id = 1 AND kind = 'k' AND id > 0 \
             AND retired_at IS NULL ORDER BY id ASC LIMIT 10",
            "documents_by_scope_kind",
        ),
        (
            "SELECT record FROM entries WHERE conversation_id = 1 AND id >= 0 AND id <= 10 ORDER BY id DESC LIMIT 10",
            "entries_by_conversation",
        ),
        (
            "SELECT record FROM entries WHERE conversation_id = 1 AND head IS NOT NULL AND id <= 10 \
             ORDER BY id DESC LIMIT 1",
            "entry_heads_by_conversation",
        ),
        (
            "SELECT record FROM tasks WHERE status = 'pending' AND id > 0 ORDER BY id ASC LIMIT 10",
            "tasks_by_status",
        ),
        (
            "SELECT record FROM submissions WHERE conversation_id = 1 AND request_id = 'r'",
            "submissions_by_request",
        ),
        (
            "SELECT seq, version, content FROM document_revisions WHERE document_id = 1 AND kind = 'base' \
             AND seq <= 10 ORDER BY seq DESC LIMIT 1",
            // `document_bases` or the primary key walking back from `at`; never a scan.
            "SEARCH document_revisions",
        ),
        (
            "SELECT seq, kind, version, content FROM document_revisions WHERE document_id = 1 AND seq > 5 \
             AND seq <= 10 ORDER BY seq",
            "PRIMARY KEY",
        ),
    ];
    for (sql, index) in cases {
        let detail = plan(sql);
        assert!(detail.contains(index), "{sql}\n{detail}");
        assert!(
            !detail.contains("SCAN documents") && !detail.contains("SCAN document_revisions"),
            "{detail}"
        );
    }
}

#[tokio::test]
async fn close_waits_and_refuses_later_calls() {
    let dir = TempDir::new();
    let storage = open(&path_in(&dir)).await;
    create_root(&storage).await;
    let read = storage.conversation(ROOT_CONVERSATION_ID, &ctx());
    let (read, closed) = tokio::join!(read, storage.close(&ctx()));
    assert!(read.unwrap().is_some());
    closed.unwrap();
    let error = storage.commit(Vec::new(), &ctx()).await.unwrap_err();
    assert!(matches!(error, StorageError::Closed), "{error:?}");
    storage.close(&ctx()).await.unwrap();
}
