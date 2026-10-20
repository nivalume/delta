//! Tests of the `sqlite` module, ported from upstream `test/sqlite/migrations.test.ts` and
//! `test/sqlite/node.test.ts`.

#![cfg(feature = "sqlite")]
#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use pretty_assertions::assert_eq;
use rusqlite::OptionalExtension;

use delta_chord::sqlite::{Database, Migration, OpenOptions, SqliteError, apply_migrations, validate_schema_prefix};

/// A directory under the system temp directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let name = format!(
            "delta-chord-sqlite-{}-{nanos}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let path = std::env::temp_dir().join(name);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

async fn memory() -> Database {
    Database::open_in_memory(OpenOptions::default()).await.unwrap()
}

async fn exec(db: &Database, sql: &'static str) {
    db.call(move |connection| Ok(connection.execute_batch(sql)?))
        .await
        .unwrap();
}

async fn ints(db: &Database, sql: &'static str) -> Vec<i64> {
    db.call(move |connection| {
        let mut statement = connection.prepare(sql)?;
        let rows = statement.query_map([], |row| row.get(0))?;
        Ok(rows.collect::<Result<Vec<i64>, _>>()?)
    })
    .await
    .unwrap()
}

async fn schema_names(db: &Database) -> Vec<String> {
    db.call(|connection| {
        let mut statement = connection.prepare("SELECT name FROM sqlite_schema ORDER BY name")?;
        let rows = statement.query_map([], |row| row.get(0))?;
        Ok(rows.collect::<Result<Vec<String>, _>>()?)
    })
    .await
    .unwrap()
}

async fn pragma_int(db: &Database, pragma: &'static str) -> i64 {
    db.call(move |connection| Ok(connection.query_row(&format!("PRAGMA {pragma}"), [], |row| row.get(0))?))
        .await
        .unwrap()
}

const NOTES_V1: Migration = Migration {
    version: 1,
    statements: &["CREATE TABLE notes (body TEXT)", "INSERT INTO notes VALUES ('kept')"],
};
const NOTES_V2: Migration = Migration {
    version: 2,
    statements: &["ALTER TABLE notes ADD COLUMN title TEXT"],
};

#[tokio::test]
async fn migrations_track_independent_histories_and_skip_applied_ones() {
    let db = memory().await;
    apply_migrations(&db, &[NOTES_V1], "notes_schema").await.unwrap();
    apply_migrations(&db, &[NOTES_V1], "notes_schema").await.unwrap();
    apply_migrations(&db, &[], "other_schema").await.unwrap();
    apply_migrations(&db, &[NOTES_V1, NOTES_V2], "notes_schema")
        .await
        .unwrap();

    let notes = db
        .call(|connection| {
            let mut statement = connection.prepare("SELECT body, title FROM notes")?;
            let rows = statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
            Ok(rows.collect::<Result<Vec<(String, Option<String>)>, _>>()?)
        })
        .await
        .unwrap();
    assert_eq!(notes, vec![("kept".to_owned(), None)]);
    assert_eq!(ints(&db, "SELECT version FROM notes_schema").await, vec![2]);
    assert_eq!(ints(&db, "SELECT version FROM other_schema").await, vec![0]);

    let error = apply_migrations(&db, &[NOTES_V1], "notes_schema").await.unwrap_err();
    assert!(
        matches!(
            error,
            SqliteError::NewerSchema {
                version: 2,
                supported: 1
            }
        ),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "SQLite schema version 2 is newer than supported version 1"
    );
    db.close().await.unwrap();
}

#[tokio::test]
async fn migrations_reject_invalid_versions_before_touching_the_database() {
    let db = memory().await;
    for versions in [&[0][..], &[2], &[1, 3], &[1, 1]] {
        let migrations: Vec<Migration> = versions
            .iter()
            .map(|&version| Migration {
                version,
                statements: &[],
            })
            .collect();
        let error = apply_migrations(&db, &migrations, "app_schema").await.unwrap_err();
        assert!(
            matches!(error, SqliteError::NonContiguousMigrations),
            "{versions:?}: {error:?}"
        );
        assert!(error.to_string().contains("contiguous versions starting at 1"));
    }
    assert_eq!(schema_names(&db).await, Vec::<String>::new());
    db.close().await.unwrap();
}

#[tokio::test]
async fn schema_prefixes_are_validated_before_any_sql_runs() {
    let db = memory().await;
    for prefix in [
        "",
        "1bad",
        "bad-prefix",
        "bad; DROP TABLE notes",
        "sqlite_app",
        "SQLITE_app",
        "é",
    ] {
        assert!(validate_schema_prefix(prefix).is_err(), "{prefix:?}");
        assert!(apply_migrations(&db, &[], prefix).await.is_err(), "{prefix:?}");
    }
    for prefix in ["pi_", "_private", "app2", "sqlite"] {
        validate_schema_prefix(prefix).unwrap();
    }
    assert!(
        matches!(validate_schema_prefix("bad-prefix"), Err(SqliteError::InvalidSchemaPrefix(p)) if p == "bad-prefix")
    );
    assert!(
        matches!(validate_schema_prefix("SQLITE_app"), Err(SqliteError::ReservedSchemaPrefix(p)) if p == "SQLITE_app")
    );
    assert_eq!(schema_names(&db).await, Vec::<String>::new());
    db.close().await.unwrap();
}

#[tokio::test]
async fn a_failing_migration_rolls_back_the_bootstrap_and_every_pending_migration() {
    const FIRST: Migration = Migration {
        version: 1,
        statements: &[
            "CREATE TABLE migration_first (value TEXT) STRICT",
            "INSERT INTO migration_first (value) VALUES ('retained')",
        ],
    };
    let db = memory().await;
    let failed = [
        FIRST,
        Migration {
            version: 2,
            statements: &["CREATE TABLE migration_second (value TEXT) STRICT", "THIS IS NOT SQL"],
        },
    ];
    let error = apply_migrations(&db, &failed, "app_schema").await.unwrap_err();
    assert!(matches!(error, SqliteError::Sqlite(_)), "{error:?}");
    assert_eq!(schema_names(&db).await, Vec::<String>::new());

    let fixed = [
        FIRST,
        Migration {
            version: 2,
            statements: &["CREATE TABLE migration_second (value TEXT) STRICT"],
        },
    ];
    apply_migrations(&db, &fixed, "app_schema").await.unwrap();
    assert_eq!(
        ints(&db, "SELECT version FROM app_schema WHERE singleton = 1").await,
        vec![2]
    );
    let value: String = db
        .call(|connection| Ok(connection.query_row("SELECT value FROM migration_first", [], |row| row.get(0))?))
        .await
        .unwrap();
    assert_eq!(value, "retained");
    db.close().await.unwrap();
}

#[tokio::test]
async fn the_metadata_table_has_upstreams_shape() {
    let db = memory().await;
    apply_migrations(&db, &[], "app_schema").await.unwrap();
    let error = db
        .call(|connection| Ok(connection.execute("INSERT INTO app_schema (singleton, version) VALUES (2, 0)", [])?))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("CHECK constraint failed"), "{error}");
    let error = db
        .call(|connection| Ok(connection.execute("UPDATE app_schema SET version = 'one'", [])?))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("cannot store TEXT value in INTEGER column"),
        "{error}"
    );
    db.close().await.unwrap();
}

#[tokio::test]
async fn open_configures_connection_defaults_and_accepts_explicit_settings() {
    let explicit = OpenOptions {
        wal_autocheckpoint_pages: 0,
        busy_timeout: Duration::from_millis(123),
    };
    for (options, timeout, pages) in [(OpenOptions::default(), 5_000, 1_000), (explicit, 123, 0)] {
        let db = Database::open_in_memory(options).await.unwrap();
        assert_eq!(pragma_int(&db, "busy_timeout").await, timeout);
        assert_eq!(pragma_int(&db, "wal_autocheckpoint").await, pages);
        assert_eq!(pragma_int(&db, "synchronous").await, 1);
        db.close().await.unwrap();
    }
}

#[tokio::test]
async fn a_file_database_uses_wal_and_close_truncates_the_log() {
    let dir = TempDir::new();
    let path = dir.path().join("nested").join("store.db");
    let db = Database::open(&path, OpenOptions::default()).await.unwrap();
    let mode: String = db
        .call(|connection| Ok(connection.query_row("PRAGMA journal_mode", [], |row| row.get(0))?))
        .await
        .unwrap();
    assert_eq!(mode, "wal");
    apply_migrations(&db, &[NOTES_V1], "app_schema").await.unwrap();
    let wal = dir.path().join("nested").join("store.db-wal");
    assert!(std::fs::metadata(&wal).unwrap().len() > 0);

    // A second connection sees committed work through the WAL.
    let reader = Database::open(&path, OpenOptions::default()).await.unwrap();
    assert_eq!(ints(&reader, "SELECT version FROM app_schema").await, vec![1]);
    reader.close().await.unwrap();

    db.close().await.unwrap();
    assert_eq!(std::fs::metadata(&wal).map_or(0, |metadata| metadata.len()), 0);

    let db = Database::open(&path, OpenOptions::default()).await.unwrap();
    assert_eq!(ints(&db, "SELECT version FROM app_schema").await, vec![1]);
    apply_migrations(&db, &[NOTES_V1], "app_schema").await.unwrap();
    db.close().await.unwrap();
}

#[tokio::test]
async fn open_reports_a_path_it_cannot_create() {
    let dir = TempDir::new();
    let file = dir.path().join("file");
    std::fs::write(&file, b"not a directory").unwrap();
    let error = Database::open(file.join("store.db"), OpenOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(error, SqliteError::Io(_)), "{error:?}");
}

#[tokio::test]
async fn a_transaction_commits_its_work() {
    let db = memory().await;
    let value = db
        .transaction(|transaction| {
            transaction.execute_batch("CREATE TABLE probe (value INTEGER)")?;
            transaction.execute("INSERT INTO probe (value) VALUES (?1)", [1])?;
            Ok("done")
        })
        .await
        .unwrap();
    assert_eq!(value, "done");
    assert_eq!(ints(&db, "SELECT value FROM probe").await, vec![1]);
    db.close().await.unwrap();
}

#[tokio::test]
async fn a_transaction_rolls_back_and_returns_the_closure_error() {
    let db = memory().await;
    exec(&db, "CREATE TABLE probe (value INTEGER)").await;
    let error = db
        .transaction(|transaction| -> Result<(), _> {
            transaction.execute("INSERT INTO probe (value) VALUES (?1)", [1])?;
            Err(SqliteError::other("roll back"))
        })
        .await
        .unwrap_err();
    assert!(matches!(&error, SqliteError::Other(_)), "{error:?}");
    assert_eq!(error.to_string(), "roll back");
    assert_eq!(ints(&db, "SELECT value FROM probe").await, Vec::<i64>::new());
    // The connection is usable and not left inside a transaction.
    assert!(db.call(|connection| Ok(connection.is_autocommit())).await.unwrap());
    db.close().await.unwrap();
}

#[tokio::test]
async fn a_failed_rollback_is_reported_separately_from_the_closure_error() {
    let db = memory().await;
    exec(&db, "CREATE TABLE probe (value INTEGER)").await;
    let error = db
        .transaction(|transaction| -> Result<(), _> {
            transaction.execute_batch("INSERT INTO probe (value) VALUES (1)")?;
            transaction.execute_batch("COMMIT")?;
            Err(SqliteError::other("rejected after an escaped commit"))
        })
        .await
        .unwrap_err();
    match &error {
        SqliteError::Rollback { cause, rollback } => {
            assert_eq!(cause.to_string(), "rejected after an escaped commit");
            assert!(rollback.to_string().contains("no transaction is active"), "{rollback}");
        }
        other => panic!("expected a rollback error, got {other:?}"),
    }
    assert_eq!(ints(&db, "SELECT value FROM probe").await, vec![1]);
    db.close().await.unwrap();
}

#[tokio::test]
async fn a_panicking_closure_rolls_back_and_resumes_the_panic_in_the_caller() {
    let db = memory().await;
    exec(&db, "CREATE TABLE probe (value INTEGER)").await;
    let handle = db.clone();
    let panicked = tokio::spawn(async move {
        handle
            .transaction(|transaction| -> Result<(), SqliteError> {
                transaction.execute("INSERT INTO probe (value) VALUES (?1)", [1])?;
                panic!("boom");
            })
            .await
    })
    .await
    .unwrap_err();
    assert!(panicked.is_panic());
    assert_eq!(ints(&db, "SELECT value FROM probe").await, Vec::<i64>::new());
    assert!(db.call(|connection| Ok(connection.is_autocommit())).await.unwrap());
    db.close().await.unwrap();
}

#[tokio::test]
async fn closures_run_in_the_order_they_were_sent() {
    let db = memory().await;
    exec(&db, "CREATE TABLE call_order (value INTEGER)").await;
    // Work sent while a transaction runs neither sees its uncommitted rows nor joins its rollback.
    let mut transaction = Box::pin(db.transaction(|transaction| -> Result<(), _> {
        transaction.execute("INSERT INTO call_order (value) VALUES (?1)", [1])?;
        std::thread::sleep(Duration::from_millis(20));
        Err(SqliteError::other("roll back"))
    }));
    let mut before_write = Box::pin(ints(&db, "SELECT value FROM call_order ORDER BY value"));
    let mut write = Box::pin(exec(&db, "INSERT INTO call_order (value) VALUES (2)"));
    let mut after_write = Box::pin(ints(&db, "SELECT value FROM call_order ORDER BY value"));
    // The first poll sends each closure, in this order.
    let _ = futures::poll!(&mut transaction);
    let _ = futures::poll!(&mut before_write);
    let _ = futures::poll!(&mut write);
    let _ = futures::poll!(&mut after_write);

    let error = transaction.await.unwrap_err();
    assert_eq!(error.to_string(), "roll back");
    write.await;
    assert_eq!(before_write.await, Vec::<i64>::new());
    assert_eq!(after_write.await, vec![2]);
    db.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_transactions_are_serialized() {
    let db = memory().await;
    exec(&db, "CREATE TABLE transaction_queue (value INTEGER)").await;
    let mut tasks = Vec::new();
    for value in 0..8_i64 {
        let db = db.clone();
        tasks.push(tokio::spawn(async move {
            db.transaction(move |transaction| {
                let count: i64 =
                    transaction.query_row("SELECT count(*) FROM transaction_queue", [], |row| row.get(0))?;
                std::thread::sleep(Duration::from_millis(2));
                transaction.execute(
                    "INSERT INTO transaction_queue (value) VALUES (?1)",
                    [value * 100 + count],
                )?;
                Ok(())
            })
            .await
        }));
    }
    for task in tasks {
        task.await.unwrap().unwrap();
    }
    // Each transaction saw every row committed before it, so the counts are 0..8.
    let mut counts: Vec<i64> = ints(&db, "SELECT value FROM transaction_queue")
        .await
        .into_iter()
        .map(|value| value % 100)
        .collect();
    counts.sort_unstable();
    assert_eq!(counts, (0..8).collect::<Vec<_>>());
    db.close().await.unwrap();
}

#[tokio::test]
async fn close_is_idempotent_and_later_calls_fail_with_closed() {
    let db = memory().await;
    let other = db.clone();
    let third = db.clone();
    exec(&db, "CREATE TABLE probe (value INTEGER)").await;
    db.close().await.unwrap();
    other.close().await.unwrap();

    let error = third
        .call(|connection| Ok(connection.execute_batch("SELECT 1")?))
        .await
        .unwrap_err();
    assert!(matches!(error, SqliteError::Closed), "{error:?}");
    let error = third.transaction(|_| Ok(())).await.unwrap_err();
    assert!(matches!(error, SqliteError::Closed), "{error:?}");
    let error = apply_migrations(&third, &[], "app_schema").await.unwrap_err();
    assert!(matches!(error, SqliteError::Closed), "{error:?}");
}

#[tokio::test]
async fn work_sent_before_close_completes_and_work_after_it_fails() {
    let db = memory().await;
    let other = db.clone();
    let mut before = Box::pin(other.call(|connection| {
        std::thread::sleep(Duration::from_millis(20));
        Ok(connection.query_row("SELECT 41 + 1", [], |row| row.get::<_, i64>(0))?)
    }));
    let _ = futures::poll!(&mut before);
    let mut close = Box::pin(db.close());
    let _ = futures::poll!(&mut close);
    let after = other.call(|connection| {
        Ok(connection
            .query_row("SELECT 1", [], |row| row.get::<_, i64>(0))
            .optional()?)
    });

    assert_eq!(before.await.unwrap(), 42);
    close.await.unwrap();
    assert!(matches!(after.await, Err(SqliteError::Closed)));
}
