//! SQLite storage helpers: a [`Database`] whose connection lives on a dedicated thread,
//! transactions, atomic migrations and a check for table-name prefixes.
//!
//! Corresponds to upstream `src/sqlite`. See `docs/design/sqlite.md`.

use std::error::Error as StdError;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;
use std::{fmt, fs, thread};

use rusqlite::{Connection, Transaction, TransactionBehavior};
use tokio::sync::oneshot;

/// Errors of the `sqlite` module.
#[derive(Debug, thiserror::Error)]
pub enum SqliteError {
    /// An error reported by SQLite or `rusqlite`.
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
    /// Creating the database directory or the database thread failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// The database was closed (by [`Database::close`] on this or another handle).
    #[error("SQLite database is closed")]
    Closed,
    /// Migration versions are not `1, 2, 3, ...`.
    #[error("SQLite migrations must have contiguous versions starting at 1")]
    NonContiguousMigrations,
    /// The database was migrated by newer code than the migrations given.
    #[error("SQLite schema version {version} is newer than supported version {supported}")]
    NewerSchema { version: i64, supported: u32 },
    /// The prefix is not `^[A-Za-z_][A-Za-z0-9_]*$`.
    #[error("Invalid SQLite schema prefix {0:?}")]
    InvalidSchemaPrefix(String),
    /// The prefix starts with `sqlite_` (any case), which SQLite reserves.
    #[error("SQLite schema prefix {0:?} is reserved")]
    ReservedSchemaPrefix(String),
    /// A transaction failed and rolling it back failed too, so its changes may have been
    /// committed or may still be pending.
    #[error("SQLite transaction failed and rollback failed: {cause}; rollback: {rollback}")]
    Rollback {
        /// Why the transaction did not commit: the closure's error or the `COMMIT` error.
        #[source]
        cause: Box<SqliteError>,
        /// Why `ROLLBACK` failed.
        rollback: rusqlite::Error,
    },
    /// Any other error a closure passed to [`Database::call`] or [`Database::transaction`]
    /// returns; build it with [`SqliteError::other`].
    #[error(transparent)]
    Other(Box<dyn StdError + Send + Sync>),
}

impl SqliteError {
    /// Wraps an error that is not a SQLite error, for closures run on the database.
    pub fn other(error: impl Into<Box<dyn StdError + Send + Sync>>) -> Self {
        Self::Other(error.into())
    }
}

/// Connection settings for [`Database::open`] and [`Database::open_in_memory`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenOptions {
    /// `PRAGMA wal_autocheckpoint`, in pages; 0 disables automatic checkpoints. Default 1000.
    pub wal_autocheckpoint_pages: u32,
    /// How long SQLite waits for a competing lock before failing with `SQLITE_BUSY`.
    /// Default 5 s; values above `i32::MAX` milliseconds are capped.
    pub busy_timeout: Duration,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            wal_autocheckpoint_pages: 1_000,
            busy_timeout: Duration::from_secs(5),
        }
    }
}

type Job = Box<dyn FnOnce(&mut Connection) + Send>;
type Outcome<T> = thread::Result<Result<T, SqliteError>>;

enum Message {
    Run(Job),
    Close(oneshot::Sender<Result<(), SqliteError>>),
}

/// A SQLite connection owned by a dedicated thread.
///
/// [`call`](Self::call) and [`transaction`](Self::transaction) send a synchronous closure to
/// that thread and await its result. Closures run one at a time, in the order they were sent
/// (the order in which the returned futures are first polled), so a transaction never
/// interleaves with other work. Dropping a returned future does not cancel a closure that was
/// already sent; it still runs and its result is discarded. A closure that panics resumes the
/// panic in the awaiting task; the database stays usable (an open transaction rolls back).
///
/// Handles are cheap to clone and share the connection. The thread exits when
/// [`close`](Self::close) runs on any handle or when the last handle is dropped; later calls
/// fail with [`SqliteError::Closed`].
#[derive(Clone)]
pub struct Database {
    sender: mpsc::Sender<Message>,
}

impl fmt::Debug for Database {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Database").finish_non_exhaustive()
    }
}

impl Database {
    /// Opens (creating if needed) the database file at `path`, creating its parent directory,
    /// and configures the connection: `journal_mode = WAL`, `synchronous = NORMAL`,
    /// `wal_autocheckpoint` and the busy timeout from `options`.
    ///
    /// The path `:memory:` opens a private in-memory database, as in SQLite.
    pub async fn open(path: impl AsRef<Path>, options: OpenOptions) -> Result<Self, SqliteError> {
        let path = path.as_ref().to_path_buf();
        Self::start(
            move || {
                if path != Path::new(":memory:") {
                    create_parent_dir(&path)?;
                }
                Ok(Connection::open(&path)?)
            },
            options,
        )
        .await
    }

    /// Opens a private in-memory database (for tests), configured like [`open`](Self::open).
    /// SQLite keeps in-memory databases in `journal_mode = memory`, so there is no WAL file.
    pub async fn open_in_memory(options: OpenOptions) -> Result<Self, SqliteError> {
        Self::start(|| Ok(Connection::open_in_memory()?), options).await
    }

    async fn start(
        connect: impl FnOnce() -> Result<Connection, SqliteError> + Send + 'static,
        options: OpenOptions,
    ) -> Result<Self, SqliteError> {
        let (sender, receiver) = mpsc::channel();
        let (ready, opened) = oneshot::channel();
        thread::Builder::new()
            .name("delta-chord-sqlite".to_owned())
            .spawn(move || {
                let connection = match connect().and_then(|connection| {
                    configure(&connection, options)?;
                    Ok(connection)
                }) {
                    Ok(connection) => connection,
                    Err(error) => {
                        // Nobody waits for the result when `open` was dropped.
                        let _ = ready.send(Err(error));
                        return;
                    }
                };
                if ready.send(Ok(())).is_ok() {
                    serve(connection, &receiver);
                }
            })?;
        opened.await.map_err(|_| SqliteError::Closed)??;
        Ok(Self { sender })
    }

    /// Runs `f` with the connection on the database thread and returns its result.
    pub async fn call<T, F>(&self, f: F) -> Result<T, SqliteError>
    where
        F: FnOnce(&mut Connection) -> Result<T, SqliteError> + Send + 'static,
        T: Send + 'static,
    {
        let (reply, result) = oneshot::channel::<Outcome<T>>();
        let job: Job = Box::new(move |connection| {
            let outcome = panic::catch_unwind(AssertUnwindSafe(|| f(connection)));
            // Nobody waits for the result when the caller's future was dropped.
            let _ = reply.send(outcome);
        });
        self.sender.send(Message::Run(job)).map_err(|_| SqliteError::Closed)?;
        match result.await {
            Ok(Ok(result)) => result,
            Ok(Err(payload)) => panic::resume_unwind(payload),
            Err(_) => Err(SqliteError::Closed),
        }
    }

    /// Runs `f` inside `BEGIN IMMEDIATE` ... `COMMIT` on the database thread.
    ///
    /// When `f` or `COMMIT` fails, the transaction rolls back and that error is returned. When
    /// the rollback fails too, the result is [`SqliteError::Rollback`], so the error is never
    /// mistaken for a guaranteed rollback.
    pub async fn transaction<T, F>(&self, f: F) -> Result<T, SqliteError>
    where
        F: FnOnce(&Transaction<'_>) -> Result<T, SqliteError> + Send + 'static,
        T: Send + 'static,
    {
        self.call(move |connection| run_transaction(connection, f)).await
    }

    /// Runs `PRAGMA wal_checkpoint(TRUNCATE)` and closes the connection, after every closure
    /// sent before it. The connection is closed even when the checkpoint fails. Closing a
    /// database that another handle already closed succeeds.
    pub async fn close(self) -> Result<(), SqliteError> {
        let (reply, result) = oneshot::channel();
        if self.sender.send(Message::Close(reply)).is_err() {
            return Ok(());
        }
        result.await.unwrap_or(Ok(()))
    }
}

fn create_parent_dir(path: &Path) -> std::io::Result<()> {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => fs::create_dir_all(parent),
        _ => Ok(()),
    }
}

/// The longest busy timeout SQLite accepts.
const MAX_BUSY_TIMEOUT: Duration = Duration::from_millis(i32::MAX as u64);

fn configure(connection: &Connection, options: OpenOptions) -> Result<(), SqliteError> {
    connection.busy_timeout(options.busy_timeout.min(MAX_BUSY_TIMEOUT))?;
    connection.execute_batch("PRAGMA journal_mode = WAL")?;
    connection.execute_batch("PRAGMA synchronous = NORMAL")?;
    connection.execute_batch(&format!(
        "PRAGMA wal_autocheckpoint = {}",
        options.wal_autocheckpoint_pages
    ))?;
    Ok(())
}

fn serve(mut connection: Connection, receiver: &mpsc::Receiver<Message>) {
    // `recv` fails once every handle is dropped; dropping the connection then closes it.
    while let Ok(message) = receiver.recv() {
        match message {
            Message::Run(job) => job(&mut connection),
            Message::Close(reply) => {
                // Nobody waits for the result when `close` was dropped.
                let _ = reply.send(close_connection(connection));
                return;
            }
        }
    }
}

fn close_connection(connection: Connection) -> Result<(), SqliteError> {
    let checkpoint = connection.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)");
    let closed = connection.close().map_err(|(_, error)| error);
    checkpoint?;
    closed?;
    Ok(())
}

fn run_transaction<T>(
    connection: &mut Connection,
    f: impl FnOnce(&Transaction<'_>) -> Result<T, SqliteError>,
) -> Result<T, SqliteError> {
    // A panic in `f` drops the transaction, which rolls it back.
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let outcome = f(&transaction).and_then(|value| {
        transaction.execute_batch("COMMIT")?;
        Ok(value)
    });
    match outcome {
        // The transaction is no longer active, so dropping it does nothing.
        Ok(value) => Ok(value),
        Err(cause) => match transaction.rollback() {
            Ok(()) => Err(cause),
            Err(rollback) => Err(SqliteError::Rollback {
                cause: Box::new(cause),
                rollback,
            }),
        },
    }
}

/// One schema migration: its statements run in order, inside the migration transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Migration {
    /// The schema version this migration produces; versions start at 1 and are contiguous.
    pub version: u32,
    /// SQL run with `execute_batch`; one entry may hold several statements.
    pub statements: &'static [&'static str],
}

/// Applies every pending migration in one transaction, tracking the schema version in the
/// metadata table `table` (the same table upstream uses, so database files move between pi
/// and delta).
///
/// Fails without touching the database when `table` is not a valid prefix
/// ([`validate_schema_prefix`]) or the versions are not contiguous from 1. Refuses a database
/// whose version is newer than the last migration. Any failure rolls back the metadata table
/// and every migration of this call.
pub async fn apply_migrations(db: &Database, migrations: &[Migration], table: &str) -> Result<(), SqliteError> {
    validate_schema_prefix(table)?;
    if migrations
        .iter()
        .zip(1..)
        .any(|(migration, version)| migration.version != version)
    {
        return Err(SqliteError::NonContiguousMigrations);
    }
    let migrations = migrations.to_vec();
    let table = table.to_owned();
    db.transaction(move |transaction| {
        transaction.execute_batch(&format!(
            "CREATE TABLE IF NOT EXISTS {table} (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    version INTEGER NOT NULL CHECK (version >= 0)
) STRICT"
        ))?;
        transaction.execute(
            &format!("INSERT OR IGNORE INTO {table} (singleton, version) VALUES (1, 0)"),
            [],
        )?;
        let version: i64 =
            transaction.query_row(&format!("SELECT version FROM {table} WHERE singleton = 1"), [], |row| {
                row.get(0)
            })?;
        let supported = migrations.last().map_or(0, |migration| migration.version);
        if version > i64::from(supported) {
            return Err(SqliteError::NewerSchema { version, supported });
        }
        let update = format!("UPDATE {table} SET version = ?1 WHERE singleton = 1");
        for migration in migrations
            .iter()
            .filter(|migration| i64::from(migration.version) > version)
        {
            for statement in migration.statements {
                transaction.execute_batch(statement)?;
            }
            transaction.execute(&update, [migration.version])?;
        }
        Ok(())
    })
    .await
}

/// Checks a prefix before it becomes part of a SQLite identifier: it must match
/// `^[A-Za-z_][A-Za-z0-9_]*$` and must not start with `sqlite_` (any case).
pub fn validate_schema_prefix(prefix: &str) -> Result<(), SqliteError> {
    let mut chars = prefix.chars();
    let valid = chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !valid {
        return Err(SqliteError::InvalidSchemaPrefix(prefix.to_owned()));
    }
    if prefix
        .get(..7)
        .is_some_and(|start| start.eq_ignore_ascii_case("sqlite_"))
    {
        return Err(SqliteError::ReservedSchemaPrefix(prefix.to_owned()));
    }
    Ok(())
}
