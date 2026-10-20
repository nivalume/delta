# SQLite storage helpers (`sqlite`, feature `sqlite`)

Corresponds to upstream `src/sqlite` (`database.ts`, `migrations.ts`, `schema.ts`,
`node.ts`).

## Problem

Durable stores sessions, documents and their revision batches in SQLite, and every commit
is one transaction. Upstream defines a portable async facade (`SqliteDatabase`) so the same
storage code runs on Node's `node:sqlite` and on Cloudflare Durable Objects, plus an atomic
migration runner and a check for table-name prefixes. Because the facade is async, its
transaction callback is async too, and the contract needs rules for queuing other work
behind a transaction and for calls into the database from inside one (they never settle).

delta has one driver, so it uses it directly and keeps transactions synchronous.

## `Database`

```rust
pub struct Database;

impl Database {
    pub async fn open(path: impl AsRef<Path>, options: OpenOptions) -> Result<Database, SqliteError>;
    pub async fn call<T, F>(&self, f: F) -> Result<T, SqliteError>
    where F: FnOnce(&mut rusqlite::Connection) -> Result<T, SqliteError> + Send + 'static, T: Send + 'static;
    pub async fn transaction<T, F>(&self, f: F) -> Result<T, SqliteError>
    where F: FnOnce(&rusqlite::Transaction<'_>) -> Result<T, SqliteError> + Send + 'static, T: Send + 'static;
    pub async fn close(self) -> Result<(), SqliteError>;
}
```

- The connection lives on a dedicated thread; `call` and `transaction` send a closure to it
  and await the result. Closures run one at a time in the order they were sent, so a
  transaction never interleaves with other work, without any queuing rules.
- `transaction` runs `BEGIN IMMEDIATE`, the closure, then `COMMIT`; when the closure returns
  an error, it rolls back and returns that error. A failed rollback returns
  `SqliteError::Rollback { cause, rollback }`, so a caller never mistakes the closure's
  error for a guaranteed rollback (the rule upstream states for adapters).
- The closures are synchronous, so storage code that needs several statements in one
  transaction writes them as one closure. Durable's transaction bodies are SQL only, so
  nothing is lost. Calling `Database` from inside a closure is impossible without blocking,
  which is the same restriction upstream documents ("never settles"), now visible in the
  types.
- `open` applies the pragmas of upstream's Node driver: WAL, `synchronous = NORMAL`,
  `wal_autocheckpoint = 1000`, a 5 s busy timeout. `close` runs
  `wal_checkpoint(TRUNCATE)` and closes the connection.
- Values are `rusqlite`'s types (`rusqlite::types::Value`, `ToSql`, `FromSql`). Integers stay
  exact `i64` (upstream returns JS numbers unless a column is read as `bigint`).

## Migrations

```rust
pub struct Migration { pub version: u32, pub statements: &'static [&'static str] }

pub async fn apply_migrations(db: &Database, migrations: &'static [Migration], table: &str)
    -> Result<(), SqliteError>;
pub fn validate_schema_prefix(name: &str) -> Result<(), SqliteError>;
```

Same behavior and same metadata table as upstream, so a database file created by pi
opens in delta and the other way round:

- Versions must be contiguous from 1.
- One transaction: create the metadata table if needed, insert version 0 if empty, read
  the version, refuse a database newer than the code, run the missing migrations and store
  each version.
- The metadata table is

  ```sql
  CREATE TABLE IF NOT EXISTS {table} (
      singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
      version INTEGER NOT NULL CHECK (version >= 0)
  ) STRICT
  ```

- `validate_schema_prefix` accepts `^[A-Za-z_][A-Za-z0-9_]*$` and refuses names starting
  with `sqlite_` (any case).

## Dependency

`rusqlite` with the `bundled` feature, behind this crate's `sqlite` feature (off by
default), so crates that only need revisions and state do not compile SQLite.

## Differences from upstream

- No portable `SqliteDatabase` / `SqliteExecutor` interface and no Cloudflare driver.
- Synchronous transaction closures on the database thread instead of async callbacks with
  a transaction handle.
- `rusqlite` types instead of `SqliteValue`.
