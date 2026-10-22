//! The SQLite backend on `delta_chord::sqlite`: delta's own schema (version 1), an
//! optional table prefix, one `BEGIN IMMEDIATE` transaction per commit.

mod codec;
mod read;
mod schema;
mod write;

use std::fmt;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use delta_chord::context::Context;
use delta_chord::sqlite::{Database, OpenOptions, SqliteError, apply_migrations};
use futures::FutureExt;
use tokio::sync::RwLock;

use self::codec::{MAX_ID, from_sqlite};
use self::read::Reader;
pub use self::schema::{CURRENT_SCHEMA_VERSION, Schema, migrations};
use super::ancestry;
use super::error::StorageError;
use super::scan::ScanStart;
use super::{Storage, StorageFuture};
use crate::records::{
    ConversationId, ConversationQuery, ConversationRecord, Cursor, DocumentAddress, DocumentId, DocumentPoint,
    DocumentQuery, DocumentRecord, EntryId, EntryQuery, EntryRecord, Page, RawId, RecordId, ScanOrder, Seq,
    StorageWrite, StoredDocument, StoredEntry, SubmissionId, SubmissionQuery, SubmissionRecord, TaskId, TaskQuery,
    TaskRecord,
};

/// How [`SqliteStorage::open`] opens a database file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SqliteOptions {
    /// Prefix of every table and index name; must pass
    /// [`validate_schema_prefix`](delta_chord::sqlite::validate_schema_prefix). Data is not
    /// moved when the prefix changes.
    pub prefix: Option<String>,
    pub open: OpenOptions,
}

/// Storage in a SQLite database.
///
/// Reads run in a deferred transaction, so a multi-query read (a document's base and tail,
/// an ancestry walk) observes one committed state. `next_id` is minted in memory and
/// persisted by commits; `i64::MAX` is never minted.
pub struct SqliteStorage {
    /// `None` once closed. Calls hold a read guard while they run; `close` takes the write
    /// guard, so it waits for calls underway and later calls see `None`.
    db: RwLock<Option<Database>>,
    schema: Arc<Schema>,
    next_id: Mutex<u64>,
}

impl fmt::Debug for SqliteStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SqliteStorage")
            .field("prefix", &self.schema.prefix())
            .finish_non_exhaustive()
    }
}

impl SqliteStorage {
    /// Opens (creating if needed) the database at `path` (`:memory:` for a private
    /// in-memory database) and migrates it. An invalid prefix fails before the file or its
    /// directory is created.
    pub async fn open(path: impl AsRef<Path>, options: SqliteOptions) -> Result<Self, StorageError> {
        let schema = Schema::new(options.prefix.as_deref())?;
        let db = Database::open(path, options.open).await?;
        Self::start(db, schema).await
    }

    /// Uses an open database: migrates it under `prefix` and reads its metadata. A failed
    /// open closes the database and returns the first error.
    pub async fn open_database(db: Database, prefix: Option<String>) -> Result<Self, StorageError> {
        let schema = match Schema::new(prefix.as_deref()) {
            Ok(schema) => schema,
            Err(error) => {
                // The first error is the one reported.
                let _ = db.close().await;
                return Err(error);
            }
        };
        Self::start(db, schema).await
    }

    async fn start(db: Database, schema: Schema) -> Result<Self, StorageError> {
        let schema = Arc::new(schema);
        let migrated = apply_migrations(&db, &migrations(&schema), &schema.version_table).await;
        let metadata = match migrated {
            Ok(()) => {
                let schema = Arc::clone(&schema);
                db.call(move |conn| Reader { conn, schema: &schema }.metadata().map_err(SqliteError::other))
                    .await
                    .map_err(from_sqlite)
            }
            Err(error) => Err(error.into()),
        };
        match metadata {
            Ok((next_id, _)) => Ok(Self {
                db: RwLock::new(Some(db)),
                schema,
                next_id: Mutex::new(next_id),
            }),
            Err(error) => {
                // The first error is the one reported.
                let _ = db.close().await;
                Err(error)
            }
        }
    }

    /// Runs `f` on the database thread inside a deferred (read) transaction.
    fn read<T, F>(&self, f: F) -> StorageFuture<'_, T>
    where
        F: FnOnce(&mut Reader<'_>) -> Result<T, StorageError> + Send + 'static,
        T: Send + 'static,
    {
        let schema = Arc::clone(&self.schema);
        async move {
            let guard = self.db.read().await;
            let db = guard.as_ref().ok_or(StorageError::Closed)?;
            db.call(move |conn| {
                let run = || -> Result<T, StorageError> {
                    let transaction = conn.transaction()?;
                    let value = f(&mut Reader {
                        conn: &transaction,
                        schema: &schema,
                    })?;
                    transaction.commit()?;
                    Ok(value)
                };
                run().map_err(SqliteError::other)
            })
            .await
            .map_err(from_sqlite)
        }
        .boxed()
    }

    fn bump_next_id(&self, next_id: u64) {
        let mut current = self.next_id.lock().unwrap_or_else(PoisonError::into_inner);
        *current = (*current).max(next_id);
    }
}

impl Storage for SqliteStorage {
    fn commit(&self, writes: Vec<StorageWrite>, _ctx: &Context) -> StorageFuture<'_, Seq> {
        let schema = Arc::clone(&self.schema);
        async move {
            let guard = self.db.read().await;
            let db = guard.as_ref().ok_or(StorageError::Closed)?;
            let (seq, next_id) = db
                .transaction(move |transaction| write::commit(transaction, &schema, writes).map_err(SqliteError::other))
                .await
                .map_err(from_sqlite)?;
            self.bump_next_id(next_id);
            Ok(seq)
        }
        .boxed()
    }

    fn mint_id(&self) -> StorageFuture<'_, RawId> {
        async move {
            let guard = self.db.read().await;
            if guard.is_none() {
                return Err(StorageError::Closed);
            }
            let mut next_id = self.next_id.lock().unwrap_or_else(PoisonError::into_inner);
            if *next_id >= MAX_ID {
                return Err(StorageError::IdSpaceExhausted);
            }
            let id = *next_id;
            *next_id += 1;
            Ok(RawId(id))
        }
        .boxed()
    }

    fn conversation(&self, id: ConversationId, _ctx: &Context) -> StorageFuture<'_, Option<ConversationRecord>> {
        self.read(move |reader| reader.conversation(id))
    }

    fn scan_conversations(
        &self,
        query: ConversationQuery,
        limit: usize,
        cursor: Option<Cursor>,
        _ctx: &Context,
    ) -> StorageFuture<'_, Page<ConversationRecord>> {
        self.read(move |reader| {
            let start = ScanStart::new(query.order, cursor.as_ref(), ScanOrder::Ascending, limit)?;
            Ok(start.page(reader.scan_conversations(&query, &start)?, |record| record.id.get()))
        })
    }

    fn entry(&self, id: EntryId, _ctx: &Context) -> StorageFuture<'_, Option<StoredEntry>> {
        self.read(move |reader| reader.entry(id))
    }

    fn visible_entry(
        &self,
        conversation: ConversationId,
        id: EntryId,
        _ctx: &Context,
    ) -> StorageFuture<'_, Option<StoredEntry>> {
        self.read(move |reader| ancestry::visible_entry(reader, conversation, id))
    }

    fn find_latest_head_marker(
        &self,
        conversation: ConversationId,
        at_or_before: Option<EntryId>,
        _ctx: &Context,
    ) -> StorageFuture<'_, Option<EntryRecord>> {
        self.read(move |reader| ancestry::latest_head(reader, conversation, at_or_before))
    }

    fn scan_entries(
        &self,
        query: EntryQuery,
        limit: usize,
        cursor: Option<Cursor>,
        _ctx: &Context,
    ) -> StorageFuture<'_, Page<EntryRecord>> {
        self.read(move |reader| {
            let start = ScanStart::new(query.order, cursor.as_ref(), ScanOrder::Descending, limit)?;
            let entries = ancestry::scan(
                reader,
                query.conversation_id,
                query.min_entry_id,
                query.max_entry_id,
                &start,
            )?;
            Ok(start.page(entries, |entry| entry.id.get()))
        })
    }

    fn task(&self, id: TaskId, _ctx: &Context) -> StorageFuture<'_, Option<TaskRecord>> {
        self.read(move |reader| reader.task(id))
    }

    fn scan_tasks(
        &self,
        query: TaskQuery,
        limit: usize,
        cursor: Option<Cursor>,
        _ctx: &Context,
    ) -> StorageFuture<'_, Page<TaskRecord>> {
        self.read(move |reader| {
            let start = ScanStart::new(query.order, cursor.as_ref(), ScanOrder::Ascending, limit)?;
            Ok(start.page(reader.scan_tasks(&query, &start)?, |task| task.id.get()))
        })
    }

    fn submission(&self, id: SubmissionId, _ctx: &Context) -> StorageFuture<'_, Option<SubmissionRecord>> {
        self.read(move |reader| reader.submission(id))
    }

    fn scan_submissions(
        &self,
        query: SubmissionQuery,
        limit: usize,
        cursor: Option<Cursor>,
        _ctx: &Context,
    ) -> StorageFuture<'_, Page<SubmissionRecord>> {
        self.read(move |reader| {
            let start = ScanStart::new(query.order, cursor.as_ref(), ScanOrder::Ascending, limit)?;
            Ok(start.page(reader.scan_submissions(&query, &start)?, |submission| {
                submission.id.get()
            }))
        })
    }

    fn submission_by_request(
        &self,
        conversation: ConversationId,
        request_id: &str,
        _ctx: &Context,
    ) -> StorageFuture<'_, Option<SubmissionRecord>> {
        let request_id = request_id.to_owned();
        self.read(move |reader| reader.submission_by_request(conversation, &request_id))
    }

    fn find_document(
        &self,
        address: &DocumentAddress,
        at: DocumentPoint,
        _ctx: &Context,
    ) -> StorageFuture<'_, Option<DocumentRecord>> {
        let address = address.clone();
        self.read(move |reader| reader.find_document(&address, at))
    }

    fn document(&self, id: DocumentId, at: DocumentPoint, _ctx: &Context) -> StorageFuture<'_, Option<StoredDocument>> {
        self.read(move |reader| reader.document(id, at))
    }

    fn scan_documents(
        &self,
        query: DocumentQuery,
        limit: usize,
        cursor: Option<Cursor>,
        _ctx: &Context,
    ) -> StorageFuture<'_, Page<DocumentRecord>> {
        self.read(move |reader| {
            let start = ScanStart::new(Some(ScanOrder::Ascending), cursor.as_ref(), ScanOrder::Ascending, limit)?;
            Ok(start.page(reader.scan_documents(&query, &start)?, |record| record.id.get()))
        })
    }

    fn close(&self, _ctx: &Context) -> StorageFuture<'_, ()> {
        async move {
            let db = self.db.write().await.take();
            match db {
                Some(db) => db.close().await.map_err(from_sqlite),
                None => Ok(()),
            }
        }
        .boxed()
    }
}
