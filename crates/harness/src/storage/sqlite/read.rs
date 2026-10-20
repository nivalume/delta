//! Reads on one connection: the backend's queries, also serving as the commit view.

use delta_chord::sqlite::rusqlite::types::Value;
use delta_chord::sqlite::rusqlite::{Connection, OptionalExtension, params, params_from_iter};
use serde::de::DeserializeOwned;

use super::codec::{decode, decode_content, int, uint};
use super::schema::Schema;
use crate::records::{
    ConversationId, ConversationParent, ConversationQuery, ConversationRecord, DocumentAddress, DocumentContent,
    DocumentCopySource, DocumentId, DocumentPoint, DocumentQuery, DocumentRecord, DocumentScope, EntryId, EntryRecord,
    ScanOrder, Seq, StoredDocument, StoredEntry, SubmissionId, SubmissionQuery, SubmissionRecord, TaskId, TaskQuery,
    TaskRecord,
};
use crate::storage::ancestry::{EntryReader, Lineage};
use crate::storage::batch::{CommitView, ExistingDocument, RecordType};
use crate::storage::error::StorageError;
use crate::storage::replay::{self, materialize};
use crate::storage::scan::ScanStart;

/// The queries of one schema on one connection (or transaction).
pub(super) struct Reader<'c> {
    pub(super) conn: &'c Connection,
    pub(super) schema: &'c Schema,
}

/// `WHERE` conditions with their positional parameters.
#[derive(Default)]
struct Conditions {
    sql: Vec<String>,
    params: Vec<Value>,
}

impl Conditions {
    fn push(&mut self, condition: impl Into<String>, params: impl IntoIterator<Item = Value>) {
        self.sql.push(condition.into());
        self.params.extend(params);
    }

    fn eq(&mut self, column: &str, value: impl Into<Value>) {
        self.push(format!("{column} = ?"), [value.into()]);
    }

    /// The cursor's side of an id scan.
    fn after(&mut self, start: &ScanStart) -> Result<(), StorageError> {
        if let Some(after) = start.after {
            let op = match start.order {
                ScanOrder::Ascending => ">",
                ScanOrder::Descending => "<",
            };
            self.push(format!("id {op} ?"), [Value::Integer(int(after)?)]);
        }
        Ok(())
    }

    /// Incarnations alive at `at`.
    fn alive(&mut self, at: DocumentPoint) -> Result<(), StorageError> {
        match at {
            DocumentPoint::Current => self.push("retired_at IS NULL", []),
            DocumentPoint::At(seq) => {
                let seq = Value::Integer(int(seq.0)?);
                self.push(
                    "created_at <= ? AND (retired_at IS NULL OR retired_at > ?)",
                    [seq.clone(), seq],
                );
            }
        }
        Ok(())
    }

    fn address(&mut self, address: &DocumentAddress) {
        self.eq("kind", address.kind.clone());
        self.scope(address.scope);
        self.push("key IS ?", [address.key.clone().map_or(Value::Null, Value::Text)]);
    }

    fn scope(&mut self, scope: DocumentScope) {
        self.eq("scope_kind", scope.kind().as_str().to_owned());
        // Stored owner ids fit `i64`; a larger one matches nothing.
        self.eq("owner_id", i64::try_from(scope.owner_id()).unwrap_or(-1));
    }

    fn render(&self) -> String {
        if self.sql.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", self.sql.join(" AND "))
        }
    }
}

fn order_sql(order: ScanOrder) -> &'static str {
    match order {
        ScanOrder::Ascending => "ASC",
        ScanOrder::Descending => "DESC",
    }
}

fn limit(fetch: usize) -> i64 {
    i64::try_from(fetch).unwrap_or(i64::MAX)
}

/// A range bound as an `INTEGER`: past `i64::MAX` nothing is stored, so the maximum bounds
/// the same rows.
fn saturating_int(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

impl Reader<'_> {
    /// Decodes the `record` column of every row `sql` returns.
    fn records<T: DeserializeOwned>(&self, sql: &str, params: Vec<Value>, table: &str) -> Result<Vec<T>, StorageError> {
        let mut statement = self.conn.prepare_cached(sql)?;
        let rows = statement.query_map(params_from_iter(params), |row| row.get::<_, String>(0))?;
        let mut records = Vec::new();
        for text in rows {
            records.push(decode(&text?, || format!("A {table} row"))?);
        }
        Ok(records)
    }

    fn record<T: DeserializeOwned>(
        &self,
        sql: &str,
        params: Vec<Value>,
        table: &str,
    ) -> Result<Option<T>, StorageError> {
        Ok(self.records(sql, params, table)?.into_iter().next())
    }

    /// A filtered id scan of `table`, one page plus one.
    fn scan<T: DeserializeOwned>(
        &self,
        table: &str,
        mut conditions: Conditions,
        start: &ScanStart,
    ) -> Result<Vec<T>, StorageError> {
        conditions.after(start)?;
        let sql = format!(
            "SELECT record FROM {table} {} ORDER BY id {} LIMIT {}",
            conditions.render(),
            order_sql(start.order),
            limit(start.fetch)
        );
        self.records(&sql, conditions.params, table)
    }

    pub(super) fn conversation(&self, id: ConversationId) -> Result<Option<ConversationRecord>, StorageError> {
        let sql = format!("SELECT record FROM {} WHERE id = ?", self.schema.conversations);
        self.record(&sql, vec![Value::Integer(int(id.0)?)], "conversation")
    }

    pub(super) fn scan_conversations(
        &self,
        query: &ConversationQuery,
        start: &ScanStart,
    ) -> Result<Vec<ConversationRecord>, StorageError> {
        let mut conditions = Conditions::default();
        if let Some(owner) = query.owner_conversation_id {
            conditions.eq("owner_conversation_id", int(owner.0)?);
        }
        if let Some(owner) = query.owner_task_id {
            conditions.eq("owner_task_id", int(owner.0)?);
        }
        self.scan(&self.schema.conversations, conditions, start)
    }

    pub(super) fn entry(&self, id: EntryId) -> Result<Option<StoredEntry>, StorageError> {
        let sql = format!("SELECT record, commit_seq FROM {} WHERE id = ?", self.schema.entries);
        let row = self
            .conn
            .prepare_cached(&sql)?
            .query_row(params![int(id.0)?], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .optional()?;
        let Some((record, commit_seq)) = row else {
            return Ok(None);
        };
        Ok(Some(StoredEntry {
            entry: decode(&record, || format!("Entry {id}"))?,
            commit_seq: Seq(uint(commit_seq)?),
        }))
    }

    pub(super) fn task(&self, id: TaskId) -> Result<Option<TaskRecord>, StorageError> {
        let sql = format!("SELECT record FROM {} WHERE id = ?", self.schema.tasks);
        self.record(&sql, vec![Value::Integer(int(id.0)?)], "task")
    }

    pub(super) fn scan_tasks(&self, query: &TaskQuery, start: &ScanStart) -> Result<Vec<TaskRecord>, StorageError> {
        let mut conditions = Conditions::default();
        if let Some(conversation) = query.conversation_id {
            conditions.eq("conversation_id", int(conversation.0)?);
        }
        if let Some(kind) = &query.kind {
            conditions.eq("kind", kind.clone());
        }
        if let Some(status) = query.status {
            conditions.eq("status", status.as_str().to_owned());
        }
        if let Some(marked) = query.abort_requested {
            conditions.eq("abort_requested", i64::from(marked));
        }
        if let Some(background) = query.background {
            conditions.eq("background", i64::from(background));
        }
        self.scan(&self.schema.tasks, conditions, start)
    }

    pub(super) fn submission(&self, id: SubmissionId) -> Result<Option<SubmissionRecord>, StorageError> {
        let sql = format!("SELECT record FROM {} WHERE id = ?", self.schema.submissions);
        self.record(&sql, vec![Value::Integer(int(id.0)?)], "submission")
    }

    pub(super) fn scan_submissions(
        &self,
        query: &SubmissionQuery,
        start: &ScanStart,
    ) -> Result<Vec<SubmissionRecord>, StorageError> {
        let mut conditions = Conditions::default();
        if let Some(conversation) = query.conversation_id {
            conditions.eq("conversation_id", int(conversation.0)?);
        }
        if let Some(status) = query.status {
            conditions.eq("status", status.as_str().to_owned());
        }
        self.scan(&self.schema.submissions, conditions, start)
    }

    pub(super) fn submission_by_request(
        &self,
        conversation: ConversationId,
        request_id: &str,
    ) -> Result<Option<SubmissionRecord>, StorageError> {
        let sql = format!(
            "SELECT record FROM {} WHERE conversation_id = ? AND request_id = ?",
            self.schema.submissions
        );
        let params = vec![Value::Integer(int(conversation.0)?), Value::Text(request_id.to_owned())];
        self.record(&sql, params, "submission")
    }

    pub(super) fn find_document(
        &self,
        address: &DocumentAddress,
        at: DocumentPoint,
    ) -> Result<Option<DocumentRecord>, StorageError> {
        let mut conditions = Conditions::default();
        conditions.address(address);
        conditions.alive(at)?;
        let sql = format!(
            "SELECT record FROM {} {} ORDER BY created_at DESC LIMIT 1",
            self.schema.documents,
            conditions.render()
        );
        self.record(&sql, conditions.params, "document")
    }

    fn document_record(&self, id: DocumentId) -> Result<Option<DocumentRecord>, StorageError> {
        let sql = format!("SELECT record FROM {} WHERE id = ?", self.schema.documents);
        self.record(&sql, vec![Value::Integer(int(id.0)?)], "document")
    }

    pub(super) fn document(&self, id: DocumentId, at: DocumentPoint) -> Result<Option<StoredDocument>, StorageError> {
        let Some(record) = self.document_record(id)? else {
            return Ok(None);
        };
        if matches!(at, DocumentPoint::At(_)) && !record.keeps_history() {
            return Err(replay::no_history(id));
        }
        if !record.is_alive_at(at) {
            return Ok(None);
        }
        let upper = match at {
            DocumentPoint::Current => i64::MAX,
            DocumentPoint::At(seq) => int(seq.0)?,
        };
        let document = int(id.0)?;
        let base_sql = format!(
            "SELECT seq, version, content FROM {} WHERE document_id = ? AND kind = 'base' AND seq <= ? \
             ORDER BY seq DESC LIMIT 1",
            self.schema.revisions
        );
        let base = self
            .conn
            .prepare_cached(&base_sql)?
            .query_row(params![document, upper], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, String>(2)?))
            })
            .optional()?;
        let Some((base_seq, version, content)) = base else {
            return Err(replay::missing_base(id));
        };
        let DocumentContent::Base { version, value } =
            decode_content(id.0, Seq(uint(base_seq)?), "base", version, &content)?
        else {
            return Err(replay::missing_base(id));
        };
        let tail_sql = format!(
            "SELECT seq, kind, version, content FROM {} WHERE document_id = ? AND seq > ? AND seq <= ? ORDER BY seq",
            self.schema.revisions
        );
        let mut statement = self.conn.prepare_cached(&tail_sql)?;
        let rows = statement.query_map(params![document, base_seq, upper], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?;
        let tail = rows.map(|row| {
            let (seq, kind, version, content) = row?;
            let seq = Seq(uint(seq)?);
            Ok((seq, decode_content(id.0, seq, &kind, version, &content)?))
        });
        materialize(record, version, value, tail).map(Some)
    }

    pub(super) fn scan_documents(
        &self,
        query: &DocumentQuery,
        start: &ScanStart,
    ) -> Result<Vec<DocumentRecord>, StorageError> {
        let mut conditions = Conditions::default();
        conditions.scope(query.scope);
        if let Some(kind) = &query.kind {
            conditions.eq("kind", kind.clone());
        }
        conditions.alive(query.at)?;
        self.scan(&self.schema.documents, conditions, start)
    }

    /// `(next_id, next_seq)` of the metadata row.
    pub(super) fn metadata(&self) -> Result<(u64, u64), StorageError> {
        let sql = format!(
            "SELECT next_id, next_seq FROM {} WHERE singleton = 1",
            self.schema.metadata
        );
        let row = self
            .conn
            .prepare_cached(&sql)?
            .query_row([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)))
            .optional()?;
        let (next_id, next_seq) =
            row.ok_or_else(|| StorageError::Corrupt("SQLite storage metadata is missing".to_owned()))?;
        Ok((uint(next_id)?, uint(next_seq)?))
    }
}

impl EntryReader for Reader<'_> {
    fn lineage(&mut self, conversation: ConversationId) -> Result<Lineage, StorageError> {
        let sql = format!(
            "SELECT parent_id, parent_at FROM {} WHERE id = ?",
            self.schema.conversations
        );
        let row = self
            .conn
            .prepare_cached(&sql)?
            .query_row(params![int(conversation.0)?], |row| {
                Ok((row.get::<_, Option<i64>>(0)?, row.get::<_, Option<i64>>(1)?))
            })
            .optional()?;
        Ok(match row {
            None => Lineage::Unknown,
            Some((Some(parent), Some(at))) => Lineage::Forked(ConversationParent {
                conversation_id: ConversationId(uint(parent)?),
                at: EntryId(uint(at)?),
            }),
            Some(_) => Lineage::Root,
        })
    }

    fn entry(&mut self, id: EntryId) -> Result<Option<StoredEntry>, StorageError> {
        Reader::entry(self, id)
    }

    fn entries(
        &mut self,
        conversation: ConversationId,
        min: Option<EntryId>,
        max: Option<EntryId>,
        order: ScanOrder,
        fetch: usize,
    ) -> Result<Vec<EntryRecord>, StorageError> {
        let sql = format!(
            "SELECT record FROM {} WHERE conversation_id = ? AND id >= ? AND id <= ? ORDER BY id {} LIMIT ?",
            self.schema.entries,
            order_sql(order)
        );
        let params = vec![
            Value::Integer(int(conversation.0)?),
            Value::Integer(min.map_or(0, |min| saturating_int(min.0))),
            Value::Integer(max.map_or(i64::MAX, |max| saturating_int(max.0))),
            Value::Integer(limit(fetch)),
        ];
        self.records(&sql, params, "entry")
    }

    fn latest_head(
        &mut self,
        conversation: ConversationId,
        max: Option<EntryId>,
    ) -> Result<Option<EntryRecord>, StorageError> {
        let sql = format!(
            "SELECT record FROM {} WHERE conversation_id = ? AND head IS NOT NULL AND id <= ? ORDER BY id DESC LIMIT 1",
            self.schema.entries
        );
        let max = max.map_or(i64::MAX, |max| saturating_int(max.0));
        self.record(
            &sql,
            vec![Value::Integer(int(conversation.0)?), Value::Integer(max)],
            "entry",
        )
    }
}

impl CommitView for Reader<'_> {
    fn record_type(&mut self, id: u64) -> Result<Option<RecordType>, StorageError> {
        let Ok(id) = i64::try_from(id) else {
            return Ok(None);
        };
        let sql = format!("SELECT record_type FROM {} WHERE id = ?", self.schema.record_ids);
        let name = self
            .conn
            .prepare_cached(&sql)?
            .query_row(params![id], |row| row.get::<_, String>(0))
            .optional()?;
        name.map(|name| {
            RecordType::parse(&name).ok_or_else(|| StorageError::Corrupt(format!("ID {id} has record type {name:?}")))
        })
        .transpose()
    }

    fn document(&mut self, id: DocumentId) -> Result<Option<ExistingDocument>, StorageError> {
        let Some(record) = self.document_record(id)? else {
            return Ok(None);
        };
        let sql = format!(
            "SELECT version FROM {} WHERE document_id = ? ORDER BY seq DESC LIMIT 1",
            self.schema.revisions
        );
        let newest = self
            .conn
            .prepare_cached(&sql)?
            .query_row(params![int(id.0)?], |row| row.get::<_, i64>(0))
            .optional()?;
        let newest_version = newest
            .map(|version| {
                u32::try_from(version)
                    .map_err(|_| StorageError::Corrupt(format!("Document {id} has version {version}")))
            })
            .transpose()?;
        Ok(Some(ExistingDocument { record, newest_version }))
    }

    fn current_document(&mut self, address: &DocumentAddress) -> Result<Option<DocumentId>, StorageError> {
        Ok(self
            .find_document(address, DocumentPoint::Current)?
            .map(|record| record.id))
    }

    fn materialize(&mut self, source: DocumentCopySource) -> Result<Option<StoredDocument>, StorageError> {
        Reader::document(self, source.id, source.at)
    }

    fn submission_by_request(
        &mut self,
        conversation: ConversationId,
        request_id: &str,
    ) -> Result<Option<SubmissionId>, StorageError> {
        Ok(Reader::submission_by_request(self, conversation, request_id)?.map(|record| record.id))
    }
}
