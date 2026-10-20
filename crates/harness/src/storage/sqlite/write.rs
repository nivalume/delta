//! One commit inside one `BEGIN IMMEDIATE` transaction: plan, write, update metadata.

use delta_chord::sqlite::rusqlite::{Connection, params};

use super::codec::{MAX_ID, encode, encode_content, int};
use super::read::Reader;
use super::schema::Schema;
use crate::records::{
    ConversationRecord, DocumentContent, EntryRecord, Seq, StorageWrite, SubmissionRecord, TaskRecord,
};
use crate::storage::batch::{Batch, DocumentAction, RecordType, RecordWrite, Target, plan};
use crate::storage::error::StorageError;

/// Plans and writes `writes` on `conn`, inside the caller's transaction. Returns the
/// batch's sequence and the new `next_id`.
pub(super) fn commit(
    conn: &Connection,
    schema: &Schema,
    writes: Vec<StorageWrite>,
) -> Result<(Seq, u64), StorageError> {
    let mut reader = Reader { conn, schema };
    let (next_id, next_seq) = reader.metadata()?;
    let batch = plan(writes, &mut reader)?;
    let seq = Seq(next_seq);
    let next_id = batch
        .highest_id
        .map_or(next_id, |highest| next_id.max(highest.saturating_add(1).min(MAX_ID)));
    let writer = Writer { conn, schema, seq };
    writer.batch(&batch)?;
    let sql = format!(
        "UPDATE {} SET next_id = ?, next_seq = ? WHERE singleton = 1",
        schema.metadata
    );
    conn.prepare_cached(&sql)?
        .execute(params![int(next_id)?, int(next_seq + 1)?])?;
    Ok((seq, next_id))
}

struct Writer<'c> {
    conn: &'c Connection,
    schema: &'c Schema,
    seq: Seq,
}

impl Writer<'_> {
    fn batch(&self, batch: &Batch) -> Result<(), StorageError> {
        for write in &batch.records {
            match write {
                RecordWrite::Conversation(record) => self.conversation(record)?,
                RecordWrite::Entry(entry) => self.entry(entry)?,
                RecordWrite::Task(task) => self.task(task)?,
                RecordWrite::Submission(submission) => self.submission(submission)?,
            }
        }
        for action in &batch.documents {
            self.document(action)?;
        }
        Ok(())
    }

    fn execute(&self, sql: &str, params: impl delta_chord::sqlite::rusqlite::Params) -> Result<(), StorageError> {
        self.conn.prepare_cached(sql)?.execute(params)?;
        Ok(())
    }

    /// Records `id` as owned by `table` (ignored when a replacement already owns it).
    fn claim(&self, id: u64, table: RecordType) -> Result<(), StorageError> {
        let sql = format!(
            "INSERT OR IGNORE INTO {} (id, record_type) VALUES (?, ?)",
            self.schema.record_ids
        );
        self.execute(&sql, params![int(id)?, table.as_str()])
    }

    fn conversation(&self, record: &ConversationRecord) -> Result<(), StorageError> {
        self.claim(record.id.0, RecordType::Conversation)?;
        let sql = format!(
            "INSERT INTO {} (id, parent_id, parent_at, owner_conversation_id, owner_task_id, record) \
             VALUES (?, ?, ?, ?, ?, ?)",
            self.schema.conversations
        );
        let parent = record.parent.map(|parent| (parent.conversation_id.0, parent.at.0));
        let owner = record.owner.map(|owner| (owner.conversation_id.0, owner.task_id.0));
        self.execute(
            &sql,
            params![
                int(record.id.0)?,
                parent.map(|(id, _)| int(id)).transpose()?,
                parent.map(|(_, at)| int(at)).transpose()?,
                owner.map(|(id, _)| int(id)).transpose()?,
                owner.map(|(_, task)| int(task)).transpose()?,
                encode(record)?,
            ],
        )
    }

    fn entry(&self, entry: &EntryRecord) -> Result<(), StorageError> {
        self.claim(entry.id.0, RecordType::Entry)?;
        let sql = format!(
            "INSERT INTO {} (id, conversation_id, head, commit_seq, record) VALUES (?, ?, ?, ?, ?)",
            self.schema.entries
        );
        self.execute(
            &sql,
            params![
                int(entry.id.0)?,
                int(entry.conversation_id.0)?,
                entry.head.map(|head| int(head.0)).transpose()?,
                int(self.seq.0)?,
                encode(entry)?,
            ],
        )
    }

    fn task(&self, task: &TaskRecord) -> Result<(), StorageError> {
        self.claim(task.id.0, RecordType::Task)?;
        let sql = format!(
            "INSERT OR REPLACE INTO {} (id, conversation_id, kind, status, abort_requested, background, record) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            self.schema.tasks
        );
        self.execute(
            &sql,
            params![
                int(task.id.0)?,
                int(task.conversation_id.0)?,
                task.kind,
                task.state.status().as_str(),
                task.abort_requested,
                task.background,
                encode(task)?,
            ],
        )
    }

    fn submission(&self, submission: &SubmissionRecord) -> Result<(), StorageError> {
        self.claim(submission.id.0, RecordType::Submission)?;
        let sql = format!(
            "INSERT OR REPLACE INTO {} (id, conversation_id, request_id, status, record) VALUES (?, ?, ?, ?, ?)",
            self.schema.submissions
        );
        self.execute(
            &sql,
            params![
                int(submission.id.0)?,
                int(submission.conversation_id.0)?,
                submission.request_id,
                submission.state.status().as_str(),
                encode(submission)?,
            ],
        )
    }

    /// Content before retirement; a base on a current-only incarnation and its retirement
    /// reclaim earlier revisions.
    fn document(&self, action: &DocumentAction) -> Result<(), StorageError> {
        let id = int(action.id.0)?;
        let record = action.record_at(self.seq);
        let keeps_history = record.keeps_history();
        match &action.target {
            Target::Created(_) => {
                self.claim(action.id.0, RecordType::Document)?;
                let sql = format!(
                    "INSERT INTO {} (id, kind, key, scope_kind, owner_id, created_at, retired_at, record) \
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    self.schema.documents
                );
                self.execute(
                    &sql,
                    params![
                        id,
                        record.kind,
                        record.key,
                        record.scope.kind().as_str(),
                        int(record.scope.owner_id())?,
                        int(record.created_at.0)?,
                        record.retired_at.map(|seq| int(seq.0)).transpose()?,
                        encode(&record)?,
                    ],
                )?;
            }
            Target::Existing(_) if action.retire => {
                let sql = format!(
                    "UPDATE {} SET retired_at = ?, record = ? WHERE id = ?",
                    self.schema.documents
                );
                self.execute(&sql, params![int(self.seq.0)?, encode(&record)?, id])?;
            }
            Target::Existing(_) => {}
        }
        if let Some(content) = &action.content {
            if matches!(content, DocumentContent::Base { .. }) && !keeps_history {
                self.reclaim(id)?;
            }
            let (kind, version, content) = encode_content(content)?;
            let sql = format!(
                "INSERT INTO {} (document_id, seq, kind, version, content) VALUES (?, ?, ?, ?, ?)",
                self.schema.revisions
            );
            self.execute(&sql, params![id, int(self.seq.0)?, kind, version, content])?;
        }
        if action.retire && !keeps_history {
            self.reclaim(id)?;
        }
        Ok(())
    }

    fn reclaim(&self, id: i64) -> Result<(), StorageError> {
        let sql = format!("DELETE FROM {} WHERE document_id = ?", self.schema.revisions);
        self.execute(&sql, params![id])
    }
}
