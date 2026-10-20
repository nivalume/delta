//! The storage conformance suite: every case runs against the memory backend and the SQLite
//! backend (in memory, in a file, and in a file under a table prefix).
//!
//! Ported from upstream `src/testing/storage-conformance.ts`; the cases after
//! `rejects_after_close` are delta's additions (storage.md, "Conformance suite"). The cases
//! live in `tables`, `documents` and `general`; this file runs them and holds the helpers.

#![allow(clippy::unwrap_used)] // Case functions are not `#[test]` functions.

#[path = "../support/mod.rs"]
mod support;

mod documents;
mod general;
mod tables;

use delta_chord::delta::Op;
use delta_chord::json::{Json, Path};
use delta_harness::records::{
    BaseContent, ConversationId, ConversationOwner, ConversationParent, ConversationPolicy, ConversationQuery,
    ConversationRecord, Cursor, DocumentAddress, DocumentContent, DocumentCopySource, DocumentCreate, DocumentId,
    DocumentPoint, DocumentQuery, DocumentScope, EntryId, EntryQuery, EntryRecord, Fork, History, JoinPolicy,
    ROOT_CONVERSATION_ID, RawId, RecordId, ScanOrder, Seq, StorageWrite, StoredEntry, SubmissionId, SubmissionQuery,
    SubmissionRecord, SubmissionState, SubmissionStatus, SubmissionType, TaskId, TaskOutcome, TaskOutcomeError,
    TaskQuery, TaskState, TaskStatus,
};
use delta_harness::storage::{Storage, StorageError};
use serde_json::json;

use documents::*;
use general::*;
use support::{Backend, Fixture, commit, create_root, ctx, entry, ids, mint, pending_task, reject, value};
use tables::*;

macro_rules! conformance {
    ($($case:ident),* $(,)?) => {
        mod memory {
            $(#[tokio::test] async fn $case() { super::$case(super::Backend::Memory.open().await).await; })*
        }
        mod sqlite_in_memory {
            $(#[tokio::test] async fn $case() { super::$case(super::Backend::SqliteInMemory.open().await).await; })*
        }
        mod sqlite_file {
            $(#[tokio::test] async fn $case() { super::$case(super::Backend::SqliteFile.open().await).await; })*
        }
        mod sqlite_prefixed {
            $(#[tokio::test] async fn $case() { super::$case(super::Backend::SqlitePrefixed.open().await).await; })*
        }
    };
}

conformance!(
    reserves_root_id,
    mixed_writes_are_atomic,
    returned_records_are_owned,
    prototype_like_keys_round_trip,
    entries_out_of_id_order,
    entry_cursor_after_newer_commit,
    conversation_pages,
    conversation_owner_filters,
    deep_fork_history,
    scan_orders_and_cursors,
    task_replacement_and_filters,
    task_owner_and_status_scans,
    sleeping_tasks,
    request_ids_and_submission_replacement,
    write_submissions,
    rewindable_documents,
    long_delta_tails,
    copies_and_ambiguous_sources,
    version_transitions_and_current_only_history,
    addresses_and_scope_scans,
    document_lifecycle_and_empty_lifetimes,
    document_failures_roll_back_tables,
    indexed_strings_lossless,
    global_id_namespace,
    rejects_after_close,
    copy_source_rules,
    singleton_and_empty_key,
    duplicate_request_id_rejected,
    invalid_reads_are_requests,
);

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn conversation_scope(conversation_id: ConversationId) -> DocumentScope {
    DocumentScope::Conversation { conversation_id }
}

fn policy(history: History, fork: Fork) -> ConversationPolicy {
    ConversationPolicy { history, fork }
}

fn create(
    id: DocumentId,
    kind: &str,
    scope: DocumentScope,
    policy: impl Into<Option<ConversationPolicy>>,
) -> DocumentCreate {
    DocumentCreate {
        id,
        kind: kind.to_owned(),
        key: None,
        scope,
        policy: policy.into(),
    }
}

fn create_write(record: DocumentCreate, version: u32, value: Json) -> StorageWrite {
    StorageWrite::DocumentCreate {
        record,
        content: BaseContent { version, value },
    }
}

fn delta(id: DocumentId, version: u32, ops: serde_json::Value) -> StorageWrite {
    StorageWrite::DocumentChange {
        id,
        content: DocumentContent::Delta {
            version,
            ops: serde_json::from_value(ops).unwrap(),
        },
    }
}

fn base(id: DocumentId, version: u32, value: Json) -> StorageWrite {
    StorageWrite::DocumentChange {
        id,
        content: DocumentContent::Base { version, value },
    }
}

fn copy(record: DocumentCreate, id: DocumentId, at: DocumentPoint) -> StorageWrite {
    StorageWrite::DocumentCopy {
        record,
        source: DocumentCopySource { id, at },
    }
}

fn input(id: SubmissionId, conversation_id: ConversationId, request_id: Option<&str>) -> SubmissionRecord {
    SubmissionRecord {
        id,
        conversation_id,
        request_id: request_id.map(str::to_owned),
        kind: SubmissionType::Input,
        state: SubmissionState::Queued,
    }
}

async fn document_value(storage: &dyn Storage, id: DocumentId, at: DocumentPoint) -> Option<Json> {
    storage
        .document(id, at, &ctx())
        .await
        .unwrap()
        .map(|stored| stored.value)
}

async fn entry_ids(
    storage: &dyn Storage,
    query: EntryQuery,
    limit: usize,
    cursor: Option<Cursor>,
) -> (Vec<EntryId>, Option<Cursor>) {
    let page = storage.scan_entries(query, limit, cursor, &ctx()).await.unwrap();
    (ids(&page, |entry: &EntryRecord| entry.id), page.next)
}

async fn scan_status(storage: &dyn Storage, status: TaskStatus) -> Vec<delta_harness::records::TaskRecord> {
    let query = TaskQuery {
        status: Some(status),
        ..TaskQuery::default()
    };
    storage.scan_tasks(query, 10, None, &ctx()).await.unwrap().items
}
