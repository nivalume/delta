//! The SQLite schema (version 1) and its rendering for a table prefix.

use delta_chord::sqlite::{Migration, validate_schema_prefix};

use crate::storage::error::StorageError;

/// The schema version [`migrations`] produces.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// Table names of one schema: every table and index is `{prefix}{name}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schema {
    prefix: String,
    pub(super) metadata: String,
    pub(super) record_ids: String,
    pub(super) conversations: String,
    pub(super) entries: String,
    pub(super) tasks: String,
    pub(super) submissions: String,
    pub(super) documents: String,
    pub(super) revisions: String,
    /// chord's schema-version table.
    pub(super) version_table: String,
}

impl Schema {
    /// The schema under `prefix`, which must pass
    /// [`validate_schema_prefix`](delta_chord::sqlite::validate_schema_prefix); `None` means
    /// no prefix.
    pub fn new(prefix: Option<&str>) -> Result<Self, StorageError> {
        let prefix = match prefix {
            Some(prefix) => {
                validate_schema_prefix(prefix)?;
                prefix
            }
            None => "",
        };
        let name = |table: &str| format!("{prefix}{table}");
        Ok(Self {
            prefix: prefix.to_owned(),
            metadata: name("metadata"),
            record_ids: name("record_ids"),
            conversations: name("conversations"),
            entries: name("entries"),
            tasks: name("tasks"),
            submissions: name("submissions"),
            documents: name("documents"),
            revisions: name("document_revisions"),
            version_table: name("schema"),
        })
    }

    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

/// The schema's migrations, rendered for `schema`'s prefix. Later versions append here;
/// version 1 has no predecessor.
pub fn migrations(schema: &Schema) -> Vec<Migration> {
    vec![Migration::new(1, [V1.replace("{p}", &schema.prefix)])]
}

/// Version 1. Every record keeps its JSON in `record`; the other columns are the indexed or
/// joined fields copied out of it, so reads decode `record` alone.
const V1: &str = "
CREATE TABLE {p}metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    next_id INTEGER NOT NULL CHECK (next_id >= 2),
    next_seq INTEGER NOT NULL CHECK (next_seq >= 1)
) STRICT;
INSERT INTO {p}metadata (singleton, next_id, next_seq) VALUES (1, 2, 1);

CREATE TABLE {p}record_ids (
    id INTEGER PRIMARY KEY,
    record_type TEXT NOT NULL
        CHECK (record_type IN ('conversation', 'entry', 'task', 'submission', 'document'))
) STRICT;

CREATE TABLE {p}conversations (
    id INTEGER PRIMARY KEY,
    parent_id INTEGER,
    parent_at INTEGER,
    owner_conversation_id INTEGER,
    owner_task_id INTEGER,
    record TEXT NOT NULL CHECK (json_valid(record)),
    CHECK ((parent_id IS NULL) = (parent_at IS NULL)),
    CHECK ((owner_conversation_id IS NULL) = (owner_task_id IS NULL))
) STRICT;
CREATE INDEX {p}conversations_by_owner_conversation ON {p}conversations (owner_conversation_id, id);
CREATE INDEX {p}conversations_by_owner_task ON {p}conversations (owner_task_id, id);

CREATE TABLE {p}entries (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL,
    head INTEGER,
    commit_seq INTEGER NOT NULL,
    record TEXT NOT NULL CHECK (json_valid(record))
) STRICT;
CREATE INDEX {p}entries_by_conversation ON {p}entries (conversation_id, id);
CREATE INDEX {p}entry_heads_by_conversation ON {p}entries (conversation_id, id) WHERE head IS NOT NULL;

CREATE TABLE {p}tasks (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('pending', 'running', 'waiting', 'sleeping', 'completing', 'terminal')),
    abort_requested INTEGER NOT NULL CHECK (abort_requested IN (0, 1)),
    background INTEGER NOT NULL CHECK (background IN (0, 1)),
    record TEXT NOT NULL CHECK (json_valid(record))
) STRICT;
CREATE INDEX {p}tasks_by_status ON {p}tasks (status, id);
CREATE INDEX {p}tasks_by_conversation ON {p}tasks (conversation_id, id);
CREATE INDEX {p}tasks_by_kind ON {p}tasks (kind, id);
CREATE INDEX {p}tasks_by_abort_requested ON {p}tasks (abort_requested, id);
CREATE INDEX {p}tasks_by_background ON {p}tasks (background, id);

CREATE TABLE {p}submissions (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL,
    request_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('queued', 'placed', 'done', 'unanswered')),
    record TEXT NOT NULL CHECK (json_valid(record))
) STRICT;
CREATE UNIQUE INDEX {p}submissions_by_request
    ON {p}submissions (conversation_id, request_id) WHERE request_id IS NOT NULL;
CREATE INDEX {p}submissions_by_conversation ON {p}submissions (conversation_id, id);
CREATE INDEX {p}submissions_by_status ON {p}submissions (status, id);

CREATE TABLE {p}documents (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    key TEXT,
    scope_kind TEXT NOT NULL CHECK (scope_kind IN ('session', 'conversation', 'task')),
    owner_id INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    retired_at INTEGER,
    record TEXT NOT NULL CHECK (json_valid(record)),
    CHECK (retired_at IS NULL OR retired_at >= created_at)
) STRICT;
CREATE INDEX {p}documents_by_address
    ON {p}documents (kind, scope_kind, owner_id, key, created_at);
CREATE INDEX {p}documents_by_scope ON {p}documents (scope_kind, owner_id, id);
CREATE INDEX {p}documents_by_scope_kind ON {p}documents (scope_kind, owner_id, kind, id);

CREATE TABLE {p}document_revisions (
    document_id INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('base', 'delta')),
    version INTEGER NOT NULL CHECK (version >= 1),
    content TEXT NOT NULL CHECK (json_valid(content)),
    PRIMARY KEY (document_id, seq)
) STRICT, WITHOUT ROWID;
CREATE INDEX {p}document_bases ON {p}document_revisions (document_id, seq) WHERE kind = 'base';
";
