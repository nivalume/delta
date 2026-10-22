//! Commit planning, shared by the backends: id ownership, request-id uniqueness, document
//! command normalization, copy resolution and incarnation checks (storage.md, "Commit
//! rules"). A backend supplies a read-only [`CommitView`] of its committed state, plans, and
//! applies the [`Batch`] only when planning succeeded, so a refused batch has no effect.

use std::collections::{HashMap, HashSet};
use std::fmt;

use indexmap::IndexMap;

use super::error::StorageError;
use crate::records::{
    ConversationId, ConversationRecord, DocumentAddress, DocumentContent, DocumentCopySource, DocumentCreate,
    DocumentId, DocumentRecord, EntryRecord, Seq, StorageWrite, StoredDocument, SubmissionId, SubmissionRecord,
    TaskRecord,
};

/// The table that owns an id in the global namespace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RecordType {
    Conversation,
    Entry,
    Task,
    Submission,
    Document,
}

impl RecordType {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Conversation => "conversation",
            Self::Entry => "entry",
            Self::Task => "task",
            Self::Submission => "submission",
            Self::Document => "document",
        }
    }

    pub(crate) fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "conversation" => Self::Conversation,
            "entry" => Self::Entry,
            "task" => Self::Task,
            "submission" => Self::Submission,
            "document" => Self::Document,
            _ => return None,
        })
    }

    /// Whether a write of this type creates (and so must claim a fresh id) rather than
    /// replaces.
    fn creates(self) -> bool {
        matches!(self, Self::Conversation | Self::Entry | Self::Document)
    }
}

impl fmt::Display for RecordType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An existing incarnation with the version of its newest stored revision.
#[derive(Clone, Debug)]
pub(crate) struct ExistingDocument {
    pub(crate) record: DocumentRecord,
    /// `None` when no revision is stored (a retired current-only incarnation).
    pub(crate) newest_version: Option<u32>,
}

/// The committed state a batch is planned against.
pub(crate) trait CommitView {
    fn record_type(&mut self, id: u64) -> Result<Option<RecordType>, StorageError>;
    fn document(&mut self, id: DocumentId) -> Result<Option<ExistingDocument>, StorageError>;
    fn current_document(&mut self, address: &DocumentAddress) -> Result<Option<DocumentId>, StorageError>;
    /// `Storage::document(source.id, source.at)` on committed state.
    fn materialize(&mut self, source: DocumentCopySource) -> Result<Option<StoredDocument>, StorageError>;
    fn submission_by_request(
        &mut self,
        conversation: ConversationId,
        request_id: &str,
    ) -> Result<Option<SubmissionId>, StorageError>;
}

/// A planned batch: record writes in batch order, then document actions in order of first
/// appearance.
#[derive(Debug)]
pub(crate) struct Batch {
    pub(crate) records: Vec<RecordWrite>,
    pub(crate) documents: Vec<DocumentAction>,
    /// The highest id the batch writes, for `next_id`.
    pub(crate) highest_id: Option<u64>,
}

/// A table write.
#[derive(Debug)]
pub(crate) enum RecordWrite {
    Conversation(ConversationRecord),
    Entry(EntryRecord),
    Task(TaskRecord),
    Submission(SubmissionRecord),
}

/// Every command of one batch on one incarnation, normalized: content before retirement.
#[derive(Debug)]
pub(crate) struct DocumentAction {
    pub(crate) id: DocumentId,
    pub(crate) target: Target,
    /// A copy's content is the source's resolved base.
    pub(crate) content: Option<DocumentContent>,
    pub(crate) retire: bool,
}

/// The incarnation an action applies to.
#[derive(Debug)]
pub(crate) enum Target {
    /// Created by this batch.
    Created(DocumentCreate),
    /// Committed before; its record as stored.
    Existing(DocumentRecord),
}

impl DocumentAction {
    /// The incarnation's record after a commit at `seq`.
    pub(crate) fn record_at(&self, seq: Seq) -> DocumentRecord {
        let mut record = match &self.target {
            Target::Created(create) => DocumentRecord::created(create.clone(), seq),
            Target::Existing(existing) => existing.clone(),
        };
        if self.retire {
            record.retired_at = Some(seq);
        }
        record
    }
}

/// An action before its incarnation is checked against committed state.
struct Draft {
    id: DocumentId,
    create: Option<DocumentCreate>,
    content: Option<DocumentContent>,
    retire: bool,
}

enum DocumentCommand {
    Content {
        id: DocumentId,
        content: DocumentContent,
        create: Option<DocumentCreate>,
    },
    Copy {
        create: DocumentCreate,
        source: DocumentCopySource,
    },
    Retire {
        id: DocumentId,
    },
}

fn rejected(message: impl Into<String>) -> StorageError {
    StorageError::Rejected(message.into())
}

/// Plans `writes` against `view`; any rule violation is [`StorageError::Rejected`].
pub(crate) fn plan(writes: Vec<StorageWrite>, view: &mut impl CommitView) -> Result<Batch, StorageError> {
    let mut claims = Claims::default();
    let mut requests = HashMap::<(ConversationId, String), SubmissionId>::new();
    let mut records = Vec::new();
    let mut commands = Vec::new();
    for write in writes {
        match write {
            StorageWrite::Conversation(record) => {
                claims.claim(view, record.id.0, RecordType::Conversation)?;
                records.push(RecordWrite::Conversation(record));
            }
            StorageWrite::Entry(record) => {
                claims.claim(view, record.id.0, RecordType::Entry)?;
                records.push(RecordWrite::Entry(record));
            }
            StorageWrite::Task(record) => {
                claims.claim(view, record.id.0, RecordType::Task)?;
                records.push(RecordWrite::Task(record));
            }
            StorageWrite::Submission(record) => {
                claims.claim(view, record.id.0, RecordType::Submission)?;
                check_request(view, &mut requests, &record)?;
                records.push(RecordWrite::Submission(record));
            }
            StorageWrite::DocumentCreate { record, content } => {
                claims.claim(view, record.id.0, RecordType::Document)?;
                record.check().map_err(|error| rejected(error.to_string()))?;
                commands.push(DocumentCommand::Content {
                    id: record.id,
                    content: content.into(),
                    create: Some(record),
                });
            }
            StorageWrite::DocumentCopy { record, source } => {
                claims.claim(view, record.id.0, RecordType::Document)?;
                record.check().map_err(|error| rejected(error.to_string()))?;
                commands.push(DocumentCommand::Copy { create: record, source });
            }
            StorageWrite::DocumentChange { id, content } => {
                commands.push(DocumentCommand::Content {
                    id,
                    content,
                    create: None,
                });
            }
            StorageWrite::DocumentRetire { id } => commands.push(DocumentCommand::Retire { id }),
        }
    }
    let documents = document_actions(commands, view)?;
    Ok(Batch {
        records,
        documents,
        highest_id: claims.highest,
    })
}

/// Id claims of one batch (commit rule 2).
#[derive(Default)]
struct Claims {
    claimed: HashMap<u64, RecordType>,
    highest: Option<u64>,
}

impl Claims {
    fn claim(&mut self, view: &mut impl CommitView, id: u64, table: RecordType) -> Result<(), StorageError> {
        if let Some(existing) = view.record_type(id)?
            && (table.creates() || existing != table)
        {
            return Err(rejected(format!("ID {id} already belongs to {existing}")));
        }
        if let Some(&earlier) = self.claimed.get(&id) {
            if table.creates() || earlier.creates() {
                return Err(rejected(format!("ID {id} is written more than once")));
            }
            if earlier != table {
                return Err(rejected(format!("ID {id} is written as two record types")));
            }
        }
        self.claimed.insert(id, table);
        self.highest = Some(self.highest.map_or(id, |highest| highest.max(id)));
        Ok(())
    }
}

/// A request id belongs to one submission per conversation.
fn check_request(
    view: &mut impl CommitView,
    requests: &mut HashMap<(ConversationId, String), SubmissionId>,
    record: &SubmissionRecord,
) -> Result<(), StorageError> {
    let Some(request_id) = &record.request_id else {
        return Ok(());
    };
    let key = (record.conversation_id, request_id.clone());
    let owner = match requests.get(&key) {
        Some(&owner) => Some(owner),
        None => view.submission_by_request(record.conversation_id, request_id)?,
    };
    if let Some(owner) = owner
        && owner != record.id
    {
        return Err(rejected(format!(
            "Request ID {request_id:?} already belongs to submission {owner} in conversation {}",
            record.conversation_id
        )));
    }
    requests.insert(key, record.id);
    Ok(())
}

/// Normalizes document commands per incarnation, resolves copies and checks every action
/// (commit rules 3 and 4).
fn document_actions(
    commands: Vec<DocumentCommand>,
    view: &mut impl CommitView,
) -> Result<Vec<DocumentAction>, StorageError> {
    let targets: HashSet<DocumentId> = commands
        .iter()
        .map(|command| match command {
            DocumentCommand::Content { id, .. } | DocumentCommand::Retire { id } => *id,
            DocumentCommand::Copy { create, .. } => create.id,
        })
        .collect();
    let mut drafts = IndexMap::<DocumentId, Draft>::new();
    for command in commands {
        let (id, create, content, retire) = match command {
            DocumentCommand::Content { id, content, create } => (id, create, Some(content), false),
            DocumentCommand::Copy { create, source } => {
                let content = resolve_copy(view, &create, source, &targets)?;
                (create.id, Some(create), Some(content), false)
            }
            DocumentCommand::Retire { id } => (id, None, None, true),
        };
        let action = drafts.entry(id).or_insert_with(|| Draft {
            id,
            create: None,
            content: None,
            retire: false,
        });
        if let Some(content) = content {
            if action.content.is_some() {
                return Err(rejected(format!("Document {id} has more than one content command")));
            }
            if content.version() == 0 {
                return Err(rejected(format!("Document {id} content version must be positive")));
            }
            action.content = Some(content);
            action.create = create;
        }
        if retire {
            if action.retire {
                return Err(rejected(format!("Document {id} is retired more than once")));
            }
            action.retire = true;
        }
    }
    check_actions(drafts.into_values(), view)
}

/// The child base of a copy: the source's value and version from committed state.
fn resolve_copy(
    view: &mut impl CommitView,
    create: &DocumentCreate,
    source: DocumentCopySource,
    targets: &HashSet<DocumentId>,
) -> Result<DocumentContent, StorageError> {
    let reject = |cause: String| rejected(format!("Document copy {} was rejected: {cause}", create.id));
    if targets.contains(&source.id) {
        return Err(reject(format!(
            "source document {} is changed in the copy batch",
            source.id
        )));
    }
    let stored = match view.materialize(source) {
        Ok(Some(stored)) => stored,
        Ok(None) => return Err(reject(format!("source document {} cannot be read", source.id))),
        Err(StorageError::Request(cause) | StorageError::Corrupt(cause)) => return Err(reject(cause)),
        Err(error) => return Err(error),
    };
    let record = &stored.record;
    let matches = record.scope.kind() == create.scope.kind()
        && create.policy.is_some()
        && record.kind == create.kind
        && record.key == create.key
        && record.policy == create.policy;
    if !matches {
        return Err(reject(format!(
            "source document {} does not match the copied record",
            source.id
        )));
    }
    Ok(DocumentContent::Base {
        version: stored.version,
        value: stored.value,
    })
}

/// Incarnation existence, delta versions and one current incarnation per address, counted
/// as current (0 or 1) − retired + created after the whole batch.
fn check_actions(
    drafts: impl Iterator<Item = Draft>,
    view: &mut impl CommitView,
) -> Result<Vec<DocumentAction>, StorageError> {
    let mut live = HashMap::<DocumentAddress, i64>::new();
    let mut actions = Vec::new();
    for draft in drafts {
        let id = draft.id;
        let existing = view.document(id)?;
        let target = match (draft.create, existing) {
            (None, None) => return Err(rejected(format!("Document {id} does not exist"))),
            (Some(_), Some(_)) => return Err(rejected(format!("Document {id} already exists"))),
            (Some(create), None) => Target::Created(create),
            (None, Some(existing)) => {
                if existing.record.retired_at.is_some() {
                    return Err(rejected(format!("Document {id} is retired")));
                }
                if let Some(DocumentContent::Delta { version, .. }) = &draft.content {
                    match existing.newest_version {
                        None => return Err(rejected(format!("Document {id} delta has no base"))),
                        Some(newest) if newest != *version => {
                            return Err(rejected(format!("Document {id} version transition requires a base")));
                        }
                        Some(_) => {}
                    }
                }
                Target::Existing(existing.record)
            }
        };
        let address = match &target {
            Target::Created(create) => create.address(),
            Target::Existing(record) => record.address(),
        };
        let count = match live.get(&address) {
            Some(&count) => count,
            None => i64::from(view.current_document(&address)?.is_some()),
        };
        let change = match (&target, draft.retire) {
            (Target::Existing(_), true) => -1,
            (Target::Created(_), false) => 1,
            _ => 0,
        };
        live.insert(address, count + change);
        actions.push(DocumentAction {
            id,
            target,
            content: draft.content,
            retire: draft.retire,
        });
    }
    if live.values().any(|&count| count > 1) {
        return Err(rejected("Document address already has a current incarnation"));
    }
    Ok(actions)
}
