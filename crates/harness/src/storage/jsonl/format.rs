//! The JSONL backend's lines: one commit marker per commit in `main.jsonl`, and the records
//! the marker confirms in the sidecars, `doc-<id>.jsonl` and `task-<id>.jsonl`.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::records::{
    ConversationRecord, DocumentContent, DocumentCreate, DocumentId, DocumentScope, EntryRecord, History, Seq,
    SubmissionRecord, TaskId, TaskRecord, TaskStatus,
};
use crate::storage::batch::{Batch, RecordWrite, Target};
use crate::storage::error::StorageError;

/// The line format this crate writes and reads.
pub const JSONL_FORMAT_VERSION: u32 = 1;

pub(super) const MAIN_FILE: &str = "main.jsonl";
pub(super) const RECLAIM_SUFFIX: &str = ".reclaim";
const COMMIT: &str = "commit";
const RECORD: &str = "record";

/// One write of a commit marker. Table records travel in the marker itself, except live
/// tasks, which go to their sidecar; document contents go to the incarnation's sidecar and
/// the marker names them by ordinal. Same tagging as `StorageWrite`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub(super) enum MainWrite {
    Conversation(ConversationRecord),
    Entry(EntryRecord),
    /// A terminal task; a live one is a `TaskSidecar`.
    Task(TaskRecord),
    Submission(SubmissionRecord),
    TaskSidecar {
        id: TaskId,
        ordinal: u32,
    },
    DocumentCreate {
        record: DocumentCreate,
        ordinal: u32,
    },
    DocumentChange {
        id: DocumentId,
        ordinal: u32,
    },
    DocumentRetire {
        id: DocumentId,
    },
}

/// One line of `main.jsonl`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct Marker {
    pub(super) format: u32,
    #[serde(rename = "type")]
    pub(super) kind: String,
    pub(super) seq: Seq,
    pub(super) writes: Vec<MainWrite>,
}

/// What a sidecar record holds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum Payload {
    /// A live task's whole record.
    Task { value: TaskRecord },
    /// A base or delta of one incarnation.
    Document { id: DocumentId, content: DocumentContent },
}

/// One line of a sidecar.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct SidecarRecord {
    pub(super) format: u32,
    #[serde(rename = "type")]
    pub(super) kind: String,
    pub(super) seq: Seq,
    /// The record's place among every sidecar record of its commit.
    pub(super) ordinal: u32,
    pub(super) payload: Payload,
}

/// A sidecar file's name.
pub(super) fn task_file(id: TaskId) -> String {
    format!("task-{id}.jsonl")
}

pub(super) fn doc_file(id: DocumentId) -> String {
    format!("doc-{id}.jsonl")
}

/// The id of a sidecar named `doc-<id>.jsonl` or `task-<id>.jsonl` (canonical digits).
pub(super) fn sidecar_id(name: &str) -> Option<(SidecarKind, u64)> {
    let (kind, rest) = match name.strip_prefix("doc-") {
        Some(rest) => (SidecarKind::Document, rest),
        None => (SidecarKind::Task, name.strip_prefix("task-")?),
    };
    let digits = rest.strip_suffix(".jsonl")?;
    let canonical = !digits.is_empty()
        && digits.bytes().all(|byte| byte.is_ascii_digit())
        && (digits == "0" || !digits.starts_with('0'));
    canonical.then(|| digits.parse().ok()).flatten().map(|id| (kind, id))
}

/// Whether `name` is a reclamation's temporary file.
pub(super) fn is_reclaim_file(name: &str) -> bool {
    name.strip_suffix(RECLAIM_SUFFIX)
        .is_some_and(|name| sidecar_id(name).is_some())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SidecarKind {
    Document,
    Task,
}

/// Whether an incarnation keeps only its current value: every document but a rewindable
/// conversation document.
pub(super) fn current_only(create: &DocumentCreate) -> bool {
    !(matches!(create.scope, DocumentScope::Conversation { .. })
        && create
            .policy
            .is_some_and(|policy| policy.history == History::Rewindable))
}

/// A commit as lines: the marker, and each affected sidecar's records in ordinal order,
/// the sidecars in the order of their first record.
#[derive(Debug)]
pub(super) struct EncodedCommit {
    pub(super) marker: Vec<u8>,
    pub(super) sidecars: IndexMap<String, Vec<u8>>,
}

/// One JSON line.
pub(super) fn line(value: &impl Serialize) -> Result<Vec<u8>, StorageError> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Encodes a planned batch committed at `seq`. Fails only when a record does not encode,
/// before anything is written.
pub(super) fn encode(seq: Seq, batch: &Batch) -> Result<EncodedCommit, StorageError> {
    let mut writes = Vec::new();
    let mut sidecars = IndexMap::<String, Vec<u8>>::new();
    let mut ordinal = 0_u32;
    let mut add = |file: String, payload: Payload| -> Result<u32, StorageError> {
        let this = ordinal;
        ordinal += 1;
        let record = SidecarRecord {
            format: JSONL_FORMAT_VERSION,
            kind: RECORD.to_owned(),
            seq,
            ordinal: this,
            payload,
        };
        sidecars.entry(file).or_default().extend(line(&record)?);
        Ok(this)
    };
    for write in &batch.records {
        writes.push(match write {
            RecordWrite::Conversation(record) => MainWrite::Conversation(record.clone()),
            RecordWrite::Entry(record) => MainWrite::Entry(record.clone()),
            RecordWrite::Submission(record) => MainWrite::Submission(record.clone()),
            RecordWrite::Task(task) if task.state.status() == TaskStatus::Terminal => MainWrite::Task(task.clone()),
            RecordWrite::Task(task) => {
                let ordinal = add(task_file(task.id), Payload::Task { value: task.clone() })?;
                MainWrite::TaskSidecar { id: task.id, ordinal }
            }
        });
    }
    for action in &batch.documents {
        if let Some(content) = &action.content {
            let ordinal = add(
                doc_file(action.id),
                Payload::Document {
                    id: action.id,
                    content: content.clone(),
                },
            )?;
            writes.push(match &action.target {
                Target::Created(record) => MainWrite::DocumentCreate {
                    record: record.clone(),
                    ordinal,
                },
                Target::Existing(_) => MainWrite::DocumentChange { id: action.id, ordinal },
            });
        }
        if action.retire {
            writes.push(MainWrite::DocumentRetire { id: action.id });
        }
    }
    let marker = Marker {
        format: JSONL_FORMAT_VERSION,
        kind: COMMIT.to_owned(),
        seq,
        writes,
    };
    Ok(EncodedCommit {
        marker: line(&marker)?,
        sidecars,
    })
}

fn corrupt(message: String) -> StorageError {
    StorageError::Corrupt(message)
}

/// A complete line as JSON; a line that is not JSON is malformed.
fn parse_json(text: &str, name: &str, number: usize) -> Result<Value, StorageError> {
    serde_json::from_str(text).map_err(|error| corrupt(format!("Malformed complete {name} line {number}: {error}")))
}

/// Parses and checks a complete `main.jsonl` line.
pub(super) fn parse_marker(text: &str, number: usize) -> Result<Marker, StorageError> {
    let value = parse_json(text, MAIN_FILE, number)?;
    let marker: Marker = serde_json::from_value(value)
        .map_err(|error| corrupt(format!("Invalid commit marker in {MAIN_FILE} line {number}: {error}")))?;
    if marker.format != JSONL_FORMAT_VERSION || marker.kind != COMMIT || marker.seq.0 < 1 {
        return Err(corrupt(format!("Invalid commit marker in {MAIN_FILE} line {number}")));
    }
    for write in &marker.writes {
        let terminal_ok = match write {
            MainWrite::Task(task) => task.state.status() == TaskStatus::Terminal,
            _ => true,
        };
        if !terminal_ok {
            return Err(corrupt(format!(
                "Invalid terminal task write in {MAIN_FILE} line {number}"
            )));
        }
    }
    Ok(marker)
}

/// Parses and checks a complete sidecar line.
pub(super) fn parse_sidecar_record(text: &str, file: &str, number: usize) -> Result<SidecarRecord, StorageError> {
    let value = parse_json(text, file, number)?;
    let content = &value["payload"]["content"];
    if value["payload"]["type"] == "document"
        && let Err(error) = serde_json::from_value::<DocumentContent>(content.clone())
    {
        return Err(corrupt(format!(
            "Invalid document content in {file} line {number}: {error}"
        )));
    }
    let record: SidecarRecord = serde_json::from_value(value)
        .map_err(|error| corrupt(format!("Invalid sidecar record in {file} line {number}: {error}")))?;
    if record.format != JSONL_FORMAT_VERSION || record.kind != RECORD || record.seq.0 < 1 {
        return Err(corrupt(format!("Invalid sidecar record in {file} line {number}")));
    }
    match &record.payload {
        Payload::Task { value } if value.state.status() == TaskStatus::Terminal => {
            Err(corrupt(format!("Invalid live task record in {file} line {number}")))
        }
        Payload::Document { content, .. } if content.version() == 0 => {
            Err(corrupt(format!("Invalid document content in {file} line {number}")))
        }
        _ => Ok(record),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_names_are_canonical() {
        assert_eq!(sidecar_id("doc-12.jsonl"), Some((SidecarKind::Document, 12)));
        assert_eq!(sidecar_id("task-0.jsonl"), Some((SidecarKind::Task, 0)));
        for name in [
            "doc-012.jsonl",
            "doc-.jsonl",
            "doc-1a.jsonl",
            "main.jsonl",
            "doc-1.json",
            "x-1.jsonl",
        ] {
            assert_eq!(sidecar_id(name), None, "{name}");
        }
        assert!(is_reclaim_file("doc-3.jsonl.reclaim"));
        assert!(!is_reclaim_file("main.jsonl.reclaim"));
    }

    #[test]
    fn markers_and_records_round_trip_and_are_checked() {
        let marker = Marker {
            format: JSONL_FORMAT_VERSION,
            kind: COMMIT.to_owned(),
            seq: Seq(3),
            writes: vec![
                MainWrite::DocumentRetire { id: DocumentId(9) },
                MainWrite::TaskSidecar {
                    id: TaskId(4),
                    ordinal: 0,
                },
            ],
        };
        let text = String::from_utf8(line(&marker).unwrap()).unwrap();
        assert_eq!(
            text,
            "{\"format\":1,\"type\":\"commit\",\"seq\":3,\"writes\":[{\"type\":\"document_retire\",\"value\":{\"id\":9}},{\"type\":\"task_sidecar\",\"value\":{\"id\":4,\"ordinal\":0}}]}\n"
        );
        assert_eq!(parse_marker(text.trim_end(), 1).unwrap(), marker);
        assert!(parse_marker(&text.replace("\"format\":1", "\"format\":2"), 1).is_err());
        assert!(parse_marker(&text.replace("\"seq\":3", "\"seq\":0"), 1).is_err());
        assert!(
            parse_marker("{\"format\":1", 1)
                .unwrap_err()
                .to_string()
                .starts_with("Malformed complete main.jsonl line 1")
        );

        let record = SidecarRecord {
            format: JSONL_FORMAT_VERSION,
            kind: RECORD.to_owned(),
            seq: Seq(3),
            ordinal: 1,
            payload: Payload::Document {
                id: DocumentId(9),
                content: DocumentContent::Base {
                    version: 1,
                    value: delta_chord::json::Json::from(serde_json::json!({ "a": 1 })),
                },
            },
        };
        let text = String::from_utf8(line(&record).unwrap()).unwrap();
        assert_eq!(parse_sidecar_record(text.trim_end(), "doc-9.jsonl", 1).unwrap(), record);
        assert!(parse_sidecar_record(&text.replace("\"version\":1", "\"version\":0"), "doc-9.jsonl", 1).is_err());
    }
}
