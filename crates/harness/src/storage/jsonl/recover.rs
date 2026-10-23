//! Opening a JSONL directory: what a crash may have left is removed, and the records every
//! marker confirms are replayed into the memory tables.
//!
//! - A final line without its line break is torn: the file is truncated before it.
//! - Sidecar records no marker confirms are a failed commit's: the tail they start is
//!   truncated, and a confirmed record after one is corruption.
//! - A confirmed record that is missing is corruption, unless a later base or retirement of
//!   a current-only incarnation, or a task's terminal record, made it reclaimable.
//! - Reclamations a crash interrupted are redone, and temporary replacements removed.

use std::collections::{BTreeMap, HashMap, HashSet};

use delta_chord::context::Context;
use delta_chord::json::Json;

use super::Files;
use super::format::{
    MAIN_FILE, MainWrite, Marker, Payload, SidecarKind, SidecarRecord, current_only, doc_file, is_reclaim_file,
    parse_marker, parse_sidecar_record, sidecar_id, task_file,
};
use crate::env::{FileKind, RemoveOptions};
use crate::records::{BaseContent, DocumentContent, DocumentId, Seq, StorageWrite, TaskId};
use crate::storage::error::StorageError;
use crate::storage::memory::MemoryStorage;

/// One complete line: its value, where it starts and its bytes with the line break.
struct Line<T> {
    value: T,
    start: u64,
    raw: Vec<u8>,
}

/// A sidecar record's identity: its file, commit and ordinal.
type Key = (String, Seq, u32);

fn corrupt(message: impl Into<String>) -> StorageError {
    StorageError::Corrupt(message.into())
}

/// Recovers `files` into `memory`; the tasks whose newest record is live.
pub(super) async fn recover(
    files: &Files,
    memory: &MemoryStorage,
    ctx: &Context,
) -> Result<HashSet<TaskId>, StorageError> {
    let main = read_lines(files, &files.main, MAIN_FILE, ctx, parse_marker).await?;
    let mut previous = 0;
    for line in &main {
        if line.value.seq.0 <= previous {
            return Err(corrupt(format!(
                "Commit sequence does not strictly increase in {MAIN_FILE}"
            )));
        }
        previous = line.value.seq.0;
    }

    let listed = files.fs.list_dir(&files.dir, ctx).await?;
    let remove = RemoveOptions {
        recursive: false,
        force: true,
    };
    for info in &listed {
        if info.kind == FileKind::File && is_reclaim_file(&info.name) {
            files.fs.remove(&info.path, remove, ctx).await?;
        }
    }
    let mut sidecars = BTreeMap::<String, (SidecarKind, u64, String, Vec<Line<SidecarRecord>>)>::new();
    let mut records = HashMap::<Key, SidecarRecord>::new();
    for info in listed.into_iter().filter(|info| info.kind == FileKind::File) {
        let Some((kind, id)) = sidecar_id(&info.name) else {
            continue;
        };
        let name = info.name.clone();
        let lines = read_lines(files, &info.path, &name, ctx, |text, number| {
            parse_sidecar_record(text, &name, number)
        })
        .await?;
        for pair in lines.windows(2) {
            let (before, after) = (&pair[0].value, &pair[1].value);
            if (after.seq, after.ordinal) <= (before.seq, before.ordinal) {
                return Err(corrupt(format!("Sidecar records are out of order in {name}")));
            }
        }
        for line in &lines {
            records.insert((name.clone(), line.value.seq, line.value.ordinal), line.value.clone());
        }
        sidecars.insert(name, (kind, id, info.path, lines));
    }

    let state = State::of(&main, &records);
    let mut confirmed = HashSet::<Key>::new();
    for line in &main {
        let marker = &line.value;
        let writes = state.writes(marker, &records, &mut confirmed)?;
        memory
            .replay(writes, marker.seq)
            .map_err(|error| corrupt(format!("Invalid committed state at sequence {}: {error}", marker.seq)))?;
    }

    let mut reclamations = BTreeMap::new();
    for (name, (kind, id, path, lines)) in &sidecars {
        let is_confirmed =
            |line: &Line<SidecarRecord>| confirmed.contains(&(name.clone(), line.value.seq, line.value.ordinal));
        let mut unconfirmed_at = None;
        for line in lines {
            if is_confirmed(line) {
                if unconfirmed_at.is_some() {
                    return Err(corrupt(format!(
                        "Confirmed record follows an unconfirmed tail in {name}"
                    )));
                }
            } else if unconfirmed_at.is_none() {
                unconfirmed_at = Some(line.start);
            }
        }
        if let Some(at) = unconfirmed_at {
            files.fs.truncate_file(path, at, ctx).await?;
        }
        let kept: Vec<&Line<SidecarRecord>> = lines.iter().filter(|line| is_confirmed(line)).collect();
        let retained: Option<Vec<&Line<SidecarRecord>>> = match kind {
            SidecarKind::Task => state.terminal_tasks.contains(&TaskId(*id)).then(Vec::new),
            SidecarKind::Document => {
                let document = DocumentId(*id);
                if state.retired_current_only.contains(&document) {
                    Some(Vec::new())
                } else if state.latest_bases.contains_key(&document) {
                    Some(
                        kept.iter()
                            .copied()
                            .filter(|line| !state.before_latest_base(document, line.value.seq, line.value.ordinal))
                            .collect(),
                    )
                } else {
                    None
                }
            }
        };
        if let Some(retained) = retained
            && (retained.len() < kept.len() || retained.is_empty())
        {
            let content: Vec<u8> = retained.iter().flat_map(|line| line.raw.iter().copied()).collect();
            reclamations.insert(name.clone(), content);
        }
    }
    files.reclaim(reclamations).await;
    Ok(state.live_tasks)
}

/// What the markers say about the sidecars.
#[derive(Default)]
struct State {
    current_only: HashSet<DocumentId>,
    retired_current_only: HashSet<DocumentId>,
    /// The newest confirmed base of each current-only incarnation, as (commit, ordinal).
    latest_bases: HashMap<DocumentId, (Seq, u32)>,
    terminal_tasks: HashSet<TaskId>,
    live_tasks: HashSet<TaskId>,
}

impl State {
    fn of(main: &[Line<Marker>], records: &HashMap<Key, SidecarRecord>) -> Self {
        let mut state = Self::default();
        let mut retired = HashSet::new();
        let mut final_tasks = HashMap::<TaskId, bool>::new();
        for line in main {
            for write in &line.value.writes {
                match write {
                    MainWrite::DocumentCreate { record, .. } if current_only(record) => {
                        state.current_only.insert(record.id);
                    }
                    MainWrite::DocumentRetire { id } => {
                        retired.insert(*id);
                    }
                    MainWrite::Task(task) => {
                        final_tasks.insert(task.id, false);
                    }
                    MainWrite::TaskSidecar { id, .. } => {
                        final_tasks.insert(*id, true);
                    }
                    _ => {}
                }
            }
        }
        state.retired_current_only = retired.intersection(&state.current_only).copied().collect();
        for (id, live) in final_tasks {
            if live {
                state.live_tasks.insert(id);
            } else {
                state.terminal_tasks.insert(id);
            }
        }
        for line in main {
            let marker = &line.value;
            for write in &marker.writes {
                let (id, ordinal) = match write {
                    MainWrite::DocumentCreate { record, ordinal } => (record.id, *ordinal),
                    MainWrite::DocumentChange { id, ordinal } => (*id, *ordinal),
                    _ => continue,
                };
                if !state.current_only.contains(&id) {
                    continue;
                }
                let is_base = records.get(&(doc_file(id), marker.seq, ordinal)).is_some_and(|record| {
                    matches!(&record.payload, Payload::Document { id: of, content: DocumentContent::Base { .. } } if *of == id)
                });
                if is_base {
                    let at = (marker.seq, ordinal);
                    let newest = state.latest_bases.entry(id).or_insert(at);
                    *newest = (*newest).max(at);
                }
            }
        }
        state
    }

    /// Whether a record of `id` precedes its newest base, which made it reclaimable.
    fn before_latest_base(&self, id: DocumentId, seq: Seq, ordinal: u32) -> bool {
        self.latest_bases.get(&id).is_some_and(|base| (seq, ordinal) < *base)
    }

    /// The writes one marker committed, with the sidecar records it confirms.
    fn writes(
        &self,
        marker: &Marker,
        records: &HashMap<Key, SidecarRecord>,
        confirmed: &mut HashSet<Key>,
    ) -> Result<Vec<StorageWrite>, StorageError> {
        let mut writes = Vec::new();
        for write in &marker.writes {
            match write {
                MainWrite::Conversation(record) => writes.push(StorageWrite::Conversation(record.clone())),
                MainWrite::Entry(record) => writes.push(StorageWrite::Entry(record.clone())),
                MainWrite::Task(record) => writes.push(StorageWrite::Task(record.clone())),
                MainWrite::Submission(record) => writes.push(StorageWrite::Submission(record.clone())),
                MainWrite::DocumentRetire { id } => writes.push(StorageWrite::DocumentRetire { id: *id }),
                MainWrite::TaskSidecar { id, ordinal } => {
                    // A task that ended later needs none of its live records.
                    let optional = self.terminal_tasks.contains(id);
                    let record = confirm(records, confirmed, task_file(*id), marker.seq, *ordinal, optional)?;
                    match record.map(|record| record.payload) {
                        Some(Payload::Task { value }) if value.id == *id => {
                            if !optional {
                                writes.push(StorageWrite::Task(value));
                            }
                        }
                        Some(_) => {
                            return Err(corrupt(format!(
                                "Confirmed task sidecar data does not match commit {}",
                                marker.seq
                            )));
                        }
                        None => {}
                    }
                }
                MainWrite::DocumentCreate { record, ordinal } => {
                    let (reclaimed, content) = self.document(marker.seq, record.id, *ordinal, records, confirmed)?;
                    let content = match content {
                        Some(DocumentContent::Base { version, value }) if !reclaimed => BaseContent { version, value },
                        Some(DocumentContent::Delta { .. }) => {
                            return Err(corrupt(format!(
                                "Document creation lacks a confirmed base in commit {}",
                                marker.seq
                            )));
                        }
                        // Reclaimed: a later base or the retirement replaces it.
                        _ => BaseContent {
                            version: 1,
                            value: Json::from(serde_json::json!({})),
                        },
                    };
                    writes.push(StorageWrite::DocumentCreate {
                        record: record.clone(),
                        content,
                    });
                }
                MainWrite::DocumentChange { id, ordinal } => {
                    let (reclaimed, content) = self.document(marker.seq, *id, *ordinal, records, confirmed)?;
                    if let Some(content) = content.filter(|_| !reclaimed) {
                        writes.push(StorageWrite::DocumentChange { id: *id, content });
                    }
                }
            }
        }
        Ok(writes)
    }
}

impl State {
    /// A document record a marker confirms: whether a later base or the retirement made it
    /// reclaimable, and its content when it is still there.
    fn document(
        &self,
        seq: Seq,
        id: DocumentId,
        ordinal: u32,
        records: &HashMap<Key, SidecarRecord>,
        confirmed: &mut HashSet<Key>,
    ) -> Result<(bool, Option<DocumentContent>), StorageError> {
        let reclaimed = self.retired_current_only.contains(&id) || self.before_latest_base(id, seq, ordinal);
        let record = confirm(records, confirmed, doc_file(id), seq, ordinal, reclaimed)?;
        match record.map(|record| record.payload) {
            Some(Payload::Document { id: of, content }) if of == id => Ok((reclaimed, Some(content))),
            Some(_) => Err(corrupt(format!(
                "Confirmed document sidecar data does not match commit {seq}"
            ))),
            None => Ok((reclaimed, None)),
        }
    }
}

/// The record `file` holds for `(seq, ordinal)`, marked confirmed; `None` when an optional
/// one is missing.
fn confirm(
    records: &HashMap<Key, SidecarRecord>,
    confirmed: &mut HashSet<Key>,
    file: String,
    seq: Seq,
    ordinal: u32,
    optional: bool,
) -> Result<Option<SidecarRecord>, StorageError> {
    let key = (file, seq, ordinal);
    if confirmed.contains(&key) {
        return Err(corrupt("Sidecar record is confirmed more than once"));
    }
    match records.get(&key) {
        Some(record) => {
            let record = record.clone();
            confirmed.insert(key);
            Ok(Some(record))
        }
        None if optional => Ok(None),
        None => Err(corrupt(format!(
            "Missing confirmed sidecar record {} at sequence {seq}",
            key.0
        ))),
    }
}

/// The complete lines of the file at `path`: a torn final line is truncated away, and each
/// complete line must be UTF-8 and parse.
async fn read_lines<T>(
    files: &Files,
    path: &str,
    name: &str,
    ctx: &Context,
    parse: impl Fn(&str, usize) -> Result<T, StorageError>,
) -> Result<Vec<Line<T>>, StorageError> {
    let bytes = files.read(path, ctx).await?;
    let complete = match bytes.last() {
        Some(b'\n') | None => bytes.len(),
        Some(_) => {
            let complete = bytes
                .iter()
                .rposition(|byte| *byte == b'\n')
                .map_or(0, |index| index + 1);
            files.fs.truncate_file(path, complete as u64, ctx).await?;
            complete
        }
    };
    let mut lines = Vec::new();
    let mut start = 0;
    for (end, _) in bytes[..complete].iter().enumerate().filter(|(_, byte)| **byte == b'\n') {
        let number = lines.len() + 1;
        let text = std::str::from_utf8(&bytes[start..end])
            .map_err(|_| corrupt(format!("Invalid UTF-8 in complete {name} line {number}")))?;
        lines.push(Line {
            value: parse(text, number)?,
            start: start as u64,
            raw: bytes[start..=end].to_vec(),
        });
        start = end + 1;
    }
    Ok(lines)
}
