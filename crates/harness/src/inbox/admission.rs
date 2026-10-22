//! Admission and withdrawal of submissions, inside a commit.

use std::collections::HashSet;

use delta_ai::types::{Message, UserContent, UserMessage};

use super::boundary::{
    At, apply_boundary, has_queued_reset, is_display_only, is_stale, prepare_boundary, remove_items,
};
use super::{
    InboxDoc, InboxItem, InboxState, InputEntry, QueueModes, SubmissionDraft, USER_ENTRY, WhenBusy, inbox_items,
};
use super::{SubmitError, draft_to_json};
use crate::documents::DocAddress;
use crate::generation::{live_state, start_run};
use crate::records::{
    ConversationId, EntryDraft, EntryRecord, SubmissionCreate, SubmissionId, SubmissionRecord, SubmissionSettlement,
    SubmissionState, SubmissionStatus, SubmissionType,
};
use crate::session::{Tx, TxError};

/// Where an admitted submission stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionStatus {
    Queued,
    Placed,
    /// Settled at admission (a write), or settled before (a known request id).
    Done,
}

/// The submission an admission created or found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Admission {
    pub id: SubmissionId,
    pub status: AdmissionStatus,
}

/// Admits `draft` inside a commit (spec §6).
///
/// 1. A known request id returns its submission without writing; another type under the
///    same id is refused.
/// 2. A display-only write with no queued reset is appended and settled `done`, busy or idle.
/// 3. A busy input with `WhenBusy::Reject` writes nothing and fails `ConversationBusy`.
/// 4. Busy, or idle with a non-empty inbox: queued; idle then runs a `Final` boundary in the
///    same commit, so order is preserved.
/// 5. Idle with an empty inbox: a write is appended and settled `done` (`unanswered` /
///    `stale` when its head is stale); an input is placed and starts a run.
pub async fn admit_submission(
    tx: &mut Tx<'_>,
    conversation: ConversationId,
    draft: SubmissionDraft,
    now: i64,
    modes: QueueModes,
) -> Result<Admission, SubmitError> {
    if let SubmissionDraft::Input { entry: Some(entry), .. } = &draft {
        check_input_entry(entry)?;
    }
    let kind = match draft {
        SubmissionDraft::Input { .. } => SubmissionType::Input,
        SubmissionDraft::Write { .. } => SubmissionType::Write,
    };
    let request_id = draft.request().map(str::to_owned);
    if let Some(request) = &request_id
        && let Some(existing) = tx.submission_by_request(conversation, request).await?
    {
        return known(&existing, kind, request);
    }
    let create = |state| SubmissionCreate {
        conversation_id: conversation,
        request_id: request_id.clone(),
        kind,
        state,
    };
    if let SubmissionDraft::Write { entry, .. } = &draft
        && is_display_only(entry)
        && !has_queued_reset(&inbox_items(tx, conversation).await?)
    {
        let record = tx.append_entry(conversation, entry.clone()).await?;
        let submission = tx
            .create_submission(create(SubmissionState::Done {
                entry: record.id,
                answer: None,
            }))
            .await?;
        return Ok(admitted(submission.id, AdmissionStatus::Done));
    }
    let busy = live_state(tx, conversation).await?.run.is_some();
    if busy
        && matches!(
            draft,
            SubmissionDraft::Input {
                when_busy: WhenBusy::Reject,
                ..
            }
        )
    {
        return Err(SubmitError::ConversationBusy(conversation));
    }
    let mut boundary = if busy {
        None
    } else {
        Some(prepare_boundary(tx, conversation, modes).await?)
    };
    let queued_items = !inbox_items(tx, conversation).await?.is_empty();
    if busy || queued_items {
        let submission = tx.create_submission(create(SubmissionState::Queued)).await?;
        let id = submission.id;
        let item = match draft {
            SubmissionDraft::Write { entry, .. } => InboxItem::Write {
                id,
                entry: draft_to_json(&entry).map_err(TxError::callback)?,
            },
            SubmissionDraft::Input {
                content,
                when_busy,
                entry,
                ..
            } => match when_busy {
                WhenBusy::Steer => InboxItem::Steer { id, content, entry },
                WhenBusy::FollowUp | WhenBusy::Reject => InboxItem::FollowUp { id, content, entry },
            },
        };
        tx.doc(DocAddress::<InboxDoc>::new(conversation))
            .await?
            .update_as::<InboxState, _>(|state| state.items.push(item))
            .map_err(TxError::callback)?;
        let Some(boundary) = &mut boundary else {
            return Ok(admitted(id, AdmissionStatus::Queued));
        };
        let placed = apply_boundary(tx, boundary, At::Final, now).await?;
        if !placed.users.is_empty() {
            start_run(tx, conversation, placed.users.clone()).await?;
        }
        let status = match kind {
            SubmissionType::Write => AdmissionStatus::Done,
            SubmissionType::Input if placed.users.last() == Some(&id) => AdmissionStatus::Placed,
            SubmissionType::Input => AdmissionStatus::Queued,
        };
        return Ok(admitted(id, status));
    }
    let boundary = boundary.ok_or_else(|| TxError::callback("an idle admission prepares its boundary"))?;
    match draft {
        SubmissionDraft::Write { entry, .. } => {
            if is_stale(&boundary, &entry) {
                let state = SubmissionState::Unanswered {
                    entry: None,
                    reason: "stale".to_owned(),
                    detail: None,
                };
                let submission = tx.create_submission(create(state)).await?;
                return Ok(admitted(submission.id, AdmissionStatus::Done));
            }
            let record = tx.append_entry(conversation, entry).await?;
            let state = SubmissionState::Done {
                entry: record.id,
                answer: None,
            };
            let submission = tx.create_submission(create(state)).await?;
            Ok(admitted(submission.id, AdmissionStatus::Done))
        }
        SubmissionDraft::Input { content, entry, .. } => {
            let record = append_input(tx, conversation, content, entry.as_ref(), now).await?;
            let submission = tx
                .create_submission(create(SubmissionState::Placed { entry: record.id }))
                .await?;
            start_run(tx, conversation, vec![submission.id]).await?;
            Ok(admitted(submission.id, AdmissionStatus::Placed))
        }
    }
}

fn admitted(id: SubmissionId, status: AdmissionStatus) -> Admission {
    Admission { id, status }
}

/// The admission of a submission found by its request id.
fn known(existing: &SubmissionRecord, kind: SubmissionType, request: &str) -> Result<Admission, SubmitError> {
    if existing.kind != kind {
        return Err(SubmitError::RequestIdTypeMismatch {
            request_id: request.to_owned(),
            existing: existing.kind,
        });
    }
    let status = match existing.state.status() {
        SubmissionStatus::Queued => AdmissionStatus::Queued,
        SubmissionStatus::Placed => AdmissionStatus::Placed,
        SubmissionStatus::Done | SubmissionStatus::Unanswered => AdmissionStatus::Done,
    };
    Ok(admitted(existing.id, status))
}

/// An input entry's kind must be non-empty and outside the crate's `delta.` prefix.
fn check_input_entry(entry: &InputEntry) -> Result<(), SubmitError> {
    if entry.kind.is_empty() || entry.kind.starts_with("delta.") {
        return Err(SubmitError::InvalidEntryKind(entry.kind.clone()));
    }
    Ok(())
}

/// Appends the entry an input is placed as: `entry`'s kind and data, or `delta.user`, with
/// the user message of `content` stamped `now`.
pub async fn append_input(
    tx: &mut Tx<'_>,
    conversation: ConversationId,
    content: UserContent,
    entry: Option<&InputEntry>,
    now: i64,
) -> Result<EntryRecord, TxError> {
    let message = Message::User(UserMessage {
        content,
        timestamp: now,
    });
    let mut draft = match entry {
        Some(entry) => EntryDraft {
            data: entry.data.clone(),
            ..EntryDraft::new(entry.kind.clone())
        },
        None => EntryDraft::new(USER_ENTRY),
    };
    draft.model = Some(vec![message]);
    tx.append_entry(conversation, draft).await
}

/// Removes a withdrawn submission's item; the caller settles the submission.
pub async fn remove_inbox_item(tx: &mut Tx<'_>, conversation: ConversationId, id: SubmissionId) -> Result<(), TxError> {
    let items = inbox_items(tx, conversation).await?;
    let indexes: Vec<usize> = items.iter().position(|item| item.id() == id).into_iter().collect();
    remove_items(tx, conversation, &indexes).await
}

/// Withdraws every queued input of `conversation`: each settles `unanswered` with `reason`
/// and leaves the inbox; queued writes stay for later placement.
pub async fn withdraw_queued_inputs(
    tx: &mut Tx<'_>,
    conversation: ConversationId,
    reason: &str,
) -> Result<(), TxError> {
    let items = inbox_items(tx, conversation).await?;
    let mut indexes = Vec::new();
    for (index, item) in items.iter().enumerate().rev() {
        if item.is_input() {
            settle_unanswered(tx, item.id(), reason);
            indexes.push(index);
        }
    }
    remove_items(tx, conversation, &indexes).await
}

/// The scheduler's withdrawal of the queued inputs a cascade or a conversation abort
/// reached: each settles `unanswered` / `aborted` and leaves `delta.inbox`.
pub async fn withdraw_inputs(
    tx: &mut Tx<'_>,
    conversation: ConversationId,
    queued: &[SubmissionRecord],
) -> Result<(), TxError> {
    let ids: HashSet<SubmissionId> = queued.iter().map(|submission| submission.id).collect();
    for id in &ids {
        settle_unanswered(tx, *id, "aborted");
    }
    let items = inbox_items(tx, conversation).await?;
    let indexes: Vec<usize> = items
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, item)| ids.contains(&item.id()))
        .map(|(index, _)| index)
        .collect();
    remove_items(tx, conversation, &indexes).await
}

fn settle_unanswered(tx: &mut Tx<'_>, id: SubmissionId, reason: &str) {
    tx.settle_submission(
        id,
        SubmissionSettlement::Unanswered {
            reason: reason.to_owned(),
            detail: None,
        },
    );
}
