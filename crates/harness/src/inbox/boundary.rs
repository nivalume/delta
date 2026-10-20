//! Turn boundaries: placing the queued items a boundary selects (spec §6).

use delta_chord::json::Json;

use super::admission::append_input;
use super::{InboxDoc, InboxItem, QueueModes, draft_from_json, inbox_items};
use crate::agent::QueueMode;
use crate::documents::DocAddress;
use crate::records::{ConversationId, EntryDraft, EntryId, Head, SubmissionId, SubmissionSettlement};
use crate::session::{Tx, TxError};

/// Where a boundary runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum At {
    /// Between a tool round and the next request: writes and steers.
    PostTools,
    /// After an answer: writes, steers and follow-ups.
    Final,
}

/// What a boundary read before the commit's first table write, and the newest head it has
/// seen so far.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Boundary {
    conversation: ConversationId,
    modes: QueueModes,
    /// The start of the active range: the newest head marker's `head`, advanced by every
    /// head this commit places.
    pub head: Option<EntryId>,
}

/// The selected user items in ID order, and whether a reset (`head: "self"` write) was
/// placed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BoundaryResult {
    pub users: Vec<SubmissionId>,
    pub reset: bool,
}

/// Reads what a boundary needs. Table reads precede the commit's first table write, so
/// callers prepare the boundary at the start of their commit.
pub async fn prepare_boundary(
    tx: &mut Tx<'_>,
    conversation: ConversationId,
    modes: QueueModes,
) -> Result<Boundary, TxError> {
    let head = tx
        .latest_head_marker(conversation)
        .await?
        .and_then(|marker| marker.head);
    Ok(Boundary {
        conversation,
        modes,
        head,
    })
}

/// Whether a write is display-only: no `model`, `head` or `edits` (present counts, even
/// empty), so it changes nothing the model sees and needs no boundary.
pub fn is_display_only(entry: &EntryDraft) -> bool {
    entry.model.is_none() && entry.head.is_none() && entry.edits.is_none()
}

/// Whether a reset (`head: "self"` write) is queued.
pub fn has_queued_reset(items: &[InboxItem]) -> bool {
    items.iter().any(|item| match item {
        InboxItem::Write { entry, .. } => entry.get("head").and_then(Json::as_str) == Some("self"),
        _ => false,
    })
}

/// Whether a head write targets an entry before the active range, so placing it would bring
/// back history a head cut. `head: "self"` is never stale.
pub fn is_stale(boundary: &Boundary, entry: &EntryDraft) -> bool {
    matches!((entry.head, boundary.head), (Some(Head::At(target)), Some(start)) if target < start)
}

/// Places the queued items the boundary selects: every write, the first or every steer, and
/// at `Final` the first or every follow-up. A queued reset turns `PostTools` into `Final`.
/// Writes are placed first and user items after them, each in ID order; a stale write
/// settles `unanswered` / `stale`. Selected and stale items are removed positionally.
pub async fn apply_boundary(
    tx: &mut Tx<'_>,
    boundary: &mut Boundary,
    at: At,
    now: i64,
) -> Result<BoundaryResult, TxError> {
    let conversation = boundary.conversation;
    let items = inbox_items(tx, conversation).await?;
    let reset = has_queued_reset(&items);
    let final_boundary = at == At::Final || reset;
    let pick = |is_mode: fn(&InboxItem) -> bool, mode: QueueMode| -> Vec<usize> {
        let indexes = items
            .iter()
            .enumerate()
            .filter(|(_, item)| is_mode(item))
            .map(|(index, _)| index);
        match mode {
            QueueMode::All => indexes.collect(),
            QueueMode::OneAtATime => indexes.take(1).collect(),
        }
    };
    let writes: Vec<usize> = pick(|item| matches!(item, InboxItem::Write { .. }), QueueMode::All);
    let mut users = pick(|item| matches!(item, InboxItem::Steer { .. }), boundary.modes.steering);
    if final_boundary {
        users.extend(pick(
            |item| matches!(item, InboxItem::FollowUp { .. }),
            boundary.modes.follow_up,
        ));
    }
    users.sort_unstable();

    for &index in &writes {
        let InboxItem::Write { id, entry } = &items[index] else {
            continue;
        };
        let draft = draft_from_json(entry).map_err(TxError::callback)?;
        if is_stale(boundary, &draft) {
            tx.settle_submission(
                *id,
                SubmissionSettlement::Unanswered {
                    reason: "stale".to_owned(),
                    detail: None,
                },
            );
            continue;
        }
        let head = draft.head;
        let record = tx.append_entry(conversation, draft).await?;
        match head {
            Some(Head::At(target)) => boundary.head = Some(target),
            Some(Head::SelfRef) => boundary.head = Some(record.id),
            None => {}
        }
        tx.place_submission(*id, record.id);
    }
    let mut placed = Vec::with_capacity(users.len());
    for &index in &users {
        let (InboxItem::Steer { id, content, entry } | InboxItem::FollowUp { id, content, entry }) = &items[index]
        else {
            continue;
        };
        let record = append_input(tx, conversation, content.clone(), entry.as_ref(), now).await?;
        tx.place_submission(*id, record.id);
        placed.push(*id);
    }
    let mut removed: Vec<usize> = writes.into_iter().chain(users).collect();
    removed.sort_unstable_by(|a, b| b.cmp(a));
    remove_items(tx, conversation, &removed).await?;
    Ok(BoundaryResult { users: placed, reset })
}

/// Removes the items at `indexes` (descending) with one positional op each.
pub(super) async fn remove_items(
    tx: &mut Tx<'_>,
    conversation: ConversationId,
    indexes: &[usize],
) -> Result<(), TxError> {
    if indexes.is_empty() {
        return Ok(());
    }
    let mut inbox = tx.doc(DocAddress::<InboxDoc>::new(conversation)).await?;
    let mut draft = inbox.draft();
    let Some(mut items) = draft.field("items").and_then(delta_chord::delta::Draft::into_array) else {
        return Err(TxError::callback("delta.inbox has no item list"));
    };
    for &index in indexes {
        if index < items.len() {
            items.remove(index);
        }
    }
    Ok(())
}
