//! Model context from entries (spec §2.1): head markers cut the active range, context
//! edits omit or replace earlier entries, tool results are ordered after their calls, and
//! the baseline system message leads.
//!
//! A read captures its bounds with two O(1) storage reads (the newest visible entry, or
//! `at`, and the newest head marker at or before it); entries at or below the tail never
//! change, so the range is then scanned and derived without the session line. The agent
//! layer keeps each conversation's last range ([`ContextRanges`]) so a later read scans only
//! the entries after it. Corresponds to upstream `src/harness/context.ts`. See
//! `docs/design/context.md`.

mod derive;
mod order;
mod range;

use delta_ai::types::Message;
use delta_chord::context::Context;

pub use derive::derive_view;
pub use order::{MISSING_RESULT_TEXT, order_tool_results};
pub use range::ContextRanges;

use crate::records::{ConversationId, EntryId, EntryQuery, EntryRecord, ScanOrder};
use crate::storage::{Storage, StorageError};

/// Entries are scanned in pages of this many.
const SCAN_PAGE: usize = 256;

/// One derived context.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ContextView {
    /// The newest head marker in range (`H`).
    pub head: Option<EntryRecord>,
    /// The active entries: `H`, then the non-head entries after its `head`; without `H`,
    /// every entry through the tail.
    pub entries: Vec<EntryRecord>,
    /// Per active entry, its messages after edits and the excluded stop reasons, before
    /// result ordering and the baseline move.
    pub contributions: Vec<Vec<Message>>,
    /// The model context.
    pub messages: Vec<Message>,
}

/// The head marker and newest visible entry that fix one committed context range.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextBounds {
    pub head: Option<EntryRecord>,
    pub tail: EntryId,
}

/// Why a context read failed.
#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    #[error("Entry {entry} is not visible from conversation {conversation}")]
    NotVisible {
        conversation: ConversationId,
        entry: EntryId,
    },
    #[error(transparent)]
    Storage(#[from] StorageError),
}

/// The bounds of the current context, or of the context cut off at the visible entry `at`;
/// `None` for a conversation without entries.
pub async fn capture_bounds(
    storage: &dyn Storage,
    conversation: ConversationId,
    at: Option<EntryId>,
    ctx: &Context,
) -> Result<Option<ContextBounds>, ContextError> {
    let tail = match at {
        Some(entry) => {
            if storage.visible_entry(conversation, entry, ctx).await?.is_none() {
                return Err(ContextError::NotVisible { conversation, entry });
            }
            entry
        }
        None => {
            let query = EntryQuery {
                order: Some(ScanOrder::Descending),
                ..EntryQuery::new(conversation)
            };
            match storage.scan_entries(query, 1, None, ctx).await?.items.first() {
                Some(newest) => newest.id,
                None => return Ok(None),
            }
        }
    };
    let head = storage.find_latest_head_marker(conversation, Some(tail), ctx).await?;
    Ok(Some(ContextBounds { head, tail }))
}

/// The visible entries `from ..= to` (fork traversal included), oldest first.
pub(crate) async fn scan_range(
    storage: &dyn Storage,
    conversation: ConversationId,
    from: Option<EntryId>,
    to: EntryId,
    ctx: &Context,
) -> Result<Vec<EntryRecord>, StorageError> {
    let query = EntryQuery {
        conversation_id: conversation,
        min_entry_id: from,
        max_entry_id: Some(to),
        order: Some(ScanOrder::Ascending),
    };
    let mut entries = Vec::new();
    let mut cursor = None;
    loop {
        let page = storage.scan_entries(query.clone(), SCAN_PAGE, cursor, ctx).await?;
        entries.extend(page.items);
        match page.next {
            Some(next) => cursor = Some(next),
            None => return Ok(entries),
        }
    }
}

/// The context within `bounds`, scanned from storage.
pub async fn derive_context(
    storage: &dyn Storage,
    conversation: ConversationId,
    bounds: Option<&ContextBounds>,
    ctx: &Context,
) -> Result<ContextView, ContextError> {
    let Some(bounds) = bounds else {
        return Ok(ContextView::default());
    };
    let from = bounds.head.as_ref().and_then(|head| head.head);
    let entries = scan_range(storage, conversation, from, bounds.tail, ctx).await?;
    Ok(derive_view(bounds.head.as_ref(), &entries))
}

/// The committed context of `conversation`, or as of the visible entry `at`.
pub async fn read_context(
    storage: &dyn Storage,
    conversation: ConversationId,
    at: Option<EntryId>,
    ctx: &Context,
) -> Result<ContextView, ContextError> {
    let bounds = capture_bounds(storage, conversation, at, ctx).await?;
    derive_context(storage, conversation, bounds.as_ref(), ctx).await
}

#[cfg(test)]
mod tests;
