//! Entry visibility through fork ancestry, shared by the backends.
//!
//! Entry `e` is visible from conversation `c` when `e.conversation_id == c`, or `c` has a
//! parent, `e.id <= c.parent.at` and `e` is visible from the parent. The cap tightens along
//! the chain. A backend supplies an [`EntryReader`]; the walks are here.

use super::error::StorageError;
use super::scan::ScanStart;
use crate::records::{
    ConversationId, ConversationParent, ConversationRecord, EntryId, EntryRecord, ScanOrder, StoredEntry,
};

/// What a conversation descends from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lineage {
    /// The conversation does not exist.
    Unknown,
    /// It has no parent.
    Root,
    Forked(ConversationParent),
}

impl Lineage {
    /// The lineage of a conversation record, `None` when it does not exist.
    pub(crate) fn of(record: Option<&ConversationRecord>) -> Self {
        match record {
            None => Self::Unknown,
            Some(ConversationRecord { parent: None, .. }) => Self::Root,
            Some(ConversationRecord {
                parent: Some(parent), ..
            }) => Self::Forked(*parent),
        }
    }
}

/// The per-conversation reads the ancestry walks need.
pub(crate) trait EntryReader {
    fn lineage(&mut self, conversation: ConversationId) -> Result<Lineage, StorageError>;

    fn entry(&mut self, id: EntryId) -> Result<Option<StoredEntry>, StorageError>;

    /// Entries of one conversation with ids in `[min, max]` (`None`: unbounded) in `order`,
    /// at most `limit` of them.
    fn entries(
        &mut self,
        conversation: ConversationId,
        min: Option<EntryId>,
        max: Option<EntryId>,
        order: ScanOrder,
        limit: usize,
    ) -> Result<Vec<EntryRecord>, StorageError>;

    /// The newest entry of `conversation` with a head marker and an id at or below `max`.
    fn latest_head(
        &mut self,
        conversation: ConversationId,
        max: Option<EntryId>,
    ) -> Result<Option<EntryRecord>, StorageError>;
}

/// One conversation of a visibility chain with its inclusive entry cap (`None`: unbounded).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Segment {
    conversation_id: ConversationId,
    upper: Option<EntryId>,
}

/// The chain of `start` and its ancestors, child first, each capped by `max` and the fork
/// points below it; the walk stops once a cap falls below `min`. An unknown `start` is a
/// request error; an unknown ancestor is corruption.
fn segments(
    reader: &mut impl EntryReader,
    start: ConversationId,
    min: Option<EntryId>,
    max: Option<EntryId>,
) -> Result<Vec<Segment>, StorageError> {
    let mut segments = Vec::new();
    let mut current = start;
    let mut upper = max;
    loop {
        segments.push(Segment {
            conversation_id: current,
            upper,
        });
        let link = match reader.lineage(current)? {
            Lineage::Forked(link) => link,
            Lineage::Root => break,
            Lineage::Unknown if current == start => {
                return Err(StorageError::Request(format!("Unknown conversation: {start}")));
            }
            Lineage::Unknown => {
                return Err(StorageError::Corrupt(format!(
                    "Conversation {current}, an ancestor of {start}, is missing"
                )));
            }
        };
        let capped = upper.map_or(link.at, |upper| upper.min(link.at));
        if min.is_some_and(|min| capped < min) {
            break;
        }
        upper = Some(capped);
        current = link.conversation_id;
    }
    Ok(segments)
}

/// The entry `id` when it is visible from `conversation`.
pub(crate) fn visible_entry(
    reader: &mut impl EntryReader,
    conversation: ConversationId,
    id: EntryId,
) -> Result<Option<StoredEntry>, StorageError> {
    let chain = segments(reader, conversation, Some(id), Some(id))?;
    Ok(reader.entry(id)?.filter(|stored| {
        chain
            .iter()
            .any(|segment| segment.conversation_id == stored.entry.conversation_id)
    }))
}

/// The newest head marker visible from `conversation` at or below `at_or_before`.
pub(crate) fn latest_head(
    reader: &mut impl EntryReader,
    conversation: ConversationId,
    at_or_before: Option<EntryId>,
) -> Result<Option<EntryRecord>, StorageError> {
    for segment in segments(reader, conversation, None, at_or_before)? {
        if let Some(marker) = reader.latest_head(segment.conversation_id, segment.upper)? {
            return Ok(Some(marker));
        }
    }
    Ok(None)
}

/// One scan page's entries (up to `start.fetch`) visible from `conversation`, bounded by
/// `[min, max]`. Descending reads each segment newest first, child first; ascending reads
/// the segments root first, oldest first, each ending at its cap. The cursor narrows the
/// bound on the side the scan moves away from.
pub(crate) fn scan(
    reader: &mut impl EntryReader,
    conversation: ConversationId,
    min: Option<EntryId>,
    max: Option<EntryId>,
    start: &ScanStart,
) -> Result<Vec<EntryRecord>, StorageError> {
    let narrowed = match (start.after, start.order) {
        (None, _) => Some((min, max)),
        (Some(after), ScanOrder::Descending) => after
            .checked_sub(1)
            .map(|below| (min, Some(max.map_or(EntryId(below), |max| max.min(EntryId(below)))))),
        (Some(after), ScanOrder::Ascending) => after
            .checked_add(1)
            .map(|above| (Some(min.map_or(EntryId(above), |min| min.max(EntryId(above)))), max)),
    };
    let Some((min, max)) = narrowed else {
        // Nothing remains, but an unknown conversation is still refused.
        segments(reader, conversation, min, max)?;
        return Ok(Vec::new());
    };
    let mut chain = segments(reader, conversation, min, max)?;
    if start.order == ScanOrder::Ascending {
        chain.reverse();
    }
    let mut found = Vec::new();
    for segment in chain {
        if min.zip(segment.upper).is_some_and(|(min, upper)| upper < min) {
            continue;
        }
        let remaining = start.fetch - found.len();
        found.extend(reader.entries(segment.conversation_id, min, segment.upper, start.order, remaining)?);
        if found.len() >= start.fetch {
            break;
        }
    }
    Ok(found)
}
