//! Kept context ranges: the agent layer's per-conversation memory of its last context read,
//! so a later read with the same head marker scans only the entries after it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use delta_chord::context::Context;
use tokio::time::Instant;

use super::{ContextBounds, ContextError, ContextView, capture_bounds, derive_view, scan_range};
use crate::records::{ConversationId, EntryId, EntryRecord};
use crate::storage::Storage;

/// One conversation's last read: its bounds, the scanned entries (frozen: entries at or
/// below the tail never change) and the view derived from them.
struct Kept {
    bounds: ContextBounds,
    entries: Arc<[EntryRecord]>,
    view: Arc<ContextView>,
    used: Instant,
}

/// The kept ranges of a session's conversations; a session service of the agent layer.
///
/// A read with the same head marker and a newer tail scans only `tail + 1 ..= new tail` and
/// derives the view again from the kept entries plus the added ones; a shorter tail (a
/// preview at an earlier entry) filters the kept entries and keeps the longer range; a
/// different head marker rescans. A range unused for `retention` is dropped at the next read.
#[derive(Default)]
pub struct ContextRanges {
    kept: Mutex<HashMap<ConversationId, Kept>>,
}

impl std::fmt::Debug for ContextRanges {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kept = self.kept.lock().unwrap_or_else(PoisonError::into_inner);
        f.debug_struct("ContextRanges")
            .field("conversations", &kept.len())
            .finish()
    }
}

/// What a read reuses of the kept range.
enum Reuse {
    Same(Arc<ContextView>),
    Prefix(Arc<[EntryRecord]>),
    Extend(Arc<[EntryRecord]>, EntryId),
    Rescan,
}

impl ContextRanges {
    pub fn new() -> Self {
        Self::default()
    }

    /// How many conversations have a kept range.
    pub fn len(&self) -> usize {
        self.kept.lock().unwrap_or_else(PoisonError::into_inner).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The committed context of `conversation` (as of the visible entry `at`), reusing and
    /// updating its kept range.
    pub async fn read(
        &self,
        storage: &dyn Storage,
        conversation: ConversationId,
        at: Option<EntryId>,
        retention: Duration,
        ctx: &Context,
    ) -> Result<ContextView, ContextError> {
        let Some(bounds) = capture_bounds(storage, conversation, at, ctx).await? else {
            self.lock_pruned(retention).remove(&conversation);
            return Ok(ContextView::default());
        };
        let head_id = bounds.head.as_ref().map(|head| head.id);
        let reuse = match self.lock_pruned(retention).get_mut(&conversation) {
            Some(kept) if kept.bounds.head.as_ref().map(|head| head.id) == head_id => {
                kept.used = Instant::now();
                match bounds.tail.cmp(&kept.bounds.tail) {
                    std::cmp::Ordering::Equal => Reuse::Same(kept.view.clone()),
                    std::cmp::Ordering::Less => Reuse::Prefix(kept.entries.clone()),
                    std::cmp::Ordering::Greater => Reuse::Extend(kept.entries.clone(), kept.bounds.tail),
                }
            }
            _ => Reuse::Rescan,
        };
        let entries: Arc<[EntryRecord]> = match reuse {
            Reuse::Same(view) => return Ok((*view).clone()),
            Reuse::Prefix(entries) => {
                let prefix: Vec<EntryRecord> = entries
                    .iter()
                    .filter(|entry| entry.id <= bounds.tail)
                    .cloned()
                    .collect();
                return Ok(derive_view(bounds.head.as_ref(), &prefix));
            }
            Reuse::Extend(entries, tail) => {
                let added = scan_range(storage, conversation, Some(EntryId(tail.0 + 1)), bounds.tail, ctx).await?;
                entries.iter().cloned().chain(added).collect()
            }
            Reuse::Rescan => {
                let from = bounds.head.as_ref().and_then(|head| head.head);
                scan_range(storage, conversation, from, bounds.tail, ctx).await?.into()
            }
        };
        let view = Arc::new(derive_view(bounds.head.as_ref(), &entries));
        let kept = Kept {
            bounds,
            entries,
            view: view.clone(),
            used: Instant::now(),
        };
        self.lock_pruned(retention).insert(conversation, kept);
        Ok((*view).clone())
    }

    /// The kept ranges, without those unused for `retention`.
    fn lock_pruned(&self, retention: Duration) -> std::sync::MutexGuard<'_, HashMap<ConversationId, Kept>> {
        let mut kept = self.kept.lock().unwrap_or_else(PoisonError::into_inner);
        let now = Instant::now();
        kept.retain(|_, range| now.saturating_duration_since(range.used) < retention);
        kept
    }
}
