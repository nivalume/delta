//! Fork copy selection: which conversation documents a fork copies, and from where.
//!
//! | `fork` | source |
//! | --- | --- |
//! | `as_of` | the instance alive at the fork entry's commit, in the conversation that owns the entry |
//! | `current` | the parent's instance when the fork commits |
//! | `initial` | none; the child starts from `initial()` on first access |

use std::collections::HashSet;

use delta_chord::context::Context;

use super::error::TxError;
use super::guard::Guard;
use crate::records::{
    ConversationId, ConversationParent, DocumentAddress, DocumentCopySource, DocumentCreate, DocumentId, DocumentPoint,
    DocumentQuery, DocumentScope, Fork,
};

/// Documents read per scan page.
pub(super) const SCAN_PAGE: usize = 256;

/// One definition-free copy for the child: a new incarnation and its source.
pub(super) struct ForkCopy {
    pub(super) record: DocumentCreate,
    pub(super) source: DocumentCopySource,
}

/// The copies a fork of `parent` at `parent.at` creates for `child`.
pub(super) async fn select_copies(
    guard: &Guard,
    parent: ConversationParent,
    child: ConversationId,
    ctx: &Context,
) -> Result<Vec<ForkCopy>, TxError> {
    let entry = guard
        .read(ctx, |storage| {
            storage.visible_entry(parent.conversation_id, parent.at, ctx)
        })
        .await?
        .ok_or_else(|| {
            TxError::Fork(format!(
                "Entry {} is not visible from conversation {}",
                parent.at, parent.conversation_id
            ))
        })?;
    let mut selection = Selection {
        child,
        copies: Vec::new(),
        addresses: HashSet::new(),
    };
    let entry_scope = DocumentScope::Conversation {
        conversation_id: entry.entry.conversation_id,
    };
    selection
        .collect(guard, entry_scope, DocumentPoint::At(entry.commit_seq), Fork::AsOf, ctx)
        .await?;
    let parent_scope = DocumentScope::Conversation {
        conversation_id: parent.conversation_id,
    };
    selection
        .collect(guard, parent_scope, DocumentPoint::Current, Fork::Current, ctx)
        .await?;
    Ok(selection.copies)
}

struct Selection {
    child: ConversationId,
    copies: Vec<ForkCopy>,
    addresses: HashSet<DocumentAddress>,
}

impl Selection {
    /// Adds a copy of every document of `scope` alive at `at` whose policy is `fork`.
    async fn collect(
        &mut self,
        guard: &Guard,
        scope: DocumentScope,
        at: DocumentPoint,
        fork: Fork,
        ctx: &Context,
    ) -> Result<(), TxError> {
        let mut cursor = None;
        loop {
            let query = DocumentQuery { scope, at, kind: None };
            let page = guard
                .read(ctx, |storage| storage.scan_documents(query, SCAN_PAGE, cursor, ctx))
                .await?;
            for source in page.items {
                let Some(policy) = source.policy.filter(|policy| policy.fork == fork) else {
                    continue;
                };
                let id: DocumentId = guard.mint().await?;
                let record = DocumentCreate {
                    id,
                    kind: source.kind,
                    key: source.key,
                    scope: DocumentScope::Conversation {
                        conversation_id: self.child,
                    },
                    policy: Some(policy),
                };
                if !self.addresses.insert(record.address()) {
                    let member = match &record.key {
                        Some(key) => format!("{}/{key}", record.kind),
                        None => record.kind.clone(),
                    };
                    return Err(TxError::Fork(format!(
                        "Fork selects multiple source documents for {member}"
                    )));
                }
                self.copies.push(ForkCopy {
                    record,
                    source: DocumentCopySource { id: source.id, at },
                });
            }
            match page.next {
                Some(next) => cursor = Some(next),
                None => return Ok(()),
            }
        }
    }
}
