//! Submission handles and their waits.

use std::collections::HashMap;
use std::error::Error as StdError;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use delta_chord::context::Context;
use tokio::sync::watch;

use super::{HarnessError, HarnessInner};
use crate::inbox::remove_inbox_item;
use crate::records::{ConversationId, SubmissionId, SubmissionRecord, SubmissionSettlement, SubmissionStatus};
use crate::session::{CommitChange, CommitObserver, CommitPublication, TxError};

/// What withdrawing a submission did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmissionAbort {
    /// It was queued: now `unanswered` / `aborted`, and out of the inbox.
    Aborted,
    /// A placed input: its run answers it; abort the conversation to stop the run.
    AlreadyPlaced,
    /// It was settled already.
    Settled,
}

fn is_settled(record: &SubmissionRecord) -> bool {
    matches!(
        record.state.status(),
        SubmissionStatus::Done | SubmissionStatus::Unanswered
    )
}

/// The waiters of unsettled submissions, resolved by a commit observer on the line.
#[derive(Default)]
pub(super) struct SubmissionWaits {
    watches: Mutex<HashMap<SubmissionId, watch::Sender<Option<SubmissionRecord>>>>,
}

impl SubmissionWaits {
    fn subscribe(&self, id: SubmissionId) -> watch::Receiver<Option<SubmissionRecord>> {
        let mut watches = self.watches.lock().unwrap_or_else(PoisonError::into_inner);
        watches
            .entry(id)
            .or_insert_with(|| watch::Sender::new(None))
            .subscribe()
    }
}

/// Resolves the waiters of every submission a commit settles.
pub(super) struct SubmissionObserver(pub(super) Arc<SubmissionWaits>);

impl CommitObserver for SubmissionObserver {
    fn committed(&mut self, publication: &CommitPublication) -> Result<(), Box<dyn StdError + Send + Sync>> {
        let mut watches = self.0.watches.lock().unwrap_or_else(PoisonError::into_inner);
        if watches.is_empty() {
            return Ok(());
        }
        for change in &publication.changes {
            if let CommitChange::Submission(record) = change
                && is_settled(record)
                && let Some(sender) = watches.remove(&record.id)
            {
                sender.send_replace(Some(record.clone()));
            }
        }
        Ok(())
    }
}

/// A submission of a Harness: compare by id. Stateless; every call reads committed state.
#[derive(Clone)]
pub struct Submission {
    inner: Arc<HarnessInner>,
    id: SubmissionId,
}

impl fmt::Debug for Submission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Submission")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl PartialEq for Submission {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Submission {
    pub(super) fn new(inner: Arc<HarnessInner>, id: SubmissionId) -> Self {
        Self { inner, id }
    }

    pub fn id(&self) -> SubmissionId {
        self.id
    }

    /// The committed record.
    pub async fn status(&self, ctx: &Context) -> Result<SubmissionRecord, HarnessError> {
        let id = self.id;
        let record = self
            .inner
            .session
            .commit(ctx, move |tx| Box::pin(async move { tx.submission(id).await }))
            .await?;
        record.ok_or(HarnessError::UnknownSubmission(id))
    }

    /// Resolves with the record once it is `done` or `unanswered`. Cancelling `ctx` cancels
    /// only the wait; a settled submission resolves even while the Harness closes.
    pub async fn wait(&self, ctx: &Context) -> Result<SubmissionRecord, HarnessError> {
        let _ = self.inner.scheduler.resume();
        let id = self.id;
        loop {
            // Checked and registered on the line, so no settling publication falls between.
            let waits = self.inner.waits.clone();
            let found = self
                .inner
                .session
                .commit(ctx, move |tx| {
                    Box::pin(async move {
                        let record = tx.submission(id).await?.ok_or(TxError::UnknownSubmission(id))?;
                        if is_settled(&record) {
                            return Ok(Ok(record));
                        }
                        Ok(Err(waits.subscribe(id)))
                    })
                })
                .await?;
            let mut receiver = match found {
                Ok(record) => return Ok(record),
                Err(receiver) => receiver,
            };
            let closed = self.inner.session.closed();
            tokio::select! {
                changed = ctx.run(receiver.changed()) => {
                    changed.map_err(HarnessError::Cancelled)?.ok();
                }
                _ = closed => return Err(HarnessError::Closed),
            }
            if let Some(record) = receiver.borrow().clone() {
                return Ok(record);
            }
        }
    }

    /// Withdraws the submission while it is queued.
    pub async fn abort(&self, ctx: &Context) -> Result<SubmissionAbort, HarnessError> {
        abort_submission(&self.inner, self.id, None, ctx).await
    }
}

/// Withdraws a queued submission (of `conversation`, when given) and removes its inbox item.
pub(super) async fn abort_submission(
    inner: &HarnessInner,
    id: SubmissionId,
    conversation: Option<ConversationId>,
    ctx: &Context,
) -> Result<SubmissionAbort, HarnessError> {
    inner
        .session
        .commit(ctx, move |tx| {
            Box::pin(async move {
                let record = tx
                    .submission(id)
                    .await?
                    .filter(|record| conversation.is_none_or(|conversation| record.conversation_id == conversation))
                    .ok_or(TxError::UnknownSubmission(id))?;
                match record.state.status() {
                    SubmissionStatus::Queued => {
                        tx.settle_submission(
                            id,
                            SubmissionSettlement::Unanswered {
                                reason: "aborted".to_owned(),
                                detail: None,
                            },
                        );
                        remove_inbox_item(tx, record.conversation_id, id).await?;
                        Ok(SubmissionAbort::Aborted)
                    }
                    SubmissionStatus::Placed => Ok(SubmissionAbort::AlreadyPlaced),
                    SubmissionStatus::Done | SubmissionStatus::Unanswered => Ok(SubmissionAbort::Settled),
                }
            })
        })
        .await
        .map_err(HarnessError::from)
}
