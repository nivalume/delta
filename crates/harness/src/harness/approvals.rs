//! Approvals on the surface: `decide`, the pending requests, and the run abort an `Abort`
//! decision goes through.

use std::error::Error as StdError;
use std::sync::Weak;

use delta_chord::context::Context;
use futures::future::BoxFuture;

use super::{Conversation, Harness, HarnessError, HarnessInner, SCAN_PAGE};
use crate::approval::{
    ApprovalError, ApprovalId, ApprovalRequest, ApprovalsDoc, ApprovalsState, Recorded, ReviewDecision, RunAbort,
    record_decision,
};
use crate::documents::DocAddress;
use crate::inbox::QueueModes;
use crate::records::{ConversationId, ConversationQuery, TaskId};

/// [`RunAbort`] over the scheduler, holding the Harness weakly so the session service does
/// not keep it alive.
pub(super) struct SchedulerAbort(pub(super) Weak<HarnessInner>);

impl RunAbort for SchedulerAbort {
    fn abort_run<'a>(
        &'a self,
        task: TaskId,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<(), Box<dyn StdError + Send + Sync>>> {
        Box::pin(async move {
            let inner = self.0.upgrade().ok_or("The harness is closed")?;
            inner.scheduler.abort_task(task, ctx).await?;
            Ok(())
        })
    }
}

impl Conversation {
    /// Decides the pending approval request `id` of this conversation: one commit writes
    /// the decision (and the `delta.approval` entry, and a session grant for
    /// `ApprovedForSession`), then the waiting tool task is notified. A request that is not
    /// waiting for a decision is [`ApprovalError::NotPending`]; `TimedOut` is the harness's
    /// own. Asks for progress, so the decision is consumed.
    pub async fn decide(&self, id: ApprovalId, decision: ReviewDecision, ctx: &Context) -> Result<(), HarnessError> {
        if decision == ReviewDecision::TimedOut {
            return Err(HarnessError::Approval(ApprovalError::InvalidDecision("timed_out")));
        }
        self.inner.scheduler.resume()?;
        let conversation = self.id;
        let services = self.inner.services.clone();
        let now = self.inner.session.now();
        let written = decision.clone();
        self.inner
            .session
            .commit(ctx, move |tx| {
                Box::pin(async move {
                    let modes = QueueModes::from(&services.settings());
                    match record_decision(tx, conversation, id, written, now, modes).await? {
                        Recorded::Written => Ok(()),
                        Recorded::Already(_) | Recorded::Missing => Err(ApprovalError::NotPending(id).into_tx()),
                    }
                })
            })
            .await?;
        self.inner.approvals.notify(id, decision);
        Ok(())
    }

    /// The conversation's approval requests waiting for a decision, in id order.
    pub async fn pending_approvals(&self, ctx: &Context) -> Result<Vec<ApprovalRequest>, HarnessError> {
        let address = DocAddress::<ApprovalsDoc>::new(self.id);
        let Some(value) = self.inner.session.snapshot(address, ctx).await? else {
            return Ok(Vec::new());
        };
        let state: ApprovalsState = delta_chord::json::from_json(&value).map_err(HarnessError::Document)?;
        Ok(state
            .requests
            .into_iter()
            .filter(|request| request.decision.is_none())
            .collect())
    }
}

impl Harness {
    /// Decides the approval request `id` in whichever conversation its tool task runs;
    /// see [`Conversation::decide`].
    pub async fn decide(&self, id: ApprovalId, decision: ReviewDecision, ctx: &Context) -> Result<(), HarnessError> {
        let task = self
            .get_task(id.0, ctx)
            .await?
            .ok_or(HarnessError::Approval(ApprovalError::UnknownApproval(id)))?;
        Conversation::new(self.inner.clone(), task.conversation_id)
            .decide(id, decision, ctx)
            .await
    }

    /// Every conversation's approval requests waiting for a decision, for a UI that shows
    /// one queue.
    pub async fn pending_approvals(
        &self,
        ctx: &Context,
    ) -> Result<Vec<(ConversationId, ApprovalRequest)>, HarnessError> {
        let mut pending = Vec::new();
        let mut cursor = None;
        loop {
            let page = self
                .inner
                .storage
                .scan_conversations(ConversationQuery::default(), SCAN_PAGE, cursor, ctx)
                .await?;
            for conversation in &page.items {
                let requests = Conversation::new(self.inner.clone(), conversation.id)
                    .pending_approvals(ctx)
                    .await?;
                pending.extend(requests.into_iter().map(|request| (conversation.id, request)));
            }
            match page.next {
                Some(next) => cursor = Some(next),
                None => return Ok(pending),
            }
        }
    }
}
