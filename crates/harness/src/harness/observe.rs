//! Observation on the surface: the conversation view, agent events and the task graph.

use std::sync::Arc;

use delta_chord::context::Context;

use super::{Conversation, Harness, HarnessError};
use crate::observe::{AgentEventStream, TaskGraphSubscription, TaskGraphView, ViewState, ViewSubscription, attach};

impl Conversation {
    /// The conversation's view: its record, active transcript and mounted documents, as a
    /// read-only state that follows every commit. One mount is shared by every observer of
    /// the conversation and lives as long as one of them.
    pub async fn view_state(&self, ctx: &Context) -> Result<ViewState, HarnessError> {
        let (mount, ()) = self
            .inner
            .views
            .mount(&self.inner.session, self.id, ctx, |_, _| Box::pin(async { Ok(()) }))
            .await?;
        Ok(ViewState::new(mount))
    }

    /// The view's revisions: a hydration, then one update per commit that changed it.
    pub async fn watch(&self, ctx: &Context) -> Result<ViewSubscription, HarnessError> {
        Ok(self.view_state(ctx).await?.subscribe())
    }
}

impl Conversation {
    /// The conversation's agent events from now on, relative to the returned stream's
    /// snapshot; attached on the line, so no commit falls between them.
    pub async fn events(&self, ctx: &Context) -> Result<AgentEventStream, HarnessError> {
        let conversation = self.id;
        let close = self.inner.session.subscribe_close();
        let (mount, attached) = self
            .inner
            .views
            .mount(&self.inner.session, conversation, ctx, move |tx, mount| {
                Box::pin(attach(tx, mount, conversation))
            })
            .await?;
        Ok(attached.into_stream(close, mount))
    }
}

impl Harness {
    /// The Session's task graph: every live task with what it waits for and the
    /// conversations it owns, as a read-only state that follows every commit. One mount is
    /// shared by every observer; it recomputes `blocked` when the scheduler signals a
    /// change without a commit (a registry publication).
    pub async fn task_graph(&self, ctx: &Context) -> Result<TaskGraphView, HarnessError> {
        let probe = self.inner.scheduler.blocked_probe();
        let (mount, built) = self.inner.graph.mount(&self.inner.session, probe, ctx).await?;
        if built {
            let mut blocked = self.inner.scheduler.subscribe_blocked();
            let mounted = Arc::downgrade(&mount);
            let harness = Arc::downgrade(&self.inner);
            tokio::spawn(async move {
                while blocked.changed().await.is_ok() {
                    let (Some(mount), Some(inner)) = (mounted.upgrade(), harness.upgrade()) else {
                        break;
                    };
                    // A closing Session ends the refresh; nothing else can fail it.
                    let _ = mount.refresh_blocked(&inner.session, &Context::background()).await;
                }
            });
        }
        Ok(TaskGraphView::new(mount))
    }

    /// The task graph's revisions: a hydration, then one update per commit that changed it.
    pub async fn watch_task_graph(&self, ctx: &Context) -> Result<TaskGraphSubscription, HarnessError> {
        Ok(self.task_graph(ctx).await?.subscribe())
    }
}
