//! Observation on the surface: the conversation view.

use delta_chord::context::Context;

use super::{Conversation, HarnessError};
use crate::observe::{AgentEventStream, ViewState, ViewSubscription, attach};

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
