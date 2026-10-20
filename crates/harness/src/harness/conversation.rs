//! [`Conversation`]: a stateless handle on one conversation of a Harness.

use std::fmt;
use std::sync::Arc;

use delta_ai::types::{Message, UserMessage};
use delta_chord::context::Context;
use delta_chord::json;
use futures::future::BoxFuture;

use super::creation::{ConversationCreateOptions, Target, create};
use super::submission::Submission;
use super::{HarnessError, HarnessInner};
use crate::agent::{Agent, AgentChange, AgentDoc, AgentState, configure, resolve_agent};
use crate::context::{ContextView, read_context};
use crate::documents::DocAddress;
use crate::inbox::{QueueModes, RESET_ENTRY, SubmissionDraft, admit_submission};
use crate::records::{ConversationId, Cursor, EntryDraft, EntryId, EntryQuery, EntryRecord, Head, Page, ScanOrder};
use crate::session::{ReportInfo, TransactionScope, Tx, TxError};

/// The entries of a conversation a page reads: an inclusive id range and an order (default
/// descending), bound to the conversation by [`Conversation::entries`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EntryRange {
    pub min_entry_id: Option<EntryId>,
    pub max_entry_id: Option<EntryId>,
    pub order: Option<ScanOrder>,
}

/// What [`Conversation::abort`] reaches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AbortOptions {
    /// Cross background boundaries, waiting for every task the abort marked.
    pub background: bool,
}

/// One conversation of a Harness. Stateless and cheap to clone; compare by id.
#[derive(Clone)]
pub struct Conversation {
    inner: Arc<HarnessInner>,
    id: ConversationId,
}

impl fmt::Debug for Conversation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Conversation")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl PartialEq for Conversation {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Conversation {
    pub(super) fn new(inner: Arc<HarnessInner>, id: ConversationId) -> Self {
        Self { inner, id }
    }

    pub fn id(&self) -> ConversationId {
        self.id
    }

    /// Admits `draft` in one commit and returns once it is durable; queue modes are read on
    /// the line. Asks for progress.
    pub async fn submit(&self, draft: SubmissionDraft, ctx: &Context) -> Result<Submission, HarnessError> {
        self.inner.scheduler.resume()?;
        let conversation = self.id;
        let services = self.inner.services.clone();
        let now = self.inner.session.now();
        let admission = self
            .inner
            .session
            .commit(ctx, move |tx| {
                Box::pin(async move {
                    let modes = QueueModes::from(&services.settings());
                    admit_submission(tx, conversation, draft, now, modes)
                        .await
                        .map_err(crate::inbox::SubmitError::into_tx)
                })
            })
            .await?;
        Ok(Submission::new(self.inner.clone(), admission.id))
    }

    /// The conversation's agent now: the committed `delta.agent` against the current
    /// registry and settings. Never writes.
    pub async fn agent(&self, ctx: &Context) -> Result<Arc<Agent>, HarnessError> {
        let stored = self
            .inner
            .session
            .snapshot(DocAddress::<AgentDoc>::new(self.id), ctx)
            .await?;
        let state: Option<AgentState> = stored
            .map(|value| json::from_json(&value))
            .transpose()
            .map_err(|error| HarnessError::Agent(crate::agent::AgentError::AgentState(error)))?;
        let snapshot = self.inner.registry.snapshot();
        let settings = self.inner.services.settings();
        let agent = resolve_agent(state.as_ref(), &snapshot, &settings, |error, extension| {
            let info = ReportInfo::ExtensionError {
                task: None,
                conversation: Some(self.id),
                extension: Some(extension.to_owned()),
            };
            self.inner.report(&*error, &info);
        });
        Ok(Arc::new(agent))
    }

    /// Applies `change` to `delta.agent` in its own commit.
    pub async fn configure(&self, change: AgentChange, ctx: &Context) -> Result<(), HarnessError> {
        let conversation = self.id;
        self.inner
            .session
            .commit(ctx, move |tx| Box::pin(configure(tx, conversation, change)))
            .await?;
        Ok(())
    }

    /// A session commit bound to the conversation: `tx.create_task` defaults to it.
    pub async fn commit<R, F>(&self, ctx: &Context, f: F) -> Result<R, HarnessError>
    where
        F: for<'a, 'l> FnOnce(&'a mut Tx<'l>) -> BoxFuture<'a, Result<R, TxError>> + Send + 'static,
        R: Send + 'static,
    {
        let scope = TransactionScope {
            conversation_id: Some(self.id),
            task_id: None,
        };
        Ok(self.inner.session.commit_scoped(ctx, scope, f).await?)
    }

    /// The committed model context, or as of the visible entry `at`.
    pub async fn context(&self, at: Option<EntryId>, ctx: &Context) -> Result<ContextView, HarnessError> {
        Ok(read_context(&*self.inner.storage, self.id, at, ctx).await?)
    }

    /// One page of the entries visible from this conversation (fork-aware history).
    pub async fn entries(
        &self,
        range: EntryRange,
        limit: usize,
        cursor: Option<Cursor>,
        ctx: &Context,
    ) -> Result<Page<EntryRecord>, HarnessError> {
        let query = EntryQuery {
            min_entry_id: range.min_entry_id,
            max_entry_id: range.max_entry_id,
            order: range.order,
            ..EntryQuery::new(self.id)
        };
        Ok(self
            .inner
            .session
            .commit(ctx, move |tx| {
                Box::pin(async move { tx.scan_entries(query, limit, cursor).await })
            })
            .await?)
    }

    /// A child conversation that sees this one's entries through `at`, with copies of its
    /// documents by their fork policy.
    pub async fn fork(
        &self,
        at: EntryId,
        options: ConversationCreateOptions,
        ctx: &Context,
    ) -> Result<Conversation, HarnessError> {
        let target = Target::Fork {
            parent: self.id,
            at,
            ownership: options.ownership,
        };
        let ConversationCreateOptions { agent, init, .. } = options;
        let id = self
            .inner
            .session
            .commit(ctx, move |tx| Box::pin(create(tx, target, agent, init)))
            .await?;
        Ok(Conversation::new(self.inner.clone(), id))
    }

    /// Submits a reset: a `delta.reset` write that heads at itself, with the handoff as a
    /// user message. It resolves after admission; placement follows the inbox rules.
    pub async fn reset(&self, handoff: Option<String>, ctx: &Context) -> Result<(), HarnessError> {
        let now = self.inner.session.now();
        let entry = EntryDraft {
            model: handoff.map(|text| {
                vec![Message::User(UserMessage {
                    content: text.into(),
                    timestamp: now,
                })]
            }),
            head: Some(Head::SelfRef),
            ..EntryDraft::new(RESET_ENTRY)
        };
        self.submit(SubmissionDraft::write(entry), ctx).await?;
        Ok(())
    }

    /// Withdraws queued inputs, marks the tasks ordinary traversal selects and returns once
    /// the conversation is ordinarily idle. Asks for progress.
    pub async fn abort(&self, options: AbortOptions, ctx: &Context) -> Result<(), HarnessError> {
        self.inner.scheduler.resume()?;
        self.inner
            .scheduler
            .abort_conversation(self.id, options.background, ctx)
            .await?;
        Ok(())
    }

    /// Resolves once ordinary traversal from the conversation reaches no live task. Asks for
    /// progress.
    pub async fn wait_for_idle(&self, ctx: &Context) -> Result<(), HarnessError> {
        self.inner.scheduler.resume()?;
        self.inner.scheduler.wait_for_idle(Some(self.id), ctx).await?;
        Ok(())
    }
}
