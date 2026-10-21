//! The public surface: [`Harness`], its [`Conversation`]s and [`Submission`]s.
//!
//! A host opens a storage with an application-owned [`Registry`] of extensions and gets
//! conversations it can submit to, configure, fork, abort and observe. Everything below is
//! the agent layer's: the Harness assembles the Session (with the agent services and the
//! built-in creation hook) and the scheduler (with the registry as its catalog and the agent
//! layer's cleanup hooks), and it opens paused: nothing runs until [`Harness::resume`] or a
//! progress-asking call.
//!
//! Corresponds to upstream `src/harness/harness.ts` and `submissions.ts` (spec §2.2). See
//! `docs/design/harness.md`.

mod approvals;
mod conversation;
mod creation;
mod error;
mod observe;
mod submission;

use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use delta_chord::context::Context;

pub use conversation::{AbortOptions, Conversation, EntryRange};
pub use creation::{ConversationCreateOptions, ConversationInit, RootOptions};
pub use error::{HarnessError, OpenError};
pub use submission::{Submission, SubmissionAbort};

use approvals::SchedulerAbort;
use creation::{CreationHook, Target, create};
use submission::{SubmissionObserver, SubmissionWaits, abort_submission};

use crate::agent::{AgentServices, EnvBuilder, SettingsSource};
use crate::approval::{AllowAll, ApprovalPolicy, PendingApprovals};
use crate::compaction::{Compactor, TaskCompactor};
use crate::documents::DocAddress;
use crate::generation::{AgentSchedulerHooks, builtin_tasks};
use crate::model::ModelClient;
use crate::observe::ViewMounts;
use crate::records::{
    ConversationId, ConversationQuery, SubmissionId, SubmissionQuery, SubmissionRecord, SubmissionStatus, TaskId,
    TaskRecord,
};
use crate::registry::Registry;
use crate::scheduler::{
    Inspection, Scheduler, SchedulerOptions, SchedulerSettings, SettledTask, TaskAbort, Wake, WakeListener,
};
use crate::session::{Clock, ConversationHook, ObserverGuard, ReportInfo, Reporter, Services, Session, SessionOptions};
use crate::storage::Storage;
use crate::usage::{UsageDoc, UsageState, add_usage_state};

/// Scans read pages of this many records.
const SCAN_PAGE: usize = 256;

/// What a Harness runs with. Every dependency comes through here; nothing is global.
pub struct HarnessOptions {
    /// Model access, read at each use.
    pub models: Arc<dyn ModelClient>,
    /// The extensions; holds the built-in tasks.
    pub registry: Registry,
    pub settings: SettingsSource,
    /// Builds a conversation's environment; without one there is none.
    pub env: Option<Arc<dyn EnvBuilder>>,
    /// Decides tool calls' approval; default [`AllowAll`].
    pub approval: Arc<dyn ApprovalPolicy>,
    /// Selects cuts and starts compactions; default [`TaskCompactor`].
    pub compactor: Arc<dyn Compactor>,
    /// Runs in every commit that creates or forks a conversation, after the built-in hook.
    pub conversation_created: Option<Arc<dyn ConversationHook>>,
    /// Epoch milliseconds; default the system time.
    pub clock: Clock,
    pub on_wake: Option<WakeListener>,
    /// Failures that do not fail the calling operation, and once the Session's failure.
    pub on_report: Option<Reporter>,
}

impl HarnessOptions {
    /// The defaults over `models`: a fresh registry, default settings, no environment.
    pub fn new(models: Arc<dyn ModelClient>) -> Self {
        Self {
            models,
            registry: Registry::new(),
            settings: SettingsSource::default(),
            env: None,
            approval: Arc::new(AllowAll),
            compactor: Arc::new(TaskCompactor),
            conversation_created: None,
            clock: SessionOptions::default().now,
            on_wake: None,
            on_report: None,
        }
    }
}

impl fmt::Debug for HarnessOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HarnessOptions")
            .field("registry", &self.registry)
            .field("settings", &self.settings)
            .field("env", &self.env.is_some())
            .finish_non_exhaustive()
    }
}

/// What a Harness's handles share.
pub(crate) struct HarnessInner {
    session: Session,
    scheduler: Scheduler,
    storage: Arc<dyn Storage>,
    registry: Registry,
    services: Arc<AgentServices>,
    approvals: Arc<PendingApprovals>,
    views: ViewMounts,
    waits: Arc<SubmissionWaits>,
    on_report: Option<Reporter>,
    _observer: ObserverGuard,
}

impl HarnessInner {
    fn report(&self, error: &(dyn std::error::Error + Send + Sync), info: &ReportInfo) {
        if let Some(report) = &self.on_report {
            report(error, info);
        }
    }
}

/// The live tasks and the queued and placed submissions.
#[derive(Clone, Debug, PartialEq)]
pub struct HarnessInspection {
    pub scheduler: Inspection,
    /// Queued and placed submissions, in id order.
    pub submissions: Vec<SubmissionRecord>,
}

/// A durable agent harness over one storage. Cheap to clone; derefs to its [`Session`] for
/// the generic document operations. Dropping every handle without [`close`](Self::close)
/// stops everything where it is, as a crash: a reopen resumes from the checkpoints.
#[derive(Clone)]
pub struct Harness {
    inner: Arc<HarnessInner>,
}

impl fmt::Debug for Harness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Harness")
            .field("session", &self.inner.session)
            .field("scheduler", &self.inner.scheduler)
            .finish_non_exhaustive()
    }
}

impl Deref for Harness {
    type Target = Session;

    fn deref(&self) -> &Session {
        &self.inner.session
    }
}

impl Harness {
    /// Opens `storage`: the Session with the agent services and the creation hook, the
    /// scheduler with the registry as its catalog. Surviving work is reconciled and waits,
    /// paused, until [`resume`](Self::resume) or a progress-asking call.
    pub async fn open(storage: Arc<dyn Storage>, options: HarnessOptions, ctx: &Context) -> Result<Harness, OpenError> {
        let HarnessOptions {
            models,
            registry,
            settings,
            env,
            approval,
            compactor,
            conversation_created,
            clock,
            on_wake,
            on_report,
        } = options;
        let snapshot = registry.snapshot();
        for builtin in builtin_tasks() {
            if snapshot.task(builtin.name()).is_none() {
                return Err(OpenError::MissingBuiltin(builtin.name().to_owned()));
            }
        }
        let resolved = settings.resolve();
        let services = Arc::new(AgentServices::new(
            models,
            settings.clone(),
            env,
            storage.clone(),
            compactor,
            approval,
        ));
        let session_services = Services::new();
        session_services.insert(services.clone());
        let approvals = Arc::new(PendingApprovals::new());
        session_services.insert(approvals.clone());
        let session = Session::open(
            storage.clone(),
            SessionOptions {
                document_retention: Some(resolved.document_retention),
                now: clock,
                reporter: on_report.clone(),
                services: session_services,
                conversation_hook: Some(Arc::new(CreationHook {
                    host: conversation_created,
                })),
            },
        );
        let waits = Arc::new(SubmissionWaits::default());
        let observer = session.observe_commits(Box::new(SubmissionObserver(waits.clone())));
        let scheduler_settings = Arc::new(move || {
            let settings = settings.resolve();
            SchedulerSettings {
                restart_limit: settings.restart_limit,
                close_grace: settings.close_grace,
            }
        });
        let options = SchedulerOptions {
            catalog: Arc::new(registry.clone()),
            hooks: Arc::new(AgentSchedulerHooks),
            settings: scheduler_settings,
            on_wake,
        };
        let scheduler = match Scheduler::open(session.clone(), options, ctx).await {
            Ok(scheduler) => scheduler,
            Err(error) => {
                // A failing storage close fails the Session, which reports it.
                let _ = session.close(&Context::background()).await;
                return Err(error.into());
            }
        };
        let inner = Arc::new(HarnessInner {
            session,
            scheduler,
            storage,
            registry,
            services,
            approvals,
            views: ViewMounts::default(),
            waits,
            on_report,
            _observer: observer,
        });
        inner
            .approvals
            .set_run_abort(Arc::new(SchedulerAbort(Arc::downgrade(&inner))));
        Ok(Harness { inner })
    }

    /// The Session.
    pub fn session(&self) -> &Session {
        &self.inner.session
    }

    /// The scheduler.
    pub fn scheduler(&self) -> &Scheduler {
        &self.inner.scheduler
    }

    /// The registry the Harness resolves against.
    pub fn registry(&self) -> &Registry {
        &self.inner.registry
    }

    /// The agent layer's session service.
    pub fn agent_services(&self) -> &Arc<AgentServices> {
        &self.inner.services
    }

    /// Enables scheduling; idempotent.
    pub fn resume(&self) -> Result<(), HarnessError> {
        Ok(self.inner.scheduler.resume()?)
    }

    /// When the Session next needs a running process, from memory.
    pub fn wake(&self) -> Wake {
        self.inner.scheduler.wake()
    }

    /// The root conversation (id 1), created in one commit the first time; `options` are
    /// ignored when it exists.
    pub async fn root(&self, options: RootOptions, ctx: &Context) -> Result<Conversation, HarnessError> {
        let RootOptions { agent, init } = options;
        let id = self
            .inner
            .session
            .commit(ctx, move |tx| Box::pin(create(tx, Target::Root, agent, init)))
            .await?;
        Ok(Conversation::new(self.inner.clone(), id))
    }

    /// A handle on conversation `id`, when it exists.
    pub async fn conversation(&self, id: ConversationId, ctx: &Context) -> Result<Option<Conversation>, HarnessError> {
        let record = self
            .inner
            .session
            .commit(ctx, move |tx| Box::pin(async move { tx.conversation(id).await }))
            .await?;
        Ok(record.map(|record| Conversation::new(self.inner.clone(), record.id)))
    }

    /// A new conversation with its explicit ownership, agent change and `init`, in one
    /// commit.
    pub async fn create_conversation(
        &self,
        options: ConversationCreateOptions,
        ctx: &Context,
    ) -> Result<Conversation, HarnessError> {
        let ConversationCreateOptions { ownership, agent, init } = options;
        let id = self
            .inner
            .session
            .commit(ctx, move |tx| {
                Box::pin(create(tx, Target::Independent(ownership), agent, init))
            })
            .await?;
        Ok(Conversation::new(self.inner.clone(), id))
    }

    /// The committed record of task `id`.
    pub async fn get_task(&self, id: TaskId, ctx: &Context) -> Result<Option<TaskRecord>, HarnessError> {
        Ok(self
            .inner
            .session
            .commit(ctx, move |tx| Box::pin(async move { tx.task(id).await }))
            .await?)
    }

    /// The live tasks with their derived states, and the queued and placed submissions. No
    /// writes, no scheduling.
    pub async fn inspect(&self, ctx: &Context) -> Result<HarnessInspection, HarnessError> {
        let scheduler = self.inner.scheduler.inspect(ctx).await?;
        let mut submissions = Vec::new();
        for status in [SubmissionStatus::Queued, SubmissionStatus::Placed] {
            let query = SubmissionQuery {
                status: Some(status),
                ..SubmissionQuery::default()
            };
            let mut cursor = None;
            loop {
                let page = self
                    .inner
                    .storage
                    .scan_submissions(query.clone(), SCAN_PAGE, cursor, ctx)
                    .await?;
                submissions.extend(page.items);
                match page.next {
                    Some(next) => cursor = Some(next),
                    None => break,
                }
            }
        }
        submissions.sort_by_key(|submission| submission.id);
        Ok(HarnessInspection { scheduler, submissions })
    }

    /// A handle on submission `id`, when it exists.
    pub async fn submission(&self, id: SubmissionId, ctx: &Context) -> Result<Option<Submission>, HarnessError> {
        let record = self
            .inner
            .session
            .commit(ctx, move |tx| Box::pin(async move { tx.submission(id).await }))
            .await?;
        Ok(record.map(|record| Submission::new(self.inner.clone(), record.id)))
    }

    /// Withdraws a queued submission (of `conversation`, when given).
    pub async fn abort_submission(
        &self,
        id: SubmissionId,
        conversation: Option<ConversationId>,
        ctx: &Context,
    ) -> Result<SubmissionAbort, HarnessError> {
        abort_submission(&self.inner, id, conversation, ctx).await
    }

    /// Marks task `id` aborted and joins its active run invocation.
    pub async fn abort_task(&self, id: TaskId, ctx: &Context) -> Result<TaskAbort, HarnessError> {
        self.inner.scheduler.resume()?;
        Ok(self.inner.scheduler.abort_task(id, ctx).await?)
    }

    /// Clears the restart count of a parked task.
    pub async fn retry_task(&self, id: TaskId, ctx: &Context) -> Result<(), HarnessError> {
        Ok(self.inner.scheduler.retry_task(id, ctx).await?)
    }

    /// Resolves with the task's terminal record. Asks for progress.
    pub async fn wait_for_task(&self, id: TaskId, ctx: &Context) -> Result<SettledTask, HarnessError> {
        self.inner.scheduler.resume()?;
        Ok(self.inner.scheduler.wait_for_task(id, ctx).await?)
    }

    /// Resolves once no ownerless conversation has live ordinary work. Asks for progress.
    pub async fn wait_for_idle(&self, ctx: &Context) -> Result<(), HarnessError> {
        self.inner.scheduler.resume()?;
        Ok(self.inner.scheduler.wait_for_idle(None, ctx).await?)
    }

    /// The sum of every conversation's `delta.usage`.
    pub async fn usage(&self, ctx: &Context) -> Result<UsageState, HarnessError> {
        let mut total = UsageState::default();
        let mut cursor = None;
        loop {
            let page = self
                .inner
                .storage
                .scan_conversations(ConversationQuery::default(), SCAN_PAGE, cursor, ctx)
                .await?;
            for conversation in &page.items {
                let address = DocAddress::<UsageDoc>::new(conversation.id);
                if let Some(value) = self.inner.session.snapshot(address, ctx).await? {
                    let state: UsageState = delta_chord::json::from_json(&value).map_err(HarnessError::Document)?;
                    add_usage_state(&mut total, &state);
                }
            }
            match page.next {
                Some(next) => cursor = Some(next),
                None => return Ok(total),
            }
        }
    }

    /// Closes the Harness: seals admission, stops and joins every invocation, writes the
    /// tasks it stopped back as `pending`, then closes storage.
    pub async fn close(&self, ctx: &Context) -> Result<(), HarnessError> {
        Ok(self.inner.scheduler.close(ctx).await?)
    }
}
