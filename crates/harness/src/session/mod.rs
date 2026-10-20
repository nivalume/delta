//! The Session: one commit line over one storage, the document cache, observation and
//! forks.
//!
//! One tokio task (the line) owns the session state and runs jobs in arrival order, each to
//! completion. A commit is a closure that receives `&mut` [`Tx`]: it reads tables, writes
//! records and edits documents through borrowed [`Doc`] drafts over chord changes; when it
//! returns, the transaction settles into one atomic storage batch, and only what storage
//! accepted is adopted and published. Documents are chord `State`s; a [`DocumentState`]
//! observes one incarnation read-only. The first fatal storage error fails the Session
//! ([`SessionFailed`]); the host reopens the storage.
//!
//! Corresponds to upstream `src/session/{session,transaction,forks,observation}.ts`. See
//! `docs/design/session.md`.
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use delta_chord::context::Context;
//! # use delta_harness::records::{ConversationOwnership, EntryDraft};
//! # use delta_harness::session::{Session, SessionOptions};
//! # use delta_harness::storage::MemoryStorage;
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let session = Session::open(Arc::new(MemoryStorage::new()), SessionOptions::default());
//! let ctx = Context::background();
//! let entry = session
//!     .commit(&ctx, |tx| {
//!         Box::pin(async move {
//!             let conversation = tx.create_conversation(ConversationOwnership::Ownerless).await?;
//!             let entry = tx.append_entry(conversation.id, EntryDraft::new("note")).await?;
//!             Ok(entry.id)
//!         })
//!     })
//!     .await?;
//! # let _ = entry;
//! session.close(&ctx).await?;
//! # Ok(())
//! # }
//! ```

mod cache;
mod drafts;
mod error;
mod forks;
mod guard;
mod lifecycle;
mod line;
mod observation;
mod plan;
mod publish;
mod report;
mod services;
mod snapshot;
#[cfg(test)]
mod tests;
mod tx;

use std::fmt;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use delta_chord::context::Context;
use futures::FutureExt;
use futures::future::BoxFuture;
use tokio::sync::{mpsc, oneshot};

pub use drafts::Doc;
pub use error::{SessionError, SessionFailed, TxError};
pub use lifecycle::SessionEnd;
pub use observation::{DocumentState, DocumentSubscription, WatchEnd};
pub use publish::{
    COMMIT_BACKLOG, CloseSubscription, CommitChange, CommitGap, CommitObserver, CommitPublication, CommitSubscription,
    ObserverGuard,
};
pub use report::{FaultReason, HostCallback, OutdatedReason, ReportInfo, ReportTask, Reporter};
pub use services::Services;
pub use tx::{TaskCreate, TaskOptions, Tx};

#[allow(unused_imports, reason = "the scheduler binds its commits to a scope (next phase)")]
pub(crate) use tx::TransactionScope;

use cache::Cache;
use guard::Guard;
use lifecycle::Shared;
use line::{Job, Line};
use publish::Observers;

use crate::records::ConversationRecord;
use crate::storage::Storage;

/// Milliseconds since the Unix epoch, for task lifecycle times.
pub type Clock = Arc<dyn Fn() -> i64 + Send + Sync>;

/// Runs inside every transaction that creates or forks a conversation, after the
/// conversation record is staged: the layer above stages the documents every conversation
/// starts with there. Its error rolls the transaction back.
pub trait ConversationHook: Send + Sync + 'static {
    fn conversation_created<'a>(
        &'a self,
        tx: &'a mut Tx<'_>,
        record: &'a ConversationRecord,
    ) -> BoxFuture<'a, Result<(), TxError>>;
}

/// How a Session runs.
pub struct SessionOptions {
    /// How long a document no state observes stays cached after its last use: `None` keeps
    /// documents, `Some(Duration::ZERO)` drops them right after use. Default 10 minutes.
    pub document_retention: Option<Duration>,
    /// The wall clock for task times; default the system time.
    pub now: Clock,
    /// Receives the Session's failure, once.
    pub reporter: Option<Reporter>,
    /// The in-memory services of the layers above.
    pub services: Services,
    /// Runs in every transaction that creates or forks a conversation.
    pub conversation_hook: Option<Arc<dyn ConversationHook>>,
}

/// The default [`SessionOptions::document_retention`].
pub const DEFAULT_DOCUMENT_RETENTION: Duration = Duration::from_mins(10);

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            document_retention: Some(DEFAULT_DOCUMENT_RETENTION),
            now: Arc::new(system_now),
            reporter: None,
            services: Services::new(),
            conversation_hook: None,
        }
    }
}

impl fmt::Debug for SessionOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionOptions")
            .field("document_retention", &self.document_retention)
            .field("reporter", &self.reporter.is_some())
            .field("services", &self.services)
            .field("conversation_hook", &self.conversation_hook.is_some())
            .finish_non_exhaustive()
    }
}

fn system_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX))
}

/// A handle on one open session. Cheap to clone; every clone is the same session, and
/// dropping the last one closes it.
#[derive(Clone)]
pub struct Session {
    inner: Arc<Handle>,
}

struct Handle {
    jobs: mpsc::UnboundedSender<Job>,
    shared: Arc<Shared>,
}

impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("sealed", &self.shared().is_sealed())
            .field("failure", &self.shared().failure())
            .finish_non_exhaustive()
    }
}

impl Session {
    /// Opens a Session over `storage` and starts its line.
    ///
    /// # Panics
    ///
    /// Outside a tokio runtime.
    pub fn open(storage: Arc<dyn Storage>, options: SessionOptions) -> Session {
        let SessionOptions {
            document_retention,
            now,
            reporter,
            services,
            conversation_hook,
        } = options;
        let shared = Arc::new(Shared::new(reporter, services));
        let guard = Guard::new(storage, shared.clone());
        let cache = Cache::new(document_retention, shared.clone());
        let line = Line::new(guard, cache, shared.clone(), now, conversation_hook);
        let (jobs, receiver) = mpsc::unbounded_channel();
        tokio::spawn(line.run(receiver));
        Session {
            inner: Arc::new(Handle { jobs, shared }),
        }
    }

    fn shared(&self) -> &Arc<Shared> {
        &self.inner.shared
    }

    /// Runs `f` on the line as one atomic commit and returns its value.
    ///
    /// The closure's error, or a rejection at settlement, rolls everything back. Once the
    /// batch reaches storage, cancelling `ctx` (or dropping this future) does not interrupt
    /// it. A commit that writes nothing publishes nothing.
    ///
    /// ```no_run
    /// # use delta_harness::session::{Session, TxError};
    /// # use delta_harness::records::{ConversationId, EntryDraft};
    /// # async fn example(session: Session, id: ConversationId) -> Result<(), Box<dyn std::error::Error>> {
    /// # let ctx = delta_chord::context::Context::background();
    /// let entry = session
    ///     .commit(&ctx, move |tx| {
    ///         Box::pin(async move {
    ///             let conversation = tx.conversation(id).await?; // a table read, before any write
    ///             let Some(conversation) = conversation else {
    ///                 return Err(TxError::callback("no such conversation"));
    ///             };
    ///             tx.append_entry(conversation.id, EntryDraft::new("note")).await
    ///         })
    ///     })
    ///     .await?;
    /// # let _ = entry;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn commit<T, F>(&self, ctx: &Context, f: F) -> Result<T, SessionError>
    where
        F: for<'a, 'l> FnOnce(&'a mut Tx<'l>) -> BoxFuture<'a, Result<T, TxError>> + Send + 'static,
        T: Send + 'static,
    {
        self.commit_scoped(ctx, TransactionScope::default(), f).await
    }

    /// [`commit`](Self::commit) bound to a scope: the default conversation of new tasks and
    /// the task stamped on appended entries (task runtime commits).
    pub(crate) async fn commit_scoped<T, F>(
        &self,
        ctx: &Context,
        scope: TransactionScope,
        f: F,
    ) -> Result<T, SessionError>
    where
        F: for<'a, 'l> FnOnce(&'a mut Tx<'l>) -> BoxFuture<'a, Result<T, TxError>> + Send + 'static,
        T: Send + 'static,
    {
        self.shared().check_usable()?;
        let ctx = ctx.clone();
        self.run(move |line| line.commit(ctx, scope, f).boxed()).await
    }

    /// Queues `job` on the line and waits for its result. The job runs to completion even
    /// when this future is dropped.
    async fn run<T, J>(&self, job: J) -> Result<T, SessionError>
    where
        J: for<'a> FnOnce(&'a mut Line) -> BoxFuture<'a, Result<T, SessionError>> + Send + 'static,
        T: Send + 'static,
    {
        self.shared().check_usable()?;
        let (reply, result) = oneshot::channel();
        let job: Job = Box::new(move |line| {
            async move {
                let _ = reply.send(job(line).await);
            }
            .boxed()
        });
        if self.inner.jobs.send(job).is_err() {
            return Err(self.shared().terminal_error());
        }
        result.await.unwrap_or_else(|_| Err(self.shared().terminal_error()))
    }

    /// Every commit publication from now on, off the line: subscribers may call the
    /// Session. Ends at close.
    pub fn subscribe_commits(&self) -> CommitSubscription {
        CommitSubscription::new(self.shared().subscribe_commits())
    }

    /// Registers a Session component that runs on the line after each adoption, before
    /// subscribers are fed (see [`CommitObserver`]). Dropping the guard removes it.
    pub fn observe_commits(&self, observer: Box<dyn CommitObserver>) -> ObserverGuard {
        Observers::register(&self.shared().observers, observer)
    }

    /// Resolves when close begins, a failure included, after admission is sealed.
    pub fn subscribe_close(&self) -> CloseSubscription {
        CloseSubscription::new(self.shared().phase())
    }

    /// Seals admission, ends every document state, lets admitted commits settle, drops the
    /// cache and services, then closes storage. Closing twice waits for the same close.
    /// Fails only when storage failed to close; cancelling `ctx` stops the wait, not the
    /// close.
    pub async fn close(&self, ctx: &Context) -> Result<(), SessionError> {
        if self.shared().seal(SessionEnd::Closed) {
            let job: Job = Box::new(|line| line.finish_close().boxed());
            if self.inner.jobs.send(job).is_err() {
                self.shared().request_close();
            }
        }
        ctx.run(self.shared().ended()).await.map_err(SessionError::Cancelled)?;
        match self.shared().close_error() {
            Some(failed) => Err(SessionError::Failed(failed)),
            None => Ok(()),
        }
    }

    /// Resolves once the Session ended: closed, or failed and closed.
    pub fn closed(&self) -> impl Future<Output = SessionEnd> + Send + 'static {
        let shared = self.shared().clone();
        async move { shared.ended().await }
    }

    /// The failure, once the Session failed.
    pub fn failure(&self) -> Option<SessionFailed> {
        self.shared().failure()
    }

    /// The in-memory services of the layers above; dropped at close.
    pub fn services(&self) -> &Services {
        &self.shared().services
    }

    /// Fails the Session from a component whose memory fell behind storage (a scheduler
    /// bookkeeping commit that failed).
    #[allow(
        dead_code,
        reason = "the scheduler fails the Session on a bookkeeping error (next phase)"
    )]
    pub(crate) fn fail(&self, cause: Arc<dyn std::error::Error + Send + Sync>) -> SessionFailed {
        self.shared().fail(cause)
    }

    /// How many documents the cache holds, for tests of retention.
    #[doc(hidden)]
    pub async fn cached_documents(&self) -> Result<usize, SessionError> {
        self.run(|line| async move { Ok(line.cache.len()) }.boxed()).await
    }
}
