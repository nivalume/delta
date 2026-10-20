//! The commit line: one tokio task owns the session state and runs jobs strictly in
//! arrival order, each to completion.
//!
//! ```text
//! admission  healthy, not closing (at the call); ctx not cancelled (on the line)
//! closure    runs to completion with &mut Tx; reads and writes stage; documents load
//! health     a closure that caught a failed read must not commit
//! settle     prepare every change, plan document writes, validate, checkpoint_when last
//! no writes  return the closure's value; nothing is published
//! storage    commit with ctx.without_cancel(); an error has failed the Session
//! adopt      chord State::commit per plan, cache bookkeeping; a failure fails the Session
//! publish    on-line observers synchronously, then subscribers' channels
//! evict      sweep expired idle documents; release the line
//! ```

use std::error::Error as StdError;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use delta_chord::context::{Cancelled, Context};
use delta_chord::delta::{AdoptError, Op};
use futures::FutureExt;
use futures::future::BoxFuture;
use tokio::sync::mpsc;

use super::cache::{AdoptedChange, Cache};
use super::error::{SessionError, TxError, panic_message};
use super::guard::Guard;
use super::lifecycle::{SessionEnd, Shared};
use super::observation::WatchEnd;
use super::plan::{Plan, PlanChange, Settlement};
use super::publish::{CommitChange, CommitPublication, Observers};
use super::tx::{TransactionScope, Tx};
use super::{Clock, ConversationHook};
use crate::records::{DocumentScope, Seq, StorageWrite};

/// One unit of work on the line. Its future is polled to completion, never dropped on
/// caller cancellation, so the line is never left mid-transaction.
pub(super) type Job = Box<dyn for<'a> FnOnce(&'a mut Line) -> BoxFuture<'a, ()> + Send>;

/// The state the line owns.
pub(crate) struct Line {
    pub(super) guard: Guard,
    pub(super) cache: Cache,
    pub(super) shared: Arc<Shared>,
    pub(super) now: Clock,
    pub(super) hook: Option<Arc<dyn ConversationHook>>,
    /// Storage closed; later jobs are dropped unrun.
    finished: bool,
}

/// A panic outside user code: memory may be inconsistent, so it fails the Session.
#[derive(Debug, thiserror::Error)]
#[error("the commit line panicked: {0}")]
struct LinePanic(String);

impl Line {
    pub(super) fn new(
        guard: Guard,
        cache: Cache,
        shared: Arc<Shared>,
        now: Clock,
        hook: Option<Arc<dyn ConversationHook>>,
    ) -> Self {
        Self {
            guard,
            cache,
            shared,
            now,
            hook,
            finished: false,
        }
    }

    /// The line task: jobs in order, eviction on its timer and after every job, and close
    /// when asked or when the last Session handle is dropped.
    pub(super) async fn run(mut self, mut jobs: mpsc::UnboundedReceiver<Job>) {
        loop {
            if self.shared.close_requested() {
                self.finish_close().await;
            }
            let deadline = self.cache.next_sweep();
            let sleep = async move {
                match deadline {
                    Some(at) => tokio::time::sleep_until(at).await,
                    None => std::future::pending().await,
                }
            };
            let job = tokio::select! {
                job = jobs.recv() => match job {
                    Some(job) => Some(job),
                    None => break,
                },
                () = self.shared.wake.notified() => None,
                () = sleep => None,
            };
            if let Some(job) = job {
                self.run_job(job).await;
            }
            self.cache.sweep();
        }
        self.shared.seal(SessionEnd::Closed);
        self.finish_close().await;
    }

    async fn run_job(&mut self, job: Job) {
        if self.finished {
            // Admitted just before the seal and queued behind close: its caller gets the
            // Session's end.
            return;
        }
        if let Err(panic) = AssertUnwindSafe(job(self)).catch_unwind().await {
            self.shared.fail(Arc::new(LinePanic(panic_message(&*panic))));
        }
    }

    /// One commit, from admission to publication.
    pub(super) async fn commit<T, F>(&mut self, ctx: Context, scope: TransactionScope, f: F) -> Result<T, SessionError>
    where
        F: for<'a, 'l> FnOnce(&'a mut Tx<'l>) -> BoxFuture<'a, Result<T, TxError>> + Send,
        T: Send,
    {
        self.shared.check_healthy()?;
        if let Some(reason) = ctx.reason() {
            return Err(SessionError::Cancelled(Cancelled { reason }));
        }
        let staged = AssertUnwindSafe(self.stage(ctx.clone(), scope, f)).catch_unwind().await;
        let (value, settlement) = match staged {
            Ok(Ok(staged)) => staged,
            Ok(Err(error)) => return Err(error.into()),
            Err(panic) => return Err(TxError::Panic(panic_message(&*panic)).into()),
        };
        if settlement.writes.is_empty() {
            return Ok(value);
        }
        let tables = self.shared.has_listeners().then(|| table_changes(&settlement.writes));
        let context = ctx.without_cancel();
        // Once admitted, caller cancellation does not interrupt settlement.
        let seq = self.guard.commit(settlement.writes, &context).await?;
        let documents = match self.adopt(seq, settlement.plans, &context) {
            Ok(documents) => documents,
            // Storage committed; memory would be behind it.
            Err(error) => return Err(SessionError::Failed(self.shared.fail(Arc::new(error)))),
        };
        if let Some(mut changes) = tables {
            changes.extend(documents);
            self.publish(seq, changes, context);
        }
        Ok(value)
    }

    /// Runs the closure and settles the transaction; any error rolls everything back.
    async fn stage<T, F>(&mut self, ctx: Context, scope: TransactionScope, f: F) -> Result<(T, Settlement), TxError>
    where
        F: for<'a, 'l> FnOnce(&'a mut Tx<'l>) -> BoxFuture<'a, Result<T, TxError>> + Send,
    {
        let mut tx = Tx::new(self, ctx, scope);
        let result = f(&mut tx).await;
        // A closure that caught a failed read must not commit, nor succeed as if it had.
        tx.line.shared.check_healthy()?;
        let value = result?;
        let settlement = tx.settle().await?;
        Ok((value, settlement))
    }

    /// Applies the plans after storage accepted the batch and describes the document
    /// changes for the publication.
    fn adopt(&mut self, seq: Seq, plans: Vec<Plan>, ctx: &Context) -> Result<Vec<CommitChange>, AdoptError> {
        let mut changes = Vec::new();
        for plan in plans {
            let record = plan.record.at(seq, plan.retire);
            let base = plan.stored_base();
            let conversation_id = plan.conversation_id;
            let copy_source = match &plan.content {
                Some(StorageWrite::DocumentCopy { source, .. }) => Some(*source),
                _ => None,
            };
            match plan.change {
                Some(PlanChange::Loaded {
                    handle,
                    prepared,
                    version,
                }) if !plan.retire => {
                    if base.is_none() {
                        // Nothing written: the prepared (empty) change is dropped.
                        handle.touch();
                        continue;
                    }
                    let value = prepared.value().clone();
                    let ops: Arc<[Op]> = prepared.ops().into();
                    handle.commit(ctx, prepared)?;
                    let change = AdoptedChange {
                        version,
                        base,
                        value: &value,
                    };
                    self.cache.changed(&handle, change, ctx);
                    changes.push(CommitChange::Document {
                        record,
                        conversation_id,
                        version: Some(version),
                        value: Some(value),
                        ops,
                    });
                    continue;
                }
                Some(PlanChange::Created {
                    state,
                    prepared,
                    version,
                }) if !plan.retire => {
                    let value = prepared.value().clone();
                    state.commit(ctx, prepared)?;
                    self.cache
                        .install_created(record.address(), record.clone(), version, state);
                    changes.push(CommitChange::Document {
                        record,
                        conversation_id,
                        version: Some(version),
                        value: Some(value),
                        ops: Arc::from([]),
                    });
                    continue;
                }
                _ => {}
            }
            if plan.retire {
                // A created-and-retired incarnation never entered the cache.
                self.cache.retire(&record, ctx);
                changes.push(CommitChange::Document {
                    record,
                    conversation_id,
                    version: None,
                    value: None,
                    ops: Arc::from([]),
                });
            } else if let Some(source) = copy_source
                && let DocumentScope::Conversation { conversation_id } = record.scope
            {
                changes.push(CommitChange::DocumentCopy {
                    record,
                    conversation_id,
                    source,
                });
            }
        }
        Ok(changes)
    }

    /// On-line observers first (an error fails the Session), then the subscribers.
    fn publish(&mut self, seq: Seq, changes: Vec<CommitChange>, context: Context) {
        let publication = Arc::new(CommitPublication { seq, changes, context });
        if let Err(error) = Observers::run(&self.shared.observers, &publication) {
            let cause: Arc<dyn StdError + Send + Sync> = Arc::from(error);
            self.shared.fail(cause);
        }
        self.shared.send_publication(publication);
    }

    /// Ends what the Session holds, once: the cache (its states end), the observers, the
    /// subscriptions and the services; then closes storage and marks the Session ended. A
    /// failing storage close fails the Session unless it already failed.
    pub(super) async fn finish_close(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        let reason = match self.shared.failure() {
            Some(failed) => WatchEnd::SessionFailed(failed),
            None => WatchEnd::SessionClosed,
        };
        self.cache.clear(&reason);
        Observers::clear(&self.shared.observers);
        self.shared.close_commits();
        self.shared.services.clear();
        if let Err(error) = self.guard.close().await {
            self.shared.close_failed(Arc::new(error));
        }
        self.shared.finish();
    }
}

/// The table records of a batch, in write order, for the publication.
fn table_changes(writes: &[StorageWrite]) -> Vec<CommitChange> {
    writes
        .iter()
        .filter_map(|write| match write {
            StorageWrite::Conversation(record) => Some(CommitChange::Conversation(record.clone())),
            StorageWrite::Entry(record) => Some(CommitChange::Entry(record.clone())),
            StorageWrite::Task(record) => Some(CommitChange::Task(record.clone())),
            StorageWrite::Submission(record) => Some(CommitChange::Submission(record.clone())),
            _ => None,
        })
        .collect()
}

/// Runs a read job's work; a panic in a definition function (`migrate`) refuses the call.
pub(super) async fn contained<T>(work: impl Future<Output = Result<T, TxError>>) -> Result<T, SessionError> {
    match AssertUnwindSafe(work).catch_unwind().await {
        Ok(result) => result.map_err(SessionError::from),
        Err(panic) => Err(TxError::Panic(panic_message(&*panic)).into()),
    }
}
