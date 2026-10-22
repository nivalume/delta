//! The storage guard: every storage call of the Session goes through it, and the first
//! fatal error fails the Session.
//!
//! | Error | Effect |
//! | --- | --- |
//! | any error from `commit` or `mint_id` | fails the Session |
//! | a read error other than `Request`, not cancelled by the caller's context | fails the Session |
//! | `Request`; a read whose context was cancelled | fails that call only |

use std::sync::Arc;

use delta_chord::context::{CancelReason, Cancelled, Context};

use super::error::TxError;
use super::lifecycle::Shared;
use crate::records::{RawId, RecordId, Seq, StorageWrite};
use crate::storage::{Storage, StorageError, StorageFuture};

/// The Session's storage behind its failure latch.
pub(super) struct Guard {
    storage: Arc<dyn Storage>,
    shared: Arc<Shared>,
}

impl Guard {
    pub(super) fn new(storage: Arc<dyn Storage>, shared: Arc<Shared>) -> Self {
        Self { storage, shared }
    }

    /// One read. Cancellation of `ctx` ends the call with [`TxError::Cancelled`]; a
    /// `Request` error is [`TxError::Storage`]; any other error fails the Session.
    pub(super) async fn read<'a, T>(
        &'a self,
        ctx: &Context,
        call: impl FnOnce(&'a dyn Storage) -> StorageFuture<'a, T>,
    ) -> Result<T, TxError> {
        self.shared.check_healthy()?;
        let result = match ctx.run(call(&*self.storage)).await {
            Ok(result) => result,
            Err(cancelled) => return Err(TxError::Cancelled(cancelled)),
        };
        match result {
            Ok(value) => {
                // A call that returns after another one failed the Session is not used.
                self.shared.check_healthy()?;
                Ok(value)
            }
            Err(error) if error.is_request() => Err(TxError::Storage(error)),
            Err(_) if ctx.is_cancelled() => Err(TxError::Cancelled(Cancelled {
                reason: ctx.reason().unwrap_or_else(CancelReason::aborted),
            })),
            Err(error) => Err(TxError::Failed(self.fail(error))),
        }
    }

    /// A fresh id of type `I`. Never exempt: any error fails the Session.
    pub(super) async fn mint<I: RecordId>(&self) -> Result<I, TxError> {
        self.shared.check_healthy()?;
        match self.storage.mint_id().await {
            Ok(RawId(raw)) => Ok(I::from_raw(raw)),
            Err(error) => Err(TxError::Failed(self.fail(error))),
        }
    }

    /// Commits one batch. Never exempt: once admitted, whether a failed batch committed is
    /// unknown, so any error fails the Session.
    pub(super) async fn commit(&self, writes: Vec<StorageWrite>, ctx: &Context) -> Result<Seq, TxError> {
        self.shared.check_healthy()?;
        match self.storage.commit(writes, ctx).await {
            Ok(seq) => Ok(seq),
            Err(error) => Err(TxError::Failed(self.fail(error))),
        }
    }

    /// Fails the Session with storage that contradicts itself.
    pub(super) fn corrupt(&self, message: String) -> TxError {
        TxError::Failed(self.fail(StorageError::Corrupt(message)))
    }

    fn fail(&self, error: StorageError) -> super::error::SessionFailed {
        self.shared.fail(Arc::new(error))
    }

    /// Closes the backend whatever the Session's health.
    pub(super) async fn close(&self) -> Result<(), StorageError> {
        self.storage.close(&Context::background()).await
    }
}
