//! The type-erased task definition the catalog holds: [`AnyTask`], implemented for every
//! [`TaskDefinition`] by a blanket impl that converts at the boundary.

use std::fmt;
use std::sync::Arc;

use delta_chord::json::{self, Json};
use futures::future::BoxFuture;

use super::definition::{RunningTask, TaskDefinition};
use super::next::ErasedNext;
use super::runtime::{RuntimeCore, TaskRuntime};
use crate::documents::MigrateError;
use crate::records::TaskRecord;

/// A task definition behind `Arc<dyn AnyTask>`: what a [`TaskCatalog`](super::TaskCatalog)
/// lists. Implemented for every [`TaskDefinition`]; sealed.
pub trait AnyTask: sealed::Erased + Send + Sync + 'static {
    fn name(&self) -> &str;
    fn version(&self) -> u32;
    /// Whether older stored versions migrate ([`TaskDefinition::can_migrate`]).
    fn can_migrate(&self) -> bool;
}

impl<T: TaskDefinition> AnyTask for T {
    fn name(&self) -> &str {
        TaskDefinition::name(self)
    }

    fn version(&self) -> u32 {
        TaskDefinition::version(self)
    }

    fn can_migrate(&self) -> bool {
        TaskDefinition::can_migrate(self)
    }
}

impl fmt::Debug for dyn AnyTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyTask")
            .field("name", &self.name())
            .field("version", &self.version())
            .finish()
    }
}

/// Whether `a` and `b` are the same definition object (handover compares identity).
pub(super) fn same_definition(a: &Arc<dyn AnyTask>, b: &Arc<dyn AnyTask>) -> bool {
    std::ptr::addr_eq(Arc::as_ptr(a), Arc::as_ptr(b))
}

/// One phase or abort handler call: the runtime's shared core and the running record.
pub struct PhaseCall {
    pub(super) core: Arc<RuntimeCore>,
    pub(super) record: TaskRecord,
}

impl fmt::Debug for PhaseCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PhaseCall")
            .field("record", &self.record)
            .finish_non_exhaustive()
    }
}

pub(super) mod sealed {
    use super::{BoxFuture, ErasedNext, Json, MigrateError, PhaseCall};

    /// The erased calls the scheduler makes. Errors are the handler's message.
    pub trait Erased {
        fn run_phase(&self, call: PhaseCall) -> BoxFuture<'_, Result<ErasedNext, String>>;
        fn run_abort(&self, call: PhaseCall) -> BoxFuture<'_, Result<ErasedNext, String>>;
        fn migrate_json(&self, input: Json, checkpoint: Json, from: u32) -> Result<(Json, Json), MigrateError>;
    }
}

impl<T: TaskDefinition> sealed::Erased for T {
    fn run_phase(&self, call: PhaseCall) -> BoxFuture<'_, Result<ErasedNext, String>> {
        Box::pin(async move {
            let (rt, checkpoint, ctx) = runtime::<T>(call)?;
            let next = self
                .run(checkpoint, &rt, &ctx)
                .await
                .map_err(|error| error.to_string())?;
            Ok(next.erase())
        })
    }

    fn run_abort(&self, call: PhaseCall) -> BoxFuture<'_, Result<ErasedNext, String>> {
        Box::pin(async move {
            let (rt, checkpoint, ctx) = runtime::<T>(call)?;
            let next = self
                .abort(checkpoint, &rt, &ctx)
                .await
                .map_err(|error| error.to_string())?;
            Ok(next.erase())
        })
    }

    fn migrate_json(&self, input: Json, checkpoint: Json, from: u32) -> Result<(Json, Json), MigrateError> {
        let (input, checkpoint) = self.migrate(input, checkpoint, from)?;
        let input = json::to_json(&input).map_err(MigrateError::failed)?;
        let checkpoint = json::to_json(&checkpoint).map_err(MigrateError::failed)?;
        Ok((input, checkpoint))
    }
}

/// The typed runtime of a call, with the phase's checkpoint and the invocation's context.
fn runtime<T: TaskDefinition>(
    call: PhaseCall,
) -> Result<(TaskRuntime<T>, T::Checkpoint, delta_chord::context::Context), String> {
    let task = RunningTask::<T>::decode(call.record).map_err(|error| error.to_string())?;
    let checkpoint = task.checkpoint.clone();
    let ctx = call.core.invocation.ctx.clone();
    Ok((TaskRuntime::new(call.core, task), checkpoint, ctx))
}
