//! The agent layer's runtime errors.

use delta_chord::context::Cancelled;

use super::env::EnvBuildError;
use crate::context::ContextError;
use crate::scheduler::RuntimeError;

/// Why an agent-layer runtime operation failed.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// The task runs on a Session that no Harness opened.
    #[error("The task does not run in a Harness: the session has no agent services")]
    NoServices,
    /// The phase's catalog is not a Harness registry.
    #[error("The task's catalog is not a Harness registry")]
    NotARegistry,
    #[error("Stored delta.agent does not convert: {0}")]
    AgentState(serde_json::Error),
    #[error("The environment could not be built: {0}")]
    Env(EnvBuildError),
    #[error(transparent)]
    Context(#[from] ContextError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Cancelled(#[from] Cancelled),
}
