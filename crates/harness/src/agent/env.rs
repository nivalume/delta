//! The environment seam: how the agent layer gets a conversation's [`ExecutionEnv`].

use std::error::Error as StdError;
use std::sync::Arc;

use delta_chord::context::Context;
use futures::future::BoxFuture;

use crate::env::ExecutionEnv;
use crate::records::ConversationId;

/// Why a builder could not build an environment.
pub type EnvBuildError = Box<dyn StdError + Send + Sync>;

/// What an environment is built for.
#[derive(Clone, Copy, Debug)]
pub struct EnvTarget<'a> {
    pub conversation_id: ConversationId,
    /// The conversation's `delta.agent.cwd`, at the time of the call.
    pub cwd: Option<&'a str>,
}

/// Builds a conversation's environment (`HarnessOptions.env`). Called at each use and never
/// on the session line: by generation preparation for the section renderers and, from phase
/// 5, by the tool task per execution. The host caches what is expensive.
pub trait EnvBuilder: Send + Sync {
    fn build<'a>(
        &'a self,
        target: EnvTarget<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<Option<Arc<dyn ExecutionEnv>>, EnvBuildError>>;
}
