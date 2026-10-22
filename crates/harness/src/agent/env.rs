//! The environment seam: how the agent layer gets a conversation's [`ExecutionEnv`].

use std::collections::HashMap;
use std::error::Error as StdError;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use delta_chord::context::Context;
use futures::future::BoxFuture;

use crate::env::{ExecutionEnv, FileSystem, LocalEnv};
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

/// Builds [`LocalEnv`]s: one per directory, kept for the builder's life so that every call in
/// a directory shares its environment (and `cleanup` reaches its commands). A conversation
/// without a `cwd` gets the builder's default directory.
#[derive(Debug)]
pub struct LocalEnvBuilder {
    default_cwd: PathBuf,
    envs: Mutex<HashMap<String, Arc<LocalEnv>>>,
}

impl LocalEnvBuilder {
    pub fn new(default_cwd: impl Into<PathBuf>) -> Self {
        Self {
            default_cwd: default_cwd.into(),
            envs: Mutex::new(HashMap::new()),
        }
    }

    /// The environment of `cwd`, built on first use.
    pub fn env(&self, cwd: Option<&str>) -> Arc<LocalEnv> {
        let env = LocalEnv::new(cwd.map_or_else(|| self.default_cwd.clone(), PathBuf::from));
        let mut envs = self.envs.lock().unwrap_or_else(PoisonError::into_inner);
        envs.entry(env.cwd().to_owned())
            .or_insert_with(|| Arc::new(env))
            .clone()
    }
}

impl EnvBuilder for LocalEnvBuilder {
    fn build<'a>(
        &'a self,
        target: EnvTarget<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, Result<Option<Arc<dyn ExecutionEnv>>, EnvBuildError>> {
        let env: Arc<dyn ExecutionEnv> = self.env(target.cwd);
        Box::pin(async move { Ok(Some(env)) })
    }
}
