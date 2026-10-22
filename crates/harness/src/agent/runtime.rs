//! What agent-layer tasks reach through their runtime: [`AgentRuntime`] and [`HookRuntime`],
//! extension traits on the kernel's `TaskRuntime` built from the session's [`AgentServices`],
//! the phase's catalog snapshot and its per-phase data.

use std::fmt;
use std::sync::Arc;

use delta_chord::context::Context;
use delta_chord::json;
use tokio::sync::OnceCell;

use super::env::{EnvBuilder, EnvTarget};
use super::{Agent, AgentDoc, AgentError, AgentState, Settings, SettingsSource, resolve_agent};
use crate::approval::ApprovalPolicy;
use crate::compaction::Compactor;
use crate::context::{ContextRanges, ContextView};
use crate::documents::DocAddress;
use crate::env::ExecutionEnv;
use crate::model::ModelClient;
use crate::records::{ConversationId, EntryId};
use crate::registry::{HookApi, HookRunner, RegistrySnapshot, TaskHooks};
use crate::scheduler::{ReportFrom, TaskDefinition, TaskRuntime};
use crate::session::HostCallback;
use crate::storage::Storage;

/// The agent layer's session service: the host dependencies its tasks read at each use and
/// the per-session memory they share. Inserted into the session's services at open; it
/// holds no handle on the Harness, the scheduler or the Session.
pub struct AgentServices {
    models: Arc<dyn ModelClient>,
    settings: SettingsSource,
    env: Option<Arc<dyn EnvBuilder>>,
    storage: Arc<dyn Storage>,
    contexts: ContextRanges,
    compactor: Arc<dyn Compactor>,
    approval: Arc<dyn ApprovalPolicy>,
}

impl AgentServices {
    pub fn new(
        models: Arc<dyn ModelClient>,
        settings: SettingsSource,
        env: Option<Arc<dyn EnvBuilder>>,
        storage: Arc<dyn Storage>,
        compactor: Arc<dyn Compactor>,
        approval: Arc<dyn ApprovalPolicy>,
    ) -> Self {
        Self {
            models,
            settings,
            env,
            storage,
            contexts: ContextRanges::new(),
            compactor,
            approval,
        }
    }

    pub fn models(&self) -> &Arc<dyn ModelClient> {
        &self.models
    }

    /// The settings, resolved now.
    pub fn settings(&self) -> Settings {
        self.settings.resolve()
    }

    pub fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }

    /// The kept context ranges.
    pub fn contexts(&self) -> &ContextRanges {
        &self.contexts
    }

    pub fn compactor(&self) -> &Arc<dyn Compactor> {
        &self.compactor
    }

    pub fn approval(&self) -> &Arc<dyn ApprovalPolicy> {
        &self.approval
    }

    /// The environment of `conversation` in `cwd`; `None` without a builder.
    pub async fn build_env(
        &self,
        conversation: ConversationId,
        cwd: Option<&str>,
        ctx: &Context,
    ) -> Result<Option<Arc<dyn ExecutionEnv>>, AgentError> {
        let Some(builder) = &self.env else {
            return Ok(None);
        };
        let target = EnvTarget {
            conversation_id: conversation,
            cwd,
        };
        builder.build(target, ctx).await.map_err(AgentError::Env)
    }
}

impl fmt::Debug for AgentServices {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentServices")
            .field("settings", &self.settings)
            .field("env", &self.env.is_some())
            .field("contexts", &self.contexts)
            .finish_non_exhaustive()
    }
}

/// The phase's agent, resolved at first use.
#[derive(Default)]
struct PhaseAgent(OnceCell<Arc<Agent>>);

/// What agent-layer tasks reach through their runtime.
pub trait AgentRuntime {
    /// The session's agent services.
    fn agent_services(&self) -> Result<Arc<AgentServices>, AgentError>;

    /// The phase's registry snapshot.
    fn registry(&self) -> Result<RegistrySnapshot, AgentError>;

    /// The conversation's agent, resolved once per phase from the phase's snapshot, the
    /// committed `delta.agent` and the settings.
    fn agent(&self, ctx: &Context) -> impl Future<Output = Result<Arc<Agent>, AgentError>> + Send;

    /// The conversation's environment, built now in the phase agent's `cwd`.
    fn env(&self, ctx: &Context) -> impl Future<Output = Result<Option<Arc<dyn ExecutionEnv>>, AgentError>> + Send;

    /// The committed context of `conversation` (as of the visible entry `at`), through the
    /// kept ranges.
    fn context_view(
        &self,
        conversation: ConversationId,
        at: Option<EntryId>,
        ctx: &Context,
    ) -> impl Future<Output = Result<ContextView, AgentError>> + Send;
}

impl<T: TaskDefinition> AgentRuntime for TaskRuntime<T> {
    fn agent_services(&self) -> Result<Arc<AgentServices>, AgentError> {
        self.services().get::<AgentServices>().ok_or(AgentError::NoServices)
    }

    fn registry(&self) -> Result<RegistrySnapshot, AgentError> {
        RegistrySnapshot::from_catalog(self.catalog()).ok_or(AgentError::NotARegistry)
    }

    async fn agent(&self, ctx: &Context) -> Result<Arc<Agent>, AgentError> {
        let cell = self.phase_data().get_or_insert_with(PhaseAgent::default);
        let agent = cell
            .0
            .get_or_try_init(|| async {
                let services = self.agent_services()?;
                let registry = self.registry()?;
                let stored = self
                    .snapshot(DocAddress::<AgentDoc>::new(self.conversation_id()), ctx)
                    .await?;
                let state: Option<AgentState> = stored
                    .map(|value| json::from_json(&value))
                    .transpose()
                    .map_err(AgentError::AgentState)?;
                let agent = resolve_agent(state.as_ref(), &registry, &services.settings(), |error, extension| {
                    self.report(&*error, Some(ReportFrom::Extension(Some(extension.to_owned()))));
                });
                Ok::<_, AgentError>(Arc::new(agent))
            })
            .await?;
        Ok(agent.clone())
    }

    async fn env(&self, ctx: &Context) -> Result<Option<Arc<dyn ExecutionEnv>>, AgentError> {
        let services = self.agent_services()?;
        let agent = self.agent(ctx).await?;
        services
            .build_env(self.conversation_id(), agent.cwd.as_deref(), ctx)
            .await
    }

    async fn context_view(
        &self,
        conversation: ConversationId,
        at: Option<EntryId>,
        ctx: &Context,
    ) -> Result<ContextView, AgentError> {
        let services = self.agent_services()?;
        let retention = services.settings().context_retention;
        let view = services
            .contexts()
            .read(&**services.storage(), conversation, at, retention, ctx)
            .await?;
        Ok(view)
    }
}

/// The hooks of a task that declares them ([`TaskHooks`]).
pub trait HookRuntime: AgentRuntime {
    type Hooks: ?Sized;

    /// The phase agent's hooks for this task's name, in extension order.
    fn hooks(&self, ctx: &Context) -> impl Future<Output = Result<HookRunner<Self::Hooks>, AgentError>> + Send;

    /// What the hooks reach of the task.
    fn hook_api(&self) -> Result<HookApi<'_>, AgentError>;

    /// Reports a hook's ordinary failure, naming its extension.
    fn report_hook(&self, error: &(dyn std::error::Error + Send + Sync), extension: &str);
}

impl<T: TaskHooks> HookRuntime for TaskRuntime<T> {
    type Hooks = T::Hooks;

    async fn hooks(&self, ctx: &Context) -> Result<HookRunner<T::Hooks>, AgentError> {
        let agent = self.agent(ctx).await?;
        Ok(agent.hooks::<T::Hooks>(&self.task().record.kind))
    }

    fn hook_api(&self) -> Result<HookApi<'_>, AgentError> {
        let models = self.agent_services()?.models().clone();
        Ok(HookApi::new(self.task_id(), self.conversation_id(), models, self))
    }

    fn report_hook(&self, error: &(dyn std::error::Error + Send + Sync), extension: &str) {
        self.report(error, Some(ReportFrom::Extension(Some(extension.to_owned()))));
    }
}

/// Reports a failure of a host callback (the environment builder, the models) from a task.
pub(crate) fn report_host<T: TaskDefinition>(
    rt: &TaskRuntime<T>,
    error: &(dyn std::error::Error + Send + Sync),
    callback: HostCallback,
) {
    rt.report(error, Some(ReportFrom::Host(callback)));
}
