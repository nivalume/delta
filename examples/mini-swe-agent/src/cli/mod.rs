//! The command lines: `mini` and `mini-extra`.

mod extra;
mod mini;

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value as Json;

pub use extra::{ExtraArgs, run_extra};
pub use mini::{MiniArgs, mini_config, run_mini};

use crate::agent::{Agent, AgentKind, AgentOptions, Frontend, StepCallback};
use crate::env::{Environment, get_environment};
use crate::model::{MiniModelClient, ModelSettings};

/// An agent built from a merged config, with the environment it runs in (the caller cleans
/// it up).
pub struct Built {
    pub agent: Agent,
    pub env: Arc<dyn Environment>,
    pub model: ModelSettings,
}

impl std::fmt::Debug for Built {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Built")
            .field("agent", &self.agent)
            .finish_non_exhaustive()
    }
}

/// What [`build_agent`] needs beside the config.
#[derive(Default)]
pub struct BuildOptions {
    /// The agent when `agent.agent_class` names none.
    pub default_kind: Option<AgentKind>,
    /// The environment when `environment.environment_class` names none.
    pub default_environment: &'static str,
    pub frontend: Option<Arc<dyn Frontend>>,
    pub handle_ctrl_c: bool,
    pub extra_info: Vec<Json>,
    /// An environment already started, used instead of building one.
    pub env: Option<Arc<dyn Environment>>,
    pub on_step: Option<StepCallback>,
}

impl std::fmt::Debug for BuildOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuildOptions")
            .field("default_kind", &self.default_kind)
            .field("default_environment", &self.default_environment)
            .finish_non_exhaustive()
    }
}

/// The model, the environment and the agent `config` describes (upstream's `get_model`,
/// `get_environment` and `get_agent`).
pub async fn build_agent(config: &Json, options: BuildOptions) -> Result<Built, String> {
    let model = ModelSettings::new(&config["model"], None).map_err(|error| error.to_string())?;
    if !model.kwargs.ignored.is_empty() {
        eprintln!(
            "Ignoring model_kwargs delta-ai does not use: {}",
            model.kwargs.ignored.join(", ")
        );
    }
    let client = MiniModelClient::new(&model).map_err(|error| error.to_string())?;
    let model_ref = client.reference();
    let env = match options.env {
        Some(env) => env,
        None => get_environment(&config["environment"], options.default_environment)
            .await
            .map_err(|error| error.to_string())?,
    };
    let agent = Agent::new(AgentOptions {
        config: config["agent"].clone(),
        default_kind: options.default_kind.unwrap_or(AgentKind::Default),
        model: model.clone(),
        client: Arc::new(client),
        model_ref,
        env: Arc::clone(&env),
        frontend: options.frontend,
        extra_info: options.extra_info,
        handle_ctrl_c: options.handle_ctrl_c,
        on_step: options.on_step,
    })
    .map_err(|error| error.to_string())?;
    Ok(Built { agent, env, model })
}

/// Prints upstream's startup line unless `MSWEA_SILENT_STARTUP` is set.
pub fn startup_banner() {
    if crate::global::env_var("MSWEA_SILENT_STARTUP").is_some() {
        return;
    }
    eprintln!(
        "This is mini-swe-agent on delta-harness, version {}.\nLoading global config from '{}'",
        crate::trajectory::MINI_VERSION,
        crate::global::global_config_file().display()
    );
}

/// The default output of `mini`: `last_mini_run.traj.json` in the global config directory.
pub fn default_mini_output() -> PathBuf {
    crate::global::global_config_dir().join("last_mini_run.traj.json")
}
