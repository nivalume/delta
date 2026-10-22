//! The command lines: `mini` and `mini-extra`.

mod account;
mod extra;
mod mini;

use std::path::PathBuf;
use std::sync::Arc;

use delta_ai::CredentialStore;
use delta_ai::model::Model;
use serde_json::{Map, Value as Json};

pub use extra::{ExtraArgs, run_extra};
pub use mini::{MiniArgs, MiniRun, mini_config, run_mini};

use crate::agent::{Agent, AgentConfig, AgentKind, AgentOptions, Frontend, StepCallback};
use crate::auth::FileCredentialStore;
use crate::config::{self, Overrides};
use crate::env::{Environment, get_environment};
use crate::model::{BatchStats, MiniModelClient, ModelSettings, has_price, split_model_ref};

/// An agent built from a merged config, with the environment it runs in (the caller cleans
/// it up).
pub struct Built {
    pub agent: Agent,
    pub env: Arc<dyn Environment>,
    pub model: ModelSettings,
    /// The delta-ai model the agent uses.
    pub resolved: Model,
}

impl std::fmt::Debug for Built {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Built")
            .field("agent", &self.agent)
            .finish_non_exhaustive()
    }
}

/// What [`build_agent`] needs beside the config.
pub struct BuildOptions {
    pub kind: AgentKind,
    /// The environment when `environment.type` names none.
    pub default_environment: &'static str,
    pub frontend: Option<Arc<dyn Frontend>>,
    pub handle_ctrl_c: bool,
    /// Where the trajectory is saved.
    pub output: Option<PathBuf>,
    /// The trajectory's `extra`.
    pub extra: Map<String, Json>,
    /// An environment already started, used instead of building one.
    pub env: Option<Arc<dyn Environment>>,
    pub on_step: Option<StepCallback>,
    pub batch: Option<Arc<BatchStats>>,
    /// The credential store; `None`: `auth.json` in the home directory.
    pub credentials: Option<Arc<dyn CredentialStore>>,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            kind: AgentKind::Default,
            default_environment: "local",
            frontend: None,
            handle_ctrl_c: false,
            output: None,
            extra: Map::new(),
            env: None,
            on_step: None,
            batch: None,
            credentials: None,
        }
    }
}

impl std::fmt::Debug for BuildOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuildOptions")
            .field("kind", &self.kind)
            .field("default_environment", &self.default_environment)
            .field("output", &self.output)
            .finish_non_exhaustive()
    }
}

/// The credential store of the command lines: `auth.json` in the home directory.
pub fn credential_store() -> Arc<dyn CredentialStore> {
    Arc::new(FileCredentialStore::new(crate::home::auth_file()))
}

/// The model, the environment and the agent `config` describes.
pub async fn build_agent(config: &Json, options: BuildOptions) -> Result<Built, String> {
    let model = ModelSettings::new(&config["model"]).map_err(|error| error.to_string())?;
    let agent_config = AgentConfig::from_section(&config["agent"])?;
    let credentials = options.credentials.unwrap_or_else(credential_store);
    let (client, warning) =
        MiniModelClient::new(&model, &config["providers"], Some(credentials)).map_err(|error| error.to_string())?;
    if let Some(warning) = warning {
        eprintln!("Warning: {warning}");
    }
    let resolved = client.resolved().clone();
    if agent_config.cost_limit > 0.0 && !model.is_scripted() && !has_price(&resolved) {
        eprintln!(
            "Warning: {}/{} has no price, so agent.cost_limit cannot end the run.",
            resolved.provider, resolved.id
        );
    }
    let env = match options.env {
        Some(env) => env,
        None => get_environment(&config["environment"], options.default_environment)
            .await
            .map_err(|error| error.to_string())?,
    };
    let agent = Agent::new(AgentOptions {
        config: config.clone(),
        kind: options.kind,
        model: model.clone(),
        resolved: resolved.clone(),
        client: Arc::new(client),
        env: Arc::clone(&env),
        frontend: options.frontend,
        output: options.output,
        extra: options.extra,
        handle_ctrl_c: options.handle_ctrl_c,
        on_step: options.on_step,
        batch: options.batch,
    })
    .map_err(|error| error.to_string())?;
    Ok(Built {
        agent,
        env,
        model,
        resolved,
    })
}

/// The config of a run: the global config, then `specs` (`default_spec` when there are
/// none).
pub fn load_config(specs: &[String], default_spec: &str) -> Result<Json, String> {
    let global = crate::home::global_config().map_err(|error| error.to_string())?;
    layered_config(global, specs, default_spec)
}

/// `specs` (`default_spec` when there are none) merged over `global`.
pub fn layered_config(global: Json, specs: &[String], default_spec: &str) -> Result<Json, String> {
    let specs = if specs.is_empty() {
        vec![default_spec.to_owned()]
    } else {
        specs.to_vec()
    };
    eprintln!("Building agent config from specs: {specs:?}");
    let configs = config::load(&specs).map_err(|error| error.to_string())?;
    Ok(config::merge_all([global, configs]))
}

/// `overrides` with the model `-m` names, as `provider/id` or a bare id.
pub fn with_model(overrides: Overrides, model: Option<&str>) -> Overrides {
    match model {
        Some(spec) => {
            let (provider, id) = split_model_ref(spec);
            overrides
                .set("model.provider", Some(provider))
                .set("model.id", Some(id))
        }
        None => overrides,
    }
}

/// Prints the version and the home directory.
pub fn startup_banner() {
    eprintln!(
        "delta-mini {} (home: {})",
        env!("CARGO_PKG_VERSION"),
        crate::home::home_dir().display()
    );
}

/// The default output of `mini`: `last_mini_run.traj.json` in the home directory.
pub fn default_mini_output() -> PathBuf {
    crate::home::home_dir().join("last_mini_run.traj.json")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn layers_go_global_specs_key_values_then_the_command_line() {
        let global = json!({
            "model": { "provider": "anthropic", "id": "claude-sonnet-4-5", "temperature": 0.1 },
            "agent": { "cost_limit": 9.0, "step_limit": 7 },
            "providers": { "local": { "api": "openai-completions", "base_url": "u", "models": [{ "id": "m" }] } },
        });
        let specs = ["mini.yaml".to_owned(), "agent.step_limit=3".to_owned()];
        let mut config = layered_config(global, &specs, "unused.yaml").unwrap();
        // mini.yaml sets cost_limit and mode; the global config keeps what nothing else sets.
        assert_eq!(config["agent"]["cost_limit"], 3.0);
        assert_eq!(config["agent"]["step_limit"], 3);
        assert_eq!(config["model"]["temperature"], 0.1);
        assert_eq!(config["providers"]["local"]["models"][0]["id"], "m");
        with_model(Overrides::default(), Some("openai/gpt-5"))
            .set("agent.step_limit", Some(1))
            .apply(&mut config);
        assert_eq!(
            (
                &config["model"]["provider"],
                &config["model"]["id"],
                &config["agent"]["step_limit"]
            ),
            (&json!("openai"), &json!("gpt-5"), &json!(1))
        );
        let default = layered_config(Json::Null, &[], "mini_textbased.yaml").unwrap();
        assert_eq!(default["model"]["actions"], "text");
    }
}
