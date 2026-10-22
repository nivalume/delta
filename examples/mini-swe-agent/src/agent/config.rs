//! The `agent` section: upstream's `AgentConfig` and `InteractiveAgentConfig`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

/// Whether the user confirms the model's actions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// The user issues the commands.
    Human,
    /// The user confirms actions not whitelisted.
    #[default]
    Confirm,
    /// Actions run without confirmation.
    Yolo,
}

impl Mode {
    /// The mode a slash command switches to.
    pub fn from_command(command: &str) -> Option<Self> {
        match command {
            "/u" => Some(Self::Human),
            "/c" => Some(Self::Confirm),
            "/y" => Some(Self::Yolo),
            _ => None,
        }
    }
}

/// Which agent runs: upstream's `DefaultAgent` or `InteractiveAgent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentKind {
    Default,
    Interactive,
}

impl AgentKind {
    /// The kind `agent_class` names: a short name or upstream's class path.
    pub fn parse(class: &str) -> Result<Self, String> {
        match class {
            "default" | "minisweagent.agents.default.DefaultAgent" => Ok(Self::Default),
            "interactive" | "minisweagent.agents.interactive.InteractiveAgent" => Ok(Self::Interactive),
            other => Err(format!("Unknown agent type: {other} (available: default, interactive)")),
        }
    }

    /// Upstream's class path, for `info.config.agent_type`.
    pub fn path(self) -> &'static str {
        match self {
            Self::Default => "minisweagent.agents.default.DefaultAgent",
            Self::Interactive => "minisweagent.agents.interactive.InteractiveAgent",
        }
    }
}

/// The `agent` section of a config.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentConfig {
    /// The system message.
    pub system_template: String,
    /// The first user message: the task.
    pub instance_template: String,
    /// Model calls allowed; 0: no limit.
    #[serde(default)]
    pub step_limit: u64,
    /// Stop once the cost reaches this; 0: no limit.
    #[serde(default = "default_cost_limit")]
    pub cost_limit: f64,
    /// Seconds of wall-clock time allowed; 0: no limit.
    #[serde(default)]
    pub wall_time_limit_seconds: u64,
    /// Format errors in a row that end the run; 0: no limit.
    #[serde(default = "default_max_consecutive_format_errors")]
    pub max_consecutive_format_errors: u64,
    /// Where the trajectory is saved after every step.
    #[serde(default)]
    pub output_path: Option<String>,
    #[serde(default)]
    pub mode: Mode,
    /// Regular expressions; an action matching one from its start is never confirmed.
    #[serde(default)]
    pub whitelist_actions: Vec<String>,
    /// Ask before finishing: a new task, or Enter to quit.
    #[serde(default = "default_true")]
    pub confirm_exit: bool,
    #[serde(default, skip_serializing)]
    pub agent_class: String,
}

fn default_cost_limit() -> f64 {
    3.0
}

fn default_max_consecutive_format_errors() -> u64 {
    3
}

fn default_true() -> bool {
    true
}

impl AgentConfig {
    /// The config `section` holds, with its `agent_class` defaulting to `default_kind`.
    pub fn from_section(section: &Json, default_kind: AgentKind) -> Result<(Self, AgentKind), String> {
        let config: Self =
            serde_json::from_value(section.clone()).map_err(|error| format!("Invalid agent config: {error}"))?;
        let kind = if config.agent_class.is_empty() {
            default_kind
        } else {
            AgentKind::parse(&config.agent_class)?
        };
        Ok((config, kind))
    }

    /// The config as upstream's `model_dump` prints it for `kind`.
    pub fn dump(&self, kind: AgentKind) -> Json {
        let mut dump = serde_json::to_value(self).unwrap_or(Json::Null);
        if let (AgentKind::Default, Json::Object(object)) = (kind, &mut dump) {
            for key in ["mode", "whitelist_actions", "confirm_exit"] {
                object.remove(key);
            }
        }
        dump
    }

    /// The agent's part of `info.config`.
    pub fn serialize(&self, kind: AgentKind) -> Json {
        let mut config = Map::new();
        config.insert("agent".to_owned(), self.dump(kind));
        config.insert("agent_type".to_owned(), Json::String(kind.path().to_owned()));
        Json::Object(config)
    }
}
