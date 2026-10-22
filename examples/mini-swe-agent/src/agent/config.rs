//! The `agent` section.

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

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

/// Whether someone at a terminal takes part in the run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentKind {
    /// Runs on its own: no confirmations, no prompts (the SWE-bench batch).
    Default,
    /// Asks its frontend: confirmations, the end of the task, new limits, human mode.
    Interactive,
}

/// The `agent` section of a config.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentConfig {
    /// The system prompt.
    pub system_template: String,
    /// The first user message: the task.
    pub instance_template: String,
    /// Model calls allowed; 0: no limit.
    #[serde(default)]
    pub step_limit: u64,
    /// USD a run may spend; 0: no limit.
    #[serde(default = "default_cost_limit")]
    pub cost_limit: f64,
    /// Seconds a run may take; 0: no limit.
    #[serde(default)]
    pub time_limit: u64,
    /// Format errors in a row that end the run; 0: no limit.
    #[serde(default = "default_format_error_limit")]
    pub format_error_limit: u64,
    #[serde(default)]
    pub mode: Mode,
    /// Regular expressions; an action matching one from its start is never confirmed.
    #[serde(default)]
    pub whitelist: Vec<String>,
    /// Ask before finishing: a new task, or Enter to quit.
    #[serde(default = "default_true")]
    pub confirm_exit: bool,
}

fn default_cost_limit() -> f64 {
    3.0
}

fn default_format_error_limit() -> u64 {
    3
}

fn default_true() -> bool {
    true
}

impl AgentConfig {
    /// The config `section` holds.
    pub fn from_section(section: &Json) -> Result<Self, String> {
        serde_json::from_value(section.clone()).map_err(|error| format!("Invalid agent config: {error}"))
    }

    /// The config as the run uses it.
    pub fn dump(&self) -> Json {
        serde_json::to_value(self).unwrap_or(Json::Null)
    }
}
