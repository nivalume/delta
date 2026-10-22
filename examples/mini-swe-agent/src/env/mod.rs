//! Where actions run: the local machine or a Docker container (upstream's
//! `environments/local.py` and `environments/docker.py`).

mod docker;
mod local;

use std::sync::Arc;
use std::time::Duration;

use delta_chord::context::Context;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

pub use docker::{DockerEnvironment, DockerEnvironmentConfig};
pub use local::{LocalEnvironment, LocalEnvironmentConfig};

/// The first output line that submits: the rest of the output is the submission.
pub const SUBMIT_MARKER: &str = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT";

/// What one command produced, as upstream's `execute` returns it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecOutput {
    pub output: String,
    pub returncode: i64,
    pub exception_info: String,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Json>,
}

impl ExecOutput {
    /// A command that did not finish: upstream's `except Exception` branch.
    pub fn exception(output: String, exception_type: &str, exception: &str) -> Self {
        let mut extra = Map::new();
        extra.insert("exception_type".to_owned(), Json::String(exception_type.to_owned()));
        extra.insert("exception".to_owned(), Json::String(exception.to_owned()));
        Self {
            output,
            returncode: -1,
            exception_info: format!("An error occurred while executing the command: {exception}"),
            extra,
        }
    }

    /// The submission, when the output asks to submit: its first non-blank line is the
    /// marker and the command succeeded (upstream's `_check_finished`).
    pub fn submission(&self) -> Option<String> {
        if self.returncode != 0 {
            return None;
        }
        let output = self.output.trim_start();
        let mut lines = output.split_inclusive('\n');
        let first = lines.next()?;
        (first.trim() == SUBMIT_MARKER).then(|| lines.collect())
    }

    /// The output as a template value.
    pub fn to_json(&self) -> Json {
        serde_json::to_value(self).unwrap_or(Json::Null)
    }
}

/// Where the agent's commands run.
pub trait Environment: Send + Sync {
    /// Runs `command` in `cwd` (the configured one when `None`), killed after `timeout` (the
    /// configured one when `None`). Never fails: a failure is an output with return code -1.
    fn execute<'a>(
        &'a self,
        command: &'a str,
        cwd: Option<&'a str>,
        timeout: Option<Duration>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, ExecOutput>;

    /// The environment's template variables.
    fn template_vars(&self) -> Json;

    /// Its part of the trajectory: `info.config.environment` and `environment_type`.
    fn serialize(&self) -> Json;

    /// Stops what the environment started.
    fn cleanup(&self) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
}

/// Why an environment could not be built.
#[derive(Debug, thiserror::Error)]
pub enum EnvironmentError {
    #[error("Unknown environment type: {0} (available: local, docker)")]
    UnknownType(String),
    #[error("Invalid environment config: {0}")]
    Config(#[from] serde_json::Error),
    #[error("Could not start the container: {0}")]
    Start(String),
}

/// The environment kind `environment_class` names: a short name or upstream's class path.
pub fn environment_kind(class: &str) -> Result<&'static str, EnvironmentError> {
    match class {
        "local" | "minisweagent.environments.local.LocalEnvironment" => Ok("local"),
        "docker" | "minisweagent.environments.docker.DockerEnvironment" => Ok("docker"),
        other => Err(EnvironmentError::UnknownType(other.to_owned())),
    }
}

/// Builds the environment `config` describes; `environment_class` falls back to
/// `default_type`.
pub async fn get_environment(config: &Json, default_type: &str) -> Result<Arc<dyn Environment>, EnvironmentError> {
    let mut config = match config {
        Json::Object(object) => object.clone(),
        _ => Map::new(),
    };
    let class = match config.remove("environment_class") {
        Some(Json::String(class)) if !class.is_empty() => class,
        _ => default_type.to_owned(),
    };
    match environment_kind(&class)? {
        "docker" => {
            let config: DockerEnvironmentConfig = serde_json::from_value(Json::Object(config))?;
            Ok(Arc::new(DockerEnvironment::start(config).await?))
        }
        _ => {
            let config: LocalEnvironmentConfig = serde_json::from_value(Json::Object(config))?;
            Ok(Arc::new(LocalEnvironment::new(config)))
        }
    }
}

/// `{"info": {"config": {"environment": config, "environment_type": type}}}`.
fn serialized(config: &Json, environment_type: &str) -> Json {
    serde_json::json!({
        "info": {
            "config": {
                "environment": config,
                "environment_type": environment_type,
            }
        }
    })
}

/// How Python prints a `TimeoutExpired`.
fn timeout_message(command: &str, timeout: Duration) -> String {
    format!(
        "Command '{command}' timed out after {} seconds",
        crate::template::python_float(timeout.as_secs_f64()).trim_end_matches(".0")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn output(text: &str, returncode: i64) -> ExecOutput {
        ExecOutput {
            output: text.to_owned(),
            returncode,
            ..ExecOutput::default()
        }
    }

    #[test]
    fn submissions_need_the_marker_first_and_success() {
        assert_eq!(
            output("\n  COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a\n+x\n", 0).submission(),
            Some("diff --git a\n+x\n".to_owned())
        );
        assert_eq!(
            output("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", 0).submission(),
            Some(String::new())
        );
        assert_eq!(output("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", 1).submission(), None);
        assert_eq!(
            output("x\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", 0).submission(),
            None
        );
        assert_eq!(output("", 0).submission(), None);
    }

    #[test]
    fn timeouts_print_as_python() {
        assert_eq!(
            timeout_message("sleep 9", Duration::from_secs(30)),
            "Command 'sleep 9' timed out after 30 seconds"
        );
        assert_eq!(
            timeout_message("x", Duration::from_millis(1500)),
            "Command 'x' timed out after 1.5 seconds"
        );
    }
}
