//! Where actions run: this machine or a Docker container. `environment.type` picks one
//! (`local` by default); the rest of the section is that environment's config.

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

/// What one command produced.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecOutput {
    /// stdout and stderr, merged in arrival order.
    pub output: String,
    /// -1 when the command did not finish.
    pub returncode: i64,
    /// Why the command did not finish (a timeout, an interruption); empty otherwise.
    pub exception_info: String,
}

impl ExecOutput {
    /// A command that did not finish.
    pub fn exception(output: String, info: impl Into<String>) -> Self {
        Self {
            output,
            returncode: -1,
            exception_info: info.into(),
        }
    }

    /// The submission, when the output asks to submit: its first non-blank line is the
    /// marker and the command succeeded.
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

    /// The environment's section as it runs, `type` included: the template variable
    /// `environment` and the trajectory's `config.environment`.
    fn config(&self) -> Json;

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

/// The keys of the `environment` section that are not an environment's config.
const SECTION_KEYS: [&str; 2] = ["type", "startup_command"];

/// The environment type `section` names, `default_type` when it names none.
pub fn environment_type(section: &Json, default_type: &str) -> String {
    section["type"]
        .as_str()
        .filter(|kind| !kind.is_empty())
        .unwrap_or(default_type)
        .to_owned()
}

/// Builds the environment the `environment` section describes.
pub async fn get_environment(section: &Json, default_type: &str) -> Result<Arc<dyn Environment>, EnvironmentError> {
    let kind = environment_type(section, default_type);
    let mut config = match section {
        Json::Object(object) => object.clone(),
        _ => Map::new(),
    };
    for key in SECTION_KEYS {
        config.remove(key);
    }
    match kind.as_str() {
        "docker" => {
            let config: DockerEnvironmentConfig = serde_json::from_value(Json::Object(config))?;
            Ok(Arc::new(DockerEnvironment::start(config).await?))
        }
        "local" => {
            let config: LocalEnvironmentConfig = serde_json::from_value(Json::Object(config))?;
            Ok(Arc::new(LocalEnvironment::new(config)))
        }
        other => Err(EnvironmentError::UnknownType(other.to_owned())),
    }
}

/// `config` with its `type` first.
fn typed(kind: &str, config: Json) -> Json {
    let mut typed = Map::new();
    typed.insert("type".to_owned(), Json::String(kind.to_owned()));
    if let Json::Object(config) = config {
        typed.extend(config);
    }
    Json::Object(typed)
}

fn timeout_message(timeout: Duration) -> String {
    format!("The command timed out after {} seconds", timeout.as_secs_f64())
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

    #[tokio::test]
    async fn the_type_picks_the_environment() {
        let local = get_environment(&serde_json::json!({ "timeout": 5, "startup_command": "x" }), "local")
            .await
            .unwrap();
        assert_eq!(local.config()["type"], "local");
        assert_eq!(local.config()["timeout"], 5);
        assert!(matches!(
            get_environment(&serde_json::json!({ "type": "podman" }), "local").await,
            Err(EnvironmentError::UnknownType(_))
        ));
        assert!(matches!(
            get_environment(&serde_json::json!({ "environment_class": "local" }), "local").await,
            Err(EnvironmentError::Config(_))
        ));
        assert_eq!(
            timeout_message(Duration::from_millis(1500)),
            "The command timed out after 1.5 seconds"
        );
    }
}
