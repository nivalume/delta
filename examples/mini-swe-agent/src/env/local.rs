//! Commands on this machine, through the harness's [`LocalEnv`].

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use delta_chord::context::Context;
use delta_harness::env::{Command, ExecErrorCode, ExecOptions, LocalEnv, Shell};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use super::{Environment, ExecOutput, serialized, timeout_message};

/// Upstream's `LocalEnvironmentConfig`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocalEnvironmentConfig {
    /// Where commands run; empty: the current directory.
    #[serde(default)]
    pub cwd: String,
    /// Variables set over the inherited ones.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Seconds before a command is killed.
    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

fn default_timeout() -> u64 {
    30
}

impl Default for LocalEnvironmentConfig {
    fn default() -> Self {
        Self {
            cwd: String::new(),
            env: BTreeMap::new(),
            timeout: default_timeout(),
        }
    }
}

/// Runs commands with the local shell; stdout and stderr are merged in arrival order.
#[derive(Debug)]
pub struct LocalEnvironment {
    config: LocalEnvironmentConfig,
    env: LocalEnv,
}

impl LocalEnvironment {
    pub fn new(config: LocalEnvironmentConfig) -> Self {
        let cwd = if config.cwd.is_empty() {
            std::env::current_dir().unwrap_or_else(|_| ".".into())
        } else {
            config.cwd.clone().into()
        };
        Self {
            env: LocalEnv::new(cwd),
            config,
        }
    }

    pub fn config(&self) -> &LocalEnvironmentConfig {
        &self.config
    }
}

impl Environment for LocalEnvironment {
    fn execute<'a>(
        &'a self,
        command: &'a str,
        cwd: Option<&'a str>,
        timeout: Option<Duration>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, ExecOutput> {
        Box::pin(async move {
            let timeout = timeout.unwrap_or(Duration::from_secs(self.config.timeout));
            let output = Arc::new(Mutex::new(String::new()));
            let sink = Arc::clone(&output);
            let mut options = ExecOptions::default().with_timeout(timeout).on_output(move |chunk| {
                sink.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push_str(&chunk.text);
            });
            options.env = self.config.env.clone();
            if let Some(cwd) = cwd.filter(|cwd| !cwd.is_empty()) {
                options.cwd = Some(cwd.to_owned());
            }
            let result = self.env.exec(Command::shell(command), options, ctx).await;
            let text = std::mem::take(&mut *output.lock().unwrap_or_else(std::sync::PoisonError::into_inner));
            match result {
                Ok(result) => ExecOutput {
                    output: text,
                    returncode: i64::from(result.exit_code),
                    ..ExecOutput::default()
                },
                Err(error) if error.code == ExecErrorCode::Timeout => {
                    ExecOutput::exception(text, "TimeoutExpired", &timeout_message(command, timeout))
                }
                Err(error) => {
                    let exception_type = match error.code {
                        ExecErrorCode::Aborted => "KeyboardInterrupt",
                        ExecErrorCode::SpawnError | ExecErrorCode::ShellUnavailable => "OSError",
                        _ => "RuntimeError",
                    };
                    ExecOutput::exception(text, exception_type, &error.message)
                }
            }
        })
    }

    fn template_vars(&self) -> Json {
        crate::template::vars([
            serde_json::to_value(&self.config).unwrap_or(Json::Null),
            crate::template::uname(),
            crate::template::environ(),
        ])
    }

    fn serialize(&self) -> Json {
        serialized(
            &serde_json::to_value(&self.config).unwrap_or(Json::Null),
            "minisweagent.environments.local.LocalEnvironment",
        )
    }
}
