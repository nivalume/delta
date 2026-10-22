//! Commands in a Docker container, started once and stopped at cleanup.

use std::collections::BTreeMap;
use std::io::Read as _;
use std::process::Stdio;
use std::sync::Mutex;
use std::time::Duration;

use delta_chord::context::Context;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use super::{Environment, EnvironmentError, ExecOutput, timeout_message, typed};

/// The `environment` section of type `docker`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DockerEnvironmentConfig {
    pub image: String,
    /// Where commands run in the container.
    #[serde(default = "default_cwd")]
    pub cwd: String,
    /// Variables set in the container; they win over `forward_env`.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Host variables forwarded when set.
    #[serde(default)]
    pub forward_env: Vec<String>,
    /// Seconds before a command is killed.
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    /// The docker (or compatible) executable.
    #[serde(default = "default_executable")]
    pub executable: String,
    #[serde(default = "default_run_args")]
    pub run_args: Vec<String>,
    /// How long the container lives, in `sleep`'s format.
    #[serde(default = "default_container_timeout")]
    pub container_timeout: String,
    /// Seconds allowed for `docker run`, which may pull the image.
    #[serde(default = "default_pull_timeout")]
    pub pull_timeout: u64,
    /// What runs a command; the command is its last argument.
    #[serde(default = "default_interpreter")]
    pub interpreter: Vec<String>,
}

fn default_cwd() -> String {
    "/".to_owned()
}

fn default_timeout() -> u64 {
    30
}

fn default_executable() -> String {
    "docker".to_owned()
}

fn default_run_args() -> Vec<String> {
    vec!["--rm".to_owned()]
}

fn default_container_timeout() -> String {
    "2h".to_owned()
}

fn default_pull_timeout() -> u64 {
    120
}

fn default_interpreter() -> Vec<String> {
    vec!["bash".to_owned(), "-lc".to_owned()]
}

/// A running container.
#[derive(Debug)]
pub struct DockerEnvironment {
    config: DockerEnvironmentConfig,
    container_id: Mutex<Option<String>>,
}

impl DockerEnvironment {
    /// Starts the container: `docker run -d --name delta-mini-<8 hex> -w <cwd> <run_args>
    /// <image> sleep <container_timeout>`.
    pub async fn start(config: DockerEnvironmentConfig) -> Result<Self, EnvironmentError> {
        let name = format!("delta-mini-{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
        let mut command = tokio::process::Command::new(&config.executable);
        command
            .args(["run", "-d", "--name", &name, "-w", &config.cwd])
            .args(&config.run_args)
            .args([config.image.as_str(), "sleep", config.container_timeout.as_str()])
            .stdin(Stdio::null())
            .kill_on_drop(true);
        let output = tokio::time::timeout(Duration::from_secs(config.pull_timeout), command.output())
            .await
            .map_err(|_| {
                EnvironmentError::Start(format!(
                    "`{} run` timed out after {} seconds",
                    config.executable, config.pull_timeout
                ))
            })?
            .map_err(|error| EnvironmentError::Start(format!("{}: {error}", config.executable)))?;
        if !output.status.success() {
            return Err(EnvironmentError::Start(format!(
                "`{} run` exited with {}: {}",
                config.executable,
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        let container_id = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        Ok(Self {
            config,
            container_id: Mutex::new(Some(container_id)),
        })
    }

    pub fn config(&self) -> &DockerEnvironmentConfig {
        &self.config
    }

    pub fn container_id(&self) -> Option<String> {
        self.container_id
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// `docker exec -w <cwd> -e K=V... <id> <interpreter...> <command>`.
    fn exec_args(&self, container_id: &str, command: &str, cwd: &str) -> Vec<String> {
        let mut args = vec!["exec".to_owned(), "-w".to_owned(), cwd.to_owned()];
        for key in &self.config.forward_env {
            if let Ok(value) = std::env::var(key) {
                args.extend(["-e".to_owned(), format!("{key}={value}")]);
            }
        }
        for (key, value) in &self.config.env {
            args.extend(["-e".to_owned(), format!("{key}={value}")]);
        }
        args.push(container_id.to_owned());
        args.extend(self.config.interpreter.iter().cloned());
        args.push(command.to_owned());
        args
    }
}

/// Runs `executable args` with stdout and stderr on one pipe. `Ok(None)` when the timeout or
/// `ctx` killed it.
async fn run_merged(
    executable: &str,
    args: &[String],
    timeout: Duration,
    ctx: &Context,
) -> std::io::Result<(Option<i64>, String)> {
    let (mut reader, writer) = std::io::pipe()?;
    let mut command = tokio::process::Command::new(executable);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(writer.try_clone()?)
        .stderr(writer)
        .kill_on_drop(true);
    let mut child = command.spawn()?;
    // The command holds the parent's ends of the pipe; dropping it leaves the child's.
    drop(command);
    let reading = tokio::task::spawn_blocking(move || {
        let mut bytes = Vec::new();
        let _ = reader.read_to_end(&mut bytes);
        bytes
    });
    let status = tokio::select! {
        status = child.wait() => Some(status?),
        () = tokio::time::sleep(timeout) => None,
        () = ctx.cancelled() => None,
    };
    if status.is_none() {
        let _ = child.kill().await;
    }
    let bytes = reading.await.unwrap_or_default();
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let code = status.map(|status| i64::from(status.code().unwrap_or(-1)));
    Ok((code, text))
}

impl Environment for DockerEnvironment {
    fn execute<'a>(
        &'a self,
        command: &'a str,
        cwd: Option<&'a str>,
        timeout: Option<Duration>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, ExecOutput> {
        Box::pin(async move {
            let Some(container_id) = self.container_id() else {
                return ExecOutput::exception(String::new(), "The container is not running");
            };
            let cwd = cwd.filter(|cwd| !cwd.is_empty()).unwrap_or(&self.config.cwd);
            let timeout = timeout.unwrap_or(Duration::from_secs(self.config.timeout));
            let args = self.exec_args(&container_id, command, cwd);
            match run_merged(&self.config.executable, &args, timeout, ctx).await {
                Ok((Some(returncode), output)) => ExecOutput {
                    output,
                    returncode,
                    ..ExecOutput::default()
                },
                Ok((None, output)) if ctx.is_cancelled() => {
                    ExecOutput::exception(output, "The command was interrupted")
                }
                Ok((None, output)) => ExecOutput::exception(output, timeout_message(timeout)),
                Err(error) => ExecOutput::exception(String::new(), format!("The command could not run: {error}")),
            }
        })
    }

    fn config(&self) -> Json {
        typed("docker", serde_json::to_value(&self.config).unwrap_or(Json::Null))
    }

    /// Stops the container: `(timeout 60 docker stop <id> || docker rm -f <id>)`, in the
    /// background on Unix, so it outlives the process.
    fn cleanup(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            let Some(container_id) = self
                .container_id
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
            else {
                return;
            };
            let executable = &self.config.executable;
            if cfg!(unix) {
                let script = format!(
                    "(timeout 60 {executable} stop {container_id} || {executable} rm -f {container_id}) >/dev/null 2>&1 &"
                );
                let _ = std::process::Command::new("sh")
                    .args(["-c", &script])
                    .stdin(Stdio::null())
                    .status();
                return;
            }
            let quiet = |args: &[&str]| {
                let mut command = tokio::process::Command::new(executable);
                command
                    .args(args)
                    .arg(&container_id)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null());
                command
            };
            let stopped = matches!(
                tokio::time::timeout(Duration::from_secs(60), quiet(&["stop"]).status()).await,
                Ok(Ok(status)) if status.success()
            );
            if !stopped {
                let _ = quiet(&["rm", "-f"]).status().await;
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exec_lines_hold_the_config() {
        let config: DockerEnvironmentConfig = serde_json::from_value(serde_json::json!({
            "image": "python:3.11",
            "cwd": "/testbed",
            "env": { "PAGER": "cat" },
            "forward_env": ["DELTA_MINI_SURELY_UNSET"],
            "interpreter": ["bash", "-c"],
        }))
        .unwrap();
        assert_eq!(config.run_args, ["--rm"]);
        assert_eq!(config.container_timeout, "2h");
        let environment = DockerEnvironment {
            config,
            container_id: Mutex::new(Some("abc".to_owned())),
        };
        assert_eq!(
            environment.exec_args("abc", "ls -la", "/testbed"),
            [
                "exec",
                "-w",
                "/testbed",
                "-e",
                "PAGER=cat",
                "abc",
                "bash",
                "-c",
                "ls -la"
            ]
        );
    }
}
