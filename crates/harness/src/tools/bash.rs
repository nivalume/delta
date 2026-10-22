//! `bash`: a command through the environment's shell, its output's tail retained.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use delta_ai::types::{JsonObject, Tool};
use delta_chord::context::Context;
use delta_chord::output::{OutputLimits, Retain};
use futures::future::BoxFuture;
use serde::Deserialize;
use serde_json::Value;

use super::truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES};
use super::{js_number, parse, require_env};
use crate::env::{Command, ExecErrorCode, ExecOptions, ExecutionEnv, MAX_TIMEOUT, OutputChunk, SpillOptions};
use crate::tool::{
    DEFAULT_OUTPUT_LIMITS, HarnessTool, PermissionSubject, Severity, ToolApi, ToolDiagnostic, ToolError,
    ToolExecutionResult,
};

/// A command about to run, which [`BashPrepare`] may change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BashExecution {
    /// The script, after the command prefix.
    pub command: String,
    pub cwd: String,
    /// Variables set for the command.
    pub env: BTreeMap<String, String>,
    /// Start from the environment's variables.
    pub inherit_env: bool,
}

/// Changes each command before it runs: a sandbox wrapper, extra variables, another
/// directory. An error fails the call.
pub trait BashPrepare: Send + Sync {
    fn prepare<'a>(
        &'a self,
        execution: &'a mut BashExecution,
        api: &'a ToolApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<(), ToolError>>;
}

/// How [`BashTool`] runs commands.
#[derive(Clone, Default)]
pub struct BashOptions {
    /// Lines run before each command.
    pub command_prefix: Option<String>,
    pub prepare: Option<Arc<dyn BashPrepare>>,
}

impl fmt::Debug for BashOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BashOptions")
            .field("command_prefix", &self.command_prefix)
            .field("prepare", &self.prepare.is_some())
            .finish()
    }
}

/// The `bash` tool. Its output streams into a tail window; programs get
/// `{ output, truncated, fullOutputPath?, exitCode }`. A nonzero exit is an error result
/// with an `exit_code` diagnostic; output past the limits is spilled to a file named by a
/// `full_output` diagnostic.
#[derive(Debug)]
pub struct BashTool {
    definition: Tool,
    options: BashOptions,
    structured_output: Value,
}

impl BashTool {
    pub fn new(options: BashOptions) -> Self {
        let parameters = serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Bash command to execute" },
                "timeout": { "type": "number", "description": "Timeout in seconds (optional, no default timeout)" },
            },
            "required": ["command"],
        });
        let structured_output = serde_json::json!({
            "type": "object",
            "properties": {
                "output": { "type": "string", "description": "Combined stdout and stderr: the retained tail, as the model sees it" },
                "truncated": { "type": "boolean", "description": "Whether earlier output was dropped" },
                "fullOutputPath": { "type": "string", "description": "File with the complete output, when it was spilled" },
                "exitCode": { "type": "number" },
            },
            "required": ["output", "truncated", "exitCode"],
        });
        let description = format!(
            "Execute a bash command in the current working directory. Returns combined stdout and stderr. Output is truncated to last {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.",
            DEFAULT_MAX_BYTES / 1024
        );
        Self {
            definition: Tool::new("bash", description, parameters),
            options,
            structured_output,
        }
    }
}

#[derive(Deserialize)]
struct BashArguments {
    command: String,
    #[serde(default)]
    timeout: Option<f64>,
}

/// The timeout in seconds as a duration: finite, positive, at most [`MAX_TIMEOUT`].
fn validate_timeout(timeout: Option<f64>) -> Result<Option<Duration>, ToolError> {
    let Some(seconds) = timeout else { return Ok(None) };
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("Invalid timeout: must be a finite number of seconds".into());
    }
    if seconds > MAX_TIMEOUT.as_secs_f64() {
        return Err(format!("Invalid timeout: maximum is {} seconds", MAX_TIMEOUT.as_secs_f64()).into());
    }
    Ok(Some(Duration::from_secs_f64(seconds)))
}

impl HarnessTool for BashTool {
    fn definition(&self) -> &Tool {
        &self.definition
    }

    fn output_limits(&self) -> OutputLimits {
        OutputLimits {
            retain: Retain::Tail,
            ..DEFAULT_OUTPUT_LIMITS
        }
    }

    fn structured_output_schema(&self) -> Option<&Value> {
        Some(&self.structured_output)
    }

    fn permission_subject(&self, arguments: &JsonObject, _env: Option<&dyn ExecutionEnv>) -> Option<PermissionSubject> {
        let command = arguments.get("command")?.as_str()?;
        Some(PermissionSubject::Command {
            command: command.to_owned(),
        })
    }

    fn execute<'a>(
        &'a self,
        arguments: JsonObject,
        api: &'a ToolApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<ToolExecutionResult, ToolError>> {
        Box::pin(async move {
            let BashArguments { command, timeout } = parse(arguments)?;
            let duration = validate_timeout(timeout)?;
            let env = require_env(api)?;
            let mut execution = BashExecution {
                command: match &self.options.command_prefix {
                    Some(prefix) if !prefix.is_empty() => format!("{prefix}\n{command}"),
                    _ => command,
                },
                cwd: env.cwd().to_owned(),
                env: BTreeMap::new(),
                inherit_env: true,
            };
            if let Some(prepare) = &self.options.prepare {
                prepare.prepare(&mut execution, api, ctx).await?;
            }
            let options = ExecOptions {
                cwd: Some(execution.cwd),
                env: execution.env,
                inherit_env: execution.inherit_env,
                timeout: duration,
                on_output: Some(Box::new(|chunk: OutputChunk| {
                    if let Some(skipped) = chunk.skipped {
                        api.output_skipped(skipped);
                    }
                    api.output_str(&chunk.text);
                })),
                spill: Some(SpillOptions {
                    after_bytes: DEFAULT_MAX_BYTES as u64,
                    after_lines: DEFAULT_MAX_LINES as u64,
                }),
                // The environment may then omit output outside the retained tail.
                window: api.output_window(),
                stdin: None,
            };
            let result = env.exec(Command::Shell(execution.command), options, ctx).await;
            let spill_path = match &result {
                Ok(result) => result.spill_path.clone(),
                Err(error) => error.spill_path.clone(),
            };
            if let Some(path) = &spill_path {
                api.diagnostic(
                    ToolDiagnostic::new(Severity::Info, format!("Full output: {path}")).with_code("full_output"),
                );
            }
            let exit_code = match result {
                Ok(result) => result.exit_code,
                // Cancelled with the call: the task ends it, whatever is returned.
                Err(error) if error.code == ExecErrorCode::Aborted && ctx.is_cancelled() => return Err(error.into()),
                Err(error) if error.code == ExecErrorCode::Timeout => {
                    let seconds = js_number(timeout.unwrap_or_default());
                    return Err(format!("Command timed out after {seconds} seconds").into());
                }
                Err(error) if error.code == ExecErrorCode::Aborted => return Err("Command aborted".into()),
                Err(error) => return Err(error.into()),
            };
            let retained = api.retained_output();
            let mut structured = serde_json::json!({
                "output": retained.text,
                "truncated": retained.truncated,
                "exitCode": exit_code,
            });
            if let Some(path) = spill_path {
                structured["fullOutputPath"] = Value::from(path);
            }
            let mut result = ToolExecutionResult {
                structured_output: Some(structured),
                ..ToolExecutionResult::default()
            };
            if exit_code != 0 {
                result.is_error = true;
                result.diagnostics.push(ToolDiagnostic::error(
                    "exit_code",
                    format!("Command exited with code {exit_code}"),
                ));
            }
            Ok(result)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeouts_are_checked() {
        assert_eq!(validate_timeout(None).unwrap(), None);
        assert_eq!(validate_timeout(Some(1.5)).unwrap(), Some(Duration::from_millis(1500)));
        assert_eq!(
            validate_timeout(Some(0.0)).unwrap_err().to_string(),
            "Invalid timeout: must be a finite number of seconds"
        );
        assert_eq!(
            validate_timeout(Some(3e6)).unwrap_err().to_string(),
            "Invalid timeout: maximum is 2147483.647 seconds"
        );
    }
}
