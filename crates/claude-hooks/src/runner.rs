//! Running an event's handlers: commands through the conversation's environment, HTTP posts,
//! in parallel, with the outcomes memoized in the hook's task.

use std::collections::BTreeMap;
use std::time::Duration;

use delta_chord::context::Context;
use delta_harness::env::{Command, ExecOptions, OutputStream};
use delta_harness::registry::{HookApi, HookError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::HookCommand;
use crate::protocol::HookOutput;

/// What one handler returned.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Outcome {
    /// Exit 0 or a 2xx response: the JSON output, or the plain text a command printed.
    Output {
        #[serde(default)]
        output: HookOutput,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
    },
    /// Exit 2: blocks with stderr as the reason.
    Blocked { reason: String },
    /// Anything else: a non-blocking error.
    Failed { error: String },
}

/// The memo an event's outcomes are kept in, so a rerun of the phase does not run the
/// handlers again.
fn memo_name(event: &str) -> String {
    format!("claude-hooks:{event}")
}

/// Runs `handlers` of `event` with `input` and returns their outcomes in handler order;
/// a rerun returns the memoized ones.
pub(crate) async fn run_handlers(
    client: &reqwest::Client,
    api: &HookApi<'_>,
    event: &str,
    handlers: &[&HookCommand],
    input: &Value,
    ctx: &Context,
) -> Result<(Vec<Outcome>, bool), HookError> {
    let name = memo_name(event);
    if let Some(outcomes) = api.memo::<Vec<Outcome>>(&name)? {
        return Ok((outcomes, false));
    }
    if handlers.is_empty() {
        return Ok((Vec::new(), false));
    }
    let body = serde_json::to_vec(input)?;
    let runs = handlers.iter().map(|handler| {
        let timeout = handler.timeout(event);
        let body = &body;
        async move {
            match handler {
                HookCommand::Command { command, .. } => run_command(api, command, timeout, body, ctx).await,
                HookCommand::Http {
                    url,
                    headers,
                    allowed_env_vars,
                    ..
                } => run_http(client, url, headers, allowed_env_vars, timeout, input, ctx).await,
            }
        }
    });
    let outcomes = futures::future::join_all(runs).await;
    Ok((api.memo_or_insert(&name, outcomes, ctx).await?, true))
}

/// stdout as an output: a JSON object, or plain text; nothing for empty output.
fn parse_output(stdout: &str) -> Outcome {
    let trimmed = stdout.trim();
    if trimmed.starts_with('{')
        && let Ok(output) = serde_json::from_str::<HookOutput>(trimmed)
    {
        return Outcome::Output { output, text: None };
    }
    Outcome::Output {
        output: HookOutput::default(),
        text: (!trimmed.is_empty()).then(|| trimmed.to_owned()),
    }
}

async fn run_command(api: &HookApi<'_>, command: &str, timeout: Duration, input: &[u8], ctx: &Context) -> Outcome {
    let env = match api.env(ctx).await {
        Ok(Some(env)) => env,
        Ok(None) => {
            return Outcome::Failed {
                error: "No execution environment to run the hook command in".to_owned(),
            };
        }
        Err(error) => {
            return Outcome::Failed {
                error: error.to_string(),
            };
        }
    };
    let (mut stdout, mut stderr) = (String::new(), String::new());
    let options = ExecOptions::default()
        .with_env("CLAUDE_PROJECT_DIR", env.cwd())
        .with_timeout(timeout)
        .with_stdin(input)
        .on_output(|chunk| match chunk.stream {
            OutputStream::Stdout => stdout.push_str(&chunk.text),
            OutputStream::Stderr => stderr.push_str(&chunk.text),
        });
    let result = env.exec(Command::shell(command), options, ctx).await;
    match result {
        Ok(result) if result.exit_code == 0 => parse_output(&stdout),
        Ok(result) if result.exit_code == 2 => Outcome::Blocked {
            reason: match stderr.trim() {
                "" => "Blocked by hook".to_owned(),
                reason => reason.to_owned(),
            },
        },
        Ok(result) => Outcome::Failed {
            error: format!("Hook command exited with code {}: {}", result.exit_code, stderr.trim()),
        },
        Err(error) => Outcome::Failed {
            error: error.to_string(),
        },
    }
}

/// `value` with `$VAR` and `${VAR}` of the allowed variables replaced by their values
/// (empty when unset).
fn interpolate(value: &str, allowed: &[String]) -> String {
    let mut out = value.to_owned();
    for name in allowed {
        let replacement = std::env::var(name).unwrap_or_default();
        out = out.replace(&format!("${{{name}}}"), &replacement);
        out = out.replace(&format!("${name}"), &replacement);
    }
    out
}

async fn run_http(
    client: &reqwest::Client,
    url: &str,
    headers: &BTreeMap<String, String>,
    allowed: &[String],
    timeout: Duration,
    input: &Value,
    ctx: &Context,
) -> Outcome {
    let mut request = client.post(url).timeout(timeout).json(input);
    for (name, value) in headers {
        request = request.header(name, interpolate(value, allowed));
    }
    let sent = async {
        let response = request.send().await?;
        let status = response.status();
        let body = response.text().await?;
        Ok::<_, reqwest::Error>((status, body))
    };
    let (status, body) = match ctx.run(sent).await {
        Err(_) => {
            return Outcome::Failed {
                error: "The hook request was cancelled".to_owned(),
            };
        }
        Ok(Err(error)) => {
            return Outcome::Failed {
                error: format!("Hook request failed: {error}"),
            };
        }
        Ok(Ok(response)) => response,
    };
    if !status.is_success() {
        return Outcome::Failed {
            error: format!("Hook returned HTTP {status}"),
        };
    }
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Outcome::Output {
            output: HookOutput::default(),
            text: None,
        };
    }
    match serde_json::from_str::<HookOutput>(trimmed) {
        Ok(output) if trimmed.starts_with('{') => Outcome::Output { output, text: None },
        _ => Outcome::Failed {
            error: "Hook returned a body that is not a JSON object".to_owned(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stdout_is_json_or_text() {
        assert_eq!(
            parse_output(" {\"decision\":\"block\",\"reason\":\"no\"} \n"),
            Outcome::Output {
                output: HookOutput {
                    decision: Some("block".to_owned()),
                    reason: Some("no".to_owned()),
                    ..HookOutput::default()
                },
                text: None
            }
        );
        assert_eq!(
            parse_output("plain words\n"),
            Outcome::Output {
                output: HookOutput::default(),
                text: Some("plain words".to_owned())
            }
        );
        assert_eq!(
            parse_output(""),
            Outcome::Output {
                output: HookOutput::default(),
                text: None
            }
        );
    }

    #[test]
    fn headers_take_only_allowed_variables() {
        let name = "DELTA_CLAUDE_HOOKS_TEST_TOKEN";
        let allowed = [name.to_owned()];
        let value = std::env::var(name).unwrap_or_default();
        assert_eq!(
            interpolate(&format!("Bearer ${name}"), &allowed),
            format!("Bearer {value}")
        );
        assert_eq!(interpolate("Bearer $OTHER", &allowed), "Bearer $OTHER");
    }
}
