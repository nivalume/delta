//! Trajectories: what a run did, saved after every step as `format: delta-mini/1`.
//!
//! The entries hold delta-ai messages, as the model received them, each with what produced
//! it: the system prompt, the task, the model's answers with their actions and cost, the
//! observations with their command output, format errors, interruptions and the user's own
//! commands. The exit, the stats, the resolved model and the config sit beside them.

use std::io::Write as _;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use delta_ai::types::{AssistantBlock, Message, SystemBlock, SystemMessage, ToolResultMessage, UserBlock, UserMessage};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

use crate::env::ExecOutput;

/// The format this crate writes and reads.
pub const FORMAT: &str = "delta-mini/1";

/// How a run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExitStatus {
    /// An action printed the submit marker; `submission` holds the rest of its output.
    Submitted,
    /// The step or cost limit (or a batch's limit) was reached.
    LimitsExceeded,
    /// The time limit was reached.
    TimeExceeded,
    /// Too many format errors in a row.
    FormatErrors,
    /// The user interrupted and nobody could be asked how to go on.
    Interrupted,
    /// The input ended at a prompt.
    Eof,
    /// Something failed: a model call, a template, the environment. `error` says what.
    Error,
}

impl ExitStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Submitted => "submitted",
            Self::LimitsExceeded => "limits_exceeded",
            Self::TimeExceeded => "time_exceeded",
            Self::FormatErrors => "format_errors",
            Self::Interrupted => "interrupted",
            Self::Eof => "eof",
            Self::Error => "error",
        }
    }
}

impl std::fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The exit of a run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Exit {
    pub status: ExitStatus,
    /// What the run submitted; empty unless `status` is `submitted`.
    #[serde(default)]
    pub submission: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Exit {
    pub fn new(status: ExitStatus, error: Option<String>) -> Self {
        Self {
            status,
            submission: String::new(),
            error,
        }
    }
}

/// The model calls of a run, their cost and the run's duration.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Stats {
    pub calls: u64,
    /// USD.
    pub cost: f64,
    pub elapsed_seconds: f64,
}

/// What ended a run early without ending the agent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterruptReason {
    /// The user rejected the model's commands, with a comment or by switching to human mode.
    UserRejection,
    /// The user interrupted (Ctrl+C), or switched to human mode at the end.
    UserInterruption,
    /// The user gave a new task when the agent wanted to finish.
    UserNewTask,
}

/// An action: a command, and the tool call that asked for it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Action {
    pub command: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// One message of the run and what produced it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Entry {
    /// The rendered `system_template`.
    System { message: Message },
    /// The rendered `instance_template`.
    Task { message: Message },
    /// A model answer, the actions parsed from it (none after a format error) and its cost.
    Answer {
        message: Message,
        actions: Vec<Action>,
        cost: f64,
    },
    /// The rendered output of an action: a tool result for a tool call, else a user message.
    Observation { message: Message, output: ExecOutput },
    /// The rendered `format_error_template`, for an answer whose actions did not parse.
    FormatError { message: Message, error: String },
    /// A message from the user that ended a step.
    Interrupt { message: Message, reason: InterruptReason },
    /// A command the user ran in human mode.
    Human { message: Message, command: String },
}

impl Entry {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::System { .. } => "system",
            Self::Task { .. } => "task",
            Self::Answer { .. } => "answer",
            Self::Observation { .. } => "observation",
            Self::FormatError { .. } => "format_error",
            Self::Interrupt { .. } => "interrupt",
            Self::Human { .. } => "human",
        }
    }

    pub fn message(&self) -> &Message {
        match self {
            Self::System { message }
            | Self::Task { message }
            | Self::Answer { message, .. }
            | Self::Observation { message, .. }
            | Self::FormatError { message, .. }
            | Self::Interrupt { message, .. }
            | Self::Human { message, .. } => message,
        }
    }
}

/// A trajectory document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trajectory {
    pub format: String,
    /// The version of this crate that wrote it.
    pub version: String,
    /// `None` while the run goes on.
    pub exit: Option<Exit>,
    pub stats: Stats,
    /// The delta-ai model the run used.
    #[serde(default)]
    pub model: Json,
    /// The run's `agent`, `model` and `environment` sections (and `providers`), as run.
    #[serde(default)]
    pub config: Json,
    /// More information, such as the SWE-bench instance.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Json>,
    pub entries: Vec<Entry>,
}

impl Default for Trajectory {
    fn default() -> Self {
        Self {
            format: FORMAT.to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            exit: None,
            stats: Stats::default(),
            model: Json::Null,
            config: Json::Null,
            extra: Map::new(),
            entries: Vec::new(),
        }
    }
}

impl Trajectory {
    /// Reads a trajectory file; an error names a file in another format.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
        let data: Json = serde_json::from_str(&text).map_err(|error| error.to_string())?;
        match data["format"].as_str() {
            Some(FORMAT) => serde_json::from_value(data).map_err(|error| error.to_string()),
            Some(other) => Err(format!("format {other} is not {FORMAT}")),
            None => Err(format!("not a {FORMAT} trajectory")),
        }
    }

    /// Writes the trajectory to `path`.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        save_json(path, self)
    }
}

/// Writes `data` to `path` as indented JSON: a temporary file, then a rename, so a reader
/// never sees half a file.
pub fn save_json(path: &Path, data: &impl Serialize) -> std::io::Result<()> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_vec_pretty(data).map_err(std::io::Error::other)?;
    let mut name = path.file_name().map(std::ffi::OsStr::to_os_string).unwrap_or_default();
    name.push(format!(".tmp-{}", std::process::id()));
    let temporary = path.with_file_name(name);
    {
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(&text)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    std::fs::rename(&temporary, path)
}

/// Milliseconds since the epoch, as messages stamp them.
pub fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX))
}

/// A system message holding `text`.
pub fn system_message(text: &str) -> Message {
    Message::System(SystemMessage {
        content: text.into(),
        sections: None,
        tools_added: None,
        tools_removed: None,
        timestamp: now_millis(),
    })
}

/// A user message holding `text`.
pub fn user_message(text: &str) -> Message {
    Message::User(UserMessage {
        content: text.into(),
        timestamp: now_millis(),
    })
}

/// The result of the `bash` call `call_id`, holding `text`.
pub fn tool_result(call_id: &str, text: &str) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: call_id.to_owned(),
        tool_name: crate::agent::BASH_TOOL_NAME.to_owned(),
        content: vec![UserBlock::text(text)],
        details: None,
        usage: None,
        nested_calls: None,
        is_error: false,
        timestamp: now_millis(),
        duration_ms: None,
    })
}

/// The text of a message for display: its text blocks, and an assistant's tool calls as
/// fenced commands (`bash`) or the tool's name and arguments.
pub fn message_text(message: &Message) -> String {
    let mut parts: Vec<String> = Vec::new();
    match message {
        Message::System(system) => match &system.content {
            delta_ai::types::Content::Text(text) => parts.push(text.clone()),
            delta_ai::types::Content::Blocks(blocks) => parts.extend(blocks.iter().map(|block| match block {
                SystemBlock::Text(text) => text.text.clone(),
            })),
        },
        Message::User(user) => parts.push(user.content.text()),
        Message::Assistant(assistant) => {
            for block in &assistant.content {
                match block {
                    AssistantBlock::Text(text) => parts.push(text.text.clone()),
                    AssistantBlock::Thinking(_) => {}
                    AssistantBlock::ToolCall(call) => {
                        let command = call.arguments.get("command");
                        parts.push(match command {
                            Some(command) if call.name == crate::agent::BASH_TOOL_NAME => format!(
                                "```\n{}\n```",
                                command.as_str().map_or_else(|| command.to_string(), str::to_owned)
                            ),
                            _ => format!("```\n{} {}\n```", call.name, Json::Object(call.arguments.clone())),
                        });
                    }
                }
            }
        }
        Message::ToolResult(result) => parts.extend(result.content.iter().filter_map(|block| match block {
            UserBlock::Text(text) => Some(text.text.clone()),
            UserBlock::Image(_) => None,
        })),
    }
    parts.retain(|part| !part.trim().is_empty());
    parts.join("\n\n")
}

/// An assistant message's thinking, when it has any.
pub fn thinking_text(message: &Message) -> Option<String> {
    let Message::Assistant(assistant) = message else {
        return None;
    };
    let thinking: Vec<&str> = assistant
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantBlock::Thinking(thinking) if !thinking.thinking.trim().is_empty() => {
                Some(thinking.thinking.as_str())
            }
            _ => None,
        })
        .collect();
    (!thinking.is_empty()).then(|| thinking.join("\n\n"))
}

#[cfg(test)]
mod tests {
    use delta_ai::providers::faux::{faux_assistant_message, faux_text, faux_thinking, faux_tool_call};
    use serde_json::json;

    use super::*;

    #[test]
    fn trajectories_save_and_load() {
        let answer = faux_assistant_message(vec![
            faux_thinking("Hmm."),
            faux_text("Listing."),
            faux_tool_call("bash", json!({ "command": "ls" })),
        ]);
        let Some(AssistantBlock::ToolCall(call)) = answer.content.get(2).cloned() else {
            panic!("a tool call");
        };
        let mut trajectory = Trajectory::default();
        trajectory.entries.push(Entry::System {
            message: system_message("Be brief ü"),
        });
        trajectory.entries.push(Entry::Answer {
            message: Message::Assistant(answer),
            actions: vec![Action {
                command: "ls".to_owned(),
                tool_call_id: Some(call.id.clone()),
            }],
            cost: 0.5,
        });
        trajectory.entries.push(Entry::Observation {
            message: tool_result(&call.id, "<returncode>0</returncode>"),
            output: ExecOutput::default(),
        });
        trajectory.exit = Some(Exit {
            status: ExitStatus::Submitted,
            submission: "patch".to_owned(),
            error: None,
        });
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a/b.traj.json");
        trajectory.save(&path).unwrap();
        let saved: Json = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(saved["format"], FORMAT);
        assert_eq!(saved["exit"], json!({ "status": "submitted", "submission": "patch" }));
        assert_eq!(saved["entries"][1]["kind"], "answer");
        assert_eq!(saved["entries"][1]["message"]["role"], "assistant");
        assert_eq!(saved["entries"][1]["actions"][0]["tool_call_id"], json!(call.id));
        assert_eq!(saved["entries"][2]["message"]["role"], "toolResult");
        assert_eq!(saved["entries"][2]["message"]["toolCallId"], json!(call.id));
        assert_eq!(Trajectory::load(&path).unwrap(), trajectory);

        assert_eq!(message_text(trajectory.entries[0].message()), "Be brief ü");
        assert_eq!(
            message_text(trajectory.entries[1].message()),
            "Listing.\n\n```\nls\n```"
        );
        assert_eq!(thinking_text(trajectory.entries[1].message()).as_deref(), Some("Hmm."));

        std::fs::write(
            &path,
            json!({ "trajectory_format": "mini-swe-agent-1.1", "messages": [] }).to_string(),
        )
        .unwrap();
        assert!(
            Trajectory::load(&path)
                .unwrap_err()
                .contains("not a delta-mini/1 trajectory")
        );
    }
}
