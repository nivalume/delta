//! Trajectories: the `.traj.json` files upstream writes (`trajectory_format`
//! `mini-swe-agent-1.1`), built message by message in upstream's OpenAI-style shape.

use std::io::Write as _;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use delta_ai::types::{AssistantBlock, AssistantMessage};
use serde_json::{Map, Value as Json, json};

use crate::env::ExecOutput;

/// The format this crate writes.
pub const TRAJECTORY_FORMAT: &str = "mini-swe-agent-1.1";

/// `info.mini_version`: the upstream version this port follows, marked as delta's.
pub const MINI_VERSION: &str = concat!("2.4.6+delta.", env!("CARGO_PKG_VERSION"));

/// Seconds since the epoch, as Python's `time.time()`.
pub fn timestamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs_f64())
        .unwrap_or_default()
}

/// A plain message: `{"role": role, "content": content}` and `extra` when not empty.
pub fn message(role: &str, content: &str, extra: Map<String, Json>) -> Json {
    let mut message = json!({ "role": role, "content": content });
    if !extra.is_empty() {
        message["extra"] = Json::Object(extra);
    }
    message
}

/// An `extra` object from pairs.
pub fn extra<const N: usize>(pairs: [(&str, Json); N]) -> Map<String, Json> {
    pairs.into_iter().map(|(key, value)| (key.to_owned(), value)).collect()
}

/// An exit message: `{"role": "exit", "content": content, "extra": {"exit_status",
/// "submission", ...}}`.
pub fn exit_message(content: &str, exit_status: &str, submission: &str, more: Map<String, Json>) -> Json {
    let mut fields = extra([
        ("exit_status", Json::String(exit_status.to_owned())),
        ("submission", Json::String(submission.to_owned())),
    ]);
    fields.extend(more);
    message("exit", content, fields)
}

/// An answer as an OpenAI chat message: its text as `content` (null without text), its
/// reasoning as `reasoning_content`, its calls as `tool_calls`.
pub fn assistant_message(answer: &AssistantMessage, extra: Map<String, Json>) -> Json {
    let text: String = answer
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let reasoning: String = answer
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantBlock::Thinking(thinking) => Some(thinking.thinking.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let tool_calls: Vec<Json> = answer
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantBlock::ToolCall(call) => Some(json!({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": Json::Object(call.arguments.clone()).to_string(),
                },
            })),
            _ => None,
        })
        .collect();
    let mut message = json!({
        "role": "assistant",
        "content": if text.is_empty() { Json::Null } else { Json::String(text) },
    });
    if !reasoning.is_empty() {
        message["reasoning_content"] = Json::String(reasoning);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = Json::Array(tool_calls);
    }
    message["extra"] = Json::Object(extra);
    message
}

/// The observation of one action: a `tool` message answering `tool_call_id`, or a `user`
/// message for text-based and human actions (upstream's `format_*_observation_messages`).
pub fn observation_message(content: &str, output: &ExecOutput, tool_call_id: Option<&str>) -> Json {
    let mut fields = extra([
        ("raw_output", Json::String(output.output.clone())),
        ("returncode", json!(output.returncode)),
        ("timestamp", json!(timestamp())),
        ("exception_info", Json::String(output.exception_info.clone())),
    ]);
    fields.extend(output.extra.clone());
    let mut message = json!({ "content": content, "extra": fields });
    match tool_call_id {
        Some(id) => {
            message["tool_call_id"] = Json::String(id.to_owned());
            message["role"] = Json::String("tool".to_owned());
        }
        None => message["role"] = Json::String("user".to_owned()),
    }
    message
}

/// What upstream's runner reads back: the exit status and the submission.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExitInfo {
    pub exit_status: String,
    pub submission: String,
}

/// The messages of a run and what the trajectory reports about them.
#[derive(Clone, Debug, Default)]
pub struct Trajectory {
    pub messages: Vec<Json>,
}

impl Trajectory {
    /// The exit, once the last message is one.
    pub fn exit(&self) -> Option<ExitInfo> {
        let last = self.messages.last()?;
        (last["role"] == "exit").then(|| {
            let field = |key: &str| last["extra"][key].as_str().unwrap_or_default().to_owned();
            ExitInfo {
                exit_status: field("exit_status"),
                submission: field("submission"),
            }
        })
    }

    /// The trajectory document: `info`, `messages` and `trajectory_format`, merged with the
    /// model's and the environment's parts (upstream's `DefaultAgent.serialize`).
    pub fn serialize(
        &self,
        cost: f64,
        api_calls: u64,
        agent: &Json,
        extra_parts: impl IntoIterator<Item = Json>,
    ) -> Json {
        let last_extra = self.messages.last().map(|message| &message["extra"]);
        let field = |key: &str| {
            last_extra
                .and_then(|extra| extra.get(key))
                .cloned()
                .unwrap_or_else(|| Json::String(String::new()))
        };
        let base = json!({
            "info": {
                "model_stats": { "instance_cost": cost, "api_calls": api_calls },
                "config": agent,
                "mini_version": MINI_VERSION,
                "exit_status": field("exit_status"),
                "submission": field("submission"),
            },
            "messages": self.messages,
            "trajectory_format": TRAJECTORY_FORMAT,
        });
        crate::config::merge_all(std::iter::once(base).chain(extra_parts))
    }
}

/// Writes `data` to `path` as indented JSON: a temporary file, then a rename, so a reader
/// never sees half a file.
pub fn save_json(path: &Path, data: &Json) -> std::io::Result<()> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut text = Vec::new();
    write_python_indented(data, &mut text)?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    {
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(&text)?;
        file.sync_all()?;
    }
    std::fs::rename(&temporary, path)
}

/// `json.dumps(data, indent=2)`: two-space indentation, `": "` between key and value,
/// non-ASCII escaped.
fn write_python_indented(data: &Json, out: &mut Vec<u8>) -> std::io::Result<()> {
    let mut text = String::new();
    python_indented(data, 0, &mut text);
    out.write_all(text.as_bytes())
}

fn python_indented(value: &Json, depth: usize, out: &mut String) {
    let pad = |out: &mut String, depth: usize| {
        out.push('\n');
        out.push_str(&"  ".repeat(depth));
    };
    match value {
        Json::Array(items) if !items.is_empty() => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                pad(out, depth + 1);
                python_indented(item, depth + 1, out);
            }
            pad(out, depth);
            out.push(']');
        }
        Json::Object(object) if !object.is_empty() => {
            out.push('{');
            for (index, (key, item)) in object.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                pad(out, depth + 1);
                crate::template::python_dumps(&Json::String(key.clone()), None, 0, out);
                out.push_str(": ");
                python_indented(item, depth + 1, out);
            }
            pad(out, depth);
            out.push('}');
        }
        other => crate::template::python_dumps(other, None, 0, out),
    }
}

/// Reads a trajectory file.
pub fn load(path: &Path) -> std::io::Result<Json> {
    let text = std::fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(std::io::Error::other)
}

/// The text of a message for display (upstream's `get_content_string`): its content, tool
/// calls as fenced commands, observations in JSON as `<key>` lines.
pub fn content_string(message: &Json) -> String {
    let mut texts: Vec<String> = Vec::new();
    match &message["content"] {
        Json::String(content) => texts.push(format_observation(content)),
        Json::Array(items) => {
            for item in items {
                match item["type"].as_str() {
                    Some("tool_use") => texts.push(format_tool_call(&item["input"].to_string())),
                    Some("tool_result") => {
                        if let Some(content) = item["content"].as_str() {
                            texts.push(format_observation(content));
                        }
                    }
                    _ => {
                        if let Some(text) = item["text"].as_str() {
                            texts.push(text.to_owned());
                        }
                    }
                }
            }
        }
        _ => {}
    }
    for call in message["tool_calls"].as_array().into_iter().flatten() {
        let arguments = &call["function"]["arguments"];
        let arguments = arguments.as_str().map_or_else(|| arguments.to_string(), str::to_owned);
        texts.push(format_tool_call(&arguments));
    }
    match &message["output"] {
        Json::String(output) => texts.push(format_observation(output)),
        Json::Array(items) => {
            for item in items {
                match item["type"].as_str() {
                    Some("message") => {
                        for content in item["content"].as_array().into_iter().flatten() {
                            if let Some(text) = content["text"].as_str() {
                                texts.push(text.to_owned());
                            }
                        }
                    }
                    Some("function_call") => {
                        texts.push(format_tool_call(item["arguments"].as_str().unwrap_or("{}")));
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    texts.retain(|text| !text.is_empty());
    texts.join("\n\n")
}

fn format_tool_call(arguments: &str) -> String {
    if let Ok(Json::Object(arguments)) = serde_json::from_str::<Json>(arguments)
        && let Some(command) = arguments.get("command")
    {
        let command = command.as_str().map_or_else(|| command.to_string(), str::to_owned);
        return format!("```\n{command}\n```");
    }
    format!("```\n{arguments}\n```")
}

fn format_observation(content: &str) -> String {
    match serde_json::from_str::<Json>(content) {
        Ok(Json::Object(data)) if data.contains_key("returncode") => data
            .iter()
            .flat_map(|(key, value)| [format!("<{key}>"), python_str(value)])
            .collect::<Vec<_>>()
            .join("\n"),
        _ => content.to_owned(),
    }
}

/// Python's `str()` of a JSON value, for the values observations hold.
fn python_str(value: &Json) -> String {
    match value {
        Json::String(text) => text.clone(),
        Json::Null => "None".to_owned(),
        Json::Bool(true) => "True".to_owned(),
        Json::Bool(false) => "False".to_owned(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use delta_ai::providers::faux::{faux_assistant_message, faux_text, faux_tool_call};

    use super::*;

    #[test]
    fn answers_become_openai_messages() {
        let answer = faux_assistant_message(vec![
            faux_text("Listing."),
            faux_tool_call("bash", json!({ "command": "ls" })),
        ]);
        let message = assistant_message(&answer, extra([("cost", json!(0.5))]));
        assert_eq!(message["role"], "assistant");
        assert_eq!(message["content"], "Listing.");
        assert_eq!(message["tool_calls"][0]["function"]["arguments"], r#"{"command":"ls"}"#);
        assert_eq!(message["extra"]["cost"], 0.5);
        assert_eq!(content_string(&message), "Listing.\n\n```\nls\n```");
    }

    #[test]
    fn observations_render_as_upstream() {
        let output = ExecOutput {
            output: "hi\n".to_owned(),
            ..ExecOutput::default()
        };
        let tool = observation_message("<returncode>0</returncode>", &output, Some("call_1"));
        assert_eq!(
            (tool["role"].as_str(), tool["tool_call_id"].as_str()),
            (Some("tool"), Some("call_1"))
        );
        assert_eq!(tool["extra"]["raw_output"], "hi\n");
        let user = observation_message(r#"{"returncode": 0, "output": "hi"}"#, &output, None);
        assert_eq!(user["role"], "user");
        assert_eq!(content_string(&user), "<returncode>\n0\n<output>\nhi");
    }

    #[test]
    fn trajectories_save_as_python_json() {
        let mut trajectory = Trajectory::default();
        trajectory.messages.push(message("system", "Be brief ü", Map::new()));
        trajectory
            .messages
            .push(exit_message("", "Submitted", "patch", Map::new()));
        assert_eq!(
            trajectory.exit(),
            Some(ExitInfo {
                exit_status: "Submitted".to_owned(),
                submission: "patch".to_owned()
            })
        );
        let data = trajectory.serialize(
            2.0,
            2,
            &json!({ "agent": {}, "agent_type": "x" }),
            [json!({ "info": { "config": { "model_type": "m" } } })],
        );
        assert_eq!(data["info"]["config"]["model_type"], "m");
        assert_eq!(data["info"]["exit_status"], "Submitted");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a/b.traj.json");
        save_json(&path, &data).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("{\n  \"info\": {\n    \"model_stats\": {\n      \"instance_cost\": 2.0,"));
        assert!(text.contains(&format!("Be brief {}u00fc", '\\')));
        assert_eq!(load(&path).unwrap(), data);
    }
}
