//! Thread items from harness records: user input, assistant blocks and tool calls, with the
//! ids live events and `thread/read` share.

use base64::Engine;
use delta_ai::types::{ImageContent, JsonObject, Message, ToolResultMessage, UserBlock, UserContent};
use delta_harness::records::{EntryRecord, TaskId};
use delta_harness::tool::ToolDiagnostic;
use delta_protocol::v2::{
    CommandAction, CommandExecutionStatus, DynamicToolCallOutputContentItem, DynamicToolCallStatus, FileUpdateChange,
    PatchApplyStatus, PatchChangeKind, ThreadItem, UserInput,
};
use serde_json::Value;

/// The id of block `index` of the assistant message task `task` generated.
pub(crate) fn block_id(task: Option<TaskId>, index: usize) -> String {
    match task {
        Some(task) => format!("{task}:{index}"),
        None => format!("message:{index}"),
    }
}

/// A user message item.
pub(crate) fn user_message(id: String, content: &UserContent) -> ThreadItem {
    let content = content
        .to_blocks()
        .into_iter()
        .map(|block| match block {
            UserBlock::Text(text) => UserInput::text(text.text),
            UserBlock::Image(image) => UserInput::Image {
                url: format!("data:{};base64,{}", image.mime_type, image.data),
            },
        })
        .collect();
    ThreadItem::UserMessage {
        id,
        client_id: None,
        content,
    }
}

/// Why user input cannot be submitted.
#[derive(Debug)]
pub(crate) struct InputError(pub(crate) String);

/// The harness content of `input`: text as text blocks, `data:` URLs and local image files
/// as image blocks.
pub(crate) async fn user_content(input: &[UserInput]) -> Result<UserContent, InputError> {
    let mut blocks = Vec::with_capacity(input.len());
    for part in input {
        match part {
            UserInput::Text { text, .. } => blocks.push(UserBlock::text(text.clone())),
            UserInput::Image { url } => {
                let (mime, data) = url
                    .strip_prefix("data:")
                    .and_then(|rest| rest.split_once(";base64,"))
                    .ok_or_else(|| InputError(format!("Only data: image URLs are supported, not {url}")))?;
                blocks.push(UserBlock::Image(ImageContent {
                    data: data.to_owned(),
                    mime_type: mime.to_owned(),
                }));
            }
            UserInput::LocalImage { path } => {
                let mime = image_mime(path).ok_or_else(|| InputError(format!("Not an image file: {path}")))?;
                let bytes = tokio::fs::read(path)
                    .await
                    .map_err(|error| InputError(format!("Cannot read {path}: {error}")))?;
                blocks.push(UserBlock::Image(ImageContent {
                    data: base64::engine::general_purpose::STANDARD.encode(bytes),
                    mime_type: mime.to_owned(),
                }));
            }
        }
    }
    if blocks.is_empty() {
        return Err(InputError("The input is empty".to_owned()));
    }
    Ok(match blocks.as_slice() {
        [UserBlock::Text(text)] => UserContent::Text(text.text.clone()),
        _ => UserContent::Blocks(blocks),
    })
}

fn image_mime(path: &str) -> Option<&'static str> {
    let extension = path.rsplit_once('.')?.1.to_ascii_lowercase();
    Some(match extension.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => return None,
    })
}

/// A tool call the model made.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Call {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) arguments: JsonObject,
}

/// How a call ended: its result message and the diagnostics of its entry.
#[derive(Clone, Debug)]
pub(crate) struct Outcome {
    pub(crate) message: ToolResultMessage,
    pub(crate) diagnostics: Vec<ToolDiagnostic>,
}

impl Outcome {
    /// The outcome a `delta.tool-result` entry records.
    pub(crate) fn of(entry: &EntryRecord) -> Option<Self> {
        let Some([Message::ToolResult(message), ..]) = entry.model.as_deref() else {
            return None;
        };
        let diagnostics = entry
            .data
            .as_ref()
            .and_then(|data| data.get("diagnostics"))
            .and_then(|diagnostics| delta_chord::json::from_json(diagnostics).ok())
            .unwrap_or_default();
        Some(Self {
            message: message.clone(),
            diagnostics,
        })
    }

    /// A call that ended without a result entry.
    pub(crate) fn interrupted(call: &Call) -> Self {
        Self {
            message: ToolResultMessage {
                tool_call_id: call.id.clone(),
                tool_name: call.name.clone(),
                content: Vec::new(),
                details: None,
                usage: None,
                nested_calls: None,
                is_error: true,
                timestamp: 0,
                duration_ms: None,
            },
            diagnostics: vec![ToolDiagnostic::error("interrupted", "The call ended without a result")],
        }
    }

    fn has_code(&self, code: &str) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some(code))
    }

    /// Blocked by a hook, denied by the policy or by a reviewer.
    fn declined(&self) -> bool {
        self.message.is_error && self.has_code("blocked")
    }

    /// The output the model saw, without the rendered diagnostics block.
    fn output_blocks(&self) -> &[UserBlock] {
        let blocks = self.message.content.as_slice();
        match blocks.split_last() {
            Some((UserBlock::Text(last), rest))
                if !self.diagnostics.is_empty() && last.text.starts_with("<harness>") =>
            {
                rest
            }
            _ => blocks,
        }
    }

    fn output_text(&self) -> String {
        self.output_blocks()
            .iter()
            .filter_map(|block| match block {
                UserBlock::Text(text) => Some(text.text.as_str()),
                UserBlock::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn exit_code(&self) -> Option<i32> {
        let code = self
            .diagnostics
            .iter()
            .find(|diagnostic| diagnostic.code.as_deref() == Some("exit_code"))
            .and_then(|diagnostic| diagnostic.message.rsplit(' ').next())
            .and_then(|code| code.parse().ok());
        code.or((!self.message.is_error).then_some(0))
    }

    fn duration_ms(&self) -> Option<i64> {
        self.message
            .duration_ms
            .and_then(|duration| i64::try_from(duration).ok())
    }
}

fn argument<'a>(call: &'a Call, name: &str) -> &'a str {
    call.arguments.get(name).and_then(Value::as_str).unwrap_or_default()
}

/// The item of `call`: `bash` a command execution, `edit` and `write` a file change, any
/// other tool a dynamic tool call; in progress without an outcome.
pub(crate) fn tool_item(call: &Call, cwd: &str, outcome: Option<&Outcome>) -> ThreadItem {
    match call.name.as_str() {
        "bash" => {
            let command = argument(call, "command").to_owned();
            let status = match outcome {
                None => CommandExecutionStatus::InProgress,
                Some(outcome) if outcome.declined() => CommandExecutionStatus::Declined,
                Some(outcome) if outcome.message.is_error => CommandExecutionStatus::Failed,
                Some(_) => CommandExecutionStatus::Completed,
            };
            ThreadItem::CommandExecution {
                id: call.id.clone(),
                command_actions: vec![CommandAction::Unknown {
                    command: command.clone(),
                }],
                command,
                cwd: cwd.to_owned(),
                process_id: None,
                status,
                aggregated_output: outcome.map(Outcome::output_text),
                exit_code: outcome.and_then(Outcome::exit_code),
                duration_ms: outcome.and_then(Outcome::duration_ms),
            }
        }
        "edit" | "write" => {
            let status = match outcome {
                None => PatchApplyStatus::InProgress,
                Some(outcome) if outcome.declined() => PatchApplyStatus::Declined,
                Some(outcome) if outcome.message.is_error => PatchApplyStatus::Failed,
                Some(_) => PatchApplyStatus::Completed,
            };
            let path = argument(call, "path").to_owned();
            let change = if call.name == "write" {
                FileUpdateChange {
                    path,
                    kind: PatchChangeKind::Add,
                    diff: argument(call, "content").to_owned(),
                }
            } else {
                let patch = outcome
                    .and_then(|outcome| outcome.message.details.as_ref())
                    .and_then(|details| details.get("patch"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                FileUpdateChange {
                    path,
                    kind: PatchChangeKind::Update { move_path: None },
                    diff: patch.to_owned(),
                }
            };
            ThreadItem::FileChange {
                id: call.id.clone(),
                changes: vec![change],
                status,
            }
        }
        _ => {
            let status = match outcome {
                None => DynamicToolCallStatus::InProgress,
                Some(outcome) if outcome.message.is_error => DynamicToolCallStatus::Failed,
                Some(_) => DynamicToolCallStatus::Completed,
            };
            let content_items = outcome.map(|outcome| {
                outcome
                    .output_blocks()
                    .iter()
                    .map(|block| match block {
                        UserBlock::Text(text) => DynamicToolCallOutputContentItem::InputText {
                            text: text.text.clone(),
                        },
                        UserBlock::Image(image) => DynamicToolCallOutputContentItem::InputImage {
                            image_url: format!("data:{};base64,{}", image.mime_type, image.data),
                        },
                    })
                    .collect()
            });
            ThreadItem::DynamicToolCall {
                id: call.id.clone(),
                namespace: None,
                tool: call.name.clone(),
                arguments: Value::Object(call.arguments.clone()),
                status,
                content_items,
                success: outcome.map(|outcome| !outcome.message.is_error),
                duration_ms: outcome.and_then(Outcome::duration_ms),
            }
        }
    }
}

/// The kind of approval request a call's item takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ApprovalKind {
    CommandExecution,
    FileChange,
    Tool,
}

impl ApprovalKind {
    pub(crate) fn of(tool: &str) -> Self {
        match tool {
            "bash" => Self::CommandExecution,
            "edit" | "write" => Self::FileChange,
            _ => Self::Tool,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, arguments: &Value) -> Call {
        Call {
            id: "c1".to_owned(),
            name: name.to_owned(),
            arguments: arguments.as_object().cloned().unwrap_or_default(),
        }
    }

    fn outcome(text: &str, is_error: bool, diagnostics: Vec<ToolDiagnostic>) -> Outcome {
        let mut content = vec![UserBlock::text(text)];
        if !diagnostics.is_empty() {
            content.push(UserBlock::text("<harness>\n[error] x\n</harness>"));
        }
        Outcome {
            message: ToolResultMessage {
                tool_call_id: "c1".to_owned(),
                tool_name: "bash".to_owned(),
                content,
                details: None,
                usage: None,
                nested_calls: None,
                is_error,
                timestamp: 0,
                duration_ms: Some(7),
            },
            diagnostics,
        }
    }

    #[test]
    fn commands_report_output_and_exit_codes() {
        let call = call("bash", &serde_json::json!({ "command": "false" }));
        let failed = outcome(
            "nope",
            true,
            vec![ToolDiagnostic::error("exit_code", "Command exited with code 1")],
        );
        let ThreadItem::CommandExecution {
            status,
            aggregated_output,
            exit_code,
            duration_ms,
            ..
        } = tool_item(&call, "/w", Some(&failed))
        else {
            panic!("not a command")
        };
        assert_eq!(status, CommandExecutionStatus::Failed);
        assert_eq!(aggregated_output.as_deref(), Some("nope"));
        assert_eq!(exit_code, Some(1));
        assert_eq!(duration_ms, Some(7));
        let blocked = outcome("", true, vec![ToolDiagnostic::error("blocked", "Denied")]);
        let ThreadItem::CommandExecution { status, exit_code, .. } = tool_item(&call, "/w", Some(&blocked)) else {
            panic!("not a command")
        };
        assert_eq!((status, exit_code), (CommandExecutionStatus::Declined, None));
        let ThreadItem::CommandExecution { status, .. } = tool_item(&call, "/w", None) else {
            panic!("not a command")
        };
        assert_eq!(status, CommandExecutionStatus::InProgress);
    }

    #[tokio::test]
    async fn input_takes_text_and_data_urls() {
        let content = user_content(&[UserInput::text("hi")]).await.unwrap();
        assert_eq!(content, UserContent::Text("hi".to_owned()));
        let content = user_content(&[
            UserInput::text("look"),
            UserInput::Image {
                url: "data:image/png;base64,AAAA".to_owned(),
            },
        ])
        .await
        .unwrap();
        let ThreadItem::UserMessage { content, .. } = user_message("7".to_owned(), &content) else {
            panic!("not a user message")
        };
        assert_eq!(
            content[1],
            UserInput::Image {
                url: "data:image/png;base64,AAAA".to_owned()
            }
        );
        assert!(
            user_content(&[UserInput::Image {
                url: "https://example.com/a.png".to_owned()
            }])
            .await
            .is_err()
        );
        assert!(user_content(&[]).await.is_err());
    }
}
