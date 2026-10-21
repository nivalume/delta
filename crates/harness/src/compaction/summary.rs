//! The summarization request and its answer: the prompts, the serialized conversation, and
//! what counts as a summary.

use delta_ai::types::{AssistantBlock, AssistantMessage, Message, StopReason, UserBlock};

/// Longest tool result text the serialized conversation keeps.
const TOOL_RESULT_MAX_CHARS: usize = 2000;

const SUMMARY_PREFIX: &str =
    "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
const SUMMARY_SUFFIX: &str = "\n</summary>";

/// The summarizer's system prompt.
pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

/// The instructions after the serialized conversation.
pub const SUMMARIZATION_PROMPT: &str = "The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work. If the conversation starts with an earlier summary, preserve its information and fold the newer messages into it.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or \"(none)\" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or \"(none)\" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.";

/// The text of the summary entry's user message.
pub(super) fn wrap(summary: &str) -> String {
    format!("{SUMMARY_PREFIX}{summary}{SUMMARY_SUFFIX}")
}

/// The summarizer's user message: the serialized conversation, the prompt, and any focus.
pub(super) fn summary_prompt(messages: &[Message], instructions: Option<&str>) -> String {
    let focus = instructions.map_or(String::new(), |instructions| {
        format!("\n\nAdditional focus: {instructions}")
    });
    format!(
        "<conversation>\n{}\n</conversation>\n\n{SUMMARIZATION_PROMPT}{focus}",
        serialize_conversation(messages)
    )
}

/// Messages as plain text, so the summarizer reads a transcript instead of continuing it.
/// System messages and thinking are omitted (replayed thinking can make the request larger
/// than the context it summarizes); visible text, tool calls and tool results remain.
pub fn serialize_conversation(messages: &[Message]) -> String {
    let mut parts = Vec::new();
    for message in messages {
        match message {
            Message::User(user) => {
                let text = user.content.text();
                if !text.is_empty() {
                    parts.push(format!("[User]: {text}"));
                }
            }
            Message::Assistant(assistant) => {
                let texts: Vec<&str> = assistant
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        AssistantBlock::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .collect();
                let calls: Vec<String> = assistant
                    .tool_calls()
                    .map(|call| {
                        let arguments: Vec<String> = call
                            .arguments
                            .iter()
                            .map(|(key, value)| format!("{key}={value}"))
                            .collect();
                        format!("{}({})", call.name, arguments.join(", "))
                    })
                    .collect();
                if !texts.is_empty() {
                    parts.push(format!("[Assistant]: {}", texts.join("\n")));
                }
                if !calls.is_empty() {
                    parts.push(format!("[Assistant tool calls]: {}", calls.join("; ")));
                }
            }
            Message::ToolResult(result) => {
                let text = blocks_text(&result.content);
                if !text.is_empty() {
                    parts.push(format!("[Tool result]: {}", truncate(&text, TOOL_RESULT_MAX_CHARS)));
                }
            }
            Message::System(_) => {}
        }
    }
    parts.join("\n\n")
}

fn blocks_text(blocks: &[UserBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            UserBlock::Text(text) => Some(text.text.as_str()),
            UserBlock::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `text` cut to `max` characters with a note of how many were dropped.
fn truncate(text: &str, max: usize) -> String {
    let total = text.chars().count();
    if total <= max {
        return text.to_owned();
    }
    let kept: String = text.chars().take(max).collect();
    format!("{kept}\n\n[... {} more characters truncated]", total - max)
}

/// The summary of a clean `stop` with text and no tool call; anything else is not one.
pub(super) fn summary_text(message: &AssistantMessage) -> Option<String> {
    if message.stop_reason != StopReason::Stop || message.tool_calls().next().is_some() {
        return None;
    }
    let text: Vec<&str> = message
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    let text = text.join("\n").trim().to_owned();
    (!text.is_empty()).then_some(text)
}

/// Why `message` is not a summary.
pub(super) fn summary_failure(message: &AssistantMessage) -> String {
    match message.stop_reason {
        StopReason::Error | StopReason::Aborted => format!(
            "Summarization failed: {}",
            message
                .error_message
                .clone()
                .unwrap_or_else(|| stop_reason_name(message.stop_reason).to_owned())
        ),
        StopReason::Length => "Summarization hit the token limit; the summary is incomplete".to_owned(),
        _ if message.tool_calls().next().is_some() => "Summarization attempted to call a tool".to_owned(),
        _ => "Summarization produced no text".to_owned(),
    }
}

fn stop_reason_name(reason: StopReason) -> &'static str {
    match reason {
        StopReason::Error => "error",
        StopReason::Aborted => "aborted",
        _ => "stopped",
    }
}
