//! Rough token estimates for context accounting.

use crate::transcript::get_system_message_text;
use crate::types::{AssistantBlock, Message, StopReason, Usage, UserBlock, UserContent};

const CHARS_PER_TOKEN: f64 = 3.5;
const ESTIMATED_IMAGE_CHARS: usize = 4800;

/// Length in UTF-16 code units, as upstream counts characters.
fn js_len(text: &str) -> usize {
    text.encode_utf16().count()
}

fn tokens_for_chars(chars: usize) -> u64 {
    (chars as f64 / CHARS_PER_TOKEN).ceil() as u64
}

/// Tokens a request's context used, as reported by the provider.
pub fn calculate_context_tokens(usage: &Usage) -> u64 {
    if usage.total_tokens > 0 {
        usage.total_tokens
    } else {
        usage.input + usage.output + usage.cache_read + usage.cache_write
    }
}

pub fn estimate_text_tokens(text: &str) -> u64 {
    tokens_for_chars(js_len(text))
}

fn blocks_chars(blocks: &[UserBlock]) -> usize {
    blocks
        .iter()
        .map(|block| match block {
            UserBlock::Text(text) => js_len(&text.text),
            UserBlock::Image(_) => ESTIMATED_IMAGE_CHARS,
        })
        .sum()
}

/// Estimated tokens of one message.
pub fn estimate_message_tokens(message: &Message) -> u64 {
    match message {
        Message::System(system) => {
            let tools = |json: Option<String>| json.map_or(0, |json| estimate_text_tokens(&json));
            estimate_text_tokens(&get_system_message_text(system))
                + tools(
                    system
                        .tools_added
                        .as_ref()
                        .filter(|t| !t.is_empty())
                        .and_then(|t| serde_json::to_string(t).ok()),
                )
                + tools(
                    system
                        .tools_removed
                        .as_ref()
                        .filter(|t| !t.is_empty())
                        .and_then(|t| serde_json::to_string(t).ok()),
                )
        }
        Message::User(user) => match &user.content {
            UserContent::Text(text) => estimate_text_tokens(text),
            UserContent::Blocks(blocks) => tokens_for_chars(blocks_chars(blocks)),
        },
        Message::ToolResult(result) => tokens_for_chars(blocks_chars(&result.content)),
        Message::Assistant(assistant) => {
            let chars = assistant
                .content
                .iter()
                .map(|block| match block {
                    AssistantBlock::Text(text) => js_len(&text.text),
                    AssistantBlock::Thinking(thinking) => js_len(&thinking.thinking),
                    AssistantBlock::ToolCall(call) => {
                        js_len(&call.name) + serde_json::to_string(&call.arguments).map_or(0, |json| js_len(&json))
                    }
                })
                .sum();
            tokens_for_chars(chars)
        }
    }
}

/// Estimate of the tokens a context occupies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextUsageEstimate {
    pub tokens: u64,
    /// Tokens reported by the last applicable assistant usage.
    pub usage_tokens: u64,
    /// Estimated tokens of messages after that response.
    pub trailing_tokens: u64,
    pub last_usage_index: Option<usize>,
}

fn message_timestamp(message: &Message) -> i64 {
    match message {
        Message::System(m) => m.timestamp,
        Message::User(m) => m.timestamp,
        Message::Assistant(m) => m.timestamp,
        Message::ToolResult(m) => m.timestamp,
    }
}

/// Estimate context tokens: the last successful response's reported usage plus estimates of
/// the messages after it, or estimates of every message when no usage applies.
pub fn estimate_context_tokens(messages: &[Message]) -> ContextUsageEstimate {
    let mut latest_prefix_timestamp = i64::MIN;
    let mut usage_info: Option<(u64, usize)> = None;
    for (index, message) in messages.iter().enumerate() {
        if let Message::Assistant(assistant) = message {
            // A newer prefix message inserted after this response (e.g. a compaction
            // summary) means its usage cannot describe the current prefix.
            let applies = assistant.timestamp >= latest_prefix_timestamp;
            let tokens = calculate_context_tokens(&assistant.usage);
            if applies && !matches!(assistant.stop_reason, StopReason::Aborted | StopReason::Error) && tokens > 0 {
                usage_info = Some((tokens, index));
            }
        }
        latest_prefix_timestamp = latest_prefix_timestamp.max(message_timestamp(message));
    }
    if let Some((usage_tokens, index)) = usage_info {
        let trailing_tokens: u64 = messages[index + 1..].iter().map(estimate_message_tokens).sum();
        return ContextUsageEstimate {
            tokens: usage_tokens + trailing_tokens,
            usage_tokens,
            trailing_tokens,
            last_usage_index: Some(index),
        };
    }
    let tokens = messages.iter().map(estimate_message_tokens).sum();
    ContextUsageEstimate {
        tokens,
        usage_tokens: 0,
        trailing_tokens: tokens,
        last_usage_index: None,
    }
}
