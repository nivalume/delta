//! Replay of a transcript for a target model: cross-provider hand-off rules, image
//! downgrade for non-vision models, tool call id normalization and synthetic results for
//! orphaned tool calls.

use std::collections::{HashMap, HashSet};

use crate::model::Model;
use crate::types::{
    AssistantBlock, AssistantMessage, ImageContent, Message, StopReason, TextContent, ToolCall, ToolResultMessage,
    UserBlock, UserContent, now_ms,
};

const NON_VISION_USER_IMAGE_PLACEHOLDER: &str = "(image omitted: model does not support images)";
const NON_VISION_TOOL_IMAGE_PLACEHOLDER: &str = "(tool image omitted: model does not support images)";
const ERROR_TOOL_IMAGE_PLACEHOLDER: &str = "(image attached after the tool results)";

/// An error tool result split into a text-only result and the images to send after the
/// tool results of the turn.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitErrorToolResult {
    /// The result content with each run of images replaced by a note.
    pub content: Vec<TextContent>,
    /// Text naming the tool call, to send before `images`.
    pub caption: String,
    pub images: Vec<ImageContent>,
}

/// Split the images out of an error tool result, for APIs that reject images in error
/// results (Anthropic, and Claude on Bedrock). `None` when no split is needed.
pub fn split_error_tool_result_images(message: &ToolResultMessage) -> Option<SplitErrorToolResult> {
    if !message.is_error {
        return None;
    }
    let images: Vec<ImageContent> = message
        .content
        .iter()
        .filter_map(|block| match block {
            UserBlock::Image(image) => Some(image.clone()),
            UserBlock::Text(_) => None,
        })
        .collect();
    if images.is_empty() {
        return None;
    }
    Some(SplitErrorToolResult {
        content: replace_images_with_placeholder(&message.content, ERROR_TOOL_IMAGE_PLACEHOLDER),
        caption: format!(
            "Images from the error result of tool call {} ({}):",
            message.tool_call_id, message.tool_name
        ),
        images,
    })
}

fn replace_images_with_placeholder(content: &[UserBlock], placeholder: &str) -> Vec<TextContent> {
    let mut result = Vec::with_capacity(content.len());
    let mut previous_was_placeholder = false;
    for block in content {
        match block {
            UserBlock::Image(_) => {
                if !previous_was_placeholder {
                    result.push(TextContent::new(placeholder));
                }
                previous_was_placeholder = true;
            }
            UserBlock::Text(text) => {
                previous_was_placeholder = text.text == placeholder;
                result.push(text.clone());
            }
        }
    }
    result
}

fn texts_to_blocks(texts: Vec<TextContent>) -> Vec<UserBlock> {
    texts.into_iter().map(UserBlock::Text).collect()
}

fn downgrade_unsupported_images(message: Message) -> Message {
    match message {
        Message::User(mut user) => {
            if let UserContent::Blocks(blocks) = &user.content {
                user.content = UserContent::Blocks(texts_to_blocks(replace_images_with_placeholder(
                    blocks,
                    NON_VISION_USER_IMAGE_PLACEHOLDER,
                )));
            }
            Message::User(user)
        }
        Message::ToolResult(mut result) => {
            result.content = texts_to_blocks(replace_images_with_placeholder(
                &result.content,
                NON_VISION_TOOL_IMAGE_PLACEHOLDER,
            ));
            Message::ToolResult(result)
        }
        other => other,
    }
}

/// Normalizes a tool call id from another model, e.g. for APIs that restrict id characters.
pub type NormalizeToolCallId<'a> = &'a dyn Fn(&str, &Model, &AssistantMessage) -> String;

fn transform_assistant(
    mut assistant: AssistantMessage,
    model: &Model,
    normalize_tool_call_id: Option<NormalizeToolCallId<'_>>,
    id_map: &mut HashMap<String, String>,
) -> AssistantMessage {
    let same_model = assistant.provider == model.provider && assistant.api == model.api && assistant.model == model.id;
    let source = assistant.clone();
    assistant.content = std::mem::take(&mut assistant.content)
        .into_iter()
        .filter_map(|block| match block {
            AssistantBlock::Thinking(thinking) => {
                // Thinking from another model is dropped: signatures and encrypted reasoning
                // only work with the model that produced them, and readable thinking sent as
                // text makes the new model treat it as part of the reply.
                if !same_model {
                    return None;
                }
                // Keep redacted and signed blocks even when empty (encrypted reasoning).
                if thinking.is_redacted() || thinking.thinking_signature.as_deref().is_some_and(|s| !s.is_empty()) {
                    return Some(AssistantBlock::Thinking(thinking));
                }
                (!thinking.thinking.trim().is_empty()).then_some(AssistantBlock::Thinking(thinking))
            }
            AssistantBlock::Text(text) => {
                if same_model {
                    Some(AssistantBlock::Text(text))
                } else {
                    Some(AssistantBlock::Text(TextContent::new(text.text)))
                }
            }
            AssistantBlock::ToolCall(mut call) => {
                if !same_model {
                    call.thought_signature = None;
                    if let Some(normalize) = normalize_tool_call_id {
                        let normalized = normalize(&call.id, model, &source);
                        if normalized != call.id {
                            id_map.insert(call.id.clone(), normalized.clone());
                            call.id = normalized;
                        }
                    }
                }
                Some(AssistantBlock::ToolCall(call))
            }
        })
        .collect();
    assistant
}

/// Prepare `messages` for `model`.
///
/// - Images become placeholders when the model has no image input.
/// - Assistant messages from another model lose thinking blocks, text signatures and tool
///   call thought signatures; tool call ids may be normalized (results follow the mapping).
/// - Errored and aborted assistant messages are dropped.
/// - Tool calls without results get synthetic error results ("No result provided") before
///   the next assistant or user message; system messages in between move after them.
pub fn transform_messages(
    messages: &[Message],
    model: &Model,
    normalize_tool_call_id: Option<NormalizeToolCallId<'_>>,
) -> Vec<Message> {
    let mut id_map: HashMap<String, String> = HashMap::new();
    let vision = model.supports_images();
    let transformed: Vec<Message> = messages
        .iter()
        .cloned()
        .map(|message| {
            if vision {
                message
            } else {
                downgrade_unsupported_images(message)
            }
        })
        .map(|message| match message {
            Message::ToolResult(mut result) => {
                if let Some(id) = id_map.get(&result.tool_call_id) {
                    result.tool_call_id.clone_from(id);
                }
                Message::ToolResult(result)
            }
            Message::Assistant(assistant) => Message::Assistant(transform_assistant(
                assistant,
                model,
                normalize_tool_call_id,
                &mut id_map,
            )),
            other => other,
        })
        .collect();

    let mut result = Vec::with_capacity(transformed.len());
    let mut pending: Vec<ToolCall> = Vec::new();
    let mut answered: HashSet<String> = HashSet::new();
    let mut held_system: Vec<Message> = Vec::new();

    let close_pending = |result: &mut Vec<Message>,
                         pending: &mut Vec<ToolCall>,
                         answered: &mut HashSet<String>,
                         held: &mut Vec<Message>| {
        for call in pending.drain(..) {
            if !answered.contains(&call.id) {
                result.push(Message::ToolResult(ToolResultMessage {
                    tool_call_id: call.id,
                    tool_name: call.name,
                    content: vec![UserBlock::text("No result provided")],
                    is_error: true,
                    timestamp: now_ms(),
                    ..ToolResultMessage::default()
                }));
            }
        }
        answered.clear();
        result.append(held);
    };

    for message in transformed {
        match message {
            Message::Assistant(assistant) => {
                close_pending(&mut result, &mut pending, &mut answered, &mut held_system);
                // Incomplete turns are not replayed; the model retries from the last valid state.
                if matches!(assistant.stop_reason, StopReason::Error | StopReason::Aborted) {
                    continue;
                }
                let calls: Vec<ToolCall> = assistant.tool_calls().cloned().collect();
                if !calls.is_empty() {
                    pending = calls;
                    answered.clear();
                }
                result.push(Message::Assistant(assistant));
            }
            Message::ToolResult(tool_result) => {
                answered.insert(tool_result.tool_call_id.clone());
                result.push(Message::ToolResult(tool_result));
            }
            Message::System(system) => {
                if pending.is_empty() {
                    result.push(Message::System(system));
                } else {
                    held_system.push(Message::System(system));
                }
            }
            Message::User(user) => {
                close_pending(&mut result, &mut pending, &mut answered, &mut held_system);
                result.push(Message::User(user));
            }
        }
    }
    close_pending(&mut result, &mut pending, &mut answered, &mut held_system);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SystemMessage, ThinkingContent};
    use serde_json::json;

    fn model() -> Model {
        serde_json::from_value(
            json!({"id": "m2", "name": "M", "api": "anthropic-messages", "provider": "anthropic",
            "baseUrl": "x", "reasoning": true, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}, "contextWindow": 1, "maxTokens": 1}),
        )
        .unwrap()
    }

    #[test]
    fn hand_off_drops_foreign_thinking_and_fills_orphans() {
        let mut assistant = AssistantMessage::empty("openai-responses", "openai", "gpt");
        assistant.stop_reason = StopReason::ToolUse;
        assistant.content = vec![
            AssistantBlock::Thinking(ThinkingContent::new("hmm")),
            AssistantBlock::Text(TextContent {
                text: "x".into(),
                text_signature: Some("s".into()),
            }),
            AssistantBlock::ToolCall(ToolCall {
                id: "call|1".into(),
                name: "read".into(),
                ..ToolCall::default()
            }),
        ];
        let messages = vec![
            Message::user("hi"),
            Message::Assistant(assistant),
            Message::System(SystemMessage {
                content: "later".into(),
                ..SystemMessage::default()
            }),
            Message::user(UserContent::Blocks(vec![UserBlock::Image(ImageContent::default())])),
        ];
        let normalize = |id: &str, _: &Model, _: &AssistantMessage| id.replace('|', "_");
        let out = transform_messages(&messages, &model(), Some(&normalize));
        let roles: Vec<_> = out.iter().map(Message::role).collect();
        assert_eq!(roles, ["user", "assistant", "toolResult", "system", "user"]);
        let Message::Assistant(a) = &out[1] else { panic!() };
        assert_eq!(a.content.len(), 2);
        assert_eq!(a.content[0], AssistantBlock::Text(TextContent::new("x")));
        let Message::ToolResult(r) = &out[2] else { panic!() };
        assert_eq!(r.tool_call_id, "call_1");
        let Message::User(u) = &out[4] else { panic!() };
        assert_eq!(u.content.text(), NON_VISION_USER_IMAGE_PLACEHOLDER);
    }
}
