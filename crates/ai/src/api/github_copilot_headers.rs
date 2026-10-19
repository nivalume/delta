//! GitHub Copilot's per-request headers. Port of upstream `api/github-copilot-headers.ts`.
//!
//! The `anthropic-messages`, `openai-completions` and `openai-responses` APIs add these
//! headers when `model.provider == "github-copilot"`, computed from the resolved transcript.

use crate::types::{Message, ProviderHeaders, UserBlock, UserContent};

/// The `X-Initiator` value: `"agent"` when the last message is not a user message (a
/// follow-up after assistant or tool messages), else `"user"`. Copilot uses it to tell
/// user-initiated requests from agent-initiated ones.
pub fn infer_copilot_initiator(messages: &[Message]) -> &'static str {
    match messages.last() {
        Some(last) if !matches!(last, Message::User(_)) => "agent",
        _ => "user",
    }
}

/// Whether a user or tool result message carries an image. Copilot requires the
/// `Copilot-Vision-Request` header for such requests. A user message with plain string
/// content has no images.
pub fn has_copilot_vision_input(messages: &[Message]) -> bool {
    let has_image = |blocks: &[UserBlock]| blocks.iter().any(|block| matches!(block, UserBlock::Image(_)));
    messages.iter().any(|message| match message {
        Message::User(user) => match &user.content {
            UserContent::Blocks(blocks) => has_image(blocks),
            UserContent::Text(_) => false,
        },
        Message::ToolResult(result) => has_image(&result.content),
        _ => false,
    })
}

/// `X-Initiator`, `Openai-Intent: conversation-edits` and, when `has_images`,
/// `Copilot-Vision-Request: true`.
pub fn build_copilot_dynamic_headers(messages: &[Message], has_images: bool) -> ProviderHeaders {
    let mut headers = ProviderHeaders::new();
    headers.insert("X-Initiator".into(), Some(infer_copilot_initiator(messages).into()));
    headers.insert("Openai-Intent".into(), Some("conversation-edits".into()));
    if has_images {
        headers.insert("Copilot-Vision-Request".into(), Some("true".into()));
    }
    headers
}

/// The dynamic headers for a GitHub Copilot model, `None` for every other provider (the
/// `model.provider === "github-copilot"` branch each upstream API repeats).
pub fn copilot_dynamic_headers(provider: &str, messages: &[Message]) -> Option<ProviderHeaders> {
    (provider == "github-copilot").then(|| build_copilot_dynamic_headers(messages, has_copilot_vision_input(messages)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AssistantMessage, ImageContent, TextContent, ToolResultMessage, UserMessage};

    fn user(content: UserContent) -> Message {
        Message::User(UserMessage { content, timestamp: 1 })
    }

    fn image() -> UserBlock {
        UserBlock::Image(ImageContent {
            data: "AAAA".into(),
            mime_type: "image/png".into(),
        })
    }

    fn tool_result(content: Vec<UserBlock>) -> Message {
        Message::ToolResult(ToolResultMessage {
            tool_call_id: "call_1".into(),
            tool_name: "read".into(),
            content,
            ..ToolResultMessage::default()
        })
    }

    fn assistant() -> Message {
        Message::Assistant(AssistantMessage::empty("openai-responses", "github-copilot", "gpt-5"))
    }

    #[test]
    fn initiator_is_user_only_when_the_last_message_is_a_user_message() {
        assert_eq!(infer_copilot_initiator(&[]), "user");
        assert_eq!(infer_copilot_initiator(&[user("hi".into())]), "user");
        assert_eq!(infer_copilot_initiator(&[user("hi".into()), assistant()]), "agent");
        assert_eq!(
            infer_copilot_initiator(&[user("hi".into()), assistant(), tool_result(vec![])]),
            "agent"
        );
    }

    #[test]
    fn vision_input_counts_images_in_user_and_tool_result_messages() {
        assert!(!has_copilot_vision_input(&[user("hi".into())]));
        let text = UserBlock::Text(TextContent::new("hi"));
        assert!(!has_copilot_vision_input(&[user(UserContent::Blocks(vec![text]))]));
        assert!(has_copilot_vision_input(&[user(UserContent::Blocks(vec![image()]))]));
        assert!(has_copilot_vision_input(&[
            user("hi".into()),
            tool_result(vec![image()])
        ]));
    }

    #[test]
    fn dynamic_headers_add_the_vision_marker_only_with_images() {
        let messages = [user("hi".into())];
        let headers = build_copilot_dynamic_headers(&messages, false);
        assert_eq!(headers.get("X-Initiator"), Some(&Some("user".into())));
        assert_eq!(headers.get("Openai-Intent"), Some(&Some("conversation-edits".into())));
        assert!(!headers.contains_key("Copilot-Vision-Request"));
        let headers = build_copilot_dynamic_headers(&messages, true);
        assert_eq!(headers.get("Copilot-Vision-Request"), Some(&Some("true".into())));
        assert!(copilot_dynamic_headers("openai", &messages).is_none());
        assert!(copilot_dynamic_headers("github-copilot", &messages).is_some());
    }
}
