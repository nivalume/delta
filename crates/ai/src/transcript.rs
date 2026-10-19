//! System messages in the transcript: normalization of a [`Context`], and replay of every
//! system message into the current prompt and tool set.

use indexmap::IndexMap;

use crate::types::{Context, Message, SystemContent, SystemMessage, Tool, ToolReference, TranscriptContext};

/// Build the leading system message for a prompt and tool set. `None` when both are empty.
pub fn create_initial_system_message(system_prompt: Option<&str>, tools: Option<&[Tool]>) -> Option<SystemMessage> {
    let prompt = system_prompt.unwrap_or_default();
    let tools = tools.unwrap_or_default();
    if prompt.is_empty() && tools.is_empty() {
        return None;
    }
    Some(SystemMessage {
        content: SystemContent::Text(prompt.to_owned()),
        tools_added: (!tools.is_empty()).then(|| tools.to_vec()),
        ..SystemMessage::default()
    })
}

/// Fold `Context::system_prompt` and `Context::tools` into a leading system message. This is
/// the entry point that produces a [`TranscriptContext`].
pub fn normalize_context(context: &Context) -> TranscriptContext {
    let initial = create_initial_system_message(context.system_prompt.as_deref(), context.tools.as_deref());
    let mut messages = Vec::with_capacity(context.messages.len() + 1);
    if let Some(initial) = initial {
        messages.push(Message::System(initial));
    }
    messages.extend(context.messages.iter().cloned());
    TranscriptContext::from_messages(messages)
}

fn system_messages(messages: &[Message]) -> impl Iterator<Item = &SystemMessage> {
    messages.iter().filter_map(|message| match message {
        Message::System(system) => Some(system),
        _ => None,
    })
}

/// The leading system message, if the transcript starts with one.
pub fn get_initial_system_message(messages: &[Message]) -> Option<&SystemMessage> {
    match messages.first() {
        Some(Message::System(system)) => Some(system),
        _ => None,
    }
}

/// The messages after the leading system message, for APIs that carry the prompt outside
/// the message list.
pub fn without_initial_system_message(messages: &[Message]) -> &[Message] {
    if get_initial_system_message(messages).is_some() {
        &messages[1..]
    } else {
        messages
    }
}

/// The tools available after applying every transcript delta in order.
pub fn get_current_tools(messages: &[Message]) -> Vec<Tool> {
    let mut tools: IndexMap<String, Tool> = IndexMap::new();
    for system in system_messages(messages) {
        for removed in system.tools_removed.iter().flatten() {
            tools.shift_remove(&removed.name);
        }
        for added in system.tools_added.iter().flatten() {
            tools.insert(added.name.clone(), added.clone());
        }
    }
    tools.into_values().collect()
}

/// Replay every system message into one leading system message holding the current prompt
/// and tools. Later `content` is appended, `sections` are patched by name.
pub fn get_current_system_message(messages: &[Message]) -> Option<SystemMessage> {
    let mut content = Vec::new();
    let mut sections: IndexMap<String, String> = IndexMap::new();
    let mut timestamp = None;
    for system in system_messages(messages) {
        timestamp.get_or_insert(system.timestamp);
        let text = system.content.text();
        if !text.is_empty() {
            content.push(text);
        }
        for (name, value) in system.sections.iter().flatten() {
            match value {
                Some(value) => {
                    sections.insert(name.clone(), value.clone());
                }
                None => {
                    sections.shift_remove(name);
                }
            }
        }
    }
    let tools = get_current_tools(messages);
    if timestamp.is_none() && tools.is_empty() {
        return None;
    }
    Some(SystemMessage {
        content: SystemContent::Text(content.join("\n\n")),
        sections: (!sections.is_empty())
            .then(|| sections.into_iter().map(|(name, value)| (name, Some(value))).collect()),
        tools_added: (!tools.is_empty()).then_some(tools),
        tools_removed: None,
        timestamp: timestamp.unwrap_or(0),
    })
}

/// A system message as a complete prompt: its content followed by its sections.
pub fn get_system_message_text(message: &SystemMessage) -> String {
    let mut parts = vec![message.content.text()];
    parts.extend(message.sections.iter().flatten().filter_map(|(_, value)| value.clone()));
    parts.retain(|part| !part.is_empty());
    parts.join("\n\n")
}

/// The current system prompt text after replaying every system message.
pub fn get_current_system_prompt(messages: &[Message]) -> String {
    get_current_system_message(messages)
        .map(|message| get_system_message_text(&message))
        .unwrap_or_default()
}

/// Render a later system message for APIs that accept system messages mid-conversation.
/// Section changes are framed by name so the model can relate them to the leading prompt.
pub fn render_system_message_update(message: &SystemMessage) -> String {
    let mut parts = Vec::new();
    let text = message.content.text();
    if !text.is_empty() {
        parts.push(text);
    }
    for (name, value) in message.sections.iter().flatten() {
        parts.push(match value {
            None => format!("Removed system prompt section \"{name}\"."),
            Some(value) => format!("Updated system prompt section \"{name}\":\n\n{value}"),
        });
    }
    parts.join("\n\n")
}

/// Rebuild the transcript for APIs without mid-conversation system messages: the replayed
/// system message leads, and every later system message is dropped.
pub fn collapse_system_messages(context: &TranscriptContext) -> TranscriptContext {
    let head = get_current_system_message(context.messages());
    let mut messages: Vec<Message> = Vec::with_capacity(context.messages().len());
    if let Some(head) = head {
        messages.push(Message::System(head));
    }
    messages.extend(
        context
            .messages()
            .iter()
            .filter(|message| !matches!(message, Message::System(_)))
            .cloned(),
    );
    TranscriptContext::from_messages(messages)
}

/// Keep later system messages in place when the model accepts them; otherwise collapse them.
pub fn resolve_transcript(context: &TranscriptContext, supports_mid_convo_system_messages: bool) -> TranscriptContext {
    if supports_mid_convo_system_messages {
        context.clone()
    } else {
        collapse_system_messages(context)
    }
}

/// Whether two tools declare the same interface to the model.
pub fn declarations_equal(left: &Tool, right: &Tool) -> bool {
    left.name == right.name
        && left.description == right.description
        && left.parameters == right.parameters
        && left.constrained_sampling == right.constrained_sampling
}

/// Tool changes between two complete tool states. A changed definition is a removal
/// followed by an addition.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolStateChanges {
    pub tools_added: Vec<Tool>,
    pub tools_removed: Vec<ToolReference>,
}

/// Compare two complete tool states.
pub fn get_tool_state_changes(previous: &[Tool], current: &[Tool]) -> ToolStateChanges {
    let find = |tools: &[Tool], name: &str| tools.iter().find(|tool| tool.name == name).cloned();
    ToolStateChanges {
        tools_added: current
            .iter()
            .filter(|tool| find(previous, &tool.name).is_none_or(|old| !declarations_equal(&old, tool)))
            .cloned()
            .collect(),
        tools_removed: previous
            .iter()
            .filter(|tool| find(current, &tool.name).is_none_or(|new| !declarations_equal(tool, &new)))
            .map(|tool| ToolReference {
                name: tool.name.clone(),
            })
            .collect(),
    }
}

/// Every definition referenced by transcript tool state, in first-declaration order (the
/// latest definition of each name).
pub fn get_declared_tools(messages: &[Message]) -> Vec<Tool> {
    let mut definitions: IndexMap<String, Tool> = IndexMap::new();
    for system in system_messages(messages) {
        for tool in system.tools_added.iter().flatten() {
            definitions.insert(tool.name.clone(), tool.clone());
        }
    }
    definitions.into_values().collect()
}

/// Whether tool history contains a removal or same-name redeclaration that an addition-only
/// transport cannot replay.
pub fn has_non_additive_tool_changes(messages: &[Message]) -> bool {
    let mut declared = std::collections::HashSet::new();
    for system in system_messages(messages) {
        if system.tools_removed.as_ref().is_some_and(|removed| !removed.is_empty()) {
            return true;
        }
        for tool in system.tools_added.iter().flatten() {
            if !declared.insert(tool.name.as_str()) {
                return true;
            }
        }
    }
    false
}

/// Split of tool declarations between the top-level request field and in-place additions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TranscriptTools {
    /// Tools sent in the top-level request field.
    pub request_tools: Vec<Tool>,
    /// Whether later system messages carry their own `tools_added` as in-place additions.
    /// When false, `request_tools` holds the complete current tool set.
    pub anchors_additions: bool,
}

/// Transports that can anchor additions at a system message keep the initial tools at the
/// top and load later ones where they appear; that only works when no tool was removed or
/// redeclared, so everything else sends the current tool list.
pub fn resolve_transcript_tools(messages: &[Message], supports_tool_additions: bool) -> TranscriptTools {
    let anchors_additions = supports_tool_additions && !has_non_additive_tool_changes(messages);
    TranscriptTools {
        request_tools: if anchors_additions {
            get_initial_system_message(messages)
                .and_then(|system| system.tools_added.clone())
                .unwrap_or_default()
        } else {
            get_current_tools(messages)
        },
        anchors_additions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(name: &str) -> Tool {
        Tool::new(name, name, json!({"type": "object"}))
    }

    #[test]
    fn replays_sections_and_tools() {
        let messages: Vec<Message> = vec![
            SystemMessage {
                content: "You are helpful.".into(),
                sections: Some(
                    [("rules".to_owned(), Some("<rules>Be brief.</rules>".to_owned()))]
                        .into_iter()
                        .collect(),
                ),
                tools_added: Some(vec![tool("read")]),
                timestamp: 1,
                ..SystemMessage::default()
            }
            .into(),
            Message::user("hi"),
            SystemMessage {
                content: "".into(),
                sections: Some(
                    [("rules".to_owned(), Some("<rules>Be thorough.</rules>".to_owned()))]
                        .into_iter()
                        .collect(),
                ),
                tools_removed: Some(vec![ToolReference { name: "read".into() }]),
                timestamp: 3,
                ..SystemMessage::default()
            }
            .into(),
        ];
        assert_eq!(
            get_current_system_prompt(&messages),
            "You are helpful.\n\n<rules>Be thorough.</rules>"
        );
        assert!(get_current_tools(&messages).is_empty());
        assert!(has_non_additive_tool_changes(&messages));
        let collapsed = collapse_system_messages(&TranscriptContext::from_messages(messages));
        assert_eq!(collapsed.messages().len(), 2);
    }

    #[test]
    fn normalize_adds_leading_system_message() {
        let context = Context::new(vec![Message::user("hi")])
            .with_system_prompt("p")
            .with_tools(vec![tool("a")]);
        let transcript = normalize_context(&context);
        assert_eq!(transcript.messages().len(), 2);
        assert_eq!(get_current_tools(transcript.messages()).len(), 1);
        assert_eq!(normalize_context(&Context::new(vec![])).messages().len(), 0);
    }
}
