//! Tool result ordering (spec §2.1, rules 7 and 8).

use std::collections::HashMap;

use delta_ai::types::{Message, ToolCall, ToolResultMessage, UserBlock};

/// The text of a result synthesized for a call whose result the history lacks.
pub const MISSING_RESULT_TEXT: &str = "Tool result unavailable: history ends before this call completed.";

/// Places each assistant's tool results, found before the next assistant message, directly
/// after it in call order; synthesizes an error result for every call without one and drops
/// results without a preceding call.
pub fn order_tool_results(messages: &[Message]) -> Vec<Message> {
    let mut ordered = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        match message {
            Message::ToolResult(_) => {}
            Message::Assistant(assistant) => {
                ordered.push(message.clone());
                let calls: Vec<&ToolCall> = assistant.tool_calls().collect();
                if calls.is_empty() {
                    continue;
                }
                let mut results: HashMap<&str, &Message> = HashMap::new();
                for candidate in messages[index + 1..]
                    .iter()
                    .take_while(|candidate| !matches!(candidate, Message::Assistant(_)))
                {
                    if let Message::ToolResult(result) = candidate {
                        results.entry(result.tool_call_id.as_str()).or_insert(candidate);
                    }
                }
                for call in calls {
                    match results.get(call.id.as_str()) {
                        Some(result) => ordered.push((*result).clone()),
                        None => ordered.push(missing_result(call, assistant.timestamp)),
                    }
                }
            }
            _ => ordered.push(message.clone()),
        }
    }
    ordered
}

fn missing_result(call: &ToolCall, timestamp: i64) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: call.id.clone(),
        tool_name: call.name.clone(),
        content: vec![UserBlock::text(MISSING_RESULT_TEXT)],
        details: Some(serde_json::json!({ "reason": "missing_result" })),
        is_error: true,
        timestamp,
        ..ToolResultMessage::default()
    })
}
