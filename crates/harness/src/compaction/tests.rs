#![allow(clippy::unwrap_used)]

use delta_ai::estimate::estimate_message_tokens;
use delta_ai::types::{AssistantBlock, AssistantMessage, Message, TextContent, ToolCall, ToolResultMessage, UserBlock};

use super::*;
use crate::context::derive_view;
use crate::model::ModelRef;
use crate::records::{EntryRecord, ROOT_CONVERSATION_ID};

fn entry(id: u64, message: Message) -> EntryRecord {
    let mut entry = EntryRecord::new(EntryId(id), ROOT_CONVERSATION_ID, "test");
    entry.model = Some(vec![message]);
    entry
}

fn user(text: &str) -> Message {
    Message::user(text)
}

fn assistant(text: &str, calls: &[&str]) -> Message {
    let mut message = AssistantMessage::empty("fake", "fake", "model");
    message.content.push(AssistantBlock::Text(TextContent::new(text)));
    for id in calls {
        message.content.push(AssistantBlock::ToolCall(ToolCall {
            id: (*id).to_owned(),
            name: "read".to_owned(),
            arguments: [("path".to_owned(), serde_json::json!("a.rs"))].into_iter().collect(),
            thought_signature: None,
            namespace: None,
            arguments_error: None,
        }));
    }
    Message::Assistant(message)
}

fn result(call: &str, text: &str) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: call.to_owned(),
        tool_name: "read".to_owned(),
        content: vec![UserBlock::text(text)],
        details: None,
        usage: None,
        nested_calls: None,
        is_error: false,
        timestamp: 0,
        duration_ms: None,
    })
}

fn tokens(view: &ContextView, from: usize) -> u64 {
    view.contributions[from..]
        .iter()
        .flatten()
        .map(estimate_message_tokens)
        .sum()
}

#[test]
fn the_cut_keeps_a_tool_round_together() {
    // The spec example: the walk reaches the limit at the long result; the cut moves to the
    // next candidate, the assistant after it.
    let entries = vec![
        entry(1, user("start")),
        entry(2, assistant("reading", &["c1"])),
        entry(3, result("c1", &"x".repeat(120_000))),
        entry(4, assistant("read it", &[])),
        entry(5, user("thanks")),
        entry(6, assistant("welcome", &[])),
    ];
    let view = derive_view(None, &entries);
    let cut = select_cut(&view, 20_000).unwrap();
    assert_eq!(view.entries[cut].id, EntryId(4));
    let summarized = summarized_messages(&view, cut);
    let roles: Vec<&str> = summarized.iter().map(Message::role).collect();
    assert_eq!(roles, ["user", "assistant", "toolResult"]);
    // Nothing to compact below the limit.
    assert_eq!(select_cut(&view, u64::MAX), None);
}

#[test]
fn a_user_entry_before_a_pending_result_is_no_candidate() {
    let entries = vec![
        entry(1, user(&"old ".repeat(2000))),
        entry(2, assistant("calling", &["c1"])),
        entry(3, user("steering")),
        entry(4, result("c1", "done")),
        entry(5, assistant("answer", &[])),
    ];
    let view = derive_view(None, &entries);
    // The walk reaches the limit at the user entry 3; its call's result still follows.
    let cut = select_cut(&view, tokens(&view, 2)).unwrap();
    assert_eq!(view.entries[cut].id, EntryId(5));
}

#[test]
fn nothing_before_the_cut_means_no_cut() {
    let entries = vec![entry(1, assistant(&"long ".repeat(10_000), &[]))];
    let view = derive_view(None, &entries);
    assert_eq!(select_cut(&view, 10), None);
}

#[test]
fn conversations_serialize_as_transcripts() {
    let messages = vec![
        user("fix the bug"),
        assistant("looking", &["c1"]),
        result("c1", &"y".repeat(2005)),
    ];
    let text = serialize_conversation(&messages);
    let parts: Vec<&str> = text.split("\n\n").collect();
    assert_eq!(parts[0], "[User]: fix the bug");
    assert_eq!(parts[1], "[Assistant]: looking");
    assert_eq!(parts[2], r#"[Assistant tool calls]: read(path="a.rs")"#);
    assert!(parts[3].starts_with("[Tool result]: yyy"));
    assert_eq!(parts[4], "[... 5 more characters truncated]");
}

#[test]
fn checkpoints_are_flat() {
    let request = SummaryRequest {
        attempt: 2,
        model: ModelRef::new("fake", "model-1"),
        thinking_level: delta_ai::model::ModelThinkingLevel::Off,
        stream_options: crate::model::ConversationStreamOptions::default(),
        max_tokens: 800,
        tail: EntryId(9),
        first_kept: EntryId(4),
    };
    let retry = CompactionCheckpoint::Retry { request, until: 5 };
    let json = serde_json::to_value(&retry).unwrap();
    assert_eq!(json["phase"], "retry");
    assert_eq!(json["attempt"], 2);
    assert_eq!(json["first_kept"], 4);
    assert_eq!(json["until"], 5);
    assert_eq!(serde_json::from_value::<CompactionCheckpoint>(json).unwrap(), retry);
    assert_eq!(
        serde_json::to_value(CompactionCheckpoint::Select).unwrap(),
        serde_json::json!({ "phase": "select" })
    );
    assert_eq!(
        serde_json::to_value(CompactionResult::default()).unwrap(),
        serde_json::json!({})
    );
}
