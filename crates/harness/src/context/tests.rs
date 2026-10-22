//! Context derivation rules on fixed entries (upstream `harness-context`).

use delta_ai::types::{
    AssistantBlock, AssistantMessage, Message, StopReason, SystemContent, SystemMessage, TextContent, ToolCall,
    ToolResultMessage, UserBlock, UserMessage,
};
use indexmap::IndexMap;

use super::{MISSING_RESULT_TEXT, derive_view, order_tool_results};
use crate::prompt::SYSTEM_ENTRY;
use crate::records::{ContextEdit, ConversationId, EditAction, EntryId, EntryRecord};

fn user(text: &str) -> Message {
    Message::User(UserMessage {
        content: text.into(),
        timestamp: 0,
    })
}

fn assistant(text: &str, stop: StopReason) -> Message {
    Message::Assistant(AssistantMessage {
        content: vec![AssistantBlock::Text(TextContent::new(text))],
        stop_reason: stop,
        ..AssistantMessage::default()
    })
}

fn calling(ids: &[&str]) -> Message {
    let content = ids
        .iter()
        .map(|id| {
            AssistantBlock::ToolCall(ToolCall {
                id: (*id).to_owned(),
                name: "read".to_owned(),
                ..ToolCall::default()
            })
        })
        .collect();
    Message::Assistant(AssistantMessage {
        content,
        stop_reason: StopReason::ToolUse,
        timestamp: 7,
        ..AssistantMessage::default()
    })
}

fn result(id: &str) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: id.to_owned(),
        tool_name: "read".to_owned(),
        content: vec![UserBlock::text(format!("result {id}"))],
        ..ToolResultMessage::default()
    })
}

fn system(key: &str) -> Message {
    let mut sections = IndexMap::new();
    sections.insert(key.to_owned(), Some(format!("<{key}>")));
    Message::System(SystemMessage {
        content: SystemContent::Text(String::new()),
        sections: Some(sections),
        ..SystemMessage::default()
    })
}

fn entry(id: u64, kind: &str, model: Vec<Message>) -> EntryRecord {
    let mut record = EntryRecord::new(EntryId(id), ConversationId(1), kind);
    record.model = Some(model);
    record
}

fn texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .map(|message| match message {
            Message::User(user) => format!("user:{}", user.content.text()),
            Message::Assistant(assistant) if assistant.tool_calls().next().is_some() => {
                let ids: Vec<&str> = assistant.tool_calls().map(|call| call.id.as_str()).collect();
                format!("calls:{}", ids.join(","))
            }
            Message::Assistant(assistant) => format!("assistant:{}", assistant.text()),
            Message::ToolResult(result) => format!("result:{}", result.tool_call_id),
            Message::System(system) => {
                let keys: Vec<&String> = system.sections.iter().flatten().map(|(key, _)| key).collect();
                format!("system:{keys:?}")
            }
        })
        .collect()
}

#[test]
fn edits_omit_and_replace_with_the_newest_edit_winning() {
    let mut first_edit = EntryRecord::new(EntryId(3), ConversationId(1), "note");
    first_edit.edits = Some(vec![ContextEdit {
        target: EntryId(1),
        action: EditAction::Replace(vec![user("first replacement")]),
    }]);
    let mut second_edit = EntryRecord::new(EntryId(4), ConversationId(1), "note");
    second_edit.edits = Some(vec![
        ContextEdit {
            target: EntryId(1),
            action: EditAction::Replace(vec![user("second replacement")]),
        },
        ContextEdit {
            target: EntryId(2),
            action: EditAction::Omit,
        },
    ]);
    let entries = vec![
        entry(1, "delta.user", vec![user("original")]),
        entry(2, "delta.assistant", vec![assistant("dropped", StopReason::Stop)]),
        first_edit,
        second_edit,
    ];
    let view = derive_view(None, &entries);
    assert_eq!(texts(&view.messages), ["user:second replacement"]);
    assert_eq!(view.entries.len(), 4);
    assert!(view.contributions[1].is_empty());
}

#[test]
fn a_head_marker_starts_the_active_range_and_drops_older_markers() {
    let mut old_marker = entry(2, "delta.compaction", vec![user("old summary")]);
    old_marker.head = Some(EntryId(1));
    let mut marker = entry(5, "delta.compaction", vec![user("summary")]);
    marker.head = Some(EntryId(3));
    let entries = [
        old_marker,
        entry(3, "delta.user", vec![user("kept")]),
        entry(4, "delta.assistant", vec![assistant("answer", StopReason::Stop)]),
        marker.clone(),
        entry(6, "delta.user", vec![user("after")]),
    ];
    // The scan starts at the marker's head; the older marker before it is out of range.
    let view = derive_view(Some(&marker), &entries[1..]);
    assert_eq!(
        texts(&view.messages),
        ["user:summary", "user:kept", "assistant:answer", "user:after"]
    );
    assert_eq!(view.head.as_ref().map(|head| head.id), Some(EntryId(5)));
    let ids: Vec<u64> = view.entries.iter().map(|entry| entry.id.0).collect();
    assert_eq!(ids, [5, 3, 4, 6]);
}

#[test]
fn failed_aborted_and_deferred_answers_contribute_nothing() {
    let entries = vec![
        entry(1, "delta.user", vec![user("question")]),
        entry(2, "delta.assistant", vec![assistant("partial", StopReason::Aborted)]),
        entry(3, "delta.assistant", vec![assistant("boom", StopReason::Error)]),
        entry(4, "delta.assistant", vec![assistant("later", StopReason::Deferred)]),
        entry(5, "delta.assistant", vec![assistant("answer", StopReason::Stop)]),
    ];
    let view = derive_view(None, &entries);
    assert_eq!(texts(&view.messages), ["user:question", "assistant:answer"]);
}

#[test]
fn results_follow_their_calls_missing_ones_are_synthesized_and_strays_dropped() {
    let messages = vec![
        user("go"),
        calling(&["a", "b"]),
        user("steer"),
        result("b"),
        result("stray"),
        calling(&["c"]),
        result("a"),
    ];
    let ordered = order_tool_results(&messages);
    assert_eq!(
        texts(&ordered),
        [
            "user:go",
            "calls:a,b",
            "result:a",
            "result:b",
            "user:steer",
            "calls:c",
            "result:c"
        ]
    );
    // `a` came after the next assistant message: it is missing for the first call.
    let Message::ToolResult(missing) = &ordered[2] else {
        panic!("a result")
    };
    assert!(missing.is_error);
    assert_eq!(missing.content, vec![UserBlock::text(MISSING_RESULT_TEXT)]);
    assert_eq!(missing.timestamp, 7);
    assert_eq!(missing.details, Some(serde_json::json!({ "reason": "missing_result" })));
    let Message::ToolResult(found) = &ordered[3] else {
        panic!("b result")
    };
    assert!(!found.is_error);
}

#[test]
fn the_baseline_leads_when_only_user_messages_precede_it() {
    let entries = vec![
        entry(1, "delta.user", vec![user("first")]),
        entry(2, SYSTEM_ENTRY, vec![system("tools")]),
    ];
    let view = derive_view(None, &entries);
    assert_eq!(texts(&view.messages), ["system:[\"tools\"]", "user:first"]);
    // The stored contributions keep the committed order.
    assert!(matches!(view.contributions[0][0], Message::User(_)));
}

#[test]
fn a_later_system_message_stays_in_place_without_a_baseline() {
    let mut carrying = entry(1, "imported", vec![system("own")]);
    carrying.model = Some(vec![user("imported")]);
    carrying.edits = Some(vec![ContextEdit {
        target: EntryId(9),
        action: EditAction::Replace(vec![system("own")]),
    }]);
    let entries = vec![
        carrying,
        entry(2, "delta.assistant", vec![assistant("answer", StopReason::Stop)]),
        entry(3, SYSTEM_ENTRY, vec![system("tools")]),
    ];
    let view = derive_view(None, &entries);
    assert_eq!(
        texts(&view.messages),
        ["user:imported", "assistant:answer", "system:[\"tools\"]"]
    );
}

#[test]
fn after_a_head_marker_the_baseline_leads_even_past_kept_answers() {
    let mut marker = entry(5, "delta.compaction", vec![user("summary")]);
    marker.head = Some(EntryId(2));
    // The rebaseline omits the retained system entry before the marker.
    let mut baseline = entry(6, SYSTEM_ENTRY, vec![system("baseline")]);
    baseline.edits = Some(vec![ContextEdit {
        target: EntryId(4),
        action: EditAction::Omit,
    }]);
    let entries = vec![
        entry(2, "delta.user", vec![user("kept")]),
        entry(3, "delta.assistant", vec![assistant("kept answer", StopReason::Stop)]),
        entry(4, SYSTEM_ENTRY, vec![system("old")]),
        marker.clone(),
        baseline,
        entry(7, "delta.user", vec![user("next")]),
    ];
    let view = derive_view(Some(&marker), &entries);
    assert_eq!(
        texts(&view.messages),
        [
            "system:[\"baseline\"]",
            "user:summary",
            "user:kept",
            "assistant:kept answer",
            "user:next"
        ]
    );
}
