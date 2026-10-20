//! Inbox JSON shapes and boundary predicates.

use delta_ai::types::{Message, UserContent};

use super::{InboxItem, InputEntry, draft_from_json, draft_to_json, has_queued_reset, is_display_only};
use crate::records::{EntryDraft, EntryId, Head, SubmissionId};

#[test]
fn items_are_mode_tagged_snake_case() {
    let item = InboxItem::FollowUp {
        id: SubmissionId(7),
        content: UserContent::Text("hi".to_owned()),
        entry: Some(InputEntry {
            kind: "app.note".to_owned(),
            data: None,
        }),
    };
    assert_eq!(
        serde_json::to_value(&item).unwrap(),
        serde_json::json!({ "mode": "follow_up", "id": 7, "content": "hi", "entry": { "kind": "app.note" } })
    );
}

#[test]
fn write_drafts_round_trip_with_self_heads() {
    let draft = EntryDraft {
        model: Some(vec![Message::user("handoff")]),
        head: Some(Head::SelfRef),
        data: Some(delta_chord::json::Json::Null),
        ..EntryDraft::new("delta.reset")
    };
    let json = draft_to_json(&draft).unwrap();
    assert_eq!(json.get("head").and_then(delta_chord::json::Json::as_str), Some("self"));
    assert_eq!(draft_from_json(&json).unwrap(), draft);
    let items = vec![InboxItem::Write {
        id: SubmissionId(1),
        entry: json,
    }];
    assert!(has_queued_reset(&items));
    let numeric = EntryDraft {
        head: Some(Head::At(EntryId(3))),
        ..EntryDraft::new("app.summary")
    };
    assert_eq!(draft_from_json(&draft_to_json(&numeric).unwrap()).unwrap(), numeric);
}

#[test]
fn display_only_means_no_model_head_or_edits() {
    assert!(is_display_only(&EntryDraft::new("note")));
    let empty_edits = EntryDraft {
        edits: Some(Vec::new()),
        ..EntryDraft::new("note")
    };
    assert!(!is_display_only(&empty_edits));
}
