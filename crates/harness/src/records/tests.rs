//! JSON shapes and round trips of the records, and their `check()` rules.

use delta_ai::types::{Message, UserMessage};
use delta_chord::delta::Op;
use delta_chord::json::{Json, Path};
use pretty_assertions::assert_eq;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use super::*;

/// Asserts that `record` serializes to `expected` and reads back equal.
fn round_trip<T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug>(record: &T, expected: Value) {
    assert_eq!(serde_json::to_value(record).unwrap(), expected);
    let back: T = serde_json::from_value(expected).unwrap();
    assert_eq!(&back, record);
}

fn data(value: Value) -> Json {
    Json::from(value)
}

#[test]
fn conversations_omit_absent_edges() {
    round_trip(&ConversationRecord::new(ROOT_CONVERSATION_ID), json!({ "id": 1 }));
    round_trip(
        &ConversationRecord {
            id: ConversationId(7),
            parent: Some(ConversationParent {
                conversation_id: ConversationId(1),
                at: EntryId(42),
            }),
            owner: Some(ConversationOwner {
                conversation_id: ConversationId(1),
                task_id: TaskId(40),
            }),
        },
        json!({
            "id": 7,
            "parent": { "conversation_id": 1, "at": 42 },
            "owner": { "conversation_id": 1, "task_id": 40 },
        }),
    );
}

#[test]
fn ids_are_numbers_past_javascripts_safe_range() {
    let id = EntryId(u64::MAX);
    assert_eq!(serde_json::to_string(&id).unwrap(), u64::MAX.to_string());
    assert_eq!(EntryId::from_raw(9).get(), 9);
    assert_eq!(TaskId::from(RawId(3)), TaskId(3));
    assert_eq!(Seq(4).to_string(), "4");
}

#[test]
fn entries_keep_null_data_and_flat_edits() {
    let mut entry = EntryRecord::new(EntryId(5), ROOT_CONVERSATION_ID, "delta.user");
    entry.data = Some(Json::Null);
    entry.head = Some(EntryId(3));
    entry.by_task_id = Some(TaskId(4));
    let message = Message::User(UserMessage {
        content: "hi".into(),
        timestamp: 1,
    });
    entry.edits = Some(vec![
        ContextEdit {
            target: EntryId(2),
            action: EditAction::Omit,
        },
        ContextEdit {
            target: EntryId(3),
            action: EditAction::Replace(vec![message.clone()]),
        },
    ]);
    entry.model = Some(vec![message]);
    round_trip(
        &entry,
        json!({
            "id": 5,
            "conversation_id": 1,
            "kind": "delta.user",
            "model": [{ "role": "user", "content": "hi", "timestamp": 1 }],
            "data": null,
            "head": 3,
            "edits": [
                { "target": 2, "action": "omit" },
                {
                    "target": 3,
                    "action": "replace",
                    "messages": [{ "role": "user", "content": "hi", "timestamp": 1 }],
                },
            ],
            "by_task_id": 4,
        }),
    );
    round_trip(
        &EntryRecord::new(EntryId(6), ROOT_CONVERSATION_ID, "note"),
        json!({ "id": 6, "conversation_id": 1, "kind": "note" }),
    );
}

#[test]
fn malformed_edits_are_refused() {
    for edit in [
        json!({ "target": 2, "action": "omit", "messages": [] }),
        json!({ "target": 2, "action": "replace" }),
        json!({ "target": 2, "action": "drop" }),
    ] {
        assert!(serde_json::from_value::<ContextEdit>(edit.clone()).is_err(), "{edit}");
    }
}

#[test]
fn typed_entry_kinds_convert_data() {
    #[derive(Debug, PartialEq, Serialize, serde::Deserialize)]
    struct Note {
        text: String,
    }
    const NOTE: EntryKind<Note> = EntryKind::new("app.note");

    let draft = NOTE.encode(TypedEntryDraft::new(Note { text: "hi".into() })).unwrap();
    assert_eq!(draft.kind, "app.note");
    assert_eq!(draft.data, Some(data(json!({ "text": "hi" }))));

    let mut record = EntryRecord::new(EntryId(2), ROOT_CONVERSATION_ID, "app.note");
    record.data = draft.data;
    let typed = NOTE.decode(record.clone()).unwrap().unwrap();
    assert_eq!(typed.data, Note { text: "hi".into() });

    record.kind = "other".into();
    assert!(NOTE.decode(record.clone()).unwrap().is_none());
    record.kind = "app.note".into();
    record.data = Some(data(json!(3)));
    assert!(matches!(NOTE.decode(record), Err(RecordError::EntryData { .. })));
}

fn submission(kind: SubmissionType, state: SubmissionState) -> SubmissionRecord {
    SubmissionRecord {
        id: SubmissionId(9),
        conversation_id: ROOT_CONVERSATION_ID,
        request_id: Some("r1".into()),
        kind,
        state,
    }
}

#[test]
fn submissions_flatten_their_status() {
    round_trip(
        &submission(SubmissionType::Input, SubmissionState::Placed { entry: EntryId(12) }),
        json!({ "id": 9, "conversation_id": 1, "request_id": "r1", "type": "input", "status": "placed", "entry": 12 }),
    );
    round_trip(
        &submission(
            SubmissionType::Write,
            SubmissionState::Done {
                entry: EntryId(12),
                answer: None,
            },
        ),
        json!({ "id": 9, "conversation_id": 1, "request_id": "r1", "type": "write", "status": "done", "entry": 12 }),
    );
    round_trip(
        &submission(
            SubmissionType::Input,
            SubmissionState::Unanswered {
                entry: None,
                reason: "closed".into(),
                detail: Some(Json::Null),
            },
        ),
        json!({
            "id": 9, "conversation_id": 1, "request_id": "r1", "type": "input",
            "status": "unanswered", "reason": "closed", "detail": null,
        }),
    );
    assert_eq!(
        submission(SubmissionType::Input, SubmissionState::Queued)
            .state
            .status(),
        SubmissionStatus::Queued
    );
}

#[test]
fn submission_check_refuses_illegal_combinations() {
    let done = |answer| SubmissionState::Done {
        entry: EntryId(1),
        answer,
    };
    let unanswered = |entry| SubmissionState::Unanswered {
        entry,
        reason: "failed".into(),
        detail: None,
    };
    for (kind, state) in [
        (SubmissionType::Write, SubmissionState::Placed { entry: EntryId(1) }),
        (SubmissionType::Write, done(Some(EntryId(2)))),
        (SubmissionType::Write, unanswered(Some(EntryId(1)))),
        (SubmissionType::Input, done(None)),
    ] {
        let error = submission(kind, state).check().unwrap_err();
        assert!(error.to_string().starts_with("Submission 9: "), "{error}");
    }
    for (kind, state) in [
        (SubmissionType::Input, SubmissionState::Placed { entry: EntryId(1) }),
        (SubmissionType::Input, done(Some(EntryId(2)))),
        (SubmissionType::Input, unanswered(Some(EntryId(1)))),
        (SubmissionType::Write, done(None)),
        (SubmissionType::Write, unanswered(None)),
    ] {
        submission(kind, state).check().unwrap();
    }
}

fn task(state: TaskState) -> TaskRecord {
    TaskRecord::new(
        TaskId(40),
        ROOT_CONVERSATION_ID,
        "test.task",
        1,
        data(json!({ "value": 40 })),
        state,
    )
}

#[test]
fn tasks_tag_states_and_outcomes_in_snake_case() {
    round_trip(
        &task(TaskState::Pending {
            checkpoint: data(json!({ "phase": "ready" })),
        }),
        json!({
            "id": 40, "conversation_id": 1, "kind": "test.task", "version": 1, "input": { "value": 40 },
            "background": false, "abort_requested": false,
            "state": { "status": "pending", "checkpoint": { "phase": "ready" } },
        }),
    );

    let mut waiting = task(TaskState::Waiting {
        checkpoint: Json::Null,
        on: vec![TaskId(41)],
        policy: JoinPolicy::AllSettled,
    });
    waiting.owner = Some(TaskId(39));
    waiting.abandon_on_restart = true;
    waiting.abort_reason = Some(AbortReason::Restart);
    waiting.started_at = Some(1_000);
    waiting.restarts = Some(2);
    waiting.memos = Some([("kept".to_owned(), Json::Bool(true))].into_iter().collect());
    waiting.trace = Some(data(json!({ "traceparent": "00-1" })));
    waiting.links = Some(vec![data(json!({ "span": 1 }))]);
    round_trip(
        &waiting,
        json!({
            "id": 40, "conversation_id": 1, "kind": "test.task", "version": 1, "input": { "value": 40 },
            "owner": 39, "background": false, "abort_requested": false, "abort_reason": "restart",
            "abandon_on_restart": true, "started_at": 1000, "restarts": 2,
            "state": { "status": "waiting", "checkpoint": null, "on": [41], "policy": "all_settled" },
            "memos": { "kept": true },
            "trace": { "traceparent": "00-1" },
            "links": [{ "span": 1 }],
        }),
    );

    let terminal = task(TaskState::Terminal {
        outcome: TaskOutcome::Failed {
            error: TaskOutcomeError::new("boom"),
            result: Some(Json::Null),
        },
    });
    let value = serde_json::to_value(&terminal).unwrap();
    assert_eq!(
        value["state"],
        json!({ "status": "terminal", "outcome": { "status": "failed", "error": { "message": "boom" }, "result": null } })
    );
    assert_eq!(serde_json::from_value::<TaskRecord>(value).unwrap(), terminal);
    assert_eq!(terminal.state.status(), TaskStatus::Terminal);
    assert!(terminal.state.checkpoint().is_none());
    assert!(terminal.state.outcome().is_some());
}

#[test]
fn completed_outcomes_require_a_result() {
    let missing = json!({ "status": "completed" });
    assert!(serde_json::from_value::<TaskOutcome>(missing).is_err());
    let null = serde_json::from_value::<TaskOutcome>(json!({ "status": "completed", "result": null })).unwrap();
    assert_eq!(null, TaskOutcome::Completed { result: Json::Null });
}

#[test]
fn tasks_accept_unknown_fields() {
    let mut value = serde_json::to_value(task(TaskState::Running { checkpoint: Json::Null })).unwrap();
    value["future_field"] = json!(1);
    let read: TaskRecord = serde_json::from_value(value).unwrap();
    assert_eq!(read, task(TaskState::Running { checkpoint: Json::Null }));
}

#[test]
fn task_check_refuses_memos_after_the_outcome() {
    let mut completing = task(TaskState::Completing {
        outcome: TaskOutcome::Completed { result: Json::Null },
    });
    completing.check().unwrap();
    completing.memos = Some(indexmap::IndexMap::new());
    assert_eq!(
        completing.check().unwrap_err().to_string(),
        "Task 40: a completing task keeps no memos"
    );
}

fn live_record() -> DocumentRecord {
    DocumentRecord {
        id: DocumentId(12),
        kind: "delta.live".into(),
        key: None,
        scope: DocumentScope::Conversation {
            conversation_id: ROOT_CONVERSATION_ID,
        },
        policy: Some(ConversationPolicy {
            history: History::Latest,
            fork: Fork::Initial,
        }),
        created_at: Seq(3),
        retired_at: None,
    }
}

#[test]
fn documents_flatten_their_policy() {
    let record = live_record();
    assert_eq!(
        serde_json::to_string(&record).unwrap(),
        r#"{"id":12,"kind":"delta.live","scope":{"kind":"conversation","conversation_id":1},"history":"latest","fork":"initial","created_at":3}"#
    );
    round_trip(
        &DocumentRecord {
            key: Some("call-7".into()),
            retired_at: Some(Seq(9)),
            ..record
        },
        json!({
            "id": 12, "kind": "delta.live", "key": "call-7",
            "scope": { "kind": "conversation", "conversation_id": 1 },
            "history": "latest", "fork": "initial", "created_at": 3, "retired_at": 9,
        }),
    );
    round_trip(
        &DocumentRecord {
            id: DocumentId(2),
            kind: "s".into(),
            key: Some(String::new()),
            scope: DocumentScope::Task { task_id: TaskId(4) },
            policy: None,
            created_at: Seq(1),
            retired_at: None,
        },
        json!({ "id": 2, "kind": "s", "key": "", "scope": { "kind": "task", "task_id": 4 }, "created_at": 1 }),
    );
}

#[test]
fn document_check_matches_policy_to_scope() {
    let mut record = live_record();
    record.check().unwrap();
    record.policy = Some(ConversationPolicy {
        history: History::Latest,
        fork: Fork::AsOf,
    });
    assert!(record.check().is_err());
    record.policy = None;
    assert!(record.check().is_err());
    record.scope = DocumentScope::Session;
    record.check().unwrap();
    record.policy = Some(ConversationPolicy {
        history: History::Rewindable,
        fork: Fork::AsOf,
    });
    assert!(record.check().is_err());
}

#[test]
fn document_lifetimes_are_half_open() {
    let mut record = live_record();
    record.retired_at = Some(Seq(9));
    assert!(!record.is_alive_at(DocumentPoint::At(Seq(2))));
    assert!(record.is_alive_at(DocumentPoint::At(Seq(3))));
    assert!(record.is_alive_at(DocumentPoint::At(Seq(8))));
    assert!(!record.is_alive_at(DocumentPoint::At(Seq(9))));
    assert!(!record.is_alive_at(DocumentPoint::Current));
    assert!(!record.keeps_history());
    record.policy = Some(ConversationPolicy {
        history: History::Rewindable,
        fork: Fork::AsOf,
    });
    assert!(record.keeps_history());
}

#[test]
fn points_contents_and_writes_have_stable_shapes() {
    round_trip(&DocumentPoint::Current, json!("current"));
    round_trip(&DocumentPoint::At(Seq(4)), json!(4));
    assert!(serde_json::from_value::<DocumentPoint>(json!("latest")).is_err());

    let delta = DocumentContent::Delta {
        version: 2,
        ops: vec![Op::Set(Path::from_iter(["count"]), Json::from(2))],
    };
    round_trip(
        &delta,
        json!({ "kind": "delta", "version": 2, "ops": [["s", ["count"], 2]] }),
    );
    assert_eq!(delta.version(), 2);

    round_trip(
        &StorageWrite::DocumentChange {
            id: DocumentId(5),
            content: delta,
        },
        json!({
            "type": "document_change",
            "value": { "id": 5, "content": { "kind": "delta", "version": 2, "ops": [["s", ["count"], 2]] } },
        }),
    );
    round_trip(
        &StorageWrite::Submission(submission(SubmissionType::Input, SubmissionState::Queued)),
        json!({
            "type": "submission",
            "value": { "id": 9, "conversation_id": 1, "request_id": "r1", "type": "input", "status": "queued" },
        }),
    );
    round_trip(
        &StorageWrite::DocumentCopy {
            record: DocumentCreate {
                id: DocumentId(6),
                kind: "k".into(),
                key: None,
                scope: DocumentScope::Session,
                policy: None,
            },
            source: DocumentCopySource {
                id: DocumentId(5),
                at: DocumentPoint::Current,
            },
        },
        json!({
            "type": "document_copy",
            "value": {
                "record": { "id": 6, "kind": "k", "scope": { "kind": "session" } },
                "source": { "id": 5, "at": "current" },
            },
        }),
    );
}
