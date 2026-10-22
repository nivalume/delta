//! Definition semantics, addresses and the access checks.

use delta_chord::delta::Op;
use delta_chord::json;
use delta_chord::json::Json;
use pretty_assertions::assert_eq;

use super::*;
use crate::records::{
    ConversationId, ConversationPolicy, DocumentAddress, DocumentId, DocumentRecord, DocumentScope, DocumentScopeKind,
    Fork, History, Seq, TaskId,
};

struct Inbox;

impl DocDefinition for Inbox {
    const KIND: &'static str = "delta.inbox";
    const VERSION: u32 = 1;
    type Scope = ConversationScope<Latest, ForkInitial>;
    type Seed = ();

    fn initial((): &()) -> Json {
        json!({ "items": [] })
    }

    fn checkpoint_when(value: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        value
            .get("items")
            .and_then(Json::as_array)
            .is_some_and(|items| items.is_empty())
    }
}

/// Version 3; migrates from 2 by renaming a field, refuses 1, fails on a bad value.
struct Agent;

impl DocDefinition for Agent {
    const KIND: &'static str = "delta.agent";
    const VERSION: u32 = 3;
    type Scope = ConversationScope<Rewindable, ForkAsOf>;
    type Seed = ();

    fn initial((): &()) -> Json {
        json!({ "model": null })
    }

    fn migrate(value: Json, from: u32) -> Result<Json, MigrateError> {
        match from {
            2 => match value.get("selection") {
                Some(Json::String(model)) => Ok(json!({ "model": model.as_str() })),
                Some(Json::Null) => Ok(Json::from(1)),
                _ => Err(MigrateError::failed("selection is missing")),
            },
            from => Err(MigrateError::Unsupported { from }),
        }
    }

    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        true
    }
}

struct NestedCall;

impl DocDefinition for NestedCall {
    const KIND: &'static str = "delta.tool.nested-call";
    const VERSION: u32 = 1;
    type Scope = TaskScope;
    type Seed = Option<String>;
    const FAMILY: bool = true;

    fn initial(seed: &Option<String>) -> Json {
        match seed {
            Some(name) => json!({ "name": name.as_str() }),
            None => Json::Null,
        }
    }

    fn checkpoint_when(_: &Json, _: &[Op], info: CheckpointInfo) -> bool {
        info.deltas_since_base >= 31
    }
}

struct Settings;

impl DocDefinition for Settings {
    const KIND: &'static str = "app.settings";
    const VERSION: u32 = 1;
    type Scope = SessionScope;
    type Seed = ();

    fn initial((): &()) -> Json {
        json!({})
    }

    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        false
    }
}

/// Compiles only for rewindable conversation documents.
fn rewindable<D: ConversationDoc<History = Rewindable>>() -> &'static str {
    D::KIND
}

fn record(semantics: Semantics, key: Option<&str>, scope: DocumentScope) -> DocumentRecord {
    DocumentRecord {
        id: DocumentId(5),
        kind: semantics.kind.to_owned(),
        key: key.map(str::to_owned),
        scope,
        policy: semantics.policy,
        created_at: Seq(1),
        retired_at: None,
    }
}

#[test]
fn semantics_come_from_the_types() {
    assert_eq!(
        semantics::<Inbox>(),
        Semantics {
            kind: "delta.inbox",
            version: 1,
            scope: DocumentScopeKind::Conversation,
            policy: Some(ConversationPolicy {
                history: History::Latest,
                fork: Fork::Initial,
            }),
            family: false,
        }
    );
    assert_eq!(
        semantics::<Agent>().policy,
        Some(ConversationPolicy {
            history: History::Rewindable,
            fork: Fork::AsOf,
        })
    );
    let nested = semantics::<NestedCall>();
    assert_eq!(
        (nested.scope, nested.policy, nested.family),
        (DocumentScopeKind::Task, None, true)
    );
    assert_eq!(semantics::<Settings>().scope, DocumentScopeKind::Session);
    assert_eq!(rewindable::<Agent>(), "delta.agent");
}

#[test]
fn addresses_carry_scope_and_key() {
    let inbox = DocAddress::<Inbox>::new(ConversationId(7));
    assert_eq!(
        inbox.to_address(),
        DocumentAddress {
            kind: "delta.inbox".into(),
            scope: DocumentScope::Conversation {
                conversation_id: ConversationId(7),
            },
            key: None,
        }
    );
    assert_eq!(inbox.owner(), ConversationId(7));
    assert_eq!(inbox.clone(), inbox);

    let member = DocAddress::<NestedCall>::member(TaskId(4), "c7");
    assert_eq!(member.key(), Some("c7"));
    assert_eq!(member.scope(), DocumentScope::Task { task_id: TaskId(4) });
    assert_eq!(member.to_address().key.as_deref(), Some("c7"));
    assert_eq!(DocAddress::<Settings>::new(()).scope(), DocumentScope::Session);
    assert!(format!("{member:?}").contains("delta.tool.nested-call"));
}

#[test]
fn records_must_match_the_definition() {
    let conversation = DocumentScope::Conversation {
        conversation_id: ConversationId(1),
    };
    check_record::<Inbox>(&record(semantics::<Inbox>(), None, conversation)).unwrap();

    let mut rewindable = record(semantics::<Inbox>(), None, conversation);
    rewindable.policy = Some(ConversationPolicy {
        history: History::Rewindable,
        fork: Fork::Initial,
    });
    let error = check_record::<Inbox>(&rewindable).unwrap_err();
    assert_eq!(
        error.to_string(),
        "Document 5 (delta.inbox) does not match the supplied definition semantics"
    );

    let keyed = record(semantics::<Inbox>(), Some("k"), conversation);
    assert!(check_record::<Inbox>(&keyed).is_err());
    let task_scope = DocumentScope::Task { task_id: TaskId(4) };
    check_record::<NestedCall>(&record(semantics::<NestedCall>(), Some(""), task_scope)).unwrap();
    assert!(check_record::<NestedCall>(&record(semantics::<NestedCall>(), None, task_scope)).is_err());
    assert!(check_record::<Settings>(&record(semantics::<Inbox>(), None, conversation)).is_err());
}

#[test]
fn initial_values_must_be_objects() {
    assert_eq!(initial_value::<Inbox>(&()).unwrap(), json!({ "items": [] }));
    assert_eq!(
        initial_value::<NestedCall>(&Some("read".into())).unwrap(),
        json!({ "name": "read" })
    );
    let error = initial_value::<NestedCall>(&None).unwrap_err();
    assert_eq!(
        error.to_string(),
        "Document delta.tool.nested-call initial value is not an object"
    );
}

#[test]
fn upgrades_follow_the_version_table() {
    let id = DocumentId(5);
    let same = upgrade::<Agent>(id, 3, json!({ "model": "a" })).unwrap();
    assert_eq!(
        same,
        Upgraded {
            value: json!({ "model": "a" }),
            migrated: false
        }
    );
    let migrated = upgrade::<Agent>(id, 2, json!({ "selection": "b" })).unwrap();
    assert_eq!(
        migrated,
        Upgraded {
            value: json!({ "model": "b" }),
            migrated: true
        }
    );

    let newer = upgrade::<Agent>(id, 4, json!({})).unwrap_err();
    assert_eq!(newer.to_string(), "Document 5 (delta.agent) has newer version 4 than 3");
    let unsupported = upgrade::<Agent>(id, 1, json!({})).unwrap_err();
    assert_eq!(
        unsupported.to_string(),
        "Document 5 (delta.agent) requires migration from version 1"
    );
    let failed = upgrade::<Agent>(id, 2, json!({})).unwrap_err();
    assert_eq!(
        failed.to_string(),
        "Document delta.agent migration from version 2 failed"
    );
    assert!(std::error::Error::source(&failed).is_some());
    let not_object = upgrade::<Agent>(id, 2, json!({ "selection": null })).unwrap_err();
    assert!(matches!(
        not_object,
        DocumentError::NotAnObject {
            origin: ValueSource::Migrated,
            ..
        }
    ));
    assert!(matches!(
        upgrade::<Inbox>(id, 0, json!({})),
        Err(DocumentError::MigrationRequired { .. })
    ));
}

#[test]
fn checkpoint_predicates_see_the_tail_length() {
    let value = json!({ "items": [] });
    assert!(Inbox::checkpoint_when(&value, &[], CheckpointInfo::default()));
    assert!(!NestedCall::checkpoint_when(
        &value,
        &[],
        CheckpointInfo { deltas_since_base: 30 }
    ));
    assert!(NestedCall::checkpoint_when(
        &value,
        &[],
        CheckpointInfo { deltas_since_base: 31 }
    ));
}
