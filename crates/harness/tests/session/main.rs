//! The Session: commits across records and documents, document planning, cache and
//! retention, observation, forks, snapshots, failures and close.
//!
//! Ported from upstream `test/session-*.test.ts` where the behaviour applies, adapted to
//! delta's decided differences (chord's no-op and merge rules, one document edited at a
//! time). Most cases run against the memory backend and SQLite in a temp file.

#![allow(clippy::unwrap_used)] // Case functions are not `#[test]` functions.

#[path = "../support/mod.rs"]
mod support;

/// Runs each case against memory and SQLite in a file.
macro_rules! cases {
    ($($case:ident),* $(,)?) => {
        mod memory {
            $(#[tokio::test] async fn $case() { super::$case(super::Backend::Memory).await; })*
        }
        mod sqlite {
            $(#[tokio::test] async fn $case() { super::$case(super::Backend::SqliteFile).await; })*
        }
    };
}

mod documents;
mod failures;
mod forks;
mod incarnations;
mod observation;
mod retention;
mod tables;

use std::sync::Arc;

use delta_chord::context::Context;
use delta_chord::delta::Op;
use delta_chord::json::Json;
use delta_harness::documents::{
    CheckpointInfo, ConversationScope, DocDefinition, ForkAsOf, ForkCurrent, ForkInitial, Latest, MigrateError,
    Rewindable, SessionScope, TaskScope,
};
use delta_harness::records::{
    ConversationId, ConversationOwnership, EntryDraft, EntryRecord, ROOT_CONVERSATION_ID, StorageWrite, TaskId,
    TaskRecord, TaskState,
};
use delta_harness::session::{Session, SessionError, SessionOptions, TaskCreate, TaskOptions, TxError};
use delta_harness::storage::Storage;

pub(crate) use support::{Backend, TempDir};

pub(crate) fn ctx() -> Context {
    Context::background()
}

/// An open Session with its storage, which outlives it.
pub(crate) struct Env {
    pub(crate) session: Session,
    pub(crate) storage: Arc<dyn Storage>,
    pub(crate) _dir: Option<TempDir>,
}

impl Env {
    pub(crate) async fn open(backend: Backend) -> Self {
        Self::open_with(backend, SessionOptions::default()).await
    }

    pub(crate) async fn open_with(backend: Backend, options: SessionOptions) -> Self {
        let fixture = backend.open().await;
        let session = Session::open(fixture.storage.clone(), options);
        Self {
            session,
            storage: fixture.storage,
            _dir: fixture.dir,
        }
    }

    /// A new ownerless conversation.
    pub(crate) async fn conversation(&self) -> ConversationId {
        self.session
            .commit(&ctx(), |tx| {
                Box::pin(async move { Ok(tx.create_conversation(ConversationOwnership::Ownerless).await?.id) })
            })
            .await
            .unwrap()
    }

    /// Appends an entry of `kind` to `conversation`.
    pub(crate) async fn append(&self, conversation: ConversationId, kind: &str) -> EntryRecord {
        let draft = EntryDraft::new(kind);
        self.session
            .commit(&ctx(), move |tx| {
                Box::pin(async move { tx.append_entry(conversation, draft).await })
            })
            .await
            .unwrap()
    }

    /// A pending task of `conversation`, created through the Session.
    pub(crate) async fn task(&self, conversation: ConversationId) -> TaskId {
        self.session
            .commit(&ctx(), move |tx| {
                Box::pin(async move {
                    tx.create_task_record(task_create(), TaskOptions::conversation(conversation))
                        .await
                })
            })
            .await
            .unwrap()
    }

    /// Writes `record` straight to storage (states the Session API does not produce).
    pub(crate) async fn write_task(&self, record: TaskRecord) {
        self.storage
            .commit(vec![StorageWrite::Task(record)], &ctx())
            .await
            .unwrap();
    }

    /// Writes the root conversation straight to storage.
    pub(crate) async fn root(&self) -> ConversationId {
        let record = delta_harness::records::ConversationRecord::new(ROOT_CONVERSATION_ID);
        self.storage
            .commit(vec![StorageWrite::Conversation(record)], &ctx())
            .await
            .unwrap();
        ROOT_CONVERSATION_ID
    }
}

pub(crate) fn task_create() -> TaskCreate {
    TaskCreate {
        kind: "test.task".to_owned(),
        version: 1,
        input: delta_chord::json!({ "n": 1 }),
        checkpoint: delta_chord::json!({ "phase": "start" }),
    }
}

/// A task record of `conversation` in `state`.
pub(crate) fn task_record(id: u64, conversation: ConversationId, state: TaskState) -> TaskRecord {
    TaskRecord::new(TaskId(id), conversation, "test.task", 1, Json::Null, state)
}

/// The `TxError` inside a refused call.
pub(crate) fn tx_error(error: SessionError) -> TxError {
    match error {
        SessionError::Tx(error) => error,
        other => panic!("expected a rejected transaction, got {other:?}"),
    }
}

// ─── Definitions ────────────────────────────────────────────────────────────

/// A conversation document: latest, not copied by forks, never checkpoints.
pub(crate) struct Notes;
impl DocDefinition for Notes {
    const KIND: &'static str = "test.notes";
    const VERSION: u32 = 1;
    type Scope = ConversationScope<Latest, ForkInitial>;
    type Seed = ();
    fn initial((): &()) -> Json {
        delta_chord::json!({ "items": [] })
    }
    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        false
    }
}

/// [`Notes`]' kind with other semantics.
pub(crate) struct NotesRewindable;
impl DocDefinition for NotesRewindable {
    const KIND: &'static str = "test.notes";
    const VERSION: u32 = 1;
    type Scope = ConversationScope<Rewindable, ForkInitial>;
    type Seed = ();
    fn initial((): &()) -> Json {
        delta_chord::json!({ "items": [] })
    }
    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        false
    }
}

/// A rewindable conversation document copied as of the fork entry.
pub(crate) struct Agent;
impl DocDefinition for Agent {
    const KIND: &'static str = "test.agent";
    const VERSION: u32 = 1;
    type Scope = ConversationScope<Rewindable, ForkAsOf>;
    type Seed = ();
    fn initial((): &()) -> Json {
        delta_chord::json!({ "turn": 0 })
    }
    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        false
    }
}

/// A latest conversation document copied as the parent has it when the fork commits.
pub(crate) struct Current;
impl DocDefinition for Current {
    const KIND: &'static str = "test.current";
    const VERSION: u32 = 1;
    type Scope = ConversationScope<Latest, ForkCurrent>;
    type Seed = ();
    fn initial((): &()) -> Json {
        delta_chord::json!({ "value": 0 })
    }
    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        false
    }
}

/// A session document whose every third change is a base.
pub(crate) struct Counter;
impl DocDefinition for Counter {
    const KIND: &'static str = "test.counter";
    const VERSION: u32 = 1;
    type Scope = SessionScope;
    type Seed = ();
    fn initial((): &()) -> Json {
        delta_chord::json!({ "count": 0 })
    }
    fn checkpoint_when(_: &Json, _: &[Op], info: CheckpointInfo) -> bool {
        info.deltas_since_base >= 2
    }
}

/// [`Counter`] at version 2: migration adds `"migrated": true`.
pub(crate) struct CounterV2;
impl DocDefinition for CounterV2 {
    const KIND: &'static str = "test.counter";
    const VERSION: u32 = 2;
    type Scope = SessionScope;
    type Seed = ();
    fn initial((): &()) -> Json {
        delta_chord::json!({ "count": 0, "migrated": false })
    }
    fn migrate(value: Json, from: u32) -> Result<Json, MigrateError> {
        assert_eq!(from, 1);
        let mut object = value.as_object().cloned().ok_or(MigrateError::Unsupported { from })?;
        object.make_mut().insert("migrated".into(), Json::Bool(true));
        Ok(Json::Object(object))
    }
    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        false
    }
}

/// [`Counter`] at version 3 without a migration.
pub(crate) struct CounterV3;
impl DocDefinition for CounterV3 {
    const KIND: &'static str = "test.counter";
    const VERSION: u32 = 3;
    type Scope = SessionScope;
    type Seed = ();
    fn initial((): &()) -> Json {
        delta_chord::json!({})
    }
    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        false
    }
}

/// [`Counter`] at version 2 whose migration yields a non-object.
pub(crate) struct CounterBadMigration;
impl DocDefinition for CounterBadMigration {
    const KIND: &'static str = "test.counter";
    const VERSION: u32 = 2;
    type Scope = SessionScope;
    type Seed = ();
    fn initial((): &()) -> Json {
        delta_chord::json!({})
    }
    fn migrate(_: Json, _: u32) -> Result<Json, MigrateError> {
        Ok(Json::from(7))
    }
    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        false
    }
}

/// A session document whose initial value is not an object.
pub(crate) struct BadInitial;
impl DocDefinition for BadInitial {
    const KIND: &'static str = "test.bad-initial";
    const VERSION: u32 = 1;
    type Scope = SessionScope;
    type Seed = ();
    fn initial((): &()) -> Json {
        Json::from("not an object")
    }
    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        false
    }
}

/// A session document whose checkpoint predicate panics on `"panic": true`.
pub(crate) struct Panicky;
impl DocDefinition for Panicky {
    const KIND: &'static str = "test.panicky";
    const VERSION: u32 = 1;
    type Scope = SessionScope;
    type Seed = ();
    fn initial((): &()) -> Json {
        delta_chord::json!({ "panic": false })
    }
    fn checkpoint_when(value: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        assert!(
            value.get("panic") != Some(&Json::Bool(true)),
            "checkpoint predicate panicked"
        );
        false
    }
}

/// A task document.
pub(crate) struct Scratch;
impl DocDefinition for Scratch {
    const KIND: &'static str = "test.scratch";
    const VERSION: u32 = 1;
    type Scope = TaskScope;
    type Seed = ();
    fn initial((): &()) -> Json {
        delta_chord::json!({ "steps": [] })
    }
    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        false
    }
}

/// A conversation family seeded with a name.
pub(crate) struct Member;
impl DocDefinition for Member {
    const KIND: &'static str = "test.member";
    const VERSION: u32 = 1;
    type Scope = ConversationScope<Latest, ForkCurrent>;
    type Seed = String;
    const FAMILY: bool = true;
    fn initial(seed: &String) -> Json {
        delta_chord::json!({ "name": seed.as_str() })
    }
    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        false
    }
}
