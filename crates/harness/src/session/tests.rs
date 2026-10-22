//! The crate-internal surface the scheduler and the layers above build on: scoped commits,
//! whole-record task replacement, terminal settlement, the root bootstrap, the
//! conversation hook and failing the Session from a component.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use delta_chord::context::Context;
use delta_chord::delta::Op;
use delta_chord::json;
use delta_chord::json::Json;
use futures::future::BoxFuture;

use super::{
    ConversationHook, Session, SessionEnd, SessionError, SessionOptions, TaskCreate, TaskOptions, TransactionScope, Tx,
    TxError,
};
use crate::documents::{CheckpointInfo, ConversationScope, DocAddress, DocDefinition, ForkInitial, Latest, TaskScope};
use crate::records::{
    ConversationId, ConversationOwnership, ConversationRecord, DocumentPoint, EntryDraft, ROOT_CONVERSATION_ID, TaskId,
    TaskOutcome, TaskState, TaskStatus,
};
use crate::storage::{MemoryStorage, Storage};

struct Scratch;
impl DocDefinition for Scratch {
    const KIND: &'static str = "test.scratch";
    const VERSION: u32 = 1;
    type Scope = TaskScope;
    type Seed = ();
    fn initial((): &()) -> Json {
        json!({ "steps": [] })
    }
    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        false
    }
}

struct Inbox;
impl DocDefinition for Inbox {
    const KIND: &'static str = "test.inbox";
    const VERSION: u32 = 1;
    type Scope = ConversationScope<Latest, ForkInitial>;
    type Seed = ();
    fn initial((): &()) -> Json {
        json!({ "items": [] })
    }
    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        false
    }
}

/// Stages an inbox in every new conversation, as the agent layer stages its documents.
struct StageInbox;
impl ConversationHook for StageInbox {
    fn conversation_created<'a>(
        &'a self,
        tx: &'a mut Tx<'_>,
        record: &'a ConversationRecord,
    ) -> BoxFuture<'a, Result<(), TxError>> {
        let id = record.id;
        Box::pin(async move {
            tx.doc(DocAddress::<Inbox>::new(id)).await?;
            Ok(())
        })
    }
}

fn ctx() -> Context {
    Context::background()
}

fn open() -> (Session, Arc<MemoryStorage>) {
    let storage = Arc::new(MemoryStorage::new());
    let options = SessionOptions {
        now: Arc::new(|| 1_000),
        conversation_hook: Some(Arc::new(StageInbox)),
        ..SessionOptions::default()
    };
    (Session::open(storage.clone(), options), storage)
}

fn create() -> TaskCreate {
    TaskCreate {
        kind: "test.task".to_owned(),
        version: 1,
        input: Json::Null,
        checkpoint: json!({ "phase": "start" }),
    }
}

async fn bootstrap(session: &Session) -> (ConversationId, TaskId) {
    session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                let root = tx.create_root_conversation().await?;
                assert_eq!(tx.staged_conversations().count(), 1);
                let task = tx
                    .create_task_record(create(), TaskOptions::conversation(root.id))
                    .await?;
                assert_eq!(tx.staged_tasks().count(), 1);
                Ok((root.id, task))
            })
        })
        .await
        .unwrap()
}

#[tokio::test]
async fn the_root_bootstrap_runs_the_conversation_hook() {
    let (session, _) = open();
    let (root, _) = bootstrap(&session).await;
    assert_eq!(root, ROOT_CONVERSATION_ID);
    let inbox = session.snapshot(DocAddress::<Inbox>::new(root), &ctx()).await.unwrap();
    assert_eq!(inbox, Some(json!({ "items": [] })));
}

#[tokio::test]
async fn scoped_commits_stamp_entries_and_default_the_conversation() {
    let (session, storage) = open();
    let (root, task) = bootstrap(&session).await;
    let scope = TransactionScope {
        conversation_id: Some(root),
        task_id: Some(task),
    };
    let (entry, child) = session
        .commit_scoped(&ctx(), scope, move |tx| {
            Box::pin(async move {
                let entry = tx.append_entry(root, EntryDraft::new("by task")).await?;
                let child = tx.create_task_record(create(), TaskOptions::default()).await?;
                Ok((entry, child))
            })
        })
        .await
        .unwrap();
    assert_eq!(entry.by_task_id, Some(task));
    let child = storage.task(child, &ctx()).await.unwrap().unwrap();
    assert_eq!(child.conversation_id, root);
}

#[tokio::test]
async fn set_task_replaces_whole_records_and_stamps_times() {
    let (session, storage) = open();
    let (root, task) = bootstrap(&session).await;
    let running = session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let mut record = tx.task(task).await?.unwrap();
                record.state = TaskState::Running {
                    checkpoint: json!({ "phase": "work" }),
                };
                tx.set_task(record)?;
                tx.doc(DocAddress::<Scratch>::new(task)).await?;
                Ok(tx.staged_tasks().next().cloned().unwrap())
            })
        })
        .await
        .unwrap();
    assert_eq!((running.started_at, running.ended_at), (Some(1_000), None));
    // A second terminal candidate and a conversation change are refused.
    let error = session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let mut record = tx.task(task).await?.unwrap();
                record.state = TaskState::Terminal {
                    outcome: TaskOutcome::Completed { result: Json::Null },
                };
                tx.set_task(record.clone())?;
                tx.set_task(record)
            })
        })
        .await
        .unwrap_err();
    assert!(matches!(error, SessionError::Tx(TxError::InvalidTask(_))));
    let error = session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let mut record = tx.task(task).await?.unwrap();
                record.conversation_id = ConversationId(99);
                tx.set_task(record)
            })
        })
        .await
        .unwrap_err();
    assert!(matches!(error, SessionError::Tx(TxError::TaskConversationChange(_))));
    // Terminal settlement retires the task's documents, those created in the same commit too.
    session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let mut record = tx.task(task).await?.unwrap();
                record.state = TaskState::Terminal {
                    outcome: TaskOutcome::Completed { result: Json::Null },
                };
                tx.set_task(record)?;
                let refused = tx.doc(DocAddress::<Scratch>::new(task)).await.map(|_| ());
                assert!(matches!(refused, Err(TxError::TerminalTask(_))));
                Ok(())
            })
        })
        .await
        .unwrap();
    let terminal = storage.task(task, &ctx()).await.unwrap().unwrap();
    assert_eq!(terminal.state.status(), TaskStatus::Terminal);
    assert_eq!((terminal.started_at, terminal.ended_at), (Some(1_000), Some(1_000)));
    let address = DocAddress::<Scratch>::new(task).to_address();
    assert!(
        storage
            .find_document(&address, DocumentPoint::Current, &ctx())
            .await
            .unwrap()
            .is_none()
    );
    // A replacement of a terminal task is refused at settlement.
    let error = session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let mut record = tx.task(task).await?.unwrap();
                record.abort_requested = true;
                tx.set_task(record)
            })
        })
        .await
        .unwrap_err();
    assert!(matches!(error, SessionError::Tx(TxError::TerminalTask(_))));
    let _ = root;
}

#[tokio::test]
async fn new_task_documents_retire_with_a_task_turning_terminal_in_the_same_commit() {
    let (session, storage) = open();
    let (root, _) = bootstrap(&session).await;
    let task = session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let task = tx.create_task_record(create(), TaskOptions::conversation(root)).await?;
                tx.doc(DocAddress::<Scratch>::new(task)).await?;
                let mut record = tx.staged_tasks().next().cloned().unwrap();
                record.state = TaskState::Terminal {
                    outcome: TaskOutcome::Completed { result: Json::Null },
                };
                tx.set_task(record)?;
                Ok(task)
            })
        })
        .await
        .unwrap();
    let address = DocAddress::<Scratch>::new(task).to_address();
    assert!(
        storage
            .find_document(&address, DocumentPoint::Current, &ctx())
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn a_component_can_fail_the_session() {
    let (session, _) = open();
    bootstrap(&session).await;
    let failed = session.fail(Arc::new(std::io::Error::other("index out of step")));
    assert_eq!(failed.cause.to_string(), "index out of step");
    assert!(matches!(session.closed().await, SessionEnd::Failed(_)));
    let error = session
        .commit(&ctx(), |tx| {
            Box::pin(async move { tx.create_conversation(ConversationOwnership::Ownerless).await })
        })
        .await
        .unwrap_err();
    assert!(matches!(error, SessionError::Failed(_)));
}
