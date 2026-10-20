//! Table reads and writes (upstream `session-tables.test.ts`).

use delta_chord::json;
use delta_harness::records::{
    ConversationOwnership, EntryDraft, EntryKind, EntryQuery, Head, SubmissionCreate, SubmissionSettlement,
    SubmissionState, SubmissionType, TaskOutcome, TaskState, TaskStatus, TypedEntryDraft,
};
use delta_harness::session::{TaskOptions, TxError};
use serde::{Deserialize, Serialize};

use super::*;

cases!(
    reads_only_before_the_first_table_write,
    documents_stay_available_after_table_writes,
    creates_records_with_minted_ids,
    closure_errors_roll_everything_back,
    validates_owners_atomically,
    rejects_invalid_child_tasks,
    places_and_settles_submissions,
    typed_entries_round_trip,
    scans_pass_limits_and_cursors,
    task_documents_follow_their_task,
    unknown_conversations_reject,
);

async fn reads_only_before_the_first_table_write(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    let error = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                assert!(tx.conversation(conversation).await?.is_some());
                assert!(tx.latest_head_marker(conversation).await?.is_none());
                tx.append_entry(conversation, EntryDraft::new("a")).await?;
                tx.conversation(conversation).await
            })
        })
        .await
        .unwrap_err();
    let error = tx_error(error);
    assert!(matches!(error, TxError::ReadAfterWrite { method: "conversation" }));
    assert_eq!(
        error.to_string(),
        "Tx::conversation() cannot read tables after the first table write"
    );
    // Settlement and placement count as table writes too.
    let error = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.place_submission(
                    delta_harness::records::SubmissionId(999),
                    delta_harness::records::EntryId(1),
                );
                tx.task(TaskId(5)).await
            })
        })
        .await
        .unwrap_err();
    assert!(matches!(tx_error(error), TxError::ReadAfterWrite { method: "task" }));
}

async fn documents_stay_available_after_table_writes(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    let (task, value) = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let task = tx
                    .create_task_record(task_create(), TaskOptions::conversation(conversation))
                    .await?;
                let mut notes = tx.doc(DocAddress::<Notes>::new(conversation)).await?;
                notes.draft().field("items").unwrap().into_array().unwrap().push("one");
                tx.append_entry(conversation, EntryDraft::new("between")).await?;
                // Reacquiring returns the same open change.
                let mut notes = tx.doc(DocAddress::<Notes>::new(conversation)).await?;
                notes.draft().field("items").unwrap().into_array().unwrap().push("two");
                let mut scratch = tx.doc(DocAddress::<Scratch>::new(task)).await?;
                scratch.set(json!({ "steps": [1] }));
                Ok((
                    task,
                    tx.doc(DocAddress::<Notes>::new(conversation)).await?.value().clone(),
                ))
            })
        })
        .await
        .unwrap();
    assert_eq!(value, json!({ "items": ["one", "two"] }));
    let notes = env
        .session
        .snapshot(DocAddress::<Notes>::new(conversation), &ctx())
        .await
        .unwrap();
    assert_eq!(notes, Some(json!({ "items": ["one", "two"] })));
    let scratch = env
        .session
        .snapshot(DocAddress::<Scratch>::new(task), &ctx())
        .await
        .unwrap();
    assert_eq!(scratch, Some(json!({ "steps": [1] })));
}

async fn creates_records_with_minted_ids(backend: Backend) {
    let env = Env::open(backend).await;
    let (conversation, entry, head, task, submission) = env
        .session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                let conversation = tx.create_conversation(ConversationOwnership::Ownerless).await?;
                let entry = tx.append_entry(conversation.id, EntryDraft::new("first")).await?;
                let mut draft = EntryDraft::new("marker");
                draft.head = Some(Head::SelfRef);
                let head = tx.append_entry(conversation.id, draft).await?;
                let task = tx
                    .create_task_record(task_create(), TaskOptions::conversation(conversation.id))
                    .await?;
                let submission = tx
                    .create_submission(SubmissionCreate {
                        conversation_id: conversation.id,
                        request_id: Some("r1".to_owned()),
                        kind: SubmissionType::Input,
                        state: SubmissionState::Queued,
                    })
                    .await?;
                Ok((conversation, entry, head, task, submission))
            })
        })
        .await
        .unwrap();
    assert!(conversation.id.0 < entry.id.0 && entry.id < head.id);
    assert_eq!(head.head, Some(head.id));
    assert_eq!(entry.by_task_id, None);
    let stored = env.storage.entry(entry.id, &ctx()).await.unwrap().unwrap();
    assert_eq!(stored.entry, entry);
    let task = env.storage.task(task, &ctx()).await.unwrap().unwrap();
    assert_eq!(
        task.state,
        TaskState::Pending {
            checkpoint: json!({ "phase": "start" })
        }
    );
    assert_eq!(task.conversation_id, conversation.id);
    assert_eq!(task.started_at, None);
    let by_request = env
        .storage
        .submission_by_request(conversation.id, "r1", &ctx())
        .await
        .unwrap();
    assert_eq!(by_request, Some(submission));
}

async fn closure_errors_roll_everything_back(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    let error = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.append_entry(conversation, EntryDraft::new("lost")).await?;
                let mut notes = tx.doc(DocAddress::<Notes>::new(conversation)).await?;
                notes.set(json!({ "items": ["lost"] }));
                let mut counter = tx.doc(DocAddress::<Counter>::new(())).await?;
                counter.set(json!({ "count": 9 }));
                Err::<(), _>(TxError::callback("the closure's own error"))
            })
        })
        .await
        .unwrap_err();
    assert!(matches!(tx_error(error), TxError::Callback(_)));
    let page = env
        .storage
        .scan_entries(EntryQuery::new(conversation), 10, None, &ctx())
        .await
        .unwrap();
    assert!(page.items.is_empty());
    assert_eq!(
        env.session
            .snapshot(DocAddress::<Notes>::new(conversation), &ctx())
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        env.session
            .snapshot(DocAddress::<Counter>::new(()), &ctx())
            .await
            .unwrap(),
        None
    );
    // The Session stays usable.
    env.append(conversation, "kept").await;
    assert!(env.session.failure().is_none());
}

async fn validates_owners_atomically(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    let outcome = TaskOutcome::Completed { result: Json::Null };
    env.write_task(task_record(
        900,
        conversation,
        TaskState::Terminal {
            outcome: outcome.clone(),
        },
    ))
    .await;
    env.write_task(task_record(901, conversation, TaskState::Completing { outcome }))
        .await;
    let mut marked = task_record(902, conversation, TaskState::Running { checkpoint: Json::Null });
    marked.abort_requested = true;
    env.write_task(marked).await;
    env.write_task(task_record(
        903,
        conversation,
        TaskState::Running { checkpoint: Json::Null },
    ))
    .await;

    for (owner, status) in [(900, "terminal"), (901, "completing"), (902, "abort-marked")] {
        let error = env
            .session
            .commit(&ctx(), move |tx| {
                Box::pin(async move {
                    tx.create_conversation(ConversationOwnership::Task(TaskId(owner)))
                        .await?;
                    Ok(())
                })
            })
            .await
            .unwrap_err();
        match tx_error(error) {
            TxError::OwnerUnavailable {
                what,
                task,
                status: found,
            } => {
                assert_eq!((what, task, found), ("Conversation owner task", TaskId(owner), status));
            }
            other => panic!("unexpected {other:?}"),
        }
        let error = env
            .session
            .commit(&ctx(), move |tx| {
                Box::pin(async move {
                    tx.create_task_record(task_create(), TaskOptions::owned_by(TaskId(owner)))
                        .await
                })
            })
            .await
            .unwrap_err();
        assert!(matches!(
            tx_error(error),
            TxError::OwnerUnavailable { what: "Task owner", .. }
        ));
    }
    let error = env
        .session
        .commit(&ctx(), |tx| {
            Box::pin(async move { tx.create_conversation(ConversationOwnership::Task(TaskId(12345))).await })
        })
        .await
        .unwrap_err();
    assert!(matches!(tx_error(error), TxError::UnknownTask(TaskId(12345))));

    // A live owner: the persisted owner edge comes from the task's conversation.
    let (child, task) = env
        .session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                let child = tx.create_conversation(ConversationOwnership::Task(TaskId(903))).await?;
                let task = tx
                    .create_task_record(task_create(), TaskOptions::owned_by(TaskId(903)))
                    .await?;
                Ok((child, task))
            })
        })
        .await
        .unwrap();
    let owner = child.owner.unwrap();
    assert_eq!((owner.conversation_id, owner.task_id), (conversation, TaskId(903)));
    let task = env.storage.task(task, &ctx()).await.unwrap().unwrap();
    assert_eq!((task.owner, task.conversation_id), (Some(TaskId(903)), conversation));
}

async fn rejects_invalid_child_tasks(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    let other = env.conversation().await;
    let owner = env.task(conversation).await;
    let background = TaskOptions {
        background: true,
        ..TaskOptions::owned_by(owner)
    };
    let elsewhere = TaskOptions {
        conversation_id: Some(other),
        ..TaskOptions::owned_by(owner)
    };
    for options in [background, elsewhere, TaskOptions::default()] {
        let error = env
            .session
            .commit(&ctx(), move |tx| {
                Box::pin(async move { tx.create_task_record(task_create(), options).await })
            })
            .await
            .unwrap_err();
        assert!(matches!(tx_error(error), TxError::InvalidTask(_)));
    }
}

async fn places_and_settles_submissions(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    let create = move |kind| SubmissionCreate {
        conversation_id: conversation,
        request_id: None,
        kind,
        state: SubmissionState::Queued,
    };
    let (input, write) = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let input = tx.create_submission(create(SubmissionType::Input)).await?;
                let write = tx.create_submission(create(SubmissionType::Write)).await?;
                Ok((input.id, write.id))
            })
        })
        .await
        .unwrap();
    // Settling a queued input as done is illegal.
    let error = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.settle_submission(
                    input,
                    SubmissionSettlement::Done {
                        answer: delta_harness::records::EntryId(1),
                    },
                );
                Ok(())
            })
        })
        .await
        .unwrap_err();
    assert!(matches!(tx_error(error), TxError::Submission(_)));
    // Placement and settlement in one commit resolve against the latest candidate.
    let (entry, answer) = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let entry = tx.append_entry(conversation, EntryDraft::new("input")).await?;
                let answer = tx.append_entry(conversation, EntryDraft::new("answer")).await?;
                tx.place_submission(input, entry.id);
                tx.settle_submission(input, SubmissionSettlement::Done { answer: answer.id });
                tx.place_submission(write, entry.id);
                Ok((entry.id, answer.id))
            })
        })
        .await
        .unwrap();
    let input_record = env.storage.submission(input, &ctx()).await.unwrap().unwrap();
    assert_eq!(
        input_record.state,
        SubmissionState::Done {
            entry,
            answer: Some(answer)
        }
    );
    let write_record = env.storage.submission(write, &ctx()).await.unwrap().unwrap();
    assert_eq!(write_record.state, SubmissionState::Done { entry, answer: None });
    // A settled record stays.
    env.session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let unanswered = SubmissionSettlement::Unanswered {
                    reason: "late".to_owned(),
                    detail: None,
                };
                tx.settle_submission(input, unanswered);
                Ok(())
            })
        })
        .await
        .unwrap();
    let again = env.storage.submission(input, &ctx()).await.unwrap().unwrap();
    assert_eq!(again, input_record);
    let error = env
        .session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                tx.place_submission(
                    delta_harness::records::SubmissionId(4242),
                    delta_harness::records::EntryId(1),
                );
                Ok(())
            })
        })
        .await
        .unwrap_err();
    assert!(matches!(tx_error(error), TxError::UnknownSubmission(_)));
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Note {
    text: String,
}

const NOTE: EntryKind<Note> = EntryKind::new("test.note");

async fn typed_entries_round_trip(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    let typed = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.append_typed_entry(
                    &NOTE,
                    conversation,
                    TypedEntryDraft::new(Note { text: "hi".to_owned() }),
                )
                .await
            })
        })
        .await
        .unwrap();
    assert_eq!(typed.record.data, Some(json!({ "text": "hi" })));
    let other = env.append(conversation, "test.other").await;
    let (read, missing) = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let read = tx.typed_entry(&NOTE, typed.record.id).await?;
                let missing = tx.typed_entry(&NOTE, other.id).await?;
                Ok((read, missing))
            })
        })
        .await
        .unwrap();
    assert_eq!(read.unwrap().data, Note { text: "hi".to_owned() });
    assert!(missing.is_none());
}

async fn scans_pass_limits_and_cursors(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    for kind in ["a", "b", "c"] {
        env.append(conversation, kind).await;
    }
    let kinds = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let first = tx.scan_entries(EntryQuery::new(conversation), 2, None).await?;
                let second = tx.scan_entries(EntryQuery::new(conversation), 2, first.next).await?;
                assert!(second.next.is_none());
                Ok(first
                    .items
                    .into_iter()
                    .chain(second.items)
                    .map(|entry| entry.kind)
                    .collect::<Vec<_>>())
            })
        })
        .await
        .unwrap();
    assert_eq!(kinds, ["c", "b", "a"]);
    // A zero limit is an invalid read: it fails the call only.
    let error = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move { tx.scan_entries(EntryQuery::new(conversation), 0, None).await })
        })
        .await
        .unwrap_err();
    assert!(matches!(error, SessionError::Storage(_)), "{error:?}");
    assert!(env.session.failure().is_none());
}

async fn task_documents_follow_their_task(backend: Backend) {
    let env = Env::open(backend).await;
    let conversation = env.conversation().await;
    let outcome = TaskOutcome::Completed { result: Json::Null };
    env.write_task(task_record(700, conversation, TaskState::Terminal { outcome }))
        .await;
    let error = env
        .session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Scratch>::new(TaskId(700))).await?;
                Ok(())
            })
        })
        .await
        .unwrap_err();
    assert!(matches!(tx_error(error), TxError::TerminalTask(TaskId(700))));
    let error = env
        .session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Scratch>::new(TaskId(701))).await?;
                Ok(())
            })
        })
        .await
        .unwrap_err();
    assert!(matches!(tx_error(error), TxError::UnknownTask(TaskId(701))));
    let task = env.task(conversation).await;
    let status = env.storage.task(task, &ctx()).await.unwrap().unwrap().state.status();
    assert_eq!(status, TaskStatus::Pending);
    env.session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Scratch>::new(task)).await?;
                Ok(())
            })
        })
        .await
        .unwrap();
    assert!(
        env.session
            .snapshot(DocAddress::<Scratch>::new(task), &ctx())
            .await
            .unwrap()
            .is_some()
    );
}

async fn unknown_conversations_reject(backend: Backend) {
    let env = Env::open(backend).await;
    let missing = ConversationId(777);
    let error = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move { tx.append_entry(missing, EntryDraft::new("x")).await })
        })
        .await
        .unwrap_err();
    assert!(matches!(
        tx_error(error),
        TxError::UnknownConversation(ConversationId(777))
    ));
    let error = env
        .session
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Notes>::new(missing)).await?;
                Ok(())
            })
        })
        .await
        .unwrap_err();
    assert!(matches!(tx_error(error), TxError::UnknownConversation(_)));
    // The root conversation exists once written.
    let root = env.root().await;
    env.append(root, "root entry").await;
}

use delta_harness::documents::DocAddress;
