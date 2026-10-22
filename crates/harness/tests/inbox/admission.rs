//! Admission: busy rules, request ids, display-only and idle writes, input entries.

use std::sync::Arc;

use delta_ai::types::Message;
use delta_harness::harness::{HarnessError, SubmissionAbort};
use delta_harness::inbox::{InputEntry, SubmissionDraft, SubmitError, WhenBusy};
use delta_harness::records::{EntryDraft, EntryId, Head, SubmissionState, SubmissionType};
use tokio::sync::Notify;

use crate::Backend;
use crate::support::fixture::{EPOCH, Fixture, chat, ctx, entries, eventually, kinds, live, settled};
use crate::support::model::Reply;

cases!(
    a_busy_conversation_rejects_input_that_asks_to_be_rejected,
    a_request_id_admits_once,
    a_display_only_write_lands_at_once_even_while_busy,
    an_idle_write_is_done_or_stale,
    input_entries_take_their_own_kind,
    abort_withdraws_only_queued_submissions,
);

async fn a_busy_conversation_rejects_input_that_asks_to_be_rejected(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let gate = Arc::new(Notify::new());
    fixture.model.push([Reply::text("first").gate(&gate)]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let first = conversation
        .submit(SubmissionDraft::input("one"), &ctx())
        .await
        .unwrap();
    let rejected = conversation
        .submit(SubmissionDraft::input("two").when_busy(WhenBusy::Reject), &ctx())
        .await;
    assert!(matches!(
        rejected,
        Err(HarnessError::Submit(SubmitError::ConversationBusy(id))) if id == conversation.id()
    ));
    gate.notify_one();
    settled(&first).await;
    assert_eq!(
        kinds(&entries(&conversation).await),
        ["delta.user", "delta.system", "delta.assistant"]
    );
    // Idle, the same input is admitted.
    fixture.model.push([Reply::text("second")]);
    let second = conversation
        .submit(SubmissionDraft::input("two").when_busy(WhenBusy::Reject), &ctx())
        .await
        .unwrap();
    assert!(matches!(settled(&second).await.state, SubmissionState::Done { .. }));
    harness.close(&ctx()).await.unwrap();
}

async fn a_request_id_admits_once(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture.model.push([Reply::text("once")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let draft = SubmissionDraft::input("hi").request_id("r1");
    let first = conversation.submit(draft.clone(), &ctx()).await.unwrap();
    settled(&first).await;
    let again = conversation.submit(draft, &ctx()).await.unwrap();
    assert_eq!(again.id(), first.id());
    assert_eq!(fixture.model.requests().len(), 1);
    let write = SubmissionDraft::write(EntryDraft::new("note")).request_id("r1");
    let mismatch = conversation.submit(write, &ctx()).await;
    assert!(matches!(
        mismatch,
        Err(HarnessError::Submit(SubmitError::RequestIdTypeMismatch {
            existing: SubmissionType::Input,
            ..
        }))
    ));
    harness.close(&ctx()).await.unwrap();
}

async fn a_display_only_write_lands_at_once_even_while_busy(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let gate = Arc::new(Notify::new());
    fixture.model.push([Reply::text("answer").gate(&gate)]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let input = conversation.submit(SubmissionDraft::input("hi"), &ctx()).await.unwrap();
    // The run's request is in flight: its system entry precedes the note.
    eventually(|| async { fixture.model.requests().len() == 1 }).await;
    let note = EntryDraft {
        data: Some(delta_chord::json!({ "text": "for the UI" })),
        ..EntryDraft::new("app.note")
    };
    let write = conversation.submit(SubmissionDraft::write(note), &ctx()).await.unwrap();
    let state = write.status(&ctx()).await.unwrap().state;
    assert!(matches!(state, SubmissionState::Done { answer: None, .. }), "{state:?}");
    assert!(live(&harness, &conversation).await.run.is_some());
    gate.notify_one();
    settled(&input).await;
    assert_eq!(
        kinds(&entries(&conversation).await),
        ["delta.user", "delta.system", "app.note", "delta.assistant"]
    );
    harness.close(&ctx()).await.unwrap();
}

async fn an_idle_write_is_done_or_stale(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let marker = EntryDraft {
        model: Some(vec![Message::user("context restart")]),
        head: Some(Head::SelfRef),
        ..EntryDraft::new("app.marker")
    };
    let placed = conversation
        .submit(SubmissionDraft::write(marker), &ctx())
        .await
        .unwrap();
    let SubmissionState::Done { entry, .. } = placed.status(&ctx()).await.unwrap().state else {
        panic!("done")
    };
    // A head older than the active range would bring cut history back: stale.
    let stale = EntryDraft {
        head: Some(Head::At(EntryId(entry.0 - 1))),
        ..EntryDraft::new("app.marker")
    };
    let refused = conversation
        .submit(SubmissionDraft::write(stale), &ctx())
        .await
        .unwrap();
    let state = refused.status(&ctx()).await.unwrap().state;
    assert!(matches!(&state, SubmissionState::Unanswered { reason, entry: None, .. } if reason == "stale"));
    assert_eq!(kinds(&entries(&conversation).await), ["app.marker"]);
    // Writes never start a run.
    assert!(fixture.model.requests().is_empty());
    harness.close(&ctx()).await.unwrap();
}

async fn input_entries_take_their_own_kind(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture.model.push([Reply::text("noted")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let draft = |kind: &str| SubmissionDraft::Input {
        content: "from the app".into(),
        when_busy: WhenBusy::FollowUp,
        entry: Some(InputEntry {
            kind: kind.to_owned(),
            data: Some(delta_chord::json!({ "source": "test" })),
        }),
        request_id: None,
    };
    for invalid in ["", "delta.user"] {
        let refused = conversation.submit(draft(invalid), &ctx()).await;
        assert!(matches!(
            refused,
            Err(HarnessError::Submit(SubmitError::InvalidEntryKind(_)))
        ));
    }
    let submission = conversation.submit(draft("app.input"), &ctx()).await.unwrap();
    settled(&submission).await;
    let transcript = entries(&conversation).await;
    assert_eq!(transcript[0].kind, "app.input");
    assert_eq!(transcript[0].data, Some(delta_chord::json!({ "source": "test" })));
    let Some([Message::User(user)]) = transcript[0].model.as_deref() else {
        panic!("a user message")
    };
    // Stamped with the Harness clock at placement.
    assert!(user.timestamp >= EPOCH && user.timestamp <= fixture.now());
    harness.close(&ctx()).await.unwrap();
}

async fn abort_withdraws_only_queued_submissions(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let gate = Arc::new(Notify::new());
    fixture.model.push([Reply::text("answer").gate(&gate)]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let placed = conversation
        .submit(SubmissionDraft::input("one"), &ctx())
        .await
        .unwrap();
    let queued = conversation
        .submit(SubmissionDraft::input("two"), &ctx())
        .await
        .unwrap();
    assert_eq!(queued.abort(&ctx()).await.unwrap(), SubmissionAbort::Aborted);
    assert_eq!(placed.abort(&ctx()).await.unwrap(), SubmissionAbort::AlreadyPlaced);
    let state = queued.status(&ctx()).await.unwrap().state;
    assert!(matches!(&state, SubmissionState::Unanswered { reason, .. } if reason == "aborted"));
    let inbox = harness
        .inspect(&ctx())
        .await
        .unwrap()
        .submissions
        .into_iter()
        .map(|submission| submission.id)
        .collect::<Vec<_>>();
    assert_eq!(inbox, [placed.id()]);
    gate.notify_one();
    settled(&placed).await;
    assert_eq!(queued.abort(&ctx()).await.unwrap(), SubmissionAbort::Settled);
    // The withdrawn item left the inbox, so nothing follows the answer.
    eventually(|| async { harness.inspect(&ctx()).await.unwrap().scheduler.tasks.is_empty() }).await;
    assert_eq!(fixture.model.requests().len(), 1);
    harness.close(&ctx()).await.unwrap();
}
