//! Placement at turn boundaries: follow-ups after the answer, steers after a tool round,
//! queue modes, queued writes and resets.

use std::sync::Arc;

use delta_ai::types::Message;
use delta_harness::agent::QueueMode;
use delta_harness::harness::{Conversation, Submission};
use delta_harness::inbox::{SubmissionDraft, WhenBusy};
use delta_harness::records::{EntryDraft, SubmissionState};
use tokio::sync::Notify;

use crate::Backend;
use crate::support::fixture::{Fixture, chat, ctx, entries, eventually, kinds, live, settled};
use crate::support::model::Reply;
use crate::support::tools::{ToolScript, extension, tool};

cases!(
    a_follow_up_waits_for_the_answer_and_starts_the_next_run,
    a_steer_lands_after_the_tool_round_and_joins_the_run,
    follow_ups_are_placed_one_at_a_time_or_all,
    a_queued_reset_lands_before_queued_input_and_cuts_the_context,
);

fn answer_of(state: &SubmissionState) -> Option<delta_harness::records::EntryId> {
    match state {
        SubmissionState::Done { answer, .. } => *answer,
        other => panic!("done: {other:?}"),
    }
}

fn user_texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::User(user) => Some(user.content.text()),
            _ => None,
        })
        .collect()
}

async fn submit(conversation: &Conversation, draft: SubmissionDraft) -> Submission {
    conversation.submit(draft, &ctx()).await.unwrap()
}

async fn a_follow_up_waits_for_the_answer_and_starts_the_next_run(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let gate = Arc::new(Notify::new());
    fixture
        .model
        .push([Reply::text("first answer").gate(&gate), Reply::text("second answer")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let first = submit(&conversation, SubmissionDraft::input("one")).await;
    let second = submit(&conversation, SubmissionDraft::input("two")).await;
    assert!(matches!(
        second.status(&ctx()).await.unwrap().state,
        SubmissionState::Queued
    ));
    gate.notify_one();
    let first_answer = answer_of(&settled(&first).await.state);
    let second_answer = answer_of(&settled(&second).await.state);
    assert_ne!(first_answer, second_answer);
    let transcript = entries(&conversation).await;
    assert_eq!(
        kinds(&transcript),
        [
            "delta.user",
            "delta.system",
            "delta.assistant",
            "delta.user",
            "delta.assistant"
        ]
    );
    let requests = fixture.model.requests();
    assert_eq!(user_texts(&requests[1].messages), ["one", "two"]);
    harness.close(&ctx()).await.unwrap();
}

async fn a_steer_lands_after_the_tool_round_and_joins_the_run(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let script = ToolScript {
        gate: Some(Arc::new(Notify::new())),
        ..ToolScript::default()
    };
    fixture
        .registry
        .install(extension(&script, vec![tool("read")]))
        .unwrap();
    fixture
        .model
        .push([Reply::calls(&[("c1", "read")]), Reply::text("Read, and steered")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let first = submit(&conversation, SubmissionDraft::input("read it")).await;
    eventually(|| async { script.started() == ["c1"] }).await;
    let steer = submit(
        &conversation,
        SubmissionDraft::input("also check the date").when_busy(WhenBusy::Steer),
    )
    .await;
    script.gate.as_ref().unwrap().notify_one();
    let first_answer = answer_of(&settled(&first).await.state);
    let steer_answer = answer_of(&settled(&steer).await.state);
    // The steer joined the run: both settle with the same answer.
    assert_eq!(first_answer, steer_answer);
    let transcript = entries(&conversation).await;
    assert_eq!(
        kinds(&transcript),
        [
            "delta.user",
            "delta.system",
            "delta.assistant",
            "delta.tool-result",
            "delta.user",
            "delta.assistant"
        ]
    );
    let requests = fixture.model.requests();
    let roles: Vec<&str> = requests[1].messages.iter().map(Message::role).collect();
    assert_eq!(roles, ["system", "user", "assistant", "toolResult", "user"]);
    harness.close(&ctx()).await.unwrap();
}

async fn follow_ups_are_placed_one_at_a_time_or_all(backend: Backend) {
    for (mode, runs) in [(QueueMode::OneAtATime, 3), (QueueMode::All, 2)] {
        let fixture = Fixture::new(backend).await;
        fixture.settings(|settings| settings.follow_up_mode = Some(mode));
        let gate = Arc::new(Notify::new());
        fixture
            .model
            .push([Reply::text("a").gate(&gate), Reply::text("b"), Reply::text("c")]);
        let harness = fixture.open().await;
        let conversation = chat(&harness).await;
        let first = submit(&conversation, SubmissionDraft::input("one")).await;
        let second = submit(&conversation, SubmissionDraft::input("two")).await;
        let third = submit(&conversation, SubmissionDraft::input("three")).await;
        gate.notify_one();
        for submission in [&first, &second, &third] {
            settled(submission).await;
        }
        assert_eq!(fixture.model.requests().len(), runs, "{mode:?}");
        let second_answer = answer_of(&second.status(&ctx()).await.unwrap().state);
        let third_answer = answer_of(&third.status(&ctx()).await.unwrap().state);
        assert_eq!(second_answer == third_answer, mode == QueueMode::All);
        harness.close(&ctx()).await.unwrap();
    }
}

async fn a_queued_reset_lands_before_queued_input_and_cuts_the_context(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let gate = Arc::new(Notify::new());
    fixture
        .model
        .push([Reply::text("old answer").gate(&gate), Reply::text("fresh answer")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let first = submit(&conversation, SubmissionDraft::input("old question")).await;
    let follow_up = submit(&conversation, SubmissionDraft::input("new question")).await;
    conversation.reset(Some("Start over".to_owned()), &ctx()).await.unwrap();
    // A display-only write behind a queued reset queues too, so it lands after the reset.
    let note = submit(&conversation, SubmissionDraft::write(EntryDraft::new("app.note"))).await;
    assert!(matches!(
        note.status(&ctx()).await.unwrap().state,
        SubmissionState::Queued
    ));
    gate.notify_one();
    settled(&first).await;
    settled(&follow_up).await;
    let transcript = entries(&conversation).await;
    assert_eq!(
        kinds(&transcript),
        [
            "delta.user",
            "delta.system",
            "delta.assistant",
            "delta.reset",
            "app.note",
            "delta.user",
            "delta.system",
            "delta.assistant"
        ]
    );
    // The follow-up ran in the context the reset started, with a new baseline.
    let requests = fixture.model.requests();
    assert_eq!(user_texts(&requests[1].messages), ["Start over", "new question"]);
    assert!(matches!(requests[1].messages[0], Message::System(_)));
    assert!(live(&harness, &conversation).await.run.is_none());
    harness.close(&ctx()).await.unwrap();
}
