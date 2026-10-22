//! A run from submission to answer.

use delta_ai::types::{Message, StopReason};
use delta_harness::inbox::SubmissionDraft;
use delta_harness::records::{SubmissionState, TaskOutcome};

use crate::Backend;
use crate::support::fixture::{Fixture, MODEL_ID, chat, ctx, entries, idle, kinds, live, settled};
use crate::support::model::Reply;

cases!(
    an_input_runs_to_an_answer,
    usage_is_recorded_per_model,
    the_provider_session_id_is_stable
);

async fn an_input_runs_to_an_answer(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture.model.push([Reply::text("Hello there")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("Hi"), &ctx()).await.unwrap();
    let record = settled(&submission).await;
    let SubmissionState::Done { entry, answer } = record.state else {
        panic!("done: {record:?}")
    };
    let transcript = entries(&conversation).await;
    assert_eq!(kinds(&transcript), ["delta.user", "delta.system", "delta.assistant"]);
    assert_eq!(transcript[0].id, entry);
    assert_eq!(Some(transcript[2].id), answer);
    let Some([Message::Assistant(message)]) = transcript[2].model.as_deref() else {
        panic!("an assistant entry")
    };
    assert_eq!(message.text(), "Hello there");
    assert_eq!(message.stop_reason, StopReason::Stop);
    // The generation was stamped on its entries and completed with the answer.
    let generation = transcript[2].by_task_id.unwrap();
    let task = harness.get_task(generation, &ctx()).await.unwrap().unwrap();
    assert_eq!(task.kind, "delta.generation");
    let outcome = task.state.outcome().cloned().unwrap();
    assert!(matches!(outcome, TaskOutcome::Completed { .. }));
    idle(&conversation).await;
    let state = live(&harness, &conversation).await;
    assert!(state.run.is_none() && state.generation.is_none());
    // The request sent the user message with the baseline leading.
    let requests = fixture.model.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].model, MODEL_ID);
    assert!(matches!(
        requests[0].messages.as_slice(),
        [Message::System(_), Message::User(_)]
    ));
    harness.close(&ctx()).await.unwrap();
}

async fn usage_is_recorded_per_model(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture
        .model
        .push([Reply::text("one").usage(10, 2), Reply::text("two").usage(5, 1)]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let first = conversation.submit(SubmissionDraft::input("a"), &ctx()).await.unwrap();
    settled(&first).await;
    let second = conversation.submit(SubmissionDraft::input("b"), &ctx()).await.unwrap();
    settled(&second).await;
    let usage = harness.usage(&ctx()).await.unwrap();
    let total = &usage.models[&format!("fake/{MODEL_ID}")];
    assert_eq!((total.input, total.output, total.total_tokens), (15, 3, 18));
    assert!(usage.tools.is_empty());
    // The second request carried the first exchange.
    let requests = fixture.model.requests();
    assert_eq!(requests[1].messages.len(), 4);
    harness.close(&ctx()).await.unwrap();
}

async fn the_provider_session_id_is_stable(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture.model.push([Reply::text("one"), Reply::text("two")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    for text in ["a", "b"] {
        let submission = conversation.submit(SubmissionDraft::input(text), &ctx()).await.unwrap();
        settled(&submission).await;
    }
    let requests = fixture.model.requests();
    let ids: Vec<Option<String>> = requests.iter().map(|request| request.session_id.clone()).collect();
    assert!(ids[0].as_ref().is_some_and(|id| id.len() == 36));
    assert_eq!(ids[0], ids[1]);
    harness.close(&ctx()).await.unwrap();
}
