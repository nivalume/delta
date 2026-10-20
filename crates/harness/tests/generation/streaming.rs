//! Streaming partials into `delta.live` on a throttle, on tokio's paused clock.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use delta_ai::types::AssistantMessage;
use delta_chord::json::{self, Json};
use delta_harness::documents::DocAddress;
use delta_harness::generation::LiveDoc;
use delta_harness::inbox::SubmissionDraft;

use crate::Backend;
use crate::support::fixture::{Fixture, chat, ctx, settled};
use crate::support::model::Reply;

paused_cases!(partials_commit_at_most_once_per_interval);

async fn partials_commit_at_most_once_per_interval(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let words: Vec<String> = (1..=10).map(|n| format!("w{n}")).collect();
    fixture
        .model
        .push([Reply::text(&words.join(" ")).step(Duration::from_millis(30))]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let state = harness
        .document_state(DocAddress::<LiveDoc>::new(conversation.id()), &ctx())
        .await
        .unwrap()
        .unwrap();
    let values: Arc<Mutex<Vec<Json>>> = Arc::default();
    let mut subscription = state.subscribe();
    let collected = values.clone();
    let collector = tokio::spawn(async move {
        while let Some(delivery) = subscription.next().await {
            assert_eq!(delivery.skipped, 0);
            collected.lock().unwrap().push(delivery.value);
        }
    });
    let start = fixture.now();
    let submission = conversation
        .submit(SubmissionDraft::input("stream"), &ctx())
        .await
        .unwrap();
    settled(&submission).await;
    assert!(fixture.now() - start >= 300);
    harness.close(&ctx()).await.unwrap();
    collector.await.unwrap();
    let values = values.lock().unwrap().clone();
    let partials: Vec<String> = values
        .iter()
        .filter_map(|value| value.get("generation")?.get("message").cloned())
        .map(|message| json::from_json::<AssistantMessage>(&message).unwrap().text())
        .collect();
    // 300 ms of streaming at one commit per 100 ms, with the trailing edge.
    assert!((2..=4).contains(&partials.len()), "partials: {partials:?}");
    let full = words.join(" ");
    for pair in partials.windows(2) {
        assert!(pair[1].len() > pair[0].len());
    }
    for partial in &partials {
        assert!(full.starts_with(partial.as_str()), "{partial:?}");
    }
    // The classifying commit cleared the generation: nothing runs, a base was stored.
    let last = values.last().unwrap();
    assert!(last.get("generation").is_none() && last.get("run").is_none());
}
