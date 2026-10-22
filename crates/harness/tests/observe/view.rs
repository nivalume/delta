//! The view mount: hydration, advancing per commit, head markers, sharing.

use delta_chord::state::DeliveryKind;
use delta_harness::harness::Conversation;
use delta_harness::inbox::SubmissionDraft;
use delta_harness::observe::{ConversationView, ViewState};
use delta_harness::records::SubmissionState;

use crate::Backend;
use crate::support::fixture::{Fixture, chat, ctx, entries, eventually, idle, settled};
use crate::support::model::Reply;
use crate::support::tools::{ToolScript, extension, tool};

cases!(
    a_view_follows_a_run_with_tools,
    a_summary_rebases_the_transcript,
    observers_share_one_mount,
);

async fn run(conversation: &Conversation, text: &str) -> SubmissionState {
    let submission = conversation.submit(SubmissionDraft::input(text), &ctx()).await.unwrap();
    settled(&submission).await.state
}

/// The view as rebuilt from storage, once no other observer keeps the mount.
async fn rebuilt(conversation: &Conversation, held: ViewState) -> ConversationView {
    let followed = held.view().unwrap();
    drop(held);
    let fresh = conversation.view_state(&ctx()).await.unwrap();
    assert_eq!(fresh.sequence(), 0, "a new mount");
    let rebuilt = fresh.view().unwrap();
    assert_eq!(followed, rebuilt);
    rebuilt
}

async fn a_view_follows_a_run_with_tools(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let script = ToolScript::default();
    fixture
        .registry
        .install(extension(&script, vec![tool("read")]))
        .unwrap();
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let view = conversation.view_state(&ctx()).await.unwrap();
    let mut subscription = view.subscribe();
    let hydrated = subscription.next().await.unwrap();
    assert_eq!(hydrated.kind, DeliveryKind::Hydrate);
    let initial = view.view().unwrap();
    assert_eq!(initial.conversation.id, conversation.id());
    assert!(initial.entries.is_empty());
    assert!(initial.docs.contains_key("delta.agent"));

    fixture
        .model
        .push([Reply::calls(&[("c1", "read"), ("c2", "read")]), Reply::text("All read")]);
    assert!(matches!(
        run(&conversation, "read twice").await,
        SubmissionState::Done { .. }
    ));
    idle(&conversation).await;
    // Other observers of the last commit (the submission's waiter) may wake first.
    let transcript = entries(&conversation).await;
    eventually(|| async { view.view().unwrap().entries == transcript }).await;
    let mut updates = 0;
    while let Some(delivery) = subscription.try_next() {
        assert_eq!(delivery.kind, DeliveryKind::Update);
        assert!(!delivery.ops.is_empty());
        updates += 1;
    }
    assert!(updates > 3, "one update per commit that changed the view: {updates}");
    let view_now = view.view().unwrap();
    for kind in ["delta.live", "delta.inbox", "delta.provider", "delta.usage"] {
        assert!(view_now.docs.contains_key(kind), "{kind} is mounted");
    }
    drop(subscription);
    rebuilt(&conversation, view).await;
    harness.close(&ctx()).await.unwrap();
}

async fn a_summary_rebases_the_transcript(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture.settings(|settings| {
        settings.compaction = Some(delta_harness::agent::CompactionSettings {
            policy: delta_harness::agent::PartialCompactionPolicy {
                keep_recent_tokens: Some(10),
                ..Default::default()
            },
            ..Default::default()
        });
    });
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let view = conversation.view_state(&ctx()).await.unwrap();
    fixture.model.push([Reply::text(&"A long first answer. ".repeat(20))]);
    run(&conversation, "First question").await;
    fixture.model.push([Reply::text("Second answer")]);
    run(&conversation, &"Second question. ".repeat(20)).await;
    fixture.model.push_summaries([Reply::text("Summary.")]);
    let task = conversation.compact(None, &ctx()).await.unwrap();
    harness.wait_for_task(task, &ctx()).await.unwrap();
    idle(&conversation).await;
    eventually(|| async { view.view().unwrap().entries[0].kind == "delta.compaction" }).await;
    let followed = view.view().unwrap();
    let first = &followed.entries[0];
    assert_eq!(first.kind, "delta.compaction");
    assert!(followed.entries[1..].iter().all(|entry| entry.head.is_none()));
    assert!(followed.entries.len() < entries(&conversation).await.len());
    let rebuilt = rebuilt(&conversation, view).await;
    assert_eq!(rebuilt.entries[0].id, first.id);
    harness.close(&ctx()).await.unwrap();
}

async fn observers_share_one_mount(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let first = conversation.view_state(&ctx()).await.unwrap();
    let second = conversation.view_state(&ctx()).await.unwrap();
    fixture.model.push([Reply::text("Hi")]);
    run(&conversation, "Hello").await;
    idle(&conversation).await;
    let transcript = entries(&conversation).await;
    eventually(|| async { first.view().unwrap().entries == transcript }).await;
    assert!(first.sequence() > 0);
    assert_eq!(first.sequence(), second.sequence());
    assert_eq!(first.value(), second.value());
    harness.close(&ctx()).await.unwrap();
}
