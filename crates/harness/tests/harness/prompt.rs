//! The positional system prompt across agent and registry changes.

use std::sync::Arc;

use delta_ai::types::Message;
use delta_harness::agent::{AgentChange, Change};
use delta_harness::harness::Conversation;
use delta_harness::inbox::SubmissionDraft;
use delta_harness::prompt::{SYSTEM_ENTRY, TextSection};
use delta_harness::registry::Extension;

use crate::Backend;
use crate::support::fixture::{Fixture, chat, ctx, entries, settled};
use crate::support::model::Reply;
use crate::support::tools::tool;

cases!(only_changed_sections_and_tools_are_appended);

async fn ask(conversation: &Conversation, text: &str) {
    let submission = conversation.submit(SubmissionDraft::input(text), &ctx()).await.unwrap();
    settled(&submission).await;
}

/// The section patches of the conversation's `delta.system` entries, in order.
async fn system_patches(conversation: &Conversation) -> Vec<Vec<(String, Option<String>)>> {
    entries(conversation)
        .await
        .into_iter()
        .filter(|entry| entry.kind == SYSTEM_ENTRY)
        .map(|entry| match entry.model.as_deref() {
            Some([Message::System(system)]) => system
                .sections
                .iter()
                .flatten()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
            other => panic!("a system message: {other:?}"),
        })
        .collect()
}

async fn only_changed_sections_and_tools_are_appended(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture
        .registry
        .install(
            Extension::new("app")
                .with_section(Arc::new(TextSection::new("role", "You review code")))
                .with_tool(tool("read")),
        )
        .unwrap();
    fixture
        .model
        .push([Reply::text("1"), Reply::text("2"), Reply::text("3"), Reply::text("4")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    ask(&conversation, "first").await;
    // Nothing changed: no new system entry.
    ask(&conversation, "second").await;
    assert_eq!(system_patches(&conversation).await.len(), 1);
    // A reinstall that changes one section appends only that section.
    fixture
        .registry
        .install(
            Extension::new("app")
                .with_section(Arc::new(TextSection::new("role", "You review tests")))
                .with_tool(tool("read")),
        )
        .unwrap();
    ask(&conversation, "third").await;
    let patches = system_patches(&conversation).await;
    assert_eq!(patches.len(), 2);
    assert_eq!(
        patches[1],
        [("role".to_owned(), Some("<role>\nYou review tests\n</role>".to_owned()))]
    );
    // Removing the instructions removes that section only.
    conversation
        .configure(
            AgentChange {
                instructions: Change::Clear,
                ..AgentChange::default()
            },
            &ctx(),
        )
        .await
        .unwrap();
    ask(&conversation, "fourth").await;
    let patches = system_patches(&conversation).await;
    assert_eq!(patches.len(), 3);
    assert_eq!(patches[2], [("instructions".to_owned(), None)]);
    // The last request replays to the current sections, leading with the baseline.
    let requests = fixture.model.requests();
    let last = &requests[3].messages;
    assert!(matches!(last[0], Message::System(_)));
    let shown = delta_harness::prompt::replay_sections(last);
    let keys: Vec<&String> = shown.keys().collect();
    assert_eq!(keys, ["role"]);
    harness.close(&ctx()).await.unwrap();
}
