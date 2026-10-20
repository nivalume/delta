//! Conversations: the root, creation with explicit ownership, the creation hook's documents,
//! configuration and resolution, forks, bound commits and context reads.

use std::sync::{Arc, Mutex};

use delta_chord::json;
use delta_harness::agent::{AgentChange, AgentDoc, AgentState, Change};
use delta_harness::documents::DocAddress;
use delta_harness::generation::{LiveDoc, ProviderDoc};
use delta_harness::harness::{ConversationCreateOptions, HarnessError, RootOptions};
use delta_harness::inbox::{InboxDoc, SubmissionDraft};
use delta_harness::model::ModelRef;
use delta_harness::records::{ConversationId, EntryDraft, ROOT_CONVERSATION_ID};
use delta_harness::session::{ConversationHook, Tx, TxError};
use delta_harness::usage::UsageDoc;
use futures::future::BoxFuture;

use crate::Backend;
use crate::support::fixture::{Fixture, INSTRUCTIONS, chat, ctx, model_ref, settled};
use crate::support::model::Reply;

cases!(
    the_root_is_created_once,
    every_conversation_starts_with_the_builtin_documents,
    configuration_is_stored_and_resolved,
    a_fork_copies_the_agent_as_of_its_entry,
    a_bound_commit_and_context_reads,
);

async fn the_root_is_created_once(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let harness = fixture.open().await;
    let root = chat(&harness).await;
    assert_eq!(root.id(), ROOT_CONVERSATION_ID);
    // A second call ignores its agent change.
    let again = harness
        .root(
            RootOptions {
                agent: Some(AgentChange {
                    instructions: Change::Set("other".to_owned()),
                    ..AgentChange::default()
                }),
                init: None,
            },
            &ctx(),
        )
        .await
        .unwrap();
    assert_eq!(again, root);
    assert_eq!(
        root.agent(&ctx()).await.unwrap().instructions.as_deref(),
        Some(INSTRUCTIONS)
    );
    assert!(
        harness
            .conversation(ConversationId(999), &ctx())
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(harness.conversation(root.id(), &ctx()).await.unwrap(), Some(root));
    harness.close(&ctx()).await.unwrap();
}

/// A host hook that records the conversations it saw.
struct Recorder(Arc<Mutex<Vec<ConversationId>>>);

impl ConversationHook for Recorder {
    fn conversation_created<'a>(
        &'a self,
        _tx: &'a mut Tx<'_>,
        record: &'a delta_harness::records::ConversationRecord,
    ) -> BoxFuture<'a, Result<(), TxError>> {
        self.0.lock().unwrap().push(record.id);
        Box::pin(async { Ok(()) })
    }
}

async fn every_conversation_starts_with_the_builtin_documents(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut options = fixture.options();
    options.conversation_created = Some(Arc::new(Recorder(seen.clone())));
    let harness = delta_harness::harness::Harness::open(fixture.storage.clone(), options, &ctx())
        .await
        .unwrap();
    let conversation = harness
        .create_conversation(ConversationCreateOptions::ownerless(), &ctx())
        .await
        .unwrap();
    let id = conversation.id();
    assert_eq!(*seen.lock().unwrap(), [id]);
    let live = harness.snapshot(DocAddress::<LiveDoc>::new(id), &ctx()).await.unwrap();
    assert_eq!(live, Some(delta_chord::json!({})));
    let inbox = harness.snapshot(DocAddress::<InboxDoc>::new(id), &ctx()).await.unwrap();
    assert_eq!(inbox, Some(delta_chord::json!({ "items": [] })));
    let usage = harness.snapshot(DocAddress::<UsageDoc>::new(id), &ctx()).await.unwrap();
    assert_eq!(usage, Some(delta_chord::json!({ "models": {}, "tools": {} })));
    let agent = harness.snapshot(DocAddress::<AgentDoc>::new(id), &ctx()).await.unwrap();
    assert_eq!(agent, Some(delta_chord::json!({})));
    let provider = harness
        .snapshot(DocAddress::<ProviderDoc>::new(id), &ctx())
        .await
        .unwrap()
        .unwrap();
    let session_id = provider.get("session_id").and_then(json::Json::as_str).unwrap();
    // UUIDv7: version nibble 7.
    assert_eq!(session_id.as_bytes()[14], b'7');
    // A second conversation has its own identity.
    let other = harness
        .create_conversation(ConversationCreateOptions::ownerless(), &ctx())
        .await
        .unwrap();
    let other_provider = harness
        .snapshot(DocAddress::<ProviderDoc>::new(other.id()), &ctx())
        .await
        .unwrap()
        .unwrap();
    assert_ne!(other_provider, provider);
    harness.close(&ctx()).await.unwrap();
}

async fn configuration_is_stored_and_resolved(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let harness = fixture.open().await;
    let conversation = harness
        .create_conversation(
            ConversationCreateOptions::ownerless().with_agent(AgentChange {
                model: Change::Set(model_ref()),
                cwd: Change::Set("/work".to_owned()),
                ..AgentChange::default()
            }),
            &ctx(),
        )
        .await
        .unwrap();
    conversation
        .configure(
            AgentChange {
                cwd: Change::Clear,
                instructions: Change::Set("Short answers".to_owned()),
                ..AgentChange::default()
            },
            &ctx(),
        )
        .await
        .unwrap();
    let stored = harness
        .snapshot(DocAddress::<AgentDoc>::new(conversation.id()), &ctx())
        .await
        .unwrap()
        .unwrap();
    let state: AgentState = json::from_json(&stored).unwrap();
    assert_eq!(state.model, Some(model_ref()));
    assert_eq!(state.cwd, None);
    let agent = conversation.agent(&ctx()).await.unwrap();
    assert_eq!(agent.instructions.as_deref(), Some("Short answers"));
    let keys: Vec<&str> = agent.sections.iter().map(|section| section.key()).collect();
    assert_eq!(keys, ["instructions"]);
    // A configuration change appends nothing and starts nothing.
    assert!(fixture.model.requests().is_empty());
    harness.close(&ctx()).await.unwrap();
}

async fn a_fork_copies_the_agent_as_of_its_entry(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture.model.push([Reply::text("first")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("hi"), &ctx()).await.unwrap();
    let settled = settled(&submission).await;
    let delta_harness::records::SubmissionState::Done {
        answer: Some(answer), ..
    } = settled.state
    else {
        panic!("done")
    };
    conversation
        .configure(
            AgentChange {
                model: Change::Set(ModelRef::new("fake", "other")),
                ..AgentChange::default()
            },
            &ctx(),
        )
        .await
        .unwrap();
    let fork = conversation
        .fork(answer, ConversationCreateOptions::ownerless(), &ctx())
        .await
        .unwrap();
    assert_eq!(fork.agent(&ctx()).await.unwrap().model, Some(model_ref()));
    assert_eq!(
        conversation.agent(&ctx()).await.unwrap().model,
        Some(ModelRef::new("fake", "other"))
    );
    // The fork sees the parent's history through the fork entry and runs on its own.
    let view = fork.context(None, &ctx()).await.unwrap();
    assert_eq!(view.messages.len(), 3);
    let parent = harness
        .snapshot(DocAddress::<ProviderDoc>::new(conversation.id()), &ctx())
        .await
        .unwrap();
    let child = harness
        .snapshot(DocAddress::<ProviderDoc>::new(fork.id()), &ctx())
        .await
        .unwrap();
    assert_ne!(parent, child);
    fixture.model.push([Reply::text("from the fork")]);
    let submission = fork.submit(SubmissionDraft::input("continue"), &ctx()).await.unwrap();
    assert!(matches!(
        crate::support::fixture::settled(&submission).await.state,
        delta_harness::records::SubmissionState::Done { .. }
    ));
    let requests = fixture.model.requests();
    assert_eq!(requests[1].messages.len(), 4);
    harness.close(&ctx()).await.unwrap();
}

async fn a_bound_commit_and_context_reads(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let entry = conversation
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                let draft = EntryDraft {
                    model: Some(vec![delta_ai::types::Message::user("written directly")]),
                    ..EntryDraft::new("app.note")
                };
                tx.append_entry(ROOT_CONVERSATION_ID, draft).await
            })
        })
        .await
        .unwrap();
    let view = conversation.context(None, &ctx()).await.unwrap();
    assert_eq!(view.entries.len(), 1);
    assert_eq!(view.messages.len(), 1);
    let at = conversation.context(Some(entry.id), &ctx()).await.unwrap();
    assert_eq!(at, view);
    let other = harness
        .create_conversation(ConversationCreateOptions::ownerless(), &ctx())
        .await
        .unwrap();
    let invisible = other.context(Some(entry.id), &ctx()).await;
    assert!(matches!(invisible, Err(HarnessError::Context(_))));
    harness.close(&ctx()).await.unwrap();
}
