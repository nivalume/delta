//! Context reads over storage: fork caps, cut-off reads and head markers.

use delta_ai::types::Message;
use delta_harness::harness::{Conversation, ConversationCreateOptions, HarnessError};
use delta_harness::records::{EntryDraft, EntryId, Head};

use crate::Backend;
use crate::support::fixture::{Fixture, chat, ctx};

cases!(
    a_fork_sees_its_parent_through_the_fork_entry,
    a_read_cut_off_at_an_entry_ends_there
);

/// Appends a user-message entry with `text`.
async fn say(conversation: &Conversation, text: &str) -> EntryId {
    let draft = EntryDraft {
        model: Some(vec![Message::user(text)]),
        ..EntryDraft::new("delta.user")
    };
    let id = conversation.id();
    conversation
        .commit(&ctx(), move |tx| {
            Box::pin(async move { Ok(tx.append_entry(id, draft).await?.id) })
        })
        .await
        .unwrap()
}

fn texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::User(user) => Some(user.content.text()),
            _ => None,
        })
        .collect()
}

async fn a_fork_sees_its_parent_through_the_fork_entry(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let harness = fixture.open().await;
    let parent = chat(&harness).await;
    say(&parent, "one").await;
    let at = say(&parent, "two").await;
    say(&parent, "parent only").await;
    let child = parent
        .fork(at, ConversationCreateOptions::ownerless(), &ctx())
        .await
        .unwrap();
    say(&child, "child only").await;
    let view = child.context(None, &ctx()).await.unwrap();
    assert_eq!(texts(&view.messages), ["one", "two", "child only"]);
    // A head marker in the child cuts the inherited history.
    let marker = EntryDraft {
        model: Some(vec![Message::user("summary")]),
        head: Some(Head::At(at)),
        ..EntryDraft::new("app.summary")
    };
    let id = child.id();
    child
        .commit(&ctx(), move |tx| {
            Box::pin(async move { tx.append_entry(id, marker).await })
        })
        .await
        .unwrap();
    let view = child.context(None, &ctx()).await.unwrap();
    assert_eq!(texts(&view.messages), ["summary", "two", "child only"]);
    let parent_view = parent.context(None, &ctx()).await.unwrap();
    assert_eq!(texts(&parent_view.messages), ["one", "two", "parent only"]);
    harness.close(&ctx()).await.unwrap();
}

async fn a_read_cut_off_at_an_entry_ends_there(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let empty = conversation.context(None, &ctx()).await.unwrap();
    assert!(empty.entries.is_empty() && empty.messages.is_empty());
    let first = say(&conversation, "one").await;
    say(&conversation, "two").await;
    let view = conversation.context(Some(first), &ctx()).await.unwrap();
    assert_eq!(texts(&view.messages), ["one"]);
    let other = harness
        .create_conversation(ConversationCreateOptions::ownerless(), &ctx())
        .await
        .unwrap();
    assert!(matches!(
        other.context(Some(first), &ctx()).await,
        Err(HarnessError::Context(_))
    ));
    harness.close(&ctx()).await.unwrap();
}
