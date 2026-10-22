//! The `delta.compaction` task: blocking compactions (threshold, overflow) the generation
//! waits for, conversation-owned ones (background, manual) placed as write submissions,
//! retries, hooks, abort and failures.
//!
//! Ported from upstream `test/harness-compaction.test.ts` where it applies. Every case runs
//! against memory and SQLite in a temp file with a scripted model; summarization requests
//! take the fake model's summary queue. Timed cases run on a paused clock against memory.

#![allow(clippy::unwrap_used, clippy::too_many_lines)]

#[path = "../support/mod.rs"]
mod support;

/// Runs each case against memory and SQLite in a file, on a multi-threaded runtime.
macro_rules! cases {
    ($($case:ident),* $(,)?) => {
        mod memory {
            $(#[tokio::test(flavor = "multi_thread", worker_threads = 4)] async fn $case() { super::$case(crate::Backend::Memory).await; })*
        }
        mod sqlite {
            $(#[tokio::test(flavor = "multi_thread", worker_threads = 4)] async fn $case() { super::$case(crate::Backend::SqliteFile).await; })*
        }
    };
}

/// [`cases!`] with tokio's clock paused, memory only (SQLite runs on its own thread).
macro_rules! paused_cases {
    ($($case:ident),* $(,)?) => {
        mod paused {
            $(#[tokio::test(start_paused = true)] async fn $case() { super::$case(crate::Backend::Memory).await; })*
        }
    };
}

mod blocking;
mod placement;

use delta_ai::types::Message;
use delta_harness::agent::{CompactionSettings, PartialCompactionPolicy};
use delta_harness::harness::Conversation;
use delta_harness::inbox::SubmissionDraft;
use delta_harness::records::{EntryRecord, SubmissionState};
pub(crate) use support::Backend;

use crate::support::fixture::{Fixture, ctx, entries, settled};
use crate::support::model::Reply;

/// A first question with a long answer, so the second run has history before a cut.
const HISTORY: &str = "An answer long enough to be worth summarizing. ";

/// Keeps only a few tokens verbatim, so any second question is a cut.
fn keep_little(fixture: &Fixture, change: impl FnOnce(&mut PartialCompactionPolicy)) {
    fixture.settings(|settings| {
        let mut policy = PartialCompactionPolicy {
            keep_recent_tokens: Some(10),
            ..PartialCompactionPolicy::default()
        };
        change(&mut policy);
        settings.compaction = Some(CompactionSettings {
            policy,
            ..CompactionSettings::default()
        });
    });
}

/// Submits `text` and waits until it settles.
async fn run(conversation: &Conversation, text: &str) -> SubmissionState {
    let submission = conversation.submit(SubmissionDraft::input(text), &ctx()).await.unwrap();
    settled(&submission).await.state
}

/// Runs a first exchange, so there is something to compact.
async fn history(fixture: &Fixture, conversation: &Conversation) {
    fixture.model.push([Reply::text(&HISTORY.repeat(20))]);
    assert!(matches!(
        run(conversation, "First question").await,
        SubmissionState::Done { .. }
    ));
}

/// The summary entries of the transcript.
async fn summaries(conversation: &Conversation) -> Vec<EntryRecord> {
    entries(conversation)
        .await
        .into_iter()
        .filter(|entry| entry.kind == "delta.compaction")
        .collect()
}

/// The text of an entry's only user message.
fn user_text(entry: &EntryRecord) -> String {
    match entry.model.as_deref() {
        Some([Message::User(user)]) => user.content.text(),
        other => panic!("a user message: {other:?}"),
    }
}

/// The summarization requests the fake model received.
fn summary_requests(fixture: &Fixture) -> Vec<crate::support::model::Recorded> {
    fixture
        .model
        .requests()
        .into_iter()
        .filter(|request| request.max_tokens.is_some())
        .collect()
}
