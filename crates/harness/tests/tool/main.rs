//! The `delta.tool` task: the pipeline (resolution, repair, validation, hooks, approval
//! policy), execution with output windows and progress, settlement, recovery after a crash,
//! abort, the structured output contract and nested calls.
//!
//! Ported from upstream `test/harness-tool.test.ts`, `harness-tool-recovery`,
//! `harness-nested-tools` and `truncate.test.ts` where they apply. Every case runs against
//! the memory backend and SQLite in a temp file with a scripted model; timed cases run on
//! a paused clock against memory.

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

mod basic;
mod hooks;
mod nested;
mod progress;
mod recovery;
mod structured;

use delta_ai::types::{Message, ToolResultMessage, UserBlock};
use delta_harness::harness::Conversation;
use delta_harness::inbox::SubmissionDraft;
use delta_harness::records::SubmissionState;
pub(crate) use support::Backend;

use crate::support::fixture::{ctx, entries, settled};

/// Submits `text` and waits until it settles.
async fn run(conversation: &Conversation, text: &str) -> SubmissionState {
    let submission = conversation.submit(SubmissionDraft::input(text), &ctx()).await.unwrap();
    settled(&submission).await.state
}

/// The tool results of the transcript, in order.
async fn results(conversation: &Conversation) -> Vec<ToolResultMessage> {
    entries(conversation)
        .await
        .into_iter()
        .filter(|entry| entry.kind == "delta.tool-result")
        .map(|entry| match entry.model.as_deref() {
            Some([Message::ToolResult(result)]) => result.clone(),
            other => panic!("a tool result: {other:?}"),
        })
        .collect()
}

/// The text items of a result, in order.
fn texts(result: &ToolResultMessage) -> Vec<&str> {
    result
        .content
        .iter()
        .filter_map(|block| match block {
            UserBlock::Text(text) => Some(text.text.as_str()),
            UserBlock::Image(_) => None,
        })
        .collect()
}
