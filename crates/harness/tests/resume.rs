//! End to end: one conversation with the built-in tools over durable storage, across a
//! clean close and a crash while a command runs. Each reopened Harness continues where the
//! last one stopped: committed turns stay, the interrupted call settles `interrupted` with
//! the output it had published, and the next request carries the whole transcript.

#![allow(clippy::unwrap_used, clippy::too_many_lines)]

mod support;

use std::sync::Arc;

use delta_ai::types::{Message, ToolResultMessage, UserBlock};
use delta_harness::agent::LocalEnvBuilder;
use delta_harness::env::Shell;
use delta_harness::harness::{Conversation, Harness};
use delta_harness::inbox::SubmissionDraft;
use delta_harness::records::SubmissionState;
use delta_harness::tools::{BashOptions, coding_tools};

use support::fixture::{Fixture, PATIENCE, chat, ctx, entries, eventually, live, settled};
use support::model::Reply;
use support::{Backend, TempDir};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn memory() {
    a_conversation_resumes_across_restarts(Backend::Memory).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sqlite() {
    a_conversation_resumes_across_restarts(Backend::SqliteFile).await;
}

/// Drops every handle without closing, as a crash.
async fn crash(harness: Harness) {
    let closed = harness.session().closed();
    drop(harness);
    tokio::time::timeout(PATIENCE, closed)
        .await
        .expect("the old Session ends");
}

async fn run(conversation: &Conversation, text: &str) -> SubmissionState {
    let submission = conversation.submit(SubmissionDraft::input(text), &ctx()).await.unwrap();
    settled(&submission).await.state
}

/// The tool results of the transcript, in order.
async fn results(conversation: &Conversation) -> Vec<ToolResultMessage> {
    entries(conversation)
        .await
        .into_iter()
        .filter_map(|entry| match entry.model.as_deref() {
            Some([Message::ToolResult(result)]) => Some(result.clone()),
            _ => None,
        })
        .collect()
}

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

async fn a_conversation_resumes_across_restarts(backend: Backend) {
    let dir = TempDir::new();
    let mut fixture = Fixture::new(backend).await;
    let envs = Arc::new(LocalEnvBuilder::new(dir.path()));
    fixture.env = Some(envs.clone());
    fixture.registry.install(coding_tools(BashOptions::default())).unwrap();

    // Turn 1 writes a file and ends; the Harness closes cleanly.
    fixture.model.push([
        Reply::calls_with(&[(
            "c1",
            "write",
            serde_json::json!({ "path": "notes.txt", "content": "draft\n" }),
        )]),
        Reply::text("Written"),
    ]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    assert!(matches!(
        run(&conversation, "write notes").await,
        SubmissionState::Done { .. }
    ));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("notes.txt")).unwrap(),
        "draft\n"
    );
    drop(conversation);
    harness.close(&ctx()).await.unwrap();

    // Turn 2: the process dies while a command runs, after it published some output.
    fixture.model.push([Reply::calls_with(&[(
        "c2",
        "bash",
        serde_json::json!({ "command": "echo started; sleep 30" }),
    )])]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation
        .submit(SubmissionDraft::input("run it"), &ctx())
        .await
        .unwrap();
    let id = submission.id();
    eventually(|| async {
        live(&harness, &conversation)
            .await
            .tools
            .into_iter()
            .flatten()
            .any(|slot| slot.progress.output.as_deref() == Some("started\n"))
    })
    .await;
    drop((submission, conversation));
    crash(harness).await;

    // Reopened: `bash` is not replay-safe, so the call settles `interrupted` with the
    // output it had published, and the turn goes on to the model's answer.
    fixture.model.push([Reply::text("It was interrupted")]);
    let harness = fixture.open().await;
    let submission = harness.submission(id, &ctx()).await.unwrap().unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    let conversation = chat(&harness).await;
    let results_so_far = results(&conversation).await;
    assert_eq!(results_so_far.len(), 2);
    assert!(!results_so_far[0].is_error);
    let interrupted = &results_so_far[1];
    assert!(interrupted.is_error);
    assert_eq!(
        texts(interrupted),
        [
            "started\n",
            "<harness>\n[error] Tool bash was interrupted and may have partially run\n</harness>"
        ]
    );

    // Turn 3 reads what turn 1 wrote; its request carries every earlier result.
    fixture.model.push([
        Reply::calls_with(&[("c3", "read", serde_json::json!({ "path": "notes.txt" }))]),
        Reply::text("It says draft"),
    ]);
    assert!(matches!(
        run(&conversation, "read notes").await,
        SubmissionState::Done { .. }
    ));
    let all = results(&conversation).await;
    assert_eq!(texts(&all[2]), ["draft\n"]);
    let last = fixture.model.requests().pop().unwrap();
    let called: Vec<&str> = last
        .messages
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult(result) => Some(result.tool_call_id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(called, ["c1", "c2", "c3"]);
    drop(conversation);
    harness.close(&ctx()).await.unwrap();
    // The crashed command may outlive its Session; the environment owner ends it.
    envs.env(None).cleanup(&ctx()).await;
}
