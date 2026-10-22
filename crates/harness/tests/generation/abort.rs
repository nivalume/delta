//! Abort of a running generation, deferred responses, and recovery after a crash.

use std::sync::Arc;

use delta_ai::types::{Message, StopReason, ToolResultMessage, UserBlock};
use delta_harness::harness::{AbortOptions, Harness};
use delta_harness::inbox::SubmissionDraft;
use delta_harness::records::{EntryRecord, SubmissionState, TaskOutcome};
use tokio::sync::Notify;

use crate::Backend;
use crate::support::fixture::{Fixture, PATIENCE, chat, ctx, entries, eventually, kinds, live, settled};
use crate::support::model::Reply;
use crate::support::tools::{ToolScript, extension, sequential_tool, tool};

cases!(
    abort_ends_a_streaming_generation,
    abort_during_a_round_answers_unstarted_calls,
    a_crash_mid_request_converts_the_partial_and_requests_again,
);
paused_cases!(a_deferred_response_is_polled, abort_cancels_a_polled_response);

fn assistant(entry: &EntryRecord) -> &delta_ai::types::AssistantMessage {
    match entry.model.as_deref() {
        Some([Message::Assistant(message)]) => message,
        other => panic!("an assistant entry: {other:?}"),
    }
}

/// Waits until the conversation's `delta.live` holds a partial.
async fn partial_committed(harness: &Harness, conversation: &delta_harness::harness::Conversation) {
    eventually(|| async {
        live(harness, conversation)
            .await
            .generation
            .is_some_and(|generation| generation.message.is_some())
    })
    .await;
}

async fn abort_ends_a_streaming_generation(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let gate = Arc::new(Notify::new());
    fixture.model.push([Reply::text("partial answer").gate(&gate)]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("hi"), &ctx()).await.unwrap();
    partial_committed(&harness, &conversation).await;
    tokio::time::timeout(PATIENCE, conversation.abort(AbortOptions::default(), &ctx()))
        .await
        .unwrap()
        .unwrap();
    let state = settled(&submission).await.state;
    assert!(
        matches!(&state, SubmissionState::Unanswered { reason, .. } if reason == "aborted"),
        "{state:?}"
    );
    let transcript = entries(&conversation).await;
    assert_eq!(kinds(&transcript), ["delta.user", "delta.system", "delta.assistant"]);
    let message = assistant(&transcript[2]);
    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert_eq!(message.text(), "partial answer");
    let task = harness
        .get_task(transcript[2].by_task_id.unwrap(), &ctx())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.state.outcome(), Some(TaskOutcome::Aborted { .. })));
    let state = live(&harness, &conversation).await;
    assert!(state.run.is_none() && state.generation.is_none());
    harness.close(&ctx()).await.unwrap();
}

async fn abort_during_a_round_answers_unstarted_calls(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let script = ToolScript {
        gate: Some(Arc::new(Notify::new())),
        ..ToolScript::default()
    };
    fixture
        .registry
        .install(extension(&script, vec![tool("read"), sequential_tool("write")]))
        .unwrap();
    fixture.model.push([Reply::calls(&[("c1", "write"), ("c2", "read")])]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    eventually(|| async { script.started() == ["c1"] }).await;
    tokio::time::timeout(PATIENCE, conversation.abort(AbortOptions::default(), &ctx()))
        .await
        .unwrap()
        .unwrap();
    let state = settled(&submission).await.state;
    assert!(matches!(&state, SubmissionState::Unanswered { reason, .. } if reason == "aborted"));
    let transcript = entries(&conversation).await;
    // The started call settled an `aborted` result from its slot; the call that never
    // started got its `aborted` result from the generation.
    let results: Vec<&ToolResultMessage> = transcript
        .iter()
        .filter(|entry| entry.kind == "delta.tool-result")
        .map(|entry| match entry.model.as_deref() {
            Some([Message::ToolResult(result)]) => result,
            _ => panic!("a result"),
        })
        .collect();
    let calls: Vec<&str> = results.iter().map(|result| result.tool_call_id.as_str()).collect();
    assert_eq!(calls, ["c1", "c2"]);
    assert!(results.iter().all(|result| result.is_error));
    assert_eq!(
        results[0].content,
        vec![UserBlock::text("<harness>\n[error] Tool write was aborted\n</harness>")]
    );
    assert_eq!(script.started(), ["c1"]);
    assert!(live(&harness, &conversation).await.tools.is_none());
    harness.close(&ctx()).await.unwrap();
}

async fn a_crash_mid_request_converts_the_partial_and_requests_again(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let gate = Arc::new(Notify::new());
    fixture.model.push([Reply::text("half an answer").gate(&gate)]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("hi"), &ctx()).await.unwrap();
    let id = submission.id();
    partial_committed(&harness, &conversation).await;
    // A crash: every handle dropped without a close.
    let closed = harness.session().closed();
    drop((submission, conversation, harness));
    tokio::time::timeout(PATIENCE, closed)
        .await
        .expect("the old Session ends");

    fixture.model.push([Reply::text("A whole answer")]);
    let harness = fixture.open().await;
    let submission = harness.submission(id, &ctx()).await.unwrap().unwrap();
    let state = settled(&submission).await.state;
    assert!(matches!(state, SubmissionState::Done { .. }), "{state:?}");
    let conversation = chat(&harness).await;
    let transcript = entries(&conversation).await;
    assert_eq!(
        kinds(&transcript),
        ["delta.user", "delta.system", "delta.assistant", "delta.assistant"]
    );
    let converted = assistant(&transcript[2]);
    assert_eq!(
        (converted.stop_reason, converted.text().as_str()),
        (StopReason::Aborted, "half an answer")
    );
    assert_eq!(assistant(&transcript[3]).text(), "A whole answer");
    // The same generation reran its request with the same messages.
    assert_eq!(transcript[2].by_task_id, transcript[3].by_task_id);
    let task = harness
        .get_task(transcript[3].by_task_id.unwrap(), &ctx())
        .await
        .unwrap()
        .unwrap();
    assert!(task.restarts.is_none(), "a completed task keeps no restart count");
    let requests = fixture.model.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].messages, requests[1].messages);
    harness.close(&ctx()).await.unwrap();
}

async fn a_deferred_response_is_polled(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture.model.push([Reply::deferred("d1", 1_000)]);
    fixture
        .model
        .push_polls([Reply::deferred("d1", 500), Reply::text("Finally")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let start = fixture.now();
    let submission = conversation
        .submit(SubmissionDraft::input("slow"), &ctx())
        .await
        .unwrap();
    let state = settled(&submission).await.state;
    assert!(matches!(state, SubmissionState::Done { .. }), "{state:?}");
    assert!(fixture.now() - start >= 1_500);
    let transcript = entries(&conversation).await;
    // A deferred response leaves no entry; only the redeemed answer is appended.
    assert_eq!(kinds(&transcript), ["delta.user", "delta.system", "delta.assistant"]);
    assert_eq!(assistant(&transcript[2]).text(), "Finally");
    harness.close(&ctx()).await.unwrap();
}

async fn abort_cancels_a_polled_response(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture.model.push([Reply::deferred("d2", 60_000)]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation
        .submit(SubmissionDraft::input("slow"), &ctx())
        .await
        .unwrap();
    eventually(|| async {
        live(&harness, &conversation)
            .await
            .generation
            .is_some_and(|generation| generation.deferred.is_some())
    })
    .await;
    conversation.abort(AbortOptions::default(), &ctx()).await.unwrap();
    let state = settled(&submission).await.state;
    assert!(matches!(&state, SubmissionState::Unanswered { reason, .. } if reason == "aborted"));
    assert_eq!(fixture.model.cancelled(), ["d2"]);
    harness.close(&ctx()).await.unwrap();
}
