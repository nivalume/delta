//! Generation hooks: the `before_request` chain, observers, the `on_yield` continuation and
//! reported hook failures.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use delta_ai::types::{AssistantMessage, Message, StopReason, UserContent};
use delta_chord::context::Context;
use delta_harness::inbox::SubmissionDraft;
use delta_harness::records::{EntryId, SubmissionState};
use delta_harness::registry::{Extension, GenerationHooks, HookApi, HookResult, RequestMessages};
use delta_harness::session::ReportInfo;
use futures::future::BoxFuture;

use crate::Backend;
use crate::support::fixture::{Fixture, chat, ctx, entries, kinds, settled};
use crate::support::model::Reply;
use crate::support::tools::{ToolScript, extension, tool};

cases!(hooks_shape_requests_observe_responses_and_continue_runs);

#[derive(Default)]
struct Recorder {
    responses: Mutex<Vec<StopReason>>,
    rounds: Mutex<Vec<(EntryId, Vec<EntryId>)>>,
    continued: AtomicBool,
}

impl GenerationHooks for Recorder {
    fn before_request<'a>(
        &'a self,
        request: &'a RequestMessages,
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<Vec<Message>>> {
        let mut messages = request.messages.clone();
        messages.push(Message::user("injected"));
        Box::pin(async move { Ok(Some(messages)) })
    }

    fn after_response<'a>(
        &'a self,
        message: &'a AssistantMessage,
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<()>> {
        self.responses.lock().unwrap().push(message.stop_reason);
        Box::pin(async { Ok(None) })
    }

    fn on_yield<'a>(
        &'a self,
        _answer: &'a AssistantMessage,
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<UserContent>> {
        let first = !self.continued.swap(true, Ordering::SeqCst);
        Box::pin(async move { Ok(first.then(|| UserContent::from("keep going"))) })
    }

    fn after_tools<'a>(
        &'a self,
        assistant: EntryId,
        results: &'a [EntryId],
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<()>> {
        self.rounds.lock().unwrap().push((assistant, results.to_vec()));
        Box::pin(async { Ok(None) })
    }
}

/// Hooks that fail; the chain goes on without them.
struct Failing;

impl GenerationHooks for Failing {
    fn before_request<'a>(
        &'a self,
        _request: &'a RequestMessages,
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<Vec<Message>>> {
        Box::pin(async { Err("hook failed".into()) })
    }
}

async fn hooks_shape_requests_observe_responses_and_continue_runs(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let script = ToolScript::default();
    fixture
        .registry
        .install(extension(&script, vec![tool("read")]))
        .unwrap();
    let recorder = Arc::new(Recorder::default());
    let hooks: Arc<dyn GenerationHooks> = recorder.clone();
    fixture
        .registry
        .install(Extension::new("recorder").with_hooks("delta.generation", hooks))
        .unwrap();
    let failing: Arc<dyn GenerationHooks> = Arc::new(Failing);
    fixture
        .registry
        .install(Extension::new("failing").with_hooks("delta.generation", failing))
        .unwrap();
    fixture.model.push([
        Reply::calls(&[("c1", "read")]),
        Reply::text("first answer"),
        Reply::text("second answer"),
    ]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("hi"), &ctx()).await.unwrap();
    let SubmissionState::Done { answer, .. } = settled(&submission).await.state else {
        panic!("done")
    };
    // `before_request` replaced every request's messages; the stored transcript has none of it.
    let requests = fixture.model.requests();
    assert_eq!(requests.len(), 3);
    for request in &requests {
        let Some(Message::User(last)) = request.messages.last() else {
            panic!("a user message last")
        };
        assert_eq!(last.content.text(), "injected");
    }
    let transcript = entries(&conversation).await;
    assert_eq!(
        kinds(&transcript),
        [
            "delta.user",
            "delta.system",
            "delta.assistant",
            "delta.tool-result",
            "delta.assistant",
            "delta.user",
            "delta.assistant"
        ]
    );
    // The continuation kept the run's input open until the second answer.
    assert_eq!(answer, Some(transcript[6].id));
    let Some([Message::User(continuation)]) = transcript[5].model.as_deref() else {
        panic!("the continuation")
    };
    assert_eq!(continuation.content.text(), "keep going");
    assert_eq!(
        *recorder.responses.lock().unwrap(),
        [StopReason::ToolUse, StopReason::Stop, StopReason::Stop]
    );
    assert_eq!(
        *recorder.rounds.lock().unwrap(),
        [(transcript[2].id, vec![transcript[3].id])]
    );
    let failures = fixture
        .reported()
        .into_iter()
        .filter(|(error, info)| {
            error == "hook failed"
                && matches!(info, ReportInfo::ExtensionError { extension: Some(name), .. } if name == "failing")
        })
        .count();
    assert_eq!(failures, 3);
    harness.close(&ctx()).await.unwrap();
}
