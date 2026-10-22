//! Generation hooks: the `before_request` chain, observers, the `on_yield` continuation and
//! reported hook failures.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use delta_ai::types::{AssistantMessage, Message, StopReason, UserContent};
use delta_chord::context::Context;
use delta_harness::inbox::SubmissionDraft;
use delta_harness::records::{EntryId, SubmissionState};
use delta_harness::registry::{
    Extension, GenerationHooks, HookApi, HookResult, InputDecision, RequestMessages, RunInput, Yielded,
};
use delta_harness::session::ReportInfo;
use futures::future::BoxFuture;

use crate::Backend;
use crate::support::fixture::{Fixture, chat, ctx, entries, kinds, settled};
use crate::support::model::Reply;
use crate::support::tools::{ToolScript, extension, tool};

cases!(
    hooks_shape_requests_observe_responses_and_continue_runs,
    on_input_blocks_a_run,
    on_input_context_is_added_once_per_run,
    continuations_are_counted_and_capped,
);

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
        yielded: &'a Yielded,
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<UserContent>> {
        assert_eq!(yielded.continuations, u32::from(self.continued.load(Ordering::SeqCst)));
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
    assert_eq!(transcript[5].data, Some(delta_chord::json!({ "continuation": true })));
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

/// `on_input` hooks: block input containing "forbidden", add context otherwise; counts calls.
#[derive(Default)]
struct Gate {
    calls: Mutex<Vec<Vec<String>>>,
}

impl GenerationHooks for Gate {
    fn on_input<'a>(
        &'a self,
        input: &'a RunInput,
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<InputDecision>> {
        let texts: Vec<String> = input
            .entries
            .iter()
            .filter_map(|entry| match entry.model.as_deref() {
                Some([Message::User(user)]) => Some(user.content.text()),
                _ => None,
            })
            .collect();
        self.calls.lock().unwrap().push(texts.clone());
        Box::pin(async move {
            if texts.iter().any(|text| text.contains("forbidden")) {
                return Ok(Some(InputDecision::Block {
                    reason: "no forbidden words".to_owned(),
                }));
            }
            Ok(Some(InputDecision::Context(format!("checked {}", texts.join(",")))))
        })
    }
}

fn gate(fixture: &Fixture) -> Arc<Gate> {
    let gate = Arc::new(Gate::default());
    let hooks: Arc<dyn GenerationHooks> = gate.clone();
    fixture
        .registry
        .install(Extension::new("gate").with_hooks("delta.generation", hooks))
        .unwrap();
    gate
}

async fn on_input_blocks_a_run(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    gate(&fixture);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation
        .submit(SubmissionDraft::input("say something forbidden"), &ctx())
        .await
        .unwrap();
    let SubmissionState::Unanswered { reason, detail, .. } = settled(&submission).await.state else {
        panic!("unanswered")
    };
    assert_eq!(reason, "blocked");
    assert_eq!(detail, Some(delta_chord::json!("no forbidden words")));
    assert!(fixture.model.requests().is_empty(), "no request was made");
    assert_eq!(kinds(&entries(&conversation).await), ["delta.user"]);
    harness.close(&ctx()).await.unwrap();
}

async fn on_input_context_is_added_once_per_run(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let script = ToolScript::default();
    fixture
        .registry
        .install(extension(&script, vec![tool("read")]))
        .unwrap();
    let gate = gate(&fixture);
    // A tool round: the successor generation does not run the hooks again.
    fixture
        .model
        .push([Reply::calls(&[("c1", "read")]), Reply::text("answer")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("hi"), &ctx()).await.unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    assert_eq!(*gate.calls.lock().unwrap(), [vec!["hi".to_owned()]]);
    let transcript = entries(&conversation).await;
    assert_eq!(&kinds(&transcript)[..2], ["delta.user", "delta.context"]);
    let Some([Message::System(context)]) = transcript[1].model.as_deref() else {
        panic!("a system message")
    };
    assert_eq!(context.content.text(), "checked hi");
    // The model saw the context.
    let first = &fixture.model.requests()[0];
    assert!(first.messages.iter().any(|message| matches!(
        message,
        Message::System(system) if system.content.text() == "checked hi"
    )));
    harness.close(&ctx()).await.unwrap();
}

/// Continues every answer, recording the counts it saw.
#[derive(Default)]
struct Persistent {
    seen: Mutex<Vec<u32>>,
}

impl GenerationHooks for Persistent {
    fn on_yield<'a>(
        &'a self,
        yielded: &'a Yielded,
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<UserContent>> {
        self.seen.lock().unwrap().push(yielded.continuations);
        Box::pin(async { Ok(Some(UserContent::from("again"))) })
    }
}

async fn continuations_are_counted_and_capped(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture.settings(|settings| settings.max_continuations = Some(2));
    let persistent = Arc::new(Persistent::default());
    let hooks: Arc<dyn GenerationHooks> = persistent.clone();
    fixture
        .registry
        .install(Extension::new("persistent").with_hooks("delta.generation", hooks))
        .unwrap();
    fixture
        .model
        .push([Reply::text("one"), Reply::text("two"), Reply::text("three")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    assert_eq!(*persistent.seen.lock().unwrap(), [0, 1, 2]);
    assert_eq!(fixture.model.requests().len(), 3);
    assert!(
        fixture
            .reported()
            .iter()
            .any(|(error, _)| error == "The run took 2 continuations; the next one is ignored")
    );
    harness.close(&ctx()).await.unwrap();
}
