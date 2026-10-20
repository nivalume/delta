//! Tool rounds: owned tool tasks, waits, controls and the handover to the next generation.

use delta_ai::transcript::get_current_tools;
use delta_ai::types::{Message, ToolResultMessage, UserBlock};
use delta_harness::agent::{AgentChange, Change, ToolFilter};
use delta_harness::harness::Conversation;
use delta_harness::inbox::SubmissionDraft;
use delta_harness::records::{EntryRecord, SubmissionState, TaskOutcome};
use delta_harness::tool::ToolControl;

use crate::Backend;
use crate::support::fixture::{Fixture, chat, ctx, entries, idle, kinds, live, settled};
use crate::support::model::Reply;
use crate::support::tools::{ToolScript, extension, sequential_tool, tool};

cases!(
    a_tool_round_hands_over_to_the_next_generation,
    an_unoffered_call_gets_an_unavailable_result,
    a_sequential_round_runs_one_call_at_a_time,
    terminate_ends_the_run_with_the_tool_calling_answer,
    add_tools_offers_a_tool_to_the_next_request,
    a_handoff_resets_the_context,
);

fn result_of(entry: &EntryRecord) -> &ToolResultMessage {
    match entry.model.as_deref() {
        Some([Message::ToolResult(result)]) => result,
        other => panic!("a tool result: {other:?}"),
    }
}

async fn run(conversation: &Conversation, text: &str) -> SubmissionState {
    let submission = conversation.submit(SubmissionDraft::input(text), &ctx()).await.unwrap();
    settled(&submission).await.state
}

async fn a_tool_round_hands_over_to_the_next_generation(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let script = ToolScript::default();
    fixture
        .registry
        .install(extension(&script, vec![tool("read")]))
        .unwrap();
    fixture.model.push([
        Reply::calls(&[("c1", "read"), ("c2", "read")]).usage(3, 1),
        Reply::text("All read").usage(4, 1),
    ]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let SubmissionState::Done { answer, .. } = run(&conversation, "read twice").await else {
        panic!("done")
    };
    let transcript = entries(&conversation).await;
    assert_eq!(
        kinds(&transcript),
        [
            "delta.user",
            "delta.system",
            "delta.assistant",
            "delta.tool-result",
            "delta.tool-result",
            "delta.assistant"
        ]
    );
    assert_eq!(answer, Some(transcript[5].id));
    assert_eq!(result_of(&transcript[3]).content, vec![UserBlock::text("read done")]);
    // The first generation completed with the tool-calling answer and owned the tool tasks.
    let first = transcript[2].by_task_id.unwrap();
    let tool_task = transcript[3].by_task_id.unwrap();
    let record = harness.get_task(tool_task, &ctx()).await.unwrap().unwrap();
    assert_eq!((record.kind.as_str(), record.owner), ("delta.tool", Some(first)));
    let generation = harness.get_task(first, &ctx()).await.unwrap().unwrap();
    let Some(TaskOutcome::Completed { result }) = generation.state.outcome() else {
        panic!("completed")
    };
    assert_eq!(
        result.get("entry_id").and_then(delta_chord::json::Json::as_u64),
        Some(transcript[2].id.0)
    );
    // The successor ran the second request with both results after their calls.
    assert_ne!(transcript[5].by_task_id, Some(first));
    let requests = fixture.model.requests();
    let roles: Vec<&str> = requests[1].messages.iter().map(Message::role).collect();
    assert_eq!(roles, ["system", "user", "assistant", "toolResult", "toolResult"]);
    idle(&conversation).await;
    let state = live(&harness, &conversation).await;
    assert!(state.run.is_none() && state.tools.is_none());
    let usage = harness.usage(&ctx()).await.unwrap();
    assert_eq!(usage.models.values().map(|usage| usage.input).sum::<u64>(), 7);
    harness.close(&ctx()).await.unwrap();
}

async fn an_unoffered_call_gets_an_unavailable_result(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture
        .model
        .push([Reply::calls(&[("c1", "missing")]), Reply::text("Sorry")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    assert!(matches!(run(&conversation, "go").await, SubmissionState::Done { .. }));
    let transcript = entries(&conversation).await;
    let result = result_of(&transcript[3]);
    assert!(result.is_error);
    assert_eq!(
        result.content,
        vec![UserBlock::text(
            "<harness>\n[error] Tool missing is not available\n</harness>"
        )]
    );
    // No tool task was created for it.
    assert_eq!(transcript[3].by_task_id, transcript[2].by_task_id);
    let diagnostics = transcript[3]
        .data
        .as_ref()
        .and_then(|data| data.get("diagnostics"))
        .unwrap();
    assert_eq!(
        diagnostics
            .get_index(0)
            .and_then(|diagnostic| diagnostic.get("code"))
            .and_then(|code| code.as_str()),
        Some("tool_unavailable")
    );
    harness.close(&ctx()).await.unwrap();
}

async fn a_sequential_round_runs_one_call_at_a_time(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let script = ToolScript::default();
    fixture
        .registry
        .install(extension(&script, vec![tool("read"), sequential_tool("write")]))
        .unwrap();
    fixture.model.push([
        Reply::calls(&[("c1", "read"), ("c2", "write"), ("c3", "read")]),
        Reply::text("Done"),
    ]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    assert!(matches!(run(&conversation, "go").await, SubmissionState::Done { .. }));
    assert_eq!(script.started(), ["c1", "c2", "c3"]);
    let transcript = entries(&conversation).await;
    let tasks: Vec<_> = transcript
        .iter()
        .filter(|entry| entry.kind == "delta.tool-result")
        .map(|entry| entry.by_task_id.unwrap())
        .collect();
    assert_eq!(tasks.len(), 3);
    for pair in tasks.windows(2) {
        let earlier = harness.get_task(pair[0], &ctx()).await.unwrap().unwrap();
        let later = harness.get_task(pair[1], &ctx()).await.unwrap().unwrap();
        assert!(later.started_at.unwrap() >= earlier.ended_at.unwrap());
    }
    harness.close(&ctx()).await.unwrap();
}

async fn terminate_ends_the_run_with_the_tool_calling_answer(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let script = ToolScript::default();
    script.control(
        "read",
        ToolControl {
            terminate: true,
            ..ToolControl::default()
        },
    );
    fixture
        .registry
        .install(extension(&script, vec![tool("read")]))
        .unwrap();
    fixture.model.push([Reply::calls(&[("c1", "read"), ("c2", "read")])]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let SubmissionState::Done { answer, .. } = run(&conversation, "go").await else {
        panic!("done")
    };
    let transcript = entries(&conversation).await;
    assert_eq!(answer, Some(transcript[2].id));
    assert_eq!(fixture.model.requests().len(), 1);
    idle(&conversation).await;
    assert!(live(&harness, &conversation).await.run.is_none());
    harness.close(&ctx()).await.unwrap();
}

async fn add_tools_offers_a_tool_to_the_next_request(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let script = ToolScript::default();
    script.control(
        "loader",
        ToolControl {
            add_tools: vec!["write".to_owned()],
            ..ToolControl::default()
        },
    );
    fixture
        .registry
        .install(extension(&script, vec![tool("loader"), tool("write")]))
        .unwrap();
    fixture
        .model
        .push([Reply::calls(&[("c1", "loader")]), Reply::text("Loaded")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let change = AgentChange {
        tools: Change::Set(ToolFilter::Exactly(vec!["loader".to_owned()])),
        ..AgentChange::default()
    };
    conversation.configure(change, &ctx()).await.unwrap();
    run(&conversation, "load").await;
    let requests = fixture.model.requests();
    let offered = |index: usize| -> Vec<String> {
        get_current_tools(&requests[index].messages)
            .into_iter()
            .map(|tool| tool.name)
            .collect()
    };
    assert_eq!(offered(0), ["loader"]);
    assert_eq!(offered(1), ["loader", "write"]);
    harness.close(&ctx()).await.unwrap();
}

async fn a_handoff_resets_the_context(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let script = ToolScript::default();
    script.control(
        "handoff",
        ToolControl {
            handoff: Some("Continue with the plan".to_owned()),
            ..ToolControl::default()
        },
    );
    fixture
        .registry
        .install(extension(&script, vec![tool("handoff")]))
        .unwrap();
    fixture
        .model
        .push([Reply::calls(&[("c1", "handoff")]), Reply::text("Fresh start")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let SubmissionState::Done { answer, .. } = run(&conversation, "go").await else {
        panic!("done")
    };
    let transcript = entries(&conversation).await;
    assert_eq!(answer, Some(transcript[2].id));
    let reset = transcript.iter().find(|entry| entry.kind == "delta.reset").unwrap();
    assert_eq!(reset.head, Some(reset.id));
    // The next run's context starts at the reset, with a new baseline.
    run(&conversation, "next").await;
    let requests = fixture.model.requests();
    let texts: Vec<String> = requests[1]
        .messages
        .iter()
        .filter_map(|message| match message {
            Message::User(user) => Some(user.content.text()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, ["Continue with the plan", "next"]);
    assert!(matches!(requests[1].messages[0], Message::System(_)));
    harness.close(&ctx()).await.unwrap();
}
