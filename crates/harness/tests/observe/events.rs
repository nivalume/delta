//! Agent events: a run with tools, a streamed partial, tool progress, approvals,
//! compactions, the snapshot and the backlog.

use std::sync::Arc;
use std::time::Duration;

use delta_ai::types::{Message, Tool};
use delta_harness::agent::{AgentChange, Change, PartialProgressPolicy};
use delta_harness::approval::{AskAll, ReviewDecision};
use delta_harness::harness::{Conversation, Harness};
use delta_harness::inbox::SubmissionDraft;
use delta_harness::observe::{AgentEvent, AgentEventStream, EVENT_BACKLOG, MessageChange, OutputChange};
use delta_harness::registry::Extension;
use delta_harness::tool::{FnHarnessTool, ToolExecutionResult};

use crate::Backend;
use crate::support::fixture::{Fixture, PATIENCE, chat, ctx, entries, idle};
use crate::support::model::Reply;
use crate::support::tools::{ToolScript, extension, tool};

cases!(
    a_run_with_tools_yields_upstream_events,
    a_streamed_answer_starts_once_and_updates,
    tool_progress_becomes_updates,
    approvals_and_compactions_have_events,
    a_stream_starts_at_its_snapshot,
);

/// Every event until a batch holds `run_end`.
async fn until_run_end(stream: &mut AgentEventStream) -> Vec<AgentEvent> {
    let mut events = Vec::new();
    loop {
        let batch = tokio::time::timeout(PATIENCE, stream.next())
            .await
            .expect("a batch")
            .expect("the stream is open");
        let done = batch.iter().any(|event| matches!(event, AgentEvent::RunEnd { .. }));
        events.extend(batch);
        if done {
            return events;
        }
    }
}

fn kinds(events: &[AgentEvent]) -> Vec<&'static str> {
    events.iter().map(AgentEvent::kind).collect()
}

fn count(events: &[AgentEvent], kind: &str) -> usize {
    events.iter().filter(|event| event.kind() == kind).count()
}

async fn submit(conversation: &Conversation, text: &str) {
    conversation.submit(SubmissionDraft::input(text), &ctx()).await.unwrap();
}

async fn a_run_with_tools_yields_upstream_events(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let script = ToolScript::default();
    fixture
        .registry
        .install(extension(&script, vec![tool("read")]))
        .unwrap();
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let mut stream = conversation.events(&ctx()).await.unwrap();
    assert!(stream.snapshot().entries.is_empty());
    fixture
        .model
        .push([Reply::calls(&[("c1", "read"), ("c2", "read")]), Reply::text("All read")]);
    submit(&conversation, "read twice").await;
    let events = until_run_end(&mut stream).await;
    let kinds = kinds(&events);
    assert_eq!(kinds.first(), Some(&"message_start"), "{kinds:?}");
    assert_eq!(count(&events, "run_start"), 1);
    assert_eq!(count(&events, "run_end"), 1);
    assert_eq!(count(&events, "turn_start"), 2, "{kinds:?}");
    assert_eq!(count(&events, "turn_end"), 2, "{kinds:?}");
    assert_eq!(count(&events, "tool_execution_start"), 2, "{kinds:?}");
    let run_start = kinds.iter().position(|kind| *kind == "run_start").unwrap();
    assert_eq!(kinds[run_start + 1], "turn_start");
    // Each tool end carries its result entry and directly precedes its message_start.
    let mut ended = Vec::new();
    for (index, event) in events.iter().enumerate() {
        if let AgentEvent::ToolExecutionEnd { call, entry, .. } = event {
            assert!(entry.is_some(), "{call:?}");
            assert_eq!(kinds[index + 1], "message_start");
            assert!(matches!(
                &events[index + 1],
                AgentEvent::MessageStart { message: Message::ToolResult(result), .. } if result.tool_call_id == call.tool_call_id
            ));
            ended.push(call.tool_call_id.clone());
        }
    }
    ended.sort();
    assert_eq!(ended, ["c1", "c2"]);
    // Every appended model entry ends once; the transcript holds them all.
    let ends: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageEnd { entry } => Some(entry.id),
            _ => None,
        })
        .collect();
    let transcript: Vec<_> = entries(&conversation)
        .await
        .into_iter()
        .filter(|entry| entry.model.is_some())
        .map(|entry| entry.id)
        .collect();
    assert_eq!(ends, transcript);
    assert!(count(&events, "submission") >= 2, "placed, then done");
    assert!(count(&events, "usage_changed") >= 1);
    // Events serialize with their upstream tags.
    let json = serde_json::to_value(&events[run_start]).unwrap();
    assert_eq!(json["type"], "run_start");
    let back: AgentEvent = serde_json::from_value(json).unwrap();
    assert_eq!(back, events[run_start]);
    harness.close(&ctx()).await.unwrap();
}

async fn a_streamed_answer_starts_once_and_updates(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture.settings(|settings| {
        settings.progress = Some(PartialProgressPolicy {
            partial_interval: Some(Duration::from_millis(1)),
            ..PartialProgressPolicy::default()
        });
    });
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let mut stream = conversation.events(&ctx()).await.unwrap();
    fixture
        .model
        .push([Reply::text("one two three four five six").step(Duration::from_millis(15))]);
    submit(&conversation, "count").await;
    let events = until_run_end(&mut stream).await;
    let partial_start = events
        .iter()
        .position(|event| {
            matches!(
                event,
                AgentEvent::MessageStart {
                    message: Message::Assistant(_),
                    ..
                }
            )
        })
        .expect("the partial starts");
    let deltas: String = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageUpdate { changes, .. } => Some(changes),
            _ => None,
        })
        .flatten()
        .filter_map(|change| match change {
            MessageChange::TextDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert!(!deltas.is_empty(), "{:?}", kinds(&events));
    // The streamed answer's entry ends without a second start.
    let starts = events[partial_start + 1..]
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::MessageStart {
                    message: Message::Assistant(_),
                    ..
                }
            )
        })
        .count();
    assert_eq!(starts, 0, "{:?}", kinds(&events));
    // The partial names the generation task that appends its entry.
    let AgentEvent::MessageStart { task_id, .. } = &events[partial_start] else {
        unreachable!()
    };
    assert!(task_id.is_some());
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::MessageEnd { entry } if entry.kind == "delta.assistant" && entry.by_task_id == *task_id
    )));
    harness.close(&ctx()).await.unwrap();
}

async fn tool_progress_becomes_updates(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let tool = FnHarnessTool::new(
        Tool::new("build", "Builds", serde_json::json!({ "type": "object" })),
        |_arguments, api, ctx| {
            Box::pin(async move {
                api.output_str("compiling\n");
                api.details(serde_json::json!({ "step": 1 }), ctx).await?;
                api.output_str("linking\n");
                api.details(serde_json::json!({ "step": 2 }), ctx).await?;
                Ok(ToolExecutionResult::text("built"))
            })
        },
    );
    fixture
        .registry
        .install(Extension::new("tools").with_tool(Arc::new(tool)))
        .unwrap();
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let mut stream = conversation.events(&ctx()).await.unwrap();
    fixture
        .model
        .push([Reply::calls(&[("c1", "build")]), Reply::text("Built")]);
    submit(&conversation, "build").await;
    let events = until_run_end(&mut stream).await;
    let mut output = String::new();
    let mut steps = Vec::new();
    for event in &events {
        if let AgentEvent::ToolExecutionUpdate {
            output: change,
            details,
            call,
            ..
        } = event
        {
            assert_eq!(call.tool_call_id, "c1");
            match change {
                Some(OutputChange::Delta { trim_start, append }) => {
                    assert_eq!(*trim_start, None);
                    output.push_str(append.as_deref().unwrap_or_default());
                }
                Some(OutputChange::Set { output: whole }) => output.clone_from(whole),
                None => {}
            }
            if let Some(details) = details {
                steps.push(details.get("step").and_then(delta_chord::json::Json::as_u64));
            }
        }
    }
    assert_eq!(output, "compiling\nlinking\n");
    assert_eq!(steps, [Some(1), Some(2)]);
    let start = events
        .iter()
        .position(|event| matches!(event, AgentEvent::ToolExecutionStart { .. }))
        .unwrap();
    let first_update = events
        .iter()
        .position(|event| matches!(event, AgentEvent::ToolExecutionUpdate { .. }))
        .unwrap();
    assert!(start < first_update);
    harness.close(&ctx()).await.unwrap();
}

async fn approvals_and_compactions_have_events(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture
        .registry
        .install(Extension::new("tools").with_tool(tool("read")))
        .unwrap();
    fixture.settings(|settings| {
        settings.compaction = Some(delta_harness::agent::CompactionSettings {
            policy: delta_harness::agent::PartialCompactionPolicy {
                keep_recent_tokens: Some(10),
                ..Default::default()
            },
            ..Default::default()
        });
    });
    let mut options = fixture.options();
    options.approval = Arc::new(AskAll);
    let harness = Harness::open(fixture.storage.clone(), options, &ctx()).await.unwrap();
    let conversation = chat(&harness).await;
    let mut stream = conversation.events(&ctx()).await.unwrap();
    fixture
        .model
        .push([Reply::calls(&[("c1", "read")]), Reply::text(&"Read it. ".repeat(30))]);
    submit(&conversation, "read").await;
    let requested = loop {
        let batch = tokio::time::timeout(PATIENCE, stream.next()).await.unwrap().unwrap();
        if let Some(requests) = batch.into_iter().find_map(|event| match event {
            AgentEvent::ApprovalUpdate { requests } => Some(requests),
            _ => None,
        }) {
            break requests;
        }
    };
    assert_eq!(requested.len(), 1);
    conversation
        .decide(requested[0].id, ReviewDecision::Approved, &ctx())
        .await
        .unwrap();
    let events = until_run_end(&mut stream).await;
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ApprovalUpdate { requests } if requests.is_empty()))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::EntryAppended { entry } if entry.kind == "delta.approval"))
    );

    fixture.model.push([Reply::text("Second")]);
    submit(&conversation, &"Second question. ".repeat(20)).await;
    until_run_end(&mut stream).await;
    fixture.model.push_summaries([Reply::text("Summary.")]);
    let task = conversation.compact(None, &ctx()).await.unwrap();
    let mut seen = Vec::new();
    while !seen.contains(&"compaction_end") {
        let batch = tokio::time::timeout(PATIENCE, stream.next()).await.unwrap().unwrap();
        for event in &batch {
            if let AgentEvent::CompactionStart { task_id, .. } | AgentEvent::CompactionEnd { task_id, .. } = event {
                assert_eq!(*task_id, task);
            }
        }
        seen.extend(kinds(&batch));
    }
    let start = seen.iter().position(|kind| *kind == "compaction_start").unwrap();
    assert!(start < seen.iter().position(|kind| *kind == "compaction_end").unwrap());
    harness.close(&ctx()).await.unwrap();
}

async fn a_stream_starts_at_its_snapshot(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    fixture.model.push([Reply::text("Hi")]);
    submit(&conversation, "Hello").await;
    idle(&conversation).await;
    let transcript = entries(&conversation).await;
    crate::support::fixture::eventually(|| async {
        conversation.view_state(&ctx()).await.unwrap().view().unwrap().entries == transcript
    })
    .await;
    let mut stream = conversation.events(&ctx()).await.unwrap();
    assert_eq!(stream.snapshot().entries, transcript);
    assert!(stream.snapshot().run.is_none());
    assert!(stream.try_next().is_none());

    // A consumer that falls behind gets one snapshot of the newest state.
    for index in 0..=EVENT_BACKLOG {
        let change = AgentChange {
            instructions: Change::Set(format!("Instructions {index}")),
            ..AgentChange::default()
        };
        conversation.configure(change, &ctx()).await.unwrap();
    }
    let batch = tokio::time::timeout(PATIENCE, stream.next()).await.unwrap().unwrap();
    let [AgentEvent::Snapshot(snapshot)] = batch.as_slice() else {
        panic!("one snapshot: {:?}", kinds(&batch))
    };
    assert_eq!(
        snapshot.agent.instructions.as_deref(),
        Some(format!("Instructions {EVENT_BACKLOG}").as_str())
    );
    assert!(stream.try_next().is_none());
    harness.close(&ctx()).await.unwrap();
}
