//! Tests of the stateful `Agent`, ported from upstream `test/agent.test.ts`.

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

mod common;

use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use tau_agent::{
    Agent, AgentError, AgentEvent, AgentMessage, AgentOptions, AgentTool, AgentToolResult, AgentTurnDecision,
    QueueMode, StreamFn, ToolUpdates,
};
use tau_ai::transcript::get_current_system_message;
use tau_ai::types::ToolReference;
use tau_ai::{
    AssistantMessageEvent, AssistantMessageEventStream, DoneReason, Message, ModelThinkingLevel, StopReason,
    SystemMessage, Tool, ToolResultMessage, UserBlock,
};

use common::*;

fn new_agent(stream_fn: StreamFn) -> Agent {
    Agent::new(AgentOptions::new(stream_fn))
}

fn agent_with(stream_fn: StreamFn, configure: impl FnOnce(&mut AgentOptions)) -> Agent {
    let mut options = AgentOptions::new(stream_fn);
    configure(&mut options);
    Agent::new(options)
}

fn unused_stream_fn() -> StreamFn {
    stream_fn(|_, _, _, _| panic!("Unexpected stream call"))
}

/// Answers every request with "done" and records the plain-string user texts it saw.
fn recording(requests: Shared<Vec<Vec<String>>>) -> StreamFn {
    stream_fn(move |_, _, context, _| {
        lock(&requests).push(user_texts(context));
        respond(assistant_text("done"))
    })
}

fn tool_use_message(calls: Vec<(&str, &str)>) -> tau_ai::AssistantMessage {
    assistant(
        calls
            .into_iter()
            .map(|(id, name)| tool_call(id, name, json!({})))
            .collect(),
        StopReason::ToolUse,
    )
}

/// Records every event a listener receives.
fn record_events(agent: &Agent) -> Shared<Vec<Arc<AgentEvent>>> {
    let events = shared(Vec::new());
    let recorded = events.clone();
    agent.subscribe(move |event, _| {
        let recorded = recorded.clone();
        async move { lock(&recorded).push(event) }
    });
    events
}

fn spawn_prompt(agent: &Agent, text: &'static str) -> tokio::task::JoinHandle<Result<(), AgentError>> {
    let agent = agent.clone();
    tokio::spawn(async move { agent.prompt_text(text, Vec::new()).await })
}

async fn tick() {
    tokio::time::sleep(Duration::from_millis(10)).await;
}

#[test]
fn creates_an_agent_with_default_state() {
    let agent = new_agent(unused_stream_fn());

    agent.with_state(|state| {
        assert_eq!(state.model.id, "unknown");
        assert_eq!(state.thinking_level, ModelThinkingLevel::Off);
        assert!(state.tools.is_empty());
        assert!(state.messages.is_empty());
        assert!(!state.is_streaming);
        assert!(state.streaming_message.is_none());
        assert!(state.pending_tool_calls.is_empty());
        assert_eq!(state.error_message, None);
    });
}

#[test]
fn creates_an_agent_with_custom_initial_state() {
    let custom_model = tau_ai::Model {
        id: "gpt-4o-mini".into(),
        ..model()
    };
    let agent = agent_with(unused_stream_fn(), |options| {
        options.initial_state.system_prompt = Some("You are a helpful assistant.".into());
        options.initial_state.model = Some(custom_model.clone());
        options.initial_state.thinking_level = ModelThinkingLevel::Low;
    });

    let expected: AgentMessage = SystemMessage {
        content: "You are a helpful assistant.".into(),
        timestamp: 0,
        ..SystemMessage::default()
    }
    .into();
    assert_eq!(agent.messages(), [expected]);
    assert_eq!(agent.model(), custom_model);
    agent.with_state(|state| assert_eq!(state.thinking_level, ModelThinkingLevel::Low));
}

fn system(message: &AgentMessage) -> &SystemMessage {
    match message {
        AgentMessage::Llm(Message::System(system)) => system,
        other => panic!("expected a system message, got {other:?}"),
    }
}

fn tool_names(tools: Option<&Vec<Tool>>) -> Vec<String> {
    tools.into_iter().flatten().map(|tool| tool.name.clone()).collect()
}

fn echo_input_tool() -> Arc<dyn AgentTool> {
    Arc::new(tau_agent::FnTool::new(
        Tool::new("echo", "Echo input", empty_schema()),
        |_, _| async { Ok(AgentToolResult::text("echo")) },
    ))
}

#[test]
fn converts_initial_prompt_and_tools_into_transcript_state() {
    let agent = agent_with(unused_stream_fn(), |options| {
        options.initial_state.system_prompt = Some("You are helpful.".into());
        options.initial_state.tools = vec![echo_input_tool()];
    });

    let messages = agent.messages();
    let initial = system(&messages[0]);
    assert_eq!(initial.content.text(), "You are helpful.");
    assert_eq!(tool_names(initial.tools_added.as_ref()), ["echo"]);
}

#[tokio::test]
async fn declares_tool_loadout_changes_to_the_model_before_the_next_request() {
    let requests = shared(Vec::<Vec<String>>::new());
    let recorded = requests.clone();
    let agent = agent_with(
        stream_fn(move |_, _, context, _| {
            let declarations = context
                .messages()
                .iter()
                .filter_map(|message| match message {
                    Message::System(system) => Some(system),
                    _ => None,
                })
                .flat_map(|system| {
                    let added = tool_names(system.tools_added.as_ref()).join(",");
                    let removed: Vec<String> = system
                        .tools_removed
                        .iter()
                        .flatten()
                        .map(|tool| tool.name.clone())
                        .collect();
                    [format!("+{added}"), format!("-{}", removed.join(","))]
                })
                .collect();
            lock(&recorded).push(declarations);
            respond(assistant_text("done"))
        }),
        |options| {
            options.initial_state.system_prompt = Some("You are helpful.".into());
            options.initial_state.tools = vec![noop_tool("first")];
        },
    );

    agent.prompt_text("one", Vec::new()).await.unwrap();
    agent.set_tools(vec![noop_tool("second")]);
    agent.prompt_text("two", Vec::new()).await.unwrap();
    agent.prompt_text("three", Vec::new()).await.unwrap();

    assert_eq!(
        *lock(&requests),
        [
            vec!["+first", "-"],
            vec!["+first", "-", "+second", "-first"],
            vec!["+first", "-", "+second", "-first"],
        ]
    );
    let messages = agent.messages();
    let update = messages
        .iter()
        .find(|message| message.is_system() && system(message).tools_removed.is_some())
        .map(system)
        .unwrap();
    assert_eq!(
        update,
        &SystemMessage {
            content: "".into(),
            tools_added: Some(vec![Tool::new("second", "second tool", empty_schema())]),
            tools_removed: Some(vec![ToolReference { name: "first".into() }]),
            timestamp: update.timestamp,
            ..SystemMessage::default()
        }
    );
}

fn system_from_json(value: serde_json::Value) -> AgentMessage {
    serde_json::from_value(value).unwrap()
}

#[tokio::test]
async fn merges_tool_changes_into_a_pending_system_message() {
    let system_counts = shared(Vec::new());
    let recorded = system_counts.clone();
    let agent = agent_with(
        stream_fn(move |_, _, context, _| {
            let count = context
                .messages()
                .iter()
                .filter(|message| matches!(message, Message::System(_)))
                .count();
            lock(&recorded).push(count);
            respond(assistant_text("done"))
        }),
        |options| options.initial_state.system_prompt = Some("You are helpful.".into()),
    );

    agent.set_tools(vec![echo_input_tool()]);
    agent
        .prompt(vec![
            system_from_json(json!({
                "role": "system", "content": "", "sections": {"skills": "<skills>x</skills>"}, "timestamp": 1
            })),
            tau_ai::UserMessage {
                content: "hi".into(),
                timestamp: 2,
            }
            .into(),
        ])
        .await
        .unwrap();

    assert_eq!(*lock(&system_counts), [2]);
    assert_eq!(
        serde_json::to_value(&agent.messages()[1]).unwrap(),
        json!({
            "role": "system",
            "content": "",
            "sections": {"skills": "<skills>x</skills>"},
            "toolsAdded": [{"name": "echo", "description": "Echo input", "parameters": empty_schema()}],
            "timestamp": 1,
        })
    );
}

#[tokio::test]
async fn rewrites_pending_tool_declarations_to_match_the_executable_set() {
    let agent = agent_with(scripted(vec![assistant_text("done")]), |options| {
        options.initial_state.system_prompt = Some("You are helpful.".into());
        options.initial_state.tools = vec![noop_tool("first")];
    });

    // The pending message claims to add `second` and remove `first`, but the executable set
    // still has `first` and lacks `second`: the executable set wins.
    agent
        .prompt(vec![
            system_from_json(json!({
                "role": "system",
                "content": "",
                "sections": {"note": "<note>x</note>"},
                "toolsAdded": [{"name": "second", "description": "second tool", "parameters": empty_schema()}],
                "toolsRemoved": [{"name": "first"}],
                "timestamp": 1,
            })),
            tau_ai::UserMessage {
                content: "hi".into(),
                timestamp: 2,
            }
            .into(),
        ])
        .await
        .unwrap();

    let messages = agent.messages();
    assert_eq!(
        serde_json::to_value(&messages[1]).unwrap(),
        json!({"role": "system", "content": "", "sections": {"note": "<note>x</note>"}, "timestamp": 1})
    );
    let llm: Vec<Message> = messages.iter().filter_map(|m| m.as_llm().cloned()).collect();
    let current = get_current_system_message(&llm).unwrap();
    assert_eq!(tool_names(current.tools_added.as_ref()), ["first"]);
}

#[test]
fn restores_the_transcript_baseline_when_reset() {
    let agent = agent_with(unused_stream_fn(), |options| {
        options.initial_state.system_prompt = Some("You are helpful.".into());
        options.initial_state.tools = vec![echo_input_tool()];
        options.initial_state.messages = vec![
            tau_ai::UserMessage {
                content: "old".into(),
                timestamp: 1,
            }
            .into(),
        ];
    });

    agent.reset().unwrap();

    let messages = agent.messages();
    assert_eq!(messages.len(), 1);
    let initial = system(&messages[0]);
    assert_eq!(initial.content.text(), "You are helpful.");
    assert_eq!(tool_names(initial.tools_added.as_ref()), ["echo"]);
}

#[test]
fn subscribes_to_events() {
    let agent = new_agent(unused_stream_fn());
    let count = Arc::new(AtomicUsize::new(0));
    let counter = count.clone();
    let subscription = agent.subscribe(move |_, _| {
        counter.fetch_add(1, Ordering::SeqCst);
        async {}
    });

    // No initial event on subscribe.
    assert_eq!(count.load(Ordering::SeqCst), 0);
    // State setters don't emit events.
    agent.set_thinking_level(ModelThinkingLevel::Low);
    assert_eq!(count.load(Ordering::SeqCst), 0);
    agent.with_state(|state| assert_eq!(state.thinking_level, ModelThinkingLevel::Low));

    agent.unsubscribe(subscription);
    agent.set_thinking_level(ModelThinkingLevel::High);
    assert_eq!(count.load(Ordering::SeqCst), 0);
}

// Upstream throws from the stream function; a tau stream function reports failures in its
// stream, so the response fails before generation starts.
#[tokio::test]
async fn emits_full_lifecycle_events_for_failed_runs() {
    let agent = new_agent(stream_fn(|_, _, _, _| {
        respond(failed(StopReason::Error, "provider exploded"))
    }));
    let events = record_events(&agent);

    agent.prompt_text("hello", Vec::new()).await.unwrap();

    let kinds: Vec<&str> = lock(&events).iter().map(|event| event.kind()).collect();
    assert_eq!(
        kinds,
        [
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end",
        ]
    );
    let messages = agent.messages();
    let last = messages.last().unwrap().as_assistant().unwrap();
    assert_eq!(last.stop_reason, StopReason::Error);
    assert_eq!(last.error_message.as_deref(), Some("provider exploded"));
    assert_eq!(agent.error_message().as_deref(), Some("provider exploded"));
}

#[tokio::test]
async fn awaits_async_subscribers_before_prompt_resolves() {
    let barrier = Gate::default();
    let agent = new_agent(scripted(vec![assistant_text("ok")]));
    let listener_finished = Arc::new(AtomicBool::new(false));
    let (gate, finished) = (barrier.clone(), listener_finished.clone());
    agent.subscribe(move |event, _| {
        let (gate, finished) = (gate.clone(), finished.clone());
        async move {
            if event.kind() == "agent_end" {
                gate.wait().await;
                finished.store(true, Ordering::SeqCst);
            }
        }
    });

    let prompt = spawn_prompt(&agent, "hello");
    tick().await;
    assert!(!prompt.is_finished());
    assert!(!listener_finished.load(Ordering::SeqCst));
    assert!(agent.is_streaming());

    barrier.open();
    prompt.await.unwrap().unwrap();

    assert!(listener_finished.load(Ordering::SeqCst));
    assert!(!agent.is_streaming());
}

#[tokio::test]
async fn wait_for_idle_waits_for_async_subscribers() {
    let barrier = Gate::default();
    let agent = new_agent(scripted(vec![assistant_text("ok")]));
    let gate = barrier.clone();
    agent.subscribe(move |event, _| {
        let gate = gate.clone();
        async move {
            if let AgentEvent::MessageEnd { message } = &*event
                && message.role() == "assistant"
            {
                gate.wait().await;
            }
        }
    });

    let prompt = spawn_prompt(&agent, "hello");
    tokio::task::yield_now().await;
    let idle_agent = agent.clone();
    let idle = tokio::spawn(async move { idle_agent.wait_for_idle().await });
    tick().await;
    assert!(!idle.is_finished());
    assert!(agent.is_streaming());

    barrier.open();
    prompt.await.unwrap().unwrap();
    idle.await.unwrap();
    assert!(!agent.is_streaming());
}

#[tokio::test]
async fn passes_the_active_cancellation_token_to_subscribers() {
    let agent = new_agent(abortable());
    let received = shared(None::<CancellationToken>);
    let recorded = received.clone();
    agent.subscribe(move |event, cancel| {
        if event.kind() == "agent_start" {
            *lock(&recorded) = Some(cancel);
        }
        async {}
    });

    let prompt = spawn_prompt(&agent, "hello");
    tick().await;
    let token = lock(&received).clone().unwrap();
    assert!(!token.is_cancelled());

    agent.abort();
    prompt.await.unwrap().unwrap();
    assert!(token.is_cancelled());
}

fn tool_update_count(events: &[Arc<AgentEvent>]) -> usize {
    events
        .iter()
        .filter(|event| event.kind() == "tool_execution_update")
        .count()
}

#[tokio::test]
async fn ignores_tool_updates_after_the_tool_execution_settles() {
    let captured = shared(None::<ToolUpdates>);
    let slot = captured.clone();
    let delayed = tool("delayed_tool", empty_schema(), move |_, context| {
        let slot = slot.clone();
        async move {
            *lock(&slot) = Some(context.updates.clone());
            context.updates.send(AgentToolResult {
                details: Some(json!({"status": "running"})),
                ..AgentToolResult::text("running")
            });
            Ok(AgentToolResult {
                details: Some(json!({"status": "done"})),
                terminate: true,
                ..AgentToolResult::text("ok")
            })
        }
    });
    let agent = agent_with(
        scripted(vec![tool_use_message(vec![("call-1", "delayed_tool")])]),
        |options| options.initial_state.tools = vec![delayed],
    );
    let events = record_events(&agent);

    agent.prompt_text("run tool", Vec::new()).await.unwrap();
    let count_after_prompt = lock(&events).len();
    lock(&captured).as_ref().unwrap().send(AgentToolResult {
        details: Some(json!({"status": "late"})),
        ..AgentToolResult::text("late")
    });
    tokio::task::yield_now().await;

    let events = lock(&events);
    assert_eq!(tool_update_count(&events), 1);
    assert_eq!(events.len(), count_after_prompt);
}

#[tokio::test]
async fn ignores_a_settled_parallel_tool_update_while_another_tool_is_still_running() {
    let slow_started = Gate::default();
    let settled_ended = Gate::default();
    let release_slow = Gate::default();
    let captured = shared(None::<ToolUpdates>);
    let slot = captured.clone();
    let settled = tool("settled_tool", empty_schema(), move |_, context| {
        let slot = slot.clone();
        async move {
            *lock(&slot) = Some(context.updates.clone());
            Ok(AgentToolResult {
                terminate: true,
                ..AgentToolResult::text("done")
            })
        }
    });
    let (started, release) = (slow_started.clone(), release_slow.clone());
    let slow = tool("slow_tool", empty_schema(), move |_, _| {
        let (started, release) = (started.clone(), release.clone());
        async move {
            started.open();
            release.wait().await;
            Ok(AgentToolResult {
                terminate: true,
                ..AgentToolResult::text("done")
            })
        }
    });
    let agent = agent_with(
        scripted(vec![tool_use_message(vec![
            ("call-1", "settled_tool"),
            ("call-2", "slow_tool"),
        ])]),
        |options| options.initial_state.tools = vec![settled, slow],
    );
    let events = record_events(&agent);
    let ended = settled_ended.clone();
    agent.subscribe(move |event, _| {
        if let AgentEvent::ToolExecutionEnd { tool_call_id, .. } = &*event
            && tool_call_id == "call-1"
        {
            ended.open();
        }
        async {}
    });

    let prompt = spawn_prompt(&agent, "run tools");
    slow_started.wait().await;
    settled_ended.wait().await;
    let count_before_late_update = lock(&events).len();
    lock(&captured).as_ref().unwrap().send(AgentToolResult {
        details: Some(json!({"status": "late"})),
        ..AgentToolResult::text("late")
    });
    tokio::task::yield_now().await;
    assert_eq!(lock(&events).len(), count_before_late_update);

    release_slow.open();
    prompt.await.unwrap().unwrap();
    assert_eq!(tool_update_count(&lock(&events)), 0);
}

#[test]
fn updates_state_with_setters() {
    let agent = new_agent(unused_stream_fn());

    let new_model = tau_ai::Model {
        id: "gemini-2.5-flash".into(),
        ..model()
    };
    agent.set_model(new_model.clone());
    assert_eq!(agent.model(), new_model);

    agent.set_thinking_level(ModelThinkingLevel::High);
    agent.with_state(|state| assert_eq!(state.thinking_level, ModelThinkingLevel::High));

    agent.set_tools(vec![noop_tool("test")]);
    agent.with_state(|state| {
        assert_eq!(state.tools.len(), 1);
        assert_eq!(state.tools[0].definition().name, "test");
    });

    let messages = vec![user("Hello")];
    agent.set_messages(messages.clone());
    assert_eq!(agent.messages(), messages);

    let appended: AgentMessage = assistant_text("Hi").into();
    agent.with_state_mut(|state| state.messages.push(appended.clone()));
    assert_eq!(agent.messages().len(), 2);
    assert_eq!(agent.messages()[1], appended);

    agent.set_messages(Vec::new());
    assert!(agent.messages().is_empty());
}

#[test]
fn supports_the_steering_message_queue() {
    let agent = new_agent(unused_stream_fn());
    let message: AgentMessage = user("Steering message");
    agent.steer(message.clone());

    // Queued, not yet in the transcript.
    assert!(!agent.messages().contains(&message));
    assert!(agent.has_queued_messages());
}

#[test]
fn supports_the_follow_up_message_queue() {
    let agent = new_agent(unused_stream_fn());
    let message: AgentMessage = user("Follow-up message");
    agent.follow_up(message.clone());

    // Queued, not yet in the transcript.
    assert!(!agent.messages().contains(&message));
    assert!(agent.has_queued_messages());
}

#[test]
fn abort_without_an_active_run_does_nothing() {
    let agent = new_agent(unused_stream_fn());
    agent.abort();
    assert!(agent.cancel_token().is_none());
}

#[tokio::test]
async fn rejects_reset_while_processing_without_corrupting_the_transcript() {
    let started = Gate::default();
    let release = Gate::default();
    let (stream_started, stream_release) = (started.clone(), release.clone());
    let agent = new_agent(Arc::new(move |model, _, _| {
        let (sender_started, sender_release) = (stream_started.clone(), stream_release.clone());
        let (mut sender, stream) = AssistantMessageEventStream::channel();
        let partial = tau_ai::AssistantMessage::empty(&model.api, &model.provider, &model.id);
        tokio::spawn(async move {
            sender.push(AssistantMessageEvent::Start { message: partial });
            sender_started.open();
            sender_release.wait().await;
            sender.push(AssistantMessageEvent::Done {
                reason: DoneReason::Stop,
                message: assistant_text("Done"),
            });
        });
        Box::pin(async move { stream })
    }));

    let prompt = spawn_prompt(&agent, "Hello");
    started.wait().await;
    tick().await;

    assert!(agent.is_streaming());
    assert_eq!(roles(&agent.messages()), ["user"]);
    assert_eq!(agent.reset(), Err(AgentError::ResetWhileRunning));
    assert!(agent.is_streaming());
    assert_eq!(roles(&agent.messages()), ["user"]);

    release.open();
    prompt.await.unwrap().unwrap();
    assert!(!agent.is_streaming());
    assert_eq!(roles(&agent.messages()), ["user", "assistant"]);
}

#[tokio::test]
async fn rejects_prompt_while_streaming() {
    let agent = new_agent(abortable());

    let first = spawn_prompt(&agent, "First message");
    tick().await;
    assert!(agent.is_streaming());

    let error = agent.prompt_text("Second message", Vec::new()).await.unwrap_err();
    assert_eq!(error, AgentError::AlreadyRunning);
    assert!(error.to_string().starts_with("Agent is already processing a prompt."));

    agent.abort();
    first.await.unwrap().unwrap();
}

#[tokio::test]
async fn rejects_continue_while_streaming() {
    let agent = new_agent(abortable());

    let first = spawn_prompt(&agent, "First message");
    tick().await;
    assert!(agent.is_streaming());

    assert_eq!(agent.continue_run().await, Err(AgentError::ContinueWhileRunning));

    agent.abort();
    first.await.unwrap().unwrap();
}

#[tokio::test]
async fn continue_processes_queued_follow_up_messages_after_an_assistant_turn() {
    let agent = new_agent(scripted(vec![assistant_text("Processed")]));
    agent.set_messages(vec![
        tau_ai::UserMessage {
            content: tau_ai::UserContent::Blocks(vec![UserBlock::text("Initial")]),
            timestamp: tau_ai::types::now_ms() - 10,
        }
        .into(),
        assistant_text("Initial response").into(),
    ]);
    agent.follow_up(tau_ai::UserMessage::new(tau_ai::UserContent::Blocks(vec![
        UserBlock::text("Queued follow-up"),
    ])));

    agent.continue_run().await.unwrap();

    let messages = agent.messages();
    assert!(
        messages
            .iter()
            .any(|message| user_text(message).as_deref() == Some("Queued follow-up"))
    );
    assert_eq!(messages.last().unwrap().role(), "assistant");
}

async fn continue_keeps_steering_semantics_for_assistant_tail(mode: QueueMode, expected_requests: usize) {
    let requests = shared(Vec::new());
    let agent = agent_with(recording(requests.clone()), |options| options.steering_mode = mode);
    agent.set_messages(vec![user("Initial"), assistant_text("Initial response").into()]);
    agent.steer(user_message("Steering 1"));
    agent.steer(user_message("Steering 2"));

    agent.continue_run().await.unwrap();

    let requests = lock(&requests);
    assert_eq!(requests.len(), expected_requests);
    assert!(requests[0].contains(&"Steering 1".to_owned()));
    if mode == QueueMode::OneAtATime {
        assert!(!requests[0].contains(&"Steering 2".to_owned()));
        assert!(requests[1].contains(&"Steering 2".to_owned()));
    } else {
        assert!(requests[0].contains(&"Steering 2".to_owned()));
    }
}

#[tokio::test]
async fn continue_keeps_one_at_a_time_steering_semantics_for_assistant_tail_fallback() {
    continue_keeps_steering_semantics_for_assistant_tail(QueueMode::OneAtATime, 2).await;
}

#[tokio::test]
async fn continue_keeps_all_steering_semantics_for_assistant_tail_fallback() {
    continue_keeps_steering_semantics_for_assistant_tail(QueueMode::All, 1).await;
}

// Upstream's legacy `prepareNextTurn(signal)` callback: the hook gets the run's cancellation.
#[tokio::test]
async fn prepare_next_turn_receives_the_active_cancellation_token() {
    let saw_active_token = Arc::new(AtomicBool::new(false));
    let saw = saw_active_token.clone();
    let requests = Arc::new(AtomicUsize::new(0));
    let count = requests.clone();
    let agent = agent_with(
        stream_fn(move |index, _, _, _| {
            count.fetch_add(1, Ordering::SeqCst);
            respond(if index == 0 {
                tool_use_message(vec![("tool-1", "noop")])
            } else {
                assistant_text("done")
            })
        }),
        |options| {
            options.initial_state.tools = vec![noop_tool("noop")];
            options.hooks = Arc::new(TestHooks::new().prepare_next_turn(move |_, cancel| {
                saw.store(!cancel.is_cancelled(), Ordering::SeqCst);
                None
            }));
        },
    );

    agent.prompt_text("start", Vec::new()).await.unwrap();

    assert_eq!(requests.load(Ordering::SeqCst), 2);
    assert!(saw_active_token.load(Ordering::SeqCst));
}

#[tokio::test]
async fn forwards_finish_turn_through_agent_options_with_the_active_cancellation_token() {
    let saw_active_token = Arc::new(AtomicBool::new(false));
    let context_roles = shared(Vec::<String>::new());
    let (saw, recorded) = (saw_active_token.clone(), context_roles.clone());
    let requests = Arc::new(AtomicUsize::new(0));
    let count = requests.clone();
    let agent = agent_with(
        stream_fn(move |index, _, _, _| {
            count.fetch_add(1, Ordering::SeqCst);
            respond(if index == 0 {
                tool_use_message(vec![("tool-1", "noop")])
            } else {
                assistant_text("should not run")
            })
        }),
        |options| {
            options.initial_state.tools = vec![noop_tool("noop")];
            options.hooks = Arc::new(TestHooks::new().finish_turn(move |turn, cancel| {
                saw.store(!cancel.is_cancelled(), Ordering::SeqCst);
                *lock(&recorded) = roles(&turn.context.messages);
                Some(AgentTurnDecision::End)
            }));
        },
    );

    agent.prompt_text("start", Vec::new()).await.unwrap();

    assert_eq!(requests.load(Ordering::SeqCst), 1);
    assert!(saw_active_token.load(Ordering::SeqCst));
    assert_eq!(*lock(&context_roles), ["system", "user", "assistant", "toolResult"]);
}

async fn rejects_queued_continuation_without_draining_queues(messages: Vec<AgentMessage>) {
    let agent = agent_with(unused_stream_fn(), |options| options.initial_state.messages = messages);
    let steering: AgentMessage = user("steering");
    let follow_up: AgentMessage = user("follow-up");
    agent.steer(steering.clone());
    agent.follow_up(follow_up.clone());

    let error = agent.continue_run().await.unwrap_err();
    assert_eq!(error, AgentError::NoMessages);
    assert_eq!(error.to_string(), "No messages to continue from");
    assert_eq!(agent.peek_queued_messages(), [steering]);
    agent.clear_steering_queue();
    assert_eq!(agent.peek_queued_messages(), [follow_up]);
}

#[tokio::test]
async fn rejects_a_queued_continuation_from_empty_context_without_draining_queues() {
    rejects_queued_continuation_without_draining_queues(Vec::new()).await;
}

#[tokio::test]
async fn rejects_a_queued_continuation_from_system_only_context_without_draining_queues() {
    rejects_queued_continuation_without_draining_queues(vec![
        SystemMessage {
            content: "system only".into(),
            timestamp: 1,
            ..SystemMessage::default()
        }
        .into(),
    ])
    .await;
}

async fn defers_follow_up_on_first_continuation_request(messages: Vec<AgentMessage>) {
    let requests = shared(Vec::new());
    let agent = agent_with(recording(requests.clone()), |options| {
        options.initial_state.messages = messages;
    });
    agent.follow_up(user_message("follow-up"));

    agent.continue_run().await.unwrap();

    let requests = lock(&requests);
    assert_eq!(requests.len(), 2);
    assert!(!requests[0].contains(&"follow-up".to_owned()));
    assert!(requests[1].contains(&"follow-up".to_owned()));
}

#[tokio::test]
async fn defers_follow_up_input_on_the_first_continuation_request_from_a_user_tail() {
    defers_follow_up_on_first_continuation_request(vec![user("existing user")]).await;
}

#[tokio::test]
async fn defers_follow_up_input_on_the_first_continuation_request_from_a_tool_result_tail() {
    defers_follow_up_on_first_continuation_request(vec![
        user("existing user"),
        tool_use_message(vec![("call-1", "noop")]).into(),
        ToolResultMessage {
            tool_call_id: "call-1".into(),
            tool_name: "noop".into(),
            content: vec![UserBlock::text("done")],
            details: None,
            usage: None,
            nested_calls: None,
            is_error: false,
            timestamp: 1,
            duration_ms: None,
        }
        .into(),
    ])
    .await;
}

async fn polls_steering_at_continuation_startup(mode: QueueMode, expected_requests: usize) {
    let requests = shared(Vec::new());
    let agent = agent_with(recording(requests.clone()), |options| {
        options.initial_state.messages = vec![user("existing")];
        options.steering_mode = mode;
    });
    agent.steer(user_message("first"));
    agent.steer(user_message("second"));

    agent.continue_run().await.unwrap();

    let requests = lock(&requests);
    assert_eq!(requests.len(), expected_requests);
    assert!(requests[0].contains(&"first".to_owned()));
    if mode == QueueMode::OneAtATime {
        assert!(!requests[0].contains(&"second".to_owned()));
        assert!(requests[1].contains(&"second".to_owned()));
    } else {
        assert!(requests[0].contains(&"second".to_owned()));
    }
}

#[tokio::test]
async fn polls_one_at_a_time_steering_at_continuation_startup() {
    polls_steering_at_continuation_startup(QueueMode::OneAtATime, 2).await;
}

#[tokio::test]
async fn polls_all_steering_at_continuation_startup() {
    polls_steering_at_continuation_startup(QueueMode::All, 1).await;
}

#[tokio::test]
async fn keeps_steering_ahead_of_follow_up_from_a_non_assistant_continuation_tail() {
    let requests = shared(Vec::new());
    let agent = agent_with(recording(requests.clone()), |options| {
        options.initial_state.messages = vec![user("existing")];
    });
    agent.steer(user_message("steering"));
    agent.follow_up(user_message("follow-up"));

    agent.continue_run().await.unwrap();

    let requests = lock(&requests);
    assert_eq!(requests.len(), 2);
    assert!(requests[0].contains(&"steering".to_owned()));
    assert!(!requests[0].contains(&"follow-up".to_owned()));
    assert!(requests[1].contains(&"follow-up".to_owned()));
}

/// Steers `message` when the assistant message ends.
fn steer_on_assistant_end(agent: &Agent, message: AgentMessage) {
    let steering = agent.clone();
    agent.subscribe(move |event, _| {
        if let AgentEvent::MessageEnd { message: ended } = &*event
            && ended.role() == "assistant"
        {
            steering.steer(message.clone());
        }
        async {}
    });
}

async fn keeps_queues_on_failed_response(reason: StopReason) {
    let queued_during_response: AgentMessage = user("steering");
    let follow_up: AgentMessage = user("follow-up");
    let agent = agent_with(
        stream_fn(move |_, _, _, _| respond(failed(reason, reason.as_str()))),
        |options| {
            options.hooks = Arc::new(TestHooks::new().finish_turn(|_, _| Some(AgentTurnDecision::Continue)));
        },
    );
    agent.follow_up(follow_up.clone());
    steer_on_assistant_end(&agent, queued_during_response.clone());

    agent.prompt_text("start", Vec::new()).await.unwrap();

    assert_eq!(agent.peek_queued_messages(), [queued_during_response]);
    agent.clear_steering_queue();
    assert_eq!(agent.peek_queued_messages(), [follow_up]);
}

#[tokio::test]
async fn keeps_queues_on_an_error_response_even_when_finish_turn_requests_continuation() {
    keeps_queues_on_failed_response(StopReason::Error).await;
}

#[tokio::test]
async fn keeps_queues_on_an_aborted_response_even_when_finish_turn_requests_continuation() {
    keeps_queues_on_failed_response(StopReason::Aborted).await;
}

#[tokio::test]
async fn keeps_queues_when_finish_turn_ends_the_run() {
    let queued_during_response: AgentMessage = user("steering");
    let follow_up: AgentMessage = user("follow-up");
    let agent = agent_with(scripted(vec![assistant_text("done")]), |options| {
        options.hooks = Arc::new(TestHooks::new().finish_turn(|_, _| Some(AgentTurnDecision::End)));
    });
    agent.follow_up(follow_up.clone());
    steer_on_assistant_end(&agent, queued_during_response.clone());

    agent.prompt_text("start", Vec::new()).await.unwrap();

    assert_eq!(agent.peek_queued_messages(), [queued_during_response]);
    agent.clear_steering_queue();
    assert_eq!(agent.peek_queued_messages(), [follow_up]);
}

#[test]
fn previews_the_next_selected_queued_messages_without_consuming_them() {
    let agent = agent_with(unused_stream_fn(), |options| {
        options.steering_mode = QueueMode::OneAtATime;
        options.follow_up_mode = QueueMode::All;
    });
    let first: AgentMessage = user("first steering");
    let second: AgentMessage = user("second steering");
    let follow_up: AgentMessage = user("follow-up");
    agent.steer(first.clone());
    agent.steer(second);
    agent.follow_up(follow_up.clone());

    assert_eq!(agent.peek_queued_messages(), std::slice::from_ref(&first));
    assert_eq!(agent.peek_queued_messages(), [first]);
    agent.clear_steering_queue();
    assert_eq!(agent.peek_queued_messages(), [follow_up]);
}

#[tokio::test]
async fn forwards_provider_stream_event_observers_through_agent_options() {
    let provider_events = shared(Vec::new());
    let recorded = provider_events.clone();
    let agent = agent_with(
        Arc::new(|model, _, options| {
            let (mut sender, stream) = AssistantMessageEventStream::channel();
            let observer = options.base.on_provider_stream_event.clone().unwrap();
            tokio::spawn(async move {
                observer(&json!({"request_cost": 0.01}), &model).await.unwrap();
                sender.push(AssistantMessageEvent::Done {
                    reason: DoneReason::Stop,
                    message: assistant_text("ok"),
                });
            });
            Box::pin(async move { stream })
        }),
        |options| {
            options.on_provider_stream_event = Some(Arc::new(move |data, _| {
                lock(&recorded).push(data.clone());
                Box::pin(async { Ok(()) })
            }));
        },
    );

    agent.prompt_text("hello", Vec::new()).await.unwrap();

    assert_eq!(*lock(&provider_events), [json!({"request_cost": 0.01})]);
}

#[tokio::test]
async fn forwards_session_id_to_stream_function_options() {
    let received = shared(None::<String>);
    let recorded = received.clone();
    let agent = agent_with(
        stream_fn(move |_, _, _, options| {
            lock(&recorded).clone_from(&options.base.session_id);
            respond(assistant_text("ok"))
        }),
        |options| options.session_id = Some("session-abc".into()),
    );

    agent.prompt_text("hello", Vec::new()).await.unwrap();
    assert_eq!(lock(&received).as_deref(), Some("session-abc"));

    agent.set_session_id(Some("session-def".into()));
    agent.prompt_text("hello again", Vec::new()).await.unwrap();
    assert_eq!(lock(&received).as_deref(), Some("session-def"));
}
