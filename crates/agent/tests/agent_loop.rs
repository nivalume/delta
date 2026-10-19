//! Tests of the low-level loop, ported from upstream `test/agent-loop.test.ts`.

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

mod common;

use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use delta_agent::{
    AfterToolCallResult, AgentContext, AgentEvent, AgentHooks, AgentLoopError, AgentLoopTurnUpdate, AgentMessage,
    AgentRequestUpdate, AgentTool, AgentToolResult, AgentTurnDecision, BeforeToolCallContext, BeforeToolCallResult,
    CustomMessage, FnTool, NoCustom, RunToolCallOptions, ToolContext, ToolError, ToolExecutionMode, agent_loop,
    agent_loop_continue, run_agent_loop, run_tool_call,
};
use delta_ai::types::UsageCost;
use delta_ai::{
    JsonObject, Message, ModelThinkingLevel, StopReason, SystemMessage, ThinkingLevel, Tool, Usage, UserBlock,
};

use common::*;

fn usage(base: u64) -> Usage {
    let base_cost = base as f64 / 10.0;
    Usage {
        input: base,
        output: base + 1,
        cache_read: base + 2,
        cache_write: base + 3,
        total_tokens: 4 * base + 6,
        cost: UsageCost {
            input: base_cost,
            output: base_cost + 0.1,
            cache_read: base_cost + 0.2,
            cache_write: base_cost + 0.3,
            total: 4.0 * base_cost + 0.6,
        },
        ..Usage::default()
    }
}

fn tool_call_message(calls: Vec<(&str, &str, Value)>) -> delta_ai::AssistantMessage {
    assistant(
        calls
            .into_iter()
            .map(|(id, name, arguments)| tool_call(id, name, arguments))
            .collect(),
        StopReason::ToolUse,
    )
}

/// Requests: the first answers with `first`, later ones with "done".
fn first_then_done(first: delta_ai::AssistantMessage, calls: Arc<AtomicUsize>) -> delta_agent::StreamFn {
    stream_fn(move |index, _, _, _| {
        calls.fetch_add(1, Ordering::SeqCst);
        respond(if index == 0 {
            first.clone()
        } else {
            assistant_text("done")
        })
    })
}

fn message_role<C: CustomMessage>(event: &AgentEvent<C>) -> Option<String> {
    match event {
        AgentEvent::MessageEnd { message } => Some(message.role().to_owned()),
        _ => None,
    }
}

#[tokio::test]
async fn emits_events_with_agent_message_types() {
    let stream = agent_loop(
        vec![user("Hello")],
        AgentContext::<NoCustom>::default(),
        config(TestHooks::new()),
        CancellationToken::new(),
        scripted(vec![assistant_text("Hi there!")]),
    );
    let (events, messages) = collect(stream).await;

    assert_eq!(roles(&messages), ["user", "assistant"]);
    let kinds = kinds(&events);
    for kind in [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ] {
        assert!(kinds.contains(&kind), "missing {kind}");
    }
}

#[tokio::test]
async fn builds_provider_context_exclusively_from_transcript_messages() {
    let initial = SystemMessage {
        content: "Transcript prompt".into(),
        tools_added: Some(Vec::new()),
        timestamp: 1,
        ..SystemMessage::default()
    };
    let seen = shared(Vec::new());
    let recorded = seen.clone();
    let stream = agent_loop(
        vec![initial.clone().into(), user("Hello")],
        AgentContext::<NoCustom>::default(),
        config(TestHooks::new()),
        CancellationToken::new(),
        stream_fn(move |_, _, context, _| {
            *lock(&recorded) = context.messages().to_vec();
            respond(assistant_text("done"))
        }),
    );
    stream.result().await;

    // The provider receives a transcript: the prompt is the first message.
    assert_eq!(lock(&seen)[0], Message::System(initial));
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Custom {
    role: String,
    text: String,
    timestamp: i64,
}

impl CustomMessage for Custom {
    fn role(&self) -> &str {
        &self.role
    }

    fn timestamp(&self) -> i64 {
        self.timestamp
    }
}

fn custom(role: &str, text: &str) -> AgentMessage<Custom> {
    AgentMessage::Custom(Custom {
        role: role.into(),
        text: text.into(),
        timestamp: delta_ai::types::now_ms(),
    })
}

fn conversation_only<C: CustomMessage>(messages: &[AgentMessage<C>]) -> Vec<Message> {
    messages
        .iter()
        .filter_map(AgentMessage::as_llm)
        .filter(|message| matches!(message.role(), "user" | "assistant" | "toolResult"))
        .cloned()
        .collect()
}

#[tokio::test]
async fn handles_custom_message_types_via_convert_to_llm() {
    let converted = shared(Vec::new());
    let recorded = converted.clone();
    let hooks = TestHooks::<Custom>::default().convert_to_llm(move |messages: &[AgentMessage<Custom>]| {
        // Notifications are not LLM messages: drop them.
        let messages = conversation_only(messages);
        lock(&recorded).clone_from(&messages);
        messages
    });
    let context = AgentContext {
        messages: vec![custom("notification", "This is a notification")],
        tools: Vec::new(),
    };
    let stream = agent_loop(
        vec![user("Hello")],
        context,
        config(hooks),
        CancellationToken::new(),
        scripted(vec![assistant_text("Response")]),
    );
    collect(stream).await;

    let converted = lock(&converted);
    assert_eq!(converted.len(), 1);
    assert_eq!(converted[0].role(), "user");
}

#[tokio::test]
async fn applies_transform_context_before_convert_to_llm() {
    let transformed = Arc::new(AtomicUsize::new(0));
    let converted = Arc::new(AtomicUsize::new(0));
    let (transformed_count, converted_count) = (transformed.clone(), converted.clone());
    let hooks = TestHooks::new()
        .transform_context(move |messages: Vec<AgentMessage>| {
            // Keep only the last 2 messages.
            let pruned = messages[messages.len() - 2..].to_vec();
            transformed_count.store(pruned.len(), Ordering::SeqCst);
            pruned
        })
        .convert_to_llm(move |messages| {
            let messages = conversation_only(messages);
            converted_count.store(messages.len(), Ordering::SeqCst);
            messages
        });
    let context = AgentContext {
        messages: vec![
            user("old message 1"),
            assistant_text("old response 1").into(),
            user("old message 2"),
            assistant_text("old response 2").into(),
        ],
        tools: Vec::new(),
    };
    let stream = agent_loop(
        vec![user("new message")],
        context,
        config(hooks),
        CancellationToken::new(),
        scripted(vec![assistant_text("Response")]),
    );
    collect(stream).await;

    assert_eq!(transformed.load(Ordering::SeqCst), 2);
    assert_eq!(converted.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn handles_tool_calls_and_results() {
    let executed = shared(Vec::new());
    let observed_usage = shared(None);
    let recorded = observed_usage.clone();
    let tool_usage = usage(1);
    let patched_usage = usage(5);
    let patched = patched_usage.clone();
    let returned_usage = tool_usage.clone();
    let log = executed.clone();
    let echo = tool("echo", value_schema(), move |args, _| {
        let log = log.clone();
        let usage = returned_usage.clone();
        async move {
            log.lock().unwrap().push(args["value"].clone());
            Ok(AgentToolResult {
                details: Some(json!({ "value": args["value"] })),
                usage: Some(usage),
                ..AgentToolResult::text(format!("echoed: {}", args["value"].as_str().unwrap()))
            })
        }
    });
    let hooks = TestHooks::new().after_tool_call(move |context| {
        *lock(&recorded) = context.result.usage.clone();
        Some(AfterToolCallResult {
            usage: Some(patched.clone()),
            ..AfterToolCallResult::default()
        })
    });
    let stream = agent_loop(
        vec![user("echo something")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![echo],
        },
        config(hooks),
        CancellationToken::new(),
        first_then_done(
            tool_call_message(vec![("tool-1", "echo", json!({"value": "hello"}))]),
            Arc::default(),
        ),
    );
    let (events, messages) = collect(stream).await;

    assert_eq!(*lock(&executed), [json!("hello")]);
    assert!(events.iter().any(|event| event.kind() == "tool_execution_start"));
    let Some(AgentEvent::ToolExecutionEnd { is_error, .. }) =
        events.iter().find(|event| event.kind() == "tool_execution_end")
    else {
        panic!("no tool_execution_end")
    };
    assert!(!is_error);
    assert_eq!(*lock(&observed_usage), Some(tool_usage));
    assert_eq!(tool_results(&messages)[0].usage, Some(patched_usage));
}

/// Blocks the call with id "blocked" after a slow check.
struct SlowBeforeToolCall;

impl AgentHooks for SlowBeforeToolCall {
    fn before_tool_call<'a>(
        &'a self,
        context: BeforeToolCallContext<'a>,
        _cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Option<BeforeToolCallResult>> {
        Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            (context.tool_call.id == "blocked").then(|| BeforeToolCallResult {
                block: true,
                reason: Some("no".into()),
                terminate: false,
            })
        })
    }
}

// #10549
#[tokio::test]
async fn records_how_long_execute_took_on_the_tool_result_excluding_hooks() {
    let echo = tool("echo", value_schema(), |args, _| async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        Ok(AgentToolResult::text(args["value"].as_str().unwrap()))
    });
    let mut config = config(TestHooks::new());
    config.hooks = Arc::new(SlowBeforeToolCall);
    let stream = agent_loop(
        vec![user("go")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![echo],
        },
        config,
        CancellationToken::new(),
        first_then_done(
            tool_call_message(vec![
                ("ran", "echo", json!({"value": "a"})),
                ("blocked", "echo", json!({"value": "b"})),
            ]),
            Arc::default(),
        ),
    );
    let (_, messages) = collect(stream).await;

    let results = tool_results(&messages);
    let (ran, blocked) = (&results[0], &results[1]);
    let duration = ran.duration_ms.unwrap();
    assert!((25..100).contains(&duration), "duration {duration}");
    assert!(blocked.is_error);
    assert_eq!(blocked.duration_ms, None);
}

#[tokio::test]
async fn does_not_execute_tool_calls_from_a_length_truncated_assistant_message() {
    let executed = shared(Vec::new());
    let calls = Arc::new(AtomicUsize::new(0));
    // Output hit the token limit mid tool call: the salvaged arguments validate but may be
    // truncated, so nothing in this message may execute.
    let truncated = assistant(
        vec![tool_call("tool-1", "echo", json!({"value": "hel"}))],
        StopReason::Length,
    );
    let stream = agent_loop(
        vec![user("echo something")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![echo_tool(executed.clone())],
        },
        config(TestHooks::new()),
        CancellationToken::new(),
        first_then_done(truncated, calls.clone()),
    );
    let (events, messages) = collect(stream).await;

    assert!(lock(&executed).is_empty());
    let Some(AgentEvent::ToolExecutionEnd { is_error, result, .. }) =
        events.iter().find(|event| event.kind() == "tool_execution_end")
    else {
        panic!("no tool_execution_end")
    };
    assert!(is_error);
    assert!(first_text(&result.content).contains("output token limit"));
    // The loop continues so the model can re-issue the tool call.
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(messages.last().unwrap().role(), "assistant");
}

/// `edit` takes a list of edits; `prepare_arguments` folds a legacy top-level edit into it.
struct EditTool {
    definition: Tool,
    executed: Shared<Vec<Value>>,
}

impl AgentTool for EditTool {
    fn definition(&self) -> &Tool {
        &self.definition
    }

    fn prepare_arguments(&self, arguments: JsonObject) -> JsonObject {
        let (Some(Value::String(old_text)), Some(Value::String(new_text))) =
            (arguments.get("oldText"), arguments.get("newText"))
        else {
            return arguments;
        };
        let mut edits = arguments
            .get("edits")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        edits.push(json!({ "oldText": old_text, "newText": new_text }));
        let mut prepared = JsonObject::new();
        prepared.insert("edits".into(), Value::Array(edits));
        prepared
    }

    fn execute(&self, arguments: Value, _context: ToolContext) -> BoxFuture<'_, Result<AgentToolResult, ToolError>> {
        Box::pin(async move {
            let count = arguments["edits"].as_array().map_or(0, Vec::len);
            lock(&self.executed).push(arguments["edits"].clone());
            Ok(AgentToolResult {
                details: Some(json!({ "count": count })),
                ..AgentToolResult::text(format!("edited {count}"))
            })
        })
    }
}

#[tokio::test]
async fn prepares_tool_arguments_for_validation() {
    let executed = shared(Vec::new());
    let edit = EditTool {
        definition: Tool::new(
            "edit",
            "Edit tool",
            json!({
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": { "type": "string" },
                                "newText": { "type": "string" },
                            },
                            "required": ["oldText", "newText"],
                        },
                    },
                },
                "required": ["edits"],
            }),
        ),
        executed: executed.clone(),
    };
    let stream = agent_loop(
        vec![user("edit something")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![Arc::new(edit)],
        },
        config(TestHooks::new()),
        CancellationToken::new(),
        first_then_done(
            tool_call_message(vec![(
                "tool-1",
                "edit",
                json!({"oldText": "before", "newText": "after"}),
            )]),
            Arc::default(),
        ),
    );
    collect(stream).await;

    assert_eq!(*lock(&executed), [json!([{ "oldText": "before", "newText": "after" }])]);
}

/// Records whether the call with value "second" ran while "first" was still blocked on
/// `release`.
#[derive(Clone, Default)]
struct Overlap {
    release: Gate,
    first_resolved: Arc<AtomicBool>,
    parallel_observed: Arc<AtomicBool>,
}

impl Overlap {
    fn tool(&self, name: &str, mode: Option<ToolExecutionMode>) -> Arc<dyn AgentTool> {
        let overlap = self.clone();
        let tool = FnTool::new(
            Tool::new(name, format!("{name} tool"), value_schema()),
            move |args: Value, _| {
                let overlap = overlap.clone();
                async move {
                    let value = args["value"].as_str().unwrap().to_owned();
                    if value == "first" {
                        overlap.release.wait().await;
                        overlap.first_resolved.store(true, Ordering::SeqCst);
                    }
                    if value == "second" && !overlap.first_resolved.load(Ordering::SeqCst) {
                        overlap.parallel_observed.store(true, Ordering::SeqCst);
                    }
                    Ok(AgentToolResult::text(format!("echoed: {value}")))
                }
            },
        );
        match mode {
            Some(mode) => Arc::new(tool.with_execution_mode(mode)),
            None => Arc::new(tool),
        }
    }

    /// First request: two calls to `name`, then open the gate after 20 ms.
    fn stream_fn(&self, first: (&str, &str), second: (&str, &str)) -> delta_agent::StreamFn {
        let release = self.release.clone();
        let message = tool_call_message(vec![
            ("tool-1", first.0, json!({ "value": first.1 })),
            ("tool-2", second.0, json!({ "value": second.1 })),
        ]);
        stream_fn(move |index, _, _, _| {
            if index > 0 {
                return respond(assistant_text("done"));
            }
            let release = release.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(20)).await;
                release.open();
            });
            respond(message.clone())
        })
    }

    fn parallel_observed(&self) -> bool {
        self.parallel_observed.load(Ordering::SeqCst)
    }
}

fn tool_result_message_ids<C>(events: &[AgentEvent<C>]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageEnd {
                message: AgentMessage::Llm(Message::ToolResult(result)),
            } => Some(result.tool_call_id.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn emits_tool_execution_end_in_completion_order_but_persists_tool_results_in_source_order() {
    let overlap = Overlap::default();
    let mut config = config(TestHooks::new());
    config.tool_execution = ToolExecutionMode::Parallel;
    let stream = agent_loop(
        vec![user("echo both")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![overlap.tool("echo", None)],
        },
        config,
        CancellationToken::new(),
        overlap.stream_fn(("echo", "first"), ("echo", "second")),
    );
    let (events, _) = collect(stream).await;

    let execution_end_ids: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.as_str()),
            _ => None,
        })
        .collect();
    let turn_result_ids: Vec<String> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::TurnEnd { tool_results, .. } => Some(tool_results),
            _ => None,
        })
        .flatten()
        .map(|result| result.tool_call_id.clone())
        .collect();

    assert!(overlap.parallel_observed());
    assert_eq!(execution_end_ids, ["tool-2", "tool-1"]);
    assert_eq!(tool_result_message_ids(&events), ["tool-1", "tool-2"]);
    assert_eq!(turn_result_ids, ["tool-1", "tool-2"]);
}

#[tokio::test]
async fn injects_queued_messages_after_all_tool_calls_complete() {
    let executed = shared(Vec::new());
    let delivered = Arc::new(AtomicBool::new(false));
    let saw_interrupt = Arc::new(AtomicBool::new(false));
    let mut config = config(TestHooks::new());
    config.tool_execution = ToolExecutionMode::Sequential;
    let log = executed.clone();
    config.get_steering_messages = Some(source(move || {
        // Deliver steering once a tool has run.
        if !lock(&log).is_empty() && !delivered.swap(true, Ordering::SeqCst) {
            vec![user("interrupt")]
        } else {
            Vec::new()
        }
    }));
    let saw = saw_interrupt.clone();
    let first = tool_call_message(vec![
        ("tool-1", "echo", json!({"value": "first"})),
        ("tool-2", "echo", json!({"value": "second"})),
    ]);
    let stream = agent_loop(
        vec![user("start")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![echo_tool(executed.clone())],
        },
        config,
        CancellationToken::new(),
        stream_fn(move |index, _, context, _| {
            if index == 1 {
                saw.store(
                    user_texts(context).iter().any(|text| text == "interrupt"),
                    Ordering::SeqCst,
                );
            }
            respond(if index == 0 {
                first.clone()
            } else {
                assistant_text("done")
            })
        }),
    );
    let (events, _) = collect(stream).await;

    // Both tools execute before steering is injected.
    assert_eq!(*lock(&executed), [json!("first"), json!("second")]);
    let tool_ends: Vec<bool> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionEnd { is_error, .. } => Some(*is_error),
            _ => None,
        })
        .collect();
    assert_eq!(tool_ends, [false, false]);

    let sequence: Vec<String> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageStart {
                message: AgentMessage::Llm(Message::ToolResult(result)),
            } => Some(format!("tool:{}", result.tool_call_id)),
            AgentEvent::MessageStart {
                message:
                    AgentMessage::Llm(Message::User(delta_ai::UserMessage {
                        content: delta_ai::UserContent::Text(text),
                        ..
                    })),
            } => Some(text.clone()),
            _ => None,
        })
        .collect();
    let position = |item: &str| sequence.iter().position(|entry| entry == item).unwrap();
    assert!(position("tool:tool-1") < position("interrupt"));
    assert!(position("tool:tool-2") < position("interrupt"));
    // The interrupt is in the context of the second request.
    assert!(saw_interrupt.load(Ordering::SeqCst));
}

#[tokio::test]
async fn forces_sequential_execution_when_a_tool_has_sequential_mode_even_with_default_parallel_config() {
    let overlap = Overlap::default();
    let stream = agent_loop(
        vec![user("run both")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![overlap.tool("slow", Some(ToolExecutionMode::Sequential))],
        },
        config(TestHooks::new()),
        CancellationToken::new(),
        overlap.stream_fn(("slow", "first"), ("slow", "second")),
    );
    let (events, _) = collect(stream).await;

    // The second call does not start before the first finishes.
    assert!(!overlap.parallel_observed());
    assert_eq!(tool_result_message_ids(&events), ["tool-1", "tool-2"]);
}

#[tokio::test]
async fn forces_sequential_execution_when_one_of_multiple_tools_has_sequential_mode() {
    let order = shared(Vec::<String>::new());
    let release = Gate::default();
    let (slow_order, slow_release) = (order.clone(), release.clone());
    let slow = FnTool::new(Tool::new("slow", "Slow tool", value_schema()), move |args: Value, _| {
        let (order, release) = (slow_order.clone(), slow_release.clone());
        async move {
            let value = args["value"].as_str().unwrap().to_owned();
            lock(&order).push(format!("slow:{value}"));
            if value == "a" {
                release.wait().await;
            }
            Ok(AgentToolResult::text(format!("slow: {value}")))
        }
    })
    .with_execution_mode(ToolExecutionMode::Sequential);
    let fast_order = order.clone();
    // No execution mode: parallel by default.
    let fast = tool("fast", value_schema(), move |args, _| {
        let order = fast_order.clone();
        async move {
            let value = args["value"].as_str().unwrap().to_owned();
            lock(&order).push(format!("fast:{value}"));
            Ok(AgentToolResult::text(format!("fast: {value}")))
        }
    });
    let message = tool_call_message(vec![
        ("tool-1", "slow", json!({"value": "a"})),
        ("tool-2", "fast", json!({"value": "b"})),
    ]);
    let stream = agent_loop(
        vec![user("run both")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![Arc::new(slow), fast],
        },
        config(TestHooks::new()),
        CancellationToken::new(),
        stream_fn(move |index, _, _, _| {
            if index > 0 {
                return respond(assistant_text("done"));
            }
            let release = release.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(20)).await;
                release.open();
            });
            respond(message.clone())
        }),
    );
    collect(stream).await;

    // The fast tool does not run before the slow one finishes.
    let order = lock(&order);
    assert_eq!(order[0], "slow:a");
    assert!(order.contains(&"fast:b".to_owned()));
}

#[tokio::test]
async fn allows_parallel_execution_when_all_tools_have_parallel_mode() {
    let overlap = Overlap::default();
    let stream = agent_loop(
        vec![user("echo both")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![overlap.tool("echo", Some(ToolExecutionMode::Parallel))],
        },
        config(TestHooks::new()),
        CancellationToken::new(),
        overlap.stream_fn(("echo", "first"), ("echo", "second")),
    );
    collect(stream).await;

    assert!(overlap.parallel_observed());
}

fn terminating_echo() -> Arc<dyn AgentTool> {
    tool("echo", value_schema(), |args, _| async move {
        Ok(AgentToolResult {
            terminate: true,
            ..AgentToolResult::text(args["value"].as_str().unwrap())
        })
    })
}

#[tokio::test]
async fn runs_finish_turn_after_tool_result_messages_and_before_turn_end() {
    let ordering = shared(Vec::<String>::new());
    let hook_ordering = ordering.clone();
    let hooks = TestHooks::new().finish_turn(move |turn, _| {
        lock(&hook_ordering).push("finishTurn".into());
        assert_eq!(turn.tool_results.len(), 1);
        assert_eq!(turn.context.messages.last().unwrap().role(), "toolResult");
        None
    });
    let sink_ordering = ordering.clone();
    let sink = FnSink(move |event: AgentEvent| match &event {
        AgentEvent::MessageEnd { message } => lock(&sink_ordering).push(format!("message_end:{}", message.role())),
        AgentEvent::TurnEnd { .. } => lock(&sink_ordering).push("turn_end".into()),
        _ => {}
    });
    run_agent_loop(
        vec![user("echo")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![terminating_echo()],
        },
        config(hooks),
        &sink,
        CancellationToken::new(),
        &scripted(vec![tool_call_message(vec![(
            "tool-1",
            "echo",
            json!({"value": "hello"}),
        )])]),
    )
    .await;

    let ordering = lock(&ordering);
    assert_eq!(
        ordering[ordering.len() - 3..],
        ["message_end:toolResult", "finishTurn", "turn_end"]
    );
}

async fn finish_turn_runs_for_failed_assistant(reason: StopReason) {
    let ordering = shared(Vec::<String>::new());
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let steering_polls = Arc::new(AtomicUsize::new(0));
    let follow_up_polls = Arc::new(AtomicUsize::new(0));
    let hook_ordering = ordering.clone();
    let mut config = config(TestHooks::new().finish_turn(move |turn, _| {
        assert_eq!(turn.message.stop_reason, reason);
        lock(&hook_ordering).push("finishTurn".into());
        Some(AgentTurnDecision::Continue)
    }));
    let polls = steering_polls.clone();
    config.get_steering_messages = Some(source(move || {
        polls.fetch_add(1, Ordering::SeqCst);
        Vec::new()
    }));
    let polls = follow_up_polls.clone();
    config.get_follow_up_messages = Some(source(move || {
        polls.fetch_add(1, Ordering::SeqCst);
        vec![user("queued")]
    }));
    let sink_ordering = ordering.clone();
    let sink = FnSink(move |event: AgentEvent| {
        if event.kind() == "turn_end" {
            lock(&sink_ordering).push("turn_end".into());
        }
    });
    let calls = provider_calls.clone();
    run_agent_loop(
        vec![user("run")],
        AgentContext::default(),
        config,
        &sink,
        CancellationToken::new(),
        &stream_fn(move |_, _, _, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            respond(failed(reason, reason.as_str()))
        }),
    )
    .await;

    assert_eq!(*lock(&ordering), ["finishTurn", "turn_end"]);
    assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
    assert_eq!(steering_polls.load(Ordering::SeqCst), 1);
    assert_eq!(follow_up_polls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn runs_finish_turn_for_an_error_assistant_before_turn_end_without_changing_the_hard_exit() {
    finish_turn_runs_for_failed_assistant(StopReason::Error).await;
}

#[tokio::test]
async fn runs_finish_turn_for_an_aborted_assistant_before_turn_end_without_changing_the_hard_exit() {
    finish_turn_runs_for_failed_assistant(StopReason::Aborted).await;
}

#[tokio::test]
async fn action_end_skips_queue_polling_and_next_turn_preparation() {
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let steering_polls = Arc::new(AtomicUsize::new(0));
    let follow_up_polls = Arc::new(AtomicUsize::new(0));
    let prepare_calls = Arc::new(AtomicUsize::new(0));
    let prepares = prepare_calls.clone();
    let mut config = config(
        TestHooks::new()
            .finish_turn(|_, _| Some(AgentTurnDecision::End))
            .prepare_next_turn(move |_, _| {
                prepares.fetch_add(1, Ordering::SeqCst);
                None
            }),
    );
    let polls = steering_polls.clone();
    config.get_steering_messages = Some(source(move || {
        polls.fetch_add(1, Ordering::SeqCst);
        Vec::new()
    }));
    let polls = follow_up_polls.clone();
    config.get_follow_up_messages = Some(source(move || {
        polls.fetch_add(1, Ordering::SeqCst);
        vec![user("queued")]
    }));
    let calls = provider_calls.clone();
    let stream = agent_loop(
        vec![user("run")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![noop_tool("noop")],
        },
        config,
        CancellationToken::new(),
        stream_fn(move |_, _, _, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            respond(tool_call_message(vec![("tool-1", "noop", json!({}))]))
        }),
    );
    stream.result().await;

    assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
    assert_eq!(steering_polls.load(Ordering::SeqCst), 1);
    assert_eq!(follow_up_polls.load(Ordering::SeqCst), 0);
    assert_eq!(prepare_calls.load(Ordering::SeqCst), 0);
}

/// `finish_turn` that requests continuation on its first call only.
fn continue_once(finish_calls: Arc<AtomicUsize>) -> TestHooks {
    TestHooks::new().finish_turn(move |_, _| {
        let calls = finish_calls.fetch_add(1, Ordering::SeqCst) + 1;
        (calls == 1).then_some(AgentTurnDecision::Continue)
    })
}

#[tokio::test]
async fn makes_exactly_one_context_only_request_when_no_natural_request_satisfies_continuation() {
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let finish_calls = Arc::new(AtomicUsize::new(0));
    let calls = provider_calls.clone();
    let stream = agent_loop(
        vec![user("run")],
        AgentContext::default(),
        config(continue_once(finish_calls.clone())),
        CancellationToken::new(),
        stream_fn(move |index, _, _, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            respond(assistant_text(&format!("response {}", index + 1)))
        }),
    );
    stream.result().await;

    assert_eq!(provider_calls.load(Ordering::SeqCst), 2);
    assert_eq!(finish_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn lets_a_natural_tool_result_request_satisfy_continuation() {
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let finish_calls = Arc::new(AtomicUsize::new(0));
    let stream = agent_loop(
        vec![user("run")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![noop_tool("noop")],
        },
        config(continue_once(finish_calls.clone())),
        CancellationToken::new(),
        first_then_done(
            tool_call_message(vec![("tool-1", "noop", json!({}))]),
            provider_calls.clone(),
        ),
    );
    stream.result().await;

    assert_eq!(provider_calls.load(Ordering::SeqCst), 2);
    assert_eq!(finish_calls.load(Ordering::SeqCst), 2);
}

async fn natural_queued_request_satisfies_continuation(steering: bool) {
    let queued = if steering { "steering" } else { "follow-up" };
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let finish_calls = Arc::new(AtomicUsize::new(0));
    let second_request_users = shared(Vec::new());
    let mut config = config(continue_once(finish_calls.clone()));
    let steering_polls = AtomicUsize::new(0);
    config.get_steering_messages = Some(source(move || {
        let polls = steering_polls.fetch_add(1, Ordering::SeqCst) + 1;
        if steering && polls == 2 {
            vec![user(queued)]
        } else {
            Vec::new()
        }
    }));
    let follow_up_delivered = AtomicBool::new(false);
    config.get_follow_up_messages = Some(source(move || {
        if steering || follow_up_delivered.swap(true, Ordering::SeqCst) {
            Vec::new()
        } else {
            vec![user(queued)]
        }
    }));
    let (calls, recorded) = (provider_calls.clone(), second_request_users.clone());
    let stream = agent_loop(
        vec![user("run")],
        AgentContext::default(),
        config,
        CancellationToken::new(),
        stream_fn(move |index, _, context, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            if index == 1 {
                lock(&recorded).extend(user_texts(context));
            }
            respond(assistant_text("done"))
        }),
    );
    stream.result().await;

    assert_eq!(provider_calls.load(Ordering::SeqCst), 2);
    assert_eq!(finish_calls.load(Ordering::SeqCst), 2);
    assert!(lock(&second_request_users).contains(&queued.to_owned()));
}

#[tokio::test]
async fn lets_a_natural_steering_request_satisfy_continuation() {
    natural_queued_request_satisfies_continuation(true).await;
}

#[tokio::test]
async fn lets_a_natural_follow_up_request_satisfy_continuation() {
    natural_queued_request_satisfies_continuation(false).await;
}

#[tokio::test]
async fn prepares_the_initial_request_after_pending_messages_and_can_replace_request_state() {
    let replacement = delta_ai::Model {
        id: "replacement".into(),
        name: "replacement".into(),
        ..model()
    };
    let canonical = user_message("canonical projection");
    let steering = user_message("steering");
    let completed = shared(Vec::<AgentMessage>::new());
    let prepare_calls = Arc::new(AtomicUsize::new(0));
    let (calls, seen, expected_steering) = (prepare_calls.clone(), completed.clone(), steering.clone());
    let (replacement_model, canonical_message) = (replacement.clone(), canonical.clone());
    let hooks = TestHooks::new().prepare_request(move |request| {
        calls.fetch_add(1, Ordering::SeqCst);
        let steering = AgentMessage::from(expected_steering.clone());
        assert!(lock(&seen).contains(&steering));
        assert!(request.context.messages.contains(&steering));
        Some(AgentRequestUpdate {
            context: Some(AgentContext {
                messages: vec![canonical_message.clone().into()],
                tools: request.context.tools.clone(),
            }),
            model: Some(replacement_model.clone()),
            thinking_level: Some(ModelThinkingLevel::High),
        })
    });
    let mut config = config(hooks);
    let delivered = AtomicBool::new(false);
    config.get_steering_messages = Some(source(move || {
        if delivered.swap(true, Ordering::SeqCst) {
            Vec::new()
        } else {
            vec![steering.clone().into()]
        }
    }));
    let recorded = completed.clone();
    let sink = FnSink(move |event: AgentEvent| {
        if let AgentEvent::MessageEnd { message } = event {
            lock(&recorded).push(message);
        }
    });
    run_agent_loop(
        vec![user("prompt")],
        AgentContext::default(),
        config,
        &sink,
        CancellationToken::new(),
        &stream_fn(move |_, model, context, options| {
            assert_eq!(model.id, "replacement");
            assert_eq!(context.messages(), [Message::User(canonical.clone())]);
            assert_eq!(options.reasoning, Some(ThinkingLevel::High));
            respond(assistant_text("done"))
        }),
    )
    .await;

    assert_eq!(prepare_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn does_not_poll_steering_after_prepare_request() {
    let queue = shared(Vec::<AgentMessage>::new());
    let included = shared(Vec::new());
    let preparations = Arc::new(AtomicUsize::new(0));
    let steering_polls = Arc::new(AtomicUsize::new(0));
    let (late_queue, prepared) = (queue.clone(), preparations.clone());
    let mut config = config(TestHooks::new().prepare_request(move |_| {
        if prepared.fetch_add(1, Ordering::SeqCst) == 0 {
            lock(&late_queue).push(user("late steering"));
        }
        None
    }));
    let (polls, drained) = (steering_polls.clone(), queue.clone());
    config.get_steering_messages = Some(source(move || {
        polls.fetch_add(1, Ordering::SeqCst);
        lock(&drained).drain(..).collect()
    }));
    let recorded = included.clone();
    let stream = agent_loop(
        vec![user("run")],
        AgentContext::default(),
        config,
        CancellationToken::new(),
        stream_fn(move |_, _, context, _| {
            lock(&recorded).push(user_texts(context).contains(&"late steering".to_owned()));
            respond(assistant_text("done"))
        }),
    );
    stream.result().await;

    assert_eq!(*lock(&included), [false, true]);
    assert_eq!(preparations.load(Ordering::SeqCst), 2);
    // Startup, post-turn delivery, then the final natural-stop check.
    assert_eq!(steering_polls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn uses_prepare_next_turn_snapshot_before_continuing() {
    let prepare_calls = Arc::new(AtomicUsize::new(0));
    let llm_calls = Arc::new(AtomicUsize::new(0));
    let second_turn_has_update = Arc::new(AtomicBool::new(false));
    let calls = prepare_calls.clone();
    let hooks = TestHooks::new().prepare_next_turn(move |turn, _| {
        if calls.fetch_add(1, Ordering::SeqCst) > 0 {
            return None;
        }
        Some(AgentLoopTurnUpdate {
            context: Some(turn.context.clone()),
            messages: Some(vec![
                SystemMessage {
                    content: "updated guidance".into(),
                    timestamp: 1,
                    ..SystemMessage::default()
                }
                .into(),
            ]),
            ..AgentLoopTurnUpdate::default()
        })
    });
    let (requests, has_update) = (llm_calls.clone(), second_turn_has_update.clone());
    let first = tool_call_message(vec![("tool-1", "echo", json!({"value": "hello"}))]);
    let stream = agent_loop(
        vec![user("echo something")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![echo_tool(shared(Vec::new()))],
        },
        config(hooks),
        CancellationToken::new(),
        stream_fn(move |index, _, context, _| {
            requests.fetch_add(1, Ordering::SeqCst);
            if index == 1 {
                has_update.store(
                    context.messages().iter().any(|message| match message {
                        Message::System(system) => system.content.text() == "updated guidance",
                        _ => false,
                    }),
                    Ordering::SeqCst,
                );
            }
            respond(if index == 0 {
                first.clone()
            } else {
                assistant_text("done")
            })
        }),
    );
    collect(stream).await;

    assert_eq!(llm_calls.load(Ordering::SeqCst), 2);
    assert_eq!(prepare_calls.load(Ordering::SeqCst), 1);
    assert!(second_turn_has_update.load(Ordering::SeqCst));
}

#[tokio::test]
async fn picks_up_steering_queued_during_prepare_next_turn_before_the_next_request() {
    let queue = shared(Vec::<AgentMessage>::new());
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let second_included = Arc::new(AtomicBool::new(false));
    let late_queue = queue.clone();
    let mut config = config(TestHooks::new().prepare_next_turn(move |_, _| {
        lock(&late_queue).push(user("late steering"));
        None
    }));
    let drained = queue.clone();
    config.get_steering_messages = Some(source(move || lock(&drained).drain(..).collect()));
    let (calls, included) = (provider_calls.clone(), second_included.clone());
    let first = tool_call_message(vec![("tool-1", "noop", json!({}))]);
    let stream = agent_loop(
        vec![user("run")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![noop_tool("noop")],
        },
        config,
        CancellationToken::new(),
        stream_fn(move |index, _, context, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            if index == 1 {
                included.store(
                    user_texts(context).contains(&"late steering".to_owned()),
                    Ordering::SeqCst,
                );
            }
            respond(if index == 0 {
                first.clone()
            } else {
                assistant_text("done")
            })
        }),
    );
    stream.result().await;

    assert_eq!(provider_calls.load(Ordering::SeqCst), 2);
    assert!(second_included.load(Ordering::SeqCst));
}

#[tokio::test]
async fn action_end_receives_finalized_turn_context_and_stops_before_queue_polling() {
    let executed = shared(Vec::new());
    let steering_polls = Arc::new(AtomicUsize::new(0));
    let follow_up_polls = Arc::new(AtomicUsize::new(0));
    let callback_result_ids = shared(Vec::<String>::new());
    let callback_roles = shared(Vec::<String>::new());
    let (result_ids, context_roles) = (callback_result_ids.clone(), callback_roles.clone());
    let mut config = config(TestHooks::new().finish_turn(move |turn, _| {
        *lock(&result_ids) = turn
            .tool_results
            .iter()
            .map(|result| result.tool_call_id.clone())
            .collect();
        *lock(&context_roles) = roles(&turn.context.messages);
        Some(AgentTurnDecision::End)
    }));
    let polls = steering_polls.clone();
    config.get_steering_messages = Some(source(move || {
        polls.fetch_add(1, Ordering::SeqCst);
        Vec::new()
    }));
    let polls = follow_up_polls.clone();
    config.get_follow_up_messages = Some(source(move || {
        polls.fetch_add(1, Ordering::SeqCst);
        vec![user("follow up should stay queued")]
    }));
    let llm_calls = Arc::new(AtomicUsize::new(0));
    let stream = agent_loop(
        vec![user("echo something")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![echo_tool(executed.clone())],
        },
        config,
        CancellationToken::new(),
        first_then_done(
            tool_call_message(vec![("tool-1", "echo", json!({"value": "hello"}))]),
            llm_calls.clone(),
        ),
    );
    let (events, messages) = collect(stream).await;

    assert_eq!(llm_calls.load(Ordering::SeqCst), 1);
    assert_eq!(*lock(&executed), [json!("hello")]);
    assert_eq!(steering_polls.load(Ordering::SeqCst), 1);
    assert_eq!(follow_up_polls.load(Ordering::SeqCst), 0);
    assert_eq!(*lock(&callback_result_ids), ["tool-1"]);
    assert_eq!(*lock(&callback_roles), ["system", "user", "assistant", "toolResult"]);
    // The context declares no tools, so the loop announces the loadout with a system message.
    assert_eq!(roles(&messages), ["system", "user", "assistant", "toolResult"]);
    assert_eq!(
        kinds(&events),
        [
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "tool_execution_start",
            "tool_execution_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end",
        ]
    );
}

/// Every request answers with the same tool call; a second request would answer "done".
fn same_tool_call_then_done(calls: Arc<AtomicUsize>) -> delta_agent::StreamFn {
    first_then_done(
        tool_call_message(vec![("tool-1", "echo", json!({"value": "hello"}))]),
        calls,
    )
}

#[tokio::test]
async fn stops_after_a_tool_batch_when_every_tool_result_sets_terminate() {
    let llm_calls = Arc::new(AtomicUsize::new(0));
    let stream = agent_loop(
        vec![user("echo something")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![terminating_echo()],
        },
        config(TestHooks::new()),
        CancellationToken::new(),
        same_tool_call_then_done(llm_calls.clone()),
    );
    let (events, messages) = collect(stream).await;

    assert_eq!(llm_calls.load(Ordering::SeqCst), 1);
    assert_eq!(roles(&messages), ["system", "user", "assistant", "toolResult"]);
    assert_eq!(events.iter().filter(|event| event.kind() == "turn_end").count(), 1);
}

#[tokio::test]
async fn stops_after_a_blocked_tool_call_when_before_tool_call_sets_terminate() {
    let executed = shared(Vec::new());
    let llm_calls = Arc::new(AtomicUsize::new(0));
    let hooks = TestHooks::new().before_tool_call(|_| {
        Some(BeforeToolCallResult {
            block: true,
            reason: Some("Blocked by policy".into()),
            terminate: true,
        })
    });
    let stream = agent_loop(
        vec![user("echo something")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![echo_tool(executed.clone())],
        },
        config(hooks),
        CancellationToken::new(),
        same_tool_call_then_done(llm_calls.clone()),
    );
    let (_, messages) = collect(stream).await;

    let result = &tool_results(&messages)[0];
    assert!(lock(&executed).is_empty());
    assert_eq!(llm_calls.load(Ordering::SeqCst), 1);
    assert!(result.is_error);
    assert!(result.content.contains(&UserBlock::text("Blocked by policy")));
}

#[tokio::test]
async fn continues_after_a_mixed_batch_with_one_terminating_blocked_call() {
    let executed = shared(Vec::new());
    let llm_calls = Arc::new(AtomicUsize::new(0));
    let mut config = config(TestHooks::new().before_tool_call(|context| {
        (context.args["value"] == "first").then(|| BeforeToolCallResult {
            block: true,
            reason: Some("Blocked first".into()),
            terminate: true,
        })
    }));
    config.tool_execution = ToolExecutionMode::Parallel;
    let stream = agent_loop(
        vec![user("echo both")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![echo_tool(executed.clone())],
        },
        config,
        CancellationToken::new(),
        first_then_done(
            tool_call_message(vec![
                ("tool-1", "echo", json!({"value": "first"})),
                ("tool-2", "echo", json!({"value": "second"})),
            ]),
            llm_calls.clone(),
        ),
    );
    collect(stream).await;

    assert_eq!(*lock(&executed), [json!("second")]);
    assert_eq!(llm_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn continues_after_parallel_tool_calls_when_not_all_tool_results_terminate() {
    let echo = tool("echo", value_schema(), |args, _| async move {
        Ok(AgentToolResult {
            terminate: args["value"] == "first",
            ..AgentToolResult::text(format!("echoed: {}", args["value"].as_str().unwrap()))
        })
    });
    let llm_calls = Arc::new(AtomicUsize::new(0));
    let mut config = config(TestHooks::new());
    config.tool_execution = ToolExecutionMode::Parallel;
    let stream = agent_loop(
        vec![user("echo both")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![echo],
        },
        config,
        CancellationToken::new(),
        first_then_done(
            tool_call_message(vec![
                ("tool-1", "echo", json!({"value": "first"})),
                ("tool-2", "echo", json!({"value": "second"})),
            ]),
            llm_calls.clone(),
        ),
    );
    let (_, messages) = collect(stream).await;

    assert_eq!(llm_calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        roles(&messages),
        ["system", "user", "assistant", "toolResult", "toolResult", "assistant"]
    );
}

#[tokio::test]
async fn allows_after_tool_call_to_mark_a_tool_batch_as_terminating() {
    let llm_calls = Arc::new(AtomicUsize::new(0));
    let hooks = TestHooks::new().after_tool_call(|_| {
        Some(AfterToolCallResult {
            terminate: Some(true),
            ..AfterToolCallResult::default()
        })
    });
    let stream = agent_loop(
        vec![user("echo something")],
        AgentContext {
            messages: Vec::new(),
            tools: vec![echo_tool(shared(Vec::new()))],
        },
        config(hooks),
        CancellationToken::new(),
        same_tool_call_then_done(llm_calls.clone()),
    );
    collect(stream).await;

    assert_eq!(llm_calls.load(Ordering::SeqCst), 1);
}

// agent_loop_continue

#[tokio::test]
async fn continue_rejects_a_context_without_messages() {
    let result = agent_loop_continue(
        AgentContext::<NoCustom>::default(),
        config(TestHooks::new()),
        CancellationToken::new(),
        stream_fn(|_, _, _, _| panic!("Unexpected stream call")),
    );
    let Err(error) = result else {
        panic!("expected an error")
    };
    assert_eq!(error, AgentLoopError::NoMessages);
    assert_eq!(error.to_string(), "Cannot continue: no messages in context");
}

#[tokio::test]
async fn continues_from_existing_context_without_emitting_user_message_events() {
    let stream = agent_loop_continue(
        AgentContext::<NoCustom> {
            messages: vec![user("Hello")],
            tools: Vec::new(),
        },
        config(TestHooks::new()),
        CancellationToken::new(),
        scripted(vec![assistant_text("Response")]),
    )
    .unwrap();
    let (events, messages) = collect(stream).await;

    // Only the new assistant message, not the existing user message.
    assert_eq!(roles(&messages), ["assistant"]);
    let ends: Vec<String> = events.iter().filter_map(message_role).collect();
    assert_eq!(ends, ["assistant"]);
}

#[tokio::test]
async fn continue_allows_custom_message_types_as_last_message() {
    // The caller converts the custom message to a user message.
    let hooks = TestHooks::<Custom>::default().convert_to_llm(|messages: &[AgentMessage<Custom>]| {
        messages
            .iter()
            .filter_map(|message| match message {
                AgentMessage::Custom(custom) if custom.role == "custom" => Some(Message::User(delta_ai::UserMessage {
                    content: custom.text.clone().into(),
                    timestamp: custom.timestamp,
                })),
                AgentMessage::Custom(_) => None,
                AgentMessage::Llm(message) => Some(message.clone()),
            })
            .collect()
    });
    let stream = agent_loop_continue(
        AgentContext {
            messages: vec![custom("custom", "Hook content")],
            tools: Vec::new(),
        },
        config(hooks),
        CancellationToken::new(),
        scripted(vec![assistant_text("Response to custom message")]),
    )
    .unwrap();
    let (_, messages) = collect(stream).await;

    assert_eq!(roles(&messages), ["assistant"]);
}

// run_tool_call

/// `echo` sends one partial update and returns structured content.
fn structured_echo() -> Arc<dyn AgentTool> {
    tool("echo", value_schema(), |args, context: ToolContext| async move {
        context.updates.send(AgentToolResult {
            details: Some(json!({})),
            ..AgentToolResult::text("partial")
        });
        Ok(AgentToolResult {
            details: Some(json!({})),
            structured_content: Some(json!({ "value": args["value"] })),
            ..AgentToolResult::text(args["value"].as_str().unwrap())
        })
    })
}

#[tokio::test]
async fn run_tool_call_validates_runs_the_hooks_and_reports_failures_as_error_outcomes() {
    let failing = tool("failing", empty_schema(), |_, _| async {
        Ok(AgentToolResult {
            details: Some(json!({ "partial": true })),
            is_error: true,
            ..AgentToolResult::text("bad")
        })
    });
    let tools = vec![structured_echo(), failing];
    let hook_calls = shared(Vec::<String>::new());
    let (before_calls, after_calls) = (hook_calls.clone(), hook_calls.clone());
    let config = config(
        TestHooks::new()
            .before_tool_call(move |context| {
                lock(&before_calls).push(format!("before {}", context.tool_call.id));
                (context.args["value"] == "blocked").then(|| BeforeToolCallResult {
                    block: true,
                    reason: Some("nope".into()),
                    terminate: false,
                })
            })
            .after_tool_call(move |context| {
                lock(&after_calls).push(format!("after {}", context.tool_call.id));
                None
            }),
    );
    let updates = shared(Vec::new());
    let recorded = updates.clone();
    let on_update = move |partial: AgentToolResult| lock(&recorded).push(partial);
    let assistant_message = assistant(Vec::new(), StopReason::Stop);
    let context = AgentContext::default();
    let run = async |id: &str, name: &str, arguments: Value| {
        run_tool_call(
            &call(id, name, arguments),
            RunToolCallOptions {
                tools: &tools,
                assistant_message: &assistant_message,
                context: &context,
                config: &config,
                cancel: CancellationToken::new(),
                on_update: Some(&on_update),
            },
        )
        .await
    };

    let ran = run("a", "echo", json!({"value": "a"})).await;
    assert_eq!(ran.tool_call.id, "a");
    assert_eq!(ran.result.structured_content, Some(json!({"value": "a"})));
    assert!(!ran.is_error);
    assert!(run("b", "echo", json!({"value": {"nested": true}})).await.is_error);
    let blocked = run("c", "echo", json!({"value": "blocked"})).await;
    assert_eq!(blocked.result.content, [UserBlock::text("nope")]);
    assert!(blocked.is_error);
    let missing = run("d", "missing", json!({})).await;
    assert_eq!(missing.result.content, [UserBlock::text("Tool missing not found")]);
    assert!(missing.is_error);
    // Error results keep their details.
    let failed = run("e", "failing", json!({})).await;
    assert_eq!(failed.result.details, Some(json!({"partial": true})));
    assert!(failed.is_error);

    assert_eq!(
        *lock(&updates),
        [AgentToolResult {
            details: Some(json!({})),
            ..AgentToolResult::text("partial")
        }]
    );
    // Validation failures and unknown tools never reach the hooks; blocked calls skip
    // after_tool_call.
    assert_eq!(
        *lock(&hook_calls),
        ["before a", "after a", "before c", "before e", "after e"]
    );
}

#[tokio::test]
async fn run_tool_call_lets_after_tool_call_replace_structured_content_and_drops_it_when_only_content_is_replaced() {
    let redacted = vec![UserBlock::text("redacted")];
    let overrides = [
        AfterToolCallResult {
            content: Some(redacted.clone()),
            ..AfterToolCallResult::default()
        },
        AfterToolCallResult {
            structured_content: Some(json!({"value": "replaced"})),
            ..AfterToolCallResult::default()
        },
        AfterToolCallResult {
            content: Some(redacted),
            structured_content: Some(json!({"value": "both"})),
            ..AfterToolCallResult::default()
        },
        AfterToolCallResult {
            details: Some(json!({"note": "kept"})),
            ..AfterToolCallResult::default()
        },
    ];
    let tools = vec![structured_echo()];
    let assistant_message = assistant(Vec::new(), StopReason::Stop);
    let context = AgentContext::default();
    let mut seen = Vec::new();
    for after in overrides {
        let config = config(TestHooks::new().after_tool_call(move |_| Some(after.clone())));
        let outcome = run_tool_call(
            &call("x", "echo", json!({"value": "original"})),
            RunToolCallOptions {
                tools: &tools,
                assistant_message: &assistant_message,
                context: &context,
                config: &config,
                cancel: CancellationToken::new(),
                on_update: None,
            },
        )
        .await;
        seen.push(outcome.result.structured_content);
    }

    assert_eq!(
        seen,
        [
            None,
            Some(json!({"value": "replaced"})),
            Some(json!({"value": "both"})),
            Some(json!({"value": "original"})),
        ]
    );
}
