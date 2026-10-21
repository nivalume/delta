//! Nested calls: results returned to the caller, keys, reattachment after a crash and
//! abandonment under a caller that is not replay-safe.

use std::sync::{Arc, Mutex};

use delta_ai::types::{JsonObject, Tool};
use delta_harness::documents::DocAddress;
use delta_harness::harness::Harness;
use delta_harness::inbox::SubmissionDraft;
use delta_harness::records::{SubmissionState, TaskId};
use delta_harness::registry::Extension;
use delta_harness::tool::{
    FnHarnessTool, HarnessTool, NestedCallDoc, NestedCallOptions, ToolExecutionResult, ToolReplay,
};

use crate::support::fixture::{Fixture, PATIENCE, chat, ctx, entries, eventually, live, settled};
use crate::support::model::Reply;
use crate::{Backend, results, run, texts};

cases!(
    a_nested_call_returns_to_its_caller,
    keys_reattach_reject_and_mismatch,
    a_safe_caller_reattaches_after_a_crash,
    an_unsafe_callers_nested_call_is_abandoned_after_a_crash,
);

/// `caller`: makes the nested calls listed in its `calls` argument (`{ tool, args, key? }`)
/// in order and answers one JSON line per call: the result's task, error flag, value and
/// diagnostic codes, or the error.
pub(crate) fn caller(replay: ToolReplay, runs: &Arc<Mutex<u32>>) -> Arc<dyn HarnessTool> {
    let runs = runs.clone();
    let tool = FnHarnessTool::new(
        Tool::new("caller", "Calls tools", serde_json::json!({ "type": "object" })),
        move |arguments, api, ctx| {
            *runs.lock().unwrap() += 1;
            Box::pin(async move {
                let calls = arguments
                    .get("calls")
                    .and_then(serde_json::Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let mut lines = Vec::new();
                for call in calls {
                    let name = call["tool"].as_str().unwrap_or_default();
                    let args: JsonObject = call["args"].as_object().cloned().unwrap_or_default();
                    let options = NestedCallOptions {
                        key: call["key"].as_str().map(str::to_owned),
                        progress: None,
                    };
                    let line = match api.execute_tool(name, args, options, ctx).await {
                        Ok(result) => {
                            let codes: Vec<Option<String>> =
                                result.diagnostics.iter().map(|d| d.code.clone()).collect();
                            serde_json::json!({
                                "task": result.task_id.0,
                                "is_error": result.is_error,
                                "value": result.structured_output,
                                "codes": codes,
                            })
                        }
                        Err(error) => serde_json::json!({ "error": error.to_string() }),
                    };
                    lines.push(line.to_string());
                }
                Ok(ToolExecutionResult::text(lines.join("\n")))
            })
        },
    )
    .with_replay(replay);
    Arc::new(tool)
}

/// `lookup`: answers `found <q>`.
fn lookup() -> Arc<dyn HarnessTool> {
    Arc::new(FnHarnessTool::new(
        Tool::new("lookup", "Looks up", serde_json::json!({ "type": "object" })),
        |arguments, _api, _ctx| {
            Box::pin(async move {
                let q = arguments.get("q").and_then(serde_json::Value::as_str).unwrap_or("");
                Ok(ToolExecutionResult::text(format!("found {q}")))
            })
        },
    ))
}

/// The lines the caller answered, parsed.
async fn lines(fixture: &Fixture, harness: &Harness, calls: serde_json::Value) -> Vec<serde_json::Value> {
    fixture.model.push([
        Reply::calls_with(&[("c1", "caller", serde_json::json!({ "calls": calls }))]),
        Reply::text("Done"),
    ]);
    let conversation = chat(harness).await;
    assert!(matches!(run(&conversation, "go").await, SubmissionState::Done { .. }));
    let results = results(&conversation).await;
    let text = texts(results.last().unwrap()).join("");
    text.lines().map(|line| serde_json::from_str(line).unwrap()).collect()
}

async fn a_nested_call_returns_to_its_caller(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    fixture
        .registry
        .install(
            Extension::new("tools")
                .with_tool(caller(ToolReplay::Unsafe, &runs))
                .with_tool(lookup()),
        )
        .unwrap();
    let harness = fixture.open().await;
    let calls = serde_json::json!([{ "tool": "lookup", "args": { "q": "keys" } }]);
    let lines = lines(&fixture, &harness, calls).await;
    assert_eq!(lines[0]["value"], "found keys");
    assert_eq!(lines[0]["is_error"], false);
    let conversation = chat(&harness).await;
    // Only the caller's result is in the transcript.
    assert_eq!(results(&conversation).await.len(), 1);
    let transcript = entries(&conversation).await;
    let entry = transcript
        .iter()
        .find(|entry| entry.kind == "delta.tool-result")
        .unwrap();
    let parent = entry.by_task_id.unwrap();
    let child = TaskId(lines[0]["task"].as_u64().unwrap());
    let record = harness.get_task(child, &ctx()).await.unwrap().unwrap();
    assert_eq!((record.kind.as_str(), record.owner), ("delta.tool", Some(parent)));
    assert!(live(&harness, &conversation).await.nested_tools.is_none());
    harness.close(&ctx()).await.unwrap();
}

async fn keys_reattach_reject_and_mismatch(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    fixture
        .registry
        .install(
            Extension::new("tools")
                .with_tool(caller(ToolReplay::Unsafe, &runs))
                .with_tool(lookup()),
        )
        .unwrap();
    let harness = fixture.open().await;
    let calls = serde_json::json!([
        { "tool": "lookup", "args": { "q": "a" }, "key": "k" },
        { "tool": "lookup", "args": { "q": "a" }, "key": "k" },
        { "tool": "lookup", "args": { "q": "b" }, "key": "k" },
        { "tool": "lookup", "args": { "q": "c" }, "key": "12" },
        { "tool": "lookup", "args": { "q": "d" } },
    ]);
    let lines = lines(&fixture, &harness, calls).await;
    assert_eq!(lines[0]["value"], "found a");
    assert_eq!(lines[1]["task"], lines[0]["task"], "the same key is the same call");
    assert!(
        lines[2]["error"]
            .as_str()
            .unwrap()
            .contains("another tool or other arguments")
    );
    assert!(
        lines[3]["error"]
            .as_str()
            .unwrap()
            .contains("Invalid nested call key \"12\"")
    );
    // The fifth call's default key is its position.
    assert_eq!(lines[4]["value"], "found d");
    let conversation = chat(&harness).await;
    let transcript = entries(&conversation).await;
    let parent = transcript
        .iter()
        .find(|entry| entry.kind == "delta.tool-result")
        .unwrap()
        .by_task_id
        .unwrap();
    let member = harness
        .snapshot(DocAddress::<NestedCallDoc>::member(parent, "5"), &ctx())
        .await
        .unwrap();
    // Task documents retire with their task.
    assert!(member.is_none());
    harness.close(&ctx()).await.unwrap();
}

/// `slow`: on its first `stuck_runs` executions reports a step and waits until its context
/// ends; then answers `slow done`.
fn slow(replay: ToolReplay, runs: &Arc<Mutex<u32>>, stuck_runs: u32) -> Arc<dyn HarnessTool> {
    let runs = runs.clone();
    Arc::new(
        FnHarnessTool::new(
            Tool::new("slow", "Takes a while", serde_json::json!({ "type": "object" })),
            move |_arguments, api, ctx| {
                let run = {
                    let mut runs = runs.lock().unwrap();
                    *runs += 1;
                    *runs
                };
                Box::pin(async move {
                    if run > stuck_runs {
                        return Ok(ToolExecutionResult::text("slow done"));
                    }
                    api.details(serde_json::json!({ "run": run }), ctx).await?;
                    ctx.run(std::future::pending::<()>()).await?;
                    Ok(ToolExecutionResult::default())
                })
            },
        )
        .with_replay(replay),
    )
}

/// Starts one caller call of `slow`, waits until the nested call published its step, and
/// crashes; returns the submission's id.
async fn crash_mid_nested_call(fixture: &Fixture) -> delta_harness::records::SubmissionId {
    let calls = serde_json::json!([{ "tool": "slow", "args": {} }]);
    fixture.model.push([Reply::calls_with(&[(
        "c1",
        "caller",
        serde_json::json!({ "calls": calls }),
    )])]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    let id = submission.id();
    eventually(|| async {
        live(&harness, &conversation)
            .await
            .nested_tools
            .into_iter()
            .flatten()
            .any(|slot| slot.progress.details.is_some() && slot.call_id == "c1/1")
    })
    .await;
    let closed = harness.session().closed();
    drop((submission, conversation, harness));
    tokio::time::timeout(PATIENCE, closed)
        .await
        .expect("the old Session ends");
    id
}

async fn a_safe_caller_reattaches_after_a_crash(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let (caller_runs, slow_runs) = (Arc::new(Mutex::new(0)), Arc::new(Mutex::new(0)));
    fixture
        .registry
        .install(
            Extension::new("tools")
                .with_tool(caller(ToolReplay::Safe, &caller_runs))
                .with_tool(slow(ToolReplay::Safe, &slow_runs, 1)),
        )
        .unwrap();
    let id = crash_mid_nested_call(&fixture).await;

    fixture.model.push([Reply::text("Done")]);
    let harness = fixture.open().await;
    let submission = harness.submission(id, &ctx()).await.unwrap().unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    let conversation = chat(&harness).await;
    let results = results(&conversation).await;
    let line: serde_json::Value = serde_json::from_str(&texts(&results[0]).join("")).unwrap();
    assert_eq!(line["value"], "slow done");
    assert_eq!(*caller_runs.lock().unwrap(), 2);
    assert_eq!(
        *slow_runs.lock().unwrap(),
        2,
        "the nested call reran once, under one task"
    );
    let inspection = harness.inspect(&ctx()).await.unwrap();
    assert!(inspection.scheduler.tasks.is_empty(), "{inspection:?}");
    harness.close(&ctx()).await.unwrap();
}

async fn an_unsafe_callers_nested_call_is_abandoned_after_a_crash(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let (caller_runs, slow_runs) = (Arc::new(Mutex::new(0)), Arc::new(Mutex::new(0)));
    fixture
        .registry
        .install(
            Extension::new("tools")
                .with_tool(caller(ToolReplay::Unsafe, &caller_runs))
                .with_tool(slow(ToolReplay::Safe, &slow_runs, 1)),
        )
        .unwrap();
    let id = crash_mid_nested_call(&fixture).await;

    fixture.model.push([Reply::text("Done")]);
    let harness = fixture.open().await;
    let submission = harness.submission(id, &ctx()).await.unwrap().unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    let conversation = chat(&harness).await;
    let results = results(&conversation).await;
    assert_eq!(
        texts(&results[0]),
        ["<harness>\n[error] Tool caller was interrupted and may have partially run\n</harness>"]
    );
    assert_eq!(*caller_runs.lock().unwrap(), 1);
    assert_eq!(*slow_runs.lock().unwrap(), 1, "the abandoned nested call did not rerun");
    let state = live(&harness, &conversation).await;
    assert!(state.nested_tools.is_none());
    harness.close(&ctx()).await.unwrap();
}
