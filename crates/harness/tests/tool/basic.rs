//! A call from resolution to its result entry: arguments repaired, coerced or rejected, an
//! error from `execute`, and what the next request sees.

use std::sync::Arc;

use delta_ai::types::{Message, Tool};
use delta_harness::records::{SubmissionState, TaskOutcome};
use delta_harness::registry::Extension;
use delta_harness::tool::{FnHarnessTool, HarnessTool, ToolExecutionResult};

use crate::support::fixture::{Fixture, chat, ctx, entries, live};
use crate::support::model::Reply;
use crate::{Backend, results, run, texts};

cases!(
    a_call_runs_and_its_result_reaches_the_next_request,
    arguments_are_repaired_coerced_or_rejected,
    an_error_from_execute_fails_the_task,
);

/// `add`: `{ a: integer, b: integer }`; repairs `{ "pair": [a, b] }`.
fn add() -> Arc<dyn HarnessTool> {
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "a": { "type": "integer" }, "b": { "type": "integer" } },
        "required": ["a", "b"],
    });
    let tool = FnHarnessTool::new(Tool::new("add", "Add two numbers", schema), |arguments, _api, _ctx| {
        Box::pin(async move {
            let number = |key: &str| arguments.get(key).and_then(serde_json::Value::as_i64).unwrap_or(0);
            Ok(ToolExecutionResult::text((number("a") + number("b")).to_string()))
        })
    })
    .with_prepare_arguments(|mut arguments| {
        if arguments.contains_key("broken") {
            return Err("cannot repair".into());
        }
        if let Some(serde_json::Value::Array(pair)) = arguments.remove("pair") {
            arguments.insert("a".to_owned(), pair[0].clone());
            arguments.insert("b".to_owned(), pair[1].clone());
        }
        Ok(arguments)
    });
    Arc::new(tool)
}

async fn a_call_runs_and_its_result_reaches_the_next_request(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture
        .registry
        .install(Extension::new("math").with_tool(add()))
        .unwrap();
    fixture.model.push([
        Reply::calls_with(&[("c1", "add", serde_json::json!({ "a": 2, "b": 3 }))]),
        Reply::text("It is 5"),
    ]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    assert!(matches!(run(&conversation, "add").await, SubmissionState::Done { .. }));
    let results = results(&conversation).await;
    assert_eq!(results.len(), 1);
    assert_eq!(texts(&results[0]), ["5"]);
    assert!(!results[0].is_error);
    assert!(results[0].duration_ms.is_some());
    let transcript = entries(&conversation).await;
    let entry = transcript
        .iter()
        .find(|entry| entry.kind == "delta.tool-result")
        .unwrap();
    let task = harness
        .get_task(entry.by_task_id.unwrap(), &ctx())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.state.outcome(), Some(TaskOutcome::Completed { .. })));
    // The second request carries the result after its call.
    let requests = fixture.model.requests();
    let last = requests[1].messages.last().unwrap();
    assert!(matches!(last, Message::ToolResult(result) if result.tool_call_id == "c1"));
    let state = live(&harness, &conversation).await;
    assert!(state.tools.is_none() && state.nested_tools.is_none());
    harness.close(&ctx()).await.unwrap();
}

async fn arguments_are_repaired_coerced_or_rejected(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture
        .registry
        .install(Extension::new("math").with_tool(add()))
        .unwrap();
    fixture.model.push([
        Reply::calls_with(&[
            ("c1", "add", serde_json::json!({ "pair": [1, 2] })),
            ("c2", "add", serde_json::json!({ "a": "5", "b": 1 })),
            ("c3", "add", serde_json::json!({ "a": 1 })),
            ("c4", "add", serde_json::json!({ "broken": true })),
            ("c5", "subtract", serde_json::json!({})),
        ]),
        Reply::text("Done"),
    ]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    run(&conversation, "add").await;
    let results = results(&conversation).await;
    let by_call = |id: &str| results.iter().find(|result| result.tool_call_id == id).unwrap();
    assert_eq!(texts(by_call("c1")), ["3"]);
    assert_eq!(texts(by_call("c2")), ["6"]);
    for id in ["c3", "c4"] {
        let result = by_call(id);
        assert!(result.is_error);
        assert!(result.duration_ms.is_none(), "rejected before intent");
        let text = texts(result).join("");
        assert!(text.starts_with("<harness>\n[error] "), "{text}");
    }
    assert!(
        texts(by_call("c3"))
            .join("")
            .contains("Validation failed for tool \"add\"")
    );
    assert!(texts(by_call("c4")).join("").contains("cannot repair"));
    assert_eq!(
        texts(by_call("c5")),
        ["<harness>\n[error] Tool subtract is not available\n</harness>"]
    );
    harness.close(&ctx()).await.unwrap();
}

async fn an_error_from_execute_fails_the_task(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let failing = FnHarnessTool::new(
        Tool::new("fail", "Fails", serde_json::json!({ "type": "object" })),
        |_arguments, api, _ctx| {
            Box::pin(async move {
                api.output_str("partial output\n");
                Err("disk full".into())
            })
        },
    );
    fixture
        .registry
        .install(Extension::new("broken").with_tool(Arc::new(failing)))
        .unwrap();
    fixture
        .model
        .push([Reply::calls(&[("c1", "fail")]), Reply::text("It failed")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    assert!(matches!(run(&conversation, "fail").await, SubmissionState::Done { .. }));
    let results = results(&conversation).await;
    assert!(results[0].is_error);
    assert_eq!(
        texts(&results[0]),
        ["partial output\n", "<harness>\n[error] disk full\n</harness>"]
    );
    let transcript = entries(&conversation).await;
    let entry = transcript
        .iter()
        .find(|entry| entry.kind == "delta.tool-result")
        .unwrap();
    let task = harness
        .get_task(entry.by_task_id.unwrap(), &ctx())
        .await
        .unwrap()
        .unwrap();
    let Some(TaskOutcome::Failed { error, result }) = task.state.outcome() else {
        panic!("failed: {:?}", task.state)
    };
    assert_eq!(error.message, "disk full");
    assert!(result.is_some(), "a failed call still names its result entry");
    harness.close(&ctx()).await.unwrap();
}
