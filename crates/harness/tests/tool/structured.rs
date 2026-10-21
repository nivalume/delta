//! The structured output contract, for nested and model-issued calls.

use std::sync::{Arc, Mutex};

use delta_ai::types::Tool;
use delta_harness::registry::Extension;
use delta_harness::tool::{FnHarnessTool, HarnessTool, ToolExecutionResult, ToolReplay};

use crate::nested::caller;
use crate::support::fixture::{Fixture, chat, ctx};
use crate::support::model::Reply;
use crate::{Backend, results, run, texts};

cases!(
    nested_results_follow_the_schema_contract,
    a_model_issued_call_only_loses_a_broken_value,
);

/// `count`: declares `{ n: integer }` and returns its `value` argument as structured output
/// (or none without one).
fn count() -> Arc<dyn HarnessTool> {
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "n": { "type": "integer" } },
        "required": ["n"],
    });
    Arc::new(
        FnHarnessTool::new(
            Tool::new("count", "Counts", serde_json::json!({ "type": "object" })),
            |arguments, _api, _ctx| {
                Box::pin(async move {
                    Ok(ToolExecutionResult {
                        structured_output: arguments.get("value").cloned(),
                        ..ToolExecutionResult::text("counted")
                    })
                })
            },
        )
        .with_structured_output_schema(schema),
    )
}

async fn nested_results_follow_the_schema_contract(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Arc::new(Mutex::new(0));
    fixture
        .registry
        .install(
            Extension::new("tools")
                .with_tool(caller(ToolReplay::Unsafe, &runs))
                .with_tool(count()),
        )
        .unwrap();
    let calls = serde_json::json!([
        { "tool": "count", "args": { "value": { "n": 3 } } },
        { "tool": "count", "args": { "value": { "n": "three" } } },
        { "tool": "count", "args": {} },
    ]);
    fixture.model.push([
        Reply::calls_with(&[("c1", "caller", serde_json::json!({ "calls": calls }))]),
        Reply::text("Done"),
    ]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    run(&conversation, "go").await;
    let results = results(&conversation).await;
    let lines: Vec<serde_json::Value> = texts(&results[0])
        .join("")
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines[0]["value"], serde_json::json!({ "n": 3 }));
    assert_eq!(lines[0]["is_error"], false);
    for broken in &lines[1..] {
        assert_eq!(broken["is_error"], true);
        assert_eq!(broken["value"], serde_json::Value::Null);
        assert_eq!(broken["codes"], serde_json::json!(["invalid_structured_output"]));
    }
    harness.close(&ctx()).await.unwrap();
}

async fn a_model_issued_call_only_loses_a_broken_value(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let loose = FnHarnessTool::new(
        Tool::new(
            "loose",
            "Returns a value without a schema",
            serde_json::json!({ "type": "object" }),
        ),
        |_arguments, _api, _ctx| {
            Box::pin(async move {
                Ok(ToolExecutionResult {
                    structured_output: Some(serde_json::json!({ "x": 1 })),
                    ..ToolExecutionResult::text("loose")
                })
            })
        },
    );
    fixture
        .registry
        .install(Extension::new("tools").with_tool(Arc::new(loose)))
        .unwrap();
    fixture
        .model
        .push([Reply::calls(&[("c1", "loose")]), Reply::text("Done")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    run(&conversation, "go").await;
    let results = results(&conversation).await;
    assert_eq!(texts(&results[0]), ["loose"]);
    assert!(!results[0].is_error);
    assert!(
        fixture
            .reported()
            .iter()
            .any(|(message, _)| message.contains("without declaring a schema"))
    );
    harness.close(&ctx()).await.unwrap();
}
