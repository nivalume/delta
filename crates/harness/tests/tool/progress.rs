//! Output windows, progress published into the slot, bounding and diagnostics.

use std::sync::{Arc, Mutex};

use delta_ai::types::{Tool, UserBlock};
use delta_chord::output::{OutputLimits, OutputSkip, Retain};
use delta_harness::harness::Harness;
use delta_harness::registry::Extension;
use delta_harness::tool::{FnHarnessTool, HarnessTool, Severity, ToolDiagnostic, ToolExecutionResult};
use tokio::sync::Notify;

use crate::support::fixture::{Fixture, chat, ctx, eventually, live};
use crate::support::model::Reply;
use crate::{Backend, results, run, texts};

paused_cases!(
    progress_is_published_while_the_call_runs,
    a_tail_window_keeps_the_end,
    explicit_text_is_bounded,
    diagnostics_keep_their_order,
);

fn declaration(name: &str) -> Tool {
    Tool::new(name, "Reports", serde_json::json!({ "type": "object" }))
}

/// Installs `tool`, runs one call of it and returns the result's text items.
async fn call(fixture: &Fixture, tool: FnHarnessTool) -> (Harness, Vec<String>) {
    let name = tool.name().to_owned();
    fixture
        .registry
        .install(Extension::new("tools").with_tool(Arc::new(tool)))
        .unwrap();
    fixture
        .model
        .push([Reply::calls(&[("c1", &name)]), Reply::text("Done")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    run(&conversation, "go").await;
    let results = results(&conversation).await;
    let texts = texts(&results[0]).into_iter().map(str::to_owned).collect();
    (harness, texts)
}

async fn progress_is_published_while_the_call_runs(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let gate = Arc::new(Notify::new());
    let tool = {
        let gate = gate.clone();
        FnHarnessTool::new(declaration("watch"), move |_arguments, api, ctx| {
            let gate = gate.clone();
            Box::pin(async move {
                api.output_str("first\n");
                api.diagnostic(ToolDiagnostic::new(Severity::Info, "warming up"));
                api.details(serde_json::json!({ "percent": 50 }), ctx).await?;
                ctx.run(gate.notified()).await?;
                api.output_str("second\n");
                Ok(ToolExecutionResult::default())
            })
        })
    };
    fixture
        .registry
        .install(Extension::new("tools").with_tool(Arc::new(tool)))
        .unwrap();
    fixture
        .model
        .push([Reply::calls(&[("c1", "watch")]), Reply::text("Done")]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation
        .submit(delta_harness::inbox::SubmissionDraft::input("go"), &ctx())
        .await
        .unwrap();
    eventually(|| async {
        let state = live(&harness, &conversation).await;
        state
            .tools
            .into_iter()
            .flatten()
            .any(|slot| slot.progress.details.is_some())
    })
    .await;
    let state = live(&harness, &conversation).await;
    let slot = &state.tools.as_ref().unwrap()[0];
    assert_eq!(slot.progress.output.as_deref(), Some("first\n"));
    assert_eq!(slot.progress.details, Some(delta_chord::json!({ "percent": 50 })));
    assert_eq!(slot.progress.diagnostics.as_ref().map(Vec::len), Some(1));
    gate.notify_one();
    crate::support::fixture::settled(&submission).await;
    let results = results(&conversation).await;
    assert_eq!(
        texts(&results[0]),
        ["first\nsecond\n", "<harness>\n[info] warming up\n</harness>"]
    );
    assert_eq!(results[0].details, Some(serde_json::json!({ "percent": 50 })));
    let state = live(&harness, &conversation).await;
    assert!(state.tools.is_none(), "the round's presentation ends with the run");
    harness.close(&ctx()).await.unwrap();
}

async fn a_tail_window_keeps_the_end(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let windows = Arc::new(Mutex::new(Vec::new()));
    let tool = {
        let windows = windows.clone();
        FnHarnessTool::new(declaration("tail"), move |_arguments, api, _ctx| {
            windows
                .lock()
                .unwrap()
                .push(api.output_window().map(|window| window.max_bytes));
            Box::pin(async move {
                api.output(b"0123456789");
                api.output_skipped(OutputSkip {
                    bytes: 100,
                    newlines: 0,
                    ends_with_newline: false,
                });
                api.output(b"abcdefghij");
                Ok(ToolExecutionResult::default())
            })
        })
        .with_output_limits(OutputLimits {
            max_bytes: 8,
            max_lines: 100,
            retain: Retain::Tail,
        })
    };
    let (harness, texts) = call(&fixture, tool).await;
    assert_eq!(
        texts,
        [
            "cdefghij",
            "<harness>\n[warn] Output truncated to its end: 0 lines, 112 bytes dropped\n</harness>"
        ]
    );
    assert_eq!(*windows.lock().unwrap(), [Some(8)]);
    harness.close(&ctx()).await.unwrap();

    // A head window cannot skip: the misuse is reported, not fatal.
    let fixture = Fixture::new(backend).await;
    let tool = FnHarnessTool::new(declaration("head"), |_arguments, api, _ctx| {
        Box::pin(async move {
            api.output_skipped(OutputSkip::default());
            api.output_str(if api.output_window().is_none() {
                "kept"
            } else {
                "a window"
            });
            Ok(ToolExecutionResult::default())
        })
    });
    let (harness, texts) = call(&fixture, tool).await;
    assert_eq!(texts, ["kept"]);
    assert!(
        fixture
            .reported()
            .iter()
            .any(|(message, _)| message.contains("tail-retaining"))
    );
    harness.close(&ctx()).await.unwrap();
}

async fn explicit_text_is_bounded(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let tool = FnHarnessTool::new(declaration("dump"), |_arguments, _api, _ctx| {
        Box::pin(async move {
            let lines: String = (0..3000).map(|line: u32| line.to_string() + "\n").collect();
            Ok(ToolExecutionResult {
                output: Some(vec![UserBlock::text(lines)]),
                ..ToolExecutionResult::default()
            })
        })
    });
    let (harness, texts) = call(&fixture, tool).await;
    assert_eq!(texts.len(), 2);
    assert_eq!(texts[0].lines().count(), 2000);
    assert!(texts[0].starts_with("0\n1\n"));
    let dropped: usize = (2000..3000).map(|line: usize| line.to_string().len() + 1).sum();
    assert_eq!(
        texts[1],
        format!("<harness>\n[warn] Output truncated to its beginning: 1000 lines, {dropped} bytes dropped\n</harness>")
    );
    harness.close(&ctx()).await.unwrap();
}

async fn diagnostics_keep_their_order(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let tool = FnHarnessTool::new(declaration("diagnose"), |_arguments, api, _ctx| {
        Box::pin(async move {
            api.diagnostic(ToolDiagnostic::new(Severity::Info, "reported first"));
            Ok(ToolExecutionResult {
                diagnostics: vec![ToolDiagnostic::new(Severity::Warn, "returned second")],
                ..ToolExecutionResult::text("ok")
            })
        })
    });
    let (harness, texts) = call(&fixture, tool).await;
    assert_eq!(
        texts,
        [
            "ok",
            "<harness>\n[info] reported first\n[warn] returned second\n</harness>"
        ]
    );
    harness.close(&ctx()).await.unwrap();
}
