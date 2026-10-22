//! `bash`: output, exit codes, timeouts, spilled output, the command prefix and `prepare`,
//! and the structured output a program gets.

use std::sync::Arc;

use delta_ai::types::Tool;
use delta_chord::context::Context;
use delta_harness::registry::Extension;
use delta_harness::tool::{FnHarnessTool, NestedCallOptions, ToolApi, ToolError, ToolExecutionResult};
use delta_harness::tools::{BashExecution, BashOptions, BashPrepare};
use futures::future::BoxFuture;

use crate::{Backend, Workspace};

cases!(
    output_and_exit_codes,
    timeouts_fail_with_the_output_so_far,
    long_output_keeps_its_tail_and_spills,
    the_prefix_and_prepare_shape_the_command,
    programs_get_structured_output,
);

async fn output_and_exit_codes(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    let called = workspace
        .call("bash", serde_json::json!({ "command": "echo hello; echo there >&2" }))
        .await;
    assert!(!called.message.is_error, "{:?}", called.diagnostics);
    // Two pipes: their lines interleave in either order.
    let mut lines: Vec<&str> = called.text().lines().collect();
    lines.sort_unstable();
    assert_eq!(lines, ["hello", "there"]);
    assert!(called.text().ends_with('\n'));
    assert!(called.diagnostics.is_empty());

    let called = workspace
        .call("bash", serde_json::json!({ "command": "echo oops; exit 3" }))
        .await;
    assert!(called.message.is_error);
    assert_eq!(called.text(), "oops\n");
    assert_eq!(called.diagnostic("exit_code").message, "Command exited with code 3");

    // Git Bash's `pwd` prints `/c/...`; `pwd -W` prints the Windows path.
    let pwd = if cfg!(windows) { "pwd -W" } else { "pwd" };
    let called = workspace.call("bash", serde_json::json!({ "command": pwd })).await;
    let cwd = std::fs::canonicalize(workspace.dir.path()).unwrap();
    let printed = std::fs::canonicalize(called.text().trim_end()).unwrap();
    assert_eq!(printed, cwd);
    workspace.close().await;
}

async fn timeouts_fail_with_the_output_so_far(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    let called = workspace
        .call(
            "bash",
            serde_json::json!({ "command": "echo started; sleep 10", "timeout": 0.5 }),
        )
        .await;
    assert!(called.message.is_error);
    assert_eq!(called.text(), "started\n");
    assert_eq!(
        called.diagnostic("tool_error").message,
        "Command timed out after 0.5 seconds"
    );

    let called = workspace
        .call("bash", serde_json::json!({ "command": "true", "timeout": 0 }))
        .await;
    assert_eq!(
        called.diagnostic("tool_error").message,
        "Invalid timeout: must be a finite number of seconds"
    );
    workspace.close().await;
}

async fn long_output_keeps_its_tail_and_spills(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    let called = workspace
        .call("bash", serde_json::json!({ "command": "seq 1 3000" }))
        .await;
    assert!(!called.message.is_error, "{:?}", called.diagnostics);
    let expected: String = (1001..=3000).map(|n| format!("{n}\n")).collect::<Vec<_>>().concat();
    assert_eq!(called.text(), expected);
    let full = called.diagnostic("full_output");
    let path = full.message.strip_prefix("Full output: ").unwrap();
    let spilled = std::fs::read_to_string(path).unwrap();
    assert_eq!(spilled.lines().count(), 3000);
    assert_eq!(
        called.diagnostic("truncated").message,
        "Output truncated to its end: 1000 lines, 3893 bytes dropped"
    );
    std::fs::remove_file(path).unwrap();
    workspace.close().await;
}

/// Runs every command in `sub` with `FOO=bar`.
struct InSubdirectory;

impl BashPrepare for InSubdirectory {
    fn prepare<'a>(
        &'a self,
        execution: &'a mut BashExecution,
        _api: &'a ToolApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, Result<(), ToolError>> {
        Box::pin(async move {
            execution.cwd = format!("{}/sub", execution.cwd);
            execution.env.insert("FOO".to_owned(), "bar".to_owned());
            Ok(())
        })
    }
}

async fn the_prefix_and_prepare_shape_the_command(backend: Backend) {
    let options = BashOptions {
        command_prefix: Some("GREETING=hi".to_owned()),
        prepare: Some(Arc::new(InSubdirectory)),
    };
    let workspace = Workspace::with_bash(backend, options).await;
    std::fs::create_dir_all(workspace.path("sub")).unwrap();
    let called = workspace
        .call(
            "bash",
            serde_json::json!({ "command": "echo \"$GREETING $FOO $(basename \"$PWD\")\"" }),
        )
        .await;
    assert_eq!(called.text(), "hi bar sub\n");
    workspace.close().await;
}

/// `probe`: runs `bash` as a nested call and returns its structured output as text.
fn probe() -> Extension {
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "command": { "type": "string" } },
        "required": ["command"],
    });
    let tool = FnHarnessTool::new(Tool::new("probe", "Run bash", schema), |arguments, api, ctx| {
        Box::pin(async move {
            let result = api
                .execute_tool("bash", arguments, NestedCallOptions::default(), ctx)
                .await?;
            Ok(ToolExecutionResult::text(
                serde_json::to_string(&result.structured_output).unwrap(),
            ))
        })
    });
    Extension::new("probe").with_tool(Arc::new(tool))
}

async fn programs_get_structured_output(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    workspace.fixture.registry.install(probe()).unwrap();
    let called = workspace
        .call("probe", serde_json::json!({ "command": "echo out; exit 2" }))
        .await;
    let structured: serde_json::Value = serde_json::from_str(called.text()).unwrap();
    assert_eq!(
        structured,
        serde_json::json!({ "output": "out\n", "truncated": false, "exitCode": 2 })
    );
    workspace.close().await;
}
