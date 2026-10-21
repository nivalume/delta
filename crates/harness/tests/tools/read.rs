//! `read`: lines, offsets and limits, truncation, long lines, images, and a missing
//! environment.

use delta_harness::tool::Severity;
use delta_harness::tools::{BashOptions, DEFAULT_MAX_BYTES, coding_tools};

use crate::support::fixture::{Fixture, chat, ctx};
use crate::support::model::Reply;
use crate::{Backend, Workspace};

cases!(
    a_file_is_read_whole,
    offset_and_limit_select_lines,
    long_files_are_truncated_with_a_continuation,
    a_line_over_the_limit_shows_its_start,
    images_are_refused,
    paths_are_normalized_and_marks_dropped,
    a_missing_file_or_environment_fails,
);

async fn a_file_is_read_whole(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    workspace.write("notes.txt", "one\ntwo\n");
    let called = workspace.call("read", serde_json::json!({ "path": "notes.txt" })).await;
    assert!(!called.message.is_error);
    assert_eq!(called.text(), "one\ntwo\n");
    assert!(called.diagnostics.is_empty());
    assert_eq!(called.message.details, None);

    workspace.write("empty.txt", "");
    let called = workspace.call("read", serde_json::json!({ "path": "empty.txt" })).await;
    assert!(!called.message.is_error);
    assert!(called.output().is_empty());
    workspace.close().await;
}

async fn offset_and_limit_select_lines(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    let lines: Vec<String> = (1..=10).map(|n| format!("line {n}")).collect();
    workspace.write("ten.txt", lines.join("\n"));

    let called = workspace
        .call(
            "read",
            serde_json::json!({ "path": "ten.txt", "offset": 3, "limit": 2 }),
        )
        .await;
    assert_eq!(called.text(), "line 3\nline 4");
    assert_eq!(called.messages(), ["6 more lines in file. Use offset=5 to continue."]);
    assert_eq!(called.diagnostics[0].severity, Severity::Info);
    assert_eq!(called.diagnostics[0].code, None);

    let called = workspace
        .call("read", serde_json::json!({ "path": "ten.txt", "offset": 9 }))
        .await;
    assert_eq!(called.text(), "line 9\nline 10");
    assert!(called.diagnostics.is_empty());

    let called = workspace
        .call("read", serde_json::json!({ "path": "ten.txt", "offset": 11 }))
        .await;
    assert!(called.message.is_error);
    assert_eq!(
        called.diagnostic("tool_error").message,
        "Offset 11 is beyond end of file (10 lines total)"
    );
    workspace.close().await;
}

async fn long_files_are_truncated_with_a_continuation(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    let lines: Vec<String> = (1..=2500).map(|n| format!("line {n}")).collect();
    workspace.write("long.txt", lines.join("\n") + "\n");
    let called = workspace.call("read", serde_json::json!({ "path": "long.txt" })).await;
    assert_eq!(called.text(), lines[..2000].join("\n"));
    let truncated = called.diagnostic("truncated");
    assert_eq!(truncated.severity, Severity::Info);
    assert_eq!(
        truncated.message,
        "Showing lines 1-2000 of 2501. Use offset=2001 to continue."
    );
    let details = called.message.details.as_ref().unwrap();
    assert_eq!(details["truncation"]["truncated_by"], "lines");
    assert_eq!(details["truncation"]["total_lines"], 2500);
    assert_eq!(details["truncation"]["output_lines"], 2000);

    // Wide lines hit the byte limit first.
    let wide: Vec<String> = (0..100).map(|n| format!("{n:04}{}", "x".repeat(1020))).collect();
    workspace.write("wide.txt", wide.join("\n"));
    let called = workspace
        .call("read", serde_json::json!({ "path": "wide.txt", "offset": 2 }))
        .await;
    let shown = called.text();
    assert_eq!(shown, wide[1..50].join("\n"));
    assert!(shown.len() <= DEFAULT_MAX_BYTES);
    assert_eq!(
        called.diagnostic("truncated").message,
        "Showing lines 2-50 of 100 (50.0KB limit). Use offset=51 to continue."
    );
    workspace.close().await;
}

async fn a_line_over_the_limit_shows_its_start(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    // A multi-byte character straddles the limit: the cut falls before it.
    let line = format!("{}é{}", "a".repeat(DEFAULT_MAX_BYTES - 1), "b".repeat(100));
    workspace.write("one.txt", format!("short\n{line}\nend\n"));
    let called = workspace
        .call("read", serde_json::json!({ "path": "one.txt", "offset": 2 }))
        .await;
    assert_eq!(called.text(), "a".repeat(DEFAULT_MAX_BYTES - 1));
    let truncated = called.diagnostic("truncated");
    assert_eq!(truncated.severity, Severity::Warn);
    assert_eq!(
        truncated.message,
        "Line 2 is 50.1KB, exceeds the 50.0KB limit; showing its first 50.0KB. Use bash: sed -n '2p' one.txt | tail -c +51200"
    );
    let truncation = &called.message.details.as_ref().unwrap()["truncation"];
    assert_eq!(truncation["output_bytes"], DEFAULT_MAX_BYTES - 1);
    assert_eq!(truncation["output_lines"], 1);
    assert_eq!(truncation["first_line_exceeds_limit"], true);
    workspace.close().await;
}

async fn images_are_refused(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    workspace.write("shot.png", b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR");
    let called = workspace.call("read", serde_json::json!({ "path": "shot.png" })).await;
    assert!(called.message.is_error);
    assert!(called.output().is_empty());
    assert_eq!(
        called.diagnostic("unsupported_image").message,
        "shot.png is an image (image/png); image input is not supported in this build"
    );
    workspace.close().await;
}

async fn paths_are_normalized_and_marks_dropped(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    workspace.write("dir/marked.txt", "\u{feff}hello\n");
    let called = workspace
        .call("read", serde_json::json!({ "path": "@dir/marked.txt" }))
        .await;
    assert_eq!(called.text(), "hello\n");

    // A macOS screenshot name, written by the model with an ASCII space.
    workspace.write("Shot 10.00.00\u{202f}PM.txt", "screenshot\n");
    let called = workspace
        .call("read", serde_json::json!({ "path": "Shot 10.00.00 PM.txt" }))
        .await;
    assert_eq!(called.text(), "screenshot\n");

    // An absolute path.
    let absolute = crate::shown(&workspace.path("dir/marked.txt"));
    let called = workspace.call("read", serde_json::json!({ "path": absolute })).await;
    assert_eq!(called.text(), "hello\n");
    workspace.close().await;
}

async fn a_missing_file_or_environment_fails(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    let called = workspace.call("read", serde_json::json!({ "path": "nope.txt" })).await;
    assert!(called.message.is_error);
    assert!(
        called.diagnostic("tool_error").message.contains("nope.txt"),
        "{:?}",
        called.diagnostics
    );
    workspace.close().await;

    // No environment builder: an ordinary error result.
    let fixture = Fixture::new(backend).await;
    fixture.registry.install(coding_tools(BashOptions::default())).unwrap();
    fixture.model.push([
        Reply::calls_with(&[("c1", "read", serde_json::json!({ "path": "x" }))]),
        Reply::text("Done"),
    ]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    crate::support::fixture::settled(
        &conversation
            .submit(delta_harness::inbox::SubmissionDraft::input("go"), &ctx())
            .await
            .unwrap(),
    )
    .await;
    let entries = crate::support::fixture::entries(&conversation).await;
    let result = entries.iter().find(|entry| entry.kind == "delta.tool-result").unwrap();
    let rendered = format!("{:?}", result.model);
    assert!(
        rendered.contains("No execution environment is configured"),
        "{rendered}"
    );
    harness.close(&ctx()).await.unwrap();
}
