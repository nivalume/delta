//! `edit`: replacements with a diff and a patch, line endings and marks kept, fuzzy
//! matching, repaired argument shapes, and the errors the model sees.

use crate::{Backend, Workspace};

cases!(
    replacements_apply_with_a_diff_and_a_patch,
    line_endings_and_marks_are_kept,
    fuzzy_matches_keep_untouched_lines,
    repaired_shapes_and_errors,
);

async fn replacements_apply_with_a_diff_and_a_patch(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    workspace.write("src.rs", "fn a() {}\nfn b() {}\nfn c() {}\n");
    let called = workspace
        .call(
            "edit",
            serde_json::json!({
                "path": "src.rs",
                "edits": [
                    { "oldText": "fn a() {}", "newText": "fn a() { 1 }" },
                    { "oldText": "fn c() {}", "newText": "fn c() { 3 }" },
                ],
            }),
        )
        .await;
    assert!(!called.message.is_error, "{:?}", called.diagnostics);
    assert_eq!(called.text(), "Successfully replaced 2 block(s) in src.rs.");
    assert_eq!(workspace.read("src.rs"), "fn a() { 1 }\nfn b() {}\nfn c() { 3 }\n");
    let details = called.message.details.as_ref().unwrap();
    assert_eq!(details["firstChangedLine"], 1);
    let diff = details["diff"].as_str().unwrap();
    assert!(
        diff.contains("-1 fn a() {}") && diff.contains("+1 fn a() { 1 }"),
        "{diff}"
    );
    assert!(diff.contains(" 2 fn b() {}"), "{diff}");
    let patch = details["patch"].as_str().unwrap();
    assert!(
        patch.starts_with("--- src.rs\n+++ src.rs\n@@ -1,3 +1,3 @@\n"),
        "{patch}"
    );
    workspace.close().await;
}

async fn line_endings_and_marks_are_kept(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    workspace.write("crlf.txt", "\u{feff}one\r\ntwo\r\nthree\r\n");
    let called = workspace
        .call(
            "edit",
            serde_json::json!({ "path": "crlf.txt", "edits": [{ "oldText": "two\nthree", "newText": "2\n3" }] }),
        )
        .await;
    assert!(!called.message.is_error, "{:?}", called.diagnostics);
    assert_eq!(workspace.read("crlf.txt"), "\u{feff}one\r\n2\r\n3\r\n");
    workspace.close().await;
}

async fn fuzzy_matches_keep_untouched_lines(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    // Line one has trailing spaces and a smart quote; the model writes plain quotes.
    workspace.write("text.md", "it\u{2019}s   \nsay \u{201c}hi\u{201d}\nend\n");
    let called = workspace
        .call(
            "edit",
            serde_json::json!({ "path": "text.md", "edits": [{ "oldText": "say \"hi\"", "newText": "say \"bye\"" }] }),
        )
        .await;
    assert!(!called.message.is_error, "{:?}", called.diagnostics);
    assert_eq!(workspace.read("text.md"), "it\u{2019}s   \nsay \"bye\"\nend\n");
    assert_eq!(called.message.details.as_ref().unwrap()["firstChangedLine"], 2);
    workspace.close().await;
}

async fn repaired_shapes_and_errors(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    workspace.write("f.txt", "alpha\nbeta\nalpha\n");

    // A top-level pair and `edits` as a JSON string are repaired.
    let called = workspace
        .call(
            "edit",
            serde_json::json!({ "path": "f.txt", "oldText": "beta", "newText": "BETA" }),
        )
        .await;
    assert!(!called.message.is_error, "{:?}", called.diagnostics);
    let edits = serde_json::json!([{ "oldText": "BETA", "newText": "beta" }]).to_string();
    let called = workspace
        .call("edit", serde_json::json!({ "path": "f.txt", "edits": edits }))
        .await;
    assert!(!called.message.is_error, "{:?}", called.diagnostics);
    assert_eq!(workspace.read("f.txt"), "alpha\nbeta\nalpha\n");

    let error = |called: &crate::Called| {
        assert!(called.message.is_error);
        called.diagnostic("tool_error").message.clone()
    };
    let called = workspace
        .call(
            "edit",
            serde_json::json!({ "path": "f.txt", "edits": [{ "oldText": "alpha", "newText": "x" }] }),
        )
        .await;
    assert_eq!(
        error(&called),
        "Found 2 occurrences of the text in f.txt. The text must be unique. Please provide more context to make it unique."
    );
    let called = workspace
        .call("edit", serde_json::json!({ "path": "f.txt", "edits": [] }))
        .await;
    assert_eq!(
        error(&called),
        "Edit tool input is invalid. edits must contain at least one replacement."
    );
    let called = workspace
        .call(
            "edit",
            serde_json::json!({ "path": "gone.txt", "edits": [{ "oldText": "a", "newText": "b" }] }),
        )
        .await;
    assert_eq!(error(&called), "Could not edit file: gone.txt. Error code: not_found.");
    std::fs::create_dir_all(workspace.path("dir")).unwrap();
    let called = workspace
        .call(
            "edit",
            serde_json::json!({ "path": "dir", "edits": [{ "oldText": "a", "newText": "b" }] }),
        )
        .await;
    assert_eq!(error(&called), "Could not edit file: dir. Path is not a file.");
    assert_eq!(workspace.read("f.txt"), "alpha\nbeta\nalpha\n");
    workspace.close().await;
}
