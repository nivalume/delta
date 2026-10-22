//! `write`: replaces a file's contents, creating it and its parent directories.

use crate::{Backend, Workspace};

cases!(a_file_and_its_parents_are_created_or_replaced);

async fn a_file_and_its_parents_are_created_or_replaced(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    let called = workspace
        .call(
            "write",
            serde_json::json!({ "path": "a/b/new.txt", "content": "first\n" }),
        )
        .await;
    assert!(!called.message.is_error);
    assert_eq!(called.text(), "Successfully wrote to a/b/new.txt");
    assert_eq!(workspace.read("a/b/new.txt"), "first\n");

    let called = workspace
        .call(
            "write",
            serde_json::json!({ "path": "a/b/new.txt", "content": "second" }),
        )
        .await;
    assert_eq!(called.text(), "Successfully wrote to a/b/new.txt");
    assert_eq!(workspace.read("a/b/new.txt"), "second");

    // A directory in the way.
    let called = workspace
        .call("write", serde_json::json!({ "path": "a/b", "content": "x" }))
        .await;
    assert!(called.message.is_error);
    workspace.close().await;
}
