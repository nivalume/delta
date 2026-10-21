//! The file mutation queue: mutations of one file, under any of its names, run one at a time
//! in order; other files never wait.

use std::sync::{Arc, Mutex};

use delta_harness::env::LocalEnv;
use delta_harness::tools::FileMutationQueue;
use tokio::sync::{Notify, oneshot};

use crate::support::TempDir;
use crate::support::fixture::ctx;
use crate::{Backend, Workspace};

cases!(parallel_edits_of_one_file_all_land);

async fn parallel_edits_of_one_file_all_land(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    let lines: Vec<String> = (0..8).map(|n| format!("line {n}")).collect();
    workspace.write("shared.txt", lines.join("\n"));
    let calls: Vec<(&str, serde_json::Value)> = (0..8)
        .map(|n| {
            let edit = serde_json::json!({ "oldText": format!("line {n}"), "newText": format!("LINE {n}") });
            ("edit", serde_json::json!({ "path": "shared.txt", "edits": [edit] }))
        })
        .collect();
    let results = workspace.calls(&calls).await;
    assert!(results.iter().all(|called| !called.message.is_error), "{results:?}");
    let expected: Vec<String> = (0..8).map(|n| format!("LINE {n}")).collect();
    assert_eq!(workspace.read("shared.txt"), expected.join("\n"));
    workspace.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_file_waits_and_others_do_not() {
    let dir = TempDir::new();
    let real = dir.path().join("real");
    std::fs::create_dir_all(&real).unwrap();
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let env = Arc::new(LocalEnv::new(dir.path()));
    let queue = FileMutationQueue::new();
    let log = Arc::new(Mutex::new(Vec::new()));
    let gate = Arc::new(Notify::new());

    let mutation = |name: &'static str, path: String, wait: Option<(oneshot::Sender<()>, Arc<Notify>)>| {
        let (env, queue, log) = (env.clone(), queue.clone(), log.clone());
        tokio::spawn(async move {
            let ctx = ctx();
            queue
                .run(&*env, &path, &ctx, async {
                    if let Some((started, gate)) = wait {
                        started.send(()).unwrap();
                        gate.notified().await;
                    }
                    log.lock().unwrap().push(name);
                })
                .await
                .unwrap();
        })
    };

    // The file does not exist yet: both names resolve through the canonical directory.
    let (started, holding) = oneshot::channel();
    let first = mutation(
        "first",
        real.join("new.txt").to_str().unwrap().to_owned(),
        Some((started, gate.clone())),
    );
    holding.await.unwrap();
    let second = mutation("second", "link/new.txt".to_owned(), None);
    mutation("other", "real/other.txt".to_owned(), None).await.unwrap();
    assert_eq!(*log.lock().unwrap(), ["other"]);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert_eq!(*log.lock().unwrap(), ["other"], "the second mutation waits");
    gate.notify_one();
    first.await.unwrap();
    second.await.unwrap();
    assert_eq!(*log.lock().unwrap(), ["other", "first", "second"]);
}
