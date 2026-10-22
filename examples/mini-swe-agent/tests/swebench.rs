//! The SWE-bench batch runner end to end, with a stand-in `docker` that runs commands on this
//! machine and logs its arguments.

#![cfg(unix)]
#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

use std::os::unix::fs::PermissionsExt as _;
use std::path::Path;

use delta_mini::swebench::{SwebenchArgs, run_batch};
use pretty_assertions::assert_eq;
use serde_json::{Value as Json, json};

/// A `docker` that answers `run` with a container id, runs `exec`'s command with `bash -c`
/// in `workdir`, and logs every call.
fn fake_docker(dir: &Path, workdir: &Path) -> std::path::PathBuf {
    let log = dir.join("docker.log");
    let script = format!(
        r#"#!/bin/bash
echo "$@" >> '{log}'
case "$1" in
  run) echo "container-$RANDOM" ;;
  exec)
    command="${{@: -1}}"
    cd '{workdir}' && exec bash -c "$command"
    ;;
  *) exit 0 ;;
esac
"#,
        log = log.display(),
        workdir = workdir.display()
    );
    let path = dir.join("docker");
    std::fs::write(&path, script).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn call(id: &str, command: &str) -> Json {
    json!({
        "role": "assistant",
        "content": "",
        "tool_calls": [{ "id": id, "type": "function", "function": { "name": "bash", "arguments": json!({ "command": command }).to_string() } }],
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn a_batch_writes_trajectories_predictions_and_a_report() {
    let dir = tempfile::tempdir().unwrap();
    let workdir = dir.path().join("testbed");
    std::fs::create_dir(&workdir).unwrap();
    let docker = fake_docker(dir.path(), &workdir);
    let dataset = dir.path().join("instances.jsonl");
    std::fs::write(
        &dataset,
        [
            json!({ "instance_id": "repo__one-1", "problem_statement": "Fix one." }),
            json!({ "instance_id": "repo__two-2", "problem_statement": "Fix two." }),
            json!({ "instance_id": "other__three-3", "problem_statement": "Skipped by the filter." }),
        ]
        .map(|instance| instance.to_string())
        .join("\n"),
    )
    .unwrap();
    let overrides = dir.path().join("test.yaml");
    let config = json!({
        "environment": { "executable": docker.display().to_string() },
        "model": {
            "model_class": "deterministic_toolcall",
            "model_name": "deterministic",
            "outputs": [
                call("call_1", "echo working"),
                call("call_2", "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\ndiff --git a/x b/x\\n'"),
            ],
        },
    });
    std::fs::write(&overrides, config.to_string()).unwrap();
    let output = dir.path().join("out");

    let summary = run_batch(SwebenchArgs {
        subset: dataset.display().to_string(),
        split: "test".to_owned(),
        slice: String::new(),
        filter: "repo__".to_owned(),
        shuffle: false,
        output: output.display().to_string(),
        workers: 2,
        model: None,
        model_class: None,
        redo_existing: false,
        config: vec!["swebench.yaml".to_owned(), overrides.display().to_string()],
        environment_class: None,
        datasets_server: None,
    })
    .await
    .unwrap();

    assert_eq!(summary.instances, 2);
    let mut submitted: Vec<String> = summary.overview["instances_by_exit_status"]["Submitted"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_str().unwrap().to_owned())
        .collect();
    submitted.sort();
    assert_eq!(submitted, ["repo__one-1", "repo__two-2"]);

    let preds: Json = serde_json::from_str(&std::fs::read_to_string(output.join("preds.json")).unwrap()).unwrap();
    assert_eq!(
        preds["repo__one-1"],
        json!({ "model_name_or_path": "deterministic", "instance_id": "repo__one-1", "model_patch": "diff --git a/x b/x\n" })
    );
    assert!(preds.get("other__three-3").is_none());

    let trajectory: Json =
        serde_json::from_str(&std::fs::read_to_string(output.join("repo__two-2/repo__two-2.traj.json")).unwrap())
            .unwrap();
    assert_eq!(trajectory["instance_id"], "repo__two-2");
    assert_eq!(trajectory["info"]["exit_status"], "Submitted");
    assert_eq!(
        trajectory["info"]["config"]["environment"]["image"],
        "docker.io/swebench/sweb.eval.x86_64.repo_1776_two-2:latest"
    );
    assert_eq!(
        trajectory["info"]["config"]["agent_type"],
        "minisweagent.agents.default.DefaultAgent"
    );
    assert!(
        trajectory["messages"][1]["content"]
            .as_str()
            .unwrap()
            .contains("Fix two.")
    );

    let reports: Vec<_> = std::fs::read_dir(&output)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("exit_statuses_"))
        .collect();
    assert_eq!(reports.len(), 1);

    let log = std::fs::read_to_string(dir.path().join("docker.log")).unwrap();
    assert!(log.contains("run -d --name minisweagent-"), "{log}");
    assert!(
        log.contains("-w /testbed --rm docker.io/swebench/sweb.eval.x86_64.repo_1776_one-1:latest sleep 2h"),
        "{log}"
    );
    assert!(log.contains("exec -w /testbed -e BASH_ENV=/root/.bashrc"), "{log}");
    assert!(log.contains("bash -c echo working"), "{log}");

    // A second run skips what preds.json holds.
    let again = run_batch(SwebenchArgs {
        subset: dataset.display().to_string(),
        split: "test".to_owned(),
        slice: String::new(),
        filter: "repo__".to_owned(),
        shuffle: false,
        output: output.display().to_string(),
        workers: 1,
        model: None,
        model_class: None,
        redo_existing: false,
        config: vec!["swebench.yaml".to_owned(), overrides.display().to_string()],
        environment_class: None,
        datasets_server: None,
    })
    .await
    .unwrap();
    assert_eq!(again.instances, 0);
}
