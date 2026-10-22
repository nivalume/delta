//! A provider the config defines, end to end: an OpenAI-compatible server (wiremock) answers
//! the agent's requests with recorded SSE bodies.

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use delta_ai::{ApiKeyCredential, Credential, CredentialStore, InMemoryCredentialStore};
use delta_mini::agent::AgentKind;
use delta_mini::cli::{BuildOptions, build_agent};
use delta_mini::config;
use delta_mini::trajectory::ExitStatus;
use pretty_assertions::assert_eq;
use serde_json::{Value as Json, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// An OpenAI chat completion stream calling bash with `command`.
fn bash_call(id: &str, command: &str) -> String {
    let chunks = [
        json!({ "id": "c", "choices": [{ "index": 0, "delta": { "role": "assistant", "content": "Running.", "tool_calls": [{
            "index": 0, "id": id, "type": "function",
            "function": { "name": "bash", "arguments": json!({ "command": command }).to_string() },
        }] } }] }),
        json!({ "id": "c", "choices": [{ "index": 0, "delta": {}, "finish_reason": "tool_calls" }],
                "usage": { "prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100 } }),
    ];
    let mut body = String::new();
    for chunk in chunks {
        let _ = write!(body, "data: {chunk}\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    body
}

/// The bodies in order, the last one again after them.
struct Answers {
    bodies: Vec<String>,
    next: AtomicUsize,
}

impl Respond for Answers {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        let index = self.next.fetch_add(1, Ordering::SeqCst).min(self.bodies.len() - 1);
        ResponseTemplate::new(200).set_body_raw(self.bodies[index].clone(), "text/event-stream")
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_configured_provider_runs_the_agent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(Answers {
            bodies: vec![
                bash_call("call_1", "echo hello"),
                bash_call("call_2", "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\ndone\\n'"),
            ],
            next: AtomicUsize::new(0),
        })
        .mount(&server)
        .await;
    let dir = tempfile::tempdir().unwrap();
    let mut config = config::load(&["mini.yaml".to_owned()]).unwrap();
    config::merge(
        &mut config,
        json!({
            "model": {
                "provider": "local",
                "id": "coder",
                "temperature": 0.25,
                "max_tokens": 512,
                "sampling": { "top_p": 0.5 },
            },
            "providers": {
                "local": {
                    "name": "Local server",
                    "api": "openai-completions",
                    "base_url": format!("{}/v1", server.uri()),
                    "api_key_env": ["DELTA_MINI_TEST_SURELY_UNSET_KEY"],
                    "models": [{ "id": "coder", "cost": { "input": 1.0, "output": 2.0 } }],
                },
            },
            "environment": { "cwd": dir.path().display().to_string() },
        }),
    );
    // A key stored with `mini-extra login` wins over the provider's variables.
    let credentials = Arc::new(InMemoryCredentialStore::new());
    credentials
        .modify(
            "local",
            Box::new(|_| {
                Box::pin(async {
                    Ok(Some(Credential::ApiKey(ApiKeyCredential {
                        key: Some("sk-local".to_owned()),
                        env: None,
                    })))
                })
            }),
        )
        .await
        .unwrap();
    let output_path = dir.path().join("run.traj.json");
    let built = build_agent(
        &config,
        BuildOptions {
            kind: AgentKind::Default,
            output: Some(output_path.clone()),
            credentials: Some(credentials),
            ..BuildOptions::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(
        (built.resolved.provider.as_str(), built.resolved.api.as_str()),
        ("local", "openai-completions")
    );
    let output = built.agent.run("Fix the bug.", serde_json::Map::new()).await.unwrap();
    built.env.cleanup().await;

    assert_eq!(output.exit.status, ExitStatus::Submitted, "{:?}", output.exit);
    assert_eq!(output.exit.submission, "done\n");
    let trajectory = &output.trajectory;
    assert_eq!(trajectory.model["provider"], "local");
    assert_eq!(trajectory.config["providers"]["local"]["name"], "Local server");
    assert_eq!(trajectory.stats.calls, 2);
    // 1000 input tokens at $1/M and 100 output tokens at $2/M, twice.
    assert!(
        (trajectory.stats.cost - 0.0024).abs() < 1e-9,
        "{}",
        trajectory.stats.cost
    );
    assert_eq!(
        serde_json::to_value(&trajectory.entries[2]).unwrap()["message"]["api"],
        "openai-completions"
    );

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0].headers.get("authorization").unwrap().to_str().unwrap(),
        "Bearer sk-local"
    );
    let first: Json = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(first["model"], "coder");
    assert_eq!(first["temperature"], 0.25);
    assert_eq!(first["top_p"], 0.5);
    assert_eq!(
        first.get("max_tokens").or_else(|| first.get("max_completion_tokens")),
        Some(&json!(512))
    );
    assert_eq!(first["tools"][0]["function"]["name"], "bash");
    let second: Json = serde_json::from_slice(&requests[1].body).unwrap();
    let result = second["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "tool")
        .unwrap();
    assert_eq!(result["tool_call_id"], "call_1");
    assert!(result["content"].to_string().contains("hello"), "{result}");
}
