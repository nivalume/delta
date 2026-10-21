//! Codex's v2 wire samples round trip through the types; the schema snapshot is current.

#![allow(clippy::unwrap_used, clippy::too_many_lines)] // Test helpers outside #[test] functions.

use delta_protocol::delta::{
    PermissionMode, PermissionsUpdateParams, ToolApprovalDecision, ToolRequestApprovalResponse,
};
use delta_protocol::v2::{
    CommandExecutionApprovalDecision, CommandExecutionRequestApprovalResponse, ExecPolicyAmendment, ThreadItem,
    ThreadSession, ThreadStatus, TurnStatus, UserInput,
};
use delta_protocol::{
    ClientNotification, ClientRequest, JsonRpcMessage, RequestId, ServerNotification, ServerNotificationEnvelope,
    ServerRequest, json_schema,
};
use pretty_assertions::assert_eq;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

/// Parses `sample` as `T` and checks it writes back the same JSON.
fn round_trip<T: Serialize + DeserializeOwned>(sample: &Value) -> T {
    let parsed: T = serde_json::from_value(sample.clone()).unwrap_or_else(|error| panic!("{error}: {sample}"));
    assert_eq!(&serde_json::to_value(&parsed).unwrap(), sample);
    parsed
}

fn thread(id: &str) -> Value {
    json!({
        "id": id,
        "sessionId": id,
        "forkedFromId": null,
        "parentThreadId": null,
        "preview": "fix the build",
        "ephemeral": false,
        "modelProvider": "anthropic",
        "model": "claude-test",
        "createdAt": 1_700_000_000,
        "updatedAt": 1_700_000_100,
        "status": { "type": "idle" },
        "path": null,
        "cwd": "/work",
        "cliVersion": "1.0.0",
        "source": "appServer",
        "name": null,
        "turns": []
    })
}

#[test]
fn client_requests_carry_method_id_and_params() {
    let initialize = json!({
        "method": "initialize",
        "id": 0,
        "params": {
            "clientInfo": { "name": "codex_vscode", "title": "Codex VS Code Extension", "version": "0.1.0" },
            "capabilities": { "experimentalApi": true, "optOutNotificationMethods": ["item/reasoning/textDelta"] }
        }
    });
    let request: ClientRequest = round_trip(&initialize);
    assert_eq!(request.id(), &RequestId::Integer(0));
    assert_eq!(request.method(), "initialize");

    let start = json!({
        "method": "turn/start",
        "id": "req-7",
        "params": {
            "threadId": "t1",
            "input": [
                { "type": "text", "text": "Run the tests", "text_elements": [] },
                { "type": "localImage", "path": "/tmp/screen.png" }
            ],
            "cwd": null,
            "approvalPolicy": "on-request",
            "model": null
        }
    });
    let ClientRequest::TurnStart { request_id, params } = round_trip(&start) else {
        panic!("not turn/start");
    };
    assert_eq!(request_id, RequestId::from("req-7"));
    assert_eq!(params.input[0], UserInput::text("Run the tests"));

    for sample in [
        json!({ "method": "thread/start", "id": 1, "params": { "model": "gpt-5", "cwd": "/work", "approvalPolicy": "never", "baseInstructions": null, "developerInstructions": null } }),
        json!({ "method": "thread/resume", "id": 2, "params": { "threadId": "t1", "model": null, "cwd": null, "approvalPolicy": "untrusted" } }),
        json!({ "method": "thread/fork", "id": 3, "params": { "threadId": "t1", "lastTurnId": "u2", "model": null, "cwd": null, "approvalPolicy": null } }),
        json!({ "method": "thread/list", "id": 4, "params": { "cursor": "c", "limit": 20 } }),
        json!({ "method": "thread/read", "id": 5, "params": { "threadId": "t1", "includeTurns": true } }),
        json!({ "method": "turn/steer", "id": 6, "params": { "threadId": "t1", "input": [{ "type": "image", "url": "data:image/png;base64,AA==" }], "expectedTurnId": "u1", "clientUserMessageId": null } }),
        json!({ "method": "turn/interrupt", "id": 7, "params": { "threadId": "t1", "turnId": "u1" } }),
        json!({ "method": "delta/permissions/read", "id": 8, "params": { "threadId": "t1" } }),
        json!({ "method": "delta/permissionRules/read", "id": 9 }),
        json!({ "method": "delta/permissionRules/remove", "id": 10, "params": { "rule": "Bash(git status:*)" } }),
    ] {
        let request: ClientRequest = round_trip(&sample);
        assert_eq!(request.method(), sample["method"]);
        assert!(delta_protocol::CLIENT_REQUEST_METHODS.contains(&request.method()));
    }

    // Omitted optional fields parse; `thread/list` may come without params.
    let list: ClientRequest = serde_json::from_value(json!({ "method": "thread/list", "id": 4 })).unwrap();
    assert!(matches!(list, ClientRequest::ThreadList { .. }));
    let update: ClientRequest = serde_json::from_value(json!({
        "method": "delta/permissions/update",
        "id": 11,
        "params": { "threadId": "t1", "mode": "acceptEdits", "add": { "deny": ["Bash(rm:*)"] } }
    }))
    .unwrap();
    let ClientRequest::PermissionsUpdate { params, .. } = update else {
        panic!("not delta/permissions/update");
    };
    assert_eq!(
        params,
        PermissionsUpdateParams {
            thread_id: "t1".into(),
            mode: Some(PermissionMode::AcceptEdits),
            add: delta_protocol::delta::PermissionRules {
                deny: vec!["Bash(rm:*)".into()],
                ..Default::default()
            },
            ..Default::default()
        }
    );
}

#[test]
fn unknown_methods_do_not_parse_as_requests() {
    let message: JsonRpcMessage = serde_json::from_value(json!({ "method": "account/read", "id": 1 })).unwrap();
    let JsonRpcMessage::Request(request) = message else {
        panic!("not a request");
    };
    assert!(!delta_protocol::CLIENT_REQUEST_METHODS.contains(&request.method.as_str()));
    assert!(serde_json::from_value::<ClientRequest>(json!({ "method": "account/read", "id": 1 })).is_err());
}

#[test]
fn messages_omit_the_jsonrpc_field_and_classify_by_shape() {
    let cases = [
        (json!({ "id": 1, "method": "thread/start", "params": {} }), "request"),
        (json!({ "method": "initialized" }), "notification"),
        (json!({ "id": "a", "result": {} }), "response"),
        (
            json!({ "id": 2, "error": { "code": -32601, "message": "no such method" } }),
            "error",
        ),
    ];
    for (sample, kind) in cases {
        let message: JsonRpcMessage = serde_json::from_value(sample.clone()).unwrap();
        let actual = match &message {
            JsonRpcMessage::Request(_) => "request",
            JsonRpcMessage::Notification(_) => "notification",
            JsonRpcMessage::Response(_) => "response",
            JsonRpcMessage::Error(_) => "error",
        };
        assert_eq!(actual, kind, "{sample}");
        let written = serde_json::to_value(&message).unwrap();
        assert_eq!(written, sample);
        assert!(written.get("jsonrpc").is_none());
    }
    // A client that sends the field anyway is understood.
    let message: JsonRpcMessage = serde_json::from_value(json!({ "jsonrpc": "2.0", "method": "initialized" })).unwrap();
    assert!(matches!(message, JsonRpcMessage::Notification(_)));
    assert_eq!(
        serde_json::from_value::<ClientNotification>(json!({ "method": "initialized" })).unwrap(),
        ClientNotification::Initialized
    );
}

#[test]
fn responses_follow_codex() {
    let session = json!({
        "thread": thread("t1"),
        "model": "claude-test",
        "modelProvider": "anthropic",
        "serviceTier": null,
        "cwd": "/work",
        "approvalPolicy": "on-request",
        "approvalsReviewer": "user",
        "sandbox": { "type": "dangerFullAccess" },
        "reasoningEffort": "medium"
    });
    let session: ThreadSession = round_trip(&session);
    assert_eq!(session.thread.status, ThreadStatus::Idle);

    let mut active = thread("t2");
    active["status"] = json!({ "type": "active", "activeFlags": ["waitingOnApproval"] });
    active["source"] = json!({ "subAgent": { "thread_spawn": { "parent_thread_id": "t1" } } });
    active["turns"] = json!([{
        "id": "u1",
        "items": [
            { "type": "userMessage", "id": "u1", "clientId": null, "content": [{ "type": "text", "text": "hi", "text_elements": [] }] },
            { "type": "reasoning", "id": "g1:0", "summary": [], "content": ["thinking"] },
            { "type": "agentMessage", "id": "g1:1", "text": "Running it." },
            {
                "type": "commandExecution", "id": "call_1", "command": "cargo test", "cwd": "/work", "processId": null,
                "status": "completed", "commandActions": [{ "type": "unknown", "command": "cargo test" }],
                "aggregatedOutput": "ok\n", "exitCode": 0, "durationMs": 1200
            },
            {
                "type": "fileChange", "id": "call_2", "status": "declined",
                "changes": [
                    { "path": "src/a.rs", "kind": { "type": "update", "move_path": null }, "diff": "@@ -1 +1 @@\n-a\n+b\n" },
                    { "path": "src/b.rs", "kind": { "type": "add" }, "diff": "+b\n" }
                ]
            },
            {
                "type": "dynamicToolCall", "id": "call_3", "namespace": null, "tool": "grep", "arguments": { "pattern": "fn" },
                "status": "completed", "contentItems": [{ "type": "inputText", "text": "a.rs:1" }], "success": true, "durationMs": 3
            },
            { "type": "contextCompaction", "id": "c1" }
        ],
        "status": "interrupted",
        "error": null,
        "startedAt": 1_700_000_000,
        "completedAt": 1_700_000_010,
        "durationMs": 10_000
    }]);
    let list = json!({ "data": [active], "nextCursor": null, "backwardsCursor": "b" });
    let list: delta_protocol::v2::ThreadListResponse = round_trip(&list);
    let turn = &list.data[0].turns[0];
    assert_eq!(turn.status, TurnStatus::Interrupted);
    let ids: Vec<&str> = turn.items.iter().map(ThreadItem::id).collect();
    assert_eq!(ids, ["u1", "g1:0", "g1:1", "call_1", "call_2", "call_3", "c1"]);

    round_trip::<delta_protocol::v2::TurnSteerResponse>(&json!({ "turnId": "u1" }));
    round_trip::<delta_protocol::v2::TurnInterruptResponse>(&json!({}));
    round_trip::<delta_protocol::v2::InitializeResponse>(&json!({
        "userAgent": "delta/1.0.0", "codexHome": "/home/me/.delta", "platformFamily": "unix", "platformOs": "linux"
    }));
}

#[test]
fn notifications_carry_params_and_emission_time() {
    let samples = [
        json!({ "method": "thread/started", "params": { "thread": thread("t1") }, "emittedAtMs": 1 }),
        json!({ "method": "thread/status/changed", "params": { "threadId": "t1", "status": { "type": "systemError" } }, "emittedAtMs": 2 }),
        json!({ "method": "turn/started", "params": { "threadId": "t1", "turn": { "id": "u1", "items": [], "status": "inProgress", "error": null, "startedAt": 5, "completedAt": null, "durationMs": null } } }),
        json!({ "method": "turn/completed", "params": { "threadId": "t1", "turn": { "id": "u1", "items": [], "status": "failed", "error": { "message": "overloaded", "codexErrorInfo": "other", "additionalDetails": null }, "startedAt": 5, "completedAt": 9, "durationMs": 4000 } } }),
        json!({ "method": "item/started", "params": { "item": { "type": "agentMessage", "id": "g1:0", "text": "" }, "threadId": "t1", "turnId": "u1", "startedAtMs": 10 } }),
        json!({ "method": "item/completed", "params": { "item": { "type": "agentMessage", "id": "g1:0", "text": "Hello" }, "threadId": "t1", "turnId": "u1", "completedAtMs": 11 } }),
        json!({ "method": "item/agentMessage/delta", "params": { "threadId": "t1", "turnId": "u1", "itemId": "g1:0", "delta": "Hel" } }),
        json!({ "method": "item/commandExecution/outputDelta", "params": { "threadId": "t1", "turnId": "u1", "itemId": "call_1", "delta": "ok\n" } }),
        json!({ "method": "item/reasoning/textDelta", "params": { "threadId": "t1", "turnId": "u1", "itemId": "g1:1", "delta": "hm", "contentIndex": 0 } }),
        json!({ "method": "serverRequest/resolved", "params": { "threadId": "t1", "requestId": 4 } }),
        json!({ "method": "error", "params": { "error": { "message": "rate limited", "codexErrorInfo": null, "additionalDetails": "retry in 2 s" }, "willRetry": true, "threadId": "t1", "turnId": "u1" } }),
        json!({ "method": "thread/tokenUsage/updated", "params": {
            "threadId": "t1", "turnId": "u1",
            "tokenUsage": {
                "total": { "totalTokens": 30, "inputTokens": 20, "cachedInputTokens": 5, "cacheWriteInputTokens": 0, "outputTokens": 10, "reasoningOutputTokens": 2 },
                "last": { "totalTokens": 30, "inputTokens": 20, "cachedInputTokens": 5, "cacheWriteInputTokens": 0, "outputTokens": 10, "reasoningOutputTokens": 2 },
                "modelContextWindow": 200_000
            }
        } }),
        json!({ "method": "delta/snapshot", "params": { "threadId": "t1", "snapshot": { "entries": [] } }, "emittedAtMs": 3 }),
    ];
    for sample in samples {
        let envelope: ServerNotificationEnvelope = round_trip(&sample);
        assert_eq!(envelope.notification.method(), sample["method"]);
        assert_eq!(
            envelope.emitted_at_ms,
            sample.get("emittedAtMs").and_then(Value::as_i64)
        );
    }
    let bare: ServerNotification = serde_json::from_value(json!({
        "method": "serverRequest/resolved", "params": { "threadId": "t1", "requestId": "r" }
    }))
    .unwrap();
    assert!(matches!(bare, ServerNotification::ServerRequestResolved(_)));
}

#[test]
fn approval_requests_and_decisions_follow_codex() {
    let command = json!({
        "method": "item/commandExecution/requestApproval",
        "id": 0,
        "params": {
            "threadId": "t1", "turnId": "u1", "itemId": "call_1", "startedAtMs": 10,
            "reason": "Bash is not allowed by a rule", "command": "git push", "cwd": "/work",
            "proposedExecpolicyAmendment": ["git", "push"]
        }
    });
    let request: ServerRequest = round_trip(&command);
    assert_eq!(request.thread_id(), "t1");
    let file = json!({
        "method": "item/fileChange/requestApproval",
        "id": 1,
        "params": { "threadId": "t1", "turnId": "u1", "itemId": "call_2", "startedAtMs": 11, "reason": null, "grantRoot": null }
    });
    round_trip::<ServerRequest>(&file);
    let tool = json!({
        "method": "delta/tool/requestApproval",
        "id": 2,
        "params": { "threadId": "t1", "turnId": "u1", "itemId": "call_3", "startedAtMs": 12, "tool": "fetch", "arguments": { "url": "https://example.com" }, "proposedRule": "fetch" }
    });
    assert_eq!(round_trip::<ServerRequest>(&tool).id(), &RequestId::Integer(2));

    for (sample, decision) in [
        (
            json!({ "decision": "accept" }),
            CommandExecutionApprovalDecision::Accept,
        ),
        (
            json!({ "decision": "acceptForSession" }),
            CommandExecutionApprovalDecision::AcceptForSession,
        ),
        (
            json!({ "decision": "decline" }),
            CommandExecutionApprovalDecision::Decline,
        ),
        (
            json!({ "decision": "cancel" }),
            CommandExecutionApprovalDecision::Cancel,
        ),
        (
            json!({ "decision": { "acceptWithExecpolicyAmendment": { "execpolicy_amendment": ["git", "push"] } } }),
            CommandExecutionApprovalDecision::AcceptWithExecpolicyAmendment {
                execpolicy_amendment: ExecPolicyAmendment {
                    command: vec!["git".into(), "push".into()],
                },
            },
        ),
    ] {
        let response: CommandExecutionRequestApprovalResponse = round_trip(&sample);
        assert_eq!(response.decision, decision);
    }
    round_trip::<delta_protocol::v2::FileChangeRequestApprovalResponse>(&json!({ "decision": "acceptForSession" }));
    let response: ToolRequestApprovalResponse =
        round_trip(&json!({ "decision": { "acceptWithRule": { "rule": "fetch" } } }));
    assert_eq!(
        response.decision,
        ToolApprovalDecision::AcceptWithRule { rule: "fetch".into() }
    );
}

#[test]
fn schema_snapshot_is_current() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("schema/v2.json");
    let schema = serde_json::to_string_pretty(&json_schema()).unwrap() + "\n";
    if std::env::var_os("DELTA_UPDATE_SCHEMA").is_some() {
        std::fs::write(&path, &schema).unwrap();
        return;
    }
    let snapshot = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(
        snapshot == schema,
        "schema/v2.json is stale; run `DELTA_UPDATE_SCHEMA=1 cargo test -p delta-protocol`"
    );
}
