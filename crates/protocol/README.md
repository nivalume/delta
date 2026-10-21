# delta-protocol

The UI protocol of [delta-harness](../harness): a subset of Codex's
[app-server](https://github.com/openai/codex/tree/main/codex-rs/app-server) v2 protocol, so a
client written for Codex's app-server can drive a delta harness, plus delta's own methods
under `delta/`. This crate holds the types only; `delta-app-server` serves them. The design
notes are in [`docs/design/protocol.md`](docs/design/protocol.md), and
[`schema/v2.json`](schema/v2.json) is the JSON Schema of every message.

Messages are JSON-RPC 2.0 without the `"jsonrpc"` field, one per line:

```rust
use delta_protocol::{ClientRequest, RequestId, ServerNotification, ServerNotificationEnvelope};

let request: ClientRequest = serde_json::from_str(
    r#"{"method":"turn/start","id":3,"params":{"threadId":"t1","input":[{"type":"text","text":"hi"}]}}"#,
)?;
assert_eq!(request.id(), &RequestId::Integer(3));

let line = r#"{"method":"item/agentMessage/delta","params":{"threadId":"t1","turnId":"u1","itemId":"g1:0","delta":"Hel"},"emittedAtMs":1700000000000}"#;
let envelope: ServerNotificationEnvelope = serde_json::from_str(line)?;
assert!(matches!(envelope.notification, ServerNotification::AgentMessageDelta(_)));
# Ok::<(), serde_json::Error>(())
```

| Direction | Methods |
| --- | --- |
| client → server requests | `initialize`, `thread/start`, `thread/resume`, `thread/fork`, `thread/list`, `thread/read`, `turn/start`, `turn/steer`, `turn/interrupt`, `delta/permissions/read`, `delta/permissions/update`, `delta/permissionRules/read`, `delta/permissionRules/remove` |
| client → server notifications | `initialized` |
| server → client notifications | `error`, `thread/started`, `thread/status/changed`, `thread/tokenUsage/updated`, `turn/started`, `turn/completed`, `item/started`, `item/completed`, `item/agentMessage/delta`, `item/commandExecution/outputDelta`, `item/reasoning/textDelta`, `serverRequest/resolved`, `delta/snapshot` |
| server → client requests | `item/commandExecution/requestApproval`, `item/fileChange/requestApproval`, `delta/tool/requestApproval` |

Run `DELTA_UPDATE_SCHEMA=1 cargo test -p delta-protocol` after changing a type to rewrite
the schema snapshot.
