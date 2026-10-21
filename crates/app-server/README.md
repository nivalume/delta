# delta-app-server

A JSON-RPC server for [delta-harness](../harness) that speaks
[delta-protocol](../protocol): a subset of Codex's app-server v2 protocol, so a client written
for Codex's app-server (an editor extension, a TUI) drives a delta harness. One
`AppServer::serve` call runs one connection over any reader and writer: newline-delimited
JSON without the `"jsonrpc"` field. The design notes are in
[`docs/design/app-server.md`](docs/design/app-server.md).

```rust,no_run
# async fn example(harness: delta_harness::harness::Harness) -> Result<(), Box<dyn std::error::Error>> {
use delta_app_server::{AppServer, AppServerOptions};
use delta_chord::context::Context;
use delta_harness::model::ModelRef;

let mut options = AppServerOptions::new(std::env::current_dir()?.to_string_lossy());
options.model = Some(ModelRef::new("anthropic", "claude-sonnet-4-5"));
let server = AppServer::new(harness, options);
// One connection over stdio.
server.serve(tokio::io::stdin(), tokio::io::stdout(), &Context::background()).await?;
# Ok(())
# }
```

| Codex | delta |
| --- | --- |
| thread | a conversation |
| turn | an input and the work up to its answer; the turn id is the input's submission id |
| `agentMessage`, `reasoning` | the blocks of an assistant message, `<generation task>:<block index>` |
| `commandExecution` | a `bash` call, its output streamed as `outputDelta` |
| `fileChange` | an `edit` (with its patch) or `write` call |
| `dynamicToolCall` | any other tool call |
| `requestApproval` | a pending approval of the harness's `ApprovalPolicy` |
| `acceptWithExecpolicyAmendment` | `ApprovedWithRule`: an allow rule for every thread |
