# delta-harness

A durable agent harness. Conversations, model turns, tool calls and application state are
committed to storage before anyone sees them, and reopening the storage continues where the
last process stopped. It is delta's counterpart of
[`@earendil-works/pi-durable`](https://github.com/earendil-works/pi/tree/main/packages/durable).
The design notes are in [`docs/design/`](docs/design/README.md).

What it provides:

- Storage: a `Storage` trait with memory and SQLite backends, and a conformance suite both
  pass.
- Conversations with an inbox (steer, follow-up and write submissions), forks, and a
  transcript of entries that hold `delta_ai` messages.
- Durable tasks: generation (streaming, retries, deferred responses, tool rounds), one task
  per tool call, and compaction. Each phase commits its outcome. After a crash a
  replay-safe call runs again; any other call settles `interrupted` with the output it had
  published.
- Approvals in the style of codex-rs: a policy asks, the request is stored, and a decision
  (approve, approve for the session, deny, abort, or a timeout) settles it, before or after
  a restart.
- Observation: a live view of each conversation, agent events derived from it, and a task
  graph.
- An execution environment (`env`, with `LocalEnv`) and the built-in `read`, `write`,
  `edit` and `bash` tools (`tools`). `read` sends images within the model's limits; the
  `images` feature adds `ImageCrateProcessor`, which resizes and converts the rest.

## Quick start

A Harness over SQLite with the coding tools and a scripted model that runs one command:

```rust
# #[tokio::main(flavor = "multi_thread")]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
use std::sync::Arc;

use delta_ai::Models;
use delta_ai::providers::faux::{FauxOptions, FauxProvider, faux_assistant_message, faux_text, faux_tool_call};
use delta_chord::context::Context;
use delta_harness::agent::{AgentChange, Change, LocalEnvBuilder};
use delta_harness::harness::{EntryRange, Harness, HarnessOptions, RootOptions};
use delta_harness::inbox::SubmissionDraft;
use delta_harness::model::{ModelRef, ModelsClient};
use delta_harness::records::SubmissionState;
use delta_harness::registry::Registry;
use delta_harness::storage::{SqliteOptions, SqliteStorage};
use delta_harness::tools::{BashOptions, coding_tools};

// The model calls `bash`, then answers.
let faux = FauxProvider::new(FauxOptions::default());
faux.set_responses(vec![
    faux_assistant_message(vec![faux_tool_call("bash", serde_json::json!({ "command": "echo hello" }))]).into(),
    faux_assistant_message(vec![faux_text("The command printed hello.")]).into(),
]);
let models = Models::default();
models.set_provider(faux.clone());

let dir = std::env::temp_dir().join(format!("delta-harness-readme-{}", std::process::id()));
std::fs::create_dir_all(&dir)?;
let storage = SqliteStorage::open(dir.join("harness.db"), SqliteOptions::default()).await?;

let registry = Registry::new();
registry.install(coding_tools(BashOptions::default()))?;
let mut options = HarnessOptions::new(Arc::new(ModelsClient::new(models)));
options.registry = registry;
// Tools reach files and commands through the conversation's environment.
options.env = Some(Arc::new(LocalEnvBuilder::new(&dir)));

let ctx = Context::background();
let harness = Harness::open(Arc::new(storage), options, &ctx).await?;
let agent = AgentChange {
    model: Change::Set(ModelRef::new("faux", "faux-1")),
    instructions: Change::Set("You are a coding agent.".to_owned()),
    ..AgentChange::default()
};
let conversation = harness.root(RootOptions { agent: Some(agent), init: None }, &ctx).await?;

let submission = conversation.submit(SubmissionDraft::input("Say hello from the shell"), &ctx).await?;
let record = submission.wait(&ctx).await?;
let SubmissionState::Done { answer: Some(answer), .. } = record.state else {
    panic!("answered: {:?}", record.state);
};
let range = EntryRange { min_entry_id: Some(answer), max_entry_id: Some(answer), order: None };
let page = conversation.entries(range, 1, None, &ctx).await?;
let Some([delta_ai::Message::Assistant(message)]) = page.items[0].model.as_deref() else {
    panic!("an answer");
};
assert_eq!(message.text(), "The command printed hello.");

harness.close(&ctx).await?;
std::fs::remove_dir_all(&dir)?;
# Ok(())
# }
```

`Harness::open` reconciles what a previous process left and stays paused until a call that
needs progress (here `wait`) or `Harness::resume`.

## Approvals

`HarnessOptions.approval` decides each call before it runs: allow, deny, or ask. An asked
call stores its request in the conversation and waits for `Conversation::decide`, up to the
policy's timeout:

```rust
# #[tokio::main(flavor = "multi_thread")]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
# use std::sync::Arc;
# use delta_ai::Models;
# use delta_ai::providers::faux::{FauxOptions, FauxProvider, faux_assistant_message, faux_text, faux_tool_call};
# use delta_chord::context::Context;
# use delta_harness::agent::{AgentChange, Change, LocalEnvBuilder};
# use delta_harness::harness::{Harness, HarnessOptions, RootOptions};
# use delta_harness::inbox::SubmissionDraft;
# use delta_harness::model::{ModelRef, ModelsClient};
# use delta_harness::registry::Registry;
# use delta_harness::storage::MemoryStorage;
# use delta_harness::tools::{BashOptions, coding_tools};
# let faux = FauxProvider::new(FauxOptions::default());
# faux.set_responses(vec![
#     faux_assistant_message(vec![faux_tool_call("bash", serde_json::json!({ "command": "rm -rf build" }))]).into(),
#     faux_assistant_message(vec![faux_text("I will not delete it.")]).into(),
# ]);
# let models = Models::default();
# models.set_provider(faux.clone());
# let registry = Registry::new();
# registry.install(coding_tools(BashOptions::default()))?;
# let dir = std::env::temp_dir();
use delta_harness::approval::{AskAll, ReviewDecision};

let mut options = HarnessOptions::new(Arc::new(ModelsClient::new(models)));
options.registry = registry;
options.env = Some(Arc::new(LocalEnvBuilder::new(&dir)));
options.approval = Arc::new(AskAll);

let ctx = Context::background();
let harness = Harness::open(Arc::new(MemoryStorage::new()), options, &ctx).await?;
# let agent = AgentChange { model: Change::Set(ModelRef::new("faux", "faux-1")), ..AgentChange::default() };
# let conversation = harness.root(RootOptions { agent: Some(agent), init: None }, &ctx).await?;
let submission = conversation.submit(SubmissionDraft::input("Clean the build"), &ctx).await?;
harness.resume()?;

// A UI lists the waiting requests and answers one.
let request = loop {
    if let Some(request) = conversation.pending_approvals(&ctx).await?.pop() {
        break request;
    }
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
};
assert_eq!(request.name, "bash");
let denied = ReviewDecision::Denied { rejection: Some("not now".to_owned()) };
conversation.decide(request.id, denied, &ctx).await?;

// The call ends `blocked` without running, and the model sees why.
submission.wait(&ctx).await?;
assert!(conversation.pending_approvals(&ctx).await?.is_empty());
harness.close(&ctx).await?;
# Ok(())
# }
```

`ReviewDecision::ApprovedForSession` also grants every later call with the same approval
key (by default the call's canonical arguments).

`RulesPolicy` is the policy for a coding agent: Claude Code's `Tool(specifier)` rules
(`Bash(cargo test:*)`, `Read(./.env)`, `Edit(src/**)`; deny > ask > allow) and modes
(`default`, `accept_edits`, `plan`, `dont_ask`, `bypass`), per conversation through
`Conversation::update_permissions`. `ReviewDecision::ApprovedWithRule { rule }` approves a
call and adds its proposed rule for every conversation. A decision committed while no process
runs is read by the call when the storage is reopened; a request with no answer ends
`blocked` at its stored deadline.

## Observing

- `Conversation::view_state` and `Conversation::watch` give the conversation's view: the
  committed transcript, the live generation and tool slots, the inbox, usage and
  approvals, advanced by each commit.
- `Conversation::events` turns the same commits into `AgentEvent`s with upstream's names
  (`message_update`, `tool_execution_update`, `approval_update`, ...). A consumer that falls
  behind gets one `snapshot` instead of the missed events.
- `Harness::task_graph` and `watch_task_graph` show every live task, what it waits on, the
  conversations it owns, and why it is blocked.

## Extending

Extensions go into a `Registry`; built-in tasks, documents and tools use the same public
traits, so an extension can do anything a built-in does.

| Trait | For |
| --- | --- |
| `HarnessTool` | Tools (`FnHarnessTool` and `typed_harness_tool` are the short forms) |
| `TaskDefinition` | Other durable work, with its own checkpoints and phases |
| `DocDefinition` | Other durable state in a conversation |
| `GenerationHooks`, `ToolHooks`, `CompactionHooks` | Changing requests, calls, results and compaction |
| `ApprovalPolicy` | Permission rules (`RulesPolicy` is the built-in one) |
| `ModelClient` | Model access; `ModelsClient` wraps `delta_ai::Models` |
| `EnvBuilder`, `ExecutionEnv` | Where tools run; `LocalEnvBuilder` and `LocalEnv` are the local ones |
| `Storage` | Other backends |

## Differences from pi-durable

delta-harness keeps pi-durable's model (one durable task per unit of work, commits as the
only source of truth) and drops wire compatibility with it: records, documents and task
kinds use delta's own `delta.*` names and a SQLite schema of their own. Messages inside
entries stay `delta_ai` types, which are wire compatible with pi. Each design note lists
its other differences from upstream.

Not in this version: code mode, remote environments, JSONL storage.
