//! The task graph: live tasks with their waits and owned conversations, and `blocked`.

use std::sync::Arc;

use delta_chord::context::Context;
use delta_chord::json::Json;
use delta_harness::inbox::SubmissionDraft;
use delta_harness::observe::{TaskGraph, TaskGraphState, TaskGraphView};
use delta_harness::registry::Extension;
use delta_harness::scheduler::{BlockedReason, Next, TaskDefinition, TaskError, TaskRuntime};
use delta_harness::session::TaskOptions;
use tokio::sync::Notify;

use crate::Backend;
use crate::support::fixture::{Fixture, chat, ctx, eventually, idle};
use crate::support::model::Reply;
use crate::support::tools::{ToolScript, extension, tool};

cases!(the_graph_follows_live_tasks, a_missing_definition_shows_as_blocked);

/// The graph as rebuilt from storage, once no other observer keeps the mount.
async fn rebuilt(harness: &delta_harness::harness::Harness, held: TaskGraphView) -> TaskGraph {
    let followed = held.graph().unwrap();
    drop(held);
    let rebuilt = harness.task_graph(&ctx()).await.unwrap().graph().unwrap();
    assert_eq!(followed, rebuilt);
    rebuilt
}

async fn the_graph_follows_live_tasks(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let script = ToolScript {
        gate: Some(Arc::new(Notify::new())),
        ..ToolScript::default()
    };
    fixture
        .registry
        .install(extension(&script, vec![tool("read")]))
        .unwrap();
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let graph = harness.task_graph(&ctx()).await.unwrap();
    assert!(graph.graph().unwrap().tasks.is_empty());
    fixture
        .model
        .push([Reply::calls(&[("c1", "read")]), Reply::text("Read")]);
    conversation
        .submit(SubmissionDraft::input("read"), &ctx())
        .await
        .unwrap();
    eventually(|| async { script.started() == ["c1"] }).await;
    eventually(|| async {
        let tasks = graph.graph().unwrap().tasks;
        tasks.len() == 2
            && tasks
                .values()
                .any(|node| matches!(node.state, TaskGraphState::Running { .. }))
    })
    .await;
    let tasks = graph.graph().unwrap().tasks;
    let generation = tasks.values().find(|node| node.kind == "delta.generation").unwrap();
    let tool_task = tasks.values().find(|node| node.kind == "delta.tool").unwrap();
    assert_eq!(tool_task.owner, Some(generation.id));
    let TaskGraphState::Waiting { phase, on, .. } = &generation.state else {
        panic!("waiting: {generation:?}")
    };
    assert_eq!((phase.as_str(), on.as_slice()), ("tools", [tool_task.id].as_slice()));
    let TaskGraphState::Running { phase } = &tool_task.state else {
        panic!("running: {tool_task:?}")
    };
    assert_eq!(phase, "execute");
    let graph = {
        rebuilt(&harness, graph).await;
        harness.task_graph(&ctx()).await.unwrap()
    };
    script.gate.as_ref().unwrap().notify_one();
    idle(&conversation).await;
    eventually(|| async { graph.graph().unwrap().tasks.is_empty() }).await;
    harness.close(&ctx()).await.unwrap();
}

/// A task that never finishes on its own.
#[derive(Clone, Copy, Debug, Default)]
struct Forever;

impl TaskDefinition for Forever {
    type Input = Json;
    type Checkpoint = Json;
    type Result = Json;

    fn name(&self) -> &'static str {
        "test.forever"
    }

    fn version(&self) -> u32 {
        1
    }

    fn initial(&self, _input: &Json) -> Json {
        delta_chord::json!({ "phase": "wait" })
    }

    async fn run<'a>(
        &'a self,
        _phase: Json,
        _rt: &'a TaskRuntime<Self>,
        ctx: &'a Context,
    ) -> Result<Next<Self>, TaskError> {
        ctx.cancelled().await;
        Ok(Next::fail("cancelled"))
    }

    async fn abort<'a>(
        &'a self,
        _phase: Json,
        _rt: &'a TaskRuntime<Self>,
        _ctx: &'a Context,
    ) -> Result<Next<Self>, TaskError> {
        Ok(Next::abort(None))
    }
}

async fn a_missing_definition_shows_as_blocked(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    fixture
        .registry
        .install(Extension::new("forever").with_task(Arc::new(Forever)))
        .unwrap();
    // Opened paused: the task stays pending.
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let task = conversation
        .commit(&ctx(), |tx| {
            Box::pin(async move { Ok(tx.create_task(&Forever, Json::Null, TaskOptions::default()).await?.id()) })
        })
        .await
        .unwrap();
    let graph = harness.task_graph(&ctx()).await.unwrap();
    eventually(|| async { graph.graph().unwrap().tasks.contains_key(&task.0.to_string()) }).await;
    let node = graph.graph().unwrap().tasks[&task.0.to_string()].clone();
    assert!(matches!(&node.state, TaskGraphState::Pending { phase } if phase == "wait"));
    assert_eq!(node.blocked, None);
    fixture.registry.uninstall("forever");
    eventually(|| async {
        graph.graph().unwrap().tasks[&task.0.to_string()].blocked == Some(BlockedReason::MissingTask)
    })
    .await;
    let json = serde_json::to_value(graph.graph().unwrap()).unwrap();
    assert_eq!(json["tasks"][task.0.to_string()]["blocked"], "missing_task");
    assert_eq!(json["tasks"][task.0.to_string()]["state"]["status"], "pending");
    harness.close(&ctx()).await.unwrap();
}
