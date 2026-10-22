//! `tool_search` and deferred tools: a deferred tool is offered once a search loads it, loads
//! are recorded by name in `model_tools`, and they hold per branch and across a reset.
//!
//! Ported from upstream `test/tool-search.test.ts`.

use std::sync::Arc;

use delta_ai::types::{Message, Tool, UserBlock};
use delta_harness::agent::{AgentChange, AgentDoc, AgentState, Change, ModelToolFilter};
use delta_harness::documents::DocAddress;
use delta_harness::harness::{Conversation, ConversationCreateOptions, Harness};
use delta_harness::inbox::SubmissionDraft;
use delta_harness::records::EntryRecord;
use delta_harness::registry::Extension;
use delta_harness::tool::{FnHarnessTool, HarnessTool, ToolControl, ToolExecutionResult};
use delta_harness::tools::tool_search_extension;

use crate::Backend;
use crate::support::fixture::{Fixture, chat, ctx, entries, settled};
use crate::support::model::Reply;

cases!(
    a_deferred_tool_is_offered_once_tool_search_loads_it,
    loaded_tools_are_appended_in_load_order,
    loads_are_recorded_by_name,
    a_loaded_tool_survives_its_extension_leaving_for_a_while,
    loads_hold_per_branch_and_after_a_reset,
    an_unrelated_query_loads_nothing,
);

/// A tool answering `<name> ran`, with an optional issue number.
fn tool(name: &str, description: &str, deferred: bool) -> Arc<dyn HarnessTool> {
    let answer = format!("{name} ran");
    Arc::new(
        FnHarnessTool::new(
            Tool::new(
                name,
                description,
                serde_json::json!({
                    "type": "object",
                    "properties": { "number": { "type": "number", "description": "Issue number" } },
                }),
            ),
            move |_arguments, _api, _ctx| {
                let answer = answer.clone();
                Box::pin(async move { Ok(ToolExecutionResult::text(answer)) })
            },
        )
        .with_deferred(deferred),
    )
}

/// Installs `tool` as its own extension, `tool:<name>`.
fn add_tool(fixture: &Fixture, tool: Arc<dyn HarnessTool>) {
    let name = format!("tool:{}", tool.name());
    fixture.registry.install(Extension::new(name).with_tool(tool)).unwrap();
}

fn get_issue() -> Arc<dyn HarnessTool> {
    tool("github_get_issue", "Get a GitHub issue by number", true)
}

/// `alpha`, the deferred `github_get_issue` and `github_list_repos`, then `omega`, and
/// `tool_search` last.
async fn setup(backend: Backend) -> Fixture {
    let fixture = Fixture::new(backend).await;
    add_tool(&fixture, tool("alpha", "Alpha tool", false));
    add_tool(&fixture, get_issue());
    add_tool(
        &fixture,
        tool("github_list_repos", "List the repositories of a user", true),
    );
    add_tool(&fixture, tool("omega", "Omega tool", false));
    fixture.registry.install(tool_search_extension()).unwrap();
    fixture
}

fn call(id: &str, name: &str, arguments: serde_json::Value) -> Reply {
    Reply::calls_with(&[(id, name, arguments)])
}

async fn go(conversation: &Conversation, text: &str) {
    let submission = conversation.submit(SubmissionDraft::input(text), &ctx()).await.unwrap();
    settled(&submission).await;
}

fn names(tools: &[Arc<dyn HarnessTool>]) -> Vec<String> {
    tools.iter().map(|tool| tool.name().to_owned()).collect()
}

async fn offered(conversation: &Conversation) -> Vec<String> {
    names(&conversation.agent(&ctx()).await.unwrap().tools)
}

async fn model_tools(harness: &Harness, conversation: &Conversation) -> Option<ModelToolFilter> {
    let value = harness
        .snapshot(DocAddress::<AgentDoc>::new(conversation.id()), &ctx())
        .await
        .unwrap()?;
    let state: AgentState = delta_chord::json::from_json(&value).unwrap();
    state.model_tools
}

fn edit(add: &[&str]) -> ModelToolFilter {
    ModelToolFilter::Edit {
        remove: Vec::new(),
        add: add.iter().map(|name| (*name).to_owned()).collect(),
    }
}

/// The tools each `delta.system` entry adds and whether it removes any, oldest first.
fn system_messages(entries: &[EntryRecord]) -> Vec<(Vec<String>, bool)> {
    entries
        .iter()
        .filter(|entry| entry.kind == "delta.system")
        .map(|entry| match entry.model.as_deref() {
            Some([Message::System(system)]) => (
                system
                    .tools_added
                    .iter()
                    .flatten()
                    .map(|tool| tool.name.clone())
                    .collect(),
                system.tools_removed.is_some(),
            ),
            other => panic!("a system message: {other:?}"),
        })
        .collect()
}

fn tool_result_texts(entries: &[EntryRecord]) -> Vec<String> {
    entries
        .iter()
        .filter(|entry| entry.kind == "delta.tool-result")
        .map(|entry| match entry.model.as_deref() {
            Some([Message::ToolResult(result)]) => result
                .content
                .iter()
                .filter_map(|block| match block {
                    UserBlock::Text(text) => Some(text.text.clone()),
                    UserBlock::Image(_) => None,
                })
                .collect(),
            other => panic!("a tool result: {other:?}"),
        })
        .collect()
}

async fn a_deferred_tool_is_offered_once_tool_search_loads_it(backend: Backend) {
    let fixture = setup(backend).await;
    fixture.model.push([
        call(
            "c1",
            "tool_search",
            serde_json::json!({ "query": "github issue", "limit": 1 }),
        ),
        call("c2", "github_get_issue", serde_json::json!({ "number": 42 })),
        Reply::text("done"),
    ]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    assert_eq!(offered(&conversation).await, ["alpha", "omega", "tool_search"]);
    let agent = conversation.agent(&ctx()).await.unwrap();
    assert_eq!(names(&agent.deferred), ["github_get_issue", "github_list_repos"]);
    // Deferred tools stay callable by other tools, such as code mode scripts.
    assert!(names(&agent.callable).contains(&"github_get_issue".to_owned()));
    go(&conversation, "go").await;
    let transcript = entries(&conversation).await;
    assert_eq!(
        tool_result_texts(&transcript),
        [
            "Loaded 1 tool. They are available from your next call:\n- github_get_issue: Get a GitHub issue by number",
            "github_get_issue ran",
        ]
    );
    assert_eq!(
        model_tools(&harness, &conversation).await,
        Some(edit(&["github_get_issue"]))
    );
    assert_eq!(
        offered(&conversation).await,
        ["alpha", "omega", "tool_search", "github_get_issue"]
    );
    // A later search sees only the tools not loaded yet.
    let agent = conversation.agent(&ctx()).await.unwrap();
    assert_eq!(names(&agent.deferred), ["github_list_repos"]);
    // The load appended the declaration: nothing was removed and declared again, so the
    // cached prefix holds.
    let messages = system_messages(&transcript);
    assert_eq!(messages[0].0, ["alpha", "omega", "tool_search"]);
    assert_eq!(messages[1], (vec!["github_get_issue".to_owned()], false));
    harness.close(&ctx()).await.unwrap();
}

async fn loaded_tools_are_appended_in_load_order(backend: Backend) {
    let fixture = setup(backend).await;
    fixture.model.push([
        call(
            "c1",
            "tool_search",
            serde_json::json!({ "query": "repositories", "limit": 1 }),
        ),
        call("c2", "tool_search", serde_json::json!({ "query": "issue", "limit": 1 })),
        Reply::text("done"),
    ]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    go(&conversation, "go").await;
    assert_eq!(
        offered(&conversation).await,
        ["alpha", "omega", "tool_search", "github_list_repos", "github_get_issue"]
    );
    let loads = system_messages(&entries(&conversation).await)[1..].to_vec();
    assert_eq!(
        loads,
        [
            (vec!["github_list_repos".to_owned()], false),
            (vec!["github_get_issue".to_owned()], false),
        ]
    );
    harness.close(&ctx()).await.unwrap();
}

async fn loads_are_recorded_by_name(backend: Backend) {
    let fixture = setup(backend).await;
    let grow = FnHarnessTool::new(
        Tool::new("grow", "grow", serde_json::json!({ "type": "object" })),
        |_arguments, _api, _ctx| {
            Box::pin(async {
                Ok(ToolExecutionResult {
                    output: Some(Vec::new()),
                    control: Some(ToolControl {
                        add_tools: vec!["github_get_issue".to_owned(), "alpha".to_owned(), "ghost".to_owned()],
                        ..ToolControl::default()
                    }),
                    ..ToolExecutionResult::default()
                })
            })
        },
    );
    add_tool(&fixture, Arc::new(grow));
    fixture.model.push([
        call("c1", "grow", serde_json::json!({})),
        Reply::text("done"),
        call("c2", "grow", serde_json::json!({})),
        Reply::text("done"),
    ]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    go(&conversation, "go").await;
    let loaded = edit(&["github_get_issue", "alpha", "ghost"]);
    assert_eq!(model_tools(&harness, &conversation).await, Some(loaded.clone()));
    assert_eq!(
        offered(&conversation).await,
        ["alpha", "omega", "tool_search", "grow", "github_get_issue"]
    );
    // A name in both remove and add is not offered.
    let both = ModelToolFilter::Edit {
        remove: vec!["github_get_issue".to_owned()],
        add: vec!["github_get_issue".to_owned()],
    };
    let change = AgentChange {
        model_tools: Change::Set(both),
        ..AgentChange::default()
    };
    conversation.configure(change, &ctx()).await.unwrap();
    assert!(!offered(&conversation).await.contains(&"github_get_issue".to_owned()));
    // Loading it again drops it from remove.
    go(&conversation, "again").await;
    assert_eq!(model_tools(&harness, &conversation).await, Some(loaded));
    assert!(offered(&conversation).await.contains(&"github_get_issue".to_owned()));
    harness.close(&ctx()).await.unwrap();
}

async fn a_loaded_tool_survives_its_extension_leaving_for_a_while(backend: Backend) {
    let fixture = setup(backend).await;
    fixture.model.push([
        call(
            "c1",
            "tool_search",
            serde_json::json!({ "query": "github issue", "limit": 1 }),
        ),
        Reply::text("done"),
    ]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    go(&conversation, "go").await;
    // The server goes away: its extension is installed again without the tool.
    fixture
        .registry
        .install(Extension::new("tool:github_get_issue"))
        .unwrap();
    assert!(!offered(&conversation).await.contains(&"github_get_issue".to_owned()));
    add_tool(&fixture, get_issue());
    assert!(offered(&conversation).await.contains(&"github_get_issue".to_owned()));
    // With an explicit list, a load appends to it as before.
    let change = AgentChange {
        model_tools: Change::Set(ModelToolFilter::Exactly(vec![
            "alpha".to_owned(),
            "tool_search".to_owned(),
        ])),
        ..AgentChange::default()
    };
    conversation.configure(change, &ctx()).await.unwrap();
    fixture.model.push([
        call(
            "c2",
            "tool_search",
            serde_json::json!({ "query": "repositories", "limit": 1 }),
        ),
        Reply::text("done"),
    ]);
    go(&conversation, "more").await;
    assert_eq!(
        model_tools(&harness, &conversation).await,
        Some(ModelToolFilter::Exactly(vec![
            "alpha".to_owned(),
            "tool_search".to_owned(),
            "github_list_repos".to_owned(),
        ]))
    );
    harness.close(&ctx()).await.unwrap();
}

async fn loads_hold_per_branch_and_after_a_reset(backend: Backend) {
    let fixture = setup(backend).await;
    fixture.model.push([
        call(
            "c1",
            "tool_search",
            serde_json::json!({ "query": "github issue", "limit": 1 }),
        ),
        Reply::text("done"),
    ]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    go(&conversation, "go").await;
    let transcript = entries(&conversation).await;
    let before = conversation
        .fork(transcript[0].id, ConversationCreateOptions::ownerless(), &ctx())
        .await
        .unwrap();
    let after = conversation
        .fork(
            transcript.last().unwrap().id,
            ConversationCreateOptions::ownerless(),
            &ctx(),
        )
        .await
        .unwrap();
    assert!(!offered(&before).await.contains(&"github_get_issue".to_owned()));
    assert!(offered(&after).await.contains(&"github_get_issue".to_owned()));
    conversation.reset(None, &ctx()).await.unwrap();
    fixture.model.push([Reply::text("done")]);
    go(&conversation, "fresh").await;
    let baseline = system_messages(&entries(&conversation).await).pop().unwrap();
    assert_eq!(baseline.0, ["alpha", "omega", "tool_search", "github_get_issue"]);
    harness.close(&ctx()).await.unwrap();
}

async fn an_unrelated_query_loads_nothing(backend: Backend) {
    let fixture = setup(backend).await;
    fixture.model.push([
        call("c1", "tool_search", serde_json::json!({ "query": "weather forecast" })),
        Reply::text("done"),
    ]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    go(&conversation, "go").await;
    assert_eq!(
        tool_result_texts(&entries(&conversation).await),
        ["No matching tools found."]
    );
    assert_eq!(model_tools(&harness, &conversation).await, None);
    harness.close(&ctx()).await.unwrap();
}
