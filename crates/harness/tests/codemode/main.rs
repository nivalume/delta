//! The `codemode` tool: scripts that call the conversation's tools, result mapping, the store,
//! globals, approvals of nested calls, and replay after a crash, including the reruns that
//! refuse to replay (a stored result, another version, fingerprint or deadline, divergence).
//!
//! Ported from upstream `test/codemode.test.ts` where it applies. Every case runs against the
//! memory backend and SQLite in a temp file with a scripted model.

#![allow(clippy::unwrap_used, clippy::too_many_lines)]

#[path = "../support/mod.rs"]
mod support;

/// Runs each case against memory and SQLite in a file, on a multi-threaded runtime.
macro_rules! cases {
    ($($case:ident),* $(,)?) => {
        mod memory {
            $(#[tokio::test(flavor = "multi_thread", worker_threads = 4)] async fn $case() { super::$case(crate::Backend::Memory).await; })*
        }
        mod sqlite {
            $(#[tokio::test(flavor = "multi_thread", worker_threads = 4)] async fn $case() { super::$case(crate::Backend::SqliteFile).await; })*
        }
    };
}

use std::sync::{Arc, Mutex};

use delta_ai::types::{Message, Tool, ToolResultMessage, UserBlock};
use delta_chord::context::Context;
use delta_codemode::{Sandbox, SandboxOptions};
use delta_harness::agent::LocalEnvBuilder;
use delta_harness::approval::{ApprovalCall, ApprovalPolicy, PolicyDecision, PolicyError};
use delta_harness::codemode::{
    CODEMODE_MODULE_VERSION, CodemodeGlobal, CodemodeJournalDoc, CodemodeNamespace, CodemodeOptions, CodemodeRunDoc,
    CodemodeStoreDoc, CodemodeToolDetails, JournalState, RunState, StoreState, StoredResult, codemode_extension,
};
use delta_harness::documents::DocAddress;
use delta_harness::harness::{AbortOptions, Conversation, Harness};
use delta_harness::inbox::SubmissionDraft;
use delta_harness::records::{SubmissionId, SubmissionState, TaskId};
use delta_harness::registry::{Extension, HookApi};
use delta_harness::tool::{
    FnHarnessTool, HarnessTool, NestedCallDoc, NestedCallState, ToolApi, ToolError, ToolExecutionResult, ToolReplay,
};
use futures::future::BoxFuture;
use serde_json::Value;

pub(crate) use support::Backend;
use support::fixture::{Fixture, PATIENCE, chat, ctx, entries, eventually, live, settled};
use support::model::Reply;

cases!(
    a_script_calls_tools_in_parallel,
    results_map_as_tools_declare,
    the_store_keeps_a_successful_scripts_writes,
    globals_answer_and_unknown_tools_throw,
    nested_calls_are_approved_one_by_one,
    options_and_timeouts,
    an_abort_ends_the_script_and_its_calls,
    a_crashed_script_resumes_without_running_finished_calls_again,
    a_rerun_returns_a_stored_result,
    a_rerun_refuses_another_version_fingerprint_or_deadline,
    a_rerun_that_diverges_ends_with_an_error,
    a_held_call_whose_key_made_another_call_diverges,
    long_output_and_images_go_to_files,
    the_section_lists_tools_and_scripts_find_the_rest,
);

/// Counts how often each test tool and global ran.
#[derive(Clone, Default)]
struct Runs(Arc<Mutex<Vec<String>>>);

impl Runs {
    fn record(&self, name: &str) -> usize {
        let mut runs = self.0.lock().unwrap();
        runs.push(name.to_owned());
        runs.iter().filter(|run| *run == name).count()
    }

    fn count(&self, name: &str) -> usize {
        self.0.lock().unwrap().iter().filter(|run| *run == name).count()
    }
}

/// `lookup` (no schema): `found <q>`; with `fail`, an error result with text.
fn lookup(runs: &Runs) -> Arc<dyn HarnessTool> {
    let runs = runs.clone();
    Arc::new(FnHarnessTool::new(
        Tool::new("lookup", "Looks a word up", serde_json::json!({ "type": "object" })),
        move |arguments, _api, _ctx| {
            runs.record("lookup");
            Box::pin(async move {
                let q = arguments
                    .get("q")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                if arguments.get("fail").is_some() {
                    return Ok(ToolExecutionResult::error(format!("lookup failed for {q}")));
                }
                Ok(ToolExecutionResult::text(format!("found {q}")))
            })
        },
    ))
}

/// `probe` (structured `{ code }`): an error result with its value when `code` is not 0, as
/// `bash` with a nonzero exit; with `bare`, an error result without one.
fn probe() -> Arc<dyn HarnessTool> {
    let tool = FnHarnessTool::new(
        Tool::new("probe", "Probes", serde_json::json!({ "type": "object" })),
        |arguments, _api, _ctx| {
            Box::pin(async move {
                if arguments.get("bare").is_some() {
                    return Ok(ToolExecutionResult::error("probe broke"));
                }
                let code = arguments.get("code").and_then(Value::as_i64).unwrap_or(0);
                Ok(ToolExecutionResult {
                    structured_output: Some(serde_json::json!({ "code": code })),
                    is_error: code != 0,
                    ..ToolExecutionResult::text(format!("code {code}"))
                })
            })
        },
    )
    .with_structured_output_schema(serde_json::json!({
        "type": "object",
        "properties": { "code": { "type": "integer" } },
        "required": ["code"],
    }));
    Arc::new(tool)
}

/// `count`: answers how often it ran.
fn count(runs: &Runs) -> Arc<dyn HarnessTool> {
    let runs = runs.clone();
    Arc::new(FnHarnessTool::new(
        Tool::new("count", "Counts", serde_json::json!({ "type": "object" })),
        move |_arguments, _api, _ctx| {
            let run = runs.record("count");
            Box::pin(async move { Ok(ToolExecutionResult::text(format!("count {run}"))) })
        },
    ))
}

/// `slow` (replay-safe): on its first `stuck_runs` executions reports a step and waits until
/// its context ends; then answers `slow done`.
fn slow(runs: &Runs, stuck_runs: usize) -> Arc<dyn HarnessTool> {
    let runs = runs.clone();
    Arc::new(
        FnHarnessTool::new(
            Tool::new("slow", "Takes a while", serde_json::json!({ "type": "object" })),
            move |_arguments, api, ctx| {
                let run = runs.record("slow");
                Box::pin(async move {
                    if run > stuck_runs {
                        return Ok(ToolExecutionResult::text("slow done"));
                    }
                    api.details(serde_json::json!({ "run": run }), ctx).await?;
                    ctx.run(std::future::pending::<()>()).await?;
                    Ok(ToolExecutionResult::default())
                })
            },
        )
        .with_replay(ToolReplay::Safe),
    )
}

/// `host.stamp()`: answers how often it ran.
struct Stamp(Runs);

impl CodemodeGlobal for Stamp {
    #[allow(clippy::unnecessary_literal_bound)] // The trait's signature.
    fn name(&self) -> &str {
        "host.stamp"
    }

    fn execute<'a>(
        &'a self,
        args: Option<Value>,
        _api: &'a ToolApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, Result<Option<Value>, ToolError>> {
        Box::pin(async move {
            if args.is_some_and(|args| args.get("fail").is_some()) {
                return Err("stamp refused".into());
            }
            Ok(Some(Value::from(self.0.record("stamp"))))
        })
    }
}

/// Installs code mode with the test tools and `host.stamp`.
fn install(fixture: &Fixture, runs: &Runs, stuck_runs: usize) {
    let sandbox = Arc::new(Sandbox::new(SandboxOptions::default()));
    let options = CodemodeOptions::new(sandbox)
        .with_global(Arc::new(Stamp(runs.clone())))
        .with_namespace(CodemodeNamespace::new("probes", "probe").with_description("Probing tools."));
    fixture
        .registry
        .install(codemode_extension(options).with_tool(lookup(runs)))
        .unwrap();
    fixture
        .registry
        .install(
            Extension::new("tools")
                .with_tool(probe())
                .with_tool(count(runs))
                .with_tool(slow(runs, stuck_runs)),
        )
        .unwrap();
}

/// The tool results of the transcript, in order.
async fn results(conversation: &Conversation) -> Vec<ToolResultMessage> {
    entries(conversation)
        .await
        .into_iter()
        .filter(|entry| entry.kind == "delta.tool-result")
        .map(|entry| match entry.model.as_deref() {
            Some([Message::ToolResult(result)]) => result.clone(),
            other => panic!("a tool result: {other:?}"),
        })
        .collect()
}

/// The text of a result without its header (which carries the wall time).
fn body(result: &ToolResultMessage) -> String {
    let texts: Vec<&str> = result
        .content
        .iter()
        .filter_map(|block| match block {
            UserBlock::Text(text) => Some(text.text.as_str()),
            UserBlock::Image(_) => None,
        })
        .collect();
    assert!(
        texts[0].starts_with("Script completed\nWall time ") || texts[0].starts_with("Script failed\nWall time "),
        "{texts:?}"
    );
    texts[1..].join("")
}

/// The first text of a result.
fn first_text(result: &ToolResultMessage) -> &str {
    match &result.content[0] {
        UserBlock::Text(text) => &text.text,
        UserBlock::Image(_) => panic!("an image"),
    }
}

fn details(result: &ToolResultMessage) -> CodemodeToolDetails {
    serde_json::from_value(result.details.clone().unwrap()).unwrap()
}

/// Runs `code` as one `codemode` call and returns its result.
async fn script(fixture: &Fixture, harness: &Harness, code: &str) -> ToolResultMessage {
    fixture.model.push([
        Reply::calls_with(&[("c1", "codemode", serde_json::json!({ "code": code }))]),
        Reply::text("Done"),
    ]);
    let conversation = chat(harness).await;
    let before = results(&conversation).await.len();
    let submission = conversation.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    let results = results(&conversation).await;
    assert_eq!(results.len(), before + 1);
    results.last().unwrap().clone()
}

async fn a_script_calls_tools_in_parallel(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Runs::default();
    install(&fixture, &runs, 0);
    let harness = fixture.open().await;
    let result = script(
        &fixture,
        &harness,
        r#"
const words = ["a", "b", "c"];
const found = await Promise.all(words.map((q) => tools.lookup({ q })));
text(found.join(", "));
console.log("tools:", ALL_TOOLS.map((tool) => tool.name).sort().join(" "));
return { probe: await tools.probe({ code: 0 }) };
"#,
    )
    .await;
    assert!(!result.is_error);
    assert_eq!(
        body(&result),
        "==> text 1/2 <==\nfound a, found b, found c\n==> text 2/2 <==\n{\"probe\":{\"code\":0}}\n<console_output>\ntools: count lookup probe slow\n</console_output>"
    );
    let details = details(&result);
    let calls: Vec<(u64, &str, &str, &str)> = details
        .calls
        .iter()
        .map(|call| (call.id, call.name.as_str(), call.args.as_str(), call.status.as_str()))
        .collect();
    assert_eq!(
        calls,
        [
            (1, "lookup", r#"{"q":"a"}"#, "ok"),
            (2, "lookup", r#"{"q":"b"}"#, "ok"),
            (3, "lookup", r#"{"q":"c"}"#, "ok"),
            (4, "probe", r#"{"code":0}"#, "ok"),
        ]
    );
    // Only the script's result is in the transcript; its nested calls left no slots.
    let conversation = chat(&harness).await;
    assert_eq!(results(&conversation).await.len(), 1);
    assert!(live(&harness, &conversation).await.nested_tools.is_none());
    harness.close(&ctx()).await.unwrap();
}

async fn results_map_as_tools_declare(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Runs::default();
    install(&fixture, &runs, 0);
    let harness = fixture.open().await;
    let result = script(
        &fixture,
        &harness,
        r#"
const exit = await tools.probe({ code: 2 });
text(`structured error resolves: ${JSON.stringify(exit)}`);
for (const call of [() => tools.lookup({ q: "x", fail: true }), () => tools.probe({ bare: true })]) {
  try {
    await call();
  } catch (error) {
    text(`${error.message} / ${JSON.stringify(error.structuredOutput)}`);
  }
}
throw new TypeError("last");
"#,
    )
    .await;
    assert!(result.is_error);
    let body = body(&result);
    assert!(
        body.starts_with(
            "==> text 1/3 <==\nstructured error resolves: {\"code\":2}\n==> text 2/3 <==\nlookup failed for x / \"lookup failed for x\"\n==> text 3/3 <==\nprobe broke / undefined\nScript error:\nTypeError: last\n"
        ),
        "{body}"
    );
    assert!(
        body.ends_with(
            "Tool calls made before the failure (they are not undone): probe (error), lookup (error), probe (error)"
        ),
        "{body}"
    );
    let details = details(&result);
    assert_eq!(details.calls[1].error.as_deref(), Some("lookup failed for x"));
    assert_eq!(details.calls[2].error.as_deref(), Some("probe broke"));
    harness.close(&ctx()).await.unwrap();
}

async fn the_store_keeps_a_successful_scripts_writes(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Runs::default();
    install(&fixture, &runs, 0);
    let harness = fixture.open().await;
    let stored = script(
        &fixture,
        &harness,
        r#"store("n", 1); store("gone", true); return load("n");"#,
    )
    .await;
    assert_eq!(body(&stored), "1");
    let failed = script(&fixture, &harness, r#"store("n", 99); throw new Error("no");"#).await;
    assert!(failed.is_error);
    let next = script(
        &fixture,
        &harness,
        r#"store("n", load("n") + 1); store("gone", undefined); return [load("n"), load("gone")];"#,
    )
    .await;
    assert_eq!(body(&next), "[2,null]");
    let conversation = chat(&harness).await;
    let value = harness
        .snapshot(DocAddress::<CodemodeStoreDoc>::new(conversation.id()), &ctx())
        .await
        .unwrap()
        .unwrap();
    let store: StoreState = delta_chord::json::from_json(&value).unwrap();
    assert_eq!(Value::Object(store.values), serde_json::json!({ "n": 2 }));
    harness.close(&ctx()).await.unwrap();
}

async fn globals_answer_and_unknown_tools_throw(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Runs::default();
    install(&fixture, &runs, 0);
    let harness = fixture.open().await;
    let result = script(
        &fixture,
        &harness,
        r#"
text(String(await host.stamp()));
try { await host.stamp({ fail: true }); } catch (error) { text(error.message); }
try { tools.codemode({ code: "" }); } catch (error) { text(error.constructor.name); }
try { await tools.lookup("word"); } catch (error) { text(error.message); }
"#,
    )
    .await;
    assert_eq!(
        body(&result),
        "==> text 1/4 <==\n1\n==> text 2/4 <==\nstamp refused\n==> text 3/4 <==\nTypeError\n==> text 4/4 <==\nTool lookup expects an object of arguments"
    );
    // Globals are not tool calls; a call refused for its arguments made no nested call.
    let calls: Vec<&str> = details(&result).calls.iter().map(|call| call.status.as_str()).collect();
    assert_eq!(calls, ["error"]);
    assert_eq!(runs.count("lookup"), 0);
    harness.close(&ctx()).await.unwrap();
}

/// Denies `count`, allows the rest.
struct DenyCount;

impl ApprovalPolicy for DenyCount {
    fn decide<'a>(
        &'a self,
        call: &'a ApprovalCall<'a>,
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, Result<PolicyDecision, PolicyError>> {
        Box::pin(async move {
            Ok(if call.call.call.name == "count" {
                PolicyDecision::Deny {
                    reason: "counting is off".to_owned(),
                }
            } else {
                PolicyDecision::Allow
            })
        })
    }
}

async fn nested_calls_are_approved_one_by_one(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Runs::default();
    install(&fixture, &runs, 0);
    let mut options = fixture.options();
    options.approval = Arc::new(DenyCount);
    let harness = Harness::open(fixture.storage.clone(), options, &ctx()).await.unwrap();
    let result = script(
        &fixture,
        &harness,
        r#"
const word = await tools.lookup({ q: "ok" });
try { await tools.count({}); } catch (error) { text(`${word}; ${error.message}`); }
"#,
    )
    .await;
    assert_eq!(
        body(&result),
        "found ok; Tool call blocked: approval denied: counting is off"
    );
    assert_eq!(runs.count("count"), 0);
    harness.close(&ctx()).await.unwrap();
}

async fn options_and_timeouts(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Runs::default();
    install(&fixture, &runs, 9);
    let harness = fixture.open().await;
    let invalid = script(&fixture, &harness, "// @options: {\"colour\": 1}\nreturn 1;").await;
    assert!(invalid.is_error);
    assert_eq!(
        first_text(&invalid),
        "<harness>\n[error] @options only supports `max_output_tokens` and `timeout_ms`; got `colour`\n</harness>"
    );
    let spinning = script(
        &fixture,
        &harness,
        "// @options: {\"timeout_ms\": 200}\nwhile (true) {}",
    )
    .await;
    assert!(spinning.is_error);
    assert!(
        body(&spinning).starts_with("Script error:\nScript timed out: "),
        "{}",
        body(&spinning)
    );
    let waiting = script(
        &fixture,
        &harness,
        "// @options: {\"timeout_ms\": 1000}\nawait tools.slow({});",
    )
    .await;
    assert!(
        body(&waiting).starts_with("Script error:\nScript timed out: "),
        "{}",
        body(&waiting)
    );
    // A loaded machine may reach the deadline before the script made its call.
    let calls: Vec<&str> = details(&waiting)
        .calls
        .iter()
        .map(|call| call.status.as_str())
        .collect();
    assert!(
        calls.len() <= 1 && calls.iter().all(|status| *status == "cancelled"),
        "{calls:?}"
    );
    harness.close(&ctx()).await.unwrap();
}

async fn an_abort_ends_the_script_and_its_calls(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Runs::default();
    install(&fixture, &runs, 9);
    fixture.model.push([Reply::calls_with(&[(
        "c1",
        "codemode",
        serde_json::json!({ "code": "await tools.slow({});" }),
    )])]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    slow_started(&harness, &conversation).await;
    tokio::time::timeout(PATIENCE, conversation.abort(AbortOptions::default(), &ctx()))
        .await
        .unwrap()
        .unwrap();
    let state = settled(&submission).await.state;
    assert!(matches!(&state, SubmissionState::Unanswered { reason, .. } if reason == "aborted"));
    let results = results(&conversation).await;
    assert_eq!(
        first_text(&results[0]),
        "<harness>\n[error] Tool codemode was aborted\n</harness>"
    );
    let state = live(&harness, &conversation).await;
    assert!(state.nested_tools.is_none() && state.tools.is_none());
    let inspection = harness.inspect(&ctx()).await.unwrap();
    assert!(inspection.scheduler.tasks.is_empty(), "{inspection:?}");
    harness.close(&ctx()).await.unwrap();
}

/// Waits until a `slow` nested call published its step; returns the code mode call's task.
async fn slow_started(harness: &Harness, conversation: &Conversation) -> TaskId {
    let found = Arc::new(Mutex::new(None));
    eventually(|| {
        let found = found.clone();
        async move {
            let slot = live(harness, conversation)
                .await
                .nested_tools
                .into_iter()
                .flatten()
                .find(|slot| slot.name == "slow" && slot.progress.details.is_some());
            let parent = slot.map(|slot| slot.parent_task_id);
            *found.lock().unwrap() = parent;
            parent.is_some()
        }
    })
    .await;
    found.lock().unwrap().unwrap()
}

/// The script of the crash cases: a call, a global, then two parallel calls, one of them stuck
/// on the first run.
const CRASHING: &str = r#"
const first = await tools.count({});
const stamp = await host.stamp();
const [second, slow] = await Promise.all([tools.count({}), tools.slow({})]);
text(`${first}, stamp ${stamp}, ${second}, ${slow}, ${Math.random() < 2}, ${Date.now() > 0}`);
store("done", true);
return "end";
"#;

/// Starts one `codemode` call of [`CRASHING`], waits until its `slow` call published its step,
/// and crashes; returns the submission and the code mode call's task.
async fn crash_mid_script(fixture: &Fixture) -> (SubmissionId, TaskId) {
    fixture.model.push([Reply::calls_with(&[(
        "c1",
        "codemode",
        serde_json::json!({ "code": CRASHING }),
    )])]);
    let harness = fixture.open().await;
    let conversation = chat(&harness).await;
    let submission = conversation.submit(SubmissionDraft::input("go"), &ctx()).await.unwrap();
    let id = submission.id();
    let task = slow_started(&harness, &conversation).await;
    // The parallel count's result is journaled and acknowledged too.
    eventually(|| async {
        let journal = journal(&harness, task).await;
        journal.entries.len() == 3 && journal.entries[2].issued_after.is_some()
    })
    .await;
    let closed = harness.session().closed();
    drop((submission, conversation, harness));
    tokio::time::timeout(PATIENCE, closed)
        .await
        .expect("the old Session ends");
    (id, task)
}

/// The journal of the code mode call `task`.
async fn journal(harness: &Harness, task: TaskId) -> JournalState {
    let value = harness
        .snapshot(DocAddress::<CodemodeJournalDoc>::new(task), &ctx())
        .await
        .unwrap()
        .unwrap();
    delta_chord::json::from_json(&value).unwrap()
}

/// Reopens, lets the submission settle and returns the code mode result.
async fn resume(fixture: &Fixture, harness: &Harness, id: SubmissionId) -> ToolResultMessage {
    fixture.model.push([Reply::text("Done")]);
    let submission = harness.submission(id, &ctx()).await.unwrap().unwrap();
    assert!(matches!(settled(&submission).await.state, SubmissionState::Done { .. }));
    let conversation = chat(harness).await;
    results(&conversation).await.remove(0)
}

async fn a_crashed_script_resumes_without_running_finished_calls_again(backend: Backend) {
    let uncrashed = {
        let fixture = Fixture::new(backend).await;
        let runs = Runs::default();
        install(&fixture, &runs, 0);
        let harness = fixture.open().await;
        let result = script(&fixture, &harness, CRASHING).await;
        harness.close(&ctx()).await.unwrap();
        body(&result)
    };
    assert_eq!(
        uncrashed,
        "==> text 1/2 <==\ncount 1, stamp 1, count 2, slow done, true, true\n==> text 2/2 <==\nend"
    );

    let fixture = Fixture::new(backend).await;
    let runs = Runs::default();
    install(&fixture, &runs, 1);
    let (id, task) = crash_mid_script(&fixture).await;
    let harness = fixture.open().await;
    // The journal as the crash left it: the first count, the stamp, the parallel count; the
    // slow call launched without a result.
    let journal = journal(&harness, task).await;
    let journaled: Vec<(u64, Option<u64>, bool)> = journal
        .entries
        .iter()
        .map(|entry| (entry.id, entry.issued_after, entry.global.is_some()))
        .collect();
    assert_eq!(
        journaled,
        [(1, Some(2), false), (2, Some(4), true), (3, Some(4), false)]
    );
    assert_eq!(journal.first_issued, Some(1));

    let result = resume(&fixture, &harness, id).await;
    assert!(!result.is_error, "{result:?}");
    assert_eq!(
        body(&result),
        uncrashed,
        "the resumed run saw what an uncrashed run sees"
    );
    assert_eq!(runs.count("count"), 2, "each count ran once");
    assert_eq!(runs.count("stamp"), 1, "the global's answer came from the journal");
    assert_eq!(runs.count("slow"), 2, "the stuck call reran once, under its own task");
    let calls: Vec<&str> = details(&result).calls.iter().map(|call| call.status.as_str()).collect();
    assert_eq!(calls, ["ok", "ok", "ok"]);
    let conversation = chat(&harness).await;
    let store = harness
        .snapshot(DocAddress::<CodemodeStoreDoc>::new(conversation.id()), &ctx())
        .await
        .unwrap()
        .unwrap();
    let store: StoreState = delta_chord::json::from_json(&store).unwrap();
    assert_eq!(Value::Object(store.values), serde_json::json!({ "done": true }));
    let inspection = harness.inspect(&ctx()).await.unwrap();
    assert!(inspection.scheduler.tasks.is_empty(), "{inspection:?}");
    harness.close(&ctx()).await.unwrap();
}

/// Edits the run document of `task` while nothing runs.
async fn edit_run(harness: &Harness, task: TaskId, edit: impl FnOnce(&mut RunState) + Send + 'static) {
    harness
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let mut run = tx.doc(DocAddress::<CodemodeRunDoc>::new(task)).await?;
                run.update_as::<RunState, _>(edit)
                    .map_err(delta_harness::session::TxError::callback)?;
                Ok(())
            })
        })
        .await
        .unwrap();
}

async fn a_rerun_returns_a_stored_result(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Runs::default();
    install(&fixture, &runs, 1);
    let (id, task) = crash_mid_script(&fixture).await;
    let harness = fixture.open().await;
    edit_run(&harness, task, |run| {
        run.result = Some(StoredResult {
            output: vec![UserBlock::text("decided before the crash")],
            is_error: false,
            details: None,
        });
    })
    .await;
    let result = resume(&fixture, &harness, id).await;
    assert_eq!(first_text(&result), "decided before the crash");
    assert_eq!(runs.count("count"), 2, "no VM ran again");
    harness.close(&ctx()).await.unwrap();
}

async fn a_rerun_refuses_another_version_fingerprint_or_deadline(backend: Backend) {
    type Edit = fn(&mut RunState);
    let cases: [(Edit, &str); 3] = [
        (
            |run| run.start.as_mut().unwrap().version = 99,
            "The script cannot resume: it started under code mode version 99, this is ",
        ),
        (
            |run| run.start.as_mut().unwrap().fingerprint = Some("another VM".to_owned()),
            "The script cannot resume: the sandbox changed since it started (another VM -> ",
        ),
        (
            |run| {
                let start = run.start.as_mut().unwrap();
                start.deadline = Some(start.started_at + 1);
            },
            "Script timed out: its deadline passed while it was interrupted.",
        ),
    ];
    for (edit, expected) in cases {
        let fixture = Fixture::new(backend).await;
        let runs = Runs::default();
        install(&fixture, &runs, 1);
        let (id, task) = crash_mid_script(&fixture).await;
        let harness = fixture.open().await;
        edit_run(&harness, task, edit).await;
        let result = resume(&fixture, &harness, id).await;
        assert!(result.is_error);
        assert!(first_text(&result).starts_with(expected), "{}", first_text(&result));
        if expected.contains("version") {
            assert!(first_text(&result).ends_with(&format!("this is {CODEMODE_MODULE_VERSION}.")));
        }
        // Nothing new ran: the counts before the crash stay the only ones.
        assert_eq!(runs.count("count"), 2);
        let calls: Vec<&str> = details(&result).calls.iter().map(|call| call.status.as_str()).collect();
        assert!(calls.iter().all(|status| *status != "running"), "{calls:?}");
        harness.close(&ctx()).await.unwrap();
    }
}

async fn a_rerun_that_diverges_ends_with_an_error(backend: Backend) {
    type Edit = fn(&mut JournalState);
    let cases: [(Edit, &str); 3] = [
        (
            |journal| journal.entries[0].issued_after = Some(7),
            "The script took a different path when resumed after a restart: after result 1 it had made 2 calls, where it had made 7",
        ),
        (
            |journal| journal.first_issued = Some(3),
            "The script took a different path when resumed after a restart: its first run made 1 calls, where it had made 3",
        ),
        (
            |journal| "host.other".clone_into(&mut journal.entries[1].global.as_mut().unwrap().name),
            "The script took a different path when resumed after a restart: call 2 is host.stamp(), but was a call to host.other",
        ),
    ];
    for (edit, expected) in cases {
        let fixture = Fixture::new(backend).await;
        let runs = Runs::default();
        install(&fixture, &runs, 1);
        let (id, task) = crash_mid_script(&fixture).await;
        let harness = fixture.open().await;
        harness
            .commit(&ctx(), move |tx| {
                Box::pin(async move {
                    let mut journal = tx.doc(DocAddress::<CodemodeJournalDoc>::new(task)).await?;
                    journal
                        .update_as::<JournalState, _>(edit)
                        .map_err(delta_harness::session::TxError::callback)?;
                    Ok(())
                })
            })
            .await
            .unwrap();
        let result = resume(&fixture, &harness, id).await;
        assert!(result.is_error);
        assert_eq!(first_text(&result), expected);
        assert_eq!(runs.count("count"), 2, "the diverged script admitted no new call");
        harness.close(&ctx()).await.unwrap();
    }
}

async fn a_held_call_whose_key_made_another_call_diverges(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Runs::default();
    install(&fixture, &runs, 1);
    let (id, task) = crash_mid_script(&fixture).await;
    let harness = fixture.open().await;
    // The slow call (4) has no journal entry, so the rerun holds it and checks its key.
    harness
        .commit(&ctx(), move |tx| {
            Box::pin(async move {
                let mut member = tx
                    .doc_member(DocAddress::<NestedCallDoc>::member(task, "c4"), ())
                    .await?;
                member
                    .update_as::<NestedCallState, _>(|state| state.name = Some("lookup".to_owned()))
                    .map_err(delta_harness::session::TxError::callback)?;
                Ok(())
            })
        })
        .await
        .unwrap();
    let result = resume(&fixture, &harness, id).await;
    assert!(result.is_error);
    assert_eq!(
        first_text(&result),
        "The script took a different path when resumed after a restart: call 4 is tools.slow({}), but was tools.lookup({})"
    );
    assert_eq!(runs.count("lookup"), 0);
    harness.close(&ctx()).await.unwrap();
}

/// A 1x1 PNG.
const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

async fn long_output_and_images_go_to_files(backend: Backend) {
    let dir = support::TempDir::new();
    let mut fixture = Fixture::new(backend).await;
    fixture.env = Some(Arc::new(LocalEnvBuilder::new(dir.path())));
    let runs = Runs::default();
    install(&fixture, &runs, 0);
    let harness = fixture.open().await;
    let code = format!(
        "// @options: {{\"max_output_tokens\": 50}}\nfor (let i = 0; i < 100; i++) text(`line ${{i}}`);\nimage(\"data:image/png;base64,{PNG}\");"
    );
    let result = script(&fixture, &harness, &code).await;
    assert!(!result.is_error);
    let body = body(&result);
    assert!(
        body.starts_with("Warning: truncated output (original token count: "),
        "{body}"
    );
    let full = details(&result).full_output_path.unwrap();
    assert!(
        body.contains(&format!("[Full output: {full} (read with offset/limit)]")),
        "{body}"
    );
    let text = std::fs::read_to_string(&full).unwrap();
    assert!(
        text.starts_with("==> text 1/100 <==\nline 0\n") && text.ends_with("line 99"),
        "{text}"
    );
    let image = std::path::Path::new(&full).with_file_name("image-1.png");
    assert!(
        body.contains(&format!("[Image saved to {} (image/png, ", image.display())),
        "{body}"
    );
    assert_eq!(
        std::fs::read(&image).unwrap(),
        base64::Engine::decode(&base64::engine::general_purpose::STANDARD, PNG).unwrap()
    );
    assert!(result.content.iter().any(|block| matches!(block, UserBlock::Image(_))));
    std::fs::remove_dir_all(std::path::Path::new(&full).parent().unwrap()).unwrap();
    harness.close(&ctx()).await.unwrap();
}

async fn the_section_lists_tools_and_scripts_find_the_rest(backend: Backend) {
    let fixture = Fixture::new(backend).await;
    let runs = Runs::default();
    install(&fixture, &runs, 0);
    let harness = fixture.open().await;
    let result = script(
        &fixture,
        &harness,
        r#"
const names = (found) => found.map((tool) => tool.name).join(",");
text(names(await searchTools("word")));
text(names(await searchTools("probes", { namespace: "probes" })));
text(JSON.stringify(await describeNamespace("probes")));
text((await describeTool("count")).split("\n")[0]);
text(String(await describeTool("nothing")));
try { await searchTools(3); } catch (error) { text(error.message); }
try { await searchTools("x", { limit: 0 }); } catch (error) { text(error.message); }
"#,
    )
    .await;
    let texts: Vec<String> = body(&result)
        .split("\n==> ")
        .map(|item| item.split_once("<==\n").unwrap().1.to_owned())
        .collect();
    assert_eq!(
        texts,
        [
            "lookup",
            "probe",
            r#"{"name":"probes","description":"Probing tools.","tools":["probe"]}"#,
            "Counts",
            "undefined",
            "searchTools() expects a query string",
            "searchTools() limit must be a positive integer",
        ]
    );
    // The system prompt lists the callable tools by namespace; `codemode` itself is not one.
    let conversation = chat(&harness).await;
    let section = entries(&conversation)
        .await
        .into_iter()
        .find_map(|entry| match entry.model.as_deref() {
            Some([Message::System(system)]) => system
                .sections
                .iter()
                .flatten()
                .find(|(key, _)| key.as_str() == "codemode")
                .and_then(|(_, text)| text.clone()),
            _ => None,
        })
        .unwrap();
    assert!(
        section.starts_with(
            "<codemode>\nTools scripts can call (`await tools.<name>(args)`):\n\n### `lookup`\nLooks a word up"
        ),
        "{section}"
    );
    assert!(
        section.contains("\n\n## probes\nProbing tools.\n\n### `probe`\nProbes"),
        "{section}"
    );
    assert!(!section.contains("### `codemode`"), "{section}");
    harness.close(&ctx()).await.unwrap();
}
