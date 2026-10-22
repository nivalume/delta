//! One execution of the `codemode` tool: a first run, or a rerun after a restart that replays
//! the journal. Port of upstream `CodemodeCall` (`codemode/codemode.ts`); see
//! `docs/design/codemode.md`.
//!
//! The host side is one loop that owns the call's state: it takes the run's events and the
//! results of the calls it launched one at a time, and awaits its commits inline, so one commit
//! is in flight at a time and nothing is delivered or launched before the fingerprint is
//! checked. What arrives meanwhile waits in the run's channel or in the set of running calls.

use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasher, Hasher};
use std::sync::Arc;
use std::time::Duration;

use delta_ai::types::{JsonObject, UserBlock};
use delta_chord::context::Context;
use delta_chord::delta::Draft;
use delta_chord::json::{self, Json};
use delta_codemode::{
    CallTarget, CodemodeResult, DeclaredGlobal, DeclaredTool, HostCall, Outcome, Run, RunEvent, RunHandle, RunOptions,
    parse_codemode_source, to_codemode_identifier,
};
use futures::FutureExt as _;
use futures::future::BoxFuture;
use futures::stream::{FuturesUnordered, StreamExt};
use serde_json::{Map, Value};
use tokio::sync::Semaphore;

use super::describe::sample_of;
use super::documents::{
    CallStatus, CodemodeJournalDoc, CodemodeNestedCall, CodemodeRunDoc, CodemodeStart, CodemodeStoreDoc,
    CodemodeToolDetails, JournalEntry, JournalState, JournaledGlobal, RunState, StartNamespace, StartTool, StoreState,
    StoredResult,
};
use super::models::{
    CATALOG_GLOBALS, GENERATE_IMAGES_TOOL, MAX_CONCURRENT_MODEL_CALLS, MODEL_CALLS, catalog_call, count_images,
    model_call, model_call_arguments,
};
use super::output::{self, CallSummary, FormatOptions, Limits, OutputFile};
use super::{CODEMODE_INTERNAL_PREFIX, CODEMODE_MODULE_VERSION, CodemodeOptions, namespace_of};
use crate::documents::DocAddress;
use crate::search::{SearchDocument, rank_documents};
use crate::session::TxError;
use crate::tool::{
    NestedCallDoc, NestedCallOptions, NestedCallState, NestedToolExecutionResult, Severity, ToolApi, ToolApiError,
    ToolError, ToolExecutionResult,
};

/// Globals for finding tools, answered from the start data.
const DISCOVERY_GLOBALS: [&str; 3] = ["searchTools", "describeTool", "describeNamespace"];
const DEFAULT_SEARCH_LIMIT: usize = 8;

/// Characters of a call's arguments and error kept for `details`.
const ARGS_PREVIEW_CHARS: usize = 200;
const ERROR_PREVIEW_CHARS: usize = 500;

/// The tool's output limits, which the formatted result stays inside.
pub(super) const LIMITS: Limits = Limits {
    max_bytes: 4 * 1024 * 1024,
    max_lines: 100_000,
};

/// `text` cut to `max` characters, ending in `...` when cut.
fn cut(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    let kept: String = text.chars().take(max.saturating_sub(3)).collect();
    format!("{kept}...")
}

/// `tools.<name>(<args>)` or `<global>(<args>)`, as divergence messages name a call.
fn describe(call: &HostCall) -> String {
    let prefix = if call.target == CallTarget::Tool { "tools." } else { "" };
    let args = call.args.as_ref().map(Value::to_string).unwrap_or_default();
    format!("{prefix}{}({args})", call.name)
}

/// An error result with `message`.
fn error_result(message: impl Into<String>) -> StoredResult {
    StoredResult {
        output: vec![UserBlock::text(message.into())],
        is_error: true,
        details: None,
    }
}

impl From<StoredResult> for ToolExecutionResult {
    fn from(stored: StoredResult) -> Self {
        Self {
            output: Some(stored.output),
            is_error: stored.is_error,
            details: stored.details,
            ..Self::default()
        }
    }
}

/// The text of a schema-less tool's collapsed output: a string, one image, or an array of text
/// and image blocks.
fn text_of(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(items)) => {
            let texts: Vec<&str> = items
                .iter()
                .filter(|item| item.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|item| item.get("text").and_then(Value::as_str))
                .collect();
            texts.join("\n")
        }
        _ => String::new(),
    }
}

/// What a script gets for a nested call. A tool with a schema resolves with its value, also on
/// an error result that carries one (`bash`'s `{ output, exitCode }` for a nonzero exit); a
/// schema-less tool resolves with its text output, images left out. Any other error rejects
/// with its text: a schema-less tool's output, else its error diagnostics.
pub(super) fn outcome_of(name: &str, structured: bool, result: &NestedToolExecutionResult) -> Outcome {
    let value = if structured {
        result.structured_output.clone()
    } else {
        Some(Value::String(text_of(result.structured_output.as_ref())))
    };
    if !result.is_error || (structured && value.is_some()) {
        return Outcome::ok(value);
    }
    let text = if structured {
        String::new()
    } else {
        text_of(result.structured_output.as_ref())
    };
    let diagnostics = error_diagnostics(result);
    let message = if !text.is_empty() {
        text
    } else if !diagnostics.is_empty() {
        diagnostics
    } else {
        format!("Tool {name} failed")
    };
    Outcome::Err {
        message,
        properties: value.map(|value| Map::from_iter([("structuredOutput".to_owned(), value)])),
    }
}

fn error_diagnostics(result: &NestedToolExecutionResult) -> String {
    let messages: Vec<&str> = result
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity == Severity::Error)
        .map(|diagnostic| diagnostic.message.as_str())
        .collect();
    messages.join("\n")
}

/// A seed for `Math.random()`, from the standard library's per-process random keys.
fn random_seed() -> u32 {
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u64(u64::from(std::process::id()));
    #[allow(clippy::cast_possible_truncation)] // Any 32 bits of the hash do.
    let seed = hasher.finish() as u32;
    seed
}

/// Ends the run when the call's future is dropped, so a cancelled call stops its VM.
struct AbortOnDrop(RunHandle);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// What the first commit found.
enum Found {
    /// A stored outcome: returned without a VM.
    Done(StoredResult),
    Start {
        start: Box<CodemodeStart>,
        journal: JournalState,
        rerun: bool,
    },
}

/// Runs the `codemode` tool with `source`.
pub(super) async fn execute(
    options: &CodemodeOptions,
    source: &str,
    api: &ToolApi<'_>,
    ctx: &Context,
) -> Result<ToolExecutionResult, ToolError> {
    let parsed = parse_codemode_source(source)?;
    let agent = api.agent();
    let tools: Vec<StartTool> = agent
        .callable
        .iter()
        .map(|tool| StartTool {
            name: tool.name().to_owned(),
            js_name: to_codemode_identifier(tool.name()),
            description: sample_of(&**tool),
            structured: tool.structured_output_schema().is_some(),
            hidden: tool.name().starts_with(CODEMODE_INTERNAL_PREFIX),
        })
        .collect();
    let mut namespaces: Vec<StartNamespace> = Vec::new();
    for tool in agent
        .callable
        .iter()
        .filter(|tool| !tool.name().starts_with(CODEMODE_INTERNAL_PREFIX))
    {
        let Some(namespace) = namespace_of(tool.name(), &options.namespaces) else {
            continue;
        };
        let index = match namespaces.iter().position(|entry| entry.name == namespace.name) {
            Some(index) => index,
            None => {
                namespaces.push(StartNamespace {
                    name: namespace.name.clone(),
                    description: namespace.description.clone(),
                    instructions: namespace.instructions.clone(),
                    tools: Vec::new(),
                });
                namespaces.len() - 1
            }
        };
        namespaces[index].tools.push(tool.name().to_owned());
    }
    let model_globals: Vec<&str> = if options.models {
        CATALOG_GLOBALS
            .iter()
            .copied()
            .chain(MODEL_CALLS.iter().map(|(global, _)| *global))
            .collect()
    } else {
        Vec::new()
    };
    let globals = DISCOVERY_GLOBALS
        .iter()
        .chain(&model_globals)
        .map(|name| DeclaredGlobal {
            name: (*name).to_owned(),
            spread: true,
        })
        .chain(options.globals.iter().map(|global| DeclaredGlobal {
            name: global.name().to_owned(),
            spread: global.spread(),
        }))
        .collect();
    let now = api.now();
    let timeout = parsed.options.timeout_ms;
    let max_output_tokens = parsed
        .options
        .max_output_tokens
        .unwrap_or(output::DEFAULT_MAX_OUTPUT_TOKENS);
    let candidate = CodemodeStart {
        version: CODEMODE_MODULE_VERSION,
        started_at: now,
        seed: random_seed(),
        fingerprint: None,
        deadline: timeout.map(|timeout| now.saturating_add(i64::try_from(timeout).unwrap_or(i64::MAX))),
        tools,
        namespaces,
        globals,
        max_output_tokens,
        store: Map::new(),
    };
    let task = api.task_id();
    let conversation = api.conversation_id();
    let found = api
        .commit(ctx, move |tx| {
            Box::pin(async move {
                let run = tx.doc(DocAddress::<CodemodeRunDoc>::new(task)).await?;
                let state: RunState = json::from_json(run.value()).map_err(TxError::callback)?;
                if let Some(result) = state.result {
                    return Ok(Found::Done(result));
                }
                let rerun = state.start.is_some();
                let start = match state.start {
                    Some(start) => start,
                    None => {
                        let store = tx.doc(DocAddress::<CodemodeStoreDoc>::new(conversation)).await?;
                        let store: StoreState = json::from_json(store.value()).map_err(TxError::callback)?;
                        let start = CodemodeStart {
                            store: store.values,
                            ..candidate
                        };
                        let mut run = tx.doc(DocAddress::<CodemodeRunDoc>::new(task)).await?;
                        run.update_as::<RunState, _>(|state| state.start = Some(start.clone()))
                            .map_err(TxError::callback)?;
                        start
                    }
                };
                let journal = tx.doc(DocAddress::<CodemodeJournalDoc>::new(task)).await?;
                let journal: JournalState = json::from_json(journal.value()).map_err(TxError::callback)?;
                Ok(Found::Start {
                    start: Box::new(start),
                    journal,
                    rerun,
                })
            })
        })
        .await?;
    let (start, journal, rerun) = match found {
        Found::Done(result) => return Ok(result.into()),
        Found::Start { start, journal, rerun } => (*start, journal, rerun),
    };

    let mut call = Call::new(options, api, ctx, start, journal, rerun);
    if call.start.version != CODEMODE_MODULE_VERSION {
        let message = format!(
            "The script cannot resume: it started under code mode version {}, this is {CODEMODE_MODULE_VERSION}.",
            call.start.version
        );
        return Ok(call.decide(error_result(message)).await?.into());
    }
    let remaining = call.start.deadline.map(|deadline| deadline - api.now());
    if remaining.is_some_and(|remaining| remaining <= 0) {
        let message = "Script timed out: its deadline passed while it was interrupted.";
        return Ok(call.decide(error_result(message)).await?.into());
    }
    let run = options.sandbox.start(
        &parsed.code,
        RunOptions {
            #[allow(clippy::cast_precision_loss)] // Epoch ms are far below 2^52.
            started_at: call.start.started_at as f64,
            seed: call.start.seed,
            tools: call
                .start
                .tools
                .iter()
                .filter(|tool| !tool.hidden)
                .map(|tool| DeclaredTool {
                    name: tool.name.clone(),
                    js_name: Some(tool.js_name.clone()),
                    description: tool.description.clone(),
                })
                .collect(),
            globals: call.start.globals.clone(),
            store: call.start.store.clone(),
            timeout: remaining.map(|remaining| Duration::from_millis(remaining.unsigned_abs())),
            cancel: None,
        },
    )?;
    call.drive(run).await
}

/// How a tool call of the script ended, as a [`CodemodeNestedCall`] row records it.
struct Row(CodemodeNestedCall);

/// A call the loop launched, when it lands.
struct Landed {
    call: HostCall,
    /// Whether the result goes to the journal (live) or comes from it (replay).
    live: bool,
    result: CallResult,
}

enum CallResult {
    Tool(Result<NestedToolExecutionResult, ToolApiError>),
    /// A tool call refused without a nested call.
    Refused(Outcome),
    Global(Result<Outcome, ToolError>),
}

/// A finished live call waiting for its journal commit.
struct Finished {
    id: u64,
    at: i64,
    outcome: Outcome,
    global: Option<JournaledGlobal>,
}

/// One step of the loop.
enum Step {
    Event(Option<RunEvent>),
    Landed(Landed),
}

struct Call<'a> {
    options: &'a CodemodeOptions,
    api: &'a ToolApi<'a>,
    ctx: &'a Context,
    start: CodemodeStart,
    /// The journal as this attempt found it; replay delivers exactly these.
    replayed: Vec<JournalEntry>,
    /// Entries in the journal, replayed and live; the next delivery is `length + 1`.
    journal_length: usize,
    /// Replayed entries delivered so far.
    next_replayed: usize,
    replaying: bool,
    /// Whether an earlier attempt ran this call: its nested call keys may already exist.
    rerun: bool,
    /// Whether live calls may launch; before, they are held (see `release`).
    released: bool,
    /// The count an earlier attempt recorded after the script's first run.
    first_issued: Option<u64>,
    /// The highest issued count known to be committed: calls up to it may launch.
    durable_issued: u64,
    /// Live calls waiting for their issued count to be committed.
    waiting: Vec<HostCall>,
    /// Journal index of each replayed call id.
    replay_index: HashMap<u64, usize>,
    /// Calls the VM issued so far; its call ids are 1, 2, ...
    issued: u64,
    /// Replay: results of journaled calls, ready to deliver in journal order.
    ready: HashMap<u64, Outcome>,
    /// Replay: calls the journal does not have, admitted once replay is done.
    held: Vec<HostCall>,
    /// Live: finished calls waiting for their journal commit.
    queue: Vec<Finished>,
    /// Acknowledged counts to record, by result number (0: the script's first run).
    acks: Vec<(u64, u64)>,
    /// Whether the loop commits the queue and the counts before its next step.
    pump_due: bool,
    /// A divergence found where the loop could not await, decided before its next step.
    divergence: Option<String>,
    /// The outcome the call decided other than the script's own (divergence, mismatch).
    ended: Option<StoredResult>,
    /// Why the call failed.
    failure: Option<ToolError>,
    /// The script's tool calls for `details`, by VM call id.
    rows: BTreeMap<u64, Row>,
    running: FuturesUnordered<BoxFuture<'a, Landed>>,
    handle: Option<RunHandle>,
    /// Slots of `models.classify()` and `models.generateImages()` calls.
    model_slots: Arc<Semaphore>,
    /// Images `models.generateImages()` returned, to notice a script that never shows them.
    generated_images: usize,
}

impl<'a> Call<'a> {
    fn new(
        options: &'a CodemodeOptions,
        api: &'a ToolApi<'a>,
        ctx: &'a Context,
        start: CodemodeStart,
        journal: JournalState,
        rerun: bool,
    ) -> Self {
        let durable_issued = journal
            .entries
            .iter()
            .filter_map(|entry| entry.issued_after)
            .fold(journal.first_issued.unwrap_or(0), u64::max);
        Self {
            options,
            api,
            ctx,
            start,
            replay_index: journal
                .entries
                .iter()
                .enumerate()
                .map(|(index, entry)| (entry.id, index))
                .collect(),
            journal_length: journal.entries.len(),
            next_replayed: 0,
            replaying: !journal.entries.is_empty(),
            replayed: journal.entries,
            rerun,
            released: false,
            first_issued: journal.first_issued,
            durable_issued,
            waiting: Vec::new(),
            issued: 0,
            ready: HashMap::new(),
            held: Vec::new(),
            queue: Vec::new(),
            acks: Vec::new(),
            pump_due: false,
            divergence: None,
            ended: None,
            failure: None,
            rows: BTreeMap::new(),
            running: FuturesUnordered::new(),
            handle: None,
            model_slots: Arc::new(Semaphore::new(MAX_CONCURRENT_MODEL_CALLS)),
            generated_images: 0,
        }
    }

    /// Whether nothing more may happen: the call decided, diverged or failed.
    fn stopped(&self) -> bool {
        self.ended.is_some() || self.failure.is_some() || self.divergence.is_some()
    }

    async fn drive(mut self, mut run: Run) -> Result<ToolExecutionResult, ToolError> {
        let _guard = AbortOnDrop(run.handle());
        self.handle = Some(run.handle());
        let result = loop {
            let step = tokio::select! {
                biased;
                event = run.next_event() => Step::Event(event),
                Some(landed) = self.running.next(), if !self.running.is_empty() => Step::Landed(landed),
            };
            match step {
                Step::Event(Some(RunEvent::Started(fingerprint))) => self.check_fingerprint(fingerprint).await,
                Step::Event(Some(RunEvent::Call(call))) => self.on_call(call),
                Step::Event(Some(RunEvent::Idle { seq, issued })) => self.on_idle(seq, issued).await,
                Step::Event(Some(RunEvent::Finished(result))) => break *result,
                Step::Event(None) => {
                    break CodemodeResult {
                        error: Some(delta_codemode::CodemodeError::new(
                            delta_codemode::ErrorKind::Sandbox,
                            "The run ended without a result",
                        )),
                        ..CodemodeResult::default()
                    };
                }
                Step::Landed(landed) => {
                    self.on_landed(landed);
                    // Results that are in too join the same journal commit.
                    while let Some(Some(landed)) = self.running.next().now_or_never() {
                        self.on_landed(landed);
                    }
                }
            }
            if let Some(message) = self.divergence.take() {
                self.diverge(message).await;
            }
            if !self.stopped() && self.pump_due {
                self.pump_due = false;
                self.pump().await;
            }
            if let Some(failure) = self.failure.take() {
                return Err(failure);
            }
            if let Some(ended) = self.ended.take() {
                return Ok(ended.into());
            }
        };
        // A replay that ends where the live run went on is a divergence: within the replayed
        // part, the first run included (a rerun without journal entries replays that much too).
        if self.rerun
            && let Some(during) = result.during
            && usize::try_from(during.seq).is_ok_and(|seq| seq <= self.replayed.len())
        {
            let seq = usize::try_from(during.seq).unwrap_or(usize::MAX);
            // The live run went on: it acknowledged that result, or it made a call that later
            // results answer, which the replay never made.
            let acknowledged = if seq == 0 {
                !self.replayed.is_empty() || self.first_issued.is_some()
            } else {
                self.replayed[seq - 1].issued_after.is_some()
            };
            if acknowledged || self.replayed[seq..].iter().any(|entry| entry.id > during.issued) {
                let message = format!(
                    "The script took a different path when resumed after a restart: it ended after result {seq}, where it went on before."
                );
                return Ok(self.decide(error_result(message)).await?.into());
            }
        }
        Ok(self.complete(&result).await?.into())
    }

    fn deliver(&mut self, seq: usize, at: i64, id: u64, outcome: Outcome) {
        let Some(handle) = &self.handle else { return };
        #[allow(clippy::cast_precision_loss)] // Epoch ms are far below 2^52.
        let delivered = handle.deliver(seq as u64, at as f64, id, outcome);
        if let Err(error) = delivered {
            self.fail(Box::new(error));
        }
    }

    fn fail(&mut self, error: ToolError) {
        if self.failure.is_none() {
            self.failure = Some(error);
        }
    }

    /// Stores an outcome the call decided; returns it.
    async fn decide(&mut self, decided: StoredResult) -> Result<StoredResult, ToolError> {
        let result = StoredResult {
            details: Some(serde_json::to_value(self.details(None))?),
            ..decided
        };
        let task = self.api.task_id();
        let stored = result.clone();
        self.api
            .commit(self.ctx, move |tx| {
                Box::pin(async move {
                    let mut run = tx.doc(DocAddress::<CodemodeRunDoc>::new(task)).await?;
                    run.update_as::<RunState, _>(|state| {
                        state.result.get_or_insert(stored);
                    })
                    .map_err(TxError::callback)?;
                    Ok(())
                })
            })
            .await?;
        Ok(result)
    }

    /// Ends the call with `message`: stores it as the outcome; the loop then ends the VM with
    /// an error the script cannot catch.
    async fn diverge(&mut self, message: String) {
        if self.ended.is_some() || self.failure.is_some() {
            return;
        }
        let result = error_result(format!(
            "The script took a different path when resumed after a restart: {message}"
        ));
        match self.decide(result).await {
            Ok(result) => self.ended = Some(result),
            Err(error) => self.fail(error),
        }
    }

    /// The script's own outcome, stored with its store writes in one transaction.
    async fn complete(&mut self, result: &CodemodeResult) -> Result<StoredResult, ToolError> {
        let details = self.details(None);
        let calls: Vec<CallSummary<'_>> = details
            .calls
            .iter()
            .map(|call| CallSummary {
                name: &call.name,
                status: call.status.as_str(),
            })
            .collect();
        let mut files = RunFiles {
            api: self.api,
            ctx: self.ctx,
            dir: None,
        };
        let formatted = output::format_result(
            result,
            FormatOptions {
                generated_images: self.generated_images,
                wall_ms: self.api.now() - self.start.started_at,
                max_output_tokens: self.start.max_output_tokens,
                calls,
                limits: LIMITS,
            },
            &mut files,
        )
        .await;
        let details = CodemodeToolDetails {
            full_output_path: formatted.full_output_path,
            ..details
        };
        let tool = StoredResult {
            output: formatted.content,
            is_error: !result.is_ok(),
            details: Some(serde_json::to_value(details)?),
        };
        let writes = result.is_ok().then(|| result.store_writes.clone());
        let task = self.api.task_id();
        let conversation = self.api.conversation_id();
        let stored = self
            .api
            .commit(self.ctx, move |tx| {
                Box::pin(async move {
                    let mut run = tx.doc(DocAddress::<CodemodeRunDoc>::new(task)).await?;
                    let state: RunState = json::from_json(run.value()).map_err(TxError::callback)?;
                    if let Some(result) = state.result {
                        return Ok(result);
                    }
                    run.update_as::<RunState, _>(|state| state.result = Some(tool.clone()))
                        .map_err(TxError::callback)?;
                    if let Some(writes) = writes.filter(|writes| !writes.set.is_empty() || !writes.delete.is_empty()) {
                        let mut store = tx.doc(DocAddress::<CodemodeStoreDoc>::new(conversation)).await?;
                        let mut draft = store.draft();
                        let Some(mut values) = draft.field("values").and_then(Draft::into_object) else {
                            return Err(TxError::callback(std::io::Error::other(
                                "delta.codemode.store has no values",
                            )));
                        };
                        for key in &writes.delete {
                            values.remove(key);
                        }
                        for (key, value) in writes.set {
                            values.insert(key, Json::from(value));
                        }
                    }
                    Ok(tool)
                })
            })
            .await?;
        Ok(stored)
    }

    async fn check_fingerprint(&mut self, fingerprint: String) {
        match &self.start.fingerprint {
            Some(recorded) if *recorded != fingerprint => {
                let message = format!(
                    "The script cannot resume: the sandbox changed since it started ({recorded} -> {fingerprint})."
                );
                match self.decide(error_result(message)).await {
                    Ok(result) => self.ended = Some(result),
                    Err(error) => self.fail(error),
                }
                return;
            }
            Some(_) => {}
            None => {
                let task = self.api.task_id();
                let recorded = fingerprint.clone();
                let committed = self
                    .api
                    .commit(self.ctx, move |tx| {
                        Box::pin(async move {
                            let mut run = tx.doc(DocAddress::<CodemodeRunDoc>::new(task)).await?;
                            run.update_as::<RunState, _>(|state| {
                                if let Some(start) = &mut state.start {
                                    start.fingerprint = Some(recorded);
                                }
                            })
                            .map_err(TxError::callback)?;
                            Ok(())
                        })
                    })
                    .await;
                if let Err(error) = committed {
                    self.fail(Box::new(error));
                    return;
                }
                self.start.fingerprint = Some(fingerprint);
            }
        }
        // Nothing was delivered before: start now. A rerun without a journal releases its
        // calls at the script's first acknowledgment instead.
        if self.replaying {
            self.deliver_replayed();
        } else if !self.rerun {
            self.release().await;
        }
    }

    fn on_call(&mut self, call: HostCall) {
        if self.stopped() {
            return;
        }
        // A script's call of `models.classify()` or `models.generateImages()` is the call of its
        // internal tool.
        let call = match model_call(&call.name) {
            Some(tool) if call.target == CallTarget::Global => HostCall {
                id: call.id,
                target: CallTarget::Tool,
                name: tool.to_owned(),
                args: Some(model_call_arguments(call.args.as_ref())),
            },
            _ => call,
        };
        if call.target == CallTarget::Tool {
            self.started(&call);
        }
        self.issued = self.issued.max(call.id);
        let index = if self.replaying {
            self.replay_index.get(&call.id).copied()
        } else {
            None
        };
        let Some(index) = index else {
            if self.released {
                self.waiting.push(call);
                self.launch_covered();
            } else {
                self.held.push(call);
            }
            return;
        };
        let entry = &self.replayed[index];
        if let Some(global) = &entry.global {
            if call.target != CallTarget::Global || call.name != global.name || call.args != global.args {
                let message = format!(
                    "call {} is {}, but was a call to {}",
                    call.id,
                    describe(&call),
                    global.name
                );
                self.diverge_later(message);
                return;
            }
            self.ready.insert(call.id, global.outcome.clone());
            self.deliver_replayed();
            return;
        }
        if call.target != CallTarget::Tool {
            let message = format!("call {} is {}, but was a tool call", call.id, describe(&call));
            self.diverge_later(message);
            return;
        }
        self.launch_tool(call, false);
    }

    /// A divergence found where the loop cannot await: decided before its next step.
    fn diverge_later(&mut self, message: String) {
        self.divergence.get_or_insert(message);
    }

    /// Replay: delivers journaled results in journal order, each once the VM has issued its call.
    fn deliver_replayed(&mut self) {
        while self.next_replayed < self.replayed.len() && !self.stopped() {
            let entry = &self.replayed[self.next_replayed];
            if entry.id > self.issued || !self.ready.contains_key(&entry.id) {
                return;
            }
            let (id, at) = (entry.id, entry.at);
            let outcome = self.ready.remove(&id).unwrap_or_else(|| unreachable!());
            self.next_replayed += 1;
            self.deliver(self.next_replayed, at, id, outcome);
        }
    }

    async fn on_idle(&mut self, seq: u64, issued: u64) {
        if self.stopped() {
            return;
        }
        let index = usize::try_from(seq).unwrap_or(usize::MAX);
        // A count an earlier attempt recorded must match; a missing one is recorded now.
        if self.rerun && index <= self.replayed.len() {
            let recorded = if index == 0 {
                self.first_issued
            } else {
                self.replayed[index - 1].issued_after
            };
            if let Some(recorded) = recorded.filter(|recorded| *recorded != issued) {
                let message = if index == 0 {
                    format!("its first run made {issued} calls, where it had made {recorded}")
                } else {
                    format!("after result {seq} it had made {issued} calls, where it had made {recorded}")
                };
                self.diverge(message).await;
                return;
            }
        }
        self.acks.push((seq, issued));
        if !self.replaying {
            // A rerun without a journal: every call an earlier attempt could have admitted is
            // made by now.
            if index == 0 && self.rerun {
                self.release().await;
            }
            // Committed at once when the script has processed everything delivered: calls it
            // made wait for it, and a script waiting on a long call is where a restart usually
            // finds it. Otherwise with the next commit; the VM catches up without waiting.
            if index == self.journal_length {
                self.pump_due = true;
            }
            return;
        }
        // Caught up: the VM processed everything delivered and waits.
        if index != self.next_replayed {
            return;
        }
        if let Some(next) = self.replayed.get(index) {
            if next.id > issued {
                let message = format!("it waits without making call {}, which it made before", next.id);
                self.diverge(message).await;
            }
            return;
        }
        // The last replayed result is acknowledged: continue live, admitting the calls the
        // journal did not have once their count is committed.
        self.replaying = false;
        self.pump().await;
        self.release().await;
    }

    /// Lets the held calls launch, and later live calls, each once its issued count is
    /// committed. On a rerun, an earlier attempt may have admitted some of them under their
    /// keys, as other calls: when the replay releases, its counts matched every recorded one, so
    /// it has made every call an earlier attempt could have admitted. All held calls are
    /// checked first, whatever their target, and a divergence launches none. Calls made after
    /// this are new.
    async fn release(&mut self) {
        if self.stopped() || self.released {
            return;
        }
        let calls = std::mem::take(&mut self.held);
        if self.rerun {
            for call in &calls {
                match Self::earlier(self.api, self.ctx, call).await {
                    Err(error) => {
                        self.fail(Box::new(error));
                        return;
                    }
                    Ok(Some(other)) => {
                        let message = format!("call {} is {}, but was {other}", call.id, describe(call));
                        self.diverge(message).await;
                        return;
                    }
                    Ok(None) => {}
                }
            }
        }
        self.released = true;
        self.waiting.extend(calls);
        self.launch_covered();
    }

    /// Launches the waiting calls whose issued count is committed.
    fn launch_covered(&mut self) {
        if !self.released || self.stopped() {
            return;
        }
        let durable = self.durable_issued;
        let (covered, waiting): (Vec<HostCall>, Vec<HostCall>) = std::mem::take(&mut self.waiting)
            .into_iter()
            .partition(|call| call.id <= durable);
        self.waiting = waiting;
        for call in covered {
            self.launch(call);
        }
    }

    /// The call an earlier attempt made under this call's key, when it is another one: another
    /// target, tool, or arguments.
    async fn earlier(api: &ToolApi<'_>, ctx: &Context, call: &HostCall) -> Result<Option<String>, ToolApiError> {
        let address = DocAddress::<NestedCallDoc>::member(api.task_id(), format!("c{}", call.id));
        let Some(value) = api.snapshot(address, ctx).await? else {
            return Ok(None);
        };
        let state: NestedCallState = json::from_json(&value).map_err(crate::scheduler::RuntimeError::Codec)?;
        let (Some(_), Some(name)) = (state.task_id, state.name) else {
            return Ok(None);
        };
        let arguments = state.arguments.unwrap_or_default();
        if call.target == CallTarget::Tool
            && name == call.name
            && call
                .args
                .as_ref()
                .map_or(arguments.is_empty(), |args| args.as_object() == Some(&arguments))
        {
            return Ok(None);
        }
        Ok(Some(format!("tools.{name}({})", Value::Object(arguments))))
    }

    /// Runs one live call; its result is journaled when it lands.
    fn launch(&mut self, call: HostCall) {
        if self.stopped() {
            return;
        }
        match call.target {
            CallTarget::Tool => self.launch_tool(call, true),
            CallTarget::Global if DISCOVERY_GLOBALS.contains(&call.name.as_str()) => {
                let outcome = match discover(&self.start, &call) {
                    Ok(value) => Outcome::ok(value),
                    Err(message) => Outcome::err(message),
                };
                self.running.push(Box::pin(std::future::ready(Landed {
                    call,
                    live: true,
                    result: CallResult::Global(Ok(outcome)),
                })));
            }
            CallTarget::Global if CATALOG_GLOBALS.contains(&call.name.as_str()) => {
                let (api, ctx) = (self.api, self.ctx);
                self.running.push(Box::pin(async move {
                    let result = match api.models() {
                        Ok(models) => match catalog_call(&call.name, call.args.as_ref(), &*models, ctx).await {
                            Ok(value) => Ok(Outcome::ok(value)),
                            Err(_) if ctx.is_cancelled() => Err("The call was cancelled".into()),
                            Err(message) => Ok(Outcome::err(message)),
                        },
                        Err(error) => Err(Box::new(error) as ToolError),
                    };
                    Landed {
                        call,
                        live: true,
                        result: CallResult::Global(result),
                    }
                }));
            }
            CallTarget::Global => {
                let (api, ctx, options) = (self.api, self.ctx, self.options);
                self.running.push(Box::pin(async move {
                    let result = call_global(options, &call, api, ctx).await;
                    Landed {
                        call,
                        live: true,
                        result: CallResult::Global(result),
                    }
                }));
            }
        }
    }

    fn launch_tool(&mut self, call: HostCall, live: bool) {
        let (api, ctx) = (self.api, self.ctx);
        let slots = call
            .name
            .starts_with(CODEMODE_INTERNAL_PREFIX)
            .then(|| self.model_slots.clone());
        self.running.push(Box::pin(async move {
            // A model call waits for a slot; one still waiting when the script ends is dropped
            // with the loop, so it never starts.
            let _permit = match slots {
                Some(slots) => slots.acquire_owned().await.ok(),
                None => None,
            };
            // Passed as the script gave them; a script that passes no object gets an error
            // without a nested call, on every run alike.
            let arguments: JsonObject = match &call.args {
                None => JsonObject::new(),
                Some(Value::Object(arguments)) => arguments.clone(),
                Some(_) => {
                    let message = format!("Tool {} expects an object of arguments", call.name);
                    return Landed {
                        call,
                        live,
                        result: CallResult::Refused(Outcome::err(message)),
                    };
                }
            };
            let options = NestedCallOptions {
                key: Some(format!("c{}", call.id)),
                progress: None,
            };
            let result = api.execute_tool(&call.name, arguments, options, ctx).await;
            Landed {
                call,
                live,
                result: CallResult::Tool(result),
            }
        }));
    }

    fn on_landed(&mut self, landed: Landed) {
        if self.stopped() {
            return;
        }
        let Landed { call, live, result } = landed;
        let (outcome, global) = match result {
            CallResult::Tool(Ok(result)) => {
                let structured = self
                    .start
                    .tools
                    .iter()
                    .any(|tool| tool.name == call.name && tool.structured);
                let outcome = outcome_of(&call.name, structured, &result);
                self.finish_row(call.id, &outcome, &result);
                if call.name == GENERATE_IMAGES_TOOL {
                    self.generated_images += count_images(result.structured_output.as_ref());
                }
                (outcome, None)
            }
            CallResult::Refused(outcome) => {
                if let Some(row) = self.rows.get_mut(&call.id) {
                    row.0.status = CallStatus::Error;
                    if let Outcome::Err { message, .. } = &outcome {
                        row.0.error = Some(cut(message, ERROR_PREVIEW_CHARS));
                    }
                }
                (outcome, None)
            }
            CallResult::Global(Ok(outcome)) => {
                let global = JournaledGlobal {
                    name: call.name.clone(),
                    args: call.args.clone(),
                    outcome: outcome.clone(),
                };
                (outcome, Some(global))
            }
            CallResult::Tool(Err(error)) => {
                self.landed_error(&call, Box::new(error));
                return;
            }
            CallResult::Global(Err(error)) => {
                self.landed_error(&call, error);
                return;
            }
        };
        if live {
            self.queue.push(Finished {
                id: call.id,
                at: self.api.now(),
                outcome,
                global,
            });
            self.pump_due = true;
        } else {
            self.ready.insert(call.id, outcome);
            self.deliver_replayed();
        }
    }

    /// A call that did not end with a result: the call failed when its context ended; else the
    /// key names another tool or other arguments (or a divergence found earlier), so this
    /// attempt took another path.
    fn landed_error(&mut self, call: &HostCall, error: ToolError) {
        if self.ctx.is_cancelled() {
            self.fail(error);
            return;
        }
        let message = format!("call {} is {}: {error}", call.id, describe(call));
        self.diverge_later(message);
    }

    /// The script made a tool call: its row appears, running, also while it waits to launch.
    fn started(&mut self, call: &HostCall) {
        self.rows.entry(call.id).or_insert_with(|| {
            let args = call.args.as_ref().map(Value::to_string).unwrap_or_default();
            Row(CodemodeNestedCall {
                id: call.id,
                name: call
                    .name
                    .strip_prefix(CODEMODE_INTERNAL_PREFIX)
                    .unwrap_or(&call.name)
                    .to_owned(),
                args: cut(&args, ARGS_PREVIEW_CHARS),
                status: CallStatus::Running,
                duration_ms: None,
                error: None,
            })
        });
    }

    /// The call ended: an error result is an error row, also when the script got its value
    /// (`bash`'s nonzero exit).
    fn finish_row(&mut self, id: u64, outcome: &Outcome, result: &NestedToolExecutionResult) {
        let Some(Row(row)) = self.rows.get_mut(&id) else {
            return;
        };
        row.status = if result.is_error {
            CallStatus::Error
        } else {
            CallStatus::Ok
        };
        if result.duration_ms.is_some() {
            row.duration_ms = result.duration_ms;
        }
        let error = match outcome {
            Outcome::Err { message, .. } => message.clone(),
            Outcome::Ok { .. } => error_diagnostics(result),
        };
        if result.is_error && !error.is_empty() {
            row.error = Some(cut(&error, ERROR_PREVIEW_CHARS));
        }
    }

    /// The rows in call order; calls still running when the script ended were cancelled.
    fn details(&self, full_output_path: Option<String>) -> CodemodeToolDetails {
        let calls = self
            .rows
            .values()
            .map(|Row(row)| {
                let mut row = row.clone();
                if row.status == CallStatus::Running {
                    row.status = CallStatus::Cancelled;
                }
                row
            })
            .collect();
        CodemodeToolDetails {
            calls,
            full_output_path,
        }
    }

    /// Journals finished calls and acknowledged counts in one commit, then delivers the calls
    /// and launches the calls the counts cover.
    async fn pump(&mut self) {
        if self.stopped() || (self.queue.is_empty() && self.acks.is_empty()) {
            return;
        }
        let batch = std::mem::take(&mut self.queue);
        let acks = std::mem::take(&mut self.acks);
        let entries: Result<Vec<Json>, serde_json::Error> = batch
            .iter()
            .map(|item| {
                json::to_json(&JournalEntry {
                    id: item.id,
                    at: item.at,
                    issued_after: None,
                    global: item.global.clone(),
                })
            })
            .collect();
        let entries = match entries {
            Ok(entries) => entries,
            Err(error) => {
                self.fail(Box::new(error));
                return;
            }
        };
        let task = self.api.task_id();
        let counts = acks.clone();
        let committed = self
            .api
            .commit(self.ctx, move |tx| {
                Box::pin(async move {
                    let mut journal = tx.doc(DocAddress::<CodemodeJournalDoc>::new(task)).await?;
                    let mut draft = journal.draft();
                    let Some(mut root) = draft.object() else {
                        return Err(TxError::callback(std::io::Error::other(
                            "delta.codemode.journal is not an object",
                        )));
                    };
                    for (seq, issued) in counts {
                        let count = Json::from(Value::from(issued));
                        if seq == 0 {
                            if !root.contains_key("first_issued") {
                                root.insert("first_issued", count);
                            }
                            continue;
                        }
                        let index = usize::try_from(seq - 1).unwrap_or(usize::MAX);
                        let mut list = root.field_or_insert("entries", Json::from(Vec::<Json>::new()));
                        if let Some(mut entry) = list.index(index).and_then(Draft::into_object)
                            && !entry.contains_key("issued_after")
                        {
                            entry.insert("issued_after", count);
                        }
                    }
                    if !entries.is_empty()
                        && let Some(mut list) = root
                            .field_or_insert("entries", Json::from(Vec::<Json>::new()))
                            .into_array()
                    {
                        list.extend(entries);
                    }
                    Ok(())
                })
            })
            .await;
        if let Err(error) = committed {
            self.fail(Box::new(error));
            return;
        }
        if let Some(max) = acks.iter().map(|(_, issued)| *issued).max() {
            self.durable_issued = self.durable_issued.max(max);
        }
        for item in batch {
            self.journal_length += 1;
            self.deliver(self.journal_length, item.at, item.id, item.outcome);
        }
        self.launch_covered();
    }
}

/// `searchTools()`, `describeTool()`, `describeNamespace()`: answered from the start data, so a
/// replay sees what the first run saw.
fn discover(start: &CodemodeStart, call: &HostCall) -> Result<Option<Value>, String> {
    let args: &[Value] = call.args.as_ref().and_then(Value::as_array).map_or(&[], Vec::as_slice);
    let tools: Vec<&StartTool> = start.tools.iter().filter(|tool| !tool.hidden).collect();
    let namespace_of_tool = |name: &str| {
        start
            .namespaces
            .iter()
            .find(|namespace| namespace.tools.iter().any(|tool| tool == name))
    };
    let matches_namespace = |namespace: &StartNamespace, query: &str| {
        namespace.name == query || to_codemode_identifier(&namespace.name) == to_codemode_identifier(query)
    };
    match call.name.as_str() {
        "describeTool" => {
            let Some(name) = args.first().and_then(Value::as_str) else {
                return Err("describeTool() expects a tool name".to_owned());
            };
            Ok(tools
                .iter()
                .find(|tool| tool.name == name || tool.js_name == name)
                .map(|tool| Value::String(tool.description.clone())))
        }
        "describeNamespace" => {
            let Some(name) = args.first().and_then(Value::as_str) else {
                return Err("describeNamespace() expects a namespace name".to_owned());
            };
            let Some(namespace) = start
                .namespaces
                .iter()
                .find(|namespace| matches_namespace(namespace, name))
            else {
                return Ok(None);
            };
            let mut described = Map::new();
            described.insert("name".to_owned(), Value::String(namespace.name.clone()));
            if let Some(description) = &namespace.description {
                described.insert("description".to_owned(), Value::String(description.clone()));
            }
            if let Some(instructions) = &namespace.instructions {
                described.insert("instructions".to_owned(), Value::String(instructions.clone()));
            }
            let members = namespace
                .tools
                .iter()
                .map(|tool| Value::String(to_codemode_identifier(tool)))
                .collect();
            described.insert("tools".to_owned(), Value::Array(members));
            Ok(Some(Value::Object(described)))
        }
        _ => {
            let Some(query) = args.first().and_then(Value::as_str) else {
                return Err("searchTools() expects a query string".to_owned());
            };
            let options = args.get(1).filter(|options| !options.is_null());
            let limit = match options.and_then(|options| options.get("limit")) {
                None | Some(Value::Null) => DEFAULT_SEARCH_LIMIT,
                Some(limit) => limit
                    .as_u64()
                    .filter(|limit| *limit > 0)
                    .and_then(|limit| usize::try_from(limit).ok())
                    .ok_or("searchTools() limit must be a positive integer")?,
            };
            let namespace = match options.and_then(|options| options.get("namespace")) {
                None | Some(Value::Null) => None,
                Some(Value::String(namespace)) => Some(namespace.as_str()),
                Some(_) => return Err("searchTools() namespace must be a string".to_owned()),
            };
            let documents: Vec<SearchDocument> = tools
                .iter()
                .filter_map(|tool| {
                    let own = namespace_of_tool(&tool.name);
                    if let Some(namespace) = namespace
                        && !own.is_some_and(|own| matches_namespace(own, namespace))
                    {
                        return None;
                    }
                    let text = [
                        tool.name.as_str(),
                        &tool.name.replace('_', " "),
                        &tool.description,
                        own.map_or("", |own| own.name.as_str()),
                        own.and_then(|own| own.description.as_deref()).unwrap_or(""),
                    ]
                    .join(" ");
                    Some(SearchDocument {
                        name: tool.name.clone(),
                        text,
                    })
                })
                .collect();
            let found = rank_documents(query, &documents, limit)
                .into_iter()
                .filter_map(|name| tools.iter().find(|tool| tool.name == name))
                .map(|tool| serde_json::json!({ "name": tool.js_name, "description": tool.description }))
                .collect();
            Ok(Some(Value::Array(found)))
        }
    }
}

/// A script's call of a host global.
async fn call_global(
    options: &CodemodeOptions,
    call: &HostCall,
    api: &ToolApi<'_>,
    ctx: &Context,
) -> Result<Outcome, ToolError> {
    let Some(global) = options.globals.iter().find(|global| global.name() == call.name) else {
        return Ok(Outcome::err(format!("Unknown global {}", call.name)));
    };
    match global.execute(call.args.clone(), api, ctx).await {
        // `None` stays undefined: the journal omits the value, and a replay resolves with
        // undefined again.
        Ok(value) => Ok(Outcome::ok(value)),
        Err(error) if ctx.is_cancelled() => Err(error),
        Err(error) => Ok(Outcome::err(error.to_string())),
    }
}

/// Writes the result's files under fixed names into the run's output directory, so a rerun
/// overwrites them.
struct RunFiles<'a> {
    api: &'a ToolApi<'a>,
    ctx: &'a Context,
    dir: Option<String>,
}

impl RunFiles<'_> {
    /// The run's output directory: the recorded one, else a new temp directory, recorded before
    /// any file is written.
    async fn dir(&mut self) -> Result<String, String> {
        if let Some(dir) = &self.dir {
            return Ok(dir.clone());
        }
        let Some(env) = self.api.env() else {
            return Err("no execution environment".to_owned());
        };
        let task = self.api.task_id();
        let snapshot = self
            .api
            .snapshot(DocAddress::<CodemodeRunDoc>::new(task), self.ctx)
            .await
            .map_err(|error| error.to_string())?;
        let recorded = snapshot
            .map(|value| json::from_json::<RunState>(&value))
            .transpose()
            .map_err(|error| error.to_string())?
            .and_then(|state| state.output_dir);
        let dir = match recorded {
            Some(dir) => dir,
            None => {
                let created = env
                    .create_temp_dir(Some("delta-codemode-"), self.ctx)
                    .await
                    .map_err(|error| error.to_string())?;
                self.api
                    .commit(self.ctx, move |tx| {
                        Box::pin(async move {
                            let mut run = tx.doc(DocAddress::<CodemodeRunDoc>::new(task)).await?;
                            run.update_as::<RunState, _>(|state| state.output_dir.get_or_insert(created).clone())
                                .map_err(TxError::callback)
                        })
                    })
                    .await
                    .map_err(|error| error.to_string())?
            }
        };
        self.dir = Some(dir.clone());
        Ok(dir)
    }
}

impl OutputFile for RunFiles<'_> {
    fn write<'a>(&'a mut self, name: &'a str, data: &'a [u8]) -> BoxFuture<'a, Result<String, String>> {
        Box::pin(async move {
            let dir = self.dir().await?;
            let Some(env) = self.api.env() else {
                return Err("no execution environment".to_owned());
            };
            // The environment's separator: `\` on Windows, so the path reads as one.
            let path = env.join_path(&[&dir, name]).map_err(|error| error.to_string())?;
            env.write_file(&path, data, self.ctx)
                .await
                .map_err(|error| error.to_string())?;
            Ok(path)
        })
    }
}
