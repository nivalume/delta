//! Host side of a run. Port of upstream `runtime/execution.ts`: the host owns everything the
//! script must not influence (the tool table, call records, output, the deadline, aborts and the
//! decoding of what the VM reports); the VM runs on its own thread (`vm.rs`).
//!
//! Lifecycle of one run: the VM thread starts, its messages arrive on a channel the run's driver
//! reads, tool calls run here (`execute`) or go to the caller (`start`), and results go back to
//! the VM. The first terminal event wins: `Done`, a crash, the deadline or an abort. Finishing
//! cancels pending calls, stops the VM and settles the result; later events are ignored.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::future::Future;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use futures::future::BoxFuture;
use serde_json::{Map, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::identifier::{is_identifier, to_codemode_identifier};
use crate::types::{
    CallRecord, CallStatus, CallTarget, CodemodeError, CodemodeResult, DeclaredGlobal, DeclaredTool, During, ErrorKind,
    HostCall, Outcome, OutputItem, StoreWrites,
};
use crate::vm::{self, FromVm, ToVm, VmHandle, VmStart};

/// `execute`'s deadline unless the sandbox sets another.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_mins(5);

const RESERVED_GLOBALS: [&str; 9] = [
    "tools",
    "ALL_TOOLS",
    "console",
    "text",
    "image",
    "exit",
    "globalThis",
    "store",
    "load",
];

/// A function scripts call: a tool (`tools.<name>(args)`) or a global (`<name>(args)`).
pub trait CodemodeTool: Send + Sync + fmt::Debug {
    /// Tools: any name; scripts use `to_codemode_identifier(name)` and the name itself. Globals:
    /// an identifier, or `namespace.member`.
    fn name(&self) -> &str;

    /// Listed in `ALL_TOOLS` and shown in declarations.
    #[allow(clippy::unnecessary_literal_bound)] // Implementations return their own text.
    fn description(&self) -> &str {
        ""
    }

    /// Schema of the argument, for declarations.
    fn input_schema(&self) -> Option<&Value> {
        None
    }

    /// Schema of the resolved value, for declarations.
    fn output_schema(&self) -> Option<&Value> {
        None
    }

    /// Globals only: receive the whole argument list as an array.
    fn spread(&self) -> bool {
        false
    }

    /// Globals only: the TypeScript parameter list and return type for declarations, replacing
    /// the rendering from the schemas.
    fn signature(&self) -> Option<&str> {
        None
    }

    /// `args` is what the script passed after a JSON round trip (`None`: undefined). The token
    /// is cancelled when the script ends without waiting for the call, or the run ends. An
    /// `Err` rejects in the script with an `Error` carrying the message.
    fn execute(&self, args: Option<Value>, cancel: CancellationToken) -> BoxFuture<'_, Result<Option<Value>, String>>;
}

type ToolFn =
    dyn Fn(Option<Value>, CancellationToken) -> BoxFuture<'static, Result<Option<Value>, String>> + Send + Sync;

/// A [`CodemodeTool`] from a closure.
pub struct FnCodemodeTool {
    name: String,
    description: String,
    input_schema: Option<Value>,
    output_schema: Option<Value>,
    spread: bool,
    run: Box<ToolFn>,
}

impl FnCodemodeTool {
    pub fn new<F, R>(name: impl Into<String>, run: F) -> Self
    where
        F: Fn(Option<Value>, CancellationToken) -> R + Send + Sync + 'static,
        R: Future<Output = Result<Option<Value>, String>> + Send + 'static,
    {
        Self {
            name: name.into(),
            description: String::new(),
            input_schema: None,
            output_schema: None,
            spread: false,
            run: Box::new(move |args, cancel| Box::pin(run(args, cancel))),
        }
    }

    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    #[must_use]
    pub fn with_schemas(mut self, input: Option<Value>, output: Option<Value>) -> Self {
        self.input_schema = input;
        self.output_schema = output;
        self
    }

    #[must_use]
    pub fn spread(mut self) -> Self {
        self.spread = true;
        self
    }
}

impl fmt::Debug for FnCodemodeTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FnCodemodeTool")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl CodemodeTool for FnCodemodeTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> Option<&Value> {
        self.input_schema.as_ref()
    }

    fn output_schema(&self) -> Option<&Value> {
        self.output_schema.as_ref()
    }

    fn spread(&self) -> bool {
        self.spread
    }

    fn execute(&self, args: Option<Value>, cancel: CancellationToken) -> BoxFuture<'_, Result<Option<Value>, String>> {
        (self.run)(args, cancel)
    }
}

/// Why a sandbox or run cannot be set up.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SandboxError {
    #[error("Tool \"{0}\" is already registered")]
    DuplicateTool(String),
    #[error("Tool script names must be unique")]
    DuplicateScriptName,
    #[error("Invalid global name \"{0}\"")]
    InvalidGlobal(String),
    #[error("Global \"{0}\" is already registered")]
    DuplicateGlobal(String),
    #[error("Global \"{0}\" conflicts with the namespace \"{0}\"")]
    NamespaceConflict(String),
    #[error("started_at must be a non-negative number of milliseconds since the epoch")]
    StartedAt,
    #[error("The sandbox thread could not start: {0}")]
    Thread(String),
}

/// Why `Run::deliver` refused a result.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DeliverError {
    #[error("The clock value of result {0} is not a finite number")]
    Clock(u64),
    #[error("Result number {0} is not a positive integer used once")]
    Seq(u64),
    #[error("Call {0} is not waiting for a result")]
    NotWaiting(u64),
}

/// Checks global names: identifiers or `namespace.member`, not shadowing a built-in global or
/// each other.
fn check_globals<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<(), SandboxError> {
    let mut seen = HashSet::new();
    let mut namespaces = HashSet::new();
    for name in names {
        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() > 2 || !parts.iter().all(|part| is_identifier(part)) || RESERVED_GLOBALS.contains(&parts[0]) {
            return Err(SandboxError::InvalidGlobal(name.to_owned()));
        }
        if !seen.insert(name.to_owned()) {
            return Err(SandboxError::DuplicateGlobal(name.to_owned()));
        }
        if parts.len() == 2 {
            namespaces.insert(parts[0].to_owned());
        }
    }
    if let Some(name) = namespaces.iter().find(|namespace| seen.contains(*namespace)) {
        return Err(SandboxError::NamespaceConflict(name.clone()));
    }
    Ok(())
}

/// Options every run of a sandbox shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SandboxOptions {
    /// `execute`'s deadline, including time spent in tools; `None` disables it. Default
    /// [`DEFAULT_TIMEOUT`]. A deterministic run has only the deadline it is given.
    pub timeout: Option<Duration>,
    /// Bytes the VM may allocate; beyond it allocations fail inside the script as
    /// `InternalError: out of memory`.
    pub memory_limit: Option<usize>,
    /// Interrupt polls a run may use before it fails with `timeout`, limiting a script that
    /// computes without awaiting anything; part of the fingerprint. Default none: the deadline
    /// and aborts interrupt a computing script anyway.
    pub interrupt_budget: Option<u64>,
}

impl Default for SandboxOptions {
    fn default() -> Self {
        Self {
            timeout: Some(DEFAULT_TIMEOUT),
            memory_limit: None,
            interrupt_budget: None,
        }
    }
}

/// Options of one `execute`.
#[derive(Clone, Debug, Default)]
pub struct ExecuteOptions {
    /// Overrides the sandbox's deadline.
    pub timeout: Option<Duration>,
    /// Values the script reads with `load(key)`; its `store()` calls come back as
    /// `store_writes`.
    pub store: Map<String, Value>,
    /// Ends the run with `aborted`.
    pub cancel: Option<CancellationToken>,
}

/// Options of a deterministic run (`start`).
#[derive(Clone, Debug, Default)]
pub struct RunOptions {
    /// The VM clock's start, in ms since the epoch: `Date.now()` until the first result.
    pub started_at: f64,
    /// Seeds `Math.random()`.
    pub seed: u32,
    pub tools: Vec<DeclaredTool>,
    pub globals: Vec<DeclaredGlobal>,
    pub store: Map<String, Value>,
    /// The run's deadline; none by default.
    pub timeout: Option<Duration>,
    /// Ends the run with `aborted`: an error the script cannot catch.
    pub cancel: Option<CancellationToken>,
}

/// Something a deterministic run reports.
#[derive(Clone, Debug, PartialEq)]
pub enum RunEvent {
    /// The VM's fingerprint, before the script runs and before any of its calls.
    Started(String),
    /// The script made a call; answer it with [`RunHandle::deliver`], now or later.
    Call(HostCall),
    /// The VM processed result `seq` (0: the script's first run) and waits again, having issued
    /// `issued` calls in total.
    Idle { seq: u64, issued: u64 },
    /// The run's outcome; the last event.
    Finished(Box<CodemodeResult>),
}

/// Runs model-written JavaScript: `tools.<name>(args)` for every registered tool, `ALL_TOOLS`,
/// `text`, `image`, `exit`, `console.*`, `store` / `load`, and the registered globals; nothing
/// else. Each run gets its own VM on its own thread.
#[derive(Debug)]
pub struct Sandbox {
    tools: Vec<Arc<dyn CodemodeTool>>,
    globals: Vec<Arc<dyn CodemodeTool>>,
    options: SandboxOptions,
}

impl Sandbox {
    pub fn new(options: SandboxOptions) -> Self {
        Self {
            tools: Vec::new(),
            globals: Vec::new(),
            options,
        }
    }

    /// Adds a tool; a second tool with the same name is refused.
    pub fn with_tool(mut self, tool: Arc<dyn CodemodeTool>) -> Result<Self, SandboxError> {
        if self.tools.iter().any(|existing| existing.name() == tool.name()) {
            return Err(SandboxError::DuplicateTool(tool.name().to_owned()));
        }
        self.tools.push(tool);
        Ok(self)
    }

    /// Adds a global, an identifier or `namespace.member` that does not shadow a built-in.
    pub fn with_global(mut self, global: Arc<dyn CodemodeTool>) -> Result<Self, SandboxError> {
        check_globals(self.globals.iter().map(|global| global.name()).chain([global.name()]))?;
        self.globals.push(global);
        Ok(self)
    }

    pub fn tools(&self) -> &[Arc<dyn CodemodeTool>] {
        &self.tools
    }

    pub fn globals(&self) -> &[Arc<dyn CodemodeTool>] {
        &self.globals
    }

    pub fn options(&self) -> SandboxOptions {
        self.options
    }

    /// The fingerprint deterministic runs of this sandbox report.
    pub fn fingerprint(&self) -> String {
        vm::fingerprint(self.options.memory_limit, self.options.interrupt_budget)
    }

    /// Runs `code`, an async function body (`return` and top-level `await` work), calling this
    /// sandbox's tools and globals. Never fails for script failures: those are in the result.
    pub async fn execute(&self, code: &str, options: ExecuteOptions) -> CodemodeResult {
        let tools: Vec<DeclaredTool> = self
            .tools
            .iter()
            .map(|tool| DeclaredTool {
                name: tool.name().to_owned(),
                js_name: None,
                description: tool.description().to_owned(),
            })
            .collect();
        let globals: Vec<DeclaredGlobal> = self
            .globals
            .iter()
            .map(|global| DeclaredGlobal {
                name: global.name().to_owned(),
                spread: global.spread(),
            })
            .collect();
        let start = vm_start(code, &tools, &globals, &options.store, &self.options, None);
        let executors = Executors {
            tools: self
                .tools
                .iter()
                .map(|tool| (tool.name().to_owned(), tool.clone()))
                .collect(),
            globals: self
                .globals
                .iter()
                .map(|global| (global.name().to_owned(), global.clone()))
                .collect(),
        };
        let cancel = options.cancel.unwrap_or_default();
        let timeout = options.timeout.or(self.options.timeout);
        let (shared, from_vm) = match launch(start, &cancel) {
            Ok(launched) => launched,
            Err(error) => {
                return CodemodeResult::failed(CodemodeError::new(ErrorKind::Sandbox, error.to_string()));
            }
        };
        drive(shared, from_vm, Mode::Execute(executors), timeout, cancel).await
    }

    /// Runs `code` deterministically, with the caller answering its calls: the same script,
    /// `started_at`, seed, tools, globals and store, answered with the same results in the same
    /// `seq` order, makes the same calls with the same arguments and produces the same output on
    /// any VM with the same fingerprint, so a host can replay it after a restart. The sandbox's
    /// own tools and globals are not used; every call is a [`RunEvent::Call`]. The run's driver
    /// is a task on the current Tokio runtime.
    pub fn start(&self, code: &str, options: RunOptions) -> Result<Run, SandboxError> {
        if !options.started_at.is_finite() || options.started_at < 0.0 {
            return Err(SandboxError::StartedAt);
        }
        let mut names = HashSet::new();
        let mut script_names = HashSet::new();
        for tool in &options.tools {
            if !names.insert(tool.name.as_str()) {
                return Err(SandboxError::DuplicateTool(tool.name.clone()));
            }
            let js_name = tool
                .js_name
                .clone()
                .unwrap_or_else(|| to_codemode_identifier(&tool.name));
            if !script_names.insert(js_name) {
                return Err(SandboxError::DuplicateScriptName);
            }
        }
        check_globals(options.globals.iter().map(|global| global.name.as_str()))?;
        let start = vm_start(
            code,
            &options.tools,
            &options.globals,
            &options.store,
            &self.options,
            Some((options.started_at, options.seed)),
        );
        let cancel = options.cancel.unwrap_or_default();
        let (shared, from_vm) = launch(start, &cancel).map_err(|error| SandboxError::Thread(error.to_string()))?;
        let (events, receiver) = mpsc::unbounded_channel();
        let handle = RunHandle {
            shared: shared.clone(),
            cancel: cancel.clone(),
        };
        tokio::spawn(drive(shared, from_vm, Mode::Run(events), options.timeout, cancel));
        Ok(Run {
            handle,
            events: receiver,
        })
    }
}

fn vm_start(
    code: &str,
    tools: &[DeclaredTool],
    globals: &[DeclaredGlobal],
    store: &Map<String, Value>,
    options: &SandboxOptions,
    deterministic: Option<(f64, u32)>,
) -> VmStart {
    let tools: Vec<Value> = tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "name": tool.name,
                "jsName": tool.js_name.clone().unwrap_or_else(|| to_codemode_identifier(&tool.name)),
                "description": tool.description,
            })
        })
        .collect();
    let globals: Vec<Value> = globals
        .iter()
        .map(|global| serde_json::json!({ "name": global.name, "spread": global.spread }))
        .collect();
    let store: Map<String, Value> = store
        .iter()
        .map(|(key, value)| (key.clone(), Value::String(value.to_string())))
        .collect();
    VmStart {
        code: code.to_owned(),
        tools_json: Value::Array(tools).to_string(),
        globals_json: Value::Array(globals).to_string(),
        store_json: Value::Object(store).to_string(),
        memory_limit: options.memory_limit,
        interrupt_budget: options.interrupt_budget,
        deterministic,
    }
}

/// A deterministic run: its events, and a handle to answer calls.
#[derive(Debug)]
pub struct Run {
    handle: RunHandle,
    events: mpsc::UnboundedReceiver<RunEvent>,
}

impl Run {
    /// A handle for answering calls from elsewhere.
    pub fn handle(&self) -> RunHandle {
        self.handle.clone()
    }

    /// The next event; `None` after [`RunEvent::Finished`].
    pub async fn next_event(&mut self) -> Option<RunEvent> {
        self.events.recv().await
    }

    pub fn deliver(&self, seq: u64, now: f64, id: u64, outcome: Outcome) -> Result<(), DeliverError> {
        self.handle.deliver(seq, now, id, outcome)
    }
}

/// Answers a deterministic run's calls.
#[derive(Clone)]
pub struct RunHandle {
    shared: Arc<Shared>,
    cancel: CancellationToken,
}

impl fmt::Debug for RunHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunHandle")
            .field("finished", &self.finished())
            .finish_non_exhaustive()
    }
}

impl RunHandle {
    /// Answers call `id` as result number `seq`. Results are numbered 1, 2, … without gaps, and
    /// the VM applies them strictly in that order, whatever order they arrive in. While it
    /// processes this one its clock reads `now` (ms since the epoch), or the previous value if
    /// `now` is earlier. A result for a finished run is dropped.
    pub fn deliver(&self, seq: u64, now: f64, id: u64, outcome: Outcome) -> Result<(), DeliverError> {
        let mut state = self.shared.lock();
        if state.finished {
            return Ok(());
        }
        if !now.is_finite() {
            return Err(DeliverError::Clock(seq));
        }
        if seq < 1 || state.delivered.contains(&seq) {
            return Err(DeliverError::Seq(seq));
        }
        if !state.pending.contains_key(&id) {
            return Err(DeliverError::NotWaiting(id));
        }
        state.delivered.insert(seq);
        let pending = state.pending.remove(&id).unwrap_or_else(|| unreachable!());
        state.settle_record(&pending, outcome.is_ok());
        let (ok, payload, properties) = encode(outcome);
        if let Some(vm) = &state.vm {
            vm.post(ToVm::Result {
                id,
                ok,
                payload,
                seq: Some(seq),
                now: Some(now),
                properties,
            });
        }
        Ok(())
    }

    /// Ends the run with `aborted`, an error the script cannot catch.
    pub fn abort(&self) {
        self.cancel.cancel();
    }

    /// Whether the run has its outcome; calls can no longer be answered.
    pub fn finished(&self) -> bool {
        self.shared.lock().finished
    }
}

fn encode(outcome: Outcome) -> (bool, Option<String>, Option<String>) {
    match outcome {
        Outcome::Ok { value } => (true, value.map(|value| value.to_string()), None),
        Outcome::Err { message, properties } => (
            false,
            Some(message),
            properties.map(|properties| Value::Object(properties).to_string()),
        ),
    }
}

/// A call waiting for its answer.
struct Pending {
    record: Option<usize>,
    started: Instant,
    cancel: CancellationToken,
}

/// What the driver and `RunHandle::deliver` share.
#[derive(Default)]
struct State {
    vm: Option<VmHandle>,
    finished: bool,
    pending: HashMap<u64, Pending>,
    delivered: HashSet<u64>,
    output: Vec<OutputItem>,
    calls: Vec<CallRecord>,
}

impl State {
    fn settle_record(&mut self, pending: &Pending, ok: bool) {
        if let Some(record) = pending.record.and_then(|index| self.calls.get_mut(index)) {
            record.status = if ok { CallStatus::Ok } else { CallStatus::Error };
            record.duration_ms = pending.started.elapsed().as_secs_f64() * 1000.0;
        }
    }
}

struct Shared(Mutex<State>);

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn launch(
    start: VmStart,
    cancel: &CancellationToken,
) -> std::io::Result<(Arc<Shared>, mpsc::UnboundedReceiver<FromVm>)> {
    let (events, from_vm) = mpsc::unbounded_channel();
    let vm = if cancel.is_cancelled() {
        None
    } else {
        Some(vm::spawn(start, events)?)
    };
    let shared = Arc::new(Shared(Mutex::new(State { vm, ..State::default() })));
    Ok((shared, from_vm))
}

/// The tools and globals `execute` runs itself.
struct Executors {
    tools: HashMap<String, Arc<dyn CodemodeTool>>,
    globals: HashMap<String, Arc<dyn CodemodeTool>>,
}

enum Mode {
    Execute(Executors),
    Run(mpsc::UnboundedSender<RunEvent>),
}

type Answer = (u64, Result<Option<Value>, String>);

/// A payload the prelude sent is not what it should be: a script that patched built-ins can
/// make it send malformed data.
struct BridgeError(String);

fn bridge_json(json: &str, what: &str) -> Result<Value, BridgeError> {
    serde_json::from_str(json).map_err(|_| BridgeError(format!("{what} is not valid JSON")))
}

fn store_writes(json: &str) -> Result<StoreWrites, BridgeError> {
    let Value::Array(entries) = bridge_json(json, "store writes")? else {
        return Err(BridgeError("store writes are not an array".to_owned()));
    };
    let mut writes = StoreWrites::default();
    for entry in entries {
        match entry.as_array().map(Vec::as_slice) {
            Some([Value::String(key)]) => writes.delete.push(key.clone()),
            Some([Value::String(key), Value::String(json)]) => {
                let value = bridge_json(json, &format!("store value for {}", Value::String(key.clone())))?;
                writes.set.insert(key.clone(), value);
            }
            _ => return Err(BridgeError("store writes contain a malformed entry".to_owned())),
        }
    }
    Ok(writes)
}

fn script_error(json: &str) -> Result<CodemodeError, BridgeError> {
    let value = bridge_json(json, "script error")?;
    let Value::Object(fields) = value else {
        return Err(BridgeError("script error is not an object".to_owned()));
    };
    let text = |key: &str| -> Result<Option<String>, BridgeError> {
        match fields.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(text)) => Ok(Some(text.clone())),
            Some(_) => Err(BridgeError("script error is malformed".to_owned())),
        }
    };
    let message = text("message")?.ok_or_else(|| BridgeError("script error is malformed".to_owned()))?;
    Ok(CodemodeError {
        kind: ErrorKind::Script,
        name: text("name")?,
        message,
        stack: text("stack")?,
    })
}

/// The run's driver: reads the VM's messages, runs or forwards calls, and finishes the run.
async fn drive(
    shared: Arc<Shared>,
    mut from_vm: mpsc::UnboundedReceiver<FromVm>,
    mode: Mode,
    timeout: Option<Duration>,
    cancel: CancellationToken,
) -> CodemodeResult {
    let deadline = timeout.map(|timeout| tokio::time::Instant::now() + timeout);
    let sleep = async move {
        match deadline {
            Some(deadline) => tokio::time::sleep_until(deadline).await,
            None => std::future::pending().await,
        }
    };
    tokio::pin!(sleep);
    // Tool calls run as tasks of their own, so a call the run cancels still sees its token.
    let (answered, mut answers) = mpsc::unbounded_channel::<Answer>();
    let outcome: Done = loop {
        tokio::select! {
            biased;
            () = cancel.cancelled() => {
                break failed(ErrorKind::Aborted, "Execution aborted");
            }
            () = &mut sleep => {
                let timeout = timeout.unwrap_or_default();
                break failed(ErrorKind::Timeout, format!("Execution timed out after {} ms", timeout.as_millis()));
            }
            message = from_vm.recv() => {
                let Some(message) = message else {
                    break failed(ErrorKind::Sandbox, "The sandbox VM exited");
                };
                match receive(&shared, &mode, message, &answered) {
                    Ok(None) => {}
                    Ok(Some(done)) => break done,
                    Err(BridgeError(problem)) => break failed(
                        ErrorKind::Sandbox,
                        format!("Sandbox bridge broken: {problem}. The script may have modified built-ins such as a prototype's toJSON."),
                    ),
                }
            }
            Some((id, answer)) = answers.recv() => {
                let mut state = shared.lock();
                // Cancelled by the finish: the record keeps "cancelled".
                let Some(pending) = state.pending.remove(&id) else { continue };
                state.settle_record(&pending, answer.is_ok());
                let (ok, payload) = match answer {
                    Ok(value) => (true, value.map(|value| value.to_string())),
                    Err(message) => (false, Some(message)),
                };
                if let Some(vm) = &state.vm {
                    vm.post(ToVm::Result { id, ok, payload, seq: None, now: None, properties: None });
                }
            }
        }
    };
    let result = finish(&shared, outcome);
    if let Mode::Run(events) = &mode {
        let _ = events.send(RunEvent::Finished(Box::new(result.clone())));
    }
    result
}

/// How the run ended, and in a deterministic run that the script ended itself, where.
type Done = (Result<(Option<Value>, StoreWrites), CodemodeError>, Option<During>);

fn failed(kind: ErrorKind, message: impl Into<String>) -> Done {
    (Err(CodemodeError::new(kind, message)), None)
}

/// Handles one VM message; `Some` when the run is decided.
fn receive(
    shared: &Shared,
    mode: &Mode,
    message: FromVm,
    answered: &mpsc::UnboundedSender<Answer>,
) -> Result<Option<Done>, BridgeError> {
    match message {
        FromVm::Output(item) => shared.lock().output.push(item),
        FromVm::Call { id, global, name, args } => {
            let args = args.map(|args| bridge_json(&args, "call arguments")).transpose()?;
            let mut state = shared.lock();
            if state.pending.contains_key(&id) {
                return Err(BridgeError(format!("duplicate call id {id}")));
            }
            let record = (!global).then(|| {
                state.calls.push(CallRecord {
                    name: name.clone(),
                    status: CallStatus::Cancelled,
                    duration_ms: 0.0,
                });
                state.calls.len() - 1
            });
            let cancel = CancellationToken::new();
            state.pending.insert(
                id,
                Pending {
                    record,
                    started: Instant::now(),
                    cancel: cancel.clone(),
                },
            );
            drop(state);
            match mode {
                Mode::Execute(executors) => {
                    let table = if global { &executors.globals } else { &executors.tools };
                    let tool = table.get(&name).cloned();
                    let kind = if global { "global" } else { "tool" };
                    let answered = answered.clone();
                    tokio::spawn(async move {
                        let answer = match tool {
                            Some(tool) => tool.execute(args, cancel).await,
                            None => Err(format!("Unknown {kind} \"{name}\"")),
                        };
                        // A finished run no longer listens.
                        let _ = answered.send((id, answer));
                    });
                }
                Mode::Run(events) => {
                    let target = if global { CallTarget::Global } else { CallTarget::Tool };
                    let _ = events.send(RunEvent::Call(HostCall { id, target, name, args }));
                }
            }
        }
        FromVm::Started(fingerprint) => {
            if let Mode::Run(events) = mode {
                let _ = events.send(RunEvent::Started(fingerprint));
            }
        }
        FromVm::Idle { seq, issued } => {
            if let Mode::Run(events) = mode {
                let _ = events.send(RunEvent::Idle { seq, issued });
            }
        }
        FromVm::Done {
            ok,
            value,
            error,
            writes,
            during,
        } => {
            // Decoded before anything is decided.
            if !ok {
                let error = script_error(error.as_deref().unwrap_or("{}"))?;
                return Ok(Some((Err(error), during)));
            }
            let value = value.map(|value| bridge_json(&value, "return value")).transpose()?;
            let writes = store_writes(writes.as_deref().unwrap_or("[]"))?;
            return Ok(Some((Ok((value, writes)), during)));
        }
        FromVm::Crash(message) => return Ok(Some(failed(ErrorKind::Sandbox, message))),
        FromVm::Exhausted => {
            return Ok(Some(failed(
                ErrorKind::Timeout,
                "Execution exceeded its interrupt budget",
            )));
        }
    }
    Ok(None)
}

/// Decides the run: cancels pending calls, stops the VM and builds the result.
fn finish(shared: &Shared, (outcome, during): Done) -> CodemodeResult {
    let mut state = shared.lock();
    state.finished = true;
    let now = Instant::now();
    for pending in std::mem::take(&mut state.pending).into_values() {
        if let Some(record) = pending.record.and_then(|index| state.calls.get_mut(index)) {
            record.duration_ms = now.duration_since(pending.started).as_secs_f64() * 1000.0;
        }
        pending.cancel.cancel();
    }
    if let Some(vm) = state.vm.take() {
        vm.stop();
    }
    let output = std::mem::take(&mut state.output);
    let calls = std::mem::take(&mut state.calls);
    match outcome {
        Ok((value, store_writes)) => CodemodeResult {
            error: None,
            value,
            output,
            calls,
            store_writes,
            during,
        },
        Err(error) => CodemodeResult {
            error: Some(error),
            value: None,
            output,
            calls,
            store_writes: StoreWrites::default(),
            during,
        },
    }
}

impl fmt::Debug for Shared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Shared").finish_non_exhaustive()
    }
}
