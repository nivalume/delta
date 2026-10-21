//! One QuickJS VM driving one script, on a thread of its own. Port of upstream
//! `runtime/vm.ts`: the prelude builds the script environment around a single bridge function,
//! and every message for the host leaves through `events`, synchronously from inside VM
//! callbacks. The thread then waits for host messages; it never decides on its own that it is
//! finished, and ends when the host sends `Stop` or drops its sender.
//!
//! Promise jobs run one at a time with an interrupt poll every [`JOBS_PER_POLL`] jobs, so a
//! script such as `while (true) await null` (one short job per iteration, which QuickJS's own
//! opcode-counted polling rarely sees) stays interruptible.
//!
//! Deterministic mode (`Sandbox::start`): `deterministic.js` runs before the prelude and fixes
//! the clock, randomness and timezone; results carry a `seq` and are applied strictly in that
//! order, buffering early ones; the VM reports `Started` with its fingerprint before the script
//! runs and `Idle` with the number of calls issued after each drain.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;

use rquickjs::context::EvalOptions;
use rquickjs::function::Rest;
use rquickjs::{Context, Ctx, Function, Object, Persistent, Runtime, Value};
use tokio::sync::mpsc::UnboundedSender;

use crate::types::{During, OutputItem};

/// The prelude of upstream pi-codemode, with its limits substituted.
pub(crate) const PRELUDE_SOURCE: &str = include_str!("prelude.js");
/// What deterministic runs evaluate before the prelude.
pub(crate) const DETERMINISTIC_SOURCE: &str = include_str!("deterministic.js");

/// The engine, as the fingerprint names it; `engine_matches_the_lockfile` checks it.
pub(crate) const ENGINE: &str = "rquickjs 0.14.0, quickjs-ng 0.16.2";
/// Bumped whenever this module or the preludes change what a deterministic script observes,
/// beyond what the fingerprint's other parts capture.
const DETERMINISTIC_VERSION: u32 = 1;
/// QuickJS's guard on its own stack: deep recursion becomes a catchable `RangeError` well
/// before the thread's stack ([`THREAD_STACK`]) runs out. Sized for unoptimized builds too: a
/// JavaScript call takes about 1.4 KB of C stack in a Linux debug build and over 4 KB in a
/// macOS one, so 16 MiB allows a few thousand nested calls everywhere.
pub(crate) const MAX_STACK_SIZE: usize = 16 * 1024 * 1024;
/// Reserved address space, committed only as the stack grows.
const THREAD_STACK: usize = 64 * 1024 * 1024;
/// Promise jobs run between two interrupt polls while draining the job queue.
const JOBS_PER_POLL: u32 = 50;

/// FNV-1a over UTF-16 code units, as 8 hex digits (as upstream identifies its prelude).
pub(crate) fn fnv1a(text: &str) -> String {
    let mut hash: u32 = 0x811c_9dc5;
    for unit in text.encode_utf16() {
        hash ^= u32::from(unit);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    format!("{hash:08x}")
}

/// Everything a VM needs to run one script.
#[derive(Clone, Debug)]
pub(crate) struct VmStart {
    pub(crate) code: String,
    /// `{ name, jsName, description }` per tool.
    pub(crate) tools_json: String,
    /// `{ name, spread }` per global.
    pub(crate) globals_json: String,
    /// Key to JSON text.
    pub(crate) store_json: String,
    pub(crate) memory_limit: Option<usize>,
    pub(crate) interrupt_budget: Option<u64>,
    /// `(started_at, seed)` of a deterministic run.
    pub(crate) deterministic: Option<(f64, u32)>,
}

/// A message for the VM.
#[derive(Debug)]
pub(crate) enum ToVm {
    /// `payload` is the JSON value when `ok` (`None`: undefined), else the error message.
    Result {
        id: u64,
        ok: bool,
        payload: Option<String>,
        seq: Option<u64>,
        now: Option<f64>,
        /// `ok: false`: a JSON object whose properties the script's `Error` gets.
        properties: Option<String>,
    },
    Stop,
}

/// A message from the VM.
#[derive(Debug)]
pub(crate) enum FromVm {
    Call {
        id: u64,
        global: bool,
        name: String,
        args: Option<String>,
    },
    Output(OutputItem),
    /// `writes` is a JSON array of `[key, json]` and `[key]`.
    Done {
        ok: bool,
        value: Option<String>,
        error: Option<String>,
        writes: Option<String>,
        during: Option<During>,
    },
    /// The VM failed outside the script's control.
    Crash(String),
    /// The interrupt budget ran out.
    Exhausted,
    Started(String),
    Idle {
        seq: u64,
        issued: u64,
    },
}

/// The host's handle on a VM thread.
pub(crate) struct VmHandle {
    sender: mpsc::Sender<ToVm>,
    stop: Arc<AtomicBool>,
}

impl VmHandle {
    pub(crate) fn post(&self, message: ToVm) {
        // A VM that already exited drops what it is sent.
        let _ = self.sender.send(message);
    }

    /// Interrupts the script at its next poll and ends the thread.
    pub(crate) fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = self.sender.send(ToVm::Stop);
    }
}

impl Drop for VmHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Starts the VM thread; `Err` only when the thread cannot be created.
pub(crate) fn spawn(start: VmStart, events: UnboundedSender<FromVm>) -> std::io::Result<VmHandle> {
    let (sender, receiver) = mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    thread::Builder::new()
        .name("codemode-vm".to_owned())
        .stack_size(THREAD_STACK)
        .spawn(move || {
            let post = Post {
                sender: events,
                exhausted: Arc::new(AtomicBool::new(false)),
            };
            if let Err(message) = run(&start, &receiver, &flag, &post) {
                post.send(message);
            }
        })?;
    Ok(VmHandle { sender, stop })
}

/// Sends to the host. Once the interrupt budget ran out, the terminal message the interrupt
/// causes (a crash or a failed `done`) is reported as `Exhausted` instead.
#[derive(Clone)]
struct Post {
    sender: UnboundedSender<FromVm>,
    exhausted: Arc<AtomicBool>,
}

impl Post {
    fn send(&self, message: FromVm) {
        let terminal = matches!(message, FromVm::Done { .. } | FromVm::Crash(_));
        let message = if terminal && self.exhausted.load(Ordering::SeqCst) {
            FromVm::Exhausted
        } else {
            message
        };
        // A host that went away ends the run at the next receive.
        let _ = self.sender.send(message);
    }
}

/// What the bridge and the thread loop share.
struct Shared {
    post: Post,
    /// Calls the script has issued, tools and globals.
    issued: Cell<u64>,
    /// The result being processed, 0 during the first run.
    current: Cell<u64>,
    deterministic: bool,
}

/// The prelude's API, kept across `Context::with` calls.
struct Api {
    run: Persistent<Function<'static>>,
    settle: Persistent<Function<'static>>,
    stalled: Persistent<Function<'static>>,
    set_now: Option<Persistent<Function<'static>>>,
}

/// The error the script threw, as upstream's `{ name, message, stack }` JSON.
fn exception_json(ctx: &Ctx<'_>) -> String {
    let caught = ctx.catch();
    let object = caught.as_object();
    let text = |key: &str| {
        object
            .and_then(|object| object.get::<_, Option<String>>(key).ok().flatten())
            .filter(|text| !text.is_empty())
    };
    let name = text("name").unwrap_or_else(|| "Error".to_owned());
    let message = text("message").unwrap_or_else(|| {
        caught
            .as_string()
            .and_then(|text| text.to_string().ok())
            .unwrap_or_default()
    });
    let head = if message.is_empty() {
        name.clone()
    } else {
        format!("{name}: {message}")
    };
    let stack = text("stack").map_or_else(|| head.clone(), |stack| format!("{head}\n{}", stack.trim_end()));
    serde_json::json!({ "name": name, "message": message, "stack": stack }).to_string()
}

fn to_text(value: Option<&Value<'_>>) -> Option<String> {
    let value = value?;
    if value.is_undefined() || value.is_null() {
        return None;
    }
    if let Some(text) = value.as_string() {
        return text.to_string().ok();
    }
    value.as_number().map(|number| number.to_string())
}

fn bridge<'js>(ctx: &Ctx<'js>, shared: Rc<Shared>) -> rquickjs::Result<Function<'js>> {
    Function::new(ctx.clone(), move |args: Rest<Value<'js>>| {
        let args = args.0;
        let kind = to_text(args.first()).unwrap_or_default();
        let (a, b, c) = (args.get(1), args.get(2), args.get(3));
        match kind.as_str() {
            "call" | "global" => {
                shared.issued.set(shared.issued.get() + 1);
                shared.post.send(FromVm::Call {
                    id: a.and_then(Value::as_number).map_or(0, |id| id as u64),
                    global: kind == "global",
                    name: to_text(b).unwrap_or_default(),
                    args: to_text(c),
                });
            }
            "output" => {
                let item = match to_text(a).as_deref() {
                    Some("image") => OutputItem::Image {
                        data: to_text(b).unwrap_or_default(),
                        mime_type: to_text(c).unwrap_or_default(),
                    },
                    Some(kind) => OutputItem::Text {
                        text: to_text(b).unwrap_or_default(),
                        console: kind == "console",
                    },
                    None => return,
                };
                shared.post.send(FromVm::Output(item));
            }
            "done" => {
                // The host stops the VM on `Done`, so this drain's `Idle` never arrives.
                let during = shared.deterministic.then(|| During {
                    seq: shared.current.get(),
                    issued: shared.issued.get(),
                });
                let ok = a.and_then(Value::as_bool).unwrap_or(false);
                shared.post.send(FromVm::Done {
                    ok,
                    value: if ok { to_text(b) } else { None },
                    error: if ok { None } else { to_text(b) },
                    writes: to_text(c),
                    during,
                });
            }
            _ => {}
        }
    })
}

#[allow(clippy::needless_pass_by_value)] // `map_err` passes the error by value.
fn crash(error: rquickjs::Error) -> FromVm {
    FromVm::Crash(format!("Sandbox VM failed: {error}"))
}

/// A call into the VM that failed without the script being able to catch it (an interrupt, or
/// an engine failure).
fn uncaught(ctx: &Ctx<'_>) -> FromVm {
    FromVm::Crash(format!("Execution interrupted: {}", exception_json(ctx)))
}

/// Runs queued jobs, then fails a script that waits on nothing that can ever resume it.
fn drain(runtime: &Runtime, context: &Context, api: &Api, stop: &AtomicBool) -> Result<(), FromVm> {
    let mut jobs: u32 = 0;
    loop {
        match runtime.execute_pending_job() {
            Ok(true) => {}
            Ok(false) => break,
            // A job only fails for uncatchable errors, such as an interrupt.
            Err(_) => return Err(FromVm::Crash("Promise job failed".to_owned())),
        }
        jobs += 1;
        if jobs.is_multiple_of(JOBS_PER_POLL) && stop.load(Ordering::SeqCst) {
            return Err(FromVm::Crash("Execution interrupted".to_owned()));
        }
    }
    context.with(|ctx| {
        let stalled = api.stalled.clone().restore(&ctx).map_err(crash)?;
        stalled.call::<_, Value<'_>>(()).map_err(|_| uncaught(&ctx))?;
        Ok(())
    })
}

/// Hands the VM one result.
fn apply(context: &Context, api: &Api, message: ToVm) -> Result<(), FromVm> {
    let ToVm::Result {
        id,
        ok,
        payload,
        properties,
        ..
    } = message
    else {
        return Ok(());
    };
    context.with(|ctx| {
        let settle = api.settle.clone().restore(&ctx).map_err(crash)?;
        settle
            .call::<_, Value<'_>>((id as f64, ok, payload, properties))
            .map_err(|_| uncaught(&ctx))?;
        Ok(())
    })
}

fn eval<'js>(ctx: &Ctx<'js>, source: impl Into<Vec<u8>>, filename: &str) -> Result<Value<'js>, ()> {
    let mut options = EvalOptions::default();
    options.strict = false;
    options.filename = Some(filename.to_owned());
    ctx.eval_with_options::<Value<'_>, _>(source, options).map_err(|_| ())
}

/// The fingerprint a deterministic run reports: what the VM is, so a replay can refuse a
/// different one.
pub(crate) fn fingerprint(memory_limit: Option<usize>, interrupt_budget: Option<u64>) -> String {
    let heap = memory_limit.map_or_else(|| "none".to_owned(), |limit| limit.to_string());
    let budget = interrupt_budget.map_or_else(|| "none".to_owned(), |budget| budget.to_string());
    format!(
        "{ENGINE}, prelude {}, deterministic {} {DETERMINISTIC_VERSION}, stack {MAX_STACK_SIZE}, heap {heap}, budget {budget}",
        fnv1a(PRELUDE_SOURCE),
        fnv1a(DETERMINISTIC_SOURCE),
    )
}

fn setup(context: &Context, start: &VmStart, shared: &Rc<Shared>) -> Result<Api, FromVm> {
    context.with(|ctx| {
        let set_now = match start.deterministic {
            Some((started_at, seed)) => {
                let install = eval(&ctx, DETERMINISTIC_SOURCE, "codemode-deterministic.js")
                    .map_err(|()| FromVm::Crash(format!("The deterministic prelude failed: {}", exception_json(&ctx))))?
                    .into_function()
                    .ok_or_else(|| FromVm::Crash("The deterministic prelude is not a function".to_owned()))?;
                let clock: Object<'_> = install.call((started_at, seed)).map_err(|_| {
                    FromVm::Crash(format!("The deterministic prelude failed: {}", exception_json(&ctx)))
                })?;
                let set_now: Function<'_> = clock.get("setNow").map_err(crash)?;
                Some(Persistent::save(&ctx, set_now))
            }
            None => None,
        };
        let prelude = eval(&ctx, PRELUDE_SOURCE, "codemode-prelude.js")
            .map_err(|()| FromVm::Crash(format!("The prelude failed: {}", exception_json(&ctx))))?
            .into_function()
            .ok_or_else(|| FromVm::Crash("The prelude is not a function".to_owned()))?;
        let api: Object<'_> = prelude
            .call((
                bridge(&ctx, shared.clone()).map_err(crash)?,
                start.tools_json.as_str(),
                start.globals_json.as_str(),
                start.store_json.as_str(),
            ))
            .map_err(|_| FromVm::Crash(format!("The prelude failed: {}", exception_json(&ctx))))?;
        let run: Function<'_> = api.get("run").map_err(crash)?;
        let settle: Function<'_> = api.get("settle").map_err(crash)?;
        let stalled: Function<'_> = api.get("stalled").map_err(crash)?;
        Ok(Api {
            run: Persistent::save(&ctx, run),
            settle: Persistent::save(&ctx, settle),
            stalled: Persistent::save(&ctx, stalled),
            set_now,
        })
    })
}

/// Compiles the script and starts it; `false` when it does not compile (reported as `Done`).
fn begin(context: &Context, start: &VmStart, api: &Api, post: &Post) -> Result<bool, FromVm> {
    context.with(|ctx| {
        // The prefix shares the first line with the script, so reported line numbers match
        // the script as written.
        let source = format!("(async (tools, console) => {{{}\n}})", start.code);
        let Ok(script) = eval(&ctx, source, "codemode.js") else {
            post.send(FromVm::Done {
                ok: false,
                value: None,
                error: Some(exception_json(&ctx)),
                writes: None,
                during: None,
            });
            return Ok(false);
        };
        let run = api.run.clone().restore(&ctx).map_err(crash)?;
        run.call::<_, Value<'_>>((script,)).map_err(|_| uncaught(&ctx))?;
        Ok(true)
    })
}

fn run(start: &VmStart, receiver: &mpsc::Receiver<ToVm>, stop: &Arc<AtomicBool>, post: &Post) -> Result<(), FromVm> {
    let runtime = Runtime::new().map_err(crash)?;
    if let Some(limit) = start.memory_limit {
        runtime.set_memory_limit(limit);
    }
    runtime.set_max_stack_size(MAX_STACK_SIZE);
    {
        let stop = stop.clone();
        let exhausted = post.exhausted.clone();
        let budget = start.interrupt_budget;
        let mut polls: u64 = 0;
        runtime.set_interrupt_handler(Some(Box::new(move || {
            if stop.load(Ordering::SeqCst) || exhausted.load(Ordering::SeqCst) {
                return true;
            }
            polls += 1;
            if budget.is_some_and(|budget| polls > budget) {
                exhausted.store(true, Ordering::SeqCst);
                return true;
            }
            false
        })));
    }
    let context = Context::full(&runtime).map_err(crash)?;
    let deterministic = start.deterministic.is_some();
    let shared = Rc::new(Shared {
        post: post.clone(),
        issued: Cell::new(0),
        current: Cell::new(0),
        deterministic,
    });
    let api = setup(&context, start, &shared)?;
    if deterministic {
        post.send(FromVm::Started(fingerprint(start.memory_limit, start.interrupt_budget)));
    }
    if begin(&context, start, &api, post)? {
        drain(&runtime, &context, &api, stop)?;
        if deterministic {
            post.send(FromVm::Idle {
                seq: 0,
                issued: shared.issued.get(),
            });
        }
    }

    let mut next_seq: u64 = 1;
    let mut early: BTreeMap<u64, ToVm> = BTreeMap::new();
    while let Ok(message) = receiver.recv() {
        let ToVm::Result { seq, .. } = &message else { break };
        if !deterministic {
            apply(&context, &api, message)?;
            drain(&runtime, &context, &api, stop)?;
            continue;
        }
        let Some(seq) = *seq else {
            return Err(FromVm::Crash("A result without a sequence number".to_owned()));
        };
        if seq < next_seq || early.contains_key(&seq) {
            return Err(FromVm::Crash(format!(
                "Result has sequence number {seq}, expected {next_seq} or later, once"
            )));
        }
        early.insert(seq, message);
        while let Some(message) = early.remove(&next_seq) {
            if let ToVm::Result { now: Some(now), .. } = &message
                && let Some(set_now) = &api.set_now
            {
                let now = *now;
                context.with(|ctx| {
                    let set_now = set_now.clone().restore(&ctx).map_err(crash)?;
                    set_now.call::<_, Value<'_>>((now,)).map_err(crash)?;
                    Ok::<_, FromVm>(())
                })?;
            }
            shared.current.set(next_seq);
            apply(&context, &api, message)?;
            drain(&runtime, &context, &api, stop)?;
            post.send(FromVm::Idle {
                seq: next_seq,
                issued: shared.issued.get(),
            });
            next_seq += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fingerprint names the engine; the lockfile pins it.
    #[test]
    fn engine_matches_the_lockfile() {
        let lock = include_str!("../../../Cargo.lock");
        let version = lock
            .split("[[package]]")
            .find(|package| package.contains("name = \"rquickjs\"\n"))
            .and_then(|package| package.lines().find_map(|line| line.strip_prefix("version = ")))
            .map(|version| version.trim_matches('"'));
        let version = version.unwrap_or_default();
        assert!(
            ENGINE.starts_with(&format!("rquickjs {version},")),
            "update ENGINE (and the QuickJS-NG version) for rquickjs {version}"
        );
    }

    #[test]
    fn fingerprints_name_the_engine_preludes_and_limits() {
        let fingerprint = fingerprint(Some(1024), None);
        assert!(fingerprint.starts_with(ENGINE));
        assert!(fingerprint.contains(&format!("prelude {}", fnv1a(PRELUDE_SOURCE))));
        assert!(fingerprint.ends_with("heap 1024, budget none"));
        assert_eq!(fnv1a(""), "811c9dc5");
    }
}
