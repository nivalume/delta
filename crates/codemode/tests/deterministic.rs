//! `Sandbox::start`, ported from upstream `deterministic.test.ts`: a script run with the caller
//! answering its calls, so a durable host can replay it after a restart.

#![allow(clippy::unwrap_used, clippy::too_many_lines)] // Test helpers outside #[test] functions.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use delta_codemode::{
    CallStatus, CallTarget, CodemodeResult, DeclaredGlobal, DeclaredTool, DeliverError, During, ErrorKind,
    ExecuteOptions, HostCall, Outcome, OutputItem, RunEvent, RunHandle, RunOptions, Sandbox, SandboxOptions,
};
use pretty_assertions::assert_eq;
use serde_json::{Value, json};

/// 2026-01-15T10:00:00Z.
const STARTED_AT: f64 = 1_768_471_200_000.0;

/// What a run reported.
#[derive(Debug, Default)]
struct Recorded {
    calls: Vec<HostCall>,
    idles: Vec<(u64, u64)>,
    events: Vec<String>,
    fingerprint: Option<String>,
}

type Answer = Box<dyn Fn(&HostCall) -> Outcome + Send + Sync>;

struct Started {
    handle: RunHandle,
    recorded: Arc<Mutex<Recorded>>,
    result: tokio::task::JoinHandle<CodemodeResult>,
}

impl Started {
    async fn result(self) -> CodemodeResult {
        tokio::time::timeout(Duration::from_secs(20), self.result)
            .await
            .unwrap()
            .unwrap()
    }

    fn calls(&self) -> Vec<HostCall> {
        self.recorded.lock().unwrap().calls.clone()
    }

    async fn wait_for(&self, check: impl Fn(&Recorded) -> bool) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while !check(&self.recorded.lock().unwrap()) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
    }
}

fn options() -> RunOptions {
    RunOptions {
        started_at: STARTED_AT,
        seed: 1,
        tools: vec![
            DeclaredTool {
                name: "echo".to_owned(),
                ..DeclaredTool::default()
            },
            DeclaredTool {
                name: "wait".to_owned(),
                ..DeclaredTool::default()
            },
        ],
        timeout: Some(Duration::from_secs(20)),
        ..RunOptions::default()
    }
}

/// Starts `code`. Every call but `wait` is answered at once (by default with its argument) as
/// result 1, 2, … with the clock at `STARTED_AT + 1000 * seq`; `wait` is left to the test.
fn start(sandbox: &Sandbox, code: &str, options: RunOptions, answer: Option<Answer>) -> Started {
    let answer = answer.unwrap_or_else(|| Box::new(|call: &HostCall| Outcome::ok(call.args.clone())));
    let mut run = sandbox.start(code, options).unwrap();
    let handle = run.handle();
    let recorded = Arc::new(Mutex::new(Recorded::default()));
    let record = recorded.clone();
    let answering = handle.clone();
    let result = tokio::spawn(async move {
        let mut seq = 0;
        while let Some(event) = run.next_event().await {
            match event {
                RunEvent::Started(fingerprint) => {
                    let mut recorded = record.lock().unwrap();
                    recorded.fingerprint = Some(fingerprint);
                    recorded.events.push("started".to_owned());
                }
                RunEvent::Call(call) => {
                    {
                        let mut recorded = record.lock().unwrap();
                        recorded.events.push(format!("call {}", call.id));
                        recorded.calls.push(call.clone());
                    }
                    if call.name != "wait" {
                        seq += 1;
                        let now = STARTED_AT + 1000.0 * seq as f64;
                        answering.deliver(seq, now, call.id, answer(&call)).unwrap();
                    }
                }
                RunEvent::Idle { seq, issued } => {
                    let mut recorded = record.lock().unwrap();
                    recorded.idles.push((seq, issued));
                    recorded.events.push(format!("idle {seq}"));
                }
                RunEvent::Finished(result) => return *result,
            }
        }
        panic!("the run ended without a result")
    });
    Started {
        handle,
        recorded,
        result,
    }
}

fn sandbox() -> Sandbox {
    Sandbox::new(SandboxOptions::default())
}

fn value(result: &CodemodeResult) -> Value {
    assert!(result.is_ok(), "{:?}", result.error);
    result.value.clone().unwrap_or(Value::Null)
}

/// Call durations are wall-clock time; everything else of two equal runs is the same.
fn timeless(mut result: CodemodeResult) -> CodemodeResult {
    for call in &mut result.calls {
        call.duration_ms = 0.0;
    }
    result
}

/// The first `Math.random()` with seed 1, pinned: a change here changes what replayed scripts
/// see, so it must change the fingerprint too.
const PINNED_FIRST_RANDOM: f64 = 0.627_073_936_184_761_8;
/// The first two `Math.random()` values with seed 0.
const PINNED_SEED_ZERO: [f64; 2] = [0.266_429_208_221_508_06, 0.223_272_026_674_465_44];

#[tokio::test]
async fn same_start_seed_and_results_make_the_same_calls_and_output() {
    let code = r"
        const values = [Date.now(), performance.now(), Math.random(), Math.random(), new Date().toISOString()];
        const first = await tools.echo(values);
        text(JSON.stringify([Date.now(), performance.now(), Math.random()]));
        return await tools.echo(first.concat(Math.random()));
    ";
    let sandbox = sandbox();
    let one = start(&sandbox, code, options(), None);
    let two = start(&sandbox, code, options(), None);
    let (one_calls, two_calls) = (one.recorded.clone(), two.recorded.clone());
    let (a, b) = (one.result().await, two.result().await);
    assert!(a.is_ok(), "{:?}", a.error);
    assert_eq!(timeless(a.clone()), timeless(b));
    let calls = one_calls.lock().unwrap().calls.clone();
    assert_eq!(calls, two_calls.lock().unwrap().calls);
    // The clock: started_at, then the time of the result being processed; performance.now()
    // counts from started_at.
    let args = calls[0].args.clone().unwrap();
    assert_eq!((args[0].as_f64(), args[1].as_f64()), (Some(STARTED_AT), Some(0.0)));
    assert_eq!(args[4], "2026-01-15T10:00:00.000Z");
    let OutputItem::Text { text, .. } = &a.output[0] else {
        panic!()
    };
    let printed: Vec<f64> = serde_json::from_str(text).unwrap();
    assert_eq!(printed[..2], [STARTED_AT + 1000.0, 1000.0]);
    let random = args[2].as_f64().unwrap();
    // Bit for bit: replays depend on it.
    assert_eq!(random.to_bits(), PINNED_FIRST_RANDOM.to_bits(), "{random}");

    // The seed decides the random values, not the start time.
    let later = start(
        &sandbox,
        code,
        RunOptions {
            started_at: STARTED_AT + 5000.0,
            ..options()
        },
        None,
    );
    let reseeded = start(&sandbox, code, RunOptions { seed: 2, ..options() }, None);
    let (later_calls, reseeded_calls) = (later.recorded.clone(), reseeded.recorded.clone());
    later.result().await;
    reseeded.result().await;
    assert_eq!(
        later_calls.lock().unwrap().calls[0].args.as_ref().unwrap()[2].as_f64(),
        Some(random)
    );
    assert_ne!(
        reseeded_calls.lock().unwrap().calls[0].args.as_ref().unwrap()[2].as_f64(),
        Some(random)
    );
    let zero = start(
        &sandbox,
        "return [Math.random(), Math.random()]",
        RunOptions { seed: 0, ..options() },
        None,
    );
    assert_eq!(value(&zero.result().await), json!(PINNED_SEED_ZERO));
}

#[tokio::test]
async fn dates_read_the_virtual_clock_in_utc() {
    let sandbox = sandbox();
    let run = start(
        &sandbox,
        r#"
        const local = new Date(2026, 0, 15, 10, 30);
        return [
            new Date(0).getHours(), new Date(0).getTimezoneOffset(), typeof WeakRef, typeof FinalizationRegistry,
            new Date().getHours(), String(new Date(0)), new Date(0).toLocaleString(), local.toISOString(),
            new Date("2026-01-15T10:00:00").getTime() === Date.now(), Date.parse("2026-01-15") === Date.UTC(2026, 0, 15),
            Date(), new Date(0) instanceof Date, typeof Date.now
        ];
    "#,
        options(),
        None,
    );
    assert_eq!(
        value(&run.result().await),
        json!([
            0,
            0,
            "undefined",
            "undefined",
            10,
            "Thu Jan 01 1970 00:00:00 GMT+0000",
            "01/01/1970, 12:00:00 AM",
            "2026-01-15T10:30:00.000Z",
            true,
            true,
            "Thu Jan 15 2026 10:00:00 GMT+0000",
            true,
            "function"
        ])
    );
}

/// Run by `timezone_does_not_reach_deterministic_runs` with `TZ=Asia/Kolkata`.
#[tokio::test]
#[ignore = "run in a child process with a timezone set"]
async fn timezone_probe() {
    let code =
        "return [new Date(0).getHours(), new Date(0).getTimezoneOffset(), typeof WeakRef, typeof FinalizationRegistry]";
    let sandbox = sandbox();
    let run = start(&sandbox, code, options(), None);
    assert_eq!(value(&run.result().await), json!([0, 0, "undefined", "undefined"]));
    // execute() is unchanged: every intrinsic, and the host's timezone. On Windows QuickJS
    // reads the system time zone (`GetTimeZoneInformation`), not `TZ`, so only Unix shows the
    // child's timezone.
    let executed = value(&sandbox.execute(code, ExecuteOptions::default()).await);
    assert_eq!(
        executed.as_array().unwrap()[2..],
        [json!("function"), json!("function")]
    );
    if cfg!(unix) {
        assert_eq!(executed, json!([5, -330, "function", "function"]));
    }
}

#[test]
fn timezone_does_not_reach_deterministic_runs() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "timezone_probe", "--quiet"])
        .env("TZ", "Asia/Kolkata")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[tokio::test]
async fn results_apply_in_seq_order_whatever_order_they_arrive_in() {
    let sandbox = sandbox();
    let run = start(
        &sandbox,
        r#"const order = [];
        await Promise.all([tools.wait("a").then(() => order.push("a")), tools.wait("b").then(() => order.push("b"))]);
        return order;"#,
        options(),
        None,
    );
    run.wait_for(|recorded| recorded.calls.len() == 2 && recorded.idles.len() == 1)
        .await;
    let calls = run.calls();
    // b is result 1 and a result 2, but a arrives first.
    run.handle
        .deliver(2, STARTED_AT + 2000.0, calls[0].id, Outcome::ok(None))
        .unwrap();
    run.handle
        .deliver(1, STARTED_AT + 1000.0, calls[1].id, Outcome::ok(None))
        .unwrap();
    let recorded = run.recorded.clone();
    assert_eq!(value(&run.result().await), json!(["b", "a"]));
    // No idle after the last result: the script finished in that drain and the host stopped
    // the VM.
    assert_eq!(recorded.lock().unwrap().idles, [(0, 2), (1, 2)]);
}

#[tokio::test]
async fn the_fingerprint_comes_first_and_idles_count_issued_calls() {
    let sandbox = Sandbox::new(SandboxOptions {
        memory_limit: Some(64 * 1024 * 1024),
        ..SandboxOptions::default()
    });
    let run = start(
        &sandbox,
        "await tools.echo(1); await Promise.all([tools.echo(2), tools.echo(3)]);",
        options(),
        None,
    );
    let recorded = run.recorded.clone();
    assert!(run.result().await.is_ok());
    let recorded = recorded.lock().unwrap();
    assert_eq!(recorded.events[..3], ["started", "call 1", "idle 0"]);
    assert_eq!(recorded.idles, [(0, 1), (1, 3), (2, 3)]);
    let calls: Vec<(u64, CallTarget, &str, Option<Value>)> = recorded
        .calls
        .iter()
        .map(|call| (call.id, call.target, call.name.as_str(), call.args.clone()))
        .collect();
    assert_eq!(
        calls,
        [
            (1, CallTarget::Tool, "echo", Some(json!(1))),
            (2, CallTarget::Tool, "echo", Some(json!(2))),
            (3, CallTarget::Tool, "echo", Some(json!(3))),
        ]
    );
    let fingerprint = recorded.fingerprint.clone().unwrap();
    assert!(fingerprint.starts_with("rquickjs "), "{fingerprint}");
    assert!(
        fingerprint.contains(", prelude ") && fingerprint.ends_with("heap 67108864, budget none"),
        "{fingerprint}"
    );
    assert_eq!(sandbox.fingerprint(), fingerprint);
    let other = Sandbox::new(SandboxOptions {
        memory_limit: Some(32 * 1024 * 1024),
        ..SandboxOptions::default()
    });
    assert_ne!(other.fingerprint(), fingerprint);
}

#[tokio::test]
async fn globals_share_call_numbers_and_errors_reject() {
    let sandbox = sandbox();
    let run = start(
        &sandbox,
        r#"const found = await search("x", 2);
        try { await tools.echo("fail"); } catch (error) { text(error.message); }
        return found;"#,
        RunOptions {
            globals: vec![DeclaredGlobal {
                name: "search".to_owned(),
                spread: true,
            }],
            ..options()
        },
        Some(Box::new(|call: &HostCall| {
            if call.args == Some(json!("fail")) {
                Outcome::err("no")
            } else if call.name == "search" {
                Outcome::ok(json!(["found"]))
            } else {
                Outcome::ok(call.args.clone())
            }
        })),
    );
    let recorded = run.recorded.clone();
    let result = run.result().await;
    let recorded = recorded.lock().unwrap();
    assert_eq!(
        recorded.calls[0],
        HostCall {
            id: 1,
            target: CallTarget::Global,
            name: "search".to_owned(),
            args: Some(json!(["x", 2]))
        }
    );
    assert_eq!(value(&result), json!(["found"]));
    assert_eq!(
        result.output,
        [OutputItem::Text {
            text: "no".to_owned(),
            console: false
        }]
    );
    assert_eq!((recorded.calls[1].id, recorded.calls[1].target), (2, CallTarget::Tool));
    assert_eq!(recorded.idles[..2], [(0, 1), (1, 2)]);
    assert_eq!(result.calls.len(), 1);
    assert_eq!(result.calls[0].status, CallStatus::Error);
}

#[tokio::test]
async fn the_clock_never_goes_backwards() {
    let sandbox = sandbox();
    let run = start(
        &sandbox,
        "const before = Date.now(); await tools.wait(1); return [before, Date.now(), performance.now()]",
        options(),
        None,
    );
    run.wait_for(|recorded| recorded.calls.len() == 1).await;
    assert_eq!(
        run.handle.deliver(1, f64::NAN, 1, Outcome::ok(None)),
        Err(DeliverError::Clock(1))
    );
    // A resumed host whose clock is behind the run's start.
    run.handle
        .deliver(1, STARTED_AT - 2000.0, 1, Outcome::ok(None))
        .unwrap();
    let started = STARTED_AT as i64;
    assert_eq!(value(&run.result().await), json!([started, started, 0]));
}

#[tokio::test]
async fn results_report_where_the_script_ended() {
    let sandbox = sandbox();
    let returned = start(
        &sandbox,
        "await tools.echo(1); return await tools.echo(2)",
        options(),
        None,
    );
    let thrown = start(
        &sandbox,
        "await tools.echo(1); await Promise.all([tools.echo(2), tools.echo(3)]); throw new Error('x')",
        options(),
        None,
    );
    let first = start(&sandbox, "return 1", options(), None);
    let returned = returned.result().await;
    assert_eq!(
        (value(&returned), returned.during),
        (json!(2), Some(During { seq: 2, issued: 2 }))
    );
    let thrown = thrown.result().await;
    assert!(!thrown.is_ok());
    assert_eq!(thrown.during, Some(During { seq: 3, issued: 3 }));
    assert_eq!(first.result().await.during, Some(During { seq: 0, issued: 0 }));
    assert_eq!(
        sandbox.execute("return 1", ExecuteOptions::default()).await.during,
        None
    );
}

#[tokio::test]
async fn rejections_carry_the_answers_properties() {
    let sandbox = sandbox();
    let run = start(
        &sandbox,
        "try { await tools.echo(1); } catch (e) { return [e instanceof Error, e.message, e.name, e.structuredOutput]; }",
        options(),
        Some(Box::new(|_: &HostCall| Outcome::Err {
            message: "denied".to_owned(),
            properties: json!({ "structuredOutput": { "code": 7 }, "name": "Ignored", "message": "ignored" })
                .as_object()
                .cloned(),
        })),
    );
    assert_eq!(
        value(&run.result().await),
        json!([true, "denied", "Error", { "code": 7 }])
    );
}

#[tokio::test]
async fn script_names_are_unique_and_taken_from_the_run() {
    let sandbox = sandbox();
    let shared = sandbox.start(
        "",
        RunOptions {
            tools: vec![
                DeclaredTool {
                    name: "a-b".to_owned(),
                    ..DeclaredTool::default()
                },
                DeclaredTool {
                    name: "a_b".to_owned(),
                    ..DeclaredTool::default()
                },
            ],
            ..options()
        },
    );
    assert!(shared.is_err());
    let run = start(
        &sandbox,
        "return [typeof tools.renamed, Object.keys(tools), ALL_TOOLS.map((tool) => tool.name)]",
        RunOptions {
            tools: vec![DeclaredTool {
                name: "my-tool".to_owned(),
                js_name: Some("renamed".to_owned()),
                description: String::new(),
            }],
            ..options()
        },
        None,
    );
    assert_eq!(
        value(&run.result().await),
        json!(["function", ["renamed", "my-tool"], ["renamed"]])
    );
}

#[tokio::test]
async fn an_abort_ends_the_run_with_an_error_the_script_cannot_catch() {
    let sandbox = sandbox();
    let run = start(
        &sandbox,
        "try { await tools.wait(1); } catch { text(\"caught\"); }\ntext(\"after\");",
        options(),
        None,
    );
    run.wait_for(|recorded| recorded.calls.len() == 1).await;
    assert!(!run.handle.finished());
    run.handle.abort();
    let handle = run.handle.clone();
    let result = run.result().await;
    let error = result.error.unwrap();
    assert_eq!(error.kind, ErrorKind::Aborted);
    assert!(result.output.is_empty());
    assert!(handle.finished());
    // A finished run drops later answers.
    assert_eq!(handle.deliver(1, STARTED_AT, 1, Outcome::ok(None)), Ok(()));
}

#[tokio::test]
async fn deliveries_are_checked() {
    let sandbox = sandbox();
    let run = start(
        &sandbox,
        "await Promise.all([tools.wait(1), tools.wait(2)])",
        options(),
        None,
    );
    run.wait_for(|recorded| recorded.calls.len() == 2).await;
    let handle = &run.handle;
    assert_eq!(
        handle.deliver(1, STARTED_AT, 99, Outcome::ok(None)),
        Err(DeliverError::NotWaiting(99))
    );
    handle.deliver(1, STARTED_AT, 1, Outcome::ok(None)).unwrap();
    assert_eq!(
        handle.deliver(1, STARTED_AT, 2, Outcome::ok(None)),
        Err(DeliverError::Seq(1))
    );
    assert_eq!(
        handle.deliver(0, STARTED_AT, 2, Outcome::ok(None)),
        Err(DeliverError::Seq(0))
    );
    handle.deliver(2, STARTED_AT, 2, Outcome::ok(None)).unwrap();
    assert!(run.result().await.is_ok());
}

#[tokio::test]
async fn execute_reads_the_real_clock_and_randomness() {
    let sandbox = sandbox();
    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as f64;
    let first = sandbox
        .execute("return [Date.now(), Math.random()]", ExecuteOptions::default())
        .await;
    let second = sandbox
        .execute("return [Date.now(), Math.random()]", ExecuteOptions::default())
        .await;
    let first = value(&first);
    assert!(first[0].as_f64().unwrap() >= before);
    assert_ne!(value(&second)[1], first[1]);
}
