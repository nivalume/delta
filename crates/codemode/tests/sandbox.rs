//! `Sandbox::execute`, ported from upstream `sandbox.test.ts`.

#![allow(clippy::unwrap_used, clippy::too_many_lines)] // Test helpers outside #[test] functions.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use delta_codemode::{
    CallStatus, CodemodeResult, CodemodeTool, ErrorKind, ExecuteOptions, FnCodemodeTool, MAX_OUTPUT_CHARS,
    MAX_OUTPUT_ITEMS, OutputItem, Sandbox, SandboxError, SandboxOptions,
};
use pretty_assertions::assert_eq;
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

// Base64 of the leading bytes of each format; image() only inspects the signature.
const PNG: &str = "iVBORw0KGgo=";
const JPEG: &str = "/9j/4A==";
const GIF: &str = "R0lGODlh";
const WEBP: &str = "UklGRgAAAABXRUJQ";

fn tool(
    name: &str,
    run: impl Fn(Option<Value>) -> Result<Option<Value>, String> + Send + Sync + 'static,
) -> Arc<dyn CodemodeTool> {
    let run = Arc::new(run);
    Arc::new(FnCodemodeTool::new(name, move |args, _| {
        let run = run.clone();
        async move { run(args) }
    }))
}

fn echo() -> Arc<dyn CodemodeTool> {
    tool("echo", Ok)
}

fn sandbox(tools: Vec<Arc<dyn CodemodeTool>>) -> Sandbox {
    let mut sandbox = Sandbox::new(SandboxOptions {
        timeout: Some(Duration::from_secs(10)),
        ..SandboxOptions::default()
    });
    for tool in tools {
        sandbox = sandbox.with_tool(tool).unwrap();
    }
    sandbox
}

async fn run(sandbox: &Sandbox, code: &str) -> CodemodeResult {
    sandbox.execute(code, ExecuteOptions::default()).await
}

fn value(result: &CodemodeResult) -> &Value {
    assert!(result.is_ok(), "{:?}", result.error);
    result.value.as_ref().unwrap_or(&Value::Null)
}

fn error(result: &CodemodeResult) -> &delta_codemode::CodemodeError {
    result
        .error
        .as_ref()
        .unwrap_or_else(|| panic!("succeeded: {:?}", result.value))
}

fn text(text: &str) -> OutputItem {
    OutputItem::Text {
        text: text.to_owned(),
        console: false,
    }
}

fn image(data: &str, mime_type: &str) -> OutputItem {
    OutputItem::Image {
        data: data.to_owned(),
        mime_type: mime_type.to_owned(),
    }
}

#[tokio::test]
async fn returns_values_after_a_json_round_trip() {
    let sandbox = sandbox(Vec::new());
    let result = run(&sandbox, "return { a: 1, b: [true, 'x'] }").await;
    assert_eq!(value(&result), &json!({ "a": 1, "b": [true, "x"] }));
    assert!(result.output.is_empty() && result.calls.is_empty());
    assert_eq!(value(&run(&sandbox, "return 'plain'").await), "plain");
    let empty = run(&sandbox, "").await;
    assert!(empty.is_ok() && empty.value.is_none());
    assert_eq!(
        value(&run(&sandbox, "const x = await Promise.resolve(41); return x + 1").await),
        42
    );
}

#[tokio::test]
async fn output_is_collected_in_order() {
    let sandbox = sandbox(Vec::new());
    let result = run(
        &sandbox,
        &format!(
            r#"
            console.log("hello", 1, {{ a: 1 }});
            text({{ json: true }});
            text(undefined);
            text(7);
            image("data:image/png;base64,{PNG}");
            image({{ image_url: "data:image/jpeg;base64,{JPEG}" }});
            image({{ type: "image", data: "{GIF}", mimeType: "image/gif" }});
            image("data:image/png;base64,{WEBP}");
            image({{ type: "image", data: "{PNG}" }});
            console.error(new Error("bad"));
            return null;
        "#
        ),
    )
    .await;
    assert!(result.is_ok());
    let (last, rest) = result.output.split_last().unwrap();
    assert_eq!(
        rest,
        [
            OutputItem::Text {
                text: "hello 1 {\"a\":1}".to_owned(),
                console: true
            },
            text("{\"json\":true}"),
            text("undefined"),
            text("7"),
            image(PNG, "image/png"),
            image(JPEG, "image/jpeg"),
            image(GIF, "image/gif"),
            // The MIME type comes from the data, not from the declared type.
            image(WEBP, "image/webp"),
            image(PNG, "image/png"),
        ]
    );
    let OutputItem::Text { text, console: true } = last else {
        panic!("{last:?}")
    };
    assert!(text.starts_with("Error: bad"), "{text}");
}

#[tokio::test]
async fn invalid_text_and_image_arguments_throw() {
    let sandbox = sandbox(Vec::new());
    let result = run(
        &sandbox,
        r#"
        const errors = [];
        const circular = {};
        circular.self = circular;
        for (const run of [
            () => text(circular),
            () => image(""),
            () => image("https://example.com/a.png"),
            () => image("data:image/png,raw"),
            () => image({ type: "text", text: "x" }),
            () => image({ type: "image", data: "" }),
            () => image(42),
            () => image("data:image/png;base64,AAAA!"),
            () => image("data:image/png;base64,AAAAA"),
            () => image("data:image/png;base64,AA=A"),
            () => image("data:image/png;base64,"),
            () => image("data:image/png;base64,AAAA\n[Output truncated]"),
            () => image({ type: "image", data: "AAAA!", mimeType: "image/png" }),
            () => image("data:image/png;base64,AAAA"),
            () => image("data:image/png;base64,QUJD"),
            () => image("data:image/jpeg;base64,/9j/9w=="),
        ]) {
            try { run(); errors.push("no error"); } catch (error) { errors.push(error.name + ": " + error.message); }
        }
        return errors;
    "#,
    )
    .await;
    assert!(result.output.is_empty());
    let errors: Vec<String> = serde_json::from_value(value(&result).clone()).unwrap();
    assert!(
        errors[0].starts_with("TypeError: ") && errors[0].contains("ircular"),
        "{}",
        errors[0]
    );
    let expects = "TypeError: image expects a non-empty image URL string, an object with image_url, or an image block { type: \"image\", data, mimeType }";
    let mut expected = vec![
        expects.to_owned(),
        "TypeError: remote image URLs are not supported in tool outputs. Pass a base64 data URI instead".to_owned(),
        "TypeError: invalid image output. Pass a base64 data URI instead".to_owned(),
        "TypeError: image only accepts image blocks, got \"text\"".to_owned(),
        "TypeError: image expected image block data".to_owned(),
        expects.to_owned(),
    ];
    expected.extend(std::iter::repeat_n(
        "TypeError: invalid image output. The image data is not valid base64 (truncated or corrupted?)".to_owned(),
        6,
    ));
    expected.extend(std::iter::repeat_n(
        "TypeError: invalid image output. The image data is not a PNG, JPEG, GIF, or WebP image".to_owned(),
        3,
    ));
    assert_eq!(errors[1..], expected);
}

#[tokio::test]
async fn wrapped_and_large_base64_images_are_accepted() {
    let sandbox = sandbox(Vec::new());
    let large = format!("iVBORw0KGgoA{}", "QUJD".repeat(256 * 1024));
    let result = run(
        &sandbox,
        &format!("image(\"data:image/png;base64,iVBORw0K\\r\\nGgo=\\n\");\nimage(\"data:image/png;base64,{large}\");"),
    )
    .await;
    assert!(result.is_ok(), "{:?}", result.error);
    assert_eq!(result.output, [image(PNG, "image/png"), image(&large, "image/png")]);
}

#[tokio::test]
async fn exit_ends_successfully_keeping_output_and_store_writes() {
    let sandbox = sandbox(vec![echo()]);
    let result = run(
        &sandbox,
        r#"
        text("before");
        store("k", 1);
        await tools.echo(1);
        try { exit(); } catch {}
        text("after");
        return "unreachable";
    "#,
    )
    .await;
    assert!(result.is_ok() && result.value.is_none());
    assert_eq!(result.output, [text("before")]);
    assert_eq!(result.store_writes.set.get("k"), Some(&json!(1)));
}

#[tokio::test]
async fn errors_keep_output_and_point_at_script_lines() {
    let sandbox = sandbox(Vec::new());
    let result = run(&sandbox, "text(\"partial\");\nthrow new Error(\"boom\")").await;
    assert!(!result.is_ok());
    assert_eq!(result.output, [text("partial")]);

    let syntax = run(&sandbox, "const a = 1;\nconst b = ;\nreturn a").await;
    let syntax = error(&syntax);
    assert_eq!(
        (syntax.kind, syntax.name.as_deref()),
        (ErrorKind::Script, Some("SyntaxError"))
    );
    assert!(
        syntax.stack.as_deref().unwrap_or_default().contains("codemode.js:2"),
        "{syntax:?}"
    );

    let thrown = run(&sandbox, "const a = 1;\nthrow new TypeError('boom ' + a)").await;
    let thrown = error(&thrown);
    assert_eq!(
        (thrown.kind, thrown.name.as_deref(), thrown.message.as_str()),
        (ErrorKind::Script, Some("TypeError"), "boom 1")
    );
    assert!(thrown.stack.as_deref().unwrap_or_default().contains("codemode.js:2"));

    let stacked = run(
        &sandbox,
        "console.log(new Error('inner'));\nthrow new RangeError('outer')",
    )
    .await;
    let stack = error(&stacked).stack.clone().unwrap_or_default();
    assert!(stack.starts_with("RangeError: outer\n    at "), "{stack}");
    assert!(
        stack.contains("codemode.js:2") && !stack.contains("codemode-prelude.js"),
        "{stack}"
    );
    let OutputItem::Text { text: inner, .. } = &stacked.output[0] else {
        panic!()
    };
    assert!(
        inner.starts_with("Error: inner\n    at ") && inner.contains("codemode.js:1"),
        "{inner}"
    );

    let object = run(&sandbox, "throw { code: 7 }").await;
    assert_eq!(error(&object).message, "{\"code\":7}");
    let bigint = run(&sandbox, "return 10n").await;
    assert_eq!(error(&bigint).name.as_deref(), Some("TypeError"));
    let numeric = run(
        &sandbox,
        "const error = new Error('x'); error.message = 42; throw error;",
    )
    .await;
    assert_eq!(
        (error(&numeric).name.as_deref(), error(&numeric).message.as_str()),
        (Some("Error"), "42")
    );
}

#[tokio::test]
async fn tools_are_async_functions_with_call_records() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorded = seen.clone();
    let sandbox = sandbox(vec![tool("add", move |args| {
        let args = args.unwrap_or_default();
        recorded.lock().unwrap().push(args.clone());
        Ok(Some(
            json!({ "sum": args["a"].as_i64().unwrap_or(0) + args["b"].as_i64().unwrap_or(0) }),
        ))
    })]);
    let result = run(
        &sandbox,
        "const first = await tools.add({ a: 1, b: 2 });\nconst second = await tools.add({ a: first.sum, b: 10 });\nreturn second.sum;",
    )
    .await;
    assert_eq!(value(&result), 13);
    assert_eq!(
        *seen.lock().unwrap(),
        [json!({ "a": 1, "b": 2 }), json!({ "a": 3, "b": 10 })]
    );
    let calls: Vec<(&str, CallStatus)> = result
        .calls
        .iter()
        .map(|call| (call.name.as_str(), call.status))
        .collect();
    assert_eq!(calls, [("add", CallStatus::Ok), ("add", CallStatus::Ok)]);
    assert!(result.calls.iter().all(|call| call.duration_ms >= 0.0));
}

#[tokio::test]
async fn concurrent_calls_and_normalized_names() {
    let delay = Arc::new(FnCodemodeTool::new("delay", |args, _| async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        Ok(args)
    }));
    let sandbox = sandbox(vec![echo(), delay]);
    let result = run(
        &sandbox,
        "const [a, b, c] = await Promise.all([tools.delay(1), tools.delay(2), tools.echo(3)]);\nreturn { values: [a, b, c], names: Object.keys(tools) };",
    )
    .await;
    assert_eq!(
        value(&result),
        &json!({ "values": [1, 2, 3], "names": ["echo", "delay"] })
    );

    let named = Sandbox::new(SandboxOptions::default())
        .with_tool(Arc::new(
            FnCodemodeTool::new("my-tool", |_, _| async { Ok(Some(json!("dash"))) }).with_description("Dashes"),
        ))
        .unwrap()
        .with_tool(Arc::new(
            FnCodemodeTool::new("my_tool", |_, _| async { Ok(Some(json!("underscore"))) }).with_description("Shadowed"),
        ))
        .unwrap()
        .with_tool(Arc::new(FnCodemodeTool::new("mcp__docs__search", |_, _| async {
            Ok(Some(json!("mcp")))
        })))
        .unwrap();
    let result = run(
        &named,
        "try { ALL_TOOLS.push({}); } catch {}\nreturn { all: ALL_TOOLS, calls: [await tools.my_tool(), await tools[\"my-tool\"](), await tools.mcp__docs__search()] };",
    )
    .await;
    assert_eq!(
        value(&result),
        &json!({
            "all": [{ "name": "my_tool", "description": "Dashes" }, { "name": "mcp__docs__search", "description": "" }],
            "calls": ["dash", "dash", "mcp"]
        })
    );
    assert!(matches!(
        Sandbox::new(SandboxOptions::default())
            .with_tool(echo())
            .unwrap()
            .with_tool(echo()),
        Err(SandboxError::DuplicateTool(_))
    ));
}

#[tokio::test]
async fn undefined_errors_and_unknown_tools() {
    let sandbox = sandbox(vec![
        tool("noop", Ok),
        tool("fail", |_| Err("tool exploded".to_owned())),
    ]);
    assert_eq!(
        value(&run(&sandbox, "return [await tools.noop(), await tools.noop(null)]").await),
        &json!([null, null])
    );
    let result = run(
        &sandbox,
        "try { await tools.fail(); return 'no error'; } catch (error) { return { isError: error instanceof Error, message: error.message }; }",
    )
    .await;
    assert_eq!(value(&result), &json!({ "isError": true, "message": "tool exploded" }));
    assert_eq!(result.calls[0].status, CallStatus::Error);
    let missing = run(&sandbox, "return await tools.missing()").await;
    assert_eq!(
        (error(&missing).kind, error(&missing).name.as_deref()),
        (ErrorKind::Script, Some("TypeError"))
    );
}

#[tokio::test]
async fn unawaited_calls_are_cancelled_when_the_script_returns() {
    let aborted = Arc::new(AtomicBool::new(false));
    let flag = aborted.clone();
    let slow = Arc::new(FnCodemodeTool::new("slow", move |_, cancel: CancellationToken| {
        let flag = flag.clone();
        async move {
            cancel.cancelled().await;
            flag.store(true, Ordering::SeqCst);
            Err("aborted".to_owned())
        }
    }));
    let sandbox = sandbox(vec![slow]);
    let result = run(&sandbox, "tools.slow(); return 'early'").await;
    assert_eq!(value(&result), "early");
    assert_eq!(result.calls[0].status, CallStatus::Cancelled);
    tokio::time::timeout(Duration::from_secs(5), async {
        while !aborted.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn missing_members_name_close_matches() {
    let sandbox = sandbox(vec![echo(), tool("web-search", |_| Ok(Some(json!(""))))]);
    let attempt = |expression: &str| {
        let code = format!("return {expression};");
        let sandbox = &sandbox;
        async move {
            let result = run(sandbox, &code).await;
            match result.error {
                Some(error) => Value::String(error.message),
                None => result.value.unwrap_or(Value::Null),
            }
        }
    };
    assert_eq!(
        attempt("tools.Echo").await,
        "tools.Echo does not exist. Did you mean tools.echo? ALL_TOOLS lists every tool; searchTools(query) finds tools by topic. Check for a member with \"Echo\" in tools."
    );
    assert!(
        attempt("tools.websearch")
            .await
            .as_str()
            .unwrap()
            .contains("Did you mean tools.web_search?")
    );
    assert!(
        attempt("tools.nothing")
            .await
            .as_str()
            .unwrap()
            .contains("Available: echo, web_search.")
    );
    assert_eq!(
        attempt("['echo' in tools, 'nothing' in tools, String(tools.toString), JSON.stringify(tools)]").await,
        json!([true, false, "undefined", "{}"])
    );
}

#[tokio::test]
async fn store_reads_the_snapshot_and_reports_writes() {
    let sandbox = sandbox(Vec::new());
    let store: Map<String, Value> = json!({ "counter": 41, "old": "x", "obj": { "a": 1 } })
        .as_object()
        .cloned()
        .unwrap();
    let result = sandbox
        .execute(
            r#"
            const seen = load("counter");
            store("counter", seen + 1);
            store("list", [1, { a: null }]);
            store("old", undefined);
            const value = load("obj"); value.a = 2;
            const kept = { b: 1 }; store("kept", kept); kept.b = 2;
            return [seen, load("counter"), load("missing"), load("old"), load("obj").a, load("kept").b];
        "#,
            ExecuteOptions {
                store,
                ..ExecuteOptions::default()
            },
        )
        .await;
    assert_eq!(value(&result), &json!([41, 42, null, null, 1, 1]));
    assert_eq!(
        Value::Object(result.store_writes.set.clone()),
        json!({ "counter": 42, "list": [1, { "a": null }], "kept": { "b": 1 } })
    );
    assert_eq!(result.store_writes.delete, ["old"]);

    let invalid = run(
        &sandbox,
        r#"
        const attempt = (fn) => { try { fn(); return "ok"; } catch (error) { return error.name; } };
        return [
            attempt(() => store(1, "x")),
            attempt(() => load({})),
            attempt(() => store("fn", () => 1)),
            attempt(() => store("big", "x".repeat(300 * 1024))),
            attempt(() => { for (let i = 0; i < 8; i++) store("k" + i, "x".repeat(200 * 1024)); }),
        ];
    "#,
    )
    .await;
    assert_eq!(
        value(&invalid),
        &json!(["TypeError", "TypeError", "TypeError", "RangeError", "RangeError"])
    );
    let oversized = run(&sandbox, "store(\"img\", \"x\".repeat(300 * 1024));").await;
    let message = &error(&oversized).message;
    assert!(
        message.contains("store(\"img\") value has 307202 characters of JSON"),
        "{message}"
    );
    assert!(message.contains("Show images with image()"));
    // A failed run reports no writes.
    let failed = run(&sandbox, "store('a', 1); throw new Error('x')").await;
    assert!(failed.store_writes.set.is_empty());
}

#[tokio::test]
async fn globals_are_top_level_functions_without_call_records() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorded = seen.clone();
    let sandbox = sandbox(vec![echo()])
        .with_global(tool("attach", move |args| {
            recorded.lock().unwrap().push(args.unwrap_or(Value::Null));
            Ok(None)
        }))
        .unwrap();
    let result = run(
        &sandbox,
        "await attach({ ref: 1 });\nattach(\"not awaited\");\nreturn [typeof attach, typeof globalThis.attach, await tools.echo(2)];",
    )
    .await;
    assert_eq!(value(&result), &json!(["function", "function", 2]));
    assert_eq!(result.calls.len(), 1);
    // Messages are handled in order, so an unawaited global still runs before the script ends.
    assert_eq!(*seen.lock().unwrap(), [json!({ "ref": 1 }), json!("not awaited")]);

    let listed = Arc::new(Mutex::new(Vec::new()));
    let recorded = listed.clone();
    let namespaced = Sandbox::new(SandboxOptions::default())
        .with_global(Arc::new(
            FnCodemodeTool::new("models.list", move |args, _| {
                recorded.lock().unwrap().push(args.unwrap_or(Value::Null));
                async { Ok(None) }
            })
            .spread(),
        ))
        .unwrap()
        .with_global(tool("models.first", Ok))
        .unwrap();
    let result = run(
        &namespaced,
        "await models.list(\"classifier\", undefined, 3);\nawait models.list();\ntry { models.extra = 1; } catch {}\nreturn [Object.keys(models), await models.first(\"a\", \"ignored\"), \"extra\" in models];",
    )
    .await;
    assert_eq!(value(&result), &json!([["list", "first"], "a", false]));
    assert_eq!(*listed.lock().unwrap(), [json!(["classifier", null, 3]), json!([])]);
    let typo = run(&namespaced, "await models.firstt();").await;
    assert_eq!(
        error(&typo).message,
        "models.firstt does not exist. Did you mean models.first? Check for a member with \"firstt\" in models."
    );
}

#[test]
fn invalid_and_reserved_global_names_are_refused() {
    for name in [
        "a.b.c",
        "a.",
        ".a",
        "tools.x",
        "store.x",
        "a.not-valid",
        "not-valid",
        "tools",
        "console",
        "store",
        "load",
    ] {
        let result = Sandbox::new(SandboxOptions::default()).with_global(tool(name, Ok));
        assert!(matches!(result, Err(SandboxError::InvalidGlobal(_))), "{name}");
    }
    let conflict = Sandbox::new(SandboxOptions::default())
        .with_global(tool("models", Ok))
        .unwrap()
        .with_global(tool("models.list", Ok));
    assert!(matches!(conflict, Err(SandboxError::NamespaceConflict(_))));
}

#[tokio::test]
async fn loops_time_out_and_stalled_scripts_fail() {
    let sandbox = sandbox(vec![echo()]);
    for code in ["while (true) {}", "while (true) await null"] {
        let started = Instant::now();
        let result = sandbox
            .execute(
                code,
                ExecuteOptions {
                    timeout: Some(Duration::from_millis(200)),
                    ..ExecuteOptions::default()
                },
            )
            .await;
        assert_eq!(error(&result).kind, ErrorKind::Timeout, "{code}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }
    let stalled = run(
        &sandbox,
        "await tools.echo(1); await new Promise(() => {}); return 'never'",
    )
    .await;
    assert_eq!(error(&stalled).kind, ErrorKind::Script);
    assert!(error(&stalled).message.contains("can never settle"));
    // Returning while a call is still pending is not a stall.
    assert_eq!(value(&run(&sandbox, "tools.echo(2); return 'early'").await), "early");
    let unlimited = Sandbox::new(SandboxOptions {
        timeout: None,
        ..SandboxOptions::default()
    })
    .with_tool(Arc::new(FnCodemodeTool::new("wait", |_, _| async {
        tokio::time::sleep(Duration::from_millis(50)).await;
        Ok(Some(json!("late")))
    })))
    .unwrap();
    assert_eq!(value(&run(&unlimited, "return await tools.wait()").await), "late");
}

#[tokio::test]
async fn interrupt_budgets_end_computing_scripts() {
    let sandbox = Sandbox::new(SandboxOptions {
        timeout: None,
        interrupt_budget: Some(20_000),
        ..SandboxOptions::default()
    });
    let result = run(&sandbox, "while (true) {}").await;
    assert_eq!(error(&result).kind, ErrorKind::Timeout);
    assert!(error(&result).message.contains("interrupt budget"));
}

#[tokio::test]
async fn aborts_cancel_in_flight_calls() {
    let cancelled = Arc::new(AtomicBool::new(false));
    let flag = cancelled.clone();
    let (called, mut wait_called) = tokio::sync::mpsc::unbounded_channel();
    let hang = Arc::new(FnCodemodeTool::new("hang", move |_, cancel: CancellationToken| {
        let flag = flag.clone();
        let _ = called.send(());
        async move {
            cancel.cancelled().await;
            flag.store(true, Ordering::SeqCst);
            Err("cancelled".to_owned())
        }
    }));
    let sandbox = sandbox(vec![hang]);
    let already = CancellationToken::new();
    already.cancel();
    let result = sandbox
        .execute(
            "await tools.hang()",
            ExecuteOptions {
                cancel: Some(already),
                ..ExecuteOptions::default()
            },
        )
        .await;
    assert_eq!(error(&result).kind, ErrorKind::Aborted);

    let cancel = CancellationToken::new();
    let pending = sandbox.execute(
        "await tools.hang(); return 'never'",
        ExecuteOptions {
            cancel: Some(cancel.clone()),
            ..ExecuteOptions::default()
        },
    );
    let abort = async {
        wait_called.recv().await;
        cancel.cancel();
    };
    let (aborted, ()) = tokio::join!(pending, abort);
    assert_eq!(error(&aborted).kind, ErrorKind::Aborted);
    assert_eq!(aborted.calls[0].status, CallStatus::Cancelled);
    tokio::time::timeout(Duration::from_secs(5), async {
        while !cancelled.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn output_past_the_limits_fails_the_script() {
    let sandbox = sandbox(Vec::new());
    for print in ["text(s)", "console.log(s)", "image(\"data:image/png;base64,\" + p)"] {
        let result = run(
            &sandbox,
            &format!(
                "const s = \"x\".repeat(1 << 20);\nconst p = \"iVBORw0KGgoA\" + \"A\".repeat(1 << 20);\nfor (;;) {{ try {{ {print}; }} catch {{}} }}"
            ),
        )
        .await;
        let failure = error(&result);
        assert_eq!(
            (failure.kind, failure.name.as_deref()),
            (ErrorKind::Script, Some("RangeError")),
            "{print}"
        );
        assert!(failure.message.contains("script output exceeded"));
        let chars: usize = result
            .output
            .iter()
            .map(|item| match item {
                OutputItem::Text { text, .. } => text.len(),
                OutputItem::Image { data, .. } => data.len(),
            })
            .sum();
        assert!(
            chars <= MAX_OUTPUT_CHARS && chars > MAX_OUTPUT_CHARS - (2 << 20),
            "{chars}"
        );
    }
    let empty = run(&sandbox, "for (;;) text(\"\");").await;
    assert_eq!(error(&empty).name.as_deref(), Some("RangeError"));
    assert_eq!(empty.output.len(), MAX_OUTPUT_ITEMS);
}

#[tokio::test]
async fn runs_do_not_share_state_and_recursion_is_catchable() {
    let sandbox = sandbox(Vec::new());
    let (a, b, c) = tokio::join!(
        run(
            &sandbox,
            "globalThis.shared = 'a'; await null; return globalThis.shared"
        ),
        run(
            &sandbox,
            "globalThis.shared = 'b'; await null; return globalThis.shared"
        ),
        run(&sandbox, "return typeof globalThis.shared"),
    );
    assert_eq!([value(&a), value(&b), value(&c)], ["a", "b", "undefined"]);
    let deep = run(
        &sandbox,
        "let depth = 0;\nfunction dive() { depth++; dive(); }\ntry { dive(); } catch (error) { return [error.name, depth > 1000]; }",
    )
    .await;
    assert_eq!(value(&deep), &json!(["RangeError", true]));
    let memory = Sandbox::new(SandboxOptions {
        memory_limit: Some(32 * 1024 * 1024),
        ..SandboxOptions::default()
    });
    let result = run(
        &memory,
        "const chunks = []; try { for (;;) chunks.push('x'.repeat(1 << 20) + chunks.length); } catch (error) { return String(error); }",
    )
    .await;
    assert!(
        value(&result).as_str().unwrap_or_default().contains("out of memory"),
        "{result:?}"
    );
}

#[tokio::test]
async fn there_are_no_escape_hatches() {
    let sandbox = sandbox(vec![echo()]);
    let globals = run(
        &sandbox,
        "return [typeof process, typeof require, typeof module, typeof setTimeout, typeof fetch, typeof WebAssembly, typeof std, typeof os, typeof globalThis.constructor]",
    )
    .await;
    assert_eq!(
        value(&globals),
        &json!([
            "undefined",
            "undefined",
            "undefined",
            "undefined",
            "undefined",
            "undefined",
            "undefined",
            "undefined",
            "function"
        ])
    );
    let eval = run(
        &sandbox,
        "return [eval(\"typeof process\"), new Function(\"return typeof process\")(), tools.echo.constructor(\"return typeof require\")(), (async () => {}).constructor(\"return typeof setTimeout\")() instanceof Promise];",
    )
    .await;
    assert_eq!(value(&eval), &json!(["undefined", "undefined", "undefined", true]));
    let import = run(
        &sandbox,
        "try { await import(\"node:fs\"); return \"imported\"; } catch (error) { return error.constructor.name; }",
    )
    .await;
    assert_ne!(value(&import), "imported");
    let patched = run(
        &sandbox,
        r#"
        Array.prototype.toJSON = () => null;
        Object.prototype.toJSON = () => 5;
        Promise.prototype.then = () => {};
        Map.prototype.get = () => undefined;
        globalThis.JSON = { stringify: () => "x", parse: () => "x" };
        store("k", [1]);
        return [await tools.echo([2]), JSON.stringify({ a: 1 })];
    "#,
    )
    .await;
    assert_eq!(value(&patched), &json!([[2], "{\"a\":1}"]));
    assert_eq!(patched.store_writes.set.get("k"), Some(&json!([1])));
    let frozen = run(
        &sandbox,
        r#"
        return [
            Object.getPrototypeOf(function* () {}).prototype,
            Object.getPrototypeOf(async function () {}),
            Object.getPrototypeOf(Int8Array).prototype,
            Object.getPrototypeOf([][Symbol.iterator]()),
            Object.getPrototypeOf(Object.getPrototypeOf([][Symbol.iterator]())),
            Object.getPrototypeOf(new Map()[Symbol.iterator]()),
            Object.getPrototypeOf(/a/[Symbol.matchAll]("")),
        ].every((object) => Object.isFrozen(object));
    "#,
    )
    .await;
    assert_eq!(value(&frozen), true);
    let overrides = run(
        &sandbox,
        r#"
        const object = {};
        object.toString = () => "custom";
        function Legacy() {}
        Legacy.prototype = Object.create(Error.prototype);
        Legacy.prototype.constructor = Legacy;
        const bare = new Error();
        bare.message = "set later";
        class MyError extends Error {
            constructor(message) {
                super(message);
                this.name = "MyError";
            }
        }
        let patched = "silent";
        try { Error.prototype.name = "Patched"; } catch (error) { patched = error.constructor.name; }
        return [String(object), new Legacy().constructor === Legacy, bare.message, new MyError("x").name, Error.prototype.name, patched];
    "#,
    )
    .await;
    assert_eq!(
        value(&overrides),
        &json!(["custom", true, "set later", "MyError", "Error", "TypeError"])
    );
    let tools = run(
        &sandbox,
        "try { tools.echo = () => 'nope'; } catch {}\ntry { tools.extra = () => 'nope'; } catch {}\ntry { globalThis.tools = null; } catch {}\nreturn [\"extra\" in tools, await tools.echo('still')];",
    )
    .await;
    assert_eq!(value(&tools), &json!([false, "still"]));
}
