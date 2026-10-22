# delta-codemode

Runs model-written JavaScript in a QuickJS VM whose only capability is calling the functions the
host injects. Nested tool calls never enter the model's context; only the script's output and
return value do. Port of [`@earendil-works/pi-codemode`](https://github.com/earendil-works/pi/tree/main/packages/codemode);
the design notes are in [`docs/design/sandbox.md`](docs/design/sandbox.md) and
[`docs/design/declarations.md`](docs/design/declarations.md).

```rust
# async fn example() {
use std::sync::Arc;

use delta_codemode::{ExecuteOptions, FnCodemodeTool, Sandbox, SandboxOptions};
use serde_json::json;

let read = FnCodemodeTool::new("read", |args, _cancel| async move {
    let path = args.and_then(|args| args["path"].as_str().map(str::to_owned)).unwrap_or_default();
    Ok(Some(json!(format!("contents of {path}"))))
});
let sandbox = Sandbox::new(SandboxOptions::default()).with_tool(Arc::new(read)).unwrap();
let result = sandbox
    .execute(
        r#"const source = await tools.read({ path: "package.json" });
        text("bytes " + source.length);
        return source.split(" ").at(-1);"#,
        ExecuteOptions::default(),
    )
    .await;
assert!(result.is_ok());
assert_eq!(result.value, Some(json!("package.json")));
# }
```

Scripts see `tools.<name>(args)`, `ALL_TOOLS`, `text`, `image`, `exit`, `console.*`, `store` /
`load` and the host's globals, and nothing else: no timers, `fetch`, modules or host objects.
Each run gets its own VM on its own thread, with a deadline, an optional memory limit and an
optional interrupt budget.

`Sandbox::start` runs a script deterministically: the host answers every call, results are
applied in the order the host numbers them, and the clock, randomness and timezone are fixed, so
the same answers in the same order make the same calls and output. `delta-harness`'s durable
`codemode` tool replays scripts after a restart this way.
