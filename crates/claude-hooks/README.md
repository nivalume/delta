# delta-claude-hooks

Claude Code's [hooks](https://code.claude.com/docs/en/hooks) for
[delta-harness](../harness): the `hooks` object of a Claude Code settings file, run from the
harness's hook seams. Command handlers run through the conversation's execution
environment with the event's JSON on standard input; HTTP handlers receive it as a POST
body. The design notes are in [`docs/design/hooks.md`](docs/design/hooks.md).

```rust,no_run
# fn example(registry: &delta_harness::registry::Registry, options: &mut delta_harness::harness::HarnessOptions) -> Result<(), Box<dyn std::error::Error>> {
use std::sync::Arc;

use delta_claude_hooks::{ClaudeHooks, ClaudeHooksOptions, HooksConfig};
use delta_harness::approval::AskAll;

let settings: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(".claude/settings.json")?)?;
let hooks = ClaudeHooks::new(HooksConfig::from_settings(&settings)?, ClaudeHooksOptions::default());
registry.install(hooks.extension())?;
// PermissionRequest handlers answer what the policy would ask.
options.approval = hooks.permission_policy(Arc::new(AskAll));
# Ok(())
# }
```

| Event | Harness seam | What the handlers can do |
| --- | --- | --- |
| `PreToolUse` | `ToolHooks::before_tool` | block (exit 2, `deny`, `decision: "block"`), ask or allow (`permissionDecision`), replace the input (`updatedInput`), add context |
| `PermissionRequest` | `ClaudeHooks::permission_policy` | allow or deny what the policy would ask |
| `PostToolUse`, `PostToolUseFailure` | `ToolHooks::after_tool` | replace the output (`updatedToolOutput`), add a reason or context |
| `UserPromptSubmit` | `GenerationHooks::on_input` | block the run, add context |
| `Stop`, `SubagentStop` | `GenerationHooks::on_yield` | continue the run with a reason |
| `PreCompact` | `CompactionHooks::before_compact` | decline the compaction |

Other events (`SessionStart`, `SessionEnd`, `Notification`, `PostToolBatch`, ...) are
listed by `HooksConfig::unsupported` and not run.
