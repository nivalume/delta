# Claude Code hooks (`delta-claude-hooks`)

A bridge from Claude Code's hook protocol to delta-harness's hook seams
([harness.md](../../../harness/docs/design/harness.md), "Hooks"). It is an ordinary
`Extension` plus an approval policy decorator, built only on the harness's public traits.
There is no upstream package; Claude Code's documentation is the reference.

## Configuration

`HooksConfig::from_value` reads the `hooks` object of a settings file: event names mapped
to matcher groups `{ matcher, hooks: [handler] }`. A handler is
`{ "type": "command", "command", "timeout"? }` or
`{ "type": "http", "url", "headers"?, "allowedEnvVars"?, "timeout"? }` (seconds; 600 by
default, 30 for `UserPromptSubmit`). A matcher that is absent, empty or `*` matches all; one
of letters, digits, `_` and `|` lists exact names; anything else is an unanchored regular
expression. Tool events match the tool's Claude Code name (`bash` is `Bash`, `read` `Read`,
`write` `Write`, `edit` `Edit`) or its own; `PreCompact` matches `manual` or `auto`; other
events run every group. Every matching handler runs, in parallel.

## Running a handler

The input is the event's JSON: `session_id` (the conversation id), `transcript_path`
(`ClaudeHooksOptions.transcript_path`, empty by default), `cwd`, `permission_mode` (the
conversation's `delta.permissions` mode in Claude Code's names), `hook_event_name`, and the
event's fields. Tool inputs are the call's arguments, with `path` also given as `file_path`;
an `updatedInput` with `file_path` is mapped back to `path`.

- A command runs through `HookApi::env()` (`ExecOptions.stdin` carries the input,
  `CLAUDE_PROJECT_DIR` is the environment's directory). Exit 0: stdout is a JSON output, or
  plain text; exit 2: blocks with stderr; any other exit is a non-blocking error.
- An HTTP handler is a POST of the input. A 2xx response with an empty body or a JSON
  object is an output; anything else is a non-blocking error. Header values take `$VAR` /
  `${VAR}` only from `allowedEnvVars`.

Non-blocking errors go to `ClaudeHooksOptions.on_error`. The outcomes of an event are
memoized in the hook's task (`claude-hooks:<event>`): a phase that runs again after a
restart reuses them instead of running the handlers again.

## Events

| Event | Seam | Mapping |
| --- | --- | --- |
| `PreToolUse` | `before_tool` | exit 2, `decision: "block"`, `continue: false` or `permissionDecision: "deny"` block the call (reason: stderr, `reason`, `stopReason` or `permissionDecisionReason`); otherwise the strongest of `defer` > `ask` > `allow` across handlers becomes `BeforeTool.permission` (`defer` is no decision); the first `updatedInput` replaces the arguments; every `additionalContext` follows the result |
| `PermissionRequest` | `permission_policy(inner)` | runs only when `inner` asks; exit 2 or `decision.behavior: "deny"` denies (`message`), `"allow"` allows; anything else keeps the request; `updatedInput` is reported and ignored |
| `PostToolUse` / `PostToolUseFailure` | `after_tool` (by `is_error`) | `updatedToolOutput` replaces the output; a block's reason, exit 2's stderr and `additionalContext` are appended as one text item |
| `UserPromptSubmit` | `on_input` | `prompt` is the run's user text; exit 2 or a block ends the run `unanswered` `blocked`; plain stdout and `additionalContext` become a `delta.context` entry |
| `Stop` / `SubagentStop` | `on_yield` | `SubagentStop` when the conversation has an owner; `stop_hook_active` is `continuations > 0`; exit 2 or `decision: "block"` continues with the reason (the harness caps continuations at `max_continuations`) |
| `PreCompact` | `before_compact` | `trigger` and `custom_instructions`; exit 2 or a block declines the compaction |

## Differences from Claude Code

- Unsupported events: `SessionStart`, `SessionEnd`, `Notification`, `PostToolBatch`,
  `PreModelSwitch` and the others with no harness seam. `HooksConfig::unsupported` names them.
- Handler types `mcp_tool`, `prompt` and `agent`, `async` handlers and `onFailure` are not
  supported; `if` conditions are not evaluated.
- `continue: false` blocks what the event can block instead of stopping the whole agent.
- `systemMessage` and `suppressOutput` have no effect.
- Tool inputs keep delta's argument names (plus `file_path`), not Claude Code's tool
  schemas.
