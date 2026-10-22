# Permission rules and modes (`approval::rules`, `approval::permissions`)

The rules engine over the approval seam ([approval.md](approval.md)): Claude Code's
`Tool(specifier)` rules and permission modes, durable per conversation, with "always
allow" rules that an approval adds harness-wide in the deciding commit (Codex's execpolicy
amendment). Everything here is an ordinary `ApprovalPolicy`; the tool task does not change.

## Problem

`AllowAll`, `DenyAll` and `AskAll` are all-or-nothing. A coding agent needs to run
`cargo test` without asking, never read `.env`, ask before `git push`, let a user switch a
conversation to "accept edits" or "plan" for a while, and turn one approval into a rule so
the same command is not asked again. Those choices must survive a restart, follow a fork
as of its fork point, and reach the conversations a task creates (sub-agents).

## Records

```rust
#[serde(rename_all = "snake_case")]
pub enum PermissionMode { Default, AcceptEdits, Plan, DontAsk, Bypass }

pub struct PermissionRules { pub allow: Vec<String>, pub ask: Vec<String>, pub deny: Vec<String> }  // each defaults to []

pub struct PermissionsState {
    pub mode: Option<PermissionMode>,     // None: the policy's default mode
    pub rules: PermissionRules,
    pub directories: Vec<String>,         // more directories reads (and accept_edits writes) may reach
}
```

| Kind | Type | Scope | History / fork | Content |
| --- | --- | --- | --- | --- |
| `delta.permissions` | `PermissionsDoc` | conversation | rewindable, fork as of | `PermissionsState` |
| `delta.permission_rules` | `PermissionRulesDoc` | session | — | `PermissionRules` added by approvals (only `allow` is written) |

`delta.permissions` is created on first write. A conversation created with an owner (a
task-owned conversation, a sub-agent) starts with a copy of its owner conversation's
document, as `delta.agent` is copied. An absent document is the empty state.

`ReviewDecision::ApprovedWithRule { rule }` approves the call and, in the same commit as
the decision, appends `rule` to `delta.permission_rules.allow` (once). `ApprovalRequest`
and `PolicyDecision::Ask` gain `proposed_rule: Option<String>`, the rule a UI offers for
"always allow".

## The tool seam

```rust
pub enum PathAccess { Read, Write }
pub enum PermissionSubject { Command { command: String }, Path { path: String /* absolute */, access: PathAccess } }  // serde: tagged "type"

trait HarnessTool {
    /// What rules match this call on; `None`: only rules naming the tool.
    fn permission_subject(&self, arguments: &JsonObject, env: Option<&dyn ExecutionEnv>) -> Option<PermissionSubject> { None }
}
```

`bash` gives `Command { command }` (the model's command, before the command prefix and
`BashPrepare`); `read` gives a read path, `write` and `edit` a write path, made absolute
with the environment's `absolute_path` after the tools' path normalization. The tool task
builds the environment for the policy (an environment failure is reported and the call is
decided without one) and passes `subject` and `cwd` (the environment's, else the agent's)
in `ApprovalCall`.

## Rules

A rule is `Tool` or `Tool(specifier)`; `Tool(*)` is `Tool`.

- `Bash`, `Read`, `Edit` and `Write` (any case) name subjects, not tools: `Bash` matches
  a command subject, `Read` a read path, `Edit` and `Write` a write path. Any other name
  matches the tool of that name exactly, and takes no specifier.
- **Command specifiers.** `*` matches any text, spaces included. A trailing ` *` also
  matches the bare command (`Bash(ls *)` matches `ls` and `ls -la`, not `lsof`); a trailing
  `:*` means the same. Without `*` the command must be equal. A command is split into
  sub-commands at `&&`, `||`, `;`, `|`, `|&`, `&` and newlines outside quotes,
  `$(...)` and backticks; each is trimmed, leading `NAME=value` assignments and the
  wrappers `timeout`, `time`, `nice`, `nohup`, `stdbuf`, `command`, `builtin` (with their
  options) are removed. An allow rule set allows a command when every sub-command matches
  one of its rules; a deny or ask rule applies when any sub-command matches. A command that
  does not split (unbalanced quotes) is never allowed by a rule; deny and ask rules then
  match the whole text.
- **Path specifiers.** Anchors: `//path` is absolute, `~/path` is below the home
  directory, `/path` is below the policy's `root` (a rule using it without a root is
  invalid), `./path` and `path` are below the call's `cwd`. In the rest, `*` matches within
  one path segment, `**` any number of segments, `?` one character. A pattern matches a
  path equal to it or below it. Matching is lexical (no symbolic links resolved). On
  Windows, paths are matched in Claude Code's POSIX form (`rules::rule_path`): `C:\Users\me`
  is `/c/Users/me`, so a rule names it `//c/Users/me/**`; proposed rules use the same form.
- `Edit`/`Write` rules apply to write subjects and `Read` rules to read subjects; a `Read`
  deny or ask rule also applies to writes of the paths it names.

Rules are validated where they are written (`Conversation::update_permissions`,
`Conversation::decide`, `RulesPolicy::new`); a rule in a document that does not parse is
ignored.

## `RulesPolicy`

```rust
pub struct PermissionDefaults {
    pub mode: PermissionMode,
    pub rules: PermissionRules,
    pub directories: Vec<String>,
    pub root: Option<String>,       // the anchor of `/path` rules
    pub home: Option<String>,       // the anchor of `~/path` rules; `new` reads the process's home when unset
}
impl RulesPolicy { pub fn new(defaults: PermissionDefaults) -> Result<Self, RuleError> }
```

The host sets it as `HarnessOptions.approval`. For a call it reads the conversation's
`delta.permissions` and the session's `delta.permission_rules` through `HookApi` and
decides, in order:

1. a deny rule matches: `Deny { reason: "denied by rule <rule>" }`;
2. an ask rule matches: `Ask` (no proposed rule);
3. mode `bypass`: `Allow`;
4. mode `plan` and a command or write subject: `Deny { reason: "plan mode: <tool> may not run" }`;
5. an allow rule matches (the defaults', the session's, the conversation's): `Allow`;
6. a read inside `cwd` or a directory: `Allow`; in mode `accept_edits` also a write
   inside them;
7. otherwise mode `dont_ask`: `Deny { reason: "<tool> would need approval (dont_ask mode)" }`;
   any other mode: `Ask` with a proposed rule.

Proposed rules: a single-command `Bash(<word> <word>:*)` (one word when the second starts
with `-`), none for a compound command; `Edit(//<parent dir>/**)` for a write;
`Read(//<parent dir>/**)` for a read; the tool's name for a call without a subject.

The decision is memoized in the call's task (`delta.permissions:decision`), so a rerun of
the call phase after a crash decides the same way even if the documents changed.

## Surface and observation

- `Conversation::permissions(ctx) -> PermissionsState`,
  `Conversation::update_permissions(PermissionsChange { mode: Change<PermissionMode>, add: PermissionRules, remove: PermissionRules, directories: Change<Vec<String>> }, ctx)`,
  `Conversation::set_permission_mode(mode, ctx)`.
- `Harness::permission_rules(ctx) -> PermissionRules`,
  `Harness::remove_permission_rule(rule, ctx)`.
- `Conversation::decide(id, ApprovedWithRule { rule }, ctx)` refuses a rule that does not
  parse (`ApprovalError::InvalidRule`).
- `delta.permissions` is mounted in the conversation view; `Snapshot.permissions` and
  `AgentEvent::PermissionsChanged { permissions }` follow it.

## Differences from Claude Code

- Rule names are subject kinds (`Bash`, `Read`, `Edit`/`Write`) or exact tool names;
  `Tool(param:value)`, tool-name globs, `WebFetch(domain:...)` and MCP server rules are
  not supported.
- Relative path patterns are anchored at the call's directory; gitignore's "a pattern
  without a slash matches at any depth" is not applied (write `**/name`).
- No protected paths and no special casing of `rm` in modes; `plan` denies instead of
  asking.
- Rules from approvals are harness-wide (a session document), not written to a settings
  file.

## Differences from upstream

Upstream pi-durable has no rules or modes.
