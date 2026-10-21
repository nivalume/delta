# UI protocol (`delta-protocol`)

The protocol a user interface speaks to a delta harness. It is a subset of Codex's
app-server v2 protocol (`codex-rs/app-server-protocol`), so a client written for Codex's
app-server works against `delta-app-server`, plus methods under `delta/` for what Codex has
no word for. pi has no UI protocol package; pi's RPC mode is the closest upstream piece, and
this protocol replaces it rather than ports it. The crate holds types only and does not
depend on the harness; `delta-app-server` maps them onto `Harness` and `Conversation`.

## Wire

JSON-RPC 2.0 without the `"jsonrpc"` field, one message per line (Codex's framing over
stdio). A field `"jsonrpc"` sent by a client is ignored. `RequestId` is a string or an
integer. Errors use JSON-RPC's codes: `-32700` parse, `-32600` invalid request (before
`initialize`, a stale `expectedTurnId`), `-32601` unknown method, `-32602` invalid params,
`-32603` server failure.

Each direction's messages are one enum tagged by `method`:

| Enum | JSON |
| --- | --- |
| `ClientRequest` | `{"method", "id", "params"}` |
| `ClientNotification` | `{"method"}` (`initialized`) |
| `ServerNotificationEnvelope` | `{"method", "params", "emittedAtMs"?}` |
| `ServerRequest` | `{"method", "id", "params"}` |

Responses are `{"id", "result"}` or `{"id", "error": {"code", "message", "data"?}}`; each
request variant's documentation names its result type. Field names are camelCase, with
the exceptions Codex has: `UserInput::Text.text_elements`, `PatchChangeKind::Update.move_path`
and the fields of `CommandExecutionApprovalDecision`'s struct variant
(`execpolicy_amendment`) are snake case. `AskForApproval` is kebab case (`untrusted`,
`on-request`, `never`); `ApprovalsReviewer` is `user` or `auto_review`.

`schema/v2.json` is the JSON Schema (draft 2020-12) of every message and result; a test
compares it with `json_schema()` and `DELTA_UPDATE_SCHEMA=1` rewrites it.

## Connection

The client sends `initialize` (`clientInfo`, `capabilities.experimentalApi`,
`capabilities.optOutNotificationMethods`) and then the `initialized` notification. Any other
request before `initialize` is answered with `-32600`. The result names the server
(`userAgent` `delta/<version>`), `codexHome` (the server's data directory) and the
platform. `optOutNotificationMethods` suppresses the named notification methods for the
connection.

## Concepts

| Codex | delta |
| --- | --- |
| thread | a conversation (`ConversationId`); `sessionId` is the same id |
| turn | a run: the user input that starts it, the generations and tool calls it settles, until the conversation yields. The turn id is the run's first input `SubmissionId` |
| item | a unit inside a turn, below |
| `thread/start` | `Harness::create_conversation` |
| `thread/resume` | `Harness::conversation`; pending approvals are sent again |
| `thread/fork` | `Conversation::fork` as of the turn's last entry |
| `thread/list` | `Harness::conversations` (newest first; the cursor is opaque) |
| `thread/read` | the conversation's record, and with `includeTurns` its turns rebuilt from entries and submission records |
| `turn/start` | `Conversation::submit` as a follow-up |
| `turn/steer` | `Conversation::submit` as steering, if `expectedTurnId` is the running turn |
| `turn/interrupt` | `Conversation::abort` of the running turn |
| `thread/status/changed` | `idle` when the conversation yields; `active` with `waitingOnApproval` while an approval is pending; `systemError` when a task failed |

Items and their ids:

| Item | Source | Id |
| --- | --- | --- |
| `userMessage` | a user entry | the submission id |
| `agentMessage` | a text block of a generation | `<generation task id>:<block index>` |
| `reasoning` | a thinking block | `<generation task id>:<block index>` |
| `commandExecution` | a `bash` call; output as `item/commandExecution/outputDelta` | the tool call id |
| `fileChange` | an `edit` or `write` call; the change as a unified diff | the tool call id |
| `dynamicToolCall` | any other tool call | the tool call id |
| `contextCompaction` | a compaction | the compaction task id |

Approvals are server requests. A pending approval of a `bash` call is
`item/commandExecution/requestApproval`; of `edit` or `write`,
`item/fileChange/requestApproval`; of any other tool, `delta/tool/requestApproval`. Each
follows an `item/started` for the call, so the item exists before its approval. Decisions
map onto `ReviewDecision`:

| Decision | `ReviewDecision` |
| --- | --- |
| `accept` | `Approved` |
| `acceptForSession` | `ApprovedForSession` |
| `acceptWithExecpolicyAmendment` | `ApprovedWithRule`, the rule `Bash(<amendment joined by spaces>:*)` |
| `acceptWithRule` (delta) | `ApprovedWithRule` with the given rule |
| `decline` | `Denied` |
| `cancel` | `Abort` |

`proposedExecpolicyAmendment` is the policy's proposed rule when it is a `Bash(<prefix>:*)`
rule, split on spaces. When an approval is decided (by this client, another client or a
timeout) the server sends `serverRequest/resolved`.

## delta's methods

| Method | Kind | What |
| --- | --- | --- |
| `delta/permissions/read` | request | a thread's `delta.permissions`: mode, effective mode, rules, directories |
| `delta/permissions/update` | request | sets or clears the mode, adds and removes rules, replaces directories |
| `delta/permissionRules/read` | request | the harness-wide allow rules approvals added |
| `delta/permissionRules/remove` | request | removes one of them |
| `delta/tool/requestApproval` | server request | approval of a tool that is neither a command nor a file change |
| `delta/snapshot` | notification | the thread's whole `Snapshot`, sent when the client fell behind its events |

Permission modes are camelCase here (`default`, `acceptEdits`, `plan`, `dontAsk`, `bypass`)
and snake case in the harness's document.

## Differences from Codex

- A subset: no accounts, login, config, models, MCP, skills, plugins, filesystem, `command/exec`,
  realtime, review, or `thread/archive`, `thread/name/set`, `thread/rollback` and the other
  thread methods not listed above. Unknown methods are answered with `-32601`.
- Items: no `plan`, `webSearch`, `mcpToolCall`, `imageView`, `enteredReviewMode` or
  `collabAgentToolCall`. `commandActions` is always one `unknown` action. `reasoning.summary`
  is empty; the thinking text is in `content`.
- `approvalPolicy` and `sandbox` are reported, not obeyed: the harness's `ApprovalPolicy`
  decides, so a thread reports `on-request` and `dangerFullAccess`. Codex's
  `execpolicy_amendment` becomes a delta permission rule.
- A turn is a run. A `turn/start` while a run is in progress joins that run, so its
  `turn/started` names the running turn's id, where Codex would queue a separate turn.
- `delta/snapshot` replaces replay for a client that fell behind; Codex has no equivalent.
- `thread/list` ignores Codex's filters (`modelProviders`, `sourceKinds`, `archived`) and
  sorts newest first.
