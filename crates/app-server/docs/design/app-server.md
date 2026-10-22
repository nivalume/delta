# App server (`delta-app-server`)

Serves a `Harness` to UI clients over [delta-protocol](../../../protocol/docs/design/protocol.md),
the Codex app-server v2 subset. There is no upstream package: pi's RPC mode is the closest
piece, and Codex's `codex-rs/app-server` is the reference for behavior. v1 is a library;
a host binary passes it stdio or a socket.

## Connection

`AppServer::serve(reader, writer, ctx)` runs one connection: lines in, lines out, one writer
task fed by a queue so requests, notifications and server requests never interleave
within a line. Requests are handled one at a time, in order; a handler never waits for the
client, so answers to approval requests are read while turns run. The connection ends when
the reader reaches end of input, `ctx` is cancelled or the harness closes; a client that
closed its side first is not an error. Ending it stops the connection's subscriptions;
pending approvals stay pending in the harness.

Before `initialize` every request is refused with `-32600`. `initialize` records
`optOutNotificationMethods`; the `initialized` notification and any other notification
from the client are accepted and ignored.

## Threads

A connection follows the threads it starts, resumes or forks, and the threads it starts or
steers turns on: one subscription per thread on `Conversation::events`, with its own
translator. Following attaches to the events before the request's work, so nothing falls
between the response and the first notification; the subscription starts after the
response is written. `thread/started` follows the response of `thread/start` and
`thread/fork`.

| Request | Harness |
| --- | --- |
| `thread/start` | `create_conversation` (ownerless) with the model (`AppServerOptions.model` when absent), the cwd (`AppServerOptions.cwd` when absent) and `baseInstructions` + `developerInstructions` as `instructions` |
| `thread/resume` | the conversation, its model and cwd overrides by `configure`; the response carries its turns; a thread the connection already follows gets its pending approval requests again, under their ids |
| `thread/fork` | `fork` at the answer of `lastTurnId`, or at the last visible entry |
| `thread/list` | `Harness::conversations`, newest first; the cursor is the storage cursor as JSON text; `status` is `notLoaded` for threads the connection does not follow |
| `thread/read` | the record, and with `includeTurns` the turns rebuilt from entries and submissions |
| `turn/start` | `submit` as a follow-up; the turn id is the submission id |
| `turn/steer` | `submit` as a steer, when `expectedTurnId` is the running turn (`-32600` otherwise) |
| `turn/interrupt` | `abort` when the turn runs; a turn not started yet has its queued input withdrawn |

A model is `<provider>/<model id>`, or a model id of `AppServerOptions.model`'s provider.
A thread's `createdAt` and `updatedAt` are the timestamps of its first and last entries'
messages; `preview` is the first user message, 200 characters at most.

## Translation

The translator turns each commit's batch of agent events into notifications, in order:

- **Turns.** A turn opens at the entry of an input placed while no turn is open
  (`turn/started`). Inputs placed before the first input's answer (steering) join it. It
  closes at the end of the batch that holds its first input's answer, or that settles the
  first input otherwise (`turn/completed`: `completed`; `interrupted` for an `aborted`
  input; `failed` with the reason for any other unanswered input). A follow-up placed after
  the answer opens the next turn, so turns and inputs line up the same way live and in
  `thread/read`. An input from this connection's `turn/start` that is withdrawn before it
  is placed starts and completes its turn at once, so the client is not left waiting.
- **Messages.** The streamed partial (`message_start` with its generation task, then
  `message_update`) starts an item per text or thinking block and sends what each block
  grew by as `item/agentMessage/delta` or `item/reasoning/textDelta`; the committed answer
  sends the tail the partial did not carry, then `item/completed` with the whole text. A
  block that was never streamed starts and completes at once.
- **Tool calls.** `tool_execution_start` starts the call's item; appended output of `bash`
  is `item/commandExecution/outputDelta`, and the result's output past what was streamed is
  sent before `item/completed`. A call that ends without having started (declined, blocked,
  not offered) starts and completes at once. Nested calls are not items.
- **Approvals.** `approval_update` is diffed against the requests the client has: a new
  request starts its call's item and becomes a server request with a connection-wide
  integer id; a request no longer pending (decided here, by another client, by timeout)
  sends `serverRequest/resolved`. The client's answer is mapped to a `ReviewDecision` and
  passed to `Conversation::decide`; an error answer leaves the approval pending.
- **Others.** `usage_changed` sends `thread/tokenUsage/updated` (the sum of the model
  buckets, and the last answer's usage); `auto_retry_start` sends `error` with
  `willRetry: true`; `task_failed` sends `error`; compactions are `contextCompaction`
  items. After each batch the thread's status (`idle`, `active`, `active` with
  `waitingOnApproval`) is sent when it changed.
- **Lag.** When the subscription fell behind, the harness sends one snapshot: the server
  sends `delta/snapshot` with it, then reconciles: the partial's missing text, calls that
  started or ended, approvals that appeared or resolved, and the turn that ended.

`thread/read` replays the visible entries through the same translator (with the
submissions of the conversation and its fork ancestors, and each entry's message time as
the clock), so its turns and item ids are those the live notifications used.

## Differences from Codex's app-server

- One connection per `serve` call; no websocket transport, no process management.
- Approvals come from the harness's `ApprovalPolicy` (for example `RulesPolicy`), not from
  `approvalPolicy` and `sandbox`, which are reported only. `acceptWithExecpolicyAmendment`
  stores `Bash(<command prefix>:*)` as an allow rule for every thread.
- Approvals of tools other than `bash`, `edit` and `write` are `delta/tool/requestApproval`.
- `contextCompaction` items in `thread/read` are named by the task that appended the
  compaction entry, which can differ from the compaction task live notifications name.
- A thread's `turns` in `thread/start`, `thread/resume` and `thread/fork` responses carry
  items; `turn/started` and `turn/completed` carry none.
