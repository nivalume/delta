# Code mode (`codemode` feature)

The `codemode` tool runs a model-written script that calls the conversation's callable tools,
and survives a restart. Port of `@earendil-works/pi-durable/codemode` (spec §7.6) on
[`delta-codemode`](../../../codemode/docs/design/sandbox.md). It is built only on the public
seams: a `HarnessTool`, nested calls (`ToolApi::execute_tool`), task-scoped documents and a
prompt section. It is behind the `codemode` feature; `codemode_extension(options)` installs it.

## Problem

A script is in-memory VM state. A restart while

```js
const results = await Promise.all(paths.map((path) => tools.read({ path })));
```

waits for the last of 200 reads would lose the script and every read it made.

## Solution: deterministic replay

The VM only runs when the host hands it a result. Given the same start data, clock, seed and
results in the same order, it makes the same calls with the same arguments
(`Sandbox::start`). The tool is `ToolReplay::Safe`: after a restart `execute` runs again,
starts the script again from the stored start data, and hands it the results it got before, in
the order it got them, from a journal written before each delivery. Nested tool calls reattach
to their tasks by key (`c<VM call id>`), so they never run twice unless their own replay policy
reruns them; their results come from `NestedCallDoc`. Then the script continues live.

## The tool

`codemode`: `callers MODEL` (scripts cannot start scripts), `replay Safe`, parameters
`{ code: string }`, `output_limits` 4 MiB and 100,000 lines, constrained sampling with
`CODEMODE_SOURCE_GRAMMAR` (applied by `delta_ai` for models that support grammar tools). Its
description is fixed: the script environment and its globals. The `codemode` prompt section
lists the callable tools (below).

A tool resolves in the script as in the coding agent: one that declares a
`structured_output_schema` resolves with its `structured_output`, also on an error result that
carries one (`bash` with a nonzero exit resolves with `{ output, exitCode, … }`); a schema-less
tool resolves with its text output, images left out. Every other error rejects, with the
schema-less tool's text as message when it is not empty, else its error diagnostics' messages,
and the value, if any, as `error.structuredOutput`. Which tools declare a schema is recorded in
the start data, so a resumed script maps results as it began.

The result's content is a `Script completed` / `Script failed` header with the wall time since
`started_at`, the text items (`==> text N/M <==` when there are several), the `console` lines
in one `<console_output>` block, the returned value, and on failure the error with the calls
already made. Text past `max_output_tokens` (default 10,000; the options line may set it) keeps
its start and end, with the complete text in `output.txt`; images are saved to
`image-<n>.<ext>` with their paths next to them. The files go to a temp directory of the
environment that the run creates once and records (`output_dir`) before writing to it, so a
rerun overwrites them. `details` lists the script's tool calls in call order
(`CodemodeToolDetails { calls: [{ id, name, args (cut to 200), status ok | error | cancelled,
duration_ms, error (cut to 500) }], full_output_path? }`); a call still running when the script
ended is `cancelled`.

## Documents

| Kind | Scope | Content |
| --- | --- | --- |
| `delta.codemode.run` | task | `start`, written once before the VM starts: module version, `started_at`, seed, the VM's fingerprint once reported, the `deadline` from `timeout_ms`, the tools (`name`, `js_name`, `description`, `structured`, `hidden`), namespaces and globals the script sees, `max_output_tokens`, the store snapshot; `output_dir`; `result`, the outcome, written once |
| `delta.codemode.journal` | task | `entries`, one per result delivered to the VM, in delivery order: the call id, the clock value `at`, `issued_after` (the calls the VM had issued after processing it, from its acknowledgment), and for a global its name, arguments and outcome; `first_issued`, the calls issued after the script's first run |
| `delta.codemode.store` | conversation, rewindable, fork as of | `values`: what scripts keep with `store()`; a successful script's writes are applied per key in the commit that writes its `result` |

## Live

A finished call's result is appended to the journal and then delivered as result number
`entry index + 1`, with the clock at the entry's time; one journal commit is in flight at a
time and later results join the next. Acknowledged counts (`Idle(seq, issued)`) are recorded
with the next journal commit, or at once when the VM has processed everything delivered. A call
is launched only once the count of the drain that made it is committed, so every key an attempt
admitted is at most its last recorded count.

A run the sandbox ends as `aborted` without the call having asked for it decides nothing: the
call stores no outcome and ends with its invocation, so a later attempt replays the script.

## Rerun

A rerun returns a stored `result` without a VM. Otherwise it ends the call with a stored error
result, admitting no new nested call, when the module version differs or the deadline passed; it
starts the VM from `start` and does the same when the fingerprint the VM reports differs, before
delivering or reattaching anything. It then delivers the journal in order, each entry once the VM
has issued its call; a tool's result comes from its nested call by key, a global's from the
journal after comparing name and arguments. Calls the journal does not have are held until the
last journaled result is acknowledged.

Divergence ends the call with a stored error result, then stops the VM with an error the script
cannot catch: a count that differs from a recorded `issued_after` or `first_issued`, a VM that
waits without having issued the next entry's call, a key or global called with another name or
arguments, a script that ends while processing an entry whose `issued_after` was recorded (or
during its first run with a non-empty journal), or before issuing a call that a later entry
answers. On a rerun, live calls are held until the acknowledgment of the last journaled result,
or of the first run when the journal is empty; every held call's key is checked against its
stored nested call, and only when none differs are they launched; later calls are new. Nested
calls admitted before the restart may resume on their own whatever the rerun decides; the
call's cleanup aborts them.

## Discovery

The `codemode` prompt section lists `agent.callable` (without internal or deferred tools) as
TypeScript declarations (`render_tool_sample`), grouped by namespace: a tool belongs to the
`CodemodeNamespace { name, description, instructions, prefix }` with the longest matching
prefix. Within an inline budget (default 3,000 estimated tokens at 4 characters each), every
group places its cheapest remaining tool in turn, ungrouped tools first, then namespaces by name;
what does not fit is said so and found with `searchTools()`.

Scripts find tools with `ALL_TOOLS`, `searchTools(query, { limit?, namespace? })` (BM25 over
name, description and namespace, default limit 8), `describeTool(name)` and
`describeNamespace(name)`, answered from the start data and journaled like other globals.

## Models

With `models` (default on), scripts get the model catalog (`models.getModelsOfType`,
`getAvailableOfType`, `getModelOfType`: journaled globals, configured headers removed) and
`models.classify()` and `models.generateImages()`. Those two are paid effects, so each runs as an
internal tool of the extension, `delta.codemode.models.classify` and
`delta.codemode.models.generateImages` (`callers TOOLS`, with a schema): the script's call of the
global becomes a nested call of the tool, which has its own task, live row, abort, usage and
replay policy. Tools named `delta.codemode.*` are recorded in the start data as `hidden`: never
in the section, `tools`, `ALL_TOOLS` or search. At most four such calls of one script run at
once, and a call waiting for a slot when the script has ended is not started. A script that
received generated images and showed none gets a note in its result.

## Differences from upstream

- Document kinds are `delta.codemode.*`; internal tools are `delta.codemode.models.*`.
- The sandbox is a native QuickJS thread ([sandbox.md](../../../codemode/docs/design/sandbox.md)),
  so the fingerprint differs from upstream's and a run started under pi does not resume here.
- The VM's call ids reach the host through a channel, so the host side is a single task that
  owns the run's state instead of callbacks.
