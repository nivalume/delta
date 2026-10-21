# Sandbox (`delta-codemode`)

Runs model-written JavaScript in a QuickJS VM whose only capability is calling the functions
the host injects. It is the port of `@earendil-works/pi-codemode` (`src/runtime/*`, `types.ts`).
The crate has no harness dependency; the harness's durable `codemode` tool
([harness codemode.md](../../../harness/docs/design/codemode.md)) builds on `Sandbox::start`.

## Script environment

`code` is the body of an async function: `return` and top-level `await` work. Inside it:

- `tools.<name>(args)` returns a promise. Arguments and results make a JSON round trip; a failed
  call rejects with an `Error` carrying the message and the outcome's extra properties. Tool
  names are also identifiers (`to_codemode_identifier`: invalid characters become `_`), so
  `my-tool` is `tools.my_tool` as well as `tools["my-tool"]`. Reading a member that does not
  exist throws a `TypeError` naming close matches.
- `ALL_TOOLS` lists `{ name, description }` for every tool.
- `text(value)` and `console.log/info/warn/error/debug` append text items (console items are
  marked `console`); `image(urlOrItem)` appends an image from a base64 `data:` URL,
  `{ image_url }` or an image block, checked against the PNG, JPEG, GIF and WebP signatures.
- `exit()` ends the script successfully at once, keeping output and store writes.
- Globals are top-level functions (`a.b` names group into a frozen `a` object); `spread` globals
  receive the whole argument list.
- `store(key, value)` and `load(key)` work synchronously on a snapshot of JSON values; a
  successful result reports the writes (`set`, `delete`). One value is at most 256 Ki characters
  of JSON, all of them at most 1 Mi.
- Nothing else: no timers, `fetch`, modules or `WebAssembly`; `eval` only makes more code in
  the same VM. Output is limited to 16 Mi characters and 100,000 items.

The prelude is upstream's `PRELUDE_SOURCE`, evaluated unchanged: it freezes the built-ins
(with accessors for the commonly overridden properties), keeps the host bridge in a closure,
and builds the environment above. A script waiting on a promise nothing can settle (no call
pending) fails at once.

## Running a script

A dedicated OS thread owns each run's `rquickjs::Runtime` and `Context`, so a computing script
never blocks the async host. The host and the VM thread exchange the messages of upstream's
`protocol.ts`:

- VM to host: `Call { id, target: tool | global, name, args }`, `Output(item)`,
  `Done { ok, value | error, writes, during }`, `Crash { message }`, and in a deterministic run
  `Started { fingerprint }` and `Idle { seq, issued }`.
- Host to VM: `Result { id, ok, payload, seq, now, properties }`, and `Stop`.

Promise jobs run one at a time, with an interrupt poll every 50 jobs, so `while (true) await
null` stays interruptible. The interrupt handler fires when the host sets the run's stop flag
(deadline, abort, finish) or, if one is set, when the interrupt budget is spent.

Limits: `memory_limit` (allocations beyond it fail inside the script as `InternalError: out of
memory`), QuickJS's stack guard at 4 MiB on a 16 MiB thread (deep recursion is a catchable
`RangeError`), the deadline (`timeout`, default 300 s for `execute`, none for `start`) and the
optional interrupt budget.

`Sandbox::execute(code, options)` runs the host's own tools: each call runs concurrently on the
host, and a call still pending when the script ends is cancelled through its
`CancellationToken` and reported `cancelled`. It never fails for script errors; the result's
`error.kind` is `script`, `timeout`, `aborted` or `sandbox`.

## Deterministic runs

`Sandbox::start(code, options)` runs a script so a host can run it again after a restart and get
the same behavior. The run declares its tools and globals; every call goes to the host as a
`RunEvent::Call`, answered with `Run::deliver(seq, now, id, outcome)`. The same script,
`started_at`, seed, tools (with their `js_name`), globals and store, answered with the same
results in the same order, makes the same calls with the same arguments and the same output on
any VM with the same fingerprint:

- Time: `Date.now()`, `new Date()`, `Date()` and `performance.now()` read a virtual clock:
  `started_at`, then the `now` of the result being processed, never going backwards.
- Randomness: `Math.random()` is mulberry32 over the seed.
- Timezone: the local-time `Date` methods, `getTimezoneOffset` and the string forms behave as
  UTC; a `Date` built from components reads them as UTC.
- `WeakRef` and `FinalizationRegistry` do not exist.
- Results are applied strictly in `seq` order (1, 2, …); an early one waits for its turn.

The VM reports `Started` with its fingerprint before the script runs and `Idle(seq, issued)`
after processing each result (0: the first run), so a host can check a replay step by step.
`Done` carries `during`: the result being processed when the script ended and the calls issued
by then.

## Differences from upstream

- A native QuickJS (`rquickjs`, QuickJS-NG) on its own thread instead of QuickJS compiled to
  WebAssembly in a worker thread. Upstream fixes time and randomness at the WASI level and sets
  the timezone through the engine; here the prelude replaces `Date`, `Math.random` and
  `performance` before freezing the built-ins. The fingerprint names the rquickjs and QuickJS
  versions, a hash of both preludes, the deterministic version and the limits.
- One sandbox: no `InlineCodemodeSandbox`, `RemoteCodemodeSandbox`, Cloudflare Durable Object
  or worker lifetime hooks (`acquireWorker`, `onWorkerStarted`). The thread is always
  interruptible, so the interrupt budget is optional.
- Tools implement the `CodemodeTool` trait; cancellation is a `CancellationToken` instead of an
  `AbortSignal`.
