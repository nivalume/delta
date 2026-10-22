# Contexts (`context`)

Corresponds to upstream `src/context/index.ts`.

## Problem

Durable passes one value through every operation of a run: the turn, each tool call, each
storage commit. It carries cancellation (with a reason: a user abort, a shutdown, a
superseded run), and values scoped to the invocation. Upstream makes this an immutable
`Context` passed explicitly, never ambient; deriving a context never changes its parent.

`delta_ai` and `delta_agent` already use `tokio_util::sync::CancellationToken` and treat
dropping a future as cancelling it. delta's `Context` builds on both.

## `Context`

```rust
#[derive(Clone)]
pub struct Context { /* Arc to an immutable node: parent, cancellation, value */ }

impl Context {
    pub fn background() -> Context;                       // never cancelled, no values

    // Values
    pub fn with_value<T: Send + Sync + 'static>(&self, key: &'static ContextKey<T>, value: T) -> Context;
    pub fn value<T: Send + Sync + 'static>(&self, key: &'static ContextKey<T>) -> Option<&T>;

    // Cancellation
    pub fn with_cancel(&self) -> (Context, Canceller);    // cancelled by the parent or by the canceller
    pub fn with_token(&self, token: CancellationToken) -> Context;   // by the parent or by an outside token
    pub fn without_cancel(&self) -> Context;              // same values, never cancelled: cleanup only
    pub fn is_cancelled(&self) -> bool;
    pub fn reason(&self) -> Option<CancelReason>;
    pub async fn cancelled(&self);
    pub async fn run<F: Future>(&self, future: F) -> Result<F::Output, Cancelled>;
    pub fn token(&self) -> CancellationToken;             // for delta_ai options and delta_agent
}

pub struct ContextKey<T> { /* name */ }
impl<T> ContextKey<T> { pub const fn new(name: &'static str) -> Self; }

pub struct Canceller;
impl Canceller {
    pub fn cancel(&self);                          // reason: Aborted
    pub fn cancel_with(&self, reason: CancelReason);
}

pub struct CancelReason(Arc<dyn Error + Send + Sync>);   // new(error), message(text), aborted()
```

### Values

- A key is a `static`: `static USER: ContextKey<String> = ContextKey::new("user");`. Keys
  compare by address, so two keys of the same type are different keys, as upstream's
  symbol tokens.
- Lookup walks the chain of derived contexts (O(depth), as upstream).

### Cancellation

- `with_cancel` is the common case: the new context's token is a child of the parent's
  token, so cancelling the parent cancels it, and no task is needed.
- `with_token` joins an outside token that is not a child of the parent's (upstream's
  `withAbortSignal(signal, context)`, which uses `AbortSignal.any`). `is_cancelled`,
  `cancelled` and `run` check both. `token()` on such a context, and `with_cancel` below it,
  need one token for both: a small task joins them, so they need a tokio runtime. The task
  ends when the last context that shares it is dropped.
- `reason()` is the reason given to the `Canceller` closest to this context that fired, or
  a generic "aborted" reason when cancellation came from a bare `CancellationToken`.
- `without_cancel` keeps the values and drops every cancellation source, for work that
  must finish (rollback, cleanup), as upstream.

### Waiting

`ctx.run(future)` resolves with the future's output, or with `Cancelled` (carrying the
reason) when the context is cancelled first. On cancellation the future is dropped, which
cancels it, as everywhere in delta. Upstream's `awaitWithContext` only stops waiting and
lets the promise run on, because a JS promise cannot be cancelled; a caller that wants
that in delta spawns the future and runs the `JoinHandle`.

## Telemetry

Upstream's `Context` always carries a telemetry span (`withTelemetry`, `traceIn`). delta
has no telemetry crate yet (`delta_ai` waits for it too), so `Context` has no span. When
delta's telemetry is designed, the span becomes a context member there. Code that wants
tracing meanwhile uses `tracing` spans directly.

## Differences from upstream

- `CancellationToken` plus a reason instead of `AbortSignal`.
- `run` cancels the awaited future on cancellation instead of only abandoning it.
- No `TODO_CONTEXT`.
- No telemetry member yet.
