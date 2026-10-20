# Bounded output and paced progress (`output`)

Corresponds to upstream `src/output.ts`.

## Problem

Tools and processes produce output of any size: a build log, a `cat` of a large file, a
command that never stops. The agent keeps a bounded window of it (the first lines for some
tools, the last lines for others), counts what it dropped, and shows the window live while
the tool runs. Live display commits the window to a document, and committing on every
chunk would flood storage and observers. Upstream solves this with `OutputBuffer` (a head
or tail window at a cost proportional to each chunk) and `Progress` (commits paced by
their size).

## Limits and slices

```rust
pub struct OutputLimits { pub max_bytes: usize, pub max_lines: usize, pub retain: Retain }
pub enum Retain { Head, Tail }

pub struct OutputSlice<'a> { pub text: &'a str, pub dropped_bytes: usize, pub dropped_lines: usize }

pub fn bound_output<'a>(text: &'a str, limits: &OutputLimits) -> OutputSlice<'a>;
pub fn sanitize_output(text: &str) -> Cow<'_, str>;
pub fn char_end(bytes: &[u8], index: usize) -> usize;
```

- `bound_output` keeps whole lines within both limits: the first lines for `Head`, the last
  for `Tail`. A single line longer than `max_bytes` is cut at the byte limit on a character
  boundary. The result borrows the input; no copy, no re-encoding (upstream encodes the
  string to bytes and decodes the slice back).
- `sanitize_output` removes C0 control characters other than tab and newline, and
  U+FFF9–U+FFFB. It borrows when there is nothing to remove.
- `char_end(bytes, index)` is the last UTF-8 character boundary at or before `index`
  (clamped to the length) in a byte buffer that may not be valid UTF-8 (durable's `read`
  tool cuts raw lines with it).
  For `&str`, `str::floor_char_boundary` does the same.
- Line counting is as upstream: a final line without a newline counts as a line.

## `OutputBuffer`

```rust
pub struct OutputBuffer;

impl OutputBuffer {
    pub fn new(limits: OutputLimits) -> Self;
    pub fn push_str(&mut self, chunk: &str);
    pub fn push_bytes(&mut self, chunk: &[u8]);
    pub fn skip(&mut self, skipped: OutputSkip);        // tail only
    pub fn end(&mut self);
    pub fn snapshot(&mut self) -> BoundedOutput;
    pub fn stored_bytes(&self) -> usize;
}

pub struct OutputSkip { pub bytes: usize, pub newlines: usize, pub ends_with_newline: bool }
pub struct BoundedOutput { pub text: String, pub dropped_bytes: usize, pub dropped_lines: usize }
```

- Byte chunks are decoded as a stream: an incomplete character at the end of a chunk waits
  for the next one; invalid bytes become U+FFFD; a byte-order mark is dropped only at the
  very start of the stream. A string chunk, a skip or `end()` flushes an incomplete
  character as U+FFFD. (Upstream uses `TextDecoder` in stream mode; delta uses
  `<[u8]>::utf8_chunks`.)
- Totals over the whole stream (bytes, newlines, whether it ends with a newline) are kept, so
  `dropped_bytes` and `dropped_lines` are exact.
- **Head**: stores decoded text until it holds more than `max_bytes` bytes or at least
  `max_lines` newlines, then only counts. It keeps at most the first `max_bytes + 1` bytes:
  nothing past them can be in the window, and the extra byte tells whether a cut at
  `max_bytes` falls inside a character.
- **Tail**: stores the last `max_bytes + 1` bytes in a ring buffer. A window of the last
  lines never spans more than `max_bytes` bytes, and the one extra byte tells whether the
  window starts at a line start. Each pushed byte is copied once. Upstream keeps a list of
  chunks and drops whole leading chunks, then trims to a margin at each snapshot.
- `skip` records output a producer omitted before the next chunk (for example a process
  whose output was read from the end of a log file). The output pushed after it, before the
  next snapshot, must by itself exceed the tail window (more than `max_bytes` bytes or more
  than `max_lines` newlines), so the stored text is discarded and only the totals grow. It
  panics with `Retain::Head`, where upstream throws. Upstream passes the skip with the
  chunk that follows it (`push(chunk, skipped)`); `skip` followed by `push_str` is the same.
- `stored_bytes` is at most `max_bytes + 1` in both modes (upstream: the limits plus one
  chunk, trimmed to a margin at each snapshot).
- `snapshot` returns the window, sanitized, and the dropped totals.

## `Progress`

```rust
pub struct Progress<E>;
pub const PROGRESS_BYTES_PER_SECOND: usize = 100 * 1024;

impl<E: Send + Sync + 'static> Progress<E> {
    pub fn spawn<W, F>(write: W, min_interval: Duration) -> Self
    where W: FnMut() -> F + Send + 'static, F: Future<Output = Result<usize, E>> + Send + 'static;

    pub fn mark(&self);
    pub fn mark_and_wait(&self) -> impl Future<Output = Result<(), Arc<E>>> + Send + use<E>;
    pub fn errors(&self) -> impl Stream<Item = Arc<E>> + Send + Unpin + use<E>;
    pub async fn stop(self) -> PendingWaiters<E>;
}

pub struct PendingWaiters<E>;

impl<E> PendingWaiters<E> {
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn settle(self, result: Result<(), Arc<E>>);
}
```

- `write` commits the current value and returns how many bytes it wrote. It runs on a
  tokio task owned by `Progress` (`spawn` must be called within a runtime). Errors are
  shared as `Arc<E>` between waiters and error streams, hence `E: Send + Sync`.
- Pacing as upstream: the first `mark` after an idle period commits at once; each commit
  delays the next one by at least `min_interval` and by its size at 100 KiB/s
  (`PROGRESS_BYTES_PER_SECOND`). At most one commit is in flight; marks made meanwhile
  coalesce into the next commit.
- `mark_and_wait` marks when it is called, not when its future is first polled, and the
  future does not borrow the `Progress`: it can be awaited after `stop`, as upstream's
  promise can. It resolves with the commit that includes this mark, or that commit's error.
  A waiter handed back by `stop` resolves only through `PendingWaiters::settle`; dropping
  the `PendingWaiters` unsettled leaves it pending, as an unsettled promise upstream.
- `errors` streams the errors of commits that fail after it is called; the stream ends when
  the progress stops or is dropped.
- `stop` stops scheduling, waits for the commit in flight, and returns the waiters that no
  commit has settled yet. The caller makes the final commit itself and settles them, as
  upstream (durable's final tool result is written by the caller, not by `Progress`). A
  mark the commit task has not picked up yet is left to that final commit (upstream starts
  a commit synchronously inside `mark`). A panic in `write` resumes in `stop`.
- Dropping a `Progress` without `stop` lets the commit in flight finish and ends the task.
- Timers are tokio timers. Tests pause tokio's clock (`#[tokio::test(start_paused = true)]`)
  instead of injecting a clock.

## Differences from upstream

- Bytes are the native unit. `utf8_byte_length` is `str::len` and is not ported; upstream
  needs it because JS strings are UTF-16 (and counts a lone surrogate as three bytes, which
  cannot occur in Rust).
- `OutputSlice` has no `bytes` field; it is `text.len()`.
- Tail retention uses a byte ring instead of a chunk list with margin trimming, and head
  retention stores at most `max_bytes + 1` bytes, so snapshots never compact.
- `skip` is a call of its own instead of `push`'s second argument.
- No injected `ProgressClock`; tokio's paused clock covers tests. Durable passes its own
  durable timers to upstream's `Progress`; whether delta's durable needs the same is
  decided when it is designed, and a clock parameter can be added then.
- Write errors go to waiters and to the `errors()` stream instead of an `onError` callback.
