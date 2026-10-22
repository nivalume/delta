//! Bounded output streams and paced progress commits.
//!
//! [`OutputBuffer`] keeps a head or tail window of a stream at a cost proportional to each
//! chunk, [`bound_output`] cuts a string to such a window, and [`Progress`] paces commits of
//! a changing value by their size. See `docs/design/output.md`.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::fmt;
use std::future::Future;
use std::panic::resume_unwind;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use futures::Stream;
use tokio::sync::{Notify, mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::Instant;

const NEWLINE: u8 = b'\n';
const REPLACEMENT: &str = "\u{FFFD}";
const BYTE_ORDER_MARK: char = '\u{FEFF}';

/// Retention limits of one output stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputLimits {
    /// Most UTF-8 bytes the window holds.
    pub max_bytes: usize,
    /// Most lines the window holds; a final line without a newline counts.
    pub max_lines: usize,
    /// Which end of the stream the window keeps.
    pub retain: Retain,
}

/// Which end of a stream an [`OutputLimits`] window keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Retain {
    /// The first lines.
    Head,
    /// The last lines.
    Tail,
}

/// An exact slice of the input within the limits, and what it left out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputSlice<'a> {
    pub text: &'a str,
    pub dropped_bytes: usize,
    pub dropped_lines: usize,
}

/// Retained, sanitized output and what the limits dropped from the whole stream.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BoundedOutput {
    pub text: String,
    pub dropped_bytes: usize,
    pub dropped_lines: usize,
}

/// Output a producer omitted before the next chunk, measured on the decoded text: every
/// U+FFFD counts as three bytes, and no sanitizing is applied.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OutputSkip {
    /// UTF-8 byte length of the omitted text.
    pub bytes: usize,
    /// Newlines (U+000A) in the omitted text.
    pub newlines: usize,
    /// Whether the omitted text ends with a newline.
    pub ends_with_newline: bool,
}

/// Remove control characters that break display and transcripts: C0 controls other than tab
/// and newline, and U+FFF9–U+FFFB. Borrows when there is nothing to remove.
pub fn sanitize_output(text: &str) -> Cow<'_, str> {
    let Some(first) = text.find(is_invalid_output) else {
        return Cow::Borrowed(text);
    };
    let mut sanitized = String::with_capacity(text.len());
    sanitized.push_str(&text[..first]);
    sanitized.extend(text[first..].chars().filter(|&character| !is_invalid_output(character)));
    Cow::Owned(sanitized)
}

fn is_invalid_output(character: char) -> bool {
    matches!(character, '\0'..='\x08' | '\x0b'..='\x1f' | '\u{FFF9}'..='\u{FFFB}')
}

/// Bound `text` to whole lines within the limits: the first lines for [`Retain::Head`], the
/// last lines for [`Retain::Tail`]. The result borrows an exact slice of the input, trailing
/// newline included. A single line longer than `max_bytes` is cut at the byte limit on a
/// character boundary.
pub fn bound_output<'a>(text: &'a str, limits: &OutputLimits) -> OutputSlice<'a> {
    let (from, to) = window(text.as_bytes(), limits);
    // `window` cuts only after a newline or on a character boundary.
    let kept = &text[from..to];
    OutputSlice {
        text: kept,
        dropped_bytes: text.len() - kept.len(),
        dropped_lines: line_count(text.as_bytes()) - line_count(kept.as_bytes()),
    }
}

/// The last UTF-8 character boundary at or before `index` (clamped to the length) in a byte
/// buffer that may not be valid UTF-8.
pub fn char_end(bytes: &[u8], index: usize) -> usize {
    let mut end = index.min(bytes.len());
    while end > 0 && bytes.get(end).is_some_and(|&byte| is_continuation(byte)) {
        end -= 1;
    }
    end
}

/// The first character boundary at or after `index`.
fn char_start(bytes: &[u8], index: usize) -> usize {
    let mut start = index;
    while bytes.get(start).is_some_and(|&byte| is_continuation(byte)) {
        start += 1;
    }
    start
}

fn is_continuation(byte: u8) -> bool {
    byte & 0xc0 == 0x80
}

/// The kept byte range of `bytes` under `limits`.
fn window(bytes: &[u8], limits: &OutputLimits) -> (usize, usize) {
    match limits.retain {
        Retain::Head => (0, head_end(bytes, limits)),
        Retain::Tail => (tail_start(bytes, limits), bytes.len()),
    }
}

fn head_end(bytes: &[u8], limits: &OutputLimits) -> usize {
    if limits.max_lines == 0 || limits.max_bytes == 0 {
        return 0;
    }
    let end = newlines(bytes)
        .nth(limits.max_lines - 1)
        .map_or(bytes.len(), |index| index + 1);
    if end <= limits.max_bytes {
        return end;
    }
    match bytes[..limits.max_bytes].iter().rposition(|&byte| byte == NEWLINE) {
        Some(newline) => newline + 1,
        None => char_end(bytes, limits.max_bytes),
    }
}

fn tail_start(bytes: &[u8], limits: &OutputLimits) -> usize {
    let len = bytes.len();
    if limits.max_lines == 0 || limits.max_bytes == 0 {
        return len;
    }
    // A trailing newline ends the last line rather than starting another.
    let body = bytes.strip_suffix(&[NEWLINE]).unwrap_or(bytes);
    let start = newlines(body)
        .rev()
        .nth(limits.max_lines - 1)
        .map_or(0, |index| index + 1);
    if len - start <= limits.max_bytes {
        return start;
    }
    // The first line starting inside the byte window, or a cut of the last line when it
    // alone is too long.
    let from = len - limits.max_bytes;
    match bytes[from - 1..].iter().position(|&byte| byte == NEWLINE) {
        Some(offset) if from + offset < len => from + offset,
        _ => char_start(bytes, from),
    }
}

/// Positions of the newlines in `bytes`, first to last.
fn newlines(bytes: &[u8]) -> impl DoubleEndedIterator<Item = usize> + '_ {
    bytes
        .iter()
        .enumerate()
        .filter(|&(_, &byte)| byte == NEWLINE)
        .map(|(index, _)| index)
}

#[allow(clippy::naive_bytecount)] // No `bytecount` dependency for chunk-sized counts.
fn count_newlines(bytes: &[u8]) -> usize {
    bytes.iter().filter(|&&byte| byte == NEWLINE).count()
}

/// Lines in `bytes`; a final line without a newline counts.
fn line_count(bytes: &[u8]) -> usize {
    match bytes.last() {
        None => 0,
        Some(&last) => lines(count_newlines(bytes), last == NEWLINE),
    }
}

/// Lines of text with `newlines` newlines; a final unterminated line counts.
fn lines(newlines: usize, terminated: bool) -> usize {
    newlines + usize::from(!terminated)
}

/// Bounded running output of one stream, such as a tool call's or a process's.
///
/// Accepting a chunk costs time proportional to the chunk, and at most `max_bytes + 1`
/// bytes are stored: head retention stops storing once the window is full, tail retention
/// keeps the last `max_bytes + 1` bytes in a ring. Totals of the whole stream are kept so the
/// dropped counts stay exact.
pub struct OutputBuffer {
    limits: OutputLimits,
    /// Head: the start of the stream. Tail: its last bytes, or those since the last skip.
    stored: VecDeque<u8>,
    /// Head: newlines in `stored`.
    stored_newlines: usize,
    /// Head: the window is complete; later output is only counted.
    full: bool,
    /// Bytes of an incomplete character at the end of the last byte chunk.
    pending: Vec<u8>,
    /// Whether output has started; a byte-order mark is dropped only before that.
    started: bool,
    total_bytes: usize,
    total_newlines: usize,
    ends_with_newline: bool,
}

impl fmt::Debug for OutputBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OutputBuffer")
            .field("limits", &self.limits)
            .field("stored_bytes", &self.stored.len())
            .field("total_bytes", &self.total_bytes)
            .field("total_newlines", &self.total_newlines)
            .finish_non_exhaustive()
    }
}

impl OutputBuffer {
    pub fn new(limits: OutputLimits) -> Self {
        Self {
            limits,
            stored: VecDeque::new(),
            stored_newlines: 0,
            full: false,
            pending: Vec::new(),
            started: false,
            total_bytes: 0,
            total_newlines: 0,
            // An empty stream has no lines.
            ends_with_newline: true,
        }
    }

    /// Bytes currently held; at most `max_bytes + 1`.
    pub fn stored_bytes(&self) -> usize {
        self.stored.len()
    }

    /// Accept decoded text. An incomplete character left by an earlier byte chunk becomes
    /// U+FFFD first.
    pub fn push_str(&mut self, chunk: &str) {
        self.flush_pending();
        if chunk.is_empty() {
            return;
        }
        self.started = true;
        self.accept(chunk.as_bytes());
    }

    /// Accept raw output, decoded as a UTF-8 stream: an incomplete character at the end waits
    /// for the next chunk, invalid bytes become U+FFFD, and a byte-order mark is dropped only
    /// at the very start of the stream.
    pub fn push_bytes(&mut self, chunk: &[u8]) {
        let rest = self.complete_pending(chunk);
        let mut pieces = rest.utf8_chunks().peekable();
        while let Some(piece) = pieces.next() {
            self.emit(piece.valid());
            let invalid = piece.invalid();
            if invalid.is_empty() {
                continue;
            }
            if pieces.peek().is_none() && is_incomplete(invalid) {
                self.pending.extend_from_slice(invalid);
            } else {
                self.emit(REPLACEMENT);
            }
        }
    }

    /// Record output a producer omitted before the next chunk, such as the start of a log
    /// file read from its end. Only the totals grow: the output pushed after the skip, before
    /// the next snapshot, must by itself exceed the tail window (more than `max_bytes` bytes
    /// or more than `max_lines` newlines), so nothing before it is in the window.
    ///
    /// # Panics
    ///
    /// With [`Retain::Head`], whose window would need the omitted text.
    pub fn skip(&mut self, skipped: OutputSkip) {
        assert!(
            self.limits.retain == Retain::Tail,
            "skipped output requires tail retention"
        );
        self.flush_pending();
        self.started = true;
        if skipped.bytes == 0 {
            return;
        }
        self.total_bytes += skipped.bytes;
        self.total_newlines += skipped.newlines;
        self.ends_with_newline = skipped.ends_with_newline;
        self.stored.clear();
    }

    /// End the stream: an incomplete trailing character becomes U+FFFD.
    pub fn end(&mut self) {
        self.flush_pending();
    }

    /// The window, sanitized, and what the limits dropped from the whole stream.
    pub fn snapshot(&mut self) -> BoundedOutput {
        // The stored bytes yield the window of the whole stream: head stores its first
        // `max_bytes + 1` bytes, tail its last ones, and the extra byte tells where the
        // window's first line starts.
        let stored = self.stored.make_contiguous();
        let (from, to) = window(stored, &self.limits);
        let kept = &stored[from..to];
        let kept_lines = line_count(kept);
        let decoded = String::from_utf8_lossy(kept);
        let text = match sanitize_output(&decoded) {
            Cow::Owned(sanitized) => sanitized,
            Cow::Borrowed(_) => decoded.into_owned(),
        };
        BoundedOutput {
            dropped_bytes: self.total_bytes - kept.len(),
            dropped_lines: lines(self.total_newlines, self.ends_with_newline) - kept_lines,
            text,
        }
    }

    /// Complete the pending character with the first bytes of `chunk`; returns the rest.
    fn complete_pending<'c>(&mut self, chunk: &'c [u8]) -> &'c [u8] {
        if self.pending.is_empty() {
            return chunk;
        }
        let mut pending = std::mem::take(&mut self.pending);
        let mut rest = chunk;
        loop {
            let Some((&byte, after)) = rest.split_first() else {
                // Still incomplete.
                self.pending = pending;
                return rest;
            };
            pending.push(byte);
            match std::str::from_utf8(&pending) {
                Ok(character) => {
                    self.emit(character);
                    rest = after;
                    break;
                }
                Err(error) if error.error_len().is_none() => rest = after,
                Err(_) => {
                    // The byte does not continue the character; it starts the rest.
                    self.emit(REPLACEMENT);
                    break;
                }
            }
        }
        pending.clear();
        self.pending = pending;
        rest
    }

    fn flush_pending(&mut self) {
        if !self.pending.is_empty() {
            self.pending.clear();
            self.emit(REPLACEMENT);
        }
    }

    /// Accept decoded text of a byte chunk.
    fn emit(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let text = if self.started {
            text
        } else {
            self.started = true;
            text.strip_prefix(BYTE_ORDER_MARK).unwrap_or(text)
        };
        self.accept(text.as_bytes());
    }

    fn accept(&mut self, bytes: &[u8]) {
        let Some(&last) = bytes.last() else { return };
        let newlines = count_newlines(bytes);
        self.total_bytes += bytes.len();
        self.total_newlines += newlines;
        self.ends_with_newline = last == NEWLINE;
        let capacity = self.limits.max_bytes.saturating_add(1);
        match self.limits.retain {
            Retain::Head => {
                if self.full {
                    return;
                }
                // Nothing past the first `max_bytes + 1` bytes can be in the window.
                let stored = &bytes[..bytes.len().min(capacity - self.stored.len())];
                self.stored.extend(stored);
                self.stored_newlines += if stored.len() == bytes.len() {
                    newlines
                } else {
                    count_newlines(stored)
                };
                self.full = self.stored.len() > self.limits.max_bytes || self.stored_newlines >= self.limits.max_lines;
            }
            Retain::Tail => {
                if bytes.len() >= capacity {
                    self.stored.clear();
                    self.stored.extend(&bytes[bytes.len() - capacity..]);
                } else {
                    let excess = (self.stored.len() + bytes.len()).saturating_sub(capacity);
                    self.stored.drain(..excess);
                    self.stored.extend(bytes);
                }
            }
        }
    }
}

/// Whether `bytes` are the start of a character that more bytes could complete.
fn is_incomplete(bytes: &[u8]) -> bool {
    std::str::from_utf8(bytes).is_err_and(|error| error.error_len().is_none())
}

/// Each progress commit also buys a pause proportional to what it wrote.
pub const PROGRESS_BYTES_PER_SECOND: usize = 100 * 1024;

type Waiter<E> = oneshot::Sender<Result<(), Arc<E>>>;

/// Paced commits of a changing value.
///
/// The first [`mark`](Self::mark) after an idle period commits at once; each commit then
/// delays the next by at least `min_interval` and by its written size at
/// [`PROGRESS_BYTES_PER_SECOND`]. At most one commit is in flight; marks made meanwhile
/// coalesce into the next one. Commits run on a tokio task owned by the `Progress`; dropping
/// it without [`stop`](Self::stop) lets the commit in flight finish and schedules no more.
pub struct Progress<E> {
    shared: Arc<Shared<E>>,
    task: JoinHandle<()>,
}

struct Shared<E> {
    state: Mutex<ProgressState<E>>,
    /// Wakes the commit task on a mark or a stop.
    wake: Notify,
}

struct ProgressState<E> {
    dirty: bool,
    stopped: bool,
    /// Waiters for the next commit.
    waiters: Vec<Waiter<E>>,
    error_subscribers: Vec<mpsc::UnboundedSender<Arc<E>>>,
}

impl<E> Shared<E> {
    fn lock(&self) -> MutexGuard<'_, ProgressState<E>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn signal_stop(&self) {
        self.lock().stopped = true;
        self.wake.notify_one();
    }
}

impl<E> fmt::Debug for Progress<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.shared.lock();
        f.debug_struct("Progress")
            .field("dirty", &state.dirty)
            .field("stopped", &state.stopped)
            .field("waiters", &state.waiters.len())
            .finish_non_exhaustive()
    }
}

impl<E: Send + Sync + 'static> Progress<E> {
    /// Start committing with `write`, which commits the current value and returns how many
    /// bytes it wrote. Must be called within a tokio runtime.
    pub fn spawn<W, F>(write: W, min_interval: Duration) -> Self
    where
        W: FnMut() -> F + Send + 'static,
        F: Future<Output = Result<usize, E>> + Send + 'static,
    {
        let shared = Arc::new(Shared {
            state: Mutex::new(ProgressState {
                dirty: false,
                stopped: false,
                waiters: Vec::new(),
                error_subscribers: Vec::new(),
            }),
            wake: Notify::new(),
        });
        let task = tokio::spawn(commit_loop(Arc::clone(&shared), write, min_interval));
        Self { shared, task }
    }

    /// Schedule a commit.
    pub fn mark(&self) {
        self.shared.lock().dirty = true;
        self.shared.wake.notify_one();
    }

    /// Schedule a commit now; the future resolves with the commit that includes this mark,
    /// or with that commit's error. After [`stop`](Self::stop), only
    /// [`PendingWaiters::settle`] resolves it; it never resolves if the waiters are dropped
    /// unsettled.
    pub fn mark_and_wait(&self) -> impl Future<Output = Result<(), Arc<E>>> + Send + use<E> {
        let (sender, receiver) = oneshot::channel();
        {
            let mut state = self.shared.lock();
            state.waiters.push(sender);
            state.dirty = true;
        }
        self.shared.wake.notify_one();
        async move {
            match receiver.await {
                Ok(result) => result,
                Err(_) => std::future::pending().await,
            }
        }
    }

    /// Errors of commits that fail from now on. The stream ends when the progress stops.
    pub fn errors(&self) -> impl Stream<Item = Arc<E>> + Send + Unpin + use<E> {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        self.shared.lock().error_subscribers.push(sender);
        futures::stream::poll_fn(move |cx| receiver.poll_recv(cx))
    }

    /// Stop scheduling, wait for the commit in flight, and return the waiters no commit has
    /// settled yet. The caller makes the final commit and settles them. A mark the commit
    /// task has not picked up yet is left to that final commit.
    pub async fn stop(mut self) -> PendingWaiters<E> {
        self.shared.signal_stop();
        if let Err(error) = (&mut self.task).await
            && error.is_panic()
        {
            resume_unwind(error.into_panic());
        }
        let mut state = self.shared.lock();
        state.error_subscribers.clear();
        PendingWaiters {
            waiters: std::mem::take(&mut state.waiters),
        }
    }
}

impl<E> Drop for Progress<E> {
    fn drop(&mut self) {
        self.shared.signal_stop();
    }
}

/// Waiters of [`Progress::mark_and_wait`] that no commit settled before
/// [`Progress::stop`].
pub struct PendingWaiters<E> {
    waiters: Vec<Waiter<E>>,
}

impl<E> fmt::Debug for PendingWaiters<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingWaiters")
            .field("len", &self.waiters.len())
            .finish()
    }
}

impl<E> PendingWaiters<E> {
    pub fn len(&self) -> usize {
        self.waiters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.waiters.is_empty()
    }

    /// Resolve every waiter with the outcome of the final commit.
    pub fn settle(mut self, result: Result<(), Arc<E>>) {
        let Some(last) = self.waiters.pop() else { return };
        // A waiter whose future was dropped no longer listens.
        for waiter in self.waiters {
            let _ = waiter.send(result.clone());
        }
        let _ = last.send(result);
    }
}

/// The commit task of a [`Progress`].
async fn commit_loop<E, W, F>(shared: Arc<Shared<E>>, mut write: W, min_interval: Duration)
where
    W: FnMut() -> F,
    F: Future<Output = Result<usize, E>>,
{
    let mut next_at: Option<Instant> = None;
    loop {
        // Idle until a mark.
        loop {
            {
                let state = shared.lock();
                if state.stopped {
                    return;
                }
                if state.dirty {
                    break;
                }
            }
            shared.wake.notified().await;
        }
        // The pause the last commit bought; marks meanwhile coalesce.
        if let Some(at) = next_at {
            let sleep = tokio::time::sleep_until(at);
            tokio::pin!(sleep);
            loop {
                tokio::select! {
                    () = &mut sleep => break,
                    () = shared.wake.notified() => {
                        if shared.lock().stopped {
                            return;
                        }
                    }
                }
            }
        }
        let waiters = {
            let mut state = shared.lock();
            if state.stopped {
                return;
            }
            state.dirty = false;
            std::mem::take(&mut state.waiters)
        };
        let started = Instant::now();
        let result = match write().await {
            Ok(bytes) => {
                next_at = Some(started + min_interval.max(size_pause(bytes)));
                Ok(())
            }
            Err(error) => {
                next_at = Some(started + min_interval);
                Err(Arc::new(error))
            }
        };
        for waiter in waiters {
            // A waiter whose future was dropped no longer listens.
            let _ = waiter.send(result.clone());
        }
        if let Err(error) = result {
            shared
                .lock()
                .error_subscribers
                .retain(|subscriber| subscriber.send(Arc::clone(&error)).is_ok());
        }
    }
}

/// The pause a commit of `bytes` bytes buys at [`PROGRESS_BYTES_PER_SECOND`].
fn size_pause(bytes: usize) -> Duration {
    let nanos = bytes as u128 * 1_000_000_000 / PROGRESS_BYTES_PER_SECOND as u128;
    Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX))
}
