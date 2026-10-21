//! What a running call reports: its bounded output window, details and diagnostics, published
//! into its slot by paced progress commits.
//!
//! `delta_chord::output::Progress` drives the pace from its own tokio task, so its write
//! closure cannot hold the borrowed task runtime. The closure snapshots what changed and
//! sends it, with a reply channel, to [`serve`], which runs in the invocation beside the
//! tool's `execute` and commits each update through the runtime. Stopping the `Progress`
//! drops the closure and with it the channel, which ends `serve`.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use delta_chord::context::Context;
use delta_chord::delta::overlap;
use delta_chord::json::Json;
use delta_chord::output::{BoundedOutput, OutputBuffer, OutputLimits, OutputSkip, PendingWaiters, Progress, Retain};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};

use super::result::ToolDiagnostic;
use super::task::ToolTask;
use crate::generation::{SlotProgress, edit_live};
use crate::records::TaskId;
use crate::scheduler::{RuntimeError, TaskRuntime};

/// How far back the byte estimate looks for a sliding window's overlap.
const OVERLAP_SCAN: usize = 64 * 1024;

/// What one progress commit writes into the slot.
#[derive(Debug)]
pub(super) struct SlotUpdate {
    /// The window and its dropped counts, when they changed.
    output: Option<BoundedOutput>,
    /// The details, when they changed.
    details: Option<Value>,
    /// Diagnostics reported since the last commit.
    diagnostics: Vec<ToolDiagnostic>,
}

impl SlotUpdate {
    fn is_empty(&self) -> bool {
        self.output.is_none() && self.details.is_none() && self.diagnostics.is_empty()
    }

    fn apply(self, progress: &mut SlotProgress) {
        if let Some(output) = self.output {
            let count = |n: usize| (n > 0).then(|| u64::try_from(n).unwrap_or(u64::MAX));
            progress.output = (!output.text.is_empty()).then_some(output.text);
            progress.dropped_bytes = count(output.dropped_bytes);
            progress.dropped_lines = count(output.dropped_lines);
        }
        if let Some(details) = self.details {
            progress.details = Some(Json::from(details));
        }
        if !self.diagnostics.is_empty() {
            progress
                .diagnostics
                .get_or_insert_with(Vec::new)
                .extend(self.diagnostics);
        }
    }
}

/// The reported state of one invocation and what its last progress commit carried.
struct Reported {
    buffer: OutputBuffer,
    details: Option<Value>,
    details_changed: bool,
    diagnostics: Vec<ToolDiagnostic>,
    published_window: BoundedOutput,
    published_diagnostics: usize,
}

impl Reported {
    /// What changed since the last update, and an estimate of its written size.
    fn update(&mut self) -> (SlotUpdate, usize) {
        let window = self.buffer.snapshot();
        let mut bytes = 0;
        let output = (window != self.published_window).then(|| {
            let previous = &self.published_window.text;
            bytes += window.text.len() - overlap(previous, &window.text, OVERLAP_SCAN.min(previous.len()));
            self.published_window = window.clone();
            window
        });
        let details = std::mem::take(&mut self.details_changed)
            .then(|| self.details.clone())
            .flatten();
        bytes += details.as_ref().map_or(0, |details| details.to_string().len());
        let diagnostics = self.diagnostics[self.published_diagnostics..].to_vec();
        bytes += diagnostics
            .iter()
            .map(|diagnostic| diagnostic.message.len())
            .sum::<usize>();
        self.published_diagnostics = self.diagnostics.len();
        let update = SlotUpdate {
            output,
            details,
            diagnostics,
        };
        (update, bytes)
    }
}

/// A progress update on its way to [`serve`], with the reply its commit resolves.
pub(super) type Request = (SlotUpdate, oneshot::Sender<Result<(), RuntimeError>>);

/// The reporting side of one invocation: what `ToolApi` writes into, and the `Progress`
/// that publishes it.
pub(super) struct Reporter {
    state: Arc<Mutex<Reported>>,
    limits: OutputLimits,
    progress: Option<Progress<RuntimeError>>,
    task: TaskId,
}

impl Reporter {
    /// A reporter with `limits`; with `interval`, it publishes through the returned channel
    /// at that pace.
    pub(super) fn new(
        limits: OutputLimits,
        interval: Option<Duration>,
        task: TaskId,
    ) -> (Self, Option<mpsc::UnboundedReceiver<Request>>) {
        let state = Arc::new(Mutex::new(Reported {
            buffer: OutputBuffer::new(limits),
            details: None,
            details_changed: false,
            diagnostics: Vec::new(),
            published_window: BoundedOutput::default(),
            published_diagnostics: 0,
        }));
        let Some(interval) = interval else {
            let reporter = Self {
                state,
                limits,
                progress: None,
                task,
            };
            return (reporter, None);
        };
        let (sender, receiver) = mpsc::unbounded_channel::<Request>();
        let shared = state.clone();
        let write = move || {
            let (update, bytes) = lock(&shared).update();
            let sender = sender.clone();
            async move {
                if update.is_empty() {
                    return Ok(0);
                }
                let (reply, replied) = oneshot::channel();
                sender.send((update, reply)).map_err(|_| RuntimeError::Ended(task))?;
                replied.await.map_err(|_| RuntimeError::Ended(task))??;
                Ok(bytes)
            }
        };
        let reporter = Self {
            state,
            limits,
            progress: Some(Progress::spawn(write, interval)),
            task,
        };
        (reporter, Some(receiver))
    }

    pub(super) fn limits(&self) -> OutputLimits {
        self.limits
    }

    fn mark(&self) {
        if let Some(progress) = &self.progress {
            progress.mark();
        }
    }

    pub(super) fn push_bytes(&self, chunk: &[u8]) {
        lock(&self.state).buffer.push_bytes(chunk);
        self.mark();
    }

    pub(super) fn push_str(&self, chunk: &str) {
        lock(&self.state).buffer.push_str(chunk);
        self.mark();
    }

    /// Records an omission; `false` (and nothing recorded) for a head window, which would
    /// need the omitted text.
    pub(super) fn skip(&self, skip: OutputSkip) -> bool {
        if self.limits.retain != Retain::Tail {
            return false;
        }
        lock(&self.state).buffer.skip(skip);
        self.mark();
        true
    }

    pub(super) fn diagnostic(&self, diagnostic: ToolDiagnostic) {
        lock(&self.state).diagnostics.push(diagnostic);
        self.mark();
    }

    /// The window as it stands.
    pub(super) fn retained(&self) -> BoundedOutput {
        lock(&self.state).buffer.snapshot()
    }

    /// Replaces the details; resolves after the commit that carries them.
    pub(super) async fn details(&self, value: Value) -> Result<(), Arc<RuntimeError>> {
        {
            let mut state = lock(&self.state);
            state.details = Some(value);
            state.details_changed = true;
        }
        match &self.progress {
            Some(progress) => progress.mark_and_wait().await,
            None => Ok(()),
        }
    }

    /// Stops publishing and returns what was reported, with the `details` waiters the
    /// settle commit resolves.
    pub(super) async fn finish(self) -> Reports {
        let waiters = match self.progress {
            Some(progress) => Some(progress.stop().await),
            None => None,
        };
        let mut state = lock(&self.state);
        state.buffer.end();
        Reports {
            output: state.buffer.snapshot(),
            retain: self.limits.retain,
            details: state.details.clone(),
            diagnostics: std::mem::take(&mut state.diagnostics),
            waiters: Waiters {
                pending: waiters,
                task: self.task,
            },
        }
    }
}

fn lock(state: &Mutex<Reported>) -> MutexGuard<'_, Reported> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What an invocation reported, once it stopped.
#[derive(Debug)]
pub(super) struct Reports {
    pub(super) output: BoundedOutput,
    pub(super) retain: Retain,
    pub(super) details: Option<Value>,
    pub(super) diagnostics: Vec<ToolDiagnostic>,
    pub(super) waiters: Waiters,
}

/// `details` waiters no progress commit resolved: the settle commit resolves them after
/// its writes; dropped unresolved (the commit never ran), they fail.
#[derive(Debug)]
pub(super) struct Waiters {
    pending: Option<PendingWaiters<RuntimeError>>,
    task: TaskId,
}

impl Waiters {
    /// No waiters, for a settlement without an invocation's reports.
    pub(super) fn none(task: TaskId) -> Self {
        Self { pending: None, task }
    }

    pub(super) fn resolve(mut self) {
        if let Some(pending) = self.pending.take() {
            pending.settle(Ok(()));
        }
    }
}

impl Drop for Waiters {
    fn drop(&mut self) {
        if let Some(pending) = self.pending.take() {
            pending.settle(Err(Arc::new(RuntimeError::Ended(self.task))));
        }
    }
}

/// Commits each progress update into the task's slot until the reporter stops; a commit
/// writes nothing once the slot is gone.
pub(super) async fn serve(
    rt: &TaskRuntime<ToolTask>,
    receiver: Option<mpsc::UnboundedReceiver<Request>>,
    ctx: &Context,
) {
    let Some(mut receiver) = receiver else {
        return;
    };
    let conversation = rt.conversation_id();
    let task = rt.task_id();
    while let Some((update, reply)) = receiver.recv().await {
        let committed = rt
            .commit(ctx, move |tx, _| {
                Box::pin(async move {
                    edit_live(tx, conversation, |live| {
                        if let Some(progress) = live.slot_progress_mut(task) {
                            update.apply(progress);
                        }
                    })
                    .await?;
                    Ok(((), None))
                })
            })
            .await;
        // The waiter may have stopped listening.
        let _ = reply.send(committed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reported(limits: OutputLimits) -> Reported {
        Reported {
            buffer: OutputBuffer::new(limits),
            details: None,
            details_changed: false,
            diagnostics: Vec::new(),
            published_window: BoundedOutput::default(),
            published_diagnostics: 0,
        }
    }

    #[test]
    fn updates_carry_only_changes() {
        let limits = OutputLimits {
            max_bytes: 10,
            max_lines: 100,
            retain: Retain::Tail,
        };
        let mut state = reported(limits);
        state.buffer.push_str("abcdef");
        let (update, bytes) = state.update();
        assert_eq!(
            update.output.as_ref().map(|output| output.text.as_str()),
            Some("abcdef")
        );
        assert_eq!(bytes, 6);
        state.buffer.push_str("ghijkl");
        state.diagnostics.push(ToolDiagnostic::error("x", "boom"));
        let (update, bytes) = state.update();
        let output = update.output.unwrap();
        assert_eq!((output.text.as_str(), output.dropped_bytes), ("cdefghijkl", 2));
        assert_eq!(update.diagnostics.len(), 1);
        assert_eq!(bytes, 6 + 4);
        let (update, bytes) = state.update();
        assert!(update.is_empty());
        assert_eq!(bytes, 0);

        let mut progress = SlotProgress::default();
        state.details = Some(serde_json::json!({ "n": 1 }));
        state.details_changed = true;
        state.update().0.apply(&mut progress);
        assert_eq!(progress.details, Some(delta_chord::json!({ "n": 1 })));
        assert_eq!(progress.output, None);
    }
}
