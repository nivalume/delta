//! Compaction as the generation sees it: the thresholds it checks before a request, and the
//! [`Compactor`] seam that selects a cut and starts a `delta.compaction` task.
//!
//! Phase 4 of the crate implements the generation's side (spec §8.3, §8.7): the context
//! estimate, the blocking and background thresholds, the overflow path and waiting for a
//! blocking compaction. Range selection and the `delta.compaction` task itself arrive with
//! phase 5 behind [`Compactor`]; until then the default [`NoCompaction`] finds no cut, so
//! no threshold fires and an overflow fails the run. See `docs/design/compaction.md`.

use delta_ai::estimate::{calculate_context_tokens, estimate_message_tokens};
use delta_ai::types::Message;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

use crate::agent::CompactionPolicy;
use crate::context::ContextView;
use crate::records::{ConversationId, TaskId};
use crate::session::{Tx, TxError};

/// Why a compaction runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CompactionReason {
    Manual,
    Threshold,
    Overflow,
}

/// The input of a `delta.compaction` task.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionInput {
    pub reason: CompactionReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// The seam between generation and compaction.
pub trait Compactor: Send + Sync {
    /// Range selection over `view`: the index into `view.entries` of the first entry a
    /// summary keeps, or `None` when there is nothing to compact.
    fn select_cut(&self, view: &ContextView, keep_recent_tokens: u64) -> Option<usize>;

    /// Creates a `delta.compaction` task in `tx`: owned by the generation `owner` (blocking,
    /// the generation waits for it), or by the conversation (background unless manual), and
    /// adds its status to `delta.live.compactions` in the same commit.
    fn create<'a>(
        &'a self,
        tx: &'a mut Tx<'_>,
        conversation: ConversationId,
        input: CompactionInput,
        owner: Option<TaskId>,
    ) -> BoxFuture<'a, Result<TaskId, TxError>>;
}

/// The default [`Compactor`] until the compaction task lands: nothing is ever compacted.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoCompaction;

impl Compactor for NoCompaction {
    fn select_cut(&self, _view: &ContextView, _keep_recent_tokens: u64) -> Option<usize> {
        None
    }

    fn create<'a>(
        &'a self,
        _tx: &'a mut Tx<'_>,
        _conversation: ConversationId,
        _input: CompactionInput,
        _owner: Option<TaskId>,
    ) -> BoxFuture<'a, Result<TaskId, TxError>> {
        Box::pin(async { Err(TxError::callback("No compaction task is installed")) })
    }
}

/// The estimated tokens of `view`'s context plus `extra` (the planned system messages):
/// from the newest assistant message after the head marker whose usage is nonzero, its
/// reported context tokens plus an estimate of every later message; without one, an
/// estimate of every message.
pub fn estimate_context(view: &ContextView, extra: &[Message]) -> u64 {
    let after = view.head.as_ref().map(|head| head.id);
    let measured = view
        .entries
        .iter()
        .zip(&view.contributions)
        .rev()
        .filter(|(entry, _)| after.is_none_or(|after| entry.id > after))
        .find_map(|(_, contribution)| {
            contribution.iter().rev().find(|message| match message {
                Message::Assistant(assistant) => calculate_context_tokens(&assistant.usage) > 0,
                _ => false,
            })
        });
    let (from, base) = match measured {
        Some(message @ Message::Assistant(assistant)) => (
            view.messages
                .iter()
                .rposition(|candidate| candidate == message)
                .map_or(0, |index| index + 1),
            calculate_context_tokens(&assistant.usage),
        ),
        _ => (0, 0),
    };
    base + view.messages[from..]
        .iter()
        .chain(extra)
        .map(estimate_message_tokens)
        .sum::<u64>()
}

/// Which threshold compaction preparation starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Threshold {
    /// Above `context_window - reserve_tokens`: compact first and prepare again.
    Blocking,
    /// Above the background threshold: start one beside the request.
    Background,
}

/// The threshold `view` plus `planned` crosses, when the policy is enabled, the window is
/// positive and range selection finds a cut.
pub fn threshold(
    view: &ContextView,
    planned: &[Message],
    context_window: u64,
    policy: &CompactionPolicy,
    compactor: &dyn Compactor,
) -> Option<Threshold> {
    if !policy.enabled || context_window == 0 {
        return None;
    }
    let tokens = estimate_context(view, planned);
    let blocking = context_window.saturating_sub(policy.reserve_tokens);
    let background = blocking.saturating_sub(policy.background_tokens);
    let over = if tokens > blocking {
        Threshold::Blocking
    } else if policy.background_tokens > 0 && tokens > background {
        Threshold::Background
    } else {
        return None;
    };
    compactor.select_cut(view, policy.keep_recent_tokens).map(|_| over)
}
