//! Range selection (spec §8.7) and the context estimate the thresholds (§8.3) compare.

use delta_ai::estimate::{calculate_context_tokens, estimate_message_tokens};
use delta_ai::types::Message;

use super::Compactor;
use crate::agent::CompactionPolicy;
use crate::context::{ContextView, order_tool_results};

/// The index into `view.entries` of the first entry a summary keeps, or `None` when there is
/// nothing to compact. Walks back from the tail until `keep_recent_tokens` are kept, then
/// cuts at the first candidate at or after that entry, or the newest candidate when none
/// follows: an entry whose contribution starts with a user or assistant message, never a
/// tool result, and never a user entry that a result of the preceding assistant's calls
/// still follows. `None` as well when nothing before the cut contributes a message.
pub fn select_cut(view: &ContextView, keep_recent_tokens: u64) -> Option<usize> {
    let contributions = &view.contributions;
    let start = usize::from(view.head.is_some());
    let candidates: Vec<usize> = (start..contributions.len())
        .filter(|&index| is_candidate(contributions, index))
        .collect();
    let mut kept = 0;
    let mut cut = None;
    for index in (start..contributions.len()).rev() {
        kept += contributions[index].iter().map(estimate_message_tokens).sum::<u64>();
        if kept < keep_recent_tokens {
            continue;
        }
        cut = candidates
            .iter()
            .copied()
            .find(|&candidate| candidate >= index)
            .or_else(|| candidates.last().copied());
        break;
    }
    let cut = cut?;
    contributions[start..cut]
        .iter()
        .any(|contribution| !contribution.is_empty())
        .then_some(cut)
}

/// Whether the entry at `index` may start the kept range.
fn is_candidate(contributions: &[Vec<Message>], index: usize) -> bool {
    match contributions[index].first() {
        Some(Message::Assistant(_)) => true,
        Some(Message::User(_)) => {
            // A result of the preceding assistant's calls that follows this entry, before the
            // next assistant, belongs before it.
            let calls: Vec<&str> = contributions[..index]
                .iter()
                .rev()
                .find_map(|contribution| {
                    contribution.iter().rev().find_map(|message| match message {
                        Message::Assistant(assistant) => Some(assistant),
                        _ => None,
                    })
                })
                .map(|assistant| assistant.tool_calls().map(|call| call.id.as_str()).collect())
                .unwrap_or_default();
            if calls.is_empty() {
                return true;
            }
            for (after, contribution) in contributions.iter().enumerate().skip(index) {
                for (position, message) in contribution.iter().enumerate() {
                    match message {
                        Message::Assistant(_) if after > index || position > 0 => return true,
                        Message::ToolResult(result) if calls.contains(&result.tool_call_id.as_str()) => return false,
                        _ => {}
                    }
                }
            }
            true
        }
        _ => false,
    }
}

/// The model messages of the entries before `cut`, head marker first, ordered as model
/// context orders them (tool results after their calls).
pub fn summarized_messages(view: &ContextView, cut: usize) -> Vec<Message> {
    let messages: Vec<Message> = view.contributions[..cut].iter().flatten().cloned().collect();
    order_tool_results(&messages)
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
