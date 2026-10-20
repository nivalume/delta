//! The usage ledger: `delta.usage`, one conversation's own spend (spec §8.6).
//!
//! Every built-in writer of a `delta.assistant` entry adds the message's usage under its
//! `provider/model` in the same commit; every writer of a `delta.tool-result` with usage adds
//! it under the tool's name. Failed and aborted attempts count. Nothing stores a
//! cross-conversation total: `Harness::usage` sums the documents. Corresponds to upstream
//! `src/harness/usage.ts`. See `docs/design/generation.md`, "Usage".

use delta_ai::types::Usage;
use delta_chord::delta::Op;
use delta_chord::json::Json;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::documents::{CheckpointInfo, ConversationScope, DocAddress, DocDefinition, ForkInitial, Latest};
use crate::records::ConversationId;
use crate::session::{Tx, TxError};

/// The `delta.usage` document: conversation scope, `latest`, a fork starts at zero, a
/// complete base on every change.
#[derive(Clone, Copy, Debug, Default)]
pub struct UsageDoc;

impl DocDefinition for UsageDoc {
    const KIND: &'static str = "delta.usage";
    const VERSION: u32 = 1;
    type Scope = ConversationScope<Latest, ForkInitial>;
    type Seed = ();

    fn initial((): &()) -> Json {
        delta_chord::json!({ "models": {}, "tools": {} })
    }

    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        true
    }
}

/// The value of a `delta.usage` document.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageState {
    /// Assistant entries and summarization attempts, keyed `provider/model_id`.
    #[serde(default)]
    pub models: IndexMap<String, Usage>,
    /// Tool results, keyed by tool name; their usage has no model identity.
    #[serde(default)]
    pub tools: IndexMap<String, Usage>,
}

/// Which side of the ledger a usage goes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UsageBucket {
    Models,
    Tools,
}

/// Adds `usage` to one bucket of `conversation`'s `delta.usage`, in the commit that records
/// the response or result.
pub async fn record_usage(
    tx: &mut Tx<'_>,
    conversation: ConversationId,
    bucket: UsageBucket,
    key: &str,
    usage: &Usage,
) -> Result<(), TxError> {
    let mut doc = tx.doc(DocAddress::<UsageDoc>::new(conversation)).await?;
    doc.update_as::<UsageState, _>(|state| {
        let totals = match bucket {
            UsageBucket::Models => &mut state.models,
            UsageBucket::Tools => &mut state.tools,
        };
        match totals.get_mut(key) {
            Some(total) => add_usage(total, usage),
            None => {
                totals.insert(key.to_owned(), usage.clone());
            }
        }
    })
    .map_err(TxError::callback)
}

/// Adds every counter of `usage` to `total`; the optional counters once either side
/// reports them.
pub fn add_usage(total: &mut Usage, usage: &Usage) {
    total.input += usage.input;
    total.output += usage.output;
    total.cache_read += usage.cache_read;
    total.cache_write += usage.cache_write;
    total.total_tokens += usage.total_tokens;
    if let Some(write) = usage.cache_write_1h {
        total.cache_write_1h = Some(total.cache_write_1h.unwrap_or(0) + write);
    }
    if let Some(reasoning) = usage.reasoning {
        total.reasoning = Some(total.reasoning.unwrap_or(0) + reasoning);
    }
    total.cost.input += usage.cost.input;
    total.cost.output += usage.cost.output;
    total.cost.cache_read += usage.cost.cache_read;
    total.cost.cache_write += usage.cost.cache_write;
    total.cost.total += usage.cost.total;
}

/// Adds every bucket of `state` into `sum`.
pub fn add_usage_state(sum: &mut UsageState, state: &UsageState) {
    for (totals, entries) in [(&mut sum.models, &state.models), (&mut sum.tools, &state.tools)] {
        for (key, usage) in entries {
            match totals.get_mut(key) {
                Some(total) => add_usage(total, usage),
                None => {
                    totals.insert(key.clone(), usage.clone());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(input: u64, reasoning: Option<u64>) -> Usage {
        Usage {
            input,
            output: 1,
            total_tokens: input + 1,
            reasoning,
            ..Usage::default()
        }
    }

    #[test]
    fn optional_counters_appear_once_either_side_reports_them() {
        let mut total = usage(2, None);
        add_usage(&mut total, &usage(3, Some(4)));
        assert_eq!((total.input, total.output, total.reasoning), (5, 2, Some(4)));
        add_usage(&mut total, &usage(1, None));
        assert_eq!(total.reasoning, Some(4));
        assert_eq!(total.cache_write_1h, None);
    }

    #[test]
    fn states_sum_per_key() {
        let mut sum = UsageState::default();
        let mut one = UsageState::default();
        one.models.insert("p/m".to_owned(), usage(1, None));
        one.tools.insert("read".to_owned(), usage(2, None));
        add_usage_state(&mut sum, &one);
        add_usage_state(&mut sum, &one);
        assert_eq!(sum.models["p/m"].input, 2);
        assert_eq!(sum.tools["read"].input, 4);
    }
}
