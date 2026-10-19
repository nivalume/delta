//! Helpers that map [`SimpleStreamOptions`] onto API-native options.

use crate::estimate::estimate_context_tokens;
use crate::model::{Model, ModelThinkingLevel, ThinkingLevel};
use crate::options::{SimpleStreamOptions, StreamOptions, ThinkingBudgets};
use crate::types::{JsonObject, TranscriptContext};

const CONTEXT_SAFETY_TOKENS: u64 = 4096;
const MIN_MAX_TOKENS: u64 = 1;

/// Tokens always left for the answer when a thinking budget shares the response ceiling.
pub const MIN_ANSWER_TOKENS: u64 = 1024;

/// Default thinking budgets per level.
pub const DEFAULT_THINKING_BUDGETS: ThinkingBudgets = ThinkingBudgets {
    minimal: Some(1024),
    low: Some(2048),
    medium: Some(8192),
    high: Some(16384),
};

/// Cap `max_tokens` to what fits in the context window after the estimated context.
pub fn clamp_max_tokens_to_context(model: &Model, context: &TranscriptContext, max_tokens: u64) -> u64 {
    if model.context_window == 0 {
        return max_tokens.max(MIN_MAX_TOKENS);
    }
    let used = estimate_context_tokens(context.messages()).tokens + CONTEXT_SAFETY_TOKENS;
    let available = model.context_window.saturating_sub(used);
    max_tokens.min(available.max(MIN_MAX_TOKENS))
}

/// Model sampling defaults, then thinking-level overrides, then request parameters.
pub fn resolve_sampling_params(
    model: &Model,
    thinking_level: ModelThinkingLevel,
    request: Option<&JsonObject>,
) -> Option<JsonObject> {
    let level = model.clamp_thinking_level(thinking_level);
    let by_level = model
        .sampling_params_by_thinking_level
        .as_ref()
        .and_then(|map| map.get(&level));
    if model.sampling_params.is_none() && by_level.is_none() && request.is_none() {
        return None;
    }
    let mut merged = JsonObject::new();
    for source in [model.sampling_params.as_ref(), by_level, request]
        .into_iter()
        .flatten()
    {
        for (key, value) in source {
            merged.insert(key.clone(), value.clone());
        }
    }
    Some(merged)
}

/// The shared [`StreamOptions`] of a simple request: sampling parameters resolved for the
/// reasoning level and `max_tokens` clamped to the context window.
pub fn build_base_options(model: &Model, context: &TranscriptContext, options: &SimpleStreamOptions) -> StreamOptions {
    let level = options
        .reasoning
        .map_or(ModelThinkingLevel::Off, ModelThinkingLevel::from);
    let mut base = options.base.clone();
    base.sampling_params = resolve_sampling_params(model, level, options.base.sampling_params.as_ref());
    base.max_tokens = Some(clamp_max_tokens_to_context(
        model,
        context,
        options.base.max_tokens.unwrap_or(model.max_tokens),
    ));
    base.api = None;
    base
}

/// `xhigh` and `max` become `high` for budget-based APIs.
pub fn clamp_reasoning(level: ThinkingLevel) -> ThinkingLevel {
    match level {
        ThinkingLevel::Xhigh | ThinkingLevel::Max => ThinkingLevel::High,
        other => other,
    }
}

/// Token budget for a reasoning level.
pub fn thinking_budget_for_level(level: ThinkingLevel, custom: Option<&ThinkingBudgets>) -> u64 {
    let pick = |budgets: &ThinkingBudgets| match clamp_reasoning(level) {
        ThinkingLevel::Minimal => budgets.minimal,
        ThinkingLevel::Low => budgets.low,
        ThinkingLevel::Medium => budgets.medium,
        _ => budgets.high,
    };
    custom
        .and_then(pick)
        .or_else(|| pick(&DEFAULT_THINKING_BUDGETS))
        .unwrap_or(1024)
}

/// Cap a thinking budget so at least [`MIN_ANSWER_TOKENS`] remain under a shared ceiling.
pub fn clamp_thinking_budget_to_answer_room(budget: u64, ceiling: u64) -> u64 {
    budget.min(ceiling.saturating_sub(MIN_ANSWER_TOKENS))
}

/// Output ceiling and thinking budget for budget-based thinking. `base_max_tokens` of `None`
/// means no explicit caller cap: use the model cap and fit thinking inside it.
pub fn adjust_max_tokens_for_thinking(
    base_max_tokens: Option<u64>,
    model_max_tokens: u64,
    level: ThinkingLevel,
    custom: Option<&ThinkingBudgets>,
) -> (u64, u64) {
    let mut budget = thinking_budget_for_level(level, custom);
    let max_tokens = base_max_tokens.map_or(model_max_tokens, |base| (base + budget).min(model_max_tokens));
    if max_tokens <= budget {
        budget = clamp_thinking_budget_to_answer_room(budget, max_tokens);
    }
    (max_tokens, budget)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_budget_fits_ceiling() {
        assert_eq!(
            adjust_max_tokens_for_thinking(Some(1000), 64000, ThinkingLevel::Medium, None),
            (9192, 8192)
        );
        assert_eq!(
            adjust_max_tokens_for_thinking(None, 4000, ThinkingLevel::High, None),
            (4000, 2976)
        );
        assert_eq!(thinking_budget_for_level(ThinkingLevel::Max, None), 16384);
        let custom = ThinkingBudgets {
            low: Some(5),
            ..ThinkingBudgets::default()
        };
        assert_eq!(thinking_budget_for_level(ThinkingLevel::Low, Some(&custom)), 5);
        assert_eq!(thinking_budget_for_level(ThinkingLevel::Medium, Some(&custom)), 8192);
    }
}
