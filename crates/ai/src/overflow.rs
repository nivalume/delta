//! Context overflow detection: whether a failed or truncated response means the request did
//! not fit the model's context window.
//!
//! Port of upstream `utils/overflow.ts`. Providers word the error differently; the patterns
//! below cover the ones pi has seen, and two usage-based checks cover providers that overflow
//! silently.

use std::sync::LazyLock;

use regex::{Regex, RegexSet};

use crate::types::{AssistantMessage, StopReason};

/// Error texts of a request that exceeded the context window or the request size limit.
///
/// - Anthropic: "prompt is too long: 213462 tokens > 200000 maximum", `request_too_large`
/// - OpenAI: "Your input exceeds the context window of this model"
/// - OpenAI-compatible / LiteLLM: "exceeds the model's maximum context length of 131072 tokens",
///   "exceeds model's maximum context length (262144)"
/// - Google: "The input token count (1196265) exceeds the maximum number of tokens allowed"
/// - xAI: "This model's maximum prompt length is 131072 but the request contains 537812 tokens"
/// - Groq: "Please reduce the length of the messages or completion"
/// - OpenRouter: "maximum context length is X tokens", "exceeds the maximum allowed input length of Y tokens"
/// - Together AI: "The input (X tokens) is longer than the model's context length (Y tokens)."
/// - GitHub Copilot: "prompt token count of X exceeds the limit of Y"
/// - llama.cpp, LM Studio, MiniMax, Kimi, Mistral, DS4, z.ai, Ollama, DashScope: see each line
/// - Any provider: HTTP 413, as a raw status or behind the "<provider> API error (413):" prefix
static OVERFLOW: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new([
        r"(?i)prompt (?:is )?too long",
        r"(?i)prompt exceeds max length",
        r"(?i)request_too_large",
        r"(?i)input is too long for requested model",
        r"(?i)exceeds the context window",
        r"(?i)exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))",
        r"(?i)input token count.*exceeds the maximum",
        r"(?i)maximum prompt length is \d+",
        r"(?i)reduce the length of the messages",
        r"(?i)maximum context length is \d+ tokens",
        r"(?i)exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",
        r"(?i)input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
        r"(?i)exceeds the limit of \d+",
        r"(?i)exceeds the available context size",
        r"(?i)greater than the context length",
        r"(?i)context window exceeds limit",
        r"(?i)exceeded model token limit",
        r"(?i)too large for model with \d+ maximum context length",
        r"(?i)prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?",
        r"(?i)model_context_window_exceeded",
        r"(?i)prompt too long; exceeded (?:max )?context length",
        r"(?i)range of input length should be",
        r"(?i)context[_ ]length[_ ]exceeded",
        r"(?i)too many tokens",
        r"(?i)token limit exceeded",
        r"(?i)^(?:[\w -]+ API error \(413\):|413(?:\s|:|$))",
    ])
    .expect("overflow patterns compile")
});

/// Cerebras reports an overflow as a bodyless 400 or 413.
static CEREBRAS_BODYLESS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^4(?:00|13)\s*(?:status code)?\s*\(no body\)").expect("pattern compiles"));

/// Errors that match an overflow pattern but are not overflows (Bedrock's "ThrottlingException:
/// Too many tokens, please wait", rate limits).
static NOT_OVERFLOW: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new([
        r"(?i)^(Throttling error|Service unavailable):",
        r"(?i)rate limit",
        r"(?i)too many requests",
    ])
    .expect("non-overflow patterns compile")
});

/// Whether `message` means the request did not fit the context window.
///
/// - A failed response whose error text matches an overflow pattern (and no rate-limit
///   pattern).
/// - With `context_window`: a successful response whose input exceeds the window (z.ai
///   accepts some overflows silently), or a length stop with no output whose input fills the
///   window (Xiaomi MiMo truncates the input instead of failing).
pub fn is_context_overflow(message: &AssistantMessage, context_window: Option<u64>) -> bool {
    if message.stop_reason == StopReason::Error
        && let Some(error) = message.error_message.as_deref()
        && !NOT_OVERFLOW.is_match(error)
        && (OVERFLOW.is_match(error) || (message.provider == "cerebras" && CEREBRAS_BODYLESS.is_match(error)))
    {
        return true;
    }
    let Some(window) = context_window.filter(|&window| window > 0) else {
        return false;
    };
    let input = message.usage.input + message.usage.cache_read;
    match message.stop_reason {
        StopReason::Stop => input > window,
        StopReason::Length => message.usage.output == 0 && input as f64 >= window as f64 * 0.99,
        _ => false,
    }
}

/// Whether a length stop ended below the intended output limit, so the response may have been
/// cut by context pressure and one compact-and-retry attempt is worthwhile.
/// `desired_max_output` is the limit before any clamping to the context.
pub fn is_recoverable_length(message: &AssistantMessage, desired_max_output: u64) -> bool {
    message.stop_reason == StopReason::Length && desired_max_output > 0 && message.usage.output < desired_max_output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error(text: &str, provider: &str) -> AssistantMessage {
        let mut message = AssistantMessage::empty("openai-completions", provider, "qwen3.5:35b");
        message.stop_reason = StopReason::Error;
        message.error_message = Some(text.to_owned());
        message
    }

    fn length_stop(input: u64, cache_read: u64, output: u64) -> AssistantMessage {
        let mut message = AssistantMessage::empty("openai-completions", "test-provider", "test-model");
        message.stop_reason = StopReason::Length;
        message.usage.input = input;
        message.usage.cache_read = cache_read;
        message.usage.output = output;
        message
    }

    #[test]
    fn request_size_overflow_is_provider_independent() {
        for text in [
            r#"github-copilot API error (413): {"message":"failed to parse request","code":""}"#,
            "custom-provider API error (413): Payload Too Large",
            r#"413 {"error":{"message":"failed to parse request","code":""}}"#,
            "413",
        ] {
            assert!(is_context_overflow(&error(text, "ollama"), Some(1_000_000)), "{text}");
            assert!(is_context_overflow(&error(text, "ollama"), None), "{text}");
        }
        for text in [
            r#"github-copilot API error (400): {"message":"failed to parse request","code":""}"#,
            "custom-provider API error (429): Too many requests",
            r#"github-copilot API error (400): {"message":"failed to parse request: invalid value 413"}"#,
            "custom-provider API error (400): upstream mentioned OpenAI API error (413): invalid input",
            "custom-provider API error (4130): invalid input",
            "4130 failed to parse request",
        ] {
            assert!(!is_context_overflow(&error(text, "ollama"), None), "{text}");
        }
    }

    #[test]
    fn only_failed_responses_match_error_text() {
        let mut message = error(
            r#"OpenAI API error (413): {"message":"failed to parse request"}"#,
            "openai",
        );
        message.stop_reason = StopReason::Aborted;
        assert!(!is_context_overflow(&message, None));
        message.stop_reason = StopReason::Stop;
        assert!(!is_context_overflow(&message, None));
    }

    #[test]
    fn provider_wordings() {
        for text in [
            "prompt too long; exceeded max context length by 100 tokens",
            r#"{"code":"1261","message":"Prompt too long"}"#,
            r#"{"code":"1261","message":"Prompt exceeds max length"}"#,
            "The input (265330 tokens) is longer than the model's context length (262144 tokens).",
            "Requested token count exceeds the model's maximum context length of 131072 tokens",
            "Input length (265330) exceeds model's maximum context length (262144).",
            "Input length 300000 exceeds the maximum allowed input length of 262144 tokens.",
            "Prompt has 300,000 tokens, but the configured context size is 262,144 tokens",
        ] {
            assert!(is_context_overflow(&error(text, "ollama"), None), "{text}");
        }
        assert!(!is_context_overflow(&error("model 'llama3' not found", "ollama"), None));
    }

    #[test]
    fn cerebras_bodyless_errors() {
        assert!(is_context_overflow(
            &error("400 status code (no body)", "cerebras"),
            None
        ));
        assert!(!is_context_overflow(
            &error("400 status code (no body)", "openai"),
            None
        ));
    }

    #[test]
    fn rate_limits_are_not_overflows() {
        for text in [
            "Throttling error: Too many tokens, please wait before trying again.",
            "Service unavailable: The service is temporarily unavailable.",
            "Rate limit reached: too many tokens per minute",
            "Too many requests. Please slow down.",
        ] {
            assert!(
                !is_context_overflow(&error(text, "amazon-bedrock"), Some(200_000)),
                "{text}"
            );
        }
    }

    #[test]
    fn silent_and_length_overflows() {
        let mut silent = AssistantMessage::empty("openai-completions", "zai", "glm");
        silent.stop_reason = StopReason::Stop;
        silent.usage.input = 150_000;
        silent.usage.cache_read = 60_000;
        assert!(is_context_overflow(&silent, Some(200_000)));
        assert!(!is_context_overflow(&silent, None));

        assert!(is_context_overflow(&length_stop(58, 1_048_512, 0), Some(1_048_576)));
        assert!(!is_context_overflow(&length_stop(1000, 0, 4096), Some(200_000)));
        assert!(!is_context_overflow(&length_stop(100, 0, 0), Some(200_000)));
    }

    #[test]
    fn recoverable_length_stops() {
        assert!(is_recoverable_length(&length_stop(3, 253_584, 16), 128_000));
        assert!(!is_recoverable_length(&length_stop(4062, 0, 1024), 1024));
        assert!(is_recoverable_length(&length_stop(100, 0, 0), 128_000));
    }
}
