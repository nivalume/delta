//! Retry classification for failed responses and the backoff between attempts.
//!
//! Port of upstream `utils/retry.ts` without `retryAssistantCall`: callers own the retry loop
//! (the durable harness sleeps between attempts as a task state, not in memory). Handle
//! context overflow first ([`crate::overflow::is_context_overflow`]); it is not retryable as is.

use std::sync::LazyLock;
use std::time::Duration;

use regex::Regex;

use crate::types::{AssistantMessage, StopReason};

/// Account limits that look like throttling but do not clear in seconds.
static NOT_RETRYABLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        "(?i){}",
        [
            "GoUsageLimitError",
            "FreeUsageLimitError",
            "Monthly usage limit reached",
            "available balance",
            "insufficient_quota",
            "out of budget",
            "quota exceeded",
            "billing",
            "subscription_sharing_usage_limit_exceeded",
        ]
        .join("|")
    ))
    .expect("non-retryable pattern compiles")
});

/// Transient provider, gateway and transport failures.
static RETRYABLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        "(?i){}",
        [
            // Provider load, HTTP status and server-side failures.
            "overloaded",
            "server_busy",
            "servers are currently busy",
            "currently experiencing high demand",
            "model is at capacity",
            "rate.?limit",
            "too many requests",
            "429",
            "500",
            "502",
            "503",
            "504",
            "520",
            "524",
            "service.?(?:is.?)?(?:temporarily.?)?unavailable",
            "server.?error",
            "internal.?error",
            // Gateways relaying an upstream failure.
            "provider.?returned.?(?:an.?)?error",
            "exceeded request buffer limit while retrying upstream",
            "unable to forward upstream",
            "upstream.?request.?failed",
            "upstream_stream_read_error",
            "stream was interrupted",
            "http/?2 stream failed",
            // Network and transport.
            "network.?error",
            "connection.?error",
            "connection.?refused",
            "connection.?lost",
            "other side closed",
            "fetch failed",
            r"request to \S+ failed, reason:",
            "getaddrinfo",
            "ENOTFOUND",
            "EAI_AGAIN",
            "upstream.?connect",
            "reset before headers",
            "socket hang up",
            "socket connection was closed",
            "timed? out",
            "timeout",
            "terminated",
            "websocket.?closed",
            "websocket.?error",
            // Streams that ended early.
            "ended without",
            "stream ended before message_stop",
            "stream ended before a terminal response event",
            "http2 request did not get a response",
            "pending stream has been canceled",
            // A provider-requested delay over the cap goes to the caller's policy.
            "retry delay",
            // Explicit retry guidance.
            "you can retry your request",
            "try your request again",
            "please retry your request",
            "ResourceExhausted",
            "subscription_sharing_usage_unavailable",
            "subscription_sharing_user_unavailable",
        ]
        .join("|")
    ))
    .expect("retryable pattern compiles")
});

/// Whether a failed response looks like a transient provider or transport error, so the
/// request is worth sending again. Quota and billing exhaustion is not retryable.
pub fn is_retryable_assistant_error(message: &AssistantMessage) -> bool {
    if message.stop_reason != StopReason::Error {
        return false;
    }
    let Some(error) = message.error_message.as_deref().filter(|error| !error.is_empty()) else {
        return false;
    };
    !NOT_RETRYABLE.is_match(error) && RETRYABLE.is_match(error)
}

/// The longest delay [`RetryPolicy::delay`] returns unless the policy sets its own cap.
pub const DEFAULT_MAX_RETRY_DELAY: Duration = Duration::from_mins(1);

/// Bounded retries with exponential backoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    pub enabled: bool,
    /// Retries after the first attempt; 0 means none.
    pub max_retries: u32,
    /// Delay before the first retry; each later retry doubles it.
    pub base_delay: Duration,
    /// Cap of each delay; [`DEFAULT_MAX_RETRY_DELAY`] when `None`.
    pub max_delay: Option<Duration>,
}

impl RetryPolicy {
    /// The delay before retry number `attempt` (1-based): `base_delay * 2^(attempt - 1)`,
    /// capped.
    pub fn delay(&self, attempt: u32) -> Duration {
        let cap = self.max_delay.unwrap_or(DEFAULT_MAX_RETRY_DELAY);
        let factor = 2u32.checked_pow(attempt.saturating_sub(1)).unwrap_or(u32::MAX);
        self.base_delay.checked_mul(factor).unwrap_or(Duration::MAX).min(cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failed(text: &str) -> AssistantMessage {
        let mut message = AssistantMessage::empty("faux", "faux", "faux");
        message.stop_reason = StopReason::Error;
        message.error_message = Some(text.to_owned());
        message
    }

    #[test]
    fn transient_errors_are_retryable() {
        for text in [
            "An error occurred while processing your request. You can retry your request, or contact us.",
            r#"{"message":"The system encountered an unexpected error during processing. Try your request again."}"#,
            "ResourceExhausted: Worker local total request limit reached (288/48)",
            "The socket connection was closed unexpectedly.",
            "Error: exceeded request buffer limit while retrying upstream",
            "The pending stream has been canceled (caused by: getaddrinfo ENOTFOUND bedrock-runtime.us-east-1.amazonaws.com)",
            "connect ENOTFOUND api.example.com",
            "EAI_AGAIN api.example.com",
            "The pending stream has been canceled",
            "The service is temporarily unavailable. Please retry later.",
            "Service unavailable: The service is temporarily unavailable.",
            "request to https://oauth2.googleapis.com/token failed, reason:",
            "unknown: Unable to forward upstream stream",
            "upstream_error: Upstream request failed",
            "upstream_stream_read_error: Upstream response stream was interrupted",
            "Upstream HTTP/2 stream failed",
            "OpenAI Responses stream ended before a terminal response event",
            "The system is currently experiencing high demand and cannot process your request.",
            "subscription_sharing_usage_unavailable: Usage cannot be checked.",
            "subscription_sharing_user_unavailable: User cannot be loaded.",
            "overloaded_error",
            "520 status code (no body)",
            "524 status code (no body)",
        ] {
            assert!(is_retryable_assistant_error(&failed(text)), "{text}");
        }
    }

    #[test]
    fn limits_and_successes_are_not_retryable() {
        assert!(!is_retryable_assistant_error(&failed("429 quota exceeded")));
        assert!(!is_retryable_assistant_error(&failed(
            r#"OpenAI API error (429): {"code":"subscription_sharing_usage_limit_exceeded","message":"Usage limit reached."}"#
        )));
        assert!(!is_retryable_assistant_error(&AssistantMessage::empty(
            "faux", "faux", "faux"
        )));
        assert!(!is_retryable_assistant_error(&failed("")));
    }

    #[test]
    fn delays_double_and_are_capped() {
        let policy = RetryPolicy {
            enabled: true,
            max_retries: 3,
            base_delay: Duration::from_secs(2),
            max_delay: None,
        };
        assert_eq!(policy.delay(1), Duration::from_secs(2));
        assert_eq!(policy.delay(3), Duration::from_secs(8));
        assert_eq!(policy.delay(6), Duration::from_mins(1));
        assert_eq!(policy.delay(200), Duration::from_mins(1));
        let capped = RetryPolicy {
            max_delay: Some(Duration::from_secs(5)),
            ..policy
        };
        assert_eq!(capped.delay(5), Duration::from_secs(5));
        let zero = RetryPolicy {
            max_delay: Some(Duration::ZERO),
            ..policy
        };
        assert_eq!(zero.delay(5), Duration::ZERO);
    }
}
