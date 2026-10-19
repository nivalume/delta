//! RFC 8628 device authorization polling. Port of upstream `auth/oauth/device-code.ts`.
//!
//! The flow-specific part (the token request and the mapping of its response) is the
//! `poll` closure; this module owns the timing: the initial interval, the 1 s floor,
//! `slow_down` handling, the expiry deadline and cancellation.

use std::future::Future;
use std::time::Duration;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::error::{ModelsError, ModelsErrorCode};

/// Error text when the login is cancelled.
pub const CANCEL_MESSAGE: &str = "Login cancelled";
/// Error text when the device code expires before the user approves it.
pub const TIMEOUT_MESSAGE: &str = "Device flow timed out";
/// Error text when the device code expires after at least one `slow_down` response.
pub const SLOW_DOWN_TIMEOUT_MESSAGE: &str = "Device flow timed out after one or more slow_down responses. This is often caused by clock drift in WSL or VM environments. Please sync or restart the VM clock and try again.";

const MINIMUM_INTERVAL_MS: u64 = 1000;
/// RFC 8628 section 3.2: if the authorization server omits `interval`, the client must use 5 seconds.
const DEFAULT_POLL_INTERVAL_SECONDS: f64 = 5.0;
/// RFC 8628 section 3.5: `slow_down` means the polling interval must increase by 5 seconds.
const SLOW_DOWN_INTERVAL_INCREMENT_MS: u64 = 5000;

/// Outcome of one token poll (upstream `OAuthDeviceCodePollResult<T>`).
#[derive(Clone, Debug, PartialEq)]
pub enum OAuthDeviceCodePollResult<T> {
    /// `authorization_pending`: poll again after the current interval.
    Pending,
    /// `slow_down`: use the server's new interval when it sends a positive one, otherwise
    /// add 5 seconds to the current interval.
    SlowDown { interval_seconds: Option<f64> },
    /// A terminal error; the flow fails with `message`.
    Failed { message: String },
    /// The user approved; the flow returns `value`.
    Complete(T),
}

/// Timing options of [`poll_oauth_device_code_flow`] (upstream `OAuthDeviceCodePollOptions`
/// without `poll`, which is a separate argument here).
#[derive(Clone, Debug, Default)]
pub struct OAuthDeviceCodePollOptions {
    /// The device authorization response's `interval`. Default 5 s; never below 1 s.
    pub interval_seconds: Option<f64>,
    /// The device authorization response's `expires_in`. `None` polls until cancelled.
    pub expires_in_seconds: Option<f64>,
    /// Sleep one interval before the first poll instead of polling at once.
    pub wait_before_first_poll: bool,
    /// Cancels the flow with [`CANCEL_MESSAGE`], also during a sleep.
    pub cancel: CancellationToken,
}

fn cancelled() -> ModelsError {
    ModelsError::new(ModelsErrorCode::Aborted, CANCEL_MESSAGE)
}

/// Sleep for `duration` unless `cancel` fires first; then fail with `cancel_message`
/// (code [`ModelsErrorCode::Aborted`]). Fails at once when `cancel` already fired.
pub async fn abortable_sleep(
    duration: Duration,
    cancel: &CancellationToken,
    cancel_message: &str,
) -> Result<(), ModelsError> {
    if cancel.is_cancelled() {
        return Err(ModelsError::new(ModelsErrorCode::Aborted, cancel_message));
    }
    tokio::select! {
        () = cancel.cancelled() => Err(ModelsError::new(ModelsErrorCode::Aborted, cancel_message)),
        () = tokio::time::sleep(duration) => Ok(()),
    }
}

fn millis(seconds: f64) -> u64 {
    // `Math.floor(seconds * 1000)`, saturating for huge values.
    (seconds * 1000.0).floor().max(0.0) as u64
}

/// Poll a device-code token endpoint until `poll` completes or fails, the device code
/// expires, or `options.cancel` fires.
///
/// Timing follows RFC 8628 as upstream does:
/// - the interval starts at `interval_seconds` (default 5 s) with a floor of 1 s;
/// - the first poll runs at once unless `wait_before_first_poll` is set;
/// - `SlowDown { interval_seconds: Some(n) }` with a finite `n > 0` sets the interval to `n`
///   seconds (floor 1 s); otherwise the interval grows by 5 s;
/// - sleeps never pass the deadline (`expires_in_seconds` after the call).
///
/// Errors: `Failed { message }` → `message` (code `OAuth`); expiry → [`TIMEOUT_MESSAGE`],
/// or [`SLOW_DOWN_TIMEOUT_MESSAGE`] after a `slow_down` (code `OAuth`); cancellation →
/// [`CANCEL_MESSAGE`] (code `Aborted`). An error returned by `poll` itself is passed through.
///
/// Time is [`tokio::time::Instant`], so tests can drive the flow with paused time.
pub async fn poll_oauth_device_code_flow<T, F, Fut>(
    options: OAuthDeviceCodePollOptions,
    mut poll: F,
) -> Result<T, ModelsError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<OAuthDeviceCodePollResult<T>, ModelsError>>,
{
    let start = Instant::now();
    let deadline = options
        .expires_in_seconds
        .map(|seconds| start + Duration::from_millis(millis(seconds)));
    let remaining =
        |deadline: Option<Instant>| deadline.map(|deadline| deadline.saturating_duration_since(Instant::now()));
    let mut interval_ms = MINIMUM_INTERVAL_MS.max(millis(
        options.interval_seconds.unwrap_or(DEFAULT_POLL_INTERVAL_SECONDS),
    ));

    let mut slow_down_responses = 0u32;
    if options.wait_before_first_poll {
        let wait = match remaining(deadline) {
            Some(remaining) => remaining.min(Duration::from_millis(interval_ms)),
            None => Duration::from_millis(interval_ms),
        };
        if !wait.is_zero() {
            abortable_sleep(wait, &options.cancel, CANCEL_MESSAGE).await?;
        }
    }

    while deadline.is_none_or(|deadline| Instant::now() < deadline) {
        if options.cancel.is_cancelled() {
            return Err(cancelled());
        }

        match poll().await? {
            OAuthDeviceCodePollResult::Complete(value) => return Ok(value),
            OAuthDeviceCodePollResult::Failed { message } => {
                return Err(ModelsError::new(ModelsErrorCode::OAuth, message));
            }
            OAuthDeviceCodePollResult::SlowDown { interval_seconds } => {
                slow_down_responses += 1;
                // Use the server-provided interval when given (GitHub reports the new required
                // minimum in `interval`); trusting only a client-tracked value risks polling
                // early forever under WSL/VM clock drift. Otherwise apply RFC 8628 section 3.5.
                interval_ms = match interval_seconds {
                    Some(seconds) if seconds.is_finite() && seconds > 0.0 => MINIMUM_INTERVAL_MS.max(millis(seconds)),
                    _ => MINIMUM_INTERVAL_MS.max(interval_ms + SLOW_DOWN_INTERVAL_INCREMENT_MS),
                };
            }
            OAuthDeviceCodePollResult::Pending => {}
        }

        let wait = match remaining(deadline) {
            Some(remaining) if remaining.is_zero() => break,
            Some(remaining) => remaining.min(Duration::from_millis(interval_ms)),
            None => Duration::from_millis(interval_ms),
        };
        abortable_sleep(wait, &options.cancel, CANCEL_MESSAGE).await?;
    }

    Err(ModelsError::new(
        ModelsErrorCode::OAuth,
        if slow_down_responses > 0 {
            SLOW_DOWN_TIMEOUT_MESSAGE
        } else {
            TIMEOUT_MESSAGE
        },
    ))
}
