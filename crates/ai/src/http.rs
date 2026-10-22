//! HTTP plumbing shared by the hand-written API implementations: the default client,
//! header assembly, retries with server-requested delays, error bodies and SSE decoding.

use bytes::{Buf, BytesMut};
use futures::{Stream, StreamExt};
use indexmap::IndexMap;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use std::sync::{Mutex, PoisonError};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::error::{Error, Result};
use crate::options::{ProviderResponse, StreamOptions};
use crate::types::ProviderHeaders;

/// Default request timeout, as the vendor SDKs use.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_mins(10);
/// Default client-side retries.
pub const DEFAULT_MAX_RETRIES: u32 = 2;
/// Default cap on server-requested retry delays.
pub const DEFAULT_MAX_RETRY_DELAY: Duration = Duration::from_mins(1);
/// Longest error body kept in error messages.
pub const MAX_PROVIDER_ERROR_BODY_CHARS: usize = 4000;

/// The default client of the runtime that last asked for one. Pooled connections belong to
/// the runtime that opened them and fail once it shuts down, so a client is never shared
/// across runtimes (separate `#[tokio::test]` runtimes, or an app running several).
static DEFAULT_CLIENT: Mutex<Option<(tokio::runtime::Id, reqwest::Client)>> = Mutex::new(None);

fn new_default_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(user_agent())
        .connect_timeout(Duration::from_secs(30))
        .build()
        .unwrap_or_default()
}

/// `delta (<os>; <arch>)`.
pub fn user_agent() -> String {
    format!(
        "delta/{} ({}; {})",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}

/// The client of a request: `options.http_client` or the default client of the current
/// Tokio runtime.
pub fn client(options: &StreamOptions) -> reqwest::Client {
    options.http_client.clone().unwrap_or_else(default_client)
}

fn default_client() -> reqwest::Client {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return new_default_client();
    };
    let id = runtime.id();
    let mut slot = DEFAULT_CLIENT.lock().unwrap_or_else(PoisonError::into_inner);
    match &*slot {
        Some((cached, client)) if *cached == id => client.clone(),
        _ => {
            let client = new_default_client();
            *slot = Some((id, client.clone()));
            client
        }
    }
}

/// Merge header sources in order; later sources win, names compare case-insensitively and
/// `None` values delete earlier ones.
pub fn build_headers<'a>(sources: impl IntoIterator<Item = &'a ProviderHeaders>) -> Result<HeaderMap> {
    let mut merged: IndexMap<String, (String, String)> = IndexMap::new();
    for source in sources {
        for (name, value) in source {
            let key = name.to_ascii_lowercase();
            merged.shift_remove(&key);
            if let Some(value) = value {
                merged.insert(key, (name.clone(), value.clone()));
            }
        }
    }
    let mut headers = HeaderMap::with_capacity(merged.len());
    for (name, value) in merged.into_values() {
        let name =
            HeaderName::from_bytes(name.as_bytes()).map_err(|_| Error::msg(format!("Invalid header name: {name}")))?;
        let value =
            HeaderValue::from_str(&value).map_err(|_| Error::msg(format!("Invalid value for header {name}")))?;
        headers.insert(name, value);
    }
    Ok(headers)
}

/// Header map from `(name, value)` pairs.
pub fn headers_of<const N: usize>(pairs: [(&str, &str); N]) -> ProviderHeaders {
    pairs
        .into_iter()
        .map(|(name, value)| (name.to_owned(), Some(value.to_owned())))
        .collect()
}

/// Status and headers for `on_response`.
pub fn provider_response(response: &reqwest::Response) -> ProviderResponse {
    ProviderResponse {
        status: response.status().as_u16(),
        headers: response
            .headers()
            .iter()
            .filter_map(|(name, value)| Some((name.as_str().to_owned(), value.to_str().ok()?.to_owned())))
            .collect(),
    }
}

/// Whether `c` is whitespace for JavaScript's `String.prototype.trim` (Unicode `White_Space`
/// except U+0085, plus U+FEFF).
pub fn is_js_whitespace(c: char) -> bool {
    (c.is_whitespace() && c != '\u{85}') || c == '\u{feff}'
}

/// JavaScript's `String.prototype.trim`.
pub fn js_trim(text: &str) -> &str {
    text.trim_matches(is_js_whitespace)
}

/// An error response body as upstream keeps it (`extractBody` in `utils/error-body.ts`):
/// trimmed, `None` when empty, truncated to [`MAX_PROVIDER_ERROR_BODY_CHARS`].
pub fn error_body_text(body: &str) -> Option<String> {
    let trimmed = js_trim(body);
    (!trimmed.is_empty()).then(|| truncate_error_text(trimmed, MAX_PROVIDER_ERROR_BODY_CHARS))
}

/// Truncate error text to `max_chars` characters, noting how much was dropped.
pub fn truncate_error_text(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_owned();
    }
    let kept: String = text.chars().take(max_chars).collect();
    format!("{kept}... [truncated {} chars]", count - max_chars)
}

/// Extract a readable message from an error response body: the `error.message` /
/// `message` / `error` field of a JSON body, else the trimmed text.
pub fn error_body_message(body: &str) -> Option<String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
        let candidates = [
            value.pointer("/error/message"),
            value.pointer("/message"),
            value.pointer("/error"),
            value.pointer("/detail"),
            value.pointer("/0/error/message"),
        ];
        for candidate in candidates.into_iter().flatten() {
            if let Some(text) = candidate.as_str().filter(|text| !text.is_empty()) {
                return Some(truncate_error_text(text, MAX_PROVIDER_ERROR_BODY_CHARS));
            }
        }
    }
    Some(truncate_error_text(trimmed, MAX_PROVIDER_ERROR_BODY_CHARS))
}

/// Server-requested retry delay from `retry-after-ms` or `retry-after` (seconds or HTTP date).
pub fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let header = |name: &str| headers.get(name).and_then(|value| value.to_str().ok()).map(str::trim);
    if let Some(ms) = header("retry-after-ms").and_then(|value| value.parse::<f64>().ok())
        && ms.is_finite()
        && ms >= 0.0
    {
        return Some(Duration::from_secs_f64(ms / 1000.0));
    }
    let value = header("retry-after")?;
    if let Ok(seconds) = value.parse::<f64>() {
        return (seconds.is_finite() && seconds >= 0.0).then(|| Duration::from_secs_f64(seconds));
    }
    let date = httpdate_to_unix(value)?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs_f64();
    Some(Duration::from_secs_f64((date - now).max(0.0)))
}

/// Parse an IMF-fixdate (`Sun, 06 Nov 1994 08:49:37 GMT`) to Unix seconds.
pub(crate) fn httpdate_to_unix(value: &str) -> Option<f64> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    if parts.len() != 6 || parts[5] != "GMT" {
        return None;
    }
    let day: i64 = parts[1].parse().ok()?;
    let month = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    .iter()
    .position(|m| *m == parts[2])? as i64
        + 1;
    let year: i64 = parts[3].parse().ok()?;
    let time: Vec<i64> = parts[4].split(':').map(|p| p.parse().ok()).collect::<Option<_>>()?;
    if time.len() != 3 {
        return None;
    }
    // Days from civil (Howard Hinnant's algorithm).
    let (y, m) = if month <= 2 {
        (year - 1, month + 9)
    } else {
        (year, month - 3)
    };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some((days * 86_400 + time[0] * 3600 + time[1] * 60 + time[2]) as f64)
}

/// Whether the vendor SDK retry policy retries this response: `x-should-retry` wins, else
/// 408, 409, 429 and 5xx.
pub fn is_retryable_status(status: u16, headers: &HeaderMap) -> bool {
    match headers.get("x-should-retry").and_then(|value| value.to_str().ok()) {
        Some("true") => return true,
        Some("false") => return false,
        _ => {}
    }
    matches!(status, 408 | 409 | 429) || status >= 500
}

/// Exponential backoff with jitter of the vendor SDK retry policy.
pub(crate) fn backoff(retry_index: u32) -> Duration {
    let base = (0.5 * 2f64.powi(retry_index as i32)).min(8.0);
    // Deterministic jitter source without a RNG dependency.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    let jitter = f64::from(nanos % 1000) / 1000.0 * 0.25;
    Duration::from_secs_f64(base * (1.0 - jitter))
}

/// Sleep for `delay` unless `cancel` fires first (`Error::Aborted`).
pub(crate) async fn sleep_or_cancel(delay: Duration, cancel: Option<&CancellationToken>) -> Result<()> {
    match cancel {
        Some(cancel) => tokio::select! {
            () = cancel.cancelled() => Err(Error::Aborted),
            () = tokio::time::sleep(delay) => Ok(()),
        },
        None => {
            tokio::time::sleep(delay).await;
            Ok(())
        }
    }
}

/// Await `future` unless `cancel` fires first.
pub async fn or_cancel<T>(cancel: Option<&CancellationToken>, future: impl Future<Output = Result<T>>) -> Result<T> {
    match cancel {
        Some(cancel) => tokio::select! {
            () = cancel.cancelled() => Err(Error::Aborted),
            result = future => result,
        },
        None => future.await,
    }
}

/// Send a request built by `build` with the vendor SDK retry policy, interruptible by
/// `options.cancel`. Returns the first successful response; error responses become
/// [`Error::Status`] with the parsed body message, prefixed `"<status> "`, and the raw body
/// ([`error_body_text`]) for APIs that format the message themselves.
pub async fn send_with_retry(
    options: &StreamOptions,
    mut build: impl FnMut() -> Result<reqwest::RequestBuilder>,
) -> Result<reqwest::Response> {
    let max_retries = options.max_retries.unwrap_or(DEFAULT_MAX_RETRIES);
    let max_delay = options.max_retry_delay.unwrap_or(DEFAULT_MAX_RETRY_DELAY);
    let timeout = options.timeout.unwrap_or(DEFAULT_TIMEOUT);
    let cancel = options.cancel.as_ref();
    let mut retry_index = 0;
    loop {
        let request = build()?.timeout(timeout);
        let outcome = or_cancel(cancel, async { request.send().await.map_err(Error::from) }).await;
        let (error, delay) = match outcome {
            Ok(response) if response.status().is_success() => return Ok(response),
            Ok(response) => {
                let status = response.status().as_u16();
                let headers = response.headers().clone();
                let body = or_cancel(cancel, async { response.text().await.map_err(Error::from) }).await?;
                let detail = error_body_message(&body).unwrap_or_else(|| "(no body)".into());
                let server_delay = retry_after(&headers);
                let error = Error::Status {
                    status,
                    message: format!("{status} {detail}"),
                    body: error_body_text(&body),
                    retry_after_ms: server_delay.map(|delay| delay.as_millis() as u64),
                };
                if retry_index >= max_retries || !is_retryable_status(status, &headers) {
                    return Err(error);
                }
                (error, server_delay)
            }
            Err(Error::Aborted) => return Err(Error::Aborted),
            // Connection errors are retried like status-less SDK errors.
            Err(error) if retry_index < max_retries && matches!(&error, Error::Http(e) if !e.is_builder()) => {
                (error, None)
            }
            Err(error) => return Err(error),
        };
        let delay = match delay {
            Some(delay) if !max_delay.is_zero() && delay > max_delay => {
                return Err(Error::msg(format!(
                    "Server requested {}s retry delay (max: {}s). {error}",
                    delay.as_secs_f64().ceil(),
                    max_delay.as_secs_f64().ceil()
                )));
            }
            Some(delay) => delay,
            None => backoff(retry_index),
        };
        retry_index += 1;
        sleep_or_cancel(delay, cancel).await?;
    }
}

/// One server-sent event.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SseEvent {
    /// `event:` field; empty when absent.
    pub event: String,
    /// `data:` lines joined with `\n`.
    pub data: String,
    pub id: Option<String>,
}

/// Incremental SSE decoder per the WHATWG event-stream format. Lines end in `\n`, `\r\n`
/// or `\r`; bytes are buffered until a line is complete, so UTF-8 sequences split across
/// chunks decode correctly.
#[derive(Debug, Default)]
pub struct SseDecoder {
    buffer: BytesMut,
    event: String,
    data: String,
    has_data: bool,
    id: Option<String>,
    /// A `\r` ended the previous chunk; a leading `\n` in the next chunk belongs to it.
    pending_cr: bool,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed bytes and collect the events they complete.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        let mut chunk = chunk;
        if self.pending_cr && chunk.first() == Some(&b'\n') {
            chunk = &chunk[1..];
        }
        self.pending_cr = false;
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(end) = self.buffer.iter().position(|b| *b == b'\n' || *b == b'\r') {
            let line = self.buffer.split_to(end);
            let terminator = self.buffer[0];
            self.buffer.advance(1);
            if terminator == b'\r' {
                if self.buffer.first() == Some(&b'\n') {
                    self.buffer.advance(1);
                } else if self.buffer.is_empty() {
                    self.pending_cr = true;
                }
            }
            let line = String::from_utf8_lossy(&line);
            if let Some(event) = self.line(&line) {
                events.push(event);
            }
        }
        events
    }

    /// Flush a final event that was not followed by a blank line.
    pub fn finish(&mut self) -> Option<SseEvent> {
        if !self.buffer.is_empty() {
            let line = String::from_utf8_lossy(&self.buffer.split()).into_owned();
            if let Some(event) = self.line(&line) {
                return Some(event);
            }
        }
        self.dispatch()
    }

    fn line(&mut self, line: &str) -> Option<SseEvent> {
        if line.is_empty() {
            return self.dispatch();
        }
        if line.starts_with(':') {
            return None;
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "event" => value.clone_into(&mut self.event),
            "data" => {
                if self.has_data {
                    self.data.push('\n');
                }
                self.data.push_str(value);
                self.has_data = true;
            }
            "id" if !value.contains('\0') => self.id = Some(value.to_owned()),
            _ => {}
        }
        None
    }

    fn dispatch(&mut self) -> Option<SseEvent> {
        let event = std::mem::take(&mut self.event);
        if !self.has_data {
            return None;
        }
        self.has_data = false;
        Some(SseEvent {
            event,
            data: std::mem::take(&mut self.data),
            id: self.id.clone(),
        })
    }
}

/// Decode a response body into SSE events. Ends after the body ends; read errors and
/// cancellation surface as `Err`.
pub fn sse_events(
    response: reqwest::Response,
    cancel: Option<CancellationToken>,
) -> impl Stream<Item = Result<SseEvent>> + Send {
    let state = (
        response.bytes_stream(),
        SseDecoder::new(),
        std::collections::VecDeque::new(),
        false,
        cancel,
    );
    futures::stream::unfold(
        state,
        |(mut body, mut decoder, mut queue, mut ended, cancel)| async move {
            loop {
                if let Some(event) = queue.pop_front() {
                    return Some((Ok(event), (body, decoder, queue, ended, cancel)));
                }
                if ended {
                    return None;
                }
                let next = match &cancel {
                    Some(token) => tokio::select! {
                        () = token.cancelled() => {
                            return Some((Err(Error::Aborted), (body, decoder, queue, true, cancel)));
                        }
                        next = body.next() => next,
                    },
                    None => body.next().await,
                };
                match next {
                    Some(Ok(chunk)) => queue.extend(decoder.push(&chunk)),
                    Some(Err(error)) => return Some((Err(Error::from(error)), (body, decoder, queue, true, cancel))),
                    None => {
                        ended = true;
                        queue.extend(decoder.finish());
                    }
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_handles_split_lines_crlf_and_multiline_data() {
        let mut decoder = SseDecoder::new();
        let mut events = decoder.push(b"event: message_start\r");
        events.extend(decoder.push(b"\ndata: {\"a\":"));
        events.extend(decoder.push(b"1}\r\n\r\n: ping\n\ndata: x\ndata: y\n\n"));
        events.extend(decoder.push("data: \u{1F600}".as_bytes().split_at(8).0));
        events.extend(decoder.push(&"data: \u{1F600}".as_bytes()[8..]));
        events.extend(decoder.finish());
        assert_eq!(
            events,
            vec![
                SseEvent {
                    event: "message_start".into(),
                    data: "{\"a\":1}".into(),
                    id: None
                },
                SseEvent {
                    event: String::new(),
                    data: "x\ny".into(),
                    id: None
                },
                SseEvent {
                    event: String::new(),
                    data: "\u{1F600}".into(),
                    id: None
                },
            ]
        );
    }

    #[test]
    fn error_bodies_and_retry_headers() {
        assert_eq!(
            error_body_message(r#"{"error":{"message":"bad key"}}"#).as_deref(),
            Some("bad key")
        );
        assert_eq!(error_body_message("  plain  ").as_deref(), Some("plain"));
        assert_eq!(error_body_message(""), None);
        // The raw body is kept trimmed (JavaScript whitespace) and capped as upstream does.
        assert_eq!(error_body_text("\u{feff} {\"a\":1}\n").as_deref(), Some("{\"a\":1}"));
        assert_eq!(error_body_text(" \n "), None);
        let long = "x".repeat(MAX_PROVIDER_ERROR_BODY_CHARS + 50);
        assert_eq!(
            error_body_text(&long),
            Some(format!(
                "{}... [truncated 50 chars]",
                "x".repeat(MAX_PROVIDER_ERROR_BODY_CHARS)
            ))
        );
        let mut headers = HeaderMap::new();
        headers.insert("retry-after", HeaderValue::from_static("2"));
        assert_eq!(retry_after(&headers), Some(Duration::from_secs(2)));
        headers.insert("retry-after-ms", HeaderValue::from_static("150"));
        assert_eq!(retry_after(&headers), Some(Duration::from_millis(150)));
        assert_eq!(httpdate_to_unix("Sun, 06 Nov 1994 08:49:37 GMT"), Some(784_111_777.0));
        assert!(is_retryable_status(529, &HeaderMap::new()));
        assert!(!is_retryable_status(400, &HeaderMap::new()));
    }

    #[test]
    fn header_sources_merge_and_delete() {
        let a = headers_of([("X-Api-Key", "1"), ("Accept", "a")]);
        let b: ProviderHeaders = [("x-api-key".to_owned(), None)].into_iter().collect();
        let headers = build_headers([&a, &b]).unwrap();
        assert_eq!(headers.len(), 1);
        assert_eq!(headers.get("accept").unwrap(), "a");
    }
}
