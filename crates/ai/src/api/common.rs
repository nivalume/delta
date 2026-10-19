//! Plumbing shared by the API implementations: the stream task, request callbacks and
//! small conversions.

use serde_json::Value;
use std::future::Future;
use tokio_util::sync::CancellationToken;

use crate::error::{Error, Result};
use crate::events::{AssistantMessageEventStream, MessageBuilder};
use crate::http;
use crate::model::{Model, calculate_cost};
use crate::options::StreamOptions;
use crate::types::{AssistantMessage, StopReason, Usage};

/// A fresh, pending output message for `model`.
pub fn new_output(model: &Model) -> AssistantMessage {
    AssistantMessage::empty(&model.api, &model.provider, &model.id)
}

/// Run an API request on a Tokio task and return its stream.
///
/// `run` receives the [`MessageBuilder`] and hands it back with its result. The task then
/// ends the stream the way every API does:
/// - `Err`, or cancellation: an `error` event (`aborted` when cancelled) carrying any
///   partial content;
/// - `Ok` with stop reason `error`/`aborted`: an `error` event with the message's
///   `error_message`;
/// - otherwise `done`.
///
/// Implementations return `Err(Error::Aborted)` when they notice the consumer dropped the
/// stream ([`MessageBuilder::is_closed`]); nothing is sent then.
pub fn spawn_stream<F, Fut>(model: &Model, cancel: Option<CancellationToken>, run: F) -> AssistantMessageEventStream
where
    F: FnOnce(MessageBuilder) -> Fut + Send + 'static,
    Fut: Future<Output = (MessageBuilder, Result<()>)> + Send + 'static,
{
    let (sender, stream) = AssistantMessageEventStream::channel();
    let builder = MessageBuilder::new(sender, new_output(model));
    tokio::spawn(async move {
        let (mut builder, result) = run(builder).await;
        if builder.is_closed() {
            return;
        }
        let cancelled = cancel.as_ref().is_some_and(CancellationToken::is_cancelled);
        match result {
            Err(error) => {
                let aborted = cancelled || matches!(error, Error::Aborted);
                let message = if aborted && matches!(error, Error::Aborted) {
                    "Request was aborted".to_owned()
                } else {
                    error.to_string()
                };
                builder.fail(message, aborted);
            }
            Ok(()) if cancelled => builder.fail("Request was aborted", true),
            Ok(()) => match builder.output.stop_reason {
                StopReason::Error | StopReason::Aborted => {
                    let message = builder
                        .output
                        .error_message
                        .clone()
                        .unwrap_or_else(|| "An unknown error occurred".into());
                    let aborted = builder.output.stop_reason == StopReason::Aborted;
                    builder.fail(message, aborted);
                }
                _ => {
                    builder.end_open_tool_calls();
                    builder.finish();
                }
            },
        }
    });
    stream
}

/// Apply `options.on_payload`: the returned payload replaces `payload` when present.
pub async fn apply_on_payload(options: &StreamOptions, model: &Model, payload: Value) -> Result<Value> {
    match &options.on_payload {
        Some(hook) => match hook(&payload, model).await {
            Ok(Some(replaced)) => Ok(replaced),
            Ok(None) => Ok(payload),
            Err(error) => Err(Error::msg(error.to_string())),
        },
        None => Ok(payload),
    }
}

/// Call `options.on_response` with the response status and headers.
pub async fn notify_response(options: &StreamOptions, model: &Model, response: &reqwest::Response) -> Result<()> {
    if let Some(hook) = &options.on_response {
        hook(&http::provider_response(response), model)
            .await
            .map_err(|error| Error::msg(error.to_string()))?;
    }
    Ok(())
}

/// Call `options.on_provider_stream_event` with a parsed provider event.
pub async fn observe_provider_event(options: &StreamOptions, model: &Model, event: &Value) -> Result<()> {
    if let Some(hook) = &options.on_provider_stream_event {
        hook(event, model)
            .await
            .map_err(|error| Error::msg(error.to_string()))?;
    }
    Ok(())
}

/// Recompute `total_tokens` from the components and the cost from the model's prices.
pub fn finalize_usage(model: &Model, usage: &mut Usage) {
    usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write;
    calculate_cost(&model.cost, usage);
}

/// Fail with a missing-key error unless an API key or an `authorization`-like header is set.
pub fn require_auth(model: &Model, options: &StreamOptions, header_names: &[&str]) -> Result<()> {
    if options.api_key.as_deref().is_some_and(|key| !key.is_empty()) {
        return Ok(());
    }
    let has_header = options.headers.iter().flatten().any(|(name, value)| {
        value.is_some()
            && header_names
                .iter()
                .any(|candidate| name.eq_ignore_ascii_case(candidate))
    });
    if has_header {
        return Ok(());
    }
    Err(Error::msg(format!("No API key for provider: {}", model.provider)))
}

/// Join a base URL and a path with exactly one slash.
pub fn join_url(base: &str, path: &str) -> String {
    format!("{}/{}", base.trim_end_matches('/'), path.trim_start_matches('/'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_urls() {
        assert_eq!(join_url("https://a/v1/", "/chat"), "https://a/v1/chat");
        assert_eq!(join_url("https://a/v1", "chat"), "https://a/v1/chat");
    }
}
