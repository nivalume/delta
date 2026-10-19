//! Plumbing shared by the classifier APIs (upstream `api/classifier-shared.ts`): the JSON POST
//! with bearer auth, request hooks, a timeout per attempt and provider retries; usage parsing;
//! response field checks and error messages.

use reqwest::header::HeaderMap;
use serde_json::Value;
use std::time::Duration;

use crate::api::common::{apply_request_on_payload, notify_request_response};
use crate::classifier::{ClassifierOptions, ClassifierResult, ClassifierStopReason};
use crate::error::{Error, Result};
use crate::http;
use crate::model::{ClassifierModel, calculate_cost};
use crate::options::{ProviderResponse, RequestOptions};
use crate::types::{JsonObject, ProviderHeaders, Usage};

/// `value` as a JSON object (upstream `isRecord`: an object, not an array or `null`).
pub fn as_record(value: Option<&Value>) -> Option<&JsonObject> {
    value.and_then(Value::as_object)
}

/// A finite number, else `"<label> returned an invalid <field>"` (upstream `requiredNumber`).
pub fn required_number(label: &str, value: Option<&Value>, field: &str) -> Result<f64> {
    value
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
        .ok_or_else(|| Error::msg(format!("{label} returned an invalid {field}")))
}

/// Request headers: `content-type`, `authorization: Bearer <key>` when a key is given, then the
/// model headers and the request headers (case-insensitive, `None` deletes).
pub fn request_headers(
    model: &ClassifierModel,
    api_key: Option<&str>,
    headers: Option<&ProviderHeaders>,
) -> Result<HeaderMap> {
    let mut defaults = http::headers_of([("content-type", "application/json")]);
    if let Some(api_key) = api_key {
        defaults.insert("authorization".into(), Some(format!("Bearer {api_key}")));
    }
    let model_headers: ProviderHeaders = model
        .headers
        .iter()
        .flatten()
        .map(|(name, value)| (name.clone(), Some(value.clone())))
        .collect();
    let none = ProviderHeaders::new();
    http::build_headers([&defaults, &model_headers, headers.unwrap_or(&none)])
}

/// Why one attempt failed, and whether the retry policy may repeat it.
struct Failure {
    error: Error,
    /// Upstream retries errors that carry a status field (HTTP errors and timeouts) only.
    retryable: bool,
    /// Server-requested delay from `retry-after-ms` / `retry-after`.
    delay: Option<Duration>,
}

impl From<Error> for Failure {
    fn from(error: Error) -> Self {
        Self {
            error,
            retryable: false,
            delay: None,
        }
    }
}

impl From<reqwest::Error> for Failure {
    fn from(error: reqwest::Error) -> Self {
        Error::from(error).into()
    }
}

async fn attempt(
    label: &str,
    client: &reqwest::Client,
    url: &str,
    headers: &HeaderMap,
    body: &[u8],
) -> std::result::Result<(ProviderResponse, Value), Failure> {
    let response = client
        .post(url)
        .headers(headers.clone())
        .body(body.to_vec())
        .send()
        .await?;
    let status = response.status().as_u16();
    if !response.status().is_success() {
        let response_headers = response.headers().clone();
        let text = response.text().await?;
        let delay = http::retry_after(&response_headers);
        return Err(Failure {
            error: Error::Status {
                status,
                message: format!("{label} returned {status}"),
                body: http::error_body_text(&text),
                retry_after_ms: delay.map(|delay| delay.as_millis() as u64),
            },
            retryable: http::is_retryable_status(status, &response_headers),
            delay,
        });
    }
    let observed = http::provider_response(&response);
    let bytes = response.bytes().await?;
    let json = serde_json::from_slice(&bytes).map_err(Error::from)?;
    Ok((observed, json))
}

/// POST `payload` as JSON to `url` and return the response status, headers and parsed body.
///
/// Each attempt gets its own `options.timeout` (no timeout when unset), which fails it with
/// `Request timed out after <ms>ms`. HTTP errors (`"<label> returned <status>"` with the body)
/// and timeouts are retried with the vendor SDK policy up to `options.max_retries` times
/// (default 2), except `no_retry_statuses`; other failures are not retried. Cancellation
/// fails with [`Error::Aborted`].
pub async fn post_json<M>(
    label: &str,
    url: &str,
    headers: &HeaderMap,
    payload: &Value,
    options: &RequestOptions<M>,
    no_retry_statuses: &[u16],
) -> Result<(ProviderResponse, Value)> {
    let client = http::client(&options.transport_options());
    let body = serde_json::to_vec(payload)?;
    let cancel = options.cancel.as_ref();
    let max_retries = options.max_retries.unwrap_or(http::DEFAULT_MAX_RETRIES);
    let mut retries_remaining = max_retries;
    loop {
        let request = attempt(label, &client, url, headers, &body);
        let timed = async {
            match options.timeout {
                Some(timeout) => tokio::time::timeout(timeout, request).await.unwrap_or_else(|_| {
                    Err(Failure {
                        error: Error::msg(format!("Request timed out after {}ms", timeout.as_millis())),
                        retryable: true,
                        delay: None,
                    })
                }),
                None => request.await,
            }
        };
        let outcome = match cancel {
            Some(cancel) => tokio::select! {
                () = cancel.cancelled() => return Err(Error::Aborted),
                outcome = timed => outcome,
            },
            None => timed.await,
        };
        let failure = match outcome {
            Ok(result) => return Ok(result),
            Err(failure) => failure,
        };
        if options.is_cancelled() {
            return Err(Error::Aborted);
        }
        let no_retry = matches!(&failure.error, Error::Status { status, .. } if no_retry_statuses.contains(status));
        if retries_remaining == 0 || !failure.retryable || no_retry {
            return Err(failure.error);
        }
        let retry_index = max_retries - retries_remaining;
        retries_remaining -= 1;
        let max_delay = options.max_retry_delay.unwrap_or(http::DEFAULT_MAX_RETRY_DELAY);
        let delay = match failure.delay {
            Some(delay) if !max_delay.is_zero() && delay > max_delay => {
                return Err(Error::msg(format!(
                    "Server requested {}s retry delay (max: {}s). {}",
                    delay.as_secs_f64().ceil(),
                    max_delay.as_secs_f64().ceil(),
                    failure.error
                )));
            }
            Some(delay) => delay,
            None => http::backoff(retry_index),
        };
        http::sleep_or_cancel(delay, cancel).await?;
    }
}

/// Post one classifier request with bearer auth and the `on_payload` / `on_response` hooks
/// (upstream `postClassifierRequest`). Returns the parsed response body.
pub async fn post_classifier_request(
    label: &str,
    url: &str,
    model: &ClassifierModel,
    body: Value,
    options: &ClassifierOptions,
    no_retry_statuses: &[u16],
) -> Result<Value> {
    let options = &options.base;
    let Some(api_key) = options.api_key.as_deref().filter(|key| !key.is_empty()) else {
        return Err(Error::msg(format!("No API key for provider: {}", model.provider)));
    };
    let payload = apply_request_on_payload(options, model, body).await?;
    let headers = request_headers(model, Some(api_key), options.headers.as_ref())?;
    let (response, json) = post_json(label, url, &headers, &payload, options, no_retry_statuses).await?;
    notify_request_response(options, model, &response).await?;
    Ok(json)
}

fn token_count(value: Option<&Value>) -> u64 {
    value
        .and_then(Value::as_f64)
        .filter(|count| count.is_finite() && *count > 0.0)
        .map_or(0, |count| count as u64)
}

/// Usage from a `{ input_tokens, output_tokens }` object, priced from the model catalog like
/// chat usage (upstream `parseClassifierUsage`). A missing or malformed usage object gives
/// `None` instead of failing the result.
pub fn parse_classifier_usage(value: Option<&Value>, model: &ClassifierModel) -> Option<Usage> {
    let usage = as_record(value)?;
    if !usage.contains_key("input_tokens") && !usage.contains_key("output_tokens") {
        return None;
    }
    let input = token_count(usage.get("input_tokens"));
    let output = token_count(usage.get("output_tokens"));
    let mut usage = Usage {
        input,
        output,
        total_tokens: input + output,
        ..Usage::default()
    };
    calculate_cost(&model.cost, &mut usage);
    Some(usage)
}

/// The text of a classifier error result (upstream
/// `formatProviderError(normalizeProviderError(error), prefix)`): HTTP errors become
/// `"<prefix> (<status>): <body>"`, or `"<prefix> (<status>): <message>"` without a body;
/// cancellation is `Request aborted`; other errors keep their message.
pub fn format_classifier_error(error: &Error, prefix: &str) -> String {
    match error {
        Error::Aborted => "Request aborted".into(),
        Error::Status {
            status, message, body, ..
        } => match body {
            Some(body) if !message.contains(body.as_str()) => format!("{prefix} ({status}): {body}"),
            _ => format!("{prefix} ({status}): {message}"),
        },
        other => other.to_string(),
    }
}

/// Turn `output` into an error result: no answers, stop reason `aborted` when the request
/// was cancelled, else `error`. Usage already set is kept (the request was billed).
pub fn fail_result(output: &mut ClassifierResult, options: &ClassifierOptions, message: String) {
    output.answers.clear();
    output.stop_reason = if options.base.is_cancelled() {
        ClassifierStopReason::Aborted
    } else {
        ClassifierStopReason::Error
    };
    output.error_message = Some(message);
}

/// `base` without trailing slashes, as a directory URL, joined with `path` (upstream
/// `new URL(path, `${base.replace(/\/+$/u, "")}/`)`).
pub fn endpoint_url(base: &str, path: &str) -> Result<String> {
    let base = format!("{}/", base.trim_end_matches('/'));
    url::Url::parse(&base)
        .and_then(|base| base.join(path))
        .map(String::from)
        .map_err(|_| Error::msg("Invalid URL"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelCost, ModelCostTier};
    use serde_json::json;

    fn model(cost: ModelCost) -> ClassifierModel {
        ClassifierModel {
            id: "m".into(),
            name: "m".into(),
            api: "test".into(),
            provider: "test".into(),
            base_url: "https://example.test/v1".into(),
            input: Vec::new(),
            input_limits: None,
            cost,
            headers: None,
            context_window: 1000,
        }
    }

    #[test]
    fn parses_and_prices_usage() {
        let cost = ModelCost {
            input: 0.1,
            tiers: Some(vec![ModelCostTier {
                input_tokens_above: 272_000.0,
                input: 0.2,
                ..ModelCostTier::default()
            }]),
            ..ModelCost::default()
        };
        let usage = parse_classifier_usage(
            Some(&json!({ "input_tokens": 300_000, "output_tokens": 0 })),
            &model(cost),
        );
        assert!((usage.unwrap().cost.total - 0.06).abs() < 1e-12);
        let partial = parse_classifier_usage(
            Some(&json!({ "input_tokens": "many", "output_tokens": 3 })),
            &model(ModelCost::default()),
        )
        .unwrap();
        assert_eq!((partial.input, partial.output, partial.total_tokens), (0, 3, 3));
        assert!(parse_classifier_usage(Some(&json!({ "cost": 0.1 })), &model(ModelCost::default())).is_none());
        assert!(parse_classifier_usage(Some(&json!([1])), &model(ModelCost::default())).is_none());
        assert!(parse_classifier_usage(None, &model(ModelCost::default())).is_none());
    }

    #[test]
    fn formats_errors_like_upstream() {
        let status = |body: Option<&str>| Error::Status {
            status: 400,
            message: "Label returned 400".into(),
            body: body.map(str::to_owned),
            retry_after_ms: None,
        };
        assert_eq!(
            format_classifier_error(&status(Some("bad input")), "Label error"),
            "Label error (400): bad input"
        );
        assert_eq!(
            format_classifier_error(&status(None), "Label error"),
            "Label error (400): Label returned 400"
        );
        assert_eq!(
            format_classifier_error(&Error::Aborted, "Label error"),
            "Request aborted"
        );
        assert_eq!(format_classifier_error(&Error::msg("plain"), "Label error"), "plain");
    }

    #[test]
    fn joins_endpoint_urls() {
        assert_eq!(
            endpoint_url("https://api.typesafe.ai/v1/", "systemone").unwrap(),
            "https://api.typesafe.ai/v1/systemone"
        );
        assert_eq!(
            endpoint_url("https://openrouter.ai/api/v1//", "systemone").unwrap(),
            "https://openrouter.ai/api/v1/systemone"
        );
        assert_eq!(
            endpoint_url("not a url", "systemone").unwrap_err().to_string(),
            "Invalid URL"
        );
    }
}
