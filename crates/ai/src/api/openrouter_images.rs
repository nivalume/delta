//! Image generation over OpenRouter's chat completions endpoint (upstream
//! `api/openrouter-images.ts`).
//!
//! Upstream sends the request through the OpenAI SDK. This module sends the same
//! non-streaming `POST <baseUrl>/chat/completions` request directly: the input blocks become
//! one user message, `modalities` asks for images (plus text when the model can return it),
//! and the response's `message.content` and `message.images` data URLs become the output
//! blocks.

use futures::future::BoxFuture;
use serde_json::{Value, json};

use crate::api::common::{
    apply_request_on_payload, format_provider_error, join_url, notify_request_response, openai_sdk_error,
};
use crate::error::{Error, Result};
use crate::http;
use crate::images::{AssistantImages, ImagesContext, ImagesOptions, ImagesStopReason, ProviderImages};
use crate::model::{ImageModel, OutputModality};
use crate::types::{ImageContent, ProviderHeaders, Usage, UsageCost, UserBlock};

/// API id of this implementation.
pub const API: &str = "openrouter-images";

/// The request body (upstream `buildParams`).
fn build_params(model: &ImageModel, context: &ImagesContext) -> Value {
    let content: Vec<Value> = context
        .input
        .iter()
        .map(|block| match block {
            UserBlock::Text(text) => json!({ "type": "text", "text": text.text }),
            UserBlock::Image(image) => json!({
                "type": "image_url",
                "image_url": { "url": format!("data:{};base64,{}", image.mime_type, image.data) }
            }),
        })
        .collect();
    let modalities = if model.output.contains(&OutputModality::Text) {
        json!(["image", "text"])
    } else {
        json!(["image"])
    };
    json!({
        "model": model.id,
        "messages": [{ "role": "user", "content": content }],
        "stream": false,
        "modalities": modalities
    })
}

/// A token count as upstream reads it (`value || 0`).
fn tokens(value: Option<&Value>) -> u64 {
    value
        .and_then(Value::as_f64)
        .filter(|count| count.is_finite() && *count > 0.0)
        .map_or(0, |count| count as u64)
}

/// Usage of an OpenAI-style `usage` object, priced at the model's base rates (upstream
/// `parseUsage`; cached tokens that were written in this request count as cache writes).
fn parse_usage(raw: &Value, model: &ImageModel) -> Usage {
    let prompt_tokens = tokens(raw.get("prompt_tokens"));
    let details = raw.get("prompt_tokens_details");
    let reported_cached = tokens(details.and_then(|details| details.get("cached_tokens")));
    let cache_write = tokens(details.and_then(|details| details.get("cache_write_tokens")));
    let cache_read = if cache_write > 0 {
        reported_cached.saturating_sub(cache_write)
    } else {
        reported_cached
    };
    let input = prompt_tokens.saturating_sub(cache_read + cache_write);
    let output = tokens(raw.get("completion_tokens"));
    let price = |rate: f64, count: u64| rate / 1_000_000.0 * count as f64;
    let mut cost = UsageCost {
        input: price(model.cost.input, input),
        output: price(model.cost.output, output),
        cache_read: price(model.cost.cache_read, cache_read),
        cache_write: price(model.cost.cache_write, cache_write),
        total: 0.0,
    };
    cost.total = cost.input + cost.output + cost.cache_read + cost.cache_write;
    Usage {
        input,
        output,
        cache_read,
        cache_write,
        total_tokens: input + output + cache_read + cache_write,
        cost,
        ..Usage::default()
    }
}

/// `data:<mime>;base64,<data>` split as upstream's `/^data:([^;]+);base64,(.+)$/`.
fn parse_data_url(url: &str) -> Option<ImageContent> {
    let rest = url.strip_prefix("data:")?;
    let (mime_type, data) = rest.split_once(';')?;
    let data = data.strip_prefix("base64,")?;
    let line_break = |c: char| matches!(c, '\n' | '\r' | '\u{2028}' | '\u{2029}');
    if mime_type.is_empty() || data.is_empty() || data.contains(line_break) {
        return None;
    }
    Some(ImageContent {
        data: data.to_owned(),
        mime_type: mime_type.to_owned(),
    })
}

/// The output blocks of the first choice: its text content, then its data-URL images.
fn parse_output(response: &Value) -> Result<Vec<UserBlock>> {
    let Some(choices) = response.get("choices").and_then(Value::as_array) else {
        return Err(Error::msg("OpenRouter returned a response without choices"));
    };
    let Some(choice) = choices.first() else {
        return Ok(Vec::new());
    };
    let Some(message) = choice.get("message").filter(|message| message.is_object()) else {
        return Err(Error::msg("OpenRouter returned a choice without a message"));
    };
    let mut output = Vec::new();
    if let Some(text) = message
        .get("content")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        output.push(UserBlock::text(text));
    }
    for image in message.get("images").and_then(Value::as_array).into_iter().flatten() {
        let url = match image.get("image_url") {
            Some(Value::String(url)) => Some(url.as_str()),
            Some(object) => object.get("url").and_then(Value::as_str),
            None => None,
        };
        if let Some(image) = url.and_then(parse_data_url) {
            output.push(UserBlock::Image(image));
        }
    }
    Ok(output)
}

/// An HTTP error as upstream reports it: the OpenAI SDK error message, or `<status>: <body>`
/// when the message lacks the body.
fn format_http_error(error: Error) -> Error {
    let Error::Status {
        status,
        body,
        retry_after_ms,
        ..
    } = error
    else {
        return error;
    };
    let (sdk_message, error_object) = openai_sdk_error(status, body.as_deref());
    Error::Status {
        status,
        message: format_provider_error(status, &sdk_message, error_object.as_ref(), None),
        body,
        retry_after_ms,
    }
}

async fn run(
    model: &ImageModel,
    context: &ImagesContext,
    options: &ImagesOptions,
    output: &mut AssistantImages,
) -> Result<()> {
    let options = &options.base;
    let Some(api_key) = options.api_key.as_deref().filter(|key| !key.is_empty()) else {
        return Err(Error::msg(format!("No API key for provider: {}", model.provider)));
    };
    let params = apply_request_on_payload(options, model, build_params(model, context)).await?;

    let defaults = http::headers_of([
        ("Accept", "application/json"),
        ("Content-Type", "application/json"),
        ("Authorization", &format!("Bearer {api_key}")),
        ("User-Agent", &http::user_agent()),
    ]);
    // Upstream spreads `{ ...model.headers, ...options.headers }` (case-sensitive) and drops
    // `null` values before handing the rest to the SDK as default headers.
    let mut custom: ProviderHeaders = model
        .headers
        .iter()
        .flatten()
        .map(|(name, value)| (name.clone(), Some(value.clone())))
        .collect();
    for (name, value) in options.headers.iter().flatten() {
        custom.insert(name.clone(), value.clone());
    }
    custom.retain(|_, value| value.is_some());
    let headers = http::build_headers([&defaults, &custom])?;
    let url = join_url(&model.base_url, "chat/completions");
    let body = serde_json::to_vec(&params)?;

    // The OpenAI SDK runs with `maxRetries: 0`; upstream retries only when `max_retries` is set.
    let mut transport = options.transport_options();
    transport.max_retries = Some(options.max_retries.unwrap_or(0));
    let client = http::client(&transport);
    let response = http::send_with_retry(&transport, || {
        Ok(client.post(&url).headers(headers.clone()).body(body.clone()))
    })
    .await
    .map_err(format_http_error)?;
    notify_request_response(options, model, &http::provider_response(&response)).await?;
    let response: Value = http::or_cancel(options.cancel.as_ref(), async {
        let bytes = response.bytes().await?;
        Ok(serde_json::from_slice(&bytes)?)
    })
    .await?;

    output.response_id = response.get("id").and_then(Value::as_str).map(str::to_owned);
    if let Some(usage) = response.get("usage").filter(|usage| usage.is_object()) {
        output.usage = Some(parse_usage(usage, model));
    }
    output.output = parse_output(&response)?;
    Ok(())
}

/// Generate images with an OpenRouter image model. Never fails: errors are results with stop
/// reason `error` (`aborted` when `options.base.cancel` fired).
pub async fn generate_images(model: &ImageModel, context: &ImagesContext, options: ImagesOptions) -> AssistantImages {
    let mut output = AssistantImages::empty(model);
    if let Err(error) = run(model, context, &options, &mut output).await {
        output.stop_reason = if options.base.is_cancelled() {
            ImagesStopReason::Aborted
        } else {
            ImagesStopReason::Error
        };
        output.error_message = Some(match error {
            Error::Aborted => "Request aborted".into(),
            other => other.to_string(),
        });
    }
    output
}

/// The `openrouter-images` image implementation.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenRouterImagesApi;

impl ProviderImages for OpenRouterImagesApi {
    fn generate_images<'a>(
        &'a self,
        model: &'a ImageModel,
        context: &'a ImagesContext,
        options: ImagesOptions,
    ) -> BoxFuture<'a, AssistantImages> {
        Box::pin(generate_images(model, context, options))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelCost;

    #[test]
    fn splits_data_urls_like_upstream() {
        assert_eq!(
            parse_data_url("data:image/png;base64,ZmFrZQ=="),
            Some(ImageContent {
                data: "ZmFrZQ==".into(),
                mime_type: "image/png".into()
            })
        );
        assert_eq!(parse_data_url("https://example.com/a.png"), None);
        assert_eq!(parse_data_url("data:;base64,ZmFrZQ=="), None);
        assert_eq!(parse_data_url("data:image/png;utf8,abc"), None);
        assert_eq!(parse_data_url("data:image/png;base64,"), None);
        assert_eq!(parse_data_url("data:image/png;base64,ab\ncd"), None);
    }

    #[test]
    fn prices_usage_at_base_rates() {
        let model = ImageModel {
            id: "m".into(),
            name: "m".into(),
            api: API.into(),
            provider: "openrouter".into(),
            base_url: "https://openrouter.ai/api/v1".into(),
            input: Vec::new(),
            input_limits: None,
            cost: ModelCost {
                input: 1.0,
                output: 2.0,
                cache_read: 0.5,
                cache_write: 4.0,
                tiers: None,
            },
            headers: None,
            output: vec![OutputModality::Image],
        };
        let usage = parse_usage(
            &json!({
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "prompt_tokens_details": { "cached_tokens": 30, "cache_write_tokens": 20 }
            }),
            &model,
        );
        assert_eq!(
            (
                usage.input,
                usage.output,
                usage.cache_read,
                usage.cache_write,
                usage.total_tokens
            ),
            (70, 10, 10, 20, 110)
        );
        let expected = (70.0 + 20.0 + 5.0 + 80.0) / 1_000_000.0;
        assert!((usage.cost.total - expected).abs() < 1e-15);
    }
}
