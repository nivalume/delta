//! System One models on the Cloudflare Workers AI REST endpoint (upstream
//! `api/cloudflare-workers-ai-system-one.ts`): `POST /accounts/{account}/ai/run` with
//! `{ model, input }`.
//!
//! The REST API wraps the model output in Cloudflare's API envelope. Third-party models such
//! as `typesafe/jev` add a run record:
//! `{ success, result: { state: "Completed", result: { answers, usage } } }`
//! (<https://developers.cloudflare.com/ai/models/typesafe/jev/>). Cloudflare-hosted models such
//! as `@cf/cloudflare/clef` return the output directly: `{ success, result: { model, answers,
//! usage } }` (<https://developers.cloudflare.com/workers-ai/models/clef/>).
//!
//! Catalog base URLs carry the `{CLOUDFLARE_ACCOUNT_ID}` placeholder; the
//! `cloudflare-workers-ai` provider fills it in before this API runs
//! ([`crate::providers::cloudflare::CloudflareClassifier`]).

use futures::future::BoxFuture;
use serde_json::Value;

use crate::api::classifier_shared::endpoint_url;
use crate::api::system_one_shared::{SystemOneTransport, classify_system_one};
use crate::classifier::{ClassifierContext, ClassifierOptions, ClassifierResult, ProviderClassifier};
use crate::error::{Error, Result};
use crate::model::ClassifierModel;
use crate::types::JsonObject;

/// API id of this implementation.
pub const API: &str = "cloudflare-workers-ai-system-one";

const LABEL: &str = "Cloudflare Workers AI";

fn cloudflare_error_message(errors: Option<&Value>) -> String {
    let messages: Vec<&str> = errors
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|error| error.as_object()?.get("message")?.as_str())
        .collect();
    if messages.is_empty() {
        format!("{LABEL} request failed")
    } else {
        format!("{LABEL} error: {}", messages.join("; "))
    }
}

/// JavaScript's `String(value)` for a JSON value (`undefined` when absent).
fn js_string(value: Option<&Value>) -> String {
    match value {
        None => "undefined".into(),
        Some(Value::Null) => "null".into(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Object(_)) => "[object Object]".into(),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item {
                Value::Null => String::new(),
                other => js_string(Some(other)),
            })
            .collect::<Vec<_>>()
            .join(","),
        Some(other) => other.to_string(),
    }
}

fn url(model: &ClassifierModel) -> Result<String> {
    endpoint_url(&model.base_url, "run")
}

fn payload(model: &ClassifierModel, request: Value) -> Value {
    let mut payload = JsonObject::new();
    payload.insert("model".into(), model.id.clone().into());
    payload.insert("input".into(), request);
    Value::Object(payload)
}

fn unexpected() -> Error {
    Error::msg(format!("{LABEL} returned an unexpected response"))
}

fn output(body: Value) -> Result<JsonObject> {
    let Value::Object(mut body) = body else {
        return Err(unexpected());
    };
    if body.get("success") == Some(&Value::Bool(false)) {
        return Err(Error::msg(cloudflare_error_message(body.get("errors"))));
    }
    let Some(Value::Object(mut result)) = body.remove("result") else {
        return Err(unexpected());
    };
    if result.contains_key("answers") {
        return Ok(result);
    }
    if result.get("state").and_then(Value::as_str) != Some("Completed") {
        return Err(Error::msg(format!(
            "{LABEL} run did not complete (state: {})",
            js_string(result.get("state"))
        )));
    }
    match result.remove("result") {
        Some(Value::Object(output)) => Ok(output),
        _ => Err(unexpected()),
    }
}

const TRANSPORT: SystemOneTransport = SystemOneTransport {
    api: API,
    label: LABEL,
    url,
    payload,
    output,
};

/// Cloudflare Workers AI System One classification with public `bool` questions mapped to
/// wire-level `noul` values. `model.base_url` must be resolved (no account placeholder).
pub async fn classify(
    model: &ClassifierModel,
    context: &ClassifierContext,
    options: ClassifierOptions,
) -> ClassifierResult {
    classify_system_one(&TRANSPORT, model, context, options).await
}

/// The `cloudflare-workers-ai-system-one` classifier implementation.
#[derive(Clone, Copy, Debug, Default)]
pub struct CloudflareWorkersAiSystemOneApi;

impl ProviderClassifier for CloudflareWorkersAiSystemOneApi {
    fn classify<'a>(
        &'a self,
        model: &'a ClassifierModel,
        context: &'a ClassifierContext,
        options: ClassifierOptions,
    ) -> BoxFuture<'a, ClassifierResult> {
        Box::pin(classify(model, context, options))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unwraps_cloudflare_envelopes() {
        let direct = output(json!({ "success": true, "result": { "answers": {} } })).unwrap();
        assert!(direct.contains_key("answers"));
        let run = output(json!({ "success": true, "result": { "state": "Completed", "result": { "answers": {} } } }))
            .unwrap();
        assert!(run.contains_key("answers"));
        let message = |body: Value| output(body).unwrap_err().to_string();
        assert_eq!(
            message(json!({ "success": true, "result": { "state": 3 } })),
            "Cloudflare Workers AI run did not complete (state: 3)"
        );
        assert_eq!(
            message(json!({ "success": true, "result": {} })),
            "Cloudflare Workers AI run did not complete (state: undefined)"
        );
        assert_eq!(
            message(json!({ "success": false, "errors": [{ "code": 1 }] })),
            "Cloudflare Workers AI request failed"
        );
        assert_eq!(
            message(json!({ "success": true, "result": [] })),
            "Cloudflare Workers AI returned an unexpected response"
        );
    }
}
