//! TypeSafe's native System One protocol (upstream `api/typesafe-system-one.ts`):
//! `POST <baseUrl>/systemone` with `{ model, state, questions }`. OpenRouter, OpenCode Zen,
//! Vercel AI Gateway and llama.cpp serve the same protocol, so their providers use this API
//! with their own base URLs.

use futures::future::BoxFuture;
use serde_json::Value;

use crate::api::classifier_shared::endpoint_url;
use crate::api::system_one_shared::{SystemOneTransport, classify_system_one};
use crate::classifier::{ClassifierContext, ClassifierOptions, ClassifierResult, ProviderClassifier};
use crate::error::{Error, Result};
use crate::model::ClassifierModel;
use crate::types::JsonObject;

/// API id of this implementation.
pub const API: &str = "typesafe-system-one";

const LABEL: &str = "System One API";

fn url(model: &ClassifierModel) -> Result<String> {
    endpoint_url(&model.base_url, "systemone")
}

fn payload(model: &ClassifierModel, request: Value) -> Value {
    let mut payload = JsonObject::new();
    payload.insert("model".into(), model.id.clone().into());
    if let Value::Object(request) = request {
        payload.extend(request);
    }
    Value::Object(payload)
}

fn output(body: Value) -> Result<JsonObject> {
    match body {
        Value::Object(body) => Ok(body),
        _ => Err(Error::msg(format!("{LABEL} returned an unexpected response"))),
    }
}

const TRANSPORT: SystemOneTransport = SystemOneTransport {
    api: API,
    label: LABEL,
    url,
    payload,
    output,
};

/// TypeSafe System One classification with public `bool` questions mapped to wire-level
/// `noul` values. `options.temperature` is ignored: System One has no such field.
pub async fn classify(
    model: &ClassifierModel,
    context: &ClassifierContext,
    options: ClassifierOptions,
) -> ClassifierResult {
    classify_system_one(&TRANSPORT, model, context, options).await
}

/// The `typesafe-system-one` classifier implementation.
#[derive(Clone, Copy, Debug, Default)]
pub struct TypeSafeSystemOneApi;

impl ProviderClassifier for TypeSafeSystemOneApi {
    fn classify<'a>(
        &'a self,
        model: &'a ClassifierModel,
        context: &'a ClassifierContext,
        options: ClassifierOptions,
    ) -> BoxFuture<'a, ClassifierResult> {
        Box::pin(classify(model, context, options))
    }
}
