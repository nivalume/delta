//! Image generation: request and result types and the contract of image API
//! implementations. Run image models with [`crate::Models::generate_images`].

use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

use crate::model::ImageModel;
use crate::options::RequestOptions;
use crate::types::{JsonObject, Usage, UserBlock, now_ms};

/// Input block of an image request: text or a base64 image.
pub type ImagesInputContent = UserBlock;
/// Output block of an image result: text or a base64 image.
pub type ImagesOutputContent = UserBlock;

/// Input of an image request.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ImagesContext {
    pub input: Vec<ImagesInputContent>,
}

impl ImagesContext {
    pub fn new(input: Vec<ImagesInputContent>) -> Self {
        Self { input }
    }
}

/// How an image request ended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ImagesStopReason {
    #[default]
    Stop,
    Error,
    Aborted,
}

impl ImagesStopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Error => "error",
            Self::Aborted => "aborted",
        }
    }
}

/// Final result of an image request.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantImages {
    pub api: String,
    pub provider: String,
    pub model: String,
    pub output: Vec<ImagesOutputContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    pub stop_reason: ImagesStopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

impl AssistantImages {
    /// An empty result for `model` with stop reason `stop`, stamped with the current time.
    pub fn empty(model: &ImageModel) -> Self {
        Self {
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            timestamp: now_ms(),
            ..Self::default()
        }
    }

    /// An error result (`aborted` when `aborted`), as upstream `imageErrorResult`.
    pub fn error(model: &ImageModel, message: impl Into<String>, aborted: bool) -> Self {
        Self {
            stop_reason: if aborted {
                ImagesStopReason::Aborted
            } else {
                ImagesStopReason::Error
            },
            error_message: Some(message.into()),
            ..Self::empty(model)
        }
    }
}

/// Options of an image request.
#[derive(Clone, Debug, Default)]
pub struct ImagesOptions {
    pub base: RequestOptions<ImageModel>,
    /// Request metadata; APIs extract the fields they understand and ignore the rest.
    pub metadata: Option<JsonObject>,
}

impl From<RequestOptions<ImageModel>> for ImagesOptions {
    fn from(base: RequestOptions<ImageModel>) -> Self {
        Self { base, metadata: None }
    }
}

/// The contract of an image-generation API implementation (upstream `ProviderImages`).
///
/// An API module exposes a unit struct implementing it; providers register it under the API
/// id with [`crate::CreateProviderOptions::images`]. The future never fails: request,
/// response and cancellation errors are returned as a result with stop reason `error` or
/// `aborted`. Auth is already resolved into `options.base` when [`crate::Models`] calls it.
pub trait ProviderImages: Send + Sync {
    fn generate_images<'a>(
        &'a self,
        model: &'a ImageModel,
        context: &'a ImagesContext,
        options: ImagesOptions,
    ) -> BoxFuture<'a, AssistantImages>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ImageContent;
    use serde_json::json;

    #[test]
    fn round_trips_upstream_json() {
        let value = json!({
            "api": "openrouter-images",
            "provider": "openrouter",
            "model": "m",
            "output": [
                { "type": "text", "text": "here" },
                { "type": "image", "data": "aGk=", "mimeType": "image/png" }
            ],
            "responseId": "r1",
            "stopReason": "stop",
            "timestamp": 1
        });
        let result: AssistantImages = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(
            result.output[1],
            UserBlock::Image(ImageContent {
                data: "aGk=".into(),
                mime_type: "image/png".into()
            })
        );
        assert_eq!(serde_json::to_value(&result).unwrap(), value);
        let context: ImagesContext =
            serde_json::from_value(json!({ "input": [{ "type": "text", "text": "a red circle" }] })).unwrap();
        assert_eq!(context.input, vec![UserBlock::text("a red circle")]);
    }
}
