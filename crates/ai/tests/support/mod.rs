//! Helpers shared by the image, classifier and deferred-response tests.

#![allow(dead_code, unreachable_pub)]

use futures::future::BoxFuture;
use std::collections::HashMap;
use std::sync::Arc;

use delta_ai::auth::{ApiKeyInput, AuthContext};
use delta_ai::model::OutputModality;
use delta_ai::{
    ApiKeyAuth, AssistantMessage, AssistantMessageEventStream, AuthResult, ClassifierModel, ImageModel, InputModality,
    Model, ModelCost, ModelsError, ProviderApi, ProviderAuth, ProviderStreams, SimpleStreamOptions, StopReason,
    StreamOptions, TranscriptContext, env_api_key_auth,
};

/// Auth context with a fixed environment and no files.
pub struct FakeAuthContext(pub HashMap<String, String>);

impl FakeAuthContext {
    pub fn with_env(env: &[(&str, &str)]) -> Arc<dyn AuthContext> {
        Arc::new(Self(
            env.iter()
                .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
                .collect(),
        ))
    }
}

impl AuthContext for FakeAuthContext {
    fn env(&self, name: &str) -> Option<String> {
        self.0.get(name).cloned()
    }

    fn file_exists(&self, _path: &str) -> bool {
        false
    }
}

/// API-key auth that always resolves to the same result.
pub struct StaticAuth(pub AuthResult);

impl ApiKeyAuth for StaticAuth {
    fn name(&self) -> &'static str {
        "Test key"
    }

    fn resolve<'a>(&'a self, _input: ApiKeyInput<'a>) -> BoxFuture<'a, Result<Option<AuthResult>, ModelsError>> {
        let result = self.0.clone();
        Box::pin(async move { Ok(Some(result)) })
    }
}

/// Keyless auth, or env-var auth when `env_var` is set (upstream `testProvider`).
pub fn test_auth(env_var: Option<&str>) -> ProviderAuth {
    match env_var {
        Some(var) => ProviderAuth::api_key(env_api_key_auth("Test key", &[var])),
        None => ProviderAuth::keyless("Test"),
    }
}

/// Chat implementation that answers every request with an empty `stop` message.
pub struct NoChat;

impl ProviderStreams for NoChat {
    fn stream(
        &self,
        model: &Model,
        _context: &TranscriptContext,
        _options: StreamOptions,
    ) -> AssistantMessageEventStream {
        let mut message = AssistantMessage::empty(&model.api, &model.provider, &model.id);
        message.stop_reason = StopReason::Stop;
        AssistantMessageEventStream::from_final(message)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.stream(model, context, options.base)
    }
}

/// A chat API map with [`NoChat`] under `api`.
pub fn chat_api(api: &str) -> ProviderApi {
    ProviderApi::ByApi([(api.to_owned(), Arc::new(NoChat) as Arc<dyn ProviderStreams>)].into())
}

pub fn zero_cost() -> ModelCost {
    ModelCost::default()
}

pub fn image_model(provider: &str, id: &str) -> ImageModel {
    ImageModel {
        id: id.into(),
        name: id.into(),
        api: "test-images".into(),
        provider: provider.into(),
        base_url: "https://example.test/v1".into(),
        input: vec![InputModality::Text],
        input_limits: None,
        cost: zero_cost(),
        headers: None,
        output: vec![OutputModality::Image],
    }
}

pub fn classifier_model(provider: &str, id: &str) -> ClassifierModel {
    ClassifierModel {
        id: id.into(),
        name: id.into(),
        api: "test-classifier".into(),
        provider: provider.into(),
        base_url: "https://example.test/v1".into(),
        input: vec![InputModality::Text],
        input_limits: None,
        cost: zero_cost(),
        headers: None,
        context_window: 1000,
    }
}

pub fn chat_model(provider: &str, api: &str, id: &str) -> Model {
    serde_json::from_value(serde_json::json!({
        "id": id,
        "name": id,
        "api": api,
        "provider": provider,
        "baseUrl": "https://example.test/v1",
        "reasoning": false,
        "input": ["text"],
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
        "contextWindow": 1000,
        "maxTokens": 100
    }))
    .expect("valid chat model")
}
