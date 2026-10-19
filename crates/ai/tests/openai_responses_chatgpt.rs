//! Sign in with ChatGPT in the `openai-responses` API. Ported from upstream
//! `test/openai-responses-chatgpt-sign-in.test.ts`, `test/openai-responses-usage-limit.test.ts`
//! and the Sign in with ChatGPT case of `test/openai-responses-mid-convo-effort.test.ts`.
//!
//! Sign in with ChatGPT is detected only for `https://api.openai.com/v1`, so the payload tests
//! capture the request body in `on_payload` and stop the request there instead of sending it.

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

use delta_ai::api::openai_responses;
use delta_ai::compat::{ModelCompat, OpenAIResponsesCompat};
use delta_ai::options::{OpenAIResponsesOptions, ReasoningEffort};
use delta_ai::{
    ApiOptions, AssistantMessage, CacheRetention, Context, Message, Model, StopReason, StreamOptions, normalize_context,
};
use futures::future::BoxFuture;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const USAGE_HINT: &str = "Check your ChatGPT usage: https://chatgpt.com/settings/usage";

fn model(base_url: &str, compat: Option<OpenAIResponsesCompat>) -> Model {
    let mut model: Model = serde_json::from_value(json!({
        "id": "gpt-5-mini", "name": "GPT-5 Mini", "api": "openai-responses", "provider": "openai",
        "baseUrl": base_url, "reasoning": true, "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 400_000, "maxTokens": 128_000
    }))
    .unwrap();
    model.compat = compat.map(ModelCompat::OpenAIResponses);
    model
}

fn openai_model() -> Model {
    model("https://api.openai.com/v1", None)
}

fn context() -> delta_ai::TranscriptContext {
    normalize_context(&Context::new(vec![Message::user("hi")]).with_system_prompt(""))
}

/// Capture the payload and stop the request before it is sent.
async fn run_captured(model: &Model, options: StreamOptions) -> (Value, AssistantMessage) {
    let captured: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let slot = captured.clone();
    let options = StreamOptions {
        on_payload: Some(Arc::new(move |payload: &Value, _model: &Model| {
            *slot.lock().unwrap() = Some(payload.clone());
            let stop: BoxFuture<'static, _> = Box::pin(async { Err("captured".into()) });
            stop
        })),
        ..options
    };
    let message = openai_responses::stream(model, &context(), options).result().await;
    assert_eq!(message.error_message.as_deref(), Some("captured"));
    let payload = captured
        .lock()
        .unwrap()
        .take()
        .expect("request payload was not captured");
    (payload, message)
}

async fn capture_payload(api_key: &str, model: &Model) -> Value {
    let options = StreamOptions {
        api_key: Some(api_key.into()),
        max_tokens: Some(1000),
        temperature: Some(0.5),
        cache_retention: Some(CacheRetention::Long),
        ..StreamOptions::default()
    };
    run_captured(model, options).await.0
}

#[tokio::test]
async fn omits_request_fields_that_token_sharing_rejects() {
    let payload = capture_payload("chatgpt-access-token", &openai_model()).await;
    assert!(payload.get("max_output_tokens").is_none());
    assert!(payload.get("temperature").is_none());
    assert!(payload.get("prompt_cache_retention").is_none());
}

#[tokio::test]
async fn omits_prompt_cache_options_on_models_with_explicit_prompt_cache_mode() {
    let explicit_cache_model = model(
        "https://api.openai.com/v1",
        Some(OpenAIResponsesCompat {
            supports_explicit_prompt_cache_mode: Some(true),
            ..OpenAIResponsesCompat::default()
        }),
    );
    let sign_in = capture_payload("chatgpt-access-token", &explicit_cache_model).await;
    let api_key = capture_payload("sk-proj-test", &explicit_cache_model).await;
    assert!(sign_in.get("prompt_cache_options").is_none());
    assert_eq!(api_key["prompt_cache_options"], json!({"ttl": "30m"}));
}

#[tokio::test]
async fn keeps_those_fields_for_openai_api_keys() {
    let payload = capture_payload("sk-proj-test", &openai_model()).await;
    assert_eq!(payload["max_output_tokens"], 1000);
    assert_eq!(payload["temperature"], 0.5);
    assert_eq!(payload["prompt_cache_retention"], "24h");
}

#[tokio::test]
async fn keeps_those_fields_for_other_openai_compatible_endpoints() {
    let payload = capture_payload("gateway-key", &model("https://gateway.example.com/v1", None)).await;
    assert_eq!(payload["max_output_tokens"], 1000);
    assert_eq!(payload["temperature"], 0.5);
    assert_eq!(payload["prompt_cache_retention"], "24h");
}

#[tokio::test]
async fn sends_the_level_as_reasoning_effort_with_sign_in_with_chatgpt() {
    let mid_convo = model(
        "https://api.openai.com/v1",
        Some(OpenAIResponsesCompat {
            supports_mid_convo_effort: Some(true),
            ..OpenAIResponsesCompat::default()
        }),
    );
    let options = |api_key: &str| StreamOptions {
        api_key: Some(api_key.into()),
        api: Some(ApiOptions::OpenAIResponses(OpenAIResponsesOptions {
            reasoning_effort: Some(ReasoningEffort::Low),
            ..OpenAIResponsesOptions::default()
        })),
        ..StreamOptions::default()
    };
    let is_update = |item: &Value| item["type"] == "configuration_update";

    let (payload, message) = run_captured(&mid_convo, options("chatgpt-access-token")).await;
    assert_eq!(payload["reasoning"]["effort"], "low");
    assert!(!payload["input"].as_array().unwrap().iter().any(is_update));
    assert_eq!(message.provider_thinking_level, None);

    // An API key sends the level in a configuration_update item.
    let (payload, message) = run_captured(&mid_convo, options("sk-test")).await;
    assert_eq!(payload["reasoning"]["effort"], "medium");
    assert!(payload["input"].as_array().unwrap().iter().any(is_update));
    assert_eq!(message.provider_thinking_level.as_deref(), Some("low"));
}

async fn error_message(response: ResponseTemplate) -> String {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(response)
        .mount(&server)
        .await;
    let options = StreamOptions {
        api_key: Some("test".into()),
        max_retries: Some(0),
        http_client: Some(reqwest::Client::builder().no_proxy().build().unwrap()),
        ..StreamOptions::default()
    };
    let message = openai_responses::stream(&model(&format!("{}/v1", server.uri()), None), &context(), options)
        .result()
        .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    message.error_message.unwrap()
}

#[tokio::test]
async fn links_to_chatgpt_usage_when_the_request_is_rejected() {
    let body = json!({"error": {"code": "subscription_sharing_usage_limit_exceeded",
        "message": "Usage limit reached.", "type": "rate_limit_error"}});
    let message = error_message(ResponseTemplate::new(429).set_body_json(body)).await;
    assert!(message.contains("subscription_sharing_usage_limit_exceeded"));
    assert!(message.ends_with(&format!("\n{USAGE_HINT}")));
}

#[tokio::test]
async fn links_to_chatgpt_usage_when_the_stream_fails() {
    let event = json!({"type": "response.failed", "sequence_number": 0,
        "response": {"id": "resp_failed", "status": "failed",
            "error": {"code": "subscription_sharing_usage_limit_exceeded", "message": "Usage limit reached."}}});
    let body = format!("event: response.failed\ndata: {event}\n\n");
    let message = error_message(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")).await;
    assert_eq!(
        message,
        format!("subscription_sharing_usage_limit_exceeded: Usage limit reached.\n{USAGE_HINT}")
    );
}

#[tokio::test]
async fn leaves_other_errors_alone() {
    let body = json!({"error": {"code": "server_error", "message": "boom"}});
    let message = error_message(ResponseTemplate::new(400).set_body_json(body)).await;
    assert!(!message.contains(USAGE_HINT));
}
