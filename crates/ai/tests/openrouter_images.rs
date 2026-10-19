//! Tests of the `openrouter-images` API against local servers (port of upstream
//! `test/openrouter-images.test.ts`, whose mocked OpenAI SDK becomes a mock HTTP server, and
//! the provider part of `test/images.test.ts`).

#![allow(clippy::unwrap_used)] // Test helpers outside `#[test]` functions.

use delta_ai::api::openrouter_images::generate_images;
use delta_ai::model::OutputModality;
use delta_ai::providers::builtin::builtin_models;
use delta_ai::{
    AnyModel, CreateModelsOptions, ImageContent, ImageModel, ImagesContext, ImagesOptions, ImagesStopReason,
    InputModality, ModelCost, ModelType, RequestOptions, UserBlock,
};
use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn model(base_url: &str, id: &str, output: Vec<OutputModality>) -> ImageModel {
    ImageModel {
        id: id.into(),
        name: id.into(),
        api: "openrouter-images".into(),
        provider: "openrouter".into(),
        base_url: base_url.into(),
        input: vec![InputModality::Text, InputModality::Image],
        input_limits: None,
        cost: ModelCost {
            input: 0.015,
            output: 0.03,
            ..ModelCost::default()
        },
        headers: None,
        output,
    }
}

fn context() -> ImagesContext {
    ImagesContext::new(vec![UserBlock::text("Generate a dog")])
}

fn options(api_key: &str) -> ImagesOptions {
    RequestOptions {
        api_key: Some(api_key.into()),
        ..RequestOptions::default()
    }
    .into()
}

fn response() -> Value {
    json!({
        "id": "img-1",
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 34,
            "prompt_tokens_details": { "cached_tokens": 0 }
        },
        "choices": [{
            "message": {
                "content": "Here is your image.",
                "images": [{ "image_url": "data:image/png;base64,ZmFrZS1wbmc=" }]
            }
        }]
    })
}

async fn serve(status: u16, body: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(status).set_body_json(body))
        .mount(&server)
        .await;
    server
}

async fn last_request(server: &MockServer) -> Request {
    server.received_requests().await.unwrap().pop().unwrap()
}

#[tokio::test]
async fn returns_text_plus_images_in_final_output() {
    let server = serve(200, response()).await;
    let mut model = model(
        &format!("{}/api/v1", server.uri()),
        "google/gemini-3.1-flash-image-preview",
        vec![OutputModality::Text, OutputModality::Image],
    );
    model.headers = Some([("HTTP-Referer".to_owned(), "https://example.com".to_owned())].into());

    let output = generate_images(&model, &context(), options("test")).await;

    assert_eq!(output.stop_reason, ImagesStopReason::Stop, "{:?}", output.error_message);
    assert_eq!(output.response_id.as_deref(), Some("img-1"));
    assert_eq!(output.output[0], UserBlock::text("Here is your image."));
    assert_eq!(
        output.output[1],
        UserBlock::Image(ImageContent {
            data: "ZmFrZS1wbmc=".into(),
            mime_type: "image/png".into()
        })
    );
    let usage = output.usage.unwrap();
    assert_eq!((usage.input, usage.output, usage.total_tokens), (12, 34, 46));
    assert!((usage.cost.total - (12.0 * 0.015 + 34.0 * 0.03) / 1_000_000.0).abs() < 1e-15);

    let request = last_request(&server).await;
    let params: Value = request.body_json().unwrap();
    assert_eq!(params["stream"], json!(false));
    assert_eq!(params["modalities"], json!(["image", "text"]));
    assert_eq!(params["model"], json!("google/gemini-3.1-flash-image-preview"));
    assert_eq!(
        params["messages"][0],
        json!({ "role": "user", "content": [{ "type": "text", "text": "Generate a dog" }] })
    );
    let header = |name: &str| request.headers.get(name).and_then(|value| value.to_str().ok());
    assert_eq!(header("authorization"), Some("Bearer test"));
    assert_eq!(header("http-referer"), Some("https://example.com"));
}

#[tokio::test]
async fn passes_through_cancellation_and_returns_aborted_result() {
    let server = serve(200, response()).await;
    let model = model(
        &format!("{}/api/v1", server.uri()),
        "black-forest-labs/flux.2-pro",
        vec![OutputModality::Image],
    );
    let cancel = CancellationToken::new();
    cancel.cancel();
    let mut options = options("test");
    options.base.cancel = Some(cancel);

    let output = generate_images(&model, &context(), options).await;

    assert_eq!(output.stop_reason, ImagesStopReason::Aborted);
    assert_eq!(output.error_message.as_deref(), Some("Request aborted"));
}

#[tokio::test]
async fn generate_images_resolves_the_final_assistant_images_result() {
    let server = serve(200, response()).await;
    let model = model(
        &format!("{}/api/v1", server.uri()),
        "black-forest-labs/flux.2-pro",
        vec![OutputModality::Image],
    );

    let output = generate_images(&model, &context(), options("test")).await;

    assert!(output.output.iter().any(|block| matches!(block, UserBlock::Image(_))));
    // Image-only models must not request text output.
    let params: Value = last_request(&server).await.body_json().unwrap();
    assert_eq!(params["modalities"], json!(["image"]));
}

#[tokio::test]
async fn sends_image_input_as_data_urls_and_reads_object_image_urls() {
    let server = serve(
        200,
        json!({
            "id": "img-2",
            "choices": [{ "message": { "content": "", "images": [
                { "image_url": { "url": "data:image/jpeg;base64,anBn" } },
                { "image_url": { "url": "https://example.com/not-inline.png" } }
            ] } }]
        }),
    )
    .await;
    let model = model(&format!("{}/api/v1", server.uri()), "m", vec![OutputModality::Image]);
    let context = ImagesContext::new(vec![
        UserBlock::text("Make it blue"),
        UserBlock::Image(ImageContent {
            data: "aW1n".into(),
            mime_type: "image/png".into(),
        }),
    ]);

    let output = generate_images(&model, &context, options("test")).await;

    assert_eq!(output.stop_reason, ImagesStopReason::Stop);
    assert_eq!(
        output.output,
        [UserBlock::Image(ImageContent {
            data: "anBn".into(),
            mime_type: "image/jpeg".into()
        })]
    );
    assert_eq!(output.usage, None);
    let params: Value = last_request(&server).await.body_json().unwrap();
    assert_eq!(
        params["messages"][0]["content"][1],
        json!({ "type": "image_url", "image_url": { "url": "data:image/png;base64,aW1n" } })
    );
}

#[tokio::test]
async fn calls_request_hooks_and_reports_errors_as_results() {
    let server = serve(200, response()).await;
    let model = model(&format!("{}/api/v1", server.uri()), "m", vec![OutputModality::Image]);
    let statuses = Arc::new(Mutex::new(Vec::new()));
    let mut options = options("test");
    options.base.on_payload = Some(Arc::new(|payload: &Value, _model: &ImageModel| {
        let mut payload = payload.clone();
        payload["extra"] = json!(true);
        Box::pin(async move { Ok(Some(payload)) })
    }));
    let seen = statuses.clone();
    options.base.on_response = Some(Arc::new(move |response, _model| {
        seen.lock().unwrap().push(response.status);
        Box::pin(async { Ok(()) })
    }));

    let output = generate_images(&model, &context(), options).await;

    assert_eq!(output.stop_reason, ImagesStopReason::Stop);
    assert_eq!(*statuses.lock().unwrap(), [200]);
    let params: Value = last_request(&server).await.body_json().unwrap();
    assert_eq!(params["extra"], json!(true));

    let missing_key = generate_images(&model, &context(), ImagesOptions::default()).await;
    assert_eq!(missing_key.stop_reason, ImagesStopReason::Error);
    assert_eq!(
        missing_key.error_message.as_deref(),
        Some("No API key for provider: openrouter")
    );

    let failing = serve(401, json!({ "error": { "message": "Invalid key", "code": 401 } })).await;
    let model = self::model(&format!("{}/api/v1", failing.uri()), "m", vec![OutputModality::Image]);
    let result = generate_images(&model, &context(), self::options("bad")).await;
    assert_eq!(result.stop_reason, ImagesStopReason::Error);
    // As upstream's `formatProviderError`: the SDK message ("401 Invalid key") lacks the
    // error body, so the body is shown after the status.
    assert_eq!(
        result.error_message.as_deref(),
        Some(r#"401: {"message":"Invalid key","code":401}"#)
    );
    // Upstream runs the SDK without retries unless `max_retries` is set.
    assert_eq!(failing.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn merges_model_and_request_headers_as_sdk_default_headers() {
    let server = serve(200, response()).await;
    let mut model = model(&format!("{}/api/v1", server.uri()), "m", vec![OutputModality::Image]);
    model.headers = Some(
        [
            ("X-Title".to_owned(), "model".to_owned()),
            ("HTTP-Referer".to_owned(), "https://model.example".to_owned()),
        ]
        .into(),
    );
    let mut options = options("test");
    options.base.headers = Some(
        [
            ("x-title".to_owned(), Some("request".to_owned())),
            ("HTTP-Referer".to_owned(), None),
            // `None` values are dropped before the SDK sees them; they cannot remove defaults.
            ("Authorization".to_owned(), None),
        ]
        .into(),
    );

    generate_images(&model, &context(), options).await;

    let request = last_request(&server).await;
    let header = |name: &str| request.headers.get(name).and_then(|value| value.to_str().ok());
    assert_eq!(header("x-title"), Some("request"));
    assert_eq!(header("http-referer"), None);
    assert_eq!(header("authorization"), Some("Bearer test"));
}

#[tokio::test]
async fn builtin_openrouter_provider_generates_images() {
    let server = serve(200, response()).await;
    let models = builtin_models(CreateModelsOptions::default());
    let provider = models.provider("openrouter").unwrap();
    assert!(provider.supports_images());
    let Some(AnyModel::Image(mut model)) =
        models.get_model_of_type(ModelType::Image, "openrouter", "google/gemini-2.5-flash-image")
    else {
        panic!("missing OpenRouter image model");
    };
    model.base_url = format!("{}/api/v1", server.uri());

    let result = models.generate_images(&model, &context(), options("explicit")).await;

    assert_eq!(result.stop_reason, ImagesStopReason::Stop, "{:?}", result.error_message);
    assert!(result.output.iter().any(|block| matches!(block, UserBlock::Image(_))));
    let request = last_request(&server).await;
    assert_eq!(
        request.headers.get("authorization").unwrap().to_str().unwrap(),
        "Bearer explicit"
    );
}
