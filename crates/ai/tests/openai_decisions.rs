//! Tests of the `openai-decisions` API against local servers (port of upstream
//! `test/openai-decisions.test.ts`).

#![allow(clippy::unwrap_used)] // Test helpers outside `#[test]` functions.

use delta_ai::api::openai_decisions::classify;
use delta_ai::model::ModelCostTier;
use delta_ai::{
    ClassifierAnswer, ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierStopReason, ImageContent,
    InputModality, ModelCost, RequestOptions,
};
use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

fn model(base_url: &str) -> ClassifierModel {
    ClassifierModel {
        id: "gpt-6-luna".into(),
        name: "GPT-6 Luna".into(),
        api: "openai-decisions".into(),
        provider: "openai".into(),
        base_url: base_url.into(),
        input: vec![InputModality::Text, InputModality::Image],
        input_limits: None,
        cost: ModelCost {
            input: 0.1,
            tiers: Some(vec![ModelCostTier {
                input_tokens_above: 272_000.0,
                input: 0.2,
                ..ModelCostTier::default()
            }]),
            ..ModelCost::default()
        },
        headers: None,
        context_window: 922_000,
    }
}

fn context() -> ClassifierContext {
    serde_json::from_value(json!({
        "state": { "text": "The deployment succeeded, thank you." },
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Classify the message",
                "criteria": { "success": "Successful", "failure": "" }
            },
            "satisfaction": {
                "type": "score",
                "instructions": "Score satisfaction",
                "criteria": ["low", "neutral", "high"]
            },
            "approved": {
                "type": "bool",
                "instructions": "Does the user approve?",
                "criteria": { "true": "Approval", "false": "No approval" }
            }
        }
    }))
    .unwrap()
}

/// Response shape from the API reference and live `gpt-6-luna` requests.
fn wire_answers() -> Vec<Value> {
    vec![
        json!({
            "type": "choice",
            "name": "category",
            "choice": "success",
            "probabilities": [
                { "value": "success", "probability": 0.9 },
                { "value": "failure", "probability": 0.1 }
            ],
            "confidence": 0.8
        }),
        json!({
            "type": "score",
            "name": "satisfaction",
            "score": 1.8,
            "probabilities": [
                { "value": 0, "label": "low", "probability": 0.05 },
                { "value": 1, "label": "neutral", "probability": 0.1 },
                { "value": 2, "label": "high", "probability": 0.85 }
            ],
            "confidence": 0.7
        }),
        json!({ "type": "predicate", "name": "approved", "probability": 0.95 }),
    ]
}

fn wire_usage() -> Value {
    json!({
        "input_tokens": 164,
        "input_tokens_details": { "cached_tokens": 0, "cache_write_tokens": 0 },
        "output_tokens": 0,
        "output_tokens_details": { "reasoning_tokens": 0 },
        "total_tokens": 164
    })
}

fn image() -> ImageContent {
    ImageContent {
        data: "aW1hZ2U=".into(),
        mime_type: "image/png".into(),
    }
}

fn options(api_key: &str) -> ClassifierOptions {
    RequestOptions {
        api_key: Some(api_key.into()),
        ..RequestOptions::default()
    }
    .into()
}

async fn serve(response: impl Respond + 'static) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST")).respond_with(response).mount(&server).await;
    server
}

fn ok(body: Value) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(body)
}

async fn requests(server: &MockServer) -> Vec<Request> {
    server.received_requests().await.unwrap()
}

/// A model whose base URL is the server's `/v1`.
fn served_model(server: &MockServer) -> ClassifierModel {
    model(&format!("{}/v1", server.uri()))
}

#[tokio::test]
async fn maps_questions_to_decisions_types_and_answers_back_by_name() {
    // Answers out of question order: they are matched by name.
    let mut reversed = wire_answers();
    reversed.reverse();
    let server = serve(ok(
        json!({ "model": "gpt-6-luna", "answers": reversed, "usage": wire_usage() }),
    ))
    .await;
    let mut options = options("secret");
    options.temperature = Some(1.5);

    let result = classify(&served_model(&server), &context(), options).await;

    let sent = requests(&server).await;
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].url.path(), "/v1/decisions");
    assert_eq!(sent[0].headers.get("authorization").unwrap(), "Bearer secret");
    assert_eq!(
        sent[0].body_json::<Value>().unwrap(),
        json!({
            "model": "gpt-6-luna",
            "input": serde_json::to_string(&context().state).unwrap(),
            "questions": [
                {
                    "type": "choice",
                    "name": "category",
                    "instructions": "Classify the message",
                    // Empty descriptions are omitted.
                    "choices": [{ "value": "success", "description": "Successful" }, { "value": "failure" }]
                },
                {
                    "type": "score",
                    "name": "satisfaction",
                    "instructions": "Score satisfaction",
                    "levels": [{ "label": "low" }, { "label": "neutral" }, { "label": "high" }]
                },
                {
                    "type": "predicate",
                    "name": "approved",
                    "instructions": "Does the user approve?\n\nTrue means: Approval\nFalse means: No approval"
                }
            ]
        })
    );
    assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
    assert_eq!(
        serde_json::to_value(&result.answers).unwrap(),
        json!({
            "category": {
                "type": "choice",
                "choice": "success",
                "probabilities": { "success": 0.9, "failure": 0.1 },
                "confidence": 0.8
            },
            "satisfaction": { "type": "score", "score": 1.8, "confidence": 0.7 },
            "approved": { "type": "bool", "probability": 0.95 }
        })
    );
    let usage = result.usage.unwrap();
    assert_eq!(
        (usage.input, usage.output, usage.cache_read, usage.total_tokens),
        (164, 0, 0, 164)
    );
    assert!((usage.cost.total - 0.000_016_4).abs() < 1e-12);
}

#[tokio::test]
async fn prices_long_context_requests_at_the_long_context_input_rate() {
    let server = serve(ok(
        json!({ "answers": wire_answers(), "usage": { "input_tokens": 300_000, "output_tokens": 0 } }),
    ))
    .await;

    let result = classify(&served_model(&server), &context(), options("secret")).await;

    assert!((result.usage.unwrap().cost.total - 0.06).abs() < 1e-12);
}

#[tokio::test]
async fn sends_images_after_the_state_in_one_user_message() {
    let server = serve(ok(json!({ "answers": wire_answers() }))).await;
    let jpeg = ImageContent {
        mime_type: "image/jpeg".into(),
        ..image()
    };
    let context = ClassifierContext {
        images: Some(vec![image(), jpeg]),
        ..context()
    };

    let result = classify(&served_model(&server), &context, options("secret")).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
    let payload: Value = requests(&server).await[0].body_json().unwrap();
    assert_eq!(
        payload["input"],
        json!([{
            "role": "user",
            "content": [
                { "type": "input_text", "text": serde_json::to_string(&context.state).unwrap() },
                { "type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U=" },
                { "type": "input_image", "image_url": "data:image/jpeg;base64,aW1hZ2U=" }
            ]
        }])
    );
}

#[tokio::test]
async fn rejects_more_than_128_images_before_sending() {
    let server = serve(ok(json!({ "answers": wire_answers() }))).await;
    let context = ClassifierContext {
        images: Some(vec![image(); 129]),
        ..context()
    };

    let result = classify(&served_model(&server), &context, options("secret")).await;

    assert!(requests(&server).await.is_empty());
    assert_eq!(result.stop_reason, ClassifierStopReason::Error);
    assert!(result.error_message.unwrap().contains("at most 128 images, got 129"));
}

#[tokio::test]
async fn fails_the_result_when_a_question_is_refused_and_keeps_the_billed_usage() {
    let answers = wire_answers();
    let server = serve(ok(json!({
        "answers": [answers[0], answers[1], { "type": "refusal", "name": "approved" }],
        "usage": wire_usage()
    })))
    .await;

    let result = classify(&served_model(&server), &context(), options("secret")).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Error);
    assert!(result.answers.is_empty());
    assert_eq!(
        result.error_message.as_deref(),
        Some("OpenAI Decisions refused to answer approved")
    );
    assert_eq!(result.usage.unwrap().input, 164);
}

#[tokio::test]
async fn returns_missing_and_mistyped_answers_as_classifier_errors() {
    let answers = wire_answers();
    let missing_server = serve(ok(json!({ "answers": answers[..2] }))).await;
    let mistyped_server = serve(ok(json!({
        "answers": [answers[0], answers[1], { "type": "score", "name": "approved" }]
    })))
    .await;

    let missing = classify(&served_model(&missing_server), &context(), options("secret")).await;
    let mistyped = classify(&served_model(&mistyped_server), &context(), options("secret")).await;

    assert_eq!(missing.stop_reason, ClassifierStopReason::Error);
    assert!(
        missing
            .error_message
            .unwrap()
            .contains("did not return an answer for approved")
    );
    assert_eq!(mistyped.stop_reason, ClassifierStopReason::Error);
    assert!(
        mistyped
            .error_message
            .unwrap()
            .contains("did not return a predicate answer for approved")
    );
}

#[tokio::test]
async fn preserves_prototype_sensitive_question_ids_in_answers() {
    let server = serve(ok(
        json!({ "answers": [{ "type": "predicate", "name": "__proto__", "probability": 0.75 }] }),
    ))
    .await;
    let context: ClassifierContext = serde_json::from_value(json!({
        "state": {},
        "questions": {
            "__proto__": { "type": "bool", "instructions": "Is this true?", "criteria": { "true": "Yes", "false": "No" } }
        }
    }))
    .unwrap();

    let result = classify(&served_model(&server), &context, options("secret")).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
    assert_eq!(
        result.answers["__proto__"],
        ClassifierAnswer::Bool(delta_ai::classifier::ClassifierBoolAnswer { probability: 0.75 })
    );
}

#[tokio::test]
async fn does_not_retry_gateway_timeouts_and_explains_them_instead_of_returning_the_html_page() {
    let server = serve(
        ResponseTemplate::new(504)
            .insert_header("retry-after-ms", "0")
            .set_body_string("<!DOCTYPE html><html>Gateway time-out</html>"),
    )
    .await;

    // Default retries: the same input would time out again.
    let result = classify(&served_model(&server), &context(), options("secret")).await;

    assert_eq!(requests(&server).await.len(), 1);
    assert_eq!(result.stop_reason, ClassifierStopReason::Error);
    let message = result.error_message.unwrap();
    assert!(message.contains("OpenAI Decisions error (504): the request timed out at the gateway"));
    assert!(!message.contains("<html>"));
}

/// Answers the first request with 503, later ones with the answers.
struct BusyOnce(AtomicUsize);

impl Respond for BusyOnce {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            ResponseTemplate::new(503)
                .insert_header("retry-after-ms", "0")
                .set_body_string("busy")
        } else {
            ok(json!({ "answers": wire_answers() }))
        }
    }
}

#[tokio::test]
async fn still_retries_other_server_errors() {
    let server = serve(BusyOnce(AtomicUsize::new(0))).await;

    let result = classify(&served_model(&server), &context(), options("secret")).await;

    assert_eq!(requests(&server).await.len(), 2);
    assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
}

#[tokio::test]
async fn includes_the_api_error_body_for_other_http_failures() {
    let server = serve(ResponseTemplate::new(400).set_body_json(json!({
        "error": { "message": "Decision input exceeds the token limit.", "type": "invalid_request_error" }
    })))
    .await;
    let mut options = options("secret");
    options.base.max_retries = Some(0);

    let result = classify(&served_model(&server), &context(), options).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Error);
    let message = result.error_message.unwrap();
    assert!(message.contains("OpenAI Decisions error (400)"), "{message}");
    assert!(message.contains("Decision input exceeds the token limit."), "{message}");
}

#[tokio::test]
async fn rejects_models_for_other_classifier_apis_and_missing_api_keys() {
    let server = serve(ok(json!({ "answers": wire_answers() }))).await;
    let mut other = served_model(&server);
    other.api = "typesafe-system-one".into();

    let other_api = classify(&other, &context(), options("secret")).await;
    let no_key = classify(&served_model(&server), &context(), ClassifierOptions::default()).await;

    assert!(requests(&server).await.is_empty());
    assert!(
        other_api
            .error_message
            .unwrap()
            .contains("Unsupported classifier API: typesafe-system-one")
    );
    assert!(
        no_key
            .error_message
            .unwrap()
            .contains("No API key for provider: openai")
    );
}
