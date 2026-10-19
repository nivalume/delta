//! Tests of the `typesafe-system-one` API against local servers (port of upstream
//! `test/typesafe-system-one.test.ts`).

#![allow(clippy::unwrap_used)] // Test helpers outside `#[test]` functions.

use delta_ai::api::typesafe_system_one::classify;
use delta_ai::classifier::{ClassifierBoolAnswer, ClassifierChoiceAnswer, ClassifierScoreAnswer};
use delta_ai::{
    ClassifierAnswer, ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierStopReason, ImageContent,
    InputModality, ModelCost, RequestOptions,
};
use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

fn model(base_url: &str) -> ClassifierModel {
    ClassifierModel {
        id: "jev-latest".into(),
        name: "Jev".into(),
        api: "typesafe-system-one".into(),
        provider: "typesafe".into(),
        base_url: base_url.into(),
        input: vec![InputModality::Text],
        input_limits: None,
        cost: ModelCost::default(),
        headers: None,
        context_window: 64000,
    }
}

fn context() -> ClassifierContext {
    serde_json::from_value(json!({
        "state": { "text": "The deployment succeeded, thank you." },
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Classify the message",
                "criteria": { "success": "Successful", "failure": "Failed" }
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

fn wire_answers() -> Value {
    json!({
        "category": {
            "type": "choice",
            "choice": "success",
            "probabilities": { "success": 0.9, "failure": 0.1 },
            "confidence": 0.8
        },
        "satisfaction": { "type": "score", "score": 2, "confidence": 0.7 },
        "approved": { "type": "noul", "noul": 0.95 }
    })
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

async fn requests(server: &MockServer) -> Vec<Request> {
    server.received_requests().await.unwrap()
}

fn ok(body: Value) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(body)
}

#[tokio::test]
async fn maps_public_bool_questions_and_answers_to_typesafe_noul_values() {
    let server = serve(ok(json!({ "answers": wire_answers() }))).await;
    let model = model(&format!("{}/v1/", server.uri()));
    let mut options = options("secret");
    options.temperature = Some(1.5);

    let result = classify(&model, &context(), options).await;

    let sent = requests(&server).await;
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].url.path(), "/v1/systemone");
    let payload: Value = sent[0].body_json().unwrap();
    assert_eq!(payload["model"], json!("jev-latest"));
    assert_eq!(payload["questions"]["category"]["type"], json!("choice"));
    assert_eq!(payload["questions"]["satisfaction"]["type"], json!("score"));
    assert_eq!(payload["questions"]["approved"]["type"], json!("noul"));
    // System One has no temperature field; the option is ignored.
    assert!(payload.get("temperature").is_none());
    assert_eq!(sent[0].headers.get("authorization").unwrap(), "Bearer secret");

    assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
    assert_eq!(
        result.answers["approved"],
        ClassifierAnswer::Bool(ClassifierBoolAnswer { probability: 0.95 })
    );
    let ClassifierAnswer::Choice(ClassifierChoiceAnswer { choice, .. }) = &result.answers["category"] else {
        panic!("expected a choice answer");
    };
    assert_eq!(choice, "success");
    assert_eq!(
        result.answers["satisfaction"],
        ClassifierAnswer::Score(ClassifierScoreAnswer {
            score: 2.0,
            confidence: 0.7
        })
    );
    assert_eq!(result.usage, None);

    let priced_server = serve(ok(
        json!({ "answers": wire_answers(), "usage": { "input_tokens": 308, "output_tokens": 23 } }),
    ))
    .await;
    let mut priced = self::model(&format!("{}/v1/", priced_server.uri()));
    priced.cost.input = 0.042;
    let priced_result = classify(&priced, &context(), self::options("secret")).await;
    let usage = priced_result.usage.unwrap();
    assert_eq!((usage.input, usage.output, usage.total_tokens), (308, 23, 331));
    assert!((usage.cost.total - 0.000_012_936).abs() < 1e-12);
}

#[tokio::test]
async fn posts_openrouter_system_one_requests_to_its_typesafe_compatible_endpoint() {
    // Response shape observed from the live OpenRouter endpoint.
    let server = serve(ok(json!({
        "id": "gen-dec-1",
        "provider": "TypeSafe",
        "answers": wire_answers(),
        "usage": { "input_tokens": 308, "output_tokens": 23, "cost": 0.000_012_936 }
    })))
    .await;
    let mut model = model(&format!("{}/api/v1", server.uri()));
    model.id = "typesafe/jev-1.13".into();
    model.provider = "openrouter".into();
    model.cost.input = 0.042;

    let result = classify(&model, &context(), options("secret")).await;

    let sent = requests(&server).await;
    assert_eq!(sent[0].url.path(), "/api/v1/systemone");
    let payload: Value = sent[0].body_json().unwrap();
    assert_eq!(payload["model"], json!("typesafe/jev-1.13"));
    assert_eq!(payload["state"], json!(context().state));
    assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
    assert_eq!(
        result.answers["approved"],
        ClassifierAnswer::Bool(ClassifierBoolAnswer { probability: 0.95 })
    );
    // Priced from the catalog like chat usage; matches OpenRouter's reported cost.
    assert!((result.usage.unwrap().cost.total - 0.000_012_936).abs() < 1e-12);
}

#[tokio::test]
async fn rejects_models_for_other_classifier_apis() {
    let server = serve(ok(json!({ "answers": wire_answers() }))).await;
    let mut model = model(&server.uri());
    model.api = "cloudflare-workers-ai-system-one".into();

    let result = classify(&model, &context(), options("secret")).await;

    assert!(requests(&server).await.is_empty());
    assert_eq!(result.stop_reason, ClassifierStopReason::Error);
    assert!(
        result
            .error_message
            .unwrap()
            .contains("Unsupported classifier API: cloudflare-workers-ai-system-one")
    );
}

#[tokio::test]
async fn rejects_image_input_before_sending() {
    let server = serve(ok(json!({ "answers": wire_answers() }))).await;
    let mut model = model(&server.uri());
    model.input = vec![InputModality::Text, InputModality::Image];
    let context = ClassifierContext {
        images: Some(vec![ImageContent {
            data: "aW1hZ2U=".into(),
            mime_type: "image/png".into(),
        }]),
        ..context()
    };

    let result = classify(&model, &context, options("secret")).await;

    assert!(requests(&server).await.is_empty());
    assert_eq!(result.stop_reason, ClassifierStopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("System One API does not support image input")
    );
}

#[tokio::test]
async fn merges_headers_case_insensitively_and_supports_none_suppression() {
    let server = serve(ok(json!({ "answers": wire_answers() }))).await;
    let mut model = model(&server.uri());
    model.headers = Some(
        [
            ("authorization".to_owned(), "Bearer model".to_owned()),
            ("X-Source".to_owned(), "model".to_owned()),
        ]
        .into(),
    );

    let mut first = options("secret");
    first.base.headers = Some(
        [
            ("Authorization".to_owned(), Some("Bearer request".to_owned())),
            ("x-source".to_owned(), Some("request".to_owned())),
        ]
        .into(),
    );
    classify(&model, &context(), first).await;
    let mut second = options("secret");
    second.base.headers = Some([("Authorization".to_owned(), None)].into());
    classify(&model, &context(), second).await;

    let sent = requests(&server).await;
    let values = |request: &Request, name: &str| request.headers.get_all(name).iter().count();
    assert_eq!(sent[0].headers.get("authorization").unwrap(), "Bearer request");
    assert_eq!(sent[0].headers.get("x-source").unwrap(), "request");
    assert_eq!(values(&sent[0], "authorization"), 1);
    assert!(sent[1].headers.get("authorization").is_none());
}

#[tokio::test]
async fn preserves_prototype_sensitive_question_ids_in_answers() {
    let server = serve(ok(
        json!({ "answers": { "__proto__": { "type": "noul", "noul": 0.75 } } }),
    ))
    .await;
    let context: ClassifierContext = serde_json::from_value(json!({
        "state": {},
        "questions": {
            "__proto__": { "type": "bool", "instructions": "Is this true?", "criteria": { "true": "Yes", "false": "No" } }
        }
    }))
    .unwrap();

    let result = classify(&model(&server.uri()), &context, options("secret")).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
    assert_eq!(result.answers.keys().collect::<Vec<_>>(), ["__proto__"]);
    assert_eq!(
        serde_json::to_value(&result.answers).unwrap()["__proto__"],
        json!({ "type": "bool", "probability": 0.75 })
    );
}

#[tokio::test]
async fn reports_request_timeouts_separately_from_caller_cancellation() {
    let server = serve(ok(json!({ "answers": wire_answers() })).set_delay(Duration::from_secs(5))).await;
    let mut options = options("secret");
    options.base.timeout = Some(Duration::from_millis(5));
    options.base.max_retries = Some(0);

    let result = classify(&model(&server.uri()), &context(), options).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Error);
    assert_eq!(result.error_message.as_deref(), Some("Request timed out after 5ms"));
}

/// Answers the first request with `first`, later ones with `then`.
struct Sequence {
    calls: AtomicUsize,
    first: ResponseTemplate,
    then: ResponseTemplate,
}

impl Respond for Sequence {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            self.first.clone()
        } else {
            self.then.clone()
        }
    }
}

#[tokio::test]
async fn creates_a_fresh_timeout_for_every_retry_attempt() {
    // Each attempt takes 600 ms: together they exceed the 1 s timeout, one alone does not.
    let delay = Duration::from_millis(600);
    let server = serve(Sequence {
        calls: AtomicUsize::new(0),
        first: ResponseTemplate::new(500)
            .insert_header("retry-after-ms", "0")
            .set_body_string("retry")
            .set_delay(delay),
        then: ok(json!({ "answers": wire_answers() })).set_delay(delay),
    })
    .await;
    let mut options = options("secret");
    options.base.timeout = Some(Duration::from_secs(1));
    options.base.max_retries = Some(1);

    let result = classify(&model(&server.uri()), &context(), options).await;

    assert_eq!(
        result.stop_reason,
        ClassifierStopReason::Stop,
        "{:?}",
        result.error_message
    );
    assert_eq!(requests(&server).await.len(), 2);
}

#[tokio::test]
async fn returns_malformed_responses_as_classifier_errors() {
    let server = serve(ok(
        json!({ "answers": {}, "usage": { "input_tokens": 10, "output_tokens": 2 } }),
    ))
    .await;

    let result = classify(&model(&server.uri()), &context(), options("secret")).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Error);
    assert!(result.answers.is_empty());
    assert!(
        result
            .error_message
            .as_deref()
            .unwrap()
            .contains("did not return an answer for category")
    );
    // The request was billed, so its usage is kept.
    let usage = result.usage.unwrap();
    assert_eq!((usage.input, usage.output), (10, 2));
}

#[tokio::test]
async fn ignores_malformed_usage() {
    let server = serve(ok(
        json!({ "answers": wire_answers(), "usage": { "input_tokens": "many", "output_tokens": 3 } }),
    ))
    .await;
    let without_tokens_server = serve(ok(json!({ "answers": wire_answers(), "usage": { "cost": 0.1 } }))).await;

    let result = classify(&model(&server.uri()), &context(), options("secret")).await;
    let without_tokens = classify(&model(&without_tokens_server.uri()), &context(), options("secret")).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
    let usage = result.usage.unwrap();
    assert_eq!((usage.input, usage.output, usage.total_tokens), (0, 3, 3));
    assert_eq!(without_tokens.stop_reason, ClassifierStopReason::Stop);
    assert_eq!(without_tokens.usage, None);
}

#[tokio::test]
async fn reports_http_errors_with_their_body_and_requires_an_api_key() {
    let server = serve(ResponseTemplate::new(401).set_body_string("invalid key")).await;
    let mut options = options("secret");
    options.base.max_retries = Some(0);

    let result = classify(&model(&server.uri()), &context(), options).await;
    let no_key = classify(&model(&server.uri()), &context(), ClassifierOptions::default()).await;

    assert_eq!(
        result.error_message.as_deref(),
        Some("System One API error (401): invalid key")
    );
    assert_eq!(
        no_key.error_message.as_deref(),
        Some("No API key for provider: typesafe")
    );
    assert_eq!(requests(&server).await.len(), 1);
}
