//! Tests of the `cloudflare-workers-ai-system-one` API through the `cloudflare-workers-ai`
//! provider, against local servers (port of upstream
//! `test/cloudflare-workers-ai-system-one.test.ts`).

#![allow(clippy::unwrap_used)] // Test helpers outside `#[test]` functions.

use delta_ai::classifier::{ClassifierBoolAnswer, ClassifierChoiceAnswer};
use delta_ai::providers::catalog::builtin_model_of_type;
use delta_ai::providers::cloudflare::cloudflare_workers_ai_provider;
use delta_ai::{
    AnyModel, ClassifierAnswer, ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierStopReason, ModelType,
    Models, RequestOptions,
};
use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

fn context() -> ClassifierContext {
    serde_json::from_value(json!({
        "state": { "message": "Help! My payouts have been failing for 3 days." },
        "questions": {
            "is_urgent": {
                "type": "bool",
                "instructions": "Does this convey urgency?",
                "criteria": { "true": "Explicitly time-sensitive", "false": "No urgency expressed" }
            },
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": { "billing": "Payments", "technical": "Bugs" }
            }
        }
    }))
    .unwrap()
}

/// Model output from <https://developers.cloudflare.com/ai/models/typesafe/jev/>.
fn jev_output() -> Value {
    json!({
        "model": "jev-1.13.0",
        "answers": {
            "is_urgent": { "type": "noul", "noul": 0.95 },
            "department": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.8,
                "probabilities": { "billing": 0.87, "technical": 0.13 }
            }
        },
        "usage": { "input_tokens": 426, "output_tokens": 73 }
    })
}

/// REST envelope observed from the live `/ai/run` endpoint.
fn rest_response(state: &str, result: &Value) -> Value {
    json!({
        "result": { "state": state, "result": result, "gatewayMetadata": { "keySource": "Unified" } },
        "success": true,
        "errors": [],
        "messages": []
    })
}

/// Cloudflare-hosted output observed from a live `/ai/run` call, question ids renamed to
/// match `context`. The envelope carries the output directly, without a run record.
fn clef_output() -> Value {
    json!({
        "model": "clef",
        "answers": {
            "is_urgent": { "type": "noul", "noul": 0.9912 },
            "department": {
                "type": "choice",
                "choice": "technical",
                "probabilities": { "billing": 0.1632, "technical": 0.8368 },
                "confidence": 0.4538
            }
        },
        "usage": { "input_tokens": 222, "output_tokens": 0 }
    })
}

fn setup() -> Models {
    let models = Models::default();
    models.set_provider(cloudflare_workers_ai_provider());
    models
}

/// A catalog classifier whose endpoint host is the mock server; the placeholder stays.
fn classifier(models: &Models, id: &str, server: &MockServer) -> ClassifierModel {
    let Some(AnyModel::Classifier(mut model)) =
        models.get_model_of_type(ModelType::Classifier, "cloudflare-workers-ai", id)
    else {
        panic!("missing Cloudflare {id} model");
    };
    model.base_url = model.base_url.replace("https://api.cloudflare.com", &server.uri());
    assert!(model.base_url.contains("{CLOUDFLARE_ACCOUNT_ID}"), "{}", model.base_url);
    model
}

fn auth() -> ClassifierOptions {
    RequestOptions {
        api_key: Some("cf-key".into()),
        env: Some([("CLOUDFLARE_ACCOUNT_ID".to_owned(), "account-id".to_owned())].into()),
        ..RequestOptions::default()
    }
    .into()
}

async fn serve(body: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    server
}

#[test]
fn exposes_jev_only_through_classifier_catalog_accessors() {
    let models = setup();
    let jev = models
        .get_model_of_type(ModelType::Classifier, "cloudflare-workers-ai", "typesafe/jev")
        .unwrap();
    assert_eq!(
        Some(jev.clone()),
        builtin_model_of_type(ModelType::Classifier, "cloudflare-workers-ai", "typesafe/jev")
    );
    assert_eq!(jev.api(), "cloudflare-workers-ai-system-one");
    assert!(models.get_model("cloudflare-workers-ai", "typesafe/jev").is_none());
}

#[tokio::test]
async fn runs_jev_through_the_account_scoped_ai_run_endpoint() {
    let server = serve(rest_response("Completed", &jev_output())).await;
    let models = setup();
    let jev = classifier(&models, "typesafe/jev", &server);

    let result = models.classify(&jev, &context(), auth()).await;

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests[0].url.path(), "/client/v4/accounts/account-id/ai/run");
    let payload: Value = requests[0].body_json().unwrap();
    assert_eq!(payload["model"], json!("typesafe/jev"));
    assert_eq!(payload["input"]["state"], json!(context().state));
    assert_eq!(payload["input"]["questions"]["is_urgent"]["type"], json!("noul"));
    assert_eq!(payload["input"]["questions"]["department"]["type"], json!("choice"));
    assert_eq!(requests[0].headers.get("authorization").unwrap(), "Bearer cf-key");

    assert_eq!(
        result.stop_reason,
        ClassifierStopReason::Stop,
        "{:?}",
        result.error_message
    );
    assert_eq!(
        result.answers["is_urgent"],
        ClassifierAnswer::Bool(ClassifierBoolAnswer { probability: 0.95 })
    );
    let ClassifierAnswer::Choice(ClassifierChoiceAnswer { choice, confidence, .. }) = &result.answers["department"]
    else {
        panic!("expected a choice answer");
    };
    assert_eq!((choice.as_str(), *confidence), ("billing", 0.8));
    let usage = result.usage.unwrap();
    assert_eq!((usage.input, usage.output, usage.total_tokens), (426, 73, 499));
}

#[tokio::test]
async fn runs_clef_models_through_ai_run_and_parses_their_direct_output() {
    for (id, input_price) in [("@cf/cloudflare/clef", 0.24), ("@cf/cloudflare/clef-flash", 0.09)] {
        let server = serve(json!({ "result": clef_output(), "success": true, "errors": [], "messages": [] })).await;
        let models = setup();
        let clef = classifier(&models, id, &server);

        let result = models.classify(&clef, &context(), auth()).await;

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests[0].url.path(), "/client/v4/accounts/account-id/ai/run");
        let payload: Value = requests[0].body_json().unwrap();
        assert_eq!(payload["model"], json!(id));
        assert_eq!(payload["input"]["state"], json!(context().state));
        assert_eq!(payload["input"]["questions"]["is_urgent"]["type"], json!("noul"));
        assert_eq!(
            result.stop_reason,
            ClassifierStopReason::Stop,
            "{id}: {:?}",
            result.error_message
        );
        assert_eq!(
            result.answers["is_urgent"],
            ClassifierAnswer::Bool(ClassifierBoolAnswer { probability: 0.9912 })
        );
        let ClassifierAnswer::Choice(ClassifierChoiceAnswer { choice, confidence, .. }) = &result.answers["department"]
        else {
            panic!("expected a choice answer");
        };
        assert_eq!((choice.as_str(), *confidence), ("technical", 0.4538));
        let usage = result.usage.unwrap();
        assert_eq!((usage.input, usage.output, usage.total_tokens), (222, 0, 222));
        assert!((usage.cost.input - 222.0 * input_price / 1_000_000.0).abs() < 1e-12);
    }
}

#[tokio::test]
async fn reports_runs_that_did_not_complete() {
    let server = serve(rest_response("Queued", &Value::Null)).await;
    let models = setup();
    let jev = classifier(&models, "typesafe/jev", &server);

    let result = models.classify(&jev, &context(), auth()).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Error);
    assert!(
        result
            .error_message
            .unwrap()
            .contains("run did not complete (state: Queued)")
    );
}

#[tokio::test]
async fn reports_cloudflare_envelope_errors() {
    let server =
        serve(json!({ "success": false, "errors": [{ "code": 5007, "message": "No such model" }], "result": null }))
            .await;
    let models = setup();
    let jev = classifier(&models, "typesafe/jev", &server);

    let result = models.classify(&jev, &context(), auth()).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Error);
    assert!(
        result
            .error_message
            .unwrap()
            .contains("Cloudflare Workers AI error: No such model")
    );
}
