//! Tests of the `llama-cpp-classify` API against a fake `llama-server` (port of upstream
//! `test/llama-cpp-classify.test.ts`).

#![allow(clippy::unwrap_used)]
// Test helpers outside `#[test]` functions.
// Upstream compares probabilities with `toEqual`; the port checks the same exact values.
#![allow(clippy::float_cmp)]

use delta_ai::api::llama_cpp_classify::{
    answer_from_probabilities, classify, label_probabilities, llama_server_root, peak_confidence, render_question,
};
use delta_ai::classifier::{ClassifierChoiceAnswer, ClassifierScoreAnswer, ClassifierScoreQuestion};
use delta_ai::{
    ClassifierAnswer, ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierQuestion, ClassifierStopReason,
    InputModality, ModelCost, RequestOptions,
};
use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

static SERVER_COUNT: AtomicUsize = AtomicUsize::new(0);

/// A fresh server root per model: label tokens are cached per server and model, and mock
/// servers reuse ports.
fn model(server: &MockServer) -> ClassifierModel {
    let count = SERVER_COUNT.fetch_add(1, Ordering::SeqCst);
    ClassifierModel {
        id: "qwen".into(),
        name: "qwen".into(),
        api: "llama-cpp-classify".into(),
        provider: "llama.cpp".into(),
        base_url: format!("{}/llama-{count}/v1", server.uri()),
        input: vec![InputModality::Text],
        input_limits: None,
        cost: ModelCost::default(),
        headers: None,
        context_window: 32768,
    }
}

type Next = Box<dyn Fn(&str, u64) -> Vec<(&'static str, f64)> + Send + Sync>;
type Template = Box<dyn Fn(&[Value]) -> String + Send + Sync>;
type Tokenize = Box<dyn Fn(&str) -> Vec<u32> + Send + Sync>;

/// Token ids: one per character, the character code.
fn char_tokens(content: &str) -> Vec<u32> {
    content.chars().map(u32::from).collect()
}

/// Maps multi-character labels to single tokens, as a real vocabulary would.
fn word_tokens(content: &str) -> Vec<u32> {
    let mut tokens = Vec::new();
    for part in content.split_inclusive('\n') {
        let (word, newline) = match part.strip_suffix('\n') {
            Some(word) => (word, true),
            None => (part, false),
        };
        match word {
            "Yes" => tokens.push(89),
            "No" => tokens.push(78),
            other => tokens.extend(char_tokens(other)),
        }
        if newline {
            tokens.push(10);
        }
    }
    tokens
}

/// Completion log-probabilities for bool (Yes/No), score and letter labels; the fake tokenizer
/// maps a label to its first character.
fn answer_by_prompt(prompt: &str, _depth: u64) -> Vec<(&'static str, f64)> {
    if prompt.contains("Answer Yes or No.") {
        vec![("Y", -0.05), ("N", -3.0)]
    } else if prompt.contains("Answer with one level number.") {
        vec![("2", -0.2), ("1", -1.8), ("0", -4.0)]
    } else {
        vec![("B", -0.3), ("A", -1.5), ("C", -3.0)]
    }
}

struct FakeServer {
    next: Next,
    template: Option<Template>,
    tokenize: Tokenize,
}

impl Default for FakeServer {
    fn default() -> Self {
        Self {
            next: Box::new(|_, _| vec![("A", -0.1), ("B", -2.5)]),
            template: None,
            tokenize: Box::new(char_tokens),
        }
    }
}

impl Respond for FakeServer {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = request.body_json().unwrap();
        let path = request.url.path();
        if path.ends_with("/tokenize") {
            let tokens = (self.tokenize)(body["content"].as_str().unwrap());
            return ResponseTemplate::new(200).set_body_json(json!({ "tokens": tokens }));
        }
        if path.ends_with("/apply-template") {
            let messages = body["messages"].as_array().unwrap();
            let prompt = match &self.template {
                Some(template) => template(messages),
                None => {
                    let mut rendered = String::new();
                    for message in messages {
                        let role = message["role"].as_str().unwrap();
                        let content = message["content"].as_str().unwrap();
                        write!(rendered, "<|{role}|>\n{content}\n").unwrap();
                    }
                    rendered.push_str("<|assistant|>\n");
                    rendered
                }
            };
            return ResponseTemplate::new(200).set_body_json(json!({ "prompt": prompt }));
        }
        if path.ends_with("/completion") {
            let next = (self.next)(body["prompt"].as_str().unwrap(), body["n_probs"].as_u64().unwrap());
            let top_logprobs: Vec<Value> = next
                .into_iter()
                .map(|(token, logprob)| {
                    json!({ "id": u32::from(token.chars().next().unwrap()), "token": token, "bytes": [], "logprob": logprob })
                })
                .collect();
            return ResponseTemplate::new(200).set_body_json(json!({
                "content": "A",
                "completion_probabilities": [{ "id": 65, "token": "A", "top_logprobs": top_logprobs }]
            }));
        }
        ResponseTemplate::new(404).set_body_string("not found")
    }
}

async fn serve(fake: impl Respond + 'static) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST")).respond_with(fake).mount(&server).await;
    server
}

async fn requests(server: &MockServer) -> Vec<Request> {
    server.received_requests().await.unwrap()
}

fn ending_with(requests: &[Request], suffix: &str) -> Vec<Value> {
    requests
        .iter()
        .filter(|request| request.url.path().ends_with(suffix))
        .map(|request| request.body_json().unwrap())
        .collect()
}

fn completion_depths(requests: &[Request]) -> Vec<u64> {
    ending_with(requests, "/completion")
        .iter()
        .map(|body| body["n_probs"].as_u64().unwrap())
        .collect()
}

fn context() -> ClassifierContext {
    serde_json::from_value(json!({
        "state": { "message": "Help! My payouts have been failing for 3 days." },
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": { "billing": "Payments and refunds", "technical": "Bugs and outages", "sales": "" }
            },
            "urgent": {
                "type": "bool",
                "instructions": "Does this convey urgency?",
                "criteria": { "true": "The user needs help soon", "false": "No time pressure" }
            },
            "severity": { "type": "score", "instructions": "How severe is this?", "criteria": ["low", "medium", "high"] }
        }
    }))
    .unwrap()
}

fn pick(instructions: &str) -> ClassifierContext {
    serde_json::from_value(json!({
        "state": {},
        "questions": { "pick": { "type": "choice", "instructions": instructions, "criteria": { "a": "", "b": "" } } }
    }))
    .unwrap()
}

fn options() -> ClassifierOptions {
    ClassifierOptions::default()
}

#[tokio::test]
async fn answers_choice_bool_and_score_questions_from_label_log_probabilities() {
    let server = serve(FakeServer {
        next: Box::new(answer_by_prompt),
        tokenize: Box::new(word_tokens),
        ..FakeServer::default()
    })
    .await;
    let model = model(&server);
    let options: ClassifierOptions = RequestOptions {
        api_key: Some("local".into()),
        ..RequestOptions::default()
    }
    .into();

    let result = classify(&model, &context(), options).await;

    assert_eq!(result.error_message, None);
    assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
    let choice = label_probabilities(&[-1.5, -0.3, -3.0], 1.0);
    assert_eq!(
        result.answers["team"],
        ClassifierAnswer::Choice(ClassifierChoiceAnswer {
            choice: "technical".into(),
            probabilities: [
                ("billing".to_owned(), choice[0]),
                ("technical".to_owned(), choice[1]),
                ("sales".to_owned(), choice[2]),
            ]
            .into_iter()
            .collect(),
            confidence: peak_confidence(&choice),
        })
    );
    assert_eq!(
        serde_json::to_value(&result.answers["urgent"]).unwrap(),
        json!({ "type": "bool", "probability": label_probabilities(&[-0.05, -3.0], 1.0)[0] })
    );
    let levels = label_probabilities(&[-4.0, -1.8, -0.2], 1.0);
    assert_eq!(
        result.answers["severity"],
        ClassifierAnswer::Score(ClassifierScoreAnswer {
            score: levels[1] + 2.0 * levels[2],
            confidence: peak_confidence(&levels),
        })
    );

    let root = model.base_url.strip_suffix("/v1").unwrap();
    let root_path = root.strip_prefix(&server.uri()).unwrap();
    let sent = requests(&server).await;
    for request in &sent {
        assert!(
            request.url.path().starts_with(&format!("{root_path}/")),
            "{}",
            request.url
        );
        let body: Value = request.body_json().unwrap();
        assert_eq!(body["model"], json!("qwen"));
        assert_eq!(request.headers.get("authorization").unwrap(), "Bearer local");
    }
    let completion = &ending_with(&sent, "/completion")[0];
    assert_eq!(completion["n_predict"], json!(1));
    assert_eq!(completion["n_probs"], json!(256));
    assert_eq!(completion["post_sampling_probs"], json!(false));
    assert_eq!(completion["cache_prompt"], json!(true));
    let template = &ending_with(&sent, "/apply-template")[0];
    assert_eq!(template["chat_template_kwargs"], json!({ "enable_thinking": false }));
}

#[test]
fn repeats_the_state_around_all_questions_and_ends_with_this_questions_labels() {
    let rendered = render_question(&context(), "team").unwrap();
    let state = "State:\n{\n \"message\": \"Help! My payouts have been failing for 3 days.\"\n}";
    assert_eq!(rendered.labels, ["A", "B", "C"]);
    assert_eq!(rendered.keys, ["billing", "technical", "sales"]);
    assert_eq!(
        rendered.content,
        [
            state,
            "",
            "Task: answer each of the following questions about the state.",
            "",
            "Question: Which team should handle this?",
            "",
            "Options:",
            "- billing: Payments and refunds",
            "- technical: Bugs and outages",
            "- sales",
            "",
            "Question: Does this convey urgency?",
            "",
            "Yes means: The user needs help soon",
            "No means: No time pressure",
            "",
            "Question: How severe is this?",
            "",
            "Levels:",
            "0. low",
            "1. medium",
            "2. high",
            "",
            state,
            "",
            "Question: Which team should handle this?",
            "",
            "Options:",
            "A. billing: Payments and refunds",
            "B. technical: Bugs and outages",
            "C. sales",
            "",
            "Answer with one letter.",
        ]
        .join("\n")
    );
}

#[test]
fn shares_everything_before_the_final_question_across_the_questions_of_a_request() {
    let prefix = |id: &str| {
        let content = render_question(&context(), id).unwrap().content;
        content[..content.rfind("Question:").unwrap()].to_owned()
    };
    assert_eq!(prefix("urgent"), prefix("team"));
    assert_eq!(prefix("severity"), prefix("team"));
    assert!(
        render_question(&context(), "urgent")
            .unwrap()
            .content
            .ends_with("No means: No time pressure\n\nAnswer Yes or No.")
    );
    assert!(
        render_question(&context(), "severity")
            .unwrap()
            .content
            .ends_with("2. high\n\nAnswer with one level number.")
    );
    assert_eq!(
        render_question(&context(), "missing").unwrap_err().to_string(),
        "Unknown question: missing"
    );
}

#[tokio::test]
async fn divides_label_log_probabilities_by_the_temperature() {
    let server = serve(FakeServer::default()).await;
    let mut options = options();
    options.temperature = Some(2.0);

    let result = classify(&model(&server), &pick("Pick one"), options).await;

    let expected = label_probabilities(&[-0.1 / 2.0, -2.5 / 2.0], 1.0);
    let ClassifierAnswer::Choice(answer) = &result.answers["pick"] else {
        panic!("expected a choice answer: {:?}", result.error_message);
    };
    assert_eq!(answer.probabilities["a"], expected[0]);
    assert_eq!(answer.probabilities["b"], expected[1]);
    assert_eq!(label_probabilities(&[-0.1, -2.5], 2.0), expected);
}

#[tokio::test]
async fn rejects_non_positive_temperatures_before_sending_requests() {
    let server = serve(FakeServer::default()).await;
    let mut options = options();
    options.temperature = Some(0.0);

    let result = classify(&model(&server), &context(), options).await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Error);
    assert!(
        result
            .error_message
            .unwrap()
            .contains("Temperature must be a positive number, got 0")
    );
    assert!(requests(&server).await.is_empty());
}

#[tokio::test]
async fn retries_deeper_readouts_when_a_label_is_missing_and_fails_without_inventing_zeros() {
    let deep = serve(FakeServer {
        next: Box::new(|_, depth| {
            if depth < 4096 {
                vec![("A", -0.1)]
            } else {
                vec![("A", -0.1), ("B", -9.0)]
            }
        }),
        ..FakeServer::default()
    })
    .await;
    let recovered = classify(&model(&deep), &pick("Pick one"), options()).await;
    assert_eq!(recovered.stop_reason, ClassifierStopReason::Stop);
    assert_eq!(completion_depths(&requests(&deep).await), [256, 4096]);

    let never = serve(FakeServer {
        next: Box::new(|_, _| vec![("A", -0.1)]),
        ..FakeServer::default()
    })
    .await;
    let failed = classify(&model(&never), &pick("Pick one"), options()).await;
    assert_eq!(failed.stop_reason, ClassifierStopReason::Error);
    assert!(failed.answers.is_empty());
    assert!(
        failed
            .error_message
            .unwrap()
            .contains("did not rank labels B for pick within the top 32768 tokens")
    );
    assert_eq!(completion_depths(&requests(&never).await), [256, 4096, 32768]);
}

#[tokio::test]
async fn reports_labels_that_get_no_probability() {
    let server = serve(FakeServer {
        next: Box::new(|_, _| vec![("A", -1e30), ("B", -3e38)]),
        ..FakeServer::default()
    })
    .await;

    let result = classify(&model(&server), &pick("Pick"), options()).await;

    assert_eq!(
        result.error_message.as_deref(),
        Some("qwen gave no probability to any answer label for pick")
    );
}

#[tokio::test]
async fn closes_a_reasoning_block_the_template_leaves_open() {
    let server = serve(FakeServer {
        template: Some(Box::new(|_| "<|assistant|>\n<think>".into())),
        ..FakeServer::default()
    })
    .await;

    classify(&model(&server), &pick("Pick"), options()).await;

    let completion = &ending_with(&requests(&server).await, "/completion")[0];
    assert_eq!(completion["prompt"], json!("<|assistant|>\n<think></think>"));
}

#[tokio::test]
async fn reads_labels_in_reply_position_and_rejects_labels_that_are_not_one_token() {
    // A tokenizer that merges a newline with a following letter falls back to the label alone.
    let merging = serve(FakeServer {
        tokenize: Box::new(|content| {
            if content.starts_with('\n') && content.chars().count() > 1 {
                vec![1000]
            } else {
                char_tokens(content)
            }
        }),
        ..FakeServer::default()
    })
    .await;
    let merged = classify(&model(&merging), &pick("Pick"), options()).await;
    assert_eq!(
        merged.stop_reason,
        ClassifierStopReason::Stop,
        "{:?}",
        merged.error_message
    );

    // The default fake tokenizer splits "Yes" into three tokens.
    let split = serve(FakeServer::default()).await;
    let context: ClassifierContext = serde_json::from_value(json!({
        "state": {},
        "questions": { "ok": { "type": "bool", "instructions": "OK?", "criteria": { "true": "", "false": "" } } }
    }))
    .unwrap();
    let result = classify(&model(&split), &context, options()).await;
    assert_eq!(result.stop_reason, ClassifierStopReason::Error);
    assert!(
        result
            .error_message
            .unwrap()
            .contains("Label \"Yes\" is not a single token for qwen")
    );
}

#[tokio::test]
async fn caches_label_tokens_per_server_and_model() {
    let server = serve(FakeServer::default()).await;
    let model = model(&server);

    classify(&model, &pick("Pick"), options()).await;
    let first = ending_with(&requests(&server).await, "/tokenize").len();
    classify(&model, &pick("Pick"), options()).await;

    assert!(first > 0);
    assert_eq!(ending_with(&requests(&server).await, "/tokenize").len(), first);
}

#[tokio::test]
async fn validates_option_counts_before_sending_requests() {
    let server = serve(FakeServer::default()).await;
    let criteria: serde_json::Map<String, Value> = (0..63).map(|index| (format!("option{index}"), json!(""))).collect();
    let too_many: ClassifierContext = serde_json::from_value(json!({
        "state": {},
        "questions": { "pick": { "type": "choice", "instructions": "Pick", "criteria": criteria } }
    }))
    .unwrap();
    let too_few: ClassifierContext = serde_json::from_value(json!({
        "state": {},
        "questions": { "rate": { "type": "score", "instructions": "Rate", "criteria": ["only"] } }
    }))
    .unwrap();

    let too_many = classify(&model(&server), &too_many, options()).await;
    let too_few = classify(&model(&server), &too_few, options()).await;

    assert!(
        too_many
            .error_message
            .unwrap()
            .contains("A choice question needs 2 to 62 options, got 63")
    );
    assert!(
        too_few
            .error_message
            .unwrap()
            .contains("A score question needs 2 to 10 levels, got 1")
    );
    assert!(requests(&server).await.is_empty());
}

#[tokio::test]
async fn passes_completion_payloads_and_responses_through_the_request_hooks() {
    let server = serve(FakeServer::default()).await;
    let payloads = Arc::new(Mutex::new(Vec::new()));
    let statuses = Arc::new(Mutex::new(Vec::new()));
    let mut options = options();
    let seen = payloads.clone();
    options.base.on_payload = Some(Arc::new(move |payload: &Value, _model: &ClassifierModel| {
        seen.lock().unwrap().push(payload.clone());
        let mut payload = payload.clone();
        payload["id_slot"] = json!(1);
        Box::pin(async move { Ok(Some(payload)) })
    }));
    let seen = statuses.clone();
    options.base.on_response = Some(Arc::new(move |response, _model| {
        seen.lock().unwrap().push(response.status);
        Box::pin(async { Ok(()) })
    }));

    classify(&model(&server), &pick("Pick"), options).await;

    let payloads = payloads.lock().unwrap().clone();
    assert_eq!(payloads.len(), 1);
    assert_eq!(payloads[0]["n_predict"], json!(1));
    assert_eq!(*statuses.lock().unwrap(), [200]);
    assert_eq!(
        ending_with(&requests(&server).await, "/completion")[0]["id_slot"],
        json!(1)
    );
}

#[tokio::test]
async fn reports_server_errors_and_cancellation() {
    let failing_server =
        serve(ResponseTemplate::new(400).set_body_string(r#"{"error":{"message":"context overflow"}}"#)).await;
    let mut options = options();
    options.base.max_retries = Some(0);
    let failing = classify(&model(&failing_server), &context(), options).await;
    assert_eq!(failing.stop_reason, ClassifierStopReason::Error);
    let message = failing.error_message.unwrap();
    assert!(message.contains("llama.cpp error (400)"), "{message}");
    assert!(message.contains("context overflow"), "{message}");

    let server = serve(FakeServer::default()).await;
    let cancel = CancellationToken::new();
    cancel.cancel();
    let mut options = self::options();
    options.base.cancel = Some(cancel);
    let aborted = classify(&model(&server), &context(), options).await;
    assert_eq!(aborted.stop_reason, ClassifierStopReason::Aborted);
}

#[tokio::test]
async fn rejects_models_for_other_classifier_apis() {
    let server = serve(FakeServer::default()).await;
    let mut model = model(&server);
    model.api = "typesafe-system-one".into();

    let result = classify(&model, &context(), options()).await;

    assert!(
        result
            .error_message
            .unwrap()
            .contains("Unsupported classifier API: typesafe-system-one")
    );
    assert!(requests(&server).await.is_empty());
}

#[test]
fn derives_the_server_root_from_openai_compatible_base_urls() {
    assert_eq!(llama_server_root("http://127.0.0.1:8080/v1/"), "http://127.0.0.1:8080");
    assert_eq!(
        llama_server_root("https://example.com/prefix/v1"),
        "https://example.com/prefix"
    );
    assert_eq!(llama_server_root("http://127.0.0.1:8080"), "http://127.0.0.1:8080");
}

#[test]
fn computes_typesafes_confidence_and_expected_scores() {
    assert!((peak_confidence(&[0.89, 0.06, 0.05]) - 0.835).abs() < 1e-9);
    assert_eq!(peak_confidence(&[0.5, 0.5]), 0.0);
    assert_eq!(peak_confidence(&[1.0, 0.0, 0.0]), 1.0);
    let question = ClassifierQuestion::Score(ClassifierScoreQuestion {
        instructions: String::new(),
        criteria: vec!["a".into(), "b".into(), "c".into()],
    });
    let keys = ["0".to_owned(), "1".to_owned(), "2".to_owned()];
    let ClassifierAnswer::Score(answer) = answer_from_probabilities(&question, &keys, &[0.2, 0.3, 0.5]) else {
        panic!("expected a score answer");
    };
    assert!((answer.score - 1.3).abs() < 1e-12);
    assert_eq!(answer.confidence, peak_confidence(&[0.2, 0.3, 0.5]));
}
