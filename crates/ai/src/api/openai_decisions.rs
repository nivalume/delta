//! OpenAI's Decisions API (upstream `api/openai-decisions.ts`): `POST /v1/decisions` with
//! `{ model, input, questions }`. <https://developers.openai.com/api/docs/guides/decisions>
//!
//! The state is sent as JSON text. With images, the input becomes one user message with the
//! state as `input_text` followed by `input_image` data URLs. Questions map to Decisions
//! types: `choice` to `choice`, `score` to `score` and `bool` to `predicate`. Predicates have
//! no criteria field, so the meanings of true and false are appended to the instructions.
//!
//! Only OpenAI API keys work: Sign in with ChatGPT tokens are rejected on this route.

use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde_json::{Value, json};
use std::collections::HashMap;

use crate::api::classifier_shared::{
    as_record, endpoint_url, fail_result, format_classifier_error, parse_classifier_usage, post_classifier_request,
    required_number,
};
use crate::classifier::{
    ClassifierAnswer, ClassifierBoolAnswer, ClassifierBoolQuestion, ClassifierChoiceAnswer, ClassifierContext,
    ClassifierOptions, ClassifierQuestion, ClassifierResult, ClassifierScoreAnswer, ProviderClassifier,
};
use crate::error::{Error, Result};
use crate::model::ClassifierModel;
use crate::types::JsonObject;

/// API id of this implementation.
pub const API: &str = "openai-decisions";

const LABEL: &str = "OpenAI Decisions";

/// The endpoint accepts at most this many image parts per request.
pub const MAX_IMAGES: usize = 128;

/// Cloudflare in front of api.openai.com answers 504 with an HTML page when a request runs
/// longer than about five seconds. Large inputs, currently above roughly 600K tokens, hit this
/// limit, and retrying the same input runs into it again, so 504 is not retried.
const NO_RETRY_STATUSES: &[u16] = &[504];

fn predicate_instructions(question: &ClassifierBoolQuestion) -> String {
    let meanings: Vec<String> = [
        ("True", &question.criteria.when_true),
        ("False", &question.criteria.when_false),
    ]
    .into_iter()
    .filter(|(_, meaning)| !meaning.is_empty())
    .map(|(answer, meaning)| format!("{answer} means: {meaning}"))
    .collect();
    if meanings.is_empty() {
        question.instructions.clone()
    } else {
        format!("{}\n\n{}", question.instructions, meanings.join("\n"))
    }
}

fn wire_question(name: &str, question: &ClassifierQuestion) -> Value {
    match question {
        ClassifierQuestion::Choice(question) => {
            let choices: Vec<Value> = question
                .criteria
                .iter()
                .map(|(value, description)| {
                    if description.is_empty() {
                        json!({ "value": value })
                    } else {
                        json!({ "value": value, "description": description })
                    }
                })
                .collect();
            json!({ "type": "choice", "name": name, "instructions": question.instructions, "choices": choices })
        }
        ClassifierQuestion::Score(question) => {
            let levels: Vec<Value> = question
                .criteria
                .iter()
                .map(|label| json!({ "label": label }))
                .collect();
            json!({ "type": "score", "name": name, "instructions": question.instructions, "levels": levels })
        }
        ClassifierQuestion::Bool(question) => {
            json!({ "type": "predicate", "name": name, "instructions": predicate_instructions(question) })
        }
    }
}

fn wire_input(context: &ClassifierContext) -> Result<Value> {
    let state = serde_json::to_string(&context.state)?;
    let images = context.images.as_deref().unwrap_or_default();
    if images.is_empty() {
        return Ok(state.into());
    }
    if images.len() > MAX_IMAGES {
        return Err(Error::msg(format!(
            "{LABEL} accepts at most {MAX_IMAGES} images, got {}",
            images.len()
        )));
    }
    let mut content = vec![json!({ "type": "input_text", "text": state })];
    content.extend(images.iter().map(|image| {
        json!({ "type": "input_image", "image_url": format!("data:{};base64,{}", image.mime_type, image.data) })
    }));
    Ok(json!([{ "role": "user", "content": content }]))
}

fn choice_probabilities(value: Option<&Value>, id: &str) -> Result<IndexMap<String, f64>> {
    let invalid = || Error::msg(format!("{LABEL} returned invalid probabilities for {id}"));
    let entries = value.and_then(Value::as_array).ok_or_else(invalid)?;
    entries
        .iter()
        .map(|entry| {
            let entry = entry.as_object().ok_or_else(invalid)?;
            let value = entry.get("value").and_then(Value::as_str).ok_or_else(invalid)?;
            let probability = required_number(
                LABEL,
                entry.get("probability"),
                &format!("probability for {id}.{value}"),
            )?;
            Ok((value.to_owned(), probability))
        })
        .collect()
}

fn parse_answer(id: &str, question: &ClassifierQuestion, answer: &JsonObject) -> Result<ClassifierAnswer> {
    let kind = answer.get("type").and_then(Value::as_str);
    if kind == Some("refusal") {
        return Err(Error::msg(format!("{LABEL} refused to answer {id}")));
    }
    let confidence = || required_number(LABEL, answer.get("confidence"), &format!("confidence for {id}"));
    Ok(match question {
        ClassifierQuestion::Choice(_) => {
            let (Some("choice"), Some(choice)) = (kind, answer.get("choice").and_then(Value::as_str)) else {
                return Err(Error::msg(format!("{LABEL} did not return a choice answer for {id}")));
            };
            ClassifierAnswer::Choice(ClassifierChoiceAnswer {
                choice: choice.to_owned(),
                probabilities: choice_probabilities(answer.get("probabilities"), id)?,
                confidence: confidence()?,
            })
        }
        ClassifierQuestion::Score(_) => {
            if kind != Some("score") {
                return Err(Error::msg(format!("{LABEL} did not return a score answer for {id}")));
            }
            ClassifierAnswer::Score(ClassifierScoreAnswer {
                score: required_number(LABEL, answer.get("score"), &format!("score for {id}"))?,
                confidence: confidence()?,
            })
        }
        ClassifierQuestion::Bool(_) => {
            if kind != Some("predicate") {
                return Err(Error::msg(format!(
                    "{LABEL} did not return a predicate answer for {id}"
                )));
            }
            ClassifierAnswer::Bool(ClassifierBoolAnswer {
                probability: required_number(LABEL, answer.get("probability"), &format!("probability for {id}"))?,
            })
        }
    })
}

fn parse_answers(value: Option<&Value>, context: &ClassifierContext) -> Result<IndexMap<String, ClassifierAnswer>> {
    let Some(value) = value.and_then(Value::as_array) else {
        return Err(Error::msg(format!("{LABEL} returned an unexpected response")));
    };
    let mut by_name: HashMap<&str, &JsonObject> = HashMap::new();
    for answer in value.iter().filter_map(Value::as_object) {
        if let Some(name) = answer.get("name").and_then(Value::as_str) {
            by_name.insert(name, answer);
        }
    }
    context
        .questions
        .iter()
        .map(|(id, question)| {
            let Some(answer) = by_name.get(id.as_str()) else {
                return Err(Error::msg(format!("{LABEL} did not return an answer for {id}")));
            };
            Ok((id.clone(), parse_answer(id, question, answer)?))
        })
        .collect()
}

fn error_message(error: &Error) -> String {
    if matches!(error, Error::Status { status: 504, .. }) {
        return format!(
            "{LABEL} error (504): the request timed out at the gateway. Very large inputs (above roughly 600K tokens) currently exceed its time limit."
        );
    }
    format_classifier_error(error, &format!("{LABEL} error"))
}

async fn run(
    model: &ClassifierModel,
    context: &ClassifierContext,
    options: &ClassifierOptions,
    output: &mut ClassifierResult,
) -> Result<()> {
    if model.api != API {
        return Err(Error::msg(format!("Unsupported classifier API: {}", model.api)));
    }
    let url = endpoint_url(&model.base_url, "decisions")?;
    let questions: Vec<Value> = context
        .questions
        .iter()
        .map(|(id, question)| wire_question(id, question))
        .collect();
    let payload = json!({ "model": model.id, "input": wire_input(context)?, "questions": questions });
    let body = post_classifier_request(LABEL, &url, model, payload, options, NO_RETRY_STATUSES).await?;
    let Some(body) = as_record(Some(&body)) else {
        return Err(Error::msg(format!("{LABEL} returned an unexpected response")));
    };
    // Set before parsing answers: a request with malformed or refused answers was still billed.
    if let Some(usage) = parse_classifier_usage(body.get("usage"), model) {
        output.usage = Some(usage);
    }
    output.answers = parse_answers(body.get("answers"), context)?;
    Ok(())
}

/// Classification through OpenAI's Decisions API. `options.temperature` is ignored.
pub async fn classify(
    model: &ClassifierModel,
    context: &ClassifierContext,
    options: ClassifierOptions,
) -> ClassifierResult {
    let mut output = ClassifierResult::empty(model);
    if let Err(error) = run(model, context, &options, &mut output).await {
        fail_result(&mut output, &options, error_message(&error));
    }
    output
}

/// The `openai-decisions` classifier implementation.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAIDecisionsApi;

impl ProviderClassifier for OpenAIDecisionsApi {
    fn classify<'a>(
        &'a self,
        model: &'a ClassifierModel,
        context: &'a ClassifierContext,
        options: ClassifierOptions,
    ) -> BoxFuture<'a, ClassifierResult> {
        Box::pin(classify(model, context, options))
    }
}
