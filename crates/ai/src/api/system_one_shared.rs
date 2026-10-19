//! TypeSafe's System One protocol over different services (upstream
//! `api/system-one-shared.ts`). A [`SystemOneTransport`] holds what differs between services:
//! the endpoint, the request envelope and the response envelope.

use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use crate::api::classifier_shared::{
    as_record, fail_result, format_classifier_error, parse_classifier_usage, post_classifier_request, required_number,
};
use crate::classifier::{
    ClassifierAnswer, ClassifierBoolAnswer, ClassifierChoiceAnswer, ClassifierContext, ClassifierOptions,
    ClassifierQuestion, ClassifierResult, ClassifierScoreAnswer,
};
use crate::error::{Error, Result};
use crate::model::ClassifierModel;
use crate::types::JsonObject;

/// Differences between services that serve System One models.
#[derive(Clone, Copy, Debug)]
pub struct SystemOneTransport {
    /// Classifier API implemented by this transport.
    pub api: &'static str,
    /// Service name used in error messages.
    pub label: &'static str,
    /// Absolute request URL.
    pub url: fn(&ClassifierModel) -> Result<String>,
    /// Wraps the System One request (`{ state, questions }`) in the service's request envelope.
    pub payload: fn(&ClassifierModel, Value) -> Value,
    /// Extracts the System One output (`{ answers, usage }`) from the service's response
    /// envelope.
    pub output: fn(Value) -> Result<JsonObject>,
}

fn probabilities(label: &str, value: Option<&Value>, id: &str) -> Result<IndexMap<String, f64>> {
    let Some(value) = as_record(value) else {
        return Err(Error::msg(format!("{label} returned invalid probabilities for {id}")));
    };
    value
        .iter()
        .map(|(key, probability)| {
            Ok((
                key.clone(),
                required_number(label, Some(probability), &format!("probability for {id}.{key}"))?,
            ))
        })
        .collect()
}

fn parse_answers(
    label: &str,
    value: Option<&Value>,
    context: &ClassifierContext,
) -> Result<IndexMap<String, ClassifierAnswer>> {
    let Some(value) = as_record(value) else {
        return Err(Error::msg(format!("{label} returned an unexpected response")));
    };
    let mut answers = IndexMap::new();
    for (id, question) in &context.questions {
        let Some(answer) = as_record(value.get(id)) else {
            return Err(Error::msg(format!("{label} did not return an answer for {id}")));
        };
        let kind = answer.get("type").and_then(Value::as_str);
        let parsed = match question {
            ClassifierQuestion::Choice(_) => {
                let (Some("choice"), Some(choice)) = (kind, answer.get("choice").and_then(Value::as_str)) else {
                    return Err(Error::msg(format!("{label} did not return a choice answer for {id}")));
                };
                ClassifierAnswer::Choice(ClassifierChoiceAnswer {
                    choice: choice.to_owned(),
                    probabilities: probabilities(label, answer.get("probabilities"), id)?,
                    confidence: required_number(label, answer.get("confidence"), &format!("confidence for {id}"))?,
                })
            }
            ClassifierQuestion::Score(_) => {
                if kind != Some("score") {
                    return Err(Error::msg(format!("{label} did not return a score answer for {id}")));
                }
                ClassifierAnswer::Score(ClassifierScoreAnswer {
                    score: required_number(label, answer.get("score"), &format!("score for {id}"))?,
                    confidence: required_number(label, answer.get("confidence"), &format!("confidence for {id}"))?,
                })
            }
            ClassifierQuestion::Bool(_) => {
                if kind != Some("noul") {
                    return Err(Error::msg(format!("{label} did not return a bool answer for {id}")));
                }
                ClassifierAnswer::Bool(ClassifierBoolAnswer {
                    probability: required_number(label, answer.get("noul"), &format!("probability for {id}"))?,
                })
            }
        };
        answers.insert(id.clone(), parsed);
    }
    Ok(answers)
}

/// The System One request: public `bool` questions become TypeSafe's wire-level `noul` type.
fn wire_request(context: &ClassifierContext) -> Result<Value> {
    let mut questions = Map::new();
    for (id, question) in &context.questions {
        let mut wire = serde_json::to_value(question)?;
        if matches!(question, ClassifierQuestion::Bool(_))
            && let Some(object) = wire.as_object_mut()
        {
            object.insert("type".into(), "noul".into());
        }
        questions.insert(id.clone(), wire);
    }
    Ok(json!({ "state": context.state, "questions": questions }))
}

async fn run(
    transport: &SystemOneTransport,
    model: &ClassifierModel,
    context: &ClassifierContext,
    options: &ClassifierOptions,
    output: &mut ClassifierResult,
) -> Result<()> {
    if model.api != transport.api {
        return Err(Error::msg(format!("Unsupported classifier API: {}", model.api)));
    }
    if context.images.as_ref().is_some_and(|images| !images.is_empty()) {
        return Err(Error::msg(format!("{} does not support image input", transport.label)));
    }
    let url = (transport.url)(model)?;
    let payload = (transport.payload)(model, wire_request(context)?);
    let body = post_classifier_request(transport.label, &url, model, payload, options, &[]).await?;
    let result = (transport.output)(body)?;
    // Set before parsing answers: a request with malformed answers was still billed.
    if let Some(usage) = parse_classifier_usage(result.get("usage"), model) {
        output.usage = Some(usage);
    }
    output.answers = parse_answers(transport.label, result.get("answers"), context)?;
    Ok(())
}

/// Run one System One classification over `transport` (upstream `classifySystemOne`).
pub async fn classify_system_one(
    transport: &SystemOneTransport,
    model: &ClassifierModel,
    context: &ClassifierContext,
    options: ClassifierOptions,
) -> ClassifierResult {
    let mut output = ClassifierResult::empty(model);
    if let Err(error) = run(transport, model, context, &options, &mut output).await {
        let message = format_classifier_error(&error, &format!("{} error", transport.label));
        fail_result(&mut output, &options, message);
    }
    output
}
