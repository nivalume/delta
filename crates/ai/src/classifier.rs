//! Classification: structured JSON state (and optionally images) answered by typed
//! questions. Run classifier models with [`crate::Models::classify`].

use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::error::{ModelsError, ModelsErrorCode};
use crate::model::{ClassifierModel, InputModality};
use crate::options::RequestOptions;
use crate::types::{ImageContent, JsonObject, Usage, now_ms};

/// Pick one of several named options.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClassifierChoiceQuestion {
    pub instructions: String,
    /// Option name to its meaning, in order.
    pub criteria: IndexMap<String, String>,
}

/// Score on an ordered scale; `criteria` lists the levels from lowest to highest.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClassifierScoreQuestion {
    pub instructions: String,
    pub criteria: Vec<String>,
}

/// Meanings of the two answers of a bool question (JSON `{ "true": .., "false": .. }`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClassifierBoolCriteria {
    #[serde(rename = "true")]
    pub when_true: String,
    #[serde(rename = "false")]
    pub when_false: String,
}

/// Yes/no question.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClassifierBoolQuestion {
    pub instructions: String,
    pub criteria: ClassifierBoolCriteria,
}

/// One question of a classifier request, tagged by `type`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ClassifierQuestion {
    Choice(ClassifierChoiceQuestion),
    Score(ClassifierScoreQuestion),
    Bool(ClassifierBoolQuestion),
}

/// Input of a classifier request.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClassifierContext {
    pub state: JsonObject,
    /// Images judged together with `state`. Only models whose `input` includes `image`
    /// accept them; other models return an error result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<ImageContent>>,
    /// Questions by name, in request order.
    pub questions: IndexMap<String, ClassifierQuestion>,
}

/// Answer to a choice question.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClassifierChoiceAnswer {
    pub choice: String,
    pub probabilities: IndexMap<String, f64>,
    pub confidence: f64,
}

/// Answer to a score question.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClassifierScoreAnswer {
    pub score: f64,
    pub confidence: f64,
}

/// Answer to a bool question: the probability of `true`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClassifierBoolAnswer {
    pub probability: f64,
}

/// One answer, tagged by `type` like its question.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ClassifierAnswer {
    Choice(ClassifierChoiceAnswer),
    Score(ClassifierScoreAnswer),
    Bool(ClassifierBoolAnswer),
}

/// How a classifier request ended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ClassifierStopReason {
    #[default]
    Stop,
    Error,
    Aborted,
}

impl ClassifierStopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Error => "error",
            Self::Aborted => "aborted",
        }
    }
}

/// Final result of a classifier request.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassifierResult {
    pub api: String,
    pub provider: String,
    pub model: String,
    /// Answers by question name.
    pub answers: IndexMap<String, ClassifierAnswer>,
    /// Token usage and its cost at the model's catalog price, when the service reports
    /// token counts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    pub stop_reason: ClassifierStopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

impl ClassifierResult {
    /// An empty result for `model` with stop reason `stop`, stamped with the current time.
    pub fn empty(model: &ClassifierModel) -> Self {
        Self {
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            timestamp: now_ms(),
            ..Self::default()
        }
    }

    /// An error result (`aborted` when `aborted`), as upstream `classifierErrorResult`.
    pub fn error(model: &ClassifierModel, message: impl Into<String>, aborted: bool) -> Self {
        Self {
            stop_reason: if aborted {
                ClassifierStopReason::Aborted
            } else {
                ClassifierStopReason::Error
            },
            error_message: Some(message.into()),
            ..Self::empty(model)
        }
    }
}

/// Options of a classifier request.
#[derive(Clone, Debug, Default)]
pub struct ClassifierOptions {
    pub base: RequestOptions<ClassifierModel>,
    /// Divides the answer logits by this value before they are normalized into
    /// probabilities. Above 1 softens the distribution, below 1 sharpens it. Must be
    /// positive. APIs that cannot apply it ignore it.
    pub temperature: Option<f64>,
}

impl From<RequestOptions<ClassifierModel>> for ClassifierOptions {
    fn from(base: RequestOptions<ClassifierModel>) -> Self {
        Self {
            base,
            temperature: None,
        }
    }
}

/// The contract of a classifier API implementation (upstream `ProviderClassifier`).
///
/// An API module exposes a unit struct implementing it; providers register it under the API
/// id with [`crate::CreateProviderOptions::classifier`]. The future never fails: errors are
/// returned as a result with stop reason `error` or `aborted`. Auth is already resolved into
/// `options.base` when [`crate::Models`] calls it.
pub trait ProviderClassifier: Send + Sync {
    fn classify<'a>(
        &'a self,
        model: &'a ClassifierModel,
        context: &'a ClassifierContext,
        options: ClassifierOptions,
    ) -> BoxFuture<'a, ClassifierResult>;
}

/// Rejects classifier images for models whose catalog entry does not accept image input
/// (upstream `assertClassifierInputSupported`).
pub fn check_classifier_input(model: &ClassifierModel, context: &ClassifierContext) -> Result<(), ModelsError> {
    let has_images = context.images.as_ref().is_some_and(|images| !images.is_empty());
    if has_images && !model.input.contains(&InputModality::Image) {
        return Err(ModelsError::new(
            ModelsErrorCode::Provider,
            format!("Model {}/{} does not accept image input", model.provider, model.id),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn questions_and_answers_keep_upstream_json_and_order() {
        let context = json!({
            "state": { "text": "yes" },
            "questions": {
                "satisfaction": {
                    "type": "score",
                    "instructions": "Score user satisfaction.",
                    "criteria": ["dissatisfied", "neutral", "satisfied"]
                },
                "category": {
                    "type": "choice",
                    "instructions": "Classify the message.",
                    "criteria": { "zeta": "last letter", "alpha": "first letter" }
                },
                "approved": {
                    "type": "bool",
                    "instructions": "Does the user approve?",
                    "criteria": { "true": "Approval", "false": "No approval" }
                }
            }
        });
        let parsed: ClassifierContext = serde_json::from_value(context.clone()).unwrap();
        assert_eq!(
            parsed.questions.keys().collect::<Vec<_>>(),
            ["satisfaction", "category", "approved"]
        );
        let ClassifierQuestion::Choice(choice) = &parsed.questions["category"] else {
            panic!("expected a choice question");
        };
        assert_eq!(choice.criteria.keys().collect::<Vec<_>>(), ["zeta", "alpha"]);
        let ClassifierQuestion::Bool(approved) = &parsed.questions["approved"] else {
            panic!("expected a bool question");
        };
        assert_eq!(approved.criteria.when_true, "Approval");
        assert_eq!(serde_json::to_value(&parsed).unwrap(), context);

        let result = json!({
            "api": "test-classifier",
            "provider": "test",
            "model": "m",
            "answers": {
                "category": { "type": "choice", "choice": "alpha", "probabilities": { "zeta": 0.25, "alpha": 0.75 }, "confidence": 0.5 },
                "satisfaction": { "type": "score", "score": 2.0, "confidence": 0.8 },
                "approved": { "type": "bool", "probability": 0.9 }
            },
            "stopReason": "stop",
            "timestamp": 1
        });
        let parsed: ClassifierResult = serde_json::from_value(result.clone()).unwrap();
        assert_eq!(
            parsed.answers["approved"],
            ClassifierAnswer::Bool(ClassifierBoolAnswer { probability: 0.9 })
        );
        assert_eq!(serde_json::to_value(&parsed).unwrap(), result);
    }
}
