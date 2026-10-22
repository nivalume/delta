//! `models.*` for code mode scripts: the model catalog, and classifier and image model calls.
//! Port of upstream `codemode/models.ts`.
//!
//! The catalog functions are host globals, journaled like any global. `models.classify()` and
//! `models.generateImages()` are paid effects, so each runs as an internal tool (`callers`
//! tools only, a reserved name, hidden from scripts' tool listings): a nested durable call with
//! its own task, live row, abort, usage accounting and replay policy. The code mode tool turns a
//! script's call of the global into a call of that tool.

use std::sync::Arc;

use delta_ai::classifier::{ClassifierContext, ClassifierResult, ClassifierStopReason};
use delta_ai::images::{AssistantImages, ImagesContext, ImagesStopReason};
use delta_ai::model::{AnyModel, ModelType};
use delta_ai::types::{Tool, Usage};
use delta_chord::context::Context;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::model::ModelClient;
use crate::tool::{Callers, FnHarnessTool, HarnessTool, ToolError, ToolExecutionResult};

/// The internal tool behind `models.classify()`.
pub const CLASSIFY_TOOL: &str = "delta.codemode.models.classify";
/// The internal tool behind `models.generateImages()`.
pub const GENERATE_IMAGES_TOOL: &str = "delta.codemode.models.generateImages";

/// The globals a script calls, with their internal tools.
pub(super) const MODEL_CALLS: [(&str, &str); 2] = [
    ("models.classify", CLASSIFY_TOOL),
    ("models.generateImages", GENERATE_IMAGES_TOOL),
];

/// The catalog functions, host globals of the `models` namespace.
pub(super) const CATALOG_GLOBALS: [&str; 3] = [
    "models.getModelsOfType",
    "models.getAvailableOfType",
    "models.getModelOfType",
];

/// `models.classify()` and `models.generateImages()` calls one script may have in flight; the
/// rest wait.
pub const MAX_CONCURRENT_MODEL_CALLS: usize = 4;

/// What the tool description says about `models`.
pub(super) fn models_description() -> String {
    [
        "`models` (all functions return promises):",
        "- `models.getModelsOfType(type, provider?)`, `models.getAvailableOfType(type, provider?)` (configured auth only), `models.getModelOfType(type, provider, id)`: the catalog; `type` is \"chat\", \"image\", or \"classifier\".",
        "- `models.classify(model, { state: { ... }, images?: [{ type: \"image\", data, mimeType }], questions: { <id>: { type: \"choice\", instructions, criteria: { <label>: <meaning> } } | { type: \"score\", instructions, criteria: [<lowest level>, ..., <highest level>] } | { type: \"bool\", instructions, criteria: { true: <meaning>, false: <meaning> } } } })` resolves to `{ answers: { <id>: ... }, stopReason, errorMessage?, usage? }`.",
        "- `models.generateImages(model, { input: [{ type: \"text\", text: <prompt> }, ...optional { type: \"image\", data, mimeType }] })` resolves to `{ output: [image blocks], stopReason, errorMessage?, usage? }`; show each image with `image(block)`.",
        &format!("At most {MAX_CONCURRENT_MODEL_CALLS} classify/generateImages calls run at once; more wait."),
    ]
    .join("\n")
}

/// The internal tool a script's call of global `name` becomes.
pub(super) fn model_call(name: &str) -> Option<&'static str> {
    MODEL_CALLS
        .iter()
        .find(|(global, _)| *global == name)
        .map(|(_, tool)| *tool)
}

const MODEL_TYPES: [ModelType; 3] = [ModelType::Chat, ModelType::Image, ModelType::Classifier];

fn type_name(model_type: ModelType) -> &'static str {
    match model_type {
        ModelType::Chat => "chat",
        ModelType::Image => "image",
        ModelType::Classifier => "classifier",
    }
}

/// A script value in an error message: `undefined`, `a string`, `an array`, or its keys
/// (`{ prompt }`).
fn describe_value(value: Option<&Value>) -> String {
    match value {
        None => "undefined".to_owned(),
        Some(Value::Null) => "null".to_owned(),
        Some(Value::Array(items)) if items.is_empty() => "an empty array".to_owned(),
        Some(Value::Array(_)) => "an array".to_owned(),
        Some(Value::Object(object)) if object.is_empty() => "{}".to_owned(),
        Some(Value::Object(object)) => {
            let keys: Vec<&str> = object.keys().take(6).map(String::as_str).collect();
            let more = if object.len() > 6 { ", ..." } else { "" };
            format!("{{ {}{more} }}", keys.join(", "))
        }
        Some(Value::String(_)) => "a string".to_owned(),
        Some(Value::Number(_)) => "a number".to_owned(),
        Some(Value::Bool(_)) => "a boolean".to_owned(),
    }
}

/// `an image`, `a classifier`.
fn with_article(word: &str) -> String {
    let article = if word.starts_with(['a', 'e', 'i', 'o', 'u']) {
        "an"
    } else {
        "a"
    };
    format!("{article} {word}")
}

fn to_model_type(value: Option<&Value>) -> Result<ModelType, String> {
    let name = value.and_then(Value::as_str);
    MODEL_TYPES
        .into_iter()
        .find(|model_type| Some(type_name(*model_type)) == name)
        .ok_or_else(|| {
            let shown = value.map_or_else(|| "undefined".to_owned(), Value::to_string);
            format!("Unknown model type {shown}. Use \"chat\", \"image\", or \"classifier\".")
        })
}

fn to_provider(value: Option<&Value>) -> Result<Option<&str>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(provider)) => Ok(Some(provider)),
        Some(_) => Err("provider must be a string".to_owned()),
    }
}

/// A catalog entry for scripts. `headers` is dropped because configured headers can carry
/// credentials.
fn to_model_info(model: &AnyModel) -> Value {
    let mut info = serde_json::to_value(model).unwrap_or(Value::Null);
    if let Value::Object(object) = &mut info {
        object.remove("headers");
    }
    info
}

/// The catalog functions: what a script's call of `name` with `args` (the argument list)
/// resolves to.
pub(super) async fn catalog_call(
    name: &str,
    args: Option<&Value>,
    models: &dyn ModelClient,
    ctx: &Context,
) -> Result<Option<Value>, String> {
    let list: &[Value] = args.and_then(Value::as_array).map_or(&[], Vec::as_slice);
    let (model_type, provider, id) = (list.first(), list.get(1), list.get(2));
    let infos = |models: Vec<AnyModel>| Some(Value::Array(models.iter().map(to_model_info).collect()));
    match name {
        "models.getModelsOfType" => Ok(infos(
            models.models_of_type(to_model_type(model_type)?, to_provider(provider)?),
        )),
        "models.getAvailableOfType" => {
            let available = models
                .available_of_type(to_model_type(model_type)?, to_provider(provider)?, ctx)
                .await
                .map_err(|error| error.to_string())?;
            Ok(infos(available))
        }
        _ => {
            let (Some(Value::String(provider)), Some(Value::String(id))) = (provider, id) else {
                let shown: Vec<String> = list.iter().map(|value| describe_value(Some(value))).collect();
                return Err(format!(
                    "models.getModelOfType(type, provider, id) expects three strings, got ({}). The provider and the id are separate arguments, for example models.getModelOfType(\"classifier\", \"typesafe\", \"jev-latest\").",
                    shown.join(", ")
                ));
            };
            let model = models.model_of_type(to_model_type(model_type)?, provider, id);
            Ok(model.as_ref().map(to_model_info))
        }
    }
}

const CLASSIFIER_CONTEXT_SHAPE: &str = "{ state: { ... }, images?: [{ type: \"image\", data: <base64>, mimeType }], questions: { <id>: { type: \"choice\", instructions, criteria: { <label>: <meaning> } } | { type: \"score\", instructions, criteria: [<lowest level>, ..., <highest level>] } | { type: \"bool\", instructions, criteria: { true: <meaning>, false: <meaning> } } } }";

fn is_image_block(value: &Value) -> bool {
    value.get("type").and_then(Value::as_str) == Some("image")
        && value.get("data").is_some_and(Value::is_string)
        && value.get("mimeType").is_some_and(Value::is_string)
}

/// Checks a script's classifier context, so mistakes fail with the expected shape instead of a
/// provider error.
fn check_classifier_context(context: Option<&Value>) -> Result<ClassifierContext, String> {
    let fail = |problem: String| format!("models.classify() {problem}. Expected context: {CLASSIFIER_CONTEXT_SHAPE}.");
    let Some(Value::Object(object)) = context else {
        return Err(fail(format!(
            "expects a context object as its second argument, got {}",
            describe_value(context)
        )));
    };
    let state = object.get("state");
    if !state.is_some_and(Value::is_object) {
        return Err(fail(format!(
            "context.state must be an object, got {}",
            describe_value(state)
        )));
    }
    if let Some(images) = object.get("images") {
        let Value::Array(images) = images else {
            return Err(fail(format!(
                "context.images must be an array, got {}",
                describe_value(Some(images))
            )));
        };
        if let Some((index, image)) = images.iter().enumerate().find(|(_, image)| !is_image_block(image)) {
            return Err(fail(format!(
                "context.images[{index}] must be an image block, got {}",
                describe_value(Some(image))
            )));
        }
    }
    let questions = object.get("questions");
    let Some(Value::Object(questions)) =
        questions.filter(|questions| questions.as_object().is_some_and(|q| !q.is_empty()))
    else {
        return Err(fail(format!(
            "context.questions must map question IDs to questions, got {}",
            describe_value(questions)
        )));
    };
    let strings = |values: &mut dyn Iterator<Item = &Value>| {
        let mut any = false;
        for value in values {
            if !value.is_string() {
                return false;
            }
            any = true;
        }
        any
    };
    for (id, question) in questions {
        let at = format!("context.questions.{id}");
        let Value::Object(fields) = question else {
            return Err(fail(format!(
                "{at} must be a question object, got {}",
                describe_value(Some(question))
            )));
        };
        if !fields.get("instructions").is_some_and(Value::is_string) {
            return Err(fail(format!("{at}.instructions must be a string")));
        }
        let criteria = fields.get("criteria");
        match fields.get("type").and_then(Value::as_str) {
            Some("choice") => {
                if !criteria
                    .and_then(Value::as_object)
                    .is_some_and(|map| strings(&mut map.values()))
                {
                    return Err(fail(format!(
                        "{at} is a \"choice\" question, so criteria must map each label to its meaning"
                    )));
                }
            }
            Some("score") => {
                if !criteria
                    .and_then(Value::as_array)
                    .is_some_and(|levels| strings(&mut levels.iter()))
                {
                    return Err(fail(format!(
                        "{at} is a \"score\" question, so criteria must list the levels as strings, lowest first"
                    )));
                }
            }
            Some("bool") => {
                let both = criteria.is_some_and(|criteria| {
                    criteria.get("true").is_some_and(Value::is_string)
                        && criteria.get("false").is_some_and(Value::is_string)
                });
                if !both {
                    return Err(fail(format!(
                        "{at} is a \"bool\" question, so criteria must be {{ true: string, false: string }}"
                    )));
                }
            }
            _ => {
                let shown = fields
                    .get("type")
                    .map_or_else(|| "undefined".to_owned(), Value::to_string);
                return Err(fail(format!(
                    "{at}.type must be \"choice\", \"score\", or \"bool\", got {shown}"
                )));
            }
        }
    }
    serde_json::from_value(Value::Object(object.clone())).map_err(|error| fail(error.to_string()))
}

/// Checks a script's image context, so mistakes such as `{ prompt }` fail with the expected
/// shape.
fn check_images_context(context: Option<&Value>) -> Result<ImagesContext, String> {
    let fail = |problem: String| {
        format!(
            "models.generateImages() {problem}. Expected context: {{ input: [{{ type: \"text\", text: <prompt> }}, ...optional {{ type: \"image\", data: <base64>, mimeType }} references] }}."
        )
    };
    let Some(Value::Object(object)) = context else {
        return Err(fail(format!(
            "expects a context object as its second argument, got {}",
            describe_value(context)
        )));
    };
    let input = object.get("input");
    let Some(Value::Array(blocks)) = input.filter(|input| input.as_array().is_some_and(|blocks| !blocks.is_empty()))
    else {
        return Err(fail(format!(
            "context.input must be a non-empty array of blocks, got {}",
            describe_value(input)
        )));
    };
    for (index, block) in blocks.iter().enumerate() {
        let text = block.get("type").and_then(Value::as_str) == Some("text")
            && block.get("text").is_some_and(Value::is_string);
        if !text && !is_image_block(block) {
            return Err(fail(format!(
                "context.input[{index}] must be a text or image block, got {}",
                describe_value(Some(block))
            )));
        }
    }
    serde_json::from_value(Value::Object(object.clone())).map_err(|error| fail(error.to_string()))
}

/// Resolves the script's model by provider and id only, so a script-supplied `baseUrl` or
/// `headers` never receive the credentials, with errors that say what to do.
fn resolve_model(
    name: &str,
    model_type: ModelType,
    model: Option<&Value>,
    models: &dyn ModelClient,
) -> Result<AnyModel, String> {
    let wanted = type_name(model_type);
    let list_hint = format!("List the {wanted} models you can use with models.getAvailableOfType(\"{wanted}\").");
    let provider = model.and_then(|model| model.get("provider")).and_then(Value::as_str);
    let id = model.and_then(|model| model.get("id")).and_then(Value::as_str);
    let (Some(provider), Some(id), Some(Value::Object(_))) = (provider, id, model) else {
        let undefined_hint = if matches!(model, None | Some(Value::Null)) {
            " models.getModelOfType() returns undefined for an unknown provider or id."
        } else {
            ""
        };
        return Err(format!(
            "{name}() expects {} model as its first argument, got {}.{undefined_hint} {list_hint}",
            with_article(wanted),
            describe_value(model)
        ));
    };
    if let Some(resolved) = models.model_of_type(model_type, provider, id) {
        return Ok(resolved);
    }
    let actual = MODEL_TYPES
        .into_iter()
        .find(|other| *other != model_type && models.model_of_type(*other, provider, id).is_some());
    Err(match actual {
        Some(actual) => format!(
            "\"{provider}/{id}\" is {} model, not {} model. {list_hint}",
            with_article(type_name(actual)),
            with_article(wanted)
        ),
        None => format!("Unknown {wanted} model \"{provider}/{id}\". {list_hint}"),
    })
}

/// The result of a model call as the tool returns it: the whole result for the script, usage
/// counted by the Harness.
fn model_result(
    label: &str,
    ok: bool,
    stop_reason: &str,
    error_message: Option<&str>,
    usage: Option<&Usage>,
    value: &impl Serialize,
) -> Result<ToolExecutionResult, ToolError> {
    let text = if ok {
        format!("{label} done")
    } else {
        format!("{label} {stop_reason}: {}", error_message.unwrap_or_default())
    };
    Ok(ToolExecutionResult {
        structured_output: Some(serde_json::to_value(value)?),
        is_error: !ok,
        usage: usage.cloned(),
        ..ToolExecutionResult::text(text)
    })
}

fn model_tool_definition(name: &str, description: &str) -> Tool {
    Tool::new(
        name,
        description,
        serde_json::json!({
            "type": "object",
            "properties": { "model": {}, "context": {} },
            "required": ["model", "context"],
        }),
    )
}

/// The internal tools behind `models.classify()` and `models.generateImages()`.
pub(super) fn model_tools() -> [Arc<dyn HarnessTool>; 2] {
    let classify = FnHarnessTool::new(
        model_tool_definition(CLASSIFY_TOOL, "models.classify() of code mode scripts."),
        |arguments, api, ctx| {
            Box::pin(async move {
                let models = api.models()?;
                let model = resolve_model(
                    "models.classify",
                    ModelType::Classifier,
                    arguments.get("model"),
                    &*models,
                )?;
                let AnyModel::Classifier(model) = model else {
                    return Err("models.classify() resolved a model of another type".into());
                };
                let context = check_classifier_context(arguments.get("context"))?;
                let result: ClassifierResult = models.classify(&model, &context, ctx).await;
                let label = format!("classify {}/{}", model.provider, model.id);
                model_result(
                    &label,
                    result.stop_reason == ClassifierStopReason::Stop,
                    result.stop_reason.as_str(),
                    result.error_message.as_deref(),
                    result.usage.as_ref(),
                    &result,
                )
            })
        },
    )
    .with_structured_output_schema(serde_json::json!({}))
    .with_callers(Callers::TOOLS);
    let generate_images = FnHarnessTool::new(
        model_tool_definition(GENERATE_IMAGES_TOOL, "models.generateImages() of code mode scripts."),
        |arguments, api, ctx| {
            Box::pin(async move {
                let models = api.models()?;
                let model = resolve_model(
                    "models.generateImages",
                    ModelType::Image,
                    arguments.get("model"),
                    &*models,
                )?;
                let AnyModel::Image(model) = model else {
                    return Err("models.generateImages() resolved a model of another type".into());
                };
                let context = check_images_context(arguments.get("context"))?;
                let result: AssistantImages = models.generate_images(&model, &context, ctx).await;
                let label = format!("generateImages {}/{}", model.provider, model.id);
                let stop_reason = match result.stop_reason {
                    ImagesStopReason::Stop => "stop",
                    ImagesStopReason::Error => "error",
                    ImagesStopReason::Aborted => "aborted",
                };
                model_result(
                    &label,
                    result.stop_reason == ImagesStopReason::Stop,
                    stop_reason,
                    result.error_message.as_deref(),
                    result.usage.as_ref(),
                    &result,
                )
            })
        },
    )
    .with_structured_output_schema(serde_json::json!({}))
    .with_callers(Callers::TOOLS);
    [Arc::new(classify), Arc::new(generate_images)]
}

/// Image blocks in a `models.generateImages()` result.
pub(super) fn count_images(value: Option<&Value>) -> usize {
    value
        .and_then(|value| value.get("output"))
        .and_then(Value::as_array)
        .map_or(0, |output| {
            output
                .iter()
                .filter(|item| item.get("type").and_then(Value::as_str) == Some("image"))
                .count()
        })
}

/// A script's call of `models.classify()` or `models.generateImages()` as the arguments of its
/// internal tool.
pub(super) fn model_call_arguments(args: Option<&Value>) -> Value {
    let list: &[Value] = args.and_then(Value::as_array).map_or(&[], Vec::as_slice);
    let mut arguments = Map::new();
    arguments.insert("model".to_owned(), list.first().cloned().unwrap_or(Value::Null));
    arguments.insert("context".to_owned(), list.get(1).cloned().unwrap_or(Value::Null));
    Value::Object(arguments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_are_described_without_their_content() {
        assert_eq!(describe_value(None), "undefined");
        assert_eq!(
            describe_value(Some(&serde_json::json!({ "prompt": "x" }))),
            "{ prompt }"
        );
        assert_eq!(describe_value(Some(&serde_json::json!("x"))), "a string");
        assert_eq!(with_article("image"), "an image");
        assert_eq!(with_article("classifier"), "a classifier");
    }

    #[test]
    fn contexts_are_checked_against_their_shape() {
        let error = check_images_context(Some(&serde_json::json!({ "prompt": "a cat" }))).unwrap_err();
        assert!(
            error.starts_with(
                "models.generateImages() context.input must be a non-empty array of blocks, got undefined."
            ),
            "{error}"
        );
        let context = check_images_context(Some(
            &serde_json::json!({ "input": [{ "type": "text", "text": "a cat" }] }),
        ));
        assert_eq!(context.unwrap().input.len(), 1);
        let question = serde_json::json!({ "type": "bool", "instructions": "Is it?", "criteria": { "true": "yes" } });
        let error = check_classifier_context(Some(
            &serde_json::json!({ "state": {}, "questions": { "q": question } }),
        ))
        .unwrap_err();
        assert!(
            error.starts_with(
                "models.classify() context.questions.q is a \"bool\" question, so criteria must be { true: string, false: string }."
            ),
            "{error}"
        );
        let question = serde_json::json!({ "type": "choice", "instructions": "Which?", "criteria": { "a": "first", "b": "second" } });
        let context = check_classifier_context(Some(
            &serde_json::json!({ "state": { "x": 1 }, "questions": { "q": question } }),
        ));
        assert_eq!(context.unwrap().questions.len(), 1);
    }

    #[test]
    fn model_calls_become_tool_arguments() {
        assert_eq!(
            model_call_arguments(Some(&serde_json::json!([{ "provider": "p", "id": "m" }]))),
            serde_json::json!({ "model": { "provider": "p", "id": "m" }, "context": null })
        );
        assert_eq!(model_call("models.classify"), Some(CLASSIFY_TOOL));
        assert_eq!(model_call("models.getModelsOfType"), None);
        assert_eq!(
            count_images(Some(
                &serde_json::json!({ "output": [{ "type": "image" }, { "type": "text" }] })
            )),
            1
        );
    }
}
