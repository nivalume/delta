//! Models: upstream's model config and classes on delta-ai.
//!
//! Upstream reaches models through litellm; here the name resolves to a delta-ai model
//! (`anthropic/claude-sonnet-4-5` is provider `anthropic`, model `claude-sonnet-4-5`) and the
//! harness streams it through [`MiniModelClient`], which adds the sampling options
//! `model_kwargs` sets. The deterministic class replays scripted outputs through delta-ai's
//! faux provider.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use delta_ai::model::{Model, ThinkingLevel};
use delta_ai::options::{SimpleStreamOptions, StreamOptions, Transport};
use delta_ai::providers::builtin::builtin_models;
use delta_ai::providers::faux::{
    FauxModelDefinition, FauxOptions, FauxProvider, FauxResponse, faux_assistant_message, faux_text,
};
use delta_ai::types::{AssistantBlock, AssistantMessage, DeferredHandle, JsonObject, StopReason, ToolCall};
use delta_ai::{AssistantMessageEventStream, CreateModelsOptions, Models};
use delta_chord::context::Context;
use delta_harness::model::{ModelClient, ModelError, ModelRef, ModelRequest, ModelsClient};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

/// The default observation template of upstream's model classes.
pub const DEFAULT_OBSERVATION_TEMPLATE: &str = "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>";

/// The default format error template of the tool-call classes.
pub const DEFAULT_TOOLCALL_FORMAT_ERROR_TEMPLATE: &str = "{{ error }}";

/// The default format error template of the text-based classes.
pub const DEFAULT_TEXTBASED_FORMAT_ERROR_TEMPLATE: &str =
    "Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions.";

/// The default `action_regex` of the text-based classes.
pub const DEFAULT_ACTION_REGEX: &str = r"```mswea_bash_command\s*\n(.*?)\n```";

/// How the model states its actions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionFormat {
    /// Calls of the `bash` tool.
    Toolcall,
    /// One fenced block matched by `action_regex`.
    Textbased,
}

/// A model class: upstream's `model_class` shortcuts and class paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelClass {
    pub format: ActionFormat,
    /// Scripted outputs instead of a provider.
    pub deterministic: bool,
    /// Every name is an OpenRouter model id.
    pub openrouter: bool,
    /// Upstream's class path, for `info.config.model_type`.
    pub path: &'static str,
}

impl ModelClass {
    /// The class `model_class` names; empty: the default (litellm, tool calls).
    pub fn parse(model_class: &str) -> Result<Self, ModelConfigError> {
        use ActionFormat::{Textbased, Toolcall};
        let class = |format, deterministic, openrouter, path| Self {
            format,
            deterministic,
            openrouter,
            path,
        };
        Ok(match model_class {
            "" | "litellm" | "minisweagent.models.litellm_model.LitellmModel" => {
                class(Toolcall, false, false, "minisweagent.models.litellm_model.LitellmModel")
            }
            "litellm_response" | "minisweagent.models.litellm_response_model.LitellmResponseModel" => class(
                Toolcall,
                false,
                false,
                "minisweagent.models.litellm_response_model.LitellmResponseModel",
            ),
            "litellm_textbased" | "minisweagent.models.litellm_textbased_model.LitellmTextbasedModel" => class(
                Textbased,
                false,
                false,
                "minisweagent.models.litellm_textbased_model.LitellmTextbasedModel",
            ),
            "openrouter" | "minisweagent.models.openrouter_model.OpenRouterModel" => class(
                Toolcall,
                false,
                true,
                "minisweagent.models.openrouter_model.OpenRouterModel",
            ),
            "openrouter_response" | "minisweagent.models.openrouter_response_model.OpenRouterResponseModel" => class(
                Toolcall,
                false,
                true,
                "minisweagent.models.openrouter_response_model.OpenRouterResponseModel",
            ),
            "openrouter_textbased" | "minisweagent.models.openrouter_textbased_model.OpenRouterTextbasedModel" => {
                class(
                    Textbased,
                    false,
                    true,
                    "minisweagent.models.openrouter_textbased_model.OpenRouterTextbasedModel",
                )
            }
            "portkey" | "minisweagent.models.portkey_model.PortkeyModel" => {
                class(Toolcall, false, false, "minisweagent.models.portkey_model.PortkeyModel")
            }
            "requesty" | "minisweagent.models.requesty_model.RequestyModel" => class(
                Toolcall,
                false,
                false,
                "minisweagent.models.requesty_model.RequestyModel",
            ),
            "deterministic" | "minisweagent.models.test_models.DeterministicModel" => class(
                Textbased,
                true,
                false,
                "minisweagent.models.test_models.DeterministicModel",
            ),
            "deterministic_toolcall" | "minisweagent.models.test_models.DeterministicToolcallModel" => class(
                Toolcall,
                true,
                false,
                "minisweagent.models.test_models.DeterministicToolcallModel",
            ),
            other => return Err(ModelConfigError::UnknownClass(other.to_owned())),
        })
    }
}

/// Why a model could not be set up.
#[derive(Debug, thiserror::Error)]
pub enum ModelConfigError {
    #[error(
        "Unknown model class: {0} (available: litellm, litellm_textbased, litellm_response, openrouter, openrouter_textbased, openrouter_response, portkey, requesty, deterministic, deterministic_toolcall)"
    )]
    UnknownClass(String),
    #[error("No default model set. Please run `mini-extra config setup` to set one.")]
    NoModel,
    #[error("Unknown model {0}: no provider of delta-ai serves it")]
    UnknownModel(String),
    #[error("Invalid model config: {0}")]
    Invalid(#[from] serde_json::Error),
    #[error("Invalid model_kwargs.{key}: {reason}")]
    Kwarg { key: String, reason: String },
}

/// The `model` section of a config.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub model_name: String,
    #[serde(default, skip_serializing)]
    pub model_class: String,
    /// Options for the API: `temperature`, `max_tokens` / `max_completion_tokens`,
    /// `reasoning_effort`, and sampling options passed through to OpenAI-compatible APIs.
    #[serde(default)]
    pub model_kwargs: Map<String, Json>,
    #[serde(default)]
    pub litellm_model_registry: Option<String>,
    /// Accepted for compatibility; delta-ai places cache markers itself.
    #[serde(default)]
    pub set_cache_control: Option<String>,
    #[serde(default = "default_cost_tracking")]
    pub cost_tracking: String,
    #[serde(default)]
    pub format_error_template: Option<String>,
    #[serde(default = "default_observation_template")]
    pub observation_template: String,
    /// Accepted for compatibility; multimodal content is not extracted.
    #[serde(default)]
    pub multimodal_regex: String,
    #[serde(default)]
    pub action_regex: Option<String>,
    /// The deterministic class's scripted outputs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<Json>,
    #[serde(default = "default_cost_per_call")]
    pub cost_per_call: f64,
}

fn default_cost_tracking() -> String {
    crate::global::env_var("MSWEA_COST_TRACKING").unwrap_or_else(|| "default".to_owned())
}

fn default_observation_template() -> String {
    DEFAULT_OBSERVATION_TEMPLATE.to_owned()
}

fn default_cost_per_call() -> f64 {
    1.0
}

/// A model config resolved for a run: class, name, templates and options.
#[derive(Clone, Debug)]
pub struct ModelSettings {
    pub config: ModelConfig,
    pub class: ModelClass,
    pub format_error_template: String,
    pub action_regex: regex::Regex,
    pub kwargs: StreamKwargs,
}

impl ModelSettings {
    /// Resolves `config` (the `model` section) with the model name from the command line,
    /// the config or `MSWEA_MODEL_NAME`, in that order.
    pub fn new(config: &Json, cli_model_name: Option<&str>) -> Result<Self, ModelConfigError> {
        let mut config: ModelConfig = serde_json::from_value(if config.is_object() {
            config.clone()
        } else {
            Json::Object(Map::new())
        })?;
        let class = ModelClass::parse(&config.model_class)?;
        let given = cli_model_name
            .filter(|name| !name.is_empty())
            .or(Some(config.model_name.as_str()).filter(|name| !name.is_empty()))
            .map(str::to_owned);
        config.model_name = match given {
            Some(name) => name,
            None if class.deterministic => "deterministic".to_owned(),
            None => crate::global::env_var("MSWEA_MODEL_NAME").ok_or(ModelConfigError::NoModel)?,
        };
        let lower = config.model_name.to_lowercase();
        if config.set_cache_control.is_none()
            && ["anthropic", "sonnet", "opus", "claude"]
                .iter()
                .any(|name| lower.contains(name))
        {
            config.set_cache_control = Some("default_end".to_owned());
        }
        let format_error_template = config.format_error_template.clone().unwrap_or_else(|| {
            match class.format {
                ActionFormat::Toolcall => DEFAULT_TOOLCALL_FORMAT_ERROR_TEMPLATE,
                ActionFormat::Textbased => DEFAULT_TEXTBASED_FORMAT_ERROR_TEMPLATE,
            }
            .to_owned()
        });
        let pattern = config.action_regex.as_deref().unwrap_or(DEFAULT_ACTION_REGEX);
        // Python's `re.DOTALL`.
        let action_regex = regex::RegexBuilder::new(pattern)
            .dot_matches_new_line(true)
            .build()
            .map_err(|error| ModelConfigError::Kwarg {
                key: "action_regex".to_owned(),
                reason: error.to_string(),
            })?;
        let kwargs = StreamKwargs::parse(&config.model_kwargs)?;
        Ok(Self {
            config,
            class,
            format_error_template,
            action_regex,
            kwargs,
        })
    }

    /// The config as upstream's `model_dump` prints it, for templates and the trajectory.
    pub fn dump(&self) -> Json {
        let mut dump = serde_json::to_value(&self.config).unwrap_or(Json::Null);
        if let Some(object) = dump.as_object_mut() {
            object.insert(
                "format_error_template".to_owned(),
                Json::String(self.format_error_template.clone()),
            );
            if self.class.format == ActionFormat::Toolcall {
                object.remove("action_regex");
            } else {
                object.insert(
                    "action_regex".to_owned(),
                    Json::String(self.action_regex.as_str().to_owned()),
                );
            }
            if !self.class.deterministic {
                object.remove("cost_per_call");
            }
        }
        dump
    }

    /// `{"info": {"config": {"model": ..., "model_type": ...}}}`.
    pub fn serialize(&self) -> Json {
        serde_json::json!({
            "info": {
                "config": {
                    "model": self.dump(),
                    "model_type": self.class.path,
                }
            }
        })
    }

    /// The cost of the `index`th call of a deterministic model: its output's `extra.cost`,
    /// else `cost_per_call`.
    pub fn deterministic_cost(&self, index: usize) -> f64 {
        self.config
            .outputs
            .get(index)
            .and_then(|output| output.pointer("/extra/cost"))
            .and_then(Json::as_f64)
            .unwrap_or(self.config.cost_per_call)
    }
}

/// What `model_kwargs` sets on each request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StreamKwargs {
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub reasoning: Option<ThinkingLevel>,
    /// Passed into the request body of OpenAI-compatible APIs.
    pub sampling_params: Option<JsonObject>,
    /// Keys nothing here understands, reported once.
    pub ignored: Vec<String>,
}

/// Sampling options merged into OpenAI-compatible request bodies as they are.
const PASSED_THROUGH: [&str; 7] = [
    "top_p",
    "top_k",
    "seed",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "min_p",
];

impl StreamKwargs {
    pub fn parse(kwargs: &Map<String, Json>) -> Result<Self, ModelConfigError> {
        let invalid = |key: &str, reason: &str| ModelConfigError::Kwarg {
            key: key.to_owned(),
            reason: reason.to_owned(),
        };
        let mut parsed = Self::default();
        let mut sampling = JsonObject::new();
        for (key, value) in kwargs {
            match key.as_str() {
                "temperature" => {
                    parsed.temperature = Some(value.as_f64().ok_or_else(|| invalid(key, "not a number"))?);
                }
                "max_tokens" | "max_completion_tokens" | "max_output_tokens" => {
                    parsed.max_tokens = Some(value.as_u64().ok_or_else(|| invalid(key, "not a whole number"))?);
                }
                "reasoning_effort" => {
                    let expected = "expected none, minimal, low, medium, high, xhigh or max";
                    parsed.reasoning = match value {
                        Json::Null => None,
                        Json::String(level) => match level.as_str() {
                            "none" => None,
                            "minimal" => Some(ThinkingLevel::Minimal),
                            "low" => Some(ThinkingLevel::Low),
                            "medium" => Some(ThinkingLevel::Medium),
                            "high" => Some(ThinkingLevel::High),
                            "xhigh" => Some(ThinkingLevel::Xhigh),
                            "max" => Some(ThinkingLevel::Max),
                            _ => return Err(invalid(key, expected)),
                        },
                        _ => return Err(invalid(key, expected)),
                    };
                }
                // litellm's own switches.
                "drop_params" | "num_retries" => {}
                key if PASSED_THROUGH.contains(&key) => {
                    sampling.insert(key.to_owned(), value.clone());
                }
                _ => parsed.ignored.push(key.clone()),
            }
        }
        parsed.sampling_params = (!sampling.is_empty()).then_some(sampling);
        Ok(parsed)
    }
}

/// litellm provider prefixes whose delta-ai provider has another id.
const PROVIDER_ALIASES: [(&str, &str); 9] = [
    ("gemini", "google"),
    ("vertex_ai", "google-vertex"),
    ("vertex_ai_beta", "google-vertex"),
    ("bedrock", "amazon-bedrock"),
    ("bedrock_converse", "amazon-bedrock"),
    ("fireworks_ai", "fireworks"),
    ("moonshot", "moonshotai"),
    ("azure_ai", "azure"),
    ("together_ai", "together"),
];

/// The delta-ai model `name` names for `class`.
///
/// - OpenRouter classes: the name (without an `openrouter/` prefix) is an OpenRouter id.
/// - `provider/id`: the provider (litellm prefixes mapped) and the id.
/// - A bare id: the first provider whose catalog has it.
///
/// An id a known provider's catalog lacks gets that provider's API and endpoint, with no
/// price (litellm passes unknown ids through as well).
pub fn resolve_model(models: &Models, name: &str, class: ModelClass) -> Result<Model, ModelConfigError> {
    let (provider, id) = if class.openrouter {
        (
            "openrouter".to_owned(),
            name.strip_prefix("openrouter/").unwrap_or(name),
        )
    } else if let Some((prefix, id)) = name.split_once('/') {
        let provider = PROVIDER_ALIASES
            .iter()
            .find(|(alias, _)| *alias == prefix)
            .map_or(prefix, |(_, provider)| provider);
        if models.provider(provider).is_some() {
            (provider.to_owned(), id)
        } else {
            return find_bare(models, name);
        }
    } else {
        return find_bare(models, name);
    };
    if let Some(model) = models.get_model(&provider, id) {
        return Ok(model);
    }
    let template = models
        .get_models(Some(&provider))
        .into_iter()
        .next()
        .ok_or_else(|| ModelConfigError::UnknownModel(name.to_owned()))?;
    Ok(Model {
        id: id.to_owned(),
        name: id.to_owned(),
        cost: delta_ai::model::ModelCost::default(),
        ..template
    })
}

fn find_bare(models: &Models, name: &str) -> Result<Model, ModelConfigError> {
    models
        .get_models(None)
        .into_iter()
        .find(|model| model.id == name)
        .ok_or_else(|| ModelConfigError::UnknownModel(name.to_owned()))
}

/// A deterministic class's scripted output as a delta-ai answer: its `content` as text and its
/// OpenAI-format `tool_calls` as tool calls.
pub fn scripted_answer(output: &Json) -> AssistantMessage {
    let mut blocks = Vec::new();
    if let Some(text) = output
        .get("content")
        .and_then(Json::as_str)
        .filter(|text| !text.is_empty())
    {
        blocks.push(faux_text(text));
    }
    for call in output.get("tool_calls").and_then(Json::as_array).into_iter().flatten() {
        let function = call.get("function").unwrap_or(&Json::Null);
        let arguments_text = function.get("arguments").and_then(Json::as_str).unwrap_or("{}");
        let (arguments, arguments_error) = match serde_json::from_str::<Json>(arguments_text) {
            Ok(Json::Object(arguments)) => (arguments, None),
            Ok(_) => (
                JsonObject::new(),
                Some("Tool call arguments are not a JSON object".to_owned()),
            ),
            Err(error) => (JsonObject::new(), Some(error.to_string())),
        };
        blocks.push(AssistantBlock::ToolCall(ToolCall {
            id: call
                .get("id")
                .and_then(Json::as_str)
                .map_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple()), str::to_owned),
            name: function
                .get("name")
                .and_then(Json::as_str)
                .unwrap_or_default()
                .to_owned(),
            arguments,
            arguments_error,
            ..ToolCall::default()
        }));
    }
    faux_assistant_message(blocks)
}

/// The provider id of deterministic models.
pub const DETERMINISTIC_PROVIDER: &str = "deterministic";

/// The harness's model access for one run: the resolved model, streamed with the
/// `model_kwargs` options.
pub struct MiniModelClient {
    models: Models,
    inner: ModelsClient,
    model: Model,
    kwargs: StreamKwargs,
}

impl std::fmt::Debug for MiniModelClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MiniModelClient")
            .field("model", &self.model.id)
            .field("provider", &self.model.provider)
            .finish_non_exhaustive()
    }
}

impl MiniModelClient {
    /// The client for `settings`: delta-ai's built-in providers (credentials from the
    /// environment), or the scripted outputs of a deterministic class.
    pub fn new(settings: &ModelSettings) -> Result<Self, ModelConfigError> {
        if settings.class.deterministic {
            let faux = FauxProvider::new(FauxOptions {
                provider: Some(DETERMINISTIC_PROVIDER.to_owned()),
                models: vec![FauxModelDefinition::new(settings.config.model_name.clone())],
                ..FauxOptions::default()
            });
            faux.set_responses(
                settings
                    .config
                    .outputs
                    .iter()
                    .map(|output| FauxResponse::from(scripted_answer(output)))
                    .collect(),
            );
            let model = faux.model();
            let models = Models::default();
            models.set_provider(faux);
            return Ok(Self::over(models, model, settings.kwargs.clone()));
        }
        let models = builtin_models(CreateModelsOptions {
            auth_context: Some(Arc::new(crate::global::DotenvAuthContext)),
            ..CreateModelsOptions::default()
        });
        let model = resolve_model(&models, &settings.config.model_name, settings.class)?;
        Ok(Self::over(models, model, settings.kwargs.clone()))
    }

    /// A client over `models` that streams `model`.
    pub fn over(models: Models, model: Model, kwargs: StreamKwargs) -> Self {
        Self {
            inner: ModelsClient::new(models.clone()),
            models,
            model,
            kwargs,
        }
    }

    /// The reference the conversation's agent names.
    pub fn reference(&self) -> ModelRef {
        ModelRef::new(&self.model.provider, &self.model.id)
    }

    pub fn resolved(&self) -> &Model {
        &self.model
    }
}

impl ModelClient for MiniModelClient {
    fn model(&self, reference: &ModelRef) -> Option<Model> {
        if reference.provider == self.model.provider && reference.model_id == self.model.id {
            return Some(self.model.clone());
        }
        self.inner.model(reference)
    }

    fn stream(&self, request: ModelRequest<'_>, ctx: &Context) -> AssistantMessageEventStream {
        let options = request.options;
        let base = StreamOptions {
            cancel: Some(ctx.token()),
            headers: options.headers.clone(),
            timeout: options.timeout_ms.map(Duration::from_millis),
            max_retries: options.max_retries,
            max_retry_delay: options.max_retry_delay_ms.map(Duration::from_millis),
            transport: options.transport.map(Transport::from),
            cache_retention: options.cache_retention,
            session_id: request.session_id.map(str::to_owned),
            metadata: options.metadata.clone(),
            max_tokens: self.kwargs.max_tokens.or(request.max_tokens),
            temperature: self.kwargs.temperature,
            sampling_params: self.kwargs.sampling_params.clone(),
            ..StreamOptions::default()
        };
        let mut simple = SimpleStreamOptions::from(base);
        simple.reasoning = self.kwargs.reasoning.or_else(|| request.thinking_level.reasoning());
        let context = delta_ai::types::Context::new(request.messages);
        self.models.stream_simple(request.model, &context, simple)
    }

    fn fetch_deferred<'a>(
        &'a self,
        model: &'a Model,
        handle: &'a DeferredHandle,
        ctx: &'a Context,
    ) -> BoxFuture<'a, AssistantMessage> {
        self.inner.fetch_deferred(model, handle, ctx)
    }

    fn cancel_deferred<'a>(
        &'a self,
        model: &'a Model,
        handle: &'a DeferredHandle,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<(), ModelError>> {
        self.inner.cancel_deferred(model, handle, ctx)
    }
}

/// The cost of an answer: delta-ai's price of its usage.
pub fn answer_cost(message: &AssistantMessage) -> f64 {
    message.usage.cost.total
}

/// upstream's `finish_reason` for a stop reason.
pub fn finish_reason(stop_reason: StopReason) -> &'static str {
    match stop_reason {
        StopReason::Length => "length",
        StopReason::ToolUse => "tool_calls",
        StopReason::Error | StopReason::Aborted => "error",
        _ => "stop",
    }
}

/// Every call and its cost in this process, with upstream's optional limits
/// (`MSWEA_GLOBAL_COST_LIMIT`, `MSWEA_GLOBAL_CALL_LIMIT`).
#[derive(Debug)]
pub struct GlobalModelStats {
    totals: Mutex<(f64, u64)>,
    cost_limit: f64,
    call_limit: u64,
}

impl GlobalModelStats {
    /// The process's stats.
    pub fn get() -> &'static Arc<Self> {
        static STATS: OnceLock<Arc<GlobalModelStats>> = OnceLock::new();
        STATS.get_or_init(|| {
            let parse = |name: &str| crate::global::env_var(name).and_then(|value| value.parse::<f64>().ok());
            Arc::new(Self {
                totals: Mutex::new((0.0, 0)),
                cost_limit: parse("MSWEA_GLOBAL_COST_LIMIT").unwrap_or(0.0),
                call_limit: parse("MSWEA_GLOBAL_CALL_LIMIT").unwrap_or(0.0) as u64,
            })
        })
    }

    /// Adds a call; an error once a limit is exceeded.
    pub fn add(&self, cost: f64) -> Result<(), String> {
        let mut totals = self.totals.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        totals.0 += cost;
        totals.1 += 1;
        let (cost, calls) = *totals;
        if (self.cost_limit > 0.0 && cost > self.cost_limit) || (self.call_limit > 0 && calls + 1 > self.call_limit) {
            return Err(format!("Global cost/call limit exceeded: ${cost:.4} / {calls}"));
        }
        Ok(())
    }

    /// Total cost and calls so far.
    pub fn totals(&self) -> (f64, u64) {
        *self.totals.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_resolve_to_delta_models() {
        let models = builtin_models(CreateModelsOptions::default());
        let toolcall = ModelClass::parse("").unwrap();
        let anthropic = models.get_models(Some("anthropic")).into_iter().next().unwrap();
        let resolved = resolve_model(&models, &format!("anthropic/{}", anthropic.id), toolcall).unwrap();
        assert_eq!(
            (resolved.provider.as_str(), resolved.id.as_str()),
            ("anthropic", anthropic.id.as_str())
        );
        let bare = resolve_model(&models, &anthropic.id, toolcall).unwrap();
        assert_eq!(bare.provider, "anthropic");
        let google = models.get_models(Some("google")).into_iter().next().unwrap();
        assert_eq!(
            resolve_model(&models, &format!("gemini/{}", google.id), toolcall)
                .unwrap()
                .provider,
            "google"
        );
        let unknown = resolve_model(&models, "anthropic/claude-from-the-future", toolcall).unwrap();
        assert_eq!(
            (unknown.id.as_str(), unknown.api.as_str(), unknown.cost.input),
            ("claude-from-the-future", anthropic.api.as_str(), 0.0)
        );
        let openrouter = ModelClass::parse("openrouter").unwrap();
        let routed = resolve_model(&models, "minimax/minimax-m2", openrouter).unwrap();
        assert_eq!(
            (routed.provider.as_str(), routed.id.as_str()),
            ("openrouter", "minimax/minimax-m2")
        );
        assert!(matches!(
            resolve_model(&models, "no-such-model", toolcall),
            Err(ModelConfigError::UnknownModel(_))
        ));
    }

    #[test]
    fn kwargs_become_stream_options() {
        let kwargs = serde_json::json!({
            "temperature": 0.0, "max_tokens": 1000, "reasoning_effort": "high",
            "top_p": 0.9, "drop_params": true, "custom_llm_provider": "x",
        });
        let parsed = StreamKwargs::parse(kwargs.as_object().unwrap()).unwrap();
        assert_eq!(parsed.temperature, Some(0.0));
        assert_eq!(parsed.max_tokens, Some(1000));
        assert_eq!(parsed.reasoning, Some(ThinkingLevel::High));
        assert_eq!(parsed.sampling_params.unwrap()["top_p"], 0.9);
        assert_eq!(parsed.ignored, ["custom_llm_provider"]);
        assert!(StreamKwargs::parse(serde_json::json!({ "temperature": "hot" }).as_object().unwrap()).is_err());
    }

    #[test]
    fn classes_pick_their_defaults() {
        let settings = ModelSettings::new(
            &serde_json::json!({ "model_name": "anthropic/claude-sonnet-4-5", "model_class": "litellm_textbased" }),
            None,
        )
        .unwrap();
        assert_eq!(settings.class.format, ActionFormat::Textbased);
        assert_eq!(settings.format_error_template, DEFAULT_TEXTBASED_FORMAT_ERROR_TEMPLATE);
        assert_eq!(settings.config.set_cache_control.as_deref(), Some("default_end"));
        assert_eq!(settings.dump()["action_regex"], DEFAULT_ACTION_REGEX);
        let overridden = ModelSettings::new(&serde_json::json!({ "model_name": "a/b" }), Some("openai/gpt-5")).unwrap();
        assert_eq!(overridden.config.model_name, "openai/gpt-5");
        assert_eq!(overridden.format_error_template, "{{ error }}");
        assert!(overridden.dump().get("action_regex").is_none());
        assert!(matches!(
            ModelSettings::new(&serde_json::json!({ "model_class": "anthropic" }), Some("x")),
            Err(ModelConfigError::UnknownClass(_))
        ));
    }

    #[test]
    fn scripted_outputs_become_answers() {
        let answer = scripted_answer(&serde_json::json!({
            "role": "assistant",
            "content": "Let me look.",
            "tool_calls": [{ "id": "call_1", "type": "function", "function": { "name": "bash", "arguments": "{\"command\": \"ls\"}" } }],
        }));
        assert_eq!(answer.stop_reason, StopReason::ToolUse);
        let AssistantBlock::ToolCall(call) = &answer.content[1] else {
            panic!("a tool call");
        };
        assert_eq!(
            (call.id.as_str(), call.arguments["command"].as_str()),
            ("call_1", Some("ls"))
        );
        let broken = scripted_answer(&serde_json::json!({
            "tool_calls": [{ "id": "c", "function": { "name": "bash", "arguments": "{" } }],
        }));
        let AssistantBlock::ToolCall(call) = &broken.content[0] else {
            panic!("a tool call");
        };
        assert!(call.arguments_error.is_some());
    }
}
