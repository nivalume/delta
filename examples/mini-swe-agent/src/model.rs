//! Models: the `model` and `providers` sections, on delta-ai.
//!
//! `model.provider` and `model.id` name a delta-ai model: one of its built-in providers' or
//! one `providers` defines (built with delta-ai's `create_provider`). The harness streams it
//! through [`MiniModelClient`], which adds `temperature`, `max_tokens` and `sampling`.
//! Provider `faux` replays `model.script` through delta-ai's faux provider.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use delta_ai::api::anthropic_messages::AnthropicMessagesApi;
use delta_ai::api::google_generative_ai::GoogleGenerativeAiApi;
use delta_ai::api::mistral_conversations::MistralConversationsApi;
use delta_ai::api::openai_completions::OpenAICompletionsApi;
use delta_ai::api::openai_responses::OpenAIResponsesApi;
use delta_ai::model::{AnyModel, Model, ModelThinkingLevel};
use delta_ai::options::{SimpleStreamOptions, StreamOptions, Transport};
use delta_ai::providers::builtin::builtin_models;
use delta_ai::providers::faux::{FauxModelDefinition, FauxOptions, FauxProvider, FauxResponse, faux_assistant_message};
use delta_ai::types::{AssistantBlock, AssistantMessage, DeferredHandle, JsonObject};
use delta_ai::{
    AssistantMessageEventStream, CreateModelsOptions, CreateProviderOptions, CredentialStore, Models, ProviderApi,
    ProviderAuth, ProviderStreams, create_provider, env_api_key_auth,
};
use delta_chord::context::Context;
use delta_harness::model::{ModelClient, ModelError, ModelRef, ModelRequest, ModelsClient};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json, json};

/// The default observation template.
pub const DEFAULT_OBSERVATION_TEMPLATE: &str = "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>";

/// The default format error template of tool-call actions.
pub const DEFAULT_TOOL_CALLS_FORMAT_ERROR_TEMPLATE: &str = "{{ error }}";

/// The default format error template of text actions.
pub const DEFAULT_TEXT_FORMAT_ERROR_TEMPLATE: &str =
    "Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions.";

/// The default `action_regex` of text actions.
pub const DEFAULT_ACTION_REGEX: &str = r"```mswea_bash_command\s*\n(.*?)\n```";

/// The provider whose answers `model.script` holds.
pub const FAUX_PROVIDER: &str = "faux";

/// How the model states its actions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionFormat {
    /// Calls of the `bash` tool.
    #[default]
    ToolCalls,
    /// One fenced block matched by `action_regex`.
    Text,
}

/// Why a model could not be set up.
#[derive(Debug, thiserror::Error)]
pub enum ModelConfigError {
    #[error("No model configured. Run `mini-extra config setup`, or pass `-m provider/model-id`.")]
    NoModel,
    #[error("Unknown provider {provider} (providers: {available})")]
    UnknownProvider { provider: String, available: String },
    #[error("No provider has a model {0}; `mini-extra models --all` lists them")]
    UnknownModel(String),
    #[error("Several providers have a model {id}: {candidates}; name one as provider/{id}")]
    AmbiguousModel { id: String, candidates: String },
    #[error("Invalid model config: {0}")]
    Invalid(serde_json::Error),
    #[error("Invalid provider {id}: {reason}")]
    Provider { id: String, reason: String },
    #[error("Invalid model.action_regex: {0}")]
    ActionRegex(String),
    #[error("Invalid model.script[{index}]: {reason}")]
    Script { index: usize, reason: String },
}

/// The `model` section of a config.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// A delta-ai provider id: built in, defined under `providers`, or `faux`.
    #[serde(default)]
    pub provider: String,
    /// The model id in the provider's catalog.
    #[serde(default)]
    pub id: String,
    /// The conversation's thinking level; unset: the harness's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ModelThinkingLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// The output token limit of each call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Sampling parameters added to the request body (`top_p`, `seed`, ...).
    #[serde(default, skip_serializing_if = "JsonObject::is_empty")]
    pub sampling: JsonObject,
    /// Replaces the model's base URL (a proxy, a local server).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Headers added to the model's.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub actions: ActionFormat,
    /// Text actions: the pattern of an action; its first group is the command.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_regex: Option<String>,
    #[serde(default = "default_observation_template")]
    pub observation_template: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format_error_template: Option<String>,
    /// Provider `faux`: the answers, in order. Each is `{content: [AssistantBlock], cost}`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub script: Vec<Json>,
}

fn default_observation_template() -> String {
    DEFAULT_OBSERVATION_TEMPLATE.to_owned()
}

/// A provider the config defines: delta-ai's `create_provider` with one implementation per
/// API its models use.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    /// Display name; default: the id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The API of models that name none: `openai-completions`, `openai-responses`,
    /// `anthropic-messages`, `google-generative-ai` or `mistral-conversations`.
    pub api: String,
    /// The base URL of models that name none.
    pub base_url: String,
    /// Variables holding the API key, first set wins; a key stored with `mini-extra login`
    /// wins over them. Without any, the provider needs no key.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub api_key_env: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    /// delta-ai model definitions (camelCase JSON); every field but `id` has a default.
    #[serde(default)]
    pub models: Vec<Json>,
}

/// The APIs a config's provider can use.
pub const PROVIDER_APIS: [&str; 5] = [
    "openai-completions",
    "openai-responses",
    "anthropic-messages",
    "google-generative-ai",
    "mistral-conversations",
];

fn provider_streams(api: &str) -> Option<Arc<dyn ProviderStreams>> {
    Some(match api {
        "openai-completions" => Arc::new(OpenAICompletionsApi),
        "openai-responses" => Arc::new(OpenAIResponsesApi),
        "anthropic-messages" => Arc::new(AnthropicMessagesApi),
        "google-generative-ai" => Arc::new(GoogleGenerativeAiApi),
        "mistral-conversations" => Arc::new(MistralConversationsApi),
        _ => return None,
    })
}

impl ProviderConfig {
    /// The provider `id` this config defines.
    pub fn build(&self, id: &str) -> Result<delta_ai::provider::BuiltProvider, ModelConfigError> {
        let invalid = |reason: String| ModelConfigError::Provider {
            id: id.to_owned(),
            reason,
        };
        if self.models.is_empty() {
            return Err(invalid("no models".to_owned()));
        }
        let mut models = Vec::new();
        for definition in &self.models {
            let model_id = definition
                .get("id")
                .and_then(Json::as_str)
                .ok_or_else(|| invalid("a model without an id".to_owned()))?;
            let mut model = json!({
                "id": model_id,
                "name": model_id,
                "api": self.api,
                "provider": id,
                "baseUrl": self.base_url,
                "input": ["text"],
                "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0 },
                "reasoning": false,
                "contextWindow": 128_000,
                "maxTokens": 16_384,
            });
            crate::config::merge(&mut model, definition.clone());
            model["provider"] = json!(id);
            let model: Model =
                serde_json::from_value(model).map_err(|error| invalid(format!("model {model_id}: {error}")))?;
            models.push(model);
        }
        let mut apis = BTreeMap::new();
        for api in models.iter().map(|model| model.api.clone()).collect::<BTreeSet<_>>() {
            let streams = provider_streams(&api)
                .ok_or_else(|| invalid(format!("unknown api {api} (available: {})", PROVIDER_APIS.join(", "))))?;
            apis.insert(api, streams);
        }
        let name = self.name.clone().unwrap_or_else(|| id.to_owned());
        let auth = if self.api_key_env.is_empty() {
            ProviderAuth::keyless(name.clone())
        } else {
            let vars: Vec<&str> = self.api_key_env.iter().map(String::as_str).collect();
            ProviderAuth::api_key(env_api_key_auth(format!("{name} API key"), &vars))
        };
        let mut options = CreateProviderOptions::new(
            id,
            auth,
            models.into_iter().map(AnyModel::Chat).collect(),
            ProviderApi::ByApi(apis.into_iter().collect()),
        )
        .name(name)
        .base_url(self.base_url.clone());
        if !self.headers.is_empty() {
            options.headers = Some(
                self.headers
                    .iter()
                    .map(|(name, value)| (name.clone(), Some(value.clone())))
                    .collect(),
            );
        }
        Ok(create_provider(options))
    }
}

/// The `providers` section.
pub fn provider_configs(section: &Json) -> Result<BTreeMap<String, ProviderConfig>, ModelConfigError> {
    match section {
        Json::Null => Ok(BTreeMap::new()),
        section => serde_json::from_value(section.clone()).map_err(|error| ModelConfigError::Provider {
            id: "providers".to_owned(),
            reason: error.to_string(),
        }),
    }
}

/// delta-ai's built-in providers and the ones `providers` defines (which replace a built-in
/// provider of the same id), with credentials from `credentials` (in memory when `None`).
pub fn build_models(
    providers: &Json,
    credentials: Option<Arc<dyn CredentialStore>>,
) -> Result<Models, ModelConfigError> {
    let models = builtin_models(CreateModelsOptions {
        credentials,
        ..CreateModelsOptions::default()
    });
    for (id, config) in provider_configs(providers)? {
        models.set_provider(config.build(&id)?);
    }
    Ok(models)
}

/// `-m`'s value as provider and model id: `provider/id`, split at the first `/`, or a bare
/// id (an empty provider).
pub fn split_model_ref(spec: &str) -> (String, String) {
    match spec.split_once('/') {
        Some((provider, id)) if !provider.is_empty() && !id.is_empty() => (provider.to_owned(), id.to_owned()),
        _ => (String::new(), spec.to_owned()),
    }
}

/// The model `provider` and `id` name, and a warning to print about it.
///
/// - No provider: the one model with that id across every provider.
/// - A known provider: its model, or, for an id its catalog lacks, its first model's API and
///   base URL with that id and no price.
/// - An unknown provider: `provider/id` as a bare id, which covers ids that hold a `/`.
pub fn resolve_model(models: &Models, provider: &str, id: &str) -> Result<(Model, Option<String>), ModelConfigError> {
    if id.is_empty() {
        return Err(ModelConfigError::NoModel);
    }
    if provider.is_empty() {
        return find_bare(models, id).map(|model| (model, None));
    }
    if models.provider(provider).is_none() {
        return find_bare(models, &format!("{provider}/{id}"))
            .map(|model| (model, None))
            .map_err(|_| ModelConfigError::UnknownProvider {
                provider: provider.to_owned(),
                available: models
                    .providers()
                    .iter()
                    .map(|provider| provider.id().to_owned())
                    .collect::<Vec<_>>()
                    .join(", "),
            });
    }
    if let Some(model) = models.get_model(provider, id) {
        return Ok((model, None));
    }
    let template = models
        .get_models(Some(provider))
        .into_iter()
        .next()
        .ok_or_else(|| ModelConfigError::UnknownModel(format!("{provider}/{id}")))?;
    let warning = format!(
        "{provider} has no model {id} in its catalog; sending it with the API of {} and no price",
        template.id
    );
    Ok((
        Model {
            id: id.to_owned(),
            name: id.to_owned(),
            cost: delta_ai::model::ModelCost::default(),
            ..template
        },
        Some(warning),
    ))
}

fn find_bare(models: &Models, id: &str) -> Result<Model, ModelConfigError> {
    let mut found: Vec<Model> = models
        .get_models(None)
        .into_iter()
        .filter(|model| model.id == id)
        .collect();
    match found.len() {
        0 => Err(ModelConfigError::UnknownModel(id.to_owned())),
        1 => Ok(found.remove(0)),
        _ => Err(ModelConfigError::AmbiguousModel {
            id: id.to_owned(),
            candidates: found
                .iter()
                .map(|model| format!("{}/{}", model.provider, model.id))
                .collect::<Vec<_>>()
                .join(", "),
        }),
    }
}

/// A `model` section resolved for a run: its config with the defaults its action format
/// implies.
#[derive(Clone, Debug)]
pub struct ModelSettings {
    pub config: ModelConfig,
    pub format_error_template: String,
    pub action_regex: regex::Regex,
}

impl ModelSettings {
    pub fn new(section: &Json) -> Result<Self, ModelConfigError> {
        let config: ModelConfig = serde_json::from_value(if section.is_object() {
            section.clone()
        } else {
            Json::Object(Map::new())
        })
        .map_err(ModelConfigError::Invalid)?;
        let format_error_template = config.format_error_template.clone().unwrap_or_else(|| {
            match config.actions {
                ActionFormat::ToolCalls => DEFAULT_TOOL_CALLS_FORMAT_ERROR_TEMPLATE,
                ActionFormat::Text => DEFAULT_TEXT_FORMAT_ERROR_TEMPLATE,
            }
            .to_owned()
        });
        // `.` matches line breaks: a command may span lines.
        let action_regex = regex::RegexBuilder::new(config.action_regex.as_deref().unwrap_or(DEFAULT_ACTION_REGEX))
            .dot_matches_new_line(true)
            .build()
            .map_err(|error| ModelConfigError::ActionRegex(error.to_string()))?;
        Ok(Self {
            config,
            format_error_template,
            action_regex,
        })
    }

    /// Whether `model.script` answers instead of a provider.
    pub fn is_scripted(&self) -> bool {
        self.config.provider == FAUX_PROVIDER
    }

    /// The config as the run uses it, for templates and the trajectory.
    pub fn dump(&self) -> Json {
        let mut dump = serde_json::to_value(&self.config).unwrap_or(Json::Null);
        if let Some(object) = dump.as_object_mut() {
            object.insert(
                "format_error_template".to_owned(),
                Json::String(self.format_error_template.clone()),
            );
            if self.config.actions == ActionFormat::Text {
                object.insert(
                    "action_regex".to_owned(),
                    Json::String(self.action_regex.as_str().to_owned()),
                );
            }
        }
        dump
    }

    /// The scripted answers and their costs.
    pub fn script(&self) -> Result<Vec<(AssistantMessage, f64)>, ModelConfigError> {
        self.config
            .script
            .iter()
            .enumerate()
            .map(|(index, item)| script_answer(index, item))
            .collect()
    }
}

/// One `model.script` item: `{content: [AssistantBlock], cost}`. A tool call without an id
/// gets `call_<item>_<block>`, one without arguments `{}`.
pub fn script_answer(index: usize, item: &Json) -> Result<(AssistantMessage, f64), ModelConfigError> {
    let invalid = |reason: String| ModelConfigError::Script { index, reason };
    let object = item.as_object().ok_or_else(|| invalid("not an object".to_owned()))?;
    if let Some(key) = object.keys().find(|key| !["content", "cost"].contains(&key.as_str())) {
        return Err(invalid(format!("unknown field {key}")));
    }
    let mut content = object.get("content").cloned().unwrap_or_else(|| json!([]));
    for (block_index, block) in content.as_array_mut().into_iter().flatten().enumerate() {
        if block["type"] == "toolCall" {
            if block.get("id").is_none() {
                block["id"] = json!(format!("call_{index}_{block_index}"));
            }
            if block.get("arguments").is_none() {
                block["arguments"] = json!({});
            }
        }
    }
    let blocks: Vec<AssistantBlock> = serde_json::from_value(content).map_err(|error| invalid(error.to_string()))?;
    let cost = match object.get("cost") {
        None => 0.0,
        Some(cost) => cost
            .as_f64()
            .ok_or_else(|| invalid("cost is not a number".to_owned()))?,
    };
    Ok((faux_assistant_message(blocks), cost))
}

/// What the client adds to every request.
#[derive(Clone, Debug, Default, PartialEq)]
struct RequestSettings {
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    sampling: Option<JsonObject>,
}

/// The harness's model access for one run: the resolved model, streamed with the `model`
/// section's request settings.
pub struct MiniModelClient {
    models: Models,
    inner: ModelsClient,
    model: Model,
    settings: RequestSettings,
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
    /// The client for `settings`, over the built-in providers and `providers`, with
    /// `credentials`; or the faux provider with the script. Also returns a warning to print.
    pub fn new(
        settings: &ModelSettings,
        providers: &Json,
        credentials: Option<Arc<dyn CredentialStore>>,
    ) -> Result<(Self, Option<String>), ModelConfigError> {
        let config = &settings.config;
        let request = RequestSettings {
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            sampling: (!config.sampling.is_empty()).then(|| config.sampling.clone()),
        };
        if settings.is_scripted() {
            let id = if config.id.is_empty() { "scripted" } else { &config.id };
            let faux = FauxProvider::new(FauxOptions {
                provider: Some(FAUX_PROVIDER.to_owned()),
                models: vec![FauxModelDefinition::new(id)],
                ..FauxOptions::default()
            });
            faux.set_responses(
                settings
                    .script()?
                    .into_iter()
                    .map(|(answer, _)| FauxResponse::from(answer))
                    .collect(),
            );
            let model = faux.model();
            let models = Models::default();
            models.set_provider(faux);
            return Ok((Self::over(models, model, request), None));
        }
        let models = build_models(providers, credentials)?;
        let (mut model, warning) = resolve_model(&models, &config.provider, &config.id)?;
        if let Some(base_url) = &config.base_url {
            model.base_url.clone_from(base_url);
        }
        if !config.headers.is_empty() {
            let headers = model.headers.get_or_insert_with(Default::default);
            for (name, value) in &config.headers {
                headers.insert(name.clone(), value.clone());
            }
        }
        Ok((Self::over(models, model, request), warning))
    }

    fn over(models: Models, model: Model, settings: RequestSettings) -> Self {
        Self {
            inner: ModelsClient::new(models.clone()),
            models,
            model,
            settings,
        }
    }

    /// The reference the conversation's agent names.
    pub fn reference(&self) -> ModelRef {
        ModelRef::new(&self.model.provider, &self.model.id)
    }

    /// The model this client streams.
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
            max_tokens: self.settings.max_tokens.or(request.max_tokens),
            temperature: self.settings.temperature,
            sampling_params: self.settings.sampling.clone(),
            ..StreamOptions::default()
        };
        let mut simple = SimpleStreamOptions::from(base);
        simple.reasoning = request.thinking_level.reasoning();
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

/// Whether `model` has a price: without one, calls cost nothing and a cost limit never trips.
pub fn has_price(model: &Model) -> bool {
    let cost = &model.cost;
    cost.input > 0.0 || cost.output > 0.0 || cost.cache_read > 0.0 || cost.cache_write > 0.0
}

/// The calls and the cost of a whole batch, with its limits.
#[derive(Debug, Default)]
pub struct BatchStats {
    totals: Mutex<(f64, u64)>,
    cost_limit: f64,
    call_limit: u64,
}

impl BatchStats {
    /// Stats with limits; 0: no limit.
    pub fn new(cost_limit: f64, call_limit: u64) -> Self {
        Self {
            totals: Mutex::new((0.0, 0)),
            cost_limit,
            call_limit,
        }
    }

    /// Adds a call.
    pub fn add(&self, cost: f64) {
        let mut totals = self.totals.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        totals.0 += cost;
        totals.1 += 1;
    }

    /// Why the batch may start no more calls: a limit is reached.
    pub fn exceeded(&self) -> Option<String> {
        let (cost, calls) = self.totals();
        ((self.cost_limit > 0.0 && cost >= self.cost_limit) || (self.call_limit > 0 && calls >= self.call_limit))
            .then(|| format!("The batch's limits are reached: ${cost:.4} in {calls} calls"))
    }

    /// Total cost and calls so far.
    pub fn totals(&self) -> (f64, u64) {
        *self.totals.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use delta_ai::types::StopReason;

    use super::*;

    #[test]
    fn model_refs_split_at_the_first_slash() {
        assert_eq!(
            split_model_ref("anthropic/claude-sonnet-4-5"),
            ("anthropic".to_owned(), "claude-sonnet-4-5".to_owned())
        );
        assert_eq!(
            split_model_ref("openrouter/minimax/minimax-m2"),
            ("openrouter".to_owned(), "minimax/minimax-m2".to_owned())
        );
        assert_eq!(split_model_ref("gpt-5"), (String::new(), "gpt-5".to_owned()));
        assert_eq!(split_model_ref("/x"), (String::new(), "/x".to_owned()));
    }

    #[test]
    fn models_resolve_by_provider_and_id() {
        let models = build_models(&Json::Null, None).unwrap();
        let anthropic = models.get_models(Some("anthropic")).into_iter().next().unwrap();
        let (resolved, warning) = resolve_model(&models, "anthropic", &anthropic.id).unwrap();
        assert_eq!((resolved.provider.as_str(), warning), ("anthropic", None));

        let routed = models
            .get_models(Some("openrouter"))
            .into_iter()
            .find(|model| model.id.contains('/'))
            .unwrap();
        let (provider, id) = split_model_ref(&format!("openrouter/{}", routed.id));
        assert_eq!(resolve_model(&models, &provider, &id).unwrap().0.id, routed.id);

        let (unknown, warning) = resolve_model(&models, "anthropic", "claude-from-the-future").unwrap();
        assert_eq!(
            (unknown.id.as_str(), unknown.api.as_str(), has_price(&unknown)),
            ("claude-from-the-future", anthropic.api.as_str(), false)
        );
        assert!(warning.unwrap().contains("no model claude-from-the-future"));

        // An id two providers serve needs its provider.
        let mut by_id: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for model in models.get_models(None) {
            by_id.entry(model.id).or_default().push(model.provider);
        }
        let (shared, _) = by_id.iter().find(|(_, providers)| providers.len() > 1).unwrap();
        assert!(matches!(
            resolve_model(&models, "", shared),
            Err(ModelConfigError::AmbiguousModel { .. })
        ));
        let (unique, _) = by_id.iter().find(|(_, providers)| providers.len() == 1).unwrap();
        assert_eq!(resolve_model(&models, "", unique).unwrap().0.id, *unique);
        assert!(matches!(
            resolve_model(&models, "", "no-such-model"),
            Err(ModelConfigError::UnknownModel(_))
        ));
        assert!(matches!(
            resolve_model(&models, "no-such-provider", "x"),
            Err(ModelConfigError::UnknownProvider { .. })
        ));
        assert!(matches!(resolve_model(&models, "", ""), Err(ModelConfigError::NoModel)));
    }

    #[test]
    fn providers_are_built_from_the_config() {
        let providers = json!({
            "local": {
                "name": "Local server",
                "api": "openai-completions",
                "base_url": "http://localhost:8000/v1",
                "models": [
                    { "id": "qwen3-coder", "contextWindow": 32768, "cost": { "input": 1.0 } },
                    { "id": "other", "api": "anthropic-messages", "baseUrl": "http://localhost:9000" },
                ],
            },
        });
        let models = build_models(&providers, None).unwrap();
        let provider = models.provider("local").unwrap();
        assert_eq!(provider.name(), "Local server");
        let qwen = models.get_model("local", "qwen3-coder").unwrap();
        assert_eq!(
            (
                qwen.api.as_str(),
                qwen.base_url.as_str(),
                qwen.context_window,
                qwen.cost.input,
                qwen.cost.output
            ),
            ("openai-completions", "http://localhost:8000/v1", 32768, 1.0, 0.0)
        );
        assert_eq!(models.get_model("local", "other").unwrap().api, "anthropic-messages");
        assert!(models.provider("anthropic").is_some(), "the built-in providers stay");

        let invalid = |providers: Json| build_models(&providers, None).unwrap_err().to_string();
        assert!(
            invalid(json!({ "x": { "api": "nope", "base_url": "u", "models": [{ "id": "m" }] } }))
                .contains("unknown api nope")
        );
        assert!(
            invalid(json!({ "x": { "api": "openai-completions", "base_url": "u", "models": [] } }))
                .contains("no models")
        );
        assert!(
            invalid(json!({ "x": { "api": "openai-completions", "base_url": "u", "model": [] } }))
                .contains("unknown field")
        );
    }

    #[test]
    fn settings_take_their_format_s_defaults() {
        let text = ModelSettings::new(&json!({ "provider": "anthropic", "id": "x", "actions": "text" })).unwrap();
        assert_eq!(text.format_error_template, DEFAULT_TEXT_FORMAT_ERROR_TEMPLATE);
        assert_eq!(text.dump()["action_regex"], DEFAULT_ACTION_REGEX);
        let calls = ModelSettings::new(&json!({ "provider": "anthropic", "id": "x", "thinking": "high" })).unwrap();
        assert_eq!(calls.config.actions, ActionFormat::ToolCalls);
        assert_eq!(calls.config.thinking, Some(ModelThinkingLevel::High));
        assert_eq!(calls.format_error_template, "{{ error }}");
        assert!(calls.dump().get("action_regex").is_none());
        assert!(matches!(
            ModelSettings::new(&json!({ "model_name": "a/b" })),
            Err(ModelConfigError::Invalid(_))
        ));
        assert!(matches!(
            ModelSettings::new(&json!({ "actions": "text", "action_regex": "(" })),
            Err(ModelConfigError::ActionRegex(_))
        ));
    }

    #[test]
    fn script_items_become_answers() {
        let (answer, cost) = script_answer(
            2,
            &json!({
                "content": [
                    { "type": "text", "text": "Let me look." },
                    { "type": "toolCall", "name": "bash", "arguments": { "command": "ls" } },
                ],
                "cost": 0.5,
            }),
        )
        .unwrap();
        assert_eq!((answer.stop_reason, cost), (StopReason::ToolUse, 0.5));
        let AssistantBlock::ToolCall(call) = &answer.content[1] else {
            panic!("a tool call");
        };
        assert_eq!(
            (call.id.as_str(), call.arguments["command"].as_str()),
            ("call_2_1", Some("ls"))
        );
        let (text, cost) = script_answer(0, &json!({ "content": [{ "type": "text", "text": "done" }] })).unwrap();
        assert_eq!((text.stop_reason, cost), (StopReason::Stop, 0.0));
        assert!(script_answer(0, &json!({ "content": [], "outputs": 1 })).is_err());
        assert!(script_answer(0, &json!({ "content": [{ "type": "image" }] })).is_err());
    }

    #[test]
    fn batch_limits_are_reached_at_their_value() {
        let stats = BatchStats::new(1.0, 5);
        stats.add(0.4);
        stats.add(0.4);
        assert_eq!(stats.exceeded(), None);
        stats.add(0.4);
        assert!(stats.exceeded().unwrap().contains("$1.2000 in 3 calls"));
        let calls = BatchStats::new(0.0, 1);
        assert_eq!(calls.exceeded(), None);
        calls.add(0.0);
        assert!(calls.exceeded().is_some());
        assert_eq!(BatchStats::default().exceeded(), None);
    }
}
