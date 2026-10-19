//! The `openai-responses` API: `POST {baseUrl}/responses` with SSE streaming.
//!
//! Port of upstream `src/api/openai-responses.ts`. Conversion and stream processing live in
//! [`super::openai_responses_shared`].
//!
//! Not ported: Sign in with ChatGPT specifics (request fields it rejects, the usage-limit
//! hint), GitHub Copilot dynamic headers, and the OpenAI Codex (ChatGPT subscription) API.

use serde_json::{Map, Value, json};

use crate::api::common::{self, apply_on_payload, join_url, notify_response, require_auth, spawn_stream};
use crate::api::openai_responses_shared::{
    ConvertResponsesMessagesOptions, ConvertResponsesToolsOptions, GrammarToolInputProperties, ResponsesStreamOptions,
    clamp_openai_prompt_cache_key, convert_responses_messages, convert_responses_tools,
    create_grammar_tool_input_properties, process_responses_stream, responses_compat,
};
use crate::compat::SessionAffinityFormat;
use crate::error::{Error, Result};
use crate::events::{AssistantMessageEventStream, MessageBuilder};
use crate::http;
use crate::model::{CacheRetention, Model, ModelThinkingLevel};
use crate::options::{
    ApiOptions, OpenAIResponsesOptions, ReasoningEffort, SimpleStreamOptions, StreamOptions, ToolChoice,
};
use crate::provider::ProviderStreams;
use crate::simple_options::{build_base_options, resolve_sampling_params};
use crate::transcript::{get_declared_tools, resolve_transcript, resolve_transcript_tools};
use crate::types::{ProviderHeaders, StopReason, TranscriptContext, Usage};

/// Providers whose tool call ids are already in the Responses `call_id|item_id` shape.
const OPENAI_TOOL_CALL_PROVIDERS: [&str; 3] = ["openai", "openai-codex", "opencode"];
/// OpenAI Responses rejects `max_output_tokens` below 16.
const OPENAI_RESPONSES_MIN_OUTPUT_TOKENS: u64 = 16;
/// Fixed `reasoning.effort` while `configuration_update` items carry the thinking level.
const CONFIGURATION_UPDATE_REQUEST_EFFORT: &str = "medium";

/// The `openai-responses` API.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAIResponsesApi;

impl ProviderStreams for OpenAIResponsesApi {
    fn stream(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        stream(model, context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        stream_simple(model, context, options)
    }
}

/// Compat settings with upstream defaults applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedCompat {
    pub(crate) supports_mid_convo_system_messages: bool,
    pub(crate) session_affinity_format: SessionAffinityFormat,
    pub(crate) supports_long_cache_retention: bool,
    pub(crate) supports_strict_mode: bool,
    pub(crate) supports_openai_grammar_tools: bool,
    pub(crate) supports_additional_tools: bool,
    pub(crate) supports_tool_search: bool,
    pub(crate) supports_explicit_prompt_cache_mode: bool,
    pub(crate) supports_max_output_tokens: bool,
    pub(crate) supports_mid_convo_effort: bool,
}

fn detect_session_affinity_format(model: &Model) -> SessionAffinityFormat {
    if model.provider == "openrouter" || model.base_url.contains("openrouter.ai") {
        SessionAffinityFormat::Openrouter
    } else {
        SessionAffinityFormat::Openai
    }
}

fn get_compat(model: &Model) -> ResolvedCompat {
    let compat = responses_compat(model);
    ResolvedCompat {
        supports_mid_convo_system_messages: compat.supports_mid_convo_system_messages.unwrap_or(false),
        session_affinity_format: compat
            .session_affinity_format
            .unwrap_or_else(|| detect_session_affinity_format(model)),
        supports_long_cache_retention: compat.supports_long_cache_retention.unwrap_or(true),
        supports_strict_mode: compat.supports_strict_mode.unwrap_or(false),
        supports_openai_grammar_tools: compat.supports_openai_grammar_tools.unwrap_or(false),
        supports_additional_tools: compat.supports_additional_tools.unwrap_or(false),
        supports_tool_search: compat.supports_tool_search.unwrap_or(false),
        supports_explicit_prompt_cache_mode: compat.supports_explicit_prompt_cache_mode.unwrap_or(false),
        supports_max_output_tokens: compat.supports_max_output_tokens.unwrap_or(true),
        supports_mid_convo_effort: compat.supports_mid_convo_effort.unwrap_or(false),
    }
}

/// The thinking level a reasoning effort names, for thinking level map lookups.
pub(crate) fn effort_level(effort: &ReasoningEffort) -> Option<ModelThinkingLevel> {
    match effort {
        ReasoningEffort::Minimal => Some(ModelThinkingLevel::Minimal),
        ReasoningEffort::Low => Some(ModelThinkingLevel::Low),
        ReasoningEffort::Medium => Some(ModelThinkingLevel::Medium),
        ReasoningEffort::High => Some(ModelThinkingLevel::High),
        ReasoningEffort::Xhigh => Some(ModelThinkingLevel::Xhigh),
        ReasoningEffort::Max => Some(ModelThinkingLevel::Max),
        ReasoningEffort::None => None,
        ReasoningEffort::Other(value) => ModelThinkingLevel::ALL
            .into_iter()
            .find(|level| level.as_str() == value),
    }
}

/// The provider value of a requested effort: the model's thinking level map entry, else the
/// effort itself.
pub(crate) fn mapped_effort(model: &Model, effort: &ReasoningEffort) -> String {
    effort_level(effort)
        .and_then(|level| model.mapped_thinking_level(level).flatten())
        .unwrap_or(effort.as_str())
        .to_owned()
}

/// `cache_retention`, else `PI_CACHE_RETENTION=long`, else short.
fn resolve_cache_retention(options: &StreamOptions) -> CacheRetention {
    if let Some(retention) = options.cache_retention {
        return retention;
    }
    if options.env_var("PI_CACHE_RETENTION").as_deref() == Some("long") {
        return CacheRetention::Long;
    }
    CacheRetention::Short
}

/// Effort to send in `configuration_update` items, or `None` to send it as `reasoning.effort`.
fn get_configuration_update_effort(
    model: &Model,
    api_options: &OpenAIResponsesOptions,
    compat: &ResolvedCompat,
) -> Option<String> {
    // Not ported: Sign in with ChatGPT rejects configuration_update items upstream.
    if !compat.supports_mid_convo_effort {
        return None;
    }
    let effort = api_options.reasoning_effort.as_ref()?;
    Some(mapped_effort(model, effort))
}

fn get_prompt_cache_retention(compat: &ResolvedCompat, retention: CacheRetention) -> Option<&'static str> {
    (retention == CacheRetention::Long
        && compat.supports_long_cache_retention
        && !compat.supports_explicit_prompt_cache_mode)
        .then_some("24h")
}

fn get_prompt_cache_options(compat: &ResolvedCompat, retention: CacheRetention) -> Option<Value> {
    if !compat.supports_explicit_prompt_cache_mode {
        return None;
    }
    match retention {
        CacheRetention::None => Some(json!({"mode": "explicit"})),
        CacheRetention::Long if compat.supports_long_cache_retention => Some(json!({"ttl": "30m"})),
        _ => None,
    }
}

fn service_tier_cost_multiplier(model: &Model, service_tier: Option<&str>) -> f64 {
    match service_tier {
        Some("flex") => 0.5,
        Some("priority" | "fast") => {
            if model.id == "gpt-5.5" {
                2.5
            } else {
                2.0
            }
        }
        _ => 1.0,
    }
}

/// Scale the cost of `usage` by the OpenAI service tier price multiplier.
pub fn apply_service_tier_pricing(model: &Model, usage: &mut Usage, service_tier: Option<&str>) {
    let multiplier = service_tier_cost_multiplier(model, service_tier);
    if (multiplier - 1.0).abs() < f64::EPSILON {
        return;
    }
    let cost = &mut usage.cost;
    cost.input *= multiplier;
    cost.output *= multiplier;
    cost.cache_read *= multiplier;
    cost.cache_write *= multiplier;
    cost.total = cost.input + cost.output + cost.cache_read + cost.cache_write;
}

/// API options of the request; options of other APIs are ignored.
fn api_options_of(options: &StreamOptions) -> OpenAIResponsesOptions {
    match &options.api {
        Some(ApiOptions::OpenAIResponses(api)) => api.clone(),
        _ => OpenAIResponsesOptions::default(),
    }
}

pub(crate) fn tool_choice_value(choice: ToolChoice) -> Value {
    Value::String(
        match choice {
            ToolChoice::Auto => "auto",
            ToolChoice::None => "none",
        }
        .into(),
    )
}

/// Insert `value` into `params`, as `Object.assign` does: existing keys keep their position.
pub(crate) fn merge_sampling_params(params: &mut Map<String, Value>, sampling: Option<crate::types::JsonObject>) {
    for (key, value) in sampling.into_iter().flatten() {
        params.insert(key, value);
    }
}

/// Build the request body.
fn build_params(
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
    api_options: &OpenAIResponsesOptions,
    compat: &ResolvedCompat,
    grammar: &GrammarToolInputProperties,
    configuration_update_effort: Option<&str>,
) -> Result<Value> {
    let transcript_tools = resolve_transcript_tools(
        context.messages(),
        compat.supports_additional_tools || compat.supports_tool_search,
    );
    let tool_options = ConvertResponsesToolsOptions {
        supports_strict_mode: Some(compat.supports_strict_mode),
        supports_openai_grammar_tools: Some(compat.supports_openai_grammar_tools),
        ..ConvertResponsesToolsOptions::default()
    };
    let input = convert_responses_messages(
        model,
        context,
        &OPENAI_TOOL_CALL_PROVIDERS,
        &ConvertResponsesMessagesOptions {
            grammar_tool_input_properties: Some(grammar),
            supports_mid_convo_system_messages: compat.supports_mid_convo_system_messages,
            supports_additional_tools: compat.supports_additional_tools,
            supports_tool_search: compat.supports_tool_search,
            tool_options,
            configuration_update_effort: configuration_update_effort.map(str::to_owned),
            ..ConvertResponsesMessagesOptions::default()
        },
    )?;

    // Not ported: Sign in with ChatGPT (a non-`sk-` key sent to api.openai.com) omits
    // prompt_cache_retention, prompt_cache_options, max_output_tokens and temperature upstream.
    let retention = resolve_cache_retention(options);
    let mut params = Map::new();
    params.insert("model".into(), model.id.clone().into());
    params.insert("input".into(), Value::Array(input));
    params.insert("stream".into(), Value::Bool(true));
    if retention != CacheRetention::None
        && let Some(key) = clamp_openai_prompt_cache_key(options.session_id.as_deref())
    {
        params.insert("prompt_cache_key".into(), key.into());
    }
    if let Some(value) = get_prompt_cache_retention(compat, retention) {
        params.insert("prompt_cache_retention".into(), value.into());
    }
    if let Some(value) = get_prompt_cache_options(compat, retention) {
        params.insert("prompt_cache_options".into(), value);
    }
    params.insert("store".into(), Value::Bool(false));

    if let Some(max_tokens) = options.max_tokens.filter(|max| *max > 0)
        && compat.supports_max_output_tokens
    {
        params.insert(
            "max_output_tokens".into(),
            max_tokens.max(OPENAI_RESPONSES_MIN_OUTPUT_TOKENS).into(),
        );
    }
    if let Some(temperature) = options.temperature {
        params.insert("temperature".into(), json!(temperature));
    }
    if let Some(tier) = &api_options.service_tier {
        params.insert("service_tier".into(), tier.clone().into());
    }
    if !transcript_tools.request_tools.is_empty() {
        params.insert(
            "tools".into(),
            Value::Array(convert_responses_tools(&transcript_tools.request_tools, &tool_options)?),
        );
    }
    if let Some(choice) = &api_options.tool_choice {
        params.insert("tool_choice".into(), choice.clone());
    }

    let summary = api_options
        .reasoning_summary
        .clone()
        .flatten()
        .filter(|summary| !summary.is_empty());
    let reasoning_effort = api_options
        .reasoning_effort
        .clone()
        .or_else(|| summary.is_some().then_some(ReasoningEffort::Medium));
    if model.reasoning {
        if let Some(effort) = &reasoning_effort {
            let requested = match &api_options.reasoning_effort {
                Some(requested) => mapped_effort(model, requested),
                None => effort.as_str().to_owned(),
            };
            let effort = if configuration_update_effort.is_some() {
                CONFIGURATION_UPDATE_REQUEST_EFFORT.to_owned()
            } else {
                requested
            };
            params.insert(
                "reasoning".into(),
                json!({"effort": effort, "summary": summary.as_deref().unwrap_or("auto")}),
            );
            params.insert("include".into(), json!(["reasoning.encrypted_content"]));
        } else if model.provider != "github-copilot"
            && model.mapped_thinking_level(ModelThinkingLevel::Off) != Some(None)
        {
            let off = model
                .mapped_thinking_level(ModelThinkingLevel::Off)
                .flatten()
                .unwrap_or("none");
            params.insert("reasoning".into(), json!({"effort": off}));
        }
        if model.provider == "xai" {
            params.insert("include".into(), json!(["reasoning.encrypted_content"]));
        }
    }

    // Last, so model and request sampling parameters override named request fields.
    let level = reasoning_effort
        .as_ref()
        .and_then(effort_level)
        .unwrap_or(ModelThinkingLevel::Off);
    merge_sampling_params(
        &mut params,
        resolve_sampling_params(model, level, options.sampling_params.as_ref()),
    );
    Ok(Value::Object(params))
}

fn build_request_headers(
    model: &Model,
    options: &StreamOptions,
    api_key: &str,
    session_id: Option<&str>,
) -> Result<reqwest::header::HeaderMap> {
    let compat = get_compat(model);
    let authorization = format!("Bearer {api_key}");
    let user_agent = http::user_agent();
    let mut defaults = http::headers_of([
        ("Authorization", authorization.as_str()),
        ("Accept", "application/json"),
        ("Content-Type", "application/json"),
        ("User-Agent", user_agent.as_str()),
    ]);
    for (name, value) in model.headers.iter().flatten() {
        defaults.insert(name.clone(), Some(value.clone()));
    }
    // Not ported: GitHub Copilot dynamic headers.
    if let Some(session_id) = session_id {
        let mut set = |name: &str| {
            defaults.insert(name.to_owned(), Some(session_id.to_owned()));
        };
        match compat.session_affinity_format {
            SessionAffinityFormat::Openrouter => set("x-session-id"),
            SessionAffinityFormat::Openai => {
                set("session_id");
                set("x-client-request-id");
            }
            SessionAffinityFormat::OpenaiNosession => set("x-client-request-id"),
        }
    }
    let empty = ProviderHeaders::new();
    http::build_headers([&defaults, options.headers.as_ref().unwrap_or(&empty)])
}

/// Format an error like upstream's `formatProviderError(normalizeProviderError(error),
/// "<prefix> API error")` over the OpenAI SDK error: HTTP errors become
/// `"<prefix> (<status>): <error JSON>"` when the body has an `error` object, else
/// `"<prefix> (<status>): <SDK message>"` (e.g. `"... (502): 502 Bad Gateway"`). Other errors
/// keep their message.
pub(crate) fn format_provider_error(error: Error, prefix: &str) -> Error {
    match error {
        Error::Status { status, body, .. } => {
            let (message, error_object) = common::openai_sdk_error(status, body.as_deref());
            Error::msg(common::format_provider_error(
                status,
                &message,
                error_object.as_ref(),
                Some(prefix),
            ))
        }
        other => other,
    }
}

async fn run(
    builder: &mut MessageBuilder,
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
) -> Result<()> {
    let compat = get_compat(model);
    let api_options = api_options_of(options);
    let configuration_update_effort = get_configuration_update_effort(model, &api_options, &compat);
    if let Some(effort) = &configuration_update_effort {
        builder.output.provider_thinking_level = Some(effort.clone());
    }

    require_auth(model, options, &["authorization", "cf-aig-authorization"])?;
    let api_key = options
        .api_key
        .clone()
        .filter(|key| !key.is_empty())
        .unwrap_or_else(|| "unused".into());
    let retention = resolve_cache_retention(options);
    let cache_session_id = options
        .session_id
        .as_deref()
        .filter(|_| retention != CacheRetention::None);
    let grammar = create_grammar_tool_input_properties(
        &get_declared_tools(context.messages()),
        compat.supports_openai_grammar_tools,
    )?;
    let headers = build_request_headers(model, options, &api_key, cache_session_id)?;
    let params = build_params(
        model,
        context,
        options,
        &api_options,
        &compat,
        &grammar,
        configuration_update_effort.as_deref(),
    )?;
    let params = apply_on_payload(options, model, params).await?;
    let body = serde_json::to_vec(&params)?;
    let url = join_url(&model.base_url, "responses");
    let client = http::client(options);
    let response = http::send_with_retry(options, || {
        Ok(client.post(&url).headers(headers.clone()).body(body.clone()))
    })
    .await?;
    notify_response(options, model, &response).await?;
    builder.start();

    process_responses_stream(
        http::sse_events(response, options.cancel.clone()),
        builder,
        model,
        options,
        ResponsesStreamOptions {
            service_tier: api_options.service_tier.clone(),
            grammar_tool_input_properties: grammar,
            resolve_service_tier: None,
            apply_service_tier_pricing: Some(apply_service_tier_pricing),
        },
    )
    .await?;

    if builder.output.stop_reason == StopReason::Pending {
        return Err(Error::msg("OpenAI Responses stream ended without a stop reason"));
    }
    Ok(())
}

/// Stream a response from the OpenAI Responses API.
pub fn stream(model: &Model, context: &TranscriptContext, options: StreamOptions) -> AssistantMessageEventStream {
    let context = resolve_transcript(context, get_compat(model).supports_mid_convo_system_messages);
    let target = model.clone();
    let prefix = if model.provider == "openai" {
        "OpenAI".to_owned()
    } else {
        model.provider.clone()
    };
    spawn_stream(model, options.cancel.clone(), move |mut builder| async move {
        let result = run(&mut builder, &target, &context, &options).await;
        // Not ported: the ChatGPT usage-limit hint appended to Sign in with ChatGPT errors.
        let result = result.map_err(|error| format_provider_error(error, &format!("{prefix} API error")));
        (builder, result)
    })
}

/// Stream with the provider-neutral reasoning options: the reasoning level, clamped to the
/// model's supported levels, becomes `reasoning_effort` (`off` sends none).
#[allow(clippy::needless_pass_by_value)] // The API contract takes the options by value.
pub fn stream_simple(
    model: &Model,
    context: &TranscriptContext,
    options: SimpleStreamOptions,
) -> AssistantMessageEventStream {
    let mut base = build_base_options(model, context, &options);
    base.api = Some(ApiOptions::OpenAIResponses(simple_api_options(model, &options)));
    stream(model, context, base)
}

/// The Responses options of a simple request.
pub(crate) fn simple_api_options(model: &Model, options: &SimpleStreamOptions) -> OpenAIResponsesOptions {
    let reasoning_effort = options
        .reasoning
        .map(|level| model.clamp_thinking_level(level.into()))
        .and_then(ModelThinkingLevel::reasoning)
        .map(ReasoningEffort::from);
    OpenAIResponsesOptions {
        reasoning_effort,
        tool_choice: options.tool_choice.map(tool_choice_value),
        ..OpenAIResponsesOptions::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::normalize_context;
    use crate::types::{Context, Message, Tool};
    use pretty_assertions::assert_eq;

    fn model(id: &str, map: Option<Value>) -> Model {
        let mut value = json!({
            "id": id, "name": id, "api": "openai-responses", "provider": "openai",
            "baseUrl": "https://api.openai.com/v1", "reasoning": true, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 400_000, "maxTokens": 128_000
        });
        if let Some(map) = map {
            value["thinkingLevelMap"] = map;
        }
        serde_json::from_value(value).unwrap()
    }

    fn params(model: &Model, options: &StreamOptions) -> Value {
        let context = normalize_context(&Context::new(vec![Message::user("hi")]).with_system_prompt("sys"));
        let compat = get_compat(model);
        let api_options = api_options_of(options);
        let effort = get_configuration_update_effort(model, &api_options, &compat);
        build_params(
            model,
            &context,
            options,
            &api_options,
            &compat,
            &GrammarToolInputProperties::new(),
            effort.as_deref(),
        )
        .unwrap()
    }

    fn with_api(api: OpenAIResponsesOptions) -> StreamOptions {
        StreamOptions {
            api: Some(ApiOptions::OpenAIResponses(api)),
            ..StreamOptions::default()
        }
    }

    #[test]
    fn reasoning_effort_mapping() {
        // Off: "none" unless the model maps off to null.
        let m = model("gpt-5.4", None);
        assert_eq!(
            params(&m, &StreamOptions::default())["reasoning"],
            json!({"effort": "none"})
        );
        let no_off = model("gpt-5", Some(json!({"off": null})));
        assert!(params(&no_off, &StreamOptions::default()).get("reasoning").is_none());
        // Requested efforts go through the thinking level map.
        let mapped = model("gpt-x", Some(json!({"xhigh": "extra"})));
        let body = params(
            &mapped,
            &with_api(OpenAIResponsesOptions {
                reasoning_effort: Some(ReasoningEffort::Xhigh),
                reasoning_summary: Some(Some("detailed".into())),
                ..OpenAIResponsesOptions::default()
            }),
        );
        assert_eq!(body["reasoning"], json!({"effort": "extra", "summary": "detailed"}));
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        // A summary alone requests medium effort.
        let body = params(
            &m,
            &with_api(OpenAIResponsesOptions {
                reasoning_summary: Some(Some("concise".into())),
                ..OpenAIResponsesOptions::default()
            }),
        );
        assert_eq!(body["reasoning"], json!({"effort": "medium", "summary": "concise"}));
        // Non-reasoning models send no reasoning.
        let mut plain = model("gpt-4o", None);
        plain.reasoning = false;
        assert!(params(&plain, &StreamOptions::default()).get("reasoning").is_none());
    }

    #[test]
    fn request_fields_and_order() {
        let m = model("gpt-5.4", None);
        let mut options = with_api(OpenAIResponsesOptions {
            service_tier: Some("priority".into()),
            tool_choice: Some(json!("required")),
            ..OpenAIResponsesOptions::default()
        });
        options.max_tokens = Some(4);
        options.temperature = Some(0.5);
        options.session_id = Some("s".repeat(70));
        options.sampling_params = Some(
            [("temperature".to_owned(), json!(0.2)), ("top_p".to_owned(), json!(0.9))]
                .into_iter()
                .collect(),
        );
        let body = params(&m, &options);
        let keys: Vec<&str> = body.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "model",
                "input",
                "stream",
                "prompt_cache_key",
                "store",
                "max_output_tokens",
                "temperature",
                "service_tier",
                "tool_choice",
                "reasoning",
                "top_p"
            ]
        );
        assert_eq!(body["max_output_tokens"], 16);
        assert_eq!(body["temperature"], 0.2);
        assert_eq!(body["prompt_cache_key"], "s".repeat(64));
        assert_eq!(body["store"], false);
    }

    #[test]
    fn prompt_cache_fields_per_compat() {
        let mut m = model("gpt-5.4", None);
        let long = StreamOptions {
            cache_retention: Some(CacheRetention::Long),
            session_id: Some("session-2".into()),
            ..StreamOptions::default()
        };
        assert_eq!(params(&m, &long)["prompt_cache_retention"], "24h");
        let none = StreamOptions {
            cache_retention: Some(CacheRetention::None),
            session_id: Some("session-2".into()),
            ..StreamOptions::default()
        };
        let body = params(&m, &none);
        assert!(body.get("prompt_cache_key").is_none() && body.get("prompt_cache_options").is_none());

        m.compat = Some(crate::compat::ModelCompat::OpenAIResponses(
            crate::compat::OpenAIResponsesCompat {
                supports_explicit_prompt_cache_mode: Some(true),
                ..Default::default()
            },
        ));
        assert_eq!(params(&m, &none)["prompt_cache_options"], json!({"mode": "explicit"}));
        let body = params(&m, &long);
        assert_eq!(body["prompt_cache_options"], json!({"ttl": "30m"}));
        assert!(body.get("prompt_cache_retention").is_none());

        m.compat = Some(crate::compat::ModelCompat::OpenAIResponses(
            crate::compat::OpenAIResponsesCompat {
                supports_long_cache_retention: Some(false),
                supports_max_output_tokens: Some(false),
                ..Default::default()
            },
        ));
        let mut long_capped = long.clone();
        long_capped.max_tokens = Some(1024);
        let body = params(&m, &long_capped);
        assert!(body.get("prompt_cache_retention").is_none() && body.get("max_output_tokens").is_none());

        // PI_CACHE_RETENTION=long through the provider env.
        let m = model("gpt-5.4", None);
        let env = StreamOptions {
            env: Some(
                [("PI_CACHE_RETENTION".to_owned(), "long".to_owned())]
                    .into_iter()
                    .collect(),
            ),
            ..StreamOptions::default()
        };
        assert_eq!(params(&m, &env)["prompt_cache_retention"], "24h");
    }

    #[test]
    fn mid_convo_effort_sends_constant_effort() {
        let mut m = model(
            "gpt-6-luna",
            Some(json!({"off": "none", "minimal": null, "low": "low", "medium": "medium", "high": "high"})),
        );
        m.compat = Some(crate::compat::ModelCompat::OpenAIResponses(
            crate::compat::OpenAIResponsesCompat {
                supports_mid_convo_effort: Some(true),
                ..Default::default()
            },
        ));
        let body = params(
            &m,
            &with_api(OpenAIResponsesOptions {
                reasoning_effort: Some(ReasoningEffort::Low),
                ..OpenAIResponsesOptions::default()
            }),
        );
        assert_eq!(body["reasoning"]["effort"], "medium");
        assert_eq!(
            body["input"].as_array().unwrap().last().unwrap(),
            &json!({"type": "configuration_update", "reasoning": {"effort": "low"}})
        );
        // Thinking off sends the off effort and no update items.
        let body = params(&m, &StreamOptions::default());
        assert_eq!(body["reasoning"]["effort"], "none");
        assert!(
            !body["input"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["type"] == "configuration_update")
        );
    }

    #[test]
    fn xai_requests_encrypted_reasoning() {
        let mut m = model("grok-4.5", Some(json!({"off": null})));
        m.provider = "xai".into();
        let body = params(&m, &StreamOptions::default());
        assert!(body.get("reasoning").is_none());
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    }

    #[test]
    fn tools_and_strict_mode() {
        let m = model("gpt-5.4", None);
        let context = normalize_context(&Context::new(vec![Message::user("hi")]).with_tools(vec![Tool::new(
            "ping",
            "Ping",
            json!({"type": "object", "properties": {"value": {"type": "string"}}}),
        )]));
        let compat = get_compat(&m);
        let body = build_params(
            &m,
            &context,
            &StreamOptions::default(),
            &OpenAIResponsesOptions::default(),
            &compat,
            &GrammarToolInputProperties::new(),
            None,
        )
        .unwrap();
        assert_eq!(
            body["tools"],
            json!([{"type": "function", "name": "ping", "description": "Ping",
                    "parameters": {"type": "object", "properties": {"value": {"type": "string"}}}}])
        );
    }

    #[test]
    fn service_tier_pricing() {
        let mut usage = Usage::default();
        usage.cost.input = 1.0;
        usage.cost.output = 2.0;
        apply_service_tier_pricing(&model("gpt-5.5", None), &mut usage, Some("priority"));
        assert!((usage.cost.total - 7.5).abs() < 1e-9);
        let mut usage = Usage::default();
        usage.cost.input = 1.0;
        apply_service_tier_pricing(&model("gpt-6-luna", None), &mut usage, Some("fast"));
        assert!((usage.cost.total - 2.0).abs() < 1e-9);
        apply_service_tier_pricing(&model("gpt-5.4", None), &mut usage, Some("flex"));
        assert!((usage.cost.total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn http_errors_get_the_provider_prefix() {
        let status = |status: u16, body: Option<&str>| Error::Status {
            status,
            message: String::new(),
            body: body.map(str::to_owned),
            retry_after_ms: None,
        };
        // The SDK's `error` object is printed as JSON after the prefix.
        let body = r#"{"error":{"message":"Invalid model","type":"invalid_request_error","param":null,"code":null}}"#;
        assert_eq!(
            format_provider_error(status(400, Some(body)), "OpenAI API error").to_string(),
            r#"OpenAI API error (400): {"message":"Invalid model","type":"invalid_request_error","param":null,"code":null}"#
        );
        // A body without an `error` object keeps the SDK message.
        assert_eq!(
            format_provider_error(status(502, Some("Bad Gateway")), "OpenAI API error").to_string(),
            "OpenAI API error (502): 502 Bad Gateway"
        );
        assert_eq!(
            format_provider_error(
                status(403, Some(r#"{"error":"blocked by gateway WAF"}"#)),
                "xai API error"
            )
            .to_string(),
            r#"xai API error (403): 403 "blocked by gateway WAF""#
        );
        assert_eq!(
            format_provider_error(status(500, None), "Azure OpenAI API error").to_string(),
            "Azure OpenAI API error (500): 500 status code (no body)"
        );
        assert_eq!(
            format_provider_error(Error::msg("boom"), "OpenAI API error").to_string(),
            "boom"
        );
    }

    #[test]
    fn simple_options_map_reasoning_and_tool_choice() {
        let m = model("gpt-5.4", Some(json!({"minimal": null})));
        let options = SimpleStreamOptions {
            reasoning: Some(crate::model::ThinkingLevel::Minimal),
            tool_choice: Some(ToolChoice::None),
            ..SimpleStreamOptions::default()
        };
        let api = simple_api_options(&m, &options);
        assert_eq!(api.reasoning_effort, Some(ReasoningEffort::Low));
        assert_eq!(api.tool_choice, Some(json!("none")));
        let mut plain = m.clone();
        plain.reasoning = false;
        assert_eq!(simple_api_options(&plain, &options).reasoning_effort, None);
    }
}
