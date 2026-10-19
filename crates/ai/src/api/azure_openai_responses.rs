//! The `azure-openai-responses` API: Azure OpenAI's v1 Responses endpoint,
//! `POST {baseUrl}/responses?api-version=<version>` with an `api-key` header.
//!
//! Port of upstream `src/api/azure-openai-responses.ts` and `src/api/azure-openai-config.ts`.
//! Azure models ship without a usable base URL: the endpoint resolves per request from the
//! options, `AZURE_OPENAI_BASE_URL` / `AZURE_OPENAI_RESOURCE_NAME`, or `model.base_url`, and is
//! normalized to `/openai/v1` on Azure hosts. The request `model` is the deployment name.

use serde_json::{Map, Value, json};
use std::collections::HashMap;
use url::Url;

use crate::api::common::{apply_on_payload, notify_response, spawn_stream};
use crate::api::openai_responses::{format_provider_error, mapped_effort, merge_sampling_params, simple_api_options};
use crate::api::openai_responses_shared::{
    ConvertResponsesMessagesOptions, ConvertResponsesToolsOptions, GrammarToolInputProperties, ResponsesStreamOptions,
    clamp_openai_prompt_cache_key, convert_responses_messages, convert_responses_tools,
    create_grammar_tool_input_properties, process_responses_stream, responses_compat,
};
use crate::compat::OpenAIResponsesCompat;
use crate::error::{Error, Result};
use crate::events::{AssistantMessageEventStream, MessageBuilder};
use crate::http;
use crate::model::{Model, ModelThinkingLevel};
use crate::options::{ApiOptions, AzureOpenAIResponsesOptions, ReasoningEffort, SimpleStreamOptions, StreamOptions};
use crate::provider::ProviderStreams;
use crate::simple_options::{build_base_options, resolve_sampling_params};
use crate::transcript::{get_declared_tools, resolve_transcript, resolve_transcript_tools};
use crate::types::{ProviderHeaders, StopReason, TranscriptContext};

/// Providers whose tool call ids are already in the Responses `call_id|item_id` shape.
const AZURE_TOOL_CALL_PROVIDERS: [&str; 4] = ["openai", "openai-codex", "opencode", "azure"];
/// OpenAI Responses rejects `max_output_tokens` below 16.
const OPENAI_RESPONSES_MIN_OUTPUT_TOKENS: u64 = 16;
const DEFAULT_AZURE_API_VERSION: &str = "v1";
const API_ID: &str = "azure-openai-responses";

/// The `azure-openai-responses` API.
#[derive(Clone, Copy, Debug, Default)]
pub struct AzureOpenAIResponsesApi;

impl ProviderStreams for AzureOpenAIResponsesApi {
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

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.filter(|value| !value.is_empty())
}

/// `AZURE_OPENAI_DEPLOYMENT_NAME_MAP`: `model=deployment` pairs separated by commas.
fn parse_deployment_name_map(value: Option<&str>) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for entry in value.unwrap_or_default().split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        // JavaScript `split("=", 2)`: the first two pieces, anything after a second `=` dropped.
        let mut parts = entry.split('=');
        let (Some(model_id), Some(deployment)) = (parts.next(), parts.next()) else {
            continue;
        };
        if model_id.is_empty() || deployment.is_empty() {
            continue;
        }
        map.insert(model_id.trim().to_owned(), deployment.trim().to_owned());
    }
    map
}

/// The deployment name sent as the request `model`: `azure_deployment_name`, else the
/// model's entry in `AZURE_OPENAI_DEPLOYMENT_NAME_MAP`, else the model id.
pub fn resolve_deployment_name(model: &Model, azure: &AzureOpenAIResponsesOptions, options: &StreamOptions) -> String {
    if let Some(name) = non_empty(azure.azure_deployment_name.as_deref()) {
        return name.to_owned();
    }
    parse_deployment_name_map(options.env_var("AZURE_OPENAI_DEPLOYMENT_NAME_MAP").as_deref())
        .remove(&model.id)
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| model.id.clone())
}

/// Ensure Azure hosts use `/openai/v1` as the base path; other URLs are kept as they are.
fn normalize_azure_base_url(base_url: &str) -> Result<String> {
    let trimmed = base_url.trim().trim_end_matches('/');
    let mut url = Url::parse(trimmed).map_err(|_| Error::msg(format!("Invalid Azure OpenAI base URL: {base_url}")))?;
    let host = url.host_str().unwrap_or_default();
    let is_azure_host = host.ends_with(".openai.azure.com")
        || host.ends_with(".cognitiveservices.azure.com")
        || host.ends_with(".ai.azure.com");
    let path = url.path().trim_end_matches('/');
    if is_azure_host && matches!(path, "" | "/" | "/openai" | "/openai/v1/responses") {
        url.set_path("/openai/v1");
        url.set_query(None);
    }
    Ok(url.to_string().trim_end_matches('/').to_owned())
}

/// The Azure endpoint: `azure_base_url`, `AZURE_OPENAI_BASE_URL`, a URL built from
/// `azure_resource_name` / `AZURE_OPENAI_RESOURCE_NAME`, or `model.base_url`, normalized.
pub fn resolve_azure_base_url(
    model: &Model,
    azure: &AzureOpenAIResponsesOptions,
    options: &StreamOptions,
) -> Result<String> {
    let env_base_url = options.env_var("AZURE_OPENAI_BASE_URL");
    let base_url = non_empty(azure.azure_base_url.as_deref().map(str::trim))
        .or_else(|| non_empty(env_base_url.as_deref().map(str::trim)))
        .map(str::to_owned);
    let resource_name = non_empty(azure.azure_resource_name.as_deref())
        .map(str::to_owned)
        .or_else(|| options.env_var("AZURE_OPENAI_RESOURCE_NAME"));
    let resolved = base_url
        .or_else(|| resource_name.map(|name| format!("https://{name}.openai.azure.com/openai/v1")))
        .or_else(|| non_empty(Some(&model.base_url)).map(str::to_owned))
        .ok_or_else(|| {
            Error::msg(
                "Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, \
                 or pass azureBaseUrl, azureResourceName, or model.baseUrl.",
            )
        })?;
    normalize_azure_base_url(&resolved)
}

/// The resolved endpoint and API version (`azure_api_version`, `AZURE_OPENAI_API_VERSION`,
/// else `v1`).
pub fn resolve_azure_config(
    model: &Model,
    azure: &AzureOpenAIResponsesOptions,
    options: &StreamOptions,
) -> Result<(String, String)> {
    let base_url = resolve_azure_base_url(model, azure, options)?;
    let api_version = non_empty(azure.azure_api_version.as_deref())
        .map(str::to_owned)
        .or_else(|| options.env_var("AZURE_OPENAI_API_VERSION"))
        .unwrap_or_else(|| DEFAULT_AZURE_API_VERSION.into());
    Ok((base_url, api_version))
}

/// `{base}/responses?api-version=<version>`. The version query replaces any query of the base
/// URL, as the SDK's default query does.
fn responses_url(base_url: &str, api_version: &str) -> Result<String> {
    let mut url = Url::parse(base_url).map_err(|_| Error::msg(format!("Invalid Azure OpenAI base URL: {base_url}")))?;
    let path = format!("{}/responses", url.path().trim_end_matches('/'));
    url.set_path(&path);
    url.set_query(None);
    url.set_fragment(None);
    url.query_pairs_mut().append_pair("api-version", api_version);
    Ok(url.to_string())
}

fn azure_options_of(options: &StreamOptions) -> AzureOpenAIResponsesOptions {
    match &options.api {
        Some(ApiOptions::AzureOpenAIResponses(azure)) => azure.clone(),
        _ => AzureOpenAIResponsesOptions::default(),
    }
}

fn build_params(
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
    azure: &AzureOpenAIResponsesOptions,
    deployment_name: &str,
    grammar: &GrammarToolInputProperties,
) -> Result<Value> {
    let compat: OpenAIResponsesCompat = responses_compat(model);
    let supports_additional_tools = compat.supports_additional_tools.unwrap_or(false);
    let supports_tool_search = compat.supports_tool_search.unwrap_or(false);
    let tool_options = ConvertResponsesToolsOptions {
        supports_strict_mode: Some(compat.supports_strict_mode.unwrap_or(true)),
        supports_openai_grammar_tools: Some(compat.supports_openai_grammar_tools.unwrap_or(false)),
        ..ConvertResponsesToolsOptions::default()
    };
    let transcript_tools =
        resolve_transcript_tools(context.messages(), supports_additional_tools || supports_tool_search);
    let input = convert_responses_messages(
        model,
        context,
        &AZURE_TOOL_CALL_PROVIDERS,
        &ConvertResponsesMessagesOptions {
            grammar_tool_input_properties: Some(grammar),
            supports_mid_convo_system_messages: compat.supports_mid_convo_system_messages.unwrap_or(false),
            supports_additional_tools,
            supports_tool_search,
            tool_options,
            ..ConvertResponsesMessagesOptions::default()
        },
    )?;

    let api = &azure.responses;
    let mut params = Map::new();
    params.insert("model".into(), deployment_name.into());
    params.insert("input".into(), Value::Array(input));
    params.insert("stream".into(), Value::Bool(true));
    if let Some(key) = clamp_openai_prompt_cache_key(options.session_id.as_deref()) {
        params.insert("prompt_cache_key".into(), key.into());
    }
    params.insert("store".into(), Value::Bool(false));
    if let Some(max_tokens) = options.max_tokens.filter(|max| *max > 0) {
        params.insert(
            "max_output_tokens".into(),
            max_tokens.max(OPENAI_RESPONSES_MIN_OUTPUT_TOKENS).into(),
        );
    }
    if let Some(temperature) = options.temperature {
        params.insert("temperature".into(), json!(temperature));
    }
    if !transcript_tools.request_tools.is_empty() {
        params.insert(
            "tools".into(),
            Value::Array(convert_responses_tools(&transcript_tools.request_tools, &tool_options)?),
        );
    }
    if let Some(choice) = &api.tool_choice {
        params.insert("tool_choice".into(), choice.clone());
    }

    let summary = api
        .reasoning_summary
        .clone()
        .flatten()
        .filter(|summary| !summary.is_empty());
    let reasoning_effort = api
        .reasoning_effort
        .clone()
        .or_else(|| summary.is_some().then_some(ReasoningEffort::Medium));
    if model.reasoning {
        if let Some(effort) = &reasoning_effort {
            let effort = match &api.reasoning_effort {
                Some(requested) => mapped_effort(model, requested),
                None => effort.as_str().to_owned(),
            };
            params.insert(
                "reasoning".into(),
                json!({"effort": effort, "summary": summary.as_deref().unwrap_or("auto")}),
            );
            params.insert("include".into(), json!(["reasoning.encrypted_content"]));
        } else if model.mapped_thinking_level(ModelThinkingLevel::Off) != Some(None) {
            let off = model
                .mapped_thinking_level(ModelThinkingLevel::Off)
                .flatten()
                .unwrap_or("none");
            params.insert("reasoning".into(), json!({"effort": off}));
        }
    }

    // Last, so model and request sampling parameters override named request fields.
    let level = reasoning_effort
        .as_ref()
        .and_then(crate::api::openai_responses::effort_level)
        .unwrap_or(ModelThinkingLevel::Off);
    merge_sampling_params(
        &mut params,
        resolve_sampling_params(model, level, options.sampling_params.as_ref()),
    );
    Ok(Value::Object(params))
}

async fn run(
    builder: &mut MessageBuilder,
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
) -> Result<()> {
    let azure = azure_options_of(options);
    let deployment_name = resolve_deployment_name(model, &azure, options);
    let Some(api_key) = non_empty(options.api_key.as_deref()) else {
        return Err(Error::msg(format!("No API key for provider: {}", model.provider)));
    };
    let user_agent = http::user_agent();
    let mut defaults = http::headers_of([
        ("api-key", api_key),
        ("Accept", "application/json"),
        ("Content-Type", "application/json"),
        ("User-Agent", user_agent.as_str()),
    ]);
    for (name, value) in model.headers.iter().flatten() {
        defaults.insert(name.clone(), Some(value.clone()));
    }
    let empty = ProviderHeaders::new();
    let headers = http::build_headers([&defaults, options.headers.as_ref().unwrap_or(&empty)])?;
    let (base_url, api_version) = resolve_azure_config(model, &azure, options)?;

    let compat = responses_compat(model);
    let grammar = create_grammar_tool_input_properties(
        &get_declared_tools(context.messages()),
        compat.supports_openai_grammar_tools.unwrap_or(false),
    )?;
    let params = build_params(model, context, options, &azure, &deployment_name, &grammar)?;
    let params = apply_on_payload(options, model, params).await?;
    let body = serde_json::to_vec(&params)?;
    let url = responses_url(&base_url, &api_version)?;
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
            grammar_tool_input_properties: grammar,
            ..ResponsesStreamOptions::default()
        },
    )
    .await?;

    if builder.output.stop_reason == StopReason::Pending {
        return Err(Error::msg("Azure OpenAI Responses stream ended without a stop reason"));
    }
    Ok(())
}

/// Stream a response from Azure OpenAI Responses.
pub fn stream(model: &Model, context: &TranscriptContext, options: StreamOptions) -> AssistantMessageEventStream {
    let supports_mid_convo = responses_compat(model)
        .supports_mid_convo_system_messages
        .unwrap_or(false);
    let context = resolve_transcript(context, supports_mid_convo);
    let target = model.clone();
    spawn_stream(model, options.cancel.clone(), move |mut builder| async move {
        builder.output.api = API_ID.into();
        let result = run(&mut builder, &target, &context, &options).await;
        let result = result.map_err(|error| format_provider_error(error, "Azure OpenAI API error"));
        (builder, result)
    })
}

/// Stream with the provider-neutral reasoning options. As upstream, only the base options,
/// the tool choice and the clamped reasoning level are passed on; the endpoint resolves from
/// the environment or `model.base_url`.
#[allow(clippy::needless_pass_by_value)] // The API contract takes the options by value.
pub fn stream_simple(
    model: &Model,
    context: &TranscriptContext,
    options: SimpleStreamOptions,
) -> AssistantMessageEventStream {
    let mut base = build_base_options(model, context, &options);
    base.api = Some(ApiOptions::AzureOpenAIResponses(AzureOpenAIResponsesOptions {
        responses: simple_api_options(model, &options),
        ..AzureOpenAIResponsesOptions::default()
    }));
    stream(model, context, base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn model(base_url: &str) -> Model {
        serde_json::from_value(json!({
            "id": "gpt-4o-mini", "name": "GPT-4o mini", "api": "azure-openai-responses", "provider": "azure",
            "baseUrl": base_url, "reasoning": false, "input": ["text", "image"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 128_000, "maxTokens": 16_384
        }))
        .unwrap()
    }

    fn env(pairs: &[(&str, &str)]) -> StreamOptions {
        StreamOptions {
            env: Some(pairs.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect()),
            ..StreamOptions::default()
        }
    }

    #[test]
    fn base_url_normalization() {
        for (input, expected) in [
            (
                "https://marc-quicktests-resource.cognitiveservices.azure.com",
                "https://marc-quicktests-resource.cognitiveservices.azure.com/openai/v1",
            ),
            (
                "https://marc-quicktests-resource.ai.azure.com",
                "https://marc-quicktests-resource.ai.azure.com/openai/v1",
            ),
            (
                "https://my-resource.openai.azure.com",
                "https://my-resource.openai.azure.com/openai/v1",
            ),
            (
                "https://my-resource.openai.azure.com/",
                "https://my-resource.openai.azure.com/openai/v1",
            ),
            (
                "https://my-resource.cognitiveservices.azure.com/openai",
                "https://my-resource.cognitiveservices.azure.com/openai/v1",
            ),
            (
                "https://my-resource.cognitiveservices.azure.com/openai/v1",
                "https://my-resource.cognitiveservices.azure.com/openai/v1",
            ),
            (
                "https://my-resource.services.ai.azure.com/openai/v1/responses",
                "https://my-resource.services.ai.azure.com/openai/v1",
            ),
            ("https://my-proxy.example.com/v1", "https://my-proxy.example.com/v1"),
            (
                "https://my-resource.openai.azure.com/openai?api-version=2024-12-01",
                "https://my-resource.openai.azure.com/openai/v1",
            ),
            (
                "https://my-proxy.example.com/v1?custom=true",
                "https://my-proxy.example.com/v1?custom=true",
            ),
        ] {
            let options = env(&[("AZURE_OPENAI_BASE_URL", input)]);
            assert_eq!(
                resolve_azure_base_url(&model(""), &AzureOpenAIResponsesOptions::default(), &options).unwrap(),
                expected,
                "{input}"
            );
        }
        let options = env(&[("AZURE_OPENAI_BASE_URL", "not-a-url")]);
        let error = resolve_azure_base_url(&model(""), &AzureOpenAIResponsesOptions::default(), &options).unwrap_err();
        assert!(error.to_string().contains("Invalid Azure OpenAI base URL"));
    }

    #[test]
    fn base_url_sources() {
        let none = AzureOpenAIResponsesOptions::default();
        let blank = env(&[]);
        let error = resolve_azure_base_url(&model(""), &none, &blank).unwrap_err();
        assert!(error.to_string().starts_with("Azure OpenAI base URL is required."));
        let resource = AzureOpenAIResponsesOptions {
            azure_resource_name: Some("res".into()),
            ..AzureOpenAIResponsesOptions::default()
        };
        assert_eq!(
            resolve_azure_base_url(&model(""), &resource, &blank).unwrap(),
            "https://res.openai.azure.com/openai/v1"
        );
        let env_resource = env(&[("AZURE_OPENAI_RESOURCE_NAME", "my-resource")]);
        assert_eq!(
            resolve_azure_base_url(&model("https://ignored.example.com"), &none, &env_resource).unwrap(),
            "https://my-resource.openai.azure.com/openai/v1"
        );
        let explicit = AzureOpenAIResponsesOptions {
            azure_base_url: Some(" https://x.openai.azure.com ".into()),
            ..AzureOpenAIResponsesOptions::default()
        };
        assert_eq!(
            resolve_azure_base_url(&model(""), &explicit, &env_resource).unwrap(),
            "https://x.openai.azure.com/openai/v1"
        );
        assert_eq!(
            resolve_azure_base_url(&model("https://m.openai.azure.com"), &none, &blank).unwrap(),
            "https://m.openai.azure.com/openai/v1"
        );
    }

    #[test]
    fn api_version_and_deployment_names() {
        let m = model("https://r.openai.azure.com");
        let none = AzureOpenAIResponsesOptions::default();
        assert_eq!(resolve_azure_config(&m, &none, &env(&[])).unwrap().1, "v1");
        let versioned = AzureOpenAIResponsesOptions {
            azure_api_version: Some("2025-01-01".into()),
            ..AzureOpenAIResponsesOptions::default()
        };
        assert_eq!(resolve_azure_config(&m, &versioned, &env(&[])).unwrap().1, "2025-01-01");
        assert_eq!(
            resolve_azure_config(&m, &none, &env(&[("AZURE_OPENAI_API_VERSION", "preview")]))
                .unwrap()
                .1,
            "preview"
        );
        assert_eq!(resolve_deployment_name(&m, &none, &env(&[])), "gpt-4o-mini");
        let mapped = env(&[(
            "AZURE_OPENAI_DEPLOYMENT_NAME_MAP",
            " other=x, gpt-4o-mini=my-deploy=ignored ,bad",
        )]);
        assert_eq!(resolve_deployment_name(&m, &none, &mapped), "my-deploy");
        let explicit = AzureOpenAIResponsesOptions {
            azure_deployment_name: Some("explicit".into()),
            ..AzureOpenAIResponsesOptions::default()
        };
        assert_eq!(resolve_deployment_name(&m, &explicit, &mapped), "explicit");
    }

    #[test]
    fn responses_url_replaces_the_query() {
        assert_eq!(
            responses_url("https://r.openai.azure.com/openai/v1", "v1").unwrap(),
            "https://r.openai.azure.com/openai/v1/responses?api-version=v1"
        );
        assert_eq!(
            responses_url("https://my-proxy.example.com/v1?custom=true", "2025-04-01").unwrap(),
            "https://my-proxy.example.com/v1/responses?api-version=2025-04-01"
        );
    }
}
