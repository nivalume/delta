//! Amazon Bedrock Converse streaming (`bedrock-converse-stream`) through the official AWS SDK
//! for Rust (`aws-sdk-bedrockruntime`, `ConverseStream`).
//!
//! Port of upstream `api/bedrock-converse-stream.ts`. The request is first built as JSON in the
//! shape of the SDK's `ConverseStreamCommand` input (camelCase members, binary members such as
//! image bytes and redacted reasoning base64-encoded), so `on_payload` sees and may replace the
//! same payload as upstream. The JSON is then converted to the SDK's typed input.
//!
//! Provider stream events reach `on_provider_stream_event` in the same JSON shape
//! (`{"contentBlockDelta": {...}}`, `{"throttlingException": {"message": ...}}`, ...).

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use aws_config::BehaviorVersion;
use aws_config::profile::ProfileFileCredentialsProvider;
use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::config::interceptors::BeforeDeserializationInterceptorContextRef;
use aws_sdk_bedrockruntime::config::retry::RetryConfig;
use aws_sdk_bedrockruntime::config::timeout::TimeoutConfig;
use aws_sdk_bedrockruntime::config::{
    ConfigBag, Credentials, Intercept, Region, RuntimeComponents, StalledStreamProtectionConfig, Token,
};
use aws_sdk_bedrockruntime::error::{BoxError, ProvideErrorMetadata, SdkError};
use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError;
use aws_sdk_bedrockruntime::types as bedrock;
use aws_sdk_bedrockruntime::types::error::ConverseStreamOutputError;
use aws_smithy_types::event_stream::RawMessage;
use aws_smithy_types::{Blob, Document, Number};
use base64::Engine as _;
use base64::engine::DecodePaddingMode;
use base64::engine::general_purpose::{GeneralPurpose, GeneralPurposeConfig, STANDARD};
use indexmap::IndexMap;
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use crate::api::common::{apply_on_payload, observe_provider_event, spawn_stream};
use crate::api::constrained_sampling::{get_json_schema_tool_parameters, resolve_json_schema_strict_sampling};
use crate::compat::ModelCompat;
use crate::error::{Error, Result};
use crate::events::{AssistantMessageEventStream, MessageBuilder};
use crate::http::{DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, MAX_PROVIDER_ERROR_BODY_CHARS, truncate_error_text};
use crate::json::parse_streaming_json;
use crate::model::{CacheRetention, Model, ModelThinkingLevel, ThinkingLevel, calculate_cost};
use crate::options::{
    AnthropicToolChoice, ApiOptions, BedrockOptions, ProviderResponse, SimpleStreamOptions, StreamOptions,
    ThinkingBudgets, ThinkingDisplay, ToolChoice,
};
use crate::provider::ProviderStreams;
use crate::simple_options::{
    adjust_max_tokens_for_thinking, build_base_options, clamp_max_tokens_to_context, clamp_reasoning,
};
use crate::transcript::{
    collapse_system_messages, get_current_tools, get_initial_system_message, get_system_message_text,
    without_initial_system_message,
};
use crate::transform::{split_error_tool_result_images, transform_messages};
use crate::types::{
    AssistantBlock, AssistantMessageDiagnostic, JsonObject, Message, ProviderEnv, ProviderHeaders, StopReason, Tool,
    TranscriptContext, UserBlock, UserContent,
};

const EMPTY_TEXT_PLACEHOLDER: &str = "<empty>";
const THINKING_BINDING_CONTROLS_BETA: &str = "thinking-binding-controls-2026-08-01";
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
/// Matches the placeholder the Anthropic API path uses for redacted thinking.
const REDACTED_THINKING_PLACEHOLDER: &str = "[Reasoning redacted]";
const FAILURE_DIAGNOSTIC_TYPE: &str = "bedrock_response_failure";
/// Over-long header values are dropped rather than truncated: a truncated request id is not a
/// request id.
const MAX_DIAGNOSTIC_VALUE_CHARS: usize = 200;
/// Some models reject the account/profile's configured data retention mode; point users at the
/// AWS docs explaining how to configure a supported mode.
const DATA_RETENTION_DOCS_URL: &str = "https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html";

/// Human-readable prefixes for Bedrock exception names. Downstream retry logic matches patterns
/// like `server.?error` and `service.?unavailable`, so the legacy prefixes are kept rather than
/// the raw exception names.
const ERROR_PREFIXES: [(&str, &str); 5] = [
    ("InternalServerException", "Internal server error"),
    ("ModelStreamErrorException", "Model stream error"),
    ("ValidationException", "Validation error"),
    ("ThrottlingException", "Throttling error"),
    ("ServiceUnavailableException", "Service unavailable"),
];

/// The `bedrock-converse-stream` API.
#[derive(Clone, Copy, Debug, Default)]
pub struct BedrockConverseStreamApi;

impl ProviderStreams for BedrockConverseStreamApi {
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

/// Stream a Bedrock `ConverseStream` response.
pub fn stream(model: &Model, context: &TranscriptContext, options: StreamOptions) -> AssistantMessageEventStream {
    // Bedrock has no mid-conversation system messages; fold them into the leading prompt.
    let context = collapse_system_messages(context);
    let owned_model = model.clone();
    spawn_stream(model, options.cancel.clone(), move |mut builder| async move {
        let result = Box::pin(run(&owned_model, &context, &options, &mut builder)).await;
        (builder, result)
    })
}

/// Map the provider-neutral simple options onto Bedrock options, then [`stream`].
#[allow(clippy::needless_pass_by_value)] // Same signature as every API's `stream_simple`.
pub fn stream_simple(
    model: &Model,
    context: &TranscriptContext,
    options: SimpleStreamOptions,
) -> AssistantMessageEventStream {
    let mut base = build_base_options(model, context, &options);
    let mut bedrock = BedrockOptions {
        tool_choice: options.tool_choice.map(|choice| match choice {
            ToolChoice::Auto => AnthropicToolChoice::Auto,
            ToolChoice::None => AnthropicToolChoice::None,
        }),
        ..BedrockOptions::default()
    };
    if let Some(reasoning) = options.reasoning {
        bedrock.reasoning = Some(reasoning);
        bedrock.thinking_budgets = options.thinking_budgets;
        if is_anthropic_claude_model(model) && !supports_adaptive_thinking(model) {
            // `base.max_tokens` is always set (clamped to the context window); `None` would mean
            // no caller cap and let the helper use the model cap.
            let (adjusted_max_tokens, thinking_budget) = adjust_max_tokens_for_thinking(
                base.max_tokens,
                model.max_tokens,
                reasoning,
                options.thinking_budgets.as_ref(),
            );
            let max_tokens = clamp_max_tokens_to_context(model, context, adjusted_max_tokens);
            let budget = thinking_budget.min(max_tokens.saturating_sub(1024));
            let mut budgets = options.thinking_budgets.unwrap_or_default();
            match clamp_reasoning(reasoning) {
                ThinkingLevel::Minimal => budgets.minimal = Some(budget),
                ThinkingLevel::Low => budgets.low = Some(budget),
                ThinkingLevel::Medium => budgets.medium = Some(budget),
                _ => budgets.high = Some(budget),
            }
            base.max_tokens = Some(max_tokens);
            bedrock.thinking_budgets = Some(budgets);
        }
    }
    base.api = Some(ApiOptions::BedrockConverseStream(bedrock));
    stream(model, context, base)
}

fn bedrock_options(options: &StreamOptions) -> BedrockOptions {
    match &options.api {
        Some(ApiOptions::BedrockConverseStream(bedrock)) => bedrock.clone(),
        _ => BedrockOptions::default(),
    }
}

// ---------------------------------------------------------------------------------------------
// Environment and client configuration
// ---------------------------------------------------------------------------------------------

/// Environment lookups of one request: scoped `options.env` values first, then the process
/// environment. Blank values count as unset.
struct EnvLookup<'a> {
    scoped: Option<&'a ProviderEnv>,
    process: &'a (dyn Fn(&str) -> Option<String> + Sync),
}

impl EnvLookup<'_> {
    fn get(&self, name: &str) -> Option<String> {
        self.scoped(name).or_else(|| self.process(name))
    }

    fn scoped(&self, name: &str) -> Option<String> {
        self.scoped
            .and_then(|env| env.get(name))
            .filter(|value| !value.trim().is_empty())
            .cloned()
    }

    fn process(&self, name: &str) -> Option<String> {
        (self.process)(name).filter(|value| !value.trim().is_empty())
    }
}

fn process_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn non_empty(value: Option<&str>) -> Option<String> {
    value.filter(|value| !value.is_empty()).map(str::to_owned)
}

/// Static SigV4 credentials.
#[derive(Clone, PartialEq, Eq)]
struct StaticCredentials {
    access_key_id: String,
    secret_access_key: String,
    session_token: Option<String>,
}

impl fmt::Debug for StaticCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StaticCredentials")
            .field("access_key_id", &self.access_key_id)
            .finish_non_exhaustive()
    }
}

/// What upstream passes to `new BedrockRuntimeClient(config)`.
#[derive(Clone, Default, PartialEq, Eq)]
struct ClientPlan {
    region: Option<String>,
    /// Explicit endpoint; `None` lets the SDK derive it from the region.
    endpoint: Option<String>,
    profile: Option<String>,
    credentials: Option<StaticCredentials>,
    /// Bedrock API key; replaces SigV4 with `Authorization: Bearer`.
    bearer_token: Option<String>,
    /// `AWS_BEDROCK_SKIP_AUTH=1`: sign with dummy credentials, never with a bearer token.
    skip_auth: bool,
}

impl fmt::Debug for ClientPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClientPlan")
            .field("region", &self.region)
            .field("endpoint", &self.endpoint)
            .field("profile", &self.profile)
            .field("credentials", &self.credentials)
            .field("bearer_token", &self.bearer_token.as_ref().map(|_| "<redacted>"))
            .field("skip_auth", &self.skip_auth)
            .finish()
    }
}

fn configured_region(bedrock: &BedrockOptions, env: &EnvLookup<'_>) -> Option<String> {
    non_empty(bedrock.region.as_deref())
        .or_else(|| env.get("AWS_REGION"))
        .or_else(|| env.get("AWS_DEFAULT_REGION"))
}

fn configured_credentials(env: &EnvLookup<'_>) -> Option<StaticCredentials> {
    Some(StaticCredentials {
        access_key_id: env.get("AWS_ACCESS_KEY_ID")?,
        secret_access_key: env.get("AWS_SECRET_ACCESS_KEY")?,
        session_token: env.get("AWS_SESSION_TOKEN"),
    })
}

fn is_region_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

/// Region of a standard Bedrock runtime endpoint (`bedrock-runtime[-fips].<region>.amazonaws.com[.cn]`).
fn standard_endpoint_region(base_url: &str) -> Option<String> {
    if base_url.is_empty() {
        return None;
    }
    let url = url::Url::parse(base_url).ok()?;
    let host = url.host_str()?.to_ascii_lowercase();
    let rest = host
        .strip_prefix("bedrock-runtime-fips.")
        .or_else(|| host.strip_prefix("bedrock-runtime."))?;
    let region = rest
        .strip_suffix(".amazonaws.com.cn")
        .or_else(|| rest.strip_suffix(".amazonaws.com"))?;
    is_region_name(region).then(|| region.to_owned())
}

/// Region embedded in an inference profile ARN (`arn:aws[-partition]:bedrock:<region>:...`).
fn arn_region(model_id: &str) -> Option<String> {
    let mut parts = model_id.splitn(5, ':');
    let (scheme, partition, service, region) = (parts.next()?, parts.next()?, parts.next()?, parts.next()?);
    parts.next()?;
    let partition_ok = partition == "aws" || partition.strip_prefix("aws-").is_some_and(is_region_name);
    (scheme == "arn" && partition_ok && service == "bedrock" && is_region_name(region)).then(|| region.to_owned())
}

/// Pin the model's base URL as the endpoint unless it is a standard AWS endpoint and a region or
/// ambient `AWS_PROFILE` is configured. Custom endpoints (VPC, proxies) always win.
fn should_use_explicit_endpoint(base_url: &str, configured_region: Option<&str>, ambient_profile: bool) -> bool {
    if standard_endpoint_region(base_url).is_none() {
        return true;
    }
    configured_region.is_none() && !ambient_profile
}

/// Resolve region, endpoint, profile and auth exactly as upstream configures the SDK client.
fn resolve_client_plan(
    model: &Model,
    bedrock: &BedrockOptions,
    api_key: Option<&str>,
    env: &EnvLookup<'_>,
) -> ClientPlan {
    // A profile configured through pi's auth flow (the `profile` option or a scoped
    // `AWS_PROFILE`) wins over ambient access keys (upstream #6957).
    let options_profile = non_empty(bedrock.profile.as_deref()).or_else(|| env.scoped("AWS_PROFILE"));
    let mut plan = ClientPlan {
        profile: options_profile.clone().or_else(|| env.get("AWS_PROFILE")),
        ..ClientPlan::default()
    };
    let region = configured_region(bedrock, env);
    let ambient_profile = env.process("AWS_PROFILE").is_some();
    let endpoint_region = standard_endpoint_region(&model.base_url);
    let explicit_endpoint = should_use_explicit_endpoint(&model.base_url, region.as_deref(), ambient_profile);
    if explicit_endpoint && !model.base_url.is_empty() {
        plan.endpoint = Some(model.base_url.clone());
    }

    plan.skip_auth = env.get("AWS_BEDROCK_SKIP_AUTH").as_deref() == Some("1");
    let bearer_token = non_empty(bedrock.bearer_token.as_deref())
        .or_else(|| non_empty(api_key))
        .or_else(|| env.get("AWS_BEARER_TOKEN_BEDROCK"));

    // Region: ARN-embedded > explicit option > env vars > standard endpoint > us-east-1, unless an
    // ambient profile supplies it.
    plan.region = if let Some(region) = arn_region(&model.id) {
        Some(region)
    } else if region.is_some() {
        region
    } else if let Some(endpoint_region) = endpoint_region.filter(|_| explicit_endpoint) {
        Some(endpoint_region)
    } else if ambient_profile {
        None
    } else {
        Some("us-east-1".to_owned())
    };

    if plan.skip_auth {
        // Proxies that need no authentication.
        plan.credentials = Some(StaticCredentials {
            access_key_id: "dummy-access-key".into(),
            secret_access_key: "dummy-secret-key".into(),
            session_token: None,
        });
    } else {
        if options_profile.is_none() {
            plan.credentials = configured_credentials(env);
        }
        plan.bearer_token = bearer_token;
    }
    plan
}

/// Build an auth scheme preference without naming `AuthSchemeId`, which the SDK does not
/// re-export: the element type is inferred from `AuthSchemePreference: From<[AuthSchemeId; N]>`.
fn scheme_preference<T: From<&'static str>>(scheme: &'static str) -> [T; 1] {
    [T::from(scheme)]
}

async fn build_client(plan: &ClientPlan, options: &StreamOptions) -> Client {
    // Not ported: upstream's Node request handlers for proxies from `options.env` and
    // `AWS_BEDROCK_FORCE_HTTP1`. The SDK's default HTTPS client reads HTTP(S)_PROXY/NO_PROXY from
    // the process environment and negotiates the HTTP version itself.
    let mut loader = aws_config::defaults(BehaviorVersion::latest());
    if let Some(profile) = &plan.profile {
        loader = loader.profile_name(profile);
    }
    if let Some(region) = &plan.region {
        loader = loader.region(Region::new(region.clone()));
    }
    if let Some(credentials) = &plan.credentials {
        loader = loader.credentials_provider(Credentials::new(
            credentials.access_key_id.clone(),
            credentials.secret_access_key.clone(),
            credentials.session_token.clone(),
            None,
            "delta-bedrock",
        ));
    } else if let Some(profile) = &plan.profile {
        // The JS SDK skips environment credentials when a profile is configured; the Rust default
        // chain does not, so the profile provider is used directly.
        loader = loader.credentials_provider(ProfileFileCredentialsProvider::builder().profile_name(profile).build());
    }
    let shared = loader.load().await;

    let max_retries = options.max_retries.unwrap_or(DEFAULT_MAX_RETRIES);
    let timeout = options.timeout.unwrap_or(DEFAULT_TIMEOUT);
    let mut config = aws_sdk_bedrockruntime::config::Builder::from(&shared)
        .retry_config(RetryConfig::standard().with_max_attempts(max_retries.saturating_add(1)))
        .timeout_config(TimeoutConfig::builder().operation_timeout(timeout).build())
        // The event stream is guarded by the idle timeout in `drive` instead; long reasoning
        // pauses must not trip the SDK's throughput check.
        .stalled_stream_protection(StalledStreamProtectionConfig::disabled());
    if let Some(endpoint) = &plan.endpoint {
        config = config.endpoint_url(endpoint);
    }
    if let Some(token) = &plan.bearer_token {
        config = config
            .bearer_token(Token::new(token.clone(), None))
            .auth_scheme_preference(scheme_preference("httpBearerAuth"));
    } else if plan.skip_auth {
        // The SDK picks up AWS_BEARER_TOKEN_BEDROCK on its own; skip-auth proxies want SigV4.
        config = config.auth_scheme_preference(scheme_preference("sigv4"));
    }
    Client::from_conf(config.build())
}

// ---------------------------------------------------------------------------------------------
// Model capabilities
// ---------------------------------------------------------------------------------------------

/// Model id and name variants; names cover application inference profile ARNs.
fn model_match_candidates(model: &Model) -> Vec<String> {
    let mut values = vec![model.id.as_str()];
    if !model.name.is_empty() {
        values.push(model.name.as_str());
    }
    values
        .into_iter()
        .flat_map(|value| {
            let lower = value.to_lowercase();
            let mut normalized = String::with_capacity(lower.len());
            let mut in_separator = false;
            for ch in lower.chars() {
                if ch.is_whitespace() || matches!(ch, '_' | '.' | ':') {
                    if !in_separator {
                        normalized.push('-');
                    }
                    in_separator = true;
                } else {
                    normalized.push(ch);
                    in_separator = false;
                }
            }
            [lower, normalized]
        })
        .collect()
}

fn matches_any(model: &Model, needles: &[&str]) -> bool {
    model_match_candidates(model)
        .iter()
        .any(|candidate| needles.iter().any(|needle| candidate.contains(needle)))
}

/// Opus 4.6+, Sonnet 4.6 and Claude 5 models.
fn supports_adaptive_thinking(model: &Model) -> bool {
    matches_any(
        model,
        &[
            "opus-4-6",
            "opus-4-7",
            "opus-4-8",
            "opus-5",
            "sonnet-4-6",
            "sonnet-5",
            "haiku-5",
            "fable-5",
        ],
    )
}

const XHIGH_FAMILIES: [&str; 6] = ["opus-4-7", "opus-4-8", "opus-5", "sonnet-5", "haiku-5", "fable-5"];

fn supports_native_xhigh_effort(model: &Model) -> bool {
    matches_any(model, &XHIGH_FAMILIES)
}

/// Opus 4.6 and Sonnet 4.6 reject `thinking.block_binding` ("Extra inputs are not permitted").
fn supports_thinking_block_binding(model: &Model) -> bool {
    matches_any(model, &XHIGH_FAMILIES)
}

fn is_anthropic_claude_model(model: &Model) -> bool {
    let id = model.id.to_lowercase();
    let name = model.name.to_lowercase();
    id.contains("anthropic.claude")
        || id.contains("anthropic/claude")
        || name.contains("anthropic.claude")
        || name.contains("anthropic/claude")
        || name.contains("claude")
}

/// Claude 3.5 Haiku, 3.7 Sonnet, 4.x and 5 models. Application inference profiles may only
/// name the model in `model.name`; as a last resort `AWS_BEDROCK_FORCE_CACHE=1` enables cache
/// points. Amazon Nova caches automatically.
fn supports_prompt_caching(model: &Model, env: &EnvLookup<'_>) -> bool {
    if !matches_any(model, &["claude"]) {
        return env.get("AWS_BEDROCK_FORCE_CACHE").as_deref() == Some("1");
    }
    matches_any(
        model,
        &[
            "fable-5",
            "opus-5",
            "sonnet-5",
            "haiku-5",
            "-4-",
            "claude-3-7-sonnet",
            "claude-3-5-haiku",
        ],
    )
}

/// Only Anthropic models accept `reasoningContent.reasoningText.signature`.
fn supports_thinking_signature(model: &Model) -> bool {
    is_anthropic_claude_model(model)
}

fn is_gov_cloud_target(model: &Model, region: Option<&str>) -> bool {
    if region.is_some_and(|region| region.to_lowercase().starts_with("us-gov-")) {
        return true;
    }
    let id = model.id.to_lowercase();
    id.starts_with("us-gov.") || id.starts_with("arn:aws-us-gov:")
}

fn supports_strict_mode(model: &Model) -> bool {
    match &model.compat {
        Some(ModelCompat::Bedrock(compat)) => compat.supports_strict_mode.unwrap_or(false),
        Some(ModelCompat::Other(extra)) => extra
            .get("supportsStrictMode")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        _ => false,
    }
}

fn mapped_level(model: &Model, level: ThinkingLevel) -> Option<String> {
    model
        .mapped_thinking_level(ModelThinkingLevel::from(level))
        .flatten()
        .map(str::to_owned)
}

fn map_thinking_level_to_effort(model: &Model, level: ThinkingLevel) -> String {
    if level == ThinkingLevel::Xhigh && supports_native_xhigh_effort(model) {
        return "xhigh".into();
    }
    if let Some(mapped) = mapped_level(model, level) {
        return mapped;
    }
    match level {
        ThinkingLevel::Minimal | ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        _ => "high",
    }
    .into()
}

/// Default budgets for budget-based Claude thinking; extended levels clamp to `high`.
fn default_thinking_budget(level: ThinkingLevel) -> u64 {
    match level {
        ThinkingLevel::Minimal => 1024,
        ThinkingLevel::Low => 2048,
        ThinkingLevel::Medium => 8192,
        _ => 16384,
    }
}

fn custom_thinking_budget(budgets: Option<&ThinkingBudgets>, level: ThinkingLevel) -> Option<u64> {
    let budgets = budgets?;
    match level {
        ThinkingLevel::Minimal => budgets.minimal,
        ThinkingLevel::Low => budgets.low,
        ThinkingLevel::Medium => budgets.medium,
        _ => budgets.high,
    }
}

/// OpenAI GPT models (GPT-5.x, GPT-6) take a nested `reasoning.effort` and reject `minimal`.
fn openai_gpt_effort(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Minimal | ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::Xhigh => "xhigh",
        ThinkingLevel::Max => "max",
    }
}

/// gpt-oss takes a flat `reasoning_effort` and only accepts low, medium and high.
fn openai_gpt_oss_effort(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Minimal | ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        _ => "high",
    }
}

fn build_additional_model_request_fields(
    model: &Model,
    bedrock: &BedrockOptions,
    region: Option<&str>,
) -> Option<Value> {
    let reasoning = bedrock.reasoning?;
    if !model.reasoning {
        return None;
    }

    if is_anthropic_claude_model(model) {
        // GovCloud Bedrock rejects the Claude thinking.display field.
        let gov_cloud = is_gov_cloud_target(model, region);
        let display = (!gov_cloud).then_some(match bedrock.thinking_display {
            Some(ThinkingDisplay::Omitted) => "omitted",
            _ => "summarized",
        });
        // Replayed signed thinking blocks are bound to the system prompt and tools they were
        // created with; Bedrock 400s on replay after either changes unless stale blocks are
        // dropped. Skipped on GovCloud like display.
        let block_binding = !gov_cloud && supports_thinking_block_binding(model);
        let adaptive = supports_adaptive_thinking(model);
        let mut result = Map::new();
        if adaptive {
            let mut thinking = Map::new();
            thinking.insert("type".into(), json!("adaptive"));
            if let Some(display) = display {
                thinking.insert("display".into(), json!(display));
            }
            if block_binding {
                thinking.insert(
                    "block_binding".into(),
                    json!({"prefix_mismatch_behavior": "drop_block"}),
                );
            }
            result.insert("thinking".into(), Value::Object(thinking));
            result.insert(
                "output_config".into(),
                json!({"effort": map_thinking_level_to_effort(model, reasoning)}),
            );
            if block_binding {
                result.insert("anthropic_beta".into(), json!([THINKING_BINDING_CONTROLS_BETA]));
            }
        } else {
            // Custom budgets only cover token-based levels through high.
            let budget = custom_thinking_budget(bedrock.thinking_budgets.as_ref(), clamp_reasoning(reasoning))
                .unwrap_or_else(|| default_thinking_budget(reasoning));
            let mut thinking = Map::new();
            thinking.insert("type".into(), json!("enabled"));
            thinking.insert("budget_tokens".into(), json!(budget));
            if let Some(display) = display {
                thinking.insert("display".into(), json!(display));
            }
            result.insert("thinking".into(), Value::Object(thinking));
        }
        if !adaptive && bedrock.interleaved_thinking.unwrap_or(true) {
            result.insert("anthropic_beta".into(), json!([INTERLEAVED_THINKING_BETA]));
        }
        return Some(Value::Object(result));
    }

    if matches_any(model, &["gpt-oss"]) {
        return Some(json!({"reasoning_effort": openai_gpt_oss_effort(reasoning)}));
    }
    if matches_any(model, &["gpt-"]) {
        let effort = mapped_level(model, reasoning).unwrap_or_else(|| openai_gpt_effort(reasoning).to_owned());
        return Some(json!({"reasoning": {"effort": effort}}));
    }
    None
}

// ---------------------------------------------------------------------------------------------
// Request payload (upstream `ConverseStreamCommand` input as JSON)
// ---------------------------------------------------------------------------------------------

/// Defaults to short; `PI_CACHE_RETENTION=long` is honored for backward compatibility.
fn resolve_cache_retention(options: &StreamOptions, env: &EnvLookup<'_>) -> CacheRetention {
    if let Some(retention) = options.cache_retention {
        return retention;
    }
    if env.get("PI_CACHE_RETENTION").as_deref() == Some("long") {
        return CacheRetention::Long;
    }
    CacheRetention::Short
}

fn cache_point(retention: CacheRetention) -> Value {
    if retention == CacheRetention::Long {
        json!({"cachePoint": {"type": "default", "ttl": "1h"}})
    } else {
        json!({"cachePoint": {"type": "default"}})
    }
}

fn build_system_prompt(
    prompt: Option<&str>,
    model: &Model,
    retention: CacheRetention,
    env: &EnvLookup<'_>,
) -> Option<Value> {
    let prompt = prompt.filter(|prompt| !prompt.is_empty())?;
    let mut blocks = vec![json!({"text": prompt})];
    if retention != CacheRetention::None && supports_prompt_caching(model, env) {
        blocks.push(cache_point(retention));
    }
    Some(Value::Array(blocks))
}

/// Bedrock tool use ids allow `[a-zA-Z0-9_-]{1,64}`. Characters are counted in UTF-16 code units
/// as upstream does.
fn normalize_tool_call_id(id: &str) -> String {
    let mut sanitized = String::with_capacity(id.len());
    for ch in id.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            sanitized.push(ch);
        } else {
            for _ in 0..ch.len_utf16() {
                sanitized.push('_');
            }
        }
    }
    sanitized.truncate(64);
    sanitized
}

fn non_blank_text_block(text: &str) -> Option<Value> {
    (!text.trim().is_empty()).then(|| json!({"text": text}))
}

fn required_text_block(text: &str) -> Value {
    non_blank_text_block(text).unwrap_or_else(|| json!({"text": EMPTY_TEXT_PLACEHOLDER}))
}

/// Copy a JSON document without empty property names, which Bedrock rejects.
fn sanitize_bedrock_document(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(sanitize_bedrock_document).collect()),
        Value::Object(object) => Value::Object(
            object
                .iter()
                .filter(|(key, _)| !key.is_empty())
                .map(|(key, value)| (key.clone(), sanitize_bedrock_document(value)))
                .collect(),
        ),
        other => other.clone(),
    }
}

const FORGIVING_BASE64: GeneralPurpose = GeneralPurpose::new(
    &base64::alphabet::STANDARD,
    GeneralPurposeConfig::new()
        .with_decode_padding_mode(DecodePaddingMode::Indifferent)
        .with_decode_allow_trailing_bits(true),
);

/// Base64 decode like the browser `atob`: ASCII whitespace and missing padding are tolerated.
fn decode_base64_forgiving(data: &str) -> Option<Vec<u8>> {
    let compact: String = data
        .chars()
        .filter(|ch| !matches!(ch, '\t' | '\n' | '\u{0c}' | '\r' | ' '))
        .collect();
    FORGIVING_BASE64.decode(compact).ok()
}

fn image_format(mime_type: &str) -> Result<&'static str> {
    match mime_type {
        "image/jpeg" | "image/jpg" => Ok("jpeg"),
        "image/png" => Ok("png"),
        "image/gif" => Ok("gif"),
        "image/webp" => Ok("webp"),
        other => Err(Error::msg(format!("Unknown image type: {other}"))),
    }
}

fn image_block(mime_type: &str, data: &str) -> Result<Value> {
    let format = image_format(mime_type)?;
    let bytes =
        decode_base64_forgiving(data).ok_or_else(|| Error::msg("Failed to decode image data: invalid base64"))?;
    Ok(json!({"image": {"source": {"bytes": STANDARD.encode(bytes)}, "format": format}}))
}

fn tool_result_content(content: &[UserBlock]) -> Result<Vec<Value>> {
    let mut result = Vec::new();
    for block in content {
        match block {
            UserBlock::Image(image) => result.push(image_block(&image.mime_type, &image.data)?),
            UserBlock::Text(text) => result.extend(non_blank_text_block(&text.text)),
        }
    }
    if result.is_empty() {
        result.push(json!({"text": EMPTY_TEXT_PLACEHOLDER}));
    }
    Ok(result)
}

fn convert_assistant_content(content: &[AssistantBlock], model: &Model) -> Vec<Value> {
    let mut blocks = Vec::new();
    for block in content {
        match block {
            AssistantBlock::Text(text) => blocks.extend(non_blank_text_block(&text.text)),
            AssistantBlock::ToolCall(call) => blocks.push(json!({"toolUse": {
                "toolUseId": call.id,
                "name": call.name,
                "input": sanitize_bedrock_document(&Value::Object(call.arguments.clone())),
            }})),
            AssistantBlock::Thinking(thinking) => {
                // Encrypted reasoning is opaque: replay the stored payload as `redactedContent`.
                // A signature that is not base64 drops the block instead of failing the request.
                if thinking.is_redacted() {
                    if let Some(bytes) = thinking
                        .thinking_signature
                        .as_deref()
                        .filter(|signature| !signature.is_empty())
                        .and_then(decode_base64_forgiving)
                        .filter(|bytes| !bytes.is_empty())
                    {
                        blocks.push(json!({"reasoningContent": {"redactedContent": STANDARD.encode(bytes)}}));
                    }
                    continue;
                }
                if thinking.thinking.trim().is_empty() {
                    continue;
                }
                if supports_thinking_signature(model) {
                    // A replayed reasoning block without a signature is rejected; fall back to
                    // plain text, matching Anthropic.
                    match thinking
                        .thinking_signature
                        .as_deref()
                        .filter(|signature| !signature.trim().is_empty())
                    {
                        Some(signature) => blocks.push(json!({"reasoningContent": {"reasoningText": {
                            "text": thinking.thinking,
                            "signature": signature,
                        }}})),
                        None => blocks.push(json!({"text": thinking.thinking})),
                    }
                } else {
                    // Other models reject the signature field.
                    blocks.push(json!({"reasoningContent": {"reasoningText": {"text": thinking.thinking}}}));
                }
            }
        }
    }
    blocks
}

fn convert_messages(
    context: &TranscriptContext,
    model: &Model,
    retention: CacheRetention,
    env: &EnvLookup<'_>,
) -> Result<Vec<Value>> {
    let normalize = |id: &str, _: &Model, _: &crate::types::AssistantMessage| normalize_tool_call_id(id);
    let messages = transform_messages(
        without_initial_system_message(context.messages()),
        model,
        Some(&normalize),
    );
    let claude = is_anthropic_claude_model(model);
    let mut result: Vec<Value> = Vec::new();
    let mut index = 0;
    while index < messages.len() {
        match &messages[index] {
            Message::User(user) => {
                let mut content = Vec::new();
                match &user.content {
                    UserContent::Text(text) => content.push(required_text_block(text)),
                    UserContent::Blocks(blocks) => {
                        for block in blocks {
                            match block {
                                UserBlock::Text(text) => content.extend(non_blank_text_block(&text.text)),
                                UserBlock::Image(image) => content.push(image_block(&image.mime_type, &image.data)?),
                            }
                        }
                        if content.is_empty() {
                            content.push(json!({"text": EMPTY_TEXT_PLACEHOLDER}));
                        }
                    }
                }
                result.push(json!({"role": "user", "content": content}));
            }
            Message::Assistant(assistant) => {
                // Bedrock rejects messages with empty content arrays (e.g. from aborted requests).
                if !assistant.content.is_empty() {
                    let blocks = convert_assistant_content(&assistant.content, model);
                    if !blocks.is_empty() {
                        result.push(json!({"role": "assistant", "content": blocks}));
                    }
                }
            }
            Message::ToolResult(_) => {
                // Bedrock requires all consecutive tool results in one user message. Claude
                // rejects error tool results with anything but text, so their images follow the
                // results of the turn (upstream #10794); other models keep them in place.
                let mut tool_results = Vec::new();
                let mut error_images = Vec::new();
                while let Some(Message::ToolResult(tool_message)) = messages.get(index) {
                    let split = if claude {
                        split_error_tool_result_images(tool_message)
                    } else {
                        None
                    };
                    let content = match &split {
                        Some(split) => {
                            let texts: Vec<UserBlock> = split.content.iter().cloned().map(UserBlock::Text).collect();
                            tool_result_content(&texts)?
                        }
                        None => tool_result_content(&tool_message.content)?,
                    };
                    tool_results.push(json!({"toolResult": {
                        "toolUseId": tool_message.tool_call_id,
                        "content": content,
                        "status": if tool_message.is_error { "error" } else { "success" },
                    }}));
                    if let Some(split) = split {
                        error_images.push(json!({"text": split.caption}));
                        for image in &split.images {
                            error_images.push(image_block(&image.mime_type, &image.data)?);
                        }
                    }
                    index += 1;
                }
                tool_results.extend(error_images);
                result.push(json!({"role": "user", "content": tool_results}));
                continue;
            }
            Message::System(_) => {}
        }
        index += 1;
    }

    // Cache point on the last user message for supported Claude models.
    if retention != CacheRetention::None
        && supports_prompt_caching(model, env)
        && let Some(last) = result.last_mut()
        && last["role"] == "user"
        && let Some(Value::Array(content)) = last.get_mut("content")
    {
        content.push(cache_point(retention));
    }
    Ok(result)
}

fn convert_tool_config(
    tools: &[Tool],
    tool_choice: Option<&AnthropicToolChoice>,
    supports_strict: bool,
) -> Result<Option<Value>> {
    if tools.is_empty() || tool_choice == Some(&AnthropicToolChoice::None) {
        return Ok(None);
    }
    let mut specs = Vec::with_capacity(tools.len());
    for tool in tools {
        let strict = resolve_json_schema_strict_sampling(tool, supports_strict, None)?;
        let schema = get_json_schema_tool_parameters(tool, strict)?;
        let mut spec = Map::new();
        spec.insert("name".into(), json!(tool.name));
        spec.insert("description".into(), json!(tool.description));
        spec.insert("inputSchema".into(), json!({"json": schema}));
        if strict == Some(true) {
            spec.insert("strict".into(), Value::Bool(true));
        }
        specs.push(json!({"toolSpec": spec}));
    }
    let mut config = Map::new();
    config.insert("tools".into(), Value::Array(specs));
    match tool_choice {
        Some(AnthropicToolChoice::Auto) => {
            config.insert("toolChoice".into(), json!({"auto": {}}));
        }
        Some(AnthropicToolChoice::Any) => {
            config.insert("toolChoice".into(), json!({"any": {}}));
        }
        Some(AnthropicToolChoice::Tool { name }) => {
            config.insert("toolChoice".into(), json!({"tool": {"name": name}}));
        }
        Some(AnthropicToolChoice::None) | None => {}
    }
    Ok(Some(Value::Object(config)))
}

fn build_payload(
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
    bedrock: &BedrockOptions,
    env: &EnvLookup<'_>,
) -> Result<Value> {
    let retention = resolve_cache_retention(options, env);
    let max_tokens = options
        .max_tokens
        .or_else(|| is_anthropic_claude_model(model).then_some(model.max_tokens));
    let system_prompt = get_initial_system_message(context.messages()).map(get_system_message_text);

    let mut payload = Map::new();
    payload.insert("modelId".into(), json!(model.id));
    payload.insert(
        "messages".into(),
        Value::Array(convert_messages(context, model, retention, env)?),
    );
    if let Some(system) = build_system_prompt(system_prompt.as_deref(), model, retention, env) {
        payload.insert("system".into(), system);
    }
    let mut inference = Map::new();
    if let Some(max_tokens) = max_tokens {
        inference.insert("maxTokens".into(), json!(max_tokens));
    }
    if let Some(temperature) = options.temperature {
        inference.insert("temperature".into(), json!(temperature));
    }
    payload.insert("inferenceConfig".into(), Value::Object(inference));
    if let Some(tool_config) = convert_tool_config(
        &get_current_tools(context.messages()),
        bedrock.tool_choice.as_ref(),
        supports_strict_mode(model),
    )? {
        payload.insert("toolConfig".into(), tool_config);
    }
    let region = configured_region(bedrock, env);
    if let Some(fields) = build_additional_model_request_fields(model, bedrock, region.as_deref()) {
        payload.insert("additionalModelRequestFields".into(), fields);
    }
    if let Some(metadata) = &bedrock.request_metadata {
        payload.insert("requestMetadata".into(), json!(metadata));
    }
    Ok(Value::Object(payload))
}

// ---------------------------------------------------------------------------------------------
// JSON payload -> SDK input
// ---------------------------------------------------------------------------------------------

/// The typed `ConverseStream` input.
#[derive(Debug, Default)]
struct ConverseRequest {
    model_id: String,
    messages: Vec<bedrock::Message>,
    system: Option<Vec<bedrock::SystemContentBlock>>,
    inference_config: Option<bedrock::InferenceConfiguration>,
    tool_config: Option<bedrock::ToolConfiguration>,
    additional_model_request_fields: Option<Document>,
    additional_model_response_field_paths: Option<Vec<String>>,
    request_metadata: Option<HashMap<String, String>>,
}

fn invalid(what: impl fmt::Display) -> Error {
    Error::msg(format!("Invalid Bedrock payload: {what}"))
}

fn built<T>(result: std::result::Result<T, aws_sdk_bedrockruntime::error::BuildError>) -> Result<T> {
    result.map_err(invalid)
}

fn as_object<'a>(value: &'a Value, what: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| invalid(format!("{what} must be an object")))
}

fn as_array<'a>(value: &'a Value, what: &str) -> Result<&'a Vec<Value>> {
    value
        .as_array()
        .ok_or_else(|| invalid(format!("{what} must be an array")))
}

fn str_field(object: &Map<String, Value>, key: &str, what: &str) -> Result<String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| invalid(format!("{what}.{key} must be a string")))
}

fn opt_str_field(object: &Map<String, Value>, key: &str) -> Option<String> {
    object.get(key).and_then(Value::as_str).map(str::to_owned)
}

/// The single member of a tagged union object, e.g. `{"text": "..."}`.
fn union_member<'a>(value: &'a Value, what: &str) -> Result<(&'a str, &'a Value)> {
    let object = as_object(value, what)?;
    let mut members = object.iter().filter(|(_, value)| !value.is_null());
    match (members.next(), members.next()) {
        (Some((key, value)), None) => Ok((key.as_str(), value)),
        _ => Err(invalid(format!("{what} must have exactly one member"))),
    }
}

fn blob_field(value: &Value, what: &str) -> Result<Blob> {
    let text = value
        .as_str()
        .ok_or_else(|| invalid(format!("{what} must be a base64 string")))?;
    decode_base64_forgiving(text)
        .map(Blob::new)
        .ok_or_else(|| invalid(format!("{what} is not valid base64")))
}

/// JSON to a Smithy document. Objects lose their key order (the SDK stores them in a `HashMap`).
fn json_to_document(value: &Value) -> Document {
    match value {
        Value::Null => Document::Null,
        Value::Bool(flag) => Document::Bool(*flag),
        Value::Number(number) => {
            if let Some(unsigned) = number.as_u64() {
                Document::Number(Number::PosInt(unsigned))
            } else if let Some(signed) = number.as_i64() {
                Document::Number(Number::NegInt(signed))
            } else {
                Document::Number(Number::Float(number.as_f64().unwrap_or(0.0)))
            }
        }
        Value::String(text) => Document::String(text.clone()),
        Value::Array(items) => Document::Array(items.iter().map(json_to_document).collect()),
        Value::Object(object) => Document::Object(
            object
                .iter()
                .map(|(key, value)| (key.clone(), json_to_document(value)))
                .collect(),
        ),
    }
}

/// A Smithy document as JSON. Object keys are sorted for a stable shape.
fn document_to_json(document: &Document) -> Value {
    match document {
        Document::Null => Value::Null,
        Document::Bool(flag) => Value::Bool(*flag),
        Document::Number(Number::PosInt(value)) => json!(value),
        Document::Number(Number::NegInt(value)) => json!(value),
        Document::Number(Number::Float(value)) => json!(value),
        Document::String(text) => Value::String(text.clone()),
        Document::Array(items) => Value::Array(items.iter().map(document_to_json).collect()),
        Document::Object(object) => {
            let mut entries: Vec<_> = object.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key.clone(), document_to_json(value)))
                    .collect(),
            )
        }
    }
}

fn to_i32(value: &Value, what: &str) -> Result<i32> {
    value
        .as_i64()
        .and_then(|number| i32::try_from(number).ok())
        .ok_or_else(|| invalid(format!("{what} must be a 32-bit integer")))
}

fn to_f32(value: &Value, what: &str) -> Result<f32> {
    value
        .as_f64()
        .map(|number| number as f32)
        .ok_or_else(|| invalid(format!("{what} must be a number")))
}

fn image_from_json(value: &Value) -> Result<bedrock::ImageBlock> {
    let object = as_object(value, "image")?;
    let format = str_field(object, "format", "image")?;
    let source = as_object(
        object.get("source").ok_or_else(|| invalid("image.source is missing"))?,
        "image.source",
    )?;
    let bytes = blob_field(
        source
            .get("bytes")
            .ok_or_else(|| invalid("image.source.bytes is missing"))?,
        "image.source.bytes",
    )?;
    built(
        bedrock::ImageBlock::builder()
            .format(bedrock::ImageFormat::from(format.as_str()))
            .source(bedrock::ImageSource::Bytes(bytes))
            .build(),
    )
}

fn cache_point_from_json(value: &Value) -> Result<bedrock::CachePointBlock> {
    let object = as_object(value, "cachePoint")?;
    let kind = opt_str_field(object, "type").unwrap_or_else(|| "default".into());
    built(
        bedrock::CachePointBlock::builder()
            .r#type(bedrock::CachePointType::from(kind.as_str()))
            .set_ttl(opt_str_field(object, "ttl").map(|ttl| bedrock::CacheTtl::from(ttl.as_str())))
            .build(),
    )
}

fn tool_result_block_from_json(value: &Value) -> Result<bedrock::ToolResultContentBlock> {
    let (member, inner) = union_member(value, "toolResult.content[]")?;
    Ok(match member {
        "text" => bedrock::ToolResultContentBlock::Text(
            inner
                .as_str()
                .ok_or_else(|| invalid("toolResult text must be a string"))?
                .to_owned(),
        ),
        "image" => bedrock::ToolResultContentBlock::Image(image_from_json(inner)?),
        "json" => bedrock::ToolResultContentBlock::Json(json_to_document(inner)),
        other => return Err(invalid(format!("unsupported tool result block {other:?}"))),
    })
}

fn reasoning_from_json(value: &Value) -> Result<bedrock::ReasoningContentBlock> {
    let (member, inner) = union_member(value, "reasoningContent")?;
    Ok(match member {
        "reasoningText" => {
            let object = as_object(inner, "reasoningContent.reasoningText")?;
            bedrock::ReasoningContentBlock::ReasoningText(built(
                bedrock::ReasoningTextBlock::builder()
                    .text(str_field(object, "text", "reasoningText")?)
                    .set_signature(opt_str_field(object, "signature"))
                    .build(),
            )?)
        }
        "redactedContent" => {
            bedrock::ReasoningContentBlock::RedactedContent(blob_field(inner, "reasoningContent.redactedContent")?)
        }
        other => return Err(invalid(format!("unsupported reasoning block {other:?}"))),
    })
}

fn content_block_from_json(value: &Value) -> Result<bedrock::ContentBlock> {
    let (member, inner) = union_member(value, "content block")?;
    Ok(match member {
        "text" => bedrock::ContentBlock::Text(
            inner
                .as_str()
                .ok_or_else(|| invalid("text must be a string"))?
                .to_owned(),
        ),
        "image" => bedrock::ContentBlock::Image(image_from_json(inner)?),
        "cachePoint" => bedrock::ContentBlock::CachePoint(cache_point_from_json(inner)?),
        "reasoningContent" => bedrock::ContentBlock::ReasoningContent(reasoning_from_json(inner)?),
        "toolUse" => {
            let object = as_object(inner, "toolUse")?;
            bedrock::ContentBlock::ToolUse(built(
                bedrock::ToolUseBlock::builder()
                    .tool_use_id(str_field(object, "toolUseId", "toolUse")?)
                    .name(str_field(object, "name", "toolUse")?)
                    .input(json_to_document(object.get("input").unwrap_or(&Value::Null)))
                    .build(),
            )?)
        }
        "toolResult" => {
            let object = as_object(inner, "toolResult")?;
            let content = match object.get("content") {
                Some(content) => as_array(content, "toolResult.content")?
                    .iter()
                    .map(tool_result_block_from_json)
                    .collect::<Result<Vec<_>>>()?,
                None => Vec::new(),
            };
            bedrock::ContentBlock::ToolResult(built(
                bedrock::ToolResultBlock::builder()
                    .tool_use_id(str_field(object, "toolUseId", "toolResult")?)
                    .set_content(Some(content))
                    .set_status(
                        opt_str_field(object, "status").map(|status| bedrock::ToolResultStatus::from(status.as_str())),
                    )
                    .build(),
            )?)
        }
        other => return Err(invalid(format!("unsupported content block {other:?}"))),
    })
}

fn message_from_json(value: &Value) -> Result<bedrock::Message> {
    let object = as_object(value, "message")?;
    let role = str_field(object, "role", "message")?;
    let content = as_array(
        object
            .get("content")
            .ok_or_else(|| invalid("message.content is missing"))?,
        "message.content",
    )?
    .iter()
    .map(content_block_from_json)
    .collect::<Result<Vec<_>>>()?;
    built(
        bedrock::Message::builder()
            .role(bedrock::ConversationRole::from(role.as_str()))
            .set_content(Some(content))
            .build(),
    )
}

fn system_block_from_json(value: &Value) -> Result<bedrock::SystemContentBlock> {
    let (member, inner) = union_member(value, "system block")?;
    Ok(match member {
        "text" => bedrock::SystemContentBlock::Text(
            inner
                .as_str()
                .ok_or_else(|| invalid("system text must be a string"))?
                .to_owned(),
        ),
        "cachePoint" => bedrock::SystemContentBlock::CachePoint(cache_point_from_json(inner)?),
        other => return Err(invalid(format!("unsupported system block {other:?}"))),
    })
}

fn inference_config_from_json(value: &Value) -> Result<bedrock::InferenceConfiguration> {
    let object = as_object(value, "inferenceConfig")?;
    let mut builder = bedrock::InferenceConfiguration::builder();
    if let Some(max_tokens) = object.get("maxTokens").filter(|value| !value.is_null()) {
        builder = builder.max_tokens(to_i32(max_tokens, "inferenceConfig.maxTokens")?);
    }
    if let Some(temperature) = object.get("temperature").filter(|value| !value.is_null()) {
        builder = builder.temperature(to_f32(temperature, "inferenceConfig.temperature")?);
    }
    if let Some(top_p) = object.get("topP").filter(|value| !value.is_null()) {
        builder = builder.top_p(to_f32(top_p, "inferenceConfig.topP")?);
    }
    if let Some(stops) = object.get("stopSequences").filter(|value| !value.is_null()) {
        let stops = as_array(stops, "inferenceConfig.stopSequences")?
            .iter()
            .map(|stop| {
                stop.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| invalid("stop sequences must be strings"))
            })
            .collect::<Result<Vec<_>>>()?;
        builder = builder.set_stop_sequences(Some(stops));
    }
    Ok(builder.build())
}

fn tool_from_json(value: &Value) -> Result<bedrock::Tool> {
    let (member, inner) = union_member(value, "tool")?;
    match member {
        "toolSpec" => {
            let object = as_object(inner, "toolSpec")?;
            let schema = object
                .get("inputSchema")
                .and_then(|schema| schema.get("json"))
                .ok_or_else(|| invalid("toolSpec.inputSchema.json is missing"))?;
            Ok(bedrock::Tool::ToolSpec(built(
                bedrock::ToolSpecification::builder()
                    .name(str_field(object, "name", "toolSpec")?)
                    .set_description(opt_str_field(object, "description"))
                    .input_schema(bedrock::ToolInputSchema::Json(json_to_document(schema)))
                    .set_strict(object.get("strict").and_then(Value::as_bool))
                    .build(),
            )?))
        }
        "cachePoint" => Ok(bedrock::Tool::CachePoint(cache_point_from_json(inner)?)),
        other => Err(invalid(format!("unsupported tool {other:?}"))),
    }
}

fn tool_config_from_json(value: &Value) -> Result<bedrock::ToolConfiguration> {
    let object = as_object(value, "toolConfig")?;
    let tools = as_array(
        object
            .get("tools")
            .ok_or_else(|| invalid("toolConfig.tools is missing"))?,
        "toolConfig.tools",
    )?
    .iter()
    .map(tool_from_json)
    .collect::<Result<Vec<_>>>()?;
    let tool_choice = match object.get("toolChoice").filter(|value| !value.is_null()) {
        None => None,
        Some(choice) => Some(match union_member(choice, "toolChoice")? {
            ("auto", _) => bedrock::ToolChoice::Auto(bedrock::AutoToolChoice::builder().build()),
            ("any", _) => bedrock::ToolChoice::Any(bedrock::AnyToolChoice::builder().build()),
            ("tool", inner) => bedrock::ToolChoice::Tool(built(
                bedrock::SpecificToolChoice::builder()
                    .name(str_field(
                        as_object(inner, "toolChoice.tool")?,
                        "name",
                        "toolChoice.tool",
                    )?)
                    .build(),
            )?),
            (other, _) => return Err(invalid(format!("unsupported tool choice {other:?}"))),
        }),
    };
    built(
        bedrock::ToolConfiguration::builder()
            .set_tools(Some(tools))
            .set_tool_choice(tool_choice)
            .build(),
    )
}

impl ConverseRequest {
    /// Convert the JSON payload. Members the SDK input does not model are ignored, as the JS
    /// SDK's command serializer does.
    fn from_payload(payload: &Value) -> Result<Self> {
        let object = as_object(payload, "payload")?;
        let present = |key: &str| object.get(key).filter(|value| !value.is_null());
        let mut request = Self {
            model_id: str_field(object, "modelId", "payload")?,
            ..Self::default()
        };
        if let Some(messages) = present("messages") {
            request.messages = as_array(messages, "messages")?
                .iter()
                .map(message_from_json)
                .collect::<Result<_>>()?;
        }
        if let Some(system) = present("system") {
            request.system = Some(
                as_array(system, "system")?
                    .iter()
                    .map(system_block_from_json)
                    .collect::<Result<_>>()?,
            );
        }
        if let Some(inference) = present("inferenceConfig") {
            request.inference_config = Some(inference_config_from_json(inference)?);
        }
        if let Some(tool_config) = present("toolConfig") {
            request.tool_config = Some(tool_config_from_json(tool_config)?);
        }
        if let Some(fields) = present("additionalModelRequestFields") {
            request.additional_model_request_fields = Some(json_to_document(fields));
        }
        if let Some(paths) = present("additionalModelResponseFieldPaths") {
            request.additional_model_response_field_paths = Some(
                as_array(paths, "additionalModelResponseFieldPaths")?
                    .iter()
                    .filter_map(|path| path.as_str().map(str::to_owned))
                    .collect(),
            );
        }
        if let Some(metadata) = present("requestMetadata") {
            request.request_metadata = Some(
                as_object(metadata, "requestMetadata")?
                    .iter()
                    .map(|(key, value)| {
                        value
                            .as_str()
                            .map(|value| (key.clone(), value.to_owned()))
                            .ok_or_else(|| invalid("requestMetadata values must be strings"))
                    })
                    .collect::<Result<_>>()?,
            );
        }
        Ok(request)
    }
}

// ---------------------------------------------------------------------------------------------
// Custom headers and response capture
// ---------------------------------------------------------------------------------------------

/// `host` and `x-amz-*` take part in the SigV4 canonical request; `authorization` belongs to
/// SigV4 or the bearer token. Compared case-insensitively.
fn is_reserved_header(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.starts_with("x-amz-") || lower == "authorization" || lower == "host"
}

/// Caller headers to set on the request (merged case-insensitively, `None` deletes), without
/// reserved headers. `None` when nothing is left to set.
fn custom_headers(headers: Option<&ProviderHeaders>) -> Result<Option<Vec<(String, String)>>> {
    let mut merged: IndexMap<String, (String, String)> = IndexMap::new();
    for (name, value) in headers.into_iter().flatten() {
        let key = name.to_ascii_lowercase();
        merged.shift_remove(&key);
        if let Some(value) = value {
            merged.insert(key, (name.clone(), value.clone()));
        }
    }
    let mut result = Vec::new();
    for (name, value) in merged.into_values() {
        if is_reserved_header(&name) {
            continue;
        }
        http::HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| Error::msg(format!("Invalid header name: {name}")))?;
        http::HeaderValue::from_str(&value).map_err(|_| Error::msg(format!("Invalid value for header {name}")))?;
        result.push((name, value));
    }
    Ok((!result.is_empty()).then_some(result))
}

/// Records the status and headers of each received HTTP response before deserialization, so
/// `on_response` sees gateway headers the SDK output does not model.
#[derive(Clone, Debug, Default)]
struct ResponseCapture(Arc<Mutex<Option<ProviderResponse>>>);

impl ResponseCapture {
    fn take(&self) -> Option<ProviderResponse> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).take()
    }
}

impl Intercept for ResponseCapture {
    fn name(&self) -> &'static str {
        "DeltaBedrockResponseCapture"
    }

    fn read_before_deserialization(
        &self,
        context: &BeforeDeserializationInterceptorContextRef<'_>,
        _runtime_components: &RuntimeComponents,
        _cfg: &mut ConfigBag,
    ) -> std::result::Result<(), BoxError> {
        let response = context.response();
        let captured = ProviderResponse {
            status: response.status().as_u16(),
            headers: response
                .headers()
                .iter()
                .map(|(name, value)| (name.to_ascii_lowercase(), value.to_owned()))
                .collect(),
        };
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(captured);
        Ok(())
    }
}

fn normalize_diagnostic_value(value: Option<&str>) -> Option<String> {
    let trimmed = value?.trim();
    (!trimmed.is_empty() && trimmed.chars().count() <= MAX_DIAGNOSTIC_VALUE_CHARS).then(|| trimmed.to_owned())
}

fn request_id_header<'a>(get: impl Fn(&str) -> Option<&'a str>) -> Option<String> {
    ["x-amzn-requestid", "x-amzn-request-id", "x-amz-request-id"]
        .into_iter()
        .find_map(|name| normalize_diagnostic_value(get(name)))
}

// ---------------------------------------------------------------------------------------------
// Failures
// ---------------------------------------------------------------------------------------------

/// A failed request, with the metadata recorded in the `bedrock_response_failure` diagnostic.
#[derive(Debug, Default, PartialEq)]
struct FailureInfo {
    /// Message without the exception prefix.
    core: String,
    /// Modeled Bedrock exception name (or the SDK's `Unknown`); adds the human-readable prefix.
    service_name: Option<String>,
    status: Option<u16>,
    error_code: Option<String>,
    request_id: Option<String>,
}

impl FailureInfo {
    fn plain(message: impl Into<String>) -> Self {
        Self {
            core: message.into(),
            ..Self::default()
        }
    }

    /// The error message: `<prefix>: <core>` for Bedrock exceptions, plus a hint for data
    /// retention mode errors.
    fn message(&self) -> String {
        let hint = if self.core.to_lowercase().contains("data retention mode") {
            format!(" See {DATA_RETENTION_DOCS_URL} for supported data retention modes.")
        } else {
            String::new()
        };
        match &self.service_name {
            Some(name) => {
                let prefix = ERROR_PREFIXES
                    .iter()
                    .find(|(exception, _)| exception == name)
                    .map_or(name.as_str(), |(_, prefix)| prefix);
                format!("{prefix}: {}{hint}", self.core)
            }
            None => format!("{}{hint}", self.core),
        }
    }

    /// Diagnostic details; unknown fields are omitted, never guessed.
    fn diagnostic_details(&self, fallback_request_id: Option<&str>) -> Option<JsonObject> {
        let mut details = JsonObject::new();
        if let Some(status) = self.status {
            details.insert("status".into(), json!(status));
        }
        // Modeled Bedrock errors end in `Exception`, unlike transport failures or `Unknown`.
        if let Some(code) = self
            .error_code
            .as_deref()
            .filter(|code| code.ends_with("Exception"))
            .and_then(|code| normalize_diagnostic_value(Some(code)))
        {
            details.insert("errorCode".into(), json!(code));
        }
        if let Some(request_id) = self
            .request_id
            .clone()
            .or_else(|| fallback_request_id.map(str::to_owned))
        {
            details.insert("requestId".into(), json!(request_id));
        }
        (!details.is_empty()).then_some(details)
    }
}

#[derive(Debug)]
enum Failure {
    Aborted,
    Failed(FailureInfo),
}

impl From<Error> for Failure {
    fn from(error: Error) -> Self {
        match error {
            Error::Aborted => Self::Aborted,
            other => Self::Failed(FailureInfo::plain(other.to_string())),
        }
    }
}

impl From<FailureInfo> for Failure {
    fn from(info: FailureInfo) -> Self {
        Self::Failed(info)
    }
}

/// An error and its sources, joined with `: ` (segments already contained in the previous one
/// are skipped).
fn error_chain(error: &(dyn std::error::Error + 'static)) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut current = Some(error);
    while let Some(error) = current {
        let text = error.to_string();
        if !text.is_empty() && !parts.last().is_some_and(|last| last.contains(&text)) {
            parts.push(text);
        }
        current = error.source();
    }
    parts.join(": ")
}

/// Map a failed `ConverseStream` call (before the event stream starts).
fn send_failure(error: &SdkError<ConverseStreamError>) -> FailureInfo {
    let SdkError::ServiceError(context) = error else {
        return FailureInfo::plain(error_chain(error));
    };
    let service = context.err();
    let raw = context.raw();
    let status = raw.status().as_u16();
    let name = service.code().unwrap_or("Unknown").to_owned();
    // Surface the raw body (with status) when the SDK found no message in it; this stops a
    // gateway 403 from collapsing to `Unknown: UnknownError`.
    let core = match service.message().filter(|message| !message.is_empty()) {
        Some(message) => message.to_owned(),
        None => {
            let body = raw
                .body()
                .bytes()
                .map(|bytes| String::from_utf8_lossy(bytes).trim().to_owned())
                .unwrap_or_default();
            if body.is_empty() {
                "UnknownError".to_owned()
            } else {
                format!(
                    "{status}: {}",
                    truncate_error_text(&body, MAX_PROVIDER_ERROR_BODY_CHARS)
                )
            }
        }
    };
    FailureInfo {
        core,
        service_name: Some(name.clone()),
        status: Some(status),
        error_code: Some(name),
        request_id: request_id_header(|header| raw.headers().get(header)),
    }
}

/// Map an error of the event stream. Modeled stream exceptions also return the provider event
/// they arrived as, for `on_provider_stream_event`.
fn stream_failure(error: &SdkError<ConverseStreamOutputError, RawMessage>) -> (Option<Value>, FailureInfo) {
    let Some(service) = error.as_service_error() else {
        return (None, FailureInfo::plain(error_chain(error)));
    };
    let member = match service {
        ConverseStreamOutputError::InternalServerException(_) => Some("InternalServerException"),
        ConverseStreamOutputError::ModelStreamErrorException(_) => Some("ModelStreamErrorException"),
        ConverseStreamOutputError::ValidationException(_) => Some("ValidationException"),
        ConverseStreamOutputError::ThrottlingException(_) => Some("ThrottlingException"),
        ConverseStreamOutputError::ServiceUnavailableException(_) => Some("ServiceUnavailableException"),
        _ => None,
    };
    let message = service
        .message()
        .filter(|message| !message.is_empty())
        .map(str::to_owned);
    match member {
        Some(name) => {
            let mut event_name = name.to_owned();
            event_name[..1].make_ascii_lowercase();
            let mut event = Map::new();
            event.insert(event_name, json!({"message": message}));
            let event = Value::Object(event);
            let info = FailureInfo {
                core: message.unwrap_or_else(|| "UnknownError".into()),
                service_name: Some(name.to_owned()),
                error_code: Some(name.to_owned()),
                ..FailureInfo::default()
            };
            (Some(event), info)
        }
        None => {
            // An unmodeled error frame, named after its error code.
            let code = service.code().map(str::to_owned);
            let core = message.or_else(|| code.clone()).unwrap_or_else(|| error_chain(error));
            (
                None,
                FailureInfo {
                    core,
                    error_code: code,
                    ..FailureInfo::default()
                },
            )
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Response stream
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockKind {
    Text,
    Thinking,
    ToolCall,
}

/// Streaming scratch state of one content block; never stored on the block itself.
#[derive(Debug)]
struct BlockState {
    content_index: usize,
    kind: BlockKind,
    /// Bedrock `contentBlockIndex` while the block is open.
    bedrock_index: Option<i32>,
    /// Encrypted reasoning deltas, joined into `thinking_signature` when the block ends.
    redacted: Option<Vec<u8>>,
}

#[derive(Debug, Default)]
struct StreamState {
    blocks: Vec<BlockState>,
}

impl StreamState {
    fn find(&self, bedrock_index: i32) -> Option<usize> {
        self.blocks
            .iter()
            .position(|block| block.bedrock_index == Some(bedrock_index))
    }

    fn handle(
        &mut self,
        event: bedrock::ConverseStreamOutput,
        model: &Model,
        builder: &mut MessageBuilder,
    ) -> std::result::Result<(), FailureInfo> {
        match event {
            bedrock::ConverseStreamOutput::MessageStart(start) => {
                if start.role != bedrock::ConversationRole::Assistant {
                    return Err(FailureInfo::plain(
                        "Unexpected assistant message start but got user message start instead",
                    ));
                }
                builder.start();
            }
            bedrock::ConverseStreamOutput::ContentBlockStart(start) => {
                if let Some(bedrock::ContentBlockStart::ToolUse(tool_use)) = start.start {
                    let content_index = builder.start_tool_call(tool_use.tool_use_id, tool_use.name, JsonObject::new());
                    self.blocks.push(BlockState {
                        content_index,
                        kind: BlockKind::ToolCall,
                        bedrock_index: Some(start.content_block_index),
                        redacted: None,
                    });
                }
            }
            bedrock::ConverseStreamOutput::ContentBlockDelta(delta) => {
                self.handle_delta(delta.content_block_index, delta.delta, builder);
            }
            bedrock::ConverseStreamOutput::ContentBlockStop(stop) => {
                self.handle_stop(stop.content_block_index, builder);
            }
            bedrock::ConverseStreamOutput::MessageStop(stop) => {
                let raw = stop.stop_reason.as_str().to_owned();
                let (reason, error_message) = map_stop_reason(Some(&raw));
                builder.output.raw_stop_reason = Some(raw);
                builder.output.stop_reason = reason;
                if error_message.is_some() {
                    builder.output.error_message = error_message;
                }
            }
            bedrock::ConverseStreamOutput::Metadata(metadata) => {
                if let Some(usage) = metadata.usage {
                    apply_usage(model, &usage, builder);
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_delta(
        &mut self,
        bedrock_index: i32,
        delta: Option<bedrock::ContentBlockDelta>,
        builder: &mut MessageBuilder,
    ) {
        let position = self.find(bedrock_index);
        match delta {
            Some(bedrock::ContentBlockDelta::Text(text)) => {
                // There is no contentBlockStart for text blocks; the first delta opens one.
                let position = position.unwrap_or_else(|| {
                    let content_index = builder.start_text();
                    self.blocks.push(BlockState {
                        content_index,
                        kind: BlockKind::Text,
                        bedrock_index: Some(bedrock_index),
                        redacted: None,
                    });
                    self.blocks.len() - 1
                });
                let block = &self.blocks[position];
                if block.kind == BlockKind::Text {
                    builder.text_delta(block.content_index, &text);
                }
            }
            Some(bedrock::ContentBlockDelta::ToolUse(tool_use)) => {
                if let Some(block) = position.map(|position| &self.blocks[position])
                    && block.kind == BlockKind::ToolCall
                {
                    builder.tool_call_delta(block.content_index, &tool_use.input);
                }
            }
            Some(bedrock::ContentBlockDelta::ReasoningContent(reasoning)) => {
                let position = position.unwrap_or_else(|| {
                    let content_index = builder.start_thinking(String::new(), false, Some(String::new()));
                    self.blocks.push(BlockState {
                        content_index,
                        kind: BlockKind::Thinking,
                        bedrock_index: Some(bedrock_index),
                        redacted: None,
                    });
                    self.blocks.len() - 1
                });
                let block = &mut self.blocks[position];
                if block.kind != BlockKind::Thinking {
                    return;
                }
                let index = block.content_index;
                let is_redacted = builder
                    .output
                    .content
                    .get(index)
                    .and_then(AssistantBlock::as_thinking)
                    .is_some_and(crate::types::ThinkingContent::is_redacted);
                match reasoning {
                    bedrock::ReasoningContentBlockDelta::Text(text) => builder.thinking_delta(index, &text),
                    // `thinking_signature` holds either an Anthropic signature or an opaque
                    // redacted payload, never both.
                    bedrock::ReasoningContentBlockDelta::Signature(signature) => {
                        if !signature.is_empty() && !is_redacted {
                            builder.append_thinking_signature(index, &signature);
                        }
                    }
                    bedrock::ReasoningContentBlockDelta::RedactedContent(bytes) => {
                        let bytes = bytes.into_inner();
                        if bytes.is_empty() {
                            return;
                        }
                        // Encrypted reasoning from non-Anthropic models (e.g. OpenAI GPT-5.6),
                        // kept verbatim in `thinking_signature` and replayed on the next turn.
                        if !is_redacted {
                            if let Some(AssistantBlock::Thinking(thinking)) = builder.output.content.get_mut(index) {
                                thinking.redacted = Some(true);
                            }
                            builder.set_thinking_signature(index, Some(String::new()));
                            builder.thinking_delta(index, REDACTED_THINKING_PLACEHOLDER);
                        }
                        block.redacted.get_or_insert_with(Vec::new).extend_from_slice(&bytes);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn handle_stop(&mut self, bedrock_index: i32, builder: &mut MessageBuilder) {
        let Some(position) = self.find(bedrock_index) else {
            return;
        };
        self.blocks[position].bedrock_index = None;
        let index = self.blocks[position].content_index;
        match self.blocks[position].kind {
            BlockKind::Text => builder.end_text(index),
            BlockKind::Thinking => {
                Self::flush_redacted(&mut self.blocks[position], builder);
                builder.end_thinking(index);
            }
            BlockKind::ToolCall => builder.end_tool_call(index),
        }
    }

    /// Encode buffered encrypted reasoning into `thinking_signature`.
    fn flush_redacted(block: &mut BlockState, builder: &mut MessageBuilder) {
        if let Some(bytes) = block.redacted.take() {
            builder.set_thinking_signature(block.content_index, Some(STANDARD.encode(bytes)));
        }
    }

    /// A stream can settle without stopping every block: flush scratch state of all blocks.
    fn finalize(&mut self, builder: &mut MessageBuilder) {
        for block in &mut self.blocks {
            Self::flush_redacted(block, builder);
        }
    }

    /// Keep the best-effort arguments of tool calls that never finished, for a failed message.
    fn keep_partial_tool_arguments(&self, builder: &mut MessageBuilder) {
        for block in &self.blocks {
            if block.kind != BlockKind::ToolCall || block.bedrock_index.is_none() {
                continue;
            }
            if let Some(json) = builder.tool_call_json(block.content_index).map(str::to_owned)
                && let Some(call) = builder.tool_call_mut(block.content_index)
            {
                call.arguments = parse_streaming_json(&json);
            }
        }
    }
}

fn map_stop_reason(reason: Option<&str>) -> (StopReason, Option<String>) {
    match reason {
        Some("end_turn" | "stop_sequence") => (StopReason::Stop, None),
        Some("max_tokens" | "model_context_window_exceeded") => (StopReason::Length, None),
        Some("tool_use") => (StopReason::ToolUse, None),
        Some(reason) if !reason.is_empty() => (StopReason::Error, Some(format!("Provider stopped with: {reason}"))),
        _ => (StopReason::Error, None),
    }
}

fn apply_usage(model: &Model, usage: &bedrock::TokenUsage, builder: &mut MessageBuilder) {
    let count = |value: i32| u64::try_from(value).unwrap_or(0);
    let target = &mut builder.output.usage;
    target.input = count(usage.input_tokens);
    target.output = count(usage.output_tokens);
    target.cache_read = usage.cache_read_input_tokens.map_or(0, count);
    target.cache_write = usage.cache_write_input_tokens.map_or(0, count);
    target.cache_write_1h = usage.cache_details.as_ref().map(|details| {
        details
            .iter()
            .filter(|detail| detail.ttl == bedrock::CacheTtl::OneHour)
            .map(|detail| count(detail.input_tokens))
            .sum()
    });
    target.total_tokens = match count(usage.total_tokens) {
        0 => target.input + target.output,
        total => total,
    };
    calculate_cost(&model.cost, target);
}

/// The provider event as JSON, in the shape of the JS SDK's stream items.
fn event_to_json(event: &bedrock::ConverseStreamOutput) -> Value {
    match event {
        bedrock::ConverseStreamOutput::MessageStart(start) => {
            json!({"messageStart": {"role": start.role.as_str()}})
        }
        bedrock::ConverseStreamOutput::ContentBlockStart(start) => {
            let inner = match &start.start {
                Some(bedrock::ContentBlockStart::ToolUse(tool_use)) => {
                    json!({"toolUse": {"toolUseId": tool_use.tool_use_id, "name": tool_use.name}})
                }
                Some(bedrock::ContentBlockStart::Image(_)) => json!({"image": {}}),
                Some(bedrock::ContentBlockStart::ToolResult(_)) => json!({"toolResult": {}}),
                _ => json!({}),
            };
            json!({"contentBlockStart": {"contentBlockIndex": start.content_block_index, "start": inner}})
        }
        bedrock::ConverseStreamOutput::ContentBlockDelta(delta) => {
            let inner = match &delta.delta {
                Some(bedrock::ContentBlockDelta::Text(text)) => json!({"text": text}),
                Some(bedrock::ContentBlockDelta::ToolUse(tool_use)) => json!({"toolUse": {"input": tool_use.input}}),
                Some(bedrock::ContentBlockDelta::ReasoningContent(reasoning)) => {
                    let reasoning = match reasoning {
                        bedrock::ReasoningContentBlockDelta::Text(text) => json!({"text": text}),
                        bedrock::ReasoningContentBlockDelta::Signature(signature) => json!({"signature": signature}),
                        bedrock::ReasoningContentBlockDelta::RedactedContent(bytes) => {
                            json!({"redactedContent": STANDARD.encode(bytes.as_ref())})
                        }
                        _ => json!({}),
                    };
                    json!({"reasoningContent": reasoning})
                }
                Some(bedrock::ContentBlockDelta::Citation(_)) => json!({"citation": {}}),
                Some(bedrock::ContentBlockDelta::Image(_)) => json!({"image": {}}),
                Some(bedrock::ContentBlockDelta::ToolResult(_)) => json!({"toolResult": []}),
                _ => json!({}),
            };
            json!({"contentBlockDelta": {"contentBlockIndex": delta.content_block_index, "delta": inner}})
        }
        bedrock::ConverseStreamOutput::ContentBlockStop(stop) => {
            json!({"contentBlockStop": {"contentBlockIndex": stop.content_block_index}})
        }
        bedrock::ConverseStreamOutput::MessageStop(stop) => {
            let mut inner = Map::new();
            inner.insert("stopReason".into(), json!(stop.stop_reason.as_str()));
            if let Some(fields) = &stop.additional_model_response_fields {
                inner.insert("additionalModelResponseFields".into(), document_to_json(fields));
            }
            json!({"messageStop": inner})
        }
        bedrock::ConverseStreamOutput::Metadata(metadata) => {
            let mut inner = Map::new();
            if let Some(usage) = &metadata.usage {
                let mut value = Map::new();
                value.insert("inputTokens".into(), json!(usage.input_tokens));
                value.insert("outputTokens".into(), json!(usage.output_tokens));
                value.insert("totalTokens".into(), json!(usage.total_tokens));
                if let Some(tokens) = usage.cache_read_input_tokens {
                    value.insert("cacheReadInputTokens".into(), json!(tokens));
                }
                if let Some(tokens) = usage.cache_write_input_tokens {
                    value.insert("cacheWriteInputTokens".into(), json!(tokens));
                }
                if let Some(details) = &usage.cache_details {
                    let details: Vec<Value> = details
                        .iter()
                        .map(|detail| json!({"ttl": detail.ttl.as_str(), "inputTokens": detail.input_tokens}))
                        .collect();
                    value.insert("cacheDetails".into(), Value::Array(details));
                }
                inner.insert("usage".into(), Value::Object(value));
            }
            if let Some(metrics) = &metadata.metrics {
                inner.insert("metrics".into(), json!({"latencyMs": metrics.latency_ms}));
            }
            json!({"metadata": inner})
        }
        _ => json!({"$unknown": {}}),
    }
}

// ---------------------------------------------------------------------------------------------
// Request driver
// ---------------------------------------------------------------------------------------------

async fn cancellable<T>(
    cancel: Option<&CancellationToken>,
    future: impl Future<Output = T>,
) -> std::result::Result<T, Failure> {
    match cancel {
        Some(cancel) => tokio::select! {
            () = cancel.cancelled() => Err(Failure::Aborted),
            value = future => Ok(value),
        },
        None => Ok(future.await),
    }
}

async fn run(
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
    builder: &mut MessageBuilder,
) -> Result<()> {
    let mut state = StreamState::default();
    // Kept outside `drive` so a mid-stream failure can still be correlated: exceptions inside
    // the event stream carry no HTTP metadata of their own.
    let mut request_id = None;
    let outcome = Box::pin(drive(model, context, options, builder, &mut state, &mut request_id)).await;
    if builder.is_closed() {
        return Err(Error::Aborted);
    }
    if outcome.is_err() {
        state.keep_partial_tool_arguments(builder);
    }
    state.finalize(builder);
    match outcome {
        Ok(()) => Ok(()),
        Err(Failure::Aborted) => Err(Error::Aborted),
        Err(Failure::Failed(_)) if options.is_cancelled() => Err(Error::Aborted),
        Err(Failure::Failed(info)) => {
            if let Some(details) = info.diagnostic_details(request_id.as_deref()) {
                builder.output.push_diagnostic(AssistantMessageDiagnostic::new(
                    FAILURE_DIAGNOSTIC_TYPE,
                    None,
                    Some(details),
                ));
            }
            builder.output.stop_reason = StopReason::Error;
            builder.output.error_message = Some(info.message());
            Ok(())
        }
    }
}

async fn drive(
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
    builder: &mut MessageBuilder,
    state: &mut StreamState,
    request_id: &mut Option<String>,
) -> std::result::Result<(), Failure> {
    let cancel = options.cancel.as_ref();
    let bedrock = bedrock_options(options);
    let env = EnvLookup {
        scoped: options.env.as_ref(),
        process: &process_env,
    };
    let plan = resolve_client_plan(model, &bedrock, options.api_key.as_deref(), &env);
    let payload = build_payload(model, context, options, &bedrock, &env)?;
    let payload = apply_on_payload(options, model, payload).await?;
    let request = ConverseRequest::from_payload(&payload)?;
    let headers = custom_headers(options.headers.as_ref())?;

    let client = cancellable(cancel, Box::pin(build_client(&plan, options))).await?;
    let capture = ResponseCapture::default();
    let mut operation = client
        .converse_stream()
        .model_id(request.model_id)
        .set_messages(Some(request.messages))
        .set_system(request.system)
        .set_inference_config(request.inference_config)
        .set_tool_config(request.tool_config)
        .set_additional_model_request_fields(request.additional_model_request_fields)
        .set_additional_model_response_field_paths(request.additional_model_response_field_paths)
        .set_request_metadata(request.request_metadata)
        .customize()
        .interceptor(capture.clone());
    if let Some(headers) = headers {
        // Runs before signing, so the headers are covered by the SigV4 signature.
        operation = operation.mutate_request(move |request| {
            for (name, value) in &headers {
                request.headers_mut().remove(name.as_str());
                let _ = request.headers_mut().try_insert(name.clone(), value.clone());
            }
        });
    }
    let mut response = match cancellable(cancel, operation.send()).await? {
        Ok(response) => response,
        Err(error) => return Err(send_failure(&error).into()),
    };

    let observed = capture.take().unwrap_or_else(|| ProviderResponse {
        status: 200,
        headers: aws_sdk_bedrockruntime::operation::RequestId::request_id(&response)
            .map(|id| [("x-amzn-requestid".to_owned(), id.to_owned())].into_iter().collect())
            .unwrap_or_default(),
    });
    *request_id = request_id_header(|name| observed.headers.get(name).map(String::as_str))
        .or_else(|| normalize_diagnostic_value(aws_sdk_bedrockruntime::operation::RequestId::request_id(&response)));
    if let Some(on_response) = &options.on_response {
        on_response(&observed, model)
            .await
            .map_err(|error| Failure::from(Error::msg(error.to_string())))?;
    }

    let idle_timeout = options.timeout.unwrap_or(DEFAULT_TIMEOUT);
    loop {
        if builder.is_closed() {
            return Err(Failure::Aborted);
        }
        let received = cancellable(cancel, tokio::time::timeout(idle_timeout, response.stream.recv())).await?;
        match received {
            Err(_) => {
                return Err(FailureInfo::plain(format!(
                    "Bedrock stream timed out: no event for {}s",
                    idle_timeout.as_secs_f64()
                ))
                .into());
            }
            Ok(Ok(Some(event))) => {
                observe_provider_event(options, model, &event_to_json(&event)).await?;
                state.handle(event, model, builder)?;
            }
            Ok(Ok(None)) => break,
            Ok(Err(error)) => {
                let (event, info) = stream_failure(&error);
                if let Some(event) = event {
                    observe_provider_event(options, model, &event).await?;
                }
                return Err(info.into());
            }
        }
    }

    if options.is_cancelled() {
        return Err(Failure::Aborted);
    }
    match builder.output.stop_reason {
        StopReason::Pending => Err(FailureInfo::plain("Bedrock stream ended without a stop reason").into()),
        StopReason::Error | StopReason::Aborted => Err(FailureInfo::plain(
            builder
                .output
                .error_message
                .clone()
                .unwrap_or_else(|| "An unknown error occurred".into()),
        )
        .into()),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests;
