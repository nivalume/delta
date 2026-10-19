//! `google-vertex`: Gemini models on Vertex AI. Port of upstream `src/api/google-vertex.ts`.
//!
//! Upstream streams through `@google/genai` in Vertex mode (REST). This module uses the
//! official Google Cloud client libraries for Rust instead: `google-cloud-aiplatform-v1`
//! (`PredictionService::stream_generate_content`, a gRPC server stream) and
//! `google-cloud-auth` (Application Default Credentials, a credentials file, or a Vertex API
//! key).
//!
//! Request building, replay rules and the stream state machine live in
//! [`super::google_shared`]. The request is first rendered as the REST JSON body the Gemini
//! API module sends ([`super::google_generative_ai::request_body`]), which is also what
//! `@google/genai` sends to Vertex, so `on_payload` sees and may replace the same JSON. The
//! JSON is then converted to the proto `GenerateContentRequest` through the crate's serde
//! implementation (proto JSON field names, base64 bytes, enum names). Responses are converted
//! from the proto types directly.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use google_cloud_aiplatform_v1::client::PredictionService;
use google_cloud_aiplatform_v1::model::candidate::FinishReason;
use google_cloud_aiplatform_v1::model::{GenerateContentRequest, GenerateContentResponse, Part as ProtoPart};
use google_cloud_auth::credentials::{
    self as gcp_credentials, Credentials, api_key_credentials, external_account, impersonated, service_account,
    user_account,
};
use serde_json::Value;

use crate::api::common::{apply_on_payload, observe_provider_event, spawn_stream};
use crate::api::google_generative_ai::request_body;
use crate::api::google_shared::{
    Candidate, RequestSettings, ResolvedGoogleThinkingLevel, ResponseChunk, ResponseFunctionCall, ResponsePart,
    StreamState, UsageMetadata, build_request, check_finished, resolve_google_thinking_level, retry_google_request,
    simple_tool_choice, to_google_thinking_level, uses_google_thinking_level,
};
use crate::error::{Error, Result};
use crate::events::{AssistantMessageEventStream, MessageBuilder};
use crate::http;
use crate::model::{Model, ModelThinkingLevel, ThinkingLevel};
use crate::options::{
    ApiOptions, GoogleThinking, GoogleVertexOptions, SimpleStreamOptions, StreamOptions, ThinkingBudgets,
};
use crate::provider::ProviderStreams;
use crate::simple_options::build_base_options;
use crate::types::{ProviderHeaders, TranscriptContext};

/// `options.api_key` value that means "use Application Default Credentials".
const GCP_VERTEX_CREDENTIALS_MARKER: &str = "gcp-vertex-credentials";
/// Endpoint for the `global` location and for API keys (express mode).
const GLOBAL_ENDPOINT: &str = "https://aiplatform.googleapis.com";
const MISSING_FINISH_REASON: &str = "Google Vertex stream ended without a finish reason";

/// The `google-vertex` API.
#[derive(Clone, Copy, Debug, Default)]
pub struct GoogleVertexApi;

impl ProviderStreams for GoogleVertexApi {
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

/// Stream a Vertex AI response. API options come from [`ApiOptions::GoogleVertex`].
///
/// Authentication: a Vertex API key in `options.api_key` (express mode), else Application
/// Default Credentials (`GOOGLE_APPLICATION_CREDENTIALS` from `options.env` or the process
/// environment, else gcloud's well-known file, else the metadata server) with a project
/// (`project`, `GOOGLE_CLOUD_PROJECT`, `GCLOUD_PROJECT`) and a location (`location`,
/// `GOOGLE_CLOUD_LOCATION`).
pub fn stream(model: &Model, context: &TranscriptContext, options: StreamOptions) -> AssistantMessageEventStream {
    spawn_vertex_stream(model, context, options, None)
}

/// [`stream`] through an existing client instead of one built from the options. The client's
/// endpoint, credentials and policies apply; `options` still selects the model resource name
/// (API key or project and location), the request and the retry policy.
pub fn stream_with_client(
    model: &Model,
    context: &TranscriptContext,
    options: StreamOptions,
    client: PredictionService,
) -> AssistantMessageEventStream {
    spawn_vertex_stream(model, context, options, Some(client))
}

fn spawn_vertex_stream(
    model: &Model,
    context: &TranscriptContext,
    options: StreamOptions,
    client: Option<PredictionService>,
) -> AssistantMessageEventStream {
    let owned_model = model.clone();
    let context = context.clone();
    let cancel = options.cancel.clone();
    spawn_stream(model, cancel, move |mut builder| async move {
        let result = Box::pin(run(&owned_model, &context, &options, client, &mut builder)).await;
        (builder, result)
    })
}

/// Stream with provider-neutral options: `reasoning` and `thinking_budgets` become the
/// `thinking` option, `tool_choice` the Gemini tool choice. As upstream, `project` and
/// `location` are not carried over; they come from `options.base.env` or the environment.
#[allow(clippy::needless_pass_by_value)] // Owned options are the API contract.
pub fn stream_simple(
    model: &Model,
    context: &TranscriptContext,
    options: SimpleStreamOptions,
) -> AssistantMessageEventStream {
    let mut base = build_base_options(model, context, &options);
    let thinking = match simple_thinking(model, options.reasoning, options.thinking_budgets.as_ref()) {
        Ok(thinking) => thinking,
        Err(error) => {
            return spawn_stream(model, base.cancel.clone(), move |builder| async move {
                (builder, Err(error))
            });
        }
    };
    base.api = Some(ApiOptions::GoogleVertex(GoogleVertexOptions {
        tool_choice: options.tool_choice.map(simple_tool_choice),
        thinking: Some(thinking),
        ..GoogleVertexOptions::default()
    }));
    stream(model, context, base)
}

/// The `thinking` option of a simple request (upstream `streamSimple`): disabled without a
/// reasoning level or when the model clamps it to `off`; otherwise a level for models with
/// discrete levels and a token budget ([`get_google_budget`]) for the rest.
fn simple_thinking(
    model: &Model,
    reasoning: Option<ThinkingLevel>,
    budgets: Option<&ThinkingBudgets>,
) -> Result<GoogleThinking> {
    let disabled = GoogleThinking {
        enabled: false,
        ..GoogleThinking::default()
    };
    let Some(reasoning) = reasoning else {
        return Ok(disabled);
    };
    let clamped = model.clamp_thinking_level(reasoning.into());
    if clamped == ModelThinkingLevel::Off {
        return Ok(disabled);
    }
    let resolved = resolve_google_thinking_level(model, clamped)?;
    if uses_google_thinking_level(model) {
        return Ok(GoogleThinking {
            enabled: true,
            budget_tokens: None,
            level: Some(to_google_thinking_level(resolved)),
        });
    }
    Ok(GoogleThinking {
        enabled: true,
        budget_tokens: Some(get_google_budget(model, resolved, budgets)),
        level: None,
    })
}

/// Default token budget for a level on budget-based Gemini models on Vertex; `-1` is dynamic.
///
/// Upstream's Vertex table has no `2.5-flash-lite` row (the Gemini API table does), so
/// Flash-Lite uses the `2.5-flash` budgets here.
pub fn get_google_budget(model: &Model, level: ResolvedGoogleThinkingLevel, custom: Option<&ThinkingBudgets>) -> i64 {
    let custom_budget = custom.and_then(|budgets| match level {
        ResolvedGoogleThinkingLevel::Minimal => budgets.minimal,
        ResolvedGoogleThinkingLevel::Low => budgets.low,
        ResolvedGoogleThinkingLevel::Medium => budgets.medium,
        ResolvedGoogleThinkingLevel::High => budgets.high,
    });
    if let Some(budget) = custom_budget {
        return budget as i64;
    }
    let table: [i64; 4] = if model.id.contains("2.5-pro") {
        [128, 2048, 8192, 32768]
    } else if model.id.contains("2.5-flash") {
        [128, 2048, 8192, 24576]
    } else {
        return -1;
    };
    match level {
        ResolvedGoogleThinkingLevel::Minimal => table[0],
        ResolvedGoogleThinkingLevel::Low => table[1],
        ResolvedGoogleThinkingLevel::Medium => table[2],
        ResolvedGoogleThinkingLevel::High => table[3],
    }
}

async fn run(
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
    client: Option<PredictionService>,
    builder: &mut MessageBuilder,
) -> Result<()> {
    if options.is_cancelled() {
        return Err(Error::Aborted);
    }
    // Not ported: upstream rejects a custom `fetch`. The SDK has its own transport, so
    // `options.http_client` is not used.
    let vertex = match &options.api {
        Some(ApiOptions::GoogleVertex(vertex)) => vertex.clone(),
        _ => GoogleVertexOptions::default(),
    };
    let env = |name: &str| options.env_var(name);
    let target = resolve_target(model, &vertex, options.api_key.as_deref(), &env)?;
    let client = match client {
        Some(client) => client,
        None => build_client(&target, options).await?,
    };

    let request = build_request(
        model,
        context,
        &RequestSettings {
            temperature: options.temperature,
            max_tokens: options.max_tokens,
            tool_choice: vertex.tool_choice,
            thinking: vertex.thinking.as_ref(),
        },
    )?;
    let body = apply_on_payload(options, model, request_body(&request)).await?;
    let request = proto_request(body, &target.model)?;

    let mut responses = retry_google_request(options, || {
        let client = client.clone();
        let request = request.clone();
        // The SDK's request future is large; keep it off the stream task's stack.
        Box::pin(async move {
            client
                .stream_generate_content()
                .with_request(request)
                .send()
                .await
                .map_err(|error| vertex_error(&error))
        })
    })
    .await?;
    // Upstream's Vertex adapter calls no `on_response`; the gRPC stream has no HTTP response.
    builder.start();

    let mut state = StreamState::new();
    loop {
        let next = http::or_cancel(options.cancel.as_ref(), async { Ok(responses.next().await) }).await?;
        let Some(item) = next else {
            break;
        };
        if builder.is_closed() {
            return Err(Error::Aborted);
        }
        let response = item.map_err(|error| vertex_error(&error))?;
        if options.on_provider_stream_event.is_some() {
            observe_provider_event(options, model, &response_json(&response)).await?;
        }
        state.apply_chunk(builder, model, &response_chunk(&response));
    }
    state.finish(builder);
    if options.is_cancelled() {
        return Err(Error::Aborted);
    }
    check_finished(&builder.output, MISSING_FINISH_REASON)
}

// ---------------------------------------------------------------------------
// Client configuration
// ---------------------------------------------------------------------------

/// How a request authenticates.
#[derive(Clone, Debug, PartialEq, Eq)]
enum VertexAuth {
    /// A Vertex API key (express mode): no project or location.
    ApiKey(String),
    /// Application Default Credentials, or the credentials file named by
    /// `GOOGLE_APPLICATION_CREDENTIALS`.
    Adc { credentials_file: Option<String> },
}

/// Where and how a request is sent.
#[derive(Clone, Debug, PartialEq, Eq)]
struct VertexTarget {
    auth: VertexAuth,
    /// gRPC endpoint (scheme and authority).
    endpoint: String,
    /// `GenerateContentRequest.model`: the full resource name.
    model: String,
}

/// The Vertex API key in `options.api_key`, or `None` for ADC: no key, the
/// `gcp-vertex-credentials` marker, or a `<placeholder>` such as `<authenticated>`.
fn resolve_api_key(api_key: Option<&str>) -> Option<String> {
    let key = api_key?.trim();
    let placeholder =
        key.len() > 2 && key.starts_with('<') && key.ends_with('>') && !key[1..key.len() - 1].contains('>');
    if key.is_empty() || key == GCP_VERTEX_CREDENTIALS_MARKER || placeholder {
        return None;
    }
    Some(key.to_owned())
}

fn non_empty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn resolve_project(vertex: &GoogleVertexOptions, env: &dyn Fn(&str) -> Option<String>) -> Result<String> {
    non_empty(vertex.project.as_deref())
        .or_else(|| env("GOOGLE_CLOUD_PROJECT"))
        .or_else(|| env("GCLOUD_PROJECT"))
        .ok_or_else(|| {
            Error::msg(
                "Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options.",
            )
        })
}

fn resolve_location(vertex: &GoogleVertexOptions, env: &dyn Fn(&str) -> Option<String>) -> Result<String> {
    non_empty(vertex.location.as_deref())
        .or_else(|| env("GOOGLE_CLOUD_LOCATION"))
        .ok_or_else(|| {
            Error::msg("Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options.")
        })
}

/// A model base URL to use, skipping generated `{location}` placeholders (upstream
/// `resolveCustomBaseUrl`).
fn custom_base_url(base_url: &str) -> Option<&str> {
    let trimmed = base_url.trim();
    (!trimmed.is_empty() && !trimmed.contains("{location}")).then_some(trimmed)
}

/// The scheme and authority of a base URL: gRPC paths are fixed, so a path in the URL is not
/// used. Unparseable values are used as given.
fn grpc_origin(base_url: &str) -> String {
    match url::Url::parse(base_url) {
        Ok(url) if url.has_host() => url.origin().ascii_serialization(),
        _ => base_url.trim_end_matches('/').to_owned(),
    }
}

/// The endpoint for a location, as the Google Gen AI SDKs choose it: `global` (and API keys)
/// use the global host, the `us`/`eu` multi-regions their regional endpoints, other
/// locations `{location}-aiplatform.googleapis.com`.
fn location_endpoint(location: Option<&str>) -> String {
    match location {
        None | Some("global") => GLOBAL_ENDPOINT.to_owned(),
        Some(region @ ("us" | "eu")) => format!("https://aiplatform.{region}.rep.googleapis.com"),
        Some(location) => format!("https://{location}-aiplatform.googleapis.com"),
    }
}

/// `publishers/{publisher}/models/{model}` for a model id (the SDK's `tModel` in Vertex
/// mode). Ids already starting with `publishers/`, `projects/` or `models/` are kept.
fn model_path(id: &str) -> Result<String> {
    if id.is_empty() {
        return Err(Error::msg("model is required and must be a string"));
    }
    if id.contains("..") || id.contains('?') || id.contains('&') {
        return Err(Error::msg("invalid model parameter"));
    }
    if id.starts_with("publishers/") || id.starts_with("projects/") || id.starts_with("models/") {
        return Ok(id.to_owned());
    }
    let mut segments = id.split('/');
    match (segments.next(), segments.next()) {
        (Some(publisher), Some(name)) => Ok(format!("publishers/{publisher}/models/{name}")),
        _ => Ok(format!("publishers/google/models/{id}")),
    }
}

/// Resolve authentication, endpoint and model resource name (upstream `createClient` /
/// `createClientWithApiKey`).
fn resolve_target(
    model: &Model,
    vertex: &GoogleVertexOptions,
    api_key: Option<&str>,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<VertexTarget> {
    let base_url = custom_base_url(&model.base_url)
        .map(str::to_owned)
        .or_else(|| env("GOOGLE_VERTEX_BASE_URL").and_then(|url| non_empty(Some(&url))));
    let path = model_path(&model.id)?;
    if let Some(key) = resolve_api_key(api_key) {
        return Ok(VertexTarget {
            auth: VertexAuth::ApiKey(key),
            endpoint: base_url.as_deref().map_or_else(|| location_endpoint(None), grpc_origin),
            model: path,
        });
    }
    let project = resolve_project(vertex, env)?;
    let location = resolve_location(vertex, env)?;
    let model = if path.starts_with("projects/") {
        path
    } else {
        format!("projects/{project}/locations/{location}/{path}")
    };
    Ok(VertexTarget {
        auth: VertexAuth::Adc {
            credentials_file: env("GOOGLE_APPLICATION_CREDENTIALS"),
        },
        endpoint: base_url
            .as_deref()
            .map_or_else(|| location_endpoint(Some(&location)), grpc_origin),
        model,
    })
}

/// Credentials from a credentials file, by its `type` (as ADC loads them).
fn credentials_from_file(path: &str) -> Result<Credentials> {
    let contents = std::fs::read_to_string(path)
        .map_err(|error| Error::msg(format!("Failed to read Google credentials file {path}: {error}")))?;
    let json: Value = serde_json::from_str(&contents)
        .map_err(|error| Error::msg(format!("Failed to parse Google credentials file {path}: {error}")))?;
    let kind = json.get("type").and_then(Value::as_str).unwrap_or_default().to_owned();
    let built = match kind.as_str() {
        "service_account" => service_account::Builder::new(json).build(),
        "authorized_user" => user_account::Builder::new(json).build(),
        "impersonated_service_account" => impersonated::Builder::new(json).build(),
        "external_account" => external_account::Builder::new(json).build(),
        _ => {
            return Err(Error::msg(format!(
                "Unsupported Google credentials type in {path}: {}",
                if kind.is_empty() { "(missing)" } else { &kind }
            )));
        }
    };
    built.map_err(|error| Error::msg(format!("Failed to load Google credentials from {path}: {error}")))
}

fn credentials(auth: &VertexAuth) -> Result<Credentials> {
    match auth {
        VertexAuth::ApiKey(key) => Ok(api_key_credentials::Builder::new(key.clone()).build()),
        VertexAuth::Adc {
            credentials_file: Some(path),
        } => credentials_from_file(path),
        VertexAuth::Adc { credentials_file: None } => gcp_credentials::Builder::default().build().map_err(|error| {
            Error::msg(format!(
                "Failed to load Google Application Default Credentials: {error}"
            ))
        }),
    }
}

/// `options.headers` as client-wide request metadata. The SDK sets `user-agent`,
/// `authorization`, `x-goog-api-key`, `x-goog-api-client`, `x-goog-user-project` and
/// `x-goog-request-params` itself and drops custom values for them.
fn custom_headers(options: &StreamOptions) -> Result<reqwest::header::HeaderMap> {
    http::build_headers([options.headers.as_ref().unwrap_or(&ProviderHeaders::new())])
}

async fn build_client(target: &VertexTarget, options: &StreamOptions) -> Result<PredictionService> {
    let credentials = credentials(&target.auth)?;
    let headers = custom_headers(options)?;
    let mut builder = PredictionService::builder()
        .with_endpoint(target.endpoint.clone())
        .with_credentials(credentials)
        // A gRPC deadline covers the whole stream, like the HTTP timeout of the other APIs.
        .with_attempt_timeout(options.timeout.unwrap_or(http::DEFAULT_TIMEOUT));
    if !headers.is_empty() {
        builder = builder.with_extension(headers);
    }
    builder
        .build()
        .await
        .map_err(|error| Error::msg(format!("Failed to create the Vertex AI client: {error}")))
}

// ---------------------------------------------------------------------------
// Request and response conversion
// ---------------------------------------------------------------------------

/// The REST JSON request body (after `on_payload`) as the proto request for `model`. A
/// `model` field set by `on_payload` is kept. Fields the proto does not define (such as
/// `functionCall.id` and `functionResponse.id`) cannot be sent over gRPC and are dropped.
pub fn proto_request(body: Value, model: &str) -> Result<GenerateContentRequest> {
    let mut request: GenerateContentRequest = serde_json::from_value(body)
        .map_err(|error| Error::msg(format!("Invalid Google Vertex request payload: {error}")))?;
    if request.model.is_empty() {
        model.clone_into(&mut request.model);
    }
    Ok(request)
}

/// The REST name of a finish reason; `None` when unset. Unknown numeric values become their
/// number.
fn finish_reason_name(reason: &FinishReason) -> Option<String> {
    if matches!(reason, FinishReason::Unspecified) {
        return None;
    }
    reason
        .name()
        .map(str::to_owned)
        .or_else(|| reason.value().map(|value| value.to_string()))
}

/// A proto count; zero (the proto default, omitted in REST JSON) and negative values count
/// as absent.
fn count(value: i32) -> Option<u64> {
    u64::try_from(value).ok().filter(|value| *value > 0)
}

fn response_part(part: &ProtoPart) -> ResponsePart {
    ResponsePart {
        text: part.text().cloned(),
        thought: part.thought.then_some(true),
        thought_signature: (!part.thought_signature.is_empty()).then(|| STANDARD.encode(&part.thought_signature)),
        function_call: part.function_call().map(|call| ResponseFunctionCall {
            // The v1 proto has no function call id.
            id: None,
            name: (!call.name.is_empty()).then(|| call.name.clone()),
            args: call.args.clone(),
        }),
    }
}

/// Convert a streamed proto `GenerateContentResponse` (first candidate only). Thought
/// signatures (bytes) become standard base64 text, as REST returns them.
pub fn response_chunk(response: &GenerateContentResponse) -> ResponseChunk {
    let candidate = response.candidates.first().map(|candidate| Candidate {
        parts: candidate
            .content
            .as_ref()
            .map(|content| content.parts.iter().map(response_part).collect())
            .unwrap_or_default(),
        finish_reason: finish_reason_name(&candidate.finish_reason),
    });
    let usage = response.usage_metadata.as_ref().map(|usage| UsageMetadata {
        prompt_token_count: count(usage.prompt_token_count),
        cached_content_token_count: count(usage.cached_content_token_count),
        candidates_token_count: count(usage.candidates_token_count),
        thoughts_token_count: count(usage.thoughts_token_count),
        total_token_count: count(usage.total_token_count),
    });
    ResponseChunk {
        response_id: (!response.response_id.is_empty()).then(|| response.response_id.clone()),
        candidate,
        usage,
    }
}

/// The JSON given to `on_provider_stream_event`: the crate's serde form of the response
/// (REST field names, base64 bytes), with `finishReason` as its REST name. Other enum
/// fields keep the crate's numeric form.
pub fn response_json(response: &GenerateContentResponse) -> Value {
    let mut value = serde_json::to_value(response).unwrap_or(Value::Null);
    if let Some(candidates) = value.get_mut("candidates").and_then(Value::as_array_mut) {
        for (json, candidate) in candidates.iter_mut().zip(&response.candidates) {
            if let (Some(object), Some(name)) = (json.as_object_mut(), finish_reason_name(&candidate.finish_reason)) {
                object.insert("finishReason".into(), Value::String(name));
            }
        }
    }
    value
}

/// HTTP status equivalent of a gRPC status code (`google.rpc.Code` mapping).
fn grpc_http_status(code: &str) -> Option<u16> {
    Some(match code {
        "CANCELLED" => 499,
        "UNKNOWN" | "INTERNAL" | "DATA_LOSS" => 500,
        "INVALID_ARGUMENT" | "FAILED_PRECONDITION" | "OUT_OF_RANGE" => 400,
        "DEADLINE_EXCEEDED" => 504,
        "NOT_FOUND" => 404,
        "ALREADY_EXISTS" | "ABORTED" => 409,
        "PERMISSION_DENIED" => 403,
        "UNAUTHENTICATED" => 401,
        "RESOURCE_EXHAUSTED" => 429,
        "UNIMPLEMENTED" => 501,
        "UNAVAILABLE" => 503,
        _ => return None,
    })
}

/// An SDK error as a stream error. Service errors become [`Error::Status`] with the HTTP
/// equivalent of the gRPC code and the message `"<status> <message>"`, so the shared retry
/// policy and message-based classification work as for REST.
fn vertex_error(error: &google_cloud_aiplatform_v1::Error) -> Error {
    let status = error.status().and_then(|status| {
        let http_status = grpc_http_status(status.code.name()).or_else(|| error.http_status_code())?;
        let message = if status.message.is_empty() {
            status.code.name().to_owned()
        } else {
            status.message.clone()
        };
        Some((http_status, message))
    });
    match status.or_else(|| error.http_status_code().map(|code| (code, error.to_string()))) {
        Some((status, message)) => Error::Status {
            status,
            message: format!("{status} {message}"),
            retry_after_ms: None,
        },
        None => Error::msg(error.to_string()),
    }
}

#[cfg(test)]
mod tests;
