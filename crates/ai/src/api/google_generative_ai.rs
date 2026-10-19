//! `google-generative-ai`: the Gemini API (`generativelanguage.googleapis.com`). Port of
//! upstream `src/api/google-generative-ai.ts`.
//!
//! Upstream streams through `@google/genai`; this module sends the same REST request by
//! hand: `POST {baseUrl}/models/{model}:streamGenerateContent?alt=sse` with the key in the
//! `x-goog-api-key` header. Request building, replay rules and the stream state machine live
//! in [`super::google_shared`]; this module holds the REST JSON shapes and the transport.

use futures::StreamExt;
use serde_json::{Value, json};

use crate::api::common::{
    apply_on_payload, join_url, notify_response, observe_provider_event, require_auth, spawn_stream,
};
use crate::api::google_shared::{
    Candidate, Content, FunctionDeclaration, GenerateRequest, InlineData, Part, RequestSettings, ResponseChunk,
    ResponseFunctionCall, ResponsePart, StreamState, ThinkingConfig, ToolParameters, UsageMetadata, build_request,
    check_finished, format_google_http_error, google_thinking_level_name, simple_thinking, simple_tool_choice,
};
use crate::error::{Error, Result};
use crate::events::{AssistantMessageEventStream, MessageBuilder};
use crate::http;
use crate::model::Model;
use crate::options::{ApiOptions, GoogleOptions, SimpleStreamOptions, StreamOptions};
use crate::provider::ProviderStreams;
use crate::simple_options::build_base_options;
use crate::types::{JsonObject, ProviderHeaders, TranscriptContext};

/// Default endpoint root when the model has no base URL (the SDK's default).
const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
/// API version appended to the default or `GOOGLE_GEMINI_BASE_URL` root. A model base URL
/// already includes it.
const API_VERSION: &str = "v1beta";
const MISSING_FINISH_REASON: &str = "Google stream ended without a finish reason";

/// The `google-generative-ai` API.
#[derive(Clone, Copy, Debug, Default)]
pub struct GoogleGenerativeAiApi;

impl ProviderStreams for GoogleGenerativeAiApi {
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

/// Stream a Gemini API response. API options come from [`ApiOptions::GoogleGenerativeAi`].
pub fn stream(model: &Model, context: &TranscriptContext, options: StreamOptions) -> AssistantMessageEventStream {
    let owned_model = model.clone();
    let context = context.clone();
    let cancel = options.cancel.clone();
    spawn_stream(model, cancel, move |mut builder| async move {
        let result = run(&owned_model, &context, &options, &mut builder).await;
        (builder, result)
    })
}

/// Stream with provider-neutral options: `reasoning` and `thinking_budgets` become the
/// `thinking` option, `tool_choice` the Gemini tool choice.
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
    base.api = Some(ApiOptions::GoogleGenerativeAi(GoogleOptions {
        tool_choice: options.tool_choice.map(simple_tool_choice),
        thinking: Some(thinking),
    }));
    stream(model, context, base)
}

async fn run(
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
    builder: &mut MessageBuilder,
) -> Result<()> {
    require_auth(model, options, &["x-goog-api-key"])?;
    let google = match &options.api {
        Some(ApiOptions::GoogleGenerativeAi(google)) => google.clone(),
        _ => GoogleOptions::default(),
    };
    let request = build_request(
        model,
        context,
        &RequestSettings {
            temperature: options.temperature,
            max_tokens: options.max_tokens,
            tool_choice: google.tool_choice,
            thinking: google.thinking.as_ref(),
        },
    )?;
    let body = apply_on_payload(options, model, request_body(&request)).await?;
    let url = endpoint(model, options)?;
    let mut defaults = http::headers_of([
        ("Content-Type", "application/json"),
        ("User-Agent", http::user_agent().as_str()),
    ]);
    if let Some(key) = options.api_key.as_deref().filter(|key| !key.is_empty()) {
        defaults.insert("x-goog-api-key".into(), Some(key.to_owned()));
    }
    let headers = http::build_headers([&defaults, options.headers.as_ref().unwrap_or(&ProviderHeaders::new())])?;
    let client = http::client(options);
    let response = http::send_with_retry(options, || Ok(client.post(&url).headers(headers.clone()).json(&body)))
        .await
        .map_err(format_google_http_error)?;
    notify_response(options, model, &response).await?;
    builder.start();

    let mut state = StreamState::new();
    let mut events = Box::pin(http::sse_events(response, options.cancel.clone()));
    while let Some(event) = events.next().await {
        if builder.is_closed() {
            return Err(Error::Aborted);
        }
        let event = event?;
        let data = event.data.trim();
        // Proxies may inject the OpenAI end marker.
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let value: Value = serde_json::from_str(data)
            .map_err(|error| Error::msg(format!("Failed to parse Google stream chunk: {error}")))?;
        if let Some(error) = stream_error(&value) {
            return Err(error);
        }
        observe_provider_event(options, model, &value).await?;
        state.apply_chunk(builder, model, &parse_chunk(&value));
    }
    state.finish(builder);
    if options.is_cancelled() {
        return Err(Error::Aborted);
    }
    check_finished(&builder.output, MISSING_FINISH_REASON)
}

/// `{base}/models/{model}:streamGenerateContent?alt=sse`. The base is the model's base URL
/// (which includes the API version), else `GOOGLE_GEMINI_BASE_URL` or the default host plus
/// `v1beta`. Model ids already prefixed `models/` or `tunedModels/` are used as-is.
fn endpoint(model: &Model, options: &StreamOptions) -> Result<String> {
    let id = model.id.as_str();
    if id.is_empty() || id.contains("..") || id.contains('?') || id.contains('&') {
        return Err(Error::msg("invalid model parameter"));
    }
    let base = if model.base_url.is_empty() {
        let root = options
            .env_var("GOOGLE_GEMINI_BASE_URL")
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());
        join_url(root.trim(), API_VERSION)
    } else {
        model.base_url.clone()
    };
    let path = if id.starts_with("models/") || id.starts_with("tunedModels/") {
        id.to_owned()
    } else {
        format!("models/{id}")
    };
    Ok(format!("{}:streamGenerateContent?alt=sse", join_url(&base, &path)))
}

// ---------------------------------------------------------------------------
// REST request JSON
// ---------------------------------------------------------------------------

fn inline_data_json(data: &InlineData) -> Value {
    json!({"inlineData": {"mimeType": data.mime_type, "data": data.data}})
}

fn part_json(part: &Part) -> Value {
    let mut object = JsonObject::new();
    match part {
        Part::Text {
            text,
            thought,
            thought_signature,
        } => {
            if *thought {
                object.insert("thought".into(), Value::Bool(true));
            }
            object.insert("text".into(), Value::String(text.clone()));
            if let Some(signature) = thought_signature {
                object.insert("thoughtSignature".into(), Value::String(signature.clone()));
            }
        }
        Part::InlineData(data) => return inline_data_json(data),
        Part::FunctionCall {
            id,
            name,
            args,
            thought_signature,
        } => {
            let mut call = JsonObject::new();
            call.insert("name".into(), Value::String(name.clone()));
            call.insert("args".into(), Value::Object(args.clone()));
            if let Some(id) = id {
                call.insert("id".into(), Value::String(id.clone()));
            }
            object.insert("functionCall".into(), Value::Object(call));
            if let Some(signature) = thought_signature {
                object.insert("thoughtSignature".into(), Value::String(signature.clone()));
            }
        }
        Part::FunctionResponse {
            id,
            name,
            response,
            parts,
        } => {
            let mut function_response = JsonObject::new();
            function_response.insert("name".into(), Value::String(name.clone()));
            function_response.insert("response".into(), Value::Object(response.clone()));
            if !parts.is_empty() {
                function_response.insert("parts".into(), parts.iter().map(inline_data_json).collect());
            }
            if let Some(id) = id {
                function_response.insert("id".into(), Value::String(id.clone()));
            }
            object.insert("functionResponse".into(), Value::Object(function_response));
        }
    }
    Value::Object(object)
}

fn content_json(content: &Content) -> Value {
    json!({
        "role": content.role.as_str(),
        "parts": content.parts.iter().map(part_json).collect::<Vec<_>>(),
    })
}

fn declaration_json(declaration: &FunctionDeclaration) -> Value {
    let mut object = JsonObject::new();
    object.insert("name".into(), Value::String(declaration.name.clone()));
    object.insert("description".into(), Value::String(declaration.description.clone()));
    match &declaration.parameters {
        ToolParameters::JsonSchema(schema) => object.insert("parametersJsonSchema".into(), schema.clone()),
        ToolParameters::OpenApi(schema) => object.insert("parameters".into(), schema.clone()),
    };
    Value::Object(object)
}

fn thinking_config_json(config: &ThinkingConfig) -> Value {
    let mut object = JsonObject::new();
    if let Some(include) = config.include_thoughts {
        object.insert("includeThoughts".into(), Value::Bool(include));
    }
    if let Some(level) = &config.thinking_level {
        object.insert(
            "thinkingLevel".into(),
            Value::String(google_thinking_level_name(level).to_owned()),
        );
    }
    if let Some(budget) = config.thinking_budget {
        object.insert("thinkingBudget".into(), Value::from(budget));
    }
    Value::Object(object)
}

/// The REST `GenerateContentRequest` body, shaped as `@google/genai` sends it: the system
/// instruction as a user content, generation settings (always present, possibly empty) in
/// `generationConfig`, tools as one `functionDeclarations` entry.
pub fn request_body(request: &GenerateRequest) -> Value {
    let mut body = JsonObject::new();
    body.insert("contents".into(), request.contents.iter().map(content_json).collect());
    if let Some(system) = &request.system_instruction {
        body.insert(
            "systemInstruction".into(),
            json!({"role": "user", "parts": [{"text": system}]}),
        );
    }
    if let Some(tools) = &request.tools {
        body.insert(
            "tools".into(),
            json!([{"functionDeclarations": tools.iter().map(declaration_json).collect::<Vec<_>>()}]),
        );
    }
    if let Some(mode) = request.function_calling_mode {
        body.insert(
            "toolConfig".into(),
            json!({"functionCallingConfig": {"mode": mode.as_str()}}),
        );
    }
    let mut generation = JsonObject::new();
    if let Some(temperature) = request.temperature {
        generation.insert("temperature".into(), json!(temperature));
    }
    if let Some(max_tokens) = request.max_output_tokens {
        generation.insert("maxOutputTokens".into(), json!(max_tokens));
    }
    if let Some(config) = &request.thinking_config {
        generation.insert("thinkingConfig".into(), thinking_config_json(config));
    }
    body.insert("generationConfig".into(), Value::Object(generation));
    Value::Object(body)
}

// ---------------------------------------------------------------------------
// REST response JSON
// ---------------------------------------------------------------------------

/// An `{"error": {...}}` envelope inside the stream, worded like `@google/genai`:
/// `got status: <status>. <chunk JSON>`.
fn stream_error(value: &Value) -> Option<Error> {
    let error = value.get("error").filter(|error| !error.is_null())?;
    let status = error.get("status").map_or_else(
        || "undefined".to_owned(),
        |status| status.as_str().map_or_else(|| status.to_string(), str::to_owned),
    );
    Some(Error::msg(format!("got status: {status}. {value}")))
}

fn string_field(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_owned)
}

/// A count as a number or a numeric string (proto JSON int64).
fn count_field(value: &Value, key: &str) -> Option<u64> {
    match value.get(key)? {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

fn parse_part(value: &Value) -> ResponsePart {
    ResponsePart {
        text: string_field(value, "text"),
        thought: value.get("thought").and_then(Value::as_bool),
        thought_signature: string_field(value, "thoughtSignature"),
        function_call: value
            .get("functionCall")
            .filter(|call| call.is_object())
            .map(|call| ResponseFunctionCall {
                id: string_field(call, "id"),
                name: string_field(call, "name"),
                args: call.get("args").and_then(Value::as_object).cloned(),
            }),
    }
}

/// Parse a streamed `GenerateContentResponse` leniently: unknown fields and part kinds are
/// ignored, malformed fields count as absent.
pub fn parse_chunk(value: &Value) -> ResponseChunk {
    let candidate = value
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|candidates| candidates.first())
        .map(|candidate| Candidate {
            parts: candidate
                .pointer("/content/parts")
                .and_then(Value::as_array)
                .map(|parts| parts.iter().filter(|part| part.is_object()).map(parse_part).collect())
                .unwrap_or_default(),
            finish_reason: string_field(candidate, "finishReason"),
        });
    let usage = value
        .get("usageMetadata")
        .filter(|usage| usage.is_object())
        .map(|usage| UsageMetadata {
            prompt_token_count: count_field(usage, "promptTokenCount"),
            cached_content_token_count: count_field(usage, "cachedContentTokenCount"),
            candidates_token_count: count_field(usage, "candidatesTokenCount"),
            thoughts_token_count: count_field(usage, "thoughtsTokenCount"),
            total_token_count: count_field(usage, "totalTokenCount"),
        });
    ResponseChunk {
        response_id: string_field(value, "responseId"),
        candidate,
        usage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::google_shared::{FunctionCallingMode, Role};
    use crate::options::GoogleThinkingLevel;
    use pretty_assertions::assert_eq;

    fn model(base_url: &str, id: &str) -> Model {
        serde_json::from_value(
            json!({"id": id, "name": id, "api": "google-generative-ai", "provider": "google",
            "baseUrl": base_url, "reasoning": true, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 1000, "maxTokens": 100}),
        )
        .unwrap()
    }

    #[test]
    fn endpoints() {
        let options = StreamOptions::default();
        assert_eq!(
            endpoint(
                &model("https://generativelanguage.googleapis.com/v1beta", "gemini-2.5-flash"),
                &options
            )
            .unwrap(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );
        assert_eq!(
            endpoint(&model("https://proxy.example.com/gemini/", "tunedModels/x"), &options).unwrap(),
            "https://proxy.example.com/gemini/tunedModels/x:streamGenerateContent?alt=sse"
        );
        let env = StreamOptions {
            env: Some([("GOOGLE_GEMINI_BASE_URL".to_owned(), "https://env.test/".to_owned())].into()),
            ..StreamOptions::default()
        };
        assert_eq!(
            endpoint(&model("", "gemini-2.5-flash"), &env).unwrap(),
            "https://env.test/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );
        assert!(endpoint(&model("https://x", "a?key=1"), &options).is_err());
    }

    #[test]
    fn request_body_shape() {
        let request = GenerateRequest {
            contents: vec![
                Content {
                    role: Role::Model,
                    parts: vec![
                        Part::Text {
                            text: "hm".into(),
                            thought: true,
                            thought_signature: Some("c2ln".into()),
                        },
                        Part::FunctionCall {
                            id: Some("c1".into()),
                            name: "read".into(),
                            args: JsonObject::new(),
                            thought_signature: None,
                        },
                    ],
                },
                Content {
                    role: Role::User,
                    parts: vec![Part::FunctionResponse {
                        id: None,
                        name: "read".into(),
                        response: json!({"output": "x"}).as_object().unwrap().clone(),
                        parts: vec![InlineData {
                            mime_type: "image/png".into(),
                            data: "AA==".into(),
                        }],
                    }],
                },
            ],
            system_instruction: Some("sys".into()),
            temperature: Some(0.5),
            max_output_tokens: Some(64),
            tools: Some(vec![FunctionDeclaration {
                name: "read".into(),
                description: "Read".into(),
                parameters: ToolParameters::JsonSchema(json!({"type": "object"})),
            }]),
            function_calling_mode: Some(FunctionCallingMode::Any),
            thinking_config: Some(ThinkingConfig {
                include_thoughts: Some(true),
                thinking_budget: None,
                thinking_level: Some(GoogleThinkingLevel::Other("THINKING_LEVEL_UNSPECIFIED".into())),
            }),
        };
        assert_eq!(
            request_body(&request),
            json!({
                "contents": [
                    {"role": "model", "parts": [
                        {"thought": true, "text": "hm", "thoughtSignature": "c2ln"},
                        {"functionCall": {"name": "read", "args": {}, "id": "c1"}}
                    ]},
                    {"role": "user", "parts": [{"functionResponse": {"name": "read", "response": {"output": "x"},
                        "parts": [{"inlineData": {"mimeType": "image/png", "data": "AA=="}}]}}]}
                ],
                "systemInstruction": {"role": "user", "parts": [{"text": "sys"}]},
                "tools": [{"functionDeclarations": [{"name": "read", "description": "Read",
                    "parametersJsonSchema": {"type": "object"}}]}],
                "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
                "generationConfig": {"temperature": 0.5, "maxOutputTokens": 64,
                    "thinkingConfig": {"includeThoughts": true, "thinkingLevel": "THINKING_LEVEL_UNSPECIFIED"}}
            })
        );
        assert_eq!(
            request_body(&GenerateRequest::default()),
            json!({"contents": [], "generationConfig": {}})
        );
    }

    #[test]
    fn parses_chunks_leniently() {
        let chunk = parse_chunk(&json!({
            "responseId": "r1",
            "modelVersion": "gemini-x",
            "candidates": [{"content": {"role": "model", "parts": [
                {"text": "a", "thought": true, "thoughtSignature": "c2ln"},
                {"executableCode": {"code": "print(1)"}},
                "junk",
                {"functionCall": {"name": "f", "args": {"x": 1}}},
                {"text": 5}
            ]}, "finishReason": "STOP", "safetyRatings": []}, {"content": {"parts": [{"text": "ignored"}]}}],
            "usageMetadata": {"promptTokenCount": 3, "totalTokenCount": "7", "promptTokensDetails": []}
        }));
        let candidate = chunk.candidate.unwrap();
        assert_eq!(chunk.response_id.as_deref(), Some("r1"));
        assert_eq!(candidate.parts.len(), 4);
        assert_eq!(candidate.parts[0].thought_signature.as_deref(), Some("c2ln"));
        assert_eq!(candidate.parts[1], ResponsePart::default());
        assert_eq!(
            candidate.parts[2].function_call.as_ref().unwrap().name.as_deref(),
            Some("f")
        );
        assert_eq!(candidate.parts[3].text, None);
        assert_eq!(candidate.finish_reason.as_deref(), Some("STOP"));
        let usage = chunk.usage.unwrap();
        assert_eq!((usage.prompt_token_count, usage.total_token_count), (Some(3), Some(7)));
        assert_eq!(parse_chunk(&json!({})), ResponseChunk::default());
    }

    #[test]
    fn stream_error_envelopes() {
        let value = json!({"error": {"code": 429, "message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}});
        assert_eq!(
            stream_error(&value).unwrap().to_string(),
            format!("got status: RESOURCE_EXHAUSTED. {value}")
        );
        assert!(stream_error(&json!({"error": null, "candidates": []})).is_none());
        assert_eq!(
            stream_error(&json!({"error": {"code": 500}})).unwrap().to_string(),
            r#"got status: undefined. {"error":{"code":500}}"#
        );
    }
}
