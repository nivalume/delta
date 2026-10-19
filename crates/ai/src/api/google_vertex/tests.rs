use super::*;
use crate::api::google_shared::{
    Content, FunctionCallingMode, FunctionDeclaration, GenerateRequest, InlineData, Part, Role, ThinkingConfig,
    ToolParameters,
};
use crate::options::GoogleThinkingLevel;
use crate::types::JsonObject;
use bytes::Bytes;
use google_cloud_aiplatform_v1::model::function_calling_config::Mode;
use google_cloud_aiplatform_v1::model::generate_content_response::UsageMetadata as ProtoUsage;
use google_cloud_aiplatform_v1::model::generation_config::thinking_config::ThinkingLevel as ProtoThinkingLevel;
use google_cloud_aiplatform_v1::model::{Candidate as ProtoCandidate, Content as ProtoContent, FunctionCall};
use google_cloud_gax::error::rpc::{Code, Status};
use pretty_assertions::assert_eq;
use serde_json::json;
use std::collections::BTreeMap;

fn model(id: &str, base_url: &str) -> Model {
    serde_json::from_value(json!({
        "id": id, "name": id, "api": "google-vertex", "provider": "google-vertex",
        "baseUrl": base_url, "reasoning": true, "input": ["text", "image"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 1000, "maxTokens": 100
    }))
    .unwrap()
}

const PLACEHOLDER_URL: &str = "https://{location}-aiplatform.googleapis.com";

fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
    let map: BTreeMap<String, String> = pairs.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect();
    move |name| map.get(name).cloned()
}

fn vertex(project: Option<&str>, location: Option<&str>) -> GoogleVertexOptions {
    GoogleVertexOptions {
        project: project.map(str::to_owned),
        location: location.map(str::to_owned),
        ..GoogleVertexOptions::default()
    }
}

// ---------------------------------------------------------------------------
// Authentication, endpoints and model names (google-vertex-api-key-resolution.test.ts)
// ---------------------------------------------------------------------------

#[test]
fn api_key_markers_fall_back_to_adc() {
    assert_eq!(resolve_api_key(None), None);
    assert_eq!(resolve_api_key(Some("")), None);
    assert_eq!(resolve_api_key(Some("   ")), None);
    assert_eq!(resolve_api_key(Some("gcp-vertex-credentials")), None);
    assert_eq!(resolve_api_key(Some("<authenticated>")), None);
    assert_eq!(resolve_api_key(Some(" <authenticated> ")), None);
    assert_eq!(resolve_api_key(Some("<>")).as_deref(), Some("<>"));
    assert_eq!(resolve_api_key(Some("<a>b>")).as_deref(), Some("<a>b>"));
    assert_eq!(resolve_api_key(Some(" AIzaKey ")).as_deref(), Some("AIzaKey"));
}

#[test]
fn model_paths() {
    assert_eq!(
        model_path("gemini-2.5-flash").unwrap(),
        "publishers/google/models/gemini-2.5-flash"
    );
    assert_eq!(
        model_path("anthropic/claude-x").unwrap(),
        "publishers/anthropic/models/claude-x"
    );
    assert_eq!(model_path("a/b/c").unwrap(), "publishers/a/models/b");
    for kept in [
        "publishers/x/models/y",
        "projects/p/locations/l/endpoints/e",
        "models/m",
    ] {
        assert_eq!(model_path(kept).unwrap(), kept);
    }
    assert_eq!(model_path("a..b").unwrap_err().to_string(), "invalid model parameter");
    assert!(model_path("a?b").is_err());
    assert!(model_path("a&b").is_err());
    assert!(model_path("").is_err());
}

#[test]
fn api_key_target_uses_the_global_endpoint_without_project() {
    let target = resolve_target(
        &model("gemini-3-flash-preview", PLACEHOLDER_URL),
        &vertex(Some("p"), Some("us-central1")),
        Some("AIzaSyExampleRealisticLookingApiKey123456"),
        &env_of(&[]),
    )
    .unwrap();
    assert_eq!(
        target,
        VertexTarget {
            auth: VertexAuth::ApiKey("AIzaSyExampleRealisticLookingApiKey123456".into()),
            endpoint: "https://aiplatform.googleapis.com".into(),
            model: "publishers/google/models/gemini-3-flash-preview".into(),
        }
    );
}

#[test]
fn adc_targets_by_location() {
    for (location, endpoint) in [
        ("us-central1", "https://us-central1-aiplatform.googleapis.com"),
        ("europe-west4", "https://europe-west4-aiplatform.googleapis.com"),
        ("global", "https://aiplatform.googleapis.com"),
        ("us", "https://aiplatform.us.rep.googleapis.com"),
        ("eu", "https://aiplatform.eu.rep.googleapis.com"),
    ] {
        for api_key in [None, Some("<authenticated>"), Some("gcp-vertex-credentials")] {
            let target = resolve_target(
                &model("gemini-3-flash-preview", PLACEHOLDER_URL),
                &vertex(Some("test-project"), Some(location)),
                api_key,
                &env_of(&[]),
            )
            .unwrap();
            assert_eq!(target.auth, VertexAuth::Adc { credentials_file: None });
            assert_eq!(target.endpoint, endpoint);
            assert_eq!(
                target.model,
                format!("projects/test-project/locations/{location}/publishers/google/models/gemini-3-flash-preview")
            );
        }
    }
}

#[test]
fn project_location_and_credentials_file_from_env() {
    let env = env_of(&[
        ("GCLOUD_PROJECT", "gcloud-project"),
        ("GOOGLE_CLOUD_LOCATION", "asia-east1"),
        ("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/sa.json"),
    ]);
    let target = resolve_target(&model("gemini-2.5-pro", ""), &vertex(None, Some(" ")), None, &env).unwrap();
    assert_eq!(
        target.auth,
        VertexAuth::Adc {
            credentials_file: Some("/tmp/sa.json".into())
        }
    );
    assert_eq!(target.endpoint, "https://asia-east1-aiplatform.googleapis.com");
    assert_eq!(
        target.model,
        "projects/gcloud-project/locations/asia-east1/publishers/google/models/gemini-2.5-pro"
    );
    // GOOGLE_CLOUD_PROJECT wins over GCLOUD_PROJECT; explicit options win over both.
    let env = env_of(&[
        ("GOOGLE_CLOUD_PROJECT", "cloud-project"),
        ("GCLOUD_PROJECT", "gcloud-project"),
        ("GOOGLE_CLOUD_LOCATION", "asia-east1"),
    ]);
    let target = resolve_target(&model("m", ""), &vertex(None, None), None, &env).unwrap();
    assert!(target.model.starts_with("projects/cloud-project/locations/asia-east1/"));
    let target = resolve_target(&model("m", ""), &vertex(Some("opt"), Some("us-east5")), None, &env).unwrap();
    assert!(target.model.starts_with("projects/opt/locations/us-east5/"));
    // A full resource name is used as-is.
    let target = resolve_target(
        &model("projects/x/locations/y/endpoints/z", ""),
        &vertex(Some("p"), Some("l")),
        None,
        &env_of(&[]),
    )
    .unwrap();
    assert_eq!(target.model, "projects/x/locations/y/endpoints/z");
}

#[test]
fn missing_project_or_location() {
    let m = model("gemini-2.5-flash", PLACEHOLDER_URL);
    assert_eq!(
        resolve_target(&m, &vertex(None, None), None, &env_of(&[]))
            .unwrap_err()
            .to_string(),
        "Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options."
    );
    assert_eq!(
        resolve_target(&m, &vertex(Some("p"), None), Some("<authenticated>"), &env_of(&[]))
            .unwrap_err()
            .to_string(),
        "Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options."
    );
}

#[test]
fn custom_base_urls() {
    let opts = vertex(Some("test-project"), Some("us-central1"));
    let target = resolve_target(&model("m", "https://proxy.example.com"), &opts, None, &env_of(&[])).unwrap();
    assert_eq!(target.endpoint, "https://proxy.example.com");
    assert_eq!(
        target.model,
        "projects/test-project/locations/us-central1/publishers/google/models/m"
    );
    let target = resolve_target(
        &model("m", "https://proxy.example.com"),
        &opts,
        Some("AIzaKey"),
        &env_of(&[]),
    )
    .unwrap();
    assert_eq!(target.endpoint, "https://proxy.example.com");
    assert_eq!(target.model, "publishers/google/models/m");
    // The path (API version, collection) is not part of a gRPC endpoint.
    let target = resolve_target(
        &model(
            "m",
            "https://proxy.example.com:8443/v1/projects/test-project/locations/global",
        ),
        &opts,
        None,
        &env_of(&[]),
    )
    .unwrap();
    assert_eq!(target.endpoint, "https://proxy.example.com:8443");
    let target = resolve_target(&model("m", "http://localhost:9000/"), &opts, None, &env_of(&[])).unwrap();
    assert_eq!(target.endpoint, "http://localhost:9000");
    // GOOGLE_VERTEX_BASE_URL applies when the model has no usable base URL.
    let env = env_of(&[("GOOGLE_VERTEX_BASE_URL", "https://env.example.com/")]);
    let target = resolve_target(&model("m", PLACEHOLDER_URL), &opts, None, &env).unwrap();
    assert_eq!(target.endpoint, "https://env.example.com");
    let target = resolve_target(&model("m", "https://model.example.com"), &opts, None, &env).unwrap();
    assert_eq!(target.endpoint, "https://model.example.com");
}

fn temp_file(name: &str, contents: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("tau-vertex-{}-{name}", std::process::id()));
    std::fs::write(&path, contents).unwrap();
    path
}

#[test]
fn credentials_files() {
    // Valid files are not built here: token-based credentials start refreshing at once.
    let unknown = temp_file("unknown.json", r#"{"type": "something_else"}"#);
    let path = unknown.to_str().unwrap();
    assert_eq!(
        credentials_from_file(path).unwrap_err().to_string(),
        format!("Unsupported Google credentials type in {path}: something_else")
    );
    let invalid = temp_file("invalid.json", "not json");
    assert!(
        credentials_from_file(invalid.to_str().unwrap())
            .unwrap_err()
            .to_string()
            .starts_with("Failed to parse Google credentials file")
    );
    assert!(
        credentials_from_file("/nonexistent/tau-vertex.json")
            .unwrap_err()
            .to_string()
            .starts_with("Failed to read Google credentials file /nonexistent/tau-vertex.json")
    );
    for path in [unknown, invalid] {
        let _ = std::fs::remove_file(path);
    }
}

#[tokio::test]
async fn api_key_credentials_send_the_key() {
    // API key credentials need no token exchange.
    let credentials = credentials(&VertexAuth::ApiKey("AIzaKey".into())).unwrap();
    let headers = credentials.headers(::http::Extensions::new()).await.unwrap();
    let google_cloud_auth::credentials::CacheableResource::New { data, .. } = headers else {
        panic!("expected new headers");
    };
    assert_eq!(data.get("x-goog-api-key").unwrap(), "AIzaKey");
}

// ---------------------------------------------------------------------------
// Thinking budgets
// ---------------------------------------------------------------------------

#[test]
fn vertex_budget_table() {
    use ResolvedGoogleThinkingLevel::{High, Low, Medium, Minimal};
    let budgets = |id: &str| {
        let m = model(id, "");
        [Minimal, Low, Medium, High].map(|level| get_google_budget(&m, level, None))
    };
    assert_eq!(budgets("gemini-2.5-pro"), [128, 2048, 8192, 32768]);
    assert_eq!(budgets("gemini-2.5-flash"), [128, 2048, 8192, 24576]);
    // No Flash-Lite row on Vertex (unlike the Gemini API).
    assert_eq!(budgets("gemini-2.5-flash-lite"), [128, 2048, 8192, 24576]);
    assert_eq!(budgets("gemini-2.0-flash"), [-1; 4]);
    let custom = ThinkingBudgets {
        low: Some(77),
        ..ThinkingBudgets::default()
    };
    assert_eq!(get_google_budget(&model("gemini-2.5-pro", ""), Low, Some(&custom)), 77);
    assert_eq!(
        get_google_budget(&model("gemini-2.5-pro", ""), High, Some(&custom)),
        32768
    );
}

#[test]
fn simple_thinking_options() {
    let flash = model("gemini-2.5-flash-lite", "");
    assert_eq!(
        simple_thinking(&flash, Some(ThinkingLevel::Minimal), None).unwrap(),
        GoogleThinking {
            enabled: true,
            budget_tokens: Some(128),
            level: None
        }
    );
    assert_eq!(
        simple_thinking(&flash, None, None).unwrap(),
        GoogleThinking {
            enabled: false,
            budget_tokens: None,
            level: None
        }
    );
    let gemini3 = model("gemini-3-pro-preview", "");
    assert_eq!(
        simple_thinking(&gemini3, Some(ThinkingLevel::Medium), None).unwrap(),
        GoogleThinking {
            enabled: true,
            budget_tokens: None,
            level: Some(GoogleThinkingLevel::Medium)
        }
    );
}

// ---------------------------------------------------------------------------
// Request conversion
// ---------------------------------------------------------------------------

fn object(value: Value) -> JsonObject {
    match value {
        Value::Object(object) => object,
        other => panic!("not an object: {other}"),
    }
}

#[test]
fn request_converts_to_proto() {
    let request = GenerateRequest {
        contents: vec![
            Content {
                role: Role::User,
                parts: vec![
                    Part::Text {
                        text: "look".into(),
                        thought: false,
                        thought_signature: None,
                    },
                    Part::InlineData(InlineData {
                        mime_type: "image/png".into(),
                        data: "aW1n".into(),
                    }),
                ],
            },
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
                        args: object(json!({"path": "a"})),
                        thought_signature: Some("dG9vbA==".into()),
                    },
                ],
            },
            Content {
                role: Role::User,
                parts: vec![Part::FunctionResponse {
                    id: Some("c1".into()),
                    name: "read".into(),
                    response: object(json!({"output": "x"})),
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
            parameters: ToolParameters::JsonSchema(
                json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            ),
        }]),
        function_calling_mode: Some(FunctionCallingMode::Validated),
        thinking_config: Some(ThinkingConfig {
            include_thoughts: Some(true),
            thinking_budget: None,
            thinking_level: Some(GoogleThinkingLevel::High),
        }),
    };
    let proto = proto_request(
        request_body(&request),
        "projects/p/locations/l/publishers/google/models/m",
    )
    .unwrap();
    assert_eq!(proto.model, "projects/p/locations/l/publishers/google/models/m");

    let system = proto.system_instruction.as_ref().unwrap();
    assert_eq!(system.role, "user");
    assert_eq!(system.parts[0].text().map(String::as_str), Some("sys"));

    assert_eq!(proto.contents.len(), 3);
    let user = &proto.contents[0];
    assert_eq!(user.role, "user");
    assert_eq!(user.parts[0].text().map(String::as_str), Some("look"));
    let image = user.parts[1].inline_data().unwrap();
    assert_eq!(
        (image.mime_type.as_str(), image.data.as_ref()),
        ("image/png", b"img".as_slice())
    );

    let assistant = &proto.contents[1];
    assert_eq!(assistant.role, "model");
    assert!(assistant.parts[0].thought);
    assert_eq!(assistant.parts[0].thought_signature.as_ref(), b"sig");
    let call = assistant.parts[1].function_call().unwrap();
    assert_eq!(call.name, "read");
    assert_eq!(call.args, Some(object(json!({"path": "a"}))));
    assert_eq!(assistant.parts[1].thought_signature.as_ref(), b"tool");
    assert!(!assistant.parts[1].thought);

    let response = proto.contents[2].parts[0].function_response().unwrap();
    assert_eq!(response.name, "read");
    assert_eq!(response.response, Some(object(json!({"output": "x"}))));
    let nested = response.parts[0].inline_data().unwrap();
    assert_eq!(
        (nested.mime_type.as_str(), nested.data.as_ref()),
        ("image/png", [0u8].as_slice())
    );

    let declaration = &proto.tools[0].function_declarations[0];
    assert_eq!(
        (declaration.name.as_str(), declaration.description.as_str()),
        ("read", "Read")
    );
    assert_eq!(
        declaration.parameters_json_schema,
        Some(json!({"type": "object", "properties": {"path": {"type": "string"}}}))
    );
    assert!(declaration.parameters.is_none());
    let mode = &proto
        .tool_config
        .as_ref()
        .unwrap()
        .function_calling_config
        .as_ref()
        .unwrap()
        .mode;
    assert_eq!(*mode, Mode::Validated);

    let generation = proto.generation_config.as_ref().unwrap();
    assert_eq!(generation.temperature, Some(0.5));
    assert_eq!(generation.max_output_tokens, Some(64));
    let thinking = generation.thinking_config.as_ref().unwrap();
    assert_eq!(thinking.include_thoughts, Some(true));
    assert_eq!(thinking.thinking_level, Some(ProtoThinkingLevel::High));
    assert_eq!(thinking.thinking_budget, None);
}

#[test]
fn request_payload_edge_cases() {
    // An empty request still has a generation config, as upstream's SDK body.
    let proto = proto_request(request_body(&GenerateRequest::default()), "m").unwrap();
    assert!(proto.contents.is_empty());
    assert!(proto.generation_config.is_some());
    assert!(proto.tool_config.is_none());

    // on_payload may add proto fields, set the model, and use budgets and modes by name.
    let proto = proto_request(
        json!({
            "model": "projects/o/locations/l/publishers/google/models/other",
            "contents": [],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
            "generationConfig": {"topK": 3, "thinkingConfig": {"thinkingBudget": -1}},
            "labels": {"team": "tau"}
        }),
        "projects/p/locations/l/publishers/google/models/m",
    )
    .unwrap();
    assert_eq!(proto.model, "projects/o/locations/l/publishers/google/models/other");
    let generation = proto.generation_config.unwrap();
    assert_eq!(generation.top_k, Some(3.0));
    assert_eq!(generation.thinking_config.unwrap().thinking_budget, Some(-1));
    assert_eq!(proto.labels.get("team").map(String::as_str), Some("tau"));
    assert_eq!(
        proto.tool_config.unwrap().function_calling_config.unwrap().mode,
        Mode::Any
    );

    let error = proto_request(json!({"contents": "nope"}), "m").unwrap_err().to_string();
    assert!(error.starts_with("Invalid Google Vertex request payload: "), "{error}");
    let error = proto_request(json!({"generationConfig": {"maxOutputTokens": 1u64 << 40}}), "m")
        .unwrap_err()
        .to_string();
    assert!(error.starts_with("Invalid Google Vertex request payload: "), "{error}");
}

// ---------------------------------------------------------------------------
// Response conversion
// ---------------------------------------------------------------------------

fn candidate(parts: Vec<ProtoPart>, finish_reason: FinishReason) -> ProtoCandidate {
    ProtoCandidate::new()
        .set_content(ProtoContent::new().set_role("model").set_parts(parts))
        .set_finish_reason(finish_reason)
}

#[test]
fn response_chunks_from_proto() {
    let mut args = JsonObject::new();
    args.insert("x".into(), json!(1));
    let response = GenerateContentResponse::new()
        .set_response_id("resp_1")
        .set_model_version("gemini-x")
        .set_candidates([
            candidate(
                vec![
                    ProtoPart::new()
                        .set_text("a")
                        .set_thought(true)
                        .set_thought_signature(Bytes::from_static(b"sig")),
                    ProtoPart::new().set_text(""),
                    ProtoPart::new().set_function_call(FunctionCall::new().set_name("f").set_args(args.clone())),
                    ProtoPart::new().set_function_call(FunctionCall::new()),
                    ProtoPart::new(),
                ],
                FinishReason::Stop,
            ),
            candidate(vec![ProtoPart::new().set_text("ignored")], FinishReason::Safety),
        ])
        .set_usage_metadata(
            ProtoUsage::new()
                .set_prompt_token_count(10)
                .set_cached_content_token_count(4)
                .set_candidates_token_count(5)
                .set_thoughts_token_count(0)
                .set_total_token_count(15),
        );
    let chunk = response_chunk(&response);
    assert_eq!(chunk.response_id.as_deref(), Some("resp_1"));
    let first = chunk.candidate.unwrap();
    assert_eq!(first.finish_reason.as_deref(), Some("STOP"));
    assert_eq!(
        first.parts,
        vec![
            ResponsePart {
                text: Some("a".into()),
                thought: Some(true),
                thought_signature: Some("c2ln".into()),
                function_call: None,
            },
            ResponsePart {
                text: Some(String::new()),
                ..ResponsePart::default()
            },
            ResponsePart {
                function_call: Some(ResponseFunctionCall {
                    id: None,
                    name: Some("f".into()),
                    args: Some(args),
                }),
                ..ResponsePart::default()
            },
            ResponsePart {
                function_call: Some(ResponseFunctionCall::default()),
                ..ResponsePart::default()
            },
            ResponsePart::default(),
        ]
    );
    assert_eq!(
        chunk.usage,
        Some(UsageMetadata {
            prompt_token_count: Some(10),
            cached_content_token_count: Some(4),
            candidates_token_count: Some(5),
            thoughts_token_count: None,
            total_token_count: Some(15),
        })
    );

    // Defaults are absent fields, as in REST JSON.
    assert_eq!(
        response_chunk(&GenerateContentResponse::new()),
        ResponseChunk::default()
    );
    let chunk = response_chunk(
        &GenerateContentResponse::new()
            .set_candidates([ProtoCandidate::new()])
            .set_usage_metadata(ProtoUsage::new()),
    );
    assert_eq!(chunk.candidate, Some(Candidate::default()));
    assert_eq!(chunk.usage, Some(UsageMetadata::default()));
}

#[test]
fn finish_reason_names() {
    assert_eq!(finish_reason_name(&FinishReason::Unspecified), None);
    assert_eq!(
        finish_reason_name(&FinishReason::MaxTokens).as_deref(),
        Some("MAX_TOKENS")
    );
    assert_eq!(
        finish_reason_name(&FinishReason::MalformedFunctionCall).as_deref(),
        Some("MALFORMED_FUNCTION_CALL")
    );
    assert_eq!(finish_reason_name(&FinishReason::from(42)).as_deref(), Some("42"));
    assert_eq!(
        finish_reason_name(&FinishReason::from("SOMETHING_NEW")).as_deref(),
        Some("SOMETHING_NEW")
    );
}

#[test]
fn provider_event_json_uses_rest_names() {
    let response = GenerateContentResponse::new()
        .set_response_id("r")
        .set_candidates([candidate(
            vec![
                ProtoPart::new()
                    .set_text("hi")
                    .set_thought_signature(Bytes::from_static(b"sig")),
            ],
            FinishReason::Stop,
        )]);
    assert_eq!(
        response_json(&response),
        json!({
            "candidates": [{"content": {"role": "model", "parts": [{"text": "hi", "thoughtSignature": "c2ln"}]},
                "finishReason": "STOP"}],
            "responseId": "r"
        })
    );
    let pending = GenerateContentResponse::new().set_candidates([ProtoCandidate::new()]);
    assert_eq!(response_json(&pending), json!({"candidates": [{}]}));
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

fn service_error(code: Code, message: &str) -> google_cloud_aiplatform_v1::Error {
    google_cloud_aiplatform_v1::Error::service(Status::default().set_code(code).set_message(message))
}

#[test]
fn sdk_errors_map_to_http_statuses() {
    for (code, status) in [
        (Code::ResourceExhausted, 429),
        (Code::Unavailable, 503),
        (Code::InvalidArgument, 400),
        (Code::PermissionDenied, 403),
        (Code::Unauthenticated, 401),
        (Code::NotFound, 404),
        (Code::DeadlineExceeded, 504),
        (Code::Internal, 500),
        (Code::Aborted, 409),
    ] {
        match vertex_error(&service_error(code, "boom")) {
            Error::Status {
                status: actual,
                message,
                retry_after_ms,
            } => {
                assert_eq!(actual, status, "{code:?}");
                assert_eq!(message, format!("{status} boom"));
                assert_eq!(retry_after_ms, None);
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(
        vertex_error(&service_error(Code::Unavailable, "")).to_string(),
        "503 UNAVAILABLE"
    );
    let timeout = google_cloud_aiplatform_v1::Error::timeout("deadline");
    assert!(matches!(vertex_error(&timeout), Error::Message(message) if message.contains("deadline")));
    let http_error = google_cloud_aiplatform_v1::Error::http(502, ::http::HeaderMap::new(), Bytes::from_static(b"bad"));
    assert!(matches!(vertex_error(&http_error), Error::Status { status: 502, .. }));
}
