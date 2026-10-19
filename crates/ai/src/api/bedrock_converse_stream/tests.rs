use super::*;
use crate::events::AssistantMessageEventStream;
use crate::transcript::normalize_context;
use crate::types::{
    AssistantMessage, Context, ImageContent, SystemMessage, TextContent, ThinkingContent, ToolCall, ToolResultMessage,
    Usage, UserMessage,
};
use crate::types::{ConstrainedSampling, StrictMode};
use futures::StreamExt;
use pretty_assertions::assert_eq;
use std::collections::BTreeMap;

const CATALOG: &str = include_str!("../../../data/providers/amazon-bedrock.json");

fn catalog_model(id: &str) -> Model {
    let catalog: Value = serde_json::from_str(CATALOG).unwrap();
    let entry = catalog["bedrock-converse-stream"][format!("chat:{id}")].clone();
    assert!(!entry.is_null(), "{id} is not in the catalog");
    serde_json::from_value(entry).unwrap()
}

fn sonnet_45() -> Model {
    let mut model = catalog_model("us.anthropic.claude-sonnet-4-5-20250929-v1:0");
    model.input = vec![crate::model::InputModality::Text, crate::model::InputModality::Image];
    model
}

fn nova() -> Model {
    let mut model = sonnet_45();
    model.id = "amazon.nova-lite-v1:0".into();
    model.name = "Nova Lite".into();
    model.reasoning = false;
    model.compat = None;
    model
}

fn gpt_56() -> Model {
    let mut model = catalog_model("global.openai.gpt-5.6-sol");
    model.id = "global.openai.gpt-5.6-terra".into();
    model.name = "GPT-5.6 Terra (Global)".into();
    model
}

fn no_process_env(_: &str) -> Option<String> {
    None
}

fn env_of(pairs: &[(&str, &str)]) -> ProviderEnv {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect()
}

fn lookup<'a>(scoped: Option<&'a ProviderEnv>, process: &'a (dyn Fn(&str) -> Option<String> + Sync)) -> EnvLookup<'a> {
    EnvLookup { scoped, process }
}

fn assistant(model: &Model, content: Vec<AssistantBlock>, stop_reason: StopReason) -> Message {
    Message::Assistant(AssistantMessage {
        content,
        stop_reason,
        usage: Usage::default(),
        ..AssistantMessage::empty("bedrock-converse-stream", "amazon-bedrock", &model.id)
    })
}

fn tool_result(id: &str, name: &str, content: Vec<UserBlock>, is_error: bool) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: id.into(),
        tool_name: name.into(),
        content,
        is_error,
        ..ToolResultMessage::default()
    })
}

fn image(data: &str) -> UserBlock {
    UserBlock::Image(ImageContent {
        data: data.into(),
        mime_type: "image/png".into(),
    })
}

fn payload_with(model: &Model, context: &Context, options: &StreamOptions) -> Value {
    let transcript = collapse_system_messages(&normalize_context(context));
    let bedrock = bedrock_options(options);
    let env = lookup(options.env.as_ref(), &no_process_env);
    build_payload(model, &transcript, options, &bedrock, &env).unwrap()
}

fn payload(model: &Model, messages: Vec<Message>) -> Value {
    let options = StreamOptions {
        cache_retention: Some(CacheRetention::None),
        ..StreamOptions::default()
    };
    payload_with(model, &Context::new(messages), &options)
}

fn reasoning_payload(model: &Model, bedrock: BedrockOptions) -> Value {
    let options = StreamOptions {
        cache_retention: Some(CacheRetention::None),
        api: Some(ApiOptions::BedrockConverseStream(BedrockOptions {
            reasoning: bedrock.reasoning.or(Some(ThinkingLevel::High)),
            ..bedrock
        })),
        ..StreamOptions::default()
    };
    payload_with(model, &Context::new(vec![Message::user("Hello")]), &options)
}

// ---------------------------------------------------------------------------------------------
// Client configuration
// ---------------------------------------------------------------------------------------------

fn plan_for(model: &Model, bedrock: &BedrockOptions, scoped: &[(&str, &str)], process: &[(&str, &str)]) -> ClientPlan {
    let scoped = env_of(scoped);
    let process: BTreeMap<String, String> = env_of(process);
    let process_fn = move |name: &str| process.get(name).cloned();
    let env = lookup(Some(&scoped), &process_fn);
    resolve_client_plan(model, bedrock, None, &env)
}

#[test]
fn explicit_and_scoped_profiles_win_over_ambient_access_keys() {
    let model = catalog_model("us.anthropic.claude-opus-4-8");
    let keys = [
        ("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE"),
        ("AWS_SECRET_ACCESS_KEY", "secretexample"),
    ];
    let explicit = BedrockOptions {
        profile: Some("explicit-profile".into()),
        ..BedrockOptions::default()
    };
    let plan = plan_for(&model, &explicit, &[], &keys);
    assert_eq!(plan.profile.as_deref(), Some("explicit-profile"));
    assert_eq!(plan.credentials, None);

    let plan = plan_for(
        &model,
        &BedrockOptions::default(),
        &[("AWS_PROFILE", "scoped-profile")],
        &keys,
    );
    assert_eq!(plan.profile.as_deref(), Some("scoped-profile"));
    assert_eq!(plan.credentials, None);
}

#[test]
fn ambient_access_keys_are_used_without_a_configured_profile() {
    let model = catalog_model("us.anthropic.claude-opus-4-8");
    let keys = [
        ("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE"),
        ("AWS_SECRET_ACCESS_KEY", "secretexample"),
    ];
    let expected = Some(StaticCredentials {
        access_key_id: "AKIAEXAMPLE".into(),
        secret_access_key: "secretexample".into(),
        session_token: None,
    });
    let plan = plan_for(&model, &BedrockOptions::default(), &[], &keys);
    assert_eq!(plan.profile, None);
    assert_eq!(plan.credentials, expected);

    // Only an ambient profile: the keys are still passed explicitly.
    let mut with_profile = keys.to_vec();
    with_profile.push(("AWS_PROFILE", "ambient-profile"));
    let plan = plan_for(&model, &BedrockOptions::default(), &[], &with_profile);
    assert_eq!(plan.profile.as_deref(), Some("ambient-profile"));
    assert_eq!(plan.credentials, expected);
}

#[test]
fn scoped_env_credentials_and_session_token() {
    let model = catalog_model("us.anthropic.claude-opus-4-8");
    let plan = plan_for(
        &model,
        &BedrockOptions::default(),
        &[
            ("AWS_ACCESS_KEY_ID", "scoped"),
            ("AWS_SECRET_ACCESS_KEY", "secret"),
            ("AWS_SESSION_TOKEN", "session"),
        ],
        &[("AWS_ACCESS_KEY_ID", "ambient"), ("AWS_SECRET_ACCESS_KEY", "ambient")],
    );
    assert_eq!(
        plan.credentials,
        Some(StaticCredentials {
            access_key_id: "scoped".into(),
            secret_access_key: "secret".into(),
            session_token: Some("session".into()),
        })
    );
}

#[test]
fn standard_endpoints_are_not_pinned_when_a_region_is_configured() {
    let model = catalog_model("us.anthropic.claude-opus-4-8");
    let plan = plan_for(&model, &BedrockOptions::default(), &[], &[("AWS_REGION", "us-east-2")]);
    assert_eq!(plan.region.as_deref(), Some("us-east-2"));
    assert_eq!(plan.endpoint, None);
}

#[test]
fn region_is_derived_from_a_standard_endpoint_without_region_or_profile() {
    let model = catalog_model("eu.anthropic.claude-sonnet-4-5-20250929-v1:0");
    assert_eq!(model.base_url, "https://bedrock-runtime.eu-central-1.amazonaws.com");
    let plan = plan_for(&model, &BedrockOptions::default(), &[], &[]);
    assert_eq!(plan.endpoint.as_deref(), Some(model.base_url.as_str()));
    assert_eq!(plan.region.as_deref(), Some("eu-central-1"));
}

#[test]
fn missing_regions_with_explicit_scoped_and_ambient_profiles() {
    let model = catalog_model("eu.anthropic.claude-sonnet-4-5-20250929-v1:0");
    let explicit = BedrockOptions {
        profile: Some("bedrock-profile".into()),
        ..BedrockOptions::default()
    };
    let plan = plan_for(&model, &explicit, &[], &[]);
    assert_eq!(plan.profile.as_deref(), Some("bedrock-profile"));
    assert_eq!(plan.endpoint.as_deref(), Some(model.base_url.as_str()));
    assert_eq!(plan.region.as_deref(), Some("eu-central-1"));

    let plan = plan_for(
        &model,
        &BedrockOptions::default(),
        &[("AWS_PROFILE", "scoped-bedrock-profile")],
        &[],
    );
    assert_eq!(plan.profile.as_deref(), Some("scoped-bedrock-profile"));
    assert_eq!(plan.endpoint.as_deref(), Some(model.base_url.as_str()));
    assert_eq!(plan.region.as_deref(), Some("eu-central-1"));

    let plan = plan_for(
        &model,
        &BedrockOptions::default(),
        &[],
        &[("AWS_PROFILE", "ambient-bedrock-profile")],
    );
    assert_eq!(plan.profile.as_deref(), Some("ambient-bedrock-profile"));
    assert_eq!(plan.endpoint, None);
    assert_eq!(plan.region, None);
}

#[test]
fn custom_endpoints_are_always_passed_through() {
    let mut model = catalog_model("us.anthropic.claude-opus-4-8");
    model.base_url = "https://bedrock-vpc.example.com".into();
    let plan = plan_for(&model, &BedrockOptions::default(), &[], &[("AWS_REGION", "us-west-2")]);
    assert_eq!(plan.endpoint.as_deref(), Some("https://bedrock-vpc.example.com"));
    assert_eq!(plan.region.as_deref(), Some("us-west-2"));
}

#[test]
fn inference_profile_arns_carry_their_region() {
    let mut model = catalog_model("us.anthropic.claude-opus-4-8");
    model.id = "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abc123".into();
    let plan = plan_for(&model, &BedrockOptions::default(), &[], &[("AWS_REGION", "us-east-1")]);
    assert_eq!(plan.region.as_deref(), Some("us-west-2"));

    model.id = "arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:application-inference-profile/abc123".into();
    let plan = plan_for(&model, &BedrockOptions::default(), &[], &[("AWS_REGION", "us-east-1")]);
    assert_eq!(plan.region.as_deref(), Some("us-gov-west-1"));
    assert_eq!(arn_region("arn:aws:bedrock:us-west-2"), None);
    assert_eq!(arn_region("arn:aws:s3:us-west-2:1:x"), None);
}

#[test]
fn ambient_profile_is_kept_for_custom_model_ids() {
    let mut model = catalog_model("us.anthropic.claude-opus-4-8");
    model.id = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/example".into();
    let plan = plan_for(
        &model,
        &BedrockOptions::default(),
        &[],
        &[("AWS_PROFILE", "bedrock-profile")],
    );
    assert_eq!(plan.profile.as_deref(), Some("bedrock-profile"));
    assert_eq!(plan.bearer_token, None);
}

#[test]
fn bearer_token_sources_and_skip_auth() {
    let model = catalog_model("us.anthropic.claude-opus-4-8");
    let scoped = ProviderEnv::new();
    let env = lookup(Some(&scoped), &no_process_env);
    let plan = resolve_client_plan(&model, &BedrockOptions::default(), Some("bedrock-api-key"), &env);
    assert_eq!(plan.bearer_token.as_deref(), Some("bedrock-api-key"));
    assert_eq!(plan.region.as_deref(), Some("us-east-1"));

    let explicit = BedrockOptions {
        bearer_token: Some("explicit".into()),
        ..BedrockOptions::default()
    };
    let plan = resolve_client_plan(&model, &explicit, Some("api-key"), &env);
    assert_eq!(plan.bearer_token.as_deref(), Some("explicit"));

    let plan = plan_for(
        &model,
        &BedrockOptions::default(),
        &[("AWS_BEARER_TOKEN_BEDROCK", "from-env")],
        &[],
    );
    assert_eq!(plan.bearer_token.as_deref(), Some("from-env"));

    let plan = plan_for(
        &model,
        &BedrockOptions::default(),
        &[("AWS_BEARER_TOKEN_BEDROCK", "from-env"), ("AWS_BEDROCK_SKIP_AUTH", "1")],
        &[("AWS_ACCESS_KEY_ID", "a"), ("AWS_SECRET_ACCESS_KEY", "b")],
    );
    assert!(plan.skip_auth);
    assert_eq!(plan.bearer_token, None);
    assert_eq!(
        plan.credentials.map(|credentials| credentials.access_key_id),
        Some("dummy-access-key".into())
    );
}

#[test]
fn endpoint_region_parsing() {
    assert_eq!(
        standard_endpoint_region("https://bedrock-runtime-fips.us-gov-west-1.amazonaws.com"),
        Some("us-gov-west-1".into())
    );
    assert_eq!(
        standard_endpoint_region("https://bedrock-runtime.cn-north-1.amazonaws.com.cn/"),
        Some("cn-north-1".into())
    );
    assert_eq!(standard_endpoint_region("https://bedrock-vpc.example.com"), None);
    assert_eq!(standard_endpoint_region("not a url"), None);
    assert_eq!(standard_endpoint_region(""), None);
}

// ---------------------------------------------------------------------------------------------
// Request payload
// ---------------------------------------------------------------------------------------------

#[test]
fn strict_tool_use_is_gated_by_model_capability() {
    let mut tool = Tool::new(
        "lookup",
        "Look up a value",
        json!({"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}),
    );
    tool.constrained_sampling = Some(ConstrainedSampling::JsonSchema {
        strict: StrictMode::Require,
    });
    let context = Context::new(vec![Message::user("Use the tool")]).with_tools(vec![tool.clone()]);
    let options = StreamOptions {
        cache_retention: Some(CacheRetention::None),
        ..StreamOptions::default()
    };
    let sent = payload_with(&sonnet_45(), &context, &options);
    let spec = &sent["toolConfig"]["tools"][0]["toolSpec"];
    assert_eq!(spec["strict"], json!(true));
    assert_eq!(spec["inputSchema"]["json"]["additionalProperties"], json!(false));

    // Required strictness on a model without strict tools fails the request.
    let transcript = normalize_context(&context);
    let env = lookup(None, &no_process_env);
    let error = build_payload(&nova(), &transcript, &options, &BedrockOptions::default(), &env).unwrap_err();
    assert!(error.to_string().contains("strict tools are unsupported"), "{error}");

    tool.constrained_sampling = Some(ConstrainedSampling::JsonSchema {
        strict: StrictMode::Prefer,
    });
    let context = Context::new(vec![Message::user("Use the tool")]).with_tools(vec![tool]);
    let sent = payload_with(&nova(), &context, &options);
    assert_eq!(sent["toolConfig"]["tools"][0]["toolSpec"].get("strict"), None);
}

#[test]
fn tool_choice_and_request_metadata() {
    let tool = Tool::new("read", "Read", json!({"type": "object"}));
    let context = Context::new(vec![Message::user("hi")]).with_tools(vec![tool]);
    let with_choice = |choice: Option<AnthropicToolChoice>| {
        let options = StreamOptions {
            cache_retention: Some(CacheRetention::None),
            api: Some(ApiOptions::BedrockConverseStream(BedrockOptions {
                tool_choice: choice,
                request_metadata: Some([("team".to_owned(), "infra".to_owned())].into_iter().collect()),
                ..BedrockOptions::default()
            })),
            ..StreamOptions::default()
        };
        payload_with(&nova(), &context, &options)
    };
    assert_eq!(with_choice(None)["toolConfig"].get("toolChoice"), None);
    assert_eq!(
        with_choice(Some(AnthropicToolChoice::Auto))["toolConfig"]["toolChoice"],
        json!({"auto": {}})
    );
    assert_eq!(
        with_choice(Some(AnthropicToolChoice::Any))["toolConfig"]["toolChoice"],
        json!({"any": {}})
    );
    assert_eq!(
        with_choice(Some(AnthropicToolChoice::Tool { name: "read".into() }))["toolConfig"]["toolChoice"],
        json!({"tool": {"name": "read"}})
    );
    let none = with_choice(Some(AnthropicToolChoice::None));
    assert_eq!(none.get("toolConfig"), None);
    assert_eq!(none["requestMetadata"], json!({"team": "infra"}));
}

#[test]
fn inference_config_defaults_to_the_model_cap_for_claude_only() {
    let claude = payload(&sonnet_45(), vec![Message::user("hi")]);
    assert_eq!(claude["inferenceConfig"], json!({"maxTokens": 64000}));
    let other = payload(&nova(), vec![Message::user("hi")]);
    assert_eq!(other["inferenceConfig"], json!({}));
    let options = StreamOptions {
        max_tokens: Some(100),
        temperature: Some(0.5),
        cache_retention: Some(CacheRetention::None),
        ..StreamOptions::default()
    };
    let explicit = payload_with(&nova(), &Context::new(vec![Message::user("hi")]), &options);
    assert_eq!(
        explicit["inferenceConfig"],
        json!({"maxTokens": 100, "temperature": 0.5})
    );
    assert_eq!(explicit["modelId"], json!("amazon.nova-lite-v1:0"));
}

#[test]
fn blank_user_content_becomes_a_placeholder() {
    let sent = payload(&sonnet_45(), vec![Message::user("   ")]);
    assert_eq!(
        sent["messages"],
        json!([{"role": "user", "content": [{"text": "<empty>"}]}])
    );

    let blocks = Message::User(UserMessage {
        content: UserContent::Blocks(vec![UserBlock::text(""), UserBlock::text("hello")]),
        timestamp: 0,
    });
    let sent = payload(&sonnet_45(), vec![blocks]);
    assert_eq!(sent["messages"][0]["content"], json!([{"text": "hello"}]));

    let empty = Message::User(UserMessage {
        content: UserContent::Blocks(vec![UserBlock::text(" ")]),
        timestamp: 0,
    });
    let sent = payload(&sonnet_45(), vec![empty]);
    assert_eq!(sent["messages"][0]["content"], json!([{"text": "<empty>"}]));
}

#[test]
fn blank_assistant_text_is_skipped_and_empty_messages_dropped() {
    let model = sonnet_45();
    let sent = payload(
        &model,
        vec![
            Message::user("hi"),
            assistant(
                &model,
                vec![AssistantBlock::Text(TextContent::new("  "))],
                StopReason::Stop,
            ),
            assistant(
                &model,
                vec![
                    AssistantBlock::Text(TextContent::new("hello")),
                    AssistantBlock::Text(TextContent::new("")),
                ],
                StopReason::Stop,
            ),
        ],
    );
    assert_eq!(
        sent["messages"],
        json!([
            {"role": "user", "content": [{"text": "hi"}]},
            {"role": "assistant", "content": [{"text": "hello"}]}
        ])
    );
}

#[test]
fn blank_tool_result_content_becomes_a_placeholder() {
    let model = sonnet_45();
    let sent = payload(
        &model,
        vec![
            assistant(
                &model,
                vec![AssistantBlock::ToolCall(ToolCall {
                    id: "tool-1".into(),
                    name: "tool".into(),
                    ..ToolCall::default()
                })],
                StopReason::ToolUse,
            ),
            tool_result("tool-1", "tool", vec![UserBlock::text("")], false),
        ],
    );
    assert_eq!(
        sent["messages"][1]["content"][0]["toolResult"],
        json!({"toolUseId": "tool-1", "content": [{"text": "<empty>"}], "status": "success"})
    );
}

#[test]
fn claude_error_tool_result_images_follow_the_tool_results() {
    let model = sonnet_45();
    let sent = payload(
        &model,
        vec![
            assistant(
                &model,
                vec![
                    AssistantBlock::ToolCall(ToolCall {
                        id: "call-1".into(),
                        name: "codemode".into(),
                        ..ToolCall::default()
                    }),
                    AssistantBlock::ToolCall(ToolCall {
                        id: "call-2".into(),
                        name: "read".into(),
                        ..ToolCall::default()
                    }),
                ],
                StopReason::ToolUse,
            ),
            tool_result(
                "call-1",
                "codemode",
                vec![
                    UserBlock::text("Script failed"),
                    image("AAAA"),
                    UserBlock::text("Script error"),
                ],
                true,
            ),
            tool_result("call-2", "read", vec![image("BBBB")], false),
        ],
    );
    let png = |data: &str| json!({"image": {"source": {"bytes": data}, "format": "png"}});
    assert_eq!(
        sent["messages"][1]["content"],
        json!([
            {"toolResult": {"toolUseId": "call-1", "content": [
                {"text": "Script failed"},
                {"text": "(image attached after the tool results)"},
                {"text": "Script error"}
            ], "status": "error"}},
            {"toolResult": {"toolUseId": "call-2", "content": [png("BBBB")], "status": "success"}},
            {"text": "Images from the error result of tool call call-1 (codemode):"},
            png("AAAA")
        ])
    );
}

#[test]
fn other_models_keep_images_in_error_tool_results() {
    let model = nova();
    let sent = payload(
        &model,
        vec![
            assistant(
                &model,
                vec![AssistantBlock::ToolCall(ToolCall {
                    id: "call-1".into(),
                    name: "codemode".into(),
                    ..ToolCall::default()
                })],
                StopReason::ToolUse,
            ),
            tool_result(
                "call-1",
                "codemode",
                vec![UserBlock::text("Script failed"), image("AAAA")],
                true,
            ),
        ],
    );
    assert_eq!(
        sent["messages"][1]["content"],
        json!([{"toolResult": {"toolUseId": "call-1", "content": [
            {"text": "Script failed"},
            {"image": {"source": {"bytes": "AAAA"}, "format": "png"}}
        ], "status": "error"}}])
    );
}

#[test]
fn replayed_tool_input_drops_empty_property_names() {
    let model = sonnet_45();
    let mut arguments = JsonObject::new();
    arguments.insert("path".into(), json!("/workspace/foobar/file.js"));
    arguments.insert(
        "edits".into(),
        json!([{"oldText": "first", "newText": "updated first"},
               {"oldText": "second", "newText": "updated second", "": ""}]),
    );
    let sent = payload(
        &model,
        vec![
            assistant(
                &model,
                vec![AssistantBlock::ToolCall(ToolCall {
                    id: "tool-1".into(),
                    name: "edit".into(),
                    arguments,
                    ..ToolCall::default()
                })],
                StopReason::ToolUse,
            ),
            tool_result("tool-1", "edit", vec![UserBlock::text("done")], false),
            Message::user("Continue"),
        ],
    );
    assert_eq!(
        sent["messages"][0]["content"][0]["toolUse"]["input"],
        json!({"path": "/workspace/foobar/file.js", "edits": [
            {"oldText": "first", "newText": "updated first"},
            {"oldText": "second", "newText": "updated second"}
        ]})
    );
}

#[test]
fn thinking_replay_depends_on_signature_support() {
    let claude = sonnet_45();
    let signed = AssistantBlock::Thinking(ThinkingContent {
        thinking: "plan".into(),
        thinking_signature: Some("sig".into()),
        redacted: None,
    });
    let unsigned = AssistantBlock::Thinking(ThinkingContent::new("unsigned plan"));
    let sent = payload(
        &claude,
        vec![
            Message::user("hi"),
            assistant(&claude, vec![signed.clone(), unsigned.clone()], StopReason::Stop),
        ],
    );
    assert_eq!(
        sent["messages"][1]["content"],
        json!([
            {"reasoningContent": {"reasoningText": {"text": "plan", "signature": "sig"}}},
            {"text": "unsigned plan"}
        ])
    );

    let gpt = gpt_56();
    let sent = payload(
        &gpt,
        vec![Message::user("hi"), assistant(&gpt, vec![signed], StopReason::Stop)],
    );
    assert_eq!(
        sent["messages"][1]["content"],
        json!([{"reasoningContent": {"reasoningText": {"text": "plan"}}}])
    );
}

#[test]
fn redacted_reasoning_is_replayed_as_redacted_content_before_tool_use() {
    let model = gpt_56();
    let encoded = STANDARD.encode(b"\x00\x01encrypted-reasoning\xff");
    let redacted = AssistantBlock::Thinking(ThinkingContent {
        thinking: String::new(),
        thinking_signature: Some(encoded.clone()),
        redacted: Some(true),
    });
    let mut arguments = JsonObject::new();
    arguments.insert("path".into(), json!("/tmp/a.txt"));
    let sent = payload(
        &model,
        vec![
            Message::user("read the file"),
            assistant(
                &model,
                vec![
                    redacted,
                    AssistantBlock::ToolCall(ToolCall {
                        id: "tool-1".into(),
                        name: "read".into(),
                        arguments,
                        ..ToolCall::default()
                    }),
                ],
                StopReason::ToolUse,
            ),
            tool_result("tool-1", "read", vec![UserBlock::text("file body")], false),
        ],
    );
    assert_eq!(
        sent["messages"][1]["content"],
        json!([
            {"reasoningContent": {"redactedContent": encoded}},
            {"toolUse": {"toolUseId": "tool-1", "name": "read", "input": {"path": "/tmp/a.txt"}}}
        ])
    );

    // A signature that is not base64 drops the block instead of failing the request.
    let broken = AssistantBlock::Thinking(ThinkingContent {
        thinking: String::new(),
        thinking_signature: Some("not base64!".into()),
        redacted: Some(true),
    });
    let sent = payload(
        &model,
        vec![
            Message::user("hi"),
            assistant(
                &model,
                vec![broken, AssistantBlock::Text(TextContent::new("done"))],
                StopReason::Stop,
            ),
        ],
    );
    assert_eq!(sent["messages"][1]["content"], json!([{"text": "done"}]));
}

#[test]
fn foreign_assistant_messages_are_normalized_for_replay() {
    let model = sonnet_45();
    let mut foreign = AssistantMessage::empty("openai-responses", "openai", "gpt-5");
    foreign.stop_reason = StopReason::ToolUse;
    foreign.content = vec![
        AssistantBlock::Thinking(ThinkingContent {
            thinking: "foreign".into(),
            thinking_signature: Some("x".into()),
            redacted: None,
        }),
        AssistantBlock::Text(TextContent {
            text: "calling".into(),
            text_signature: Some("msg_1".into()),
        }),
        AssistantBlock::ToolCall(ToolCall {
            id: "call_1|fc_abc+/=".into(),
            name: "read".into(),
            ..ToolCall::default()
        }),
    ];
    let sent = payload(
        &model,
        vec![Message::user("hi"), Message::Assistant(foreign), Message::user("next")],
    );
    assert_eq!(
        sent["messages"],
        json!([
            {"role": "user", "content": [{"text": "hi"}]},
            {"role": "assistant", "content": [
                {"text": "calling"},
                {"toolUse": {"toolUseId": "call_1_fc_abc___", "name": "read", "input": {}}}
            ]},
            // Orphaned tool call: a synthetic error result is inserted.
            {"role": "user", "content": [{"toolResult": {"toolUseId": "call_1_fc_abc___",
                "content": [{"text": "No result provided"}], "status": "error"}}]},
            {"role": "user", "content": [{"text": "next"}]}
        ])
    );
}

#[test]
fn tool_call_ids_are_sanitized_and_capped() {
    assert_eq!(normalize_tool_call_id("call|1"), "call_1");
    assert_eq!(normalize_tool_call_id(&"a".repeat(80)).len(), 64);
    // Astral characters count as two UTF-16 code units, as upstream.
    assert_eq!(normalize_tool_call_id("a\u{1F600}b"), "a__b");
}

#[test]
fn images_are_decoded_and_unknown_types_rejected() {
    let model = sonnet_45();
    let user = |mime: &str, data: &str| {
        Message::User(UserMessage {
            content: UserContent::Blocks(vec![UserBlock::Image(ImageContent {
                data: data.into(),
                mime_type: mime.into(),
            })]),
            timestamp: 0,
        })
    };
    // Whitespace and missing padding are tolerated like `atob`.
    let sent = payload(&model, vec![user("image/jpg", "AAE\nC")]);
    assert_eq!(
        sent["messages"][0]["content"][0],
        json!({"image": {"source": {"bytes": "AAEC"}, "format": "jpeg"}})
    );
    let transcript = normalize_context(&Context::new(vec![user("image/bmp", "AAAA")]));
    let env = lookup(None, &no_process_env);
    let error = build_payload(
        &model,
        &transcript,
        &StreamOptions::default(),
        &BedrockOptions::default(),
        &env,
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "Unknown image type: image/bmp");
}

#[test]
fn cache_points_for_claude_models() {
    let model = sonnet_45();
    let context = Context::new(vec![Message::user("Hello")]).with_system_prompt("You are helpful.");
    let sent = payload_with(&model, &context, &StreamOptions::default());
    assert_eq!(
        sent["system"],
        json!([{"text": "You are helpful."}, {"cachePoint": {"type": "default"}}])
    );
    assert_eq!(
        sent["messages"][0]["content"],
        json!([{"text": "Hello"}, {"cachePoint": {"type": "default"}}])
    );

    let long = StreamOptions {
        cache_retention: Some(CacheRetention::Long),
        ..StreamOptions::default()
    };
    let sent = payload_with(&model, &context, &long);
    assert_eq!(
        sent["system"][1],
        json!({"cachePoint": {"type": "default", "ttl": "1h"}})
    );

    let env_long = StreamOptions {
        env: Some(env_of(&[("PI_CACHE_RETENTION", "long")])),
        ..StreamOptions::default()
    };
    let sent = payload_with(&model, &context, &env_long);
    assert_eq!(
        sent["system"][1],
        json!({"cachePoint": {"type": "default", "ttl": "1h"}})
    );

    let none = StreamOptions {
        cache_retention: Some(CacheRetention::None),
        ..StreamOptions::default()
    };
    let sent = payload_with(&model, &context, &none);
    assert_eq!(sent["system"], json!([{"text": "You are helpful."}]));

    // Nova caches automatically; AWS_BEDROCK_FORCE_CACHE forces cache points for opaque ids.
    let sent = payload_with(&nova(), &context, &StreamOptions::default());
    assert_eq!(sent["system"], json!([{"text": "You are helpful."}]));
    let mut opaque = nova();
    opaque.id = "arn:aws:bedrock:us-east-1:1:application-inference-profile/x".into();
    opaque.name = "My profile".into();
    let forced = StreamOptions {
        env: Some(env_of(&[("AWS_BEDROCK_FORCE_CACHE", "1")])),
        ..StreamOptions::default()
    };
    let sent = payload_with(&opaque, &context, &forced);
    assert_eq!(sent["system"][1], json!({"cachePoint": {"type": "default"}}));
}

#[test]
fn cache_points_via_application_inference_profile_names() {
    let mut model = catalog_model("global.anthropic.claude-opus-4-6-v1");
    model.id = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-profile".into();
    model.name = "Claude Sonnet 4.6".into();
    let context = Context::new(vec![Message::user("Hello")]).with_system_prompt("You are helpful.");
    let sent = payload_with(&model, &context, &StreamOptions::default());
    assert_eq!(sent["system"].as_array().unwrap().len(), 2);
    let last = sent["messages"][0]["content"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
        .clone();
    assert!(last.get("cachePoint").is_some());
}

#[test]
fn mid_conversation_system_messages_fold_into_the_prompt() {
    let model = nova();
    let messages = vec![
        Message::System(SystemMessage {
            content: "First.".into(),
            ..SystemMessage::default()
        }),
        Message::user("hi"),
        Message::System(SystemMessage {
            content: "Second.".into(),
            ..SystemMessage::default()
        }),
        Message::user("again"),
    ];
    let sent = payload(&model, messages);
    assert_eq!(sent["system"], json!([{"text": "First.\n\nSecond."}]));
    assert_eq!(sent["messages"].as_array().unwrap().len(), 2);
}

// ---------------------------------------------------------------------------------------------
// Thinking payloads
// ---------------------------------------------------------------------------------------------

fn adaptive_with_binding() -> Value {
    json!({"type": "adaptive", "display": "summarized", "block_binding": {"prefix_mismatch_behavior": "drop_block"}})
}

#[test]
fn adaptive_thinking_with_block_binding_for_new_claude_models() {
    let mut opus_48 = catalog_model("global.anthropic.claude-opus-4-6-v1");
    opus_48.id = "global.anthropic.claude-opus-4-8-v1".into();
    opus_48.name = "Claude Opus 4.8 (Global)".into();
    for model in [
        opus_48,
        catalog_model("global.anthropic.claude-fable-5"),
        catalog_model("global.anthropic.claude-sonnet-5"),
        catalog_model("global.anthropic.claude-opus-5"),
        catalog_model("global.anthropic.claude-opus-5-5"),
    ] {
        let fields = &reasoning_payload(&model, BedrockOptions::default())["additionalModelRequestFields"];
        assert_eq!(fields["thinking"], adaptive_with_binding(), "{}", model.id);
        assert_eq!(fields["output_config"], json!({"effort": "high"}), "{}", model.id);
        assert_eq!(
            fields["anthropic_beta"],
            json!([THINKING_BINDING_CONTROLS_BETA]),
            "{}",
            model.id
        );
        let xhigh = reasoning_payload(
            &model,
            BedrockOptions {
                reasoning: Some(ThinkingLevel::Xhigh),
                ..BedrockOptions::default()
            },
        );
        assert_eq!(
            xhigh["additionalModelRequestFields"]["output_config"],
            json!({"effort": "xhigh"}),
            "{}",
            model.id
        );
    }
}

#[test]
fn block_binding_is_omitted_for_46_models() {
    for id in [
        "global.anthropic.claude-opus-4-6-v1",
        "global.anthropic.claude-sonnet-4-6",
    ] {
        let fields = &reasoning_payload(&catalog_model(id), BedrockOptions::default())["additionalModelRequestFields"];
        assert_eq!(fields["thinking"], json!({"type": "adaptive", "display": "summarized"}));
        assert_eq!(fields.get("anthropic_beta"), None);
    }
    // xhigh is not native on 4.6: the thinking level map decides, else high.
    let fields = &reasoning_payload(
        &catalog_model("global.anthropic.claude-opus-4-6-v1"),
        BedrockOptions {
            reasoning: Some(ThinkingLevel::Max),
            thinking_display: Some(ThinkingDisplay::Omitted),
            ..BedrockOptions::default()
        },
    )["additionalModelRequestFields"];
    assert_eq!(fields["output_config"], json!({"effort": "max"}));
    assert_eq!(fields["thinking"]["display"], json!("omitted"));
}

#[test]
fn gov_cloud_targets_omit_display_and_binding() {
    let mut model = sonnet_45();
    model.id = "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0".into();
    model.name = "Claude Sonnet 4.5 (GovCloud)".into();
    let fields = &reasoning_payload(&model, BedrockOptions::default())["additionalModelRequestFields"];
    assert_eq!(fields["thinking"], json!({"type": "enabled", "budget_tokens": 16384}));
    assert_eq!(fields["anthropic_beta"], json!([INTERLEAVED_THINKING_BETA]));

    let mut opus = catalog_model("global.anthropic.claude-opus-4-6-v1");
    opus.id = "global.anthropic.claude-opus-4-8-v1".into();
    opus.name = "Claude Opus 4.8 (Global)".into();
    let fields = &reasoning_payload(
        &opus,
        BedrockOptions {
            region: Some("us-gov-west-1".into()),
            ..BedrockOptions::default()
        },
    )["additionalModelRequestFields"];
    assert_eq!(fields["thinking"], json!({"type": "adaptive"}));
    assert_eq!(fields["output_config"], json!({"effort": "high"}));
    assert_eq!(fields.get("anthropic_beta"), None);
}

#[test]
fn budget_thinking_uses_custom_budgets_and_interleaved_flag() {
    let model = sonnet_45();
    let fields = &reasoning_payload(
        &model,
        BedrockOptions {
            reasoning: Some(ThinkingLevel::Max),
            thinking_budgets: Some(ThinkingBudgets {
                high: Some(5000),
                ..ThinkingBudgets::default()
            }),
            interleaved_thinking: Some(false),
            ..BedrockOptions::default()
        },
    )["additionalModelRequestFields"];
    assert_eq!(
        *fields,
        json!({"thinking": {"type": "enabled", "budget_tokens": 5000, "display": "summarized"}})
    );
    let fields = &reasoning_payload(
        &model,
        BedrockOptions {
            reasoning: Some(ThinkingLevel::Minimal),
            ..BedrockOptions::default()
        },
    )["additionalModelRequestFields"];
    assert_eq!(fields["thinking"]["budget_tokens"], json!(1024));
    assert_eq!(fields["anthropic_beta"], json!([INTERLEAVED_THINKING_BETA]));
}

#[test]
fn application_inference_profiles_match_by_name() {
    let mut model = catalog_model("global.anthropic.claude-opus-4-6-v1");
    model.id = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-profile".into();
    model.name = "Claude Opus 4.6".into();
    let fields = &reasoning_payload(&model, BedrockOptions::default())["additionalModelRequestFields"];
    assert_eq!(fields["thinking"], json!({"type": "adaptive", "display": "summarized"}));
    assert_eq!(fields["output_config"], json!({"effort": "high"}));

    model.name = "Claude Sonnet 4.5".into();
    let fields = &reasoning_payload(&model, BedrockOptions::default())["additionalModelRequestFields"];
    assert_eq!(fields["thinking"]["type"], json!("enabled"));
    assert!(fields["thinking"]["budget_tokens"].is_u64());
    assert_eq!(fields["anthropic_beta"], json!([INTERLEAVED_THINKING_BETA]));
}

#[test]
fn openai_reasoning_payloads() {
    for (level, effort) in [
        (ThinkingLevel::Minimal, "low"),
        (ThinkingLevel::Low, "low"),
        (ThinkingLevel::Medium, "medium"),
        (ThinkingLevel::High, "high"),
        (ThinkingLevel::Xhigh, "xhigh"),
        (ThinkingLevel::Max, "max"),
    ] {
        for id in [
            "global.openai.gpt-6-sol",
            "us.openai.gpt-6-luna",
            "global.openai.gpt-5.6-sol",
        ] {
            let mut model = catalog_model(id);
            // The catalog maps xhigh only; drop the map to check the built-in table.
            model.thinking_level_map = None;
            let sent = reasoning_payload(
                &model,
                BedrockOptions {
                    reasoning: Some(level),
                    ..BedrockOptions::default()
                },
            );
            assert_eq!(
                sent["additionalModelRequestFields"],
                json!({"reasoning": {"effort": effort}}),
                "{id}"
            );
        }
    }

    let mut model = catalog_model("global.openai.gpt-6-sol");
    model.id = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-profile".into();
    model.name = "GPT-6 Sol".into();
    let sent = reasoning_payload(
        &model,
        BedrockOptions {
            reasoning: Some(ThinkingLevel::Medium),
            ..BedrockOptions::default()
        },
    );
    assert_eq!(
        sent["additionalModelRequestFields"],
        json!({"reasoning": {"effort": "medium"}})
    );

    let oss = catalog_model("openai.gpt-oss-120b-1:0");
    for (level, effort) in [
        (ThinkingLevel::Minimal, "low"),
        (ThinkingLevel::Medium, "medium"),
        (ThinkingLevel::Xhigh, "high"),
    ] {
        let sent = reasoning_payload(
            &oss,
            BedrockOptions {
                reasoning: Some(level),
                ..BedrockOptions::default()
            },
        );
        assert_eq!(
            sent["additionalModelRequestFields"],
            json!({"reasoning_effort": effort})
        );
    }

    let off = payload(&catalog_model("global.openai.gpt-6-sol"), vec![Message::user("hi")]);
    assert_eq!(off.get("additionalModelRequestFields"), None);
}

// ---------------------------------------------------------------------------------------------
// Payload -> SDK input
// ---------------------------------------------------------------------------------------------

#[test]
fn payload_converts_to_sdk_input() {
    let model = sonnet_45();
    let mut tool = Tool::new(
        "read",
        "Read a file",
        json!({"type": "object", "properties": {"path": {"type": "string"}}}),
    );
    tool.constrained_sampling = Some(ConstrainedSampling::JsonSchema {
        strict: StrictMode::Prefer,
    });
    let encoded = STANDARD.encode(b"opaque");
    let context = Context::new(vec![
        Message::User(UserMessage {
            content: UserContent::Blocks(vec![UserBlock::text("look"), image("AAEC")]),
            timestamp: 0,
        }),
        assistant(
            &model,
            vec![
                AssistantBlock::Thinking(ThinkingContent {
                    thinking: "plan".into(),
                    thinking_signature: Some("sig".into()),
                    redacted: None,
                }),
                AssistantBlock::Thinking(ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: Some(encoded),
                    redacted: Some(true),
                }),
                AssistantBlock::ToolCall(ToolCall {
                    id: "t1".into(),
                    name: "read".into(),
                    arguments: [("path".to_owned(), json!("a"))].into_iter().collect(),
                    ..ToolCall::default()
                }),
            ],
            StopReason::ToolUse,
        ),
        tool_result("t1", "read", vec![UserBlock::text("body"), image("AAEC")], false),
    ])
    .with_system_prompt("sys")
    .with_tools(vec![tool]);
    let options = StreamOptions {
        temperature: Some(0.25),
        cache_retention: Some(CacheRetention::Long),
        api: Some(ApiOptions::BedrockConverseStream(BedrockOptions {
            reasoning: Some(ThinkingLevel::Low),
            tool_choice: Some(AnthropicToolChoice::Tool { name: "read".into() }),
            request_metadata: Some([("k".to_owned(), "v".to_owned())].into_iter().collect()),
            ..BedrockOptions::default()
        })),
        ..StreamOptions::default()
    };
    let sent = payload_with(&model, &context, &options);
    let request = ConverseRequest::from_payload(&sent).unwrap();
    assert_eq!(request.model_id, model.id);
    assert_eq!(request.messages.len(), 3);
    assert_eq!(request.messages[0].role, bedrock::ConversationRole::User);
    let first = &request.messages[0].content;
    assert_eq!(first[0], bedrock::ContentBlock::Text("look".into()));
    let bedrock::ContentBlock::Image(image_block) = &first[1] else {
        panic!("expected an image: {first:?}")
    };
    assert_eq!(image_block.format, bedrock::ImageFormat::Png);
    assert_eq!(
        image_block.source,
        Some(bedrock::ImageSource::Bytes(Blob::new(vec![0, 1, 2])))
    );
    let assistant_content = &request.messages[1].content;
    assert!(matches!(
        &assistant_content[0],
        bedrock::ContentBlock::ReasoningContent(bedrock::ReasoningContentBlock::ReasoningText(text))
            if text.text == "plan" && text.signature.as_deref() == Some("sig")
    ));
    assert_eq!(
        assistant_content[1],
        bedrock::ContentBlock::ReasoningContent(bedrock::ReasoningContentBlock::RedactedContent(Blob::new(
            b"opaque".to_vec()
        )))
    );
    let bedrock::ContentBlock::ToolUse(tool_use) = &assistant_content[2] else {
        panic!("expected tool use")
    };
    assert_eq!(tool_use.tool_use_id, "t1");
    assert_eq!(document_to_json(&tool_use.input), json!({"path": "a"}));
    let results = &request.messages[2].content;
    let bedrock::ContentBlock::ToolResult(result) = &results[0] else {
        panic!("expected tool result")
    };
    assert_eq!(result.status, Some(bedrock::ToolResultStatus::Success));
    assert_eq!(result.content.len(), 2);
    let bedrock::ContentBlock::CachePoint(point) = results.last().unwrap() else {
        panic!("expected cache point")
    };
    assert_eq!(point.ttl, Some(bedrock::CacheTtl::OneHour));

    let system = request.system.unwrap();
    assert_eq!(system[0], bedrock::SystemContentBlock::Text("sys".into()));
    assert!(matches!(system[1], bedrock::SystemContentBlock::CachePoint(_)));
    let inference = request.inference_config.unwrap();
    assert_eq!(inference.max_tokens, Some(64000));
    assert_eq!(inference.temperature, Some(0.25));
    let tools = request.tool_config.unwrap();
    let bedrock::Tool::ToolSpec(spec) = &tools.tools[0] else {
        panic!("expected tool spec")
    };
    assert_eq!(spec.name, "read");
    assert_eq!(spec.strict, Some(true));
    assert!(matches!(&tools.tool_choice, Some(bedrock::ToolChoice::Tool(choice)) if choice.name == "read"));
    let fields = document_to_json(&request.additional_model_request_fields.unwrap());
    assert_eq!(fields["thinking"]["budget_tokens"], json!(2048));
    assert_eq!(
        request.request_metadata,
        Some([("k".to_owned(), "v".to_owned())].into_iter().collect())
    );
}

#[test]
fn payload_conversion_reports_invalid_members() {
    let error = ConverseRequest::from_payload(&json!({"messages": []})).unwrap_err();
    assert_eq!(
        error.to_string(),
        "Invalid Bedrock payload: payload.modelId must be a string"
    );
    let error = ConverseRequest::from_payload(&json!({
        "modelId": "m",
        "messages": [{"role": "user", "content": [{"video": {}}]}]
    }))
    .unwrap_err();
    assert!(
        error.to_string().contains("unsupported content block \"video\""),
        "{error}"
    );
    // Unknown top-level members are ignored, like the JS SDK command serializer.
    let request = ConverseRequest::from_payload(&json!({
        "modelId": "m", "messages": [], "guardrailConfig": {}, "inferenceConfig": {"topP": 0.5, "stopSequences": ["x"]}
    }))
    .unwrap();
    let inference = request.inference_config.unwrap();
    assert_eq!(inference.top_p, Some(0.5));
    assert_eq!(inference.stop_sequences, Some(vec!["x".to_owned()]));
}

#[test]
fn documents_round_trip_numbers() {
    let value = json!({"a": 1, "b": -2, "c": 1.5, "d": [true, null, "s"], "e": {}});
    assert_eq!(document_to_json(&json_to_document(&value)), value);
}

// ---------------------------------------------------------------------------------------------
// Response stream
// ---------------------------------------------------------------------------------------------

fn new_builder(model: &Model) -> (MessageBuilder, AssistantMessageEventStream) {
    let (sender, stream) = AssistantMessageEventStream::channel();
    (
        MessageBuilder::new(sender, crate::api::common::new_output(model)),
        stream,
    )
}

fn message_start() -> bedrock::ConverseStreamOutput {
    bedrock::ConverseStreamOutput::MessageStart(
        bedrock::MessageStartEvent::builder()
            .role(bedrock::ConversationRole::Assistant)
            .build()
            .unwrap(),
    )
}

fn block_delta(index: i32, delta: bedrock::ContentBlockDelta) -> bedrock::ConverseStreamOutput {
    bedrock::ConverseStreamOutput::ContentBlockDelta(
        bedrock::ContentBlockDeltaEvent::builder()
            .content_block_index(index)
            .delta(delta)
            .build()
            .unwrap(),
    )
}

fn block_stop(index: i32) -> bedrock::ConverseStreamOutput {
    bedrock::ConverseStreamOutput::ContentBlockStop(
        bedrock::ContentBlockStopEvent::builder()
            .content_block_index(index)
            .build()
            .unwrap(),
    )
}

fn tool_start(index: i32, id: &str, name: &str) -> bedrock::ConverseStreamOutput {
    bedrock::ConverseStreamOutput::ContentBlockStart(
        bedrock::ContentBlockStartEvent::builder()
            .content_block_index(index)
            .start(bedrock::ContentBlockStart::ToolUse(
                bedrock::ToolUseBlockStart::builder()
                    .tool_use_id(id)
                    .name(name)
                    .build()
                    .unwrap(),
            ))
            .build()
            .unwrap(),
    )
}

fn tool_delta(index: i32, input: &str) -> bedrock::ConverseStreamOutput {
    block_delta(
        index,
        bedrock::ContentBlockDelta::ToolUse(bedrock::ToolUseBlockDelta::builder().input(input).build().unwrap()),
    )
}

fn message_stop(reason: &str) -> bedrock::ConverseStreamOutput {
    bedrock::ConverseStreamOutput::MessageStop(
        bedrock::MessageStopEvent::builder()
            .stop_reason(bedrock::StopReason::from(reason))
            .build()
            .unwrap(),
    )
}

fn redacted_delta(index: i32, bytes: &[u8]) -> bedrock::ConverseStreamOutput {
    block_delta(
        index,
        bedrock::ContentBlockDelta::ReasoningContent(bedrock::ReasoningContentBlockDelta::RedactedContent(Blob::new(
            bytes.to_vec(),
        ))),
    )
}

fn feed(
    model: &Model,
    events: Vec<bedrock::ConverseStreamOutput>,
) -> (MessageBuilder, StreamState, AssistantMessageEventStream) {
    let (mut builder, stream) = new_builder(model);
    let mut state = StreamState::default();
    for event in events {
        state.handle(event, model, &mut builder).unwrap();
    }
    (builder, state, stream)
}

async fn kinds(builder: MessageBuilder, mut stream: AssistantMessageEventStream) -> Vec<&'static str> {
    builder.finish();
    let mut kinds = Vec::new();
    while let Some(event) = stream.next().await {
        kinds.push(event.kind());
    }
    kinds
}

#[tokio::test]
async fn text_tool_calls_and_stop_reasons() {
    let model = sonnet_45();
    let (builder, _state, stream) = feed(
        &model,
        vec![
            message_start(),
            block_delta(0, bedrock::ContentBlockDelta::Text("Hel".into())),
            block_delta(0, bedrock::ContentBlockDelta::Text("lo".into())),
            block_stop(0),
            tool_start(1, "tool-1", "edit"),
            tool_start(2, "tool-2", "read"),
            tool_delta(
                1,
                r#"{"path":"/workspace/foobar/file.js","edits":[{"oldText":"second","#,
            ),
            tool_delta(2, r#"{"path":"b"}"#),
            tool_delta(1, r#""newText":"updated second","":""}]}"#),
            block_stop(2),
            block_stop(1),
            message_stop("tool_use"),
        ],
    );
    assert_eq!(builder.output.stop_reason, StopReason::ToolUse);
    assert_eq!(builder.output.raw_stop_reason.as_deref(), Some("tool_use"));
    assert_eq!(builder.output.text(), "Hello");
    let calls: Vec<_> = builder.output.tool_calls().cloned().collect();
    assert_eq!(calls[0].id, "tool-1");
    assert_eq!(
        Value::Object(calls[0].arguments.clone()),
        json!({"path": "/workspace/foobar/file.js", "edits": [{"oldText": "second", "newText": "updated second", "": ""}]})
    );
    assert_eq!(Value::Object(calls[1].arguments.clone()), json!({"path": "b"}));
    assert_eq!(
        kinds(builder, stream).await,
        [
            "start",
            "text_start",
            "text_delta",
            "text_delta",
            "text_end",
            "toolcall_start",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_delta",
            "toolcall_delta",
            "toolcall_end",
            "toolcall_end",
            "done"
        ]
    );
}

#[test]
fn stop_reason_mapping() {
    assert_eq!(map_stop_reason(Some("end_turn")), (StopReason::Stop, None));
    assert_eq!(map_stop_reason(Some("stop_sequence")), (StopReason::Stop, None));
    assert_eq!(map_stop_reason(Some("max_tokens")), (StopReason::Length, None));
    assert_eq!(
        map_stop_reason(Some("model_context_window_exceeded")),
        (StopReason::Length, None)
    );
    assert_eq!(map_stop_reason(Some("tool_use")), (StopReason::ToolUse, None));
    assert_eq!(
        map_stop_reason(Some("guardrail_intervened")),
        (
            StopReason::Error,
            Some("Provider stopped with: guardrail_intervened".into())
        )
    );
    assert_eq!(map_stop_reason(None), (StopReason::Error, None));

    let (builder, _, _stream) = feed(&sonnet_45(), vec![message_start(), message_stop("content_filtered")]);
    assert_eq!(builder.output.stop_reason, StopReason::Error);
    assert_eq!(builder.output.raw_stop_reason.as_deref(), Some("content_filtered"));
    assert_eq!(
        builder.output.error_message.as_deref(),
        Some("Provider stopped with: content_filtered")
    );
}

#[test]
fn user_message_start_is_an_error() {
    let model = sonnet_45();
    let (mut builder, _stream) = new_builder(&model);
    let start = bedrock::ConverseStreamOutput::MessageStart(
        bedrock::MessageStartEvent::builder()
            .role(bedrock::ConversationRole::User)
            .build()
            .unwrap(),
    );
    let error = StreamState::default().handle(start, &model, &mut builder).unwrap_err();
    assert_eq!(
        error.core,
        "Unexpected assistant message start but got user message start instead"
    );
}

#[tokio::test]
async fn signed_thinking() {
    let model = sonnet_45();
    let reasoning = |delta: bedrock::ReasoningContentBlockDelta| {
        block_delta(0, bedrock::ContentBlockDelta::ReasoningContent(delta))
    };
    let (builder, _, stream) = feed(
        &model,
        vec![
            message_start(),
            reasoning(bedrock::ReasoningContentBlockDelta::Text("Let me ".into())),
            reasoning(bedrock::ReasoningContentBlockDelta::Text("think".into())),
            reasoning(bedrock::ReasoningContentBlockDelta::Signature("sig-a".into())),
            reasoning(bedrock::ReasoningContentBlockDelta::Signature("sig-b".into())),
            block_stop(0),
            block_delta(1, bedrock::ContentBlockDelta::Text("Answer".into())),
            block_stop(1),
            message_stop("end_turn"),
        ],
    );
    assert_eq!(
        builder.output.content[0],
        AssistantBlock::Thinking(ThinkingContent {
            thinking: "Let me think".into(),
            thinking_signature: Some("sig-asig-b".into()),
            redacted: None,
        })
    );
    assert_eq!(
        builder.output.content[1],
        AssistantBlock::Text(TextContent::new("Answer"))
    );
    assert_eq!(
        kinds(builder, stream).await,
        [
            "start",
            "thinking_start",
            "thinking_delta",
            "thinking_delta",
            "thinking_end",
            "text_start",
            "text_delta",
            "text_end",
            "done"
        ]
    );
}

#[test]
fn redacted_reasoning_is_preserved_and_joined() {
    let model = gpt_56();
    let bytes = b"\x01\x02encrypted\xfe\xff reasoning";
    let (head, tail) = bytes.split_at(7);
    let (builder, _, _stream) = feed(
        &model,
        vec![
            message_start(),
            redacted_delta(0, head),
            redacted_delta(0, tail),
            block_stop(0),
            block_delta(1, bedrock::ContentBlockDelta::Text("done".into())),
            block_stop(1),
            message_stop("end_turn"),
        ],
    );
    assert_eq!(builder.output.stop_reason, StopReason::Stop);
    assert_eq!(
        builder.output.content,
        vec![
            AssistantBlock::Thinking(ThinkingContent {
                thinking: REDACTED_THINKING_PLACEHOLDER.into(),
                thinking_signature: Some(STANDARD.encode(bytes)),
                redacted: Some(true),
            }),
            AssistantBlock::Text(TextContent::new("done")),
        ]
    );
    // A signature delta never mixes into a redacted payload.
    let (builder, _, _stream) = feed(
        &model,
        vec![
            message_start(),
            redacted_delta(0, b"abc"),
            block_delta(
                0,
                bedrock::ContentBlockDelta::ReasoningContent(bedrock::ReasoningContentBlockDelta::Signature(
                    "sig".into(),
                )),
            ),
            block_stop(0),
        ],
    );
    let thinking = builder.output.content[0].as_thinking().unwrap();
    assert_eq!(thinking.thinking_signature, Some(STANDARD.encode(b"abc")));
}

#[test]
fn redacted_reasoning_is_encoded_without_block_stop() {
    let model = gpt_56();
    let (mut builder, mut state, _stream) = feed(
        &model,
        vec![message_start(), redacted_delta(0, b"payload"), message_stop("end_turn")],
    );
    state.finalize(&mut builder);
    let thinking = builder.output.content[0].as_thinking().unwrap();
    assert_eq!(thinking.thinking_signature, Some(STANDARD.encode(b"payload")));
    assert!(thinking.is_redacted());
    // The persisted message carries no scratch fields.
    let value = serde_json::to_value(&builder.output.content[0]).unwrap();
    assert_eq!(
        value,
        json!({"type": "thinking", "thinking": "[Reasoning redacted]", "thinkingSignature": STANDARD.encode(b"payload"), "redacted": true})
    );
}

#[test]
fn partial_tool_arguments_are_kept_on_failure() {
    let model = sonnet_45();
    let (mut builder, state, _stream) = feed(
        &model,
        vec![
            message_start(),
            tool_start(0, "t", "write"),
            tool_delta(0, r#"{"path": "a.t"#),
        ],
    );
    state.keep_partial_tool_arguments(&mut builder);
    let call = builder.output.tool_calls().next().unwrap();
    assert_eq!(Value::Object(call.arguments.clone()), json!({"path": "a.t"}));
}

#[test]
fn usage_and_one_hour_cache_write_cost() {
    let model = catalog_model("us.anthropic.claude-opus-4-8");
    let detail = |ttl: &str, tokens: i32| {
        bedrock::CacheDetail::builder()
            .ttl(bedrock::CacheTtl::from(ttl))
            .input_tokens(tokens)
            .build()
            .unwrap()
    };
    let usage = bedrock::TokenUsage::builder()
        .input_tokens(100)
        .output_tokens(5)
        .total_tokens(1_000_105)
        .cache_write_input_tokens(1_000_000)
        .cache_details(detail("1h", 150_000))
        .cache_details(detail("5m", 600_000))
        .cache_details(detail("1h", 250_000))
        .build()
        .unwrap();
    let metadata =
        bedrock::ConverseStreamOutput::Metadata(bedrock::ConverseStreamMetadataEvent::builder().usage(usage).build());
    let (builder, _, _stream) = feed(&model, vec![message_start(), metadata]);
    let usage = &builder.output.usage;
    assert_eq!(usage.cache_write, 1_000_000);
    assert_eq!(usage.cache_write_1h, Some(400_000));
    assert_eq!(usage.total_tokens, 1_000_105);
    let expected = (600_000.0 * model.cost.cache_write + 400_000.0 * model.cost.input * 2.0) / 1_000_000.0;
    assert!((usage.cost.cache_write - expected).abs() < 1e-9);

    // Without cache details the 1h split stays unset; a zero total falls back to input + output.
    let usage = bedrock::TokenUsage::builder()
        .input_tokens(3)
        .output_tokens(4)
        .total_tokens(0)
        .build()
        .unwrap();
    let metadata =
        bedrock::ConverseStreamOutput::Metadata(bedrock::ConverseStreamMetadataEvent::builder().usage(usage).build());
    let (builder, _, _stream) = feed(&model, vec![metadata]);
    assert_eq!(builder.output.usage.cache_write_1h, None);
    assert_eq!(builder.output.usage.total_tokens, 7);
}

#[test]
fn provider_events_as_json() {
    assert_eq!(
        event_to_json(&message_start()),
        json!({"messageStart": {"role": "assistant"}})
    );
    assert_eq!(
        event_to_json(&tool_start(1, "t", "read")),
        json!({"contentBlockStart": {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "t", "name": "read"}}}})
    );
    assert_eq!(
        event_to_json(&redacted_delta(0, b"ab")),
        json!({"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"redactedContent": "YWI="}}}})
    );
    let stop = bedrock::ConverseStreamOutput::MessageStop(
        bedrock::MessageStopEvent::builder()
            .stop_reason(bedrock::StopReason::EndTurn)
            .additional_model_response_fields(json_to_document(&json!({"source": "test"})))
            .build()
            .unwrap(),
    );
    assert_eq!(
        event_to_json(&stop),
        json!({"messageStop": {"stopReason": "end_turn", "additionalModelResponseFields": {"source": "test"}}})
    );
    assert_eq!(
        event_to_json(&block_stop(3)),
        json!({"contentBlockStop": {"contentBlockIndex": 3}})
    );
}

// ---------------------------------------------------------------------------------------------
// Failures and headers
// ---------------------------------------------------------------------------------------------

#[test]
fn failure_messages_and_diagnostics() {
    let info = FailureInfo {
        core: "The provided model identifier is invalid.".into(),
        service_name: Some("ValidationException".into()),
        status: Some(400),
        error_code: Some("ValidationException".into()),
        request_id: Some("req-1".into()),
    };
    assert_eq!(
        info.message(),
        "Validation error: The provided model identifier is invalid."
    );
    assert_eq!(
        Value::Object(info.diagnostic_details(None).unwrap()),
        json!({"status": 400, "errorCode": "ValidationException", "requestId": "req-1"})
    );

    let unknown = FailureInfo {
        core: "403: <html>Forbidden</html>".into(),
        service_name: Some("Unknown".into()),
        status: Some(403),
        error_code: Some("Unknown".into()),
        request_id: Some("req".into()),
    };
    assert_eq!(unknown.message(), "Unknown: 403: <html>Forbidden</html>");
    assert_eq!(
        Value::Object(unknown.diagnostic_details(None).unwrap()),
        json!({"status": 403, "requestId": "req"})
    );

    let long = FailureInfo {
        status: Some(400),
        error_code: Some(format!("{}Exception", "E".repeat(5000))),
        request_id: normalize_diagnostic_value(Some(&"R".repeat(5000))),
        ..FailureInfo::default()
    };
    assert_eq!(
        Value::Object(long.diagnostic_details(Some("fallback")).unwrap()),
        json!({"status": 400, "requestId": "fallback"})
    );
    assert_eq!(FailureInfo::plain("socket hang up").diagnostic_details(None), None);
    let timeout = FailureInfo {
        core: "Connection timed out".into(),
        error_code: Some("TimeoutError".into()),
        ..FailureInfo::default()
    };
    assert_eq!(
        Value::Object(timeout.diagnostic_details(Some("req")).unwrap()),
        json!({"requestId": "req"})
    );

    let retention = FailureInfo::plain("data retention mode 'default' is not available for this model");
    assert!(retention.message().ends_with(&format!(
        " See {DATA_RETENTION_DOCS_URL} for supported data retention modes."
    )));
    let throttled = FailureInfo {
        core: "slow down".into(),
        service_name: Some("ThrottlingException".into()),
        ..FailureInfo::default()
    };
    assert_eq!(throttled.message(), "Throttling error: slow down");
}

#[test]
fn custom_headers_skip_reserved_names() {
    let headers: ProviderHeaders = [
        ("authorization", Some("evil")),
        ("x-amz-date", Some("evil")),
        ("x-allowed", Some("ok")),
        ("Authorization", Some("evil2")),
        ("X-Amz-Date", Some("evil2")),
        ("HOST", Some("evil3")),
        ("X-Removed", Some("x")),
        ("x-removed", None),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_owned(), value.map(str::to_owned)))
    .collect();
    assert_eq!(
        custom_headers(Some(&headers)).unwrap(),
        Some(vec![("x-allowed".to_owned(), "ok".to_owned())])
    );
    assert_eq!(custom_headers(None).unwrap(), None);
    assert_eq!(custom_headers(Some(&ProviderHeaders::new())).unwrap(), None);
    let invalid: ProviderHeaders = [("bad header".to_owned(), Some("v".to_owned()))].into_iter().collect();
    assert!(custom_headers(Some(&invalid)).is_err());
}

// ---------------------------------------------------------------------------------------------
// stream / stream_simple without network (the payload hook stops the request)
// ---------------------------------------------------------------------------------------------

async fn captured_payload(model: &Model, context: &Context, simple: SimpleStreamOptions) -> Value {
    let captured = Arc::new(Mutex::new(None));
    let sink = captured.clone();
    let mut simple = simple;
    simple.base.on_payload = Some(Arc::new(move |payload: &Value, _: &Model| {
        *sink.lock().unwrap() = Some(payload.clone());
        Box::pin(async { Err::<Option<Value>, _>("payload captured".into()) })
    }));
    let message = stream_simple(model, &normalize_context(context), simple).result().await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_message.as_deref(), Some("payload captured"));
    assert_eq!(message.diagnostics, None);
    captured.lock().unwrap().take().unwrap()
}

#[tokio::test]
async fn stream_simple_maps_reasoning_to_budgets() {
    let model = sonnet_45();
    let context = Context::new(vec![Message::user("hi")]);
    let simple = SimpleStreamOptions {
        base: StreamOptions {
            max_tokens: Some(1000),
            ..StreamOptions::default()
        },
        reasoning: Some(ThinkingLevel::Medium),
        tool_choice: Some(ToolChoice::Auto),
        ..SimpleStreamOptions::default()
    };
    let sent = captured_payload(&model, &context, simple).await;
    // max_tokens = base + budget (capped by the model), budget kept below max_tokens - 1024.
    assert_eq!(sent["inferenceConfig"]["maxTokens"], json!(9192));
    assert_eq!(
        sent["additionalModelRequestFields"]["thinking"]["budget_tokens"],
        json!(8168)
    );

    // Adaptive models keep the clamped max tokens and send an effort.
    let adaptive = catalog_model("global.anthropic.claude-sonnet-4-6");
    let simple = SimpleStreamOptions {
        reasoning: Some(ThinkingLevel::Low),
        ..SimpleStreamOptions::default()
    };
    let sent = captured_payload(&adaptive, &context, simple).await;
    assert_eq!(sent["inferenceConfig"]["maxTokens"], json!(adaptive.max_tokens));
    assert_eq!(
        sent["additionalModelRequestFields"]["output_config"],
        json!({"effort": "low"})
    );

    // No reasoning: no thinking fields.
    let sent = captured_payload(&model, &context, SimpleStreamOptions::default()).await;
    assert_eq!(sent.get("additionalModelRequestFields"), None);
}

#[tokio::test]
async fn stream_fails_before_network_on_invalid_input() {
    let model = sonnet_45();
    let context = normalize_context(&Context::new(vec![Message::User(UserMessage {
        content: UserContent::Blocks(vec![UserBlock::Image(ImageContent {
            data: "AAAA".into(),
            mime_type: "image/tiff".into(),
        })]),
        timestamp: 0,
    })]));
    let message = stream(&model, &context, StreamOptions::default()).result().await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_message.as_deref(), Some("Unknown image type: image/tiff"));
}

#[tokio::test]
async fn aborted_before_start() {
    let model = sonnet_45();
    let cancel = CancellationToken::new();
    cancel.cancel();
    let options = StreamOptions {
        cancel: Some(cancel),
        ..StreamOptions::default()
    };
    let message = stream(
        &model,
        &normalize_context(&Context::new(vec![Message::user("hi")])),
        options,
    )
    .result()
    .await;
    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert_eq!(message.diagnostics, None);
}
