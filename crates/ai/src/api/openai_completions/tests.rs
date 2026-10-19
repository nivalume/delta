//! Unit tests of request building, message conversion and chunk helpers.

use super::*;
use crate::model::ThinkingLevel;
use crate::options::ThinkingBudgets;
use crate::transcript::normalize_context;
use crate::types::{
    Context, ImageContent, SystemMessage, TextContent, ThinkingContent, ToolCall, ToolReference, UserMessage,
};
use pretty_assertions::assert_eq;

fn model_json(provider: &str, base_url: &str, id: &str, reasoning: bool, compat: Option<Value>) -> Model {
    let mut value = json!({
        "id": id, "name": id, "api": "openai-completions", "provider": provider, "baseUrl": base_url,
        "reasoning": reasoning, "input": ["text", "image"],
        "cost": {"input": 1, "output": 2, "cacheRead": 0.5, "cacheWrite": 0},
        "contextWindow": 128_000, "maxTokens": 16384
    });
    if let Some(compat) = compat {
        value["compat"] = compat;
    }
    serde_json::from_value(value).unwrap()
}

fn openai() -> Model {
    model_json("openai", "https://api.openai.com/v1", "gpt-4o-mini", false, None)
}

fn local(compat: Value) -> Model {
    model_json("local-vllm", "http://localhost:8000/v1", "glm", true, Some(compat))
}

fn ctx(messages: Vec<Message>) -> TranscriptContext {
    normalize_context(&Context::new(messages))
}

fn ctx_with(system: &str, tools: Vec<Tool>, messages: Vec<Message>) -> TranscriptContext {
    normalize_context(&Context::new(messages).with_system_prompt(system).with_tools(tools))
}

fn read_tool() -> Tool {
    Tool::new(
        "read",
        "Read a file",
        json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
    )
}

fn hi() -> Vec<Message> {
    vec![Message::user("Hi")]
}

fn with_effort(effort: Option<ThinkingLevel>) -> StreamOptions {
    StreamOptions {
        api: Some(ApiOptions::OpenAICompletions(OpenAICompletionsOptions {
            reasoning_effort: effort.map(ReasoningEffort::from),
            ..OpenAICompletionsOptions::default()
        })),
        ..StreamOptions::default()
    }
}

fn params(model: &Model, context: &TranscriptContext, options: &StreamOptions) -> Value {
    build_params(model, context, options).unwrap()
}

fn assistant(content: Vec<AssistantBlock>, model: &Model) -> Message {
    let mut message = AssistantMessage::empty(&model.api, &model.provider, &model.id);
    message.content = content;
    message.stop_reason = if message.tool_calls().next().is_some() {
        StopReason::ToolUse
    } else {
        StopReason::Stop
    };
    Message::Assistant(message)
}

fn call(id: &str, name: &str, arguments: Value) -> AssistantBlock {
    AssistantBlock::ToolCall(ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: match arguments {
            Value::Object(object) => object,
            _ => JsonObject::new(),
        },
        ..ToolCall::default()
    })
}

fn result(id: &str, name: &str, content: Vec<UserBlock>) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: id.into(),
        tool_name: name.into(),
        content,
        ..ToolResultMessage::default()
    })
}

fn convert(model: &Model, context: &TranscriptContext) -> Vec<Value> {
    convert_messages(model, context, &get_compat(model), &HashMap::new()).unwrap()
}

// ---------------------------------------------------------------------------
// Compat detection
// ---------------------------------------------------------------------------

#[test]
fn detects_compat_from_provider_and_url() {
    let c = detect_compat(&openai());
    assert!(c.supports_store && c.supports_developer_role && c.supports_reasoning_effort);
    assert_eq!(c.max_tokens_field, MaxTokensField::MaxCompletionTokens);
    assert_eq!(c.thinking_format, ThinkingFormat::Openai);
    assert!(!c.supports_strict_mode);
    assert!(!c.send_session_affinity_headers);

    let deepseek = detect_compat(&model_json("x", "https://API.DeepSeek.com/v1", "m", true, None));
    assert_eq!(deepseek.thinking_format, ThinkingFormat::Deepseek);
    assert_eq!(deepseek.max_tokens_field, MaxTokensField::MaxTokens);
    assert!(deepseek.requires_reasoning_content_on_assistant_messages && !deepseek.supports_store);

    let zai = detect_compat(&model_json("zai", "https://api.z.ai/api/paas/v4", "glm", true, None));
    assert_eq!(zai.thinking_format, ThinkingFormat::Zai);
    assert!(!zai.supports_reasoning_effort);

    let xai = detect_compat(&model_json("xai", "https://api.x.ai/v1", "grok", true, None));
    assert!(!xai.supports_reasoning_effort && !xai.supports_store && !xai.supports_developer_role);
    assert_eq!(xai.max_tokens_field, MaxTokensField::MaxCompletionTokens);

    let chutes = detect_compat(&model_json("c", "https://llm.chutes.ai/v1", "m", true, None));
    assert_eq!(chutes.max_tokens_field, MaxTokensField::MaxTokens);

    let together = detect_compat(&model_json("together", "https://api.together.xyz/v1", "m", true, None));
    assert_eq!(together.thinking_format, ThinkingFormat::Together);
    assert!(!together.supports_long_cache_retention);

    let open_router = detect_compat(&model_json(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "anthropic/claude",
        true,
        None,
    ));
    assert_eq!(open_router.thinking_format, ThinkingFormat::Openrouter);
    assert!(open_router.supports_developer_role && open_router.send_session_affinity_headers);
    assert_eq!(open_router.session_affinity_format, SessionAffinityFormat::Openrouter);
    assert_eq!(open_router.cache_control_format.as_deref(), Some("anthropic"));
    let qwen_or = detect_compat(&model_json(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "qwen/q",
        true,
        None,
    ));
    assert!(!qwen_or.supports_developer_role && qwen_or.cache_control_format.is_none());

    let ant_ling = detect_compat(&model_json("ant-ling", "https://api.ant-ling.com/v1", "m", true, None));
    assert_eq!(ant_ling.thinking_format, ThinkingFormat::AntLing);
    let gateway = detect_compat(&model_json(
        "x",
        "https://gateway.ai.cloudflare.com/v1/a/b/compat",
        "m",
        true,
        None,
    ));
    assert_eq!(gateway.max_tokens_field, MaxTokensField::MaxTokens);
    assert!(!gateway.supports_reasoning_effort && !gateway.supports_long_cache_retention);
    for (provider, url) in [
        ("cerebras", "https://api.cerebras.ai/v1"),
        ("nvidia", "https://integrate.api.nvidia.com/v1"),
        ("opencode", "https://opencode.ai/zen/v1"),
        ("moonshotai", "https://api.moonshot.ai/v1"),
        ("cloudflare-workers-ai", "https://api.cloudflare.com/client/v4"),
    ] {
        assert!(
            !detect_compat(&model_json(provider, url, "m", true, None)).supports_store,
            "{provider}"
        );
    }
}

#[test]
fn explicit_compat_overrides_detection() {
    let model = model_json(
        "deepseek",
        "https://api.deepseek.com",
        "m",
        true,
        Some(
            json!({"maxTokensField": "max_completion_tokens", "thinkingFormat": "openai", "supportsStrictMode": true, "vllmPriority": 5}),
        ),
    );
    let c = get_compat(&model);
    assert_eq!(c.max_tokens_field, MaxTokensField::MaxCompletionTokens);
    assert_eq!(c.thinking_format, ThinkingFormat::Openai);
    assert!(c.supports_strict_mode);
    assert!(c.requires_reasoning_content_on_assistant_messages);
    assert_eq!(c.vllm_priority, Some(5.0));
}

// ---------------------------------------------------------------------------
// Request parameters
// ---------------------------------------------------------------------------

#[test]
fn builds_the_basic_request() {
    let context = ctx_with("sys", vec![], hi());
    let options = StreamOptions {
        max_tokens: Some(100),
        temperature: Some(0.5),
        ..StreamOptions::default()
    };
    assert_eq!(
        params(&openai(), &context, &options),
        json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "Hi"}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "store": false,
            "max_completion_tokens": 100,
            "temperature": 0.5
        })
    );
}

#[test]
fn uses_max_tokens_and_drops_store_for_non_standard_servers() {
    let model = model_json("deepseek", "https://api.deepseek.com", "deepseek-chat", false, None);
    let options = StreamOptions {
        max_tokens: Some(1234),
        ..StreamOptions::default()
    };
    let body = params(&model, &ctx(hi()), &options);
    assert_eq!(body["max_tokens"], 1234);
    assert!(body.get("max_completion_tokens").is_none() && body.get("store").is_none());
    let model = local(json!({"supportsUsageInStreaming": false}));
    assert!(params(&model, &ctx(hi()), &options).get("stream_options").is_none());
}

#[test]
fn instruction_role_follows_reasoning_and_developer_support() {
    let context = ctx_with("sys", vec![], hi());
    let reasoning = model_json("openai", "https://api.openai.com/v1", "o3", true, None);
    assert_eq!(convert(&reasoning, &context)[0]["role"], "developer");
    assert_eq!(convert(&openai(), &context)[0]["role"], "system");
    let qwen = model_json("openrouter", "https://openrouter.ai/api/v1", "qwen/qwen3", true, None);
    assert_eq!(convert(&qwen, &context)[0]["role"], "system");
    let claude = model_json(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "anthropic/claude-x",
        true,
        None,
    );
    assert_eq!(convert(&claude, &context)[0]["role"], "developer");
}

#[test]
fn prompt_cache_key_and_retention() {
    let context = ctx_with("sys", vec![], hi());
    let session = |retention: Option<CacheRetention>, id: &str| StreamOptions {
        session_id: Some(id.into()),
        cache_retention: retention,
        ..StreamOptions::default()
    };
    let body = params(&openai(), &context, &session(None, "session-123"));
    assert_eq!(body["prompt_cache_key"], "session-123");
    assert!(body.get("prompt_cache_retention").is_none());

    let body = params(&openai(), &context, &session(Some(CacheRetention::Long), "s"));
    assert_eq!(body["prompt_cache_retention"], "24h");

    let body = params(&openai(), &context, &session(None, &"x".repeat(67)));
    assert_eq!(body["prompt_cache_key"], "x".repeat(64));

    let body = params(&openai(), &context, &session(Some(CacheRetention::None), "s"));
    assert!(body.get("prompt_cache_key").is_none());

    let proxy = model_json(
        "openai",
        "https://proxy.example.com/v1",
        "m",
        false,
        Some(json!({"supportsLongCacheRetention": false})),
    );
    let body = params(&proxy, &context, &session(Some(CacheRetention::Long), "s"));
    assert!(body.get("prompt_cache_key").is_none() && body.get("prompt_cache_retention").is_none());
    let proxy = model_json("openai", "https://proxy.example.com/v1", "m", false, None);
    assert_eq!(
        params(&proxy, &context, &session(Some(CacheRetention::Long), "s"))["prompt_cache_key"],
        "s"
    );
    assert!(
        params(&proxy, &context, &session(None, "s"))
            .get("prompt_cache_key")
            .is_none()
    );

    let mut env_options = session(None, "session-env");
    env_options.env = Some([("PI_CACHE_RETENTION".to_owned(), "long".to_owned())].into());
    assert_eq!(
        params(&openai(), &context, &env_options)["prompt_cache_retention"],
        "24h"
    );
}

#[test]
fn applies_anthropic_cache_markers() {
    let model = model_json(
        "openrouter",
        "https://example.com/v1",
        "custom-qwen",
        true,
        Some(json!({"cacheControlFormat": "anthropic"})),
    );
    let context = ctx_with("System prompt", vec![read_tool()], vec![Message::user("Hello")]);
    let body = params(&model, &context, &StreamOptions::default());
    let marker = json!({"type": "ephemeral"});
    assert_eq!(body["messages"][0]["content"][0]["cache_control"], marker);
    assert_eq!(body["tools"][0]["cache_control"], marker);
    assert_eq!(
        body["messages"][1]["content"],
        json!([{"type": "text", "text": "Hello", "cache_control": marker}])
    );

    let long = StreamOptions {
        cache_retention: Some(CacheRetention::Long),
        ..StreamOptions::default()
    };
    assert_eq!(
        params(&model, &context, &long)["tools"][0]["cache_control"],
        json!({"type": "ephemeral", "ttl": "1h"})
    );

    let none = StreamOptions {
        cache_retention: Some(CacheRetention::None),
        ..StreamOptions::default()
    };
    let body = params(&model, &context, &none);
    assert!(body["messages"][0]["content"].is_string());
    assert!(body["tools"][0].get("cache_control").is_none());

    // The conversation marker moves to the last tool result.
    let context = ctx_with(
        "System prompt",
        vec![read_tool()],
        vec![
            Message::user("Read the file"),
            assistant(vec![call("call_1", "read", json!({"path": "README.md"}))], &model),
            result("call_1", "read", vec![UserBlock::text("file contents")]),
        ],
    );
    let body = params(&model, &context, &StreamOptions::default());
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages[1]["content"], "Read the file");
    let last = messages.last().unwrap();
    assert_eq!(last["role"], "tool");
    assert_eq!(last["content"][0]["cache_control"], marker);
}

#[test]
fn tools_tool_choice_and_tool_history() {
    let context = ctx_with("", vec![read_tool()], hi());
    let body = params(&openai(), &context, &StreamOptions::default());
    // Unknown endpoints do not get `strict`.
    assert_eq!(
        body["tools"],
        json!([{"type": "function", "function": {"name": "read", "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}])
    );

    let strict_model = model_json(
        "openai",
        "https://api.openai.com/v1",
        "gpt-5",
        false,
        Some(json!({"supportsStrictMode": true})),
    );
    let mut tool = read_tool();
    tool.name = "strict_read".into();
    tool.parameters = json!({"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "number"}}, "required": ["path"]});
    tool.constrained_sampling = Some(crate::types::ConstrainedSampling::JsonSchema {
        strict: crate::types::StrictMode::Prefer,
    });
    let body = params(
        &strict_model,
        &ctx_with("", vec![tool, read_tool()], hi()),
        &StreamOptions::default(),
    );
    assert_eq!(body["tools"][0]["function"]["strict"], true);
    assert_eq!(
        body["tools"][0]["function"]["parameters"]["required"],
        json!(["path", "offset"])
    );
    assert_eq!(body["tools"][1]["function"]["strict"], false);

    // tool_choice is forwarded even without tools.
    let options = StreamOptions {
        api: Some(ApiOptions::OpenAICompletions(OpenAICompletionsOptions {
            tool_choice: Some(json!("required")),
            ..OpenAICompletionsOptions::default()
        })),
        ..StreamOptions::default()
    };
    let body = params(&openai(), &ctx(hi()), &options);
    assert_eq!(body["tool_choice"], "required");
    assert!(body.get("tools").is_none());

    // Tool history without current tools sends `tools: []`.
    let model = openai();
    let history = ctx(vec![
        Message::user("read"),
        assistant(vec![call("c1", "read", json!({"path": "a"}))], &model),
        result("c1", "read", vec![UserBlock::text("ok")]),
    ]);
    assert_eq!(params(&model, &history, &StreamOptions::default())["tools"], json!([]));

    // z.ai tool_stream only with tools.
    let zai = model_json(
        "zai",
        "https://api.z.ai/api/coding/paas/v4",
        "glm-5",
        true,
        Some(json!({"zaiToolStream": true})),
    );
    assert_eq!(params(&zai, &context, &StreamOptions::default())["tool_stream"], true);
    assert!(
        params(&zai, &ctx(hi()), &StreamOptions::default())
            .get("tool_stream")
            .is_none()
    );
}

#[test]
fn grammar_tools_and_strict_requirements() {
    let model = model_json(
        "openai",
        "https://api.openai.com/v1",
        "gpt-5",
        false,
        Some(json!({"supportsOpenAIGrammarTools": true})),
    );
    let mut tool = Tool::new(
        "apply_patch",
        "Apply a patch",
        json!({"type": "object", "properties": {"patch": {"type": "string"}}, "required": ["patch"]}),
    );
    tool.constrained_sampling = Some(crate::types::ConstrainedSampling::Grammar {
        variants: crate::types::GrammarVariants {
            openai_lark: Some("start: /.+/".into()),
            openai_regex: None,
        },
    });
    let history = vec![
        Message::user("patch"),
        assistant(vec![call("c1", "apply_patch", json!({"patch": "diff"}))], &model),
        result("c1", "apply_patch", vec![UserBlock::text("done")]),
    ];
    let body = params(
        &model,
        &ctx_with("", vec![tool.clone()], history),
        &StreamOptions::default(),
    );
    assert_eq!(
        body["tools"],
        json!([{"type": "custom", "custom": {"name": "apply_patch", "description": "Apply a patch",
            "format": {"type": "grammar", "grammar": {"syntax": "lark", "definition": "start: /.+/"}}}}])
    );
    assert_eq!(
        body["messages"][1]["tool_calls"],
        json!([{"id": "c1", "type": "custom", "custom": {"name": "apply_patch", "input": "diff"}}])
    );

    // Without grammar support the tool falls back to a function tool.
    let plain = params(&openai(), &ctx_with("", vec![tool], hi()), &StreamOptions::default());
    assert_eq!(plain["tools"][0]["type"], "function");

    let mut required = read_tool();
    required.constrained_sampling = Some(crate::types::ConstrainedSampling::JsonSchema {
        strict: crate::types::StrictMode::Require,
    });
    let error = build_params(
        &openai(),
        &ctx_with("", vec![required], hi()),
        &StreamOptions::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("strict tools are unsupported"), "{error}");
}

#[test]
fn reasoning_effort_formats() {
    let effort = |model: &Model, level: Option<ThinkingLevel>| params(model, &ctx(hi()), &with_effort(level));

    let o3 = model_json("openai", "https://api.openai.com/v1", "o3", true, None);
    assert_eq!(effort(&o3, Some(ThinkingLevel::High))["reasoning_effort"], "high");
    assert!(effort(&o3, None).get("reasoning_effort").is_none());
    // Non-reasoning models never get reasoning fields.
    assert!(
        effort(&openai(), Some(ThinkingLevel::High))
            .get("reasoning_effort")
            .is_none()
    );

    let mut mapped = o3.clone();
    mapped.thinking_level_map = Some(
        [
            (ModelThinkingLevel::Off, Some("none".to_owned())),
            (ModelThinkingLevel::High, Some("default".to_owned())),
        ]
        .into(),
    );
    assert_eq!(
        effort(&mapped, Some(ThinkingLevel::High))["reasoning_effort"],
        "default"
    );
    assert_eq!(effort(&mapped, None)["reasoning_effort"], "none");

    let zai = model_json(
        "zai",
        "https://api.z.ai/api/paas/v4",
        "glm",
        true,
        Some(json!({"supportsReasoningEffort": true})),
    );
    let body = effort(&zai, Some(ThinkingLevel::Medium));
    assert_eq!(body["thinking"], json!({"type": "enabled", "clear_thinking": false}));
    assert_eq!(body["reasoning_effort"], "medium");
    let body = effort(&zai, None);
    assert_eq!(body["thinking"], json!({"type": "disabled"}));
    assert!(body.get("reasoning_effort").is_none());

    let qwen = local(json!({"thinkingFormat": "qwen"}));
    assert_eq!(effort(&qwen, Some(ThinkingLevel::Low))["enable_thinking"], true);
    assert_eq!(effort(&qwen, Some(ThinkingLevel::Low))["reasoning_effort"], "low");
    assert_eq!(effort(&qwen, None)["enable_thinking"], false);

    let qwen_template = local(json!({"thinkingFormat": "qwen-chat-template"}));
    assert_eq!(
        effort(&qwen_template, None)["chat_template_kwargs"],
        json!({"enable_thinking": false, "preserve_thinking": true})
    );

    let deepseek = model_json("deepseek", "https://api.deepseek.com", "deepseek-v4", true, None);
    let body = effort(&deepseek, Some(ThinkingLevel::High));
    assert_eq!(body["thinking"], json!({"type": "enabled"}));
    assert_eq!(body["reasoning_effort"], "high");
    assert_eq!(effort(&deepseek, None)["thinking"], json!({"type": "disabled"}));
    let mut no_off = deepseek.clone();
    no_off.thinking_level_map = Some([(ModelThinkingLevel::Off, None)].into());
    assert!(effort(&no_off, None).get("thinking").is_none());

    let open_router = model_json("openrouter", "https://openrouter.ai/api/v1", "x/y", true, None);
    let body = effort(&open_router, Some(ThinkingLevel::High));
    assert_eq!(body["reasoning"], json!({"effort": "high"}));
    assert!(body.get("reasoning_effort").is_none());
    assert_eq!(effort(&open_router, None)["reasoning"], json!({"effort": "none"}));

    let together = model_json("together", "https://api.together.xyz/v1", "m", true, None);
    assert_eq!(
        effort(&together, Some(ThinkingLevel::High))["reasoning"],
        json!({"enabled": true})
    );
    assert!(
        effort(&together, Some(ThinkingLevel::High))
            .get("reasoning_effort")
            .is_none()
    );

    let mut ant_ling = model_json("ant-ling", "https://api.ant-ling.com/v1", "ling", true, None);
    assert!(effort(&ant_ling, Some(ThinkingLevel::High)).get("reasoning").is_none());
    ant_ling.thinking_level_map = Some([(ModelThinkingLevel::High, Some("deep".to_owned()))].into());
    assert_eq!(
        effort(&ant_ling, Some(ThinkingLevel::High))["reasoning"],
        json!({"effort": "deep"})
    );

    let string_thinking = local(json!({"thinkingFormat": "string-thinking"}));
    assert_eq!(effort(&string_thinking, Some(ThinkingLevel::Low))["thinking"], "low");
    assert_eq!(effort(&string_thinking, None)["thinking"], "none");

    let baseten = local(
        json!({"thinkingFormat": "baseten", "chatTemplateArgs": {"enable_thinking": {"$var": "thinking.enabled"}}}),
    );
    let body = effort(&baseten, Some(ThinkingLevel::Medium));
    assert_eq!(body["chat_template_args"], json!({"enable_thinking": true}));
    assert_eq!(body["reasoning_effort"], "medium");
    assert!(effort(&baseten, None).get("reasoning_effort").is_none());
}

#[test]
fn chat_template_kwargs_and_thinking_budgets() {
    let model = local(json!({
        "thinkingFormat": "chat-template",
        "chatTemplateKwargs": {
            "enable_thinking": {"$var": "thinking.enabled"},
            "effort": {"$var": "thinking.effort", "omitWhenOff": true},
            "thinking_budget": {"$var": "thinking.budget"},
            "static": "x"
        }
    }));
    let body = params(&model, &ctx(hi()), &with_effort(Some(ThinkingLevel::High)));
    assert_eq!(
        body["chat_template_kwargs"],
        json!({"enable_thinking": true, "effort": "high", "thinking_budget": 16384 - 1024, "static": "x"})
    );
    assert!(body.get("thinking_token_budget").is_none());
    let body = params(&model, &ctx(hi()), &with_effort(None));
    assert_eq!(
        body["chat_template_kwargs"],
        json!({"enable_thinking": false, "static": "x"})
    );

    let budget = |compat: Value, level: Option<ThinkingLevel>, budgets: Option<ThinkingBudgets>, max: Option<u64>| {
        let options = StreamOptions {
            max_tokens: max,
            api: Some(ApiOptions::OpenAICompletions(OpenAICompletionsOptions {
                reasoning_effort: level.map(ReasoningEffort::from),
                thinking_budgets: budgets,
                ..OpenAICompletionsOptions::default()
            })),
            ..StreamOptions::default()
        };
        params(&local(compat), &ctx(hi()), &options)
    };
    let medium = Some(ThinkingBudgets {
        medium: Some(4096),
        high: Some(8192),
        ..ThinkingBudgets::default()
    });
    let vllm = json!({"supportsThinkingTokenBudget": true});
    assert_eq!(
        budget(vllm.clone(), Some(ThinkingLevel::Medium), medium, None)["thinking_token_budget"],
        4096
    );
    assert!(
        budget(vllm.clone(), None, medium, None)
            .get("thinking_token_budget")
            .is_none()
    );
    assert_eq!(
        budget(vllm.clone(), Some(ThinkingLevel::Max), medium, None)["thinking_token_budget"],
        8192
    );
    assert_eq!(
        budget(vllm.clone(), Some(ThinkingLevel::High), None, None)["thinking_token_budget"],
        16384 - 1024
    );
    assert_eq!(
        budget(vllm, Some(ThinkingLevel::High), medium, Some(4096))["thinking_token_budget"],
        4096 - 1024
    );
    let field = json!({"thinkingFormat": "zai", "supportsThinkingTokenBudget": true, "thinkingTokenBudgetField": "thinking_budget"});
    let body = budget(field, Some(ThinkingLevel::Medium), medium, None);
    assert_eq!(body["thinking_budget"], 4096);
    assert!(body.get("thinking_token_budget").is_none());
    let none = budget(
        json!({"thinkingFormat": "zai"}),
        Some(ThinkingLevel::Medium),
        medium,
        None,
    );
    assert!(none.get("thinking_budget").is_none() && none.get("thinking_token_budget").is_none());
}

#[test]
fn priority_routing_and_sampling_params() {
    let model = local(json!({
        "vllmPriority": 5,
        "openRouterRouting": {"order": ["a"]},
        "vercelGatewayRouting": {"only": ["x"], "models": ["y"]}
    }));
    let options = StreamOptions {
        temperature: Some(0.2),
        sampling_params: Some(json!({"temperature": 0.9, "top_k": 20}).as_object().cloned().unwrap()),
        ..StreamOptions::default()
    };
    let body = params(&model, &ctx(hi()), &options);
    assert_eq!(body["priority"], json!(5));
    assert_eq!(body["provider"], json!({"order": ["a"]}));
    assert_eq!(body["providerOptions"], json!({"gateway": {"only": ["x"]}}));
    assert_eq!(body["temperature"], 0.9);
    assert_eq!(body["top_k"], 20);
    assert!(
        params(&openai(), &ctx(hi()), &StreamOptions::default())
            .get("priority")
            .is_none()
    );

    let mut by_level = local(json!({}));
    by_level.sampling_params = Some(json!({"top_p": 0.8}).as_object().cloned().unwrap());
    by_level.sampling_params_by_thinking_level = Some(
        [(
            ModelThinkingLevel::High,
            json!({"top_p": 0.95}).as_object().cloned().unwrap(),
        )]
        .into(),
    );
    assert_eq!(
        params(&by_level, &ctx(hi()), &with_effort(Some(ThinkingLevel::High)))["top_p"],
        0.95
    );
    assert_eq!(params(&by_level, &ctx(hi()), &with_effort(None))["top_p"], 0.8);
}

// ---------------------------------------------------------------------------
// Message conversion
// ---------------------------------------------------------------------------

#[test]
fn user_content_and_tool_result_images() {
    let model = openai();
    let image = || {
        UserBlock::Image(ImageContent {
            data: "ZmFrZQ==".into(),
            mime_type: "image/png".into(),
        })
    };
    let context = ctx(vec![Message::User(UserMessage {
        content: UserContent::Blocks(vec![UserBlock::text(""), image()]),
        timestamp: 1,
    })]);
    assert_eq!(
        convert(&model, &context),
        vec![
            json!({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZQ=="}}]})
        ]
    );
    // A user message whose blocks are all empty text is dropped.
    let empty = ctx(vec![Message::User(UserMessage {
        content: UserContent::Blocks(vec![UserBlock::text("")]),
        timestamp: 1,
    })]);
    assert!(convert(&model, &empty).is_empty());

    let context = ctx(vec![
        Message::user("Read the images"),
        assistant(
            vec![
                call("tool-1", "read", json!({"path": "1.png"})),
                call("tool-2", "read", json!({"path": "2.png"})),
            ],
            &model,
        ),
        result("tool-1", "read", vec![image()]),
        result("tool-2", "read", vec![UserBlock::text("  \n"), image()]),
        Message::user("next"),
    ]);
    let messages = convert(&model, &context);
    let roles: Vec<_> = messages.iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(roles, ["user", "assistant", "tool", "tool", "user", "user"]);
    assert_eq!(messages[2]["content"], "(see attached image)");
    assert_eq!(messages[4]["content"][0]["text"], "Attached image(s) from tool result:");
    assert_eq!(messages[4]["content"].as_array().unwrap().len(), 3);
    assert_eq!(
        messages[1]["tool_calls"][0],
        json!({"id": "tool-1", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"1.png\"}"}})
    );
    assert!(messages[1]["content"].is_null());

    // Non-vision models get placeholders and no image message.
    let mut text_only = model.clone();
    text_only.input = vec![crate::model::InputModality::Text];
    let messages = convert(&text_only, &context);
    assert_eq!(messages.len(), 5);

    for text in ["", "\r\n", " \n\t"] {
        let context = ctx(vec![
            Message::user("run"),
            assistant(vec![call("t", "bash", json!({}))], &model),
            result("t", "bash", vec![UserBlock::text(text)]),
        ]);
        assert_eq!(convert(&model, &context)[2]["content"], "(no tool output)");
    }
}

#[test]
fn tool_result_name_and_assistant_bridge() {
    let model = local(json!({"requiresToolResultName": true, "requiresAssistantAfterToolResult": true}));
    let context = ctx(vec![
        Message::user("run"),
        assistant(vec![call("t", "bash", json!({"c": 1}))], &model),
        result("t", "bash", vec![UserBlock::text("ok")]),
        Message::user("thanks"),
    ]);
    let messages = convert(&model, &context);
    assert_eq!(messages[1]["content"], "");
    assert_eq!(
        messages[2],
        json!({"role": "tool", "content": "ok", "tool_call_id": "t", "name": "bash"})
    );
    assert_eq!(
        messages[3],
        json!({"role": "assistant", "content": "I have processed the tool results."})
    );
    assert_eq!(messages[4]["role"], "user");
}

#[test]
fn replays_thinking_per_compat() {
    let model = local(json!({}));
    let thinking = |text: &str, signature: Option<&str>| {
        AssistantBlock::Thinking(ThinkingContent {
            thinking: text.into(),
            thinking_signature: signature.map(str::to_owned),
            redacted: None,
        })
    };
    let text = |text: &str| AssistantBlock::Text(TextContent::new(text));
    let context = ctx(vec![
        Message::user("q"),
        assistant(
            vec![
                thinking("step 1", Some("reasoning_content")),
                thinking("step 2", None),
                text("answer"),
            ],
            &model,
        ),
        Message::user("again"),
    ]);
    let messages = convert(&model, &context);
    assert_eq!(
        messages[1],
        json!({"role": "assistant", "content": "answer", "reasoning_content": "step 1\nstep 2"})
    );

    let as_text = local(json!({"requiresThinkingAsText": true}));
    let context_as_text = ctx(vec![
        Message::user("q"),
        assistant(vec![thinking("step 1", Some("reasoning")), text("answer")], &as_text),
    ]);
    assert_eq!(
        convert(&as_text, &context_as_text)[1]["content"],
        json!([{"type": "text", "text": "step 1"}, {"type": "text", "text": "answer"}])
    );

    // Unknown signature fields are not replayed; DeepSeek-style servers get an empty field.
    let required = local(json!({"requiresReasoningContentOnAssistantMessages": true}));
    let context = ctx(vec![
        Message::user("q"),
        assistant(
            vec![thinking("hm", Some("opaque")), call("c", "read", json!({}))],
            &required,
        ),
        result("c", "read", vec![UserBlock::text("x")]),
    ]);
    let message = &convert(&required, &context)[1];
    assert_eq!(message["reasoning_content"], "");
    assert!(message.get("opaque").is_none());

    // OpenCode Go replays `reasoning` as `reasoning_content`.
    let go = model_json("opencode-go", "https://opencode.ai/zen/go/v1", "kimi", true, None);
    let context = ctx(vec![
        Message::user("q"),
        assistant(vec![thinking("t", Some("reasoning")), text("a")], &go),
    ]);
    assert_eq!(convert(&go, &context)[1]["reasoning_content"], "t");

    // Empty assistant messages are skipped.
    let context = ctx(vec![
        Message::user("q"),
        assistant(vec![text("  ")], &model),
        Message::user("r"),
    ]);
    assert_eq!(convert(&model, &context).len(), 2);
}

#[test]
fn replays_reasoning_details() {
    let model = model_json(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "google/gemini",
        true,
        None,
    );
    let detail = json!({"type": "reasoning.encrypted", "id": "call_1", "data": "enc"});
    let signed = AssistantBlock::Thinking(ThinkingContent {
        thinking: "visible".into(),
        thinking_signature: Some(json!([detail]).to_string()),
        redacted: None,
    });
    let context = ctx(vec![
        Message::user("q"),
        assistant(vec![signed, call("call_1", "read", json!({"path": "a"}))], &model),
        result("call_1", "read", vec![UserBlock::text("x")]),
    ]);
    let message = &convert(&model, &context)[1];
    assert_eq!(message["reasoning_details"], json!([detail]));
    assert!(message.get("reasoning").is_none());

    // Older transcripts stored the encrypted detail on the tool call.
    let mut legacy = ToolCall {
        id: "call_1".into(),
        name: "read".into(),
        ..ToolCall::default()
    };
    legacy.thought_signature = Some(detail.to_string());
    let context = ctx(vec![
        Message::user("q"),
        assistant(vec![AssistantBlock::ToolCall(legacy)], &model),
        result("call_1", "read", vec![UserBlock::text("x")]),
    ]);
    assert_eq!(convert(&model, &context)[1]["reasoning_details"], json!([detail]));
}

#[test]
fn normalizes_foreign_tool_call_ids() {
    let model = openai();
    let mut foreign = AssistantMessage::empty("openai-responses", "openai-codex", "gpt-5-codex");
    foreign.stop_reason = StopReason::ToolUse;
    let long_id = format!("call_abc|fc_{}", "x".repeat(60));
    foreign.content = vec![
        call("call_1|fc+/=1", "read", json!({})),
        call(&long_id, "read", json!({})),
        AssistantBlock::Thinking(ThinkingContent::new("foreign thinking")),
    ];
    let context = ctx(vec![
        Message::user("q"),
        Message::Assistant(foreign),
        result("call_1|fc+/=1", "read", vec![UserBlock::text("1")]),
        result(&long_id, "read", vec![UserBlock::text("2")]),
    ]);
    let messages = convert(&model, &context);
    assert_eq!(messages[1]["tool_calls"][0]["id"], "call_1_fc___1");
    assert_eq!(messages[1]["tool_calls"][1]["id"], "call_abc_sbtd1t3g");
    assert!(messages[1].get("reasoning_content").is_none());
    assert_eq!(messages[2]["tool_call_id"], "call_1_fc___1");
    assert_eq!(messages[3]["tool_call_id"], "call_abc_sbtd1t3g");

    assert_eq!(normalize_tool_call_id(&"a".repeat(50), "openai"), "a".repeat(40));
    assert_eq!(normalize_tool_call_id(&"a".repeat(50), "groq"), "a".repeat(50));
    assert_eq!(normalize_tool_call_id("call|", "x"), "call");
    assert_eq!(normalize_tool_call_id("c\u{1F600}|i", "x"), "c___i");
}

#[test]
fn orphaned_tool_calls_get_results() {
    let model = openai();
    let context = ctx(vec![
        Message::user("q"),
        assistant(vec![call("c1", "read", json!({}))], &model),
        Message::user("never mind"),
    ]);
    let messages = convert(&model, &context);
    assert_eq!(messages[2]["role"], "tool");
    assert_eq!(messages[2]["content"], "No result provided");
    assert_eq!(messages[3]["content"], "never mind");
}

#[test]
fn mid_conversation_system_messages_and_tool_additions() {
    let model = local(json!({"supportsMidConvoSystemMessages": true, "supportsMidConvoToolAdditions": true}));
    let later = SystemMessage {
        content: "Now be brief.".into(),
        tools_added: Some(vec![Tool::new("grep", "Search", json!({"type": "object"}))]),
        ..SystemMessage::default()
    };
    let context = ctx_with(
        "sys",
        vec![read_tool()],
        vec![Message::user("q"), Message::System(later), Message::user("r")],
    );
    let body = params(&model, &context, &StreamOptions::default());
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(body["tools"].as_array().unwrap().len(), 1);
    assert_eq!(messages[2]["role"], "system");
    assert_eq!(messages[2]["tools"][0]["function"]["name"], "grep");
    assert_eq!(messages[3], json!({"role": "developer", "content": "Now be brief."}));

    // Without support, system messages collapse into the leading prompt and tools merge.
    let plain = local(json!({}));
    let body = params(&plain, &context, &StreamOptions::default());
    assert_eq!(body["messages"][0]["content"], "sys\n\nNow be brief.");
    assert_eq!(body["tools"].as_array().unwrap().len(), 2);

    // A removal cannot be anchored: the request carries the current tool list.
    let removal = SystemMessage {
        tools_removed: Some(vec![ToolReference { name: "read".into() }]),
        ..SystemMessage::default()
    };
    let context = ctx_with(
        "sys",
        vec![read_tool()],
        vec![Message::user("q"), Message::System(removal)],
    );
    let body = params(&model, &context, &StreamOptions::default());
    assert!(body.get("tools").is_none());
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[test]
fn usage_and_stop_reasons() {
    let model = openai();
    let usage = parse_chunk_usage(
        &json!({"prompt_tokens": 100, "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 50, "cache_write_tokens": 30},
            "completion_tokens_details": {"reasoning_tokens": 2}}),
        &model,
    );
    assert_eq!(
        (
            usage.input,
            usage.output,
            usage.cache_read,
            usage.cache_write,
            usage.total_tokens
        ),
        (20, 5, 50, 30, 105)
    );
    assert_eq!(usage.reasoning, Some(2));
    for raw in [
        json!({"prompt_tokens": 100, "completion_tokens": 5, "prompt_cache_hit_tokens": 40}),
        json!({"prompt_tokens": 100, "completion_tokens": 5, "cached_tokens": 40}),
        json!({"prompt_tokens": 100, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": null}, "cached_tokens": 40}),
    ] {
        let usage = parse_chunk_usage(&raw, &model);
        assert_eq!((usage.input, usage.cache_read, usage.total_tokens), (60, 40, 105));
        assert_eq!(usage.reasoning, Some(0));
    }
    let usage = parse_chunk_usage(&json!({"prompt_tokens": null, "completion_tokens": null}), &model);
    assert_eq!(usage.total_tokens, 0);

    assert_eq!(map_stop_reason("stop"), (StopReason::Stop, None));
    assert_eq!(map_stop_reason("end"), (StopReason::Stop, None));
    assert_eq!(map_stop_reason("length"), (StopReason::Length, None));
    assert_eq!(map_stop_reason("tool_calls"), (StopReason::ToolUse, None));
    assert_eq!(map_stop_reason("function_call"), (StopReason::ToolUse, None));
    assert_eq!(
        map_stop_reason("content_filter"),
        (StopReason::Error, Some("Provider finish_reason: content_filter".into()))
    );
}

#[test]
fn merges_reasoning_detail_deltas() {
    let mut details = Vec::new();
    let push = |details: &mut Vec<JsonObject>, value: Value| {
        append_reasoning_detail(details, value.as_object().unwrap());
    };
    push(
        &mut details,
        json!({"type": "reasoning.text", "text": "The", "index": 0}),
    );
    push(
        &mut details,
        json!({"type": "reasoning.text", "text": " user", "signature": "sig", "format": "f", "index": 0}),
    );
    push(
        &mut details,
        json!({"type": "reasoning.summary", "summary": "A", "index": 0}),
    );
    push(
        &mut details,
        json!({"type": "reasoning.summary", "summary": "B", "format": "f"}),
    );
    push(
        &mut details,
        json!({"type": "reasoning.encrypted", "id": "x", "data": "d"}),
    );
    push(&mut details, json!({"type": "reasoning.summary", "summary": "C"}));
    let values: Vec<Value> = details.into_iter().map(Value::Object).collect();
    assert_eq!(
        Value::Array(values),
        json!([
            {"type": "reasoning.text", "text": "The user", "index": 0, "signature": "sig", "format": "f"},
            {"type": "reasoning.summary", "summary": "AB", "index": 0, "format": "f"},
            {"type": "reasoning.encrypted", "id": "x", "data": "d"},
            {"type": "reasoning.summary", "summary": "C"}
        ])
    );

    // `??=` and `||=` assign undefined when the later delta lacks the field.
    let mut details = Vec::new();
    push(
        &mut details,
        json!({"type": "reasoning.text", "text": "a", "id": null, "format": ""}),
    );
    push(&mut details, json!({"type": "reasoning.text", "text": "b"}));
    assert_eq!(
        Value::Object(details[0].clone()).to_string(),
        r#"{"type":"reasoning.text","text":"ab"}"#
    );

    assert!(!is_reasoning_detail(&json!({"type": "reasoning.text", "text": 1})));
    assert!(!is_reasoning_detail(&json!({"type": "other"})));
    assert!(!is_reasoning_detail(
        &json!({"type": "reasoning.summary", "summary": "s", "format": null})
    ));
}

#[test]
fn in_stream_error_chunks() {
    let error = |chunk: Value| chunk_error(&chunk).map(|error| error.to_string());
    assert_eq!(
        error(json!({"error": {"message": "Overloaded", "code": 529, "metadata": {"raw": "upstream overloaded"}}})),
        Some("Overloaded\nupstream overloaded".into())
    );
    assert_eq!(
        error(json!({"error": {"message": "Provider returned error: boom", "metadata": {"raw": "boom"}}})),
        Some("Provider returned error: boom".into())
    );
    assert_eq!(error(json!({"error": "blocked"})), Some("\"blocked\"".into()));
    assert_eq!(
        error(json!({"detail": "Not authenticated"})),
        Some("Not authenticated".into())
    );
    assert_eq!(
        error(json!({"object": "error", "message": "bad", "code": 400})),
        Some("bad".into())
    );
    assert_eq!(error(json!({"error": null, "id": "x", "choices": []})), None);
    assert_eq!(error(json!({"id": "x", "model": "m"})), None);
    assert_eq!(error(json!({"choices": [], "usage": null})), None);
}
