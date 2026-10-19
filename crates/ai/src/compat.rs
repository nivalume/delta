//! Per-API compatibility settings carried on [`crate::Model::compat`].
//!
//! Every field is optional: unset fields fall back to API defaults or URL-based
//! auto-detection. Unknown fields are kept in `extra` so custom model files round-trip.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::model::ModelCost;
use crate::types::JsonObject;

/// Session-affinity header format used to route related requests consistently.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SessionAffinityFormat {
    /// `session_id`, `x-client-request-id` and `x-session-affinity`.
    Openai,
    /// `x-client-request-id` and `x-session-affinity`.
    OpenaiNosession,
    /// `x-session-id`.
    Openrouter,
}

/// Format of reasoning parameters on OpenAI-compatible completions endpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ThinkingFormat {
    Openai,
    Openrouter,
    Deepseek,
    Together,
    Baseten,
    Zai,
    Qwen,
    ChatTemplate,
    QwenChatTemplate,
    StringThinking,
    AntLing,
}

/// Top-level request field that caps reasoning tokens on OpenAI-compatible servers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingTokenBudgetField {
    /// vLLM.
    ThinkingTokenBudget,
    /// Qwen, DashScope, SGLang.
    ThinkingBudget,
    /// llama.cpp.
    ThinkingBudgetTokens,
}

/// Field name for the output token limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaxTokensField {
    MaxCompletionTokens,
    MaxTokens,
}

/// A pi-controlled value in `chat_template_kwargs`/`chat_template_args`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinkingVariable {
    #[serde(rename = "thinking.enabled")]
    Enabled,
    #[serde(rename = "thinking.effort")]
    Effort,
    #[serde(rename = "thinking.budget")]
    Budget,
}

/// Value of a chat template kwarg: a literal, or a `$var` reference to a thinking setting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatTemplateKwargValue {
    Var {
        #[serde(rename = "$var")]
        var: ThinkingVariable,
        #[serde(rename = "omitWhenOff", default, skip_serializing_if = "Option::is_none")]
        omit_when_off: Option<bool>,
    },
    Literal(Value),
}

/// Compatibility settings for OpenAI-compatible completions APIs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAICompletionsCompat {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_store: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_developer_role: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_reasoning_effort: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_usage_in_streaming: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_finish_reason: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens_field: Option<MaxTokensField>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_tool_result_name: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_assistant_after_tool_result: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_thinking_as_text: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_reasoning_content_on_assistant_messages: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_format: Option<ThinkingFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<IndexMap<String, ChatTemplateKwargValue>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_args: Option<IndexMap<String, ChatTemplateKwargValue>>,
    /// OpenRouter provider routing preferences, sent as the `provider` request field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_router_routing: Option<JsonObject>,
    /// Vercel AI Gateway routing preferences.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vercel_gateway_routing: Option<JsonObject>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zai_tool_stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_token_budget_field: Option<ThinkingTokenBudgetField>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_thinking_token_budget: Option<bool>,
    #[serde(
        default,
        rename = "supportsOpenAIGrammarTools",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_openai_grammar_tools: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_mid_convo_system_messages: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_mid_convo_tool_additions: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_strict_mode: Option<bool>,
    /// Only `"anthropic"` is defined.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control_format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub send_session_affinity_headers: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_affinity_format: Option<SessionAffinityFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vllm_priority: Option<f64>,
    #[serde(flatten)]
    pub extra: JsonObject,
}

/// Compatibility settings for OpenAI Responses APIs (also Azure and Codex Responses).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAIResponsesCompat {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_developer_role: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_mid_convo_system_messages: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_affinity_format: Option<SessionAffinityFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_strict_mode: Option<bool>,
    #[serde(
        default,
        rename = "supportsOpenAIGrammarTools",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_openai_grammar_tools: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_additional_tools: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_tool_search: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_explicit_prompt_cache_mode: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_mid_convo_effort: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_max_output_tokens: Option<bool>,
    #[serde(flatten)]
    pub extra: JsonObject,
}

/// An Anthropic server-side refusal fallback model with local pricing metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnthropicAllowedFallbackModel {
    pub provider: String,
    pub model: String,
    pub cost: ModelCost,
}

/// Compatibility settings for Anthropic Messages-compatible APIs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicMessagesCompat {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_eager_tool_input_streaming: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub send_session_affinity_headers: Option<bool>,
    /// Only `openrouter` is defined; unset sends `x-session-affinity`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_affinity_format: Option<SessionAffinityFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_cache_control_on_tools: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_temperature: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force_adaptive_thinking: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_empty_signature: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_strict_tools: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_mid_convo_effort: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_mid_convo_system_messages: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_mid_convo_tool_changes: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_fallback_models: Option<Vec<AnthropicAllowedFallbackModel>>,
    #[serde(flatten)]
    pub extra: JsonObject,
}

/// Compatibility settings for Amazon Bedrock models.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockCompat {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_strict_mode: Option<bool>,
    #[serde(flatten)]
    pub extra: JsonObject,
}

/// Compatibility settings for the Mistral chat API.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MistralConversationsCompat {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_mid_convo_system_messages: Option<bool>,
    #[serde(flatten)]
    pub extra: JsonObject,
}

/// Compatibility settings of a chat model, typed by its API.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ModelCompat {
    OpenAICompletions(OpenAICompletionsCompat),
    /// `openai-responses`, `azure-openai-responses` and `openai-codex-responses`.
    OpenAIResponses(OpenAIResponsesCompat),
    AnthropicMessages(AnthropicMessagesCompat),
    Bedrock(BedrockCompat),
    Mistral(MistralConversationsCompat),
    /// APIs without a typed compat schema.
    Other(JsonObject),
}

impl ModelCompat {
    /// Parse `value` with the compat schema of `api`.
    pub fn from_value(api: &str, value: Value) -> Result<Self, serde_json::Error> {
        Ok(match api {
            "openai-completions" => Self::OpenAICompletions(serde_json::from_value(value)?),
            "openai-responses" | "azure-openai-responses" | "openai-codex-responses" => {
                Self::OpenAIResponses(serde_json::from_value(value)?)
            }
            "anthropic-messages" => Self::AnthropicMessages(serde_json::from_value(value)?),
            "bedrock-converse-stream" => Self::Bedrock(serde_json::from_value(value)?),
            "mistral-conversations" => Self::Mistral(serde_json::from_value(value)?),
            _ => Self::Other(serde_json::from_value(value)?),
        })
    }

    /// Whether the exact model accepts system messages mid-conversation.
    pub fn supports_mid_convo_system_messages(&self) -> bool {
        match self {
            Self::OpenAICompletions(c) => c.supports_mid_convo_system_messages,
            Self::OpenAIResponses(c) => c.supports_mid_convo_system_messages,
            Self::AnthropicMessages(c) => c.supports_mid_convo_system_messages,
            Self::Mistral(c) => c.supports_mid_convo_system_messages,
            Self::Bedrock(_) | Self::Other(_) => None,
        }
        .unwrap_or(false)
    }
}
