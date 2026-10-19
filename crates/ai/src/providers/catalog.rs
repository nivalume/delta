//! The generated built-in model catalog.
//!
//! The data files (`data/providers/<provider>.json`) are the catalog shipped by
//! `@earendil-works/pi-ai` (see `scripts/sync-models.sh`). Each file groups entries by API id
//! and keys them as `"<type>:<model id>"`. A provider's file is parsed on first access.

use serde_json::Value;
use std::collections::HashMap;
use std::sync::{LazyLock, OnceLock};

use crate::model::{AnyModel, Model, ModelType};

const CATALOG: &[(&str, &str)] = &[
    (
        "amazon-bedrock",
        include_str!("../../data/providers/amazon-bedrock.json"),
    ),
    ("ant-ling", include_str!("../../data/providers/ant-ling.json")),
    ("anthropic", include_str!("../../data/providers/anthropic.json")),
    ("azure", include_str!("../../data/providers/azure.json")),
    ("baseten", include_str!("../../data/providers/baseten.json")),
    ("cerebras", include_str!("../../data/providers/cerebras.json")),
    (
        "cloudflare-ai-gateway",
        include_str!("../../data/providers/cloudflare-ai-gateway.json"),
    ),
    (
        "cloudflare-workers-ai",
        include_str!("../../data/providers/cloudflare-workers-ai.json"),
    ),
    ("deepseek", include_str!("../../data/providers/deepseek.json")),
    ("fireworks", include_str!("../../data/providers/fireworks.json")),
    (
        "github-copilot",
        include_str!("../../data/providers/github-copilot.json"),
    ),
    ("google", include_str!("../../data/providers/google.json")),
    ("google-vertex", include_str!("../../data/providers/google-vertex.json")),
    ("groq", include_str!("../../data/providers/groq.json")),
    ("huggingface", include_str!("../../data/providers/huggingface.json")),
    ("kimi-coding", include_str!("../../data/providers/kimi-coding.json")),
    ("meta", include_str!("../../data/providers/meta.json")),
    ("minimax", include_str!("../../data/providers/minimax.json")),
    ("minimax-cn", include_str!("../../data/providers/minimax-cn.json")),
    ("mistral", include_str!("../../data/providers/mistral.json")),
    ("moonshotai", include_str!("../../data/providers/moonshotai.json")),
    ("moonshotai-cn", include_str!("../../data/providers/moonshotai-cn.json")),
    ("nvidia", include_str!("../../data/providers/nvidia.json")),
    ("openai", include_str!("../../data/providers/openai.json")),
    ("openai-codex", include_str!("../../data/providers/openai-codex.json")),
    ("opencode", include_str!("../../data/providers/opencode.json")),
    ("opencode-go", include_str!("../../data/providers/opencode-go.json")),
    ("openrouter", include_str!("../../data/providers/openrouter.json")),
    (
        "qwen-token-plan",
        include_str!("../../data/providers/qwen-token-plan.json"),
    ),
    (
        "qwen-token-plan-cn",
        include_str!("../../data/providers/qwen-token-plan-cn.json"),
    ),
    (
        "qwen-token-plan-individual",
        include_str!("../../data/providers/qwen-token-plan-individual.json"),
    ),
    ("radius", include_str!("../../data/providers/radius.json")),
    ("together", include_str!("../../data/providers/together.json")),
    ("typesafe", include_str!("../../data/providers/typesafe.json")),
    (
        "vercel-ai-gateway",
        include_str!("../../data/providers/vercel-ai-gateway.json"),
    ),
    ("xai", include_str!("../../data/providers/xai.json")),
    ("xiaomi", include_str!("../../data/providers/xiaomi.json")),
    (
        "xiaomi-token-plan-ams",
        include_str!("../../data/providers/xiaomi-token-plan-ams.json"),
    ),
    (
        "xiaomi-token-plan-cn",
        include_str!("../../data/providers/xiaomi-token-plan-cn.json"),
    ),
    (
        "xiaomi-token-plan-sgp",
        include_str!("../../data/providers/xiaomi-token-plan-sgp.json"),
    ),
    ("zai", include_str!("../../data/providers/zai.json")),
    ("zai-coding-cn", include_str!("../../data/providers/zai-coding-cn.json")),
];

static PARSED: LazyLock<HashMap<&'static str, OnceLock<Vec<AnyModel>>>> =
    LazyLock::new(|| CATALOG.iter().map(|(id, _)| (*id, OnceLock::new())).collect());

fn parse(provider: &str, json: &str) -> Vec<AnyModel> {
    let Ok(Value::Object(apis)) = serde_json::from_str::<Value>(json) else {
        tracing::warn!(provider, "invalid built-in catalog data");
        return Vec::new();
    };
    let mut models = Vec::new();
    for (_, entries) in apis {
        let Value::Object(entries) = entries else { continue };
        for (key, value) in entries {
            match serde_json::from_value::<AnyModel>(value) {
                Ok(model) => models.push(model),
                Err(error) => tracing::warn!(provider, key, %error, "skipping invalid built-in model"),
            }
        }
    }
    models
}

/// Provider ids present in the built-in catalog.
pub fn builtin_provider_ids() -> Vec<&'static str> {
    CATALOG.iter().map(|(id, _)| *id).collect()
}

/// Every built-in model of a provider, of every type.
pub fn all_builtin_models(provider: &str) -> &'static [AnyModel] {
    let Some((id, json)) = CATALOG.iter().find(|(id, _)| *id == provider) else {
        return &[];
    };
    PARSED[id].get_or_init(|| parse(id, json))
}

/// Built-in chat models of a provider.
pub fn builtin_models(provider: &str) -> Vec<Model> {
    all_builtin_models(provider)
        .iter()
        .filter_map(AnyModel::as_chat)
        .cloned()
        .collect()
}

/// A built-in chat model.
pub fn builtin_model(provider: &str, id: &str) -> Option<Model> {
    all_builtin_models(provider)
        .iter()
        .filter_map(AnyModel::as_chat)
        .find(|model| model.id == id)
        .cloned()
}

/// A built-in model of one type.
pub fn builtin_model_of_type(model_type: ModelType, provider: &str, id: &str) -> Option<AnyModel> {
    all_builtin_models(provider)
        .iter()
        .find(|model| model.model_type() == model_type && model.id() == id)
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_catalog_entry_parses() {
        for (provider, json) in CATALOG {
            let value: Value = serde_json::from_str(json).unwrap();
            let expected: usize = value
                .as_object()
                .unwrap()
                .values()
                .map(|v| v.as_object().unwrap().len())
                .sum();
            let models = all_builtin_models(provider);
            assert_eq!(models.len(), expected, "{provider}");
            assert!(models.iter().all(|model| model.provider() == *provider), "{provider}");
        }
    }

    #[test]
    fn looks_up_models() {
        let model = builtin_model("anthropic", "claude-sonnet-4-5").unwrap();
        assert_eq!(model.api, "anthropic-messages");
        assert!(model.reasoning);
        assert!(builtin_model_of_type(ModelType::Image, "openrouter", "google/gemini-2.5-flash-image").is_some());
    }
}
