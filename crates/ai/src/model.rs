//! Model catalog entries: chat, image-generation and classifier models.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

use crate::compat::ModelCompat;
use crate::types::{JsonObject, Usage, UsageCost};

/// Pi reasoning levels accepted by the simple stream options.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ThinkingLevel {
    pub fn as_str(self) -> &'static str {
        ModelThinkingLevel::from(self).as_str()
    }
}

/// Thinking levels a model can run at, including `off`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelThinkingLevel {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ModelThinkingLevel {
    /// Every level, from `off` to `max`.
    pub const ALL: [Self; 7] = [
        Self::Off,
        Self::Minimal,
        Self::Low,
        Self::Medium,
        Self::High,
        Self::Xhigh,
        Self::Max,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }

    /// The reasoning level, or `None` for `off`.
    pub fn reasoning(self) -> Option<ThinkingLevel> {
        match self {
            Self::Off => None,
            Self::Minimal => Some(ThinkingLevel::Minimal),
            Self::Low => Some(ThinkingLevel::Low),
            Self::Medium => Some(ThinkingLevel::Medium),
            Self::High => Some(ThinkingLevel::High),
            Self::Xhigh => Some(ThinkingLevel::Xhigh),
            Self::Max => Some(ThinkingLevel::Max),
        }
    }
}

impl From<ThinkingLevel> for ModelThinkingLevel {
    fn from(level: ThinkingLevel) -> Self {
        match level {
            ThinkingLevel::Minimal => Self::Minimal,
            ThinkingLevel::Low => Self::Low,
            ThinkingLevel::Medium => Self::Medium,
            ThinkingLevel::High => Self::High,
            ThinkingLevel::Xhigh => Self::Xhigh,
            ThinkingLevel::Max => Self::Max,
        }
    }
}

/// Provider value sent for each thinking level. `None` marks a level unsupported; a missing
/// key uses the provider default through `high`, while `xhigh` and `max` are opt-in.
pub type ThinkingLevelMap = BTreeMap<ModelThinkingLevel, Option<String>>;

/// Input modality of a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputModality {
    Text,
    Image,
}

/// Output modality of an image model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputModality {
    Text,
    Image,
}

/// Prompt cache retention preference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheRetention {
    None,
    #[default]
    Short,
    Long,
}

/// Best-effort prompt cache lifetime in seconds per retention tier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelPromptCache {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub short: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub long: Option<f64>,
}

/// Request-wide pricing tier. The highest matching input threshold applies to the full request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCostTier {
    pub input_tokens_above: f64,
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

/// Prices in USD per million tokens.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiers: Option<Vec<ModelCostTier>>,
}

/// Cache-safe image resize profile.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelImageResizeOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_height: Option<u32>,
    /// Maximum base64-encoded payload size in bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jpeg_quality: Option<u8>,
}

/// Image input limits of a model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelImageInputLimits {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resize: Option<ModelImageResizeOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_per_message: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_per_request: Option<u32>,
}

/// Provider input limits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInputLimits {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_request_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<ModelImageInputLimits>,
}

/// Chat model: usable with `stream()` and friends.
///
/// Models are plain serializable data; persisting "which model was this conversation using"
/// is a `serde_json::to_string` away.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", try_from = "ModelRepr")]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: String,
    pub provider: String,
    pub base_url: String,
    pub input: Vec<InputModality>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_limits: Option<ModelInputLimits>,
    pub cost: ModelCost,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<IndexMap<String, String>>,
    pub reasoning: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache: Option<ModelPromptCache>,
    pub context_window: u64,
    pub max_tokens: u64,
    /// Default sampling parameters; per-request keys override them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<JsonObject>,
    /// Sampling parameter overrides selected by the effective thinking level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params_by_thinking_level: Option<BTreeMap<ModelThinkingLevel, JsonObject>>,
    /// Compatibility overrides. If unset, API implementations auto-detect from `base_url`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<ModelCompat>,
}

/// Deserialization shape of [`Model`]: `compat` is parsed with the schema of `api`.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelRepr {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    id: String,
    name: String,
    api: String,
    provider: String,
    base_url: String,
    input: Vec<InputModality>,
    #[serde(default)]
    input_limits: Option<ModelInputLimits>,
    cost: ModelCost,
    #[serde(default)]
    headers: Option<IndexMap<String, String>>,
    #[serde(default)]
    reasoning: bool,
    #[serde(default)]
    thinking_level_map: Option<ThinkingLevelMap>,
    #[serde(default)]
    prompt_cache: Option<ModelPromptCache>,
    #[serde(default)]
    context_window: u64,
    #[serde(default)]
    max_tokens: u64,
    #[serde(default)]
    sampling_params: Option<JsonObject>,
    #[serde(default)]
    sampling_params_by_thinking_level: Option<BTreeMap<ModelThinkingLevel, JsonObject>>,
    #[serde(default)]
    compat: Option<Value>,
}

impl TryFrom<ModelRepr> for Model {
    type Error = String;

    fn try_from(repr: ModelRepr) -> Result<Self, Self::Error> {
        if let Some(kind) = repr.kind.as_deref()
            && kind != "chat"
        {
            return Err(format!(
                "model {}/{} has type {kind:?}, expected a chat model",
                repr.provider, repr.id
            ));
        }
        let compat = repr
            .compat
            .filter(|value| !value.is_null())
            .map(|value| ModelCompat::from_value(&repr.api, value))
            .transpose()
            .map_err(|error| format!("invalid compat for {}/{}: {error}", repr.provider, repr.id))?;
        Ok(Self {
            id: repr.id,
            name: repr.name,
            api: repr.api,
            provider: repr.provider,
            base_url: repr.base_url,
            input: repr.input,
            input_limits: repr.input_limits,
            cost: repr.cost,
            headers: repr.headers,
            reasoning: repr.reasoning,
            thinking_level_map: repr.thinking_level_map,
            prompt_cache: repr.prompt_cache,
            context_window: repr.context_window,
            max_tokens: repr.max_tokens,
            sampling_params: repr.sampling_params,
            sampling_params_by_thinking_level: repr.sampling_params_by_thinking_level,
            compat,
        })
    }
}

impl Model {
    pub fn supports_images(&self) -> bool {
        self.input.contains(&InputModality::Image)
    }

    /// Thinking levels this model accepts, from `off` up.
    pub fn supported_thinking_levels(&self) -> Vec<ModelThinkingLevel> {
        if !self.reasoning {
            return vec![ModelThinkingLevel::Off];
        }
        ModelThinkingLevel::ALL
            .into_iter()
            .filter(|level| {
                let mapped = self.thinking_level_map.as_ref().and_then(|map| map.get(level));
                match mapped {
                    Some(None) => false,
                    Some(Some(_)) => true,
                    None => !matches!(level, ModelThinkingLevel::Xhigh | ModelThinkingLevel::Max),
                }
            })
            .collect()
    }

    /// The nearest supported level: `level` itself, else the next higher, else the next lower.
    pub fn clamp_thinking_level(&self, level: ModelThinkingLevel) -> ModelThinkingLevel {
        let available = self.supported_thinking_levels();
        if available.contains(&level) {
            return level;
        }
        let all = ModelThinkingLevel::ALL;
        let index = all.iter().position(|candidate| *candidate == level).unwrap_or(0);
        all[index..]
            .iter()
            .chain(all[..index].iter().rev())
            .find(|candidate| available.contains(candidate))
            .copied()
            .unwrap_or(ModelThinkingLevel::Off)
    }

    /// Provider value mapped for `level`, if the model maps it.
    pub fn mapped_thinking_level(&self, level: ModelThinkingLevel) -> Option<Option<&str>> {
        self.thinking_level_map
            .as_ref()
            .and_then(|map| map.get(&level))
            .map(Option::as_deref)
    }

    /// Whether the exact model accepts system messages mid-conversation.
    pub fn supports_mid_convo_system_messages(&self) -> bool {
        self.compat
            .as_ref()
            .is_some_and(ModelCompat::supports_mid_convo_system_messages)
    }
}

/// Image-generation model: usable with `generate_images()` only.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageModel {
    pub id: String,
    pub name: String,
    pub api: String,
    pub provider: String,
    pub base_url: String,
    pub input: Vec<InputModality>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_limits: Option<ModelInputLimits>,
    pub cost: ModelCost,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<IndexMap<String, String>>,
    /// Always includes `image`; `text` means the model can also return text blocks.
    pub output: Vec<OutputModality>,
}

/// Structured classifier model: usable with `classify()` only.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassifierModel {
    pub id: String,
    pub name: String,
    pub api: String,
    pub provider: String,
    pub base_url: String,
    pub input: Vec<InputModality>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_limits: Option<ModelInputLimits>,
    pub cost: ModelCost,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<IndexMap<String, String>>,
    pub context_window: u64,
}

/// What a catalog entry is for. Decides which `Models` operation accepts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Chat,
    Image,
    Classifier,
}

/// Anything a provider can list. Serialized with a `type` tag; a missing tag means chat.
#[derive(Clone, Debug, PartialEq)]
pub enum AnyModel {
    Chat(Model),
    Image(ImageModel),
    Classifier(ClassifierModel),
}

impl AnyModel {
    pub fn model_type(&self) -> ModelType {
        match self {
            Self::Chat(_) => ModelType::Chat,
            Self::Image(_) => ModelType::Image,
            Self::Classifier(_) => ModelType::Classifier,
        }
    }

    pub fn id(&self) -> &str {
        match self {
            Self::Chat(m) => &m.id,
            Self::Image(m) => &m.id,
            Self::Classifier(m) => &m.id,
        }
    }

    pub fn provider(&self) -> &str {
        match self {
            Self::Chat(m) => &m.provider,
            Self::Image(m) => &m.provider,
            Self::Classifier(m) => &m.provider,
        }
    }

    pub fn api(&self) -> &str {
        match self {
            Self::Chat(m) => &m.api,
            Self::Image(m) => &m.api,
            Self::Classifier(m) => &m.api,
        }
    }

    pub fn cost(&self) -> &ModelCost {
        match self {
            Self::Chat(m) => &m.cost,
            Self::Image(m) => &m.cost,
            Self::Classifier(m) => &m.cost,
        }
    }

    pub fn headers(&self) -> Option<&IndexMap<String, String>> {
        match self {
            Self::Chat(m) => m.headers.as_ref(),
            Self::Image(m) => m.headers.as_ref(),
            Self::Classifier(m) => m.headers.as_ref(),
        }
    }

    pub fn as_chat(&self) -> Option<&Model> {
        match self {
            Self::Chat(m) => Some(m),
            _ => None,
        }
    }

    /// Same type, provider and id.
    pub fn same_model(&self, other: &Self) -> bool {
        self.model_type() == other.model_type() && self.id() == other.id() && self.provider() == other.provider()
    }
}

impl From<Model> for AnyModel {
    fn from(model: Model) -> Self {
        Self::Chat(model)
    }
}

impl From<ImageModel> for AnyModel {
    fn from(model: ImageModel) -> Self {
        Self::Image(model)
    }
}

impl From<ClassifierModel> for AnyModel {
    fn from(model: ClassifierModel) -> Self {
        Self::Classifier(model)
    }
}

impl Serialize for AnyModel {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let (kind, value) = match self {
            Self::Chat(m) => ("chat", serde_json::to_value(m)),
            Self::Image(m) => ("image", serde_json::to_value(m)),
            Self::Classifier(m) => ("classifier", serde_json::to_value(m)),
        };
        let mut value = value.map_err(serde::ser::Error::custom)?;
        if let Value::Object(object) = &mut value {
            object.insert("type".into(), Value::String(kind.into()));
        }
        value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AnyModel {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let kind = value.get("type").and_then(Value::as_str).unwrap_or("chat").to_owned();
        let result = match kind.as_str() {
            "chat" => serde_json::from_value(value).map(Self::Chat),
            "image" => serde_json::from_value(value).map(Self::Image),
            "classifier" => serde_json::from_value(value).map(Self::Classifier),
            other => return Err(serde::de::Error::custom(format!("unknown model type {other:?}"))),
        };
        result.map_err(serde::de::Error::custom)
    }
}

/// Fill `usage.cost` from the model's prices and return it.
pub fn calculate_cost(cost: &ModelCost, usage: &mut Usage) -> UsageCost {
    let input_tokens = (usage.input + usage.cache_read + usage.cache_write) as f64;
    let (mut input, mut output, mut cache_read, mut cache_write) =
        (cost.input, cost.output, cost.cache_read, cost.cache_write);
    let mut matched = -1.0;
    for tier in cost.tiers.iter().flatten() {
        if input_tokens > tier.input_tokens_above && tier.input_tokens_above > matched {
            (input, output, cache_read, cache_write) = (tier.input, tier.output, tier.cache_read, tier.cache_write);
            matched = tier.input_tokens_above;
        }
    }
    // Anthropic charges 2x base input for 1h cache writes.
    let long_write = usage.cache_write_1h.unwrap_or(0).min(usage.cache_write) as f64;
    let short_write = usage.cache_write as f64 - long_write;
    let c = &mut usage.cost;
    c.input = input / 1_000_000.0 * usage.input as f64;
    c.output = output / 1_000_000.0 * usage.output as f64;
    c.cache_read = cache_read / 1_000_000.0 * usage.cache_read as f64;
    c.cache_write = (cache_write * short_write + input * 2.0 * long_write) / 1_000_000.0;
    c.total = c.input + c.output + c.cache_read + c.cache_write;
    *c
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn model(reasoning: bool, map: Option<Value>) -> Model {
        let mut value = json!({"id": "m", "name": "M", "api": "openai-completions", "provider": "p",
            "baseUrl": "http://x", "reasoning": reasoning, "input": ["text"],
            "cost": {"input": 1, "output": 2, "cacheRead": 0.5, "cacheWrite": 1.5},
            "contextWindow": 1000, "maxTokens": 100});
        if let Some(map) = map {
            value["thinkingLevelMap"] = map;
        }
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn thinking_levels() {
        assert_eq!(
            model(false, None).supported_thinking_levels(),
            vec![ModelThinkingLevel::Off]
        );
        let m = model(true, Some(json!({"minimal": null, "xhigh": "xhigh"})));
        assert_eq!(
            m.supported_thinking_levels(),
            vec![
                ModelThinkingLevel::Off,
                ModelThinkingLevel::Low,
                ModelThinkingLevel::Medium,
                ModelThinkingLevel::High,
                ModelThinkingLevel::Xhigh
            ]
        );
        assert_eq!(
            m.clamp_thinking_level(ModelThinkingLevel::Minimal),
            ModelThinkingLevel::Low
        );
        assert_eq!(
            m.clamp_thinking_level(ModelThinkingLevel::Max),
            ModelThinkingLevel::Xhigh
        );
    }

    #[test]
    fn cost_with_tiers_and_long_cache_writes() {
        let mut m = model(false, None);
        m.cost.tiers = Some(vec![ModelCostTier {
            input_tokens_above: 1000.0,
            input: 10.0,
            output: 20.0,
            cache_read: 1.0,
            cache_write: 15.0,
        }]);
        let mut usage = Usage {
            input: 1_000_000,
            output: 1_000_000,
            cache_write: 2,
            cache_write_1h: Some(1),
            ..Usage::default()
        };
        let cost = calculate_cost(&m.cost, &mut usage);
        assert!((cost.input - 10.0).abs() < 1e-9);
        assert!((cost.output - 20.0).abs() < 1e-9);
        assert!((cost.cache_write - (15.0 + 20.0) / 1_000_000.0).abs() < 1e-12);
    }

    #[test]
    fn any_model_type_tag() {
        let any: AnyModel = serde_json::from_value(
            json!({"type": "image", "id": "i", "name": "I", "api": "openrouter-images",
            "provider": "openrouter", "baseUrl": "x", "input": ["text"], "output": ["image"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}),
        )
        .unwrap();
        assert_eq!(any.model_type(), ModelType::Image);
        assert_eq!(serde_json::to_value(&any).unwrap()["type"], "image");
        let chat = AnyModel::Chat(model(false, None));
        let back: AnyModel = serde_json::from_value(serde_json::to_value(&chat).unwrap()).unwrap();
        assert_eq!(back, chat);
    }
}
