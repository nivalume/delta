//! Unified LLM API: provider collections, auth resolution, streaming events, token and cost
//! tracking, tool calling and cross-provider context hand-off.
//!
//! Rust port of [`@earendil-works/pi-ai`](https://github.com/earendil-works/pi/tree/main/packages/ai).
//! Design notes live in `docs/design/` of this crate.

pub mod api;
pub mod auth;
pub mod compat;
pub mod error;
pub mod estimate;
pub mod events;
pub mod http;
pub mod json;
pub mod model;
pub mod models;
pub mod options;
pub mod provider;
pub mod providers;
pub mod simple_options;
pub mod transcript;
pub mod transform;
pub mod types;
pub mod validation;

pub use auth::{
    ApiKeyAuth, ApiKeyCredential, AuthContext, AuthResult, AuthType, Credential, CredentialStore,
    InMemoryCredentialStore, ModelAuth, OAuthAuth, OAuthCredential, ProviderAuth, env_api_key_auth,
};
pub use error::{ModelsError, ModelsErrorCode};
pub use events::{
    AssistantMessageEvent, AssistantMessageEventStream, DoneReason, ErrorReason, EventSender, MessageBuilder,
};
pub use model::{
    AnyModel, CacheRetention, ClassifierModel, ImageModel, InputModality, Model, ModelCost, ModelThinkingLevel,
    ModelType, ThinkingLevel, calculate_cost,
};
pub use models::{CreateModelsOptions, Models, ModelsRefreshOptions, ModelsRefreshResult, ModelsStore};
pub use options::{ApiOptions, SimpleStreamOptions, StreamOptions, ToolChoice};
pub use provider::{CreateProviderOptions, Provider, ProviderApi, ProviderStreams, create_provider, lazy_stream};
pub use transcript::{get_current_system_prompt, get_current_tools, normalize_context};
pub use types::{
    AssistantBlock, AssistantMessage, Context, ImageContent, JsonObject, Message, StopReason, SystemMessage,
    TextContent, ThinkingContent, Tool, ToolCall, ToolResultMessage, TranscriptContext, Usage, UserBlock, UserContent,
    UserMessage,
};

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
