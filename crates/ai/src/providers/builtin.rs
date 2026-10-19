//! Factories of the built-in providers.
//!
//! Upstream has one module per provider (`src/providers/<id>.ts`). Most of them only pair
//! an environment variable with an API implementation, so delta keeps those in one table and
//! writes out the providers with their own behavior (Azure, OpenCode, Bedrock, Vertex AI).
//! The Cloudflare providers live in [`crate::providers::cloudflare`], GitHub Copilot in
//! [`crate::providers::github_copilot`], Radius in [`crate::providers::radius`].
//!
//! All upstream built-in providers are built in. OAuth login is ported for every provider that
//! offers it upstream: Anthropic, OpenRouter, xAI, OpenAI (Sign in with ChatGPT), OpenAI Codex,
//! Kimi For Coding, Meta, GitHub Copilot and Radius ([`crate::oauth`]).

use futures::future::BoxFuture;
use std::collections::HashMap;
use std::sync::Arc;

use crate::api::anthropic_messages::AnthropicMessagesApi;
use crate::api::azure_openai_responses::{AzureOpenAIResponsesApi, resolve_azure_base_url, resolve_deployment_name};
use crate::api::google_generative_ai::GoogleGenerativeAiApi;
use crate::api::mistral_conversations::MistralConversationsApi;
use crate::api::openai_codex_responses::OpenAICodexResponsesApi;
use crate::api::openai_completions::OpenAICompletionsApi;
use crate::api::openai_decisions::{self, OpenAIDecisionsApi};
use crate::api::openai_responses::OpenAIResponsesApi;
use crate::api::openrouter_images::{self, OpenRouterImagesApi};
use crate::api::typesafe_system_one::{self, TypeSafeSystemOneApi};
use crate::auth::{Credential, OAuthAuth, ProviderAuth, env_api_key_auth};
use crate::error::{ModelsError, ModelsErrorCode};
use crate::events::AssistantMessageEventStream;
use crate::model::{AnyModel, Model, ModelType};
use crate::models::{CreateModelsOptions, Models};
use crate::oauth::{
    AnthropicOAuth, KimiCodingOAuth, MetaOAuth, OpenAIChatGPTOAuth, OpenAICodexOAuth, OpenRouterOAuth, XaiOAuth,
};
use crate::options::{AzureOpenAIResponsesOptions, OnPayload, SimpleStreamOptions, StreamOptions};
use crate::provider::{
    BuiltProvider, CreateProviderOptions, Provider, ProviderApi, ProviderStreams, create_provider, lazy_stream,
};
use crate::providers::catalog::all_builtin_models;
use crate::types::TranscriptContext;

/// Which API implementations a table-driven provider uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Apis {
    AnthropicMessages,
    OpenAICompletions,
    OpenAIResponses,
    GoogleGenerativeAi,
    Mistral,
    /// `anthropic-messages` and `openai-completions`, by `model.api`.
    AnthropicAndCompletions,
}

/// `(id, name, base url, env vars, APIs)` of the providers that only need an API key.
const SIMPLE_PROVIDERS: &[(&str, &str, &str, &[&str], Apis)] = &[
    (
        "ant-ling",
        "Ant Ling",
        "https://api.ant-ling.com/v1",
        &["ANT_LING_API_KEY"],
        Apis::OpenAICompletions,
    ),
    (
        "baseten",
        "Baseten",
        "https://inference.baseten.co/v1",
        &["BASETEN_API_KEY"],
        Apis::OpenAICompletions,
    ),
    (
        "cerebras",
        "Cerebras",
        "https://api.cerebras.ai/v1",
        &["CEREBRAS_API_KEY"],
        Apis::OpenAICompletions,
    ),
    (
        "deepseek",
        "DeepSeek",
        "https://api.deepseek.com",
        &["DEEPSEEK_API_KEY"],
        Apis::OpenAICompletions,
    ),
    (
        "fireworks",
        "Fireworks",
        "https://api.fireworks.ai/inference",
        &["FIREWORKS_API_KEY"],
        Apis::AnthropicAndCompletions,
    ),
    (
        "google",
        "Google",
        "https://generativelanguage.googleapis.com/v1beta",
        &["GEMINI_API_KEY"],
        Apis::GoogleGenerativeAi,
    ),
    (
        "groq",
        "Groq",
        "https://api.groq.com/openai/v1",
        &["GROQ_API_KEY"],
        Apis::OpenAICompletions,
    ),
    (
        "huggingface",
        "Hugging Face",
        "https://router.huggingface.co/v1",
        &["HF_TOKEN"],
        Apis::OpenAICompletions,
    ),
    (
        "kimi-coding",
        "Kimi For Coding",
        "https://api.kimi.com/coding/v1",
        &["KIMI_API_KEY"],
        Apis::OpenAICompletions,
    ),
    (
        "meta",
        "Meta",
        "https://api.meta.ai/v1",
        &["META_API_KEY"],
        Apis::OpenAIResponses,
    ),
    (
        "minimax",
        "MiniMax",
        "https://api.minimax.io/anthropic",
        &["MINIMAX_API_KEY"],
        Apis::AnthropicMessages,
    ),
    (
        "minimax-cn",
        "MiniMax CN",
        "https://api.minimaxi.com/anthropic",
        &["MINIMAX_CN_API_KEY"],
        Apis::AnthropicMessages,
    ),
    (
        "mistral",
        "Mistral",
        "https://api.mistral.ai",
        &["MISTRAL_API_KEY"],
        Apis::Mistral,
    ),
    (
        "moonshotai",
        "Moonshot AI",
        "https://api.moonshot.ai/v1",
        &["MOONSHOT_API_KEY"],
        Apis::OpenAICompletions,
    ),
    (
        "moonshotai-cn",
        "Moonshot AI CN",
        "https://api.moonshot.cn/v1",
        &["MOONSHOT_API_KEY"],
        Apis::OpenAICompletions,
    ),
    (
        "nvidia",
        "NVIDIA",
        "https://integrate.api.nvidia.com/v1",
        &["NVIDIA_API_KEY"],
        Apis::OpenAICompletions,
    ),
    (
        "openai",
        "OpenAI",
        "https://api.openai.com/v1",
        &["OPENAI_API_KEY"],
        Apis::OpenAIResponses,
    ),
    (
        "openrouter",
        "OpenRouter",
        "https://openrouter.ai/api/v1",
        &["OPENROUTER_API_KEY"],
        Apis::AnthropicAndCompletions,
    ),
    (
        "qwen-token-plan",
        "Qwen Token Plan",
        "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        &["QWEN_TOKEN_PLAN_API_KEY"],
        Apis::OpenAICompletions,
    ),
    (
        "qwen-token-plan-cn",
        "Qwen Token Plan CN",
        "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        &["QWEN_TOKEN_PLAN_CN_API_KEY"],
        Apis::OpenAICompletions,
    ),
    (
        "qwen-token-plan-individual",
        "Qwen Token Plan Individual",
        "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        &["QWEN_TOKEN_PLAN_API_KEY"],
        Apis::OpenAICompletions,
    ),
    (
        "together",
        "Together",
        "https://api.together.ai/v1",
        &["TOGETHER_API_KEY"],
        Apis::OpenAICompletions,
    ),
    (
        "vercel-ai-gateway",
        "Vercel AI Gateway",
        "https://ai-gateway.vercel.sh",
        &["AI_GATEWAY_API_KEY"],
        Apis::AnthropicMessages,
    ),
    (
        "xai",
        "xAI",
        "https://api.x.ai/v1",
        &["XAI_API_KEY"],
        Apis::OpenAIResponses,
    ),
    (
        "xiaomi",
        "Xiaomi",
        "https://api.xiaomimimo.com/v1",
        &["XIAOMI_API_KEY"],
        Apis::OpenAICompletions,
    ),
    (
        "xiaomi-token-plan-ams",
        "Xiaomi Token Plan AMS",
        "https://token-plan-ams.xiaomimimo.com/v1",
        &["XIAOMI_TOKEN_PLAN_AMS_API_KEY"],
        Apis::OpenAICompletions,
    ),
    (
        "xiaomi-token-plan-cn",
        "Xiaomi Token Plan CN",
        "https://token-plan-cn.xiaomimimo.com/v1",
        &["XIAOMI_TOKEN_PLAN_CN_API_KEY"],
        Apis::OpenAICompletions,
    ),
    (
        "xiaomi-token-plan-sgp",
        "Xiaomi Token Plan SGP",
        "https://token-plan-sgp.xiaomimimo.com/v1",
        &["XIAOMI_TOKEN_PLAN_SGP_API_KEY"],
        Apis::OpenAICompletions,
    ),
    (
        "zai",
        "Z.AI",
        "https://api.z.ai/api/coding/paas/v4",
        &["ZAI_API_KEY"],
        Apis::OpenAICompletions,
    ),
    (
        "zai-coding-cn",
        "Z.AI Coding CN",
        "https://open.bigmodel.cn/api/coding/paas/v4",
        &["ZAI_CODING_CN_API_KEY"],
        Apis::OpenAICompletions,
    ),
];

fn by_api(entries: Vec<(&str, Arc<dyn ProviderStreams>)>) -> ProviderApi {
    ProviderApi::ByApi(
        entries
            .into_iter()
            .map(|(api, streams)| (api.to_owned(), streams))
            .collect::<HashMap<_, _>>(),
    )
}

fn apis(kind: Apis) -> ProviderApi {
    match kind {
        Apis::AnthropicMessages => ProviderApi::single(AnthropicMessagesApi),
        Apis::OpenAICompletions => ProviderApi::single(OpenAICompletionsApi),
        Apis::OpenAIResponses => ProviderApi::single(OpenAIResponsesApi),
        Apis::GoogleGenerativeAi => ProviderApi::single(GoogleGenerativeAiApi),
        Apis::Mistral => ProviderApi::single(MistralConversationsApi),
        Apis::AnthropicAndCompletions => by_api(vec![
            ("anthropic-messages", Arc::new(AnthropicMessagesApi)),
            ("openai-completions", Arc::new(OpenAICompletionsApi)),
        ]),
    }
}

fn catalog_provider(
    id: &str,
    name: &str,
    base_url: Option<&str>,
    auth: ProviderAuth,
    api: ProviderApi,
) -> BuiltProvider {
    let mut options = CreateProviderOptions::new(id, auth, all_builtin_models(id).to_vec(), api).name(name);
    options.base_url = base_url.map(str::to_owned);
    create_provider(with_operations(id, options))
}

/// The image and classifier implementations of a built-in provider (upstream `images` and
/// `classifiers` in `src/providers/<id>.ts`). The Cloudflare providers register theirs in
/// [`crate::providers::cloudflare`].
fn with_operations(id: &str, options: CreateProviderOptions) -> CreateProviderOptions {
    match id {
        // OpenRouter also serves TypeSafe's System One protocol at /api/v1/systemone.
        "openrouter" => options
            .images(openrouter_images::API, OpenRouterImagesApi)
            .classifier(typesafe_system_one::API, TypeSafeSystemOneApi),
        // OpenCode Zen serves System One at /inference/v1/systemone, AI Gateway at
        // /typesafe/v1/systemone.
        "typesafe" | "opencode" | "vercel-ai-gateway" => {
            options.classifier(typesafe_system_one::API, TypeSafeSystemOneApi)
        }
        "openai" => {
            let mut options = options.classifier(openai_decisions::API, OpenAIDecisionsApi);
            options.filter_all_models = Some(Arc::new(openai_filter_all_models));
            options
        }
        _ => options,
    }
}

/// Sign in with ChatGPT tokens only reach the Responses API; the Decisions API rejects them,
/// so the `openai` provider hides its classifier models while it uses an OAuth credential.
fn openai_filter_all_models(models: &[AnyModel], credential: Option<&Credential>) -> Vec<AnyModel> {
    match credential {
        Some(Credential::OAuth(_)) => models
            .iter()
            .filter(|model| model.model_type() != ModelType::Classifier)
            .cloned()
            .collect(),
        _ => models.to_vec(),
    }
}

/// TypeSafe: System One classifier models only (`TYPESAFE_API_KEY`).
pub fn typesafe_provider() -> BuiltProvider {
    catalog_provider(
        "typesafe",
        "TypeSafe",
        None,
        ProviderAuth::api_key(env_api_key_auth("TypeSafe API key", &["TYPESAFE_API_KEY"])),
        ProviderApi::none(),
    )
}

/// The API-key method's display name: upstream's `envApiKeyAuth` name, usually `"<name> API key"`.
fn api_key_label(id: &str, name: &str) -> String {
    match id {
        "google" => "Gemini API key".into(),
        "huggingface" => "Hugging Face token".into(),
        "kimi-coding" => "Kimi API key".into(),
        "meta" => "Meta Model API key".into(),
        "moonshotai-cn" => "Moonshot AI API key".into(),
        _ => format!("{name} API key"),
    }
}

/// A built-in provider by id, or `None` for unknown or not yet ported providers.
pub fn builtin_provider(id: &str) -> Option<Arc<dyn Provider>> {
    if let Some((id, name, base_url, env, kind)) = SIMPLE_PROVIDERS.iter().find(|entry| entry.0 == id) {
        let mut auth = ProviderAuth::api_key(env_api_key_auth(api_key_label(id, name), env));
        auth.oauth = builtin_oauth(id);
        return Some(Arc::new(catalog_provider(id, name, Some(base_url), auth, apis(*kind))));
    }
    let provider: Arc<dyn Provider> = match id {
        "anthropic" => Arc::new(anthropic_provider()),
        "openai-codex" => Arc::new(openai_codex_provider()),
        "azure" => Arc::new(azure_provider()),
        "opencode" => Arc::new(opencode_provider("opencode", "OpenCode Zen")),
        "opencode-go" => Arc::new(opencode_provider("opencode-go", "OpenCode Go")),
        "cloudflare-ai-gateway" => Arc::new(crate::providers::cloudflare::cloudflare_ai_gateway_provider()),
        "cloudflare-workers-ai" => Arc::new(crate::providers::cloudflare::cloudflare_workers_ai_provider()),
        "github-copilot" => Arc::new(crate::providers::github_copilot::github_copilot_provider()),
        "radius" => Arc::new(crate::providers::radius::radius_provider(
            crate::providers::radius::RadiusProviderOptions::default(),
        )),
        "typesafe" => Arc::new(typesafe_provider()),
        #[cfg(feature = "bedrock")]
        "amazon-bedrock" => Arc::new(cloud::amazon_bedrock_provider()),
        #[cfg(feature = "vertex")]
        "google-vertex" => Arc::new(vertex::google_vertex_provider()),
        _ => return None,
    };
    Some(provider)
}

/// The OAuth login of a table-driven provider, next to its API key (upstream
/// `auth.oauth` in `src/providers/<id>.ts`).
fn builtin_oauth(id: &str) -> Option<Arc<dyn OAuthAuth>> {
    let oauth: Arc<dyn OAuthAuth> = match id {
        "openai" => Arc::new(OpenAIChatGPTOAuth::default()),
        "openrouter" => Arc::new(OpenRouterOAuth::default()),
        "xai" => Arc::new(XaiOAuth::default()),
        "kimi-coding" => Arc::new(KimiCodingOAuth::default()),
        "meta" => Arc::new(MetaOAuth::default()),
        _ => return None,
    };
    Some(oauth)
}

/// Ids of the providers [`builtin_provider`] builds, in catalog order.
pub fn builtin_provider_ids() -> Vec<&'static str> {
    crate::providers::catalog::builtin_provider_ids()
        .into_iter()
        .filter(|id| builtin_provider(id).is_some())
        .collect()
}

/// Every built-in provider.
pub fn builtin_providers() -> Vec<Arc<dyn Provider>> {
    builtin_provider_ids()
        .into_iter()
        .filter_map(builtin_provider)
        .collect()
}

/// A [`Models`] collection with every built-in provider registered.
pub fn builtin_models(options: CreateModelsOptions) -> Models {
    let models = Models::new(options);
    for provider in builtin_providers() {
        models.set_provider_arc(provider);
    }
    models
}

/// OpenAI Codex: the ChatGPT Plus/Pro subscription's Codex backend, OAuth only (upstream
/// `providers/openai-codex.ts`).
pub fn openai_codex_provider() -> BuiltProvider {
    catalog_provider(
        "openai-codex",
        "OpenAI Codex (legacy)",
        Some("https://chatgpt.com/backend-api"),
        ProviderAuth {
            api_key: None,
            oauth: Some(Arc::new(OpenAICodexOAuth::default())),
        },
        ProviderApi::single(OpenAICodexResponsesApi),
    )
}

/// Anthropic: `ANTHROPIC_API_KEY`, `ANTHROPIC_OAUTH_TOKEN`, or `ANTHROPIC_AUTH_TOKEN` sent as a
/// bearer token; OAuth login with a Claude Pro/Max subscription.
pub fn anthropic_provider() -> BuiltProvider {
    catalog_provider(
        "anthropic",
        "Anthropic",
        Some("https://api.anthropic.com"),
        ProviderAuth::api_key(anthropic_auth::AnthropicApiKeyAuth).with_oauth(AnthropicOAuth::default()),
        ProviderApi::single(AnthropicMessagesApi),
    )
}

mod anthropic_auth {
    use super::BoxFuture;
    use crate::auth::{ApiKeyAuth, ApiKeyCredential, ApiKeyInput, AuthInteraction, AuthPrompt, AuthResult, ModelAuth};
    use crate::error::ModelsError;
    use crate::http::headers_of;

    /// Upstream order: stored key, `ANTHROPIC_AUTH_TOKEN` (bearer), `ANTHROPIC_OAUTH_TOKEN`,
    /// `ANTHROPIC_API_KEY`. Workload identity federation is not ported.
    #[derive(Debug)]
    pub(super) struct AnthropicApiKeyAuth;

    impl ApiKeyAuth for AnthropicApiKeyAuth {
        fn name(&self) -> &'static str {
            "Anthropic API key"
        }

        fn resolve<'a>(&'a self, input: ApiKeyInput<'a>) -> BoxFuture<'a, Result<Option<AuthResult>, ModelsError>> {
            Box::pin(async move {
                if let Some(credential) = input.credential
                    && let Some(key) = credential.key.clone().filter(|key| !key.is_empty())
                {
                    return Ok(Some(AuthResult {
                        auth: ModelAuth {
                            api_key: Some(key),
                            ..ModelAuth::default()
                        },
                        env: credential.env.clone(),
                        source: Some("stored credential".into()),
                    }));
                }
                if let Some(token) = input.ctx.env("ANTHROPIC_AUTH_TOKEN") {
                    return Ok(Some(AuthResult {
                        auth: ModelAuth {
                            headers: Some(headers_of([("Authorization", &format!("Bearer {token}"))])),
                            ..ModelAuth::default()
                        },
                        env: None,
                        source: Some("ANTHROPIC_AUTH_TOKEN".into()),
                    }));
                }
                for var in ["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"] {
                    if let Some(key) = input.ctx.env(var) {
                        return Ok(Some(AuthResult {
                            auth: ModelAuth {
                                api_key: Some(key),
                                ..ModelAuth::default()
                            },
                            env: None,
                            source: Some(var.into()),
                        }));
                    }
                }
                Ok(None)
            })
        }

        fn supports_login(&self) -> bool {
            true
        }

        fn login<'a>(
            &'a self,
            interaction: &'a dyn AuthInteraction,
        ) -> BoxFuture<'a, Result<ApiKeyCredential, ModelsError>> {
            Box::pin(async move {
                let key = interaction
                    .prompt(AuthPrompt::Secret {
                        message: "Enter Anthropic API key".into(),
                        placeholder: None,
                    })
                    .await?;
                Ok(ApiKeyCredential {
                    key: Some(key),
                    env: None,
                })
            })
        }
    }
}

/// Wraps an API so the request's `model` field is the Azure deployment name and the base
/// URL comes from the Azure settings, resolved inside the stream so a missing endpoint is a
/// stream error.
struct AzureStreams<S> {
    inner: S,
}

impl<S: ProviderStreams + Clone + 'static> AzureStreams<S> {
    fn prepare(model: &Model, mut options: StreamOptions) -> Result<(Model, StreamOptions), ModelsError> {
        let azure = AzureOpenAIResponsesOptions::default();
        let base_url = resolve_azure_base_url(model, &azure, &options)
            .map_err(|error| ModelsError::new(ModelsErrorCode::Provider, error.to_string()))?;
        let deployment = resolve_deployment_name(model, &azure, &options);
        if deployment != model.id {
            let previous = options.on_payload.take();
            let hook: OnPayload = Arc::new(move |payload, payload_model| {
                let mut params = payload.clone();
                if let Some(object) = params.as_object_mut() {
                    object.insert("model".into(), deployment.clone().into());
                }
                let previous = previous.clone();
                let payload_model = payload_model.clone();
                Box::pin(async move {
                    match previous {
                        Some(previous) => Ok(previous(&params, &payload_model).await?.or(Some(params))),
                        None => Ok(Some(params)),
                    }
                })
            });
            options.on_payload = Some(hook);
        }
        let mut model = model.clone();
        model.base_url = base_url;
        Ok((model, options))
    }
}

impl<S: ProviderStreams + Clone + 'static> ProviderStreams for AzureStreams<S> {
    fn stream(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        let inner = self.inner.clone();
        let (model_c, context) = (model.clone(), context.clone());
        lazy_stream(model, options.cancel.clone(), async move {
            let (model, options) = Self::prepare(&model_c, options)?;
            Ok(inner.stream(&model, &context, options))
        })
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let inner = self.inner.clone();
        let (model_c, context) = (model.clone(), context.clone());
        lazy_stream(model, options.base.cancel.clone(), async move {
            let mut options = options;
            let (model, base) = Self::prepare(&model_c, std::mem::take(&mut options.base))?;
            options.base = base;
            Ok(inner.stream_simple(&model, &context, options))
        })
    }
}

/// Azure OpenAI: Responses models plus Chat Completions models on the same resource.
pub fn azure_provider() -> BuiltProvider {
    catalog_provider(
        "azure",
        "Azure",
        None,
        ProviderAuth::api_key(env_api_key_auth("Azure OpenAI API key", &["AZURE_OPENAI_API_KEY"])),
        by_api(vec![
            ("azure-openai-responses", Arc::new(AzureOpenAIResponsesApi)),
            (
                "openai-completions",
                Arc::new(AzureStreams {
                    inner: OpenAICompletionsApi,
                }),
            ),
        ]),
    )
}

const OPENCODE_SESSION_HEADER: &str = "x-opencode-session";

/// Adds OpenCode's per-conversation routing header from `session_id`.
#[derive(Clone)]
struct OpenCodeStreams {
    inner: Arc<dyn ProviderStreams>,
}

fn with_session_header(mut options: StreamOptions) -> StreamOptions {
    let Some(session) = options.session_id.clone() else {
        return options;
    };
    let headers = options.headers.get_or_insert_with(Default::default);
    if !headers
        .keys()
        .any(|name| name.eq_ignore_ascii_case(OPENCODE_SESSION_HEADER))
    {
        headers.insert(OPENCODE_SESSION_HEADER.into(), Some(session));
    }
    options
}

impl ProviderStreams for OpenCodeStreams {
    fn stream(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        self.inner.stream(model, context, with_session_header(options))
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        mut options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        options.base = with_session_header(options.base);
        self.inner.stream_simple(model, context, options)
    }
}

fn opencode_provider(id: &str, name: &str) -> BuiltProvider {
    let wrap = |inner: Arc<dyn ProviderStreams>| -> Arc<dyn ProviderStreams> { Arc::new(OpenCodeStreams { inner }) };
    catalog_provider(
        id,
        name,
        None,
        ProviderAuth::api_key(env_api_key_auth("OpenCode API key", &["OPENCODE_API_KEY"])),
        by_api(vec![
            ("anthropic-messages", wrap(Arc::new(AnthropicMessagesApi))),
            ("google-generative-ai", wrap(Arc::new(GoogleGenerativeAiApi))),
            ("openai-completions", wrap(Arc::new(OpenAICompletionsApi))),
            ("openai-responses", wrap(Arc::new(OpenAIResponsesApi))),
        ]),
    )
}

#[cfg(feature = "bedrock")]
mod cloud {
    use super::{BoxFuture, ProviderApi, catalog_provider};
    use crate::api::bedrock_converse_stream::BedrockConverseStreamApi;
    use crate::auth::{
        ApiKeyAuth, ApiKeyCredential, ApiKeyInput, AuthEvent, AuthInfoLink, AuthInteraction, AuthPrompt, AuthResult,
        AuthSelectOption, ModelAuth, ProviderAuth,
    };
    use crate::error::{ModelsError, ModelsErrorCode};
    use crate::provider::BuiltProvider;
    use crate::types::ProviderEnv;

    /// Bedrock accepts a bearer token or the AWS SDK's default credential chain; resolve
    /// detects ambient AWS credentials without copying them into the credential store.
    #[derive(Debug)]
    struct BedrockAuth;

    fn result(source: &str, env: Option<ProviderEnv>) -> AuthResult {
        AuthResult {
            auth: ModelAuth::default(),
            env,
            source: Some(source.into()),
        }
    }

    impl ApiKeyAuth for BedrockAuth {
        fn name(&self) -> &'static str {
            "AWS credentials or bearer token"
        }

        fn resolve<'a>(&'a self, input: ApiKeyInput<'a>) -> BoxFuture<'a, Result<Option<AuthResult>, ModelsError>> {
            Box::pin(async move {
                let env = |name: &str| input.ctx.env(name);
                let credential = input.credential;
                if let Some(credential) = credential
                    && let Some(key) = credential.key.clone().filter(|key| !key.is_empty())
                {
                    return Ok(Some(AuthResult {
                        auth: ModelAuth {
                            api_key: Some(key),
                            ..ModelAuth::default()
                        },
                        env: credential.env.clone(),
                        source: Some("stored credential".into()),
                    }));
                }
                if env("AWS_BEARER_TOKEN_BEDROCK").is_some() {
                    return Ok(Some(result("AWS_BEARER_TOKEN_BEDROCK", None)));
                }
                let stored_profile = credential
                    .and_then(|c| c.env.as_ref())
                    .and_then(|e| e.get("AWS_PROFILE"));
                if stored_profile.is_some() || env("AWS_PROFILE").is_some() {
                    let source = if stored_profile.is_some() {
                        "stored credential"
                    } else {
                        "AWS_PROFILE"
                    };
                    return Ok(Some(result(source, credential.and_then(|c| c.env.clone()))));
                }
                if env("AWS_ACCESS_KEY_ID").is_some() && env("AWS_SECRET_ACCESS_KEY").is_some() {
                    return Ok(Some(result("AWS access keys", None)));
                }
                if env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI").is_some()
                    || env("AWS_CONTAINER_CREDENTIALS_FULL_URI").is_some()
                {
                    return Ok(Some(result("ECS task role", None)));
                }
                if env("AWS_WEB_IDENTITY_TOKEN_FILE").is_some() {
                    return Ok(Some(result("web identity token", None)));
                }
                Ok(None)
            })
        }

        fn supports_login(&self) -> bool {
            true
        }

        fn login<'a>(
            &'a self,
            interaction: &'a dyn AuthInteraction,
        ) -> BoxFuture<'a, Result<ApiKeyCredential, ModelsError>> {
            Box::pin(async move {
                let option = |id: &str, label: &str| AuthSelectOption {
                    id: id.into(),
                    label: label.into(),
                    description: None,
                };
                let method = interaction
                    .prompt(AuthPrompt::Select {
                        message: "Select Amazon Bedrock authentication method:".into(),
                        options: vec![
                            option("bearer-token", "Bearer token"),
                            option("aws-profile", "AWS profile"),
                            option("credential-chain", "Existing AWS credential chain"),
                        ],
                    })
                    .await?;
                if method == "bearer-token" {
                    let key = interaction
                        .prompt(AuthPrompt::Secret {
                            message: "Enter Amazon Bedrock bearer token".into(),
                            placeholder: None,
                        })
                        .await?;
                    return Ok(ApiKeyCredential {
                        key: Some(key),
                        env: None,
                    });
                }
                interaction.notify(AuthEvent::Info {
                    message: "Amazon Bedrock supports AWS profiles, IAM credentials, and role-based credentials."
                        .into(),
                    links: vec![AuthInfoLink {
                        url: "https://docs.aws.amazon.com/sdkref/latest/guide/standardized-credentials.html".into(),
                        label: Some("AWS credential provider chain".into()),
                    }],
                });
                match method.as_str() {
                    "aws-profile" => {
                        let profile = interaction
                            .prompt(AuthPrompt::Text {
                                message: "Enter AWS profile name".into(),
                                placeholder: None,
                            })
                            .await?;
                        Ok(ApiKeyCredential {
                            key: None,
                            env: Some([("AWS_PROFILE".to_owned(), profile)].into()),
                        })
                    }
                    "credential-chain" => {
                        interaction
                            .prompt(AuthPrompt::Text {
                                message: "Configure AWS credentials, then press Enter to continue".into(),
                                placeholder: None,
                            })
                            .await?;
                        Ok(ApiKeyCredential::default())
                    }
                    other => Err(ModelsError::new(
                        ModelsErrorCode::Auth,
                        format!("Unknown Amazon Bedrock auth method: {other}"),
                    )),
                }
            })
        }
    }

    pub(super) fn amazon_bedrock_provider() -> BuiltProvider {
        catalog_provider(
            "amazon-bedrock",
            "Amazon Bedrock",
            None,
            ProviderAuth::api_key(BedrockAuth),
            ProviderApi::single(BedrockConverseStreamApi),
        )
    }
}

#[cfg(feature = "vertex")]
mod vertex {
    use super::{BoxFuture, ProviderApi, catalog_provider};
    use crate::api::google_vertex::GoogleVertexApi;
    use crate::auth::{
        ApiKeyAuth, ApiKeyCredential, ApiKeyInput, AuthEvent, AuthInfoLink, AuthInteraction, AuthPrompt, AuthResult,
        AuthSelectOption, ModelAuth, ProviderAuth,
    };
    use crate::error::{ModelsError, ModelsErrorCode};
    use crate::provider::BuiltProvider;
    use crate::types::ProviderEnv;

    const VERTEX_ADC_PATH: &str = "~/.config/gcloud/application_default_credentials.json";

    /// Vertex accepts an API key or Application Default Credentials. ADC also needs a project
    /// and a location, which the API implementation reads from the environment.
    #[derive(Debug)]
    struct VertexAuth;

    impl ApiKeyAuth for VertexAuth {
        fn name(&self) -> &'static str {
            "Google Cloud credentials"
        }

        fn resolve<'a>(&'a self, input: ApiKeyInput<'a>) -> BoxFuture<'a, Result<Option<AuthResult>, ModelsError>> {
            Box::pin(async move {
                let credential = input.credential;
                let stored = |name: &str| {
                    credential
                        .and_then(|credential| credential.env.as_ref())
                        .and_then(|env| env.get(name).cloned())
                };
                let env = |name: &str| stored(name).or_else(|| input.ctx.env(name));

                let stored_key = credential.and_then(|credential| credential.key.clone());
                let (key, source) = match stored_key {
                    Some(key) => (Some(key), "stored credential"),
                    None => (input.ctx.env("GOOGLE_CLOUD_API_KEY"), "GOOGLE_CLOUD_API_KEY"),
                };
                if let Some(key) = key.filter(|key| !key.is_empty()) {
                    return Ok(Some(AuthResult {
                        auth: ModelAuth {
                            api_key: Some(key),
                            ..ModelAuth::default()
                        },
                        env: None,
                        source: Some(source.into()),
                    }));
                }

                let credentials_path = env("GOOGLE_APPLICATION_CREDENTIALS");
                let has_credentials = input
                    .ctx
                    .file_exists(credentials_path.as_deref().unwrap_or(VERTEX_ADC_PATH));
                let project = env("GOOGLE_CLOUD_PROJECT").or_else(|| env("GCLOUD_PROJECT"));
                let location = env("GOOGLE_CLOUD_LOCATION");
                if has_credentials && project.is_some() && location.is_some() {
                    return Ok(Some(AuthResult {
                        auth: ModelAuth::default(),
                        env: credential.and_then(|credential| credential.env.clone()),
                        source: Some(
                            if credential.is_some() {
                                "stored credential"
                            } else {
                                "gcloud application default credentials"
                            }
                            .into(),
                        ),
                    }));
                }
                Ok(None)
            })
        }

        fn supports_login(&self) -> bool {
            true
        }

        fn login<'a>(
            &'a self,
            interaction: &'a dyn AuthInteraction,
        ) -> BoxFuture<'a, Result<ApiKeyCredential, ModelsError>> {
            Box::pin(async move {
                let option = |id: &str, label: &str| AuthSelectOption {
                    id: id.into(),
                    label: label.into(),
                    description: None,
                };
                let text = |message: &str| AuthPrompt::Text {
                    message: message.into(),
                    placeholder: None,
                };
                let method = interaction
                    .prompt(AuthPrompt::Select {
                        message: "Select Google Vertex AI authentication method:".into(),
                        options: vec![
                            option("api-key", "Google Cloud API key"),
                            option("adc", "Application Default Credentials"),
                            option("service-account", "Service account credentials file"),
                        ],
                    })
                    .await?;
                if method == "api-key" {
                    let key = interaction
                        .prompt(AuthPrompt::Secret {
                            message: "Enter Google Cloud API key".into(),
                            placeholder: None,
                        })
                        .await?;
                    return Ok(ApiKeyCredential {
                        key: Some(key),
                        env: None,
                    });
                }
                if method != "adc" && method != "service-account" {
                    return Err(ModelsError::new(
                        ModelsErrorCode::Auth,
                        format!("Unknown Google Vertex AI auth method: {method}"),
                    ));
                }
                interaction.notify(AuthEvent::Info {
                    message: if method == "adc" {
                        "Run `gcloud auth application-default login`, then provide the project and location."
                    } else {
                        "Provide a service account credentials file, project, and location."
                    }
                    .into(),
                    links: vec![AuthInfoLink {
                        url: "https://cloud.google.com/docs/authentication/provide-credentials-adc".into(),
                        label: Some("Application Default Credentials".into()),
                    }],
                });
                let project = interaction.prompt(text("Enter Google Cloud project ID")).await?;
                let location = interaction.prompt(text("Enter Google Cloud location")).await?;
                let mut env = ProviderEnv::new();
                env.insert("GOOGLE_CLOUD_PROJECT".into(), project);
                env.insert("GOOGLE_CLOUD_LOCATION".into(), location);
                if method == "service-account" {
                    let path = interaction
                        .prompt(text("Enter service account credentials file path"))
                        .await?;
                    env.insert("GOOGLE_APPLICATION_CREDENTIALS".into(), path);
                }
                Ok(ApiKeyCredential {
                    key: None,
                    env: Some(env),
                })
            })
        }
    }

    pub(super) fn google_vertex_provider() -> BuiltProvider {
        catalog_provider(
            "google-vertex",
            "Google Vertex AI",
            None,
            ProviderAuth::api_key(VertexAuth),
            ProviderApi::single(GoogleVertexApi),
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::auth::AuthContext;
        use std::collections::BTreeMap;

        #[derive(Default)]
        struct Ctx {
            env: BTreeMap<&'static str, &'static str>,
            files: Vec<&'static str>,
        }

        impl AuthContext for Ctx {
            fn env(&self, name: &str) -> Option<String> {
                self.env.get(name).map(|value| (*value).to_owned())
            }

            fn file_exists(&self, path: &str) -> bool {
                self.files.contains(&path)
            }
        }

        async fn resolve(ctx: &Ctx, credential: Option<&ApiKeyCredential>) -> Option<AuthResult> {
            VertexAuth.resolve(ApiKeyInput { ctx, credential }).await.unwrap()
        }

        #[tokio::test]
        async fn api_key_from_store_or_env() {
            let stored = ApiKeyCredential {
                key: Some("stored".into()),
                env: None,
            };
            let ctx = Ctx {
                env: [("GOOGLE_CLOUD_API_KEY", "env-key")].into(),
                ..Ctx::default()
            };
            let result = resolve(&ctx, Some(&stored)).await.unwrap();
            assert_eq!(result.auth.api_key.as_deref(), Some("stored"));
            assert_eq!(result.source.as_deref(), Some("stored credential"));

            let result = resolve(&ctx, None).await.unwrap();
            assert_eq!(result.auth.api_key.as_deref(), Some("env-key"));
            assert_eq!(result.source.as_deref(), Some("GOOGLE_CLOUD_API_KEY"));
        }

        #[tokio::test]
        async fn adc_needs_credentials_project_and_location() {
            let mut ctx = Ctx {
                env: [("GOOGLE_CLOUD_PROJECT", "p")].into(),
                files: vec![VERTEX_ADC_PATH],
            };
            assert!(resolve(&ctx, None).await.is_none());

            ctx.env.insert("GOOGLE_CLOUD_LOCATION", "us-central1");
            let result = resolve(&ctx, None).await.unwrap();
            assert_eq!(result.auth.api_key, None);
            assert_eq!(result.source.as_deref(), Some("gcloud application default credentials"));

            ctx.files.clear();
            assert!(resolve(&ctx, None).await.is_none());
        }

        #[tokio::test]
        async fn stored_service_account_env() {
            let stored = ApiKeyCredential {
                key: None,
                env: Some(
                    [
                        ("GOOGLE_CLOUD_PROJECT".to_owned(), "p".to_owned()),
                        ("GOOGLE_CLOUD_LOCATION".to_owned(), "global".to_owned()),
                        ("GOOGLE_APPLICATION_CREDENTIALS".to_owned(), "/keys/sa.json".to_owned()),
                    ]
                    .into(),
                ),
            };
            let ctx = Ctx {
                files: vec!["/keys/sa.json"],
                ..Ctx::default()
            };
            let result = resolve(&ctx, Some(&stored)).await.unwrap();
            assert_eq!(result.env, stored.env);
            assert_eq!(result.source.as_deref(), Some("stored credential"));
        }

        #[test]
        fn provider_serves_the_vertex_catalog() {
            let provider = google_vertex_provider();
            assert_eq!(crate::provider::Provider::id(&provider), "google-vertex");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_key_names_match_upstream() {
        for (id, name) in [
            ("google", "Gemini API key"),
            ("huggingface", "Hugging Face token"),
            ("kimi-coding", "Kimi API key"),
            ("meta", "Meta Model API key"),
            ("moonshotai-cn", "Moonshot AI API key"),
            ("groq", "Groq API key"),
        ] {
            let provider = builtin_provider(id).unwrap();
            assert_eq!(provider.auth().api_key.as_ref().unwrap().name(), name, "{id}");
        }
    }

    #[test]
    fn builds_every_supported_catalog_provider() {
        let ids = builtin_provider_ids();
        for expected in [
            "anthropic",
            "openai",
            "google",
            "mistral",
            "azure",
            "openrouter",
            "groq",
            "opencode",
        ] {
            assert!(ids.contains(&expected), "{expected}");
        }
        let models = builtin_models(CreateModelsOptions::default());
        let model = models.get_model("anthropic", "claude-sonnet-4-5").unwrap();
        assert_eq!(model.api, "anthropic-messages");
        assert!(models.get_model("openai", "gpt-4o-mini").is_some());
        assert!(models.get_model("github-copilot", "claude-sonnet-4.6").is_some());
    }

    #[test]
    fn opencode_adds_session_header_once() {
        let options = StreamOptions {
            session_id: Some("s1".into()),
            ..StreamOptions::default()
        };
        let headers = with_session_header(options).headers.unwrap();
        assert_eq!(headers.get(OPENCODE_SESSION_HEADER), Some(&Some("s1".to_owned())));
        let mut explicit = StreamOptions {
            session_id: Some("s1".into()),
            ..StreamOptions::default()
        };
        explicit.headers = Some(
            [("X-OpenCode-Session".to_owned(), Some("mine".to_owned()))]
                .into_iter()
                .collect(),
        );
        let headers = with_session_header(explicit).headers.unwrap();
        assert_eq!(headers.len(), 1);
        assert_eq!(headers.get("X-OpenCode-Session"), Some(&Some("mine".to_owned())));
    }
}
