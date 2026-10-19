//! Cloudflare AI Gateway and Cloudflare Workers AI.
//!
//! Ports upstream `providers/cloudflare-auth.ts` (auth), `providers/cloudflare-stream.ts`
//! (endpoint placeholders, for chat and classifier requests), `providers/cloudflare-ai-gateway.ts` and
//! `providers/cloudflare-workers-ai.ts` (provider factories). The endpoints themselves live in
//! [`crate::api::cloudflare`].
//!
//! Catalog base URLs carry `{CLOUDFLARE_ACCOUNT_ID}` and `{CLOUDFLARE_GATEWAY_ID}`
//! placeholders. Auth resolves the account and gateway ids into the provider env, and
//! [`CloudflareStreams`] fills the placeholders in from `options.env` before the request goes
//! to the wrapped API implementation.

use futures::future::BoxFuture;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use crate::api::anthropic_messages::AnthropicMessagesApi;
use crate::api::cloudflare_workers_ai_system_one::CloudflareWorkersAiSystemOneApi;
use crate::api::openai_completions::OpenAICompletionsApi;
use crate::api::openai_responses::OpenAIResponsesApi;
use crate::auth::{
    ApiKeyAuth, ApiKeyCredential, ApiKeyInput, AuthContext, AuthInteraction, AuthPrompt, AuthResult, ModelAuth,
    ProviderAuth,
};
use crate::classifier::{ClassifierContext, ClassifierOptions, ClassifierResult, ProviderClassifier};
use crate::error::ModelsError;
use crate::events::AssistantMessageEventStream;
use crate::model::{AnyModel, ClassifierModel, Model, ModelType};
use crate::options::{SimpleStreamOptions, StreamOptions};
use crate::provider::{BuiltProvider, CreateProviderOptions, ProviderApi, ProviderStreams, create_provider};
use crate::providers::catalog::all_builtin_models;
use crate::types::{ProviderEnv, ProviderHeaders, TranscriptContext};

/// Env var of the Cloudflare API token.
pub const CLOUDFLARE_API_KEY: &str = "CLOUDFLARE_API_KEY";
/// Env var of the Cloudflare account id.
pub const CLOUDFLARE_ACCOUNT_ID: &str = "CLOUDFLARE_ACCOUNT_ID";
/// Env var of the AI Gateway id.
pub const CLOUDFLARE_GATEWAY_ID: &str = "CLOUDFLARE_GATEWAY_ID";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CloudflareAuthKind {
    WorkersAi,
    AiGateway,
}

/// Per-field merge: prefer the credential value, fall back to ambient env. A credential
/// carrying only the API key must still pick up the account / gateway id from the
/// environment.
fn resolve_value(name: &str, ctx: &dyn AuthContext, credential: Option<&ApiKeyCredential>) -> Option<String> {
    let from_credential = credential.and_then(|credential| {
        if name == CLOUDFLARE_API_KEY {
            credential.key.clone()
        } else {
            credential.env.as_ref().and_then(|env| env.get(name).cloned())
        }
    });
    from_credential.or_else(|| ctx.env(name))
}

struct ResolvedCloudflareEnv {
    api_key: String,
    env: ProviderEnv,
    source: &'static str,
}

fn resolve_cloudflare_env(
    kind: CloudflareAuthKind,
    ctx: &dyn AuthContext,
    credential: Option<&ApiKeyCredential>,
) -> Option<ResolvedCloudflareEnv> {
    let api_key = resolve_value(CLOUDFLARE_API_KEY, ctx, credential);
    let account_id = resolve_value(CLOUDFLARE_ACCOUNT_ID, ctx, credential);
    let gateway_id = match kind {
        CloudflareAuthKind::AiGateway => resolve_value(CLOUDFLARE_GATEWAY_ID, ctx, credential),
        CloudflareAuthKind::WorkersAi => None,
    };

    // Empty strings count as missing, as in upstream's truthiness checks.
    let api_key = api_key.filter(|value| !value.is_empty())?;
    let account_id = account_id.filter(|value| !value.is_empty())?;
    let gateway_id = gateway_id.filter(|value| !value.is_empty());
    if kind == CloudflareAuthKind::AiGateway && gateway_id.is_none() {
        return None;
    }

    let mut env = ProviderEnv::new();
    env.insert(CLOUDFLARE_ACCOUNT_ID.into(), account_id);
    if let Some(gateway_id) = gateway_id {
        env.insert(CLOUDFLARE_GATEWAY_ID.into(), gateway_id);
    }
    Some(ResolvedCloudflareEnv {
        api_key,
        env,
        source: if credential.is_some() {
            "stored credential"
        } else {
            CLOUDFLARE_API_KEY
        },
    })
}

fn text_prompt(message: &str) -> AuthPrompt {
    AuthPrompt::Text {
        message: message.into(),
        placeholder: None,
    }
}

fn secret_prompt(message: &str) -> AuthPrompt {
    AuthPrompt::Secret {
        message: message.into(),
        placeholder: None,
    }
}

/// Workers AI auth: `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID`, from the stored
/// credential (key and `env`) or the environment, merged per field. The key is sent as the
/// API key; the account id goes into the resolved provider env.
#[derive(Clone, Copy, Debug, Default)]
pub struct CloudflareWorkersAiAuth;

/// Upstream `cloudflareWorkersAIAuth()`.
pub fn cloudflare_workers_ai_auth() -> CloudflareWorkersAiAuth {
    CloudflareWorkersAiAuth
}

impl ApiKeyAuth for CloudflareWorkersAiAuth {
    fn name(&self) -> &'static str {
        "Cloudflare API key"
    }

    fn resolve<'a>(&'a self, input: ApiKeyInput<'a>) -> BoxFuture<'a, Result<Option<AuthResult>, ModelsError>> {
        Box::pin(async move {
            let Some(resolved) = resolve_cloudflare_env(CloudflareAuthKind::WorkersAi, input.ctx, input.credential)
            else {
                return Ok(None);
            };
            Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some(resolved.api_key),
                    ..ModelAuth::default()
                },
                env: Some(resolved.env),
                source: Some(resolved.source.into()),
            }))
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
            let key = interaction.prompt(secret_prompt("Enter Cloudflare API key")).await?;
            let account_id = interaction.prompt(text_prompt("Enter Cloudflare account ID")).await?;
            Ok(ApiKeyCredential {
                key: Some(key),
                env: Some([(CLOUDFLARE_ACCOUNT_ID.to_owned(), account_id)].into()),
            })
        })
    }
}

/// AI Gateway auth: `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_GATEWAY_ID`,
/// merged per field like [`CloudflareWorkersAiAuth`]. The key is sent as
/// `cf-aig-authorization: Bearer <key>`, and `Authorization` / `x-api-key` are suppressed so
/// the gateway does not mistake a placeholder for a BYOK provider key. An explicit
/// `Authorization` request header (BYOK) still wins over the suppression.
#[derive(Clone, Copy, Debug, Default)]
pub struct CloudflareAiGatewayAuth;

/// Upstream `cloudflareAIGatewayAuth()`.
pub fn cloudflare_ai_gateway_auth() -> CloudflareAiGatewayAuth {
    CloudflareAiGatewayAuth
}

impl ApiKeyAuth for CloudflareAiGatewayAuth {
    fn name(&self) -> &'static str {
        "Cloudflare API key"
    }

    fn resolve<'a>(&'a self, input: ApiKeyInput<'a>) -> BoxFuture<'a, Result<Option<AuthResult>, ModelsError>> {
        Box::pin(async move {
            let Some(resolved) = resolve_cloudflare_env(CloudflareAuthKind::AiGateway, input.ctx, input.credential)
            else {
                return Ok(None);
            };
            let headers: ProviderHeaders = [
                (
                    "cf-aig-authorization".to_owned(),
                    Some(format!("Bearer {}", resolved.api_key)),
                ),
                ("Authorization".to_owned(), None),
                ("x-api-key".to_owned(), None),
            ]
            .into_iter()
            .collect();
            Ok(Some(AuthResult {
                auth: ModelAuth {
                    headers: Some(headers),
                    ..ModelAuth::default()
                },
                env: Some(resolved.env),
                source: Some(resolved.source.into()),
            }))
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
            let key = interaction.prompt(secret_prompt("Enter Cloudflare API key")).await?;
            let account_id = interaction.prompt(text_prompt("Enter Cloudflare account ID")).await?;
            let gateway_id = interaction
                .prompt(text_prompt("Enter Cloudflare AI Gateway ID"))
                .await?;
            Ok(ApiKeyCredential {
                key: Some(key),
                env: Some(
                    [
                        (CLOUDFLARE_ACCOUNT_ID.to_owned(), account_id),
                        (CLOUDFLARE_GATEWAY_ID.to_owned(), gateway_id),
                    ]
                    .into(),
                ),
            })
        })
    }
}

/// `base_url` with `{CLOUDFLARE_ACCOUNT_ID}` and `{CLOUDFLARE_GATEWAY_ID}` replaced by the
/// values in `env`. A placeholder without a value in `env` stays as it is.
pub fn resolve_cloudflare_base_url(base_url: &str, env: &ProviderEnv) -> String {
    let mut url = base_url.to_owned();
    for name in [CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_GATEWAY_ID] {
        if let Some(value) = env.get(name) {
            url = url.replace(&format!("{{{name}}}"), value);
        }
    }
    url
}

/// `model` with its endpoint placeholders filled in from the resolved provider env
/// (upstream `resolveCloudflareModel`). Borrows the model when nothing changes.
pub fn resolve_cloudflare_model<'a>(model: &'a Model, env: Option<&ProviderEnv>) -> Cow<'a, Model> {
    let Some(env) = env else {
        return Cow::Borrowed(model);
    };
    let base_url = resolve_cloudflare_base_url(&model.base_url, env);
    if base_url == model.base_url {
        return Cow::Borrowed(model);
    }
    let mut model = model.clone();
    model.base_url = base_url;
    Cow::Owned(model)
}

/// Wraps an API implementation so Cloudflare account/gateway endpoint placeholders
/// materialize from `options.env` before dispatch (upstream `cloudflareStreams`).
#[derive(Clone)]
pub struct CloudflareStreams {
    inner: Arc<dyn ProviderStreams>,
}

impl std::fmt::Debug for CloudflareStreams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CloudflareStreams").finish_non_exhaustive()
    }
}

/// Wrap `inner` in [`CloudflareStreams`].
pub fn cloudflare_streams(inner: impl ProviderStreams + 'static) -> CloudflareStreams {
    CloudflareStreams { inner: Arc::new(inner) }
}

impl ProviderStreams for CloudflareStreams {
    fn stream(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        let model = resolve_cloudflare_model(model, options.env.as_ref());
        self.inner.stream(&model, context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let model = resolve_cloudflare_model(model, options.base.env.as_ref());
        self.inner.stream_simple(&model, context, options)
    }
}

/// Classifier counterpart of [`CloudflareStreams`] (upstream `cloudflareClassifier`): fills in
/// the endpoint placeholders of `model.base_url` from `options.base.env` before dispatch.
#[derive(Clone)]
pub struct CloudflareClassifier {
    inner: Arc<dyn ProviderClassifier>,
}

impl std::fmt::Debug for CloudflareClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CloudflareClassifier").finish_non_exhaustive()
    }
}

/// Wrap `inner` in [`CloudflareClassifier`].
pub fn cloudflare_classifier(inner: impl ProviderClassifier + 'static) -> CloudflareClassifier {
    CloudflareClassifier { inner: Arc::new(inner) }
}

impl ProviderClassifier for CloudflareClassifier {
    fn classify<'a>(
        &'a self,
        model: &'a ClassifierModel,
        context: &'a ClassifierContext,
        options: ClassifierOptions,
    ) -> BoxFuture<'a, ClassifierResult> {
        let resolved = options
            .base
            .env
            .as_ref()
            .map(|env| resolve_cloudflare_base_url(&model.base_url, env))
            .filter(|base_url| *base_url != model.base_url);
        match resolved {
            None => self.inner.classify(model, context, options),
            Some(base_url) => {
                let model = ClassifierModel {
                    base_url,
                    ..model.clone()
                };
                Box::pin(async move { self.inner.classify(&model, context, options).await })
            }
        }
    }
}

fn catalog_models(provider: &str, types: &[ModelType]) -> Vec<AnyModel> {
    all_builtin_models(provider)
        .iter()
        .filter(|model| types.contains(&model.model_type()))
        .cloned()
        .collect()
}

/// Cloudflare AI Gateway: Anthropic, OpenAI Responses and Workers AI (`/compat`) models
/// behind one gateway.
pub fn cloudflare_ai_gateway_provider() -> BuiltProvider {
    // The api map is pinned to all three APIs: models.dev's gateway catalog drops and
    // restores `workers-ai/*` (openai-completions) entries over time.
    let api: HashMap<String, Arc<dyn ProviderStreams>> = HashMap::from([
        (
            "anthropic-messages".to_owned(),
            Arc::new(cloudflare_streams(AnthropicMessagesApi)) as Arc<dyn ProviderStreams>,
        ),
        (
            "openai-completions".to_owned(),
            Arc::new(cloudflare_streams(OpenAICompletionsApi)) as Arc<dyn ProviderStreams>,
        ),
        (
            "openai-responses".to_owned(),
            Arc::new(cloudflare_streams(OpenAIResponsesApi)) as Arc<dyn ProviderStreams>,
        ),
    ]);
    create_provider(
        CreateProviderOptions::new(
            "cloudflare-ai-gateway",
            ProviderAuth::api_key(cloudflare_ai_gateway_auth()),
            catalog_models("cloudflare-ai-gateway", &[ModelType::Chat]),
            ProviderApi::ByApi(api),
        )
        .name("Cloudflare AI Gateway"),
    )
}

/// Cloudflare Workers AI: chat models over the OpenAI-compatible endpoint and System One
/// classifier models over the REST `/ai/run` endpoint.
pub fn cloudflare_workers_ai_provider() -> BuiltProvider {
    create_provider(
        CreateProviderOptions::new(
            "cloudflare-workers-ai",
            ProviderAuth::api_key(cloudflare_workers_ai_auth()),
            catalog_models("cloudflare-workers-ai", &[ModelType::Chat, ModelType::Classifier]),
            ProviderApi::single(cloudflare_streams(OpenAICompletionsApi)),
        )
        .name("Cloudflare Workers AI")
        .classifier(
            crate::api::cloudflare_workers_ai_system_one::API,
            cloudflare_classifier(CloudflareWorkersAiSystemOneApi),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{AuthEvent, AuthResolutionOverrides};
    use crate::models::{CreateModelsOptions, Models};
    use crate::provider::Provider;
    use crate::transcript::normalize_context;
    use crate::types::Context;
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Ctx {
        env: BTreeMap<&'static str, &'static str>,
    }

    impl AuthContext for Ctx {
        fn env(&self, name: &str) -> Option<String> {
            self.env.get(name).map(|value| (*value).to_owned())
        }

        fn file_exists(&self, _path: &str) -> bool {
            false
        }
    }

    fn models_with_env(env: &[(&'static str, &'static str)], provider: BuiltProvider) -> Models {
        let models = Models::new(CreateModelsOptions {
            auth_context: Some(Arc::new(Ctx {
                env: env.iter().copied().collect(),
            })),
            ..CreateModelsOptions::default()
        });
        models.set_provider(provider);
        models
    }

    async fn get_auth(models: &Models, provider: &str) -> Option<AuthResult> {
        models
            .get_auth(provider, &AuthResolutionOverrides::default())
            .await
            .unwrap()
    }

    fn env_of(pairs: &[(&str, &str)]) -> ProviderEnv {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[tokio::test]
    async fn workers_ai_requires_account_config_and_returns_scoped_env() {
        let missing_account = models_with_env(&[("CLOUDFLARE_API_KEY", "cf-key")], cloudflare_workers_ai_provider());
        assert!(!missing_account.get_models(Some("cloudflare-workers-ai")).is_empty());
        assert!(get_auth(&missing_account, "cloudflare-workers-ai").await.is_none());

        let configured = models_with_env(
            &[
                ("CLOUDFLARE_API_KEY", "cf-key"),
                ("CLOUDFLARE_ACCOUNT_ID", "account-id"),
            ],
            cloudflare_workers_ai_provider(),
        );
        let result = get_auth(&configured, "cloudflare-workers-ai").await.unwrap();
        assert_eq!(
            result.auth,
            ModelAuth {
                api_key: Some("cf-key".into()),
                ..ModelAuth::default()
            }
        );
        assert_eq!(result.env, Some(env_of(&[("CLOUDFLARE_ACCOUNT_ID", "account-id")])));
        assert_eq!(result.source.as_deref(), Some("CLOUDFLARE_API_KEY"));
    }

    #[tokio::test]
    async fn ai_gateway_requires_account_and_gateway_config_and_returns_scoped_env_headers() {
        let missing_gateway = models_with_env(
            &[
                ("CLOUDFLARE_API_KEY", "cf-key"),
                ("CLOUDFLARE_ACCOUNT_ID", "account-id"),
            ],
            cloudflare_ai_gateway_provider(),
        );
        assert!(get_auth(&missing_gateway, "cloudflare-ai-gateway").await.is_none());

        let configured = models_with_env(
            &[
                ("CLOUDFLARE_API_KEY", "cf-key"),
                ("CLOUDFLARE_ACCOUNT_ID", "account-id"),
                ("CLOUDFLARE_GATEWAY_ID", "gateway-id"),
            ],
            cloudflare_ai_gateway_provider(),
        );
        let result = get_auth(&configured, "cloudflare-ai-gateway").await.unwrap();
        let expected: ProviderHeaders = [
            ("cf-aig-authorization".to_owned(), Some("Bearer cf-key".to_owned())),
            ("Authorization".to_owned(), None),
            ("x-api-key".to_owned(), None),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            result.auth,
            ModelAuth {
                headers: Some(expected),
                ..ModelAuth::default()
            }
        );
        assert_eq!(
            result.env,
            Some(env_of(&[
                ("CLOUDFLARE_ACCOUNT_ID", "account-id"),
                ("CLOUDFLARE_GATEWAY_ID", "gateway-id"),
            ]))
        );
    }

    #[tokio::test]
    async fn stored_credential_merges_with_env_per_field() {
        let ctx = Ctx {
            env: [
                ("CLOUDFLARE_API_KEY", "env-key"),
                ("CLOUDFLARE_ACCOUNT_ID", "env-account"),
                ("CLOUDFLARE_GATEWAY_ID", "env-gateway"),
            ]
            .into(),
        };
        // Key only: account and gateway ids come from the environment.
        let key_only = ApiKeyCredential {
            key: Some("stored-key".into()),
            env: None,
        };
        let result = CloudflareAiGatewayAuth
            .resolve(ApiKeyInput {
                ctx: &ctx,
                credential: Some(&key_only),
            })
            .await
            .unwrap()
            .unwrap();
        let headers = result.auth.headers.unwrap();
        assert_eq!(
            headers.get("cf-aig-authorization"),
            Some(&Some("Bearer stored-key".to_owned()))
        );
        assert_eq!(
            result.env,
            Some(env_of(&[
                ("CLOUDFLARE_ACCOUNT_ID", "env-account"),
                ("CLOUDFLARE_GATEWAY_ID", "env-gateway"),
            ]))
        );
        assert_eq!(result.source.as_deref(), Some("stored credential"));

        // Stored env values win over ambient ones; a stored env without a key uses the env key.
        let env_only = ApiKeyCredential {
            key: None,
            env: Some(env_of(&[("CLOUDFLARE_ACCOUNT_ID", "stored-account")])),
        };
        let result = CloudflareWorkersAiAuth
            .resolve(ApiKeyInput {
                ctx: &ctx,
                credential: Some(&env_only),
            })
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.auth.api_key.as_deref(), Some("env-key"));
        assert_eq!(result.env, Some(env_of(&[("CLOUDFLARE_ACCOUNT_ID", "stored-account")])));

        // An empty stored key does not fall back to the environment.
        let empty_key = ApiKeyCredential {
            key: Some(String::new()),
            env: None,
        };
        let result = CloudflareWorkersAiAuth
            .resolve(ApiKeyInput {
                ctx: &ctx,
                credential: Some(&empty_key),
            })
            .await
            .unwrap();
        assert!(result.is_none());
    }

    struct Answers {
        answers: Mutex<Vec<&'static str>>,
        prompts: Mutex<Vec<String>>,
    }

    impl Answers {
        fn new(answers: &[&'static str]) -> Self {
            Self {
                answers: Mutex::new(answers.iter().rev().copied().collect()),
                prompts: Mutex::default(),
            }
        }
    }

    impl AuthInteraction for Answers {
        fn prompt(&self, prompt: AuthPrompt) -> BoxFuture<'_, Result<String, ModelsError>> {
            let label = match prompt {
                AuthPrompt::Text { message, .. } => format!("text: {message}"),
                AuthPrompt::Secret { message, .. } => format!("secret: {message}"),
                other => format!("{other:?}"),
            };
            self.prompts.lock().unwrap().push(label);
            let answer = self.answers.lock().unwrap().pop().unwrap_or_default();
            Box::pin(async move { Ok(answer.to_owned()) })
        }

        fn notify(&self, _event: AuthEvent) {}
    }

    #[tokio::test]
    async fn login_prompts_for_key_and_ids() {
        let interaction = Answers::new(&["key", "account"]);
        let credential = CloudflareWorkersAiAuth.login(&interaction).await.unwrap();
        assert_eq!(
            credential,
            ApiKeyCredential {
                key: Some("key".into()),
                env: Some(env_of(&[("CLOUDFLARE_ACCOUNT_ID", "account")])),
            }
        );
        assert_eq!(
            *interaction.prompts.lock().unwrap(),
            ["secret: Enter Cloudflare API key", "text: Enter Cloudflare account ID"]
        );

        let interaction = Answers::new(&["key", "account", "gateway"]);
        let credential = CloudflareAiGatewayAuth.login(&interaction).await.unwrap();
        assert_eq!(
            credential,
            ApiKeyCredential {
                key: Some("key".into()),
                env: Some(env_of(&[
                    ("CLOUDFLARE_ACCOUNT_ID", "account"),
                    ("CLOUDFLARE_GATEWAY_ID", "gateway"),
                ])),
            }
        );
        assert_eq!(
            interaction.prompts.lock().unwrap().last().map(String::as_str),
            Some("text: Enter Cloudflare AI Gateway ID")
        );
    }

    /// Records the base URL each call receives.
    #[derive(Clone, Default)]
    struct Capture {
        urls: Arc<Mutex<Vec<String>>>,
    }

    impl ProviderStreams for Capture {
        fn stream(&self, model: &Model, _: &TranscriptContext, _: StreamOptions) -> AssistantMessageEventStream {
            self.urls.lock().unwrap().push(model.base_url.clone());
            AssistantMessageEventStream::channel().1
        }

        fn stream_simple(
            &self,
            model: &Model,
            _: &TranscriptContext,
            _: SimpleStreamOptions,
        ) -> AssistantMessageEventStream {
            self.urls.lock().unwrap().push(model.base_url.clone());
            AssistantMessageEventStream::channel().1
        }
    }

    fn gateway_model() -> Model {
        serde_json::from_value(serde_json::json!({
            "id": "model", "name": "model", "api": "openai-completions", "provider": "cloudflare-ai-gateway",
            "baseUrl": crate::api::cloudflare::CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL,
            "reasoning": false, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 1000, "maxTokens": 100
        }))
        .unwrap()
    }

    #[test]
    fn materializes_the_model_endpoint_before_dispatch() {
        let capture = Capture::default();
        let streams = cloudflare_streams(capture.clone());
        let model = gateway_model();
        let context = normalize_context(&Context::new(Vec::new()));
        let env = env_of(&[
            ("CLOUDFLARE_ACCOUNT_ID", "account"),
            ("CLOUDFLARE_GATEWAY_ID", "gateway"),
        ]);

        let options = StreamOptions {
            env: Some(env),
            ..StreamOptions::default()
        };
        drop(streams.stream(&model, &context, options.clone()));
        drop(streams.stream_simple(
            &model,
            &context,
            SimpleStreamOptions {
                base: options,
                ..SimpleStreamOptions::default()
            },
        ));

        assert_eq!(
            *capture.urls.lock().unwrap(),
            [
                "https://gateway.ai.cloudflare.com/v1/account/gateway/openai",
                "https://gateway.ai.cloudflare.com/v1/account/gateway/openai",
            ]
        );
    }

    #[test]
    fn keeps_placeholders_when_the_provider_env_does_not_resolve_them() {
        let capture = Capture::default();
        let streams = cloudflare_streams(capture.clone());
        let model = gateway_model();
        let context = normalize_context(&Context::new(Vec::new()));

        drop(streams.stream_simple(&model, &context, SimpleStreamOptions::default()));
        let options = StreamOptions {
            env: Some(env_of(&[("CLOUDFLARE_ACCOUNT_ID", "account")])),
            ..StreamOptions::default()
        };
        drop(streams.stream(&model, &context, options));

        assert_eq!(
            *capture.urls.lock().unwrap(),
            [
                model.base_url.as_str(),
                "https://gateway.ai.cloudflare.com/v1/account/{CLOUDFLARE_GATEWAY_ID}/openai",
            ]
        );
    }

    #[test]
    fn providers_serve_their_catalogs() {
        let gateway = cloudflare_ai_gateway_provider();
        assert_eq!(gateway.id(), "cloudflare-ai-gateway");
        assert_eq!(gateway.name(), "Cloudflare AI Gateway");
        let apis: std::collections::BTreeSet<_> = gateway.models().into_iter().map(|model| model.api).collect();
        assert_eq!(
            apis.into_iter().collect::<Vec<_>>(),
            ["anthropic-messages", "openai-completions", "openai-responses"]
        );

        let workers = cloudflare_workers_ai_provider();
        assert_eq!(workers.name(), "Cloudflare Workers AI");
        assert!(workers.models().iter().all(|model| model.api == "openai-completions"));
        assert!(
            workers
                .all_models()
                .iter()
                .any(|model| model.model_type() == ModelType::Classifier && model.id() == "@cf/cloudflare/clef")
        );
    }
}
