//! The `github-copilot` provider. Port of upstream `providers/github-copilot.ts`.
//!
//! Claude models use `anthropic-messages`, the others `openai-completions` or
//! `openai-responses`; each API adds Copilot's dynamic headers
//! ([`crate::api::github_copilot_headers`]). Auth is a Copilot token from
//! `COPILOT_GITHUB_TOKEN` or the GitHub device-code login ([`GitHubCopilotOAuth`]), whose
//! credential carries the account's API base URL and model list.

use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::api::anthropic_messages::AnthropicMessagesApi;
use crate::api::openai_completions::OpenAICompletionsApi;
use crate::api::openai_responses::OpenAIResponsesApi;
use crate::auth::{Credential, ProviderAuth, env_api_key_auth};
use crate::model::Model;
use crate::oauth::GitHubCopilotOAuth;
use crate::oauth::github_copilot::INDIVIDUAL_BASE_URL;
use crate::provider::{BuiltProvider, CreateProviderOptions, ProviderApi, ProviderStreams, create_provider};
use crate::providers::catalog::all_builtin_models;

/// Keep only the models in an OAuth credential's `availableModelIds`. API-key credentials,
/// no credential, or a missing or malformed list keep every model.
pub fn filter_github_copilot_models(models: Vec<Model>, credential: Option<&Credential>) -> Vec<Model> {
    let Some(Credential::OAuth(credential)) = credential else {
        return models;
    };
    let Some(Value::Array(ids)) = credential.extra.get("availableModelIds") else {
        return models;
    };
    let Some(available) = ids.iter().map(Value::as_str).collect::<Option<HashSet<&str>>>() else {
        return models;
    };
    models
        .into_iter()
        .filter(|model| available.contains(model.id.as_str()))
        .collect()
}

/// GitHub Copilot: `COPILOT_GITHUB_TOKEN` or OAuth login with a Copilot subscription.
pub fn github_copilot_provider() -> BuiltProvider {
    let auth = ProviderAuth::api_key(env_api_key_auth("GitHub Copilot token", &["COPILOT_GITHUB_TOKEN"]))
        .with_oauth(GitHubCopilotOAuth::default());
    let apis: HashMap<String, Arc<dyn ProviderStreams>> = [
        (
            "anthropic-messages",
            Arc::new(AnthropicMessagesApi) as Arc<dyn ProviderStreams>,
        ),
        ("openai-completions", Arc::new(OpenAICompletionsApi)),
        ("openai-responses", Arc::new(OpenAIResponsesApi)),
    ]
    .into_iter()
    .map(|(api, streams)| (api.to_owned(), streams))
    .collect();
    let mut options = CreateProviderOptions::new(
        "github-copilot",
        auth,
        all_builtin_models("github-copilot").to_vec(),
        ProviderApi::ByApi(apis),
    )
    .name("GitHub Copilot")
    .base_url(INDIVIDUAL_BASE_URL);
    options.filter_models = Some(Arc::new(filter_github_copilot_models));
    create_provider(options)
}
