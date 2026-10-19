//! The Radius gateway provider. Port of upstream `providers/radius.ts` and
//! `providers/radius-config.ts`.
//!
//! Radius speaks the `pi-messages` API ([`crate::api::pi_messages`]). The provider ships the
//! public `radius.pi.dev` catalog for synchronous lookups and replaces it with the gateway's
//! `/v1/config` catalog once a refresh with configured auth has run. Custom gateways start
//! with no models.

use futures::future::BoxFuture;
use serde_json::Value;
use std::sync::{Arc, PoisonError, RwLock};
use tokio_util::sync::CancellationToken;

use crate::api::pi_messages::PiMessagesApi;
use crate::auth::{Credential, OAuthCredential, ProviderAuth, env_api_key_auth};
use crate::error::{ModelsError, ModelsErrorCode};
use crate::events::AssistantMessageEventStream;
use crate::model::{AnyModel, Model};
use crate::oauth::radius::{RadiusOAuthOptions, create_radius_oauth};
use crate::options::{SimpleStreamOptions, StreamOptions};
use crate::provider::{ModelsPublication, ModelsStoreEntry, Persist, Provider, ProviderStreams, RefreshModelsContext};
use crate::providers::catalog::builtin_models;
use crate::types::{JsonObject, TranscriptContext, now_ms};

/// The public Radius gateway.
pub const DEFAULT_RADIUS_GATEWAY: &str = "https://radius.pi.dev";
/// Longest error body kept in config errors, in characters.
const MAX_HTTP_BODY_CHARS: usize = 512;

// ---------------------------------------------------------------------------------------------
// Gateway config (radius-config.ts)
// ---------------------------------------------------------------------------------------------

/// The catalog a gateway serves at `/v1/config` (upstream `RadiusGatewayConfig`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RadiusGatewayConfig {
    /// Base URL of the gateway's `pi-messages` endpoint.
    pub base_url: String,
    /// Model entries (`id`, `name`, `reasoning`, `thinkingLevelMap`, `input`, `cost`,
    /// `contextWindow`, `maxTokens`, ...) as the gateway sent them. Invalid entries are
    /// dropped when the config is read.
    pub models: Vec<JsonObject>,
}

/// Upstream `isRadiusGatewayModel`.
fn is_radius_gateway_model(value: &Value) -> bool {
    let Value::Object(model) = value else { return false };
    model.get("id").is_some_and(Value::is_string)
        && model.get("name").is_some_and(Value::is_string)
        && model.get("reasoning").is_some_and(Value::is_boolean)
        && model.get("input").is_some_and(Value::is_array)
        && model.get("cost").is_some_and(Value::is_object)
        && model.get("contextWindow").is_some_and(Value::is_number)
        && model.get("maxTokens").is_some_and(Value::is_number)
}

/// Upstream `sanitizeRadiusGatewayConfig`: `None` unless `baseUrl` is a string and `models`
/// an array; invalid model entries are dropped.
fn sanitize_radius_gateway_config(config: &Value) -> Option<RadiusGatewayConfig> {
    let Value::Object(config) = config else { return None };
    let base_url = config.get("baseUrl")?.as_str()?;
    let models = config.get("models")?.as_array()?;
    Some(RadiusGatewayConfig {
        base_url: base_url.to_owned(),
        models: models
            .iter()
            .filter(|model| is_radius_gateway_model(model))
            .filter_map(|model| model.as_object().cloned())
            .collect(),
    })
}

/// Add `https://` when the value has no `http(s)://` scheme and remove trailing slashes.
pub fn normalize_radius_gateway_url(value: &str) -> String {
    let has_scheme = ["http://", "https://"].iter().any(|scheme| {
        value
            .get(..scheme.len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(scheme))
    });
    let with_scheme = if has_scheme {
        value.to_owned()
    } else {
        format!("https://{value}")
    };
    with_scheme.trim_end_matches('/').to_owned()
}

/// The gateway config cached in an OAuth credential by the implementation that predates the
/// models store (`gatewayConfig`).
pub fn get_radius_credential_config(credential: Option<&OAuthCredential>) -> Option<RadiusGatewayConfig> {
    sanitize_radius_gateway_config(credential?.extra.get("gatewayConfig")?)
}

/// The config's models as `pi-messages` models of `provider_id`. Entries that do not form a
/// valid model are skipped.
pub fn get_radius_models_from_config(provider_id: &str, config: &RadiusGatewayConfig) -> Vec<Model> {
    config
        .models
        .iter()
        .filter_map(|model| {
            let mut model = model.clone();
            model.insert("api".into(), "pi-messages".into());
            model.insert("provider".into(), provider_id.into());
            model.insert("baseUrl".into(), config.base_url.clone().into());
            match serde_json::from_value::<Model>(Value::Object(model)) {
                Ok(model) => Some(model),
                Err(error) => {
                    tracing::warn!(provider = provider_id, %error, "skipping invalid Radius gateway model");
                    None
                }
            }
        })
        .collect()
}

/// The models of the gateway config cached in an OAuth credential.
pub fn get_radius_models(provider_id: &str, credential: Option<&OAuthCredential>) -> Vec<Model> {
    get_radius_credential_config(credential)
        .map(|config| get_radius_models_from_config(provider_id, &config))
        .unwrap_or_default()
}

/// Trimmed, at most 512 characters plus `…`.
fn truncate_http_body(body: &str) -> String {
    let trimmed = crate::http::js_trim(body);
    match trimmed.char_indices().nth(MAX_HTTP_BODY_CHARS) {
        Some((end, _)) => format!("{}…", &trimmed[..end]),
        None => trimmed.to_owned(),
    }
}

fn source_error(message: impl Into<String>) -> ModelsError {
    ModelsError::new(ModelsErrorCode::ModelSource, message)
}

/// `GET <gateway>/v1/config`, with `Authorization: Bearer <api_key>` when a key is given.
pub async fn load_radius_gateway_config(
    gateway: &str,
    api_key: Option<&str>,
    cancel: &CancellationToken,
) -> Result<RadiusGatewayConfig, ModelsError> {
    let url = url::Url::parse(gateway)
        .and_then(|base| base.join("/v1/config"))
        .map_err(|error| source_error(format!("Invalid URL: {gateway} ({error})")))?;
    let mut request = crate::http::client(&StreamOptions::default())
        .get(url)
        .header("accept", "application/json");
    if let Some(api_key) = api_key.filter(|key| !key.is_empty()) {
        request = request.header("authorization", format!("Bearer {api_key}"));
    }
    let load = async {
        let response = request
            .send()
            .await
            .map_err(|error| ModelsError::with_source(ModelsErrorCode::ModelSource, "Request failed", error))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|error| ModelsError::with_source(ModelsErrorCode::ModelSource, "Request failed", error))?;
        if !status.is_success() {
            return Err(source_error(format!(
                "Could not load Radius config from {gateway}: {}: {}",
                status.as_u16(),
                truncate_http_body(&text)
            )));
        }
        let value: Value = serde_json::from_str(&text).map_err(|error| source_error(error.to_string()))?;
        sanitize_radius_gateway_config(&value)
            .ok_or_else(|| source_error(format!("Invalid Radius config from {gateway}")))
    };
    if cancel.is_cancelled() {
        return Err(ModelsError::aborted());
    }
    tokio::select! {
        () = cancel.cancelled() => Err(ModelsError::aborted()),
        result = load => result,
    }
}

// ---------------------------------------------------------------------------------------------
// Provider (radius.ts)
// ---------------------------------------------------------------------------------------------

/// Options of [`radius_provider`] (upstream `RadiusProviderOptions`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RadiusProviderOptions {
    /// Provider id. Default `radius`.
    pub id: Option<String>,
    /// Display name. Default `Radius`.
    pub name: Option<String>,
    /// Gateway URL. Default [`DEFAULT_RADIUS_GATEWAY`].
    pub gateway: Option<String>,
}

/// Radius gateway provider with a persisted, dynamically refreshed catalog.
pub struct RadiusProvider {
    id: String,
    name: String,
    gateway: String,
    auth: ProviderAuth,
    /// The shipped catalog; used until a gateway catalog is known.
    baseline: Vec<Model>,
    /// Gateway catalog for this account. Radius organization owners can disable models, so
    /// once known it replaces the baseline instead of overlaying it.
    dynamic: Arc<RwLock<Option<Vec<Model>>>>,
}

impl std::fmt::Debug for RadiusProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RadiusProvider")
            .field("id", &self.id)
            .field("gateway", &self.gateway)
            .finish_non_exhaustive()
    }
}

/// A Radius provider. Only the default gateway gets the shipped catalog as its baseline.
pub fn radius_provider(options: RadiusProviderOptions) -> RadiusProvider {
    let id = options.id.unwrap_or_else(|| "radius".to_owned());
    let name = options.name.unwrap_or_else(|| "Radius".to_owned());
    let gateway = normalize_radius_gateway_url(options.gateway.as_deref().unwrap_or(DEFAULT_RADIUS_GATEWAY));
    let baseline = if gateway == normalize_radius_gateway_url(DEFAULT_RADIUS_GATEWAY) {
        builtin_models("radius")
            .into_iter()
            .map(|mut model| {
                model.provider.clone_from(&id);
                model
            })
            .collect()
    } else {
        Vec::new()
    };
    let auth = ProviderAuth::api_key(env_api_key_auth("Radius API key", &["RADIUS_API_KEY"])).with_oauth(
        create_radius_oauth(RadiusOAuthOptions {
            name: name.clone(),
            gateway: gateway.clone(),
        }),
    );
    RadiusProvider {
        id,
        name,
        gateway,
        auth,
        baseline,
        dynamic: Arc::default(),
    }
}

impl RadiusProvider {
    /// The normalized gateway URL.
    pub fn gateway(&self) -> &str {
        &self.gateway
    }

    /// Replace the baseline catalog, the models served before a gateway catalog is known.
    /// Upstream derives it from the gateway only; this lets tests and private deployments
    /// pair a custom gateway with a shipped list.
    #[must_use]
    pub fn with_baseline_models(mut self, models: Vec<Model>) -> Self {
        self.baseline = models;
        self
    }

    fn set_dynamic(&self, models: Vec<Model>) -> Box<dyn FnOnce() + Send> {
        let dynamic = self.dynamic.clone();
        Box::new(move || {
            *dynamic.write().unwrap_or_else(PoisonError::into_inner) = Some(models);
        })
    }

    async fn refresh(&self, context: RefreshModelsContext) -> Result<(), ModelsError> {
        let stored = context.stored.as_ref();
        if let Some(stored) = stored {
            let restored: Vec<Model> = stored
                .models
                .iter()
                .filter_map(AnyModel::as_chat)
                .filter(|model| model.provider == self.id)
                .cloned()
                .collect();
            let published = (context.publish)(ModelsPublication {
                persist: Persist::Keep,
                update: Some(self.set_dynamic(restored)),
            })
            .await;
            if !published {
                return Ok(());
            }
        }

        // Import catalogs cached by the implementation that predates the models store.
        if stored.is_none()
            && let Some(Credential::OAuth(credential)) = &context.credential
        {
            let legacy = get_radius_models(&self.id, Some(credential));
            if !legacy.is_empty() {
                let published = (context.publish)(ModelsPublication {
                    persist: Persist::Write(store_entry(&legacy)),
                    update: Some(self.set_dynamic(legacy)),
                })
                .await;
                if !published {
                    return Ok(());
                }
            }
        }

        if !context.allow_network || context.cancel.is_cancelled() {
            return Ok(());
        }
        let api_key = match &context.credential {
            Some(Credential::OAuth(credential)) => Some(credential.access.as_str()),
            Some(Credential::ApiKey(credential)) => credential.key.as_deref(),
            None => None,
        };
        let config = load_radius_gateway_config(&self.gateway, api_key, &context.cancel).await?;
        if context.cancel.is_cancelled() {
            return Ok(());
        }
        let refreshed = get_radius_models_from_config(&self.id, &config);
        (context.publish)(ModelsPublication {
            persist: Persist::Write(store_entry(&refreshed)),
            update: Some(self.set_dynamic(refreshed)),
        })
        .await;
        Ok(())
    }
}

fn store_entry(models: &[Model]) -> ModelsStoreEntry {
    ModelsStoreEntry {
        models: models.iter().cloned().map(AnyModel::Chat).collect(),
        checked_at: Some(now_ms()),
        ..ModelsStoreEntry::default()
    }
}

impl Provider for RadiusProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn auth(&self) -> &ProviderAuth {
        &self.auth
    }

    fn models(&self) -> Vec<Model> {
        self.dynamic
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
            .unwrap_or_else(|| self.baseline.clone())
    }

    fn is_dynamic(&self) -> bool {
        true
    }

    fn refresh_models(&self, context: RefreshModelsContext) -> BoxFuture<'_, Result<(), ModelsError>> {
        Box::pin(self.refresh(context))
    }

    fn stream(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        PiMessagesApi.stream(model, context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        PiMessagesApi.stream_simple(model, context, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalizes_gateway_urls() {
        assert_eq!(
            normalize_radius_gateway_url("radius.example//"),
            "https://radius.example"
        );
        assert_eq!(
            normalize_radius_gateway_url("HTTP://localhost:8788/"),
            "HTTP://localhost:8788"
        );
        assert_eq!(
            normalize_radius_gateway_url(DEFAULT_RADIUS_GATEWAY),
            DEFAULT_RADIUS_GATEWAY
        );
    }

    #[test]
    fn sanitizes_gateway_configs() {
        assert_eq!(sanitize_radius_gateway_config(&json!({"models": []})), None);
        assert_eq!(
            sanitize_radius_gateway_config(&json!({"baseUrl": "x", "models": {}})),
            None
        );
        let config = sanitize_radius_gateway_config(&json!({
            "baseUrl": "https://radius.example/v1",
            "models": [
                {"id": "ok", "name": "Ok", "reasoning": false, "input": ["text"],
                 "cost": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0},
                 "contextWindow": 1000, "maxTokens": 100, "lab": "Example"},
                {"id": "no-cost", "name": "No cost", "reasoning": false, "input": ["text"],
                 "contextWindow": 1000, "maxTokens": 100},
                "not a model"
            ]
        }))
        .unwrap();
        assert_eq!(config.models.len(), 1);
        let models = get_radius_models_from_config("radius-dev", &config);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].api, "pi-messages");
        assert_eq!(models[0].provider, "radius-dev");
        assert_eq!(models[0].base_url, "https://radius.example/v1");
    }

    #[test]
    fn reads_legacy_credential_configs() {
        let credential: OAuthCredential = serde_json::from_value(json!({
            "access": "a", "refresh": "r", "expires": 0,
            "gatewayConfig": {"baseUrl": "https://radius.example/v1", "models": [
                {"id": "m", "name": "M", "reasoning": true, "input": ["text", "image"],
                 "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                 "contextWindow": 1, "maxTokens": 1}
            ]}
        }))
        .unwrap();
        let models = get_radius_models("radius", Some(&credential));
        assert_eq!(models.len(), 1);
        assert!(models[0].reasoning);
        assert!(get_radius_models("radius", None).is_empty());
    }

    #[test]
    fn truncates_error_bodies() {
        assert_eq!(truncate_http_body("  short \n"), "short");
        let long = "x".repeat(600);
        assert_eq!(truncate_http_body(&long), format!("{}…", "x".repeat(512)));
    }
}
