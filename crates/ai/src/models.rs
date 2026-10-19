//! The [`Models`] collection: registered providers plus auth application and request
//! convenience. Providers own request behavior; `Models` resolves auth and delegates each
//! request to the provider that owns the model.

use futures::future::{BoxFuture, join_all};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use tokio_util::sync::CancellationToken;

use crate::auth::{
    AuthCheck, AuthContext, AuthInteraction, AuthResolutionOverrides, AuthResult, AuthType, Credential,
    CredentialStore, DefaultAuthContext, InMemoryCredentialStore, LoginOptions, ModifyFn, merge_headers,
    read_credential, refresh_stored_oauth_credential, resolve_provider_auth,
};
use crate::classifier::{ClassifierContext, ClassifierOptions, ClassifierResult, check_classifier_input};
use crate::deferred::{DeferredCancelOptions, DeferredFetchOptions};
use crate::error::{Error, ModelsError, ModelsErrorCode};
use crate::events::AssistantMessageEventStream;
use crate::images::{AssistantImages, ImagesContext, ImagesOptions};
use crate::model::{AnyModel, ClassifierModel, ImageModel, Model, ModelType};
use crate::options::{RequestOptions, SimpleStreamOptions, StreamOptions, TransformHeaders};
use crate::provider::{ModelsPublication, ModelsStoreEntry, Persist, Provider, RefreshModelsContext, lazy_stream};
use crate::transcript::normalize_context;
use crate::types::{AssistantMessage, Context, DeferredHandle, ProviderEnv, ProviderHeaders, now_ms};

/// Persistent dynamic model catalogs keyed by provider id.
pub trait ModelsStore: Send + Sync {
    fn read<'a>(&'a self, provider_id: &'a str) -> BoxFuture<'a, Result<Option<ModelsStoreEntry>, ModelsError>>;
    fn write<'a>(&'a self, provider_id: &'a str, entry: ModelsStoreEntry) -> BoxFuture<'a, Result<(), ModelsError>>;
    fn delete<'a>(&'a self, provider_id: &'a str) -> BoxFuture<'a, Result<(), ModelsError>>;
}

/// Default in-memory [`ModelsStore`].
#[derive(Debug, Default)]
pub struct InMemoryModelsStore {
    entries: Mutex<HashMap<String, ModelsStoreEntry>>,
}

impl ModelsStore for InMemoryModelsStore {
    fn read<'a>(&'a self, provider_id: &'a str) -> BoxFuture<'a, Result<Option<ModelsStoreEntry>, ModelsError>> {
        Box::pin(async move {
            Ok(self
                .entries
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(provider_id)
                .cloned())
        })
    }

    fn write<'a>(&'a self, provider_id: &'a str, entry: ModelsStoreEntry) -> BoxFuture<'a, Result<(), ModelsError>> {
        Box::pin(async move {
            self.entries
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(provider_id.to_owned(), entry);
            Ok(())
        })
    }

    fn delete<'a>(&'a self, provider_id: &'a str) -> BoxFuture<'a, Result<(), ModelsError>> {
        Box::pin(async move {
            self.entries
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(provider_id);
            Ok(())
        })
    }
}

/// Options of [`Models::new`].
#[derive(Clone, Default)]
pub struct CreateModelsOptions {
    pub credentials: Option<Arc<dyn CredentialStore>>,
    pub models_store: Option<Arc<dyn ModelsStore>>,
    pub auth_context: Option<Arc<dyn AuthContext>>,
}

impl fmt::Debug for CreateModelsOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CreateModelsOptions").finish_non_exhaustive()
    }
}

/// Options of [`Models::refresh`].
#[derive(Clone, Debug, Default)]
pub struct ModelsRefreshOptions {
    /// Default: true. False restores persisted catalogs without network access.
    pub allow_network: Option<bool>,
    /// Restrict refresh to these provider ids.
    pub providers: Option<Vec<String>>,
    /// Bypass provider freshness checks.
    pub force: bool,
    pub cancel: Option<CancellationToken>,
}

/// Result of [`Models::refresh`]. Provider errors are returned, not raised.
#[derive(Debug, Default)]
pub struct ModelsRefreshResult {
    pub aborted: bool,
    pub errors: HashMap<String, ModelsError>,
}

#[derive(Default)]
struct RefreshState {
    generations: HashMap<String, u64>,
    controllers: HashMap<String, (u64, CancellationToken)>,
    publication_locks: HashMap<String, Arc<tokio::sync::Mutex<()>>>,
}

struct Inner {
    providers: RwLock<Vec<Arc<dyn Provider>>>,
    credentials: Arc<dyn CredentialStore>,
    models_store: Arc<dyn ModelsStore>,
    auth_context: Arc<dyn AuthContext>,
    refresh: Mutex<RefreshState>,
}

/// Runtime collection of providers. Cheap to clone; clones share state.
#[derive(Clone)]
pub struct Models {
    inner: Arc<Inner>,
}

impl fmt::Debug for Models {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ids: Vec<String> = self
            .providers()
            .iter()
            .map(|provider| provider.id().to_owned())
            .collect();
        f.debug_struct("Models")
            .field("providers", &ids)
            .finish_non_exhaustive()
    }
}

impl Default for Models {
    fn default() -> Self {
        Self::new(CreateModelsOptions::default())
    }
}

impl Models {
    pub fn new(options: CreateModelsOptions) -> Self {
        Self {
            inner: Arc::new(Inner {
                providers: RwLock::default(),
                credentials: options
                    .credentials
                    .unwrap_or_else(|| Arc::new(InMemoryCredentialStore::new())),
                models_store: options
                    .models_store
                    .unwrap_or_else(|| Arc::new(InMemoryModelsStore::default())),
                auth_context: options.auth_context.unwrap_or_else(|| Arc::new(DefaultAuthContext)),
                refresh: Mutex::default(),
            }),
        }
    }

    /// The credential store.
    pub fn credentials(&self) -> &Arc<dyn CredentialStore> {
        &self.inner.credentials
    }

    /// Upsert by provider id.
    pub fn set_provider(&self, provider: impl Provider + 'static) {
        self.set_provider_arc(Arc::new(provider));
    }

    /// Upsert a shared provider by id.
    pub fn set_provider_arc(&self, provider: Arc<dyn Provider>) {
        self.supersede_refresh(provider.id());
        let mut providers = self.inner.providers.write().unwrap_or_else(PoisonError::into_inner);
        match providers.iter_mut().find(|existing| existing.id() == provider.id()) {
            Some(existing) => *existing = provider,
            None => providers.push(provider),
        }
    }

    pub fn delete_provider(&self, id: &str) {
        self.supersede_refresh(id);
        self.inner
            .providers
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .retain(|provider| provider.id() != id);
    }

    pub fn clear_providers(&self) {
        let ids: Vec<String> = self
            .providers()
            .iter()
            .map(|provider| provider.id().to_owned())
            .collect();
        for id in ids {
            self.delete_provider(&id);
        }
    }

    pub fn providers(&self) -> Vec<Arc<dyn Provider>> {
        self.inner
            .providers
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    pub fn provider(&self, id: &str) -> Option<Arc<dyn Provider>> {
        self.inner
            .providers
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .find(|provider| provider.id() == id)
            .cloned()
    }

    fn selected(&self, provider: Option<&str>) -> Vec<Arc<dyn Provider>> {
        match provider {
            Some(id) => self.provider(id).into_iter().collect(),
            None => self.providers(),
        }
    }

    /// Last-known chat models of one provider or all.
    pub fn get_models(&self, provider: Option<&str>) -> Vec<Model> {
        self.selected(provider)
            .iter()
            .flat_map(|provider| provider.models())
            .collect()
    }

    /// Chat model lookup against last-known lists.
    pub fn get_model(&self, provider: &str, id: &str) -> Option<Model> {
        self.provider(provider)?
            .models()
            .into_iter()
            .find(|model| model.id == id)
    }

    /// Last-known models of every type.
    pub fn get_all_models(&self, provider: Option<&str>) -> Vec<AnyModel> {
        self.selected(provider)
            .iter()
            .flat_map(|provider| provider.all_models())
            .collect()
    }

    /// Last-known models of one type.
    pub fn get_models_of_type(&self, model_type: ModelType, provider: Option<&str>) -> Vec<AnyModel> {
        self.get_all_models(provider)
            .into_iter()
            .filter(|model| model.model_type() == model_type)
            .collect()
    }

    /// Lookup of a model of one type.
    pub fn get_model_of_type(&self, model_type: ModelType, provider: &str, id: &str) -> Option<AnyModel> {
        self.get_models_of_type(model_type, Some(provider))
            .into_iter()
            .find(|model| model.id() == id)
    }

    fn supersede_refresh(&self, provider_id: &str) -> u64 {
        let mut state = self.inner.refresh.lock().unwrap_or_else(PoisonError::into_inner);
        let generation = state.generations.entry(provider_id.to_owned()).or_default();
        *generation += 1;
        let generation = *generation;
        if let Some((_, previous)) = state.controllers.remove(provider_id) {
            previous.cancel();
        }
        generation
    }

    fn begin_refresh(&self, provider_id: &str, parent: &CancellationToken) -> (u64, CancellationToken) {
        let generation = self.supersede_refresh(provider_id);
        let token = parent.child_token();
        self.inner
            .refresh
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .controllers
            .insert(provider_id.to_owned(), (generation, token.clone()));
        (generation, token)
    }

    fn is_current(&self, provider_id: &str, generation: u64) -> bool {
        self.inner
            .refresh
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .generations
            .get(provider_id)
            == Some(&generation)
    }

    fn publisher(&self, provider_id: &str, generation: u64, cancel: CancellationToken) -> crate::provider::PublishFn {
        let models = self.clone();
        let provider_id = provider_id.to_owned();
        Arc::new(move |publication: ModelsPublication| {
            let models = models.clone();
            let provider_id = provider_id.clone();
            let cancel = cancel.clone();
            Box::pin(async move {
                let lock = models
                    .inner
                    .refresh
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .publication_locks
                    .entry(provider_id.clone())
                    .or_default()
                    .clone();
                let _guard = lock.lock().await;
                if cancel.is_cancelled() || !models.is_current(&provider_id, generation) {
                    return false;
                }
                let store = &models.inner.models_store;
                let persisted = match publication.persist {
                    Persist::Keep => Ok(()),
                    Persist::Write(entry) => store.write(&provider_id, entry).await,
                    Persist::Delete => store.delete(&provider_id).await,
                };
                if persisted.is_err() || cancel.is_cancelled() || !models.is_current(&provider_id, generation) {
                    return false;
                }
                if let Some(update) = publication.update {
                    update();
                }
                true
            })
        })
    }

    /// Refresh configured dynamic providers concurrently (all when `providers` is unset).
    /// Errors and cancellation are returned; static, unknown and unconfigured providers are
    /// skipped.
    pub async fn refresh(&self, options: ModelsRefreshOptions) -> ModelsRefreshResult {
        let allow_network = options.allow_network.unwrap_or(true);
        let caller = options.cancel.clone().unwrap_or_default();
        if caller.is_cancelled() {
            return ModelsRefreshResult {
                aborted: true,
                errors: HashMap::new(),
            };
        }
        let refreshable: Vec<Arc<dyn Provider>> = self
            .providers()
            .into_iter()
            .filter(|provider| provider.is_dynamic())
            .filter(|provider| {
                options
                    .providers
                    .as_ref()
                    .is_none_or(|ids| ids.iter().any(|id| id == provider.id()))
            })
            .collect();
        let tasks = refreshable.into_iter().map(|provider| {
            let models = self.clone();
            let caller = caller.clone();
            let force = options.force;
            async move {
                let id = provider.id().to_owned();
                let (generation, token) = models.begin_refresh(&id, &caller);
                let operation = models.refresh_provider(provider, allow_network, force, generation, token.clone());
                let result = tokio::select! {
                    () = token.cancelled() => Ok(()),
                    result = operation => result,
                };
                {
                    let mut state = models.inner.refresh.lock().unwrap_or_else(PoisonError::into_inner);
                    if state
                        .controllers
                        .get(&id)
                        .is_some_and(|(current, _)| *current == generation)
                    {
                        state.controllers.remove(&id);
                    }
                }
                (id, result.err().filter(|_| !token.is_cancelled()))
            }
        });
        let results = tokio::select! {
            () = caller.cancelled() => Vec::new(),
            results = join_all(tasks) => results,
        };
        ModelsRefreshResult {
            aborted: caller.is_cancelled(),
            errors: results
                .into_iter()
                .filter_map(|(id, error)| error.map(|error| (id, error)))
                .collect(),
        }
    }

    async fn refresh_provider(
        &self,
        provider: Arc<dyn Provider>,
        allow_network: bool,
        force: bool,
        generation: u64,
        cancel: CancellationToken,
    ) -> Result<(), ModelsError> {
        let id = provider.id().to_owned();
        let stored_credential = read_credential(self.inner.credentials.as_ref(), &id).await;
        let phase = |credential: Option<Credential>, network: bool| {
            let models = self.clone();
            let provider = provider.clone();
            let cancel = cancel.clone();
            let id = id.clone();
            async move {
                let stored = models.inner.models_store.read(&id).await?;
                provider
                    .refresh_models(RefreshModelsContext {
                        credential,
                        stored,
                        publish: models.publisher(&id, generation, cancel.clone()),
                        allow_network: network,
                        force: network && force,
                        cancel,
                        base_url: provider.base_url().map(str::to_owned),
                    })
                    .await
            }
        };
        // Restore cached provider state before auth resolution or network access.
        let stored = stored_credential.as_ref().ok().cloned().flatten();
        phase(stored.clone(), false).await?;
        let stored = stored_credential?;
        if !allow_network || cancel.is_cancelled() {
            return Ok(());
        }
        let Some(credential) = self.resolve_refresh_credential(provider.as_ref(), stored).await? else {
            return Ok(());
        };
        phase(Some(credential), true).await
    }

    async fn resolve_refresh_credential(
        &self,
        provider: &dyn Provider,
        stored: Option<Credential>,
    ) -> Result<Option<Credential>, ModelsError> {
        if let Some(Credential::OAuth(credential)) = &stored {
            let Some(oauth) = provider.auth().oauth.clone() else {
                return Ok(None);
            };
            if now_ms() < credential.expires {
                return Ok(stored);
            }
            let refreshed = refresh_stored_oauth_credential(
                self.inner.credentials.clone(),
                provider.id().to_owned(),
                oauth,
                |credential, _| now_ms() >= credential.expires,
                0,
            )
            .await?;
            return Ok(refreshed.map(Credential::OAuth));
        }
        let Some(api_key) = provider.auth().api_key.clone() else {
            return Ok(None);
        };
        let credential = match &stored {
            Some(Credential::ApiKey(credential)) => Some(credential),
            _ => None,
        };
        let result = api_key
            .resolve(crate::auth::ApiKeyInput {
                ctx: self.inner.auth_context.as_ref(),
                credential,
            })
            .await?;
        Ok(result.map(|result| {
            Credential::ApiKey(crate::auth::ApiKeyCredential {
                key: result.auth.api_key,
                env: result.env,
            })
        }))
    }

    async fn check_provider_auth(
        &self,
        provider: &dyn Provider,
        credential: Option<&Credential>,
    ) -> Result<Option<AuthCheck>, ModelsError> {
        if let Some(Credential::OAuth(_)) = credential {
            return Ok(provider.auth().oauth.as_ref().map(|_| AuthCheck {
                source: Some("OAuth".into()),
                auth_type: AuthType::OAuth,
            }));
        }
        let Some(api_key) = provider.auth().api_key.clone() else {
            return Ok(None);
        };
        let api_credential = match credential {
            Some(Credential::ApiKey(credential)) => Some(credential),
            _ => None,
        };
        let input = crate::auth::ApiKeyInput {
            ctx: self.inner.auth_context.as_ref(),
            credential: api_credential,
        };
        if let Some(check) = api_key.check(input) {
            return check.await.map_err(|error| {
                ModelsError::with_source(
                    ModelsErrorCode::Auth,
                    format!("API key auth check failed for provider {}", provider.id()),
                    error,
                )
            });
        }
        let resolution = resolve_provider_auth(
            provider.id(),
            provider.auth(),
            &self.inner.credentials,
            self.inner.auth_context.as_ref(),
            &AuthResolutionOverrides::default(),
        )
        .await?;
        Ok(resolution.map(|resolution| AuthCheck {
            source: resolution.source,
            auth_type: AuthType::ApiKey,
        }))
    }

    /// Whether a provider has complete auth configuration, without refreshing OAuth.
    pub async fn check_auth(&self, provider_id: &str) -> Result<Option<AuthCheck>, ModelsError> {
        let Some(provider) = self.provider(provider_id) else {
            return Ok(None);
        };
        let credential = read_credential(self.inner.credentials.as_ref(), provider_id).await?;
        self.check_provider_auth(provider.as_ref(), credential.as_ref()).await
    }

    async fn authenticated(
        &self,
        provider_id: Option<&str>,
    ) -> Result<Vec<(Arc<dyn Provider>, Option<Credential>)>, ModelsError> {
        let checks = self.selected(provider_id).into_iter().map(|provider| async move {
            let credential = read_credential(self.inner.credentials.as_ref(), provider.id()).await?;
            let check = self.check_provider_auth(provider.as_ref(), credential.as_ref()).await?;
            Ok::<_, ModelsError>(check.map(|_| (provider, credential)))
        });
        let mut available = Vec::new();
        for result in join_all(checks).await {
            if let Some(entry) = result? {
                available.push(entry);
            }
        }
        Ok(available)
    }

    /// Chat models whose providers have complete auth configuration.
    pub async fn get_available(&self, provider_id: Option<&str>) -> Result<Vec<Model>, ModelsError> {
        Ok(self
            .authenticated(provider_id)
            .await?
            .into_iter()
            .flat_map(|(provider, credential)| provider.filter_models(provider.models(), credential.as_ref()))
            .collect())
    }

    /// Models of every type whose providers have complete auth configuration.
    pub async fn get_all_available(&self, provider_id: Option<&str>) -> Result<Vec<AnyModel>, ModelsError> {
        let mut result = Vec::new();
        for (provider, credential) in self.authenticated(provider_id).await? {
            let models = provider.all_models();
            if let Some(filtered) = provider.filter_all_models(&models, credential.as_ref()) {
                result.extend(filtered);
                continue;
            }
            let chat: Vec<String> = provider
                .filter_models(provider.models(), credential.as_ref())
                .into_iter()
                .map(|model| model.id)
                .collect();
            result.extend(
                models
                    .into_iter()
                    .filter(|model| model.model_type() != ModelType::Chat || chat.iter().any(|id| id == model.id())),
            );
        }
        Ok(result)
    }

    /// Models of one type whose providers have complete auth configuration.
    pub async fn get_available_of_type(
        &self,
        model_type: ModelType,
        provider_id: Option<&str>,
    ) -> Result<Vec<AnyModel>, ModelsError> {
        Ok(self
            .get_all_available(provider_id)
            .await?
            .into_iter()
            .filter(|model| model.model_type() == model_type)
            .collect())
    }

    /// Resolve provider-scoped auth. `Ok(None)` when the provider is unknown or unconfigured.
    pub async fn get_auth(
        &self,
        provider_id: &str,
        overrides: &AuthResolutionOverrides,
    ) -> Result<Option<AuthResult>, ModelsError> {
        let Some(provider) = self.provider(provider_id) else {
            return Ok(None);
        };
        resolve_provider_auth(
            provider.id(),
            provider.auth(),
            &self.inner.credentials,
            self.inner.auth_context.as_ref(),
            overrides,
        )
        .await
    }

    /// Provider auth plus the model's static headers.
    pub async fn get_model_auth(
        &self,
        model: &AnyModel,
        overrides: &AuthResolutionOverrides,
    ) -> Result<Option<AuthResult>, ModelsError> {
        let Some(mut result) = self.get_auth(model.provider(), overrides).await? else {
            return Ok(None);
        };
        if let Some(headers) = model.headers() {
            let model_headers: ProviderHeaders = headers
                .iter()
                .map(|(name, value)| (name.clone(), Some(value.clone())))
                .collect();
            result.auth.headers = merge_headers(result.auth.headers.as_ref(), Some(&model_headers));
        }
        Ok(Some(result))
    }

    /// Run a provider-owned login flow and persist its credential.
    pub async fn login(
        &self,
        provider_id: &str,
        auth_type: AuthType,
        interaction: &dyn AuthInteraction,
        options: &LoginOptions,
    ) -> Result<Credential, ModelsError> {
        let provider = self
            .provider(provider_id)
            .ok_or_else(|| ModelsError::new(ModelsErrorCode::Provider, format!("Unknown provider: {provider_id}")))?;
        let unsupported = || {
            ModelsError::new(
                ModelsErrorCode::Auth,
                format!("{} does not support {auth_type:?} login", provider.name()),
            )
        };
        let cancel = interaction.cancel();
        let login = async {
            match auth_type {
                AuthType::OAuth => {
                    let oauth = provider.auth().oauth.clone().ok_or_else(unsupported)?;
                    oauth.login(interaction, options).await.map(Credential::OAuth)
                }
                AuthType::ApiKey => {
                    let api_key = provider.auth().api_key.clone().filter(|auth| auth.supports_login());
                    let api_key = api_key.ok_or_else(unsupported)?;
                    api_key.login(interaction).await.map(Credential::ApiKey)
                }
            }
        };
        let credential = tokio::select! {
            () = cancel.cancelled() => return Err(ModelsError::aborted()),
            credential = login => credential?,
        };
        let stored = credential.clone();
        let modify: ModifyFn = Box::new(move |_| Box::pin(async move { Ok(Some(stored)) }));
        self.inner
            .credentials
            .modify(provider_id, modify)
            .await
            .map_err(|error| {
                ModelsError::with_source(
                    ModelsErrorCode::Auth,
                    format!("Credential store modify failed for {provider_id}"),
                    error,
                )
            })?;
        Ok(credential)
    }

    /// Remove the stored credential of a provider.
    pub async fn logout(&self, provider_id: &str) -> Result<(), ModelsError> {
        self.inner.credentials.delete(provider_id).await.map_err(|error| {
            ModelsError::with_source(
                ModelsErrorCode::Auth,
                format!("Credential store delete failed for {provider_id}"),
                error,
            )
        })
    }

    fn require_provider(&self, provider_id: &str) -> Result<Arc<dyn Provider>, ModelsError> {
        self.provider(provider_id)
            .ok_or_else(|| ModelsError::new(ModelsErrorCode::Provider, format!("Unknown provider: {provider_id}")))
    }

    /// Resolve auth and merge it into the request fields: explicit options win per field,
    /// then the `transform_headers` hook runs last. Returns the base URL auth resolved, if
    /// any. Auth resolution races `cancel`, as upstream's abort signal.
    async fn resolve_request(&self, model: AnyModel, request: RequestAuth<'_>) -> Result<Option<String>, ModelsError> {
        let RequestAuth {
            api_key,
            env,
            headers,
            transform_headers,
            cancel,
        } = request;
        let overrides = AuthResolutionOverrides {
            api_key: api_key.clone(),
            env: env.clone(),
            min_oauth_validity: None,
        };
        let resolve = self.get_model_auth(&model, &overrides);
        let resolution = match cancel {
            Some(cancel) => tokio::select! {
                biased;
                () = cancel.cancelled() => return Err(ModelsError::aborted()),
                resolution = resolve => resolution,
            },
            None => resolve.await,
        }?;
        let resolution = resolution.ok_or_else(|| {
            ModelsError::new(
                ModelsErrorCode::Auth,
                format!("Provider is not configured: {}", model.provider()),
            )
        })?;
        let auth = resolution.auth;
        *api_key = api_key.take().or(auth.api_key);
        let mut merged = merge_headers(auth.headers.as_ref(), headers.as_ref());
        if let Some(transform) = transform_headers.take() {
            merged = Some(transform(merged.unwrap_or_default()).await);
        }
        *headers = merged;
        if resolution.env.is_some() || env.is_some() {
            let mut merged = resolution.env.unwrap_or_default();
            merged.extend(env.take().unwrap_or_default());
            *env = Some(merged);
        }
        Ok(auth.base_url)
    }

    /// [`Self::resolve_request`] for chat stream options.
    async fn apply_auth(
        &self,
        model: &Model,
        mut options: StreamOptions,
    ) -> Result<(Model, StreamOptions), ModelsError> {
        let cancel = options.cancel.clone();
        let request = RequestAuth {
            api_key: &mut options.api_key,
            env: &mut options.env,
            headers: &mut options.headers,
            transform_headers: &mut options.transform_headers,
            cancel: cancel.as_ref(),
        };
        let base_url = self.resolve_request(AnyModel::Chat(model.clone()), request).await?;
        let mut request_model = model.clone();
        if let Some(base_url) = base_url {
            request_model.base_url = base_url;
        }
        Ok((request_model, options))
    }

    /// [`Self::resolve_request`] for [`RequestOptions`]; returns the resolved base URL.
    async fn apply_request_auth<M>(
        &self,
        model: AnyModel,
        options: &mut RequestOptions<M>,
    ) -> Result<Option<String>, ModelsError> {
        let cancel = options.cancel.clone();
        let request = RequestAuth {
            api_key: &mut options.api_key,
            env: &mut options.env,
            headers: &mut options.headers,
            transform_headers: &mut options.transform_headers,
            cancel: cancel.as_ref(),
        };
        self.resolve_request(model, request).await
    }

    /// Provider of a chat model with a deferred capability, or the upstream error.
    fn require_deferred_provider(
        &self,
        model: &Model,
        supported: impl Fn(&dyn Provider) -> bool,
    ) -> Result<Arc<dyn Provider>, ModelsError> {
        let provider = self.require_provider(&model.provider)?;
        if !supported(provider.as_ref()) {
            return Err(ModelsError::new(
                ModelsErrorCode::Provider,
                format!("Provider {} does not support deferred responses", model.provider),
            ));
        }
        Ok(provider)
    }

    /// Stream a response with the model's API options. Auth resolves through the provider.
    pub fn stream(&self, model: &Model, context: &Context, options: StreamOptions) -> AssistantMessageEventStream {
        let transcript = normalize_context(context);
        let models = self.clone();
        let target = model.clone();
        lazy_stream(model, options.cancel.clone(), async move {
            let provider = models.require_provider(&target.provider)?;
            let (request_model, options) = models.apply_auth(&target, options).await?;
            Ok(provider.stream(&request_model, &transcript, options))
        })
    }

    /// Await the final message of [`Self::stream`].
    pub async fn complete(&self, model: &Model, context: &Context, options: StreamOptions) -> AssistantMessage {
        self.stream(model, context, options).result().await
    }

    /// Stream with the provider-neutral reasoning options.
    pub fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let transcript = normalize_context(context);
        let models = self.clone();
        let target = model.clone();
        lazy_stream(model, options.base.cancel.clone(), async move {
            let provider = models.require_provider(&target.provider)?;
            let mut options = options;
            let (request_model, base) = models.apply_auth(&target, std::mem::take(&mut options.base)).await?;
            options.base = base;
            Ok(provider.stream_simple(&request_model, &transcript, options))
        })
    }

    /// Await the final message of [`Self::stream_simple`].
    pub async fn complete_simple(
        &self,
        model: &Model,
        context: &Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessage {
        self.stream_simple(model, context, options).result().await
    }

    /// Poll or redeem a deferred response through the owning provider, with auth resolved
    /// like [`Self::stream`]. A response that is still running ends with stop reason
    /// `deferred`. Providers without deferred support produce an error stream.
    pub fn stream_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: DeferredFetchOptions,
    ) -> AssistantMessageEventStream {
        let models = self.clone();
        let target = model.clone();
        let handle = handle.clone();
        lazy_stream(model, options.base.cancel.clone(), async move {
            let provider = models.require_deferred_provider(&target, |provider| provider.supports_fetch_deferred())?;
            let mut options = options;
            let base_url = models
                .apply_request_auth(AnyModel::Chat(target.clone()), &mut options.base)
                .await?;
            let mut request_model = target;
            if let Some(base_url) = base_url {
                request_model.base_url = base_url;
            }
            Ok(provider.fetch_deferred(&request_model, &handle, options))
        })
    }

    /// Await the final message of [`Self::stream_deferred`].
    pub async fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: DeferredFetchOptions,
    ) -> AssistantMessage {
        self.stream_deferred(model, handle, options).result().await
    }

    /// Best-effort cancellation of a deferred response through the owning provider.
    pub async fn cancel_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        mut options: DeferredCancelOptions,
    ) -> Result<(), Error> {
        let provider = self.require_deferred_provider(model, |provider| provider.supports_cancel_deferred())?;
        let base_url = self
            .apply_request_auth(AnyModel::Chat(model.clone()), &mut options)
            .await?;
        let mut request_model = model.clone();
        if let Some(base_url) = base_url {
            request_model.base_url = base_url;
        }
        provider.cancel_deferred(&request_model, handle, options).await
    }

    /// Generate images through the owning provider, with auth resolved like
    /// [`Self::stream`]. Never fails: unknown providers, unconfigured auth and providers
    /// without image generation return a result with stop reason `error`.
    pub async fn generate_images(
        &self,
        model: &ImageModel,
        context: &ImagesContext,
        mut options: ImagesOptions,
    ) -> AssistantImages {
        let prepared = async {
            let provider = self.require_provider(&model.provider)?;
            if !provider.supports_images() {
                return Err(ModelsError::new(
                    ModelsErrorCode::Provider,
                    format!("Provider {} does not support image generation", model.provider),
                ));
            }
            let base_url = self
                .apply_request_auth(AnyModel::Image(model.clone()), &mut options.base)
                .await?;
            Ok((provider, base_url))
        }
        .await;
        match prepared {
            Ok((provider, base_url)) => {
                let mut request_model = model.clone();
                if let Some(base_url) = base_url {
                    request_model.base_url = base_url;
                }
                provider.generate_images(&request_model, context, options).await
            }
            Err(error) => AssistantImages::error(model, error.to_string(), options.base.is_cancelled()),
        }
    }

    /// Classify structured state through the owning provider. Never fails: errors, including
    /// images sent to a model without image input, return a result with stop reason `error`.
    pub async fn classify(
        &self,
        model: &ClassifierModel,
        context: &ClassifierContext,
        mut options: ClassifierOptions,
    ) -> ClassifierResult {
        let prepared = async {
            check_classifier_input(model, context)?;
            let provider = self.require_provider(&model.provider)?;
            if !provider.supports_classify() {
                return Err(ModelsError::new(
                    ModelsErrorCode::Provider,
                    format!("Provider {} does not support classification", model.provider),
                ));
            }
            let base_url = self
                .apply_request_auth(AnyModel::Classifier(model.clone()), &mut options.base)
                .await?;
            Ok((provider, base_url))
        }
        .await;
        match prepared {
            Ok((provider, base_url)) => {
                let mut request_model = model.clone();
                if let Some(base_url) = base_url {
                    request_model.base_url = base_url;
                }
                provider.classify(&request_model, context, options).await
            }
            Err(error) => ClassifierResult::error(model, error.to_string(), options.base.is_cancelled()),
        }
    }
}

/// Mutable request fields that [`Models::resolve_request`] merges auth into.
struct RequestAuth<'a> {
    api_key: &'a mut Option<String>,
    env: &'a mut Option<ProviderEnv>,
    headers: &'a mut Option<ProviderHeaders>,
    transform_headers: &'a mut Option<TransformHeaders>,
    cancel: Option<&'a CancellationToken>,
}
