//! Providers: the runtime unit that owns a model catalog, auth and request behavior.

use futures::StreamExt;
use futures::future::BoxFuture;
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::sync::{Arc, RwLock};
use tokio_util::sync::CancellationToken;

use crate::auth::{Credential, ProviderAuth};
use crate::classifier::{ClassifierContext, ClassifierOptions, ClassifierResult, ProviderClassifier};
use crate::deferred::{DeferredCancelOptions, DeferredFetchOptions};
use crate::error::{Error, ModelsError, ModelsErrorCode};
use crate::events::{AssistantMessageEvent, AssistantMessageEventStream, ErrorReason};
use crate::images::{AssistantImages, ImagesContext, ImagesOptions, ProviderImages};
use crate::model::{AnyModel, ClassifierModel, ImageModel, Model, ModelType};
use crate::options::{SimpleStreamOptions, StreamOptions};
use crate::types::{AssistantMessage, DeferredHandle, ProviderHeaders, StopReason, TranscriptContext, now_ms};

/// The stream contract of an API implementation (a wire protocol such as
/// `anthropic-messages`). Implementations receive a normalized transcript and always return
/// a stream; failures after that are encoded in the stream.
///
/// APIs that support deferred responses also override the deferred methods and report them
/// through `supports_fetch_deferred` / `supports_cancel_deferred` (upstream: the optional
/// `fetchDeferred` / `cancelDeferred` members and `lazyApi` capabilities).
pub trait ProviderStreams: Send + Sync {
    fn stream(&self, model: &Model, context: &TranscriptContext, options: StreamOptions)
    -> AssistantMessageEventStream;
    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream;

    /// Whether [`Self::fetch_deferred`] is implemented.
    fn supports_fetch_deferred(&self) -> bool {
        false
    }

    /// Poll or redeem a deferred response. A response that is still running ends with stop
    /// reason `deferred` and the handle.
    fn fetch_deferred(
        &self,
        model: &Model,
        _handle: &DeferredHandle,
        _options: DeferredFetchOptions,
    ) -> AssistantMessageEventStream {
        AssistantMessageEventStream::from_final(error_message(
            model,
            "API does not support deferred responses",
            false,
            now_ms(),
        ))
    }

    /// Whether [`Self::cancel_deferred`] is implemented.
    fn supports_cancel_deferred(&self) -> bool {
        false
    }

    /// Best-effort cancellation of a deferred response.
    fn cancel_deferred<'a>(
        &'a self,
        _model: &'a Model,
        _handle: &'a DeferredHandle,
        _options: DeferredCancelOptions,
    ) -> BoxFuture<'a, Result<(), Error>> {
        Box::pin(async { Err(Error::msg("API cannot cancel deferred responses")) })
    }
}

/// Persisted dynamic model catalog of a provider.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsStoreEntry {
    /// Models of every type. Stored models of types this version does not know are dropped
    /// when the entry is deserialized.
    #[serde(deserialize_with = "deserialize_known_models")]
    pub models: Vec<AnyModel>,
    /// Unix timestamp from the remote catalog's Last-Modified header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_modified: Option<i64>,
    /// Unix timestamp of the last completed remote check.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_at: Option<i64>,
    /// Opaque ETag validator, echoed back as If-None-Match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub etag: Option<String>,
}

fn deserialize_known_models<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Vec<AnyModel>, D::Error> {
    use serde::Deserialize;
    let values = Vec::<serde_json::Value>::deserialize(deserializer)?;
    let mut models = Vec::with_capacity(values.len());
    for value in values {
        let kind = value.get("type").and_then(serde_json::Value::as_str).unwrap_or("chat");
        if matches!(kind, "chat" | "image" | "classifier") {
            models.push(serde_json::from_value(value).map_err(serde::de::Error::custom)?);
        }
    }
    Ok(models)
}

/// Storage change of a [`ModelsPublication`].
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Persist {
    /// Leave storage unchanged.
    #[default]
    Keep,
    Write(ModelsStoreEntry),
    Delete,
}

/// Generation-checked publication of a refreshed catalog.
#[derive(Default)]
pub struct ModelsPublication {
    pub persist: Persist,
    /// Synchronous update of provider-private in-memory catalog state, run only when the
    /// publication is still current.
    pub update: Option<Box<dyn FnOnce() + Send>>,
}

impl fmt::Debug for ModelsPublication {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelsPublication")
            .field("persist", &self.persist)
            .finish_non_exhaustive()
    }
}

/// Publishes a refresh result; resolves `false` when superseded or cancelled.
pub type PublishFn = Arc<dyn Fn(ModelsPublication) -> BoxFuture<'static, bool> + Send + Sync>;

/// Input of [`Provider::refresh_models`].
#[derive(Clone)]
pub struct RefreshModelsContext {
    /// Effective configured credential; OAuth credentials are refreshed before network use.
    pub credential: Option<Credential>,
    /// Provider-scoped catalog snapshot captured before this refresh phase.
    pub stored: Option<ModelsStoreEntry>,
    pub publish: PublishFn,
    /// False during offline/cache-only initialization.
    pub allow_network: bool,
    /// Bypass provider freshness checks.
    pub force: bool,
    /// Always present; honor it for blocking work.
    pub cancel: CancellationToken,
    /// Base URL of the registered provider.
    pub base_url: Option<String>,
}

impl fmt::Debug for RefreshModelsContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RefreshModelsContext")
            .field("allow_network", &self.allow_network)
            .field("force", &self.force)
            .finish_non_exhaustive()
    }
}

/// A provider: identity, auth, a model catalog and the operations its models support.
pub trait Provider: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn base_url(&self) -> Option<&str> {
        None
    }
    fn headers(&self) -> Option<&ProviderHeaders> {
        None
    }
    /// Every provider has auth semantics, even ambient or keyless ones.
    fn auth(&self) -> &ProviderAuth;

    /// Current chat models. Dynamic providers return the list as of the last refresh.
    fn models(&self) -> Vec<Model>;

    /// Current models of every type.
    fn all_models(&self) -> Vec<AnyModel> {
        self.models().into_iter().map(AnyModel::Chat).collect()
    }

    /// Whether [`Self::refresh_models`] does anything.
    fn is_dynamic(&self) -> bool {
        false
    }

    /// Restore `context.stored` and optionally fetch a newer list. Implementations keep their
    /// previous list on failure and publish through `context.publish`.
    fn refresh_models(&self, _context: RefreshModelsContext) -> BoxFuture<'_, Result<(), ModelsError>> {
        Box::pin(async { Ok(()) })
    }

    /// Credential-specific chat model availability.
    fn filter_models(&self, models: Vec<Model>, _credential: Option<&Credential>) -> Vec<Model> {
        models
    }

    /// Credential-specific availability across model types. `None` applies
    /// [`Self::filter_models`] to chat models and keeps every other model.
    fn filter_all_models(&self, _models: &[AnyModel], _credential: Option<&Credential>) -> Option<Vec<AnyModel>> {
        None
    }

    /// Stream a normalized transcript with the model's API options.
    fn stream(&self, model: &Model, context: &TranscriptContext, options: StreamOptions)
    -> AssistantMessageEventStream;

    /// Stream with provider-neutral reasoning options.
    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream;

    /// Whether [`Self::fetch_deferred`] is implemented (upstream: `fetchDeferred` exists).
    fn supports_fetch_deferred(&self) -> bool {
        false
    }

    /// Poll or redeem a deferred response.
    fn fetch_deferred(
        &self,
        model: &Model,
        _handle: &DeferredHandle,
        _options: DeferredFetchOptions,
    ) -> AssistantMessageEventStream {
        let message = format!("Provider {} does not support deferred responses", self.id());
        AssistantMessageEventStream::from_final(error_message(model, message, false, now_ms()))
    }

    /// Whether [`Self::cancel_deferred`] is implemented (upstream: `cancelDeferred` exists).
    fn supports_cancel_deferred(&self) -> bool {
        false
    }

    /// Best-effort cancellation of a deferred response.
    fn cancel_deferred<'a>(
        &'a self,
        _model: &'a Model,
        _handle: &'a DeferredHandle,
        _options: DeferredCancelOptions,
    ) -> BoxFuture<'a, Result<(), Error>> {
        let message = format!("Provider {} does not support deferred responses", self.id());
        Box::pin(async move { Err(ModelsError::new(ModelsErrorCode::Provider, message).into()) })
    }

    /// Whether [`Self::generate_images`] is implemented (upstream: `generateImages` exists).
    fn supports_images(&self) -> bool {
        false
    }

    /// Generate images. Never fails: errors are results with stop reason `error`.
    fn generate_images<'a>(
        &'a self,
        model: &'a ImageModel,
        _context: &'a ImagesContext,
        options: ImagesOptions,
    ) -> BoxFuture<'a, AssistantImages> {
        let message = format!("Provider {} does not support image generation", self.id());
        let aborted = options.base.is_cancelled();
        Box::pin(async move { AssistantImages::error(model, message, aborted) })
    }

    /// Whether [`Self::classify`] is implemented (upstream: `classify` exists).
    fn supports_classify(&self) -> bool {
        false
    }

    /// Classify structured state. Never fails: errors are results with stop reason `error`.
    fn classify<'a>(
        &'a self,
        model: &'a ClassifierModel,
        _context: &'a ClassifierContext,
        options: ClassifierOptions,
    ) -> BoxFuture<'a, ClassifierResult> {
        let message = format!("Provider {} does not support classification", self.id());
        let aborted = options.base.is_cancelled();
        Box::pin(async move { ClassifierResult::error(model, message, aborted) })
    }
}

/// An assistant error message for `model`.
pub fn error_message(model: &Model, message: impl Into<String>, aborted: bool, timestamp: i64) -> AssistantMessage {
    AssistantMessage {
        stop_reason: if aborted {
            StopReason::Aborted
        } else {
            StopReason::Error
        },
        error_message: Some(message.into()),
        timestamp,
        ..AssistantMessage::empty(&model.api, &model.provider, &model.id)
    }
}

/// A stream returned synchronously while async setup (auth resolution and the like) runs
/// behind it. Setup failures end the stream with an `error` event. Events of the inner
/// stream are forwarded; dropping the outer stream drops the inner one.
///
/// Requires a Tokio runtime.
pub fn lazy_stream<F>(model: &Model, cancel: Option<CancellationToken>, setup: F) -> AssistantMessageEventStream
where
    F: Future<Output = Result<AssistantMessageEventStream, ModelsError>> + Send + 'static,
{
    let started_at = now_ms();
    let model = model.clone();
    let (mut sender, outer) = AssistantMessageEventStream::channel();
    tokio::spawn(async move {
        match setup.await {
            Ok(mut inner) => {
                let mut finished = false;
                while let Some(event) = inner.next().await {
                    if sender.is_closed() {
                        return;
                    }
                    finished |= event.is_terminal();
                    sender.push(event);
                }
                if !finished {
                    let error = error_message(&model, "Stream ended without a final message", false, started_at);
                    sender.push(AssistantMessageEvent::Error {
                        reason: ErrorReason::Error,
                        error,
                    });
                }
            }
            Err(error) => {
                let aborted = error.code() == ModelsErrorCode::Aborted
                    || cancel.as_ref().is_some_and(CancellationToken::is_cancelled);
                let message = error_message(&model, error.to_string(), aborted, started_at);
                let reason = if aborted {
                    ErrorReason::Aborted
                } else {
                    ErrorReason::Error
                };
                sender.push(AssistantMessageEvent::Error { reason, error: message });
            }
        }
    });
    outer
}

/// Fetches a dynamic model list for [`create_provider`].
pub type FetchModels =
    Arc<dyn Fn(RefreshModelsContext) -> BoxFuture<'static, Result<Vec<AnyModel>, ModelsError>> + Send + Sync>;
/// Credential-specific chat model filter.
pub type FilterModels = Arc<dyn Fn(Vec<Model>, Option<&Credential>) -> Vec<Model> + Send + Sync>;
/// Credential-specific filter across model types.
pub type FilterAllModels = Arc<dyn Fn(&[AnyModel], Option<&Credential>) -> Vec<AnyModel> + Send + Sync>;

/// Chat implementation of a provider built by [`create_provider`].
#[derive(Clone)]
pub enum ProviderApi {
    /// One implementation streams every chat model.
    Single(Arc<dyn ProviderStreams>),
    /// Mixed-API providers dispatch on `model.api`.
    ByApi(HashMap<String, Arc<dyn ProviderStreams>>),
}

impl fmt::Debug for ProviderApi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Single(_) => f.write_str("Single"),
            Self::ByApi(map) => f.debug_tuple("ByApi").field(&map.keys().collect::<Vec<_>>()).finish(),
        }
    }
}

impl ProviderApi {
    pub fn single(streams: impl ProviderStreams + 'static) -> Self {
        Self::Single(Arc::new(streams))
    }

    /// No chat implementation, for providers that only generate images or classify
    /// (upstream: `api` omitted). An empty `ByApi` map.
    pub fn none() -> Self {
        Self::ByApi(HashMap::new())
    }

    fn for_model(&self, model: &Model) -> Option<&Arc<dyn ProviderStreams>> {
        match self {
            Self::Single(streams) => Some(streams),
            Self::ByApi(map) => map.get(&model.api),
        }
    }

    fn implementations(&self) -> Vec<&Arc<dyn ProviderStreams>> {
        match self {
            Self::Single(streams) => vec![streams],
            Self::ByApi(map) => map.values().collect(),
        }
    }
}

/// Parts of a provider built by [`create_provider`].
#[derive(Clone)]
pub struct CreateProviderOptions {
    pub id: String,
    /// Display name. Default: `id`.
    pub name: Option<String>,
    pub base_url: Option<String>,
    pub headers: Option<ProviderHeaders>,
    pub auth: ProviderAuth,
    /// Static baseline models of every type.
    pub models: Vec<AnyModel>,
    /// Fetch a dynamic overlay; restored and published transactionally.
    pub fetch_models: Option<FetchModels>,
    pub filter_models: Option<FilterModels>,
    pub filter_all_models: Option<FilterAllModels>,
    /// Chat implementation. [`ProviderApi::none`] when the provider only has image or
    /// classifier implementations.
    pub api: ProviderApi,
    /// Image-generation implementations keyed by `model.api`.
    pub images: HashMap<String, Arc<dyn ProviderImages>>,
    /// Classifier implementations keyed by `model.api`.
    pub classifiers: HashMap<String, Arc<dyn ProviderClassifier>>,
}

impl fmt::Debug for CreateProviderOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CreateProviderOptions")
            .field("id", &self.id)
            .field("models", &self.models.len())
            .field("api", &self.api)
            .field("images", &self.images.keys().collect::<Vec<_>>())
            .field("classifiers", &self.classifiers.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl CreateProviderOptions {
    pub fn new(id: impl Into<String>, auth: ProviderAuth, models: Vec<AnyModel>, api: ProviderApi) -> Self {
        Self {
            id: id.into(),
            name: None,
            base_url: None,
            headers: None,
            auth,
            models,
            fetch_models: None,
            filter_models: None,
            filter_all_models: None,
            api,
            images: HashMap::new(),
            classifiers: HashMap::new(),
        }
    }

    /// Register the image-generation implementation of an image API id.
    #[must_use]
    pub fn images(mut self, api: impl Into<String>, implementation: impl ProviderImages + 'static) -> Self {
        self.images.insert(api.into(), Arc::new(implementation));
        self
    }

    /// Register the classifier implementation of a classifier API id.
    #[must_use]
    pub fn classifier(mut self, api: impl Into<String>, implementation: impl ProviderClassifier + 'static) -> Self {
        self.classifiers.insert(api.into(), Arc::new(implementation));
        self
    }

    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    #[must_use]
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }
}

/// Provider built from parts by [`create_provider`].
pub struct BuiltProvider {
    options: CreateProviderOptions,
    name: String,
    dynamic: Arc<RwLock<Vec<AnyModel>>>,
}

impl fmt::Debug for BuiltProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuiltProvider")
            .field("id", &self.options.id)
            .finish_non_exhaustive()
    }
}

/// Build a provider from parts. A single `api` streams all chat models; a map dispatches
/// on `model.api`, and a model whose API has no entry produces a stream error. The `images`
/// and `classifiers` maps dispatch on `model.api` the same way.
///
/// # Panics
///
/// When `api`, `images` and `classifiers` are all empty (upstream throws). Use
/// [`try_create_provider`] to get the error instead.
pub fn create_provider(options: CreateProviderOptions) -> BuiltProvider {
    match try_create_provider(options) {
        Ok(provider) => provider,
        Err(error) => panic!("{error}"),
    }
}

/// [`create_provider`] that returns an error when the provider has no implementation at all:
/// at least one of `api`, `images` or `classifiers` must hold an implementation.
pub fn try_create_provider(options: CreateProviderOptions) -> Result<BuiltProvider, ModelsError> {
    if options.api.implementations().is_empty() && options.images.is_empty() && options.classifiers.is_empty() {
        return Err(ModelsError::new(
            ModelsErrorCode::Provider,
            format!(
                "Provider {}: at least one of \"api\", \"images\", or \"classifiers\" is required.",
                options.id
            ),
        ));
    }
    let name = options.name.clone().unwrap_or_else(|| options.id.clone());
    Ok(BuiltProvider {
        options,
        name,
        dynamic: Arc::default(),
    })
}

impl BuiltProvider {
    fn current_models(&self) -> Vec<AnyModel> {
        let mut merged = self.options.models.clone();
        let dynamic = self.dynamic.read().unwrap_or_else(std::sync::PoisonError::into_inner);
        for model in dynamic.iter() {
            match merged
                .iter_mut()
                .find(|entry| entry.model_type() == model.model_type() && entry.id() == model.id())
            {
                Some(entry) => *entry = model.clone(),
                None => merged.push(model.clone()),
            }
        }
        merged
    }

    fn dispatch(
        &self,
        model: &Model,
        cancel: Option<CancellationToken>,
        run: impl FnOnce(&Arc<dyn ProviderStreams>) -> AssistantMessageEventStream,
    ) -> AssistantMessageEventStream {
        match self.options.api.for_model(model) {
            Some(streams) => run(streams),
            None => {
                let message = format!(
                    "Provider {} has no API implementation for \"{}\"",
                    self.options.id, model.api
                );
                lazy_stream(model, cancel, async move {
                    Err(ModelsError::new(ModelsErrorCode::Stream, message))
                })
            }
        }
    }
}

impl Provider for BuiltProvider {
    fn id(&self) -> &str {
        &self.options.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn base_url(&self) -> Option<&str> {
        self.options.base_url.as_deref()
    }

    fn headers(&self) -> Option<&ProviderHeaders> {
        self.options.headers.as_ref()
    }

    fn auth(&self) -> &ProviderAuth {
        &self.options.auth
    }

    fn models(&self) -> Vec<Model> {
        self.current_models()
            .into_iter()
            .filter_map(|model| match model {
                AnyModel::Chat(model) => Some(model),
                _ => None,
            })
            .collect()
    }

    fn all_models(&self) -> Vec<AnyModel> {
        self.current_models()
    }

    fn is_dynamic(&self) -> bool {
        self.options.fetch_models.is_some()
    }

    fn refresh_models(&self, context: RefreshModelsContext) -> BoxFuture<'_, Result<(), ModelsError>> {
        Box::pin(async move {
            let Some(fetch) = self.options.fetch_models.clone() else {
                return Ok(());
            };
            let id = self.options.id.clone();
            if let Some(stored) = &context.stored {
                let restored: Vec<AnyModel> = stored
                    .models
                    .iter()
                    .filter(|model| model.provider() == id)
                    .cloned()
                    .collect();
                let dynamic = self.dynamic.clone();
                let published = (context.publish)(ModelsPublication {
                    persist: Persist::Keep,
                    update: Some(Box::new(move || {
                        *dynamic.write().unwrap_or_else(std::sync::PoisonError::into_inner) = restored;
                    })),
                })
                .await;
                if !published {
                    return Ok(());
                }
            }
            if !context.allow_network || context.cancel.is_cancelled() {
                return Ok(());
            }
            let publish = context.publish.clone();
            let cancel = context.cancel.clone();
            let fetched = fetch(context).await?;
            if cancel.is_cancelled() {
                return Ok(());
            }
            let dynamic = self.dynamic.clone();
            let models = fetched.clone();
            publish(ModelsPublication {
                persist: Persist::Write(ModelsStoreEntry {
                    models: fetched,
                    checked_at: Some(now_ms()),
                    ..Default::default()
                }),
                update: Some(Box::new(move || {
                    *dynamic.write().unwrap_or_else(std::sync::PoisonError::into_inner) = models;
                })),
            })
            .await;
            Ok(())
        })
    }

    fn filter_models(&self, models: Vec<Model>, credential: Option<&Credential>) -> Vec<Model> {
        match &self.options.filter_models {
            Some(filter) => filter(models, credential),
            None => models,
        }
    }

    fn filter_all_models(&self, models: &[AnyModel], credential: Option<&Credential>) -> Option<Vec<AnyModel>> {
        self.options
            .filter_all_models
            .as_ref()
            .map(|filter| filter(models, credential))
    }

    fn stream(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        let cancel = options.cancel.clone();
        self.dispatch(model, cancel, |streams| streams.stream(model, context, options))
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let cancel = options.base.cancel.clone();
        self.dispatch(model, cancel, |streams| streams.stream_simple(model, context, options))
    }

    fn supports_fetch_deferred(&self) -> bool {
        self.options
            .api
            .implementations()
            .iter()
            .any(|streams| streams.supports_fetch_deferred())
    }

    fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: DeferredFetchOptions,
    ) -> AssistantMessageEventStream {
        match self.options.api.for_model(model) {
            Some(streams) if streams.supports_fetch_deferred() => streams.fetch_deferred(model, handle, options),
            _ => {
                let message = format!(
                    "Provider {} does not support deferred responses for \"{}\"",
                    self.options.id, model.api
                );
                lazy_stream(model, options.base.cancel.clone(), async move {
                    Err(ModelsError::new(ModelsErrorCode::Provider, message))
                })
            }
        }
    }

    fn supports_cancel_deferred(&self) -> bool {
        self.options
            .api
            .implementations()
            .iter()
            .any(|streams| streams.supports_cancel_deferred())
    }

    fn cancel_deferred<'a>(
        &'a self,
        model: &'a Model,
        handle: &'a DeferredHandle,
        options: DeferredCancelOptions,
    ) -> BoxFuture<'a, Result<(), Error>> {
        match self.options.api.for_model(model) {
            Some(streams) if streams.supports_cancel_deferred() => streams.cancel_deferred(model, handle, options),
            _ => {
                let message = format!(
                    "Provider {} cannot cancel deferred responses for \"{}\"",
                    self.options.id, model.api
                );
                Box::pin(async move { Err(ModelsError::new(ModelsErrorCode::Provider, message).into()) })
            }
        }
    }

    fn supports_images(&self) -> bool {
        !self.options.images.is_empty()
    }

    fn generate_images<'a>(
        &'a self,
        model: &'a ImageModel,
        context: &'a ImagesContext,
        options: ImagesOptions,
    ) -> BoxFuture<'a, AssistantImages> {
        match self.options.images.get(&model.api) {
            Some(implementation) => implementation.generate_images(model, context, options),
            None => {
                let message = format!(
                    "Provider {} has no image generation implementation for \"{}\"",
                    self.options.id, model.api
                );
                Box::pin(async move { AssistantImages::error(model, message, false) })
            }
        }
    }

    fn supports_classify(&self) -> bool {
        !self.options.classifiers.is_empty()
    }

    fn classify<'a>(
        &'a self,
        model: &'a ClassifierModel,
        context: &'a ClassifierContext,
        options: ClassifierOptions,
    ) -> BoxFuture<'a, ClassifierResult> {
        match self.options.classifiers.get(&model.api) {
            Some(implementation) => implementation.classify(model, context, options),
            None => {
                let message = format!(
                    "Provider {} has no classifier implementation for \"{}\"",
                    self.options.id, model.api
                );
                Box::pin(async move { ClassifierResult::error(model, message, false) })
            }
        }
    }
}

/// Whether a model has the given type.
pub fn is_model_type(model: &AnyModel, model_type: ModelType) -> bool {
    model.model_type() == model_type
}
