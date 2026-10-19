//! Provider auth: credentials, credential stores, API-key and OAuth auth methods, and the
//! resolution order shared by every [`crate::Models`] operation.
//!
//! A stored credential owns its provider: the environment is consulted only when nothing
//! is stored, and a failed OAuth refresh never silently falls back to an environment key.

use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::error::{ModelsError, ModelsErrorCode};
use crate::types::{JsonObject, ProviderEnv, ProviderHeaders, now_ms};

/// Request auth for a single model request.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelAuth {
    pub api_key: Option<String>,
    pub headers: Option<ProviderHeaders>,
    pub base_url: Option<String>,
}

/// Stored API-key credential. `env` holds provider-scoped config such as account ids.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiKeyCredential {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<ProviderEnv>,
}

/// Stored OAuth credential. Provider-specific fields live in `extra`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OAuthCredential {
    pub refresh: String,
    pub access: String,
    /// Expiry as a Unix timestamp in milliseconds.
    pub expires: i64,
    #[serde(flatten)]
    pub extra: JsonObject,
}

/// One type-tagged credential per provider, the shape of pi's `auth.json` entries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Credential {
    #[serde(rename = "api_key")]
    ApiKey(ApiKeyCredential),
    #[serde(rename = "oauth")]
    OAuth(OAuthCredential),
}

impl Credential {
    pub fn auth_type(&self) -> AuthType {
        match self {
            Self::ApiKey(_) => AuthType::ApiKey,
            Self::OAuth(_) => AuthType::OAuth,
        }
    }
}

/// Auth method kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuthType {
    #[serde(rename = "api_key")]
    ApiKey,
    #[serde(rename = "oauth")]
    OAuth,
}

/// Non-secret credential metadata for account/status enumeration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CredentialInfo {
    pub provider_id: String,
    pub auth_type: AuthType,
}

/// Write function of [`CredentialStore::modify`]: receives the current credential and
/// returns the new one, or `None` to leave the entry unchanged.
pub type ModifyFn =
    Box<dyn FnOnce(Option<Credential>) -> BoxFuture<'static, Result<Option<Credential>, ModelsError>> + Send>;

/// App-owned credential storage keyed by provider id, one credential per provider.
///
/// `modify` is the only write path, so every mutation is a serialized read-modify-write;
/// OAuth refresh runs inside it so concurrent requests (and processes, where the store
/// supports cross-process locking) cannot double-refresh a rotated token.
pub trait CredentialStore: Send + Sync {
    /// The stored credential, possibly expired. `Ok(None)` when missing.
    fn read<'a>(&'a self, provider_id: &'a str) -> BoxFuture<'a, Result<Option<Credential>, ModelsError>>;
    /// Stored credential metadata, without resolving secrets or running key commands.
    fn list(&self) -> BoxFuture<'_, Result<Vec<CredentialInfo>, ModelsError>>;
    /// Serialized read-modify-write. Resolves with the post-write credential.
    fn modify<'a>(
        &'a self,
        provider_id: &'a str,
        f: ModifyFn,
    ) -> BoxFuture<'a, Result<Option<Credential>, ModelsError>>;
    /// Remove a credential (logout), serialized against `modify`.
    fn delete<'a>(&'a self, provider_id: &'a str) -> BoxFuture<'a, Result<(), ModelsError>>;
}

/// Default in-memory credential store. Writes are serialized per provider.
#[derive(Debug, Default)]
pub struct InMemoryCredentialStore {
    credentials: std::sync::Mutex<HashMap<String, Credential>>,
    locks: std::sync::Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

impl InMemoryCredentialStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock_for(&self, provider_id: &str) -> Arc<Mutex<()>> {
        let mut locks = self.locks.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        locks.entry(provider_id.to_owned()).or_default().clone()
    }

    fn get(&self, provider_id: &str) -> Option<Credential> {
        self.credentials
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(provider_id)
            .cloned()
    }
}

impl CredentialStore for InMemoryCredentialStore {
    fn read<'a>(&'a self, provider_id: &'a str) -> BoxFuture<'a, Result<Option<Credential>, ModelsError>> {
        Box::pin(async move { Ok(self.get(provider_id)) })
    }

    fn list(&self) -> BoxFuture<'_, Result<Vec<CredentialInfo>, ModelsError>> {
        Box::pin(async move {
            let credentials = self
                .credentials
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut list: Vec<CredentialInfo> = credentials
                .iter()
                .map(|(id, credential)| CredentialInfo {
                    provider_id: id.clone(),
                    auth_type: credential.auth_type(),
                })
                .collect();
            list.sort_by(|a, b| a.provider_id.cmp(&b.provider_id));
            Ok(list)
        })
    }

    fn modify<'a>(
        &'a self,
        provider_id: &'a str,
        f: ModifyFn,
    ) -> BoxFuture<'a, Result<Option<Credential>, ModelsError>> {
        Box::pin(async move {
            let lock = self.lock_for(provider_id);
            let _guard = lock.lock().await;
            let current = self.get(provider_id);
            let next = f(current.clone()).await?;
            match next {
                Some(next) => {
                    self.credentials
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .insert(provider_id.to_owned(), next.clone());
                    Ok(Some(next))
                }
                None => Ok(current),
            }
        })
    }

    fn delete<'a>(&'a self, provider_id: &'a str) -> BoxFuture<'a, Result<(), ModelsError>> {
        Box::pin(async move {
            let lock = self.lock_for(provider_id);
            let _guard = lock.lock().await;
            self.credentials
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(provider_id);
            Ok(())
        })
    }
}

/// Environment access for auth resolution. Injectable for tests and sandboxes.
pub trait AuthContext: Send + Sync {
    /// The value of an environment variable; blank values count as unset.
    fn env(&self, name: &str) -> Option<String>;
    /// Whether a file exists. Supports a leading `~`.
    fn file_exists(&self, path: &str) -> bool;
}

/// The process environment and file system.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultAuthContext;

impl AuthContext for DefaultAuthContext {
    fn env(&self, name: &str) -> Option<String> {
        std::env::var(name).ok().filter(|value| !value.trim().is_empty())
    }

    fn file_exists(&self, path: &str) -> bool {
        expand_home(path).is_some_and(|path| std::path::Path::new(&path).exists())
    }
}

/// Expand a leading `~` to the home directory.
pub fn expand_home(path: &str) -> Option<String> {
    match path.strip_prefix('~') {
        Some(rest) => {
            let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).ok()?;
            Some(format!("{home}{rest}"))
        }
        None => Some(path.to_owned()),
    }
}

/// `env` values first, then the base context.
struct OverlayAuthContext<'a> {
    base: &'a dyn AuthContext,
    env: &'a ProviderEnv,
}

impl AuthContext for OverlayAuthContext<'_> {
    fn env(&self, name: &str) -> Option<String> {
        self.env
            .get(name)
            .filter(|value| !value.is_empty())
            .cloned()
            .or_else(|| self.base.env(name))
    }

    fn file_exists(&self, path: &str) -> bool {
        self.base.file_exists(path)
    }
}

/// A value from `env`, else from the process environment. Blank values count as unset.
pub fn env_value(env: Option<&ProviderEnv>, name: &str) -> Option<String> {
    env.and_then(|env| env.get(name))
        .filter(|value| !value.trim().is_empty())
        .cloned()
        .or_else(|| DefaultAuthContext.env(name))
}

/// Result of resolving auth for a provider or model.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AuthResult {
    pub auth: ModelAuth,
    /// Provider-scoped config resolved from credentials and ambient context.
    pub env: Option<ProviderEnv>,
    /// Label for status UI: `ANTHROPIC_API_KEY`, `OAuth`, `stored credential`.
    pub source: Option<String>,
}

/// Outcome of a side-effect-free auth availability check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthCheck {
    pub source: Option<String>,
    pub auth_type: AuthType,
}

/// Option shown to the user by a `select` prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthSelectOption {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
}

/// Prompt shown during login.
#[derive(Clone, Debug)]
pub enum AuthPrompt {
    Text {
        message: String,
        placeholder: Option<String>,
    },
    Secret {
        message: String,
        placeholder: Option<String>,
    },
    /// `prompt()` returns the selected option id.
    Select {
        message: String,
        options: Vec<AuthSelectOption>,
    },
    /// A code pasted by the user, raced against a local callback server; `cancel` fires
    /// when the server wins.
    ManualCode {
        message: String,
        placeholder: Option<String>,
        cancel: CancellationToken,
    },
}

/// A link in an info event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthInfoLink {
    pub url: String,
    pub label: Option<String>,
}

/// Login progress shown to the user.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuthEvent {
    Info {
        message: String,
        links: Vec<AuthInfoLink>,
    },
    AuthUrl {
        url: String,
        instructions: Option<String>,
    },
    DeviceCode {
        user_code: String,
        verification_uri: String,
        interval_seconds: Option<u64>,
        expires_in_seconds: Option<u64>,
    },
    Progress {
        message: String,
    },
}

/// Login interaction callbacks serving both API-key and OAuth flows.
pub trait AuthInteraction: Send + Sync {
    /// The entered/selected string. Errors on cancel.
    fn prompt(&self, prompt: AuthPrompt) -> BoxFuture<'_, Result<String, ModelsError>>;
    fn notify(&self, event: AuthEvent);
    /// Cancels the whole login flow.
    fn cancel(&self) -> CancellationToken {
        CancellationToken::new()
    }
}

/// App-supplied context for `Models::login`.
#[derive(Clone, Default)]
pub struct LoginOptions {
    /// Stable id of this app installation; called only by flows that need it.
    pub device_id: Option<Arc<dyn Fn() -> String + Send + Sync>>,
    /// Name this app introduces itself with during login.
    pub agent_name: Option<String>,
}

impl fmt::Debug for LoginOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoginOptions")
            .field("agent_name", &self.agent_name)
            .finish_non_exhaustive()
    }
}

/// Input of [`ApiKeyAuth::resolve`] and [`ApiKeyAuth::check`].
#[derive(Clone, Copy)]
pub struct ApiKeyInput<'a> {
    pub ctx: &'a dyn AuthContext,
    pub credential: Option<&'a ApiKeyCredential>,
}

impl fmt::Debug for ApiKeyInput<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApiKeyInput")
            .field("has_credential", &self.credential.is_some())
            .finish_non_exhaustive()
    }
}

/// API-key auth: stored key/provider env plus ambient sources (env vars, AWS profiles, ADC
/// files). Every provider has one, even keyless local servers, whose `resolve` reports
/// whether the provider is configured.
pub trait ApiKeyAuth: Send + Sync {
    /// Display name, e.g. "Anthropic API key".
    fn name(&self) -> &str;

    /// Resolve auth from the stored credential and/or ambient sources, merging per field.
    /// `Ok(None)` means not configured.
    fn resolve<'a>(&'a self, input: ApiKeyInput<'a>) -> BoxFuture<'a, Result<Option<AuthResult>, ModelsError>>;

    /// Side-effect-free availability check, for `resolve` implementations that run commands
    /// or do request-time work. `None` makes `Models` check by resolving.
    fn check<'a>(&'a self, _input: ApiKeyInput<'a>) -> Option<BoxFuture<'a, Result<Option<AuthCheck>, ModelsError>>> {
        None
    }

    /// Whether [`Self::login`] is implemented. Ambient-only providers have no login.
    fn supports_login(&self) -> bool {
        false
    }

    /// Interactive setup (prompt for key/provider env).
    fn login<'a>(
        &'a self,
        _interaction: &'a dyn AuthInteraction,
    ) -> BoxFuture<'a, Result<ApiKeyCredential, ModelsError>> {
        let name = self.name().to_owned();
        Box::pin(async move {
            Err(ModelsError::new(
                ModelsErrorCode::Auth,
                format!("{name} does not support login"),
            ))
        })
    }
}

/// OAuth auth. The `refresh`/`to_auth` split lets `Models` own the locked refresh.
pub trait OAuthAuth: Send + Sync {
    /// Display name, e.g. "Anthropic (Claude Pro/Max)".
    fn name(&self) -> &str;
    /// Whether access is backed by a provider subscription.
    fn is_subscription(&self) -> bool {
        false
    }
    /// Selector label of the login option.
    fn login_label(&self) -> Option<&str> {
        None
    }
    fn login<'a>(
        &'a self,
        interaction: &'a dyn AuthInteraction,
        options: &'a LoginOptions,
    ) -> BoxFuture<'a, Result<OAuthCredential, ModelsError>>;
    /// Exchange the refresh token. Runs under the credential-store lock.
    fn refresh(&self, credential: OAuthCredential) -> BoxFuture<'_, Result<OAuthCredential, ModelsError>>;
    /// Side-effect-free derivation of request auth from a valid credential.
    fn to_auth<'a>(&'a self, credential: &'a OAuthCredential) -> BoxFuture<'a, Result<ModelAuth, ModelsError>>;
}

/// Auth methods of a provider. At least one is present.
#[derive(Clone, Default)]
pub struct ProviderAuth {
    pub api_key: Option<Arc<dyn ApiKeyAuth>>,
    pub oauth: Option<Arc<dyn OAuthAuth>>,
}

impl fmt::Debug for ProviderAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderAuth")
            .field("api_key", &self.api_key.as_ref().map(|auth| auth.name().to_owned()))
            .field("oauth", &self.oauth.as_ref().map(|auth| auth.name().to_owned()))
            .finish()
    }
}

impl ProviderAuth {
    pub fn api_key(auth: impl ApiKeyAuth + 'static) -> Self {
        Self {
            api_key: Some(Arc::new(auth)),
            oauth: None,
        }
    }

    /// Keyless auth for local servers: always configured, no key.
    pub fn keyless(name: impl Into<String>) -> Self {
        Self::api_key(KeylessAuth { name: name.into() })
    }

    #[must_use]
    pub fn with_oauth(mut self, oauth: impl OAuthAuth + 'static) -> Self {
        self.oauth = Some(Arc::new(oauth));
        self
    }
}

/// Standard API-key auth: a stored key wins, otherwise the first set env var. Its `login`
/// prompts for the key.
#[derive(Clone, Debug)]
pub struct EnvApiKeyAuth {
    name: String,
    env_vars: Vec<String>,
}

/// [`EnvApiKeyAuth`] for `env_vars`.
pub fn env_api_key_auth(name: impl Into<String>, env_vars: &[&str]) -> EnvApiKeyAuth {
    EnvApiKeyAuth {
        name: name.into(),
        env_vars: env_vars.iter().map(|var| (*var).to_owned()).collect(),
    }
}

impl ApiKeyAuth for EnvApiKeyAuth {
    fn name(&self) -> &str {
        &self.name
    }

    fn resolve<'a>(&'a self, input: ApiKeyInput<'a>) -> BoxFuture<'a, Result<Option<AuthResult>, ModelsError>> {
        Box::pin(async move {
            if let Some(credential) = input.credential
                && let Some(key) = credential.key.as_ref().filter(|key| !key.is_empty())
            {
                return Ok(Some(AuthResult {
                    auth: ModelAuth {
                        api_key: Some(key.clone()),
                        ..ModelAuth::default()
                    },
                    env: credential.env.clone(),
                    source: Some("stored credential".into()),
                }));
            }
            for var in &self.env_vars {
                if let Some(value) = input.ctx.env(var) {
                    return Ok(Some(AuthResult {
                        auth: ModelAuth {
                            api_key: Some(value),
                            ..ModelAuth::default()
                        },
                        env: None,
                        source: Some(var.clone()),
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
                    message: format!("Enter {}", self.name),
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

/// Always-configured auth without a key, for local servers.
#[derive(Clone, Debug)]
pub struct KeylessAuth {
    name: String,
}

impl ApiKeyAuth for KeylessAuth {
    fn name(&self) -> &str {
        &self.name
    }

    fn resolve<'a>(&'a self, input: ApiKeyInput<'a>) -> BoxFuture<'a, Result<Option<AuthResult>, ModelsError>> {
        Box::pin(async move {
            let credential = input.credential;
            Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: credential.and_then(|credential| credential.key.clone()),
                    ..ModelAuth::default()
                },
                env: credential.and_then(|credential| credential.env.clone()),
                source: None,
            }))
        })
    }
}

/// Overrides of one auth resolution.
#[derive(Clone, Debug, Default)]
pub struct AuthResolutionOverrides {
    pub api_key: Option<String>,
    pub env: Option<ProviderEnv>,
    /// Required remaining OAuth validity; at least five minutes.
    pub min_oauth_validity: Option<Duration>,
}

const DEFAULT_OAUTH_MINIMUM_VALIDITY_MS: i64 = 5 * 60 * 1000;
const DEFAULT_OAUTH_REFRESH_TIMEOUT: Duration = Duration::from_secs(15);

/// Resolve provider auth: an explicit key, else the stored credential (refreshing OAuth
/// when it expires within five minutes), else ambient sources.
pub async fn resolve_provider_auth(
    provider_id: &str,
    auth: &ProviderAuth,
    credentials: &Arc<dyn CredentialStore>,
    ctx: &dyn AuthContext,
    overrides: &AuthResolutionOverrides,
) -> Result<Option<AuthResult>, ModelsError> {
    let overlay;
    let ctx: &dyn AuthContext = match &overrides.env {
        Some(env) => {
            overlay = OverlayAuthContext { base: ctx, env };
            &overlay
        }
        None => ctx,
    };
    if let (Some(key), Some(api_key)) = (&overrides.api_key, &auth.api_key) {
        let credential = ApiKeyCredential {
            key: Some(key.clone()),
            env: overrides.env.clone(),
        };
        return resolve_api_key(api_key.as_ref(), provider_id, ctx, Some(&credential)).await;
    }
    let stored = read_credential(credentials.as_ref(), provider_id).await?;
    match stored {
        Some(Credential::OAuth(credential)) => match &auth.oauth {
            Some(oauth) => resolve_stored_oauth(credentials, provider_id, oauth, credential, overrides).await,
            None => Ok(None),
        },
        Some(Credential::ApiKey(mut credential)) => match &auth.api_key {
            Some(api_key) => {
                if let Some(env) = &overrides.env {
                    credential.env.get_or_insert_with(ProviderEnv::new).extend(env.clone());
                }
                resolve_api_key(api_key.as_ref(), provider_id, ctx, Some(&credential)).await
            }
            None => Ok(None),
        },
        None => match &auth.api_key {
            Some(api_key) => resolve_api_key(api_key.as_ref(), provider_id, ctx, None).await,
            None => Ok(None),
        },
    }
}

async fn resolve_api_key(
    auth: &dyn ApiKeyAuth,
    provider_id: &str,
    ctx: &dyn AuthContext,
    credential: Option<&ApiKeyCredential>,
) -> Result<Option<AuthResult>, ModelsError> {
    auth.resolve(ApiKeyInput { ctx, credential }).await.map_err(|error| {
        ModelsError::with_source(
            ModelsErrorCode::Auth,
            format!("API key auth failed for provider {provider_id}"),
            error,
        )
    })
}

pub(crate) async fn read_credential(
    credentials: &dyn CredentialStore,
    provider_id: &str,
) -> Result<Option<Credential>, ModelsError> {
    credentials.read(provider_id).await.map_err(|error| {
        ModelsError::with_source(
            ModelsErrorCode::Auth,
            format!("Credential store read failed for {provider_id}"),
            error,
        )
    })
}

/// Refresh a stored OAuth credential under the credential-store lock and persist it before
/// the lock is released. `needs_refresh` is re-checked under the lock, so concurrent callers
/// refresh only once.
///
/// The refresh runs in its own task: once started, the provider may already have rotated
/// the refresh token, so dropping the caller's future must not discard the result. It is
/// bounded by a 15 s timeout instead.
pub async fn refresh_stored_oauth_credential(
    credentials: Arc<dyn CredentialStore>,
    provider_id: String,
    oauth: Arc<dyn OAuthAuth>,
    needs_refresh: fn(&OAuthCredential, i64) -> bool,
    min_validity_ms: i64,
) -> Result<Option<OAuthCredential>, ModelsError> {
    let task = tokio::spawn(async move {
        let id = provider_id.clone();
        let modify: ModifyFn = Box::new(move |current| {
            Box::pin(async move {
                let Some(Credential::OAuth(current)) = current else {
                    return Ok(None);
                };
                if !needs_refresh(&current, min_validity_ms) {
                    return Ok(None);
                }
                match tokio::time::timeout(DEFAULT_OAUTH_REFRESH_TIMEOUT, oauth.refresh(current)).await {
                    Ok(Ok(fresh)) => Ok(Some(Credential::OAuth(fresh))),
                    Ok(Err(error)) => Err(ModelsError::with_source(
                        ModelsErrorCode::OAuth,
                        format!("OAuth refresh failed for {id}"),
                        error,
                    )),
                    Err(_) => Err(ModelsError::new(
                        ModelsErrorCode::OAuth,
                        format!("OAuth refresh timed out for {id}"),
                    )),
                }
            })
        });
        let post = credentials.modify(&provider_id, modify).await.map_err(|error| {
            if error.code() == ModelsErrorCode::OAuth {
                error
            } else {
                ModelsError::with_source(
                    ModelsErrorCode::Auth,
                    format!("Credential store modify failed for {provider_id}"),
                    error,
                )
            }
        })?;
        Ok(match post {
            Some(Credential::OAuth(credential)) => Some(credential),
            _ => None,
        })
    });
    task.await
        .map_err(|error| ModelsError::with_source(ModelsErrorCode::OAuth, "OAuth refresh task failed", error))?
}

fn expires_within(credential: &OAuthCredential, min_validity_ms: i64) -> bool {
    now_ms() + min_validity_ms >= credential.expires
}

async fn resolve_stored_oauth(
    credentials: &Arc<dyn CredentialStore>,
    provider_id: &str,
    oauth: &Arc<dyn OAuthAuth>,
    stored: OAuthCredential,
    overrides: &AuthResolutionOverrides,
) -> Result<Option<AuthResult>, ModelsError> {
    let requested = overrides.min_oauth_validity.map(|d| d.as_millis() as i64);
    let min_validity = DEFAULT_OAUTH_MINIMUM_VALIDITY_MS.max(requested.unwrap_or(0));
    let mut credential = stored;
    if expires_within(&credential, min_validity) {
        // The optimistic check said expired; the authoritative check runs under the lock.
        let Some(post) = refresh_stored_oauth_credential(
            credentials.clone(),
            provider_id.to_owned(),
            oauth.clone(),
            expires_within,
            min_validity,
        )
        .await?
        else {
            return Ok(None); // logged out meanwhile
        };
        credential = post;
        if requested.is_some() && expires_within(&credential, min_validity) {
            return Err(ModelsError::new(
                ModelsErrorCode::OAuth,
                format!("OAuth refresh returned a token that expires too soon for {provider_id}"),
            ));
        }
    }
    let auth = oauth.to_auth(&credential).await.map_err(|error| {
        ModelsError::with_source(
            ModelsErrorCode::OAuth,
            format!("OAuth auth derivation failed for {provider_id}"),
            error,
        )
    })?;
    Ok(Some(AuthResult {
        auth,
        env: None,
        source: Some("OAuth".into()),
    }))
}

/// Merge header maps; names compare case-insensitively and `overrides` win.
pub fn merge_headers(base: Option<&ProviderHeaders>, overrides: Option<&ProviderHeaders>) -> Option<ProviderHeaders> {
    if base.is_none() && overrides.is_none() {
        return None;
    }
    let mut merged = base.cloned().unwrap_or_default();
    for (name, value) in overrides.into_iter().flatten() {
        merged.retain(|existing, _| !existing.eq_ignore_ascii_case(name));
        merged.insert(name.clone(), value.clone());
    }
    Some(merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TestOAuth {
        refreshes: AtomicUsize,
    }

    impl OAuthAuth for TestOAuth {
        fn name(&self) -> &'static str {
            "test"
        }
        fn login<'a>(
            &'a self,
            _: &'a dyn AuthInteraction,
            _: &'a LoginOptions,
        ) -> BoxFuture<'a, Result<OAuthCredential, ModelsError>> {
            Box::pin(async { Err(ModelsError::new(ModelsErrorCode::OAuth, "no")) })
        }
        fn refresh(&self, credential: OAuthCredential) -> BoxFuture<'_, Result<OAuthCredential, ModelsError>> {
            self.refreshes.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                tokio::time::sleep(Duration::from_millis(10)).await;
                Ok(OAuthCredential {
                    access: "fresh".into(),
                    expires: now_ms() + 3_600_000,
                    ..credential
                })
            })
        }
        fn to_auth<'a>(&'a self, credential: &'a OAuthCredential) -> BoxFuture<'a, Result<ModelAuth, ModelsError>> {
            Box::pin(async move {
                Ok(ModelAuth {
                    api_key: Some(credential.access.clone()),
                    ..ModelAuth::default()
                })
            })
        }
    }

    #[tokio::test]
    async fn concurrent_requests_refresh_once() {
        let store: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
        store
            .modify(
                "p",
                Box::new(|_| {
                    Box::pin(async {
                        Ok(Some(Credential::OAuth(OAuthCredential {
                            refresh: "r".into(),
                            access: "old".into(),
                            expires: 0,
                            extra: JsonObject::new(),
                        })))
                    })
                }),
            )
            .await
            .unwrap();
        let oauth = Arc::new(TestOAuth {
            refreshes: AtomicUsize::new(0),
        });
        let auth = ProviderAuth {
            api_key: None,
            oauth: Some(oauth.clone()),
        };
        let overrides = AuthResolutionOverrides::default();
        let (a, b) = tokio::join!(
            resolve_provider_auth("p", &auth, &store, &DefaultAuthContext, &overrides),
            resolve_provider_auth("p", &auth, &store, &DefaultAuthContext, &overrides),
        );
        assert_eq!(a.unwrap().unwrap().auth.api_key.as_deref(), Some("fresh"));
        assert_eq!(b.unwrap().unwrap().auth.api_key.as_deref(), Some("fresh"));
        assert_eq!(oauth.refreshes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn stored_key_wins_over_env_and_explicit_key_wins_over_stored() {
        let store: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
        let auth = ProviderAuth::api_key(env_api_key_auth("Test key", &["DELTA_TEST_UNSET_KEY_VAR"]));
        let none = resolve_provider_auth(
            "p",
            &auth,
            &store,
            &DefaultAuthContext,
            &AuthResolutionOverrides::default(),
        )
        .await
        .unwrap();
        assert!(none.is_none());
        store
            .modify(
                "p",
                Box::new(|_| {
                    Box::pin(async {
                        Ok(Some(Credential::ApiKey(ApiKeyCredential {
                            key: Some("s".into()),
                            env: None,
                        })))
                    })
                }),
            )
            .await
            .unwrap();
        let stored = resolve_provider_auth(
            "p",
            &auth,
            &store,
            &DefaultAuthContext,
            &AuthResolutionOverrides::default(),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(stored.auth.api_key.as_deref(), Some("s"));
        let explicit = AuthResolutionOverrides {
            api_key: Some("x".into()),
            ..AuthResolutionOverrides::default()
        };
        let result = resolve_provider_auth("p", &auth, &store, &DefaultAuthContext, &explicit)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.auth.api_key.as_deref(), Some("x"));
    }

    #[test]
    fn headers_merge_case_insensitively() {
        let base: ProviderHeaders = [("Authorization".to_owned(), Some("a".to_owned()))]
            .into_iter()
            .collect();
        let over: ProviderHeaders = [("authorization".to_owned(), None)].into_iter().collect();
        let merged = merge_headers(Some(&base), Some(&over)).unwrap();
        assert_eq!(merged.len(), 1);
        assert_eq!(merged.get("authorization"), Some(&None));
    }
}
