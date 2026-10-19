//! OAuth login flows of the built-in providers and the helpers they share. Port of
//! upstream `src/auth/oauth/` and `src/utils/oauth-page.ts`.
//!
//! Shared helpers, for porting further flows one-to-one:
//!
//! - [`pkce`]: [`generate_pkce`] (upstream `generatePKCE`).
//! - [`callback_server`]: [`start_oauth_callback_server`] and
//!   [`wait_for_callback_or_manual_input`] (upstream `startOAuthCallbackServer`,
//!   `waitForCallbackOrManualInput`).
//! - [`device_code`]: [`poll_oauth_device_code_flow`] and [`abortable_sleep`] (upstream
//!   `pollOAuthDeviceCodeFlow`, `abortableSleep`).
//! - [`page`]: [`oauth_success_html`] and [`oauth_error_html`].
//! - [`oauth_callback_host`]: `PI_OAUTH_CALLBACK_HOST` or `127.0.0.1`.
//!
//! Flows: [`anthropic`] (Claude Pro/Max), [`openrouter`] (PKCE that mints an API key),
//! [`xai`], [`kimi_coding`] and [`meta`] (device code), [`github_copilot`] (GitHub device code
//! plus the Copilot token exchange), [`radius`] (browser or device code against a Radius
//! gateway), [`openai_codex`] (ChatGPT Plus/Pro for the Codex backend: browser or device code)
//! and [`openai_chatgpt`] (Sign in with ChatGPT for the OpenAI API). Each flow is a struct
//! implementing [`crate::auth::OAuthAuth`]; `Default` (for Radius, [`create_radius_oauth`]) uses
//! upstream's endpoints, and the endpoint fields are public so tests (and proxies) can point them
//! elsewhere.
//!
//! Flow errors are [`ModelsError`]s: cancellation is `Login cancelled` with code
//! [`ModelsErrorCode::Aborted`]; everything else has code [`ModelsErrorCode::OAuth`] and
//! upstream's message text.

pub mod anthropic;
pub mod callback_server;
pub mod device_code;
pub mod github_copilot;
pub mod kimi_coding;
pub mod meta;
pub mod openai_chatgpt;
pub mod openai_codex;
pub mod openrouter;
pub mod page;
pub mod pkce;
pub mod radius;
pub mod xai;

pub use anthropic::AnthropicOAuth;
pub use callback_server::{
    CallbackOrManual, OAuthCallbackComplete, OAuthCallbackServer, OAuthCallbackServerOptions,
    start_oauth_callback_server, wait_for_callback_or_manual_input,
};
pub use device_code::{
    CANCEL_MESSAGE, OAuthDeviceCodePollOptions, OAuthDeviceCodePollResult, SLOW_DOWN_TIMEOUT_MESSAGE, TIMEOUT_MESSAGE,
    abortable_sleep, poll_oauth_device_code_flow,
};
pub use github_copilot::GitHubCopilotOAuth;
pub use kimi_coding::KimiCodingOAuth;
pub use meta::MetaOAuth;
pub use openai_chatgpt::OpenAIChatGPTOAuth;
pub use openai_codex::OpenAICodexOAuth;
pub use openrouter::OpenRouterOAuth;
pub use page::{oauth_error_html, oauth_success_html};
pub use pkce::{Pkce, generate_pkce};
pub use radius::{RadiusOAuth, RadiusOAuthOptions, create_radius_oauth};
pub use xai::XaiOAuth;

use crate::auth::env_value;
use crate::error::{ModelsError, ModelsErrorCode};

/// Environment variable that overrides the loopback callback host.
pub const OAUTH_CALLBACK_HOST_ENV: &str = "PI_OAUTH_CALLBACK_HOST";

/// The loopback callback host: `PI_OAUTH_CALLBACK_HOST` when set, else `127.0.0.1`
/// (upstream `getProviderEnvValue("PI_OAUTH_CALLBACK_HOST") || "127.0.0.1"`).
pub fn oauth_callback_host() -> String {
    env_value(None, OAUTH_CALLBACK_HOST_ENV).unwrap_or_else(|| "127.0.0.1".to_owned())
}

/// `Login cancelled` with code [`ModelsErrorCode::Aborted`].
pub fn login_cancelled() -> ModelsError {
    ModelsError::new(ModelsErrorCode::Aborted, CANCEL_MESSAGE)
}

/// An OAuth flow error (code [`ModelsErrorCode::OAuth`]) with exactly `message`.
pub fn oauth_error(message: impl Into<String>) -> ModelsError {
    ModelsError::new(ModelsErrorCode::OAuth, message)
}

/// Encode `pairs` as `application/x-www-form-urlencoded` (upstream `URLSearchParams`
/// serialization: spaces become `+`).
pub fn form_urlencode<'a>(pairs: impl IntoIterator<Item = (&'a str, &'a str)>) -> String {
    let mut serializer = url::form_urlencoded::Serializer::new(String::new());
    for (name, value) in pairs {
        serializer.append_pair(name, value);
    }
    serializer.finish()
}

/// The first value of `name` in a URL query or form body (upstream
/// `URLSearchParams.get`). A leading `?` is ignored.
pub fn search_param(query: &str, name: &str) -> Option<String> {
    let query = query.strip_prefix('?').unwrap_or(query);
    url::form_urlencoded::parse(query.as_bytes())
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.into_owned())
}

/// The HTTP client of OAuth requests: the runtime's default client of [`crate::http`]
/// (delta user agent, 30 s connect timeout, system proxy).
pub fn oauth_http_client() -> reqwest::Client {
    crate::http::client(&crate::options::StreamOptions::default())
}

/// An error's message followed by `; cause=<message>` for each error in its `source()`
/// chain (the Rust counterpart of upstream Anthropic's `formatErrorDetails`, without the
/// JavaScript error name and stack).
pub fn format_error_details(error: &(dyn std::error::Error + 'static)) -> String {
    let mut details = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        details.push_str("; cause=");
        details.push_str(&cause.to_string());
        source = cause.source();
    }
    details
}

/// The first value of `name` in `url`'s query (upstream `url.searchParams.get(name)`).
pub fn url_search_param(url: &url::Url, name: &str) -> Option<String> {
    url.query_pairs()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.into_owned())
}
