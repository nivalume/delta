//! Loopback OAuth redirect handler shared by the browser sign-in flows. Port of upstream
//! `auth/oauth/callback-server.ts`.
//!
//! The server is a minimal HTTP/1.1 handler on a Tokio [`TcpListener`]: it reads one
//! request head per connection, answers with an HTML page from [`crate::oauth::page`] and
//! closes the connection. Only `GET <path>` is served.

use futures::future::BoxFuture;
use std::fmt;
use std::future::Future;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::auth::{AuthInteraction, AuthPrompt};
use crate::error::{ModelsError, ModelsErrorCode};
use crate::oauth::device_code::CANCEL_MESSAGE;
use crate::oauth::page::{oauth_error_html, oauth_success_html};
use crate::oauth::url_search_param;

/// Largest request head the server reads before answering `400`.
const MAX_REQUEST_HEAD_BYTES: usize = 64 * 1024;

/// Finishes the sign-in with the authorization code (upstream `complete`).
pub type OAuthCallbackComplete<T> = Arc<dyn Fn(String) -> BoxFuture<'static, Result<T, ModelsError>> + Send + Sync>;

/// Options of [`start_oauth_callback_server`] (upstream `OAuthCallbackServerOptions<T>`).
pub struct OAuthCallbackServerOptions<T> {
    /// Provider name used on the browser page and in errors, for example `OpenAI`.
    pub provider_name: String,
    /// Address to listen on, for example `127.0.0.1`.
    pub host: String,
    /// Port to listen on; `0` picks a free port.
    pub port: u16,
    /// Callback path, for example `/callback`.
    pub path: String,
    /// Host in `redirect_uri` when it differs from `host`, for example `localhost`.
    pub redirect_host: Option<String>,
    /// Expected `state` parameter. `None` when the provider does not send one.
    pub state: Option<String>,
    /// Finishes the sign-in with the received code before the browser page is sent, so the
    /// page can show exchange failures. Pass `|code| async move { Ok(code) }` to exchange
    /// the code later.
    pub complete: OAuthCallbackComplete<T>,
    /// Cancels the wait with `Login cancelled` (upstream `signal`).
    pub cancel: Option<CancellationToken>,
    /// Fails the wait with `<provider_name> sign-in timed out` after this long.
    pub timeout: Option<Duration>,
}

impl<T> OAuthCallbackServerOptions<T> {
    /// Options with the required fields; `redirect_host`, `state`, `cancel` and `timeout`
    /// start unset.
    pub fn new<F, Fut>(
        provider_name: impl Into<String>,
        host: impl Into<String>,
        port: u16,
        path: impl Into<String>,
        complete: F,
    ) -> Self
    where
        F: Fn(String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<T, ModelsError>> + Send + 'static,
    {
        Self {
            provider_name: provider_name.into(),
            host: host.into(),
            port,
            path: path.into(),
            redirect_host: None,
            state: None,
            complete: Arc::new(move |code| Box::pin(complete(code))),
            cancel: None,
            timeout: None,
        }
    }

    #[must_use]
    pub fn redirect_host(mut self, redirect_host: impl Into<String>) -> Self {
        self.redirect_host = Some(redirect_host.into());
        self
    }

    #[must_use]
    pub fn state(mut self, state: impl Into<String>) -> Self {
        self.state = Some(state.into());
        self
    }

    #[must_use]
    pub fn cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = Some(cancel);
        self
    }

    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

impl<T> fmt::Debug for OAuthCallbackServerOptions<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OAuthCallbackServerOptions")
            .field("provider_name", &self.provider_name)
            .field("host", &self.host)
            .field("port", &self.port)
            .field("path", &self.path)
            .field("redirect_host", &self.redirect_host)
            .field("has_state", &self.state.is_some())
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

type WaitResult<T> = Result<Option<T>, ModelsError>;

struct Settlement<T> {
    claimed: bool,
    settled: bool,
    sender: Option<oneshot::Sender<WaitResult<T>>>,
}

struct Shared<T> {
    provider_name: String,
    path: String,
    state: Option<String>,
    complete: OAuthCallbackComplete<T>,
    settlement: Mutex<Settlement<T>>,
    /// Fires when the wait settles, so the timeout/cancel watcher stops.
    settled: CancellationToken,
}

impl<T> Shared<T> {
    fn lock(&self) -> std::sync::MutexGuard<'_, Settlement<T>> {
        self.settlement.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn finish(&self, result: WaitResult<T>) {
        let sender = {
            let mut settlement = self.lock();
            if settlement.settled {
                return;
            }
            settlement.settled = true;
            settlement.sender.take()
        };
        self.settled.cancel();
        if let Some(sender) = sender {
            let _ = sender.send(result);
        }
    }
}

/// A running loopback callback server (upstream `OAuthCallbackServer<T>`).
///
/// Dropping it closes it, like [`Self::close`].
pub struct OAuthCallbackServer<T> {
    redirect_uri: String,
    shared: Arc<Shared<T>>,
    receiver: Mutex<Option<oneshot::Receiver<WaitResult<T>>>>,
    shutdown: CancellationToken,
}

impl<T> fmt::Debug for OAuthCallbackServer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OAuthCallbackServer")
            .field("redirect_uri", &self.redirect_uri)
            .finish_non_exhaustive()
    }
}

impl<T: Send + 'static> OAuthCallbackServer<T> {
    /// `http://<redirect_host or host>:<bound port><path>`; IPv6 hosts are bracketed.
    pub fn redirect_uri(&self) -> &str {
        &self.redirect_uri
    }

    /// The result of `complete`, or `Ok(None)` after [`Self::cancel`]. Fails when the
    /// provider redirects with an error, `complete` fails, the cancel token fires, the
    /// timeout elapses or the server is closed.
    ///
    /// The result can be taken once; a second call fails.
    pub async fn wait(&self) -> Result<Option<T>, ModelsError> {
        let receiver = self.receiver.lock().unwrap_or_else(PoisonError::into_inner).take();
        let Some(receiver) = receiver else {
            return Err(ModelsError::new(
                ModelsErrorCode::OAuth,
                "OAuth callback result was already taken",
            ));
        };
        receiver.await.unwrap_or_else(|_| Err(closed_error()))
    }

    /// Stop waiting for the browser unless a callback is already being completed.
    pub fn cancel(&self) {
        let claimed = self.shared.lock().claimed;
        if !claimed {
            self.shared.finish(Ok(None));
        }
    }

    /// Stop listening. A pending wait fails with `OAuth callback server closed`.
    pub fn close(&self) {
        self.shared.finish(Err(closed_error()));
        self.shutdown.cancel();
    }
}

impl<T> Drop for OAuthCallbackServer<T> {
    fn drop(&mut self) {
        // `close()` without the `T: Send` bound.
        self.shared.finish(Err(closed_error()));
        self.shutdown.cancel();
    }
}

fn closed_error() -> ModelsError {
    ModelsError::new(ModelsErrorCode::OAuth, "OAuth callback server closed")
}

fn cancelled_error() -> ModelsError {
    ModelsError::new(ModelsErrorCode::Aborted, CANCEL_MESSAGE)
}

/// Start a loopback callback server and return once it listens.
///
/// Fails with `Login cancelled` when `options.cancel` already fired, and with the bind
/// error (its [`std::io::Error`] is the error's `source()`) when the port is taken; it
/// never falls back to another port. Must be called inside a Tokio runtime.
pub async fn start_oauth_callback_server<T: Send + 'static>(
    options: OAuthCallbackServerOptions<T>,
) -> Result<OAuthCallbackServer<T>, ModelsError> {
    if options.cancel.as_ref().is_some_and(CancellationToken::is_cancelled) {
        return Err(cancelled_error());
    }
    let listener = TcpListener::bind((options.host.as_str(), options.port))
        .await
        .map_err(|error| {
            ModelsError::with_source(
                ModelsErrorCode::OAuth,
                format!(
                    "OAuth callback server failed to listen on {}:{}",
                    options.host, options.port
                ),
                error,
            )
        })?;
    let port = listener
        .local_addr()
        .map_err(|error| {
            ModelsError::with_source(
                ModelsErrorCode::OAuth,
                "OAuth callback server did not bind to TCP",
                error,
            )
        })?
        .port();

    let (sender, receiver) = oneshot::channel();
    let shared = Arc::new(Shared {
        provider_name: options.provider_name,
        path: options.path,
        state: options.state,
        complete: options.complete,
        settlement: Mutex::new(Settlement {
            claimed: false,
            settled: false,
            sender: Some(sender),
        }),
        settled: CancellationToken::new(),
    });
    let shutdown = CancellationToken::new();

    tokio::spawn(accept_loop(listener, shared.clone(), shutdown.clone()));

    if options.cancel.is_some() || options.timeout.is_some() {
        let shared = shared.clone();
        let cancel = options.cancel.unwrap_or_default();
        let timeout = options.timeout;
        tokio::spawn(async move {
            let timer = async {
                match timeout {
                    Some(timeout) => tokio::time::sleep(timeout).await,
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                () = shared.settled.cancelled() => {}
                () = cancel.cancelled() => shared.finish(Err(cancelled_error())),
                () = timer => {
                    let message = format!("{} sign-in timed out", shared.provider_name);
                    shared.finish(Err(ModelsError::new(ModelsErrorCode::OAuth, message)));
                }
            }
        });
    }

    let redirect_host = options.redirect_host.unwrap_or(options.host);
    let redirect_host = if redirect_host.contains(':') {
        format!("[{redirect_host}]")
    } else {
        redirect_host
    };
    Ok(OAuthCallbackServer {
        redirect_uri: format!("http://{redirect_host}:{port}{}", shared.path),
        shared,
        receiver: Mutex::new(Some(receiver)),
        shutdown,
    })
}

async fn accept_loop<T: Send + 'static>(listener: TcpListener, shared: Arc<Shared<T>>, shutdown: CancellationToken) {
    loop {
        let accepted = tokio::select! {
            biased;
            () = shutdown.cancelled() => break,
            accepted = listener.accept() => accepted,
        };
        match accepted {
            // A connection accepted after `close()` is dropped unanswered.
            Ok(_) if shutdown.is_cancelled() => break,
            Ok((stream, _)) => {
                tokio::spawn(handle_connection(stream, shared.clone()));
            }
            Err(error) => {
                shared.finish(Err(ModelsError::with_source(
                    ModelsErrorCode::OAuth,
                    "OAuth callback server failed",
                    error,
                )));
                break;
            }
        }
    }
}

/// The request line's method and target, or `None` for a malformed or oversized head.
pub(crate) async fn read_request_head(stream: &mut TcpStream) -> Option<(String, String)> {
    let mut head = Vec::with_capacity(1024);
    let mut chunk = [0u8; 4096];
    loop {
        if head.windows(4).any(|window| window == b"\r\n\r\n") || head.windows(2).any(|window| window == b"\n\n") {
            break;
        }
        if head.len() > MAX_REQUEST_HEAD_BYTES {
            return None;
        }
        let read = stream.read(&mut chunk).await.ok()?;
        if read == 0 {
            break;
        }
        head.extend_from_slice(&chunk[..read]);
    }
    let head = String::from_utf8_lossy(&head);
    let mut parts = head.lines().next()?.split_whitespace();
    let method = parts.next()?.to_owned();
    let target = parts.next()?.to_owned();
    Some((method, target))
}

pub(crate) async fn send_page(stream: &mut TcpStream, status: u16, html: &str) {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        500 => "Internal Server Error",
        _ => "Bad Gateway",
    };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: text/html; charset=utf-8\r\ncache-control: no-store\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{html}",
        html.len()
    );
    let _ = stream.write_all(response.as_bytes()).await;
    let _ = stream.shutdown().await;
}

enum Claim {
    Code(String),
    Handled,
    ProviderError(String),
    MissingCode,
}

/// Claim the sign-in for the callback's code, under the settlement lock.
fn claim<T>(shared: &Shared<T>, url: &url::Url) -> Claim {
    let mut settlement = shared.lock();
    if settlement.claimed || settlement.settled {
        return Claim::Handled;
    }
    if let Some(error) = url_search_param(url, "error").filter(|error| !error.is_empty()) {
        return Claim::ProviderError(url_search_param(url, "error_description").unwrap_or(error));
    }
    let Some(code) = url_search_param(url, "code").filter(|code| !code.is_empty()) else {
        return Claim::MissingCode;
    };
    settlement.claimed = true;
    Claim::Code(code)
}

async fn handle_connection<T: Send + 'static>(mut stream: TcpStream, shared: Arc<Shared<T>>) {
    let Some((method, target)) = read_request_head(&mut stream).await else {
        send_page(&mut stream, 400, &oauth_error_html("Malformed request.", None)).await;
        return;
    };
    let url = url::Url::parse("http://localhost/").and_then(|base| base.join(&target));
    let url = match url {
        Ok(url) if method == "GET" && url.path() == shared.path => url,
        _ => {
            send_page(&mut stream, 404, &oauth_error_html("Callback route not found.", None)).await;
            return;
        }
    };
    if let Some(expected) = &shared.state
        && url_search_param(&url, "state").as_deref() != Some(expected.as_str())
    {
        send_page(&mut stream, 400, &oauth_error_html("State mismatch.", None)).await;
        return;
    }
    let provider = shared.provider_name.as_str();
    let code = match claim(&shared, &url) {
        Claim::Code(code) => code,
        Claim::Handled => {
            let html = oauth_error_html("This sign-in has already been handled.", None);
            send_page(&mut stream, 409, &html).await;
            return;
        }
        Claim::ProviderError(description) => {
            let html = oauth_error_html(&format!("{provider} authorization failed."), Some(&description));
            send_page(&mut stream, 400, &html).await;
            shared.finish(Err(ModelsError::new(
                ModelsErrorCode::OAuth,
                format!("{provider} authorization failed: {description}"),
            )));
            return;
        }
        Claim::MissingCode => {
            send_page(&mut stream, 400, &oauth_error_html("Missing authorization code.", None)).await;
            return;
        }
    };
    match (shared.complete)(code).await {
        Ok(value) => {
            let html = oauth_success_html(&format!("Signed in to {provider}. You may now close this page."));
            send_page(&mut stream, 200, &html).await;
            shared.finish(Ok(Some(value)));
        }
        Err(error) => {
            let html = oauth_error_html(&format!("{provider} sign-in failed."), Some(error.message()));
            send_page(&mut stream, 502, &html).await;
            shared.finish(Err(error));
        }
    }
}

/// Result of [`wait_for_callback_or_manual_input`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallbackOrManual<T> {
    /// The browser reached the callback server; the value `complete` returned.
    Callback(T),
    /// The user pasted a code or redirect URL.
    Manual(String),
}

/// Wait for the browser callback, or for the user to paste the code or redirect URL when
/// the browser cannot reach the loopback server (for example over SSH). Without a
/// callback server only the manual prompt is used.
///
/// The manual prompt is an [`AuthPrompt::ManualCode`] whose `cancel` token fires once this
/// function returns, so UIs can dismiss it. When the prompt answers first, the callback
/// server is cancelled; a callback that is already being completed still wins. A failed
/// prompt fails the wait; a failed callback wait fails it too.
pub async fn wait_for_callback_or_manual_input<T: Send + 'static>(
    interaction: &dyn AuthInteraction,
    callback: Option<&OAuthCallbackServer<T>>,
    message: impl Into<String>,
    placeholder: impl Into<String>,
) -> Result<CallbackOrManual<T>, ModelsError> {
    let manual_cancel = CancellationToken::new();
    let _abort_manual = manual_cancel.clone().drop_guard();
    let mut manual = interaction.prompt(AuthPrompt::ManualCode {
        message: message.into(),
        placeholder: Some(placeholder.into()),
        cancel: manual_cancel,
    });
    let Some(callback) = callback else {
        return manual.await.map(CallbackOrManual::Manual);
    };

    let wait = callback.wait();
    tokio::pin!(wait);
    let mut manual_result: Option<Result<String, ModelsError>> = None;
    let value = loop {
        tokio::select! {
            value = &mut wait => break value?,
            result = &mut manual, if manual_result.is_none() => {
                callback.cancel();
                manual_result = Some(result);
            }
        }
    };
    if let Some(Err(error)) = manual_result {
        return Err(error);
    }
    if let Some(value) = value {
        return Ok(CallbackOrManual::Callback(value));
    }
    let input = match manual_result {
        Some(result) => result?,
        None => manual.await?,
    };
    Ok(CallbackOrManual::Manual(input))
}
