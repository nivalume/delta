//! Tests of the shared OAuth helpers: the loopback callback server, the race against a
//! pasted code, and device-code polling. Ported from upstream
//! `test/oauth-callback-server.test.ts` and `test/oauth-device-code.test.ts`.
//!
//! Callback tests use real loopback connections; device-code tests run on paused Tokio time.

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

use delta_ai::auth::{AuthEvent, AuthInteraction, AuthPrompt};
use delta_ai::oauth::{
    CallbackOrManual, OAuthCallbackServer, OAuthCallbackServerOptions, OAuthDeviceCodePollOptions,
    OAuthDeviceCodePollResult, SLOW_DOWN_TIMEOUT_MESSAGE, TIMEOUT_MESSAGE, oauth_callback_host,
    poll_oauth_device_code_flow, start_oauth_callback_server, wait_for_callback_or_manual_input,
};
use delta_ai::{ModelsError, ModelsErrorCode};
use futures::future::BoxFuture;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn http() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

fn callback_url(redirect_uri: &str, params: &[(&str, &str)]) -> String {
    let mut url = url::Url::parse(redirect_uri).unwrap();
    for (name, value) in params {
        url.query_pairs_mut().append_pair(name, value);
    }
    url.to_string()
}

struct Page {
    status: u16,
    content_type: Option<String>,
    body: String,
}

async fn get(url: &str) -> Page {
    let response = http().get(url).send().await.unwrap();
    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get("content-type")
        .map(|value| value.to_str().unwrap().to_owned());
    Page {
        status,
        content_type,
        body: response.text().await.unwrap(),
    }
}

fn options(
    complete: impl Fn(String) -> BoxFuture<'static, Result<String, ModelsError>> + Send + Sync + 'static,
) -> OAuthCallbackServerOptions<String> {
    OAuthCallbackServerOptions::new("Example", "127.0.0.1", 0, "/callback", complete).state("expected-state")
}

fn completed(code: String) -> BoxFuture<'static, Result<String, ModelsError>> {
    Box::pin(async move { Ok(format!("completed:{code}")) })
}

async fn start(options: OAuthCallbackServerOptions<String>) -> OAuthCallbackServer<String> {
    start_oauth_callback_server(options).await.unwrap()
}

type PromptFn = Box<dyn Fn(AuthPrompt) -> BoxFuture<'static, Result<String, ModelsError>> + Send + Sync>;

struct Interaction {
    prompt: PromptFn,
}

impl AuthInteraction for Interaction {
    fn prompt(&self, prompt: AuthPrompt) -> BoxFuture<'_, Result<String, ModelsError>> {
        (self.prompt)(prompt)
    }

    fn notify(&self, _event: AuthEvent) {}
}

/// A manual prompt that stays open until its cancel token fires.
fn pending_prompt(seen: Arc<Mutex<Option<CancellationToken>>>) -> PromptFn {
    Box::new(move |prompt| {
        let AuthPrompt::ManualCode { cancel, .. } = prompt else {
            panic!("unexpected prompt");
        };
        *seen.lock().unwrap() = Some(cancel.clone());
        Box::pin(async move {
            cancel.cancelled().await;
            Err(ModelsError::new(ModelsErrorCode::Aborted, "prompt aborted"))
        })
    })
}

fn answer(text: &'static str) -> PromptFn {
    Box::new(move |_| Box::pin(async move { Ok(text.to_owned()) }))
}

// ---------------------------------------------------------------------------
// OAuth callback server
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ignores_stray_requests_and_resolves_with_the_completed_code() {
    let server = start(options(completed)).await;
    let redirect = server.redirect_uri().to_owned();
    let pattern = redirect.strip_prefix("http://127.0.0.1:").unwrap();
    assert!(pattern.ends_with("/callback"));
    assert!(pattern.trim_end_matches("/callback").parse::<u16>().is_ok());

    let base = url::Url::parse(&redirect).unwrap();
    assert_eq!(get(base.join("/other").unwrap().as_str()).await.status, 404);
    let wrong_state = get(&callback_url(&redirect, &[("code", "c"), ("state", "other")])).await;
    assert_eq!(wrong_state.status, 400);
    assert_eq!(wrong_state.content_type.as_deref(), Some("text/html; charset=utf-8"));
    assert!(wrong_state.body.contains("State mismatch."));
    let post = http()
        .post(callback_url(&redirect, &[("code", "c"), ("state", "expected-state")]))
        .send()
        .await
        .unwrap();
    assert_eq!(post.status().as_u16(), 404);
    let missing_code = get(&callback_url(&redirect, &[("state", "expected-state")])).await;
    assert_eq!(missing_code.status, 400);

    let success = get(&callback_url(
        &redirect,
        &[("code", "the-code"), ("state", "expected-state")],
    ))
    .await;
    assert_eq!(success.status, 200);
    assert_eq!(success.content_type.as_deref(), Some("text/html; charset=utf-8"));
    assert!(success.body.contains("Authentication successful"));
    assert!(success.body.contains("Signed in to Example."));
    assert!(success.body.contains(r##"fill="#F09082""##));
    assert!(success.body.contains(r##"fill="#4D9ABF""##));
    assert!(success.body.contains(r##"fill="#F1BE58""##));
    assert_eq!(server.wait().await.unwrap().as_deref(), Some("completed:the-code"));
    server.close();
}

#[tokio::test]
async fn uses_the_redirect_host_and_skips_the_state_check_when_none_is_expected() {
    let mut opts = options(completed).redirect_host("localhost");
    opts.state = None;
    let server = start(opts).await;
    let redirect = server.redirect_uri().to_owned();
    assert!(redirect.starts_with("http://localhost:"));
    assert!(redirect.ends_with("/callback"));
    let url = callback_url(&redirect, &[("code", "no-state")]).replace("localhost", "127.0.0.1");
    assert_eq!(get(&url).await.status, 200);
    assert_eq!(server.wait().await.unwrap().as_deref(), Some("completed:no-state"));
}

#[tokio::test]
async fn shows_completion_failures_on_the_page_and_rejects_the_wait() {
    let server = start(options(|_| {
        Box::pin(async { Err(ModelsError::new(ModelsErrorCode::OAuth, "token exchange failed")) })
    }))
    .await;
    let failure = get(&callback_url(
        server.redirect_uri(),
        &[("code", "c"), ("state", "expected-state")],
    ))
    .await;
    assert_eq!(failure.status, 502);
    assert!(failure.body.contains("Example sign-in failed."));
    assert!(failure.body.contains("token exchange failed"));
    let error = server.wait().await.unwrap_err();
    assert!(error.message().contains("token exchange failed"));
}

#[tokio::test]
async fn rejects_the_wait_when_the_provider_redirects_with_an_error() {
    let server = start(options(completed)).await;
    let failure = get(&callback_url(
        server.redirect_uri(),
        &[
            ("error", "access_denied"),
            ("error_description", "User denied access"),
            ("state", "expected-state"),
        ],
    ))
    .await;
    assert_eq!(failure.status, 400);
    assert!(failure.body.contains("User denied access"));
    let error = server.wait().await.unwrap_err();
    assert_eq!(error.message(), "Example authorization failed: User denied access");
}

#[tokio::test]
async fn completes_only_the_first_callback() {
    let release = CancellationToken::new();
    let started = CancellationToken::new();
    let (release_in, started_in) = (release.clone(), started.clone());
    let server = start(options(move |_| {
        let (release, started) = (release_in.clone(), started_in.clone());
        Box::pin(async move {
            started.cancel();
            release.cancelled().await;
            Ok("done".to_owned())
        })
    }))
    .await;
    let url = callback_url(server.redirect_uri(), &[("code", "c"), ("state", "expected-state")]);
    let first = tokio::spawn({
        let url = url.clone();
        async move { get(&url).await }
    });
    started.cancelled().await;
    assert_eq!(get(&url).await.status, 409);
    // A claimed callback keeps completing even when the caller switches to manual input.
    server.cancel();
    release.cancel();
    assert_eq!(first.await.unwrap().status, 200);
    assert_eq!(server.wait().await.unwrap().as_deref(), Some("done"));
}

#[tokio::test]
async fn resolves_with_none_after_cancel() {
    let server = start(options(completed)).await;
    server.cancel();
    assert_eq!(server.wait().await.unwrap(), None);
    let late = get(&callback_url(
        server.redirect_uri(),
        &[("code", "c"), ("state", "expected-state")],
    ))
    .await;
    assert_eq!(late.status, 409);
}

#[tokio::test]
async fn rejects_the_wait_on_cancel_and_on_timeout() {
    let cancel = CancellationToken::new();
    let aborted = start(options(completed).cancel(cancel.clone())).await;
    cancel.cancel();
    let error = aborted.wait().await.unwrap_err();
    assert_eq!(error.message(), "Login cancelled");
    assert_eq!(error.code(), ModelsErrorCode::Aborted);

    let timed_out = start(options(completed).timeout(Duration::from_millis(10))).await;
    assert_eq!(
        timed_out.wait().await.unwrap_err().message(),
        "Example sign-in timed out"
    );

    let already = CancellationToken::new();
    already.cancel();
    let error = start_oauth_callback_server(options(completed).cancel(already))
        .await
        .unwrap_err();
    assert_eq!(error.message(), "Login cancelled");
}

#[tokio::test]
async fn fails_instead_of_picking_another_port_when_the_requested_port_is_taken() {
    let blocker = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = blocker.local_addr().unwrap().port();
    let mut opts = options(completed);
    opts.port = port;
    let error = start_oauth_callback_server(opts).await.unwrap_err();
    let io = std::error::Error::source(&error)
        .and_then(|source| source.downcast_ref::<std::io::Error>())
        .unwrap();
    assert_eq!(io.kind(), std::io::ErrorKind::AddrInUse);
}

#[tokio::test]
async fn refuses_connections_after_close() {
    let server = start(options(completed)).await;
    let url = callback_url(server.redirect_uri(), &[("code", "c"), ("state", "expected-state")]);
    server.close();
    assert_eq!(
        server.wait().await.unwrap_err().message(),
        "OAuth callback server closed"
    );
    assert!(http().get(&url).send().await.is_err());
}

#[test]
fn callback_host_defaults_to_ipv4_loopback() {
    if std::env::var_os("PI_OAUTH_CALLBACK_HOST").is_none() {
        assert_eq!(oauth_callback_host(), "127.0.0.1");
    }
}

// ---------------------------------------------------------------------------
// wait_for_callback_or_manual_input
// ---------------------------------------------------------------------------

fn plain_server_options() -> OAuthCallbackServerOptions<String> {
    OAuthCallbackServerOptions::new("Example", "127.0.0.1", 0, "/callback", |code| async move { Ok(code) })
}

#[tokio::test]
async fn returns_the_browser_callback_and_aborts_the_manual_prompt() {
    let seen = Arc::new(Mutex::new(None));
    let server = start(plain_server_options()).await;
    let interaction = Interaction {
        prompt: pending_prompt(seen.clone()),
    };
    let url = callback_url(server.redirect_uri(), &[("code", "from-browser")]);
    let browser = tokio::spawn(async move { get(&url).await });
    let result = wait_for_callback_or_manual_input(&interaction, Some(&server), "paste", server.redirect_uri())
        .await
        .unwrap();
    assert_eq!(result, CallbackOrManual::Callback("from-browser".to_owned()));
    assert!(seen.lock().unwrap().as_ref().unwrap().is_cancelled());
    assert_eq!(browser.await.unwrap().status, 200);
}

#[tokio::test]
async fn returns_pasted_input_and_stops_waiting_for_the_browser() {
    let server = start(plain_server_options()).await;
    let interaction = Interaction {
        prompt: answer("pasted"),
    };
    let result = wait_for_callback_or_manual_input(&interaction, Some(&server), "paste", server.redirect_uri())
        .await
        .unwrap();
    assert_eq!(result, CallbackOrManual::Manual("pasted".to_owned()));
}

#[tokio::test]
async fn uses_only_the_manual_prompt_without_a_callback_server() {
    let interaction = Interaction {
        prompt: answer("pasted"),
    };
    let result = wait_for_callback_or_manual_input::<String>(&interaction, None, "paste", "http://localhost/callback")
        .await
        .unwrap();
    assert_eq!(result, CallbackOrManual::Manual("pasted".to_owned()));
}

#[tokio::test]
async fn propagates_manual_prompt_failures() {
    let server = start(plain_server_options()).await;
    let interaction = Interaction {
        prompt: Box::new(|_| Box::pin(async { Err(ModelsError::new(ModelsErrorCode::Aborted, "prompt cancelled")) })),
    };
    let error = wait_for_callback_or_manual_input(&interaction, Some(&server), "paste", server.redirect_uri())
        .await
        .unwrap_err();
    assert_eq!(error.message(), "prompt cancelled");
}

// ---------------------------------------------------------------------------
// Device-code polling (paused time)
// ---------------------------------------------------------------------------

type Results = Arc<Mutex<Vec<OAuthDeviceCodePollResult<String>>>>;

/// Runs the flow with scripted poll results; returns the result and each poll's offset.
async fn run_poll(
    options: OAuthDeviceCodePollOptions,
    results: Vec<OAuthDeviceCodePollResult<String>>,
) -> (Result<String, ModelsError>, Vec<Duration>) {
    let start = Instant::now();
    let times = Arc::new(Mutex::new(Vec::new()));
    let results: Results = Arc::new(Mutex::new(results));
    let result = poll_oauth_device_code_flow(options, || {
        let times = times.clone();
        let results = results.clone();
        async move {
            times.lock().unwrap().push(start.elapsed());
            let mut results = results.lock().unwrap();
            assert!(!results.is_empty(), "Unexpected extra poll");
            Ok(results.remove(0))
        }
    })
    .await;
    let times = times.lock().unwrap().clone();
    (result, times)
}

fn poll_options(interval: f64, expires_in: f64) -> OAuthDeviceCodePollOptions {
    OAuthDeviceCodePollOptions {
        interval_seconds: Some(interval),
        expires_in_seconds: Some(expires_in),
        ..OAuthDeviceCodePollOptions::default()
    }
}

fn complete() -> OAuthDeviceCodePollResult<String> {
    OAuthDeviceCodePollResult::Complete("token".to_owned())
}

#[tokio::test(start_paused = true)]
async fn polls_immediately_and_returns_the_completed_value() {
    let (result, times) = run_poll(
        poll_options(2.0, 30.0),
        vec![OAuthDeviceCodePollResult::Pending, complete()],
    )
    .await;
    assert_eq!(result.unwrap(), "token");
    assert_eq!(times, [Duration::ZERO, Duration::from_secs(2)]);
}

#[tokio::test(start_paused = true)]
async fn can_wait_before_the_first_poll() {
    let options = OAuthDeviceCodePollOptions {
        wait_before_first_poll: true,
        ..poll_options(2.0, 30.0)
    };
    let (result, times) = run_poll(options, vec![complete()]).await;
    assert_eq!(result.unwrap(), "token");
    assert_eq!(times, [Duration::from_secs(2)]);
}

#[tokio::test(start_paused = true)]
async fn increases_the_interval_by_5_seconds_after_slow_down_without_a_server_interval() {
    let (result, times) = run_poll(
        poll_options(2.0, 900.0),
        vec![
            OAuthDeviceCodePollResult::SlowDown { interval_seconds: None },
            complete(),
        ],
    )
    .await;
    assert_eq!(result.unwrap(), "token");
    assert_eq!(times, [Duration::ZERO, Duration::from_secs(7)]);
}

#[tokio::test(start_paused = true)]
async fn honors_a_server_provided_slow_down_interval() {
    let (result, times) = run_poll(
        poll_options(2.0, 900.0),
        vec![
            OAuthDeviceCodePollResult::SlowDown {
                interval_seconds: Some(30.0),
            },
            complete(),
        ],
    )
    .await;
    assert_eq!(result.unwrap(), "token");
    assert_eq!(times, [Duration::ZERO, Duration::from_secs(30)]);
}

#[tokio::test(start_paused = true)]
async fn cancels_an_in_flight_wait() {
    let cancel = CancellationToken::new();
    let options = OAuthDeviceCodePollOptions {
        cancel: cancel.clone(),
        ..poll_options(5.0, 30.0)
    };
    let flow = tokio::spawn(run_poll(options, vec![OAuthDeviceCodePollResult::Pending; 10]));
    tokio::task::yield_now().await;
    cancel.cancel();
    let (result, _) = flow.await.unwrap();
    let error = result.unwrap_err();
    assert_eq!(error.message(), "Login cancelled");
    assert_eq!(error.code(), ModelsErrorCode::Aborted);
}

#[tokio::test(start_paused = true)]
async fn times_out_at_the_deadline_with_the_slow_down_hint() {
    let (result, times) = run_poll(poll_options(2.0, 5.0), vec![OAuthDeviceCodePollResult::Pending; 3]).await;
    assert_eq!(result.unwrap_err().message(), TIMEOUT_MESSAGE);
    // The last sleep is cut to the deadline.
    assert_eq!(times, [Duration::ZERO, Duration::from_secs(2), Duration::from_secs(4)]);

    let (result, _) = run_poll(
        poll_options(2.0, 5.0),
        vec![OAuthDeviceCodePollResult::SlowDown { interval_seconds: None }],
    )
    .await;
    assert_eq!(result.unwrap_err().message(), SLOW_DOWN_TIMEOUT_MESSAGE);
}

#[tokio::test(start_paused = true)]
async fn fails_with_the_poll_message_and_floors_the_interval() {
    let (result, times) = run_poll(
        poll_options(0.2, 30.0),
        vec![
            OAuthDeviceCodePollResult::Pending,
            OAuthDeviceCodePollResult::Failed {
                message: "denied".into(),
            },
        ],
    )
    .await;
    let error = result.unwrap_err();
    assert_eq!(error.message(), "denied");
    assert_eq!(error.code(), ModelsErrorCode::OAuth);
    assert_eq!(times, [Duration::ZERO, Duration::from_secs(1)]);
}
