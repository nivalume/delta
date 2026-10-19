//! Deferred responses: ports of the deferred cases of upstream `test/providers.test.ts`
//! (`createProvider` capabilities, `Models` auth for fetch and cancel, the faux provider).

#![allow(clippy::unwrap_used)] // Test helpers outside `#[test]` functions.

mod support;

use futures::StreamExt;
use futures::future::BoxFuture;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use delta_ai::error::Error;
use delta_ai::providers::faux::{
    FauxDeferredOptions, FauxOptions, FauxProvider, FauxResponse, faux_assistant_message, faux_text,
};
use delta_ai::types::ProviderHeaders;
use delta_ai::{
    AssistantBlock, AssistantMessage, AssistantMessageEventStream, AuthResult, Context, CreateProviderOptions,
    DeferredCancelOptions, DeferredFetchOptions, DeferredHandle, DeferredRequest, DeferredWindow, Message, Model,
    ModelAuth, Models, Provider, ProviderApi, ProviderAuth, ProviderStreams, RequestOptions, SimpleStreamOptions,
    StopReason, StreamOptions, TranscriptContext, create_provider,
};
use support::{NoChat, StaticAuth, chat_model};

type Fetched = Arc<Mutex<Vec<(Model, DeferredFetchOptions)>>>;
type Cancelled = Arc<Mutex<Vec<DeferredCancelOptions>>>;

/// Chat API with deferred support: `fetch_deferred` answers `stop`, `cancel_deferred`
/// records its options. Cancellation support is optional.
#[derive(Clone, Default)]
struct DeferredStreams {
    can_cancel: bool,
    fetched: Fetched,
    cancelled: Cancelled,
}

impl ProviderStreams for DeferredStreams {
    fn stream(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        NoChat.stream(model, context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        NoChat.stream_simple(model, context, options)
    }

    fn supports_fetch_deferred(&self) -> bool {
        true
    }

    fn fetch_deferred(
        &self,
        model: &Model,
        _handle: &DeferredHandle,
        options: DeferredFetchOptions,
    ) -> AssistantMessageEventStream {
        self.fetched.lock().unwrap().push((model.clone(), options));
        let mut message = AssistantMessage::empty(&model.api, &model.provider, &model.id);
        message.stop_reason = StopReason::Stop;
        AssistantMessageEventStream::from_final(message)
    }

    fn supports_cancel_deferred(&self) -> bool {
        self.can_cancel
    }

    fn cancel_deferred<'a>(
        &'a self,
        _model: &'a Model,
        _handle: &'a DeferredHandle,
        options: DeferredCancelOptions,
    ) -> BoxFuture<'a, Result<(), Error>> {
        self.cancelled.lock().unwrap().push(options);
        Box::pin(async { Ok(()) })
    }
}

fn handle_for(model: &Model) -> DeferredHandle {
    DeferredHandle {
        provider: model.provider.clone(),
        model_id: model.id.clone(),
        api: model.api.clone(),
        id: "response-1".into(),
        expires_at: None,
        poll_after_ms: None,
        data: None,
    }
}

fn header<'a>(headers: Option<&'a ProviderHeaders>, name: &str) -> Option<&'a str> {
    headers?
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .and_then(|(_, value)| value.as_deref())
}

fn context() -> Context {
    Context::new(vec![Message::user("hi")])
}

#[tokio::test]
async fn exposes_only_declared_deferred_capabilities() {
    let model = chat_model("mixed", "api-a", "model-a");
    let provider = create_provider(CreateProviderOptions::new(
        "mixed",
        ProviderAuth::keyless("Test"),
        vec![model.clone().into()],
        ProviderApi::single(DeferredStreams::default()),
    ));
    assert!(provider.supports_fetch_deferred());
    assert!(!provider.supports_cancel_deferred());
    let fetched = provider
        .fetch_deferred(&model, &handle_for(&model), DeferredFetchOptions::default())
        .result()
        .await;
    assert_eq!(fetched.stop_reason, StopReason::Stop);

    let plain = create_provider(CreateProviderOptions::new(
        "plain",
        ProviderAuth::keyless("Test"),
        Vec::new(),
        ProviderApi::single(NoChat),
    ));
    assert!(!plain.supports_fetch_deferred());
    assert!(!plain.supports_cancel_deferred());
}

#[tokio::test]
async fn applies_resolved_request_options_to_deferred_fetch_and_cancellation() {
    let model = chat_model("deferred-provider", "api-a", "model-a");
    let streams = DeferredStreams {
        can_cancel: true,
        ..DeferredStreams::default()
    };
    let auth = StaticAuth(AuthResult {
        auth: ModelAuth {
            api_key: Some("provider-key".into()),
            headers: Some(
                [
                    ("Authorization".to_owned(), Some("Bearer provider".to_owned())),
                    ("X-Shared".to_owned(), Some("provider".to_owned())),
                ]
                .into(),
            ),
            base_url: Some("https://resolved.test/v1".into()),
        },
        env: Some(BTreeMap::from([
            ("PROVIDER_ONLY".to_owned(), "provider".to_owned()),
            ("SHARED".to_owned(), "provider".to_owned()),
        ])),
        source: None,
    });
    let models = Models::default();
    models.set_provider(create_provider(CreateProviderOptions::new(
        "deferred-provider",
        ProviderAuth::api_key(auth),
        vec![model.clone().into()],
        ProviderApi::single(streams.clone()),
    )));
    let handle = handle_for(&model);

    let fetch_options = DeferredFetchOptions {
        base: RequestOptions {
            timeout: Some(Duration::from_millis(100)),
            api_key: Some("request-key".into()),
            headers: Some(
                [
                    ("X-Request".to_owned(), Some("request".to_owned())),
                    ("x-shared".to_owned(), Some("request".to_owned())),
                ]
                .into(),
            ),
            env: Some(BTreeMap::from([
                ("REQUEST_ONLY".to_owned(), "request".to_owned()),
                ("SHARED".to_owned(), "request".to_owned()),
            ])),
            transform_headers: Some(Arc::new(|mut headers| {
                Box::pin(async move {
                    headers.insert("X-Transformed".into(), Some("yes".into()));
                    headers
                })
            })),
            ..RequestOptions::default()
        },
        wait: Some(Duration::from_millis(50)),
    };
    models.fetch_deferred(&model, &handle, fetch_options).await;
    let cancel_options = DeferredCancelOptions {
        timeout: Some(Duration::from_millis(200)),
        transform_headers: Some(Arc::new(|mut headers| {
            Box::pin(async move {
                headers.insert("X-Cancel".into(), Some("yes".into()));
                headers
            })
        })),
        ..RequestOptions::default()
    };
    models.cancel_deferred(&model, &handle, cancel_options).await.unwrap();

    let fetched = streams.fetched.lock().unwrap();
    let (fetched_model, fetched) = &fetched[0];
    assert_eq!(fetched_model.base_url, "https://resolved.test/v1");
    assert_eq!(fetched.wait, Some(Duration::from_millis(50)));
    assert_eq!(fetched.base.timeout, Some(Duration::from_millis(100)));
    assert_eq!(fetched.base.api_key.as_deref(), Some("request-key"));
    let headers = fetched.base.headers.as_ref();
    assert_eq!(header(headers, "Authorization"), Some("Bearer provider"));
    assert_eq!(header(headers, "X-Request"), Some("request"));
    assert_eq!(header(headers, "x-shared"), Some("request"));
    assert_eq!(header(headers, "X-Transformed"), Some("yes"));
    assert_eq!(
        fetched.base.env,
        Some(BTreeMap::from([
            ("PROVIDER_ONLY".to_owned(), "provider".to_owned()),
            ("REQUEST_ONLY".to_owned(), "request".to_owned()),
            ("SHARED".to_owned(), "request".to_owned()),
        ]))
    );

    let cancelled = &streams.cancelled.lock().unwrap()[0];
    assert_eq!(cancelled.timeout, Some(Duration::from_millis(200)));
    assert_eq!(cancelled.api_key.as_deref(), Some("provider-key"));
    let headers = cancelled.headers.as_ref();
    assert_eq!(header(headers, "Authorization"), Some("Bearer provider"));
    assert_eq!(header(headers, "X-Shared"), Some("provider"));
    assert_eq!(header(headers, "X-Cancel"), Some("yes"));
    assert_eq!(
        cancelled.env,
        Some(BTreeMap::from([
            ("PROVIDER_ONLY".to_owned(), "provider".to_owned()),
            ("SHARED".to_owned(), "provider".to_owned()),
        ]))
    );
}

#[tokio::test]
async fn reports_providers_and_apis_without_deferred_support() {
    let models = Models::default();
    let plain = chat_model("plain", "api-a", "model-a");
    models.set_provider(create_provider(CreateProviderOptions::new(
        "plain",
        ProviderAuth::keyless("Test"),
        vec![plain.clone().into()],
        ProviderApi::single(NoChat),
    )));
    let fetched = models
        .fetch_deferred(&plain, &handle_for(&plain), DeferredFetchOptions::default())
        .await;
    assert_eq!(fetched.stop_reason, StopReason::Error);
    assert_eq!(
        fetched.error_message.as_deref(),
        Some("Provider plain does not support deferred responses")
    );
    let cancelled = models
        .cancel_deferred(&plain, &handle_for(&plain), DeferredCancelOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        cancelled.to_string(),
        "Provider plain does not support deferred responses"
    );

    // A mixed provider whose model API has no deferred implementation.
    let other = chat_model("mixed", "api-b", "model-b");
    models.set_provider(create_provider(CreateProviderOptions::new(
        "mixed",
        ProviderAuth::keyless("Test"),
        vec![other.clone().into()],
        ProviderApi::ByApi(
            [
                (
                    "api-a".to_owned(),
                    Arc::new(DeferredStreams {
                        can_cancel: true,
                        ..DeferredStreams::default()
                    }) as Arc<dyn ProviderStreams>,
                ),
                ("api-b".to_owned(), Arc::new(NoChat) as Arc<dyn ProviderStreams>),
            ]
            .into(),
        ),
    )));
    let fetched = models
        .fetch_deferred(&other, &handle_for(&other), DeferredFetchOptions::default())
        .await;
    assert_eq!(
        fetched.error_message.as_deref(),
        Some("Provider mixed does not support deferred responses for \"api-b\"")
    );
    let cancelled = models
        .cancel_deferred(&other, &handle_for(&other), DeferredCancelOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        cancelled.to_string(),
        "Provider mixed cannot cancel deferred responses for \"api-b\""
    );

    let ghost = chat_model("ghost", "api-a", "m");
    let unknown = models
        .fetch_deferred(&ghost, &handle_for(&ghost), DeferredFetchOptions::default())
        .await;
    assert_eq!(unknown.error_message.as_deref(), Some("Unknown provider: ghost"));
}

#[tokio::test]
async fn faux_submits_polls_and_redeems_deferred_responses() {
    let faux = FauxProvider::new(FauxOptions {
        deferred: Some(FauxDeferredOptions {
            pending_fetches: 1,
            poll_after_ms: Some(25),
        }),
        ..FauxOptions::default()
    });
    let models = Models::default();
    models.set_provider(faux.clone());
    faux.set_responses(vec![faux_assistant_message(vec![faux_text("ready")]).into()]);
    let model = faux.model();

    let options = SimpleStreamOptions {
        deferred: Some(DeferredRequest::with_window(DeferredWindow::OneHour)),
        ..SimpleStreamOptions::default()
    };
    let mut submission = models.stream_simple(&model, &context(), options);
    let mut kinds = Vec::new();
    let mut deferred = None;
    while let Some(event) = submission.next().await {
        kinds.push(event.kind());
        if let Some(message) = event.final_message() {
            deferred = Some(message.clone());
        }
    }
    let deferred = deferred.unwrap();
    assert_eq!(kinds, ["start", "done"]);
    assert_eq!(deferred.stop_reason, StopReason::Deferred);
    assert!(deferred.content.is_empty());
    let handle = deferred.deferred.clone().expect("deferred handle");
    assert_eq!(handle.provider, model.provider);
    assert_eq!(handle.model_id, model.id);
    assert_eq!(handle.api, model.api);
    assert!(!handle.id.is_empty());
    assert_eq!(handle.poll_after_ms, Some(25));
    assert_eq!(handle.expires_at, None);
    assert_eq!(handle.data, None);

    let pending = models
        .fetch_deferred(&model, &handle, DeferredFetchOptions::default())
        .await;
    assert_eq!(pending.stop_reason, StopReason::Deferred);
    assert_eq!(pending.deferred.as_ref(), Some(&handle));

    let ready = models
        .fetch_deferred(
            &model,
            &handle,
            DeferredFetchOptions {
                wait: Some(Duration::ZERO),
                ..DeferredFetchOptions::default()
            },
        )
        .await;
    assert_eq!(ready.stop_reason, StopReason::Stop);
    assert_eq!(ready.content, [faux_text("ready")]);
    assert!(ready.usage.total_tokens > 0);
    let state = faux.state();
    assert_eq!(state.call_count, 1);
    assert_eq!(state.deferred_fetch_count, 2);
}

#[tokio::test]
async fn faux_records_cancellation_and_returns_deferred_fetch_failures_in_band() {
    let faux = FauxProvider::new(FauxOptions::default());
    let models = Models::default();
    models.set_provider(faux.clone());
    let failing: FauxResponse = FauxResponse::Factory(Arc::new(|_, _, model| AssistantMessage {
        stop_reason: StopReason::Error,
        error_message: Some("deferred failed".into()),
        ..AssistantMessage::empty(&model.api, &model.provider, &model.id)
    }));
    faux.set_responses(vec![
        failing,
        faux_assistant_message(vec![faux_text("cancelled")]).into(),
    ]);
    let model = faux.model();
    let deferred = || SimpleStreamOptions {
        deferred: Some(DeferredRequest::default()),
        ..SimpleStreamOptions::default()
    };

    let failed_submission = models.complete_simple(&model, &context(), deferred()).await;
    let handle = failed_submission.deferred.expect("deferred handle");
    let failed = models
        .fetch_deferred(&model, &handle, DeferredFetchOptions::default())
        .await;
    assert_eq!(failed.stop_reason, StopReason::Error);
    assert_eq!(failed.error_message.as_deref(), Some("deferred failed"));

    let cancelled_submission = models.complete_simple(&model, &context(), deferred()).await;
    let handle = cancelled_submission.deferred.expect("deferred handle");
    models
        .cancel_deferred(&model, &handle, DeferredCancelOptions::default())
        .await
        .unwrap();
    assert_eq!(faux.state().cancelled_deferred, std::slice::from_ref(&handle));
    let cancelled = models
        .fetch_deferred(&model, &handle, DeferredFetchOptions::default())
        .await;
    assert_eq!(cancelled.stop_reason, StopReason::Error);
    assert!(cancelled.error_message.unwrap().contains("was cancelled"));
}

#[tokio::test]
async fn faux_rejects_unknown_deferred_handles() {
    let faux = FauxProvider::new(FauxOptions::default());
    let models = Models::default();
    models.set_provider(faux.clone());
    let model = faux.model();
    let unknown = models
        .fetch_deferred(&model, &handle_for(&model), DeferredFetchOptions::default())
        .await;
    assert_eq!(
        unknown.error_message.as_deref(),
        Some("Unknown faux deferred response: response-1")
    );
    // Without the deferred option, a simple request is answered directly.
    faux.set_responses(vec![faux_assistant_message(vec![faux_text("now")]).into()]);
    let direct = models
        .complete_simple(&model, &context(), SimpleStreamOptions::default())
        .await;
    assert_eq!(
        direct.content,
        [AssistantBlock::Text(delta_ai::TextContent::new("now"))]
    );
    assert!(direct.deferred.is_none());
}
