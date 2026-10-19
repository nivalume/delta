//! Port of upstream `test/images-models.test.ts`: image models in `Models`, image provider
//! dispatch and `Models::generate_images`.

#![allow(clippy::unwrap_used)] // Test helpers outside `#[test]` functions.

mod support;

use futures::future::BoxFuture;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

use delta_ai::auth::AuthResolutionOverrides;
use delta_ai::models::InMemoryModelsStore;
use delta_ai::provider::{FetchModels, is_model_type};
use delta_ai::providers::builtin::builtin_models;
use delta_ai::providers::catalog::{all_builtin_models, builtin_model_of_type, builtin_models as catalog_models};
use delta_ai::{
    AnyModel, AssistantImages, AuthResult, CreateModelsOptions, CreateProviderOptions, ImageContent, ImageModel,
    ImagesContext, ImagesOptions, ImagesStopReason, ModelAuth, ModelType, Models, ModelsRefreshOptions, ModelsStore,
    ProviderApi, ProviderAuth, ProviderImages, RequestOptions, UserBlock, create_provider, try_create_provider,
};
use support::{FakeAuthContext, NoChat, StaticAuth, chat_api, chat_model, image_model, test_auth};

type Calls = Arc<Mutex<Vec<(ImageModel, ImagesOptions)>>>;

fn ok_result(model: &ImageModel) -> AssistantImages {
    AssistantImages {
        output: vec![UserBlock::Image(ImageContent {
            data: "aGk=".into(),
            mime_type: "image/png".into(),
        })],
        ..AssistantImages::empty(model)
    }
}

/// Records each call and returns an image.
#[derive(Clone, Default)]
struct RecordingImages {
    calls: Calls,
}

impl ProviderImages for RecordingImages {
    fn generate_images<'a>(
        &'a self,
        model: &'a ImageModel,
        _context: &'a ImagesContext,
        options: ImagesOptions,
    ) -> BoxFuture<'a, AssistantImages> {
        self.calls.lock().unwrap().push((model.clone(), options));
        Box::pin(async move { ok_result(model) })
    }
}

/// Upstream `testProvider`: a `test-chat` chat API plus image implementations for `image_apis`.
fn test_provider(
    id: &str,
    models: Option<Vec<AnyModel>>,
    env_var: Option<&str>,
    calls: &Calls,
    image_apis: &[&str],
) -> delta_ai::provider::BuiltProvider {
    let models = models.unwrap_or_else(|| vec![image_model(id, "model-a").into()]);
    let mut options = CreateProviderOptions::new(id, test_auth(env_var), models, chat_api("test-chat"));
    for api in image_apis {
        options = options.images(*api, RecordingImages { calls: calls.clone() });
    }
    create_provider(options)
}

fn context() -> ImagesContext {
    ImagesContext::new(vec![UserBlock::text("a red circle")])
}

fn ids(models: &[AnyModel]) -> Vec<&str> {
    models.iter().map(AnyModel::id).collect()
}

fn image_of(models: &Models, provider: &str, id: &str) -> ImageModel {
    match models.get_model_of_type(ModelType::Image, provider, id) {
        Some(AnyModel::Image(model)) => model,
        other => panic!("expected an image model, got {other:?}"),
    }
}

#[test]
fn treats_models_without_a_type_as_chat_models() {
    let chat = chat_model("p", "test-chat", "c");
    let untyped = serde_json::to_value(&chat).unwrap();
    assert!(untyped.get("type").is_none());
    let parsed: AnyModel = serde_json::from_value(untyped).unwrap();
    assert_eq!(parsed.model_type(), ModelType::Chat);
    let mut typed = serde_json::to_value(&chat).unwrap();
    typed["type"] = "chat".into();
    assert_eq!(
        serde_json::from_value::<AnyModel>(typed).unwrap().model_type(),
        ModelType::Chat
    );
    let image: AnyModel = image_model("p", "i").into();
    assert_eq!(image.model_type(), ModelType::Image);
    let chat: AnyModel = chat.into();
    assert!(is_model_type(&chat, ModelType::Chat));
    assert!(!is_model_type(&chat, ModelType::Image));
    assert!(is_model_type(&image, ModelType::Image));
    // `hasApi` never matches an image model, even on an equal api string.
    assert!(image.as_chat().is_none());
}

#[tokio::test]
async fn lists_models_without_a_type_as_chat_models_at_create_provider_boundaries() {
    let fetch: FetchModels =
        Arc::new(|_| Box::pin(async { Ok(vec![chat_model("legacy", "test-chat", "dynamic").into()]) }));
    let mut options = CreateProviderOptions::new(
        "legacy",
        ProviderAuth::keyless("Test"),
        vec![
            chat_model("legacy", "test-chat", "static").into(),
            image_model("legacy", "static").into(),
        ],
        ProviderApi::single(NoChat),
    );
    options.fetch_models = Some(fetch);
    let models = Models::default();
    models.set_provider(create_provider(options));
    let provider = models.provider("legacy").unwrap();

    let chat_ids = |provider: &dyn delta_ai::Provider| provider.models().into_iter().map(|m| m.id).collect::<Vec<_>>();
    assert_eq!(chat_ids(provider.as_ref()), ["static"]);
    models
        .refresh(ModelsRefreshOptions {
            providers: Some(vec!["legacy".into()]),
            ..ModelsRefreshOptions::default()
        })
        .await;
    assert_eq!(chat_ids(provider.as_ref()), ["static", "dynamic"]);
    assert_eq!(
        ids(&models.get_models_of_type(ModelType::Image, Some("legacy"))),
        ["static"]
    );
}

#[test]
fn lists_chat_image_and_all_models_through_typed_accessors() {
    let calls = Calls::default();
    let models = Models::default();
    models.set_provider(test_provider(
        "p1",
        Some(vec![
            chat_model("p1", "test-chat", "c1").into(),
            image_model("p1", "i1").into(),
            image_model("p1", "i2").into(),
        ]),
        None,
        &calls,
        &["test-images"],
    ));
    models.set_provider(test_provider(
        "p2",
        Some(vec![image_model("p2", "i3").into()]),
        None,
        &calls,
        &["test-images"],
    ));

    let chat: Vec<String> = models.get_models(None).into_iter().map(|m| m.id).collect();
    assert_eq!(chat, ["c1"]);
    assert_eq!(ids(&models.get_models_of_type(ModelType::Chat, None)), ["c1"]);
    assert_eq!(
        ids(&models.get_models_of_type(ModelType::Image, None)),
        ["i1", "i2", "i3"]
    );
    assert_eq!(
        ids(&models.get_models_of_type(ModelType::Image, Some("p1"))),
        ["i1", "i2"]
    );
    assert!(models.get_models_of_type(ModelType::Classifier, None).is_empty());
    assert_eq!(ids(&models.get_all_models(None)), ["c1", "i1", "i2", "i3"]);

    assert_eq!(models.get_model("p1", "c1").map(|m| m.id).as_deref(), Some("c1"));
    assert!(models.get_model("p1", "i1").is_none());
    assert!(models.get_model_of_type(ModelType::Chat, "p1", "c1").is_some());
    assert!(models.get_model_of_type(ModelType::Image, "p1", "i1").is_some());
    assert!(models.get_model_of_type(ModelType::Image, "p1", "c1").is_none());
}

#[tokio::test]
async fn splits_available_models_by_type() {
    let calls = Calls::default();
    let models = Models::new(CreateModelsOptions {
        auth_context: Some(FakeAuthContext::with_env(&[("KEY", "k")])),
        ..CreateModelsOptions::default()
    });
    models.set_provider(test_provider(
        "p1",
        Some(vec![
            chat_model("p1", "test-chat", "c1").into(),
            image_model("p1", "i1").into(),
        ]),
        Some("KEY"),
        &calls,
        &["test-images"],
    ));
    models.set_provider(test_provider(
        "p2",
        Some(vec![image_model("p2", "i2").into()]),
        Some("MISSING"),
        &calls,
        &["test-images"],
    ));

    let available: Vec<String> = models
        .get_available(None)
        .await
        .unwrap()
        .into_iter()
        .map(|m| m.id)
        .collect();
    assert_eq!(available, ["c1"]);
    assert_eq!(
        ids(&models.get_available_of_type(ModelType::Chat, None).await.unwrap()),
        ["c1"]
    );
    assert_eq!(
        ids(&models.get_available_of_type(ModelType::Image, None).await.unwrap()),
        ["i1"]
    );
    assert_eq!(ids(&models.get_all_available(None).await.unwrap()), ["c1", "i1"]);
}

#[tokio::test]
async fn resolves_auth_through_the_provider_and_merges_it_into_image_requests() {
    let calls = Calls::default();
    let models = Models::new(CreateModelsOptions {
        auth_context: Some(FakeAuthContext::with_env(&[("TEST_KEY", "env-key")])),
        ..CreateModelsOptions::default()
    });
    models.set_provider(test_provider("p1", None, Some("TEST_KEY"), &calls, &["test-images"]));
    let model = image_of(&models, "p1", "model-a");
    let any = AnyModel::Image(model.clone());

    let key = |result: Option<AuthResult>| result.and_then(|result| result.auth.api_key);
    let defaults = AuthResolutionOverrides::default();
    assert_eq!(
        key(models.get_model_auth(&any, &defaults).await.unwrap()).as_deref(),
        Some("env-key")
    );
    assert_eq!(
        key(models.get_auth("p1", &defaults).await.unwrap()).as_deref(),
        Some("env-key")
    );
    let explicit = AuthResolutionOverrides {
        api_key: Some("explicit-key".into()),
        ..AuthResolutionOverrides::default()
    };
    assert_eq!(
        key(models.get_model_auth(&any, &explicit).await.unwrap()).as_deref(),
        Some("explicit-key")
    );

    let result = models
        .generate_images(&model, &context(), ImagesOptions::default())
        .await;
    assert_eq!(result.stop_reason, ImagesStopReason::Stop);
    assert_eq!(calls.lock().unwrap()[0].1.base.api_key.as_deref(), Some("env-key"));

    let options = ImagesOptions::from(RequestOptions {
        api_key: Some("explicit".into()),
        ..RequestOptions::default()
    });
    models.generate_images(&model, &context(), options).await;
    assert_eq!(calls.lock().unwrap()[1].1.base.api_key.as_deref(), Some("explicit"));
}

#[tokio::test]
async fn merges_provider_resolved_env_and_applies_header_transforms() {
    let calls = Calls::default();
    let models = Models::default();
    let auth = StaticAuth(AuthResult {
        auth: ModelAuth {
            api_key: Some("provider-key".into()),
            headers: Some([("x-base".to_owned(), Some("1".to_owned()))].into()),
            base_url: None,
        },
        env: Some(BTreeMap::from([
            ("PROVIDER_ONLY".to_owned(), "provider".to_owned()),
            ("SHARED".to_owned(), "provider".to_owned()),
        ])),
        source: None,
    });
    models.set_provider(create_provider(
        CreateProviderOptions::new(
            "p1",
            ProviderAuth::api_key(auth),
            vec![image_model("p1", "model-a").into()],
            ProviderApi::none(),
        )
        .images("test-images", RecordingImages { calls: calls.clone() }),
    ));
    let model = image_of(&models, "p1", "model-a");

    let options = ImagesOptions::from(RequestOptions {
        api_key: Some("request-key".into()),
        env: Some(BTreeMap::from([
            ("REQUEST_ONLY".to_owned(), "request".to_owned()),
            ("SHARED".to_owned(), "request".to_owned()),
        ])),
        transform_headers: Some(Arc::new(|mut headers| {
            Box::pin(async move {
                headers.insert("x-extra".into(), Some("2".into()));
                headers
            })
        })),
        ..RequestOptions::default()
    });
    models.generate_images(&model, &context(), options).await;

    let calls = calls.lock().unwrap();
    let options = &calls[0].1.base;
    assert_eq!(options.api_key.as_deref(), Some("request-key"));
    assert_eq!(
        options.env,
        Some(BTreeMap::from([
            ("PROVIDER_ONLY".to_owned(), "provider".to_owned()),
            ("REQUEST_ONLY".to_owned(), "request".to_owned()),
            ("SHARED".to_owned(), "request".to_owned()),
        ]))
    );
    assert_eq!(
        options.headers,
        Some(
            [
                ("x-base".to_owned(), Some("1".to_owned())),
                ("x-extra".to_owned(), Some("2".to_owned()))
            ]
            .into()
        )
    );
    assert!(options.transform_headers.is_none());
}

#[tokio::test]
async fn returns_error_results_instead_of_rejecting() {
    let models = Models::new(CreateModelsOptions {
        auth_context: Some(FakeAuthContext::with_env(&[])),
        ..CreateModelsOptions::default()
    });

    let ghost = models
        .generate_images(&image_model("ghost", "m"), &context(), ImagesOptions::default())
        .await;
    assert_eq!(ghost.stop_reason, ImagesStopReason::Error);
    assert!(ghost.error_message.unwrap().contains("Unknown provider: ghost"));

    // Unconfigured auth is an error, matching stream().
    let calls = Calls::default();
    models.set_provider(test_provider("p1", None, Some("MISSING"), &calls, &["test-images"]));
    let model = image_of(&models, "p1", "model-a");
    let unconfigured_auth = models
        .get_model_auth(&model.clone().into(), &AuthResolutionOverrides::default())
        .await
        .unwrap();
    assert!(unconfigured_auth.is_none());
    let unconfigured = models
        .generate_images(&model, &context(), ImagesOptions::default())
        .await;
    assert_eq!(unconfigured.stop_reason, ImagesStopReason::Error);
    assert!(unconfigured.error_message.unwrap().contains("not configured"));
    assert!(calls.lock().unwrap().is_empty());

    let cancel = CancellationToken::new();
    cancel.cancel();
    let options = ImagesOptions::from(RequestOptions {
        cancel: Some(cancel),
        ..RequestOptions::default()
    });
    let cancelled = models.generate_images(&model, &context(), options).await;
    assert_eq!(cancelled.stop_reason, ImagesStopReason::Aborted);
    assert!(calls.lock().unwrap().is_empty());

    // A provider without any images implementation rejects image models it lists.
    models.set_provider(create_provider(CreateProviderOptions::new(
        "chat-only",
        ProviderAuth::keyless("Test"),
        vec![image_model("chat-only", "i").into()],
        ProviderApi::single(NoChat),
    )));
    let unsupported = models
        .generate_images(
            &image_of(&models, "chat-only", "i"),
            &context(),
            ImagesOptions::default(),
        )
        .await;
    assert_eq!(unsupported.stop_reason, ImagesStopReason::Error);
    assert!(
        unsupported
            .error_message
            .unwrap()
            .contains("does not support image generation")
    );

    // An images map without the model's api yields a provider error result.
    models.set_provider(test_provider(
        "wrong-api",
        Some(vec![image_model("wrong-api", "i").into()]),
        None,
        &calls,
        &["other-images"],
    ));
    let missing_api = models
        .generate_images(
            &image_of(&models, "wrong-api", "i"),
            &context(),
            ImagesOptions::default(),
        )
        .await;
    assert_eq!(missing_api.stop_reason, ImagesStopReason::Error);
    assert!(
        missing_api
            .error_message
            .unwrap()
            .contains("no image generation implementation for \"test-images\"")
    );
}

#[test]
fn requires_at_least_one_concrete_operation_implementation() {
    let empty = || CreateProviderOptions::new("empty", ProviderAuth::keyless("Test"), Vec::new(), ProviderApi::none());
    let message = "at least one of \"api\", \"images\", or \"classifiers\"";
    let error = try_create_provider(empty()).unwrap_err();
    assert!(error.to_string().contains(message), "{error}");
    assert!(try_create_provider(empty().images("test-images", RecordingImages::default())).is_ok());
    assert!(try_create_provider(empty()).is_err());
}

#[test]
#[should_panic(expected = "at least one of \"api\", \"images\", or \"classifiers\" is required")]
fn create_provider_panics_without_any_implementation() {
    create_provider(CreateProviderOptions::new(
        "empty",
        ProviderAuth::keyless("Test"),
        Vec::new(),
        ProviderApi::none(),
    ));
}

#[tokio::test]
async fn supports_dynamic_providers_listing_image_models_via_refresh() {
    let fetches = Arc::new(Mutex::new(0));
    let store = Arc::new(InMemoryModelsStore::default());
    let models = Models::new(CreateModelsOptions {
        models_store: Some(store.clone()),
        ..CreateModelsOptions::default()
    });
    let counter = fetches.clone();
    let fetch: FetchModels = Arc::new(move |_| {
        *counter.lock().unwrap() += 1;
        Box::pin(async {
            Ok(vec![
                image_model("dyn", "listed").into(),
                chat_model("dyn", "test-chat", "chat").into(),
            ])
        })
    });
    let mut options = CreateProviderOptions::new("dyn", ProviderAuth::keyless("Test"), Vec::new(), ProviderApi::none())
        .images("test-images", RecordingImages::default());
    options.fetch_models = Some(fetch);
    models.set_provider(create_provider(options));

    assert!(models.get_all_models(Some("dyn")).is_empty());
    let result = models
        .refresh(ModelsRefreshOptions {
            providers: Some(vec!["dyn".into()]),
            ..ModelsRefreshOptions::default()
        })
        .await;
    assert!(result.errors.is_empty());
    assert_eq!(*fetches.lock().unwrap(), 1);
    assert!(models.get_model_of_type(ModelType::Image, "dyn", "listed").is_some());
    assert!(models.get_model("dyn", "chat").is_some());
    let stored = store.read("dyn").await.unwrap().unwrap();
    assert_eq!(ids(&stored.models), ["listed", "chat"]);
}

#[test]
fn keeps_existing_builtin_reads_chat_only() {
    let chat = catalog_models("openrouter");
    let all = all_builtin_models("openrouter");
    let images: Vec<&AnyModel> = all.iter().filter(|m| m.model_type() == ModelType::Image).collect();
    let classifiers = all.iter().filter(|m| m.model_type() == ModelType::Classifier).count();

    assert!(!images.is_empty());
    assert!(chat.iter().all(|model| model.context_window > 0));
    assert_eq!(chat.len() + images.len() + classifiers, all.len());
    assert!(matches!(
        builtin_model_of_type(ModelType::Image, "openrouter", "black-forest-labs/flux.2-pro"),
        Some(AnyModel::Image(_))
    ));
}

#[tokio::test]
async fn builtin_models_expose_openrouter_image_models_under_the_openrouter_provider() {
    let models = builtin_models(CreateModelsOptions {
        auth_context: Some(FakeAuthContext::with_env(&[("OPENROUTER_API_KEY", "or-key")])),
        ..CreateModelsOptions::default()
    });
    let provider = models.provider("openrouter").unwrap();
    let images = models.get_models_of_type(ModelType::Image, Some("openrouter"));
    assert!(!images.is_empty());
    assert!(
        provider
            .all_models()
            .iter()
            .any(|model| model.model_type() == ModelType::Image)
    );
    assert!(images.iter().all(|model| model.api() == "openrouter-images"));
    assert!(
        models
            .get_models_of_type(ModelType::Image, None)
            .iter()
            .all(|model| model.provider() == "openrouter")
    );

    // One upstream id can expose separate chat and image operations.
    let chat = models.get_model("openrouter", "google/gemini-3-pro-image").unwrap();
    let image = models
        .get_model_of_type(ModelType::Image, "openrouter", "google/gemini-3-pro-image")
        .unwrap();
    assert_eq!(chat.api, "openai-completions");
    assert_eq!(image.api(), "openrouter-images");

    // One credential covers both.
    let defaults = AuthResolutionOverrides::default();
    let key = |result: Option<AuthResult>| result.and_then(|result| result.auth.api_key);
    assert_eq!(
        key(models.get_model_auth(&images[0], &defaults).await.unwrap()).as_deref(),
        Some("or-key")
    );
    assert_eq!(
        key(models.get_model_auth(&chat.into(), &defaults).await.unwrap()).as_deref(),
        Some("or-key")
    );
    // Upstream also checks `provider.generateImages`; the `openrouter-images` API is not
    // ported yet.
}
