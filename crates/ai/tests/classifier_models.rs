//! Port of upstream `test/classifier-models.test.ts`: the `Models` layer, and the built-in
//! providers' classifier APIs against local servers.

#![allow(clippy::unwrap_used)] // Test helpers outside `#[test]` functions.

mod support;

use futures::future::BoxFuture;
use indexmap::IndexMap;
use std::sync::{Arc, Mutex};

use support::chat_api;

use delta_ai::classifier::{ClassifierBoolAnswer, ClassifierBoolCriteria, ClassifierBoolQuestion};
use delta_ai::oauth::OpenRouterOAuth;
use delta_ai::providers::builtin::{builtin_models, builtin_provider};
use delta_ai::providers::catalog::{all_builtin_models, builtin_model_of_type};
use delta_ai::{
    AnyModel, ApiKeyCredential, ClassifierAnswer, ClassifierContext, ClassifierModel, ClassifierOptions,
    ClassifierQuestion, ClassifierResult, ClassifierStopReason, CreateModelsOptions, CreateProviderOptions, Credential,
    CredentialStore, ImageContent, InMemoryCredentialStore, JsonObject, ModelType, Models, OAuthCredential, Provider,
    ProviderApi, ProviderAuth, ProviderClassifier, RequestOptions, create_provider,
};
use serde_json::{Value, json};
use support::{chat_model, classifier_model};
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

fn context() -> ClassifierContext {
    let mut questions = IndexMap::new();
    questions.insert(
        "approved".to_owned(),
        ClassifierQuestion::Bool(ClassifierBoolQuestion {
            instructions: "Does this express approval?".into(),
            criteria: ClassifierBoolCriteria {
                when_true: "Approval".into(),
                when_false: "No approval".into(),
            },
        }),
    );
    ClassifierContext {
        state: serde_json::json!({ "text": "yes" }).as_object().unwrap().clone(),
        images: None,
        questions,
    }
}

/// Answers every request with `approved: 0.9` and counts calls.
#[derive(Clone, Default)]
struct Approver {
    calls: Arc<Mutex<usize>>,
}

impl ProviderClassifier for Approver {
    fn classify<'a>(
        &'a self,
        model: &'a ClassifierModel,
        _context: &'a ClassifierContext,
        _options: ClassifierOptions,
    ) -> BoxFuture<'a, ClassifierResult> {
        *self.calls.lock().unwrap() += 1;
        Box::pin(async move {
            let mut result = ClassifierResult::empty(model);
            result.answers.insert(
                "approved".into(),
                ClassifierAnswer::Bool(ClassifierBoolAnswer { probability: 0.9 }),
            );
            result
        })
    }
}

fn classifier_of(models: &Models, provider: &str, id: &str) -> Option<ClassifierModel> {
    match models.get_model_of_type(ModelType::Classifier, provider, id)? {
        AnyModel::Classifier(model) => Some(model),
        _ => None,
    }
}

#[tokio::test]
async fn keeps_chat_and_classifier_entries_with_the_same_provider_and_id_separate() {
    let chat = chat_model("test", "test-chat", "shared");
    let classifier = classifier_model("test", "shared");
    let models = Models::default();
    models.set_provider(create_provider(
        CreateProviderOptions::new(
            "test",
            ProviderAuth::keyless("Test"),
            vec![chat.clone().into(), classifier.clone().into()],
            ProviderApi::none(),
        )
        .classifier("test-classifier", Approver::default()),
    ));

    assert_eq!(models.get_model("test", "shared"), Some(chat));
    assert_eq!(classifier_of(&models, "test", "shared"), Some(classifier.clone()));
    assert_eq!(
        models.get_models_of_type(ModelType::Classifier, None),
        vec![AnyModel::Classifier(classifier.clone())]
    );
    assert_eq!(models.get_all_models(None).len(), 2);
    assert_eq!(
        models.get_available_of_type(ModelType::Classifier, None).await.unwrap(),
        vec![AnyModel::Classifier(classifier.clone())]
    );
    let result = models
        .classify(&classifier, &context(), ClassifierOptions::default())
        .await;
    assert_eq!(
        result.answers["approved"],
        ClassifierAnswer::Bool(ClassifierBoolAnswer { probability: 0.9 })
    );
}

#[tokio::test]
async fn returns_error_results_for_unknown_and_unsupported_providers() {
    let models = Models::default();
    let ghost = models
        .classify(
            &classifier_model("ghost", "m"),
            &context(),
            ClassifierOptions::default(),
        )
        .await;
    assert_eq!(ghost.stop_reason, ClassifierStopReason::Error);
    assert_eq!(ghost.error_message.as_deref(), Some("Unknown provider: ghost"));

    models.set_provider(create_provider(CreateProviderOptions::new(
        "images-only",
        ProviderAuth::keyless("Test"),
        vec![classifier_model("images-only", "c").into()],
        chat_api("test-chat"),
    )));
    let unsupported = models
        .classify(
            &classifier_model("images-only", "c"),
            &context(),
            ClassifierOptions::default(),
        )
        .await;
    assert_eq!(
        unsupported.error_message.as_deref(),
        Some("Provider images-only does not support classification")
    );

    models.set_provider(create_provider(
        CreateProviderOptions::new(
            "wrong-api",
            ProviderAuth::keyless("Test"),
            Vec::new(),
            ProviderApi::none(),
        )
        .classifier("other-classifier", Approver::default()),
    ));
    let missing = models
        .classify(
            &classifier_model("wrong-api", "c"),
            &context(),
            ClassifierOptions::default(),
        )
        .await;
    assert_eq!(
        missing.error_message.as_deref(),
        Some("Provider wrong-api has no classifier implementation for \"test-classifier\"")
    );
}

#[test]
fn exposes_jev_only_through_classifier_catalog_accessors() {
    let Some(AnyModel::Classifier(jev)) = builtin_model_of_type(ModelType::Classifier, "typesafe", "jev-latest") else {
        panic!("missing TypeSafe Jev model");
    };
    assert_eq!(jev.api, "typesafe-system-one");
    assert_eq!(jev.provider, "typesafe");
    assert_eq!(jev.context_window, 64000);
    assert_eq!(all_builtin_models("typesafe"), [AnyModel::Classifier(jev.clone())]);

    let models = builtin_models(CreateModelsOptions::default());
    assert!(models.get_model("typesafe", "jev-latest").is_none());
    assert_eq!(classifier_of(&models, "typesafe", "jev-latest"), Some(jev));
}

/// `model` with the scheme and host of its base URL replaced by the mock server's.
fn on_server(mut model: ClassifierModel, server: &MockServer) -> ClassifierModel {
    let url = url::Url::parse(&model.base_url).unwrap();
    model.base_url = format!("{}{}", server.uri(), url.path());
    model
}

fn secret() -> ClassifierOptions {
    RequestOptions {
        api_key: Some("secret".into()),
        ..RequestOptions::default()
    }
    .into()
}

async fn serve(body: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    server
}

#[tokio::test]
async fn routes_typesafe_compatible_jev_models_to_their_endpoints() {
    let models = builtin_models(CreateModelsOptions::default());
    for (provider, id, url) in [
        (
            "vercel-ai-gateway",
            "typesafe-ai/jev",
            "https://ai-gateway.vercel.sh/typesafe/v1/systemone",
        ),
        ("opencode", "jev-1.13", "https://opencode.ai/inference/v1/systemone"),
        (
            "opencode",
            "jev-1.13-free",
            "https://opencode.ai/inference/v1/systemone",
        ),
    ] {
        let jev = classifier_of(&models, provider, id).unwrap_or_else(|| panic!("missing {provider} Jev model"));
        assert_eq!(jev.api, "typesafe-system-one");
        assert_eq!(jev.context_window, 32000);
        assert!(models.get_model(provider, id).is_none());
        assert_eq!(format!("{}/systemone", jev.base_url), url);

        let server = serve(json!({ "model": id, "answers": { "approved": { "type": "noul", "noul": 0.8 } } })).await;
        let result = models.classify(&on_server(jev, &server), &context(), secret()).await;

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].url.path(), url::Url::parse(url).unwrap().path());
        let body: Value = requests[0].body_json().unwrap();
        assert_eq!(body["model"], json!(id));
        assert_eq!(body["state"], json!(context().state));
        assert_eq!(requests[0].headers.get("authorization").unwrap(), "Bearer secret");
        assert_eq!(
            result.stop_reason,
            ClassifierStopReason::Stop,
            "{:?}",
            result.error_message
        );
        assert_eq!(
            result.answers["approved"],
            ClassifierAnswer::Bool(ClassifierBoolAnswer { probability: 0.8 })
        );
    }
}

#[tokio::test]
async fn rejects_images_for_classifier_models_without_image_input_before_calling_the_provider() {
    let classifier = classifier_model("test", "text-only");
    let approver = Approver::default();
    let models = Models::default();
    models.set_provider(create_provider(
        CreateProviderOptions::new(
            "test",
            ProviderAuth::keyless("Test"),
            vec![classifier.clone().into()],
            ProviderApi::none(),
        )
        .classifier("test-classifier", approver.clone()),
    ));

    let image = ImageContent {
        data: "aW1hZ2U=".into(),
        mime_type: "image/png".into(),
    };
    let result = models
        .classify(
            &classifier,
            &ClassifierContext {
                images: Some(vec![image]),
                ..context()
            },
            ClassifierOptions::default(),
        )
        .await;
    let without_images = models
        .classify(
            &classifier,
            &ClassifierContext {
                images: Some(Vec::new()),
                ..context()
            },
            ClassifierOptions::default(),
        )
        .await;

    assert_eq!(result.stop_reason, ClassifierStopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("Model test/text-only does not accept image input")
    );
    assert_eq!(without_images.stop_reason, ClassifierStopReason::Stop);
    assert_eq!(*approver.calls.lock().unwrap(), 1);
}

#[test]
fn lists_openai_gpt_6_luna_as_a_separate_classifier_entry() {
    let models = builtin_models(CreateModelsOptions::default());
    let luna = classifier_of(&models, "openai", "gpt-6-luna").expect("missing OpenAI Decisions model");
    assert_eq!(luna.api, "openai-decisions");
    assert_eq!(
        luna.input,
        [delta_ai::InputModality::Text, delta_ai::InputModality::Image]
    );
    assert_eq!(luna.context_window, 922_000);
    // The chat entry with the same id stays separate.
    assert_eq!(
        models
            .get_model("openai", "gpt-6-luna")
            .map(|model| model.api)
            .as_deref(),
        Some("openai-responses")
    );
}

#[tokio::test]
async fn routes_openai_gpt_6_luna_through_the_decisions_api_with_images() {
    let models = builtin_models(CreateModelsOptions::default());
    let luna = classifier_of(&models, "openai", "gpt-6-luna").expect("missing OpenAI Decisions model");
    assert_eq!(luna.base_url, "https://api.openai.com/v1");
    let server = serve(json!({ "answers": [{ "type": "predicate", "name": "approved", "probability": 0.8 }] })).await;
    let context = ClassifierContext {
        images: Some(vec![ImageContent {
            data: "aW1hZ2U=".into(),
            mime_type: "image/png".into(),
        }]),
        ..context()
    };

    let result = models.classify(&on_server(luna, &server), &context, secret()).await;

    let requests = server.received_requests().await.unwrap();
    assert_eq!(
        requests.iter().map(|request| request.url.path()).collect::<Vec<_>>(),
        ["/v1/decisions"]
    );
    assert_eq!(
        result.stop_reason,
        ClassifierStopReason::Stop,
        "{:?}",
        result.error_message
    );
    assert_eq!(
        result.answers["approved"],
        ClassifierAnswer::Bool(ClassifierBoolAnswer { probability: 0.8 })
    );
}

/// The built-in `openai` provider with an OAuth method attached, as it is once ChatGPT
/// OAuth is ported. The login flow itself is never run.
struct OpenAIWithOAuth {
    inner: Arc<dyn Provider>,
    auth: ProviderAuth,
}

impl OpenAIWithOAuth {
    fn new() -> Self {
        let inner = builtin_provider("openai").unwrap();
        let mut auth = inner.auth().clone();
        auth.oauth.get_or_insert_with(|| Arc::new(OpenRouterOAuth::default()));
        Self { inner, auth }
    }
}

impl Provider for OpenAIWithOAuth {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn auth(&self) -> &ProviderAuth {
        &self.auth
    }

    fn models(&self) -> Vec<delta_ai::Model> {
        self.inner.models()
    }

    fn all_models(&self) -> Vec<AnyModel> {
        self.inner.all_models()
    }

    fn filter_models(&self, models: Vec<delta_ai::Model>, credential: Option<&Credential>) -> Vec<delta_ai::Model> {
        self.inner.filter_models(models, credential)
    }

    fn filter_all_models(&self, models: &[AnyModel], credential: Option<&Credential>) -> Option<Vec<AnyModel>> {
        self.inner.filter_all_models(models, credential)
    }

    fn stream(
        &self,
        model: &delta_ai::Model,
        context: &delta_ai::TranscriptContext,
        options: delta_ai::StreamOptions,
    ) -> delta_ai::AssistantMessageEventStream {
        self.inner.stream(model, context, options)
    }

    fn stream_simple(
        &self,
        model: &delta_ai::Model,
        context: &delta_ai::TranscriptContext,
        options: delta_ai::SimpleStreamOptions,
    ) -> delta_ai::AssistantMessageEventStream {
        self.inner.stream_simple(model, context, options)
    }
}

async fn models_with_credential(credential: Credential) -> Models {
    let store = Arc::new(InMemoryCredentialStore::new());
    store
        .modify(
            "openai",
            Box::new(move |_| Box::pin(async move { Ok(Some(credential)) })),
        )
        .await
        .unwrap();
    let models = builtin_models(CreateModelsOptions {
        credentials: Some(store),
        ..CreateModelsOptions::default()
    });
    models.set_provider(OpenAIWithOAuth::new());
    models
}

#[tokio::test]
async fn lists_openai_decisions_models_only_for_api_key_credentials() {
    let with_api_key = models_with_credential(Credential::ApiKey(ApiKeyCredential {
        key: Some("secret".into()),
        env: None,
    }))
    .await;
    let with_oauth = models_with_credential(Credential::OAuth(OAuthCredential {
        access: "access".into(),
        refresh: "refresh".into(),
        expires: i64::MAX,
        extra: JsonObject::new(),
    }))
    .await;

    let ids = |models: Vec<AnyModel>| models.iter().map(|model| model.id().to_owned()).collect::<Vec<_>>();
    assert_eq!(
        ids(with_api_key
            .get_available_of_type(ModelType::Classifier, Some("openai"))
            .await
            .unwrap()),
        ["gpt-6-luna"]
    );
    assert!(
        with_oauth
            .get_available_of_type(ModelType::Classifier, Some("openai"))
            .await
            .unwrap()
            .is_empty()
    );
    // Chat models stay available with ChatGPT OAuth.
    assert!(
        with_oauth
            .get_available(Some("openai"))
            .await
            .unwrap()
            .iter()
            .any(|model| model.id == "gpt-6-luna")
    );
}

#[test]
fn the_openai_provider_hides_classifier_models_from_oauth_credentials() {
    let provider = builtin_provider("openai").unwrap();
    let models = provider.all_models();
    let oauth = Credential::OAuth(OAuthCredential {
        access: "access".into(),
        refresh: "refresh".into(),
        expires: 0,
        extra: JsonObject::new(),
    });
    let api_key = Credential::ApiKey(ApiKeyCredential {
        key: Some("secret".into()),
        env: None,
    });
    let classifiers = |credential: Option<&Credential>| {
        provider
            .filter_all_models(&models, credential)
            .unwrap()
            .into_iter()
            .filter(|model| model.model_type() == ModelType::Classifier)
            .count()
    };
    assert_eq!(classifiers(Some(&api_key)), 1);
    assert_eq!(classifiers(None), 1);
    assert_eq!(classifiers(Some(&oauth)), 0);
    // Chat models are kept.
    assert_eq!(
        provider.filter_all_models(&models, Some(&oauth)).unwrap().len(),
        models.len() - 1
    );
}

#[test]
fn routes_openrouter_classifier_models_through_the_system_one_api() {
    let models = builtin_models(CreateModelsOptions::default());
    let classifiers: Vec<&AnyModel> = all_builtin_models("openrouter")
        .iter()
        .filter(|model| model.model_type() == ModelType::Classifier)
        .collect();
    assert!(!classifiers.is_empty());
    for model in classifiers {
        let AnyModel::Classifier(classifier) = model else {
            unreachable!()
        };
        assert_eq!(classifier.api, "typesafe-system-one");
        assert_eq!(classifier.base_url, "https://openrouter.ai/api/v1");
        assert!(models.get_model("openrouter", &classifier.id).is_none());
        assert_eq!(
            classifier_of(&models, "openrouter", &classifier.id).as_ref(),
            Some(classifier)
        );
    }
}
