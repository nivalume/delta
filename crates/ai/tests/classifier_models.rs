//! Port of upstream `test/classifier-models.test.ts`, as far as it tests the `Models` layer.
//! The cases that call a concrete classifier API (TypeSafe System One, OpenAI Decisions)
//! belong with those API implementations.

#![allow(clippy::unwrap_used)] // Test helpers outside `#[test]` functions.

mod support;

use futures::future::BoxFuture;
use indexmap::IndexMap;
use std::sync::{Arc, Mutex};

use support::chat_api;

use delta_ai::classifier::{ClassifierBoolAnswer, ClassifierBoolCriteria, ClassifierBoolQuestion};
use delta_ai::providers::builtin::builtin_models;
use delta_ai::providers::catalog::{all_builtin_models, builtin_model_of_type};
use delta_ai::{
    AnyModel, ClassifierAnswer, ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierQuestion,
    ClassifierResult, ClassifierStopReason, CreateModelsOptions, CreateProviderOptions, ImageContent, ModelType,
    Models, ProviderApi, ProviderAuth, ProviderClassifier, create_provider,
};
use support::{chat_model, classifier_model};

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
    assert_eq!(all_builtin_models("typesafe"), [AnyModel::Classifier(jev)]);
    // Upstream also checks the built-in `typesafe` provider, which is not ported yet.
}

#[test]
fn lists_typesafe_compatible_jev_models_as_classifiers_only() {
    let models = builtin_models(CreateModelsOptions::default());
    for (provider, id) in [
        ("vercel-ai-gateway", "typesafe-ai/jev"),
        ("opencode", "jev-1.13"),
        ("opencode", "jev-1.13-free"),
    ] {
        let jev = classifier_of(&models, provider, id).unwrap_or_else(|| panic!("missing {provider} Jev model"));
        assert_eq!(jev.api, "typesafe-system-one");
        assert_eq!(jev.context_window, 32000);
        assert!(models.get_model(provider, id).is_none());
    }
    // Upstream also sends a request to each TypeSafe-compatible endpoint; that belongs with
    // the `typesafe-system-one` API.
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
    // Upstream also sends a Decisions request; that belongs with the `openai-decisions` API.
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
