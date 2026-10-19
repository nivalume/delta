//! API implementations: one module per wire protocol. Each exposes a unit struct that
//! implements [`crate::ProviderStreams`], plus `stream` and `stream_simple` functions. Image
//! and classifier APIs implement [`crate::ProviderImages`] or [`crate::ProviderClassifier`]
//! instead, plus a `generate_images` or `classify` function.

pub mod common;
pub mod constrained_sampling;

pub mod anthropic_messages;
pub mod azure_openai_responses;
#[cfg(feature = "bedrock")]
pub mod bedrock_converse_stream;
pub mod cloudflare;
pub mod github_copilot_headers;
pub mod google_generative_ai;
pub mod google_shared;
#[cfg(feature = "vertex")]
pub mod google_vertex;
pub mod mistral_conversations;
pub mod openai_codex_responses;
pub mod openai_completions;
pub mod openai_responses;
pub mod openai_responses_shared;
pub mod pi_messages;

// Image generation and classification.
pub mod classifier_shared;
pub mod cloudflare_workers_ai_system_one;
pub mod llama_cpp_classify;
pub mod openai_decisions;
pub mod openrouter_images;
pub mod system_one_shared;
pub mod typesafe_system_one;
