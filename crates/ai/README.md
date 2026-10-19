# delta-ai

Unified LLM API for Rust with provider collections, automatic auth resolution, token and cost tracking, and context hand-off between models in the middle of a session.

`delta-ai` (imported as `delta_ai`) is the Rust port of [`@earendil-works/pi-ai`](https://github.com/earendil-works/pi/tree/main/packages/ai). It is part of [delta](https://github.com/nivalume/pi-rs). Serialized types (messages, models, credentials) keep upstream's JSON shape, so contexts and `auth.json` files move between pi, pi-py and delta.

**Note**: The chat catalog only includes models that support tool calling. The catalog also lists image and classifier models from upstream, but this crate cannot run them yet (see [Image Generation](#image-generation) and [Classification](#classification)).

**Status**: in development. Nothing is published to crates.io yet. The design notes, including every difference from upstream, are in [docs/design/README.md](docs/design/README.md).

## Table of Contents

- [Supported Providers](#supported-providers)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Providers and Models](#providers-and-models)
  - [Provider Factories](#provider-factories)
  - [All Built-in Providers](#all-built-in-providers)
  - [Querying Models](#querying-models)
  - [Static Catalog Reads](#static-catalog-reads)
  - [Dynamic Providers](#dynamic-providers)
- [Auth](#auth)
  - [How Auth Resolves](#how-auth-resolves)
  - [Transforming Request Headers](#transforming-request-headers)
  - [Credential Store](#credential-store)
  - [Environment Variables](#environment-variables)
- [Tools](#tools)
  - [Defining Tools](#defining-tools)
  - [Handling Tool Calls](#handling-tool-calls)
  - [Streaming Tool Calls with Partial JSON](#streaming-tool-calls-with-partial-json)
  - [Validating Tool Arguments](#validating-tool-arguments)
  - [Complete Event Reference](#complete-event-reference)
- [Image Input](#image-input)
- [Image Generation](#image-generation)
- [Classification](#classification)
- [Thinking/Reasoning](#thinkingreasoning)
  - [Unified Interface](#unified-interface-stream_simplecomplete_simple)
  - [Provider-Specific Options](#provider-specific-options-streamcomplete)
  - [Streaming Thinking Content](#streaming-thinking-content)
- [Stop Reasons](#stop-reasons)
- [Telemetry](#telemetry)
- [Error Handling](#error-handling)
  - [Aborting Requests](#aborting-requests)
  - [Continuing After Abort](#continuing-after-abort)
  - [Debugging Provider Payloads](#debugging-provider-payloads)
  - [Observing Provider Stream Events](#observing-provider-stream-events)
- [Custom Providers](#custom-providers)
  - [create_provider()](#create_provider)
  - [Calling API Implementations Directly](#calling-api-implementations-directly)
  - [OpenAI Compatibility Settings](#openai-compatibility-settings)
- [Faux Provider for Tests](#faux-provider-for-tests)
- [Cross-Provider Handoffs](#cross-provider-handoffs)
- [System Messages](#system-messages)
- [Context Serialization](#context-serialization)
- [Browser Usage](#browser-usage)
- [Bundling and Tree Shaking](#bundling-and-tree-shaking)
- [OAuth Providers](#oauth-providers)
  - [Vertex AI](#vertex-ai)
  - [CLI Login](#cli-login)
  - [Programmatic OAuth](#programmatic-oauth)
- [Migrating from the Old Global API](#migrating-from-the-old-global-api)
- [Development](#development)
- [License](#license)

## Supported Providers

`builtin_provider_ids()` returns the providers below. Two of them need a Cargo feature (see [Bundling and Tree Shaking](#bundling-and-tree-shaking)).

| Provider | Id | API | Cargo feature |
| --- | --- | --- | --- |
| OpenAI | `openai` | `openai-responses` | |
| Ant Ling | `ant-ling` | `openai-completions` | |
| Azure OpenAI | `azure` | `azure-openai-responses`, `openai-completions` | |
| DeepSeek | `deepseek` | `openai-completions` | |
| NVIDIA NIM | `nvidia` | `openai-completions` | |
| Anthropic | `anthropic` | `anthropic-messages` | |
| Google | `google` | `google-generative-ai` | |
| Vertex AI | `google-vertex` | `google-vertex` | `vertex` |
| Mistral | `mistral` | `mistral-conversations` | |
| Groq | `groq` | `openai-completions` | |
| Cerebras | `cerebras` | `openai-completions` | |
| Cloudflare AI Gateway | `cloudflare-ai-gateway` | `anthropic-messages`, `openai-completions`, `openai-responses` | |
| Cloudflare Workers AI | `cloudflare-workers-ai` | `openai-completions` | |
| xAI | `xai` | `openai-responses` | |
| OpenRouter | `openrouter` | `openai-completions`, `anthropic-messages` | |
| Vercel AI Gateway | `vercel-ai-gateway` | `anthropic-messages` | |
| ZAI Coding Plan (Global, China) | `zai`, `zai-coding-cn` | `openai-completions` | |
| MiniMax (Global, China) | `minimax`, `minimax-cn` | `anthropic-messages` | |
| Together AI | `together` | `openai-completions` | |
| Baseten | `baseten` | `openai-completions` | |
| Hugging Face | `huggingface` | `openai-completions` | |
| Moonshot AI (Global, China) | `moonshotai`, `moonshotai-cn` | `openai-completions` | |
| Amazon Bedrock | `amazon-bedrock` | `bedrock-converse-stream` | `bedrock` |
| OpenCode Zen | `opencode` | `anthropic-messages`, `openai-completions`, `openai-responses`, `google-generative-ai` | |
| OpenCode Go | `opencode-go` | same as OpenCode Zen | |
| Fireworks | `fireworks` | `openai-completions`, `anthropic-messages` | |
| Kimi For Coding | `kimi-coding` | `openai-completions` | |
| Meta | `meta` | `openai-responses` | |
| Qwen Token Plan (existing, Individual, China) | `qwen-token-plan`, `qwen-token-plan-individual`, `qwen-token-plan-cn` | `openai-completions` | |
| Xiaomi MiMo (API billing, Token Plan China/Amsterdam/Singapore) | `xiaomi`, `xiaomi-token-plan-cn`, `xiaomi-token-plan-ams`, `xiaomi-token-plan-sgp` | `openai-completions` | |

Any OpenAI-compatible API (Ollama, vLLM, LM Studio, and so on) works through [`create_provider()`](#create_provider).

Not built in yet: OpenAI Codex, Radius, TypeSafe and GitHub Copilot. Their catalogs ship with the crate, but their APIs or auth flows are not ported.

## Installation

```bash
cargo add delta-ai
```

The crate is not on crates.io yet. Until it is, use a git dependency:

```bash
cargo add delta-ai --git https://github.com/nivalume/pi-rs
```

```toml
[dependencies]
delta-ai = { git = "https://github.com/nivalume/pi-rs", features = ["bedrock", "vertex"] }
```

Requests run on Tokio, so a Tokio runtime is required. The examples also use `futures` (for `StreamExt`), `serde_json`, and `tokio-util` (for `CancellationToken`). Upstream re-exports TypeBox for tool schemas; in Rust a tool schema is a plain `serde_json::Value`.

## Quick Start

You build a `Models` collection of providers and stream through it. The quickest start registers every built-in provider; apps that want fewer providers register them one by one (see [Provider Factories](#provider-factories)).

```rust,no_run
use delta_ai::providers::builtin::builtin_models;
use delta_ai::{
    AssistantBlock, AssistantMessageEvent, Context, CreateModelsOptions, Message, StreamOptions, Tool,
    ToolResultMessage,
};
use futures::StreamExt;
use serde_json::json;

#[tokio::main]
async fn main() {
    // A Models collection with every built-in provider registered
    let models = builtin_models(CreateModelsOptions::default());

    // Synchronous lookup against the collection
    let model = models.get_model("openai", "gpt-4o-mini").expect("model in catalog");

    // Tool parameters are a JSON Schema
    let tools = vec![Tool::new(
        "get_time",
        "Get the current time",
        json!({
            "type": "object",
            "properties": {
                "timezone": { "type": "string", "description": "Optional timezone (e.g., America/New_York)" }
            }
        }),
    )];

    // A conversation context (serializable and transferable between models)
    let mut context = Context::new(vec![Message::user("What time is it?")])
        .with_system_prompt("You are a helpful assistant.")
        .with_tools(tools);

    // Option 1: streaming with all event types.
    // Auth resolves through the provider (OPENAI_API_KEY from the environment here).
    let mut s = models.stream(&model, &context, StreamOptions::default());
    while let Some(event) = s.next().await {
        match event {
            AssistantMessageEvent::Start { message } => println!("Starting with {}", message.model),
            AssistantMessageEvent::TextStart { .. } => println!("\n[Text started]"),
            AssistantMessageEvent::TextDelta { delta, .. } => print!("{delta}"),
            AssistantMessageEvent::TextEnd { .. } => println!("\n[Text ended]"),
            AssistantMessageEvent::ThinkingStart { .. } => println!("[Model is thinking...]"),
            AssistantMessageEvent::ThinkingDelta { delta, .. } => print!("{delta}"),
            AssistantMessageEvent::ThinkingEnd { .. } => println!("[Thinking complete]"),
            AssistantMessageEvent::ToolCallStart { content_index, .. } => {
                println!("\n[Tool call started: index {content_index}]");
            }
            AssistantMessageEvent::ToolCallDelta { content_index, .. } => {
                // Partial tool arguments are being streamed
                let partial = s.partial().and_then(|message| message.content.get(content_index));
                if let Some(AssistantBlock::ToolCall(call)) = partial {
                    println!("[Streaming args for {}]", call.name);
                }
            }
            AssistantMessageEvent::ToolCallEnd { tool_call, .. } => {
                println!("\nTool called: {}", tool_call.name);
                println!("Arguments: {}", json!(tool_call.arguments));
            }
            AssistantMessageEvent::Done { reason, .. } => println!("\nFinished: {reason:?}"),
            AssistantMessageEvent::Error { error, .. } => {
                eprintln!("Error: {}", error.error_message.unwrap_or_default());
            }
        }
    }

    // Get the final message after streaming, add it to the context
    let final_message = s.result().await;
    context.messages.push(final_message.clone().into());

    // Handle tool calls if any
    let tool_calls: Vec<_> = final_message.tool_calls().cloned().collect();
    for call in &tool_calls {
        let result = if call.name == "get_time" {
            let timezone = call.arguments.get("timezone").and_then(|tz| tz.as_str()).unwrap_or("UTC");
            format!("{:?} ({timezone})", std::time::SystemTime::now())
        } else {
            "Unknown tool".to_owned()
        };
        // Add the tool result to the context (results can also carry images)
        context.messages.push(ToolResultMessage::text(call, result, false).into());
    }

    // Continue if there were tool calls
    if !tool_calls.is_empty() {
        let continuation = models.complete(&model, &context, StreamOptions::default()).await;
        println!("After tool execution: {}", continuation.text());
        context.messages.push(continuation.into());
    }

    println!("Total tokens: {} in, {} out", final_message.usage.input, final_message.usage.output);
    println!("Cost: ${:.4}", final_message.usage.cost.total);

    // Option 2: get the complete response without streaming
    let response = models.complete(&model, &context, StreamOptions::default()).await;
    for block in &response.content {
        match block {
            AssistantBlock::Text(text) => println!("{}", text.text),
            AssistantBlock::ToolCall(call) => println!("Tool: {}({})", call.name, json!(call.arguments)),
            AssistantBlock::Thinking(_) => {}
        }
    }
}
```

`Models::stream` returns the event stream at once; auth resolution and the request run in a Tokio task. `AssistantMessageEventStream` implements `futures::Stream`. `result()` drains the stream and returns the final message. Dropping the stream cancels the request.

Snippets in the rest of this README assume a `models` collection set up like this (with the relevant providers registered), plus a `model` and a `context`. These names are declared in hidden lines of each example.

## Providers and Models

A **provider** is the runtime unit: it owns its model catalog, its auth (API key resolution, OAuth refresh) and its request behavior. It implements the `Provider` trait. A `Models` collection holds providers and routes every request to the provider that owns the model. `Models` is cheap to clone; clones share state.

Providers share **API implementations** (the wire protocols, `ProviderStreams`): Anthropic models use `anthropic-messages`, OpenAI uses `openai-responses`, while Groq, Cerebras, OpenRouter and most others share `openai-completions`. Mixed-API providers (OpenRouter, OpenCode Zen, Fireworks, Azure, Cloudflare AI Gateway) dispatch per model on `model.api`.

### Provider Factories

Upstream has one module per provider. Most providers only pair an environment variable with an API implementation, so this crate keeps them in one table in `providers::builtin` and builds them by id. Anthropic and Azure also have their own factory functions. The Cloudflare providers are in `providers::cloudflare` (`cloudflare_ai_gateway_provider()`, `cloudflare_workers_ai_provider()`).

```rust
use delta_ai::Models;
use delta_ai::providers::builtin::{anthropic_provider, builtin_provider};

let models = Models::default();
models.set_provider(anthropic_provider());
models.set_provider_arc(builtin_provider("openrouter").expect("built-in provider"));
assert_eq!(models.providers().len(), 2);
```

`builtin_provider(id)` returns `None` for unknown providers, for providers that are not ported, and for `amazon-bedrock` / `google-vertex` when their Cargo feature is off. Every catalog is compiled into the binary; a provider's catalog is parsed on first access.

### All Built-in Providers

For apps that want everything (as in Quick Start):

```rust
use delta_ai::CreateModelsOptions;
use delta_ai::providers::builtin::{builtin_models, builtin_provider_ids, builtin_providers};

// A Models collection with every built-in provider registered
let models = builtin_models(CreateModelsOptions::default());

let ids = builtin_provider_ids();
assert!(ids.contains(&"anthropic"));
#[cfg(feature = "bedrock")]
assert!(ids.contains(&"amazon-bedrock"));
#[cfg(not(feature = "vertex"))]
assert!(!ids.contains(&"google-vertex"));

assert_eq!(builtin_providers().len(), models.providers().len());
```

`builtin_models()` takes the same options as `Models::new()` (`credentials`, `models_store`, `auth_context`). `builtin_providers()` returns the providers if you want to register them on your own collection.

### Querying Models

Reads are synchronous and return the last-known lists:

```rust
# #[tokio::main(flavor = "current_thread")]
# async fn main() -> Result<(), delta_ai::ModelsError> {
use delta_ai::providers::builtin::builtin_models;
use delta_ai::{AnyModel, CreateModelsOptions, ModelType};

let models = builtin_models(CreateModelsOptions::default());

let providers = models.providers(); // registered providers
let provider = models.provider("anthropic"); // one provider

let all = models.get_models(None); // every chat model across providers
let anthropic_models = models.get_models(Some("anthropic"));
let model = models.get_model("anthropic", "claude-sonnet-4-5");

for m in &anthropic_models {
    println!("{}: {}", m.id, m.name);
    println!("  API: {}", m.api);
    println!("  Context: {} tokens", m.context_window);
    println!("  Vision: {}", m.supports_images());
    println!("  Reasoning: {}", m.reasoning);
}

let images = models.get_models_of_type(ModelType::Image, Some("openrouter")); // Vec<AnyModel>
let image = models.get_model_of_type(ModelType::Image, "openrouter", "google/gemini-2.5-flash-image");
let everything = models.get_all_models(None); // Vec<AnyModel>
// Models whose provider has auth configured (environment, stored credentials)
let available = models.get_available(None).await?;
let available_images = models.get_available_of_type(ModelType::Image, None).await?;

for model in &everything {
    if let AnyModel::Image(image) = model {
        // image: &ImageModel
    }
}
# Ok(())
# }
```

`get_models()` / `get_model()` / `get_available()` return chat models (`Model`), which the stream methods accept. The `*_of_type` reads and `get_all_models()` / `get_all_available()` return `AnyModel`, an enum with `Chat`, `Image` and `Classifier` variants. Match on the variant, or use `AnyModel::model_type()` and `delta_ai::provider::is_model_type()`.

IDs are unique within each provider and type; one upstream model may have separate entries for different operations. On a provider, `Provider::models()` returns chat models and `Provider::all_models()` returns every type.

`Model::api` is a string. Upstream narrows the TypeScript type with `hasApi()`; in Rust you pass per-API options as an `ApiOptions` variant, and a variant that does not match `model.api` is ignored (see [Provider-Specific Options](#provider-specific-options-streamcomplete)).

### Static Catalog Reads

For tooling that wants the built-in catalog without a collection, use `providers::catalog`. Ids are plain strings; there are no generated literal types.

```rust
use delta_ai::ModelType;
use delta_ai::providers::catalog::{
    all_builtin_models, builtin_model, builtin_model_of_type, builtin_models, builtin_provider_ids,
};

let model = builtin_model("openai", "gpt-4o-mini").expect("in catalog");
assert_eq!(model.api, "openai-responses");
let flux = builtin_model_of_type(ModelType::Image, "openrouter", "google/gemini-2.5-flash-image");
let jev = builtin_model_of_type(ModelType::Classifier, "typesafe", "jev-latest");
let providers = builtin_provider_ids(); // every provider in the catalog, ported or not
let openrouter_chat = builtin_models("openrouter"); // Vec<Model>
let openrouter_all = all_builtin_models("openrouter"); // &[AnyModel]
assert!(flux.is_some() && jev.is_some());
```

`catalog::builtin_provider_ids()` lists every provider in the catalog (42 at pi-ai 1.3.0). `builtin::builtin_provider_ids()` lists only the providers this build can construct.

### Dynamic Providers

Providers may have dynamic model lists (a llama.cpp server, a live listing). Reads stay synchronous; fetching is an explicit async call:

```rust
# #[tokio::main(flavor = "current_thread")]
# async fn main() {
# let models = delta_ai::Models::default();
use delta_ai::ModelsRefreshOptions;

// get_models() returns the last-known list (empty before the first refresh)
let result = models
    .refresh(ModelsRefreshOptions {
        providers: Some(vec!["llamacpp".into()]), // refresh one provider
        ..Default::default()
    })
    .await;
// Refresh every dynamic provider concurrently, best effort
let result = models.refresh(ModelsRefreshOptions::default()).await;
let fresh = models.get_model("llamacpp", "qwen3-30b");
# }
```

Built-in providers are static, so `refresh()` skips them. Radius, the one upstream provider that is both static and dynamic, is not ported. See [create_provider()](#create_provider) for building a dynamic provider.

## Auth

Every provider owns its auth: how API keys resolve (stored credentials, environment variables, ambient sources such as AWS profiles or gcloud ADC) and, where a provider supplies one, OAuth refresh.

### How Auth Resolves

When you call `models.stream()`, the collection resolves auth through the owning provider and merges it into the request. Explicit per-request values always win:

```rust,no_run
# use delta_ai::{Context, Model, Models, StreamOptions};
# async fn example(models: Models, model: Model, context: Context) -> Result<(), delta_ai::ModelsError> {
use delta_ai::AnyModel;
use delta_ai::auth::AuthResolutionOverrides;

// Resolved through the provider (env var, stored credential, OAuth token):
models.complete(&model, &context, StreamOptions::default()).await;

// Explicit key wins over anything the provider would resolve:
let options = StreamOptions {
    api_key: Some("sk-explicit".into()),
    ..Default::default()
};
models.complete(&model, &context, options).await;

// Inspect resolution without making a request
let overrides = AuthResolutionOverrides::default();
let provider_auth = models.get_auth(&model.provider, &overrides).await?;
let model_auth = models.get_model_auth(&AnyModel::Chat(model.clone()), &overrides).await?;

match model_auth {
    Some(result) => {
        // e.g. "ANTHROPIC_API_KEY", "OAuth", "stored credential"
        println!("configured via {:?}", result.source);
        println!("{:?}", result.auth.headers); // provider auth headers + model.headers
    }
    None => println!("not configured"),
}
# Ok(())
# }
```

Rust has no overloading, so upstream's `getAuth(providerId)` and `getAuth(model)` are `get_auth()` and `get_model_auth()`. Both resolve credentials, refresh expired OAuth when needed, and may return an auth-derived `api_key`, `headers` or `base_url`. They return `Ok(None)` for unconfigured providers and `Err(ModelsError)` when something is broken (`ModelsErrorCode::OAuth`: token refresh failed, credential kept for re-login; `ModelsErrorCode::Auth`: key resolution or credential store failure). Request paths report the same failures as stream errors.

Cancellation in Rust is dropping the future, so `get_auth()`, `check_auth()`, `get_available()` and `logout()` take no signal. `login()` is cancelled through `AuthInteraction::cancel()`. OAuth refresh runs in its own Tokio task, so a dropped caller does not lose a rotated token.

### Transforming Request Headers

`StreamOptions::transform_headers` runs once after provider auth, `model.headers` and explicit `headers` have been merged, and before provider dispatch:

```rust,no_run
# use delta_ai::{Context, Model, Models};
# async fn example(models: Models, model: Model, context: Context) {
use delta_ai::http::headers_of;
use delta_ai::options::TransformHeaders;
use delta_ai::types::now_ms;
use delta_ai::{SimpleStreamOptions, StreamOptions};
use std::sync::Arc;

let transform: TransformHeaders = Arc::new(|mut headers| {
    Box::pin(async move {
        headers.insert("X-Request-ID".into(), Some(format!("req-{}", now_ms())));
        headers
    })
});
let options = SimpleStreamOptions {
    base: StreamOptions {
        headers: Some(headers_of([("X-Client", "my-app")])),
        transform_headers: Some(transform),
        ..Default::default()
    },
    ..Default::default()
};
let response = models.complete_simple(&model, &context, options).await;
# }
```

The ordering is:

```text
provider auth headers -> model.headers -> explicit options.headers -> transform_headers -> Provider::stream*()
```

Header names are merged case-insensitively. Explicit headers override auth and model headers, and the transform has final control. A `None` value suppresses an API default header of the same name.

Upstream puts `transformHeaders` in a `Models`-only option type. Here it is a field of `StreamOptions`; `Models` takes it out before it calls the provider, so providers never see it. Use it instead of calling `get_model_auth()` before `stream()`, which would resolve auth twice.

### Credential Store

Stored credentials (API keys entered interactively, OAuth tokens) live in a `CredentialStore`: one type-tagged credential per provider. The crate ships `InMemoryCredentialStore`; apps inject persistent storage by implementing the trait:

```rust
# #[tokio::main(flavor = "current_thread")]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
use delta_ai::auth::{AuthType, ModifyFn};
use delta_ai::providers::builtin::azure_provider;
use delta_ai::{CreateModelsOptions, Credential, CredentialStore, InMemoryCredentialStore, Models};
use serde_json::json;
use std::sync::Arc;

let store: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new()); // or a file-backed store
let models = Models::new(CreateModelsOptions {
    credentials: Some(store.clone()),
    ..Default::default()
});
// builtin_models() takes the same options
models.set_provider(azure_provider());

// The JSON shape of pi's auth.json entries
let credential: Credential = serde_json::from_value(json!({
    "type": "api_key",
    "key": "...",
    "env": { "AZURE_OPENAI_RESOURCE_NAME": "my-resource" }
}))?;

// modify() is the only write path: a serialized read-modify-write
let write: ModifyFn = Box::new(move |_current| Box::pin(async move { Ok(Some(credential)) }));
store.modify("azure", write).await?;

let check = models.check_auth("azure").await?.expect("configured");
assert_eq!(check.auth_type, AuthType::ApiKey);
assert_eq!(check.source.as_deref(), Some("stored credential"));
# Ok(())
# }
```

The trait has four methods: `read(provider_id)`, `list()` for non-secret `CredentialInfo { provider_id, auth_type }` metadata, `modify(provider_id, f)` and `delete(provider_id)`. `list()` must not resolve secrets or run configured key commands. OAuth token refresh runs inside `modify`, so concurrent requests (and processes, when the store locks across processes) cannot double-refresh a rotated token. A stored credential *owns* its provider: environment variables are only consulted when nothing is stored, and a failed refresh never falls back to an environment key.

API-key credentials use the same `type` discriminator as pi's `auth.json` and can carry provider-scoped env/config values in `env`, as shown above.

### Environment Variables

Built-in providers resolve these environment variables:

| Provider | Environment Variable(s) |
|----------|------------------------|
| OpenAI | `OPENAI_API_KEY` |
| Ant Ling | `ANT_LING_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_BASE_URL` (e.g. `https://{resource}.openai.azure.com`) or `AZURE_OPENAI_RESOURCE_NAME`. Root endpoints are normalized to `/openai/v1`. Optional: `AZURE_OPENAI_API_VERSION` (default `v1`), `AZURE_OPENAI_DEPLOYMENT_NAME_MAP`. |
| Anthropic | `ANTHROPIC_API_KEY`, `ANTHROPIC_OAUTH_TOKEN`, or `ANTHROPIC_AUTH_TOKEN` (sent as a bearer token) |
| DeepSeek | `DEEPSEEK_API_KEY` |
| NVIDIA NIM | `NVIDIA_API_KEY` |
| Google | `GEMINI_API_KEY` |
| Vertex AI | `GOOGLE_CLOUD_API_KEY`, or `GOOGLE_CLOUD_PROJECT` (or `GCLOUD_PROJECT`) + `GOOGLE_CLOUD_LOCATION` + ADC |
| Mistral | `MISTRAL_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Cerebras | `CEREBRAS_API_KEY` |
| Cloudflare AI Gateway | `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_GATEWAY_ID` |
| Cloudflare Workers AI | `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` |
| xAI | `XAI_API_KEY` |
| Fireworks | `FIREWORKS_API_KEY` |
| Together AI | `TOGETHER_API_KEY` |
| Baseten | `BASETEN_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Vercel AI Gateway | `AI_GATEWAY_API_KEY` |
| ZAI Coding Plan (Global) | `ZAI_API_KEY` |
| ZAI Coding Plan (China) | `ZAI_CODING_CN_API_KEY` |
| MiniMax (Global) | `MINIMAX_API_KEY` |
| MiniMax (China) | `MINIMAX_CN_API_KEY` |
| Moonshot AI / Moonshot AI (China) | `MOONSHOT_API_KEY` |
| Hugging Face | `HF_TOKEN` |
| OpenCode Zen / OpenCode Go | `OPENCODE_API_KEY` |
| Kimi For Coding | `KIMI_API_KEY` |
| Meta | `META_API_KEY` |
| Qwen Token Plan (existing catalog, Individual) | `QWEN_TOKEN_PLAN_API_KEY` |
| Qwen Token Plan (China) | `QWEN_TOKEN_PLAN_CN_API_KEY` |
| Xiaomi MiMo (API billing) | `XIAOMI_API_KEY` |
| Xiaomi MiMo Token Plan (China) | `XIAOMI_TOKEN_PLAN_CN_API_KEY` |
| Xiaomi MiMo Token Plan (Amsterdam) | `XIAOMI_TOKEN_PLAN_AMS_API_KEY` |
| Xiaomi MiMo Token Plan (Singapore) | `XIAOMI_TOKEN_PLAN_SGP_API_KEY` |

`qwen-token-plan-individual` and `qwen-token-plan` share the international endpoint and `QWEN_TOKEN_PLAN_API_KEY`. The Individual provider lists only the models documented for Individual subscriptions. Stored credentials are provider-scoped, so save the key under the provider id you register.

Amazon Bedrock (feature `bedrock`) resolves ambient AWS credentials (`AWS_BEARER_TOKEN_BEDROCK`, `AWS_PROFILE`, `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`, ECS task roles, web identity tokens); its API-key login flow supports bearer tokens, AWS profiles and the existing credential chain. Vertex AI (feature `vertex`) resolves either an API key or Application Default Credentials plus project and location, with an API-key login flow for API keys, ADC and service-account files.

The Cloudflare providers read each value from the stored credential first (the key, and `CLOUDFLARE_ACCOUNT_ID` / `CLOUDFLARE_GATEWAY_ID` in its `env`) and fall back to the environment per value, so a stored key can be combined with ids from the environment. Their login flows prompt for the key and the ids. AI Gateway sends the key as `cf-aig-authorization: Bearer <key>` and suppresses `Authorization` and `x-api-key`; an explicit `Authorization` request header passes a provider key through the gateway (BYOK).

```rust
use delta_ai::Credential;
use serde_json::json;

// A stored Cloudflare AI Gateway credential, as pi writes it to auth.json
let credential: Credential = serde_json::from_value(json!({
    "type": "api_key",
    "key": "...",
    "env": {
        "CLOUDFLARE_ACCOUNT_ID": "account-id",
        "CLOUDFLARE_GATEWAY_ID": "gateway-id"
    }
}))
.unwrap();
assert!(matches!(credential, Credential::ApiKey(_)));
```

The `env` method of `AuthContext` reads these variables. Inject your own `AuthContext` through `CreateModelsOptions::auth_context` for tests or sandboxes.

## Tools

Tools let a model call external systems. A tool's parameters are a JSON Schema (`serde_json::Value`). Upstream builds schemas with TypeBox; in Rust you write the schema as JSON, or generate it with a crate of your choice. Arguments are validated against the schema with the `jsonschema` crate, after the same lenient coercions upstream applies.

### Defining Tools

```rust
use delta_ai::Tool;
use serde_json::json;

let weather_tool = Tool::new(
    "get_weather",
    "Get current weather for a location",
    json!({
        "type": "object",
        "properties": {
            "location": { "type": "string", "description": "City name or coordinates" },
            "units": { "type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius" }
        },
        "required": ["location"]
    }),
);

// For Google API compatibility, write string enums as `"type": "string"` plus `"enum"`.
// `anyOf`/`const` patterns are not supported by Google.

let book_meeting_tool = Tool::new(
    "book_meeting",
    "Schedule a meeting",
    json!({
        "type": "object",
        "properties": {
            "title": { "type": "string", "minLength": 1 },
            "startTime": { "type": "string", "format": "date-time" },
            "endTime": { "type": "string", "format": "date-time" },
            "attendees": { "type": "array", "items": { "type": "string", "format": "email" }, "minItems": 1 }
        },
        "required": ["title", "startTime", "endTime", "attendees"]
    }),
);
```

### Constrained Sampling for Tools

Tools can opt in to provider-side constrained sampling. For JSON-schema tools, `StrictMode::Prefer` uses provider-side strict schema enforcement when supported and otherwise falls back to normal tool calling. `StrictMode::Require` fails the request when the active provider or model cannot honor it. `constrained_sampling: None` opts out; in JSON, `"constrainedSampling": false` means the same.

```rust
use delta_ai::Tool;
use delta_ai::types::{ConstrainedSampling, GrammarVariants, StrictMode};
use serde_json::json;

let mut strict_tool = Tool::new(
    "edit_file",
    "Edit a file",
    json!({
        "type": "object",
        "properties": { "path": { "type": "string" }, "content": { "type": "string" } },
        "required": ["path", "content"],
        "additionalProperties": false
    }),
);
strict_tool.constrained_sampling = Some(ConstrainedSampling::JsonSchema {
    strict: StrictMode::Prefer,
});

let mut patch_tool = Tool::new(
    "apply_patch",
    "Apply a patch",
    json!({
        "type": "object",
        "properties": { "input": { "type": "string" } },
        "required": ["input"],
        "additionalProperties": false
    }),
);
patch_tool.constrained_sampling = Some(ConstrainedSampling::Grammar {
    variants: GrammarVariants {
        openai_lark: Some("start: /.+/s".into()),
        openai_regex: None,
    },
});
```

Strict JSON-schema constrained sampling is supported for OpenAI, Anthropic, supported Amazon Bedrock Converse models, Mistral, and Gemini 3 tool calls through the Google Generative AI and Vertex adapters. Google uses `VALIDATED` function-calling mode (or `ANY` when explicitly requested); earlier Gemini versions fall back for `Prefer` and reject `Require`. Bedrock strict-tool support comes from catalog metadata; custom Bedrock models can set `compat.supportsStrictMode`.

OpenAI Responses and Chat Completions can also emit grammar-constrained custom tools with OpenAI Lark or regex grammars. If both variants are set, Lark wins. Grammar tools are used only when the model's compat sets `supportsOpenAIGrammarTools` (the catalog sets it for GPT-5+ models on endpoints that pass custom tools through); otherwise the tool falls back to a normal function tool. Grammar-capable models reject a grammar configuration without a supported variant. A native grammar tool must have an object parameter schema with exactly one required string property.

### Handling Tool Calls

Tool results use content blocks and can include both text and images:

```rust,no_run
# use delta_ai::{Model, Models, StreamOptions};
# async fn execute_weather_api(_: &delta_ai::JsonObject) -> serde_json::Value { serde_json::Value::Null }
# async fn example(models: Models, model: Model, weather_tool: delta_ai::Tool) -> Result<(), Box<dyn std::error::Error>> {
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use delta_ai::types::now_ms;
use delta_ai::{Context, ImageContent, Message, ToolResultMessage, UserBlock};

let mut context =
    Context::new(vec![Message::user("What is the weather in London?")]).with_tools(vec![weather_tool]);

let response = models.complete(&model, &context, StreamOptions::default()).await;
context.messages.push(response.clone().into());

// Check for tool calls in the response
for call in response.tool_calls() {
    // Execute your tool with the arguments.
    // See "Validating Tool Arguments" for validation.
    let result = execute_weather_api(&call.arguments).await;

    // Add a text tool result
    context.messages.push(ToolResultMessage::text(call, result.to_string(), false).into());
}

// Tool results can also include images (for vision-capable models)
let image = std::fs::read("chart.png")?;
context.messages.push(
    ToolResultMessage {
        tool_call_id: "tool_xyz".into(),
        tool_name: "generate_chart".into(),
        content: vec![
            UserBlock::text("Generated chart showing temperature trends"),
            UserBlock::Image(ImageContent {
                data: STANDARD.encode(image),
                mime_type: "image/png".into(),
            }),
        ],
        is_error: false,
        timestamp: now_ms(),
        ..Default::default()
    }
    .into(),
);
# Ok(())
# }
```

### Streaming Tool Calls with Partial JSON

During streaming, tool call arguments are parsed as they arrive, so a UI can update before the arguments are complete. Events carry no `partial` message in this crate; `AssistantMessageEventStream::partial()` returns the message as of the event just taken:

```rust,no_run
# use delta_ai::{Context, Model, Models, StreamOptions};
# async fn example(models: Models, model: Model, context: Context) {
use delta_ai::{AssistantBlock, AssistantMessageEvent};
use futures::StreamExt;

let mut s = models.stream(&model, &context, StreamOptions::default());

while let Some(event) = s.next().await {
    match event {
        AssistantMessageEvent::ToolCallDelta { content_index, .. } => {
            // The arguments hold the best-effort parse of the JSON received so far.
            // BE DEFENSIVE: they may be incomplete.
            let block = s.partial().and_then(|message| message.content.get(content_index));
            if let Some(AssistantBlock::ToolCall(call)) = block
                && call.name == "write_file"
                && let Some(path) = call.arguments.get("path").and_then(|v| v.as_str())
            {
                // Show the file path before the content is complete
                println!("Writing to: {path}");
                // Content might be partial or missing
                if let Some(content) = call.arguments.get("content").and_then(|v| v.as_str()) {
                    let preview: String = content.chars().take(100).collect();
                    println!("Content preview: {preview}...");
                }
            }
        }
        AssistantMessageEvent::ToolCallEnd { tool_call, .. } => {
            // Here the arguments are complete (but not yet validated)
            println!("Tool completed: {} {:?}", tool_call.name, tool_call.arguments);
        }
        _ => {}
    }
}
# }
```

**Important notes about partial tool arguments:**
- After a `ToolCallDelta` event, `arguments` in `partial()` holds the best-effort parse of partial JSON
- Fields may be missing or incomplete; always check for existence before use
- String values may be truncated mid-word
- Arrays may be incomplete
- Nested objects may be partially populated
- `arguments` is always an object (`JsonObject`), empty at minimum
- The Google APIs do not stream function call arguments. You receive a single `ToolCallDelta` with the full arguments.
- If the complete arguments text is not a JSON object, `ToolCall::arguments` is empty and `ToolCall::arguments_error` holds the reason; validation then fails with that message.

### Validating Tool Arguments

When you run your own tool execution loop, use `validate_tool_call` to validate arguments before passing them to your tools:

```rust,no_run
# use delta_ai::{Context, Model, Models, StreamOptions, Tool};
# async fn execute_my_tool(_: &str, _: serde_json::Value) -> String { String::new() }
# async fn example(models: Models, model: Model, mut context: Context, tools: Vec<Tool>) {
use delta_ai::validation::validate_tool_call;
use delta_ai::{AssistantMessageEvent, ToolResultMessage};
use futures::StreamExt;

let mut s = models.stream(&model, &context.clone().with_tools(tools.clone()), StreamOptions::default());

while let Some(event) = s.next().await {
    if let AssistantMessageEvent::ToolCallEnd { tool_call, .. } = event {
        // Validate arguments against the tool's schema
        match validate_tool_call(&tools, &tool_call) {
            Ok(arguments) => {
                let result = execute_my_tool(&tool_call.name, arguments).await;
                context.messages.push(ToolResultMessage::text(&tool_call, result, false).into());
            }
            // Validation failed: return the error as a tool result so the model can retry
            Err(error) => {
                context.messages.push(ToolResultMessage::text(&tool_call, error.to_string(), true).into());
            }
        }
    }
}
# }
```

`validate_tool_call` returns the coerced arguments as a `serde_json::Value` (for example `"5"` becomes `5` when the schema asks for a number). `validate_tool_arguments(tool, call)` validates against one tool.

### Complete Event Reference

Successful generation follows `Start → updates* → Done`. A failure after generation starts follows `Start → updates* → Error`. Request setup (auth, building the request) may fail before generation starts; the stream then contains only `Error`. Calling an API implementation directly without auth also gives an `Error` event; nothing panics or returns early.

Upstream gives every non-terminal event a shared, mutable `partial` message. In this crate, events carry no `partial`. The stream reduces each event it yields into its own copy, and `AssistantMessageEventStream::partial()` returns the exact message as of the last yielded event. To make that possible, some events carry extra block data (tool call id and name at start, signatures at end). Text and ordinary thinking blocks start empty and grow through matching deltas until the authoritative `*End`; redacted thinking may be complete at `ThinkingStart` and emit no deltas. Usage and response ids are only updated by the terminal event.

All `AssistantMessageEvent` variants, with upstream's event type name (also returned by `event.kind()`):

| Variant | Upstream type | Description | Fields |
|---------|---------------|-------------|--------|
| `Start` | `start` | Stream begins | `message`: response metadata, normally with no content |
| `TextStart` | `text_start` | Text block starts | `content_index` |
| `TextDelta` | `text_delta` | Text chunk received | `content_index`, `delta` |
| `TextEnd` | `text_end` | Text block complete | `content_index`, `content`: full text, `signature` |
| `ThinkingStart` | `thinking_start` | Thinking block starts | `content_index`, `thinking`, `redacted`, `signature` |
| `ThinkingDelta` | `thinking_delta` | Thinking chunk received | `content_index`, `delta` |
| `ThinkingEnd` | `thinking_end` | Thinking block complete | `content_index`, `content`: full thinking, `signature` |
| `ToolCallStart` | `toolcall_start` | Tool call begins | `content_index`, `id`, `name`, `arguments` (provider-specific, often empty) |
| `ToolCallDelta` | `toolcall_delta` | Tool arguments streaming | `content_index`, `delta`: JSON chunk |
| `ToolCallEnd` | `toolcall_end` | Tool call complete | `content_index`, `tool_call`: complete, but not schema-validated |
| `Done` | `done` | Stream complete | `reason`: `DoneReason` (`Stop`, `Length`, `ToolUse`), `message`: final assistant message |
| `Error` | `error` | Error occurred | `reason`: `ErrorReason` (`Error`, `Aborted`), `error`: assistant message with partial content and `error_message` |

Events for different content blocks are not guaranteed to be contiguous. Providers may emit deltas for text, thinking and tool calls in the same upstream chunk, so events may interleave, for example `TextStart`, `TextDelta`, `ToolCallStart`, `TextDelta`, `ToolCallDelta`. Use `content_index` to associate each delta or end event with its block.

## Image Input

Models with vision capabilities can process images. Check `model.input` (or `model.supports_images()`). When a context with images is sent to a model without image input, the images are replaced with placeholder text.

```rust,no_run
# use delta_ai::{Models, StreamOptions};
# async fn example(models: Models) -> Result<(), Box<dyn std::error::Error>> {
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use delta_ai::{AssistantBlock, Context, ImageContent, Message, UserBlock, UserContent};

let model = models.get_model("openai", "gpt-4o-mini").expect("model in catalog");

// Check if the model supports images
if model.supports_images() {
    println!("Model supports vision");
}

let image = STANDARD.encode(std::fs::read("image.png")?);

let context = Context::new(vec![Message::user(UserContent::Blocks(vec![
    UserBlock::text("What is in this image?"),
    UserBlock::Image(ImageContent {
        data: image,
        mime_type: "image/png".into(),
    }),
]))]);
let response = models.complete(&model, &context, StreamOptions::default()).await;

// Access the response
for block in &response.content {
    if let AssistantBlock::Text(text) = block {
        println!("{}", text.text);
    }
}
# Ok(())
# }
```

## Image Generation

Not ported yet. The catalog contains image models (`ImageModel`, listed by `get_models_of_type(ModelType::Image, ..)`), but there is no `generate_images()` and providers have no image implementations.

## Classification

Not ported yet. The catalog contains classifier models (`ClassifierModel`, listed by `get_models_of_type(ModelType::Classifier, ..)`), but there is no `classify()` and none of the classifier APIs (TypeSafe System One, OpenAI Decisions, llama.cpp) are implemented.

## Thinking/Reasoning

Many models can show their reasoning. Check `model.reasoning`. Reasoning options sent to a non-reasoning model are ignored.

### Unified Interface (stream_simple/complete_simple)

```rust,no_run
# use delta_ai::Models;
# async fn example(models: Models) {
use delta_ai::{AssistantBlock, Context, Message, ModelThinkingLevel, SimpleStreamOptions, ThinkingLevel};

// Many models across providers support thinking/reasoning
let model = models.get_model("anthropic", "claude-sonnet-4-5").expect("model in catalog");
// or models.get_model("openai", "gpt-5-mini")
// or models.get_model("google", "gemini-2.5-flash")

// Check if the model supports reasoning
if model.reasoning {
    println!("Model supports reasoning/thinking");
}

// Use the simplified reasoning option
let options = SimpleStreamOptions {
    reasoning: Some(ThinkingLevel::Medium), // Minimal, Low, Medium, High, Xhigh, Max
    ..Default::default()
};
let context = Context::new(vec![Message::user("Solve: 2x + 5 = 13")]);
let response = models.complete_simple(&model, &context, options).await;

// Access thinking and text blocks
for block in &response.content {
    match block {
        AssistantBlock::Thinking(thinking) => println!("Thinking: {}", thinking.thinking),
        AssistantBlock::Text(text) => println!("Response: {}", text.text),
        AssistantBlock::ToolCall(_) => {}
    }
}

// xhigh and max are opt-in, model-specific levels
let has_xhigh = model.supported_thinking_levels().contains(&ModelThinkingLevel::Xhigh);
# }
```

`Xhigh` and `Max` are model-specific, opt-in levels. Use `model.supported_thinking_levels()` to see whether a model exposes them, and `model.clamp_thinking_level()` to pick the nearest supported level.

### Provider-Specific Options (stream/complete)

`models.stream()` / `complete()` take `StreamOptions`, whose `api` field holds the options of one API as an `ApiOptions` variant. A variant that does not match `model.api` is ignored, so no type narrowing is needed:

```rust,no_run
# use delta_ai::{Context, Models};
# async fn example(models: Models, context: Context) {
use delta_ai::options::{AnthropicOptions, GoogleOptions, GoogleThinking, OpenAIResponsesOptions, ReasoningEffort};
use delta_ai::{ApiOptions, StreamOptions};

// OpenAI reasoning (o1, o3, gpt-5)
let openai_model = models.get_model("openai", "gpt-5-mini").expect("model in catalog");
let options = StreamOptions {
    api: Some(ApiOptions::OpenAIResponses(OpenAIResponsesOptions {
        reasoning_effort: Some(ReasoningEffort::Medium),
        reasoning_summary: Some(Some("detailed".into())), // OpenAI Responses API only
        ..Default::default()
    })),
    ..Default::default()
};
models.complete(&openai_model, &context, options).await;

// Anthropic thinking
let anthropic_model = models.get_model("anthropic", "claude-sonnet-4-5").expect("model in catalog");
let options = StreamOptions {
    api: Some(ApiOptions::AnthropicMessages(AnthropicOptions {
        thinking_enabled: Some(true),
        thinking_budget_tokens: Some(8192), // Optional token limit
        ..Default::default()
    })),
    ..Default::default()
};
models.complete(&anthropic_model, &context, options).await;

// Google Gemini thinking
let google_model = models.get_model("google", "gemini-2.5-flash").expect("model in catalog");
let options = StreamOptions {
    api: Some(ApiOptions::GoogleGenerativeAi(GoogleOptions {
        thinking: Some(GoogleThinking {
            enabled: true,
            budget_tokens: Some(8192), // -1 for dynamic, 0 to disable
            level: None,
        }),
        ..Default::default()
    })),
    ..Default::default()
};
models.complete(&google_model, &context, options).await;
# }
```

The option types live in `delta_ai::options`: `AnthropicOptions`, `OpenAICompletionsOptions`, `OpenAIResponsesOptions`, `AzureOpenAIResponsesOptions`, `GoogleOptions`, `GoogleVertexOptions`, `MistralOptions`, `BedrockOptions`. Custom APIs read `ApiOptions::Custom(JsonObject)`.

### Streaming Thinking Content

When streaming, thinking content is delivered through specific events:

```rust,no_run
# use delta_ai::{Context, Model, Models};
# async fn example(models: Models, model: Model, context: Context) {
use delta_ai::{AssistantMessageEvent, SimpleStreamOptions, ThinkingLevel};
use futures::StreamExt;

let options = SimpleStreamOptions {
    reasoning: Some(ThinkingLevel::High),
    ..Default::default()
};
let mut s = models.stream_simple(&model, &context, options);

while let Some(event) = s.next().await {
    match event {
        AssistantMessageEvent::ThinkingStart { .. } => println!("[Model started thinking]"),
        AssistantMessageEvent::ThinkingDelta { delta, .. } => print!("{delta}"), // Stream thinking content
        AssistantMessageEvent::ThinkingEnd { .. } => println!("\n[Thinking complete]"),
        _ => {}
    }
}
# }
```

## Stop Reasons

Every `AssistantMessage` has a `stop_reason` field (`StopReason`) that tells how the generation ended:

- `Pending`: only in partial messages, while the stop reason is not known yet
- `Stop`: the final message the model will produce this turn
- `Length`: output hit the maximum token limit
- `ToolUse`: the model is calling tools and expects tool results
- `Error`: an error occurred during generation
- `Aborted`: the request was cancelled

`StopReason` also has a `Deferred` variant for upstream's deferred responses; no ported API produces it. The JSON values are upstream's strings (`"stop"`, `"toolUse"`, ...); `StopReason::as_str()` returns them.

`AssistantMessage` may also have `response_id`, a provider-specific upstream response or message id when the API exposes one. Do not assume it is always present.

## Telemetry

Not ported yet. Upstream records spans and metrics through `@earendil-works/pi-telemetry`; this crate will add them once `delta-telemetry` is ported.

## Error Handling

Request failures after a stream is returned never panic or return early: when a request ends with an error (including aborts and tool call validation errors), the stream emits an `Error` event and the final message carries the details. Setup failures may emit `Error` without `Start`; failures after generation begins emit `Start`, any observed updates, then `Error`.

```rust,no_run
# use delta_ai::{Context, Model, Models, StreamOptions};
# async fn example(models: Models, model: Model, context: Context) {
use delta_ai::AssistantMessageEvent;
use futures::StreamExt;

let mut s = models.stream(&model, &context, StreamOptions::default());

// In streaming
while let Some(event) = s.next().await {
    if let AssistantMessageEvent::Error { reason, error } = event {
        // reason is ErrorReason::Error or ErrorReason::Aborted.
        // error is the AssistantMessage with partial content.
        eprintln!("Error ({reason:?}): {:?}", error.error_message);
        println!("Partial content: {:?}", error.content);
    }
}

// The final message has the error details
let message = s.result().await;
if message.stop_reason.is_failure() {
    eprintln!("Request failed: {:?}", message.error_message);
    // message.content holds any partial content received before the error
    // message.usage holds partial token counts and costs
}
# }
```

When using a `Models` collection, auth failures (OAuth refresh failed, provider not configured, unknown provider) surface as a stream error with `StopReason::Error`. Calling an API implementation directly without an API key also gives a stream error (`No API key for provider: ...`).

Non-request operations (`get_auth()`, `check_auth()`, `login()`, credential and model stores) return `Result<_, ModelsError>`. `ModelsError::code()` returns a `ModelsErrorCode`.

### Aborting Requests

`StreamOptions::cancel` takes a `tokio_util::sync::CancellationToken`. Aborted requests have `StopReason::Aborted`:

```rust,no_run
# use delta_ai::{Model, Models};
# async fn example(models: Models, model: Model) {
use delta_ai::{AssistantMessageEvent, Context, ErrorReason, Message, StopReason, StreamOptions};
use futures::StreamExt;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

let cancel = CancellationToken::new();

// Abort after 2 seconds
let timer = cancel.clone();
tokio::spawn(async move {
    tokio::time::sleep(Duration::from_secs(2)).await;
    timer.cancel();
});

let context = Context::new(vec![Message::user("Write a long story")]);
let options = StreamOptions {
    cancel: Some(cancel),
    ..Default::default()
};
let mut s = models.stream(&model, &context, options);

while let Some(event) = s.next().await {
    match event {
        AssistantMessageEvent::TextDelta { delta, .. } => print!("{delta}"),
        AssistantMessageEvent::Error { reason, error } => {
            let label = if reason == ErrorReason::Aborted { "Aborted" } else { "Error" };
            println!("{label}: {:?}", error.error_message);
        }
        _ => {}
    }
}

// Get the result (may be partial if aborted)
let response = s.result().await;
if response.stop_reason == StopReason::Aborted {
    println!("Request was aborted: {:?}", response.error_message);
    println!("Partial content received: {:?}", response.content);
    println!("Tokens used: {:?}", response.usage);
}
# }
```

Dropping the stream also cancels the request: the producer sees the closed channel, stops reading and closes the HTTP connection. Upstream keeps the request running in that case. To let a request finish without reading its events, await `result()`.

### Continuing After Abort

Aborted messages can stay in the conversation context, and the conversation continues in later requests:

```rust,no_run
# use delta_ai::{Model, Models};
# async fn example(models: Models, model: Model) {
use delta_ai::{Context, Message, StreamOptions};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

let mut context = Context::new(vec![Message::user("Explain quantum computing in detail")]);

// The first request is aborted after 2 seconds
let cancel = CancellationToken::new();
let timer = cancel.clone();
tokio::spawn(async move {
    tokio::time::sleep(Duration::from_secs(2)).await;
    timer.cancel();
});
let options = StreamOptions {
    cancel: Some(cancel),
    ..Default::default()
};
let partial = models.complete(&model, &context, options).await;

// Add the partial response to the context
context.messages.push(partial.into());
context.messages.push(Message::user("Please continue"));

// Continue the conversation
let continuation = models.complete(&model, &context, StreamOptions::default()).await;
# }
```

When the context is sent, assistant messages with stop reason `Error` or `Aborted` are dropped (`transform::transform_messages`, same as upstream's message transform). The model therefore does not see the partial answer; it sees the user's follow-up.

### Debugging Provider Payloads

Use the `on_payload` callback to inspect the request payload sent to the provider. This helps with request formatting issues and provider validation errors.

```rust,no_run
# use delta_ai::{Context, Model, Models};
# async fn example(models: Models, model: Model, context: Context) {
use delta_ai::StreamOptions;
use delta_ai::options::OnPayload;
use std::sync::Arc;

let on_payload: OnPayload = Arc::new(|payload, _model| {
    println!("Provider payload: {payload:#}");
    // Ok(Some(value)) replaces the payload; an error fails the request
    Box::pin(async { Ok(None) })
});
let options = StreamOptions {
    on_payload: Some(on_payload),
    ..Default::default()
};
let response = models.complete(&model, &context, options).await;
# }
```

The callback works with `stream`, `complete`, `stream_simple` and `complete_simple`. `on_response` is called with the HTTP status and headers before the body is read.

### Observing Provider Stream Events

Use `on_provider_stream_event` to inspect provider-specific fields that `AssistantMessage` does not include. The callback receives each parsed provider event (`serde_json::Value`) before it is normalized. It is not guaranteed to be the original HTTP bytes or SSE frame.

```rust,no_run
# use delta_ai::{Context, Models};
# async fn example(models: Models, context: Context) {
use delta_ai::StreamOptions;
use delta_ai::http::headers_of;
use delta_ai::options::OnProviderStreamEvent;
use std::sync::Arc;

let model = models.get_model("openrouter", "openrouter/auto").expect("model in catalog");
let observe: OnProviderStreamEvent = Arc::new(|data, _model| {
    if let Some(metadata) = data.get("openrouter_metadata") {
        println!("{metadata}");
    }
    Box::pin(async { Ok(()) })
});
let options = StreamOptions {
    headers: Some(headers_of([("X-OpenRouter-Metadata", "enabled")])),
    on_provider_stream_event: Some(observe),
    ..Default::default()
};
let response = models.complete(&model, &context, options).await;
# }
```

Callbacks are awaited in stream order, so slow callbacks delay stream consumption, and a returned error fails the request. The Bedrock and Vertex adapters use official SDKs and can only expose the fields the SDK keeps.

## Custom Providers

### create_provider()

`create_provider()` builds a provider from parts: identity, auth, a model list and an API implementation (`ProviderApi`). Use it for local inference servers, proxies, or any OpenAI- or Anthropic-compatible endpoint. Models are plain data; the JSON form below is the same as upstream's model literals.

```rust,no_run
# async fn example(context: delta_ai::Context) -> Result<(), Box<dyn std::error::Error>> {
use delta_ai::api::openai_completions::OpenAICompletionsApi;
use delta_ai::{CreateProviderOptions, Model, Models, ProviderApi, ProviderAuth, StreamOptions, create_provider};
use serde_json::json;

let ollama_model: Model = serde_json::from_value(json!({
    "id": "llama-3.1-8b",
    "name": "Llama 3.1 8B (Ollama)",
    "api": "openai-completions",
    "provider": "ollama",
    "baseUrl": "http://localhost:11434/v1",
    "reasoning": false,
    "input": ["text"],
    "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
    "contextWindow": 128000,
    "maxTokens": 32000
}))?;

let ollama = create_provider(
    CreateProviderOptions::new(
        "ollama",
        // Every provider declares auth; keyless local servers resolve as configured with no key.
        ProviderAuth::keyless("Ollama"),
        vec![ollama_model.into()],
        ProviderApi::single(OpenAICompletionsApi),
    )
    .name("Ollama")
    .base_url("http://localhost:11434/v1"),
);

let models = Models::default();
models.set_provider(ollama);

let model = models.get_model("ollama", "llama-3.1-8b").expect("registered");
models.complete(&model, &context, StreamOptions::default()).await;
# Ok(())
# }
```

For providers with real keys, `env_api_key_auth(display_name, env_vars)` gives the standard behavior (a stored credential wins, then the first set environment variable):

```rust
use delta_ai::api::openai_completions::OpenAICompletionsApi;
use delta_ai::{CreateProviderOptions, ProviderApi, ProviderAuth, create_provider, env_api_key_auth};

let proxy = create_provider(CreateProviderOptions::new(
    "my-proxy",
    ProviderAuth::api_key(env_api_key_auth("My proxy API key", &["MY_PROXY_API_KEY"])),
    vec![/* ... */],
    ProviderApi::single(OpenAICompletionsApi),
));
```

Mixed-API providers pass a map keyed by `model.api`; each model dispatches to its API's implementation. A model whose API has no entry gets an error stream.

```rust
use delta_ai::api::anthropic_messages::AnthropicMessagesApi;
use delta_ai::api::openai_responses::OpenAIResponsesApi;
use delta_ai::{
    CreateProviderOptions, ProviderApi, ProviderAuth, ProviderStreams, create_provider, env_api_key_auth,
};
use std::collections::HashMap;
use std::sync::Arc;

let apis: HashMap<String, Arc<dyn ProviderStreams>> = HashMap::from([
    ("anthropic-messages".to_owned(), Arc::new(AnthropicMessagesApi) as Arc<dyn ProviderStreams>),
    ("openai-responses".to_owned(), Arc::new(OpenAIResponsesApi)),
]);
let gateway = create_provider(CreateProviderOptions::new(
    "my-gateway",
    ProviderAuth::api_key(env_api_key_auth("Gateway key", &["GATEWAY_API_KEY"])),
    vec![/* models with api "anthropic-messages" or "openai-responses" */],
    ProviderApi::ByApi(apis),
));
```

Provider-wide endpoint or request changes belong in the provider's API implementation: wrap the `ProviderStreams` you pass as `api` so every request goes through the change before dispatch. The built-in Azure, OpenCode and Cloudflare providers do this; the Cloudflare providers fill account/gateway placeholders in the base URL from the resolved provider env (`providers::cloudflare::CloudflareStreams`).

```rust
use delta_ai::api::openai_completions::OpenAICompletionsApi;
use delta_ai::{
    AssistantMessageEventStream, CreateProviderOptions, Model, ProviderApi, ProviderAuth, ProviderStreams,
    SimpleStreamOptions, StreamOptions, TranscriptContext, create_provider, env_api_key_auth,
};

struct TenantStreams<S> {
    inner: S,
    tenant: String,
}

impl<S> TenantStreams<S> {
    fn with_tenant(&self, model: &Model) -> Model {
        let mut model = model.clone();
        model.base_url = model.base_url.replace("{tenant}", &self.tenant);
        model
    }
}

impl<S: ProviderStreams> ProviderStreams for TenantStreams<S> {
    fn stream(&self, model: &Model, context: &TranscriptContext, options: StreamOptions) -> AssistantMessageEventStream {
        self.inner.stream(&self.with_tenant(model), context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.inner.stream_simple(&self.with_tenant(model), context, options)
    }
}

let tenant_gateway = create_provider(CreateProviderOptions::new(
    "tenant-gateway",
    ProviderAuth::api_key(env_api_key_auth("Gateway key", &["GATEWAY_API_KEY"])),
    vec![/* ... */],
    ProviderApi::single(TenantStreams {
        inner: OpenAICompletionsApi,
        tenant: "acme".into(),
    }),
));
```

Dynamic model lists use `fetch_models`, which can return models of every type. `Models::refresh()` refreshes every configured dynamic provider and passes its effective API-key or refreshed OAuth credential in `RefreshModelsContext::credential`. A `ModelsStore` persists dynamic catalogs; both stores default to in-memory implementations.

```rust
# #[tokio::main(flavor = "current_thread")]
# async fn main() {
use delta_ai::api::openai_completions::OpenAICompletionsApi;
use delta_ai::models::InMemoryModelsStore;
use delta_ai::provider::{FetchModels, RefreshModelsContext};
use delta_ai::{
    AnyModel, CreateModelsOptions, CreateProviderOptions, Models, ModelsError, ModelsRefreshOptions, ProviderApi,
    ProviderAuth, create_provider,
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

async fn fetch_models_from_server(
    base_url: Option<String>,
    cancel: CancellationToken,
) -> Result<Vec<AnyModel>, ModelsError> {
    // GET {base_url}/models and map each entry to a Model. Honor `cancel` while waiting.
    let _ = (base_url, cancel);
    Ok(Vec::new())
}

let models = Models::new(CreateModelsOptions {
    models_store: Some(Arc::new(InMemoryModelsStore::default())),
    ..Default::default()
});

let fetch: FetchModels = Arc::new(|context: RefreshModelsContext| {
    Box::pin(fetch_models_from_server(context.base_url, context.cancel))
});
let mut options = CreateProviderOptions::new(
    "llamacpp",
    ProviderAuth::keyless("llama.cpp"),
    Vec::new(),
    ProviderApi::single(OpenAICompletionsApi),
)
.base_url("http://localhost:8080/v1");
options.fetch_models = Some(fetch);

models.set_provider(create_provider(options));
let result = models.refresh(ModelsRefreshOptions::default()).await;
if result.aborted {
    println!("refresh cancelled");
}
for (provider, error) in &result.errors {
    eprintln!("{provider}: {error}");
}
# }
```

`Models::refresh()` runs until done when `ModelsRefreshOptions::cancel` is unset. Providers always receive a `RefreshModelsContext::cancel` token and must honor it for network requests and other blocking work. When the caller's token fires, `refresh()` returns at once with `aborted: true`.

Use `providers: Some(vec!["openrouter".into()])` to restrict work to selected providers, `allow_network: Some(false)` to restore persisted catalogs without network access, or `force: true` to bypass provider freshness checks. Model reads stay synchronous and return the last restored or refreshed list.

`create_provider()` handles publication and persistence. A hand-written `Provider::refresh_models()` receives the read-only `context.stored` snapshot and publishes through `context.publish(ModelsPublication { persist, update })`. `Persist::Keep` leaves storage unchanged, `Persist::Write(entry)` writes it, and `Persist::Delete` deletes it. Publication is generation-checked: `publish` resolves `false` when a newer refresh or `set_provider()` superseded it, and `update` then does not run. Put in-memory catalog changes in `update` rather than changing state before publication.

Custom models can carry `headers` (e.g. proxies behind bot detection) and `compat` settings. `Models::get_model_auth()` includes the model headers, and the stream methods merge them before explicit request headers and `transform_headers`. See [OpenAI Compatibility Settings](#openai-compatibility-settings).

Some OpenAI-compatible servers do not understand the `developer` role used for reasoning-capable models. For those, set `compat.supportsDeveloperRole` to `false` so the system prompt is sent as a `system` message. If the server also does not support `reasoning_effort`, set `compat.supportsReasoningEffort` to `false`. This commonly applies to Ollama, vLLM, SGLang and similar servers.

Use the model-level `thinkingLevelMap` to describe model-specific thinking controls. Keys are thinking levels (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`). Missing levels up to `high` use provider defaults; `xhigh` and `max` are opt-in and need a non-null entry. String values are sent to the provider, `null` marks a level unsupported, and maps may skip levels.

```rust
use delta_ai::compat::ModelCompat;
use delta_ai::{Model, ModelThinkingLevel};
use serde_json::json;

let ollama_reasoning_model: Model = serde_json::from_value(json!({
    "id": "gpt-oss:20b",
    "name": "GPT-OSS 20B (Ollama)",
    "api": "openai-completions",
    "provider": "ollama",
    "baseUrl": "http://localhost:11434/v1",
    "reasoning": true,
    "input": ["text"],
    "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
    "contextWindow": 131072,
    "maxTokens": 32000,
    "thinkingLevelMap": {
        "minimal": null,
        "low": null,
        "medium": null,
        "high": "high",
        "xhigh": null
    },
    "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
    }
}))
.expect("valid model");

assert_eq!(
    ollama_reasoning_model.supported_thinking_levels(),
    [ModelThinkingLevel::Off, ModelThinkingLevel::High]
);
assert!(matches!(ollama_reasoning_model.compat, Some(ModelCompat::OpenAICompletions(_))));
```

### Calling API Implementations Directly

Each API implementation is a module under `delta_ai::api` with free functions `stream` and `stream_simple`, plus a unit struct that implements `ProviderStreams`. Direct calls bypass provider auth and context normalization: pass `api_key` explicitly and normalize the context with `normalize_context()`:

```rust,no_run
# use delta_ai::{Context, Model};
# fn example(claude_model: Model, context: Context) {
use delta_ai::api::anthropic_messages;
use delta_ai::options::AnthropicOptions;
use delta_ai::{ApiOptions, StreamOptions, normalize_context};

let options = StreamOptions {
    api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
    api: Some(ApiOptions::AnthropicMessages(AnthropicOptions {
        thinking_enabled: Some(true),
        thinking_budget_tokens: Some(2048),
        ..Default::default()
    })),
    ..Default::default()
};
let s = anthropic_messages::stream(&claude_model, &normalize_context(&context), options);
# }
```

Built-in API implementations:

| API id | Module | `ProviderStreams` struct | Options (`ApiOptions` variant) |
|--------|--------|--------------------------|--------------------------------|
| `anthropic-messages` | `api::anthropic_messages` | `AnthropicMessagesApi` | `AnthropicMessages(AnthropicOptions)` |
| `openai-completions` | `api::openai_completions` | `OpenAICompletionsApi` | `OpenAICompletions(OpenAICompletionsOptions)` |
| `openai-responses` | `api::openai_responses` | `OpenAIResponsesApi` | `OpenAIResponses(OpenAIResponsesOptions)` |
| `azure-openai-responses` | `api::azure_openai_responses` | `AzureOpenAIResponsesApi` | `AzureOpenAIResponses(AzureOpenAIResponsesOptions)` |
| `google-generative-ai` | `api::google_generative_ai` | `GoogleGenerativeAiApi` | `GoogleGenerativeAi(GoogleOptions)` |
| `google-vertex` (feature `vertex`) | `api::google_vertex` | `GoogleVertexApi` | `GoogleVertex(GoogleVertexOptions)` |
| `mistral-conversations` | `api::mistral_conversations` | `MistralConversationsApi` | `MistralConversations(MistralOptions)` |
| `bedrock-converse-stream` (feature `bedrock`) | `api::bedrock_converse_stream` | `BedrockConverseStreamApi` | `BedrockConverseStream(BedrockOptions)` |

`openai-codex-responses` is not ported. The hand-written APIs (Anthropic, OpenAI, Google Generative AI, Mistral) are always compiled; they use `reqwest` and the helpers in `delta_ai::http`. Bedrock and Vertex use the vendors' official Rust SDKs. Upstream's custom `fetch` and injected SDK `client` options map to `StreamOptions::http_client` (a `reqwest::Client`).

### OpenAI Compatibility Settings

The `openai-completions` API is implemented by many providers with small differences. By default the library detects compatibility settings from `baseUrl` for known OpenAI-compatible providers (Cerebras, xAI, DeepSeek, NVIDIA NIM, Together AI, zAI, OpenCode, Cloudflare Workers AI and others). For custom proxies or unknown endpoints, set the model's `compat` field. `Model::compat` is a `ModelCompat` enum, parsed with the schema that matches `model.api`; for `openai-completions` it is `OpenAICompletionsCompat`, for the Responses APIs `OpenAIResponsesCompat`. All fields are optional and unknown fields are kept in `extra`.

`OpenAICompletionsCompat` (JSON name, then meaning):

| Field | Meaning |
|-------|---------|
| `supportsStore` | Provider supports the `store` field (default: true) |
| `supportsDeveloperRole` | Provider supports the `developer` role instead of `system` (default: true) |
| `supportsReasoningEffort` | Provider supports `reasoning_effort` (default: true) |
| `supportsUsageInStreaming` | Provider supports `stream_options: { include_usage: true }` (default: true) |
| `supportsStrictMode` | Provider supports `strict` in tool definitions (default: false; set in the catalog for capable models) |
| `supportsOpenAIGrammarTools` | Emit OpenAI custom Lark/regex grammar tools; false falls back to function tools (default: false) |
| `supportsMidConvoSystemMessages` | Model accepts system messages after the conversation started; false folds them into the leading prompt (default: false) |
| `supportsMidConvoToolAdditions` | System messages can add tools mid-conversation via Kimi-style `tools` system messages; needs `supportsMidConvoSystemMessages` (default: false) |
| `sendSessionAffinityHeaders` | Send session-affinity data from `session_id` (default: true for OpenRouter, false otherwise) |
| `sessionAffinityFormat` | `openai`, `openai-nosession` or `openrouter` (default: detected) |
| `maxTokensField` | `max_completion_tokens` or `max_tokens` (default: `max_completion_tokens`) |
| `requiresToolResultName` | Tool results need the `name` field (default: false) |
| `requiresAssistantAfterToolResult` | Tool results must be followed by an assistant message (default: false) |
| `requiresThinkingAsText` | Thinking blocks must be sent as text (default: false) |
| `requiresReasoningContentOnAssistantMessages` | Replayed assistant messages need empty `reasoning_content` when reasoning is on (default: detected for DeepSeek) |
| `thinkingFormat` | `openai`, `openrouter`, `deepseek`, `together`, `baseten`, `zai`, `qwen`, `chat-template`, `qwen-chat-template`, `string-thinking` or `ant-ling` (default: `openai`) |
| `chatTemplateKwargs` / `chatTemplateArgs` | `chat_template_kwargs` / `chat_template_args` values; `{ "$var": "thinking.enabled" }` (or `thinking.effort`, `thinking.budget`) inserts a thinking value |
| `thinkingTokenBudgetField` | Top-level field that caps reasoning tokens: `thinking_token_budget`, `thinking_budget` or `thinking_budget_tokens` (off by default) |
| `supportsThinkingTokenBudget` | Alias for `thinkingTokenBudgetField: "thinking_token_budget"` (default: false) |
| `cacheControlFormat` | `anthropic`: Anthropic-style `cache_control` on the system prompt, last tool and last user/assistant text |
| `openRouterRouting` / `vercelGatewayRouting` | Routing preferences for OpenRouter / Vercel AI Gateway |

The struct also has `supportsFinishReason`, `zaiToolStream`, `supportsLongCacheRetention` and `vllmPriority`, which upstream's compat schema defines but its README does not describe.

`OpenAIResponsesCompat` has `supportsDeveloperRole`, `sessionAffinityFormat` (affects headers only, not the `prompt_cache_key` body field), `supportsLongCacheRetention` (`prompt_cache_retention: "24h"`, default true), `supportsStrictMode`, `supportsOpenAIGrammarTools`, `supportsMidConvoSystemMessages`, `supportsAdditionalTools`, `supportsToolSearch` and a few more.

OpenRouter requests send `x-session-id` from `session_id` when prompt caching is enabled. Chat Completions and Anthropic Messages both detect OpenRouter endpoints unless `sendSessionAffinityHeaders` is false. On Anthropic-compatible models, `sessionAffinityFormat: "openrouter"` selects `x-session-id`; when unset, `x-session-affinity` is used. Explicit request headers take precedence over generated headers.

If `compat` is not set, the library falls back to URL-based detection. If `compat` is partly set, unset fields use the detected defaults. This is useful for:

- **LiteLLM proxies**: may not support the `store` field
- **Custom inference servers**: may use non-standard field names
- **Self-hosted endpoints**: may support a different set of features

## Faux Provider for Tests

`FauxProvider` is an in-memory provider with scripted responses for tests and demos. It needs no network:

```rust
# #[tokio::main(flavor = "current_thread")]
# async fn main() {
use delta_ai::providers::faux::{
    FauxModelDefinition, FauxOptions, FauxProvider, faux_assistant_message, faux_text, faux_thinking,
    faux_tool_call,
};
use delta_ai::{CacheRetention, Context, Message, Models, StreamOptions, ToolResultMessage};
use futures::StreamExt;
use serde_json::json;

// Set tokens_per_second to pace output in real time
let faux = FauxProvider::new(FauxOptions::default());

let models = Models::default();
models.set_provider(faux.clone());

let model = faux.model();
let mut context = Context::new(vec![Message::user("Summarize Cargo.toml and then call echo")]);

faux.set_responses(vec![
    faux_assistant_message(vec![
        faux_thinking("Need to inspect package metadata first."),
        faux_tool_call("echo", json!({ "text": "Cargo.toml" })),
    ])
    .into(),
]);

let options = StreamOptions {
    session_id: Some("session-1".into()),
    cache_retention: Some(CacheRetention::Short),
    ..Default::default()
};
let first = models.complete(&model, &context, options).await;
let call = first.tool_calls().next().expect("tool call").clone();
context.messages.push(first.into());
context.messages.push(ToolResultMessage::text(&call, "Cargo.toml contents here", false).into());

faux.set_responses(vec![
    faux_assistant_message(vec![
        faux_thinking("Now I can summarize the tool output."),
        faux_text("Here is the summary."),
    ])
    .into(),
]);

let mut s = models.stream(&model, &context, StreamOptions::default());
while let Some(event) = s.next().await {
    println!("{}", event.kind());
}

// Optional: several faux models for model-switching tests
let multi_model = FauxProvider::new(FauxOptions {
    provider: Some("faux-multi".into()),
    models: vec![
        FauxModelDefinition::new("faux-fast"),
        FauxModelDefinition {
            reasoning: true,
            ..FauxModelDefinition::new("faux-thinker")
        },
    ],
    ..Default::default()
});
models.set_provider(multi_model.clone());
let thinker = multi_model.model_by_id("faux-thinker").expect("faux model");

assert!(thinker.reasoning);
assert_eq!(faux.pending_response_count(), 0);
assert_eq!(faux.state().call_count, 2);
# }
```

Notes:
- Responses are consumed from a queue in request start order.
- If the queue is empty, the faux provider returns an assistant error message with `error_message: "No more faux responses queued"`.
- Use `faux.set_responses(..)` to replace the remaining queue and `faux.append_responses(..)` to add more. `FauxResponse::Factory` builds a reply from the request context.
- `faux.model()` returns the first model and `faux.model_by_id(id)` a specific one.
- Use `faux_assistant_message(..)` for scripted replies; its stop reason is `ToolUse` when it contains tool calls, else `Stop`. Use `faux_text(..)`, `faux_thinking(..)` and `faux_tool_call(..)` to build content blocks.
- Usage is estimated at roughly 1 token per 4 characters. When `session_id` is set and cache retention is not `None`, prompt cache reads and writes are simulated.
- Tool call arguments stream in chunks through `ToolCallDelta` events.
- By default chunks are emitted as fast as possible. Set `tokens_per_second` to pace them in real time.
- `FauxProvider` itself implements `Provider`, and clones share the queue (upstream separates the handle from the provider). For independent concurrent flows, create separate faux providers with distinct `provider` ids.

## Cross-Provider Handoffs

The library supports handoffs between providers within one conversation. You can switch models mid-conversation and keep the context, including text, tool calls and tool results.

When messages from one model are sent to a different model, they are transformed for compatibility:

- **User and tool result messages** are passed through unchanged (images become placeholder text for models without image input)
- **Assistant messages from the same provider, API and model** are kept as they are, including thinking blocks
- **Assistant messages from a different model** lose their thinking blocks. Signatures and encrypted reasoning only work with the model that produced them, and readable thinking sent as plain text makes the new model treat it as part of the reply. Text signatures and tool call thought signatures are removed too, and tool call ids are normalized where the target API restricts them
- **Tool calls and regular text** are otherwise kept unchanged
- **Tool calls without results** get a synthetic error result ("No result provided")

```rust,no_run
# async fn example() {
use delta_ai::providers::builtin::{anthropic_provider, builtin_provider};
use delta_ai::{Context, Message, Models, SimpleStreamOptions, StreamOptions, ThinkingLevel};

let models = Models::default();
models.set_provider(anthropic_provider());
models.set_provider_arc(builtin_provider("openai").expect("built-in provider"));
models.set_provider_arc(builtin_provider("google").expect("built-in provider"));

let mut context = Context::default();

// Start with Claude
let claude = models.get_model("anthropic", "claude-sonnet-4-5").expect("model in catalog");
context.messages.push(Message::user("What is 25 * 18?"));
let options = SimpleStreamOptions {
    reasoning: Some(ThinkingLevel::Medium),
    ..Default::default()
};
context.messages.push(models.complete_simple(&claude, &context, options).await.into());

// Switch to GPT-5: it sees Claude's answer, but not Claude's thinking
let gpt5 = models.get_model("openai", "gpt-5-mini").expect("model in catalog");
context.messages.push(Message::user("Is that calculation correct?"));
context.messages.push(models.complete(&gpt5, &context, StreamOptions::default()).await.into());

// Switch to Gemini
let gemini = models.get_model("google", "gemini-2.5-flash").expect("model in catalog");
context.messages.push(Message::user("What was the original question?"));
let gemini_response = models.complete(&gemini, &context, StreamOptions::default()).await;
# }
```

All providers can handle messages from other providers: text, tool calls and results (including images), and thinking blocks (dropped when the model changes). Assistant messages that ended with an error or an abort are not replayed. You can start with a fast model, switch to a more capable one for complex reasoning, or continue after a provider outage.

## System Messages

`Context::system_prompt` and `Context::tools` are shorthand for a leading system message. The public entry points (`Models::stream()`, `stream_simple()`, `complete()`, `complete_simple()`) take a `Context` and call `normalize_context()` once. Everything below them, including `Provider::stream()`, `ProviderStreams` and the API modules, receives the resulting `TranscriptContext`, which only has messages. Only `normalize_context()` and the transcript helpers can build a `TranscriptContext`. The transcript can also carry later system messages that change the prompt or the tool set without rewriting the history.

`SystemMessage` has these fields:

| Field | Meaning |
|-------|---------|
| `content` | Leading message: the base prompt. Later messages: added instructions |
| `sections` | Named prompt sections; later messages replace them by name, `None` removes one |
| `tools_added` | Tools that become available here |
| `tools_removed` | Tools that stop being available here |
| `timestamp` | Milliseconds since the Unix epoch |

Sections are opaque text rendered verbatim after `content`, joined by blank lines. Keep each one self-delimiting (a tag, a heading) so the model can relate an update to the original. Replaying every system message in order gives the current prompt and tools; the replay helpers take the message list:

```rust
use delta_ai::{Message, get_current_system_prompt, get_current_tools};
use serde_json::json;

let messages: Vec<Message> = serde_json::from_value(json!([
    { "role": "system", "content": "You are helpful.", "sections": { "rules": "<rules>Be brief.</rules>" },
      "toolsAdded": [{ "name": "read", "description": "Read a file", "parameters": { "type": "object" } }],
      "timestamp": 1 },
    { "role": "user", "content": "hi", "timestamp": 2 },
    { "role": "system", "content": "", "sections": { "rules": "<rules>Be thorough.</rules>" },
      "toolsRemoved": [{ "name": "read" }], "timestamp": 3 }
]))
.expect("valid transcript");

assert_eq!(get_current_system_prompt(&messages), "You are helpful.\n\n<rules>Be thorough.</rules>");
assert!(get_current_tools(&messages).is_empty());
```

A custom `Provider` or `ProviderStreams` reads the prompt and tools the same way from `context.messages()`; a `TranscriptContext` has no `system_prompt` or `tools`.

Models that accept system messages mid-conversation (`supportsMidConvoSystemMessages` in the model's compat, set by the catalog for verified models) receive each later system message in place, so the cached prefix stays intact; section changes are framed by name. Every other model receives `transcript::collapse_system_messages(..)`: the replayed prompt and current tools as the leading system message, with later system messages dropped. Anthropic models that also set `supportsMidConvoToolChanges` send tool changes as native blocks (`inline-tools-2026-09-15` beta): the top-level tool list holds the initial tools plus a stable deferred placeholder, later tools are defined in `tool_addition` blocks, and removals are `tool_removal` references, so tool changes do not invalidate the prompt cache. That needs at least one initial tool; otherwise the current tool list is sent at the top level. OpenAI Responses models with `supportsAdditionalTools` or `supportsToolSearch` anchor additive tool changes at their message; everything else sends the current tool list at the top level.

## Context Serialization

`Context` implements `serde::Serialize` and `Deserialize`, so you can persist conversations, implement chat history, or move contexts between services. The JSON is the same as upstream's:

```rust,no_run
# use delta_ai::Models;
# async fn example(models: Models) -> Result<(), Box<dyn std::error::Error>> {
use delta_ai::{Context, Message, StreamOptions};

let mut context = Context::new(vec![Message::user("What is Rust?")]).with_system_prompt("You are a helpful assistant.");

let model = models.get_model("openai", "gpt-4o-mini").expect("model in catalog");
let response = models.complete(&model, &context, StreamOptions::default()).await;
context.messages.push(response.into());

// Serialize the entire context
let serialized = serde_json::to_string(&context)?;

// Save to a database, a file, etc.
std::fs::write("conversation.json", &serialized)?;

// Later: deserialize and continue the conversation
let mut restored: Context = serde_json::from_str(&std::fs::read_to_string("conversation.json")?)?;
restored.messages.push(Message::user("Tell me more about its type system"));

// Continue with any model
let new_model = models.get_model("anthropic", "claude-haiku-4-5").expect("model in catalog");
let continuation = models.complete(&new_model, &restored, StreamOptions::default()).await;
# Ok(())
# }
```

Models are plain serializable data too, with no functions or implementations attached, so persisting "which model was this conversation using" is one `serde_json::to_string(&model)`.

> **Note**: If the context contains images (base64, as in [Image Input](#image-input)), they are serialized too.

## Browser Usage

Not applicable. `delta-ai` is a native library that runs on Tokio; there is no browser or WebAssembly build. A web app should call a backend that uses the library and keeps the API keys.

## Bundling and Tree Shaking

Rust has no bundler or tree shaking step. Cargo features choose what is compiled:

| Feature | API | Dependencies |
|---------|-----|--------------|
| (default) | `anthropic-messages`, `openai-completions`, `openai-responses`, `azure-openai-responses`, `google-generative-ai`, `mistral-conversations` | `reqwest` (hand-written clients) |
| `bedrock` | `bedrock-converse-stream` (provider `amazon-bedrock`) | `aws-sdk-bedrockruntime`, `aws-config`, `aws-smithy-types` |
| `vertex` | `google-vertex` (provider `google-vertex`) | `google-cloud-aiplatform-v1`, `google-cloud-auth`, `google-cloud-gax` |
| `full` | all of the above | |

```toml
[dependencies]
delta-ai = { git = "https://github.com/nivalume/pi-rs", features = ["bedrock"] }
```

Rules:

- The default build has no vendor SDK dependencies.
- `bedrock` and `vertex` add the official AWS and Google Cloud SDKs. Without the feature, the API module does not exist and `builtin_provider()` returns `None` for that provider. Its catalog entries stay listed in `providers::catalog`.
- Every catalog is embedded with `include_str!` (about 1700 entries at pi-ai 1.3.0). A provider's catalog is parsed the first time it is read.
- There is no per-provider split. `builtin_provider()` matches on the id, so calling it links every built-in provider of the enabled features.

### Provider-Scoped Environment Overrides

Set `env` in `StreamOptions` to scope provider configuration to a request. Values in `env` are used before process environment variables for provider auth and configuration such as Cloudflare account IDs, Azure OpenAI settings, Vertex project/location, Bedrock settings and `PI_CACHE_RETENTION`.

```rust,no_run
# use delta_ai::{Context, Models};
# async fn example(models: Models, context: Context) {
use delta_ai::StreamOptions;
use delta_ai::types::ProviderEnv;

let model = models.get_model("azure", "gpt-4o-mini").expect("model in catalog");

let options = StreamOptions {
    env: Some(ProviderEnv::from([
        ("AZURE_OPENAI_API_KEY".to_owned(), "...".to_owned()),
        ("AZURE_OPENAI_RESOURCE_NAME".to_owned(), "my-resource".to_owned()),
    ])),
    ..Default::default()
};
let response = models.complete(&model, &context, options).await;
# }
```

Use this when one process needs different provider settings per request, or when ambient environment variables should not leak into a provider call. Proxy settings (`HTTP_PROXY`, `HTTPS_PROXY`) are read by the HTTP client from the process environment, not from `env`.

## OAuth Providers

Several providers support OAuth authentication instead of static API keys:

- **Anthropic** (Claude Pro/Max subscription)
- **OpenRouter** (OAuth PKCE that mints a user-controlled API key)
- **xAI** (SuperGrok or X Premium subscription, device-code login)

Upstream also offers OAuth for OpenAI (Sign in with ChatGPT), OpenAI Codex, GitHub Copilot, Kimi For Coding, Meta and Radius. Those flows are being ported; until then the built-in providers among them authenticate with API keys.

Each of these providers carries an `OAuthAuth` in `provider.auth().oauth` with three operations: `login(interaction, options)` uses the provider-neutral `AuthInteraction::prompt()`/`notify()` protocol and returns a credential, `refresh(credential)` refreshes expiring credentials, and `to_auth(credential)` derives request auth. Login is cancelled through `AuthInteraction::cancel()`. Refresh is automatic: `models.get_auth(provider_id, ..)` and request paths refresh expired tokens inside `CredentialStore::modify`, so concurrent requests cannot double-refresh. OpenRouter's flow returns a permanent API key, so its refresh is a no-op. An Anthropic OAuth access token is used as the API key; the `anthropic-messages` API recognizes it and sends it as a bearer token with Claude Code's identity and beta headers.

```rust,no_run
# #[tokio::main(flavor = "current_thread")]
# async fn main() -> Result<(), delta_ai::ModelsError> {
use delta_ai::auth::{AuthEvent, AuthInteraction, AuthPrompt, AuthType, LoginOptions};
use delta_ai::providers::builtin::anthropic_provider;
use delta_ai::{Context, Message, Models, ModelsError, StreamOptions};
use futures::future::BoxFuture;

struct Terminal;

impl AuthInteraction for Terminal {
    fn prompt(&self, prompt: AuthPrompt) -> BoxFuture<'_, Result<String, ModelsError>> {
        Box::pin(async move {
            match prompt {
                // Anthropic asks for the login method first.
                AuthPrompt::Select { options, .. } => Ok(options[0].id.clone()),
                // Raced against a local callback server; `cancel` fires when the server wins.
                AuthPrompt::ManualCode { message, cancel, .. } => {
                    println!("{message}");
                    cancel.cancelled().await;
                    Err(ModelsError::new(delta_ai::ModelsErrorCode::Aborted, "dismissed"))
                }
                AuthPrompt::Text { message, .. } | AuthPrompt::Secret { message, .. } => {
                    Ok(read_line(&message))
                }
            }
        })
    }

    fn notify(&self, event: AuthEvent) {
        match event {
            AuthEvent::Info { message, links } => {
                println!("{message}");
                for link in links {
                    println!("{}: {}", link.label.as_deref().unwrap_or("More information"), link.url);
                }
            }
            AuthEvent::AuthUrl { url, .. } => println!("Open: {url}"),
            AuthEvent::DeviceCode { user_code, verification_uri, .. } => {
                println!("Code: {user_code} at {verification_uri}");
            }
            AuthEvent::Progress { message } => println!("{message}"),
        }
    }
}

# fn read_line(_: &str) -> String { String::new() }
let models = Models::default(); // pass a persistent CredentialStore in real code
models.set_provider(anthropic_provider());

// Login: Models drives the flow and persists the credential
models
    .login("anthropic", AuthType::OAuth, &Terminal, &LoginOptions::default())
    .await?;

// From here on, requests resolve and refresh the token automatically
let model = models.get_model("anthropic", "claude-sonnet-4-5").expect("model");
let context = Context::new(vec![Message::user("Hello")]);
models.complete(&model, &context, StreamOptions::default()).await;

// Logout
models.logout("anthropic").await?;
# Ok(())
# }
```

API-key login works today through the same `Models::login` call with `AuthType::ApiKey`. Standard providers prompt for the key; Bedrock and Vertex offer a choice of methods:

```rust
# #[tokio::main(flavor = "current_thread")]
# async fn main() -> Result<(), delta_ai::ModelsError> {
use delta_ai::auth::{AuthEvent, AuthInteraction, AuthPrompt, AuthType, LoginOptions};
use delta_ai::providers::builtin::builtin_provider;
use delta_ai::{Credential, Models, ModelsError};
use futures::future::BoxFuture;

struct Terminal;

impl AuthInteraction for Terminal {
    fn prompt(&self, prompt: AuthPrompt) -> BoxFuture<'_, Result<String, ModelsError>> {
        // Show the prompt and read the answer; a fixed answer keeps the example short.
        if let AuthPrompt::Secret { message, .. } = &prompt {
            println!("{message}");
        }
        Box::pin(async { Ok("sk-or-...".to_owned()) })
    }

    fn notify(&self, event: AuthEvent) {
        match event {
            AuthEvent::Info { message, links } => {
                println!("{message}");
                for link in links {
                    println!("{}: {}", link.label.as_deref().unwrap_or("More information"), link.url);
                }
            }
            AuthEvent::AuthUrl { url, .. } => println!("Open: {url}"),
            AuthEvent::DeviceCode { user_code, verification_uri, .. } => {
                println!("Code: {user_code} at {verification_uri}");
            }
            AuthEvent::Progress { message } => println!("{message}"),
        }
    }
}

let models = Models::default(); // pass a persistent CredentialStore in real code
models.set_provider_arc(builtin_provider("openrouter").expect("built-in provider"));

// Models drives the flow and stores the credential
let credential = models
    .login("openrouter", AuthType::ApiKey, &Terminal, &LoginOptions::default())
    .await?;
assert!(matches!(credential, Credential::ApiKey(_)));

// Logout
models.logout("openrouter").await?;
# Ok(())
# }
```

### Vertex AI

Vertex AI (feature `vertex`) accepts either a Google Cloud API key or Application Default Credentials (ADC). Its API-key login flow can configure either method:

- **API key**: set `GOOGLE_CLOUD_API_KEY` or pass `api_key` in the request options.
- **Local development (ADC)**: run `gcloud auth application-default login`.
- **CI/Production (ADC)**: set `GOOGLE_APPLICATION_CREDENTIALS` to a service account JSON key file.

With ADC, also set `GOOGLE_CLOUD_PROJECT` (or `GCLOUD_PROJECT`) and `GOOGLE_CLOUD_LOCATION`, or pass `project` / `location` in `GoogleVertexOptions`. With an API key, project and location are not required.

```bash
# Local (uses your user credentials)
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="my-project"
export GOOGLE_CLOUD_LOCATION="us-central1"

# CI/Production (service account key file)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

Official docs: [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials)

### CLI Login

The quickest way to authenticate is the `delta-ai` binary, built with the `cli` feature:

```bash
cargo install delta-ai --features cli
delta-ai login              # interactive provider selection
delta-ai login anthropic    # login to specific provider
delta-ai list               # list available providers
```

Credentials are saved to `auth.json` in the current directory, in the same JSON shape as pi's `auth.json` (`{"anthropic": {"type": "oauth", "refresh": ..., "access": ..., "expires": ...}}`), so a file-backed `CredentialStore` can read them. Only providers with an OAuth flow are listed.

### Programmatic OAuth

Use the provider-owned `OAuthAuth` through `Models::login`, as above; it composes with `CredentialStore` and gets locked automatic refresh through `Models`. The flows are also public in `delta_ai::oauth` (`AnthropicOAuth`, `OpenRouterOAuth`, `XaiOAuth`), with their endpoints as public fields. To add OAuth to a custom provider, implement `OAuthAuth` and attach it with `ProviderAuth::with_oauth`. The building blocks of the built-in flows are public too:

- `generate_pkce()`: PKCE verifier and S256 challenge;
- `start_oauth_callback_server(options)` and `wait_for_callback_or_manual_input(..)`: a loopback redirect server raced against a pasted code or redirect URL;
- `poll_oauth_device_code_flow(options, poll)`: RFC 8628 device-code polling with `slow_down` and expiry handling.

The callback host defaults to `127.0.0.1`; set `PI_OAUTH_CALLBACK_HOST` to change it (for example inside containers).

Provider notes:

**Azure OpenAI (Responses)**: Uses the Responses API, plus Chat Completions for the catalog's one `openai-completions` model. Set `AZURE_OPENAI_API_KEY` and either `AZURE_OPENAI_BASE_URL` or `AZURE_OPENAI_RESOURCE_NAME`. `AZURE_OPENAI_BASE_URL` accepts `https://<resource>.openai.azure.com` and `https://<resource>.cognitiveservices.azure.com`; root endpoints are normalized to `.../openai/v1`. `AZURE_OPENAI_API_VERSION` (default `v1`) overrides the API version. Deployment names default to the model ids; override them with `AzureOpenAIResponsesOptions::azure_deployment_name` or `AZURE_OPENAI_DEPLOYMENT_NAME_MAP`, which holds comma-separated `model-id=deployment` pairs (for example `gpt-4o-mini=my-deployment,gpt-4o=prod`). Legacy deployment-based URLs are not supported.

OpenAI Codex and GitHub Copilot are not ported.

## Migrating from the Old Global API

Not applicable. This crate never had upstream's old global API (`getModel()`, global `stream()`, `registerApiProvider()`, ...) and has no compat entrypoint. Use `Models` and the provider factories. The `delta_ai::compat` module is unrelated: it holds the per-API model compatibility settings.

## Development

Run the checks from the repository root:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

Tests must not hit the network. Provider tests replay recorded SSE bodies or use `wiremock`. The examples in this README are compiled as doctests.

### Adding a New Provider

Adding a provider touches several files. API implementations live in `src/api/`, the provider table and factories in `src/providers/builtin.rs`, and the catalog data in `data/providers/`. Upstream design wins: write down any deviation in the design notes and confirm it with the maintainer before you implement it. Repository rules are in [AGENTS.md](../../AGENTS.md).

#### 1. Core Types (`src/options.rs`, `src/compat.rs`)

Rust has no `KnownApi` / `KnownProvider` string unions; API and provider ids are strings. For a new API:

- Add an options struct and an `ApiOptions` variant in `src/options.rs`
- If the API has its own compat schema, add a struct and a `ModelCompat` variant in `src/compat.rs`, and map the API id in `ModelCompat::from_value`
- Keep upstream's JSON shape for everything that is serialized

#### 2. API Implementation (`src/api/<api>.rs`, only for a new API)

Create `src/api/<api>.rs` (for example `bedrock_converse_stream.rs`) and register it in `src/api/mod.rs`. It exports:

- A unit struct that implements `ProviderStreams`, plus free functions `stream` and `stream_simple`
- Message conversion from the `TranscriptContext` to the provider format; read the prompt and tools with `transcript::get_initial_system_message()`, `get_current_tools()` and `resolve_transcript()`, and prepare history with `transform::transform_messages()`
- Tool conversion if the provider supports tools (shared strict-schema and grammar logic is in `api::constrained_sampling`)
- Response parsing that emits events through `MessageBuilder`, run inside `api::common::spawn_stream`

Use the vendor's official Rust SDK when one exists and put it behind a Cargo feature (as `bedrock` and `vertex` do). Otherwise write the client on `reqwest` and `delta_ai::http` (`send_with_retry`, `sse_events`). Do not use third-party SDK crates. Zed's provider crates are GPL-3.0: read them for behavior only, never copy code; `docs/research/provider-references.md` lists the behaviors to cover.

#### 3. Model Generation (`data/providers/`, `scripts/sync-models.sh`)

The catalog is not generated here. `scripts/sync-models.sh [version]` copies `dist/providers/data/*.json` from the `@earendil-works/pi-ai` npm package into `crates/ai/data/providers/`. For a provider that upstream already ships, run the script. Then add the file to the `CATALOG` table in `src/providers/catalog.rs`. The test `every_catalog_entry_parses` checks that every entry parses.

#### 4. Provider Factory (`src/providers/builtin.rs`)

- A provider that only pairs an environment variable with an API goes into the `SIMPLE_PROVIDERS` table
- A provider with its own behavior gets a factory function built on `create_provider()` with the catalog, auth and API implementation
- Auth: `env_api_key_auth` for standard key providers, a custom `ApiKeyAuth` for ambient auth (AWS profiles, ADC), an `OAuthAuth` where an OAuth flow exists
- Add the id to the `match` in `builtin_provider()`, behind `#[cfg(feature = "...")]` if it needs a feature

#### 5. Tests (`tests/`)

Add `tests/<api>.rs` for a new API, or extend the existing one. Replay recorded SSE bodies from `tests/fixtures/` or serve them with `wiremock`. Cover the behaviors of upstream's test suite:

- Basic streaming and tool use
- Token usage and cost reporting
- Request cancellation
- Empty messages
- Context overflow errors
- Images in user messages and tool results (if supported)
- Unicode handling
- Tool calls without results
- Cross-provider context replay
- Auth resolution for the provider

#### 6. Coding Agent Integration

Not part of this crate. When the agent crates in this repository need a default model or help text for the new provider, update them there.

#### 7. Documentation

- Write or update the design note `docs/design/api-<api>.md` (and the index in [docs/design/README.md](docs/design/README.md)). List every difference from upstream and every pending item.
- Update this README: the [Supported Providers](#supported-providers) table, provider-specific options or auth requirements, and the [Environment Variables](#environment-variables) table.

#### 8. Changelog

This crate has no changelog yet. Describe the change in the commit message.

## License

MIT. See [LICENSE](LICENSE).
