# Image generation, classification and deferred responses (`images`, `classifier`, `deferred`)

## Purpose

The framework for the operations besides chat streaming: image generation (`Models::generate_images`), structured classification (`Models::classify`) and deferred chat responses (`SimpleStreamOptions::deferred`, `Models::stream_deferred` / `fetch_deferred` / `cancel_deferred`). This layer holds the request and result types, the contracts that API implementations fulfil, provider dispatch by API id, auth resolution and the error results. The concrete image and classifier APIs are separate API modules built on these interfaces: `openrouter-images` ([api-openrouter-images.md](api-openrouter-images.md)), `typesafe-system-one` and `cloudflare-workers-ai-system-one` ([api-system-one.md](api-system-one.md)), `openai-decisions` ([api-openai-decisions.md](api-openai-decisions.md)) and `llama-cpp-classify` ([api-llama-cpp-classify.md](api-llama-cpp-classify.md)).

## Upstream source files

- `src/types.ts`: `ImagesContext`, `AssistantImages`, `ImagesOptions`, `ClassifierQuestion`, `ClassifierContext`, `ClassifierAnswer`, `ClassifierResult`, `ClassifierOptions`, `ProviderImages`, `ProviderClassifier`, `DeferredHandle`, `DeferredFetchOptions`, `DeferredCancelOptions`, `SimpleStreamOptions.deferred`, `ProviderStreams.fetchDeferred` / `cancelDeferred`.
- `src/models.ts`: the optional `Provider` methods, `Models.generateImages` / `classify` / `streamDeferred` / `fetchDeferred` / `cancelDeferred`, `createProvider({ images, classifiers })`.
- `src/utils/model-operations.ts`: `assertClassifierInputSupported`, `imageErrorResult`, `classifierErrorResult`.
- `src/api/lazy.ts`: `LazyApiCapabilities` (`fetchDeferred`, `cancelDeferred`).
- `src/providers/faux.ts`: deferred support of the faux provider.
- Tests: `test/images-models.test.ts`, `test/classifier-models.test.ts`, the deferred cases of `test/providers.test.ts`.

## Types

| Upstream | Rust |
| --- | --- |
| `ImagesInputContent` / `ImagesOutputContent` (`TextContent \| ImageContent`) | aliases of `UserBlock` |
| `ImagesContext { input }` | `images::ImagesContext` |
| `ImagesStopReason` (`stop`, `error`, `aborted`) | `ImagesStopReason` enum |
| `AssistantImages` | struct of the same name; `AssistantImages::error(model, message, aborted)` is `imageErrorResult` |
| `ImagesOptions` | `ImagesOptions { base: RequestOptions<ImageModel>, metadata }` |
| `ClassifierChoiceQuestion` / `ClassifierScoreQuestion` / `ClassifierBoolQuestion` | structs of the same names in `classifier`; `ClassifierQuestion` is an enum tagged by `type` |
| bool criteria `{ true, false }` | `ClassifierBoolCriteria { when_true, when_false }`, serialized as `true` / `false` |
| `ClassifierContext { state, images?, questions }` | `ClassifierContext`; `questions` is an `IndexMap`, so request order is kept |
| `ClassifierAnswer` (choice, score, bool) | enum tagged by `type`; `probabilities` and `answers` are `IndexMap`s |
| `ClassifierStopReason` / `ClassifierResult` | enum and struct of the same names; `ClassifierResult::error` is `classifierErrorResult` |
| `ClassifierOptions { temperature }` | `ClassifierOptions { base: RequestOptions<ClassifierModel>, temperature }` |
| `DeferredHandle` | `types::DeferredHandle` (already part of `AssistantMessage::deferred`) |
| `SimpleStreamOptions.deferred: boolean \| { window }` | `SimpleStreamOptions::deferred: Option<DeferredRequest>`; `true` is `Some(DeferredRequest::default())`, `{ window: "1h" }` is `DeferredRequest::with_window(DeferredWindow::OneHour)` |
| `DeferredFetchOptions { wait }` | `DeferredFetchOptions { base: RequestOptions<Model>, wait: Option<Duration> }` |
| `DeferredCancelOptions` | alias of `RequestOptions<Model>` |
| `ProviderRequestOptions<TModel>` | `options::RequestOptions<M>`; its callbacks are `OnPayloadFor<M>` / `OnResponseFor<M>` |

JSON shapes match upstream (camelCase fields, `type` tags, optional fields omitted). The tests in `images.rs` and `classifier.rs` round-trip upstream JSON.

## Implementation contracts

An API module implements one trait on a unit struct, as chat APIs implement `ProviderStreams`:

```rust
pub trait ProviderImages: Send + Sync {
    fn generate_images<'a>(&'a self, model: &'a ImageModel, context: &'a ImagesContext,
        options: ImagesOptions) -> BoxFuture<'a, AssistantImages>;
}

pub trait ProviderClassifier: Send + Sync {
    fn classify<'a>(&'a self, model: &'a ClassifierModel, context: &'a ClassifierContext,
        options: ClassifierOptions) -> BoxFuture<'a, ClassifierResult>;
}
```

The future never fails: request, response and cancellation errors become a result with stop reason `error` or `aborted` (`AssistantImages::error`, `ClassifierResult::error`). When `Models` calls the implementation, auth is resolved into `options.base` (`api_key`, merged `headers`, merged `env`) and `model.base_url` carries a base URL resolved by auth. `RequestOptions::transport_options()` gives the auth and transport settings as `StreamOptions`, so an implementation can use `http::client` and `http::send_with_retry`.

`ProviderStreams` gains the deferred methods of upstream's optional members. The capability is a separate `supports_*` method, like `LazyApiCapabilities`:

| Method | Default |
| --- | --- |
| `supports_fetch_deferred()` | `false` |
| `fetch_deferred(model, handle, DeferredFetchOptions) -> AssistantMessageEventStream` | error stream `API does not support deferred responses` |
| `supports_cancel_deferred()` | `false` |
| `cancel_deferred(model, handle, DeferredCancelOptions) -> BoxFuture<Result<(), Error>>` | `Err("API cannot cancel deferred responses")` |

A response that is still running ends with stop reason `deferred` (event `done` with reason `deferred`) and `AssistantMessage::deferred` set to the handle.

## Provider

The `Provider` trait gets the optional upstream methods as default methods, each paired with a capability check that `Models` reads before it resolves auth (upstream checks whether the method exists):

| Upstream | Rust | Default |
| --- | --- | --- |
| `fetchDeferred?` | `supports_fetch_deferred()` + `fetch_deferred(..)` | `false`; error stream |
| `cancelDeferred?` | `supports_cancel_deferred()` + `cancel_deferred(..)` | `false`; `Err` |
| `generateImages?` | `supports_images()` + `generate_images(..)` | `false`; error result `Provider <id> does not support image generation` |
| `classify?` | `supports_classify()` + `classify(..)` | `false`; error result `Provider <id> does not support classification` |

Existing `Provider` implementations keep compiling.

## `create_provider`

`CreateProviderOptions` has two new maps keyed by `model.api`: `images: HashMap<String, Arc<dyn ProviderImages>>` and `classifiers: HashMap<String, Arc<dyn ProviderClassifier>>`, filled with the builder methods `.images(api, implementation)` and `.classifier(api, implementation)`.

Upstream makes `api` optional when `images` or `classifiers` is given. Rust keeps `api: ProviderApi` and the four-argument `CreateProviderOptions::new`, so existing callers are unchanged; an image- or classifier-only provider passes `ProviderApi::none()` (an empty `ByApi` map). As upstream, a provider without any implementation is rejected with `Provider <id>: at least one of "api", "images", or "classifiers" is required.`: `try_create_provider` returns that error, `create_provider` panics with it (upstream throws).

Dispatch of the built provider:

- `supports_images` / `supports_classify`: the map is not empty. A model whose API has no entry gets an error result `Provider <id> has no image generation implementation for "<api>"` / `Provider <id> has no classifier implementation for "<api>"`.
- `supports_fetch_deferred` / `supports_cancel_deferred`: at least one chat implementation reports the capability. A model whose API implementation lacks it gets `Provider <id> does not support deferred responses for "<api>"` (an error stream) / `Provider <id> cannot cancel deferred responses for "<api>"`.

The built-in wrappers (`AzureStreams`, `OpenCodeStreams`) do not forward the deferred methods; upstream's provider wrappers do not either, and no built-in API implements deferred responses.

## Models

| Upstream | Rust |
| --- | --- |
| `generateImages(model, context, options)` | `generate_images(&ImageModel, &ImagesContext, ImagesOptions) -> AssistantImages` |
| `classify(model, context, options)` | `classify(&ClassifierModel, &ClassifierContext, ClassifierOptions) -> ClassifierResult` |
| `streamDeferred(model, handle, options)` | `stream_deferred(&Model, &DeferredHandle, DeferredFetchOptions) -> AssistantMessageEventStream` |
| `fetchDeferred(model, handle, options)` | `fetch_deferred(..) -> AssistantMessage` (the final message of `stream_deferred`) |
| `cancelDeferred(model, handle, options)` | `cancel_deferred(&Model, &DeferredHandle, DeferredCancelOptions) -> Result<(), Error>` |

Order of checks, as upstream:

- `generate_images`: unknown provider (`Unknown provider: <id>`) → provider without image support (`Provider <id> does not support image generation`) → auth (`Provider is not configured: <id>`) → `provider.generate_images`.
- `classify`: images sent to a model whose `input` lacks `image` (`Model <provider>/<id> does not accept image input`, before any provider call; an empty image list is accepted) → unknown provider → `Provider <id> does not support classification` → auth → `provider.classify`.
- `stream_deferred` / `cancel_deferred`: unknown provider → `Provider <id> does not support deferred responses` → auth → provider.

`generate_images` and `classify` never fail: every error before the provider call becomes a result with stop reason `error`, or `aborted` when the cancellation token has fired. `stream_deferred` reports errors in the stream like `stream`; `cancel_deferred` returns them.

Auth goes through the same code as `stream` (`Models::resolve_request`): an explicit `api_key` wins, headers merge in the order provider auth → `model.headers` → `options.headers` → `transform_headers`, resolved `env` is overlaid by the request's `env`, and a resolved base URL replaces `model.base_url`. Auth resolution races the cancellation token (upstream passes the abort signal to `getAuth`), so a cancelled request never reaches the provider; this also applies to `stream` and `stream_simple`.

Credential-specific availability needs no new code in this layer: `get_all_available` / `get_available_of_type` apply `Provider::filter_all_models`. Upstream's only rule of this kind (the `openai` provider hides its `openai-decisions` classifier models while it uses ChatGPT OAuth) lives in the provider factory, `providers::builtin` (see [api-openai-decisions.md](api-openai-decisions.md#availability-with-chatgpt-oauth)).

## Faux provider

As upstream, the faux provider supports deferred responses but not images or classification:

- `FauxOptions::deferred: Option<FauxDeferredOptions { pending_fetches, poll_after_ms }>`.
- `stream_simple` with `deferred` set takes the next queued reply, stores it with the request context and answers with an empty message of stop reason `deferred` (events `start`, `done`). The handle has `poll_after_ms` when configured and a generated id. An empty queue still answers `No more faux responses queued`.
- `fetch_deferred` counts `FauxState::deferred_fetch_count`. It answers the handle again while `pending_fetches` remain, then the stored reply (resolved once, with the usage estimate and prompt cache of the submission). Unknown handles fail with `Unknown faux deferred response: <id>`, cancelled ones with `Faux deferred response was cancelled: <id>`.
- `cancel_deferred` records the handle in `FauxState::cancelled_deferred` and marks it cancelled.
- `on_response` is called with status 200 for submissions, fetches and cancellations, as upstream.

## Built-in implementations

| Provider | Images | Classifiers |
| --- | --- | --- |
| `openrouter` | `openrouter-images` | `typesafe-system-one` |
| `typesafe` | | `typesafe-system-one` |
| `opencode`, `vercel-ai-gateway` | | `typesafe-system-one` |
| `cloudflare-workers-ai` | | `cloudflare-workers-ai-system-one` |
| `openai` | | `openai-decisions` (hidden for OAuth credentials) |

`llama-cpp-classify` has no built-in provider; applications register it on their own (upstream registers it in the coding agent's llama extension).

## Rust-specific differences

- Typed entry points: `generate_images` takes `&ImageModel`, `classify` takes `&ClassifierModel` and the stream methods take `&Model`, so upstream's runtime checks `is not an image model` / `is not a classifier model` / `is not a chat model` cannot fail and are not ported.
- Optional interface members become default trait methods plus `supports_*` capability checks (see above). `lazyApi` has no Rust counterpart (API modules are compiled in); its capabilities are the `supports_*` methods of `ProviderStreams`.
- `api` stays required in `CreateProviderOptions`; `ProviderApi::none()` stands for an omitted `api`. `create_provider` panics where upstream throws; `try_create_provider` returns the error.
- Option types wrap `RequestOptions<M>` in a `base` field, like `SimpleStreamOptions::base`. `wait` is a `Duration`.
- `cancel_deferred` returns `crate::error::Error` (provider and auth failures are `Error::Models`), since implementations fail with HTTP errors.
- The faux response factory is synchronous and cannot reject; a factory that fails returns an error `AssistantMessage`, which `fetch_deferred` replays the same way upstream replays a rejection.
- The global `generateImages()` with its images API registry (`src/images.ts`, `src/images-api-registry.ts`) is not ported, like the global chat `stream()`; use `Models::generate_images` or call an API implementation directly.

## Not ported

- `ProviderImagesOptions` (`ImagesOptions & Record<string, unknown>`): no API-specific image options exist upstream yet. If an image API needs them, `ImagesOptions` can get an `api: Option<JsonObject>` field like `StreamOptions::api`.
- Telemetry spans for deferred polls (`pi.ai.poll`), with the rest of telemetry.
- `test/image-model-data.test.ts` tests upstream's catalog build script, which delta does not port (see [catalog-and-faux.md](catalog-and-faux.md)). The cases of `test/images.test.ts`, `test/images-models.test.ts` and `test/classifier-models.test.ts` that call a concrete API are ported with those APIs (`tests/openrouter_images.rs`, `tests/images_models.rs`, `tests/classifier_models.rs`).

## Improvements pending confirmation

- The faux provider fails a scripted error reply with empty content without a `start` event, while upstream emits `start` then `error` for it (upstream emits `error` alone only for request failures). This predates this module and applies to deferred fetches that replay a failed reply too.
