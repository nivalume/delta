# The Models collection and Provider (`models`, `provider`)

Upstream: `src/models.ts`, `src/models-store.ts`, `src/api/lazy.ts`, `src/utils/abort.ts`.

## Provider

```rust
pub trait Provider: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn base_url(&self) -> Option<&str>;
    fn headers(&self) -> Option<&ProviderHeaders>;
    fn auth(&self) -> &ProviderAuth;
    fn models(&self) -> Vec<Model>;              // upstream getModels()
    fn all_models(&self) -> Vec<AnyModel>;       // upstream getAllModels()
    fn is_dynamic(&self) -> bool;                // whether upstream refreshModels exists
    fn refresh_models(&self, ctx: RefreshModelsContext) -> BoxFuture<'_, Result<(), ModelsError>>;
    fn filter_models(&self, models: Vec<Model>, credential: Option<&Credential>) -> Vec<Model>;
    fn filter_all_models(&self, models: &[AnyModel], credential: Option<&Credential>) -> Option<Vec<AnyModel>>;
    fn stream(&self, model: &Model, ctx: &TranscriptContext, options: StreamOptions) -> AssistantMessageEventStream;
    fn stream_simple(&self, model: &Model, ctx: &TranscriptContext, options: SimpleStreamOptions) -> AssistantMessageEventStream;
}
```

Optional methods of the TS interface are trait methods with default implementations in Rust. `is_dynamic` replaces the "does the method exist" check. The generic parameter of upstream `Provider<TApi>` only gives types to direct callers of the factory function; Rust drops it. `generateImages`, `classify`, `fetchDeferred` and `cancelDeferred` are not added to the trait in the first phase. They will be added later as methods with default implementations, which does not break existing implementations.

## `create_provider`

`create_provider(CreateProviderOptions) -> BuiltProvider`, corresponding to upstream `createProvider()`:

- `models`: the static baseline models of each type.
- `api: ProviderApi`: `Single(Arc<dyn ProviderStreams>)` serves all chat models; `ByApi(HashMap<api, ..>)` dispatches on `model.api`, and a model with no matching implementation gets an error stream (`ModelsErrorCode::Stream`).
- `fetch_models`: the dynamic model list. A refresh first restores `stored` (keeping only models that belong to this provider), then fetches when network access is allowed. The result is written to the store through `publish` and replaces the in-memory dynamic list. Dynamic models override baseline models by (type, id).
- `filter_models` / `filter_all_models`: filter the available models by credential (for example, OpenAI hides classifier models when OAuth is used).

`ProviderStreams` is the contract for API implementations (`stream`, `stream_simple`). Provider-scoped endpoint rewrites are done by wrapping `ProviderStreams`, as in upstream.

## Models

`Models` is a cheaply cloneable handle (an `Arc` inside); clones share state. It corresponds to upstream `Models` + `MutableModels`:

| Upstream | Rust |
| --- | --- |
| `createModels(options)` | `Models::new(CreateModelsOptions)`, `Models::default()` |
| `setProvider` / `deleteProvider` / `clearProviders` | `set_provider`, `set_provider_arc`, `delete_provider`, `clear_providers` |
| `getProviders` / `getProvider` | `providers`, `provider` |
| `getModels` / `getModel` / `getAllModels` / `getModelsOfType` / `getModelOfType` | snake_case methods with the same names |
| `checkAuth` / `getAvailable` / `getAvailableOfType` / `getAllAvailable` | Methods with the same names |
| `getAuth(providerId)` / `getAuth(model)` | `get_auth(provider_id, overrides)` / `get_model_auth(model, overrides)` (Rust has no overloading) |
| `login` / `logout` | Methods with the same names |
| `refresh` | `refresh(ModelsRefreshOptions) -> ModelsRefreshResult` |
| `stream` / `complete` / `streamSimple` / `completeSimple` | Methods with the same names |

### Request path

`stream` / `stream_simple`:

1. `normalize_context` turns the `Context` into a `TranscriptContext` (synchronously).
2. `lazy_stream` returns the event stream immediately. A background task looks up the provider (unknown provider → error stream), resolves auth with `apply_auth`, calls `provider.stream(...)` and forwards the events.
3. `apply_auth`: an explicit `api_key` wins; otherwise the key resolved by the provider is used. Request headers are merged in the order "provider auth headers → `model.headers` → `options.headers` → `transform_headers`", with case-insensitive names. `env` is merged (the request's values override the resolved ones). When auth returns a `base_url`, it replaces the model's `base_url`. If the provider is not configured, the error is `Provider is not configured: <id>`.

A failure during preparation emits a single `Error` event. If the cancellation token has fired, `reason` is `aborted`.

### Dynamic refresh

`refresh` refreshes the selected dynamic providers concurrently. Errors are collected per provider and returned, not thrown. For each provider:

1. Read the stored credential. Run `refresh_models` once in offline mode first to restore the cached catalog (even if reading the credential failed).
2. When network access is allowed, resolve the credential for the refresh: an expired OAuth credential is refreshed inside the credential lock; an API key goes through `resolve`. With no credential, skip.
3. Run `refresh_models` again in online mode.

The generation check is the same as upstream: `set_provider`, `delete_provider` or a new refresh invalidates the old refresh and cancels it. `publish` runs serially inside a per-provider publish lock and checks the generation and cancellation state both before and after writing the store. A stale publish returns `false` and does not run `update`. When the caller cancels, `refresh` returns `aborted: true` immediately.

### Model store

The `ModelsStore` trait (`read`, `write`, `delete`); the default is `InMemoryModelsStore`. `ModelsStoreEntry { models, last_modified, checked_at, etag }` drops models of unknown type when deserialized.
