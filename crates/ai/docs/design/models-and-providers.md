# Models 集合与 Provider（`models`、`provider`）

上游：`src/models.ts`、`src/models-store.ts`、`src/api/lazy.ts`、`src/utils/abort.ts`。

## Provider

```rust
pub trait Provider: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn base_url(&self) -> Option<&str>;
    fn headers(&self) -> Option<&ProviderHeaders>;
    fn auth(&self) -> &ProviderAuth;
    fn models(&self) -> Vec<Model>;              // 上游 getModels()
    fn all_models(&self) -> Vec<AnyModel>;       // 上游 getAllModels()
    fn is_dynamic(&self) -> bool;                // 上游 refreshModels 是否存在
    fn refresh_models(&self, ctx: RefreshModelsContext) -> BoxFuture<'_, Result<(), ModelsError>>;
    fn filter_models(&self, models: Vec<Model>, credential: Option<&Credential>) -> Vec<Model>;
    fn filter_all_models(&self, models: &[AnyModel], credential: Option<&Credential>) -> Option<Vec<AnyModel>>;
    fn stream(&self, model: &Model, ctx: &TranscriptContext, options: StreamOptions) -> AssistantMessageEventStream;
    fn stream_simple(&self, model: &Model, ctx: &TranscriptContext, options: SimpleStreamOptions) -> AssistantMessageEventStream;
}
```

TS 接口里的可选方法在 Rust 里是带默认实现的 trait 方法；`is_dynamic` 替代“方法是否存在”的判断。上游 `Provider<TApi>` 的泛型只用于给工厂函数的直接调用者提供类型，Rust 里去掉。`generateImages`、`classify`、`fetchDeferred`、`cancelDeferred` 第一阶段不加入 trait，以后以带默认实现的方法加入，不破坏已有实现。

## `create_provider`

`create_provider(CreateProviderOptions) -> BuiltProvider`，对应上游 `createProvider()`：

- `models`：各类型的静态基线模型。
- `api: ProviderApi`：`Single(Arc<dyn ProviderStreams>)` 服务全部 chat 模型；`ByApi(HashMap<api, ..>)` 按 `model.api` 分派，没有对应实现的模型得到一个错误流（`ModelsErrorCode::Stream`）。
- `fetch_models`：动态模型列表。刷新时先恢复 `stored`（只保留属于本 provider 的模型），允许联网时再拉取，结果通过 `publish` 写入存储并替换内存中的动态列表；动态模型按（类型，id）覆盖基线模型。
- `filter_models` / `filter_all_models`：按凭据过滤可用模型（例如 OpenAI 用 OAuth 时隐藏分类器模型）。

`ProviderStreams` 是 API 实现的契约（`stream`、`stream_simple`）；provider 范围的端点改写通过包装 `ProviderStreams` 实现，与上游一致。

## Models

`Models` 是可廉价克隆的句柄（内部 `Arc`），克隆共享状态。对应上游 `Models` + `MutableModels`：

| 上游 | Rust |
| --- | --- |
| `createModels(options)` | `Models::new(CreateModelsOptions)`、`Models::default()` |
| `setProvider` / `deleteProvider` / `clearProviders` | `set_provider`、`set_provider_arc`、`delete_provider`、`clear_providers` |
| `getProviders` / `getProvider` | `providers`、`provider` |
| `getModels` / `getModel` / `getAllModels` / `getModelsOfType` / `getModelOfType` | 同名 snake_case 方法 |
| `checkAuth` / `getAvailable` / `getAvailableOfType` / `getAllAvailable` | 同名方法 |
| `getAuth(providerId)` / `getAuth(model)` | `get_auth(provider_id, overrides)` / `get_model_auth(model, overrides)`（Rust 无重载） |
| `login` / `logout` | 同名方法 |
| `refresh` | `refresh(ModelsRefreshOptions) -> ModelsRefreshResult` |
| `stream` / `complete` / `streamSimple` / `completeSimple` | 同名方法 |

### 请求路径

`stream` / `stream_simple`：

1. `normalize_context` 把 `Context` 变成 `TranscriptContext`（同步）。
2. `lazy_stream` 立即返回事件流，后台任务执行：查找 provider（未知 provider → 错误流）；`apply_auth` 解析鉴权；调用 `provider.stream(...)` 并转发事件。
3. `apply_auth`：显式 `api_key` 优先，否则用 provider 解析出的 key；请求头按“provider 鉴权头 → `model.headers` → `options.headers` → `transform_headers`”的顺序合并，名字不区分大小写；`env` 合并（请求的覆盖解析出的）；鉴权给出 `base_url` 时替换模型的 `base_url`。provider 未配置时返回 `Provider is not configured: <id>` 错误。

准备阶段失败只发一个 `Error` 事件；取消令牌已触发时 `reason` 为 `aborted`。

### 动态刷新

`refresh` 并发刷新选中的动态 provider，错误按 provider 收集返回，不抛出。每个 provider：

1. 读取存储的凭据；先以离线模式运行一次 `refresh_models`，恢复缓存的目录（即使凭据读取失败）。
2. 允许联网时解析刷新用的凭据：OAuth 已过期则在凭据锁内刷新；API key 走 `resolve`。没有凭据则跳过。
3. 以联网模式再运行一次 `refresh_models`。

代数（generation）检查与上游相同：`set_provider`、`delete_provider` 或新的刷新会使旧刷新失效并取消它；`publish` 在每个 provider 的发布锁内串行执行，写存储前后都检查代数和取消状态，过期的发布返回 `false` 且不执行 `update`。调用方取消时 `refresh` 立即返回 `aborted: true`。

### 模型存储

`ModelsStore` trait（`read`、`write`、`delete`），默认 `InMemoryModelsStore`。`ModelsStoreEntry { models, last_modified, checked_at, etag }`，反序列化时丢弃类型未知的模型。
