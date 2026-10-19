# 鉴权（`auth`）

上游：`src/auth/types.ts`、`src/auth/resolve.ts`、`src/auth/helpers.ts`、`src/auth/context.ts`、`src/auth/credential-store.ts`。

## 类型

| 上游 | Rust |
| --- | --- |
| `ModelAuth { apiKey, headers, baseUrl }` | `ModelAuth` |
| `ApiKeyCredential` / `OAuthCredential` / `Credential` | 同名；`Credential` 用 `type` 标签（`api_key` / `oauth`），与 pi 的 `auth.json` 格式一致；OAuth 的额外字段放在 `extra` |
| `CredentialStore` | trait：`read`、`list`、`modify`、`delete` |
| `InMemoryCredentialStore` | 同名，按 provider 串行化写入 |
| `AuthContext` | trait：`env(name)`、`file_exists(path)`；默认实现 `DefaultAuthContext` |
| `ApiKeyAuth` | trait：`name`、`resolve`、可选 `check`、可选 `login` |
| `OAuthAuth` | trait：`name`、`login`、`refresh`、`to_auth` |
| `ProviderAuth { apiKey?, oauth? }` | `ProviderAuth { api_key, oauth }`（`Arc<dyn ..>`） |
| `envApiKeyAuth(name, envVars)` | `env_api_key_auth(name, &[..])` |
| 本地无 key 服务器的 `resolve: () => ({ auth: {} })` | `ProviderAuth::keyless(name)` |
| `AuthInteraction` / `AuthPrompt` / `AuthEvent` | 同名 trait / 枚举 |

`CredentialStore::modify` 是唯一的写入路径：传入的函数拿到当前凭据，返回新凭据或 `None`（不变）。存储实现负责对同一 provider 串行化（支持时跨进程加锁，例如文件锁）。

## 解析顺序（`resolve_provider_auth`）

与上游一致：

1. 显式 `api_key`（请求选项）且 provider 有 API key 鉴权：用它构造临时凭据解析。
2. 有存储的凭据：
   - OAuth 且 provider 支持 OAuth：剩余有效期不足 5 分钟（或调用方要求的更长时间）时在锁内刷新，然后 `to_auth`。
   - API key 且 provider 支持：用存储的 key 和 env 解析（请求的 env 覆盖存储的）。
   - 类型不匹配：视为未配置，不回退到环境变量。
3. 没有存储的凭据：用环境变量等环境来源解析。

存储的凭据拥有该 provider：只有没有存储凭据时才看环境变量；OAuth 刷新失败不会悄悄改用环境变量里的 key。

请求的 `env` 覆盖层：`OverlayAuthContext` 先查请求的 env，再查进程环境。空白值视为未设置。

## OAuth 刷新加锁

`refresh_stored_oauth_credential`：在 `modify` 内重新检查是否需要刷新（双重检查），需要时调用 `oauth.refresh`，结果在释放锁前写入存储。并发请求只刷新一次（有测试覆盖）。

上游里调用方的 signal 只取消“等锁”，刷新一旦开始就忽略 signal，只受 15 秒超时约束，以免丢失已轮换的 refresh token。Rust 里取消就是丢弃 future，所以整个刷新放在独立的 Tokio 任务里，调用方丢弃 future 不影响刷新完成和写回；刷新本身仍有 15 秒超时。

错误码：刷新失败 → `OAuth`（凭据保留，重新登录可修复）；存储读写失败或 API key 解析失败 → `Auth`。

## 第一阶段范围

- 已实现：类型、内存存储、解析顺序、加锁刷新、`env_api_key_auth`、`keyless`、`Models::login/logout` 框架。
- 未实现：各 provider 的具体 OAuth 登录流程（Anthropic、OpenAI ChatGPT、Codex、GitHub Copilot、OpenRouter 等）、回调服务器、设备码流程、PKCE。它们放在后续阶段，按 provider 逐个移植。
