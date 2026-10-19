# delta-ai（`delta_ai`）设计文档

`delta-ai` 是 [`@earendil-works/pi-ai`](https://github.com/earendil-works/pi/tree/main/packages/ai) 的 Rust 移植，crates.io 包名 `delta-ai`，代码里 `use delta_ai::...`。设计以上游为准；Rust 语言层面必须改动的地方在各文档的“与上游的差异”里列出，属于改进或行为变化的放在“待确认”里，确认前不实现。

## 文档索引

| 文档 | 模块 | 内容 |
| --- | --- | --- |
| [types.md](types.md) | `types`、`model`、`compat` | 消息、内容块、模型、兼容性配置，JSON 形状与上游一致 |
| [events.md](events.md) | `events` | 事件协议、`AssistantMessageEventStream`、`MessageBuilder` |
| [models-and-providers.md](models-and-providers.md) | `models`、`provider` | `Models` 集合、`Provider` trait、`create_provider`、动态模型刷新 |
| [auth.md](auth.md) | `auth` | 凭据、凭据存储、API key 与 OAuth、解析顺序与加锁刷新 |
| [transcript.md](transcript.md) | `transcript`、`transform`、`estimate`、`simple_options` | system 消息重放、跨 provider 切换、token 估算、思考预算 |
| [http.md](http.md) | `http`、`json`、`api::common`、`api::constrained_sampling` | SSE 解码、重试、错误体、流式 JSON 解析、API 任务框架、共享的约束采样 |
| [catalog-and-faux.md](catalog-and-faux.md) | `providers::catalog`、`providers::faux` | 内置模型目录、测试用 faux provider |
| `api-*.md` | `api::*` | 各 wire 协议的实现，每个 API 一篇 |

## 总体结构

![delta_ai 运行时结构](diagrams/architecture.png)

（可交互版本：[diagrams/architecture.html](diagrams/architecture.html)）

和上游一样分三层：

1. **`Models`**：应用持有的集合。负责按 provider 解析鉴权、合并请求头、规范化上下文，再把请求交给拥有该模型的 provider。
2. **`Provider`**：运行时单元，拥有模型目录、鉴权方式（API key / OAuth）和请求行为。内置 provider 通过 `create_provider` 组装。
3. **API 实现**（`ProviderStreams`）：wire 协议，例如 `anthropic-messages`、`openai-completions`。多个 provider 共用同一个 API 实现（xAI、Groq、OpenRouter 等都走 `openai-completions`）。

一次 `Models::stream` 请求的时序：

![Models::stream 时序](diagrams/stream-sequence.png)

（可交互版本：[diagrams/stream-sequence.html](diagrams/stream-sequence.html)）

`Models::stream` 同步返回事件流；鉴权解析和 provider 分派在后台任务里进行（对应上游的 `lazyStream`）。API 实现在自己的 Tokio 任务里发请求、解析 SSE、通过 `MessageBuilder` 发事件。消费端每取出一个事件就把它归约进自己的 `partial()`。

## crate 组织

```
crates/ai/
  data/providers/*.json   # 上游生成的模型目录（scripts/sync-models.sh 同步）
  docs/design/            # 本目录
  src/
    types.rs model.rs compat.rs options.rs error.rs
    events.rs             # 事件协议与事件流
    auth.rs               # 凭据与鉴权
    provider.rs models.rs # Provider trait 与 Models 集合
    transcript.rs transform.rs estimate.rs simple_options.rs json.rs
    http.rs               # 手写客户端共用的 HTTP/SSE 工具
    api/                  # 每个 wire 协议一个模块
    providers/            # 模型目录、faux、各 provider 工厂
```

上游用子路径导入和懒加载控制打包体积；Rust 用 cargo feature 控制编译体积。手写的 API（Anthropic、OpenAI、Gemini、Mistral）没有额外依赖，始终编译；依赖官方 SDK 的两个 API 放在 feature 后面：

| feature | API | 依赖 |
| --- | --- | --- |
| `bedrock` | `bedrock-converse-stream` | `aws-sdk-bedrockruntime`、`aws-config`（AWS 官方） |
| `vertex` | `google-vertex` | `google-cloud-aiplatform-v1`、`google-cloud-auth`（Google 官方） |
| `full` | 以上全部 | |

## provider 实现策略

有官方 Rust SDK 的用官方 SDK，没有的手写（`reqwest` + `delta_ai::http`），不依赖第三方 SDK crate。2026-10 的调研结果：

| 厂商 | 官方 Rust SDK | 做法 |
| --- | --- | --- |
| Anthropic | 无（官方 SDK：Python、TS、C#、Go、Java、PHP、Ruby） | 手写 |
| OpenAI | 无（官方 SDK：TS、Python、.NET、Java、Go、Ruby） | 手写 |
| Google Gemini API | 无（官方 SDK：Python、JS、Go、Java、C#） | 手写 |
| Google Vertex AI | 有：`google-cloud-aiplatform-v1` | 官方 crate |
| AWS Bedrock | 有：`aws-sdk-bedrockruntime` | 官方 crate |
| Mistral | 无 | 手写 |

手写客户端参考了 Zed 和 litellm-rs 的实现来避坑，结论见 [`docs/research/provider-references.md`](../../../../docs/research/provider-references.md)。Zed 的 provider crate 是 GPL-3.0，只参考行为，不复制代码。

## 已确认的设计决定

| 问题 | 决定 |
| --- | --- |
| 事件里的 `partial` | 消费端归约：事件不带 `partial`，`AssistantMessageEventStream::partial()` 返回截至刚取出事件时的消息。见 [events.md](events.md) |
| Vertex AI | 用官方 `google-cloud-aiplatform-v1` + `google-cloud-auth` |
| 各 API 的专有选项 | `StreamOptions` 放公共字段，`api: Option<ApiOptions>` 是类型化枚举，每个内置 API 一个变体，另有 `Custom(JsonObject)` |
| 包名 | crates.io 用 `delta-*`（`pi-*` 被上游占用），本包 `delta-ai`，库名 `delta_ai` |
| 丢弃事件流 | 取消请求：`AssistantMessageEventStream` 被 drop 后，生产者检测到通道关闭（`MessageBuilder::is_closed`）就停止读取并断开 HTTP 连接。上游 JS 里请求会继续跑完。需要跑完的场景先调用 `result()` |

## 与上游的差异（Rust 语言层面）

这些是语言层面的改写，不改变可观察行为（除标注的条目外）：

- `AbortSignal` → `tokio_util::sync::CancellationToken`（`StreamOptions::cancel`）。公共异步操作在 Rust 里通过丢弃 future 取消，所以凭据存储、模型存储的方法不带 signal 参数。
- `fetch` 选项 → `http_client: Option<reqwest::Client>`；`timeoutMs` 等毫秒数 → `Duration`。
- 回调（`on_payload`、`on_response`、`on_provider_stream_event`）是 `Arc<dyn Fn(..) -> BoxFuture<Result<..>>>`，返回错误时请求失败，与上游抛异常的语义一致。
- `transformHeaders` 放在 `StreamOptions::transform_headers`，由 `Models` 消费后清空，provider 不会看到它（上游是 `Models` 专属的选项类型）。
- `AuthContext` 的 `env`/`file_exists` 是同步方法（上游返回 Promise，但实现都是同步读取）。
- OAuth 刷新在独立的 Tokio 任务里执行，调用方的 future 被丢弃时刷新仍会完成并写回，用来替代上游“刷新开始后忽略 signal、只受超时约束”的做法。
- 持久化的模型目录里，类型未知的模型在反序列化 `ModelsStoreEntry` 时丢弃（上游在读取后过滤）。
- 字符串长度相关的估算（token 估算、faux 的缓存模拟）按 UTF-16 码元计数，与上游 JS 的 `length` 一致。

## 待确认

第一阶段不移植的部分：遥测（等 `delta-telemetry` 移植后接入）、图片生成、分类器、deferred 响应、OAuth 登录流程（`Models::login` 的框架已在，具体 provider 的流程未写）、OpenAI Codex 与 WebSocket 传输、GitHub Copilot、Cloudflare、Radius（`pi-messages`）。模型目录里已有这些模型的数据。

各 API 的待确认项见对应的 `api-*.md`。
