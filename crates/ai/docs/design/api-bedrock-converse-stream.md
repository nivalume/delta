# API 设计：bedrock-converse-stream

## 目的

`bedrock-converse-stream` 通过 Amazon Bedrock 的 `ConverseStream` 接口流式调用 Bedrock 上的模型（Claude、Nova、OpenAI GPT/gpt-oss 等），
把 Bedrock 的事件流转换成 delta 统一的 `AssistantMessageEvent` 协议。

按 `AGENTS.md` 的规定，Bedrock 使用 AWS 官方 Rust SDK（`aws-sdk-bedrockruntime` + `aws-config`），不手写 SigV4 和事件流解析。
整个模块在 `bedrock` feature 之后（`#[cfg(feature = "bedrock")]`），未开启时不编译 AWS SDK。

模块：`crates/ai/src/api/bedrock_converse_stream.rs`，单元测试在 `bedrock_converse_stream/tests.rs`，端到端测试在
`crates/ai/tests/bedrock_converse_stream.rs`。

对外接口：

- `BedrockConverseStreamApi`（单元结构体，实现 `ProviderStreams`）；
- `stream(model, context, StreamOptions)`；
- `stream_simple(model, context, SimpleStreamOptions)`。

API 专属选项取自 `options.api` 中的 `ApiOptions::BedrockConverseStream(BedrockOptions)`，其他变体忽略。
兼容设置取自 `ModelCompat::Bedrock`（目前只有 `supportsStrictMode`）。

## 上游源文件

- `packages/ai/src/api/bedrock-converse-stream.ts`（主体，1416 行）。
- `packages/ai/src/api/constrained-sampling.ts`（严格工具 schema 转换，在共享模块 `api::constrained_sampling` 中）。
- `packages/ai/src/api/bedrock-converse-stream.lazy.ts`、`src/bedrock-provider.ts`：只负责懒加载 Node 专用模块，
  Rust 里由 feature 开关代替，没有对应代码。
- 测试：`packages/ai/test/bedrock-*.test.ts`；Python 移植 `pi_ai/api/bedrock_converse_stream.py` 作为第二参考。

## 请求映射

### 流程

1. `stream()` 先用 `collapse_system_messages` 把对话中途的 system 消息并入开头的系统提示（Bedrock 不支持中途 system 消息）。
2. 解析客户端配置（region、endpoint、profile、凭证、bearer token），见下节。
3. 构造 JSON 形式的请求体，形状与上游 `ConverseStreamCommand` 的输入一致（camelCase 成员；图片字节和加密推理等二进制成员用 base64 字符串表示）。
4. 调用 `on_payload`，回调可以替换整个 JSON。
5. 把 JSON 转换为 SDK 的强类型输入（`ConverseRequest::from_payload`）。SDK 不认识的顶层成员被忽略，与 JS SDK 命令序列化行为一致。
6. 用 `aws_config` 构造客户端并发送，读取事件流。

### 客户端配置（与上游 Node 分支一致）

| 项 | 规则 |
| --- | --- |
| profile | `BedrockOptions::profile`，否则 `options.env` 中的 `AWS_PROFILE`（"显式 profile"），否则进程环境的 `AWS_PROFILE` |
| region | 推理配置 ARN 中的 region > `BedrockOptions::region` > `AWS_REGION` > `AWS_DEFAULT_REGION` > 标准 endpoint 中的 region（仅在 endpoint 被固定时）> 没有环境 `AWS_PROFILE` 时用 `us-east-1`；否则交给 SDK 从 profile 解析 |
| endpoint | `model.base_url` 不是标准 `bedrock-runtime[-fips].<region>.amazonaws.com[.cn]` 时总是使用；是标准 endpoint 时，只有在既没配置 region 也没有环境 `AWS_PROFILE` 时才固定 |
| 静态凭证 | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`（+ `AWS_SESSION_TOKEN`），先查 `options.env` 再查进程环境；有显式 profile 时不使用（上游 #6957） |
| bearer token | `BedrockOptions::bearer_token` > `options.api_key` > `AWS_BEARER_TOKEN_BEDROCK`；设置后 `auth_scheme_preference = httpBearerAuth`，不做 SigV4 |
| `AWS_BEDROCK_SKIP_AUTH=1` | 使用 dummy 静态凭证，不使用 bearer token，并把认证方案固定为 SigV4（SDK 自己也会读取 `AWS_BEARER_TOKEN_BEDROCK`，需要覆盖） |

环境变量查找：`options.env` 中的值优先，然后是进程环境，空白值视为未设置。

### 请求体字段

| 字段 | 来源 |
| --- | --- |
| `modelId` | `model.id` |
| `messages` | `transform_messages`（tool call id 归一化为 `[a-zA-Z0-9_-]`、最长 64，长度按 UTF-16 计算），再按下表转换 |
| `system` | 开头 system 消息的完整文本（含 sections）；支持缓存的模型追加 `cachePoint` |
| `inferenceConfig` | `maxTokens`：`options.max_tokens`，Claude 模型缺省用 `model.max_tokens`；`temperature` |
| `toolConfig` | 当前工具集；无工具或 `tool_choice = None` 时省略；`toolChoice` 为 `auto` / `any` / `tool{name}` |
| `additionalModelRequestFields` | 推理配置，见下文 |
| `requestMetadata` | `BedrockOptions::request_metadata` |

消息转换：

- 用户消息：字符串内容为空白时发送 `<empty>`；块内容中空白文本块被丢弃，全部丢弃时发送 `<empty>`；图片按 MIME 类型映射为 `jpeg/png/gif/webp`，
  其它类型报错 `Unknown image type: ...`。base64 解码与浏览器 `atob` 一样宽松（容忍空白和缺失填充）。
- 助手消息：内容为空的消息跳过；空白文本跳过；工具调用参数去掉空键名（Bedrock 拒绝空属性名，只影响发送的副本）；
  thinking：redacted 块按 base64 解码后作为 `reasoningContent.redactedContent` 回放（解码失败则丢弃该块）；
  Claude 模型带签名时发 `reasoningText{text, signature}`，没有签名时降级为普通文本；非 Claude 模型只发 `reasoningText{text}`。
  所有块被过滤掉后整条消息跳过。
- 工具结果：连续的工具结果合并成一条 user 消息；空内容发送 `<empty>`；Claude 模型的错误结果中的图片用
  `split_error_tool_result_images` 拆出，放在本轮所有工具结果之后（上游 #10794），其它模型保留原位。
- 缓存点：缓存保留策略不是 `none` 且模型支持提示缓存（Claude 3.5 Haiku、3.7 Sonnet、4.x、5，或 `AWS_BEDROCK_FORCE_CACHE=1`）时，
  在 system 和最后一条 user 消息末尾追加 `{"cachePoint": {"type": "default"}}`，`long` 时带 `"ttl": "1h"`。
  缓存策略缺省为 `short`，`PI_CACHE_RETENTION=long` 时为 `long`。

严格工具：`compat.supportsStrictMode` 为真且工具声明 `constrainedSampling: json_schema` 时，把 schema 转为严格子集并设置 `strict: true`；
不能转换时，`prefer` 退回普通工具，`require` 报错；模型不支持严格模式时 `require` 报错。

### 推理（`additionalModelRequestFields`）

只在 `BedrockOptions::reasoning` 有值且 `model.reasoning` 为真时发送。模型族通过 id 和 name 匹配（兼容应用推理配置 ARN）。

- Claude，自适应思考模型（Opus 4.6/4.7/4.8/5、Sonnet 4.6/5、Haiku 5、Fable 5）：
  `thinking: {type: "adaptive", display, block_binding?}`、`output_config: {effort}`；
  支持 block binding 的模型（4.6 以外）加 `block_binding: {prefix_mismatch_behavior: "drop_block"}` 和
  `anthropic_beta: ["thinking-binding-controls-2026-08-01"]`。
  effort：`xhigh` 在原生支持的模型上直接用；否则查 `thinking_level_map`；否则 minimal/low→low、medium→medium、其余→high。
- Claude，预算思考模型：`thinking: {type: "enabled", budget_tokens, display}`，预算取自定义预算（xhigh/max 按 high），否则默认表；
  `interleaved_thinking`（缺省 true）时加 `anthropic_beta: ["interleaved-thinking-2025-05-14"]`。
- `display` 缺省 `summarized`；GovCloud（region 以 `us-gov-` 开头，或 id 以 `us-gov.` / `arn:aws-us-gov:` 开头）不发送 display 和 block binding。
- gpt-oss：`reasoning_effort`（low/medium/high）；其它 GPT：`reasoning: {effort}`（优先 `thinking_level_map`）。

### `stream_simple`

与上游 `streamSimple` 相同：`build_base_options` + `tool_choice`（Auto/None 映射为 Bedrock 的 auto/none）。
没有 reasoning 时不发推理字段；Claude 自适应模型和非 Claude 模型直接传 `reasoning` 和自定义预算；
Claude 预算模型用 `adjust_max_tokens_for_thinking` 计算 `max_tokens`，再按上下文窗口截断，
并把对应级别的预算设为 `min(预算, max(0, max_tokens - 1024))`。

### 自定义请求头

`options.headers` 按名称大小写不敏感合并（`None` 删除），跳过保留头 `x-amz-*`、`authorization`、`host`，
通过 SDK 的 `mutate_request`（在签名前执行）写入请求，因此参与 SigV4 签名。没有可写入的头时不注册拦截器。
`options.headers` 已包含 `Models` 合并的模型头。

## 流事件映射

| Bedrock 事件 | 处理 |
| --- | --- |
| `messageStart` | role 必须是 assistant，否则报错；`builder.start()` |
| `contentBlockStart.toolUse` | `start_tool_call(toolUseId, name)`，记录 Bedrock 块索引 |
| `contentBlockDelta.text` | 该索引没有打开的块时先 `start_text`（文本块没有 start 事件），然后 `text_delta` |
| `contentBlockDelta.toolUse` | 对应工具块的 `tool_call_delta` |
| `contentBlockDelta.reasoningContent.text` | 没有块时 `start_thinking`（签名初值为空字符串，与上游一致），然后 `thinking_delta` |
| `contentBlockDelta.reasoningContent.signature` | 非 redacted 块时追加到 `thinking_signature` |
| `contentBlockDelta.reasoningContent.redactedContent` | 第一次出现时把块标记为 redacted、签名清空、追加占位文本 `[Reasoning redacted]`（发一次 `thinking_delta`）；字节累积到临时缓冲 |
| `contentBlockStop` | 文本 `end_text`；thinking 先把累积字节编码为 base64 写入签名再 `end_thinking`；工具 `end_tool_call`（解析完整参数）；块的 Bedrock 索引清除 |
| `messageStop` | 记录 `raw_stop_reason`；`end_turn`/`stop_sequence`→stop，`max_tokens`/`model_context_window_exceeded`→length，`tool_use`→toolUse，其它→error 且 `Provider stopped with: <reason>` |
| `metadata.usage` | input/output/cacheRead/cacheWrite；`cacheWrite1h` 为 `cacheDetails` 中 ttl=1h 的合计（没有 details 时不设置）；`total_tokens` 用 Bedrock 的 totalTokens，为 0 时用 input+output；按模型价格计算费用（1h 缓存写入按 2 倍输入价） |
| 未知事件/增量类型 | 忽略 |
| 流内异常（`internalServerException` 等五种） | 先交给 `on_provider_stream_event`，再作为错误结束 |

每个事件在处理前都以上游 JS SDK 的形状（如 `{"contentBlockDelta": {...}}`，二进制为 base64）传给 `on_provider_stream_event`。

流结束后：已取消→aborted；stop reason 仍为 pending→`Bedrock stream ended without a stop reason`；stop reason 为 error→以其消息失败。
成功或失败时都会把仍在缓冲的加密推理写回签名（流可能不发 `contentBlockStop` 就结束）。

## 错误与重试

- 错误消息与上游 `formatBedrockError` 一致：Bedrock 异常加可读前缀（`Validation error:`、`Throttling error:`、`Internal server error:`、
  `Model stream error:`、`Service unavailable:`，其余用异常名本身，SDK 无法识别时为 `Unknown`）；
  服务端没有给出 message 时用 `"<status>: <body>"`（截断到 4000 字符），避免网关 403 变成 `Unknown: UnknownError`；
  包含 "data retention mode" 时追加文档链接提示。
- 诊断：失败（非 abort）时追加 `bedrock_response_failure` 诊断，`details` 只包含已知字段：`status`、`errorCode`（以 `Exception` 结尾、
  不超过 200 字符）、`requestId`（来自错误响应头，或成功响应的 `x-amzn-requestid` 作为流内错误的回退）。`error_message` 不受影响。
- `on_response`：通过拦截器在反序列化前记录每次 HTTP 响应的状态和全部响应头，请求成功后对最后一次（成功的）响应调用一次，
  所以网关自定义头也能看到。
- 重试：SDK 标准重试策略，`max_attempts = max_retries + 1`（缺省 3 次，与 JS SDK 缺省相同）。
- 超时：`options.timeout`（缺省 10 分钟）作为 SDK 的 operation timeout（覆盖发送和全部重试），并作为事件流的空闲超时（两个事件之间的最长等待）。
  SDK 的 stalled-stream 保护被关闭，避免长时间推理停顿被误判。
- 取消：`options.cancel` 与 SDK 构造、`send()` 以及每次 `recv()` 做 `select`，取消后丢弃 future 即中断请求；消费者丢弃流时同样停止。
- 失败时保留已收到的部分内容；未结束的工具调用保留已流入 JSON 的尽力解析结果（与上游 partial 消息一致）。

## Rust 特有的差异

- 使用 AWS 官方 SDK 而非上游的 JS SDK：请求先构造为 JSON（供 `on_payload` 使用）再转换为 SDK 类型。`on_payload` 返回的 JSON 中
  SDK 不支持的成员会被忽略，格式错误的成员报 `Invalid Bedrock payload: ...`。
- SDK 的 `Document` 使用 `HashMap`，工具 schema、工具参数和 `additionalModelRequestFields` 在线上的对象键顺序不固定（上游 JS 保持插入顺序）。
- Rust 的默认凭证链总是先读环境变量；为保持上游"显式 profile 优先于环境密钥"的语义，有 profile 且没有静态凭证时直接使用
  `ProfileFileCredentialsProvider`（覆盖 SSO、credential_process、assume role，但不包括 ECS/IMDS）。
- `aws-config` 不导出 `AuthSchemeId`，认证方案偏好通过类型推断构造（`scheme_preference`）。
- Rust 字符串不会包含未配对代理项，上游的 `sanitizeSurrogates` 是空操作，未移植。
- 流内建模异常（如 `throttlingException`）在 Rust SDK 中是带类型的错误，诊断里会带上 `errorCode`；上游测试里这种异常是裸对象，只有 `requestId`。
- 上游在 Node 之外（浏览器）的 region 回退分支不适用，未移植。
- 未移植（见代码中 `// Not ported` 注释）：`options.env` 中的代理设置和 `AWS_BEDROCK_FORCE_HTTP1`。SDK 默认 HTTPS 客户端读取进程环境的
  `HTTP(S)_PROXY`/`NO_PROXY`，HTTP 版本由客户端自行协商。`options.http_client`（reqwest）对本 API 无效。
- `max_retry_delay` 没有映射：SDK 的退避策略不读取 `retry-after`，没有对应概念。
- `thinking` 块在流中途变成 redacted 时，`MessageBuilder` 没有对应事件，消费端的 partial 消息直到结束事件才看到 `redacted: true`；最终消息正确。

## 测试

- 单元测试（`bedrock_converse_stream/tests.rs`，52 个）：客户端配置（凭证优先级、endpoint/region 解析、ARN、bearer token、skip auth）、
  请求体（严格工具、tool choice、metadata、占位文本、错误结果图片拆分、空键名、thinking 回放、redacted 回放、跨模型回放和孤立工具调用、
  缓存点、推理字段的全部分支）、JSON→SDK 转换、事件处理（文本、并行工具调用、签名、redacted 拼接、无 stop 事件、用量与 1h 缓存费用）、
  错误格式和诊断、`stream_simple` 的预算映射。
- 端到端测试（`tests/bedrock_converse_stream.rs`，16 个）：真实 SDK 客户端连接 wiremock，响应体是用 `aws-smithy-eventstream`
  编码的事件流帧。覆盖文本+用量+响应头、thinking+并行工具调用、redacted 往返、线上请求体形状、`on_payload` 替换、
  自定义头参与签名、bearer token、HTTP 服务错误、网关错误、5xx 重试、流内异常、错误 stop reason、缺少 stop reason、取消、超时、跨模型回放。
- 新增开发依赖：`aws-smithy-eventstream = "0.61"`（工作区和 `crates/ai` 的 dev-dependencies），仅用于在测试中编码事件流帧。

## 待确认的改进

- 缓存 `SdkConfig`/`Client`：上游每次请求新建客户端（这里保持一致），凭证链（SSO、assume role、IMDS）每次都会重新解析；pi-py 按配置缓存客户端。
- `Document` 键顺序不稳定可能影响 Bedrock 端的提示缓存命中（工具定义位于缓存前缀中）。可选方案：在签名前用保持顺序的 JSON 重写请求体，或等待 SDK 支持有序文档。
- 支持 `options.env` 中的代理设置（需要直接依赖 `aws-smithy-http-client` 构造 HTTP 客户端）。
- 把更多 `ConverseStream` 成员（`guardrailConfig`、`performanceConfig`、`serviceTier` 等）纳入 `on_payload` 的 JSON→SDK 转换。
