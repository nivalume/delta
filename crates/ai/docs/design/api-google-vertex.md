# API 设计：google-vertex

## 目的

`google-vertex` 通过 Vertex AI 流式调用 Gemini 模型，把响应转换成 delta 统一的 `AssistantMessageEvent` 协议。

上游通过 `@google/genai` 的 Vertex 模式发 REST 请求。按 `AGENTS.md`，Vertex AI 用 Google Cloud 官方 Rust 客户端库：

- `google-cloud-aiplatform-v1`（feature `prediction-service`）：`PredictionService::stream_generate_content`，这是 gRPC 服务端流；
- `google-cloud-auth`：Application Default Credentials（ADC）、凭证文件、Vertex API key。

整个模块在 `vertex` feature 之后（`#[cfg(feature = "vertex")]`）。

模块：`crates/ai/src/api/google_vertex.rs`，单元测试在 `google_vertex/tests.rs`，端到端测试在 `crates/ai/tests/google_vertex.rs`。

对外接口：

- `GoogleVertexApi`（单元结构体，实现 `ProviderStreams`）；
- `stream(model, context, StreamOptions)`、`stream_simple(model, context, SimpleStreamOptions)`；
- `stream_with_client(model, context, StreamOptions, PredictionService)`：用调用方给的客户端（自定义端点、凭证、传输或测试桩）发请求，其余流程与 `stream` 相同；
- `get_google_budget`（Vertex 的思考预算表）、`proto_request`、`response_chunk`、`response_json`（转换函数，公开以便测试和复用）。

API 专属选项取自 `ApiOptions::GoogleVertex(GoogleVertexOptions)`（`tool_choice`、`thinking`、`project`、`location`），其他变体忽略。

## 上游源文件

- `packages/ai/src/api/google-vertex.ts`：`stream`、`streamSimple`、客户端创建、`getGoogleBudget`。
- `packages/ai/src/api/google-shared.ts`：消息/工具转换、思考配置、签名、结束原因、重试。Rust 侧已移植为 `api::google_shared`（见 `api-google-generative-ai.md`），本模块直接复用，不修改。
- `packages/ai/src/providers/google-vertex.ts`：provider 工厂和认证解析（`GOOGLE_CLOUD_API_KEY`、ADC 文件检查），移植在 `providers/builtin.rs` 的 `vertex` 模块（`VertexAuth`，含登录流程）。本模块只做上游 API 模块自己做的解析。
- 测试：`test/google-vertex-api-key-resolution.test.ts`、`google-raw-stop-reason.test.ts`、`google-thinking-level-map.test.ts` 中的 Vertex 部分；pi-py 的 `google_vertex.py` 和 `test_google_streams.py` 作为第二参考。

## 与 google_shared 的分工

`google_shared` 只处理普通数据（`GenerateRequest`、`ResponseChunk`、`StreamState`），与传输无关。本模块只做：

1. 认证、端点、模型资源名的解析和客户端构造；
2. `GenerateRequest` → proto `GenerateContentRequest`；
3. proto `GenerateContentResponse` → `ResponseChunk`，交给同一个 `StreamState`；
4. 结束时 `check_finished(output, "Google Vertex stream ended without a finish reason")`。

### 请求转换：走 REST JSON + serde

生成的 proto 类型实现了 serde，字段名就是 REST/proto JSON 的 camelCase 名，bytes 用 base64，枚举接受名字字符串（也接受数字）。因此请求先用 Gemini 模块的 `google_generative_ai::request_body` 渲染成 REST JSON（这正是 `@google/genai` 发给 Vertex 的请求体），交给 `on_payload`，再 `serde_json::from_value::<GenerateContentRequest>` 转成 proto，最后填 `model`。

这样做的理由：

- `on_payload` 看到的 JSON 与 Gemini 模块相同，也与上游 SDK 实际发送的请求体字段一致，回调可以按 REST 字段名修改（如 `generationConfig.topK`、`labels`），proto 定义的字段都能生效；
- 不用手写一套 `Part`/`Content`/`Tool` 的 proto 构造代码；
- 思考签名、图片等 base64 文本由 serde 解码成 bytes。

注意：

- proto 没有定义的字段（例如 `functionCall.id`、`functionResponse.id`）会进入生成类型的未知字段表，gRPC 编码时被丢弃。
- JSON 无法转成 proto 时（类型不对、整数越界、非法 base64）请求在发送前失败：`Invalid Google Vertex request payload: <原因>`。
- `on_payload` 返回的 JSON 若带 `model` 字段，则保留它（必须是完整资源名）；否则用解析出的资源名。

### 响应转换：直接读 proto

响应方向不走 JSON：生成类型序列化枚举时输出数字（`finishReason: 1`），与 REST 不同。`response_chunk` 直接读 proto 字段：

- 只看第一个候选；
- part：`text()`、`thought`（`true` 才记为思考）、`thought_signature`（非空 bytes 用标准 base64 编码成文本，与 REST 返回的形式相同）、`function_call()`（`name` 为空视为缺失，`args` 原样）；
- `finish_reason`：`FINISH_REASON_UNSPECIFIED`（proto 默认值）视为没有结束原因；已知值用其名字（`STOP`、`SAFETY`……）；未知的数字值用数字文本；
- `usage_metadata`：proto3 标量没有“缺失”，0 和负数视为缺失。REST JSON 也省略 0 值，所以结果与 Gemini 模块一致（`promptTokenCount` 为 0 时不更新用量）；
- `response_id`：空串视为缺失。

`on_provider_stream_event` 拿到的是 `response_json`：生成类型的 serde JSON（REST 字段名、base64 bytes、省略默认值），其中 `candidates[].finishReason` 换成名字字符串；其他枚举字段（安全评级、`trafficType` 等）保持数字。只有设置了回调才序列化。

## 请求映射

### 流程

1. 已取消：直接以 `aborted` 结束，不发请求。
2. 解析认证、端点和模型资源名（顺序与上游相同：API key，否则 project，再 location）。
3. 构造客户端（`stream_with_client` 跳过这一步）。
4. `google_shared::build_request` → `request_body` → `on_payload` → `proto_request`。
5. `retry_google_request` 包住 `stream_generate_content().send()`；成功后 `start`。
6. 逐个读流（`http::or_cancel` 让取消立即生效），每个响应先交给 `on_provider_stream_event`，再交给 `StreamState::apply_chunk`。

### 认证

- `options.api_key` 去掉首尾空白后非空、不是 `gcp-vertex-credentials`、不是 `<...>` 形式的占位符（如 `<authenticated>`）：用 Vertex API key（express 模式），`api_key_credentials` 发 `x-goog-api-key`，不需要 project/location。
- 否则用 ADC：
  - project：`GoogleVertexOptions::project` → `GOOGLE_CLOUD_PROJECT` → `GCLOUD_PROJECT`；都没有时报错 `Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options.`
  - location：`GoogleVertexOptions::location` → `GOOGLE_CLOUD_LOCATION`；没有时报错 `Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options.`
  - 凭证：`GOOGLE_APPLICATION_CREDENTIALS` 有值时（上游 `keyFilename`）读这个文件，按 `type` 构造：`service_account`、`authorized_user`、`impersonated_service_account`、`external_account`；其他类型报错 `Unsupported Google credentials type in <path>: <type>`，读/解析失败分别报 `Failed to read/parse Google credentials file <path>: …`。没有时用 `google-cloud-auth` 的默认 ADC（gcloud 的 well-known 文件，再退到 metadata server）。
- 环境变量都先查 `options.env`，再查进程环境，空白值视为未设置（`StreamOptions::env_var`）。
- 上游 API 模块本身不读 `GOOGLE_CLOUD_API_KEY`（由 provider 工厂解析成 `apiKey`），这里也不读。

### 端点

gRPC 的路径是固定的（`/google.cloud.aiplatform.v1.PredictionService/StreamGenerateContent`），端点只取 scheme 和 authority：

1. 模型的 `base_url`（去空白，非空且不含 `{location}` 占位符，即上游 `resolveCustomBaseUrl`）；
2. 否则 `GOOGLE_VERTEX_BASE_URL`（JS SDK 从进程环境读它）；
3. 否则按 location：API key 和 `global` 用 `https://aiplatform.googleapis.com`；多区域 `us`/`eu` 用 `https://aiplatform.{us|eu}.rep.googleapis.com`（与 Python SDK、pi-py 一致）；其余 `https://{location}-aiplatform.googleapis.com`（SDK 的区域端点）。

内置目录里 Vertex 模型的 `baseUrl` 是 `https://{location}-aiplatform.googleapis.com`，因此默认走第 3 条。

### 模型资源名

先按 SDK 的 `tModel`（Vertex 模式）得到路径：id 含 `..`、`?`、`&` 时报 `invalid model parameter`；以 `publishers/`、`projects/`、`models/` 开头的原样使用；含 `/` 的取前两段作 `publishers/{a}/models/{b}`；其余为 `publishers/google/models/{id}`。

- API key：直接用这个路径；
- ADC：路径不以 `projects/` 开头时加前缀 `projects/{project}/locations/{location}/`。

### 请求头

上游把 `{User-Agent: pi UA, ...model.headers, ...options.headers}` 设为 SDK 的 `httpOptions.headers`。这里 `options.headers`（`Models` 已合并模型的 headers；值为 `None` 的项删除）作为客户端级的请求元数据（`ClientBuilder::with_extension(HeaderMap)`）。SDK 会丢弃 `user-agent`、`authorization`、`x-goog-api-key`、`x-goog-api-client`、`x-goog-user-project`、`x-goog-request-params` 的自定义值，由它自己设置。User-Agent 通过 `RequestOptionsBuilder::with_user_agent` 按请求设置：默认 delta 的 User-Agent，`options.headers` 里有 `user-agent` 时用它的值，与上游一致。

### 请求体

与 Gemini 模块相同（见 `api-google-generative-ai.md`“请求体”一节），字段在 proto 中的对应：

| `GenerateRequest` 字段 | REST JSON | proto 字段 |
| --- | --- | --- |
| `contents` | `contents[]` | `contents`（`role`、`parts`：`text`、`thought`、`thoughtSignature`、`inlineData`、`functionCall`、`functionResponse`） |
| `system_instruction` | `systemInstruction: {role: "user", parts: [{text}]}` | `system_instruction` |
| `temperature` | `generationConfig.temperature` | `generation_config.temperature`（f32） |
| `max_output_tokens` | `generationConfig.maxOutputTokens` | `generation_config.max_output_tokens`（i32） |
| `thinking_config` | `generationConfig.thinkingConfig` | `generation_config.thinking_config`（`include_thoughts`、`thinking_level` 枚举、`thinking_budget` i32） |
| `tools` | `tools: [{functionDeclarations: [{name, description, parametersJsonSchema}]}]` | `tools[0].function_declarations[].parameters_json_schema` |
| `function_calling_mode` | `toolConfig.functionCallingConfig.mode` | `tool_config.function_calling_config.mode`（`AUTO`/`NONE`/`ANY`/`VALIDATED`） |

`model` 字段是解析出的资源名。消息转换、思考配置、严格工具 schema、函数调用模式的规则全部来自 `google_shared`，与 Gemini 模块相同（包括需要 id 的模型给工具调用加 id，但 v1 proto 没有 id 字段，见下）。

### `stream_simple`

与上游 `streamSimple` 相同：`build_base_options`，`tool_choice` 映射，`reasoning` 为空或被钳制为 `off` 时 `thinking: {enabled: false}`，否则用 `thinkingLevel` 的模型发级别，其余发预算。

预算用 Vertex 自己的表（上游 `google-vertex.ts` 的 `getGoogleBudget`）：自定义 `thinking_budgets` 优先，`2.5-pro` 为 128/2048/8192/32768，`2.5-flash`（**包括** `2.5-flash-lite`，上游 Vertex 表没有 Flash-Lite 一行）为 128/2048/8192/24576，其他 `-1`。因为这张表与 Gemini API 的不同，`google_shared::simple_thinking` 不能直接用，本模块复制了它的逻辑（见“需要的核心改动”）。

与上游一样，`stream_simple` 不保留 `project`/`location`（`build_base_options` 清掉 `api`），它们只能来自 `options.env` 或进程环境。

## 流事件映射

| proto 响应 | 事件 / 输出字段 |
| --- | --- |
| `send()` 成功（gRPC 响应头已到） | `Start` |
| `response_id` | `response_id`，保留第一个非空值 |
| part `text`，`thought: true` | 当前不是思考块时结束当前块并 `ThinkingStart`；`ThinkingDelta` |
| part `text`，非思考 | 当前不是文本块时结束当前块并 `TextStart`；`TextDelta` |
| part `thought_signature`（文本/思考 part） | base64 文本，作为当前块的签名，保留最后一个非空值 |
| part `function_call` | 结束当前块；`ToolCallStart`、`ToolCallDelta`（`args` 的紧凑 JSON）、`ToolCallEnd`；非空签名记到工具调用上 |
| 函数调用 id | v1 proto 没有，总是生成 `<name>_<毫秒时间戳>_<计数>`（跨 chunk 唯一） |
| `finish_reason` | `raw_stop_reason` 为其名字；`STOP` → `stop`，`MAX_TOKENS` → `length`，其他 → `error`；已有工具调用且为 `stop` 时改为 `toolUse` |
| `usage_metadata`（`prompt_token_count` > 0 时） | `input = prompt - cached`，`output = candidates + thoughts`，`cache_read = cached`，`reasoning = thoughts`，`total_tokens` 取服务器值，费用 `calculate_cost` |
| 流中出现错误状态 | 立即失败，不重试，已收到的内容保留 |
| 流结束 | 结束当前块；已取消 → `aborted`；没有结束原因 → `Google Vertex stream ended without a finish reason`；结束原因映射为 `error` → `Provider stopped with: <raw>`；否则 `Done` |

## 错误、重试、超时与取消

- SDK 错误转换（`vertex_error`）：带 gRPC 状态的服务错误按 `google.rpc.Code` 的 HTTP 映射转成 `Error::Status`（`INVALID_ARGUMENT`/`FAILED_PRECONDITION`/`OUT_OF_RANGE` 400，`UNAUTHENTICATED` 401，`PERMISSION_DENIED` 403，`NOT_FOUND` 404，`ALREADY_EXISTS`/`ABORTED` 409，`RESOURCE_EXHAUSTED` 429，`CANCELLED` 499，`UNKNOWN`/`INTERNAL`/`DATA_LOSS` 500，`UNIMPLEMENTED` 501，`UNAVAILABLE` 503，`DEADLINE_EXCEEDED` 504），消息为 `"<status> <status.message>"`（消息为空时用状态码名）。只有 HTTP 状态的错误（代理、负载均衡返回）同样转成 `Error::Status`。其他错误（连接、认证、超时、序列化）用 SDK 的错误文本。
- 重试：`google_shared::retry_google_request`（上游 `retryGoogleRequest`）包住打开流的调用，`Error::Status` 为 408/409/429/5xx 时按 `max_retries`（默认 2）指数退避重试；其他错误和流中错误不重试。gax 的重试策略不作用于服务端流，所以不用它。没有 `retry_after_ms`（gRPC 的 `RetryInfo` 需要 `google-cloud-rpc`，上游 SDK 也从不使用服务端延迟）。
- 超时：`options.timeout`（默认 10 分钟）设为客户端的 `attempt_timeout`，gRPC 把它作为整个调用（包括流）的 deadline，与其他 API 的 HTTP 超时语义相同。用 `stream_with_client` 时以调用方客户端的配置为准。
- 取消：请求前检查；打开流和重试等待由 `retry_google_request` 响应取消；读流时每次 `next()` 都与 `options.cancel` 竞争。取消后以 `aborted` 结束，消息 `Request was aborted`，保留部分内容。消费者丢弃流时停止读取。
- `on_payload`、`on_provider_stream_event` 返回错误时请求失败。没有 `on_response`（上游 Vertex 适配器不调用它，gRPC 流也没有 HTTP 响应）。

## Rust 特有差异

- 传输是 gRPC，不是 REST：
  - 模型的 `base_url` 只用 scheme 和 authority，路径（API 版本、`projects/.../locations/...` 集合路径）不用。上游的 `baseUrlResourceScope: COLLECTION` 和“URL 已含版本时 `apiVersion: ""`”在 gRPC 下没有对应。
  - 自定义 `base_url` 时上游不加 `projects/{p}/locations/{l}` 前缀；gRPC 的 `model` 必须是完整资源名，所以 ADC 模式总是加前缀。
  - v1 proto 的 `FunctionCall`/`FunctionResponse` 没有 `id`：Gemini 3 等模型的调用 id 不会发出（上游 REST 会发），响应里的 id 也拿不到，工具调用 id 总是本地生成。Vertex 按顺序和名字配对调用与结果。
- `on_payload` 拿到的是 REST 请求体（与 Gemini 模块相同），不是 SDK 的 `{model, contents, config}`；proto 没有的字段不会发出。
- `on_provider_stream_event` 拿到的是 proto 的 serde JSON，见上；非 `finishReason` 的枚举是数字。
- 错误消息用 `"<status> <message>"`，上游是 SDK 拼出的 JSON 正文。
- 重试次数默认 2（上游 Google 适配器默认 0），与 Gemini 模块相同。
- 上游没有给 Vertex SDK 设超时；这里应用 `options.timeout`（核心的统一语义）。
- SDK 另外发送自己的 `x-goog-api-client`。
- `options.http_client` 不使用（上游对自定义 `fetch` 报错，这里没有这个检查）。
- 凭证文件按 `type` 分派，支持的类型同 `google-cloud-auth` 的 ADC（不含 `gdch_service_account`）；上游 `google-auth-library` 的 `keyFilename` 另外支持 `.p12`/`.pem` 等旧格式。
- 预先取消时消息是 `Request was aborted`（上游 `buildParams` 里是 `Request aborted`）。
- 未知的数字结束原因记为数字文本；REST 下会是服务器给的名字。两者都映射为 `error`。
- 每个请求新建客户端和凭证（与上游每次 `new GoogleGenAI` 相同）。
- `sanitizeSurrogates` 不需要：Rust 字符串不含孤立代理项。

## 未移植

- 上游拒绝自定义 `fetch` 的检查（见上）。
- 遥测：上游 Vertex 适配器没有。

## provider-references 检查

- 密钥不进 URL：API key 由 SDK 放在 `x-goog-api-key` 元数据里。
- 出现函数调用时 `STOP` 改为 `toolUse`（`MAX_TOKENS` 保持 `length`）：是。
- 未知结束原因不导致崩溃：映射为 `error` 并保留原值。
- 空签名视为无；非思考文本 part 上的签名保留；思考 part 作为思考块：是。
- 无 id 的调用生成跨 chunk 唯一的 id：是（v1 proto 总是无 id）。
- 用量等待 `promptTokenCount`、饱和减法、输出含 `thoughtsTokenCount`：是。
- 流中错误：gRPC 状态立即失败，不重试；没有结束原因时报错：是。
- 宽松解析：proto 类型由 SDK 解码，未知字段/枚举值不致命（未知枚举保留为数字）。
- `parametersJsonSchema`：是；`$defs`/`definitions` 内联未实现（与 Gemini 模块相同，见其文档）。
- 重试：408/409/429/5xx 重试，连接错误不重试（与上游 `retryGoogleRequest` 一致）。

## 核心改动

- `vertex` feature 引入可选依赖 `google-cloud-gax`（官方 crate，`aiplatform` 已间接依赖它），用于 `with_user_agent`。
- `google_shared::simple_thinking_with` 接受预算函数；Vertex 传入自己的 `get_google_budget`（没有 2.5-flash-lite 行，与上游一致）。

## 待确认的改进

- 按（端点、认证方式、key/项目/凭证文件、请求头）缓存 `PredictionService`，复用 gRPC 连接和令牌缓存（pi-py 这样做；上游每次新建客户端）。
- 读取 gRPC 错误详情里的 `RetryInfo.retry_delay` 作为重试延迟。
- `stream_simple` 保留 `options.base.api` 中 `GoogleVertexOptions` 的 `project`/`location`（上游丢弃）。
- 把目录里 `baseUrl` 的 `{location}` 占位符替换成解析出的 location，而不是忽略它（结果相同，只是更直接）。
- 用 `model_version` 填 `response_model`（上游不填）。
- API key（express 模式）走 gRPC 时，`model` 为 `publishers/google/models/{id}` 是否被服务端接受，需要用真实 key 验证；不接受时改为 REST 传输或要求 project/location。
- 如果需要发送函数调用 id（Gemini 3），只能换成带 `id` 字段的 API 版本或 REST 传输。
- 流式函数参数（`partialArgs`/`willContinue`、`streamFunctionCallArguments`）未使用，与上游相同。
