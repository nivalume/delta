# API：mistral-conversations

## 目的

`mistral-conversations` 对接 Mistral 原生的 Chat Completions 接口：`POST {baseUrl}/v1/chat/completions`，`stream: true`，用 SSE 返回增量。上游通过 Mistral SDK 的 HTTP 层调用该接口；tau 没有 Mistral 官方 Rust SDK，所以在 `reqwest` + `tau_ai::http` 上手写客户端，请求字段与上游发送的完全一致。

实现位于 `crates/ai/src/api/mistral_conversations.rs`，对外提供：

- `MistralConversationsApi`（单元结构体，实现 `ProviderStreams`）；
- `stream(model, context, options)`；
- `stream_simple(model, context, options)`。

API 专属选项来自 `options.api` 中的 `ApiOptions::MistralConversations(MistralOptions)`，其他变体一律忽略。模型兼容设置为 `ModelCompat::Mistral`，目前只用到 `supportsMidConvoSystemMessages`。

## 上游源文件

- `packages/ai/src/api/mistral-conversations.ts`（主体）
- `packages/ai/src/api/constrained-sampling.ts`（严格 JSON Schema，本 API 用到的部分）
- `packages/ai/src/utils/hash.ts`（`shortHash`）
- 测试：`test/mistral-http-transport.test.ts`、`test/mistral-raw-stop-reason.test.ts`、`test/mistral-reasoning-mode.test.ts`、`test/mistral-tool-schema.test.ts`
- 参考：pi-py `pi_ai/api/mistral_conversations.py` 与 `tests/test_mistral_conversations.py`

## 请求映射

### 流程

1. `resolve_transcript`：模型 `compat.supportsMidConvoSystemMessages` 为真时保留对话中途的 system 消息，否则折叠成开头的一条。
2. 必须有 `options.api_key`（与上游一致，只有 `authorization` 头不算），否则报错 `No API key for provider: <provider>`。
3. `transform_messages`，并传入 Mistral 工具调用 id 规范化函数（见下文）。
4. 按 Mistral SDK 的 camelCase 形状构造请求体，交给 `on_payload`（它看到的就是 camelCase，例如 `maxTokens`、`promptMode`、`promptCacheKey`）。
5. 发送前把 camelCase 字段改名为线上的 snake_case（`to_wire_payload`）。
6. 收到响应头后调用 `on_response`（成功和失败的响应都会调用），状态码成功才发出 `start` 事件并开始读 SSE。

### 请求体字段

| SDK 字段（`on_payload` 所见） | 线上字段 | 来源 |
| --- | --- | --- |
| `model` | `model` | `model.id` |
| `stream` | `stream` | 固定 `true` |
| `messages` | `messages` | 见下表 |
| `tools` | `tools` | 当前工具集（`get_current_tools`），为空则不发送 |
| `temperature` | `temperature` | `options.temperature` |
| `maxTokens` | `max_tokens` | `options.max_tokens` |
| `toolChoice` | `tool_choice` | `MistralOptions.tool_choice`：`auto`/`none`/`any`/`required` 原样；函数对象只保留 `{type: "function", function: {name}}` |
| `promptMode` | `prompt_mode` | `MistralOptions.prompt_mode`（非空才发） |
| `reasoningEffort` | `reasoning_effort` | `MistralOptions.reasoning_effort`（非空才发） |
| `promptCacheKey` | `prompt_cache_key` | `options.session_id`，条件是 `cache_retention != none` 且 session id 非空 |

`on_payload` 返回的新请求体同样经过改名，因此调用方可以按 SDK 写法加入 `topP`、`randomSeed`、`responseFormat.jsonSchema.schemaDefinition`、`presencePenalty`、`frequencyPenalty`、`parallelToolCalls`、`safePrompt` 等字段，线上会变成 `top_p`、`random_seed`、`response_format.json_schema.schema` 等。消息内的 `toolCalls`/`toolCallId` 和内容块的 `imageUrl`、`documentUrl`、`documentName`、`fileId`、`referenceIds`、`inputAudio` 也会改名。

### 消息转换

| 内部消息 | Mistral 消息 |
| --- | --- |
| 第一条 system | `{role: "system", content}`，内容为 `get_system_message_text`（正文加各 section）；为空则不发送 |
| 之后的 system（仅当兼容设置允许时还存在） | `{role: "system", content}`，内容为 `render_system_message_update` |
| user，字符串内容 | `{role: "user", content: "<文本>"}` |
| user，块内容 | 文本块 `{type: "text", text}`；模型支持图片时图片块 `{type: "image_url", imageUrl: "data:<mime>;base64,<data>"}`；全部被过滤且原来有图片时改发占位文本 `(image omitted: model does not support images)` |
| assistant | `{role: "assistant", prefix: false, content?, toolCalls?}`；文本去空白后为空的块跳过；thinking 块变成 `{type: "thinking", thinking: [{type: "text", text}]}`；工具调用变成 `{id, type: "function", function: {name, arguments: <紧凑 JSON 字符串>}, index: 0}`；内容和工具调用都为空时整条不发送 |
| toolResult | `{role: "tool", toolCallId, name, content}`；第一块是文本（见下），模型支持图片时其后附上图片块 |

工具结果文本规则（与上游一致）：文本各块用换行连接并去首尾空白；出错时加前缀 `[tool error] `；有文本时，若含图片而模型不支持图片，追加 `\n[tool image omitted: model does not support images]`；无文本时依次为 `(see attached image)`、`(image omitted: model does not support images)`、`(no tool output)`。

跨模型回放由 `transform_messages` 处理：外来 assistant 消息的 thinking 被丢弃，出错或中止的 assistant 消息不回放，无结果的工具调用补一条 `No result provided` 错误结果，不支持图片的模型把图片换成占位文本。

### 工具调用 id 规范化

Mistral 只接受 9 位字母数字 id。移植上游的规范化函数：

- 去掉所有非 ASCII 字母数字字符；第一次尝试时若正好 9 位则直接使用；
- 否则用 `shortHash`（cyrb53 风格，按 UTF-16 码元计算，base36 输出）对去掉符号后的字符串（为空则用原 id）求哈希，取前 9 个字母数字字符；
- 每个请求一张映射表：同一个源 id 永远映射到同一个结果；结果被另一个源 id 占用时用 `<seed>:<attempt>` 重新哈希。

例：`call_abc123|fc_xyz` → `nrregu2qr`（与 pi、pi-py 相同）。规范化只作用于外来模型的 assistant 消息，工具结果随映射改写。

### 工具定义与严格模式

每个工具发送为 `{type: "function", function: {name, description, parameters, strict}}`。工具带 `constrainedSampling: {type: "json_schema"}` 时，尝试把参数 schema 转成严格子集（所有属性都列入 `required`，原本可选的属性改为 `anyOf: [原 schema, {type: "null"}]`，每个对象加 `additionalProperties: false`）：

- 转换成功：发送转换后的 schema，`strict: true`；
- 转换失败且 `strict: "prefer"`：发送原 schema，`strict: false`；
- 转换失败且 `strict: "require"`：请求前报错 `Tool "<name>" requires JSON-schema constrained sampling, but <原因>.`。

不支持的关键字、联合类型、元组等判断与上游 `makeStrictJsonSchema` 一一对应。上游的 `stripSymbolKeys` 用来去掉 TypeBox 的 symbol 键，Rust 的 `serde_json::Value` 没有这类键，因此不需要。

### 请求头

合并顺序：API 默认头，然后 `options.headers`（名称不区分大小写，`None` 删除默认头）。

| 头 | 值 |
| --- | --- |
| `User-Agent` | `tau/<版本> (<os>; <arch>)` |
| `accept` | `text/event-stream` |
| `authorization` | `Bearer <api_key>` |
| `content-type` | `application/json` |
| `x-affinity` | 会话亲和：启用提示缓存（同 `promptCacheKey` 的条件）且任何头来源都没有提到 `x-affinity`（包括设为 `None` 的删除项）时，填入 `session_id` |

### `stream_simple`

与上游 `streamSimple` 一致：

- `build_base_options`：按推理级别解析采样参数，`max_tokens` 夹到上下文窗口；
- `tool_choice`：`Auto` → `"auto"`，`None` → `"none"`；
- 推理级别先用 `clamp_thinking_level` 夹到模型支持的级别，`off` 视为未开启；
- 模型 `reasoning` 为真且有 `thinkingLevelMap`：发送 `reasoning_effort`，取映射值；映射缺失或为 `null` 时用 `"high"`；未开启推理时取映射里 `off` 的值（例如 `"none"`，为 `null` 则不发送）；
- 模型 `reasoning` 为真但没有映射（Magistral）：开启推理时发送 `prompt_mode: "reasoning"`；
- 非推理模型不发送任何推理控制字段。

## 流事件映射

每个 SSE `data` 去首尾空白；空数据跳过，`[DONE]` 结束读取。数据必须是带 `choices` 数组的 JSON 对象，否则报错 `Invalid Mistral streaming event`。每个合法块先交给 `on_provider_stream_event`，再处理。

| Mistral 块内容 | 处理 |
| --- | --- |
| 顶层 `id` | 第一个非空值写入 `response_id` |
| 顶层 `usage` | `input = prompt_tokens - 缓存`，`cache_read = 缓存`，`output = completion_tokens`，`cache_write = 0`，`total_tokens` 优先取 `total_tokens`，否则求和；随后按模型价格计算费用。缓存 token 依次从 `promptTokensDetails.cachedTokens`、`prompt_tokens_details.cached_tokens`、`promptTokenDetails.cachedTokens`、`prompt_token_details.cached_tokens`、`numCachedTokens`、`num_cached_tokens` 取第一个非空值，并限制在 `[0, prompt_tokens]` |
| `choices[0].finish_reason`（非空） | 写入 `raw_stop_reason`，并映射停止原因（见下表） |
| `delta.content` 为字符串 | 文本增量：`text_start`（需要时）+ `text_delta` |
| `delta.content` 数组中的字符串或 `{type: "text"}` | 同上 |
| `delta.content` 数组中的 `{type: "thinking", thinking: [{text}]}` | 各段文本拼接后作为思考增量：`thinking_start`（需要时）+ `thinking_delta` |
| 空字符串或空文本 | 跳过，不开新块（GLM 模型会在思考和工具调用周围发空增量，开新块会把思考拆成多块，Mistral 回放时拒绝） |
| 未知块类型 | 跳过 |
| `delta.tool_calls[]` | 先结束当前文本/思考块（`text_end`/`thinking_end`）；按 `index`（没有则按 id）找到已有调用或新建一个（`toolcall_start`）；参数为字符串时原样追加，为对象时序列化后追加（`toolcall_delta`） |
| 流结束 | 结束当前文本/思考块，再按出现顺序结束所有工具调用（`toolcall_end`，参数在此完整解析；解析失败写入 `ToolCall::arguments_error`，不是流错误） |

工具调用 id：取 `id`，但空字符串和 Mistral 续传块里的字面量 `"null"` 视为没有 id；没有 id 时用 `derive("toolcall:<index 或 0>")` 生成 9 位 id。

块与块之间文本和思考在同一类型时持续追加，类型切换时先结束前一个块。

### 停止原因

| `finish_reason` | `stop_reason` | `error_message` |
| --- | --- | --- |
| `stop` | `stop` | — |
| `length`、`model_length` | `length` | — |
| `tool_calls` | `toolUse` | — |
| `error` | `error` | `Provider stopped with: error (server error)`（含 "server error"，上层重试逻辑据此认为可重试） |
| 其他 | `error` | `Provider stopped with: <原值>` |

流结束时：若已取消，结果为 `aborted`；若从未收到 `finish_reason`，报错 `Mistral stream ended without a finish reason`；停止原因为 `error` 时以 `error` 事件结束（保留已收到的内容）；其余情况发出 `done`。

## 错误与重试

- 不做客户端重试（上游直接调用 `fetch`，忽略 `maxRetries`）。`max_retries`、`max_retry_delay` 对本 API 无效。
- 超时只覆盖等待响应头：默认 60 秒（上游 `timeoutMs ?? 60000`），由 `options.timeout` 覆盖；超时报错 `Mistral response headers timed out after <毫秒>ms`。响应体读取不设总时限，长时间的思考流不会被截断（上游 #10609）。
- HTTP 错误：先调用 `on_response`，再读响应体，错误信息为 `Mistral API error (<状态码>): <去空白后的原始响应体，最多 4000 字符>`；响应体为空时用状态短语，没有短语时为 `Request failed with status <状态码>`。内部以 `Error::Status` 表示，并记录 `retry-after`。
- 取消：`options.cancel` 在等待响应头、读响应体和读 SSE 时都会生效，流以 `aborted`、`Request was aborted` 结束；消费者丢弃流时生产者停止读取。
- 请求阶段（发出 `start` 之前）的失败只产生一个 `error` 事件。

## Rust 实现差异

- 上游在 API 内部再合并一次 `model.headers`；tau 由 `Models` 在调用 API 前把模型头合并进 `options.headers`（并在其后运行 `transform_headers`），API 只合并默认头和 `options.headers`。`x-affinity` 的“显式指定”判断因此只看 `options.headers`，效果相同。直接调用 `stream` 时需要自己把模型头放进 `options.headers`。
- `User-Agent` 用 tau 自己的值，而不是 `pi (...)`。
- `stream_simple` 在上游缺少 API key 时同步抛错；Rust 返回只含一个 `error` 事件的流，消息相同。
- 上游对每个字符串做 `sanitizeSurrogates`；Rust 字符串保证是合法 UTF-8，不需要。
- 事件不携带共享的 `partial`。工具调用每收到一段参数，就用 `parse_streaming_json` 更新输出消息中的参数（与上游的部分解析一致），结束时再完整解析。空的参数增量不产生 `toolcall_delta` 事件（`MessageBuilder` 的统一行为），上游会发出一个空增量。
- SSE 解析使用共享的 `http::SseDecoder`（处理 `\r\n`、`\r`、多行 `data`、跨块的 UTF-8），上游有自己的分段逻辑；对合法流结果相同。
- 自定义 `fetch`（上游 `options.fetch`）没有移植，`StreamOptions::http_client` 起相同作用。
- JSON 解析失败的错误文本来自 `serde_json`，与 JavaScript 的 `SyntaxError` 文本不同。
- 对上游会因类型错误而崩溃的输入，Rust 采取宽松处理：`tool_choice` 是缺少 `function.name` 的对象时原样发送；`delta` 缺失、`content` 为非字符串非数组、内容块为 `null`、工具调用缺少 `function` 时跳过对应部分。
- 以下两条来自 `docs/research/provider-references.md`，只让客户端更健壮，对合法输入结果不变，已实现：
  - 同一个 `index` 上出现另一个真实 id 时视为新的工具调用（上游会把两个调用的参数拼在一起，得到无效 JSON）；
  - 工具调用名称只在非空时设置：首块名称为空、后续块给出名称时补上。
- 严格 JSON Schema 转换和 `shortHash` 暂时作为私有函数放在本模块中（见“待确认的改进”）。
- 测试中每个用例使用自己的 `reqwest::Client`：共享默认客户端的连接池属于打开连接的那个 Tokio 运行时，`#[tokio::test]` 每个用例一个运行时，复用连接会失败。

## 待确认的改进

1. 按 `options.max_retries` 用 `http::send_with_retry` 的策略重试 408/409/429/5xx（pi-py 已支持 `max_retries`，默认 0）。上游不重试，暂未实现。
2. 流内错误对象（例如 `{"object": "error", "message": ...}`）目前只报 `Invalid Mistral streaming event`，可以把其中的 `message` 带出来。
3. 匿名工具调用（先无 id、后有真实 id）沿用生成的 id；可考虑改用后到的真实 id。
4. 响应体空闲超时：上游把卡住的响应体交给 HTTP 客户端的空闲超时，tau 的默认客户端没有读取超时。
5. 块中的 `model` 字段可写入 `response_model`（上游未写）。
6. 严格 JSON Schema 转换（上游 `constrained-sampling.ts`）和 `shortHash`（上游 `utils/hash.ts`）其他 API 也会用到，建议移到共享模块。
