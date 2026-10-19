# API：openai-responses（含 Responses 共享层）

## 目的

实现 OpenAI Responses API（`POST {baseUrl}/responses`，SSE 流式）。模块分两层：

- `api::openai_responses_shared`：Responses 家族共用的输入项转换、工具转换和流事件处理。`openai-responses` 与 `azure-openai-responses` 都用它，以后的 `openai-codex-responses` 也可以复用。
- `api::openai_responses`：单元结构体 `OpenAIResponsesApi`，负责兼容配置、请求体、请求头、缓存与会话亲和、服务等级计价，以及 `stream` / `stream_simple`。

客户端手写在 `reqwest` + `tau_ai::http` 之上，不依赖 OpenAI SDK。

## 上游源文件

- `packages/ai/src/api/openai-responses-shared.ts`：`convertResponsesMessages`、`convertResponsesTools`、`processResponsesStream`、`mapStopReason`。
- `packages/ai/src/api/openai-responses.ts`：`getCompat`、`buildParams`、`createClient`、`stream`、`streamSimple`、服务等级计价。
- `packages/ai/src/api/constrained-sampling.ts`：严格 JSON Schema 与语法（grammar）工具。只移植了本 API 用到的部分，放在共享模块内部。
- `packages/ai/src/api/openai-prompt-cache.ts`：`prompt_cache_key` 截断到 64 个码点。
- `packages/ai/src/utils/hash.ts`：`shortHash`，与 TypeScript 逐位一致（按 UTF-16 码元、`Math.imul` 语义、36 进制输出）。
- 测试参照：`test/openai-responses-*.test.ts`、`test/cache-retention.test.ts`（Responses 部分）、`test/xai-responses.test.ts`（请求部分），以及 pi-py 的 `tests/test_openai_responses.py`。

## 请求映射

### URL 与请求头

- URL：`join_url(model.base_url, "responses")`。
- 请求头合并顺序（后者覆盖前者，名字不区分大小写，`None` 删除）：
  1. API 默认：`Authorization: Bearer <api_key>`（没有 key 但有 `authorization` / `cf-aig-authorization` 头时用 `unused`）、`Accept`、`Content-Type: application/json`、`User-Agent: tau/...`；
  2. `model.headers`；
  3. 会话亲和头（见下）；
  4. `options.headers`。
- 没有 API key、也没有 `authorization` / `cf-aig-authorization` 头时报错 `No API key for provider: <provider>`。
- 会话亲和：只有 `session_id` 存在且缓存保留不是 `none` 时才发送。格式取 `compat.sessionAffinityFormat`，未设置时 provider 为 `openrouter` 或 baseUrl 含 `openrouter.ai` 用 `openrouter`，否则用 `openai`。
  - `openai`：`session_id` 和 `x-client-request-id`；
  - `openai-nosession`：只发 `x-client-request-id`；
  - `openrouter`：`x-session-id`。

### 兼容配置默认值

| 字段 | 默认 |
| --- | --- |
| `supportsDeveloperRole` | true（推理模型的指令用 `developer` 角色，否则 `system`） |
| `supportsMidConvoSystemMessages` | false（为 false 时后续 system 消息折叠进开头的提示词） |
| `sessionAffinityFormat` | 按 provider / baseUrl 推断 |
| `supportsLongCacheRetention` | true |
| `supportsStrictMode` | false |
| `supportsOpenAIGrammarTools` | false |
| `supportsAdditionalTools` / `supportsToolSearch` | false |
| `supportsExplicitPromptCacheMode` | false |
| `supportsMaxOutputTokens` | true |
| `supportsMidConvoEffort` | false |

模型的 `compat` 若不是 Responses 结构（例如自定义 JSON），会被宽松地按 Responses 结构重读，失败时用默认值。

### 请求体字段（按插入顺序）

| 字段 | 规则 |
| --- | --- |
| `model` | `model.id` |
| `input` | 见“输入项转换” |
| `stream` | `true` |
| `prompt_cache_key` | 缓存保留不是 `none` 时取 `session_id`，截断到 64 个码点 |
| `prompt_cache_retention` | 保留为 `long`、`supportsLongCacheRetention` 且不是显式缓存模式时为 `"24h"` |
| `prompt_cache_options` | 仅显式缓存模式：保留 `none` → `{"mode":"explicit"}`；`long` 且支持长缓存 → `{"ttl":"30m"}` |
| `store` | `false` |
| `max_output_tokens` | `max_tokens > 0` 且 `supportsMaxOutputTokens` 时发送，最小 16 |
| `temperature` | 设置时发送 |
| `service_tier` | `OpenAIResponsesOptions::service_tier` |
| `tools` | 请求级工具（见“工具转换”），为空时不发 |
| `tool_choice` | 原样发送 `OpenAIResponsesOptions::tool_choice` |
| `reasoning` / `include` | 见下 |
| 采样参数 | 最后合并 `resolve_sampling_params(model, level, options.sampling_params)`，已有的键原位覆盖 |

缓存保留：`options.cache_retention`，否则 `PI_CACHE_RETENTION=long` 时为 `long`，否则 `short`。

推理参数（仅 `model.reasoning`）：

- 实际 effort = `reasoning_effort`，没有时若设置了非空 `reasoning_summary` 则为 `medium`。
- 有 effort：`reasoning = {effort, summary}`，effort 经 `thinkingLevelMap` 映射（映射值为 `null` 时用原值），summary 默认 `auto`；同时 `include = ["reasoning.encrypted_content"]`。
- 没有 effort：provider 不是 `github-copilot` 且 `thinkingLevelMap.off` 不为 `null` 时发送 `reasoning = {effort: thinkingLevelMap.off ?? "none"}`。
- provider 为 `xai` 时总是 `include = ["reasoning.encrypted_content"]`。
- 对话中途改 effort（`supportsMidConvoEffort` 且请求了 effort）：`reasoning.effort` 固定为 `medium`，真实等级写进 `configuration_update` 输入项（见下），并记入输出消息的 `provider_thinking_level`。这样之前的输入不会变，前缀缓存不失效。

### 输入项转换（`convert_responses_messages`）

1. `resolve_transcript` 按 `supportsMidConvoSystemMessages` 保留或折叠后续 system 消息。
2. `transform_messages` 处理跨模型交接，工具调用 id 规范化规则：
   - 目标 provider 不在允许列表（OpenAI：`openai`、`openai-codex`、`opencode`）或 id 里没有 `|`：把 `[^a-zA-Z0-9_-]` 替换成 `_`（按 UTF-16 码元计），截断 64，去掉结尾 `_`。
   - 否则 id 是 `call_id|item_id`：`call_id` 同上规范化；来自其他 provider 或 API 的 `item_id` 变成 `fc_<shortHash>`，否则规范化；不以 `fc_` 开头的再补上 `fc_`。
3. 逐条消息（`msg_index` 只对非开头 system 消息计数，`continue` 跳过的消息不计数，与上游一致）：
   - 开头 system 消息：`{role: developer|system, content: 提示词全文}`（`include_system_prompt` 为 false 时省略）。
   - 后续 system 消息：先按需输出工具新增项（见下），再输出 `render_system_message_update` 渲染的文本。
   - user：字符串 → 一个 `input_text`；块列表 → `input_text` / `input_image`（`detail: auto`，data URL）；空块列表跳过。
   - assistant：
     - thinking：签名是 Responses 推理项的 JSON，原样解析回输入项（含 `encrypted_content`）；
     - text：从 `textSignature` 解析消息 id 和 phase（`TextSignatureV1` JSON 或旧的纯字符串 id；只保留 `commentary` / `final_answer`）。没有 id 时用 `msg_pi_<i>`，同一条消息的第 N 个文本块用 `msg_pi_<i>_<N>`；超过 64 个字符的 id 变为 `msg_<shortHash>`；
     - toolCall：语法工具发 `custom_tool_call`（`input` 取对应参数），否则 `function_call`（`arguments` 为紧凑 JSON）。同 provider/API 但不同模型时去掉 item id；item id 前缀与类型不符（`fc_` / `ctc_`）也去掉。`namespace` 只回放给同一模型；
     - 输出为空的 assistant 消息跳过；需要时在它前面插入 `configuration_update`（同一模型且记录了 `provider_thinking_level`）。
   - toolResult：`function_call_output` 或 `custom_tool_call_output`，`call_id` 取 `|` 前的部分。输出：无图片或模型不支持图片时为字符串（文本为空白时为 `(no tool output)`，只有图片时为 `(see attached image)`）；否则为 `input_text` + `input_image` 数组。
4. 最后若启用中途 effort，追加 `{type: configuration_update, reasoning: {effort}}`。

工具新增项（`supportsAdditionalTools` 或 `supportsToolSearch`，且工具历史只有新增时）：请求级 `tools` 只放开头 system 消息的工具，后续 system 消息新增的工具在原位置加载：

- `supportsAdditionalTools`：`{type: additional_tools, role: developer, tools}`；
- 否则 `supportsToolSearch`：一对 `tool_search_call` / `tool_search_output`（`execution: client`、`status: completed`），`call_id = pi_tool_load_<shortHash("system:<i>:<names>")>`，输出里的工具带 `defer_loading: true`。

### 工具转换（`convert_responses_tools`）

- 语法工具（`constrainedSampling.type = grammar` 且支持 grammar 工具）：`{type: custom, name, description, format: {type: grammar, syntax: lark|regex, definition}}`。优先 lark。参数 schema 必须是只有一个必填字符串属性的对象，否则报错。
- 其他工具：`{type: function, name, description, parameters, strict?}`。
  - `constrainedSampling.type = json_schema` 且支持严格模式、schema 可转为严格子集时 `strict = true`，`parameters` 换成严格 schema（所有属性必填，原本可选的属性改成 `anyOf [原 schema, null]`，`additionalProperties: false`）。
  - `strict: require` 而无法满足时报错；`prefer` 退回非严格。
  - 支持严格模式时才发送 `strict` 键。

## 流事件映射

SSE 层先模拟 OpenAI SDK：`data: [DONE]` 结束读取；无法解析为 JSON 的数据报错；带非空 `error` 字段的负载以其 `error.message` 报错（不交给回调）。其余负载先交给 `on_provider_stream_event`，再归一化。未知事件类型和条目类型直接跳过。

| Responses 事件 | 处理 |
| --- | --- |
| `response.created` | 记录 `response_id` |
| `response.output_item.added`（reasoning） | 新建 thinking 块，`thinking_start` |
| `response.output_item.added`（message） | 新建 text 块，`text_start`；`phase = final_answer` 时临时把停止原因设为 `stop` |
| `response.output_item.added`（function_call） | 新建工具调用，id 为 `call_id|id`，`toolcall_start` |
| `response.output_item.added`（custom_tool_call） | 新建工具调用，参数为 `{<输入属性>: input}`，`toolcall_start` |
| `response.reasoning_summary_text.delta` / `response.reasoning_text.delta` | `thinking_delta` |
| `response.reasoning_summary_part.done` | `thinking_delta("\n\n")` |
| `response.output_text.delta` / `response.refusal.delta` | `text_delta` |
| `response.function_call_arguments.delta` | 追加参数 JSON，`toolcall_delta` |
| `response.function_call_arguments.done` | 以完整参数替换；若是已收到内容的延续，补发剩余部分为 `toolcall_delta` |
| `response.custom_tool_call_input.delta` / `.done` | 原始输入重新编码为 `{"<属性>":"..."}` 的 JSON 增量，`toolcall_delta`；输入必须单调增长 |
| `response.output_item.done`（reasoning） | 文本取 summary（`\n\n` 连接），否则 content，否则已累积文本；签名为整个推理项的 JSON；`thinking_end` |
| `response.output_item.done`（message） | 文本取 `output_text` / `refusal` 拼接；签名 `{"v":1,"id":...,"phase":...}`；`text_end` |
| `response.output_item.done`（function_call / custom_tool_call） | 用条目的最终参数解析（失败时 `arguments_error`），设置 `namespace`，`toolcall_end` |
| `response.completed` / `response.incomplete` | 终止事件：回填推理项的 `encrypted_content`、`response_id`、用量、费用、停止原因 |
| `response.failed` | 报错 `<code>: <message>`，或 `incomplete: <reason>`，或 `Unknown error (no error details in response)`；记录 `raw_stop_reason` |
| `error` | 报错 `Error Code <code>: <message>` |

槽位按事件的 `output_index` 区分。缺少 `output_index` 的事件共用一个槽位（与上游 `Map` 的 `undefined` 键一致），因此并行工具调用缺少 `output_index` 时会留下未完成的调用并报错，而不是把参数混在一起执行。

### 用量与费用

- `input = input_tokens - cached_tokens - cache_write_tokens`（不小于 0），`cache_read = cached_tokens`，`cache_write = cache_write_tokens`，`output = output_tokens`，`reasoning = reasoning_tokens`（总是设置），`total_tokens` 用服务端的 `total_tokens`（不重新计算）。
- 费用由 `calculate_cost` 计算；`openai-responses` 再按服务等级乘系数：`flex` ×0.5，`priority` / `fast` ×2（`gpt-5.5` 为 ×2.5）。服务等级优先取响应里的 `service_tier`，没有时取请求的。

### 停止原因

| `status` | 结果 |
| --- | --- |
| 缺失、`completed`、`in_progress`、`queued` | `stop` |
| `incomplete` + `max_output_tokens` | `length` |
| `incomplete` + 其他原因 | `error`，`Response incomplete: <reason>` |
| `incomplete` 无原因 | `error`，`Response incomplete without a provider reason` |
| `failed`、`cancelled` | `error`（无消息，最终为 `An unknown error occurred`） |
| 其他 | 报错 `Unhandled stop reason: <status>` |

`raw_stop_reason` 为 `status`，有 incomplete 原因时为 `<status>.<reason>`。结果为 `stop` 且有工具调用时改为 `toolUse`。

流结束时：没有收到终止事件报错 `OpenAI Responses stream ended before a terminal response event`；停止原因为 `toolUse` 但有工具调用没收到 `output_item.done` 时报错 `OpenAI Responses stream completed with an unfinished tool call: <name> (<id>)`。

## 错误与重试

- 发送使用 `http::send_with_retry`：重试 408/409/429/5xx 和连接错误，遵守 `retry-after-ms` / `retry-after` 和 `max_retry_delay`，可被 `options.cancel` 中断。
- HTTP 错误格式为 `<前缀> API error (<status>): <消息>`，前缀在 provider 为 `openai` 时是 `OpenAI`，否则是 provider id。其他错误（流内错误、配置错误）保留原消息。
- `start` 事件在响应被接受、`on_response` 回调之后发出。之后的失败以 `error` 事件结束并带上已收到的内容；取消时停止原因为 `aborted`，消息为 `Request was aborted`；消费方丢弃流时读循环返回 `Error::Aborted` 并停止。

## Rust 特有差异

1. 事件不携带共享的 `partial`。`phase = final_answer` 设置的临时 `stop` 停止原因只体现在最终消息里，消费方的 partial 看不到这个中间状态。
2. `MessageBuilder::end_tool_call` 会用它自己累积的流式 JSON 解析参数。上游用 `output_item.done` 里的完整参数。实现上先设置、调用 `end_tool_call` 后再覆盖一次，所以最终消息与上游一致；但当服务端最终参数不是已流式内容的延续时（例如参数格式错误的用例），`toolcall_end` 事件里的参数是按流式内容解析的。见“需要的核心改动”。
3. `output_item.added` 里已带的函数参数会作为一个 `toolcall_delta` 发出（上游只放进 `partialJson`，不发事件），这样 builder 解析的 JSON 与上游的 `partialJson` 一致。
4. `MessageBuilder` 跳过空的 text/thinking/toolcall 增量，上游会发出空增量事件。
5. 停止原因为 `length` 时，`spawn_stream` 会对仍未完成的工具调用调用 `end_open_tool_calls` 发出 `toolcall_end`；上游不发。
6. HTTP 错误的正文：上游打印 SDK 解析出的 `error` 对象 JSON，这里用核心 `Error::Status` 提取的消息（`error.message` 等字段）。
7. `sanitizeSurrogates` 不需要：Rust 字符串不会含孤立代理项。
8. `stream_simple` 缺少 API key 时上游同步抛错，这里返回以同样消息结束的错误流。
9. 约束采样、`shortHash`、prompt cache key 截断目前作为共享模块的私有/公开辅助函数存在（核心里还没有对应模块）。
10. SSE 数据不是 JSON 时的错误消息格式为 `Could not parse message into JSON: <原因>: <数据>`（SDK 抛出的是 JSON 解析异常本身）。
11. 用量字段接受浮点数并取整，负数按 0 处理。

## 未移植

- Sign in with ChatGPT：上游对 `provider = openai`、`baseUrl = https://api.openai.com/v1` 且 key 不以 `sk-` 开头的请求省略 `prompt_cache_retention`、`prompt_cache_options`、`max_output_tokens`、`temperature`，不发 `configuration_update`，并在 `subscription_sharing_usage_limit_exceeded` 错误后追加 ChatGPT 用量链接。按本次范围未移植，代码中有 `// Not ported` 注释。
- GitHub Copilot 动态请求头（`X-Initiator`、`Copilot-Vision-Request`）。
- OpenAI Codex（ChatGPT 订阅）API `openai-codex-responses`、WebSocket 传输、遥测、`fetch` 选项（由 `StreamOptions::http_client` 替代）。

## 需要的核心改动（未在本次实现）

- `events.rs`：给 `MessageBuilder` 增加 `end_tool_call_with(index: usize, arguments: JsonObject, arguments_error: Option<String>)`（或替换已累积 JSON 的方法），让 `toolcall_end` 事件直接带上 API 给出的最终参数，消除差异 2。
- 新增共享模块（例如 `crates/ai/src/constrained_sampling.rs` 和 `hash.rs`）：`openai-completions` 也需要严格 schema、语法工具和 `shortHash`，目前放在本模块里。

## 待确认的改进

- 未知 `status` 目前按上游报错（`Unhandled stop reason`）。provider-references 建议未知结束原因不应是硬错误，可以改为 `stop` 并保留 `raw_stop_reason`。
- HTTP 错误可以像上游一样带上原始 `error` 对象 JSON，需要核心 `Error::Status` 保留原始正文。
- `function_call_arguments.done` / `custom_tool_call_input.done` 缺少参数字段时目前忽略（上游会抛 TypeError）；可以考虑明确报错。
- 缺少 `item.id` 的工具调用 id 会是 `call_x|undefined`（与上游和 pi-py 一致），可以考虑改为只用 `call_id`。
