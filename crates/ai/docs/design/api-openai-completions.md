# openai-completions API

## 目的

`openai-completions` 是 OpenAI Chat Completions 协议的实现，也覆盖所有 OpenAI 兼容端点
（DeepSeek、z.ai、Moonshot、Together、OpenRouter、Vercel AI Gateway、Cloudflare、NVIDIA NIM、
vLLM、llama.cpp、SGLang 等）。服务端之间的差异由模型上的 `OpenAICompletionsCompat` 描述；
未设置的字段按 provider id 和 base URL 自动探测。

模块：`crates/ai/src/api/openai_completions.rs`（单元结构体 `OpenAICompletionsApi`，实现
`ProviderStreams`）。严格 JSON schema 工具和 OpenAI grammar 自定义工具在共享模块
`api/constrained_sampling.rs`，`shortHash` 在 `api::common::short_hash`；为兼容旧路径，本模块重新导出
`openai_completions::constrained_sampling` 和 `openai_completions::short_hash`。选项为
`ApiOptions::OpenAICompletions(OpenAICompletionsOptions)`。

## 上游源文件

- `packages/ai/src/api/openai-completions.ts`（主体：compat 探测、请求参数、消息转换、流处理）
- `packages/ai/src/api/openai-prompt-cache.ts`（`prompt_cache_key` 截断到 64 个字符）
- `packages/ai/src/api/constrained-sampling.ts`（严格 schema 转换、grammar 工具）
- `packages/ai/src/utils/hash.ts`（`shortHash`，用于工具调用 id 归一化）
- 参考测试：`test/openai-completions-*.test.ts`、`constrained-sampling.test.ts`；pi-py 的
  `tests/test_openai_completions.py`。

## Compat 解析

`get_compat(model)` = `detect_compat(model)` 被 `model.compat` 中显式设置的字段逐项覆盖
（`vllmPriority` 只取显式值）。探测规则与上游一致：

| 条件 | 结果 |
| --- | --- |
| 非标准端点：NVIDIA、Cerebras、xAI、Together、chutes.ai、DeepSeek、z.ai、Moonshot、OpenCode、Cloudflare Workers AI / AI Gateway、Ant Ling | 不发 `store`；不用 developer 角色 |
| chutes.ai、DeepSeek、Moonshot、Cloudflare AI Gateway、Together、NVIDIA、Ant Ling、z.ai | 用 `max_tokens`，否则 `max_completion_tokens` |
| xAI、z.ai、Moonshot、Together、AI Gateway、NVIDIA、Ant Ling | 不支持 `reasoning_effort` |
| OpenRouter | `thinkingFormat=openrouter`；默认发会话亲和头（`x-session-id`）；只有 `anthropic/`、`openai/` 模型用 developer 角色 |
| provider 为 `openrouter` 且模型 id 以 `anthropic/` 开头 | `cacheControlFormat=anthropic` |
| DeepSeek / z.ai / Together / Ant Ling | 对应的 `thinkingFormat`；DeepSeek 还要求 assistant 消息带 `reasoning_content` |
| Together、Cloudflare、NVIDIA、Ant Ling | 不支持长缓存保留 |
| 其他 | `thinkingFormat=openai`，`supportsStrictMode=false`（OpenAI 兼容不代表支持严格工具） |

## 请求映射

`POST {baseUrl}/chat/completions`（`baseUrl` 为空时用 `https://api.openai.com/v1`，与 OpenAI SDK 一致）。

请求头按顺序合并（后者覆盖前者，`None` 删除）：`Accept`/`Content-Type: application/json`、
`Authorization: Bearer <key>`、`User-Agent`、`model.headers`、会话亲和头、`options.headers`。
没有 API key 时，若 `authorization` 或 `cf-aig-authorization` 头非空，key 记为 `unused`（与上游
相同，可用 `Authorization: None` 删除默认头）；否则流以 `No API key for provider: <p>` 失败。

会话亲和头只在 `sendSessionAffinityHeaders` 且缓存保留不为 `none` 且有 `session_id` 时发送：
`openrouter` 格式发 `x-session-id`；其他格式发 `x-client-request-id`、`x-session-affinity`，
`openai` 格式另加 `session_id`。

请求体字段（顺序与上游相同）：

| 字段 | 来源 |
| --- | --- |
| `model`、`messages`、`stream: true` | 模型 id、转换后的消息 |
| `prompt_cache_key` | base URL 含 `api.openai.com` 且保留不为 `none`，或长保留且 `supportsLongCacheRetention`；值为截断到 64 字符的 `session_id` |
| `prompt_cache_retention: "24h"` | 长保留且 `supportsLongCacheRetention` |
| `stream_options.include_usage` | `supportsUsageInStreaming` 不为 false |
| `store: false` | `supportsStore` |
| `max_tokens` / `max_completion_tokens` | `options.max_tokens`（大于 0 时），字段名由 `maxTokensField` 决定 |
| `temperature` | `options.temperature` |
| `tools` | 当前工具（见下）；没有工具但历史中有工具调用/结果时发 `tools: []`（LiteLLM 代理 Anthropic 时需要） |
| `tool_stream: true` | 有工具且 `zaiToolStream` |
| `tool_choice` | API 选项中的原始值 |
| `priority` | `vllmPriority` |
| 推理字段 | 见下表 |
| `thinking_token_budget` / `thinking_budget` / `thinking_budget_tokens` | `thinkingTokenBudgetField`（`supportsThinkingTokenBudget` 时默认第一个）且有预算 |
| `provider` | 显式 `openRouterRouting` |
| `providerOptions.gateway` | 显式 `vercelGatewayRouting` 的 `only`/`order` |
| 采样参数 | 最后合并：模型 `samplingParams`、按推理级别的覆盖、请求 `sampling_params`，覆盖同名字段 |

缓存保留：`options.cache_retention`，否则环境变量 `PI_CACHE_RETENTION=long`，否则 `short`。

推理字段（仅 `model.reasoning` 时；`effort` 为请求的推理级别，`map[x]` 为 `thinkingLevelMap`）：

| `thinkingFormat` | 字段 |
| --- | --- |
| `zai` | `thinking: {type: enabled, clear_thinking: false}` 或 `{type: disabled}`；支持时 `reasoning_effort`（未映射用级别名，映射为 null 则不发） |
| `qwen` | `enable_thinking`；支持时 `reasoning_effort = map[effort] ?? effort` |
| `qwen-chat-template` | `chat_template_kwargs: {enable_thinking, preserve_thinking: true}` |
| `chat-template` | `chat_template_kwargs`：字面值原样；`$var` 为 `thinking.enabled`（布尔）、`thinking.effort`（映射值或级别名）、`thinking.budget`（截断后的预算）；`omitWhenOff` 在关闭推理时省略 |
| `baseten` | `chat_template_args`（同上），支持时 `reasoning_effort`（关闭时取 `map.off`） |
| `deepseek` | `thinking: {type: enabled}`，关闭时 `{type: disabled}`（`map.off` 为 null 则不发）；支持时 `reasoning_effort` |
| `openrouter` | `reasoning: {effort}`，关闭时 `{effort: map.off ?? "none"}`（`map.off` 为 null 则不发） |
| `ant-ling` | 只在 `map[effort]` 为字符串时发 `reasoning: {effort}` |
| `together` | `reasoning: {enabled}`；支持时 `reasoning_effort` |
| `string-thinking` | `thinking: map[effort] ?? effort`，关闭时 `map.off ?? "none"` |
| `openai` | 支持时 `reasoning_effort = map[effort] ?? effort`；关闭时只在 `map.off` 是字符串时发送 |

思考预算：只在有推理级别时计算，`xhigh`/`max` 按 `high`，上限为 `max_tokens`（未给出时为模型
`maxTokens`）减 1024，结果为 0 时不发。

### 工具

`convert_tools`：`supportsOpenAIGrammarTools` 且工具声明 grammar 约束时发
`{type: custom, custom: {name, description, format: {type: grammar, grammar: {syntax, definition}}}}`
（lark 优先，其次 regex；没有可用变体或参数 schema 不是“恰好一个必填字符串属性”时报错）。
否则发 function 工具；`supportsStrictMode` 时带 `strict`，`json_schema` 约束可转换时为 `true` 并发送
严格化后的 schema（所有属性必填、可选属性改为 `anyOf [..., null]`、`additionalProperties: false`），
`require` 而无法满足时请求失败。

`supportsMidConvoSystemMessages` 且 `supportsMidConvoToolAdditions`，且工具历史只有新增时，顶层
`tools` 只发初始工具，后续系统消息新增的工具在原位置以 `{role: system, tools: [...]}`（Kimi 风格）
插入；否则 `tools` 为当前完整工具列表。

### 消息转换

先 `resolve_transcript`（不支持中途系统消息时合并为一条），再 `transform_messages`，工具调用 id
归一化：含 `|` 的 Responses 风格 id 拼成 `{call}_{item}`（非 `[a-zA-Z0-9_-]` 按 UTF-16 单元替换为
`_`），超过 40 字符时取前缀加 `shortHash` 前 8 位；provider 为 `openai` 的其他 id 截断到 40 个
UTF-16 单元。

- 系统消息：第一条发完整文本（含 sections），之后的发 `render_system_message_update`；角色在
  推理模型且 `supportsDeveloperRole` 时为 `developer`，否则 `system`。
- 用户消息：字符串原样；块列表去掉空文本，图片为 `image_url` data URL；全空时跳过。
- assistant：文本块（去掉空白文本）拼成字符串 `content`；无内容时为 `null`（
  `requiresAssistantAfterToolResult` 时为 `""`）。思考：`requiresThinkingAsText` 时作为文本 part 放在
  最前；否则若第一个思考块的签名是 `reasoning_content`/`reasoning`/`reasoning_text`，以该字段回放
  （opencode-go 的 `reasoning` 改为 `reasoning_content`）。签名为 `reasoning_details` JSON 时回放
  `reasoning_details`（优先于原始字段）；老记录中工具调用 `thoughtSignature` 里的加密 detail 也会回放。
  `requiresReasoningContentOnAssistantMessages` 时补 `reasoning_content: ""`。grammar 工具调用回放为
  `{type: custom, custom: {name, input}}`。既无内容也无工具调用的消息跳过。
- 工具结果：连续的结果逐条发 `role: tool`（文本为空时用 `(see attached image)` 或
  `(no tool output)`，`requiresToolResultName` 时带 `name`）；模型支持图片时，图片在这批结果之后
  合并为一条用户消息 `Attached image(s) from tool result:`。
- `requiresAssistantAfterToolResult`：工具结果后紧跟用户消息时插入
  `I have processed the tool results.`。
- `cacheControlFormat=anthropic` 且保留不为 `none`：给第一条 system/developer 消息、最后一个工具、
  最后一条带文本的 user/assistant/tool 消息加 `cache_control: {type: ephemeral}`（长保留且支持时
  加 `ttl: "1h"`），字符串内容先转成单个文本 part。

## 流事件映射

SSE 每个 `data:` 是一个 chunk；`[DONE]` 结束读取，空 data 跳过，无法解析的 JSON 使请求失败。
`on_provider_stream_event` 对每个解析出的 chunk 调用（包括 `null`），错误 chunk 除外。

| Chunk 内容 | 处理 |
| --- | --- |
| HTTP 响应被接受 | `on_response`，然后 `start` |
| 非空 `error` | 请求失败：`error.message`（非字符串时 JSON 化，无 message 时整个 error JSON 化）；`error.metadata.raw` 不在消息中时换行追加 |
| 没有 `choices`/`usage`/`id`，但有 `detail`、`message` 或 `object: "error"` | 请求失败，消息为 `detail`/`message` |
| `id` | 第一个非空值成为 `response_id` |
| `model` | 与请求的模型 id 不同且非空时成为 `response_model`（只取第一个） |
| `usage`（或 `choices[0].usage`，Moonshot） | 覆盖 usage：缓存读取取 `prompt_tokens_details.cached_tokens` → `prompt_cache_hit_tokens` → `cached_tokens`；缓存写入取 `prompt_tokens_details.cache_write_tokens`；`input = prompt - 读 - 写`；`reasoning = completion_tokens_details.reasoning_tokens`（缺省 0）；按模型价格计费 |
| `finish_reason` | `raw_stop_reason`；`stop`/`end` → stop，`length` → length，`tool_calls`/`function_call` → toolUse，其他 → error（`Provider finish_reason: <x>`） |
| `delta.reasoning_content` / `reasoning` / `reasoning_text` | 第一个非空字段 → `thinking_start`（签名为字段名）/`thinking_delta` |
| `delta.reasoning_details` | 合法条目保存在内存（相邻 text/summary 合并，加密条目独立）；必要时开始思考块；块结束时 JSON 化写入思考签名 |
| `delta.content` | `text_start` / `text_delta`（思考先于文本处理） |
| `delta.tool_calls[]` | 先按 `index`、再按 `id` 找调用；都没有且只有一个调用时归入该调用，多个时请求失败；找不到则 `toolcall_start`。迟到的 id 补给匿名调用，名称只在非空时设置。`function.arguments` → `toolcall_delta`；`custom.input` 渲染为 `{"<属性>":"<转义输入>"}` 的增量 JSON |
| 流结束 | 按内容顺序 `text_end`/`thinking_end`/`toolcall_end`（参数解析失败记入 `arguments_error`） |

结束规则：取消 → aborted；没有 `finish_reason` 且 `supportsFinishReason=false` → 有工具调用为
toolUse，否则 stop；stop reason 为 error → 以其消息失败；`supportsFinishReason` 而没收到
`finish_reason` → `Stream ended without finish_reason`。工具调用无论 finish reason 是什么都会发出。

## 错误与重试

- 请求通过 `http::send_with_retry`：408/409/429/5xx 与连接错误重试，遵守 `retry-after-ms` /
  `retry-after`，超过 `max_retry_delay` 立即失败；取消随时生效。
- HTTP 错误消息与上游一致：先按 OpenAI SDK 得到 `APIError`（消息 `"<status> <error.message>"`，正文
  没有 `error` 对象时为 `"<status> <文本>"` 或 `"<status> status code (no body)"`），再按
  `formatProviderError`：正文的 `error` 是非空对象且消息里不含它时为 `"<status>: <error JSON>"`，否则为
  SDK 消息。OpenRouter 的 `error.metadata.raw` 不在消息里时另起一行追加。原始正文来自
  `Error::Status.body`，辅助函数在 `api::common`（`openai_sdk_error`、`format_provider_error`）。
- 失败时（包括 HTTP 错误前、流中错误、取消）流以 `error` 事件结束，保留已收到的内容；已收集的
  `reasoning_details` 仍写入思考签名。
- 取消：读取循环结束，已打开的块正常结束，然后以 `Request was aborted`（aborted）失败，与上游
  （SDK 在取消时静默结束迭代）事件顺序一致。

## Rust 特有差异

- 事件通过 `MessageBuilder` 产生：空的 `toolcall_delta`（上游在只带 id/name 的 chunk 上也会发空增量）
  不发出；`thinking_start` 事件携带签名。
- grammar 工具调用的参数由 builder 解析流出的 JSON 得到，结果与上游 `{property: input}` 相同。
  若同一个调用先以 function 增量开始、后改为 custom（上游会丢弃已有的 function 参数缓冲），
  builder 中已累积的 JSON 无法清除，参数可能解析失败。
- HTTP 错误正文先去掉首尾空白并截断到 4000 字符（`Error::Status.body`）：超过 4000 字符的 JSON 正文
  无法解析，按文本处理；`error` 对象由 `serde_json` 重新序列化，浮点数写法可能与 `JSON.stringify`
  不同（如 `1.0`）。连接错误的文本来自 reqwest 而不是 SDK 的 `Connection error.`。
- 默认重试次数由核心决定（`DEFAULT_MAX_RETRIES = 2`），上游 `retryProviderRequest` 默认不重试。
- 不发送 OpenAI SDK 的 `X-Stainless-*` 头；`User-Agent` 为核心的 `delta/<版本> (...)`。
- Rust 字符串不含孤立代理项，`sanitizeSurrogates` 无需移植。
- JavaScript 语义：`trim()` 的空白集合（含 `\u{feff}`、不含 `\u{85}`）、`shortHash` 按 UTF-16 单元、
  `??=`/`||=` 对 `reasoning_details` 字段的赋值（来源缺失时删除字段）都按上游实现。
- 整数值的 `f64`（如 `vllmPriority`、`temperature`）序列化为整数，与 `JSON.stringify` 一致。
- `ChatTemplateKwargValue::Literal` 中的对象原样发送（上游把任何对象都当作 `$var` 引用处理）。
- `ReasoningEffort::None` / `Other(s)` 按字符串处理：作为推理已开启，按名称查 `thinkingLevelMap`，
  不在级别中的名称没有思考预算，采样参数按 `off` 级别解析——与上游传入非标准字符串时相同。
- 未移植：GitHub Copilot 动态头（`x-initiator`、`openai-intent`、视觉请求头）、遥测。

以下行为来自 `docs/research/provider-references.md`，只让客户端对非法输入更宽容或报错更明确，
不改变合法输入的结果：

- 工具调用 `index` 为负数时视为缺失（上游会按 `-1` 合并所有调用）。
- 既无 id 也无 index 的增量归入唯一进行中的调用；有多个调用时报错（上游会新建一个匿名调用）。
- `function.arguments` 不是字符串时按 JSON 文本追加（上游会拼出 `[object Object]`）。
- 没有补全字段的错误包（`{"detail": ...}`、`{"object": "error"}`）直接报错，而不是以
  `Stream ended without finish_reason` 结束。

注意：provider-references 中“同一 index 上出现不同 id 时开始新调用”的规则与上游冲突——上游测试
（Kimi 流中途更换 id）要求按 index 合并，这里遵循上游。

## 待确认的改进

- 流内 `overloaded`/`529` 类错误是否按可重试处理（上游不重试）。
- provider-references 的“同一 index 不同 id 视为新调用”规则是否需要对某些 provider 生效
  （与上游 Kimi 行为冲突，目前未采用）。
