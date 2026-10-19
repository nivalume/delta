# API 设计：`anthropic-messages`

## 目的

`tau_ai::api::anthropic_messages` 实现 Anthropic Messages 协议（流式）。凡是 `model.api == "anthropic-messages"` 的模型都走这里：Anthropic 官方端点、OAuth（Claude Code 订阅）、GitHub Copilot 的 Claude 端点，以及 OpenRouter、Vercel AI Gateway、Cloudflare AI Gateway、Fireworks、Kimi 等兼容端点。

模块对外提供：

- `AnthropicMessagesApi`：单元结构体，实现 `ProviderStreams`，直接委托给下面两个函数。
- `stream(model, context, options)`：API 专属选项取自 `options.api` 中的 `ApiOptions::AnthropicMessages(AnthropicOptions)`，其他变体忽略。
- `stream_simple(model, context, options)`：把 `reasoning` / `thinking_budgets` / `tool_choice` 映射为 `AnthropicOptions`，再调用 `stream`。
- 几个纯函数与常量：`normalize_tool_call_id`、`to_claude_code_name`、`from_claude_code_name`、`is_oauth_token`、`CLAUDE_CODE_VERSION`、`CLAUDE_CODE_SYSTEM_PROMPT`、`CLAUDE_CODE_TOOLS`。

上游通过 Anthropic SDK 发请求（`client.beta.messages.create(params).asResponse()`），然后自己解析 SSE 响应体。按 AGENTS.md 的规则（Anthropic 没有官方 Rust SDK，不引入第三方 SDK），这里用 `reqwest` + `tau_ai::http` 手写 SDK 在线上发出的同一个请求。

## 上游源文件

- `packages/ai/src/api/anthropic-messages.ts`（主体：选项、请求构建、SSE 解析、事件映射、`streamSimple`）
- `packages/ai/src/api/constrained-sampling.ts`（严格 JSON schema 转换，`resolveJsonSchemaStrictSampling`；Rust 里是共享模块 `api::constrained_sampling`）
- `packages/ai/src/api/simple-options.ts`（`buildBaseOptions`、`adjustMaxTokensForThinking`、`clampMaxTokensToContext`，已在 `simple_options.rs`）
- `packages/ai/src/api/transform-messages.ts`（`transformMessages`、`splitErrorToolResultImages`，已在 `transform.rs`）
- `packages/ai/src/utils/provider-retry.ts`（重试策略，已在 `http.rs`）
- `packages/ai/src/utils/transcript.ts`、`utils/text.ts`（系统消息回放，已在 `transcript.rs`）
- 参考：pi-py `pi_ai/api/anthropic_messages.py` 及 `tests/test_anthropic_*.py`

移植的测试用例来自上游 `test/anthropic-*.test.ts`、`cache-retention.test.ts`、`interleaved-thinking.test.ts`、`max-thinking.test.ts`（其中需要真实网络的 e2e 用例只移植了可离线验证的断言），以及 pi-py 的三个测试文件。

## 请求映射

### 端点与认证

- `POST {baseUrl}/v1/messages?beta=true`。`?beta=true` 是 SDK 的 `beta.messages` 路径固有的查询参数，pi 和 pi-py 发出的请求都带它。`baseUrl` 为空字符串时用 `https://api.anthropic.com`（与 JS SDK 的 `baseURL || 默认值` 一致）。
- 认证检查与上游 `assertRequestAuth` 相同：有非空 `api_key`，或 `options.headers` 里有非空白的 `authorization` / `x-api-key` / `cf-aig-authorization`，否则以 `No API key for provider: <provider>` 失败（流里只有一个 `error` 事件）。
- 三种认证模式：
  - `provider == "github-copilot"`：`Authorization: Bearer <key>`，不做 OAuth 改写。
  - `api_key` 包含 `sk-ant-oat`：OAuth（Claude Code）模式，`Authorization: Bearer <token>`，附加 `user-agent: claude-cli/2.1.280`、`x-app: cli`，betas 加 `claude-code-20250219`、`oauth-2025-04-20`，系统提示前置 `You are Claude Code, Anthropic's official CLI for Claude.`，工具名按 Claude Code 规范大小写改写，响应中的工具名再映射回调用方的名字。
  - 其他：有 key 时发 `X-Api-Key`；没有 key 时由请求头自带认证（如 `Authorization: Bearer <ANTHROPIC_AUTH_TOKEN>`、Cloudflare 的 `cf-aig-authorization`）。

### 请求头

合并顺序与 SDK 一致，后者覆盖前者，名字大小写不敏感，`None` 删除前面同名头：

1. SDK 默认：`Accept: application/json`、`anthropic-version: 2023-06-01`
2. 认证头（`X-Api-Key` 或 `Authorization`）
3. 客户端默认头（上游 `mergeClientHeaders`）：`User-Agent: tau/...`、`accept`、`anthropic-dangerous-direct-browser-access: true`；OAuth 模式的 `user-agent`/`x-app`；API key 模式的会话亲和头；`model.headers`；`options.headers`
4. `Content-Type: application/json`
5. 由请求体字段生成的头：`betas` → `anthropic-beta`（逗号连接），`user_profile_id` → `anthropic-user-profile-id`。这两个字段在 `on_payload` 之后从请求体里取出，与 SDK 的行为相同。

会话亲和：仅在 API key 模式、缓存保留不是 `none`、有 `session_id` 且 `compat.sendSessionAffinityHeaders`（默认：OpenRouter 为真）时发送；`sessionAffinityFormat == openrouter`（OpenRouter 默认）用 `x-session-id`，否则 `x-session-affinity`。

### Betas（`getBetaFeatures`）

`model.headers` 或 `options.headers` 中出现 `anthropic-beta` 时，它的值（去空白、去重）替换整个列表，`None` 表示不发任何 beta。否则按顺序收集：

| 条件 | beta |
| --- | --- |
| OAuth | `claude-code-20250219`、`oauth-2025-04-20` |
| 有工具且 `supportsEagerToolInputStreaming == false` | `fine-grained-tool-streaming-2025-05-14` |
| `model.reasoning`、`thinking_enabled == true`、`interleaved_thinking != false`、非 `forceAdaptiveThinking` | `interleaved-thinking-2025-05-14` |
| `allowedFallbackModels` 非空 | `server-side-fallback-2026-07-01` |
| `supportsMidConvoEffort` | `mid-conversation-output-config-2026-07-01`、`thinking-binding-controls-2026-08-01` |
| 原生工具变更 | `inline-tools-2026-09-15` |

### 请求体

字段顺序与上游相同：`model`、`messages`、`max_tokens`（`options.max_tokens`，否则 `model.max_tokens`）、`stream: true`、`betas`、`system`、`temperature`、`tools`、`thinking`、`output_config`、`metadata`、`tool_choice`、`fallbacks`。`on_payload` 返回的替换体会被强制设回 `stream: true`。

- **转录**：`resolve_transcript` 按 `compat.supportsMidConvoSystemMessages` 保留或折叠中途系统消息；`transform_messages` 处理跨模型回放（丢弃外来 thinking 与文本签名、补 `No result provided` 孤立工具结果、非视觉模型图片占位），工具调用 id 用 `normalize_tool_call_id` 规整为 `^[a-zA-Z0-9_-]{1,64}$`（非法字符按 UTF-16 单元逐个替换为 `_`，与上游正则一致，再截到 64 个字符）。
- **系统提示**：首条系统消息的完整文本（内容 + sections）作为 `system` 文本块，带 `cache_control`。
- **用户消息**：纯文本空白则跳过；块列表中空白文本块被过滤，全空则跳过；图片转 `{"type":"image","source":{"type":"base64",...}}`。
- **助手消息**：空白文本跳过；`redacted` thinking 回放为 `redacted_thinking`；有签名的 thinking 原样回放（thinking 文本可为空）；无签名的 thinking 变为文本块，`compat.allowEmptySignature` 时保留为 `signature: ""` 的 thinking 块；工具调用为 `tool_use`。
- **工具结果**：连续的工具结果合为一个 user 消息。纯文本时 `content` 是 `\n` 连接的字符串，含图片时是块列表（只有图片时前置 `(see attached image)`）。错误结果中的图片用 `transform::split_error_tool_result_images` 拆出，放在本轮所有 `tool_result` 之后（Anthropic 拒绝含图片的错误结果）。
- **中途系统消息**（模型支持时）：渲染为 `role: "system"` 消息（`render_system_message_update`），并推迟到下一条助手消息之前（或末尾）发送，因为 `tool_result` 必须紧跟 `tool_use`。
- **原生工具变更**：`supportsMidConvoSystemMessages && supportsMidConvoToolChanges` 且首条系统消息有工具时启用。顶层 `tools` 固定为初始工具（最后一个带缓存断点）加 `__pi_deferred_placeholder__`（`defer_loading: true`）；之后的变更以 `tool_removal` / `tool_addition` 块放在系统消息里，同名重定义只发 addition。否则顶层 `tools` 是当前工具全集。
- **工具定义**：`name`（OAuth 时改写）、`description`、`eager_input_streaming: true`（`supportsEagerToolInputStreaming`，默认真）、`strict: true`（严格模式时）、`input_schema`、最后一个工具的 `cache_control`（`supportsCacheControlOnTools`，默认真）。非严格时 `input_schema` 只保留 `type/properties/required`；严格时为严格化后的完整 schema 再覆盖这三个键。
- **严格工具**：`constrainedSampling: {type: json_schema}` 的工具在 `compat.supportsStrictTools` 时尝试严格化（调用共享模块 `api::constrained_sampling` 的 `resolve_json_schema_strict_sampling`，传入 Anthropic 的关键字检查 `is_strict_unsupported_keyword`：`minimum`、`maxItems`、非 0/1 的 `minItems`、不在白名单内的 `format` 等）。`prefer` 无法严格化时退回非严格，`require` 则请求失败。
- **缓存**：保留策略为 `options.cache_retention`，否则 `PI_CACHE_RETENTION=long`（先查 `options.env`，再查进程环境）为 long，否则 short。`none` 不发 `cache_control`；long 且 `supportsLongCacheRetention`（默认真）时为 `{"type":"ephemeral","ttl":"1h"}`。断点位置：系统块、最后一个工具、最后一条 user/system 消息的最后一个 text/image/tool_result/tool_addition/tool_removal 块（字符串内容改写为带 `cache_control` 的文本块）。
- **thinking**：
  - `model.reasoning`、`thinking_enabled == false`、且 `thinkingLevelMap.off` 不是 `null`：`{"type":"disabled"}`。
  - `supportsMidConvoEffort`（且未禁用）：`{"type":"adaptive","display":...,"block_binding":{"prefix_mismatch_behavior":"drop_block"}}`，`output_config: {"effort":"high"}`；消息中在每条带已知 effort（同 provider、`providerThinkingLevel` 为合法 effort）的回放助手消息前插入 `{"role":"system","content":[],"output_config":{"effort":...}}`，末尾插入本轮 effort（默认 high），并把本轮 effort 写入输出的 `provider_thinking_level`。
  - `thinking_enabled == true`：`forceAdaptiveThinking` 时 `{"type":"adaptive","display":...}`，有 effort 时加 `output_config`；否则 `{"type":"enabled","budget_tokens": 预算或 1024,"display":...}`。`display` 默认 `summarized`。
- **temperature**：仅在未开启 thinking、非 `supportsMidConvoEffort`、`supportsTemperature`（默认真）时发送。`sampling_params` 不用于本 API（与上游一致）。
- **metadata**：只取字符串 `user_id`。
- **tool_choice**：`auto`/`any`/`none` → `{"type":...}`，指定工具 → `{"type":"tool","name":...}`。
- **fallbacks**：`allowedFallbackModels` 非空时发 `[{"model":...}]`。

### `stream_simple`

与上游 `streamSimple` 相同：

- 用 `build_base_options` 得到基础选项（按上下文窗口收紧 `max_tokens`），`tool_choice` 映射为 `AnthropicToolChoice`。
- 无 `reasoning`：`thinking_enabled = false`。
- `forceAdaptiveThinking`：`thinking_enabled = true`，effort 取 `thinkingLevelMap` 中的字符串，否则 minimal/low → low、medium → medium、其余 → high。
- 其他：`adjust_max_tokens_for_thinking` 求出上限与预算，再 `clamp_max_tokens_to_context`，预算取 `min(预算, max(0, max_tokens - 1024))`。

## 流事件映射

只处理 SSE 事件名为 `message_start`、`message_delta`、`message_stop`、`content_block_start`、`content_block_delta`、`content_block_stop` 的事件（数据用 `parse_json_with_repair` 解析）；`ping`、`[DONE]`、代理扩展事件被跳过；`event: error` 直接以原始 data 作为错误消息结束流。每个已解析事件先交给 `on_provider_stream_event`。HTTP 响应被接受、`on_response` 调用之后才发 `start`。

| Anthropic 事件 | 处理 | tau 事件 |
| --- | --- | --- |
| `message_start` | `response_id`、`input_transformations`；`message.model` 与请求模型不同则记 `response_model`，若它在 `allowedFallbackModels` 中则改用其价格；初始 usage（`cacheWrite1h` 取 `cache_creation.ephemeral_1h_input_tokens`，缺省 0），重算总量与费用 | 无 |
| `content_block_start` `text` | 新文本块；`content_block.text` 非空时作为第一个增量 | `text_start`（+ `text_delta`） |
| `content_block_start` `thinking` | 新 thinking 块，初始 thinking 与签名（缺省为空串）随开始事件给出 | `thinking_start` |
| `content_block_start` `redacted_thinking` | thinking 块 `[Reasoning redacted]`，`redacted: true`，签名为 `data` | `thinking_start` |
| `content_block_start` `tool_use` | 工具调用（OAuth 时名字映射回调用方工具名），初始参数为 `input` | `toolcall_start` |
| `content_block_start` `fallback` | 已有输出时报错 `Anthropic performed an unsupported mid-output model fallback`，否则忽略 | 无 |
| `content_block_delta` `text_delta` | 追加文本 | `text_delta` |
| `content_block_delta` `thinking_delta` | 追加 thinking | `thinking_delta` |
| `content_block_delta` `input_json_delta` | 追加参数 JSON，并用 `parse_streaming_json` 更新部分参数 | `toolcall_delta` |
| `content_block_delta` `signature_delta` | 追加 thinking 签名 | 无 |
| `content_block_stop` | 结束对应块；工具调用参数由完整 JSON 解析（无 JSON 时为 `{}`，非对象或无法解析时记 `arguments_error`） | `text_end` / `thinking_end` / `toolcall_end` |
| `message_delta` | `input_transformations`；`stop_reason` → `raw_stop_reason` 与停止原因；usage 逐字段覆盖（缺失字段保留 `message_start` 的值），含 `cache_creation.ephemeral_1h_input_tokens` 与 `output_tokens_details.thinking_tokens`（→ `reasoning`）；重算总量与费用 | 无 |
| `message_stop` | 标记流正常结束 | 无 |
| 其他事件/块/增量类型 | 跳过 | 无 |

块按 Anthropic 的 `index` 定位：每个 index 维护一个未结束块的 FIFO 队列，等价于上游 `findIndex(b => b.index === event.index)`（结束时删除 index）。

停止原因：`end_turn`、`pause_turn`、`stop_sequence` → `stop`；`max_tokens` → `length`；`tool_use` → `toolUse`；`refusal` → `error`（消息为 `stop_details.explanation`，否则 `The model refused to complete the request`）；`sensitive` → `error`（`Provider stopped with: sensitive`）；其他 → `Unhandled stop reason: <值>` 错误（`raw_stop_reason` 已记录）。

流结束后依次检查：见过 `message_start` 但没有 `message_stop` → `Anthropic stream ended before message_stop`；已取消 → aborted；没有停止原因 → `Anthropic stream ended without a stop reason`；停止原因为 error → 以其消息失败。成功时若有 `input_transformations`（取最后一次出现的列表），追加 `anthropic_input_transformations` 诊断，`details.transformations` 每项只保留非空的 `type`/`path`/`reason`。

## 错误与重试

- 请求通过 `http::send_with_retry` 发送。上游的 `retryProviderRequest` 在未指定 `maxRetries` 时**不重试**，所以这里把 `options.max_retries` 的默认值设为 0（核心默认是 2）。指定后按 SDK 策略重试 408/409/429/5xx/连接错误，遵守 `x-should-retry`、`retry-after-ms`、`retry-after` 与 `max_retry_delay`。
- 严格工具降级：请求被拒（HTTP 400）且消息包含 `The compiled grammar is too large`、`Too many strict tools (` 或 `Schemas contains too many parameters with union types (` 之一，且本次请求里有被严格化的 `prefer` 工具、尚未降级、未取消时，追加 `anthropic_strict_tools_fallback` 诊断（`details.reason = "strict_tool_limits_exceeded"`），把 `prefer` 工具改为非严格后重建请求再发一次（重新调用 `on_payload`）。`require` 工具保持严格。只降级一次，第二次被拒直接报错；已开始的流（包括流内 `event: error`）不降级。
- 取消：每次构建请求前、等待响应与重试退避期间、读取每个 SSE 事件时检查 `options.cancel`，以 `Request was aborted` 结束为 aborted，并保留已收到的部分内容。消费者丢弃流（`builder.is_closed()`）时同样停止。
- `on_payload` / `on_response` / `on_provider_stream_event` 回调失败时以其错误消息结束流。

## Rust 实现上的差异

- **事件形状**：tau 的事件不携带共享 `partial`。`content_block_start` 自带非空文本时，额外发一个 `text_delta`，使消费端归约出的部分消息包含这段文本；thinking 的初始文本和签名随 `thinking_start` 给出。上游只发 `text_start`/`thinking_start`。
- **未结束的工具调用**：成功结束时 `spawn_stream` 会为仍未收到 `content_block_stop` 的工具调用补发 `toolcall_end`（共享契约），上游不补发。
- **HTTP 错误消息**：与 SDK 的 `APIError.makeMessage` 一样由 `Error::Status` 的原始响应体拼出：JSON 正文是 `"<status> <整个 JSON body>"`（如 `400 {"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long"}}`），否则是 `"<status> <文本>"`，空正文是 `"<status> status code (no body)"`。差异：正文先去掉首尾空白并截断到 4000 字符（`Error::Status.body`），超过 4000 字符的 JSON 正文无法解析，按文本处理；JSON 由 `serde_json` 重新序列化，浮点数写法可能与 `JSON.stringify` 不同（如 `1.0`）。连接错误、超时的文字是 reqwest 的而不是 SDK 的 `Connection error.` / `Request timed out.`。
- **严格工具降级的判断**：与上游一样要求状态码 400、错误体的 `error.type == "invalid_request_error"`，且消息包含上述限制文本之一。
- **SSE 解析**：使用共享的 `http::SseDecoder`。只有 `event:` 没有 `data:` 的事件会被它丢弃（上游会以空 data 解析失败）；解析失败消息中的 `raw=` 由事件名和 data 行重建，不含注释/`id:` 行；JSON 错误文字是 serde 的，不是 JS 的 `SyntaxError`。解析结果不是对象时同样报 `Could not parse Anthropic SSE event ...`。
- **字符串清理**：上游对发送文本调用 `sanitizeSurrogates` 去除孤立代理项。Rust `String` 不可能含孤立代理项，所以无需处理。
- **用量字段**：`message_start` 缺少 `usage` 时按 0 处理（上游会抛 `TypeError`）。token 数为浮点时截断为整数。
- **自定义客户端**：上游的 `options.client`（注入 SDK 客户端，如 `AnthropicVertex`）没有对应物；可通过 `StreamOptions::http_client` 注入 `reqwest::Client`。
- **SDK 遥测头**：不发送 `X-Stainless-*`（SDK 自身的遥测，非协议所需）。

## 未移植

- **工作负载身份联合**（`ANTHROPIC_FEDERATION_RULE_ID`、`ANTHROPIC_ORGANIZATION_ID`、`ANTHROPIC_IDENTITY_TOKEN_FILE`、`ANTHROPIC_WORKSPACE_ID`、`ANTHROPIC_SERVICE_ACCOUNT_ID`）：上游把 OIDC 令牌交换与缓存交给 SDK。本次不在范围内，没有 key/认证头时一律报 `No API key for provider: ...`。需要时可在本模块内实现 `POST /v1/oauth/token`（jwt-bearer）交换、按配置缓存令牌、401 时作废重换，并让 `stream_simple`/`stream` 的认证检查接受该配置。
- **GitHub Copilot 动态头**（`X-Initiator`、`Openai-Intent`、视觉输入标记）：只保留 Copilot 的 Bearer 认证，代码中有 `// Not ported` 注释。
- 遥测、OAuth 登录流程、延迟响应、WebSocket 传输：按任务说明不在范围内。

## 待确认的改进

- 流内 `overloaded_error`（`event: error`）目前与上游一样直接失败。provider-references.md 建议把它当作可重试错误，但流已开始、事件已发出，重试会改变事件序列，需要确认是否以及如何做（例如仅在尚无内容块时重试）。
- 上游在 `message_start` 缺少 `usage` 时抛出异常，这里按 0 处理；如需严格一致可改回报错。
- `baseUrl` 为空时 pi-py 会读 `ANTHROPIC_BASE_URL`，上游 JS 直接用默认地址；这里跟随上游 JS。
