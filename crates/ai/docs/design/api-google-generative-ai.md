# Gemini API（`google-generative-ai`）与 Google 共享逻辑（`api::google_shared`）

`google-generative-ai` 对接 Gemini API（`generativelanguage.googleapis.com`）。上游通过 `@google/genai` SDK 发请求；这里没有官方 Rust SDK，按 AGENTS.md 手写 REST 客户端，请求和响应的 JSON 形状对齐该 SDK 实际发送和接收的内容。

`google_shared` 放 Gemini API 与 Vertex AI（`google-vertex`）共用的逻辑。Vertex AI 以后用官方 `google-cloud-aiplatform-v1` crate 和它的 proto 类型，所以共享模块只处理普通数据，不依赖传输方式。

## 上游源文件

- `packages/ai/src/api/google-shared.ts`：消息/工具转换、思考配置、思考签名、结束原因、重试。
- `packages/ai/src/api/google-generative-ai.ts`：`stream`、`streamSimple`、`buildParams`、`getGoogleBudget`。
- `packages/ai/src/api/constrained-sampling.ts`：严格 JSON Schema（`google-shared` 依赖它，Rust 核心里还没有，见“需要的核心改动”）。
- 测试：`test/google-raw-stop-reason.test.ts`、`google-shared-convert-tools.test.ts`、`google-shared-gemini3-unsigned-tool-call.test.ts`、`google-shared-image-tool-result-routing.test.ts`、`google-shared-retry.test.ts`、`google-shared-signed-empty-blocks.test.ts`、`google-thinking-level-map.test.ts`、`google-thinking-signature.test.ts`；另参考 pi-py 的 `test_google_shared.py`、`test_google_streams.py`。

## 模块划分

`google_shared`（与传输无关，Vertex 直接复用）：

- 思考：`resolve_google_thinking_level`、`uses_google_thinking_level`、`get_disabled_google_thinking_config`、`resolve_thinking_config`（上游 `buildParams` 里的思考分支）、`get_google_budget`、`simple_thinking`（上游 `streamSimple` 的思考分支）、`ThinkingConfig`。
- 签名：`is_thinking_part`、`retain_thought_signature`、`is_valid_thought_signature`、`resolve_thought_signature`。
- 模型能力：`gemini_major_version`、`requires_tool_call_id`、`supports_multimodal_function_response`、`supports_google_strict_tool_sampling`、`normalize_tool_call_id`。
- 请求：`convert_messages` → `Vec<Content>`（`Part` 枚举：文本/思考、内联数据、functionCall、functionResponse）；`convert_tools` → `Vec<FunctionDeclaration>`；`resolve_google_function_calling_mode`；`build_request` → `GenerateRequest`（上游 `GenerateContentParameters` 去掉 model）。
- Schema：`sanitize_for_open_api`、`make_strict_json_schema`、`resolve_json_schema_strict_sampling`、`get_json_schema_tool_parameters`。
- 响应：`ResponseChunk`/`Candidate`/`ResponsePart`/`UsageMetadata`（只含流处理用到的字段），`StreamState`（上游流循环的状态机），`check_finished`、`map_stop_reason`、`parse_usage`。
- 重试：`retry_google_request`，给不走 `http::send_with_retry` 的传输（Vertex SDK）用。

`google_generative_ai`（REST 传输）：`GoogleGenerativeAiApi`、`stream`、`stream_simple`、`request_body`（`GenerateRequest` → REST JSON）、`parse_chunk`（REST JSON → `ResponseChunk`）、端点和请求头。

Vertex 模块要做的只有：把 `GenerateRequest` 转成 proto 请求，把 proto 响应转成 `ResponseChunk`（思考签名在 proto 里是字节，转成 base64 文本），然后调用同一个 `StreamState`，结束时用 `check_finished(output, "Google Vertex stream ended without a finish reason")`。

## 请求映射

### 端点与请求头

- `POST {base}/{model}:streamGenerateContent?alt=sse`。
- `base`：模型的 `base_url`（已含 API 版本，与上游 `apiVersion: ""` 一致）；为空时用 `GOOGLE_GEMINI_BASE_URL`（先查 `options.env`，再查进程环境）或 `https://generativelanguage.googleapis.com`，再加 `/v1beta`（SDK 的默认值）。
- 模型路径：id 已以 `models/` 或 `tunedModels/` 开头时原样使用，否则加 `models/`（SDK `tModel`）。id 含 `..`、`?`、`&` 时报错 "invalid model parameter"（SDK 同样拒绝）。
- 请求头（`build_headers` 合并顺序：默认值 → `options.headers`）：`Content-Type: application/json`、`User-Agent: tau/…`、`x-goog-api-key: <key>`。密钥只放请求头，不进 URL。`options.headers` 可以覆盖其中任何一个（与上游显式头覆盖默认 User-Agent 的行为一致）。模型的 `headers` 由 `Models` 合并进 `options.headers`。

### 请求体

| `GenerateRequest` 字段 | REST JSON | 来源 |
| --- | --- | --- |
| `contents` | `contents[]`：`{role: "user"/"model", parts}` | `convert_messages` |
| `system_instruction` | `systemInstruction: {role: "user", parts: [{text}]}`（SDK `tContent` 的形状） | 折叠后首条系统消息的文本（含 sections），为空不发 |
| `temperature` | `generationConfig.temperature` | `options.temperature` |
| `max_output_tokens` | `generationConfig.maxOutputTokens` | `options.max_tokens` |
| `thinking_config` | `generationConfig.thinkingConfig: {includeThoughts, thinkingLevel, thinkingBudget}` | `resolve_thinking_config` |
| `tools` | `tools: [{functionDeclarations: [{name, description, parametersJsonSchema}]}]` | 折叠后的当前工具集 |
| `function_calling_mode` | `toolConfig.functionCallingConfig.mode` | 有工具时由 `resolve_google_function_calling_mode` 决定 |

`generationConfig` 总是存在（可能为空对象），与 SDK 一致。Gemini 不支持对话中途的系统消息：中途系统消息全部折叠进 `systemInstruction`，中途新增的工具并入 `tools`。`options.sampling_params`、`cache_retention`、`session_id`、`metadata` 不适用于本 API，忽略。

### 思考配置

`thinking` 选项来自 `ApiOptions::GoogleGenerativeAi`（其他 API 的选项忽略）：

- `enabled` 且模型 `reasoning`：`{includeThoughts: true}`，再加 `thinkingLevel`（有 level 时）或 `thinkingBudget`（有预算时）。
- 显式关闭且模型 `reasoning`：`get_disabled_google_thinking_config`。不用 `thinkingLevel` 的模型发 `{thinkingBudget: 0}`；用 `thinkingLevel` 的模型把 `off` 按思考级别表钳制，仍是 `off` 时发 `{thinkingBudget: 0}`，否则发钳制后的最低级别（如 Gemini 3.1 Pro 不能关闭思考，发 `LOW`）。
- 未设置 `thinking`：不发 `thinkingConfig`。

`uses_google_thinking_level`：id（小写）匹配 `gemini-3(.N)-pro|flash`、`gemini-4` 及以后、`gemini-flash-latest`、`gemini-flash-lite-latest`、`gemma-4`/`gemma4`。上游用正则；这里没有 regex 依赖，手写了等价的匹配，并用表驱动测试覆盖。

`resolve_google_thinking_level`：取模型思考级别表里该级别的值（不区分大小写），没有映射时用级别名本身；结果必须是 minimal/low/medium/high，否则报错 `Unsupported Google thinking level mapping for <provider>/<id>: <level> -> <映射值|null|undefined>`。

### `stream_simple`

与上游 `streamSimple` 相同：

- `build_base_options`（`max_tokens` 按上下文窗口钳制）。
- `tool_choice`：`Auto`/`None` 映射为 Gemini 的同名选项。
- 没有 `reasoning`，或模型把它钳制为 `off`：`thinking: {enabled: false}`。
- 否则解析成标准级别；用 `thinkingLevel` 的模型发级别，其余模型发预算：自定义 `thinking_budgets` 优先，否则 `2.5-pro` 为 128/2048/8192/32768，`2.5-flash-lite` 为 512/2048/8192/24576，`2.5-flash` 为 128/2048/8192/24576（minimal/low/medium/high），其他模型 `-1`（动态）。
- 级别映射非法时，上游同步抛错；这里返回只有一个 `Error` 事件的流。

### 消息转换（`convert_messages`）

先 `collapse_system_messages` 并去掉首条系统消息，再 `transform_messages`（需要 id 的模型用 `normalize_tool_call_id`：`[a-zA-Z0-9_-]` 以外的字符换成 `_`，截到 64）。规则：

- 用户消息：字符串 → 一个文本 part；块 → 文本 part / `inlineData`；没有 part 时跳过。
- 助手消息：同一 provider 且同一模型（不比较 api，与上游一致）才保留签名和思考。
  - 文本：签名经 `resolve_thought_signature`（同模型且是合法 base64）后作为 `thoughtSignature`；空白文本只有带签名时才保留（Gemini 会给可见文本为空的 part 签名并要求回传）。
  - 思考：同模型时发 `{thought: true, text, thoughtSignature?}`，空白且无签名时丢弃；不同模型时作为普通文本（实际上 `transform_messages` 已经丢掉了外来思考，与上游相同）。
  - 工具调用：`functionCall {name, args, id?}`，需要 id 的模型（`claude-*`、`gpt-oss-*`、Gemini 3+）才带 id；签名规则同上。不会添加 `skip_thought_signature_validator`。
  - 没有 part 的助手消息跳过；`error`/`aborted` 的助手消息由 `transform_messages` 丢弃。
- 工具结果：文本块用 `\n` 拼接，只有空白时视为无文本；模型支持图片时才取图片。`response` 为 `{output: …}`（错误时 `{error: …}`），值是文本，或无文本有图片时 `"(see attached image)"`，否则空串。Gemini 3+（及非 Gemini 模型）把图片放进 `functionResponse.parts`；Gemini 3 以下在其后追加一个 `"Tool result image:"` 加图片的用户回合。连续的工具结果合并进上一个含 `functionResponse` 的用户回合。
- 孤立的工具调用由 `transform_messages` 补上 `"No result provided"` 错误结果。

### 工具与 Schema

- 默认发 `parametersJsonSchema`（完整 JSON Schema，原样发送，包括 `$schema`、`$ref`、`$defs`）。`use_parameters = true`（上游给 Cloud Code Assist 的 Claude 模型用）时发旧的 `parameters`，并递归去掉 `$schema`、`$id`、`$anchor`、`$dynamicAnchor`、`$vocabulary`、`$comment`、`$defs`、`definitions`（保留 `$ref`）。
- 工具声明了 `constrained_sampling: json_schema` 且模型是 Gemini 3+ 时，参数转成严格形式（所有属性必填，可选属性变为可空，`additionalProperties: false`），模式为 `VALIDATED`。不支持严格模式时：`prefer` 退回普通工具，`require` 报错 `Tool "<name>" requires JSON-schema constrained sampling, but …`。
- 函数调用模式：`none`/`any` 原样；否则有严格工具时 `VALIDATED`；否则按 `tool_choice`（未设置则不发 `toolConfig`）。没有工具时不发 `tools` 和 `toolConfig`。

## 流事件映射

每个 SSE `data:` 是一个 `GenerateContentResponse` JSON。空数据和代理注入的 `[DONE]` 跳过；`{"error": …}` 信封立即失败；其余先交给 `on_provider_stream_event`，再由 `parse_chunk` 宽松解析（只看第一个候选；未知字段和未知 part 类型忽略；类型不对的字段视为缺失），然后交给 `StreamState::apply_chunk`。

| Gemini 响应 | 事件 / 输出字段 |
| --- | --- |
| HTTP 响应被接受 | `on_response`，然后 `Start` |
| `responseId` | `response_id`，保留第一个非空值 |
| part `text`，`thought: true` | 当前不是思考块时结束当前块并 `ThinkingStart`；`ThinkingDelta`（空串不发事件） |
| part `text`，非思考 | 当前不是文本块时结束当前块并 `TextStart`；`TextDelta` |
| part `thoughtSignature`（文本/思考 part） | 当前块的 `thinking_signature`/`text_signature`，保留最后一个非空值，随 `ThinkingEnd`/`TextEnd` 发出 |
| part `functionCall` | 结束当前块；`ToolCallStart`（参数为 `args`）、`ToolCallDelta`（`args` 的紧凑 JSON）、`ToolCallEnd`；非空 `thoughtSignature` 记到工具调用上 |
| `functionCall.id` 缺失、为空或与本消息已有调用重复 | 生成 `<name>_<毫秒时间戳>_<进程内递增计数>`，跨 chunk 唯一 |
| `finishReason` | `raw_stop_reason` 原样保存；`STOP` → `stop`，`MAX_TOKENS` → `length`，其他（含未知值）→ `error`；已有工具调用且为 `stop` 时改为 `toolUse` |
| `usageMetadata`（含 `promptTokenCount` 时） | 见下 |
| 流结束 | 结束当前块；没有结束原因 → 错误 "Google stream ended without a finish reason"；结束原因映射为 `error` → 错误 "Provider stopped with: <raw>"；否则 `Done` |

用量：`input = promptTokenCount - cachedContentTokenCount`（饱和减法），`output = candidatesTokenCount + thoughtsTokenCount`，`cache_read = cachedContentTokenCount`，`cache_write = 0`，`reasoning = thoughtsTokenCount`，`total_tokens = totalTokenCount`（与上游一样取服务器的值，不重算），费用用 `calculate_cost`。每个带用量的 chunk 覆盖前值，但没有 `promptTokenCount` 的用量不覆盖（只有部分字段的早期报告不会替换完整的报告）。计数接受数字或数字字符串。

## 错误与重试

- 缺少密钥：`No API key for provider: <provider>`。`require_auth` 也接受 `options.headers` 里的 `x-goog-api-key`。
- HTTP 错误：`http::send_with_retry`，消息 `"<status> <error.message>"`（非 JSON 正文取文本），见 `http.md`。可重试：408、409、429、5xx、连接错误，`retry-after-ms`/`retry-after` 生效，超过 `max_retry_delay` 立即失败。
- 流内错误信封：`got status: <error.status>. <整个 chunk 的 JSON>`（SDK 的措辞），不重试，已收到的内容保留在错误消息里。
- 无法解析的 chunk：`Failed to parse Google stream chunk: <原因>`。
- 取消：请求前、重试等待中、读流中都响应 `options.cancel`，以 `aborted` 结束，保留部分内容；消费者丢弃流时停止读取。
- `on_payload`、`on_response`、`on_provider_stream_event` 返回错误时请求失败。
- `retry_google_request`（给 Vertex 用）：对 `Error::Status` 的 408/409/429/5xx 重试，用 `retry_after_ms` 或指数退避，其他错误（传输错误、流内错误）不重试；次数默认值与核心一致。

## Rust 特有差异

- `on_payload` 拿到的是 REST 请求体（`contents`、`systemInstruction`、`tools`、`toolConfig`、`generationConfig`），而不是 SDK 的 `{model, contents, config}`；修改要按 REST 字段名写（如 `generationConfig.topK`）。返回值原样作为请求体发送。
- `on_provider_stream_event` 拿到的是 chunk 的原始 JSON，而不是 SDK 对象（字段名相同）。
- HTTP 错误消息用核心的 `"<status> <消息>"` 格式；上游是 SDK 拼出的整段 JSON 正文。溢出检测等按消息内容匹配的逻辑不受影响。
- 重试走核心策略：`max_retries` 默认 2（上游 Google 适配器默认 0），会使用 `retry-after` 头（SDK 的 `ApiError` 没有头，上游从不使用），连接错误也重试（上游不重试）。
- 预先取消时消息是 "Request was aborted"（上游 `buildParams` 里是 "Request aborted"）。
- 结束原因：REST 拿到的是字符串，用上游的 `mapStopReasonString` 语义，未知值映射为 `error`，消息 "Provider stopped with: <raw>"；上游 SDK 路径对未知枚举值抛 "Unhandled stop reason: <raw>"。两者结果都是 `error`。
- 流内错误信封：SDK 只在原始 chunk 本身是 JSON 且 `code` 在 400–599 时抛错；`alt=sse` 下的信封实际被当作空响应忽略，最后报 "stream ended without a finish reason"。这里任何非空 `error` 都立即失败（与 pi-py 一致）。
- `sanitizeSurrogates` 不需要：Rust 字符串不可能含孤立代理项。
- 思考签名按收到的 base64 文本保存和回传（与 JS SDK 一致；pi-py 的 SDK 会转成 URL-safe base64）。
- 事件不带共享的 `partial`；`ToolCallStart` 不带签名，签名在 `ToolCallEnd` 的完整调用里。空文本 part 不发 `TextDelta`（`MessageBuilder` 跳过空增量），但仍会开块并记录签名。
- 用量只在含 `promptTokenCount` 时更新，减法饱和（provider-references 的要求；对正常流结果不变）。
- 跳过 `[DONE]`、宽松解析 part（结构体加可选字段，而不是标签枚举）、未知 part 类型忽略（provider-references 的要求）。
- 工具调用 id 计数器是进程级 `AtomicU64`，上游是模块级计数器。
- 严格 JSON Schema（上游 `constrained-sampling.ts`）暂时实现在 `google_shared` 里（核心没有），见下。

## 未移植

- 上游拒绝自定义 `fetch`：Rust 用 `options.http_client`，没有对应检查。
- SDK 自带的 `x-goog-api-client` 头和 User-Agent 前缀（`google-genai-sdk/…`）：手写客户端不是 SDK，不冒充。
- 遥测：上游 Google 适配器没有。

## provider-references 检查

- `x-goog-api-key` 头，密钥不进 URL：是。
- 出现 `functionCall` 时 `STOP` 改为 `toolUse`：是（`MAX_TOKENS` 保持 `length`）。
- 未知结束原因不导致崩溃：映射为 `error` 并保留原值。
- 空 `thoughtSignature` 视为无；非思考文本 part 上的签名保留：是。
- 思考 part 作为思考块，不并入文本：是。
- 无 id 的调用生成跨 chunk 唯一的 id：是（另外重复 id 也重新生成，与上游相同）。
- 用量等待 `promptTokenCount`、饱和减法、输出含 `thoughtsTokenCount`：是。
- 流内 `{"error": …}` 信封：是。
- part 宽松解析、未知类型不致命、`[DONE]` 跳过、无结束事件时报错：是。
- `parametersJsonSchema`：是。`$defs`/`definitions` 引用内联：未实现，见下。

## 需要的核心改动（未做，留给维护者）

- 把 `make_strict_json_schema`、`resolve_json_schema_strict_sampling`、`get_json_schema_tool_parameters`（上游 `constrained-sampling.ts`）移到核心的共享模块（如 `crates/ai/src/constrained_sampling.rs`），OpenAI、Anthropic、Bedrock 也要用；目前在 `google_shared` 里。
- 可选：`http` 暴露一个通用的 `retry(options, f)`（或公开 `backoff`），`retry_google_request` 就不必复制退避计算。
- 可选：`options::GoogleThinkingLevel` 加 `as_str()` 返回线上枚举名；目前用 `google_shared::google_thinking_level_name`。

## 待确认的改进

- 按 provider-references 内联 `$defs`/`definitions` 引用（递归引用变 `{}`，无法解析的引用保持原样）。上游原样发送 `parametersJsonSchema`，所以没有实现。
- 用响应的 `modelVersion` 填 `response_model`（上游不填）。
- HTTP 错误消息是否改成上游的 JSON 正文格式，以便与 pi 的消息逐字一致。
- 流内 `UNAVAILABLE`/`RESOURCE_EXHAUSTED` 错误是否像 Anthropic 的 `overloaded_error` 一样可重试（目前与上游一致，不重试）。
- `toolUsePromptTokenCount` 没有计入 `input`（`total_tokens` 取服务器值，所以可能与各分量之和不等）。
- Gemini 3 以下的“工具结果图片”回合会打断函数响应的合并，后面的工具结果（包括合成的孤立结果）落到新的用户回合，与上游相同；可以考虑把图片回合放到所有函数响应之后。
