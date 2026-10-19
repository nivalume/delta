# 消息、模型与兼容性配置（`types`、`model`、`compat`）

上游：`src/types.ts`、`src/providers/model-schema.ts`、`src/providers/compat-schema.ts`、`src/utils/model-operations.ts`、`src/models.ts`（`calculateCost`、思考级别函数）。

## 序列化兼容

所有可序列化类型的 JSON 形状与上游相同：字段 camelCase，内容块用 `type` 标签，消息用 `role` 标签，可选字段缺省时不输出。由 pi（TS）、pi-py 和 tau 序列化的上下文、模型、凭据可以互相读取。`types.rs` 的测试用上游格式的 JSON 做往返校验。

## 消息与内容

| 上游 | Rust |
| --- | --- |
| `TextContent` / `ThinkingContent` / `ImageContent` / `ToolCall` | 同名结构体 |
| 用户消息与工具结果的内容 `(TextContent \| ImageContent)[]` | `UserBlock` 枚举 |
| 助手消息的内容 `(TextContent \| ThinkingContent \| ToolCall)[]` | `AssistantBlock` 枚举 |
| `string \| Block[]` | `Content<B>`（untagged：`Text(String)` 或 `Blocks(Vec<B>)`），`UserContent`、`SystemContent` 是它的别名 |
| `Message` 联合 | `Message` 枚举：`System`、`User`、`Assistant`、`ToolResult` |
| `JsonObject` | `serde_json::Map`（开启 `preserve_order`，保持键顺序） |
| `StopReason` 字符串 | `StopReason` 枚举 |
| `timestamp: number`（毫秒） | `i64` |

`ToolResultMessage::details` 在上游是泛型 `TDetails`（编译期检查 JSON 兼容）；Rust 里是 `Option<serde_json::Value>`，调用方自行序列化。

`Tool::parameters` 是 JSON Schema（`serde_json::Value`），对应上游的 TypeBox schema。`constrainedSampling: false` 反序列化为 `None`，与省略同义。

`Context` 是公共入口的输入，`system_prompt` 和 `tools` 是首条 system 消息的简写。`TranscriptContext` 只能由 `normalize_context` 和 transcript 辅助函数构造（构造函数是 `pub(crate)`），对应上游用品牌类型阻止原始 `Context` 进入 provider 代码。

## 模型

| 上游 | Rust |
| --- | --- |
| `Model<TApi>` | `Model`（`api: String`） |
| `ImageModel` / `ClassifierModel` | 同名结构体 |
| `AnyModel` | `AnyModel` 枚举，序列化时带 `type`，缺少 `type` 视为 chat |
| `ThinkingLevel` / `ModelThinkingLevel` | 同名枚举 |
| `ThinkingLevelMap` | `BTreeMap<ModelThinkingLevel, Option<String>>`：键缺失与值为 `null` 含义不同 |
| `getSupportedThinkingLevels` / `clampThinkingLevel` | `Model::supported_thinking_levels` / `clamp_thinking_level` |
| `calculateCost` | `calculate_cost`：按 tier 选价格，1h 缓存写入按 2 倍输入价计 |

上游 `Model<TApi>` 的 `compat` 字段类型随 `api` 变化。Rust 里 `Model::compat` 是 `ModelCompat` 枚举，反序列化时先读 `api` 再用对应 schema 解析（`#[serde(try_from = "ModelRepr")]`）：

| `api` | 变体 |
| --- | --- |
| `openai-completions` | `OpenAICompletions(OpenAICompletionsCompat)` |
| `openai-responses`、`azure-openai-responses`、`openai-codex-responses` | `OpenAIResponses(OpenAIResponsesCompat)` |
| `anthropic-messages` | `AnthropicMessages(AnthropicMessagesCompat)` |
| `bedrock-converse-stream` | `Bedrock(BedrockCompat)` |
| `mistral-conversations` | `Mistral(MistralConversationsCompat)` |
| 其他 | `Other(JsonObject)` |

兼容性结构体的字段都是 `Option`，未设置时由 API 实现按默认值或 URL 自动检测。上游 schema 允许额外属性（`additionalProperties: true`），所以每个结构体有 `#[serde(flatten)] extra` 保存未知字段，自定义模型文件可以无损往返。

`Model` 序列化时不输出 `type`（chat 是默认类型）；`AnyModel` 序列化时总是输出 `type`。

## 选项（`options`）

`StreamOptions` 对应上游 `StreamOptions` + `ProviderRequestOptions`；`SimpleStreamOptions { base, tool_choice, reasoning, thinking_budgets }` 对应上游 `SimpleStreamOptions`（`deferred` 不在第一阶段）。各 API 的专有选项放在 `StreamOptions::api: Option<ApiOptions>`，与 `model.api` 不匹配的变体被忽略：

| `ApiOptions` 变体 | 上游类型 |
| --- | --- |
| `AnthropicMessages(AnthropicOptions)` | `AnthropicOptions` |
| `OpenAICompletions(OpenAICompletionsOptions)` | `OpenAICompletionsOptions` |
| `OpenAIResponses(OpenAIResponsesOptions)` | `OpenAIResponsesOptions` |
| `AzureOpenAIResponses(AzureOpenAIResponsesOptions)` | `AzureOpenAIResponsesOptions` |
| `GoogleGenerativeAi(GoogleOptions)` | `GoogleOptions` |
| `GoogleVertex(GoogleVertexOptions)` | `GoogleVertexOptions` |
| `MistralConversations(MistralOptions)` | `MistralOptions` |
| `BedrockConverseStream(BedrockOptions)` | `BedrockOptions` |
| `Custom(JsonObject)` | 自定义 API 的 `Record<string, unknown>` |

上游 Anthropic 选项里的 `client`（注入预构建的 SDK 客户端）在手写实现里没有对应物；需要替换传输时用 `StreamOptions::http_client`。

`StreamOptions` 的 `Debug` 输出会隐藏 API key，只列出 env 和 header 的名字。

## 错误（`error`）

- `ModelsError { code, message, source }`：对应上游 `ModelsError`。`code` 多了 `Aborted`（上游用 `AbortError`）。构造时把 cause 的消息拼进 `message`，只显示 `to_string()` 的调用方也能看到原因（上游行为）。
- `Error`：API 实现内部的错误类型，不会从流里抛出：流以 `Error` 事件结束，`error_message` 是它的 `to_string()`。HTTP 错误是 `Error::Status { status, message, retry_after_ms }`。
