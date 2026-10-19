# Messages, models and compat configuration (`types`, `model`, `compat`)

Upstream: `src/types.ts`, `src/providers/model-schema.ts`, `src/providers/compat-schema.ts`, `src/utils/model-operations.ts`, `src/models.ts` (`calculateCost`, thinking level functions).

## Serialization compatibility

Every serializable type has the same JSON shape as upstream: camelCase fields, content blocks tagged by `type`, messages tagged by `role`, optional fields omitted when absent. Contexts, models and credentials serialized by pi (TS), pi-py and delta can be read by each other. The tests in `types.rs` round-trip JSON in upstream format.

## Messages and content

| Upstream | Rust |
| --- | --- |
| `TextContent` / `ThinkingContent` / `ImageContent` / `ToolCall` | Structs with the same names |
| User message and tool result content `(TextContent \| ImageContent)[]` | `UserBlock` enum |
| Assistant message content `(TextContent \| ThinkingContent \| ToolCall)[]` | `AssistantBlock` enum |
| `string \| Block[]` | `Content<B>` (untagged: `Text(String)` or `Blocks(Vec<B>)`); `UserContent` and `SystemContent` are aliases of it |
| `Message` union | `Message` enum: `System`, `User`, `Assistant`, `ToolResult` |
| `JsonObject` | `serde_json::Map` (with `preserve_order` enabled to keep key order) |
| `StopReason` string | `StopReason` enum |
| `timestamp: number` (milliseconds) | `i64` |

`ToolResultMessage::details` is a generic `TDetails` in upstream (JSON compatibility checked at compile time). In Rust it is `Option<serde_json::Value>`, and the caller serializes it.

`Tool::parameters` is a JSON Schema (`serde_json::Value`), corresponding to upstream's TypeBox schema. `constrainedSampling: false` deserializes to `None`, which means the same as omitting it.

`Context` is the input to the public entry points; `system_prompt` and `tools` are shorthand for the first system message. `TranscriptContext` can only be built by `normalize_context` and the transcript helper functions (the constructor is `pub(crate)`). This matches upstream's use of a branded type to keep a raw `Context` out of provider code.

## Models

| Upstream | Rust |
| --- | --- |
| `Model<TApi>` | `Model` (`api: String`) |
| `ImageModel` / `ClassifierModel` | Structs with the same names |
| `AnyModel` | `AnyModel` enum, serialized with `type`; a missing `type` means chat |
| `ThinkingLevel` / `ModelThinkingLevel` | Enums with the same names |
| `ThinkingLevelMap` | `BTreeMap<ModelThinkingLevel, Option<String>>`: a missing key and a `null` value mean different things |
| `getSupportedThinkingLevels` / `clampThinkingLevel` | `Model::supported_thinking_levels` / `clamp_thinking_level` |
| `calculateCost` | `calculate_cost`: picks the price by tier; 1h cache writes are charged at 2x the input price |

In upstream `Model<TApi>`, the type of the `compat` field depends on `api`. In Rust, `Model::compat` is the `ModelCompat` enum. Deserialization reads `api` first and then parses with the matching schema (`#[serde(try_from = "ModelRepr")]`):

| `api` | Variant |
| --- | --- |
| `openai-completions` | `OpenAICompletions(OpenAICompletionsCompat)` |
| `openai-responses`, `azure-openai-responses`, `openai-codex-responses` | `OpenAIResponses(OpenAIResponsesCompat)` |
| `anthropic-messages` | `AnthropicMessages(AnthropicMessagesCompat)` |
| `bedrock-converse-stream` | `Bedrock(BedrockCompat)` |
| `mistral-conversations` | `Mistral(MistralConversationsCompat)` |
| Others | `Other(JsonObject)` |

All fields of the compat structs are `Option`. When a field is unset, the API implementation uses a default or detects the value from the URL. The upstream schema allows extra properties (`additionalProperties: true`), so each struct has a `#[serde(flatten)] extra` that keeps unknown fields. Custom model files round-trip without loss.

`Model` does not emit `type` when serialized (chat is the default type). `AnyModel` always emits `type`.

## Options (`options`)

`StreamOptions` corresponds to upstream `StreamOptions` + `ProviderRequestOptions`. `SimpleStreamOptions { base, tool_choice, reasoning, thinking_budgets, deferred }` corresponds to upstream `SimpleStreamOptions`. The generic `RequestOptions<M>` is upstream `ProviderRequestOptions<TModel>` for the image, classifier and deferred options; those types are described in [images-classifiers-deferred.md](images-classifiers-deferred.md). API-specific options go in `StreamOptions::api: Option<ApiOptions>`. A variant that does not match `model.api` is ignored:

| `ApiOptions` variant | Upstream type |
| --- | --- |
| `AnthropicMessages(AnthropicOptions)` | `AnthropicOptions` |
| `OpenAICompletions(OpenAICompletionsOptions)` | `OpenAICompletionsOptions` |
| `OpenAIResponses(OpenAIResponsesOptions)` | `OpenAIResponsesOptions` |
| `AzureOpenAIResponses(AzureOpenAIResponsesOptions)` | `AzureOpenAIResponsesOptions` |
| `GoogleGenerativeAi(GoogleOptions)` | `GoogleOptions` |
| `GoogleVertex(GoogleVertexOptions)` | `GoogleVertexOptions` |
| `MistralConversations(MistralOptions)` | `MistralOptions` |
| `BedrockConverseStream(BedrockOptions)` | `BedrockOptions` |
| `Custom(JsonObject)` | `Record<string, unknown>` for custom APIs |

The `client` field in upstream's Anthropic options (inject a prebuilt SDK client) has no counterpart in the hand-written implementation. To replace the transport, use `StreamOptions::http_client`.

The `Debug` output of `StreamOptions` hides the API key and lists only the names of env vars and headers.

## Errors (`error`)

- `ModelsError { code, message, source }`: corresponds to upstream `ModelsError`. `code` has an extra `Aborted` (upstream uses `AbortError`). The constructor appends the cause's message to `message`, so callers that only display `to_string()` still see the cause (upstream behavior).
- `Error`: the internal error type of API implementations. It is never thrown from the stream: the stream ends with an `Error` event whose `error_message` is its `to_string()`. HTTP errors are `Error::Status { status, message, retry_after_ms }`.
