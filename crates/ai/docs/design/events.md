# 事件协议（`events`）

上游：`src/types.ts`（`AssistantMessageEvent`）、`src/utils/event-stream.ts`、`src/api/lazy.ts`。

## 协议

成功：`Start → 更新事件* → Done`。生成开始后失败：`Start → 更新事件* → Error`。请求准备阶段（鉴权、构造请求）失败时，流里只有一个 `Error`。更新事件和 `Done` 不会出现在 `Start` 之前，终止事件之后不再有事件。

| 事件 | 上游 type | 字段 | 说明 |
| --- | --- | --- | --- |
| `Start` | `start` | `message` | 响应元数据，通常没有内容块 |
| `TextStart` | `text_start` | `content_index` | 文本块开始，内容为空 |
| `TextDelta` | `text_delta` | `content_index`、`delta` | |
| `TextEnd` | `text_end` | `content_index`、`content`、`signature` | `content` 为最终文本 |
| `ThinkingStart` | `thinking_start` | `content_index`、`thinking`、`redacted`、`signature` | 普通思考为空；被屏蔽的思考可能在开始时就完整 |
| `ThinkingDelta` | `thinking_delta` | `content_index`、`delta` | |
| `ThinkingEnd` | `thinking_end` | `content_index`、`content`、`signature` | |
| `ToolCallStart` | `toolcall_start` | `content_index`、`id`、`name`、`arguments` | `arguments` 由 provider 决定，通常为空 |
| `ToolCallDelta` | `toolcall_delta` | `content_index`、`delta` | 参数 JSON 的片段 |
| `ToolCallEnd` | `toolcall_end` | `content_index`、`tool_call` | 完整的工具调用，未做 schema 校验 |
| `Done` | `done` | `reason`、`message` | `reason`：stop / length / toolUse / deferred |
| `Error` | `error` | `reason`、`error` | `reason`：error / aborted；`error` 带部分内容和 `error_message` |

不同内容块的事件可以交错（例如 `text_start, text_delta, toolcall_start, text_delta, toolcall_delta`），消费者按 `content_index` 关联。

## 与上游的差异：`partial` 改为消费端归约

上游每个非终止事件都带 `partial`：provider 正在修改的同一个 `AssistantMessage` 对象，不是事件发生时的快照。Rust 里共享可变对象需要加锁，而且语义含糊，所以（经确认）改为：

- 事件本身不带 `partial`。
- `AssistantMessageEventStream` 每产出一个事件，就把它归约进自己持有的消息副本；`partial()` 返回截至刚产出事件时的消息，是精确的事件时刻快照。
- 为了让消费端能归约，事件补充了块数据：`ToolCallStart` 带 `id`、`name`、初始参数；`ThinkingStart` 带被屏蔽思考的完整内容和签名；`TextEnd`/`ThinkingEnd` 带最终签名。
- 工具调用参数：消费端累积 `ToolCallDelta` 的 JSON 片段，每次用流式 JSON 解析得到当前参数（与上游每个 delta 重新解析一致）。
- 只调用 `result()` 的消费者不做归约，没有额外开销。

响应元数据（usage、response id 等）在流式过程中只在终止事件里更新，`partial()` 的这些字段可能滞后；这与上游“partial 不是快照”的约定相容。

## 类型

```rust
let (sender, stream) = AssistantMessageEventStream::channel();
```

- **`EventSender`**：生产端。终止事件之后的事件被忽略。给终止消息写 `duration_ms`：从 sender 创建起用单调时钟计时；消息已有 `duration_ms`，或其 `timestamp` 早于流创建时间（转发别处开始的响应）时不计时，与上游一致。
- **`AssistantMessageEventStream`**：消费端，实现 `futures::Stream<Item = AssistantMessageEvent>`。`partial()` 见上；`result().await` 读到终止事件为止并返回最终消息；生产者没有发终止事件就结束时，返回 `stop_reason = error` 的消息（"Stream ended without a final message"）。`from_final(message)` 构造只含一个终止事件的流。
- **`MessageBuilder`**：API 实现用来构建响应的辅助类型。它持有 `output: AssistantMessage`，所有内容修改都通过它的方法完成，方法同时发出对应事件，保证生产端和消费端的消息一致：

| 方法 | 事件 |
| --- | --- |
| `start()` | `Start`（只发一次） |
| `start_text()` / `text_delta()` / `end_text()` | `TextStart` / `TextDelta` / `TextEnd` |
| `start_thinking()` / `thinking_delta()` / `end_thinking()` | `ThinkingStart` / `ThinkingDelta` / `ThinkingEnd` |
| `start_tool_call()` / `tool_call_delta()` / `end_tool_call()` | `ToolCallStart` / `ToolCallDelta` / `ToolCallEnd` |
| `finish()` | `Done`（`stop_reason` 为 error/aborted 时发 `Error`） |
| `fail(message, aborted)` | `Error` |

`end_tool_call` 用 `json::parse_tool_call_arguments` 解析累积的参数文本：不是 JSON 对象时参数为空，并在 `ToolCall::arguments_error` 里记录错误和原文，供工具校验把问题反馈给模型（上游行为）。没有收到任何参数片段时保留开始时的参数。

## 运行时要求

API 实现和 `lazy_stream` 在 Tokio 任务里运行，所以需要 Tokio 运行时。通道无界，消费者慢时事件在队列里等待，与上游的内存队列一致。消费者丢弃流后（`is_closed()`），生产者停止工作（见 README 的待确认第 1 条）。
