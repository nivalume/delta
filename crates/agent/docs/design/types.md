# 消息、事件与工具结果（`types`）

对应上游 `src/types.ts`。

## `AgentMessage<C>` 与自定义消息

对话记录里除了 LLM 消息，应用还可以放自己的消息（例如 UI 通知、压缩摘要）。上游通过 TypeScript 声明合并扩展 `AgentMessage` 联合类型；delta 用类型参数：

```rust
pub trait CustomMessage: Clone + Debug + Send + Sync + Serialize + DeserializeOwned + 'static {
    fn role(&self) -> &str;      // 不能与 LLM 消息的 role 重名
    fn timestamp(&self) -> i64;  // 毫秒
}

#[serde(untagged)]
pub enum AgentMessage<C = NoCustom> {
    Llm(Message),
    Custom(C),
}
```

- `NoCustom` 是空枚举，默认参数。不用自定义消息的应用写 `Agent`、`AgentMessage` 即可。
- `untagged` 序列化：`Llm` 变体序列化成内部的 `Message`，所以没有自定义消息的对话记录和上游 JSON 完全相同。反序列化先按 `Message`（按 `role` 区分）尝试，失败再按 `C`。
- 自定义消息不会发给模型。`AgentHooks::convert_to_llm` 的默认实现丢弃它们；需要把它们转成 user 消息等的应用自己实现。
- `role()`、`timestamp()`、`as_llm()`、`as_assistant()`、`is_system()` 是常用访问器；`From<Message>`、`From<AssistantMessage>`、`From<UserMessage>` 等用于构造。

## 工具结果

```rust
pub struct AgentToolResult {
    pub content: Vec<UserBlock>,              // 发给模型的文本或图片
    pub details: Option<Value>,               // 给日志和 UI，不发给模型
    pub structured_content: Option<Value>,    // 匹配工具 output schema 的结构化结果，不发给模型
    pub usage: Option<Usage>,                 // 工具自身的用量（例如子 agent）
    pub is_error: bool,
    pub terminate: bool,
}
```

JSON 形状与上游一致（camelCase，`false`/`None` 字段省略）。

- `is_error`：工具可以不返回 `Err` 而直接报告失败，这样 `details` 和 `structured_content` 会保留。
- `terminate`：提示在当前这批工具调用后停止。只有这批里**每个**最终结果都设置了它才生效，与上游相同。
- `AgentToolResult::text(..)` 构造文本结果；`AgentToolResult::error(..)` 构造循环在失败时记录的结果（文本 + 空 `details` 对象，与上游一致）。

`AgentToolCallOutcome` 是钩子处理后的最终结果，带 `tool_call`、`is_error` 和 `duration_ms`（工具没有执行时为 `None`）。

## 事件

```rust
pub enum AgentEvent<C = NoCustom> {
    AgentStart,
    AgentEnd { messages: Vec<AgentMessage<C>> },
    TurnStart,
    TurnEnd { message: AgentMessage<C>, tool_results: Vec<ToolResultMessage> },
    MessageStart { message: AgentMessage<C> },
    MessageUpdate { message: Arc<AssistantMessage>, assistant_message_event: AssistantMessageEvent },
    MessageEnd { message: AgentMessage<C> },
    ToolExecutionStart { tool_call_id, tool_name, args },
    ToolExecutionUpdate { tool_call_id, tool_name, args, partial_result },
    ToolExecutionEnd { tool_call_id, tool_name, result, is_error, duration_ms },
}
```

`kind()` 返回上游的事件类型名（`agent_start`、`message_update` 等），用于日志和跨语言对照。

事件顺序与上游相同：

- 一次运行：`AgentStart`，然后若干轮，最后 `AgentEnd`。`AgentEnd.messages` 是本次运行新增的消息。
- 一轮：`TurnStart`，本轮输入消息的 `MessageStart`/`MessageEnd`，助手回复的 `MessageStart`、若干 `MessageUpdate`、`MessageEnd`，工具执行事件，工具结果消息的 `MessageStart`/`MessageEnd`，最后 `TurnEnd`。
- system、user、tool result 和自定义消息只有 `MessageStart`/`MessageEnd`；`MessageUpdate` 只用于正在流式生成的助手消息。

### `MessageUpdate` 的快照

上游的 `message_update` 事件携带流式过程中被原地修改的 partial 对象。Rust 里事件要能被监听器保存、跨任务传递，所以 `message` 是 `Arc<AssistantMessage>` 快照：循环每取出一个 provider 事件，就把 `AssistantMessageEventStream::partial()` 克隆一次放进 `Arc`。`Agent` 把同一个 `Arc` 存进 `state.streaming_message`，监听器拿到的是同一份数据，不会再次克隆。

代价是每个 delta 一次消息克隆。消息在流式过程中通常只有几 KB 到几十 KB，这个开销相对网络延迟可以忽略。

### 不实现 `Serialize`

`AgentEvent` 只在进程内传递，不实现序列化。上游也没有把事件当作持久化格式；需要跨进程传事件的应用（例如 RPC 模式）自己定义 wire 格式。

## 钩子的返回值

| 类型 | 用于 | 说明 |
| --- | --- | --- |
| `BeforeToolCallResult { block, reason, terminate }` | `before_tool_call` | `block` 时不执行工具，记录错误结果，文本默认 `Tool execution was blocked` |
| `AfterToolCallResult { content, details, structured_content, is_error, usage, terminate }` | `after_tool_call` | 逐字段覆盖，`None` 保留原值，不做深合并。替换 `content` 但没给 `structured_content` 时丢弃原结构化结果，因为它可能不再匹配 |
| `AgentTurnDecision::{Continue, End}` | `finish_turn` | `Continue` 保证再发一次请求（已有工具结果或队列消息会触发请求时不额外加一轮）；`End` 在 `TurnEnd` 后结束运行，不读队列 |

## 其他枚举

- `ToolExecutionMode::{Sequential, Parallel}`，默认 `Parallel`，序列化为小写。
- `QueueMode::{All, OneAtATime}`，默认 `OneAtATime`，序列化为 `all` / `one-at-a-time`。
