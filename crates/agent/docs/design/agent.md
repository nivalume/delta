# `Agent`（`agent`）

对应上游 `src/agent.ts`。`Agent<C>` 持有状态，把底层循环包装成可以多次 `prompt` 的对象。

## 句柄

`Agent<C>` 内部是 `Arc<Inner>`，`clone()` 得到共享同一个 agent 的句柄。一个任务 `await agent.prompt(..)` 的同时，另一个任务可以 `steer`、`abort` 或读状态。上游是单线程 JS 对象，不需要这层。

内部用 `std::sync::Mutex` 保护状态、设置、监听器和两个队列。锁只在同步代码里短暂持有，不跨 `await`。

## 构造

```rust
let agent: Agent = Agent::new(AgentOptions {
    initial_state: AgentInitialState {
        system_prompt: Some("You are helpful.".into()),
        model: Some(model),
        tools: vec![Arc::new(tool)],
        ..Default::default()
    },
    hooks: Arc::new(MyHooks),
    ..AgentOptions::new(models_stream_fn(models))
});
```

- `AgentOptions::new(stream_fn)` 给出其余字段的默认值：`DefaultHooks`、两个队列都是 `OneAtATime`、传输 `Auto`、工具并行执行。
- `AgentOptions` 还包括 `session_id`、`thinking_budgets`、`transport`、`max_retry_delay`、`on_payload`、`on_response`、`on_provider_stream_event`，每次运行时放进 `SimpleStreamOptions`。
- `AgentInitialState`：`messages` 不以 system 消息开头时，`system_prompt` 和 `tools` 组成开头的 system 消息（与上游一致：system prompt 和工具声明存在对话记录里）。没有 `model` 时使用占位模型 `unknown`。

## 状态

```rust
pub struct AgentState<C> {
    pub model: Model,
    pub thinking_level: ModelThinkingLevel,
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub messages: Vec<AgentMessage<C>>,
    pub is_streaming: bool,
    pub streaming_message: Option<StreamingMessage<C>>,
    pub pending_tool_calls: HashSet<String>,
    pub error_message: Option<String>,
}
```

- 读：`with_state(|state| ..)`；写：`with_state_mut(|state| ..)`。常用字段有 getter/setter（`messages()`、`set_model()`、`set_tools()`、`system_prompt()` 等）。
- `system_prompt()` 从对话记录的 system 消息重放得到，没有单独的字段。
- `set_tools()` 只改可执行工具；与对话记录里声明的差异由循环在下一次请求前写成 system 消息（见 [loop.md](loop.md#工具声明变更declare_tool_changes)）。
- `streaming_message` 是 `StreamingMessage::Message(AgentMessage<C>)`（`MessageStart` 到 `MessageEnd` 之间）或 `StreamingMessage::Assistant(Arc<AssistantMessage>)`（最近一次 `MessageUpdate` 的快照）。上游这个字段的类型就是 `AgentMessage`；Rust 里拆成两个变体，以便直接存 `MessageUpdate` 的 `Arc`，不再克隆。

### 状态归约

`Agent` 实现 `AgentEventSink`。每个事件先归约进状态，再交给监听器，与上游 `processEvents` 相同：

| 事件 | 状态变化 |
| --- | --- |
| `MessageStart` | `streaming_message = Message(msg)` |
| `MessageUpdate` | `streaming_message = Assistant(arc)` |
| `MessageEnd` | 清空 `streaming_message`，消息追加到 `messages` |
| `ToolExecutionStart` / `End` | 加入 / 移出 `pending_tool_calls` |
| `TurnEnd` | 助手消息有 `error_message` 时写入 `state.error_message` |
| `AgentEnd` | 清空 `streaming_message` |

## 监听器

```rust
pub type AgentListener<C> =
    Arc<dyn Fn(Arc<AgentEvent<C>>, CancellationToken) -> BoxFuture<'static, ()> + Send + Sync>;
```

- `subscribe(listener)` 返回 `SubscriptionId`，`unsubscribe(id)` 取消订阅（上游 `subscribe` 返回取消函数）。
- 监听器按订阅顺序依次 await，事件以 `Arc` 共享，不为每个监听器克隆。
- 监听器的执行计入运行：`AgentEnd` 的监听器全部返回后，agent 才变为空闲。

## 运行

| 方法 | 上游 | 说明 |
| --- | --- | --- |
| `prompt(messages)` | `prompt(messages)` | 用一批消息开始运行 |
| `prompt_text(text, images)` | `prompt(text, images)` | 文本加可选图片组成一条 user 消息 |
| `continue_run()` | `continue()` | 末尾不是 assistant 时重试；是 assistant 时依次尝试一批 steering 消息、一批 follow-up 消息，都没有则 `AssistantTail` |
| `abort()` | `abort()` | 取消当前运行 |
| `wait_for_idle()` | `waitForIdle()` | 当前运行和它的监听器都结束后返回 |
| `reset()` | `reset()` | 清空对话和队列，保留重放得到的 system 消息（prompt 和工具基线）。运行中调用返回 `AlreadyRunning` |

`continue` 是 Rust 关键字，所以叫 `continue_run`。

已有运行时 `prompt`/`continue_run` 返回 `AgentError::AlreadyRunning`，错误文本与上游相同。

运行开始时设置 `is_streaming = true`、清空 `streaming_message` 和 `error_message`，创建本次运行的 `CancellationToken`。运行结束时由一个 drop guard 清理（`is_streaming = false`、清空 `pending_tool_calls`、通知空闲）。`prompt` 返回的 future 在运行结束前被 drop 时，guard 同样清理状态，并取消运行令牌，让仍持有令牌的工具停止；这种情况下不会发出 `AgentEnd`。

`wait_for_idle` 用 `tokio::sync::watch` 实现，多个任务可以同时等待。

## 队列

- `steer(msg)`：在当前助手回复（及其工具调用）结束后注入。
- `follow_up(msg)`：只在 agent 本来要停下时注入。
- 模式由 `set_steering_mode` / `set_follow_up_mode` 设置：`OneAtATime` 每次取一条，`All` 一次取全部。
- `clear_steering_queue`、`clear_follow_up_queue`、`clear_all_queues`、`has_queued_messages`、`peek_queued_messages`（下一轮会取走的消息，steering 优先，不消费）。

运行开始前排队的 steering 消息会在第一次请求前注入。`continue_run` 用 steering 批次重启运行时，跳过开头那次 steering 读取，避免 `OneAtATime` 模式一次注入两条。

## 与上游的差异

- 可克隆的共享句柄，内部加锁。
- 状态通过闭包读写；`streaming_message` 拆成两个变体。
- `subscribe` 返回 `SubscriptionId`。
- 错误用 `Result<(), AgentError>` 返回。上游 `runWithLifecycle` 捕获循环抛出的异常并合成一条失败的助手消息；Rust 的循环不会以错误结束（流的失败编码在流里，钩子不返回 `Result`），所以没有这条路径。
- `prompt` 的 future 被 drop 会取消运行。上游的 promise 无法取消，只能 `abort()`。
