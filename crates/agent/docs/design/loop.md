# 底层循环与钩子（`agent_loop`、`hooks`）

对应上游 `src/agent-loop.ts`、`src/stream-fn.ts` 和 `types.ts` 里的 `AgentLoopConfig`。

## 入口

| 函数 | 上游 | 说明 |
| --- | --- | --- |
| `run_agent_loop(prompts, context, config, sink, cancel, stream_fn)` | `runAgentLoop` | 把 `prompts` 加入上下文并发出事件，然后进入循环。返回本次新增的消息 |
| `run_agent_loop_continue(context, config, sink, cancel, stream_fn)` | `runAgentLoopContinue` | 不加新消息，从当前上下文继续（重试）。最后一条消息不能是 assistant，否则 `AgentLoopError` |
| `agent_loop(..)` / `agent_loop_continue(..)` | `agentLoop` / `agentLoopContinue` | 在 Tokio 任务里运行上面两个函数，返回 `AgentEventStream` |

`run_*` 版本把事件交给 `AgentEventSink`，并 await 每次 `emit`：事件接收端处理完一个事件，循环才继续。`Agent` 用这个保证监听器按顺序执行且计入运行结束（见 [agent.md](agent.md)）。

`agent_loop` 版本把事件放进无界通道，循环不等消费者。`AgentEventStream` 实现 `Stream<Item = AgentEvent<C>>`；`result()` 排空事件并返回新增消息。运行结束前 drop 这个流会取消运行：循环拿到的是调用方令牌的子令牌，drop 只取消子令牌，不影响调用方的令牌。

## 配置

```rust
pub struct AgentLoopConfig<C> {
    pub model: Model,
    pub stream_options: SimpleStreamOptions,   // reasoning 即思考级别；cancel、api_key 每次请求由循环设置
    pub tool_execution: ToolExecutionMode,
    pub hooks: Arc<dyn AgentHooks<C>>,
    pub get_steering_messages: Option<MessageSource<C>>,
    pub get_follow_up_messages: Option<MessageSource<C>>,
}
```

上游 `AgentLoopConfig` 继承 `SimpleStreamOptions`，并把各个回调平铺在同一个对象上。delta 把请求选项放在 `stream_options` 字段，把回调收进 `hooks`。`MessageSource<C>` 是返回 `BoxFuture<Vec<AgentMessage<C>>>` 的闭包，对应上游的 `getSteeringMessages` / `getFollowUpMessages`。

## `StreamFn`

```rust
pub type StreamFn = Arc<
    dyn Fn(Model, TranscriptContext, SimpleStreamOptions) -> BoxFuture<'static, AssistantMessageEventStream> + Send + Sync,
>;
```

- 输入是已规范化的 `TranscriptContext`（system prompt 和工具声明都在 system 消息里）。
- 失败必须编码在返回的流里（`Error` 事件），不能 panic。
- `models_stream_fn(models)` 把 `delta_ai::Models::stream_simple` 适配成 `StreamFn`，是常规用法；`stream_proxy`（见 [proxy.md](proxy.md)）是另一种实现；测试里直接构造。
- 上游有全局的 `setDefaultStreamFn`，调用方不传 `streamFn` 时使用。delta 不设全局状态，`StreamFn` 必须显式传入。

## `AgentHooks<C>`

上游的回调合并成一个 trait，每个方法都有默认实现，应用只实现需要的方法：

| 方法 | 上游 | 调用时机 | 默认 |
| --- | --- | --- | --- |
| `convert_to_llm(messages)` | `convertToLlm` | 每次请求前，`transform_context` 之后 | 保留 LLM 消息，丢弃自定义消息 |
| `transform_context(messages, cancel)` | `transformContext` | 每次请求前，在 `convert_to_llm` 之前（裁剪、注入上下文） | 原样返回 |
| `get_api_key(provider)` | `getApiKey` | 每次请求前，用于短期 token | `None`：使用 `stream_options` 里配置的 key |
| `before_tool_call(ctx, cancel)` | `beforeToolCall` | 参数校验后、执行前 | 不拦截 |
| `after_tool_call(ctx, cancel)` | `afterToolCall` | 执行后、`ToolExecutionEnd` 前 | 不覆盖 |
| `finish_turn(turn, cancel)` | `finishTurn` | 本轮助手消息和工具结果都已加入上下文、`TurnEnd` 之前 | 不干预。回复失败时决定被忽略 |
| `prepare_request(request, cancel)` | `prepareRequest` | 每次请求前（包括第一次），待追加消息已加入之后 | 不修改。不读队列 |
| `prepare_next_turn(turn, cancel)` | `prepareNextTurn` | 循环继续时，`TurnEnd` 之后、下一轮 `TurnStart` 之前 | 不修改 |

钩子的输入是借用的上下文结构体（`BeforeToolCallContext<'a, C>`、`AgentTurnContext<'a, C>` 等），方法签名形如 `fn(&'a self, ctx: Ctx<'a, C>, cancel: &'a CancellationToken) -> BoxFuture<'a, Option<R>>`。把这八个回调写成独立的 `Arc<dyn Fn>` 需要每个都带高阶生命周期约束，trait 方法更好写也更好实现。`DefaultHooks` 是全部使用默认实现的空结构体。

`prepare_request` 返回 `AgentRequestUpdate { context, model, thinking_level }`，`prepare_next_turn` 返回 `AgentLoopTurnUpdate`（多一个 `messages`：下一次请求前要追加的消息，会正常发出生命周期事件）。替换的值对本次及之后的请求都生效。

## 循环

流程图见 [README.md](README.md#总体结构)。和上游逐步对应：

1. **开始**：`AgentStart`、`TurnStart`。`run_agent_loop` 先把 prompts 经 `declare_tool_changes` 处理后加入上下文并发出 `MessageStart`/`MessageEnd`。然后读一次 steering 队列（运行启动前排队的消息）。
2. **准备本轮**：不是第一轮时调用 `prepare_next_turn`；如果此时没有待处理消息，再读一次 steering 队列（准备可能很慢，例如压缩上下文），然后发 `TurnStart`。
3. **追加待处理消息**：`prepare_next_turn` 给的消息加上队列消息，经 `declare_tool_changes` 后逐条发事件并加入上下文。
4. **`prepare_request`**。
5. **流式请求**：`transform_context` → `convert_to_llm` → `normalize_context`，解析 API key，调用 `StreamFn`。provider 事件转成 `MessageStart` / `MessageUpdate` / `MessageEnd`。流在没有 `Done`/`Error` 的情况下结束时，记为错误 `Stream ended without a final message`。最终消息的 `thinking_level` 记录请求的级别。
6. **失败**：回复以 `error`/`aborted` 结束时，调用 `finish_turn`（忽略决定），发 `TurnEnd`、`AgentEnd`，结束。
7. **工具**：有工具调用时执行（见 [tools.md](tools.md)），结果加入上下文。全部结果都要求 `terminate` 时，本批之后不再因工具调用而继续。
8. **结束本轮**：`finish_turn`，`TurnEnd`。决定为 `End` 时发 `AgentEnd` 并结束。
9. **继续条件**：读 steering 队列。还有工具调用要继续或有 steering 消息时回到第 2 步。否则读 follow-up 队列，有消息时回到第 2 步；都没有但 `finish_turn` 返回了 `Continue` 时，再跑一轮只用当前上下文的请求；否则发 `AgentEnd` 结束。

### 工具声明变更（`declare_tool_changes`）

`context.tools` 是运行时可以执行的工具；对话记录里的 system 消息声明模型可以调用哪些工具（`delta_ai` 的 transcript 设计，见 [delta_ai transcript.md](../../../ai/docs/design/transcript.md)）。两者不一致时（例如应用调用了 `Agent::set_tools`），下一次请求前把差异写成 system 消息上的 `tools_added` / `tools_removed`：

- 待追加的消息里有 system 消息时，改写最后一条：它的工具字段替换为差异。
- 否则在第一条非 system 消息前插入一条新的 system 消息。
- 没有差异且调用方的 system 消息本身没声明变更时，原样保留。

这个函数是公开的，与上游一样，应用可以在自己构造消息时使用。

## 与上游的差异

- 配置拆成 `stream_options` + `hooks` + 队列来源，见上。
- 没有全局默认 `StreamFn`。
- 上下文里存的是 `Done`/`Error` 事件携带的最终消息；上游存的是流式过程中原地修改的 partial 对象，内容相同。
- `agent_loop` 返回的流被 drop 时取消运行；上游的 `EventStream` 没有取消语义。
