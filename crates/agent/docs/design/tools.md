# 工具（`tool`）

对应上游 `types.ts` 里的 `AgentTool` 和 `agent-loop.ts` 里的工具执行部分。

## `AgentTool` trait

上游的 `AgentTool` 是一个对象接口：继承 `Tool`（名称、描述、参数 schema），加上 `label`、`execute` 和几个可选字段。tau 用 trait：

```rust
pub trait AgentTool: Send + Sync {
    fn definition(&self) -> &Tool;                                   // 发给模型的声明
    fn label(&self) -> &str { &self.definition().name }
    fn execution_mode(&self) -> Option<ToolExecutionMode> { None }
    fn output_schema(&self) -> Option<&Value> { None }
    fn replay(&self) -> Option<ToolReplay> { None }
    fn prepare_arguments(&self, arguments: JsonObject) -> JsonObject { arguments }
    fn execute(&self, arguments: Value, context: ToolContext)
        -> BoxFuture<'_, Result<AgentToolResult, ToolError>>;
}
```

| 上游 | tau | 说明 |
| --- | --- | --- |
| `name`、`description`、`parameters`（继承自 `Tool`） | `definition() -> &Tool` | 直接复用 `tau_ai::Tool`，包括 `constrained_sampling` |
| `label` | `label()` | 默认取工具名 |
| `executionMode` | `execution_mode()` | 任一调用的工具是 `Sequential`，整批顺序执行 |
| `outputSchema` | `output_schema()` | 只是声明，循环不校验 |
| `replay` | `replay()` | `ToolReplay::{Never, Safe}`，供持久化执行的宿主使用，循环不读取 |
| `prepareArguments` | `prepare_arguments()` | 在 schema 校验之前处理原始参数 |
| `execute(toolCallId, params, signal, onUpdate)` | `execute(arguments, ToolContext)` | 见下 |

工具以 `Arc<dyn AgentTool>` 存放在 `AgentContext::tools` 和 `AgentState::tools` 里。

### `ToolContext`

上游 `execute` 的位置参数合并成一个结构体：

```rust
pub struct ToolContext {
    pub tool_call_id: String,
    pub cancel: CancellationToken,   // 本次运行的取消令牌
    pub updates: ToolUpdates,        // 上游的 onUpdate 回调
}
```

`ToolUpdates::send(partial)` 发出 `ToolExecutionUpdate` 事件。它是无界通道的发送端，工具不需要 await。`execute` 返回之前发出的更新都会送达（返回后先排空通道）；返回之后的调用被忽略，与上游“`onUpdate` 在工具结束后无效”一致。`ToolUpdates::none()` 用于在循环外直接调用工具。

### 失败

`execute` 返回 `Err(ToolError)`（`Box<dyn Error + Send + Sync>`），对应上游抛异常：循环记录 `AgentToolResult::error(err.to_string())`，`is_error = true`。也可以返回 `is_error: true` 的结果来保留 `details`。

## 便捷实现

### `FnTool`

声明加闭包：

```rust
let tool = FnTool::new(Tool::new("echo", "Echo the input", schema), |args, _ctx| async move {
    Ok(AgentToolResult::text(args["text"].as_str().unwrap_or_default()))
})
.with_label("Echo")
.with_execution_mode(ToolExecutionMode::Sequential);
```

### `typed_tool`（feature `schemars`，默认开启）

参数是 Rust 类型，schema 由 `schemars` 生成，校验通过后反序列化：

```rust
#[derive(Deserialize, JsonSchema)]
struct GetTime {
    /// IANA timezone name.
    timezone: Option<String>,
}

let tool = typed_tool("get_time", "Get the current time", |args: GetTime, _ctx| async move {
    Ok(AgentToolResult::text(format!("12:00 in {}", args.timezone.unwrap_or_else(|| "UTC".into()))))
});
```

`schema_for::<T>()` 生成工具 schema 所需的形状：draft-07，子 schema 内联（不产生 `$ref`/`definitions`，部分 provider 不支持引用），去掉 `$schema` 和根 `title`。上游用 TypeBox 在运行时构造 schema，Rust 里对应的是从类型派生。

## 一次工具调用的步骤

与上游相同，每个调用依次经过：

1. **查找工具**：按名称在 `context.tools` 里找。找不到记录错误结果 `Tool {name} not found`。
2. **`prepare_arguments`**：处理原始参数。
3. **校验**：`tau_ai::validation::validate_tool_arguments`，包括上游的类型强制转换（字符串数字转数字等）。失败时错误文本发给模型。
4. **`before_tool_call` 钩子**：可以阻止执行。钩子返回后若运行已取消，记录 `Operation aborted`。
5. **执行**：`execute`，期间转发 `ToolExecutionUpdate`，记录耗时。
6. **`after_tool_call` 钩子**：逐字段覆盖结果。
7. **发事件**：`ToolExecutionEnd`，然后是工具结果消息的 `MessageStart`/`MessageEnd`。

第 1–4 步失败的调用不执行，`duration_ms` 为 `None`。

### 顺序与并行

- **顺序**：每个调用走完全部步骤再开始下一个；运行取消后不再开始新的调用。
- **并行**（默认）：先按顺序做第 1–4 步（`ToolExecutionStart` 按源顺序发出），再并发执行通过的调用。`ToolExecutionEnd` 按完成顺序发出，工具结果消息按源顺序发出。
- 回复因 `length` 停止时，所有工具调用的参数都可能被截断，循环直接为每个调用记录错误结果，不执行（上游行为）。

### `run_tool_call`

调用其他工具的工具（例如上游 `examples/mcp-codemode`）用 `run_tool_call` 执行嵌套调用。它走同样的第 1–6 步，所以钩子对嵌套调用同样生效，但不发事件、不产生消息，失败以 `is_error` 的结果返回。工具的中间结果交给 `RunToolCallOptions::on_update`（上游 `onUpdate`）。

## 与上游的差异

- trait 代替对象接口；`execute` 的位置参数合并成 `ToolContext`。
- 参数 schema 是 `serde_json::Value`（上游是 TypeBox 对象）。`typed_tool` 提供类型化的写法。
- `details` 是 `Option<Value>`，上游是泛型 `TDetails`。需要类型化的应用自己序列化/反序列化。
