# 对话记录处理（`transcript`、`transform`、`estimate`、`simple_options`）

上游：`src/utils/transcript.ts`、`src/utils/text.ts`、`src/api/transform-messages.ts`、`src/utils/estimate.ts`、`src/api/simple-options.ts`。四个模块都是纯函数，与上游逐一对应。

## system 消息（`transcript`）

`Context::system_prompt` 和 `tools` 是首条 system 消息的简写，`normalize_context` 把它们折叠进去。之后的 system 消息可以在对话中途修改提示词和工具集：`content` 追加指令，`sections` 按名字替换（`None` 删除），`tools_added` / `tools_removed` 增删工具。按顺序重放所有 system 消息得到当前提示词和工具。

| 函数 | 作用 |
| --- | --- |
| `normalize_context` | 生成 `TranscriptContext` |
| `get_current_tools` | 重放工具增删；同名工具重新定义时保持原位置（与 JS `Map.set` 一致） |
| `get_current_system_message` / `get_current_system_prompt` | 重放提示词与 sections |
| `collapse_system_messages` | 不支持中途 system 消息的模型：重放结果作为首条消息，其余 system 消息删除 |
| `resolve_transcript(ctx, supports)` | 支持时原样保留，否则折叠 |
| `render_system_message_update` | 支持中途 system 消息的 API 渲染更新文本 |
| `resolve_transcript_tools` | 能锚定新增工具的传输：顶层只放初始工具；有删除或重定义时发送当前完整工具列表 |
| `get_tool_state_changes`、`get_declared_tools`、`has_non_additive_tool_changes`、`declarations_equal` | 工具状态比较 |

上游已标记废弃的 `hasToolRedefinitions` 没有移植。`declarations_equal` 直接比较字段（上游借 JSON 序列化比较，目的相同）。

## 跨 provider 切换（`transform`）

`transform_messages(messages, model, normalize_tool_call_id)`：

- 模型不支持图片：用户消息和工具结果里的图片换成占位文本（连续多张只放一个占位）。
- 来自其他模型（provider、api、id 任一不同）的助手消息：删除思考块；文本去掉签名；工具调用去掉 `thought_signature`，可选地规范化 id（例如 Anthropic 要求 `^[a-zA-Z0-9_-]{1,64}$`），后续工具结果跟随映射。
- 同一模型的思考块：有签名或被屏蔽的保留（即使文本为空，加密推理需要回放）；无签名且文本为空的删除。
- 出错或中止的助手消息跳过，不回放。
- 没有结果的工具调用：在下一条助手或用户消息之前插入错误结果 "No result provided"；夹在工具调用和结果之间的 system 消息移到结果之后。

`split_error_tool_result_images`：Anthropic 和 Bedrock 上的 Claude 拒绝错误工具结果里的图片，把图片拆出来放到本轮工具结果之后。

## token 估算（`estimate`）

每 token 3.5 个字符，图片按 4800 个字符计，字符数按 UTF-16 码元计（与上游 JS 一致）。`estimate_context_tokens` 优先用最后一条可用助手消息报告的 usage，再加上它之后消息的估算；若之后插入了时间更早的前缀消息（例如压缩摘要），该 usage 不再适用。

## 简化选项（`simple_options`）

| 函数 | 作用 |
| --- | --- |
| `build_base_options` | 公共选项：按思考级别合并采样参数（模型默认 → 级别覆盖 → 请求参数），`max_tokens` 限制在上下文窗口内（预留 4096） |
| `clamp_reasoning` | `xhigh`、`max` 对按预算思考的 API 降为 `high` |
| `thinking_budget_for_level` | 默认预算：minimal 1024、low 2048、medium 8192、high 16384，可按级别覆盖 |
| `adjust_max_tokens_for_thinking` | 思考预算与回答共享上限时，至少给回答留 1024 token |
