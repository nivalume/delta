# tau

[pi](https://github.com/earendil-works/pi) 各个库的 Rust 移植版。

| crate | 导入名 | 对应上游 | 用途 |
| --- | --- | --- | --- |
| [`tau-llm`](crates/ai) | `tau_ai` | `@earendil-works/pi-ai` | 统一的 LLM API：provider 集合、鉴权解析、流式事件、token 与费用统计、工具调用、跨 provider 切换上下文 |
| [`tau-agent-runtime`](crates/agent) | `tau_agent` | `@earendil-works/pi-agent-core` | 有状态的 agent：工具执行（并行或顺序）、steering 与 follow-up 队列、生命周期事件和钩子 |

状态：开发中，尚未发布。

## 许可证

MIT，见 [LICENSE](LICENSE)。
