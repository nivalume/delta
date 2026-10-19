# tau

Rust port of [pi](https://github.com/earendil-works/pi)'s libraries.

| Crate | Import | Upstream | What it does |
| --- | --- | --- | --- |
| [`tau-llm`](crates/ai) | `tau_ai` | `@earendil-works/pi-ai` | Unified LLM API: provider collections, auth resolution, streaming events, token and cost tracking, tool calling, cross-provider context hand-off |
| [`tau-agent-runtime`](crates/agent) | `tau_agent` | `@earendil-works/pi-agent-core` | Stateful agent: tool execution (parallel or sequential), steering and follow-up queues, lifecycle events and hooks |

Status: in development; nothing is published yet.

中文说明见 [README.zh-CN.md](README.zh-CN.md)。

## License

MIT. See [LICENSE](LICENSE).
