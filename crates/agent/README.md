# tau-agent-runtime

Stateful LLM agent for Rust, built on [`tau-llm`](https://crates.io/crates/tau-llm). Rust port of
[`@earendil-works/pi-agent-core`](https://github.com/earendil-works/pi/tree/main/packages/agent);
part of [tau](https://github.com/nivalume/pi-rs).

Import it as `tau_agent`.

- Tool execution, parallel by default or sequential, with argument validation and `before`/`after` hooks
- Steering and follow-up queues
- Lifecycle events (`agent_start`, `turn_start`, `message_update`, `tool_execution_end`, ...)
- App-defined transcript messages through a type parameter
- A low-level loop (`agent_loop`) for callers that manage state themselves, and `stream_proxy` for
  routing model calls through your own server

```rust
use std::sync::Arc;
use tau_agent::{Agent, AgentEvent, AgentInitialState, AgentOptions, AgentToolResult, models_stream_fn, typed_tool};

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct GetTime {
    /// IANA timezone name.
    timezone: Option<String>,
}

let get_time = typed_tool("get_time", "Get the current time", |args: GetTime, _ctx| async move {
    Ok(AgentToolResult::text(format!("12:00 in {}", args.timezone.unwrap_or_else(|| "UTC".into()))))
});

let agent: Agent = Agent::new(AgentOptions {
    initial_state: AgentInitialState {
        system_prompt: Some("You are a helpful assistant.".into()),
        model: Some(model),
        tools: vec![Arc::new(get_time)],
        ..Default::default()
    },
    ..AgentOptions::new(models_stream_fn(models))
});

agent.subscribe(|event, _cancel| async move {
    if let AgentEvent::MessageUpdate { message, .. } = &*event {
        println!("{}", message.text());
    }
});

agent.prompt_text("What time is it in Tokyo?", Vec::new()).await?;
```

Design notes (Chinese) are in [`docs/design/`](docs/design/README.md).

## License

MIT
