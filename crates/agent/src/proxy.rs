//! Stream function for apps that route model calls through their own server. The server
//! holds the provider credentials and streams events with the partial message stripped;
//! the client rebuilds the message.

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use tau_ai::events::MessageBuilder;
use tau_ai::http::sse_events;
use tau_ai::options::ThinkingBudgets;
use tau_ai::{
    AssistantMessage, AssistantMessageEventStream, Model, SimpleStreamOptions, StopReason, ToolCall, TranscriptContext,
    Usage,
};

/// Events the proxy server sends.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProxyAssistantMessageEvent {
    Start,
    TextStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
    },
    TextDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
    },
    TextEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        #[serde(rename = "contentSignature", default, skip_serializing_if = "Option::is_none")]
        content_signature: Option<String>,
    },
    ThinkingStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
    },
    ThinkingDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
    },
    ThinkingEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        #[serde(rename = "contentSignature", default, skip_serializing_if = "Option::is_none")]
        content_signature: Option<String>,
    },
    #[serde(rename = "toolcall_start")]
    ToolCallStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
    },
    #[serde(rename = "toolcall_delta")]
    ToolCallDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "toolcall_end")]
    ToolCallEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        #[serde(rename = "toolCall")]
        tool_call: ToolCall,
    },
    Done {
        reason: StopReason,
        usage: Usage,
        #[serde(rename = "providerThinkingLevel", default, skip_serializing_if = "Option::is_none")]
        provider_thinking_level: Option<String>,
    },
    Error {
        reason: StopReason,
        #[serde(rename = "errorMessage", default, skip_serializing_if = "Option::is_none")]
        error_message: Option<String>,
        usage: Usage,
        #[serde(rename = "providerThinkingLevel", default, skip_serializing_if = "Option::is_none")]
        provider_thinking_level: Option<String>,
    },
}

/// Options of [`stream_proxy`].
#[derive(Clone, Debug)]
pub struct ProxyStreamOptions {
    /// Request options; the serializable subset is sent to the server.
    pub options: SimpleStreamOptions,
    /// Bearer token of the proxy server.
    pub auth_token: String,
    /// Proxy server URL, e.g. `https://genai.example.com`.
    pub proxy_url: String,
}

fn thinking_budgets_json(budgets: Option<&ThinkingBudgets>) -> Value {
    budgets.map_or(Value::Null, |budgets| {
        json!({"minimal": budgets.minimal, "low": budgets.low, "medium": budgets.medium, "high": budgets.high})
    })
}

/// The serializable request options, as upstream sends them.
fn request_options(options: &SimpleStreamOptions) -> Value {
    let base = &options.base;
    let mut object = serde_json::Map::new();
    let mut put = |key: &str, value: Value| {
        if !value.is_null() {
            object.insert(key.to_owned(), value);
        }
    };
    put("temperature", json!(base.temperature));
    put("samplingParams", json!(base.sampling_params));
    put("maxTokens", json!(base.max_tokens));
    put("reasoning", json!(options.reasoning));
    put("cacheRetention", json!(base.cache_retention));
    put("sessionId", json!(base.session_id));
    put("headers", json!(base.headers));
    put("metadata", json!(base.metadata));
    put(
        "transport",
        base.transport.map_or(Value::Null, |transport| {
            json!(match transport {
                tau_ai::options::Transport::Sse => "sse",
                tau_ai::options::Transport::Websocket => "websocket",
                tau_ai::options::Transport::WebsocketCached => "websocket-cached",
                tau_ai::options::Transport::Auto => "auto",
            })
        }),
    );
    let budgets = thinking_budgets_json(options.thinking_budgets.as_ref());
    put(
        "thinkingBudgets",
        match budgets {
            Value::Object(map) => Value::Object(map.into_iter().filter(|(_, value)| !value.is_null()).collect()),
            other => other,
        },
    );
    put(
        "maxRetryDelayMs",
        json!(base.max_retry_delay.map(|delay| delay.as_millis() as u64)),
    );
    Value::Object(object)
}

/// Stream through a proxy server: `POST {proxy_url}/api/stream` with the model, context and
/// options, then rebuild the message from the server's SSE events.
pub fn stream_proxy(
    model: &Model,
    context: &TranscriptContext,
    options: ProxyStreamOptions,
) -> AssistantMessageEventStream {
    let (sender, stream) = AssistantMessageEventStream::channel();
    let model = model.clone();
    let context = context.clone();
    tokio::spawn(async move {
        let mut builder = MessageBuilder::new(sender, AssistantMessage::empty(&model.api, &model.provider, &model.id));
        let cancel = options.options.base.cancel.clone();
        match run(&mut builder, &model, &context, &options, cancel.as_ref()).await {
            Ok(()) => builder.finish(),
            Err(message) => {
                let aborted = cancel.as_ref().is_some_and(CancellationToken::is_cancelled);
                let message = if aborted {
                    "Request aborted by user".to_owned()
                } else {
                    message
                };
                builder.fail(message, aborted);
            }
        }
    });
    stream
}

async fn run(
    builder: &mut MessageBuilder,
    model: &Model,
    context: &TranscriptContext,
    options: &ProxyStreamOptions,
    cancel: Option<&CancellationToken>,
) -> Result<(), String> {
    let client = tau_ai::http::client(&options.options.base);
    let body = json!({"model": model, "context": context, "options": request_options(&options.options)});
    let request = client
        .post(format!("{}/api/stream", options.proxy_url.trim_end_matches('/')))
        .bearer_auth(&options.auth_token)
        .json(&body)
        .send();
    let response = match cancel {
        Some(cancel) => tokio::select! {
            () = cancel.cancelled() => return Err("Request aborted by user".into()),
            response = request => response,
        },
        None => request.await,
    }
    .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        let status = response.status();
        let mut message = format!(
            "Proxy error: {} {}",
            status.as_u16(),
            status.canonical_reason().unwrap_or("")
        );
        if let Ok(body) = response.json::<Value>().await
            && let Some(error) = body.get("error").and_then(Value::as_str)
        {
            message = format!("Proxy error: {error}");
        }
        return Err(message);
    }
    let events = sse_events(response, cancel.cloned());
    futures::pin_mut!(events);
    let mut terminal = false;
    while let Some(event) = events.next().await {
        let event = event.map_err(|error| error.to_string())?;
        if builder.is_closed() {
            return Ok(());
        }
        let data = event.data.trim();
        if data.is_empty() {
            continue;
        }
        let proxy_event: ProxyAssistantMessageEvent = serde_json::from_str(data).map_err(|error| error.to_string())?;
        if apply(builder, proxy_event)? {
            terminal = true;
            break;
        }
    }
    if !terminal {
        // A clean EOF without done/error: the server dropped the response mid-stream.
        return Err("Connection closed by proxy server before the response completed".into());
    }
    Ok(())
}

/// Apply one proxy event; `Ok(true)` after a terminal event.
fn apply(builder: &mut MessageBuilder, event: ProxyAssistantMessageEvent) -> Result<bool, String> {
    use ProxyAssistantMessageEvent as E;
    let index_matches = |builder: &MessageBuilder, index: usize| index == builder.output.content.len();
    match event {
        E::Start => builder.start(),
        E::TextStart { content_index } => {
            if !index_matches(builder, content_index) {
                return Err(format!("Unexpected content index {content_index}"));
            }
            builder.start_text();
        }
        E::TextDelta { content_index, delta } => {
            require_kind(builder, content_index, "text", "text_delta")?;
            builder.text_delta(content_index, &delta);
        }
        E::TextEnd {
            content_index,
            content_signature,
        } => {
            require_kind(builder, content_index, "text", "text_end")?;
            builder.set_text_signature(content_index, content_signature);
            builder.end_text(content_index);
        }
        E::ThinkingStart { content_index } => {
            if !index_matches(builder, content_index) {
                return Err(format!("Unexpected content index {content_index}"));
            }
            builder.start_thinking(String::new(), false, None);
        }
        E::ThinkingDelta { content_index, delta } => {
            require_kind(builder, content_index, "thinking", "thinking_delta")?;
            builder.thinking_delta(content_index, &delta);
        }
        E::ThinkingEnd {
            content_index,
            content_signature,
        } => {
            require_kind(builder, content_index, "thinking", "thinking_end")?;
            builder.set_thinking_signature(content_index, content_signature);
            builder.end_thinking(content_index);
        }
        E::ToolCallStart {
            content_index,
            id,
            tool_name,
        } => {
            if !index_matches(builder, content_index) {
                return Err(format!("Unexpected content index {content_index}"));
            }
            builder.start_tool_call(id, tool_name, serde_json::Map::new());
        }
        E::ToolCallDelta { content_index, delta } => {
            require_kind(builder, content_index, "toolCall", "toolcall_delta")?;
            builder.tool_call_delta(content_index, &delta);
        }
        E::ToolCallEnd {
            content_index,
            tool_call,
        } => {
            if builder.tool_call_mut(content_index).is_none() {
                return Ok(false);
            }
            builder.end_tool_call_with(content_index, tool_call);
        }
        E::Done {
            reason,
            usage,
            provider_thinking_level,
        } => {
            builder.output.stop_reason = reason;
            builder.output.usage = usage;
            if provider_thinking_level.is_some() {
                builder.output.provider_thinking_level = provider_thinking_level;
            }
            return Ok(true);
        }
        E::Error {
            reason,
            error_message,
            usage,
            provider_thinking_level,
        } => {
            builder.output.stop_reason = reason;
            builder.output.error_message = error_message;
            builder.output.usage = usage;
            if provider_thinking_level.is_some() {
                builder.output.provider_thinking_level = provider_thinking_level;
            }
            return Ok(true);
        }
    }
    Ok(false)
}

fn require_kind(builder: &mut MessageBuilder, index: usize, kind: &str, event: &str) -> Result<(), String> {
    let matches = match builder.output.content.get(index) {
        Some(tau_ai::AssistantBlock::Text(_)) => kind == "text",
        Some(tau_ai::AssistantBlock::Thinking(_)) => kind == "thinking",
        Some(tau_ai::AssistantBlock::ToolCall(_)) => kind == "toolCall",
        None => false,
    };
    if matches {
        Ok(())
    } else {
        Err(format!("Received {event} for non-{kind} content"))
    }
}
