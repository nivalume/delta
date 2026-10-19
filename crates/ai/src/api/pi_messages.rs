//! `pi-messages` API: streams pi's own message protocol directly to a backend. Port of
//! upstream `api/pi-messages.ts`.
//!
//! The request is a single `POST <baseUrl>/messages` of `{model, context, options}`; the
//! response is an SSE stream of serialized assistant-message events ([`PiMessagesEvent`])
//! ending with a `done` or `error` event. This is the wire protocol of the Radius gateway,
//! but any backend that implements it works, e.g. a custom provider whose models use
//! `"api": "pi-messages"`.
//!
//! The context is sent as delta serializes it, which is upstream's JSON shape. It is not
//! transformed: the backend sees the transcript exactly as the caller built it.

use futures::StreamExt;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::api::common;
use crate::error::{Error, Result};
use crate::events::{AssistantMessageEventStream, MessageBuilder};
use crate::http;
use crate::model::{CacheRetention, Model};
use crate::options::{ApiOptions, PiMessagesOptions, SimpleStreamOptions, StreamOptions, ToolChoice};
use crate::provider::ProviderStreams;
use crate::types::{
    AssistantBlock, AssistantMessageDiagnostic, DiagnosticErrorInfo, JsonObject, StopReason, ToolCall,
    TranscriptContext, Usage, now_ms,
};

/// Longest response body kept in a failure diagnostic, in characters.
const MAX_DIAGNOSTIC_STRING: usize = 8192;

/// The `pi-messages` API.
#[derive(Clone, Copy, Debug, Default)]
pub struct PiMessagesApi;

impl ProviderStreams for PiMessagesApi {
    fn stream(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        stream(model, context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        stream_simple(model, context, options)
    }
}

// ---------------------------------------------------------------------------------------------
// Wire events
// ---------------------------------------------------------------------------------------------

/// Impact summary of a server-side message rewrite (e.g. a gateway policy).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiMessagesRewriteImpact {
    pub policy_id: String,
    pub policy_version: i64,
    pub changed: bool,
    pub token_count_change: i64,
    pub message_count_change: i64,
    pub system_prompt_changed: bool,
    /// Fields this version does not know; kept in the diagnostic.
    #[serde(flatten)]
    pub extra: JsonObject,
}

/// Serialized assistant-message event as sent by a pi-messages backend.
///
/// `done` reasons are `stop`, `length` or `toolUse`; `error` reasons are `aborted` or
/// `error`. A backend written in Rust can serialize these values to produce the wire format.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PiMessagesEvent {
    Start,
    #[serde(rename_all = "camelCase")]
    TextStart {
        content_index: usize,
    },
    #[serde(rename_all = "camelCase")]
    TextDelta {
        content_index: usize,
        delta: String,
    },
    #[serde(rename_all = "camelCase")]
    TextEnd {
        content_index: usize,
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content_signature: Option<String>,
    },
    #[serde(rename_all = "camelCase")]
    ThinkingStart {
        content_index: usize,
    },
    #[serde(rename_all = "camelCase")]
    ThinkingDelta {
        content_index: usize,
        delta: String,
    },
    #[serde(rename_all = "camelCase")]
    ThinkingEnd {
        content_index: usize,
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content_signature: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        redacted: Option<bool>,
    },
    #[serde(rename = "toolcall_start", rename_all = "camelCase")]
    ToolCallStart {
        content_index: usize,
        id: String,
        tool_name: String,
    },
    #[serde(rename = "toolcall_delta", rename_all = "camelCase")]
    ToolCallDelta {
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "toolcall_end", rename_all = "camelCase")]
    ToolCallEnd {
        content_index: usize,
        /// Serialized as a content block (`{"type": "toolCall", ...}`).
        #[serde(with = "tool_call_block")]
        tool_call: ToolCall,
    },
    #[serde(rename_all = "camelCase")]
    Done {
        reason: StopReason,
        #[serde(default)]
        usage: Usage,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        response_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_thinking_level: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rewrite: Option<PiMessagesRewriteImpact>,
    },
    #[serde(rename_all = "camelCase")]
    Error {
        reason: StopReason,
        #[serde(default)]
        usage: Usage,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error_message: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        response_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_thinking_level: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rewrite: Option<PiMessagesRewriteImpact>,
    },
}

/// Event types of [`PiMessagesEvent`]. Other types are skipped.
const EVENT_TYPES: [&str; 12] = [
    "start",
    "text_start",
    "text_delta",
    "text_end",
    "thinking_start",
    "thinking_delta",
    "thinking_end",
    "toolcall_start",
    "toolcall_delta",
    "toolcall_end",
    "done",
    "error",
];

/// A tool call on the wire is a content block: it carries `"type": "toolCall"`.
mod tool_call_block {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use crate::types::{AssistantBlock, ToolCall};

    pub(super) fn serialize<S: Serializer>(call: &ToolCall, serializer: S) -> Result<S::Ok, S::Error> {
        AssistantBlock::ToolCall(call.clone()).serialize(serializer)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<ToolCall, D::Error> {
        ToolCall::deserialize(deserializer)
    }
}

// ---------------------------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------------------------

/// Stream a response from a pi-messages backend (`POST <baseUrl>/messages`).
pub fn stream(model: &Model, context: &TranscriptContext, options: StreamOptions) -> AssistantMessageEventStream {
    let context = context.clone();
    let request_model = model.clone();
    common::spawn_stream(model, options.cancel.clone(), move |mut builder| async move {
        let result = match run(&request_model, &context, &options, &mut builder).await {
            Ok(()) => Ok(()),
            Err(failure) => {
                // Upstream reports thrown errors with a fresh message: no partial content.
                builder.output = common::new_output(&request_model);
                let (error, diagnostic) = match failure {
                    Failure::Response { message, diagnostic } => (Error::msg(message), Some(diagnostic)),
                    Failure::Other(error) => (error, None),
                };
                if let Some(diagnostic) = diagnostic
                    && !options.is_cancelled()
                {
                    builder.output.push_diagnostic(*diagnostic);
                }
                Err(error)
            }
        };
        (builder, result)
    })
}

/// Stream with simple options. `reasoning` and `tool_choice` are forwarded as they are; the
/// backend applies the model's thinking levels. `debug` is kept from
/// [`ApiOptions::PiMessages`] when present.
#[allow(clippy::needless_pass_by_value)] // The signature is shared by every API module.
pub fn stream_simple(
    model: &Model,
    context: &TranscriptContext,
    options: SimpleStreamOptions,
) -> AssistantMessageEventStream {
    let SimpleStreamOptions {
        mut base,
        tool_choice,
        reasoning,
        ..
    } = options;
    let debug = match &base.api {
        Some(ApiOptions::PiMessages(options)) => options.debug,
        _ => None,
    };
    base.api = Some(ApiOptions::PiMessages(PiMessagesOptions {
        reasoning,
        tool_choice: tool_choice.map(|choice| {
            Value::from(match choice {
                ToolChoice::Auto => "auto",
                ToolChoice::None => "none",
            })
        }),
        debug,
    }));
    stream(model, context, base)
}

/// Why a request failed before a terminal event.
enum Failure {
    /// A non-2xx response (upstream `PiMessagesResponseError`).
    Response {
        message: String,
        diagnostic: Box<AssistantMessageDiagnostic>,
    },
    Other(Error),
}

impl From<Error> for Failure {
    fn from(error: Error) -> Self {
        Self::Other(error)
    }
}

impl From<serde_json::Error> for Failure {
    fn from(error: serde_json::Error) -> Self {
        Self::Other(error.into())
    }
}

async fn run(
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
    builder: &mut MessageBuilder,
) -> std::result::Result<(), Failure> {
    let Some(api_key) = options.api_key.as_deref().filter(|key| !key.is_empty()) else {
        return Err(Error::msg(format!("No API key provided for provider \"{}\"", model.provider)).into());
    };
    let pi = match &options.api {
        Some(ApiOptions::PiMessages(pi)) => pi.clone(),
        _ => PiMessagesOptions::default(),
    };

    let url = messages_url(model, pi.debug == Some(true))?;
    let payload = build_payload(model, context, options, &pi);
    let payload = common::apply_on_payload(options, model, payload).await?;

    let bearer = format!("Bearer {api_key}");
    let defaults = http::headers_of([
        ("authorization", bearer.as_str()),
        ("accept", "text/event-stream"),
        ("content-type", "application/json"),
    ]);
    let headers = http::build_headers([Some(&defaults), options.headers.as_ref()].into_iter().flatten())?;
    // One request without client retries or timeout, like upstream's plain `fetch`; only
    // `cancel` stops it.
    let request = http::client(options)
        .post(url.clone())
        .headers(headers)
        .body(payload.to_string());
    let cancel = options.cancel.as_ref();
    let response = http::or_cancel(cancel, async { request.send().await.map_err(Error::from) }).await?;
    common::notify_response(options, model, &response).await?;

    let status = response.status();
    if !status.is_success() {
        let body = http::or_cancel(cancel, async { response.text().await.map_err(Error::from) }).await?;
        return Err(response_failure(model, &url, status, &body));
    }

    let mut events = std::pin::pin!(http::sse_events(response, options.cancel.clone()));
    let mut converter = EventConverter::default();
    while let Some(event) = events.next().await {
        if builder.is_closed() {
            return Err(Error::Aborted.into());
        }
        let event = event?;
        let data = event.data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let value: Value = serde_json::from_str(data)?;
        common::observe_provider_event(options, model, &value).await?;
        if converter.apply(value, builder)? {
            // `done` and `error` end the stream through `spawn_stream` with the stop reason.
            return Ok(());
        }
    }
    Err(Error::msg(format!("{} stream ended without a terminal event", model.provider)).into())
}

/// `<baseUrl>/messages`, with `debug=1` when requested.
fn messages_url(model: &Model, debug: bool) -> Result<url::Url> {
    let raw = format!("{}/messages", model.base_url.trim_end_matches('/'));
    let mut url = url::Url::parse(&raw).map_err(|error| Error::msg(format!("Invalid URL: {raw} ({error})")))?;
    if debug {
        // `URLSearchParams.set`: replace any existing `debug` parameter.
        let pairs: Vec<(String, String)> = url
            .query_pairs()
            .filter(|(name, _)| name != "debug")
            .map(|(name, value)| (name.into_owned(), value.into_owned()))
            .collect();
        url.query_pairs_mut()
            .clear()
            .extend_pairs(pairs)
            .append_pair("debug", "1");
    }
    Ok(url)
}

/// `{model, context, options}`. Unset options are left out, as `JSON.stringify` drops
/// `undefined`.
fn build_payload(model: &Model, context: &TranscriptContext, options: &StreamOptions, pi: &PiMessagesOptions) -> Value {
    let mut request_options = JsonObject::new();
    let mut set = |name: &str, value: Option<Value>| {
        if let Some(value) = value {
            request_options.insert(name.to_owned(), value);
        }
    };
    set("temperature", options.temperature.map(Value::from));
    set("maxTokens", options.max_tokens.map(Value::from));
    set("reasoning", pi.reasoning.map(|level| json!(level)));
    set(
        "cacheRetention",
        resolve_cache_retention(options).map(|retention| json!(retention)),
    );
    set("sessionId", options.session_id.clone().map(Value::from));
    set("toolChoice", pi.tool_choice.clone());
    json!({
        "model": model.id,
        "context": context,
        "options": request_options,
    })
}

/// The requested retention. When unset the backend default applies; only the legacy
/// `PI_CACHE_RETENTION=long` opt-in is mapped.
fn resolve_cache_retention(options: &StreamOptions) -> Option<CacheRetention> {
    if options.cache_retention.is_some() {
        return options.cache_retention;
    }
    (options.env_var("PI_CACHE_RETENTION").as_deref() == Some("long")).then_some(CacheRetention::Long)
}

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// The body's `error` object, when the body is JSON with one (`parsePiMessagesErrorBody`).
fn parse_error_body(body: &str) -> Option<JsonObject> {
    match serde_json::from_str::<Value>(body) {
        Ok(Value::Object(mut parsed)) => match parsed.remove("error") {
            Some(Value::Object(error)) => Some(error),
            _ => None,
        },
        _ => None,
    }
}

fn truncate_diagnostic_string(value: &str) -> String {
    match value.char_indices().nth(MAX_DIAGNOSTIC_STRING) {
        Some((end, _)) => format!("{}…", &value[..end]),
        None => value.to_owned(),
    }
}

/// `<status> <statusText>: <error.message or body>[ (<error.code>)]`, with the
/// `pi_messages_response_failure` diagnostic.
fn response_failure(model: &Model, url: &url::Url, status: StatusCode, body: &str) -> Failure {
    let error = parse_error_body(body);
    let field = |name: &str| {
        error
            .as_ref()
            .and_then(|error| error.get(name))
            .and_then(Value::as_str)
            .map(str::to_owned)
    };
    let error_message = field("message");
    let code = field("code");
    let status_text = status.canonical_reason().unwrap_or("");
    let code_suffix = code
        .as_deref()
        .filter(|code| !code.is_empty())
        .map(|code| format!(" ({code})"))
        .unwrap_or_default();
    let message = format!(
        "{} {status_text}: {}{code_suffix}",
        status.as_u16(),
        error_message.as_deref().unwrap_or(body)
    );

    let mut details = JsonObject::new();
    details.insert("version".into(), json!(1));
    details.insert("provider".into(), json!(model.provider));
    details.insert("model".into(), json!(model.id));
    details.insert("url".into(), json!(url.as_str()));
    details.insert("status".into(), json!(status.as_u16()));
    details.insert("statusText".into(), json!(status_text));
    match error {
        Some(error) => {
            details.insert("error".into(), Value::Object(error));
        }
        None => {
            details.insert("body".into(), json!(truncate_diagnostic_string(body)));
        }
    }
    details.insert("timestampMs".into(), json!(now_ms()));

    let diagnostic = AssistantMessageDiagnostic {
        kind: "pi_messages_response_failure".into(),
        timestamp: now_ms(),
        error: Some(DiagnosticErrorInfo {
            name: Some("PiMessagesResponseError".into()),
            message: message.clone(),
            stack: None,
            code: code.map(Value::String),
        }),
        details: Some(details),
    };
    Failure::Response {
        message,
        diagnostic: Box::new(diagnostic),
    }
}

// ---------------------------------------------------------------------------------------------
// Event conversion
// ---------------------------------------------------------------------------------------------

/// Replays wire events through the [`MessageBuilder`]. Wire content indexes are mapped to
/// the builder's; events for unknown indexes are skipped.
#[derive(Debug, Default)]
struct EventConverter {
    indexes: std::collections::HashMap<usize, usize>,
}

impl EventConverter {
    /// Apply one wire event. Returns `true` for the terminal `done`/`error` event.
    fn apply(&mut self, value: Value, builder: &mut MessageBuilder) -> Result<bool> {
        let known = value
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| EVENT_TYPES.contains(&kind));
        if !known {
            return Ok(false);
        }
        let event: PiMessagesEvent = serde_json::from_value(value)?;
        if !matches!(event, PiMessagesEvent::Done { .. } | PiMessagesEvent::Error { .. }) {
            // Content events need a started stream even if the backend sent no `start`.
            builder.start();
        }
        match event {
            PiMessagesEvent::Start => {}
            PiMessagesEvent::TextStart { content_index } => {
                let index = builder.start_text();
                self.indexes.insert(content_index, index);
            }
            PiMessagesEvent::TextDelta { content_index, delta } => {
                if let Some(&index) = self.indexes.get(&content_index) {
                    builder.text_delta(index, &delta);
                }
            }
            PiMessagesEvent::TextEnd {
                content_index,
                content,
                content_signature,
            } => {
                if let Some(&index) = self.indexes.get(&content_index) {
                    if let Some(AssistantBlock::Text(text)) = builder.output.content.get_mut(index) {
                        text.text = content;
                        text.text_signature = content_signature;
                    }
                    builder.end_text(index);
                }
            }
            PiMessagesEvent::ThinkingStart { content_index } => {
                let index = builder.start_thinking(String::new(), false, None);
                self.indexes.insert(content_index, index);
            }
            PiMessagesEvent::ThinkingDelta { content_index, delta } => {
                if let Some(&index) = self.indexes.get(&content_index) {
                    builder.thinking_delta(index, &delta);
                }
            }
            PiMessagesEvent::ThinkingEnd {
                content_index,
                content,
                content_signature,
                redacted,
            } => {
                if let Some(&index) = self.indexes.get(&content_index) {
                    if let Some(AssistantBlock::Thinking(thinking)) = builder.output.content.get_mut(index) {
                        thinking.thinking = content;
                        thinking.thinking_signature = content_signature;
                        thinking.redacted = redacted;
                    }
                    builder.end_thinking(index);
                }
            }
            PiMessagesEvent::ToolCallStart {
                content_index,
                id,
                tool_name,
            } => {
                let index = builder.start_tool_call(id, tool_name, JsonObject::new());
                self.indexes.insert(content_index, index);
            }
            PiMessagesEvent::ToolCallDelta { content_index, delta } => {
                if let Some(&index) = self.indexes.get(&content_index) {
                    builder.tool_call_delta(index, &delta);
                }
            }
            PiMessagesEvent::ToolCallEnd {
                content_index,
                tool_call,
            } => {
                if let Some(&index) = self.indexes.get(&content_index) {
                    builder.end_tool_call_with(index, tool_call);
                }
            }
            PiMessagesEvent::Done {
                reason,
                usage,
                response_id,
                provider_thinking_level,
                rewrite,
            } => {
                finish(builder, reason, usage, response_id, provider_thinking_level, rewrite);
                return Ok(true);
            }
            PiMessagesEvent::Error {
                reason,
                usage,
                error_message,
                response_id,
                provider_thinking_level,
                rewrite,
            } => {
                finish(builder, reason, usage, response_id, provider_thinking_level, rewrite);
                builder.output.error_message = error_message;
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// Copy the terminal event's metadata onto the output.
fn finish(
    builder: &mut MessageBuilder,
    reason: StopReason,
    usage: Usage,
    response_id: Option<String>,
    provider_thinking_level: Option<String>,
    rewrite: Option<PiMessagesRewriteImpact>,
) {
    let output = &mut builder.output;
    output.stop_reason = reason;
    output.usage = usage;
    output.response_id = response_id;
    if provider_thinking_level.is_some() {
        output.provider_thinking_level = provider_thinking_level;
    }
    if let Some(rewrite) = rewrite {
        let details = match serde_json::to_value(rewrite) {
            Ok(Value::Object(details)) => details,
            _ => JsonObject::new(),
        };
        output.push_diagnostic(AssistantMessageDiagnostic {
            kind: "pi_messages_rewrite".into(),
            timestamp: now_ms(),
            error: None,
            details: Some(details),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_keep_the_wire_shape() {
        let wire = json!([
            {"type": "start"},
            {"type": "text_start", "contentIndex": 0},
            {"type": "text_delta", "contentIndex": 0, "delta": "Hi"},
            {"type": "text_end", "contentIndex": 0, "content": "Hi", "contentSignature": "sig"},
            {"type": "thinking_end", "contentIndex": 1, "content": "", "redacted": true},
            {"type": "toolcall_start", "contentIndex": 2, "id": "call_1", "toolName": "read"},
            {"type": "toolcall_end", "contentIndex": 2,
             "toolCall": {"type": "toolCall", "id": "call_1", "name": "read", "arguments": {"path": "a"}}},
            {"type": "done", "reason": "toolUse",
             "usage": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 3,
                       "cost": {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0}},
             "providerThinkingLevel": "high",
             "rewrite": {"policyId": "p", "policyVersion": 2, "changed": true, "tokenCountChange": -5,
                         "messageCountChange": 0, "systemPromptChanged": false, "note": "x"}},
            {"type": "error", "reason": "aborted",
             "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                       "cost": {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0}},
             "errorMessage": "stopped"}
        ]);
        let events: Vec<PiMessagesEvent> = serde_json::from_value(wire.clone()).unwrap();
        assert!(matches!(
            &events[7],
            PiMessagesEvent::Done { reason: StopReason::ToolUse, rewrite: Some(rewrite), .. }
                if rewrite.token_count_change == -5 && rewrite.extra["note"] == "x"
        ));
        assert_eq!(serde_json::to_value(&events).unwrap(), wire);
    }

    #[test]
    fn response_errors_format_message_and_code() {
        let model: Model = serde_json::from_value(json!({
            "id": "auto", "name": "Auto", "api": "pi-messages", "provider": "radius",
            "baseUrl": "http://localhost/v1", "reasoning": false, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 1, "maxTokens": 1
        }))
        .unwrap();
        let url = url::Url::parse("http://localhost/v1/messages").unwrap();
        let Failure::Response { message, diagnostic } = response_failure(
            &model,
            &url,
            StatusCode::UNAUTHORIZED,
            r#"{"error":{"message":"Token expired","code":"unauthorized"}}"#,
        ) else {
            panic!("expected a response failure");
        };
        assert_eq!(message, "401 Unauthorized: Token expired (unauthorized)");
        let details = diagnostic.details.unwrap();
        assert_eq!(
            details["error"],
            json!({"message": "Token expired", "code": "unauthorized"})
        );
        assert!(!details.contains_key("body"));
        assert_eq!(diagnostic.error.unwrap().code, Some(json!("unauthorized")));

        let Failure::Response { message, diagnostic } =
            response_failure(&model, &url, StatusCode::BAD_GATEWAY, "upstream down")
        else {
            panic!("expected a response failure");
        };
        assert_eq!(message, "502 Bad Gateway: upstream down");
        assert_eq!(diagnostic.details.unwrap()["body"], "upstream down");

        let long = "x".repeat(MAX_DIAGNOSTIC_STRING + 1);
        assert_eq!(
            truncate_diagnostic_string(&long),
            format!("{}…", "x".repeat(MAX_DIAGNOSTIC_STRING))
        );
    }

    #[test]
    fn debug_replaces_an_existing_parameter() {
        let mut model: Model = serde_json::from_value(json!({
            "id": "auto", "name": "Auto", "api": "pi-messages", "provider": "radius",
            "baseUrl": "http://localhost/v1//", "reasoning": false, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 1, "maxTokens": 1
        }))
        .unwrap();
        assert_eq!(
            messages_url(&model, false).unwrap().as_str(),
            "http://localhost/v1/messages"
        );
        assert_eq!(
            messages_url(&model, true).unwrap().as_str(),
            "http://localhost/v1/messages?debug=1"
        );
        model.base_url = "not a url".into();
        assert!(messages_url(&model, false).is_err());
    }
}
