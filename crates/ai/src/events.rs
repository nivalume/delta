//! The assistant message event protocol and its stream.
//!
//! Successful generation follows `Start → updates* → Done`. A failure after generation
//! starts follows `Start → updates* → Error`. Request setup may fail before generation
//! starts; the stream then contains only `Error`.
//!
//! Unlike upstream, events do not carry a shared mutable `partial` message. The consumer
//! side ([`AssistantMessageEventStream`]) reduces the events it yields into its own copy, so
//! [`AssistantMessageEventStream::partial`] is the exact message as of the last yielded event.
//! Events therefore carry the block data a reducer needs (tool call id and name at start,
//! signatures at end).

use futures::Stream;
use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};
use std::time::Instant;
use tokio::sync::mpsc;

use crate::json::{parse_streaming_json, parse_tool_call_arguments};
use crate::types::{
    AssistantBlock, AssistantMessage, JsonObject, StopReason, TextContent, ThinkingContent, ToolCall, now_ms,
};

/// Stop reasons of a successful stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoneReason {
    Stop,
    Length,
    ToolUse,
    Deferred,
}

impl From<DoneReason> for StopReason {
    fn from(reason: DoneReason) -> Self {
        match reason {
            DoneReason::Stop => Self::Stop,
            DoneReason::Length => Self::Length,
            DoneReason::ToolUse => Self::ToolUse,
            DoneReason::Deferred => Self::Deferred,
        }
    }
}

impl DoneReason {
    /// The done reason for a stop reason, or `None` for `pending`, `error` and `aborted`.
    pub fn from_stop_reason(reason: StopReason) -> Option<Self> {
        match reason {
            StopReason::Stop => Some(Self::Stop),
            StopReason::Length => Some(Self::Length),
            StopReason::ToolUse => Some(Self::ToolUse),
            StopReason::Deferred => Some(Self::Deferred),
            StopReason::Pending | StopReason::Error | StopReason::Aborted => None,
        }
    }
}

/// Stop reasons of a failed stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorReason {
    Error,
    Aborted,
}

impl From<ErrorReason> for StopReason {
    fn from(reason: ErrorReason) -> Self {
        match reason {
            ErrorReason::Error => Self::Error,
            ErrorReason::Aborted => Self::Aborted,
        }
    }
}

/// One event of an assistant response.
///
/// Events for different content blocks may interleave; use `content_index` to associate
/// each delta/end event with its block.
#[derive(Clone, Debug, PartialEq)]
pub enum AssistantMessageEvent {
    /// Generation started. `message` holds the response metadata, normally with no content.
    Start {
        message: AssistantMessage,
    },
    /// A text block starts (empty).
    TextStart {
        content_index: usize,
    },
    TextDelta {
        content_index: usize,
        delta: String,
    },
    /// A text block is complete. `content` is authoritative.
    TextEnd {
        content_index: usize,
        content: String,
        signature: Option<String>,
    },
    /// A thinking block starts. Ordinary thinking starts empty; redacted thinking may be
    /// complete here (`thinking` set) and emit no deltas.
    ThinkingStart {
        content_index: usize,
        thinking: String,
        redacted: bool,
        signature: Option<String>,
    },
    ThinkingDelta {
        content_index: usize,
        delta: String,
    },
    /// A thinking block is complete. `content` and `signature` are authoritative.
    ThinkingEnd {
        content_index: usize,
        content: String,
        signature: Option<String>,
    },
    /// A tool call starts. `arguments` is provider-specific (often empty).
    ToolCallStart {
        content_index: usize,
        id: String,
        name: String,
        arguments: JsonObject,
    },
    /// A chunk of the tool call's JSON arguments.
    ToolCallDelta {
        content_index: usize,
        delta: String,
    },
    /// The complete, but not schema-validated, tool call.
    ToolCallEnd {
        content_index: usize,
        tool_call: ToolCall,
    },
    /// Terminal: the final message.
    Done {
        reason: DoneReason,
        message: AssistantMessage,
    },
    /// Terminal: the failed message with any partial content, `error_message` set.
    Error {
        reason: ErrorReason,
        error: AssistantMessage,
    },
}

impl AssistantMessageEvent {
    /// Upstream event type name (`text_delta`, `done`, ...).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Start { .. } => "start",
            Self::TextStart { .. } => "text_start",
            Self::TextDelta { .. } => "text_delta",
            Self::TextEnd { .. } => "text_end",
            Self::ThinkingStart { .. } => "thinking_start",
            Self::ThinkingDelta { .. } => "thinking_delta",
            Self::ThinkingEnd { .. } => "thinking_end",
            Self::ToolCallStart { .. } => "toolcall_start",
            Self::ToolCallDelta { .. } => "toolcall_delta",
            Self::ToolCallEnd { .. } => "toolcall_end",
            Self::Done { .. } => "done",
            Self::Error { .. } => "error",
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done { .. } | Self::Error { .. })
    }

    /// The final message of a terminal event.
    pub fn final_message(&self) -> Option<&AssistantMessage> {
        match self {
            Self::Done { message, .. } => Some(message),
            Self::Error { error, .. } => Some(error),
            _ => None,
        }
    }

    fn into_final_message(self) -> Option<AssistantMessage> {
        match self {
            Self::Done { message, .. } => Some(message),
            Self::Error { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// Producer half of an [`AssistantMessageEventStream`].
///
/// Stamps `duration_ms` on the terminal message, measured with a monotonic clock from the
/// sender's creation, unless the message already has one or its timestamp predates the
/// stream (a forwarded response that started elsewhere stays untimed).
#[derive(Debug)]
pub struct EventSender {
    tx: mpsc::UnboundedSender<AssistantMessageEvent>,
    started_at_ms: i64,
    started_at: Instant,
    done: bool,
}

impl EventSender {
    /// Send an event. Events after the terminal event are ignored.
    pub fn push(&mut self, mut event: AssistantMessageEvent) {
        if self.done {
            return;
        }
        match &mut event {
            AssistantMessageEvent::Done { message, .. } | AssistantMessageEvent::Error { error: message, .. } => {
                self.time(message);
                self.done = true;
            }
            _ => {}
        }
        let _ = self.tx.send(event);
    }

    fn time(&self, message: &mut AssistantMessage) {
        if message.duration_ms.is_none() && message.timestamp >= self.started_at_ms {
            message.duration_ms = Some(self.started_at.elapsed().as_millis() as u64);
        }
    }

    /// Whether the consumer dropped the stream. Producers stop work early when it did.
    pub fn is_closed(&self) -> bool {
        self.tx.is_closed()
    }

    /// Whether a terminal event was sent.
    pub fn is_done(&self) -> bool {
        self.done
    }
}

/// Stream of [`AssistantMessageEvent`]s of one response.
///
/// Iterate it with [`futures::StreamExt::next`], or await [`Self::result`] for the final
/// message. Dropping the stream tells the producer to stop; built-in API implementations
/// cancel their HTTP request.
#[derive(Debug)]
pub struct AssistantMessageEventStream {
    rx: mpsc::UnboundedReceiver<AssistantMessageEvent>,
    partial: Option<AssistantMessage>,
    tool_json: HashMap<usize, String>,
    result: Option<AssistantMessage>,
    terminated: bool,
}

impl AssistantMessageEventStream {
    /// A connected sender/stream pair.
    pub fn channel() -> (EventSender, Self) {
        let (tx, rx) = mpsc::unbounded_channel();
        let sender = EventSender {
            tx,
            started_at_ms: now_ms(),
            started_at: Instant::now(),
            done: false,
        };
        let stream = Self {
            rx,
            partial: None,
            tool_json: HashMap::new(),
            result: None,
            terminated: false,
        };
        (sender, stream)
    }

    /// A stream that holds a single terminal event.
    pub fn from_final(message: AssistantMessage) -> Self {
        let (mut tx, stream) = Self::channel();
        let event = match DoneReason::from_stop_reason(message.stop_reason) {
            Some(reason) => AssistantMessageEvent::Done { reason, message },
            None => AssistantMessageEvent::Error {
                reason: if message.stop_reason == StopReason::Aborted {
                    ErrorReason::Aborted
                } else {
                    ErrorReason::Error
                },
                error: message,
            },
        };
        tx.push(event);
        stream
    }

    /// The response so far, as of the last event this stream yielded. `None` before `Start`.
    ///
    /// Tool call arguments hold the best-effort parse of the JSON received so far.
    pub fn partial(&self) -> Option<&AssistantMessage> {
        self.result.as_ref().or(self.partial.as_ref())
    }

    /// Drain the stream and return the final message. A producer that ends without a
    /// terminal event yields an error message.
    pub async fn result(mut self) -> AssistantMessage {
        if let Some(result) = self.result.take() {
            return result;
        }
        while let Some(event) = self.rx.recv().await {
            if let Some(message) = event.into_final_message() {
                return message;
            }
        }
        let mut message = self.partial.take().unwrap_or_default();
        message.stop_reason = StopReason::Error;
        message.error_message = Some("Stream ended without a final message".into());
        message
    }

    fn apply(&mut self, event: &AssistantMessageEvent) {
        use AssistantMessageEvent as E;
        if let Some(message) = event.final_message() {
            self.result = Some(message.clone());
            self.partial = None;
            self.tool_json.clear();
            return;
        }
        if let E::Start { message } = event {
            self.partial = Some(message.clone());
            return;
        }
        let Some(partial) = self.partial.as_mut() else { return };
        let content = &mut partial.content;
        match event {
            E::TextStart { content_index } => {
                set_block(content, *content_index, AssistantBlock::Text(TextContent::default()));
            }
            E::TextDelta { content_index, delta } => {
                if let Some(AssistantBlock::Text(text)) = content.get_mut(*content_index) {
                    text.text.push_str(delta);
                }
            }
            E::TextEnd {
                content_index,
                content: text,
                signature,
            } => {
                set_block(
                    content,
                    *content_index,
                    AssistantBlock::Text(TextContent {
                        text: text.clone(),
                        text_signature: signature.clone(),
                    }),
                );
            }
            E::ThinkingStart {
                content_index,
                thinking,
                redacted,
                signature,
            } => {
                set_block(
                    content,
                    *content_index,
                    AssistantBlock::Thinking(ThinkingContent {
                        thinking: thinking.clone(),
                        thinking_signature: signature.clone(),
                        redacted: redacted.then_some(true),
                    }),
                );
            }
            E::ThinkingDelta { content_index, delta } => {
                if let Some(AssistantBlock::Thinking(thinking)) = content.get_mut(*content_index) {
                    thinking.thinking.push_str(delta);
                }
            }
            E::ThinkingEnd {
                content_index,
                content: text,
                signature,
            } => {
                if let Some(AssistantBlock::Thinking(thinking)) = content.get_mut(*content_index) {
                    thinking.thinking.clone_from(text);
                    if signature.is_some() {
                        thinking.thinking_signature.clone_from(signature);
                    }
                }
            }
            E::ToolCallStart {
                content_index,
                id,
                name,
                arguments,
            } => {
                set_block(
                    content,
                    *content_index,
                    AssistantBlock::ToolCall(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                        ..ToolCall::default()
                    }),
                );
            }
            E::ToolCallDelta { content_index, delta } => {
                let buffer = self.tool_json.entry(*content_index).or_default();
                buffer.push_str(delta);
                if let Some(AssistantBlock::ToolCall(call)) = content.get_mut(*content_index) {
                    call.arguments = parse_streaming_json(buffer);
                }
            }
            E::ToolCallEnd {
                content_index,
                tool_call,
            } => {
                self.tool_json.remove(content_index);
                set_block(content, *content_index, AssistantBlock::ToolCall(tool_call.clone()));
            }
            E::Start { .. } | E::Done { .. } | E::Error { .. } => {}
        }
    }
}

fn set_block(content: &mut Vec<AssistantBlock>, index: usize, block: AssistantBlock) {
    if index < content.len() {
        content[index] = block;
    } else {
        while content.len() < index {
            content.push(AssistantBlock::Text(TextContent::default()));
        }
        content.push(block);
    }
}

impl Stream for AssistantMessageEventStream {
    type Item = AssistantMessageEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        if self.terminated {
            return Poll::Ready(None);
        }
        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(event)) => {
                self.apply(&event);
                if event.is_terminal() {
                    self.terminated = true;
                }
                Poll::Ready(Some(event))
            }
            Poll::Ready(None) => {
                self.terminated = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Producer-side helper that owns the response being built and emits matching events.
///
/// API implementations mutate the message only through this builder (plus metadata such as
/// usage and response ids, which do not produce events), so the consumer's reduced copy
/// stays in step.
#[derive(Debug)]
pub struct MessageBuilder {
    sender: EventSender,
    /// The response being built.
    pub output: AssistantMessage,
    tool_json: HashMap<usize, String>,
    started: bool,
}

impl MessageBuilder {
    pub fn new(sender: EventSender, output: AssistantMessage) -> Self {
        Self {
            sender,
            output,
            tool_json: HashMap::new(),
            started: false,
        }
    }

    /// Whether the consumer dropped the stream.
    pub fn is_closed(&self) -> bool {
        self.sender.is_closed()
    }

    pub fn is_started(&self) -> bool {
        self.started
    }

    /// Emit `Start` once.
    pub fn start(&mut self) {
        if !self.started {
            self.started = true;
            self.sender.push(AssistantMessageEvent::Start {
                message: self.output.clone(),
            });
        }
    }

    pub fn start_text(&mut self) -> usize {
        let index = self.output.content.len();
        self.output.content.push(AssistantBlock::Text(TextContent::default()));
        self.sender
            .push(AssistantMessageEvent::TextStart { content_index: index });
        index
    }

    pub fn text_delta(&mut self, index: usize, delta: &str) {
        if delta.is_empty() {
            return;
        }
        if let Some(AssistantBlock::Text(text)) = self.output.content.get_mut(index) {
            text.text.push_str(delta);
            self.sender.push(AssistantMessageEvent::TextDelta {
                content_index: index,
                delta: delta.to_owned(),
            });
        }
    }

    pub fn set_text_signature(&mut self, index: usize, signature: Option<String>) {
        if let Some(AssistantBlock::Text(text)) = self.output.content.get_mut(index) {
            text.text_signature = signature;
        }
    }

    pub fn end_text(&mut self, index: usize) {
        if let Some(AssistantBlock::Text(text)) = self.output.content.get(index) {
            self.sender.push(AssistantMessageEvent::TextEnd {
                content_index: index,
                content: text.text.clone(),
                signature: text.text_signature.clone(),
            });
        }
    }

    /// Start a thinking block. Redacted thinking passes its complete text and signature here.
    pub fn start_thinking(&mut self, thinking: String, redacted: bool, signature: Option<String>) -> usize {
        let index = self.output.content.len();
        self.output.content.push(AssistantBlock::Thinking(ThinkingContent {
            thinking: thinking.clone(),
            thinking_signature: signature.clone(),
            redacted: redacted.then_some(true),
        }));
        self.sender.push(AssistantMessageEvent::ThinkingStart {
            content_index: index,
            thinking,
            redacted,
            signature,
        });
        index
    }

    pub fn thinking_delta(&mut self, index: usize, delta: &str) {
        if delta.is_empty() {
            return;
        }
        if let Some(AssistantBlock::Thinking(thinking)) = self.output.content.get_mut(index) {
            thinking.thinking.push_str(delta);
            self.sender.push(AssistantMessageEvent::ThinkingDelta {
                content_index: index,
                delta: delta.to_owned(),
            });
        }
    }

    pub fn append_thinking_signature(&mut self, index: usize, signature: &str) {
        if let Some(AssistantBlock::Thinking(thinking)) = self.output.content.get_mut(index) {
            thinking
                .thinking_signature
                .get_or_insert_with(String::new)
                .push_str(signature);
        }
    }

    pub fn set_thinking_signature(&mut self, index: usize, signature: Option<String>) {
        if let Some(AssistantBlock::Thinking(thinking)) = self.output.content.get_mut(index) {
            thinking.thinking_signature = signature;
        }
    }

    pub fn end_thinking(&mut self, index: usize) {
        if let Some(AssistantBlock::Thinking(thinking)) = self.output.content.get(index) {
            self.sender.push(AssistantMessageEvent::ThinkingEnd {
                content_index: index,
                content: thinking.thinking.clone(),
                signature: thinking.thinking_signature.clone(),
            });
        }
    }

    /// Start a tool call with provider-specific initial arguments.
    pub fn start_tool_call(&mut self, id: String, name: String, arguments: JsonObject) -> usize {
        let index = self.output.content.len();
        self.output.content.push(AssistantBlock::ToolCall(ToolCall {
            id: id.clone(),
            name: name.clone(),
            arguments: arguments.clone(),
            ..ToolCall::default()
        }));
        self.tool_json.insert(index, String::new());
        self.sender.push(AssistantMessageEvent::ToolCallStart {
            content_index: index,
            id,
            name,
            arguments,
        });
        index
    }

    /// Mutable access to a tool call, e.g. to set its id, signature or namespace.
    pub fn tool_call_mut(&mut self, index: usize) -> Option<&mut ToolCall> {
        match self.output.content.get_mut(index) {
            Some(AssistantBlock::ToolCall(call)) => Some(call),
            _ => None,
        }
    }

    /// Append a chunk of JSON arguments.
    pub fn tool_call_delta(&mut self, index: usize, delta: &str) {
        if delta.is_empty() {
            return;
        }
        if let Some(buffer) = self.tool_json.get_mut(&index) {
            buffer.push_str(delta);
            self.sender.push(AssistantMessageEvent::ToolCallDelta {
                content_index: index,
                delta: delta.to_owned(),
            });
        }
    }

    /// The JSON arguments received so far.
    pub fn tool_call_json(&self, index: usize) -> Option<&str> {
        self.tool_json.get(&index).map(String::as_str)
    }

    /// Finish a tool call. Arguments are parsed from the streamed JSON when any arrived;
    /// otherwise the start arguments stay.
    pub fn end_tool_call(&mut self, index: usize) {
        let json = self.tool_json.remove(&index);
        let Some(AssistantBlock::ToolCall(call)) = self.output.content.get_mut(index) else {
            return;
        };
        if let Some(json) = json.filter(|json| !json.trim().is_empty()) {
            let (arguments, error) = parse_tool_call_arguments(&json);
            call.arguments = arguments;
            call.arguments_error = error;
        }
        let tool_call = call.clone();
        self.sender.push(AssistantMessageEvent::ToolCallEnd {
            content_index: index,
            tool_call,
        });
    }

    /// Finish a tool call with a complete call from elsewhere (e.g. a proxy server),
    /// ignoring any streamed JSON.
    pub fn end_tool_call_with(&mut self, index: usize, tool_call: ToolCall) {
        self.tool_json.remove(&index);
        let Some(AssistantBlock::ToolCall(call)) = self.output.content.get_mut(index) else {
            return;
        };
        *call = tool_call.clone();
        self.sender.push(AssistantMessageEvent::ToolCallEnd {
            content_index: index,
            tool_call,
        });
    }

    /// End every block that is still open (tool calls with pending JSON).
    pub fn end_open_tool_calls(&mut self) {
        let mut open: Vec<usize> = self.tool_json.keys().copied().collect();
        open.sort_unstable();
        for index in open {
            self.end_tool_call(index);
        }
    }

    /// Emit `Done` with the output's stop reason (`stop` when still pending).
    pub fn finish(mut self) {
        self.start();
        self.tool_json.clear();
        if self.output.stop_reason == StopReason::Pending {
            self.output.stop_reason = StopReason::Stop;
        }
        match DoneReason::from_stop_reason(self.output.stop_reason) {
            Some(reason) => self.sender.push(AssistantMessageEvent::Done {
                reason,
                message: self.output,
            }),
            None => {
                let reason = if self.output.stop_reason == StopReason::Aborted {
                    ErrorReason::Aborted
                } else {
                    ErrorReason::Error
                };
                self.sender.push(AssistantMessageEvent::Error {
                    reason,
                    error: self.output,
                });
            }
        }
    }

    /// Emit `Error` with `message` (`aborted` when `aborted`).
    pub fn fail(mut self, message: impl Into<String>, aborted: bool) {
        self.tool_json.clear();
        let reason = if aborted {
            ErrorReason::Aborted
        } else {
            ErrorReason::Error
        };
        self.output.stop_reason = reason.into();
        self.output.error_message = Some(message.into());
        self.sender.push(AssistantMessageEvent::Error {
            reason,
            error: self.output,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use serde_json::json;

    #[tokio::test]
    async fn partial_tracks_events_and_result_is_final() {
        let (sender, mut stream) = AssistantMessageEventStream::channel();
        let mut builder = MessageBuilder::new(sender, AssistantMessage::empty("api", "p", "m"));
        builder.start();
        let text = builder.start_text();
        builder.text_delta(text, "Hel");
        let call = builder.start_tool_call("c1".into(), "write".into(), JsonObject::new());
        builder.text_delta(text, "lo");
        builder.tool_call_delta(call, r#"{"path": "a.t"#);
        builder.tool_call_delta(call, r#"xt"}"#);
        builder.end_text(text);
        builder.end_tool_call(call);
        builder.output.stop_reason = StopReason::ToolUse;
        builder.finish();

        let mut kinds = Vec::new();
        while let Some(event) = stream.next().await {
            kinds.push(event.kind());
            if let AssistantMessageEvent::ToolCallDelta { content_index, .. } = &event {
                let partial = stream.partial().unwrap();
                let call = partial.content[*content_index].as_tool_call().unwrap();
                assert!(call.arguments["path"].as_str().unwrap().starts_with("a.t"));
            }
        }
        assert_eq!(
            kinds,
            [
                "start",
                "text_start",
                "text_delta",
                "toolcall_start",
                "text_delta",
                "toolcall_delta",
                "toolcall_delta",
                "text_end",
                "toolcall_end",
                "done"
            ]
        );
        let message = stream.partial().unwrap();
        assert_eq!(message.text(), "Hello");
        assert_eq!(
            serde_json::Value::Object(message.tool_calls().next().unwrap().arguments.clone()),
            json!({"path": "a.txt"})
        );
        assert!(message.duration_ms.is_some());
    }

    #[tokio::test]
    async fn result_without_terminal_event_is_error() {
        let (sender, stream) = AssistantMessageEventStream::channel();
        drop(sender);
        let message = stream.result().await;
        assert_eq!(message.stop_reason, StopReason::Error);
    }
}
