//! The JSON-RPC envelope as Codex's app-server speaks it: JSON-RPC 2.0 shapes without the
//! `"jsonrpc"` field, one message per line.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A request's id: a string or an integer, as the client chose.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum RequestId {
    String(String),
    Integer(i64),
}

impl From<i64> for RequestId {
    fn from(id: i64) -> Self {
        Self::Integer(id)
    }
}

impl From<&str> for RequestId {
    fn from(id: &str) -> Self {
        Self::String(id.to_owned())
    }
}

/// A request: a method that gets a response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct JsonRpcRequest {
    pub id: RequestId,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

/// A notification: a method without a response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct JsonRpcNotification {
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

/// A successful response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct JsonRpcResponse {
    pub id: RequestId,
    pub result: Value,
}

/// A failed request's error.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct JsonRpcErrorBody {
    pub code: i64,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// A failed response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct JsonRpcError {
    pub id: RequestId,
    pub error: JsonRpcErrorBody,
}

/// Any message on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum JsonRpcMessage {
    Request(JsonRpcRequest),
    Notification(JsonRpcNotification),
    Response(JsonRpcResponse),
    Error(JsonRpcError),
}

/// The request does not parse as JSON-RPC.
pub const PARSE_ERROR: i64 = -32700;
/// The request is not valid here (before `initialize`, a stale turn id, ...).
pub const INVALID_REQUEST: i64 = -32600;
/// No such method.
pub const METHOD_NOT_FOUND: i64 = -32601;
/// The params do not parse.
pub const INVALID_PARAMS: i64 = -32602;
/// The server failed.
pub const INTERNAL_ERROR: i64 = -32603;
