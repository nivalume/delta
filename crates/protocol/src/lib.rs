//! The UI protocol of `delta-harness`: a subset of Codex's app-server v2 protocol, so a
//! client written for Codex's app-server drives a delta harness, plus delta's own methods
//! under `delta/`. Types only; `delta-app-server` serves them.
//!
//! Messages are JSON-RPC 2.0 without the `"jsonrpc"` field, one per line. The client sends
//! [`ClientRequest`]s and [`ClientNotification`]s; the server sends
//! [`ServerNotificationEnvelope`]s and [`ServerRequest`]s. Each enum is tagged by `method`,
//! so a request's JSON is `{"method": "thread/start", "id": 1, "params": {...}}`. See
//! `docs/design/protocol.md`.

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

pub mod delta;
pub mod jsonrpc;
pub mod v2;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub use jsonrpc::{
    INTERNAL_ERROR, INVALID_PARAMS, INVALID_REQUEST, JsonRpcError, JsonRpcErrorBody, JsonRpcMessage,
    JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, METHOD_NOT_FOUND, PARSE_ERROR, RequestId,
};

use delta::{
    PermissionRulesRemoveParams, PermissionsReadParams, PermissionsUpdateParams, SnapshotNotification,
    ToolRequestApprovalParams,
};
use v2::{
    CommandExecutionRequestApprovalParams, ErrorNotification, FileChangeRequestApprovalParams, InitializeParams,
    ItemCompletedNotification, ItemDeltaNotification, ItemStartedNotification, ReasoningTextDeltaNotification,
    ServerRequestResolvedNotification, ThreadForkParams, ThreadListParams, ThreadReadParams, ThreadResumeParams,
    ThreadStartParams, ThreadStartedNotification, ThreadStatusChangedNotification, ThreadTokenUsageUpdatedNotification,
    TurnInterruptParams, TurnNotification, TurnStartParams, TurnSteerParams,
};

/// A request the client sends; the comment names the response type.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "method")]
pub enum ClientRequest {
    /// [`v2::InitializeResponse`]; the first request of a connection.
    #[serde(rename = "initialize")]
    Initialize {
        #[serde(rename = "id")]
        request_id: RequestId,
        params: InitializeParams,
    },
    /// [`v2::ThreadStartResponse`].
    #[serde(rename = "thread/start")]
    ThreadStart {
        #[serde(rename = "id")]
        request_id: RequestId,
        params: ThreadStartParams,
    },
    /// [`v2::ThreadResumeResponse`].
    #[serde(rename = "thread/resume")]
    ThreadResume {
        #[serde(rename = "id")]
        request_id: RequestId,
        params: ThreadResumeParams,
    },
    /// [`v2::ThreadForkResponse`].
    #[serde(rename = "thread/fork")]
    ThreadFork {
        #[serde(rename = "id")]
        request_id: RequestId,
        params: ThreadForkParams,
    },
    /// [`v2::ThreadListResponse`].
    #[serde(rename = "thread/list")]
    ThreadList {
        #[serde(rename = "id")]
        request_id: RequestId,
        #[serde(default)]
        params: ThreadListParams,
    },
    /// [`v2::ThreadReadResponse`].
    #[serde(rename = "thread/read")]
    ThreadRead {
        #[serde(rename = "id")]
        request_id: RequestId,
        params: ThreadReadParams,
    },
    /// [`v2::TurnStartResponse`].
    #[serde(rename = "turn/start")]
    TurnStart {
        #[serde(rename = "id")]
        request_id: RequestId,
        params: TurnStartParams,
    },
    /// [`v2::TurnSteerResponse`].
    #[serde(rename = "turn/steer")]
    TurnSteer {
        #[serde(rename = "id")]
        request_id: RequestId,
        params: TurnSteerParams,
    },
    /// [`v2::TurnInterruptResponse`].
    #[serde(rename = "turn/interrupt")]
    TurnInterrupt {
        #[serde(rename = "id")]
        request_id: RequestId,
        params: TurnInterruptParams,
    },
    /// [`delta::PermissionsResponse`].
    #[serde(rename = "delta/permissions/read")]
    PermissionsRead {
        #[serde(rename = "id")]
        request_id: RequestId,
        params: PermissionsReadParams,
    },
    /// [`delta::PermissionsResponse`].
    #[serde(rename = "delta/permissions/update")]
    PermissionsUpdate {
        #[serde(rename = "id")]
        request_id: RequestId,
        params: PermissionsUpdateParams,
    },
    /// [`delta::PermissionRulesResponse`].
    #[serde(rename = "delta/permissionRules/read")]
    PermissionRulesRead {
        #[serde(rename = "id")]
        request_id: RequestId,
    },
    /// [`delta::PermissionRulesResponse`].
    #[serde(rename = "delta/permissionRules/remove")]
    PermissionRulesRemove {
        #[serde(rename = "id")]
        request_id: RequestId,
        params: PermissionRulesRemoveParams,
    },
}

impl ClientRequest {
    pub fn id(&self) -> &RequestId {
        match self {
            Self::Initialize { request_id, .. }
            | Self::ThreadStart { request_id, .. }
            | Self::ThreadResume { request_id, .. }
            | Self::ThreadFork { request_id, .. }
            | Self::ThreadList { request_id, .. }
            | Self::ThreadRead { request_id, .. }
            | Self::TurnStart { request_id, .. }
            | Self::TurnSteer { request_id, .. }
            | Self::TurnInterrupt { request_id, .. }
            | Self::PermissionsRead { request_id, .. }
            | Self::PermissionsUpdate { request_id, .. }
            | Self::PermissionRulesRead { request_id }
            | Self::PermissionRulesRemove { request_id, .. } => request_id,
        }
    }

    pub fn method(&self) -> &'static str {
        match self {
            Self::Initialize { .. } => "initialize",
            Self::ThreadStart { .. } => "thread/start",
            Self::ThreadResume { .. } => "thread/resume",
            Self::ThreadFork { .. } => "thread/fork",
            Self::ThreadList { .. } => "thread/list",
            Self::ThreadRead { .. } => "thread/read",
            Self::TurnStart { .. } => "turn/start",
            Self::TurnSteer { .. } => "turn/steer",
            Self::TurnInterrupt { .. } => "turn/interrupt",
            Self::PermissionsRead { .. } => "delta/permissions/read",
            Self::PermissionsUpdate { .. } => "delta/permissions/update",
            Self::PermissionRulesRead { .. } => "delta/permissionRules/read",
            Self::PermissionRulesRemove { .. } => "delta/permissionRules/remove",
        }
    }
}

/// The methods [`ClientRequest`] knows; another method is answered with
/// [`METHOD_NOT_FOUND`].
pub const CLIENT_REQUEST_METHODS: [&str; 13] = [
    "initialize",
    "thread/start",
    "thread/resume",
    "thread/fork",
    "thread/list",
    "thread/read",
    "turn/start",
    "turn/steer",
    "turn/interrupt",
    "delta/permissions/read",
    "delta/permissions/update",
    "delta/permissionRules/read",
    "delta/permissionRules/remove",
];

/// A notification the client sends.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "method")]
pub enum ClientNotification {
    /// Sent once the `initialize` response arrived.
    #[serde(rename = "initialized")]
    Initialized,
}

/// A notification the server sends.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "method", content = "params")]
pub enum ServerNotification {
    #[serde(rename = "error")]
    Error(ErrorNotification),
    #[serde(rename = "thread/started")]
    ThreadStarted(ThreadStartedNotification),
    #[serde(rename = "thread/status/changed")]
    ThreadStatusChanged(ThreadStatusChangedNotification),
    #[serde(rename = "thread/tokenUsage/updated")]
    ThreadTokenUsageUpdated(ThreadTokenUsageUpdatedNotification),
    #[serde(rename = "turn/started")]
    TurnStarted(TurnNotification),
    #[serde(rename = "turn/completed")]
    TurnCompleted(TurnNotification),
    #[serde(rename = "item/started")]
    ItemStarted(ItemStartedNotification),
    #[serde(rename = "item/completed")]
    ItemCompleted(ItemCompletedNotification),
    #[serde(rename = "item/agentMessage/delta")]
    AgentMessageDelta(ItemDeltaNotification),
    #[serde(rename = "item/commandExecution/outputDelta")]
    CommandExecutionOutputDelta(ItemDeltaNotification),
    #[serde(rename = "item/reasoning/textDelta")]
    ReasoningTextDelta(ReasoningTextDeltaNotification),
    #[serde(rename = "serverRequest/resolved")]
    ServerRequestResolved(ServerRequestResolvedNotification),
    #[serde(rename = "delta/snapshot")]
    Snapshot(SnapshotNotification),
}

impl ServerNotification {
    pub fn method(&self) -> &'static str {
        match self {
            Self::Error(_) => "error",
            Self::ThreadStarted(_) => "thread/started",
            Self::ThreadStatusChanged(_) => "thread/status/changed",
            Self::ThreadTokenUsageUpdated(_) => "thread/tokenUsage/updated",
            Self::TurnStarted(_) => "turn/started",
            Self::TurnCompleted(_) => "turn/completed",
            Self::ItemStarted(_) => "item/started",
            Self::ItemCompleted(_) => "item/completed",
            Self::AgentMessageDelta(_) => "item/agentMessage/delta",
            Self::CommandExecutionOutputDelta(_) => "item/commandExecution/outputDelta",
            Self::ReasoningTextDelta(_) => "item/reasoning/textDelta",
            Self::ServerRequestResolved(_) => "serverRequest/resolved",
            Self::Snapshot(_) => "delta/snapshot",
        }
    }
}

/// A server notification as written: `{"method", "params", "emittedAtMs"}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ServerNotificationEnvelope {
    #[serde(flatten)]
    pub notification: ServerNotification,
    /// Unix milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emitted_at_ms: Option<i64>,
}

/// A request the server sends; the comment names the response type.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "method")]
pub enum ServerRequest {
    /// [`v2::CommandExecutionRequestApprovalResponse`].
    #[serde(rename = "item/commandExecution/requestApproval")]
    CommandExecutionRequestApproval {
        #[serde(rename = "id")]
        request_id: RequestId,
        params: CommandExecutionRequestApprovalParams,
    },
    /// [`v2::FileChangeRequestApprovalResponse`].
    #[serde(rename = "item/fileChange/requestApproval")]
    FileChangeRequestApproval {
        #[serde(rename = "id")]
        request_id: RequestId,
        params: FileChangeRequestApprovalParams,
    },
    /// [`delta::ToolRequestApprovalResponse`].
    #[serde(rename = "delta/tool/requestApproval")]
    ToolRequestApproval {
        #[serde(rename = "id")]
        request_id: RequestId,
        params: ToolRequestApprovalParams,
    },
}

impl ServerRequest {
    pub fn id(&self) -> &RequestId {
        match self {
            Self::CommandExecutionRequestApproval { request_id, .. }
            | Self::FileChangeRequestApproval { request_id, .. }
            | Self::ToolRequestApproval { request_id, .. } => request_id,
        }
    }

    /// The thread the request belongs to.
    pub fn thread_id(&self) -> &str {
        match self {
            Self::CommandExecutionRequestApproval { params, .. } => &params.thread_id,
            Self::FileChangeRequestApproval { params, .. } => &params.thread_id,
            Self::ToolRequestApproval { params, .. } => &params.thread_id,
        }
    }
}

/// The JSON Schema of every message and response, as `schema/v2.json` holds it.
pub fn json_schema() -> serde_json::Value {
    use schemars::generate::SchemaSettings;

    let mut generator = SchemaSettings::draft2020_12().into_generator();
    let mut roots = serde_json::Map::new();
    macro_rules! roots {
        ($($ty:ty),* $(,)?) => {$(
            roots.insert(stringify!($ty).rsplit("::").next().unwrap_or_default().trim().to_owned(), generator.subschema_for::<$ty>().to_value());
        )*};
    }
    roots!(
        ClientRequest,
        ClientNotification,
        ServerNotificationEnvelope,
        ServerRequest,
        JsonRpcResponse,
        JsonRpcError,
        v2::InitializeResponse,
        v2::ThreadSession,
        v2::ThreadListResponse,
        v2::ThreadReadResponse,
        v2::TurnStartResponse,
        v2::TurnSteerResponse,
        v2::TurnInterruptResponse,
        v2::CommandExecutionRequestApprovalResponse,
        v2::FileChangeRequestApprovalResponse,
        delta::PermissionsResponse,
        delta::PermissionRulesResponse,
        delta::ToolRequestApprovalResponse,
    );
    serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "delta-protocol",
        "roots": roots,
        "$defs": generator.definitions(),
    })
}
