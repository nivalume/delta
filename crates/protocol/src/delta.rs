//! delta's extensions, under the `delta/` method prefix: a conversation's permissions, the
//! harness-wide rules approvals added, approval of tools that are neither commands nor file
//! changes, and the snapshot a client gets when it fell behind.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// How much a thread's tool calls may do without asking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum PermissionMode {
    /// Reads inside the working directories run; anything else not allowed by a rule asks.
    #[default]
    Default,
    /// Also writes inside the working directories run.
    AcceptEdits,
    /// Commands and writes are denied.
    Plan,
    /// What would ask is denied.
    DontAsk,
    /// Everything runs, except what a deny or ask rule names.
    Bypass,
}

/// Rules in Claude Code's `Tool(specifier)` grammar, by what they decide.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PermissionRules {
    #[serde(default)]
    pub allow: Vec<String>,
    #[serde(default)]
    pub ask: Vec<String>,
    #[serde(default)]
    pub deny: Vec<String>,
}

/// A thread's permissions.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Permissions {
    /// The mode the thread set; `None`: the server's default.
    #[serde(default)]
    pub mode: Option<PermissionMode>,
    /// The mode in effect.
    pub effective_mode: PermissionMode,
    pub rules: PermissionRules,
    /// More directories reads (and `acceptEdits` writes) reach.
    #[serde(default)]
    pub directories: Vec<String>,
}

/// `delta/permissions/read`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PermissionsReadParams {
    pub thread_id: String,
}

/// What `delta/permissions/read` and `delta/permissions/update` answer.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PermissionsResponse {
    pub permissions: Permissions,
}

/// `delta/permissions/update`: removals apply before additions; rules are matched by text.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PermissionsUpdateParams {
    pub thread_id: String,
    /// Sets the mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<PermissionMode>,
    /// Returns to the server's default mode; `mode` wins when both are given.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub clear_mode: bool,
    #[serde(default)]
    pub add: PermissionRules,
    #[serde(default)]
    pub remove: PermissionRules,
    /// Replaces the extra directories.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub directories: Option<Vec<String>>,
}

/// `delta/permissionRules/read` takes no params; the allow rules approvals added for every
/// thread.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PermissionRulesResponse {
    pub rules: PermissionRules,
}

/// `delta/permissionRules/remove`; answers [`PermissionRulesResponse`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PermissionRulesRemoveParams {
    pub rule: String,
}

/// `delta/tool/requestApproval`: a call of a tool that is neither a command nor a file change.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ToolRequestApprovalParams {
    pub thread_id: String,
    pub turn_id: String,
    pub item_id: String,
    pub started_at_ms: i64,
    pub tool: String,
    pub arguments: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// The rule `acceptWithRule` would add when the client names none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposed_rule: Option<String>,
}

/// The answer to an approval request of any kind.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum ToolApprovalDecision {
    Accept,
    AcceptForSession,
    /// Accepts and adds an allow rule for every thread.
    AcceptWithRule {
        rule: String,
    },
    Decline,
    Cancel,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ToolRequestApprovalResponse {
    pub decision: ToolApprovalDecision,
}

/// `delta/snapshot`: the thread's whole state, sent when the client fell behind its events.
/// Items in flight continue with their deltas; anything before is in the snapshot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotNotification {
    pub thread_id: String,
    /// The harness's `Snapshot` as JSON.
    pub snapshot: Value,
}
