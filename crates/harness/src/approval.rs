//! The approval seam: the policy that decides, per tool call, whether the call runs, is
//! denied or must be approved.
//!
//! The tool task asks the policy once per call, after the `before_tool` chain, on the final
//! arguments: `Allow` runs the call, `Deny` settles it `blocked`. Durable requests, decisions,
//! grants and the waiting phase for `Ask` are the approval phase's (until it lands, `Ask`
//! settles `blocked` too). See `docs/design/approval.md`.

use std::error::Error as StdError;
use std::fmt;
use std::time::Duration;

use delta_chord::context::Context;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

use crate::records::{ConversationId, TaskId};
use crate::registry::HookApi;
use crate::tool::{HarnessTool, ToolHookCall};

/// An approval request's id: the id of the tool task that asks, which asks at most once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ApprovalId(pub TaskId);

impl fmt::Display for ApprovalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// A policy failure; the call is denied with its text (fail closed).
pub type PolicyError = Box<dyn StdError + Send + Sync>;

/// The call a policy decides on.
pub struct ApprovalCall<'a> {
    pub conversation_id: ConversationId,
    pub task_id: TaskId,
    /// The call with its final arguments.
    pub call: &'a ToolHookCall,
    pub tool: &'a dyn HarnessTool,
    pub cwd: Option<&'a str>,
}

impl fmt::Debug for ApprovalCall<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApprovalCall")
            .field("conversation_id", &self.conversation_id)
            .field("task_id", &self.task_id)
            .field("call", &self.call)
            .field("cwd", &self.cwd)
            .finish_non_exhaustive()
    }
}

/// What a policy decides (Codex's `ExecApprovalRequirement`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolicyDecision {
    Allow,
    Deny {
        reason: String,
    },
    Ask {
        timeout: Option<Duration>,
        prompt: Option<String>,
    },
}

/// Decides whether a tool call needs approval. It runs off the session line with the
/// task's [`HookApi`] (ids, models, memos); a crash before the call's intent commit may run
/// it again.
pub trait ApprovalPolicy: Send + Sync {
    fn decide<'a>(
        &'a self,
        call: &'a ApprovalCall<'a>,
        api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<PolicyDecision, PolicyError>>;
}

impl fmt::Debug for dyn ApprovalPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApprovalPolicy").finish_non_exhaustive()
    }
}

/// The default policy: every call runs, as upstream without approvals.
#[derive(Clone, Copy, Debug, Default)]
pub struct AllowAll;

impl ApprovalPolicy for AllowAll {
    fn decide<'a>(
        &'a self,
        _call: &'a ApprovalCall<'a>,
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, Result<PolicyDecision, PolicyError>> {
        Box::pin(async { Ok(PolicyDecision::Allow) })
    }
}

/// Denies every call with `reason`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DenyAll {
    pub reason: String,
}

impl ApprovalPolicy for DenyAll {
    fn decide<'a>(
        &'a self,
        _call: &'a ApprovalCall<'a>,
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, Result<PolicyDecision, PolicyError>> {
        let reason = self.reason.clone();
        Box::pin(async move { Ok(PolicyDecision::Deny { reason }) })
    }
}
