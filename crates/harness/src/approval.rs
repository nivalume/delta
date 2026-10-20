//! The approval seam: the policy that decides, per tool call, whether the call runs, is
//! denied or must be approved.
//!
//! Phase 4 of the crate defines the policy and its default, [`AllowAll`], so hosts pass one
//! through `HarnessOptions.approval` from the start. The tool task asks it once per call,
//! after the `before_tool` chain, on the final arguments; requests, decisions, grants and
//! the waiting phase arrive with the tool task in phase 5. See `docs/design/approval.md`.

use std::error::Error as StdError;
use std::fmt;
use std::time::Duration;

use delta_ai::types::ToolCall;
use delta_chord::context::Context;
use futures::future::BoxFuture;

use crate::records::{ConversationId, TaskId};
use crate::tool::HarnessTool;

/// A policy failure; the call is denied with its text (fail closed).
pub type PolicyError = Box<dyn StdError + Send + Sync>;

/// The call a policy decides on.
pub struct ApprovalCall<'a> {
    pub conversation_id: ConversationId,
    pub task_id: TaskId,
    /// The call with its final arguments.
    pub call: &'a ToolCall,
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

/// Decides whether a tool call needs approval.
pub trait ApprovalPolicy: Send + Sync {
    fn decide<'a>(
        &'a self,
        call: &'a ApprovalCall<'a>,
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
        _ctx: &'a Context,
    ) -> BoxFuture<'a, Result<PolicyDecision, PolicyError>> {
        Box::pin(async { Ok(PolicyDecision::Allow) })
    }
}
