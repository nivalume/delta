//! [`ApprovalPolicy`]: decides, per tool call, whether it runs, is denied or must be approved.

use std::error::Error as StdError;
use std::fmt;
use std::time::Duration;

use delta_chord::context::Context;
use futures::future::BoxFuture;

use crate::records::{ConversationId, TaskId};
use crate::registry::{HookApi, HookPermission};
use crate::tool::{HarnessTool, PermissionSubject, ToolHookCall};

/// A policy failure; the call is denied with its text (fail closed).
pub type PolicyError = Box<dyn StdError + Send + Sync>;

/// The call a policy decides on.
pub struct ApprovalCall<'a> {
    pub conversation_id: ConversationId,
    pub task_id: TaskId,
    /// The call with its final arguments.
    pub call: &'a ToolHookCall,
    pub tool: &'a dyn HarnessTool,
    /// What permission rules match the call on (`HarnessTool::permission_subject`).
    pub subject: Option<PermissionSubject>,
    /// The call's working directory: its environment's, else the agent's.
    pub cwd: Option<&'a str>,
    /// What the `before_tool` hooks said about approval. The tool task asks when they said
    /// `Ask` whatever the policy decides; `Allow` is the policy's to honor.
    pub hook: Option<HookPermission>,
}

impl fmt::Debug for ApprovalCall<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApprovalCall")
            .field("conversation_id", &self.conversation_id)
            .field("task_id", &self.task_id)
            .field("call", &self.call)
            .field("subject", &self.subject)
            .field("cwd", &self.cwd)
            .field("hook", &self.hook)
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
        /// The rule a UI offers to always allow such calls (`ApprovedWithRule`).
        proposed_rule: Option<String>,
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

/// Asks for every call, with the settings' timeout.
#[derive(Clone, Copy, Debug, Default)]
pub struct AskAll;

impl ApprovalPolicy for AskAll {
    fn decide<'a>(
        &'a self,
        _call: &'a ApprovalCall<'a>,
        _api: &'a HookApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, Result<PolicyDecision, PolicyError>> {
        Box::pin(async {
            Ok(PolicyDecision::Ask {
                timeout: None,
                prompt: None,
                proposed_rule: None,
            })
        })
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
