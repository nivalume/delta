//! The hook traits of the built-in tasks besides the generation's: [`ToolHooks`].

use delta_ai::types::JsonObject;
use delta_chord::context::Context;
use futures::future::BoxFuture;

use super::hooks::{HookApi, HookResult};
use crate::tool::{ToolExecutionResult, ToolHookCall};

/// What a `before_tool` hook asks: other arguments, or a block.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BeforeTool {
    /// Replaces the call's arguments; later hooks see the replacement, and they are
    /// validated again after the chain.
    pub arguments: Option<JsonObject>,
    /// Blocks the call with this text; the first block wins and ends the chain.
    pub block: Option<String>,
}

/// Hooks of `delta.tool`. Every method defaults to doing nothing. They run off the session
/// line in the call's first phase and are not run again by recovery: the arguments they
/// produced are in the intent checkpoint.
///
/// | Hook | Composition | Ordinary error |
/// | --- | --- | --- |
/// | `before_tool` | argument replacement chain; the first block wins | blocks the call with its text |
/// | `after_tool` | result replacement chain | reported, chain continues |
pub trait ToolHooks: Send + Sync {
    /// Sees the call before the approval policy; may replace its arguments or block it.
    fn before_tool<'a>(
        &'a self,
        call: &'a ToolHookCall,
        api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<BeforeTool>> {
        let _ = (call, api, ctx);
        Box::pin(async { Ok(None) })
    }

    /// Replaces the result before it is committed (redaction, rewriting); later hooks see
    /// the replacement.
    fn after_tool<'a>(
        &'a self,
        call: &'a ToolHookCall,
        result: &'a ToolExecutionResult,
        api: &'a HookApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, HookResult<ToolExecutionResult>> {
        let _ = (call, result, api, ctx);
        Box::pin(async { Ok(None) })
    }
}
