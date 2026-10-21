//! The built-in coding tools: `read`, `write`, `edit` and `bash`.
//!
//! Each is a plain [`HarnessTool`] that reaches files and commands only through the call's
//! [`ExecutionEnv`](crate::env::ExecutionEnv), so it runs wherever the conversation does. A
//! call without an environment fails `No execution environment is configured`. All four are
//! not replay-safe and callable by the model and by tools. `write` and `edit` of one file run
//! one at a time through a shared [`FileMutationQueue`]. [`coding_tools`] bundles the four;
//! nothing installs it automatically.
//!
//! Corresponds to upstream `src/tools/` and `src/truncate.ts`. See `docs/design/tools.md`.

mod bash;
mod edit;
mod edit_diff;
mod mutation;
mod paths;
mod read;
mod truncate;
mod write;

use std::sync::Arc;

use delta_ai::types::JsonObject;
use delta_chord::context::Context;
use serde::de::DeserializeOwned;

pub use bash::{BashExecution, BashOptions, BashPrepare, BashTool};
pub use edit::EditTool;
pub use mutation::FileMutationQueue;
pub use paths::{normalize_tool_path, resolve_read_path, resolve_tool_path};
pub use read::ReadTool;
pub use truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, Limit, Totals, Truncation, format_size, truncate_head, truncate_head_of,
};
pub use write::WriteTool;

use crate::env::ExecutionEnv;
use crate::registry::Extension;
use crate::tool::{ToolApi, ToolError};

/// The name of the [`coding_tools`] extension.
pub const CODING_TOOLS: &str = "coding-tools";

/// `read`, `write`, `edit` and `bash` (run with `bash`), with one mutation queue for `write`
/// and `edit`.
pub fn coding_tools(bash: BashOptions) -> Extension {
    let queue = FileMutationQueue::new();
    Extension::new(CODING_TOOLS)
        .with_tool(Arc::new(ReadTool::new()))
        .with_tool(Arc::new(WriteTool::new(queue.clone())))
        .with_tool(Arc::new(EditTool::new(queue)))
        .with_tool(Arc::new(BashTool::new(bash)))
}

/// The call's environment.
fn require_env<'a>(api: &'a ToolApi<'_>) -> Result<&'a dyn ExecutionEnv, ToolError> {
    api.env()
        .map(|env| &**env)
        .ok_or_else(|| "No execution environment is configured".into())
}

/// The validated arguments as the tool's type.
fn parse<T: DeserializeOwned>(arguments: JsonObject) -> Result<T, ToolError> {
    Ok(serde_json::from_value(serde_json::Value::Object(arguments))?)
}

/// Fails once the call is cancelled, between the steps of a mutation.
fn check_aborted(ctx: &Context) -> Result<(), ToolError> {
    if ctx.is_cancelled() {
        return Err("Operation aborted".into());
    }
    Ok(())
}

/// A number as JavaScript prints it in the common cases: integers without a fraction.
fn js_number(value: f64) -> String {
    if value == 0.0 {
        "0".to_owned()
    } else if value.fract() == 0.0 && value.abs() < 1e21 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::js_number;

    #[test]
    fn numbers_print_as_upstream() {
        assert_eq!(js_number(3.0), "3");
        assert_eq!(js_number(2.5), "2.5");
        assert_eq!(js_number(-0.0), "0");
    }
}
