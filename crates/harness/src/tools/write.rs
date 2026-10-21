//! `write`: replace a file's contents, creating it and its parents.

use delta_ai::types::{JsonObject, Tool};
use delta_chord::context::Context;
use futures::future::BoxFuture;
use serde::Deserialize;

use super::mutation::FileMutationQueue;
use super::paths::resolve_tool_path;
use super::{check_aborted, parse, path_subject, require_env};
use crate::env::ExecutionEnv;
use crate::tool::{HarnessTool, PathAccess, PermissionSubject, ToolApi, ToolError, ToolExecutionResult};

/// The `write` tool; its mutations of a file wait for the queue's earlier ones.
#[derive(Debug)]
pub struct WriteTool {
    definition: Tool,
    queue: FileMutationQueue,
}

impl WriteTool {
    pub fn new(queue: FileMutationQueue) -> Self {
        let parameters = serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to write (relative or absolute)" },
                "content": { "type": "string", "description": "Content to write to the file" },
            },
            "required": ["path", "content"],
        });
        let description = "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.";
        Self {
            definition: Tool::new("write", description, parameters),
            queue,
        }
    }
}

#[derive(Deserialize)]
struct WriteArguments {
    path: String,
    content: String,
}

impl HarnessTool for WriteTool {
    fn definition(&self) -> &Tool {
        &self.definition
    }

    fn permission_subject(&self, arguments: &JsonObject, env: Option<&dyn ExecutionEnv>) -> Option<PermissionSubject> {
        path_subject(arguments, env, PathAccess::Write)
    }

    fn execute<'a>(
        &'a self,
        arguments: JsonObject,
        api: &'a ToolApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<ToolExecutionResult, ToolError>> {
        Box::pin(async move {
            let WriteArguments { path, content } = parse(arguments)?;
            let env = require_env(api)?;
            let absolute = resolve_tool_path(env, &path)?;
            let write = async {
                check_aborted(ctx)?;
                env.write_file(&absolute, content.as_bytes(), ctx).await?;
                check_aborted(ctx)?;
                Ok(ToolExecutionResult::text(format!("Successfully wrote to {path}")))
            };
            self.queue.run(env, &absolute, ctx, write).await?
        })
    }
}
