//! The `codemode` tool and extension.

use std::sync::Arc;

use delta_ai::types::{JsonObject, Tool};
use delta_chord::context::Context;
use delta_chord::output::{OutputLimits, Retain};
use futures::future::BoxFuture;

use super::CodemodeOptions;
use super::run::{self, LIMITS};
use crate::registry::Extension;
use crate::tool::{Callers, HarnessTool, ToolApi, ToolError, ToolExecutionResult, ToolReplay};

pub const CODEMODE_TOOL_NAME: &str = "codemode";

const DESCRIPTION_INTRO: &str = "Run JavaScript that calls other tools. The input is raw JavaScript (not JSON, no code fence), run as an async function body in a QuickJS sandbox: top-level `await` and `return` work. No Node, file system, network, or timers.
- `await tools.<name>({ ...args })` resolves to a string, or an object if the tool's declaration says so, and rejects with an Error on failure. A tool whose declaration returns an object resolves on failure too when it returned that object; check its fields. Calls still running when the script ends are cancelled.
- The script survives a restart: it resumes where it was, without running finished calls again.
- Optional first line: `// @options: {\"max_output_tokens\": 10000, \"timeout_ms\": 60000}`";

/// The `codemode` tool's description: the script environment and its globals, the same for
/// every conversation.
pub fn codemode_description() -> String {
    let globals = [
        "Globals:",
        "- `text(value)`, `image(dataUrlOrImageBlock)`, `console.log(...)`, and top-level `return` add output; `exit()` ends the script. With several text items, each starts with a `==> text N/M <==` line, and `console` lines follow the other output in one `<console_output>` block. `image()` also saves the image to a temp file and the result names its path.",
        "- `store(key, value)` and `load(key)` keep JSON values across codemode calls.",
        "- `ALL_TOOLS` lists `{ name, description }` for every tool scripts can call.",
    ];
    [DESCRIPTION_INTRO.to_owned(), globals.join("\n")].join("\n\n")
}

/// The `codemode` tool: scripts call the conversation's callable tools as
/// `tools.<name>(args)`.
struct CodemodeTool {
    definition: Tool,
    options: CodemodeOptions,
}

impl HarnessTool for CodemodeTool {
    fn definition(&self) -> &Tool {
        &self.definition
    }

    // Scripts cannot start scripts.
    fn callers(&self) -> Callers {
        Callers::MODEL
    }

    fn replay(&self) -> ToolReplay {
        ToolReplay::Safe
    }

    // Outputs are bounded by the script's own token budget; these only stop a runaway.
    fn output_limits(&self) -> OutputLimits {
        OutputLimits {
            max_bytes: LIMITS.max_bytes,
            max_lines: LIMITS.max_lines,
            retain: Retain::Head,
        }
    }

    fn execute<'a>(
        &'a self,
        arguments: JsonObject,
        api: &'a ToolApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<ToolExecutionResult, ToolError>> {
        Box::pin(async move {
            let Some(code) = arguments.get("code").and_then(serde_json::Value::as_str) else {
                return Err("codemode expects `code`, a string of JavaScript".into());
            };
            run::execute(&self.options, code, api, ctx).await
        })
    }
}

/// The `codemode` tool.
pub fn codemode_tool(options: CodemodeOptions) -> Arc<dyn HarnessTool> {
    let parameters = serde_json::json!({
        "type": "object",
        "properties": { "code": { "type": "string", "description": "Raw JavaScript source." } },
        "required": ["code"],
    });
    Arc::new(CodemodeTool {
        definition: Tool::new(CODEMODE_TOOL_NAME, codemode_description(), parameters),
        options,
    })
}

/// The `codemode` extension: the `codemode` tool.
pub fn codemode_extension(options: CodemodeOptions) -> Extension {
    Extension::new("codemode").with_tool(codemode_tool(options))
}
