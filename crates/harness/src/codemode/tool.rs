//! The `codemode` tool and extension.

use std::sync::Arc;

use delta_ai::types::{ConstrainedSampling, GrammarVariants, JsonObject, Tool};
use delta_chord::context::Context;
use delta_chord::output::{OutputLimits, Retain};
use delta_codemode::CODEMODE_SOURCE_GRAMMAR;
use futures::future::BoxFuture;

use super::CodemodeOptions;
use super::describe::CodemodeSection;
use super::models::{model_tools, models_description};
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
pub fn codemode_description(models: bool) -> String {
    let globals = [
        "Globals:",
        "- `text(value)`, `image(dataUrlOrImageBlock)`, `console.log(...)`, and top-level `return` add output; `exit()` ends the script. With several text items, each starts with a `==> text N/M <==` line, and `console` lines follow the other output in one `<console_output>` block. `image()` also saves the image to a temp file and the result names its path.",
        "- `store(key, value)` and `load(key)` keep JSON values across codemode calls.",
        "- `ALL_TOOLS`, `await searchTools(query, { limit?, namespace? })`, `await describeTool(name)`, `await describeNamespace(name)`: find tools the `codemode` section does not list.",
    ];
    let mut parts = vec![DESCRIPTION_INTRO.to_owned(), globals.join("\n")];
    if models {
        parts.push(models_description());
    }
    parts.push(
        "The `codemode` section of the system prompt lists the tools scripts can call, by namespace, as far as space allows."
            .to_owned(),
    );
    parts.join("\n\n")
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
        definition: Tool {
            // Capable models write the script as raw text instead of a JSON-escaped string;
            // `delta_ai` applies it only for them.
            constrained_sampling: Some(ConstrainedSampling::Grammar {
                variants: GrammarVariants {
                    openai_lark: Some(CODEMODE_SOURCE_GRAMMAR.to_owned()),
                    openai_regex: None,
                },
            }),
            ..Tool::new(CODEMODE_TOOL_NAME, codemode_description(options.models), parameters)
        },
        options,
    })
}

/// The `codemode` extension: the `codemode` tool, the `codemode` prompt section listing the
/// tools scripts can call, and with `models` the internal tools behind `models.classify()` and
/// `models.generateImages()`.
pub fn codemode_extension(options: CodemodeOptions) -> Extension {
    let section = CodemodeSection {
        namespaces: options.namespaces.clone(),
        budget: options.inline_budget,
    };
    let models = options.models;
    let mut extension = Extension::new("codemode")
        .with_tool(codemode_tool(options))
        .with_section(Arc::new(section));
    if models {
        for tool in model_tools() {
            extension = extension.with_tool(tool);
        }
    }
    extension
}

#[cfg(test)]
mod tests {
    use delta_codemode::{Sandbox, SandboxOptions};

    use super::*;

    #[test]
    fn the_tool_declares_its_grammar_and_the_extension_its_internal_tools() {
        let options = CodemodeOptions::new(Arc::new(Sandbox::new(SandboxOptions::default())));
        let tool = codemode_tool(options.clone());
        let Some(ConstrainedSampling::Grammar { variants }) = &tool.definition().constrained_sampling else {
            panic!("a grammar");
        };
        assert_eq!(variants.openai_lark.as_deref(), Some(CODEMODE_SOURCE_GRAMMAR));
        assert_eq!(tool.callers(), Callers::MODEL);
        assert!(
            tool.definition()
                .description
                .contains("`models` (all functions return promises):")
        );
        assert!(!codemode_description(false).contains("`models`"));
        let names: Vec<String> = model_tools().iter().map(|tool| tool.name().to_owned()).collect();
        assert_eq!(names, [super::super::CLASSIFY_TOOL, super::super::GENERATE_IMAGES_TOOL]);
        assert!(model_tools().iter().all(|tool| tool.callers() == Callers::TOOLS));
    }
}
