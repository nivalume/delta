//! `tool_search`: the model searches the deferred tools it is not offered yet, and the matches
//! are loaded with `control.add_tools`, so the next request offers them. Loading writes the
//! conversation's `model_tools` (its `add`), so loaded tools stay offered on that branch, also
//! after a compaction. Port of upstream `src/tool-search/tool-search.ts`.

use std::sync::Arc;

use delta_ai::types::{JsonObject, Tool};
use delta_chord::context::Context;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::registry::Extension;
use crate::search::{SearchDocument, rank_documents};
use crate::tool::{Callers, HarnessTool, ToolApi, ToolControl, ToolError, ToolExecutionResult, ToolReplay};

pub const TOOL_SEARCH_TOOL_NAME: &str = "tool_search";
pub const DEFAULT_TOOL_SEARCH_LIMIT: usize = 8;

/// The name of the [`tool_search_extension`].
pub const TOOL_SEARCH_EXTENSION: &str = "tool-search";

/// Fixed, so it does not change while deferred tools are registered, for example as MCP
/// servers connect.
const DESCRIPTION: &str = "# Tool discovery\n\nSearches over deferred tool metadata with BM25 and exposes matching tools for the next model call.\n\nSome of the tools, such as tools of MCP servers, may not have been provided to you upfront, and you should use this tool (`tool_search`) to search for the required tools. For MCP tool discovery, always use `tool_search`.";

/// `tool_search` result details: the tools it loaded.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSearchDetails {
    pub loaded: Vec<String>,
}

/// Schema descriptions and property names, recursively.
fn schema_text<'a>(schema: &'a Value, parts: &mut Vec<&'a str>) {
    let Value::Object(schema) = schema else {
        return;
    };
    if let Some(Value::String(description)) = schema.get("description") {
        parts.push(description);
    }
    if let Some(Value::Object(properties)) = schema.get("properties") {
        for (name, property) in properties {
            parts.push(name);
            schema_text(property, parts);
        }
    }
    if let Some(items) = schema.get("items") {
        schema_text(items, parts);
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(variants)) = schema.get(key) {
            for variant in variants {
                schema_text(variant, parts);
            }
        }
    }
}

/// The search text of a tool: its name, the name with `_` as spaces, its description, and its
/// parameters' text.
fn search_text(tool: &Tool) -> String {
    let spaced = tool.name.replace('_', " ");
    let mut parts = vec![tool.name.as_str(), spaced.as_str(), tool.description.as_str()];
    schema_text(&tool.parameters, &mut parts);
    parts.retain(|part| !part.trim().is_empty());
    parts.join(" ")
}

/// The search limit from the arguments: a positive integer, [`DEFAULT_TOOL_SEARCH_LIMIT`] when
/// absent.
fn limit_of(arguments: &JsonObject) -> Result<usize, ToolError> {
    let Some(limit) = arguments.get("limit").filter(|limit| !limit.is_null()) else {
        return Ok(DEFAULT_TOOL_SEARCH_LIMIT);
    };
    let whole = limit.as_u64().or_else(|| {
        limit
            .as_f64()
            .filter(|limit| limit.fract() == 0.0 && *limit >= 1.0 && *limit <= 9_007_199_254_740_991.0)
            .map(|limit| {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // A safe integer.
                let limit = limit as u64;
                limit
            })
    });
    whole
        .filter(|limit| *limit > 0)
        .map(|limit| usize::try_from(limit).unwrap_or(usize::MAX))
        .ok_or_else(|| "limit must be a positive integer".into())
}

/// The `tool_search` tool.
#[derive(Debug)]
pub struct ToolSearchTool {
    definition: Tool,
}

impl ToolSearchTool {
    pub fn new() -> Self {
        let parameters = serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Search query for deferred tools." },
                "limit": {
                    "type": "number",
                    "description": format!("Maximum number of tools to return. Defaults to {DEFAULT_TOOL_SEARCH_LIMIT}."),
                },
            },
            "required": ["query"],
        });
        Self {
            definition: Tool::new(TOOL_SEARCH_TOOL_NAME, DESCRIPTION, parameters),
        }
    }
}

impl Default for ToolSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl HarnessTool for ToolSearchTool {
    fn definition(&self) -> &Tool {
        &self.definition
    }

    // Searching changes what the model is offered; scripts have their own `searchTools()`.
    fn callers(&self) -> Callers {
        Callers::MODEL
    }

    fn replay(&self) -> ToolReplay {
        ToolReplay::Safe
    }

    fn execute<'a>(
        &'a self,
        arguments: JsonObject,
        api: &'a ToolApi<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, Result<ToolExecutionResult, ToolError>> {
        Box::pin(async move {
            let query = arguments.get("query").and_then(Value::as_str).unwrap_or_default();
            if query.trim().is_empty() {
                return Err("query must not be empty".into());
            }
            let limit = limit_of(&arguments)?;
            let candidates = &api.agent().deferred;
            let documents: Vec<SearchDocument> = candidates
                .iter()
                .map(|tool| SearchDocument {
                    name: tool.name().to_owned(),
                    text: search_text(tool.definition()),
                })
                .collect();
            let loaded = rank_documents(query, &documents, limit);
            let text = if loaded.is_empty() {
                "No matching tools found.".to_owned()
            } else {
                let lines: Vec<String> = loaded
                    .iter()
                    .map(|name| {
                        let description = candidates
                            .iter()
                            .find(|tool| tool.name() == name)
                            .map(|tool| tool.definition().description.trim())
                            .unwrap_or_default();
                        let first = description.lines().next().unwrap_or_default();
                        format!("- {name}: {first}")
                    })
                    .collect();
                let plural = if loaded.len() == 1 { "" } else { "s" };
                format!(
                    "Loaded {} tool{plural}. They are available from your next call:\n{}",
                    loaded.len(),
                    lines.join("\n")
                )
            };
            let control = (!loaded.is_empty()).then(|| ToolControl {
                add_tools: loaded.clone(),
                ..ToolControl::default()
            });
            Ok(ToolExecutionResult {
                details: Some(serde_json::to_value(ToolSearchDetails { loaded })?),
                control,
                ..ToolExecutionResult::text(text)
            })
        })
    }
}

/// The `tool_search` extension: install it where the model should find deferred tools.
pub fn tool_search_extension() -> Extension {
    Extension::new(TOOL_SEARCH_EXTENSION).with_tool(Arc::new(ToolSearchTool::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_search_text_has_the_names_descriptions_and_parameter_text() {
        let tool = Tool::new(
            "github_get_issue",
            "Get an issue",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "number": { "type": "number", "description": "Issue number" },
                    "labels": { "type": "array", "items": { "anyOf": [{ "description": "A label name" }] } },
                    "blank": { "description": "  " },
                },
            }),
        );
        assert_eq!(
            search_text(&tool),
            "github_get_issue github get issue Get an issue number Issue number labels A label name blank"
        );
    }

    #[test]
    fn limits_are_positive_integers() {
        let arguments = |limit: Value| {
            let mut arguments = JsonObject::new();
            arguments.insert("limit".to_owned(), limit);
            arguments
        };
        assert_eq!(limit_of(&JsonObject::new()).unwrap(), DEFAULT_TOOL_SEARCH_LIMIT);
        assert_eq!(limit_of(&arguments(Value::from(3))).unwrap(), 3);
        assert_eq!(limit_of(&arguments(serde_json::json!(2.0))).unwrap(), 2);
        for bad in [serde_json::json!(0), serde_json::json!(1.5), serde_json::json!(-1)] {
            assert_eq!(
                limit_of(&arguments(bad)).unwrap_err().to_string(),
                "limit must be a positive integer"
            );
        }
    }
}
