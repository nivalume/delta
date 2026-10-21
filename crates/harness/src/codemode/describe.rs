//! What the model reads about code mode: the tool's declarations as scripts see them, and the
//! `codemode` prompt section, which lists the tools scripts can call as TypeScript declarations,
//! grouped by namespace and limited to a token budget. Tools that do not fit are found with
//! `searchTools()`. The section renders from the conversation's callable tools, so it changes
//! with them; the tool's description never does. Port of upstream `codemode/describe.ts`.

use std::collections::HashSet;
use std::sync::Arc;

use delta_chord::context::Context;
use delta_codemode::{ToolDeclaration, render_tool_sample, to_codemode_identifier};
use futures::future::BoxFuture;
use serde_json::Value;

use super::{CODEMODE_INTERNAL_PREFIX, CodemodeNamespace, namespace_of};
use crate::prompt::{PromptInput, PromptSection, SectionError};
use crate::tool::HarnessTool;

/// Default token budget of the tool declarations in the `codemode` section.
pub const DEFAULT_CODEMODE_INLINE_BUDGET: usize = 3000;
/// Characters per token when estimating the cost of a tool section.
const CHARS_PER_TOKEN: usize = 4;

/// A tool's description and TypeScript declaration; a tool without an output schema resolves
/// with its text.
pub(super) fn sample_of(tool: &dyn HarnessTool) -> String {
    let definition = tool.definition();
    let text: Value = serde_json::json!({ "type": "string" });
    render_tool_sample(
        &ToolDeclaration {
            name: &definition.name,
            description: &definition.description,
            input_schema: Some(&definition.parameters),
            output_schema: Some(tool.structured_output_schema().unwrap_or(&text)),
        },
        None,
    )
}

/// `` ### `id` (`raw name`) `` followed by the tool's description and declaration.
fn render_tool_section(tool: &dyn HarnessTool) -> String {
    let name = tool.name();
    let id = to_codemode_identifier(name);
    let heading = if id == name {
        format!("### `{id}`")
    } else {
        format!("### `{id}` (`{name}`)")
    };
    format!("{heading}\n{}", sample_of(tool).trim())
}

struct CatalogEntry {
    name: String,
    section: String,
    cost: usize,
}

struct CatalogGroup<'a> {
    namespace: Option<&'a CodemodeNamespace>,
    entries: Vec<CatalogEntry>,
}

/// Picks the tool sections that fit the budget: in each round every group (tools without a
/// namespace first, then namespaces by name) places its cheapest remaining tool; a group whose
/// next tool does not fit drops out while the others continue. Every namespace is represented
/// before any namespace is complete.
fn select_catalog(groups: &[CatalogGroup<'_>], budget: usize) -> HashSet<String> {
    let mut queues: Vec<Vec<&CatalogEntry>> = groups
        .iter()
        .map(|group| {
            let mut entries: Vec<&CatalogEntry> = group.entries.iter().collect();
            entries.sort_by_key(|entry| entry.cost);
            entries.reverse();
            entries
        })
        .filter(|queue| !queue.is_empty())
        .collect();
    let mut shown = HashSet::new();
    let mut remaining = budget;
    while !queues.is_empty() {
        queues.retain_mut(|queue| {
            let Some(next) = queue.last() else { return false };
            if next.cost > remaining {
                return false;
            }
            remaining -= next.cost;
            shown.insert(next.name.clone());
            queue.pop();
            !queue.is_empty()
        });
    }
    shown
}

/// The `codemode` section: the tools scripts can call, grouped by namespace, with their
/// declarations as far as the budget allows. `None` without callable tools.
pub fn codemode_section(
    tools: &[Arc<dyn HarnessTool>],
    namespaces: &[CodemodeNamespace],
    budget: usize,
) -> Option<String> {
    if tools.is_empty() {
        return None;
    }
    let mut groups: Vec<CatalogGroup<'_>> = vec![CatalogGroup {
        namespace: None,
        entries: Vec::new(),
    }];
    for tool in tools {
        let namespace = namespace_of(tool.name(), namespaces);
        let index = match groups
            .iter()
            .position(|group| group.namespace.map(|own| &own.name) == namespace.map(|namespace| &namespace.name))
        {
            Some(index) => index,
            None => {
                groups.push(CatalogGroup {
                    namespace,
                    entries: Vec::new(),
                });
                groups.len() - 1
            }
        };
        let section = render_tool_section(&**tool);
        let cost = section.chars().count().div_ceil(CHARS_PER_TOKEN);
        groups[index].entries.push(CatalogEntry {
            name: tool.name().to_owned(),
            section,
            cost,
        });
    }
    // Stable: tools without a namespace first, then namespaces by name.
    groups.sort_by(|a, b| match (a.namespace, b.namespace) {
        (None, None) => std::cmp::Ordering::Equal,
        (None, Some(_)) => std::cmp::Ordering::Less,
        (Some(_), None) => std::cmp::Ordering::Greater,
        (Some(a), Some(b)) => a.name.cmp(&b.name),
    });
    let shown = select_catalog(&groups, budget);

    let mut sections = vec!["Tools scripts can call (`await tools.<name>(args)`):".to_owned()];
    for group in &groups {
        if group.entries.is_empty() {
            continue;
        }
        let visible: Vec<&CatalogEntry> = group
            .entries
            .iter()
            .filter(|entry| shown.contains(&entry.name))
            .collect();
        if let Some(namespace) = group.namespace {
            let listing = if visible.len() == group.entries.len() {
                ""
            } else if visible.is_empty() {
                " (tools not listed)"
            } else {
                " (some tools not listed)"
            };
            let description = namespace
                .description
                .as_deref()
                .map(str::trim)
                .filter(|description| !description.is_empty())
                .map(|description| format!("\n{description}"))
                .unwrap_or_default();
            sections.push(format!("## {}{listing}{description}", namespace.name));
        } else if visible.len() < group.entries.len() {
            sections.push("Some tools are not listed: find them with `searchTools()`.".to_owned());
        }
        sections.extend(visible.into_iter().map(|entry| entry.section.clone()));
    }
    Some(sections.join("\n\n"))
}

/// The `codemode` prompt section of the extension.
pub(super) struct CodemodeSection {
    pub(super) namespaces: Vec<CodemodeNamespace>,
    pub(super) budget: usize,
}

impl PromptSection for CodemodeSection {
    #[allow(clippy::unnecessary_literal_bound)] // The trait's signature.
    fn key(&self) -> &str {
        "codemode"
    }

    fn render<'a>(
        &'a self,
        input: &'a PromptInput<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, Result<Option<String>, SectionError>> {
        // Deferred tools are found with `searchTools()`, as the coding agent leaves them out too.
        let tools: Vec<Arc<dyn HarnessTool>> = input
            .agent
            .callable
            .iter()
            .filter(|tool| !tool.name().starts_with(CODEMODE_INTERNAL_PREFIX) && !tool.deferred())
            .cloned()
            .collect();
        let text = codemode_section(&tools, &self.namespaces, self.budget);
        Box::pin(async move { Ok(text) })
    }
}

#[cfg(test)]
mod tests {
    use delta_ai::types::Tool;

    use super::*;
    use crate::tool::{FnHarnessTool, ToolExecutionResult};

    fn tool(name: &str, description: &str) -> Arc<dyn HarnessTool> {
        Arc::new(FnHarnessTool::new(
            Tool::new(
                name,
                description,
                serde_json::json!({ "type": "object", "properties": { "q": { "type": "string" } } }),
            ),
            |_arguments, _api, _ctx| Box::pin(async { Ok(ToolExecutionResult::text("")) }),
        ))
    }

    #[test]
    fn tools_are_listed_by_namespace() {
        let tools = [
            tool("read", "Reads a file."),
            tool("mcp__github__issue_read", "Reads an issue."),
        ];
        let namespaces = [CodemodeNamespace::new("github", "mcp__github__").with_description("GitHub's API.")];
        let section = codemode_section(&tools, &namespaces, DEFAULT_CODEMODE_INLINE_BUDGET).unwrap();
        assert_eq!(
            section,
            "Tools scripts can call (`await tools.<name>(args)`):\n\n### `read`\nReads a file.\n\ncodemode tool declaration:\n```ts\ndeclare const tools: { read(args: { q?: string; }): Promise<string>; };\n```\n\n## github\nGitHub's API.\n\n### `mcp__github__issue_read`\nReads an issue.\n\ncodemode tool declaration:\n```ts\ndeclare const tools: { mcp__github__issue_read(args: { q?: string; }): Promise<string>; };\n```"
        );
        assert!(codemode_section(&[], &namespaces, 100).is_none());
    }

    #[test]
    fn every_namespace_is_represented_before_any_is_complete() {
        let long = "x".repeat(400);
        let tools = [
            tool("my-tool", "Short."),
            tool("a__one", "One."),
            tool("a__two", &long),
            tool("b__one", "One."),
        ];
        let namespaces = [CodemodeNamespace::new("a", "a__"), CodemodeNamespace::new("b", "b__")];
        let cost = |name: &str| {
            render_tool_section(&*tools.iter().find(|tool| tool.name() == name).unwrap().clone())
                .chars()
                .count()
                .div_ceil(CHARS_PER_TOKEN)
        };
        let budget = cost("my-tool") + cost("a__one") + cost("b__one");
        let section = codemode_section(&tools, &namespaces, budget).unwrap();
        assert!(section.contains("### `my_tool` (`my-tool`)"), "{section}");
        assert!(section.contains("## a (some tools not listed)"), "{section}");
        assert!(
            section.contains("### `a__one`") && !section.contains("### `a__two`"),
            "{section}"
        );
        assert!(section.contains("## b\n\n### `b__one`"), "{section}");
        let tiny = codemode_section(&tools, &namespaces, 1).unwrap();
        assert!(
            tiny.contains("Some tools are not listed: find them with `searchTools()`.")
                && tiny.contains("## a (tools not listed)"),
            "{tiny}"
        );
    }
}
