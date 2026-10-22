//! `edit`: exact-text replacements in one file, with a diff and a patch as details.

use delta_ai::types::{JsonObject, Tool};
use delta_chord::context::Context;
use futures::future::BoxFuture;
use serde::Deserialize;
use serde_json::Value;

use super::edit_diff::{
    Edit, apply_edits, detect_line_ending, diff_string, normalize_to_lf, restore_line_endings, strip_bom, unified_patch,
};
use super::mutation::FileMutationQueue;
use super::paths::resolve_tool_path;
use super::{check_aborted, parse, path_subject, require_env};
use crate::env::{ExecutionEnv, FileError, FileKind};
use crate::tool::{HarnessTool, PathAccess, PermissionSubject, ToolApi, ToolError, ToolExecutionResult};

/// The `edit` tool. Details: `{ diff, patch, firstChangedLine? }`. Its mutations of a file
/// wait for the queue's earlier ones.
#[derive(Debug)]
pub struct EditTool {
    definition: Tool,
    queue: FileMutationQueue,
}

impl EditTool {
    pub fn new(queue: FileMutationQueue) -> Self {
        let parameters = serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to edit (relative or absolute)" },
                "edits": {
                    "type": "array",
                    "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {
                                "type": "string",
                                "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call.",
                            },
                            "newText": { "type": "string", "description": "Replacement text for this targeted edit." },
                        },
                        "required": ["oldText", "newText"],
                    },
                },
            },
            "required": ["path", "edits"],
        });
        let description = "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.";
        Self {
            definition: Tool::new("edit", description, parameters),
            queue,
        }
    }
}

#[derive(Deserialize)]
struct EditArguments {
    path: String,
    edits: Vec<Edit>,
}

fn is_single_edit(value: &Value) -> bool {
    value.get("oldText").is_some_and(Value::is_string) && value.get("newText").is_some_and(Value::is_string)
}

/// Repairs the shapes models commonly send: `edits` as a JSON string or as one edit object,
/// and a top-level `oldText`/`newText` pair, appended to `edits`.
fn prepare_edit_arguments(mut arguments: JsonObject) -> JsonObject {
    let repaired = match arguments.get("edits") {
        Some(Value::String(text)) => match serde_json::from_str::<Value>(text) {
            Ok(parsed @ Value::Array(_)) => Some(parsed),
            Ok(parsed) if is_single_edit(&parsed) => Some(Value::Array(vec![parsed])),
            _ => None,
        },
        Some(edit) if is_single_edit(edit) => Some(Value::Array(vec![edit.clone()])),
        _ => None,
    };
    if let Some(edits) = repaired {
        arguments.insert("edits".to_owned(), edits);
    }
    if arguments.get("oldText").is_some_and(Value::is_string) && arguments.get("newText").is_some_and(Value::is_string)
    {
        let old_text = arguments.remove("oldText").unwrap_or_default();
        let new_text = arguments.remove("newText").unwrap_or_default();
        let mut edits = match arguments.remove("edits") {
            Some(Value::Array(edits)) => edits,
            _ => Vec::new(),
        };
        edits.push(serde_json::json!({ "oldText": old_text, "newText": new_text }));
        arguments.insert("edits".to_owned(), Value::Array(edits));
    }
    arguments
}

fn access_error(path: &str, error: &FileError) -> ToolError {
    format!("Could not edit file: {path}. Error code: {}.", error.code).into()
}

impl HarnessTool for EditTool {
    fn definition(&self) -> &Tool {
        &self.definition
    }

    fn prepare_arguments(&self, arguments: JsonObject) -> Result<JsonObject, ToolError> {
        Ok(prepare_edit_arguments(arguments))
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
            let EditArguments { path, edits } = parse(arguments)?;
            if edits.is_empty() {
                return Err("Edit tool input is invalid. edits must contain at least one replacement.".into());
            }
            let env = require_env(api)?;
            let absolute = resolve_tool_path(env, &path)?;
            let edit = edit_file(env, &absolute, &path, &edits, ctx);
            self.queue.run(env, &absolute, ctx, edit).await?
        })
    }
}

/// The edit inside the file's mutation slot.
async fn edit_file(
    env: &dyn ExecutionEnv,
    absolute: &str,
    path: &str,
    edits: &[Edit],
    ctx: &Context,
) -> Result<ToolExecutionResult, ToolError> {
    check_aborted(ctx)?;
    let info = env
        .file_info(absolute, ctx)
        .await
        .map_err(|error| access_error(path, &error))?;
    if !matches!(info.kind, FileKind::File | FileKind::Symlink) {
        return Err(format!("Could not edit file: {path}. Path is not a file.").into());
    }
    let text = env
        .read_text_file(absolute, ctx)
        .await
        .map_err(|error| access_error(path, &error))?;
    check_aborted(ctx)?;

    let (bom, content) = strip_bom(&text);
    let ending = detect_line_ending(content);
    let original = normalize_to_lf(content);
    let edited = apply_edits(&original, edits, path)?;
    check_aborted(ctx)?;

    let mut written = String::from(if bom { "\u{feff}" } else { "" });
    written.push_str(&restore_line_endings(&edited, ending));
    env.write_file(absolute, written.as_bytes(), ctx)
        .await
        .map_err(|error| access_error(path, &error))?;
    check_aborted(ctx)?;

    let (diff, first_changed_line) = diff_string(&original, &edited);
    let mut details = serde_json::json!({ "diff": diff, "patch": unified_patch(path, &original, &edited) });
    if let Some(line) = first_changed_line {
        details["firstChangedLine"] = Value::from(line);
    }
    Ok(ToolExecutionResult {
        details: Some(details),
        ..ToolExecutionResult::text(format!("Successfully replaced {} block(s) in {path}.", edits.len()))
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn object(value: &Value) -> JsonObject {
        value.as_object().cloned().unwrap_or_default()
    }

    #[test]
    fn common_argument_shapes_are_repaired() {
        let edit = json!({ "oldText": "a", "newText": "b" });
        let expected = object(&json!({ "path": "f", "edits": [edit.clone()] }));
        assert_eq!(
            prepare_edit_arguments(object(&json!({ "path": "f", "edits": edit.to_string() }))),
            expected
        );
        assert_eq!(prepare_edit_arguments(expected.clone()), expected);
        assert_eq!(
            prepare_edit_arguments(object(&json!({ "path": "f", "edits": edit.clone() }))),
            expected
        );
        assert_eq!(
            prepare_edit_arguments(object(&json!({ "path": "f", "oldText": "a", "newText": "b" }))),
            expected
        );
        let both = prepare_edit_arguments(object(&json!({
            "path": "f", "edits": [{ "oldText": "x", "newText": "y" }], "oldText": "a", "newText": "b",
        })));
        assert_eq!(both["edits"], json!([{ "oldText": "x", "newText": "y" }, edit]));
        // Not JSON: left for validation to reject.
        let broken = object(&json!({ "path": "f", "edits": "[{" }));
        assert_eq!(prepare_edit_arguments(broken.clone()), broken);
    }
}
