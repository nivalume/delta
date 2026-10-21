//! What handlers receive and return: Claude Code's hook input and output JSON.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A handler's JSON output (all fields optional), as Claude Code defines it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HookOutput {
    /// `false` stops: the bridge blocks what the event allows blocking.
    #[serde(default, rename = "continue", skip_serializing_if = "Option::is_none")]
    pub continue_: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_message: Option<String>,
    /// `"block"` (with `reason`) for the events that take it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hook_specific_output: Option<Value>,
}

impl HookOutput {
    /// A field of `hookSpecificOutput`.
    pub fn specific(&self, field: &str) -> Option<&Value> {
        self.hook_specific_output.as_ref()?.get(field)
    }

    /// A string field of `hookSpecificOutput`.
    pub fn specific_str(&self, field: &str) -> Option<&str> {
        self.specific(field)?.as_str()
    }

    /// Whether the output blocks: `decision: "block"` or `continue: false`; with the
    /// reason to show.
    pub fn block(&self) -> Option<String> {
        if self.decision.as_deref() == Some("block") {
            return Some(self.reason.clone().unwrap_or_else(|| "Blocked by hook".to_owned()));
        }
        if self.continue_ == Some(false) {
            return Some(self.stop_reason.clone().unwrap_or_else(|| "Stopped by hook".to_owned()));
        }
        None
    }
}

/// The name Claude Code's hooks know a built-in tool by (`bash` is `Bash`); other tools
/// keep their names.
pub fn claude_tool_name(name: &str) -> String {
    match name {
        "bash" => "Bash".to_owned(),
        "read" => "Read".to_owned(),
        "write" => "Write".to_owned(),
        "edit" => "Edit".to_owned(),
        other => other.to_owned(),
    }
}

/// A call's arguments as hooks see them: `path` is also given as Claude Code's
/// `file_path`.
pub(crate) fn claude_tool_input(arguments: &Map<String, Value>) -> Value {
    let mut input = arguments.clone();
    if let Some(path) = arguments.get("path").filter(|path| path.is_string())
        && !input.contains_key("file_path")
    {
        input.insert("file_path".to_owned(), path.clone());
    }
    Value::Object(input)
}

/// A hook's `updatedInput` as arguments: `file_path` back to `path` when the tool takes
/// `path`.
pub(crate) fn arguments_from(updated: &Value, original: &Map<String, Value>) -> Option<Map<String, Value>> {
    let mut arguments = updated.as_object()?.clone();
    if original.contains_key("path")
        && !arguments.contains_key("path")
        && let Some(path) = arguments.remove("file_path")
    {
        arguments.insert("path".to_owned(), path);
    } else if original.contains_key("path") && !original.contains_key("file_path") {
        arguments.remove("file_path");
    }
    Some(arguments)
}

/// Claude Code's name of a delta permission mode (snake case), `default` without one.
pub(crate) fn claude_permission_mode(mode: Option<&str>) -> &'static str {
    match mode {
        Some("accept_edits") => "acceptEdits",
        Some("plan") => "plan",
        Some("dont_ask") => "dontAsk",
        Some("bypass") => "bypassPermissions",
        _ => "default",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outputs_parse_upstream_examples() {
        let pre: HookOutput = serde_json::from_value(serde_json::json!({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Destructive command blocked"
            }
        }))
        .unwrap();
        assert_eq!(pre.specific_str("permissionDecision"), Some("deny"));
        let stop: HookOutput =
            serde_json::from_value(serde_json::json!({ "decision": "block", "reason": "Run the tests" })).unwrap();
        assert_eq!(stop.block().as_deref(), Some("Run the tests"));
        let halt: HookOutput =
            serde_json::from_value(serde_json::json!({ "continue": false, "stopReason": "Done for today" })).unwrap();
        assert_eq!(halt.block().as_deref(), Some("Done for today"));
        assert_eq!(HookOutput::default().block(), None);
    }

    #[test]
    fn tool_names_and_inputs_map_both_ways() {
        assert_eq!(claude_tool_name("bash"), "Bash");
        assert_eq!(claude_tool_name("mcp__x"), "mcp__x");
        let arguments = serde_json::json!({ "path": "a.txt", "content": "x" });
        let arguments = arguments.as_object().unwrap();
        let input = claude_tool_input(arguments);
        assert_eq!(input["file_path"], "a.txt");
        let updated = serde_json::json!({ "file_path": "b.txt", "content": "y" });
        let back = arguments_from(&updated, arguments).unwrap();
        assert_eq!(
            Value::Object(back),
            serde_json::json!({ "path": "b.txt", "content": "y" })
        );
        assert_eq!(claude_permission_mode(Some("bypass")), "bypassPermissions");
        assert_eq!(claude_permission_mode(None), "default");
    }
}
