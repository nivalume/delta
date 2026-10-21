//! Argument validation before execution and the structured output contract after it.

use delta_ai::types::{JsonObject, ToolCall};
use delta_ai::validation::validate_tool_arguments;
use serde_json::Value;

use super::definition::HarnessTool;

/// The call's arguments validated (and coerced) against the tool's parameters; the error
/// message lists every violation.
pub(super) fn arguments(tool: &dyn HarnessTool, call: &ToolCall) -> Result<JsonObject, String> {
    match validate_tool_arguments(tool.definition(), call) {
        Ok(Value::Object(arguments)) => Ok(arguments),
        Ok(other) => Err(format!(
            "Arguments of tool \"{}\" are not an object: {other}",
            call.name
        )),
        Err(error) => Err(error.0),
    }
}

/// Why `value` breaks `schema`; `None` when it matches. A schema that does not compile
/// cannot reject.
pub(super) fn structured(schema: &Value, value: &Value) -> Option<String> {
    let validator = jsonschema::validator_for(schema).ok()?;
    let errors: Vec<String> = validator
        .iter_errors(value)
        .map(|error| format!("{}: {error}", error.instance_path()))
        .collect();
    (!errors.is_empty()).then(|| errors.join("; "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_output_is_checked_against_its_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "n": { "type": "integer" } },
            "required": ["n"],
        });
        assert_eq!(structured(&schema, &serde_json::json!({ "n": 1 })), None);
        let problem = structured(&schema, &serde_json::json!({ "n": "x" })).unwrap();
        assert!(problem.contains("/n"), "{problem}");
        assert_eq!(structured(&serde_json::json!({ "type": 5 }), &Value::Null), None);
    }
}
