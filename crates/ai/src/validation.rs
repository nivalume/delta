//! Validation of tool call arguments against the tool's JSON Schema, with the same lenient
//! coercions upstream applies before checking (`"5"` → `5`, `null` for optional fields, ...).

use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use crate::types::{Tool, ToolCall};

/// A validation failure; the message is meant to be returned to the model.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ToolValidationError(pub String);

static VALIDATORS: LazyLock<Mutex<HashMap<String, Option<Arc<jsonschema::Validator>>>>> = LazyLock::new(Mutex::default);

/// Compiled validator of a schema, cached by its serialization. `None` for schemas that
/// do not compile.
fn validator(schema: &Value) -> Option<Arc<jsonschema::Validator>> {
    let key = schema.to_string();
    let mut cache = VALIDATORS.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(cached) = cache.get(&key) {
        return cached.clone();
    }
    if cache.len() > 1024 {
        cache.clear();
    }
    let compiled = jsonschema::validator_for(schema).ok().map(Arc::new);
    cache.insert(key, compiled.clone());
    compiled
}

fn check(schema: &Value, value: &Value) -> Option<bool> {
    validator(schema).map(|validator| validator.is_valid(value))
}

fn schema_types(schema: &Value) -> Vec<&str> {
    match schema.get("type") {
        Some(Value::String(kind)) => vec![kind.as_str()],
        Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

fn matches_json_type(value: &Value, kind: &str) -> bool {
    match kind {
        "number" => value.is_number(),
        "integer" => value.as_f64().is_some_and(|n| n.fract() == 0.0),
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "null" => value.is_null(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    }
}

fn number_value(number: f64) -> Value {
    if number.fract() == 0.0 && number.abs() < 9.007_199_254_740_992e15 {
        Value::from(number as i64)
    } else {
        serde_json::Number::from_f64(number).map_or(Value::Null, Value::Number)
    }
}

/// Coerce a primitive toward `kind`; returns the value unchanged when it does not apply.
fn coerce_primitive(value: &Value, kind: &str) -> Value {
    match (kind, value) {
        ("number" | "integer", Value::Null) => Value::from(0),
        ("number", Value::String(text)) if !text.trim().is_empty() => match text.trim().parse::<f64>() {
            Ok(number) if number.is_finite() => number_value(number),
            _ => value.clone(),
        },
        ("integer", Value::String(text)) if !text.trim().is_empty() => match text.trim().parse::<f64>() {
            Ok(number) if number.is_finite() && number.fract() == 0.0 => number_value(number),
            _ => value.clone(),
        },
        ("number" | "integer", Value::Bool(flag)) => Value::from(i64::from(*flag)),
        ("boolean", Value::Null) => Value::Bool(false),
        ("boolean", Value::String(text)) if text == "true" => Value::Bool(true),
        ("boolean", Value::String(text)) if text == "false" => Value::Bool(false),
        ("boolean", Value::Number(number)) if number.as_f64() == Some(1.0) => Value::Bool(true),
        ("boolean", Value::Number(number)) if number.as_f64() == Some(0.0) => Value::Bool(false),
        ("string", Value::Null) => Value::String(String::new()),
        ("string", Value::Number(number)) => Value::String(number.to_string()),
        ("string", Value::Bool(flag)) => Value::String(flag.to_string()),
        ("null", Value::String(text)) if text.is_empty() => Value::Null,
        ("null", Value::Number(number)) if number.as_f64() == Some(0.0) => Value::Null,
        ("null", Value::Bool(false)) => Value::Null,
        _ => value.clone(),
    }
}

fn coerce_union(value: Value, schemas: &[Value]) -> Value {
    if schemas.iter().any(|schema| check(schema, &value) == Some(true)) {
        return value;
    }
    for schema in schemas {
        let coerced = coerce_with_schema(value.clone(), schema);
        if check(schema, &coerced) == Some(true) {
            return coerced;
        }
    }
    value
}

/// Apply upstream's schema-guided coercions to `value`.
fn coerce_with_schema(value: Value, schema: &Value) -> Value {
    let mut next = value;
    if let Some(Value::Array(all_of)) = schema.get("allOf") {
        for nested in all_of {
            next = coerce_with_schema(next, nested);
        }
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(Value::Array(options)) = schema.get(key) {
            next = coerce_union(next, options);
        }
    }
    let kinds = schema_types(schema);
    let matches_member = kinds.len() > 1 && kinds.iter().any(|kind| matches_json_type(&next, kind));
    if !kinds.is_empty() && !matches_member {
        for kind in &kinds {
            let candidate = coerce_primitive(&next, kind);
            if candidate != next {
                next = candidate;
                break;
            }
        }
    }
    if kinds.contains(&"object")
        && let Value::Object(object) = &mut next
    {
        coerce_object(object, schema);
    }
    if kinds.contains(&"array")
        && let Value::Array(items) = &mut next
    {
        match schema.get("items") {
            Some(Value::Array(item_schemas)) => {
                for (item, item_schema) in items.iter_mut().zip(item_schemas) {
                    *item = coerce_with_schema(std::mem::take(item), item_schema);
                }
            }
            Some(item_schema @ Value::Object(_)) => {
                for item in items.iter_mut() {
                    *item = coerce_with_schema(std::mem::take(item), item_schema);
                }
            }
            _ => {}
        }
    }
    next
}

fn coerce_object(object: &mut Map<String, Value>, schema: &Value) {
    let properties = schema.get("properties").and_then(Value::as_object);
    if let Some(properties) = properties {
        for (key, property_schema) in properties {
            if let Some(value) = object.get_mut(key) {
                *value = coerce_with_schema(std::mem::take(value), property_schema);
            }
        }
    }
    if let Some(additional @ Value::Object(_)) = schema.get("additionalProperties") {
        for (key, value) in object.iter_mut() {
            if properties.is_some_and(|properties| properties.contains_key(key)) {
                continue;
            }
            *value = coerce_with_schema(std::mem::take(value), additional);
        }
    }
}

/// Drop `null` values of optional properties whose schema does not accept `null`.
fn normalize_optional_nulls(value: &mut Value, schema: &Value) {
    match value {
        Value::Array(items) => match schema.get("items") {
            Some(Value::Array(item_schemas)) => {
                for (item, item_schema) in items.iter_mut().zip(item_schemas) {
                    normalize_optional_nulls(item, item_schema);
                }
            }
            Some(item_schema) => {
                for item in items.iter_mut() {
                    normalize_optional_nulls(item, item_schema);
                }
            }
            None => {}
        },
        Value::Object(object) => {
            let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
                return;
            };
            let required: Vec<&str> = schema
                .get("required")
                .and_then(Value::as_array)
                .map(|required| required.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            for (key, property_schema) in properties {
                let Some(entry) = object.get_mut(key) else { continue };
                if entry.is_null()
                    && !required.contains(&key.as_str())
                    && property_schema.get("$ref").is_none()
                    && check(property_schema, &Value::Null) == Some(false)
                {
                    object.remove(key);
                } else {
                    normalize_optional_nulls(entry, property_schema);
                }
            }
        }
        _ => {}
    }
}

fn format_path(error: &jsonschema::ValidationError<'_>) -> String {
    let base = error
        .instance_path()
        .to_string()
        .trim_start_matches('/')
        .replace('/', ".");
    if let jsonschema::error::ValidationErrorKind::Required { property } = error.kind() {
        let property = property.as_str().map_or_else(|| property.to_string(), str::to_owned);
        return if base.is_empty() {
            property
        } else {
            format!("{base}.{property}")
        };
    }
    if base.is_empty() { "root".into() } else { base }
}

/// Validate a tool call's arguments against `tool.parameters` and return the (coerced)
/// arguments. The error message lists every violation and the received arguments.
pub fn validate_tool_arguments(tool: &Tool, call: &ToolCall) -> Result<Value, ToolValidationError> {
    if let Some(error) = &call.arguments_error {
        return Err(ToolValidationError(format!(
            "Invalid arguments for tool \"{}\": {error}",
            call.name
        )));
    }
    let mut args = Value::Object(call.arguments.clone());
    normalize_optional_nulls(&mut args, &tool.parameters);
    let args = coerce_with_schema(args, &tool.parameters);
    let Some(validator) = validator(&tool.parameters) else {
        // Upstream compiles any schema; an uncompilable schema cannot reject arguments.
        return Ok(args);
    };
    let errors: Vec<String> = validator
        .iter_errors(&args)
        .map(|error| format!("  - {}: {}", format_path(&error), error.masked()))
        .collect();
    if errors.is_empty() {
        return Ok(args);
    }
    let received = serde_json::to_string_pretty(&call.arguments).unwrap_or_default();
    Err(ToolValidationError(format!(
        "Validation failed for tool \"{}\":\n{}\n\nReceived arguments:\n{received}",
        call.name,
        errors.join("\n")
    )))
}

/// Find the tool named by the call and validate its arguments.
pub fn validate_tool_call(tools: &[Tool], call: &ToolCall) -> Result<Value, ToolValidationError> {
    let tool = tools
        .iter()
        .find(|tool| tool.name == call.name)
        .ok_or_else(|| ToolValidationError(format!("Tool \"{}\" not found", call.name)))?;
    validate_tool_arguments(tool, call)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(arguments: &Value) -> ToolCall {
        ToolCall {
            id: "1".into(),
            name: "t".into(),
            arguments: arguments.as_object().unwrap().clone(),
            ..ToolCall::default()
        }
    }

    fn tool() -> Tool {
        Tool::new(
            "t",
            "test",
            json!({
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "ratio": {"type": "number"},
                    "flag": {"type": "boolean"},
                    "name": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "mode": {"anyOf": [{"type": "integer"}, {"type": "string", "enum": ["auto"]}]},
                    "note": {"type": "string"}
                },
                "required": ["count"],
                "additionalProperties": false
            }),
        )
    }

    #[test]
    fn coerces_like_upstream() {
        let args = validate_tool_arguments(
            &tool(),
            &call(&json!({"count": "3", "ratio": "0.5", "flag": "true", "name": 7, "tags": [1, true], "mode": "2", "note": null})),
        )
        .unwrap();
        assert_eq!(
            args,
            json!({"count": 3, "ratio": 0.5, "flag": true, "name": "7", "tags": ["1", "true"], "mode": 2})
        );
    }

    #[test]
    fn reports_violations_with_paths() {
        let error = validate_tool_arguments(&tool(), &call(&json!({"flag": "maybe", "extra": 1}))).unwrap_err();
        assert!(error.0.starts_with("Validation failed for tool \"t\":"), "{}", error.0);
        assert!(error.0.contains("  - count:"), "{}", error.0);
        assert!(error.0.contains("  - flag:"), "{}", error.0);
        assert!(error.0.contains("Received arguments:"), "{}", error.0);
        let missing = validate_tool_call(
            &[tool()],
            &ToolCall {
                name: "x".into(),
                ..ToolCall::default()
            },
        )
        .unwrap_err();
        assert_eq!(missing.0, "Tool \"x\" not found");
    }

    #[test]
    fn arguments_error_fails_validation() {
        let mut bad = call(&json!({}));
        bad.arguments_error = Some("bad json".into());
        assert!(
            validate_tool_arguments(&tool(), &bad)
                .unwrap_err()
                .0
                .contains("bad json")
        );
    }
}
