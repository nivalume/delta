//! Provider-side constrained sampling for tools: strict JSON-schema tools and OpenAI
//! grammar (`custom`) tools.
//!
//! Port of upstream `api/constrained-sampling.ts`, shared by every API that sends strict or
//! grammar tools.

use serde_json::Value;
use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::types::{ConstrainedSampling, StrictMode, Tool};

/// Keywords that strict JSON-schema mode rejects.
const UNSUPPORTED_STRICT_SCHEMA_KEYS: [&str; 16] = [
    "$ref",
    "$defs",
    "definitions",
    "allOf",
    "oneOf",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
];

/// Whether a provider's strict mode rejects this schema keyword with this value (upstream
/// `UnsupportedStrictSchemaKeywordCheck`).
pub type UnsupportedStrictSchemaKeywordCheck = fn(&str, &Value) -> bool;

/// A schema that strict mode cannot express. The message names the offending construct.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnsupportedStrictSchema(pub String);

impl std::fmt::Display for UnsupportedStrictSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn unsupported(message: impl Into<String>) -> UnsupportedStrictSchema {
    UnsupportedStrictSchema(message.into())
}

fn is_structured_schema(schema: &Value) -> bool {
    let Some(object) = schema.as_object() else {
        return false;
    };
    let has_type = |name: &str| match object.get("type") {
        Some(Value::String(kind)) => kind == name,
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind.as_str() == Some(name)),
        _ => false,
    };
    has_type("object") || has_type("array") || object.contains_key("properties") || object.contains_key("items")
}

fn schema_allows_null(schema: &Value) -> bool {
    let Some(object) = schema.as_object() else {
        return false;
    };
    match object.get("type") {
        Some(Value::String(kind)) if kind == "null" => return true,
        Some(Value::Array(kinds)) if kinds.iter().any(|kind| kind.as_str() == Some("null")) => return true,
        _ => {}
    }
    if object.get("const") == Some(&Value::Null) {
        return true;
    }
    if let Some(Value::Array(values)) = object.get("enum")
        && values.contains(&Value::Null)
    {
        return true;
    }
    matches!(object.get("anyOf"), Some(Value::Array(variants)) if variants.iter().any(schema_allows_null))
}

fn make_node_strict(
    schema: &mut Value,
    is_unsupported_keyword: Option<UnsupportedStrictSchemaKeywordCheck>,
) -> std::result::Result<(), UnsupportedStrictSchema> {
    let Some(object) = schema.as_object_mut() else {
        return Err(unsupported("boolean schemas are unsupported"));
    };
    for key in UNSUPPORTED_STRICT_SCHEMA_KEYS {
        if object.contains_key(key) {
            return Err(unsupported(format!("{key} schemas are unsupported")));
        }
    }
    if let Some(is_unsupported_keyword) = is_unsupported_keyword {
        for (key, value) in object.iter() {
            if is_unsupported_keyword(key, value) {
                // `Value`'s `Display` is compact JSON, as upstream's `JSON.stringify(value)`.
                return Err(unsupported(format!("{key}: {value} is unsupported")));
            }
        }
    }

    if let Some(any_of) = object.get_mut("anyOf") {
        let Some(variants) = any_of.as_array_mut().filter(|variants| !variants.is_empty()) else {
            return Err(unsupported("anyOf must contain at least one schema"));
        };
        for variant in variants {
            if is_structured_schema(variant) {
                return Err(unsupported("object and array unions are unsupported"));
            }
            make_node_strict(variant, is_unsupported_keyword)?;
        }
    }

    if let Some(items) = object.get_mut("items") {
        if items.is_array() {
            return Err(unsupported("tuple schemas are unsupported"));
        }
        make_node_strict(items, is_unsupported_keyword)?;
    }

    let is_object_schema = object.get("type").and_then(Value::as_str) == Some("object");
    if object.contains_key("properties") && !is_object_schema {
        return Err(unsupported("properties require type object"));
    }
    if !is_object_schema {
        return Ok(());
    }
    if object
        .get("additionalProperties")
        .is_some_and(|value| *value != Value::Bool(false))
    {
        return Err(unsupported("schema-valued or true additionalProperties is unsupported"));
    }
    if object.get("properties").is_some_and(|value| !value.is_object()) {
        return Err(unsupported("object properties must be a schema map"));
    }
    let required: Vec<String> = match object.get("required") {
        None => Vec::new(),
        Some(Value::Array(keys)) => keys
            .iter()
            .map(|key| key.as_str().map(str::to_owned))
            .collect::<Option<_>>()
            .ok_or_else(|| unsupported("object required must be a string array"))?,
        Some(_) => return Err(unsupported("object required must be a string array")),
    };

    let names: Vec<String> = object
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| properties.keys().cloned().collect())
        .unwrap_or_default();
    if required.iter().any(|key| !names.contains(key)) {
        return Err(unsupported("required contains an unknown property"));
    }
    if let Some(properties) = object.get_mut("properties").and_then(Value::as_object_mut) {
        for (key, property) in properties.iter_mut() {
            make_node_strict(property, is_unsupported_keyword)?;
            if !required.contains(key) && !schema_allows_null(property) {
                let inner = std::mem::take(property);
                *property = serde_json::json!({"anyOf": [inner, {"type": "null"}]});
            }
        }
    }
    object.insert(
        "required".into(),
        Value::Array(names.into_iter().map(Value::String).collect()),
    );
    object.insert("additionalProperties".into(), Value::Bool(false));
    Ok(())
}

/// Convert a tool schema to the strict subset expected by provider constrained sampling.
/// The input is not modified. `is_unsupported_keyword` lets a provider reject extra
/// keywords its strict mode does not accept.
pub fn make_strict_json_schema(
    schema: &Value,
    is_unsupported_keyword: Option<UnsupportedStrictSchemaKeywordCheck>,
) -> std::result::Result<Value, UnsupportedStrictSchema> {
    let mut cloned = schema.clone();
    if !cloned.is_object() {
        return Err(unsupported("root schema must have type object"));
    }
    make_node_strict(&mut cloned, is_unsupported_keyword)?;
    if cloned.get("type").and_then(Value::as_str) != Some("object") {
        return Err(unsupported("root schema must have type object"));
    }
    Ok(cloned)
}

/// The parameters to send for a JSON-schema tool: the strict form when `strict` is
/// `Some(true)`, else the tool's own schema. As upstream, no provider keyword check runs
/// here: [`resolve_json_schema_strict_sampling`] has already applied it.
pub fn get_json_schema_tool_parameters(tool: &Tool, strict: Option<bool>) -> Result<Value> {
    if strict == Some(true) {
        make_strict_json_schema(&tool.parameters, None).map_err(|error| Error::msg(error.0))
    } else {
        Ok(tool.parameters.clone())
    }
}

/// Whether a JSON-schema tool is sent in strict mode: `Some(true)` when strict, `None`
/// otherwise. Fails when the tool requires strict sampling that cannot be honored.
/// `is_unsupported_keyword` lets a provider reject extra keywords its strict mode does not
/// accept, so `prefer` tools fall back to non-strict.
pub fn resolve_json_schema_strict_sampling(
    tool: &Tool,
    supports_strict_mode: bool,
    is_unsupported_keyword: Option<UnsupportedStrictSchemaKeywordCheck>,
) -> Result<Option<bool>> {
    let Some(ConstrainedSampling::JsonSchema { strict }) = &tool.constrained_sampling else {
        return Ok(None);
    };
    if supports_strict_mode {
        return match make_strict_json_schema(&tool.parameters, is_unsupported_keyword) {
            Ok(_) => Ok(Some(true)),
            Err(_) if *strict != StrictMode::Require => Ok(None),
            Err(error) => Err(Error::msg(format!(
                "Tool \"{}\" requires JSON-schema constrained sampling, but {error}.",
                tool.name
            ))),
        };
    }
    if *strict == StrictMode::Require {
        return Err(Error::msg(format!(
            "Tool \"{}\" requires JSON-schema constrained sampling, but strict tools are unsupported.",
            tool.name
        )));
    }
    Ok(None)
}

/// Grammar format of an OpenAI custom tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrammarFormat {
    Lark,
    Regex,
}

impl GrammarFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lark => "lark",
            Self::Regex => "regex",
        }
    }
}

/// A resolved grammar constraint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrammarConstrainedSampling {
    pub format: GrammarFormat,
    pub definition: String,
    /// The single required string property that carries the raw grammar input.
    pub input_property: String,
}

fn infer_grammar_input_property(tool: &Tool) -> std::result::Result<String, String> {
    let schema = &tool.parameters;
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return Err("grammar constrained sampling requires an object parameter schema".into());
    }
    let property = match schema.get("required") {
        Some(Value::Array(required)) if required.len() == 1 => required[0].as_str().map(str::to_owned),
        _ => None,
    }
    .ok_or_else(|| "grammar constrained sampling requires exactly one required string property".to_owned())?;
    let entry = schema
        .get("properties")
        .and_then(|properties| properties.get(&property))
        .filter(|entry| js_truthy(entry));
    let Some(entry) = entry else {
        return Err(format!(
            "grammar constrained sampling requires a properties entry for {property}"
        ));
    };
    if entry.get("type").and_then(Value::as_str) != Some("string") {
        return Err(format!(
            "grammar constrained sampling property {property} must have type string"
        ));
    }
    Ok(property)
}

/// The grammar constraint of a tool when the provider supports OpenAI grammar tools.
pub fn resolve_grammar_constrained_sampling(
    tool: &Tool,
    supports_openai_grammar_tools: bool,
) -> Result<Option<GrammarConstrainedSampling>> {
    let Some(ConstrainedSampling::Grammar { variants }) = &tool.constrained_sampling else {
        return Ok(None);
    };
    if !supports_openai_grammar_tools {
        return Ok(None);
    }
    let usable = |definition: &Option<String>| definition.clone().filter(|text| !text.trim().is_empty());
    let (format, definition) = match (usable(&variants.openai_lark), usable(&variants.openai_regex)) {
        (Some(lark), _) => (GrammarFormat::Lark, lark),
        (None, Some(regex)) => (GrammarFormat::Regex, regex),
        (None, None) => {
            return Err(Error::msg(format!(
                "Tool \"{}\" cannot use grammar constrained sampling: no supported grammar variant was provided.",
                tool.name
            )));
        }
    };
    let input_property = infer_grammar_input_property(tool).map_err(|message| {
        Error::msg(format!(
            "Tool \"{}\" cannot use grammar constrained sampling: {message}.",
            tool.name
        ))
    })?;
    Ok(Some(GrammarConstrainedSampling {
        format,
        definition,
        input_property,
    }))
}

/// Grammar tool name to its input property, for every grammar tool the provider accepts.
pub fn create_grammar_tool_input_properties(
    tools: &[Tool],
    supports_openai_grammar_tools: bool,
) -> Result<HashMap<String, String>> {
    let mut properties = HashMap::new();
    for tool in tools {
        if let Some(grammar) = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)? {
            properties.insert(tool.name.clone(), grammar.input_property);
        }
    }
    Ok(properties)
}

/// The raw grammar input of a stored grammar tool call.
pub fn get_grammar_tool_input<'a>(
    tool_name: &str,
    arguments: &'a serde_json::Map<String, Value>,
    input_property: &str,
) -> Result<&'a str> {
    arguments.get(input_property).and_then(Value::as_str).ok_or_else(|| {
        Error::msg(format!(
            "Grammar tool call \"{tool_name}\" requires argument \"{input_property}\" to be a string."
        ))
    })
}

/// Append-only JSON rendering of a streamed grammar input: `{"<property>":"<escaped input>"}`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GrammarToolInputJsonBuffer {
    pub input: String,
    pub started: bool,
    pub closed: bool,
}

/// The JSON text to append when the grammar input grows to `next_input` (and closes, when
/// `close`). `Ok(None)` when nothing needs to be appended.
pub fn append_grammar_tool_input_json_delta(
    buffer: &mut GrammarToolInputJsonBuffer,
    input_property: &str,
    next_input: &str,
    close: bool,
) -> Result<Option<String>> {
    if buffer.closed {
        if close && next_input == buffer.input {
            return Ok(None);
        }
        return Err(Error::msg(format!(
            "grammar tool input for property \"{input_property}\" changed after it was closed"
        )));
    }
    let Some(input_delta) = next_input.strip_prefix(buffer.input.as_str()) else {
        return Err(Error::msg(format!(
            "grammar tool input for property \"{input_property}\" changed non-monotonically"
        )));
    };
    if !close && input_delta.is_empty() {
        return Ok(None);
    }
    let mut delta = String::new();
    if !buffer.started {
        delta.push('{');
        delta.push_str(&json_string(input_property));
        delta.push_str(":\"");
        buffer.started = true;
    }
    let escaped = json_string(input_delta);
    delta.push_str(&escaped[1..escaped.len() - 1]);
    next_input.clone_into(&mut buffer.input);
    if close {
        delta.push_str("\"}");
        buffer.closed = true;
    }
    Ok(Some(delta))
}

/// `JSON.stringify` of a string.
fn json_string(text: &str) -> String {
    Value::String(text.to_owned()).to_string()
}

/// JavaScript truthiness of a JSON value (objects and arrays are truthy).
pub(crate) fn js_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|n| n != 0.0 && !n.is_nan()),
        Value::String(text) => !text.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GrammarVariants;
    use serde_json::json;

    fn tool(parameters: Value, constrained_sampling: Option<ConstrainedSampling>) -> Tool {
        Tool {
            name: "sample_tool".into(),
            description: "Sample tool".into(),
            parameters,
            constrained_sampling,
        }
    }

    fn prefer() -> ConstrainedSampling {
        ConstrainedSampling::JsonSchema {
            strict: StrictMode::Prefer,
        }
    }

    fn require() -> ConstrainedSampling {
        ConstrainedSampling::JsonSchema {
            strict: StrictMode::Require,
        }
    }

    #[test]
    fn derives_strict_schemas_without_changing_the_tool() {
        let parameters = json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "number"},
                "metadata": {"type": "object", "properties": {"enabled": {"type": "boolean"}}},
                "nullable": {"anyOf": [{"type": "string"}, {"type": "null"}]}
            },
            "required": ["path", "metadata"]
        });
        let strict = make_strict_json_schema(&parameters, None).unwrap();
        assert!(parameters.get("additionalProperties").is_none());
        assert_eq!(parameters["required"], json!(["path", "metadata"]));
        assert_eq!(
            strict,
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                    "metadata": {
                        "type": "object",
                        "properties": {"enabled": {"anyOf": [{"type": "boolean"}, {"type": "null"}]}},
                        "required": ["enabled"],
                        "additionalProperties": false
                    },
                    "nullable": {"anyOf": [{"type": "string"}, {"type": "null"}]}
                },
                "required": ["path", "offset", "metadata", "nullable"],
                "additionalProperties": false
            })
        );
    }

    #[test]
    fn falls_back_or_rejects_unsupported_schemas() {
        let cases = [
            (
                json!({"type": "object", "properties": {"metadata": {"type": "object", "properties": {}, "additionalProperties": {"type": "string"}}}, "required": ["metadata"]}),
                "additionalProperties is unsupported",
            ),
            (
                json!({"allOf": [{"type": "object"}, {"type": "object"}]}),
                "allOf schemas are unsupported",
            ),
            (
                json!({"type": "object", "properties": {"value": {"anyOf": [{"type": "object", "properties": {"nested": {"type": "string"}}}, {"type": "null"}]}}, "required": ["value"]}),
                "object and array unions are unsupported",
            ),
            (
                json!({"type": "object", "properties": {"child": {"$ref": "https://example.com/child.json"}}, "required": ["child"]}),
                "$ref schemas are unsupported",
            ),
        ];
        for (parameters, error) in cases {
            let message = make_strict_json_schema(&parameters, None).unwrap_err().0;
            assert!(message.contains(error), "{message}");
            let mut t = tool(parameters, Some(prefer()));
            assert_eq!(resolve_json_schema_strict_sampling(&t, true, None).unwrap(), None);
            t.constrained_sampling = Some(require());
            let message = resolve_json_schema_strict_sampling(&t, true, None)
                .unwrap_err()
                .to_string();
            assert!(message.contains(error), "{message}");
        }
    }

    #[test]
    fn strict_requirements_without_support() {
        let t = tool(json!({"type": "object", "properties": {}}), Some(require()));
        assert_eq!(
            resolve_json_schema_strict_sampling(&t, false, None)
                .unwrap_err()
                .to_string(),
            "Tool \"sample_tool\" requires JSON-schema constrained sampling, but strict tools are unsupported."
        );
        let t = tool(json!({"type": "object", "properties": {}}), Some(prefer()));
        assert_eq!(resolve_json_schema_strict_sampling(&t, false, None).unwrap(), None);
        assert_eq!(resolve_json_schema_strict_sampling(&t, true, None).unwrap(), Some(true));
    }

    #[test]
    fn strict_schemas_recurse_into_items_and_keep_nullable_types() {
        let schema = json!({"type": "object", "properties": {
            "a": {"type": "string"},
            "b": {"type": "array", "items": {"type": "object", "properties": {"c": {"type": ["number", "null"]}}}}
        }, "required": ["a"]});
        assert_eq!(
            make_strict_json_schema(&schema, None).unwrap(),
            json!({"type": "object", "properties": {
                "a": {"type": "string"},
                "b": {"anyOf": [
                    {"type": "array", "items": {"type": "object", "properties": {"c": {"type": ["number", "null"]}},
                        "required": ["c"], "additionalProperties": false}},
                    {"type": "null"}
                ]}
            }, "required": ["a", "b"], "additionalProperties": false})
        );
        assert_eq!(
            make_strict_json_schema(&json!({"type": "object"}), None).unwrap(),
            json!({"type": "object", "required": [], "additionalProperties": false})
        );
    }

    #[test]
    fn strict_schema_rejection_messages() {
        let cases = [
            (json!(true), "root schema must have type object"),
            (json!({"type": "string"}), "root schema must have type object"),
            (json!({"type": "object", "$defs": {}}), "$defs schemas are unsupported"),
            (json!({"type": "object", "oneOf": []}), "oneOf schemas are unsupported"),
            (
                json!({"type": "object", "properties": {"a": {"anyOf": [{"type": "object"}]}}}),
                "object and array unions are unsupported",
            ),
            (json!({"type": "array", "items": [{}]}), "tuple schemas are unsupported"),
            (
                json!({"type": "string", "properties": {}}),
                "properties require type object",
            ),
            (
                json!({"type": "object", "additionalProperties": true}),
                "schema-valued or true additionalProperties is unsupported",
            ),
            (
                json!({"type": "object", "properties": []}),
                "object properties must be a schema map",
            ),
            (
                json!({"type": "object", "required": ["x"]}),
                "required contains an unknown property",
            ),
            (
                json!({"type": "object", "required": [1]}),
                "object required must be a string array",
            ),
            (
                json!({"type": "object", "properties": {"a": false}}),
                "boolean schemas are unsupported",
            ),
            (
                json!({"type": "object", "anyOf": []}),
                "anyOf must contain at least one schema",
            ),
        ];
        for (schema, expected) in cases {
            assert_eq!(
                make_strict_json_schema(&schema, None).unwrap_err().0,
                expected,
                "{schema}"
            );
        }
    }

    #[test]
    fn provider_keyword_checks_reject_nested_keywords() {
        fn no_minimum(key: &str, value: &Value) -> bool {
            key == "minimum" || (key == "format" && value != "uri")
        }
        let schema = json!({"type": "object", "properties": {"a": {"type": "integer", "minimum": 1}}});
        assert!(make_strict_json_schema(&schema, None).is_ok());
        assert_eq!(
            make_strict_json_schema(&schema, Some(no_minimum)).unwrap_err().0,
            "minimum: 1 is unsupported"
        );
        let schema = json!({"type": "object", "properties": {"a": {"type": "array", "items": {"type": "string", "format": "regex"}}}});
        assert_eq!(
            make_strict_json_schema(&schema, Some(no_minimum)).unwrap_err().0,
            "format: \"regex\" is unsupported"
        );
        let schema = json!({"type": "object", "properties": {"a": {"type": "string", "format": "uri"}}});
        assert!(make_strict_json_schema(&schema, Some(no_minimum)).is_ok());

        // `prefer` falls back to non-strict; `require` names the keyword.
        let schema = json!({"type": "object", "properties": {"a": {"type": "number", "minimum": 0}}});
        let mut t = tool(schema, Some(prefer()));
        assert_eq!(resolve_json_schema_strict_sampling(&t, true, None).unwrap(), Some(true));
        assert_eq!(
            resolve_json_schema_strict_sampling(&t, true, Some(no_minimum)).unwrap(),
            None
        );
        t.constrained_sampling = Some(require());
        assert_eq!(
            resolve_json_schema_strict_sampling(&t, true, Some(no_minimum))
                .unwrap_err()
                .to_string(),
            "Tool \"sample_tool\" requires JSON-schema constrained sampling, but minimum: 0 is unsupported."
        );
    }

    #[test]
    fn resolves_grammar_tools() {
        let grammar = |variants: GrammarVariants| Some(ConstrainedSampling::Grammar { variants });
        let params = json!({"type": "object", "properties": {"payload": {"type": "string"}}, "required": ["payload"]});
        let t = tool(
            params.clone(),
            grammar(GrammarVariants {
                openai_lark: Some("start: /[a-z]+/".into()),
                openai_regex: Some("x".into()),
            }),
        );
        assert_eq!(
            resolve_grammar_constrained_sampling(&t, true).unwrap(),
            Some(GrammarConstrainedSampling {
                format: GrammarFormat::Lark,
                definition: "start: /[a-z]+/".into(),
                input_property: "payload".into()
            })
        );
        assert_eq!(resolve_grammar_constrained_sampling(&t, false).unwrap(), None);
        let t = tool(params.clone(), grammar(GrammarVariants::default()));
        assert_eq!(
            resolve_grammar_constrained_sampling(&t, true).unwrap_err().to_string(),
            "Tool \"sample_tool\" cannot use grammar constrained sampling: no supported grammar variant was provided."
        );
        let t = tool(
            json!({"type": "object", "properties": {"payload": {"type": "number"}}, "required": ["payload"]}),
            grammar(GrammarVariants {
                openai_lark: None,
                openai_regex: Some("[0-9]+".into()),
            }),
        );
        assert_eq!(
            resolve_grammar_constrained_sampling(&t, true).unwrap_err().to_string(),
            "Tool \"sample_tool\" cannot use grammar constrained sampling: grammar constrained sampling property payload must have type string."
        );
    }

    #[test]
    fn grammar_input_json_deltas_are_append_only() {
        let mut buffer = GrammarToolInputJsonBuffer::default();
        let mut text = String::new();
        for (input, close) in [("a\"b", false), ("a\"b", false), ("a\"bc\n", false), ("a\"bc\n", true)] {
            if let Some(delta) = append_grammar_tool_input_json_delta(&mut buffer, "patch", input, close).unwrap() {
                text.push_str(&delta);
            }
        }
        assert_eq!(text, r#"{"patch":"a\"bc\n"}"#);
        assert_eq!(
            serde_json::from_str::<Value>(&text).unwrap(),
            json!({"patch": "a\"bc\n"})
        );
        assert!(
            append_grammar_tool_input_json_delta(&mut buffer, "patch", "a\"bc\n", true)
                .unwrap()
                .is_none()
        );
        assert!(append_grammar_tool_input_json_delta(&mut buffer, "patch", "other", false).is_err());
        let mut fresh = GrammarToolInputJsonBuffer::default();
        assert_eq!(
            append_grammar_tool_input_json_delta(&mut fresh, "input", "", true)
                .unwrap()
                .as_deref(),
            Some(r#"{"input":""}"#)
        );
    }
}
