//! TypeScript declarations of the script API for model-facing text. Port of upstream
//! `declarations.ts`: tools become members of `declare const tools`, globals `declare
//! function` statements, and `ns.member` globals members of `declare const ns`. Schemas only
//! shape the text; values are not validated against them.

use std::collections::HashSet;

use serde_json::{Map, Value};

use crate::identifier::{is_identifier, to_codemode_identifier};
use crate::sandbox::CodemodeTool;

const INDENT: &str = "  ";
/// Largest rendered input type, in UTF-16 units, before it becomes `unknown`.
pub const DEFAULT_INPUT_SCHEMA_MAX_CHARS: usize = 16_000;
/// Local `$ref` expansions per rendered schema, so shared definitions cannot blow up the output.
const MAX_REF_EXPANSIONS: usize = 32;
/// JSON Schema keywords with a number or string value shown as `@keyword value` property tags.
const TAG_KEYWORDS: [&str; 11] = [
    "minimum",
    "exclusiveMinimum",
    "maximum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minItems",
    "maxItems",
];
/// Tag keywords whose values all hold at once, so only the largest (lower) or smallest (upper)
/// matters.
const LOWER_BOUNDS: [&str; 4] = ["minimum", "exclusiveMinimum", "minLength", "minItems"];
const UPPER_BOUNDS: [&str; 4] = ["maximum", "exclusiveMaximum", "maxLength", "maxItems"];
/// Longest property tag; longer ones (large defaults, long patterns) are left out.
const MAX_TAG_CHARS: usize = 80;

/// A tool as declarations show it.
#[derive(Clone, Copy, Debug, Default)]
pub struct ToolDeclaration<'a> {
    pub name: &'a str,
    pub description: &'a str,
    pub input_schema: Option<&'a Value>,
    pub output_schema: Option<&'a Value>,
}

impl<'a> ToolDeclaration<'a> {
    pub fn of(tool: &'a dyn CodemodeTool) -> Self {
        Self {
            name: tool.name(),
            description: tool.description(),
            input_schema: tool.input_schema(),
            output_schema: tool.output_schema(),
        }
    }
}

/// A global as declarations show it.
#[derive(Clone, Copy, Debug, Default)]
pub struct GlobalDeclaration<'a> {
    pub name: &'a str,
    pub description: &'a str,
    pub input_schema: Option<&'a Value>,
    pub output_schema: Option<&'a Value>,
    /// The TypeScript parameter list and return type, replacing the rendering from the schemas.
    pub signature: Option<&'a str>,
}

impl<'a> GlobalDeclaration<'a> {
    pub fn of(global: &'a dyn CodemodeTool) -> Self {
        Self {
            name: global.name(),
            description: global.description(),
            input_schema: global.input_schema(),
            output_schema: global.output_schema(),
            signature: global.signature(),
        }
    }
}

fn utf16_len(text: &str) -> usize {
    text.encode_utf16().count()
}

/// A number as JavaScript's `String(number)` writes it, for the common cases.
fn js_number(number: &serde_json::Number) -> String {
    if let Some(integer) = number.as_i64() {
        return integer.to_string();
    }
    if let Some(integer) = number.as_u64() {
        return integer.to_string();
    }
    let value = number.as_f64().unwrap_or(f64::NAN);
    if value.fract() == 0.0 && value.abs() < 1e21 {
        return format!("{value:.0}");
    }
    value.to_string()
}

/// `JSON.stringify(value)`, with numbers as JavaScript writes them.
fn js_json(value: &Value) -> String {
    match value {
        Value::Number(number) => js_number(number),
        Value::Array(items) => format!("[{}]", items.iter().map(js_json).collect::<Vec<_>>().join(",")),
        Value::Object(fields) => format!(
            "{{{}}}",
            fields
                .iter()
                .map(|(key, value)| format!("{}:{}", Value::String(key.clone()), js_json(value)))
                .collect::<Vec<_>>()
                .join(",")
        ),
        other => other.to_string(),
    }
}

/// Renders TypeScript declarations of the tools and globals.
pub fn render_declarations(tools: &[ToolDeclaration<'_>], globals: &[GlobalDeclaration<'_>]) -> String {
    let mut sections = Vec::new();
    if !tools.is_empty() {
        let members: Vec<String> = tools
            .iter()
            .map(|tool| {
                format!(
                    "{}{INDENT}{}",
                    doc_comment(tool.description, INDENT),
                    render_tool_signature(tool, None)
                )
            })
            .collect();
        sections.push(format!("declare const tools: {{\n{}\n}};", members.join("\n")));
    }
    let mut namespaces: Vec<(&str, Vec<String>)> = Vec::new();
    for global in globals {
        let Some((namespace, member)) = global.name.split_once('.') else {
            sections.push(render_global(&format!("declare function {}", global.name), global, ""));
            continue;
        };
        let rendered = render_global(member, global, INDENT);
        match namespaces.iter_mut().find(|(name, _)| *name == namespace) {
            Some((_, members)) => members.push(rendered),
            None => namespaces.push((namespace, vec![rendered])),
        }
    }
    for (namespace, members) in namespaces {
        sections.push(format!("declare const {namespace}: {{\n{}\n}};", members.join("\n")));
    }
    sections.join("\n\n")
}

/// One tool as a member of the `tools` object: `name(args: T): Promise<R>;` with the name as the
/// identifier scripts use. Input types longer than `input_max_chars` (default
/// [`DEFAULT_INPUT_SCHEMA_MAX_CHARS`]) render as `unknown`.
pub fn render_tool_signature(tool: &ToolDeclaration<'_>, input_max_chars: Option<usize>) -> String {
    let input = tool.input_schema.map_or_else(
        || "unknown".to_owned(),
        |schema| schema_to_type(schema, Some(input_max_chars.unwrap_or(DEFAULT_INPUT_SCHEMA_MAX_CHARS))),
    );
    format!(
        "{}(args: {input}): Promise<{}>;",
        to_codemode_identifier(tool.name),
        render_tool_output_type(tool.output_schema)
    )
}

/// A tool's description followed by its declaration, for tool listings.
pub fn render_tool_sample(tool: &ToolDeclaration<'_>, input_max_chars: Option<usize>) -> String {
    let declaration = format!(
        "declare const tools: {{ {} }};",
        render_tool_signature(tool, input_max_chars)
    );
    format!(
        "{}\n\ncodemode tool declaration:\n```ts\n{declaration}\n```",
        tool.description.trim()
    )
}

/// The type a tool call resolves to: the output schema's type, `unknown` without a schema.
pub fn render_tool_output_type(schema: Option<&Value>) -> String {
    schema.map_or_else(|| "unknown".to_owned(), |schema| schema_to_type(schema, None))
}

fn render_global(head: &str, global: &GlobalDeclaration<'_>, indent: &str) -> String {
    let comment = doc_comment(global.description, indent);
    if let Some(signature) = global.signature {
        return format!("{comment}{indent}{head}{signature};");
    }
    let input = global
        .input_schema
        .map_or_else(|| "unknown".to_owned(), |schema| schema_to_type(schema, None));
    let output = global
        .output_schema
        .map_or_else(|| "unknown".to_owned(), |schema| schema_to_type(schema, None));
    format!("{comment}{indent}{head}(args: {input}): Promise<{output}>;")
}

fn doc_comment(description: &str, indent: &str) -> String {
    let text = description.trim();
    if text.is_empty() {
        return String::new();
    }
    let text = text.replace("*/", "*\\/");
    let lines: Vec<&str> = text
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .collect();
    if let [line] = lines.as_slice() {
        return format!("{indent}/** {line} */\n");
    }
    let body: Vec<String> = lines
        .iter()
        .map(|line| {
            if line.is_empty() {
                format!("{indent} *")
            } else {
                format!("{indent} * {line}")
            }
        })
        .collect();
    format!("{indent}/**\n{}\n{indent} */\n", body.join("\n"))
}

fn property_key(name: &str) -> String {
    if is_identifier(name) {
        name.to_owned()
    } else {
        Value::String(name.to_owned()).to_string()
    }
}

fn union(types: Vec<String>) -> String {
    let mut unique: Vec<String> = Vec::new();
    for entry in types {
        if !unique.contains(&entry) {
            unique.push(entry);
        }
    }
    if unique.iter().any(|entry| entry == "unknown") {
        return "unknown".to_owned();
    }
    if unique.is_empty() {
        "never".to_owned()
    } else {
        unique.join(" | ")
    }
}

/// Converts a JSON Schema to a TypeScript type expression: objects on one line (`{ a: string;
/// b?: number; }`) with properties sorted by name, or one property per line with `//` comments
/// when a property has a description or constraints the type cannot express (`// @integer
/// @minimum 1 @default 20`); `Array<T>` for arrays. Local references resolve against `schema`;
/// recursive and remote references render as `unknown`. A result longer than `max_chars` is
/// rendered again without the constraint tags, and is `unknown` if it is still too long.
pub fn schema_to_type(schema: &Value, max_chars: Option<usize>) -> String {
    let render = |tags: bool| {
        let mut context = SchemaContext {
            root: schema,
            resolving: HashSet::new(),
            expansions: 0,
            tags,
        };
        to_type(schema, &mut context)
    };
    let rendered = render(true);
    let Some(max_chars) = max_chars else { return rendered };
    if utf16_len(&rendered) <= max_chars {
        return rendered;
    }
    let untagged = render(false);
    if utf16_len(&untagged) <= max_chars {
        untagged
    } else {
        "unknown".to_owned()
    }
}

struct SchemaContext<'a> {
    root: &'a Value,
    /// References being expanded on the current path, to stop at recursive types.
    resolving: HashSet<String>,
    expansions: usize,
    /// Whether property comments include constraint tags.
    tags: bool,
}

fn decode_segment(segment: &str) -> String {
    let mut bytes = Vec::with_capacity(segment.len());
    let raw = segment.as_bytes();
    let mut index = 0;
    while index < raw.len() {
        if raw[index] == b'%'
            && index + 2 < raw.len()
            && let Ok(byte) = u8::from_str_radix(&segment[index + 1..index + 3], 16)
        {
            bytes.push(byte);
            index += 3;
            continue;
        }
        bytes.push(raw[index]);
        index += 1;
    }
    String::from_utf8_lossy(&bytes).replace("~1", "/").replace("~0", "~")
}

fn resolve_ref<'a>(reference: &str, root: &'a Value) -> Option<&'a Value> {
    if reference != "#" && !reference.starts_with("#/") {
        return None;
    }
    let mut current = root;
    for segment in reference
        .get(2..)
        .unwrap_or_default()
        .split('/')
        .filter(|segment| !segment.is_empty())
    {
        current = current.as_object()?.get(&decode_segment(segment))?;
    }
    matches!(current, Value::Bool(_) | Value::Object(_)).then_some(current)
}

fn to_type(schema: &Value, context: &mut SchemaContext<'_>) -> String {
    let fields = match schema {
        Value::Bool(false) => return "never".to_owned(),
        Value::Object(fields) => fields,
        _ => return "unknown".to_owned(),
    };
    if let Some(Value::String(reference)) = fields.get("$ref") {
        if context.resolving.contains(reference) || context.expansions >= MAX_REF_EXPANSIONS {
            return "unknown".to_owned();
        }
        let Some(target) = resolve_ref(reference, context.root) else {
            return "unknown".to_owned();
        };
        context.expansions += 1;
        context.resolving.insert(reference.clone());
        let rendered = to_type(target, context);
        context.resolving.remove(reference);
        return rendered;
    }
    if let Some(constant) = fields.get("const") {
        return js_json(constant);
    }
    if let Some(Value::Array(values)) = fields.get("enum") {
        return union(values.iter().map(js_json).collect());
    }
    let variants = match (fields.get("anyOf"), fields.get("oneOf")) {
        (Some(Value::Array(variants)), _) | (_, Some(Value::Array(variants))) => Some(variants),
        _ => None,
    };
    if let Some(variants) = variants {
        return union(variants.iter().map(|variant| to_type(variant, context)).collect());
    }
    if let Some(Value::Array(parts)) = fields.get("allOf") {
        let parts: Vec<String> = parts
            .iter()
            .map(|part| to_type(part, context))
            .filter(|part| part != "unknown")
            .collect();
        if parts.is_empty() {
            return "unknown".to_owned();
        }
        return parts
            .iter()
            .map(|part| {
                if part.contains(" | ") {
                    format!("({part})")
                } else {
                    part.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(" & ");
    }
    match fields.get("type") {
        Some(Value::Array(types)) => {
            let rendered = types
                .iter()
                .map(|entry| {
                    let mut variant = fields.clone();
                    variant.insert("type".to_owned(), entry.clone());
                    to_type(&Value::Object(variant), context)
                })
                .collect();
            union(rendered)
        }
        Some(Value::String(kind)) => match kind.as_str() {
            "string" => "string".to_owned(),
            "number" | "integer" => "number".to_owned(),
            "boolean" => "boolean".to_owned(),
            "null" => "null".to_owned(),
            "array" => array_type(fields, context),
            "object" => object_type(fields, context),
            _ => "unknown".to_owned(),
        },
        None => {
            if ["properties", "additionalProperties", "required"]
                .iter()
                .any(|key| fields.contains_key(*key))
            {
                return object_type(fields, context);
            }
            if fields.contains_key("items") || fields.contains_key("prefixItems") {
                return array_type(fields, context);
            }
            "unknown".to_owned()
        }
        Some(_) => "unknown".to_owned(),
    }
}

fn array_type(fields: &Map<String, Value>, context: &mut SchemaContext<'_>) -> String {
    if let Some(items) = fields.get("items")
        && !items.is_array()
    {
        return format!("Array<{}>", to_type(items, context));
    }
    let tuple = match (fields.get("prefixItems"), fields.get("items")) {
        (Some(Value::Array(items)), _) | (_, Some(Value::Array(items))) => items.as_slice(),
        _ => &[],
    };
    if tuple.is_empty() {
        return "unknown[]".to_owned();
    }
    format!(
        "[{}]",
        tuple
            .iter()
            .map(|item| to_type(item, context))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn is_line_terminator(c: char) -> bool {
    matches!(c, '\r' | '\n' | '\u{2028}' | '\u{2029}')
}

fn description_of(property: &Value) -> String {
    property
        .get("description")
        .and_then(Value::as_str)
        .map(|description| description.trim().to_owned())
        .unwrap_or_default()
}

/// Constraints of a property its TypeScript type cannot express, as JSDoc-style tags:
/// `@integer @minimum 1 @maximum 100 @default 20`. Empty when there are none.
fn constraint_tags(schema: &Value) -> String {
    let Some(property) = schema.as_object() else {
        return String::new();
    };
    // Optional fields such as Pydantic's `int | None` keep their constraints in the non-null
    // branch; the branch's constraints apply together with the property's, so both are read.
    let variants = property
        .get("anyOf")
        .filter(|variants| variants.is_array())
        .or_else(|| property.get("oneOf"))
        .and_then(Value::as_array);
    let non_null: Vec<&Value> = variants
        .map(|variants| {
            variants
                .iter()
                .filter(|variant| variant.get("type").and_then(Value::as_str) != Some("null"))
                .collect()
        })
        .unwrap_or_default();
    let mut branches = vec![property];
    if let [branch] = non_null.as_slice()
        && let Some(branch) = branch.as_object()
    {
        branches.push(branch);
    }
    // Draft-04 and OpenAPI 3.0 make `minimum`/`maximum` exclusive with a boolean
    // `exclusiveMinimum`/`exclusiveMaximum`.
    let schemas: Vec<Map<String, Value>> = branches
        .into_iter()
        .map(|schema| {
            let mut schema = schema.clone();
            for (exclusive, bound) in [("exclusiveMinimum", "minimum"), ("exclusiveMaximum", "maximum")] {
                if schema.get(exclusive) == Some(&Value::Bool(true)) {
                    let value = schema.remove(bound).unwrap_or(Value::Null);
                    schema.insert(exclusive.to_owned(), value);
                }
            }
            schema
        })
        .collect();

    let mut tags: Vec<String> = Vec::new();
    let is_integer = |kind: Option<&Value>| match kind {
        Some(Value::String(kind)) => kind == "integer",
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind.as_str() == Some("integer")),
        _ => false,
    };
    if schemas.iter().any(|schema| is_integer(schema.get("type"))) {
        tags.push("@integer".to_owned());
    }
    for keyword in TAG_KEYWORDS {
        let values: Vec<&Value> = schemas.iter().filter_map(|schema| schema.get(keyword)).collect();
        let lower = LOWER_BOUNDS.contains(&keyword);
        if lower || UPPER_BOUNDS.contains(&keyword) {
            let numbers: Vec<&serde_json::Number> = values
                .iter()
                .filter_map(|value| match value {
                    Value::Number(number) => Some(number),
                    _ => None,
                })
                .collect();
            let chosen = numbers.into_iter().reduce(|best, number| {
                let (a, b) = (best.as_f64().unwrap_or(f64::NAN), number.as_f64().unwrap_or(f64::NAN));
                if (lower && b > a) || (!lower && b < a) {
                    number
                } else {
                    best
                }
            });
            if let Some(number) = chosen {
                tags.push(format!("@{keyword} {}", js_number(number)));
            }
            continue;
        }
        let mut seen: Vec<&Value> = Vec::new();
        for value in values {
            if seen.contains(&value) {
                continue;
            }
            seen.push(value);
            match value {
                Value::Number(number) => tags.push(format!("@{keyword} {}", js_number(number))),
                Value::String(text) => tags.push(format!("@{keyword} {text}")),
                _ => {}
            }
        }
    }
    if schemas
        .iter()
        .any(|schema| schema.get("uniqueItems") == Some(&Value::Bool(true)))
    {
        tags.push("@uniqueItems".to_owned());
    }
    // Annotations, unlike constraints, come from the property itself.
    if let Some(default) = property.get("default").filter(|default| !default.is_null()) {
        tags.push(format!("@default {}", js_json(default)));
    }
    if property.get("deprecated") == Some(&Value::Bool(true)) {
        tags.push("@deprecated".to_owned());
    }
    tags.into_iter()
        .filter(|tag| utf16_len(tag) <= MAX_TAG_CHARS)
        .map(|tag| {
            tag.chars()
                .map(|c| {
                    if is_line_terminator(c) {
                        format!("\\u{:04x}", u32::from(c))
                    } else {
                        c.to_string()
                    }
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn object_type(fields: &Map<String, Value>, context: &mut SchemaContext<'_>) -> String {
    let empty = Map::new();
    let properties = fields.get("properties").and_then(Value::as_object).unwrap_or(&empty);
    let required: Vec<&str> = fields
        .get("required")
        .and_then(Value::as_array)
        .map(|required| required.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut names: Vec<&String> = properties.keys().collect();
    names.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
    let mut members: Vec<String> = names
        .iter()
        .map(|name| {
            let optional = if required.contains(&name.as_str()) { "" } else { "?" };
            format!(
                "{}{optional}: {};",
                property_key(name),
                to_type(&properties[name.as_str()], context)
            )
        })
        .collect();
    match fields.get("additionalProperties") {
        Some(Value::Bool(false)) => {}
        Some(additional) => {
            let rendered = if additional == &Value::Bool(true) {
                "unknown".to_owned()
            } else {
                to_type(additional, context)
            };
            members.push(format!("[key: string]: {rendered};"));
        }
        None if names.is_empty() => members.push("[key: string]: unknown;".to_owned()),
        None => {}
    }
    if members.is_empty() {
        return "{}".to_owned();
    }
    let comments: Vec<Vec<String>> = names
        .iter()
        .map(|name| {
            let property = &properties[name.as_str()];
            let mut lines: Vec<String> = description_of(property)
                .split(is_line_terminator)
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(str::to_owned)
                .collect();
            let tags = if context.tags {
                constraint_tags(property)
            } else {
                String::new()
            };
            if !tags.is_empty() {
                lines.push(tags);
            }
            lines
        })
        .collect();
    if comments.iter().all(Vec::is_empty) {
        return format!("{{ {} }}", members.join(" "));
    }
    let mut lines = vec!["{".to_owned()];
    for (index, comment) in comments.iter().enumerate() {
        for line in comment {
            lines.push(format!("{INDENT}// {line}"));
        }
        lines.push(format!(
            "{INDENT}{}",
            members[index].replace('\n', &format!("\n{INDENT}"))
        ));
    }
    for member in &members[names.len()..] {
        lines.push(format!("{INDENT}{member}"));
    }
    lines.push("}".to_owned());
    lines.join("\n")
}
