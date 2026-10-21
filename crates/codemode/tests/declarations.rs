//! Declarations and source parsing, ported from upstream `declarations.test.ts` and
//! `source.test.ts`.

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

use delta_codemode::{
    GlobalDeclaration, SourceOptions, ToolDeclaration, parse_codemode_source, render_declarations, render_tool_sample,
    render_tool_signature, schema_to_type,
};
use pretty_assertions::assert_eq;
use serde_json::{Value, json};

fn ty(schema: &Value) -> String {
    schema_to_type(schema, None)
}

#[test]
fn primitives_literals_and_unions() {
    assert_eq!(ty(&json!({ "type": "string" })), "string");
    assert_eq!(ty(&json!({ "type": "integer" })), "number");
    assert_eq!(ty(&json!({ "type": ["string", "null"] })), "string | null");
    assert_eq!(ty(&json!({ "const": "a" })), "\"a\"");
    assert_eq!(ty(&json!({ "enum": ["a", 1, null] })), "\"a\" | 1 | null");
    assert_eq!(
        ty(&json!({ "anyOf": [{ "type": "string" }, { "type": "number" }] })),
        "string | number"
    );
    assert_eq!(ty(&json!({ "anyOf": [{ "type": "string" }, {}] })), "unknown");
    assert_eq!(
        ty(&json!({ "allOf": [{ "anyOf": [{ "type": "string" }, { "type": "number" }] }, { "const": 1 }] })),
        "(string | number) & 1"
    );
    assert_eq!(ty(&json!({ "$ref": "#/defs/x" })), "unknown");
    assert_eq!(ty(&json!(true)), "unknown");
    assert_eq!(ty(&json!(false)), "never");
}

#[test]
fn objects_on_one_line_with_sorted_properties() {
    assert_eq!(
        ty(&json!({
            "type": "object",
            "properties": { "city": { "type": "string" }, "max-lines": { "type": "number" } },
            "required": ["city"],
            "additionalProperties": false
        })),
        "{ city: string; \"max-lines\"?: number; }"
    );
    assert_eq!(
        ty(&json!({ "type": "object", "additionalProperties": { "type": "number" } })),
        "{ [key: string]: number; }"
    );
    assert_eq!(ty(&json!({ "type": "object" })), "{ [key: string]: unknown; }");
    assert_eq!(
        ty(&json!({ "type": "object", "properties": {}, "additionalProperties": false })),
        "{}"
    );
}

#[test]
fn property_descriptions_on_comment_lines() {
    assert_eq!(
        ty(&json!({
            "type": "object",
            "properties": {
                "weather": {
                    "type": "array",
                    "description": "look up weather for a given list of locations",
                    "items": { "type": "object", "properties": { "location": { "type": "string" } }, "required": ["location"] }
                }
            },
            "required": ["weather"]
        })),
        "{\n  // look up weather for a given list of locations\n  weather: Array<{ location: string; }>;\n}"
    );
    assert_eq!(
        ty(&json!({
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "description": "Outer",
                    "properties": { "inner": { "type": "string", "description": "Inner" } }
                }
            }
        })),
        "{\n  // Outer\n  outer?: {\n    // Inner\n    inner?: string;\n  };\n}"
    );
}

#[test]
fn constraints_on_property_comment_lines() {
    assert_eq!(
        ty(&json!({
            "type": "object",
            "properties": {
                "timeoutMs": { "type": "integer", "minimum": 0, "maximum": 30000, "default": 0, "description": "Wait in milliseconds." },
                "id": { "type": "string", "pattern": "^[a-z]+$", "maxLength": 8 },
                "tags": { "type": "array", "items": { "type": "string" }, "maxItems": 5, "uniqueItems": true, "deprecated": true },
                "name": { "type": "string" }
            },
            "required": ["id"]
        })),
        [
            "{",
            "  // @maxLength 8 @pattern ^[a-z]+$",
            "  id: string;",
            "  name?: string;",
            "  // @maxItems 5 @uniqueItems @deprecated",
            "  tags?: Array<string>;",
            "  // Wait in milliseconds.",
            "  // @integer @minimum 0 @maximum 30000 @default 0",
            "  timeoutMs?: number;",
            "}",
        ]
        .join("\n")
    );
}

#[test]
fn constraints_of_optional_fields_from_the_non_null_branch() {
    assert_eq!(
        ty(&json!({
            "type": "object",
            "properties": {
                "limit": { "anyOf": [{ "type": "integer", "minimum": 1, "maximum": 100 }, { "type": "null" }], "default": null, "title": "Limit" }
            }
        })),
        "{\n  // @integer @minimum 1 @maximum 100\n  limit?: number | null;\n}"
    );
    assert_eq!(
        ty(&json!({
            "type": "object",
            "properties": {
                "n": {
                    "anyOf": [{ "type": "integer", "minimum": 10, "maximum": 50, "format": "int32" }, { "type": "null" }],
                    "minimum": 0,
                    "maximum": 20,
                    "format": "port"
                }
            }
        })),
        "{\n  // @integer @minimum 10 @maximum 20 @format port @format int32\n  n?: number | null;\n}"
    );
}

#[test]
fn draft_04_boolean_exclusive_bounds() {
    assert_eq!(
        ty(&json!({
            "type": "object",
            "properties": {
                "n": { "type": "integer", "minimum": 0, "exclusiveMinimum": true, "maximum": 10, "exclusiveMaximum": false },
                "m": { "anyOf": [{ "type": "number", "maximum": 5, "exclusiveMaximum": true }, { "type": "null" }], "maximum": 8 }
            }
        })),
        [
            "{",
            "  // @maximum 8 @exclusiveMaximum 5",
            "  m?: number | null;",
            "  // @integer @exclusiveMinimum 0 @maximum 10",
            "  n?: number;",
            "}",
        ]
        .join("\n")
    );
}

#[test]
fn long_tags_left_out_and_line_terminators_escaped() {
    assert_eq!(
        ty(
            &json!({ "type": "object", "properties": { "config": { "type": "object", "default": { "text": "x".repeat(100) } } } })
        ),
        "{ config?: { [key: string]: unknown; }; }"
    );
    assert_eq!(
        ty(&json!({
            "type": "object",
            "properties": {
                "sep": { "type": "string", "pattern": "^[\r\n\u{2028}\u{2029}]$", "default": "\u{2028}", "description": "a\u{2028}b" }
            }
        })),
        [
            "{",
            "  // a",
            "  // b",
            "  // @pattern ^[\\u000d\\u000a\\u2028\\u2029]$ @default \"\\u2028\"",
            "  sep?: string;",
            "}",
        ]
        .join("\n")
    );
}

#[test]
fn constraint_tags_dropped_before_unknown() {
    let schema = json!({ "type": "object", "properties": { "n": { "type": "integer", "minimum": 1 } } });
    assert_eq!(ty(&schema), "{\n  // @integer @minimum 1\n  n?: number;\n}");
    assert_eq!(schema_to_type(&schema, Some(20)), "{ n?: number; }");
    assert_eq!(schema_to_type(&schema, Some(10)), "unknown");
}

#[test]
fn local_references_resolve_and_recursion_stops() {
    let schema = json!({
        "type": "object",
        "properties": {
            "item": { "$ref": "#/$defs/Item" },
            "legacy": { "$ref": "#/definitions/Legacy" },
            "remote": { "$ref": "https://example.com/schema.json" }
        },
        "required": ["item"],
        "$defs": {
            "Item": { "type": "object", "properties": { "id": { "type": "string" }, "parent": { "$ref": "#/$defs/Item" } }, "required": ["id"] }
        },
        "definitions": { "Legacy": { "enum": ["a", "b"] } }
    });
    assert_eq!(
        ty(&schema),
        "{ item: { id: string; parent?: unknown; }; legacy?: \"a\" | \"b\"; remote?: unknown; }"
    );
}

#[test]
fn arrays_tuples_and_budgets() {
    assert_eq!(
        ty(&json!({ "type": "array", "items": { "type": "string" } })),
        "Array<string>"
    );
    assert_eq!(
        ty(&json!({ "type": "array", "prefixItems": [{ "type": "string" }, { "type": "number" }] })),
        "[string, number]"
    );
    assert_eq!(ty(&json!({ "type": "array" })), "unknown[]");
    let properties: serde_json::Map<String, Value> = (0..50)
        .map(|index| (format!("field{index}"), json!({ "type": "string" })))
        .collect();
    let schema = json!({ "type": "object", "properties": properties });
    assert_eq!(schema_to_type(&schema, Some(100)), "unknown");
    assert!(ty(&schema).contains("field49?: string;"));
}

#[test]
fn tool_signatures_samples_and_declarations() {
    let input = json!({ "type": "object", "properties": { "city": { "type": "string" } }, "required": ["city"], "additionalProperties": false });
    let output = json!({ "type": "object", "properties": { "ok": { "type": "boolean" } }, "required": ["ok"] });
    let tool = ToolDeclaration {
        name: "hidden-dynamic-tool",
        input_schema: Some(&input),
        output_schema: Some(&output),
        ..ToolDeclaration::default()
    };
    assert_eq!(
        render_tool_signature(&tool, None),
        "hidden_dynamic_tool(args: { city: string; }): Promise<{ ok: boolean; }>;"
    );
    let free = ToolDeclaration {
        name: "free",
        ..ToolDeclaration::default()
    };
    assert_eq!(
        render_tool_signature(&free, None),
        "free(args: unknown): Promise<unknown>;"
    );
    let mcp = json!({
        "type": "object",
        "properties": { "content": { "type": "array", "items": { "type": "object" } }, "isError": { "type": "boolean" }, "_meta": { "type": "object" } },
        "required": ["content"]
    });
    let empty = json!({ "type": "object", "properties": {} });
    let plain = ToolDeclaration {
        name: "plain",
        input_schema: Some(&empty),
        output_schema: Some(&mcp),
        ..ToolDeclaration::default()
    };
    assert_eq!(
        render_tool_signature(&plain, None),
        "plain(args: { [key: string]: unknown; }): Promise<{ _meta?: { [key: string]: unknown; }; content: Array<{ [key: string]: unknown; }>; isError?: boolean; }>;"
    );
    let string = json!({ "type": "string" });
    let foo = ToolDeclaration {
        name: "foo",
        description: "bar",
        input_schema: Some(&string),
        output_schema: None,
    };
    assert_eq!(
        render_tool_sample(&foo, None),
        "bar\n\ncodemode tool declaration:\n```ts\ndeclare const tools: { foo(args: string): Promise<unknown>; };\n```"
    );

    let read_input = json!({ "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] });
    let tools = [
        ToolDeclaration {
            name: "read",
            description: "Read a file.\nSecond line.",
            input_schema: Some(&read_input),
            output_schema: Some(&string),
        },
        ToolDeclaration {
            name: "remote-api",
            ..ToolDeclaration::default()
        },
    ];
    let globals = [GlobalDeclaration {
        name: "attach",
        description: "Attach it.",
        input_schema: Some(&string),
        ..GlobalDeclaration::default()
    }];
    assert_eq!(
        render_declarations(&tools, &globals),
        [
            "declare const tools: {",
            "  /**",
            "   * Read a file.",
            "   * Second line.",
            "   */",
            "  read(args: { path: string; }): Promise<string>;",
            "  remote_api(args: unknown): Promise<unknown>;",
            "};",
            "",
            "/** Attach it. */",
            "declare function attach(args: string): Promise<unknown>;",
        ]
        .join("\n")
    );
    let globals = [
        GlobalDeclaration {
            name: "models.list",
            description: "List models.",
            signature: Some("(type: string): Promise<string[]>"),
            ..GlobalDeclaration::default()
        },
        GlobalDeclaration {
            name: "models.get",
            input_schema: Some(&string),
            ..GlobalDeclaration::default()
        },
        GlobalDeclaration {
            name: "plain",
            signature: Some("(): void"),
            ..GlobalDeclaration::default()
        },
    ];
    assert_eq!(
        render_declarations(&[], &globals),
        [
            "declare function plain(): void;",
            "",
            "declare const models: {",
            "  /** List models. */",
            "  list(type: string): Promise<string[]>;",
            "  get(args: string): Promise<unknown>;",
            "};",
        ]
        .join("\n")
    );
    let x = [ToolDeclaration {
        name: "x",
        description: "a */ b",
        ..ToolDeclaration::default()
    }];
    assert!(render_declarations(&x, &[]).contains("/** a *\\/ b */"));
}

#[test]
fn sources_parse_with_their_options_line() {
    let plain = parse_codemode_source("text('hi')").unwrap();
    assert_eq!(
        (plain.code.as_str(), plain.options),
        ("text('hi')", SourceOptions::default())
    );
    assert_eq!(
        parse_codemode_source("// just a comment\nreturn 1").unwrap().code,
        "// just a comment\nreturn 1"
    );
    let parsed = parse_codemode_source("// @options: {\"timeout_ms\": 10}\nconst a = 1;\ntext(a)").unwrap();
    assert_eq!(parsed.code, "\nconst a = 1;\ntext(a)");
    assert_eq!(parsed.options.timeout_ms, Some(10));
    let options = parse_codemode_source("  // @options:{\"max_output_tokens\":0,\"timeout_ms\":1500}\r\ntext(1)")
        .unwrap()
        .options;
    assert_eq!((options.max_output_tokens, options.timeout_ms), (Some(0), Some(1500)));
    let empty = parse_codemode_source("// @options: {}\ntext(1)").unwrap();
    assert_eq!(
        (empty.code.as_str(), empty.options),
        ("\ntext(1)", SourceOptions::default())
    );
    assert_eq!(
        parse_codemode_source("// @optionsx {}\ntext(1)").unwrap().options,
        SourceOptions::default()
    );
    assert_eq!(
        parse_codemode_source("text(\"// @options: {}\")").unwrap().options,
        SourceOptions::default()
    );
}

#[test]
fn misplaced_options_lines_and_invalid_sources_are_refused() {
    for (input, line) in [
        (
            "// count things\n// @options: {\"timeout_ms\": 1000}\nwhile (true) {}",
            2,
        ),
        ("\n  // @options: {\"timeout_ms\": 1000}\nwhile (true) {}", 2),
        ("text(1)\r\n// @options: {\"timeout_ms\": 1}", 2),
        (
            "// @options: {\"timeout_ms\": 1}\ntext(1)\n// @options: {\"timeout_ms\": 2}",
            3,
        ),
    ] {
        let error = parse_codemode_source(input).unwrap_err();
        assert_eq!(
            error.0,
            format!("The // @options: line must be the first line of the input; found one on line {line}")
        );
    }
    for (input, message) in [
        ("", "Expected JavaScript source text (non-empty)"),
        ("  \n", "Expected JavaScript source text (non-empty)"),
        (
            "// @options:\ntext(1)",
            "@options must be a JSON object with supported fields",
        ),
        (
            "// @options: {timeout_ms: 1}\ntext(1)",
            "@options must be valid JSON with supported fields",
        ),
        (
            "// @options: [1]\ntext(1)",
            "@options must be a JSON object with supported fields",
        ),
        (
            "// @options: {\"yield\": 1}\ntext(1)",
            "@options only supports `max_output_tokens` and `timeout_ms`; got `yield`",
        ),
        (
            "// @options: {\"max_output_tokens\": 1.5}\ntext(1)",
            "@options field `max_output_tokens` must be a non-negative safe integer",
        ),
        (
            "// @options: {\"timeout_ms\": 0}\ntext(1)",
            "@options field `timeout_ms` must be a positive integer",
        ),
        (
            "// @options: {\"timeout_ms\": 1}",
            "The @options line must be followed by JavaScript source on subsequent lines",
        ),
        (
            "// @options: {\"timeout_ms\": 1}\n  \n",
            "The @options line must be followed by JavaScript source on subsequent lines",
        ),
    ] {
        let error = parse_codemode_source(input).unwrap_err();
        assert!(error.0.contains(message), "{input:?}: {}", error.0);
    }
}
