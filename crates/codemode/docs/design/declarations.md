# Declarations and source (`delta-codemode`)

Model-facing text for code mode: TypeScript declarations of the script API, and the format of the
code mode tool's argument. Ports of upstream `declarations.ts`, `source.ts` and `identifier.ts`.

## Declarations

`render_declarations(tools, globals)` renders tools as members of `declare const tools`, globals
as `declare function` statements, and `ns.member` globals as members of `declare const ns`.
Descriptions become doc comments; JSON Schemas become types (`schema_to_type`):

- objects with property comments from `description` and constraint tags such as
  `// @integer @minimum 1 @default 20` (tags are dropped first when a type is too long; tags
  longer than 80 characters are left out), unions from `anyOf`/`oneOf`/`type` arrays, `enum` and
  `const` as literal types, arrays and tuples;
- local `$ref`s (`#/$defs/…`, `#/definitions/…`) expanded at most 32 times per schema;
  recursive and remote references render as `unknown`;
- an input type longer than `DEFAULT_INPUT_SCHEMA_MAX_CHARS` (16,000) renders as `unknown`.

`render_tool_signature`, `render_tool_output_type` and `render_tool_sample` render one tool for
the harness's prompt section and `describeTool()`. Schemas only shape the text; values are not
validated against them.

## Source format

`parse_codemode_source(input)` accepts JavaScript whose first line may be
`// @options: {"max_output_tokens": N, "timeout_ms": N}`. The options line is replaced by an
empty line so line numbers in stack traces still match. Empty input, invalid JSON, unknown
fields, out-of-range values, an options line without code, or an options line below the first
line are `SourceError`s with upstream's messages. `CODEMODE_SOURCE_GRAMMAR` is upstream's Lark
grammar, unchanged, for providers with grammar-constrained tool input.

## Differences from upstream

None in behavior; the functions take Rust types (`ToolDeclaration`, `GlobalDeclaration`) instead
of `CodemodeTool` objects.
