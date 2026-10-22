//! The code mode source format: JavaScript, optionally preceded by one options line.
//!
//! ```js
//! // @options: {"max_output_tokens": 2000, "timeout_ms": 30000}
//! const text = await tools.read({ path: "package.json" });
//! text(JSON.parse(text).name);
//! ```

use serde_json::Value;

pub const CODEMODE_OPTIONS_PREFIX: &str = "// @options:";

const SUPPORTED_FIELDS: [&str; 2] = ["max_output_tokens", "timeout_ms"];
const SUPPORTED_FIELDS_TEXT: &str = "`max_output_tokens` and `timeout_ms`";
/// Largest delay `setTimeout` supports, which bounds `timeout_ms` (as upstream).
const MAX_TIMEOUT_MS: u64 = 2_147_483_647;
const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

/// Lark grammar for providers with grammar-constrained tool input. It only fixes the shape of
/// the options line; the options JSON and the code are checked by [`parse_codemode_source`].
pub const CODEMODE_SOURCE_GRAMMAR: &str = r"
start: options_source | plain_source
options_source: OPTIONS_LINE NEWLINE SOURCE
plain_source: SOURCE

OPTIONS_LINE: /[ \t]*\/\/ @options:[^\r\n]*/
NEWLINE: /\r?\n/
SOURCE: /[\s\S]+/
";

/// The options line's fields.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SourceOptions {
    /// Token budget for the script's output.
    pub max_output_tokens: Option<u64>,
    /// Hard deadline for the whole script in milliseconds, including tool calls.
    pub timeout_ms: Option<u64>,
}

/// A parsed source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedSource {
    /// The script with the options line replaced by an empty line, so line numbers are
    /// unchanged.
    pub code: String,
    pub options: SourceOptions,
}

/// Why a source does not parse.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct SourceError(pub String);

fn safe_integer(value: &Value) -> Option<u64> {
    let number = value.as_u64().or_else(|| {
        value
            .as_f64()
            .filter(|number| number.fract() == 0.0 && *number >= 0.0)
            .map(|number| number as u64)
    })?;
    (number <= MAX_SAFE_INTEGER).then_some(number)
}

fn parse_options(directive: &str) -> Result<SourceOptions, SourceError> {
    let not_object = || {
        SourceError(format!(
            "@options must be a JSON object with supported fields {SUPPORTED_FIELDS_TEXT}"
        ))
    };
    if directive.is_empty() {
        return Err(not_object());
    }
    let value: Value = serde_json::from_str(directive).map_err(|error| {
        SourceError(format!(
            "@options must be valid JSON with supported fields {SUPPORTED_FIELDS_TEXT}: {error}"
        ))
    })?;
    let Value::Object(fields) = value else {
        return Err(not_object());
    };
    if let Some(key) = fields.keys().find(|key| !SUPPORTED_FIELDS.contains(&key.as_str())) {
        return Err(SourceError(format!(
            "@options only supports {SUPPORTED_FIELDS_TEXT}; got `{key}`"
        )));
    }
    let mut options = SourceOptions::default();
    if let Some(value) = fields.get("max_output_tokens") {
        let tokens = safe_integer(value).ok_or_else(|| {
            SourceError("@options field `max_output_tokens` must be a non-negative safe integer".to_owned())
        })?;
        options.max_output_tokens = Some(tokens);
    }
    if let Some(value) = fields.get("timeout_ms") {
        let timeout = safe_integer(value)
            .filter(|timeout| (1..=MAX_TIMEOUT_MS).contains(timeout))
            .ok_or_else(|| {
                SourceError(format!(
                    "@options field `timeout_ms` must be a positive integer up to {MAX_TIMEOUT_MS}"
                ))
            })?;
        options.timeout_ms = Some(timeout);
    }
    Ok(options)
}

/// Splits an optional first-line `// @options: {...}` from the script. Empty input, invalid
/// options, and an options line that is not the first line (which would otherwise be ignored
/// without notice) are errors.
pub fn parse_codemode_source(input: &str) -> Result<ParsedSource, SourceError> {
    if input.trim().is_empty() {
        return Err(SourceError(
            "Expected JavaScript source text (non-empty). Provide JS only, optionally with a first line `// @options: {\"max_output_tokens\": 1000}`.".to_owned(),
        ));
    }
    if let Some(line) = input
        .split('\n')
        .enumerate()
        .skip(1)
        .find(|(_, line)| line.trim_start().starts_with(CODEMODE_OPTIONS_PREFIX))
        .map(|(index, _)| index + 1)
    {
        return Err(SourceError(format!(
            "The {CODEMODE_OPTIONS_PREFIX} line must be the first line of the input; found one on line {line}"
        )));
    }
    let first = input.split('\n').next().unwrap_or_default();
    let first = first.strip_suffix('\r').unwrap_or(first);
    let trimmed = first.trim_start();
    let Some(directive) = trimmed.strip_prefix(CODEMODE_OPTIONS_PREFIX) else {
        return Ok(ParsedSource {
            code: input.to_owned(),
            options: SourceOptions::default(),
        });
    };
    let code = input.find('\n').map_or("", |newline| &input[newline..]);
    if code.trim().is_empty() {
        return Err(SourceError(
            "The @options line must be followed by JavaScript source on subsequent lines".to_owned(),
        ));
    }
    Ok(ParsedSource {
        code: code.to_owned(),
        options: parse_options(directive.trim())?,
    })
}
