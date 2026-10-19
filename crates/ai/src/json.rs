//! JSON helpers for tool-call arguments: repair of malformed string literals, and
//! best-effort parsing of incomplete JSON while arguments stream in.

use serde_json::{Map, Number, Value};
use std::fmt::Write as _;

use crate::types::JsonObject;

/// Repair malformed JSON string literals: escape raw control characters inside strings
/// and double backslashes before invalid escape characters.
pub fn repair_json(json: &str) -> String {
    let mut repaired = String::with_capacity(json.len());
    let chars: Vec<char> = json.chars().collect();
    let mut in_string = false;
    let mut index = 0;
    while index < chars.len() {
        let ch = chars[index];
        if !in_string {
            repaired.push(ch);
            if ch == '"' {
                in_string = true;
            }
            index += 1;
            continue;
        }
        match ch {
            '"' => {
                repaired.push(ch);
                in_string = false;
            }
            '\\' => match chars.get(index + 1) {
                Some('u')
                    if chars.len() >= index + 6 && chars[index + 2..index + 6].iter().all(char::is_ascii_hexdigit) =>
                {
                    repaired.extend(&chars[index..index + 6]);
                    index += 5;
                }
                Some(next @ ('"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't')) => {
                    repaired.push('\\');
                    repaired.push(*next);
                    index += 1;
                }
                _ => repaired.push_str("\\\\"),
            },
            '\u{08}' => repaired.push_str("\\b"),
            '\u{0c}' => repaired.push_str("\\f"),
            '\n' => repaired.push_str("\\n"),
            '\r' => repaired.push_str("\\r"),
            '\t' => repaired.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(repaired, "\\u{:04x}", c as u32);
            }
            c => repaired.push(c),
        }
        index += 1;
    }
    repaired
}

/// Parse JSON, retrying once with [`repair_json`] when the first attempt fails.
pub fn parse_json_with_repair(json: &str) -> Result<Value, serde_json::Error> {
    match serde_json::from_str(json) {
        Ok(value) => Ok(value),
        Err(error) => {
            let repaired = repair_json(json);
            if repaired == json {
                return Err(error);
            }
            serde_json::from_str(&repaired)
        }
    }
}

/// Parse the model's complete arguments text of a finished tool call.
///
/// Text that is not a JSON object yields empty arguments plus an error message that
/// includes the received text, so tool validation can report it back to the model.
pub fn parse_tool_call_arguments(json: &str) -> (JsonObject, Option<String>) {
    if json.trim().is_empty() {
        return (JsonObject::new(), None);
    }
    match parse_json_with_repair(json) {
        Ok(Value::Object(object)) => (object, None),
        Ok(_) => (
            JsonObject::new(),
            Some(format!(
                "Tool arguments must be a JSON object\n\nReceived arguments:\n{json}"
            )),
        ),
        Err(error) => (
            JsonObject::new(),
            Some(format!("{error}\n\nReceived arguments:\n{json}")),
        ),
    }
}

/// Best-effort parse of potentially incomplete JSON while it streams. Always returns an
/// object; fields may be missing, strings truncated and arrays incomplete.
pub fn parse_streaming_json(partial: &str) -> JsonObject {
    if partial.trim().is_empty() {
        return JsonObject::new();
    }
    let value = parse_json_with_repair(partial)
        .ok()
        .or_else(|| parse_partial_json(partial))
        .or_else(|| parse_partial_json(&repair_json(partial)));
    match value {
        Some(Value::Object(object)) => object,
        _ => JsonObject::new(),
    }
}

/// Parse a JSON prefix, closing whatever is still open at the end of the input. Returns
/// `None` for text that is invalid rather than merely truncated.
pub fn parse_partial_json(text: &str) -> Option<Value> {
    let mut parser = PartialParser {
        chars: text.chars().collect(),
        pos: 0,
    };
    parser.skip_ws();
    let value = parser.value()?;
    parser.skip_ws();
    (parser.pos == parser.chars.len()).then_some(value?)
}

struct PartialParser {
    chars: Vec<char>,
    pos: usize,
}

/// `Some(None)`: input ended before a value started. `None`: invalid input.
type Parsed = Option<Option<Value>>;

impl PartialParser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn eof(&self) -> bool {
        self.pos >= self.chars.len()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\n' | '\r' | '\t')) {
            self.pos += 1;
        }
    }

    fn value(&mut self) -> Parsed {
        match self.peek() {
            None => Some(None),
            Some('{') => self.object(),
            Some('[') => self.array(),
            Some('"') => self.string().map(|s| s.map(Value::String)),
            Some('t') => self.literal("true", Value::Bool(true)),
            Some('f') => self.literal("false", Value::Bool(false)),
            Some('n') => self.literal("null", Value::Null),
            Some('-' | '0'..='9') => self.number(),
            Some(_) => None,
        }
    }

    fn literal(&mut self, word: &str, value: Value) -> Parsed {
        for expected in word.chars() {
            match self.peek() {
                None => return Some(Some(value)),
                Some(c) if c == expected => self.pos += 1,
                Some(_) => return None,
            }
        }
        Some(Some(value))
    }

    fn number(&mut self) -> Parsed {
        let start = self.pos;
        while matches!(self.peek(), Some('-' | '+' | '.' | 'e' | 'E' | '0'..='9')) {
            self.pos += 1;
        }
        let mut text: String = self.chars[start..self.pos].iter().collect();
        if let Ok(value) = serde_json::from_str::<Number>(&text) {
            return Some(Some(Value::Number(value)));
        }
        if !self.eof() {
            return None;
        }
        // A truncated number: drop trailing characters until it parses.
        while !text.is_empty() {
            text.pop();
            if let Ok(value) = serde_json::from_str::<Number>(&text) {
                return Some(Some(Value::Number(value)));
            }
        }
        Some(None)
    }

    #[allow(clippy::option_option)]
    /// A string; at end of input, the text so far (an incomplete escape is dropped).
    fn string(&mut self) -> Option<Option<String>> {
        self.pos += 1;
        let mut out = String::new();
        loop {
            let Some(c) = self.peek() else { return Some(Some(out)) };
            self.pos += 1;
            match c {
                '"' => return Some(Some(out)),
                '\\' => {
                    let Some(escaped) = self.peek() else {
                        return Some(Some(out));
                    };
                    self.pos += 1;
                    match escaped {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'b' => out.push('\u{08}'),
                        'f' => out.push('\u{0c}'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => {
                            let Some(code) = self.hex4() else {
                                if self.eof() {
                                    return Some(Some(out));
                                }
                                return None;
                            };
                            if (0xD800..0xDC00).contains(&code) {
                                // High surrogate: needs a following low surrogate.
                                if self.chars.get(self.pos) == Some(&'\\') && self.chars.get(self.pos + 1) == Some(&'u')
                                {
                                    self.pos += 2;
                                    match self.hex4() {
                                        Some(low) if (0xDC00..0xE000).contains(&low) => {
                                            let combined = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                                            out.push(char::from_u32(combined).unwrap_or('\u{fffd}'));
                                        }
                                        Some(_) => out.push('\u{fffd}'),
                                        None if self.eof() => return Some(Some(out)),
                                        None => return None,
                                    }
                                } else if self.eof() {
                                    return Some(Some(out));
                                } else {
                                    out.push('\u{fffd}');
                                }
                            } else {
                                out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                            }
                        }
                        _ => return None,
                    }
                }
                c => out.push(c),
            }
        }
    }

    fn hex4(&mut self) -> Option<u32> {
        if self.pos + 4 > self.chars.len() {
            self.pos = self.chars.len();
            return None;
        }
        let text: String = self.chars[self.pos..self.pos + 4].iter().collect();
        let code = u32::from_str_radix(&text, 16).ok()?;
        self.pos += 4;
        Some(code)
    }

    fn array(&mut self) -> Parsed {
        self.pos += 1;
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                None => return Some(Some(Value::Array(items))),
                Some(']') => {
                    self.pos += 1;
                    return Some(Some(Value::Array(items)));
                }
                _ => {}
            }
            match self.value()? {
                Some(value) => items.push(value),
                None => return Some(Some(Value::Array(items))),
            }
            self.skip_ws();
            match self.peek() {
                None => return Some(Some(Value::Array(items))),
                Some(',') => self.pos += 1,
                Some(']') => {
                    self.pos += 1;
                    return Some(Some(Value::Array(items)));
                }
                Some(_) => return None,
            }
        }
    }

    fn object(&mut self) -> Parsed {
        self.pos += 1;
        let mut map = Map::new();
        loop {
            self.skip_ws();
            match self.peek() {
                None => return Some(Some(Value::Object(map))),
                Some('}') => {
                    self.pos += 1;
                    return Some(Some(Value::Object(map)));
                }
                Some('"') => {}
                Some(_) => return None,
            }
            let key = self.string()?;
            let Some(key) = key else {
                return Some(Some(Value::Object(map)));
            };
            self.skip_ws();
            match self.peek() {
                None => return Some(Some(Value::Object(map))),
                Some(':') => self.pos += 1,
                Some(_) => return None,
            }
            self.skip_ws();
            match self.value()? {
                Some(value) => {
                    map.insert(key, value);
                }
                None => return Some(Some(Value::Object(map))),
            }
            self.skip_ws();
            match self.peek() {
                None => return Some(Some(Value::Object(map))),
                Some(',') => self.pos += 1,
                Some('}') => {
                    self.pos += 1;
                    return Some(Some(Value::Object(map)));
                }
                Some(_) => return None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn partial_objects() {
        assert_eq!(
            Value::Object(parse_streaming_json(r#"{"path": "src/ma"#)),
            json!({"path": "src/ma"})
        );
        assert_eq!(Value::Object(parse_streaming_json(r#"{"a": 1, "b"#)), json!({"a": 1}));
        assert_eq!(
            Value::Object(parse_streaming_json(r#"{"a": [1, 2, {"c": tr"#)),
            json!({"a": [1, 2, {"c": true}]})
        );
        assert_eq!(Value::Object(parse_streaming_json(r#"{"a": -"#)), json!({}));
        assert_eq!(Value::Object(parse_streaming_json(r#"{"a": 1.5e"#)), json!({"a": 1.5}));
        assert_eq!(Value::Object(parse_streaming_json(r#"{"a": "x\"#)), json!({"a": "x"}));
        assert_eq!(
            Value::Object(parse_streaming_json(r#"{"a": "😀"}"#)),
            json!({"a": "😀"})
        );
        assert_eq!(Value::Object(parse_streaming_json("[1,2]")), json!({}));
        assert_eq!(Value::Object(parse_streaming_json("{\"a\" x")), json!({}));
    }

    #[test]
    fn repairs_control_characters_and_bad_escapes() {
        assert_eq!(repair_json("{\"a\": \"x\ny\\q\"}"), "{\"a\": \"x\\ny\\\\q\"}");
        let (args, error) = parse_tool_call_arguments("{\"a\": \"x\ny\"}");
        assert_eq!(Value::Object(args), json!({"a": "x\ny"}));
        assert!(error.is_none());
        let (args, error) = parse_tool_call_arguments("[1]");
        assert!(args.is_empty());
        assert!(error.unwrap().contains("Received arguments:\n[1]"));
    }
}
