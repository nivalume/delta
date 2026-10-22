//! Templates: Jinja through minijinja, configured as upstream configures jinja2
//! (`StrictUndefined`, no trailing newline kept), with a `tojson` that prints what jinja2's
//! prints, so a rendered observation is the same text upstream sends.

use std::fmt::Write as _;

use minijinja::value::{Kwargs, Serde};
use minijinja::{Environment, Error, ErrorKind, UndefinedBehavior, Value};
use serde_json::{Map, Value as Json};

/// Renders `template` with `vars`, which must be a JSON object. An undefined variable is an
/// error, as with jinja2's `StrictUndefined`.
pub fn render(template: &str, vars: &Json) -> Result<String, Error> {
    environment().render_str(template, Value::from(Serde(vars)))
}

fn environment() -> Environment<'static> {
    let mut environment = Environment::new();
    environment.set_undefined_behavior(UndefinedBehavior::Strict);
    environment.add_filter("tojson", tojson);
    environment
}

/// jinja2's `tojson`: `json.dumps` with sorted keys and ASCII escapes, then `<`, `>`, `&` and
/// `'` escaped for HTML.
// minijinja passes keyword arguments by value.
#[allow(clippy::needless_pass_by_value)]
fn tojson(value: &Value, kwargs: Kwargs) -> Result<Value, Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    kwargs.assert_all_used()?;
    let json =
        serde_json::to_value(value).map_err(|error| Error::new(ErrorKind::BadSerialization, error.to_string()))?;
    let mut out = String::new();
    python_dumps(&json, indent, 0, &mut out);
    let escaped = out
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026")
        .replace('\'', "\\u0027");
    Ok(Value::from_safe_string(escaped))
}

/// `value` as Python's `json.dumps(value, sort_keys=True, indent=indent)` prints it.
pub fn python_dumps(value: &Json, indent: Option<usize>, depth: usize, out: &mut String) {
    let newline = |out: &mut String, depth: usize| {
        if let Some(indent) = indent {
            out.push('\n');
            out.push_str(&" ".repeat(indent * depth));
        }
    };
    let item_separator = if indent.is_some() { "," } else { ", " };
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
        Json::Number(number) => out.push_str(&python_number(number)),
        Json::String(text) => python_string(text, out),
        Json::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(item_separator);
                }
                newline(out, depth + 1);
                python_dumps(item, indent, depth + 1, out);
            }
            newline(out, depth);
            out.push(']');
        }
        Json::Object(object) => {
            if object.is_empty() {
                out.push_str("{}");
                return;
            }
            let mut keys: Vec<&String> = object.keys().collect();
            keys.sort();
            out.push('{');
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    out.push_str(item_separator);
                }
                newline(out, depth + 1);
                python_string(key, out);
                out.push_str(": ");
                python_dumps(&object[key], indent, depth + 1, out);
            }
            newline(out, depth);
            out.push('}');
        }
    }
}

/// A number as Python's `repr` prints it: integers as they are, floats with at least one
/// decimal and Python's exponent form.
fn python_number(number: &serde_json::Number) -> String {
    if number.is_i64() || number.is_u64() {
        return number.to_string();
    }
    let value = number.as_f64().unwrap_or_default();
    python_float(value)
}

/// A float as Python's `repr(float)` prints it.
pub fn python_float(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity" } else { "-Infinity" }.to_owned();
    }
    // Python switches to exponent form outside 1e-4 <= |x| < 1e16.
    let magnitude = value.abs();
    if magnitude != 0.0 && !(1e-4..1e16).contains(&magnitude) {
        let formatted = format!("{value:e}");
        let (mantissa, exponent) = formatted.split_once('e').unwrap_or((&formatted, "0"));
        let exponent: i32 = exponent.parse().unwrap_or_default();
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!("{mantissa}e{sign}{:02}", exponent.abs());
    }
    let formatted = format!("{value:?}");
    if formatted.contains('.') || formatted.contains('e') {
        formatted
    } else {
        format!("{formatted}.0")
    }
}

/// A string as Python's `json.dumps` prints it with `ensure_ascii=True`.
fn python_string(text: &str, out: &mut String) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(character),
            _ => {
                let mut units = [0_u16; 2];
                for unit in character.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}

/// Merges `layers` into one object, later layers winning (upstream's `recursive_merge`).
pub fn vars(layers: impl IntoIterator<Item = Json>) -> Json {
    crate::config::merge_all(layers.into_iter().map(|layer| match layer {
        Json::Object(_) => layer,
        _ => Json::Object(Map::new()),
    }))
}

/// What Python's `platform.uname()` reports, as template variables.
pub fn uname() -> Json {
    static UNAME: std::sync::OnceLock<Json> = std::sync::OnceLock::new();
    UNAME.get_or_init(read_uname).clone()
}

fn read_uname() -> Json {
    let system = match std::env::consts::OS {
        "linux" => "Linux".to_owned(),
        "macos" => "Darwin".to_owned(),
        "windows" => "Windows".to_owned(),
        other => other.to_owned(),
    };
    let field = |flag: &str| -> Option<String> {
        if cfg!(windows) {
            return None;
        }
        let output = std::process::Command::new("uname").arg(flag).output().ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    let machine = field("-m").unwrap_or_else(|| match std::env::consts::ARCH {
        "x86_64" if cfg!(windows) => "AMD64".to_owned(),
        arch => arch.to_owned(),
    });
    let processor = field("-p")
        .filter(|processor| processor != "unknown")
        .unwrap_or_default();
    let node = field("-n")
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .unwrap_or_default();
    serde_json::json!({
        "system": system,
        "node": node,
        "release": field("-r").unwrap_or_default(),
        "version": field("-v").unwrap_or_default(),
        "machine": machine,
        "processor": processor,
    })
}

/// The process environment as template variables (upstream's local environment adds
/// `os.environ`).
pub fn environ() -> Json {
    Json::Object(
        std::env::vars()
            .map(|(key, value)| (key, Json::String(value)))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_undefined_and_slices() {
        let vars = serde_json::json!({ "output": { "output": "abcdef", "returncode": 0 } });
        assert_eq!(
            render(
                "{{ output.output[:2] }}|{{ output.output[-2:] }}|{{ output.output | length - 4 }}",
                &vars
            )
            .unwrap(),
            "ab|ef|2"
        );
        assert!(render("{{ missing }}", &vars).is_err());
        assert_eq!(
            render("{% if missing is defined %}x{% else %}y{% endif %}", &vars).unwrap(),
            "y"
        );
        assert_eq!(render("a\n", &vars).unwrap(), "a");
    }

    #[test]
    fn tojson_prints_what_jinja2_prints() {
        let vars = serde_json::json!({ "text": "<a href='x'>&é😀\n\"", "object": { "b": [1, 2.5], "a": null } });
        let escape = |code: &str| format!("{}u{code}", '\\');
        let expected = format!(
            "\"{}a href={}x{}{}{}{}{}{}\\n\\\"\"",
            escape("003c"),
            escape("0027"),
            escape("0027"),
            escape("003e"),
            escape("0026"),
            escape("00e9"),
            escape("d83d"),
            escape("de00"),
        );
        assert_eq!(render("{{ text | tojson }}", &vars).unwrap(), expected);
        assert_eq!(
            render("{{ object | tojson }}", &vars).unwrap(),
            r#"{"a": null, "b": [1, 2.5]}"#
        );
        assert_eq!(
            render("{{ object | tojson(indent=2) }}", &vars).unwrap(),
            "{\n  \"a\": null,\n  \"b\": [\n    1,\n    2.5\n  ]\n}"
        );
    }

    #[test]
    fn floats_print_as_python_repr() {
        assert_eq!(python_float(0.0), "0.0");
        assert_eq!(python_float(3.0), "3.0");
        assert_eq!(python_float(0.1), "0.1");
        assert_eq!(python_float(1e-5), "1e-05");
        assert_eq!(python_float(1.5e20), "1.5e+20");
    }
}
