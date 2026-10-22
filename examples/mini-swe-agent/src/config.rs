//! Configs: YAML files merged in order, with `key.path=value` overrides, found the way
//! upstream finds them (`minisweagent/config/__init__.py`). The configs upstream ships are
//! built in, so `-c mini.yaml` or `-c swebench.yaml` works from anywhere.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

/// The configs upstream ships, by their path under its `config/` directory.
pub const BUILTIN_CONFIGS: [(&str, &str); 6] = [
    ("mini.yaml", include_str!("../config/mini.yaml")),
    ("default.yaml", include_str!("../config/default.yaml")),
    ("mini_textbased.yaml", include_str!("../config/mini_textbased.yaml")),
    (
        "benchmarks/swebench.yaml",
        include_str!("../config/benchmarks/swebench.yaml"),
    ),
    (
        "benchmarks/swebench_backticks.yaml",
        include_str!("../config/benchmarks/swebench_backticks.yaml"),
    ),
    (
        "benchmarks/swebench_xml.yaml",
        include_str!("../config/benchmarks/swebench_xml.yaml"),
    ),
];

/// Why a config could not be built.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Could not find config file for {spec} (tried: {tried})")]
    NotFound { spec: String, tried: String },
    #[error("Invalid config spec {0:?}: empty config key")]
    EmptyKey(String),
    #[error("Could not read {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("Invalid YAML in {path}: {source}")]
    Yaml {
        path: String,
        #[source]
        source: serde_saphyr::Error,
    },
    #[error("Invalid {section} config: {source}")]
    Invalid {
        section: &'static str,
        #[source]
        source: serde_json::Error,
    },
}

/// Where a config spec resolved to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigSource {
    File(PathBuf),
    Builtin(&'static str),
}

/// The text of a built-in config: by its path (`benchmarks/swebench.yaml`) or its file name
/// alone (`swebench.yaml`), with or without `.yaml`.
pub fn builtin_config(name: &str) -> Option<(&'static str, &'static str)> {
    let name = if Path::new(name).extension().is_some_and(|extension| extension == "yaml") {
        name.to_owned()
    } else {
        format!("{name}.yaml")
    };
    BUILTIN_CONFIGS
        .iter()
        .find(|(path, _)| {
            *path == name
                || ["extra/", "benchmarks/"]
                    .iter()
                    .any(|dir| *path == format!("{dir}{name}"))
        })
        .map(|(path, text)| (*path, *text))
}

/// Resolves a config file spec as upstream's `get_config_path`: the path itself, then
/// `$MSWEA_CONFIG_DIR`, then the built-in configs (also under `extra/` and `benchmarks/`).
/// A spec without the `.yaml` suffix gets it.
pub fn config_source(spec: &str) -> Result<ConfigSource, ConfigError> {
    let mut path = PathBuf::from(spec);
    if path.extension().is_none_or(|extension| extension != "yaml") {
        path.set_extension("yaml");
    }
    let config_dir = std::env::var_os("MSWEA_CONFIG_DIR").map_or_else(|| PathBuf::from("."), PathBuf::from);
    let candidates = [path.clone(), config_dir.join(&path)];
    if let Some(found) = candidates.iter().find(|candidate| candidate.is_file()) {
        return Ok(ConfigSource::File(found.clone()));
    }
    let name = path.to_string_lossy();
    if let Some((builtin, _)) = builtin_config(&name) {
        return Ok(ConfigSource::Builtin(builtin));
    }
    let tried: Vec<String> = candidates
        .iter()
        .map(|candidate| candidate.display().to_string())
        .chain(["builtin", "builtin/extra", "builtin/benchmarks"].map(|dir| format!("{dir}/{name}")))
        .collect();
    Err(ConfigError::NotFound {
        spec: spec.to_owned(),
        tried: tried.join(", "),
    })
}

/// Parses YAML text into JSON values. Line breaks read as `\n` whatever the file uses (a
/// Windows checkout turns the built-in configs to CRLF), as the YAML spec normalizes them.
pub fn parse_yaml(text: &str, path: &str) -> Result<Value, ConfigError> {
    let text = &text.replace("\r\n", "\n");
    let value: Option<Value> = serde_saphyr::from_str(text).map_err(|source| ConfigError::Yaml {
        path: path.to_owned(),
        source,
    })?;
    let mut value = value.unwrap_or_else(|| Value::Object(Map::new()));
    drop_invented_line_break(text, &mut value);
    Ok(value)
}

/// A block scalar (`|`, `>`) that ends the file without a line break has none to keep, as
/// PyYAML reads it; saphyr adds one. Upstream's `default.yaml` ends that way.
fn drop_invented_line_break(text: &str, value: &mut Value) {
    if text.ends_with('\n') {
        return;
    }
    let last_style = saphyr_parser::Parser::new_from_str(text)
        .filter_map(Result::ok)
        .filter_map(|(event, _)| match event {
            saphyr_parser::Event::Scalar(_, style, ..) => Some(style),
            _ => None,
        })
        .last();
    if !matches!(
        last_style,
        Some(saphyr_parser::ScalarStyle::Literal | saphyr_parser::ScalarStyle::Folded)
    ) {
        return;
    }
    if let Some(Value::String(scalar)) = last_leaf(value)
        && scalar.ends_with('\n')
    {
        scalar.pop();
    }
}

/// The last scalar of `value` in document order.
fn last_leaf(value: &mut Value) -> Option<&mut Value> {
    match value {
        Value::Object(object) => object.values_mut().last().and_then(last_leaf),
        Value::Array(items) => items.last_mut().and_then(last_leaf),
        leaf => Some(leaf),
    }
}

/// A config from a spec: `key.path=value` (the value parsed as JSON when it is JSON, else
/// kept as a string), or a config file.
pub fn config_from_spec(spec: &str) -> Result<Value, ConfigError> {
    if let Some((key, value)) = spec.split_once('=') {
        let keys: Vec<&str> = key.split('.').collect();
        if keys.iter().any(|key| key.is_empty()) {
            return Err(ConfigError::EmptyKey(spec.to_owned()));
        }
        let value = serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_owned()));
        let mut config = Value::Object(Map::new());
        set_path(&mut config, &keys, value);
        return Ok(config);
    }
    match config_source(spec)? {
        ConfigSource::Builtin(path) => {
            let text = builtin_config(path).map(|(_, text)| text).unwrap_or_default();
            parse_yaml(text, path)
        }
        ConfigSource::File(path) => {
            let text = std::fs::read_to_string(&path).map_err(|source| ConfigError::Read {
                path: path.display().to_string(),
                source,
            })?;
            parse_yaml(&text, &path.display().to_string())
        }
    }
}

/// Sets `value` at `keys` in `config`, creating the objects on the way.
pub fn set_path(config: &mut Value, keys: &[&str], value: Value) {
    let Some((last, parents)) = keys.split_last() else {
        *config = value;
        return;
    };
    let mut current = config;
    for key in parents {
        current = object_mut(current)
            .entry((*key).to_owned())
            .or_insert_with(|| Value::Object(Map::new()));
    }
    object_mut(current).insert((*last).to_owned(), value);
}

/// `value`'s object, after replacing anything else with an empty one.
fn object_mut(value: &mut Value) -> &mut Map<String, Value> {
    if !value.is_object() {
        *value = Value::Object(Map::new());
    }
    match value {
        Value::Object(object) => object,
        _ => unreachable!("replaced with an object above"),
    }
}

/// Merges `overlay` into `base`: objects key by key, recursively; anything else replaces.
pub fn merge(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Object(base), Value::Object(overlay)) => {
            for (key, value) in overlay {
                match base.get_mut(&key) {
                    Some(existing) if existing.is_object() && value.is_object() => merge(existing, value),
                    _ => {
                        base.insert(key, value);
                    }
                }
            }
        }
        (base, overlay) => *base = overlay,
    }
}

/// Merges `configs` in order, later ones winning (upstream's `recursive_merge`).
pub fn merge_all(configs: impl IntoIterator<Item = Value>) -> Value {
    let mut merged = Value::Object(Map::new());
    for config in configs {
        merge(&mut merged, config);
    }
    merged
}

/// The configs of `specs`, merged in order.
pub fn load(specs: &[String]) -> Result<Value, ConfigError> {
    let configs = specs
        .iter()
        .map(|spec| config_from_spec(spec))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(merge_all(configs))
}

/// Command-line overrides: each `Some` value lands at its path; `None` changes nothing
/// (upstream's `UNSET`).
#[derive(Clone, Debug, Default)]
pub struct Overrides(Vec<(&'static str, Value)>);

impl Overrides {
    #[must_use]
    pub fn set(mut self, path: &'static str, value: Option<impl Into<Value>>) -> Self {
        if let Some(value) = value {
            self.0.push((path, value.into()));
        }
        self
    }

    pub fn apply(self, config: &mut Value) {
        for (path, value) in self.0 {
            let keys: Vec<&str> = path.split('.').collect();
            set_path(config, &keys, value);
        }
    }
}

/// The object at `key` of `config`, empty when absent.
pub fn section(config: &Value, key: &str) -> Value {
    config
        .get(key)
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_are_files_builtins_or_key_values() {
        let overlay = config_from_spec("model.model_kwargs.temperature=0.5").unwrap();
        assert_eq!(
            overlay,
            serde_json::json!({ "model": { "model_kwargs": { "temperature": 0.5 } } })
        );
        assert_eq!(
            config_from_spec("agent.mode=yolo").unwrap(),
            serde_json::json!({ "agent": { "mode": "yolo" } })
        );
        assert!(matches!(config_from_spec("a..b=1"), Err(ConfigError::EmptyKey(_))));
        let mini = config_from_spec("mini.yaml").unwrap();
        assert_eq!(mini["agent"]["mode"], "confirm");
        assert_eq!(config_from_spec("swebench").unwrap()["environment"]["cwd"], "/testbed");
        assert!(matches!(
            config_from_spec("nope.yaml"),
            Err(ConfigError::NotFound { .. })
        ));
    }

    #[test]
    fn block_scalars_read_as_pyyaml_reads_them() {
        let parse = |text: &str| parse_yaml(text, "test").unwrap()["a"].clone();
        assert_eq!(parse("a: |\n  x"), "x");
        assert_eq!(parse("a: |\n  x\n"), "x\n");
        assert_eq!(parse("a: >\n  x\n  y"), "x y");
        assert_eq!(parse("a: \"x\\n\""), "x\n");
        assert_eq!(parse("a: [1, \"x\\n\"]")[1], "x\n");
        assert_eq!(parse("a: |\n  x\nb: 1"), "x\n");
        assert_eq!(parse("a: |\r\n  x\r\n  y\r\n"), "x\ny\n");
        for name in ["default.yaml", "mini_textbased.yaml"] {
            let config = config_from_spec(name).unwrap();
            let template = config["model"]["format_error_template"].as_str().unwrap();
            assert!(template.ends_with("{%- endif %}"), "{name}");
        }
    }

    #[test]
    fn later_configs_win_and_objects_merge() {
        let merged = merge_all([
            serde_json::json!({ "agent": { "mode": "confirm", "cost_limit": 3.0 }, "model": { "a": 1 } }),
            serde_json::json!({ "agent": { "mode": "yolo" }, "model": "replaced" }),
        ]);
        assert_eq!(
            merged,
            serde_json::json!({ "agent": { "mode": "yolo", "cost_limit": 3.0 }, "model": "replaced" })
        );
        let mut config = merged;
        Overrides::default()
            .set("agent.cost_limit", Some(1.5))
            .set("model.model_name", None::<String>)
            .apply(&mut config);
        assert_eq!(config["agent"]["cost_limit"], 1.5);
    }
}
