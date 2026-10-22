//! Templates: Jinja through minijinja with strict undefined (an unknown variable is an
//! error) and minijinja's own filters, `tojson` among them.

use minijinja::value::Serde;
use minijinja::{Environment, Error, UndefinedBehavior, Value};
use serde_json::{Map, Value as Json};

/// Renders `template` with `vars`, which must be a JSON object.
pub fn render(template: &str, vars: &Json) -> Result<String, Error> {
    environment().render_str(template, Value::from(Serde(vars)))
}

fn environment() -> Environment<'static> {
    let mut environment = Environment::new();
    environment.set_undefined_behavior(UndefinedBehavior::Strict);
    environment
}

/// Merges `layers` into one object, later layers winning.
pub fn vars(layers: impl IntoIterator<Item = Json>) -> Json {
    crate::config::merge_all(layers.into_iter().map(|layer| match layer {
        Json::Object(_) => layer,
        _ => Json::Object(Map::new()),
    }))
}

/// The machine the agent runs on, as template variables: `system` (`Linux`, `Darwin`,
/// `Windows`), `release`, `version` and `machine` as `uname` reports them.
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
    serde_json::json!({
        "system": system,
        "release": field("-r").unwrap_or_default(),
        "version": field("-v").unwrap_or_default(),
        "machine": field("-m").unwrap_or_else(|| std::env::consts::ARCH.to_owned()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_undefined_slices_and_tojson() {
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
        assert_eq!(render("{{ output.output | tojson }}", &vars).unwrap(), "\"abcdef\"");
    }
}
