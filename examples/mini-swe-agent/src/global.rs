//! The global config: a `.env` file in the global config directory (upstream's
//! `global_config_file`) holding the default model and API keys, and `mini-extra config`.
//!
//! Upstream loads the file into the process environment. Here its values stay in this
//! module: [`env_var`] reads the process environment first and the file second (the file never
//! overrides, as `load_dotenv` does not), and [`DotenvAuthContext`] gives delta-ai's
//! providers the same view for their credentials.

use std::collections::BTreeMap;
use std::io::IsTerminal as _;
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};

use delta_ai::auth::{AuthContext, DefaultAuthContext};

/// The global config directory: `$MSWEA_GLOBAL_CONFIG_DIR`, else `mini-swe-agent` in the
/// user's config directory.
pub fn global_config_dir() -> PathBuf {
    std::env::var_os("MSWEA_GLOBAL_CONFIG_DIR")
        .filter(|dir| !dir.is_empty())
        .map_or_else(
            || {
                dirs::config_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("mini-swe-agent")
            },
            PathBuf::from,
        )
}

/// The `.env` file in the global config directory.
pub fn global_config_file() -> PathBuf {
    global_config_dir().join(".env")
}

fn values() -> &'static RwLock<BTreeMap<String, String>> {
    static VALUES: OnceLock<RwLock<BTreeMap<String, String>>> = OnceLock::new();
    VALUES.get_or_init(|| RwLock::new(read_dotenv(&global_config_file())))
}

/// Reads the global config file again.
pub fn reload() {
    *values().write().unwrap_or_else(std::sync::PoisonError::into_inner) = read_dotenv(&global_config_file());
}

/// A variable: the process environment's, else the global config file's. Blank counts as
/// unset.
pub fn env_var(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .or_else(|| {
            values()
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(name)
                .cloned()
        })
        .filter(|value| !value.trim().is_empty())
}

/// The global config file's values the process environment does not set.
pub fn file_only_values() -> BTreeMap<String, String> {
    values()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .filter(|(key, _)| std::env::var_os(key).is_none())
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

/// delta-ai's view of the environment, with the global config file under it.
#[derive(Clone, Copy, Debug, Default)]
pub struct DotenvAuthContext;

impl AuthContext for DotenvAuthContext {
    fn env(&self, name: &str) -> Option<String> {
        env_var(name)
    }

    fn file_exists(&self, path: &str) -> bool {
        DefaultAuthContext.file_exists(path)
    }
}

/// The `KEY=VALUE` pairs of a `.env` file: `export ` prefixes, `#` comments and single or
/// double quotes as python-dotenv reads them.
pub fn parse_dotenv(text: &str) -> BTreeMap<String, String> {
    let mut values = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        values.insert(key.trim().to_owned(), unquote(value.trim()));
    }
    values
}

fn unquote(value: &str) -> String {
    let quoted = |quote: char| value.len() >= 2 && value.starts_with(quote) && value.ends_with(quote);
    if quoted('\'') {
        return value[1..value.len() - 1].to_owned();
    }
    if quoted('"') {
        return value[1..value.len() - 1]
            .replace("\\n", "\n")
            .replace("\\\"", "\"")
            .replace("\\\\", "\\");
    }
    // An unquoted value ends at an inline comment.
    value
        .split_once(" #")
        .map_or(value, |(value, _)| value)
        .trim_end()
        .to_owned()
}

fn read_dotenv(path: &Path) -> BTreeMap<String, String> {
    std::fs::read_to_string(path)
        .map(|text| parse_dotenv(&text))
        .unwrap_or_default()
}

/// Sets `key` in the file at `path` as python-dotenv's `set_key` writes it: `KEY='value'`,
/// replacing the key's line or appending one.
pub fn set_key(path: &Path, key: &str, value: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let line = format!("{key}='{}'", value.replace('\'', "\\'"));
    let mut found = false;
    let mut lines: Vec<String> = text
        .lines()
        .map(|existing| {
            if line_key(existing) == Some(key) {
                found = true;
                line.clone()
            } else {
                existing.to_owned()
            }
        })
        .collect();
    if !found {
        lines.push(line);
    }
    std::fs::write(path, lines.join("\n") + "\n")
}

/// Removes `key` from the file at `path`; whether it was there.
pub fn unset_key(path: &Path, key: &str) -> std::io::Result<bool> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(false);
    };
    let kept: Vec<&str> = text.lines().filter(|line| line_key(line) != Some(key)).collect();
    let removed = kept.len() != text.lines().count();
    std::fs::write(path, kept.join("\n") + "\n")?;
    Ok(removed)
}

fn line_key(line: &str) -> Option<&str> {
    let line = line.trim();
    let line = line.strip_prefix("export ").unwrap_or(line);
    line.split_once('=').map(|(key, _)| key.trim())
}

/// What `mini-extra config setup` prints first.
pub const SETUP_HELP: &str = "To get started, we need to set up your global config file.

You can edit it manually or use the `mini-extra config set` or `mini-extra config edit` commands.

This setup will ask you for your model and an API key.

Here's a few popular models and the required API keys:

anthropic/claude-sonnet-4-5 (ANTHROPIC_API_KEY)
openai/gpt-5 or openai/gpt-5-mini (OPENAI_API_KEY)
gemini/gemini-2.5-pro (GEMINI_API_KEY)

Note: Please always include the provider (e.g., \"openai/\") in the model name.

You can leave any setting blank to skip it.
";

/// Asks for a line on the terminal; `None` at the end of input.
fn ask(question: &str, default: Option<&str>) -> Option<String> {
    let mut editor = rustyline::DefaultEditor::new().ok()?;
    let line = match default.filter(|default| !default.is_empty()) {
        Some(default) => editor.readline_with_initial(question, (default, "")),
        None => editor.readline(question),
    };
    line.ok().map(|line| line.trim().to_owned())
}

/// `mini-extra config setup`: the default model and an API key.
pub fn setup() -> std::io::Result<()> {
    let file = global_config_file();
    println!("{SETUP_HELP}");
    let current = env_var("MSWEA_MODEL_NAME");
    let model = ask(
        "Enter your default model (e.g., anthropic/claude-sonnet-4-5): ",
        current.as_deref(),
    )
    .unwrap_or_default();
    if !model.is_empty() {
        set_key(&file, "MSWEA_MODEL_NAME", &model)?;
    }
    println!("If you already have your API keys set as environment variables, you can ignore the next question.");
    let key_name = ask("Enter your API key name (e.g., ANTHROPIC_API_KEY): ", None).unwrap_or_default();
    let mut key_value = String::new();
    if !key_name.is_empty() {
        let current = env_var(&key_name);
        key_value = ask("Enter your API key value (e.g., sk-1234567890): ", current.as_deref()).unwrap_or_default();
        if !key_value.is_empty() {
            set_key(&file, &key_name, &key_value)?;
        }
    }
    if key_value.is_empty() {
        println!("API key setup not completed. Totally fine if you have your keys as environment variables.");
    }
    set_key(&file, "MSWEA_CONFIGURED", "true")?;
    reload();
    println!("\nConfig finished. If you want to revisit it, run `mini-extra config setup`.");
    Ok(())
}

/// Runs the setup once, when nothing is configured and someone is at the terminal.
pub fn configure_if_first_time() -> std::io::Result<()> {
    if env_var("MSWEA_CONFIGURED").is_some() || !std::io::stdin().is_terminal() {
        return Ok(());
    }
    setup()
}

/// `mini-extra config set [KEY [VALUE]]`.
pub fn set(key: Option<String>, value: Option<String>) -> std::io::Result<()> {
    let Some(key) = key.or_else(|| ask("Enter the key to set: ", None)) else {
        return Ok(());
    };
    let Some(value) = value.or_else(|| ask(&format!("Enter the value for {key}: "), None)) else {
        return Ok(());
    };
    set_key(&global_config_file(), &key, &value)?;
    reload();
    Ok(())
}

/// `mini-extra config unset [KEY]`.
pub fn unset(key: Option<String>) -> std::io::Result<()> {
    let Some(key) = key.or_else(|| ask("Enter the key to unset: ", None)) else {
        return Ok(());
    };
    unset_key(&global_config_file(), &key)?;
    reload();
    Ok(())
}

/// `mini-extra config edit`: the file in `$EDITOR` (default `nano`).
pub fn edit() -> std::io::Result<()> {
    let file = global_config_file();
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "nano".to_owned());
    std::process::Command::new(editor).arg(&file).status()?;
    reload();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dotenv_files_read_and_write_as_python_dotenv() {
        let values = parse_dotenv(
            "# comment\nMSWEA_MODEL_NAME='anthropic/claude-sonnet-4-5'\nexport A=\"x\\ny\"\nB=plain # note\nEMPTY=\n",
        );
        assert_eq!(values["MSWEA_MODEL_NAME"], "anthropic/claude-sonnet-4-5");
        assert_eq!(values["A"], "x\ny");
        assert_eq!(values["B"], "plain");
        assert_eq!(values["EMPTY"], "");

        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("sub/.env");
        set_key(&file, "A", "1").unwrap();
        set_key(&file, "B", "2").unwrap();
        set_key(&file, "A", "3").unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "A='3'\nB='2'\n");
        assert!(unset_key(&file, "A").unwrap());
        assert!(!unset_key(&file, "A").unwrap());
        assert_eq!(
            parse_dotenv(&std::fs::read_to_string(&file).unwrap()),
            BTreeMap::from([("B".to_owned(), "2".to_owned())])
        );
    }
}
