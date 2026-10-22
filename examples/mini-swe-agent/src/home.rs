//! The home directory: `$DELTA_MINI_HOME`, else `delta-mini` in the user's config directory.
//! It holds the global config (`config.yaml`, the lowest config layer of every run), the
//! credentials (`auth.json`), the prompt history and the last runs' trajectories.

use std::path::PathBuf;

use serde_json::{Map, Value as Json};

use crate::config::{self, ConfigError};

/// The home directory.
pub fn home_dir() -> PathBuf {
    std::env::var_os("DELTA_MINI_HOME")
        .filter(|dir| !dir.is_empty())
        .map_or_else(
            || {
                dirs::config_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("delta-mini")
            },
            PathBuf::from,
        )
}

/// The global config file.
pub fn config_file() -> PathBuf {
    home_dir().join("config.yaml")
}

/// The credentials file, in the shape of pi's `auth.json`.
pub fn auth_file() -> PathBuf {
    home_dir().join("auth.json")
}

/// The global config; empty when the file does not exist.
pub fn global_config() -> Result<Json, ConfigError> {
    let path = config_file();
    match std::fs::read_to_string(&path) {
        Ok(text) => config::parse_yaml(&text, &path.display().to_string()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Json::Object(Map::new())),
        Err(source) => Err(ConfigError::Read {
            path: path.display().to_string(),
            source,
        }),
    }
}

/// Sets `model.provider` and `model.id` in the global config file, keeping its other values
/// (comments are not kept).
pub fn set_default_model(provider: &str, id: &str) -> Result<PathBuf, String> {
    let mut global = global_config().map_err(|error| error.to_string())?;
    config::set_path(&mut global, &["model", "provider"], Json::String(provider.to_owned()));
    config::set_path(&mut global, &["model", "id"], Json::String(id.to_owned()));
    let text = serde_saphyr::to_string(&global).map_err(|error| error.to_string())?;
    let path = config_file();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| format!("Could not create {}: {error}", parent.display()))?;
    }
    std::fs::write(&path, text).map_err(|error| format!("Could not write {}: {error}", path.display()))?;
    Ok(path)
}

/// Opens the global config file in `$EDITOR` (`nano`, or `notepad` on Windows).
pub fn edit_config() -> std::io::Result<()> {
    let path = config_file();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if !path.exists() {
        std::fs::write(
            &path,
            "# delta-mini global config: the lowest layer of every run's config.\n",
        )?;
    }
    let editor = std::env::var("EDITOR")
        .ok()
        .filter(|editor| !editor.trim().is_empty())
        .unwrap_or_else(|| if cfg!(windows) { "notepad" } else { "nano" }.to_owned());
    let status = std::process::Command::new(editor).arg(&path).status()?;
    if !status.success() {
        return Err(std::io::Error::other(format!("the editor exited with {status}")));
    }
    Ok(())
}
