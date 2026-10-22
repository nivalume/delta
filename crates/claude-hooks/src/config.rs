//! The `hooks` object of Claude Code's settings: events, matcher groups and handlers.

use std::collections::BTreeMap;
use std::time::Duration;

use regex::Regex;
use serde::Deserialize;

/// Events this bridge runs; the others are kept in [`HooksConfig::unsupported`].
pub const SUPPORTED_EVENTS: [&str; 8] = [
    "PreToolUse",
    "PermissionRequest",
    "PostToolUse",
    "PostToolUseFailure",
    "UserPromptSubmit",
    "Stop",
    "SubagentStop",
    "PreCompact",
];

/// A handler's default timeout; `UserPromptSubmit` handlers default to 30 s.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_mins(10);
const PROMPT_TIMEOUT: Duration = Duration::from_secs(30);

/// Why a configuration does not load.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Invalid hooks configuration: {0}")]
    Shape(#[from] serde_json::Error),
    #[error("Invalid matcher {matcher:?}: {source}")]
    Matcher {
        matcher: String,
        #[source]
        source: regex::Error,
    },
}

/// One handler: a command run through the conversation's environment, or an HTTP POST.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum HookCommand {
    Command {
        command: String,
        /// Seconds.
        #[serde(default)]
        timeout: Option<f64>,
    },
    Http {
        url: String,
        /// Values may name `$VAR` or `${VAR}` from `allowedEnvVars`.
        #[serde(default)]
        headers: BTreeMap<String, String>,
        #[serde(default, rename = "allowedEnvVars")]
        allowed_env_vars: Vec<String>,
        /// Seconds.
        #[serde(default)]
        timeout: Option<f64>,
    },
}

impl HookCommand {
    /// The handler's timeout for `event`.
    pub fn timeout(&self, event: &str) -> Duration {
        let seconds = match self {
            Self::Command { timeout, .. } | Self::Http { timeout, .. } => *timeout,
        };
        match seconds {
            Some(seconds) if seconds.is_finite() && seconds > 0.0 => Duration::from_secs_f64(seconds),
            _ if event == "UserPromptSubmit" => PROMPT_TIMEOUT,
            _ => DEFAULT_TIMEOUT,
        }
    }
}

/// What a group's `matcher` matches: every value, exact names, or a regular expression.
#[derive(Clone, Debug)]
pub enum Matcher {
    All,
    Names(Vec<String>),
    Pattern(Regex),
}

impl Matcher {
    /// `*`, empty or absent: all; letters, digits, `_` and `|`: exact names; else an
    /// unanchored regular expression.
    pub fn parse(matcher: Option<&str>) -> Result<Self, ConfigError> {
        let text = matcher.unwrap_or("").trim();
        if text.is_empty() || text == "*" {
            return Ok(Self::All);
        }
        if text.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '|') {
            return Ok(Self::Names(text.split('|').map(str::to_owned).collect()));
        }
        Regex::new(text)
            .map(Self::Pattern)
            .map_err(|source| ConfigError::Matcher {
                matcher: text.to_owned(),
                source,
            })
    }

    /// Whether any of `values` matches.
    pub fn matches(&self, values: &[&str]) -> bool {
        match self {
            Self::All => true,
            Self::Names(names) => values.iter().any(|value| names.iter().any(|name| name == value)),
            Self::Pattern(regex) => values.iter().any(|value| regex.is_match(value)),
        }
    }
}

#[derive(Deserialize)]
struct RawGroup {
    #[serde(default)]
    matcher: Option<String>,
    #[serde(default)]
    hooks: Vec<HookCommand>,
}

/// A matcher and its handlers.
#[derive(Clone, Debug)]
pub struct MatcherGroup {
    pub matcher: Matcher,
    pub hooks: Vec<HookCommand>,
}

/// The `hooks` object of a settings file, by event name.
#[derive(Clone, Debug, Default)]
pub struct HooksConfig {
    events: BTreeMap<String, Vec<MatcherGroup>>,
}

impl HooksConfig {
    /// Parses the `hooks` object (`{ "PreToolUse": [{ "matcher": "Bash", "hooks": [...] }] }`).
    pub fn from_value(value: serde_json::Value) -> Result<Self, ConfigError> {
        let raw: BTreeMap<String, Vec<RawGroup>> = serde_json::from_value(value)?;
        let mut events = BTreeMap::new();
        for (event, groups) in raw {
            let groups = groups
                .into_iter()
                .map(|group| {
                    Ok(MatcherGroup {
                        matcher: Matcher::parse(group.matcher.as_deref())?,
                        hooks: group.hooks,
                    })
                })
                .collect::<Result<Vec<_>, ConfigError>>()?;
            events.insert(event, groups);
        }
        Ok(Self { events })
    }

    /// Parses a whole settings file and takes its `hooks` object; none is an empty
    /// configuration.
    pub fn from_settings(settings: &serde_json::Value) -> Result<Self, ConfigError> {
        match settings.get("hooks") {
            Some(hooks) => Self::from_value(hooks.clone()),
            None => Ok(Self::default()),
        }
    }

    /// The handlers of `event` whose matcher matches one of `values`, with their group
    /// index; an event without a matcher target passes no values and runs every group.
    pub fn handlers(&self, event: &str, values: &[&str]) -> Vec<(usize, &HookCommand)> {
        self.events
            .get(event)
            .into_iter()
            .flatten()
            .enumerate()
            .filter(|(_, group)| values.is_empty() || group.matcher.matches(values))
            .flat_map(|(index, group)| group.hooks.iter().map(move |hook| (index, hook)))
            .collect()
    }

    /// Configured events this bridge does not run (`SessionStart`, `Notification`, ...).
    pub fn unsupported(&self) -> Vec<&str> {
        self.events
            .keys()
            .map(String::as_str)
            .filter(|event| !SUPPORTED_EVENTS.contains(event))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matchers_take_all_names_or_patterns() {
        assert!(Matcher::parse(None).unwrap().matches(&["Bash"]));
        assert!(Matcher::parse(Some("*")).unwrap().matches(&["anything"]));
        let names = Matcher::parse(Some("Edit|Write")).unwrap();
        assert!(names.matches(&["Write"]));
        assert!(!names.matches(&["WriteFile"]));
        let pattern = Matcher::parse(Some("mcp__.*")).unwrap();
        assert!(pattern.matches(&["mcp__github__search"]));
        assert!(Matcher::parse(Some("(")).is_err());
    }

    #[test]
    fn configurations_load_and_name_unsupported_events() {
        let config = HooksConfig::from_value(serde_json::json!({
            "PreToolUse": [
                { "matcher": "Bash", "hooks": [{ "type": "command", "command": "check.sh", "timeout": 5 }] },
                { "hooks": [{ "type": "http", "url": "http://localhost/hook", "headers": { "Authorization": "Bearer $TOKEN" }, "allowedEnvVars": ["TOKEN"] }] }
            ],
            "SessionStart": [{ "hooks": [{ "type": "command", "command": "hello" }] }]
        }))
        .unwrap();
        assert_eq!(config.handlers("PreToolUse", &["Bash"]).len(), 2);
        assert_eq!(config.handlers("PreToolUse", &["Read"]).len(), 1);
        let (_, hook) = config.handlers("PreToolUse", &["Bash"])[0];
        assert_eq!(hook.timeout("PreToolUse"), Duration::from_secs(5));
        let (_, http) = config.handlers("PreToolUse", &["Read"])[0];
        assert_eq!(http.timeout("UserPromptSubmit"), Duration::from_secs(30));
        assert_eq!(config.unsupported(), ["SessionStart"]);
    }
}
