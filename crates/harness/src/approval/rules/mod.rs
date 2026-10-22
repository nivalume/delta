//! Permission rules: Claude Code's `Tool(specifier)` grammar, and how a rule matches a call
//! (`docs/design/permissions.md`).

mod bash;
mod path;

use std::fmt;
use std::str::FromStr;

pub use bash::split_command;
pub use path::{Anchor, Anchors, PathPattern, rule_path, windows_rule_path};

use crate::tool::{PathAccess, PermissionSubject};

/// Why a rule does not parse.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RuleError {
    #[error("Rule is empty")]
    Empty,
    #[error("Rule {0:?} has unbalanced parentheses")]
    Unbalanced(String),
    #[error("Rule {0:?} has an empty specifier")]
    EmptySpecifier(String),
    /// Only `Bash`, `Read`, `Edit` and `Write` take a specifier.
    #[error("Rule {0:?}: only Bash, Read, Edit and Write take a specifier")]
    UnsupportedSpecifier(String),
    /// A `/path` rule where no root is configured.
    #[error("Rule {0:?} is relative to the root, and no root is configured")]
    NoRoot(String),
}

/// What a rule matches.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Matcher {
    /// Command subjects; `None`: every command.
    Command(Option<String>),
    /// Path subjects of `access`; `None`: every path.
    Path(PathAccess, Option<PathPattern>),
    /// The tool of this name, whatever its subject.
    Tool(String),
}

/// One rule, as written (`Bash(cargo test:*)`, `Read(./.env)`, `my_tool`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rule {
    text: String,
    matcher: Matcher,
}

impl FromStr for Rule {
    type Err = RuleError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let text = text.trim();
        if text.is_empty() {
            return Err(RuleError::Empty);
        }
        let (name, specifier) = match text.find('(') {
            None if text.contains(')') => return Err(RuleError::Unbalanced(text.to_owned())),
            None => (text, None),
            Some(open) => {
                let Some(inner) = text[open + 1..].strip_suffix(')') else {
                    return Err(RuleError::Unbalanced(text.to_owned()));
                };
                let inner = inner.trim();
                if inner.is_empty() {
                    return Err(RuleError::EmptySpecifier(text.to_owned()));
                }
                (text[..open].trim(), (inner != "*").then_some(inner))
            }
        };
        if name.is_empty() {
            return Err(RuleError::Empty);
        }
        let matcher = match name.to_ascii_lowercase().as_str() {
            "bash" => Matcher::Command(specifier.map(str::to_owned)),
            "read" => Matcher::Path(PathAccess::Read, specifier.map(PathPattern::parse)),
            "edit" | "write" => Matcher::Path(PathAccess::Write, specifier.map(PathPattern::parse)),
            _ if specifier.is_some() => return Err(RuleError::UnsupportedSpecifier(text.to_owned())),
            _ => Matcher::Tool(name.to_owned()),
        };
        Ok(Self {
            text: text.to_owned(),
            matcher,
        })
    }
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

/// The call a rule is matched against.
#[derive(Clone, Copy, Debug)]
pub struct RuleCall<'a> {
    pub tool: &'a str,
    pub subject: Option<&'a PermissionSubject>,
    pub anchors: Anchors<'a>,
}

impl Rule {
    /// The rule as written.
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// Whether the rule depends on the policy's root (`/path`).
    pub fn needs_root(&self) -> bool {
        matches!(&self.matcher, Matcher::Path(_, Some(pattern)) if pattern.anchor == Anchor::Root)
    }

    /// Whether the rule, as a deny or ask rule, applies to `call`: a command rule when any
    /// sub-command matches (the whole text when it does not split), a `Read` rule also for
    /// writes.
    pub fn restricts(&self, call: &RuleCall<'_>) -> bool {
        match (&self.matcher, call.subject) {
            (Matcher::Tool(name), _) => name == call.tool,
            (Matcher::Command(None), Some(PermissionSubject::Command { .. })) => true,
            (Matcher::Command(Some(pattern)), Some(PermissionSubject::Command { command })) => {
                match split_command(command) {
                    Some(parts) => parts.iter().any(|part| bash::command_matches(pattern, part)),
                    None => bash::command_matches(pattern, command.trim()),
                }
            }
            (Matcher::Path(access, pattern), Some(PermissionSubject::Path { path, access: reach })) => {
                (access == reach || *access == PathAccess::Read)
                    && pattern
                        .as_ref()
                        .is_none_or(|pattern| pattern.matches(path, call.anchors))
            }
            _ => false,
        }
    }

    /// Whether the rule, as an allow rule, matches the sub-command `part`.
    fn allows_command(&self, part: &str) -> bool {
        match &self.matcher {
            Matcher::Command(None) => true,
            Matcher::Command(Some(pattern)) => bash::command_matches(pattern, part),
            _ => false,
        }
    }

    /// Whether the rule, as an allow rule, matches a call that is not a command.
    fn allows(&self, call: &RuleCall<'_>) -> bool {
        match (&self.matcher, call.subject) {
            (Matcher::Tool(name), _) => name == call.tool,
            (Matcher::Path(access, pattern), Some(PermissionSubject::Path { path, access: reach })) => {
                access == reach
                    && pattern
                        .as_ref()
                        .is_none_or(|pattern| pattern.matches(path, call.anchors))
            }
            _ => false,
        }
    }
}

/// Whether `rules`, as allow rules, allow `call`: a command when every sub-command matches
/// one of them (never a command that does not split), anything else when one matches. A
/// rule naming the tool allows every call of it.
pub fn allowed_by<'r>(rules: impl IntoIterator<Item = &'r Rule> + Clone, call: &RuleCall<'_>) -> bool {
    if rules
        .clone()
        .into_iter()
        .any(|rule| matches!(&rule.matcher, Matcher::Tool(name) if name == call.tool))
    {
        return true;
    }
    match call.subject {
        Some(PermissionSubject::Command { command }) => split_command(command).is_some_and(|parts| {
            parts
                .iter()
                .all(|part| rules.clone().into_iter().any(|rule| rule.allows_command(part)))
        }),
        _ => rules.into_iter().any(|rule| rule.allows(call)),
    }
}

/// The rule a UI offers to always allow a call like this one: `Bash(<stem>:*)` for a single
/// command (none for a compound one), `Edit(//<dir>/**)` or `Read(//<dir>/**)` for a path,
/// the tool's name without a subject.
pub fn propose(tool: &str, subject: Option<&PermissionSubject>) -> Option<String> {
    match subject {
        None => Some(tool.to_owned()),
        Some(PermissionSubject::Command { command }) => match split_command(command)?.as_slice() {
            [part] => Some(format!("Bash({}:*)", bash::command_stem(part))),
            _ => None,
        },
        Some(PermissionSubject::Path { path, access }) => {
            let kind = match access {
                PathAccess::Read => "Read",
                PathAccess::Write => "Edit",
            };
            let parent = path::parent_rule_path(path);
            let glob = if parent == "//" {
                "//**".to_owned()
            } else {
                format!("{parent}/**")
            };
            Some(format!("{kind}({glob})"))
        }
    }
}

/// Whether `path` is inside `directory` (absolute, or relative to `cwd`).
pub fn is_within(path: &str, directory: &str, cwd: Option<&str>) -> bool {
    if directory.starts_with('/') {
        return path::is_within(path, directory);
    }
    cwd.is_some_and(|cwd| path::is_within(path, &format!("{}/{directory}", cwd.trim_end_matches('/'))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(text: &str) -> Rule {
        text.parse().unwrap()
    }

    fn command(text: &str) -> PermissionSubject {
        PermissionSubject::Command {
            command: text.to_owned(),
        }
    }

    fn path(path: &str, access: PathAccess) -> PermissionSubject {
        PermissionSubject::Path {
            path: path.to_owned(),
            access,
        }
    }

    fn call<'a>(tool: &'a str, subject: &'a PermissionSubject) -> RuleCall<'a> {
        RuleCall {
            tool,
            subject: Some(subject),
            anchors: Anchors {
                cwd: Some("/work"),
                home: Some("/home/me"),
                root: None,
            },
        }
    }

    #[test]
    fn rules_parse_and_print_as_written() {
        assert_eq!(rule("Bash(cargo test:*)").to_string(), "Bash(cargo test:*)");
        assert_eq!(rule("  Read(./.env) ").as_str(), "Read(./.env)");
        assert_eq!(
            rule("Bash(*)"),
            Rule {
                text: "Bash(*)".to_owned(),
                matcher: Matcher::Command(None)
            }
        );
        assert_eq!(
            rule("bash(echo (a))").matcher,
            Matcher::Command(Some("echo (a)".to_owned()))
        );
        assert_eq!(rule("my_tool").matcher, Matcher::Tool("my_tool".to_owned()));
        assert!(rule("Edit(/src/**)").needs_root());
        assert_eq!("".parse::<Rule>(), Err(RuleError::Empty));
        assert_eq!(
            "Bash(ls".parse::<Rule>(),
            Err(RuleError::Unbalanced("Bash(ls".to_owned()))
        );
        assert_eq!(
            "Bash()".parse::<Rule>(),
            Err(RuleError::EmptySpecifier("Bash()".to_owned()))
        );
        assert_eq!(
            "fetch(x)".parse::<Rule>(),
            Err(RuleError::UnsupportedSpecifier("fetch(x)".to_owned()))
        );
    }

    #[test]
    fn deny_and_ask_rules_match_any_sub_command() {
        let push = command("cargo fmt && git push origin");
        assert!(rule("Bash(git push:*)").restricts(&call("bash", &push)));
        assert!(!rule("Bash(git pull:*)").restricts(&call("bash", &push)));
        // Unsplittable: matched against the whole text.
        assert!(rule("Bash(echo *)").restricts(&call("bash", &command("echo 'open"))));
        assert!(rule("Bash").restricts(&call("bash", &push)));
        assert!(
            rule("bash").restricts(&call("other", &push)),
            "subject kinds, not tool names"
        );
    }

    #[test]
    fn allow_rules_need_every_sub_command() {
        let rules = [rule("Bash(cargo test:*)"), rule("Bash(cargo fmt)")];
        let both = command("cargo fmt && cargo test --all");
        assert!(allowed_by(&rules, &call("bash", &both)));
        let more = command("cargo fmt && rm -rf target");
        assert!(!allowed_by(&rules, &call("bash", &more)));
        let open = command("cargo test 'x");
        assert!(
            !allowed_by(&[rule("Bash")], &call("bash", &open)),
            "an unsplittable command is never allowed"
        );
        assert!(allowed_by(&[rule("bash")], &call("bash", &command("ls; pwd"))));
        assert!(allowed_by(&[rule("other")], &call("other", &command("anything"))));
    }

    #[test]
    fn path_rules_follow_access() {
        let read_env = path("/work/.env", PathAccess::Read);
        let write_env = path("/work/.env", PathAccess::Write);
        assert!(rule("Read(./.env)").restricts(&call("read", &read_env)));
        assert!(
            rule("Read(./.env)").restricts(&call("write", &write_env)),
            "a Read deny covers writes"
        );
        assert!(!rule("Edit(./.env)").restricts(&call("read", &read_env)));
        assert!(allowed_by(
            &[rule("Edit(src/**)")],
            &call("write", &path("/work/src/a.rs", PathAccess::Write))
        ));
        assert!(!allowed_by(
            &[rule("Read(src/**)")],
            &call("write", &path("/work/src/a.rs", PathAccess::Write))
        ));
        assert!(allowed_by(&[rule("Write")], &call("edit", &write_env)));
    }

    #[test]
    fn proposals() {
        assert_eq!(
            propose("bash", Some(&command("git push origin main"))).as_deref(),
            Some("Bash(git push:*)")
        );
        assert_eq!(propose("bash", Some(&command("ls -la"))).as_deref(), Some("Bash(ls:*)"));
        assert_eq!(propose("bash", Some(&command("a && b"))), None);
        assert_eq!(
            propose("write", Some(&path("/work/src/a.rs", PathAccess::Write))).as_deref(),
            Some("Edit(//work/src/**)")
        );
        assert_eq!(
            propose("read", Some(&path("/a.txt", PathAccess::Read))).as_deref(),
            Some("Read(//**)")
        );
        assert_eq!(propose("fetch", None).as_deref(), Some("fetch"));
        for proposed in ["Bash(git push:*)", "Edit(//work/src/**)", "Read(//**)"] {
            proposed.parse::<Rule>().unwrap();
        }
        let allowed = rule("Edit(//work/src/**)");
        assert!(allowed_by(
            &[allowed],
            &call("write", &path("/work/src/a.rs", PathAccess::Write))
        ));
    }

    #[test]
    fn directories_contain_paths() {
        assert!(is_within("/work/a", "/work", None));
        assert!(is_within("/work/docs/a", "docs", Some("/work")));
        assert!(!is_within("/work/docs/a", "docs", None));
    }
}
