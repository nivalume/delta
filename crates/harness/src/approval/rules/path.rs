//! Path patterns for rule matching: anchors and segment globs, matched lexically.

/// What a path pattern is relative to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Anchor {
    /// `//path`
    Absolute,
    /// `~/path`
    Home,
    /// `/path`: the policy's root.
    Root,
    /// `./path` or `path`: the call's directory.
    Cwd,
}

/// A path pattern: its anchor and the glob after it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathPattern {
    pub anchor: Anchor,
    /// Without the anchor and leading slashes; may be empty (the anchor itself).
    pub glob: String,
}

/// The directories anchors stand for.
#[derive(Clone, Copy, Debug, Default)]
pub struct Anchors<'a> {
    pub cwd: Option<&'a str>,
    pub home: Option<&'a str>,
    pub root: Option<&'a str>,
}

impl PathPattern {
    pub fn parse(text: &str) -> Self {
        let (anchor, glob) = if let Some(rest) = text.strip_prefix("//") {
            (Anchor::Absolute, rest)
        } else if text == "~" {
            (Anchor::Home, "")
        } else if let Some(rest) = text.strip_prefix("~/") {
            (Anchor::Home, rest)
        } else if let Some(rest) = text.strip_prefix('/') {
            (Anchor::Root, rest)
        } else if text == "." {
            (Anchor::Cwd, "")
        } else {
            (Anchor::Cwd, text.strip_prefix("./").unwrap_or(text))
        };
        Self {
            anchor,
            glob: glob.trim_start_matches('/').to_owned(),
        }
    }

    /// The absolute glob, `None` when the anchor's directory is unknown.
    pub fn resolve(&self, anchors: Anchors<'_>) -> Option<String> {
        let base = match self.anchor {
            Anchor::Absolute => "/",
            Anchor::Home => anchors.home?,
            Anchor::Root => anchors.root?,
            Anchor::Cwd => anchors.cwd?,
        };
        Some(format!("{}/{}", base.trim_end_matches('/'), self.glob))
    }

    /// Whether `path` (absolute) is the pattern's path or below it.
    pub fn matches(&self, path: &str, anchors: Anchors<'_>) -> bool {
        let Some(glob) = self.resolve(anchors) else {
            return false;
        };
        let pattern = segments(&glob);
        let path = segments(path);
        match_segments(&pattern, &path)
    }
}

/// Whether `path` (absolute) is `directory` or below it, lexically.
pub(super) fn is_within(path: &str, directory: &str) -> bool {
    let directory = segments(directory);
    let path = segments(path);
    path.len() >= directory.len() && path[..directory.len()] == directory[..]
}

/// The segments of an absolute path with `.` and `..` resolved lexically.
pub(super) fn segments(path: &str) -> Vec<&str> {
    let mut out: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            segment => out.push(segment),
        }
    }
    out
}

/// The parent directory of an absolute path, as `//dir` for a rule.
pub(super) fn parent_rule_path(path: &str) -> String {
    let mut parts = segments(path);
    parts.pop();
    format!("//{}", parts.join("/"))
}

/// Pattern segments against path segments; the path may continue past the pattern.
fn match_segments(pattern: &[&str], path: &[&str]) -> bool {
    let Some((first, rest)) = pattern.split_first() else {
        return true;
    };
    if *first == "**" {
        return (0..=path.len()).any(|skip| match_segments(rest, &path[skip..]));
    }
    match path.split_first() {
        Some((segment, path_rest)) => segment_matches(first, segment) && match_segments(rest, path_rest),
        None => false,
    }
}

/// One segment: `*` any characters, `?` one character.
fn segment_matches(pattern: &str, segment: &str) -> bool {
    fn go(pattern: &[char], text: &[char]) -> bool {
        match pattern.split_first() {
            None => text.is_empty(),
            Some(('*', rest)) => (0..=text.len()).any(|skip| go(rest, &text[skip..])),
            Some(('?', rest)) => !text.is_empty() && go(rest, &text[1..]),
            Some((c, rest)) => text.first() == Some(c) && go(rest, &text[1..]),
        }
    }
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = segment.chars().collect();
    go(&pattern, &text)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ANCHORS: Anchors<'static> = Anchors {
        cwd: Some("/work/project"),
        home: Some("/home/me"),
        root: Some("/repo"),
    };

    fn matches(pattern: &str, path: &str) -> bool {
        PathPattern::parse(pattern).matches(path, ANCHORS)
    }

    #[test]
    fn anchors_resolve() {
        assert!(matches("//etc/passwd", "/etc/passwd"));
        assert!(matches("~/.ssh", "/home/me/.ssh/id_rsa"));
        assert!(matches("/config", "/repo/config/app.toml"));
        assert!(matches("./.env", "/work/project/.env"));
        assert!(matches(".env", "/work/project/.env"));
        assert!(!matches(".env", "/work/project/sub/.env"));
        assert!(matches(".", "/work/project/anything"));
        let rootless = Anchors { root: None, ..ANCHORS };
        assert!(!PathPattern::parse("/config").matches("/repo/config", rootless));
    }

    #[test]
    fn globs_match_segments() {
        assert!(matches("src/**", "/work/project/src/a/b.rs"));
        assert!(matches("src/**/*.rs", "/work/project/src/a/b.rs"));
        assert!(matches("src/**/*.rs", "/work/project/src/b.rs"));
        assert!(!matches("src/*.rs", "/work/project/src/a/b.toml"));
        assert!(matches("**/.env", "/work/project/deep/down/.env"));
        assert!(matches("file?.txt", "/work/project/file1.txt"));
        assert!(!matches("file?.txt", "/work/project/file10.txt"));
        assert!(matches("src/../docs", "/work/project/docs/x"));
    }

    #[test]
    fn within_and_parents() {
        assert!(is_within("/work/project/a", "/work/project"));
        assert!(is_within("/work/project", "/work/project/"));
        assert!(!is_within("/work/projects", "/work/project"));
        assert_eq!(parent_rule_path("/work/project/a.txt"), "//work/project");
        assert_eq!(parent_rule_path("/a.txt"), "//");
    }
}
