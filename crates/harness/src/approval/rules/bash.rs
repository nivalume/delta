//! Shell commands for rule matching: splitting a command into sub-commands, stripping
//! assignments and wrappers, and matching a sub-command against a command pattern.

/// Wrappers that run the command after them; rules match what they wrap.
const WRAPPERS: [&str; 7] = ["timeout", "time", "nice", "nohup", "stdbuf", "command", "builtin"];

/// The sub-commands of `command`: split at `&&`, `||`, `;`, `|`, `|&`, `&` and newlines
/// outside quotes, `$(...)` and backticks; trimmed, without leading `NAME=value`
/// assignments and wrappers, empty ones dropped. `None` when quotes, `$(` or backticks are
/// unbalanced.
pub fn split_command(command: &str) -> Option<Vec<String>> {
    let chars: Vec<char> = command.chars().collect();
    let mut parts = Vec::new();
    let mut current = String::new();
    let (mut single, mut double, mut backtick) = (false, false, false);
    let mut depth = 0usize;
    let mut index = 0;
    while index < chars.len() {
        let c = chars[index];
        let next = chars.get(index + 1).copied();
        let nested = single || double || backtick || depth > 0;
        match c {
            '\\' if !single => {
                current.push(c);
                if let Some(next) = next {
                    current.push(next);
                    index += 1;
                }
            }
            '\'' if !double => {
                single = !single;
                current.push(c);
            }
            '"' if !single => {
                double = !double;
                current.push(c);
            }
            '`' if !single => {
                backtick = !backtick;
                current.push(c);
            }
            '$' if !single && next == Some('(') => {
                depth += 1;
                current.push_str("$(");
                index += 1;
            }
            ')' if !single && depth > 0 => {
                depth -= 1;
                current.push(c);
            }
            ';' | '\n' if !nested => parts.push(std::mem::take(&mut current)),
            '|' if !nested => {
                if matches!(next, Some('|' | '&')) {
                    index += 1;
                }
                parts.push(std::mem::take(&mut current));
            }
            '&' if !nested && next == Some('&') => {
                index += 1;
                parts.push(std::mem::take(&mut current));
            }
            // `2>&1`, `&>file` and `>&2` redirect; they do not separate.
            '&' if !nested && !is_redirection(&chars, index) => parts.push(std::mem::take(&mut current)),
            _ => current.push(c),
        }
        index += 1;
    }
    if single || double || backtick || depth > 0 {
        return None;
    }
    parts.push(current);
    Some(
        parts
            .iter()
            .map(|part| strip_prefixes(part.trim()))
            .filter(|part| !part.is_empty())
            .collect(),
    )
}

/// Whether the `&` at `index` belongs to a redirection (`>&`, `<&`, `&>`).
fn is_redirection(chars: &[char], index: usize) -> bool {
    let before = index.checked_sub(1).map(|at| chars[at]);
    matches!(before, Some('>' | '<')) || chars.get(index + 1) == Some(&'>')
}

/// Where the first word of `text` ends (quotes respected).
fn word_end(text: &str) -> usize {
    let (mut single, mut double) = (false, false);
    for (index, c) in text.char_indices() {
        match c {
            '\'' if !double => single = !single,
            '"' if !single => double = !double,
            c if c.is_whitespace() && !single && !double => return index,
            _ => {}
        }
    }
    text.len()
}

/// `text` split into its first word and the rest, trimmed.
fn first_word(text: &str) -> (&str, &str) {
    let end = word_end(text);
    (&text[..end], text[end..].trim_start())
}

/// Whether `option` of `wrapper` is followed by its value as a separate word.
fn takes_value(wrapper: &str, option: &str) -> bool {
    matches!(
        (wrapper, option),
        ("nice", "-n" | "--adjustment")
            | ("timeout", "-s" | "-k" | "--signal" | "--kill-after")
            | ("stdbuf", "-i" | "-o" | "-e")
    )
}

fn is_assignment(word: &str) -> bool {
    word.split_once('=').is_some_and(|(name, _)| {
        name.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    })
}

/// `part` without leading assignments and wrappers (with their options; `timeout`'s
/// duration and `nice -n`'s value too).
fn strip_prefixes(part: &str) -> String {
    let mut rest = part;
    loop {
        let (word, after) = first_word(rest);
        if word.is_empty() {
            return String::new();
        }
        if is_assignment(word) {
            rest = after;
            continue;
        }
        if !WRAPPERS.contains(&word) {
            return rest.to_owned();
        }
        rest = after;
        loop {
            let (option, after) = first_word(rest);
            if !option.starts_with('-') || option == "-" {
                break;
            }
            rest = after;
            if takes_value(word, option) {
                rest = first_word(rest).1;
            }
        }
        if word == "timeout" {
            rest = first_word(rest).1;
        }
    }
}

/// Whether the sub-command `command` matches `pattern`: `*` matches any text; a trailing
/// ` *` or `:*` also matches the bare command.
pub(super) fn command_matches(pattern: &str, command: &str) -> bool {
    let prefix = pattern.strip_suffix(":*").or_else(|| pattern.strip_suffix(" *"));
    match prefix {
        Some(base) => wildcard(base, command) || wildcard(&format!("{base} *"), command),
        None => wildcard(pattern, command),
    }
}

/// `*` matches any (possibly empty) text; everything else literally.
fn wildcard(pattern: &str, text: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();
    let (mut p, mut t) = (0, 0);
    // The last `*` and the text position it was tried at, for backtracking.
    let mut star: Option<(usize, usize)> = None;
    while t < text.len() {
        if p < pattern.len() && pattern[p] == '*' {
            star = Some((p, t));
            p += 1;
        } else if p < pattern.len() && pattern[p] == text[t] {
            p += 1;
            t += 1;
        } else if let Some((at, from)) = star {
            p = at + 1;
            t = from + 1;
            star = Some((at, from + 1));
        } else {
            return false;
        }
    }
    pattern[p..].iter().all(|&c| c == '*')
}

/// The words of a sub-command for a proposed rule: its first word, and the second unless
/// it is an option.
pub(super) fn command_stem(part: &str) -> String {
    let (first, rest) = first_word(part);
    let (second, _) = first_word(rest);
    if second.is_empty() || second.starts_with('-') {
        first.to_owned()
    } else {
        format!("{first} {second}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(command: &str) -> Vec<String> {
        split_command(command).unwrap()
    }

    #[test]
    fn splits_at_operators_outside_quotes() {
        assert_eq!(
            split("ls && echo 'a && b' | wc -l; pwd"),
            ["ls", "echo 'a && b'", "wc -l", "pwd"]
        );
        assert_eq!(split("a || b |& c & d\ne"), ["a", "b", "c", "d", "e"]);
        assert_eq!(
            split("echo \"x; y\" $(a; b) `c | d`"),
            ["echo \"x; y\" $(a; b) `c | d`"]
        );
        assert_eq!(split("make 2>&1 >&2 &> log"), ["make 2>&1 >&2 &> log"]);
        assert_eq!(split("echo a\\;b"), ["echo a\\;b"]);
        assert_eq!(split("  ;  "), Vec::<String>::new());
    }

    #[test]
    fn unbalanced_commands_do_not_split() {
        assert_eq!(split_command("echo 'open"), None);
        assert_eq!(split_command("echo $(date"), None);
        assert_eq!(split_command("echo `x"), None);
    }

    #[test]
    fn assignments_and_wrappers_are_stripped() {
        assert_eq!(split("FOO=1 BAR=\"a b\" cargo test"), ["cargo test"]);
        assert_eq!(split("timeout -s KILL 30 nice -n 5 nohup git push"), ["git push"]);
        assert_eq!(split("time -p command -v ls"), ["ls"]);
        assert_eq!(split("stdbuf -o0 tail -f log"), ["tail -f log"]);
    }

    #[test]
    fn patterns_match_prefixes_and_wildcards() {
        assert!(command_matches("ls *", "ls"));
        assert!(command_matches("ls *", "ls -la"));
        assert!(!command_matches("ls *", "lsof"));
        assert!(command_matches("npm run test:*", "npm run test"));
        assert!(command_matches("npm run test:*", "npm run test -- --watch"));
        assert!(!command_matches("npm run test:*", "npm run testing"));
        assert!(command_matches("git * --dry-run", "git push origin --dry-run"));
        assert!(command_matches("cargo test", "cargo test"));
        assert!(!command_matches("cargo test", "cargo test --all"));
        assert!(command_matches("*", "anything at all"));
    }

    #[test]
    fn stems_skip_options() {
        assert_eq!(command_stem("git push origin main"), "git push");
        assert_eq!(command_stem("ls -la"), "ls");
        assert_eq!(command_stem("pwd"), "pwd");
    }
}
