//! The paths a model writes, made into paths of the environment.

use std::collections::HashSet;

use delta_chord::context::Context;
use unicode_normalization::UnicodeNormalization;

use crate::env::{ExecutionEnv, FileResult};

const NARROW_NO_BREAK_SPACE: char = '\u{202f}';

fn is_unicode_space(c: char) -> bool {
    matches!(
        c,
        '\u{a0}' | '\u{2000}'..='\u{200a}' | '\u{202f}' | '\u{205f}' | '\u{3000}'
    )
}

/// Unicode spaces become ASCII spaces; a leading `@` (a model's file mention) is dropped.
pub fn normalize_tool_path(path: &str) -> String {
    let normalized: String = path
        .chars()
        .map(|c| if is_unicode_space(c) { ' ' } else { c })
        .collect();
    match normalized.strip_prefix('@') {
        Some(rest) => rest.to_owned(),
        None => normalized,
    }
}

/// The normalized `path` as an absolute path of `env`.
pub fn resolve_tool_path(env: &dyn ExecutionEnv, path: &str) -> FileResult<String> {
    env.absolute_path(&normalize_tool_path(path))
}

/// ` AM.` and ` PM.` (any case) with a narrow no-break space before them, as macOS names
/// screenshots.
fn narrow_meridiem(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    let mut rest = path;
    while let Some(index) = rest.find(' ') {
        out.push_str(&rest[..index]);
        let after = &rest[index + 1..];
        let bytes = after.as_bytes();
        let meridiem = bytes.len() >= 3
            && matches!(bytes[0], b'A' | b'a' | b'P' | b'p')
            && matches!(bytes[1], b'M' | b'm')
            && bytes[2] == b'.';
        out.push(if meridiem { NARROW_NO_BREAK_SPACE } else { ' ' });
        rest = after;
    }
    out.push_str(rest);
    out
}

/// [`resolve_tool_path`], or the first variant that exists: with a narrow no-break space
/// before ` AM.`/` PM.`, in NFD, with `'` as `’`, or NFD with `’`. Without one, the plain
/// resolution.
pub async fn resolve_read_path(env: &dyn ExecutionEnv, path: &str, ctx: &Context) -> FileResult<String> {
    let resolved = resolve_tool_path(env, path)?;
    let nfd: String = resolved.nfd().collect();
    let variants = [
        resolved.clone(),
        narrow_meridiem(&resolved),
        nfd.clone(),
        resolved.replace('\'', "\u{2019}"),
        nfd.replace('\'', "\u{2019}"),
    ];
    let mut tried = HashSet::new();
    for variant in variants {
        if tried.insert(variant.clone()) && env.exists(&variant, ctx).await? {
            return Ok(variant);
        }
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_paths_are_normalized() {
        assert_eq!(normalize_tool_path("@src/main.rs"), "src/main.rs");
        assert_eq!(normalize_tool_path("a\u{a0}b\u{3000}c"), "a b c");
        assert_eq!(normalize_tool_path("a@b"), "a@b");
    }

    #[test]
    fn meridiems_take_a_narrow_space() {
        assert_eq!(
            narrow_meridiem("Shot 10.00.00 pm.png and AM. x"),
            "Shot 10.00.00\u{202f}pm.png and\u{202f}AM. x"
        );
        assert_eq!(narrow_meridiem("a b"), "a b");
    }
}
