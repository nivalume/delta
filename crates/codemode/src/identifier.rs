//! The identifier a script uses for a tool.

/// Characters that are not valid in a JavaScript identifier become `_`: `mcp__docs__search`
/// stays as is, `my-tool` becomes `my_tool`, an empty name `_`.
pub fn to_codemode_identifier(name: &str) -> String {
    let mut identifier = String::with_capacity(name.len());
    for c in name.chars() {
        let valid = c.is_ascii_alphabetic() || c == '_' || c == '$' || (!identifier.is_empty() && c.is_ascii_digit());
        identifier.push(if valid { c } else { '_' });
    }
    if identifier.is_empty() {
        identifier.push('_');
    }
    identifier
}

/// Whether `name` is a plain JavaScript identifier (ASCII).
pub(crate) fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_' || c == '$')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_become_identifiers() {
        assert_eq!(to_codemode_identifier("mcp__docs__search"), "mcp__docs__search");
        assert_eq!(to_codemode_identifier("my-tool"), "my_tool");
        assert_eq!(to_codemode_identifier("1st"), "_st");
        assert_eq!(to_codemode_identifier("a.b"), "a_b");
        assert_eq!(to_codemode_identifier("é"), "_");
        assert_eq!(to_codemode_identifier(""), "_");
        assert!(is_identifier("$x1"));
        assert!(!is_identifier("1x"));
        assert!(!is_identifier(""));
    }
}
