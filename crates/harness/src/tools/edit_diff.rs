//! `edit`'s text work: line endings, exact and fuzzy matching, applying replacements, and the
//! diff and patch of the result. Ported from upstream `src/tools/edit-diff.ts`; offsets are
//! byte offsets.

use std::borrow::Cow;

use similar::{ChangeTag, DiffTag, TextDiff};
use unicode_normalization::UnicodeNormalization;

/// Lines of context in the diff and the patch.
const CONTEXT_LINES: usize = 4;

/// One replacement as the model sends it.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Edit {
    pub(super) old_text: String,
    pub(super) new_text: String,
}

/// `\r\n` when the first `\r\n` comes before (at) the first `\n`.
pub(super) fn detect_line_ending(content: &str) -> &'static str {
    match (content.find("\r\n"), content.find('\n')) {
        (Some(crlf), Some(lf)) if crlf < lf => "\r\n",
        _ => "\n",
    }
}

/// `\r\n` and lone `\r` as `\n`.
pub(super) fn normalize_to_lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

pub(super) fn restore_line_endings(text: &str, ending: &str) -> String {
    if ending == "\r\n" {
        text.replace('\n', "\r\n")
    } else {
        text.to_owned()
    }
}

/// The text without a leading byte-order mark, and whether it had one.
pub(super) fn strip_bom(content: &str) -> (bool, &str) {
    match content.strip_prefix('\u{feff}') {
        Some(text) => (true, text),
        None => (false, content),
    }
}

/// Whitespace as JavaScript's `trimEnd` sees it.
fn is_js_whitespace(c: char) -> bool {
    (c.is_whitespace() && c != '\u{85}') || c == '\u{feff}'
}

/// The text for fuzzy matching: NFKC, trailing whitespace stripped per line, smart quotes
/// as ASCII quotes, Unicode dashes as `-`, special spaces as ` `.
pub(super) fn normalize_for_fuzzy_match(text: &str) -> String {
    let nfkc: String = text.nfkc().collect();
    let trimmed: Vec<&str> = nfkc
        .split('\n')
        .map(|line| line.trim_end_matches(is_js_whitespace))
        .collect();
    trimmed
        .join("\n")
        .chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{201a}' | '\u{201b}' => '\'',
            '\u{201c}' | '\u{201d}' | '\u{201e}' | '\u{201f}' => '"',
            '\u{2010}'..='\u{2015}' | '\u{2212}' => '-',
            '\u{a0}' | '\u{2002}'..='\u{200a}' | '\u{202f}' | '\u{205f}' | '\u{3000}' => ' ',
            c => c,
        })
        .collect()
}

/// Where an edit's old text was found.
struct Found {
    index: usize,
    length: usize,
    /// The offsets are in the fuzzy normalization of the content, not in the content.
    fuzzy: bool,
}

/// `old` in `content`, exactly or else in fuzzy-normalized space.
fn fuzzy_find(content: &str, old: &str) -> Option<Found> {
    if let Some(index) = content.find(old) {
        return Some(Found {
            index,
            length: old.len(),
            fuzzy: false,
        });
    }
    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old = normalize_for_fuzzy_match(old);
    let index = fuzzy_content.find(&fuzzy_old)?;
    Some(Found {
        index,
        length: fuzzy_old.len(),
        fuzzy: true,
    })
}

fn count_occurrences(content: &str, old: &str) -> usize {
    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old = normalize_for_fuzzy_match(old);
    fuzzy_content.matches(fuzzy_old.as_str()).count()
}

/// One located edit.
#[derive(Clone, Debug)]
struct Replacement {
    edit: usize,
    index: usize,
    length: usize,
    new_text: String,
}

/// Applies `replacements` (sorted, disjoint, offsets from `offset`) to `content`, last first.
fn apply_replacements(content: &str, replacements: &[Replacement], offset: usize) -> String {
    let mut result = content.to_owned();
    for replacement in replacements.iter().rev() {
        let start = replacement.index - offset;
        result.replace_range(start..start + replacement.length, &replacement.new_text);
    }
    result
}

/// `[start, end)` of each line of `content`, with its newline.
fn line_spans(content: &str) -> Vec<(usize, usize)> {
    let mut offset = 0;
    content
        .split_inclusive('\n')
        .map(|line| {
            let span = (offset, offset + line.len());
            offset = span.1;
            span
        })
        .collect()
}

/// The lines `[start, end)` a replacement touches.
fn replacement_lines(lines: &[(usize, usize)], replacement: &Replacement) -> Result<(usize, usize), String> {
    const OUTSIDE: &str = "Replacement range is outside the base content.";
    let start = replacement.index;
    let end = replacement.index + replacement.length;
    let first = lines
        .iter()
        .position(|&(from, to)| start >= from && start < to)
        .ok_or(OUTSIDE)?;
    let mut last = first;
    while last < lines.len() && lines[last].1 < end {
        last += 1;
    }
    if last >= lines.len() {
        return Err(OUTSIDE.to_owned());
    }
    Ok((first, last + 1))
}

/// Applies replacements located in `base` (a normalized view of `original` with the same
/// lines) so that only the lines they touch are taken from `base`; every other line keeps
/// its bytes from `original`.
fn apply_preserving_unchanged_lines(
    original: &str,
    base: &str,
    replacements: &[Replacement],
) -> Result<String, String> {
    let original_lines: Vec<&str> = original.split_inclusive('\n').collect();
    let base_lines = line_spans(base);
    if original_lines.len() != base_lines.len() {
        return Err("Cannot preserve unchanged lines because the base content has a different line count.".to_owned());
    }
    let mut groups: Vec<(usize, usize, Vec<Replacement>)> = Vec::new();
    let mut sorted = replacements.to_vec();
    sorted.sort_by_key(|replacement| replacement.index);
    for replacement in sorted {
        let (start, end) = replacement_lines(&base_lines, &replacement)?;
        if let Some(group) = groups.last_mut().filter(|group| start < group.1) {
            group.1 = group.1.max(end);
            group.2.push(replacement);
            continue;
        }
        groups.push((start, end, vec![replacement]));
    }
    let mut result = String::with_capacity(original.len());
    let mut next = 0;
    for (start, end, replacements) in &groups {
        result.extend(original_lines[next..*start].iter().copied());
        let from = base_lines[*start].0;
        let to = base_lines[end - 1].1;
        result.push_str(&apply_replacements(&base[from..to], replacements, from));
        next = *end;
    }
    result.extend(original_lines[next..].iter().copied());
    Ok(result)
}

/// Applies `edits` to LF-normalized `content`, every one matched against the same original.
/// When any needs fuzzy matching, all are located in fuzzy-normalized space and only the
/// lines they touch are taken from it. Returns the new content; the error is the message
/// for the model.
pub(super) fn apply_edits(content: &str, edits: &[Edit], path: &str) -> Result<String, String> {
    let several = edits.len() > 1;
    let edits: Vec<Edit> = edits
        .iter()
        .map(|edit| Edit {
            old_text: normalize_to_lf(&edit.old_text),
            new_text: normalize_to_lf(&edit.new_text),
        })
        .collect();
    for (index, edit) in edits.iter().enumerate() {
        if edit.old_text.is_empty() {
            return Err(if several {
                format!("edits[{index}].oldText must not be empty in {path}.")
            } else {
                format!("oldText must not be empty in {path}.")
            });
        }
    }
    let fuzzy = edits
        .iter()
        .any(|edit| fuzzy_find(content, &edit.old_text).is_some_and(|found| found.fuzzy));
    let base: Cow<'_, str> = if fuzzy {
        Cow::Owned(normalize_for_fuzzy_match(content))
    } else {
        Cow::Borrowed(content)
    };

    let mut replacements = Vec::with_capacity(edits.len());
    for (index, edit) in edits.iter().enumerate() {
        let Some(found) = fuzzy_find(&base, &edit.old_text) else {
            return Err(if several {
                format!(
                    "Could not find edits[{index}] in {path}. The oldText must match exactly including all whitespace and newlines."
                )
            } else {
                format!(
                    "Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
                )
            });
        };
        let occurrences = count_occurrences(&base, &edit.old_text);
        if occurrences > 1 {
            return Err(if several {
                format!(
                    "Found {occurrences} occurrences of edits[{index}] in {path}. Each oldText must be unique. Please provide more context to make it unique."
                )
            } else {
                format!(
                    "Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
                )
            });
        }
        replacements.push(Replacement {
            edit: index,
            index: found.index,
            length: found.length,
            new_text: edit.new_text.clone(),
        });
    }
    replacements.sort_by_key(|replacement| replacement.index);
    for pair in replacements.windows(2) {
        if pair[0].index + pair[0].length > pair[1].index {
            return Err(format!(
                "edits[{}] and edits[{}] overlap in {path}. Merge them into one edit or target disjoint regions.",
                pair[0].edit, pair[1].edit
            ));
        }
    }
    let new_content = if fuzzy {
        apply_preserving_unchanged_lines(content, &base, &replacements)?
    } else {
        apply_replacements(&base, &replacements, 0)
    };
    if new_content == content {
        return Err(if several {
            format!("No changes made to {path}. The replacements produced identical content.")
        } else {
            format!(
                "No changes made to {path}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
            )
        });
    }
    Ok(new_content)
}

/// A unified patch of `path` with file headers only.
pub(super) fn unified_patch(path: &str, old: &str, new: &str) -> String {
    TextDiff::from_lines(old, new)
        .unified_diff()
        .context_radius(CONTEXT_LINES)
        .header(path, path)
        .to_string()
}

/// A run of lines the diff keeps, removes or adds.
struct Part<'a> {
    tag: ChangeTag,
    lines: Vec<&'a str>,
}

fn parts<'a>(old: &'a str, new: &'a str) -> Vec<Part<'a>> {
    let old_lines: Vec<&str> = old.split_inclusive('\n').collect();
    let new_lines: Vec<&str> = new.split_inclusive('\n').collect();
    let mut parts: Vec<Part<'a>> = Vec::new();
    let mut push = |tag: ChangeTag, lines: &[&'a str]| {
        let lines = lines.iter().map(|line| line.strip_suffix('\n').unwrap_or(line));
        match parts.last_mut() {
            Some(part) if part.tag == tag => part.lines.extend(lines),
            _ => parts.push(Part {
                tag,
                lines: lines.collect(),
            }),
        }
    };
    for op in TextDiff::from_lines(old, new).ops() {
        let (tag, old_range, new_range) = op.as_tag_tuple();
        match tag {
            DiffTag::Equal => push(ChangeTag::Equal, &old_lines[old_range]),
            DiffTag::Delete => push(ChangeTag::Delete, &old_lines[old_range]),
            DiffTag::Insert => push(ChangeTag::Insert, &new_lines[new_range]),
            DiffTag::Replace => {
                push(ChangeTag::Delete, &old_lines[old_range]);
                push(ChangeTag::Insert, &new_lines[new_range]);
            }
        }
    }
    parts
}

/// Context lines, numbered by the old content; both counters advance.
fn context(lines: &[&str], width: usize, lines_at: &mut (usize, usize), output: &mut Vec<String>) {
    for line in lines {
        output.push(numbered(' ', lines_at.0, width, line));
        lines_at.0 += 1;
        lines_at.1 += 1;
    }
}

fn numbered(sign: char, number: usize, width: usize, line: &str) -> String {
    format!("{sign}{number:>width$} {line}")
}

/// The diff for people: changed lines with `+`/`-` and their line number, up to four lines
/// of context around them, `...` for skipped lines; and the first changed line of the new
/// content.
pub(super) fn diff_string(old: &str, new: &str) -> (String, Option<usize>) {
    let parts = parts(old, new);
    let width = old.split('\n').count().max(new.split('\n').count()).to_string().len();
    let skipped = format!(" {:width$} ...", "");
    let mut output = Vec::new();
    // The next line number of the old and of the new content.
    let mut at = (1, 1);
    let mut last_was_change = false;
    let mut first_changed = None;
    for (index, part) in parts.iter().enumerate() {
        let lines = &part.lines;
        if part.tag != ChangeTag::Equal {
            first_changed.get_or_insert(at.1);
            for line in lines {
                if part.tag == ChangeTag::Insert {
                    output.push(numbered('+', at.1, width, line));
                    at.1 += 1;
                } else {
                    output.push(numbered('-', at.0, width, line));
                    at.0 += 1;
                }
            }
            last_was_change = true;
            continue;
        }
        let next_is_change = parts.get(index + 1).is_some_and(|next| next.tag != ChangeTag::Equal);
        let skip = |count: usize, at: &mut (usize, usize), output: &mut Vec<String>| {
            if count > 0 {
                output.push(skipped.clone());
                at.0 += count;
                at.1 += count;
            }
        };
        match (last_was_change, next_is_change) {
            (true, true) if lines.len() <= CONTEXT_LINES * 2 => context(lines, width, &mut at, &mut output),
            (true, true) => {
                context(&lines[..CONTEXT_LINES], width, &mut at, &mut output);
                skip(lines.len() - CONTEXT_LINES * 2, &mut at, &mut output);
                context(&lines[lines.len() - CONTEXT_LINES..], width, &mut at, &mut output);
            }
            (true, false) => {
                let shown = lines.len().min(CONTEXT_LINES);
                context(&lines[..shown], width, &mut at, &mut output);
                skip(lines.len() - shown, &mut at, &mut output);
            }
            (false, true) => {
                let hidden = lines.len().saturating_sub(CONTEXT_LINES);
                skip(hidden, &mut at, &mut output);
                context(&lines[hidden..], width, &mut at, &mut output);
            }
            (false, false) => {
                at.0 += lines.len();
                at.1 += lines.len();
            }
        }
        last_was_change = false;
    }
    (output.join("\n"), first_changed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edit(old: &str, new: &str) -> Edit {
        Edit {
            old_text: old.to_owned(),
            new_text: new.to_owned(),
        }
    }

    #[test]
    fn line_endings_and_marks() {
        assert_eq!(detect_line_ending("a\r\nb\n"), "\r\n");
        assert_eq!(detect_line_ending("a\nb\r\n"), "\n");
        assert_eq!(detect_line_ending("a"), "\n");
        assert_eq!(normalize_to_lf("a\r\nb\rc"), "a\nb\nc");
        assert_eq!(restore_line_endings("a\nb", "\r\n"), "a\r\nb");
        assert_eq!(strip_bom("\u{feff}x"), (true, "x"));
    }

    #[test]
    fn fuzzy_normalization() {
        assert_eq!(
            normalize_for_fuzzy_match("\u{201c}a\u{201d} \u{2014} it\u{2019}s  \nb\u{a0}c"),
            "\"a\" - it's\nb c"
        );
        // NFKC folds compatibility forms.
        assert_eq!(normalize_for_fuzzy_match("\u{ff21}"), "A");
    }

    #[test]
    fn exact_edits_apply_against_the_original() {
        let content = "one\ntwo\nthree\n";
        let edits = [edit("one", "1"), edit("three", "3")];
        assert_eq!(apply_edits(content, &edits, "f").unwrap(), "1\ntwo\n3\n");
    }

    #[test]
    fn fuzzy_edits_keep_untouched_lines() {
        // The first line has trailing spaces and a smart quote; the edit targets line two
        // through fuzzy matching (its quote differs), and line one keeps its bytes.
        let content = "it\u{2019}s  \nsay \u{201c}hi\u{201d}\n";
        let edits = [edit("say \"hi\"", "say \"bye\"")];
        assert_eq!(
            apply_edits(content, &edits, "f").unwrap(),
            "it\u{2019}s  \nsay \"bye\"\n"
        );
    }

    #[test]
    fn edit_errors_name_the_problem() {
        let content = "a\nb\na\n";
        assert_eq!(
            apply_edits(content, &[edit("x", "y")], "f").unwrap_err(),
            "Could not find the exact text in f. The old text must match exactly including all whitespace and newlines."
        );
        assert_eq!(
            apply_edits(content, &[edit("a", "y")], "f").unwrap_err(),
            "Found 2 occurrences of the text in f. The text must be unique. Please provide more context to make it unique."
        );
        assert_eq!(
            apply_edits(content, &[edit("a\nb", "y"), edit("b\na", "z")], "f").unwrap_err(),
            "edits[0] and edits[1] overlap in f. Merge them into one edit or target disjoint regions."
        );
        assert_eq!(
            apply_edits(content, &[edit("", "y")], "f").unwrap_err(),
            "oldText must not be empty in f."
        );
        assert!(
            apply_edits(content, &[edit("b", "b")], "f")
                .unwrap_err()
                .starts_with("No changes made to f.")
        );
    }

    #[test]
    fn diffs_number_changed_lines_with_context() {
        let old = (1..=20).map(|n| format!("line {n}\n")).collect::<Vec<_>>().concat();
        let new = old.replace("line 10\n", "ten\n");
        let (diff, first) = diff_string(&old, &new);
        assert_eq!(first, Some(10));
        let lines: Vec<&str> = diff.lines().collect();
        assert_eq!(lines[0], "    ...");
        assert_eq!(lines[1], "  6 line 6");
        assert!(lines.contains(&"-10 line 10"));
        assert!(lines.contains(&"+10 ten"));
        assert_eq!(lines.last(), Some(&"    ..."));
        let patch = unified_patch("f", &old, &new);
        assert!(patch.starts_with("--- f\n+++ f\n@@ -6,9 +6,9 @@\n"), "{patch}");
    }
}
