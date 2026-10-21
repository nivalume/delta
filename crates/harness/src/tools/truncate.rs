//! Head truncation of a text known whole or by a prefix and its totals, for `read`. Streamed
//! output is bounded by `delta_chord::output::OutputBuffer` instead.

use delta_chord::output::OutputLimits;
use serde::{Deserialize, Serialize};

/// Lines `read` shows at most, and the line limit of the default output window.
pub const DEFAULT_MAX_LINES: usize = 2000;
/// Bytes `read` shows at most, and the byte limit of the default output window.
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;

/// Which limit cut a text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Limit {
    Lines,
    Bytes,
}

/// How a text was cut. Lines are counted ignoring a trailing newline; empty text has none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Truncation {
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncated_by: Option<Limit>,
    pub total_lines: usize,
    pub total_bytes: usize,
    /// Complete lines kept.
    pub output_lines: usize,
    pub output_bytes: usize,
    /// Always `false` for head truncation, which keeps whole lines only.
    pub last_line_partial: bool,
    /// The first line alone is longer than `max_bytes`; nothing was kept.
    pub first_line_exceeds_limit: bool,
    pub max_lines: usize,
    pub max_bytes: usize,
}

/// The size of a whole text, for [`truncate_head_of`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    /// Lines, counted as [`truncate_head`] counts them.
    pub lines: usize,
    /// UTF-8 bytes.
    pub bytes: usize,
}

/// The lines of `text` for counting: none for empty text, no empty line after a trailing
/// newline.
fn counted_lines(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut lines: Vec<&str> = text.split('\n').collect();
    if text.ends_with('\n') {
        lines.pop();
    }
    lines
}

/// The whole lines at the start of `text` within both limits of `limits` (its `retain` is
/// ignored). A first line longer than `max_bytes` keeps nothing and sets
/// `first_line_exceeds_limit`.
pub fn truncate_head<'a>(text: &'a str, limits: &OutputLimits) -> (&'a str, Truncation) {
    let totals = Totals {
        lines: counted_lines(text).len(),
        bytes: text.len(),
    };
    truncate_head_of(text, totals, limits)
}

/// [`truncate_head`] of a text known by `prefix` and its `totals`. The prefix must be the
/// whole text, longer than `max_bytes + 1` bytes, or hold at least `max_lines` newlines; the
/// result then equals [`truncate_head`] of the whole text.
pub fn truncate_head_of<'a>(prefix: &'a str, totals: Totals, limits: &OutputLimits) -> (&'a str, Truncation) {
    let (max_lines, max_bytes) = (limits.max_lines, limits.max_bytes);
    let truncation = Truncation {
        truncated: false,
        truncated_by: None,
        total_lines: totals.lines,
        total_bytes: totals.bytes,
        output_lines: totals.lines,
        output_bytes: totals.bytes,
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    };
    if totals.lines <= max_lines && totals.bytes <= max_bytes {
        return (prefix, truncation);
    }
    let lines = counted_lines(prefix);
    if lines.first().map_or(0, |line| line.len()) > max_bytes {
        let truncation = Truncation {
            truncated: true,
            truncated_by: Some(Limit::Bytes),
            output_lines: 0,
            output_bytes: 0,
            first_line_exceeds_limit: true,
            ..truncation
        };
        return ("", truncation);
    }
    let mut kept = 0;
    let mut bytes = 0;
    let mut by_bytes = false;
    for (index, line) in lines.iter().take(max_lines).enumerate() {
        let size = line.len() + usize::from(index > 0);
        if bytes + size > max_bytes {
            by_bytes = true;
            break;
        }
        kept += 1;
        bytes += size;
    }
    // Without a byte break, only omitted lines prove the line limit was reached; otherwise a
    // trailing newline exceeded the bytes.
    let truncated_by = if !by_bytes && kept < totals.lines {
        Limit::Lines
    } else {
        Limit::Bytes
    };
    let truncation = Truncation {
        truncated: true,
        truncated_by: Some(truncated_by),
        output_lines: kept,
        output_bytes: bytes,
        ..truncation
    };
    // The kept lines are the start of the prefix.
    (&prefix[..bytes], truncation)
}

/// `bytes` for people: `512B`, `1.5KB`, `2.0MB`.
pub fn format_size(bytes: u64) -> String {
    const KIB: u64 = 1024;
    #[allow(clippy::cast_precision_loss)]
    let size = bytes as f64;
    if bytes < KIB {
        format!("{bytes}B")
    } else if bytes < KIB * KIB {
        format!("{:.1}KB", size / 1024.0)
    } else {
        format!("{:.1}MB", size / (1024.0 * 1024.0))
    }
}

#[cfg(test)]
mod tests {
    use delta_chord::output::Retain;

    use super::*;

    fn limits(max_lines: usize, max_bytes: usize) -> OutputLimits {
        OutputLimits {
            max_bytes,
            max_lines,
            retain: Retain::Head,
        }
    }

    #[test]
    fn text_within_the_limits_is_kept() {
        let (text, truncation) = truncate_head("a\nb\n", &limits(2, 10));
        assert_eq!(text, "a\nb\n");
        assert!(!truncation.truncated);
        assert_eq!((truncation.total_lines, truncation.total_bytes), (2, 4));
        assert_eq!(truncate_head("", &limits(0, 0)).1.total_lines, 0);
    }

    #[test]
    fn whole_lines_are_kept_until_a_limit() {
        let (text, truncation) = truncate_head("one\ntwo\nthree\n", &limits(2, 100));
        assert_eq!(text, "one\ntwo");
        assert_eq!(truncation.truncated_by, Some(Limit::Lines));
        assert_eq!((truncation.output_lines, truncation.output_bytes), (2, 7));

        let (text, truncation) = truncate_head("one\ntwo\nthree", &limits(10, 8));
        assert_eq!(text, "one\ntwo");
        assert_eq!(truncation.truncated_by, Some(Limit::Bytes));

        // Every line fits, but the trailing newline does not.
        let (text, truncation) = truncate_head("one\ntwo\n", &limits(10, 7));
        assert_eq!(text, "one\ntwo");
        assert_eq!(truncation.truncated_by, Some(Limit::Bytes));
    }

    #[test]
    fn a_first_line_over_the_byte_limit_keeps_nothing() {
        let (text, truncation) = truncate_head("abcdef\nx", &limits(10, 4));
        assert_eq!(text, "");
        assert!(truncation.first_line_exceeds_limit);
        assert_eq!(truncation.truncated_by, Some(Limit::Bytes));
    }

    #[test]
    fn a_long_enough_prefix_gives_the_whole_texts_result() {
        let whole = (0..50)
            .map(|line| format!("line {line}\n"))
            .collect::<Vec<_>>()
            .concat();
        let limits = limits(10, 1000);
        let expected = truncate_head(&whole, &limits);
        let totals = Totals {
            lines: 50,
            bytes: whole.len(),
        };
        let prefix_end = whole.match_indices('\n').nth(9).unwrap().0 + 1;
        assert_eq!(truncate_head_of(&whole[..prefix_end], totals, &limits), expected);
    }

    #[test]
    fn sizes_format_as_upstream() {
        assert_eq!(format_size(512), "512B");
        assert_eq!(format_size(1536), "1.5KB");
        assert_eq!(format_size(50 * 1024), "50.0KB");
        assert_eq!(format_size(2 * 1024 * 1024), "2.0MB");
    }
}
