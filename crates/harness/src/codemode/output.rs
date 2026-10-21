//! A script's result as the model reads it: a `Script completed` / `Script failed` header with
//! the wall time, the script's text items (`==> text N/M <==` when there are several), its
//! `console` lines in one `<console_output>` block, the returned value, and on failure the error
//! with the calls already made. Text past the token budget is cut to its start and end, with
//! the complete text in a file; images are saved to files too, with their path next to them,
//! because a script has no other way to hand their bytes on. Port of upstream
//! `codemode/output.ts`.

use std::collections::HashMap;
use std::fmt::Write as _;

use base64::Engine as _;
use delta_ai::types::{ImageContent, UserBlock};
use delta_codemode::{CodemodeResult, ErrorKind, OutputItem};
use futures::future::BoxFuture;
use serde_json::Value;

/// Lines of the result besides the script's text: the header, markers, notes.
const LINE_MARGIN: usize = 100;
/// Default token budget for script output.
pub const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 10_000;
/// Characters per token when estimating.
const CHARS_PER_TOKEN: usize = 4;

/// Writes a file of the result under a fixed name: its path, or why it could not be written.
pub(super) trait OutputFile: Send {
    fn write<'a>(&'a mut self, name: &'a str, data: &'a [u8]) -> BoxFuture<'a, Result<String, String>>;
}

/// The tool's output limits, which the Harness enforces by cutting: results stay inside them.
#[derive(Clone, Copy, Debug)]
pub(super) struct Limits {
    pub(super) max_bytes: usize,
    pub(super) max_lines: usize,
}

/// One tool call as the error summary names it.
pub(super) struct CallSummary<'a> {
    pub(super) name: &'a str,
    pub(super) status: &'a str,
}

pub(super) struct FormatOptions<'a> {
    /// Images `models.generateImages()` returned; when the script showed none, the result
    /// says so.
    pub(super) generated_images: usize,
    pub(super) wall_ms: i64,
    pub(super) max_output_tokens: u64,
    pub(super) calls: Vec<CallSummary<'a>>,
    pub(super) limits: Limits,
}

/// The formatted content, and the path of the complete text when it was cut.
pub(super) struct Formatted {
    pub(super) content: Vec<UserBlock>,
    pub(super) full_output_path: Option<String>,
}

fn image_extension(mime_type: &str) -> &'static str {
    match mime_type {
        "image/png" => ".png",
        "image/jpeg" => ".jpg",
        "image/gif" => ".gif",
        "image/webp" => ".webp",
        _ => ".bin",
    }
}

/// Like the script's `text()`: strings as is, other values as compact JSON.
fn value_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Lays out the script's output so the model can tell items apart: with more than one text
/// item, each starts with a `==> text N/M <==` line; `console` lines follow all other output in
/// one `<console_output>` block.
fn format_items(output: Vec<OutputItem>) -> Vec<UserBlock> {
    let total = output
        .iter()
        .filter(|item| matches!(item, OutputItem::Text { console: false, .. }))
        .count();
    let mut items = Vec::new();
    let mut console = Vec::new();
    let mut index = 0;
    for item in output {
        match item {
            OutputItem::Image { data, mime_type } => items.push(UserBlock::Image(ImageContent { data, mime_type })),
            OutputItem::Text { text, console: true } => console.push(text),
            OutputItem::Text { text, console: false } => {
                index += 1;
                let text = if total > 1 {
                    format!("==> text {index}/{total} <==\n{text}")
                } else {
                    text
                };
                items.push(UserBlock::text(text));
            }
        }
    }
    if !console.is_empty() {
        items.push(UserBlock::text(format!(
            "<console_output>\n{}\n</console_output>",
            console.join("\n")
        )));
    }
    items
}

/// Joins adjacent text items into one, each part starting on its own line.
fn join_adjacent_text(items: Vec<UserBlock>) -> Vec<UserBlock> {
    let mut joined = Vec::new();
    let mut text: Option<String> = None;
    for item in items {
        match item {
            UserBlock::Text(part) => {
                let current = text.get_or_insert_with(String::new);
                if !current.is_empty() && !current.ends_with('\n') {
                    current.push('\n');
                }
                current.push_str(&part.text);
            }
            image @ UserBlock::Image(_) => {
                if let Some(text) = text.take() {
                    joined.push(UserBlock::text(text));
                }
                joined.push(image);
            }
        }
    }
    if let Some(text) = text {
        joined.push(UserBlock::text(text));
    }
    joined
}

/// The byte offset after the first `count` characters of `text`.
fn char_offset(text: &str, count: usize) -> usize {
    text.char_indices().nth(count).map_or(text.len(), |(offset, _)| offset)
}

/// Applies the token budget: when the combined text exceeds it, the text items become one item
/// that keeps the start and end of the text, and images follow it. The complete text is written
/// to `output.txt`.
async fn truncate(
    items: Vec<UserBlock>,
    max_tokens: u64,
    limits: Limits,
    write: &mut dyn OutputFile,
) -> (Vec<UserBlock>, Option<String>) {
    let texts: Vec<&str> = items
        .iter()
        .filter_map(|item| match item {
            UserBlock::Text(text) => Some(text.text.as_str()),
            UserBlock::Image(_) => None,
        })
        .collect();
    let combined = texts.join("\n");
    let chars = combined.chars().count();
    // Within the limits too: a character is at most 4 UTF-8 bytes, and the rest of the result
    // is a few short lines.
    let tokens = usize::try_from(max_tokens).unwrap_or(usize::MAX);
    let budget = tokens.saturating_mul(CHARS_PER_TOKEN).min(limits.max_bytes / 4);
    let line_budget = limits.max_lines.saturating_sub(LINE_MARGIN) / 2;
    let lines = combined.split('\n').count();
    if texts.is_empty() || (chars <= budget && lines <= 2 * line_budget) {
        return (items, None);
    }
    let head_chars = budget / 2;
    let tail_chars = budget - head_chars;
    let head: Vec<&str> = combined[..char_offset(&combined, head_chars)]
        .split('\n')
        .take(line_budget)
        .collect();
    let head = head.join("\n");
    let tail = if tail_chars > 0 {
        let start = char_offset(&combined, chars.saturating_sub(tail_chars));
        let tail: Vec<&str> = combined[start..].split('\n').collect();
        tail[tail.len().saturating_sub(line_budget)..].join("\n")
    } else {
        String::new()
    };
    let removed = chars - head.chars().count() - tail.chars().count();
    let mut text = format!(
        "Warning: truncated output (original token count: {})\nTotal output lines: {lines}\n\n{head}…{} tokens truncated…{tail}",
        chars.div_ceil(CHARS_PER_TOKEN),
        removed.div_ceil(CHARS_PER_TOKEN),
    );
    let spilled = write.write("output.txt", combined.as_bytes()).await;
    match &spilled {
        Ok(path) => {
            let _ = write!(text, "\n\n[Full output: {path} (read with offset/limit)]");
        }
        Err(error) => {
            let _ = write!(text, "\n\n[Could not save the full output: {error}]");
        }
    }
    let mut truncated = vec![UserBlock::text(text)];
    truncated.extend(items.into_iter().filter(|item| matches!(item, UserBlock::Image(_))));
    (truncated, spilled.ok())
}

/// Saves each image to a file and puts a text item with its path before it; an image shown
/// twice is saved once.
async fn save_images(items: Vec<UserBlock>, write: &mut dyn OutputFile) -> Vec<UserBlock> {
    let mut labels: HashMap<String, String> = HashMap::new();
    let mut saved = Vec::new();
    for item in items {
        let UserBlock::Image(image) = &item else {
            saved.push(item);
            continue;
        };
        let label = match labels.get(&image.data) {
            Some(label) => label.clone(),
            None => {
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(&image.data)
                    .unwrap_or_default();
                let kind = format!("{}, {} bytes", image.mime_type, bytes.len());
                let name = format!("image-{}{}", labels.len() + 1, image_extension(&image.mime_type));
                // A failed write must not discard the result of a script whose calls already
                // ran: it becomes part of the label.
                let label = match write.write(&name, &bytes).await {
                    Ok(path) => format!("[Image saved to {path} ({kind})]"),
                    Err(error) => format!("[Image ({kind}) could not be saved: {error}]"),
                };
                labels.insert(image.data.clone(), label.clone());
                label
            }
        };
        saved.push(UserBlock::text(label));
        saved.push(item);
    }
    saved
}

fn format_error(result: &CodemodeResult, calls: &[CallSummary<'_>]) -> String {
    let Some(error) = &result.error else {
        return String::new();
    };
    let head = match error.kind {
        ErrorKind::Script => error
            .stack
            .clone()
            .unwrap_or_else(|| format!("{}: {}", error.name.as_deref().unwrap_or("Error"), error.message)),
        ErrorKind::Timeout => format!("Script timed out: {}", error.message),
        ErrorKind::Aborted => format!("Script aborted: {}", error.message),
        ErrorKind::Sandbox => format!("Script sandbox failed: {}", error.message),
    };
    let summary = if calls.is_empty() {
        "No tool calls were made.".to_owned()
    } else {
        let calls: Vec<String> = calls
            .iter()
            .map(|call| format!("{} ({})", call.name, call.status))
            .collect();
        format!(
            "Tool calls made before the failure (they are not undone): {}",
            calls.join(", ")
        )
    };
    format!("{head}\n\n{summary}")
}

/// The tool result's content for a script's result.
pub(super) async fn format_result(
    result: &CodemodeResult,
    options: FormatOptions<'_>,
    write: &mut dyn OutputFile,
) -> Formatted {
    let mut output = result.output.clone();
    if result.is_ok()
        && let Some(value) = &result.value
    {
        output.push(OutputItem::Text {
            text: value_text(value),
            console: false,
        });
    }
    let mut items = format_items(output);
    if !result.is_ok() {
        items.push(UserBlock::text(format!(
            "Script error:\n{}",
            format_error(result, &options.calls)
        )));
    }
    let generated = options.generated_images;
    if generated > 0 && !items.iter().any(|item| matches!(item, UserBlock::Image(_))) {
        let plural = if generated == 1 { "" } else { "s" };
        items.push(UserBlock::text(format!(
            "Note: models.generateImages() returned {generated} image{plural} that the script did not show. Show each image block of result.output with image(block)."
        )));
    }
    let (items, full_output_path) = truncate(
        join_adjacent_text(items),
        options.max_output_tokens,
        options.limits,
        write,
    )
    .await;
    // After truncation, which joins the text items and moves images after them, so each path
    // stays next to its image.
    let content = join_adjacent_text(save_images(items, write).await);
    let status = if result.is_ok() {
        "Script completed"
    } else {
        "Script failed"
    };
    #[allow(clippy::cast_precision_loss)] // Wall times are far below 2^52 ms.
    let seconds = options.wall_ms as f64 / 1000.0;
    let header = format!("{status}\nWall time {seconds:.1} seconds\nOutput:\n");
    let mut blocks = vec![UserBlock::text(header)];
    blocks.extend(content);
    Formatted {
        content: blocks,
        full_output_path,
    }
}

#[cfg(test)]
mod tests {
    use delta_codemode::CodemodeError;

    use super::*;

    /// Records writes; fails names in `failing`.
    #[derive(Default)]
    struct Files {
        written: Vec<(String, Vec<u8>)>,
        failing: Vec<&'static str>,
    }

    impl OutputFile for Files {
        fn write<'a>(&'a mut self, name: &'a str, data: &'a [u8]) -> BoxFuture<'a, Result<String, String>> {
            Box::pin(async move {
                if self.failing.contains(&name) {
                    return Err("disk full".to_owned());
                }
                self.written.push((name.to_owned(), data.to_vec()));
                Ok(format!("/tmp/run/{name}"))
            })
        }
    }

    const LIMITS: Limits = Limits {
        max_bytes: 4 * 1024 * 1024,
        max_lines: 100_000,
    };

    fn options(max_output_tokens: u64) -> FormatOptions<'static> {
        FormatOptions {
            generated_images: 0,
            wall_ms: 1250,
            max_output_tokens,
            calls: Vec::new(),
            limits: LIMITS,
        }
    }

    fn texts(content: &[UserBlock]) -> Vec<String> {
        content
            .iter()
            .map(|block| match block {
                UserBlock::Text(text) => text.text.clone(),
                UserBlock::Image(image) => format!("<{}>", image.mime_type),
            })
            .collect()
    }

    fn text(text: &str, console: bool) -> OutputItem {
        OutputItem::Text {
            text: text.to_owned(),
            console,
        }
    }

    #[tokio::test]
    async fn items_are_numbered_and_console_lines_follow() {
        let result = CodemodeResult {
            output: vec![text("one", false), text("log", true), text("two", false)],
            value: Some(serde_json::json!({ "n": 1 })),
            ..CodemodeResult::default()
        };
        let formatted = format_result(&result, options(10_000), &mut Files::default()).await;
        assert_eq!(
            texts(&formatted.content),
            [
                "Script completed\nWall time 1.2 seconds\nOutput:\n",
                "==> text 1/3 <==\none\n==> text 2/3 <==\ntwo\n==> text 3/3 <==\n{\"n\":1}\n<console_output>\nlog\n</console_output>",
            ]
        );
        assert!(formatted.full_output_path.is_none());
    }

    #[tokio::test]
    async fn failures_name_the_error_and_the_calls() {
        let result = CodemodeResult {
            error: Some(CodemodeError {
                stack: Some("Error: boom\n    at <anonymous> (codemode.js:2)".to_owned()),
                ..CodemodeError::new(ErrorKind::Script, "boom")
            }),
            ..CodemodeResult::default()
        };
        let mut options = options(10_000);
        options.calls = vec![CallSummary {
            name: "read",
            status: "ok",
        }];
        let formatted = format_result(&result, options, &mut Files::default()).await;
        assert_eq!(
            texts(&formatted.content)[1],
            "Script error:\nError: boom\n    at <anonymous> (codemode.js:2)\n\nTool calls made before the failure (they are not undone): read (ok)"
        );
        assert!(texts(&formatted.content)[0].starts_with("Script failed\n"));
    }

    #[tokio::test]
    async fn long_output_keeps_its_ends_and_spills_to_a_file() {
        let long = (0..400).fold(String::new(), |mut long, line| {
            let _ = writeln!(long, "line {line}");
            long
        });
        let result = CodemodeResult {
            output: vec![text(&long, false)],
            ..CodemodeResult::default()
        };
        let mut files = Files::default();
        let formatted = format_result(&result, options(100), &mut files).await;
        let body = &texts(&formatted.content)[1];
        assert!(
            body.starts_with("Warning: truncated output (original token count: "),
            "{body}"
        );
        assert!(body.contains("line 0\n") && body.contains("line 399"), "{body}");
        assert!(body.contains("tokens truncated…"), "{body}");
        assert!(body.ends_with("[Full output: /tmp/run/output.txt (read with offset/limit)]"));
        assert_eq!(formatted.full_output_path.as_deref(), Some("/tmp/run/output.txt"));
        assert_eq!(files.written[0].1, long.as_bytes());
    }

    #[tokio::test]
    async fn images_are_saved_once_and_labelled() {
        let png = base64::engine::general_purpose::STANDARD.encode([0x89, b'P', b'N', b'G']);
        let image = OutputItem::Image {
            data: png,
            mime_type: "image/png".to_owned(),
        };
        let gif = OutputItem::Image {
            data: "R0lG".to_owned(),
            mime_type: "image/gif".to_owned(),
        };
        let result = CodemodeResult {
            output: vec![image.clone(), text("between", false), image, gif],
            ..CodemodeResult::default()
        };
        let mut files = Files {
            failing: vec!["image-2.gif"],
            ..Files::default()
        };
        let formatted = format_result(&result, options(10_000), &mut files).await;
        assert_eq!(
            texts(&formatted.content)[1..],
            [
                "[Image saved to /tmp/run/image-1.png (image/png, 4 bytes)]",
                "<image/png>",
                "between\n[Image saved to /tmp/run/image-1.png (image/png, 4 bytes)]",
                "<image/png>",
                "[Image (image/gif, 3 bytes) could not be saved: disk full]",
                "<image/gif>",
            ]
        );
        assert_eq!(files.written.len(), 1);
    }
}
