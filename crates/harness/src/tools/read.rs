//! `read`: a text file's lines, bounded, with how to continue as diagnostics.

use std::sync::Arc;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use delta_ai::model::Model;
use delta_ai::types::{ImageContent, JsonObject, Tool, UserBlock};
use delta_chord::context::Context;
use delta_chord::output::{OutputLimits, Retain, char_end};
use futures::future::BoxFuture;
use serde::Deserialize;

use super::image::{ImageLimits, ImageProcessor, base64_len, detect as detect_image, image_dimensions, is_inline};
use super::paths::resolve_read_path;
use super::truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, Limit, Totals, format_size, truncate_head_of};
use super::{js_number, parse, path_subject, require_env};
use crate::env::{BinaryReader, ExecutionEnv, FileInfo, FileResult, LineScan, OpenOptions, Utf8Decoder};
use crate::tool::{
    HarnessTool, PathAccess, PermissionSubject, Severity, ToolApi, ToolDiagnostic, ToolError, ToolExecutionResult,
};

/// Bytes read per call while reading the head of the selection.
const READ_CHUNK: usize = 64 * 1024;
const BYTE_ORDER_MARK: [u8; 3] = [0xef, 0xbb, 0xbf];
/// JavaScript's largest safe integer, the bound of a line index upstream scans by.
const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;

const LIMITS: OutputLimits = OutputLimits {
    max_bytes: DEFAULT_MAX_BYTES,
    max_lines: DEFAULT_MAX_LINES,
    retain: Retain::Head,
};

/// How [`ReadTool`] reads images.
#[derive(Clone, Debug, Default)]
pub struct ReadOptions {
    /// Resizes and converts images; without one, inline images within the model's limits
    /// pass through as they are.
    pub image_processor: Option<Arc<dyn ImageProcessor>>,
}

/// The `read` tool: text files' lines, or one image. Details of a text read:
/// `{ truncation? }`, the [`Truncation`](super::Truncation) of the shown text.
#[derive(Debug)]
pub struct ReadTool {
    definition: Tool,
    images: Option<Arc<dyn ImageProcessor>>,
}

impl Default for ReadTool {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadTool {
    pub fn new() -> Self {
        Self::with_options(ReadOptions::default())
    }

    pub fn with_options(options: ReadOptions) -> Self {
        let formats = if options.image_processor.is_some() {
            "jpg, png, gif, webp, bmp"
        } else {
            "jpg, png, gif, webp"
        };
        let description = format!(
            "Read the contents of a file. Supports text files and images ({formats}). Images are sent as attachments. For text files, output is truncated to {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
            DEFAULT_MAX_BYTES / 1024
        );
        let parameters = serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to read (relative or absolute)" },
                "offset": { "type": "number", "description": "Line number to start reading from (1-indexed)" },
                "limit": { "type": "number", "description": "Maximum number of lines to read" },
            },
            "required": ["path"],
        });
        Self {
            definition: Tool::new("read", description, parameters),
            images: options.image_processor,
        }
    }
}

#[derive(Deserialize)]
struct ReadArguments {
    path: String,
    #[serde(default)]
    offset: Option<f64>,
    #[serde(default)]
    limit: Option<f64>,
}

impl HarnessTool for ReadTool {
    fn definition(&self) -> &Tool {
        &self.definition
    }

    fn permission_subject(&self, arguments: &JsonObject, env: Option<&dyn ExecutionEnv>) -> Option<PermissionSubject> {
        path_subject(arguments, env, PathAccess::Read)
    }

    fn execute<'a>(
        &'a self,
        arguments: JsonObject,
        api: &'a ToolApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<ToolExecutionResult, ToolError>> {
        Box::pin(async move {
            let arguments: ReadArguments = parse(arguments)?;
            let env = require_env(api)?;
            let absolute = resolve_read_path(env, &arguments.path, ctx).await?;
            let reader = env.open_binary_reader(&absolute, OpenOptions::default(), ctx).await?;
            let result = read_opened(&*reader, &arguments, api, self.images.as_deref(), ctx).await;
            reader.close(ctx).await;
            result
        })
    }
}

/// A concurrent writer can change the file between the scan and the reads. Appending (a
/// growing log) leaves the scanned text as it was; an image must be whole. Otherwise the file
/// is read again once.
async fn read_opened(
    reader: &dyn BinaryReader,
    arguments: &ReadArguments,
    api: &ToolApi<'_>,
    processor: Option<&dyn ImageProcessor>,
    ctx: &Context,
) -> Result<ToolExecutionResult, ToolError> {
    for _ in 0..2 {
        let before = reader.info(ctx).await?;
        let image = detect_image(reader, before.size, ctx).await?;
        let result = match image {
            Some(mime) => {
                let model = ImageModel::of(api);
                read_image(reader, &before, &arguments.path, mime, processor, &model, ctx).await?
            }
            None => read_text(reader, arguments, ctx).await?,
        };
        let after = reader.info(ctx).await?;
        let unchanged = after.size == before.size && after.mtime_ms == before.mtime_ms;
        if unchanged || (image.is_none() && after.size > before.size) {
            return Ok(result);
        }
    }
    Err(format!("{} changed while it was read", arguments.path).into())
}

/// What the conversation's model takes: whether images, and within which limits. A model
/// that does not resolve counts as one that takes images within the defaults.
struct ImageModel {
    vision: bool,
    limits: ImageLimits,
}

impl ImageModel {
    fn of(api: &ToolApi<'_>) -> Self {
        let model = api
            .agent()
            .model
            .as_ref()
            .and_then(|reference| api.models().ok()?.model(reference));
        Self {
            vision: model.as_ref().is_none_or(Model::supports_images),
            limits: ImageLimits::for_model(model.as_ref()),
        }
    }
}

/// One image item, prepared by `processor` or, without one, as it is when its format is
/// inline and it fits the limits (its size checked before reading it, its dimensions from
/// its header); otherwise an error result saying why.
async fn read_image(
    reader: &dyn BinaryReader,
    info: &FileInfo,
    path: &str,
    mime: &str,
    processor: Option<&dyn ImageProcessor>,
    model: &ImageModel,
    ctx: &Context,
) -> Result<ToolExecutionResult, ToolError> {
    let limits = model.limits;
    let mut notes = Vec::new();
    let (data, prepared_mime) = if let Some(processor) = processor {
        let bytes = read_whole(reader, info.size, ctx).await?;
        let Some(prepared) = processor.prepare(&bytes, mime, limits, ctx).await else {
            return Ok(image_error(format!(
                "{path} is an image ({mime}) that cannot be prepared for the model"
            )));
        };
        if let Some(from) = &prepared.converted_from {
            notes.push(format!("Converted from {from} to {}.", prepared.mime));
        }
        if let Some((from, to)) = prepared.resized {
            let scale = f64::from(from.width) / f64::from(to.width);
            notes.push(format!(
                "Resized from {}x{} to {}x{}. Multiply coordinates by {scale:.2} to map them to the original.",
                from.width, from.height, to.width, to.height
            ));
        }
        (prepared.data, prepared.mime)
    } else {
        if !is_inline(mime) {
            return Ok(image_error(format!(
                "{path} is an image ({mime}) that needs converting, and no image processor is configured"
            )));
        }
        let unshrinkable = "and no image processor is configured to shrink it";
        if base64_len(info.size) > limits.max_bytes {
            return Ok(image_error(format!(
                "{path} is an image ({mime}) of {}, too large to send, {unshrinkable}",
                format_size(info.size)
            )));
        }
        let bytes = read_whole(reader, info.size, ctx).await?;
        let Some(size) = image_dimensions(&bytes, mime) else {
            return Ok(image_error(format!(
                "{path} is an image ({mime}) whose size cannot be read"
            )));
        };
        if size.width > limits.max_width || size.height > limits.max_height {
            return Ok(image_error(format!(
                "{path} is an image ({mime}) of {}x{}, larger than {}x{}, {unshrinkable}",
                size.width, size.height, limits.max_width, limits.max_height
            )));
        }
        (bytes, mime.to_owned())
    };
    if !model.vision {
        notes.push("The current model does not support images; it sees a placeholder instead.".to_owned());
    }
    let mut message = format!("Read image file [{prepared_mime}].");
    for note in notes {
        message.push(' ');
        message.push_str(&note);
    }
    let image = ImageContent {
        data: STANDARD.encode(data),
        mime_type: prepared_mime,
    };
    Ok(ToolExecutionResult {
        output: Some(vec![UserBlock::Image(image)]),
        diagnostics: vec![ToolDiagnostic::new(Severity::Info, message).with_code("image")],
        ..ToolExecutionResult::default()
    })
}

/// The whole opened file.
async fn read_whole(reader: &dyn BinaryReader, size: u64, ctx: &Context) -> Result<Vec<u8>, ToolError> {
    let length = usize::try_from(size)?;
    Ok(reader.read(0, length, ctx).await?)
}

fn image_error(message: String) -> ToolExecutionResult {
    ToolExecutionResult {
        output: Some(Vec::new()),
        is_error: true,
        diagnostics: vec![ToolDiagnostic::error("unsupported_image", message)],
        ..ToolExecutionResult::default()
    }
}

/// A JavaScript `slice` index that a scan can take: a non-negative safe integer.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn scan_index(value: f64) -> Option<u64> {
    (0.0..=MAX_SAFE_INTEGER).contains(&value).then_some(value as u64)
}

/// The read result for the opened file. It equals decoding the whole file, splitting it on
/// `\n`, slicing the lines as JavaScript's `slice` does and bounding them with
/// `truncate_head`, while reading one scan's worth of the file plus the head.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
async fn read_text(
    reader: &dyn BinaryReader,
    arguments: &ReadArguments,
    ctx: &Context,
) -> Result<ToolExecutionResult, ToolError> {
    let ReadArguments {
        path, offset, limit, ..
    } = arguments;
    let start_line = match offset {
        Some(offset) if *offset != 0.0 => (offset - 1.0).max(0.0),
        _ => 0.0,
    };
    let start_display = start_line + 1.0;
    let slice_start = start_line.trunc();
    // One pass finds the line count and the selection; a selection past the last line ends
    // with it, and an empty one (a zero or negative limit) is scanned as one line and then
    // ignored. A start beyond any file is scanned from 0 only to count lines.
    let scan_start = scan_index(slice_start).unwrap_or(0);
    let scan_end = limit
        .map(|limit| ((scan_start + 1) as f64).max((start_line + limit).trunc()))
        .and_then(scan_index);
    let mut scan = reader.scan_lines(scan_start, scan_end, ctx).await?;
    let total_lines = scan.newlines as f64 + 1.0;
    if start_line >= total_lines {
        return Err(format!(
            "Offset {} is beyond end of file ({} lines total)",
            js_number(offset.unwrap_or(0.0)),
            js_number(total_lines)
        )
        .into());
    }

    let mut user_limited = None;
    let mut selected = total_lines - slice_start;
    if let Some(limit) = limit {
        let end_line = (start_line + limit).min(total_lines);
        user_limited = Some(end_line - start_line);
        // `slice` counts a negative end from the end, which only the line count tells.
        let relative_end = end_line.trunc();
        let slice_end = if relative_end < 0.0 {
            (total_lines + relative_end).max(0.0)
        } else {
            relative_end
        };
        selected = (slice_end - slice_start).max(0.0);
        if selected > 0.0 && relative_end < 0.0 {
            scan = reader.scan_lines(scan_start, Some(slice_end as u64), ctx).await?;
        }
    }
    let empty = selected == 0.0;
    // Counted as `truncate_head` counts: a trailing newline adds no line, empty text has none.
    let ends_with_newline = !empty && scan.last_line_start == scan.end && scan.last_line_start > scan.start;
    let totals = Totals {
        lines: if empty || scan.selected_bytes == 0 {
            0
        } else {
            selected as usize - usize::from(ends_with_newline)
        },
        bytes: if empty { 0 } else { scan.selected_bytes as usize },
    };
    let head = if empty {
        String::new()
    } else {
        let bom = reader.read(0, 3, ctx).await?.starts_with(&BYTE_ORDER_MARK);
        read_head(reader, &scan, bom, ctx).await?
    };

    let (shown, truncation) = truncate_head_of(&head, totals, &LIMITS);
    let mut text = shown;
    let mut diagnostics = Vec::new();
    let mut details = None;
    let start = js_number(start_display);
    if truncation.first_line_exceeds_limit {
        // The start of the line, cut at the byte limit on a character boundary. As
        // `lines[start_line]`, a fractional start line names no line.
        let integral = start_line.fract() == 0.0;
        let line = if integral {
            head.split('\n').next().unwrap_or_default()
        } else {
            ""
        };
        let size = if integral { scan.first_line_bytes } else { 0 };
        let end = char_end(line.as_bytes(), DEFAULT_MAX_BYTES);
        text = &line[..end];
        diagnostics.push(
            ToolDiagnostic::new(
                Severity::Warn,
                format!(
                    "Line {start} is {}, exceeds the {} limit; showing its first {}. Use bash: sed -n '{start}p' {path} | tail -c +{}",
                    format_size(size),
                    format_size(DEFAULT_MAX_BYTES as u64),
                    format_size(end as u64),
                    end + 1
                ),
            )
            .with_code("truncated"),
        );
        let truncation = super::Truncation {
            output_bytes: end,
            output_lines: 1,
            ..truncation
        };
        details = Some(serde_json::json!({ "truncation": truncation }));
    } else if truncation.truncated {
        let end_display = start_display + truncation.output_lines as f64 - 1.0;
        let limit_text = if truncation.truncated_by == Some(Limit::Lines) {
            String::new()
        } else {
            format!(" ({} limit)", format_size(DEFAULT_MAX_BYTES as u64))
        };
        diagnostics.push(
            ToolDiagnostic::new(
                Severity::Info,
                format!(
                    "Showing lines {start}-{} of {}{limit_text}. Use offset={} to continue.",
                    js_number(end_display),
                    js_number(total_lines),
                    js_number(end_display + 1.0)
                ),
            )
            .with_code("truncated"),
        );
        details = Some(serde_json::json!({ "truncation": truncation }));
    } else if let Some(limited) = user_limited
        && start_line + limited < total_lines
    {
        let remaining = total_lines - (start_line + limited);
        diagnostics.push(ToolDiagnostic::new(
            Severity::Info,
            format!(
                "{} more lines in file. Use offset={} to continue.",
                js_number(remaining),
                js_number(start_line + limited + 1.0)
            ),
        ));
    }

    Ok(ToolExecutionResult {
        output: Some(if text.is_empty() {
            Vec::new()
        } else {
            vec![UserBlock::text(text)]
        }),
        details,
        diagnostics,
        ..ToolExecutionResult::default()
    })
}

/// The decoded start of the selection, decoded as part of the whole file: all of it, or
/// enough for `truncate_head_of` (more than `DEFAULT_MAX_BYTES + 1` bytes, or
/// `DEFAULT_MAX_LINES` newlines).
async fn read_head(reader: &dyn BinaryReader, scan: &LineScan, bom: bool, ctx: &Context) -> FileResult<String> {
    let mut decoder = Utf8Decoder::new();
    let mut text = String::new();
    let mut newlines = 0;
    let mut position = if bom && scan.start == 0 {
        BYTE_ORDER_MARK.len() as u64
    } else {
        scan.start
    };
    while position < scan.end {
        let length = usize::try_from(scan.end - position).map_or(READ_CHUNK, |rest| rest.min(READ_CHUNK));
        let bytes = reader.read(position, length, ctx).await?;
        if bytes.is_empty() {
            break;
        }
        position += bytes.len() as u64;
        let decoded = decoder.decode_to_string(&bytes);
        newlines += decoded.matches('\n').count();
        text.push_str(&decoded);
        if newlines >= DEFAULT_MAX_LINES || text.len() > DEFAULT_MAX_BYTES + 1 {
            return Ok(text);
        }
    }
    text.push_str(&decoder.finish_to_string());
    Ok(text)
}
