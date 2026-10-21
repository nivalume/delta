//! The end of a call: the final result (`after_tool` chain, truncation, bounding, the
//! structured output contract) and the settle commit that writes it where the caller reads it.

use delta_ai::types::{ToolCall, UserBlock};
use delta_chord::context::Context;
use delta_chord::json;
use delta_chord::output::{OutputLimits, Retain, bound_output};
use serde_json::Value;

use super::definition::HarnessTool;
use super::nested::MADE_NESTED_CALLS;
use super::output::{Reports, Waiters};
use super::records::{NestedCallDoc, NestedCallState, ToolHookCall, ToolTaskInput, ToolTaskResult};
use super::result::{
    NestedToolExecutionResult, Severity, ToolDiagnostic, ToolExecutionResult, append_tool_result, harness_error,
    truncated,
};
use super::task::{Rt, ToolTask};
use super::validate;
use crate::agent::HookRuntime;
use crate::approval::{ApprovalId, remove_request};
use crate::documents::DocAddress;
use crate::generation::{LiveDoc, LiveState, NestedToolSummary, SlotProgress, SlotStatus, edit_live};
use crate::records::TaskOutcomeError;
use crate::scheduler::{Next, NextState, Outcome, TaskError};
use crate::session::TxError;
use crate::usage::{UsageBucket, record_usage};

/// How much error text a nested slot's summary keeps.
const SUMMARY_CHARS: usize = 500;

/// How the task ends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Ending {
    /// A returned result, also an error result.
    Completed,
    /// An error from `execute` or the environment, or an interruption.
    Failed(String),
    /// The abort handler's.
    Aborted,
}

/// What the settle commit writes.
#[derive(Debug)]
pub(super) struct Settlement {
    /// The call as stored: the model's, or the nested input's.
    pub(super) call: ToolCall,
    pub(super) result: ToolExecutionResult,
    /// Measured around `execute`; absent for results written before intent.
    pub(super) duration_ms: Option<u64>,
    pub(super) ending: Ending,
    pub(super) waiters: Waiters,
    /// The task's approval request, removed in the settle commit.
    pub(super) withdraw: Option<ApprovalId>,
}

impl Settlement {
    /// A harness-written result that ends the task `completed` without having run anything.
    pub(super) fn early(rt: &Rt, call: ToolCall, code: &str, message: impl Into<String>) -> Self {
        Self {
            call,
            result: harness_error(code, message),
            duration_ms: None,
            ending: Ending::Completed,
            waiters: Waiters::none(rt.task_id()),
            withdraw: None,
        }
    }

    /// Also removes the approval request `id`.
    pub(super) fn withdrawing(mut self, id: ApprovalId) -> Self {
        self.withdraw = Some(id);
        self
    }
}

/// The commit that ends the task: a model-issued call appends its `delta.tool-result` entry
/// and marks its slot done; a nested call writes its result into the caller's
/// `delta.tool.nested-call` member and its slot's summary. Both remove the nested slots
/// below the task when it made nested calls.
pub(super) fn settle(rt: &Rt, settlement: Settlement) -> Result<Next<ToolTask>, TaskError> {
    let made_nested = rt.memo::<bool>(MADE_NESTED_CALLS)?.unwrap_or(false);
    let input = rt.input().clone();
    let conversation = rt.conversation_id();
    let task = rt.task_id();
    let now = rt.now();
    Ok(rt.next(move |tx, _| {
        Box::pin(async move {
            let Settlement {
                call,
                result,
                duration_ms,
                ending,
                waiters,
                withdraw,
            } = settlement;
            if let Some(id) = withdraw {
                remove_request(tx, conversation, id).await?;
            }
            let tool_result = match input {
                ToolTaskInput::Model { .. } => {
                    let entry = append_tool_result(tx, conversation, &call, &result, now, duration_ms).await?;
                    edit_live(tx, conversation, |live| {
                        if let Some(slot) = live.tool_slot_mut(task) {
                            slot.finish();
                            slot.entry = Some(entry.id);
                        }
                        if made_nested {
                            live.remove_nested_below(task);
                        }
                    })
                    .await?;
                    let control = result.control.filter(|_| ending == Ending::Completed);
                    ToolTaskResult::Model {
                        entry_id: entry.id,
                        control,
                    }
                }
                ToolTaskInput::Nested { parent, key, .. } => {
                    if let Some(usage) = &result.usage {
                        record_usage(tx, conversation, UsageBucket::Tools, &call.name, usage).await?;
                    }
                    let summary = summary(&result, duration_ms);
                    let nested = NestedToolExecutionResult {
                        task_id: task,
                        structured_output: result.structured_output,
                        is_error: result.is_error,
                        details: result.details,
                        diagnostics: result.diagnostics,
                        usage: result.usage,
                        duration_ms,
                    };
                    tx.doc_member(DocAddress::<NestedCallDoc>::member(parent, key), ())
                        .await?
                        .update_as::<NestedCallState, _>(|state| state.result = Some(nested))
                        .map_err(TxError::callback)?;
                    edit_live(tx, conversation, |live| {
                        if let Some(slot) = live.nested_slot_mut(task) {
                            slot.status = SlotStatus::Done;
                            slot.progress = SlotProgress::default();
                            slot.summary = Some(summary);
                        }
                        if made_nested {
                            live.remove_nested_below(task);
                        }
                    })
                    .await?;
                    ToolTaskResult::Nested
                }
            };
            waiters.resolve();
            Ok(NextState::Terminal(match ending {
                Ending::Completed => Outcome::Completed(tool_result),
                Ending::Failed(message) => Outcome::Failed {
                    error: TaskOutcomeError::new(message),
                    result: Some(tool_result),
                },
                Ending::Aborted => Outcome::Aborted {
                    reason: None,
                    result: Some(tool_result),
                },
            }))
        })
    }))
}

/// A nested slot's summary: up to 500 characters of the error text, from the error
/// diagnostics or else the output.
fn summary(result: &ToolExecutionResult, duration_ms: Option<u64>) -> NestedToolSummary {
    let error = result.is_error.then(|| {
        let diagnostics: Vec<&str> = result
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.severity == Severity::Error)
            .map(|diagnostic| diagnostic.message.as_str())
            .collect();
        let text = if diagnostics.is_empty() {
            text_of(result.output.as_deref().unwrap_or_default())
        } else {
            diagnostics.join("\n")
        };
        text.chars().take(SUMMARY_CHARS).collect()
    });
    NestedToolSummary {
        is_error: result.is_error,
        duration_ms,
        usage: result.usage.clone(),
        error,
    }
}

/// The text items of `content`, joined by newlines.
fn text_of(content: &[UserBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            UserBlock::Text(text) => Some(text.text.as_str()),
            UserBlock::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The published progress of the task's slot, from the committed `delta.live`.
pub(super) async fn slot_progress(rt: &Rt, ctx: &Context) -> Result<SlotProgress, TaskError> {
    let Some(value) = rt
        .snapshot(DocAddress::<LiveDoc>::new(rt.conversation_id()), ctx)
        .await?
    else {
        return Ok(SlotProgress::default());
    };
    let mut live: LiveState = json::from_json(&value)?;
    Ok(live
        .slot_progress_mut(rt.task_id())
        .map(std::mem::take)
        .unwrap_or_default())
}

/// An error result built from what the slot published: its partial output, details and
/// diagnostics, then the `code` error and a `truncated` warning when the window dropped
/// output.
pub(super) fn from_slot(progress: SlotProgress, retain: Retain, code: &str, message: String) -> ToolExecutionResult {
    let mut result = harness_error(code, message);
    if let Some(output) = progress.output.filter(|output| !output.is_empty()) {
        result.output = Some(vec![UserBlock::text(output)]);
    }
    result.details = progress.details.map(Value::from);
    let mut diagnostics = progress.diagnostics.unwrap_or_default();
    diagnostics.append(&mut result.diagnostics);
    let (bytes, lines) = (progress.dropped_bytes.unwrap_or(0), progress.dropped_lines.unwrap_or(0));
    if bytes > 0 || lines > 0 {
        diagnostics.push(truncated(retain, lines, bytes));
    }
    result.diagnostics = diagnostics;
    result
}

/// The result as committed (tool.md, "Final result and settlement"): the returned result
/// over what the invocation reported, through the `after_tool` chain, with harness
/// diagnostics, explicit text bounded and the structured output contract applied.
pub(super) async fn final_result(
    rt: &Rt,
    ctx: &Context,
    tool: &dyn HarnessTool,
    call: &ToolHookCall,
    returned: ToolExecutionResult,
    reports: Reports,
) -> Result<(ToolExecutionResult, Waiters), TaskError> {
    let Reports {
        output: window,
        retain,
        details,
        diagnostics: reported,
        waiters,
    } = reports;
    let retained: Vec<UserBlock> = if window.text.is_empty() {
        Vec::new()
    } else {
        vec![UserBlock::text(window.text.clone())]
    };
    let mut result = returned;
    let used_window = result.output.is_none();
    if used_window {
        result.output = Some(retained.clone());
    }
    if result.details.is_none() {
        result.details = details;
    }
    let mut diagnostics = reported;
    diagnostics.append(&mut result.diagnostics);
    result.diagnostics = diagnostics;

    let mut result = after_tool(rt, call, result, ctx).await?;

    let output = result.output.take().unwrap_or_default();
    let window_kept = used_window && output == retained;
    if window_kept && (window.dropped_bytes > 0 || window.dropped_lines > 0) {
        result.diagnostics.push(truncated(
            retain,
            count(window.dropped_lines),
            count(window.dropped_bytes),
        ));
    }
    let output = if window_kept {
        output
    } else {
        let (bounded, dropped) = bound_text(output, &tool.output_limits());
        if let Some((lines, bytes)) = dropped {
            result
                .diagnostics
                .push(truncated(tool.output_limits().retain, lines, bytes));
        }
        bounded
    };
    let nested = call.parent.is_some();
    structured_contract(rt, tool, nested, &mut result, &output);
    result.output = Some(output);
    Ok((result, waiters))
}

fn count(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// The `after_tool` replacement chain; an ordinary error is reported and skipped.
async fn after_tool(
    rt: &Rt,
    call: &ToolHookCall,
    mut result: ToolExecutionResult,
    ctx: &Context,
) -> Result<ToolExecutionResult, TaskError> {
    let hooks = rt.hooks(ctx).await?;
    if hooks.is_empty() {
        return Ok(result);
    }
    let api = rt.hook_api()?;
    for (extension, hook) in hooks.iter() {
        match hook.after_tool(call, &result, &api, ctx).await {
            Ok(Some(replacement)) => result = replacement,
            Ok(None) => {}
            Err(error) => super::pipeline::hook_failed(rt, &*error, extension, ctx)?,
        }
    }
    Ok(result)
}

/// The text items of `content` bounded together: on overflow they collapse into one bounded
/// text item at the first (head) or last (tail) text item's position; other items stay.
/// Returns what was dropped, as `(lines, bytes)`.
fn bound_text(content: Vec<UserBlock>, limits: &OutputLimits) -> (Vec<UserBlock>, Option<(u64, u64)>) {
    let texts: Vec<usize> = content
        .iter()
        .enumerate()
        .filter_map(|(index, block)| matches!(block, UserBlock::Text(_)).then_some(index))
        .collect();
    let (Some(&first), Some(&last)) = (texts.first(), texts.last()) else {
        return (content, None);
    };
    let joined = text_of(&content);
    let slice = bound_output(&joined, limits);
    if slice.dropped_bytes == 0 && slice.dropped_lines == 0 {
        return (content, None);
    }
    let anchor = match limits.retain {
        Retain::Head => first,
        Retain::Tail => last,
    };
    let bounded = UserBlock::text(slice.text);
    let dropped = (count(slice.dropped_lines), count(slice.dropped_bytes));
    let mut kept = Vec::with_capacity(content.len() - texts.len() + 1);
    let mut bounded = Some(bounded);
    for (index, block) in content.into_iter().enumerate() {
        if index == anchor {
            kept.extend(bounded.take());
        } else if !matches!(block, UserBlock::Text(_)) {
            kept.push(block);
        }
    }
    (kept, Some(dropped))
}

/// The structured output contract: a tool with a schema returns a matching value (except
/// on an error result), a tool without one returns none. A broken nested result becomes an
/// error; a broken model-issued one only loses its value, and the break is reported. A
/// schema-less nested result carries its output as its value.
fn structured_contract(
    rt: &Rt,
    tool: &dyn HarnessTool,
    nested: bool,
    result: &mut ToolExecutionResult,
    output: &[UserBlock],
) {
    let schema = tool.structured_output_schema();
    let broken = match (schema, &result.structured_output) {
        (Some(_), None) if !result.is_error => Some(format!("Tool {} returned no structured output", tool.name())),
        (Some(schema), Some(value)) => validate::structured(schema, value).map(|problem| {
            format!(
                "Structured output of tool {} does not match its schema: {problem}",
                tool.name()
            )
        }),
        (None, Some(_)) => Some(format!(
            "Tool {} returned structured output without declaring a schema",
            tool.name()
        )),
        _ => None,
    };
    if let Some(problem) = broken {
        result.structured_output = None;
        if nested {
            result.is_error = true;
            result
                .diagnostics
                .push(ToolDiagnostic::error("invalid_structured_output", problem));
        } else {
            rt.report(&std::io::Error::other(problem), None);
        }
    }
    if !nested {
        return;
    }
    if schema.is_none() {
        result.structured_output = Some(output_value(output));
    } else if result.is_error
        && result.structured_output.is_none()
        && !result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == Severity::Error)
    {
        result
            .diagnostics
            .push(ToolDiagnostic::error("tool_error", text_of(output)));
    }
}

/// A schema-less nested result's output as a value: one text item as its string, one image
/// as itself, none as `""`, else the list.
fn output_value(output: &[UserBlock]) -> Value {
    match output {
        [] => Value::String(String::new()),
        [UserBlock::Text(text)] => Value::String(text.text.clone()),
        [image @ UserBlock::Image(_)] => serde_json::to_value(image).unwrap_or(Value::Null),
        many => serde_json::to_value(many).unwrap_or(Value::Null),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(blocks: &[UserBlock]) -> Vec<String> {
        blocks
            .iter()
            .map(|block| match block {
                UserBlock::Text(text) => text.text.clone(),
                UserBlock::Image(_) => "<image>".to_owned(),
            })
            .collect()
    }

    #[test]
    fn explicit_text_is_bounded_into_one_item() {
        let limits = OutputLimits {
            max_bytes: 1000,
            max_lines: 2,
            retain: Retain::Tail,
        };
        let content = vec![UserBlock::text("a\nb"), UserBlock::text("c\nd")];
        let (bounded, dropped) = bound_text(content, &limits);
        assert_eq!(texts(&bounded), vec!["c\nd"]);
        assert_eq!(dropped, Some((2, 4)));

        let fits = vec![UserBlock::text("a"), UserBlock::text("b")];
        let (kept, dropped) = bound_text(fits.clone(), &limits);
        assert_eq!((kept, dropped), (fits, None));
    }

    #[test]
    fn output_values_collapse() {
        assert_eq!(output_value(&[]), Value::String(String::new()));
        assert_eq!(output_value(&[UserBlock::text("x")]), Value::String("x".to_owned()));
        assert!(output_value(&[UserBlock::text("x"), UserBlock::text("y")]).is_array());
    }

    #[test]
    fn slot_results_keep_partial_output() {
        let progress = SlotProgress {
            output: Some("partial".to_owned()),
            dropped_bytes: Some(10),
            dropped_lines: Some(1),
            diagnostics: Some(vec![ToolDiagnostic::new(Severity::Info, "note")]),
            ..SlotProgress::default()
        };
        let result = from_slot(
            progress,
            Retain::Head,
            "interrupted",
            "Tool t was interrupted".to_owned(),
        );
        assert_eq!(texts(result.output.as_deref().unwrap()), vec!["partial"]);
        let codes: Vec<Option<&str>> = result.diagnostics.iter().map(|d| d.code.as_deref()).collect();
        assert_eq!(codes, vec![None, Some("interrupted"), Some("truncated")]);
        assert!(result.is_error);
    }

    #[test]
    fn summaries_prefer_error_diagnostics() {
        let mut result = ToolExecutionResult::error("out");
        assert_eq!(summary(&result, Some(3)).error.as_deref(), Some("out"));
        result
            .diagnostics
            .push(ToolDiagnostic::error("tool_error", "x".repeat(600)));
        assert_eq!(summary(&result, None).error.map(|text| text.len()), Some(SUMMARY_CHARS));
    }
}
