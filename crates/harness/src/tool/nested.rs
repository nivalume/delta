//! Nested calls: a tool calls another tool through `ToolApi::execute_tool`, as a `delta.tool`
//! task it owns, keyed so a rerun reattaches to the same call.

use std::sync::atomic::Ordering;

use delta_ai::types::{JsonObject, ToolCall};
use delta_chord::context::Context;
use delta_chord::json;

use super::api::{NestedCallOptions, ToolApi, ToolApiError};
use super::definition::ToolReplay;
use super::records::{NestedCallDoc, NestedCallState, ToolTaskInput};
use super::result::{NestedToolExecutionResult, ToolDiagnostic};
use super::task::ToolTask;
use crate::documents::DocAddress;
use crate::generation::{NestedToolSlot, SlotProgress, SlotStatus, edit_live};
use crate::records::{TaskId, TaskOutcome};
use crate::scheduler::RuntimeError;
use crate::session::{TaskOptions, TxError};

/// The memo a tool task records before its first nested call; settlement removes nested
/// slots only when it is set.
pub(super) const MADE_NESTED_CALLS: &str = "delta.tool.made_nested_calls";

/// Why an explicit key may not be used; `None` when it may.
pub(super) fn check_key(key: &str) -> Option<&'static str> {
    if key.is_empty() {
        Some("it is empty")
    } else if key.contains('/') {
        Some("it contains '/'")
    } else if key == "__proto__" {
        Some("it is reserved")
    } else if key.bytes().all(|byte| byte.is_ascii_digit()) && key.bytes().any(|byte| byte != b'0') {
        Some("positive integers are the default keys")
    } else {
        None
    }
}

pub(super) async fn execute_tool(
    api: &ToolApi<'_>,
    name: &str,
    arguments: JsonObject,
    options: NestedCallOptions,
    ctx: &Context,
) -> Result<NestedToolExecutionResult, ToolApiError> {
    let position = api.nested.count.fetch_add(1, Ordering::Relaxed) + 1;
    let key = match options.key {
        Some(key) => {
            if let Some(reason) = check_key(&key) {
                return Err(ToolApiError::InvalidKey { key, reason });
            }
            key
        }
        None => position.to_string(),
    };
    let rt = api.rt;
    if !api.nested.recorded.swap(true, Ordering::Relaxed) {
        rt.memo_or_insert(MADE_NESTED_CALLS, true, ctx).await?;
    }
    let parent = rt.task_id();
    let conversation = rt.conversation_id();
    let call = ToolCall {
        id: format!("{}/{key}", api.call_id),
        name: name.to_owned(),
        arguments,
        thought_signature: None,
        namespace: None,
        arguments_error: None,
    };
    let input = ToolTaskInput::Nested {
        parent,
        parent_call_id: api.call_id.to_owned(),
        key: key.clone(),
        call: call.clone(),
        progress: options.progress.filter(|progress| !progress),
    };
    let options = TaskOptions {
        owner: Some(parent),
        abandon_on_restart: api.replay != ToolReplay::Safe,
        ..TaskOptions::default()
    };
    let address = DocAddress::<NestedCallDoc>::member(parent, key.clone());
    let admitted = rt
        .commit(ctx, {
            let address = address.clone();
            move |tx, _| {
                Box::pin(async move {
                    let mut doc = tx.doc_member(address.clone(), ()).await?;
                    let state: NestedCallState = json::from_json(doc.value()).map_err(TxError::callback)?;
                    if let Some(task) = state.task_id {
                        let same = state.name.as_deref() == Some(call.name.as_str())
                            && state.arguments.as_ref() == Some(&call.arguments);
                        return Ok((same.then_some(task), None));
                    }
                    doc.update_as::<NestedCallState, _>(|state| {
                        state.name = Some(call.name.clone());
                        state.arguments = Some(call.arguments.clone());
                    })
                    .map_err(TxError::callback)?;
                    let task = tx.create_task(&ToolTask, input, options).await?.id();
                    tx.doc_member(address, ())
                        .await?
                        .update_as::<NestedCallState, _>(|state| state.task_id = Some(task))
                        .map_err(TxError::callback)?;
                    let slot = NestedToolSlot {
                        call_id: call.id,
                        name: call.name,
                        task_id: task,
                        parent_task_id: parent,
                        status: SlotStatus::Pending,
                        arguments: call.arguments,
                        progress: SlotProgress::default(),
                        summary: None,
                    };
                    edit_live(tx, conversation, |live| {
                        live.nested_tools.get_or_insert_with(Vec::new).push(slot);
                    })
                    .await?;
                    Ok((Some(task), None))
                })
            }
        })
        .await?;
    let Some(task) = admitted else {
        return Err(ToolApiError::KeyMismatch { key });
    };
    let settled = rt.wait_for_task(task, ctx).await?;
    let state: Option<NestedCallState> = rt
        .snapshot(address, ctx)
        .await?
        .map(|value| json::from_json(&value))
        .transpose()
        .map_err(RuntimeError::Codec)?;
    Ok(match state.and_then(|state| state.result) {
        Some(result) => result,
        None => missing_result(task, settled.outcome()),
    })
}

/// The result of a nested task that ended without writing one (faulted, orphaned).
fn missing_result(task: TaskId, outcome: &TaskOutcome) -> NestedToolExecutionResult {
    let diagnostic = match outcome {
        TaskOutcome::Faulted { error } => ToolDiagnostic::error("faulted", error.message.clone()),
        TaskOutcome::Orphaned { reason } => ToolDiagnostic::error("orphaned", format!("Tool call orphaned: {reason}")),
        _ => ToolDiagnostic::error("faulted", "Tool call ended without a result"),
    };
    NestedToolExecutionResult {
        is_error: true,
        diagnostics: vec![diagnostic],
        ..NestedToolExecutionResult::new(task)
    }
}

/// Aborts every live nested call `rt`'s task owns, all at once, and waits until they settled
/// into their slots.
pub(super) async fn abort_nested(
    rt: &crate::scheduler::TaskRuntime<ToolTask>,
    ctx: &Context,
) -> Result<(), RuntimeError> {
    let owned = rt.owned_tasks(ctx).await?;
    let nested: Vec<TaskId> = owned
        .iter()
        .filter(|record| record.kind == super::TOOL_TASK)
        .map(|record| record.id)
        .collect();
    let aborts = nested.into_iter().map(|id| rt.abort_owned(id, ctx));
    for result in futures::future::join_all(aborts).await {
        result?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_keys_are_checked() {
        assert_eq!(check_key("lookup"), None);
        assert_eq!(check_key("0"), None);
        assert_eq!(check_key("007a"), None);
        assert!(check_key("").is_some());
        assert!(check_key("a/b").is_some());
        assert!(check_key("__proto__").is_some());
        assert!(check_key("12").is_some());
        assert!(check_key("007").is_some());
    }
}
