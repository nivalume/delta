//! A thread's turns from its records: the visible entries replayed through the translator,
//! with the submissions of the conversation and its fork ancestors, so `thread/read` names
//! turns and items as the live notifications did.

use delta_ai::types::Message;
use delta_chord::context::Context;
use delta_harness::compaction::{COMPACTION_ENTRY, CompactionReason};
use delta_harness::harness::{Conversation, EntryRange, Harness, HarnessError};
use delta_harness::observe::{AgentEvent, ToolEventCall};
use delta_harness::records::{EntryRecord, ScanOrder, SubmissionRecord};
use delta_harness::tool::TOOL_RESULT_ENTRY;
use delta_protocol::ServerNotification;
use delta_protocol::v2::{ThreadItem, Turn};

use crate::items::Outcome;
use crate::translate::{Output, Translator};

const PAGE: usize = 256;

/// When a message was made (epoch ms).
pub(crate) fn timestamp(message: &Message) -> i64 {
    match message {
        Message::System(message) => message.timestamp,
        Message::User(message) => message.timestamp,
        Message::Assistant(message) => message.timestamp,
        Message::ToolResult(message) => message.timestamp,
    }
}

/// Every entry visible from `conversation`, oldest first.
pub(crate) async fn entries(conversation: &Conversation, ctx: &Context) -> Result<Vec<EntryRecord>, HarnessError> {
    let range = EntryRange {
        order: Some(ScanOrder::Ascending),
        ..EntryRange::default()
    };
    let mut entries = Vec::new();
    let mut cursor = None;
    loop {
        let page = conversation.entries(range, PAGE, cursor, ctx).await?;
        entries.extend(page.items);
        match page.next {
            Some(next) => cursor = Some(next),
            None => return Ok(entries),
        }
    }
}

/// The submissions of `conversation` and of the conversations it was forked from.
pub(crate) async fn submissions(
    harness: &Harness,
    conversation: &Conversation,
    ctx: &Context,
) -> Result<Vec<SubmissionRecord>, HarnessError> {
    let mut records = Vec::new();
    let mut current = Some(conversation.clone());
    while let Some(conversation) = current.take() {
        let mut cursor = None;
        loop {
            let page = conversation.submissions(PAGE, cursor, ctx).await?;
            records.extend(page.items);
            match page.next {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }
        if let Some(parent) = conversation.record(ctx).await?.parent {
            current = harness.conversation(parent.conversation_id, ctx).await?;
        }
    }
    Ok(records)
}

/// The turns `entries` hold, as `thread/read` reports them.
pub(crate) fn turns(thread_id: &str, cwd: &str, entries: &[EntryRecord], records: Vec<SubmissionRecord>) -> Vec<Turn> {
    let mut translator = Translator::new(thread_id.to_owned(), cwd.to_owned());
    translator.know(records);
    let mut turns: Vec<Turn> = Vec::new();
    for entry in entries {
        let at = entry.model.as_deref().and_then(<[Message]>::first).map(timestamp);
        if at.is_some() {
            translator.set_clock(at);
        }
        let events = if entry.kind == TOOL_RESULT_ENTRY {
            let Some(outcome) = Outcome::of(entry) else { continue };
            vec![AgentEvent::ToolExecutionEnd {
                call: ToolEventCall {
                    tool_call_id: outcome.message.tool_call_id.clone(),
                    tool_name: outcome.message.tool_name.clone(),
                    task_id: None,
                    parent_tool_call_id: None,
                    parent_task_id: None,
                },
                entry: Some(entry.clone()),
                result: None,
            }]
        } else if entry.kind == COMPACTION_ENTRY {
            let Some(task_id) = entry.by_task_id else { continue };
            vec![
                AgentEvent::CompactionStart {
                    task_id,
                    reason: CompactionReason::Manual,
                    blocking: false,
                },
                AgentEvent::CompactionEnd {
                    task_id,
                    reason: CompactionReason::Manual,
                    error: None,
                },
            ]
        } else if entry.model.is_some() {
            vec![AgentEvent::MessageEnd { entry: entry.clone() }]
        } else {
            continue;
        };
        for output in translator.batch(&events) {
            let Output::Notify(notification) = output else { continue };
            collect(&mut turns, notification);
        }
    }
    for output in translator.finish() {
        let Output::Notify(notification) = output else { continue };
        collect(&mut turns, notification);
    }
    turns
}

fn turn_of<'a>(turns: &'a mut [Turn], id: &str) -> Option<&'a mut Vec<ThreadItem>> {
    turns
        .iter_mut()
        .rev()
        .find(|turn| turn.id == id)
        .map(|turn| &mut turn.items)
}

fn collect(turns: &mut Vec<Turn>, notification: ServerNotification) {
    match notification {
        ServerNotification::TurnStarted(started) => turns.push(started.turn),
        ServerNotification::TurnCompleted(completed) => {
            if let Some(turn) = turns.iter_mut().rev().find(|turn| turn.id == completed.turn.id) {
                let items = std::mem::take(&mut turn.items);
                *turn = Turn {
                    items,
                    ..completed.turn
                };
            }
        }
        ServerNotification::ItemStarted(started) => {
            if let Some(items) = turn_of(turns, &started.turn_id)
                && !items.iter().any(|item| item.id() == started.item.id())
            {
                items.push(started.item);
            }
        }
        ServerNotification::ItemCompleted(completed) => {
            if let Some(items) = turn_of(turns, &completed.turn_id) {
                match items.iter_mut().find(|item| item.id() == completed.item.id()) {
                    Some(item) => *item = completed.item,
                    None => items.push(completed.item),
                }
            }
        }
        _ => {}
    }
}
