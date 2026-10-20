//! Prompt planning on fixed views (upstream `harness-prompt`).

use std::sync::Arc;

use delta_ai::transcript::get_current_tools;
use delta_ai::types::{Message, SystemContent, SystemMessage, Tool};
use delta_chord::context::Context;
use futures::future::BoxFuture;
use indexmap::IndexMap;

use super::{
    PromptInput, PromptSection, SYSTEM_ENTRY, SectionError, TextSection, plan_system_entries, render_sections,
    replay_sections,
};
use crate::agent::{Agent, Settings, resolve_agent};
use crate::context::{ContextView, derive_view};
use crate::records::{ConversationId, EditAction, EntryDraft, EntryId, EntryRecord};
use crate::registry::Registry;

fn sections(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(key, text)| ((*key).to_owned(), (*text).to_owned()))
        .collect()
}

fn tool(name: &str, description: &str) -> Tool {
    Tool::new(name, description, serde_json::json!({ "type": "object" }))
}

/// A view whose transcript is the planned drafts committed as entries 1, 2, ...
fn committed(drafts: &[EntryDraft]) -> ContextView {
    let entries: Vec<EntryRecord> = drafts
        .iter()
        .enumerate()
        .map(|(index, draft)| {
            let mut record = EntryRecord::new(EntryId(index as u64 + 1), ConversationId(1), draft.kind.clone());
            record.model.clone_from(&draft.model);
            record.edits.clone_from(&draft.edits);
            record
        })
        .collect();
    derive_view(None, &entries)
}

fn system(draft: &EntryDraft) -> &SystemMessage {
    match draft.model.as_deref() {
        Some([Message::System(system)]) => system,
        other => panic!("not a system draft: {other:?}"),
    }
}

#[test]
fn the_first_plan_states_everything_and_an_equal_plan_appends_nothing() {
    let desired = sections(&[("a", "A"), ("b", "B")]);
    let tools = [tool("read", "Read")];
    let first = plan_system_entries(&ContextView::default(), &desired, &tools, 1);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].kind, SYSTEM_ENTRY);
    assert_eq!(system(&first[0]).content, SystemContent::Text(String::new()));
    let view = committed(&first);
    assert_eq!(replay_sections(&view.messages), desired);
    assert_eq!(get_current_tools(&view.messages), tools);
    assert!(plan_system_entries(&view, &desired, &tools, 2).is_empty());
}

#[test]
fn only_changed_sections_are_appended() {
    let tools = [tool("read", "Read")];
    let first = plan_system_entries(&ContextView::default(), &sections(&[("a", "A"), ("b", "B")]), &tools, 1);
    let view = committed(&first);
    let patch = plan_system_entries(&view, &sections(&[("a", "A"), ("b", "B2"), ("c", "C")]), &tools, 2);
    assert_eq!(patch.len(), 1);
    let message = system(&patch[0]);
    let expected: IndexMap<String, Option<String>> = [("b", Some("B2")), ("c", Some("C"))]
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.map(str::to_owned)))
        .collect();
    assert_eq!(message.sections.as_ref(), Some(&expected));
    assert!(message.tools_added.is_none() && message.tools_removed.is_none());
    let removal = plan_system_entries(&view, &sections(&[("a", "A")]), &tools, 3);
    let expected: IndexMap<String, Option<String>> = [("b".to_owned(), None)].into_iter().collect();
    assert_eq!(system(&removal[0]).sections.as_ref(), Some(&expected));
}

#[test]
fn an_order_change_removes_every_section_and_re_adds_them() {
    let first = plan_system_entries(&ContextView::default(), &sections(&[("a", "A"), ("b", "B")]), &[], 1);
    let mut drafts = first.clone();
    let desired = sections(&[("b", "B"), ("a", "A")]);
    let pair = plan_system_entries(&committed(&first), &desired, &[], 2);
    assert_eq!(pair.len(), 2);
    assert!(
        system(&pair[0])
            .sections
            .as_ref()
            .unwrap()
            .values()
            .all(Option::is_none)
    );
    drafts.extend(pair);
    assert_eq!(replay_sections(&committed(&drafts).messages), desired);
}

#[test]
fn tools_are_patched_in_order_and_changed_declarations_replaced() {
    let first = plan_system_entries(
        &ContextView::default(),
        &IndexMap::new(),
        &[tool("read", "Read"), tool("write", "Write")],
        1,
    );
    let mut drafts = first.clone();
    // A changed declaration is removed and re-added; an addition appends.
    let desired = [tool("read", "Read"), tool("write", "Write v2"), tool("bash", "Bash")];
    let patch = plan_system_entries(&committed(&drafts), &IndexMap::new(), &desired, 2);
    assert_eq!(patch.len(), 1);
    let message = system(&patch[0]);
    let removed: Vec<&str> = message
        .tools_removed
        .iter()
        .flatten()
        .map(|tool| tool.name.as_str())
        .collect();
    assert_eq!(removed, ["write"]);
    drafts.extend(patch);
    assert_eq!(get_current_tools(&committed(&drafts).messages), desired);
    // A reorder that appending cannot give re-adds every tool.
    let reordered = [tool("bash", "Bash"), tool("read", "Read"), tool("write", "Write v2")];
    let patch = plan_system_entries(&committed(&drafts), &IndexMap::new(), &reordered, 3);
    assert_eq!(system(&patch[0]).tools_removed.as_ref().map(Vec::len), Some(3));
    drafts.extend(patch);
    assert_eq!(get_current_tools(&committed(&drafts).messages), reordered);
}

#[test]
fn a_head_marker_gets_a_complete_baseline_that_omits_earlier_system_entries() {
    let first = plan_system_entries(
        &ContextView::default(),
        &sections(&[("a", "A")]),
        &[tool("read", "R")],
        1,
    );
    let mut entries: Vec<EntryRecord> = Vec::new();
    let mut system_entry = EntryRecord::new(EntryId(1), ConversationId(1), SYSTEM_ENTRY);
    system_entry.model.clone_from(&first[0].model);
    entries.push(system_entry);
    let mut user = EntryRecord::new(EntryId(2), ConversationId(1), "delta.user");
    user.model = Some(vec![Message::user("hi")]);
    entries.push(user);
    let mut marker = EntryRecord::new(EntryId(3), ConversationId(1), "delta.compaction");
    marker.model = Some(vec![Message::user("summary")]);
    marker.head = Some(EntryId(1));
    entries.push(marker.clone());
    let view = derive_view(Some(&marker), &entries);
    // Unchanged values are restated: every later preparation finds a baseline after the marker.
    let plan = plan_system_entries(&view, &sections(&[("a", "A")]), &[tool("read", "R")], 2);
    assert_eq!(plan.len(), 1);
    let edits = plan[0].edits.as_ref().expect("omissions");
    assert_eq!(edits.len(), 1);
    assert_eq!((edits[0].target, &edits[0].action), (EntryId(1), &EditAction::Omit));
    let message = system(&plan[0]);
    assert_eq!(message.sections.as_ref().map(IndexMap::len), Some(1));
    assert_eq!(message.tools_added.as_ref().map(Vec::len), Some(1));
}

struct Failing;

impl PromptSection for Failing {
    fn key(&self) -> &'static str {
        "flaky"
    }

    fn render<'a>(
        &'a self,
        _input: &'a PromptInput<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, Result<Option<String>, SectionError>> {
        Box::pin(async { Err("render failed".into()) })
    }
}

fn agent() -> Agent {
    resolve_agent(None, &Registry::new().snapshot(), &Settings::default(), |_, _| {})
}

#[tokio::test]
async fn a_failing_section_keeps_its_shown_text_and_tags_wrap_text() {
    let agent = agent();
    let shown = sections(&[("flaky", "<flaky>\nold\n</flaky>")]);
    let input = PromptInput {
        conversation: ConversationId(1),
        agent: &agent,
        env: None,
        shown: &shown,
    };
    let rendered: Vec<Arc<dyn PromptSection>> = vec![
        Arc::new(TextSection::new("intro", "Hello")),
        Arc::new(Failing),
        Arc::new(TextSection {
            tag: false,
            ..TextSection::new("plain", "Plain")
        }),
    ];
    let mut failures = Vec::new();
    let desired = render_sections(
        &rendered,
        &input,
        |error, section| failures.push((section.key().to_owned(), error.to_string())),
        &Context::background(),
    )
    .await
    .unwrap();
    assert_eq!(
        desired,
        sections(&[
            ("intro", "<intro>\nHello\n</intro>"),
            ("flaky", "<flaky>\nold\n</flaky>"),
            ("plain", "Plain")
        ])
    );
    assert_eq!(failures, [("flaky".to_owned(), "render failed".to_owned())]);
}
