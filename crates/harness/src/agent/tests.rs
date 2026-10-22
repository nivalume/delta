//! Agent resolution (upstream `harness-registry`, the resolution half).

use std::sync::Arc;

use super::{
    AgentChange, AgentState, Change, ExtensionSelection, HarnessSettings, ModelToolFilter, Settings, ToolFilter,
    resolve_agent,
};
use crate::model::ModelRef;
use crate::prompt::TextSection;
use crate::registry::{Extension, Registry, Wrap};
use crate::tool::{Callers, HarnessTool};

struct Named {
    declaration: delta_ai::types::Tool,
    deferred: bool,
    callers: Callers,
}

impl HarnessTool for Named {
    fn definition(&self) -> &delta_ai::types::Tool {
        &self.declaration
    }

    fn deferred(&self) -> bool {
        self.deferred
    }

    fn callers(&self) -> Callers {
        self.callers
    }

    fn execute<'a>(
        &'a self,
        _arguments: delta_ai::types::JsonObject,
        _api: &'a crate::tool::ToolApi<'a>,
        _ctx: &'a delta_chord::context::Context,
    ) -> futures::future::BoxFuture<'a, Result<crate::tool::ToolExecutionResult, crate::tool::ToolError>> {
        Box::pin(async { Ok(crate::tool::ToolExecutionResult::default()) })
    }
}

fn tool(name: &str, description: &str) -> Arc<dyn HarnessTool> {
    Arc::new(Named {
        declaration: delta_ai::types::Tool::new(name, description, serde_json::json!({})),
        deferred: false,
        callers: Callers::BOTH,
    })
}

fn names(tools: &[Arc<dyn HarnessTool>]) -> Vec<String> {
    tools.iter().map(|tool| tool.name().to_owned()).collect()
}

fn registry() -> Registry {
    let registry = Registry::new();
    let deferred: Arc<dyn HarnessTool> = Arc::new(Named {
        declaration: delta_ai::types::Tool::new("search", "", serde_json::json!({})),
        deferred: true,
        callers: Callers::BOTH,
    });
    let internal: Arc<dyn HarnessTool> = Arc::new(Named {
        declaration: delta_ai::types::Tool::new("internal", "", serde_json::json!({})),
        deferred: false,
        callers: Callers::TOOLS,
    });
    registry
        .install(
            Extension::new("files")
                .with_tool(tool("read", "v1"))
                .with_tool(tool("write", ""))
                .with_section(Arc::new(TextSection::new("files", "Files"))),
        )
        .unwrap();
    registry
        .install(
            Extension::new("extra")
                .with_tool(tool("read", "v2"))
                .with_tool(deferred)
                .with_tool(internal),
        )
        .unwrap();
    registry
}

#[test]
fn every_installed_extension_by_default_with_later_tools_replacing_in_place() {
    let agent = resolve_agent(None, &registry().snapshot(), &Settings::default(), |_, _| {});
    assert_eq!(names(&agent.tools), ["read", "write"]);
    assert_eq!(agent.tools[0].definition().description, "v2");
    assert_eq!(names(&agent.deferred), ["search"]);
    assert_eq!(names(&agent.callable), ["read", "write", "search", "internal"]);
    let keys: Vec<&str> = agent.sections.iter().map(|section| section.key()).collect();
    assert_eq!(keys, ["files"]);
    assert_eq!(agent.model, None);
}

#[test]
fn selections_and_filters_apply_by_name() {
    let state = AgentState {
        model: Some(ModelRef::new("p", "m")),
        extensions: Some(ExtensionSelection::Edit {
            add: vec!["missing".to_owned()],
            remove: vec!["extra".to_owned()],
        }),
        tools: Some(ToolFilter::Remove {
            remove: vec!["write".to_owned()],
        }),
        instructions: Some("Be brief".to_owned()),
        ..AgentState::default()
    };
    let agent = resolve_agent(Some(&state), &registry().snapshot(), &Settings::default(), |_, _| {});
    assert_eq!(names(&agent.tools), ["read"]);
    assert_eq!(agent.tools[0].definition().description, "v1");
    let keys: Vec<&str> = agent.sections.iter().map(|section| section.key()).collect();
    assert_eq!(keys, ["files", "instructions"]);
    // The settings' default selection is the base of an edit.
    let settings = HarnessSettings {
        extensions: Some(vec!["extra".to_owned()]),
        ..HarnessSettings::default()
    }
    .resolve();
    let loaded = AgentState {
        model_tools: Some(ModelToolFilter::Edit {
            remove: Vec::new(),
            add: vec!["search".to_owned()],
        }),
        ..AgentState::default()
    };
    let agent = resolve_agent(Some(&loaded), &registry().snapshot(), &settings, |_, _| {});
    assert_eq!(names(&agent.tools), ["read", "search"]);
    assert!(agent.deferred.is_empty());
}

#[test]
fn a_failing_or_renaming_wrapper_drops_its_target_and_is_reported() {
    let registry = registry();
    registry
        .install(
            Extension::new("wrapper")
                .with_wrap(Wrap::Tool {
                    name: "read".to_owned(),
                    wrap: Arc::new(|_| Err("no".into())),
                })
                .with_wrap(Wrap::Tool {
                    name: "write".to_owned(),
                    wrap: Arc::new(|_| Ok(tool("renamed", ""))),
                })
                .with_wrap(Wrap::Tool {
                    name: "absent".to_owned(),
                    wrap: Arc::new(|_| Err("never called".into())),
                }),
        )
        .unwrap();
    let mut reports = Vec::new();
    let agent = resolve_agent(None, &registry.snapshot(), &Settings::default(), |error, extension| {
        reports.push((extension.to_owned(), error.to_string()));
    });
    assert!(names(&agent.tools).is_empty());
    assert_eq!(reports.len(), 2);
    assert_eq!(reports[0], ("wrapper".to_owned(), "no".to_owned()));
}

#[test]
fn changes_keep_clear_or_set_fields() {
    let mut state = AgentState {
        model: Some(ModelRef::new("p", "m")),
        cwd: Some("/a".to_owned()),
        ..AgentState::default()
    };
    AgentChange {
        model: Change::Clear,
        instructions: Change::Set("x".to_owned()),
        ..AgentChange::default()
    }
    .apply(&mut state);
    assert_eq!(state.model, None);
    assert_eq!(state.cwd.as_deref(), Some("/a"));
    assert_eq!(state.instructions.as_deref(), Some("x"));
    let json = serde_json::to_value(&AgentState {
        tools: Some(ToolFilter::Exactly(vec!["read".to_owned()])),
        model_tools: Some(ModelToolFilter::Edit {
            remove: Vec::new(),
            add: vec!["search".to_owned()],
        }),
        ..AgentState::default()
    })
    .unwrap();
    assert_eq!(
        json,
        serde_json::json!({ "tools": ["read"], "model_tools": { "add": ["search"] } })
    );
}
