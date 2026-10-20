//! Registry validation, publication and snapshots (upstream `harness-registry`).

use std::sync::Arc;

use super::{Extension, INSTRUCTIONS_KEY, Registry, RegistryError, RegistrySnapshot};
use crate::generation::{GENERATION_TASK, GenerationTask};
use crate::prompt::TextSection;
use crate::scheduler::{AnyTask, TaskCatalog};
use crate::tool::HarnessTool;

struct Named(delta_ai::types::Tool);

impl HarnessTool for Named {
    fn definition(&self) -> &delta_ai::types::Tool {
        &self.0
    }
}

fn tool(name: &str) -> Arc<dyn HarnessTool> {
    Arc::new(Named(delta_ai::types::Tool::new(name, name, serde_json::json!({}))))
}

#[test]
fn a_new_registry_holds_the_builtin_tasks_only() {
    let registry = Registry::new();
    let snapshot = registry.snapshot();
    assert!(snapshot.installed().is_empty());
    assert!(snapshot.task(GENERATION_TASK).is_some());
    assert_eq!(snapshot.generation(), 0);
}

#[test]
fn install_validates_names_and_keys_and_publishes_nothing_when_invalid() {
    let registry = Registry::new();
    let twice = Extension::new("x").with_tool(tool("read")).with_tool(tool("read"));
    assert!(matches!(
        registry.install(twice),
        Err(RegistryError::DuplicateTool { .. })
    ));
    let bad_key = Extension::new("x").with_section(Arc::new(TextSection::new("Bad", "")));
    assert!(matches!(
        registry.install(bad_key),
        Err(RegistryError::InvalidSectionKey { .. })
    ));
    let reserved = Extension::new("x").with_section(Arc::new(TextSection::new(INSTRUCTIONS_KEY, "")));
    assert!(matches!(
        registry.install(reserved),
        Err(RegistryError::ReservedSectionKey { .. })
    ));
    let builtin = Extension::new("x").with_task(Arc::new(GenerationTask));
    assert!(matches!(
        registry.install(builtin),
        Err(RegistryError::DuplicateTask { .. })
    ));
    assert_eq!(registry.snapshot().generation(), 0);
    assert!(registry.snapshot().installed().is_empty());
}

#[test]
fn reinstalling_replaces_in_place_and_snapshots_stay_immutable() {
    let registry = Registry::new();
    let changes = registry.subscribe();
    registry.install(Extension::new("a").with_tool(tool("one"))).unwrap();
    registry.install(Extension::new("b")).unwrap();
    let before = registry.snapshot();
    registry.install(Extension::new("a").with_tool(tool("two"))).unwrap();
    let after = registry.snapshot();
    let names = |snapshot: &RegistrySnapshot| -> Vec<String> {
        snapshot
            .installed()
            .iter()
            .map(|extension| extension.name.clone())
            .collect()
    };
    assert_eq!(names(&after), ["a", "b"]);
    let tools = |snapshot: &RegistrySnapshot| -> Vec<String> {
        snapshot.tools().map(|(_, tool)| tool.name().to_owned()).collect()
    };
    assert_eq!(tools(&before), ["one"]);
    assert_eq!(tools(&after), ["two"]);
    assert_eq!(after.generation(), 3);
    assert!(changes.has_changed().unwrap());
    registry.uninstall("a");
    assert_eq!(names(&registry.snapshot()), ["b"]);
    registry.uninstall("missing");
    assert_eq!(registry.snapshot().generation(), 4);
}

#[test]
fn the_catalog_snapshot_downcasts_and_names_installers() {
    let registry = Registry::new();
    let task: Arc<dyn AnyTask> = Arc::new(Probe);
    registry.install(Extension::new("ext").with_task(task.clone())).unwrap();
    let catalog = TaskCatalog::snapshot(&registry);
    let snapshot = RegistrySnapshot::from_catalog(&catalog).expect("a registry snapshot");
    assert_eq!(snapshot.generation(), 1);
    assert_eq!(catalog.extension_of(&task), Some("ext".to_owned()));
    let builtin = catalog.task(GENERATION_TASK).unwrap().clone();
    assert_eq!(catalog.extension_of(&builtin), None);
}

/// A task definition that never runs.
struct Probe;

impl crate::scheduler::TaskDefinition for Probe {
    type Input = delta_chord::json::Json;
    type Checkpoint = delta_chord::json::Json;
    type Result = delta_chord::json::Json;

    fn name(&self) -> &'static str {
        "test.probe"
    }

    fn version(&self) -> u32 {
        1
    }

    fn initial(&self, _input: &Self::Input) -> Self::Checkpoint {
        delta_chord::json!({ "phase": "start" })
    }

    async fn run<'a>(
        &'a self,
        _phase: Self::Checkpoint,
        _rt: &'a crate::scheduler::TaskRuntime<Self>,
        _ctx: &'a delta_chord::context::Context,
    ) -> Result<crate::scheduler::Next<Self>, crate::scheduler::TaskError> {
        Ok(crate::scheduler::Next::complete(delta_chord::json::Json::Null))
    }

    async fn abort<'a>(
        &'a self,
        _phase: Self::Checkpoint,
        _rt: &'a crate::scheduler::TaskRuntime<Self>,
        _ctx: &'a delta_chord::context::Context,
    ) -> Result<crate::scheduler::Next<Self>, crate::scheduler::TaskError> {
        Ok(crate::scheduler::Next::abort(None))
    }
}
