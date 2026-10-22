//! Agent resolution: a stored `delta.agent`, a registry snapshot and the settings into the
//! [`Agent`] one phase runs with.

use std::fmt;
use std::sync::Arc;

use delta_ai::model::ModelThinkingLevel;
use indexmap::IndexMap;

use super::{AgentState, ExtensionSelection, ModelToolFilter, Settings, ToolFilter};
use crate::model::ModelRef;
use crate::prompt::{PromptSection, TextSection};
use crate::registry::{Extension, HookRegistration, HookRunner, INSTRUCTIONS_KEY, RegistrySnapshot, Wrap, WrapError};
use crate::tool::{Callers, HarnessTool};

/// A resolved agent; tasks hold one `Arc<Agent>` per phase.
#[derive(Clone)]
pub struct Agent {
    /// The stored model; a request without one fails `no_model`.
    pub model: Option<ModelRef>,
    pub thinking_level: ModelThinkingLevel,
    /// The selected installed extensions, in selection order.
    pub extensions: Vec<Arc<Extension>>,
    /// The tools offered to the model, in order.
    pub tools: Vec<Arc<dyn HarnessTool>>,
    /// The enabled tools other tools may call (nested calls resolve among these).
    pub callable: Vec<Arc<dyn HarnessTool>>,
    /// Model-callable deferred tools not offered yet.
    pub deferred: Vec<Arc<dyn HarnessTool>>,
    /// The extensions' sections, then `instructions`.
    pub sections: Vec<Arc<dyn PromptSection>>,
    pub instructions: Option<String>,
    pub cwd: Option<String>,
}

impl Agent {
    /// The offered tool named `name`.
    pub fn tool(&self, name: &str) -> Option<&Arc<dyn HarnessTool>> {
        self.tools.iter().find(|tool| tool.name() == name)
    }

    /// The selected extensions' hooks for the task named `task`, in extension order, as the
    /// task's hook trait `H`; registrations of another type are skipped.
    pub fn hooks<H: ?Sized + Send + Sync + 'static>(&self, task: &str) -> HookRunner<H> {
        let handlers = self
            .extensions
            .iter()
            .flat_map(|extension| {
                extension
                    .hooks
                    .iter()
                    .filter(|registration| registration.task() == task)
                    .filter_map(HookRegistration::get::<H>)
                    .map(|hooks| (extension.name.clone(), hooks))
            })
            .collect();
        HookRunner::new(handlers)
    }

    /// The extension whose code the section `key` runs: the last selected one that provides
    /// or wraps it.
    pub fn section_extension(&self, key: &str) -> Option<&str> {
        self.extensions
            .iter()
            .rev()
            .find(|extension| {
                extension.sections.iter().any(|section| section.key() == key)
                    || extension
                        .wraps
                        .iter()
                        .any(|wrap| matches!(wrap, Wrap::Section { key: wrapped, .. } if wrapped == key))
            })
            .map(|extension| extension.name.as_str())
    }
}

impl fmt::Debug for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names =
            |tools: &[Arc<dyn HarnessTool>]| tools.iter().map(|tool| tool.name().to_owned()).collect::<Vec<_>>();
        let extensions: Vec<&str> = self
            .extensions
            .iter()
            .map(|extension| extension.name.as_str())
            .collect();
        let sections: Vec<&str> = self.sections.iter().map(|section| section.key()).collect();
        f.debug_struct("Agent")
            .field("model", &self.model)
            .field("thinking_level", &self.thinking_level)
            .field("extensions", &extensions)
            .field("tools", &names(&self.tools))
            .field("callable", &names(&self.callable))
            .field("deferred", &names(&self.deferred))
            .field("sections", &sections)
            .field("cwd", &self.cwd)
            .finish_non_exhaustive()
    }
}

/// Resolves a stored agent (absent: every field unset) against `snapshot` and `settings`.
/// A wrapper that fails or renames its target drops the target for this resolution and is
/// given to `report` with its extension's name.
pub fn resolve_agent(
    state: Option<&AgentState>,
    snapshot: &RegistrySnapshot,
    settings: &Settings,
    mut report: impl FnMut(WrapError, &str),
) -> Agent {
    let empty = AgentState::default();
    let state = state.unwrap_or(&empty);
    let extensions = select_extensions(state.extensions.as_ref(), snapshot, settings);

    let mut tools: IndexMap<String, Arc<dyn HarnessTool>> = IndexMap::new();
    let mut sections: IndexMap<String, Arc<dyn PromptSection>> = IndexMap::new();
    for extension in &extensions {
        for tool in &extension.tools {
            tools.insert(tool.name().to_owned(), tool.clone());
        }
        for section in &extension.sections {
            sections.insert(section.key().to_owned(), section.clone());
        }
    }
    for extension in &extensions {
        for wrap in &extension.wraps {
            let mut failed = |error| report(error, &extension.name);
            match wrap {
                Wrap::Tool { name, wrap } => apply_wrap(&mut tools, name, &**wrap, |tool| tool.name(), &mut failed),
                Wrap::Section { key, wrap } => {
                    apply_wrap(&mut sections, key, &**wrap, |section| section.key(), &mut failed);
                }
            }
        }
    }

    let enabled = filter_tools(tools.into_values().collect(), state.tools.as_ref());
    let model_callable: Vec<_> = enabled
        .iter()
        .filter(|tool| tool.callers().contains(Callers::MODEL))
        .cloned()
        .collect();
    let offered = offered_tools(&model_callable, state.model_tools.as_ref());
    let deferred = model_callable
        .iter()
        .filter(|tool| tool.deferred() && !offered.iter().any(|offered| Arc::ptr_eq(offered, tool)))
        .cloned()
        .collect();
    let callable = enabled
        .iter()
        .filter(|tool| tool.callers().contains(Callers::TOOLS))
        .cloned()
        .collect();

    let mut agent_sections: Vec<Arc<dyn PromptSection>> = sections.into_values().collect();
    if let Some(instructions) = &state.instructions {
        agent_sections.push(Arc::new(TextSection::new(INSTRUCTIONS_KEY, instructions.clone())));
    }
    Agent {
        model: state.model.clone(),
        thinking_level: state.thinking_level.unwrap_or(ModelThinkingLevel::Off),
        extensions,
        tools: offered,
        callable,
        deferred,
        sections: agent_sections,
        instructions: state.instructions.clone(),
        cwd: state.cwd.clone(),
    }
}

/// The selected installed extensions: the stored list, or the default selection (the
/// settings' or every installed one) edited by `add` and `remove`. Duplicates keep their
/// first position; names that are not installed are skipped.
fn select_extensions(
    stored: Option<&ExtensionSelection>,
    snapshot: &RegistrySnapshot,
    settings: &Settings,
) -> Vec<Arc<Extension>> {
    let selected: Vec<String> = match stored {
        Some(ExtensionSelection::Exactly(names)) => names.clone(),
        edit => {
            let base: Vec<String> = settings.extensions.clone().unwrap_or_else(|| {
                snapshot
                    .installed()
                    .iter()
                    .map(|extension| extension.name.clone())
                    .collect()
            });
            let (add, remove) = match edit {
                Some(ExtensionSelection::Edit { add, remove }) => (add.as_slice(), remove.as_slice()),
                _ => (&[][..], &[][..]),
            };
            base.into_iter()
                .chain(add.iter().cloned())
                .filter(|name| !remove.contains(name))
                .collect()
        }
    };
    let mut extensions: Vec<Arc<Extension>> = Vec::new();
    for name in selected {
        if extensions.iter().any(|extension| extension.name == name) {
            continue;
        }
        if let Some(extension) = snapshot.extension(&name) {
            extensions.push(extension.clone());
        }
    }
    extensions
}

/// `tools` through the stored filter: `Exactly` keeps these names in its order, `Remove`
/// drops names.
fn filter_tools(tools: Vec<Arc<dyn HarnessTool>>, filter: Option<&ToolFilter>) -> Vec<Arc<dyn HarnessTool>> {
    match filter {
        None => tools,
        Some(ToolFilter::Exactly(names)) => pick(&tools, names),
        Some(ToolFilter::Remove { remove }) => tools
            .into_iter()
            .filter(|tool| !remove.iter().any(|name| name == tool.name()))
            .collect(),
    }
}

/// The tools of `tools` named by `names`, in `names` order, each once.
fn pick(tools: &[Arc<dyn HarnessTool>], names: &[String]) -> Vec<Arc<dyn HarnessTool>> {
    let mut picked: Vec<Arc<dyn HarnessTool>> = Vec::new();
    for name in names {
        if picked.iter().any(|tool| tool.name() == name) {
            continue;
        }
        if let Some(tool) = tools.iter().find(|tool| tool.name() == name) {
            picked.push(tool.clone());
        }
    }
    picked
}

/// The tools offered to the model: `Exactly` those names (deferred ones too); `Edit` every
/// non-deferred tool but `remove`, then the deferred tools in `add` in load order.
fn offered_tools(tools: &[Arc<dyn HarnessTool>], filter: Option<&ModelToolFilter>) -> Vec<Arc<dyn HarnessTool>> {
    let (remove, add) = match filter {
        Some(ModelToolFilter::Exactly(names)) => return pick(tools, names),
        Some(ModelToolFilter::Edit { remove, add }) => (remove.as_slice(), add.as_slice()),
        None => (&[][..], &[][..]),
    };
    let removed = |tool: &Arc<dyn HarnessTool>| remove.iter().any(|name| name == tool.name());
    let mut offered: Vec<Arc<dyn HarnessTool>> = tools
        .iter()
        .filter(|tool| !tool.deferred() && !removed(tool))
        .cloned()
        .collect();
    for tool in pick(tools, add) {
        if tool.deferred() && !removed(&tool) {
            offered.push(tool);
        }
    }
    offered
}

/// Wraps the item at `target` in place; a failing or renaming wrapper drops it.
fn apply_wrap<T: ?Sized>(
    items: &mut IndexMap<String, Arc<T>>,
    target: &str,
    wrap: &(dyn Fn(Arc<T>) -> Result<Arc<T>, WrapError> + Send + Sync),
    name_of: impl Fn(&T) -> &str,
    report: &mut impl FnMut(WrapError),
) {
    let Some(item) = items.get(target).cloned() else {
        return;
    };
    match wrap(item) {
        Ok(wrapped) if name_of(&wrapped) == target => {
            items.insert(target.to_owned(), wrapped);
        }
        Ok(wrapped) => {
            items.shift_remove(target);
            report(format!("Wrapper renamed {target} to {}", name_of(&wrapped)).into());
        }
        Err(error) => {
            items.shift_remove(target);
            report(error);
        }
    }
}
