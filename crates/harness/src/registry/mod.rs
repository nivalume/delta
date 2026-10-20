//! The registry of extensions: the application-owned code (tools, prompt sections, hooks,
//! wrappers, task definitions) a Harness resolves stored names against.
//!
//! Nothing here is persisted. A conversation names its extensions in `delta.agent`; each
//! phase of a task resolves those names against the [`RegistrySnapshot`] the scheduler took
//! for it, so a restart or an in-process reinstall finds the same names and runs whatever is
//! installed now. The [`Registry`] is the scheduler's [`TaskCatalog`]: it always holds the
//! built-in tasks, and every installed extension's tasks beside them.
//!
//! Corresponds to upstream `src/harness/registry.ts` (spec §7.1, §7.5). See
//! `docs/design/harness.md`, "Registry and extensions".

mod extension;
mod hooks;

use std::any::Any;
use std::collections::HashSet;
use std::fmt;
use std::sync::{Arc, PoisonError, RwLock};

use indexmap::IndexMap;
use tokio::sync::watch;

pub use extension::{Extension, HookRegistration, SectionWrapper, ToolWrapper, Wrap, WrapError};
pub use hooks::{GenerationHooks, HookApi, HookError, HookHost, HookResult, HookRunner, RequestMessages, TaskHooks};

use crate::prompt::PromptSection;
use crate::scheduler::{AnyTask, CatalogSnapshot, TaskCatalog};
use crate::tool::HarnessTool;

/// The section key reserved for the agent's `instructions`.
pub const INSTRUCTIONS_KEY: &str = "instructions";

/// Why [`Registry::install`] refused an extension; nothing was published.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum RegistryError {
    #[error("Extension {extension} has two tools named {tool}")]
    DuplicateTool { extension: String, tool: String },
    #[error("Extension {extension} has two sections with key {key}")]
    DuplicateSection { extension: String, key: String },
    #[error("Section key {key:?} of extension {extension} must match ^[a-z][a-z0-9_-]*$")]
    InvalidSectionKey { extension: String, key: String },
    #[error("Section key {key} of extension {extension} is reserved for the agent's instructions")]
    ReservedSectionKey { extension: String, key: String },
    #[error("Task {task} of extension {extension} is already installed")]
    DuplicateTask { extension: String, task: String },
}

/// One immutable publication of a registry.
struct SnapshotState {
    generation: u64,
    extensions: Vec<Arc<Extension>>,
    /// Built-ins first, then each extension's tasks in install order, with their installer.
    tasks: IndexMap<String, (Arc<dyn AnyTask>, Option<String>)>,
}

impl SnapshotState {
    fn build(
        generation: u64,
        builtins: &[Arc<dyn AnyTask>],
        extensions: Vec<Arc<Extension>>,
    ) -> Result<Self, RegistryError> {
        let mut tasks = IndexMap::new();
        for task in builtins {
            tasks.insert(task.name().to_owned(), (task.clone(), None));
        }
        for extension in &extensions {
            for task in &extension.tasks {
                let name = task.name().to_owned();
                if tasks.contains_key(&name) {
                    return Err(RegistryError::DuplicateTask {
                        extension: extension.name.clone(),
                        task: name,
                    });
                }
                tasks.insert(name, (task.clone(), Some(extension.name.clone())));
            }
        }
        Ok(Self {
            generation,
            extensions,
            tasks,
        })
    }
}

impl CatalogSnapshot for SnapshotState {
    fn generation(&self) -> u64 {
        self.generation
    }

    fn task(&self, name: &str) -> Option<&Arc<dyn AnyTask>> {
        self.tasks.get(name).map(|(task, _)| task)
    }

    fn extension_of(&self, task: &Arc<dyn AnyTask>) -> Option<String> {
        let (installed, extension) = self.tasks.get(task.name())?;
        std::ptr::addr_eq(Arc::as_ptr(installed), Arc::as_ptr(task))
            .then(|| extension.clone())
            .flatten()
    }
}

/// An immutable registry state: the installed extensions in install order and every task
/// definition. Cheap to clone. The scheduler takes one per phase and hands it down; handlers
/// and hooks never take their own.
#[derive(Clone)]
pub struct RegistrySnapshot(Arc<SnapshotState>);

impl RegistrySnapshot {
    /// The registry snapshot behind a scheduler catalog snapshot, when the catalog is a
    /// [`Registry`].
    pub fn from_catalog(snapshot: &Arc<dyn CatalogSnapshot>) -> Option<Self> {
        let any: Arc<dyn Any + Send + Sync> = snapshot.clone();
        any.downcast::<SnapshotState>().ok().map(Self)
    }

    /// Increases with every publication.
    pub fn generation(&self) -> u64 {
        self.0.generation
    }

    /// The installed extensions, in install order.
    pub fn installed(&self) -> &[Arc<Extension>] {
        &self.0.extensions
    }

    pub fn extension(&self, name: &str) -> Option<&Arc<Extension>> {
        self.0.extensions.iter().find(|extension| extension.name == name)
    }

    /// Every installed tool with its extension, in install order; names may repeat.
    pub fn tools(&self) -> impl Iterator<Item = (&Arc<Extension>, &Arc<dyn HarnessTool>)> {
        self.0
            .extensions
            .iter()
            .flat_map(|extension| extension.tools.iter().map(move |tool| (extension, tool)))
    }

    /// Every installed section with its extension, in install order; keys may repeat.
    pub fn sections(&self) -> impl Iterator<Item = (&Arc<Extension>, &Arc<dyn PromptSection>)> {
        self.0
            .extensions
            .iter()
            .flat_map(|extension| extension.sections.iter().map(move |section| (extension, section)))
    }

    /// Every task definition, built-ins first.
    pub fn tasks(&self) -> impl Iterator<Item = &Arc<dyn AnyTask>> {
        self.0.tasks.values().map(|(task, _)| task)
    }

    pub fn task(&self, name: &str) -> Option<&Arc<dyn AnyTask>> {
        self.0.task(name)
    }
}

impl fmt::Debug for RegistrySnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self
            .installed()
            .iter()
            .map(|extension| extension.name.as_str())
            .collect();
        f.debug_struct("RegistrySnapshot")
            .field("generation", &self.generation())
            .field("installed", &names)
            .finish_non_exhaustive()
    }
}

struct Inner {
    builtins: Vec<Arc<dyn AnyTask>>,
    current: RwLock<RegistrySnapshot>,
    generation: watch::Sender<u64>,
}

/// The application-owned registry of extensions. Cheap to clone; every clone is the same
/// registry. Publication is synchronous, runs no extension code and replaces one extension
/// whole at its install position.
#[derive(Clone)]
pub struct Registry {
    inner: Arc<Inner>,
}

impl Registry {
    /// A registry holding the built-in tasks and no extension.
    pub fn new() -> Self {
        Self::with_builtins(crate::generation::builtin_tasks())
    }

    fn with_builtins(builtins: Vec<Arc<dyn AnyTask>>) -> Self {
        let empty = SnapshotState {
            generation: 0,
            extensions: Vec::new(),
            tasks: builtins
                .iter()
                .map(|task| (task.name().to_owned(), (task.clone(), None)))
                .collect(),
        };
        Self {
            inner: Arc::new(Inner {
                builtins,
                current: RwLock::new(RegistrySnapshot(Arc::new(empty))),
                generation: watch::Sender::new(0),
            }),
        }
    }

    /// Installs `extension`, replacing an installed one of the same name in place. The
    /// registry as it would be after the change is validated first; an invalid one publishes
    /// nothing.
    pub fn install(&self, extension: Extension) -> Result<(), RegistryError> {
        validate(&extension)?;
        let extension = Arc::new(extension);
        self.publish(
            |extensions| match extensions.iter().position(|installed| installed.name == extension.name) {
                Some(index) => extensions[index] = extension,
                None => extensions.push(extension),
            },
        )
    }

    /// Removes the extension named `name`; nothing when none is installed.
    pub fn uninstall(&self, name: &str) {
        if self.snapshot().extension(name).is_none() {
            return;
        }
        // Removing an extension cannot create a task name collision.
        let _ = self.publish(|extensions| extensions.retain(|installed| installed.name != name));
    }

    /// The current state.
    pub fn snapshot(&self) -> RegistrySnapshot {
        self.inner
            .current
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Changes with every publication.
    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.inner.generation.subscribe()
    }

    fn publish(&self, change: impl FnOnce(&mut Vec<Arc<Extension>>)) -> Result<(), RegistryError> {
        let generation = {
            let mut current = self.inner.current.write().unwrap_or_else(PoisonError::into_inner);
            let mut extensions = current.0.extensions.clone();
            change(&mut extensions);
            let generation = current.0.generation + 1;
            let next = SnapshotState::build(generation, &self.inner.builtins, extensions)?;
            *current = RegistrySnapshot(Arc::new(next));
            generation
        };
        self.inner.generation.send_replace(generation);
        Ok(())
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Registry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Registry").field(&self.snapshot()).finish()
    }
}

impl TaskCatalog for Registry {
    fn snapshot(&self) -> Arc<dyn CatalogSnapshot> {
        self.snapshot().0
    }

    fn subscribe(&self) -> watch::Receiver<u64> {
        Registry::subscribe(self)
    }
}

/// Whether `key` matches `^[a-z][a-z0-9_-]*$`.
fn valid_key(key: &str) -> bool {
    let mut chars = key.chars();
    chars.next().is_some_and(|first| first.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
}

/// Unique tool names and section keys within the extension; valid, unreserved keys.
fn validate(extension: &Extension) -> Result<(), RegistryError> {
    let name = || extension.name.clone();
    let mut tools = HashSet::new();
    for tool in &extension.tools {
        if !tools.insert(tool.name()) {
            return Err(RegistryError::DuplicateTool {
                extension: name(),
                tool: tool.name().to_owned(),
            });
        }
    }
    let mut keys = HashSet::new();
    for section in &extension.sections {
        let key = section.key();
        if !valid_key(key) {
            return Err(RegistryError::InvalidSectionKey {
                extension: name(),
                key: key.to_owned(),
            });
        }
        if key == INSTRUCTIONS_KEY {
            return Err(RegistryError::ReservedSectionKey {
                extension: name(),
                key: key.to_owned(),
            });
        }
        if !keys.insert(key) {
            return Err(RegistryError::DuplicateSection {
                extension: name(),
                key: key.to_owned(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
