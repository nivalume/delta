//! [`Extension`]: the unit a host installs. Tools, prompt sections, hooks, wrappers and task
//! definitions, all immutable once installed.

use std::any::Any;
use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;

use crate::prompt::PromptSection;
use crate::scheduler::AnyTask;
use crate::tool::HarnessTool;

/// Why a wrapper could not wrap its target; the target is dropped for that resolution.
pub type WrapError = Box<dyn StdError + Send + Sync>;

/// Wraps a tool by name; must keep its name.
pub type ToolWrapper = Arc<dyn Fn(Arc<dyn HarnessTool>) -> Result<Arc<dyn HarnessTool>, WrapError> + Send + Sync>;

/// Wraps a prompt section by key; must keep its key.
pub type SectionWrapper =
    Arc<dyn Fn(Arc<dyn PromptSection>) -> Result<Arc<dyn PromptSection>, WrapError> + Send + Sync>;

/// A wrapper an extension applies to the selected tools or sections of every resolution.
/// Wrappers are pure; one that finds no target does nothing.
#[derive(Clone)]
pub enum Wrap {
    Tool { name: String, wrap: ToolWrapper },
    Section { key: String, wrap: SectionWrapper },
}

impl fmt::Debug for Wrap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tool { name, .. } => f.debug_struct("Tool").field("name", name).finish_non_exhaustive(),
            Self::Section { key, .. } => f.debug_struct("Section").field("key", key).finish_non_exhaustive(),
        }
    }
}

/// One hook object registered for a task name: an `Arc<H>` of the task's hook trait `H`
/// (`dyn GenerationHooks`, say), downcast at dispatch by the task's
/// [`TaskHooks::Hooks`](super::TaskHooks::Hooks) type.
#[derive(Clone)]
pub struct HookRegistration {
    task: String,
    /// An `Arc<H>`, boxed as `Any`.
    hooks: Arc<dyn Any + Send + Sync>,
}

impl HookRegistration {
    /// Hooks of the task named `task`.
    pub fn new<H: ?Sized + Send + Sync + 'static>(task: impl Into<String>, hooks: Arc<H>) -> Self {
        Self {
            task: task.into(),
            hooks: Arc::new(hooks),
        }
    }

    /// The task name the hooks are for.
    pub fn task(&self) -> &str {
        &self.task
    }

    /// The hook object as an `H`, when it was registered as one.
    pub fn get<H: ?Sized + Send + Sync + 'static>(&self) -> Option<Arc<H>> {
        self.hooks.downcast_ref::<Arc<H>>().cloned()
    }
}

impl fmt::Debug for HookRegistration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HookRegistration")
            .field("task", &self.task)
            .finish_non_exhaustive()
    }
}

/// Code a host installs into a [`Registry`](super::Registry) under a unique name. A
/// conversation selects extensions by name in `delta.agent`; resolution skips names that are
/// not installed, so a stored name takes effect again once its extension is.
#[derive(Clone, Default)]
pub struct Extension {
    pub name: String,
    pub tools: Vec<Arc<dyn HarnessTool>>,
    pub sections: Vec<Arc<dyn PromptSection>>,
    pub hooks: Vec<HookRegistration>,
    pub wraps: Vec<Wrap>,
    pub tasks: Vec<Arc<dyn AnyTask>>,
}

impl Extension {
    /// An extension named `name` with nothing in it.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    #[must_use]
    pub fn with_tool(mut self, tool: Arc<dyn HarnessTool>) -> Self {
        self.tools.push(tool);
        self
    }

    #[must_use]
    pub fn with_section(mut self, section: Arc<dyn PromptSection>) -> Self {
        self.sections.push(section);
        self
    }

    /// Hooks `hooks` (an `Arc` of the task's hook trait) for the task named `task`.
    #[must_use]
    pub fn with_hooks<H: ?Sized + Send + Sync + 'static>(mut self, task: &str, hooks: Arc<H>) -> Self {
        self.hooks.push(HookRegistration::new(task, hooks));
        self
    }

    #[must_use]
    pub fn with_wrap(mut self, wrap: Wrap) -> Self {
        self.wraps.push(wrap);
        self
    }

    #[must_use]
    pub fn with_task(mut self, task: Arc<dyn AnyTask>) -> Self {
        self.tasks.push(task);
        self
    }
}

impl fmt::Debug for Extension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tools: Vec<&str> = self.tools.iter().map(|tool| tool.name()).collect();
        let sections: Vec<&str> = self.sections.iter().map(|section| section.key()).collect();
        let tasks: Vec<&str> = self.tasks.iter().map(|task| task.name()).collect();
        f.debug_struct("Extension")
            .field("name", &self.name)
            .field("tools", &tools)
            .field("sections", &sections)
            .field("hooks", &self.hooks)
            .field("wraps", &self.wraps)
            .field("tasks", &tasks)
            .finish()
    }
}
