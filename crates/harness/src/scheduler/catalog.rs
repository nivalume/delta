//! Where the scheduler finds task definitions: the [`TaskCatalog`] the layer above supplies
//! (the agent layer's registry), its immutable [`CatalogSnapshot`]s, and the
//! [`MemoryCatalog`] for kernels and tests without a registry.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, PoisonError, RwLock};

use tokio::sync::watch;

use super::erased::AnyTask;

/// A source of task definitions by name. Publication is a new snapshot and a bump of
/// [`subscribe`](Self::subscribe); the scheduler reconsiders blocked tasks at each.
pub trait TaskCatalog: Send + Sync + 'static {
    /// The current definitions; an immutable value.
    fn snapshot(&self) -> Arc<dyn CatalogSnapshot>;
    /// Changes with every publication.
    fn subscribe(&self) -> watch::Receiver<u64>;
}

/// One immutable publication of a [`TaskCatalog`]. The scheduler takes one per phase and
/// hands it to the phase's runtime; the layer above downcasts it (`&dyn Any`) to its own
/// snapshot type.
pub trait CatalogSnapshot: Any + Send + Sync {
    /// Increases with every publication; a cheap check before comparing definitions.
    fn generation(&self) -> u64;
    /// The definition stored tasks of `name` run under.
    fn task(&self, name: &str) -> Option<&Arc<dyn AnyTask>>;
    /// The extension that installed `task`, named in reports; none for built-ins.
    fn extension_of(&self, task: &Arc<dyn AnyTask>) -> Option<String> {
        let _ = task;
        None
    }
}

impl fmt::Debug for dyn CatalogSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CatalogSnapshot")
            .field("generation", &self.generation())
            .finish_non_exhaustive()
    }
}

/// A catalog held in memory: definitions by name, replaced or removed at will. The agent
/// layer uses its registry instead; tests and kernel-only hosts use this one.
pub struct MemoryCatalog {
    current: RwLock<Arc<MemorySnapshot>>,
    generation: watch::Sender<u64>,
}

struct MemorySnapshot {
    generation: u64,
    tasks: HashMap<String, Arc<dyn AnyTask>>,
}

impl MemoryCatalog {
    /// A catalog of `tasks`.
    pub fn new(tasks: impl IntoIterator<Item = Arc<dyn AnyTask>>) -> Self {
        let tasks = tasks.into_iter().map(|task| (task.name().to_owned(), task)).collect();
        Self {
            current: RwLock::new(Arc::new(MemorySnapshot { generation: 0, tasks })),
            generation: watch::Sender::new(0),
        }
    }

    /// Installs `task` under its name, replacing the definition there.
    pub fn install(&self, task: Arc<dyn AnyTask>) {
        self.publish(|tasks| {
            tasks.insert(task.name().to_owned(), task);
        });
    }

    /// Removes the definition of `name`.
    pub fn remove(&self, name: &str) {
        self.publish(|tasks| {
            tasks.remove(name);
        });
    }

    fn publish(&self, change: impl FnOnce(&mut HashMap<String, Arc<dyn AnyTask>>)) {
        let generation = {
            let mut current = self.current.write().unwrap_or_else(PoisonError::into_inner);
            let mut tasks = current.tasks.clone();
            change(&mut tasks);
            let generation = current.generation + 1;
            *current = Arc::new(MemorySnapshot { generation, tasks });
            generation
        };
        self.generation.send_replace(generation);
    }
}

impl Default for MemoryCatalog {
    fn default() -> Self {
        Self::new([])
    }
}

impl TaskCatalog for MemoryCatalog {
    fn snapshot(&self) -> Arc<dyn CatalogSnapshot> {
        self.current.read().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn subscribe(&self) -> watch::Receiver<u64> {
        self.generation.subscribe()
    }
}

impl CatalogSnapshot for MemorySnapshot {
    fn generation(&self) -> u64 {
        self.generation
    }

    fn task(&self, name: &str) -> Option<&Arc<dyn AnyTask>> {
        self.tasks.get(name)
    }
}

impl fmt::Debug for MemoryCatalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let current = self.current.read().unwrap_or_else(PoisonError::into_inner);
        let mut names: Vec<&str> = current.tasks.keys().map(String::as_str).collect();
        names.sort_unstable();
        f.debug_struct("MemoryCatalog")
            .field("generation", &current.generation)
            .field("tasks", &names)
            .finish()
    }
}
