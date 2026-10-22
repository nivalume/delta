//! Per-session, in-memory services of the layers above the kernel.

use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, PoisonError, RwLock};

/// A type map of per-session, in-memory services, one value per type (like
/// `http::Extensions`). Never persisted and dropped at close; the kernel stores them and never
/// reads them.
///
/// An upper layer keeps process-local state per session here without a kernel type for it:
/// the approval waiters of the agent layer, for instance, are a service it inserts at open
/// and reads from its tasks.
#[derive(Default)]
pub struct Services {
    map: RwLock<HashMap<TypeId, Entry>>,
}

struct Entry {
    name: &'static str,
    value: Arc<dyn Any + Send + Sync>,
}

impl Services {
    /// No services.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores `service` as the `T` of this session and returns the one it replaces.
    pub fn insert<T: Send + Sync + 'static>(&self, service: Arc<T>) -> Option<Arc<T>> {
        let entry = Entry {
            name: type_name::<T>(),
            value: service,
        };
        let previous = self
            .map
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(TypeId::of::<T>(), entry);
        previous.and_then(|entry| entry.value.downcast().ok())
    }

    /// The `T` of this session.
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        let map = self.map.read().unwrap_or_else(PoisonError::into_inner);
        let value = map.get(&TypeId::of::<T>())?.value.clone();
        value.downcast().ok()
    }

    /// Removes the `T` of this session.
    pub fn remove<T: Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        let previous = self
            .map
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&TypeId::of::<T>());
        previous.and_then(|entry| entry.value.downcast().ok())
    }

    /// Drops every service (at close).
    pub(super) fn clear(&self) {
        // Dropped outside the lock: a service's `Drop` may read the map.
        let map = std::mem::take(&mut *self.map.write().unwrap_or_else(PoisonError::into_inner));
        drop(map);
    }
}

impl fmt::Debug for Services {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let map = self.map.read().unwrap_or_else(PoisonError::into_inner);
        let mut names: Vec<&str> = map.values().map(|entry| entry.name).collect();
        names.sort_unstable();
        f.debug_struct("Services").field("types", &names).finish()
    }
}
