//! The file mutation queue: `write` and `edit` of one file run one at a time.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError, Weak};

use delta_chord::context::Context;
use futures::future::BoxFuture;

use crate::env::{ExecutionEnv, FileError, FileErrorCode, FileResult};

/// A file: its file system id and canonical path.
type Key = (String, String);

/// Serializes the read-modify-write of one file: calls with the same file system id and
/// canonical path, whichever environment object they got, run in the order their keys
/// resolved; other files and other file systems never wait. Not a lock against `bash` or
/// other processes.
///
/// Clones share their slots. [`coding_tools`](super::coding_tools) gives its `write` and
/// `edit` one queue; tools that build their own share it explicitly.
#[derive(Clone, Default)]
pub struct FileMutationQueue {
    slots: Arc<Mutex<HashMap<Key, Weak<tokio::sync::Mutex<()>>>>>,
}

impl fmt::Debug for FileMutationQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner).len();
        f.debug_struct("FileMutationQueue").field("slots", &slots).finish()
    }
}

impl FileMutationQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Runs `mutation` once every earlier mutation of the file at `path` has finished.
    /// Fails `aborted` when `ctx` is cancelled while it waits.
    pub async fn run<T>(
        &self,
        env: &dyn ExecutionEnv,
        path: &str,
        ctx: &Context,
        mutation: impl Future<Output = T>,
    ) -> FileResult<T> {
        let key = mutation_key(env, path, ctx).await?;
        let slot = self.slot(key);
        let _held = ctx.run(slot.lock()).await.map_err(|_| FileError::aborted())?;
        Ok(mutation.await)
    }

    /// The file's lock; unused locks are dropped as new ones are handed out.
    fn slot(&self, key: Key) -> Arc<tokio::sync::Mutex<()>> {
        let mut slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(slot) = slots.get(&key).and_then(Weak::upgrade) {
            return slot;
        }
        slots.retain(|_, slot| slot.strong_count() > 0);
        let slot = Arc::new(tokio::sync::Mutex::new(()));
        slots.insert(key, Arc::downgrade(&slot));
        slot
    }
}

async fn mutation_key(env: &dyn ExecutionEnv, path: &str, ctx: &Context) -> FileResult<Key> {
    let absolute = env.absolute_path(path)?;
    Ok((env.id().to_owned(), canonical(env, absolute, ctx).await?))
}

/// The canonical path; for a file that does not exist yet, its canonical parent joined with
/// its name, so a `write` that creates a file and a later mutation of it share one key even
/// under a symbolic link. `not_supported` falls back to the absolute path.
fn canonical<'a>(env: &'a dyn ExecutionEnv, absolute: String, ctx: &'a Context) -> BoxFuture<'a, FileResult<String>> {
    Box::pin(async move {
        match env.canonical_path(&absolute, ctx).await {
            Ok(path) => return Ok(path),
            Err(error) if error.code == FileErrorCode::NotSupported => return Ok(absolute),
            Err(error) if error.code != FileErrorCode::NotFound => return Err(error),
            Err(_) => {}
        }
        // The file system splits the path, so a name may hold characters that are
        // separators elsewhere.
        let parent = env.join_path(&[&absolute, ".."])?;
        if parent == absolute || !absolute.starts_with(&parent) {
            return Ok(absolute);
        }
        let separator = usize::from(!parent.ends_with(['/', '\\']));
        let Some(name) = absolute.get(parent.len() + separator..).map(str::to_owned) else {
            return Ok(absolute);
        };
        let parent = canonical(env, parent, ctx).await?;
        env.join_path(&[&parent, &name])
    })
}
