//! Per-session cleanup registry. Port of upstream `src/session-resources.ts`.
//!
//! APIs that keep resources per session (the OpenAI Codex WebSocket cache) register a
//! cleanup here. Apps call [`cleanup_session_resources`] when a session ends (with its id)
//! or when shutting down (without one).

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

/// Error a cleanup may return.
pub type SessionResourceCleanupError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Releases the resources of one session (`Some(id)`) or of every session (`None`)
/// (upstream `SessionResourceCleanup`).
pub type SessionResourceCleanup = Arc<dyn Fn(Option<&str>) -> Result<(), SessionResourceCleanupError> + Send + Sync>;

static NEXT_ID: AtomicU64 = AtomicU64::new(0);
static CLEANUPS: Mutex<Vec<(u64, SessionResourceCleanup)>> = Mutex::new(Vec::new());

/// Handle of a registered cleanup (upstream returns an unregister function).
#[derive(Debug)]
pub struct SessionResourceRegistration {
    id: u64,
}

impl SessionResourceRegistration {
    /// Remove the cleanup from the registry. Unregistering twice is a no-op.
    pub fn unregister(&self) {
        CLEANUPS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .retain(|(id, _)| *id != self.id);
    }
}

/// Register `cleanup` (upstream `registerSessionResourceCleanup`). Registering the same
/// `Arc` again keeps a single entry, as upstream's `Set` does; the returned handle then
/// refers to that entry.
pub fn register_session_resource_cleanup(cleanup: SessionResourceCleanup) -> SessionResourceRegistration {
    let mut cleanups = CLEANUPS.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((id, _)) = cleanups.iter().find(|(_, existing)| Arc::ptr_eq(existing, &cleanup)) {
        return SessionResourceRegistration { id: *id };
    }
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    cleanups.push((id, cleanup));
    SessionResourceRegistration { id }
}

/// The failures of [`cleanup_session_resources`] (upstream `AggregateError`).
#[derive(Debug)]
pub struct SessionResourcesCleanupError {
    pub errors: Vec<SessionResourceCleanupError>,
}

impl fmt::Display for SessionResourcesCleanupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Failed to cleanup session resources")
    }
}

impl std::error::Error for SessionResourcesCleanupError {}

/// Run every registered cleanup for `session_id` (`None`: every session), in registration
/// order (upstream `cleanupSessionResources`). All cleanups run even when some fail; the
/// failures are returned together.
pub fn cleanup_session_resources(session_id: Option<&str>) -> Result<(), SessionResourcesCleanupError> {
    // Run outside the lock so a cleanup may register or unregister cleanups.
    let cleanups: Vec<SessionResourceCleanup> = CLEANUPS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .map(|(_, cleanup)| cleanup.clone())
        .collect();
    let errors: Vec<SessionResourceCleanupError> = cleanups
        .iter()
        .filter_map(|cleanup| cleanup(session_id).err())
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(SessionResourcesCleanupError { errors })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    #[test]
    fn runs_every_cleanup_and_collects_failures() {
        let calls = Arc::new(StdMutex::new(Vec::new()));
        let recorded = calls.clone();
        let ok: SessionResourceCleanup = Arc::new(move |session: Option<&str>| {
            recorded.lock().unwrap().push(session.map(str::to_owned));
            Ok(())
        });
        let failing: SessionResourceCleanup = Arc::new(|session: Option<&str>| {
            if session == Some("session-resources-test") {
                Err("boom".into())
            } else {
                Ok(())
            }
        });
        let first = register_session_resource_cleanup(ok.clone());
        let again = register_session_resource_cleanup(ok);
        let second = register_session_resource_cleanup(failing);

        let error = cleanup_session_resources(Some("session-resources-test")).unwrap_err();
        assert_eq!(error.to_string(), "Failed to cleanup session resources");
        assert_eq!(error.errors.len(), 1);
        assert_eq!(error.errors[0].to_string(), "boom");
        // The duplicate registration ran once.
        assert_eq!(*calls.lock().unwrap(), [Some("session-resources-test".to_owned())]);

        again.unregister();
        second.unregister();
        first.unregister();
        cleanup_session_resources(Some("session-resources-test")).unwrap();
        assert_eq!(calls.lock().unwrap().len(), 1);
    }
}
