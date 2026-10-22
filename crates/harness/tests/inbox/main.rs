//! Submissions and the inbox: admission, request ids, steer / follow-up / write placement at
//! turn boundaries, resets and stale heads, queue modes, withdrawal and failed runs.
//!
//! Ported from upstream `test/harness-inbox.test.ts`, `harness-submissions`,
//! `harness-failed-run-inbox` and `harness-input-entries`. Every case runs against the
//! memory backend and SQLite in a temp file, with a scripted model.

#![allow(clippy::unwrap_used, clippy::too_many_lines)]

#[path = "../support/mod.rs"]
mod support;

/// Runs each case against memory and SQLite in a file, on a multi-threaded runtime.
macro_rules! cases {
    ($($case:ident),* $(,)?) => {
        mod memory {
            $(#[tokio::test(flavor = "multi_thread", worker_threads = 4)] async fn $case() { super::$case(crate::Backend::Memory).await; })*
        }
        mod sqlite {
            $(#[tokio::test(flavor = "multi_thread", worker_threads = 4)] async fn $case() { super::$case(crate::Backend::SqliteFile).await; })*
        }
    };
}

mod admission;
mod placement;
mod withdrawal;

pub(crate) use support::Backend;
