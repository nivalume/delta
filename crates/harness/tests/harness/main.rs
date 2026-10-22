//! The Harness surface: opening paused, conversations and their creation hook, agent
//! configuration and resolution, forks, the prompt across registry changes, inspection,
//! reports and close.
//!
//! Ported from upstream `test/harness-conversations.test.ts`, `harness-registry`,
//! `harness-lifecycle`, `harness-inspect` and `harness-reports` where they apply before the
//! tool and compaction tasks land. Every case runs against the memory backend and SQLite in
//! a temp file, with a scripted model.

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

mod conversations;
mod lifecycle;
mod prompt;

pub(crate) use support::Backend;
