//! Observation: the conversation view mount, the agent events derived from it, and the
//! task graph.
//!
//! The central check: a view advanced commit by commit equals the view rebuilt from
//! storage afterwards. Every case runs against memory and SQLite in a temp file.

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

mod events;
mod graph;
mod view;

pub(crate) use support::Backend;
