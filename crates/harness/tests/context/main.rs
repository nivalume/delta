//! Context derivation over storage: fork caps, cut-off reads, and the kept ranges that let a
//! later read scan only the entries after the last one.
//!
//! Ported from upstream `test/harness-context.test.ts` (the storage half) and the kept-range
//! half of `harness-scheduler-retention`. The derivation rules themselves are unit tests of
//! `context`. Every case runs against the memory backend and SQLite in a temp file.

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

mod ranges;
mod reads;

pub(crate) use support::Backend;
