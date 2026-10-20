//! The generation: requests, streaming partials, classification, retries, overflow, tool
//! rounds, abort and recovery.
//!
//! Ported from upstream `test/harness-generation.test.ts`, `harness-generation-recovery`,
//! `harness-live-deltas` and `spec-usage` where they apply before the tool and compaction
//! tasks land. Every case runs against the memory backend and SQLite in a temp file, with
//! a scripted model; a crash drops the Harness without closing it and reopens the storage.

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

/// [`cases!`] with tokio's clock paused, memory only (SQLite runs on its own thread).
macro_rules! paused_cases {
    ($($case:ident),* $(,)?) => {
        mod paused {
            $(#[tokio::test(start_paused = true)] async fn $case() { super::$case(crate::Backend::Memory).await; })*
        }
    };
}

mod abort;
mod basic;
mod failures;
mod hooks;
mod rounds;
mod streaming;

pub(crate) use support::Backend;
