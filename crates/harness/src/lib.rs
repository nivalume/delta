//! Durable agent harness: conversations, model turns, tool calls and application state are
//! committed to storage before they are shown, and a reopened storage resumes where it stopped.
//!
//! delta's counterpart of [`@earendil-works/pi-durable`](https://github.com/earendil-works/pi/tree/main/packages/durable).
//! Design notes live in `docs/design/` of this crate.
//!
//! Implemented so far:
//!
//! - [`records`]: ids, conversation / entry / submission / task / document records, storage
//!   writes and scan queries.
//! - [`documents`]: document definitions, typed addresses and their validation.
//! - [`storage`]: the [`Storage`](storage::Storage) contract with memory and SQLite backends.
//! - [`env`](mod@env): the execution environment (file system and shell) and its local
//!   implementation.

pub mod documents;
pub mod env;
pub mod records;
pub mod storage;
