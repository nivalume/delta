//! JSON revisions with op batches, replicated state, explicit contexts, bounded output and
//! SQLite storage helpers.
//!
//! delta's counterpart of the base library of
//! [`@earendil-works/chord`](https://github.com/earendil-works/pi/tree/main/packages/chord).
//! Design notes live in `docs/design/` of this crate.

pub mod context;
pub mod delta;
pub mod json;
pub mod output;
#[cfg(feature = "sqlite")]
pub mod sqlite;
pub mod state;
