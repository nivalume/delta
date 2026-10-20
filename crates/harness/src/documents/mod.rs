//! Document definitions: what a durable JSON document is, who owns it, what it keeps, what
//! a fork does with it and when a change becomes a complete base.
//!
//! This module holds the definition layer only: the [`DocDefinition`] trait, typed
//! addresses and the checks the Session runs on typed access. Transactions over documents
//! are the session's; incarnations and materialization are storage's. Corresponds to
//! upstream `src/documents.ts`. See `docs/design/documents.md`.

mod address;
mod definition;
mod validate;

pub use address::{DocAddress, OwnerOf};
pub use definition::{
    CheckpointInfo, ConversationDoc, ConversationScope, DocDefinition, DocScope, ForkAsOf, ForkCurrent, ForkInitial,
    ForkKind, HistoryKind, Latest, MigrateError, Rewindable, SessionScope, TaskScope,
};
pub use validate::{DocumentError, Semantics, Upgraded, ValueSource, check_record, initial_value, semantics, upgrade};

#[cfg(test)]
mod tests;
