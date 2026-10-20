//! Record ids and commit sequences: `u64` newtypes in one global numeric namespace.

use std::fmt;
use std::hash::Hash;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

mod sealed {
    pub trait Sealed {}
}

/// A freshly minted id before it has a kind. [`Storage::mint_id`](crate::storage::Storage::mint_id)
/// returns one; the Session types it with [`RecordId::from_raw`] or `From`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RawId(pub u64);

impl fmt::Display for RawId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// The five record id newtypes. Sealed: every id lives in the same namespace, so a new kind
/// of id is a storage change, not an extension point.
pub trait RecordId:
    Copy
    + Ord
    + Hash
    + fmt::Debug
    + fmt::Display
    + Serialize
    + DeserializeOwned
    + From<RawId>
    + Into<u64>
    + Send
    + Sync
    + 'static
    + sealed::Sealed
{
    /// The id with the number `raw`.
    fn from_raw(raw: u64) -> Self {
        Self::from(RawId(raw))
    }

    /// The id's number.
    fn get(self) -> u64 {
        self.into()
    }
}

macro_rules! ids {
    ($($(#[$doc:meta])* $name:ident),* $(,)?) => {$(
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub u64);

        impl sealed::Sealed for $name {}
        impl RecordId for $name {}

        impl From<RawId> for $name {
            fn from(raw: RawId) -> Self {
                Self(raw.0)
            }
        }

        impl From<$name> for u64 {
            fn from(id: $name) -> u64 {
                id.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)
            }
        }
    )*};
}

ids! {
    /// A conversation: a line of entries, possibly forked from a parent.
    ConversationId,
    /// An immutable entry of a conversation. Entry ids order entries in creation order.
    EntryId,
    /// A durable task.
    TaskId,
    /// A host submission (input or passive write).
    SubmissionId,
    /// One incarnation of a document; a recreated document gets a new id.
    DocumentId,
}

/// The root conversation's reserved id; storage never mints it.
pub const ROOT_CONVERSATION_ID: ConversationId = ConversationId(1);

/// Commit sequence: strictly increases between commits; gaps are permitted. A separate type
/// from the ids, so an entity id cannot be passed as a document point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Seq(pub u64);

impl fmt::Display for Seq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}
