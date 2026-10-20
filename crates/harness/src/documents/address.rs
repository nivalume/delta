//! Typed logical addresses of documents.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use super::definition::{DocDefinition, DocScope};
use crate::records::{DocumentAddress, DocumentScope};

/// The owner type of definition `D`'s scope.
pub type OwnerOf<D> = <<D as DocDefinition>::Scope as DocScope>::Owner;

/// The logical address of one instance of `D`: its scope owner and, for a family member,
/// its key. The argument shape of `tx.doc`, `snapshot`, `document_state` and `retire_doc`.
pub struct DocAddress<D: DocDefinition> {
    owner: OwnerOf<D>,
    key: Option<String>,
    _definition: PhantomData<fn() -> D>,
}

impl<D: DocDefinition<Seed = ()>> DocAddress<D> {
    /// The singleton of `owner`. Not available for a family (compile-time check).
    pub fn new(owner: OwnerOf<D>) -> Self {
        const { assert!(!D::FAMILY, "a family document is addressed with DocAddress::member") };
        Self {
            owner,
            key: None,
            _definition: PhantomData,
        }
    }
}

impl<D: DocDefinition> DocAddress<D> {
    /// The family member `key` of `owner`. Only for a family (compile-time check).
    pub fn member(owner: OwnerOf<D>, key: impl Into<String>) -> Self {
        const { assert!(D::FAMILY, "a singleton document is addressed with DocAddress::new") };
        Self {
            owner,
            key: Some(key.into()),
            _definition: PhantomData,
        }
    }

    pub fn owner(&self) -> OwnerOf<D> {
        self.owner
    }

    /// The member key; `None` for a singleton.
    pub fn key(&self) -> Option<&str> {
        self.key.as_deref()
    }

    /// The persisted scope.
    pub fn scope(&self) -> DocumentScope {
        D::Scope::scope(self.owner)
    }

    /// The untyped address storage looks up. Also the Session's cache key.
    pub fn to_address(&self) -> DocumentAddress {
        DocumentAddress {
            kind: D::KIND.to_owned(),
            scope: self.scope(),
            key: self.key.clone(),
        }
    }
}

impl<D: DocDefinition> Clone for DocAddress<D> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner,
            key: self.key.clone(),
            _definition: PhantomData,
        }
    }
}

impl<D: DocDefinition> PartialEq for DocAddress<D> {
    fn eq(&self, other: &Self) -> bool {
        self.owner == other.owner && self.key == other.key
    }
}

impl<D: DocDefinition> Eq for DocAddress<D> {}

impl<D: DocDefinition> Hash for DocAddress<D> {
    fn hash<S: Hasher>(&self, state: &mut S) {
        self.owner.hash(state);
        self.key.hash(state);
    }
}

impl<D: DocDefinition> fmt::Debug for DocAddress<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DocAddress")
            .field("kind", &D::KIND)
            .field("owner", &self.owner)
            .field("key", &self.key)
            .finish()
    }
}
