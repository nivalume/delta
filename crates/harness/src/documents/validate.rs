//! Checks of definitions against themselves, against persisted records and against stored
//! versions; the values `initial` and `migrate` produce.

use delta_chord::json::Json;

use super::definition::{DocDefinition, DocScope, MigrateError};
use crate::records::{ConversationPolicy, DocumentId, DocumentRecord, DocumentScopeKind};

/// What a definition claims about every incarnation it touches; compared with the persisted
/// record on typed access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Semantics {
    pub kind: &'static str,
    pub version: u32,
    pub scope: DocumentScopeKind,
    /// `Some` exactly for conversation documents.
    pub policy: Option<ConversationPolicy>,
    pub family: bool,
}

impl Semantics {
    /// Whether `record` is an incarnation this definition may read: same kind, scope kind,
    /// history, fork, and a key exactly when the definition is a family.
    pub fn matches(&self, record: &DocumentRecord) -> bool {
        record.kind == self.kind
            && record.scope.kind() == self.scope
            && record.policy == self.policy
            && record.key.is_some() == self.family
    }
}

/// The semantics of `D`. A non-positive `VERSION` fails compilation where this is
/// instantiated; scope, history and fork are already checked by the types.
pub fn semantics<D: DocDefinition>() -> Semantics {
    const { assert!(D::VERSION >= 1, "document definition versions start at 1") };
    Semantics {
        kind: D::KIND,
        version: D::VERSION,
        scope: <D::Scope as DocScope>::KIND,
        policy: <D::Scope as DocScope>::POLICY,
        family: D::FAMILY,
    }
}

/// Refuses an incarnation whose persisted semantics differ from `D`'s; a migration cannot
/// reinterpret them.
pub fn check_record<D: DocDefinition>(record: &DocumentRecord) -> Result<(), DocumentError> {
    if semantics::<D>().matches(record) {
        Ok(())
    } else {
        Err(DocumentError::DefinitionMismatch {
            id: record.id,
            kind: record.kind.clone(),
        })
    }
}

/// `D::initial(seed)`, refused unless it is an object.
pub fn initial_value<D: DocDefinition>(seed: &D::Seed) -> Result<Json, DocumentError> {
    let value = D::initial(seed);
    if value.as_object().is_some() {
        Ok(value)
    } else {
        Err(DocumentError::NotAnObject {
            kind: D::KIND.to_owned(),
            origin: ValueSource::Initial,
        })
    }
}

/// A stored value brought to `D::VERSION`.
#[derive(Clone, Debug, PartialEq)]
pub struct Upgraded {
    pub value: Json,
    /// Whether `migrate` ran; the first writable access then stores a required base.
    pub migrated: bool,
}

/// The value of incarnation `id`, stored at `stored_version`, as `D::VERSION` expects it:
/// unchanged when the versions are equal, migrated when older, refused when newer or when
/// the migration is unsupported, fails or yields a non-object.
pub fn upgrade<D: DocDefinition>(id: DocumentId, stored_version: u32, value: Json) -> Result<Upgraded, DocumentError> {
    let definition = semantics::<D>().version;
    if stored_version == definition {
        return Ok(Upgraded { value, migrated: false });
    }
    if stored_version > definition {
        return Err(DocumentError::VersionTooNew {
            id,
            kind: D::KIND.to_owned(),
            stored: stored_version,
            definition,
        });
    }
    match D::migrate(value, stored_version) {
        Ok(value) if value.as_object().is_some() => Ok(Upgraded { value, migrated: true }),
        Ok(_) => Err(DocumentError::NotAnObject {
            kind: D::KIND.to_owned(),
            origin: ValueSource::Migrated,
        }),
        Err(MigrateError::Unsupported { .. }) => Err(DocumentError::MigrationRequired {
            id,
            kind: D::KIND.to_owned(),
            stored: stored_version,
        }),
        Err(source) => Err(DocumentError::Migrate {
            kind: D::KIND.to_owned(),
            from: stored_version,
            source,
        }),
    }
}

/// Which definition function produced a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueSource {
    Initial,
    Migrated,
}

impl ValueSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Initial => "initial",
            Self::Migrated => "migrated",
        }
    }
}

/// A document access the definition layer refuses. The Session reports these as
/// rejected commits or failed reads; nothing is written.
#[derive(Debug, thiserror::Error)]
pub enum DocumentError {
    #[error("Document {id} ({kind}) does not match the supplied definition semantics")]
    DefinitionMismatch { id: DocumentId, kind: String },
    #[error("Document {id} ({kind}) has newer version {stored} than {definition}")]
    VersionTooNew {
        id: DocumentId,
        kind: String,
        stored: u32,
        definition: u32,
    },
    #[error("Document {id} ({kind}) requires migration from version {stored}")]
    MigrationRequired { id: DocumentId, kind: String, stored: u32 },
    #[error("Document {kind} migration from version {from} failed")]
    Migrate {
        kind: String,
        from: u32,
        #[source]
        source: MigrateError,
    },
    #[error("Document {kind} {} value is not an object", .origin.as_str())]
    NotAnObject { kind: String, origin: ValueSource },
}
