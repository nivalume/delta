//! Documents inside a transaction: acquisition (get or create, memoized per address),
//! fork copies read through a definition, retirement, and the borrowed [`Doc`] handle over
//! the chord [`Change`].

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use delta_chord::delta::{Change, Draft, Op};
use delta_chord::json::Json;
use delta_chord::state::State;
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::cache::DocHandle;
use super::error::TxError;
use super::tx::Tx;
use crate::documents::{
    CheckpointInfo, DocAddress, DocDefinition, DocumentError, Semantics, check_record, initial_value, semantics,
    upgrade,
};
use crate::records::{
    DocumentAddress, DocumentCopySource, DocumentCreate, DocumentId, DocumentPoint, DocumentRecord, DocumentScope, Seq,
    TaskStatus,
};

/// A definition's checkpoint predicate, kept as a function pointer for settlement.
pub(super) type CheckpointWhen = fn(&Json, &[Op], CheckpointInfo) -> bool;

/// The definition that acquired an entry.
#[derive(Clone, Copy)]
pub(super) struct Acquired {
    pub(super) semantics: Semantics,
    pub(super) checkpoint_when: CheckpointWhen,
}

/// The open change of an acquired entry and the definition that acquired it.
pub(super) struct Open {
    pub(super) change: Change,
    pub(super) acquired: Acquired,
}

/// What a staged entry is in storage and in the cache.
pub(super) enum Target {
    /// A cached incarnation; the change began from its current revision.
    Loaded {
        handle: Arc<DocHandle>,
        stored_version: u32,
        deltas_since_base: u64,
        open: Open,
    },
    /// A new incarnation (also a fork copy read through a definition).
    Created {
        record: DocumentCreate,
        state: State,
        open: Open,
    },
    /// A fork copy no definition read: a `DocumentCopy` write.
    ForkCopy {
        record: DocumentCreate,
        source: DocumentCopySource,
    },
    /// An incarnation retired without acquisition.
    Retiring { record: DocumentRecord },
}

/// One document acquisition, copy or retirement of the transaction.
pub(super) struct DocEntry {
    pub(super) address: DocumentAddress,
    /// `None` for a retirement of an absent address.
    pub(super) target: Option<Target>,
    pub(super) retire: bool,
}

impl DocEntry {
    pub(super) fn fork_copy(record: DocumentCreate, source: DocumentCopySource) -> Self {
        Self {
            address: record.address(),
            target: Some(Target::ForkCopy { record, source }),
            retire: false,
        }
    }

    /// The open change and its definition, once acquired through a definition.
    fn open(&self) -> Option<&Open> {
        match self.target.as_ref()? {
            Target::Loaded { open, .. } | Target::Created { open, .. } => Some(open),
            Target::ForkCopy { .. } | Target::Retiring { .. } => None,
        }
    }

    fn open_mut(&mut self) -> Option<&mut Open> {
        match self.target.as_mut()? {
            Target::Loaded { open, .. } | Target::Created { open, .. } => Some(open),
            Target::ForkCopy { .. } | Target::Retiring { .. } => None,
        }
    }

    /// The incarnation's id, when it has one.
    fn id(&self) -> Option<DocumentId> {
        match self.target.as_ref()? {
            Target::Loaded { handle, .. } => Some(handle.record().id),
            Target::Created { record, .. } | Target::ForkCopy { record, .. } => Some(record.id),
            Target::Retiring { record } => Some(record.id),
        }
    }
}

/// A borrowed handle on the transaction's open change of one incarnation.
///
/// It borrows the [`Tx`] mutably: while held no other `Tx` method runs, and it cannot escape
/// the commit closure. Acquiring the address again returns a handle on the same change, so
/// "edit, table write, edit again" is `doc`, `drop`, `append_entry`, `doc`. A change that
/// ends equal to its base writes and publishes nothing.
pub struct Doc<'tx, D> {
    change: &'tx mut Change,
    _definition: PhantomData<fn() -> D>,
}

impl<D: DocDefinition> Doc<'_, D> {
    /// The working value: the base with the edits so far.
    pub fn value(&self) -> &Json {
        self.change.value()
    }

    /// The revision the change started from.
    pub fn base(&self) -> &Json {
        self.change.base()
    }

    /// A draft of the whole value; edits record ops.
    pub fn draft(&mut self) -> Draft<'_> {
        self.change.draft()
    }

    /// Replaces the whole value; the batch is the diff from the previous working value.
    pub fn set(&mut self, value: Json) {
        self.change.draft().set(value);
    }

    /// Edits the value as a `T` (a serde round trip of the document) and records the diff.
    pub fn update_as<T: Serialize + DeserializeOwned, R>(
        &mut self,
        edit: impl FnOnce(&mut T) -> R,
    ) -> Result<R, serde_json::Error> {
        self.change.draft().update_as(edit)
    }
}

impl<D: DocDefinition> fmt::Debug for Doc<'_, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Doc")
            .field("kind", &D::KIND)
            .field("value", self.change.value())
            .finish()
    }
}

impl Tx<'_> {
    /// The singleton at `at`: the current incarnation, loaded and migrated if needed, or a
    /// new one from `initial`. Memoized per address.
    pub async fn doc<D: DocDefinition<Seed = ()>>(&mut self, at: DocAddress<D>) -> Result<Doc<'_, D>, TxError> {
        self.doc_member(at, ()).await
    }

    /// The document at `at`; a family member is created from `seed` (ignored when it exists).
    ///
    /// A task document requires its task, and refuses once the task has a terminal
    /// candidate. A cached value serves only definitions of its version; another version
    /// reloads from storage.
    pub async fn doc_member<D: DocDefinition>(
        &mut self,
        at: DocAddress<D>,
        seed: D::Seed,
    ) -> Result<Doc<'_, D>, TxError> {
        let address = at.to_address();
        self.check_task_documents_open(address.scope)?;
        let acquired = Acquired {
            semantics: semantics::<D>(),
            checkpoint_when: D::checkpoint_when,
        };
        let index = match self.latest.get(&address).copied() {
            Some(index) if !self.documents[index].retire && self.documents[index].open().is_some() => {
                let entry = &self.documents[index];
                if entry
                    .open()
                    .is_some_and(|open| open.acquired.semantics != acquired.semantics)
                {
                    return Err(TxError::Document(DocumentError::DefinitionMismatch {
                        id: entry.id().unwrap_or(DocumentId(0)),
                        kind: D::KIND.to_owned(),
                    }));
                }
                index
            }
            Some(index) if !self.documents[index].retire => {
                self.acquire_fork_copy::<D>(index, acquired).await?;
                index
            }
            latest => {
                // An address retired in this transaction gets a new incarnation.
                let retired_here = latest.is_some();
                let target = self.acquire::<D>(&address, &seed, retired_here, acquired).await?;
                self.push_document(DocEntry {
                    address,
                    target: Some(target),
                    retire: false,
                })
            }
        };
        let open = self.documents[index]
            .open_mut()
            .expect("an acquired document entry holds its open change");
        Ok(Doc {
            change: &mut open.change,
            _definition: PhantomData,
        })
    }

    /// The current incarnation at `at` as this transaction sees it, with its id, without
    /// acquiring or creating it: an acquired document's working value, the committed value
    /// (loaded and migrated in memory, as [`Session::snapshot`](super::Session::snapshot)
    /// does), or `None` when absent or retired here. Reads nothing from the tables.
    pub async fn peek_doc<D: DocDefinition>(
        &mut self,
        at: DocAddress<D>,
    ) -> Result<Option<(DocumentId, Json)>, TxError> {
        let address = at.to_address();
        if let Some(index) = self.latest.get(&address).copied() {
            if self.documents[index].retire {
                return Ok(None);
            }
            if matches!(self.documents[index].target, Some(Target::ForkCopy { .. })) {
                let acquired = Acquired {
                    semantics: semantics::<D>(),
                    checkpoint_when: D::checkpoint_when,
                };
                self.acquire_fork_copy::<D>(index, acquired).await?;
            }
            let entry = &self.documents[index];
            return Ok(entry.id().zip(entry.open().map(|open| open.change.value().clone())));
        }
        let loaded = self.line.cache.load::<D>(&self.line.guard, &address, &self.ctx).await?;
        Ok(loaded.map(|loaded| (loaded.handle.record().id, loaded.handle.value())))
    }

    /// Retires the current incarnation at `at` in this commit; an acquired change's final
    /// content is written first. Absent: nothing. A later `doc` in the same transaction
    /// creates a new incarnation.
    pub async fn retire_doc<D: DocDefinition>(&mut self, at: DocAddress<D>) -> Result<(), TxError> {
        let address = at.to_address();
        if let Some(index) = self.latest.get(&address).copied() {
            let entry = &mut self.documents[index];
            if entry.retire {
                return Ok(());
            }
            if let Some(Target::ForkCopy { record, .. }) = &entry.target {
                check_record::<D>(&DocumentRecord::created(record.clone(), Seq(0)))?;
            }
            entry.retire = true;
            return Ok(());
        }
        let cached = self
            .line
            .shared
            .documents
            .get(&address)
            .filter(|handle| !handle.is_ended())
            .map(|handle| handle.record().clone());
        let record = match cached {
            Some(record) => Some(record),
            None => {
                let ctx = &self.ctx;
                self.line
                    .guard
                    .read(ctx, |storage| {
                        storage.find_document(&address, DocumentPoint::Current, ctx)
                    })
                    .await?
            }
        };
        if let Some(record) = &record {
            check_record::<D>(record)?;
        }
        self.push_document(DocEntry {
            address,
            target: record.map(|record| Target::Retiring { record }),
            retire: true,
        });
        Ok(())
    }

    fn check_task_documents_open(&self, scope: DocumentScope) -> Result<(), TxError> {
        if let DocumentScope::Task { task_id } = scope
            && self
                .tasks
                .get(&task_id)
                .and_then(|task| task.write.as_ref())
                .is_some_and(|write| write.record.state.status() == TaskStatus::Terminal)
        {
            return Err(TxError::TerminalTask(task_id));
        }
        Ok(())
    }

    async fn acquire<D: DocDefinition>(
        &mut self,
        address: &DocumentAddress,
        seed: &D::Seed,
        skip_load: bool,
        acquired: Acquired,
    ) -> Result<Target, TxError> {
        if !skip_load {
            for _ in 0..2 {
                let Some(loaded) = self.line.cache.load::<D>(&self.line.guard, address, &self.ctx).await? else {
                    break;
                };
                // A handle ended by a close that sealed meanwhile is reloaded fresh.
                let Some(change) = loaded.handle.begin() else {
                    self.line.cache.forget(address, &loaded.handle);
                    continue;
                };
                return Ok(Target::Loaded {
                    handle: loaded.handle,
                    stored_version: loaded.stored_version,
                    deltas_since_base: loaded.deltas_since_base,
                    open: Open { change, acquired },
                });
            }
        }
        self.require_owner(address.scope).await?;
        let value = initial_value::<D>(seed)?;
        let id: DocumentId = self.line.guard.mint().await?;
        let record = DocumentCreate {
            id,
            kind: D::KIND.to_owned(),
            key: address.key.clone(),
            scope: address.scope,
            policy: semantics::<D>().policy,
        };
        record.check()?;
        let state = State::new(value);
        let change = state.begin();
        Ok(Target::Created {
            record,
            state,
            open: Open { change, acquired },
        })
    }

    /// A conversation document needs its conversation; a task document its task, not
    /// terminal.
    async fn require_owner(&mut self, scope: DocumentScope) -> Result<(), TxError> {
        match scope {
            DocumentScope::Session => Ok(()),
            DocumentScope::Conversation { conversation_id } => self.require_conversation(conversation_id).await,
            DocumentScope::Task { task_id } => {
                let task = self.current_task(task_id).await?.ok_or(TxError::UnknownTask(task_id))?;
                if task.state.status() == TaskStatus::Terminal {
                    return Err(TxError::TerminalTask(task_id));
                }
                Ok(())
            }
        }
    }

    /// Reads an unaccessed fork copy's source through `D`: checked against the copy's
    /// record, migrated, and turned into a creation with the final value.
    async fn acquire_fork_copy<D: DocDefinition>(&mut self, index: usize, acquired: Acquired) -> Result<(), TxError> {
        let Some(Target::ForkCopy { record, source }) = &self.documents[index].target else {
            return Err(TxError::Fork("Fork copy entry has no source".to_owned()));
        };
        let (record, source) = (record.clone(), *source);
        let ctx = &self.ctx;
        let stored = self
            .line
            .guard
            .read(ctx, |storage| storage.document(source.id, source.at, ctx))
            .await?
            .ok_or_else(|| TxError::Fork(format!("Fork source document {} cannot be read", source.id)))?;
        let matches = matches!(stored.record.scope, DocumentScope::Conversation { .. })
            && stored.record.kind == record.kind
            && stored.record.key == record.key
            && stored.record.policy == record.policy;
        if !matches {
            return Err(TxError::Fork(format!(
                "Fork source document {} does not match the copied record",
                source.id
            )));
        }
        check_record::<D>(&DocumentRecord::created(record.clone(), Seq(0)))?;
        let upgraded = upgrade::<D>(record.id, stored.version, stored.value)?;
        let state = State::new(upgraded.value);
        let change = state.begin();
        self.documents[index].target = Some(Target::Created {
            record,
            state,
            open: Open { change, acquired },
        });
        Ok(())
    }
}
