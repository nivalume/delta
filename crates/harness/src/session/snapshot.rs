//! Document reads from outside a commit: the current value (`snapshot`), the value at an
//! entry's commit (`snapshot_as_of`) and the observable state (`document_state`). Each
//! migrates in memory, writes nothing and never creates.

use delta_chord::context::Context;
use delta_chord::json::Json;
use futures::FutureExt;

use super::Session;
use super::error::{SessionError, TxError};
use super::line::{Line, contained};
use super::observation::DocumentState;
use crate::documents::{ConversationDoc, DocAddress, DocDefinition, Rewindable, check_record, upgrade};
use crate::records::{DocumentAddress, DocumentPoint, DocumentScope, EntryId};

impl Session {
    /// The current value of the document at `at`, `None` when absent.
    ///
    /// A cache hit of the definition's version is answered without queuing on the line;
    /// otherwise the line loads it (and keeps it cached). An older stored version is
    /// migrated in memory only.
    pub async fn snapshot<D: DocDefinition>(
        &self,
        at: DocAddress<D>,
        ctx: &Context,
    ) -> Result<Option<Json>, SessionError> {
        let shared = self.shared();
        shared.check_usable()?;
        let address = at.to_address();
        if let Some(handle) = shared.documents.get(&address)
            && handle.value_version() == D::VERSION
            && let Some(value) = handle.live_value()
        {
            check_record::<D>(handle.record()).map_err(TxError::Document)?;
            handle.touch();
            return Ok(Some(value));
        }
        let ctx = ctx.clone();
        self.run(move |line| contained(line.snapshot::<D>(address, ctx)).boxed())
            .await
    }

    /// The value of a rewindable conversation document as of the commit that wrote `entry`
    /// (visible from the address's conversation, possibly in an ancestor), `None` when no
    /// instance was alive then. Never starts from today's incarnation; refused with
    /// [`TxError::Fork`] when the entry is not visible.
    pub async fn snapshot_as_of<D: ConversationDoc<History = Rewindable>>(
        &self,
        at: DocAddress<D>,
        entry: EntryId,
        ctx: &Context,
    ) -> Result<Option<Json>, SessionError> {
        self.shared().check_usable()?;
        let address = at.to_address();
        let ctx = ctx.clone();
        self.run(move |line| contained(line.snapshot_as_of::<D>(address, entry, ctx)).boxed())
            .await
    }

    /// A read-only state of the current incarnation at `at`, `None` when absent (never
    /// creates). Acquisition runs on the line, so the state starts at a committed revision
    /// and follows every later one.
    pub async fn document_state<D: DocDefinition>(
        &self,
        at: DocAddress<D>,
        ctx: &Context,
    ) -> Result<Option<DocumentState>, SessionError> {
        self.shared().check_usable()?;
        let address = at.to_address();
        let ctx = ctx.clone();
        self.run(move |line| {
            async move {
                // Admitted before a seal that happened meanwhile: the seal ended every state.
                if line.shared.is_sealed() {
                    return Err(line.shared.terminal_error());
                }
                let loaded = contained(line.load::<D>(address, ctx)).await?;
                Ok(loaded.map(|handle| DocumentState::attach(handle, &line.shared)))
            }
            .boxed()
        })
        .await
    }
}

impl Line {
    async fn load<D: DocDefinition>(
        &mut self,
        address: DocumentAddress,
        ctx: Context,
    ) -> Result<Option<std::sync::Arc<super::cache::DocHandle>>, TxError> {
        self.shared.check_healthy()?;
        let loaded = self.cache.load::<D>(&self.guard, &address, &ctx).await?;
        Ok(loaded.map(|loaded| loaded.handle))
    }

    async fn snapshot<D: DocDefinition>(
        &mut self,
        address: DocumentAddress,
        ctx: Context,
    ) -> Result<Option<Json>, TxError> {
        let handle = self.load::<D>(address, ctx).await?;
        Ok(handle.map(|handle| handle.value()))
    }

    async fn snapshot_as_of<D: DocDefinition>(
        &mut self,
        mut address: DocumentAddress,
        entry: EntryId,
        ctx: Context,
    ) -> Result<Option<Json>, TxError> {
        self.shared.check_healthy()?;
        let DocumentScope::Conversation { conversation_id } = address.scope else {
            return Ok(None);
        };
        let ctx = &ctx;
        let stored = self
            .guard
            .read(ctx, |storage| storage.visible_entry(conversation_id, entry, ctx))
            .await?
            .ok_or_else(|| {
                TxError::Fork(format!(
                    "Entry {entry} is not visible from conversation {conversation_id}"
                ))
            })?;
        let at = DocumentPoint::At(stored.commit_seq);
        address.scope = DocumentScope::Conversation {
            conversation_id: stored.entry.conversation_id,
        };
        let found = self
            .guard
            .read(ctx, |storage| storage.find_document(&address, at, ctx))
            .await?;
        let Some(record) = found else {
            return Ok(None);
        };
        let document = self
            .guard
            .read(ctx, |storage| storage.document(record.id, at, ctx))
            .await?;
        let Some(document) = document else {
            return Err(self.guard.corrupt(format!(
                "Historical document {} ({}) cannot be read",
                record.id, record.kind
            )));
        };
        check_record::<D>(&document.record)?;
        let upgraded = upgrade::<D>(document.record.id, document.version, document.value)?;
        Ok(Some(upgraded.value))
    }
}
