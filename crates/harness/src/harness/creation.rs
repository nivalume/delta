//! Creating conversations: the built-in creation hook every creating commit runs, and the
//! conveniences that apply an agent change and an `init` in the same commit.

use std::fmt;
use std::sync::Arc;

use futures::future::BoxFuture;

use crate::agent::{AgentChange, configure, create_agent};
use crate::approval::create_permissions;
use crate::documents::DocAddress;
use crate::generation::{LiveDoc, ProviderDoc};
use crate::inbox::InboxDoc;
use crate::records::{ConversationId, ConversationOwnership, ConversationRecord, EntryId, ROOT_CONVERSATION_ID};
use crate::session::{ConversationHook, Tx, TxError};
use crate::usage::UsageDoc;

/// Writes in the creating commit, after the creation hook and the agent change.
pub type ConversationInit =
    Box<dyn for<'a, 'l> FnOnce(&'a mut Tx<'l>, ConversationId) -> BoxFuture<'a, Result<(), TxError>> + Send>;

/// How a new conversation starts.
pub struct ConversationCreateOptions {
    /// Hosts choose ownership explicitly; nothing infers it.
    pub ownership: ConversationOwnership,
    pub agent: Option<AgentChange>,
    pub init: Option<ConversationInit>,
}

impl ConversationCreateOptions {
    /// An ownerless conversation with nothing else.
    pub fn ownerless() -> Self {
        Self {
            ownership: ConversationOwnership::Ownerless,
            agent: None,
            init: None,
        }
    }

    #[must_use]
    pub fn with_agent(mut self, agent: AgentChange) -> Self {
        self.agent = Some(agent);
        self
    }
}

impl fmt::Debug for ConversationCreateOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConversationCreateOptions")
            .field("ownership", &self.ownership)
            .field("agent", &self.agent)
            .field("init", &self.init.is_some())
            .finish()
    }
}

/// How the root conversation starts; ignored when it exists.
#[derive(Default)]
pub struct RootOptions {
    pub agent: Option<AgentChange>,
    pub init: Option<ConversationInit>,
}

impl fmt::Debug for RootOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RootOptions")
            .field("agent", &self.agent)
            .field("init", &self.init.is_some())
            .finish()
    }
}

/// What a creating commit creates.
pub(super) enum Target {
    Root,
    Independent(ConversationOwnership),
    Fork {
        parent: ConversationId,
        at: EntryId,
        ownership: ConversationOwnership,
    },
}

/// Creates the conversation `target` names with `agent` and `init` in one commit; the root
/// is created once, and `agent` and `init` are ignored when it exists.
pub(super) async fn create(
    tx: &mut Tx<'_>,
    target: Target,
    agent: Option<AgentChange>,
    init: Option<ConversationInit>,
) -> Result<ConversationId, TxError> {
    let record = match target {
        Target::Root => {
            if tx.conversation(ROOT_CONVERSATION_ID).await?.is_some() {
                return Ok(ROOT_CONVERSATION_ID);
            }
            tx.create_root_conversation().await?
        }
        Target::Independent(ownership) => tx.create_conversation(ownership).await?,
        Target::Fork { parent, at, ownership } => tx.fork_conversation(parent, at, ownership).await?,
    };
    if let Some(change) = agent {
        configure(tx, record.id, change).await?;
    }
    if let Some(init) = init {
        init(tx, record.id).await?;
    }
    Ok(record.id)
}

/// The built-in creation hook: empty `delta.live`, `delta.inbox` and `delta.usage`, a fresh
/// `delta.provider`, the conversation's `delta.agent` (and its owner's `delta.permissions`),
/// then the host's hook.
pub(super) struct CreationHook {
    pub(super) host: Option<Arc<dyn ConversationHook>>,
}

impl ConversationHook for CreationHook {
    fn conversation_created<'a>(
        &'a self,
        tx: &'a mut Tx<'_>,
        record: &'a ConversationRecord,
    ) -> BoxFuture<'a, Result<(), TxError>> {
        Box::pin(async move {
            let id = record.id;
            tx.doc(DocAddress::<LiveDoc>::new(id)).await?;
            tx.doc(DocAddress::<InboxDoc>::new(id)).await?;
            tx.doc(DocAddress::<UsageDoc>::new(id)).await?;
            tx.doc(DocAddress::<ProviderDoc>::new(id)).await?;
            create_agent(tx, record).await?;
            create_permissions(tx, record).await?;
            if let Some(host) = &self.host {
                host.conversation_created(tx, record).await?;
            }
            Ok(())
        })
    }
}
