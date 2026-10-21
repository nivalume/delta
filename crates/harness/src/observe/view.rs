//! The conversation view (spec §9.3): one `delta_chord::State` per observed conversation,
//! holding its record, its active transcript and its mounted documents, built on the line
//! and advanced from every commit publication with the ops the commit wrote.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError, Weak};

use delta_chord::context::Context;
use delta_chord::delta::Op;
use delta_chord::json::{self, Json, Seg};
use delta_chord::state::{Delivery, State, Subscription};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::agent::AgentDoc;
use crate::approval::ApprovalsDoc;
use crate::documents::{DocAddress, DocDefinition};
use crate::generation::{LiveDoc, ProviderDoc};
use crate::inbox::InboxDoc;
use crate::records::{
    ConversationId, ConversationRecord, DocumentId, DocumentScope, EntryId, EntryQuery, EntryRecord, ScanOrder,
};
use crate::session::{CommitChange, CommitPublication, ObserverGuard, Session, SessionError, Tx, TxError};
use crate::usage::UsageDoc;

/// Scans read pages of this many entries.
const PAGE: usize = 256;

/// The documents a view mounts, by kind.
pub const MOUNTED_DOCUMENTS: [&str; 6] = [
    AgentDoc::KIND,
    LiveDoc::KIND,
    InboxDoc::KIND,
    ProviderDoc::KIND,
    UsageDoc::KIND,
    ApprovalsDoc::KIND,
];

/// The value of a conversation view.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConversationView {
    pub conversation: ConversationRecord,
    /// The head marker, then the non-head entries from its head (the active transcript).
    pub entries: Vec<EntryRecord>,
    /// The mounted documents by kind ([`MOUNTED_DOCUMENTS`]); absent documents absent.
    pub docs: IndexMap<String, Json>,
}

/// What a mount follows and how it advances.
struct Core {
    conversation: ConversationId,
    state: State,
    /// The mounted incarnation of each present document, with its version.
    docs: HashMap<String, (DocumentId, u32)>,
    /// The mounted entries: id and whether it is a head marker.
    entries: Vec<(EntryId, bool)>,
    /// Who else follows the mount's revisions (the event streams).
    listeners: Vec<Weak<dyn MountListener>>,
}

/// One revision of a mount, as a listener sees it.
pub(crate) struct Revision<'a> {
    pub(crate) before: &'a Json,
    pub(crate) after: &'a Json,
    pub(crate) publication: &'a CommitPublication,
}

/// Follows a mount's revisions on the commit line; it must not block.
pub(crate) trait MountListener: Send + Sync {
    fn advanced(&self, revision: &Revision<'_>);
}

impl Core {
    /// The view ops of `publication`: document ops, then entry ops in id order.
    fn ops(&mut self, publication: &CommitPublication) -> Vec<Op> {
        let value = self.state.value();
        let mut ops = Vec::new();
        let mut entries: Vec<&EntryRecord> = Vec::new();
        for change in &publication.changes {
            match change {
                CommitChange::Conversation(record) if record.id == self.conversation => {
                    let json = json::to_json(record).unwrap_or(Json::Null);
                    if value.get("conversation") != Some(&json) {
                        ops.push(Op::Set(path(&["conversation"]), json));
                    }
                }
                CommitChange::Document {
                    record,
                    version,
                    value: document,
                    ops: changed,
                    ..
                } if record.scope
                    == (DocumentScope::Conversation {
                        conversation_id: self.conversation,
                    })
                    && record.key.is_none()
                    && MOUNTED_DOCUMENTS.contains(&record.kind.as_str()) =>
                {
                    let kind = record.kind.as_str();
                    let mounted = self.docs.get(kind).copied();
                    let at = path(&["docs", kind]);
                    match (document, version) {
                        (None, _) => {
                            if mounted.is_some_and(|(id, _)| id == record.id) {
                                self.docs.remove(kind);
                                ops.push(Op::Delete(at));
                            }
                        }
                        (Some(document), Some(version)) => {
                            let same = mounted == Some((record.id, *version));
                            if same && !changed.is_empty() {
                                let prefix: Vec<Seg> = vec!["docs".into(), kind.into()];
                                ops.extend(changed.iter().cloned().map(|op| op.prefixed(&prefix)));
                            } else if !same || value.get("docs").and_then(|docs| docs.get(kind)) != Some(document) {
                                ops.push(Op::Set(at, document.clone()));
                            }
                            self.docs.insert(kind.to_owned(), (record.id, *version));
                        }
                        (Some(_), None) => {}
                    }
                }
                CommitChange::Entry(entry) if entry.conversation_id == self.conversation => entries.push(entry),
                _ => {}
            }
        }
        entries.sort_by_key(|entry| entry.id);
        for entry in entries {
            let json = json::to_json(entry).unwrap_or(Json::Null);
            match entry.head {
                None => {
                    ops.push(Op::Splice {
                        path: path(&["entries"]),
                        index: self.entries.len(),
                        remove: 0,
                        items: vec![json],
                    });
                    self.entries.push((entry.id, false));
                }
                Some(head) => {
                    // Every kept entry is a suffix: the non-head entries from the new head.
                    let kept = self
                        .entries
                        .iter()
                        .position(|(id, marker)| !marker && *id >= head)
                        .unwrap_or(self.entries.len());
                    ops.push(Op::Splice {
                        path: path(&["entries"]),
                        index: 0,
                        remove: kept,
                        items: vec![json],
                    });
                    self.entries.splice(..kept, [(entry.id, true)]);
                }
            }
        }
        ops
    }

    /// Advances by one publication. Never fails the Session: an op that does not apply (a
    /// bug, not a state the commits can reach) is skipped.
    fn advance(&mut self, publication: &CommitPublication) {
        let mut ops = self.ops(publication);
        if ops.is_empty() {
            return;
        }
        let before_value = self.state.value();
        if delta_chord::delta::applied(&before_value, &ops).is_err() {
            let mut value = before_value.clone();
            ops.retain(|op| delta_chord::delta::apply(&mut value, std::slice::from_ref(op)).is_ok());
        }
        // Applies: every op that did not was dropped above.
        let _ = self.state.apply(&publication.context, ops);
        if self.listeners.is_empty() {
            return;
        }
        let after = self.state.value();
        let revision = Revision {
            before: &before_value,
            after: &after,
            publication,
        };
        self.listeners.retain(|listener| match listener.upgrade() {
            Some(listener) => {
                listener.advanced(&revision);
                true
            }
            None => false,
        });
    }
}

fn path(segments: &[&str]) -> delta_chord::json::Path {
    segments
        .iter()
        .map(|segment| Seg::from(*segment))
        .collect::<Vec<_>>()
        .into()
}

/// One live mount; dropping the last handle stops it following the commits.
pub(crate) struct Mount {
    core: Arc<Mutex<Core>>,
    state: State,
    _observer: ObserverGuard,
}

impl Mount {
    pub(crate) fn state(&self) -> &State {
        &self.state
    }

    /// Adds a listener; call it on the line (inside a commit) so it misses no revision.
    pub(crate) fn listen(&self, listener: Weak<dyn MountListener>) {
        self.core
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .listeners
            .push(listener);
    }
}

/// A view as read in a transaction, with what its mount follows.
pub(crate) struct ReadView {
    pub(crate) value: Json,
    docs: HashMap<String, (DocumentId, u32)>,
    entries: Vec<(EntryId, bool)>,
}

/// Reads the active transcript and the mounted documents of `conversation` in `tx`.
pub(crate) async fn read_view(tx: &mut Tx<'_>, conversation: ConversationId) -> Result<ReadView, TxError> {
    let record = tx
        .conversation(conversation)
        .await?
        .ok_or(TxError::UnknownConversation(conversation))?;
    let head = tx.latest_head_marker(conversation).await?;
    let mut visible = Vec::new();
    let mut cursor = None;
    loop {
        let query = EntryQuery {
            min_entry_id: head.as_ref().and_then(|head| head.head),
            order: Some(ScanOrder::Ascending),
            ..EntryQuery::new(conversation)
        };
        let page = tx.scan_entries(query, PAGE, cursor).await?;
        visible.extend(page.items);
        match page.next {
            Some(next) => cursor = Some(next),
            None => break,
        }
    }
    let active: Vec<EntryRecord> = match &head {
        None => visible,
        Some(head) => std::iter::once(head.clone())
            .chain(visible.into_iter().filter(|entry| entry.head.is_none()))
            .collect(),
    };
    let ids = active.iter().map(|entry| (entry.id, entry.head.is_some())).collect();
    let mut docs = IndexMap::new();
    let mut incarnations = HashMap::new();
    macro_rules! peek {
        ($definition:ty) => {
            if let Some((id, value)) = tx.peek_doc(DocAddress::<$definition>::new(conversation)).await? {
                docs.insert(<$definition>::KIND.to_owned(), value);
                incarnations.insert(<$definition>::KIND.to_owned(), (id, <$definition>::VERSION));
            }
        };
    }
    peek!(AgentDoc);
    peek!(LiveDoc);
    peek!(InboxDoc);
    peek!(ProviderDoc);
    peek!(UsageDoc);
    peek!(ApprovalsDoc);
    let view = ConversationView {
        conversation: record,
        entries: active,
        docs,
    };
    let value = json::to_json(&view).map_err(TxError::callback)?;
    Ok(ReadView {
        value,
        docs: incarnations,
        entries: ids,
    })
}

/// The live mounts of one Session, at most one per conversation.
#[derive(Default)]
pub(crate) struct ViewMounts {
    mounts: Mutex<HashMap<ConversationId, Weak<Mount>>>,
}

impl ViewMounts {
    /// The mount of `conversation`, built on the line when none lives. `then` runs on the
    /// line right after (inside the same commit), with the mount, so a listener it adds
    /// misses no revision.
    pub(crate) async fn mount<R, F>(
        &self,
        session: &Session,
        conversation: ConversationId,
        ctx: &Context,
        then: F,
    ) -> Result<(Arc<Mount>, R), SessionError>
    where
        F: for<'a, 'l> FnOnce(&'a mut Tx<'l>, &'a Arc<Mount>) -> futures::future::BoxFuture<'a, Result<R, TxError>>
            + Send
            + 'static,
        R: Send + 'static,
    {
        let existing = self
            .mounts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&conversation)
            .and_then(Weak::upgrade);
        let registrar = session.clone();
        let (mount, result) = session
            .commit(ctx, move |tx| {
                Box::pin(async move {
                    let mount = match existing {
                        Some(mount) => mount,
                        None => {
                            let read = read_view(tx, conversation).await?;
                            let state = State::new(read.value);
                            let core = Arc::new(Mutex::new(Core {
                                conversation,
                                state: state.clone(),
                                docs: read.docs,
                                entries: read.entries,
                                listeners: Vec::new(),
                            }));
                            let following = core.clone();
                            let observer =
                                registrar.observe_commits(Box::new(move |publication: &CommitPublication| {
                                    following
                                        .lock()
                                        .unwrap_or_else(PoisonError::into_inner)
                                        .advance(publication);
                                    Ok(())
                                }));
                            Arc::new(Mount {
                                core,
                                state,
                                _observer: observer,
                            })
                        }
                    };
                    let result = then(tx, &mount).await?;
                    Ok((mount, result))
                })
            })
            .await?;
        self.mounts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(conversation, Arc::downgrade(&mount));
        Ok((mount, result))
    }
}

impl fmt::Debug for ViewMounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = self.mounts.lock().unwrap_or_else(PoisonError::into_inner).len();
        f.debug_struct("ViewMounts").field("mounts", &count).finish()
    }
}

/// A read-only conversation view that follows every commit while it, or one of its
/// subscriptions, lives.
#[derive(Clone)]
pub struct ViewState {
    mount: Arc<Mount>,
}

impl ViewState {
    pub(crate) fn new(mount: Arc<Mount>) -> Self {
        Self { mount }
    }

    /// The current value. O(1).
    pub fn value(&self) -> Json {
        self.mount.state().value()
    }

    /// The current value, typed.
    pub fn view(&self) -> Result<ConversationView, serde_json::Error> {
        json::from_json(&self.value())
    }

    /// The number of revisions the mount published.
    pub fn sequence(&self) -> u64 {
        self.mount.state().sequence()
    }

    /// Every revision from now on: a `Hydrate`, then one `Update` per commit that changed
    /// the view, with its ops.
    pub fn subscribe(&self) -> ViewSubscription {
        ViewSubscription {
            inner: self.mount.state().subscribe(),
            _mount: self.mount.clone(),
        }
    }
}

impl fmt::Debug for ViewState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ViewState")
            .field("sequence", &self.sequence())
            .finish_non_exhaustive()
    }
}

/// The deliveries of a [`ViewState`]; it keeps the mount following like the state.
pub struct ViewSubscription {
    inner: Subscription,
    _mount: Arc<Mount>,
}

impl ViewSubscription {
    /// The next delivery; `None` once the mount ended.
    pub async fn next(&mut self) -> Option<Delivery> {
        self.inner.next().await
    }

    /// The next delivery if one is queued.
    pub fn try_next(&mut self) -> Option<Delivery> {
        self.inner.try_next()
    }
}

impl fmt::Debug for ViewSubscription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ViewSubscription").finish_non_exhaustive()
    }
}
