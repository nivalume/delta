//! The definition trait and the type-level scope, history and fork markers.

use std::convert::Infallible;
use std::error::Error as StdError;
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

use delta_chord::delta::Op;
use delta_chord::json::Json;
use serde::Serialize;

use crate::records::{ConversationId, ConversationPolicy, DocumentScope, DocumentScopeKind, Fork, History, TaskId};

mod sealed {
    pub trait Sealed {}
}

/// What a document is: kind, version and migrations, initial value, owning scope, and when a
/// change is stored as a complete base.
///
/// A definition is a stateless unit type; its items are read without an instance. Scope,
/// history and fork are types ([`DocScope`]), so a session document cannot be addressed with
/// a conversation id and `as_of` without `rewindable` does not compile.
///
/// ```
/// use delta_chord::delta::Op;
/// use delta_chord::json::Json;
/// use delta_harness::documents::{CheckpointInfo, ConversationScope, DocDefinition, ForkInitial, Latest};
///
/// pub struct Inbox;
///
/// impl DocDefinition for Inbox {
///     const KIND: &'static str = "delta.inbox";
///     const VERSION: u32 = 1;
///     type Scope = ConversationScope<Latest, ForkInitial>;
///     type Seed = ();
///
///     fn initial(_: &()) -> Json {
///         delta_chord::json!({ "items": [] })
///     }
///
///     fn checkpoint_when(value: &Json, _: &[Op], _: CheckpointInfo) -> bool {
///         value.get("items").and_then(Json::as_array).is_some_and(|items| items.is_empty())
///     }
/// }
/// ```
pub trait DocDefinition: Send + Sync + 'static {
    /// Stable persisted kind; `delta.*` belongs to the crate's own definitions by convention.
    const KIND: &'static str;
    /// Positive (checked when the definition is first used, at compile time). Versions belong
    /// to stored bases and deltas, not to the incarnation.
    const VERSION: u32;
    /// Owner and lifetime: [`SessionScope`], [`ConversationScope`] or [`TaskScope`].
    type Scope: DocScope;
    /// A family's seed, given to [`initial`](Self::initial) on the first acquisition of a
    /// member; `()` for a singleton.
    type Seed: Serialize + Send + 'static;
    /// Whether instances are family members addressed by key.
    const FAMILY: bool = false;

    /// The value of a new instance. Must be an object.
    fn initial(seed: &Self::Seed) -> Json;

    /// A complete current-version value from one stored at `from < VERSION`. Pure.
    /// Default: no migration.
    fn migrate(value: Json, from: u32) -> Result<Json, MigrateError> {
        drop(value);
        Err(MigrateError::Unsupported { from })
    }

    /// Whether this change is stored as a complete base. Runs once per change, after every
    /// other validation; `|_, _, _| false` keeps only the bases that are required anyway.
    fn checkpoint_when(value: &Json, ops: &[Op], info: CheckpointInfo) -> bool;
}

/// What [`DocDefinition::checkpoint_when`] knows besides the value and the batch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CheckpointInfo {
    /// Deltas already stored after the incarnation's newest base, excluding this change.
    pub deltas_since_base: u64,
}

/// Why [`DocDefinition::migrate`] produced no value.
#[derive(Debug, thiserror::Error)]
pub enum MigrateError {
    #[error("no migration from version {from}")]
    Unsupported { from: u32 },
    #[error(transparent)]
    Failed(Box<dyn StdError + Send + Sync>),
}

impl MigrateError {
    /// Wraps the definition's own error.
    pub fn failed(error: impl Into<Box<dyn StdError + Send + Sync>>) -> Self {
        Self::Failed(error.into())
    }
}

/// A document scope at the type level: who owns an instance and, for conversation
/// documents, its history and fork policy. Sealed.
pub trait DocScope: sealed::Sealed + Send + Sync + 'static {
    /// The owner's id: `()`, [`ConversationId`] or [`TaskId`].
    type Owner: Copy + fmt::Debug + Eq + Hash + Send + Sync + 'static;
    const KIND: DocumentScopeKind;
    /// `Some` exactly for conversation scopes.
    const POLICY: Option<ConversationPolicy>;

    /// The persisted scope of the instance owned by `owner`.
    fn scope(owner: Self::Owner) -> DocumentScope;
}

/// Session documents: current only, shared by forks, retired only explicitly.
#[derive(Debug)]
pub enum SessionScope {}

/// Task documents: current only, never copied, retired with their task.
#[derive(Debug)]
pub enum TaskScope {}

/// Conversation documents with history `H` ([`Latest`] or [`Rewindable`]) and fork policy
/// `F` ([`ForkAsOf`], [`ForkCurrent`] or [`ForkInitial`]). `ForkAsOf` requires `Rewindable`.
pub struct ConversationScope<H, F>(Infallible, PhantomData<fn() -> (H, F)>);

impl<H, F> fmt::Debug for ConversationScope<H, F> {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {}
    }
}

impl sealed::Sealed for SessionScope {}
impl sealed::Sealed for TaskScope {}
impl<H, F> sealed::Sealed for ConversationScope<H, F> {}

impl DocScope for SessionScope {
    type Owner = ();
    const KIND: DocumentScopeKind = DocumentScopeKind::Session;
    const POLICY: Option<ConversationPolicy> = None;

    fn scope((): ()) -> DocumentScope {
        DocumentScope::Session
    }
}

impl DocScope for TaskScope {
    type Owner = TaskId;
    const KIND: DocumentScopeKind = DocumentScopeKind::Task;
    const POLICY: Option<ConversationPolicy> = None;

    fn scope(task_id: TaskId) -> DocumentScope {
        DocumentScope::Task { task_id }
    }
}

impl<H: HistoryKind, F: ForkKind<H>> DocScope for ConversationScope<H, F> {
    type Owner = ConversationId;
    const KIND: DocumentScopeKind = DocumentScopeKind::Conversation;
    const POLICY: Option<ConversationPolicy> = Some(ConversationPolicy {
        history: H::HISTORY,
        fork: F::FORK,
    });

    fn scope(conversation_id: ConversationId) -> DocumentScope {
        DocumentScope::Conversation { conversation_id }
    }
}

/// A conversation document's history at the type level. Sealed.
pub trait HistoryKind: sealed::Sealed + Send + Sync + 'static {
    const HISTORY: History;
}

/// Current value only.
#[derive(Debug)]
pub enum Latest {}

/// Every revision is kept and readable at any commit of the lifetime.
#[derive(Debug)]
pub enum Rewindable {}

impl sealed::Sealed for Latest {}
impl sealed::Sealed for Rewindable {}

impl HistoryKind for Latest {
    const HISTORY: History = History::Latest;
}

impl HistoryKind for Rewindable {
    const HISTORY: History = History::Rewindable;
}

/// A fork policy allowed with history `H`. Sealed.
pub trait ForkKind<H>: sealed::Sealed + Send + Sync + 'static {
    const FORK: Fork;
}

/// The child gets the value at the fork entry's commit. Rewindable documents only.
#[derive(Debug)]
pub enum ForkAsOf {}

/// The child gets the value when the fork commits.
#[derive(Debug)]
pub enum ForkCurrent {}

/// The child starts from the initial value.
#[derive(Debug)]
pub enum ForkInitial {}

impl sealed::Sealed for ForkAsOf {}
impl sealed::Sealed for ForkCurrent {}
impl sealed::Sealed for ForkInitial {}

impl ForkKind<Rewindable> for ForkAsOf {
    const FORK: Fork = Fork::AsOf;
}

impl<H: HistoryKind> ForkKind<H> for ForkCurrent {
    const FORK: Fork = Fork::Current;
}

impl<H: HistoryKind> ForkKind<H> for ForkInitial {
    const FORK: Fork = Fork::Initial;
}

/// A conversation document, with its history and fork as associated types; implemented for
/// every definition whose scope is a [`ConversationScope`]. Bounds such as
/// `D: ConversationDoc<History = Rewindable>` select rewindable documents.
pub trait ConversationDoc: DocDefinition {
    type History: HistoryKind;
    type Fork: ForkKind<Self::History>;
}

impl<D, H, F> ConversationDoc for D
where
    D: DocDefinition<Scope = ConversationScope<H, F>>,
    H: HistoryKind,
    F: ForkKind<H>,
{
    type History = H;
    type Fork = F;
}
