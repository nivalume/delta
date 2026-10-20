//! The agent: what a conversation runs with. `delta.agent` stores names (model, extension
//! selection, tool filters, instructions, cwd), never code; each task phase resolves them
//! against its registry snapshot and the live settings into an [`Agent`].
//!
//! This module also holds what agent-layer tasks reach through their runtime
//! ([`AgentRuntime`], built on the session's [`AgentServices`]) and the settings types, so
//! every built-in task and extension task resolves agents the same way. Corresponds to
//! upstream `src/harness/agent.ts` (spec §7.1). See `docs/design/harness.md`, "The agent
//! document and resolution".

mod env;
mod error;
mod resolve;
mod runtime;
mod settings;

use delta_ai::model::ModelThinkingLevel;
use delta_chord::delta::Op;
use delta_chord::json::Json;
use serde::{Deserialize, Serialize};

pub use env::{EnvBuildError, EnvBuilder, EnvTarget};
pub use error::AgentError;
pub use resolve::{Agent, resolve_agent};
pub(crate) use runtime::report_host;
pub use runtime::{AgentRuntime, AgentServices, HookRuntime};
pub use settings::{
    AfterFailedRun, CompactionPolicy, CompactionSettings, DEFAULT_RESTART_LIMIT, DEFAULT_RETRY_POLICY, HarnessSettings,
    PartialCompactionPolicy, PartialProgressPolicy, PartialRetryPolicy, ProgressPolicy, QueueMode, ResolvedCompaction,
    Settings, SettingsSource,
};

use crate::documents::{CheckpointInfo, ConversationScope, DocAddress, DocDefinition, ForkAsOf, Rewindable};
use crate::model::ModelRef;
use crate::records::{ConversationId, ConversationRecord};
use crate::session::{Tx, TxError};

/// The `delta.agent` document: conversation scope, rewindable, a fork copies it as of the
/// fork entry, a complete base on every change, `{}` initially.
#[derive(Clone, Copy, Debug, Default)]
pub struct AgentDoc;

impl DocDefinition for AgentDoc {
    const KIND: &'static str = "delta.agent";
    const VERSION: u32 = 1;
    type Scope = ConversationScope<Rewindable, ForkAsOf>;
    type Seed = ();

    fn initial((): &()) -> Json {
        delta_chord::json!({})
    }

    fn checkpoint_when(_: &Json, _: &[Op], _: CheckpointInfo) -> bool {
        true
    }
}

/// Which installed extensions a conversation selects.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExtensionSelection {
    /// Exactly these, in this order.
    Exactly(Vec<String>),
    /// The default selection with `add` appended and `remove` dropped.
    Edit {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        add: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        remove: Vec<String>,
    },
}

/// Which of the selected extensions' tools are enabled.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolFilter {
    /// Exactly these, in this order.
    Exactly(Vec<String>),
    /// Every tool but these.
    Remove { remove: Vec<String> },
}

/// Which enabled, model-callable tools are offered to the model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelToolFilter {
    /// Exactly these, in this order, deferred ones included.
    Exactly(Vec<String>),
    /// Every non-deferred tool but `remove`, then the deferred tools in `add` (loaded).
    Edit {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        remove: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        add: Vec<String>,
    },
}

/// The stored `delta.agent`: names, never code. Nothing checks that a stored name is
/// installed; resolution skips unknown names.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<ModelThinkingLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extensions: Option<ExtensionSelection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolFilter>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_tools: Option<ModelToolFilter>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
}

/// One field of an [`AgentChange`]: leave it, clear it or replace it (upstream `undefined`,
/// `null` or a value).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Change<T> {
    #[default]
    Keep,
    Clear,
    Set(T),
}

impl<T> Change<T> {
    fn apply(self, field: &mut Option<T>) {
        match self {
            Self::Keep => {}
            Self::Clear => *field = None,
            Self::Set(value) => *field = Some(value),
        }
    }
}

/// A change of `delta.agent`; every field defaults to [`Change::Keep`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AgentChange {
    pub model: Change<ModelRef>,
    pub thinking_level: Change<ModelThinkingLevel>,
    pub extensions: Change<ExtensionSelection>,
    pub tools: Change<ToolFilter>,
    pub model_tools: Change<ModelToolFilter>,
    pub instructions: Change<String>,
    pub cwd: Change<String>,
}

impl AgentChange {
    /// Applies the change to a stored state.
    pub fn apply(self, state: &mut AgentState) {
        self.model.apply(&mut state.model);
        self.thinking_level.apply(&mut state.thinking_level);
        self.extensions.apply(&mut state.extensions);
        self.tools.apply(&mut state.tools);
        self.model_tools.apply(&mut state.model_tools);
        self.instructions.apply(&mut state.instructions);
        self.cwd.apply(&mut state.cwd);
    }
}

/// Edits `conversation`'s `delta.agent` as a typed state.
async fn edit_agent<R>(
    tx: &mut Tx<'_>,
    conversation: ConversationId,
    edit: impl FnOnce(&mut AgentState) -> R,
) -> Result<R, TxError> {
    let mut doc = tx.doc(DocAddress::<AgentDoc>::new(conversation)).await?;
    doc.update_as::<AgentState, R>(edit).map_err(TxError::callback)
}

/// Applies `change` to `conversation`'s `delta.agent`. It neither starts generation nor
/// appends a system entry; the next preparation does.
pub async fn configure(tx: &mut Tx<'_>, conversation: ConversationId, change: AgentChange) -> Result<(), TxError> {
    edit_agent(tx, conversation, |state| change.apply(state)).await
}

/// A tool round's `add_tools`: an `Exactly` tool filter gets each name it lacks appended, a
/// `Remove` one loses the names, an unset one stays unset (every tool is enabled already);
/// an unset or `Edit` model filter gets each name appended to `add` (loading deferred
/// tools), an `Exactly` one gets it appended. By name, without consulting the registry.
pub async fn add_tools(tx: &mut Tx<'_>, conversation: ConversationId, added: &[String]) -> Result<(), TxError> {
    edit_agent(tx, conversation, |state| {
        match &mut state.tools {
            Some(ToolFilter::Exactly(names)) => append_missing(names, added),
            Some(ToolFilter::Remove { remove }) => remove.retain(|name| !added.contains(name)),
            None => {}
        }
        match &mut state.model_tools {
            Some(ModelToolFilter::Exactly(names)) => append_missing(names, added),
            Some(ModelToolFilter::Edit { remove, add }) => {
                remove.retain(|name| !added.contains(name));
                append_missing(add, added);
            }
            None => {
                state.model_tools = Some(ModelToolFilter::Edit {
                    remove: Vec::new(),
                    add: dedupe(added),
                });
            }
        }
    })
    .await
}

fn append_missing(names: &mut Vec<String>, added: &[String]) {
    for name in added {
        if !names.contains(name) {
            names.push(name.clone());
        }
    }
}

fn dedupe(names: &[String]) -> Vec<String> {
    let mut unique = Vec::new();
    append_missing(&mut unique, names);
    unique
}

/// The built-in part of every commit that creates a conversation, for `delta.agent`: a fork
/// keeps its `as_of` copy; a new task-owned conversation copies the stored agent of its owner
/// task's conversation; a new ownerless one starts with `{}`.
pub async fn create_agent(tx: &mut Tx<'_>, record: &ConversationRecord) -> Result<(), TxError> {
    if record.parent.is_some() {
        return Ok(());
    }
    let owner = match record.owner {
        Some(owner) => Some(
            tx.doc(DocAddress::<AgentDoc>::new(owner.conversation_id))
                .await?
                .value()
                .clone(),
        ),
        None => None,
    };
    let mut agent = tx.doc(DocAddress::<AgentDoc>::new(record.id)).await?;
    if let Some(owner) = owner {
        agent.set(owner);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
