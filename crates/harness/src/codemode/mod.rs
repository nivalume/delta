//! Code mode (`codemode` feature): a tool that runs model-written JavaScript in a QuickJS
//! sandbox ([`delta_codemode`]), whose script calls the conversation's callable tools and
//! survives a restart.
//!
//! A script is in-memory VM state, so a restart while it waits for the last of 200 reads would
//! lose the script and every read it made. The VM only runs when the host hands it a result;
//! given the same start data, clock, seed and results in the same order, it makes the same calls
//! with the same arguments. The tool is replay-safe: after a restart `execute` runs again, starts
//! the script again from the stored start data, and hands it the results it got before, in the
//! order it got them, from a journal written before each delivery. Nested tool calls reattach to
//! their tasks by key (`c<VM call id>`), so they never run twice unless their own replay policy
//! reruns them. Then the script continues live.
//!
//! Documents, scoped to the code mode call's task and gone when it settles:
//! [`CodemodeRunDoc`] (the start data and the outcome) and [`CodemodeJournalDoc`] (one entry
//! per delivered result). [`CodemodeStoreDoc`] keeps what scripts `store()`, per conversation.
//!
//! Port of upstream `@earendil-works/pi-durable/codemode` (spec §7.6); see
//! `docs/design/codemode.md`.

mod describe;
mod documents;
mod models;
mod output;
mod run;
mod search;
mod tool;

use std::fmt;
use std::sync::Arc;

use delta_chord::context::Context;
use delta_codemode::Sandbox;
use futures::future::BoxFuture;
use serde_json::Value;

pub use describe::{DEFAULT_CODEMODE_INLINE_BUDGET, codemode_section};
pub use documents::{
    CallStatus, CodemodeJournalDoc, CodemodeNestedCall, CodemodeRunDoc, CodemodeStart, CodemodeStoreDoc,
    CodemodeToolDetails, JournalEntry, JournalState, JournaledGlobal, RunState, StartNamespace, StartTool, StoreState,
    StoredResult,
};
pub use models::{CLASSIFY_TOOL, GENERATE_IMAGES_TOOL, MAX_CONCURRENT_MODEL_CALLS};
pub use output::DEFAULT_MAX_OUTPUT_TOKENS;
pub use tool::{CODEMODE_TOOL_NAME, codemode_description, codemode_extension, codemode_tool};

use crate::tool::{ToolApi, ToolError};

/// Version of everything in this module that turns stored results into what a script sees:
/// result mapping, globals, start data. Bumped when that changes, so a call started before an
/// upgrade is not replayed by different code.
pub const CODEMODE_MODULE_VERSION: u32 = 1;

/// Reserved name prefix of code mode's internal tools; they are never listed to scripts or the
/// model.
pub const CODEMODE_INTERNAL_PREFIX: &str = "delta.codemode.";

/// A group of tools named by a common prefix, such as an MCP server's `mcp__github__`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodemodeNamespace {
    pub name: String,
    pub description: Option<String>,
    pub instructions: Option<String>,
    /// Tool names starting with it belong to the namespace; the longest matching prefix wins.
    pub prefix: String,
}

impl CodemodeNamespace {
    pub fn new(name: impl Into<String>, prefix: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            instructions: None,
            prefix: prefix.into(),
        }
    }

    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
}

/// The namespace of a tool: the one with the longest prefix its name starts with.
pub(crate) fn namespace_of<'a>(name: &str, namespaces: &'a [CodemodeNamespace]) -> Option<&'a CodemodeNamespace> {
    namespaces
        .iter()
        .filter(|namespace| name.starts_with(&namespace.prefix))
        .fold(None, |found: Option<&CodemodeNamespace>, namespace| match found {
            Some(found) if found.prefix.len() >= namespace.prefix.len() => Some(found),
            _ => Some(namespace),
        })
}

/// A host function scripts call as a top-level global (`a.b` names group into an object). Its
/// result is journaled, so a replay resolves with it again without calling it.
pub trait CodemodeGlobal: Send + Sync {
    /// An identifier, or `namespace.member`.
    fn name(&self) -> &str;

    /// The call receives the whole argument list instead of the first argument.
    fn spread(&self) -> bool {
        false
    }

    /// Answers a call with what the script passed (`None` for `undefined`); `Ok(None)`
    /// resolves with `undefined`, an error rejects with its message.
    fn execute<'a>(
        &'a self,
        args: Option<Value>,
        api: &'a ToolApi<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<Option<Value>, ToolError>>;
}

impl fmt::Debug for dyn CodemodeGlobal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CodemodeGlobal")
            .field("name", &self.name())
            .finish_non_exhaustive()
    }
}

/// What the `codemode` tool runs scripts with.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CodemodeOptions {
    /// Runs the scripts; its own tools and globals are not used.
    pub sandbox: Arc<Sandbox>,
    /// Groups of tools by name prefix, such as MCP servers.
    pub namespaces: Vec<CodemodeNamespace>,
    /// Host functions scripts call as globals.
    pub globals: Vec<Arc<dyn CodemodeGlobal>>,
    /// Estimated tokens the tool declarations in the `codemode` section may use. Default
    /// [`DEFAULT_CODEMODE_INLINE_BUDGET`].
    pub inline_budget: usize,
    /// Gives scripts `models.*`: the model catalog, classifiers and image generation. Default
    /// on.
    pub models: bool,
}

impl CodemodeOptions {
    pub fn new(sandbox: Arc<Sandbox>) -> Self {
        Self {
            sandbox,
            namespaces: Vec::new(),
            globals: Vec::new(),
            inline_budget: DEFAULT_CODEMODE_INLINE_BUDGET,
            models: true,
        }
    }

    #[must_use]
    pub fn with_models(mut self, models: bool) -> Self {
        self.models = models;
        self
    }

    #[must_use]
    pub fn with_inline_budget(mut self, budget: usize) -> Self {
        self.inline_budget = budget;
        self
    }

    #[must_use]
    pub fn with_namespace(mut self, namespace: CodemodeNamespace) -> Self {
        self.namespaces.push(namespace);
        self
    }

    #[must_use]
    pub fn with_global(mut self, global: Arc<dyn CodemodeGlobal>) -> Self {
        self.globals.push(global);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_longest_prefix_names_the_namespace() {
        let namespaces = [
            CodemodeNamespace::new("github", "mcp__github__"),
            CodemodeNamespace::new("issues", "mcp__github__issue"),
        ];
        let name = |tool: &str| namespace_of(tool, &namespaces).map(|namespace| namespace.name.as_str());
        assert_eq!(name("mcp__github__issue_read"), Some("issues"));
        assert_eq!(name("mcp__github__pull"), Some("github"));
        assert_eq!(name("read"), None);
    }
}
