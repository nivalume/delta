//! The system prompt: prompt sections, their rendering, and the positional `delta.system`
//! entries that make the transcript's replayed sections and tools equal the desired ones.
//!
//! Durable has no stored prompt: the resolved agent produces the desired sections for each
//! request, and the transcript's `delta.system` entries are the only durable record of what
//! the model saw. Preparation appends the smallest patch, so provider prompt caches stay
//! stable. Corresponds to upstream `src/harness/prompt.ts` (spec §7.4). See
//! `docs/design/context.md`, "System prompt and dynamic tools".

mod plan;

use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;

use delta_chord::context::{Cancelled, Context};
use futures::future::BoxFuture;
use indexmap::IndexMap;

pub use plan::{SYSTEM_ENTRY, plan_system_entries, replay_sections};

use crate::agent::Agent;
use crate::env::ExecutionEnv;
use crate::records::ConversationId;

/// Why a section did not render; it keeps its shown text, if any.
pub type SectionError = Box<dyn StdError + Send + Sync>;

/// What a section renders from.
pub struct PromptInput<'a> {
    pub conversation: ConversationId,
    pub agent: &'a Agent,
    /// The conversation's environment, when the host builds one.
    pub env: Option<&'a dyn ExecutionEnv>,
    /// The sections in effect in the transcript, after replaying its system messages.
    pub shown: &'a IndexMap<String, String>,
}

impl fmt::Debug for PromptInput<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PromptInput")
            .field("conversation", &self.conversation)
            .field("env", &self.env.is_some())
            .field("shown", &self.shown)
            .finish_non_exhaustive()
    }
}

/// One keyed part of the system prompt. Renderers must be deterministic for equal inputs:
/// any change in rendered text appends a patch and invalidates provider prompt caches.
pub trait PromptSection: Send + Sync {
    /// Matches `^[a-z][a-z0-9_-]*$`; `instructions` is reserved.
    fn key(&self) -> &str;

    /// Whether the text is wrapped as `<key>\n...\n</key>`.
    fn tag(&self) -> bool {
        true
    }

    /// The section's text; `None` omits it.
    fn render<'a>(
        &'a self,
        input: &'a PromptInput<'a>,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Result<Option<String>, SectionError>>;
}

impl fmt::Debug for dyn PromptSection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PromptSection")
            .field("key", &self.key())
            .finish_non_exhaustive()
    }
}

/// A section with fixed text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextSection {
    pub key: String,
    pub text: String,
    pub tag: bool,
}

impl TextSection {
    /// A tagged section `key` with `text`.
    pub fn new(key: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            text: text.into(),
            tag: true,
        }
    }
}

impl PromptSection for TextSection {
    fn key(&self) -> &str {
        &self.key
    }

    fn tag(&self) -> bool {
        self.tag
    }

    fn render<'a>(
        &'a self,
        _input: &'a PromptInput<'a>,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, Result<Option<String>, SectionError>> {
        Box::pin(async move { Ok(Some(self.text.clone())) })
    }
}

/// Renders `sections` in order into the desired sections. `None` omits one; tagged text is
/// `<key>\n...\n</key>`. A section that fails keeps its `shown` text, if any, and is given to
/// `report`; a failure once `ctx` is cancelled propagates.
pub async fn render_sections(
    sections: &[Arc<dyn PromptSection>],
    input: &PromptInput<'_>,
    mut report: impl FnMut(&SectionError, &dyn PromptSection),
    ctx: &Context,
) -> Result<IndexMap<String, String>, Cancelled> {
    let mut desired = IndexMap::new();
    for section in sections {
        let key = section.key();
        match section.render(input, ctx).await {
            Ok(Some(text)) => {
                let text = if section.tag() {
                    format!("<{key}>\n{text}\n</{key}>")
                } else {
                    text
                };
                desired.insert(key.to_owned(), text);
            }
            Ok(None) => {}
            Err(error) => {
                if let Some(reason) = ctx.reason() {
                    return Err(Cancelled { reason });
                }
                report(&error, section.as_ref());
                if let Some(kept) = input.shown.get(key) {
                    desired.insert(key.to_owned(), kept.clone());
                }
            }
        }
    }
    Ok(desired)
}

#[cfg(test)]
mod tests;
