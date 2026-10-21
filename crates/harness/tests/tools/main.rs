//! The built-in tools on the local environment, called by a scripted model: `read`, `write`,
//! `edit` and `bash`, and the file mutation queue.
//!
//! Ported from upstream `test/tools/*.test.ts` where they apply. Each case runs in its own
//! temp directory, the conversation's environment, against memory and SQLite in a file.

#![allow(clippy::unwrap_used, clippy::too_many_lines)]

#[path = "../support/mod.rs"]
mod support;

/// Runs each case against memory and SQLite in a file, on a multi-threaded runtime.
macro_rules! cases {
    ($($case:ident),* $(,)?) => {
        mod memory {
            $(#[tokio::test(flavor = "multi_thread", worker_threads = 4)] async fn $case() { super::$case(crate::Backend::Memory).await; })*
        }
        mod sqlite {
            $(#[tokio::test(flavor = "multi_thread", worker_threads = 4)] async fn $case() { super::$case(crate::Backend::SqliteFile).await; })*
        }
    };
}

mod bash;
mod edit;
mod mutation;
mod read;
mod write;

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use delta_ai::types::{Message, ToolResultMessage, UserBlock};
use delta_chord::json;
use delta_harness::agent::LocalEnvBuilder;
use delta_harness::harness::{Conversation, Harness};
use delta_harness::inbox::SubmissionDraft;
use delta_harness::tool::ToolDiagnostic;
use delta_harness::tools::{BashOptions, coding_tools};
pub(crate) use support::Backend;

use crate::support::TempDir;
use crate::support::fixture::{Fixture, chat, ctx, entries, settled};
use crate::support::model::Reply;

/// A harness with the coding tools whose conversations work in a temp directory.
struct Workspace {
    fixture: Fixture,
    harness: Harness,
    conversation: Conversation,
    dir: TempDir,
    calls: AtomicU32,
}

/// One call's result as the transcript has it.
#[derive(Debug)]
struct Called {
    message: ToolResultMessage,
    diagnostics: Vec<ToolDiagnostic>,
}

impl Called {
    /// The output's text items, without the rendered diagnostics.
    fn output(&self) -> Vec<&str> {
        let texts: Vec<&str> = self
            .message
            .content
            .iter()
            .filter_map(|block| match block {
                UserBlock::Text(text) => Some(text.text.as_str()),
                UserBlock::Image(_) => None,
            })
            .collect();
        let end = texts.len() - usize::from(!self.diagnostics.is_empty());
        texts[..end].to_vec()
    }

    /// The only output text.
    fn text(&self) -> &str {
        match self.output()[..] {
            [text] => text,
            ref other => panic!("one text item: {other:?}"),
        }
    }

    /// The diagnostic with `code`.
    fn diagnostic(&self, code: &str) -> &ToolDiagnostic {
        self.diagnostics
            .iter()
            .find(|diagnostic| diagnostic.code.as_deref() == Some(code))
            .unwrap_or_else(|| panic!("a {code} diagnostic: {:?}", self.diagnostics))
    }

    fn messages(&self) -> Vec<&str> {
        self.diagnostics
            .iter()
            .map(|diagnostic| diagnostic.message.as_str())
            .collect()
    }
}

impl Workspace {
    async fn new(backend: Backend) -> Self {
        Self::with_bash(backend, BashOptions::default()).await
    }

    async fn with_bash(backend: Backend, bash: BashOptions) -> Self {
        let dir = TempDir::new();
        let mut fixture = Fixture::new(backend).await;
        fixture.env = Some(Arc::new(LocalEnvBuilder::new(dir.path())));
        fixture.registry.install(coding_tools(bash)).unwrap();
        let harness = fixture.open().await;
        let conversation = chat(&harness).await;
        Self {
            fixture,
            harness,
            conversation,
            dir,
            calls: AtomicU32::new(0),
        }
    }

    fn path(&self, name: &str) -> std::path::PathBuf {
        self.dir.path().join(name)
    }

    fn write(&self, name: &str, content: impl AsRef<[u8]>) {
        let path = self.path(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    fn read(&self, name: &str) -> String {
        std::fs::read_to_string(self.path(name)).unwrap()
    }

    /// The model calls each `(tool, arguments)` in one round, then answers; the results in
    /// call order.
    async fn calls(&self, calls: &[(&str, serde_json::Value)]) -> Vec<Called> {
        let ids: Vec<String> = calls
            .iter()
            .map(|_| format!("c{}", self.calls.fetch_add(1, Ordering::Relaxed)))
            .collect();
        let round: Vec<(&str, &str, serde_json::Value)> = calls
            .iter()
            .zip(&ids)
            .map(|((tool, arguments), id)| (id.as_str(), *tool, arguments.clone()))
            .collect();
        self.fixture
            .model
            .push([Reply::calls_with(&round), Reply::text("Done")]);
        let submission = self
            .conversation
            .submit(SubmissionDraft::input("go"), &ctx())
            .await
            .unwrap();
        settled(&submission).await;
        let mut found: Vec<Called> = entries(&self.conversation)
            .await
            .into_iter()
            .filter(|entry| entry.kind == "delta.tool-result")
            .filter_map(|entry| {
                let Some([Message::ToolResult(message)]) = entry.model.as_deref() else {
                    panic!("a tool result: {entry:?}")
                };
                ids.contains(&message.tool_call_id).then(|| Called {
                    message: message.clone(),
                    diagnostics: entry
                        .data
                        .as_ref()
                        .and_then(|data| data.get("diagnostics"))
                        .and_then(|diagnostics| json::from_json(diagnostics).ok())
                        .unwrap_or_default(),
                })
            })
            .collect();
        found.sort_by_key(|called| ids.iter().position(|id| *id == called.message.tool_call_id));
        assert_eq!(found.len(), calls.len(), "every call settled");
        found
    }

    /// One call of `tool`.
    async fn call(&self, tool: &str, arguments: serde_json::Value) -> Called {
        self.calls(&[(tool, arguments)]).await.pop().unwrap()
    }

    async fn close(self) {
        self.harness.close(&ctx()).await.unwrap();
    }
}

/// `path` as the environment prints it.
fn shown(path: &Path) -> String {
    path.to_str().unwrap().to_owned()
}
