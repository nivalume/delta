//! A JSON-RPC server for `delta-harness` speaking `delta-protocol`: a subset of Codex's
//! app-server v2 protocol, so a client written for Codex's app-server drives a delta
//! harness. [`AppServer::serve`] runs one connection over any reader and writer (stdio, a
//! socket), newline-delimited JSON without the `"jsonrpc"` field. See
//! `docs/design/app-server.md`.

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

mod connection;
mod history;
mod items;
mod threads;
mod translate;

use std::sync::Arc;

use delta_chord::context::Context;
use delta_harness::approval::PermissionMode;
use delta_harness::harness::Harness;
use delta_harness::model::ModelRef;
use tokio::io::{AsyncRead, AsyncWrite};

/// How the server answers and what threads start with.
#[derive(Clone, Debug)]
pub struct AppServerOptions {
    /// The model of a thread whose `thread/start` names none; also the provider of a bare
    /// model id.
    pub model: Option<ModelRef>,
    /// The working directory of a thread whose `thread/start` names none.
    pub cwd: String,
    /// The mode of a thread that set none, as the harness's `RulesPolicy` is configured;
    /// reported as `effectiveMode`.
    pub permission_mode: PermissionMode,
    /// `initialize`'s `userAgent`.
    pub user_agent: String,
    /// `initialize`'s `codexHome`: the server's data directory.
    pub home: String,
}

impl AppServerOptions {
    /// Options with `cwd`, no default model, the default mode and `delta/<version>`.
    pub fn new(cwd: impl Into<String>) -> Self {
        Self {
            model: None,
            cwd: cwd.into(),
            permission_mode: PermissionMode::Default,
            user_agent: format!("delta/{}", env!("CARGO_PKG_VERSION")),
            home: String::new(),
        }
    }
}

/// Why a connection ended abnormally.
#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    /// Reading or writing the connection failed.
    #[error("Connection failed: {0}")]
    Io(#[from] std::io::Error),
}

/// Serves a harness to protocol clients, one [`serve`](Self::serve) per connection.
#[derive(Clone)]
pub struct AppServer {
    harness: Harness,
    options: Arc<AppServerOptions>,
}

impl std::fmt::Debug for AppServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppServer")
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

impl AppServer {
    pub fn new(harness: Harness, options: AppServerOptions) -> Self {
        Self {
            harness,
            options: Arc::new(options),
        }
    }

    /// Runs one connection until the client closes its side, `ctx` is cancelled or the
    /// harness closes. Requests are answered in order; notifications of the threads the
    /// connection started, resumed or forked follow as they happen.
    pub async fn serve<R, W>(&self, reader: R, writer: W, ctx: &Context) -> Result<(), ServeError>
    where
        R: AsyncRead + Unpin + Send,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        connection::serve(self.harness.clone(), self.options.clone(), reader, writer, ctx).await
    }
}
