//! Claude Code's command and HTTP hooks for `delta-harness`.
//!
//! [`ClaudeHooks`] reads the `hooks` object of Claude Code's settings ([`HooksConfig`]) and
//! runs its handlers from the harness's hook seams: an ordinary [`Extension`] with tool,
//! generation and compaction hooks, plus an approval policy decorator for
//! `PermissionRequest`. Commands run through the conversation's execution environment with
//! the event's JSON on standard input; HTTP handlers get it as a POST body. Every event's
//! outcomes are memoized in the hook's task, so a phase that runs again after a restart
//! does not run its handlers again. See `docs/design/hooks.md`.
//!
//! [`Extension`]: delta_harness::registry::Extension

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

mod config;
mod extension;
mod protocol;
mod runner;

pub use config::{ConfigError, DEFAULT_TIMEOUT, HookCommand, HooksConfig, Matcher, MatcherGroup, SUPPORTED_EVENTS};
pub use extension::{CLAUDE_HOOKS, ClaudeHooks, ClaudeHooksOptions, OnError, TranscriptPath};
pub use protocol::{HookOutput, claude_tool_name};
pub use runner::Outcome;
