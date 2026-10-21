//! Durable agent harness: conversations, model turns, tool calls and application state are
//! committed to storage before they are shown, and a reopened storage resumes where it stopped.
//!
//! delta's counterpart of [`@earendil-works/pi-durable`](https://github.com/earendil-works/pi/tree/main/packages/durable).
//! Design notes live in `docs/design/` of this crate.
//!
//! The modules form layers; each depends only on the layers below it:
//!
//! - surface: [`harness`](mod@harness) ([`Harness`](harness::Harness),
//!   [`Conversation`](harness::Conversation), [`Submission`](harness::Submission)) and
//!   [`observe`] (the conversation view).
//! - agent: [`registry`] (extensions and hooks), [`agent`] (`delta.agent`, resolution,
//!   settings, the runtime of agent-layer tasks), [`inbox`], [`context`], [`prompt`],
//!   [`generation`], [`usage`], [`model`] (the [`ModelClient`](model::ModelClient) seam),
//!   [`tool`] (tools and the `delta.tool` task), [`approval`] (the policy, requests and
//!   decisions), [`compaction`] (thresholds and the `delta.compaction` task), and, with the
//!   `codemode` feature, `codemode` (scripts that call the conversation's tools).
//! - kernel: [`records`], [`documents`], [`storage`], [`session`], [`scheduler`].
//! - side: [`env`](mod@env), the execution environment and its local implementation,
//!   [`search`], the BM25 tool ranking, and [`tools`], the built-in `read`, `write`, `edit`,
//!   `bash` and `tool_search`.

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

pub mod agent;
pub mod approval;
#[cfg(feature = "codemode")]
pub mod codemode;
pub mod compaction;
pub mod context;
pub mod documents;
pub mod env;
pub mod generation;
pub mod harness;
pub mod inbox;
pub mod model;
pub mod observe;
pub mod prompt;
pub mod records;
pub mod registry;
pub mod scheduler;
pub mod search;
pub mod session;
pub mod storage;
pub mod tool;
pub mod tools;
pub mod usage;
