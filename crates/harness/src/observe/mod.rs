//! Observation (spec §9): the conversation view, a `delta_chord::State` per observed
//! conversation advanced from the commit publications, the agent events derived from its
//! revisions, and the Session's task graph.
//!
//! Every observer sees committed state only and never blocks a writer: a mount is built on
//! the commit line from the committed transcript and documents, and advanced on the line
//! with the ops each commit wrote. Corresponds to upstream `src/harness/view.ts`, `events.ts` and `task-graph.ts`. See
//! `docs/design/observe.md`.

mod events;
mod graph;
mod stream;
mod translate;
mod view;

pub use events::{AgentEvent, MessageChange, OutputChange, QueuedItem, QueuedMode, Snapshot, ToolEventCall};
pub(crate) use graph::GraphMounts;
pub use graph::{OutcomeStatus, TaskGraph, TaskGraphNode, TaskGraphState, TaskGraphSubscription, TaskGraphView};
pub(crate) use stream::attach;
pub use stream::{AgentEventStream, EVENT_BACKLOG};
pub(crate) use view::ViewMounts;
pub use view::{ConversationView, MOUNTED_DOCUMENTS, ViewState, ViewSubscription};
