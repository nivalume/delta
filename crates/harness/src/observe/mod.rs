//! Observation (spec §9): the conversation view, a `delta_chord::State` per observed
//! conversation advanced from the commit publications.
//!
//! Every observer sees committed state only and never blocks a writer: a mount is built on
//! the commit line from the committed transcript and documents, and advanced on the line
//! with the ops each commit wrote. Corresponds to upstream `src/harness/view.ts`. See
//! `docs/design/observe.md`.

mod view;

pub(crate) use view::ViewMounts;
pub use view::{ConversationView, MOUNTED_DOCUMENTS, ViewState, ViewSubscription};
