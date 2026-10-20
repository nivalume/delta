//! Durable agent harness: conversations, model turns, tool calls and application state are
//! committed to storage before they are shown, and a reopened storage resumes where it stopped.
//!
//! delta's counterpart of [`@earendil-works/pi-durable`](https://github.com/earendil-works/pi/tree/main/packages/durable).
//! Design notes live in `docs/design/` of this crate; the modules are not implemented yet.
