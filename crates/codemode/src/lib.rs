//! Runs model-written JavaScript in a QuickJS VM whose only capability is calling the functions
//! the host injects: nested tool calls never enter the model's context, only the script's output
//! and return value do. Port of `@earendil-works/pi-codemode`; see `docs/design/sandbox.md` and
//! `docs/design/declarations.md`.
//!
//! [`Sandbox::execute`] runs a script against the sandbox's own tools. [`Sandbox::start`] runs it
//! deterministically, with the host answering every call, so a host can replay it after a
//! restart from a log of its calls' results.

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

mod declarations;
mod identifier;
mod sandbox;
mod source;
mod types;
mod vm;

pub use declarations::{
    DEFAULT_INPUT_SCHEMA_MAX_CHARS, GlobalDeclaration, ToolDeclaration, render_declarations, render_tool_output_type,
    render_tool_sample, render_tool_signature, schema_to_type,
};
pub use identifier::to_codemode_identifier;
pub use sandbox::{
    CodemodeTool, DEFAULT_TIMEOUT, DeliverError, ExecuteOptions, FnCodemodeTool, Run, RunEvent, RunHandle, RunOptions,
    Sandbox, SandboxError, SandboxOptions,
};
pub use source::{
    CODEMODE_OPTIONS_PREFIX, CODEMODE_SOURCE_GRAMMAR, ParsedSource, SourceError, SourceOptions, parse_codemode_source,
};
pub use types::{
    CallRecord, CallStatus, CallTarget, CodemodeError, CodemodeResult, DeclaredGlobal, DeclaredTool, During, ErrorKind,
    HostCall, MAX_OUTPUT_CHARS, MAX_OUTPUT_ITEMS, MAX_STORE_TOTAL_CHARS, MAX_STORE_VALUE_CHARS, Outcome, OutputItem,
    StoreWrites,
};
