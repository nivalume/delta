//! Reports: failures that do not fail the operation that saw them, and the Session's own
//! failure, given to the host's [`Reporter`].

use std::error::Error as StdError;
use std::sync::Arc;

use crate::records::{ConversationId, TaskId};

/// Receives reported errors with what they came from (upstream `onReport`). Runs
/// synchronously where the error happened, so it must not block.
pub type Reporter = Arc<dyn Fn(&(dyn StdError + Send + Sync), &ReportInfo) + Send + Sync>;

/// What a reported error came from. Every task variant carries the task's id, kind and
/// conversation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReportInfo {
    /// The Session failed; reported once.
    SessionFailed,
    TaskFaulted {
        task: ReportTask,
        reason: FaultReason,
    },
    TaskParked {
        task: ReportTask,
    },
    MigrationFailed {
        task: ReportTask,
    },
    TaskOutdated {
        task: ReportTask,
        reason: OutdatedReason,
    },
    TaskError {
        task: ReportTask,
    },
    ExtensionError {
        task: Option<ReportTask>,
        conversation: Option<ConversationId>,
        extension: Option<String>,
    },
    HostError {
        callback: HostCallback,
        task: Option<ReportTask>,
    },
    /// A host listener failed; it never fails what ran it.
    ListenerError,
}

/// The task a report is about.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReportTask {
    pub id: TaskId,
    pub kind: String,
    pub conversation_id: ConversationId,
    pub extension: Option<String>,
}

/// Why a task faulted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultReason {
    /// Its handler returned an error.
    Threw,
    /// It was restarted more often than allowed.
    RestartLimit,
}

/// Why a stored task cannot run with the installed definitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutdatedReason {
    MissingTask,
    IncompatibleTask,
}

/// The host callback that failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostCallback {
    Now,
    Settings,
    Registry,
    Env,
    Models,
    OnWake,
}
