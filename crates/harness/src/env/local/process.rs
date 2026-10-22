//! Process trees of local commands: how a command's tree is killed, the registry
//! [`Shell::cleanup`](crate::env::Shell::cleanup) kills from, and exit codes.
//!
//! On Unix each command leads its own process group (`process_group(0)`), and killing the tree
//! sends `SIGKILL` to that group. The crate forbids `unsafe` and has no libc binding, so the
//! signal is sent by `/bin/sh`'s `kill -9 -<pgid>` instead of `killpg(2)`; the direct child is
//! killed through tokio as well, so a missing `/bin/sh` still stops the command itself. A
//! descendant that leaves the group (`setsid`) is out of reach, as with upstream. On Windows
//! `taskkill /F /T` kills the tree.

use std::collections::HashMap;
use std::process::{ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// The process tree of one running command, by the process id of its root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ProcessTree {
    pid: u32,
}

impl ProcessTree {
    pub(super) fn new(pid: u32) -> Self {
        Self { pid }
    }

    /// The command that kills this tree.
    fn killer(self) -> std::process::Command {
        #[cfg(unix)]
        {
            // A negative pid names the process group the command leads.
            let mut command = std::process::Command::new("/bin/sh");
            command.args(["-c", "kill -9 -\"$1\"", "sh", &self.pid.to_string()]);
            command
        }
        #[cfg(windows)]
        {
            let root = std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into());
            let mut command =
                std::process::Command::new(std::path::Path::new(&root).join("System32").join("taskkill.exe"));
            command.args(["/F", "/T", "/PID", &self.pid.to_string()]);
            command
        }
    }

    /// Kill every process of the tree and wait for the killer to finish. Best effort: a tree
    /// that already exited is not an error.
    pub(super) async fn kill(self) {
        let mut command = tokio::process::Command::from(self.killer());
        command.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
        let _ = command.status().await;
    }

    /// Kill the tree without waiting, from synchronous code such as `Drop`. The killer is
    /// reaped by a short-lived thread.
    pub(super) fn kill_detached(self) {
        let mut command = self.killer();
        command.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
        if let Ok(mut killer) = command.spawn() {
            let _ = std::thread::Builder::new()
                .name("delta-env-kill".to_owned())
                .spawn(move || killer.wait());
        }
    }
}

/// The commands an environment still runs, for [`Shell::cleanup`](crate::env::Shell::cleanup).
#[derive(Debug, Default)]
pub(super) struct ChildRegistry {
    next_id: AtomicU64,
    running: Mutex<HashMap<u64, ProcessTree>>,
}

impl ChildRegistry {
    fn running(&self) -> MutexGuard<'_, HashMap<u64, ProcessTree>> {
        self.running.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Track `tree` until the returned guard is settled or dropped.
    pub(super) fn track(&self, tree: ProcessTree) -> Tracked<'_> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.running().insert(id, tree);
        Tracked {
            registry: self,
            id,
            tree,
            settled: false,
        }
    }

    /// Kill every tracked tree and stop tracking it.
    pub(super) async fn kill_all(&self) {
        let trees: Vec<ProcessTree> = self.running().drain().map(|(_, tree)| tree).collect();
        for tree in trees {
            tree.kill().await;
        }
    }
}

/// A tracked command. Dropping it before [`settle`](Self::settle), as when the caller drops
/// the `exec` future, kills the command's tree.
#[derive(Debug)]
pub(super) struct Tracked<'r> {
    registry: &'r ChildRegistry,
    id: u64,
    tree: ProcessTree,
    settled: bool,
}

impl Tracked<'_> {
    pub(super) fn tree(&self) -> ProcessTree {
        self.tree
    }

    /// The command finished; stop tracking it.
    pub(super) fn settle(mut self) {
        self.settled = true;
    }
}

impl Drop for Tracked<'_> {
    fn drop(&mut self) {
        self.registry.running().remove(&self.id);
        if !self.settled {
            self.tree.kill_detached();
        }
    }
}

/// The exit code of a finished process: its own; `128 + signal` when a signal killed it; `1`
/// when unknown.
pub(super) fn exit_code(status: ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return 128 + signal;
        }
    }
    1
}
