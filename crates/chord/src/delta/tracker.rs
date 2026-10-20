//! Revision sequences and two-phase changes.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::json::Json;

use super::draft::{Draft, Edit};
use super::{MAX_OPS, Op, diff};

static NEXT_TRACKER: AtomicU64 = AtomicU64::new(1);

/// One sequence of revisions: the current value and how many changes were adopted.
#[derive(Debug)]
pub struct Tracker {
    value: Json,
    revision: u64,
    id: u64,
}

/// Why [`Tracker::adopt`] refused a preparation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AdoptError {
    #[error("the preparation belongs to another tracker")]
    Foreign,
    #[error("the preparation was made from revision {prepared}, but the tracker is at revision {current}")]
    Stale { prepared: u64, current: u64 },
}

impl Tracker {
    pub fn new(initial: Json) -> Self {
        Self {
            value: initial,
            revision: 0,
            id: NEXT_TRACKER.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// The current revision.
    pub fn value(&self) -> &Json {
        &self.value
    }

    /// The number of adopted preparations, empty ones included.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Start a change from the current revision. O(1).
    pub fn begin(&self) -> Change {
        Change {
            base: self.value.clone(),
            edit: Edit::new(self.value.clone()),
            revision: self.revision,
            tracker: self.id,
        }
    }

    /// Prepare `value` as a whole replacement: `["r", value]`, or an empty batch when it equals
    /// the current revision.
    pub fn prepare_replace(&self, value: Json) -> Prepared {
        if value == self.value {
            return self.prepared(self.value.clone(), Vec::new());
        }
        let ops = vec![Op::Replace(value.clone())];
        self.prepared(value, ops)
    }

    /// Prepare `value` with the ops of [`diff`] from the current revision. A value built from
    /// the current revision shares its unchanged parts, so this costs time proportional to the
    /// change.
    pub fn prepare_diff(&self, value: Json) -> Prepared {
        let ops = diff(&self.value, &value);
        if ops.is_empty() {
            return self.prepared(self.value.clone(), ops);
        }
        let ops = if ops.len() > MAX_OPS {
            vec![Op::Replace(value.clone())]
        } else {
            ops
        };
        self.prepared(value, ops)
    }

    /// Make a preparation's value the current revision and return its ops.
    ///
    /// Refuses a preparation of another tracker and one made from an earlier revision: adopting
    /// a change makes every other change begun from the same revision stale.
    pub fn adopt(&mut self, prepared: Prepared) -> Result<Vec<Op>, AdoptError> {
        if prepared.tracker != self.id {
            return Err(AdoptError::Foreign);
        }
        if prepared.revision != self.revision {
            return Err(AdoptError::Stale {
                prepared: prepared.revision,
                current: self.revision,
            });
        }
        self.value = prepared.value;
        self.revision += 1;
        Ok(prepared.ops)
    }

    fn prepared(&self, value: Json, ops: Vec<Op>) -> Prepared {
        Prepared {
            base: self.value.clone(),
            value,
            ops,
            revision: self.revision,
            tracker: self.id,
        }
    }
}

/// Edits on top of one revision. Owned and `Send`: it may stay open across `.await`, and
/// several may be open at once. Dropping it discards the edits.
pub struct Change {
    base: Json,
    edit: Edit,
    revision: u64,
    tracker: u64,
}

impl Change {
    /// A handle on the whole value.
    pub fn draft(&mut self) -> Draft<'_> {
        Draft::root(&mut self.edit)
    }

    /// The revision this change started from.
    pub fn base(&self) -> &Json {
        &self.base
    }

    /// The value with the edits so far.
    pub fn value(&self) -> &Json {
        &self.edit.working
    }

    /// The new revision and its batch. The batch is empty, and the value is the base itself,
    /// when the edits changed nothing.
    pub fn prepare(self) -> Prepared {
        let Self {
            base,
            edit,
            revision,
            tracker,
        } = self;
        let (working, ops) = edit.finish();
        let (value, ops) = match ops {
            None => (working.clone(), vec![Op::Replace(working)]),
            Some(ops) if ops.is_empty() || working == base => (base.clone(), Vec::new()),
            Some(ops) => (working, ops),
        };
        Prepared {
            base,
            value,
            ops,
            revision,
            tracker,
        }
    }
}

impl fmt::Debug for Change {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Change")
            .field("revision", &self.revision)
            .field("value", &self.edit.working)
            .finish_non_exhaustive()
    }
}

/// A prepared revision and the batch from its base. Adopt it with [`Tracker::adopt`]; dropping
/// it aborts it.
#[derive(Debug, Clone)]
pub struct Prepared {
    base: Json,
    value: Json,
    ops: Vec<Op>,
    revision: u64,
    tracker: u64,
}

impl Prepared {
    pub fn base(&self) -> &Json {
        &self.base
    }

    pub fn value(&self) -> &Json {
        &self.value
    }

    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// Whether the batch is empty (nothing changed).
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// The revision this preparation was made from.
    pub fn revision(&self) -> u64 {
        self.revision
    }
}
