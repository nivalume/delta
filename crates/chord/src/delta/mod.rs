//! Immutable JSON revisions and the op batches between them.
//!
//! A [`Tracker`] holds the current revision. A [`Change`] edits a copy through [`Draft`]
//! handles and [`prepare`](Change::prepare)s the next revision with its batch; adopting the
//! preparation makes it current. Ops serialize as upstream chord's tuples
//! (`["s", path, value]` and so on). See `docs/design/delta.md`.

mod apply;
mod diff;
mod draft;
mod tracker;

use std::fmt;

use serde::de::{self, Deserializer, IgnoredAny, SeqAccess, Visitor};
use serde::ser::{SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};

use crate::json::{Json, Kind, Path, Seg, Str};

pub use apply::{ApplyError, ApplyErrorKind, applied, apply};
pub use diff::{diff, overlap};
pub use draft::{ArrayDraft, Draft, ObjectDraft, StrDraft};
pub use tracker::{AdoptError, Change, Prepared, Tracker};

/// Batches longer than this fold into one `["r", value]`, so replaying a batch stays cheap.
pub const MAX_OPS: usize = 4096;

/// Path segments that upstream's JavaScript appliers refuse (prototype pollution). delta
/// accepts them in [`apply`], and its tracker never emits them, so batches stay readable by pi.
pub const RESERVED_SEGMENTS: [&str; 3] = ["__proto__", "constructor", "prototype"];

pub(crate) fn is_reserved(seg: &Seg) -> bool {
    seg.as_key().is_some_and(|key| RESERVED_SEGMENTS.contains(&key))
}

/// One step from a revision to the next.
#[derive(Clone, PartialEq)]
pub enum Op {
    /// `["r", value]`: replace the whole value.
    Replace(Json),
    /// `["s", path, value]`: set an object key, or an array element (`index == len` appends).
    Set(Path, Json),
    /// `["d", path]`: delete an object key or remove an array element.
    Delete(Path),
    /// `["a", path, text]`: append to a string.
    Append(Path, Str),
    /// `["t", path, count]`: remove `count` UTF-16 code units from the front of a string.
    TrimStart(Path, u64),
    /// `["p", path, index, remove, items]`: splice an array.
    Splice {
        path: Path,
        index: usize,
        remove: usize,
        items: Vec<Json>,
    },
    /// `["m", path, permutation]`: reorder an array, `new[i] = old[permutation[i]]`.
    Permute(Path, Vec<usize>),
}

impl Op {
    /// The op's tag in the tuple form.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Replace(_) => "r",
            Self::Set(..) => "s",
            Self::Delete(_) => "d",
            Self::Append(..) => "a",
            Self::TrimStart(..) => "t",
            Self::Splice { .. } => "p",
            Self::Permute(..) => "m",
        }
    }

    /// The path the op addresses; empty for `r`. For `p` and `m` it is the array's path.
    pub fn path(&self) -> &[Seg] {
        match self {
            Self::Replace(_) => &[],
            Self::Set(path, _)
            | Self::Delete(path)
            | Self::Append(path, _)
            | Self::TrimStart(path, _)
            | Self::Splice { path, .. }
            | Self::Permute(path, _) => path,
        }
    }

    /// The same op addressed under `prefix`. An `r` becomes a set of `prefix` (which must
    /// then be non-empty).
    #[must_use]
    pub fn prefixed(self, prefix: &[Seg]) -> Self {
        if prefix.is_empty() {
            return self;
        }
        match self {
            Self::Replace(value) => Self::Set(Path::from(prefix), value),
            Self::Set(path, value) => Self::Set(Path::from(prefix).concat(&path), value),
            Self::Delete(path) => Self::Delete(Path::from(prefix).concat(&path)),
            Self::Append(path, text) => Self::Append(Path::from(prefix).concat(&path), text),
            Self::TrimStart(path, count) => Self::TrimStart(Path::from(prefix).concat(&path), count),
            Self::Splice {
                path,
                index,
                remove,
                items,
            } => Self::Splice {
                path: Path::from(prefix).concat(&path),
                index,
                remove,
                items,
            },
            Self::Permute(path, permutation) => Self::Permute(Path::from(prefix).concat(&path), permutation),
        }
    }

    /// Whether a batch starts with a replacement. A batch holds at most one `r`, first.
    pub fn is_base(ops: &[Op]) -> bool {
        matches!(ops.first(), Some(Self::Replace(_)))
    }
}

impl fmt::Debug for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&serde_json::to_string(self).expect("ops always serialize"))
    }
}

// ─── Tuple form ──────────────────────────────────────────────────────────────

impl Serialize for Op {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let len = match self {
            Self::Replace(_) | Self::Delete(_) => 2,
            Self::Set(..) | Self::Append(..) | Self::TrimStart(..) | Self::Permute(..) => 3,
            Self::Splice { .. } => 5,
        };
        let mut seq = serializer.serialize_seq(Some(len))?;
        seq.serialize_element(self.tag())?;
        match self {
            Self::Replace(value) => seq.serialize_element(value)?,
            Self::Set(path, value) => {
                seq.serialize_element(path)?;
                seq.serialize_element(value)?;
            }
            Self::Delete(path) => seq.serialize_element(path)?,
            Self::Append(path, text) => {
                seq.serialize_element(path)?;
                seq.serialize_element(&**text)?;
            }
            Self::TrimStart(path, count) => {
                seq.serialize_element(path)?;
                seq.serialize_element(count)?;
            }
            Self::Splice {
                path,
                index,
                remove,
                items,
            } => {
                seq.serialize_element(path)?;
                seq.serialize_element(index)?;
                seq.serialize_element(remove)?;
                seq.serialize_element(items)?;
            }
            Self::Permute(path, permutation) => {
                seq.serialize_element(path)?;
                seq.serialize_element(permutation)?;
            }
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for Op {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(OpVisitor)
    }
}

struct OpVisitor;

fn element<'de, A: SeqAccess<'de>, T: Deserialize<'de>>(seq: &mut A, tag: &str, what: &str) -> Result<T, A::Error> {
    seq.next_element()?
        .ok_or_else(|| de::Error::custom(format_args!("\"{tag}\" op is missing its {what}")))
}

fn non_empty<E: de::Error>(path: Path, tag: &str) -> Result<Path, E> {
    if path.is_empty() {
        Err(E::custom(format_args!("\"{tag}\" op cannot target the root")))
    } else {
        Ok(path)
    }
}

impl<'de> Visitor<'de> for OpVisitor {
    type Value = Op;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an op tuple such as [\"s\", path, value]")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Op, A::Error> {
        let tag: String = seq.next_element()?.ok_or_else(|| de::Error::custom("empty op tuple"))?;
        let op = match tag.as_str() {
            "r" => Op::Replace(element(&mut seq, "r", "value")?),
            "s" => Op::Set(
                non_empty(element(&mut seq, "s", "path")?, "s")?,
                element(&mut seq, "s", "value")?,
            ),
            "d" => Op::Delete(non_empty(element(&mut seq, "d", "path")?, "d")?),
            "a" => {
                let path = non_empty(element(&mut seq, "a", "path")?, "a")?;
                let text: String = element(&mut seq, "a", "text")?;
                Op::Append(path, Str::new(text))
            }
            "t" => Op::TrimStart(
                non_empty(element(&mut seq, "t", "path")?, "t")?,
                element(&mut seq, "t", "count")?,
            ),
            "p" => Op::Splice {
                path: element(&mut seq, "p", "path")?,
                index: element(&mut seq, "p", "index")?,
                remove: element(&mut seq, "p", "remove count")?,
                items: element(&mut seq, "p", "items")?,
            },
            "m" => {
                let path = element(&mut seq, "m", "path")?;
                let permutation: Vec<usize> = element(&mut seq, "m", "permutation")?;
                if !is_permutation(&permutation) {
                    return Err(de::Error::custom("\"m\" op permutation is not a bijection"));
                }
                Op::Permute(path, permutation)
            }
            other => return Err(de::Error::custom(format_args!("unknown op {other:?}"))),
        };
        if seq.next_element::<IgnoredAny>()?.is_some() {
            return Err(de::Error::custom(format_args!("\"{tag}\" op has too many elements")));
        }
        Ok(op)
    }
}

pub(crate) fn is_permutation(permutation: &[usize]) -> bool {
    let mut seen = vec![false; permutation.len()];
    permutation
        .iter()
        .all(|&index| index < seen.len() && !std::mem::replace(&mut seen[index], true))
}

// ─── UTF-16 counts ───────────────────────────────────────────────────────────

/// UTF-16 code units of `text`: the unit of `t` counts.
pub(crate) fn utf16_len(text: &str) -> usize {
    if text.is_ascii() {
        text.len()
    } else {
        text.chars().map(char::len_utf16).sum()
    }
}

/// The byte offset after `units` UTF-16 code units of `text`, or `None` when the count is past
/// the end or ends inside a surrogate pair.
pub(crate) fn utf16_offset(text: &str, units: u64) -> Option<usize> {
    let units = usize::try_from(units).ok()?;
    if text.is_ascii() {
        return (units <= text.len()).then_some(units);
    }
    let mut counted = 0;
    for (offset, ch) in text.char_indices() {
        if counted == units {
            return Some(offset);
        }
        counted += ch.len_utf16();
        if counted > units {
            return None;
        }
    }
    (counted == units).then_some(text.len())
}

pub(crate) fn kind_error(expected: Kind, found: &Json) -> ApplyErrorKind {
    ApplyErrorKind::WrongKind {
        expected,
        found: found.kind(),
    }
}
