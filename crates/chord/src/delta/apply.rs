//! Applying op batches to a value.

use std::ops::Range;

use crate::json::{Json, Kind, Path, Seg};

use super::{Op, is_permutation, kind_error, utf16_offset};

/// An op that does not fit the value it was applied to.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("op {index} ({tag} at {path}): {kind}")]
pub struct ApplyError {
    /// Position of the op in its batch.
    pub index: usize,
    pub tag: &'static str,
    pub path: Path,
    pub kind: ApplyErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ApplyErrorKind {
    #[error("nothing at this path")]
    Missing,
    #[error("expected {expected}, found {found}")]
    WrongKind { expected: Kind, found: Kind },
    #[error("index {index} is out of range for length {len}")]
    OutOfRange { index: usize, len: usize },
    #[error("count {count} is past the end of a string of {len} UTF-16 code units or inside a surrogate pair")]
    BadTrim { count: u64, len: usize },
    #[error("cannot target the root")]
    Root,
    #[error("permutation of {permutation} elements does not fit an array of {len}")]
    BadPermutation { permutation: usize, len: usize },
}

/// Apply `ops` to `value` in order. Containers shared with other values are copied before they
/// change, so other holders of the original never see the change.
///
/// On error, a prefix of the batch may have been applied; use [`applied`] to keep the original.
pub fn apply(value: &mut Json, ops: &[Op]) -> Result<(), ApplyError> {
    for (index, op) in ops.iter().enumerate() {
        apply_op(value, op).map_err(|kind| ApplyError {
            index,
            tag: op.tag(),
            path: Path::from(op.path()),
            kind,
        })?;
    }
    Ok(())
}

/// `base` with `ops` applied. `base` is unchanged, also on error.
pub fn applied(base: &Json, ops: &[Op]) -> Result<Json, ApplyError> {
    let mut value = base.clone();
    apply(&mut value, ops)?;
    Ok(value)
}

fn apply_op(value: &mut Json, op: &Op) -> Result<(), ApplyErrorKind> {
    match op {
        Op::Replace(replacement) => {
            *value = replacement.clone();
            Ok(())
        }
        Op::Set(path, item) => {
            let (parent, last) = parent_mut(value, path)?;
            match (parent, last) {
                (Json::Object(object), Seg::Key(key)) => {
                    object.make_mut().insert(key.clone(), item.clone());
                    Ok(())
                }
                (Json::Array(array), Seg::Index(index)) => {
                    let len = array.len();
                    match (*index).cmp(&len) {
                        std::cmp::Ordering::Less => array.make_mut()[*index] = item.clone(),
                        std::cmp::Ordering::Equal => array.make_mut().push(item.clone()),
                        std::cmp::Ordering::Greater => return Err(ApplyErrorKind::OutOfRange { index: *index, len }),
                    }
                    Ok(())
                }
                (parent, last) => Err(seg_mismatch(parent, last)),
            }
        }
        Op::Delete(path) => {
            let (parent, last) = parent_mut(value, path)?;
            match (parent, last) {
                (Json::Object(object), Seg::Key(key)) => {
                    if !object.contains_key(&**key) {
                        return Err(ApplyErrorKind::Missing);
                    }
                    object.make_mut().shift_remove(&**key);
                    Ok(())
                }
                (Json::Array(array), Seg::Index(index)) => {
                    let len = array.len();
                    if *index >= len {
                        return Err(ApplyErrorKind::OutOfRange { index: *index, len });
                    }
                    array.make_mut().remove(*index);
                    Ok(())
                }
                (parent, last) => Err(seg_mismatch(parent, last)),
            }
        }
        Op::Append(path, text) => {
            if path.is_empty() {
                return Err(ApplyErrorKind::Root);
            }
            match target_mut(value, path)? {
                Json::String(string) => {
                    string.make_mut().push_str(text);
                    Ok(())
                }
                other => Err(kind_error(Kind::String, other)),
            }
        }
        Op::TrimStart(path, count) => {
            if path.is_empty() {
                return Err(ApplyErrorKind::Root);
            }
            match target_mut(value, path)? {
                Json::String(string) => {
                    let offset = utf16_offset(string, *count).ok_or_else(|| ApplyErrorKind::BadTrim {
                        count: *count,
                        len: super::utf16_len(string),
                    })?;
                    if offset > 0 {
                        string.make_mut().drain(..offset);
                    }
                    Ok(())
                }
                other => Err(kind_error(Kind::String, other)),
            }
        }
        Op::Splice {
            path,
            index,
            remove,
            items,
        } => match target_mut(value, path)? {
            Json::Array(array) => {
                let len = array.len();
                let range = splice_range(*index, *remove, len)?;
                array.make_mut().splice(range, items.iter().cloned());
                Ok(())
            }
            other => Err(kind_error(Kind::Array, other)),
        },
        Op::Permute(path, permutation) => match target_mut(value, path)? {
            Json::Array(array) => {
                if permutation.len() != array.len() || !is_permutation(permutation) {
                    return Err(ApplyErrorKind::BadPermutation {
                        permutation: permutation.len(),
                        len: array.len(),
                    });
                }
                let mut old: Vec<Option<Json>> = std::mem::take(array.make_mut()).into_iter().map(Some).collect();
                let reordered = permutation
                    .iter()
                    .map(|&from| old[from].take().expect("bijection"))
                    .collect();
                *array.make_mut() = reordered;
                Ok(())
            }
            other => Err(kind_error(Kind::Array, other)),
        },
    }
}

fn splice_range(index: usize, remove: usize, len: usize) -> Result<Range<usize>, ApplyErrorKind> {
    if index > len {
        return Err(ApplyErrorKind::OutOfRange { index, len });
    }
    let end = index.checked_add(remove).filter(|&end| end <= len);
    end.map(|end| index..end).ok_or(ApplyErrorKind::OutOfRange {
        index: index.saturating_add(remove),
        len,
    })
}

fn seg_mismatch(parent: &Json, last: &Seg) -> ApplyErrorKind {
    match last {
        Seg::Key(_) => kind_error(Kind::Object, parent),
        Seg::Index(_) => kind_error(Kind::Array, parent),
    }
}

fn target_mut<'a>(value: &'a mut Json, path: &[Seg]) -> Result<&'a mut Json, ApplyErrorKind> {
    let mut current = value;
    for seg in path {
        current = match (current, seg) {
            (Json::Object(object), Seg::Key(key)) => {
                if !object.contains_key(&**key) {
                    return Err(ApplyErrorKind::Missing);
                }
                object.make_mut().get_mut(&**key).expect("checked")
            }
            (Json::Array(array), Seg::Index(index)) => {
                let len = array.len();
                if *index >= len {
                    return Err(ApplyErrorKind::OutOfRange { index: *index, len });
                }
                &mut array.make_mut()[*index]
            }
            (parent, seg) => return Err(seg_mismatch(parent, seg)),
        };
    }
    Ok(current)
}

fn parent_mut<'a, 'p>(value: &'a mut Json, path: &'p [Seg]) -> Result<(&'a mut Json, &'p Seg), ApplyErrorKind> {
    let (last, parent) = path.split_last().ok_or(ApplyErrorKind::Root)?;
    Ok((target_mut(value, parent)?, last))
}
