//! Draft handles and the op log of a change.
//!
//! Structural edits record their op as they happen; assignments record the diff against the
//! value they replace. A handle addresses its value by path from the root of the change, so
//! recording can look at any ancestor (reserved-segment folding needs that).

use std::cmp::Ordering;
use std::ops::{Bound, RangeBounds};

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::json::{self, Json, Key, Map, Path, Seg, Str};

use super::diff::diff_into;
use super::{MAX_OPS, Op, is_reserved, utf16_len};

const VALID: &str = "a draft's path stays valid while the draft exists";

/// The working value of a change and the ops recorded so far.
pub(crate) struct Edit {
    pub(crate) working: Json,
    log: Vec<Op>,
    /// More than `MAX_OPS` ops were recorded: the batch will be one replacement.
    folded: bool,
}

impl Edit {
    pub(crate) fn new(working: Json) -> Self {
        Self {
            working,
            log: Vec::new(),
            folded: false,
        }
    }

    /// The working value and its ops; `None` when the log folded into a replacement.
    pub(crate) fn finish(self) -> (Json, Option<Vec<Op>>) {
        (self.working, (!self.folded).then_some(self.log))
    }

    /// Record an op whose edit was already made to the working value.
    fn record(&mut self, op: Op) {
        if self.folded {
            return;
        }
        let path = op.path();
        let op = if let Some(at) = path.iter().position(is_reserved) {
            // pi refuses reserved segments: set the ancestor above the first one, as it is now.
            let ancestor = &path[..at];
            let value = self.working.pointer(ancestor).expect(VALID).clone();
            if ancestor.is_empty() {
                Op::Replace(value)
            } else {
                Op::Set(Path::from(ancestor), value)
            }
        } else if path.is_empty() && matches!(op, Op::Set(..) | Op::Delete(_) | Op::Append(..) | Op::TrimStart(..)) {
            // A string or value edit at the root has no tuple form: replace the root.
            Op::Replace(self.working.clone())
        } else {
            op
        };
        self.push(op);
        if self.log.len() > MAX_OPS {
            self.folded = true;
            self.log = Vec::new();
        }
    }

    /// Record the ops that turn `old` into `new` at `path`.
    fn record_diff(&mut self, path: &Path, old: &Json, new: &Json) {
        let mut ops = Vec::new();
        diff_into(&mut path.to_vec(), old, new, &mut ops);
        for op in ops {
            self.record(op);
        }
    }

    /// Append to the log, merging with the previous op where the result is the same.
    fn push(&mut self, op: Op) {
        match &op {
            Op::Replace(_) => self.log.clear(),
            Op::Append(path, text) => {
                if let Some(Op::Append(last_path, last_text)) = self.log.last_mut()
                    && last_path == path
                {
                    last_text.make_mut().push_str(text);
                    return;
                }
            }
            Op::Splice {
                path,
                index,
                remove: 0,
                items,
            } => {
                if let Some(Op::Splice {
                    path: last_path,
                    index: last_index,
                    remove: 0,
                    items: last_items,
                }) = self.log.last_mut()
                    && last_path == path
                    && *last_index + last_items.len() == *index
                {
                    last_items.extend(items.iter().cloned());
                    return;
                }
            }
            Op::Set(path, _) | Op::Delete(path) => self.drop_superseded(path, matches!(op, Op::Delete(_))),
            _ => {}
        }
        self.log.push(op);
    }

    /// Drop the adjacent ops that an assignment or deletion at `path` makes irrelevant: those
    /// below it, and those at it except a deletion (whose effect on key order an assignment
    /// does not repeat) and, before a deletion, a set (which may have created the key). Only
    /// the adjacent run: earlier ops may address elements whose indices moved since.
    fn drop_superseded(&mut self, path: &[Seg], deleting: bool) {
        while let Some(last) = self.log.last() {
            let last_path = last.path();
            let below = last_path.len() > path.len() && last_path.starts_with(path);
            let at = last_path == path && !matches!(last, Op::Delete(_)) && !(deleting && matches!(last, Op::Set(..)));
            if !(below || at) {
                break;
            }
            self.log.pop();
        }
    }
}

// ─── Draft ───────────────────────────────────────────────────────────────────

/// A handle on one value of a change: read it, assign it, or narrow to its kind.
///
/// It borrows the change mutably, so one handle is usable at a time and none outlives the
/// change.
#[derive(Debug)]
pub struct Draft<'a> {
    edit: &'a mut Edit,
    path: Path,
}

impl std::fmt::Debug for Edit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Edit")
            .field("ops", &self.log.len())
            .field("folded", &self.folded)
            .finish_non_exhaustive()
    }
}

impl<'a> Draft<'a> {
    pub(crate) fn root(edit: &'a mut Edit) -> Self {
        Self {
            edit,
            path: Path::root(),
        }
    }

    /// Where this value is in the change's value.
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn value(&self) -> &Json {
        self.edit.working.pointer(&self.path).expect(VALID)
    }

    fn value_mut(&mut self) -> &mut Json {
        self.edit.working.pointer_mut(&self.path).expect(VALID)
    }

    /// Assign the value. Records the diff against the value it replaces.
    pub fn set(&mut self, value: impl Into<Json>) {
        let value = value.into();
        let old = std::mem::replace(self.value_mut(), value.clone());
        self.edit.record_diff(&self.path, &old, &value);
    }

    /// Change the value freely, then record the diff against what it was.
    pub fn update<R>(&mut self, edit: impl FnOnce(&mut Json) -> R) -> R {
        let old = self.value().clone();
        let result = edit(self.value_mut());
        let new = self.value().clone();
        self.edit.record_diff(&self.path, &old, &new);
        result
    }

    /// Change the value as a `T`: deserialize, run `edit`, serialize back and record the diff.
    /// Costs a serde round trip of this value.
    pub fn update_as<T, R>(&mut self, edit: impl FnOnce(&mut T) -> R) -> Result<R, serde_json::Error>
    where
        T: Serialize + DeserializeOwned,
    {
        let mut typed: T = json::from_json(self.value())?;
        let result = edit(&mut typed);
        self.set(json::to_json(&typed)?);
        Ok(result)
    }

    /// The value at `key`, when this is an object that has it.
    pub fn field(&mut self, key: &str) -> Option<Draft<'_>> {
        self.value().get(key)?;
        Some(Draft {
            edit: &mut *self.edit,
            path: self.path.join(key),
        })
    }

    /// The element at `index`, when this is an array that long.
    pub fn index(&mut self, index: usize) -> Option<Draft<'_>> {
        self.value().get_index(index)?;
        Some(Draft {
            edit: &mut *self.edit,
            path: self.path.join(index),
        })
    }

    /// The value at `path` below this one.
    pub fn at(&mut self, path: &[Seg]) -> Option<Draft<'_>> {
        self.value().pointer(path)?;
        Some(Draft {
            edit: &mut *self.edit,
            path: self.path.concat(path),
        })
    }

    pub fn object(&mut self) -> Option<ObjectDraft<'_>> {
        self.value().as_object()?;
        Some(ObjectDraft {
            edit: &mut *self.edit,
            path: self.path.clone(),
        })
    }

    pub fn array(&mut self) -> Option<ArrayDraft<'_>> {
        self.value().as_array()?;
        Some(ArrayDraft {
            edit: &mut *self.edit,
            path: self.path.clone(),
        })
    }

    pub fn string(&mut self) -> Option<StrDraft<'_>> {
        self.value().as_str()?;
        Some(StrDraft {
            edit: &mut *self.edit,
            path: self.path.clone(),
        })
    }

    pub fn into_object(self) -> Option<ObjectDraft<'a>> {
        self.value().as_object()?;
        Some(ObjectDraft {
            edit: self.edit,
            path: self.path,
        })
    }

    pub fn into_array(self) -> Option<ArrayDraft<'a>> {
        self.value().as_array()?;
        Some(ArrayDraft {
            edit: self.edit,
            path: self.path,
        })
    }

    pub fn into_string(self) -> Option<StrDraft<'a>> {
        self.value().as_str()?;
        Some(StrDraft {
            edit: self.edit,
            path: self.path,
        })
    }
}

// ─── Objects ─────────────────────────────────────────────────────────────────

/// A handle on an object of a change.
#[derive(Debug)]
pub struct ObjectDraft<'a> {
    edit: &'a mut Edit,
    path: Path,
}

impl ObjectDraft<'_> {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn value(&self) -> &Map {
        match self.edit.working.pointer(&self.path) {
            Some(Json::Object(object)) => object,
            _ => panic!("{VALID}"),
        }
    }

    fn map_mut(&mut self) -> &mut Map {
        match self.edit.working.pointer_mut(&self.path) {
            Some(Json::Object(object)) => object.make_mut(),
            _ => panic!("{VALID}"),
        }
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        self.value().get(key)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.value().contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.value().len()
    }

    pub fn is_empty(&self) -> bool {
        self.value().is_empty()
    }

    pub fn keys(&self) -> impl Iterator<Item = &Key> {
        self.value().keys()
    }

    /// The value at `key`, if present.
    pub fn field(&mut self, key: &str) -> Option<Draft<'_>> {
        self.value().get(key)?;
        Some(Draft {
            edit: &mut *self.edit,
            path: self.path.join(key),
        })
    }

    /// The value at `key`, inserting `default` first when absent.
    pub fn field_or_insert(&mut self, key: &str, default: impl Into<Json>) -> Draft<'_> {
        if !self.contains_key(key) {
            self.insert(key, default);
        }
        Draft {
            edit: &mut *self.edit,
            path: self.path.join(key),
        }
    }

    /// Set `key`. A new key records a set; replacing a value records the diff against it.
    /// Returns the replaced value.
    pub fn insert(&mut self, key: impl Into<Key>, value: impl Into<Json>) -> Option<Json> {
        let key = key.into();
        let value = value.into();
        let path = self.path.join(key.clone());
        let old = self.map_mut().insert(key, value.clone());
        match &old {
            Some(old) => self.edit.record_diff(&path, old, &value),
            None => self.edit.record(Op::Set(path, value)),
        }
        old
    }

    /// Remove `key`, keeping the order of the other keys.
    pub fn remove(&mut self, key: &str) -> Option<Json> {
        if !self.contains_key(key) {
            return None;
        }
        let removed = self.map_mut().shift_remove(key);
        self.edit.record(Op::Delete(self.path.join(key)));
        removed
    }
}

// ─── Arrays ──────────────────────────────────────────────────────────────────

/// A handle on an array of a change. Index methods panic out of bounds, as `Vec`'s do.
#[derive(Debug)]
pub struct ArrayDraft<'a> {
    edit: &'a mut Edit,
    path: Path,
}

impl ArrayDraft<'_> {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn value(&self) -> &[Json] {
        match self.edit.working.pointer(&self.path) {
            Some(Json::Array(array)) => array,
            _ => panic!("{VALID}"),
        }
    }

    fn vec_mut(&mut self) -> &mut Vec<Json> {
        match self.edit.working.pointer_mut(&self.path) {
            Some(Json::Array(array)) => array.make_mut(),
            _ => panic!("{VALID}"),
        }
    }

    pub fn get(&self, index: usize) -> Option<&Json> {
        self.value().get(index)
    }

    pub fn len(&self) -> usize {
        self.value().len()
    }

    pub fn is_empty(&self) -> bool {
        self.value().is_empty()
    }

    /// The element at `index`, if in bounds.
    pub fn index(&mut self, index: usize) -> Option<Draft<'_>> {
        self.value().get(index)?;
        Some(Draft {
            edit: &mut *self.edit,
            path: self.path.join(index),
        })
    }

    /// Assign the element at `index`, recording the diff against the old element.
    pub fn set(&mut self, index: usize, value: impl Into<Json>) -> Json {
        let len = self.len();
        assert!(index < len, "set index (is {index}) should be < len (is {len})");
        let value = value.into();
        let old = std::mem::replace(&mut self.vec_mut()[index], value.clone());
        self.edit.record_diff(&self.path.join(index), &old, &value);
        old
    }

    pub fn push(&mut self, value: impl Into<Json>) {
        let index = self.len();
        self.insert(index, value);
    }

    pub fn extend(&mut self, values: impl IntoIterator<Item = Json>) {
        let index = self.len();
        self.splice(index..index, values);
    }

    pub fn insert(&mut self, index: usize, value: impl Into<Json>) {
        let len = self.len();
        assert!(index <= len, "insertion index (is {index}) should be <= len (is {len})");
        let value = value.into();
        self.vec_mut().insert(index, value.clone());
        self.edit.record(Op::Splice {
            path: self.path.clone(),
            index,
            remove: 0,
            items: vec![value],
        });
    }

    pub fn remove(&mut self, index: usize) -> Json {
        let len = self.len();
        assert!(index < len, "removal index (is {index}) should be < len (is {len})");
        let removed = self.vec_mut().remove(index);
        self.edit.record(Op::Splice {
            path: self.path.clone(),
            index,
            remove: 1,
            items: Vec::new(),
        });
        removed
    }

    pub fn pop(&mut self) -> Option<Json> {
        let len = self.len();
        (len > 0).then(|| self.remove(len - 1))
    }

    /// Replace `range` with `items`, as `Vec::splice`; returns the removed elements.
    pub fn splice(&mut self, range: impl RangeBounds<usize>, items: impl IntoIterator<Item = Json>) -> Vec<Json> {
        let len = self.len();
        let start = match range.start_bound() {
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&end) => end + 1,
            Bound::Excluded(&end) => end,
            Bound::Unbounded => len,
        };
        assert!(
            start <= end && end <= len,
            "splice range {start}..{end} out of bounds for length {len}"
        );
        let items: Vec<Json> = items.into_iter().collect();
        let removed: Vec<Json> = self.vec_mut().splice(start..end, items.iter().cloned()).collect();
        if !removed.is_empty() || !items.is_empty() {
            self.edit.record(Op::Splice {
                path: self.path.clone(),
                index: start,
                remove: end - start,
                items,
            });
        }
        removed
    }

    pub fn truncate(&mut self, len: usize) {
        if len < self.len() {
            self.splice(len.., []);
        }
    }

    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Keep the elements for which `keep` returns true. Records one splice per removed run,
    /// from the end.
    pub fn retain(&mut self, mut keep: impl FnMut(&Json) -> bool) {
        let flags: Vec<bool> = self.value().iter().map(&mut keep).collect();
        let mut end = flags.len();
        while end > 0 {
            if flags[end - 1] {
                end -= 1;
                continue;
            }
            let mut start = end - 1;
            while start > 0 && !flags[start - 1] {
                start -= 1;
            }
            self.splice(start..end, []);
            end = start;
        }
    }

    pub fn reverse(&mut self) {
        let len = self.len();
        if len > 1 {
            self.permute((0..len).rev().collect());
        }
    }

    /// Sort stably with `compare`. Records one `m` with the permutation.
    pub fn sort_by(&mut self, mut compare: impl FnMut(&Json, &Json) -> Ordering) {
        let values = self.value();
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&a, &b| compare(&values[a], &values[b]));
        if order.iter().enumerate().any(|(position, &from)| position != from) {
            self.permute(order);
        }
    }

    /// Reorder: `new[i] = old[order[i]]`.
    fn permute(&mut self, order: Vec<usize>) {
        let reordered = order.iter().map(|&from| self.value()[from].clone()).collect();
        *self.vec_mut() = reordered;
        self.edit.record(Op::Permute(self.path.clone(), order));
    }
}

// ─── Strings ─────────────────────────────────────────────────────────────────

/// A handle on a string of a change.
#[derive(Debug)]
pub struct StrDraft<'a> {
    edit: &'a mut Edit,
    path: Path,
}

impl StrDraft<'_> {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn as_str(&self) -> &str {
        match self.edit.working.pointer(&self.path) {
            Some(Json::String(text)) => text,
            _ => panic!("{VALID}"),
        }
    }

    fn string_mut(&mut self) -> &mut String {
        match self.edit.working.pointer_mut(&self.path) {
            Some(Json::String(text)) => text.make_mut(),
            _ => panic!("{VALID}"),
        }
    }

    pub fn len(&self) -> usize {
        self.as_str().len()
    }

    pub fn is_empty(&self) -> bool {
        self.as_str().is_empty()
    }

    /// Append `text`. Records an `a`; appends in a row merge into one.
    pub fn push_str(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.string_mut().push_str(text);
        self.edit.record(Op::Append(self.path.clone(), Str::from(text)));
    }

    /// Remove the first `len` bytes. Records a `t`.
    ///
    /// # Panics
    ///
    /// When `len` is past the end or not on a character boundary.
    pub fn remove_prefix(&mut self, len: usize) {
        let text = self.as_str();
        assert!(
            text.is_char_boundary(len),
            "remove_prefix length {len} is not a character boundary"
        );
        if len == 0 {
            return;
        }
        let units = utf16_len(&text[..len]) as u64;
        self.string_mut().drain(..len);
        self.edit.record(Op::TrimStart(self.path.clone(), units));
    }

    /// Assign the string. Records an append, a trim plus an append (a sliding window), or a set.
    pub fn set(&mut self, text: impl Into<String>) {
        let value = Json::from(text.into());
        let old = std::mem::replace(self.edit.working.pointer_mut(&self.path).expect(VALID), value.clone());
        self.edit.record_diff(&self.path, &old, &value);
    }
}
