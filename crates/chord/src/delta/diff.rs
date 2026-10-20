//! Ops between two values.

use crate::json::{Array, Json, Object, Path, Seg, Str};

use super::{Op, RESERVED_SEGMENTS, utf16_len};

/// How far back into the old string a sliding-window match is searched.
const OVERLAP_SCAN: usize = 64 * 1024;
/// Length of the first probe of an overlap search.
const OVERLAP_PROBE: usize = 64;
/// Matches of a probe that are checked before an overlap search gives up.
const OVERLAP_CANDIDATES: usize = 8;

/// Ops that turn `old` into `new`.
///
/// Subtrees that `new` shares with `old` are skipped in O(1), so diffing a value derived from
/// the previous revision costs time proportional to what changed. Appended text is an `a`; a
/// string whose start was dropped and whose end grew (a sliding window) is a `t` and an `a`;
/// an insert or removal between unchanged array elements is one `p`.
pub fn diff(old: &Json, new: &Json) -> Vec<Op> {
    let mut out = Vec::new();
    diff_into(&mut Vec::new(), old, new, &mut out);
    out
}

/// Append to `out` the ops that turn `old` into `new`, both located at `path`.
pub(crate) fn diff_into(path: &mut Vec<Seg>, old: &Json, new: &Json, out: &mut Vec<Op>) {
    if Json::same(old, new) {
        return;
    }
    match (old, new) {
        (Json::String(old), Json::String(new)) if !path.is_empty() => string_diff(path, old, new, out),
        (Json::Object(old), Json::Object(new)) => object_diff(path, old, new, out),
        (Json::Array(old), Json::Array(new)) => array_diff(path, old, new, out),
        _ if old == new => {}
        _ => out.push(set(path, new)),
    }
}

fn set(path: &[Seg], value: &Json) -> Op {
    if path.is_empty() {
        Op::Replace(value.clone())
    } else {
        Op::Set(Path::from(path), value.clone())
    }
}

fn string_diff(path: &[Seg], old: &Str, new: &Str, out: &mut Vec<Op>) {
    if old.as_str() == new.as_str() {
        return;
    }
    if let Some(rest) = new.strip_prefix(old.as_str()) {
        out.push(Op::Append(Path::from(path), Str::from(rest)));
        return;
    }
    let shared = overlap(old, new, OVERLAP_SCAN);
    if shared == 0 {
        out.push(Op::Set(Path::from(path), Json::String(new.clone())));
        return;
    }
    out.push(Op::TrimStart(
        Path::from(path),
        utf16_len(&old[..old.len() - shared]) as u64,
    ));
    if shared < new.len() {
        out.push(Op::Append(Path::from(path), Str::from(&new[shared..])));
    }
}

fn object_diff(path: &mut Vec<Seg>, old: &Object, new: &Object, out: &mut Vec<Op>) {
    // A changed reserved key would put a segment pi refuses into a path: set the whole object.
    let reserved_changed = RESERVED_SEGMENTS
        .iter()
        .any(|key| match (old.get(*key), new.get(*key)) {
            (None, None) => false,
            (Some(a), Some(b)) => a != b,
            _ => true,
        });
    if reserved_changed {
        out.push(set(path, &Json::Object(new.clone())));
        return;
    }
    for key in old.keys() {
        if !new.contains_key(key) {
            path.push(Seg::Key(key.clone()));
            out.push(Op::Delete(Path::from(&path[..])));
            path.pop();
        }
    }
    for (key, value) in new.iter() {
        path.push(Seg::Key(key.clone()));
        match old.get(key) {
            None => out.push(Op::Set(Path::from(&path[..]), value.clone())),
            Some(previous) => diff_into(path, previous, value, out),
        }
        path.pop();
    }
}

fn array_diff(path: &mut Vec<Seg>, old: &Array, new: &Array, out: &mut Vec<Op>) {
    let shorter = old.len().min(new.len());
    let mut prefix = 0;
    while prefix < shorter && old[prefix] == new[prefix] {
        prefix += 1;
    }
    let mut suffix = 0;
    while suffix < shorter - prefix && old[old.len() - 1 - suffix] == new[new.len() - 1 - suffix] {
        suffix += 1;
    }
    let old_middle = &old[prefix..old.len() - suffix];
    let new_middle = &new[prefix..new.len() - suffix];

    // Elements edited in place: pair them up. With equal lengths every element pairs; otherwise
    // leading containers of one kind do (an element edited, then elements inserted after it).
    let paired = if old_middle.len() == new_middle.len() {
        old_middle.len()
    } else {
        old_middle
            .iter()
            .zip(new_middle)
            .take_while(|(a, b)| same_container_kind(a, b))
            .count()
    };
    for (offset, (previous, value)) in old_middle.iter().zip(new_middle).take(paired).enumerate() {
        path.push(Seg::Index(prefix + offset));
        diff_into(path, previous, value, out);
        path.pop();
    }
    let remove = old_middle.len() - paired;
    let items = &new_middle[paired..];
    if remove > 0 || !items.is_empty() {
        out.push(Op::Splice {
            path: Path::from(&path[..]),
            index: prefix + paired,
            remove,
            items: items.to_vec(),
        });
    }
}

fn same_container_kind(a: &Json, b: &Json) -> bool {
    matches!(
        (a, b),
        (Json::Object(_), Json::Object(_)) | (Json::Array(_), Json::Array(_))
    )
}

/// The longest suffix of `a`, within its last `scan` bytes, that is a prefix of `b`, in bytes.
///
/// Probes with a long head of `b` first (few candidates, and it finds the large overlaps a
/// rolling window produces), then with its first character. Repetitive text can match a head
/// at many positions, so each probe checks a bounded number of candidates; giving up returns 0,
/// which costs a larger op, never a wrong one.
pub fn overlap(a: &str, b: &str, scan: usize) -> usize {
    if a.is_empty() || b.is_empty() || scan == 0 {
        return 0;
    }
    let tail = &a[ceil_char_boundary(a, a.len().saturating_sub(scan))..];
    let first_char = b.chars().next().map_or(0, char::len_utf8);
    let long_probe = floor_char_boundary(b, OVERLAP_PROBE.min(b.len()));
    let probes = if long_probe > first_char {
        [long_probe, first_char]
    } else {
        [first_char, 0]
    };
    for probe in probes.into_iter().filter(|&probe| probe > 0) {
        let head = &b[..probe];
        let mut from = 0;
        for _ in 0..OVERLAP_CANDIDATES {
            let Some(found) = tail[from..].find(head) else { break };
            let start = from + found;
            let shared = tail.len() - start;
            if shared <= b.len() && b.is_char_boundary(shared) && tail[start..] == b[..shared] {
                return shared;
            }
            from = start + tail[start..].chars().next().map_or(1, char::len_utf8);
        }
    }
    0
}

fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    while !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn ceil_char_boundary(text: &str, mut index: usize) -> usize {
    while !text.is_char_boundary(index) {
        index += 1;
    }
    index
}
