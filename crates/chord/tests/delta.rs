//! Tests of `delta_chord::delta` and `delta_chord::json`: the tuple form, `apply`, the tracker
//! lifecycle, what each draft edit records, `diff`, and randomized convergence.

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

use delta_chord::delta::{AdoptError, ApplyErrorKind, MAX_OPS, Op, Tracker, applied, apply, diff, overlap};
use delta_chord::json::{Json, Path};
use delta_chord::{json, path};
use pretty_assertions::assert_eq;

fn ops_json(ops: &[Op]) -> serde_json::Value {
    serde_json::to_value(ops).unwrap()
}

/// Run `edit` on a change of `initial`; check the batch replays to the value, survives a JSON
/// round trip, and return it.
fn change(initial: Json, edit: impl FnOnce(delta_chord::delta::Draft<'_>)) -> (Json, Vec<Op>) {
    let mut tracker = Tracker::new(initial);
    let mut change = tracker.begin();
    edit(change.draft());
    let prepared = change.prepare();
    let value = prepared.value().clone();
    assert_eq!(
        applied(prepared.base(), prepared.ops()).unwrap(),
        value,
        "replay of {:?}",
        prepared.ops()
    );
    let wire: Vec<Op> = serde_json::from_str(&serde_json::to_string(prepared.ops()).unwrap()).unwrap();
    assert_eq!(wire, prepared.ops());
    let ops = tracker.adopt(prepared).unwrap();
    assert_eq!(tracker.value(), &value);
    (value, ops)
}

// ─── Wire form ───────────────────────────────────────────────────────────────

#[test]
fn ops_use_upstream_tuples() {
    let text = r#"[
        ["r", {"a": 1}],
        ["s", ["a", 0, "b"], null],
        ["d", ["tasks", "123"]],
        ["a", ["out"], "x"],
        ["t", ["out"], 3],
        ["p", [], 0, 1, [true]],
        ["m", ["list"], [1, 0]]
    ]"#;
    let ops: Vec<Op> = serde_json::from_str(text).unwrap();
    assert_eq!(ops[1], Op::Set(path!["a", 0, "b"], Json::Null));
    assert_eq!(ops[2], Op::Delete(path!["tasks", "123"]));
    assert_eq!(ops_json(&ops), serde_json::from_str::<serde_json::Value>(text).unwrap());
    assert!(Op::is_base(&ops));
}

#[test]
fn malformed_ops_are_refused() {
    for text in [
        r#"["s", [], 1]"#,
        r#"["d", []]"#,
        r#"["a", [], "x"]"#,
        r#"["t", ["x"], -1]"#,
        r#"["m", [], [0, 0]]"#,
        r#"["s", ["x"]]"#,
        r#"["d", ["x"], 1]"#,
        r#"["x", ["x"]]"#,
        r#"["s", [-1], 1]"#,
        "[]",
    ] {
        assert!(serde_json::from_str::<Op>(text).is_err(), "{text} should be refused");
    }
}

// ─── apply ───────────────────────────────────────────────────────────────────

#[test]
fn apply_leaves_shared_values_untouched() {
    let base = json!({"list": [1, 2, 3], "text": "ab", "keep": {"x": 1}});
    let ops: Vec<Op> = serde_json::from_value(serde_json::json!([
        ["p", ["list"], 1, 1, [9, 8]],
        ["a", ["text"], "c"],
        ["t", ["text"], 1],
        ["s", ["new"], {"y": 2}],
        ["d", ["keep"]],
        ["m", ["list"], [3, 2, 1, 0]],
        ["s", ["list", 4], 0]
    ]))
    .unwrap();
    let value = applied(&base, &ops).unwrap();
    assert_eq!(value, json!({"list": [3, 8, 9, 1, 0], "text": "bc", "new": {"y": 2}}));
    assert_eq!(base, json!({"list": [1, 2, 3], "text": "ab", "keep": {"x": 1}}));
}

#[test]
fn trim_counts_utf16_code_units() {
    let mut value = json!({"t": "😀é-x"});
    apply(&mut value, &[Op::TrimStart(path!["t"], 3)]).unwrap();
    assert_eq!(value, json!({"t": "-x"}));
    let error = applied(&json!({"t": "😀"}), &[Op::TrimStart(path!["t"], 1)]).unwrap_err();
    assert!(matches!(error.kind, ApplyErrorKind::BadTrim { count: 1, len: 2 }));
}

#[test]
fn apply_is_strict() {
    let base = json!({"a": [1], "s": "x"});
    let cases = [
        (r#"["d", ["missing"]]"#, "Missing"),
        (r#"["s", ["a", 2], 0]"#, "OutOfRange"),
        (r#"["p", ["a"], 0, 2, []]"#, "OutOfRange"),
        (r#"["t", ["s"], 2]"#, "BadTrim"),
        (r#"["a", ["a"], "x"]"#, "WrongKind"),
        (r#"["s", ["a", "k"], 0]"#, "WrongKind"),
        (r#"["m", ["a"], [1, 0]]"#, "BadPermutation"),
    ];
    for (text, expected) in cases {
        let op: Op = serde_json::from_str(text).unwrap();
        let error = applied(&base, &[op]).unwrap_err();
        assert!(format!("{:?}", error.kind).starts_with(expected), "{text}: {error}");
    }
}

#[test]
fn reserved_segments_apply() {
    let value = applied(&json!({}), &[Op::Set(path!["__proto__"], json!(1))]).unwrap();
    assert_eq!(value, json!({"__proto__": 1}));
}

// ─── Tracker lifecycle ───────────────────────────────────────────────────────

#[test]
fn adopting_one_change_stales_the_others() {
    let mut tracker = Tracker::new(json!({"n": 0}));
    let mut first = tracker.begin();
    let mut second = tracker.begin();
    first.draft().field("n").unwrap().set(1);
    second.draft().field("n").unwrap().set(2);
    let first = first.prepare();
    let second = second.prepare();
    assert_eq!(tracker.adopt(first).unwrap(), vec![Op::Set(path!["n"], json!(1))]);
    assert_eq!(
        tracker.adopt(second).unwrap_err(),
        AdoptError::Stale {
            prepared: 0,
            current: 1
        }
    );
    assert_eq!(tracker.value(), &json!({"n": 1}));

    let other = Tracker::new(json!(null));
    assert_eq!(tracker.adopt(other.begin().prepare()).unwrap_err(), AdoptError::Foreign);
}

#[test]
fn no_op_changes_keep_the_base() {
    let mut tracker = Tracker::new(json!({"list": [1], "text": "a", "obj": {"k": [1, 2]}}));
    let mut change = tracker.begin();
    {
        let mut draft = change.draft();
        let mut list = draft.field("list").unwrap().into_array().unwrap();
        list.push(2);
        list.pop();
        draft.field("text").unwrap().set("a");
        draft.field("obj").unwrap().set(json!({"k": [1, 2]}));
    }
    let prepared = change.prepare();
    assert!(prepared.is_empty());
    assert!(Json::same(prepared.value(), prepared.base()));
    tracker.adopt(prepared).unwrap();
    assert_eq!(tracker.revision(), 1);
}

#[test]
fn prepare_replace_and_diff() {
    let tracker = Tracker::new(json!({"a": 1}));
    assert!(tracker.prepare_replace(json!({"a": 1})).is_empty());
    assert_eq!(
        ops_json(tracker.prepare_replace(json!({"a": 2})).ops()),
        serde_json::json!([["r", {"a": 2}]])
    );
    let mut next = tracker.value().clone();
    *next.pointer_mut(&path!["a"]).unwrap() = json!(3);
    assert_eq!(
        ops_json(tracker.prepare_diff(next).ops()),
        serde_json::json!([["s", ["a"], 3]])
    );
}

// ─── Drafts ──────────────────────────────────────────────────────────────────

#[test]
fn appends_and_pushes_merge() {
    let (_, ops) = change(json!({"out": "", "items": []}), |mut draft| {
        let mut out = draft.field("out").unwrap().into_string().unwrap();
        out.push_str("a");
        out.push_str("b");
        out.push_str("");
        let mut items = draft.field("items").unwrap().into_array().unwrap();
        items.push(1);
        items.push(2);
        items.extend([json!(3)]);
    });
    assert_eq!(
        ops_json(&ops),
        serde_json::json!([["a", ["out"], "ab"], ["p", ["items"], 0, 0, [1, 2, 3]]])
    );
}

#[test]
fn assigning_a_string_diffs_it() {
    let (_, ops) = change(json!({"out": "one\n"}), |mut draft| {
        draft.field("out").unwrap().set("one\ntwo\n");
    });
    assert_eq!(ops_json(&ops), serde_json::json!([["a", ["out"], "two\n"]]));

    // A sliding window: the start was dropped and the end grew.
    let (_, ops) = change(json!({"out": "line 1\nline 2\nline 3\n"}), |mut draft| {
        draft.field("out").unwrap().set("line 2\nline 3\nline 4\n");
    });
    assert_eq!(
        ops_json(&ops),
        serde_json::json!([["t", ["out"], 7], ["a", ["out"], "line 4\n"]])
    );

    let (_, ops) = change(json!({"out": "abc"}), |mut draft| {
        draft.field("out").unwrap().set("xyz");
    });
    assert_eq!(ops_json(&ops), serde_json::json!([["s", ["out"], "xyz"]]));
}

#[test]
fn object_edits() {
    let (value, ops) = change(json!({"a": 1, "b": {"c": 2}}), |draft| {
        let mut object = draft.into_object().unwrap();
        assert_eq!(object.insert("a", 5), Some(json!(1)));
        object.insert("z", json!([]));
        object.remove("b");
        assert_eq!(object.remove("missing"), None);
        object.field_or_insert("n", 0).set(1);
    });
    assert_eq!(value, json!({"a": 5, "z": [], "n": 1}));
    // The insert of "n" is superseded by its assignment.
    assert_eq!(
        ops_json(&ops),
        serde_json::json!([["s", ["a"], 5], ["s", ["z"], []], ["d", ["b"]], ["s", ["n"], 1]])
    );
}

#[test]
fn an_assignment_drops_the_edits_below_it() {
    let (_, ops) = change(json!({"doc": {"text": "", "list": []}}), |mut draft| {
        let mut doc = draft.field("doc").unwrap();
        doc.field("text").unwrap().into_string().unwrap().push_str("x");
        doc.field("list").unwrap().into_array().unwrap().push(1);
        doc.set("replaced");
    });
    assert_eq!(ops_json(&ops), serde_json::json!([["s", ["doc"], "replaced"]]));

    let (_, ops) = change(json!({"text": "a", "made": 0}), |draft| {
        let mut object = draft.into_object().unwrap();
        object.field("text").unwrap().into_string().unwrap().push_str("b");
        object.remove("text");
        object.insert("new", 1);
        object.remove("new");
    });
    assert_eq!(
        ops_json(&ops),
        serde_json::json!([["d", ["text"]], ["s", ["new"], 1], ["d", ["new"]]])
    );
}

#[test]
fn array_edits() {
    let (value, ops) = change(json!([5, 3, 4, 1, 2]), |draft| {
        let mut array = draft.into_array().unwrap();
        array.sort_by(|a, b| a.as_i64().cmp(&b.as_i64()));
        array.retain(|item| item.as_i64() != Some(2) && item.as_i64() != Some(3));
        array.insert(0, 0);
        array.reverse();
        array.set(0, 50);
        assert_eq!(array.splice(1..2, [json!("x"), json!("y")]), vec![json!(4)]);
        array.truncate(3);
    });
    assert_eq!(value, json!([50, "x", "y"]));
    assert_eq!(
        ops_json(&ops),
        serde_json::json!([
            ["m", [], [3, 4, 1, 2, 0]],
            ["p", [], 1, 2, []],
            ["p", [], 0, 0, [0]],
            ["m", [], [3, 2, 1, 0]],
            ["s", [0], 50],
            ["p", [], 1, 1, ["x", "y"]],
            ["p", [], 3, 2, []]
        ])
    );
}

#[test]
fn root_string_edits_replace_the_root() {
    let (value, ops) = change(json!("a"), |draft| draft.into_string().unwrap().push_str("b"));
    assert_eq!(value, json!("ab"));
    assert_eq!(ops_json(&ops), serde_json::json!([["r", "ab"]]));
}

#[test]
fn reserved_segments_fold_into_the_ancestor() {
    let (_, ops) = change(json!({"names": {"constructor": "a"}}), |mut draft| {
        let mut names = draft.field("names").unwrap();
        names.field("constructor").unwrap().into_string().unwrap().push_str("b");
    });
    assert_eq!(
        ops_json(&ops),
        serde_json::json!([["s", ["names"], {"constructor": "ab"}]])
    );

    let (_, ops) = change(json!({"names": {"constructor": 1, "x": 1}}), |mut draft| {
        draft.field("names").unwrap().set(json!({"constructor": 2, "x": 1}));
    });
    assert_eq!(
        ops_json(&ops),
        serde_json::json!([["s", ["names"], {"constructor": 2, "x": 1}]])
    );

    let (_, ops) = change(json!({"prototype": 1}), |draft| {
        draft.into_object().unwrap().remove("prototype");
    });
    assert_eq!(ops_json(&ops), serde_json::json!([["r", {}]]));
}

#[test]
fn typed_updates() {
    #[derive(serde::Serialize, serde::Deserialize)]
    struct Status {
        running: bool,
        tasks: u32,
        log: String,
    }
    let (_, ops) = change(
        json!({"status": {"running": false, "tasks": 1, "log": "a"}}),
        |mut draft| {
            draft
                .field("status")
                .unwrap()
                .update_as(|status: &mut Status| {
                    status.running = true;
                    status.log.push('b');
                    status.tasks
                })
                .unwrap();
        },
    );
    assert_eq!(
        ops_json(&ops),
        serde_json::json!([["s", ["status", "running"], true], ["a", ["status", "log"], "b"]])
    );
}

#[test]
fn long_changes_fold_into_a_replacement() {
    let (value, ops) = change(json!({"n": []}), |mut draft| {
        let mut list = draft.field("n").unwrap().into_array().unwrap();
        for index in 0..=MAX_OPS {
            list.push(index);
            list.set(index, -1);
        }
    });
    assert_eq!(ops.len(), 1);
    assert!(Op::is_base(&ops));
    assert_eq!(value.get("n").unwrap().as_array().unwrap().len(), MAX_OPS + 1);
}

// ─── diff ────────────────────────────────────────────────────────────────────

#[test]
fn array_diffs_are_one_splice_around_unchanged_elements() {
    let old = json!([{"id": 1}, {"id": 2}, {"id": 3}]);
    let new = json!([{"id": 1}, {"id": 9}, {"id": 2}, {"id": 3}]);
    assert_eq!(
        ops_json(&diff(&old, &new)),
        serde_json::json!([["p", [], 1, 0, [{"id": 9}]]])
    );

    // The last element edited in place, then one appended.
    let old = json!([{"text": "a"}]);
    let new = json!([{"text": "ab"}, {"text": "c"}]);
    assert_eq!(
        ops_json(&diff(&old, &new)),
        serde_json::json!([["a", [0, "text"], "b"], ["p", [], 1, 0, [{"text": "c"}]]])
    );
}

#[test]
fn diff_skips_shared_subtrees_and_ignores_key_order() {
    let old = json!({"big": [1, 2, 3], "x": 1, "y": 2});
    let mut new = old.clone();
    *new.pointer_mut(&path!["x"]).unwrap() = json!(10);
    assert_eq!(ops_json(&diff(&old, &new)), serde_json::json!([["s", ["x"], 10]]));
    assert!(diff(&json!({"a": 1, "b": 2}), &json!({"b": 2, "a": 1})).is_empty());
    assert!(diff(&json!(1), &json!(1.0)).is_empty());
    assert_eq!(ops_json(&diff(&json!(1), &json!("1"))), serde_json::json!([["r", "1"]]));
}

#[test]
fn overlap_finds_sliding_windows() {
    assert_eq!(overlap("abcdef", "defgh", 100), 3);
    assert_eq!(overlap("abc", "xyz", 100), 0);
    assert_eq!(overlap("aaaa", "aab", 100), 2);
    assert_eq!(overlap("xé", "éy", 100), "é".len());
    assert_eq!(overlap("abcdef", "defgh", 2), 0);
}

// ─── Convergence ─────────────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn scalar(&mut self) -> Json {
        match self.below(5) {
            0 => Json::Null,
            1 => Json::from(self.below(2) == 0),
            2 => Json::from(self.below(100)),
            3 => Json::from(["a", "é", "😀", "\n", "constructor"][self.below(5)]),
            _ => json!({"k": self.below(3)}),
        }
    }
}

/// Random edits through drafts: every batch must replay to the value it was prepared with.
#[test]
fn random_edits_converge() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut tracker = Tracker::new(json!({"list": [], "text": "", "map": {}}));
    let mut replica = tracker.value().clone();
    for _ in 0..2000 {
        let mut change = tracker.begin();
        let steps = 1 + rng.below(6);
        for _ in 0..steps {
            let mut draft = change.draft();
            match rng.below(9) {
                0 => draft.field("list").unwrap().into_array().unwrap().push(rng.scalar()),
                1 => {
                    let mut list = draft.field("list").unwrap().into_array().unwrap();
                    if !list.is_empty() {
                        let index = rng.below(list.len());
                        list.remove(index);
                    }
                }
                2 => {
                    let mut list = draft.field("list").unwrap().into_array().unwrap();
                    let len = list.len();
                    let start = rng.below(len + 1);
                    let end = start + rng.below(len - start + 1);
                    let items: Vec<Json> = (0..rng.below(3)).map(|_| rng.scalar()).collect();
                    list.splice(start..end, items);
                }
                3 => draft
                    .field("list")
                    .unwrap()
                    .into_array()
                    .unwrap()
                    .sort_by(|a, b| a.to_json_string().cmp(&b.to_json_string())),
                4 => {
                    let piece = ["x", "é", "😀\n"][rng.below(3)];
                    draft.field("text").unwrap().into_string().unwrap().push_str(piece);
                }
                5 => {
                    let mut text = draft.field("text").unwrap().into_string().unwrap();
                    let cut = text.as_str().char_indices().map(|(i, _)| i).nth(1).unwrap_or(0);
                    text.remove_prefix(cut);
                }
                6 => {
                    let key = ["a", "b", "constructor"][rng.below(3)];
                    let value = rng.scalar();
                    draft.field("map").unwrap().into_object().unwrap().insert(key, value);
                }
                7 => {
                    let key = ["a", "b", "constructor"][rng.below(3)];
                    draft.field("map").unwrap().into_object().unwrap().remove(key);
                }
                _ => {
                    let current = draft.value().get("text").unwrap().as_str().unwrap().to_owned();
                    let keep = current
                        .char_indices()
                        .map(|(i, _)| i)
                        .nth(rng.below(3))
                        .unwrap_or(current.len());
                    let next = format!("{}{}", &current[keep..], ["", "y", "\n"][rng.below(3)]);
                    draft.field("text").unwrap().set(next);
                }
            }
        }
        let prepared = change.prepare();
        let text = serde_json::to_string(prepared.ops()).unwrap();
        let ops: Vec<Op> = serde_json::from_str(&text).unwrap();
        assert!(
            ops.iter()
                .all(|op| op.path().iter().all(|seg| seg.as_key() != Some("constructor"))),
            "reserved segment in a path: {text}"
        );
        apply(&mut replica, &ops).unwrap();
        let ops = tracker.adopt(prepared).unwrap();
        assert_eq!(&replica, tracker.value(), "after {ops:?}");
    }
}

#[test]
fn json_round_trips_and_paths_display() {
    let value: Json = delta_chord::json::from_str(r#"{"a":[1,2.5,"x",null,true],"b":{}}"#).unwrap();
    assert_eq!(value.to_json_string(), r#"{"a":[1,2.5,"x",null,true],"b":{}}"#);
    assert_eq!(path!["a", 0].to_string(), r#"["a",0]"#);
    assert_eq!(Path::root().to_string(), "[]");
    assert!(delta_chord::json::from_str("[1e400]").is_err());
}

fn random_value(rng: &mut Rng, depth: usize) -> Json {
    if depth == 0 {
        return rng.scalar();
    }
    match rng.below(4) {
        0 => (0..rng.below(5))
            .map(|_| random_value(rng, depth - 1))
            .collect::<Vec<_>>()
            .into(),
        1 => {
            let keys = ["a", "b", "c", "constructor", "d"];
            let mut map = delta_chord::json::Map::new();
            for _ in 0..rng.below(4) {
                map.insert(keys[rng.below(keys.len())].into(), random_value(rng, depth - 1));
            }
            map.into()
        }
        2 => Json::from(["", "abc", "bcd", "cde\n", "é😀"][rng.below(5)]),
        _ => rng.scalar(),
    }
}

/// Edit a copy of `value` at random places, sharing everything else.
fn mutate(rng: &mut Rng, value: &mut Json) {
    match value {
        Json::Array(array) if !array.is_empty() && rng.below(3) > 0 => {
            let index = rng.below(array.len());
            mutate(rng, &mut array.make_mut()[index]);
        }
        Json::Array(array) => {
            let items = array.make_mut();
            match rng.below(3) {
                0 => items.insert(rng.below(items.len() + 1), random_value(rng, 1)),
                1 if !items.is_empty() => {
                    items.remove(rng.below(items.len()));
                }
                _ => items.push(random_value(rng, 1)),
            }
        }
        Json::Object(object) if !object.is_empty() && rng.below(3) > 0 => {
            let index = rng.below(object.len());
            let map = object.make_mut();
            let (_, child) = map.get_index_mut(index).unwrap();
            mutate(rng, child);
        }
        Json::String(text) => {
            let piece = ["x", "é", "\n"][rng.below(3)];
            if rng.below(2) == 0 {
                text.make_mut().push_str(piece);
            } else {
                let cut = text.char_indices().nth(1).map_or(0, |(i, _)| i);
                text.make_mut().drain(..cut);
                text.make_mut().push_str(piece);
            }
        }
        _ => *value = random_value(rng, 2),
    }
}

#[test]
fn random_diffs_converge() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for _ in 0..3000 {
        let old = random_value(&mut rng, 3);
        let mut new = old.clone();
        for _ in 0..=rng.below(3) {
            mutate(&mut rng, &mut new);
        }
        let ops = diff(&old, &new);
        assert_eq!(applied(&old, &ops).unwrap(), new, "{old} -> {new} via {ops:?}");
        assert!(
            ops.iter()
                .all(|op| op.path().iter().all(|seg| seg.as_key() != Some("constructor"))),
            "{ops:?}"
        );
        let tracker = Tracker::new(old.clone());
        let prepared = tracker.prepare_diff(new.clone());
        assert_eq!(prepared.value(), &new);
    }
}
