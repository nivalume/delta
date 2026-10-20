//! Tests of `delta_chord::state`.

#![allow(clippy::unwrap_used)] // Test helpers outside #[test] functions.

use std::time::Duration;

use delta_chord::context::{Context, ContextKey};
use delta_chord::delta::{AdoptError, Op, apply};
use delta_chord::json;
use delta_chord::state::{DeliveryKind, State, SubscribeOptions};
use futures::StreamExt;
use pretty_assertions::assert_eq;

static TURN: ContextKey<u32> = ContextKey::new("turn");

fn ops_json(ops: &[Op]) -> serde_json::Value {
    serde_json::to_value(ops).unwrap()
}

#[tokio::test]
async fn subscribers_get_the_value_then_every_change() {
    let state = State::new(json!({"out": "", "count": 0}));
    let mut subscription = state.subscribe();
    let context = Context::background().with_value(&TURN, 7);

    let returned = state.change(&context, |mut draft| {
        draft.field("out").unwrap().into_string().unwrap().push_str("done\n");
        draft.field("count").unwrap().set(1);
        "ok"
    });
    assert_eq!(returned, "ok");
    // Empty changes are not published.
    state.change(&context, |mut draft| draft.field("count").unwrap().set(1));

    let hydrate = subscription.next().await.unwrap();
    assert_eq!(hydrate.kind, DeliveryKind::Hydrate);
    assert_eq!(hydrate.value, json!({"out": "", "count": 0}));
    assert_eq!(hydrate.sequence, 0);

    let update = subscription.next().await.unwrap();
    assert_eq!(update.kind, DeliveryKind::Update);
    assert_eq!(update.sequence, 1);
    assert_eq!(update.context.value(&TURN), Some(&7));
    assert_eq!(
        ops_json(&update.ops),
        serde_json::json!([["a", ["out"], "done\n"], ["s", ["count"], 1]])
    );
    assert_eq!(update.value, state.value());
    assert!(subscription.try_next().is_none());
    assert_eq!(state.sequence(), 1);
}

#[tokio::test]
async fn two_step_commits() {
    let state = State::new(json!({"n": 0}));
    let mut subscription = state.subscribe();
    let mut first = state.begin();
    let mut second = state.begin();
    first.draft().field("n").unwrap().set(1);
    second.draft().field("n").unwrap().set(2);
    let first = first.prepare();
    // ... the writer persists `first.ops()` here ...
    state.commit(&Context::background(), first).unwrap();
    assert!(matches!(
        state.commit(&Context::background(), second.prepare()),
        Err(AdoptError::Stale { .. })
    ));
    assert_eq!(state.value(), json!({"n": 1}));

    subscription.next().await.unwrap();
    assert_eq!(
        ops_json(&subscription.next().await.unwrap().ops),
        serde_json::json!([["s", ["n"], 1]])
    );
}

#[tokio::test]
async fn replace_publishes_a_diff() {
    let state = State::new(json!({"big": [1, 2, 3], "log": "a"}));
    let mut subscription = state.subscribe();
    // A value built from the current one shares its unchanged parts.
    let mut next = state.value();
    *next.pointer_mut(&delta_chord::path!["log"]).unwrap() = json!("ab");
    state.replace(&Context::background(), next);
    subscription.next().await.unwrap();
    assert_eq!(
        ops_json(&subscription.next().await.unwrap().ops),
        serde_json::json!([["a", ["log"], "b"]])
    );
}

#[tokio::test]
async fn a_backlog_merges_into_one_delivery() {
    let state = State::new(json!({"items": []}));
    let mut subscription = state.subscribe_with(SubscribeOptions { max_pending: 3 });
    subscription.next().await.unwrap();
    let mut replica = state.value();
    for index in 0..10 {
        state.change(&Context::background(), |mut draft| {
            draft.field("items").unwrap().into_array().unwrap().push(index);
        });
    }
    let mut deliveries = Vec::new();
    while let Some(delivery) = subscription.try_next() {
        deliveries.push(delivery);
    }
    assert!(deliveries.len() <= 3, "{} deliveries", deliveries.len());
    assert_eq!(
        deliveries.iter().map(|delivery| delivery.skipped).sum::<u64>() + deliveries.len() as u64,
        10
    );
    for delivery in &deliveries {
        apply(&mut replica, &delivery.ops).unwrap();
        assert_eq!(replica, delivery.value);
    }
    assert_eq!(deliveries.last().unwrap().sequence, 10);
    assert_eq!(replica, state.value());
}

#[tokio::test]
async fn a_hydration_backlog_stays_a_hydration() {
    let state = State::new(json!(0));
    let mut subscription = state.subscribe_with(SubscribeOptions { max_pending: 1 });
    state.change(&Context::background(), |mut draft| draft.set(1));
    state.change(&Context::background(), |mut draft| draft.set(2));
    let delivery = subscription.next().await.unwrap();
    assert_eq!(delivery.kind, DeliveryKind::Hydrate);
    assert_eq!(delivery.value, json!(2));
    assert!(delivery.ops.is_empty());
    assert_eq!(delivery.skipped, 2);
    assert!(subscription.try_next().is_none());
}

#[tokio::test]
async fn subscriptions_end_with_the_state() {
    let state = State::new(json!(0));
    let subscription = state.subscribe();
    let other = state.clone();
    let reader = tokio::spawn(async move { subscription.into_stream().collect::<Vec<_>>().await });
    tokio::time::sleep(Duration::from_millis(10)).await;
    other.change(&Context::background(), |mut draft| draft.set(1));
    drop(state);
    drop(other);
    let deliveries = tokio::time::timeout(Duration::from_secs(5), reader)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(deliveries.len(), 2);
    assert_eq!(deliveries[1].value, json!(1));
}

#[test]
fn a_panicking_change_keeps_the_value() {
    let state = State::new(json!({"n": 0}));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.change(&Context::background(), |mut draft| {
            draft.field("n").unwrap().set(5);
            panic!("boom");
        });
    }));
    assert!(result.is_err());
    assert_eq!(state.value(), json!({"n": 0}));
    state.change(&Context::background(), |mut draft| draft.field("n").unwrap().set(1));
    assert_eq!(state.value(), json!({"n": 1}));
}

#[test]
#[should_panic(expected = "inside a change of the same state")]
fn reentrant_changes_panic() {
    let state = State::new(json!(0));
    state.change(&Context::background(), |_| state.change(&Context::background(), |_| ()));
}
