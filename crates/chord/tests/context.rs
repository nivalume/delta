//! Tests of `delta_chord::context`.

use std::time::Duration;

use delta_chord::context::{Aborted, CancelReason, Context, ContextKey};
use tokio_util::sync::CancellationToken;

static USER: ContextKey<String> = ContextKey::new("user");
static OTHER_USER: ContextKey<String> = ContextKey::new("user");
static COUNT: ContextKey<u32> = ContextKey::new("count");

#[derive(Debug, thiserror::Error)]
#[error("superseded")]
struct Superseded;

#[test]
fn values_are_scoped_and_typed() {
    let root = Context::background();
    let ann = root.with_value(&USER, "ann".to_owned());
    let both = ann.with_value(&COUNT, 2);
    let bob = both.with_value(&USER, "bob".to_owned());
    assert_eq!(root.value(&USER), None);
    assert_eq!(ann.value(&USER).unwrap(), "ann");
    assert_eq!(bob.value(&USER).unwrap(), "bob");
    assert_eq!(bob.value(&COUNT), Some(&2));
    // Keys compare by identity, not by name or type.
    assert_eq!(bob.value(&OTHER_USER), None);
    assert_eq!(
        format!("{bob:?}"),
        r#"Context { values: ["user", "count", "user"], cancellable: false, cancelled: false }"#
    );
}

#[test]
fn cancellation_flows_to_children_not_parents() {
    let (parent, cancel_parent) = Context::background().with_cancel();
    let (child, cancel_child) = parent.with_cancel();
    let keep = child.without_cancel().with_value(&COUNT, 1);

    cancel_child.cancel_with(CancelReason::new(Superseded));
    assert!(child.is_cancelled());
    assert!(!parent.is_cancelled());
    assert!(child.reason().unwrap().downcast_ref::<Superseded>().is_some());

    let (other, _) = parent.with_cancel();
    cancel_parent.cancel();
    assert!(other.is_cancelled());
    assert!(other.reason().unwrap().downcast_ref::<Aborted>().is_some());
    // The child keeps the reason it was cancelled with first.
    assert!(child.reason().unwrap().downcast_ref::<Superseded>().is_some());
    assert!(!keep.is_cancelled());
    assert!(parent.token().is_cancelled());
}

#[test]
fn a_parent_reason_reaches_children() {
    let (parent, cancel) = Context::background().with_cancel();
    let (child, _) = parent.with_value(&COUNT, 1).with_cancel();
    cancel.cancel_with(CancelReason::message("shutdown"));
    assert_eq!(child.reason().unwrap().to_string(), "shutdown");
}

#[tokio::test]
async fn run_drops_the_future_on_cancellation() {
    let (context, cancel) = Context::background().with_cancel();
    assert_eq!(context.run(async { 1 }).await.unwrap(), 1);

    let dropped = CancellationToken::new();
    let guard = dropped.clone().drop_guard();
    let task = tokio::spawn({
        let context = context.clone();
        async move {
            context
                .run(async move {
                    let _guard = guard;
                    std::future::pending::<()>().await;
                })
                .await
        }
    });
    tokio::time::sleep(Duration::from_millis(10)).await;
    cancel.cancel_with(CancelReason::message("stop"));
    let error = task.await.unwrap().unwrap_err();
    assert_eq!(error.to_string(), "stop");
    assert!(dropped.is_cancelled(), "the future was dropped");
    assert!(context.run(async { 1 }).await.is_err());
}

#[tokio::test]
async fn outside_tokens_join_the_context() {
    let (parent, cancel_parent) = Context::background().with_cancel();
    let outside = CancellationToken::new();
    let joined = parent.with_token(outside.clone());
    let (child, _) = joined.with_cancel();
    let token = joined.token();
    drop(joined);

    outside.cancel();
    assert!(child.is_cancelled());
    child.cancelled().await;
    token.cancelled().await;
    assert!(child.reason().unwrap().downcast_ref::<Aborted>().is_some());
    assert!(!parent.is_cancelled());
    cancel_parent.cancel();

    // The parent side cancels the joined token too.
    let (parent, cancel_parent) = Context::background().with_cancel();
    let joined = parent.with_token(CancellationToken::new());
    let token = joined.token();
    cancel_parent.cancel();
    tokio::time::timeout(Duration::from_secs(5), token.cancelled())
        .await
        .unwrap();
}

#[tokio::test]
async fn background_is_never_cancelled() {
    let context = Context::background();
    assert!(!context.is_cancelled());
    assert!(context.reason().is_none());
    assert!(
        tokio::time::timeout(Duration::from_millis(10), context.cancelled())
            .await
            .is_err()
    );
    assert!(!context.token().is_cancelled());
}
