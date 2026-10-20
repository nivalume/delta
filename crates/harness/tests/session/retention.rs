//! Document cache retention (upstream `session-document-retention.test.ts`). Time is
//! paused, so these run on the memory backend only: SQLite answers from its own thread,
//! and paused time would advance while the runtime waits for it.

use std::time::Duration;

use delta_chord::json;
use delta_harness::documents::DocAddress;

use super::*;

fn retention(retention: Option<Duration>) -> SessionOptions {
    SessionOptions {
        document_retention: retention,
        ..SessionOptions::default()
    }
}

async fn touch_counter(env: &Env) {
    env.session
        .commit(&ctx(), |tx| {
            Box::pin(async move {
                tx.doc(DocAddress::<Counter>::new(())).await?;
                Ok(())
            })
        })
        .await
        .unwrap();
}

async fn cached(env: &Env) -> usize {
    env.session.cached_documents().await.unwrap()
}

async fn advance(seconds: u64) {
    tokio::time::advance(Duration::from_secs(seconds)).await;
    // Let the line run its timer.
    tokio::task::yield_now().await;
}

#[tokio::test(start_paused = true)]
async fn evicts_on_its_timer_and_reloads_the_same_value() {
    let env = Env::open_with(Backend::Memory, retention(Some(Duration::from_mins(1)))).await;
    touch_counter(&env).await;
    assert_eq!(cached(&env).await, 1);
    advance(30).await;
    assert_eq!(cached(&env).await, 1);
    advance(31).await;
    assert_eq!(cached(&env).await, 0);
    let value = env
        .session
        .snapshot(DocAddress::<Counter>::new(()), &ctx())
        .await
        .unwrap();
    assert_eq!(value, Some(json!({ "count": 0 })));
    assert_eq!(cached(&env).await, 1);
}

#[tokio::test(start_paused = true)]
async fn a_used_document_stays() {
    let env = Env::open_with(Backend::Memory, retention(Some(Duration::from_mins(1)))).await;
    touch_counter(&env).await;
    advance(50).await;
    // A cache hit answered without a line job still counts as a use.
    env.session
        .snapshot(DocAddress::<Counter>::new(()), &ctx())
        .await
        .unwrap();
    advance(50).await;
    assert_eq!(cached(&env).await, 1);
    advance(11).await;
    assert_eq!(cached(&env).await, 0);
}

#[tokio::test(start_paused = true)]
async fn an_observed_document_never_leaves() {
    let env = Env::open_with(Backend::Memory, retention(Some(Duration::from_mins(1)))).await;
    touch_counter(&env).await;
    let state = env
        .session
        .document_state(DocAddress::<Counter>::new(()), &ctx())
        .await
        .unwrap()
        .unwrap();
    let subscription = state.subscribe();
    drop(state);
    advance(100).await;
    assert_eq!(cached(&env).await, 1, "a subscription pins like its state");
    drop(subscription);
    advance(59).await;
    assert_eq!(cached(&env).await, 1);
    advance(62).await;
    assert_eq!(cached(&env).await, 0);
}

#[tokio::test(start_paused = true)]
async fn without_retention_documents_stay() {
    let env = Env::open_with(Backend::Memory, retention(None)).await;
    touch_counter(&env).await;
    advance(100_000).await;
    assert_eq!(cached(&env).await, 1);
}

#[tokio::test(start_paused = true)]
async fn zero_retention_drops_right_after_use_unless_observed() {
    let env = Env::open_with(Backend::Memory, retention(Some(Duration::ZERO))).await;
    let conversations = futures::future::join_all((0..20).map(|_| env.conversation())).await;
    for conversation in &conversations {
        let conversation = *conversation;
        env.session
            .commit(&ctx(), move |tx| {
                Box::pin(async move {
                    tx.doc(DocAddress::<Notes>::new(conversation)).await?;
                    Ok(())
                })
            })
            .await
            .unwrap();
    }
    assert_eq!(cached(&env).await, 0);
    let state = env
        .session
        .document_state(DocAddress::<Notes>::new(conversations[0]), &ctx())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(cached(&env).await, 1);
    drop(state);
    // Unpinning wakes the line, which sweeps.
    tokio::task::yield_now().await;
    assert_eq!(cached(&env).await, 0);
}

#[tokio::test(start_paused = true)]
async fn a_retired_incarnation_leaves_with_its_observers_ended() {
    let env = Env::open_with(Backend::Memory, retention(None)).await;
    touch_counter(&env).await;
    let state = env
        .session
        .document_state(DocAddress::<Counter>::new(()), &ctx())
        .await
        .unwrap()
        .unwrap();
    env.session
        .commit(&ctx(), |tx| {
            Box::pin(async move { tx.retire_doc(DocAddress::<Counter>::new(())).await })
        })
        .await
        .unwrap();
    assert_eq!(cached(&env).await, 0);
    assert!(state.end().is_some());
}
