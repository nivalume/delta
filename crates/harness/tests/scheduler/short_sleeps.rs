//! Sleeps of a millisecond, on the real clock: they fall due around the reservation pass that
//! reads the clock, and must still get a timer (regression: such a sleep once waited for an
//! unrelated scheduler event).

use std::time::Duration;

use delta_chord::json::Json;
use delta_harness::scheduler::Next;

use crate::{Backend, Env, Probe, erased, phase};

cases!(one_millisecond_sleeps_all_wake);

const ROUNDS: i64 = 200;

async fn one_millisecond_sleeps_all_wake(backend: Backend) {
    let probe = Probe::new("test.short-sleeper")
        .run(|rt, checkpoint| {
            Box::pin(async move {
                let round = checkpoint.get("round").and_then(Json::as_i64).unwrap_or(0);
                if round == ROUNDS {
                    return Ok(Next::complete(Json::from(round)));
                }
                let mut next = phase("sleep");
                if let Json::Object(object) = &mut next {
                    object.make_mut().insert("round".into(), Json::from(round + 1));
                }
                Ok(Next::sleep(rt.now() + 1, next))
            })
        })
        .arc();
    let env = Env::new(backend, [erased(&probe)]).await;
    let harness = env.run().await;
    let conversation = harness.conversation().await;
    let id = harness.start(&probe, conversation).await;
    let settled = tokio::time::timeout(Duration::from_secs(30), harness.settled(id))
        .await
        .expect("every short sleep woke");
    assert_eq!(settled.result::<i64>().unwrap().unwrap(), ROUNDS);
    harness.close().await;
}
